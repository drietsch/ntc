"""Convert the Salesforce xLAM function-calling corpus into NTC dataset
examples (Stage-2 schema grounding, spec §48).

Source: Salesforce/xlam-function-calling-60k (CC-BY-4.0 — see NOTICE).
Records are `{query, tools, answers}`, which is already NTC's input shape:
an utterance plus candidate tool schemas, and a typed call as the label.

Two conversions do the real work:

**Types.** xLAM annotates parameters as `str` / `int, optional` /
`List[int]` / `dict`. These map onto the canonical Tool ABI: scalars
directly, `List[T]` onto `LIST<T>` (ABI v2), and dict/nested payloads onto
OPAQUE — which makes the tool agent-only and routes such requests to
DELEGATE rather than inventing a value.

**Spans.** NTC's argument heads point into the utterance rather than
generating text, so each argument value is aligned back to the query to
recover a char span (~80% succeed). Values that do not appear literally
(normalized dates, defaults, inferred ids) are kept as bindings **without**
provenance — the model must infer them, and the span head is not supervised
on them.

Action labels follow docs/delegation.md: exactly one call → CALL; more than
one call → DELEGATE (parallel or chained work is beyond a single typed call
in V1; spec §77's planner would later reclaim the mechanical subset); no
call → NO_CALL.

Run: uv run python -m tools.convert_xlam --src ../salesforce-training-data.json \
        --out data/xlam --limit 20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets.schema import DatasetExample

# xLAM type string -> (json type, item type for arrays)
SCALARS = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
}
LIST_RE = re.compile(r"^(?:list|List)\[(\w+)\]$")


def map_type(raw: str) -> dict | None:
    """xLAM type annotation -> JSON-Schema fragment, or None if unusable."""
    t = (raw or "str").strip()
    optional = "optional" in t.lower()
    t = t.split(",")[0].strip()

    frag: dict
    if t in SCALARS:
        frag = {"type": SCALARS[t]}
    elif (m := LIST_RE.match(t)) and m.group(1) in SCALARS:
        frag = {"type": "array", "items": {"type": SCALARS[m.group(1)]}}
    elif t in ("list", "array"):
        # Untyped list: elements could be anything -> agent-only.
        frag = {"type": "array", "items": {"type": "object"}}
    elif t in ("dict", "object", "Dict", "any"):
        frag = {"type": "object"}
    else:
        return None
    frag["_optional"] = optional
    return frag


def convert_tool(tool: dict) -> dict | None:
    """xLAM tool -> NTC RawToolSchema (flat parameter style)."""
    name = tool.get("name")
    if not name:
        return None
    params: dict[str, dict] = {}
    for pname, spec in (tool.get("parameters") or {}).items():
        frag = map_type(str(spec.get("type", "str")))
        if frag is None:
            continue
        optional = frag.pop("_optional")
        # xLAM marks optionality in the type string; a declared default also
        # implies optional.
        has_default = str(spec.get("default", "")).strip() not in ("", "None")
        out = dict(frag)
        desc = (spec.get("description") or "").strip()
        if desc:
            out["description"] = desc[:160]
        if not optional and not has_default:
            out["required"] = True
        params[pname] = out
    return {
        "name": name,
        "description": (tool.get("description") or "")[:200],
        "parameters": params,
    }


def semantic_for(value, frag: dict) -> tuple[str, object] | None:
    """Pick the IR semantic type for a concrete argument value."""
    jt = frag.get("type")
    if jt == "array":
        item_json = (frag.get("items") or {}).get("type")
        if item_json == "object" or not isinstance(value, list):
            return None  # OPAQUE parameter — never bound
        items = []
        for v in value:
            if item_json == "integer" and isinstance(v, int) and not isinstance(v, bool):
                items.append({"semantic_type": "INTEGER", "value": v})
            elif item_json == "number" and isinstance(v, (int, float)):
                items.append({"semantic_type": "FLOAT", "value": float(v)})
            elif item_json == "string":
                items.append({"semantic_type": "STRING", "value": str(v)})
            elif item_json == "boolean" and isinstance(v, bool):
                items.append({"semantic_type": "BOOLEAN", "value": v})
            else:
                return None
        return ("LIST", {"items": items}) if items else None
    if jt == "object":
        return None
    if isinstance(value, bool):
        return ("BOOLEAN", value) if jt == "boolean" else None
    if isinstance(value, int) and jt == "integer":
        return ("INTEGER", value)
    if isinstance(value, (int, float)) and jt == "number":
        return ("FLOAT", float(value))
    if jt == "string":
        return ("STRING", str(value))
    # Type mismatch (e.g. "5" for an integer parameter): coerce where safe.
    if jt == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return ("INTEGER", int(value))
    return None


def find_list_span(query: str, values: list) -> tuple[int, int, str] | None:
    """Span covering a whole list region ("0.1, 0.2, 0.3"), not just its first
    element — the runtime's deterministic splitter needs the full region to
    recover every element (docs/tool-abi.md, composite value types)."""
    if not values:
        return None
    first = find_span(query, values[0])
    if first is None:
        return None
    start = first[0]
    end = first[1]
    cursor = first[1]
    for v in values[1:]:
        needle = str(v).strip()
        idx = query.lower().find(needle.lower(), cursor)
        if idx < 0:
            break  # elements not laid out contiguously; keep what we have
        cursor = idx + len(needle)
        end = cursor
    return start, end, query[start:end]


def find_span(query: str, value) -> tuple[int, int, str] | None:
    """Locate a value's surface form in the query (case-insensitive)."""
    needle = str(value).strip()
    if len(needle) < 2 and not needle.isdigit():
        return None
    lower_q, lower_n = query.lower(), needle.lower()
    idx = lower_q.find(lower_n)
    if idx < 0:
        return None
    return idx, idx + len(needle), query[idx : idx + len(needle)]


def convert(
    record: dict, rng: random.Random, split: str, max_candidates: int = 0
) -> dict | None:
    query = (record.get("query") or "").strip()
    if not query:
        return None
    answers = record["answers"]
    tools = record["tools"]
    if isinstance(answers, str):
        answers = json.loads(answers)
    if isinstance(tools, str):
        tools = json.loads(tools)

    candidates = [c for c in (convert_tool(t) for t in tools) if c]
    if not candidates:
        return None

    # xLAM occasionally offers two DIFFERENT schemas under the same tool name
    # (e.g. two `translate` APIs with `dest`/`src` vs `target`/`source`).
    # NTC's registry is keyed by tool id, so duplicates are ambiguous by
    # construction: keep the variant whose parameters actually cover the gold
    # call's arguments, and drop the rest.
    gold_args = {
        k
        for a in (answers if isinstance(answers, list) else [])
        for k in (a.get("arguments") or {})
    }
    deduped: dict[str, dict] = {}
    for c in candidates:
        prev = deduped.get(c["name"])
        if prev is None:
            deduped[c["name"]] = c
            continue
        score = lambda t: len(gold_args & set(t["parameters"]))  # noqa: E731
        if score(c) > score(prev):
            deduped[c["name"]] = c
    candidates = list(deduped.values())

    # Cap the candidate set: fusion attends over n_tools x schema_tokens, so
    # 8 long schemas make the sequence (and training) quadratically slower.
    # The gold tool is always kept; decoys are sampled, preserving the
    # hard-negative signal at a trainable size.
    if max_candidates and len(candidates) > max_candidates:
        gold_names = {a.get("name") for a in (answers if isinstance(answers, list) else [])}
        keep = [c for c in candidates if c["name"] in gold_names]
        decoys = [c for c in candidates if c["name"] not in gold_names]
        rng.shuffle(decoys)
        keep = (keep + decoys)[:max_candidates]
        rng.shuffle(keep)
        candidates = keep
    by_name = {c["name"]: c for c in candidates}

    # More than one call is beyond a single typed call (docs/delegation.md).
    if len(answers) != 1:
        action, tool_name, arguments = (
            ("DELEGATE", None, []) if len(answers) > 1 else ("NO_CALL", None, [])
        )
        gold = {"action": action, "tool": tool_name, "arguments": arguments, "unresolved": []}
        tags = ["delegate" if action == "DELEGATE" else "no_call", "xlam"]
        return finish(query, candidates, gold, split, tags)

    call = answers[0]
    name = call.get("name")
    if name not in by_name:
        return None
    schema = by_name[name]

    arguments = []
    for pname, value in (call.get("arguments") or {}).items():
        frag = schema["parameters"].get(pname)
        if frag is None:
            continue
        sem = semantic_for(value, frag)
        if sem is None:
            continue
        stype, svalue = sem
        arg: dict = {"parameter": pname, "semantic_type": stype, "value": svalue}
        found = (
            find_list_span(query, value)
            if stype == "LIST" and isinstance(value, list)
            else find_span(query, value)
        )
        if found:
            start, end, surface = found
            arg["char_span"] = {"start": start, "end": end}
            arg["surface"] = surface
        arguments.append(arg)

    gold = {"action": "CALL", "tool": name, "arguments": arguments, "unresolved": []}
    return finish(query, candidates, gold, split, ["call", "xlam"])


def finish(query: str, candidates: list[dict], gold: dict, split: str, tags: list[str]) -> dict:
    ex = {
        "id": "",
        "lang": "en",
        "utterance": query,
        "candidates": candidates,
        "gold": gold,
        "split": split,
        "tags": tags,
    }
    ex["id"] = "xlam-" + hashlib.sha256(
        json.dumps(ex, sort_keys=True).encode()
    ).hexdigest()[:14]
    return ex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("../salesforce-training-data.json"))
    parser.add_argument("--out", type=Path, default=Path("data/xlam"))
    parser.add_argument("--limit", type=int, default=20000, help="max converted examples")
    parser.add_argument("--max-delegate-share", type=float, default=0.3,
                        help="cap DELEGATE share (xLAM is 53%% multi-call)")
    parser.add_argument("--max-candidates", type=int, default=4,
                        help="cap candidates per example (0 = keep all 1-8)")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = json.loads(args.src.read_text())
    rng.shuffle(records)

    kept: list[dict] = []
    delegates = 0
    stats = {"read": 0, "invalid": 0, "unusable": 0, "delegate_dropped": 0}
    for rec in records:
        if len(kept) >= args.limit:
            break
        stats["read"] += 1
        bucket = int(hashlib.sha256(str(rec.get("id")).encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 90 else ("dev" if bucket < 95 else "test")
        ex = convert(rec, rng, split, args.max_candidates)
        if ex is None:
            stats["unusable"] += 1
            continue
        if ex["gold"]["action"] == "DELEGATE":
            if delegates >= args.limit * args.max_delegate_share:
                stats["delegate_dropped"] += 1
                continue
            delegates += 1
        try:
            DatasetExample.model_validate(ex)
        except Exception:  # noqa: BLE001 — skip anything the contract rejects
            stats["invalid"] += 1
            continue
        kept.append(ex)

    args.out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    handles = {s: (args.out / f"{s}.jsonl").open("w") for s in ("train", "dev", "test")}
    spanned = args_total = 0
    for ex in kept:
        handles[ex["split"]].write(json.dumps(ex, ensure_ascii=False) + "\n")
        counts[ex["split"]] = counts.get(ex["split"], 0) + 1
        key = f"{ex['split']}:{ex['gold']['action']}"
        counts[key] = counts.get(key, 0) + 1
        for a in ex["gold"]["arguments"]:
            args_total += 1
            spanned += "char_span" in a
    for h in handles.values():
        h.close()
    counts["arguments_with_span_pct"] = round(100 * spanned / max(1, args_total), 1)
    counts.update(stats)
    (args.out / "stats.json").write_text(json.dumps(counts, indent=2, sort_keys=True))
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
