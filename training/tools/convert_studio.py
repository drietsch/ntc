"""Convert the Pimcore Studio corpus (`specs/training/`) into NTC dataset
examples, preserving everything its label policy encodes.

The corpus is already NTC-shaped — utterance, candidate slate, typed gold —
but richer than the V1 dataset schema in four ways, all now expressible
(IR v2 / head codec v3):

- a **context frame**: linked items (the Studio selection), the host's
  identifier resolver, `selectionCount`, `studioView`;
- **argument sources**: `utterance` (char span) / `linked_item`
  (`linked_ref[s]`) / `resolver` (`resolver_token`) / `inferred`;
- **typed DELEGATE reasons** with a `suggested_tool`, and NO_CALL reasons;
- **ASK hints** naming why an argument could not be filled.

`ARRAY` becomes the IR's flat `LIST` (`item_type` + values); per-element
spans and per-element linked refs become `element_provenance`.

POLICY.md §4 annotations (`authorization`, `precedence`,
`resolution_hop_for`, `prerequisite`, `bound_context`, `note`, `limit_note`,
`context_used`) are *rationale and host inputs*, not model outputs. They are
carried through as `annotations` for eval slicing rather than turned into
prediction targets.

Run: uv run python -m tools.convert_studio --out data/studio
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SOURCE_MAP = {
    "utterance": "USER",
    "linked_item": "LINKED_ITEM",
    "resolver": "RESOLVER",
    "inferred": "MODEL",
}

# Annotations kept for slicing, never predicted.
ANNOTATION_KEYS = (
    "note", "prerequisite", "authorization", "bound_context", "context_used",
    "reason", "limit_note", "resolution_hop_for", "precedence",
)


def convert_argument(a: dict) -> dict:
    """One gold argument -> NTC gold argument."""
    out: dict = {"parameter": a["parameter"], "semantic_type": a["semantic_type"]}
    source = SOURCE_MAP[a["source"]]

    if a["semantic_type"] == "ARRAY":
        out["semantic_type"] = "LIST"
        out["item_type"] = a.get("item_type", "STRING")
        out["value"] = a["value"]
        # Per-element provenance: spans for utterance-sourced lists, refs for
        # selection-sourced ones.
        elements = a.get("elements") or []
        if elements:
            out["element_spans"] = [
                {"char_span": e["char_span"], "surface": e["surface"]} for e in elements
            ]
        if a.get("linked_refs"):
            out["linked_refs"] = a["linked_refs"]
    else:
        out["value"] = a["value"]

    if "char_span" in a:
        out["char_span"] = a["char_span"]
        out["surface"] = a.get("surface")
    if a.get("linked_ref"):
        out["linked_refs"] = [a["linked_ref"]]
    elif a.get("linked_refs") and "linked_refs" not in out:
        out["linked_refs"] = a["linked_refs"]
    if a.get("resolver_token"):
        out["resolver_token"] = a["resolver_token"]
    if a.get("composed_from"):
        out["composed_from"] = a["composed_from"]
    out["source"] = source
    return out


def subsample_candidates(rec: dict, limit: int, rng) -> None:
    """Cap the slate in place.

    Fusion attends over n_tools x schema_tokens, and Pimcore schemas run to
    541 tokens, so 8 candidates make training quadratically slower. Truncating
    the *descriptions* instead is not an option: the corpus's adversarial
    cases (`namespace_trap`, `family_dependent_symbol`) hide their
    discriminative signal inside the INFO text.

    Gold-absent slates are preserved as such — dropping the gold tool is the
    whole point of those examples, so they are subsampled without forcing it
    back in.
    """
    cands = rec["candidates"]
    if len(cands) <= limit:
        return
    gold = rec["gold"].get("tool")
    names = [c["name"] for c in cands]
    if gold and gold in names:
        keep = [c for c in cands if c["name"] == gold]
        rest = [c for c in cands if c["name"] != gold]
        rng.shuffle(rest)
        keep += rest[: limit - 1]
    else:
        keep = list(cands)
        rng.shuffle(keep)
        keep = keep[:limit]
    rng.shuffle(keep)
    rec["candidates"] = keep


def convert(rec: dict) -> dict:
    gold_in = rec["gold"]
    gold: dict = {
        "action": gold_in["action"],
        "tool": gold_in.get("tool"),
        "arguments": [convert_argument(a) for a in gold_in.get("arguments", [])],
        "unresolved": [
            {
                "parameter": u["parameter"],
                "reason": u["reason"],
                **({"hint": u["hint"]} if u.get("hint") else {}),
                **({"options": u["options"]} if u.get("options") else {}),
            }
            for u in gold_in.get("unresolved", [])
        ],
    }
    if gold_in.get("delegate_reason"):
        gold["delegate_reason"] = gold_in["delegate_reason"]
    if gold_in.get("suggested_tool"):
        gold["suggested_tool"] = gold_in["suggested_tool"]
    if gold_in.get("no_call_reason"):
        gold["no_call_reason"] = gold_in["no_call_reason"]

    ctx_in = rec.get("context") or {}
    context = {
        "linked": ctx_in.get("linked", []),
        "resolver": ctx_in.get("resolver", []),
        "locale": ctx_in.get("locale"),
    }
    if ctx_in.get("selectionCount") is not None:
        context["selection_count"] = ctx_in["selectionCount"]
    if ctx_in.get("studioView") is not None:
        context["studio_view"] = ctx_in["studioView"]

    annotations = {k: gold_in[k] for k in ANNOTATION_KEYS if k in gold_in}
    annotations["template_id"] = rec.get("template_id")
    annotations["vertical"] = rec.get("vertical")

    return {
        "id": rec["id"],
        "lang": rec["lang"],
        "utterance": rec["utterance"],
        "context": context,
        "candidates": rec["candidates"],
        "gold": gold,
        "split": rec["split"],
        "tags": rec.get("tags", []),
        "annotations": annotations,
    }


def verify(ex: dict) -> list[str]:
    """Contract checks the converter must not silently violate."""
    errs = []
    u = ex["utterance"]
    refs = {item["ref"] for item in ex["context"]["linked"]}
    names = [c["name"] for c in ex["candidates"]]
    if len(names) != len(set(names)):
        errs.append("duplicate candidate")
    g = ex["gold"]
    if g["action"] == "CALL" and g["tool"] not in names:
        errs.append("gold tool not in slate")
    if g["action"] == "DELEGATE" and not g.get("delegate_reason"):
        errs.append("DELEGATE without reason")
    if g["action"] in ("DELEGATE", "NO_CALL") and (g["tool"] or g["arguments"]):
        errs.append(f"{g['action']} carries tool/arguments")
    if g["action"] == "ASK" and not g["unresolved"]:
        errs.append("ASK without unresolved")
    for a in g["arguments"]:
        if "char_span" in a:
            s, e = a["char_span"]["start"], a["char_span"]["end"]
            if a.get("surface") and u[s:e] != a["surface"]:
                errs.append(f"span mismatch on {a['parameter']}")
        for ref in a.get("linked_refs", []):
            if ref not in refs:
                errs.append(f"dangling linked_ref {ref}")
        for el in a.get("element_spans", []):
            s, e = el["char_span"]["start"], el["char_span"]["end"]
            if u[s:e] != el["surface"]:
                errs.append(f"element span mismatch on {a['parameter']}")
    return errs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=REPO / "specs" / "training")
    parser.add_argument("--out", type=Path, default=Path("data/studio"))
    parser.add_argument("--max-candidates", type=int, default=4,
                        help="cap slate size (0 = keep the corpus's 2-8)")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    problems: Counter[str] = Counter()
    for split in ("train", "dev"):
        src = args.src / f"{split}.jsonl"
        rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
        out_rows = []
        for rec in rows:
            if args.max_candidates:
                subsample_candidates(rec, args.max_candidates, rng)
            ex = convert(rec)
            errs = verify(ex)
            if errs:
                problems.update(errs)
                continue
            out_rows.append(ex)
            stats[f"{split}:{ex['gold']['action']}"] += 1
            for a in ex["gold"]["arguments"]:
                stats[f"source:{a['source']}"] += 1
        (args.out / f"{split}.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in out_rows)
        )
        stats[split] = len(out_rows)
    # The harness expects a test split; the corpus holds out `dev`.
    (args.out / "test.jsonl").write_text((args.out / "dev.jsonl").read_text())

    report = {**stats, "rejected": dict(problems)}
    (args.out / "stats.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
