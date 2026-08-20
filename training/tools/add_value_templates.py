"""Annotate the composed-value arguments the compiler was told it could not reach.

`eval/esa.py` reports a ceiling — "N rows need a composed value, see DELEGATE" —
for every gold argument with no span behind it. On the Studio corpus that was
13.8% of all calls, and it turned out to be five (tool, argument) pairs, not a
property of the architecture:

    list_assets.parentId          218 rows, one distinct value: 1,
                                  which the schema itself declares as its default
    get_area_brick.brickIds        60 rows, a comma-joined list whose element
                                  names are verbatim in the utterance 108/120 times
    search_*.pqlFilter            258 rows, five closed shapes with 0-2 slots

None of those needs a decoder. This tool rewrites the corpus so each is
expressible:

* **parentId** — nothing to do here. The registry gains `"default": 1`, which
  the canonical ABI carries but never renders, so the deterministic backend can
  fill it without the model being told anything new (and without a retrain).
* **brickIds** — the schema gains `SEMANTIC LIST.CSV`, and the gold argument
  gains a `char_span` over the list region ("video-embed and spec-table"). The
  runtime's existing list splitter turns that into the provider's comma form.
* **pqlFilter** — the schema gains `SEMANTIC FILTER.PQL`, and the gold argument
  gains the `template` it instantiates plus a `char_span` over the region
  holding that template's slots. The filter-template head picks the shape; the
  span head fills it.

Candidate schemas are rewritten **from the registry, by name**, rather than
patched in place. A silent divergence between `examples/pimcore-tools.json` and
the schemas the model trained on has cost this project 25 accuracy points once
already; rebuilding candidates from the registry makes that divergence
impossible rather than merely unlikely.

Rows whose slot text cannot be located in the utterance are left untemplated and
counted in the report. They are the corpus's deliberate typo noise ("liferimeValue"
for `lifetimeValue`) — a span cannot recover a field name the request misspells,
and guessing which schema field was meant is a lookup the host does with
`get_queryable_fields`, not something to fake here.

Run (from training/):
    uv run python -m tools.add_value_templates --src data/studio-ask --out data/studio-tpl

The annotated registry is written beside the original as
`examples/pimcore-tools-templates.json`; the served one is left alone until a
model trained against the new schemas exists to replace it.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Semantic annotations the registry needs for the runtime to treat these
#: arguments as anything other than a literal span copy. Keyed by (tool, arg).
SEMANTICS = {
    ("search_assets", "pqlFilter"): "FILTER.PQL",
    ("search_data_objects", "pqlFilter"): "FILTER.PQL",
    ("search_documents", "pqlFilter"): "FILTER.PQL",
    ("get_area_brick", "brickIds"): "LIST.CSV",
}

#: Values the provider applies when the argument is omitted. Carried through
#: the ABI record, never rendered — declaring one costs no retrain.
DEFAULTS = {
    ("list_assets", "parentId"): 1,
}

#: How to recognise which template a gold value instantiates, and which part of
#: it has to be found in the utterance. Order matters: first match wins.
TEMPLATES = [
    ("UNPUBLISHED_PAGES", re.compile(r'^type = "page" AND published = false$'), ()),
    ("FILE_EXTENSION", re.compile(r'^filename LIKE "\*\.(?P<token>\w+)"$'), ("token",)),
    ("FIELD_IS_NULL", re.compile(r"^(?P<field>[\w.]+) IS NULL$"), ("field",)),
    ("SCORE_ABOVE", re.compile(r"^matchScore > (?P<number>\d+)$"), ("number",)),
    ("FIELD_LESS_THAN", re.compile(r"^(?P<field>[\w.]+) < (?P<number>\d+)$"), ("field", "number")),
]


def fold(s: str) -> str:
    """Case- and accent-insensitive form for locating a slot in an utterance."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn"
    )


def find(utterance: str, needle: str) -> tuple[int, int] | None:
    """Char range of `needle` in `utterance`, exact first then case/accent-folded.

    Returns character offsets into the ORIGINAL string: folding is
    length-preserving here because NFD combining marks are dropped only after
    a per-character pass, so index alignment holds for the Latin-script
    languages this corpus covers.
    """
    at = utterance.find(needle)
    if at >= 0:
        return at, at + len(needle)
    folded_u, folded_n = fold(utterance), fold(needle)
    if len(folded_u) != len(utterance):  # a mark got dropped; offsets no longer align
        return None
    at = folded_u.find(folded_n)
    return (at, at + len(needle)) if at >= 0 else None


def slot_span(utterance: str, template: str, groups: dict[str, str]) -> tuple[int, int] | None:
    """The char range the span head must mark for this template's slots.

    One span, covering every slot the pattern needs — the same contract the
    list splitter works under. For `{field} < {number}` that is from the field
    to the number; the distractor numbers these rows carry sit outside it,
    which is exactly what makes "the last number in the span" safe.
    """
    spans = []
    for name in ("field", "number", "token"):
        if name not in groups:
            continue
        # A file extension is named by whatever word contains it ("PDFs",
        # "pdf-Dateien"); the renderer matches it against the declared set, so
        # the span only has to cover a word holding it.
        needle = groups[name]
        at = find(utterance, needle)
        if at is None:
            return None
        if name == "token":
            # Widen to the whole word so the span lands on a token boundary.
            start, end = at
            while start > 0 and utterance[start - 1].isalnum():
                start -= 1
            while end < len(utterance) and utterance[end].isalnum():
                end += 1
            at = (start, end)
        spans.append(at)
    if not spans:
        return None
    return min(s for s, _ in spans), max(e for _, e in spans)


def list_region(utterance: str, items: list[str]) -> tuple[int, int] | None:
    """Char range covering every element of a comma-joined list argument."""
    spans = [find(utterance, item) for item in items]
    if any(s is None for s in spans):
        return None
    return min(s for s, _ in spans), max(e for _, e in spans)


def annotate(row: dict, registry: dict, stats: Counter) -> dict:
    # Candidates are rebuilt from the registry so the schemas the model trains
    # on cannot drift from the schemas it is served.
    for i, cand in enumerate(row["candidates"]):
        served = registry.get(cand["name"])
        if served is None:
            stats["candidate absent from registry"] += 1
            continue
        row["candidates"][i] = json.loads(json.dumps(served))

    gold = row["gold"]
    if gold["action"] != "CALL":
        return row
    tool, utterance = gold["tool"], row["utterance"]

    for arg in gold["arguments"]:
        key = (tool, arg["parameter"])
        if key in DEFAULTS:
            # Nothing to annotate: the value comes from the schema at decode
            # time. Recorded so the report accounts for every blocked row.
            stats["parentId — filled from the schema default"] += 1
            continue

        if SEMANTICS.get(key) == "LIST.CSV":
            items = [i.strip() for i in str(arg["value"]).split(",") if i.strip()]
            span = list_region(utterance, items)
            if span is None:
                stats["brickIds — element not in the utterance"] += 1
                continue
            arg["char_span"] = {"start": span[0], "end": span[1]}
            arg["surface"] = utterance[span[0] : span[1]]
            stats["brickIds — spanned as a list"] += 1
            continue

        if SEMANTICS.get(key) != "FILTER.PQL":
            continue

        value = str(arg["value"])
        for name, pattern, slots in TEMPLATES:
            m = pattern.match(value)
            if not m:
                continue
            groups = {k: v for k, v in m.groupdict().items() if v is not None}
            if not slots:
                arg["template"] = name
                stats[f"{name} — constant, no span needed"] += 1
                break
            span = slot_span(utterance, name, groups)
            if span is None:
                stats[f"{name} — slot text not in the utterance"] += 1
                break
            arg["template"] = name
            arg["char_span"] = {"start": span[0], "end": span[1]}
            arg["surface"] = utterance[span[0] : span[1]]
            stats[f"{name} — templated"] += 1
            break
        else:
            stats["pqlFilter — matches no declared template"] += 1
    return row


def patch_registry(src: Path, out: Path) -> list[dict]:
    """Write an annotated copy of the registry; the corpus is rebuilt from it.

    Deliberately a *copy*. `SEMANTIC FILTER.PQL` and `SEMANTIC LIST.CSV` are
    rendered lines, so adding them changes the schema token stream — a model
    trained without them would be served text it never saw, which is the exact
    failure that once cost this project 25 points. The served registry stays
    as it is until a model trained against this copy replaces it, and then the
    copy becomes the registry.

    (`default` is the exception: it is carried but never rendered, so it would
    have been safe to add in place. It rides along here to keep one file.)
    """
    tools = json.loads(src.read_text())
    changed = 0
    for tool in tools:
        for arg_name, param in (tool.get("parameters") or {}).items():
            key = (tool["name"], arg_name)
            if key in SEMANTICS and param.get("semantic") != SEMANTICS[key]:
                param["semantic"] = SEMANTICS[key]
                changed += 1
            if key in DEFAULTS and param.get("default") != DEFAULTS[key]:
                param["default"] = DEFAULTS[key]
                changed += 1
    out.write_text(json.dumps(tools, indent=2, ensure_ascii=False) + "\n")
    print(f"registry: {changed} parameter annotations -> {out}")
    print(f"          {src} left untouched (it serves the current checkpoints)")
    return tools


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--registry", type=Path, default=REPO / "examples" / "pimcore-tools.json")
    ap.add_argument(
        "--registry-out",
        type=Path,
        default=REPO / "examples" / "pimcore-tools-templates.json",
        help="annotated copy to write; the source registry is never modified",
    )
    args = ap.parse_args()

    tools = patch_registry(args.registry, args.registry_out)
    registry = {t["name"]: t for t in tools}

    args.out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    totals = {}
    for split in ("train", "dev", "test"):
        src = args.src / f"{split}.jsonl"
        if not src.exists():
            continue
        rows = [annotate(json.loads(line), registry, stats) for line in src.open()]
        (args.out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        totals[split] = len(rows)

    (args.out / "stats.json").write_text(json.dumps({**totals, "annotations": dict(stats)}, indent=2))
    print(f"\nrows: {totals}")
    print("\nannotations (train+dev+test, so each row is counted once per split file):")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {v:6d}  {k}")


if __name__ == "__main__":
    main()
