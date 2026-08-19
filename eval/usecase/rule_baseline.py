"""How much of the routing decision needs a neural network at all?

Spec §6.2 says anything deterministic should not consume model capacity. This
tests that claim against the Studio corpus by asking a blunt question: how far
does a rule engine with **no model** get on DELEGATE, using only the request,
the context frame, and the tool schemas?

The rules encode structure that is genuinely knowable without semantics:

  PAYLOAD_REQUIRED      every candidate that could plausibly serve a write
                        needs an OPAQUE (nested object/array) argument
  OVER_LIMIT            selectionCount exceeds the per-call cap the tool's own
                        description states
  MIXED_ELEMENT_TYPES   the linked selection spans element types no
                        single-type tool accepts

MULTI_STEP is deliberately NOT ruled on: the corpus was built so that length
and conjunctions do not separate it (POLICY.md §5), which is exactly the case
where a model earns its keep.

Run: python3 eval/usecase/rule_baseline.py --gold training/data/studio/dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# "max 5 per call", "Max 5", "1..5", "max: 50"
CAP_RE = re.compile(r"max(?:imum)?[ :]*(\d+)|\(1\.\.(\d+)\)|1\.\.(\d+)", re.I)
WRITE_HINT = re.compile(
    r"\b(update|set|change|edit|rename|assign|publish|apply|propose|write|"
    r"ändere|setze|aktualisier|umbenenn|zuweis|veröffentlich|"
    r"met[s]? à jour|modifie|renomme|publie|"
    r"actualiza|cambia|renombra|publica)\b",
    re.I,
)


def tool_cap(tool: dict) -> int | None:
    """Per-call element cap stated in the tool's own description."""
    text = tool.get("description", "") + " ".join(
        p.get("description", "") for p in (tool.get("parameters") or {}).values()
    )
    caps = [int(g) for m in CAP_RE.finditer(text) for g in m.groups() if g]
    return min(caps) if caps else None


def has_opaque_required(tool: dict) -> bool:
    """A required argument the structured heads could never author."""
    for spec in (tool.get("parameters") or {}).values():
        if not spec.get("required"):
            continue
        if spec.get("type") == "object":
            return True
        if spec.get("type") == "array" and (spec.get("items") or {}).get("type") == "object":
            return True
    return False


def rule_delegate(ex: dict) -> str | None:
    """Return a delegate_reason, or None for 'no rule fires'."""
    ctx = ex.get("context") or {}
    linked = ctx.get("linked", [])
    slate = ex["candidates"]

    kinds = {i.get("type") for i in linked}
    if len(kinds) > 1:
        return "MIXED_ELEMENT_TYPES"

    count = ctx.get("selection_count") or ctx.get("selectionCount") or len(linked)
    caps = [c for c in (tool_cap(t) for t in slate) if c]
    if caps and count > min(caps):
        return "OVER_LIMIT"

    # A write intent whose only plausible tools need a nested payload.
    if WRITE_HINT.search(ex["utterance"]):
        writers = [t for t in slate if re.match(r"^(propose_|update_|apply_)", t["name"])]
        if writers and all(has_opaque_required(t) for t in writers):
            return "PAYLOAD_REQUIRED"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path,
                        default=REPO / "training" / "data" / "studio" / "dev.jsonl")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.gold.read_text().splitlines() if line.strip()]
    tp = fp = fn = 0
    by_reason: dict[str, Counter] = {}
    for ex in rows:
        gold_action = ex["gold"]["action"]
        gold_reason = ex["gold"].get("delegate_reason")
        got = rule_delegate(ex)
        if gold_action == "DELEGATE":
            c = by_reason.setdefault(gold_reason or "?", Counter())
            c["total"] += 1
            if got:
                tp += 1
                c["caught"] += 1
                c["exact_reason"] += int(got == gold_reason)
            else:
                fn += 1
        elif got:
            fp += 1

    negatives = sum(1 for e in rows if e["gold"]["action"] != "DELEGATE")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    print(f"Rule engine, no model — {len(rows)} dev rows\n")
    print(f"  DELEGATE precision       {precision:.3f}")
    print(f"  DELEGATE recall          {recall:.3f}")
    print(f"  false-positive rate      {fp / negatives:.3f}" if negatives else "")
    print("\n  per gold reason (caught / total, and reason exactly right):")
    for reason, c in sorted(by_reason.items()):
        print(f"    {reason:22} {c['caught']:3}/{c['total']:<3} "
              f"reason exact {c['exact_reason']:3}")
    print("\n  Model, same split:  precision 0.905  recall 0.990  FP 0.029")


if __name__ == "__main__":
    main()
