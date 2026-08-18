"""Evaluation harness: pair predictions with gold labels and aggregate metrics.

Inputs are JSONL files:
- predictions: one CompileOutcome-shaped object per line, plus an `id` field;
- gold: one object per line with an `id` and either a `gold` field holding the
  GoldLabel (a full DatasetExample line) or the GoldLabel fields inline.

Usage::

    python -m eval.harness --pred preds.jsonl --gold gold.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.metrics import (
    Pair,
    ask_accuracy,
    hallucinated_arg_rate,
    no_call_precision_recall,
    required_arg_accuracy,
    tag_errors,
    tool_selection_accuracy,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            records.append(record)
    return records


def gold_label(record: dict[str, Any]) -> dict[str, Any]:
    """The GoldLabel dict of a gold record (DatasetExample line or bare label)."""
    return record["gold"] if "gold" in record else record


def pair_by_id(
    predictions: Sequence[dict[str, Any]], gold_records: Sequence[dict[str, Any]]
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Match predictions to gold records by `id`; missing either side is an error."""
    by_id = {}
    for record in gold_records:
        if record["id"] in by_id:
            raise ValueError(f"duplicate gold id `{record['id']}`")
        by_id[record["id"]] = gold_label(record)
    pairs = []
    for pred in predictions:
        pid = pred.get("id")
        if pid is None:
            raise ValueError("prediction without an `id` field")
        if pid not in by_id:
            raise ValueError(f"prediction id `{pid}` has no gold record")
        pairs.append((pid, pred, by_id.pop(pid)))
    if by_id:
        raise ValueError(f"gold ids without predictions: {sorted(by_id)}")
    return pairs


def evaluate(pairs: Sequence[Pair]) -> dict[str, Any]:
    """Aggregate all metrics over (prediction, gold) pairs."""
    no_call = no_call_precision_recall(pairs)
    error_counts: Counter[str] = Counter()
    exact = 0
    for pred, gold in pairs:
        tags = tag_errors(pred, gold)
        error_counts.update(tags)
        exact += not tags
    return {
        "n_examples": len(pairs),
        "exact_match": exact / len(pairs) if pairs else None,
        "tool_selection_accuracy": tool_selection_accuracy(pairs),
        "required_arg_accuracy": required_arg_accuracy(pairs),
        "hallucinated_arg_rate": hallucinated_arg_rate(pairs),
        "no_call_precision": no_call["precision"],
        "no_call_recall": no_call["recall"],
        "no_call_f1": no_call["f1"],
        "ask_accuracy": ask_accuracy(pairs),
        "error_counts": dict(sorted(error_counts.items())),
    }


def run_harness(pred_path: str | Path, gold_path: str | Path) -> dict[str, Any]:
    """Load both JSONL files, pair by id, and return the full report."""
    triples = pair_by_id(load_jsonl(pred_path), load_jsonl(gold_path))
    pairs: list[Pair] = [(pred, gold) for _, pred, gold in triples]
    report = evaluate(pairs)
    report["per_example"] = [
        {"id": pid, "tags": tag_errors(pred, gold)} for pid, pred, gold in triples
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path, help="predictions JSONL")
    parser.add_argument("--gold", required=True, type=Path, help="gold labels JSONL")
    args = parser.parse_args()
    print(json.dumps(run_harness(args.pred, args.gold), indent=2))


if __name__ == "__main__":
    main()
