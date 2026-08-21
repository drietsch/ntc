"""Evaluation for the Pimcore Studio corpus.

Aggregate action accuracy hides what matters here, so this reports:

- **per-reason** accuracy for DELEGATE and NO_CALL — `PAYLOAD_REQUIRED` has
  262 training examples and `MIXED_ELEMENT_TYPES` 37, so an average would be
  dominated by the common one;
- **the length-heuristic baseline** POLICY.md §5 sets as the bar: the corpus
  was rebuilt so word count alone gets ~0.40 recall at ~0.18 false positives.
  A model that does not beat that has learned nothing about delegation;
- **per-adversarial-tag** slices (`namespace_trap`, `source_conflict`,
  `family_dependent_symbol`, …) — precisely where a plausible wrong call gets
  made;
- **argument-source** accuracy (utterance / linked_item / resolver /
  inferred), since binding from the Studio selection is the capability the
  context frame exists for.

Run (from training/):
  uv run python ../eval/studio_report.py --pred <preds.jsonl> --gold data/studio/dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from esa import load_keyed  # noqa: E402


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def length_heuristic_baseline(gold: list[dict], fp_budget: float = 0.20) -> dict:
    """How well word count alone predicts DELEGATE.

    Two operating points, because they say different things: the max-F1
    threshold (usually degenerate — it fires on almost everything), and the
    best recall achievable within a false-positive budget, which is the shape
    POLICY.md §5 quotes ("0.40 recall at 0.18 FP"). The model has to beat the
    second one to have learned anything beyond "long sentences are complex".
    """
    rows = [(len(g["utterance"].split()), g["gold"]["action"] == "DELEGATE") for g in gold]
    positives = sum(1 for _, d in rows if d)
    negatives = len(rows) - positives
    best = {"threshold": 0, "recall": 0.0, "false_positive_rate": 1.0, "f1": 0.0}
    within_budget = {"threshold": 0, "recall": 0.0, "false_positive_rate": 0.0, "f1": 0.0}
    for thr in range(1, 40):
        tp = sum(1 for n, d in rows if d and n >= thr)
        fp = sum(1 for n, d in rows if not d and n >= thr)
        recall = tp / positives if positives else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / negatives if negatives else 0.0
        point = {
            "threshold": thr,
            "recall": round(recall, 3),
            "false_positive_rate": round(fpr, 3),
            "precision": round(precision, 3),
            "f1": round(f1, 3),
        }
        if f1 > best["f1"]:
            best = point
        if fpr <= fp_budget and recall > within_budget["recall"]:
            within_budget = point
    return {"max_f1": best, f"best_recall_at_fp_below_{fp_budget}": within_budget}


def pred_action(p: dict) -> str:
    return p.get("outcome", "ERROR")


def evaluate(preds: dict[str, dict], gold: list[dict]) -> dict:
    n = correct_action = 0
    per_reason: dict[str, Counter] = defaultdict(Counter)
    per_tag: dict[str, Counter] = defaultdict(Counter)
    per_lang: dict[str, Counter] = defaultdict(Counter)
    tool_hit = tool_n = 0
    src_hit = src_n = 0
    delegate = Counter()
    errors = 0

    for g in gold:
        p = preds.get(g.get("_key", g["id"]))
        if p is None or "error" in p:
            errors += 1
            continue
        n += 1
        ga, pa = g["gold"]["action"], pred_action(p)
        ok = ga == pa
        correct_action += ok
        per_lang[g["lang"]].update(total=1, correct=int(ok))

        if ga == "DELEGATE":
            reason = g["gold"].get("delegate_reason", "?")
            per_reason[f"delegate:{reason}"].update(total=1, correct=int(ok))
            delegate.update(gold_pos=1, hit=int(pa == "DELEGATE"))
        elif ga == "NO_CALL":
            per_reason[f"no_call:{g['gold'].get('no_call_reason', '?')}"].update(
                total=1, correct=int(ok)
            )
        if pa == "DELEGATE":
            delegate.update(pred_pos=1)

        for tag in g.get("tags", []):
            per_tag[tag].update(total=1, correct=int(ok))

        if ga == "CALL":
            tool_n += 1
            got = (p.get("ir") or {}).get("tool") or {}
            tool_hit += pa == "CALL" and got.get("registry_id") == g["gold"]["tool"]
            # Argument-source agreement on bound arguments.
            gold_src = {a["parameter"]: a["source"] for a in g["gold"]["arguments"]}
            for arg in (p.get("ir") or {}).get("arguments", []):
                want = gold_src.get(arg["parameter"])
                if want is None:
                    continue
                src_n += 1
                got_src = (arg.get("provenance") or {}).get("source", "MODEL")
                src_hit += got_src == want

    tp, fp_pos = delegate["hit"], delegate["pred_pos"] - delegate["hit"]
    precision = tp / delegate["pred_pos"] if delegate["pred_pos"] else 0.0
    recall = tp / delegate["gold_pos"] if delegate["gold_pos"] else 0.0
    negatives = n - delegate["gold_pos"]

    def rate(c: Counter) -> float:
        return round(c["correct"] / c["total"], 3) if c["total"] else 0.0

    return {
        "n_scored": n,
        "infer_errors": errors,
        "action_accuracy": round(correct_action / n, 3) if n else 0.0,
        "tool_selection_accuracy": round(tool_hit / tool_n, 3) if tool_n else None,
        "argument_source_accuracy": round(src_hit / src_n, 3) if src_n else None,
        "delegate": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "false_positive_rate": round(fp_pos / negatives, 3) if negatives else 0.0,
        },
        "per_reason": {k: {"n": v["total"], "accuracy": rate(v)} for k, v in sorted(per_reason.items())},
        "per_language": {k: {"n": v["total"], "accuracy": rate(v)} for k, v in sorted(per_lang.items())},
        "per_tag": {
            k: {"n": v["total"], "accuracy": rate(v)}
            for k, v in sorted(per_tag.items(), key=lambda kv: -kv[1]["total"])
        },
    }


def build_batch_input(gold: list[dict], path: Path) -> None:
    """Gold rows -> `ntc batch-infer` input, carrying the context frame."""
    with path.open("w") as f:
        for g in gold:
            ctx = g.get("context") or {}
            f.write(
                json.dumps(
                    {
                        "id": g.get("_key", g["id"]),
                        "utterance": g["utterance"],
                        "tools": g["candidates"],
                        "context": {
                            "linked": ctx.get("linked", []),
                            "resolver": ctx.get("resolver", []),
                            **({"selection_count": ctx["selection_count"]}
                               if ctx.get("selection_count") is not None else {}),
                            **({"studio_view": ctx["studio_view"]}
                               if ctx.get("studio_view") else {}),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="batch-infer output JSONL")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--make-input", type=Path, default=None,
                        help="write a batch-infer input file from the gold set and exit")
    args = parser.parse_args()

    if args.make_input:
        gold = load_keyed(args.gold)
        build_batch_input(gold, args.make_input)
        print(f"wrote {args.make_input} ({len(gold)} rows)")
        return

    gold = load_keyed(args.gold)
    preds = {}
    for row in load(args.pred):
        preds[row["id"]] = row.get("result", {"error": row.get("error")})

    report = evaluate(preds, gold)
    report["length_heuristic_baseline"] = length_heuristic_baseline(gold)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
