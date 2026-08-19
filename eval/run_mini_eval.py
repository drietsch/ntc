"""IC-2.5 + eval-matrix driver for the mini model.

1. Runs the exported `.ntc` over the dataset's dev/test splits through the
   **Rust runtime** (`ntc batch-infer`) — the runtime is the reference scorer.
2. Computes the spec §60/§61 metric matrix per split-tag × language via
   eval.harness.
3. Runs the same examples through the **PyTorch** model and reports
   Python↔Rust decision agreement (IC-2.5 gate ≥ 0.999 on action+tool).

Run (from training/):
  uv run python -m eval.run_mini_eval --model ../models/ntc-mini-v1/model.ntc \
      --data data/mini --out ../models/ntc-mini-v1/eval
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_RELEASE = REPO / "target" / "release" / "ntc"
NTC_BIN = _RELEASE if _RELEASE.exists() else REPO / "target" / "debug" / "ntc"
NOW = "2026-08-18T11:00:00+02:00"
TZ = "Europe/Berlin"


def build_batch_input(examples: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {
                        "id": ex["id"],
                        "utterance": ex["utterance"],
                        "tools": ex["candidates"],
                        "timezone": TZ,
                        "now": NOW,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_rust(model: Path, batch_in: Path, batch_out: Path) -> dict[str, dict]:
    subprocess.run(
        [
            str(NTC_BIN), "batch-infer",
            "--model", str(model),
            "--input", str(batch_in),
            "--output", str(batch_out),
        ],
        check=True,
    )
    preds = {}
    for line in batch_out.read_text().splitlines():
        row = json.loads(line)
        preds[row["id"]] = row.get("result", {"error": row.get("error")})
    return preds


def metric_matrix(examples: list[dict], preds: dict[str, dict]) -> dict:
    from eval.harness import evaluate

    def subset(pred_gold_pairs, keys):
        return [p for p in pred_gold_pairs if all(k(p[2]) for k in keys)]

    pairs = []
    for ex in examples:
        pred = preds.get(ex["id"])
        if pred is None or "error" in pred:
            continue
        pairs.append((pred, ex["gold"], ex))

    matrix: dict[str, dict] = {}
    groups: dict[str, list] = defaultdict(list)
    for pred, gold, ex in pairs:
        seen = "unseen_family" if "unseen_family" in ex["tags"] else (
            "masked_names" if "masked_names" in ex["tags"] else "seen"
        )
        groups["all"].append((pred, gold))
        groups[f"split:{seen}"].append((pred, gold))
        groups[f"lang:{ex['lang']}"].append((pred, gold))
        groups[f"{seen}×{ex['lang']}"].append((pred, gold))
    for name, group in sorted(groups.items()):
        matrix[name] = {"n": len(group), **evaluate(group)}
    errors = sum(1 for ex in examples if preds.get(ex["id"], {}).get("error"))
    matrix["_infer_errors"] = errors
    return matrix


def python_decisions(examples: list[dict], model_path: Path) -> dict[str, dict]:
    import torch
    from tokenizers import Tokenizer

    from datasets.collate import load_and_prepare, make_batch  # noqa: F401 (path setup)
    from ntc_model.io import load_ntc_model
    from ntc_model.packing import Canonicalizer, pack_batch

    cfg, model, tokenizer_bytes = load_ntc_model(model_path)
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode())
    canon = Canonicalizer(NTC_BIN)

    # Canonicalize all candidate sets in one call.
    pending, keys_per_ex = {}, []
    for ex in examples:
        keys = []
        for i, tool in enumerate(ex["candidates"]):
            key = json.dumps(tool, sort_keys=True) + f"#{i}"
            pending.setdefault(key, (tool, i))
            keys.append(key)
        keys_per_ex.append(keys)
    keys = list(pending)
    results = canon.canonicalize([pending[k][0] for k in keys], [pending[k][1] for k in keys])
    cache = dict(zip(keys, results, strict=False))

    # Batch by tool count (NO_TOOL sits at index n, so mixed counts would
    # shift the label space under padding).
    by_count: dict[int, list[int]] = {}
    for i, ex in enumerate(examples):
        by_count.setdefault(len(ex["candidates"]), []).append(i)

    decisions = {}
    for n, indices in by_count.items():
        for chunk_start in range(0, len(indices), 32):
            chunk = indices[chunk_start : chunk_start + 32]
            batch = pack_batch(
                cfg,
                tokenizer,
                [examples[i]["utterance"] for i in chunk],
                [[cache[k] for k in keys_per_ex[i]] for i in chunk],
            )
            batch.pop("utterance_lens")
            with torch.no_grad():
                out = model(**batch)
            for bi, i in enumerate(chunk):
                ex = examples[i]
                action_idx = int(out["action.logits"][bi].argmax())
                tool_idx = int(out["tool.logits"][bi, : n + 1].argmax())
                decisions[ex["id"]] = {
                    "action": ["CALL", "ASK", "NO_CALL", "DELEGATE"][action_idx],
                    "tool": ex["candidates"][tool_idx]["name"] if tool_idx < n else None,
                }
    return decisions


def rust_decisions(pred: dict) -> dict:
    ir = pred.get("ir", {})
    tool = (ir.get("tool") or {}).get("registry_id")
    # Compare pre-policy neural decisions where possible: the policy can
    # downgrade CALL→NO_CALL/ASK, so compare the IR's action (post-policy) —
    # both sides must agree because Python applies no policy. Instead compare
    # raw argmax: Python argmax vs Rust ir action is only comparable when no
    # policy fired; count separately.
    return {"action": ir.get("action"), "tool": tool}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/mini"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {}
    for split in args.splits:
        examples = [
            json.loads(line)
            for line in (args.data / f"{split}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        batch_in = args.out / f"{split}.input.jsonl"
        batch_out = args.out / f"{split}.rust.jsonl"
        build_batch_input(examples, batch_in)
        preds = run_rust(args.model, batch_in, batch_out)
        report[split] = {"metrics": metric_matrix(examples, preds)}

        # IC-2.5: Python↔Rust decision agreement (action + tool identity).
        py = python_decisions(examples, args.model)
        n = agree_action = agree_tool = 0
        disagreements = []
        for ex in examples:
            pred = preds.get(ex["id"])
            if pred is None or "error" in pred:
                continue
            r = rust_decisions(pred)
            p = py[ex["id"]]
            n += 1
            # Rust policy may downgrade; recover the neural action from the
            # tool selection: agreement is measured on the tool identity and
            # on action once policy-neutral (CALL with tool vs CALL).
            agree_tool += int(r["tool"] == p["tool"] or (r["tool"] is None and p["tool"] is None))
            agree_action += int(r["action"] == p["action"])
            if r["action"] != p["action"] or r["tool"] != p["tool"]:
                disagreements.append({"id": ex["id"], "rust": r, "python": p})
        report[split]["ic25"] = {
            "n": n,
            "action_agreement": agree_action / max(1, n),
            "tool_agreement": agree_tool / max(1, n),
            "disagreements": disagreements[:20],
        }

    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for split, r in report.items():
        m = r["metrics"]["all"]
        print(f"[{split}] n={m['n']} metrics={ {k: v for k, v in m.items() if k != 'n'} }")
        print(f"[{split}] IC-2.5 agreement: {r['ic25']['action_agreement']:.4f} action, "
              f"{r['ic25']['tool_agreement']:.4f} tool over {r['ic25']['n']}")
    print(f"report → {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
