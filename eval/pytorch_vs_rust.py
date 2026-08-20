"""Do PyTorch and the Rust runtime pick the same tool? (IC-2.5)

PyTorch reports high dev `tool_acc` while the shipping runtime, scored on the
same rows with the same weights, picks a different tool often enough to halve
executable accuracy. Exactly one of these is measuring what we think it is.

Two explanations look identical in the aggregate numbers and have completely
different fixes:

  the metric      `tool_acc` averages over every dev row, and the non-CALL rows
                  (where the answer is NO_TOOL) are easy. A model that is good
                  at "no tool applies" and bad at choosing among real tools
                  scores well. Fix: report tool accuracy on CALL rows only.
  the pipeline    the two sides disagree on the same input — packing, canonical
                  text, slate order or context handling differs between the
                  Python collator and the Rust runtime. Fix: find the divergence.

So this runs both on identical rows and reports (a) PyTorch tool accuracy split
by whether the gold action is CALL, and (b) how often the two implementations
choose the *same* tool, which is the IC-2.5 gate and does not care which is
right.

Run (from the repo root):
    python3 eval/pytorch_vs_rust.py --ckpt training/runs/studio-v2/best.pt \
        --model /tmp/studio-v2-e7cal.ntc --pred /tmp/dev-e7cal.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def pytorch_predictions(ckpt: Path, data_dir: Path, tokenizer_dir: str) -> dict[str, str | None]:
    """{row id: predicted tool name or None for NO_TOOL}, from the trainer's
    own model and collator — the same code path that produces `tool_acc`."""
    script = f'''
import json, sys, torch
from pathlib import Path
from tokenizers import Tokenizer
from datasets.collate import load_and_prepare, make_batch
from ntc_model.config import NtcArchConfig
from ntc_model.model import NtcEncoderHeadsV1

REPO = Path({str(REPO)!r})
ckpt = torch.load({str(ckpt)!r}, map_location="cpu", weights_only=False)
cfg = NtcArchConfig.model_validate(ckpt["cfg"])
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = NtcEncoderHeadsV1(cfg).to(device)
model.load_state_dict(ckpt["state_dict"]); model.eval()

tok = Tokenizer.from_file(str(REPO / "contracts" / {tokenizer_dir!r} / "tokenizer.json"))
path = Path({str(data_dir)!r}) / "dev.jsonl"
items = load_and_prepare(cfg, tok, path, skip_oversize=True)
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
by_id = {{r["id"]: r for r in rows}}

out = {{}}
with torch.no_grad():
    for i in range(0, len(items), 16):
        chunk = items[i:i+16]
        batch = make_batch(cfg, chunk)
        batch.pop("targets"); batch.pop("n_linked", None)
        logits = model(**{{k: v.to(device) for k, v in batch.items()}})["tool.logits"].cpu()
        for j, it in enumerate(chunk):
            row = by_id.get(it.id)
            n = int(batch["tool_count"][j])
            idx = int(logits[j, :n+1].argmax())
            names = [c["name"] for c in (row or {{}}).get("candidates", [])] if row else []
            out[it.id] = (
                names[idx] if row and idx < len(names) and idx < n else None
            )
print(json.dumps(out))
'''
    proc = subprocess.run(
        ["uv", "run", "python", "-c", script],
        cwd=REPO / "training", capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"pytorch inference failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True,
                        help="batch-infer output for the same rows")
    parser.add_argument("--data", type=Path, default=REPO / "training" / "data" / "studio")
    parser.add_argument("--tokenizer-dir", default="tokenizer-any")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/torch-tool-preds.json"),
                        help="reuse PyTorch predictions instead of recomputing")
    args = parser.parse_args()

    gold = [json.loads(l) for l in (args.data / "dev.jsonl").read_text().splitlines() if l.strip()]
    rust = {}
    for line in args.pred.read_text().splitlines():
        r = json.loads(line)
        res = r.get("result") or {}
        rust[r["id"]] = ((res.get("ir") or {}).get("tool") or {}).get("registry_id")

    if args.cache and args.cache.exists():
        torch_pred = json.loads(args.cache.read_text())
        print(f"(reusing PyTorch predictions from {args.cache})")
    else:
        torch_pred = pytorch_predictions(args.ckpt, args.data, args.tokenizer_dir)
        if args.cache:
            args.cache.write_text(json.dumps(torch_pred))

    call_rows = [g for g in gold if g["gold"]["action"] == "CALL"]
    other_rows = [g for g in gold if g["gold"]["action"] != "CALL"]

    def acc(rows, pred):
        hit = sum(1 for g in rows if pred.get(g["id"]) == g["gold"].get("tool"))
        return hit, len(rows)

    print(f"{len(gold)} dev rows — {len(call_rows)} CALL, {len(other_rows)} not\n")
    for label, rows in (("CALL rows", call_rows), ("non-CALL rows", other_rows)):
        h, n = acc(rows, torch_pred)
        print(f"  PyTorch tool correct, {label:14} {h:3}/{n:<4} {h / max(1, n):.1%}")
    h, n = acc(gold, torch_pred)
    print(f"  PyTorch tool correct, {'all rows':14} {h:3}/{n:<4} {h / max(1, n):.1%}"
          "   <- the number train.py logs")

    both = [g["id"] for g in gold if g["id"] in torch_pred and g["id"] in rust]
    agree = sum(1 for i in both if torch_pred[i] == rust[i])
    print(f"\n  PyTorch and Rust chose the same tool  {agree}/{len(both)}  "
          f"{agree / max(1, len(both)):.1%}   <- IC-2.5 gate (>=99.9%)")

    # The interesting population: CALL rows where PyTorch already had the
    # right tool. Whatever the runtime does with those is pure loss, and its
    # cause is either the action head or the policy, not tool selection.
    outcome = {}
    for line in args.pred.read_text().splitlines():
        r = json.loads(line)
        outcome[r["id"]] = (r.get("result") or {}).get("outcome")

    winners = [g for g in call_rows if torch_pred.get(g["id"]) == g["gold"]["tool"]]
    from collections import Counter
    got = Counter(outcome.get(g["id"]) for g in winners)
    kept = sum(1 for g in winners if rust.get(g["id"]) == g["gold"]["tool"])
    print(f"\n  Of the {len(winners)} CALL rows PyTorch got right, the runtime:")
    for k, v in got.most_common():
        print(f"    answered {str(k):10} {v:3}")
    print(f"    ...kept the same tool in the IR  {kept}/{len(winners)}  "
          f"{kept / max(1, len(winners)):.1%}")
    print("\n  A tool PyTorch chose correctly but the runtime discarded is lost to"
          "\n  the action head or ConfidencePolicy, not to tool selection.")


if __name__ == "__main__":
    main()
