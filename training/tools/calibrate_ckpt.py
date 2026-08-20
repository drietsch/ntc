"""Fit calibration temperatures onto a checkpoint saved mid-training.

`train.py` fits temperatures once, after the final epoch, and writes them into
the *final* checkpoint. `best.pt` is rewritten every epoch and carries only
`cfg` and `state_dict`, so exporting a mid-training snapshot silently produced
a model with the default temperatures (all 1.0).

That is not a cosmetic difference. The runtime's confidence policy gates on
calibrated probabilities: an uncalibrated tool head trips the ASK downgrade,
and the optional-argument threshold compares an uncalibrated presence
confidence against a value tuned on a calibrated one. Measured on the Studio
dev split, an uncalibrated epoch-7 snapshot answered ASK on 67 call-worthy
requests. The comparison against a fully-trained baseline was therefore not
measuring the model at all.

So: fit on dev, write the result back into the checkpoint, then export.

Run:
    uv run python -m tools.calibrate_ckpt --ckpt runs/studio-v2/best.pt \
        --data data/studio-aug --out /tmp/calibrated.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from datasets.collate import load_and_prepare
from ntc_model.config import NtcArchConfig
from ntc_model.model import NtcEncoderHeadsV1
from train import REPO, fit_temperatures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="dataset dir; dev.jsonl is what temperatures are fit on")
    parser.add_argument("--out", type=Path, default=None,
                        help="defaults to overwriting --ckpt")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer-any")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = NtcArchConfig.model_validate(ckpt["cfg"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = NtcEncoderHeadsV1(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    tokenizer = Tokenizer.from_file(
        str(REPO / "contracts" / args.tokenizer_dir / "tokenizer.json")
    )
    dev_items = load_and_prepare(cfg, tokenizer, args.data / "dev.jsonl", skip_oversize=True)
    calibration = fit_temperatures(model, cfg, dev_items, device)
    print(f"fitted on {len(dev_items)} dev rows: {calibration}")
    if calibration.action == 1.0 and calibration.tool == 1.0:
        print("  note: both landed on 1.0 — the grid's midpoint, so calibration "
              "is not what is limiting this checkpoint")

    ckpt["calibration"] = calibration
    out = args.out or args.ckpt
    torch.save(ckpt, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
