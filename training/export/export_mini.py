"""Export the trained mini model to `.ntc` (F16 weights, F32 norms, frozen
tokenizer, dev-fitted calibration temperatures in metadata).

Run:  uv run python -m export.export_mini --ckpt runs/mini/best.pt \
          --out ../models/ntc-mini-v1/model.ntc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from export.ntc_writer import NtcWriter
from ntc_model.config import NtcArchConfig
from ntc_model.model import NtcEncoderHeadsV1, export_tensors

REPO = Path(__file__).resolve().parents[2]


def export(ckpt_path: Path, out_path: Path, model_version: str, tokenizer_dir: str = "tokenizer") -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = NtcArchConfig(**ckpt["cfg"])
    if "calibration" in ckpt:
        cfg = cfg.model_copy(update={"calibration": ckpt["calibration"]})
    model = NtcEncoderHeadsV1(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tokenizer_bytes = (REPO / "contracts" / tokenizer_dir / "tokenizer.json").read_bytes()
    metadata = {
        "architecture": "ntc_encoder_heads_v1",
        "model_version": model_version,
        "ir_version": 1,
        "abi_version": 1,
        "head_spec_version": 1,
        "tokenizer_sha256": "",  # writer fills
        "quantization": "f16",
        "model": cfg.model_dump(),
        "semantic_types": [],
    }
    writer = NtcWriter(metadata, tokenizer_bytes)

    arrays = export_tensors(model, cfg)
    f16_overflow = 0
    for name, arr in arrays.items():
        if name.endswith(("norm.weight", "norm.bias")):
            writer.add_tensor(name, "F32", list(arr.shape), arr.astype(np.float32))
        else:
            f16 = arr.astype(np.float16)
            # Per-tensor overflow check (F16 max ≈ 65504).
            if not np.isfinite(f16).all():
                f16_overflow += 1
                writer.add_tensor(name, "F32", list(arr.shape), arr.astype(np.float32))
            else:
                writer.add_tensor(name, "F16", list(arr.shape), f16)
    if f16_overflow:
        print(f"note: {f16_overflow} tensors kept F32 due to F16 overflow")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer.write(out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("runs/mini/best.pt"))
    parser.add_argument("--out", type=Path, default=REPO / "models" / "ntc-mini-v1" / "model.ntc")
    parser.add_argument("--version", default="ntc-mini-v1")
    parser.add_argument("--tokenizer-dir", default="tokenizer",
                        help="contracts/<dir>/tokenizer.json to embed (tokenizer | tokenizer-any)")
    args = parser.parse_args()
    export(args.ckpt, args.out, args.version, args.tokenizer_dir)


if __name__ == "__main__":
    main()
