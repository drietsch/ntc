"""Export a seeded random-init tiny model to `.ntc` for conformance testing.

The tokenizer is the deterministic test tokenizer embedded in the Rust-written
fixture `fixtures/models/tiny-v1/tiny.ntc` — extracted here by parsing the
container, so the two fixtures share byte-identical tokenizer sections.

Run from `training/`::

    uv run python -m export.export_tiny [--out ...] [--seed 42]

Then verify with the Rust reader::

    ./target/debug/ntc verify fixtures/models/tiny-v1-py/tiny.ntc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from export.ntc_writer import NtcWriter, read_ntc_file
from ntc_model.config import ARCHITECTURE, NtcArchConfig, tiny_config
from ntc_model.model import NtcEncoderHeadsV1, export_tensors, tensor_specs

REPO_ROOT = Path(__file__).resolve().parents[2]
RUST_TINY_NTC = REPO_ROOT / "fixtures" / "models" / "tiny-v1" / "tiny.ntc"
DEFAULT_OUT = REPO_ROOT / "fixtures" / "models" / "tiny-v1-py" / "tiny.ntc"

IR_VERSION = 1
ABI_VERSION = 1
HEAD_SPEC_VERSION = 1


def extract_test_tokenizer(rust_ntc_path: Path = RUST_TINY_NTC) -> bytes:
    """The embedded tokenizer.json bytes of the Rust tiny fixture."""
    parsed = read_ntc_file(rust_ntc_path)
    if not parsed["sha256_ok"]:
        raise ValueError(f"{rust_ntc_path}: sha256 footer mismatch")
    return parsed["tokenizer_bytes"]


def export_tiny(
    out_path: Path = DEFAULT_OUT,
    seed: int = 42,
    cfg: NtcArchConfig | None = None,
) -> Path:
    """Build, random-init (seeded), and export the tiny model. Returns the path.

    `cfg` overrides the default tiny architecture — used by the parity test to
    build a fixture that declares value templates, so the filter-template head
    exists on both sides and can be compared.
    """
    cfg = cfg or tiny_config()
    torch.manual_seed(seed)
    model = NtcEncoderHeadsV1(cfg)

    tensors = export_tensors(model, cfg)
    metadata = {
        "architecture": ARCHITECTURE,
        "model_version": f"tiny-v1-py-seed{seed}",
        "ir_version": IR_VERSION,
        "abi_version": ABI_VERSION,
        "head_spec_version": HEAD_SPEC_VERSION,
        "quantization": "f32",
        "model": cfg.to_metadata_model(),
        "semantic_types": [],
    }
    writer = NtcWriter(metadata, extract_test_tokenizer())
    # Directory order pinned to the canonical manifest order.
    for name, _shape in tensor_specs(cfg):
        arr = tensors[name]
        writer.add_tensor(name, "F32", list(arr.shape), arr)
    writer.write(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = export_tiny(args.out, args.seed)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
