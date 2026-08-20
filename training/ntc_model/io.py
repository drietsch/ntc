"""Load a `.ntc` file back into the PyTorch model (inverse of
`ntc_model.model.export_tensors`). Pinned to the exporter by a
round-trip test (export → import → export must be identical)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from export.ntc_writer import read_ntc_file
from ntc_model.config import NtcArchConfig
from ntc_model.model import NtcEncoderHeadsV1, iter_tensor_params


def import_tensors(model: NtcEncoderHeadsV1, arrays: dict[str, np.ndarray]) -> None:
    """Copy canonical-named `[in, out]` arrays into the model in place."""
    seen = set()
    with torch.no_grad():
        for name, param, transpose in iter_tensor_params(model):
            if name not in arrays:
                raise KeyError(f"missing tensor `{name}`")
            value = arrays[name]
            if transpose:
                value = value.T
            t = torch.from_numpy(np.ascontiguousarray(value)).to(param.dtype)
            if t.shape != param.shape:
                raise ValueError(f"`{name}`: shape {tuple(t.shape)} vs param {tuple(param.shape)}")
            param.copy_(t)
            seen.add(name)
    extra = set(arrays) - seen
    if extra:
        raise KeyError(f"unused tensors in file: {sorted(extra)[:5]}…")


def decode_records(records: list[dict], data: bytes) -> dict[str, np.ndarray]:
    """Tensor directory + data section → float32 arrays (V1 dtypes)."""
    dtype_map = {"F32": np.float32, "F16": np.float16}
    out: dict[str, np.ndarray] = {}
    for r in records:
        if r["dtype"] == "BF16":
            raw = np.frombuffer(data, dtype=np.uint16, count=int(np.prod(r["shape"])),
                                offset=r["offset"])
            arr = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(
                data,
                dtype=dtype_map[r["dtype"]],
                count=int(np.prod(r["shape"])),
                offset=r["offset"],
            ).astype(np.float32)
        out[r["name"]] = arr.reshape(r["shape"]).copy()
    return out


def load_ntc_model(path: str | Path) -> tuple[NtcArchConfig, NtcEncoderHeadsV1, bytes]:
    """Read a `.ntc` file → (config, model with imported weights, tokenizer bytes)."""
    parsed = read_ntc_file(path)
    if not parsed["sha256_ok"]:
        raise ValueError(f"{path}: sha256 footer mismatch")
    cfg = NtcArchConfig(**parsed["metadata"]["model"])
    model = NtcEncoderHeadsV1(cfg)
    arrays = decode_records(parsed["records"], parsed["data"])
    import_tensors(model, arrays)
    model.eval()
    return cfg, model, parsed["tokenizer_bytes"]
