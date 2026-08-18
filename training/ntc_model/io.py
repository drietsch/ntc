"""Load a `.ntc` file back into the PyTorch model (inverse of
`ntc_model.model.export_tensors`). Pinned to the exporter by a
round-trip test (export → import → export must be identical)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import nn

from export.ntc_writer import read_ntc_file
from ntc_model.config import NtcArchConfig
from ntc_model.model import NtcEncoderHeadsV1


def iter_tensor_params(
    model: NtcEncoderHeadsV1,
) -> Iterator[tuple[str, torch.nn.Parameter, bool]]:
    """Yield (canonical_name, parameter, needs_transpose) — the same walk as
    `export_tensors`, expressed once for both directions of transfer."""

    def linear(name: str, mod: nn.Linear):
        yield f"{name}.weight", mod.weight, True
        if mod.bias is not None:
            yield f"{name}.bias", mod.bias, False

    def norm(name: str, mod: nn.LayerNorm):
        yield f"{name}.weight", mod.weight, False
        yield f"{name}.bias", mod.bias, False

    def layer(prefix: str, mod):
        for p in ("q", "k", "v", "o"):
            yield from linear(f"{prefix}.attn.{p}", getattr(mod.attn, p))
        yield from norm(f"{prefix}.attn.norm", mod.attn_norm)
        yield from linear(f"{prefix}.ffn.up", mod.ffn_up)
        yield from linear(f"{prefix}.ffn.down", mod.ffn_down)
        yield from norm(f"{prefix}.ffn.norm", mod.ffn_norm)

    yield "embeddings.word.weight", model.word_emb.weight, False
    yield "embeddings.position.weight", model.pos_emb.weight, False
    yield from norm("embeddings.norm", model.emb_norm)
    for i, mod in enumerate(model.encoder_layers):
        yield from layer(f"encoder.layer.{i}", mod)

    yield "schema.embeddings.segment_kind.weight", model.segment_kind_emb.weight, False
    yield "schema.embeddings.tool_index.weight", model.tool_index_emb.weight, False
    yield from norm("schema.embeddings.norm", model.schema_norm)
    for i, mod in enumerate(model.schema_layers):
        yield from layer(f"schema.layer.{i}", mod)

    yield "fusion.no_tool.embedding", model.no_tool, False
    for i, block in enumerate(model.fusion_blocks):
        for part, attn, nrm in (
            ("self", block.self_attn, block.self_norm),
            ("cross", block.cross_attn, block.cross_norm),
        ):
            for p in ("q", "k", "v", "o"):
                yield from linear(f"fusion.block.{i}.{part}.{p}", getattr(attn, p))
            yield from norm(f"fusion.block.{i}.{part}.norm", nrm)
        yield from linear(f"fusion.block.{i}.ffn.up", block.ffn_up)
        yield from linear(f"fusion.block.{i}.ffn.down", block.ffn_down)
        yield from norm(f"fusion.block.{i}.ffn.norm", block.ffn_norm)

    yield from linear("heads.action.dense", model.action_head.dense)
    yield from linear("heads.action.out", model.action_head.out)
    yield from linear("heads.tool.dense", model.tool_head.dense)
    yield from linear("heads.tool.out", model.tool_head.out)
    yield from linear("heads.presence.dense", model.presence_head.dense)
    yield from linear("heads.presence.out", model.presence_head.out)
    yield from linear("heads.boolean.out", model.boolean_out)
    yield from linear("heads.span.start", model.span_start)
    yield from linear("heads.span.end", model.span_end)
    yield from linear("heads.enum", model.enum_proj)
    yield from linear("heads.numeric.unit", model.numeric_unit)
    yield from linear("heads.numeric.magnitude", model.numeric_magnitude)
    yield from linear("heads.datetime.relation", model.datetime_relation)
    yield from linear("heads.datetime.weekday", model.datetime_weekday)
    yield from linear("heads.datetime.daypart", model.datetime_daypart)
    yield from linear("heads.datetime.month", model.datetime_month)


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
