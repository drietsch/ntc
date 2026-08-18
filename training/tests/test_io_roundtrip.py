"""Pins ntc_model.io's walk to export_tensors: export → import into a fresh
model → export again must be bit-identical."""

import numpy as np
import torch

from ntc_model.config import NtcArchConfig
from ntc_model.io import import_tensors
from ntc_model.model import NtcEncoderHeadsV1, export_tensors


def tiny_cfg() -> NtcArchConfig:
    return NtcArchConfig(
        hidden=32, heads=4, ffn=64, vocab=64, max_positions=128,
        encoder_layers=2, schema_layers=1, fusion_blocks=1,
        max_tools=4, max_args=4, max_enum_values=4,
        max_utterance_tokens=24, max_schema_tokens=64,
    )


def test_export_import_export_bit_identical():
    cfg = tiny_cfg()
    torch.manual_seed(123)
    a = NtcEncoderHeadsV1(cfg)
    exported = export_tensors(a, cfg)

    b = NtcEncoderHeadsV1(cfg)
    import_tensors(b, exported)
    re_exported = export_tensors(b, cfg)

    assert set(exported) == set(re_exported)
    for name, arr in exported.items():
        assert np.array_equal(arr, re_exported[name]), name
