"""Shape/contract tests for the PyTorch ntc_encoder_heads_v1 implementation."""

import numpy as np
import torch

from ntc_model.config import tiny_config
from ntc_model.model import (
    HEAD_OUTPUT_NAMES,
    MASK_VALUE,
    NtcEncoderHeadsV1,
    export_tensors,
    tensor_specs,
)


def make_inputs(cfg, b=2, t=2, a=None, e=None):
    torch.manual_seed(7)
    a = a or cfg.max_args
    e = e or cfg.max_enum_values
    lu, ls = cfg.max_utterance_tokens, cfg.max_schema_tokens

    utterance_ids = torch.randint(0, cfg.vocab, (b, lu))
    utterance_mask = torch.zeros(b, lu, dtype=torch.bool)
    utterance_mask[:, :7] = True

    schema_ids = torch.randint(0, cfg.vocab, (b, t, ls))
    schema_mask = torch.zeros(b, t, ls, dtype=torch.bool)
    schema_mask[:, :, :20] = True
    schema_kinds = torch.randint(0, 9, (b, t, ls))

    tool_count = torch.tensor([t, t - 1])
    tool_anchors = torch.ones(b, t, dtype=torch.long)
    arg_anchors = torch.randint(2, 20, (b, t, a))
    arg_mask = torch.zeros(b, t, a, dtype=torch.bool)
    arg_mask[:, :, :2] = True
    enum_anchors = torch.randint(2, 20, (b, t, a, e))
    enum_mask = torch.zeros(b, t, a, e, dtype=torch.bool)
    enum_mask[:, :, 1, :3] = True  # arg 1 has 3 enum values

    return dict(
        utterance_ids=utterance_ids,
        utterance_mask=utterance_mask,
        schema_ids=schema_ids,
        schema_mask=schema_mask,
        schema_kinds=schema_kinds,
        tool_count=tool_count,
        tool_anchors=tool_anchors,
        arg_anchors=arg_anchors,
        arg_mask=arg_mask,
        enum_anchors=enum_anchors,
        enum_mask=enum_mask,
    )


def test_forward_output_names_and_shapes():
    cfg = tiny_config()
    model = NtcEncoderHeadsV1(cfg)
    b, t, a, e, lu = 2, 2, cfg.max_args, cfg.max_enum_values, cfg.max_utterance_tokens
    with torch.no_grad():
        out = model(**make_inputs(cfg, b=b, t=t))

    assert set(out) == set(HEAD_OUTPUT_NAMES)
    assert out["action.logits"].shape == (b, 3)
    assert out["tool.logits"].shape == (b, t + 1)
    assert out["presence.logits"].shape == (b, t, a, 4)
    assert out["boolean.logits"].shape == (b, t, a, 2)
    assert out["span.start.logits"].shape == (b, t, a, lu)
    assert out["span.end.logits"].shape == (b, t, a, lu)
    assert out["enum.logits"].shape == (b, t, a, e)
    assert out["numeric.unit.logits"].shape == (b, t, a, 6)
    assert out["numeric.magnitude"].shape == (b, t, a, 1)
    assert out["datetime.relation.logits"].shape == (b, t, a, 10)
    assert out["datetime.weekday.logits"].shape == (b, t, a, 8)
    assert out["datetime.daypart.logits"].shape == (b, t, a, 6)
    assert out["datetime.month.logits"].shape == (b, t, a, 13)
    for name, tensor in out.items():
        assert not torch.isnan(tensor).any(), f"{name} contains NaN"


def test_padding_masks_applied():
    cfg = tiny_config()
    model = NtcEncoderHeadsV1(cfg)
    with torch.no_grad():
        out = model(**make_inputs(cfg))

    # Batch element 1 has tool_count=1: padded tool slot logit is f32::MIN.
    assert out["tool.logits"][1, 1].item() == MASK_VALUE
    # Padded arg slots (>= 2): logits f32::MIN, magnitude 0.
    assert (out["presence.logits"][:, :, 2:, :] == MASK_VALUE).all()
    assert (out["numeric.magnitude"][:, :, 2:, :] == 0).all()
    # Span logits masked outside the 7 real utterance tokens.
    assert (out["span.start.logits"][:, 0, 0, 7:] == MASK_VALUE).all()
    assert (out["span.start.logits"][0, 0, 0, :7] != MASK_VALUE).all()
    # Enum logits only live where enum_mask is set (arg 1, first 3 values).
    assert (out["enum.logits"][0, 0, 0, :] == MASK_VALUE).all()
    assert (out["enum.logits"][0, 0, 1, :3] != MASK_VALUE).all()
    assert (out["enum.logits"][0, 0, 1, 3:] == MASK_VALUE).all()


def test_export_matches_canonical_tensor_specs():
    cfg = tiny_config()
    torch.manual_seed(0)
    model = NtcEncoderHeadsV1(cfg)
    tensors = export_tensors(model, cfg)  # raises on any name/shape drift
    specs = tensor_specs(cfg)
    assert len(specs) == len(tensors) == 112  # pinned by the Rust tiny manifest
    for arr in tensors.values():
        assert arr.dtype == np.float32


def test_linear_weights_are_transposed_on_export():
    cfg = tiny_config()
    model = NtcEncoderHeadsV1(cfg)
    tensors = export_tensors(model, cfg)
    # nn.Linear stores [out, in]; export contract is [in, out].
    w = model.action_head.dense.weight.detach().numpy()  # [H, 2H]
    assert tensors["heads.action.dense.weight"].shape == (2 * cfg.hidden, cfg.hidden)
    assert np.array_equal(tensors["heads.action.dense.weight"], w.T)
    # Embeddings are exported as-is.
    assert np.array_equal(
        tensors["embeddings.word.weight"], model.word_emb.weight.detach().numpy()
    )
