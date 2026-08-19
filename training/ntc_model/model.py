"""PyTorch `ntc_encoder_heads_v1` — the training-side twin of the Rust CPU
reference backend (`crates/ntc-model/src/cpu.rs`).

Semantics mirrored exactly:

- post-LN transformer layers: self-attn -> add -> LayerNorm -> FFN (exact erf
  GELU) -> add -> LayerNorm;
- utterance embeddings: word + position (from 0) -> LayerNorm;
- schema encoder per tool independently: word + position + segment_kind +
  tool_index embeddings -> LayerNorm -> layers;
- fusion over the packed [T*Ls + 1 NO_TOOL] states: per block self-attn ->
  add+LN -> cross-attn (Q=packed, KV=user states) -> add+LN -> FFN -> add+LN;
- heads per contracts/heads/v1/head-spec.json (exact output names).

Masking: masked key positions get `float32.min` (= Rust `f32::MIN`) before
softmax — never `-inf`, so fully-masked rows soften to a uniform distribution
instead of NaN. Padded rows are never read by any head, so this difference
from the Rust reference (which zeroes them) cannot leak into head outputs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from ntc_model.config import SEGMENT_KINDS, NtcArchConfig

#: Additive mask value: exactly `f32::MIN` as used by the Rust reference.
MASK_VALUE = torch.finfo(torch.float32).min

#: The head-spec output tensor names (contracts/heads/v1/head-spec.json).
HEAD_OUTPUT_NAMES = (
    "action.logits",
    "tool.logits",
    "presence.logits",
    "boolean.logits",
    "span.start.logits",
    "span.end.logits",
    "enum.logits",
    "numeric.unit.logits",
    "numeric.magnitude",
    "datetime.relation.logits",
    "datetime.weekday.logits",
    "datetime.daypart.logits",
    "datetime.month.logits",
)


class Attention(nn.Module):
    """Multi-head attention: `q_states` attends over `kv_states`.

    `kv_mask[b, j] == False` masks key/value position `j` (score = f32::MIN).
    """

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)

    def forward(self, q_states: Tensor, kv_states: Tensor, kv_mask: Tensor) -> Tensor:
        b, lq, hidden = q_states.shape
        lkv = kv_states.shape[1]
        nh, hd = self.heads, self.head_dim

        q = self.q(q_states).view(b, lq, nh, hd).transpose(1, 2)  # [B, nh, Lq, hd]
        k = self.k(kv_states).view(b, lkv, nh, hd).transpose(1, 2)
        v = self.v(kv_states).view(b, lkv, nh, hd).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (hd**0.5)  # [B, nh, Lq, Lkv]
        scores = scores.masked_fill(~kv_mask[:, None, None, :], MASK_VALUE)
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)  # [B, nh, Lq, hd]
        ctx = ctx.transpose(1, 2).reshape(b, lq, hidden)
        return self.o(ctx)


class TransformerLayer(nn.Module):
    """One post-LN self-attention transformer layer."""

    def __init__(self, cfg: NtcArchConfig):
        super().__init__()
        self.attn = Attention(cfg.hidden, cfg.heads)
        self.attn_norm = nn.LayerNorm(cfg.hidden, eps=cfg.layer_norm_eps)
        self.ffn_up = nn.Linear(cfg.hidden, cfg.ffn)
        self.ffn_down = nn.Linear(cfg.ffn, cfg.hidden)
        self.ffn_norm = nn.LayerNorm(cfg.hidden, eps=cfg.layer_norm_eps)
        self.act = nn.GELU()  # exact erf GELU (matches the Rust reference)

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        states = self.attn_norm(states + self.attn(states, states, mask))
        states = self.ffn_norm(states + self.ffn_down(self.act(self.ffn_up(states))))
        return states


class FusionBlock(nn.Module):
    """Self-attn -> add+LN -> cross-attn (KV=user) -> add+LN -> FFN -> add+LN."""

    def __init__(self, cfg: NtcArchConfig):
        super().__init__()
        self.self_attn = Attention(cfg.hidden, cfg.heads)
        self.self_norm = nn.LayerNorm(cfg.hidden, eps=cfg.layer_norm_eps)
        self.cross_attn = Attention(cfg.hidden, cfg.heads)
        self.cross_norm = nn.LayerNorm(cfg.hidden, eps=cfg.layer_norm_eps)
        self.ffn_up = nn.Linear(cfg.hidden, cfg.ffn)
        self.ffn_down = nn.Linear(cfg.ffn, cfg.hidden)
        self.ffn_norm = nn.LayerNorm(cfg.hidden, eps=cfg.layer_norm_eps)
        self.act = nn.GELU()

    def forward(
        self, packed: Tensor, packed_mask: Tensor, user_states: Tensor, user_mask: Tensor
    ) -> Tensor:
        packed = self.self_norm(packed + self.self_attn(packed, packed, packed_mask))
        packed = self.cross_norm(packed + self.cross_attn(packed, user_states, user_mask))
        packed = self.ffn_norm(packed + self.ffn_down(self.act(self.ffn_up(packed))))
        return packed


class MlpHead(nn.Module):
    """`dense -> gelu -> out` MLP head."""

    def __init__(self, in_dim: int, hidden: int, classes: int):
        super().__init__()
        self.dense = nn.Linear(in_dim, hidden)
        self.out = nn.Linear(hidden, classes)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.act(self.dense(x)))


# Host element kinds a linked item can have (frozen order; index 0 = unknown).
LINKED_KINDS = ["asset", "document", "object", "data-object", "folder"]
MAX_LINKED = 8


class NtcEncoderHeadsV1(nn.Module):
    """The complete model. See the module docstring for the forward contract."""

    def __init__(self, cfg: NtcArchConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden

        # Shared embeddings (utterance + schema encoders).
        self.word_emb = nn.Embedding(cfg.vocab, h)
        self.pos_emb = nn.Embedding(cfg.max_positions, h)
        self.emb_norm = nn.LayerNorm(h, eps=cfg.layer_norm_eps)
        self.encoder_layers = nn.ModuleList(
            TransformerLayer(cfg) for _ in range(cfg.encoder_layers)
        )

        self.segment_kind_emb = nn.Embedding(SEGMENT_KINDS, h)
        self.tool_index_emb = nn.Embedding(cfg.max_tools + 1, h)
        self.schema_norm = nn.LayerNorm(h, eps=cfg.layer_norm_eps)
        self.schema_layers = nn.ModuleList(
            TransformerLayer(cfg) for _ in range(cfg.schema_layers)
        )

        self.no_tool = nn.Parameter(torch.empty(h).normal_(std=0.02))
        self.fusion_blocks = nn.ModuleList(FusionBlock(cfg) for _ in range(cfg.fusion_blocks))

        self.action_head = MlpHead(2 * h, h, cfg.action_classes)
        self.tool_head = MlpHead(h, h, 1)
        self.presence_head = MlpHead(h, h, 4)
        self.boolean_out = nn.Linear(h, 2)
        self.span_start = nn.Linear(h, h, bias=False)
        self.span_end = nn.Linear(h, h, bias=False)
        self.enum_proj = nn.Linear(h, h, bias=False)
        self.numeric_unit = nn.Linear(h, 6)
        self.numeric_magnitude = nn.Linear(h, 1)
        self.datetime_relation = nn.Linear(h, 10)
        self.datetime_weekday = nn.Linear(h, 8)
        self.datetime_daypart = nn.Linear(h, 6)
        self.datetime_month = nn.Linear(h, 13)
        # Head codec v3: why the router escalated / declined, where each value
        # comes from, which linked item it binds, and why one is unfillable.
        self.delegate_reason_head = MlpHead(2 * h, h, 4)
        self.no_call_reason_head = MlpHead(2 * h, h, 5)
        self.source_out = nn.Linear(h, 4)
        self.unresolved_reason_out = nn.Linear(h, 5)
        # Linked items are described by a small typed embedding rather than
        # text: the model needs their *kind* and position, not their key.
        self.linked_kind_emb = nn.Embedding(len(LINKED_KINDS) + 1, h)
        self.linked_pos_emb = nn.Embedding(MAX_LINKED + 1, h)
        self.entity_proj = nn.Linear(h, h, bias=False)
        self.entity_none = nn.Parameter(torch.empty(h).normal_(std=0.02))

    # -- encoders ----------------------------------------------------------

    def encode_utterance(self, ids: Tensor, mask: Tensor) -> Tensor:
        lu = ids.shape[1]
        pos = torch.arange(lu, device=ids.device)
        x = self.word_emb(ids) + self.pos_emb(pos)[None, :, :]
        x = self.emb_norm(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x

    def encode_tools(self, ids: Tensor, mask: Tensor, kinds: Tensor) -> Tensor:
        """ids/mask/kinds: [B, T, Ls]. Each tool encoded independently."""
        b, t, ls = ids.shape
        pos = torch.arange(ls, device=ids.device)
        tool_idx = torch.arange(t, device=ids.device)
        x = (
            self.word_emb(ids)
            + self.pos_emb(pos)[None, None, :, :]
            + self.segment_kind_emb(kinds)
            + self.tool_index_emb(tool_idx)[None, :, None, :]
        )
        x = self.schema_norm(x)
        x = x.view(b * t, ls, -1)
        flat_mask = mask.reshape(b * t, ls)
        for layer in self.schema_layers:
            x = layer(x, flat_mask)
        return x.view(b, t, ls, -1)

    def fuse(
        self,
        tool_states: Tensor,
        schema_mask: Tensor,
        user_states: Tensor,
        user_mask: Tensor,
    ) -> Tensor:
        """tool_states [B, T, Ls, H] -> fused packed states [B, T*Ls + 1, H]."""
        b, t, ls, h = tool_states.shape
        packed = torch.cat(
            [tool_states.reshape(b, t * ls, h), self.no_tool[None, None, :].expand(b, 1, h)],
            dim=1,
        )
        packed_mask = torch.cat(
            [
                schema_mask.reshape(b, t * ls),
                torch.ones(b, 1, dtype=torch.bool, device=tool_states.device),
            ],
            dim=1,
        )
        for block in self.fusion_blocks:
            packed = block(packed, packed_mask, user_states, user_mask)
        return packed

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        utterance_ids: Tensor,  # [B, Lu] long
        utterance_mask: Tensor,  # [B, Lu] bool
        schema_ids: Tensor,  # [B, T, Ls] long
        schema_mask: Tensor,  # [B, T, Ls] bool
        schema_kinds: Tensor,  # [B, T, Ls] long
        tool_count: Tensor,  # [B] long — number of real candidate tools
        tool_anchors: Tensor,  # [B, T] long — first schema token per tool
        arg_anchors: Tensor,  # [B, T, A] long — arg-name anchor tokens
        arg_mask: Tensor,  # [B, T, A] bool — declared (real) args
        enum_anchors: Tensor,  # [B, T, A, E] long — enum-value anchor tokens
        enum_mask: Tensor,  # [B, T, A, E] bool — declared enum values
        linked_kinds: Tensor | None = None,  # [B, N] long — context.linked types
        linked_mask: Tensor | None = None,  # [B, N] bool
    ) -> dict[str, Tensor]:
        cfg = self.cfg
        b, t, ls = schema_ids.shape
        a = arg_anchors.shape[2]
        e = enum_anchors.shape[3]
        h = cfg.hidden

        user_states = self.encode_utterance(utterance_ids, utterance_mask)
        tool_states = self.encode_tools(schema_ids, schema_mask, schema_kinds)
        fused = self.fuse(tool_states, schema_mask, user_states, utterance_mask)

        seg_off = torch.arange(t, device=fused.device) * ls  # [T]

        def gather_states(flat_idx: Tensor) -> Tensor:
            """fused [B, S, H] at flat_idx [B, N] -> [B, N, H]."""
            return torch.gather(fused, 1, flat_idx.unsqueeze(-1).expand(-1, -1, h))

        user_cls = user_states[:, 0]  # [B, H]
        global_state = fused[:, t * ls]  # NO_TOOL slot, [B, H]

        out: dict[str, Tensor] = {}

        # action: concat(user token 0, NO_TOOL fused state) -> MLP.
        out["action.logits"] = self.action_head(torch.cat([user_cls, global_state], dim=-1))

        # tool: MLP scored at each tool's first schema token + NO_TOOL.
        valid_tool = torch.arange(t, device=fused.device)[None, :] < tool_count[:, None]  # [B,T]
        tool_state = gather_states(tool_anchors + seg_off[None, :])  # [B, T, H]
        tool_logits = self.tool_head(tool_state).squeeze(-1)  # [B, T]
        tool_logits = tool_logits.masked_fill(~valid_tool, MASK_VALUE)
        no_tool_logit = self.tool_head(global_state)  # [B, 1]
        out["tool.logits"] = torch.cat([tool_logits, no_tool_logit], dim=1)  # [B, T+1]

        # Per-arg heads at arg-name anchor states.
        arg_valid = arg_mask & valid_tool[:, :, None]  # [B, T, A]
        arg_idx = (arg_anchors + seg_off[None, :, None]).reshape(b, t * a)
        arg_states = gather_states(arg_idx).view(b, t, a, h)

        def mask_arg(x: Tensor, fill: float = MASK_VALUE) -> Tensor:
            return x.masked_fill(~arg_valid[..., None], fill)

        out["presence.logits"] = mask_arg(self.presence_head(arg_states))
        out["boolean.logits"] = mask_arg(self.boolean_out(arg_states))
        out["numeric.unit.logits"] = mask_arg(self.numeric_unit(arg_states))
        out["numeric.magnitude"] = mask_arg(self.numeric_magnitude(arg_states), fill=0.0)
        out["datetime.relation.logits"] = mask_arg(self.datetime_relation(arg_states))
        out["datetime.weekday.logits"] = mask_arg(self.datetime_weekday(arg_states))
        out["datetime.daypart.logits"] = mask_arg(self.datetime_daypart(arg_states))
        out["datetime.month.logits"] = mask_arg(self.datetime_month(arg_states))

        # Span heads: bilinear (arg_state · W) · user_states over real tokens.
        span_allowed = arg_valid[:, :, :, None] & utterance_mask[:, None, None, :]  # [B,T,A,Lu]
        for name, proj in (
            ("span.start.logits", self.span_start),
            ("span.end.logits", self.span_end),
        ):
            scores = torch.einsum("btah,blh->btal", proj(arg_states), user_states)
            out[name] = scores.masked_fill(~span_allowed, MASK_VALUE)

        # Enum head: bilinear (arg_state · W) · enum-value anchor states.
        enum_idx = (enum_anchors + seg_off[None, :, None, None]).reshape(b, t * a * e)
        enum_targets = gather_states(enum_idx).view(b, t, a, e, h)
        enum_scores = (self.enum_proj(arg_states)[:, :, :, None, :] * enum_targets).sum(-1)
        enum_allowed = enum_mask & arg_valid[..., None]
        out["enum.logits"] = enum_scores.masked_fill(~enum_allowed, MASK_VALUE)

        # Escalation reasons read the same pooled state as the action head:
        # they refine one whole-utterance verdict, not a per-argument one.
        pooled = torch.cat([user_cls, global_state], dim=-1)
        out["delegate_reason.logits"] = self.delegate_reason_head(pooled)
        out["no_call_reason.logits"] = self.no_call_reason_head(pooled)

        # Per-argument: where the value comes from, and why it is unfillable.
        out["source.logits"] = mask_arg(self.source_out(arg_states))
        out["unresolved_reason.logits"] = mask_arg(self.unresolved_reason_out(arg_states))

        # Entity reference: bilinear match of each argument against the
        # linked items, plus a learned NONE slot at the end.
        n_linked = MAX_LINKED if linked_kinds is None else linked_kinds.shape[1]
        if linked_kinds is None:
            linked_kinds = torch.zeros(b, n_linked, dtype=torch.long, device=fused.device)
            linked_mask = torch.zeros(b, n_linked, dtype=torch.bool, device=fused.device)
        pos = torch.arange(n_linked, device=fused.device)
        linked_states = self.linked_kind_emb(linked_kinds) + self.linked_pos_emb(pos)[None]
        ent_scores = torch.einsum(
            "btah,bnh->btan", self.entity_proj(arg_states), linked_states
        )
        none_score = (self.entity_proj(arg_states) * self.entity_none).sum(-1, keepdim=True)
        ent_scores = torch.cat([ent_scores, none_score], dim=-1)
        allowed = torch.cat(
            [linked_mask[:, None, None, :].expand(b, t, a, n_linked),
             torch.ones(b, t, a, 1, dtype=torch.bool, device=fused.device)],
            dim=-1,
        ) & arg_valid[..., None]
        out["entity_ref.logits"] = ent_scores.masked_fill(~allowed, MASK_VALUE)

        return out


# --- canonical tensor manifest + export ------------------------------------


def tensor_specs(cfg: NtcArchConfig) -> list[tuple[str, list[int]]]:
    """Every canonical tensor name with its expected shape — mirrors
    `ntc_model::weights::tensor_specs` exactly (order included)."""
    h, f = cfg.hidden, cfg.ffn
    specs: list[tuple[str, list[int]]] = []

    specs.append(("embeddings.word.weight", [cfg.vocab, h]))
    specs.append(("embeddings.position.weight", [cfg.max_positions, h]))
    specs.append(("embeddings.norm.weight", [h]))
    specs.append(("embeddings.norm.bias", [h]))

    def layer(prefix: str) -> None:
        for p in ("q", "k", "v", "o"):
            specs.append((f"{prefix}.attn.{p}.weight", [h, h]))
            specs.append((f"{prefix}.attn.{p}.bias", [h]))
        specs.append((f"{prefix}.attn.norm.weight", [h]))
        specs.append((f"{prefix}.attn.norm.bias", [h]))
        specs.append((f"{prefix}.ffn.up.weight", [h, f]))
        specs.append((f"{prefix}.ffn.up.bias", [f]))
        specs.append((f"{prefix}.ffn.down.weight", [f, h]))
        specs.append((f"{prefix}.ffn.down.bias", [h]))
        specs.append((f"{prefix}.ffn.norm.weight", [h]))
        specs.append((f"{prefix}.ffn.norm.bias", [h]))

    for i in range(cfg.encoder_layers):
        layer(f"encoder.layer.{i}")

    specs.append(("schema.embeddings.segment_kind.weight", [SEGMENT_KINDS, h]))
    specs.append(("schema.embeddings.tool_index.weight", [cfg.max_tools + 1, h]))
    specs.append(("schema.embeddings.norm.weight", [h]))
    specs.append(("schema.embeddings.norm.bias", [h]))
    for i in range(cfg.schema_layers):
        layer(f"schema.layer.{i}")

    specs.append(("fusion.no_tool.embedding", [h]))
    for i in range(cfg.fusion_blocks):
        for part in ("self", "cross"):
            for p in ("q", "k", "v", "o"):
                specs.append((f"fusion.block.{i}.{part}.{p}.weight", [h, h]))
                specs.append((f"fusion.block.{i}.{part}.{p}.bias", [h]))
            specs.append((f"fusion.block.{i}.{part}.norm.weight", [h]))
            specs.append((f"fusion.block.{i}.{part}.norm.bias", [h]))
        specs.append((f"fusion.block.{i}.ffn.up.weight", [h, f]))
        specs.append((f"fusion.block.{i}.ffn.up.bias", [f]))
        specs.append((f"fusion.block.{i}.ffn.down.weight", [f, h]))
        specs.append((f"fusion.block.{i}.ffn.down.bias", [h]))
        specs.append((f"fusion.block.{i}.ffn.norm.weight", [h]))
        specs.append((f"fusion.block.{i}.ffn.norm.bias", [h]))

    for dense, outp, classes in (
        ("heads.action.dense", "heads.action.out", cfg.action_classes),
        ("heads.tool.dense", "heads.tool.out", 1),
        ("heads.presence.dense", "heads.presence.out", 4),
    ):
        in_dim = 2 * h if dense == "heads.action.dense" else h
        specs.append((f"{dense}.weight", [in_dim, h]))
        specs.append((f"{dense}.bias", [h]))
        specs.append((f"{outp}.weight", [h, classes]))
        specs.append((f"{outp}.bias", [classes]))
    for name in ("heads.span.start.weight", "heads.span.end.weight", "heads.enum.weight"):
        specs.append((name, [h, h]))
    specs.append(("heads.delegate_reason.dense.weight", [2 * h, h]))
    specs.append(("heads.delegate_reason.dense.bias", [h]))
    specs.append(("heads.delegate_reason.out.weight", [h, 4]))
    specs.append(("heads.delegate_reason.out.bias", [4]))
    specs.append(("heads.no_call_reason.dense.weight", [2 * h, h]))
    specs.append(("heads.no_call_reason.dense.bias", [h]))
    specs.append(("heads.no_call_reason.out.weight", [h, 5]))
    specs.append(("heads.no_call_reason.out.bias", [5]))
    specs.append(("heads.entity.proj.weight", [h, h]))
    specs.append(("heads.entity.none.embedding", [h]))
    specs.append(("context.linked_kind.weight", [len(LINKED_KINDS) + 1, h]))
    specs.append(("context.linked_pos.weight", [MAX_LINKED + 1, h]))
    for name, classes in (
        ("heads.source.out", 4),
        ("heads.unresolved_reason.out", 5),
        ("heads.boolean.out", 2),
        ("heads.numeric.unit", 6),
        ("heads.numeric.magnitude", 1),
        ("heads.datetime.relation", 10),
        ("heads.datetime.weekday", 8),
        ("heads.datetime.daypart", 6),
        ("heads.datetime.month", 13),
    ):
        specs.append((f"{name}.weight", [h, classes]))
        specs.append((f"{name}.bias", [classes]))
    return specs


def export_tensors(model: NtcEncoderHeadsV1, cfg: NtcArchConfig) -> dict[str, np.ndarray]:
    """Canonical Rust tensor names -> float32 arrays.

    `nn.Linear` stores weights `[out, in]`; the runtime contract is `[in, out]`
    (kernels compute `y = x·W + b`), so linear weights are transposed here.
    Embedding tables and LayerNorm params export as-is.
    """
    out: dict[str, np.ndarray] = {}

    def npy(t: Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    def linear(name: str, mod: nn.Linear) -> None:
        out[f"{name}.weight"] = np.ascontiguousarray(npy(mod.weight).T)
        if mod.bias is not None:
            out[f"{name}.bias"] = npy(mod.bias)

    def norm(name: str, mod: nn.LayerNorm) -> None:
        out[f"{name}.weight"] = npy(mod.weight)
        out[f"{name}.bias"] = npy(mod.bias)

    def layer(prefix: str, mod: TransformerLayer) -> None:
        for p in ("q", "k", "v", "o"):
            linear(f"{prefix}.attn.{p}", getattr(mod.attn, p))
        norm(f"{prefix}.attn.norm", mod.attn_norm)
        linear(f"{prefix}.ffn.up", mod.ffn_up)
        linear(f"{prefix}.ffn.down", mod.ffn_down)
        norm(f"{prefix}.ffn.norm", mod.ffn_norm)

    out["embeddings.word.weight"] = npy(model.word_emb.weight)
    out["embeddings.position.weight"] = npy(model.pos_emb.weight)
    norm("embeddings.norm", model.emb_norm)
    for i, mod in enumerate(model.encoder_layers):
        layer(f"encoder.layer.{i}", mod)

    out["schema.embeddings.segment_kind.weight"] = npy(model.segment_kind_emb.weight)
    out["schema.embeddings.tool_index.weight"] = npy(model.tool_index_emb.weight)
    norm("schema.embeddings.norm", model.schema_norm)
    for i, mod in enumerate(model.schema_layers):
        layer(f"schema.layer.{i}", mod)

    out["fusion.no_tool.embedding"] = npy(model.no_tool)
    for i, block in enumerate(model.fusion_blocks):
        for part, attn, nrm in (
            ("self", block.self_attn, block.self_norm),
            ("cross", block.cross_attn, block.cross_norm),
        ):
            for p in ("q", "k", "v", "o"):
                linear(f"fusion.block.{i}.{part}.{p}", getattr(attn, p))
            norm(f"fusion.block.{i}.{part}.norm", nrm)
        linear(f"fusion.block.{i}.ffn.up", block.ffn_up)
        linear(f"fusion.block.{i}.ffn.down", block.ffn_down)
        norm(f"fusion.block.{i}.ffn.norm", block.ffn_norm)

    linear("heads.action.dense", model.action_head.dense)
    linear("heads.action.out", model.action_head.out)
    linear("heads.tool.dense", model.tool_head.dense)
    linear("heads.tool.out", model.tool_head.out)
    linear("heads.presence.dense", model.presence_head.dense)
    linear("heads.presence.out", model.presence_head.out)
    linear("heads.delegate_reason.dense", model.delegate_reason_head.dense)
    linear("heads.delegate_reason.out", model.delegate_reason_head.out)
    linear("heads.no_call_reason.dense", model.no_call_reason_head.dense)
    linear("heads.no_call_reason.out", model.no_call_reason_head.out)
    linear("heads.source.out", model.source_out)
    linear("heads.unresolved_reason.out", model.unresolved_reason_out)
    linear("heads.entity.proj", model.entity_proj)
    out["heads.entity.none.embedding"] = npy(model.entity_none)
    out["context.linked_kind.weight"] = npy(model.linked_kind_emb.weight)
    out["context.linked_pos.weight"] = npy(model.linked_pos_emb.weight)
    linear("heads.boolean.out", model.boolean_out)
    linear("heads.span.start", model.span_start)
    linear("heads.span.end", model.span_end)
    linear("heads.enum", model.enum_proj)
    linear("heads.numeric.unit", model.numeric_unit)
    linear("heads.numeric.magnitude", model.numeric_magnitude)
    linear("heads.datetime.relation", model.datetime_relation)
    linear("heads.datetime.weekday", model.datetime_weekday)
    linear("heads.datetime.daypart", model.datetime_daypart)
    linear("heads.datetime.month", model.datetime_month)

    specs = tensor_specs(cfg)
    expected = {name: shape for name, shape in specs}
    if set(out) != set(expected):
        missing = sorted(set(expected) - set(out))
        extra = sorted(set(out) - set(expected))
        raise AssertionError(f"tensor name drift: missing={missing} extra={extra}")
    for name, shape in specs:
        got = list(out[name].shape)
        if got != shape:
            raise AssertionError(f"tensor `{name}`: shape {got}, expected {shape}")
    return out
