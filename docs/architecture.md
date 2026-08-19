# Architecture

See `specs/ntc_web_technical_concept.md` for the full concept. This page maps
the concept onto the code.

## Inference pipeline (spec §72)

```
CompileRequest ──► NtcTokenizer (ntc-core)             tokenize + offsets
              ──► ToolRegistry.resolve_candidates      ≤256 tools offered
              ──► shortlist (if wider than one slate)  N/slate scoring passes
              ──► ModelInputs::pack (ntc-model)        pad + segment kinds + anchors
              ──► Backend::run                         CpuRefBackend | WgpuBackend
              ──► Decoder (ntc-runtime)                logits → typed predictions
              ──► ConfidencePolicy                     spec §46 downgrades
              ──► validate (ntc-core)                  §64 taxonomy codes
              ──► normalize + serialize (ntc-runtime)  jiff dates, units → JSON
              ──► CompileOutcome { CALL | ASK | NO_CALL | DELEGATE }
```

## Candidate narrowing (spec §21–22)

The model reads a fixed-width slate; an MCP host registers everything it has.
Something must choose, and "whatever the caller passed" is not a strategy —
it makes the host responsible for the router's accuracy. So a wide tool set is
narrowed by the model itself, in two steps:

1. **Shortlist** — the set is split into slate-sized groups and each is scored.
   A tool's score is its logit's margin over **that group's own NO_TOOL**.
   NO_TOOL is the only option present in every group, so it is the one usable
   common reference; raw logits are not comparable across groups, and per-group
   softmax is worse than nothing, because a group of three strong candidates
   splits its mass while a group of three decoys does not.
2. **Decide** — one pass over the survivors *together*. Groups are scored
   independently, so the survivors have been ranked against a baseline but
   never compared with each other. Fusion's schema self-attention is what
   discriminates near-identical siblings (`get_asset` / `list_assets` /
   `search_assets`), and it only sees tools that share a slate.

Cost is about `N/slate + 1` forward passes, which is why `MAX_CANDIDATES` is a
cost bound (256) rather than a model bound. The 10,000-tool case needs the
embedding retriever of §21–22, not an exhaustive sweep; `CandidateSelector` is
the seam for it. Per-tool schema encodings are independent (block-diagonal), so
caching them across the shortlist rounds is the obvious next optimization.

`NeuralToolCompiler::shortlist` is public and returns *every* tool's score, not
just the survivors, so a host can distinguish one clear winner from a field
bunched inside the noise — the latter is an ASK signal, not a coin flip.

## Model (ntc_encoder_heads_v1)

- Utterance encoder: post-LN transformer over word+position embeddings.
- Schema encoder: per-tool (block-diagonal) over word+position+segment-kind+
  tool-index embeddings — each tool encodes independently, enabling future
  per-tool caching.
- Fusion: packed tool states + NO_TOOL pseudo-slot; per block: schema
  self-attention (cross-tool decoy discrimination) → cross-attention to user
  states → FFN; all post-LN.
- Heads: see `contracts/heads/v1/head-spec.json`. Head projections and all
  argmax/softmax run on CPU — head tensors are tiny; kernels stay few.

## Backends

`ntc_model::Backend` is the seam. `CpuRefBackend` is normative
(`docs/parity-testing.md`); `WgpuBackend` (ntc-webgpu) must reproduce it.
The wasm build uses the CPU backend until the WebGPU-in-browser milestone.

## Design invariants

- No autoregressive JSON: one structured forward pass per compile (spec §9).
- Anything deterministic never consumes model capacity (spec §6.2): dates,
  units, numbers, validation, serialization are Rust.
- The neural model is never the security boundary (spec §45).
- Single implementation for every train/serve-shared computation: canonical
  text rendering, tokenizer, anchor discovery rules.
