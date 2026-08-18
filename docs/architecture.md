# Architecture

See `specs/ntc_web_technical_concept.md` for the full concept. This page maps
the concept onto the code.

## Inference pipeline (spec §72)

```
CompileRequest ──► NtcTokenizer (ntc-core)             tokenize + offsets
              ──► ToolRegistry.resolve_candidates      ≤16 candidates (V1)
              ──► ModelInputs::pack (ntc-model)        pad + segment kinds + anchors
              ──► Backend::run                         CpuRefBackend | WgpuBackend
              ──► Decoder (ntc-runtime)                logits → typed predictions
              ──► ConfidencePolicy                     spec §46 downgrades
              ──► validate (ntc-core)                  §64 taxonomy codes
              ──► normalize + serialize (ntc-runtime)  jiff dates, units → JSON
              ──► CompileOutcome { CALL | ASK | NO_CALL }
```

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
