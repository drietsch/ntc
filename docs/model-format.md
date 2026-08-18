# `.ntc` Model Format v1

Normative reader: `crates/ntc-format` (`NtcFile::parse`, `ntc verify`).
Production writer: `training/export/ntc_writer.py`.
Dev writer (fixtures, corrupt cases): `crates/ntc-format/src/writer.rs` (`write` feature).

## Binary layout (all integers little-endian)

| offset | size | field |
|---|---|---|
| 0 | 4 | magic `NTC1` |
| 4 | 4 | `format_version: u32` = 1 |
| 8 | 8 | `metadata_len: u64` |
| 16 | 8 | `tokenizer_len: u64` |
| 24 | 8 | `directory_len: u64` |
| 32 | 8 | `data_len: u64` |
| 40 | … | metadata JSON, tokenizer.json bytes, tensor directory JSON (concatenated, in that order) |
| … | … | zero padding to the next 256-byte boundary **from file start** |
| … | `data_len` | data section — each tensor blob 256-byte aligned, zero-padded gaps |
| end−32 | 32 | sha256 of everything before the footer |

Header sections precede the data so a browser can begin per-layer GPU upload
while the body streams. 256 = WebGPU's worst-case
`min_storage_buffer_offset_alignment`, so tensor slices can be uploaded or
bound without repacking.

## Metadata JSON

```json
{
  "architecture": "ntc_encoder_heads_v1",
  "model_version": "tiny-v1-seed42",
  "ir_version": 1,
  "abi_version": 1,
  "head_spec_version": 1,
  "tokenizer_sha256": "<hex sha256 of the tokenizer bytes>",
  "quantization": "f32 | f16 | bf16",
  "model": { "hidden": 768, "heads": 12, "ffn": 3072, "vocab": 64000,
             "max_positions": 512, "encoder_layers": 12, "schema_layers": 4,
             "fusion_blocks": 3, "max_tools": 16, "max_args": 16,
             "max_enum_values": 12, "max_utterance_tokens": 96,
             "max_schema_tokens": 96, "layer_norm_eps": 1e-5,
             "calibration": {"action": 1.0, "tool": 1.0, "presence": 1.0, "value": 1.0} },
  "semantic_types": ["..."]
}
```

The runtime loads the tokenizer **only** from the embedded bytes and checks
the sha256 — model/tokenizer skew is structurally impossible.

## Tensor directory

JSON array, offset-sorted, no duplicates:

```json
{"name": "encoder.layer.3.ffn.up.weight", "dtype": "F32",
 "shape": [768, 3072], "offset": 12345600, "byte_length": 9437184,
 "xxh64": "0123456789abcdef"}
```

- `offset` is relative to the data-section start; multiple of 256.
- `xxh64` = xxh64(bytes, seed 0) as 16 lowercase hex digits.
- dtypes valid in v1: `F32`, `F16`, `BF16`. Codes `TERNARY_T2`,
  `TERNARY_PACK20`, `I8` are **reserved** (Phase 2/3); v1 readers reject them
  with a distinct "reserved for a later phase" error, never "unknown".
- Dense tensors must satisfy `byte_length = prod(shape) × element_size`.

## Tensor naming + shapes

See the doc comment in `crates/ntc-model/src/weights.rs` (the
`tensor_specs()` function is the machine-readable manifest). Conventions:

- Linear weights are **[in, out]** row-major (`y = x·W + b`); PyTorch
  exporters transpose `nn.Linear`'s `[out, in]`.
- XLM-R position rows are de-offset by the exporter (runtime indexes `0..L`).
- The single token-type embedding row is folded into position embeddings.

## Input packing (head codec `inputs.packing`)

- Utterance: tokenizer output with special tokens, truncated to
  `max_utterance_tokens`.
- Each candidate tool: canonical ABI text (see `docs/tool-abi.md`) encoded
  independently, padded to `max_schema_tokens`; a tool whose anchors don't fit
  is a loud error, not a truncation.
- Per-token segment kinds: SPECIAL 0, TOOL_HEADER 1, DESC 2, ARG_NAME 3,
  INFO 4, TYPE 5, REQUIRED 6, SEMANTIC 7, ENUM_VALUE 8, PAD 9.
- Anchors (tool / arg-name / enum-value) are the first token whose byte range
  intersects the rendered anchor range (`crates/ntc-model/src/inputs.rs`).
- Fusion packs `n_tools × max_schema_tokens` states plus one trailing NO_TOOL
  slot seeded from `fusion.no_tool.embedding`.
