# NTC-Web

A browser-native **multilingual neural compiler** that translates human intent
into typed, executable tool calls — running locally via Rust → WebAssembly →
WebGPU. Not a chatbot, not a compressed LLM: a purpose-built semantic
compiler (see `specs/ntc_web_technical_concept.md`).

```
user language → multilingual encoder → schema encoder → fusion
             → structured prediction heads → Typed Action IR
             → deterministic Rust backend → validated JSON tool call
```

The neural network resolves only *uncertain semantics* (intent, tool choice,
argument binding, references, confidence). Everything exact — JSON, types,
enums, dates, units, policy, validation — is deterministic Rust.

## Repository layout

| Path | What |
|---|---|
| `crates/ntc-core` | Typed Action IR, canonical Tool ABI + schema compiler (single implementation), registry, validation, tokenizer wrapper |
| `crates/ntc-format` | `.ntc` model container: normative reader + verifier |
| `crates/ntc-model` | Architecture config, input packing, `Backend` trait, **CPU reference forward pass** |
| `crates/ntc-webgpu` | wgpu/WGSL inference backend (parity-tested against the CPU reference) |
| `crates/ntc-runtime` | Decode → IR → confidence policy → datetime/unit normalization → validation → JSON |
| `crates/ntc-wasm` | wasm-bindgen browser API |
| `crates/ntc-cli` | `ntc` dev CLI: `gen-schemas`, `schemac`, `verify`, `fixture-gen`, `infer` |
| `contracts/` | Versioned cross-workstream contracts (IR schemas, Tool ABI, head codec, compat matrix) |
| `fixtures/` | Golden corpora: schema-ABI renderings, IR accept/reject, tiny fixture models, tolerances |
| `training/` | PyTorch model + synthetic data engine (teacher: headless Claude Code) + `.ntc` exporter |
| `eval/` | Metrics, error taxonomy, parity tooling |
| `examples/browser/` | Static browser demo |
| `docs/` | Normative docs: model format, Tool ABI, Action IR, parity testing |

## Quick start

```sh
# Everything Rust: build + test (includes CPU-reference end-to-end tests)
cargo test --workspace --features ntc-format/write

# Generate the tiny fixture model and run a compile on it
cargo run -p ntc-cli -- fixture-gen --out fixtures/models/tiny-v1
cargo run -p ntc-cli -- infer \
  --model fixtures/models/tiny-v1/tiny.ntc \
  --utterance "make a dentist appointment tomorrow afternoon" \
  --tools examples/tools.json --timezone Europe/Berlin --now 2026-08-18T11:00:00+02:00

# Canonicalize a raw tool schema (the Python pipeline shells out to this)
echo '{"name":"search_trains","description":"Search trains","parameters":{...}}' \
  | cargo run -p ntc-cli -- schemac

# Python side
cd training && uv sync && uv run pytest

# Browser demo
examples/browser/build.sh && python3 -m http.server  # then open /examples/browser/
```

## Status (V1 milestones)

- [x] Contract pack: Typed Action IR v1, canonical Tool ABI v1, head codec v1,
      `.ntc` format v1, compat matrix
- [x] Rust runtime end-to-end on the CPU reference backend (tokenize → pack →
      forward → decode → policy → normalize → validate → serialize; spec §4
      example reproduced byte-exactly)
- [x] `.ntc` reader/writer + corruption tests + fixture generator
- [x] WebGPU backend: portable WGSL kernels, per-kernel + full-backend parity
      vs the CPU reference (element tolerances and 100% decision parity on
      the tiny fixture, verified on Metal)
- [x] wasm + browser demo (CPU backend in-browser; `examples/browser/build.sh`)
- [x] Python: pydantic contract mirrors, `.ntc` exporter (Rust-verifier
      conformant), PyTorch model with head-codec-exact outputs, data-engine +
      eval-harness skeletons (85 pytest tests)
- [x] Cross-language smoke: PyTorch-exported `.ntc` loads and runs through the
      Rust runtime (`ntc infer --model fixtures/models/tiny-v1-py/tiny.ntc …`)
- [ ] Python↔Rust logit parity on identical packed inputs (needs the Python
      input packer; next milestone toward IC-1)
- [ ] Tokenizer freeze (pruned-vocab mE5), Stage 1/2 training, IC-2…IC-3
- [ ] WebGPU-in-browser (wasm + wgpu), perf hardening (A8)

The V1 scope is spec §73: EN/DE/FR/ES, single-tool calls, `CALL/ASK/NO_CALL`,
≤16 candidate tools, BF16 training / F16 browser weights. Ternary is Phase 2;
the format reserves its dtype codes.
