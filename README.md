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

It is the **fast local tier** of an agent stack: it compiles what a single
typed call can express and answers `DELEGATE` for work that needs a full LLM
agent (multi-step chains over results, bulk mutations, open-ended reasoning).
See [docs/delegation.md](docs/delegation.md).

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
| `docs/` | Normative docs: model format, Tool ABI, Action IR, parity testing, [delegation boundary](docs/delegation.md) |

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
      Rust runtime
- [x] **IC-1**: Python↔Rust logit + decision parity on identical packed inputs
      (Python packer mirrors `inputs.rs`; `training/tests/test_parity_ic1.py`)
- [x] Tokenizer **frozen** (mini-scale Unigram, EN/DE/FR/ES) with golden-vector
      parity native-Rust-side (`contracts/tokenizer/`, `fixtures/tokenizer/`)
- [x] Synthetic data engine: deterministic mini generator (1.9k validated
      examples, 4 languages, decoys/hard-negatives/name-randomization) **and**
      a live `claude -p` teacher batch through the production orchestrator
- [x] **Trained mini model** (1.4M params, Stage-2 composite loss, calibrated,
      exported to `models/ntc-mini-v1/model.ntc`): dev exact-match 95.5%,
      seen-test 93.8%, tool selection 98.8% seen / 86% masked-names
- [x] **IC-2.5**: PyTorch↔Rust-runtime decision agreement 100% on dev
      (`models/ntc-mini-v1/eval/report.json`)
- [x] **IC-3**: the trained model compiles EN/DE/FR/ES utterances to validated
      JSON **in Chrome on WebGPU** — 33–44 ms/compile steady-state, 4.5× the
      wasm-CPU path, identical outcomes (`docs/benchmarking.md`)
- [x] WebGPU-in-browser: async wgpu path (`NtcWeb.new_gpu` + `compile_async`)

- [x] **Any-word model** (`models/ntc-any-v1`, 44M params, ~89 MB — gitignored,
      rebuild recipe in .gitignore): pretrained multilingual MiniLM backbone
      with vocab pruned 250k→33.9k (0% token inflation, 0 unks), fine-tuned on
      templates + a live `claude -p` diversity batch. Understands arbitrary
      wording: "Hey, could you kill the lights in the bathroom?" → correct
      `light.set` call. Selectable in the browser demo next to the 2.8 MB mini
      model.

Mini-scale caveats (documented limits, not defects): the 1.4M model does not
generalize to unseen tool families (0% on that split — the capability the
full-scale 250M pretrained backbone exists to provide), and the 642-piece
tokenizer is case-sensitive to its training corpus. Full-scale training
(pruned-mE5 backbone, teacher-generated 500k+ corpus, A100) reuses exactly
this machinery.

The V1 scope is spec §73: EN/DE/FR/ES, single-tool calls, `CALL/ASK/NO_CALL`,
≤16 candidate tools, BF16 training / F16 browser weights. Ternary is Phase 2;
the format reserves its dtype codes.
