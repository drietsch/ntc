# NTC-Web training workstream

Python side of NTC-Web: contract models, the production `.ntc` exporter, the
PyTorch reference implementation of `ntc_encoder_heads_v1`, the synthetic data
engine skeleton, and the eval harness (in `../eval`).

The Rust crates are the source of truth for all contracts. Everything here is
pinned to them: the pydantic models validate against `contracts/*.schema.json`,
the exporter's output must pass `ntc verify`, and the PyTorch forward pass
mirrors the normative CPU reference (`crates/ntc-model/src/cpu.rs`).

## Setup

Requires [uv](https://docs.astral.sh/uv/). The project is pinned to Python 3.12
(torch has no 3.14 wheels yet):

```sh
cd training
uv sync          # installs Python 3.12 + torch, pydantic, etc. into .venv
```

## Running tests

```sh
cd training
uv run pytest    # collects training/tests AND ../eval/tests (see pyproject)
```

Test collection is configured in `pyproject.toml` (`[tool.pytest.ini_options]`
`testpaths = ["tests", "../eval/tests"]`), so one invocation covers both
workstreams. The `.ntc` conformance test shells out to `target/debug/ntc` and
is skipped automatically if the Rust CLI is not built
(`cargo build -p ntc-cli`).

Lint (both trees):

```sh
uv run ruff check . ../eval
```

## Layout

| Path | What |
| --- | --- |
| `ntc_contracts/` | Pydantic v2 mirrors of the contracts: `ir.py` (ActionIr, adjacently-tagged SemanticValue, CompileRequest), `tool_abi.py` (CanonicalTool/CanonicalArg). |
| `export/ntc_writer.py` | The **production `.ntc` v1 writer** (pure Python: struct/hashlib/xxhash/json), plus a minimal reader (`read_ntc`) for tokenizer extraction and self-checks. |
| `export/export_tiny.py` | Exports a seeded random-init tiny model to `fixtures/models/tiny-v1-py/tiny.ntc`, reusing the test tokenizer embedded in the Rust fixture. Run: `uv run python -m export.export_tiny`. |
| `ntc_model/config.py` | `NtcArchConfig` (mirrors the Rust config incl. `.ntc` metadata `model` serialization) + `tiny_config()`. |
| `ntc_model/model.py` | PyTorch `ntc_encoder_heads_v1`: post-LN encoders, schema encoder, fusion (self+cross attention, NO_TOOL slot), all heads with the exact head-spec output names; `export_tensors()` produces the canonical Rust tensor names with `[in, out]`-transposed linear weights; `tensor_specs()` mirrors `ntc_model::weights::tensor_specs`. |
| `datasets/schema.py` | Dataset example schema (`ToolFamily`, `DatasetExample` with char-offset gold spans) + invariant validators (span/surface check, ASK ⇔ unresolved, CALL ⇒ tool). |
| `synthetic/orchestrator.py` | Headless-Claude teacher driver skeleton: stage prompt builders (tool-gen / request-gen / verify-vote), `run_claude_batch` (async subprocess, bounded concurrency), validate → one repair retry → JSONL shards + `{prompt_sha256, created_at, count}` manifest. Tests use canned outputs; `claude` is never invoked in CI. |
| `../eval/metrics.py` | Metrics over (CompileOutcome, gold) pairs: tool selection, required-arg accuracy, hallucinated-arg rate, NO_CALL precision/recall/F1, ASK accuracy, error taxonomy E01–E05/E11/E12. Pure stdlib. |
| `../eval/harness.py` | JSONL pairing by id + aggregation; `python -m eval.harness --pred p.jsonl --gold g.jsonl`. |

## Conformance loop

```sh
uv run python -m export.export_tiny            # writes fixtures/models/tiny-v1-py/tiny.ntc
(cd .. && ./target/debug/ntc verify fixtures/models/tiny-v1-py/tiny.ntc --dump-manifest)
```

The pytest suite runs the same check (`tests/test_export_tiny.py`) and also
asserts the tensor name/shape set matches the Rust tiny manifest exactly
(112 tensors).

## What exists vs. what's next

Done:
- contract models + JSON Schema round-trip tests;
- `.ntc` writer + Rust-verified tiny export;
- PyTorch model matching the Rust forward semantics (shape/mask contract
  tested; numeric parity fixtures come with the parity harness);
- dataset schema with validators;
- teacher orchestration skeleton (parse/validate/repair path unit-tested);
- eval metrics + error taxonomy + harness.

Next (not yet built):
- **vocab pruning**: derive the product tokenizer subset + row-pruned
  embeddings from corpus statistics;
- **real backbone**: XLM-R-style initialization (position de-offset,
  token-type folding per `weights.rs` notes) instead of random init;
- **Stage 1/2 training**: schema-grounding pretraining + head fine-tuning,
  loss wiring against the head-spec class orderings;
- **full data engine**: drive the real `claude -p` teacher, verify-vote
  consensus, dedup/stratified splits, `ntc schemac` canonicalization in the
  loop, token-span projection from char spans;
- **parity fixtures**: dump PyTorch head outputs for fixed inputs and diff
  against the Rust CPU reference within `fixtures/tolerances.toml`.

## Objective note: presence-head class imbalance

The presence head labels **every declared argument of every candidate tool**,
so its class distribution scales with the tool set's shape:

| corpus | tools/example | args/tool | NOT_APPLICABLE share |
|---|---|---|---|
| mini (calendar/email/timer/light) | 4 | 2–3 | ~75% |
| Pimcore MCP tools | 4 | up to 8 | **94.8%** |

At ~95% the unweighted objective collapses to the majority class: the model
still selects the right tool but marks every argument NOT_APPLICABLE, so it
emits argument-free CALLs (measured: required-arg accuracy 0.05). The fix is
balanced class weighting (`train.py:class_weights`, `total / (n_classes *
count_c)`, capped at 12 so a handful of AMBIGUOUS labels cannot dominate),
plus a `present_acc` metric in the training log — plain `presence_acc` hides
the collapse because it is ~94% when the model predicts NOT_APPLICABLE for
everything.

This matters at full scale too: with 16 candidate tools the share only grows,
so the weighting (or a focal-loss variant) should stay in the objective.
