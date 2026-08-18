# NTC browser demo

Compiles natural-language utterances into tool calls entirely in the browser:
the tiny fixture model (`fixtures/models/tiny-v1/tiny.ntc`) runs on the CPU
reference backend compiled to WebAssembly. No frameworks, no CDNs, no server
component.

## Prerequisites

- Rust with the `wasm32-unknown-unknown` target
  (`rustup target add wasm32-unknown-unknown`)
- The `wasm-bindgen` CLI. `build.sh` checks that its version matches the
  `wasm-bindgen` crate version in `Cargo.lock` and installs the matching
  version via `cargo install wasm-bindgen-cli --version <x> --locked` if not.

## Build

```sh
./examples/browser/build.sh
```

This builds `ntc-wasm` for `wasm32-unknown-unknown` (release) and generates
the JS glue into `examples/browser/pkg/` (`ntc_wasm.js`,
`ntc_wasm_bg.wasm`, TypeScript definitions).

## Serve

The page fetches the model at `../../fixtures/models/tiny-v1/tiny.ntc`, so the
web server must serve the **repository root** (not `examples/browser/`):

```sh
# from the repository root
python3 -m http.server 8000
```

Then open <http://localhost:8000/examples/browser/>.

## Use

- **Utterance** — the natural-language command. The fixture model's test
  tokenizer only knows a small vocabulary (words like *make, a, dentist,
  appointment, tomorrow, afternoon, create, calendar, event, send, an, email,
  title, start, priority, low, normal, high, recipient, subject*); anything
  else maps to `<unk>`.
- **Tools** — a JSON array of NTC tool schemas registered before compiling.
  Edits take effect on the next compile.
- **Timezone** — IANA timezone used to resolve relative dates ("tomorrow
  afternoon").

Press **Compile**. The output pane shows the `CompileOutcome` JSON: an
`"outcome"` of `CALL` (with the executable `call`), `ASK` (with unresolved
required fields), or `NO_CALL`, plus the decoded intermediate representation
(`ir`).

Note: the tiny fixture model is a smoke-test artifact, not a trained model —
outcomes are structurally valid but not semantically meaningful.
