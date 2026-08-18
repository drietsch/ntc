#!/usr/bin/env bash
# Build the NTC wasm bundle into examples/browser/pkg/.
#
# Uses cargo + the wasm-bindgen CLI directly (wasm-pack is not required).
# The CLI version must match the `wasm-bindgen` crate version in Cargo.lock;
# if it does not, the matching CLI is installed via `cargo install`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$ROOT/examples/browser/pkg"
WASM="$ROOT/target/wasm32-unknown-unknown/release/ntc_wasm.wasm"

# --- wasm-bindgen CLI / Cargo.lock version check -----------------------------
LOCK_VERSION="$(awk '
  /^name = "wasm-bindgen"$/ { found = 1; next }
  found && /^version = / { gsub(/"/, "", $3); print $3; exit }
' "$ROOT/Cargo.lock")"

if [[ -z "$LOCK_VERSION" ]]; then
  echo "error: could not find wasm-bindgen in Cargo.lock" >&2
  exit 1
fi

CLI_VERSION="$(wasm-bindgen --version 2>/dev/null | awk '{print $2}' || true)"

if [[ "$CLI_VERSION" != "$LOCK_VERSION" ]]; then
  echo "wasm-bindgen CLI ($CLI_VERSION) != Cargo.lock ($LOCK_VERSION); installing matching CLI..."
  cargo install wasm-bindgen-cli --version "$LOCK_VERSION" --locked
fi

# --- build -------------------------------------------------------------------
echo "==> cargo build (wasm32-unknown-unknown, release)"
cargo build --release --target wasm32-unknown-unknown -p ntc-wasm \
  --manifest-path "$ROOT/Cargo.toml"

echo "==> wasm-bindgen -> $OUT_DIR"
wasm-bindgen "$WASM" --target web --out-dir "$OUT_DIR"

echo "==> done"
ls -lh "$OUT_DIR"
