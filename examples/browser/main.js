// NTC browser demo. Serve the REPOSITORY ROOT (not this directory) so the
// fixture model is reachable, e.g. from the repo root:
//
//     python3 -m http.server 8000
//
// then open http://localhost:8000/examples/browser/
//
// Requires ./pkg/ to exist — run ./build.sh first.

import init, { NtcWeb, version } from "./pkg/ntc_wasm.js";

const MODEL_URL = "../../fixtures/models/tiny-v1/tiny.ntc";

const $ = (id) => document.getElementById(id);
const output = $("output");
const status = $("status");
const button = $("compile");

let modelBytes = null;

function showError(e) {
  // wasm-bindgen errors arrive as plain strings; JS errors have .message.
  output.textContent = `Error: ${e instanceof Error ? e.message : e}`;
}

async function boot() {
  await init();
  $("version").textContent = `ntc-wasm v${version()}`;

  const resp = await fetch(MODEL_URL);
  if (!resp.ok) {
    throw new Error(
      `fetching ${MODEL_URL} failed (HTTP ${resp.status}). ` +
        "Serve the repository root, not examples/browser/ — see README.md."
    );
  }
  modelBytes = new Uint8Array(await resp.arrayBuffer());

  button.disabled = false;
  status.textContent = `Model loaded (${(modelBytes.length / 1024).toFixed(0)} KiB). Ready.`;
}

function compile() {
  output.textContent = "";
  let ntc = null;
  try {
    const timezone = $("timezone").value.trim() || "UTC";

    // A fresh instance per compile keeps edits to the tool list effective
    // (the registry is append-only). The tiny model loads in milliseconds.
    ntc = new NtcWeb(modelBytes, JSON.stringify({ timezone }));

    const tools = JSON.parse($("tools").value);
    if (!Array.isArray(tools)) throw new Error("tools must be a JSON array");
    for (const tool of tools) ntc.register_tool(JSON.stringify(tool));

    const started = performance.now();
    const result = ntc.compile(
      JSON.stringify({ utterance: $("utterance").value, timezone })
    );
    const ms = performance.now() - started;

    output.textContent = JSON.stringify(JSON.parse(result), null, 2);
    status.textContent = `Compiled in ${ms.toFixed(1)} ms.`;
  } catch (e) {
    showError(e);
  } finally {
    if (ntc) ntc.free();
  }
}

button.addEventListener("click", compile);
$("utterance").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !button.disabled) compile();
});

boot().catch((e) => {
  status.textContent = "Failed to start.";
  showError(e);
});
