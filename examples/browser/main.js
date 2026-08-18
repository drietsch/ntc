// NTC browser demo. Serve the REPOSITORY ROOT (not this directory) so the
// fixture model is reachable, e.g. from the repo root:
//
//     python3 -m http.server 8000
//
// then open http://localhost:8000/examples/browser/
//
// Requires ./pkg/ to exist — run ./build.sh first.

import init, { NtcWeb, version } from "./pkg/ntc_wasm.js";

let gpuAvailable = false;

// Prefer the trained mini model; fall back to the random-weight fixture.
const MODEL_URLS = [
  "../../models/ntc-mini-v1/model.ntc",
  "../../fixtures/models/tiny-v1/tiny.ntc",
];

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

  const t0 = performance.now();
  let loaded = null;
  for (const url of MODEL_URLS) {
    const resp = await fetch(url);
    if (resp.ok) {
      modelBytes = new Uint8Array(await resp.arrayBuffer());
      loaded = url;
      break;
    }
  }
  if (!loaded) {
    throw new Error(
      "no model reachable. Serve the repository root, not examples/browser/ — see README.md."
    );
  }
  const fetchMs = performance.now() - t0;

  gpuAvailable = !!navigator.gpu;
  const be = $("backend");
  if (be && !gpuAvailable) {
    be.querySelector('option[value="gpu"]').disabled = true;
    if (be.value === "gpu" || be.value === "auto") be.value = "cpu";
  }

  button.disabled = false;
  status.textContent =
    `Model ${loaded.split("/").pop()} loaded (${(modelBytes.length / 1024).toFixed(0)} KiB ` +
    `in ${fetchMs.toFixed(0)} ms). Ready.`;
}

async function compile() {
  output.textContent = "";
  let ntc = null;
  try {
    const timezone = $("timezone").value.trim() || "UTC";
    const sel = $("backend") ? $("backend").value : "auto";
    const useGpu = sel === "gpu" || (sel === "auto" && gpuAvailable);

    // A fresh instance per compile keeps edits to the tool list effective
    // (the registry is append-only). Init includes weight decode (+ GPU
    // upload on the WebGPU path) — an honest cold-start number.
    const tInit = performance.now();
    ntc = useGpu
      ? await NtcWeb.new_gpu(modelBytes, JSON.stringify({ timezone }))
      : new NtcWeb(modelBytes, JSON.stringify({ timezone }));
    const initMs = performance.now() - tInit;

    const tools = JSON.parse($("tools").value);
    if (!Array.isArray(tools)) throw new Error("tools must be a JSON array");
    for (const tool of tools) ntc.register_tool(JSON.stringify(tool));

    const started = performance.now();
    const result = await ntc.compile_async(
      JSON.stringify({ utterance: $("utterance").value, timezone })
    );
    const ms = performance.now() - started;

    output.textContent = JSON.stringify(JSON.parse(result), null, 2);
    status.textContent =
      `${ntc.backend().toUpperCase()} · init ${initMs.toFixed(0)} ms · ` +
      `compile ${ms.toFixed(1)} ms — fully local, no server.`;
  } catch (e) {
    showError(e);
  } finally {
    if (ntc) ntc.free();
  }
}

// In-distribution example utterances (the mini tokenizer is case-sensitive
// and trained on lowercase corpus text).
const EXAMPLES = {
  en: [
    "schedule a dentist appointment tomorrow afternoon for one and a half hours",
    "turn off the light in the living room",
    "set a timer for 90 minutes",
    "what does the timer tool do?",
  ],
  de: [
    "plane einen zahnarzttermin morgen nachmittag für eine stunde",
    "Drehe das Licht im Wohnzimmer ab!",
    "Mach das Licht in der Küche an.",
    "stell einen timer auf eineinhalb stunden",
    "was macht das timer-tool?",
  ],
  fr: [
    "planifie un rendez-vous chez le dentiste demain après-midi",
    "envoie un e-mail à anna müller au sujet du budget",
    "mets un minuteur de dix minutes",
  ],
  es: [
    "programa una cita con el dentista mañana por la tarde",
    "apaga la luz del salón",
    "pon un temporizador de media hora",
  ],
};
const exampleIdx = {};
for (const lang of Object.keys(EXAMPLES)) {
  const el = document.getElementById(`ex-${lang}`);
  if (!el) continue;
  el.addEventListener("click", () => {
    const list = EXAMPLES[lang];
    exampleIdx[lang] = ((exampleIdx[lang] ?? -1) + 1) % list.length;
    $("utterance").value = list[exampleIdx[lang]];
    compile();
  });
}

button.addEventListener("click", compile);
$("utterance").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !button.disabled) compile();
});

boot().catch((e) => {
  status.textContent = "Failed to start.";
  showError(e);
});
