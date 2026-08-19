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

// Selectable models; missing files are disabled at boot.
const MODELS = {
  studio: "../../models/ntc-studio-v1/model.ntc",
  any: "../../models/ntc-any-v1/model.ntc",
  mini: "../../models/ntc-mini-v1/model.ntc",
  fixture: "../../fixtures/models/tiny-v1/tiny.ntc",
};
const modelCache = new Map();

const $ = (id) => document.getElementById(id);
const output = $("output");
const status = $("status");
const button = $("compile");

let modelBytes = null;

async function loadModel(key) {
  const url = MODELS[key];
  const t0 = performance.now();
  if (!modelCache.has(key)) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`fetching ${url} failed (HTTP ${resp.status})`);
    modelCache.set(key, new Uint8Array(await resp.arrayBuffer()));
  }
  modelBytes = modelCache.get(key);
  const ms = performance.now() - t0;
  status.textContent =
    `Model ${url.split("/").pop().replace("model.ntc", key + ".ntc")} loaded ` +
    `(${(modelBytes.length / 1048576).toFixed(1)} MiB in ${ms.toFixed(0)} ms). Ready.`;
}

function showError(e) {
  // wasm-bindgen errors arrive as plain strings; JS errors have .message.
  output.textContent = `Error: ${e instanceof Error ? e.message : e}`;
}

async function boot() {
  await init();
  $("version").textContent = `ntc-wasm v${version()}`;

  gpuAvailable = !!navigator.gpu;
  const be = $("backend");
  if (be && !gpuAvailable) {
    be.querySelector('option[value="gpu"]').disabled = true;
    if (be.value === "gpu" || be.value === "auto") be.value = "cpu";
  }

  // Disable models that are not present (they are gitignored and rebuilt
  // locally), and default to the first one that is.
  const sel = $("model");
  let defaultKey = null;
  for (const [key, url] of Object.entries(MODELS)) {
    const head = await fetch(url, { method: "HEAD" });
    const opt = sel && sel.querySelector(`option[value="${key}"]`);
    if (opt) opt.disabled = !head.ok;
    if (head.ok && defaultKey === null) defaultKey = key;
  }
  if (defaultKey === null) {
    throw new Error(
      "no model reachable. Serve the repository root, not examples/browser/ — see README.md."
    );
  }
  if (sel) sel.value = defaultKey;
  await loadModel(defaultKey);
  button.disabled = false;
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

    let context;
    try {
      const raw = $("context").value.trim();
      context = raw ? JSON.parse(raw) : undefined;
    } catch (e) {
      throw new Error(`context is not valid JSON: ${e.message}`);
    }

    const started = performance.now();
    const result = await ntc.compile_async(
      JSON.stringify({ utterance: $("utterance").value, timezone, context })
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

// Studio scenarios — utterance AND selection, because the whole point is
// that "tag this" depends on what "this" is.
const SCENARIOS = [
  {
    label: "linked asset",
    utterance: "tag this with Freigegeben",
    context: { linked: [{ ref: "L1", type: "asset", id: 4711, key: "winterjacke-hero.jpg", path: "/Produktfotos/", isFolder: false }], locale: "de" },
  },
  {
    label: "bulk over limit",
    utterance: "apply the publish transition to all of these",
    context: {
      linked: [
        { ref: "L1", type: "asset", id: 4711, key: "hero.jpg", path: "/Fotos/", isFolder: false },
        { ref: "L2", type: "document", id: 3108, key: "spring-sale", path: "/en/", isFolder: false },
        { ref: "L3", type: "object", id: 36192, key: "loyalty-flyer", path: "/Rendition/", isFolder: false },
      ],
      selectionCount: 37,
      locale: "en",
    },
  },
  {
    label: "payload write",
    utterance: "update document 355 and set its headline to Autumn Sale",
    context: { linked: [], resolver: [], locale: "en" },
  },
  {
    label: "conceptual",
    utterance: "was macht das search_assets tool eigentlich?",
    context: { linked: [], locale: "de" },
  },
  {
    label: "no selection",
    utterance: "montre-moi les groupes cibles",
    context: { linked: [], locale: "fr" },
  },
];
let scenarioIdx = -1;

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

const scenarioBtn = $("scenario");
if (scenarioBtn) {
  scenarioBtn.addEventListener("click", () => {
    scenarioIdx = (scenarioIdx + 1) % SCENARIOS.length;
    const sc = SCENARIOS[scenarioIdx];
    $("utterance").value = sc.utterance;
    $("context").value = JSON.stringify(sc.context, null, 1);
    scenarioBtn.textContent = `Studio scenario: ${sc.label}`;
    compile();
  });
}

const modelSel = $("model");
if (modelSel) {
  modelSel.addEventListener("change", async () => {
    button.disabled = true;
    try {
      await loadModel(modelSel.value);
    } catch (e) {
      showError(e);
    }
    button.disabled = false;
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
