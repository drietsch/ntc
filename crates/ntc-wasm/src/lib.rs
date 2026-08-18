//! WebAssembly bindings for the Neural Tool Compiler.
//!
//! V1 runs the CPU reference backend inside the browser (WebGPU-in-wasm is a
//! later milestone). The API is JSON-in/JSON-out so the JS side never needs
//! generated types: schemas are [`RawToolSchema`] JSON, requests are
//! [`CompileRequest`] JSON, and the result is a [`CompileOutcome`] JSON string
//! tagged with an `"outcome"` field.
//!
//! The crate also compiles on native targets (`rlib`), where `#[wasm_bindgen]`
//! is inert — this keeps the wrapper testable with plain `cargo test`.

use wasm_bindgen::prelude::*;

use ntc_core::ir::CompileRequest;
use ntc_core::schema::RawToolSchema;
use ntc_model::CpuRefBackend;
use ntc_runtime::{CompilerConfig, NeuralToolCompiler};

/// The crate version (mirrors the workspace version).
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Map any displayable error onto a `JsValue` (its Display string).
fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Browser-facing handle around [`NeuralToolCompiler`] on the CPU reference
/// backend.
#[wasm_bindgen]
pub struct NtcWeb {
    compiler: NeuralToolCompiler<CpuRefBackend>,
}

#[wasm_bindgen]
impl NtcWeb {
    /// Load a `.ntc` model from bytes (a JS `Uint8Array`).
    ///
    /// `config_json` is an optional JSON object overriding compiler defaults:
    /// `{"timezone": "Europe/Berlin", "locale": "de-DE"}` — both keys
    /// optional; unknown keys are rejected to catch typos early.
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8], config_json: Option<String>) -> Result<NtcWeb, JsValue> {
        console_error_panic_hook::set_once();

        let mut config = CompilerConfig::default();
        if let Some(json) = config_json.as_deref() {
            let value: serde_json::Value = serde_json::from_str(json)
                .map_err(|e| js_err(format!("invalid config JSON: {e}")))?;
            let obj = value
                .as_object()
                .ok_or_else(|| js_err("config JSON must be an object"))?;
            for (key, v) in obj {
                let as_str = |v: &serde_json::Value| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| js_err(format!("config `{key}` must be a string")))
                };
                match key.as_str() {
                    "timezone" => config.timezone = as_str(v)?,
                    "locale" => config.locale = as_str(v)?,
                    other => {
                        return Err(js_err(format!(
                            "unknown config key `{other}` (expected `timezone` or `locale`)"
                        )))
                    }
                }
            }
        }

        let compiler = NeuralToolCompiler::load_cpu(model_bytes, config).map_err(js_err)?;
        Ok(NtcWeb { compiler })
    }

    /// Register a tool schema (a [`RawToolSchema`] JSON object). Returns the
    /// tool's registry index.
    pub fn register_tool(&mut self, schema_json: String) -> Result<u32, JsValue> {
        let schema: RawToolSchema = serde_json::from_str(&schema_json)
            .map_err(|e| js_err(format!("invalid tool schema JSON: {e}")))?;
        let id = self.compiler.register_tool(schema).map_err(js_err)?;
        Ok(id.0)
    }

    /// Compile an utterance. `request_json` is a [`CompileRequest`] JSON
    /// object; the result is the [`ntc_runtime::CompileOutcome`] serialized as
    /// JSON (`{"outcome": "CALL" | "ASK" | "NO_CALL", ...}`).
    pub fn compile(&mut self, request_json: String) -> Result<String, JsValue> {
        let request: CompileRequest = serde_json::from_str(&request_json)
            .map_err(|e| js_err(format!("invalid compile request JSON: {e}")))?;
        let outcome = self.compiler.compile(&request).map_err(js_err)?;
        serde_json::to_string(&outcome).map_err(js_err)
    }
}
