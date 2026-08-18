//! WebAssembly bindings for the Neural Tool Compiler.
//!
//! Two backends: the CPU reference (synchronous, always available) and
//! WebGPU via wgpu (async — construct with [`NtcWeb::new_gpu`] and use
//! [`NtcWeb::compile_async`]). The API is JSON-in/JSON-out so the JS side never needs
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
use ntc_webgpu::WgpuBackend;

#[allow(clippy::large_enum_variant)] // one instance per page; boxing buys nothing
enum Inner {
    Cpu(NeuralToolCompiler<CpuRefBackend>),
    Gpu(NeuralToolCompiler<WgpuBackend>),
}

/// The crate version (mirrors the workspace version).
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Parse the optional config JSON (`{"timezone", "locale"}`).
fn parse_config(config_json: Option<&str>) -> Result<CompilerConfig, JsValue> {
    let mut config = CompilerConfig::default();
    if let Some(json) = config_json {
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|e| js_err(format!("invalid config JSON: {e}")))?;
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
    Ok(config)
}

/// Map any displayable error onto a `JsValue` (its Display string).
fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Browser-facing handle around [`NeuralToolCompiler`] on the CPU reference
/// backend.
#[wasm_bindgen]
pub struct NtcWeb {
    inner: Inner,
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

        let config = parse_config(config_json.as_deref())?;
        let compiler = NeuralToolCompiler::load_cpu(model_bytes, config).map_err(js_err)?;
        Ok(NtcWeb {
            inner: Inner::Cpu(compiler),
        })
    }

    /// Load a `.ntc` model onto the **WebGPU** backend (async: adapter and
    /// device acquisition go through the browser). Use [`Self::compile_async`]
    /// with instances created this way.
    pub async fn new_gpu(
        model_bytes: Vec<u8>,
        config_json: Option<String>,
    ) -> Result<NtcWeb, JsValue> {
        console_error_panic_hook::set_once();
        let config = parse_config(config_json.as_deref())?;
        let compiler = ntc_webgpu::load_gpu(&model_bytes, config)
            .await
            .map_err(js_err)?;
        Ok(NtcWeb {
            inner: Inner::Gpu(compiler),
        })
    }

    /// Which backend this instance runs on: `"cpu"` or `"gpu"`.
    pub fn backend(&self) -> String {
        match &self.inner {
            Inner::Cpu(_) => "cpu".into(),
            Inner::Gpu(_) => "gpu".into(),
        }
    }

    /// Register a tool schema (a [`RawToolSchema`] JSON object). Returns the
    /// tool's registry index.
    pub fn register_tool(&mut self, schema_json: String) -> Result<u32, JsValue> {
        let schema: RawToolSchema = serde_json::from_str(&schema_json)
            .map_err(|e| js_err(format!("invalid tool schema JSON: {e}")))?;
        let id = match &mut self.inner {
            Inner::Cpu(c) => c.register_tool(schema),
            Inner::Gpu(c) => c.register_tool(schema),
        }
        .map_err(js_err)?;
        Ok(id.0)
    }

    /// Compile an utterance. `request_json` is a [`CompileRequest`] JSON
    /// object; the result is the [`ntc_runtime::CompileOutcome`] serialized as
    /// JSON (`{"outcome": "CALL" | "ASK" | "NO_CALL", ...}`).
    pub fn compile(&mut self, request_json: String) -> Result<String, JsValue> {
        let request: CompileRequest = serde_json::from_str(&request_json)
            .map_err(|e| js_err(format!("invalid compile request JSON: {e}")))?;
        let outcome = match &mut self.inner {
            Inner::Cpu(c) => c.compile(&request).map_err(js_err)?,
            Inner::Gpu(_) => {
                return Err(js_err("this instance runs on WebGPU — use compile_async"))
            }
        };
        serde_json::to_string(&outcome).map_err(js_err)
    }

    /// Async compile — required for GPU instances, works for CPU too.
    pub async fn compile_async(&mut self, request_json: String) -> Result<String, JsValue> {
        let request: CompileRequest = serde_json::from_str(&request_json)
            .map_err(|e| js_err(format!("invalid compile request JSON: {e}")))?;
        let outcome = match &mut self.inner {
            Inner::Cpu(c) => c.compile(&request).map_err(js_err)?,
            Inner::Gpu(c) => c.compile_async(&request).await.map_err(js_err)?,
        };
        serde_json::to_string(&outcome).map_err(js_err)
    }
}
