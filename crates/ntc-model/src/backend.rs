//! The execution contract every inference backend implements.

use std::collections::HashMap;

use ntc_core::NtcError;

use crate::inputs::ModelInputs;
use crate::tensor::Tensor;

/// Head outputs keyed by the head-codec output names
/// (`action.logits`, `tool.logits`, `presence.logits`, …).
///
/// Shape conventions (single request, B = 1):
/// - `action.logits`: `[3]`
/// - `tool.logits`: `[n_tools + 1]` (last = NO_TOOL)
/// - per-arg heads: `[n_tools, max_args, C]` — slots for args a tool does not
///   declare (and enum slots past a value list) are filled with `f32::MIN`
///   for logit tensors and `0.0` for `numeric.magnitude`.
/// - `span.*.logits`: `[n_tools, max_args, max_utterance_tokens]` with padded
///   utterance positions at `f32::MIN`.
///
/// Parity fixtures compare only the valid region (real tools/args/tokens).
#[derive(Debug, Clone)]
pub struct HeadOutputs {
    pub tensors: HashMap<String, Tensor>,
}

impl HeadOutputs {
    pub fn get(&self, name: &str) -> Result<&Tensor, NtcError> {
        self.tensors
            .get(name)
            .ok_or_else(|| NtcError::Inference(format!("missing head output `{name}`")))
    }
}

pub trait Backend {
    fn run(&mut self, inputs: &ModelInputs) -> Result<HeadOutputs, NtcError>;
}

/// Async execution contract for hosts without blocking GPU readback
/// (wasm/WebGPU: `map_async` resolves via the browser event loop, so the
/// whole inference must be awaited). Native backends implement it trivially
/// by wrapping [`Backend::run`].
pub trait AsyncBackend {
    fn run_async(
        &mut self,
        inputs: &ModelInputs,
    ) -> impl std::future::Future<Output = Result<HeadOutputs, NtcError>>;
}
