//! NTC-Web deterministic runtime (spec §42–§46): decodes head outputs into
//! the Typed Action IR, applies confidence policy, normalizes semantics
//! (dates, durations, locale), validates against the Tool ABI, and serializes
//! the executable JSON call. Generic over the inference [`Backend`].

pub mod compiler;
pub mod decode;
pub mod normalize;
pub mod policy;

pub use compiler::{CompileOutcome, CompiledCall, CompilerConfig, NeuralToolCompiler};
pub use ntc_model::Backend;
pub use policy::ConfidencePolicy;

use ntc_core::NtcError;

/// Host-side execution hook (spec §23: actual API invocation stays in the
/// host application; the compiler only produces validated calls).
pub trait ExecutionHook {
    fn execute(&mut self, call: &CompiledCall) -> Result<serde_json::Value, NtcError>;
}
