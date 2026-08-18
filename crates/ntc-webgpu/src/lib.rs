//! WebGPU inference backend for NTC-Web (V1: portable f32 kernels).
//!
//! Implements [`ntc_model::Backend`] with the transformer stack (utterance
//! encoder, schema encoder, fusion) on the GPU via WGSL compute kernels, and
//! embedding lookups + head projections on the CPU (accepted V1 split).
//! Parity-tested against `CpuRefBackend`, the normative reference.

pub mod backend;
pub mod device;
pub mod gpu;

pub use backend::WgpuBackend;
pub use device::{GpuCaps, WgpuContext};
pub use gpu::GpuExecutor;

/// WGSL kernel sources (compiled with `include_str!`), exposed so CI can
/// naga-validate them without a GPU.
pub mod kernels {
    pub const MATMUL_BIAS: &str = include_str!("kernels/matmul_bias.wgsl");
    pub const LAYERNORM: &str = include_str!("kernels/layernorm.wgsl");
    pub const ELEMENTWISE: &str = include_str!("kernels/elementwise.wgsl");
    pub const ATTN_SCORES: &str = include_str!("kernels/attn_scores.wgsl");
    pub const SOFTMAX: &str = include_str!("kernels/softmax.wgsl");
    pub const ATTN_CTX: &str = include_str!("kernels/attn_ctx.wgsl");

    /// `(file name, source)` for every kernel.
    pub const ALL: &[(&str, &str)] = &[
        ("matmul_bias.wgsl", MATMUL_BIAS),
        ("layernorm.wgsl", LAYERNORM),
        ("elementwise.wgsl", ELEMENTWISE),
        ("attn_scores.wgsl", ATTN_SCORES),
        ("softmax.wgsl", SOFTMAX),
        ("attn_ctx.wgsl", ATTN_CTX),
    ];
}

use ntc_core::NtcError;

/// Load a `.ntc` model onto the WebGPU backend and wrap it in the runtime
/// compiler. Async: adapter/device acquisition awaits the platform (browser
/// WebGPU or native).
pub async fn load_gpu(
    model_bytes: &[u8],
    config: ntc_runtime::CompilerConfig,
) -> Result<ntc_runtime::NeuralToolCompiler<WgpuBackend>, NtcError> {
    let file =
        ntc_format::NtcFile::parse(model_bytes).map_err(|e| NtcError::Format(e.to_string()))?;
    if file.metadata.architecture != ntc_model::ARCHITECTURE {
        return Err(NtcError::Format(format!(
            "unsupported architecture `{}`",
            file.metadata.architecture
        )));
    }
    let arch = ntc_model::NtcArchConfig::from_metadata(&file.metadata.model)?;
    let weights = ntc_model::ModelWeights::from_ntc(&file, &arch)?;
    let tokenizer = ntc_core::tokenizer::NtcTokenizer::from_bytes(file.tokenizer_bytes)?;
    let ctx = WgpuContext::new().await?;
    let backend = WgpuBackend::new(arch.clone(), &weights, ctx)?;
    Ok(ntc_runtime::NeuralToolCompiler::from_parts(
        arch, tokenizer, backend, config,
    ))
}
