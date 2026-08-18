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
