//! NTC-Web model layer: architecture configuration, host tensors, input
//! packing, the [`Backend`] execution contract, and the naive-but-normative
//! **CPU reference** forward pass every GPU backend is parity-tested against.
//!
//! V1 architecture family: `ntc_encoder_heads_v1` — multilingual utterance
//! encoder, per-tool schema encoder (block-diagonal), fusion (schema
//! self-attention → cross-attention to user → FFN), and structured heads per
//! `contracts/heads/v1/head-spec.json`.
//!
//! The runtime executes single requests (B = 1); batching exists only on the
//! PyTorch training side.

pub mod backend;
pub mod config;
pub mod cpu;
pub mod inputs;
pub mod ops;
pub mod tensor;
pub mod weights;

pub use backend::{Backend, HeadOutputs};
pub use config::NtcArchConfig;
pub use cpu::CpuRefBackend;
pub use inputs::{ModelInputs, SegmentKind};
pub use tensor::Tensor;
pub use weights::ModelWeights;

/// Architecture id this crate implements.
pub const ARCHITECTURE: &str = "ntc_encoder_heads_v1";

#[cfg(feature = "test-support")]
pub mod test_support;
