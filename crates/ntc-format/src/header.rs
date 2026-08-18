//! `.ntc` header and metadata section.

use serde::{Deserialize, Serialize};

pub const MAGIC: &[u8; 4] = b"NTC1";
pub const FORMAT_VERSION: u32 = 1;

/// Fixed-size portion of the header (after magic + version): section lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionLengths {
    pub metadata_len: u64,
    pub tokenizer_len: u64,
    pub directory_len: u64,
    pub data_len: u64,
}

/// The metadata JSON section. `model` is the architecture config consumed by
/// `ntc-model` (dims, layer counts, head layout); it stays schemaless here so
/// the format crate does not depend on model internals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NtcMetadata {
    /// Architecture family id, e.g. `ntc_encoder_heads_v1`.
    pub architecture: String,
    /// Training-side model version tag, e.g. `tiny-v1` or `ntc-web-250m-r3`.
    pub model_version: String,
    /// Contract versions this model was built against.
    pub ir_version: u32,
    pub abi_version: u32,
    pub head_spec_version: u32,
    /// Hex sha256 of the embedded tokenizer.json bytes.
    pub tokenizer_sha256: String,
    /// Dominant stored weight precision: `f32` | `f16` | `bf16`.
    pub quantization: String,
    /// Architecture configuration (parsed by `ntc-model`).
    pub model: serde_json::Value,
    /// Semantic type table (spec §34) — the closed vocabularies the head
    /// codec indexes into, in index order.
    #[serde(default)]
    pub semantic_types: Vec<String>,
}
