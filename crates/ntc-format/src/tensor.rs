//! Tensor directory records and views.

use serde::{Deserialize, Serialize};

/// Tensor element types. V1 readers accept F32/F16/BF16; the remaining codes
/// are **reserved** for later phases and rejected with a distinct error so a
/// Phase-2 file fails loudly rather than confusingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// Reserved (Phase 2): 2-bit ternary, 16 weights per u32.
    #[serde(rename = "TERNARY_T2")]
    TernaryT2,
    /// Reserved (Phase 3): base-3 packed ternary, 20 weights per u32.
    #[serde(rename = "TERNARY_PACK20")]
    TernaryPack20,
    /// Reserved (stretch): int8 weights + per-channel scales.
    I8,
}

impl DType {
    /// Bytes per element for the dense dtypes; `None` for packed formats.
    pub fn element_size(&self) -> Option<u64> {
        match self {
            DType::F32 => Some(4),
            DType::F16 | DType::BF16 => Some(2),
            DType::I8 => Some(1),
            DType::TernaryT2 | DType::TernaryPack20 => None,
        }
    }

    pub fn is_supported_v1(&self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }

    pub fn name(&self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::TernaryT2 => "TERNARY_T2",
            DType::TernaryPack20 => "TERNARY_PACK20",
            DType::I8 => "I8",
        }
    }
}

/// One entry in the tensor directory (JSON section of the file).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorRecord {
    /// Canonical tensor name, e.g. `encoder.layer.3.ffn.up.weight`.
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// Byte offset relative to the data-section start; multiple of 256.
    pub offset: u64,
    pub byte_length: u64,
    /// xxh64 of the tensor bytes, as a 16-hex-digit string.
    pub xxh64: String,
}

impl TensorRecord {
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// A zero-copy view of one tensor's bytes inside a parsed `.ntc` buffer.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub record: &'a TensorRecord,
    pub bytes: &'a [u8],
}
