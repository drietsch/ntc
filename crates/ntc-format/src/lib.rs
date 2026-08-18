//! The `.ntc` model container, format version 1 (spec §33–§35).
//!
//! This crate is the **normative reader**; the production writer is the
//! Python exporter (`training/export/ntc_writer.py`). A Rust writer exists
//! behind the `write` feature for fixtures and round-trip conformance tests.
//! The binary layout is documented in `docs/model-format.md`; the two
//! implementations are pinned to each other by round-trip CI and the corrupt
//! fixtures under `fixtures/ntc-format/`.
//!
//! ## Layout v1 (all integers little-endian)
//!
//! ```text
//! offset  size  field
//! 0       4     magic "NTC1"
//! 4       4     format_version u32 (= 1)
//! 8       8     metadata_len   u64   (JSON, see [`NtcMetadata`])
//! 16      8     tokenizer_len  u64   (raw tokenizer.json bytes)
//! 24      8     directory_len  u64   (JSON array of [`TensorRecord`])
//! 32      8     data_len       u64   (tensor data section)
//! 40      ...   metadata | tokenizer | directory
//! ...     ...   zero padding to a 256-byte boundary (from file start)
//! ...     ...   data section — each tensor blob 256-byte aligned
//! end-32  32    sha256 of everything before the footer
//! ```
//!
//! Header sections come first so a browser can begin per-layer GPU upload
//! while the body streams. Tensor `offset` values are relative to the data
//! section start and are all multiples of 256 (WebGPU's worst-case
//! `min_storage_buffer_offset_alignment`), so tensor slices can be bound or
//! uploaded without repacking.

pub mod header;
pub mod loader;
pub mod tensor;
pub mod verify;
#[cfg(feature = "write")]
pub mod writer;

pub use header::{NtcMetadata, FORMAT_VERSION, MAGIC};
pub use loader::NtcFile;
pub use tensor::{DType, TensorRecord, TensorView};
pub use verify::{verify, VerifyReport};

use thiserror::Error;

/// Alignment of the data section and every tensor blob within it.
pub const TENSOR_ALIGN: u64 = 256;

#[derive(Debug, Error, PartialEq)]
pub enum FormatError {
    #[error("bad magic: expected `NTC1`")]
    BadMagic,
    #[error("unsupported format version {0} (reader supports {FORMAT_VERSION})")]
    UnsupportedVersion(u32),
    #[error("file truncated: {0}")]
    Truncated(String),
    #[error("invalid metadata JSON: {0}")]
    BadMetadata(String),
    #[error("invalid tensor directory JSON: {0}")]
    BadDirectory(String),
    #[error("tensor `{name}`: {message}")]
    BadTensor { name: String, message: String },
    #[error("misaligned {what}: offset {offset} is not a multiple of {TENSOR_ALIGN}")]
    Misaligned { what: String, offset: u64 },
    #[error("checksum mismatch for {0}")]
    ChecksumMismatch(String),
    #[error("unsupported dtype `{dtype}` for tensor `{name}` (reserved for a later phase)")]
    ReservedDtype { name: String, dtype: String },
    #[error("duplicate tensor name `{0}`")]
    DuplicateTensor(String),
}
