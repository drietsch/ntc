//! Rust `.ntc` writer — dev/tooling only (`write` feature).
//!
//! The production writer is `training/export/ntc_writer.py`; this writer
//! exists for round-trip tests and for generating corrupt-case fixtures. Its
//! output must stay byte-compatible with the Python writer (pinned by the
//! conformance suite).

use sha2::{Digest, Sha256};
use xxhash_rust::xxh64::xxh64;

use crate::header::{NtcMetadata, FORMAT_VERSION, MAGIC};
use crate::tensor::{DType, TensorRecord};
use crate::{FormatError, TENSOR_ALIGN};

pub struct NtcWriter {
    metadata: NtcMetadata,
    tokenizer_bytes: Vec<u8>,
    records: Vec<TensorRecord>,
    data: Vec<u8>,
}

impl NtcWriter {
    pub fn new(mut metadata: NtcMetadata, tokenizer_bytes: Vec<u8>) -> Self {
        metadata.tokenizer_sha256 = super::loader::hex(&Sha256::digest(&tokenizer_bytes));
        Self {
            metadata,
            tokenizer_bytes,
            records: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Append a tensor; blobs are laid out in call order, 256-byte aligned.
    pub fn add_tensor(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[u64],
        bytes: &[u8],
    ) -> Result<(), FormatError> {
        if let Some(esize) = dtype.element_size() {
            let expected = shape.iter().product::<u64>() * esize;
            if expected != bytes.len() as u64 {
                return Err(FormatError::BadTensor {
                    name: name.into(),
                    message: format!(
                        "byte length {} does not match shape {shape:?} × {esize}",
                        bytes.len()
                    ),
                });
            }
        }
        let offset = (self.data.len() as u64).next_multiple_of(TENSOR_ALIGN);
        self.data.resize(offset as usize, 0);
        self.data.extend_from_slice(bytes);
        self.records.push(TensorRecord {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            offset,
            byte_length: bytes.len() as u64,
            xxh64: format!("{:016x}", xxh64(bytes, 0)),
        });
        Ok(())
    }

    pub fn finish(self) -> Vec<u8> {
        let metadata = serde_json::to_vec(&self.metadata).expect("metadata serializes");
        let directory = serde_json::to_vec(&self.records).expect("directory serializes");

        let var_start = 4 + 4 + 8 * 4;
        let dir_end = var_start as u64
            + metadata.len() as u64
            + self.tokenizer_bytes.len() as u64
            + directory.len() as u64;
        let data_start = dir_end.next_multiple_of(TENSOR_ALIGN);

        let mut out = Vec::with_capacity(data_start as usize + self.data.len() + 32);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.tokenizer_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&(directory.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.data.len() as u64).to_le_bytes());
        out.extend_from_slice(&metadata);
        out.extend_from_slice(&self.tokenizer_bytes);
        out.extend_from_slice(&directory);
        out.resize(data_start as usize, 0);
        out.extend_from_slice(&self.data);

        let digest = Sha256::digest(&out);
        out.extend_from_slice(&digest);
        out
    }
}
