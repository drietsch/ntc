//! Deep verification for `ntc verify` (the conformance CLI entry point).

use serde::Serialize;

use crate::loader::NtcFile;
use crate::FormatError;

#[derive(Debug, Serialize)]
pub struct VerifyReport {
    pub format_version: u32,
    pub architecture: String,
    pub model_version: String,
    pub quantization: String,
    pub ir_version: u32,
    pub abi_version: u32,
    pub head_spec_version: u32,
    pub tokenizer_sha256: String,
    pub tokenizer_bytes: usize,
    pub tensor_count: usize,
    pub total_tensor_bytes: u64,
    pub tensors: Vec<TensorSummary>,
}

#[derive(Debug, Serialize)]
pub struct TensorSummary {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub byte_length: u64,
}

/// Parse + structural validation + per-tensor hash check, returning a
/// manifest suitable for `--dump-manifest` JSON comparison in conformance CI.
pub fn verify(buf: &[u8]) -> Result<VerifyReport, FormatError> {
    let file = NtcFile::parse(buf)?;
    file.check_tensor_hashes()?;
    Ok(VerifyReport {
        format_version: crate::header::FORMAT_VERSION,
        architecture: file.metadata.architecture.clone(),
        model_version: file.metadata.model_version.clone(),
        quantization: file.metadata.quantization.clone(),
        ir_version: file.metadata.ir_version,
        abi_version: file.metadata.abi_version,
        head_spec_version: file.metadata.head_spec_version,
        tokenizer_sha256: file.metadata.tokenizer_sha256.clone(),
        tokenizer_bytes: file.tokenizer_bytes.len(),
        tensor_count: file.records.len(),
        total_tensor_bytes: file.records.iter().map(|r| r.byte_length).sum(),
        tensors: file
            .records
            .iter()
            .map(|r| TensorSummary {
                name: r.name.clone(),
                dtype: r.dtype.name().into(),
                shape: r.shape.clone(),
                offset: r.offset,
                byte_length: r.byte_length,
            })
            .collect(),
    })
}
