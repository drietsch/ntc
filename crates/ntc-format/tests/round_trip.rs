//! Round-trip and corruption tests for the `.ntc` container.
//! Run with: cargo test -p ntc-format --features write

#![cfg(feature = "write")]

use ntc_format::writer::NtcWriter;
use ntc_format::{DType, FormatError, NtcFile, NtcMetadata, TENSOR_ALIGN};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn tiny_metadata() -> NtcMetadata {
    NtcMetadata {
        architecture: "ntc_encoder_heads_v1".into(),
        model_version: "test-tiny".into(),
        ir_version: 1,
        abi_version: 1,
        head_spec_version: 1,
        tokenizer_sha256: String::new(), // filled by the writer
        quantization: "f32".into(),
        model: serde_json::json!({"hidden": 8, "layers": 1}),
        semantic_types: vec!["STRING".into(), "DURATION".into()],
    }
}

fn build_tiny() -> Vec<u8> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut w = NtcWriter::new(tiny_metadata(), b"{\"fake\":\"tokenizer\"}".to_vec());
    let emb: Vec<f32> = (0..16 * 8).map(|_| rng.gen_range(-1.0..1.0)).collect();
    w.add_tensor(
        "embedding.weight",
        DType::F32,
        &[16, 8],
        &bytemuck_cast(&emb),
    )
    .unwrap();
    let ffn: Vec<f32> = (0..8 * 8).map(|_| rng.gen_range(-1.0..1.0)).collect();
    w.add_tensor(
        "encoder.layer.0.ffn.weight",
        DType::F32,
        &[8, 8],
        &bytemuck_cast(&ffn),
    )
    .unwrap();
    w.finish()
}

fn bytemuck_cast(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[test]
fn round_trip() {
    let buf = build_tiny();
    let file = NtcFile::parse(&buf).unwrap();
    assert_eq!(file.metadata.model_version, "test-tiny");
    assert_eq!(file.records.len(), 2);
    file.check_tensor_hashes().unwrap();

    let emb = file.tensor("embedding.weight").unwrap();
    assert_eq!(emb.record.shape, vec![16, 8]);
    assert_eq!(emb.bytes.len(), 16 * 8 * 4);
    assert_eq!(emb.record.offset % TENSOR_ALIGN, 0);

    let report = ntc_format::verify(&buf).unwrap();
    assert_eq!(report.tensor_count, 2);
}

#[test]
fn bad_magic_rejected() {
    let mut buf = build_tiny();
    buf[0] = b'X';
    assert_eq!(NtcFile::parse(&buf).unwrap_err(), FormatError::BadMagic);
}

#[test]
fn unsupported_version_rejected() {
    let mut buf = build_tiny();
    buf[4..8].copy_from_slice(&99u32.to_le_bytes());
    assert_eq!(
        NtcFile::parse(&buf).unwrap_err(),
        FormatError::UnsupportedVersion(99)
    );
}

#[test]
fn truncation_rejected() {
    let buf = build_tiny();
    let cut = &buf[..buf.len() - 40];
    assert!(matches!(
        NtcFile::parse(cut).unwrap_err(),
        FormatError::Truncated(_)
    ));
}

#[test]
fn flipped_data_byte_fails_sha256() {
    let mut buf = build_tiny();
    let n = buf.len();
    buf[n - 100] ^= 0xff; // inside the data section
    assert!(matches!(
        NtcFile::parse(&buf).unwrap_err(),
        FormatError::ChecksumMismatch(_)
    ));
}

#[test]
fn reserved_dtype_rejected_distinctly() {
    let mut w = NtcWriter::new(tiny_metadata(), b"{}".to_vec());
    // 32 ternary weights packed into 2 u32 words (8 bytes).
    w.add_tensor("w", DType::TernaryT2, &[32], &[0u8; 8])
        .unwrap();
    let buf = w.finish();
    assert!(matches!(
        NtcFile::parse(&buf).unwrap_err(),
        FormatError::ReservedDtype { .. }
    ));
}

#[test]
fn duplicate_tensor_rejected() {
    let mut w = NtcWriter::new(tiny_metadata(), b"{}".to_vec());
    w.add_tensor("dup", DType::F32, &[2], &[0u8; 8]).unwrap();
    w.add_tensor("dup", DType::F32, &[2], &[0u8; 8]).unwrap();
    let buf = w.finish();
    assert!(matches!(
        NtcFile::parse(&buf).unwrap_err(),
        FormatError::DuplicateTensor(_)
    ));
}
