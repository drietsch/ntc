//! Zero-copy `.ntc` parser.

use std::collections::HashMap;

use sha2::{Digest, Sha256};
use xxhash_rust::xxh64::xxh64;

use crate::header::{NtcMetadata, SectionLengths, FORMAT_VERSION, MAGIC};
use crate::tensor::{TensorRecord, TensorView};
use crate::{FormatError, TENSOR_ALIGN};

const FIXED_HEADER_LEN: u64 = 4 + 4 + 8 * 4;
const FOOTER_LEN: u64 = 32;

/// A parsed `.ntc` file borrowing the underlying buffer (browser: the fetched
/// ArrayBuffer copied into wasm memory; native: an mmap or `fs::read` vec).
#[derive(Debug)]
pub struct NtcFile<'a> {
    pub metadata: NtcMetadata,
    pub tokenizer_bytes: &'a [u8],
    pub records: Vec<TensorRecord>,
    data: &'a [u8],
    by_name: HashMap<String, usize>,
}

impl<'a> NtcFile<'a> {
    /// Parse and structurally validate a `.ntc` buffer.
    ///
    /// Integrity checks performed here: magic, version, section bounds,
    /// alignment, directory sanity (bounds, overlap-free ordering,
    /// duplicates), whole-file sha256 footer, tokenizer sha256 vs metadata.
    /// Per-tensor xxh64 is *not* checked here (it is O(model size)); call
    /// [`crate::verify::verify`] for deep verification.
    pub fn parse(buf: &'a [u8]) -> Result<Self, FormatError> {
        let total = buf.len() as u64;
        if total < FIXED_HEADER_LEN + FOOTER_LEN {
            return Err(FormatError::Truncated(
                "shorter than fixed header + footer".into(),
            ));
        }
        if &buf[0..4] != MAGIC {
            return Err(FormatError::BadMagic);
        }
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(FormatError::UnsupportedVersion(version));
        }
        let lens = SectionLengths {
            metadata_len: read_u64(buf, 8),
            tokenizer_len: read_u64(buf, 16),
            directory_len: read_u64(buf, 24),
            data_len: read_u64(buf, 32),
        };

        let var_start = FIXED_HEADER_LEN;
        let meta_end = var_start
            .checked_add(lens.metadata_len)
            .ok_or_else(|| FormatError::Truncated("metadata length overflow".into()))?;
        let tok_end = meta_end
            .checked_add(lens.tokenizer_len)
            .ok_or_else(|| FormatError::Truncated("tokenizer length overflow".into()))?;
        let dir_end = tok_end
            .checked_add(lens.directory_len)
            .ok_or_else(|| FormatError::Truncated("directory length overflow".into()))?;
        let data_start = dir_end.next_multiple_of(TENSOR_ALIGN);
        let data_end = data_start
            .checked_add(lens.data_len)
            .ok_or_else(|| FormatError::Truncated("data length overflow".into()))?;
        let expected_total = data_end + FOOTER_LEN;
        if expected_total != total {
            return Err(FormatError::Truncated(format!(
                "expected {expected_total} bytes from header, file has {total}"
            )));
        }

        // Whole-file integrity: sha256 over everything before the footer.
        let mut hasher = Sha256::new();
        hasher.update(&buf[..data_end as usize]);
        let digest = hasher.finalize();
        if digest.as_slice() != &buf[data_end as usize..] {
            return Err(FormatError::ChecksumMismatch("file sha256 footer".into()));
        }

        let metadata: NtcMetadata =
            serde_json::from_slice(&buf[var_start as usize..meta_end as usize])
                .map_err(|e| FormatError::BadMetadata(e.to_string()))?;
        let tokenizer_bytes = &buf[meta_end as usize..tok_end as usize];

        let tok_sha = hex(&Sha256::digest(tokenizer_bytes));
        if tok_sha != metadata.tokenizer_sha256 {
            return Err(FormatError::ChecksumMismatch(format!(
                "tokenizer sha256 (metadata says {}, bytes hash to {tok_sha})",
                metadata.tokenizer_sha256
            )));
        }

        let records: Vec<TensorRecord> =
            serde_json::from_slice(&buf[tok_end as usize..dir_end as usize])
                .map_err(|e| FormatError::BadDirectory(e.to_string()))?;

        let mut by_name = HashMap::with_capacity(records.len());
        let mut cursor = 0u64;
        for (i, r) in records.iter().enumerate() {
            if by_name.insert(r.name.clone(), i).is_some() {
                return Err(FormatError::DuplicateTensor(r.name.clone()));
            }
            if r.offset % TENSOR_ALIGN != 0 {
                return Err(FormatError::Misaligned {
                    what: format!("tensor `{}`", r.name),
                    offset: r.offset,
                });
            }
            if r.offset < cursor {
                return Err(FormatError::BadTensor {
                    name: r.name.clone(),
                    message: "overlaps the previous tensor (directory must be offset-sorted)"
                        .into(),
                });
            }
            let end =
                r.offset
                    .checked_add(r.byte_length)
                    .ok_or_else(|| FormatError::BadTensor {
                        name: r.name.clone(),
                        message: "offset + length overflow".into(),
                    })?;
            if end > lens.data_len {
                return Err(FormatError::BadTensor {
                    name: r.name.clone(),
                    message: format!(
                        "extends to {end} beyond data section length {}",
                        lens.data_len
                    ),
                });
            }
            // Dense dtypes must be exactly shape × element size.
            if let Some(esize) = r.dtype.element_size() {
                let expected = r.element_count() * esize;
                if expected != r.byte_length {
                    return Err(FormatError::BadTensor {
                        name: r.name.clone(),
                        message: format!(
                            "byte_length {} does not match shape {:?} × {esize} = {expected}",
                            r.byte_length, r.shape
                        ),
                    });
                }
            }
            if !r.dtype.is_supported_v1() {
                return Err(FormatError::ReservedDtype {
                    name: r.name.clone(),
                    dtype: r.dtype.name().into(),
                });
            }
            cursor = end;
        }

        Ok(Self {
            metadata,
            tokenizer_bytes,
            records,
            data: &buf[data_start as usize..data_end as usize],
            by_name,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<TensorView<'_>> {
        let &i = self.by_name.get(name)?;
        let r = &self.records[i];
        Some(TensorView {
            record: r,
            bytes: &self.data[r.offset as usize..(r.offset + r.byte_length) as usize],
        })
    }

    pub fn tensors(&self) -> impl Iterator<Item = TensorView<'_>> {
        self.records.iter().map(move |r| TensorView {
            record: r,
            bytes: &self.data[r.offset as usize..(r.offset + r.byte_length) as usize],
        })
    }

    /// Deep check: per-tensor xxh64 hashes.
    pub fn check_tensor_hashes(&self) -> Result<(), FormatError> {
        for view in self.tensors() {
            let h = format!("{:016x}", xxh64(view.bytes, 0));
            if h != view.record.xxh64 {
                return Err(FormatError::ChecksumMismatch(format!(
                    "tensor `{}` xxh64 (directory says {}, bytes hash to {h})",
                    view.record.name, view.record.xxh64
                )));
            }
        }
        Ok(())
    }
}

fn read_u64(buf: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(buf[at..at + 8].try_into().unwrap())
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
