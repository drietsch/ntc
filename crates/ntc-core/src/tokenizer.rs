//! Thin wrapper over the HuggingFace `tokenizers` crate (same Rust core the
//! Python bindings wrap → exact train/serve parity). Exposes token↔byte
//! offset maps, required to resolve span-pointer head predictions back to
//! utterance text (the IR span contract is token indices).

use tokenizers::Tokenizer;

use crate::error::NtcError;

/// A tokenized sequence with byte-offset provenance per token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSeq {
    pub ids: Vec<u32>,
    /// Per-token `[start, end)` byte offsets into the original text.
    /// Special tokens carry `(0, 0)`.
    pub offsets: Vec<(usize, usize)>,
}

impl TokenSeq {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Resolve a `[start, end)` token span to the covered source text.
    /// Skips zero-width (special) tokens at the boundaries.
    pub fn span_text<'a>(&self, text: &'a str, start: u32, end: u32) -> Option<&'a str> {
        let (start, end) = (start as usize, end as usize);
        if start >= end || end > self.offsets.len() {
            return None;
        }
        let real: Vec<(usize, usize)> = self.offsets[start..end]
            .iter()
            .copied()
            .filter(|&(s, e)| e > s)
            .collect();
        let (&(first, _), &(_, last)) = (real.first()?, real.last()?);
        text.get(first..last)
    }
}

pub struct NtcTokenizer {
    inner: Tokenizer,
}

impl NtcTokenizer {
    /// Load from raw `tokenizer.json` bytes (as embedded in a `.ntc` file).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NtcError> {
        let inner = Tokenizer::from_bytes(bytes)
            .map_err(|e| NtcError::Tokenizer(format!("failed to load tokenizer.json: {e}")))?;
        Ok(Self { inner })
    }

    /// Encode a user utterance (with the tokenizer's own special tokens).
    pub fn encode_utterance(&self, text: &str) -> Result<TokenSeq, NtcError> {
        self.encode(text, true)
    }

    /// Encode canonical schema text (with special tokens; segment structure
    /// is added by the input packer, not here).
    pub fn encode_schema_text(&self, text: &str) -> Result<TokenSeq, NtcError> {
        self.encode(text, true)
    }

    fn encode(&self, text: &str, add_special: bool) -> Result<TokenSeq, NtcError> {
        let enc = self
            .inner
            .encode(text, add_special)
            .map_err(|e| NtcError::Tokenizer(format!("encode failed: {e}")))?;
        Ok(TokenSeq {
            ids: enc.get_ids().to_vec(),
            offsets: enc.get_offsets().to_vec(),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
