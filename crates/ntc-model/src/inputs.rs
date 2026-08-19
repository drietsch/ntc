//! Model input packing (head codec `inputs.packing`).
//!
//! Builds padded token/mask/segment arrays plus **anchor positions** — the
//! token indices whose fused states the heads read: each tool's first token,
//! each arg's name token, each enum value's token. Anchors are discovered via
//! tokenizer byte offsets against the canonical text's rendered line spans,
//! so they are correct for any tokenizer that reports offsets.

use ntc_core::schema::{CanonicalTool, LineKind};
use ntc_core::tokenizer::{NtcTokenizer, TokenSeq};
use ntc_core::NtcError;

use crate::config::NtcArchConfig;

/// Segment-kind vocabulary (embedding rows in
/// `schema.embeddings.segment_kind.weight`; count = [`crate::weights::SEGMENT_KINDS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SegmentKind {
    Special = 0,
    ToolHeader = 1,
    Desc = 2,
    ArgName = 3,
    Info = 4,
    Type = 5,
    Required = 6,
    Semantic = 7,
    EnumValue = 8,
    Pad = 9,
    /// `ITEM <TYPE>` — element type of a LIST argument (ABI v2).
    Item = 10,
}

impl From<LineKind> for SegmentKind {
    fn from(k: LineKind) -> Self {
        match k {
            LineKind::ToolHeader => SegmentKind::ToolHeader,
            LineKind::Desc => SegmentKind::Desc,
            LineKind::ArgName => SegmentKind::ArgName,
            LineKind::Info => SegmentKind::Info,
            LineKind::Type => SegmentKind::Type,
            LineKind::Required => SegmentKind::Required,
            LineKind::Semantic => SegmentKind::Semantic,
            LineKind::EnumValue => SegmentKind::EnumValue,
            LineKind::Item => SegmentKind::Item,
        }
    }
}

/// One candidate tool's packed input.
#[derive(Debug, Clone)]
pub struct ToolInput {
    /// Padded to `max_schema_tokens`.
    pub ids: Vec<u32>,
    pub mask: Vec<bool>,
    pub kinds: Vec<u8>,
    /// Token index of the tool's first schema token (`TOOL` keyword).
    pub tool_anchor: usize,
    /// Per declared arg: token index of the arg-name anchor.
    pub arg_anchors: Vec<usize>,
    /// Per declared arg: token index of each enum value's anchor.
    pub enum_anchors: Vec<Vec<usize>>,
}

/// Linked-item kinds the entity-reference head embeds (frozen order; 0 =
/// unknown). Mirrors `training/datasets/collate.py::LINKED_KINDS`.
pub const LINKED_KINDS: [&str; 5] = ["asset", "document", "object", "data-object", "folder"];
/// Entity-reference head width (excluding the trailing NONE slot).
pub const MAX_LINKED: usize = 8;

/// Embedding row for a host element kind.
pub fn linked_kind_id(kind: &str) -> usize {
    LINKED_KINDS
        .iter()
        .position(|k| *k == kind)
        .map_or(0, |i| i + 1)
}

#[derive(Debug, Clone)]
pub struct ModelInputs {
    /// Padded to `max_utterance_tokens`.
    pub utterance_ids: Vec<u32>,
    pub utterance_mask: Vec<bool>,
    /// The utterance's unpadded token count.
    pub utterance_len: usize,
    pub tools: Vec<ToolInput>,
    /// Kind ids of the items linked into the request, truncated to
    /// [`MAX_LINKED`]. Empty when the host supplied no context.
    pub linked_kinds: Vec<usize>,
}

impl ModelInputs {
    /// Pack a tokenized utterance and candidate tools.
    ///
    /// `utterance` is truncated to `max_utterance_tokens`. A tool whose
    /// anchors do not fit within `max_schema_tokens` is an error (spec: fail
    /// loudly rather than silently dropping arguments).
    pub fn pack(
        cfg: &NtcArchConfig,
        tokenizer: &NtcTokenizer,
        utterance: &TokenSeq,
        candidates: &[&CanonicalTool],
    ) -> Result<Self, NtcError> {
        Self::pack_with_context(cfg, tokenizer, utterance, candidates, &[])
    }

    /// Packing with the host's context frame, so the entity-reference head
    /// can bind arguments to the current selection.
    pub fn pack_with_context(
        cfg: &NtcArchConfig,
        tokenizer: &NtcTokenizer,
        utterance: &TokenSeq,
        candidates: &[&CanonicalTool],
        linked: &[ntc_core::ir::LinkedItem],
    ) -> Result<Self, NtcError> {
        if candidates.is_empty() || candidates.len() > cfg.max_tools {
            return Err(NtcError::CandidateLimit(format!(
                "{} candidates (model supports 1..={})",
                candidates.len(),
                cfg.max_tools
            )));
        }

        let lu = cfg.max_utterance_tokens;
        let n = utterance.ids.len().min(lu);
        let mut utterance_ids = vec![0u32; lu];
        let mut utterance_mask = vec![false; lu];
        utterance_ids[..n].copy_from_slice(&utterance.ids[..n]);
        utterance_mask[..n].fill(true);

        let mut tools = Vec::with_capacity(candidates.len());
        for (t, tool) in candidates.iter().enumerate() {
            tools.push(Self::pack_tool(cfg, tokenizer, tool, t)?);
        }

        Ok(Self {
            utterance_ids,
            utterance_mask,
            utterance_len: n,
            tools,
            linked_kinds: linked
                .iter()
                .take(MAX_LINKED)
                .map(|l| linked_kind_id(&l.kind))
                .collect(),
        })
    }

    fn pack_tool(
        cfg: &NtcArchConfig,
        tokenizer: &NtcTokenizer,
        tool: &CanonicalTool,
        candidate_index: usize,
    ) -> Result<ToolInput, NtcError> {
        if tool.args.len() > cfg.max_args {
            return Err(NtcError::Schema(format!(
                "tool `{}` declares {} args (model supports ≤ {})",
                tool.id,
                tool.args.len(),
                cfg.max_args
            )));
        }
        let (text, lines) = tool.to_neural_text_with_lines(candidate_index);
        let seq = tokenizer.encode_schema_text(&text)?;
        let ls = cfg.max_schema_tokens;
        if seq.ids.len() > ls {
            // Anchor completeness beats silent truncation: reject.
            return Err(NtcError::Schema(format!(
                "tool `{}`: canonical text tokenizes to {} tokens (> {ls}); \
                 shorten descriptions or raise max_schema_tokens",
                tool.id,
                seq.ids.len()
            )));
        }

        let mut ids = vec![0u32; ls];
        let mut mask = vec![false; ls];
        let mut kinds = vec![SegmentKind::Pad as u8; ls];
        ids[..seq.ids.len()].copy_from_slice(&seq.ids);
        mask[..seq.ids.len()].fill(true);

        // Kind per token: the line containing the token's byte-offset start.
        for (i, &(s, e)) in seq.offsets.iter().enumerate() {
            kinds[i] = if e <= s {
                SegmentKind::Special as u8
            } else {
                lines
                    .iter()
                    .find(|l| s >= l.range.0 && s < l.range.1)
                    .map(|l| SegmentKind::from(l.kind) as u8)
                    .unwrap_or(SegmentKind::Special as u8)
            };
        }

        let find_anchor = |range: (usize, usize)| -> Option<usize> {
            seq.offsets
                .iter()
                .position(|&(s, e)| e > s && s < range.1 && e > range.0)
        };

        // Tool anchor: first real (non-special) token.
        let tool_anchor = seq
            .offsets
            .iter()
            .position(|&(s, e)| e > s)
            .ok_or_else(|| NtcError::Schema(format!("tool `{}`: empty tokenization", tool.id)))?;

        let mut arg_anchors = Vec::with_capacity(tool.args.len());
        let mut enum_anchors = Vec::with_capacity(tool.args.len());
        for (k, arg) in tool.args.iter().enumerate() {
            let name_line = lines
                .iter()
                .find(|l| l.kind == LineKind::ArgName && l.arg_index == Some(k))
                .expect("rendered ARG line exists for every arg");
            let anchor = name_line.anchor.and_then(find_anchor).ok_or_else(|| {
                NtcError::Schema(format!(
                    "tool `{}`: no token anchor for arg `{}`",
                    tool.id, arg.name
                ))
            })?;
            arg_anchors.push(anchor);

            let mut evs = Vec::with_capacity(arg.enum_values.len());
            for j in 0..arg.enum_values.len() {
                let line = lines
                    .iter()
                    .find(|l| {
                        l.kind == LineKind::EnumValue
                            && l.arg_index == Some(k)
                            && l.enum_index == Some(j)
                    })
                    .expect("rendered ENUM line exists for every enum value");
                let anchor = line.anchor.and_then(find_anchor).ok_or_else(|| {
                    NtcError::Schema(format!(
                        "tool `{}`: no token anchor for enum value {j} of `{}`",
                        tool.id, arg.name
                    ))
                })?;
                evs.push(anchor);
            }
            enum_anchors.push(evs);
        }

        Ok(ToolInput {
            ids,
            mask,
            kinds,
            tool_anchor,
            arg_anchors,
            enum_anchors,
        })
    }
}
