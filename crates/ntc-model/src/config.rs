//! Architecture configuration, deserialized from `.ntc` metadata `model`.

use serde::{Deserialize, Serialize};

use ntc_core::NtcError;

fn default_eps() -> f32 {
    1e-5
}

/// Action-head width for models trained before `DELEGATE` existed.
fn default_action_classes() -> usize {
    3
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NtcArchConfig {
    pub hidden: usize,
    /// Attention heads; must divide `hidden`.
    pub heads: usize,
    /// FFN inner dimension.
    pub ffn: usize,
    pub vocab: usize,
    pub max_positions: usize,
    pub encoder_layers: usize,
    pub schema_layers: usize,
    pub fusion_blocks: usize,
    pub max_tools: usize,
    pub max_args: usize,
    pub max_enum_values: usize,
    pub max_utterance_tokens: usize,
    pub max_schema_tokens: usize,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f32,
    /// Action-head width: 3 = CALL/ASK/NO_CALL, 4 adds DELEGATE.
    #[serde(default = "default_action_classes")]
    pub action_classes: usize,
    /// Per-head calibration temperatures (head codec §confidence).
    #[serde(default)]
    pub calibration: Calibration,
    /// Host-declared value templates the filter-template head indexes into
    /// (head codec v4). Empty means the model has no such head.
    ///
    /// The head's class order is `NONE` at index 0 followed by this list in
    /// order, so the table is part of what the model was trained against and
    /// travels with the weights rather than with the registry.
    #[serde(default)]
    pub filter_templates: Vec<FilterTemplate>,
}

/// One host-declared way to construct an argument value the utterance does
/// not spell out literally (spec §19; head codec v4 `filter_template`).
///
/// The compiler has no decoder, so it cannot write an arbitrary query string.
/// What it can do is *choose* among the shapes a host declares and fill their
/// slots from the span it already marks — the same factored trick the datetime
/// head uses (pick `NEXT` + `FRIDAY`, don't spell out a date).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilterTemplate {
    /// Stable id, for diagnostics and for the training-side codec.
    pub id: String,
    /// The `SEMANTIC` annotation this template serves. Only arguments carrying
    /// it may select this template; every other template is masked out.
    pub semantic: String,
    /// Literal text with `{field}`, `{number}` and `{token}` placeholders,
    /// each filled deterministically from the marked span. A pattern with no
    /// placeholder is a constant and needs no span at all.
    pub pattern: String,
    /// Closed set of fillers for this pattern's `{token}` slot, if it has one.
    /// The span is matched against these rather than copied, so plurals,
    /// compounds and case never have to be undone ("PDFs" -> `pdf`). A span
    /// naming none of them leaves the argument unresolved.
    #[serde(default)]
    pub values: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    #[serde(default = "one")]
    pub action: f32,
    #[serde(default = "one")]
    pub tool: f32,
    #[serde(default = "one")]
    pub presence: f32,
    #[serde(default = "one")]
    pub value: f32,
}

fn one() -> f32 {
    1.0
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            action: 1.0,
            tool: 1.0,
            presence: 1.0,
            value: 1.0,
        }
    }
}

impl NtcArchConfig {
    pub fn from_metadata(model: &serde_json::Value) -> Result<Self, NtcError> {
        let cfg: Self = serde_json::from_value(model.clone())
            .map_err(|e| NtcError::Format(format!("bad model config: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), NtcError> {
        if self.hidden == 0 || self.heads == 0 || !self.hidden.is_multiple_of(self.heads) {
            return Err(NtcError::Format(format!(
                "hidden {} must be a positive multiple of heads {}",
                self.hidden, self.heads
            )));
        }
        if !(3..=4).contains(&self.action_classes) {
            return Err(NtcError::Format(format!(
                "action_classes must be 3 or 4, got {}",
                self.action_classes
            )));
        }
        if self.max_utterance_tokens > self.max_positions
            || self.max_schema_tokens > self.max_positions
        {
            return Err(NtcError::Format(
                "sequence limits exceed max_positions".into(),
            ));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// Width of the filter-template head: `NONE` plus one class per declared
    /// template. Zero when the model declares none, in which case the head's
    /// tensors are absent and the output is not emitted.
    pub fn filter_template_classes(&self) -> usize {
        if self.filter_templates.is_empty() {
            0
        } else {
            self.filter_templates.len() + 1
        }
    }

    /// Packed fusion sequence length: T tool segments + the NO_TOOL slot.
    pub fn packed_len(&self) -> usize {
        self.max_tools * self.max_schema_tokens + 1
    }
}
