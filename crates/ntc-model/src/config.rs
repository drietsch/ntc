//! Architecture configuration, deserialized from `.ntc` metadata `model`.

use serde::{Deserialize, Serialize};

use ntc_core::NtcError;

fn default_eps() -> f32 {
    1e-5
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
    /// Per-head calibration temperatures (head codec §confidence).
    #[serde(default)]
    pub calibration: Calibration,
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

    /// Packed fusion sequence length: T tool segments + the NO_TOOL slot.
    pub fn packed_len(&self) -> usize {
        self.max_tools * self.max_schema_tokens + 1
    }
}
