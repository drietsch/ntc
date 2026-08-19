"""Architecture configuration — mirrors `crates/ntc-model/src/config.rs`.

Serializes to exactly the `.ntc` metadata `model` object the Rust runtime
parses (`NtcArchConfig` with `deny_unknown_fields`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

ARCHITECTURE = "ntc_encoder_heads_v1"

#: Number of segment-kind embedding rows (crates/ntc-model/src/inputs.rs).
SEGMENT_KINDS = 11


class Calibration(BaseModel):
    """Per-head calibration temperatures (head codec §confidence)."""

    model_config = ConfigDict(extra="forbid")

    action: float = 1.0
    tool: float = 1.0
    presence: float = 1.0
    value: float = 1.0


class NtcArchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden: int = Field(gt=0)
    heads: int = Field(gt=0)
    ffn: int = Field(gt=0)
    vocab: int = Field(gt=0)
    max_positions: int = Field(gt=0)
    encoder_layers: int = Field(ge=0)
    schema_layers: int = Field(ge=0)
    fusion_blocks: int = Field(ge=0)
    max_tools: int = Field(gt=0)
    max_args: int = Field(gt=0)
    max_enum_values: int = Field(gt=0)
    max_utterance_tokens: int = Field(gt=0)
    max_schema_tokens: int = Field(gt=0)
    layer_norm_eps: float = 1e-5
    action_classes: int = Field(default=3, ge=3, le=4)
    calibration: Calibration = Field(default_factory=Calibration)

    @model_validator(mode="after")
    def _validate(self) -> NtcArchConfig:
        if self.hidden % self.heads != 0:
            raise ValueError(f"hidden {self.hidden} must be a multiple of heads {self.heads}")
        if (
            self.max_utterance_tokens > self.max_positions
            or self.max_schema_tokens > self.max_positions
        ):
            raise ValueError("sequence limits exceed max_positions")
        return self

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def packed_len(self, n_tools: int | None = None) -> int:
        """Packed fusion sequence length: T tool segments + the NO_TOOL slot."""
        t = self.max_tools if n_tools is None else n_tools
        return t * self.max_schema_tokens + 1

    def to_metadata_model(self) -> dict:
        """The `.ntc` metadata `model` object (exact key set the runtime parses)."""
        return self.model_dump(mode="json")


def tiny_config() -> NtcArchConfig:
    """Tiny dims — must stay identical to `ntc_model::test_support::tiny_config`."""
    return NtcArchConfig(
        hidden=32,
        heads=4,
        ffn=64,
        vocab=64,
        max_positions=128,
        encoder_layers=2,
        schema_layers=1,
        fusion_blocks=1,
        max_tools=4,
        max_args=4,
        max_enum_values=4,
        max_utterance_tokens=24,
        max_schema_tokens=64,
        layer_norm_eps=1e-5,
    )
