"""Dataset example schema for the synthetic data engine.

Gold labels are ActionIr-shaped but use **character-offset spans** into the
utterance (`char_span: [start, end)`) instead of token spans — tokenization
happens at training time against the current tokenizer, so labels stay
tokenizer-independent.

Validated invariants:
- every `char_span` must select a non-empty utterance substring, and match the
  recorded `surface` string if one is given;
- `action == ASK` ⇔ at least one unresolved entry;
- `action == CALL` ⇒ a gold tool is present (and is one of the candidates).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ntc_contracts.ir import (
    ActionState,
    SemanticType,
    UnresolvedReason,
    validate_semantic_value,
)

Lang = Literal["en", "de", "fr", "es"]
Split = Literal["train", "dev", "test"]


class RawToolSchema(BaseModel):
    """A raw (pre-canonicalization) tool schema, OpenAI-function style.

    Kept permissive on purpose: canonicalization and deep validation have
    exactly one implementation — the Rust `ntc schemac` CLI.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolFamily(BaseModel):
    """A themed group of raw tool schemas (one teacher tool-gen batch)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    domain: str
    description: str | None = None
    tools: list[RawToolSchema] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_tool_names(self) -> ToolFamily:
        names = [t.name for t in self.tools]
        if len(names) != len(set(names)):
            raise ValueError(f"tool family `{self.id}`: duplicate tool names")
        return self


class CharSpan(BaseModel):
    """`[start, end)` character offsets over the utterance."""

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _non_empty(self) -> CharSpan:
        if self.end <= self.start:
            raise ValueError(f"char_span [{self.start}, {self.end}) is empty")
        return self


class GoldArgument(BaseModel):
    """One gold argument binding (char-offset provenance)."""

    model_config = ConfigDict(extra="forbid")

    parameter: str
    semantic_type: SemanticType
    value: Any
    char_span: CharSpan | None = None
    surface: str | None = None

    @model_validator(mode="after")
    def _check_value(self) -> GoldArgument:
        self.value = validate_semantic_value(self.semantic_type, self.value)
        return self


class GoldUnresolved(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter: str
    reason: UnresolvedReason


class GoldLabel(BaseModel):
    """ActionIr-shaped gold label."""

    model_config = ConfigDict(extra="forbid")

    action: ActionState
    tool: str | None = None
    arguments: list[GoldArgument] = Field(default_factory=list)
    unresolved: list[GoldUnresolved] = Field(default_factory=list)


class DatasetExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    lang: Lang
    utterance: str = Field(min_length=1)
    candidates: list[RawToolSchema] = Field(min_length=1)
    gold: GoldLabel
    split: Split = "train"
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_spans(self) -> DatasetExample:
        for arg in self.gold.arguments:
            if arg.char_span is None:
                continue
            span = arg.char_span
            if span.end > len(self.utterance):
                raise ValueError(
                    f"arg `{arg.parameter}`: char_span [{span.start}, {span.end}) exceeds "
                    f"utterance length {len(self.utterance)}"
                )
            text = self.utterance[span.start : span.end]
            if not text:
                raise ValueError(f"arg `{arg.parameter}`: char_span selects an empty substring")
            if arg.surface is not None and text != arg.surface:
                raise ValueError(
                    f"arg `{arg.parameter}`: span text {text!r} != recorded surface "
                    f"{arg.surface!r}"
                )
        return self

    @model_validator(mode="after")
    def _check_ask_iff_unresolved(self) -> DatasetExample:
        if self.gold.action == "ASK" and not self.gold.unresolved:
            raise ValueError("action ASK requires at least one unresolved entry")
        if self.gold.action != "ASK" and self.gold.unresolved:
            raise ValueError("unresolved entries require action ASK")
        return self

    @model_validator(mode="after")
    def _check_delegate_is_bare(self) -> DatasetExample:
        """DELEGATE is a whole-utterance verdict: the router hands the request
        to a full LLM agent, so it carries no tool, arguments or unresolved."""
        if self.gold.action == "DELEGATE" and (
            self.gold.tool or self.gold.arguments or self.gold.unresolved
        ):
            raise ValueError("action DELEGATE must not carry tool/arguments/unresolved")
        return self

    @model_validator(mode="after")
    def _check_call_has_tool(self) -> DatasetExample:
        if self.gold.action == "CALL":
            if self.gold.tool is None:
                raise ValueError("action CALL requires a gold tool")
            names = {t.name for t in self.candidates}
            if self.gold.tool not in names:
                raise ValueError(
                    f"gold tool `{self.gold.tool}` is not among the candidates {sorted(names)}"
                )
        return self
