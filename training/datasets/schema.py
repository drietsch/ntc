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


class ElementSpan(BaseModel):
    """Per-element provenance for a LIST value assembled from several spans."""

    model_config = ConfigDict(extra="forbid")

    char_span: CharSpan
    surface: str


class GoldArgument(BaseModel):
    """One gold argument binding.

    Provenance is a union of four sources (see `source`): a span in the
    utterance, one or more linked items from the host's selection, a token the
    host's resolver looked up, or nothing at all when the value is inferred.
    """

    model_config = ConfigDict(extra="forbid")

    parameter: str
    semantic_type: SemanticType
    value: Any
    char_span: CharSpan | None = None
    surface: str | None = None
    #: Where the value comes from. Absent on corpora without a context frame.
    source: Literal["USER", "LINKED_ITEM", "RESOLVER", "MODEL"] | None = None
    #: Element type of a LIST value, mirroring the ABI's ITEM line.
    item_type: SemanticType | None = None
    #: Refs into `context.linked` (e.g. ["L1", "L3"]).
    linked_refs: list[str] = Field(default_factory=list)
    #: The identifier token the host's resolver looked up.
    resolver_token: str | None = None
    #: Per-element spans for a LIST read out of the utterance.
    element_spans: list[ElementSpan] = Field(default_factory=list)
    #: Spans a single value was assembled from.
    composed_from: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_value(self) -> GoldArgument:
        self.value = validate_semantic_value(self.semantic_type, self.value)
        return self


class GoldUnresolved(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter: str
    reason: UnresolvedReason
    #: Machine-readable cause, so the host can phrase the question.
    hint: str | None = None
    #: Candidate readings, when the ambiguity is enumerable.
    options: list[Any] = Field(default_factory=list)


class LinkedItem(BaseModel):
    """An element the user linked into the chat (the host's selection)."""

    model_config = ConfigDict(extra="allow")

    ref: str
    type: str
    id: int
    key: str = ""
    path: str = ""
    isFolder: bool = False  # noqa: N815 — host wire format
    className: str | None = None  # noqa: N815


class ResolverEntry(BaseModel):
    """One identifier-like token the host looked up before the model ran."""

    model_config = ConfigDict(extra="allow")

    token: str
    char_span: CharSpan | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class RequestContext(BaseModel):
    """Everything the host knew when the utterance was typed."""

    model_config = ConfigDict(extra="allow")

    linked: list[LinkedItem] = Field(default_factory=list)
    resolver: list[ResolverEntry] = Field(default_factory=list)
    selection_count: int | None = None
    studio_view: str | None = None
    locale: str | None = None


class GoldLabel(BaseModel):
    """ActionIr-shaped gold label."""

    model_config = ConfigDict(extra="forbid")

    action: ActionState
    tool: str | None = None
    arguments: list[GoldArgument] = Field(default_factory=list)
    unresolved: list[GoldUnresolved] = Field(default_factory=list)
    #: Why the router escalated; required when action is DELEGATE.
    delegate_reason: (
        Literal["PAYLOAD_REQUIRED", "OVER_LIMIT", "MULTI_STEP", "MIXED_ELEMENT_TYPES"] | None
    ) = None
    #: The tool the router believes is involved, for the agent's benefit.
    suggested_tool: str | None = None
    #: Why nothing should run.
    no_call_reason: (
        Literal["CHITCHAT", "CONCEPTUAL_QUESTION", "UNSUPPORTED_CAPABILITY",
                "OUT_OF_SCOPE", "MENTION_ONLY"] | None
    ) = None


class DatasetExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    lang: Lang
    utterance: str = Field(min_length=1)
    candidates: list[RawToolSchema] = Field(min_length=1)
    gold: GoldLabel
    #: The host's context frame; empty for corpora without a selection model.
    context: RequestContext = Field(default_factory=RequestContext)
    split: Split = "train"
    tags: list[str] = Field(default_factory=list)
    #: Rationale and host inputs (authorization, precedence, template_id, ...)
    #: kept for eval slicing. Never a prediction target.
    annotations: dict[str, Any] = Field(default_factory=dict)

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
    def _check_linked_refs_resolve(self) -> DatasetExample:
        """Every linked_ref must name an item actually in the context."""
        refs = {item.ref for item in self.context.linked}
        for arg in self.gold.arguments:
            dangling = set(arg.linked_refs) - refs
            if dangling:
                raise ValueError(
                    f"arg `{arg.parameter}`: linked_refs {sorted(dangling)} not in context.linked"
                )
        return self

    @model_validator(mode="after")
    def _check_delegate_reason(self) -> DatasetExample:
        if self.gold.action == "DELEGATE" and self.gold.delegate_reason is None:
            raise ValueError("action DELEGATE requires a delegate_reason")
        if self.gold.action != "DELEGATE" and self.gold.delegate_reason is not None:
            raise ValueError("delegate_reason requires action DELEGATE")
        return self

    @model_validator(mode="after")
    def _check_element_spans(self) -> DatasetExample:
        for arg in self.gold.arguments:
            for el in arg.element_spans:
                if self.utterance[el.char_span.start : el.char_span.end] != el.surface:
                    raise ValueError(
                        f"arg `{arg.parameter}`: element span text != recorded surface"
                    )
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
