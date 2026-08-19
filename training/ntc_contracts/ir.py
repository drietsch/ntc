"""Typed Action IR + CompileRequest (contracts/action-ir/v1).

Mirrors `crates/ntc-core/src/ir.rs` as generated into
`contracts/action-ir/v1/{action-ir,compile-request}.schema.json`.

Semantic values use adjacent tagging on the wire (spec §18):
``{"semantic_type": "DURATION", "value": {"magnitude": 1, "unit": "HOUR"}}``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

ActionState = Literal["CALL", "ASK", "NO_CALL", "DELEGATE"]
UnresolvedReason = Literal["MISSING", "AMBIGUOUS"]
DateRelation = Literal["TODAY", "TOMORROW", "YESTERDAY", "THIS", "NEXT", "LAST", "IN", "AGO"]
Weekday = Literal[
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"
]
Daypart = Literal["MORNING", "NOON", "AFTERNOON", "EVENING", "NIGHT"]
DurationUnit = Literal["SECOND", "MINUTE", "HOUR", "DAY", "WEEK"]
ProvenanceSource = Literal["USER", "CONTEXT", "MODEL"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CivilDate(_Strict):
    year: int
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)


class CivilTime(_Strict):
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)


class DurationValue(_Strict):
    magnitude: float
    unit: DurationUnit


class EnumSelection(_Strict):
    """Index + symbol into the canonical schema's enum-value list."""

    index: int = Field(ge=0)
    symbol: str


class RelativeDate(_Strict):
    relation: DateRelation
    weekday: Weekday | None = None
    offset: DurationValue | None = None


class RelativeDatetime(_Strict):
    relation: DateRelation
    weekday: Weekday | None = None
    daypart: Daypart | None = None
    time: CivilTime | None = None
    offset: DurationValue | None = None


class TextRef(_Strict):
    text: str


# --- Adjacently-tagged semantic values ------------------------------------


class StringValue(_Strict):
    semantic_type: Literal["STRING"]
    value: str


class BooleanValue(_Strict):
    semantic_type: Literal["BOOLEAN"]
    value: bool


class IntegerValue(_Strict):
    semantic_type: Literal["INTEGER"]
    value: int


class FloatValue(_Strict):
    semantic_type: Literal["FLOAT"]
    value: float


class EnumValue(_Strict):
    semantic_type: Literal["ENUM"]
    value: EnumSelection


class AbsoluteDateValue(_Strict):
    semantic_type: Literal["ABSOLUTE_DATE"]
    value: CivilDate


class RelativeDateValue(_Strict):
    semantic_type: Literal["RELATIVE_DATE"]
    value: RelativeDate


class AbsoluteDatetimeValue(_Strict):
    """RFC 3339 timestamp, e.g. ``2026-08-19T15:00:00+02:00``."""

    semantic_type: Literal["ABSOLUTE_DATETIME"]
    value: str


class RelativeDatetimeValue(_Strict):
    semantic_type: Literal["RELATIVE_DATETIME"]
    value: RelativeDatetime


class TimeOfDayValue(_Strict):
    semantic_type: Literal["TIME_OF_DAY"]
    value: CivilTime


class DaypartValue(_Strict):
    semantic_type: Literal["DAYPART"]
    value: Daypart


class DurationSemanticValue(_Strict):
    semantic_type: Literal["DURATION"]
    value: DurationValue


class PersonRefValue(_Strict):
    semantic_type: Literal["PERSON_REF"]
    value: TextRef


class LocationValue(_Strict):
    semantic_type: Literal["LOCATION"]
    value: TextRef


class ListValue(_Strict):
    """A `LIST<T>` value.

    Two encodings exist for the same thing and both are accepted here:

    - **dataset labels** carry the element type as a sibling field on the
      argument (`item_type`) and the value as a flat list, mirroring the
      source corpora's `ARRAY` shape;
    - the **runtime IR** nests `{item_type, items, element_provenance}` inside
      `value`, so a serialized call is self-describing without its schema.

    Keeping the label form flat avoids rewriting every corpus; keeping the IR
    form nested keeps a single argument interpretable on its own.
    """

    semantic_type: Literal["LIST"]
    value: list[str | int | float | bool] | dict[str, Any]


SemanticValue = Annotated[
    StringValue | BooleanValue | IntegerValue | FloatValue | EnumValue | AbsoluteDateValue | RelativeDateValue | AbsoluteDatetimeValue | RelativeDatetimeValue | TimeOfDayValue | DaypartValue | DurationSemanticValue | PersonRefValue | LocationValue | ListValue,
    Field(discriminator="semantic_type"),
]

SemanticType = Literal[
    "STRING",
    "BOOLEAN",
    "INTEGER",
    "FLOAT",
    "ENUM",
    "ABSOLUTE_DATE",
    "RELATIVE_DATE",
    "ABSOLUTE_DATETIME",
    "RELATIVE_DATETIME",
    "TIME_OF_DAY",
    "DAYPART",
    "DURATION",
    "PERSON_REF",
    "LOCATION",
    "LIST",
]

_SEMANTIC_VALUE_ADAPTER: TypeAdapter[Any] = TypeAdapter(SemanticValue)


def validate_semantic_value(semantic_type: str, value: Any) -> Any:
    """Validate an adjacently-tagged (semantic_type, value) pair.

    Returns the coerced typed value payload (e.g. a `DurationValue`).
    Raises `pydantic.ValidationError` on mismatch. Validation is strict so
    e.g. `"yes"` is not silently coerced to a BOOLEAN `True`.
    """
    validated = _SEMANTIC_VALUE_ADAPTER.validate_python(
        {"semantic_type": semantic_type, "value": value}, strict=True
    )
    return validated.value


# --- IR structure ----------------------------------------------------------


class TokenSpan(_Strict):
    start: int = Field(ge=0)
    end: int = Field(ge=0, description="Exclusive.")


class Provenance(_Strict):
    source: ProvenanceSource
    token_span: TokenSpan | None = None


class ToolSelection(_Strict):
    registry_id: str
    candidate_index: int = Field(ge=0, le=255)
    confidence: float = Field(ge=0.0, le=1.0)


class UnresolvedField(_Strict):
    parameter: str
    reason: UnresolvedReason
    confidence: float = Field(ge=0.0, le=1.0)


class ArgumentBinding(BaseModel):
    """One bound argument: binding metadata + flattened semantic value."""

    model_config = ConfigDict(extra="forbid")

    parameter: str
    confidence: float = Field(ge=0.0, le=1.0)
    semantic_type: SemanticType
    value: Any
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _check_semantic_value(self) -> ArgumentBinding:
        self.value = validate_semantic_value(self.semantic_type, self.value)
        return self

    def semantic_value(self) -> Any:
        """The (semantic_type, value) pair as a typed SemanticValue model."""
        return _SEMANTIC_VALUE_ADAPTER.validate_python(
            {"semantic_type": self.semantic_type, "value": self.value}
        )


class ActionIr(BaseModel):
    """The typed action program (contracts/action-ir/v1/action-ir.schema.json)."""

    model_config = ConfigDict(extra="forbid")

    action: ActionState
    action_confidence: float = Field(ge=0.0, le=1.0)
    ir_version: int = Field(default=1, ge=0)
    tool: ToolSelection | None = None
    arguments: list[ArgumentBinding] = Field(default_factory=list)
    unresolved: list[UnresolvedField] = Field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        """JSON-safe dict with `None` optionals omitted (matches Rust serde)."""
        return self.model_dump(mode="json", exclude_none=True)


# --- CompileRequest --------------------------------------------------------


class ContextEntity(_Strict):
    id: str
    kind: str
    display: str


class RequestContext(_Strict):
    entities: list[ContextEntity] = Field(default_factory=list)


class CompileRequest(BaseModel):
    """contracts/action-ir/v1/compile-request.schema.json."""

    model_config = ConfigDict(extra="forbid")

    utterance: str
    locale: str | None = None
    timezone: str | None = None
    now: str | None = None
    candidates: list[str] | None = None
    context: RequestContext | None = None

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
