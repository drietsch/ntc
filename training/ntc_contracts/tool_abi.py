"""Canonical Tool ABI (contracts/tool-abi/v1/tool-abi.schema.json).

Mirrors `ntc_core::schema::CanonicalTool`. Canonicalization itself has
exactly one implementation — the Rust `ntc schemac` CLI; the Python side
only validates and transports already-canonical records.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ParamType = Literal[
    "TEXT",
    "INTEGER",
    "FLOAT",
    "BOOLEAN",
    "ENUM",
    "DATE",
    "DATETIME",
    "DURATION",
    "PERSON",
    "LOCATION",
]

RiskClass = Literal["READ", "WRITE", "DESTRUCTIVE"]


class CanonicalArg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    json_type: Literal["string", "integer", "number", "boolean"]
    param_type: ParamType
    required: bool
    description: str | None = None
    semantic_type: str | None = None
    enum_values: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enum_values_iff_enum(self) -> CanonicalArg:
        if self.param_type == "ENUM" and not self.enum_values:
            raise ValueError(f"arg `{self.name}`: ENUM param_type requires enum_values")
        return self


class CanonicalTool(BaseModel):
    """The normalized Tool ABI record (spec §40)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    abi_version: int = Field(ge=0)
    description: str
    args: list[CanonicalArg] = Field(default_factory=list)
    risk: RiskClass = "WRITE"

    @model_validator(mode="after")
    def _unique_arg_names(self) -> CanonicalTool:
        names = [a.name for a in self.args]
        if len(names) != len(set(names)):
            raise ValueError(f"tool `{self.id}`: duplicate arg names")
        return self

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
