"""Pydantic v2 mirrors of the NTC-Web contracts (contracts/*).

The Rust crates are the source of truth; these models exist so the Python
training/data pipeline can construct, validate, and serialize contract
objects that round-trip against the generated JSON Schemas.
"""

from ntc_contracts.ir import (
    ActionIr,
    ActionState,
    ArgumentBinding,
    CivilDate,
    CivilTime,
    CompileRequest,
    DateRelation,
    Daypart,
    DurationUnit,
    DurationValue,
    Provenance,
    SemanticValue,
    TokenSpan,
    ToolSelection,
    UnresolvedField,
    UnresolvedReason,
    Weekday,
    validate_semantic_value,
)
from ntc_contracts.tool_abi import CanonicalArg, CanonicalTool, ParamType, RiskClass

__all__ = [
    "ActionIr",
    "ActionState",
    "ArgumentBinding",
    "CanonicalArg",
    "CanonicalTool",
    "CivilDate",
    "CivilTime",
    "CompileRequest",
    "DateRelation",
    "Daypart",
    "DurationUnit",
    "DurationValue",
    "ParamType",
    "Provenance",
    "RiskClass",
    "SemanticValue",
    "TokenSpan",
    "ToolSelection",
    "UnresolvedField",
    "UnresolvedReason",
    "Weekday",
    "validate_semantic_value",
]
