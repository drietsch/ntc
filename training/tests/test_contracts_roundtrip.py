"""Round-trip the pydantic contract models against the generated JSON Schemas."""

import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from ntc_contracts import (
    ActionIr,
    ArgumentBinding,
    CanonicalArg,
    CanonicalTool,
    CompileRequest,
    ToolSelection,
    UnresolvedField,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = REPO_ROOT / "contracts"


def load_schema(rel: str) -> dict:
    return json.loads((CONTRACTS / rel).read_text())


def validator_for(rel: str) -> jsonschema.Draft7Validator:
    schema = load_schema(rel)
    jsonschema.Draft7Validator.check_schema(schema)
    return jsonschema.Draft7Validator(schema)


def zahnarzt_ir() -> ActionIr:
    """The spec's Zahnarzttermin example: 'mach einen Zahnarzttermin morgen
    Nachmittag, eine Stunde' -> calendar.create."""
    return ActionIr(
        action="CALL",
        action_confidence=0.97,
        ir_version=1,
        tool=ToolSelection(registry_id="calendar.create", candidate_index=0, confidence=0.99),
        arguments=[
            ArgumentBinding(
                parameter="title",
                confidence=0.94,
                semantic_type="STRING",
                value="Zahnarzttermin",
                provenance={"source": "USER", "token_span": {"start": 3, "end": 4}},
            ),
            ArgumentBinding(
                parameter="start",
                confidence=0.92,
                semantic_type="RELATIVE_DATETIME",
                value={"relation": "TOMORROW", "daypart": "AFTERNOON"},
            ),
            ArgumentBinding(
                parameter="duration",
                confidence=0.90,
                semantic_type="DURATION",
                value={"magnitude": 1, "unit": "HOUR"},
            ),
        ],
    )


class TestActionIr:
    def test_zahnarzt_ir_validates_against_schema(self):
        v = validator_for("action-ir/v1/action-ir.schema.json")
        wire = zahnarzt_ir().to_wire()
        v.validate(wire)

    def test_wire_round_trips_through_pydantic(self):
        ir = zahnarzt_ir()
        wire = ir.to_wire()
        reparsed = ActionIr.model_validate(wire)
        assert reparsed == ir
        assert reparsed.to_wire() == wire

    def test_ask_ir_validates(self):
        v = validator_for("action-ir/v1/action-ir.schema.json")
        ir = ActionIr(
            action="ASK",
            action_confidence=0.81,
            unresolved=[UnresolvedField(parameter="start", reason="MISSING", confidence=0.7)],
        )
        v.validate(ir.to_wire())

    def test_all_semantic_types_validate(self):
        v = validator_for("action-ir/v1/action-ir.schema.json")
        cases = [
            ("STRING", "hello"),
            ("BOOLEAN", True),
            ("INTEGER", 42),
            ("FLOAT", 2.5),
            ("ENUM", {"index": 1, "symbol": "normal"}),
            ("ABSOLUTE_DATE", {"year": 2026, "month": 8, "day": 19}),
            ("RELATIVE_DATE", {"relation": "NEXT", "weekday": "FRIDAY"}),
            ("ABSOLUTE_DATETIME", "2026-08-19T15:00:00+02:00"),
            ("RELATIVE_DATETIME", {"relation": "IN", "offset": {"magnitude": 3, "unit": "DAY"}}),
            ("TIME_OF_DAY", {"hour": 15, "minute": 30}),
            ("DAYPART", "EVENING"),
            ("DURATION", {"magnitude": 1, "unit": "HOUR"}),
            ("PERSON_REF", {"text": "Anna"}),
            ("LOCATION", {"text": "Berlin Hbf"}),
        ]
        for semantic_type, value in cases:
            ir = ActionIr(
                action="CALL",
                action_confidence=1.0,
                tool=ToolSelection(registry_id="t.x", candidate_index=0, confidence=1.0),
                arguments=[
                    ArgumentBinding(
                        parameter="p", confidence=1.0, semantic_type=semantic_type, value=value
                    )
                ],
            )
            v.validate(ir.to_wire())

    def test_mismatched_semantic_value_rejected(self):
        with pytest.raises(ValidationError):
            ArgumentBinding(
                parameter="p",
                confidence=1.0,
                semantic_type="DURATION",
                value={"magnitude": 1, "unit": "FORTNIGHT"},
            )
        with pytest.raises(ValidationError):
            ArgumentBinding(parameter="p", confidence=1.0, semantic_type="BOOLEAN", value="yes")


class TestCompileRequest:
    def test_round_trip_against_schema(self):
        v = validator_for("action-ir/v1/compile-request.schema.json")
        req = CompileRequest(
            utterance="mach einen Zahnarzttermin morgen Nachmittag, eine Stunde",
            locale="de-DE",
            timezone="Europe/Berlin",
            now="2026-08-18T12:00:00+02:00",
            candidates=["calendar.create", "email.send"],
            context={"entities": [{"id": "p1", "kind": "PERSON", "display": "Dr. Weber"}]},
        )
        wire = req.to_wire()
        v.validate(wire)
        assert CompileRequest.model_validate(wire) == req

    def test_minimal_request_validates(self):
        v = validator_for("action-ir/v1/compile-request.schema.json")
        v.validate(CompileRequest(utterance="hi").to_wire())


class TestCanonicalTool:
    def test_round_trip_against_schema(self):
        v = validator_for("tool-abi/v1/tool-abi.schema.json")
        tool = CanonicalTool(
            id="calendar.create",
            abi_version=1,
            description="create a calendar event",
            risk="WRITE",
            args=[
                CanonicalArg(
                    name="title",
                    json_type="string",
                    param_type="TEXT",
                    required=True,
                    description="event title",
                ),
                CanonicalArg(
                    name="start", json_type="string", param_type="DATETIME", required=True
                ),
                CanonicalArg(
                    name="duration_minutes",
                    json_type="integer",
                    param_type="INTEGER",
                    required=False,
                    semantic_type="DURATION",
                ),
                CanonicalArg(
                    name="priority",
                    json_type="string",
                    param_type="ENUM",
                    required=False,
                    enum_values=["low", "normal", "high"],
                ),
            ],
        )
        wire = tool.to_wire()
        v.validate(wire)
        assert CanonicalTool.model_validate(wire) == tool

    def test_enum_without_values_rejected(self):
        with pytest.raises(ValidationError):
            CanonicalArg(name="p", json_type="string", param_type="ENUM", required=False)

    def test_parses_rust_schemac_shape(self):
        # The exact `tool` object `ntc schemac` emits for
        # fixtures/schema-abi/001-calendar-create.json (None fields omitted).
        wire = {
            "id": "calendar.create",
            "abi_version": 1,
            "description": "Create a calendar event",
            "risk": "WRITE",
            "args": [
                {
                    "name": "title",
                    "param_type": "TEXT",
                    "json_type": "string",
                    "required": True,
                    "description": "Event title",
                },
                {
                    "name": "start",
                    "param_type": "DATETIME",
                    "json_type": "string",
                    "required": True,
                },
                {
                    "name": "duration_minutes",
                    "param_type": "INTEGER",
                    "json_type": "integer",
                    "required": False,
                    "semantic_type": "DURATION",
                },
                {
                    "name": "priority",
                    "param_type": "ENUM",
                    "json_type": "string",
                    "required": False,
                    "enum_values": ["low", "normal", "high"],
                },
            ],
        }
        tool = CanonicalTool.model_validate(wire)
        assert [a.param_type for a in tool.args] == ["TEXT", "DATETIME", "INTEGER", "ENUM"]
        validator_for("tool-abi/v1/tool-abi.schema.json").validate(tool.to_wire())
