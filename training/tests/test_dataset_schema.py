"""Validator tests for the dataset example schema (valid + each violation)."""

import pytest
from pydantic import ValidationError

from datasets.schema import DatasetExample, RawToolSchema, ToolFamily

CALENDAR_TOOL = {
    "name": "calendar.create",
    "description": "Create a calendar event",
    "parameters": {
        "title": {"type": "string", "required": True},
        "start": {"type": "string", "format": "date-time", "required": True},
    },
}

UTTERANCE = "mach einen Zahnarzttermin morgen Nachmittag, eine Stunde"


def valid_example(**overrides) -> dict:
    ex = {
        "id": "ex-0001",
        "lang": "de",
        "utterance": UTTERANCE,
        "candidates": [CALENDAR_TOOL],
        "gold": {
            "action": "CALL",
            "tool": "calendar.create",
            "arguments": [
                {
                    "parameter": "title",
                    "semantic_type": "STRING",
                    "value": "Zahnarzttermin",
                    "char_span": {"start": 11, "end": 25},
                    "surface": "Zahnarzttermin",
                },
                {
                    "parameter": "start",
                    "semantic_type": "RELATIVE_DATETIME",
                    "value": {"relation": "TOMORROW", "daypart": "AFTERNOON"},
                    "char_span": {"start": 26, "end": 43},
                },
            ],
        },
        "split": "train",
        "tags": ["seed-batch"],
    }
    ex.update(overrides)
    return ex


def test_valid_call_example():
    ex = DatasetExample.model_validate(valid_example())
    assert ex.gold.action == "CALL"
    span = ex.gold.arguments[0].char_span
    assert ex.utterance[span.start : span.end] == "Zahnarzttermin"


def test_valid_ask_example():
    gold = {
        "action": "ASK",
        "tool": None,
        "unresolved": [{"parameter": "start", "reason": "MISSING"}],
    }
    ex = DatasetExample.model_validate(valid_example(gold=gold))
    assert ex.gold.unresolved[0].reason == "MISSING"


def test_valid_no_call_example():
    ex = DatasetExample.model_validate(
        valid_example(gold={"action": "NO_CALL"}, utterance="wie geht es dir?")
    )
    assert ex.gold.action == "NO_CALL"


def test_span_out_of_bounds_rejected():
    ex = valid_example()
    ex["gold"]["arguments"][0]["char_span"] = {"start": 11, "end": 999}
    with pytest.raises(ValidationError, match="exceeds utterance length"):
        DatasetExample.model_validate(ex)


def test_empty_span_rejected():
    ex = valid_example()
    ex["gold"]["arguments"][0]["char_span"] = {"start": 11, "end": 11}
    with pytest.raises(ValidationError, match="empty"):
        DatasetExample.model_validate(ex)


def test_surface_mismatch_rejected():
    ex = valid_example()
    ex["gold"]["arguments"][0]["surface"] = "Friseurtermin"
    with pytest.raises(ValidationError, match="span text"):
        DatasetExample.model_validate(ex)


def test_ask_without_unresolved_rejected():
    ex = valid_example(gold={"action": "ASK", "unresolved": []})
    with pytest.raises(ValidationError, match="ASK requires at least one unresolved"):
        DatasetExample.model_validate(ex)


def test_unresolved_without_ask_rejected():
    ex = valid_example()
    ex["gold"]["unresolved"] = [{"parameter": "start", "reason": "AMBIGUOUS"}]
    with pytest.raises(ValidationError, match="unresolved entries require action ASK"):
        DatasetExample.model_validate(ex)


def test_call_without_tool_rejected():
    ex = valid_example()
    ex["gold"]["tool"] = None
    with pytest.raises(ValidationError, match="CALL requires a gold tool"):
        DatasetExample.model_validate(ex)


def test_call_with_unknown_tool_rejected():
    ex = valid_example()
    ex["gold"]["tool"] = "email.send"
    with pytest.raises(ValidationError, match="not among the candidates"):
        DatasetExample.model_validate(ex)


def test_bad_semantic_value_rejected():
    ex = valid_example()
    ex["gold"]["arguments"][1]["value"] = {"relation": "SOMEDAY"}
    with pytest.raises(ValidationError):
        DatasetExample.model_validate(ex)


def test_tool_family_rejects_duplicate_names():
    with pytest.raises(ValidationError, match="duplicate tool names"):
        ToolFamily(
            id="fam-1",
            domain="calendar",
            tools=[RawToolSchema(name="a"), RawToolSchema(name="a")],
        )
