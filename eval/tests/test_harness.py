"""End-to-end harness tests over canned JSONL files."""

import json

import pytest
from eval.harness import evaluate, load_jsonl, pair_by_id, run_harness

GOLD_LINES = [
    {
        "id": "ex-1",
        "lang": "de",
        "utterance": "mach einen Zahnarzttermin",
        "gold": {
            "action": "CALL",
            "tool": "calendar.create",
            "arguments": [
                {"parameter": "title", "semantic_type": "STRING", "value": "Zahnarzttermin"}
            ],
            "unresolved": [],
        },
    },
    {
        "id": "ex-2",
        "gold": {
            "action": "ASK",
            "tool": None,
            "arguments": [],
            "unresolved": [{"parameter": "start", "reason": "MISSING"}],
        },
    },
    {"id": "ex-3", "gold": {"action": "NO_CALL", "tool": None, "arguments": [], "unresolved": []}},
]

PRED_LINES = [
    {  # exact match
        "id": "ex-1",
        "outcome": "CALL",
        "ir": {
            "action": "CALL",
            "action_confidence": 0.97,
            "ir_version": 1,
            "tool": {"registry_id": "calendar.create", "candidate_index": 0, "confidence": 0.99},
            "arguments": [
                {
                    "parameter": "title",
                    "confidence": 0.94,
                    "semantic_type": "STRING",
                    "value": "Zahnarzttermin",
                }
            ],
        },
        "call": {"name": "calendar.create", "arguments": {"title": "Zahnarzttermin"}},
    },
    {  # should have asked -> E01 + E11
        "id": "ex-2",
        "outcome": "CALL",
        "ir": {
            "action": "CALL",
            "action_confidence": 0.6,
            "ir_version": 1,
            "tool": {"registry_id": "calendar.create", "candidate_index": 0, "confidence": 0.5},
            "arguments": [],
        },
        "call": {"name": "calendar.create", "arguments": {}},
    },
    {  # correct NO_CALL
        "id": "ex-3",
        "outcome": "NO_CALL",
        "ir": {"action": "NO_CALL", "action_confidence": 0.9, "ir_version": 1},
    },
]


def write_jsonl(path, lines):
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


@pytest.fixture
def files(tmp_path):
    pred = tmp_path / "preds.jsonl"
    gold = tmp_path / "gold.jsonl"
    write_jsonl(pred, PRED_LINES)
    write_jsonl(gold, GOLD_LINES)
    return pred, gold


def test_run_harness_report(files):
    pred, gold = files
    report = run_harness(pred, gold)
    assert report["n_examples"] == 3
    assert report["exact_match"] == pytest.approx(2 / 3)
    assert report["tool_selection_accuracy"] == 1.0
    assert report["required_arg_accuracy"] == 1.0
    assert report["hallucinated_arg_rate"] == 0.0
    assert report["no_call_precision"] == 1.0
    assert report["no_call_recall"] == 1.0
    assert report["ask_accuracy"] == 0.0
    assert report["error_counts"] == {"E01": 1, "E11": 1}
    tags = {e["id"]: e["tags"] for e in report["per_example"]}
    assert tags == {"ex-1": [], "ex-2": ["E01", "E11"], "ex-3": []}


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n')
    assert [r["id"] for r in load_jsonl(path)] == ["a", "b"]


def test_pair_by_id_validates_coverage():
    with pytest.raises(ValueError, match="no gold record"):
        pair_by_id([{"id": "missing", "outcome": "NO_CALL"}], GOLD_LINES)
    preds = [dict(p) for p in PRED_LINES[:2]]
    with pytest.raises(ValueError, match="gold ids without predictions"):
        pair_by_id(preds, GOLD_LINES)


def test_evaluate_empty_set_is_all_undefined():
    report = evaluate([])
    assert report["n_examples"] == 0
    assert report["exact_match"] is None
    assert report["tool_selection_accuracy"] is None
    assert report["error_counts"] == {}
