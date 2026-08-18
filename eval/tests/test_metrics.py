"""Metric + error-taxonomy tests over small canned prediction/gold sets."""

import pytest
from eval.metrics import (
    ask_accuracy,
    hallucinated_arg_rate,
    no_call_precision_recall,
    required_arg_accuracy,
    tag_errors,
    tool_selection_accuracy,
    values_equal,
)


def p_arg(parameter, semantic_type, value):
    return {"parameter": parameter, "confidence": 0.9, "semantic_type": semantic_type, "value": value}


def g_arg(parameter, semantic_type, value):
    return {"parameter": parameter, "semantic_type": semantic_type, "value": value}


def pred_call(tool, args=()):
    return {
        "outcome": "CALL",
        "ir": {
            "action": "CALL",
            "action_confidence": 0.95,
            "ir_version": 1,
            "tool": {"registry_id": tool, "candidate_index": 0, "confidence": 0.9},
            "arguments": list(args),
        },
        "call": {"name": tool, "arguments": {}},
    }


def pred_ask(params):
    unresolved = [{"parameter": p, "reason": "MISSING", "confidence": 0.8} for p in params]
    return {
        "outcome": "ASK",
        "ir": {"action": "ASK", "action_confidence": 0.9, "ir_version": 1, "unresolved": unresolved},
        "unresolved": unresolved,
    }


def pred_no_call():
    return {"outcome": "NO_CALL", "ir": {"action": "NO_CALL", "action_confidence": 0.9, "ir_version": 1}}


def gold_call(tool, args=()):
    return {"action": "CALL", "tool": tool, "arguments": list(args), "unresolved": []}


def gold_ask(params):
    return {
        "action": "ASK",
        "tool": None,
        "arguments": [],
        "unresolved": [{"parameter": p, "reason": "MISSING"} for p in params],
    }


def gold_no_call():
    return {"action": "NO_CALL", "tool": None, "arguments": [], "unresolved": []}


TITLE = g_arg("title", "STRING", "Zahnarzttermin")
DURATION = g_arg("duration", "DURATION", {"magnitude": 1, "unit": "HOUR"})


class TestToolSelectionAccuracy:
    def test_counts_only_gold_call_examples(self):
        pairs = [
            (pred_call("calendar.create"), gold_call("calendar.create")),
            (pred_call("email.send"), gold_call("calendar.create")),  # wrong tool
            (pred_no_call(), gold_no_call()),  # ignored
        ]
        assert tool_selection_accuracy(pairs) == 0.5

    def test_non_call_prediction_counts_as_miss(self):
        pairs = [(pred_ask(["start"]), gold_call("calendar.create"))]
        assert tool_selection_accuracy(pairs) == 0.0

    def test_undefined_without_gold_calls(self):
        assert tool_selection_accuracy([(pred_no_call(), gold_no_call())]) is None


class TestRequiredArgAccuracy:
    def test_per_argument_scoring(self):
        pred = pred_call(
            "calendar.create",
            [
                p_arg("title", "STRING", "Zahnarzttermin"),  # correct
                p_arg("duration", "DURATION", {"magnitude": 2, "unit": "HOUR"}),  # wrong value
            ],
        )
        gold = gold_call("calendar.create", [TITLE, DURATION])
        assert required_arg_accuracy([(pred, gold)]) == 0.5

    def test_wrong_tool_scores_zero(self):
        pred = pred_call("email.send", [p_arg("title", "STRING", "Zahnarzttermin")])
        gold = gold_call("calendar.create", [TITLE])
        assert required_arg_accuracy([(pred, gold)]) == 0.0

    def test_semantic_type_mismatch_is_wrong(self):
        pred = pred_call("t", [p_arg("title", "LOCATION", {"text": "Zahnarzttermin"})])
        gold = gold_call("t", [TITLE])
        assert required_arg_accuracy([(pred, gold)]) == 0.0

    def test_undefined_without_gold_args(self):
        assert required_arg_accuracy([(pred_no_call(), gold_no_call())]) is None


class TestHallucinatedArgRate:
    def test_extra_arg_counts(self):
        pred = pred_call(
            "calendar.create",
            [p_arg("title", "STRING", "Zahnarzttermin"), p_arg("location", "LOCATION", {"text": "Berlin"})],
        )
        gold = gold_call("calendar.create", [TITLE])
        assert hallucinated_arg_rate([(pred, gold)]) == 0.5

    def test_call_on_gold_no_call_is_fully_hallucinated(self):
        pred = pred_call("calendar.create", [p_arg("title", "STRING", "x")])
        assert hallucinated_arg_rate([(pred, gold_no_call())]) == 1.0

    def test_undefined_without_predicted_args(self):
        assert hallucinated_arg_rate([(pred_no_call(), gold_no_call())]) is None


class TestNoCallPrecisionRecall:
    def test_mixed_set(self):
        pairs = [
            (pred_no_call(), gold_no_call()),  # tp
            (pred_no_call(), gold_call("t")),  # fp
            (pred_call("t"), gold_no_call()),  # fn
            (pred_call("t"), gold_call("t")),  # tn
        ]
        result = no_call_precision_recall(pairs)
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5

    def test_undefined_edges(self):
        result = no_call_precision_recall([(pred_call("t"), gold_call("t"))])
        assert result["precision"] is None
        assert result["recall"] is None
        assert result["f1"] is None


class TestAskAccuracy:
    def test_ask_covering_gold_params_is_correct(self):
        assert ask_accuracy([(pred_ask(["start", "title"]), gold_ask(["start"]))]) == 1.0

    def test_ask_missing_gold_param_is_wrong(self):
        assert ask_accuracy([(pred_ask(["title"]), gold_ask(["start"]))]) == 0.0

    def test_non_ask_prediction_is_wrong(self):
        assert ask_accuracy([(pred_call("t"), gold_ask(["start"]))]) == 0.0

    def test_undefined_without_gold_asks(self):
        assert ask_accuracy([(pred_call("t"), gold_call("t"))]) is None


class TestErrorTaxonomy:
    def test_exact_match_has_no_tags(self):
        pred = pred_call("calendar.create", [p_arg("title", "STRING", "Zahnarzttermin")])
        gold = gold_call("calendar.create", [TITLE])
        assert tag_errors(pred, gold) == []

    def test_e01_wrong_action(self):
        assert tag_errors(pred_no_call(), gold_call("t")) == ["E01"]

    def test_e02_wrong_tool(self):
        assert tag_errors(pred_call("email.send"), gold_call("calendar.create")) == ["E02"]

    def test_e03_missing_required_arg(self):
        pred = pred_call("t")
        gold = gold_call("t", [TITLE])
        assert tag_errors(pred, gold) == ["E03"]

    def test_e04_hallucinated_arg(self):
        pred = pred_call("t", [p_arg("location", "LOCATION", {"text": "Berlin"})])
        gold = gold_call("t")
        assert tag_errors(pred, gold) == ["E04"]

    def test_e05_wrong_value(self):
        pred = pred_call("t", [p_arg("title", "STRING", "Friseurtermin")])
        gold = gold_call("t", [TITLE])
        assert tag_errors(pred, gold) == ["E05"]

    def test_e11_should_have_asked(self):
        assert tag_errors(pred_call("t"), gold_ask(["start"])) == ["E01", "E11"]

    def test_e12_asked_unnecessarily(self):
        assert tag_errors(pred_ask(["start"]), gold_call("t")) == ["E01", "E12"]

    def test_tags_can_cooccur(self):
        pred = pred_call(
            "t",
            [p_arg("title", "STRING", "wrong"), p_arg("location", "LOCATION", {"text": "x"})],
        )
        gold = gold_call("t", [TITLE, DURATION])
        assert tag_errors(pred, gold) == ["E03", "E04", "E05"]


class TestValuesEqual:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (1.0, 1.0000000001, True),
            (1, 1.0, True),
            (True, 1, False),  # bool is not the number 1 here
            (True, True, True),
            ({"magnitude": 1, "unit": "HOUR"}, {"magnitude": 1.0, "unit": "HOUR"}, True),
            ({"magnitude": 1, "unit": "HOUR"}, {"magnitude": 1, "unit": "DAY"}, False),
            (["a", 1], ["a", 1], True),
            (["a"], ["a", "b"], False),
        ],
    )
    def test_cases(self, a, b, expected):
        assert values_equal(a, b) is expected
