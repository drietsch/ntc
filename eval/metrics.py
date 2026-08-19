"""Metrics over (predicted CompileOutcome JSON, gold label) pairs.

- Predictions are `CompileOutcome`-shaped dicts
  (contracts/action-ir/v1/compile-outcome.schema.json): `{"outcome": "CALL",
  "ir": {...}, "call": {...}}` etc.
- Gold labels are `GoldLabel`-shaped dicts (training/datasets/schema.py):
  `{"action", "tool", "arguments": [{"parameter", "semantic_type", "value",
  ...}], "unresolved": [{"parameter", "reason"}]}`.

Pure stdlib on purpose — the harness must not depend on the training stack.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

Pair = tuple[dict[str, Any], dict[str, Any]]  # (prediction, gold)

#: Error taxonomy. E01/E11/E12 describe the action decision (E11/E12 refine
#: E01 for the ASK boundary); E02–E05 describe CALL content and are only
#: emitted when both sides agree the example is a CALL.
ERROR_TAXONOMY = {
    "E01": "wrong action",
    "E02": "wrong tool",
    "E03": "missing required arg",
    "E04": "hallucinated arg",
    "E05": "wrong value",
    "E11": "should-have-asked (gold ASK, predicted CALL/NO_CALL)",
    "E12": "asked-unnecessarily (predicted ASK, gold CALL/NO_CALL)",
}


# --- accessors --------------------------------------------------------------


def pred_action(pred: dict[str, Any]) -> str:
    return pred["outcome"]


def pred_tool(pred: dict[str, Any]) -> str | None:
    tool = (pred.get("ir") or {}).get("tool")
    return tool.get("registry_id") if tool else None


def pred_arguments(pred: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """parameter -> {"semantic_type", "value"} from the predicted IR."""
    args = (pred.get("ir") or {}).get("arguments") or []
    return {a["parameter"]: a for a in args}


def pred_unresolved_params(pred: dict[str, Any]) -> set[str]:
    ir_unresolved = (pred.get("ir") or {}).get("unresolved") or []
    top_unresolved = pred.get("unresolved") or []
    return {u["parameter"] for u in [*ir_unresolved, *top_unresolved]}


def gold_arguments(gold: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {a["parameter"]: a for a in gold.get("arguments") or []}


def gold_unresolved_params(gold: dict[str, Any]) -> set[str]:
    return {u["parameter"] for u in gold.get("unresolved") or []}


def values_equal(a: Any, b: Any, rel_tol: float = 1e-6) -> bool:
    """Structural equality with float tolerance (bool checked before number)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def binding_correct(pred_arg: dict[str, Any] | None, gold_arg: dict[str, Any]) -> bool:
    if pred_arg is None:
        return False
    return pred_arg.get("semantic_type") == gold_arg.get("semantic_type") and values_equal(
        pred_arg.get("value"), gold_arg.get("value")
    )


# --- metrics ----------------------------------------------------------------


def tool_selection_accuracy(pairs: Sequence[Pair]) -> float | None:
    """Over gold-CALL examples: predicted CALL with the gold tool selected.

    `None` when the metric is undefined (no gold CALL examples).
    """
    total = correct = 0
    for pred, gold in pairs:
        if gold["action"] != "CALL":
            continue
        total += 1
        if pred_action(pred) == "CALL" and pred_tool(pred) == gold.get("tool"):
            correct += 1
    return correct / total if total else None


def required_arg_accuracy(pairs: Sequence[Pair]) -> float | None:
    """Over every gold argument of gold-CALL examples: bound correctly in the
    prediction (right action, right tool, matching semantic_type + value)."""
    total = correct = 0
    for pred, gold in pairs:
        if gold["action"] != "CALL":
            continue
        on_tool = pred_action(pred) == "CALL" and pred_tool(pred) == gold.get("tool")
        p_args = pred_arguments(pred)
        for name, gold_arg in gold_arguments(gold).items():
            total += 1
            if on_tool and binding_correct(p_args.get(name), gold_arg):
                correct += 1
    return correct / total if total else None


def hallucinated_arg_rate(pairs: Sequence[Pair]) -> float | None:
    """Share of predicted argument bindings that gold does not license.

    A predicted binding is hallucinated when the gold label has no argument of
    that name — including every binding of a predicted CALL whose gold action
    is not CALL or whose gold tool differs.
    """
    total = hallucinated = 0
    for pred, gold in pairs:
        if pred_action(pred) != "CALL":
            continue
        licensed = (
            set(gold_arguments(gold))
            if gold["action"] == "CALL" and pred_tool(pred) == gold.get("tool")
            else set()
        )
        for name in pred_arguments(pred):
            total += 1
            if name not in licensed:
                hallucinated += 1
    return hallucinated / total if total else None


def no_call_precision_recall(pairs: Sequence[Pair]) -> dict[str, float | None]:
    """Precision/recall/F1 of the NO_CALL decision (NO_CALL = positive)."""
    tp = fp = fn = 0
    for pred, gold in pairs:
        p = pred_action(pred) == "NO_CALL"
        g = gold["action"] == "NO_CALL"
        tp += p and g
        fp += p and not g
        fn += g and not p
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def ask_accuracy(pairs: Sequence[Pair]) -> float | None:
    """Over gold-ASK examples: predicted ASK covering every gold unresolved
    parameter (the model may surface additional ones)."""
    total = correct = 0
    for pred, gold in pairs:
        if gold["action"] != "ASK":
            continue
        total += 1
        if pred_action(pred) == "ASK" and gold_unresolved_params(gold) <= pred_unresolved_params(
            pred
        ):
            correct += 1
    return correct / total if total else None


# --- error taxonomy ---------------------------------------------------------


def delegate_accuracy(pairs: Sequence[Pair]) -> dict[str, float | None]:
    """How well the router recognizes work that belongs to a full LLM agent.

    Precision: of predicted DELEGATE, how many were gold DELEGATE.
    Recall: of gold DELEGATE, how many were predicted DELEGATE.
    """
    tp = sum(1 for p, g in pairs if pred_action(p) == "DELEGATE" and g["action"] == "DELEGATE")
    fp = sum(1 for p, g in pairs if pred_action(p) == "DELEGATE" and g["action"] != "DELEGATE")
    fn = sum(1 for p, g in pairs if pred_action(p) != "DELEGATE" and g["action"] == "DELEGATE")
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else (0.0 if (precision is not None and recall is not None) else None)
    )
    return {"delegate_precision": precision, "delegate_recall": recall, "delegate_f1": f1}


def tag_errors(pred: dict[str, Any], gold: dict[str, Any]) -> list[str]:
    """Error tags for one (prediction, gold) pair. Empty list = correct.

    Tags can co-occur; E11/E12 refine E01 at the ASK boundary.
    """
    tags: set[str] = set()
    p_action, g_action = pred_action(pred), gold["action"]

    if p_action != g_action:
        tags.add("E01")
    if g_action == "ASK" and p_action != "ASK":
        tags.add("E11")
    if p_action == "ASK" and g_action != "ASK":
        tags.add("E12")

    if p_action == "CALL" and g_action == "CALL":
        if pred_tool(pred) != gold.get("tool"):
            tags.add("E02")
        else:
            p_args = pred_arguments(pred)
            g_args = gold_arguments(gold)
            for name, gold_arg in g_args.items():
                if name not in p_args:
                    tags.add("E03")
                elif not binding_correct(p_args[name], gold_arg):
                    tags.add("E05")
            if set(p_args) - set(g_args):
                tags.add("E04")

    return sorted(tags)
