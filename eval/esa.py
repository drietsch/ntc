"""Executable semantic accuracy — spec §60's primary metric.

The goal is to call MCP servers **without an LLM**, so the question is not
"did the router classify the action correctly" but:

    of the requests that should produce a tool call, how many produced a call
    that would actually do the right thing?

That is a conjunction, and reporting its parts separately flatters the model:
right tool AND every required argument present AND every value correct AND
nothing invented. This computes the conjunction, plus the funnel that shows
where the loss happens.

DELEGATE is counted as a **miss**, not a success. It is the escape hatch: an
utterance the compiler could not turn into a call. Coverage — the share of
call-worthy requests answered with an executable call at all — is reported
alongside, because a system that delegates everything is trivially "precise".

Run (from the repo root):
    python3 eval/esa.py --pred <batch-infer output> --gold training/data/studio/dev.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def row_key(row: dict, index: int) -> str:
    """A collision-proof identity for one gold row.

    Row ids are content-derived and 31 of them cover 2-6 dev rows apiece: same
    utterance, *different* candidate slate. Keying predictions by id alone lets
    one slate's answer overwrite the others and then be scored against all of
    them — 69 of 278 call-worthy rows, worth 1.4 points. The line number is
    what actually distinguishes them, so it is part of the key.
    """
    return f'{row["id"]}#{index}'


def load_keyed(path: Path) -> list[dict]:
    """Gold rows carrying `_key`. Use this wherever predictions are joined."""
    rows = load(path)
    for i, r in enumerate(rows):
        r["_key"] = row_key(r, i)
    return rows


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def values_equal(got, want) -> bool:
    """Compare a compiled argument value with the gold value."""
    if isinstance(want, dict) and "symbol" in want:  # ENUM
        return got == want["symbol"] or got == want.get("index")
    if isinstance(want, bool) or isinstance(got, bool):
        return got is want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return abs(float(got) - float(want)) < 1e-6
    if isinstance(want, list):
        return isinstance(got, list) and len(got) == len(want) and all(
            values_equal(g, w) for g, w in zip(got, want)
        )
    if isinstance(got, str) and isinstance(want, str):
        return got.strip().lower() == want.strip().lower()
    return got == want


def needs_composed_value(arg: dict, tool: str = "", declared_defaults: set = frozenset()) -> bool:
    """True when nothing the compiler has can supply this argument's value.

    The architecture has no decoder: arguments come from a span pointing into
    the utterance, from the context frame (a linked item, the resolver), from
    a dedicated head for booleans and enums, from a value template the host
    declared (head codec v4 — the head picks the shape, the span fills its
    slots), or from a default the schema itself states. A value that fits none
    of those has to be *authored*, and no amount of training produces it.

    This function was once much more pessimistic, and the number it produced —
    an "88.8% architecture ceiling" on the Studio dev split — was mostly wrong.
    Of the 536 blocked rows in that corpus, 218 wanted `parentId: 1`, which the
    schema declares as its own default; 60 wanted a comma-joined list whose
    elements are verbatim in the utterance; and the 258 genuine PQL filters
    turned out to be five closed shapes with at most two slots. Almost none of
    it was composition. What is left here really is: a value with no shape the
    host declared and no source in the request.

    The Studio corpus has a concrete example — `pqlFilter`:

        "lista todos los archivos csv"  ->  filename LIKE "*.csv"

    That string does not occur in the utterance; it is a query-language
    expression composed from it. Rows like this are not compiler failures,
    they are the case DELEGATE exists for, and scoring them as missed calls
    understates the system against its own design.
    """
    if arg.get("source") in ("LINKED_ITEM", "RESOLVER", "CONTEXT"):
        return False
    if "char_span" in arg or arg.get("element_spans"):
        return False
    if arg.get("template"):  # a declared shape; slots come from the span
        return False
    if (tool, arg.get("parameter")) in declared_defaults:
        return False
    value = arg.get("value")
    if isinstance(value, bool):  # the boolean head needs no span
        return False
    if isinstance(value, dict) and "symbol" in value:  # enum pointer head
        return False
    return True


def declared_defaults(registry: Path | None) -> set[tuple[str, str]]:
    """(tool, argument) pairs whose schema states the value used when omitted.

    The canonical ABI carries these but never renders them, so the model is
    never told what the default is — it only decides that the argument is
    present and unsourced. Reading it off the schema is not invention, so
    these rows are not blocked.
    """
    if registry is None or not registry.exists():
        return set()
    tools = json.loads(registry.read_text())
    return {
        (t["name"], name)
        for t in tools
        for name, param in (t.get("parameters") or {}).items()
        if isinstance(param, dict) and "default" in param
    }


def ceiling(gold: list[dict], defaults: set[tuple[str, str]] = frozenset()) -> tuple[int, int]:
    """(rows the compiler has no mechanism to express, call-worthy rows)."""
    call_rows = [g for g in gold if g["gold"]["action"] == "CALL"]
    blocked = sum(
        1
        for g in call_rows
        if any(
            needs_composed_value(a, g["gold"]["tool"], defaults)
            for a in g["gold"]["arguments"]
        )
    )
    return blocked, len(call_rows)


def score(
    preds: dict[str, dict],
    gold: list[dict],
    defaults: set[tuple[str, str]] = frozenset(),
) -> dict:
    funnel = Counter()
    per_tool: dict[str, Counter] = {}
    failures: list[dict] = []

    callable_rows = [g for g in gold if g["gold"]["action"] == "CALL"]
    for g in callable_rows:
        key = g.get("_key", g["id"])
        p = preds.get(key) or {}
        want_tool = g["gold"]["tool"]
        stats = per_tool.setdefault(want_tool, Counter())
        stats["total"] += 1
        funnel["call_worthy"] += 1

        outcome = p.get("outcome")
        if outcome != "CALL":
            funnel[f"answered_{(outcome or 'ERROR').lower()}"] += 1
            failures.append({"id": g["id"], "key": key, "stage": "no_call_emitted",
                             "detail": outcome, "utterance": g["utterance"][:70]})
            continue
        funnel["emitted_a_call"] += 1

        got_tool = (p.get("call") or {}).get("name")
        if got_tool != want_tool:
            failures.append({"id": g["id"], "key": key, "stage": "wrong_tool",
                             "detail": f"{got_tool} != {want_tool}",
                             "utterance": g["utterance"][:70]})
            funnel["wrong_tool_called"] += 1  # also executed, also silent
            continue
        funnel["right_tool"] += 1
        stats["right_tool"] += 1

        got_args = (p.get("call") or {}).get("arguments") or {}
        want_args = {a["parameter"]: a["value"] for a in g["gold"]["arguments"]}

        missing = [k for k in want_args if k not in got_args]
        invented = [k for k in got_args if k not in want_args]
        wrong = [k for k in want_args if k in got_args and not values_equal(got_args[k], want_args[k])]

        if not missing and not invented and not wrong:
            funnel["executable"] += 1
            stats["executable"] += 1
        else:
            # A call that is wrong but well-formed is the worst outcome this
            # system can produce. `search_data_objects(pqlFilter="finDerechos")`
            # does not raise — it runs, returns the wrong rows, and reports
            # success. Declining is recoverable; executing the wrong thing is
            # not, so the two must never be pooled into one failure count.
            funnel["wrong_call_executed"] += 1
            failures.append({
                "id": g["id"], "key": key, "stage": "arguments",
                "detail": {"missing": missing, "invented": invented, "wrong": wrong},
                "utterance": g["utterance"][:70],
            })
            if not missing and not wrong:
                funnel["only_extra_args"] += 1

    n = funnel["call_worthy"] or 1
    blocked, call_n = ceiling(gold, defaults)
    declined = sum(v for k, v in funnel.items() if k.startswith("answered_"))
    return {
        "call_worthy_requests": funnel["call_worthy"],
        "executable_semantic_accuracy": round(funnel["executable"] / n, 4),
        "safety": {
            # Of the requests it got wrong, how many did it get wrong *loudly*?
            "wrong_call_executed": funnel["wrong_call_executed"] + funnel["wrong_tool_called"],
            "declined_instead": declined,
            "wrong_call_rate": round(
                (funnel["wrong_call_executed"] + funnel["wrong_tool_called"]) / n, 4
            ),
        },
        "ceiling": {
            "unreachable_rows": blocked,
            "ceiling": round((call_n - blocked) / max(1, call_n), 4),
        },
        "funnel": {
            "emitted a call": round(funnel["emitted_a_call"] / n, 3),
            "...with the right tool": round(funnel["right_tool"] / n, 3),
            "...and every argument correct": round(funnel["executable"] / n, 3),
        },
        "answered_instead": {
            k.replace("answered_", ""): v for k, v in funnel.items() if k.startswith("answered_")
        },
        "per_tool": {
            t: {"n": c["total"], "right_tool": c["right_tool"], "executable": c["executable"]}
            for t, c in sorted(per_tool.items(), key=lambda kv: -kv[1]["total"])
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--show-failures", type=int, default=0)
    parser.add_argument("--tools", type=Path,
                        default=Path(__file__).resolve().parents[1] / "examples"
                        / "pimcore-tools.json",
                        help="registry to read declared defaults from; an argument the "
                             "schema gives a default for is not unreachable")
    parser.add_argument("--by-language", action="store_true",
                        help="per-language slices; spec §60 asks for these rather "
                             "than an average, because a single weak language "
                             "disappears into one")
    args = parser.parse_args()

    preds = {}
    for row in load(args.pred):
        preds[row["id"]] = row.get("result") or {"error": row.get("error")}
    defaults = declared_defaults(args.tools)
    report = score(preds, load_keyed(args.gold), defaults)

    c = report["ceiling"]
    print(f"call-worthy requests            {report['call_worthy_requests']}")
    print(f"EXECUTABLE SEMANTIC ACCURACY    {report['executable_semantic_accuracy']:.1%}")
    print(f"  ceiling for this architecture {c['ceiling']:.1%}"
          f"   ({c['unreachable_rows']} rows want a value with no shape the host "
          "declared — see DELEGATE)")
    sf = report["safety"]
    print(f"\n  \033[1mwrong call executed          {sf['wrong_call_rate']:.1%}\033[0m"
          f"   ({sf['wrong_call_executed']} requests got a well-formed call that does the"
          " wrong thing)")
    print(f"  declined instead             {sf['declined_instead']}"
          "   \033[2m(recoverable: the host can escalate)\033[0m")
    print("\nfunnel (share of call-worthy requests):")
    for k, v in report["funnel"].items():
        print(f"  {k:32} {v:.1%}")
    if report["answered_instead"]:
        print("\nanswered with something else:")
        for k, v in report["answered_instead"].items():
            print(f"  {k:32} {v}")
    print("\nper tool (n / right tool / executable):")
    for t, c in list(report["per_tool"].items())[:12]:
        print(f"  {t:32} {c['n']:3}  {c['right_tool']:3}  {c['executable']:3}")

    if args.by_language:
        gold_rows = load_keyed(args.gold)
        langs = sorted({g.get("lang", "?") for g in gold_rows})
        print(f"\nper language ({'n':>4} {'executable':>11} {'right tool':>11} {'wrong call':>11}):")
        for lang in langs:
            rows = [g for g in gold_rows if g.get("lang", "?") == lang]
            r = score(preds, rows, defaults)
            print(f"  {lang:6} {r['call_worthy_requests']:4} "
                  f"{r['executable_semantic_accuracy']:10.1%} "
                  f"{r['funnel']['...with the right tool']:10.1%} "
                  f"{r['safety']['wrong_call_rate']:10.1%}")

    for f in report["failures"][: args.show_failures]:
        print(f"\n  {f['stage']}: {f['utterance']}\n    {f['detail']}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
