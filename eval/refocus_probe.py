"""Would a focused second look at the chosen tool extract better arguments?

The executable-accuracy funnel loses far more at the argument stage than at
tool selection (69% right tool -> 42% fully correct). One plausible cause is
attention budget: in a shared slate the fusion blocks must attend across every
candidate's schema, so the winning tool's arguments compete for capacity with
schemas that were only ever going to be rejected.

If that is the cause, then re-running the same model with **only the chosen
tool in the slate** should extract better arguments, since nothing else is
competing. If it is not the cause — if the argument heads are simply
undertrained — the focused pass will change little, and building a third stage
would be wasted work.

This measures the ceiling of that idea directly, by running each row with a
slate of exactly one tool: the **gold** tool. Using gold rather than the
predicted tool isolates the argument question from the selection question, and
answers "is a refocus pass worth building at all" before any is built. It is an
oracle, so its number is an upper bound, not an achievable score.

Run (from the repo root):
    python3 eval/refocus_probe.py --model models/ntc-studio-v1/model.ntc
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from esa import values_equal  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NTC = REPO / "target" / "release" / "ntc"


def infer(model: Path, lines: list[dict], backend: str) -> dict[str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        inp.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines))
        cmd = [str(NTC), "batch-infer", "--model", str(model),
               "--input", str(inp), "--output", str(outp)]
        if backend == "gpu":
            cmd.append("--gpu")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"batch-infer failed:\n{proc.stderr[-2000:]}")
        return {
            json.loads(l)["id"]: (json.loads(l).get("result") or {"error": json.loads(l).get("error")})
            for l in outp.read_text().splitlines()
        }


def args_correct(pred: dict, gold_row: dict) -> tuple[bool, dict]:
    if pred.get("outcome") != "CALL":
        return False, {"no_call": pred.get("outcome")}
    got = (pred.get("call") or {}).get("arguments") or {}
    want = {a["parameter"]: a["value"] for a in gold_row["gold"]["arguments"]}
    missing = [k for k in want if k not in got]
    invented = [k for k in got if k not in want]
    wrong = [k for k in want if k in got and not values_equal(got[k], want[k])]
    return (not missing and not invented and not wrong), {
        "missing": missing, "invented": invented, "wrong": wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=REPO / "models" / "ntc-studio-v1" / "model.ntc")
    parser.add_argument("--gold", type=Path,
                        default=REPO / "training" / "data" / "studio" / "dev.jsonl")
    parser.add_argument("--tools", type=Path,
                        default=REPO / "examples" / "pimcore-tools.json")
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    by_name = {t["name"]: t for t in json.loads(args.tools.read_text())}
    rows = [json.loads(l) for l in args.gold.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["gold"]["action"] == "CALL"]
    # Only rows that actually take arguments can distinguish the two runs.
    rows = [r for r in rows if r["gold"]["arguments"]]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("no call-with-arguments rows in the gold split")

    def slate(r):
        return [by_name[c["name"] if isinstance(c, dict) else c] for c in r["candidates"]]

    # Keyed by line: the shared run's whole point is the recorded slate, and
    # several rows share an id while carrying different slates.
    for i, r in enumerate(rows):
        r["_key"] = f'{r["id"]}#{i}'

    shared = infer(args.model, [
        {"id": r["_key"], "utterance": r["utterance"], "tools": slate(r),
         "context": r.get("context", {})} for r in rows], args.backend)
    focused = infer(args.model, [
        {"id": r["_key"], "utterance": r["utterance"],
         "tools": [by_name[r["gold"]["tool"]]],
         "context": r.get("context", {})} for r in rows], args.backend)

    stats = {"shared": 0, "focused": 0, "fixed": 0, "broken": 0}
    reasons = {"shared": [], "focused": []}
    for r in rows:
        s_ok, s_why = args_correct(shared.get(r["_key"], {}), r)
        f_ok, f_why = args_correct(focused.get(r["_key"], {}), r)
        stats["shared"] += s_ok
        stats["focused"] += f_ok
        stats["fixed"] += (f_ok and not s_ok)
        stats["broken"] += (s_ok and not f_ok)
        if not s_ok:
            reasons["shared"].append(s_why)
        if not f_ok:
            reasons["focused"].append(f_why)

    n = len(rows)
    print(f"{n} gold CALL rows that take at least one argument · {args.backend.upper()}\n")
    print(f"  arguments fully correct, shared slate (2-3 tools)   {stats['shared']:3}/{n}  "
          f"{stats['shared'] / n:.1%}")
    print(f"  arguments fully correct, focused slate (gold only)  {stats['focused']:3}/{n}  "
          f"{stats['focused'] / n:.1%}")
    print(f"\n  rows the focus fixed   {stats['fixed']:3}")
    print(f"  rows the focus broke   {stats['broken']:3}")

    for label in ("shared", "focused"):
        agg = {"missing": 0, "invented": 0, "wrong": 0, "no_call": 0}
        for w in reasons[label]:
            for k in agg:
                v = w.get(k)
                agg[k] += len(v) if isinstance(v, list) else bool(v)
        print(f"\n  {label:8} failures — " + "  ".join(f"{k} {v}" for k, v in agg.items()))

    gain = (stats["focused"] - stats["shared"]) / n
    verdict = ("worth building a focused third pass" if gain >= 0.05 else
               "not worth a third pass — the argument heads, not the slate, are the limit")
    print(f"\n  \033[1m{gain:+.1%} — {verdict}\033[0m")


if __name__ == "__main__":
    main()
