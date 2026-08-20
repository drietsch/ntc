"""Does the router still work when nobody has pre-narrowed the tools?

`eval/esa.py` scores the dev split as recorded, and the dev split hands the
model a 2–3 tool slate that always contains the gold tool. A real MCP host does
not do that. It registers everything it has — Pimcore Studio has 49 tools — and
asks. Scoring only the narrow slate measures a system that does not exist:
random guessing is already 33% there, and the "who narrows 49 to 3" question is
quietly assigned to the caller.

So this re-runs the same gold rows with **the full registry offered**, which
exercises the shortlist-then-decide cascade in `NeuralToolCompiler::compile`,
and reports the same executable-accuracy funnel. The gap between the two runs
is the honest cost of the narrowing.

It also reports what the shortlist alone achieves — whether the gold tool
survives into the deciding slate at all — because those are two different
failures with two different fixes: a tool that never made the shortlist cannot
be recovered by better argument prediction.

Run (from the repo root):
    python3 eval/wide_slate.py --model models/ntc-studio-v1/model.ntc
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from esa import score  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NTC = REPO / "target" / "release" / "ntc"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


#: Raw batch-infer rows from the most recent wide run, kept so the shortlist
#: block can be read back without paying for a second inference pass.
_LAST_WIDE_ROWS: list[dict] = []


def load_pred_rows(args, gold, all_tools) -> list[dict]:
    return _LAST_WIDE_ROWS


def run(model: Path, rows: list[dict], tools: list[dict], backend: str) -> dict[str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        with inp.open("w") as f:
            for r in rows:
                f.write(json.dumps({
                    "id": r["id"],
                    "utterance": r["utterance"],
                    "tools": tools,
                    "context": r.get("context", {}),
                }, ensure_ascii=False) + "\n")
        cmd = [str(NTC), "batch-infer", "--model", str(model),
               "--input", str(inp), "--output", str(outp)]
        if backend == "gpu":
            cmd.append("--gpu")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"batch-infer failed:\n{proc.stderr[-2000:]}")
        preds = {}
        _LAST_WIDE_ROWS.clear()
        for line in outp.read_text().splitlines():
            row = json.loads(line)
            _LAST_WIDE_ROWS.append(row)
            preds[row["id"]] = row.get("result") or {"error": row.get("error")}
        return preds


def report(title: str, rep: dict) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print(f"  executable semantic accuracy   {rep['executable_semantic_accuracy']:.1%}")
    for k, v in rep["funnel"].items():
        print(f"    {k:30} {v:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=REPO / "models" / "ntc-studio-v1" / "model.ntc")
    parser.add_argument("--gold", type=Path,
                        default=REPO / "training" / "data" / "studio" / "dev.jsonl")
    parser.add_argument("--tools", type=Path,
                        default=REPO / "examples" / "pimcore-tools.json")
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--limit", type=int, default=0, help="first N rows only")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not NTC.exists():
        sys.exit("build the CLI first: cargo build --release -p ntc-cli")

    gold = load(args.gold)
    if args.limit:
        gold = gold[: args.limit]
    all_tools = json.loads(args.tools.read_text())
    by_name = {t["name"]: t for t in all_tools}

    # Every gold tool must be offerable, or the comparison is rigged.
    missing = {g["gold"]["tool"] for g in gold
               if g["gold"]["action"] == "CALL" and g["gold"]["tool"] not in by_name}
    if missing:
        sys.exit(f"gold tools absent from the registry: {sorted(missing)}")

    print(f"{len(gold)} rows · narrow slate = as recorded (2-3 tools) · "
          f"wide slate = all {len(all_tools)} tools · {args.backend.upper()}")

    narrow_rows = [{**g, "_tools": [by_name[n] for n in
                                    [c["name"] if isinstance(c, dict) else c
                                     for c in g["candidates"]]]} for g in gold]
    # Narrow: each row keeps its own recorded slate.
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        with inp.open("w") as f:
            for r in narrow_rows:
                f.write(json.dumps({
                    "id": r["id"], "utterance": r["utterance"],
                    "tools": r["_tools"], "context": r.get("context", {}),
                }, ensure_ascii=False) + "\n")
        cmd = [str(NTC), "batch-infer", "--model", str(args.model),
               "--input", str(inp), "--output", str(outp)]
        if args.backend == "gpu":
            cmd.append("--gpu")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"batch-infer failed:\n{proc.stderr[-2000:]}")
        narrow_preds = {}
        for line in outp.read_text().splitlines():
            row = json.loads(line)
            narrow_preds[row["id"]] = row.get("result") or {"error": row.get("error")}

    wide_preds = run(args.model, gold, all_tools, args.backend)

    narrow = score(narrow_preds, gold)
    wide = score(wide_preds, gold)
    report(f"narrow slate — {len(gold)} rows, gold tool always among 2-3", narrow)
    report(f"wide slate — all {len(all_tools)} tools offered, cascade narrows", wide)

    delta = wide["executable_semantic_accuracy"] - narrow["executable_semantic_accuracy"]
    print(f"\n  \033[1mcost of not being handed the answer   {delta:+.1%}\033[0m")

    # Attribute the loss: did the gold tool survive the shortlist at all?
    # A tool that never reached the deciding pass is a stage-1 (ranking)
    # failure and needs a better scorer; one that reached it and lost is a
    # stage-2 (model) failure. The end-to-end number cannot separate them.
    kept_by_id = {}
    for row in load_pred_rows(args, gold, all_tools):
        if "shortlist" in row:
            kept_by_id[row["id"]] = row["shortlist"]["kept"]

    call_rows = [g for g in gold if g["gold"]["action"] == "CALL"]
    scored = [g for g in call_rows if g["id"] in kept_by_id]
    if scored:
        survived = [g for g in scored if g["gold"]["tool"] in kept_by_id[g["id"]]]
        print(f"\n  \033[1mshortlist recall\033[0m  {len(survived)}/{len(scored)}  "
              f"{len(survived) / len(scored):.1%}"
              "   \033[2m(gold tool reached the deciding pass)\033[0m")
        # Of those that survived, how many then won?
        wide_right = {f["id"] for f in wide["failures"] if f["stage"] == "wrong_tool"}
        won = [g for g in survived if g["id"] not in wide_right]
        print(f"  of those, chosen by the deciding pass  {len(won)}/{len(survived)}  "
              f"{len(won) / max(1, len(survived)):.1%}")
        print("\n  A gold tool that never survives the shortlist cannot be recovered"
              "\n  downstream: that is a ranking problem, not a model problem.")

    lost = [f for f in wide["failures"] if f["stage"] == "wrong_tool"]
    narrow_ok = {f["id"] for f in narrow["failures"]}
    newly_wrong = [f for f in lost if f["id"] not in narrow_ok]
    print(f"\n  wrong tool on the wide slate           {len(lost)}")
    print(f"    ...of which were right when narrowed {len(newly_wrong)}"
          f"  \033[2m(pure cost of the 49-way choice)\033[0m")

    if args.out:
        args.out.write_text(json.dumps({"narrow": narrow, "wide": wide}, indent=2,
                                       ensure_ascii=False) + "\n")
        print(f"  report -> {args.out}")


if __name__ == "__main__":
    main()
