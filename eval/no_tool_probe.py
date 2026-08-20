"""Can the router say "none of these tools fits"?

This is the one question the wide-slate cascade rests on. `shortlist` ranks a
49-tool registry by scoring it in slates of three, so 16 of every 17 slates
contain no correct tool. If the model cannot recognize that, its NO_TOOL
margins are noise and the ranking is worthless no matter how good the deciding
pass is — measured, shortlist recall was 40.5% while the deciding pass picked
correctly 80% of the time from whatever survived.

The dev split cannot answer it: every CALL row contains its gold tool, so
NO_TOOL only ever appears where nothing was asked of the model at all. So this
builds the missing case directly — take a CALL row, drop its gold tool from the
slate, backfill with a decoy — and asks what the model does.

Two numbers, and both matter:

  NO_TOOL recall     on a slate with the gold tool removed, does the model
                     decline rather than pick the nearest thing?
  intact accuracy    on the *unmodified* slate, does it still choose correctly?

The second is the guard. A model that has learned to say NO_TOOL by becoming
reluctant has not improved; it has traded one failure for another, and only
reporting both catches that.

Run (from the repo root):
    python3 eval/no_tool_probe.py --model models/ntc-studio-v2/model.ntc
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NTC = REPO / "target" / "release" / "ntc"


def infer(model: Path, lines: list[dict], backend: str) -> dict[str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        inp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
        cmd = [str(NTC), "batch-infer", "--model", str(model),
               "--input", str(inp), "--output", str(outp)]
        if backend == "gpu":
            cmd.append("--gpu")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"batch-infer failed:\n{proc.stderr[-2000:]}")
        out = {}
        for line in outp.read_text().splitlines():
            row = json.loads(line)
            out[row["id"]] = row.get("result") or {"error": row.get("error")}
        return out


def chosen_tool(pred: dict) -> str | None:
    return ((pred.get("ir") or {}).get("tool") or {}).get("registry_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gold", type=Path,
                        default=REPO / "training" / "data" / "studio" / "dev.jsonl")
    parser.add_argument("--tools", type=Path,
                        default=REPO / "examples" / "pimcore-tools.json")
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pool = {t["name"]: t for t in json.loads(args.tools.read_text())}
    rows = [json.loads(l) for l in args.gold.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["gold"]["action"] == "CALL" and r["gold"].get("tool")]
    if args.limit:
        rows = rows[: args.limit]

    intact, stripped, expect = [], [], {}
    for r in rows:
        names = [c["name"] if isinstance(c, dict) else c for c in r["candidates"]]
        gold = r["gold"]["tool"]
        if gold not in names or gold not in pool:
            continue
        slate = [pool[n] for n in names if n in pool]
        intact.append({"id": r["id"], "utterance": r["utterance"],
                       "tools": slate, "context": r.get("context", {})})

        others = [n for n in pool if n not in names]
        if not others:
            continue
        replacement = pool[rng.choice(others)]
        neg_slate = [pool[n] for n in names if n != gold and n in pool] + [replacement]
        stripped.append({"id": r["id"], "utterance": r["utterance"],
                         "tools": neg_slate, "context": r.get("context", {})})
        expect[r["id"]] = gold

    if not stripped:
        sys.exit("no rows could have their gold tool removed")

    intact_pred = infer(args.model, intact, args.backend)
    stripped_pred = infer(args.model, stripped, args.backend)

    intact_ok = sum(1 for x in intact if chosen_tool(intact_pred.get(x["id"], {})) == expect.get(x["id"]))
    declined, picked = 0, {}
    for x in stripped:
        p = stripped_pred.get(x["id"], {})
        tool = chosen_tool(p)
        if tool is None:
            declined += 1
        else:
            picked[tool] = picked.get(tool, 0) + 1

    n_i, n_s = len(intact), len(stripped)
    print(f"{n_s} rows, each scored twice — with and without its gold tool "
          f"in the slate · {args.backend.upper()}\n")
    print(f"  intact slate, correct tool chosen   {intact_ok:3}/{n_i}  {intact_ok / max(1, n_i):.1%}"
          "   \033[2m(the guard: must not regress)\033[0m")
    print(f"  gold removed, declined (NO_TOOL)    {declined:3}/{n_s}  {declined / max(1, n_s):.1%}"
          "   \033[2m(the capability the cascade needs)\033[0m")

    if picked:
        print("\n  when it did not decline, what it reached for instead:")
        for tool, count in sorted(picked.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {tool:34} {count}")


if __name__ == "__main__":
    main()
