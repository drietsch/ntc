"""Teach the model that sometimes none of the offered tools is the right one.

The Studio corpus has **zero** rows where a call-worthy request is shown a slate
that does not contain its tool. Every CALL row includes its gold tool; every row
without one is a request that was not callable to begin with. So the model has
never had to answer "this is a real request, and none of these three tools
serves it" — and NO_TOOL only ever means "nothing was asked of me".

That gap is invisible until the caller stops pre-narrowing the slate. A real
MCP host offers everything it has, so `NeuralToolCompiler::shortlist` scores the
tool set in slate-sized groups — and for a 49-tool registry, **16 of 17 groups
contain no correct tool at all**. The model has no signal for that case, so it
picks the closest-looking candidate with high confidence, and the margins the
shortlist ranks on become noise. Measured: tool accuracy 90.8% on the recorded
2-3 tool slate, 26.3% when the same rows are asked against all 49.

So this adds the missing supervision: take a CALL row, remove its gold tool from
the slate, fill the gap with decoys, and label it NO_CALL /
UNSUPPORTED_CAPABILITY — which is what the runtime *should* answer when the
registry genuinely cannot serve a request. The utterance is untouched, so the
model cannot learn to spot these from surface form; only the slate differs.

Decoys are drawn from the same registry, so a negative slate looks exactly like
a real one. Rows keep a `gold_absent` tag so their effect can be measured
separately.

Run (from training/):
    uv run python -m tools.add_gold_absent --src data/studio-aug --out data/studio-neg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tool_pool(rows: list[dict]) -> dict[str, dict]:
    """Every candidate schema seen anywhere, so decoys are real tools."""
    pool: dict[str, dict] = {}
    for r in rows:
        for c in r["candidates"]:
            if isinstance(c, dict) and "name" in c:
                pool.setdefault(c["name"], c)
    return pool


def make_negative(row: dict, pool: dict[str, dict], rng: random.Random) -> dict | None:
    """A copy of `row` whose slate is the same size but excludes the gold tool."""
    gold_tool = row["gold"].get("tool")
    if not gold_tool:
        return None
    names = [c["name"] if isinstance(c, dict) else c for c in row["candidates"]]
    if gold_tool not in names:
        return None

    # Anything except the gold tool. Keeping the other original candidates is
    # deliberate: they were chosen as plausible confusions for this utterance,
    # so the negative stays hard rather than becoming trivially unrelated.
    keep = [c for c, n in zip(row["candidates"], names, strict=False) if n != gold_tool]
    others = [n for n in pool if n != gold_tool and n not in names]
    if not others:
        return None
    keep.append(pool[rng.choice(others)])

    out = json.loads(json.dumps(row))
    out["candidates"] = keep
    out["gold"] = {
        "action": "NO_CALL",
        "tool": None,
        "arguments": [],
        "unresolved": [],
        # The request is fine; this tool set cannot serve it. That is exactly
        # what UNSUPPORTED_CAPABILITY means, and it is what the runtime should
        # answer when a registry genuinely lacks the tool.
        "no_call_reason": "UNSUPPORTED_CAPABILITY",
    }
    out["tags"] = sorted(set(out.get("tags", []) + ["gold_absent"]))
    out["annotations"] = {}
    out["id"] = "neg-" + hashlib.sha256(
        json.dumps(out, sort_keys=True).encode()
    ).hexdigest()[:14]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/studio-aug"))
    parser.add_argument("--out", type=Path, default=Path("data/studio-neg"))
    parser.add_argument("--rate", type=float, default=0.5,
                        help="negatives to add, as a fraction of CALL rows")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--also-dev", action="store_true",
                        help="add negatives to dev too (for measuring, not training)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    train = load(args.src / "train.jsonl")
    pool = tool_pool(train)
    calls = [r for r in train if r["gold"]["action"] == "CALL" and r["gold"].get("tool")]
    rng.shuffle(calls)

    added, seen = [], {r["id"] for r in train}
    for row in calls[: int(len(calls) * args.rate)]:
        neg = make_negative(row, pool, rng)
        if neg and neg["id"] not in seen:
            seen.add(neg["id"])
            added.append(neg)

    out_rows = train + added
    rng.shuffle(out_rows)
    with (args.out / "train.jsonl").open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for split in ("dev", "test"):
        src = args.src / f"{split}.jsonl"
        if not src.exists():
            continue
        if split == "dev" and args.also_dev:
            rows = load(src)
            negs = [
                n
                for r in rows
                if r["gold"]["action"] == "CALL" and r["gold"].get("tool")
                for n in [make_negative(r, tool_pool(rows), rng)]
                if n
            ]
            with (args.out / "dev.jsonl").open("w") as f:
                for r in rows + negs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"dev: {len(rows)} + {len(negs)} negatives")
        else:
            shutil.copy(src, args.out / f"{split}.jsonl")

    stats = {"base": len(train), "negatives_added": len(added), "total": len(out_rows)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
