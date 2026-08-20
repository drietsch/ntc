"""Teach the model to ask when a required argument has no source.

Third instance of the same corpus gap. Twice now a capability has been missing
not because the model was weak but because no training row ever demanded it:

  NO_TOOL      no row paired a call-worthy request with a slate lacking its
               tool, so the shortlist ranked on meaningless margins
               (`tools/add_gold_absent.py`)
  ASK          no row shows a *required* argument with nothing to fill it

The second gap has teeth. `"show me the asset"` — no id in the utterance,
nothing selected — used to compile to `get_asset(id = 44848)`, a fabricated
entity id executed against a live server. The decoder no longer invents
numbers, so it now answers ASK by falling through rather than by recognizing
the situation, and the model itself still has no signal for it. Dev cannot
measure any of this: every CALL row there has a source for every argument, so
the case is invisible by construction.

The material is already in the corpus. 1,001 CALL rows have a required argument
sourced from a linked item — `elementId`, `id`, `ids`, `className`. Removing the
linked selection leaves the argument with nothing to bind to, which is the
missing case exactly, and the correct answer becomes ASK: *which* asset?

The utterance is untouched, so the model cannot spot these from surface form —
"tag this one" reads identically whether or not anything is selected. Only the
context frame differs, which is the discrimination worth learning.

Run (from training/):
    uv run python -m tools.add_source_absent --src data/studio-neg --out data/studio-ask
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


def required_params(tools: dict[str, dict], tool_name: str) -> set[str]:
    spec = (tools.get(tool_name) or {}).get("parameters") or {}
    return {k for k, v in spec.items() if v.get("required")}


def make_unsourced(row: dict, tools: dict[str, dict]) -> dict | None:
    """Strip the linked selection, leaving a required argument unfillable."""
    ctx = row.get("context") or {}
    if not ctx.get("linked"):
        return None
    required = required_params(tools, row["gold"].get("tool") or "")
    orphaned = [
        a["parameter"]
        for a in row["gold"].get("arguments", [])
        if a.get("source") == "LINKED_ITEM" and a["parameter"] in required
    ]
    if not orphaned:
        return None

    out = json.loads(json.dumps(row))
    out["context"] = {
        k: v for k, v in ctx.items() if k not in ("linked", "selection_count", "selectionCount")
    }
    out["context"]["linked"] = []
    # The tool is still right and still knowable — only the value is missing —
    # so ASK keeps the tool and lists what it cannot fill. That is the whole
    # point of ASK over NO_CALL: the host can answer the question and retry.
    out["gold"] = {
        "action": "ASK",
        "tool": row["gold"]["tool"],
        "arguments": [
            a
            for a in row["gold"].get("arguments", [])
            if a["parameter"] not in orphaned and a.get("source") != "LINKED_ITEM"
        ],
        "unresolved": [{"parameter": p, "reason": "MISSING"} for p in orphaned],
    }
    out["tags"] = sorted(set(out.get("tags", []) + ["source_absent"]))
    out["annotations"] = {}
    out["id"] = "ask-" + hashlib.sha256(
        json.dumps(out, sort_keys=True).encode()
    ).hexdigest()[:14]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/studio-neg"))
    parser.add_argument("--out", type=Path, default=Path("data/studio-ask"))
    parser.add_argument("--tools", type=Path,
                        default=REPO / "examples" / "pimcore-tools.json")
    parser.add_argument("--rate", type=float, default=0.45,
                        help="fraction of eligible rows to add an unsourced copy of")
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    tools = {t["name"]: t for t in json.loads(args.tools.read_text())}

    train = load(args.src / "train.jsonl")
    eligible = [r for r in train if r["gold"]["action"] == "CALL" and (r.get("context") or {}).get("linked")]
    rng.shuffle(eligible)

    added, seen = [], {r["id"] for r in train}
    for row in eligible[: int(len(eligible) * args.rate)]:
        v = make_unsourced(row, tools)
        if v and v["id"] not in seen:
            seen.add(v["id"])
            added.append(v)

    out_rows = train + added
    rng.shuffle(out_rows)
    with (args.out / "train.jsonl").open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ("dev", "test"):
        src = args.src / f"{split}.jsonl"
        if src.exists():
            shutil.copy(src, args.out / f"{split}.jsonl")

    stats = {"base": len(train), "ask_added": len(added), "total": len(out_rows)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
