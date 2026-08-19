"""Merge live-teacher shards into a training set alongside the template data.

Run: uv run python -m tools.merge_data --base data/mini --live data/live --out data/any
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from datasets.schema import DatasetExample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/mini"))
    parser.add_argument("--live", type=Path, default=Path("data/live"))
    parser.add_argument("--out", type=Path, default=Path("data/any"))
    parser.add_argument("--holdout", type=float, default=0.0,
                        help="fraction of live examples held out to test (by id hash)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    holdout_rows: list[dict] = []
    shutil.copy(args.base / "dev.jsonl", args.out / "dev.jsonl")

    seen: set[str] = set()
    n_base = n_live = 0
    with (args.out / "train.jsonl").open("w") as out:
        for line in (args.base / "train.jsonl").read_text().splitlines():
            ex = json.loads(line)
            seen.add(ex["id"])
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_base += 1
        for shard in sorted(args.live.glob("*.jsonl")):
            for line in shard.read_text().splitlines():
                ex = json.loads(line)
                ex["split"] = "train"
                ex.setdefault("tags", []).append("live_teacher")
                ex["tags"] = sorted(set(ex["tags"]))
                try:
                    DatasetExample.model_validate(ex)
                except Exception as e:  # noqa: BLE001 — skip invalid, keep going
                    print(f"skip {ex.get('id', '?')}: {str(e)[:120]}")
                    continue
                if ex["id"] in seen:
                    continue
                seen.add(ex["id"])
                if args.holdout and (int(ex["id"][:8], 16) % 100) < args.holdout * 100:
                    ex["split"] = "test"
                    holdout_rows.append(ex)
                    continue
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_live += 1
    with (args.out / "test.jsonl").open("w") as tf:
        for line in (args.base / "test.jsonl").read_text().splitlines():
            tf.write(line + "\n")
        for ex in holdout_rows:
            tf.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"train: {n_base} template + {n_live} live-teacher, "
          f"{len(holdout_rows)} live held out to test -> {args.out}")


if __name__ == "__main__":
    main()
