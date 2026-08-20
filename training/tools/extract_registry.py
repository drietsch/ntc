"""Rebuild the Pimcore tool registry from the schemas the model trains on.

`examples/pimcore-tools.json` is what every eval — and the browser demo —
registers as the tool set. It had drifted badly from the corpus: 41 of 49 tools
carried a shorter description than the one in the training data and 8 carried
no description at all.

That is not cosmetic. The canonical Tool ABI renders `DESC <normalized
description>` capped at 200 characters, so a one-sentence summary and a full
description produce materially different schema text, and the tool head reads
schema states directly. Serving the short version to a model trained on the
long one cost 25 points of executable accuracy on the Studio dev split
(30.2% → 55.4%) and looked exactly like an undertrained model: wrong tools,
unfillable arguments, and a flood of ASK.

The corpus is the source of truth because it is what the weights encode. Tools
present in the registry but absent from the corpus are preserved as-is, so
hand-written scenarios that reference them keep working.

Run (from training/):
    uv run python -m tools.extract_registry --check   # CI: fail on drift
    uv run python -m tools.extract_registry           # rewrite the registry
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def corpus_schemas(data_dir: Path) -> dict[str, dict]:
    """Every distinct candidate schema in the corpus, keyed by tool name.

    A tool appears in many rows; they are identical by construction, but the
    longest description wins if that ever stops being true, since truncation
    is the failure mode this exists to prevent.
    """
    found: dict[str, dict] = {}
    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            for cand in json.loads(line)["candidates"]:
                if not isinstance(cand, dict):
                    continue
                prev = found.get(cand["name"])
                if prev is None or len(cand.get("description", "")) > len(
                    prev.get("description", "")
                ):
                    found[cand["name"]] = cand
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # specs/training is the committed source; training/data/studio is derived
    # from it and gitignored, so a drift check has to read the former to run
    # in CI at all. Their schemas are identical by construction (verified: all
    # 47 tools, byte-equal descriptions).
    parser.add_argument("--data", type=Path, default=REPO / "specs" / "training")
    parser.add_argument("--registry", type=Path,
                        default=REPO / "examples" / "pimcore-tools.json")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the registry differs from the corpus")
    args = parser.parse_args()

    corpus = corpus_schemas(args.data)
    if not corpus:
        sys.exit(f"no embedded candidate schemas under {args.data}")

    existing = {t["name"]: t for t in json.loads(args.registry.read_text())}
    merged = dict(existing)
    drifted = []
    for name, schema in corpus.items():
        if existing.get(name) != schema:
            drifted.append(name)
        merged[name] = schema

    out = [merged[n] for n in sorted(merged)]
    if args.check:
        if drifted:
            print(f"{len(drifted)} tools differ from the corpus: {sorted(drifted)[:8]}")
            print("run `uv run python -m tools.extract_registry` to resync")
            sys.exit(1)
        print(f"registry matches the corpus ({len(corpus)} tools)")
        return

    args.registry.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    kept = sorted(set(existing) - set(corpus))
    print(f"wrote {args.registry.relative_to(REPO)}: {len(out)} tools "
          f"({len(drifted)} resynced from the corpus)")
    if kept:
        print(f"  kept {len(kept)} not in the corpus: {kept}")


if __name__ == "__main__":
    main()
