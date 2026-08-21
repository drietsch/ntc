#!/usr/bin/env bash
# Export a checkpoint and score it on everything that matters, in one command.
#
# The order is deliberate: cheapest and most diagnostic first, so a checkpoint
# that regressed is obvious before the expensive wide-slate sweep runs.
#
#   1. narrow-slate ESA    the headline, comparable to every number so far
#   2. NO_TOOL probe       can it decline when no offered tool fits? — the
#                          capability wide-slate routing depends on, and one
#                          forward pass per row instead of eighteen
#   3. acceptance suite    26 hand-written scenarios, the "does it feel right" check
#   4. wide-slate ESA      all 49 tools offered — the honest production setting
#
# The optional-argument threshold is swept rather than assumed: it was worth 14
# points on v1 (28.4% -> 42.4%), whose presence head was broken, and 1.4 points
# on v2, whose head works. A fixed value credits or blames the model for a
# decode setting, and which of those it is changes per checkpoint.
#
# The registry and gold split are arguments, not constants. A model trained on
# schemas carrying `SEMANTIC FILTER.PQL` must be *served* those schemas: the
# canonical text is part of the model's input, and serving text it never saw
# has already cost this project 25 points once. Passing the wrong registry is
# now a visible mistake in the command rather than an invisible one in a
# hardcoded path.
#
# Usage: eval/run_all.sh <checkpoint.pt> <version-name> [registry.json] [gold.jsonl]
#   v1-v4:  eval/run_all.sh "$PWD/runs/studio-v4/best.pt" ntc-studio-v4
#   v5+:    eval/run_all.sh "$PWD/runs/studio-v5/best.pt" ntc-studio-v5 \
#             examples/pimcore-tools-templates.json training/data/studio-tpl/dev.jsonl
set -euo pipefail

CKPT="${1:?usage: run_all.sh <checkpoint.pt> <version-name> [registry.json] [gold.jsonl]}"
NAME="${2:?usage: run_all.sh <checkpoint.pt> <version-name> [registry.json] [gold.jsonl]}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$REPO/models/$NAME/model.ntc"
TOOLS="$(cd "$(dirname "${3:-$REPO/examples/pimcore-tools.json}")" && pwd)/$(basename "${3:-pimcore-tools.json}")"
GOLD="$(cd "$(dirname "${4:-$REPO/training/data/studio/dev.jsonl}")" && pwd)/$(basename "${4:-dev.jsonl}")"

echo "checkpoint $CKPT"
echo "registry   $TOOLS"
echo "gold       $GOLD"

cd "$REPO/training"
mkdir -p "$REPO/models/$NAME"
echo "=== exporting $CKPT -> $MODEL"
uv run python -m export.export_mini --ckpt "$CKPT" --out "$MODEL" \
  --version "$NAME" --tokenizer-dir tokenizer-any

cd "$REPO"
cargo build --release -p ntc-cli

IN=$(mktemp); OUT=$(mktemp)
python3 - "$GOLD" "$IN" "$TOOLS" <<'PY'
import json, sys
gold, out, registry = sys.argv[1], sys.argv[2], sys.argv[3]
tools = {t["name"]: t for t in json.load(open(registry))}
with open(out, "w") as f:
    for n, line in enumerate(open(gold)):
        r = json.loads(line)
        slate = [tools[c["name"] if isinstance(c, dict) else c] for c in r["candidates"]]
        # `esa.row_key`: id alone collides across rows with different slates.
        f.write(json.dumps({"id": f'{r["id"]}#{n}', "utterance": r["utterance"],
                            "tools": slate, "context": r.get("context", {})},
                           ensure_ascii=False) + "\n")
PY

echo
echo "=== 1. narrow-slate ESA, swept over the optional-argument threshold"
for T in 0.5 0.7 0.9 0.99; do
  NTC_OPTIONAL_ARG_THRESHOLD=$T ./target/release/ntc batch-infer --gpu \
    --model "$MODEL" --input "$IN" --output "$OUT" >/dev/null
  printf '  threshold %-5s ' "$T"
  python3 eval/esa.py --pred "$OUT" --gold "$GOLD" --tools "$TOOLS" | grep EXECUTABLE
done

echo
echo "=== 1b. per language (spec asks for slices, not an average)"
NTC_OPTIONAL_ARG_THRESHOLD=0.9 ./target/release/ntc batch-infer --gpu \
  --model "$MODEL" --input "$IN" --output "$OUT" >/dev/null
python3 eval/esa.py --pred "$OUT" --gold "$GOLD" --tools "$TOOLS" --by-language | tail -6

echo
echo "=== 2. can it say \"none of these tools fits\"?  (v2 baseline: 24.0% / intact 88.7%)"
python3 eval/no_tool_probe.py --model "$MODEL" --limit 150 --tools "$TOOLS" --gold "$GOLD"

echo
echo "=== 3. acceptance scenarios"
python3 eval/usecase/run.py --model "$MODEL" --tools "$TOOLS" || true   # exit 1 unless a clean sweep

echo
echo "=== 4. wide slate — all 49 tools, shortlist-then-decide"
python3 eval/wide_slate.py --model "$MODEL" --tools "$TOOLS" --gold "$GOLD" \
  --out "$REPO/eval/reports/$NAME-wide.json"

echo
echo "=== 5. does a focused re-read help the arguments? (measured -0.9% twice)"
python3 eval/refocus_probe.py --model "$MODEL" --tools "$TOOLS" --gold "$GOLD"

rm -f "$IN" "$OUT"
