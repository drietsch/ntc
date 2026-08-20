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
# Usage: eval/run_all.sh runs/studio-v2/best.pt ntc-studio-v2
set -euo pipefail

CKPT="${1:?usage: run_all.sh <checkpoint.pt> <version-name>}"
NAME="${2:?usage: run_all.sh <checkpoint.pt> <version-name>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$REPO/models/$NAME/model.ntc"
GOLD="$REPO/training/data/studio/dev.jsonl"

cd "$REPO/training"
mkdir -p "$REPO/models/$NAME"
echo "=== exporting $CKPT -> $MODEL"
uv run python -m export.export_mini --ckpt "$CKPT" --out "$MODEL" \
  --version "$NAME" --tokenizer-dir tokenizer-any

cd "$REPO"
cargo build --release -p ntc-cli

IN=$(mktemp); OUT=$(mktemp)
python3 - "$GOLD" "$IN" <<'PY'
import json, sys
gold, out = sys.argv[1], sys.argv[2]
tools = {t["name"]: t for t in json.load(open("examples/pimcore-tools.json"))}
with open(out, "w") as f:
    for line in open(gold):
        r = json.loads(line)
        slate = [tools[c["name"] if isinstance(c, dict) else c] for c in r["candidates"]]
        f.write(json.dumps({"id": r["id"], "utterance": r["utterance"],
                            "tools": slate, "context": r.get("context", {})},
                           ensure_ascii=False) + "\n")
PY

echo
echo "=== 1. narrow-slate ESA, swept over the optional-argument threshold"
for T in 0.5 0.7 0.9 0.99; do
  NTC_OPTIONAL_ARG_THRESHOLD=$T ./target/release/ntc batch-infer --gpu \
    --model "$MODEL" --input "$IN" --output "$OUT" >/dev/null
  printf '  threshold %-5s ' "$T"
  python3 eval/esa.py --pred "$OUT" --gold "$GOLD" | grep EXECUTABLE
done

echo
echo "=== 2. can it say \"none of these tools fits\"?  (v2 baseline: 24.0% / intact 88.7%)"
python3 eval/no_tool_probe.py --model "$MODEL" --limit 150

echo
echo "=== 3. acceptance scenarios"
python3 eval/usecase/run.py --model "$MODEL" || true   # exit 1 unless a clean sweep

echo
echo "=== 4. wide slate — all 49 tools, shortlist-then-decide"
python3 eval/wide_slate.py --model "$MODEL" --out "$REPO/eval/reports/$NAME-wide.json"

echo
echo "=== 5. does a focused re-read help the arguments? (measured -0.9% twice)"
python3 eval/refocus_probe.py --model "$MODEL"

rm -f "$IN" "$OUT"
