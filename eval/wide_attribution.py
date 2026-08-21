#!/usr/bin/env python3
"""Where the wide-slate loss actually sits, read off a finished wide run.

`wide_slate.py` prints the two-stage split; this reads the per-row
`attribution` block it writes and answers the question the split cannot: for
the rows whose gold tool never survived the shortlist, *where did it rank*?

That distinction decides what to fix. A gold tool sitting at rank 4 means the
cut is too tight — widen it. A gold tool at rank 30 means the scorer is wrong,
which is a far more expensive problem. The recorded conclusion for this project
("stage 1 is the ceiling") was measured at 40.5% recall and should be re-derived
whenever recall moves, because which stage dominates changes with it.

Usage: eval/wide_attribution.py [eval/reports/<name>-wide.json]
"""
import collections
import json
import sys

REPORT = (sys.argv[1] if len(sys.argv) > 1
          else "eval/reports/ntc-studio-v5-wide.json")
rows = json.load(open(REPORT))["attribution"]
n = len(rows)
print(f"call-worthy rows: {n}\n")

surv = [r for r in rows if r["survived"]]
died = [r for r in rows if not r["survived"]]
print("stage 1 — did the gold tool survive the shortlist?")
print(f"  survived            {len(surv):3d}/{n}  {len(surv)/n:.1%}")
print(f"  never shortlisted   {len(died):3d}/{n}  {len(died)/n:.1%}   <- unrecoverable\n")

print("stage 2 — of the survivors, what happened?")
s = collections.Counter(r["outcome"] for r in surv)
for k in ("right_tool", "wrong_tool", "no_call"):
    print(f"  {k:12s}      {s[k]:3d}/{len(surv)}  {s[k]/max(1, len(surv)):.1%}")
print(f"\n  stage 2 loses {s['wrong_tool'] + s['no_call']} rows to stage 1's {len(died)}.")

ranks = [r["rank"] for r in rows if r["rank"]]
kept_n = collections.Counter(len(r["kept"] or []) for r in rows)
cut = max(kept_n) if kept_n else 0
print(f"\nranking quality — gold's place by margin over NO_TOOL (cut: {dict(kept_n)})")
for k in (1, 2, 3, 5, 9, 12):
    hit = sum(1 for r in ranks if r <= k)
    mark = "   <- the cut" if k == cut else ""
    print(f"  recall@{k:<3d} {hit:3d}/{n}  {hit/n:.1%}{mark}")
print(f"  gold outside the ranking entirely: {sum(1 for r in rows if not r['rank'])}")

print("\nper language — where each one loses")
by = collections.defaultdict(collections.Counter)
for r in rows:
    c = by[r["lang"] or "?"]
    c["n"] += 1
    c["survived"] += r["survived"]
    c[r["outcome"]] += 1
print(f"  {'lang':5s} {'n':>4s} {'recall':>8s} {'right':>7s} {'wrong':>7s} {'no_call':>8s}")
for lang in sorted(by):
    c = by[lang]
    print(f"  {lang:5s} {c['n']:4d} {c['survived']/c['n']:8.1%} "
          f"{c['right_tool']/c['n']:7.1%} {c['wrong_tool']/c['n']:7.1%} "
          f"{c['no_call']/c['n']:8.1%}")
