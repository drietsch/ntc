"""Acceptance tests for the Pimcore Studio use case.

Distinct from `eval/studio_report.py`, which scores the model on the corpus's
own held-out split. These are **hand-written scenarios in the shape a real
Studio user would type**, each stating what it tests and why, run through the
shipping runtime (WebGPU by default). They answer a different question: does
the router behave sensibly on requests nobody generated from a template?

Each scenario asserts only what should be stable:
  action        — always
  tool          — when exactly one tool is defensible
  reason        — when the DELEGATE/NO_CALL reason is unambiguous
  arg_values    — when a specific value must be bound (e.g. the id in the
                  utterance, not the one in the selection)

Run (from the repo root):
    python3 eval/usecase/run.py --model models/ntc-studio-v1/model.ntc
    python3 eval/usecase/run.py --model ... --backend cpu   # parity oracle
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NTC = REPO / "target" / "release" / "ntc"
DEFAULT_TOOLS = REPO / "examples" / "pimcore-tools.json"


def load_registry(path: Path) -> dict[str, dict]:
    """The schemas to serve. Must be the ones the model trained against — the
    canonical text is part of its input, so a registry that has drifted reads
    as an undertrained model rather than as an error."""
    return {t["name"]: t for t in json.loads(path.read_text())}

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def load_scenarios(path: Path, tools: dict[str, dict]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    missing = {t for r in rows for t in r["slate"] if t not in tools}
    if missing:
        sys.exit(f"unknown tools in scenarios: {sorted(missing)}")
    return rows


def run_batch(
    model: Path, scenarios: list[dict], backend: str, tools: dict[str, dict]
) -> dict[str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        inp.write_text(
            "".join(
                json.dumps(
                    {
                        "id": s["id"],
                        "utterance": s["utterance"],
                        "tools": [tools[t] for t in s["slate"]],
                        "context": s.get("context", {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for s in scenarios
            )
        )
        cmd = [str(NTC), "batch-infer", "--model", str(model),
               "--input", str(inp), "--output", str(outp)]
        if backend == "gpu":
            cmd.append("--gpu")
        subprocess.run(cmd, check=True, capture_output=True)
        out = {}
        for line in outp.read_text().splitlines():
            row = json.loads(line)
            out[row["id"]] = row.get("result") or {"error": row.get("error")}
        return out


def check(scenario: dict, pred: dict) -> tuple[bool, list[str]]:
    """Compare a prediction against the scenario's expectations."""
    exp = scenario["expect"]
    ir = pred.get("ir") or {}
    problems: list[str] = []

    if "error" in pred:
        return False, [f"inference error: {str(pred['error'])[:70]}"]

    got_action = pred.get("outcome")
    if got_action != exp["action"]:
        problems.append(f"action {got_action} != {exp['action']}")

    if "tool" in exp:
        got_tool = (ir.get("tool") or {}).get("registry_id")
        if got_tool != exp["tool"]:
            problems.append(f"tool {got_tool} != {exp['tool']}")

    if "reason" in exp:
        got_reason = ir.get("delegate_reason") or ir.get("no_call_reason")
        if got_reason != exp["reason"]:
            problems.append(f"reason {got_reason} != {exp['reason']}")

    if "arg_values" in exp:
        args = (pred.get("call") or {}).get("arguments") or {}
        for k, want in exp["arg_values"].items():
            if args.get(k) != want:
                problems.append(f"{k}={args.get(k)!r} != {want!r}")

    return not problems, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPO / "models" / "ntc-studio-v1" / "model.ntc")
    parser.add_argument("--scenarios", type=Path, default=Path(__file__).parent / "scenarios.jsonl")
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu",
                        help="gpu = the shipping target; cpu = the parity oracle")
    parser.add_argument("--json", type=Path, default=None, help="write a machine-readable report")
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS,
                        help="registry to serve; must match what the model trained on")
    args = parser.parse_args()

    if not NTC.exists():
        sys.exit("build the CLI first: cargo build --release -p ntc-cli")
    if not args.model.exists():
        sys.exit(f"model not found: {args.model} (see .gitignore for the rebuild recipe)")

    tools = load_registry(args.tools)
    scenarios = load_scenarios(args.scenarios, tools)
    preds = run_batch(args.model, scenarios, args.backend, tools)

    print(f"{BOLD}Studio use-case acceptance — {args.model.name} on {args.backend.upper()}{RESET}\n")
    by_group: Counter[str] = Counter()
    by_group_pass: Counter[str] = Counter()
    rows = []
    for s in scenarios:
        ok, problems = check(s, preds.get(s["id"], {}))
        group = s["id"].split("-")[0]
        by_group[group] += 1
        by_group_pass[group] += ok
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {mark}  {s['id']:32} {DIM}{s['why'][:58]}{RESET}")
        if not ok:
            print(f"        {RED}{'; '.join(problems)}{RESET}")
            print(f"        {DIM}utterance: {s['utterance'][:88]}{RESET}")
        rows.append({"id": s["id"], "group": group, "passed": ok,
                     "problems": problems, "why": s["why"]})

    passed = sum(r["passed"] for r in rows)
    print(f"\n{BOLD}{passed}/{len(rows)} scenarios pass{RESET}")
    print("  by group: " + "  ".join(
        f"{g} {by_group_pass[g]}/{by_group[g]}" for g in sorted(by_group)
    ))

    if args.json:
        args.json.write_text(json.dumps(
            {"model": args.model.name, "backend": args.backend,
             "passed": passed, "total": len(rows), "scenarios": rows},
            indent=2, ensure_ascii=False) + "\n")
        print(f"  report -> {args.json}")

    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()
