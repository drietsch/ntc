"""Small LIVE teacher batch through the production orchestrator — proves the
`claude -p` path end-to-end (prompt → envelope parse → pydantic validation →
repair retry → shard + manifest). Full-scale generation uses the same code
with more prompts.

Run (from training/):  uv run python -m synthetic.run_live_batch
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from datasets.generator import TRAIN_TOOLS
from synthetic.orchestrator import build_request_gen_prompt, generate_examples, write_shard


async def main() -> None:
    candidates = [TRAIN_TOOLS["calendar.create"], TRAIN_TOOLS["email.send"], TRAIN_TOOLS["timer.set"]]
    prompts = [
        build_request_gen_prompt(candidates, lang="de", n_examples=8),
        build_request_gen_prompt(candidates, lang="fr", n_examples=8),
    ]
    report = await generate_examples(prompts, concurrency=2)
    print(
        f"teacher batch: {len(report.examples)} valid examples, "
        f"{report.retried} retried, {len(report.rejected)} rejected"
    )
    for r in report.rejected:
        print(f"  rejected ({r['stage']}): {str(r['error'])[:160]}")
    if report.examples:
        shard = write_shard(Path("data/live"), "teacher-demo-000", report.examples, prompts)
        print(f"shard → {shard}")
        ex = report.examples[0]
        print(json.dumps(ex.model_dump(), ensure_ascii=False)[:400])


if __name__ == "__main__":
    asyncio.run(main())
