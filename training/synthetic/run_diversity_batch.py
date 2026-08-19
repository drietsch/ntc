"""Live teacher batch for PHRASING DIVERSITY: colloquial/synonym-rich
utterances the template generator cannot produce, to pair with the pretrained
backbone's any-word understanding.

Run (from training/):  uv run python -m synthetic.run_diversity_batch
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from datasets.generator import TRAIN_TOOLS
from synthetic.orchestrator import build_request_gen_prompt, generate_examples, write_shard

DIVERSITY = (
    "\nIMPORTANT: maximize surface diversity. Use colloquial phrasing, synonym "
    "verbs (book/arrange/put in; besorge/leg an/richte ein; réserve/prévois; "
    "agenda/apunta), varied word order, polite forms, capitalization and "
    "punctuation as real users type. Do NOT reuse the same verb twice. Keep "
    "every char_span exactly matching its surface substring."
)

TOOLSETS = [
    ["calendar.create", "email.send", "timer.set"],
    ["light.set", "weather.lookup", "task.create"],
    ["reminder.set", "calendar.create", "light.set"],
]


async def main() -> None:
    prompts = []
    for lang in ("en", "de", "fr", "es"):
        for tools in TOOLSETS:
            candidates = [TRAIN_TOOLS[t] for t in tools]
            prompts.append(build_request_gen_prompt(candidates, lang=lang, n_examples=10) + DIVERSITY)
    report = await generate_examples(prompts, concurrency=4)
    print(
        f"teacher batch: {len(report.examples)} valid, "
        f"{report.retried} retried, {len(report.rejected)} rejected"
    )
    if report.examples:
        shard = write_shard(Path("data/live"), "diversity-000", report.examples, prompts)
        print(f"shard → {shard}")


if __name__ == "__main__":
    asyncio.run(main())
