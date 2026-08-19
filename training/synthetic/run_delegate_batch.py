"""Teacher batches for the DELEGATE action: requests the fast local router
must hand to a full LLM agent instead of compiling itself.

DELEGATE is the honest verdict when a request needs capabilities the typed
single-call compiler does not have — multi-step chains with data-dependent
filtering, bulk mutations over unknown result sets, free-form reasoning or
authoring, or anything where guessing a single call would be wrong AND
asking a clarifying question would not help.

Reference case (user-provided): "Search for all blue cars in Pimcore assets.
Then take just the cars that have a building year of 1976 and before and
activate the oldtimer field for those cars."

Run (from training/):  uv run python -m synthetic.run_delegate_batch
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from synthetic.orchestrator import build_request_gen_prompt, generate_examples, write_shard

TOOLS = {t["name"]: t for t in json.loads(Path("../examples/pimcore-tools.json").read_text())}

CLUSTERS = [
    ["search_assets", "get_asset", "list_assets", "upload_asset"],
    ["search_data_objects", "create_data_object", "propose_data_object_update", "list_classes"],
    ["search_documents", "create_document", "list_document_types", "get_document_schema"],
    ["assign_tag", "unassign_tag", "create_tag", "list_tags"],
    ["list_workflows", "get_workflow_places", "list_elements_by_workflow_state", "search_data_objects"],
]

DELEGATE_GUIDE = """
Domain: Pimcore PIM/DAM/CMS (assets, data objects/products, documents, tags,
workflows). This model is a FAST LOCAL ROUTER that emits ONE typed tool call.
Use action DELEGATE (no tool, no arguments, no unresolved) when the request
needs a full LLM agent instead of a single call:

- multi-step chains where a later step depends on earlier RESULTS
  ("search X, then for those results do Y"),
- bulk mutations over an unknown/filtered result set ("update all products
  where …", "tag every image that …"),
- conditional/comparative logic ("only the ones older than 1976", "whichever
  has the most variants"),
- open-ended authoring, analysis, explanation or reasoning ("write a product
  description", "why is this document unpublished?", "summarize the changes"),
- anything requiring judgement across several tools.

Use CALL only for a single self-contained action, ASK when ONE required
argument is missing but the intent is otherwise a single call, and NO_CALL
when the user is only mentioning/discussing tools or wants something the tool
set cannot do at all.

Produce a MIX for this batch: about 55% DELEGATE, 20% CALL, 12% ASK, 13%
NO_CALL. DELEGATE examples must be realistic editor/admin requests; vary
verbs and sentence shapes; include some with numeric ids and folder paths.
DELEGATE examples MUST have "gold": {"action": "DELEGATE", "tool": null,
"arguments": [], "unresolved": []}.

Reference DELEGATE example (English):
"Search for all blue cars in Pimcore assets. Then take just the cars that have
a building year of 1976 and before and activate the oldtimer field for those
cars."
"""


async def main() -> None:
    prompts = []
    for lang in ("en", "de", "fr", "es"):
        for cluster in CLUSTERS:
            candidates = [TOOLS[n] for n in cluster]
            prompts.append(
                build_request_gen_prompt(candidates, lang=lang, n_examples=10) + DELEGATE_GUIDE
            )
    report = await generate_examples(prompts, concurrency=3)
    print(
        f"delegate teacher batch: {len(report.examples)} valid, "
        f"{report.retried} retried, {len(report.rejected)} rejected"
    )
    for r in report.rejected[:5]:
        print(f"  rejected ({r['stage']}): {str(r['error'])[:140]}")
    if report.examples:
        shard = write_shard(Path("data/live"), "delegate-000", report.examples, prompts)
        print(f"shard → {shard}")


if __name__ == "__main__":
    asyncio.run(main())
