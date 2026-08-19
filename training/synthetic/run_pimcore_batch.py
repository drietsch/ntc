"""Live teacher batches for the PIMCORE tool set (the POC's real tools,
extracted from pimcore/pimcore-agent-bundle MCP attributes).

Clusters of ≤4 related tools form each example's candidate set (the model's
pimcore arch uses max_tools=4 — retrieval pre-narrows in the full design);
same-category clusters give naturally adversarial decoys (search_assets vs
list_assets vs get_asset).

Run (from training/):  uv run python -m synthetic.run_pimcore_batch
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from synthetic.orchestrator import build_request_gen_prompt, generate_examples, write_shard

TOOLS = {t["name"]: t for t in json.loads(Path("../examples/pimcore-tools.json").read_text())}

CLUSTERS = [
    ["search_assets", "get_asset", "list_assets", "upload_asset"],
    ["stage_asset", "get_asset_queryable_fields", "list_thumbnail_configs", "search_assets"],
    ["search_data_objects", "get_class_definition", "list_classes", "get_queryable_fields"],
    ["create_data_object", "propose_data_object_update", "get_field_schema", "search_data_objects"],
    ["search_documents", "create_document", "list_document_types", "get_document_schema"],
    ["assign_tag", "unassign_tag", "create_tag", "list_tags"],
    ["list_workflows", "get_workflow_places", "list_elements_by_workflow_state", "get_element_workflow_details"],
    ["search_assets", "search_data_objects", "search_documents", "get_element_tags"],
]

CONTEXT = (
    "\nDomain context: Pimcore PIM/DAM/CMS. Users manage assets (images, PDFs, "
    "folders like /Marketing), data objects (products, categories — classes like "
    "Product, Category), documents (pages, snippets), tags, and workflows. "
    "Write REALISTIC editor/admin requests: 'find all PDFs in the marketing "
    "folder', 'tag asset 123 with summer', 'welche klassen gibt es?', 'crée une "
    "page à propos sous la page d'accueil'. Numeric ids appear as digits in the "
    "utterance with exact char_spans. Mix CALL (~60%), ASK for a missing "
    "required argument (~20%), and NO_CALL (mention-only or unsupported, ~20%). "
    "Maximize verb and phrasing diversity; never reuse a verb twice."
)


async def main() -> None:
    missing = [n for c in CLUSTERS for n in c if n not in TOOLS]
    assert not missing, f"unknown tools in clusters: {missing}"
    prompts = []
    for lang in ("en", "de", "fr", "es"):
        for cluster in CLUSTERS:
            candidates = [TOOLS[n] for n in cluster]
            prompts.append(
                build_request_gen_prompt(candidates, lang=lang, n_examples=10) + CONTEXT
            )
    report = await generate_examples(prompts, concurrency=4)
    print(
        f"pimcore teacher batch: {len(report.examples)} valid, "
        f"{report.retried} retried, {len(report.rejected)} rejected"
    )
    for r in report.rejected[:5]:
        print(f"  rejected ({r['stage']}): {str(r['error'])[:140]}")
    if report.examples:
        shard = write_shard(Path("data/live"), "pimcore-000", report.examples, prompts)
        print(f"shard → {shard}")


if __name__ == "__main__":
    asyncio.run(main())
