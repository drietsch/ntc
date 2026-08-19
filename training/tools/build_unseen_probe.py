"""Unseen-tool probe: can the router select and bind a Pimcore tool it has
never seen during fine-tuning?

24 of the 49 extracted Pimcore tools never appear in `data/pimcore/train.jsonl`.
This builds a small hand-written eval set over exactly those, so the question
"does Stage-2 schema grounding on xLAM transfer to unseen schemas?" has an
honest, held-out answer rather than a training-set echo.

Each probe example states the required arguments literally in the utterance,
so a correct answer needs schema *reading* (find the tool by its description,
bind its declared parameters) rather than memorization. Tools whose required
arguments are OPAQUE are probed for DELEGATE instead — that is their correct
verdict (docs/delegation.md).

Run: uv run python -m tools.build_unseen_probe --out data/pimcore-unseen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets.schema import DatasetExample

REPO = Path(__file__).resolve().parents[2]
TOOLS = {t["name"]: t for t in json.loads((REPO / "examples" / "pimcore-tools.json").read_text())}

# (tool, utterance, {param: (semantic_type, value, surface|None)})
# Surfaces are literal substrings of the utterance; None = no span (inferred).
PROBES: list[tuple[str, str, dict]] = [
    ("get_data_object", "get me data objects 42, 55 and 101",
     {"ids": ("LIST", [42, 55, 101], "42, 55 and 101")}),
    ("get_document", "fetch documents 12 and 18",
     {"ids": ("LIST", [12, 18], "12 and 18")}),
    ("get_field_schema", "show the field schema for class Product",
     {"className": ("STRING", "Product", "Product")}),
    ("get_workflow_graph", "render the workflow graph for product_review",
     {"workflowName": ("STRING", "product_review", "product_review")}),
    ("get_element_workflow_details", "what workflow state is asset 812 in",
     {"elementId": ("INTEGER", 812, "812")}),
    ("list_editables", "list the editables of document 355",
     {"documentId": ("INTEGER", 355, "355")}),
    ("get_area_brick", "describe the area brick called teaser",
     {"brickName": ("STRING", "teaser", "teaser")}),
    ("describe_tool", "describe the tool search_assets",
     {"name": ("STRING", "search_assets", "search_assets")}),
    ("list_agent_templates", "list the agent templates", {}),
    ("list_target_groups", "show all target groups", {}),
    # German / French / Spanish variants — same schema-reading task.
    ("get_field_schema", "zeig mir das feldschema der klasse Product",
     {"className": ("STRING", "Product", "Product")}),
    ("list_editables", "liste die editables von dokument 355 auf",
     {"documentId": ("INTEGER", 355, "355")}),
    ("get_document", "hol die dokumente 12 und 18",
     {"ids": ("LIST", [12, 18], "12 und 18")}),
    ("get_workflow_graph", "affiche le graphe du workflow product_review",
     {"workflowName": ("STRING", "product_review", "product_review")}),
    ("get_field_schema", "muéstrame el esquema de campos de la clase Product",
     {"className": ("STRING", "Product", "Product")}),
]

# Tools whose required arguments are OPAQUE: a single typed call cannot
# express them, so DELEGATE is the correct verdict.
DELEGATE_PROBES: list[tuple[str, str]] = [
    ("apply_transition", "apply the publish transition to assets 12 and 42 in the product_review workflow"),
    ("propose_tag_assignment", "propose tagging object 42 with the summer and press tags"),
    ("update_document", "update document 355 and set its headline to Autumn Sale"),
    ("update_asset", "rename asset 812 to new-filename.jpg"),
    ("apply_global_action", "run the archive global action on documents 12 and 18"),
]

CLUSTER_SIZE = 4


def make_id(payload: dict) -> str:
    return "unseen-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:12]


def candidates_for(gold: str, rng: random.Random) -> list[dict]:
    """Gold tool plus decoys drawn from the other unseen tools."""
    pool = [n for n in TOOLS if n != gold]
    decoys = rng.sample(pool, CLUSTER_SIZE - 1)
    picked = [TOOLS[gold]] + [TOOLS[d] for d in decoys]
    rng.shuffle(picked)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/pimcore-unseen"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    rows: list[dict] = []
    for tool, utterance, params in PROBES:
        arguments = []
        for pname, (stype, value, surface) in params.items():
            arg: dict = {"parameter": pname, "semantic_type": stype}
            if stype == "LIST":
                item_type = "INTEGER" if all(isinstance(v, int) for v in value) else "STRING"
                arg["value"] = {"items": [{"semantic_type": item_type, "value": v} for v in value]}
            else:
                arg["value"] = value
            if surface:
                start = utterance.index(surface)
                arg["char_span"] = {"start": start, "end": start + len(surface)}
                arg["surface"] = surface
            arguments.append(arg)
        ex = {
            "id": "", "lang": "en", "utterance": utterance,
            "candidates": candidates_for(tool, rng),
            "gold": {"action": "CALL", "tool": tool, "arguments": arguments, "unresolved": []},
            "split": "test", "tags": ["unseen_tool", "call"],
        }
        ex["id"] = make_id(ex)
        DatasetExample.model_validate(ex)
        rows.append(ex)

    for tool, utterance in DELEGATE_PROBES:
        ex = {
            "id": "", "lang": "en", "utterance": utterance,
            "candidates": candidates_for(tool, rng),
            "gold": {"action": "DELEGATE", "tool": None, "arguments": [], "unresolved": []},
            "split": "test", "tags": ["unseen_tool", "delegate", "opaque_required"],
        }
        ex["id"] = make_id(ex)
        DatasetExample.model_validate(ex)
        rows.append(ex)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "test.jsonl").open("w") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    # The harness expects all three splits to exist.
    for split in ("train", "dev"):
        (args.out / f"{split}.jsonl").write_text("")
    print(f"{len(rows)} probe examples "
          f"({sum(1 for r in rows if r['gold']['action'] == 'CALL')} CALL / "
          f"{sum(1 for r in rows if r['gold']['action'] == 'DELEGATE')} DELEGATE) -> {args.out}")


if __name__ == "__main__":
    main()
