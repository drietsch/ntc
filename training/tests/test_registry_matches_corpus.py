"""The served tool registry must be the one the model trained on.

`examples/pimcore-tools.json` is what evals and the browser demo register. When
it drifted from the corpus — 41 of 49 tools with a shortened description, 8
with none — executable accuracy on the Studio dev split read 30.2% instead of
55.4%, with the *same weights*. Nothing failed; the model simply appeared to be
much worse than it was, and the symptoms (wrong tools, unfillable arguments, a
flood of ASK) pointed convincingly at undertraining.

The canonical Tool ABI renders `DESC <normalized description>` capped at 200
characters and the tool head reads schema states directly, so a description
that differs between training and serving is a silent, unbounded accuracy loss.
This pins it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "examples" / "pimcore-tools.json"
CORPUS = REPO / "specs" / "training"


def corpus_tools() -> dict[str, dict]:
    found: dict[str, dict] = {}
    for split in ("train", "dev"):
        path = CORPUS / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            for cand in json.loads(line).get("candidates", []):
                if isinstance(cand, dict) and "name" in cand:
                    found.setdefault(cand["name"], cand)
    return found


@pytest.mark.skipif(not (CORPUS / "dev.jsonl").exists(), reason="Studio corpus absent")
def test_registry_schemas_match_the_training_corpus():
    corpus = corpus_tools()
    assert corpus, "no embedded candidate schemas in specs/training"
    registry = {t["name"]: t for t in json.loads(REGISTRY.read_text())}

    missing = sorted(set(corpus) - set(registry))
    assert not missing, f"registry is missing tools the model trained on: {missing}"

    drifted = []
    for name, schema in corpus.items():
        if registry[name] != schema:
            got = registry[name].get("description", "")
            want = schema.get("description", "")
            drifted.append(f"{name} (description {len(got)} chars, corpus has {len(want)})")
    assert not drifted, (
        "served schemas differ from the trained ones — the model will see "
        "different canonical text than it was trained on:\n  "
        + "\n  ".join(drifted[:10])
        + "\nresync: uv run python -m tools.extract_registry"
    )


@pytest.mark.skipif(not (CORPUS / "dev.jsonl").exists(), reason="Studio corpus absent")
def test_no_tool_is_served_without_a_description():
    """An empty DESC line is the degenerate case of the same bug, and the one
    that is hardest to notice by eye in a 49-entry JSON file."""
    empty = [
        t["name"]
        for t in json.loads(REGISTRY.read_text())
        if not (t.get("description") or "").strip()
    ]
    assert not empty, f"tools served with no description: {empty}"
