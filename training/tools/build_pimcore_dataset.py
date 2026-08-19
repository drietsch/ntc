"""Assemble the Pimcore POC dataset from all sources:

  - live teacher shards (data/live/pimcore-*.jsonl, delegate-*.jsonl),
  - deterministic DELEGATE templates (data/delegate/),
  - deterministic CALL/ASK/NO_CALL templates over the real Pimcore tools
    (generated here: id/parent/query/tag patterns that the extracted schemas
    actually support, in EN/DE/FR/ES).

Splits: examples tagged `test` stay test; everything else is split
deterministically by id hash (85/7.5/7.5 train/dev/test).

Run: uv run python -m tools.build_pimcore_dataset --out data/pimcore
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

CLUSTERS = [
    ["search_assets", "get_asset", "list_assets", "upload_asset"],
    ["search_data_objects", "create_data_object", "propose_data_object_update", "list_classes"],
    ["search_documents", "create_document", "list_document_types", "get_document_schema"],
    ["assign_tag", "unassign_tag", "create_tag", "list_tags"],
    ["list_workflows", "get_workflow_places", "list_elements_by_workflow_state", "search_data_objects"],
]

# (lang) -> list of (utterance template, tool, [(param, semantic_type, slot)])
# `{q}` = search term slot, `{id}` = numeric id slot, `{tag}` = tag name slot.
CALL_PATTERNS: dict[str, list[tuple[str, str, list[tuple[str, str, str]]]]] = {
    "en": [
        ("find assets matching {q}", "search_assets", [("query", "STRING", "q")]),
        ("search the assets for {q}", "search_assets", [("query", "STRING", "q")]),
        ("show me asset {id}", "get_asset", [("id", "INTEGER", "id")]),
        ("open asset {id}", "get_asset", [("id", "INTEGER", "id")]),
        ("list the tags", "list_tags", []),
        ("which classes exist", "list_classes", []),
        ("look for data objects matching {q}", "search_data_objects", [("query", "STRING", "q")]),
        ("find documents about {q}", "search_documents", [("query", "STRING", "q")]),
        ("create a tag called {tag}", "create_tag", [("name", "STRING", "tag")]),
        ("show the workflows", "list_workflows", []),
    ],
    "de": [
        ("finde assets zu {q}", "search_assets", [("query", "STRING", "q")]),
        ("suche in den assets nach {q}", "search_assets", [("query", "STRING", "q")]),
        ("zeig mir asset {id}", "get_asset", [("id", "INTEGER", "id")]),
        ("öffne asset {id}", "get_asset", [("id", "INTEGER", "id")]),
        ("liste die tags auf", "list_tags", []),
        ("welche klassen gibt es", "list_classes", []),
        ("suche datenobjekte zu {q}", "search_data_objects", [("query", "STRING", "q")]),
        ("finde dokumente über {q}", "search_documents", [("query", "STRING", "q")]),
        ("lege einen tag namens {tag} an", "create_tag", [("name", "STRING", "tag")]),
        ("zeig die workflows", "list_workflows", []),
    ],
    "fr": [
        ("trouve les assets correspondant à {q}", "search_assets", [("query", "STRING", "q")]),
        ("montre-moi l'asset {id}", "get_asset", [("id", "INTEGER", "id")]),
        ("liste les tags", "list_tags", []),
        ("quelles classes existent", "list_classes", []),
        ("cherche des objets de données sur {q}", "search_data_objects", [("query", "STRING", "q")]),
        ("crée un tag nommé {tag}", "create_tag", [("name", "STRING", "tag")]),
    ],
    "es": [
        ("busca assets que coincidan con {q}", "search_assets", [("query", "STRING", "q")]),
        ("muéstrame el asset {id}", "get_asset", [("id", "INTEGER", "id")]),
        ("lista las etiquetas", "list_tags", []),
        ("qué clases existen", "list_classes", []),
        ("busca objetos de datos sobre {q}", "search_data_objects", [("query", "STRING", "q")]),
        ("crea una etiqueta llamada {tag}", "create_tag", [("name", "STRING", "tag")]),
    ],
}

QUERIES = {
    "en": ["summer campaign", "product sheets", "hero banner", "invoice 2025"],
    "de": ["sommerkampagne", "produktblätter", "hero banner", "rechnung 2025"],
    "fr": ["campagne d'été", "fiches produit", "bannière"],
    "es": ["campaña de verano", "fichas de producto", "banner"],
}
TAGS = {
    "en": ["summer", "archive", "press", "oldtimer"],
    "de": ["sommer", "archiv", "presse", "oldtimer"],
    "fr": ["été", "archive", "presse"],
    "es": ["verano", "archivo", "prensa"],
}
IDS = ["12", "42", "128", "812", "1503"]

ASK_PATTERNS = {
    "en": [("show me the asset", "get_asset", "id"), ("create a tag", "create_tag", "name")],
    "de": [("zeig mir das asset", "get_asset", "id"), ("lege einen tag an", "create_tag", "name")],
    "fr": [("montre-moi l'asset", "get_asset", "id"), ("crée un tag", "create_tag", "name")],
    "es": [("muéstrame el asset", "get_asset", "id"), ("crea una etiqueta", "create_tag", "name")],
}

NO_CALL = {
    "en": [
        "what does search_assets actually do?",
        "thanks, that's all for today",
        "can pimcore send an sms to a customer?",
        "how is the tag tool different from properties?",
    ],
    "de": [
        "was macht search_assets eigentlich?",
        "danke, das war's für heute",
        "kann pimcore eine sms an kunden schicken?",
    ],
    "fr": ["à quoi sert search_assets ?", "merci, c'est tout", "pimcore peut-il envoyer un sms ?"],
    "es": ["¿qué hace search_assets?", "gracias, eso es todo", "¿pimcore puede enviar un sms?"],
}


def make_id(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def assemble(lang: str, utterance: str, tool: str | None, arguments: list[dict],
             unresolved: list[dict], action: str, rng: random.Random, tags: list[str]) -> dict:
    cluster = next((c for c in CLUSTERS if tool in c), None) if tool else None
    cluster = cluster or rng.choice(CLUSTERS)
    names = list(cluster)
    if tool and tool not in names:
        names[0] = tool
    candidates = [TOOLS[n] for n in names]
    rng.shuffle(candidates)
    ex = {
        "id": "", "lang": lang, "utterance": utterance, "candidates": candidates,
        "gold": {"action": action, "tool": tool, "arguments": arguments,
                 "unresolved": unresolved},
        "split": "train", "tags": tags,
    }
    ex["id"] = make_id(ex)
    return ex


def templates(seed: int, repeats: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    for lang in ("en", "de", "fr", "es"):
        for _ in range(repeats):
            for tpl, tool, slots in CALL_PATTERNS[lang]:
                fills = {"q": rng.choice(QUERIES[lang]), "id": rng.choice(IDS),
                         "tag": rng.choice(TAGS[lang])}
                text, args = tpl, []
                for param, sem, slot in slots:
                    value = fills[slot]
                    start = tpl.index("{" + slot + "}")
                    # Build the utterance progressively so spans stay exact.
                    text = tpl.format(**fills)
                    start = text.index(value, max(0, start - 20))
                    args.append({
                        "parameter": param, "semantic_type": sem,
                        "value": int(value) if sem == "INTEGER" else value,
                        "char_span": {"start": start, "end": start + len(value)},
                        "surface": value,
                    })
                if not slots:
                    text = tpl
                out.append(assemble(lang, text, tool, args, [], "CALL", rng, ["call"]))
            for text, tool, missing in ASK_PATTERNS[lang]:
                out.append(assemble(lang, text, tool, [],
                                    [{"parameter": missing, "reason": "MISSING"}], "ASK", rng, ["ask"]))
            for text in NO_CALL[lang]:
                out.append(assemble(lang, text, None, [], [], "NO_CALL", rng, ["no_call"]))
    return out


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/pimcore"))
    parser.add_argument("--live", type=Path, default=Path("data/live"))
    parser.add_argument("--delegate", type=Path, default=Path("data/delegate"))
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    rows: list[dict] = templates(args.seed, args.repeats)
    for pattern in ("pimcore-*.jsonl", "delegate-*.jsonl"):
        for shard in sorted(args.live.glob(pattern)):
            for ex in load_jsonl(shard):
                ex.setdefault("tags", []).append("live_teacher")
                ex["tags"] = sorted(set(ex["tags"]))
                rows.append(ex)
    for split_file in ("train.jsonl", "test.jsonl"):
        path = args.delegate / split_file
        if path.exists():
            rows.extend(load_jsonl(path))

    # Validate, dedup, split.
    seen: set[str] = set()
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    dropped = 0
    for ex in rows:
        try:
            DatasetExample.model_validate(ex)
        except Exception:  # noqa: BLE001 — teacher output can be imperfect
            dropped += 1
            continue
        if ex["id"] in seen:
            continue
        seen.add(ex["id"])
        if ex.get("split") == "test":
            split = "test"
        else:
            # Teacher ids are arbitrary strings; hash them for a stable split.
            bucket = int(hashlib.sha256(ex["id"].encode()).hexdigest()[:8], 16) % 100
            split = "train" if bucket < 85 else ("dev" if bucket < 93 else "test")
        ex["split"] = split
        splits[split].append(ex)

    args.out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {"dropped_invalid": dropped}
    for split, items in splits.items():
        with (args.out / f"{split}.jsonl").open("w") as f:
            for ex in items:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        stats[split] = len(items)
        for ex in items:
            stats[f"{split}:{ex['gold']['action']}"] = stats.get(f"{split}:{ex['gold']['action']}", 0) + 1
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
