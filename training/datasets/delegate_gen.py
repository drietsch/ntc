"""Deterministic DELEGATE examples: requests that need a full LLM agent, not
a single typed call.

Composed from per-language patterns over the real Pimcore tool set so the
router learns the *shape* of agent work rather than keywords:
  - chained steps whose later stage depends on earlier results,
  - bulk mutation over a filtered/unknown result set,
  - conditional/comparative logic,
  - open-ended authoring / analysis / explanation.

The reference case (user-provided) is included verbatim per language.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets.schema import DatasetExample

REPO = Path(__file__).resolve().parents[2]

# Realistic Pimcore subject matter per language: (entity plural, attribute
# phrase, threshold phrase, mutation phrase).
SUBJECTS = {
    "en": [
        ("blue cars", "building year", "1976 and before", "activate the oldtimer field"),
        ("product images", "resolution", "under 800 pixels", "tag them as low-res"),
        ("PDF datasheets", "last modified date", "older than 2020", "move them to the archive folder"),
        ("draft pages", "publication state", "still unpublished", "set them to review"),
        ("category objects", "product count", "empty", "mark them as deprecated"),
    ],
    "de": [
        ("blaue autos", "baujahr", "1976 und älter", "das oldtimer-feld aktivieren"),
        ("produktbilder", "auflösung", "unter 800 pixel", "sie als low-res taggen"),
        ("pdf-datenblätter", "änderungsdatum", "älter als 2020", "sie ins archiv verschieben"),
        ("entwurfsseiten", "status", "noch unveröffentlicht", "sie auf review setzen"),
        ("kategorie-objekte", "produktanzahl", "leer", "sie als veraltet markieren"),
    ],
    "fr": [
        ("voitures bleues", "année de construction", "1976 ou avant", "activer le champ oldtimer"),
        ("images produit", "résolution", "sous 800 pixels", "les taguer basse résolution"),
        ("fiches PDF", "date de modification", "antérieure à 2020", "les déplacer vers les archives"),
        ("pages brouillon", "statut", "non publiées", "les passer en révision"),
    ],
    "es": [
        ("coches azules", "año de fabricación", "1976 o anterior", "activar el campo oldtimer"),
        ("imágenes de producto", "resolución", "menos de 800 píxeles", "etiquetarlas como baja resolución"),
        ("fichas PDF", "fecha de modificación", "anterior a 2020", "moverlas al archivo"),
        ("páginas borrador", "estado", "sin publicar", "ponerlas en revisión"),
    ],
}

# Chained search → filter → mutate.
CHAIN = {
    "en": "Search for all {ent} in Pimcore assets. Then take just the ones whose {attr} is {thr} and {mut}.",
    "de": "Suche alle {ent} in den Pimcore-Assets. Nimm dann nur die, deren {attr} {thr} ist, und {mut}.",
    "fr": "Cherche tous les {ent} dans les assets Pimcore, puis prends seulement ceux dont {attr} est {thr} et {mut}.",
    "es": "Busca todos los {ent} en los assets de Pimcore y luego, solo con los que tengan {attr} {thr}, {mut}.",
}

BULK = {
    "en": [
        "Find every {ent} and {mut} for all of them.",
        "Go through all {ent} and {mut} where the {attr} is {thr}.",
        "For each of the {ent} with {attr} {thr}, {mut}.",
    ],
    "de": [
        "Finde alle {ent} und {mut} bei allen davon.",
        "Geh alle {ent} durch und {mut}, wenn das {attr} {thr} ist.",
        "Für jedes der {ent} mit {attr} {thr} bitte {mut}.",
    ],
    "fr": [
        "Trouve tous les {ent} et {mut} pour chacun.",
        "Parcours les {ent} dont {attr} est {thr} et {mut}.",
    ],
    "es": [
        "Encuentra todos los {ent} y {mut} en cada uno.",
        "Revisa los {ent} cuyo {attr} sea {thr} y {mut}.",
    ],
}

COMPARE = {
    "en": [
        "Which {ent} have the largest file size, and why are they not published?",
        "Compare the {ent} in the marketing folder and tell me which ones need new metadata.",
        "Check whether any {ent} are missing alt text and fix the worst offenders.",
    ],
    "de": [
        "Welche {ent} sind am größten und warum sind sie nicht veröffentlicht?",
        "Vergleiche die {ent} im Marketing-Ordner und sag mir, welche neue Metadaten brauchen.",
        "Prüfe, ob {ent} ohne Alt-Text existieren, und korrigiere die schlimmsten.",
    ],
    "fr": [
        "Quels {ent} sont les plus volumineux et pourquoi ne sont-ils pas publiés ?",
        "Compare les {ent} du dossier marketing et dis-moi lesquels ont besoin de métadonnées.",
    ],
    "es": [
        "¿Qué {ent} ocupan más espacio y por qué no están publicados?",
        "Compara los {ent} de la carpeta de marketing y dime cuáles necesitan metadatos.",
    ],
}

AUTHOR = {
    "en": [
        "Write a product description for the new espresso machine and publish it as a page.",
        "Summarize what changed in the product catalog this week.",
        "Explain why the checkout page is stuck in the review workflow and unblock it.",
        "Draft SEO metadata for every unpublished landing page.",
    ],
    "de": [
        "Schreibe eine Produktbeschreibung für die neue Espressomaschine und veröffentliche sie als Seite.",
        "Fasse zusammen, was sich diese Woche im Produktkatalog geändert hat.",
        "Erkläre, warum die Checkout-Seite im Review-Workflow hängt, und behebe es.",
        "Entwirf SEO-Metadaten für alle unveröffentlichten Landingpages.",
    ],
    "fr": [
        "Rédige une description produit pour la nouvelle machine à espresso et publie-la.",
        "Résume les changements du catalogue produits cette semaine.",
        "Explique pourquoi la page de paiement est bloquée dans le workflow.",
    ],
    "es": [
        "Redacta una descripción de producto para la nueva cafetera y publícala.",
        "Resume qué cambió en el catálogo de productos esta semana.",
        "Explica por qué la página de pago está bloqueada en el flujo de trabajo.",
    ],
}

CLUSTERS = [
    ["search_assets", "get_asset", "list_assets", "upload_asset"],
    ["search_data_objects", "create_data_object", "propose_data_object_update", "list_classes"],
    ["search_documents", "create_document", "list_document_types", "get_document_schema"],
    ["assign_tag", "unassign_tag", "create_tag", "list_tags"],
    ["list_workflows", "get_workflow_places", "list_elements_by_workflow_state", "search_data_objects"],
]


def utterances(lang: str, rng: random.Random) -> list[str]:
    out: list[str] = []
    for ent, attr, thr, mut in SUBJECTS[lang]:
        out.append(CHAIN[lang].format(ent=ent, attr=attr, thr=thr, mut=mut))
        for tpl in BULK[lang]:
            out.append(tpl.format(ent=ent, attr=attr, thr=thr, mut=mut))
        for tpl in COMPARE[lang]:
            out.append(tpl.format(ent=ent))
    out.extend(AUTHOR[lang])
    rng.shuffle(out)
    return out


def build(seed: int = 23) -> list[dict]:
    rng = random.Random(seed)
    tools = {t["name"]: t for t in json.loads((REPO / "examples" / "pimcore-tools.json").read_text())}
    examples: list[dict] = []
    for lang in ("en", "de", "fr", "es"):
        for i, utterance in enumerate(utterances(lang, rng)):
            cluster = CLUSTERS[i % len(CLUSTERS)]
            candidates = [tools[n] for n in cluster]
            rng.shuffle(candidates)
            ex = {
                "id": "",
                "lang": lang,
                "utterance": utterance,
                "candidates": candidates,
                "gold": {"action": "DELEGATE", "tool": None, "arguments": [], "unresolved": []},
                # Hold a slice out for honest eval.
                "split": "train" if i % 5 else "test",
                "tags": ["delegate"],
            }
            ex["id"] = hashlib.sha256(
                json.dumps(ex, sort_keys=True).encode()
            ).hexdigest()[:16]
            DatasetExample.model_validate(ex)
            examples.append(ex)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/delegate"))
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    examples = build(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with (args.out / "train.jsonl").open("w") as tr, (args.out / "test.jsonl").open("w") as te:
        for ex in examples:
            (tr if ex["split"] == "train" else te).write(
                json.dumps(ex, ensure_ascii=False) + "\n"
            )
            counts[ex["split"]] = counts.get(ex["split"], 0) + 1
            counts[f"lang:{ex['lang']}"] = counts.get(f"lang:{ex['lang']}", 0) + 1
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
