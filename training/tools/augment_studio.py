"""Augment the Studio corpus where the acceptance suite says it is thin.

`eval/usecase/run.py` fails in three specific places, and each maps to a
shortage in the data rather than a modelling gap:

  robust-*       a typo or a politeness prefix flips the decision
  adversarial-*  namespace traps and mention-only phrasing fool it (~40 each)
  ask-*          an argument with no source is not recognized

So this adds, span-correctly:

- **surface noise** — politeness prefixes/suffixes, filler, casing changes and
  keyboard-adjacent typos. Every `char_span` and every `element_spans` entry is
  shifted by the inserted prefix length, and typos are only ever applied to
  text OUTSIDE a span, so recorded surfaces stay exact.
- **upsampling** of the adversarial tags, with different noise per copy so the
  model sees the trap in several surface forms rather than the same sentence
  repeatedly.

Everything is re-validated against `DatasetExample` and the spans are checked
against the utterance before being written; a row that fails is dropped, not
repaired.

Run: uv run python -m tools.augment_studio --out data/studio-aug
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from datasets.schema import DatasetExample

# Politeness / filler that real users type, per language.
PREFIXES = {
    "en": ["hi, ", "hey, ", "please ", "could you ", "quick one: ", "sorry, one more thing: "],
    "de": ["hallo, ", "hey, ", "bitte ", "kannst du ", "kurze frage: ", "sorry, noch was: "],
    "fr": ["bonjour, ", "salut, ", "peux-tu ", "petite question : ", "s'il te plaît "],
    "es": ["hola, ", "oye, ", "por favor ", "una cosa: ", "¿puedes "],
}
SUFFIXES = {
    "en": [" please", " thanks", " if you can", ""],
    "de": [" bitte", " danke", " wenn's geht", ""],
    "fr": [" s'il te plaît", " merci", ""],
    "es": [" por favor", " gracias", ""],
}

# Keyboard-adjacent substitutions, plus transposition/dropping handled below.
ADJACENT = {
    "a": "sq", "e": "wr", "i": "ou", "o": "ip", "u": "yi", "s": "ad",
    "t": "ry", "n": "bm", "r": "et", "l": "k", "c": "vx", "d": "sf",
}

ADVERSARIAL_TAGS = {
    "namespace_trap", "source_conflict", "mention_only", "ambiguous_namespace",
    "type_conflict", "not_found", "family_dependent_symbol",
    "contradicts_search_assets_description", "over_max_clamped", "enum_from_context",
}


def protected_ranges(ex: dict) -> list[tuple[int, int]]:
    """Character ranges that must survive untouched (recorded surfaces)."""
    out = []
    for a in ex["gold"].get("arguments", []):
        if "char_span" in a:
            out.append((a["char_span"]["start"], a["char_span"]["end"]))
        for el in a.get("element_spans", []):
            out.append((el["char_span"]["start"], el["char_span"]["end"]))
    return out


def shift_spans(ex: dict, delta: int) -> None:
    """Move every recorded span by `delta` (a prefix was prepended)."""
    for a in ex["gold"].get("arguments", []):
        if "char_span" in a:
            a["char_span"]["start"] += delta
            a["char_span"]["end"] += delta
        for el in a.get("element_spans", []):
            el["char_span"]["start"] += delta
            el["char_span"]["end"] += delta


def typo(text: str, protected: list[tuple[int, int]], rng: random.Random) -> str:
    """One keyboard-plausible typo outside every protected range."""
    candidates = [
        i
        for i, ch in enumerate(text)
        if ch.lower() in ADJACENT and not any(s <= i < e for s, e in protected)
    ]
    if not candidates:
        return text
    i = rng.choice(candidates)
    ch = text[i].lower()
    mode = rng.random()
    if mode < 0.45:  # adjacent-key substitution
        return text[:i] + rng.choice(ADJACENT[ch]) + text[i + 1 :]
    if mode < 0.75 and i + 1 < len(text) and not any(
        s <= i + 1 < e for s, e in protected
    ):  # transposition
        return text[:i] + text[i + 1] + text[i] + text[i + 2 :]
    return text[:i] + text[i + 1 :]  # dropped character


def noisy_variant(ex: dict, rng: random.Random, tag: str) -> dict | None:
    out = json.loads(json.dumps(ex))
    text = out["utterance"]
    lang = out["lang"]

    if rng.random() < 0.65:
        prefix = rng.choice(PREFIXES[lang])
        text = prefix + text
        shift_spans(out, len(prefix))
    if rng.random() < 0.45:
        text = text + rng.choice(SUFFIXES[lang])
    protected = protected_ranges(out)
    if rng.random() < 0.5:
        text = typo(text, protected, rng)
        # A typo changes lengths; only keep it when every span still matches.
    if rng.random() < 0.25:
        # Casing noise, applied only outside protected ranges.
        chars = list(text)
        for i, ch in enumerate(chars):
            if not any(s <= i < e for s, e in protected):
                chars[i] = ch.upper() if rng.random() < 0.12 else ch
        text = "".join(chars)

    out["utterance"] = text
    out["tags"] = sorted(set(out.get("tags", []) + [tag]))
    out["id"] = "aug-" + hashlib.sha256(
        json.dumps(out, sort_keys=True).encode()
    ).hexdigest()[:14]

    # Spans must still select exactly what they claim, or the row is useless.
    for a in out["gold"].get("arguments", []):
        if "char_span" in a and a.get("surface") is not None:
            s, e = a["char_span"]["start"], a["char_span"]["end"]
            if out["utterance"][s:e] != a["surface"]:
                return None
        for el in a.get("element_spans", []):
            s, e = el["char_span"]["start"], el["char_span"]["end"]
            if out["utterance"][s:e] != el["surface"]:
                return None
    try:
        DatasetExample.model_validate(out)
    except Exception:  # noqa: BLE001
        return None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/studio"))
    parser.add_argument("--out", type=Path, default=Path("data/studio-aug"))
    parser.add_argument("--noise-rate", type=float, default=0.45,
                        help="fraction of train rows to add a noisy copy of")
    parser.add_argument("--adversarial-copies", type=int, default=4,
                        help="extra noisy copies per adversarial-tagged row")
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    # dev/test stay pristine: augmenting them would measure the augmenter.
    for split in ("dev", "test"):
        src = args.src / f"{split}.jsonl"
        if src.exists():
            shutil.copy(src, args.out / f"{split}.jsonl")

    rows = [json.loads(line) for line in (args.src / "train.jsonl").read_text().splitlines() if line.strip()]
    out_rows = list(rows)
    stats = {"base": len(rows), "noise": 0, "adversarial": 0, "dropped": 0}

    for ex in rows:
        tags = set(ex.get("tags", []))
        if rng.random() < args.noise_rate:
            v = noisy_variant(ex, rng, "surface_noise")
            if v:
                out_rows.append(v)
                stats["noise"] += 1
            else:
                stats["dropped"] += 1
        if tags & ADVERSARIAL_TAGS:
            for _ in range(args.adversarial_copies):
                v = noisy_variant(ex, rng, "adversarial_upsample")
                if v:
                    out_rows.append(v)
                    stats["adversarial"] += 1
                else:
                    stats["dropped"] += 1

    seen: set[str] = set()
    with (args.out / "train.jsonl").open("w") as f:
        for ex in out_rows:
            if ex["id"] in seen:
                continue
            seen.add(ex["id"])
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    stats["total"] = len(seen)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
