"""Train and freeze the NTC-Web tokenizer (milestone B1, mini scale).

Trains a Unigram (SentencePiece-style) tokenizer on the mini corpus
(utterances in EN/DE/FR/ES + canonical schema texts + numeric/date/identifier
strings per spec §38), then:
- writes the frozen artifact to contracts/tokenizer/tokenizer.json,
- writes golden vectors (ids + BYTE offsets, the Rust offset space) to
  fixtures/tokenizer/vectors.jsonl for the cross-implementation parity test.

Padding id 0 is a real token (`<pad>`), matching both packers' zero-padding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers

from ntc_model.packing import Canonicalizer

REPO = Path(__file__).resolve().parents[2]

SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]

EXTRA_STRINGS = [
    "90", "1.5", "1,5", "3", "15:00", "2026-08-19", "2026-08-19T15:00:00+02:00",
    "anna.mueller@example.com", "https://example.com/docs", "fn_193 arg_0 arg_1",
    "TOOL 0", "TOOL 15", "DESC", "ARG", "INFO", "TYPE", "REQUIRED", "SEMANTIC", "ENUM",
    "TEXT", "INTEGER", "FLOAT", "BOOLEAN", "DATE", "DATETIME", "DURATION",
    "PERSON", "LOCATION", "REQUIRED 1", "REQUIRED 0",
]

VECTOR_TEXTS = [
    "make a dentist appointment tomorrow afternoon",
    "Mach morgen Nachmittag einen einstündigen Zahnarzttermin.",
    "schick eine e-mail an anna müller über das budget",
    "planifie un rendez-vous chez le dentiste demain après-midi",
    "programa una cita con el dentista mañana por la tarde",
    "set a timer for 90 minutes",
    "stell einen timer auf eineinhalb stunden",
    "what's the weather in münchen for the next 3 days",
    "turn off the light in the living room",
    "éteins la lumière dans la chambre",
    "wie wird das wetter in köln",
    "  leading and   multiple   spaces  ",
    "anna.mueller@example.com",
    "https://example.com/docs?q=1",
    "2026-08-19T15:00:00+02:00",
    "TOOL 7",
    "café münchen köln día señor",
]


def build_corpus(data_dir: Path) -> list[str]:
    corpus: list[str] = []
    tools_seen: dict[str, dict] = {}
    for split in ("train", "dev"):
        path = data_dir / f"{split}.jsonl"
        for line in path.read_text().splitlines():
            ex = json.loads(line)
            corpus.append(ex["utterance"])
            for tool in ex["candidates"]:
                tools_seen.setdefault(json.dumps(tool, sort_keys=True), tool)

    canon = Canonicalizer()
    schemas = list(tools_seen.values())
    for index in (0, 1, 2, 3):
        for result in canon.canonicalize(schemas, [index] * len(schemas)):
            corpus.append(result["text"])
    corpus.extend(EXTRA_STRINGS * 4)
    return corpus


def train(corpus: list[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        unk_token="<unk>",
    )
    tok.train_from_iterator(corpus, trainer)
    bos, eos = tok.token_to_id("<s>"), tok.token_to_id("</s>")
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B </s>",
        special_tokens=[("<s>", bos), ("</s>", eos)],
    )
    return tok


def char_to_byte_offsets(text: str, offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Python tokenizers reports char offsets; Rust reports byte offsets."""
    cum = [0]
    for ch in text:
        cum.append(cum[-1] + len(ch.encode("utf-8")))
    return [(cum[s], cum[e]) for s, e in offsets]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mini"))
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()

    corpus = build_corpus(args.data)
    print(f"corpus: {len(corpus)} lines")
    tok = train(corpus, args.vocab_size)
    print(f"vocab size: {tok.get_vocab_size()}")
    assert tok.token_to_id("<pad>") == 0, "padding id must be 0"

    out = REPO / "contracts" / "tokenizer" / "tokenizer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tok.to_str(pretty=False))
    print(f"froze {out}")

    vectors_path = REPO / "fixtures" / "tokenizer" / "vectors.jsonl"
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    with vectors_path.open("w") as f:
        for text in VECTOR_TEXTS:
            enc = tok.encode(text)
            byte_offsets = char_to_byte_offsets(text, enc.offsets)
            f.write(
                json.dumps(
                    {"text": text, "ids": list(enc.ids), "offsets": byte_offsets},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {vectors_path} ({len(VECTOR_TEXTS)} vectors)")


if __name__ == "__main__":
    main()
