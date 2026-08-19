"""Stage-0 backbone init (plan §Stage 0, mini-hardware scale): prune the
pretrained multilingual MiniLM backbone's 250k SentencePiece vocab to ~32k
and map its weights onto `ntc_encoder_heads_v1`.

Outputs:
- contracts/tokenizer-any/tokenizer.json   — frozen pruned tokenizer
- fixtures/tokenizer-any/vectors.jsonl     — golden ids+byte offsets (Rust-pinned)
- runs/backbone/init.pt                    — partial state_dict in OUR module
  names: pruned word embeddings, position embeddings (token-type row folded),
  embedding LayerNorm, all 12 encoder layers, schema layers warm-started from
  layers 0..N-1.

The pruned tokenizer decomposes ANY word of the target languages into known
subwords (top-K pieces by unigram score + corpus pieces + Latin single
chars); truly uncovered characters fall back to <unk>.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
BACKBONE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

VECTOR_TEXTS = [
    "Drehe das Licht im Wohnzimmer ab!",
    "Mach das Licht in der Küche an.",
    "schedule a dentist appointment tomorrow afternoon for one and a half hours",
    "Book a quick chat with the plumber next Friday!",
    "Éteins la lumière de la chambre, s'il te plaît.",
    "Apaga la luz del dormitorio, por favor.",
    "SET A TIMER FOR 90 MINUTES",
    "wie wird das wetter übermorgen in Köln?",
    "anna.mueller@example.com https://example.com/docs?q=1",
    "2026-08-19T15:00:00+02:00",
    "TOOL 7 DESC search available train journeys",
    "café münchen día señor Straßenbahn",
]


def probe_corpus(data_dir: Path) -> list[str]:
    corpus: list[str] = []
    for split in ("train", "dev"):
        for line in (data_dir / f"{split}.jsonl").read_text().splitlines():
            ex = json.loads(line)
            corpus.append(ex["utterance"])
    # Canonical schema texts (keywords + descriptions) via the Rust renderer.
    from ntc_model.packing import Canonicalizer

    tools: dict[str, dict] = {}
    for line in (data_dir / "train.jsonl").read_text().splitlines():
        for tool in json.loads(line)["candidates"]:
            tools.setdefault(json.dumps(tool, sort_keys=True), tool)
    canon = Canonicalizer()
    schemas = list(tools.values())
    for index in range(4):
        corpus += [r["text"] for r in canon.canonicalize(schemas, [index] * len(schemas))]
    corpus += VECTOR_TEXTS
    return corpus


def keep_latin_single(piece: str) -> bool:
    if len(piece) != 1 and not (len(piece) == 2 and piece.startswith("▁")):
        return False
    ch = piece[-1]
    if ch == "▁":
        return True
    cat = unicodedata.category(ch)
    if cat.startswith(("P", "N", "S")):
        return True
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return "LATIN" in name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mini"))
    parser.add_argument("--top-k", type=int, default=32000)
    parser.add_argument("--schema-layers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("runs/backbone"))
    args = parser.parse_args()

    hf_tok = AutoTokenizer.from_pretrained(BACKBONE)
    full = Tokenizer.from_str(hf_tok._tokenizer.to_str())
    data = json.loads(hf_tok._tokenizer.to_str())
    vocab: list[list] = data["model"]["vocab"]  # [piece, score], ids = position

    # ---- select pieces -------------------------------------------------
    keep = set(range(4))  # <s> <pad> </s> <unk> keep ids 0..3
    ranked = sorted(range(4, len(vocab)), key=lambda i: vocab[i][1], reverse=True)
    keep.update(ranked[: args.top_k])
    for i in range(4, len(vocab)):
        if keep_latin_single(vocab[i][0]):
            keep.add(i)
    corpus = probe_corpus(args.data)
    for text in corpus:
        keep.update(full.encode(text).ids)
    keep.discard(250001)  # <mask> (MLM-only)

    old_ids = sorted(keep)
    new_vocab = [vocab[i] for i in old_ids]
    id_map = {old: new for new, old in enumerate(old_ids)}
    assert [id_map[i] for i in range(4)] == [0, 1, 2, 3], "specials must keep ids 0..3"
    print(f"vocab: {len(vocab)} -> {len(new_vocab)} pieces")

    # ---- rebuild tokenizer.json ---------------------------------------
    data["model"]["vocab"] = new_vocab
    data["added_tokens"] = [t for t in data["added_tokens"] if t["content"] != "<mask>"]
    pruned = Tokenizer.from_str(json.dumps(data))

    # Coverage check: token inflation vs the full tokenizer on the corpus.
    full_count = sum(len(full.encode(t).ids) for t in corpus)
    pruned_count = sum(len(pruned.encode(t).ids) for t in corpus)
    unk_id = 3
    unks = sum(pruned.encode(t).ids.count(unk_id) for t in corpus)
    print(f"tokens on corpus: full={full_count} pruned={pruned_count} "
          f"(+{100 * (pruned_count / full_count - 1):.2f}%), <unk> count={unks}")

    out_tok = REPO / "contracts" / "tokenizer-any" / "tokenizer.json"
    out_tok.parent.mkdir(parents=True, exist_ok=True)
    out_tok.write_text(pruned.to_str(pretty=False))
    print(f"froze {out_tok}")

    vec_path = REPO / "fixtures" / "tokenizer-any" / "vectors.jsonl"
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    with vec_path.open("w") as f:
        for text in VECTOR_TEXTS:
            enc = pruned.encode(text)
            cum = [0]
            for ch in text:
                cum.append(cum[-1] + len(ch.encode("utf-8")))
            offsets = [(cum[s], cum[e]) for s, e in enc.offsets]
            f.write(json.dumps({"text": text, "ids": list(enc.ids), "offsets": offsets},
                               ensure_ascii=False) + "\n")
    print(f"wrote {vec_path}")

    # ---- backbone weights → our module names --------------------------
    model = AutoModel.from_pretrained(BACKBONE)
    sd = model.state_dict()
    rows = torch.tensor(old_ids, dtype=torch.long)
    init: dict[str, torch.Tensor] = {}
    init["word_emb.weight"] = sd["embeddings.word_embeddings.weight"][rows].clone()
    init["pos_emb.weight"] = (
        sd["embeddings.position_embeddings.weight"]
        + sd["embeddings.token_type_embeddings.weight"][0][None, :]
    ).clone()
    init["emb_norm.weight"] = sd["embeddings.LayerNorm.weight"].clone()
    init["emb_norm.bias"] = sd["embeddings.LayerNorm.bias"].clone()

    def layer(dst: str, i: int) -> None:
        src = f"encoder.layer.{i}"
        m = {
            f"{dst}.attn.q": f"{src}.attention.self.query",
            f"{dst}.attn.k": f"{src}.attention.self.key",
            f"{dst}.attn.v": f"{src}.attention.self.value",
            f"{dst}.attn.o": f"{src}.attention.output.dense",
            f"{dst}.attn_norm": f"{src}.attention.output.LayerNorm",
            f"{dst}.ffn_up": f"{src}.intermediate.dense",
            f"{dst}.ffn_down": f"{src}.output.dense",
            f"{dst}.ffn_norm": f"{src}.output.LayerNorm",
        }
        for d, s in m.items():
            init[f"{d}.weight"] = sd[f"{s}.weight"].clone()
            init[f"{d}.bias"] = sd[f"{s}.bias"].clone()

    n_layers = model.config.num_hidden_layers
    for i in range(n_layers):
        layer(f"encoder_layers.{i}", i)
    for j in range(args.schema_layers):
        layer(f"schema_layers.{j}", j)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": init}, out / "init.pt")
    meta = {
        "backbone": BACKBONE,
        "vocab": len(new_vocab),
        "hidden": model.config.hidden_size,
        "heads": model.config.num_attention_heads,
        "ffn": model.config.intermediate_size,
        "encoder_layers": n_layers,
        "layer_norm_eps": model.config.layer_norm_eps,
        "max_positions": model.config.max_position_embeddings,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out / 'init.pt'} + meta.json: {meta}")


if __name__ == "__main__":
    main()
