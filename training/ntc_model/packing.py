"""Input packing — the Python mirror of `crates/ntc-model/src/inputs.rs`.

Canonical text and its line/anchor metadata come from the Rust single
implementation via `ntc schemac` (never re-rendered here). Anchor discovery
follows the same rule as the Rust packer: a token anchors a range if it is the
first token whose span intersects it.

Offset spaces: the Rust CLI reports **byte** ranges; the Python `tokenizers`
binding reports **char** offsets. `_byte_to_char_ranges` converts once per
canonical text so both sides pick identical tokens.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ntc_model.config import NtcArchConfig

SEGMENT_SPECIAL = 0
SEGMENT_PAD = 9
SEGMENT_KIND_IDS = {
    "TOOL_HEADER": 1,
    "DESC": 2,
    "ARG_NAME": 3,
    "INFO": 4,
    "TYPE": 5,
    "REQUIRED": 6,
    "SEMANTIC": 7,
    "ENUM_VALUE": 8,
    "ITEM": 10,
}

DEFAULT_NTC_BIN = Path(__file__).resolve().parents[2] / "target" / "debug" / "ntc"


class Canonicalizer:
    """Batch access to the Rust canonical renderer (`ntc schemac`)."""

    def __init__(self, ntc_bin: str | Path = DEFAULT_NTC_BIN):
        self.ntc_bin = str(ntc_bin)

    def canonicalize(self, schemas: list[dict], indices: list[int]) -> list[dict]:
        """Each result: {id, abi_version, index, tool, text, lines}."""
        assert len(schemas) == len(indices)
        payload = "\n".join(
            json.dumps({"schema": s, "index": i}) for s, i in zip(schemas, indices, strict=False)
        )
        proc = subprocess.run(
            [self.ntc_bin, "schemac"],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def _byte_to_char_ranges(text: str, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Convert byte ranges over `text` (UTF-8) into char ranges."""
    byte_to_char: dict[int, int] = {}
    b = 0
    for ci, ch in enumerate(text):
        byte_to_char[b] = ci
        b += len(ch.encode("utf-8"))
    byte_to_char[b] = len(text)
    return [(byte_to_char[s], byte_to_char[e]) for s, e in ranges]


@dataclass
class PackedTool:
    ids: list[int]
    mask: list[bool]
    kinds: list[int]
    tool_anchor: int
    arg_anchors: list[int]
    enum_anchors: list[list[int]]


def pack_tool(cfg: NtcArchConfig, tokenizer: Tokenizer, canon: dict) -> PackedTool:
    """Mirror of `ModelInputs::pack_tool` for one canonicalized tool."""
    text: str = canon["text"]
    lines: list[dict] = canon["lines"]
    n_args = len(canon["tool"]["args"])
    if n_args > cfg.max_args:
        raise ValueError(f"tool `{canon['id']}` declares {n_args} args > {cfg.max_args}")

    # Byte → char conversion for every line range and anchor range.
    byte_ranges = [tuple(line["range"]) for line in lines]
    anchor_byte = [tuple(line["anchor"]) if line["anchor"] else None for line in lines]
    char_ranges = _byte_to_char_ranges(text, byte_ranges)
    anchor_char = _byte_to_char_ranges(
        text, [r for r in anchor_byte if r is not None]
    )
    anchor_iter = iter(anchor_char)
    anchors = [next(anchor_iter) if r is not None else None for r in anchor_byte]

    enc = tokenizer.encode(text)
    ls = cfg.max_schema_tokens
    if len(enc.ids) > ls:
        raise ValueError(
            f"tool `{canon['id']}`: {len(enc.ids)} tokens > max_schema_tokens {ls}"
        )

    ids = list(enc.ids) + [0] * (ls - len(enc.ids))
    mask = [True] * len(enc.ids) + [False] * (ls - len(enc.ids))
    kinds = [SEGMENT_PAD] * ls
    for i, (s, e) in enumerate(enc.offsets):
        if e <= s:
            kinds[i] = SEGMENT_SPECIAL
            continue
        kinds[i] = next(
            (
                SEGMENT_KIND_IDS[line["kind"]]
                for line, (cs, ce) in zip(lines, char_ranges, strict=False)
                if cs <= s < ce
            ),
            SEGMENT_SPECIAL,
        )

    def find_anchor(char_range: tuple[int, int]) -> int:
        for i, (s, e) in enumerate(enc.offsets):
            if e > s and s < char_range[1] and e > char_range[0]:
                return i
        raise ValueError(f"tool `{canon['id']}`: no token anchor for range {char_range}")

    tool_anchor = next(i for i, (s, e) in enumerate(enc.offsets) if e > s)

    arg_anchors: list[int] = []
    enum_anchors: list[list[int]] = []
    for k, arg in enumerate(canon["tool"]["args"]):
        name_line_idx = next(
            i
            for i, line in enumerate(lines)
            if line["kind"] == "ARG_NAME" and line["arg_index"] == k
        )
        arg_anchors.append(find_anchor(anchors[name_line_idx]))
        evs: list[int] = []
        for j in range(len(arg.get("enum_values") or [])):
            line_idx = next(
                i
                for i, line in enumerate(lines)
                if line["kind"] == "ENUM_VALUE"
                and line["arg_index"] == k
                and line["enum_index"] == j
            )
            evs.append(find_anchor(anchors[line_idx]))
        enum_anchors.append(evs)

    return PackedTool(ids, mask, kinds, tool_anchor, arg_anchors, enum_anchors)


def pack_batch(
    cfg: NtcArchConfig,
    tokenizer: Tokenizer,
    utterances: list[str],
    canon_tools_per_example: list[list[dict]],
) -> dict[str, torch.Tensor]:
    """Pack a batch for `NtcEncoderHeadsV1.forward`.

    T is the max tool count in the batch (padded tools are fully masked);
    A/E/Ls/Lu come from the config, matching the Rust runtime's shapes.
    """
    b = len(utterances)
    assert b == len(canon_tools_per_example)
    lu, ls = cfg.max_utterance_tokens, cfg.max_schema_tokens
    a, e = cfg.max_args, cfg.max_enum_values
    t = max(len(tools) for tools in canon_tools_per_example)

    out = {
        "utterance_ids": torch.zeros(b, lu, dtype=torch.long),
        "utterance_mask": torch.zeros(b, lu, dtype=torch.bool),
        "schema_ids": torch.zeros(b, t, ls, dtype=torch.long),
        "schema_mask": torch.zeros(b, t, ls, dtype=torch.bool),
        "schema_kinds": torch.full((b, t, ls), SEGMENT_PAD, dtype=torch.long),
        "tool_count": torch.zeros(b, dtype=torch.long),
        "tool_anchors": torch.zeros(b, t, dtype=torch.long),
        "arg_anchors": torch.zeros(b, t, a, dtype=torch.long),
        "arg_mask": torch.zeros(b, t, a, dtype=torch.bool),
        "enum_anchors": torch.zeros(b, t, a, e, dtype=torch.long),
        "enum_mask": torch.zeros(b, t, a, e, dtype=torch.bool),
    }

    utterance_lens: list[int] = []
    for bi, (utterance, tools) in enumerate(zip(utterances, canon_tools_per_example, strict=False)):
        enc = tokenizer.encode(utterance)
        n = min(len(enc.ids), lu)
        utterance_lens.append(n)
        out["utterance_ids"][bi, :n] = torch.tensor(enc.ids[:n], dtype=torch.long)
        out["utterance_mask"][bi, :n] = True
        out["tool_count"][bi] = len(tools)
        for ti, canon in enumerate(tools):
            p = pack_tool(cfg, tokenizer, canon)
            out["schema_ids"][bi, ti] = torch.tensor(p.ids, dtype=torch.long)
            out["schema_mask"][bi, ti] = torch.tensor(p.mask, dtype=torch.bool)
            out["schema_kinds"][bi, ti] = torch.tensor(p.kinds, dtype=torch.long)
            out["tool_anchors"][bi, ti] = p.tool_anchor
            for k, anchor in enumerate(p.arg_anchors):
                out["arg_anchors"][bi, ti, k] = anchor
                out["arg_mask"][bi, ti, k] = True
                for j, ea in enumerate(p.enum_anchors[k]):
                    out["enum_anchors"][bi, ti, k, j] = ea
                    out["enum_mask"][bi, ti, k, j] = True

    out["utterance_lens"] = torch.tensor(utterance_lens, dtype=torch.long)
    return out
