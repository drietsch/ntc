"""Dataset → model inputs + head targets.

Precomputes per-example packed inputs (via `ntc_model.packing`, anchors from
the Rust canonical renderer) and target tensors for every head in the codec.
Char-offset gold spans convert to token indices here (the token-span
contract), using the tokenizer's char offsets.

Label conventions: `-100` = no supervision (ignored by cross-entropy);
presence classes PRESENT 0 / MISSING 1 / AMBIGUOUS 2 / NOT_APPLICABLE 3;
NO_TOOL tool label resolves to the batch's `T` at batch-assembly time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ntc_model.config import NtcArchConfig
from ntc_model.packing import Canonicalizer, PackedTool, pack_tool

IGNORE = -100
PRESENCE = {"PRESENT": 0, "MISSING": 1, "AMBIGUOUS": 2, "NOT_APPLICABLE": 3}
RELATION = {
    "TODAY": 1, "TOMORROW": 2, "YESTERDAY": 3, "THIS": 4, "NEXT": 5,
    "LAST": 6, "IN": 7, "AGO": 8,
}
WEEKDAY = {
    "MONDAY": 1, "TUESDAY": 2, "WEDNESDAY": 3, "THURSDAY": 4, "FRIDAY": 5,
    "SATURDAY": 6, "SUNDAY": 7,
}
DAYPART = {"MORNING": 1, "NOON": 2, "AFTERNOON": 3, "EVENING": 4, "NIGHT": 5}
UNIT = {"SECOND": 1, "MINUTE": 2, "HOUR": 3, "DAY": 4, "WEEK": 5}
NO_TOOL = -1  # sentinel; becomes index T at batch time


@dataclass
class Prepared:
    """One example, packed and labeled (unbatched)."""

    id: str
    lang: str
    split: str
    tags: list[str]
    utterance: str
    utterance_ids: list[int]
    utterance_offsets: list[tuple[int, int]]
    tools: list[PackedTool]
    canon: list[dict]
    # targets
    action: int
    tool: int  # candidate index or NO_TOOL
    presence: list[list[int]]  # [n_tools][max_args]
    span_start: list[list[int]]
    span_end: list[list[int]]
    enum: list[list[int]]
    boolean: list[list[int]]
    unit: list[list[int]]
    magnitude: list[list[float]]
    magnitude_mask: list[list[bool]]
    relation: list[list[int]]
    weekday: list[list[int]]
    daypart: list[list[int]]
    month: list[list[int]]


ACTION_IDS = {"CALL": 0, "ASK": 1, "NO_CALL": 2, "DELEGATE": 3}


def char_span_to_token_span(
    offsets: list[tuple[int, int]], start: int, end: int
) -> tuple[int, int] | None:
    """First/last token intersecting the char range; inclusive end index."""
    toks = [i for i, (s, e) in enumerate(offsets) if e > s and s < end and e > start]
    if not toks:
        return None
    return toks[0], toks[-1]


def prepare_example(
    cfg: NtcArchConfig,
    tokenizer: Tokenizer,
    ex: dict,
    canon_cache: dict[str, dict],
) -> Prepared:
    a = cfg.max_args
    enc = tokenizer.encode(ex["utterance"])
    if len(enc.ids) > cfg.max_utterance_tokens:
        raise ValueError(f"{ex['id']}: utterance tokenizes to {len(enc.ids)} > Lu")

    canon = []
    for i, tool in enumerate(ex["candidates"]):
        key = json.dumps(tool, sort_keys=True) + f"#{i}"
        canon.append(canon_cache[key])
    tools = [pack_tool(cfg, tokenizer, c) for c in canon]
    n_tools = len(tools)

    gold = ex["gold"]
    action = ACTION_IDS[gold["action"]]
    tool_names = [c["id"] for c in canon]
    tool_label = tool_names.index(gold["tool"]) if gold.get("tool") else NO_TOOL

    def grid(fill):
        return [[fill] * a for _ in range(n_tools)]

    presence = grid(IGNORE)
    span_start = grid(IGNORE)
    span_end = grid(IGNORE)
    enum = grid(IGNORE)
    boolean = grid(IGNORE)
    unit = grid(IGNORE)
    magnitude = grid(0.0)
    magnitude_mask = [[False] * a for _ in range(n_tools)]
    relation = grid(IGNORE)
    weekday = grid(IGNORE)
    daypart = grid(IGNORE)
    month = grid(IGNORE)

    # Decoys (and every candidate on NO_CALL/DELEGATE): NOT_APPLICABLE.
    no_tool_action = action in (ACTION_IDS["NO_CALL"], ACTION_IDS["DELEGATE"])
    for t, c in enumerate(canon):
        if t == tool_label and not no_tool_action:
            continue
        for k in range(len(c["tool"]["args"])):
            presence[t][k] = PRESENCE["NOT_APPLICABLE"]

    if tool_label != NO_TOOL and not no_tool_action:
        t = tool_label
        args = canon[t]["tool"]["args"]
        arg_index = {arg["name"]: k for k, arg in enumerate(args)}
        bound = {arg["parameter"] for arg in gold.get("arguments", [])}
        unresolved = {u["parameter"]: u["reason"] for u in gold.get("unresolved", [])}

        if action == ACTION_IDS["CALL"]:
            # Unmentioned declared args of the gold tool: NOT_APPLICABLE.
            for name, k in arg_index.items():
                if name not in bound and name not in unresolved:
                    presence[t][k] = PRESENCE["NOT_APPLICABLE"]
        for name, reason in unresolved.items():
            presence[t][arg_index[name]] = PRESENCE[reason]

        for garg in gold.get("arguments", []):
            k = arg_index[garg["parameter"]]
            presence[t][k] = PRESENCE["PRESENT"]
            st = garg["semantic_type"]
            value = garg["value"]

            if garg.get("char_span"):
                ts = char_span_to_token_span(
                    enc.offsets, garg["char_span"]["start"], garg["char_span"]["end"]
                )
                if ts is not None:
                    span_start[t][k], span_end[t][k] = ts

            if st == "ENUM":
                enum[t][k] = value["index"]
            elif st == "BOOLEAN":
                boolean[t][k] = int(bool(value))
            elif st == "DURATION":
                unit[t][k] = UNIT[value["unit"]]
                magnitude[t][k] = math.asinh(float(value["magnitude"]))
                magnitude_mask[t][k] = True
            elif st == "INTEGER":
                magnitude[t][k] = math.asinh(float(value))
                magnitude_mask[t][k] = True
            elif st in ("RELATIVE_DATE", "RELATIVE_DATETIME"):
                relation[t][k] = RELATION[value["relation"]]
                weekday[t][k] = WEEKDAY.get(value.get("weekday") or "", 0)
                daypart[t][k] = DAYPART.get(value.get("daypart") or "", 0)
                month[t][k] = 0

    return Prepared(
        id=ex["id"], lang=ex["lang"], split=ex["split"], tags=ex.get("tags", []),
        utterance=ex["utterance"], utterance_ids=list(enc.ids),
        utterance_offsets=list(enc.offsets), tools=tools, canon=canon,
        action=action, tool=tool_label, presence=presence,
        span_start=span_start, span_end=span_end, enum=enum, boolean=boolean,
        unit=unit, magnitude=magnitude, magnitude_mask=magnitude_mask,
        relation=relation, weekday=weekday, daypart=daypart, month=month,
    )


def load_and_prepare(
    cfg: NtcArchConfig,
    tokenizer: Tokenizer,
    path: Path,
    skip_oversize: bool = False,
) -> list[Prepared]:
    examples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # One canonicalizer call for the whole file (cache keyed by schema+index).
    canonicalizer = Canonicalizer()
    pending: dict[str, tuple[dict, int]] = {}
    for ex in examples:
        for i, tool in enumerate(ex["candidates"]):
            key = json.dumps(tool, sort_keys=True) + f"#{i}"
            pending.setdefault(key, (tool, i))
    keys = list(pending)
    results = canonicalizer.canonicalize(
        [pending[k][0] for k in keys], [pending[k][1] for k in keys]
    )
    cache = dict(zip(keys, results, strict=False))

    out: list[Prepared] = []
    skipped = 0
    for ex in examples:
        try:
            out.append(prepare_example(cfg, tokenizer, ex, cache))
        except ValueError:
            # Utterance or canonical schema longer than the model's window.
            # Packing fails loudly by design; corpora with a long tail (xLAM)
            # opt into skipping instead of aborting the run.
            if not skip_oversize:
                raise
            skipped += 1
    if skipped:
        print(f"{path.name}: skipped {skipped}/{len(examples)} examples exceeding the window")
    return out


def make_batch(cfg: NtcArchConfig, items: list[Prepared]) -> dict[str, torch.Tensor]:
    """Assemble padded input + target tensors for a batch."""
    b = len(items)
    lu, ls = cfg.max_utterance_tokens, cfg.max_schema_tokens
    a, e = cfg.max_args, cfg.max_enum_values
    t = max(len(it.tools) for it in items)

    out = {
        "utterance_ids": torch.zeros(b, lu, dtype=torch.long),
        "utterance_mask": torch.zeros(b, lu, dtype=torch.bool),
        "schema_ids": torch.zeros(b, t, ls, dtype=torch.long),
        "schema_mask": torch.zeros(b, t, ls, dtype=torch.bool),
        "schema_kinds": torch.full((b, t, ls), 9, dtype=torch.long),
        "tool_count": torch.zeros(b, dtype=torch.long),
        "tool_anchors": torch.zeros(b, t, dtype=torch.long),
        "arg_anchors": torch.zeros(b, t, a, dtype=torch.long),
        "arg_mask": torch.zeros(b, t, a, dtype=torch.bool),
        "enum_anchors": torch.zeros(b, t, a, e, dtype=torch.long),
        "enum_mask": torch.zeros(b, t, a, e, dtype=torch.bool),
    }
    tgt = {
        "action": torch.zeros(b, dtype=torch.long),
        "tool": torch.zeros(b, dtype=torch.long),
        "presence": torch.full((b, t, a), IGNORE, dtype=torch.long),
        "span_start": torch.full((b, t, a), IGNORE, dtype=torch.long),
        "span_end": torch.full((b, t, a), IGNORE, dtype=torch.long),
        "enum": torch.full((b, t, a), IGNORE, dtype=torch.long),
        "boolean": torch.full((b, t, a), IGNORE, dtype=torch.long),
        "unit": torch.full((b, t, a), IGNORE, dtype=torch.long),
        "magnitude": torch.zeros(b, t, a),
        "magnitude_mask": torch.zeros(b, t, a, dtype=torch.bool),
        "relation": torch.full((b, t, a), IGNORE, dtype=torch.long),
        "weekday": torch.full((b, t, a), IGNORE, dtype=torch.long),
        "daypart": torch.full((b, t, a), IGNORE, dtype=torch.long),
        "month": torch.full((b, t, a), IGNORE, dtype=torch.long),
    }

    for bi, it in enumerate(items):
        n = len(it.utterance_ids)
        out["utterance_ids"][bi, :n] = torch.tensor(it.utterance_ids)
        out["utterance_mask"][bi, :n] = True
        out["tool_count"][bi] = len(it.tools)
        tgt["action"][bi] = it.action
        tgt["tool"][bi] = it.tool if it.tool != NO_TOOL else t
        for ti, p in enumerate(it.tools):
            out["schema_ids"][bi, ti] = torch.tensor(p.ids)
            out["schema_mask"][bi, ti] = torch.tensor(p.mask)
            out["schema_kinds"][bi, ti] = torch.tensor(p.kinds)
            out["tool_anchors"][bi, ti] = p.tool_anchor
            for k, anchor in enumerate(p.arg_anchors):
                out["arg_anchors"][bi, ti, k] = anchor
                out["arg_mask"][bi, ti, k] = True
                for j, ea in enumerate(p.enum_anchors[k]):
                    out["enum_anchors"][bi, ti, k, j] = ea
                    out["enum_mask"][bi, ti, k, j] = True
            for name in (
                "presence", "span_start", "span_end", "enum", "boolean", "unit",
                "relation", "weekday", "daypart", "month",
            ):
                vals = getattr(it, name)[ti]
                tgt[name][bi, ti, : len(vals)] = torch.tensor(vals)
            tgt["magnitude"][bi, ti, : a] = torch.tensor(it.magnitude[ti])
            tgt["magnitude_mask"][bi, ti, : a] = torch.tensor(it.magnitude_mask[ti])

    return {**out, "targets": tgt}
