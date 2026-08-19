"""Stage-2 (schema grounding) training on the mini dataset — the machinery of
spec §48/§56 at machine-feasible scale: composite structured loss over every
head in the codec, per-language dev metrics, checkpointing.

Run:  uv run python train.py --data data/mini --out runs/mini
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from datasets.collate import load_and_prepare, make_batch
from ntc_model.config import Calibration, NtcArchConfig
from ntc_model.model import NtcEncoderHeadsV1

REPO = Path(__file__).resolve().parents[1]

LOSS_WEIGHTS = {
    "action": 1.0,
    "delegate_reason": 0.5,
    "no_call_reason": 0.5,
    "source": 0.75,
    "entity_ref": 0.75,
    "unresolved_reason": 0.4,
    "tool": 1.0,
    "presence": 0.5,
    "span": 1.0,
    "enum": 0.5,
    "boolean": 0.5,
    "unit": 0.5,
    "magnitude": 0.5,
    "datetime": 0.5,
}


def backbone_config(vocab: int) -> NtcArchConfig:
    """paraphrase-multilingual-MiniLM-L12-v2 dims (runs/backbone/meta.json)."""
    return NtcArchConfig(
        hidden=384,
        heads=12,
        ffn=1536,
        vocab=vocab,
        max_positions=512,
        encoder_layers=12,
        schema_layers=2,
        fusion_blocks=2,
        max_tools=8,
        max_args=8,
        max_enum_values=4,
        max_utterance_tokens=64,
        max_schema_tokens=160,
        layer_norm_eps=1e-12,
    )


def xlam_config(vocab: int) -> NtcArchConfig:
    """Stage-2 schema-grounding arch for the xLAM corpus: 4 candidates (the
    converter caps them), wide-ish schema window (p95 of xLAM canonical texts
    is ~260 tokens), 12 args. Same backbone dims as `pimcore_config` so a
    Stage-2 checkpoint fine-tunes onto Pimcore without reshaping."""
    return NtcArchConfig(
        hidden=384,
        heads=12,
        ffn=1536,
        vocab=vocab,
        max_positions=512,
        encoder_layers=12,
        schema_layers=2,
        fusion_blocks=2,
        max_tools=4,
        max_args=12,
        max_enum_values=4,
        max_utterance_tokens=96,
        max_schema_tokens=256,
        layer_norm_eps=1e-12,
        action_classes=4,
    )


def studio_config(vocab: int) -> NtcArchConfig:
    """Arch for the Pimcore Studio corpus (specs/training).

    Schema window covers the longest canonical text in the registry
    (`propose_document_update`, 541 tokens) so nothing is silently dropped;
    the slate is capped to 4 by the converter instead, because truncating
    descriptions would delete the signal the adversarial cases test."""
    return NtcArchConfig(
        hidden=384,
        heads=12,
        ffn=1536,
        vocab=vocab,
        # The backbone ships 512 position embeddings and the longest tool
        # (`propose_document_update`) canonicalizes to 541 tokens. Rather than
        # drop 10% of the corpus — disproportionately the propose_* /
        # PAYLOAD_REQUIRED cases — the table is extended at warm-start (see
        # `extend_positions`). Only the schema encoder needs this many
        # positions: tools are encoded independently, so the window covers one
        # schema, not the packed fusion sequence.
        max_positions=576,
        encoder_layers=12,
        schema_layers=2,
        fusion_blocks=2,
        # Compute-bound choice: fusion self-attends over n_tools x
        # schema_tokens, so the corpus's full shape (4 slates x 576 tokens =
        # 2305) costs ~63 min/epoch here. Narrowing the *slate* to 3 is the
        # right lever; shortening the *window* is not — at 384 tokens 28% of
        # examples drop, and they are disproportionately the long propose_*
        # schemas, i.e. the biggest PAYLOAD_REQUIRED class. That would bias
        # the very measurement this run exists for.
        max_tools=3,
        max_args=8,
        # The registry's widest enum has 6 values (element types).
        max_enum_values=6,
        max_utterance_tokens=64,
        max_schema_tokens=576,
        layer_norm_eps=1e-12,
        action_classes=4,
    )


def pimcore_config(vocab: int) -> NtcArchConfig:
    """Backbone dims with a wide schema window for the real Pimcore tools
    (longest canonical text ≈ 332 tokens) and retrieval-narrowed candidate
    sets (max_tools=4)."""
    return NtcArchConfig(
        hidden=384,
        heads=12,
        ffn=1536,
        vocab=vocab,
        max_positions=512,
        encoder_layers=12,
        schema_layers=2,
        fusion_blocks=2,
        max_tools=4,
        max_args=8,
        max_enum_values=4,
        max_utterance_tokens=64,
        max_schema_tokens=352,
        layer_norm_eps=1e-12,
        # CALL / ASK / NO_CALL / DELEGATE — the router can hand complex
        # multi-step or open-ended work to a full LLM agent.
        action_classes=4,
    )


def mini_config(vocab: int) -> NtcArchConfig:
    return NtcArchConfig(
        hidden=128,
        heads=4,
        ffn=256,
        vocab=vocab,
        max_positions=192,
        encoder_layers=4,
        schema_layers=2,
        fusion_blocks=2,
        max_tools=8,
        max_args=8,
        max_enum_values=4,
        max_utterance_tokens=64,
        max_schema_tokens=144,
    )


def extend_positions(init_sd: dict, target: int) -> None:
    """Grow a warm-start position table to `target` rows, in place.

    New rows continue the learned drift at the tail rather than starting from
    noise: each appended row is the last pretrained row plus the average
    step across the final stretch. Positions beyond the pretrained range are
    only ever reached by long tool schemas, and the schema encoder sees them
    from the first step of fine-tuning.
    """
    pos = init_sd.get("pos_emb.weight")
    if pos is None or pos.shape[0] >= target:
        return
    have = pos.shape[0]
    tail = pos[-32:]
    drift = (tail[1:] - tail[:-1]).mean(dim=0)
    extra = torch.stack([pos[-1] + drift * (i + 1) for i in range(target - have)])
    init_sd["pos_emb.weight"] = torch.cat([pos, extra], dim=0)
    print(f"extended position embeddings {have} -> {target} for long tool schemas")


def ce(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Cross-entropy over the last dim with -100 ignore, safe on empty."""
    flat = logits.reshape(-1, logits.shape[-1]).float()
    t = target.reshape(-1)
    if (t != -100).sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(flat, t, ignore_index=-100, weight=weight)


def class_weights(counts: torch.Tensor, cap: float = 12.0, power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency class weights, softened by `power` and capped.

    Presence labels are dominated by NOT_APPLICABLE, so an unweighted
    objective collapses to the majority class and the model stops binding
    arguments at all. Fully balanced weighting over-corrects in the other
    direction — measured on the Studio corpus it made the model predict
    PRESENT almost everywhere, which showed up as 58 invented optional
    arguments on the dev split and cut executable accuracy to 28%. Neither
    extreme is right, so the correction is softened by `power` (0.5 = square
    root of inverse frequency), with the cap keeping classes that appear a
    handful of times from dominating the gradient.
    """
    freq = counts.clamp(min=1).float()
    w = (freq.sum() / (len(freq) * freq)).pow(power)
    return w.clamp(max=cap)


def presence_class_weights(items, n_classes: int = 4) -> torch.Tensor:
    counts = torch.zeros(n_classes)
    for it in items:
        for row in it.presence:
            for v in row:
                if v != -100:
                    counts[v] += 1
    return class_weights(counts)


def compute_loss(
    out: dict[str, torch.Tensor],
    tgt: dict[str, torch.Tensor],
    presence_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    parts = {
        "action": ce(out["action.logits"], tgt["action"]),
        "tool": ce(out["tool.logits"], tgt["tool"]),
        "presence": ce(out["presence.logits"], tgt["presence"], presence_weight),
        "delegate_reason": ce(out["delegate_reason.logits"], tgt["delegate_reason"]),
        "no_call_reason": ce(out["no_call_reason.logits"], tgt["no_call_reason"]),
        "source": ce(out["source.logits"], tgt["source"]),
        "entity_ref": ce(out["entity_ref.logits"], tgt["entity_ref"]),
        "unresolved_reason": ce(out["unresolved_reason.logits"], tgt["unresolved_reason"]),
        "span": ce(out["span.start.logits"], tgt["span_start"])
        + ce(out["span.end.logits"], tgt["span_end"]),
        "enum": ce(out["enum.logits"], tgt["enum"]),
        "boolean": ce(out["boolean.logits"], tgt["boolean"]),
        "unit": ce(out["numeric.unit.logits"], tgt["unit"]),
        "datetime": ce(out["datetime.relation.logits"], tgt["relation"])
        + ce(out["datetime.weekday.logits"], tgt["weekday"])
        + ce(out["datetime.daypart.logits"], tgt["daypart"])
        + ce(out["datetime.month.logits"], tgt["month"]),
    }
    mag_mask = tgt["magnitude_mask"]
    if mag_mask.any():
        pred = out["numeric.magnitude"].squeeze(-1)[mag_mask].float()
        parts["magnitude"] = F.smooth_l1_loss(pred, tgt["magnitude"][mag_mask].float())
    else:
        parts["magnitude"] = out["numeric.magnitude"].new_zeros(())
    total = sum(LOSS_WEIGHTS[k] * v for k, v in parts.items())
    return total, {k: float(v) for k, v in parts.items()}


@torch.no_grad()
def evaluate(model, cfg, items, device, batch_size=32) -> dict[str, float]:
    model.eval()
    agg = {
        "action_correct": 0, "action_total": 0,
        "tool_correct": 0, "tool_total": 0,
        "presence_correct": 0, "presence_total": 0,
        "span_correct": 0, "span_total": 0,
        "enum_correct": 0, "enum_total": 0,
        "relation_correct": 0, "relation_total": 0,
        "source_correct": 0, "source_total": 0,
        "entity_correct": 0, "entity_total": 0,
        "dreason_correct": 0, "dreason_total": 0,
    }
    for i in range(0, len(items), batch_size):
        batch = make_batch(cfg, items[i : i + batch_size])
        tgt = batch.pop("targets")
        batch.pop("n_linked", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        out = {k: v.cpu() for k, v in out.items()}

        agg["action_correct"] += int((out["action.logits"].argmax(-1) == tgt["action"]).sum())
        agg["action_total"] += len(tgt["action"])
        agg["tool_correct"] += int((out["tool.logits"].argmax(-1) == tgt["tool"]).sum())
        agg["tool_total"] += len(tgt["tool"])

        pmask = tgt["presence"] == 0  # PRESENT
        agg["present_correct"] = agg.get("present_correct", 0) + int(
            (out["presence.logits"].argmax(-1)[pmask] == 0).sum()
        )
        agg["present_total"] = agg.get("present_total", 0) + int(pmask.sum())

        for name, logits, target in (
            ("source", out["source.logits"], tgt["source"]),
            ("entity", out["entity_ref.logits"], tgt["entity_ref"]),
            ("dreason", out["delegate_reason.logits"], tgt["delegate_reason"]),
            ("presence", out["presence.logits"], tgt["presence"]),
            ("enum", out["enum.logits"], tgt["enum"]),
            ("relation", out["datetime.relation.logits"], tgt["relation"]),
        ):
            mask = target != -100
            agg[f"{name}_correct"] += int((logits.argmax(-1)[mask] == target[mask]).sum())
            agg[f"{name}_total"] += int(mask.sum())

        smask = tgt["span_start"] != -100
        s_ok = out["span.start.logits"].argmax(-1)[smask] == tgt["span_start"][smask]
        e_ok = out["span.end.logits"].argmax(-1)[smask] == tgt["span_end"][smask]
        agg["span_correct"] += int((s_ok & e_ok).sum())
        agg["span_total"] += int(smask.sum())

    model.train()
    return {
        k.replace("_correct", "_acc"): agg[k] / max(1, agg[k.replace("_correct", "_total")])
        for k in agg
        if k.endswith("_correct")
    }


def fit_temperatures(model, cfg, items, device) -> Calibration:
    """Grid-search softmax temperatures minimizing NLL on dev (B8, simple)."""
    model.eval()
    logits_action, y_action, logits_tool, y_tool = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(items), 32):
            batch = make_batch(cfg, items[i : i + 32])
            tgt = batch.pop("targets")
            batch.pop("n_linked", None)
            out = model(**{k: v.to(device) for k, v in batch.items()})
            logits_action.append(out["action.logits"].cpu())
            y_action.append(tgt["action"])
            logits_tool.append(out["tool.logits"].cpu())
            y_tool.append(tgt["tool"])
    def best_t(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        # tool.logits width varies with the batch's tool count, so aggregate
        # NLL per batch instead of concatenating.
        grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

        def nll(t: float) -> float:
            total, n = 0.0, 0
            for logits, y in pairs:
                total += float(F.cross_entropy(logits / t, y, reduction="sum"))
                n += len(y)
            return total / max(1, n)

        return min(grid, key=nll)

    model.train()
    return Calibration(
        action=best_t(list(zip(logits_action, y_action, strict=False))),
        tool=best_t(list(zip(logits_tool, y_tool, strict=False))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mini"))
    parser.add_argument("--out", type=Path, default=Path("runs/mini"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=0, help="cap train examples (overfit checks)")
    parser.add_argument("--arch", choices=["mini", "backbone", "pimcore", "xlam", "studio"], default="mini")
    parser.add_argument("--init", type=Path, default=None,
                        help="partial state_dict (runs/backbone/init.pt) to warm-start from")
    parser.add_argument("--backbone-lr", type=float, default=2e-5,
                        help="learning rate for pretrained modules when --init is given")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tok_dir = "tokenizer" if args.arch == "mini" else "tokenizer-any"
    tokenizer = Tokenizer.from_file(str(REPO / "contracts" / tok_dir / "tokenizer.json"))
    make_cfg = {
        "mini": mini_config,
        "backbone": backbone_config,
        "pimcore": pimcore_config,
        "xlam": xlam_config,
        "studio": studio_config,
    }[args.arch]
    cfg = make_cfg(tokenizer.get_vocab_size())
    print(f"device={device} arch={args.arch} vocab={cfg.vocab}")

    train_items = load_and_prepare(cfg, tokenizer, args.data / "train.jsonl", skip_oversize=True)
    dev_items = load_and_prepare(cfg, tokenizer, args.data / "dev.jsonl", skip_oversize=True)
    if args.limit:
        train_items = train_items[: args.limit]
    print(f"train={len(train_items)} dev={len(dev_items)}")

    presence_weight = presence_class_weights(train_items).to(device)
    print(
        "presence class weights "
        f"[PRESENT, MISSING, AMBIGUOUS, NOT_APPLICABLE]: "
        f"{[round(float(w), 2) for w in presence_weight]}"
    )

    model = NtcEncoderHeadsV1(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params / 1e6:.2f}M")

    pretrained_names: set[str] = set()
    if args.init:
        init_sd = torch.load(args.init, map_location="cpu", weights_only=True)["state_dict"]
        extend_positions(init_sd, cfg.max_positions)
        missing, unexpected = model.load_state_dict(init_sd, strict=False)
        assert not unexpected, f"unexpected init tensors: {unexpected[:5]}"
        pretrained_names = set(init_sd)
        print(f"warm-started {len(init_sd)} tensors from {args.init}; "
              f"{len(missing)} stay random (fusion/heads/structural)")
        groups = [
            {"params": [p for n, p in model.named_parameters() if n in pretrained_names],
             "lr": args.backbone_lr},
            {"params": [p for n, p in model.named_parameters() if n not in pretrained_names],
             "lr": args.lr},
        ]
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[g.get("lr", args.lr) for g in opt.param_groups],
        total_steps=args.epochs * math.ceil(len(train_items) / args.batch),
        pct_start=0.1,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"
    best_score = -1.0
    t0 = time.time()
    with log_path.open("w") as log:
        for epoch in range(args.epochs):
            rng.shuffle(train_items)
            epoch_loss, steps = 0.0, 0
            for i in range(0, len(train_items), args.batch):
                batch = make_batch(cfg, train_items[i : i + args.batch])
                tgt = {k: v.to(device) for k, v in batch.pop("targets").items()}
                batch.pop("n_linked", None)
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss, parts = compute_loss(out, tgt, presence_weight)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                epoch_loss += float(loss)
                steps += 1

            metrics = evaluate(model, cfg, dev_items, device)
            score = metrics["action_acc"] + metrics["tool_acc"] + metrics["span_acc"]
            row = {
                "epoch": epoch,
                "loss": epoch_loss / max(1, steps),
                **metrics,
                "elapsed_s": round(time.time() - t0, 1),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row))
            if score > best_score:
                best_score = score
                torch.save(
                    {"cfg": cfg.model_dump(), "state_dict": model.state_dict()},
                    args.out / "best.pt",
                )

    # Calibration on dev (stored into the exported .ntc metadata).
    ckpt = torch.load(args.out / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    calibration = fit_temperatures(model, cfg, dev_items, device)
    ckpt["calibration"] = calibration.model_dump()
    torch.save(ckpt, args.out / "best.pt")
    print(f"calibration: {calibration.model_dump()}")
    print(f"done in {time.time() - t0:.0f}s; best={best_score:.3f}; saved {args.out}/best.pt")


if __name__ == "__main__":
    main()
