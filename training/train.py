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
    "tool": 1.0,
    "presence": 0.5,
    "span": 1.0,
    "enum": 0.5,
    "boolean": 0.5,
    "unit": 0.5,
    "magnitude": 0.5,
    "datetime": 0.5,
}


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
        max_utterance_tokens=48,
        max_schema_tokens=144,
    )


def ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the last dim with -100 ignore, safe on empty."""
    flat = logits.reshape(-1, logits.shape[-1]).float()
    t = target.reshape(-1)
    if (t != -100).sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(flat, t, ignore_index=-100)


def compute_loss(out: dict[str, torch.Tensor], tgt: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    parts = {
        "action": ce(out["action.logits"], tgt["action"]),
        "tool": ce(out["tool.logits"], tgt["tool"]),
        "presence": ce(out["presence.logits"], tgt["presence"]),
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
    }
    for i in range(0, len(items), batch_size):
        batch = make_batch(cfg, items[i : i + batch_size])
        tgt = batch.pop("targets")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        out = {k: v.cpu() for k, v in out.items()}

        agg["action_correct"] += int((out["action.logits"].argmax(-1) == tgt["action"]).sum())
        agg["action_total"] += len(tgt["action"])
        agg["tool_correct"] += int((out["tool.logits"].argmax(-1) == tgt["tool"]).sum())
        agg["tool_total"] += len(tgt["tool"])

        for name, logits, target in (
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = Tokenizer.from_file(str(REPO / "contracts" / "tokenizer" / "tokenizer.json"))
    cfg = mini_config(tokenizer.get_vocab_size())
    print(f"device={device} vocab={cfg.vocab}")

    train_items = load_and_prepare(cfg, tokenizer, args.data / "train.jsonl")
    dev_items = load_and_prepare(cfg, tokenizer, args.data / "dev.jsonl")
    if args.limit:
        train_items = train_items[: args.limit]
    print(f"train={len(train_items)} dev={len(dev_items)}")

    model = NtcEncoderHeadsV1(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr,
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
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss, parts = compute_loss(out, tgt)
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
