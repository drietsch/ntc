"""IC-1: Python (PyTorch) ↔ Rust runtime parity on the tiny fixture model.

The same `.ntc` file (written by the Python exporter) is run through:
  (a) the PyTorch model + Python packer, and
  (b) the Rust CPU reference runtime (`ntc infer --dump-heads`)
on identical utterance + candidate tools. Gates (fixtures/tolerances.toml):
element agreement within f32 tolerances on the valid region, and 100%
decision parity (action / tool / presence / enum / datetime.relation argmax).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from ntc_model.io import load_ntc_model
from ntc_model.packing import Canonicalizer, pack_batch

REPO = Path(__file__).resolve().parents[2]
NTC_BIN = REPO / "target" / "debug" / "ntc"
MODEL = REPO / "fixtures" / "models" / "tiny-v1-py" / "tiny.ntc"
TOOLS = json.loads((REPO / "examples" / "tools.json").read_text())

UTTERANCE = "make a dentist appointment tomorrow afternoon"

pytestmark = pytest.mark.skipif(
    not NTC_BIN.exists() or not MODEL.exists(),
    reason="ntc binary or tiny-v1-py fixture missing (cargo build -p ntc-cli; export_tiny)",
)

ATOL, RTOL = 1.0e-4, 1.0e-3  # fixtures/tolerances.toml [f32]


def rust_heads() -> dict[str, dict]:
    proc = subprocess.run(
        [
            str(NTC_BIN), "infer",
            "--model", str(MODEL),
            "--utterance", UTTERANCE,
            "--tools", str(REPO / "examples" / "tools.json"),
            "--dump-heads",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def python_heads() -> dict[str, np.ndarray]:
    cfg, model, tokenizer_bytes = load_ntc_model(MODEL)
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode())
    canon = Canonicalizer(NTC_BIN).canonicalize(TOOLS, list(range(len(TOOLS))))
    batch = pack_batch(cfg, tokenizer, [UTTERANCE], [canon])
    batch.pop("utterance_lens")
    with torch.no_grad():
        out = model(**batch)
    return {k: v[0].numpy() for k, v in out.items()}


def test_ic1_logit_and_decision_parity():
    rust = rust_heads()
    py = python_heads()

    assert set(rust) == set(py), (
        f"head set mismatch: rust-only {set(rust) - set(py)}, py-only {set(py) - set(rust)}"
    )

    n_tools = len(TOOLS)
    decisions_checked = 0
    for name, entry in rust.items():
        r = np.asarray(entry["data"], dtype=np.float32).reshape(entry["shape"])
        p = py[name].astype(np.float32)
        assert r.shape == p.shape, f"{name}: rust {r.shape} vs python {p.shape}"

        # Element agreement — padding carries the identical MASK constant on
        # both sides, so comparing everything is safe; exclude the masked
        # constant from the relative check to avoid inf-scale noise.
        valid = r > np.float32(np.finfo(np.float32).min) / 2
        both = valid | (p > np.float32(np.finfo(np.float32).min) / 2)
        assert (valid == (p > np.float32(np.finfo(np.float32).min) / 2)).all(), (
            f"{name}: masked-region mismatch"
        )
        diff = np.abs(r[valid] - p[valid])
        bound = ATOL + RTOL * np.abs(p[valid])
        worst = (diff - bound).max() if diff.size else 0.0
        assert (diff <= bound).all(), f"{name}: worst excess {worst:.3e}"
        del both

    # Decision parity.
    assert int(np.argmax(np.asarray(rust["action.logits"]["data"]))) == int(
        np.argmax(py["action.logits"])
    )
    r_tool = np.asarray(rust["tool.logits"]["data"])
    assert int(np.argmax(r_tool)) == int(np.argmax(py["tool.logits"]))
    decisions_checked += 2

    args_per_tool = [len(t["parameters"]) for t in TOOLS]
    for name, _classes in (
        ("presence.logits", 4),
        ("datetime.relation.logits", 10),
    ):
        entry = rust[name]
        r = np.asarray(entry["data"], dtype=np.float32).reshape(entry["shape"])
        p = py[name]
        for t in range(n_tools):
            for k in range(args_per_tool[t]):
                assert int(np.argmax(r[t, k])) == int(np.argmax(p[t, k])), (
                    f"{name}[{t},{k}] decision mismatch"
                )
                decisions_checked += 1

    assert decisions_checked >= 10
