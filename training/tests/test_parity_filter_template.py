"""IC-1 for the filter-template head: PyTorch ↔ Rust on a fixture that has one.

The standing parity fixture declares no value templates, so neither side emits
`filter_template.logits` and the head-set assertion passes while covering
nothing. This builds a tiny model that *does* declare templates, exports it,
and compares the head the same way IC-1 compares the rest.

Worth its own file because the head was added to two implementations
independently — a PyTorch `nn.Linear` and a hand-written Rust matmul — and this
project's most expensive bugs have all been a Python/Rust divergence that
nothing errored on.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from export.export_tiny import export_tiny
from ntc_model.config import FilterTemplate, tiny_config
from ntc_model.io import load_ntc_model
from ntc_model.packing import Canonicalizer, pack_batch

REPO = Path(__file__).resolve().parents[2]
NTC_BIN = REPO / "target" / "debug" / "ntc"
RUST_TINY = REPO / "fixtures" / "models" / "tiny-v1" / "tiny.ntc"

UTTERANCE = "which Teaser have a teaserText below 199"

#: One argument the templates serve and one they do not, so the test also
#: covers that an unannotated argument is left alone.
TOOLS = [
    {
        "name": "search_objects",
        "description": "search data objects",
        "parameters": {
            "className": {"type": "string", "description": "the class"},
            "pqlFilter": {
                "type": "string",
                "description": "a pql filter",
                "semantic": "FILTER.PQL",
            },
        },
    }
]

TEMPLATES = [
    FilterTemplate(id="FIELD_IS_NULL", semantic="FILTER.PQL", pattern="{field} IS NULL"),
    FilterTemplate(
        id="FIELD_LESS_THAN", semantic="FILTER.PQL", pattern="{field} < {number}"
    ),
    FilterTemplate(
        id="UNPUBLISHED_PAGES",
        semantic="FILTER.PQL",
        pattern='type = "page" AND published = false',
    ),
]

ATOL, RTOL = 1.0e-4, 1.0e-3  # fixtures/tolerances.toml [f32]

pytestmark = pytest.mark.skipif(
    not NTC_BIN.exists() or not RUST_TINY.exists(),
    reason="ntc binary or tiny-v1 fixture missing (cargo build -p ntc-cli)",
)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory) -> Path:
    cfg = tiny_config().model_copy(update={"filter_templates": TEMPLATES})
    out = tmp_path_factory.mktemp("tpl") / "tiny-templates.ntc"
    return export_tiny(out, seed=42, cfg=cfg)


@pytest.fixture(scope="module")
def tools_path(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("tpl-tools") / "tools.json"
    p.write_text(json.dumps(TOOLS))
    return p


def rust_heads(model_path: Path, tools_path: Path) -> dict[str, dict]:
    proc = subprocess.run(
        [
            str(NTC_BIN), "infer",
            "--model", str(model_path),
            "--utterance", UTTERANCE,
            "--tools", str(tools_path),
            "--dump-heads",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def python_heads(model_path: Path) -> dict[str, np.ndarray]:
    cfg, model, tokenizer_bytes = load_ntc_model(model_path)
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode())
    canon = Canonicalizer(NTC_BIN).canonicalize(TOOLS, list(range(len(TOOLS))))
    batch = pack_batch(cfg, tokenizer, [UTTERANCE], [canon])
    batch.pop("utterance_lens")
    with torch.no_grad():
        out = model(**batch)
    return {k: v[0].numpy() for k, v in out.items()}


def test_the_head_exists_on_both_sides(model_path, tools_path):
    rust = rust_heads(model_path, tools_path)
    py = python_heads(model_path)
    assert "filter_template.logits" in rust, "Rust did not emit the head"
    assert "filter_template.logits" in py, "PyTorch did not emit the head"
    assert set(rust) == set(py), (
        f"head set mismatch: rust-only {set(rust) - set(py)}, py-only {set(py) - set(rust)}"
    )


def test_filter_template_logits_agree(model_path, tools_path):
    rust = rust_heads(model_path, tools_path)
    py = python_heads(model_path)

    entry = rust["filter_template.logits"]
    r = np.asarray(entry["data"], dtype=np.float32).reshape(entry["shape"])
    p = py["filter_template.logits"].astype(np.float32)
    assert r.shape == p.shape, f"rust {r.shape} vs python {p.shape}"
    # NONE plus one class per declared template.
    assert r.shape[-1] == len(TEMPLATES) + 1

    valid = r > np.float32(np.finfo(np.float32).min) / 2
    assert (valid == (p > np.float32(np.finfo(np.float32).min) / 2)).all(), (
        "masked-region mismatch: the two sides disagree on which arg slots exist"
    )
    diff = np.abs(r[valid] - p[valid])
    bound = ATOL + RTOL * np.abs(p[valid])
    assert (diff <= bound).all(), f"worst excess {(diff - bound).max():.3e}"


def test_template_decisions_agree(model_path, tools_path):
    """The gate that actually matters: same argmax, so the same template."""
    rust = rust_heads(model_path, tools_path)
    py = python_heads(model_path)
    entry = rust["filter_template.logits"]
    r = np.asarray(entry["data"], dtype=np.float32).reshape(entry["shape"])
    p = py["filter_template.logits"]

    checked = 0
    for t in range(len(TOOLS)):
        for k in range(len(TOOLS[t]["parameters"])):
            assert int(np.argmax(r[t, k])) == int(np.argmax(p[t, k])), (
                f"filter_template[{t},{k}]: rust picks {int(np.argmax(r[t, k]))}, "
                f"python picks {int(np.argmax(p[t, k]))}"
            )
            checked += 1
    assert checked == 2
