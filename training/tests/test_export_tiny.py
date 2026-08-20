"""Conformance: the Python-exported tiny .ntc must pass the Rust verifier."""

import json
import subprocess

import pytest

from export.export_tiny import (
    DEFAULT_OUT,
    REPO_ROOT,
    RUST_TINY_NTC,
    export_tiny,
    extract_test_tokenizer,
)
from export.ntc_writer import read_ntc_file
from ntc_model.config import tiny_config
from ntc_model.model import tensor_specs

#: Derived from the head contract, not hardcoded. A literal here drifted from
#: the model when the v3 heads landed and stayed wrong; test_model.py keeps one
#: deliberate pin so the count still cannot change silently.
EXPECTED_TENSORS = len(tensor_specs(tiny_config()))

NTC_BIN = REPO_ROOT / "target" / "debug" / "ntc"

needs_ntc_bin = pytest.mark.skipif(
    not NTC_BIN.is_file(), reason="Rust CLI not built (cargo build -p ntc-cli)"
)


def test_tokenizer_extraction_matches_rust_fixture_manifest():
    tok = extract_test_tokenizer()
    manifest = json.loads((RUST_TINY_NTC.parent / "tiny.manifest.json").read_text())
    assert len(tok) == manifest["tokenizer_bytes"]
    import hashlib

    assert hashlib.sha256(tok).hexdigest() == manifest["tokenizer_sha256"]


def test_export_writes_parseable_file(tmp_path):
    out = export_tiny(tmp_path / "tiny.ntc", seed=1)
    parsed = read_ntc_file(out)
    assert parsed["sha256_ok"]
    assert parsed["metadata"]["architecture"] == "ntc_encoder_heads_v1"
    assert len(parsed["records"]) == EXPECTED_TENSORS
    # Tokenizer bytes are byte-identical to the Rust fixture's.
    assert parsed["tokenizer_bytes"] == extract_test_tokenizer()


@needs_ntc_bin
def test_rust_verify_accepts_python_export():
    out = export_tiny(DEFAULT_OUT, seed=42)
    proc = subprocess.run(
        [str(NTC_BIN), "verify", str(out), "--dump-manifest"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, f"ntc verify failed:\n{proc.stdout}\n{proc.stderr}"
    manifest = json.loads(proc.stdout)
    assert manifest["architecture"] == "ntc_encoder_heads_v1"
    assert manifest["tensor_count"] == EXPECTED_TENSORS
    assert manifest["model_version"] == "tiny-v1-py-seed42"
    assert manifest["quantization"] == "f32"

    # Same tensor set + shapes as the Rust-written tiny fixture.
    rust_manifest = json.loads((RUST_TINY_NTC.parent / "tiny.manifest.json").read_text())
    py_tensors = {t["name"]: t["shape"] for t in manifest["tensors"]}
    rust_tensors = {t["name"]: t["shape"] for t in rust_manifest["tensors"]}
    assert py_tensors == rust_tensors
    assert manifest["tokenizer_sha256"] == rust_manifest["tokenizer_sha256"]
