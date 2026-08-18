"""Structural tests for the pure-Python `.ntc` writer."""

import hashlib
import struct

import numpy as np
import pytest
import xxhash

from export.ntc_writer import MAGIC, TENSOR_ALIGN, NtcWriter, read_ntc


def make_metadata() -> dict:
    return {
        "architecture": "ntc_encoder_heads_v1",
        "model_version": "unit-test",
        "ir_version": 1,
        "abi_version": 1,
        "head_spec_version": 1,
        "quantization": "f32",
        "model": {"hidden": 8},
    }


def test_layout_and_footer():
    w = NtcWriter(make_metadata(), b'{"tokenizer": true}')
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.ones(5, dtype=np.float32)
    w.add_tensor("a", "F32", [3, 4], a)
    w.add_tensor("b", "F32", [5], b)
    buf = w.finish()

    assert buf[:4] == MAGIC
    (version,) = struct.unpack_from("<I", buf, 4)
    assert version == 1
    metadata_len, tokenizer_len, directory_len, data_len = struct.unpack_from("<4Q", buf, 8)
    assert tokenizer_len == len(b'{"tokenizer": true}')

    # Data section starts 256-aligned from file start.
    header_end = 40 + metadata_len + tokenizer_len + directory_len
    data_start = (header_end + TENSOR_ALIGN - 1) // TENSOR_ALIGN * TENSOR_ALIGN
    assert buf[header_end:data_start] == b"\x00" * (data_start - header_end)
    assert len(buf) == data_start + data_len + 32

    # sha256 footer over everything before it.
    assert buf[-32:] == hashlib.sha256(buf[:-32]).digest()

    parsed = read_ntc(buf)
    assert parsed["sha256_ok"]
    assert parsed["metadata"]["architecture"] == "ntc_encoder_heads_v1"
    assert parsed["metadata"]["tokenizer_sha256"] == hashlib.sha256(
        b'{"tokenizer": true}'
    ).hexdigest()

    rec_a, rec_b = parsed["records"]
    assert rec_a["name"] == "a" and rec_a["shape"] == [3, 4]
    assert rec_a["offset"] == 0 and rec_a["byte_length"] == 48
    # Second tensor 256-aligned within the data section.
    assert rec_b["offset"] == 256 and rec_b["byte_length"] == 20
    assert data_len == 256 + 20

    # Tensor bytes + xxh64 (seed 0, 16 lowercase hex digits).
    data = parsed["data"]
    got_a = data[rec_a["offset"] : rec_a["offset"] + rec_a["byte_length"]]
    assert got_a == a.astype("<f4").tobytes()
    assert rec_a["xxh64"] == f"{xxhash.xxh64(got_a, seed=0).intdigest():016x}"
    assert len(rec_a["xxh64"]) == 16 and rec_a["xxh64"] == rec_a["xxh64"].lower()


def test_add_tensor_validation():
    w = NtcWriter(make_metadata(), b"{}")
    with pytest.raises(ValueError, match="byte length"):
        w.add_tensor("bad", "F32", [2, 2], np.zeros(3, dtype=np.float32))
    w.add_tensor("x", "F32", [2], np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="duplicate"):
        w.add_tensor("x", "F32", [2], np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="unsupported dtype"):
        w.add_tensor("y", "I8", [2], b"\x00\x00")
    # BF16 requires raw bytes, and the byte length is 2 per element.
    w.add_tensor("z", "BF16", [2], b"\x00\x3f\x00\x3f")
    with pytest.raises(ValueError, match="raw bytes"):
        w.add_tensor("z2", "BF16", [2], np.zeros(2, dtype=np.float32))


def test_metadata_key_validation():
    with pytest.raises(ValueError, match="missing required keys"):
        NtcWriter({"architecture": "x"}, b"{}")
