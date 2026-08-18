"""The production `.ntc` v1 writer (format spec §33–§35).

The normative *reader* is `crates/ntc-format`; this writer is pinned to it by
the conformance suite (`ntc verify` must accept every file this produces).

Layout v1 (all integers little-endian)::

    offset  size  field
    0       4     magic "NTC1"
    4       4     format_version u32 (= 1)
    8       8     metadata_len   u64   (JSON)
    16      8     tokenizer_len  u64   (raw tokenizer.json bytes)
    24      8     directory_len  u64   (JSON array of tensor records)
    32      8     data_len       u64   (tensor data section)
    40      ...   metadata | tokenizer | directory
    ...     ...   zero padding to a 256-byte boundary (from file start)
    ...     ...   data section — each tensor blob 256-byte aligned
    end-32  32    sha256 of everything before the footer

Tensor directory entry::

    {"name", "dtype" ("F32"|"F16"|"BF16"), "shape": [u64...],
     "offset" (relative to data-section start, multiple of 256),
     "byte_length", "xxh64" (16 lowercase hex digits, seed 0)}
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import xxhash

MAGIC = b"NTC1"
FORMAT_VERSION = 1
TENSOR_ALIGN = 256

_DTYPE_SIZES = {"F32": 4, "F16": 2, "BF16": 2}
_NUMPY_DTYPES = {"F32": "<f4", "F16": "<f2"}

_REQUIRED_METADATA_KEYS = (
    "architecture",
    "model_version",
    "ir_version",
    "abi_version",
    "head_spec_version",
    "quantization",
    "model",
)


def _align_up(n: int, align: int = TENSOR_ALIGN) -> int:
    return (n + align - 1) // align * align


class NtcWriter:
    """Streamed-layout `.ntc` writer.

    `metadata` must carry the required metadata keys (`tokenizer_sha256` is
    computed here; `semantic_types` defaults to `[]`).
    """

    def __init__(self, metadata: dict[str, Any], tokenizer_bytes: bytes):
        missing = [k for k in _REQUIRED_METADATA_KEYS if k not in metadata]
        if missing:
            raise ValueError(f"metadata missing required keys: {missing}")
        # Key order pinned to the Rust NtcMetadata serde order.
        self.metadata: dict[str, Any] = {
            "architecture": metadata["architecture"],
            "model_version": metadata["model_version"],
            "ir_version": metadata["ir_version"],
            "abi_version": metadata["abi_version"],
            "head_spec_version": metadata["head_spec_version"],
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
            "quantization": metadata["quantization"],
            "model": metadata["model"],
            "semantic_types": metadata.get("semantic_types", []),
        }
        self.tokenizer_bytes = bytes(tokenizer_bytes)
        self._records: list[dict[str, Any]] = []
        self._names: set[str] = set()
        self._data = bytearray()

    def add_tensor(
        self,
        name: str,
        dtype: str,
        shape: list[int] | tuple[int, ...],
        data: np.ndarray | bytes | bytearray,
    ) -> None:
        """Append a tensor; blobs are laid out in call order, 256-byte aligned.

        `data` is either raw little-endian bytes or a numpy array (converted
        to the target dtype; BF16 requires raw bytes).
        """
        if dtype not in _DTYPE_SIZES:
            raise ValueError(f"tensor `{name}`: unsupported dtype `{dtype}` (F32|F16|BF16)")
        if name in self._names:
            raise ValueError(f"duplicate tensor name `{name}`")

        shape = [int(d) for d in shape]
        if any(d < 0 for d in shape):
            raise ValueError(f"tensor `{name}`: negative dimension in {shape}")

        if isinstance(data, np.ndarray):
            if dtype not in _NUMPY_DTYPES:
                raise ValueError(
                    f"tensor `{name}`: pass raw bytes for dtype `{dtype}` (no numpy equivalent)"
                )
            blob = np.ascontiguousarray(data, dtype=_NUMPY_DTYPES[dtype]).tobytes()
        else:
            blob = bytes(data)

        expected = int(np.prod(shape, dtype=np.int64)) * _DTYPE_SIZES[dtype]
        if len(blob) != expected:
            raise ValueError(
                f"tensor `{name}`: byte length {len(blob)} does not match "
                f"shape {shape} × {_DTYPE_SIZES[dtype]}"
            )

        offset = _align_up(len(self._data))
        self._data.extend(b"\x00" * (offset - len(self._data)))
        self._data.extend(blob)
        self._names.add(name)
        self._records.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "offset": offset,
                "byte_length": len(blob),
                "xxh64": f"{xxhash.xxh64(blob, seed=0).intdigest():016x}",
            }
        )

    def finish(self) -> bytes:
        """Assemble the complete file (header, sections, padding, footer)."""
        metadata = json.dumps(self.metadata, separators=(",", ":"), ensure_ascii=False).encode()
        directory = json.dumps(self._records, separators=(",", ":"), ensure_ascii=False).encode()

        var_start = 4 + 4 + 8 * 4
        dir_end = var_start + len(metadata) + len(self.tokenizer_bytes) + len(directory)
        data_start = _align_up(dir_end)

        out = bytearray()
        out += MAGIC
        out += struct.pack("<I", FORMAT_VERSION)
        out += struct.pack(
            "<4Q", len(metadata), len(self.tokenizer_bytes), len(directory), len(self._data)
        )
        out += metadata
        out += self.tokenizer_bytes
        out += directory
        out += b"\x00" * (data_start - len(out))
        out += self._data
        out += hashlib.sha256(out).digest()
        return bytes(out)

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.finish())


def read_ntc(buf: bytes) -> dict[str, Any]:
    """Minimal `.ntc` v1 reader (for tokenizer extraction and self-checks).

    Returns {"metadata", "tokenizer_bytes", "records", "data", "sha256_ok"}.
    """
    if buf[:4] != MAGIC:
        raise ValueError("bad magic: expected `NTC1`")
    (version,) = struct.unpack_from("<I", buf, 4)
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported format version {version}")
    metadata_len, tokenizer_len, directory_len, data_len = struct.unpack_from("<4Q", buf, 8)

    pos = 40
    metadata = json.loads(buf[pos : pos + metadata_len])
    pos += metadata_len
    tokenizer_bytes = bytes(buf[pos : pos + tokenizer_len])
    pos += tokenizer_len
    records = json.loads(buf[pos : pos + directory_len])
    pos += directory_len

    data_start = _align_up(pos)
    data = bytes(buf[data_start : data_start + data_len])
    footer = buf[len(buf) - 32 :]
    sha256_ok = hashlib.sha256(buf[: len(buf) - 32]).digest() == footer

    return {
        "metadata": metadata,
        "tokenizer_bytes": tokenizer_bytes,
        "records": records,
        "data": data,
        "sha256_ok": sha256_ok,
    }


def read_ntc_file(path: str | Path) -> dict[str, Any]:
    return read_ntc(Path(path).read_bytes())
