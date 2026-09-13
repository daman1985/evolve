"""The naive round-0 codec: a stable container header wrapped around
``zlib.compress`` of the raw array bytes.

No modelling, no block framing, no scheme selection. This exists to
establish the measuring apparatus (bench + tests) and an honest floor for
later rounds to beat. It is *expected* to fail the no-expansion gate on
incompressible data -- that failure is the point of round 0, not a bug.

Bitstream layout (little-endian throughout)::

    offset  size  field
    0       4     magic b'TSC0'
    4       1     format version (uint8), currently 0
    5       1     dtype code (uint8), see DTYPE_CODES
    6       8     element count (uint64)
    14      *     payload = zlib.compress(array.tobytes(), level=6)

The header is always exactly 14 bytes.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

MAGIC = b"TSC0"
FORMAT_VERSION = 0

# Stable code -> numpy dtype string table. Codes are permanent once assigned;
# later rounds may only append new codes, never renumber or remove these.
DTYPE_CODES: dict[int, str] = {
    0: "bool",
    1: "int8",
    2: "int16",
    3: "int32",
    4: "int64",
    5: "uint8",
    6: "uint16",
    7: "uint32",
    8: "uint64",
    9: "float32",
    10: "float64",
}
_STR_TO_CODE: dict[str, int] = {v: k for k, v in DTYPE_CODES.items()}

_HEADER_STRUCT = struct.Struct("<4sBBQ")  # magic, version, dtype_code, count
_HEADER_SIZE = _HEADER_STRUCT.size  # 14


def _dtype_code_for(a: np.ndarray) -> int:
    name = a.dtype.name
    try:
        return _STR_TO_CODE[name]
    except KeyError:
        raise TypeError(
            f"tscodec does not support dtype {a.dtype!r}; supported dtypes "
            f"are {sorted(_STR_TO_CODE)}"
        ) from None


def encode(a: np.ndarray) -> bytes:
    """Encode a 1-D NumPy array of a supported numeric/bool dtype to bytes.

    Raises TypeError for unsupported ndim or dtype (e.g. 2-D, object).
    """
    if not isinstance(a, np.ndarray):
        raise TypeError(f"tscodec.encode expects a numpy.ndarray, got {type(a)!r}")
    if a.ndim != 1:
        raise TypeError(f"tscodec only supports 1-D arrays, got ndim={a.ndim}")

    dtype_code = _dtype_code_for(a)

    # Normalise to native byte order before taking raw bytes. `.astype` with
    # the native-order dtype is a no-op copy-avoiding call when already
    # native, and produces a native-order copy otherwise.
    native_dtype = np.dtype(a.dtype.str.replace(">", "=").replace("<", "="))
    if a.dtype != native_dtype or not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a, dtype=native_dtype)

    raw = a.tobytes()
    payload = zlib.compress(raw, 6)

    header = _HEADER_STRUCT.pack(MAGIC, FORMAT_VERSION, dtype_code, a.shape[0])
    return header + payload


def decode(buf) -> np.ndarray:
    """Decode bytes produced by `encode` back into a native-byte-order array."""
    buf = bytes(buf)
    if len(buf) < _HEADER_SIZE:
        raise ValueError(
            f"truncated tscodec buffer: need at least {_HEADER_SIZE} header "
            f"bytes, got {len(buf)}"
        )

    magic, version, dtype_code, count = _HEADER_STRUCT.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic: expected {MAGIC!r}, got {magic!r}")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported format version {version}; this decoder handles "
            f"version {FORMAT_VERSION}"
        )
    if dtype_code not in DTYPE_CODES:
        raise ValueError(f"unknown dtype code {dtype_code}")

    dtype = np.dtype(DTYPE_CODES[dtype_code])
    payload = buf[_HEADER_SIZE:]

    try:
        raw = zlib.decompress(payload)
    except zlib.error as exc:
        raise ValueError(f"corrupt payload: zlib decompression failed: {exc}") from exc

    expected_nbytes = count * dtype.itemsize
    if len(raw) != expected_nbytes:
        raise ValueError(
            f"decoded payload has {len(raw)} bytes, expected {expected_nbytes} "
            f"({count} elements of dtype {dtype})"
        )

    # np.frombuffer over a `bytes` payload is read-only (bytes are immutable);
    # copy so callers get an ordinary writable array, matching what `encode`
    # was given.
    arr = np.frombuffer(raw, dtype=dtype).copy()
    if arr.shape[0] != count:
        raise ValueError(
            f"decoded element count mismatch: expected {count}, got {arr.shape[0]}"
        )
    return arr
