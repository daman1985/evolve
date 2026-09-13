"""Basic round-trip and error-handling tests for the round-0 naive codec.

The auditor extends this suite with adversarial/fuzz cases; these tests only
cover the behaviours the spec calls out explicitly.
"""

import numpy as np
import pytest

from tscodec.codec import DTYPE_CODES, MAGIC, decode, encode

SUPPORTED_DTYPES = [np.dtype(name) for name in DTYPE_CODES.values()]


def _roundtrip(a: np.ndarray) -> np.ndarray:
    return decode(encode(a))


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=lambda d: d.name)
def test_roundtrip_basic(dtype):
    rng = np.random.default_rng(0)
    if dtype == np.dtype(bool):
        a = rng.integers(0, 2, size=257).astype(bool)
    elif np.issubdtype(dtype, np.floating):
        a = rng.standard_normal(257).astype(dtype)
    else:
        info = np.iinfo(dtype)
        a = rng.integers(info.min, info.max, size=257, endpoint=True, dtype=dtype)

    out = _roundtrip(a)
    assert out.dtype == a.dtype
    assert out.shape == a.shape
    assert out.tobytes() == a.tobytes()


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=lambda d: d.name)
def test_roundtrip_empty(dtype):
    a = np.array([], dtype=dtype)
    out = _roundtrip(a)
    assert out.dtype == a.dtype
    assert out.shape == (0,)
    assert out.tobytes() == b""


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=lambda d: d.name)
def test_roundtrip_single_element(dtype):
    if dtype == np.dtype(bool):
        a = np.array([True], dtype=dtype)
    else:
        a = np.array([np.iinfo(dtype).max if not np.issubdtype(dtype, np.floating) else 3.5], dtype=dtype)
    out = _roundtrip(a)
    assert out.dtype == a.dtype
    assert out.tobytes() == a.tobytes()


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=lambda d: d.name)
def test_roundtrip_all_same(dtype):
    if dtype == np.dtype(bool):
        fill = True
    elif np.issubdtype(dtype, np.floating):
        fill = 1.25
    else:
        fill = np.iinfo(dtype).max
    a = np.full(1000, fill, dtype=dtype)
    out = _roundtrip(a)
    assert out.dtype == a.dtype
    assert out.tobytes() == a.tobytes()


def test_roundtrip_random_large():
    rng = np.random.default_rng(42)
    a = rng.standard_normal(50_000).astype(np.float64)
    out = _roundtrip(a)
    assert out.tobytes() == a.tobytes()


def test_nan_and_inf_payloads_survive():
    a = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0], dtype=np.float64)
    out = _roundtrip(a)
    assert out.tobytes() == a.tobytes()
    # confirm byte-exactness actually distinguished 0.0 from -0.0 and preserved
    # the NaN bit pattern, not just numeric equality
    assert np.signbit(out[3]) and not np.signbit(out[4])


def test_int64_min_and_max():
    info = np.iinfo(np.int64)
    a = np.array([info.min, info.max, 0, -1, 1], dtype=np.int64)
    out = _roundtrip(a)
    assert out.tobytes() == a.tobytes()


def test_non_native_byte_order_int():
    rng = np.random.default_rng(1)
    native = rng.integers(-1000, 1000, size=500, dtype=np.int32)
    swapped = native.astype(native.dtype.newbyteorder())
    assert swapped.dtype.byteorder in (">", "<") and swapped.dtype != native.dtype

    out = decode(encode(swapped))
    # decode always returns native order; compare values, not raw dtype object
    assert out.dtype == native.dtype  # native.dtype already has native '=' byteorder in practice
    assert out.tobytes() == native.tobytes()
    np.testing.assert_array_equal(out, native)


def test_non_native_byte_order_float():
    rng = np.random.default_rng(2)
    native = rng.standard_normal(500).astype(np.float64)
    swapped = native.astype(native.dtype.newbyteorder())

    out = decode(encode(swapped))
    assert out.tobytes() == native.tobytes()
    np.testing.assert_array_equal(out, native)


def test_unsupported_dtype_object_raises_typeerror():
    a = np.array(["a", "b", "c"], dtype=object)
    with pytest.raises(TypeError):
        encode(a)


def test_unsupported_ndim_raises_typeerror():
    a = np.zeros((10, 10), dtype=np.float64)
    with pytest.raises(TypeError):
        encode(a)


def test_unsupported_complex_dtype_raises_typeerror():
    a = np.zeros(10, dtype=np.complex128)
    with pytest.raises(TypeError):
        encode(a)


def test_corrupt_magic_raises_valueerror():
    a = np.arange(10, dtype=np.int32)
    buf = bytearray(encode(a))
    buf[0:4] = b"XXXX"
    with pytest.raises(ValueError):
        decode(bytes(buf))


def test_bad_version_raises_valueerror():
    a = np.arange(10, dtype=np.int32)
    buf = bytearray(encode(a))
    buf[4] = 255
    with pytest.raises(ValueError):
        decode(bytes(buf))


def test_truncated_buffer_raises_valueerror():
    a = np.arange(10, dtype=np.int32)
    buf = encode(a)
    with pytest.raises(ValueError):
        decode(buf[:5])


def test_truncated_payload_raises_valueerror():
    a = np.arange(1000, dtype=np.int64)
    buf = encode(a)
    with pytest.raises(ValueError):
        decode(buf[:-10])


def test_wrong_element_count_in_header_raises_valueerror():
    a = np.arange(10, dtype=np.int32)
    buf = bytearray(encode(a))
    # bump the declared element count (bytes 6..14, little-endian uint64)
    # well beyond what the payload actually decompresses to
    buf[6:14] = (10_000).to_bytes(8, "little")
    with pytest.raises(ValueError):
        decode(bytes(buf))


def test_magic_constant_is_stable():
    assert MAGIC == b"TSC0"


def test_dtype_codes_cover_all_supported_dtypes():
    names = set(DTYPE_CODES.values())
    expected = {
        "bool", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float32", "float64",
    }
    assert names == expected
