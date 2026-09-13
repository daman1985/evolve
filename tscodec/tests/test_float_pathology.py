"""Float-specific correctness bait.

All comparisons here are `.tobytes()` comparisons, never `np.array_equal` or
`==`, because both of those lie about NaN (NaN != NaN) and are blind to the
sign of zero and to which of the ~2^52 (float64) / ~2^23 (float32) possible
NaN payloads a given NaN bit pattern encodes.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tscodec.codec import decode, encode


def _roundtrip(a: np.ndarray) -> np.ndarray:
    return decode(encode(a))


def _assert_bytes_exact(a: np.ndarray, out: np.ndarray) -> None:
    assert out.dtype == a.dtype
    assert out.shape == a.shape
    assert out.tobytes() == a.tobytes()


# ---------------------------------------------------------------------
# NaN payload preservation
# ---------------------------------------------------------------------


def _f64_from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def test_distinct_f64_nan_payloads_survive_bit_exact():
    # Quiet NaN with an arbitrary non-zero low-mantissa payload, a different
    # quiet NaN payload, and the canonical numpy NaN -- three different bit
    # patterns that all satisfy `x != x`.
    bits = [
        0x7FF8000000000001,  # quiet NaN, payload=1
        0x7FF80000DEADBEEF & 0x7FFFFFFFFFFFFFFF,  # quiet NaN, distinct payload
        0xFFF8000000000000,  # quiet NaN, sign bit set ("negative" NaN)
        0x7FF0000000000001,  # signalling NaN (quiet bit clear, payload!=0)
    ]
    values = [_f64_from_bits(b) for b in bits]
    a = np.array(values, dtype=np.float64)
    # sanity: these really are distinct bit patterns and really are NaN
    raw_bits = a.view(np.uint64)
    assert len(set(raw_bits.tolist())) == len(bits)
    assert np.all(np.isnan(a))

    out = _roundtrip(a)
    _assert_bytes_exact(a, out)
    assert np.array_equal(out.view(np.uint64), raw_bits)


def test_distinct_f32_nan_payloads_survive_bit_exact():
    bits = [
        0x7FC00001,
        0x7FDEAD00,
        0xFFC00000,
        0x7F800001,  # signalling NaN
    ]
    values = [_f32_from_bits(b) for b in bits]
    a = np.array(values, dtype=np.float32)
    raw_bits = a.view(np.uint32)
    assert len(set(raw_bits.tolist())) == len(bits)
    assert np.all(np.isnan(a))

    out = _roundtrip(a)
    _assert_bytes_exact(a, out)
    assert np.array_equal(out.view(np.uint32), raw_bits)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_signalling_nan_single_element(dtype):
    uint_dtype = np.uint32 if dtype == np.float32 else np.uint64
    exp_bits = np.iinfo(uint_dtype).bit_length if False else None
    # signalling NaN: exponent all-ones, quiet bit (top mantissa bit) clear,
    # some other mantissa bit set
    if dtype == np.float32:
        bits = 0x7F800001
    else:
        bits = 0x7FF0000000000001
    a = np.array([bits], dtype=uint_dtype).view(dtype)
    assert np.isnan(a[0])
    out = _roundtrip(a)
    _assert_bytes_exact(a, out)


# ---------------------------------------------------------------------
# signed zero, infinities, denormals
# ---------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_negative_zero_distinct_from_positive_zero(dtype):
    a = np.array([-0.0, 0.0], dtype=dtype)
    out = _roundtrip(a)
    _assert_bytes_exact(a, out)
    assert np.signbit(out[0])
    assert not np.signbit(out[1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_infinities_both_signs(dtype):
    a = np.array([np.inf, -np.inf, 0.0, -0.0], dtype=dtype)
    out = _roundtrip(a)
    _assert_bytes_exact(a, out)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_denormals_smallest_and_mixed_signs(dtype):
    tiny = np.finfo(dtype).tiny
    smallest_subnormal = np.nextafter(dtype(0.0), dtype(1.0))
    a = np.array(
        [smallest_subnormal, -smallest_subnormal, tiny, -tiny, tiny / 2, -tiny / 2, 0.0, -0.0],
        dtype=dtype,
    )
    assert smallest_subnormal != 0.0
    out = _roundtrip(a)
    _assert_bytes_exact(a, out)


def test_all_special_values_one_array_f64():
    a = np.array(
        [
            np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0,
            np.finfo(np.float64).max, np.finfo(np.float64).min,
            np.finfo(np.float64).tiny, -np.finfo(np.float64).tiny,
            np.nextafter(np.float64(0.0), np.float64(1.0)),
            -np.nextafter(np.float64(0.0), np.float64(1.0)),
            1.0, -1.0, np.pi, -np.pi,
        ],
        dtype=np.float64,
    )
    out = _roundtrip(a)
    _assert_bytes_exact(a, out)


def test_all_special_values_one_array_f32():
    a = np.array(
        [
            np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0,
            np.finfo(np.float32).max, np.finfo(np.float32).min,
            np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny,
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            -np.nextafter(np.float32(0.0), np.float32(1.0)),
            1.0, -1.0, np.float32(np.pi), -np.float32(np.pi),
        ],
        dtype=np.float32,
    )
    out = _roundtrip(a)
    _assert_bytes_exact(a, out)


# ---------------------------------------------------------------------
# float32 values that are lossy under a float64 round trip -- bait for any
# scheme that upcasts float32 to float64 internally (e.g. to compute a
# common delta/scale) and downcasts afterwards without checking exactness.
# ---------------------------------------------------------------------


def test_float32_values_lossy_under_float64_roundtrip():
    # Values deliberately built from the extreme ends of float32's mantissa
    # so that widening to float64 and back is bit-exact (it always is, since
    # float32 -> float64 is lossless and float64 -> float32 of an
    # originally-float32 value is a no-op) -- but if an encoder does
    # arithmetic in float64 (e.g. delta = f64(a) - f64(b)) and casts the
    # *result* back to float32, that intermediate can round differently
    # than performing the same subtraction natively in float32.
    rng = np.random.default_rng(2024)
    mantissa_bits = rng.integers(0, 2**23, size=2000, dtype=np.uint32)
    exponent_bits = rng.integers(1, 254, size=2000, dtype=np.uint32)  # avoid 0/255
    sign_bits = rng.integers(0, 2, size=2000, dtype=np.uint32)
    bits = (sign_bits << 31) | (exponent_bits << 23) | mantissa_bits
    a = bits.view(np.float32)
    assert np.all(np.isfinite(a))

    out = _roundtrip(a)
    _assert_bytes_exact(a, out)

    # explicit demonstration that float64 delta then downcast can lose bits,
    # to document *why* this pattern matters for future schemes (not itself
    # a round-trip assertion on the codec)
    widened = a.astype(np.float64)
    diffs64 = np.diff(widened)
    diffs32_direct = np.diff(a)
    back_down = diffs64.astype(np.float32)
    # not asserted equal on purpose -- this commonly differs, which is the
    # whole point of the bait; kept here as executable documentation.
    _ = (diffs32_direct, back_down)


@pytest.mark.parametrize("value_bits", [
    0x7F7FFFFF,  # float32 max
    0xFF7FFFFF,  # float32 min (most negative)
    0x00800000,  # float32 smallest positive normal
    0x00000001,  # float32 smallest positive subnormal
    0x34000000,  # a value with a long exact binary fraction
])
def test_float32_boundary_bit_patterns(value_bits):
    a = np.array([value_bits], dtype=np.uint32).view(np.float32)
    out = _roundtrip(a)
    _assert_bytes_exact(a, out)


def test_mixed_specials_and_denormals_at_various_lengths():
    specials = np.array(
        [np.nan, np.inf, -np.inf, 0.0, -0.0,
         np.finfo(np.float64).tiny, -np.finfo(np.float64).tiny],
        dtype=np.float64,
    )
    for n in [0, 1, 2, 3, 7, 8, 100, 4095, 4096, 4097]:
        rng = np.random.default_rng(n)
        base = rng.standard_normal(n)
        if n:
            n_special = min(n, len(specials))
            idx = rng.choice(n, size=n_special, replace=False)
            base[idx] = specials[:n_special]
        a = base.astype(np.float64)
        out = _roundtrip(a)
        _assert_bytes_exact(a, out)
