"""Shared adversarial data generators for the tscodec test suite.

Not a test module itself (no ``test_`` prefix, so pytest won't collect it);
imported by the various ``test_*.py`` files. Every generator is a pure
function of ``(dtype, n[, rng])`` so callers can reproduce any failure by
recording the inputs that were passed in.

These patterns exist specifically to bait transforms that round 1+ is
expected to introduce: delta, delta-of-delta, frame-of-reference, and
bit-packing. None of them should ever cause anything other than a correct,
byte-exact round trip -- if a future scheme computes an intermediate delta
in a fixed-width type without checking for overflow, these are the arrays
that will catch it.
"""

from __future__ import annotations

import numpy as np

INTEGER_DTYPES = [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
]
FLOAT_DTYPES = [np.float32, np.float64]
ALL_DTYPES = INTEGER_DTYPES + FLOAT_DTYPES + [np.bool_]


def _tile_to_length(values: list, n: int) -> list:
    if n == 0:
        return []
    reps = (n + len(values) - 1) // len(values)
    return (values * reps)[:n]


def alternating_extremes(dtype, n: int) -> np.ndarray:
    """[min, max, min, max, ...] -- the classic first-difference overflow
    bait: max - min overflows the dtype's own signed/unsigned range."""
    dtype = np.dtype(dtype)
    info = np.iinfo(dtype)
    return np.array(_tile_to_length([int(info.min), int(info.max)], n), dtype=dtype)


def monotonic_delta_overflow(dtype, n: int) -> np.ndarray:
    """A monotone increasing sequence spanning the full dtype range in as
    few steps as possible, so each first-difference is enormous (near the
    dtype's full width). Monotone data is exactly what a naive engineer
    expects delta-coding to shine on; this checks the huge-delta case still
    round-trips instead of silently wrapping or clipping."""
    dtype = np.dtype(dtype)
    info = np.iinfo(dtype)
    lo, hi = int(info.min), int(info.max)
    if n <= 0:
        return np.array([], dtype=dtype)
    if n == 1:
        return np.array([lo], dtype=dtype)
    step = (hi - lo) // (n - 1)
    vals = [lo + i * step for i in range(n)]
    vals[-1] = hi  # land exactly on the max, regardless of rounding
    return np.array(vals, dtype=dtype)


def delta_of_delta_wrap(dtype, n: int) -> np.ndarray:
    """A zigzag hammering both extremes and the midpoint repeatedly, so
    first differences alternate sign at full magnitude and *second*
    differences ("delta of delta") are on the order of the full dtype
    range -- the case that breaks a delta-of-delta transform implemented in
    a same-width accumulator."""
    dtype = np.dtype(dtype)
    info = np.iinfo(dtype)
    lo, hi = int(info.min), int(info.max)
    mid = 0 if lo < 0 else hi // 2
    pattern = [mid, hi, lo, hi, lo, mid, lo, hi]
    return np.array(_tile_to_length(pattern, n), dtype=dtype)


def uint_max_neighbors(dtype, n: int) -> np.ndarray:
    """Values clustered around the top of an unsigned range (or the
    positive/negative boundary of a signed range), which is where an
    off-by-one in frame-of-reference or bit-width sizing shows up."""
    dtype = np.dtype(dtype)
    info = np.iinfo(dtype)
    hi = int(info.max)
    lo = int(info.min)
    pattern = [hi - 2, hi - 1, hi, lo, lo + 1, lo + 2, 0]
    pattern = [max(lo, min(hi, v)) for v in pattern]
    return np.array(_tile_to_length(pattern, n), dtype=dtype)


def constant_with_single_outlier(dtype, n: int, outlier_index: int | None = None) -> np.ndarray:
    """A constant column except for exactly one element, which is set to
    the dtype's extreme opposite value. Structural bait for run-length /
    constant-detection schemes: they must not corrupt or drop the outlier."""
    dtype = np.dtype(dtype)
    if n == 0:
        return np.array([], dtype=dtype)
    if dtype == np.dtype(np.bool_):
        a = np.zeros(n, dtype=dtype)
        outlier = True
    elif np.issubdtype(dtype, np.floating):
        a = np.full(n, 1.5, dtype=dtype)
        outlier = dtype.type(-1.5)
    else:
        info = np.iinfo(dtype)
        a = np.full(n, info.min, dtype=dtype)
        outlier = info.max
    idx = (n // 2) if outlier_index is None else outlier_index
    a[idx] = outlier
    return a


def run_with_boundary_transition(dtype, n: int, boundary: int) -> np.ndarray:
    """A run of one value followed by a run of another, with the transition
    placed exactly at index ``boundary``. When ``boundary`` coincides with a
    codec's internal block size, this exercises run-length/constant
    detection that must not assume a run continues (or resets) across block
    edges incorrectly."""
    dtype = np.dtype(dtype)
    a = np.empty(n, dtype=dtype)
    if dtype == np.dtype(np.bool_):
        first, second = False, True
    elif np.issubdtype(dtype, np.floating):
        first, second = dtype.type(3.0), dtype.type(-3.0)
    else:
        info = np.iinfo(dtype)
        first, second = info.min, info.max
    b = max(0, min(n, boundary))
    a[:b] = first
    a[b:] = second
    return a


def quantized_with_one_off_grid(n: int, decimals: int = 2, off_index: int | None = None,
                                 dtype=np.float64) -> np.ndarray:
    """Values that look perfectly quantised to `decimals` decimal places
    except one element nudged off the grid by float epsilon -- bait for a
    quantisation-detection scheme that must fall back correctly rather than
    silently rounding the outlier onto the grid."""
    dtype = np.dtype(dtype)
    if n == 0:
        return np.array([], dtype=dtype)
    rng = np.random.default_rng(777)
    base = np.round(rng.uniform(-100, 100, size=n), decimals).astype(dtype)
    idx = (n // 2) if off_index is None else off_index
    eps = np.finfo(dtype).eps * max(abs(float(base[idx])), 1.0) * 4
    base[idx] = base[idx] + dtype.type(eps if eps != 0 else 1e-3)
    return base


def rand_incompressible(dtype, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random bit patterns for any supported dtype -- true
    incompressible data used for no-expansion checks. Floats are re-drawn on
    a non-finite hit so the array stays finite-only (matches bench corpus
    convention); pass a dtype-specific NaN/inf generator separately if you
    want those mixed in."""
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.bool_):
        return rng.integers(0, 2, size=n, dtype=np.uint8).astype(bool)
    if np.issubdtype(dtype, np.floating):
        uint_dtype = {np.dtype(np.float32): np.uint32, np.dtype(np.float64): np.uint64}[dtype]
        info = np.iinfo(uint_dtype)
        bits = rng.integers(0, info.max, size=n, endpoint=True, dtype=uint_dtype)
        values = bits.view(dtype)
        bad = ~np.isfinite(values)
        while bad.any():
            n_bad = int(bad.sum())
            bits[bad] = rng.integers(0, info.max, size=n_bad, endpoint=True, dtype=uint_dtype)
            values = bits.view(dtype)
            bad = ~np.isfinite(values)
        return values
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=n, endpoint=True, dtype=dtype)


PATTERN_FUNCS = {
    "alternating_extremes": alternating_extremes,
    "monotonic_delta_overflow": monotonic_delta_overflow,
    "delta_of_delta_wrap": delta_of_delta_wrap,
    "uint_max_neighbors": uint_max_neighbors,
}
