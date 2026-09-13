"""Structural edge cases: constant columns, single outliers, near-quantised
data with one value off the grid, and long arrays.

None of this is meaningful yet against the round-0 zlib-only codec (there is
no run-length, constant, or quantisation detector to fool), but round 1-2
are specified to add exactly those, and this is where they go wrong: an
outlier that gets silently absorbed into "the constant", a quantisation
detector that rounds an off-grid value onto the grid instead of falling
back, a run that is one element short of what the detector assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscodec.codec import DTYPE_CODES, decode, encode

from _adversarial import constant_with_single_outlier, quantized_with_one_off_grid

DTYPES = [np.dtype(name) for name in DTYPE_CODES.values()]


def _assert_roundtrip(a: np.ndarray) -> None:
    out = decode(encode(a))
    assert out.dtype == a.dtype
    assert out.shape == a.shape
    assert out.tobytes() == a.tobytes()


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: d.name)
def test_all_zeros(dtype):
    for n in [0, 1, 2, 100, 10000]:
        a = np.zeros(n, dtype=dtype)
        _assert_roundtrip(a)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: d.name)
def test_all_one_nonzero_value(dtype):
    if dtype == np.dtype(bool):
        fill = True
    elif np.issubdtype(dtype, np.floating):
        fill = -3.5
    else:
        fill = np.iinfo(dtype).max
    for n in [1, 2, 100, 10000]:
        a = np.full(n, fill, dtype=dtype)
        _assert_roundtrip(a)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: d.name)
@pytest.mark.parametrize("n", [1, 2, 3, 100, 4096, 10000])
def test_single_outlier_in_constant_column(dtype, n):
    a = constant_with_single_outlier(dtype, n)
    _assert_roundtrip(a)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: d.name)
def test_outlier_at_every_position_small_array(dtype):
    """Cheap exhaustive sweep: for a small array, put the single outlier at
    every possible index in turn, so an off-by-one in a "first/last element
    is special-cased" optimisation can't hide."""
    n = 9
    for idx in range(n):
        a = constant_with_single_outlier(dtype, n, outlier_index=idx)
        _assert_roundtrip(a)


@pytest.mark.parametrize("decimals", [0, 1, 2, 3, 6])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_quantized_with_one_value_off_grid(dtype, decimals):
    for n in [2, 3, 100, 4096]:
        a = quantized_with_one_off_grid(n, decimals=decimals, dtype=dtype)
        _assert_roundtrip(a)


def test_quantized_perfectly_on_grid_no_outlier():
    """Control case: perfectly quantised data with *no* off-grid value,
    still must round-trip byte-exactly even if a future scheme decides to
    apply a quantisation transform."""
    rng = np.random.default_rng(55)
    for decimals in [0, 1, 2, 3]:
        a = np.round(rng.uniform(-1000, 1000, size=5000), decimals)
        _assert_roundtrip(a)


def test_long_monotonic_run_no_outlier():
    a = np.arange(200_000, dtype=np.int64)
    _assert_roundtrip(a)


def test_very_long_array_random():
    rng = np.random.default_rng(9)
    a = rng.standard_normal(500_000).astype(np.float64)
    _assert_roundtrip(a)


def test_very_long_array_constant():
    a = np.full(500_000, 7, dtype=np.int32)
    _assert_roundtrip(a)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
def test_two_distinct_values_alternating(dtype):
    """Lowest possible cardinality above 1: exactly two distinct values,
    strictly alternating. Cheap bait for a dictionary/RLE scheme that
    assumes runs are long."""
    if dtype == np.float64:
        vals = np.array([1.0, -1.0])
    else:
        info = np.iinfo(dtype)
        vals = np.array([info.min, info.max], dtype=dtype)
    for n in [2, 3, 1000, 8191, 8192, 8193]:
        a = np.resize(vals, n).astype(dtype)
        _assert_roundtrip(a)
