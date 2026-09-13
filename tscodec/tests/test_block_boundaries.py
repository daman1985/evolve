"""Block-boundary edge cases.

Round 0's codec has no block framing at all (it's one zlib stream over the
whole array), so none of this can fail *yet*. It exists so that round 1,
which is specified to add per-block framing, lands on a suite that already
parametrizes over every plausible block size and the lengths immediately
around its boundaries. Whatever block size round 1 actually picks, these
tests stay meaningful without being rewritten.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscodec.codec import decode, encode

from _adversarial import (
    alternating_extremes,
    delta_of_delta_wrap,
    monotonic_delta_overflow,
    run_with_boundary_transition,
    uint_max_neighbors,
)

# Plausible block sizes a round-1 implementation might pick, per the task
# brief. Keep in sync with that list if the manager narrows it down later.
BLOCK_SIZES = [1024, 4096, 8192, 16384, 65536]

BOUNDARY_DTYPES = [np.int64, np.uint64, np.int8, np.uint8]
RANDOM_DTYPES = [np.int64, np.uint64, np.float64, np.float32, np.int8, np.uint8, np.bool_]

OVERFLOW_PATTERNS = {
    "alternating_extremes": alternating_extremes,
    "monotonic_delta_overflow": monotonic_delta_overflow,
    "delta_of_delta_wrap": delta_of_delta_wrap,
    "uint_max_neighbors": uint_max_neighbors,
}


def _lengths_around(block_size: int) -> list[int]:
    return [block_size - 1, block_size, block_size + 1, 2 * block_size + 1]


def _assert_roundtrip(a: np.ndarray) -> None:
    out = decode(encode(a))
    assert out.dtype == a.dtype, f"dtype mismatch: expected {a.dtype}, got {out.dtype}"
    assert out.shape == a.shape, f"shape mismatch: expected {a.shape}, got {out.shape}"
    assert out.tobytes() == a.tobytes()


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", RANDOM_DTYPES, ids=lambda d: np.dtype(d).name)
def test_block_boundary_random_data(block_size, dtype):
    rng = np.random.default_rng(1_000_000 + block_size)
    for n in _lengths_around(block_size):
        if dtype == np.bool_:
            a = rng.integers(0, 2, size=n).astype(bool)
        elif np.issubdtype(dtype, np.floating):
            a = rng.standard_normal(n).astype(dtype)
        else:
            info = np.iinfo(dtype)
            a = rng.integers(info.min, info.max, size=n, endpoint=True, dtype=dtype)
        _assert_roundtrip(a)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", BOUNDARY_DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("pattern_name", sorted(OVERFLOW_PATTERNS))
def test_block_boundary_overflow_bait(block_size, dtype, pattern_name):
    fn = OVERFLOW_PATTERNS[pattern_name]
    for n in _lengths_around(block_size):
        a = fn(dtype, n)
        _assert_roundtrip(a)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.bool_, np.uint8],
                          ids=lambda d: np.dtype(d).name)
def test_run_transition_exactly_at_block_boundary(block_size, dtype):
    """A run of one value followed by a run of another, with the switch
    landing on exactly index `block_size` (and, for the two-block-plus-one
    length, also at `2 * block_size`). This is the case that breaks a
    per-block constant/RLE detector that assumes a run can't span or align
    perfectly with a block edge."""
    for n in _lengths_around(block_size):
        if n <= 0:
            continue
        a = run_with_boundary_transition(dtype, n, block_size)
        _assert_roundtrip(a)
        if n > 2 * block_size:
            # second transition at the second block boundary too
            a2 = np.array(a, copy=True)
            if dtype == np.bool_:
                a2[2 * block_size:] = False
            elif np.issubdtype(dtype, np.floating):
                a2[2 * block_size:] = np.dtype(dtype).type(0.0)
            else:
                info = np.iinfo(dtype)
                a2[2 * block_size:] = info.min
            _assert_roundtrip(a2)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_block_boundary_all_same_value(block_size):
    for n in _lengths_around(block_size):
        a = np.full(n, 42, dtype=np.int64)
        _assert_roundtrip(a)
