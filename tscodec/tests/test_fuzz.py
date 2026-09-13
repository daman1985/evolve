"""Seeded randomised fuzz loop mixing dtypes, lengths (0..100000), and
adversarial value patterns.

The case count is scalable via the ``TSCODEC_FUZZ_CASES`` environment
variable so this file can run a small number of cases by default (to keep
the whole suite's runtime well under the ~90s budget every round) while
still supporting a deep run:

    TSCODEC_FUZZ_CASES=2000 python -m pytest tscodec/tests/test_fuzz.py -q

The master seed is printed at the start of the run (visible with `-s`, and
always shown by pytest on failure via the assertion message) so any failure
is reproducible from that single number plus the case index.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tscodec.codec import decode, encode

from _adversarial import (
    ALL_DTYPES,
    alternating_extremes,
    constant_with_single_outlier,
    delta_of_delta_wrap,
    monotonic_delta_overflow,
    quantized_with_one_off_grid,
    rand_incompressible,
    run_with_boundary_transition,
    uint_max_neighbors,
)

MASTER_SEED = int(os.environ.get("TSCODEC_FUZZ_SEED", "20240613"))
N_CASES = int(os.environ.get("TSCODEC_FUZZ_CASES", "400"))

print(f"\n[test_fuzz] TSCODEC_FUZZ_SEED={MASTER_SEED} TSCODEC_FUZZ_CASES={N_CASES}")


def _random_length(rng: np.random.Generator) -> int:
    bucket = rng.integers(0, 6)
    if bucket == 0:
        return 0
    if bucket == 1:
        return int(rng.integers(1, 17))
    if bucket == 2:
        return int(rng.integers(17, 1000))
    if bucket == 3:
        return int(rng.integers(1000, 20_000))
    if bucket == 4:
        return int(rng.integers(20_000, 100_001))
    # bucket 5: land exactly on a handful of plausible block-boundary-ish
    # values, since these are cheap and disproportionately likely to catch
    # future off-by-one bugs
    return int(rng.choice([1023, 1024, 1025, 4095, 4096, 4097,
                            8191, 8192, 8193, 16383, 16384, 16385,
                            65535, 65536, 65537]))


def _random_dtype(rng: np.random.Generator) -> np.dtype:
    idx = int(rng.integers(0, len(ALL_DTYPES)))
    return np.dtype(ALL_DTYPES[idx])


def _make_case(rng: np.random.Generator, case_idx: int) -> np.ndarray:
    dtype = _random_dtype(rng)
    n = _random_length(rng)
    kind = rng.integers(0, 9)

    if dtype == np.dtype(np.bool_):
        if n == 0:
            return np.array([], dtype=bool)
        if kind in (0, 1):
            return rng.integers(0, 2, size=n).astype(bool)
        if kind == 2:
            return np.zeros(n, dtype=bool)
        if kind == 3:
            return np.ones(n, dtype=bool)
        return constant_with_single_outlier(dtype, n)

    if np.issubdtype(dtype, np.floating):
        if kind == 0:
            return rand_incompressible(dtype, n, rng)
        if kind == 1:
            with np.errstate(over="ignore"):
                return (rng.standard_normal(n) * rng.choice([1.0, 1e-10, 1e10, 1e300])).astype(dtype)
        if kind == 2:
            return quantized_with_one_off_grid(n, decimals=int(rng.integers(0, 6)), dtype=dtype)
        if kind == 3 and n > 0:
            return constant_with_single_outlier(dtype, n)
        if kind == 4 and n > 0:
            base = rng.standard_normal(n).astype(dtype)
            n_special = min(n, 5)
            if n_special:
                idx = rng.choice(n, size=n_special, replace=False)
                specials = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=dtype)
                base[idx] = specials[:n_special]
            return base
        if kind == 5:
            return np.full(n, rng.choice([0.0, -0.0, 1.5, -1.5]), dtype=dtype)
        return rng.standard_normal(n).astype(dtype)

    # integer dtypes
    if kind == 0:
        return rand_incompressible(dtype, n, rng)
    if kind == 1:
        return alternating_extremes(dtype, n)
    if kind == 2:
        return monotonic_delta_overflow(dtype, n)
    if kind == 3:
        return delta_of_delta_wrap(dtype, n)
    if kind == 4:
        return uint_max_neighbors(dtype, n)
    if kind == 5 and n > 0:
        return constant_with_single_outlier(dtype, n)
    if kind == 6 and n > 0:
        boundary = int(rng.integers(0, n + 1))
        return run_with_boundary_transition(dtype, n, boundary)
    if kind == 7:
        # NB: pick the fill value by indexing a plain Python list, not
        # `rng.choice([info.min, info.max, 0])` -- numpy would first build a
        # float64 array from that list to hand to `choice`, and info.max for
        # (u)int64 (e.g. 2**64-1) is not exactly representable in float64,
        # silently rounding to 2**64 and producing an out-of-range/undefined
        # cast into the target dtype.
        info = np.iinfo(dtype)
        options = [info.min, info.max, 0]
        fill = options[int(rng.integers(0, len(options)))]
        return np.full(n, fill, dtype=dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=n, endpoint=True, dtype=dtype)


@pytest.mark.parametrize("case_idx", range(N_CASES))
def test_fuzz_case(case_idx):
    rng = np.random.default_rng([MASTER_SEED, case_idx])
    a = _make_case(rng, case_idx)
    try:
        encoded = encode(a)
        out = decode(encoded)
    except Exception:
        print(
            f"[test_fuzz] FAILURE seed={MASTER_SEED} case_idx={case_idx} "
            f"dtype={a.dtype} shape={a.shape}"
        )
        raise
    assert out.dtype == a.dtype, (
        f"seed={MASTER_SEED} case_idx={case_idx}: dtype mismatch "
        f"expected {a.dtype} got {out.dtype}"
    )
    assert out.shape == a.shape, (
        f"seed={MASTER_SEED} case_idx={case_idx}: shape mismatch "
        f"expected {a.shape} got {out.shape}"
    )
    assert out.tobytes() == a.tobytes(), (
        f"seed={MASTER_SEED} case_idx={case_idx}: byte mismatch for dtype={a.dtype}, "
        f"n={a.shape[0]} -- rerun with TSCODEC_FUZZ_SEED={MASTER_SEED} to reproduce "
        f"case {case_idx}"
    )
