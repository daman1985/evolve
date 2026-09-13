"""The no-expansion guarantee, stressed exactly where header overhead bites:
small arrays.

PLAN.md gate 3 requires ``ratio_d >= 0.99`` on whole (>=1MB) benchmark
columns; measured that way, header overhead is invisible. This file checks
the same guarantee at the sizes where a fixed per-call header cost is not
invisible: 0, 1, 2, 3, 7, 8, 16, 100, 1000, 10000 elements, for every
supported dtype, using genuinely incompressible data (uniform random bits,
matching the bench corpus's own incompressible-column generator).

Bound (from PLAN.md's framing of the guarantee, applied per-array rather
than per-corpus):

    len(encode(a)) <= max(len(a.tobytes()) * 1.01, len(a.tobytes()) + 64)

At round 0 this already holds everywhere we checked (zlib's stored-block
mode is cheap enough that the naive header+zlib codec doesn't blow the
bound at these sizes) -- see the auditor round-0 report for the actual
numbers. That is exactly the kind of thing that can regress silently once
per-block framing, bit-packing, and scheme-selection metadata show up in
round 1: a per-block header repeated over many tiny blocks, or a scheme
tag plus a raw fallback that still carries its own overhead, can blow this
bound even though the *dataset*-level ratio gate still passes fine.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from tscodec.codec import DTYPE_CODES, encode

from _adversarial import rand_incompressible

LENGTHS = [0, 1, 2, 3, 7, 8, 16, 100, 1000, 10000]
DTYPES = [np.dtype(name) for name in DTYPE_CODES.values()]


def _bound(raw_len: int) -> int:
    return max(int(raw_len * 1.01), raw_len + 64)


def _name_seed(name: str) -> int:
    """Deterministic (non-hash-randomized) integer derived from a dtype
    name, safe to fold into a numpy seed across Python processes."""
    return zlib.crc32(name.encode("ascii"))


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: d.name)
def test_no_expansion_incompressible_small(dtype, n):
    rng = np.random.default_rng([0xC0FFEE, n, _name_seed(dtype.name)])
    a = rand_incompressible(dtype, n, rng)
    enc = encode(a)
    raw_len = len(a.tobytes())
    limit = _bound(raw_len)
    assert len(enc) <= limit, (
        f"no-expansion bound violated: dtype={dtype.name} n={n} raw={raw_len} "
        f"encoded={len(enc)} bound={limit} (overhead={len(enc) - raw_len} bytes)"
    )


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: d.name)
def test_no_expansion_multiple_seeds(dtype, n):
    """Same bound, several independent random draws per (dtype, n), so a
    single lucky/unlucky seed can't hide a marginal violation."""
    for seed in range(5):
        rng = np.random.default_rng([seed, n, _name_seed(dtype.name)])
        a = rand_incompressible(dtype, n, rng)
        enc = encode(a)
        raw_len = len(a.tobytes())
        limit = _bound(raw_len)
        assert len(enc) <= limit, (
            f"no-expansion bound violated: dtype={dtype.name} n={n} seed={seed} "
            f"raw={raw_len} encoded={len(enc)} bound={limit}"
        )


def test_no_expansion_zero_length_every_dtype():
    """n=0 in isolation: header-only overhead against the +64 floor term."""
    for dtype in DTYPES:
        a = np.array([], dtype=dtype)
        enc = encode(a)
        assert len(enc) <= 64, (
            f"empty {dtype.name} array encoded to {len(enc)} bytes, "
            f"expected <= 64 (raw is 0 bytes)"
        )


@pytest.mark.parametrize("n", [6399, 6400, 6401, 65535, 65536, 65537, 131071, 131072, 131073])
def test_no_expansion_near_deflate_block_boundaries(n):
    """zlib emits stored/dynamic-Huffman blocks up to 65535 bytes; check the
    bound right around multiples of that, and around the point where the 1%
    term overtakes the +64 floor term (raw*0.01 == 64 at raw == 6400), for
    the two widest incompressible dtypes."""
    for dtype in (np.dtype(np.int64), np.dtype(np.float64)):
        rng = np.random.default_rng([n, _name_seed(dtype.name)])
        a = rand_incompressible(dtype, n, rng)
        enc = encode(a)
        raw_len = len(a.tobytes())
        limit = _bound(raw_len)
        assert len(enc) <= limit, (
            f"no-expansion bound violated: dtype={dtype.name} n={n} raw={raw_len} "
            f"encoded={len(enc)} bound={limit}"
        )
