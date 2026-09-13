"""Bitstream robustness: truncated buffers, corrupted headers, corrupted
payloads, and pure garbage must all fail cleanly (a raised exception) --
never hang, never segfault, never silently return an array of the wrong
length or a wrong dtype without raising.

This module never asserts that corruption is *detected as wrong content*
(that would require a cryptographic checksum tscodec doesn't have and isn't
asked to have) -- only that the failure mode is always a clean, catchable
exception, or, when decode happens to succeed, that its own internal
invariants (declared dtype/shape/count) still hold.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tscodec.codec import MAGIC, decode, encode

CLEAN_EXCEPTIONS = (ValueError, TypeError, EOFError)


def _sample_arrays():
    rng = np.random.default_rng(3)
    return [
        np.arange(10, dtype=np.int32),
        np.arange(1000, dtype=np.int64),
        rng.standard_normal(500).astype(np.float64),
        np.array([], dtype=np.float32),
        np.array([True, False, True], dtype=bool),
    ]


# ---------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("array_idx", range(len(_sample_arrays())))
def test_truncation_at_every_prefix_length_raises_cleanly(array_idx):
    a = _sample_arrays()[array_idx]
    buf = encode(a)
    for cut in range(len(buf)):
        prefix = buf[:cut]
        try:
            out = decode(prefix)
        except CLEAN_EXCEPTIONS:
            continue
        except Exception as exc:  # pragma: no cover - this is the failure we're hunting for
            pytest.fail(
                f"decode of a {cut}-byte truncated buffer (from a {len(buf)}-byte "
                f"encoding of a {a.dtype} array of length {a.shape[0]}) raised "
                f"{type(exc).__name__}: {exc!r} instead of a clean exception"
            )
        else:
            # Only acceptable if it happens to still be correct (extremely
            # unlikely for a truncated buffer, but not itself a crash).
            assert out.tobytes() == a.tobytes()[: len(out.tobytes())]


def test_empty_buffer_raises_cleanly():
    with pytest.raises(CLEAN_EXCEPTIONS):
        decode(b"")


def test_single_byte_buffer_raises_cleanly():
    with pytest.raises(CLEAN_EXCEPTIONS):
        decode(b"\x00")


# ---------------------------------------------------------------------
# corrupted header fields
# ---------------------------------------------------------------------


def test_corrupted_magic_every_byte_raises_cleanly():
    a = np.arange(50, dtype=np.int64)
    buf = bytearray(encode(a))
    for i in range(4):
        for bad in (0x00, 0xFF):
            corrupted = bytearray(buf)
            corrupted[i] = bad
            if bytes(corrupted[0:4]) == MAGIC:
                continue
            with pytest.raises(CLEAN_EXCEPTIONS):
                decode(bytes(corrupted))


@pytest.mark.parametrize("version", [1, 2, 42, 128, 255])
def test_every_nonzero_version_raises_cleanly(version):
    a = np.arange(50, dtype=np.int64)
    buf = bytearray(encode(a))
    buf[4] = version
    with pytest.raises(CLEAN_EXCEPTIONS):
        decode(bytes(buf))


@pytest.mark.parametrize("dtype_code", [11, 12, 50, 100, 255])
def test_unknown_dtype_code_raises_cleanly(dtype_code):
    a = np.arange(50, dtype=np.int64)
    buf = bytearray(encode(a))
    buf[5] = dtype_code
    with pytest.raises(CLEAN_EXCEPTIONS):
        decode(bytes(buf))


@pytest.mark.parametrize("count_override", [0, 1, 2**32, 2**64 - 1, 10**9])
def test_corrupted_length_field_raises_cleanly_or_is_self_consistent(count_override):
    a = np.arange(100, dtype=np.int64)
    buf = bytearray(encode(a))
    buf[6:14] = (count_override % (2**64)).to_bytes(8, "little")
    try:
        out = decode(bytes(buf))
    except CLEAN_EXCEPTIONS:
        return
    # If it didn't raise, the declared count must actually match what came
    # back -- silently returning a different length than declared is exactly
    # the failure mode this test bans.
    assert out.shape[0] == count_override % (2**64)


def test_length_field_huge_does_not_hang_or_allocate_absurdly():
    """A declared count of ~2**63 must fail fast (a cheap arithmetic/size
    check), not attempt to actually materialise anything close to that many
    elements."""
    a = np.arange(10, dtype=np.int64)
    buf = bytearray(encode(a))
    buf[6:14] = (2**63).to_bytes(8, "little")
    start = time.monotonic()
    with pytest.raises(CLEAN_EXCEPTIONS):
        decode(bytes(buf))
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"decode of a bogus huge-count header took {elapsed:.2f}s"


# ---------------------------------------------------------------------
# corrupted payload
# ---------------------------------------------------------------------


def test_random_single_byte_flips_in_payload_never_crash():
    """Flip one random payload byte at a time across many trials and many
    base arrays; every outcome must be either a clean exception or a
    self-consistent (dtype/shape-correct) array. An uncaught exception type
    (struct.error, zlib.error, IndexError, OverflowError, ...) leaking out
    of decode() is the bug this test hunts for."""
    rng = np.random.default_rng(4)
    silent_mismatches = 0
    trials = 0
    for a in _sample_arrays():
        if a.shape[0] == 0:
            continue
        buf = bytearray(encode(a))
        header_size = 14
        if len(buf) <= header_size:
            continue
        n_trials = 200
        for _ in range(n_trials):
            trials += 1
            idx = rng.integers(header_size, len(buf))
            bit = 1 << rng.integers(0, 8)
            corrupted = bytearray(buf)
            corrupted[idx] ^= bit
            try:
                out = decode(bytes(corrupted))
            except CLEAN_EXCEPTIONS:
                continue
            except Exception as exc:  # pragma: no cover
                pytest.fail(
                    f"single-byte payload flip raised {type(exc).__name__}: {exc!r} "
                    f"(array dtype={a.dtype}, len={a.shape[0]}, byte_idx={idx}, bit={bit})"
                )
            else:
                assert out.dtype == a.dtype
                assert out.shape == a.shape
                if out.tobytes() != a.tobytes():
                    silent_mismatches += 1
    # Not a hard requirement (zlib's adler32 does most of the work), but if
    # a large fraction of corruptions are silently "successful" with wrong
    # content, that's worth knowing about going forward.
    if trials:
        rate = silent_mismatches / trials
        assert rate < 0.5, (
            f"unexpectedly high rate of silently-corrupted-but-accepted payloads: "
            f"{silent_mismatches}/{trials} ({rate:.1%})"
        )


def test_corrupted_payload_declared_size_mismatch_still_raises():
    """Craft a payload that decompresses successfully but to the wrong
    number of bytes for the declared header count; decode must catch this
    via its own post-decompression length check, not just trust zlib."""
    import zlib

    raw = np.arange(5, dtype=np.int64).tobytes()  # 40 bytes
    payload = zlib.compress(raw, 6)
    from tscodec.codec import _HEADER_STRUCT, FORMAT_VERSION, MAGIC as _MAGIC

    # declare count=6 (48 bytes expected) but payload only decompresses to 40
    header = _HEADER_STRUCT.pack(_MAGIC, FORMAT_VERSION, 4, 6)  # dtype_code 4 = int64
    with pytest.raises(ValueError):
        decode(header + payload)


def test_decompression_of_oversized_payload_relative_to_declared_count_raises():
    """The mirror image: a payload that decompresses to *more* bytes than a
    small declared count promises. Demonstrates that decode() currently
    performs the full decompression before checking size (see audit notes:
    this means a small malicious/corrupted buffer can force an arbitrarily
    large decompression before being rejected -- a potential decompression-
    bomb surface, not currently guarded against). This test only asserts
    that -- for a moderate expansion factor that's safe to actually run in
    CI -- decode still raises cleanly and reasonably quickly; it does not
    prove protection against a much larger crafted bomb."""
    import zlib

    from tscodec.codec import _HEADER_STRUCT, FORMAT_VERSION, MAGIC as _MAGIC

    big_raw = bytes(5_000_000)  # 5 MB of zeros - compresses to a few KB
    payload = zlib.compress(big_raw, 6)
    assert len(payload) < 20_000  # sanity: payload is tiny relative to decompressed size

    # header declares only 10 uint8 elements (10 bytes expected)
    header = _HEADER_STRUCT.pack(_MAGIC, FORMAT_VERSION, 5, 10)  # dtype_code 5 = uint8
    start = time.monotonic()
    with pytest.raises(ValueError):
        decode(header + payload)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0


# ---------------------------------------------------------------------
# pure garbage
# ---------------------------------------------------------------------


def test_pure_random_garbage_never_crashes():
    rng = np.random.default_rng(5)
    for _ in range(500):
        n = int(rng.integers(0, 300))
        buf = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
        try:
            decode(buf)
        except CLEAN_EXCEPTIONS:
            continue
        except Exception as exc:  # pragma: no cover
            pytest.fail(
                f"decode of {n} random bytes raised {type(exc).__name__}: {exc!r} "
                f"(buf={buf!r})"
            )


def test_garbage_with_valid_magic_and_version_never_crashes():
    """Random garbage that at least starts with the right magic+version, so
    it exercises the dtype/count/payload parsing path instead of bailing out
    on the very first check."""
    rng = np.random.default_rng(6)
    for _ in range(500):
        n = int(rng.integers(14, 300))
        tail = rng.integers(0, 256, size=n - 6, dtype=np.uint8).tobytes()
        buf = MAGIC + bytes([0]) + tail
        try:
            decode(buf)
        except CLEAN_EXCEPTIONS:
            continue
        except Exception as exc:  # pragma: no cover
            pytest.fail(
                f"decode of magic-prefixed garbage raised {type(exc).__name__}: {exc!r} "
                f"(buf={buf!r})"
            )
