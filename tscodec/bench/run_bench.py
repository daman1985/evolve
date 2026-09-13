#!/usr/bin/env python3
"""Benchmark runner for tscodec.

Usage:
    python tscodec/bench/run_bench.py [--quick] [--label ROUND] [--json PATH]

Encodes and decodes every dataset in the dev and holdout corpora, verifies
bit-exact round trips, times encode/decode (best of 3), reports per-dataset
ratios plus SCORE (geomean of dev ratios) and HOLDOUT_SCORE, checks the four
measurable hard gates (lossless, no_expansion, decode_speed, encode_speed —
`fuzz` is the property test suite's job, not this script's), and reports
zlib/lzma/zstd reference lines on the dev corpus. Appends one JSON record per
run to tscodec/results/history.jsonl; --json PATH additionally writes the same
record (pretty-printed) to an arbitrary path.
"""

from __future__ import annotations

import argparse
import json
import lzma
import pathlib
import sys
import time
import zlib

import numpy as np

# `run_bench.py` is invoked as a script (`python tscodec/bench/run_bench.py`),
# so Python only puts its own directory on sys.path. Add this project's root
# (for the sibling `datasets` module and for `import tscodec` to find the
# *inner* `tscodec/tscodec/` package directory) explicitly.
#
# Careful: the project root here is .../tscodec, which itself contains a
# child directory *also* named tscodec/ (the actual package, with
# __init__.py). Putting the project root's *parent* on sys.path instead would
# make "import tscodec" resolve the outer, __init__-less project directory as
# a PEP 420 namespace package and shadow the real one -- it must be the
# project root itself that goes on sys.path.
_THIS_DIR = pathlib.Path(__file__).resolve().parent  # .../tscodec/bench
_PROJECT_DIR = _THIS_DIR.parent  # .../tscodec (contains the tscodec/ package)
for _p in (str(_PROJECT_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import datasets  # noqa: E402  (sibling module; path fixed up above)
import tscodec  # noqa: E402  (path fixed up above)

try:
    import zstandard

    _HAVE_ZSTD = True
except ImportError:
    _HAVE_ZSTD = False

REFERENCE_CAP_BYTES = 2_000_000
HISTORY_PATH = _PROJECT_DIR / "results" / "history.jsonl"

GATE_DECODE_MB_S = 100.0
GATE_ENCODE_MB_S = 25.0
GATE_MIN_RATIO = 0.99


def geomean(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if np.any(arr <= 0):
        raise ValueError("geomean requires strictly positive values")
    return float(np.exp(np.mean(np.log(arr))))


def _best_of_3(fn):
    """Run fn() three times, return (best wall time, result of the last call)."""
    best = None
    result = None
    for _ in range(3):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        if best is None or dt < best:
            best = dt
    return best, result


def _measure_dataset(ds: "datasets.Dataset", corpus_name: str) -> dict:
    a = ds.array
    raw_bytes = a.nbytes

    enc_time, encoded = _best_of_3(lambda: tscodec.encode(a))
    dec_time, decoded = _best_of_3(lambda: tscodec.decode(encoded))

    ok = (
        decoded.dtype == a.dtype
        and decoded.shape == a.shape
        and decoded.tobytes() == a.tobytes()
    )

    encoded_bytes = len(encoded)
    ratio = raw_bytes / encoded_bytes if encoded_bytes else float("inf")
    enc_mb_s = (raw_bytes / 1e6 / enc_time) if enc_time > 0 else float("inf")
    dec_mb_s = (raw_bytes / 1e6 / dec_time) if dec_time > 0 else float("inf")

    return {
        "corpus": corpus_name,
        "name": ds.name,
        "dtype": str(a.dtype),
        "n": int(a.shape[0]),
        "raw_bytes": int(raw_bytes),
        "encoded_bytes": int(encoded_bytes),
        "ratio": ratio,
        "encode_mb_s": enc_mb_s,
        "decode_mb_s": dec_mb_s,
        "roundtrip_ok": bool(ok),
        "note": ds.note,
    }


def _reference_ratio(raw_blobs: list[bytes], compress_fn) -> float:
    ratios = []
    for raw in raw_blobs:
        capped = raw[:REFERENCE_CAP_BYTES]
        comp = compress_fn(capped)
        ratios.append(len(capped) / len(comp))
    return geomean(ratios)


def main() -> int:
    ap = argparse.ArgumentParser(description="tscodec benchmark runner")
    ap.add_argument("--quick", action="store_true", help="small/fast corpus for a smoke check")
    ap.add_argument("--label", default="unlabeled", help="label recorded with this run")
    ap.add_argument("--json", default=None, help="also write this run's record to PATH")
    args = ap.parse_args()

    scale = 0.15 if args.quick else 1.0
    if args.quick:
        print("QUICK MODE - not comparable to full runs")

    dev = datasets.dev_corpus(scale=scale)
    holdout = datasets.holdout_corpus(scale=scale)

    print(
        f"(reference compression lines computed over <= "
        f"{REFERENCE_CAP_BYTES / 1e6:.1f} MB per column, for comparability across runs)"
    )
    print()

    records = []
    header = (
        f"{'name':<20}{'corpus':<9}{'dtype':<9}{'raw MB':>9}"
        f"{'enc bytes':>12}{'ratio':>9}{'enc MB/s':>11}{'dec MB/s':>11}   OK"
    )
    print(header)
    print("-" * len(header))
    for corpus_name, corpus in (("dev", dev), ("holdout", holdout)):
        for ds in corpus:
            rec = _measure_dataset(ds, corpus_name)
            records.append(rec)
            print(
                f"{rec['name']:<20}{rec['corpus']:<9}{rec['dtype']:<9}"
                f"{rec['raw_bytes'] / 1e6:>9.3f}{rec['encoded_bytes']:>12d}"
                f"{rec['ratio']:>9.3f}{rec['encode_mb_s']:>11.1f}"
                f"{rec['decode_mb_s']:>11.1f}   {'OK' if rec['roundtrip_ok'] else 'FAIL'}"
            )

    dev_records = [r for r in records if r["corpus"] == "dev"]
    holdout_records = [r for r in records if r["corpus"] == "holdout"]
    all_records = dev_records + holdout_records

    score = geomean(r["ratio"] for r in dev_records)
    holdout_score = geomean(r["ratio"] for r in holdout_records)

    lossless_ok = all(r["roundtrip_ok"] for r in all_records)
    n_fail = sum(1 for r in all_records if not r["roundtrip_ok"])
    worst = min(all_records, key=lambda r: r["ratio"])
    no_expansion_ok = worst["ratio"] >= GATE_MIN_RATIO

    dec_speed = geomean(r["decode_mb_s"] for r in dev_records)
    enc_speed = geomean(r["encode_mb_s"] for r in dev_records)

    gates = [
        (
            "lossless",
            lossless_ok,
            "all round trips bit-exact"
            if lossless_ok
            else f"{n_fail}/{len(all_records)} round trips failed",
        ),
        (
            "no_expansion",
            no_expansion_ok,
            f"worst={worst['name']} ({worst['corpus']}) ratio={worst['ratio']:.4f}",
        ),
        ("decode_speed", dec_speed >= GATE_DECODE_MB_S, f"geomean dev decode = {dec_speed:.1f} MB/s"),
        ("encode_speed", enc_speed >= GATE_ENCODE_MB_S, f"geomean dev encode = {enc_speed:.1f} MB/s"),
    ]

    print()
    print(f"SCORE (dev geomean ratio)     = {score:.4f}")
    print(f"HOLDOUT_SCORE (holdout geomean ratio) = {holdout_score:.4f}")
    print()
    for gate_name, passed, detail in gates:
        print(f"GATE {gate_name}: {'PASS' if passed else 'FAIL'} ({detail})")
    print("GATE fuzz: SKIPPED (see tests)")

    # -- reference lines (dev corpus only) --
    print()
    print("reference lines (dev corpus, geometric-mean ratio, raw bytes):")
    dev_raw = [ds.array.tobytes() for ds in dev]
    references = {}
    references["zlib-6"] = _reference_ratio(dev_raw, lambda b: zlib.compress(b, 6))
    references["lzma-6"] = _reference_ratio(dev_raw, lambda b: lzma.compress(b, preset=6))
    if _HAVE_ZSTD:
        c3 = zstandard.ZstdCompressor(level=3)
        c19 = zstandard.ZstdCompressor(level=19)
        references["zstd-3"] = _reference_ratio(dev_raw, lambda b: c3.compress(b))
        references["zstd-19"] = _reference_ratio(dev_raw, lambda b: c19.compress(b))
    else:
        references["zstd-3"] = None
        references["zstd-19"] = None

    for k, v in references.items():
        print(f"  {k:<10} {'n/a' if v is None else f'{v:.3f}'}")

    n_pass = sum(1 for _, p, _ in gates)
    n_total = len(gates)
    print()
    print(
        f"RESULT label={args.label} score={score:.4f} holdout={holdout_score:.4f} "
        f"gates={n_pass}/{n_total}"
    )

    record = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick": args.quick,
        "scale": scale,
        "datasets": records,
        "score": score,
        "holdout_score": holdout_score,
        "gates": [{"name": n, "passed": bool(p), "detail": d} for n, p, d in gates],
        "references": references,
    }

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(record, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
