# tscodec

A lossless columnar codec for numeric NumPy arrays. Dependency-light
(`numpy` + Python stdlib only in the library itself).

```python
import numpy as np
import tscodec

a = np.array([1, 2, 3, 4], dtype=np.int64)
buf = tscodec.encode(a)
b = tscodec.decode(buf)
assert b.dtype == a.dtype and (b == a).all()
```

## Status: round 0 (harness + naive baseline)

`tscodec.codec` currently implements a deliberately naive container: a small
fixed header (magic, format version, dtype code, element count) wrapped
around `zlib.compress` of the raw array bytes. There is no modelling, block
framing, or scheme selection yet — this round exists to stand up the
benchmark corpus (`tscodec/bench/datasets.py`), the benchmark runner
(`tscodec/bench/run_bench.py`), and the correctness test suite
(`tscodec/tests/`), and to record an honest floor for later rounds to beat.

Supported dtypes: `bool`, `int8/16/32/64`, `uint8/16/32/64`, `float32`,
`float64`, as 1-D arrays. `encode` raises `TypeError` on anything else.

As a naive `zlib`-over-raw-bytes codec, round 0 is expected to *fail* the
no-expansion gate on incompressible columns (`rand_int`, `rand_f64`, and
their holdout equivalents) — see `PLAN.md` and `LOG.md` for the gate
definitions and how later rounds address this with a raw-fallback path.

## Guarantees (round 0)

- Bit-exact round trip for every supported dtype, including empty arrays,
  single-element arrays, all-NaN/-0.0/int64-min payloads, and non-native
  byte order input.
- `decode` always returns a native-byte-order array.
- `decode` raises `ValueError` on bad magic, bad version, or truncated/corrupt
  input; `encode` raises `TypeError` on unsupported dtype or ndim != 1.

## Running the benchmark and tests

```
python tscodec/bench/run_bench.py --quick     # fast smoke check
python -m pytest tscodec/tests -q             # correctness suite
python tscodec/bench/run_bench.py --label round0   # full benchmark
```

Full-run results are appended to `tscodec/results/history.jsonl`; the
round-by-round narrative is in `LOG.md` at the repo root.
