# Round 0 spec — harness, corpus, and naive baseline

Author: manager. Implementer: `codec-engineer`.

Goal: build the measuring apparatus and an honest floor. **No compression cleverness
this round.** The codec you write must be deliberately naive.

## Layout to create (all under `/home/user/evolve/tscodec/`)

```
tscodec/
  __init__.py          # exports encode, decode, __version__
  codec.py             # the codec itself
bench/
  datasets.py          # deterministic corpus generators
  run_bench.py         # the benchmark runner
tests/
  test_roundtrip.py    # basic unit tests (auditor extends later)
results/               # run_bench writes here
README.md              # stub is fine this round
```

## 1. `tscodec/codec.py` — the naive codec

```python
encode(a: np.ndarray) -> bytes
decode(buf: bytes | bytes-like) -> np.ndarray
```

Requirements:
- Accepts 1-D arrays of dtype: `bool`, `int8/16/32/64`, `uint8/16/32/64`,
  `float32`, `float64`. Raise `TypeError` on anything else (2-D, object, etc.).
- Container format, little-endian throughout:
  - magic `b'TSC0'` (4 bytes)
  - format version `uint8` = 0
  - dtype code `uint8` (define a stable table `DTYPE_CODES` mapping code -> numpy
    dtype string; keep codes fixed forever, later rounds append only)
  - element count `uint64`
  - payload
- Payload this round: `zlib.compress(a.tobytes(), 6)`.
- Non-native byte order input must be handled correctly (convert to native on
  encode, note that decode always returns native order).
- `decode` must validate magic and version and raise `ValueError` on mismatch,
  truncated input, or a payload that does not produce exactly `count` elements.
- Empty arrays must round-trip, preserving dtype.

Do **not** add a raw fallback, block framing, or scheme selection this round. The
naive codec is supposed to look bad on the no-expansion gate; that is data.

## 2. `bench/datasets.py` — the corpus

Expose `dev_corpus() -> list[Dataset]` and `holdout_corpus() -> list[Dataset]`
where `Dataset` is a small dataclass `(name, array, note)`. Everything seeded with
explicit `np.random.default_rng(seed)` — running twice must produce identical bytes.
Target ~1.5 MB raw per column (so e.g. ~190k float64 elements, ~1.5M bool elements).
Add a `--quick` mode hook: a module-level function `dev_corpus(scale=1.0)` where
scale shrinks lengths proportionally.

### Dev corpus (seeds 1000..1013, one per column)

1. `ts_ms` — int64. Start at 1_700_000_000_000. Increments of 1000 ms plus integer
   jitter uniform in [-3, 3]; 0.2% of steps are a large gap (uniform 5_000..90_000 ms).
   Cumulative sum, so strictly increasing.
2. `counter` — uint64. Monotone: cumulative sum of Poisson(lam=40) increments,
   with a reset to 0 at ~8 random positions (counter restarts).
3. `gauge_quant` — float64. A slow random walk in [15.0, 35.0] (reflecting), then
   `np.round(x, 2)`. Values are exact multiples of 0.01 up to float representation.
4. `gauge_f32` — float32. Smooth sinusoid + gaussian noise, full float32 precision
   (not quantised).
5. `cat_codes` — int32. 32 distinct values drawn from a skewed (Zipf-ish) categorical
   distribution; values are arbitrary ids in [0, 100000), not 0..31.
6. `flags` — bool. ~3% True, clustered (True comes in short bursts).
7. `sparse` — int64. 95% exact zeros; the rest drawn from a lognormal-ish spread
   over roughly [1, 10**6].
8. `walk_tick` — float64. Price-like: start 100.0, random walk in units of exactly
   0.01 (integer steps in [-3, 3] times 0.01), value computed as
   `np.round(base + steps * 0.01, 2)` so it lies on the tick grid.
9. `periodic` — float64. Sawtooth with period 1024 plus small noise; not quantised.
10. `bursty` — int64 microseconds. Heavy tailed latency: mixture of a tight
    lognormal around ~200 and a 2% tail up to ~5e6.
11. `ids_runs` — int64. Run-length structured: draw a run length from
    geometric(mean ~40) and a value from 5000 distinct ids; repeat.
12. `small_ints` — int16. Values in [-400, 400], smooth-ish (random walk clipped).
13. `rand_int` — int64. `rng.integers` over the full int64 range. **Incompressible.**
14. `rand_f64` — float64. Uniform random bits reinterpreted as float64 **with NaN
    and inf bit patterns excluded** (regenerate or mask exponent so values are
    finite normals) — we want incompressible, not pathological, here.

### Holdout corpus (seeds 9000+, DIFFERENT shape parameters)

Same archetypes but deliberately shifted so a codec tuned to the dev numbers does
not automatically win: different cardinality (e.g. 7 categories and 900 categories),
different jitter scale, different quantisation (3 decimals and 0.5-unit grid),
different run-length mean, a column that switches regime halfway (first half
constant, second half noisy), a `uint32` column, a `float32` column containing
`NaN`, `+inf`, `-inf` and `-0.0` values, and one incompressible `uint64` column.
Aim for 12-14 columns. Write a one-line `note` on each.

## 3. `bench/run_bench.py` — the runner

CLI: `python tscodec/bench/run_bench.py [--quick] [--label ROUND] [--json PATH]`.

For every dataset in dev and holdout:
- `raw = a.nbytes`
- `enc = len(encode(a))`, `ratio = raw / enc`
- verify `decode(encode(a))` is bit-exact: dtype equal, shape equal,
  `.tobytes()` equal. Record pass/fail.
- timing: best-of-3 wall time for encode and for decode, each measured with
  `time.perf_counter` around a single call on the full array; throughput in MB/s
  is `raw / 1e6 / seconds`.

Reference lines, computed on `a.tobytes()` for dev datasets only:
`zlib.compress(b, 6)`, `lzma.compress(b, preset=6)`, and if `zstandard` imports,
`zstd` level 3 and level 19. Report each as a geometric-mean ratio over the dev
corpus. If `zstandard` is missing, print `n/a` — never make it a hard dependency,
and never import it from library code.

Output:
- A per-dataset table: name, dtype, raw MB, encoded bytes, ratio, enc MB/s, dec MB/s, OK.
- `SCORE` = geometric mean of dev ratios. `HOLDOUT_SCORE` = geometric mean of holdout ratios.
- Gate lines, each printed as `GATE <name>: PASS/FAIL (<detail>)`:
  - `lossless` — all round trips bit-exact (dev + holdout)
  - `no_expansion` — min ratio over all datasets >= 0.99; print the worst dataset
  - `decode_speed` — geomean dev decode MB/s >= 100
  - `encode_speed` — geomean dev encode MB/s >= 25
  - (`fuzz` is checked by the test suite, not here; print it as `SKIPPED (see tests)`)
- Reference line block.
- A final `RESULT` line: `RESULT label=<label> score=<x> holdout=<y> gates=<n_pass>/<n_total>`.
- Append one JSON object per run to `tscodec/results/history.jsonl` with all of the
  above (label, timestamp, per-dataset records, score, holdout, gates, references).

`--quick` should use scale ~0.15 so it finishes in a few seconds, and must print
`QUICK MODE - not comparable to full runs` at the top.

Note lzma on ~20MB of data is slow; if a reference line takes more than ~60s for
the whole corpus, cap the reference-line computation to the first 2 MB of each
column and say so in the output header. Ratios stay comparable that way.

## 4. `tests/test_roundtrip.py`

A modest suite (the auditor will expand it): each supported dtype round-trips,
empty array, single element, all-same, random data, non-native byte order,
unsupported dtype raises `TypeError`, corrupt magic raises `ValueError`.
Use plain `unittest` or pytest-style functions; make sure
`python -m pytest tscodec/tests -q` works.

## Definition of done for this round

`python tscodec/bench/run_bench.py --quick` runs clean, `python -m pytest
tscodec/tests -q` passes, and `python tscodec/bench/run_bench.py` produces the
full table. Report the numbers you saw.
