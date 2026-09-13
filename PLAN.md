# PLAN.md — `tscodec`: a lossless columnar codec for numeric telemetry

Manager: main thread. Workers: four subagents defined in `.claude/agents/`.
Project lives in `./tscodec/`. Round-by-round narrative lives in `./LOG.md`.

---

## 1. The problem

**Build a lossless compression codec specialised for numeric columns — the kind of
data that comes out of metrics pipelines, IoT sensors, financial ticks, and
analytics tables — that compresses them substantially better than a
general-purpose byte compressor while staying fast enough to actually use.**

Input is a 1-D NumPy array of a fixed numeric dtype (`int8..int64`, `uint8..uint64`,
`float32`, `float64`, `bool`). Output is a self-describing byte string that
decodes back to a **bit-identical** array. The deliverable is a small pure
Python + NumPy library with `encode(array) -> bytes` and `decode(bytes) -> array`,
plus a benchmark harness and a fuzz-based correctness suite.

### Why this problem has real headroom

A general byte compressor sees a `float64` column as a flat stream of bytes and
has to rediscover, from scratch and through a tiny window, structure that is
obvious if you know the data is a numeric column: values are 8-byte aligned,
consecutive values are strongly correlated, exponents and sign bits are nearly
constant, sensor readings are quantised to a few decimal places, counters are
monotonic, categorical codes live in a tiny range, and long runs repeat. The gap
between "gzip the raw bytes" and "model the column, then entropy-code the
residual" is not a few percent — on structured columns it is routinely **5–50×**,
and closing it requires a sequence of genuinely distinct ideas (delta and
delta-of-delta, frame-of-reference, bit-packing, run-length, XOR encoding,
byte-plane transposition, quantisation detection, per-block scheme selection with
a cost model, dictionary encoding). That is exactly the shape that rewards many
rounds of iteration: each round can add one more model to the arsenal and teach
the selector when to reach for it. At the same time there is a hard floor —
incompressible random data must not be expanded — so the work is not a free
one-way ratchet; every new scheme has to earn its place against a cost model.

### Practical use of the final result

A drop-in, dependency-light (`numpy` + stdlib only) column codec. Concrete uses:
shrinking Parquet-less on-disk telemetry archives, cutting the payload size of
metrics shipped over the network, storing feature-store columns, embedding a
compact time-series store in an application that cannot take a C dependency.
It is a library another program imports and builds on, not a score in a notebook.

---

## 2. Success and improvement, numerically

### The benchmark corpus

`tscodec/bench/datasets.py` deterministically generates a **dev corpus** of ~14
columns (seeded, ≥1 MB raw each) spanning realistic column archetypes:

| # | column | archetype |
|---|--------|-----------|
| 1 | `ts_ms` | monotonic millisecond timestamps, mostly-regular interval with jitter |
| 2 | `counter` | monotonically increasing uint64 counter with occasional resets |
| 3 | `gauge_quant` | float64 sensor readings quantised to 2 decimals, slow drift |
| 4 | `gauge_f32` | float32 noisy sensor readings |
| 5 | `cat_codes` | low-cardinality int32 categorical codes (32 distinct) |
| 6 | `flags` | boolean flags, ~3% true |
| 7 | `sparse` | int64, 95% zeros, occasional spikes |
| 8 | `walk_tick` | random walk on a fixed tick size (price-like, float64) |
| 9 | `periodic` | sawtooth/seasonal float64 |
| 10 | `bursty` | heavy-tailed latency samples (int64 microseconds) |
| 11 | `ids_runs` | int64 ids in long repeating runs |
| 12 | `small_ints` | int16 in a narrow band |
| 13 | `rand_int` | **incompressible** uniform random int64 |
| 14 | `rand_f64` | **incompressible** uniform random float64 |

Datasets 13–14 exist to make the score ungameable: any scheme that wins on
structure must detect these and fall back, or it pays for it.

A separate **holdout corpus** is generated with different seeds *and* different
shape parameters (different cardinalities, jitter scales, run lengths, decimal
places, a mixed-regime column, a column with NaNs and infinities). It is scored
every round but is never described to the architect or engineer in parameter
detail — it is the overfitting check.

### Primary metric

For each dataset `d`: `ratio_d = raw_bytes_d / encoded_bytes_d`.

> **SCORE = geometric mean of `ratio_d` over the dev corpus.**

Geometric mean, not arithmetic: it stops one spectacular column (sparse zeros can
hit 1000×) from drowning out everything else, and it makes a relative gain on any
column worth the same as the same relative gain anywhere else.

`HOLDOUT_SCORE` is the same quantity over the holdout corpus and is reported
alongside. A round is only considered a real improvement if **both** move up (or
holdout is flat while dev moves up for a reason the log explains).

### Hard gates — a round's result is invalid unless all pass

1. **Lossless.** For every dev and holdout dataset, `decode(encode(a))` is
   bit-identical to `a`: same dtype, same shape, and `a.tobytes() == out.tobytes()`
   (byte comparison, so NaN payloads and `-0.0` must survive exactly).
2. **Fuzz-lossless.** A property-based suite (≥2000 randomised cases: random
   dtypes, lengths 0…100k, adversarial patterns, all-NaN, all-min/max, single
   element, empty) round-trips bit-exactly.
3. **No expansion.** `ratio_d ≥ 0.99` for **every** dataset, dev and holdout. A
   codec that inflates incompressible data is not shippable.
4. **Decode throughput ≥ 100 MB/s**, geometric mean over the dev corpus, measured
   in *raw* (uncompressed) bytes per second, best of 3 timed runs.
5. **Encode throughput ≥ 25 MB/s**, same measurement convention.

### Reference lines (reported every round, not gates)

`zlib -6`, `lzma -6`, `zstd -3` and `zstd -19` applied to the raw column bytes.
These are what a competent engineer would reach for instead of this library; the
codec has to beat them on SCORE to be worth existing. (`zstandard` is used **only**
in the benchmark for comparison — the shipped codec depends on `numpy` + stdlib.)

### Improvement

"Round N improved on round N−1" means: SCORE strictly increased, all five gates
still pass, and holdout did not regress by more than 2%. Every round's numbers
are appended to `tscodec/results/history.jsonl` and to `LOG.md`.

---

## 3. Delegation structure

Four subagents, defined in `.claude/agents/`. The manager (main thread) never
writes codec code: it sets each round's objective, picks the worker, and records
results.

### `codec-architect` — model: **opus**
**Responsibility.** Given the current benchmark table (per-dataset ratio, bytes,
scheme-selection statistics, timings) and the codec's current design, produce a
written *change specification* for exactly one round: which compression technique
to add or change, why the evidence supports it, which datasets it should move and
by how much, what the bitstream layout change is, and what could go wrong. It
does not edit code.
**Why opus.** This is the only genuinely creative seat in the loop. Choosing
*which* model to add next — reading "column 8 is at 1.9× while zstd -19 gets 2.4×"
and concluding "the tick size is a constant factor, divide it out before delta" —
is diagnosis from sparse evidence, and it compounds: a bad call wastes a whole
round of the other three agents. Worth the most capable model available.

### `codec-engineer` — model: **sonnet**
**Responsibility.** Implement the architect's spec in `tscodec/`: write and edit
the encoder, decoder, bitstream format, and unit tests. Keep the public API
stable, keep the dependency set to `numpy` + stdlib, keep everything vectorised
(no per-element Python loops on hot paths), and make sure the local unit tests
pass before reporting back.
**Why sonnet.** This is well-specified mechanical work — bit-packing with NumPy,
struct headers, dtype plumbing — where the *what* has already been decided.
Sonnet is strong at exactly this kind of contained implementation, and running
the highest-throughput seat in the loop on the frontier model would spend a lot
of budget re-deriving decisions the architect already made.

### `codec-auditor` — model: **sonnet**
**Responsibility.** Adversarially attack correctness after every implementation
round. Own `tscodec/tests/`: extend the fuzz suite, hunt for dtype/overflow/
endianness/empty-input/NaN-payload bugs, check the no-expansion guarantee on
pathological inputs, and confirm the encoder never silently loses precision. It
reports failures with a minimal reproducing case and may fix test files, but does
not redesign the codec.
**Why sonnet.** Adversarial test design needs real reasoning about edge cases —
integer overflow in delta-of-delta, `int64` min, denormal floats — but it is
bounded, checkable work with an objective pass/fail signal. A separate agent from
the implementer is the point: the author of a bug is the worst person to look for
it, and a fresh context re-reads the code without the implementer's assumptions.

### `bench-runner` — model: **haiku**
**Responsibility.** Run `tscodec/bench/run_bench.py`, nothing else. Report the
resulting table verbatim: per-dataset ratios and timings, SCORE, HOLDOUT_SCORE,
gate pass/fail, reference lines. Never edits code, never interprets results,
never "fixes" a failing benchmark.
**Why haiku.** The job is: invoke a script, read its stdout, relay it. Any
reasoning added here is actively harmful — the measurement seat must be
incapable of rationalising a bad number into a good one. It runs 6+ times, so
the cheapest and fastest model is exactly right, and its narrowness is a feature.

---

## 4. Rounds and definition of done

**Six rounds, revised mid-run to four.** (See the schedule-revision entry in
`LOG.md`: rounds 2+3 were merged, and rounds 4+5 were merged, to fit the session's
usage budget. The technical ground covered is unchanged; the round themes below
stand as written, they are simply delivered in four rounds instead of six.)

- **Round 0 — harness and honest baseline.** Build the corpus generator, the
  benchmark runner, the fuzz suite, and a deliberately naive codec (header +
  `zlib` over raw bytes). Establishes the floor everything else is measured
  against. No architecture work.
- **Round 1 — the structural core.** Per-block framing, frame-of-reference +
  bit-packing for integers, delta transform, and a first cost-model-driven scheme
  selector with a raw fallback that guarantees the no-expansion gate.
- **Round 2 — the float problem.** Floats are where naive delta fails; needs its
  own model (byte-plane transposition, XOR-with-predictor, and/or detecting that
  values are really a quantised fixed-point grid).
- **Round 3 — redundancy beyond the local model.** Runs, repeats, low cardinality
  and sparsity: RLE, dictionary encoding, sparse index+value splitting.
- **Round 4 — entropy coding the residual.** Once the model is good, residuals
  are small and skewed; squeeze them with a real entropy stage instead of a
  general byte compressor.
- **Round 5 — selection, tuning, and hardening.** Better per-block cost model,
  block size tuning, scheme pruning for speed, and whatever the round-4 table
  says is still leaving bytes on the table. Finish with a full fuzz pass and the
  packaging pass that makes the library usable.

Each round: manager writes the intent paragraph in `LOG.md` → `codec-architect`
writes the spec → `codec-engineer` implements → `codec-auditor` attacks → any
failures go back to the engineer → `bench-runner` measures → manager records the
numbers in `LOG.md` and `history.jsonl` and decides whether to keep or revert.
A round whose change fails a gate or loses SCORE is **reverted**, and the log says
so; a dead end is a result, not a failure to hide.

### Done means

1. All six rounds executed and logged.
2. All five hard gates green on both dev and holdout corpora at the final commit.
3. Final SCORE **≥ 2.5× the Round 0 baseline SCORE**, and strictly greater than
   every reference line including `zstd -19`.
4. `tscodec` is importable and usable from outside its own directory, with a
   README documenting `encode`/`decode`, the guarantees, and the measured numbers.
5. `SUMMARY.md` written: what was built, the round-over-round table, and the
   final state.

If target 3 turns out to be unreachable, the run still finishes the six rounds
and `SUMMARY.md` reports the shortfall and the evidence for why — an honest
number beats a moved goalpost.
