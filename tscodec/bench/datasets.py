"""Deterministic benchmark corpus generators for tscodec.

Every generator is seeded with an explicit `np.random.default_rng(seed)` and
built entirely from vectorised NumPy operations, so calling `dev_corpus()` (or
`holdout_corpus()`) twice in separate processes produces byte-identical
arrays. `scale` shrinks column lengths proportionally (used by
`run_bench.py --quick`); it does not change the *shape* of the structure
(jitter scale, quantisation, cardinality, ...), only how many elements are
generated.

Two corpora:
  - `dev_corpus()`   — the 14 archetypes described in PLAN.md, seeds 1000-1013.
  - `holdout_corpus()` — same archetypes with deliberately different shape
    parameters (cardinality, jitter, quantisation, run length, dtype, special
    float values, a regime-switching column), seeds 9000+. This exists to
    catch overfitting to the exact dev parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Target raw size (bytes) per column at scale=1.0; `--quick` passes a smaller
# scale so the corpus generates and benchmarks in a couple of seconds.
_TARGET_BYTES = 1_500_000
_MIN_N = 16


@dataclass(frozen=True)
class Dataset:
    name: str
    array: np.ndarray
    note: str


def _n_for(dtype, scale: float) -> int:
    itemsize = np.dtype(dtype).itemsize
    n = int(round(_TARGET_BYTES * scale / itemsize))
    return max(n, _MIN_N)


# --------------------------------------------------------------------------
# individual column generators — each takes (rng, n, **shape params) and
# returns an array; dtype casting happens once in the corpus assembly loop.
# --------------------------------------------------------------------------


def _make_ts_ms(
    rng: np.random.Generator,
    n: int,
    *,
    start: int = 1_700_000_000_000,
    base_step: int = 1000,
    jitter: int = 3,
    gap_frac: float = 0.002,
    gap_lo: int = 5_000,
    gap_hi: int = 90_000,
) -> np.ndarray:
    """Monotonic ms timestamps: ~base_step cadence + jitter, rare big gaps."""
    n_steps = max(n - 1, 0)
    steps = base_step + rng.integers(-jitter, jitter + 1, size=n_steps, dtype=np.int64)
    if n_steps:
        gap_mask = rng.random(n_steps) < gap_frac
        gap_vals = rng.integers(gap_lo, gap_hi + 1, size=n_steps, dtype=np.int64)
        steps = np.where(gap_mask, gap_vals, steps)
    values = np.empty(n, dtype=np.int64)
    if n:
        values[0] = start
    if n_steps:
        values[1:] = start + np.cumsum(steps)
    return values


def _apply_periodic_resets(cum: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Subtract, from each element, the cumulative value at its most recent
    reset boundary (0 for the first segment). `positions` are the indices
    (sorted, 1 <= p < n) at which a fresh segment starts. Fully vectorised.
    """
    n = cum.shape[0]
    if positions.size == 0:
        return cum
    boundaries = np.concatenate(([0], positions))
    offsets_at_boundary = np.concatenate(([0], cum[positions - 1]))
    seg_idx = np.searchsorted(boundaries, np.arange(n), side="right") - 1
    return cum - offsets_at_boundary[seg_idx]


def _make_counter(
    rng: np.random.Generator,
    n: int,
    *,
    lam: float = 40.0,
    n_resets: int = 8,
    dtype=np.uint64,
) -> np.ndarray:
    """Monotone counter: cumsum of Poisson(lam) increments, reset to 0 at a
    handful of random positions (counter restarts)."""
    increments = rng.poisson(lam=lam, size=n).astype(np.int64)
    cum = np.cumsum(increments)
    k = min(n_resets, max(0, n - 1))
    if k > 0:
        positions = np.sort(rng.choice(np.arange(1, n), size=k, replace=False))
        cum = _apply_periodic_resets(cum, positions)
    return cum.astype(dtype)


def _reflecting_walk(rng, n, lo, hi, step_std):
    """Triangle-wave reflection of a gaussian random walk into [lo, hi]."""
    if n == 0:
        return np.empty(0, dtype=np.float64)
    steps = rng.normal(0.0, step_std, size=n)
    unbounded = np.cumsum(steps)
    span = hi - lo
    period = 2.0 * span
    m = np.mod(unbounded, period)
    reflected = np.where(m > span, period - m, m)
    return lo + reflected


def _make_gauge_quant(
    rng: np.random.Generator,
    n: int,
    *,
    lo: float = 15.0,
    hi: float = 35.0,
    step_std: float = 0.03,
    decimals: int = 2,
) -> np.ndarray:
    values = _reflecting_walk(rng, n, lo, hi, step_std)
    return np.round(values, decimals)


def _make_gauge_f32(
    rng: np.random.Generator,
    n: int,
    *,
    period: float = 500.0,
    amplitude: float = 5.0,
    center: float = 20.0,
    noise_std: float = 0.3,
) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    base = center + amplitude * np.sin(2.0 * np.pi * t / period)
    noise = rng.normal(0.0, noise_std, size=n)
    return (base + noise).astype(np.float32)


def _make_gauge_f32_special(
    rng: np.random.Generator,
    n: int,
    *,
    period: float = 300.0,
    amplitude: float = 5.0,
    center: float = 20.0,
    noise_std: float = 0.5,
) -> np.ndarray:
    """Like gauge_f32 but with NaN, +inf, -inf and -0.0 injected."""
    values = _make_gauge_f32(
        rng, n, period=period, amplitude=amplitude, center=center, noise_std=noise_std
    )
    specials = np.array([np.nan, np.inf, -np.inf, -0.0], dtype=np.float32)
    n_special = min(n, max(4, n // 5000))
    idx = rng.choice(n, size=n_special, replace=False)
    reps = np.tile(specials, int(np.ceil(n_special / specials.size)))[:n_special]
    rng.shuffle(reps)
    values[idx] = reps
    return values


def _make_cat_codes(
    rng: np.random.Generator,
    n: int,
    *,
    n_categories: int = 32,
    id_space: int = 100_000,
) -> np.ndarray:
    n_categories = max(1, min(n_categories, id_space))
    ids = rng.choice(id_space, size=n_categories, replace=False)
    ranks = np.arange(1, n_categories + 1, dtype=np.float64)
    weights = 1.0 / ranks
    weights /= weights.sum()
    choice_idx = rng.choice(n_categories, size=n, p=weights)
    return ids[choice_idx].astype(np.int64)


def _make_flags(
    rng: np.random.Generator,
    n: int,
    *,
    frac_true: float = 0.03,
    mean_burst: float = 5.0,
) -> np.ndarray:
    if n == 0:
        return np.empty(0, dtype=bool)
    n_bursts = max(1, int(round(frac_true * n / mean_burst)))
    starts = rng.integers(0, n, size=n_bursts)
    lengths = rng.geometric(p=1.0 / mean_burst, size=n_bursts)
    ends = np.minimum(starts + lengths, n)
    diff = np.zeros(n + 1, dtype=np.int64)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    coverage = np.cumsum(diff[:n])
    return coverage > 0


def _make_sparse(
    rng: np.random.Generator,
    n: int,
    *,
    zero_frac: float = 0.95,
    mean_log: float = 8.0,
    sigma_log: float = 2.0,
    hi: float = 1e6,
) -> np.ndarray:
    nonzero_mask = rng.random(n) >= zero_frac
    magnitudes = rng.lognormal(mean=mean_log, sigma=sigma_log, size=n)
    magnitudes = np.clip(magnitudes, 1.0, hi)
    values = np.zeros(n, dtype=np.int64)
    scaled = magnitudes.astype(np.int64)
    scaled = np.maximum(scaled, 1)
    values[nonzero_mask] = scaled[nonzero_mask]
    return values


def _make_walk_tick(
    rng: np.random.Generator,
    n: int,
    *,
    start: float = 100.0,
    tick: float = 0.01,
    step_range: int = 3,
) -> np.ndarray:
    if n == 0:
        return np.empty(0, dtype=np.float64)
    steps = rng.integers(-step_range, step_range + 1, size=n)
    steps[0] = 0
    cum_ticks = np.cumsum(steps)
    decimals = max(0, int(round(-np.log10(tick))))
    return np.round(start + cum_ticks * tick, decimals)


def _make_periodic(
    rng: np.random.Generator,
    n: int,
    *,
    period: int = 1024,
    noise_std: float = 0.02,
) -> np.ndarray:
    t = np.arange(n)
    saw = 2.0 * (t % period) / period - 1.0
    noise = rng.normal(0.0, noise_std, size=n)
    return saw + noise


def _make_bursty(
    rng: np.random.Generator,
    n: int,
    *,
    core_median: float = 200.0,
    core_sigma: float = 0.3,
    tail_frac: float = 0.02,
    tail_lo: float = 1e5,
    tail_hi: float = 5e6,
) -> np.ndarray:
    core = rng.lognormal(mean=np.log(core_median), sigma=core_sigma, size=n)
    tail_mask = rng.random(n) < tail_frac
    tail = rng.uniform(tail_lo, tail_hi, size=n)
    values = np.where(tail_mask, tail, core)
    return np.round(values).astype(np.int64)


def _make_ids_runs(
    rng: np.random.Generator,
    n: int,
    *,
    mean_run: float = 40.0,
    n_ids: int = 5000,
    id_space: int = 10_000_000,
) -> np.ndarray:
    if n == 0:
        return np.empty(0, dtype=np.int64)
    ids_pool = rng.integers(0, id_space, size=n_ids, dtype=np.int64)
    p = 1.0 / mean_run
    est_runs = int(n * p * 1.5) + 16
    run_lengths = rng.geometric(p=p, size=est_runs)
    cum = np.cumsum(run_lengths)
    while cum[-1] < n:
        more = rng.geometric(p=p, size=est_runs)
        run_lengths = np.concatenate([run_lengths, more])
        cum = np.cumsum(run_lengths)
    n_runs_needed = int(np.searchsorted(cum, n) + 1)
    run_lengths = run_lengths[:n_runs_needed].copy()
    overshoot = int(cum[n_runs_needed - 1] - n)
    run_lengths[-1] -= overshoot
    run_ids_idx = rng.integers(0, n_ids, size=n_runs_needed)
    values = np.repeat(ids_pool[run_ids_idx], run_lengths)
    return values.astype(np.int64)


def _make_small_ints(
    rng: np.random.Generator,
    n: int,
    *,
    lo: int = -400,
    hi: int = 400,
    step_range: int = 5,
) -> np.ndarray:
    steps = rng.integers(-step_range, step_range + 1, size=n)
    walk = np.cumsum(steps)
    return np.clip(walk, lo, hi).astype(np.int64)


def _make_rand_int(rng: np.random.Generator, n: int, *, dtype=np.int64) -> np.ndarray:
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=n, endpoint=True, dtype=dtype)


def _make_rand_float(rng: np.random.Generator, n: int, *, dtype=np.float64) -> np.ndarray:
    """Uniform random bit patterns reinterpreted as float, with any non-finite
    draw (NaN/inf, from an all-ones exponent) redrawn so the column is
    incompressible but not pathological."""
    uint_dtype = {np.dtype(np.float32): np.uint32, np.dtype(np.float64): np.uint64}[
        np.dtype(dtype)
    ]
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


def _make_regime_switch(
    rng: np.random.Generator,
    n: int,
    *,
    const_val: int = 12345,
    step_range: int = 50,
) -> np.ndarray:
    """First half constant, second half a noisy random walk from that value."""
    half = n // 2
    rest = n - half
    constant_part = np.full(half, const_val, dtype=np.int64)
    if rest:
        steps = rng.integers(-step_range, step_range + 1, size=rest)
        steps[0] = 0
        noisy_part = const_val + np.cumsum(steps)
    else:
        noisy_part = np.empty(0, dtype=np.int64)
    return np.concatenate([constant_part, noisy_part]).astype(np.int64)


# --------------------------------------------------------------------------
# corpus assembly
# --------------------------------------------------------------------------


def _build(specs, seed_base: int, scale: float) -> list[Dataset]:
    out = []
    for i, (name, dtype, fn, note) in enumerate(specs):
        seed = seed_base + i
        rng = np.random.default_rng(seed)
        n = _n_for(dtype, scale)
        arr = np.asarray(fn(rng, n), dtype=dtype)
        out.append(Dataset(name=name, array=arr, note=note))
    return out


def dev_corpus(scale: float = 1.0) -> list[Dataset]:
    specs = [
        ("ts_ms", np.int64, _make_ts_ms,
         "monotonic ms timestamps, ~1s cadence, +/-3ms jitter, 0.2% big gaps"),
        ("counter", np.uint64, _make_counter,
         "monotone counter, cumsum Poisson(lam=40), ~8 resets to 0"),
        ("gauge_quant", np.float64, _make_gauge_quant,
         "reflecting random walk in [15,35], rounded to 2 decimals"),
        ("gauge_f32", np.float32, _make_gauge_f32,
         "sinusoid (period 500) + gaussian noise, full float32 precision"),
        ("cat_codes", np.int32, _make_cat_codes,
         "32 distinct ids in [0,100000), Zipf-ish frequency"),
        ("flags", np.bool_, _make_flags,
         "~3% True, clustered in short bursts (mean length 5)"),
        ("sparse", np.int64, _make_sparse,
         "95% exact zeros, lognormal spikes clipped to [1,1e6]"),
        ("walk_tick", np.float64, _make_walk_tick,
         "price-like random walk on an exact 0.01 tick grid, start 100.0"),
        ("periodic", np.float64, _make_periodic,
         "sawtooth, period 1024, + small gaussian noise, not quantised"),
        ("bursty", np.int64, _make_bursty,
         "latency us: lognormal core (median 200) + 2% heavy tail to 5e6"),
        ("ids_runs", np.int64, _make_ids_runs,
         "run-length structured: 5000 distinct ids, geometric(mean 40) runs"),
        ("small_ints", np.int16, _make_small_ints,
         "clipped random walk in [-400,400], step in [-5,5]"),
        ("rand_int", np.int64, _make_rand_int,
         "uniform random int64 over the full range; incompressible"),
        ("rand_f64", np.float64, _make_rand_float,
         "uniform random bits as float64, finite-only; incompressible"),
    ]
    return _build(specs, seed_base=1000, scale=scale)


def holdout_corpus(scale: float = 1.0) -> list[Dataset]:
    specs = [
        ("ts_ms_wide_jitter", np.int64,
         lambda rng, n: _make_ts_ms(rng, n, jitter=20, gap_frac=0.01, gap_lo=20_000, gap_hi=500_000),
         "timestamps with much wider jitter (+/-20ms) and more frequent big gaps"),
        ("counter_u32", np.uint32,
         lambda rng, n: _make_counter(rng, n, lam=150.0, n_resets=15, dtype=np.uint32),
         "uint32 monotone counter, cumsum Poisson(lam=150), ~15 resets"),
        ("gauge_3dp", np.float64,
         lambda rng, n: _make_gauge_quant(rng, n, lo=0.0, hi=1.0, step_std=0.01, decimals=3),
         "reflecting walk in [0,1] quantised to 3 decimals"),
        ("walk_half_grid", np.float64,
         lambda rng, n: _make_walk_tick(rng, n, start=50.0, tick=0.5, step_range=4),
         "random walk on a coarse 0.5-unit tick grid"),
        ("gauge_f32_special", np.float32,
         _make_gauge_f32_special,
         "float32 sinusoid+noise with injected NaN/+inf/-inf/-0.0"),
        ("cat_codes_7", np.int32,
         lambda rng, n: _make_cat_codes(rng, n, n_categories=7, id_space=100_000),
         "only 7 distinct categorical ids (extreme low cardinality)"),
        ("cat_codes_900", np.int32,
         lambda rng, n: _make_cat_codes(rng, n, n_categories=900, id_space=1_000_000),
         "900 distinct categorical ids (high cardinality)"),
        ("flags_dense", np.bool_,
         lambda rng, n: _make_flags(rng, n, frac_true=0.15, mean_burst=20.0),
         "15% True in long bursts (mean length 20)"),
        ("sparse_80", np.int64,
         lambda rng, n: _make_sparse(rng, n, zero_frac=0.80, mean_log=6.0, sigma_log=1.5, hi=1e4),
         "80% zeros (less sparse), smaller-magnitude spikes"),
        ("periodic_short", np.float64,
         lambda rng, n: _make_periodic(rng, n, period=300, noise_std=0.08),
         "sawtooth period 300 with more noise"),
        ("bursty_hot", np.int64,
         lambda rng, n: _make_bursty(rng, n, core_median=50.0, tail_frac=0.08, tail_hi=2e6),
         "latency with a hotter tail (8% of samples spike)"),
        ("ids_runs_short", np.int64,
         lambda rng, n: _make_ids_runs(rng, n, mean_run=6.0, n_ids=50, id_space=100_000),
         "short runs (mean 6), only 50 distinct ids"),
        ("regime_switch", np.int64,
         _make_regime_switch,
         "first half constant, second half a noisy random walk"),
        ("rand_uint64", np.uint64,
         lambda rng, n: _make_rand_int(rng, n, dtype=np.uint64),
         "uniform random uint64 over the full range; incompressible"),
    ]
    return _build(specs, seed_base=9000, scale=scale)
