# Round 1 spec — the structural core: per-block framing, FOR/delta + patched bit-packing, cost-model selector

Status: specification only. Target files: `tscodec/tscodec/`. Format version goes 0 -> 1.

---

## 1. Objective

Replace the whole-column `zlib` blob with a **blocked container** (8192 elements per block) in which
every integer/bool block independently picks one of four schemes — verbatim RAW, per-block ZLIB,
frame-of-reference bit-packing, and delta + frame-of-reference bit-packing, both with a
**patched (exception) list** for outliers — chosen by a cheap histogram-based cost model, with a
verbatim fallback at both block and column level that makes the no-expansion gate structural.
Float columns keep round-0 behaviour this round; the container and scheme-id space are designed so
round 2 adds float schemes as new scheme ids with **no format break**.

---

## 2. Evidence

Round 0 (naive whole-column `zlib -6`): SCORE 5.6766 dev, 4.7767 holdout. References: zlib-6 5.677,
zstd-3 5.766, zstd-19 6.524, lzma-6 7.222.

I simulated the full design (including the exact cost model, the exact exception encoding and the
exact header/directory overhead) against the real corpus. Per-column dev results, measured not guessed:

| column | dtype | round 0 | round 1 (simulated) | why |
|---|---|---|---|---|
| `ts_ms` | int64 | **2.972** | **20.66** | deltas are 1000±3 with 0.2% gaps to 90 000. FOR base 997 ⇒ **w = 3 bits**; the ~16 gaps per 8192-block become exceptions at ~0.08 bits/elem. 3.1 bits vs 64. |
| `counter` | uint64 | **4.570** | **10.72** | Poisson(40) increments; per-block delta range ≈ 12..80 ⇒ **w = 6** (blocks containing a reset pick base 0 and patch the reset). ~6 bits vs 64. |
| `small_ints` | int16 | **15.68** | **24.82** | walk with steps in [-5,5]; 73/92 blocks are FOR on values (range ≤ 800 ⇒ w = 10), 11 pick delta (w = 4). |
| `bursty` | int64 | **4.573** | **6.55** | lognormal core ⇒ FOR base ≈ 60, **w = 10**; the 2% heavy tail (to 5·10⁶) is 160 exceptions/block at 23 bits. 10.8 bits vs 64. |
| `sparse` | int64 | **34.00** | **38.41** | 95% exact zeros ⇒ FOR base 0, **w = 0** (zero payload); the 5% spikes are exceptions at 13-bit index + 20-bit value ⇒ 1.65 bits/elem. |
| `ids_runs` | int64 | **59.56** | **63.10** | delta with **base = 0, w = 0**: 97.5% of deltas are exactly 0 and cost nothing; the 2.5% run boundaries are exceptions (13-bit index + 25-bit value). |
| `cat_codes` | int32 | 4.946 | 4.95 | FOR gives w = 17 (ids spread over [0,10⁵]) = 1.88x — far worse than zlib's 4.95x. Blocks correctly fall back to ZLIB. **Dictionary coding is round 3's job.** |
| `flags` | bool | 58.68 | 58.67 | bit-packing gives exactly 8x; zlib's run modelling gives 58x. Falls back. **RLE is round 3's job.** |
| floats | — | unchanged | unchanged | round 2. |
| `rand_int`/`rand_f64` | — | 0.9997 | **1.0000** | RAW fallback: fixed 15-byte overhead instead of zlib's 480. |

**Simulated SCORE = 7.43 dev (+30.9%), HOLDOUT = 6.00 (+25.6%).** That clears `zstd -19` (6.524) and
`lzma -6` (7.222) in one round. Holdout standouts: `regime_switch` 8.49 -> 17.43 (block-local choice
is exactly what that column was built to test), `counter_u32` 1.65 -> 4.46, `ts_ms_wide_jitter`
2.96 -> 10.08.

Three ablations, run on both corpora, that shape the spec:

| variant | dev | holdout | conclusion |
|---|---|---|---|
| full design | 7.4256 | 6.0004 | — |
| **without delta-of-delta** | **7.4256** | **6.0004** | **DELTA2 is selected on zero blocks of 28 columns. Do not implement it.** |
| without the DELTA scheme | 5.8726 | 4.9296 | delta is where the round's value is |
| without the FOR-on-values scheme | 7.1767 | 5.8901 | worth 3.3%; `sparse`/`bursty` need it |
| without per-block ZLIB | 7.3594 | 6.0004 | worth 0.9% on dev; keep |
| **zlib tried on every block** vs **triggered** (§6.4) | 7.4256 | 6.0004 | **identical — the trigger is free encode speed** |

**Do not implement delta-of-delta this round.** The plan listed it, but the evidence says it can never
win here: frame-of-reference already subtracts the *mean slope* of the block, so second differencing
only helps when the slope itself drifts within a block. Every monotone column in this corpus has a
constant-rate slope, so DELTA2 is strictly worse than DELTA+FOR (for `ts_ms` it is w = 4 vs w = 3).
Reserve scheme id 4 for it and spend the round's budget on the exception mechanism instead, which is
what actually unlocks `sparse`, `ids_runs`, `bursty` and `ts_ms`.

Block size is not ratio-critical: dev SCORE is 7.3845 / 7.4279 / 7.4256 / 7.4244 / 7.3258 at
B = 2048 / 4096 / 8192 / 16384 / 32768. **Pick B = 8192**, which is the smallest size at which per-block
Python loops are permitted, and store `log2(B)` in the header so round 5 can tune it without a format break.

---

## 3. Bitstream layout, byte-exact

Everything is **little-endian**. `I` = `dtype.itemsize` (1, 2, 4 or 8). `L` = `8*I` = the element bit
width. `UINT_L` = the unsigned NumPy dtype of width L (`uint8`/`uint16`/`uint32`/`uint64`).

### 3.1 Container header — 15 bytes, always present

```
off  size  field
0    4     magic          b'TSC0'                    (unchanged from round 0)
4    1     version        uint8 = 1                  (was 0; decoder MUST reject 0)
5    1     dtype_code     uint8, DTYPE_CODES table from round 0, unchanged
6    8     n              uint64, element count
14   1     mode           uint8 container mode
```

`mode` values:

| id | name | body |
|---|---|---|
| 0 | `SINGLE_RAW` | bytes 15.. = the column's raw little-endian bytes. Total length is exactly `15 + n*I`. |
| 1 | `SINGLE_ZLIB` | bytes 15.. = `zlib.compress(raw, 6)` |
| 2 | `BLOCKED` | §3.2 |
| 3-255 | reserved | decoder raises `ValueError` |

### 3.2 `BLOCKED` body

```
off  size            field
15   1               log2_block     uint8; B = 1 << log2_block. Round 1 emits 13 (B = 8192).
                                    Decoder accepts 6..30, rejects anything else.
16   4               n_blocks       uint32, MUST equal ceil(n / B)
20   R * n_blocks    directory      R = 12 + 3*I bytes per record, §3.3
20+R*n_blocks ...    payload region, blocks concatenated in block order
```

Block `b` covers elements `[b*B, min(n, (b+1)*B))`; write `m` for that block's element count
(`m == B` for every block but possibly the last). The payload of block `b` begins at
`payload_base + sum(plen[0:b])` and is `plen[b]` bytes long.

### 3.3 Directory record — fixed size `R = 12 + 3*I`, packed (no alignment padding)

```
off      size  field       notes
+0       1     scheme      uint8, §3.4
+1       1     w           uint8, main-stream bit width, 0..L. 0 for RAW/ZLIB_RAW.
+2       1     we          uint8, exception-value bit width, 0..L. 0 if n_exc == 0.
+3       1     reserved    uint8, MUST be 0 (round 2+ may claim it; decoder MUST NOT reject non-zero)
+4       4     n_exc       uint32, number of patched elements. 0 for RAW/ZLIB_RAW.
+8       4     plen        uint32, payload byte length of this block
+12      I     base        UINT_L, frame-of-reference base for the main stream
+12+I    I     base_e      UINT_L, frame-of-reference base for the exception values
+12+2I   I     first       UINT_L, the block's first element value (BITPACK_DELTA only)
```

Unused fields are written as 0. The decoder parses the whole directory with **one**
`np.frombuffer(buf, dtype=REC_DTYPE, count=n_blocks, offset=20)` where

```python
REC_DTYPE = np.dtype([('scheme','u1'), ('w','u1'), ('we','u1'), ('reserved','u1'),
                      ('n_exc','<u4'), ('plen','<u4'),
                      ('base', ule), ('base_e', ule), ('first', ule)])   # ule in '<u1','<u2','<u4','<u8'
```

Verify `REC_DTYPE.itemsize == 12 + 3*I` in a unit test — NumPy must not insert alignment padding
(it does not for a plain list-of-tuples spec, but assert it).

Overhead: 36 bytes/block for int64 (828 bytes on a 1.5 MB column, 0.055%), 15 bytes/block for bool
(0.18%). Acceptable; fixed size is what buys the single-`frombuffer` directory parse.

### 3.4 Scheme id table (permanent; ids are never renumbered or reused)

| id | name | round | notes |
|---|---|---|---|
| 0 | `RAW` | 1 | verbatim block bytes |
| 1 | `ZLIB_RAW` | 1 | `zlib.compress(block_bytes, 6)` |
| 2 | `BITPACK` | 1 | FOR on values + patched bit-packing |
| 3 | `BITPACK_DELTA` | 1 | FOR on first differences + patched bit-packing |
| 4 | *reserved* `BITPACK_DELTA2` | — | second differences; evidence says it never wins, do not emit |
| 5 | *reserved* `RLE` | 3 | |
| 6 | *reserved* `DICT` | 3 | |
| 7 | *reserved* `SPARSE_IDX` | 3 | |
| 8-15 | *reserved* | — | further integer schemes |
| 16 | *reserved* `FLOAT_SPLIT` | 2 | byte-plane / stream-split |
| 17 | *reserved* `FLOAT_XOR` | 2 | XOR-with-predictor |
| 18 | *reserved* `FLOAT_FIXED` | 2 | quantised-grid detection -> integer scheme |
| 19-31 | *reserved* | 2 | further float schemes |
| 32-63 | *reserved* | 4 | entropy-coded variants of 0-31 (id = base_id + 32) |
| 64-255 | *reserved* | — | |

A decoder that meets an unknown scheme id raises `ValueError`. Round 2 adds float schemes purely by
claiming ids 16-18 and changing the encoder's dtype policy in §6.6 — **no header, directory or
payload-framing change**.

### 3.5 Payload layout per scheme

Let `wi = max(1, bitlen(m - 1))` (the bit width of a within-block index; 13 for `m = 8192`).
`wi` is **derived from `m`, never stored**.

| scheme | payload bytes (in this order) | `plen` |
|---|---|---|
| `RAW` | the block's `m*I` raw little-endian bytes | `m*I` |
| `ZLIB_RAW` | `zlib.compress(block_bytes, 6)` | its length |
| `BITPACK` / `BITPACK_DELTA` | `main` ‖ `exc_idx` ‖ `exc_val` | sum of the three |

with

* `main` = `PACK(r, w)`, `len = ceil(m*w/8)`, where `r` is the block's `m` residuals;
* `exc_idx` = `PACK(idx, wi)`, `len = ceil(n_exc*wi/8)`, `idx` = the ascending within-block positions
  of the patched elements;
* `exc_val` = `PACK(ev, we)`, `len = ceil(n_exc*we/8)`;
* both exception streams are absent (zero bytes) when `n_exc == 0`.

### 3.6 `PACK` — the bit-packing primitive (normative definition)

> `PACK(v, w)` for an array `v` of `c` unsigned values, every one `< 2^w`, is the byte string of
> length `ceil(c*w/8)` in which **element `i` occupies bit positions `i*w .. i*w + w - 1`**, where bit
> position `p` is bit `p mod 8` (LSB-first) of byte `p // 8`. Bits beyond `c*w` in the final byte are 0.
> `PACK(v, 0)` is the empty byte string.

This definition is independent of how you compute it. **Implement it with the lane trick:** choose the
lane width `Lp = 8 if w<=8 else 16 if w<=16 else 32 if w<=32 else 64`, cast `v` to the unsigned dtype of
that width, zero-pad to a multiple of `Lp` elements, reshape to `(g, Lp)`, and fill a `(g, w)` output:

```
for j in range(Lp):                     # fixed Lp iterations, independent of c
    pos = j*w; word = pos // Lp; off = pos % Lp
    out[:, word] |= V[:, j] << off
    if Lp - off < w:
        out[:, word+1] |= V[:, j] >> (Lp - off)
return out.reshape(-1).view(uint8)[: (c*w + 7)//8]
```

`UNPACK` is the exact mirror (zero-pad the byte string back up to `g*w*(Lp//8)` bytes first, mask each
extracted value with `(1 << w) - 1`, return the first `c`).

I verified that **the byte output is identical for every lane width** `Lp ∈ {8,16,32,64}` with `w ≤ Lp`,
for `w = 1..32` — a unit test must assert this. The lane choice is a pure speed optimisation and it is a
large one: packing 187 500 int64 residuals at `w = 3` runs at 3304 MB/s with `Lp = 8` vs 1309 MB/s with
`Lp = 64`; unpacking 1.5 M bool residuals at `w = 1` runs at 1685 MB/s vs 143 MB/s. **`Lp = 64`
unconditionally would put the decode gate at risk on narrow dtypes.**

### 3.7 Block decode semantics

For `BITPACK` (2) and `BITPACK_DELTA` (3), with `M = 2^L` (all arithmetic mod `M`, i.e. native wrapping
in `UINT_L`):

```
r  = UNPACK(main, w, m)                       # UINT_L
t  = (r + base) mod M
if n_exc:
    idx = UNPACK(exc_idx, wi, n_exc)
    ev  = UNPACK(exc_val, we, n_exc)
    t[idx] = (ev + base_e) mod M

scheme 2 (BITPACK):        values = t
scheme 3 (BITPACK_DELTA):  t[0] = 0
                           values = (first + cumsum(t)) mod M
```

The encoder guarantees `r[0] == 0` for scheme 3 (§6.2), so index 0 is never patched and the decoder's
`t[0] = 0` is unambiguous.

---

## 4. What the encoder computes, vectorised

All work is done in `UINT_L` — **never widen to uint64**. Widening is what made the naive analysis pass
run at 19.5 MB/s on the bool column; staying in the native width keeps it above 100 MB/s.

Let `U` be the column viewed as `UINT_L` and reshaped to `(k, B)` for the `k = n // B` full blocks
(the tail block is handled as its own `(1, m)` group).

1. `base_for` — per block: for signed dtypes, `U[i, U[i].view(int_L).argmin()]`; for unsigned/bool,
   `U.min(axis=1)`.
2. `D` — `D[:, 1:] = U[:, 1:] - U[:, :-1]` (wrapping), `D[:, 0] = 0`.
3. `base_delta` — per block, the element of `D[:, 1:]` with the **minimum signed interpretation**
   (`D[:, 1:].view(int_L).argmin(axis=1)`), *regardless of the column's signedness* — deltas are
   conceptually signed and this is what makes `counter_u32` and `small_ints` work.
4. Three candidates, each a `(transform, base)` pair:

   | candidate | scheme id emitted | transform `T` | base |
   |---|---|---|---|
   | `FOR` | 2 | `U` | `base_for` |
   | `DELTA_MIN` | 3 | `D` | `base_delta` |
   | `DELTA_ZERO` | 3 | `D` | `0` |

   `DELTA_ZERO` is not an extra scheme id — it is the same scheme with `base = 0`. It is the candidate
   that wins `ids_runs` (63.10x) and the reset blocks of `counter`; without it those blocks pay
   `0 - min_delta` for every zero delta.
5. For each candidate: `R = T - base[:, None]` (wrapping); then `R[:, 0] = 0` for the two delta
   candidates.

---

## 5. The cost model

Per candidate, per block, in **bits**:

```
cost(w) = m*w + est_exc(w) * (wi + we_est)          for w in 0..L
```

`est_exc(w)` = estimated count of residuals needing more than `w` bits; `we_est` = the largest residual
bit length seen. Both come from one histogram:

* **Subsample with stride `S = 8`**: `Rs = R[:, ::8]` (at least 1 element per block).
* `bl = bitlen(Rs)` as `uint8`, computed by binary search on shifts (`log2(L)` vectorised steps,
  staying in `UINT_L`) — not `np.searchsorted`, whose `intp` output costs 8 bytes per element.
* Per-block histogram in one call:
  `H = np.bincount((bl + np.arange(k)[:,None]*(L+1)).ravel(), minlength=k*(L+1)).reshape(k, L+1)`.
* `est_exc(w) = S * sum(H[:, w+1:])` (a reversed cumsum), `we_est = ` highest non-empty bin.

Choose `w* = argmin cost(w)`, **ties broken toward the larger `w`** (fewer exceptions, faster decode,
identical size).

Then, **for the winning candidate only** (this matters: computing exact costs for all three candidates
costs 2.2x the encode time and produced bit-identical output on all 28 columns):

* `exc_mask = r >= 2^w*` (empty if `w* == L`), `n_exc = popcount`, `idx = flatnonzero(exc_mask)`
* `base_e` = the element of `T[idx]` with minimum signed interpretation (unsigned min for unsigned
  `BITPACK`); `ev = (T[idx] - base_e) mod M`; `we = bitlen(ev.max())`
* exact payload bytes `p = ceil(m*w*/8) + ceil(n_exc*wi/8) + ceil(n_exc*we/8)`

> **Do not** put the exception values in residual space (`r[idx]`) — put them in transform space
> (`T[idx]`) with their own base. That one change took `ids_runs` from 59.55x to 63.10x, because with
> `base = 0` the negative run-boundary deltas wrap to ≈2⁶⁴ and a residual-space base makes `we = 64`;
> in transform space their signed range is ±10⁷ and `we = 25`.

---

## 6. Selection procedure

### 6.1 Per block
1. If `m < 64`: the block may only use `RAW` or `ZLIB_RAW`. (Kills every degenerate packing edge case
   at a bounded cost of ≤ 504 bytes, only ever on the tail block.)
2. Otherwise compute the estimated cost of all three candidates (§5), take the argmin, compute its
   exact cost `p`.
3. `RAW` wins if `m*I <= p` (**tie goes to RAW**).
4. ZLIB trigger (§6.4). If fired and `len(zlib_bytes) < current best`, the block becomes `ZLIB_RAW`.

### 6.2 Delta bookkeeping
For a `BITPACK_DELTA` block, `first = U[b*B]`, statistics and `base_delta` come from `D[:, 1:]` only,
and `r[0]` is forced to 0 after the residual subtraction and **before** exception detection.

### 6.3 Width/lane invariants
`w ∈ [0, L]`. When `w == L` there are no exceptions by construction (every residual fits), and the
payload is exactly `m*I` bytes, so step 6.1.3 will have chosen RAW — a `BITPACK` block with `w == L`
should never be emitted. Guard `1 << w` with `w == L` handled separately (`1 << 64` overflows a
`uint64` scalar constructor).

### 6.4 The ZLIB trigger (encode-speed control)
Run `zlib.compress` on a block iff

```
(p * 8.0 / m) >= 8.0                                  # bit-packing is doing badly, OR
or  count_nonzero(D[b, 1:] == 0) / (m - 1) >= 0.5      # half the block is literal repeats -> LZ will win
```

The second term is free: `D` is already computed, so it is one vectorised `count_nonzero` over the whole
`(k, B)` array. The first term catches `cat_codes` (17 bits), `bursty` (10.8 bits) and `rand_int`
(64 bits); the second catches `flags` (94% repeats), `ids_runs` and `sparse`. Measured: this trigger
produces **bit-identical output** to trying zlib on every block, on all 28 columns, while skipping zlib
entirely on `ts_ms`, `counter` and `ts_ms_wide_jitter` (encode 288/320/273 MB/s instead of ~35).

### 6.5 Column-level fallbacks, applied in this order after the blocked body is built
1. If `count(scheme == ZLIB_RAW) >= 0.5 * n_blocks` **and** `n_blocks >= 2`: compute
   `zlib.compress(whole_column, 6)`; if `15 + len(...) < len(blocked_output)`, emit `SINGLE_ZLIB`.
   This is worth 2.6% of SCORE (`flags` 43.9x -> 58.7x, `cat_codes` 4.63x -> 4.95x, `ids_runs_short`
   18.3x -> 20.0x) and only costs encode time on columns that were already zlib-bound.
2. If the result is still longer than `15 + n*I`, emit `SINGLE_RAW`.

**Step 2 is the no-expansion guarantee and it is unconditional: `len(encode(a)) <= 15 + a.nbytes` for
every input, always.** On the 1.5 MB corpus columns that is ratio ≥ 0.99999.

### 6.6 dtype policy for round 1 (the one line round 2 changes)
```
if dtype.kind == 'f':   emit min(SINGLE_ZLIB, SINGLE_RAW)      # no float-aware scheme exists yet
elif n < 2:             emit min(SINGLE_ZLIB, SINGLE_RAW)
else:                   BLOCKED, then §6.5
```
Floats therefore keep their round-0 ratios exactly (plus a hair from the smaller header). Do **not**
spend this round on float schemes — but do not special-case floats anywhere except this dispatch, so
round 2 only has to delete the first branch.

---

## 7. Integration

| file | change |
|---|---|
| `tscodec/tscodec/bitpack.py` | **new.** `bitlen(arr) -> uint8 array`, `lane_for(w) -> int`, `pack(values, w) -> bytes`, `unpack(buf, w, count, out_dtype) -> ndarray`. Pure, no knowledge of the container. |
| `tscodec/tscodec/blocks.py` | **new.** `SCHEME_*` constants, `rec_dtype(itemsize)`, `encode_blocked(u, itemsize, log2_block) -> bytes`, `decode_blocked(buf, offset, n, udtype) -> ndarray`, plus the selector (§5, §6.1-6.4). |
| `tscodec/tscodec/codec.py` | header/mode dispatch only: `encode`, `decode`, `DTYPE_CODES` (unchanged), `FORMAT_VERSION = 1`, §6.5/§6.6 policy. Keep the public signatures. |
| `tscodec/tscodec/__init__.py` | also export `describe` (below). |
| `tscodec/bench/run_bench.py` | optional: call `tscodec.describe(encoded)` and print the per-column scheme histogram. PLAN §3 lists scheme-selection statistics as an architect input and I do not have them today. |

Add `tscodec.describe(buf) -> dict` — parses header + directory and returns
`{'mode', 'n', 'dtype', 'n_blocks', 'block_size', 'schemes': {name: count}, 'widths': {w: count},
'n_exceptions': int, 'bytes': {section: nbytes}}`. It touches no payload, so it costs nothing on the hot
path, and it is what lets me diagnose round 2 from evidence instead of guesses.

Version 0 streams are **not** decodable by the round-1 decoder (nothing has persisted any); raise
`ValueError("unsupported format version 0")`.

---

## 8. Correctness hazards, and exactly how each is handled

1. **Delta reversibility / overflow.** All transforms are wrapping arithmetic in `UINT_L`, which is exact
   modulo `2^L` and therefore bijective. `D = U[1:] - U[:-1]` over `int64`-min-to-`int64`-max wraps and the
   inverse `cumsum` wraps back. Never compute a delta in a *wider* type "to avoid overflow" — wrapping is
   the correctness mechanism, not a bug to dodge.
2. **`np.cumsum` silently widens.** `np.cumsum(x)` on `uint8`/`uint16`/`uint32` returns `uint64`
   (verified on the installed NumPy 2.4.6: `np.cumsum(np.array([200,100,50],'u1')) -> [200,300,350]`,
   dtype `uint64`). The wrap is then lost. **Every `cumsum` must pass `dtype=UINT_L` explicitly**
   (`np.cumsum(t2d, axis=1, dtype=np.uint8)` gives `[200,44,94]`, which is what we need). Same discipline
   for `np.sum`/`np.add.reduce` anywhere in the encoder.
3. **Signed min via view.** `U.view(int_L).argmin(axis=1)` then index `U` with it — never convert the
   base to a Python int and back. `int64` min views as `uint64` `2^63` and `argmin` on the `int64` view
   finds it correctly.
4. **Shift counts.** Never shift by `>= L`. In `PACK`/`UNPACK`, `off < Lp` always, and the second-word
   write is guarded by `Lp - off < w`; when `off == 0 and w == Lp` that guard is false, so no
   out-of-range shift occurs. `1 << w` for the exception threshold must be guarded when `w == L`
   (`w == L` ⇒ no exceptions; skip the comparison entirely).
5. **Scalar promotion.** Build every scalar operand with `dt.type(x)` before combining it with an array.
   NumPy 2.x NEP-50 keeps `np.uint64(5) - 1` in `uint64`, but the code must not depend on the NumPy
   version for this.
6. **NaN payloads, `-0.0`, infinities.** Floats never enter the integer path this round; `SINGLE_RAW`
   and `SINGLE_ZLIB` are byte transports, so every bit pattern survives. When round 2 adds float schemes
   they must operate on the *bit patterns* (`view(uint64)`), never on float values — no comparisons, no
   arithmetic, no `np.isnan`-driven branching that could canonicalise a NaN payload or collapse `-0.0`
   into `0.0`.
7. **`bool`.** Treated as `uint8` (`a.view(np.uint8)`), `L = 8`. Never assume the byte is 0 or 1 — a
   `bool` array can be produced by `.view()` and hold arbitrary bytes. FOR with `base = min` and
   `w = bitlen(max - min)` round-trips any byte. Decode reconstructs `uint8` and `.view(np.bool_)`.
8. **Empty input (`n == 0`).** `SINGLE_RAW`, empty payload, total 15 bytes. `decode` returns
   `np.empty(0, dtype)`. `n_blocks` would be 0; `BLOCKED` is never chosen (§6.6 `n < 2`).
9. **Single element (`n == 1`), and any `n < 2`.** `SINGLE_*` path (§6.6). Also guard `m < 2` inside the
   block encoder for safety (no delta candidate when `m < 2`); §6.1.1's `m < 64` rule already covers it.
10. **Tail block.** `m < B`. `wi = max(1, bitlen(m-1))` uses the tail's own `m`, so the exception-index
    width differs from the full blocks' — the decoder must recompute `wi` per block from `m`, not
    assume `wi = bitlen(B-1)`.
11. **All-exceptions / pathological widths.** `w == L` can never produce an exception; `RAW` wins on
    ties (§6.1.3) so a `w == L` bit-packed block is never emitted. Assert this in a unit test.
12. **Byte order.** The format is little-endian by definition. `encode` normalises the input with
    `np.ascontiguousarray(a, dtype=native)` as today, then works in explicit little-endian dtypes
    (`'<u8'` etc.) so the packed bytes are identical on a big-endian host; `decode` returns a
    native-byte-order array. For `I == 1` there is nothing to swap.
13. **`np.frombuffer` writability.** Directory and payload views taken from `bytes` are read-only; copy
    before any in-place `|=` or scatter. The returned array must be writable (round 0's contract).
14. **Truncated/corrupt input.** Validate: `len(buf) >= 15`; `n_blocks == ceil(n/B)`; `6 <= log2_block <= 30`;
    `20 + R*n_blocks + sum(plen) == len(buf)`; each `w <= L`, `we <= L`, `n_exc <= m`; unknown scheme or
    mode. Raise `ValueError`, never segfault or return a wrong-length array.
15. **Exception indices must be strictly ascending and `< m`.** They are by construction
    (`flatnonzero`); the decoder should not rely on it for memory safety — `t[idx] = ...` with an
    out-of-range index raises, which is the desired behaviour on corrupt input.

---

## 9. Expected effect

### 9.1 Ratio (simulated end-to-end on the real corpus, including all overheads)

Dev: **SCORE 5.6766 -> 7.43** (+30.9%). Holdout: **4.7767 -> 6.00** (+25.6%).

| dev column | r0 | r1 pred | | holdout column | r0 | r1 pred |
|---|---|---|---|---|---|---|
| ts_ms | 2.972 | **20.66** | | ts_ms_wide_jitter | 2.960 | **10.08** |
| counter | 4.570 | **10.72** | | counter_u32 | 1.651 | **4.46** |
| gauge_quant | 6.606 | 6.606 | | gauge_3dp | 4.791 | 4.791 |
| gauge_f32 | 1.189 | 1.189 | | walk_half_grid | 10.649 | 10.649 |
| cat_codes | 4.946 | 4.95 | | gauge_f32_special | 1.185 | 1.185 |
| flags | 58.68 | 58.67 | | cat_codes_7 | 7.814 | 7.81 |
| sparse | 34.00 | **38.41** | | cat_codes_900 | 2.615 | 2.61 |
| walk_tick | 7.666 | 7.666 | | flags_dense | 50.68 | 50.67 |
| periodic | 1.048 | 1.048 | | sparse_80 | 11.751 | 11.75 |
| bursty | 4.573 | **6.55** | | periodic_short | 1.046 | 1.046 |
| ids_runs | 59.56 | **63.10** | | bursty_hot | 5.045 | **6.54** |
| small_ints | 15.68 | **24.82** | | ids_runs_short | 20.01 | 20.00 |
| rand_int | 0.9997 | **1.0000** | | regime_switch | 8.486 | **17.43** |
| rand_f64 | 0.9997 | **1.0000** | | rand_uint64 | 0.9997 | **1.0000** |

Nothing regresses. The largest holdout mover (`regime_switch`, 2.05x) is the per-block-choice column,
which is the cleanest possible confirmation that the framing — not just the transforms — is earning.

### 9.2 Throughput

Measured on a prototype of the selector (analysis + exact cost + triggered zlib) and of the
pack/unpack primitives, on this machine:

* **Encode.** `ts_ms` 288 MB/s, `counter` 320, `regime_switch` 199, `small_ints` 117, `ids_runs` 85,
  `sparse` 67, `flags` 47, `rand_int` 37, `cat_codes` 16, `bursty` 16. Adding packing (0.5-3 ms/column)
  and the §6.5 whole-column zlib where it fires, I predict **geomean encode ≈ 38-42 MB/s** (round 0: 35.5).
  The slowest column is `cat_codes` at ~8 MB/s because it pays per-block zlib *and* the §6.5
  whole-column zlib; that is the single place where the design pays double.
* **Decode.** Unpack runs at 400-5200 MB/s with the lane optimisation; per-block exception unpacking is
  the worst case at ~4 ms on `sparse` (23 blocks x 2 streams). Predicted **geomean decode 300-600 MB/s**
  (round 0: 520.7). Gate is 100.

Both gates should pass with large margin. The specific risk is encode, and §6.4 and §6.5 are where the
knobs are: if the measured encode geomean lands below 28 MB/s, drop §6.5 step 1 first (costs 2.6% of
SCORE, recovers roughly 15% of encode time).

---

## 10. Kill criteria

Revert (or cut back to the named subset) if, on the full benchmark:

1. **SCORE < 6.8 dev.** The simulation says 7.43 with every overhead counted. Landing below 6.8 means the
   implementation diverges from the spec — find the divergence, do not tune around it. Below 5.68 is an
   outright revert.
2. **HOLDOUT < 5.4**, or holdout down while dev is up. The design has no corpus-specific constants
   except `S = 8`, the `8.0 bits/elem` and `0.5 repeat-fraction` triggers, and `B = 8192`; a holdout gap
   means one of those four is overfit. Re-derive it, do not add a fifth.
3. **Any gate fails.** In particular any `ratio < 0.99`: §6.5 step 2 makes that structurally impossible,
   so a failure there is a bug in the final length comparison, not a tuning problem.
4. **Encode geomean < 25 MB/s.** Escalation order: (a) drop §6.5 step 1; (b) raise the §6.4 repeat-fraction
   trigger from 0.5 to 0.8; (c) raise `S` from 8 to 16. Do **not** drop the DELTA_ZERO candidate — it is
   worth more than all three of those combined.
5. **Decode geomean < 100 MB/s.** Almost certainly means `PACK`/`UNPACK` was implemented with a fixed
   `Lp = 64`; check that first (it is a 7-12x factor on `bool`/`int16`).
6. **Fuzz suite finds a non-round-tripping case** that needs a *format* change rather than a code fix.

Explicitly **not** a reason to revert: `cat_codes`, `flags`, `gauge_f32` and `periodic` staying where
they are. Those four are the round-2 and round-3 targets by design, and they are the reason there is
still >2x of headroom left after this round.
