# LOG.md — `tscodec` run log

Metric: **SCORE** = geometric mean of `raw_bytes / encoded_bytes` over the 14-column
dev corpus. **HOLDOUT_SCORE** is the same over a separate corpus generated with
different seeds and shape parameters. Gates: bit-exact lossless, fuzz-lossless,
no dataset expanded (ratio >= 0.99), decode >= 100 MB/s, encode >= 25 MB/s.
See `PLAN.md` for the full definition.

---

## Round 0 — harness and honest baseline

**Intent (before the round).** Nothing can be improved before it can be measured,
so round 0 buys no compression at all — it buys the measuring apparatus and an
honest floor to measure against. I am having the engineer build three things: a
deterministic corpus generator that produces the 14 dev columns and a separately
parameterised holdout corpus (so later rounds can be checked for overfitting), a
benchmark runner that computes per-dataset ratios, throughputs, the geometric-mean
SCORE, all five gates and the four reference lines (`zlib -6`, `lzma -6`,
`zstd -3`, `zstd -19`), and a deliberately naive codec: a small header plus
`zlib.compress(a.tobytes())`. The naive codec is the point of the round. It is
what a reasonable engineer writes in ten minutes, it passes losslessness
trivially, and it is the number every later round has to justify itself against —
if round 5 cannot clear it by a wide margin then the whole modelling exercise was
theatre. I expect a SCORE somewhere around 2-4x, dragged down by the two
deliberately incompressible columns, and I expect the naive codec to *fail* the
no-expansion gate or come close to it on random data, which is itself a useful
demonstration of why the gate exists. The auditor builds the fuzz suite in
parallel with this so that round 1's first real scheme lands on top of an already
adversarial test bed.

*Delegation note.* The four agent definitions in `.claude/agents/` were written
after this session's agent registry had already loaded, so the runtime does not
expose them as named `subagent_type`s in this run. Rather than abandon the
structure, every delegation below is dispatched as a generic worker with the
role's model pinned explicitly (`opus` for the architect, `sonnet` for the
engineer and auditor, `haiku` for the bench-runner) and with the worker instructed
to read its own definition file from `.claude/agents/` as its first action. The
role files stay the single source of truth for each seat's responsibility and
boundaries; only the wiring is different. Every round below names which seat did
which piece of work.

**Result.** Harness, 14-column dev corpus, 14-column holdout corpus, 59 unit tests
and the naive codec all landed. Determinism of the corpus was verified by hashing
it across two separate processes. Measured baseline:

```
SCORE (dev geomean ratio)     = 5.6766
HOLDOUT_SCORE                 = 4.7767
GATE lossless      PASS   (all round trips bit-exact)
GATE no_expansion  PASS   (worst = rand_int, ratio 0.9997)
GATE decode_speed  PASS   (geomean dev decode 520.7 MB/s)
GATE encode_speed  PASS   (geomean dev encode  35.5 MB/s)

reference lines (dev geomean):  zlib-6 5.677 | lzma-6 7.222 | zstd-3 5.766 | zstd-19 6.524
```

Three things this changes. First, **the bar moved up**: I guessed 2-4x in the
intent paragraph and got 5.68, so "2.5x the baseline" now means reaching **14.2x**,
and the codec has to beat `lzma-6` at 7.22 rather than the 5.7 I was mentally
budgeting for. zlib is a much better column compressor than it has any right to
be, mostly because 8-byte-strided repetition inside a numeric column is exactly
what LZ77 match-finding is good at. Second, the **per-column table immediately
tells us where the headroom is**: `periodic` (1.048), `gauge_f32` (1.189) and
`counter` (4.57) are being handled terribly relative to their actual structure —
a sawtooth with period 1024 is nearly free to encode if you model it, and a
monotone counter should be a small delta stream, not a 328 KB LZ blob. The
float columns are the worst offenders and the biggest prize. Third, the engineer
reported an honest negative finding: **the no-expansion gate does not currently
bite**. At 1.5 MB per column, zlib's ~480-byte overhead on incompressible data
leaves the ratio at 0.9997, just inside the gate. The gate is true but untested.
I am not going to weaken the corpus to make it bite; instead the auditor's job
brief now explicitly includes small-array expansion tests, where header overhead
is proportionally brutal, so the guarantee is actually exercised where it can fail.

**Round 0 verdict: KEEP.** Baseline established at SCORE 5.6766.

---

## Round 1 — the structural core

**Intent (before the round).** The naive codec has no idea it is looking at a
numeric column; everything from here is about giving it that knowledge. Round 1
builds the skeleton that all later schemes hang off, and it targets the integer
columns because they are the tractable half of the problem. Three pieces. (a)
**Per-block framing**: split the column into fixed-size blocks so that scheme
choice is local — `regime_switch` in the holdout exists precisely because a column
can change character halfway through, and a whole-column decision cannot win
there. (b) **The integer model**: delta and frame-of-reference transforms followed
by bit-packing, which should turn `counter` and `ts_ms` from LZ-compressible
byte soup into a few bits per element. A monotone counter with Poisson increments
has maybe 7 bits of real entropy per value; we are currently spending 14. (c) The
piece I care most about architecturally — a **cost-model-driven selector with a
raw fallback**. The encoder must be able to try candidate schemes, estimate output
size cheaply, pick the winner, and fall back to storing the block verbatim. That
fallback is what makes the no-expansion gate structurally guaranteed rather than
accidental, and it is what will let every later round add a scheme without
risking a regression on the incompressible columns. I am sending the evidence
table to the architect rather than dictating the scheme list myself, because the
per-column gaps are the whole argument for what to build and I want the
diagnosis on the record before the implementation. Risk I am watching: bit-packing
in pure NumPy is where the encode-speed gate could break, and a per-block Python
loop over several candidate schemes multiplies that cost.

---

### Schedule revision (mid-round-1): six rounds become four

Recording a change to `PLAN.md` §4 and the reason for it, so the run stays legible.

The plan committed to six rounds without a budget model attached to it. After
setup plus round 0 the session was ~28% through its usage window with ~4.5 hours
to reset. Round 0 cost a single engineer call (~132k subagent tokens) plus manager
turns; rounds 1-5 as planned each carry four calls (architect on opus, engineer,
auditor, bench-runner), so every remaining round is *more* expensive than the one
already spent. Five of them do not fit in the remaining budget. Discovering that
at round 3 would mean abandoning the run half-finished, with the entropy-coding
and hardening work — the rounds that convert a promising codec into a usable one —
never reached.

So the remaining schedule is compressed, keeping the same technical ground:

- **Round 1** (as dispatched): per-block framing, integer transforms, bit-packing,
  cost-model scheme selection with raw fallback.
- **Round 2** (was rounds 2+3): the float model *and* redundancy beyond the local
  model — byte-plane transposition / XOR / quantisation-grid detection, plus RLE,
  dictionary and sparse encoding.
- **Round 3** (was rounds 4+5): entropy coding the residual, selector and
  block-size tuning, scheme pruning for speed, final fuzz pass and packaging.

The auditor authors tests in two rounds rather than five (after bit-packing lands,
and at the end); its suite still runs every round through the test gate, it is the
*authoring* calls that are cut. The architect keeps one opus call per round —
that is the highest value-per-token seat in the loop and the last thing worth
economising on.

What this costs: less iteration *within* each theme. The original schedule allowed
a round to land a float scheme, measure it, and refine it the following round.
Now each theme gets one shot plus whatever the final tuning round can fix. If a
merged round produces a change that fails a gate, the revert is more expensive
because more landed at once. I am accepting that risk over the alternative, which
is a run that stops before it has anything usable.

Targets are unchanged. Done still means all gates green on dev and holdout, final
SCORE >= 2.5x baseline (>= 14.19) and above every reference line including lzma-6
at 7.222. Fewer rounds does not mean a lower bar; if the bar is missed, `SUMMARY.md`
reports the shortfall honestly.
