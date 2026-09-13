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
