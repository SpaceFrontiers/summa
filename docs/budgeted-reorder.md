# Budgeted (Partial) BP Reordering

Status: design (2026-07-09), implemented.

## Problem

Full-depth BP on a large segment (18M+ docs ⇒ ~18 bisection levels × swap
passes over all postings) is hours of CPU, so big segments were reordered
rarely or never — exactly the segments where pruning quality matters most.

## Design: BP as an anytime algorithm

Stopping BP at any depth or deadline yields a valid permutation, and quality
is concave in work (top levels capture most of the log-gap gain). Because a
segment's layout _is_ the resume state, a follow-up pass warm-starts from the
previous order: early levels converge in ~0 swaps (the existing early-exit
fires) and the budget flows to deeper levels — repeated budgeted passes
converge to full-BP quality.

`BpBudget` (threaded through every reorder path):

- `min_partition_docs: Option<usize>` — depth cap; stop bisection at
  partitions of this size. **256 (= the default LSP superblock, 8 blocks ×
  32 vectors) reaches the hierarchy's coarse-pruning unit**; remaining
  levels only sharpen intra-superblock block ordering. The graph routine
  completes that chosen target normally; segment metadata deliberately
  records a depth-capped optimizer pass as `bp_converged = false` so the
  deepening ladder can later reach full block depth.
- `time_budget: Option<Duration>` — wall-clock deadline, checked between
  refinement passes and before each recursion (atomic flag across the rayon
  fan-out). Hitting it ends the pass cleanly with `converged = false`.

Per-segment metadata gains `bp_converged` (serde-default true for old
metadata) and `bp_unconverged_passes` (serde-default zero). What you cannot
budget away: each pass still rewrites the whole sparse
blob (the 4-bit grid is row-major — any permutation rewrites every row), so
partial reorder bounds CPU/memory, not per-pass IO. Passes go through the
cold-IO writer, so at least they no longer evict the cache.

## Optimizer tiering (summa-server)

- Segments `< --optimizer-large-segment-docs` (default 5M): full-depth BP.
- Larger: budgeted pass — depth capped at
  `--optimizer-partial-min-partition-docs` (default 256) and wall-clock
  capped at `--optimizer-time-budget-secs` (default 600).
- Unconverged segments (deadline fired) are revisited alongside fresh work,
  with at most one deepening pass active globally. After it completes, the
  next pass waits `--optimizer-unconverged-cooldown-secs` (default 600).
  Starting cooldown at completion is important because a full sparse rewrite
  can outlast the BP deadline; starting it at launch caused the replacement
  segment to be requeued immediately.
- The optimizer deepens a replacement lineage only while its consecutive
  budget-exhausted rewrite count is below
  `--optimizer-max-unconverged-passes` (default 3, including its initial
  partial pass). The counter follows replacement IDs, includes a partial
  merge-time BP pass, and resets after convergence or a non-reordered merge.
  `0` disables optimizer follow-up deepening; it does not disable explicitly
  configured merge-time reorder.
- Merge-time reorder uses its own `--merge-bp-budget-secs` deadline. A server
  value of `0` explicitly selects an unbudgeted pass. A truncated output joins
  the same paced deepening ladder.

Observability: `converged` in reorder logs, `bp_converged` in metadata, and
the Grafana superblock/block skip-ratio panels are the quality metric — a
partial pass shows up as high superblock skip ratio with a middling block
skip ratio that improves with each deepening pass.

## BP gain-ranking fix (found while testing)

The depth-cap regression test exposed a quality bug in the BP port: both
halves ranked docs by raw "gain of moving to the other side" and the top-mid
went left, so a misplaced left doc (wants right) and a misplaced right doc
(wants left) ranked identically and could never be exchanged. The reference
formulation negates the right half, producing one coherent
"belongs-right-ness" key; the partition takes the lowest-mid as the left
half. Fixed in `compute_gains` + the selection comparator; pinned by
`test_bp_depth_cap_separates_clusters_and_converges`. Expect measurably
better pruning from all reorders after this fix.

## Deepening ladder (2026-07-14)

Semantics change for aggressive continuous background reordering:

- A pass with a **depth floor above block granularity** (the optimizer's
  budgeted first pass on large segments, `min_partition_docs = 4096`) is now
  recorded `bp_converged = false` — by definition it has not reached
  block-level order. Previously it reported converged and large segments
  were permanently stuck at superblock-granularity order.
- **Merge-time BP is wall-clock-budgeted** (`--merge-bp-budget-secs`,
  default 600; `IndexConfig::merge_bp_time_budget`). A truncated merge pass
  still writes a valid, better-ordered segment, marked unconverged. This
  caps how long one merge holds a merge slot — previously a 64-source/18M-doc
  merge ran full-depth BP inline (10-30+ min per slot), which is how merge
  backlogs (350 segments at 30M docs) built up.
- The optimizer's **deepening is no longer starved by fresh segments**:
  under continuous ingestion fresh candidates arrive every commit, and the
  old "deepen only when idle" rule postponed deepening indefinitely. Now one
  unconverged segment is deepened per cooldown window
  (`--optimizer-unconverged-cooldown-secs`, default now 600, was 1800)
  alongside fresh work. Deepening passes run **full depth** with the wall
  clock budget — warm-starting makes already-ordered prefixes nearly free,
  so each pass pushes deeper until one beats the clock and converges.
  Only one deepening pass runs at a time, and its cooldown starts when the
  complete rewrite finishes (success, error, cancellation, or panic), not
  when it starts. This prevents long passes from continuously replacing and
  immediately requeueing their own unconverged output.

- **Merge fan-in default raised** (`TieredMergePolicy::large_scale()`
  `max_merge_at_once: 10 → 24`, baked in — no tuning flags): wide fan-in
  absorbs continuous-ingestion floods of small memtable segments in ~2.4×
  fewer merge passes. Giant merges are safe because output docs are capped
  by `max_merged_docs` and merge-time BP is wall-clock budgeted.

Net effect: merges are fast and bounded; order quality converges in the
background via warm-started passes; `bp_converged` now means
"block-granularity order reached".

## Single-pass parallelism (2026-07-14)

A single reorder pass has three phases; all three are now parallel and all
three run on one process-wide, bounded background pool. In `summa-server`,
`--optimizer-threads` sets its width when the periodic optimizer is enabled;
otherwise merge-time and manual BP share the `cores/2` fallback pool. There is
never a pool per index or per pass, and background CPU stays off the global
query pool:

1. **Forward-index build** (was fully serial). Real doc ids are assigned in
   ascending vid order, so each BMP block owns a contiguous real-id range —
   df counting uses one dense atomic vocabulary table shared by the bounded
   pool, and the CSR count/fill phases write disjoint per-block slices (safe
   `split_at_mut` carving, no locks). This avoids one vocabulary-sized hash map
   per worker. Candidate discovery retains only a budgeted low-frequency set.
2. **Graph bisection** (already parallel: `rayon::join` halves + parallel
   gain passes, wall-clock budgeted).
3. **Permuted blob write** (was fully serial). Output blocks encode
   independently; they are encoded in parallel in memory-derived windows and
   written serially in blob order. Grid spill and encoded-window budgets share
   the remaining per-pass allowance after persistent document maps.

Measured (300k docs / 14.4M postings, RamDirectory, aarch64, release):
forward index 135 ms → 50 ms (2.7×); blob-encode phase ~3.3 s → ~1.0 s;
`writer.reorder()` end-to-end 4.7 s → 2.36 s (2.0×). BP itself (1.3 s
full-depth here) is now the dominant phase, and it is budgeted/warm-started.
Evidence: `bench_forward_index_build` (`#[ignore]`) in
`summa-core/src/index/tests/bmp.rs`.

## Default changes (2026-07-14, follow-up)

- `IndexConfig::default()` now uses `TieredMergePolicy::large_scale()` (the
  server already did). Tier floors only shape _when_ segments merge, and
  merge-time BP is wall-clock budgeted, so the policy is safe at any scale.
- `bp_memory_budget_bytes` default 2 GB → 24 GB (`--bp-memory-budget-mb`
  default 24576). It is a cap, not an allocation — memory use is proportional
  to the segment actually being reordered (~4 B/posting + ~32 B/doc). Sized
  from prod evidence: a 58M-doc / 5B-posting pass estimated 20.1 GB; smaller
  budgets trimmed it by dropping highest-df dims (2 GB dropped ~10% of
  eligible dims on 18M-doc merges).

## Process-level resource bounds (2026-07-15)

Thread width and operation concurrency are intentionally independent:

- `--optimizer-threads` controls CPU parallelism inside the shared pool. A
  running BP pass is expected to keep up to that many cores busy.
- `--optimizer-concurrent-passes` (default 2, hard maximum 2) gates complete
  reorder passes across every index and every entry point: optimizer,
  merge-time, and manual. An explicit force merge pauses new background
  admission and reserves all but one slot after pre-existing merges drain.
  Concurrent passes share the CPU pool, but each may consume its own forward
  index budget and rewrite buffers.
- `--bp-memory-budget-mb` is therefore a **per-pass algorithmic working-set
  bound**. It covers record maps, forward CSR, candidate/remap storage,
  vocabulary degree arrays (including parallel recursion), and rewrite
  grid/encode windows. It does not include source/output mmap pages, directory
  implementation buffers, unrelated merge structures, or indexing. A
  conservative process estimate starts with `concurrent passes × memory
budget`, then adds those external consumers.

When the dense vocabulary table itself cannot fit, the pass emits identity or
block order and remains explicitly unconverged. Recursion parallelism is
reduced to the number of vocabulary-sized degree arrays that fit. Memory
pressure can therefore reduce reorder quality or parallelism, but cannot be
misreported as convergence and trigger an unbounded immediate loop.

The gate is acquired before expensive forward-index construction. Scheduler
permits are nonblocking, indexes are visited with a rotating start point, fresh
segments are ordered small-first, and at most one unconverged candidate is
selected first. This prevents a quiet index or repeated deepening candidate
from monopolizing capacity.

## Parallel selection and objective stopping (2026-07-23)

Coarse record-level partitions no longer build an 8-byte `usize` index for
every entity and run serial `select_nth_unstable_by`. Four exact parallel radix
histogram passes select the same `(gain.total_cmp(), old_index)` median, write
the two output halves in parallel, and accumulate moved-term degree deltas in
thread-local lazy arrays. The number of arrays is taken from the existing
degree-array budget, so this increases CPU utilization without multiplying the
configured memory bound. Partitions below 1M entities retain quickselect: its
allocation is small there, it is faster than four radix scans, and its
within-half permutation preserves the established BP symmetry breaking.

Coarse BP partitions (1M+ entities) now measure the exact negative BiMLogA
bisection cost
`Σ_{s,t} d(s,t) × (log2(d(s,t) + 1) - log2(|s|))` after each refinement.
Approximate median refinements are retained because a reciprocal move may
improve the exact objective only after an intermediate step. After a
four-iteration warm-up, two consecutive iterations that fail to improve the
best exact objective by at least `1e-6` stop that partition. The static
iteration count is now only an upper safety bound. Fine partitions retain the
cheaper gain-cooling stop because a serial full-vocabulary objective scan costs
more there than the posting passes it can avoid. Long passes log progress every
30 seconds and export aggregate depth, partition, iteration, entity-pass, swap,
duration, and stop-reason metrics.

## Gain arithmetic review (2026-09-05)

Before this pass, the gain loop recomputed two logarithms and a division per posting.
Within a refinement, each term has only two possible signed contributions,
one for each source half. Document term order and strict floating-point
operation order determine gain bits, median ties, and persisted permutations;
reassociation of this reduction is therefore not an admissible SIMD shortcut.

The implemented optimization caches those two contributions once
per active term per refinement, then gathers and sums them in the original
document term order. It reuses the cache across partitions on each admitted
degree lane and charges `8 × vocabulary × lanes` bytes plus vector descriptors
against remaining graph scratch allowance before allocation. Cache admission
does not drop extra dimensions or reduce the existing degree-lane count; when
it does not fit, the kernel retains direct computation. This changes expensive
arithmetic from posting-proportional to active-term-proportional while leaving CSR traversal,
selection, stopping rules, formats, and rewrite ownership unchanged.

Same-fixture timings, exact gain/permutation bytes, memory accounting, assembly
and validation are recorded in the
[reordering review](reordering-performance-review.md). This cache uses remaining
allowance without changing dimension selection or degree-lane admission. It
does not fix the text adapter's older incomplete memory accounting; that
adapter retains direct gain computation until its complete remaining budget
can be supplied.

## Deleted rows and optimizer admission

The existing optimizer also schedules threshold-based [row compaction](row-deletion.md).
Eligible compaction sources are excluded from fresh/deepening BP work in that scan.
Compaction shares the optimizer whole-pass gate, merge capacity and CPU pool; one
automatic compaction may run globally, followed by a separate completion cooldown.
It does not reset exhausted BP attempt counts or consume a BP attempt. Filtering a
previously reordered BMP layout invalidates convergence while retaining its order,
so any allowed follow-up BP uses the existing bounded deepening policy.

## Seismic follow-up accounting

The optimizer shares cooldown and capacity gates with Seismic maintenance, but
its stopping criterion differs from BP's bounded deepening attempts. Seismic
uses persisted `seismic_no_progress_passes`: productive published replacements
reset it and remain eligible, while consecutive unchanged-debt replacements stop
at `--optimizer-max-unconverged-passes`. Zero still disables automatic follow-ups.
Failed/cancelled publication changes no persisted progress state; existing failure
backoff remains separate. BMP's total partial-pass bound is unchanged.
