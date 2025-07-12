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
  partitions of this size. **4096 (= superblock 64 blocks × 64 docs) keeps
  nearly all of the superblock-pruning win at ~⅓ less depth** — the
  remaining levels only sharpen intra-superblock block ordering. A depth cap
  is a chosen target: the pass reports `converged = true`.
- `time_budget: Option<Duration>` — wall-clock deadline, checked between
  refinement passes and before each recursion (atomic flag across the rayon
  fan-out). Hitting it ends the pass cleanly with `converged = false`.

Per-segment metadata gains `bp_converged` (serde-default true for old
metadata). What you cannot budget: each pass still rewrites the whole sparse
blob (the 4-bit grid is row-major — any permutation rewrites every row), so
partial reorder bounds CPU/memory, not per-pass IO. Passes go through the
cold-IO writer, so at least they no longer evict the cache.

## Optimizer tiering (hermes-server)

- Segments `< --optimizer-large-segment-docs` (default 5M): full-depth BP.
- Larger: budgeted pass — depth capped at
  `--optimizer-partial-min-partition-docs` (default 4096) and wall-clock
  capped at `--optimizer-time-budget-secs` (default 600).
- Unconverged segments (deadline fired) are revisited only when no
  never-reordered work exists, at most one per
  `--optimizer-unconverged-cooldown-secs` (default 1800) — each follow-up is
  a full segment rewrite, so deepening is deliberately paced.
- Merge-time reorder stays unbudgeted: the merge already pays the rewrite
  and BP warm-starts from the (usually ordered) inputs.

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
three run on the bounded background pool (`cores/2` threads — merge-time
pool in `SegmentManager::background_cpu_pool`, optimizer pool sized
`max(--optimizer-threads, cores/2)`), keeping background CPU off the global
query pool:

1. **Forward-index build** (was fully serial). Real doc ids are assigned in
   ascending vid order, so each BMP block owns a contiguous real-id range —
   df counting is a rayon fold/reduce over blocks, and the CSR count/fill
   phases write disjoint per-block slices (safe `split_at_mut` carving, no
   locks). Also df counting aggregates per (block, dim) instead of per
   posting, which speeds up the serial path too.
2. **Graph bisection** (already parallel: `rayon::join` halves + parallel
   gain passes, wall-clock budgeted).
3. **Permuted blob write** (was fully serial). Output blocks encode
   independently; they are encoded in parallel in 4096-block windows and
   written serially in blob order (the window bounds buffered bytes).

## Default changes (2026-07-14, follow-up)

- `IndexConfig::default()` now uses `TieredMergePolicy::large_scale()` (the
  server already did). Tier floors only shape _when_ segments merge, and
  merge-time BP is wall-clock budgeted, so the policy is safe at any scale.
- `bp_memory_budget_bytes` default 2 GB → 24 GB (`--bp-memory-budget-mb`
  default 24576). It is a cap, not an allocation — memory use is proportional
  to the segment actually being reordered (~4 B/posting + ~28 B/doc). Sized
  from prod evidence: a 58M-doc / 5B-posting pass estimated 20.1 GB; smaller
  budgets trimmed it by dropping highest-df dims (2 GB dropped ~10% of
  eligible dims on 18M-doc merges).