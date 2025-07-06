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
