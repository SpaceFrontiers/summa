# Block-Level Reorder with Stats-Guided Granularity

Status: design (2026-07-09), implemented.

## Problem

Record-level BP reorder pays two heavy costs: BP over every record
(18M entities on the big segment) and a rewrite that _scatters_ every record
(per-block gather + re-serialization). But the most common reorder trigger is
a block-copy merge of already-reordered sources — where the blocks are still
internally coherent and only their **order** was scrambled by concatenation.
Recovering superblock locality there does not require touching records at
all.

## Block-level reorder

Treat each existing block as one entity whose "terms" are its header dim
list (no posting decode needed):

- BP over `num_blocks` entities (~32× fewer than vectors with the production
  block size) down to `min_partition = superblock_size` — within a
  superblock, block order is irrelevant to pruning (the executor sorts local
  blocks by UB at query time), so only superblock _assignment_ matters.
- The rewrite is a **permuted block copy**: block bytes verbatim in the new
  order, `block_data_starts` recomputed, compressed grid groups regenerated
  from exact maxima retained in block headers, `sb_grid` recomputed in the
  same bounded pass, and document maps copied per block chunk (doc-id offsets
  patched for multi-source). No record unpacking, no per-block hashmaps, no
  padding changes.

The row-major grid rewrite is bounded independently of field size: each row
is spilled and consumed in order instead of retaining `dimensions ×
superblocks` in RAM. Source/local-block ownership is resolved once per block
and reused for every dimension, avoiding a prefix search in the
`dimensions × blocks` inner loop. Record-level output encoding likewise uses
bounded windows (at most twice the BP pool width) rather than buffering the
entire rewritten blob.

What it cannot do: tighten block upper bounds (block composition is fixed).
That remains record-level BP's job.

## The decision algorithm (`BpGranularity::Auto`)

Per (segment, field), the raw signal is block coherence:

```
coherence d = total_postings / total_terms
            = average number of records sharing a (block, dim) pair
```

`d` alone is **not value-independent**: it is dominated by the corpus's
dim-frequency distribution, not by ordering quality. A rare-dim corpus
(short title vectors — a dim in only 3 documents can never contribute more
than 3) caps `d` low even when perfectly clustered, while ubiquitous dims
inflate `d` even when records are fully scrambled. An absolute cutoff
inverts the decision on both.

So the decision normalizes `d` between two per-corpus bounds, both derived
from the same per-dim record frequencies `df_t` (one streaming pass over
block headers, posting-slice lengths only — no posting decode):

```
d_rand = P / Σ_t B·(1 − (1 − 1/B)^df_t)   expected d under random record→block assignment
d_max  = P / Σ_t ceil(df_t / block_size)   perfect per-dim packing bound (overestimates
                                           achievable d → biases toward record-level)
norm   = (d − d_rand) / (d_max − d_rand)   clamped to 0..=1
```

- `norm ≈ 0`: ordering is indistinguishable from random — record-level BP
  required (with the existing size-tiered BpBudget).
- `norm ≈ 1`: blocks are as internally coherent as the data allows —
  block-level reorder recovers everything reordering can still give.
- No headroom (`d_max ≈ d_rand`, e.g. only ubiquitous dims): record-level
  BP cannot help either, so `norm = 1` and the near-free blockwise pass
  runs.

Two estimator details:

- **Singleton dims (df=1) are excluded from all aggregates.** They
  contribute one pair to actual, random, and packed counts alike — zero
  ordering signal — and on id-heavy corpora (a unique dim per record) they
  compress the bounds enough to fake "no headroom" and mask real headroom
  in the informative dims.
- **Large segments are stride-sampled** (cap: 8192 blocks ≈ 262k vectors at
  the production block size 32). All aggregates come from the sampled
  sub-population, so the estimator stays consistent at both extremes; the
  decision log reports scanned/total blocks.

Threshold: `BLOCKWISE_NORM_COHERENCE_THRESHOLD = 0.5`. Every `Auto`
decision logs `norm`, `d`, `d_rand`, `d_max`, blocks scanned, scan time,
and the chosen granularity (and emits `summa_reorder_coherence{,_norm}`)
so the default can be tuned from production data. Explicit granularity
skips the scan entirely and emits only the granularity counter.

## Alignment with depth budgets and deepening

The granularity decision and the `BpBudget` depth machinery interact; three
rules keep them consistent:

- **Unconverged sources force record-level.** A segment marked
  `bp_converged = false` (wall-clock-truncated pass) is owed a deepening
  pass, and the output of the next pass over it — standalone reorder or
  merge — is what discharges that debt. `Auto` would measure the partial
  pass's residual coherence, potentially take the blockwise path (which
  cannot deepen record clustering) and report converged, silently ending
  the cascade at partial quality. `SegmentManager::merge_granularity`
  resolves `Records` whenever any source is unconverged.
- **Depth caps are converted to block units for blockwise BP.**
  `BpBudget::min_partition_docs` is in docs; blockwise BP entities are
  blocks. Unconverted, the optimizer's default large-segment cap (256
  vectors) would read as 256 _blocks_ and stop blockwise BP above the desired
  depth as a silent no-op. With the production defaults, 256 / 32 = 8 blocks,
  exactly one LSP superblock.
- **Partial record-level output legitimately reads `norm ≈ 0`.** Budgeted
  passes (`min_partition_docs` above block size) leave intra-block order
  random, so Auto keeps routing such segments to record-level until a
  full-depth pass lands — correct, not a mis-decision. A blockwise pass
  never changes `d` or `norm` at all: blocks are copied verbatim and the
  metric is invariant under block permutation.

Where Auto applies:

- **Merge-time reorder** (`reorder_on_merge`): sources that were reordered
  produce high-`d` fields → the merge does a near-free blockwise pass
  instead of full BP. Fresh/scrambled sources → record-level as before.
- **Background optimizer / manual reorder**: same per-field decision. A
  fresh date-clustered segment (naturally coherent) gets the cheap pass; a
  shuffled one gets full BP.
- Convergence: a blockwise pass reports `bp_converged = true` unless its
  wall-clock budget fired — coherence justified the coarse target, so there
  is nothing to deepen. Record-level budgeted passes keep the existing
  deepening semantics.

## Costs

|                  | record-level                    | block-level                           |
| ---------------- | ------------------------------- | ------------------------------------- |
| BP entities      | num_records                     | num_blocks (÷32 by default)           |
| BP depth target  | block (32 vectors by default)   | LSP superblock (8 blocks)             |
| rewrite          | scatter every record            | permuted memcpy + grid nibble permute |
| improves         | block UBs + superblock locality | superblock locality only              |
| interior padding | compacted                       | preserved                             |

## Future

- Persist per-field coherence in segment metadata so the optimizer can skip
  already-coherent segments without opening them.
- Heterogeneity-focused deep passes: record-level BP only inside partitions
  whose local coherence is below threshold.
