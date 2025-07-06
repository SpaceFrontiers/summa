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

- BP over `num_blocks` entities (~64× fewer than records) down to
  `min_partition = superblock_size` — within a superblock, block order is
  irrelevant to pruning (the executor sorts local blocks by UB at query
  time), so only superblock _assignment_ matters.
- The rewrite is a **permuted block copy**: block bytes verbatim in the new
  order, `block_data_starts` recomputed, grid rows re-permuted nibble-wise
  (row-at-a-time streaming, same cost class as the merge's grid pass),
  `sb_grid` recomputed in the same pass, doc maps copied per 64-entry block
  chunk (doc-id offsets patched for multi-source). No record unpacking, no
  per-block hashmaps, no padding changes.

What it cannot do: tighten block upper bounds (block composition is fixed).
That remains record-level BP's job.

## The decision algorithm (`BpGranularity::Auto`)

Per (segment, field), from footer stats alone — zero scan cost:

```
coherence d = total_postings / total_terms
            = average number of records sharing a (block, dim) pair
```

- `d ≈ 1–2`: blocks are scrambled — every dim appears once or twice per
  block. Record-level BP required (with the existing size-tiered BpBudget).
- `d` high (records in a block share vocabulary): blocks are internally
  coherent — block-level reorder recovers everything reordering can still
  give. Threshold: `BLOCKWISE_COHERENCE_THRESHOLD = 4.0` (average of 4
  records per block sharing each dim), chosen conservatively; every decision
  logs the measured `d`, the threshold, and the chosen granularity so the
  default can be tuned from production logs.

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
| BP entities      | num_records                     | num_blocks (÷64)                      |
| BP depth target  | block (64 docs)                 | superblock (64 blocks)                |
| rewrite          | scatter every record            | permuted memcpy + grid nibble permute |
| improves         | block UBs + superblock locality | superblock locality only              |
| interior padding | compacted                       | preserved                             |

## Future

- Persist per-field coherence in segment metadata so the optimizer can skip
  already-coherent segments without opening them.
- Heterogeneity-focused deep passes: record-level BP only inside partitions
  whose local coherence is below threshold.