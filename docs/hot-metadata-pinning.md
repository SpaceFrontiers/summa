# Hot-Metadata Pinning (meta/data residency split)

Status: design (2026-07-08), Phase 1 implemented.

## Problem

Every query must touch certain small metadata sections — BMP block-offset
tables, sparse skip sections, doc-id maps, superblock grids. Hermes maps whole
segment files and slices them zero-copy, so residency of those sections is
decided by the kernel's page-cache LRU, not by us. Under memory pressure the
kernel evicts them exactly like bulk data, and every subsequent query pays
major faults on structures it cannot skip. This is the remaining half of the
`slow BMP: 14116ms` pathology (block-data prefetch fixed the bulk half;
grid/starts/doc-map faults were still unpinned).

The classic production pattern for this is a **meta/data split**: each store
keeps a small offset/index part every lookup touches (pin it in RAM) separate
from the bulk payload (page-cache or direct I/O). Hermes already has the
_layout_ half of this — every file is section-structured with lazy range
reads, and some metadata is de-facto resident because it is deserialized to
the heap (sparse dimension tables, ANN blobs after first use). What was
missing is choosing residency for the mmap-backed metadata.

## Residency of per-query structures today

| Structure       | "meta" part                              | Residency                | "data" part | Residency                                    |
| --------------- | ---------------------------------------- | ------------------------ | ----------- | -------------------------------------------- |
| Sparse MaxScore | `DimensionTable` (SoA Vecs)              | heap (de-facto pinned)   | block data  | evictable mmap                               |
|                 | skip section (`skip_bytes`)              | **pinnable**             |             |                                              |
| BMP             | `block_data_starts`, `sb_grid`, doc maps | **pinnable**             | block data  | evictable (MADV_RANDOM + WILLNEED prefetch)  |
|                 | 4-bit `grid`                             | never pinned (see below) |             |                                              |
| Dense flat      | header + doc-id map                      | **pinnable**             | raw vectors | evictable (+ RANDOM/prefetch for ANN fields) |
| ANN blobs       | heap after first search                  | de-facto pinned          | —           | —                                            |

Sizes for a representative 18.2M-doc production segment (284,690 blocks,
4,449 superblocks, SPLADE dims ≈ 105,879):

| Structure                           | Size         | Pin priority                                 |
| ----------------------------------- | ------------ | -------------------------------------------- |
| BMP `block_data_starts`             | ~2.3 MB      | 1 (every scored block does an offset lookup) |
| Sparse skip sections                | tens of MB   | 2 (every posting traversal)                  |
| Flat + BMP doc-id maps (6 B/vector) | ~110 MB each | 3 (every top-k resolution)                   |
| BMP `sb_grid` (dims × superblocks)  | ~470 MB      | 4 (every query dim reads a row)              |
| BMP 4-bit `grid` (dims × blocks/2)  | ~15 GB       | **never**                                    |

The 4-bit grid is meta-shaped but data-sized: it must stay evictable and is
treated like bulk data (rows are read contiguously per query dim, so default
readahead serves it well).

## Phase 1 (implemented): budgeted metadata pinning

`segment/pin.rs` defines a process-wide `PinPolicy`:

- `HERMES_PIN_METADATA_BUDGET_MB` — bytes of metadata to pin **per segment**
  (default 0 = disabled), or `set_pin_policy()` programmatically before the
  first segment open.
- `HERMES_PIN_MODE` — `mlock` (default; zero-copy, needs RLIMIT_MEMLOCK
  headroom) or `copy` (heap copy; no permissions needed, duplicates bytes,
  immune to eviction because production runs swapless).

At `SegmentReader::open`, sections are pinned in the priority order above
until the budget is exhausted. Fail-loud: mlock failure logs a warning and
continues; `SegmentMemoryStats` carries `pin_intended_bytes` vs
`pinned_metadata_bytes` so an operator can see the gap. Bulk data is never
pinned.

Suggested starting budget: 150–700 MB/segment depending on host RAM —
enough for offsets + skips + doc maps everywhere, plus `sb_grid` on hosts
that can afford it.

## Phase 2 (implemented): cold-IO merge writes — see `docs/cold-io.md`

Bulk reads that bypass the page cache entirely (merges, cold posting/vector
scans) so they can never evict warm pages. Requires a direct-I/O read variant
in the `Directory` layer. The merge-side `MADV_SEQUENTIAL`/`MADV_DONTNEED`
discipline already approximates most of the benefit; revisit only if Phase 1
plus the existing madvise work leaves measurable eviction churn.
