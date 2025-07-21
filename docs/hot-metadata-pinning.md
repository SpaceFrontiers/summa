# Hot-Metadata Pinning (meta/data residency split)

Status: implemented (2026-07-22).

## Problem

Every query must touch certain small metadata sections — BMP block-offset
tables, sparse skip sections, doc-id maps, and the coarse query hierarchy. Hermes maps whole
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
reads, and some metadata is resident because it is decoded to the heap (for
example, sparse dimension tables and global ANN routing artifacts). What was
missing is choosing residency for mmap-backed metadata and immutable heap
arrays touched by every ANN route.

## Residency of per-query structures today

| Structure       | "meta" part                              | Residency                | "data" part | Residency                                    |
| --------------- | ---------------------------------------- | ------------------------ | ----------- | -------------------------------------------- |
| Sparse MaxScore | `DimensionTable` (SoA Vecs)              | heap (de-facto pinned)   | block data  | evictable mmap                               |
|                 | skip section (`skip_bytes`)              | **pinnable**             |             |                                              |
| BMP             | `block_data_starts`, E row offsets, H, doc maps | **pinnable**       | block data  | evictable (MADV_RANDOM + WILLNEED prefetch)  |
|                 | 4-bit block grid D and superblock grid E | never pinned (see below) |             |                                              |
| Dense flat      | header + doc-id map                      | **pinnable**             | raw vectors | evictable (+ RANDOM/prefetch for ANN fields) |
| ANN             | global routing/centroids/PQ tables and per-segment run directory | **pinnable** | quantized codes and IDs | evictable (clustered: MADV_RANDOM + selected-run WILLNEED; flat TQ: sequential) |

Sizes for a representative 18.2M-vector segment at block size 32
(568,750 blocks, 71,094 superblocks, 278 coarse groups, SPLADE
dims ≈ 105,879):

| Structure                                      | Dense upper size | Pin priority                                      |
| ---------------------------------------------- | ---------------- | ------------------------------------------------- |
| BMP `block_data_starts`                        | ~4.34 MiB        | 2 (every scored block does an offset lookup)      |
| Sparse skip sections                           | workload-specific | 3 (every posting traversal)                      |
| BMP doc maps (6 B/padded vector)               | ~104.14 MiB      | 4 (only competitive candidates resolve through it) |
| E row-offset table                             | ~0.81 MiB        | 5 (selected E groups need one row lookup)         |
| Coarse H grid (`dims × ceil(groups / 2)`)      | ~14.04 MiB       | 5 (every query dimension sweeps its H row)        |
| Superblock E grid (`dims × ceil(SBs / 2)`)     | ~3.50 GiB        | **never**                                         |
| Block D grid (`dims × ceil(blocks / 2)`)       | ~28.04 GiB       | **never**                                         |

D and E are meta-shaped but data-sized, so their payloads stay evictable.
Global LSP scans the small H level, then touches only selected independently
addressable E groups; block traversal does the same for D. Both mappings use
`MADV_RANDOM` plus bounded `MADV_WILLNEED` ranges to avoid readahead
amplification. The sizes above are dense four-bit upper bounds; the
row-local variable-width codec is smaller whenever groups need fewer bits.

## Phase 1 (implemented): budgeted metadata pinning

`segment/pin.rs` defines a process-wide `PinPolicy`:

- `--pin-metadata-budget-mb` (or `HERMES_PIN_METADATA_BUDGET_MB`) — metadata
  budget per segment. The same bound is separately applied once to each
  index-global ANN generation, in routing-first priority order. Default 0
  disables pinning.
- `--pin-mode` (or `HERMES_PIN_MODE`) — `mlock` (default; zero-copy, needs RLIMIT_MEMLOCK
  headroom) or `copy` (heap copy; no permissions needed, duplicates bytes,
  immune to eviction because production runs swapless).

At `SegmentReader::open`, sections are pinned in the priority order above
until the budget is exhausted. Loading a trained ANN generation additionally
locks HNSW/two-level topology, parent centroids, PQ/OPQ tables, and then leaf
centroids. Each segment locks its compact cluster-run lookup directory. The
corpus-sized PQ/binary run columns and exact rerank vectors remain mmap-backed
and evictable. ANN queries prefetch only the selected physical run extents;
exact reranking applies the same bounded prefetch in synchronous and
asynchronous execution. Fail-loud: mlock failure logs a warning and continues;
`SegmentMemoryStats` carries `pin_intended_bytes` vs `pinned_metadata_bytes`
for per-segment accounting, while generation pinning logs its own totals.

Suggested starting budget: 150–300 MiB/segment depending on skip-section and
dense-field metadata — enough for offsets, H, and BMP doc maps without ever
pinning the corpus-sized D/E payloads.

## Phase 2 (implemented): cold-IO merge writes — see `docs/cold-io.md`

Bulk reads that bypass the page cache entirely (merges, cold posting/vector
scans) so they can never evict warm pages. Requires a direct-I/O read variant
in the `Directory` layer. The merge-side `MADV_SEQUENTIAL`/`MADV_DONTNEED`
discipline already approximates most of the benefit; revisit only if Phase 1
plus the existing madvise work leaves measurable eviction churn.
