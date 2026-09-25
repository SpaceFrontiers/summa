# Hot-Metadata Pinning (meta/data residency split)

Status: implemented (2026-07-22).

## Problem

Every query must touch certain small metadata sections — BMP block-offset
tables, sparse skip sections, doc-id maps, and the coarse query hierarchy. Summa maps whole
segment files and slices them zero-copy, so residency of those sections is
decided by the kernel's page-cache LRU, not by us. Under memory pressure the
kernel evicts them exactly like bulk data, and every subsequent query pays
major faults on structures it cannot skip. This is the remaining half of the
`slow BMP: 14116ms` pathology (block-data prefetch fixed the bulk half;
grid/starts/doc-map faults were still unpinned).

The classic production pattern for this is a **meta/data split**: each store
keeps a small offset/index part every lookup touches (pin it in RAM) separate
from the bulk payload (page-cache or direct I/O). Summa already has the
_layout_ half of this — every file is section-structured with lazy range
reads, and some metadata is resident because it is decoded to the heap (for
example, sparse dimension tables and global ANN routing artifacts). What was
missing is choosing residency for mmap-backed metadata and immutable heap
arrays touched by every ANN route.

## Residency of per-query structures today

| Structure       | "meta" part                                                      | Residency                 | "data" part                               | Residency                                                                       |
| --------------- | ---------------------------------------------------------------- | ------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| Sparse MaxScore | `DimensionTable` (SoA Vecs)                                      | heap (outside page cache) | block data                                | evictable mmap                                                                  |
|                 | skip section (`skip_bytes`)                                      | **pinnable**              |                                           |                                                                                 |
| BMP             | `block_data_starts`, E row offsets, H, doc maps                  | **pinnable**              | block data                                | evictable (MADV_RANDOM + WILLNEED prefetch)                                     |
|                 | 4-bit block grid D and superblock grid E                         | never pinned (see below)  |                                           |                                                                                 |
| Seismic         | term/row directories                                             | **pinnable**              | summaries, nomination rows, exact vectors | evictable mmap                                                                  |
| Dense flat      | header + doc-id map                                              | **pinnable**              | raw vectors                               | evictable (+ RANDOM/prefetch for ANN fields)                                    |
| ANN             | global routing/centroids/PQ tables and per-segment run directory | **pinnable**              | quantized codes and IDs                   | evictable (clustered: MADV_RANDOM + selected-run WILLNEED; flat TQ: sequential) |

Sizes for a representative 18.2M-vector segment at block size 32
(568,750 blocks, 71,094 superblocks, 278 coarse groups, SPLADE
dims ≈ 105,879):

| Structure                                  | Dense upper size  | Pin priority                                       |
| ------------------------------------------ | ----------------- | -------------------------------------------------- |
| BMP `block_data_starts`                    | ~4.34 MiB         | 2 (every scored block does an offset lookup)       |
| Sparse skip sections                       | workload-specific | 3 (every posting traversal)                        |
| BMP doc maps (6 B/padded vector)           | ~104.14 MiB       | 4 (only competitive candidates resolve through it) |
| E row-offset table                         | ~0.81 MiB         | 5 (selected E groups need one row lookup)          |
| Coarse H grid (`dims × ceil(groups / 2)`)  | ~14.04 MiB        | 5 (every query dimension sweeps its H row)         |
| Superblock E grid (`dims × ceil(SBs / 2)`) | ~3.50 GiB         | **never**                                          |
| Block D grid (`dims × ceil(blocks / 2)`)   | ~28.04 GiB        | **never**                                          |

D and E are meta-shaped but data-sized, so their payloads stay evictable.
Global LSP scans the small H level, then touches only selected independently
addressable E groups; block traversal does the same for D. Both mappings use
`MADV_RANDOM` plus bounded `MADV_WILLNEED` ranges to avoid readahead
amplification. The sizes above are dense four-bit upper bounds; the
row-local variable-width codec is smaller whenever groups need fewer bits.

### Fast-field header checkpoints

Single-value blockwise-linear fast fields retain at most 256 sparse header
checkpoints per reader, across all merged source blocks: at most 3 KiB of heap
payload. The directory is constructed after encoded-envelope validation and does
not grow during queries. Other codecs and multivalue columns allocate no such
directory. Values and text dictionary bytes retain their original byte owner
(file-backed with `MmapDirectory`); the checkpoint heap is not `mlock` residency or a pinned payload.

`SegmentMemoryStats.fast_field_metadata_heap_bytes` includes fast-field block
metadata plus checkpoints, and contributes to `estimated_heap_bytes()`. Row-stat
columns include their checkpoints in the existing row-stat heap counter. These
estimates still exclude existing lazy text dictionary tables and ordinal maps;
they are not a complete process heap measurement. See the
[Searchbench experiment](searchbench-comparison.md#dispatch-sparse-id-directory-and-envelope-ownership-experiment)
for the cost model and measured evidence.

### Seismic compact-directory residency

Seismic keeps its compact run directory on the heap. Each run also retains
separate zero-copy views of its term and logical-row directories. The existing
segment pin policy prioritizes term directories alongside BMP block offsets
(priority 2), then logical-row directories alongside document maps (priority 4).
Copy mode replaces only those views, and all directory lookups use them; mlock
mode locks the same extents. The original encoded run remains the owner of
forward vectors and nomination payloads and the source for byte-identical merge.
No format or additional residency policy changes are required. The default pin
budget remains zero. Copy-mode bytes count as additional sparse heap allocation;
the original mapped file extent remains file-backed and does not imply resident
RAM. Mlock consumes page-rounded OS lock capacity; logical-byte budgets do not
include this rounding or guarantee success. Disabled pinning still reports all
eligible mapped directory bytes as intended, with zero pinned bytes.

With `MmapDirectory` (the server/tool backend), summaries, nomination rows and
forward values remain evictable file-backed views. `FsDirectory`, RAM and HTTP
readers return heap-backed component buffers, so this out-of-core property does
not apply to those backends. Sparse residency accounting distinguishes heap,
mmap and pinned metadata for all supported formats. Pin failures and insufficient
budgets use the same intended/pinned/skipped/failed accounting as BMP and ANN.

The 1M-document four-source copy-merge fixture has 24.0 MB of row directories and
4.395 MB of term directories. Summary arrays occupy 2.138 GB in Seismic version 3
and 1.415 GB with version-4 lossless directory compression; version-5
cluster-ID compression reduces them to 1.328 GB. Compact
directories are eligible; encoded summary payloads remain evictable even with a
large pin budget. These are encoded sizes, not page-rounded lock costs or measured
RSS.
See the [memory-pressure evaluation](search-performance-review.md) for evidence.

On Linux, eligible copy-mode metadata is copied in 128 KiB chunks with one
additional chunk prefetched. The destination is allocated once after budget
admission; reads stay bounded even when the source mapping uses random access.
Other platforms retain their existing copy behavior.

## Phase 1 (implemented): budgeted metadata pinning

`segment/pin.rs` defines a process-wide `PinPolicy`:

- `--pin-metadata-budget-mb` (or `SUMMA_PIN_METADATA_BUDGET_MB`) — metadata
  budget per segment. The same bound is separately applied once to each
  index-global ANN generation, in routing-first priority order. Default 0
  disables pinning.
- `--pin-mode` (or `SUMMA_PIN_MODE`) — `mlock` (default; locks existing metadata
  pages and needs RLIMIT_MEMLOCK headroom) or `copy` (copies mapped metadata to
  the heap without special permissions). Heap allocations are outside the page
  cache but can still swap unless the host is swapless.

At `SegmentReader::open`, sections are pinned in the priority order above
until the budget is exhausted. The budget
is per segment, not per process, and old reader generations can overlap during
replacement. Loading a trained ANN generation additionally
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

## Reader reload ownership

Deletion-only commits must share immutable segment payloads, parsed metadata,
term/store caches and pin owners with older searchers. Only the deletion bitmap
and visibility views change; old searchers retain their original visibility.
New segment opens are bounded to two concurrent readers, including deletion
loading. Publication remains atomic: a failed or cancelled reload leaves the
previous searcher usable.

Reload memory is shared retained payloads + new segment payloads + per-generation
visibility bitmaps (roughly one bit per physical row) + at most two opens' scratch.
This bounds opening concurrency, not total index residency: the pin budget remains
per segment and old replaced segments remain alive while queries hold snapshots.
No stored format or pin-budget default changes are required.

## Phase 2 (implemented): cold-IO merge writes — see `docs/cold-io.md`

Implemented cold output uses write-behind writeback/cache-drop on Linux and
`F_NOCACHE` on macOS. Selected byte-identical ranges can use kernel-assisted
copying; other reads remain buffered/mmap-backed with advisory prefetch and
release. This reduces cache pollution; it is not a general direct-I/O read path
or a guarantee that bulk reads cannot evict warm pages. Revisit direct reads
only if pinning plus the existing advice leaves measurable eviction churn.
