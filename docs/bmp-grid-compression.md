# BMP Grid Memory: Compression Research & Roadmap

## Memory anatomy of a BMP segment (V13/V14)

Everything is mmap-backed (page cache, not heap), but resident-set pressure
is real on memory-bound deployments. Per field, per segment:

| section        | bytes                          | 18M docs, 105879 dims, block 64 |
| -------------- | ------------------------------ | ------------------------------- |
| block grid (D) | `dims × ceil(num_blocks / 2)`  | **≈ 14.9 GB**                   |
| sb_grid (E)    | `dims × ceil(num_blocks / 64)` | ≈ 465 MB                        |
| doc maps (F+G) | `6 × num_virtual_docs`         | ≈ 108 MB                        |
| block data (B) | `2 × postings + headers`       | ≈ 6-8 GB (data, not overhead)   |

The **dense block grid dominates**: every one of `dims` vocabulary rows gets
a 4-bit cell in _every_ block, even though a typical SPLADE block touches
only a small fraction of the vocabulary. Measured coherence
`d = total_postings / total_terms ≈ 2-3` at block 64 implies the average
non-zero density of the grid is `total_terms / (dims × num_blocks)` — for
the production corpus roughly **3-4% non-zero**. The dense grid stores the
other ~96% as zero nibbles.

## Phase 1 (shipped): per-field `bmp_block_size` + grid MADV_RANDOM

Grid bytes scale as `1/block_size`. Moving a field from 64 → 256:

- grid: 14.9 GB → **3.7 GB** (4×), sb_grid: 465 MB → 116 MB (18M docs);
  at 50M docs: 41 GB → **10 GB**.
- pruning granularity coarsens: a block is now 256 docs, so a superblock
  covers 16k docs. The BMP paper (Mallia, Suel & Tonellotto, SIGIR 2024)
  favors b=8-32 for in-RAM MS MARCO-scale corpora; at 100k-dim vocabularies
  and 10M+ doc segments the grid does not fit that regime — coarser blocks
  are the correct trade (see also docs/budgeted-reorder.md research notes).
- Block size is **per field, uniform across all segments** (default 64) —
  set it in SDL at index creation; there is deliberately no auto-escalation
  and no mixed-block-size support (block-copy merge and blockwise reorder
  move blocks verbatim and validate uniformity loudly).
- The block grid is now `MADV_RANDOM`: queries at production scale read 32
  bytes per (query dim, surviving superblock) at effectively random offsets,
  and default kernel readahead amplified each probe into 128 KB of page
  cache (~4000×) — marching the entire data-sized grid into memory and
  OOMing cgroup-limited deployments. With MADV_RANDOM only the touched 4 KB
  pages fault in, and they stay cleanly reclaimable.

SDL:

```sdl
field embedding: sparse_vector<u32> [indexed<format: bmp, dims: 105879,
    max_weight: 5.0, bmp_block_size: 256>]
```

### 2-bit grid impacts — MEASURED, IMPLEMENTED (`bmp_grid_bits: 2`)

Halve the dense grid (dims × blocks / 4) by quantizing block UBs to 2 bits
(ceil-quantized, recall-safe like `quantize_u8_to_u4_ceil`). Same layout and
probe pattern, with no format fork beyond a footer flag.

| grid            | topical queries (blocks/q) | background-only queries (blocks/q) |
| --------------- | -------------------------- | ---------------------------------- |
| 4-bit (shipped) | 25.7                       | 6057                               |
| 3-bit (9 lvl)   | 25.8 (+0.4%)               | 6122 (+1.1%)                       |
| 2-bit (4 lvl)   | 25.8 (+0.4%)               | 6188 (+2.2%)                       |

Why so cheap: the exact u8 `sb_grid` prunes first and shields the block
grid — coarse block bounds only act inside surviving superblocks. **2-bit
grid ≈ free** (grid 10 GB → 5 GB at 50M docs / block 64; combine with
`bmp_block_size: 256` for 2.5 GB).

**Implemented**: SDL `indexed<bmp_grid_bits: 2>` (default 4; per field,
uniform across segments — merge/blockwise validate loudly). The V14 blob
footer's reserved field carries the cell width (0 = legacy 4-bit), so
existing segments stay readable. 2-bit currently uses an unrolled scalar
accumulate/mask kernel (4 cells/byte); SIMD u2 variants are a follow-up if
profiling asks for them — the block grid is probed only inside surviving
superblocks. Exactness is pinned by
`test_bmp_grid_bits_2_exact_parity_and_roundtrip` (identical top-k vs a
4-bit index over the same corpus, through block-copy merge and BP reorder).

### Exact integer bounds (implemented 2026-07-15)

### 4-bit posting weights — MEASURED, REJECTED

Snapping posting impacts to the u4 lattice (u8 multiples of 17; emulated at
build time so quantization applies to both scores and grid maxes) costs
real quality: **recall@10 vs the 8-bit index = 95.1%** on the same corpus,
matching the classical 3-5% loss for sub-8-bit impact quantization
(Anh & Moffat lineage). The saving is only ~25% of the postings section —
NOT 50%, because a BMP posting is 2 bytes `(local_slot: u8, impact: u8)`
and only the impact half shrinks: packed, 2 B → 1.5 B per posting. Bad
trade — weights stay 8-bit.

## Phase 3 (assessed, not recommended): dropping grid rows for rare dims

Term-centric static pruning (Carmel et al., SIGIR 2001 lineage) suggests
dropping upper-bound rows for rare dims and treating their UB
conservatively (global max weight for all blocks). But a conservative UB
inflates every block bound for queries containing rare dims — exactly the
discriminative dims BMP prunes best with — and with CSR rejected the
"row costs bf_t entries" argument is moot: under the dense grid rare-dim
rows are mmap pages that queries rarely fault in anyway.

## SSD-friendliness / paging behavior

The grid is mmap-backed; "SSD-friendly" means the resident set stays
bounded and misses are served as clean 4 KB NVMe refaults:

- **MADV_RANDOM on the grid** (shipped): kills readahead amplification —
  per query the grid working set is `query_dims × surviving_superblocks ×
4 KB` instead of ×128 KB. Pages are clean and reclaimable; under memory
  pressure the kernel evicts them and cold queries pay ~20-50 µs NVMe
  refaults per page instead of the pod OOMing.
- **sb_grid stays readahead-friendly and pinnable** (priority 4 in the pin
  budget): it is swept contiguously per query dim and is 64× smaller than
  the grid — keeping it resident preserves superblock pruning, which is
  what keeps the number of grid probes small in the first place.
- The block grid is deliberately never pinned (data-sized).

## Rollout