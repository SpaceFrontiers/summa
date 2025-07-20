# BMP Grid Memory: Compression Research & Roadmap

## Memory anatomy of a BMP segment (V15)

Everything is mmap-backed (page cache, not heap), but resident-set pressure
is real on memory-bound deployments. Counts below are per field and use the
current production settings: 105,879 dimensions, 32 vectors per block, a
64-block superblock, and a dense 4-bit block grid. The cardinality is sparse
**vectors/ordinals**, not logical documents; a multi-valued field can therefore
be many times larger than its document count.

`num_blocks = ceil(num_vectors / 32)`. Sizes are decimal GB/TB, matching disk
tools; binary sizes are included where useful.

| section          | exact V15 bytes | 18M vectors | 100M vectors | 1B vectors |
| ---------------- | ---------------: | ----------: | -----------: | ---------: |
| block grid (D)   | `dims × ceil(num_blocks / 2)` | 29.78 GB (27.73 GiB) | 165.44 GB (154.07 GiB) | **1.654 TB (1.504 TiB)** |
| sb_grid (E)      | `dims × ceil(num_blocks / 64)` | 0.931 GB (0.867 GiB) | 5.170 GB (4.815 GiB) | 51.70 GB (48.15 GiB) |
| doc maps (F+G)   | `6 × num_virtual_docs` | 0.108 GB | 0.600 GB | 6.000 GB |
| block starts (A) | `8 × (num_blocks + 1)` | 4.50 MB | 25.00 MB | 250.00 MB |
| block data (B)   | `2P + 8T + 8N` | workload-dependent | workload-dependent | workload-dependent |

Thus a single 1B-vector BMP field needs about **1.706 TB** for the two grids
alone at block size 32/4-bit, before postings, document maps, segment metadata,
temporary merge output, or a second BMP field. Segment boundaries add a small
amount of row-padding overhead. Two 1B-vector fields need about **3.412 TB**
for Sections D and E alone.

In the block-data formula, `P` is the posting count, `T` is the number of
non-zero `(block, dimension)` pairs, and `N` is the number of non-empty blocks.
The blob also has at most seven alignment bytes and a 72-byte footer. These
small fixed sections do not change the conclusion: the block grid dominates.

The **dense block grid dominates**: every one of `dims` vocabulary rows gets
a 4-bit cell in _every_ block, even though a typical SPLADE block touches
only a small fraction of the vocabulary. The exact non-zero density is
`total_terms / (dims × num_blocks)` using the V15 footer counters. Smaller
blocks reduce that density while increasing the number of dense cells.

## Shipped controls: block size, grid width, and paging

Grid bytes scale approximately as `1/block_size`. Moving a field from the
current block size 32 to 256 would:

- At 18M vectors: grid 29.78 GB → **3.72 GB** and sb_grid
  0.931 GB → **0.116 GB** (8×).
- At 1B vectors: grid 1.654 TB → **206.8 GB** and sb_grid
  51.70 GB → **6.46 GB** (8×).
- Pruning granularity would coarsen by the same factor: a block would contain
  256 vectors and a 64-block superblock 16,384 vectors. This is not a free
  compression knob. The production choice remains **32** until representative
  relevance and tail-latency tests justify a larger value.
- Block size is per field and uniform across all segments. There is deliberately
  no auto-escalation or mixed-block-size support: block-copy merge and blockwise
  reorder move blocks verbatim and validate uniformity.
- The block grid is now `MADV_RANDOM`: queries at production scale read 32
  bytes per (query dim, surviving superblock) at effectively random offsets,
  and default kernel readahead amplified each probe into 128 KB of page
  cache (~4000×) — marching the entire data-sized grid into memory and
  OOMing cgroup-limited deployments. With MADV_RANDOM only the touched 4 KB
  pages fault in, and they stay cleanly reclaimable.

SDL:

```sdl
field embedding: sparse_vector<u32> [indexed<format: bmp, dims: 105879,
    max_weight: 5.0, bmp_block_size: 32, bmp_grid_bits: 4>]
```

## What “compression” means in the original BMP paper

The [SIGIR 2024 BMP paper](https://arxiv.org/pdf/2405.01117) evaluates two
representations of each dimension's block-max row:

- **BMP-Dense** stores every 8-bit maximum, including zeros.
- **BMP-Sparse**, called “compressed” in Table 1, zero-suppresses each row:
  only non-zero `(offset, maximum)` pairs are kept. The
  [reference implementation](https://github.com/pisa-engine/BMP) partitions a
  row into groups of 256 blocks and stores those pairs inside each group.
  Query evaluation scans the pairs and scatters them into a dense accumulator.

This is not entropy compression of the whole index, posting compression, or
Hermes's 4-bit packing. On MS MARCO with block size 8, the paper reports the
block-max structure shrinking from 30 GB dense to 5.5 GB compressed; at block
size 256 it reports 0.9 GB dense versus 1.2 GB compressed, so sparse storage
can even lose once rows become sufficiently dense.

Hermes does **not** store BMP-Sparse in production. It currently has:

- a dense block grid with independently addressable 4-bit cells (or optional
  2-bit cells), using ceil quantization so the bounds remain rank-safe;
- an exact dense 8-bit superblock grid;
- BP document ordering, which improves bound tightness and zero density;
- two-level superblock pruning and `MADV_RANDOM` on the block grid.

## Why the original sparse grid is wrong for the current executor

The [2025 SP paper](https://arxiv.org/html/2504.17045) introduced two-level
superblock traversal but left both grids uncompressed. Its
[2026 LSP follow-up](https://arxiv.org/html/2602.02883) directly tests
compression with this access pattern. It finds the original BMP-Sparse up to
**76× slower** because surviving superblocks cause random block-group access,
rather than BMP's original sequential full-row scan.

Correctness is cross-checked entry-exact between layouts:

| config    | density | grid memory: dense → CSR | block-UB compute: dense → CSR | probes hitting |
| --------- | ------- | ------------------------ | ----------------------------- | -------------- |
| block 64  | 3.5%    | 1562 → 227 MB (**6.9×**) | 12 → 59 ns/probe (**5.1×**)   | 94%            |
| block 256 | 10.4%   | 391 → 137 MB (**2.9×**)  | 11 → 72 ns/probe (**6.3×**)   | 99%            |

Two hypotheses failed:

1. **"Sparse rows skip absent dims"** — real query dims are Zipf-head dims,
   present in nearly every superblock (94-99% probe hit rate). The skip
   advantage barely exists; every hit pays a binary search plus a scalar
   scatter instead of one 32-byte SIMD sweep.
2. **The memory win shrinks exactly where large fields are going**: at
   block 256 density triples, so CSR yields only 2.9× — for a 6.3×
   hot-loop regression.

Note also that the dense grid is mmap-backed: rows of dims that queries
never touch are never faulted in, so the _resident_ dense grid is already
concentrated on head-dim rows — the same rows CSR must keep hot.

## Best next grid experiment: SIMDBP-256*

The 2026 LSP follow-up proposes a better representation for exactly this access
pattern:

1. Partition each dimension's block and superblock maxima into independently
   decodable groups of 256 integers.
2. Bit-pack each group at its local maximum width, so an all-zero group needs
   no payload bits and low-valued groups use fewer bits.
3. Put the group-width selectors at the front of each row so a randomly chosen
   group can be located and decoded without walking earlier payload groups.
4. Decode 256 values into `u16` lanes, allowing twice as many values per SIMD
   instruction as the conventional `u32` SIMDBP decoder.

The paper reports that `SIMDBP-256*` is up to 1.5× faster than conventional
SIMDBP-256 for random group access. On MS MARCO at block size 32, its Table 7
reports 8.0 GB for dense 8-bit block/superblock maxima, 4.2 GB for BMP-Sparse,
3.7 GB for 8-bit SIMDBP-256*, and **1.5 GB for SIMDBP-256* with 4-bit
maxima**. Unlike BMP-Sparse, its safe-search latency stays near the dense
representation.

Those ratios cannot be applied directly to Hermes: its block grid is already
4-bit, its superblock grid is 8-bit, and its BP ordering and density differ.
The remaining opportunity is the zero/low-width compression inside each
256-cell group, plus converting the superblock grid to ceil-quantized 4-bit.
The latter alone halves Section E but saves only about 3% of current total grid
bytes because a superblock spans 64 blocks.

A Hermes prototype must retain the current rank-safe semantics: calculate the
exact u8 maximum, ceil it to the stored lattice, and losslessly bit-pack that
ceiling. It must be evaluated on production-shaped, x86 data for:

- bytes per block and superblock maximum, including selectors/offsets;
- hot and cold p50/p99 latency, page faults, and decoded groups;
- exact top-k parity with the current 4-bit dense grid;
- build/reorder/streaming-merge throughput and peak memory.

This is a higher-priority experiment than increasing `bmp_block_size`: it can
remove zero and low-width storage without coarsening the 32-vector pruning
unit. It is not implemented yet.

## Existing dense-grid option: 2-bit block maxima

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
grid ≈ free** in this experiment. At the current block size 32, it changes the
100M-vector grid from 165.44 GB to 82.72 GB, and the 1B-vector grid from
1.654 TB to 827.18 GB.

**Implemented**: SDL `indexed<bmp_grid_bits: 2>` (default 4; per field,
uniform across segments — merge/blockwise validate loudly). The V15 blob
footer carries the cell width (2 or 4); older segments must be rebuilt.
2-bit currently uses an unrolled scalar
accumulate/mask kernel (4 cells/byte); SIMD u2 variants are a follow-up if
profiling asks for them — the block grid is probed only inside surviving
superblocks. Exactness is pinned by
`test_bmp_grid_bits_2_exact_parity_and_roundtrip` (identical top-k vs a
4-bit index over the same corpus, through block-copy merge and BP reorder).

## Exact integer bounds (implemented 2026-07-15)

## 4-bit posting weights: measured and rejected

Snapping posting impacts to the u4 lattice (u8 multiples of 17; emulated at
build time so quantization applies to both scores and grid maxes) costs
real quality: **recall@10 vs the 8-bit index = 95.1%** on the same corpus,
matching the classical 3-5% loss for sub-8-bit impact quantization
(Anh & Moffat lineage). The saving is only ~25% of the postings section —
NOT 50%, because a BMP posting is 2 bytes `(local_slot: u8, impact: u8)`
and only the impact half shrinks: packed, 2 B → 1.5 B per posting. Bad
trade — weights stay 8-bit.

## Dropping rare-dimension rows: assessed and rejected

Term-centric static pruning (Carmel et al., SIGIR 2001 lineage) suggests
dropping upper-bound rows for rare dims and treating their UB
conservatively (global max weight for all blocks). But a conservative UB
inflates every block bound for queries containing rare dims — exactly the
discriminative dims BMP prunes best with — and with CSR rejected the
"row costs bf_t entries" argument is moot: under the dense grid rare-dim
rows are mmap pages that queries rarely fault in anyway.

## Orthogonal follow-ups

The 2026 paper
[Forward Index Compression for Learned Sparse Retrieval](https://arxiv.org/html/2602.05445)
introduces DotVByte, which fuses component-gap decoding, query-value gathers,
and dot-product accumulation. This targets a document-oriented forward index,
not the block/superblock maximum grids. Hermes V15 instead stores a block-local
inverted Section B, so DotVByte is not a drop-in grid fix. Its fused-decode
approach is worth revisiting only if Section B becomes the measured bottleneck
after the grids are addressed.

[Seismic](https://arxiv.org/abs/2404.18812) and
[SeismicWave](https://arxiv.org/abs/2408.04443) avoid an exhaustive BMP grid
through statically pruned inverted lists, compact block summaries, and (for
SeismicWave) a proximity graph. They are approximate retrieval designs with
different build, update, and quality trade-offs, not lossless compression for
the current BMP executor. They should be compared as separate algorithms
rather than mixed into the V15 format.

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

## Recommendation