# BMP LSP/0 and Maximum-Grid Compression

Status: implemented in the BMP V18 format (2026-07-23).

The production field shape is:

```sdl
field embedding: sparse_vector<u32> [indexed<
    format: bmp,
    dims: 105879,
    max_weight: 5.0,
    bmp_block_size: 32,
    bmp_grid_bits: 4
>]
```

V18 is the only supported BMP representation. There is no compatibility
reader or migration path; rebuild the index after upgrading.

## Space anatomy

Let `n` be stored vectors/ordinals, `d` the vocabulary size, `b = 32` vectors
per block, and `c = 8` blocks per superblock. Multi-valued fields count every
ordinal, not only logical documents.

Before local bit packing, the three maximum grids would occupy:

```text
blocks      = ceil(n / b)
superblocks = ceil(blocks / c)
coarse      = ceil(superblocks / 256)
D bytes     = d × ceil(blocks / 2)       # ceil-u4 block maxima
E bytes     = d × ceil(superblocks / 2)  # ceil-u4 superblock maxima
H bytes     = d × ceil(coarse / 2)       # ceil-u4 256-superblock maxima
```

For `d = 105,879`:

| vectors | blocks | superblocks | coarse | dense D | dense E | dense H | dense total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 18M | 562,500 | 70,313 | 275 | 29.78 GB | 3.72 GB | 14.61 MB | **33.52 GB** |
| 100M | 3,125,000 | 390,625 | 1,526 | 165.44 GB | 20.68 GB | 80.79 MB | **186.20 GB** |
| 1B | 31,250,000 | 3,906,250 | 15,259 | 1.654 TB | 206.79 GB | 807.86 MB | **1.862 TB** |

Those are dense-reference sizes, not the V18 on-disk sizes. V18 omits
all-zero payload groups and uses each group’s actual local width. The encoded
size depends on term density and BP ordering, so projections captured for the
old 64-block/exact-u8 hierarchy are not valid for V18. The segment load log’s
separate D/E/H byte counts are authoritative after a rebuild.

The other persistent sections are:

```text
T = total non-empty (block, dimension) pairs
P = total document-term postings
B = total blocks
N = padded vector/ordinal count

Section B, Flat-Inv payload = 8 × non-empty_blocks + 9T + 2P
Section A, block offsets    = 8 × (B + 1)
Sections F+G, document map  = 6N
```

The `9T` term-header component is four bytes for the dimension, four bytes for
its posting start, and one byte for its exact block maximum. The final posting
start contributes the other four bytes per non-empty block. Retaining the
exact maximum costs `T` bytes, but lets a D2 BP/reorder regenerate a tight
ceil-u4 E grid without rereading `2P` posting bytes. Empty blocks have no
Section B payload.

## V18 grid representation

D, E, and H are independent dimension-major grids partitioned into 256-cell
codec groups. H has one logical cell per 256 E superblocks. Each exact
block-header maximum is ceiling-quantized before projection:

```text
u4 = ceil(exact_u8 / 17)
u2 = ceil(exact_u8 / 85)  # D only, when bmp_grid_bits = 2
```

E and H always use ceil-u4. Taking the maximum of ceiling-quantized values is
equivalent to ceiling-quantizing their exact maximum. Query-time multiplication
by 17 (or 85 for u2 D) makes every decoded cell an upper bound, never a rounded
down estimate.

A 256-cell group is encoded at its smallest sufficient bit width:

- D has local width `0..=bmp_grid_bits`.
- E and H have local width `0..=4`.
- An all-zero group has width zero and no payload.
- A group payload is exactly `width × 32` bytes.

Rows use checkpointed selectors:

```text
row_offsets: u64[dims + 1]

for each row:
    repeated every 32 groups:
        payload_checkpoint: u32  # offset in 32-byte units
        widths: packed-u4[1..=32]
    group payloads
```

The final selector record stores only live nibbles. One checkpoint plus at
most 31 width additions locates any group without walking the row. For
`C` cells and `G = ceil(C / 256)` groups:

```text
row_header  = ceil(G / 2) + 4 × ceil(G / 32)
row_payload = 32 × sum(local_group_widths)
section     = 8 × (dims + 1) + sum(row_header + row_payload)
```

Codec groups do not pad the logical BMP block or superblock counts.

## LSP/0 execution

V18 implements LSP/0 as a query-level operation:

1. Use all bounded query dimensions for candidate generation by default.
   An explicit `query<pruning: beta>` can retain only the strongest fraction,
   but is an approximation whose Recall@K must be measured.
2. Sweep the small H grid and push its coarse upper bounds into one global
   best-first frontier.
3. Expand only H cells whose bound can still enter global top-`gamma`; each
   expansion reads the corresponding independently addressable 256-cell group
   from E. Stop only when the next H bound is strictly below the current
   `gamma`-th E bound, preserving score ties exactly.
4. Partition the exact global top-`gamma` E cells back to their segments.
5. Visit selected superblocks in non-increasing SBMax order while
   `SBMax >= theta`.
6. Within a selected superblock, decode D once per candidate dimension, order
   its eight blocks by BlockMax, and apply `heap_factor` only at this level.
7. Score documents in visited blocks with the bounded full query, including
   dimensions removed from candidate generation.

The global selection is important: 50 immutable segments still visit at most
`gamma` superblocks in total, not `50 × gamma`.

Automatic gamma follows the paper’s zero-shot depths:

| requested candidate depth | gamma |
| ---: | ---: |
| `1..=10` | 250 |
| `11..=100` | 500 |
| `101..=1000` | 1,000 |
| `> 1000` | `max(2000, depth)` |

Set query `lsp_gamma: 0` for exhaustive traversal, or a positive value for an
explicit cap. `heap_factor: 1.0` and ceiling bounds are rank-safe when query
pruning is disabled; finite gamma and beta below 1.0 are intentional
candidate-generation approximations. Full-query scoring preserves relevance
within the visited candidate set.

Candidate bounds and their corresponding score contributions share the same
integer arithmetic: query-u16 times ceiling maximum accumulates in u32 and is
converted to f32 once with the document scorer’s dequantizer. This prevents a
floating-point rounding difference from lowering an unpruned bound.

## Build, merge, and BP reorder

Initial build and BP/reorder write grids from sorted
`(dimension, block, exact_u8_maximum)` entries. Exact maxima are retained in
each block header, so D, E, and H can be regenerated without scanning or
reserializing postings.

Ordinary merge remains a streaming block-copy operation:

- Section B block payloads are copied byte-for-byte.
- Aligned D groups are copied byte-for-byte.
- A shifted D boundary is decoded/repacked with one bounded 256-cell buffer.
- E values are remapped to destination superblocks. If a source superblock
  straddles a destination boundary, its ceiling bound is max-propagated to
  both destinations. The bound may be temporarily loose but cannot be unsafe.
- H is streamed from the remapped E row by reducing each 256-superblock group;
  postings are never decoded or reserialized.
- Document maps are chunked copies with document-ID offset patching.

A later BP pass rebuilds tight E/H values from exact block headers. Reordering
changes block coordinates first and then projects exact maxima into
`floor(new_block / 8)`, so ceil quantization composes correctly with any
permutation.

## CPU and paging behavior

The production codec and hot loops are Rust:

- x86_64 uses BMI2 for variable-width unpacking and SSE4.1 for u4/u8 bound
  accumulation, including the eight-block LSP unit;
- AArch64 uses bounded unpacking and NEON for the same D/E/H accumulation paths;
- other targets use a result-identical scalar fallback.

All large sections remain mmap-backed. H is approximately 256 times smaller
than dense E and is pinned with its row offsets because every BMP query sweeps
it. Only E's small row-offset table is pinned; selected E payload groups, D,
and block payload remain pageable with random-access advice.

## Match to the LSP paper

V18 follows the core design in
[Carlson et al., “Efficiency Optimizations for Superblock-based Sparse Retrieval”](https://arxiv.org/html/2602.02883v1):

- 256 independently decodable maximum cells per compressed group;
- random group access rather than BMP-Sparse’s binary-search/scatter path;
- four-bit ceiling maxima at both hierarchy levels;
- a superblock footprint no larger than 256 vectors (`b = 32`, `c = 8`);
- optional beta query pruning, strict SBMax top-gamma selection, safe
  `SBMax >= theta`, block-level eta/`heap_factor`, and full-query scoring of
  visited candidates.

Hermes’s codec is equivalent in access granularity and bounds but is not the
paper implementation’s wire format. Its selector checkpoints are interleaved
every 32 groups rather than stored as one long selector prefix, which bounds
random lookup work on very long 1B-scale rows.

The persisted H level is a Hermes exact-search acceleration around LSP/0, not
a new approximation: it is a maximum hierarchy over E and produces the same
global top-`gamma` membership as a complete E sweep.

The paper reports its best latency around block sizes 4–16. Hermes keeps the
requested block size 32 and compensates with eight-block superblocks; this
satisfies the paper’s `b × c <= 256` recommendation but is a deliberate
space/build-time tradeoff rather than a claim that 32 is the paper optimum.

## Flat-Inv versus the paper’s Fwd layout

Hermes does **not** use the paper’s document-major Fwd payload for persistent
BMP scoring. Section B is a block-local Flat-Inv layout:

```text
sorted term IDs
term offsets
(local_slot: u8, impact: u8) postings
```

The paper finds Fwd faster for small blocks, including `b = 32`, but its
Compact-Inv format explicitly assumes a vocabulary that fits two-byte term
IDs. Hermes production fields use roughly 105,879 dimensions, so a direct
Hermes Fwd layout requires four-byte IDs. At minimum, separate u32 term IDs
and u8 weights cost `5P` bytes plus document offsets, while current Section B
costs `2P + 9T + block headers`. Which representation is smaller therefore
depends on the measured `T/P` ratio; vocabulary size alone does not settle it.
Ignoring the small offset/header terms, Fwd becomes smaller only when
`T/P > 1/3`; repeated dimensions within a 32-vector block push that ratio
down and favor Flat-Inv.
Flat-Inv also lets the scorer fetch only postings for query-present terms,
whereas Fwd streams every term in every candidate document.

“Forward indexes” elsewhere in Hermes are temporary BP/reorder structures
used to compute a document or block permutation. They are memory-budgeted,
discarded after the pass, and are not the on-disk query payload discussed by
the paper.

## Validation

The test suite pins:

- every local width round trip and malformed metadata rejection;
- SSE4.1/NEON/scalar-equivalent u4 and u8 accumulation tails;
- exact BMP-versus-MaxScore top-k parity in exhaustive mode;
- u2-versus-u4 rank-safe parity;
- a single global gamma across multiple segments;
- non-aligned D merge and overlapping E/H-bound propagation;
- record and blockwise BP reorder;
- E regeneration from exact block-header maxima;
- bounded external-run construction.