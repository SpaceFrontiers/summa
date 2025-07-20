# Seismic: research assessment (2026-07-09)

Sources: Bruch, Nardini, Rulli, Venturini — "Efficient Inverted Indexes for
Approximate Retrieval over Learned Sparse Representations" (SIGIR 2024,
arXiv:2404.18812); "Pairing Clustered Inverted Indexes with kNN Graphs..."
(SeismicWave, arXiv:2408.04443); "Efficiency Optimizations for
Superblock-based Sparse Retrieval" (arXiv:2602.02883, compares SP vs
Seismic/SeismicWave).

## What Seismic is

An _approximate_ sparse index that abandons the document-space grid entirely:

- **Static pruning**: keep only the top-λ highest-impact postings per
  inverted list (λ = 6,000 on MS MARCO 8.8M — lists are capped, not
  proportional to corpus).
- **Geometric blocking**: cluster each list's postings into β blocks via
  shallow k-means over the document vectors (β = 400), so a block groups
  _similar documents within one list_ — not consecutive doc ids.
- **Per-block summaries**: coordinate-wise max vector, cropped to its α-mass
  (α = 0.4) and quantized to u8. Query processing walks the top-`cut` query
  coordinates, scores summaries, and only fetches exact vectors (forward
  index) for blocks whose summary beats `Heap.min()/heap_factor`.
- **SeismicWave** adds importance-ordered block traversal and a k-regular
  kNN-graph expansion of the result list: near-exact accuracy, up to 2.2×
  faster than Seismic.

## Headline numbers (MS MARCO, SPLADE, single thread)

- 187–531 µs/query at 90–97% accuracy — 2.6–3.7× faster than the best
  graph method (GrassRMA), 22–37× faster than prior inverted approaches,
  ~85–105× faster than impact-sorted indexes.
- Index ~6.3 GiB (vs 10.2 GiB graph); builds in ~5 min on 64 cores.
- Caveat from the SP follow-up (arXiv:2602.02883): rank-safe SP/BMP beats
  Seismic at k=1000 in some regimes, the kNN graph can _hurt_ at large k,
  and Seismic needs exhaustive grid search over (λ, β, α, cut, heap_factor)
  to hit its numbers.

## Why this matters for Hermes at 1B scale

BMP's maximum grids are O(dims × vectors / block_size) before V17's local
bit packing. At 1B vectors, `dims=105879`, `b=32`, and eight blocks per
superblock, the dense D+E reference is 1.861 TB; actual V17 size is
data-dependent because zero groups disappear and every 256-cell group uses
its local width. Seismic's memory is O(kept postings): λ caps every list, so
the index grows with _vocabulary_, not corpus × dims. That is a structurally
attractive shape for 1B vectors. The trade: approximate-only (no rank-safe
mode), parameter-sensitive, and per-list clustering makes incremental
merging awkward (blocks must be re-clustered per list — a rebuild-style
operation like our reorder, not a byte-stack merge).

## Fit with existing Hermes machinery

Most building blocks already exist:

| Seismic component               | Hermes equivalent                                         |
| ------------------------------- | --------------------------------------------------------- |
| top-λ list pruning              | `pruning` (per-list fraction) — needs absolute-λ variant  |
| α-mass summaries                | `doc_mass` logic (same cropping, applied to a max-vector) |
| u8 quantization                 | `WeightQuantization::UInt8` + max_weight scale            |
| shallow k-means                 | shared deterministic k-means++/Lloyd trainer             |
| forward index for exact scoring | BMP has the same values in block-local Flat-Inv form; a Seismic scorer would need a persistent document-major Fwd layout or an equivalent exact-vector store |
| heap_factor pruning             | same knob/semantics as BMP/MaxScore                       |
| kNN graph (Wave)                | could reuse the global dense IVF HNSW topology            |

Sketch: a third `SparseFormat::Seismic` with per-list blocked-cluster layout

- summary section; merge = concat lists then recluster (or block-stack with
  summary rebuild as the cheap path); query executor mirrors the paper's
  coordinate-at-a-time loop. Estimated scope: comparable to the BMP
  implementation (format + builder + merger + executor + tests).

## Recommendation

Not a replacement for BMP today — BMP + superblocks + BP reorder supports an
exhaustive rank-safe mode, remains merge-friendly, and is well-instrumented.
Seismic becomes the right tool when (a) corpus per segment pushes the grid
past what local compression and the chosen block size can absorb, or (b)
strictly approximate retrieval with sub-ms budgets is acceptable
product-wide.
Suggested path: prototype behind `format: seismic` on one production-shaped
segment, compare against BMP at equal recall using the new
`hermes_bmp_*`/latency metrics before committing to the format.
