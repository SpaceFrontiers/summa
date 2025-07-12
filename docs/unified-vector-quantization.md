# Unifying the Dense and Binary Vector Code Paths

Status: investigation (2026-07-07). No format change proposed yet.

## The question

Binary vectors look like "dense vectors with 1-bit quantization" — why do we
keep two field types (`dense_vector` / `binary_dense_vector`), two configs, two
query types, and two search paths?

## What is already unified

Storage is one path today. Binary fields store into the same `.vectors` flat
file as dense fields, tagged `DenseVectorQuantization::Binary` — the same
`FlatVectorData` / `LazyFlatVectorData` machinery, TOC, doc-id maps, mmap
reads, and merge streaming serve both. `FieldEntry` distinguishes them only by
config type, and the segment reader keeps both in the same `flat_vectors` map.

## What is genuinely different

The split is about the **metric and the query representation**, not storage:

|                | dense (`f32/f16/u8`)          | binary                       |
| -------------- | ----------------------------- | ---------------------------- |
| Query vector   | `Vec<f32>`                    | packed `Vec<u8>` bits        |
| Metric         | cosine / dot / L2             | Hamming                      |
| Scoring kernel | FMA dot-product SIMD          | XOR + popcount SIMD          |
| ANN structure  | k-means IVF + RaBitQ/PQ codes | k-majority IVF, exact codes  |
| Rerank         | required (codes are lossy)    | none (codes ARE the vectors) |

The last row is the deep one: for binary fields the stored code is the exact
vector, so IVF scanning gives exact distances and there is no L1/L2 refinement
stage. For dense fields the quantized code is an estimate and everything
downstream (rerank_factor, extended RaBitQ bits, flat-vector reads) exists to
correct it. Collapsing the two paths would force the binary path to carry
machinery it never needs, or fork internally on "is the code exact?" — the
same split, one level down.

## Is IVF-RaBitQ applicable to binary embeddings?

Half of it. **IVF transfers** — that's what `BinaryIvfIndex` implements
(k-majority centroids instead of k-means, same probe-nprobe-clusters shape).
**RaBitQ does not**: RaBitQ's job is to _produce_ a 1-bit(+ex) code from a
float residual and estimate float inner products from it asymmetrically. For
data that is already 1 bit/dim there is no residual magnitude to encode, no
scale factors to store, and no estimation error to correct — XOR+popcount on
the raw code is both faster and exact. Running binary data through the RaBitQ
encoder would only rotate the bits and add per-vector floats that carry no
information.

The one place RaBitQ-style thinking would apply: **asymmetric float-query vs
binary-doc scoring** (e.g. an MRL float query against binarized doc
embeddings). That is a scoring-kernel addition (`dot(f32_query, unpacked_bits)`
with a correction term), not a reason to merge index structures — and it would
slot into the existing binary field as an alternative query form.

## Recommended direction (if/when unification is wanted)

Do not merge the field types. Instead extract the two seams that are
duplicated today:

1. **A `VectorCodec` trait** covering `{F32, F16, U8, Binary}` with
   `score_batch(query, raw, out)` — `score_quantized_batch` and
   `batch_hamming_scores` become impls. The brute-force and rerank loops in
   `segment/reader/mod.rs` (async + sync × dense + binary = 4 near-identical
   scan loops) collapse to one generic loop.
2. **A `CoarseIndex` trait** for IVF probing (`rank_clusters`, `scan_cluster`)
   with k-means-f32 and k-majority-binary impls, unifying the
   builder/loader/merger wiring (`try_build_ann` / `try_build_binary_ivf`).

That removes the duplication the split creates while keeping the type-level
distinction users see (a Hamming field and a cosine field genuinely are
different things to query). Effort: moderate refactor, no format change, no
user-facing change — worth doing next time either scan loop needs a third
variant.
