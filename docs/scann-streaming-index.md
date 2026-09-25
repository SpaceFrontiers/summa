# Streaming ScaNN index

## Status

This document defines the on-disk and lifecycle contract for adding ScaNN to
Summa. It covers floating-point dense vectors and packed binary embeddings.
The implementation must preserve Summa' immutable-segment model: training is
an index-generation operation, while ordinary commits and merges only assign,
encode, and move rows within an already-published generation.

## Goals

- Declare ScaNN in SDL/JSON schemas and expose every persistent build option.
- Change a field between `ivf_tq` and `scann` through an atomic alter operation
  without serving a mixed or partially rebuilt generation.
- Train once from an index-wide sample and share the resulting routing tree and
  asymmetric-hashing codebook across every segment in the generation.
- Keep commits streaming: after training, a new segment routes and encodes its
  rows against the published model without changing that model.
- Keep normal merges streaming: merge matching leaf runs by byte copy plus
  document-ID rebasing, with no vector decode, model training, or code rewrite.
- Support an explicit routing depth as well as an automatic depth suitable for
  approximately one billion vectors.
- Support packed binary embeddings with native Hamming routing and exact
  Hamming leaf scoring; do not expand them to floats or run float AH over bits.

## Non-goals

- Online mutation of a published trained model.
- Mixing segment payloads produced by different model fingerprints.
- Silently falling back from an incompatible built payload. An accumulating
  field may search flat vectors, but corrupt or mixed generations fail loudly.

## Schema

`VectorIndexType::Scann` is a float dense-vector index. The corresponding SDL
form is:

```sdl
field embedding: dense_vector<1024, f16> [
  indexed<scann, target_vectors: 1000000000, tree_levels: 2, nprobe: 1024>
]
```

All fields are optional except the vector dimension. `num_clusters` is the
terminal leaf count for ScaNN; reusing the existing persisted field avoids two
names for the same invariant. `num_clusters` and `tree_levels` default to an
autopilot derived from corpus size and dimension. `target_vectors` is an
optional steady-state sizing hint for that autopilot. Automatic geometry uses
`max(live_vectors, target_vectors)`, so the hint can select a stable future
topology early but can never shrink geometry below the live corpus. An explicit
`num_clusters` takes precedence; the hint remains persisted but dormant until
automatic leaf sizing is restored. `target_vectors` must be positive and is
rejected for Flat and training-free TQ fields. Explicit values are validated
together. The implementation supports at most three centroid levels and 30
million leaves. Corpus-only sizing follows AlloyDB's recall-oriented balanced
guidance: `sqrt(rows)` below 100 million rows, `rows^(2/3)` through one billion,
and `min(30 million, rows^(3/4))` above one billion. One billion vectors
therefore resolve to 1,000,000 leaves in two centroid levels (a three-level
index in AlloyDB's root-inclusive terminology). The higher-quality `rows / 100`
geometry remains available explicitly when its measured recall benefit
justifies the additional training, routing, and artifact cost.

Binary fields gain a `scann` index type with the same routing/lifecycle fields:

```sdl
field hash: binary_dense_vector<1024> [
  indexed<scann, target_vectors: 1000000000, tree_levels: 3, nprobe: 1024,
          soar: selective>
]
```

Float-only AH options are rejected on binary fields at schema load time.
Binary `soar: selective` uses a strict 30% secondary-posting budget. It keeps
the vectors with the largest primary-centroid Hamming residuals and assigns
each to its nearest alternate among a bounded widened tree probe. Search
deduplicates `(document, ordinal)` results. This is SOAR-like selective
spilling, not the float algorithm's orthogonal-residual objective: packed bits
do not have a continuous residual geometry.

## Generation state

Each vector field has one published generation. Its metadata records:

- configured algorithm and persistent build parameters;
- lifecycle state: `accumulating`, `building`, or `built`;
- source and target configuration for a staged alter;
- model artifact path, format version, and content fingerprint;
- vector count and sample count used for training;
- leaf count and routing level counts.

The model artifact contains the complete resident routing plane and, for float
ScaNN, the global asymmetric-hashing codebook. Every segment payload header
stores the model fingerprint. Readers and mergers require an exact match.

At billion scale this artifact cannot use the existing 512 MiB bincode-owned
artifact path. Ten million 1024-dimensional fixed-point leaf centroids alone
are approximately 10 GB. The ScaNN tree therefore has its own checked,
versioned, zero-copy format and is opened from mmap/file slices. All offsets,
lengths, level counts, and fingerprints are validated before any region is
exposed. Configuration reports the estimated artifact and training-sample
memory before a build begins. Automatic and explicit geometry are never
silently changed by a transient training resource ceiling. A build fails or
defers with the required and available resource counts when its limits are
insufficient.

Training eligibility and sample size are derived from the selected geometry,
not configurable schema knobs. Partitioning starts at 100,000 live vectors.
The desired routing sample is `max(100,000, leaves * 200)`, bounded by the live
corpus and builder memory; the hardcoded viability floor is eight sampled rows
per leaf. This minimum is internal and is not a schema parameter. If the corpus
has not reached the partitioning floor, the field remains
in accumulating flat-search mode. If memory cannot hold the hard floor, the
build fails loudly with required and available byte counts. Commits never start
an expensive training job implicitly.

`target_vectors` affects topology selection only: hinted vectors are never
treated as observed rows or synthetic training samples. A field whose live
corpus or builder budget is below the selected geometry's hardcoded viability
floor remains in accumulating flat-search mode (or reports the insufficient
builder limit); reaching the exact floor makes it eligible to build.

The depth-band transitions intentionally raise the training floor: 100 million
rows select 215,444 leaves and require 1,723,552 samples; the first row above
one billion selects 5,623,414 leaves and requires 44,987,312 samples. Resource
limits must be raised before either topology can be published if the configured
builder ceiling is lower.

## Commit and merge

After a model is published, commit performs a bounded streaming transform:

1. Route each row through the shared tree.
2. Emit primary and optional SOAR leaf assignments.
3. For float vectors, encode centroid residuals with the shared AH codebook and
   write the exact rerank representation.
4. For binary vectors, retain the original packed code and route/score with
   Hamming distance.
5. Write leaf runs ordered by leaf ID, with locators and document ordinals.

Normal merge never retrains. Binary ANN payloads and exact-vector lookup rows
copy verbatim; only their run/span/block directories receive new offsets and
document bases. Source extents remain separate. Float AH rows are streamed through
the FastScan packer when compaction joins run-relative 32-row blocks. Neither path
reassigns leaves or changes the global model. Binary exact retrieval shares the
ANN codes by default; see [exact binary storage](binary-vector-storage.md).

## Alter operation

`alter_vector_index(field, new_config)` stages a target configuration, then
builds a complete replacement generation from the authoritative exact-vector view. It
publishes schema, model metadata, and replacement segment references in one
metadata rename. Readers continue using the old generation until that point.
On failure, the old generation remains readable and staged artifacts are
garbage-collectable.

Changing only query-time parameters such as `nprobe` updates schema metadata
without retraining. Changes to the tree, leaves, codebook, metric, storage
representation, or algorithm require a replacement generation. The supported
algorithm replacement is IVF-TQ to ScaNN or ScaNN to IVF-TQ; Flat and TQ are
accumulation/search formats, not alter targets.

Changing `target_vectors` requires a replacement generation when automatic
leaf sizing is active, because it can change the routing topology. When
`num_clusters` is explicit and unchanged, a target-only ALTER is metadata-only:
the hint is dormant and does not affect the built layout. Removing the explicit
leaf count activates the persisted hint and therefore requires a rebuild. If
the newly selected automatic IVF or ScaNN geometry is not yet viable from live
data, the ALTER remains deferred in flat accumulation until enough real vectors
are available.

## Binary ScaNN

ScaNN's partitioning idea applies to binary embeddings, but float residual AH
does not. The binary path uses medoid/majority-bit centroids, Hamming assignment,
hierarchical beam routing, optional multi-assignment, and exact XOR-popcount
inside selected leaves. It shares the generation and merge mechanics with
float ScaNN while using a distinct artifact kind and fingerprint domain.

The hot loop must dispatch safely to AVX2/AVX-512 popcount where available,
AArch64 NEON on ARM, and a scalar fallback. Recall is controlled by routing and
leaf probes; there is no lossy leaf codec once the packed bits are stored.

## Binary leaf scan pruning

Binary IVF and binary ScaNN share the exact Hamming scanner and the existing
bounded ANN collectors. The scanner checks the collector's
conservative score threshold before reading document IDs, visibility, and
ordinals. It still scores every code in each probed leaf: the probe budget,
recall, and persisted run bytes do not change. Thresholds are refreshed in
small fixed-size blocks, so a stale threshold can only admit extra candidates.
Equal scores must reach the collector to preserve document/ordinal tie-breaking.

The unbounded ordinal sink used before Sum, Avg, LogSumExp, and WeightedTopK
combination has no pruning threshold. A low individual score can still change
the combined document score. Max and distinct-vector top-k may reuse their
existing collector thresholds. The expected saving is fewer metadata loads and
heap comparisons, with no additional scratch, cache, or format.

For 256-bit codes, the same row kernel also receives a compile-time width,
letting LLVM unroll it without duplicating the Hamming implementation. Runtime
CPU checks and scalar fallback remain unchanged. See the
[performance review](search-performance-review.md) for ARM/x86 measurements and
[exact binary storage](binary-vector-storage.md) for the separate proposed
removal of duplicate flat codes.

## Verification

- Schema round trips, unknown-option rejection, invalid geometry, and binary
  float-option rejection.
- Deferred training below the derived geometry floor and successful build once
  the required routing sample is available.
- New commits reuse the exact model fingerprint.
- Merge output returns identical top-k results and never calls a trainer.
- Generation mismatch makes both open and merge fail with an actionable error.
- Crash tests around staged artifact write and metadata publication.
- IVF-TQ to ScaNN and ScaNN to IVF-TQ transitions.
- Float recall/latency/indexing tests against the existing IVF-TQ baseline and
  the Google ScaNN benchmark methodology.
- Binary recall against exact Hamming ground truth, with indexing, merge, and
  query throughput measured separately on x86_64 and AArch64.
