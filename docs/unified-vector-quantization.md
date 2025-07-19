# Unified Dense IVF Architecture

Status: implemented (2026-07-22). The serialized ANN format is version 3 and
requires rebuilding older vector artifacts.

## One global router

Float and packed-binary dense fields use the same index-wide IVF lifecycle:

- one corpus-trained coarse codebook per field and index;
- one routing implementation with `auto`, `flat`, `two_level`, and `hnsw` modes;
- HNSW coarse routing for large codebooks and exact flat routing for small ones;
- one probe plan per query, shared by every segment;
- bounded distinct-document collection rather than a heap proportional to all
  vector values or to the number of segments;
- exact flat accumulation until the explicit ANN build creates global artifacts;
- strict artifact versions, with no legacy deserialization fallback.

This removes per-segment centroid searches, per-segment codebooks, and the
former alternative production paths.

## Metric-specific leaf payloads

The global router is metric-agnostic, but leaf scanning is deliberately not:

| Field | Coarse metric | Segment payload | Leaf score | Rerank |
| --- | --- | --- | --- | --- |
| float dense | squared L2 | residual PQ bytes | asymmetric distance tables | exact, bounded to 3×k distinct docs |
| binary dense | Hamming | packed source bits | XOR + popcount | unnecessary; leaf scores are exact |

Treating an already-binary embedding as float PQ would add conversion and
approximation without information. Conversely, storing full float vectors in
every IVF leaf would lose PQ's memory and bandwidth benefit. The two scanners
therefore share routing, collection, persistence rules, and lifecycle while
retaining their optimal metric kernels.

## Billion-scale defaults

Automatic training targets 8×sqrt(N) leaves, capped by the available training
sample and a 512 MiB coarse-artifact budget. Training samples at most two
million vectors or 6 GiB. Large codebooks use hierarchical k-means for
tractable training and a compact HNSW graph (`M=32`, `efConstruction=200`);
query routing uses `efSearch=max(128, 4*nprobe)`, with `nprobe=64` by default.

Float residual PQ uses shared distance tables built once per query. Segment PQ
payloads use struct-of-arrays columns with `code_size + 6` bytes per assignment
(PQ code, document ID, and multi-value ordinal), avoiding per-vector heap
objects. Both float and binary assignment paths reuse HNSW scratch buffers.
