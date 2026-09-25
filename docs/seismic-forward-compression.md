# Seismic forward dimension compression

## Scope and invariants

[Forward Index Compression for Learned Sparse Retrieval](https://arxiv.org/pdf/2602.05445)
proposes DotVByte: sorted component gaps, one length bit per 16-bit gap,
eight gaps per control byte, independent document alignment, and fused SIMD
scoring. Its RGB permutation reorders **dimensions**, not documents.

Summa stores configured weight bytes after each vector's dimension bytes.
Adaptive dimension compression is enabled by default and reduces that same
single owner without adding another forward index. Explicitly disabling it
stores raw U32 dimensions. Weights, logical document/ordinal identity, missing vectors,
nomination, and scoring semantics remain unchanged. Dimension IDs retain the
full U32 range, including 100k vocabularies. Compatible runs remain byte-copyable
without training or coordinate translation.

## Default format

`seismic_forward_compression` defaults to `true` for newly built Seismic rows,
using adaptive U16/U24/DotVByte storage. Set it to `false` for raw U32 dimensions.
The default was enabled at the user's request after the comparison below; the
measured CPU/storage tradeoffs and architecture limits still apply. Existing
precision settings are unchanged.

The 24-byte row directory uses byte 6 for dimension encoding: 0 = raw U32,
1 = raw U16, 2 = DotVByte, 3 = raw U24; byte 7 remains reserved. DotVByte consists of
a U32 base (the first dimension), `floor(nnz / 8)` control bytes, packed
one/two-byte little-endian gaps (first gap zero), then up to seven absolute
U32 tail IDs. Prefix sums are U32, so crossing 65535 does not truncate IDs.
Weight bytes follow, with their size determined by nnz and existing precision.
The writer chooses DotVByte only when every encoded gap fits U16 and the
result is smaller than raw IDs (U16 below 65536, U24 below 16777216, otherwise
U32). Each row is independently decodable. Empty rows retain encoding 0. Uncompressible gaps use the smallest raw width
that represents every ID.

Components use envelope version 6; older versions require rebuilding.
Copy merges preserve encoded runs, including mixtures of raw and compressed
version-6 sources. There is no legacy reader path. Compaction copies surviving
forward bytes and their encoding tags. No payload-validation pass is added.
The existing envelope and row ownership admission remain outside query loops.

The decoder uses bounded eight-U32-coordinate scratch and streams into the existing
scorer and retrieval iterator. It must not allocate a full decoded vector per
candidate or fork scoring semantics. ARM NEON and x86 SSSE3 shuffle eight gaps
and accumulate U32 prefix sums.
Other targets use a portable decoder. Decoded groups feed the existing scorer
through its fold closure; packed SIMD weight multiplication/gather is not added.
This preserves its lookup/sorted-query paths, duplicate-query arithmetic, and
all four weight precisions without introducing a second scorer.

## Evaluation

Use the existing SPLADE 1M fixture and identical compiler/machine/flags for
comparisons. Measure actual dimension, forward, and total bytes; candidate
scoring/query latency; peak memory; and exact/approximate result equality.
Compare raw U32, raw U24, and DotVByte timings, and additionally measure raw U16
size on the narrow fixture, to separate narrowing from gap compression. Writer/codec tests cover every control byte, tails, zero, signed
weights, all precisions, large dimensions, partial iterator consumption, and
encoded-byte preservation through merge/compaction.

RGB dimension permutation is deferred: it requires an index-global artifact,
query/lookup translation, and a build-time memory budget. Existing document
reordering is not a substitute. Additional weight quantization is not part of
this lossless change. BMP's forward values serve hydration/reordering, while
normal BMP scoring reads inverted data; this experiment targets Seismic's
candidate-scoring owner first.

The existing query lookup allocation stays capped at 65536 entries. Queries
with larger IDs use the existing sorted merge scorer; this change does not
increase per-query lookup residency. A 100k vocabulary requires 17 bits per ID;
U24 is the byte-aligned fallback, saving 25% of dimension bytes relative to U32.
With Float16 weights, that is 5 rather than 6 bytes per coordinate (16.7% less
forward payload), before row-directory and index overhead.

## Implementation evidence

The first prototype outlined `Dimensions::fold` despite the outer precision
specialization. Linux assembly showed a call to the generic weight decoder for
every coordinate; raw exhaustive scans regressed from about 275 to 440 ms on
five queries. The fold boundary now explicitly inlines so the caller's constant
weight precision remains visible inside the loop. Dimensions and weights also
share one coordinate cursor through `fold_indexed`, rather than advancing two
independent counters. U16/U24 folds use exact byte chunks. The shared iterator
also has a compile-time raw specialization selected at the row-scoring boundary,
so raw rows do not carry compressed-format dispatch into their scoring loop.
The dimension reader, weight codec, and scorer remain single implementations.
Final measurements below supersede the prototype.

Native verification covers all control bytes, exact input extents, SIMD/scalar
identity, U16/U24/U32 boundaries, bases above 65535 (including near U32 maximum),
partial iterator consumption, all four weight precisions, signed and repeated
query terms, and byte-preserving merge/compaction. The wide-ID WASM regression
exercises groups, tails, U24, and missing matches through public search APIs.

## Storage on the 1M fixture

The fixture contains 1,000,000 vectors and 126,320,094 stored coordinates with
Float32 weights. These byte counts include each vector's controls/base/tails;
weights and persistent directory size are unchanged.

| Layout                             | Dimension bytes | Forward payload bytes (IDs + weights) |
| ---------------------------------- | --------------: | ------------------------------------: |
| Raw U32                            |     505,280,376 |                         1,010,560,752 |
| Raw U16 (narrow vocabularies only) |     252,640,188 |                           757,920,564 |
| Raw U24                            |     378,960,282 |                           884,240,658 |
| Adaptive gaps                      |     178,040,573 |                           683,320,949 |

Adaptive encoding saves 64.8% of dimension bytes, 32.4% of Float32 forward
payload, and 327,239,803 bytes across the index. Nomination payloads do not
shrink in this change. This is storage savings, not a minimum-RAM guarantee.

## Measured query tradeoff

The [recorded comparison](benchmark-results/seismic-forward/2026-09-19/README.md)
uses one Linux x86 machine, identical compiler/flags/lockfiles, and unchanged
weights, nomination payloads, candidate settings, and query results. The 1M
Float32 fixture's complete index shrinks from 2,728,494,428 to 2,401,254,625 bytes
with adaptive gaps (12.0%).

| Candidate layout | Warm top-100 mean | Warm peak RSS | 1 GiB limit mean |
| ---------------- | ----------------: | ------------: | ---------------: |
| Raw U32          |          27.99 ms |      2.00 GiB |        984.41 ms |
| Forced U24       |          31.50 ms |      1.89 GiB |         35.77 ms |
| Adaptive gaps    |          32.79 ms |      1.70 GiB |         33.97 ms |

Warm results average three balanced trials of 200 queries. The original main
binary averages 29.85 ms; the raw specialization removes the prototype's
regression. Relative to the candidate's raw rows, adaptive rows cost 17.1% more
warm ANN time and 45.8% more exhaustive time (378.16 versus 259.31 ms).
The user subsequently selected compression as the default; raw storage remains
available with `seismic_forward_compression: false`.

The memory-limit probe repeats 20 queries, with two independent cold-start
trials per format in reverse order, swap disabled. Adaptive gaps reduce total
physical reads from about 2.44 to 0.77 GiB and let this working set fit in cache;
raw rows repeatedly refault. Latency excludes the first pass, whereas the read
counters include it. This threshold effect is not a general speedup or a
minimum-RAM estimate for billions of vectors.

The shifted 100k-vector wide-ID fixture preserves exact/approximate results;
warm means are 10.23 ms raw, 10.46 ms U24, and 10.72 ms adaptive. Its gap
distribution comes from the narrow model. Representative 100k-token embeddings,
concurrent load, relevance recall, and ARM latency remain unmeasured. No RGB
permutation or new packed weight scorer is included in these numbers.
