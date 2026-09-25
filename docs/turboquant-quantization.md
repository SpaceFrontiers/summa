# TurboQuant (TQ) — training-free dense ANN codec

Status: implemented (v1 `AnnKind::TqFlat`, v1.1 adds `AnnKind::IvfTq`).
v1.2 removes IVF-PQ entirely: IVF-TQ beat it on every measured axis (see
Benchmark), so residual-PQ/OPQ, its trained codebooks, and `AnnKind 1` /
TOC type 2 are retired. Legacy `ivf_pq` schemas, trained generations, and
payloads fail loudly with a recreate-as-`ivf_tq` message; the discriminants
are reserved and never reused.

## Motivation

IVF-PQ requires corpus-trained global artifacts (coarse centroids + OPQ-PQ
codebook). Until `build_vector_index()` runs, dense queries brute-force the
flat vectors; every trained generation adds version gates, and OPQ training is
disabled on WASM. TurboQuant (Zandieh, Daliri et al., arXiv 2504.19874; see
also mayflower/pg_turboquant, MIT) is a **data-oblivious** quantizer: its
codebook is derived analytically from the geometry of the unit sphere, so a
segment can carry a compressed ANN payload from its very first build with
**zero training** and no cross-segment compatibility surface beyond fixed
constants.

`index_type: tq` is a per-segment compressed _flat_ scan: every code is
scored (no IVF pruning), so candidate recall is limited only by estimator
error at the `fetch_k` boundary, and the existing exact re-rank restores
full-precision ordering. It replaces the brute-force fallback lane, not
IVF-PQ: at large N, IVF-PQ scans `nprobe/num_clusters` of the corpus while TQ
scans all of it (at 1/8 the bytes of f32).

## Codec (bits = 4 per padded dimension, fixed)

Encode, per vector `x` (always the L2-normalized vector; cosine and unit-norm
dot coincide after normalization, exact re-rank applies the field's metric):

1. Apply the seeded rotation `R1`. Since codec v2 this is padding-free:
   three rounds of {sign flips → normalized FWHT over the largest
   power-of-two prefix → full random permutation}, each factor orthonormal
   on `R^P` with `P = dim` rounded up to even. (v1 zero-padded to the next
   power of two, inflating 768-dim codes by 33%.)
2. Stage 1: per-coordinate 3-bit code into the analytic codebook `C[8]` —
   Lloyd-Max levels for the marginal density of a unit-sphere coordinate in
   `R^P`, `f(t) ∝ (1 − t²)^((P−3)/2)` on `[−1, 1]`. Depends only on `P`.
3. Residual `r = R1·x − C[codes]`, `gamma = ‖r‖₂` (stored f32).
4. QJL sketch: 1 sign bit per coordinate of `R2·r` (`R2` an independent
   seeded rotation).
5. Nibble per coordinate: `(code << 1) | sign` → `P/2` bytes per vector.

Score estimate for query `q` (normalized, rotated once per query):

```
q1 = R1·q,  q2 = R2·q1
est(q, x) = Σᵢ q1[i]·C[code[i]]  +  gamma · sqrt(π/2)/sqrt(P) · Σᵢ ±q2[i]
```

The second term is the QJL unbiased correction of the inner product lost to
stage-1 quantization (sign `±` from the stored bit). Unbiasedness is pinned by
a statistical test, not assumed.

## Segment payload and scoring

TQ reuses the run-based ANN container (`segment/ann_disk.rs`) with
`AnnKind::TqFlat = 3`, TOC discriminant `TQ_FLAT_TYPE = 7`, one logical
cluster (`num_clusters = 1`, routing `Flat`). Runs keep explicit doc-ID and
ordinal columns, so the normal merge stays a byte-for-byte column copy.

The codes column is laid out for FastScan-style LUT16 scoring (pg_turboquant /
Faiss FastScan pattern): groups of 16 vectors ("lanes") per block:

```
block = [gamma: 16 × f32] [nibbles: P dims × 8 bytes]
```

Nibbles are dimension-major; for dim `i`, byte `j` holds lane `j` (low nibble)
and lane `j + 8` (high nibble). The final block of a run is zero-padded;
padded lanes are never read back (lane index ≥ run count).

Query time builds two 16-entry LUTs per dimension (stage-1 term and QJL term,
the latter with the `sqrt(π/2)/sqrt(P)` constant folded in), quantized to i8
with one global scale each. Kernels score 16 lanes per dimension with in-
register table lookups (`vqtbl1q_u8` on NEON, `_mm_shuffle_epi8` on
x86_64/SSSE3+), accumulate i16 in chunks of ≤ 128 dims, widen to i32, and
finish per lane as `base_sum·s_base + gamma·qjl_sum·s_qjl`. A scalar fallback
(f32 LUTs, no i8 quantization) ships alongside and is the WASM path; a test
pins SIMD ≈ scalar agreement.

`code_size` in the ANN header is the logical `P/2` bytes/vector; the container
validates the block-padded column length for `TqFlat` specifically.

## Compatibility and gates

There are no trained artifacts. `quantizer_version` carries a deterministic
FNV-1a fingerprint of `(codec version, bits, dim, P, R1/R2 seeds)`;
`codebook_version = 0`. Merge (`headers_compatible`) and query-time validation
compare that fingerprint, so any future change to seeds, bit width, or level
derivation bumps `TQ_CODEC_VERSION` and refuses to mix loudly. Merge under
`AnnWriteMode::Rebuild` (vector-generation rewrites) still byte-copies TQ
payloads — they are generation-independent.

## Schema surface

```sdl
field embedding: dense_vector<768> [indexed<tq>]
```

`VectorIndexType::Tq`. `nprobe` is meaningless for TQ (no probing) and is
warned about at parse time; `soar`/`num_clusters` likewise. `rerank_factor`
works unchanged. `build_vector_index()` skips TQ fields (nothing to train).

## Cost model (768-dim example)

| lane     | bytes/vector                    | trained artifacts        |
| -------- | ------------------------------- | ------------------------ |
| flat f32 | 3072                            | none                     |
| IVF-PQ   | 96 codes + assignment           | centroids + OPQ codebook |
| TQ       | 384 nibbles + 4 gamma (P = 768) | none                     |

Since codec v2 non-power-of-two dims pay no padding: a 768-dim vector costs
384 nibble bytes + 4 (gamma).

## Benchmark

`summa-core/src/index/tests/tq_bench.rs` (ignored test; run in release):
clustered synthetic unit-norm corpus, queries perturbed from corpus points,
ground truth = the flat index's exact cosine top-10.

Measured 2026-07-23, aarch64 (Apple Silicon, NEON kernels), 100k docs ×
768 dims, 256 corpus clusters, 100 queries, k=10, `nprobe: 64` over 1,024
trained leaves, `rerank_factor: 2.0`, mmap directory, page cache warmed
(single idle-machine run; compare rows within the table, not across runs).
**Harness note:** this table predates the harness's `force_merge`, so each
method served from several auto-flushed segments and every segment
contributed `fetch_k = k × rerank_factor` candidates to the exact re-rank.
That amplifies absolute recall for all rows equally (cross-method
comparison is unaffected); merged single-segment numbers are lower and are
the conservative baseline (next table).

| method       | build (s) | train (s) | .vectors (MB) | p50 (ms) | p95 (ms) | recall@10 |
| ------------ | --------- | --------- | ------------- | -------- | -------- | --------- |
| flat (exact) | 0.4       | —         | 293.5         | 4.85     | 5.51     | 1.000     |
| **tq**       | 0.7       | **0**     | 343.3         | **3.27** | **3.50** | 0.959     |
| ivf_pq       | 0.8       | 212.2     | 303.4         | 8.54     | 12.29    | 0.786     |
| **ivf_tq**   | 0.4       | 143.3     | 351.3         | 3.33     | 5.81     | **0.997** |

Two headline results. First, training-free `tq` beats exact scan by ~1.5× at
p50 with zero training. Second, at the same probe budget
`ivf_tq` beats `ivf_pq` on every axis: +0.21 recall@10 (0.997 vs 0.786 — the
4-bit residual estimate keeps the true top-10 alive inside the probed set
where 1-bit/dim PQ loses them), 2.6× lower p50 (no per-cluster ADC table
builds, which dominate IVF-PQ's query cost at nprobe 64), and ~33% less
training time (no PQ/OPQ stage). The asymptotic caveat applies to `tq` only:
it scans every code (O(N)/query), so at millions of vectors per segment the
probed lanes (`ivf_tq`, `ivf_pq`) win latency; `tq` remains the
zero-training lane for small/medium segments and the pre-training state.
x86_64 SSSE3/AVX2 kernels are correctness-pinned against the scalar
reference in unit tests; perf numbers on x86 should be captured before
quoting them.

Same setup on the shipped codec-v2 build, `force_merge` to one segment
(honest `rerank_factor: 2.0` semantics — exactly `fetch_k = 20` candidates
re-ranked):

| method       | build (s) | train (s) | .vectors (MB) | p50 (ms) | p95 (ms) | recall@10 |
| ------------ | --------- | --------- | ------------- | -------- | -------- | --------- |
| flat (exact) | 1.3       | —         | 293.5         | 5.62     | 6.13     | 1.000     |
| **tq**       | 1.3       | **0**     | 331.1         | 3.12     | 5.15     | 0.838     |
| **ivf_tq**   | 1.4       | 143.1     | 334.5         | **1.80** | **4.69** | **0.974** |

The recall delta between the two tables decomposes into two measured,
controlled effects — there is no hidden quality bug (each control varied
one factor on the same corpus and seeds):

- **Segment-count amplification** (~−0.06 tq): the shipped build re-run
  under the old multi-segment harness scores tq 0.947 / ivf_tq 0.992 —
  unmerged indexes union per-segment candidates before the exact re-rank,
  so recall rises with segment count. This is real serving behavior, not
  an artifact of the codec; a fully merged segment is the floor.
- **Codec v2 bits** (−0.05 tq, −0.01 ivf_tq at one segment): a
  pow2-padding control build (v1 geometry, +33% code bits) scores 0.891 /
  0.984. A full-pipeline estimator probe at dim 768 measures RMSE 0.0051
  (P=768) vs 0.0044 (P=1024) — the ratio is exactly `√(1024/768)`, i.e.
  the v2 estimator is as good as the geometry allows; the padding-free
  saving (−25% payload and scan work) is a genuine bits-for-recall trade.

Recall dials: `ivf_tq` (the default) recovers recall with `nprobe`
(0.974 at 64/1,024 here; the estimate quality supports much deeper probes
at ~linear cost). The flat `tq` lane's only dial is `rerank_factor`,
which is capped by the global candidate-oversubscription budget (2×), so
its merged-segment recall on boundary-dense corpora is a floor to be aware
of — it remains the zero-training lane, not the precision lane.

At 1M docs (same harness and arch, measured 2026-07-23 on the optimized
build: codec v2 padding-free rotation, scale-sorted leaf pruning, parallel
flat scan, 256-points-per-centroid coarse-training ceiling; 8,192 corpus
clusters, `nprobe: 64` over 1,024 leaves, recall against stored document
keys so multi-segment merges cannot permute the ground truth):

| method       | build (s) | train (s) | .vectors (MB) | p50 (ms) | p95 (ms)  | recall@10 |
| ------------ | --------- | --------- | ------------- | -------- | --------- | --------- |
| flat (exact) | 16.0      | —         | 2,935.4       | 80.56    | 113.05    | 1.000     |
| **tq**       | 16.8      | **0**     | 3,311.2       | 18.79    | 20.09     | **0.918** |
| **ivf_tq**   | 8.2       | 933.8     | 3,317.9       | **6.57** | **12.65** | 0.912     |

The crossover the caveat predicts is now visible: the probed lane is 12×
faster than exact scan and ~2.9× faster than the `tq` full scan at equal
recall (`nprobe`/`rerank_factor` are the recall dials). The training ceiling
cuts coarse training from 2,862 s (uncapped, same corpus shape) to 934 s
with no measured recall penalty. For the removal record: on the pre-removal
build at 1M, `ivf_pq` measured p50 16.04 ms (2.4× slower than optimized
`ivf_tq`) and 3,517 s training; its 1M recall was never validly measured
(the pre-fix harness compared permuted doc ids), so the 100k table above is
the recall evidence.

## IVF-TQ (`index_type: ivf_tq`)

Sub-linear probing with the training-free leaf codec: the trained global
coarse router (same `build_vector_index()` machinery, HNSW/SOAR supported)
plus TQ codes of the centroid residual `r = x̂ − c` per leaf. Training
samples, indexed ANN copies, and queries are always normalized before
routing and residual encoding; original stored vectors remain unchanged for
exact reranking. Residual norms carry ranking information, so each vector
stores `scale = ‖r‖` next to `gamma`, and the leaf estimate is

```
⟨q̂, x⟩ = ⟨q̂, c⟩ + scale · est⟨q̂, r̂⟩
```

Because residuals are encoded in absolute rotated space (not per-cluster
tables), the whole query needs **one** LUT set plus one `⟨q̂,c⟩` scalar per
probed cluster — plan build is `O(P·16 + nprobe·dim)` instead of IVF-PQ's
`nprobe` ADC tables (which dominated its measured per-query cost). Blocks are
`[16 scales][16 gammas][nibbles]`; runs are per-leaf, merges byte-copy.
On disk `quantizer_version` is the trained centroid generation and
also carries the cosine-format marker; `codebook_version` carries the codec
fingerprint. Only centroids are stored as trained artifacts
(`codebook_file` must be absent).

Unmarked, pre-cosine IVF-TQ artifacts are unsupported and rejected while the
trained generation or ANN payload is opened. Rebuild the index with a current
Summa version; the current reader and writer do not load or migrate the legacy
format.

Coarse training draws a deterministic uniform point sample directly at the
256-points-per-centroid ceiling and keeps it as one contiguous matrix. Small
codebooks use k-means++ initialization; large codebooks use bounded k-means||
rounds with at most `2K` candidates total. When a second seed is affordable, an
in-buffer held-out suffix selects by exact distortion, routed distortion, and
posting-list balance. Hierarchical child counts are proportional to parent
population.

Construction routing is intentionally wider than query routing: HNSW uses a
larger build search budget and two-level routing inspects at least four
populated parents when available. This improves permanent posting assignment
without changing the bounded production query path. Selective SOAR calibrates
to at most its target spill fraction on the calibration sample; a tied quantile
underfills rather than exceeding the posting-storage budget. IVF-TQ enables
this one-secondary, at-most-30% selective preset when `soar` is omitted;
`soar: off` explicitly disables secondary assignments.

```sdl
field e: dense_vector<768, f16> [indexed<ivf_tq, num_clusters: 1024, nprobe: 64>]
```

`num_clusters`, `nprobe`, `routing`, and `soar` all apply (unlike `tq`).

### Scan-time pruning and parallelism

IVF-TQ leaves are serialized in descending residual-scale order. At scan
time each block's best possible score is bounded by
`⟨q̂,c⟩ + max_scale · 1.3` (the 1.3 covers estimator slack); once a block's
bound cannot beat the running k-th score the rest of its run is skipped —
equality with the unpruned scan is pinned by a regression test, and skipped
block counts are logged at debug level. Flat `tq` scans fan out across the
Rayon pool above 65k vectors (per-task collectors, merged top-k).

## Explicit non-goals in v1

- No configurable bit width (4 = 3+1 fixed; header carries it for evolution).
- No WASM in-browser TQ _encoding_ (WASM reads native-built payloads;
  `LocalIndex` encode support is a follow-up).
- No L2 metric (dense scoring in Summa is cosine/dot; unchanged).
