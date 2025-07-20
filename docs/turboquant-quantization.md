# TurboQuant (TQ) — training-free dense ANN codec

Status: implemented (v1, `AnnKind::TqFlat`).

## Motivation

IVF-PQ requires corpus-trained global artifacts (coarse centroids + OPQ-PQ
codebook). Until `build_vector_index()` runs, dense queries brute-force the
flat vectors; every trained generation adds version gates, and OPQ training is
disabled on WASM. TurboQuant (Zandieh, Daliri et al., arXiv 2504.19874; see
also mayflower/pg_turboquant, MIT) is a _data-oblivious_ quantizer: its
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

## Codec (bits = 4 per padded dimension, fixed in v1)

Encode, per vector `x` (always the L2-normalized vector; cosine and unit-norm
dot coincide after normalization, exact re-rank applies the field's metric):

1. Zero-pad to `P = dim.next_power_of_two()`, apply seeded rotation
   `R1` = sign flips → normalized FWHT → permutation (orthonormal; seed is a
   codec constant).
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

| lane     | bytes/vector                     | trained artifacts        |
| -------- | -------------------------------- | ------------------------ |
| flat f32 | 3072                             | none                     |
| IVF-PQ   | 96 codes + assignment            | centroids + OPQ codebook |
| TQ       | 512 nibbles + 4 gamma (P = 1024) | none                     |

Non-power-of-two dims pay FWHT padding (768 → 1024, +33%). A structured
non-pow2 transform (Kac walk / block butterfly) is possible follow-up work.

## Explicit non-goals in v1

- No IVF-routed TQ leaves (would reuse trained coarse centroids; follow-up).
- No configurable bit width (4 = 3+1 fixed; header carries it for evolution).
- No WASM in-browser TQ _encoding_ (WASM reads native-built payloads;
  `LocalIndex` encode support is a follow-up).
- No L2 metric (dense scoring in Hermes is cosine/dot; unchanged).