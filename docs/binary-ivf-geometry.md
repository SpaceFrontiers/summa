# Binary IVF geometry: 4·sqrt(N) leaves, 128 probes

## Decision

Automatic binary IVF codebooks now train `ceil(4·sqrt(N))` leaves (clamped to
16..1,048,576), where `N` is the larger of the live corpus and `target_vectors`.
The default `nprobe` for binary IVF is 128 (`DEFAULT_BINARY_IVF_NPROBE`); binary
ScaNN and flat keep 64. Explicit `num_clusters` and `nprobe` still win. Float IVF
(8·sqrt(N)) and ScaNN geometry are unchanged. No format changes: the leaf count
already lives in the trained artifact.

Existing indexes keep their trained codebooks and persisted `nprobe`. A
retrain of an automatically sized field picks the new leaf count; when its
persisted `nprobe` is below 128, training logs a warning, because 4·sqrt(N)
leaves probed 64 times lose recall (0.942 → 0.889 recall@10 at 1M vectors).

## Evidence

Study: 1,000,000 NQ passages embedded with Cohere `embed-english-v3`
(1,024 dimensions, sign-binarized to 1,024-bit codes) and 1,000 real NQ
queries; exact Hamming and float ground truth. The baseline is Summa's own
`BinaryCoarseQuantizer::train` with flat routing. Work counts are postings
Hamming-scored; "+K" adds one centroid distance per leaf for flat routing.
Raw results, scripts, and seeds are retained outside the repository in the
workspace `.context/recall-study/`; the summary is reproduced here.

| Leaves      | Mean leaf | Postings @0.90 / 0.95 / 0.99 recall | Postings + K @0.90 / 0.95 / 0.99 |
| ----------- | --------- | ----------------------------------- | -------------------------------- |
| 1,000 (√N)  | 1,000     | 34.9k / 76.6k / 294k                | 35.9k / 77.7k / 295k             |
| 4,000 (4√N) | 250       | 19.6k / 44.6k / 201k                | 23.6k / 48.6k / 205k             |
| 16,000      | 62        | 12.2k / 31.4k / 133k                | 28.3k / 47.4k / 150k             |

4·sqrt(N) scans 1.72–1.78× fewer postings than sqrt(N) at 0.90–0.95 recall
(1.5–1.6× counting centroid routing). 16·sqrt(N) ties it at 0.95 including
routing and wins only at 0.99. At a fixed 64 probes recall falls with smaller
leaves, so the probe default doubles: 4·sqrt(N) at 128 probes scans 33.1k
postings for recall@10 0.936 (Hamming) / 0.654 (float), against 65.0k postings
for 0.942 / 0.657 at sqrt(N) and 64 probes.

The previous sqrt(N) default cited a 15M-row latency sweep. End-to-end
latency on the same 1M real codes confirms the work counts. Each geometry was
indexed into one merged `MmapDirectory` segment, with explicit `num_clusters`
and `nprobe` swept through metadata-only ALTERs. The queries were 1,000 real NQ
queries (tie-aware recall@10 against exact Hamming), two warm passes, on an
8-vCPU Xeon @ 2.9 GHz with AVX-512 VPOPCNTDQ:

| Recall@10 | sqrt(N) = 1,000 leaves    | 4·sqrt(N) = 4,000 leaves | 16·sqrt(N) = 16,000 leaves |
| --------- | ------------------------- | ------------------------ | -------------------------- |
| ≈0.89     | 0.553 ms p50 (nprobe 32)  | 0.318 ms (64)            | 0.364 ms (128)             |
| ≈0.935    | 1.082 ms p50 (nprobe 64)  | 0.586 ms (128, default)  | 0.801 ms (384)             |
| ≈0.965    | 2.140 ms p50 (nprobe 128) | 1.153 ms (256)           | 1.642 ms (768)             |

4·sqrt(N) is the latency optimum at every recall level (1.74–1.86× faster than
sqrt(N)); 16·sqrt(N) pays for routing over four times more centroids. Query
time is the Hamming scan itself (48% of all profile samples are the kernel on
the search thread), and it is memory-bandwidth bound: about 8 MB of codes per
query at roughly 8 GB/s on one core. Bytes read per query, not per-leaf overhead,
are the cost to reduce. Production-sized indexes on cold storage remain
unmeasured.

## Training cost and hierarchical quality

Codebooks with fewer than 4,096 leaves train directly; at 1M vectors and 4,000
leaves direct k-majority took 280 s on two cores (7 s at 1,000 leaves).
Hierarchical (two-level) training of the same 4,000 leaves took 6.5 s but needs
1.23–1.26× more postings at 0.90–0.95 recall (24.1k / 56.3k; second seed
24.4k / 58.6k). The threshold therefore stays at 4,096: below it quality wins
and the cost is bounded (at most ~1M sampled vectors). Billion-scale fields
always train hierarchically, so improving hierarchical leaf quality is the most
direct follow-up for them. A prototype exhaustive k-majority refinement started
from the hierarchical leaves recovers only part of the gap: one global iteration
gives 21.3k / 53.7k postings at 0.90 / 0.95 (three iterations: 21.9k / 53.2k),
against 24.1k / 56.3k unrefined and 19.6k / 44.6k for direct training. An
exhaustive pass is impractical at 200k leaves, and a bounded parent-beam pass
would recover less, so it is not adopted.

## Options evaluated and not adopted

| Option                                                    | Result on the same data                                     |
| --------------------------------------------------------- | ----------------------------------------------------------- |
| Float k-means on ±1 codes, sign-binarized centroids (BRB) | 0.97–1.10× (seed noise), 9–47× slower training              |
| Exact covering-radius / per-code triangle pruning         | ~0% pruned: leaf radius ≈364 bits vs 10th neighbor ≈319     |
| Probe while within Δ of the best centroid                 | 0.83–1.01× (worse than fixed nprobe)                        |
| SOAR-Hamming spill (λ = 1, one secondary)                 | 1.40–1.83× at 4·sqrt(N); doubles payload, needs format work |
| Early stop from current 10th distance                     | 1.1–1.35×, mostly at ≥0.97 recall                           |
| Asymmetric rescoring with a 4-bit float query             | +7–9 points float recall@10, saturating at 4× oversampling  |

SOAR spill and asymmetric rescoring are the remaining worthwhile changes; both
need design work (a versioned payload with duplicate postings, and a float
query path respectively) before implementation.

## Candidate: prefix-first two-stage scan

Because the scan is bandwidth bound, reading fewer bytes per posting matters
more than fewer instructions. Ranking probed postings by Hamming distance on a
code prefix, keeping the best M, and rescoring only those on the full code
gives, at 4·sqrt(N) on the same data:

| nprobe | Full scan: recall / bytes | 512-bit prefix, M = 1,000: recall / bytes | M = 500         |
| ------ | ------------------------- | ----------------------------------------- | --------------- |
| 64     | 0.889 / 2.15 MB           | 0.889 / 1.21 MB                           | 0.886 / 1.14 MB |
| 128    | 0.936 / 4.24 MB           | 0.935 / 2.25 MB                           | 0.930 / 2.18 MB |
| 192    | 0.955 / 6.31 MB           | 0.952 / 3.28 MB                           | 0.948 / 3.22 MB |
| 256    | 0.964 / 8.36 MB           | 0.962 / 4.31 MB                           | 0.957 / 4.25 MB |

At equal bytes (about 3.2 MB) the cascade reaches 0.952 recall against 0.917
for a full scan; at equal recall it reads 1.8–1.9× fewer bytes. A 256-bit prefix
loses too much (0.889 at M = 1,000 for nprobe 128). These embeddings are not
Matryoshka-trained, so any 512 bits are as informative as any other.
Matryoshka-trained models concentrate information in the leading dimensions, so
a short prefix of their sign codes should do better. That is untested here and
must be measured on the production model before adopting a prefix length.

On the production Matryoshka model the prefix can be much shorter. The scan is
implemented as the per-field `prefix_bits` / `prefix_rerank` option; see the
[prefix-first scan](binary-prefix-scan.md).
