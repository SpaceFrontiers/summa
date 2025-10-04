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

The previous sqrt(N) default cited a 15M-row latency sweep. This study counts
work, not latency, so per-leaf overhead (run directories, probe selection,
prefetch ranges) is not included. Merges now keep one run per leaf
([binary vector storage](binary-vector-storage.md)), which bounds that
overhead; latency should be re-measured on production-sized indexes.

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
