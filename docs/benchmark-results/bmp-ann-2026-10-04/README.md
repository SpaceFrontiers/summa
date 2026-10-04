# BMP, binary IVF, IVF-TQ and ScaNN investigation — 2026-10-04

Baseline: `9ea50c169ec7c2b9782f99e9533e8ccf8e463c72`. The retained production
change skips a 64-score binary IVF window when no score can reach the current
collector floor. A non-short-circuit boolean reduction allows native SIMD
comparisons before metadata access. Equal scores remain competitive; unbounded
collectors bypass the reduction. ARM64 and x86-64 use it; WASM retains the
existing scalar loop. There is no new allocation, scratch buffer or setting.

BMP runtime, TQ accumulation and ScaNN accumulation remain unchanged after
rejecting the experiments below. New boundary tests and reusable measurement
tools remain. IVF-PQ was already retired before this campaign; the current
quantized paths reviewed here are **IVF-TQ and ScaNN AH**.

**No index rebuild is required for these changes.** Writers, persisted layouts,
training artifacts, probe/candidate budgets, reranking and wire formats are
unchanged. This does not add support for previously retired IVF-PQ payloads.

## Measured decisions

[The complete table](tables.md) contains **149 paired comparison cells**, including
rejected candidates, medians, applicable p95 values and peak process RSS.

| Path / experiment                                                       | Evidence                                                                                                                                                                              | Decision                                                            |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Binary IVF, final guarded window rejection                              | ARM: 256-bit medians improve 6.0–26.7%; 2,560-bit improve 3.1–11.4%. x86: 256-bit five cells improve 3.3–14.3%, one regresses 2.1%; wide codes range from 3.6% faster to 4.0% slower. | Retain; useful across both architectures, with disclosed tradeoffs. |
| BMP A: search only the remaining sorted term suffix                     | Six ARM cells; uniform queries regress about 30–37%.                                                                                                                                  | Reject.                                                             |
| BMP B: contiguous touched-mask max/clear                                | Six ARM cells; no consistent gain.                                                                                                                                                    | Reject.                                                             |
| BMP C: dense 64-slot max/clear at 16 touched slots                      | 24 ARM cells mixed; four production-derived x86 replay cells regress 0.3–2.8%.                                                                                                        | Reject.                                                             |
| BMP D: accumulate a local touched word for blocks ≤64                   | 24 ARM cells; short queries benefit, wider controls regress.                                                                                                                          | Reject.                                                             |
| BMP E: restrict local-word accumulation to ≤8 query terms and ≤64 slots | 30 ARM cells including 2/4/8/60 terms and large-block controls; short-query gains still accompany regressions.                                                                        | Reject; no extra runtime dispatch retained.                         |
| TQ: widen after 256 dimensions instead of 128                           | ARM 768-dimension LUT16 kernel: 122.38 → 123.43 µs (+0.9%).                                                                                                                           | Reject at first gate; no x86 follow-up.                             |
| ScaNN: widen after 128 word pairs instead of 64                         | Three kernels on each architecture; largest improves 1.9% ARM / 0.5% x86, small controls flat or slower.                                                                              | Reject; no demonstrated query-level gain.                           |

The preliminary binary B/C measurements omitted the unbounded-floor guard.
`binary-final` tables are the evidence for the retained code. Peak RSS varies
with fixture/oracle construction and allocator behavior; it does **not** establish
a memory reduction. Added production scratch is zero. Native x86 binary runs
peak at 96.6–97.0 MiB across both variants.

## Research and existing implementation

Machine Library was used to retrieve and examine the primary BMP, Quicker ADC
and TurboQuant papers. Official Faiss implementation documentation provided
additional binary/FastScan context. No paper was treated as evidence of a Summa
speedup; all retention decisions above use this repository's measurements.

| Method                                                                | Current implementation / consequence                                                                                                                                                                                                                                                                                                           |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BMP block-local inverted scoring, hierarchical bounds and BP locality | Already represented by adaptive blocks, grids, ordered/pruned traversal and reorder support. Five remaining hot-loop hypotheses were tested above. [BMP paper](https://doi.org/10.48550/arxiv.2405.01117).                                                                                                                                     |
| Register lookup tables and packed integer accumulation                | Already used by TQ LUT16 and ScaNN FastScan, with resolved kernel dispatch and scalar fallbacks. Longer safe accumulation intervals were tested and rejected. [Quicker ADC](https://doi.org/10.1109/tpami.2019.2952606), [Faiss FastScan](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-%28FastScan%29). |
| Scalar quantization plus residual sketch                              | Already owned by IVF-TQ. Changing rotations or codebooks would require separate training/format and recall evaluation. [TurboQuant](https://doi.org/10.48550/arxiv.2504.19874).                                                                                                                                                                |
| Batched exact Hamming scoring within probed lists                     | Existing SIMD batch scoring is preserved; this campaign improves rejection of noncompetitive score windows. [Faiss binary indexes](https://github.com/facebookresearch/faiss/wiki/Binary-indexes).                                                                                                                                             |

## Fixtures and methodology

Both hosts use Rust 1.98.1, release builds and empty `RUSTFLAGS`. ARM measurements
use an Apple M4 on macOS 15.6.1. x86 measurements use a GCP Intel Xeon 2.80 GHz
(family 6/model 85), Ubuntu kernel 7.0.0, pinned to CPU 2. Rayon has one thread.
Each comparison runs baseline/candidate/candidate/baseline, on the same host,
with the same fixture and flags. Builds and packaging do not overlap retained
timing runs. Two rounds per variant are useful controls, not proof that small
effects generalize to every CPU or workload.

- **Binary:** 131,072 deterministic rows, 32 persisted leaves, widths 32/320
  bytes, 16 queries, 1/8/32 explicitly selected leaves and depths 10/100.
  Each run checks every query against an independent exhaustive Hamming oracle,
  then records seven passes (112 latency samples per cell per run).
  Writer-byte and result hashes match across all measured variants and hosts.
  This isolates warm preassigned-leaf scans; it does not measure trained routing
  recall, cold storage or whole-service latency.
- **BMP synthetic:** deterministic skewed corpus and skewed/uniform query
  streams, 200 queries each, 200K documents for A/B and 100K for C/D/E.
  Controls cover 32/256-slot blocks and 2/4/8/12/60 terms where indicated in
  the tables. The optional audit executes both distributions, depths 10/100
  and gamma zero/quarter outside timing. All 48 unique audit keys agree.
  Criterion uses 30 samples with rotating queries. Its run medians are not
  individual request latency percentiles.
- **BMP real replay:** the retained September 19 production-derived fixture
  contains 42,596 documents and approximately one million sparse vectors;
  [the original manifest](../bmp-forward/2026-09-19/query-manifest.json) identifies
  source windows. An isolated copy of the existing index was opened without
  rebuilding. 216 retained queries × depths 10/100 × gamma 0/100 gives 864
  unique cases; all IDs and score bits agree across C and baseline. One warmup
  and three recorded passes per run produce 10,368 records across four runs.
  Corpus/query contents are not committed here; fixture file hashes and result
  digests are recorded. The query-file hash was not captured separately.
- **Quantized kernels:** existing ScaNN 32/96/384-block and TQ 768-dimension
  benchmarks isolate lookup accumulation, not complete ANN queries or recall.

The x86 C synthetic BMP matrix is excluded because partial-result packaging
overlapped one timing interval. Its preceding binary, ScaNN and real replay
phases completed before that overlap and remain valid. Initial local baseline
test builds accidentally reused Cargo artifacts across mirrored source paths;
they were discarded and explicitly recompiled before the retained comparisons.
The forced-build provenance and saved binary hashes distinguish these builds.

## Reproduce and audit

Run `python3 docs/benchmark-results/bmp-ann-2026-10-04/analyze.py` to validate
hash equivalence and regenerate every table from [compressed raw measurements](evidence.json.gz).
The evidence contains raw per-query times, Criterion iterations/times, replay
result digests and per-process RSS. [Provenance](provenance.json) records hosts,
binary and fixture hashes. Rejected source changes are retained in `patches/`;
they are experiments against the baseline, not production patches to apply
together. `export_evidence.py LOG_DIRECTORY` documents normalization from the
original ignored log tree (including `results/` and `x86-final/results/`).

For new timing, build each source revision **before** running an ABBA sequence.
Use separate target directories, or force recompilation after changing a mirrored
checkout; do not trust same-package Cargo fingerprints across copied sources.

```sh
cargo test --locked -p summa-core --release --lib --no-run
# Copy the emitted test executable for each variant, then run it directly:
RAYON_NUM_THREADS=1 /path/to/saved/core-tests \
  measure_persisted_binary_leaf_scans --ignored --nocapture

BMP_BENCH_WIDE_DOCS=100000 BMP_BENCH_BLOCK_SIZE=32 \
BMP_BENCH_QUERY_DIMS=8 BMP_BENCH_AUDIT=1 RAYON_NUM_THREADS=1 \
cargo bench --locked -p summa-core --bench bmp_hot_path -- \
  --warm-up-time 0.2 --measurement-time 1 --noplot

cargo build --locked -p summa-core --release --example bmp_replay
RAYON_NUM_THREADS=1 target/release/examples/bmp_replay INDEX QUERIES_JSON 3
```

For publication-quality repeated runs, save benchmark executables and set a
distinct `CRITERION_HOME` per run; run them directly with `--bench` so no build
overlaps timing. The Linux wrapper adds `/usr/bin/time -v taskset -c 2`; macOS
uses `/usr/bin/time -l`. The replay takes the original field name `vector` and
uses maximum multi-value combination, one search worker and two Tokio workers.

## Validation and remaining work

The final validation results are recorded in [validation](validation.md).
Boundary tests cover signed/unsigned maximum accumulations around flush limits
and sparse touched slots across scoring phases/64-slot boundaries. Existing
binary regressions cover equal scores, deletions, serial/parallel search and
complete ordinal collection. The new independent oracle additionally checks
all measured binary queries and persisted bytes.

This campaign does not claim exhaustive coverage of all possible methods.
Remaining useful work is a representative **whole ANN query** recall/latency
matrix (trained routing, reranking and realistic query distributions), followed
by profiling of the dominant stage. Cold I/O, concurrent ingest, service tail
latency and WASM performance were not benchmarked here. A 320-byte specialization
was built during exploration but not timed, so no performance claim is made.
New training/layout techniques and dynamic pruning policies need their own
recall and compatibility design; they are not justified by these hot-loop data.

Both existing cloud machines were stopped and verified `TERMINATED`; their
original disks remain. No publication or deployment was performed for this task.
