# Collector selection experiment

This benchmark compares the unchanged production `ScoreCollector` with two
benchmark-only prototypes: a bounded 2k partial-selection buffer and a loser
tree. Neither replaces a production collector. Context and upstream source pins
are in the [IResearch audit](iresearch-optimization-audit.md#top-k-selection).

## Invariant and scope

All three return identical ordered document IDs, score bits, and ordinals under
Summa's total float and tie ordering. Full-sort oracle checks precede timing,
including empty/short streams, k=0/1, non-power-of-two trees, negative scores,
ties, signed zero, infinities, and NaNs. Timed fixtures contain finite positive
scores; the shared eight-score screen follows the existing text admission shape.
The baseline calls the real production `ScoreCollector`, not a copied heap.

The collector receives 100,000 candidates (128 for the short control) with
k=10/100/1000. A fixed-seed generator uses the canonical three-term BM25 scorer
outside timing. Improving and deteriorating score orders stress replacement and
rejection; rounded scores stress ties; clustered blocks stress threshold feedback.
These are synthetic streams, not traces captured from production queries.

Replay measures collector construction, screening, admission, final ordered
materialization, and destruction. Pruned mode uses exact precomputed 128-candidate
block maxima and each collector's current threshold. It measures block skipping
and admission, but excludes score computation, posting decode, and storage I/O.
Its bounds are intentionally ideal: it isolates the cost of delayed threshold
updates without claiming production MaxScore bound tightness or query latency.
The match-count/position collector, shared seeded thresholds, and parallel
cross-segment execution are outside this experiment.

The heap retains k entries; partial selection retains at most 2k; the tree retains
k entries and k cached competitor nodes. Printed backing-byte figures describe
allocation capacity from the concrete types, not process RSS or allocator
overhead. A full sort of the input is used only by the untimed oracle. The input
fixture is shared across algorithms and excluded from collector memory figures.
The loser-tree prototype caches complete Summa ordering keys in 16-byte nodes.
Upstream IResearch caches score/leaf pairs and uses different tie admission;
these results measure this tie-preserving Rust implementation, not its C++ binary
or every possible loser-tree representation.

## Reproduction

```sh
# Release-mode correctness preflight, without timing.
cargo bench --locked -p summa-core --bench collector_selection -- --test

# Run adjacent algorithm comparisons with recorded environment/source identity.
python3 scripts/check_search.py bench --bench collector_selection --save-baseline collectors-forward
COLLECTOR_REVERSE=1 python3 scripts/check_search.py bench --bench collector_selection --save-baseline collectors-reverse
```

Each case uses 300 ms warmup, one second of measurement, and 30 Criterion
samples. Reverse order checks gross order/thermal bias; it does not eliminate
machine noise. Keep CPU, compiler, features, and flags identical. These are
exploratory microbenchmarks; do not change production defaults from this alone.

## Results

September 20, 2026: Apple M4 (10 logical CPUs), macOS 15.6.1, Rust 1.98.1,
LLVM 22.1.8, default optimized bench profile, empty `RUSTFLAGS`. Both runs use
the same binary/source and input seed. No compilation or tests overlapped the
timed runs. The desktop had active background processes, and several second-run
cases slowed substantially; small differences are inconclusive. No x86 timing
claim is made. The live index host was not benchmarked under its active workload.

Times below are Criterion median microseconds per complete collection. Each cell
is **forward-order / reverse-order**; these are independent runs, not a confidence
interval. The [complete CSV](benchmark-results/collectors-2026-09-20.csv) includes
all 126 estimates and their within-run 95% bootstrap intervals. The
[manifest](benchmark-results/collectors-2026-09-20.json) records compiler, flags,
source hashes, commands, and scope.

| Workload                     |    k | Production heap | Partial selection |      Loser tree |
| ---------------------------- | ---: | --------------: | ----------------: | --------------: |
| BM25 replay, 100k candidates |   10 |   39.52 / 39.31 |     40.38 / 40.33 |   40.70 / 40.94 |
| BM25 replay, 100k candidates |  100 |   65.89 / 73.14 |     51.48 / 52.49 |   71.85 / 71.05 |
| BM25 replay, 100k candidates | 1000 | 302.91 / 338.30 |   115.52 / 134.04 | 347.35 / 384.67 |
| Clustered, block pruning     |   10 |     4.62 / 4.72 |       5.37 / 5.38 |     7.85 / 7.89 |
| Clustered, block pruning     |  100 |   34.85 / 39.22 |     14.39 / 14.86 |   47.83 / 47.36 |
| Clustered, block pruning     | 1000 | 267.52 / 288.76 |     78.95 / 85.74 | 435.21 / 350.60 |
| Short replay, 128 candidates |   10 |   0.383 / 0.462 |     0.536 / 0.639 |   0.638 / 0.683 |

Partial selection reduces BM25 replay time by **22–28% at k=100** and **60–62%
at k=1000** across the two runs. There is no meaningful top-10 BM25 win; it is
38–40% slower on the short top-10 stream. On the monotone improving stress case,
where replacements dominate, its reduction is 41–82% depending on k/run. This
adversarial case should not be substituted for a production query distribution.
The loser tree provides no consistent advantage over the heap on BM25 replay
and loses substantially in the pruning and improving-score cases.

The pruning workload also exposes the delayed cutoff's hidden cost:

|    k | Heap/tree candidates visited | Partial-selection candidates visited | Additional visits | Blocks skipped, heap / partition |
| ---: | ---------------------------: | -----------------------------------: | ----------------: | -------------------------------: |
|   10 |                         1664 |                                 1792 |              7.7% |                        769 / 768 |
|  100 |                         1792 |                                 2176 |             21.4% |                        768 / 765 |
| 1000 |                         6272 |                                 8192 |             30.6% |                        733 / 718 |

These counts match in both runs. Precomputed scores make those extra visits
cheap here; real queries would also pay scoring/decoding costs. Collector-only
speedup is therefore not an end-to-end query speedup estimate.

Retained allocation capacities, excluding the input and result materialization:

|    k |     Heap | Partial selection | Loser tree |
| ---: | -------: | ----------------: | ---------: |
|   10 |    120 B |             240 B |      280 B |
|  100 |   1200 B |            2400 B |     2800 B |
| 1000 | 12,000 B |          24,000 B |   28,000 B |

Collector objects add 48/48/72 bytes respectively. The tree caches ordering keys
as well as retained hits; the partition buffer reserves space for k extra hits.
These are layout/capacity measurements, not peak process RSS.

Decision: retain the production heap. Bounded partial selection is the promising
next end-to-end experiment for larger k, including real MaxScore threshold
feedback, shared threshold seeding, final sorting, and counted/position paths.
Do not infer a universal k crossover from this single synthetic ARM fixture.

## Validation and raw evidence

- Release correctness preflight and all timed-case oracle checks passed.
- `python3 scripts/check_search.py check` passed: format, Clippy, focused native
  tests, native-without-sync, and standalone broker compilation.
- Documentation links/inventory and `git diff --check` passed.
- Runtime/WASM code is unchanged; no WASM rebuild, full RPC run, or end-to-end
  search latency measurement was performed for these benchmark-only prototypes.

Workspace evidence (gitignored):

```text
.context/search-harness/20260920T112922.388241Z-check/
.context/search-harness/20260920T113100.802077Z-bench/
.context/search-harness/20260920T113246.139461Z-bench/
.context/search-harness/criterion/collector_*/.../collectors-{forward,reverse}/
```
