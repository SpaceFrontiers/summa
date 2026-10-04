# BMP and ANN optimization review

Completed October 4, 2026 against `9ea50c16`. The [measurement report](benchmark-results/bmp-ann-2026-10-04/README.md)
contains the research, 149 paired comparison cells, raw samples, rejected source
patches, provenance and validation. It distinguishes kernels, synthetic leaf
scans and production-derived BMP replay.

## Ownership and invariant

Sparse/dense/binary query planners own admission and candidate budgets. BMP
execution belongs to `query/bmp.rs`, block views to `segment/bmp_adaptive.rs`,
and persisted ANN scans to `segment/ann_disk.rs`. SIMD primitives and TQ/ScaNN
lookup accumulation belong to `structures`. Native sync/async use these same
scorers; portable builds retain scalar fallbacks.

This campaign changes no writer, persisted layout, training artifact, routing,
probe count, candidate depth, multi-value combination or reranking. No index
rebuild is required. IVF-PQ was already retired; the current quantized paths
reviewed are IVF-TQ and ScaNN AH. Previously retired IVF-PQ payloads remain
rejected as documented in [TurboQuant](turboquant-quantization.md).

## Retained change

Binary IVF scans already score bounded batches and refresh the collector floor
at 64-score boundaries. On ARM64/x86-64, a non-short-circuit boolean reduction
now rejects an entire window when none of its finite Hamming scores reaches
that floor. Surviving windows use the existing per-lane metadata/collector loop.
Equal scores remain competitive. An unbounded floor bypasses the reduction,
preserving complete ordinal collection without an additional score pass.
Portable scalar targets retain the original loop.

The reduction adds no allocation or scratch. Its benefit depends on how many
windows are wholly noncompetitive: final ARM medians improve 6.0–26.7% for
256-bit codes and 3.1–11.4% for 2,560-bit codes. x86 results are mixed but useful:
five of six 256-bit cases improve 3.3–14.3%, with a 2.1% regression in the smallest
top-100 case; wider codes range from 3.6% faster to 4.0% slower. These are warm,
preassigned-leaf measurements, not service throughput or routing recall claims.
All 12 byte/result hash keys and independent Hamming oracles agree.

## Rejected changes and remaining opportunities

Machine Library primary-paper research found that the major relevant methods
are already implemented: BMP block-local inverted scoring and hierarchical
bounds, TQ/ScaNN register LUT16 scans, resolved kernel dispatch, binary batch
Hamming and bounded reranking. Sources and implementation mapping are in the
[report](benchmark-results/bmp-ann-2026-10-04/README.md#research-and-existing-implementation).

Five BMP hot-loop candidates were tested: sorted-suffix lookup, contiguous-mask
max/clear, dense-word max/clear, local touched-word accumulation, and dispatching
that last loop only for short queries/small blocks. None demonstrated a robust
benefit across its controls, so no BMP runtime change remains. The real-fixture
candidate returned identical IDs/score bits for 864 unique cases but regressed
all four median cases by 0.3–2.8%.

Longer TQ and ScaNN integer accumulation windows were also rejected: TQ was
slightly slower, and ScaNN's largest kernel gained only 1.9% ARM / 0.5% x86 with
flat or slower controls. Their existing limits remain. New correctness tests
cover accumulation extremes and sparse scoring phases/word boundaries; benchmark
audit controls and a read-only persisted BMP replay example remain reusable.

Next, measure representative complete ANN queries with trained routing and
recall before changing probe/candidate budgets, layouts or training. Cold I/O,
concurrent ingest, service tail latency and WASM performance remain unmeasured.
No new default or compatibility policy is selected from these synthetic tests.
