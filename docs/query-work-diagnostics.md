# Query work diagnostics

## Purpose and invariant

Latency alone cannot distinguish extra traversal from an expensive decoder or
scorer. The optional `query-diagnostics` native feature records bounded work
counts in the existing owners; it does not introduce another query executor,
change ranking, or change persisted formats. Normal builds compile out both the
counter updates and their arguments. Diagnostic timings must never be reported
as production latency.

## Capture and cost model

Fixed-size, thread-local counters with no hot-loop allocation, locks, shared
atomics, or per-block clocks. Each capture allocates a small shared accumulator;
Synchronous Searcher segment workers inherit it explicitly and merge once on completion. A synchronous capture
restores the previous scope on return or panic. Async capture installs the scope
only during each poll, restoring it before suspension; unrelated tasks and
cancelled futures cannot leak counters into another query. Nested captures are
exclusive. Arbitrary spawned tasks are outside the capture. The synchronous Searcher segment boundary
is instrumented explicitly, including parallel segments; its scheduling and
production query paths are unchanged.

Count decoded occurrences, not distinct blocks/documents: repeated decodes count
again. Payload bytes are bytes consumed by decoders, excluding metadata, and are
not disk I/O or cache-miss measurements. BM25 units are term-document score
evaluations (or phrase-document evaluations), not distinct result documents.
Executor window/block pruning counts apply only to the paths that implement
those optimizations; zero does not prove a query performed no other pruning.

## Experiment

Run the same queries on legacy, compact/exact, and compact/quantized indexes with
identical document order, caches, and RGB disabled. Capture a cold process pass
and a repeat pass, checking protocol responses and ranked results independently.
Compare query-by-query and by family:

- Doc/TF/position blocks, decoded values and payload bytes; posting seeks.
- Exact versus lookup score evaluations, score batches and norm tables built.
- Bound evaluations, candidate/heap admissions, windows and blocks pruned.
- Phrase confirmations and position requests.
- Posting/position envelope opens, without admission scans or a validation cache.
- Adapter parse and execute wall time, plus summed segment-worker wall time,
  measured outside hot loops and used only as diagnostic phase attribution.
  With one segment, execute minus segment time exposes orchestration overhead.
  With parallel or nested segments, these wall times cannot be subtracted.

Join these counts to a separate uninstrumented latency run. Equal work with
slower queries points to per-unit cost; more work points to traversal/pruning.
Do not divide total query time by one counter and call it decoder/scorer time.
Compare Tantivy only where counter definitions and corpus geometry agree. Keep
unmatched counters explicitly unavailable. Use a separate, score-bit-equivalent
canonical-arithmetic experiment on the same quantized index if traversal counts
cannot explain the norm regression.

## Running it

Build a dedicated binary (never use it for production latency):

The configuration contains `queries` (JSONL paths), `expected` (aligned exact
counts), and `engines` (name to argument-array mapping). Optional `passes`
defaults to two and is capped at ten. The collector starts each engine once,
runs all four commands, checks each response, enforces a per-query timeout, and
retains raw logs and structured per-query counters. The first pass is process
cold, not a claim of cold OS page cache. Repeat 1 is used for comparison.

## Sparse backends

BMP is the default sparse backend; MaxScore and Seismic are explicit field
formats. Compare each using the same exact result oracle, fixture, quantization,
query pruning, and result depth. BMP's `lsp_gamma: 0` disables its approximate
superblock selection; Seismic's `exhaustive: true` scores forward vectors without
nomination. These switches do not undo index-time pruning or weight quantization.

The text score counters above exclude sparse/vector scoring and point reranking.
Seismic has separate `QueryWork` counters: `seismic_clusters` counts visited
clusters, `seismic_documents` counts eligible documents scored, and
`seismic_forward_rows` / `seismic_forward_bytes` include repeated exact backfill
and winner hydration. `seismic_filter_scans` counts selective or underfilled
filtered executions that scan forward values; `seismic_budget_truncations`
identifies finite nomination budgets reached before traversal completes.

BMP reports its own phase and work metrics through `hermes_bmp_*`: preparation,
grid scoring, prefetch, block scoring, document-map lookup, and LSP selection;
visit/skip counts and prefetched bytes explain the work behind those phases.
These are metrics captures, not fields in `QueryWork`. Sparse MaxScore has no
separate equivalent work-counter family. A zero text or Seismic counter must not
be interpreted as zero BMP or MaxScore work. This is not an index-wide profiler.
