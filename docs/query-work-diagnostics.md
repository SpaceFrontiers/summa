# Query work diagnostics

## Purpose and invariant

Latency alone cannot distinguish extra traversal from an expensive decoder or
scorer. The optional `query-diagnostics` native feature records bounded work
counts in the existing owners; it does not introduce another query executor,
change ranking, or change persisted formats. Normal builds compile out both the
counter updates and their arguments. Diagnostic timings must never be reported
as production latency.

Structures own
decoded block/value/byte counts, query scorers own score and pruning counts, and
the posting reader counts envelope opens (`postings_opened`, `positions_opened`).

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

Counter definitions and capture APIs live in
[`search_diagnostics`](../hermes-core/src/search_diagnostics.rs).
