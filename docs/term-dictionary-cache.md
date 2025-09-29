# Process-wide term dictionary cache

## Problem

## Design

The document store already solved this shape of problem with one byte-bounded
cache shared by every search reader in the process. That cache becomes a
generic `SharedBlockCache<V>` (`structures/block_cache.rs`): namespaced keys
`(directory, segment, block)`, shared read lock on hits, insertion-ordered
eviction under one byte ceiling, a per-entry admission ceiling, and
reference-counted namespaces whose entries are dropped when the last reader
of a segment closes. The store keeps its behavior and tests; SSTable readers
use the same type for decoded dictionary blocks. No code is duplicated.

- `IndexConfig.term_cache_process_bytes` (new; default 256 MiB on 64-bit
  native builds, 0 elsewhere). When non-zero, every search-time term
  dictionary opened by indexes with the same budget uses one shared cache,
  registered like the store cache. Per-segment `term_cache_blocks` and
  `term_cache_budget_bytes` apply only when it is 0; setting them while the
  shared cache is active is reported with `log::warn!` at open time.
- Merge, reorder and maintenance readers keep private per-segment caches, so a
  sequential merge cannot flood the search cache.
- Overlapping reader generations of one segment share blocks through the
  namespace; closing the last one frees its entries immediately.
- Bulk leading-block prefetch (merge-only) is skipped for shared-cache readers.
- Observability: per-reader `cached_blocks`/`cached_bytes` report the
  namespace's share; oversized blocks that bypass retention are counted in
  `SSTableStats::cache_insert_bypasses`; the process budget is logged once.
- WASM and standalone `Searcher::open` keep per-segment caches.
- Operators: `summa-server --term-cache-budget-mb` (default 256, 0 selects
  per-segment caches) and `summa-tool search --term-cache-process-bytes`.

No index format, query semantics or result changes. Memory: the ceiling is
per process and filled only by dictionary reads; point-lookup workloads retain
a few blocks per query term.

## Validation

Unit tests cover sharing across readers, namespace release, the byte ceiling
across segments, bypass accounting and budget-keyed registry sharing; an
index-level test pins one ceiling across three segments and that merges do not
fill the search cache. The 677-query exact audit passes with both policies;
paired HTTP and aarch64 measurements are in the campaign report.
