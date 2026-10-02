# Process-wide term dictionary cache

September 27, 2026. Status: implemented and the default. It doubles
broad-wildcard throughput on the 10M-document benchmark (338 → 703 QPS) for
+130 MiB anonymous RSS, and uses 1.83× less CPU for the same kind of scans on
aarch64. Other families move −4% to +2%; the new binary shows the same shifts
with per-segment caches, so they follow the build rather than the policy. [Measurements](benchmark-results/dict-scan-2026-09-27/README.md#process-wide-term-cache-the-new-default).

## Problem

Search-time term dictionaries retain decompressed SSTable blocks in a cache
owned by each segment reader: `term_cache_blocks` (default 256 blocks, about
4 MiB) with an optional per-segment byte cap. Broad dictionary scans such as
`h*band` walk a whole first-letter subtree; on the 10M-document benchmark the
42 broad wildcards touch 12,503 blocks (207 MB decoded) of one segment. With
256 blocks every such query decompresses its subtree again, and
decompression is about half of its CPU. A 16,384-block / 256 MiB per-segment
cache doubles broad-wildcard throughput with other families unchanged
([measurement](benchmark-results/dict-scan-2026-09-27/README.md)). The same
per-segment setting cannot become the default: it multiplies by segment count
and by overlapping reader generations.

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
