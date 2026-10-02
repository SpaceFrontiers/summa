# Suffix filters for cached dictionary blocks

September 27, 2026. Status: implemented for shared-cache term dictionaries.

## Problem

A broad single-star wildcard such as `h*band` must examine every term of the
`h` subtree: the literal prefix bounds the dictionary range, but the suffix
cannot be located in key order. With decoded blocks retained by the
[process-wide cache](term-dictionary-cache.md), the remaining cost is the
per-term scan itself. On the 10M-document benchmark's text field, only
**4,528 of the 34,010 blocks** in the 42 broad-wildcard ranges contain any key
ending with the pattern's last four suffix bytes
([measurement](benchmark-results/dict-scan-2026-09-27/README.md)).

## Design

Each decoded dictionary block held by the shared cache can carry a lazily
built filter of its keys' last four bytes:

- A 1 KiB bloom filter (8,192 bits, three probes from one multiplicative hash)
  plus the block's entry count. It is built as a by-product of the first
  complete scan of that block and published once (`OnceLock`); concurrent
  scanners never wait for each other.
- Its size is charged to the block when it enters the cache, so the process
  budget stays exact whether or not a filter is ever built. Per-segment caches
  (merges, standalone searchers, `term_cache_process_bytes = 0`) build no
  filters and keep their accounting unchanged.
- A single-star query whose suffix has at least four bytes passes that ending
  to the prefix scan. Only **interior** blocks — strictly between the block
  containing the prefix and the block containing its successor key — may be
  skipped. Every key of an interior block is inside the range, so a skipped
  block adds exactly its entry count to the scanned-term budget: budget errors,
  expansion limits and results are identical to a full scan.
- Bloom filters have no false negatives, so skipping never drops a match.
  Blocks without a built filter, edge blocks, shorter suffixes, regexes and
  multi-prefix lookups scan exactly as before.

No index format, merge or query-semantics change: the filter is derived from
immutable decoded bytes and dies with its cache entry.

## Validation

Unit tests pin exact results and scan-budget errors with and without skipping,
filter construction only after a full block scan, no false negatives, and
per-segment readers building none; the 677-query exact audit and paired HTTP
comparison measure the effect.
