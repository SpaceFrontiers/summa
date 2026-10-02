# Entry offsets for cached dictionary blocks

September 28, 2026. Status: implemented for shared-cache term dictionaries.
Broad wildcard and regex scans run 2.1–2.3× faster on the 10M-document
benchmark, for 3% more term-cache bytes. Under 32 clients wildcard top-k
throughput doubles, overtaking Luxir (1.11×, from 0.56×;
[campaign](benchmark-results/next-2026-09-29/README.md)).

## Problem

Wildcard, regex and prefix queries scan every term under a literal prefix.
With the [process-wide term cache](term-dictionary-cache.md) the blocks are
already decompressed, but every rejected entry still parses its `TermInfo`
(a tag and three to five varints) only to find where the next entry starts.
On the 10M index these families trailed Luxir: wildcard 0.58×, regex COUNT
0.62×. [Suffix filters](dictionary-suffix-filters.md) skip whole blocks only
for four-byte endings; many patterns end in one letter or are regexes.

## Design

A shared-cache block can carry the offset of every entry in its entry stream
(`u16`, plus the stream's end: about 2 bytes per term against 21.5 bytes of
block data on the benchmark dictionary). A scan of such a block reads each key
at its offset and jumps to the next entry without touching the value; values
are parsed only for accepted keys. Keys, matching, in-range tests, scan
budgets and results are exactly those of the front-coded scan (the entry loop
is one generic function, specialized for the two forms).

- **When.** A scan that reaches a block already cached (a hit) and without
  offsets records them as it goes. If that scan covers the whole block, the
  offsets are attached. Blocks are thus indexed on their second scan, so a
  cache too small for the scanned range never pays for recording, and blocks
  that suffix filters skip, or that point lookups touch, are never indexed.
- **Accounting.** A cached value's retained bytes must not change while it is
  cached: insertion and eviction both read them. Offsets are therefore never
  added in place: the scan builds a copy of the block (with its suffix filter)
  carrying them, and `SharedBlockCache::replace` swaps it in, subtracting the
  old entry's bytes, adding the new one's and evicting as usual. Readers
  holding the old block are unaffected. A block evicted meanwhile is not
  re-admitted.
- **Scope.** Only the process-wide cache records offsets, like suffix
  filters; per-segment caches (merges, standalone searchers,
  `term_cache_process_bytes = 0`) scan the front-coded form. Blocks over
  64 KiB (block targets above the 16 KiB default) cannot use `u16` offsets:
  their scans are counted in `SSTableStats::scan_index_bypasses` and the first
  one logs a warning.

No index format, merge or query-semantics change.

## Measurements

x86 (n2, Cascade Lake), 10M-document index, one pinned core, same process
per arm, A B A B, sums of per-query medians; every result digest is identical.

| Family               |   Before |   Entry offsets |
| -------------------- | -------: | --------------: |
| wildcard TOP_10      |   7.9 ms |  3.8 ms (0.48×) |
| wildcard_scan TOP_10 | 106.8 ms | 45.7 ms (0.43×) |
| regex TOP_10         |  24.5 ms | 11.1 ms (0.45×) |
| prefix3 TOP_10       |   0.2 ms |          0.2 ms |
| wildcard COUNT       |  40.0 ms | 36.0 ms (0.90×) |
| regex COUNT          |  32.7 ms | 20.0 ms (0.61×) |
| prefix3 COUNT        |   3.9 ms |          3.8 ms |

prefix3 scans one or two edge blocks per query, which rarely complete. Counts
over frequent-term unions spend most of their time in postings, not the
dictionary.

**Working set.** Cycling all 59 scan queries in one process under the default
256 MiB budget: the cache holds 236.8 MB before and 244.3 MB with offsets; a
steady round takes 148 ms before and 72 ms after; the first (cold) round takes
1.6% longer (966 → 982 ms) for recording and copying.

**Materialized keys (rejected).** A prototype that also stored every key
decoded (keys, `u16` ends and value offsets, about 0.87× the block) and
binary-searched the prefix start was 1.3× faster still on these scans, but
nearly doubles every scanned block: the 42 broad wildcards alone would need
about 390 MB, beyond the default budget, where eviction would turn every scan
into a miss.
