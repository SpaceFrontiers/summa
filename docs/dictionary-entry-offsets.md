# Entry offsets for cached dictionary blocks

## Problem

Wildcard, regex and prefix queries scan every term under a literal prefix.
With the [process-wide term cache](term-dictionary-cache.md) the blocks are
already decompressed, but every rejected entry still parses its `TermInfo`
(a tag and three to five varints) only to find where the next entry starts.
On the 10M index these families trailed Luxir: wildcard 0.58×, regex COUNT
0.62×. [Suffix filters](dictionary-suffix-filters.md) skip whole blocks only
for four-byte endings; many patterns end in one letter or are regexes.

## Design

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

