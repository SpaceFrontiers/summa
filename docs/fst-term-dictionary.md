# FST term dictionary: merge cost and scan feasibility

September 27, 2026. Status: **measured; FST not adopted for scans.** Merging
is not the obstacle — the FST merges faster and in bounded memory — but
enumerating an FST subtree is 4.4× slower than scanning decoded SSTable blocks,
so it does not close the broad-wildcard gap. No index format, writer or reader
changes with this note.

## Question

The broad-wildcard family (`h*band`, `a*nder`, … — one leading letter plus a
four- or five-byte suffix) walks an entire first-letter subtree of the text
field's dictionary per query. Luxir answers that family about 5.75× faster
than Summa. The Summa profile is dominated by decompressing SSTable blocks
(zstd, ~38% of samples) and decoding entries (~23%).

An FST dictionary stores keys uncompressed-but-shared in a mapped automaton,
so enumerating a subtree needs no block decompression. It was previously set
aside for two reasons:

1. The September 25 forward-FST probe was **slower** (1.95–41×).
2. Concern that FST dictionaries cannot be merged efficiently.

This note separates those concerns and measures both on the real corpus.

## Why the earlier probe was slow

That probe added an FST _next to_ the SSTable and fetched every matched value
with an SSTable point lookup — one block decompression per match, plus the
subtree walk. The dictionary did not get cheaper; it got a second index. A
replacement design must address values directly from the FST output:

- FST: key → byte offset into a value stream (monotonic, so outputs are small
  deltas along shared paths).
- Value stream: concatenated serialized `TermInfo`, the existing encoding.

A matched key then costs one `TermInfo::deserialize` at a known offset, and
rejected keys cost nothing beyond the automaton step.

## Why merging is not the obstacle

Segment merge already produces terms as one sorted stream
(`segment/merger/terms.rs` `MergedTerms`, a k-way heap over source
dictionaries) and writes them with `SSTableWriter::insert(key, term_info)`.
`fst::MapBuilder` has the same contract: strictly increasing keys, streaming
insertion, output written to any `Write`. Memory is bounded by the builder's
node registry and the unfinished-node stack (maximum key length), not by the
number of terms. So a forward FST writer is a drop-in replacement at the
existing call site, and FST sources can feed the same heap (or
`fst::map::OpBuilder::union`, which is the same k-way merge).

The genuinely hard case was the **reverse** dictionary sidecar (reversed keys
for suffix queries): its order is unrelated to the forward merge order, so the
earlier design needed a global external sort per merge. That problem does not
apply to a forward dictionary. If a reverse set is ever needed, it can also be
merged as a streaming union of the sources' reverse sets (each is sorted in
reverse-key order), filtered by membership in the new forward FST — no
external sort — but it is not part of this proposal.

## Measurement on the production dictionary

The probe ([source](benchmark-results/dict-scan-2026-09-27/fst-probe-source.rs.txt),
[results](benchmark-results/dict-scan-2026-09-27/fst-probe.json)) streams the
production `.terms` file (503,593,473 bytes, 51,874,769 keys, all fields) into
eight overlapping segment dictionaries — 70% of keys in one segment, the rest
in two or three — once per format. Each mode runs as its own process pinned to
CPU 0 of the 8-vCPU x86 validation host, so peak RSS is attributable. Both
formats carry the same `TermInfo` encoding; the FST maps key → byte offset into
the value stream.

| Merge of 8 segments → 51.9M keys | SSTable (current) |                  FST + value stream |
| -------------------------------- | ----------------: | ----------------------------------: |
| CPU                              |            62.3 s |                          **41.9 s** |
| Peak RSS above loaded sources    |          +870 MiB |                         **+23 MiB** |
| Output bytes                     |          503.6 MB | 800.2 MB (273.3 FST + 526.9 values) |

The merged SSTable has exactly the production file's size, consistent with the
probe writing the canonical format. Its merge memory was O(terms):
`SSTableWriter` kept a 16-byte bloom hash pair per key until `finish`. That is
now fixed for merges, compaction and text reordering
(`BloomSizing::PowerOfTwo`): their filters are sized
`bits_per_key × next_power_of_two(keys)`, so every such pipeline writes
identical bytes, and when the sources' total key count bounds the output
without the presized filter outgrowing the hash buffer (total × bits per key ≤
largest source × 64 bits) they build the filter while writing and fold it at
`finish`. Heavily overlapping many-way merges, where the total overstates the
output, still buffer hashes. The rewritten blooms are up to 2× larger, with
correspondingly fewer false positives; segment builders keep the exact frozen
v5 sizing. The FST builder's memory is independent of term count. Initial segment writing (with
1.3× entry duplication) takes 62.7 s for SSTable and 73.2 s for FST.

Scans use the text field's 42 broad-wildcard patterns and two three-letter
prefixes; all 44 return identical `TermInfo` sequences and identical scanned
counts in every mode. Values are medians of 21 single-thread CPU-time samples
after three warmups:

| Sum over 42 broad wildcards (54k–830k terms each)             |    CPU ms |
| ------------------------------------------------------------- | --------: |
| SSTable, 1,024-block / 16 MiB cache                           |   1,319.9 |
| SSTable, every touched block retained (12,503 blocks, 207 MB) | **850.1** |
| FST range stream + suffix check + offset read                 |   3,753.3 |

FST enumeration costs about 145 ns per visited term versus 33 ns for a decoded
SSTable block, 4.4× slower, and it is also slower for the small prefix
controls. FST point lookups are 12× faster than the SSTable's (542 vs 6,506 ns
over 179,695 random keys), but term lookups are not the gap in any benchmark
family.

## Conclusion

- **Merging is solved** for a forward FST: same streaming contract as today,
  faster and bounded. Reverse sets can merge as streaming unions too.
- **An FST does not help this workload**: subtree enumeration is the cost, and
  a decoded SSTable block enumerates 4.4× faster.
- **Block decompression is the lever.** With blocks decoded, the 42 patterns
  average 20.2 ms of scan CPU, near Luxir's ~17 ms per request. Searches used
  a 256-block (~4 MiB) per-segment cache, so each broad wildcard decompressed
  its whole subtree again; the [process-wide dictionary cache](term-dictionary-cache.md)
  now retains them by default. See
  [the campaign report](benchmark-results/dict-scan-2026-09-27/README.md).
