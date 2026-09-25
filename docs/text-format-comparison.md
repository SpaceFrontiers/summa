# Text index formats and memory: Summa versus Tantivy

The tables below describe the preserved Rounded/POS3 baseline. The new
[compact text and byte-norm implementation](compact-text-format.md) is opt-in:
separate posting headers, smaller position cursors, POS4 checkpoints and
descriptors, and query-local normalization lookups. Existing codec defaults
and old segment semantics remain unchanged. Rebuilding with the corresponding
options is required to measure the new representations.

## Scope

Same 5,032,104-document Wikipedia corpus and document order, RGB disabled,
Summa Rounded postings/POS3 positions, and Tantivy 0.26. Compare file bytes,
resident pages and anonymous memory separately. A larger file can increase
cache and bandwidth costs; file size alone does not establish a latency cause.

## Format differences verified in source

| Component         | Summa current fixture                                                                                                                  | Tantivy 0.26                                                                |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Document gaps     | Up to 127 gaps plus first ID; widths rounded to 0/8/16/32 bits                                                                         | 128 strictly increasing deltas, gap-minus-one, exact-width BitPacker4x      |
| Term frequencies  | Direct values at rounded widths                                                                                                        | Frequency-minus-one in full exact-width blocks                              |
| Posting metadata  | 8-byte block header, 16-byte L0 entry, L1 bounds, position cursors and ratios; 44-byte term footer (no impact records in this fixture) | 12-byte skip record per full positioned block, plus VInt skip-length prefix |
| Posting tail      | Rounded partial block and ordinary directory entry                                                                                     | VInt document gaps and frequencies                                          |
| Position payload  | Rounded widths in blocks of up to 128                                                                                                  | Exact-width BitPacker4x full blocks, VInt tail                              |
| Position metadata | 4-byte block header, 12-byte physical/logical directory entry, 16-byte term footer                                                     | One width byte per full block, VInt block-count prefix                      |
| Scoring lengths   | Two-byte saturated lengths                                                                                                             | One-byte quantized norms and a 256-float BM25 lookup table                  |

These are the formats actually used here. Summa also has explicit Packed,
Pfor and Simd4x options; its default is not its smallest codec. The existing
Simd4x experiment reduced total index bytes about 20% but made official top-10
2.6% slower on that measured build; see [posting codecs](posting-codecs.md).
A smaller representation must also have a competitive decoder and seek path.

Primary sources: Tantivy release source revision
`9e63fc508153ef770f9ff980c8fa2f11e8e2e6db`,
[compression](https://github.com/quickwit-oss/tantivy/blob/9e63fc508153ef770f9ff980c8fa2f11e8e2e6db/src/postings/compression/mod.rs),
[posting serializer](https://github.com/quickwit-oss/tantivy/blob/9e63fc508153ef770f9ff980c8fa2f11e8e2e6db/src/postings/serializer.rs),
[position serializer](https://github.com/quickwit-oss/tantivy/blob/9e63fc508153ef770f9ff980c8fa2f11e8e2e6db/src/positions/serializer.rs),
[BM25 lookup](https://github.com/quickwit-oss/tantivy/blob/9e63fc508153ef770f9ff980c8fa2f11e8e2e6db/src/query/bm25.rs).
Summa: [posting layout](posting-codecs.md),
[position stream](../summa-core/src/structures/postings/positions_v2.rs),
[length columns](../summa-core/src/segment/chunk_map.rs).

## Admission also affects residency

On a structural-validation cache miss, Summa checks every posting/position
block header. Headers are interleaved with payload blocks, and a Rounded block
is smaller than a 4 KiB page. Walking all headers therefore touches essentially
every payload page in that term's stream, even when later ranking would prune
most blocks. The bounded proof cache avoids repeating validation on a hit, but
does not undo pages brought into the process working set during admission.
This is a source-based explanation of additional residency, not a measured
fraction of the latency gap.

Tantivy's position reader splits a compact width directory from payload and
advances through widths to skip payload bytes. Summa's opt-in compact formats
now move structural metadata into separate directories, allowing admission
without scanning payload headers. Decoded content checks remain in the owning
block reader. Validation still rejects invalid widths, offsets, framing and
document order; skipping it could turn damaged bytes into incorrect results.
See [posting admission](../summa-core/src/structures/postings/posting/validation.rs),
[position admission](../summa-core/src/structures/postings/positions_v2.rs),
[bounded proof ownership](../summa-core/src/structures/postings/posting/reader.rs),
and [Tantivy's position reader](https://github.com/quickwit-oss/tantivy/blob/9e63fc508153ef770f9ff980c8fa2f11e8e2e6db/src/positions/reader.rs).

## Measured process residency

Fresh processes execute three complete passes of each command in order: top-10,
top-1000, top-100 plus count, count. Snapshots retain `/proc` smaps, rollup,
status and mappings. Values below are after the third count pass, hence include
pages retained from preceding commands. They are stable across all three count
passes. This is a separate residency run, not the latency-run peak RSS.

| Official workload, MiB | Selected Summa | Tantivy | Difference |
| ---------------------- | -------------: | ------: | ---------: |
| Total RSS              |        1007.29 |  574.24 |    +433.05 |
| Position-file RSS      |         664.19 |  343.77 |    +320.42 |
| Posting-file RSS       |         276.88 |  180.53 |     +96.34 |
| Length/norm-file RSS   |           9.60 |    4.80 |      +4.80 |
| Term-dictionary RSS    |          32.06 |   38.76 |      -6.70 |
| Anonymous residency    |           6.16 |    0.78 |      +5.39 |

Position pages explain **74.0%** of the total RSS difference; posting pages
explain another **22.2%**. These files together explain **96.2%** of the gap.
Anonymous residency includes allocator/runtime/stack memory; it is not a precise
heap-allocation attribution. Locked memory is zero throughout. The original
Summa build has essentially the same index-file residency as the selected build.

Standalone terms have no position-file residency: selected Summa RSS is
341.52 MiB versus 230.24 MiB for Tantivy, with the same 276.88/180.53 MiB posting
residency and 5.14/0.56 MiB anonymous residency. Tantivy's standalone top-10
snapshot is smaller (217.55 MiB); later top-1000 brings more pages in. Do not mix
snapshots from different command boundaries when comparing engines.

## Measured file-byte decomposition

All 13 format-audit members verify. Component sums reproduce every term-range
byte in both indexes. Values are MiB (2²⁰ bytes); Tantivy's whole-file framing
adds 113 posting bytes and 104 position bytes beyond these term ranges.

| Component                            | Summa MiB | Tantivy MiB | Summa / Tantivy |
| ------------------------------------ | --------: | ----------: | --------------: |
| Postings total                       |  2101.873 |    1005.718 |           2.09× |
| Document-gap payload                 |   887.465 |     660.291 |           1.34× |
| Frequency payload                    |   561.974 |     295.261 |           1.90× |
| Posting headers/directories/footers  |   652.434 |      50.166 |      **13.01×** |
| Positions total                      |  2625.295 |    1764.826 |           1.49× |
| Position payload                     |  2285.306 |    1751.398 |           1.30× |
| Position headers/directories/footers |   339.989 |      13.429 |      **25.32×** |

Posting metadata explains **54.9%** of the posting-file size gap; position
metadata explains **38.0%** of the position-file gap. Summa's posting metadata
includes 193.18 MiB of L0 entries, 166.37 MiB of per-term footers, 96.59 MiB each
of block headers and position cursors, plus L1 and ratio bounds. No impact
records are present in this timed fixture. Position metadata includes 209.62 MiB
of logical/physical directory entries, 69.87 MiB of block headers and 60.50 MiB
of term footers. These counters describe the persisted fixture, independent of
whether any page happened to be resident during a query.

Summa has 12,660,057 posting blocks, **67.1% short**, and 18,316,784 position
blocks, **46.4% short**. The copy-preserving format permits short interior blocks;
per-block overhead and per-term footers are especially expensive for small
lists. Tantivy instead has compact skip records only for full posting blocks,
one width byte per full position block, and VInt tails. A redesign must retain
Summa's ability to copy compatible encoded runs during merge.

Reading the actual Rounded values gives hypothetical payload savings of
**577,974,979 bytes** for exact-width gap/frequency packing with applicable
minus-one coding, and **568,889,062 bytes** for exact-width positions. These
estimates preserve Summa's existing block boundaries and exclude replacement
metadata; they are not a proposed complete format, a Tantivy byte-for-byte
encoding, or a query-speed prediction. Tail VInt estimates are also retained
in the audit and must not be added to overlapping exact-width estimates.

The indexes retain their documented upstream analyzer differences: Summa has
177 extra terms, 220 extra document occurrences and 247 extra positions out of
3.96 million terms, 586 million occurrences and 1.33 billion positions.
Tantivy's default analyzer excludes tokens of 40 bytes or more; Summa retains
them. All timed-query counts still match. See [analyzer differences](search-benchmark-game.md#analyzer-and-scoring-differences).
These tiny vocabulary differences do not explain gigabytes of format overhead.

## Quantization precision measured on this corpus

The arithmetic byte4 representatives match all 256 values in the pinned
Tantivy table. Applying that table to Summa's current saturated u16 lengths
would change **2,932,716 of 5,032,104 lengths (58.28%)**. Among nonzero lengths,
the mean rounding-down loss is **2.3456%**, with maximum **11.0726%**. There are
391,611 zero lengths, which remain zero. The norm payload would shrink from
10,064,208 to 5,032,104 bytes, saving **4.80 MiB**. These are length errors, not
BM25 score errors or a measured ranking/relevance change. A 256-entry BM25
component table removes repeated length arithmetic in the new opt-in scorer.
These baseline estimates are distinct from the latency and ranking measurements
of the [implemented byte-norm format](compact-text-format.md).

## Priorities from the evidence

The September 16 upstream follow-up supports keeping compression, skipping and
validation costs separate:

- PISA's [SIMD-BP128 codec](https://github.com/pisa-engine/pisa/blob/master/include/pisa/codec/simdbp.hpp)
  uses 128-value compression blocks. Its [query documentation](https://pisa.readthedocs.io/en/latest/query_index.html)
  exposes separate WAND metadata, fixed or variable bound blocks, and several
  ranked traversal algorithms. Compressed payload size alone does not select the
  best traversal or bound partition for Summa.
- IResearch's [format owner](https://github.com/iresearch-toolkit/iresearch/blob/master/core/formats/formats_10.cpp)
  uses packed full blocks and variable-integer tails; tail document deltas can
  carry a frequency-one flag. Its reader checks format versions and block size,
  but deliberately avoids full posting/position checksum scans at open. This
  supports cheap structural admission, not removing corruption handling.

These are source observations and follow-up directions, not new measurements
against either engine. Summa still needs bounded copy-compatible tails and
measured decoding paths before adopting those representations.

1. Compact posting/position metadata and separate admission metadata from
   payload pages. This addresses the largest measured RSS costs and a large
   share of file overhead.
2. Evaluate exact-width payloads and compact tails with the existing shared
   decode kernels. Keep copied runs and bounded seeks; smaller files alone did
   not make the earlier Simd4x trial faster.
3. Evaluate quantized norms with a 256-entry BM25 component table. This has a
   modest direct memory benefit and a potentially larger CPU benefit.

The traversal implementation itself did not change these defaults. A subsequent
opt-in implementation now provides compact directories and byte norms. The
requirements below motivated that implementation; its concrete layouts and
mixed-format merge rules are in [the format design](compact-text-format.md).

## Quantized length requirements

Tantivy 0.26 uses a monotone 256-value table, rounding down, and a 256-float BM25
length-component cache. The CPU opportunity is the lookup, in addition to a
one-byte-per-document column. A Summa experiment must compare both precision
and query speed against its own exhaustive scorer using the same norm values.
It cannot keep the current byte-exact score oracle as the expected new ranking.

The authoritative writer/merger must persist an explicit norm encoding and
quantizer version. Readers reject unknown encodings before scoring. Existing
u16 indexes retain their exact/saturated semantics; do not silently reinterpret
bytes or derive a corpus-sized unbudgeted heap cache at reader open. Keep
quantized bytes mmap-backed and evictable. Original total token counts and
missing-value semantics remain explicit. Actual chunk lengths and position
coordinates remain exact; quantized scoring norms are not chunk geometry.

Every block/group/impact pruning bound must use the same decoded representative
length as the scorer. Reusing the old exact-length lower bound while rounding
the scoring length down can underestimate scores and drop valid top-k hits.
Rebuild bounds with the new indexed representation, and copy compatible encoded
blocks on ordinary merges. Mixed versions require explicit compatibility rules.

Gate with exact pruned-versus-exhaustive results on the new quantized fixture,
old-format regression/byte-copy tests, cross-segment scoring, deletions,
missing/multivalue fields, native/sync/portable execution and identical-corpus
ARM/x86 latency plus memory. Report ranking overlap with the exact-length fixture
as a precision tradeoff. Do not change the default from one synthetic benchmark.

## Compact positions requirements

Summa POS3 records a 4-byte header and 12-byte physical/logical directory entry
per block, plus 16 bytes per term. The directory allows short interior blocks
so merges copy source bytes. Tantivy stores one width byte per full block and
uses VInt tails; direct adoption must not restore full-payload merge rebuilding.
An extent directory can describe copied runs of full blocks plus short tails,
with bounded coarse offsets and local width scans. This needs a new version,
checked overflow/corruption handling and measured seek/merge costs. Quantify the
current overhead first before committing to an alternative directory design.

## Reproducibility

The [verified evidence archive](benchmark-results/traversal-2026-09-15/README.md)
contains all scripts, exact input/binary hashes, 401 memory-audit members,
13 format-audit members, component-sum checks, pinned Tantivy source/license,
and all diagnostic errors and corrections. Neither audit changes an index.
