# Search Benchmark, the Game: Summa comparison

## Protocol and current scope

The upstream suite at <https://tantivy-search.github.io/bench/> uses transformed
English Wikipedia and 962 queries. Pin upstream revision
`a7c75473e91746280c5f01e69bf594ece5fca560`; preserve corpus and query SHA-256 hashes.
The transformer lowercases and replaces non-ASCII letters with spaces for every
engine. Do not introduce engine-specific stop words, stemming, document filters,
query caches, query-specific tuning, or approximate candidate budgets.

The initial adapter is an example executable using the normal `IndexWriter`,
`MmapDirectory`, `Searcher`, and the native strict query parser. It translates
only the upstream line protocol. Malformed
syntax must fail explicitly. It stores the corpus ID, indexes positioned text,
and stores the numeric sort column as a fast field, matching the upstream schema.
All three engines force merge to one segment; search uses one CPU. Summa and
Tantivy index with four workers and a 2,000,000,000-byte builder budget. Lucene
retains its upstream eight workers (`availableProcessors`) and 1024 MiB writer
RAM buffer. These are different resource controls, not process RSS caps, and
initial index builds overlapped; no cross-engine indexing-speed claim is made.
Index sizes and peak RSS include each engine's own format overhead.

Summa' existing `search_with_count` returns **scored documents**, not exact hit
counts. Never report that counter as an exact count. `COUNT` and `TOP_*_COUNT` use
the existing `collect_segment` API with `CountCollector` and a `TopKCollector`
tuple. That API preserves exact counts and ranked results. Eligible collectors
can count plain terms and two-term unions separately from ranking; other text
queries stream complete membership. Custom and position collectors retain
complete callbacks. Ranked operations use the synchronous public search boundary.
`VERIFY` compares both ranked search and exact-count collection at top-10,
top-100 and top-1000 against an explicitly exhaustive collector outside timing.
Native async and WASM share the query implementations;
the benchmark adapter itself requires native sync and changes no persisted format.

## Filtered collection batches

`PredicatedScorer` can preserve an underlying scorer's document and score
windows through deletion and query predicates. The driver must also advertise
`Scorer::supports_filtered_windows`, whose default is false. Eligible Boolean
composites opt in because batching amortizes traversal of multiple children;
leaf scorers retain scalar filtering. The wrapper checks set bits in document
order, preserves MUST score addition order, and leaves the cursor on its next
eligible match. It borrows the collector's bounded buffers and adds no scratch.
The existing score collector still allocates 16 KiB per query using score windows.
Position collection retains scalar traversal. No encoding or ranking changes.

The fixed-mask performance experiment is separate from the deletion-free
upstream workload: the same immutable index is filtered by `doc_id % 8 != 0`
through the public predicate wrapper. Mask setup is outside timing; dictionary
COUNT shortcuts are disabled because physical document frequency includes the
masked rows. Only exhaustive COUNT and top-100 plus exact count are timed.
This isolates filtering costs without changing any index payload or claiming
masked ranked-search or deletion-publication performance.

## Measurement plan

Run Summa and upstream Tantivy 0.26, then Lucene 10.4.0 where its build is
available, on the same dedicated Linux machine. Pin compiler and release flags
(`-C target-cpu=native` for both Rust engines). Keep upstream query order,
commands, warmup and repetitions, retaining all samples as well as its minimum
statistic. Also report per-query median ratios and grouped p50/p95/p99, so one
outlier cannot become an overall speed claim. Timing includes parse and pipe
round-trip, excludes index open after warmup, and excludes document hydration.

Validate matching counts and small-corpus exhaustive top-k before performance
claims. Differences in field-length quantization, BM25 scaling, and phrase
scoring must be reported. Run before/after on the same persisted index, fixture,
host, compiler, and flags without competing build/index work. Preserve raw logs,
resource measurements, revisions, dirty diff, and errors. Cold-cache measurements
are separate from the upstream warm-cache suite. Synthetic fixtures validate
behavior but cannot substitute for the full corpus.

## Optimization gate

Investigate measured slow families in the owning query/posting component and
compare competitor implementations. Document the invariant and cost model
before substantial changes; add a failing behavior regression for correctness
bugs and an exhaustive ranking oracle for pruning changes. Keep encoded formats
unchanged unless explicitly designed otherwise. No architecture-sensitive
default changes without x86/aarch64/scalar evidence. Record measured findings in
[the performance review](search-performance-review.md).

## Streaming exhaustive collection

The baseline public exhaustive `collect_segment` entry point sent a huge
limit to the ranked scorer. For a text union, MaxScore consequently accumulated
and sorted every match before the outer collector saw any document. Counting
therefore used O(matches) temporary memory even though `CountCollector` itself
is constant space; top-k plus count needs only O(k) retained results.

The implementation stays in `core/query/collector`: request the existing complete
text-membership scorer, already used for required Boolean clauses, when exhaustively collecting ordinary text terms and pure unions. Tuned,
proximity, opaque, and vector query types retain their existing execution path. It must enumerate every live match with the same score
and ordinal semantics; no score floor or approximate cutoff is permitted. Keep
the old large limit for non-text query compatibility. Text unions then use
posting cursors plus the caller's collector, with no all-hit ranked heap.
`CountCollector` can additionally declare that it does not consume scores;
collector tuples still request scoring when any child requires it. This changes
no format, ranking default, schema, or RPC semantics. Regression coverage must
check bounded allocation, counts, tuple top-k, deletions, and async/sync parity.
The 32,000-document union regression reduces total allocated bytes from 746,157
to 7,490 on the benchmark x86 host; this is allocation volume in that fixture,
not whole-process peak RSS. Full-corpus timing is reported separately.

## Runtime controls

The adapter's full usage string is:

```text
search_benchmark_game <index|serve|validate-queries> <path> [--exhaustive]
  [--indexing-threads N] [--indexing-memory-bytes N]
  [--posting-codec rounded|packed|pfor|simd4x]
  [--term-cache-blocks N] [--term-cache-bytes N] [--term-dict-block-bytes N]
  [--posting-ratio-bounds] [--posting-impact-bounds] [--no-background-merges]
```

The ratio-bound option `--posting-ratio-bounds` (also `summa-tool index` and
`IndexConfig.posting_ratio_bounds`) is off by default. New segments carry
score-independent block metadata; queries detect it automatically. Existing
segments keep their original bounds, and compatible merges preserve either
representation. See the
[ratio-bound design](posting-codecs.md#experimental-ratio-bounds-2026-09-13-follow-up).
This option does not change exact counts, ranking semantics, or posting codecs.
`--posting-impact-bounds` additionally writes per-block frequency/length impact
envelopes on multi-block lists and implies ratio bounds
(`IndexConfig::effective_posting_bounds`). Both are indexing-time options: they
apply to new segments only, and a merge never upgrades old blocks. An index
opened with bounds enabled whose segments lack them logs this once.
`--no-background-merges` installs `NoMergePolicy` for the build so the only
merge is the explicit final `force_merge`; the default tiered policy may run
intermediate merges during ingestion. Both produce one merged segment; use the
flag when ingestion memory or wall-clock must exclude background merge work.

The adapter accepts `--indexing-threads N` and `--indexing-memory-bytes N`,
validated before index creation, using existing `IndexConfig` settings. Defaults
are four workers and 2,000,000,000 bytes. This is the existing builder flush
budget divided among workers, not a whole-process RSS cap; ingestion queues,
serialization, merge work and mmap residency also contribute to peak memory.
`--posting-codec rounded|packed|pfor|simd4x`
selects the existing posting codec through `IndexConfig::posting_codec` and its
shared parser; the default remains rounded. Codec experiments use a separately
rebuilt index and are labeled separately from same-index execution changes.
`--term-dict-block-bytes N` sets the uncompressed term-dictionary block target
for the build (512..=1048576, default 16384), the same
`IndexConfig.term_dict_block_size` that `summa-tool index` exposes.
`--term-cache-blocks N` exposes the existing per-segment dictionary-block cache
capacity (0 disables it, maximum 65,536 enforced by `IndexConfig` at open,
unchanged default 256). This budgets
decompressed metadata blocks, not query answers; actual heap bytes depend on block
sizes and must be reported alongside latency. A separate 1024-block experiment
uses the same binary and index; it does not change the main run or production
default. Do not infer a universal cache setting from this repeated workload.
`--term-cache-bytes N` adds the optional per-segment byte cap on retained
decompressed dictionary blocks (`IndexConfig.term_cache_budget_bytes`; 0
disables retention entirely). It is applied together with the block cap.
Posting/position validation caches were removed when normal query reads began
trusting immutable writer output. The former `--posting-validation-cache-bytes`
option is no longer accepted; older measurements that mention it describe their
historical source snapshots.

## Competitor implementation evidence

Tantivy 0.26's [term weight](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/query/term_query/term_weight.rs)
uses document frequency for a deletion-free term count, a buffered docset loop
when scores are unnecessary, and single-scorer block WAND for ranked terms.
Its [Boolean weight](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/query/boolean_query/boolean_weight.rs)
separates score-free unions from ranked block WAND. These are general query and
collector distinctions, not special cases for this query list. Summa should
preserve the same distinction between exact membership and competitive scoring
while reusing its own posting codecs and matching semantics.

## Ranked term plan

Ordinary non-chunked terms retain streaming traversal. An experiment reused the
existing block-MaxScore executor for multi-block standalone terms with k below
document frequency. It was exact, but the broad 714-term supplement exposed the
cost of loose bounds and window bookkeeping. Even after reducing single-term
scratch writes, streaming traversal was 1.11× faster for TOP_10 and 1.19× faster
for TOP_1000 on the same cloud fixture. The final implementation therefore
retains the original term execution policy alongside the improved shared
posting decoder, seek and planner hints. Chunked and multi-term plans keep their
existing owning executors. Required/excluded clauses expose complete membership;
BM25 parameters, global statistics, ties, offsets and live-row filtering remain
unchanged. No policy threshold is tuned to particular benchmark terms.

## Exact term-count shortcut

The pilot confirms deletion-free single-term `COUNT` spends time enumerating
postings that Tantivy counts from document-frequency metadata. Allow a collector
to accept a known exact count through an optional `collect_count` hook (default:
decline without side effects). Only use it for a score/position-free collector,
a decomposed text term, a non-chunked indexed field, and a reader with no deleted
rows. Existing dictionary metadata is then the exact document count. Reuse the
reader's `text_doc_freq` method without loading a posting payload or allocating
its skip directories. Only accept a positive dictionary frequency as the
shortcut: zero can mean missing metadata and must retain normal fallback. Missing
postings retain normal fallback semantics; chunks and deletions always enumerate
live documents. Tuples retain ordinary collection unless they explicitly support
bulk counts, so no collector can be partially updated. No estimates or cached
query answers are used. Validate metadata counts against enumerated counts,
including deleted rows and repeated values.

For an ordinary Boolean query with exactly one required text term, no exclusions
and only optional text terms, the required term alone determines membership.
Expose this equivalence from the Boolean owner through a count-only query hook;
reuse the collector's same dictionary-frequency checks. Do not alter ranked
decomposition or optional score contributions. Opaque, tuned and positional
scoring combinations decline this shortcut, preserving validation and execution
semantics. A lazy-reader regression must prove neither required nor optional
posting payloads are read, while a declining collector still receives all IDs.

## Saturated block-bound correctness finding

Reviewing the ranked-term gate exposed an existing bound issue outside this
benchmark: packed block/group maximum TF saturates at 65,535, while postings and
the list footer retain 32-bit TF. A saturated value is not an upper bound for a
larger actual TF. Decode saturated packed maxima conservatively using the list's
full maximum; retain the encoded bytes unchanged. This applies to every text
MaxScore query, including unions. Test a TF above 65,535, group bounds, and
byte-identical serialization before enabling broader use of block pruning.

## Canonical BM25 arithmetic

A 384-document regression reproduces a deterministic ranking mismatch: at average
length 3, TF 2/length 3 and TF 1/length 1 tie under the shared `Bm25Params` formula,
but the block executor's rearranged precomputed coefficients put document 1 ahead
of document 0. The block executor must use the same `Bm25Params::score` arithmetic
as exhaustive term scoring, including its legacy zero/missing-length fallback.
Keep block traversal/pruning and batching; remove the alternative score formula
and unused coefficients. This can increase arithmetic per decoded block, so
measure it together with pruning. Correct document ordering is the gate, and
single-term regression scores must now agree exactly, not within an epsilon.

## Supplemental standalone terms

The official suite has only one `term` query (`the`) and one `two-phase-critic`
query. Always report family sample sizes; neither category estimates a broad
query distribution. `scripts/search_benchmark/supplemental_terms.py` extracts all
714 distinct ASCII terms from the pinned suite, without selecting by observed
speed. Benchmark those separately on the same full index, using the same
before/after binaries and upstream driver (`QUERIES` override), 10-second
warmup and five repetitions for all engines. Verify exact counts and all three
ranked depths first. These results supplement, and never replace or get pooled
into, the official 962-query results.

## Analyzer and scoring differences

Reference engines keep the analyzers in their upstream adapters. Tantivy's
[default analyzer](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/tokenizer/tokenizer_manager.rs)
includes `RemoveLongFilter::limit(40)`. Lucene's adapter uses an empty stop list
with [StandardAnalyzer](https://github.com/apache/lucene/blob/releases/lucene/10.4.0/lucene/core/src/java/org/apache/lucene/analysis/standard/StandardAnalyzer.java),
whose default maximum token length is 255 (longer tokens are split). Summa'
simple tokenizer retains the transformed ASCII tokens without those limits.
These defaults can affect vocabulary, field lengths, index size, and scores.
Do not claim identical BM25 rankings across engines or introduce a Summa-only
corpus filter to improve timing. Exact hit counts are compared on every measured
query, and Summa ranking is checked against its own exhaustive scorer. Field
length quantization also remains engine-specific. Summa and Tantivy use
BM25 k1=1.2, b=0.75; the pinned Lucene adapter explicitly uses k1=0.9, b=0.4.
Keep that upstream configuration and disclose it: the Lucene comparison follows
the benchmark protocol, but is not a comparison of identical scoring functions.

## Selective conjunction driver

The generic Boolean cursor advances every required child after each match and
seeks every child before learning the next selective candidate. This can make an
expensive phrase enumerate intervening phrase hits that a selective required term
would skip. Advance/seek only the required child with the smallest known size
hint, then use the existing intersection alignment to seek the others. Preserve
child storage and score summation order. Treat zero hints on nonempty cursors as
unknown, expose a phrase's minimum posting cardinality as its conservative hint,
and saturate summed disjunction hints rather than overflow. Native term cursors
must expose their posting cardinality too; otherwise an expensive phrase with a
known cost can incorrectly be preferred over a rare term reporting zero/unknown.
Retain the existing posting frequency as a constant-size cursor hint, including
through complete-membership wrappers; it is a planning cost, not a live hit count. No approximate
membership or two-phase query semantics are introduced. Test skipped expensive
advances, exclusions, seeks, score identity, and overflow before benchmarking.

## Preserve text score accumulation order

The full-corpus gate found a top-1000 mismatch for `west palm beach florida`.
Per-term BM25 parity alone is insufficient: window MaxScore accumulates terms in
changing bound order, while complete text membership accumulates in query order.
Floating-point addition can therefore split document ties differently. Preserve
input term order independently of traversal order. For multi-term text windows,
retain per-term contributions in bounded scratch and sum surviving candidates in
input order before heap insertion; single-term windows need no extra scratch.
The maximum extra space is 64 terms × 4096 IDs × 4 bytes (1 MiB), independent of
corpus size. Partial-score pruning must allow for rounding in differently ordered
sums using a term-count-relative margin, including global partitioning. This
keeps the public f32 formula and complete scorer unchanged; it trades bounded
scratch and final candidate additions for deterministic ranking. Measure that
cost in the full run; never tolerate changed document identities in verification.

## Two-phase required-clause traversal

The baseline required-phrase query takes roughly 529 ms versus Tantivy's 27 ms.
A selective lead alone still asks each phrase cursor to find its next _verified_
phrase, potentially decoding positions beyond the conjunction's next candidate.
The shared `Scorer` contract now has opt-in candidate advance/seek and exact
confirmation. Default implementations remain exact and require no wrapper changes.
The Boolean owner uses these methods only for required children: align their
candidate document IDs, confirm every requirement, then expose the document and
its unchanged score/positions. Failed confirmation advances the lead and retries.
Optional and excluded clauses keep exact iteration. Phrase scorers can use their
existing posting intersection as the candidate stream and their existing position
checker as confirmation. Chunk-folding wrappers retain exact default traversal.
Ordinary `DocSet` calls must always return verified matches, including when mixed
with candidate calls. Confirmation must be repeatable, respect cancellation, and
never expose stale phrase frequency. Scratch and candidate retention remain
unchanged: one cursor position and reusable positional buffers, no candidate heap.

## Existing codecs and possible metadata work

The full-corpus codec experiment rebuilds separate `packed` and `pfor` indexes
with the same adapter, schema, corpus, four workers, and indexing memory budget.
It repeats all five official commands with the same release binary and validates
every query first. This is a rebuilt-index comparison, distinct from the main
before/after comparison on identical persisted bytes. No codec default changes
follow from one host. Rebuilding may also affect document layout, so isolate
individual codec CPU costs with production-block benchmarks before attributing
all query timing changes to compression alone.

A separate proposal worth measuring is tighter block score metadata. Summa
currently combines maximum TF and minimum length, which can come from different
documents. That is conservative but can substantially overestimate a block's
best BM25 score. Lucene's
[competitive impact accumulator](https://github.com/apache/lucene/blob/releases/lucene/10.4.0/lucene/core/src/java/org/apache/lucene/codecs/CompetitiveImpactAccumulator.java)
retains competitive frequency/length pairs. Reusing that established model could
provide tighter bounds while allowing scoring parameters and global collection
statistics to change. This is an inference from the designs, not a measured
Summa speedup or an implemented format change. An experiment must first measure
bounds versus actual block maxima and decoding avoided; then budget metadata,
version the representation, preserve compatible block-copy merges, and verify
exact scores under deletions, mixed segments, global statistics, and custom
BM25 parameters. Keep the existing posting payload codecs. Do not freeze one
benchmark's scoring function into stored maximum scores.

The position stream is a separate compression opportunity: its existing 128-value
blocks round widths to 0, 8, 16 or 32 bits. On this corpus `.pos` accounts for
2,871,883,607 bytes (54% of the Summa index), so posting-codec experiments cannot
resolve most of that cost. A future position-codec experiment should reuse the
existing bounded exact-width and PFor kernels rather than add another bit packer.
Use an explicitly versioned stream with a per-block codec tag and preserve the
existing logical-start index, short interior blocks and position cursors. Legacy
rounded blocks should remain copyable during mixed-format merges; a new reader
must reject unknown versions/tags, and old readers must reject the new version.
Measure complete index size, phrase latency, decode CPU and metadata residency,
including merge/compaction byte identity and cancellation, before changing any
default. This is a proposal, not an implemented size reduction. The completed
[offline estimate](search-benchmark-results.md#offline-position-width-estimate)
measures its potential storage saving without writing a new format or measuring
query latency.

## Canonical union score scratch

The v9 run regressed ranked unions despite faster seeks and decoding. Canonical
input-order scoring cleared every term's contribution array across each complete
ID window, including absent slots. The implementation retains the same score
arrays plus one presence bit per term/slot, clears only presence words and sums
zero for absent contributions. Essential block visits and nonessential point
probes both set presence. Scratch is at most 1 MiB + 32 KiB for 64 terms and
4096 IDs; single-term queries retain their previous path. No score arithmetic,
term order, threshold, posting format or window-size default changes.

The dense and sparse M4 fixtures improved by 18.4% and 61.4% respectively; the
v11 full-corpus union TOP_10 run improved 1.59 times over v9. These are separate
measurements, not interchangeable speed claims. A repeated-window regression and
the complete corpus gate require identical score bits and ordered document IDs.

## Bounded score-free document batches

The v9 COUNT profile attributes most CPU time to individual term/Boolean cursor
calls after decoding and seeking improvements. The existing DocSet now offers
opt-in 4096-document membership windows (512 bytes), preserving forward-only
exact matching and leaving its cursor on the first match after the window. Term
cursors visit existing decoded blocks and retain their TF/position prefixes;
pure disjunctions combine child membership bits in the Boolean owner. There is
no new scorer or encoded representation. Unsupported children, phrases, chunks,
filters, conjunctions and exclusions retain scalar traversal.

The first full-corpus experiment also batched conjunctions and exclusions. It
improved union COUNT 6.9 times over v9 but slowed intersections by 8% and negated
queries by 38%. The accepted path keeps batching for supported pure disjunctions
and removes the experimental candidate-intersection API. Selective scalar and
two-phase traversal remain the default for required/excluded clauses.

Only score-free, position-free collection uses batches. CountCollector accepts
the population count; other collectors receive the same ascending IDs and zero
scores. Scoreful collection retains its existing path. Scratch is fixed per
active union depth, independent of corpus and hit count. Deadline checks surround
each window, so no partially verified window is collected after cancellation.
Tests cover nested membership and scalar fallbacks, next-document scores and
positions, all three codecs, declining collectors, deletions and cancellation.

## Single-term score scratch

Single-cursor MaxScore windows retain the existing mask, but overwrite each
matching score slot with zero-plus-score instead of clearing the whole dense
array before adding it. Only set mask bits may read scores; multi-term addition
order stays unchanged. This avoids writes to absent slots while preserving score
bits and bounded scratch. On matched M4 fixtures, dense latency fell from
353.01 to 344.06 µs and sparse latency from 15.599 to 4.778 µs (20 samples,
1-second warmup, 3-second measurement, native CPU flags and release LTO).

The 714-term cloud candidate improved over the preceding window implementation
by 1.34× for TOP_10 and 1.18× for TOP_1000, but restoring ordinary term streaming
was faster still. The final implementation keeps this scratch improvement for
other single-cursor executor uses and retains streaming ordinary terms, as
described above. The microbenchmark is not a claim about final standalone-term
query speed. No posting format or production default changes.

## Existing library decoder candidate

The packed full-index run saves 9.8% of total bytes but regresses ranked latency
by 32–42% and COUNT by 65% versus rounded on this host. Before inventing another
format, evaluate the existing
[`bitpacking::BitPacker1x` 0.9.3](https://docs.rs/bitpacking/0.9.3/bitpacking/struct.BitPacker1x.html):
it stores 32 integers as consecutive fixed-width bit representations. A separate
little-endian M4 probe matches an independent horizontal-layout oracle in 8,448
cases (widths 0–32, one to four blocks, 64 seeds), including decoded values and
output canaries. This supports layout compatibility for complete 32-value groups;
it is not a production decoder integration or a speed measurement.

A candidate can reuse 32-value groups and the existing bounded tail without
rewriting stored bytes. It must first validate actual production block bytes,
unaligned slices, every short tail, protected-page input extents, endian behavior,
PFor exception reconstruction and sync/WASM parity, then rerun same-index query
benchmarks. Multi-lane bit-pack layouts are not interchangeable with the current
horizontal payload. The final measured engine retains its validated decoder;
this probe adds no dependency or format change to Summa.

## Dictionary block-size and cache-byte experiment

The current dictionary writer targets 16 KiB of uncompressed entries per STB5
block. Existing full-corpus profiles spend substantial single-term COUNT time in
Zstd. Lucene's smaller term blocks motivate measuring smaller units of decoding;
we will first reuse Summa's Zstd codec and restart/index format.

A validated `SSTableBlockSize` value permits targets from 512 bytes to 1 MiB,
with the existing 16 KiB default. It is a flush target, not a maximum entry size;
individual entries and restart trailers can exceed it. The existing 64 MiB reader
limit still applies, and writers must refuse blocks they cannot reopen. The type
validates before writer-buffer allocation and exposes the byte count. No new
footer version is necessary: existing block addresses already store actual
compressed lengths and decoders do not assume the 16 KiB target. This must be
verified with old/new reader compatibility and a frozen default-output fixture.

`IndexConfig.term_dict_block_size`, the native and WASM builder policies, and the
existing merge/compaction/text-reorder owners carry the setting. Every term
writer uses the same SSTable writer. Compatible posting payloads remain copied;
term metadata already needs rewriting when posting offsets/doc frequencies change.
Indexing and merging CLI options must propagate the setting, including custom
builder configurations. Reorder operations that copy unchanged term dictionaries
continue to copy them.

An optional `IndexConfig.term_cache_budget_bytes` additionally bounds retained
decompressed blocks in the existing dictionary cache; the block-count cap remains
in force. `None` preserves today's policy and zero disables retention. An oversized
block remains readable but bypasses the cache. Cache entries retain no query
answers. Track retained bytes in constant time, charge before insertion, subtract
on eviction, and keep read hits free of LRU writes. Hash/deque metadata is separately
bounded by the block count; in-flight readers may retain evicted blocks, so report
RSS as well as retained cache bytes. No new answer cache or cache implementation
is proposed.

Compare 16 KiB and smaller block targets under the same **actual decompressed-byte
cap**, not just the same number of blocks. The initial experiment uses a 4 MiB
cap, with block-count limits sufficient to reach it for each target. Record the
larger block-index/FST metadata, index size, prefix behavior and cold/lazy-read
costs. Do not change defaults from the public benchmark alone. Default writer
bytes must match the frozen fixture; smaller-block output intentionally has new
block boundaries but identical term values and query results. Test boundary and
oversized entries, eviction/duplicate races, zero budgets, sync/async lookup,
prefix scans, merge/compaction/reorder and native/WASM policy propagation.

The cache audit also reproduced an existing bulk-prefetch bug: it expanded the
configured block cap to the entire table and read all compressed bytes even with
caching disabled. The fix keeps the legacy method name but warms only a bounded
leading range: at most 4 MiB of compressed input, no cache-cap expansion, and no
payload I/O with zero retention capacity. It stops before evicting entries to
make room for speculative prefetch, and report limited warmup in diagnostics.
A single decompression remains bounded by the existing 64 MiB safety limit;
this scratch is separate from retained cache bytes. Normal iteration still reads
all terms and propagates later I/O/corruption errors. The merge caller limits
concurrent prefetch operations to four; output ownership/publication is unchanged.

### Isolated dictionary fixture preparation

For the first block-size ablation, copy the already-built immutable benchmark
index into an exclusively created, unpublished fixture directory. Regenerate only
its `.terms` files with the existing `AsyncSSTableReader<TermInfo>` and canonical
`SSTableWriter`, using the pinned original Adaptive Zstd settings and a new block
target. Reject input with a trained dictionary. Compare every ordered key and
serialized `TermInfo` after reopening; hash every non-dictionary file to prove
posting bytes, document IDs, positions, norms and stored values are unchanged.
Copy root metadata last, only after verification. This is offline fixture setup,
not a production migration API or an alternative writer/publication protocol.
Failures leave an explicitly incomplete private fixture and never modify the
source index. Do not reuse an existing destination or delete unowned files.

The 16 KiB reconstruction is an additional byte-identity control. Old binaries
must read the new smaller-block fixture and agree on exhaustive ordered hit IDs,
score bits and exact counts. Compare 16 KiB, 4 KiB and 1 KiB targets with the same
4 MiB retained decompressed-byte cap and 8,192-block upper bound, selected from
storage units rather than the benchmark query set. Keep posting validation reuse
fixed at 256 KiB. Retain a legacy 256-block control to separate format effects
from cache policy. Normal build/merge/compaction/reorder propagation is covered
separately by permanent integration tests.

### Writer failure boundary follow-up

The initial oversized-block rejection occurs after serialization; a custom value
serializer can still grow the block before that check or return an error after
writing a partial value. Strengthen the same writer with a bounded `Write` view
for entry bytes (including reserved restart-trailer space), and poison the writer
after any insert failure. `finish` must refuse to publish a partial entry, and
subsequent inserts must fail. Reject excessive writes before buffer growth. This
is an ingest correctness/resource follow-up, not a search algorithm or format
change. Reproduce both an over-budget streaming serializer and an injected
mid-value failure first; retain default-output byte identity and validate native,
portable and WASM behavior. Frozen dictionary-v1 timings precede this follow-up
and must remain explicitly identified as such until the follow-up is measured.

## Proposed frame-size-aware bounded decompression

The dictionary experiment exposed an allocation cost independent of block size.
`decompress_limited` and its dictionary variant pass the 64 MiB SSTable safety
limit directly to `zstd::bulk::Decompressor::decompress`. In the enabled zstd 0.13
configuration, its `upper_bound` method returns `None` (the experimental feature
is disabled), so each miss allocates that full capacity. The caller then converts
the short decoded vector into its retained block. Document-store bounded readers
share the same compression owner and can incur the same excess reservation.

Use the stable library `get_frame_content_size` header reader as an allocation
hint, never as validation of the decompressed content. A declared first-frame
size over the caller's limit can be rejected before allocation. A known size
within the limit gives the bulk attempt its exact capacity. Unknown or malformed
headers use the existing 512 KiB bulk hint, capped by the caller's limit. Retain
the bounded streaming fallback: concatenated frames may exceed the first-frame
hint, frames may omit size, and corrupt data must still return an error. No
header parser, codec, wire format or dictionary identity is replaced. The same
owner serves native sync/async and WASM callers; no cache grows with workload.

Reproduce excessive returned capacity for small ordinary/dictionary frames first.
Cover empty frames, unknown sizes, concatenation, skippable frames, truncation,
trailing corruption, dictionary switches and exact/one-byte-too-small limits.
Compare decoded bytes with the existing library oracle. Preserve all fixture
hashes and query scores/counts; measure decoder misses, whole-query latency,
RSS and prefix scans separately from the block-size experiment. This is a new
candidate after frozen dictionary-v2, not part of its measured build.

## Proposed adaptive conjunction membership windows

The full-corpus dictionary pass leaves exact conjunction counts about 2.6× slower
than Tantivy. Before this candidate, `BooleanScorer` aligned its required cursors
individually for every matching document; only pure disjunctions exposed bounded
membership windows. The implemented experiment extends that same owner for
pure conjunctions whose children already support exact membership windows.
It does not introduce an executor, answer cache, persisted format or scorer.

Consume the most selective required child into the existing 4,096-document
mask. Intersect subsequent children into that mask. When the surviving mask has
more set bits than its 64 machine words, use the child's existing window method;
otherwise seek only those surviving candidates. This comparison is an initial
cost-model heuristic (candidate probes versus bitmap words), not a semantic
threshold or benchmark-query special case. Measure skewed/selective and dense
queries before retaining it. Stop processing a window once its mask is empty.
Scratch is two existing 512-byte masks, independent of corpus or result size;
children retain their current bounded decoder state.

Preserve child storage and score-addition order. After a window, use the existing
alignment/confirmation owner to leave the scorer on its first exact match at or
after the window end, including valid scores and position cursors. Unsupported
children, exclusions and mixed optional clauses retain scalar collection. Score
windows remain restricted to the existing disjunction implementation; an outer
union containing a now-window-capable conjunction can also select that existing
path. Each nested child still contributes its complete scalar score, preserving
floating-point addition order. The new intersection kernel handles score-free
exact membership only. Driver deadline checks before and
after a batch remain authoritative, so an interrupted negative/two-phase check
cannot publish an accepted hit. Visibility masks continue in the existing filter
wrapper.

Validation compares every emitted bit, exact count and next-document score bits
with scalar advance/seek across dense and skewed terms, nested unions/conjunctions,
empty intersections, high document IDs, deletions and sync/async execution.
Measure the same frozen index/compiler/flags before and after on ARM and x86,
retaining all ranked commands as controls. No production storage/cache defaults
change. This is subsequent work beyond frozen `summa-zstd-capacity-v1`.

The verifier must independently request score-free `COUNT`, exhaustive `VERIFY`
and Tantivy `COUNT` for each query before timing. A ranked top-k-plus-count
oracle alone does not exercise the score-free membership-window path.

## Streaming posting-merge admission correction

Review before the impact-format prototype found that `concatenate_streaming`
parses source footers and optional ratio values but does not invoke the existing
posting structural validator. Its single-source copy path can therefore preserve
invalid block headers or directories in a replacement file. Both concatenation
APIs also need explicit checked document rebasing and source-order validation;
plain addition must not wrap or panic at the document-ID boundary.

The correction now preflights all sources before any output: it reuses the
canonical structural validator for encoded inputs and checks remapped block
ranges, total counts and position-cursor arithmetic. Typed in-memory sources
reuse the same remapping check. This is O(blocks plus encoded exception entries), with bounded
metadata state and no posting decode or re-encoding. Copy/remap output must stay
byte-identical for valid sources, including the single-source fast path. I/O
failures still belong to the existing output owner; preflight validation does
not make arbitrary writers transactional.

Regressions cover malformed first/later sources, the zero-offset copy shortcut,
overflow into the reserved terminal document, reversed/overlapping source ranges,
and byte-equivalent valid merges across all codecs. The existing lifecycle owner
must refuse publication after an error. Format, search scoring and query defaults
are unchanged; this correction precedes any new impact record implementation.

### Conjunction window admission from estimated density

The first full-corpus candidate regresses conjunction COUNT by 2.2% overall.
The complete per-query analysis shows why: sparse lead lists pay bitmap setup
cost, while dense conjunctions benefit from batched intersection. The next
candidate selects windows only when the least-cost required cursor estimates
at least one candidate per bitmap word across the segment:

```text
lead.size_hint() * DOC_WINDOW_SIZE >= segment.num_docs() * DOC_WINDOW_WORDS
```

Use u64 products to avoid overflow. Both constants come from the existing
4,096-document / 64-word window; there is no query identity, fitted document-
frequency cutoff or result cache. This extends the original candidate-probe
versus bitmap-word cost model to admission before allocating/clearing a window.
The current per-window sparse-survivor probing remains useful after dense lead
candidates are intersected. Estimated density affects execution cost only;
scalar and window paths preserve the same exact membership and canonical scores.

The common native/async planner supplies the segment's document space to its
Boolean scorer. No API/CLI/format/default cache setting changes are required.
Validate sparse/dense and empty estimates, u32 document-space extremes, all
existing nested/positioned/deleted-row cases, and all 962+714 public gates. Repeat
all commands with the original scalar build and first window build as controls
on both architectures. This is a new measured candidate, not a claim that the
first window candidate passed its performance gate.

## Proposed batch length reads for every text scoring path

September 14 disassembly of the full-corpus build shows scalar `vdivss` in
`TermCursor::compute_deferred_scores` when real norms are present; the vector
`vdivps` branch serves only the TF-as-length fallback. The scalar loop repeatedly
checks the length-source variant and `OwnedBytes` representation before scoring
one document. `TermScorer` already separates gathers and scores, but its gather
still resolves those invariants per document. The same phase also reproduced an
8.2% ranked-OR-with-count regression in the traversal candidate, despite aggregate
wins. This motivates measuring the shared norm/scoring pipeline, not accepting
that regression from the aggregate alone.

Proposed ownership: the segment length-column reader provides bounded batch
reads over its existing little-endian u16 representation. It resolves the backing
byte view once and uses a borrowed slice of two-byte arrays. Plain lengths keep
zero for missing/out-of-range IDs; chunk lengths retain strict indexing and their
BM25 floor. Scalar and batch access use the same representation and semantics.
`LengthSource` dispatches once per block. Both term scoring paths then call the
existing canonical BM25 formula on contiguous frequency/length arrays. No scorer,
format, quantized norm, reciprocal approximation or float reduction is added.
Missing zero lengths still use TF, and boosted/unboosted arithmetic order stays
unchanged. Scratch adds at most 128 u32 lengths to a MaxScore block; existing term
score-window scratch keeps its size. Validation requires bit-exact top-k oracles,
chunk/missing/multivalue semantics, byte identity, portable/WASM builds and matched
full-workload ARM/x86 measurements. Vector instructions alone are not a result.

### Proposed phrase intersection driver (September 14)

The shared `PhraseScorer` currently advances posting iterator zero, computes the
maximum current document over all terms, seeks every iterator to that maximum,
and repeats even after a term disproves the candidate. Query order therefore
controls the advancing driver. The full-workload profiles attribute material CPU
to this alignment loop after posting seek itself was improved.

The proposal chooses the smallest posting list once, using the existing exact
list cardinalities. All phrase iterator and position arrays remain in query
order. One `usize` identifies the driver; no additional per-query allocation is
needed. A candidate is checked against the other lists until the first rejection;
that list's next document is a lower bound on the intersection, so the driver
seeks to it before trying again. Successful alignment still parks **every** term
on the same document before the existing position matcher runs. No scoring,
phrase-offset, slop, occurrence-count, ordinal, or persisted-byte semantics change.
The sync/async builders and candidate backfill share this same scorer.

The invariant is that each rejected candidate advances the driver to a strictly
larger document or termination, without skipping any possible intersection.
Explicit seeks, ordinary enumeration, and two-phase candidate iteration must all
use the selected driver. Deadline checks stay inside the alignment loop. Setup
is O(number of terms), with one word of additional scorer state. Successful
candidates still require all terms; failed candidates stop at their first witness.

[Tantivy 0.26's intersection](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/query/intersection.rs)
orders drivers by cost and restarts on rejection; its
[phrase scorer](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/query/phrase_query/phrase_scorer.rs)
uses that intersection. Summa will retain its own documented phrase/slop rules.
This is an unmeasured proposal. Acceptance requires complete-workload timings,
exact counts and exhaustive top-k gates on x86 and ARM, including merged streams.

## Proposed bounded score collection (September 14)

The corrected position-admission build remains slower than Tantivy. A repeated
five-process audit also confirms a 10–11% union TOP_100_COUNT regression against
the preceding phrase-driver build. Separate software profiles of all 301 unions
attribute approximately 29% of CPU to TopKCollector and 17% to the collection
driver in both builds; decoding accounts for approximately 7%. The profiles do
not establish the cause of the regression. Optimize this common execution cost
without removing structural validation or changing relevance.

The shared native/async/portable driver already receives exact membership and
scores for bounded windows. Its proposed next step passes one 64-document word
to collectors that explicitly support batching. Top-k counts the word once,
hoists heap representation/capacity checks, fills any remaining heap capacity,
and then retains a local copy of the worst result, refreshing it only after a
competitive replacement. CountCollector uses the population count. Both use
the same input membership, including noncompetitive matches. There is no new
scorer, score approximation, candidate pruning, persisted format or allocation.

Custom collectors default to the existing per-document path. Tuple collectors
opt in only when every child explicitly supports blocks; this preserves the
existing interleaved per-document behavior for custom side effects. Position
collection and deadline-bearing collection retain their existing path, including
checks before each hit. The budget-free block path is bounded to 64 documents;
all inputs still pass through the same eligibility filtering and scorer.

The invariant is identical ordered results and counts to per-document collection,
including NaNs, infinities, signed zero, ties, empty words, partial heaps, zero
limits and saturated total_seen. Tests compare score bits and IDs and preserve
custom callback ordering and cancellation. Acceptance requires full-workload
before/after timings on the same full-corpus x86 and both 100k ARM fixtures,
exact-count/exhaustive-ranking gates, memory, portable and WASM validation.

## Proposed required-clause intersection driver (September 14)

After bounded collection, full-corpus AND TOP_10 remains 681.833 µs versus
Tantivy 330.145 µs; AND COUNT remains 557.240 versus 282.576 µs. BooleanScorer
already chooses the smallest required cursor for advancement, but alignment
still reads every cursor to compute a maximum candidate and seeks every cursor
again, including the driver. This cost affects ordinary terms and composed
required clauses, in both native and async execution.

Retain the existing selected driver throughout alignment. Read its current
candidate; seek the other required cursors in query order until the first
rejection. That cursor's next candidate bounds the next possible intersection,
so seek the driver to it and restart. Do not seek the driver to its own current
candidate. Once all candidates align, run the existing two-phase confirmations,
exclusions and optional-clause positioning unchanged. Keep arrays and score
summation in query order. Ordinary merged/count windows re-enter the same loop.

This changes neither scorer ownership, candidate approximation rules, index
formats, heap residency, nor scratch. The invariant is strict forward movement
at each rejected candidate, with every required cursor confirmed before exposing
an exact hit. Cursor-local deadlines remain checked by doc/seek/confirmation.
Tests cover every driver index, interleaved gaps, optional/excluded clauses,
explicit seeks, terminal/high-ID boundaries and existing two-phase phrase cases.
Measure full workloads and exact-count/top-k gates against the frozen collection
build on both architectures before claiming improvement.

## Proposed score-required terms in window MaxScore (September 14)

The L0 impact experiment leaves official TOP10 essentially flat: 839.458 µs
versus its matching rounded control's 833.194 µs, with Tantivy at 503.951 µs.
Its frozen group-envelope successor is being measured independently. Another
traced execution gap is that window MaxScore bulk-scores every essential term's
union even when a competitive hit must contain multiple terms. Lucene's pinned
[MaxScoreBulkScorer](https://github.com/apache/lucene/blob/1160d3c8a256ee967b16df7e9e26ec39da3aea01/lucene/core/src/java/org/apache/lucene/search/MaxScoreBulkScorer.java)
uses required terms to intersect candidates first. This is distinct from the
explicit Boolean MUST driver above.

Within the existing MaxScore owner, a term is score-required when the sum of
all other window bounds is strictly below the guarded competitive threshold.
Sum those other bounds in canonical query order, including absent zero terms;
restrict this deduction to finite nonnegative bounds. Equal-score candidates
remain eligible. If at least two terms are required, choose the
rarest required term as the candidate driver. Score its bounded posting run,
intersect candidates with the other required terms, then visit remaining
essential terms only at surviving IDs. A required term can belong to the
MaxScore nonessential partition; process it once, then exclude its already
accounted bound from the remaining nonessential sum. Nonessential filtering and
canonical final score reduction retain their existing path. This is an execution plan
for the same query, not a changed query or scoring model.

Every essential cursor must finish past the window even if intersection empties
the candidates; otherwise a rejected interval could be revisited indefinitely.
Reuse metadata skipping to discard proven noncompetitive ranges. Keep actual
term contributions in the existing bounded buffers, in original query order.
The shared TermCursor owns candidate seeks and scores; do not add a scorer,
writer, cache, format or corpus-dependent decision. Detection costs at most
64 × 64 bound additions per window and uses fixed term-limit scratch; candidate
and contribution buffers retain their current caps. Keep deadline checks at
window boundaries and during any new candidate loop.

Validate score bits, ties, duplicate/reordered clauses, absent terms, predicates,
legacy/ratio/impact lists, sparse/dense distributions, seeds, approximate mode,
empty intersections and cancellation. Run complete-workload controls against
the frozen group build on identical indexes, including counts and exhaustive
ranking, both ARM fixtures, the full x86 corpus and WASM. This remains an
unmeasured proposal until those checks and comparisons finish.

## Proposed ranked conjunctions in the shared window executor

Today's pure positive text conjunction uses the general Boolean scorer. Its
rarest required clause drives membership, but matches still cross per-document
term seek/score interfaces. The full-corpus profiles show these costs across
ranked intersections. This proposal reuses the existing text cursors, bounded
window contributions, exact collector and score-required candidate intersection.
It does not introduce another scorer formula or posting writer.

The first experiment admits 2–64 plain unweighted terms on one non-chunked
indexed text field, without optional/prohibited clauses, position collection,
proximity, nested complete-membership requirements or approximate settings.
Every admitted cursor is semantically required from the first window. An
exhausted required cursor terminates the intersection; the candidate interval
starts no earlier than every required cursor's current document. The rarest
cursor supplies bounded candidates, other required cursors intersect them, and
final scores retain query-order strict f32 addition. Equal-score ties, eligibility,
global statistics and observable deadlines keep existing semantics. Missing
required postings return no matches. Unsupported compositions keep the general
scorer, and exact-count execution remains exhaustive.

The plan must not discard explicit per-term statistics when decomposing a query.
That existing decomposition boundary needs a regression test before extending its
use. Query-owned statistics are authoritative; query grouping is optional.

Scratch remains bounded by 64 terms times the existing 4,096-document window.
The cost hypothesis is less per-document polymorphism and batched canonical
scoring, offset by window bound/scratch overhead. It is not a speed claim.
Native/async/WASM and exhaustive score-bit tests precede matched ARM and complete
cloud before/after timing. No persisted or wire format changes are involved.

The first score-required union experiment regressed full-corpus rounded top-10
from 843.856 to 924.393 µs. Automatic score-based classification is therefore
removed from the conjunction candidate. Only explicit semantic requirements
select the new intersection plan. The prior group-capable binary remains the
before control. Candidate helpers retain their bounded deadline checks.

The statistics fix carries optional query-owned statistics in `TermQueryInfo`.
Single-field, per-field and filtered grouping retain the original scorer when
those statistics are present; the term remains a scoring text term, so a MUST
clause cannot be reclassified as a zero-score filter. L1 score decomposition
rejects unsupported term-local statistics rather than silently substituting
parent statistics. `TermQueryInfo` is a Rust planning type; there is no persisted
or wire-format change. External struct literals need the new `global_stats`
field (`None` for an ordinary term).

## Proposed scoring of candidate runs within a posting block

The current `TermCursor::score_candidates_sync` seeks each surviving candidate,
then `ensure_scores` computes BM25 for every posting in that block. A block has
up to 128 postings; one surviving candidate can therefore trigger 128 length
reads and score evaluations. This amortizes well for dense unions but may waste
work for selective conjunctions and optional terms. It is a code-level cost
observation, not attribution of the measured cross-engine gap.

A follow-up experiment would separate deferred TF decoding from score readiness
inside the existing cursor owner. Candidate probes would group matching document
IDs within the current block, decode its TFs once, gather lengths for those IDs
only, and run the canonical scorer over bounded contiguous scratch. Already
computed full-block scores would remain reusable. Full-window scoring retains
its existing bulk path. The same helper serves explicit conjunctions and optional
MaxScore contributions; sparse cursors retain their current representation.

At most 128 document IDs, posting slots, lengths and scores are needed per run,
with no corpus-sized cache or additional per-hit allocation. Block transitions
must invalidate both TF and score state; transitioning from candidate probes to
a full score run must preserve every score bit. Tests need mixed seek/score-run
sequences, zero/missing lengths, parameter extremes, ties, cancellation and
native/async/browser parity. Immutable index bytes must remain unchanged.
This proposal is not implemented in the first conjunction candidate.

The separate candidate-run implementation now performs one metadata-aware seek
per physical block and probes its loaded document slice with the existing SIMD
lower-bound primitive. A run retains at most 128 matching documents and their
posting/output slots. It gathers only those TFs and lengths for strict BM25
scoring, reusing complete scores when present. Deferred TF decode and full-score
readiness are separate; moving from selective probes back to a full window
reuses TFs and computes the missing scores. Both required and optional candidate
paths use this implementation, with deadline checks every 64 input candidates.
The first transition test passes across four codecs, parameter/length extremes,
missing candidates, and block boundaries. Broader validation and measurements
remain pending. The original group-capable executor and first conjunction
candidate must both be retained as controls.

## Proposed compact batches for semantic conjunctions

The first two ARM candidates regress ranked AND queries: scoring only selected
postings recovers some time, but the dense 4,096-ID window, per-term contribution
arrays and score-bound partitioning still impose work on selective intersections.
The next experiment removes that machinery from semantic AND execution while
retaining it for unions. It uses the same `MaxScoreExecutor`, `TermCursor`,
canonical BM25 implementation and exact collector; there is no format change.

The rarest term drives typed cursor alignment. When another required term lands
ahead, the lead seeks there and alignment restarts. Only fully aligned eligible
hits contribute TFs to a batch of at most 128 documents. Per-term TF rows occupy
at most 64 _ 128 _ 4 bytes; document/score/length scratch is also bounded. Each
term scores the batch with the shared canonical kernel, and per-document totals
are accumulated in original query order. No score-bound pruning is applied to
this conjunction path: all matching eligible documents are scored. The seeded
collector still retains the exact top-k and deterministic ties.

The planner eligibility restrictions stay unchanged. Complete membership/count
and positioned/nested callers retain the general scorer. Deadlines are checked
while aligning and before scoring each bounded batch. Pending unscored hits may
be discarded on expiration, with truncation observable and already collected
hits preserved. Tests must cover score bits, rare-term order, missing/duplicate
terms, predicates, block/batch boundaries, global statistics, and sync/async/WASM.
The before control is the original executor as well as the candidate-run version;
recovering a regression is not a net performance improvement.

### Proposed demand-driven frequencies in the shared posting iterator

The ordinary `BlockPostingIterator` currently decodes document IDs and all
frequencies whenever it loads a block. Its count windows also use a visitor
that supplies frequencies even though membership does not read them. This
cost affects the general Boolean, count and positional candidate paths, beyond
ranked conjunctions. The MaxScore cursor already has separate document/frequency
readiness; the ordinary iterator should use the same posting owner's decoders.

The proposed iterator keeps document IDs immediately available and initializes
a fixed 128-frequency buffer only for `term_freq`, position accounting, or a
scored posting visitor. A boxed `OnceLock` preserves the public immutable
`term_freq`/`position_cursor` methods and native Send+Sync without a mutable alias
or a per-read mutex. Its allocation is reused across blocks and point-query
scratch recycling; storage is bounded per cursor, not per corpus or match.
Document membership windows walk document slices directly. Decoding into a
slice is extracted from the existing frequency decoder, with the Vec API
retained as its adapter; there is no second codec or writer.

The invariant is identical document movement, frequency values, position
prefixes and score bits across any interleaving of seeks, count windows, scored
visitors and position reads. Changing blocks invalidates both frequency
readiness and prefix accounting. Structural admission still validates encoded
frequency payloads before use; laziness must not admit corrupt data. Encoded
bytes do not change. Tests must cover all codecs, tails, exhausted and stopped
visits, borrowed/owned cursors, scratch reuse and shared immutable frequency
reads. The extra readiness check and buffer initialization may regress scored
queries; matched complete-workload latency, memory and exhaustive verification
against the frozen compact-conjunction and original controls determine whether
the change is retained. No speedup is assumed from avoided decode calls alone.

### Proposed density-based text union execution

The dense text executor clears and addresses scratch by document-ID windows.
That work can greatly exceed the number of postings for sparse unions spread
across a large ID range. The proposed alternative stays in `MaxScoreExecutor`,
using its existing cursors, canonical BM25 batch kernel, predicate and collector.
It scans the sorted union into batches of at most 128 actual document IDs,
records only present term frequencies, scores those postings in query reduction
order and collects exact final sums. It does not prune documents by score.

The initial conservative cost gate requires at least two text cursors and
fewer than one posting per 64 IDs across their combined first/last-ID range.
This chooses compact execution only where a 4096-ID window averages fewer
than 64 postings. It uses list metadata, not query strings, fixture identity,
cached answers or observed final scores. Single terms retain their existing
specialized path; dense unions retain window MaxScore, and semantic conjunctions
retain their own alignment path. The fixed cost estimate is a hypothesis to
validate on both ARM and x86; sparse traversal trades pruning for avoiding dense
scratch and must be rejected if complete-workload measurements do not justify it.

Scratch is bounded by query terms times 128 frequencies, two membership words
per term and fixed-size document/score arrays. Absent terms are never passed to
BM25 with a fabricated zero frequency; this preserves zero-k1 and exceptional
parameter behavior. Accumulation follows the existing canonical term order,
including duplicates and ties. Eligibility is checked before frequency reads.
Deadlines are checked during traversal and between score rows, and unscored
partial batches are discarded with observable truncation. Native and async
entry points select the same in-memory text path; no format or scorer formula
changes. Tests compare ordered document IDs and score bits with exhaustive
collection across sparse/dense layouts, codecs, empty terms, predicates, wide
queries, tails and late winners before matched complete-workload measurements.

### Proposed demand-driven phrase frequency

Phrase confirmation currently counts every matching start even for document-only
collection. Exact document counts require existence of a phrase occurrence in
each matching document, while ranked collection also requires its full frequency.
The proposed scorer confirms the first occurrence and saves its monotone position
cursor state. If `score()` is called, it resumes from the next start and counts
the remaining occurrences once. A scalar `OnceLock` keeps immutable score reads
Send+Sync; a small inline copy of the saved cursor indices isolates the resumed
scan without modifying the confirmation state. Wider phrases may spill that
copy proportionally to their existing query size.

One shared scanner supports stopping after one match or finishing the frequency.
Ranked queries do not repeat the already confirmed prefix. Document-only queries
never finish the unused frequency scan. The first-term/independent-slop-interval
semantics, repeated terms, offset overflow handling and canonical BM25 formula
remain unchanged. Empty or unconfirmed candidates retain their previous scoring
behavior. Moving to another candidate clears first-match and cached-frequency
state; cancellation invalidates confirmation. Native, async, point scoring and
chunk folding continue through the same phrase scorer. No format or query syntax
changes are required.

The extra continuation setup and readiness check can cost ranked queries.
Regression coverage must compare resumed scans to a brute-force occurrence
oracle, shared immutable first score reads, repeated confirmation, mixed scored
and unscored traversal, terminal seeks, false candidates and expired budgets.
Both complete ranking and exact document count must pass the same full-corpus
verification before interpreting matched latency and memory measurements.

### Proposed ranked-cursor lower-bound execution

The compact-conjunction full-corpus profile places 34.55% of self samples in
`TermCursor::seek_prepare`. The ranked text/sparse cursor uses a linear SIMD
scan inside decoded blocks, including long jumps; the ordinary posting iterator
already probes the next document and uses slice binary search for the remaining
suffix. Tantivy 0.26 also checks a nearby document before fixed-block binary
search. This motivates an isolated traversal experiment, not a codec or BM25
change. Sampling alone does not establish its speedup.

The proposed ranked-cursor rewrite retains the current document for backward
or equal seeks, checks the next document for nearby forward seeks, then uses
`partition_point` on the unconsumed suffix. Newly loaded blocks use the same
standard-library lower bound. Its cost is one comparison for common adjacent
steps and logarithmic comparisons for long jumps, instead of a scan proportional
to the gap. Existing text L1/L0 and sparse skip lookup, lazy block loading,
frequency/ordinal readiness, terminal behavior and sync/async I/O remain owned by
`TermCursor`. No encoded bytes, query syntax, scoring formula or public settings
change. This also avoids adding a second unsafe search primitive.

Validation must interleave forward/equal/backward seeks, advances, lazy block
boundaries, exact frequency/score reads and terminal calls against a materialized
posting oracle for every codec and both synchronous and asynchronous methods.
The full exact-count/top-k gates and unchanged index manifests remain required.
A before/after comparison on the same x86 and ARM fixtures must use the frozen
original executor as well as the immediate predecessor; do not infer an overall
win from the seek profile or a standalone loop. The current prototype implements this policy; validation and matched measurement
are in progress.

### Proposed block-WAND for short text unions

The completed compact-conjunction comparison separates execution by query shape:
198 regular two-term unions take 880.244 µs in Summa versus 392.545 µs in
Tantivy; 83 three-term unions take 1550.570/1223.131 µs; 18 regular longer
unions take 2646.073/2744.842 µs. These are geometric means of per-query medians
on the same full-corpus run. The data motivates selecting different traversal
algorithms by public query shape, without consulting query text or expected
answers. It does not establish that WAND will be faster in Summa.

Prototype two-term block-WAND inside the existing ranked executor. Keep semantic
conjunctions, standalone terms and the compact low-density union path ahead of
this choice; keep score windows for wider disjunctions. The candidate applies
only to two text cursors with nonnegative finite IDF and valid BM25 parameters;
unsupported shapes retain their existing executor. No new public tuning setting,
encoded format, scorer formula or adapter-specific query path is introduced.

Select a pivot using remaining global bounds and ascending current document IDs.
Read per-block bounds around the pivot without changing loaded payload state.
If their sum cannot reach the collector threshold, advance one contributing
cursor beyond the proven interval, capped by the first later cursor's document.
Otherwise align the prefix to the pivot and score only an actual eligible match.
All score sums retain canonical query order. Use strict rejection (`bound <
threshold`) and the existing conservative threshold adjustment to retain score
ties and address ordering. Exhausted clauses contribute zero. A gap before a
future block is a zero-contribution interval ending before that block's first ID.

The cost model removes dense score/membership buffers and window partitioning
for short unions, paying bounded pivot comparisons and scoring only competitive
candidates. It may lose vectorized scoring throughput when most candidates
compete, so measure TOP_1000 and dense/low-selectivity queries explicitly. Reuse
`TermCursor` block admission, deferred TF decoding and candidate scoring. A
one-document case in the shared scoring kernel may avoid initializing an unused
128-entry gather buffer while calling the same canonical BM25 function. Scratch
is constant for two cursors, plus the existing bounded collector. Deadline checks
remain at bounded work intervals and mark truncation through the shared owner.

Validation must compare direct and routed sync/async execution with exhaustive
ordered IDs and score bits on every codec, disjoint/overlapping/duplicate clauses,
late winners, absent terms, missing norms, zero-k1, custom positive statistics,
seeded thresholds, equal-score ties, large document IDs and predicate/budget
boundaries. Run the required harness, portable build and browser tests. Preserve
index bytes, and compare the full 962-query mix plus 714-term workload against
both the frozen original executor and immediate predecessor on x86 and two ARM
layouts. The current prototype implements this policy. Its native exhaustive oracles, full harness, portable build and 25 browser
tests pass; matched timings are pending.

### Proposed incremental exact-phrase position intersection

The current phrase scorer reads every term's positions before checking any
position relationship. Long phrases therefore pay for payload reads that an
empty intermediate intersection could make unnecessary. Tantivy 0.26's
`PhraseScorer::compute_phrase_match` intersects positions incrementally and exits
when the intermediate result is empty. The same general execution policy is
applicable here, but Summa must retain its own first-term occurrence and slop
semantics; Tantivy's slop algorithm is not interchangeable.

For zero-slop phrases with at least two terms, retain original first-term starts
and intersect them in place with other terms one at a time. Order those other
terms once by document frequency, keeping document-posting traversal unchanged.
Read a later term's positions only while some first-term starts survive. On the
last term, confirm one match and save the two monotone cursors; exact scoring
resumes that final intersection once. One shared pair cursor supplies both
intermediate compaction and final existence/frequency scans. Nonzero slop keeps
the existing independent-interval scanner. Single-term behavior is unchanged.

The invariant is the multiplicity of original first-term starts that have an
occurrence at every required offset, including repeated terms and offsets.
Do not change the anchor to another term: that would require a separate proof
for duplicate-position multiplicity. Compare offsets in u64 so high positions
cannot wrap. In-place writes stay behind the unconsumed starts. Position buffers
are replaced and frequency state invalidated at the existing owner boundary,
including point backfill; skipped stale buffers are never scored. No format,
public option, scorer formula or native/async adapter changes are required.

The cost is sequential position intersection with a shrinking candidate array,
plus one query-sized order prepared once (inline for ordinary short phrases).
It can avoid later position decoding on failed candidates, at the cost of
compacting intermediate arrays. Two-term phrases use only the final pair cursor.
Measure whole phrase families and the complete workload, ranked and exact count,
on unchanged x86 and ARM fixtures; reject it if these costs outweigh savings.
Regression oracles must cover duplicates, arbitrary offsets, empty intermediate
results, overflow, resumed count, stale buffers after failure, chunk backfill,
cancellation and every posting/position codec. This is a proposal, not a measured
improvement or an implemented change.

The full-corpus compact-union result is now negative (27.9% union TOP_10
regression against its predecessor). The next candidate also removes that route
and its unused kernel, while preserving sparse/duplicate correctness coverage.
Two-term unions then select the existing short-WAND prototype; wider unions use
score windows. This rollback and the phrase change must be reported together as
the candidate's complete source delta, with family-level results separating their
intended effects. They do not change encoded bytes.

### Proposed phrase planning by selectivity (diagnostic prototype)

The isolated ARM decoded-work trace finds identical requested-position totals
for all 198 two-term phrases (195,149 values per engine), but Summa requests
64,844 versus 31,908 for three-term phrases and 237,753 versus 46,151 for longer
phrases. These are aggregate logical work counts, not latency, and the native
100k indexes have their documented engine-specific build/order differences.
Both adapters preserve all 1,676 exact counts; instrumented Summa also preserves
its exhaustive ordered ID/score-bit oracle on canonical and merged fixtures.

The current phrase scorer chooses the rarest posting lead but probes the other
terms in original order. Its exact-position pass always reads original term zero
first. The proposed planner orders all document intersection probes by document
frequency and starts position filtering at the rarest term. An original-first
anchor keeps the current path. Otherwise, rare-anchor positions are shifted into
original-first coordinates with checked subtraction, then intersected against
other non-first terms in selectivity order. Only a surviving filter reads the
original first term; the final intersection enumerates that original first
buffer, preserving its duplicate-start multiplicity and exact phrase frequency.
Nonzero slop keeps its current matching algorithm. Offset overflow/underflow,
backfill invalidation, cancellation and native/async behavior remain explicit
regression targets. No encoding, BM25 formula or query-specific route changes.

This prototype starts in an isolated instrumented copy, where decoded work and
correctness can reject it before production integration or another latency run.
The public candidate remains frozen until evidence justifies a replacement.

The first diagnostic prototype reduces phrase document-block decodes from
15,984 to 12,320 and requested positions from 497,746 to 304,139, compared
with the frozen incremental-position candidate on the same Summa ARM index.
All 1,676 counts and exhaustive ranking gates pass on both fixtures. However,
three-term requests remain excessive because forcing original term zero to the
end ignores its intermediate selectivity. The revised proposal permits it to
filter rare-anchor starts at its sorted position without modifying its own
occurrence buffer. Later terms can then be skipped on failure. Final frequency
still enumerates the untouched original-first buffer against surviving starts;
when original term zero is last, the existing lazy final-pair check is retained.
A new failing regression pins this intermediate-filter case before the revision.
These observations are work counts, not latency or a full-corpus result.

The revised diagnostic planner now requests exactly Tantivy's 273,208 positions
across all 300 ARM phrases, including identical totals in each phrase-length
group, versus the frozen Summa candidate's 497,746. Posting-block decodes are
12,320 versus 15,984; position-block decodes are 10,884 versus 15,631.
Independent cross-source oracles compare all ordered top-1000 document IDs,
raw score bits and exact counts between the prior deferred-frequency build,
the incremental-position build and the revised planner on both fixed ARM
fixtures (1,676 queries each). All pass, as do the revised planner's 15 focused
phrase tests. Source revision two is archived with all diagnostic counter code;
those binaries remain excluded from latency. Full-corpus diagnostic work is
queued after the existing timing runs.

The revised phrase planner is now integrated into the main candidate for the
required native, portable and browser checks and subsequent latency comparison.
Only phrase traversal and its tests are changed at this integration step;
retention of the other experimental executor changes remains subject to their
complete measurements. Matching decoded work does not establish matching speed.

The complete short-WAND timing rows are negative on x86 as well as ARM: rounded
union TOP_10 increases 1268.903 to 1451.435 µs and overall TOP_10 increases
899.328 to 937.045 µs, versus original 850.274 and Tantivy 509.501 µs.
The archive is still being captured. After its complete hash verification, the
next selection step will restore the deferred-frequency candidate's production
ranked executor (removing the short-WAND route, its scalar one-hit scoring
special case, and the subsequently unhelpful ranked-seek rewrite). New behavior
oracles remain routed through native and async execution. The selective phrase
planner, phrase capability correctness fixes, lazy shared posting frequencies,
and existing sparse behavior are retained for validation. The intervening
phrase-plus-WAND build is a correctness artifact, not a performance candidate.
A fresh full native check, portable build and browser suite are required after
this rollback; only the resulting selected source will enter the next timing run.

## Bounded traversal and exact-count collection

Ranked two-term conjunctions intersect decoded cursor blocks directly. Term IDs,
TF rows and query order remain associated throughout the existing 128-document
canonical score batch. General seek/decode owns block changes, validation and
errors. Predicates, mapped document folding and deadline checks remain in their
existing owners. No additional allocation or format change is introduced.

Sparse complete conjunctions use 128-document batches alongside the existing
dense 4096-document membership/score windows. Posting iterators copy IDs and
retain sorted candidate IDs within decoded blocks; frequencies are read only
when scoring or positions require them. The same kernels accept a compile-time
frequency flag. Untimed term scorers with unit boost use the canonical batch
scorer; pure compatible conjunctions preserve complete child scores in original
query order, including nested sums. Unsupported consumers retain scalar methods.
Dense windows retain priority. Count scratch is 512 bytes; a flat scored batch
uses about 4.2 KiB across the active leaf, conjunction and collector, plus about
1.2 KiB per active nested conjunction. Existing query-depth limits still apply.
The collector discards a batch that expires before publication.

Exact count-only collection can count a two-term plain text union by adding its
dictionary frequencies and subtracting the intersection count. The existing
Boolean scorer and count driver compute that overlap. Admit only indexed,
unmapped physical fields with positive DFs and no deleted documents. Missing,
fast-only, mapped, chunked and unsupported queries use ordinary collection.
An overlap larger than either DF is corruption. The shared text planner admits
only indexed fields to postings-based MaxScore; fast-only fields keep their
column scorer for unions and conjunctions as well as standalone terms.

Score-only TopK and Count collectors
can opt into ranked collection followed by an exact omitted-match count. Tuple
collectors require every child to opt in and retain the largest child limit.
Custom and position collectors keep complete callbacks. Admit only plain
positive-weight text terms or two same-field terms with no mapping or deletions.
The size gates amortize the extra traversal: 16 × max(128,k) for a
single term, or a term DF of 128 × max(128,k) before union overlap work. The
existing ranked executor adds a bounded heap alongside the caller's heap; count
omitted matches without altering scores, retained IDs or total_seen. VERIFY and
explicit exhaustive benchmark modes use a forwarding collector that declines
this capability, preserving their independent full-scoring reference.

## Optional standalone RGB experiment

Create a new eligible index with `index PATH --reorder-text` plus the desired
encoding flags. This sets the existing schema attribute and initially writes
identity mappings. Preserve that directory as the unpermuted control, then run
`reorder COPY` on a separate copy to invoke the existing core writer's standalone
RGB operation. The command logs its configured BP memory budget and core progress
to stderr. `serve` uses the persisted field mapping without additional flags.
The default index schema is unchanged. See [the implementation boundary and
validation requirements](maxscore-text-reordering.md); RGB results are separate
from the benchmark's RGB-disabled parity target.
