# Lucene 11 performance research for Summa

Research snapshot: 2026-09-13. Lucene source is pinned to
[`1160d3c8a256ee967b16df7e9e26ec39da3aea01`](https://github.com/apache/lucene/tree/1160d3c8a256ee967b16df7e9e26ec39da3aea01),
committed 2026-09-11. Lucene 11 is development work, not a released benchmark
competitor: the latest stable announcement is 10.5.1, dated 2026-08-12.
The August developer discussion still discusses preparing the 11 release.
Sources: [release announcements](https://lucene.apache.org/core/corenews.html),
[release discussion](https://www.mail-archive.com/dev@lucene.apache.org/msg318643.html).

The useful lesson for Summa is to move work through contiguous batches and
make expensive work conditional on cheap, sound bounds. Java-specific changes
are not automatically useful in Rust. The development tree also includes
optimizations released in 10.4/10.5 and work listed for 10.6; attributing all of
these to an exclusive “Lucene 11 algorithm” would be misleading.

## Findings and implementation priorities

| Mechanism                                       | Source and status                                                                                                                                                                                                                                                                                                | Summa implication                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Posting, norm and score batches                 | Pinned [TermScorer](https://github.com/apache/lucene/blob/1160d3c8a256ee967b16df7e9e26ec39da3aea01/lucene/core/src/java/org/apache/lucene/search/TermScorer.java) uses `nextPostings`, bulk norm reads and a bulk similarity scorer.                                                                             | High priority. Our complete union path repeatedly dispatched through each child for each document. A bounded score-window prototype now visits each child in sequence; a second experiment gathers lengths and computes canonical BM25 over decoded posting runs.                                                                             |
| Block-max MaxScore with required terms          | Pinned [MaxScoreBulkScorer](https://github.com/apache/lucene/blob/1160d3c8a256ee967b16df7e9e26ec39da3aea01/lucene/core/src/java/org/apache/lucene/search/MaxScoreBulkScorer.java), with the [required-term refinement](https://www.elastic.co/search-labs/blog/more-skipping-with-bm-maxscore) released earlier. | High priority for ranked unions. Summa partitions essential/nonessential terms, but its window loop still scores the essential union. When all other terms' conservative bounds cannot reach the threshold, a term becomes required. Use that to intersect candidates before scoring; preserve equal-score ties and canonical addition order. |
| Competitive frequency/norm frontier             | Pinned [CompetitiveImpactAccumulator](https://github.com/apache/lucene/blob/1160d3c8a256ee967b16df7e9e26ec39da3aea01/lucene/core/src/java/org/apache/lucene/codecs/CompetitiveImpactAccumulator.java).                                                                                                           | High priority for single terms and unions. Summa's optional ratio metadata is safe but still combines extrema from different documents. A bounded conservative frontier could tighten bounds under varying query-global statistics. Requires an explicit format and merge-copy design, not index-time fixed BM25 scores.                      |
| Phrase rejection before position initialization | [PR #15861](https://github.com/apache/lucene/pull/15861), listed under 10.5.                                                                                                                                                                                                                                     | High priority after profiling. Lucene bounds a candidate with maximum possible phrase frequency and its norm before resetting position state. Summa must derive a bound for its own phrase scoring and ordinal semantics. Apply only to competitive ranking; exact membership/counting still verifies the phrase.                             |
| 256-posting blocks                              | Pinned [ForUtil](https://github.com/apache/lucene/blob/1160d3c8a256ee967b16df7e9e26ec39da3aea01/lucene/core/src/java/org/apache/lucene/codecs/lucene104/ForUtil.java); shipped in the 10.4 format.                                                                                                               | Experiment, not an automatic default change. Summa has 128-posting blocks. Larger blocks amortize decode/skip overhead but weaken pruning and consume more candidate scratch. Measure both architectures and selective/dense workloads. Existing compatible payloads must remain copyable at merge.                                           |
| ARM vector lower-bound search                   | [PR #16254](https://github.com/apache/lucene/pull/16254), merged into main on 2026-06-14.                                                                                                                                                                                                                        | Already present in Summa's `find_first_ge_u32` NEON path. Compare generated code and whole-query profiles before changing it. Lucene's reported microbenchmark gains are not expected Summa gains.                                                                                                                                            |
| Bulk exclusion runs and two-phase masks         | [PR #16307](https://github.com/apache/lucene/pull/16307), [PR #16362](https://github.com/apache/lucene/pull/16362); development/backport work.                                                                                                                                                                   | Extend the existing filter and two-phase owners when profiling supports it. Complete run boundaries can skip rejected ranges or mask batches without per-document dispatch. Cancellation must never turn an incomplete negative check into an accepted hit.                                                                                   |
| Batch disjunction iteration                     | [PR #16394](https://github.com/apache/lucene/pull/16394), development work listed for 10.6.                                                                                                                                                                                                                      | The same principle as Summa's membership windows: avoid a heap/cursor update for every match. Useful for constant-score filters as well as exhaustive counts. Maintain an independent scalar oracle.                                                                                                                                          |
| Lazy MaxScore scratch                           | [PR #16316](https://github.com/apache/lucene/pull/16316), development work.                                                                                                                                                                                                                                      | Summa's previous single-term specialization already avoids general window buffers. Keep scratch lazy on paths that actually use it; do not allocate per-term dense arrays for paths that only need membership.                                                                                                                                |

## Changes that require a different workload

Lucene 11's [query-cache refactor](https://github.com/apache/lucene/pull/15558)
partitions locking, keys entries by query and segment, and separates stale-entry
cleanup. This addresses contention on reusable filter results. It does not
explain single-thread uncached BM25 execution speed. Summa's decompressed term
**dictionary** block cache is a different cache: it shares storage decoding,
not answers. Its measured capacity tradeoff is recorded in the
[ratio-bound report](search-benchmark-ratio-results.md). A future filter cache
needs reader-generation identity, admission policy, byte limits and invalidation
measurements. It must not be introduced just to memorize benchmark queries.

Lucene's [native vector work](https://github.com/apache/lucene/pull/15508) uses
native SIMD dot products and avoids Java/native copy overhead. This matters for
dense vectors. Summa already executes native Rust SIMD kernels, and sparse
retrieval is dominated by posting traversal and candidate selection rather than
a single contiguous dense dot product. Reuse useful kernels where applicable,
but do not transfer dense-vector headline speedups to full-text or BMP claims.

The pinned [change log](https://github.com/apache/lucene/blob/1160d3c8a256ee967b16df7e9e26ec39da3aea01/lucene/CHANGES.txt)
also adds BM25 k3 query-term saturation. That is a scoring-model option, not a
semantics-preserving acceleration. Any adoption would need an explicit query or
schema setting and separate relevance evaluation; it cannot silently replace
the benchmark's current BM25 model.

## Implementation and evidence limits

Summa retained bulk score windows, contiguous term-run BM25, and standalone
term windows. The full-corpus score-window v2 prototype improves official
TOP100+COUNT 1.462× and its complete union family 3.222×; it still takes 2.338×
Tantivy's time overall for that command. The subsequent corruption fix has a
measured 6–9% validation cost. The retained build takes 2.30–2.84× Tantivy time
across official commands.
[The report](search-benchmark-bulk-results.md) separates those candidates.

The phrase-bound prototype was removed after a paired full-corpus repeat:
phrase TOP10 improved 12.9%, union TOP10 slowed 11.2%, and the overall result was
flat. Its correctness evidence and frozen source remain available. A later removal
control did not recover the union slowdown and traded wins across workloads,
so that slowdown cannot be assigned solely to the hint. The extra hook remains
out pending a controlled explanation of its net benefit.

Final retained code passes 1,649 native tests, native without sync, and the WASM
build plus 20 tests. All 1,676 ARM query gates pass ordered ID/score-bit and exact
count checks. Full-corpus final checks pass all 1,676 gates. Timings use the same Cascade Lake
machine as the frozen baseline and Tantivy, excluding builds and profiles. No Lucene
engine timing or sparse-index speedup is claimed in this pass.

## Dictionary lookup is a separate bottleneck

The pinned [Lucene103 block-tree writer](https://github.com/apache/lucene/blob/1160d3c8a256ee967b16df7e9e26ec39da3aea01/lucene/core/src/java/org/apache/lucene/codecs/lucene103/blocktree/Lucene103BlockTreeTermsWriter.java)
uses default block targets of 25–48 items, splits oversized shared-prefix groups
into floor blocks, and chooses among uncompressed, LZ4 and lowercase-ASCII suffix
representations. Its term statistics and posting metadata are separate from
suffix bytes. This is established codec design present in the Lucene 11 tree,
not a new 11-only feature.

Summa currently compresses approximately 16 KiB SSTable blocks and includes
term values in the entry stream. Earlier count-only profiles spent about 48%
of samples in Zstd sequence decompression at the default 256-block cache;
1,024 blocks remove most of that cost but increase memory. A next controlled
experiment should vary dictionary block size and existing compression settings,
then consider an established low-overhead codec if necessary. Smaller blocks
trade more index/address metadata for less decode work per miss. Separate hot
frequency/address metadata from colder inline posting payloads only if measured
benefits justify a versioned layout. Do not replace the dictionary or change its
default cache simply to retain the benchmark's 714 terms. The subsequent dictionary experiment adds an opt-in block-size setting and a
retained-byte cache cap using the existing STB5/Zstd format. Defaults remain
unchanged. Its separate measurement and correctness protocol is recorded in the
[benchmark design](search-benchmark-game.md#dictionary-block-size-and-cache-byte-experiment).

## Full-corpus follow-up

The second bulk-scoring prototype now has full-corpus evidence: all 962 official
queries improve 1.462× on TOP100+COUNT, including 3.222× for all 301 unions.
Summa still takes 2.338× Tantivy's time overall for that command. The matched
union COUNT control is flat. See the [full tables and limits](search-benchmark-bulk-results.md).

The offline impact probe covers 780,298 blocks for all 714 distinct workload
terms. Mean Pareto cardinality is 8.506 (maximum 41); the BM25 convex envelope
retains 3.600 points on average (maximum 10, with 264 blocks over eight). At an
oracle final top-10 threshold, ideal block rejection is 69.009% for the ratio
proxy and 99.138% for the envelope/full frontier. Top-1000 gives 11.496% versus
71.600%. These numbers exclude heap warmup, metadata cost and conservative
rounding, so they measure potential rather than actual skipping or latency.
The later [impact implementation and full-corpus results](search-performance-review.md#competitive-impact-results-september-14)
show that this potential does not translate into a complete-workload win.

## Norm computation is another scoring cost

The pinned [BM25Similarity](https://github.com/apache/lucene/blob/1160d3c8a256ee967b16df7e9e26ec39da3aea01/lucene/core/src/java/org/apache/lucene/search/similarities/BM25Similarity.java)
precomputes 256 inverse norm factors from encoded length bytes. Its batch scorer
first gathers those factors, then runs a vectorizable scoring loop with one
division per hit. Summa's current canonical calculation divides both length by
average length and the TF numerator by its denominator. This helps explain why
batching alone need not close the gap.

Copying Lucene's quantized lengths or algebraic score rewrite would change
Summa's current scores. A separate, unimplemented option is a bounded lookup
of exact canonical norm factors for common integer lengths, with the existing
calculation for longer lengths. It would need query-global statistics in its
identity, shared ownership across terms, bounded memory and setup-cost accounting,
and bit-exact exhaustive tests. No lookup cache or norm-format change is included
in this pass.

An offline ARM kernel probe over all 1,644,488 postings for the 714 terms in
the 100k fixture did not justify that lookup prototype. Median canonical scoring
is 0.378 ns/value, versus 0.476 with a 16 KiB partial table and 0.438 with a
256 KiB full table; score bits agree throughout. These are warm contiguous-array
kernels, excluding posting decode, collection and query setup, not query times.
The runtime-sized lookup/gather prototype remains outside production. An x86
or statically sized table could behave differently; copying Lucene's technique
without measuring Summa's generated code would be premature.

The [posting-validation cache follow-up](search-benchmark-validation-results.md)
improves the official full-corpus workload by 5–10% while retaining structural
checks for new bytes. It leaves Summa 2.11–2.60× slower than Tantivy across those
commands; single-term dictionary COUNT remains a separate bottleneck.

The [dictionary follow-up](search-benchmark-dictionary-results.md) measures smaller
Summa STB5/Zstd blocks inspired by Lucene's block-tree granularity, with equal
cache caps and unchanged posting bytes. The final full-corpus 4 KiB candidate
improves official commands 10–18% versus 16 KiB; its TOP100+exact-count result is
1,749.443 µs versus Tantivy 852.032 µs. The dictionary grows 4.7%, and prefix and
ARM tradeoffs keep the default unchanged. A separate allocation bug was also
found in how Summa supplied its safety limit to the existing Zstd API; that fix
is measured independently. These are Summa changes informed by the research,
not measurements of Lucene 11 itself.

## Tantivy comparison: request only the posting data needed

Tantivy 0.26's [term query weight](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/query/term_query/term_query.rs)
requests `IndexRecordOption::Basic` when scoring is disabled. Its
[block reader](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/postings/block_segment_postings.rs)
then selects `SkipFreq` and omits frequency decompression in both full blocks
and vint tails. With scoring enabled, it decodes frequencies with the document
block. This is a concrete execution distinction, not evidence that it explains
the whole measured latency gap.

Summa's ordinary iterator previously decoded frequencies on every document
block load, including its membership-window visitor. The current deferred
frequency prototype removes that work for document-only consumers and can also
defer it until a positional or scoring candidate needs it. Its immutable
frequency-access API requires a different readiness mechanism from Tantivy's
request-time policy. Both models preserve the original posting values. The
[Summa validation and measurement record](search-performance-review.md#deferred-frequency-experiment-mixed-full-corpus-result)
must establish whether the extra readiness check pays off in actual queries.

## Traversal follow-up: conjunctions and within-block search

The pinned [Lucene conjunction](https://github.com/apache/lucene/blob/1160d3c8a256ee967b16df7e9e26ec39da3aea01/lucene/core/src/java/org/apache/lucene/search/ConjunctionDISI.java)
aligns the two lowest-cost iterators first and checks the remaining iterators
only after agreement. [Tantivy 0.26](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/query/intersection.rs)
also specializes its two shortest term scorers. These are general query-shape
optimizations. Summa's compact conjunction now sorts by document frequency;
its full-corpus profile is dominated by cursor traversal, while batched BM25
arithmetic occupies less than one percent of self samples.

Tantivy's [posting cursor](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/postings/segment_postings.rs)
checks a nearby document before [fixed-block binary search](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/postings/block_search.rs).
Summa's ordinary posting iterator already uses a nearby probe and suffix binary
search, but its ranked cursor still scans decoded IDs using SIMD. Applying the
ordinary cursor's lower-bound policy to the ranked owner is an unmeasured
hypothesis targeting that discrepancy. It does not require copying Tantivy's
unsafe fixed-size implementation or changing on-disk blocks.

September 24 follow-up: the [ranked pruning investigation](ranked-pruning-followup.md)
now measures that existing fixed-block primitive in Summa's ranked cursor,
alongside guarded rare-term conjunction seeks and phrase changes. Its targeted
10M cloud and smaller ARM evidence is separate from this earlier full-corpus
research snapshot; it does not establish an all-query win.

Tantivy's [phrase scorer](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/query/phrase_query/phrase_scorer.rs)
separates phrase existence from full frequency when scoring is disabled. Summa's
current demand-driven frequency candidate similarly avoids finishing a frequency
scan for membership-only queries while preserving its existing slop semantics.
A stale cache discovered by chunked backfill validation was reproduced and fixed;
this experiment has not yet established a full-corpus gain.

The same Tantivy phrase scorer also intersects position lists incrementally in
`compute_phrase_match`, returning as soon as an intermediate intersection is
empty. Summa's preceding scorer reads all term positions before checking their
relationships. The [incremental exact-phrase candidate](search-performance-review.md#incremental-exact-phrase-position-intersection-validation-in-progress)
now applies that execution policy while retaining the original first-term
occurrence multiplicity. It does not copy Tantivy's slop semantics. A baseline
reproducer and independent occurrence oracles pass after the change; full-corpus
performance is pending.

The short-WAND prototype must also be judged by generated code and complete
workloads. Its ARM disassembly still calls cursor movement, block lookup,
frequency readiness and candidate scoring from the per-candidate loop. Existing
`CachedScoreBound` already memoizes each query-local block's computed bound;
adding another arithmetic-bound cache would not remove the remaining metadata
and call overhead. Both ARM union comparisons regress. This motivates inspecting
work frequency and call boundaries, not assuming that adopting an algorithm's
name reproduces a competitor's implementation quality.

## Required and optional clauses

The pinned [ReqOptSumScorer](https://github.com/apache/lucene/blob/1160d3c8a256ee967b16df7e9e26ec39da3aea01/lucene/core/src/java/org/apache/lucene/search/ReqOptSumScorer.java)
checks block bounds and can turn optional matching into a temporary competitive
requirement when required-only scores cannot reach the cutoff. It keeps that
optimization specific to top-score collection. Tantivy's
[RequiredOptionalScorer](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/query/reqopt_scorer.rs)
uses typed required/optional children and delays optional seeking until scoring.
Summa currently uses dynamically composed complete streams for this shape.
The proposed required text windows reuse Summa's existing batched executor,
canonical arithmetic and membership-intersection helper. Neither competitor's
implementation proves an expected speedup; the complete fixed-corpus comparison
and exact oracle must establish it.
