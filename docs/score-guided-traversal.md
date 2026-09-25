# Score-guided posting traversal: research and Summa fit

September 24, 2026. Research assessment and proposed follow-up, not a claim that
the proposed index structures have been implemented. The current measured work
is tracked in [ranked pruning](ranked-pruning-followup.md).

## Follow-up membership and intersection profiles

The Unicode-word corpus profile attributes 72.6% of three common conjunction
COUNT probes to `BlockPostingIterator::fill_doc_window`. Its sorted-ID loop
updates a bitmap word in memory for every document. The retained implementation coalesces
one word's mask in a register, then updates that word once. Scratch stays at
the existing 4,096-ID window; document/TF cursor state and the encoded bytes
remain unchanged. Dense, sparse, boundary and resumed-window tests and paired
ARM/corpus measurements validate the density gate.

The unconditional grouping experiment improved the dense ARM fixture about
2.4× and `+of +s` about 1.37×, but regressed several less-dense corpus probes
20–38% in CPU time. It was rejected. A per-run density gate reduced the regressions but
left a 1–5% CPU penalty on sparse controls. The retained implementation chooses
from list-wide density once per window (at least 16 postings occupying at least
half the ID span) and dispatches to two compile-time loop variants. This removes
the density branch from decoded runs; it deliberately forgoes grouping a globally
sparse list's locally dense runs. The corpus `+of +s` probe improves 1.47× in
latency and CPU, while five sparse/common controls stay within about 3.5%.
ARM dense fixtures improve 2.12–2.40×; sparse fixtures stay within 1.1%.

Three medium conjunction probes spend 46.2% in the decoded-block intersection
kernel, 9.0% in block bounds and 6.4% in document decoding. This motivates a
separate kernel experiment, not a looser bound or candidate cap. Index-pair
order, partial-output resume positions and unsigned document IDs remain exact.

A separate ranked-pair experiment scores the rarer term's decoded block through
the existing `score_candidate_block` helper. A candidate can be omitted before
intersection only when that exact contribution plus the other term's conservative
list bound is strictly below the established heap threshold. Equal scores stay
eligible. A bounded 128-entry ID/slot projection feeds the existing intersection
kernel; final scoring, deletion predicates and logical-ID ties stay in their
current owners. Counted traversal does not enter this path. This trades extra
lead-term norm/TF work for fewer intersections and requires corpus and ARM
controls before adoption. This candidate was rejected: the corpus screen
regressed representative top-100 queries by 16–29% and introduced 3–6%
slowdowns in several controls. It is absent from production code.

### Concurrent dictionary-cache experiment

The explicit 64 MiB cache improves repeated single-query scans but regresses
concurrent broad scans. The canonical cache reads without hit promotion, while
racing duplicate insertions linearly scan and reorder the eviction deque under
the write lock. A private candidate treats duplicate insertions like ordinary
hits, preserving insertion order and the existing block/byte caps. Another
explicit configuration raises the budget to 512 MiB. These require paired
throughput, CPU and RSS evidence; neither is a production policy or default
change. Query membership, encoded bytes and corruption behavior must agree.

## What can be skipped exactly

A region can be omitted from exact top-k when its conservative query-score
upper bound is strictly below the kth score already proved by real eligible
matches. For example, a region bounded by 3.5 cannot improve a full heap whose
floor is 8.0. Its posting and position payloads need not be decoded.

For additive nonnegative term scoring, combine each query term's bound over
the **same document interval**. A missing required term rejects an AND region.
An equality needs the stable-ID tie rule; physical RGB order does not establish
logical-ID order. Exact total counts still require membership evaluation.

There are two separate decisions: which region to visit next, and whether a
region is proven unable to win. A heuristic may choose visitation order. Only
a conservative score bound, evaluated with the request's scoring parameters and
global statistics, may justify omission. Density or topical similarity alone
does not certify a high score or an uncompetitive region.

A useful structure is an array-backed skip hierarchy whose entries contain a
document-range endpoint, a payload offset, and a conservative score envelope.
Leaves cover posting blocks; parents cover several children. If a parent's
bound loses, jump past all its children. If it remains competitive, descend.
Unlike a plain skip list, this directory answers both "where is document d?"
and "can anything before this endpoint beat the current result heap?"

For example, suppose three disjoint regions have query bounds 2, 11 and 3.
Visit the second region first. If ten real matches establish a tenth score of
8, skip the other two regions without scoring their documents. The bound of 11
does not promise ten good matches; if the actual scores are low, continue into
the remaining regions. An isolated high score can also keep a large region's
bound high, which is why finer or variable partitions can matter.

There is a limit to what cheap metadata can prove. Different terms' maxima can
belong to different documents, so their sum may overestimate every real score
in a region. Phrase adjacency adds another source of looseness. A region can
therefore be unhelpful in reality yet remain competitive under a safe bound.
Measure bound looseness separately from slow heap-threshold growth; they call
for different improvements.

## Relevant primary sources

| Work                                                                                                                                                                                                         | Relevant mechanism                                                                                                                                                            | Summa implication                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [Ding and Suel, SIGIR 2011: Block-Max indexes](https://research.engineering.nyu.edu/~suel/papers/bmw.pdf)                                                                                                    | Per-block score upper bounds allow WAND to bypass postings without scoring them.                                                                                              | The basic score-aware skip mechanism already exists in Summa.                                          |
| [Mallia and Porciani, ECIR 2019: Longer Skipping](https://www.antoniomallia.it/uploads/ECIR19a.pdf)                                                                                                          | Skip consecutive blocks whose maxima do not increase; an alternative stores precomputed skip distances.                                                                       | Direct match for "skip pointers with hints" across a long low-score run.                               |
| [Bortnikov, Carmel and Golan-Gueta, WWW 2017: Conditional Skips](https://archives.iw3c2.org/www2017/proceedings/companion/p653.pdf)                                                                          | A conditional iterator combines document targets and score thresholds. A treap implementation orders by document ID while heap-ordering scores, allowing subtree skips.       | A concrete tree-based alternative; compare its memory and traversal cost with compact block metadata.  |
| [Mallia et al., SIGIR 2017: variable-sized blocks](https://pages.di.unipi.it/rossano/assets/pdf/papers/SIGIR17A.pdf)                                                                                         | Adapt bound partitions to score variation, reducing contamination of a low-score region by isolated high scores.                                                              | Consider variable bound regions if measurements show fixed boundaries are the limiting factor.         |
| [Lucene Impacts](https://lucene.apache.org/core/9_12_2/core/org/apache/lucene/index/Impacts.html) and [ImpactsSource](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/index/ImpactsSource.html) | Multiple levels expose frequency/norm envelopes and validity endpoints; shallow advancement consults this information without normal document advancement.                    | A compact hierarchy with independent metadata traversal is a closer fit than pointer-heavy skip nodes. |
| [Mackenzie, Petri and Moffat, 2021: Anytime Ranking on Document-Ordered Indexes](https://arxiv.org/abs/2104.08976)                                                                                           | A cluster-skipping index uses per-term range bounds and BoundSum to prioritize ranges. The paper distinguishes exact termination from deadline-limited approximate execution. | Closest match to combining RGB with query-dependent traversal order.                                   |

The last paper also evaluates exact range processing. Its advantage is smaller
against BMW/VBMW than against WAND/MaxScore, so adding a range directory is not
automatically a win. A bound can be safe yet too optimistic to predict where
good actual matches occur. The authors' [implementation](https://github.com/JMMackenzie/anytime-daat)
is available for reproduction.

Longer Skipping also exposes an important cost: scanning compressed metadata
for a longer jump can cost more than it saves; precomputed distances address
that overhead. The conditional-skip paper likewise finds its treap preferable
for short queries with larger skips, while the simpler iterator wins on longer
queries. Neither result justifies replacing every iterator with a tree.

[Block-Max Pruning for learned sparse retrieval, SIGIR 2024](https://research.engineering.nyu.edu/~suel/papers/pulse-sigir24.pdf)
provides another example of aggregating bounds over aligned document ranges
and processing promising ranges first. Its learned impact scores and block
evaluation layout differ from text phrase scoring. More recently,
[Yafay and Altingovde, IEEE Access 2026](https://open.metu.edu.tr/handle/11511/119623)
use historical query-result hit counts to reorder documents and raise thresholds
earlier. That is an alternative workload-dependent ordering signal, not proof
that a region can be skipped for an unseen query.

## Existing ownership and the proposed next experiment

Summa text postings already have block and L1 group bounds, including optional
ratio/impact envelopes. `phrase_block_bound` and `phrase_group_bound` consume
them; typed conjunctions have their own measured admission policy. The sparse
BMP executor separately has coarse/superblock/block bounds and prioritized
superblock processing. Its quantized scoring and bounded LSP policy are not a
drop-in replacement for exact BM25 phrase scoring.

My recommended follow-up is a **compact region-bound hierarchy plus bounded
priority traversal**, evaluated first using existing metadata:

1. Define disjoint physical-document intervals shared by the query's terms.
   Derive conservative interval bounds from all overlapping posting groups.
   Keep this metadata traversal separate from payload decoding.
2. Use a bounded priority frontier to nominate promising intervals. Within an
   interval, retain the existing posting intersection and position verifier.
   Establish the heap floor early, then skip certified losing intervals.
3. Retain a complete fallback traversal when the frontier budget is exhausted.
   Record visited intervals compactly or partition traversal so each result is
   emitted once. A frontier cap must not silently become a recall cap.
4. Only if measured bound looseness warrants it, add persisted aligned range
   envelopes or variable bound partitions. Budget bytes per posting, resident
   metadata, build time, and cold seeks before changing any default.

Current phrase traversal already coalesces consecutive losing L0 blocks,
bounded to one L1 group, and can skip a losing L1 group directly. The first
Longer Skipping experiment should therefore ablate these mechanisms before
adding wider jumps or precomputed distances. Persisted scalar-score hints would
need to remain valid under Summa's request scoring parameters and global
statistics; do not assume that a BM25 ordering precomputed for one configuration
holds for another.

This is a proposal inferred from the research and Summa's current design.
Existing group boundaries differ between terms; summing unrelated block maxima
does not establish a bound over an arbitrarily larger common interval.

The query-time cost should be proportional to the bounded number of inspected
region summaries times the query's term count, plus payloads in visited regions.
A disjoint interval frontier avoids a corpus-sized visited bitmap. Admission
should retain the ordinary iterator for small lists where scheduling overhead
cannot be amortized. An offline exhaustive audit can compare each region's bound
with its best actual score and record threshold growth against work performed.

Phrases need their own bound: term occurrence does not imply adjacency. Summa
can bound exact phrase frequency by every term's frequency only with its
unique-original-first-position certificate; otherwise the original first term
remains the safe bound. Use those frequencies in the phrase's BM25 score space,
then verify real positions. A bag-of-words bound may be conservative but loose;
substituting a bag-of-words score for the actual phrase score changes semantics.

Persisted hierarchy changes would belong to the posting writer/reader, preserve
compatible encoded leaves during merge, and require versioned validation.
Query-only scheduling over existing bounds needs no schema change. Test exact
IDs and score bits, deleted documents, repeated positions, slop, ties, deadline
boundaries and native/async parity, alongside before/after CPU, memory and
latency on both ordinary and RGB indexes. No universal speedup is implied.

## Expanded-workload experiment (September 24)

The first follow-up keeps 1.9.1's format and scoring unchanged. Its public path
is the Searchbench HTTP adapter, then the core collector, typed conjunction
cursor, and `PhraseScorer`. Native synchronous queries own the timed path;
the scorer and collector are shared with asynchronous/portable execution.
The experiment builds isolated source variants, not runtime settings:

| Variant              | Question                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Current              | Reference: L1 bounds, bounded losing-block runs, adaptive rare-term seeks, hybrid phrase pilot.                          |
| No pilot             | Does the extra proof pass amortize on the wider phrase workload?                                                         |
| Eight short blocks   | Confirm all candidates in eight short-document blocks, for both top-10 and top-100.                                      |
| Bound-priority pilot | Nominate blocks by their phrase-score envelope instead of minimum document length, with the current confirmation budget. |
| Single-block skips   | Disable L1 jumps and losing-block coalescing, retaining safe L0 and candidate bounds.                                    |
| No rare-term seeks   | Retain SIMD block intersections even when the current global/local sparsity gates would seek.                            |

The bound-priority variant is a bounded region-ordering **pilot**, not the full
common-interval frontier proposed above. It visits disjoint blocks from the
phrase-bound term; that term is required by every phrase match. At most 4,096
metadata entries are inspected, with at most 64 blocks and 512 position
confirmations for small k, or eight blocks and 1,024 confirmations for larger k.
Only k distinct, position-verified matches prove a floor. The pilot emits no
hits and restores all cursors; the normal complete traversal still owns result
collection and stable-ID ties. No heuristic nomination authorizes a skip.
This isolates whether score envelopes guide the existing bounded work budget
better than short-document hints, without adding a writer or another scorer.

Ablations preserve the current bounds and cancellation checks wherever that
mechanism remains enabled. Compare optimized counts and ranked IDs/score bits
with exhaustive enumeration, including both ordinary and RGB indexes, before
admitting timing cells. Report unsupported/budget-exceeded queries explicitly.
Cross-engine count agreement and within-Summa algorithm equivalence are
separate gates: analyzer disagreements must not prevent testing Summa's own
skipping over the wider accepted workload. Measure CPU and anonymous/resident
memory alongside throughput, using unchanged indexes, compiler, flags, affinity,
and query-cache settings. These experiments do not change production defaults.

The [completed experiment](benchmark-results/skipping-2026-09-24/skipping.md)
finds a targeted 15.4% throughput / 13.7% CPU-per-request improvement for the
bound-priority pilot on RGB high-frequency phrase top-10, with negligible
aggregate phrase benefit. Disabling longer skips hurts high-frequency sloppy
phrase top-10 by 12.3% on plain and 21.3% on RGB, while helping medium phrase
top-100 by 6.6% / 5.0%. These results support measuring skip length, metadata
cost and threshold growth before designing selective admission. They do not
establish that a general priority frontier will win, or justify changing a
default from one architecture and short screening windows.

## Plain-index bound admission investigation

The expanded workload exposes two planner restrictions: an ordinary ranked
`TermQuery` enters MaxScore only with optional ratio bounds, and the typed
ranked conjunction admits block pruning only with a document map and ratio
bounds. Plain postings already carry conservative maximum-frequency and
minimum-length bounds. The implemented query-only change admits these existing
bounds without changing index configuration or encoded bytes. The term path
continues to use the existing windowed executor when ratio bounds are absent.
The conjunction path retains the existing typed intersection and canonical
query-order score reduction; exact counted traversal remains exhaustive.

The invariant is unchanged: skip only when a conservative bound is strictly
below the proven threshold. Equal-score stable-ID ties, deletions, global IDF,
field scoring parameters, positions and complete membership must retain their
existing semantics. Admission requires supported finite scoring bounds; it
must not turn negative boosts or complete callers into truncated streams.
The cost is per-block metadata checks versus avoided document decoding and
scoring, with existing bounded scratch and no new corpus-sized allocation.
The [completed 334-query comparison](benchmark-results/plain-bounds-2026-09-24/README.md)
validates plain and RGB on the preserved corpus, exact top-k IDs/score bits,
COUNT controls, and portable execution. Plain ranked terms improve 2.70–8.24×;
RGB high/high conjunctions improve 1.32–1.94×. Some conjunction families regress
1.9–5.1%, and the ARM late-winner fixture also exposes unproductive pruning.
Selective conjunction admission remains a measured follow-up, not an assumed
benefit of broader bounds.

## Remaining throughput gap investigation

The next experiments target query setup, single-term traversal and skewed
intersections independently. Trace remains HTTP adapter → core Searcher →
planner/scorer → immutable posting reader; native and portable queries share
these owners. No experiment may change scores, stable-ID ties, complete
membership, cancellation, or request budgets to improve a timing result.

1. Measure posting-open costs before replacing the copied L1 document/bound
   arrays with borrowed little-endian views. An opening should cost fixed
   envelope parsing, not a pass over every group. Writers and merges must
   preserve their existing bytes, and unaligned input must remain supported.
2. Compare the existing single-term block traversal against general windows
   on plain bounds. This is an admission ablation, not a second scorer. The
   potential saving is window setup and earlier group skips; the risk is extra
   metadata checks when bounds admit every block.
3. Profile common-list seeks driven by rare candidates separately from scoring
   and top-k collection. Prefer the owning reader's bounded block decoders and
   existing cursor lifecycle over another per-query posting representation.

Keep analyzer outliers in the full shared-input report and separately report
queries within 5% of reference counts. An analyzer mismatch can change a rare
conjunction into tens of thousands of matches; that is not equivalent work.
Any new analyzer/index experiment needs its own indexing and correctness
provenance. Compare each candidate on unchanged index bytes before combining
it with other changes, including CPU/request and anonymous RSS. Final retained
changes require native/async/WASM checks and both x86 and ARM measurements.

An independent analyzer experiment adds an opt-in `unicode_word` tokenizer:
UAX #29 word boundaries followed only by lowercase, preserving punctuation
inside Unicode words. The existing lexical tokenizer deliberately performs
additional splitting, elision, normalization and CJK expansion. Those defaults
remain unchanged. This experiment needs a newly built index and separately
reported counts; it cannot be credited as faster execution of the old index.
Tokens over 255 Unicode scalar values are skipped with their position retained.
It is not a claim of complete Lucene StandardAnalyzer compatibility (Unicode
versions, emoji and long-token policies can differ).

For prefix/wildcard/regex top-k, the retained bounded lazy union merges existing
posting iterators in document order, using the fact that every score is one. Only the established
ranked-candidate protocol may stop when later equal-score IDs cannot enter the
heap. Ordinary advance/seek remain complete for Boolean and count callers.
Mapped physical order retains materialization into stable document order.
Expansion limits stay unchanged; iterator scratch scales with the bounded
number of expanded terms, not the sum of postings. Exhaustive requests retain
the existing bitset/vector path to avoid heap work per matching document.
