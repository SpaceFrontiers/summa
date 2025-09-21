# Score-guided posting traversal: research and Hermes fit

September 24, 2026. Research assessment and proposed follow-up, not a claim that
the proposed index structures have been implemented. The current measured work
is tracked in [ranked pruning](ranked-pruning-followup.md).

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

| Work                                                                                                                                                                                                         | Relevant mechanism                                                                                                                                                            | Hermes implication                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [Ding and Suel, SIGIR 2011: Block-Max indexes](https://research.engineering.nyu.edu/~suel/papers/bmw.pdf)                                                                                                    | Per-block score upper bounds allow WAND to bypass postings without scoring them.                                                                                              | The basic score-aware skip mechanism already exists in Hermes.                                         |
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

Hermes text postings already have block and L1 group bounds, including optional
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

A smaller independent experiment is Longer Skipping over the existing L0/L1
bounds: retain forward traversal, but coalesce consecutive losing regions into
one seek. Persisted scalar-score hints would need to remain valid under Hermes's
request scoring parameters and global statistics; do not assume that a BM25
ordering precomputed for one configuration holds for another.

This is a proposal inferred from the research and Hermes's current design.
Existing group boundaries differ between terms; summing unrelated block maxima
does not establish a bound over an arbitrarily larger common interval.

The query-time cost should be proportional to the bounded number of inspected
region summaries times the query's term count, plus payloads in visited regions.
A disjoint interval frontier avoids a corpus-sized visited bitmap. Admission
should retain the ordinary iterator for small lists where scheduling overhead
cannot be amortized. An offline exhaustive audit can compare each region's bound
with its best actual score and record threshold growth against work performed.

Phrases need their own bound: term occurrence does not imply adjacency. Hermes
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
