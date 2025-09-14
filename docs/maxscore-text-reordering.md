# MaxScore text reordering

Implemented field-local text RGB with standalone and merge-time planning,
map-preserving copy merge, and compaction. Query execution preserves stable
document IDs and canonical scores while traversing physical posting order.

## Required behavior

MaxScore text must follow BMP's field-local locality model. An opted-in field
owns a physical scoring-unit order and resolves hits back to stable document
IDs and value ordinals. Store, fast fields, unrelated text and vector fields
retain their existing IDs and bytes. `reorder` selects fields; the existing
`reorder_on_merge` policy, BP budgets, concurrency gate, cancellation and output
claims govern merge-time work. Ordinary merges continue copying compatible
encoded blocks. Reordering is an explicit exception requiring re-encoding.

Plain text remains one scoring unit per document, including aggregated values;
chunked text remains one unit per value. Reordering must preserve IDF, average
lengths, raw score bits, position/ordinal semantics, exact membership/counts,
query-global statistics and tie ordering. In particular, plain fields must not
inherit the chunked field's 90th-percentile BM25 length floor.

## Ownership and representation

The scoring implementation keeps public construction, dispatch, cursor state,
and shared scratch in `query/scoring.rs`. Private `scoring/conjunction.rs` and
`scoring/windows.rs` modules own intersection and window traversal respectively;
both implement the existing `MaxScoreExecutor` and use the same cursor, BM25
batch scorer, collector, and scratch. Ranked conjunction entry and the proven
union tail share one pruning dispatch. Exact counted conjunctions always choose
exhaustive traversal. Unit regressions live in `scoring/tests.rs`.

This source decomposition preserves score operation order, strict pruning
comparisons, logical-ID tie ordering, cancellation checks, public paths, and
persisted bytes. It adds no allocation, cache, dynamic dispatch, or traversal:
scratch remains bounded by `MAX_QUERY_TERMS` and the existing window/block sizes.

Extend `segment/chunk_map.rs` and `segment/text_reorder.rs`; do not add another
permutation writer or query executor. A plain-text map records one physical
slot per document with ordinal zero and unfloored document lengths. Persist a
distinct, version-gated section kind so older readers reject data whose physical
IDs they would otherwise mistake for document IDs. Existing unpermuted segments
remain readable, including when merged with mapped sources.

The segment reader distinguishes physical mapping from chunked scoring semantics.
Query planning uses the map for physical traversal, predicates and target probes;
corpus statistics continue using the declared scoring unit. Existing mapping,
complete-membership, position and result-folding owners handle translated hits.
Plain token positions must survive folding, rather than becoming chunk ordinals.

Merge planning computes one permutation per opted-in field over the surviving
source units before text output is written. The existing posting/position merge
writer consumes this plan, and the map writer emits the matching mapping. It
must not write a first full generation just to reopen and reorder it. No new
publication protocol or ANN rebuild is introduced. Permutation, CSR, term and
position scratch must be charged to the existing budget or spilled in bounded
runs; cancellation must discard unpublished output through its current owner.

## Merge review: bounded planning and norm preservation (September 17)

Standalone text reorder must retain untouched norm encodings, including byte
norms, through the owning chunk-map writer. It may not expand an unrelated norm
column merely because another field is reordered. New-column, copied-column and
merge output share the existing CHNK format; no additional file format is needed.

The text BP planner must charge retained field plans, vocabulary discovery,
posting input, pair/CSR construction and graph scratch before allocating them.
It reuses BMP's frequency selection and candidate fitting policy. Fixed scratch
that cannot fit is an explicit budget error. Graph construction rejects invalid
physical IDs and observes cancellation during dictionary and posting traversal.
A term rewrite must similarly admit its retained postings/positions and encoder
scratch before decoding. These checks bound lifecycle work; they do not alter
query scores or introduce a new reorder algorithm.

Explicit reorder repacks position ranges through the existing stream encoder so
permuted document boundaries do not force partially filled output blocks. The
range reader retains only bounded block scratch. Ordinary compaction keeps its
encoded-block copy policy; reorder is the explicit rebuild exception. Regression
coverage compares the repacked bytes with a fresh encoding of the same deltas.

## Integrated merge-time implementation (September 17)

The merge text stage builds BP directly over concatenated source physical
units before opening its output. A shared bounded k-way dictionary traversal
serves both posting merging and BP vocabulary/edge passes. Standalone and
merge-time text use the same planner, permutation validation, and term rewrite.
Source-local cursors retain their source index until position ranges are written.
Only opted-in fields rebuild; all others keep the encoded-copy path.

The CHNK merge writer accepts validated order/inverse slices. It streams
permuted source columns and translates each source logical slot through the inverse,
without allocating another complete map. Legacy plain inputs use the existing
budgeted identity migration; missing norms with nonzero tokens remain an error.
Permutations, source directories, source block buffers, and retained field plans
share the configured BP scratch limit. Map writing follows term rewriting to
avoid overlapping two budget-sized scratch allocations. Text plans and the
shared reorder permit are released before BMP work. Text and BMP convergence
jointly determine publication metadata; truncated BP remains observable.

Tests must compare against copy-merge plus standalone reorder, including raw
payload bytes, scores/positions, missing and multi-valued fields, mixed source
layouts, multiple fields, cancellation and failed output cleanup. No second
segment generation, new format, or new publication path is permitted.

## Candidate probing and canonical window accumulation (September 17)

The text-window executor preserves frozen RGB bytes and canonical f32 document
scores. Native sync and async search share this owner; WASM uses the portable
implementation. None of these reader changes introduce a format, configuration
default, second scorer, or retained cache. Measured selection and rejected

Optional candidate probes retain every input candidate. The shared membership
probe specializes that distinction: required membership compacts matched IDs
and scores, while optional membership changes only matching scores. A lower-bound
search is unnecessary when the current posting already reaches the candidate.
Posting slots use bytes, with a compile-time assertion on block capacity. Both
instantiations preserve block-level and 64-candidate deadline checks and use the
same deferred TF decoder and batch scorer.

When every term is essential and the prepared scorer proves finite nonnegative
contributions, visiting terms in canonical query order makes the dense window
accumulator exact. Absent positive-zero additions cannot change that sum. These
windows omit duplicate contribution-plane writes and the final fold. Windows
that defer any term retain the existing per-term storage and canonical fold;
the existing two-term commutative case also retains its admission.

### Bounded window setup and cost-aware partitioning

When the current posting block covers the window, the bound lookup returns its
cached bound without searching the skip directory. The existing whole-group
substitution and zero clamp remain intact. Other windows use the same
multi-block/group traversal and bound arithmetic.

Mapped OR queries with several globally essential terms amortize small windows.
The executor tracks evaluated windows and candidates before optional probing.
An average below 32 candidates per term doubles the minimum span, capped by the
existing 4,096-ID scratch; adequately populated windows reset it to one. Required
queries and unmapped execution retain their original spans. Every enlarged
window still computes conservative bounds over its full range. The motivating
algorithm is Lucene's `computeOuterWindowMax` in
[MaxScoreBulkScorer](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/MaxScoreBulkScorer.java).

For finite nonnegative mapped OR scorers, local bounds are ordered per posting-
list entry. Reciprocal list costs are prepared once per query in a fixed
`MAX_QUERY_TERMS` array (eight bytes per term slot on the stack). The existing
prefix-bound proof and canonical scorer retain responsibility for correctness;
there is no second executor, unbounded scratch, or retained cost cache.

### Ranked mapped-pair pruning

Two-term semantic AND top-k and proven OR-to-conjunction tails reuse the same
compile-time pruning mode in the existing typed intersection. Admission requires
a mapped field, exact heap factor, finite prepared bounds and ratio metadata on
both lists. All other conjunctions and **every exact-count traversal** retain
the exhaustive instantiation.

Until the earlier current block end, every possible match is bounded by the sum
of those two block bounds. A sum strictly below the existing conservative heap
threshold advances the block ending first; equal ends advance both. This always
makes progress and can avoid decoding TFs and later payloads. Equal-score bounds
remain eligible, preserving stable logical-ID ties. The scorer, canonical sum,
collector, cursor protocol and deadline checks remain shared. The count returned
by a pruned traversal is a work statistic, never an exact result count.

Regression tests compare public sync/async ranked results and exact-count
results, preserve ties, and observe expired budgets. A strong-prefix fixture
scores 256 of 8,192 matches; tied scores still require all 8,192, and counted
execution remains exhaustive in either case. Final validation and independent
paired timing are recorded in the performance review.
