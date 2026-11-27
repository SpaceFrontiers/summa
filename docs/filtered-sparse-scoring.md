# Filtered and floored BMP sparse scoring

Status: implemented in 2.2.7. Both mechanisms change which rows a BMP sparse
query scores; neither changes a format.

## Exact scoring for selective filters

### Problem

BMP applies a document filter to slots after it has scored their block. When a
filter leaves a few hundred documents, almost every scored slot is rejected, the
heap never fills, no bound prunes, and each segment scores up to the γ cap of
superblocks before returning little or nothing. A production index with 28
segments per shard logged 0.5–1.5 s per segment for
`+type:message +issued_at:[one day] sparse(...)` (`sbs=3000/77341,
blocks≈22000, returned=0, threshold=0`), 12–26 s per query. The cap also ends the
traversal before it reaches eligible documents outside the selected superblocks,
so those matches were lost, and a query-global plan skipped whole segments.

### Mechanism

The planner already materializes the filter as a `DocBitset`, so its
cardinality is known. When at most `EXACT_FILTER_MAX_DOCS` (4096) documents of a
segment are eligible, and they are at most 1 in 64 of its documents,
`score_bmp_eligible_exact` resolves each document's rows
(forward directory, or the ordered doc map of a never-reordered segment), scores
them with the candidate point kernel, and keeps the best `k`. It shares
retrieval's integer quantization, so a row's score equals the traversal's.

- Applies to `Boolean` MUST/MUST_NOT bitsets and to `FilteredQuery`/alive-document
  eligibility. An opaque predicate without a bitset keeps the traversal.
- A wider filter, or the alive documents of a small segment with deletions,
  keeps the traversal: its heap fills quickly and its bounds then prune, which
  the exact pass cannot.
- Bounded: at most 65,536 rows per segment pass. A larger expansion (long
  documents) or a reordered segment without forward values returns to the
  traversal; nothing is dropped.
- A query that restricts documents (`Query::restricts_documents`) now reaches
  segments in which the global LSP plan selected nothing. Those segments run only
  the exact pass; a wider filter there still returns no rows, as before.
- The exact pass ignores the cross-segment floor and never publishes one. It
  returns a superset of what a floored traversal would.
- Observable as `summa_bmp_exact_filter_segments_total{index, field}` and a
  `[planner] exact filtered sparse BMP` debug line.

Result change: selective filtered sparse queries now return every eligible
matching row, ranked exactly. Before, they returned the subset the capped
traversal happened to reach. Regression:
`test_bmp_selective_filter_scores_eligible_documents_beyond_the_gamma_cap`
(single and multi-valued, one and two segments, both filter forms); with the
exact pass disabled it returns no hits.

Not measured: latency on the production shape after the change. The work is
bounded by eligible rows instead of γ superblocks per segment.

## Opt-in document floor for multi-valued fields

A multi-valued segment with an averaging combiner used no cross-segment floor:
its raw heap ranks values, not documents, so it could neither publish nor
consume the document floor, and every segment filled its own `2 × limit` heap
(≈42,000 doc-map lookups per query at depth 300 over 25 segments).

`IndexConfig::sparse_document_floor` (`summa-server --sparse-document-floor`,
default off) lets such a segment consume the floor the searcher raises from
completed segments. Averaging combiners (`Max`, `Avg`, `LogSumExp`,
`WeightedTopK`) never score a document above its best value, so a document whose
every value is below the floor cannot enter the window. `Sum` never consumes it.
The raw heap still never publishes.

It is opt-in because it is not rank-identical: a document that stays loses its
values below the floor from its combined score, and from the ordinals it
reports. Enable it on one shard and compare `summa_bmp_blocks_scored_per_query`
and a retrieval-pool evaluation before making it the deployment default.
Regression: `test_bmp_document_floor_prunes_multi_value_segments_only_when_enabled`.
