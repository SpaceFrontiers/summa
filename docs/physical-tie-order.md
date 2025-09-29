# Physical tie order for constant-score queries on reordered fields

September 30, 2026. Status: **implemented**.

## Problem

Prefix, wildcard and regex queries score every match 1.0, so their ranked
top-k is decided entirely by the tie order, which is the logical (stable)
document ID. On an RGB-reordered field postings follow physical order, so the
`k` smallest logical IDs are known only after every match has been mapped:
pattern top-k runs at 0.02–0.05× of an ordinary build on the 10M-document
other engines order ties by internal (physical) document ID, which makes the
same query stop after `k` matches.

## Rule

On a text field with a document map (plain RGB reordering), a **constant-score
query** returns its matches in the field's **physical order**: ties among its
equal scores break by physical slot instead of logical document ID. Every
other query keeps the logical-ID tie order.

Constant-score queries are those whose matches all receive the same score:
prefix, wildcard and regex queries, a boost of one, and a Boolean query with
exactly one scoring clause that is itself constant-score (plus any number of
exclusions). A Boolean mixing a BM25 clause with a pattern keeps logical ties.
Scored queries are unchanged because their ties are rare and their executors
already prune well; changing them would touch every MaxScore path, reranking
and fusion.

The order is total and deterministic for a given segment: (score, segment,
physical slot). Offset pagination stays consistent, because the top `n + m`
of a total order extends its top `n`. Physical order depends on the build, so
the chosen tied documents may differ between an ordinary and a reordered
build of the same corpus; every tied document is an equally correct answer.

## Design

- **Query:** `Query::constant_score()` (default: `is_filter()`), overridden by
  Boolean and boost queries per the rule above. Term expansions opt into
  physical traversal for ranked streams as well as complete ones
  (`physical_union_field`), and a Boolean query does whenever it is
  constant-score.
- **Per segment:** the searcher's top-k paths (`search_segment_shared_planned`
  and its sync form) collect a constant-score query on a document-mapped field
  in physical IDs: the heap compares physical slots, the competitive-candidate
  certificate holds (`allow_equal = false`), so the stream stops once `k`
  matches are held; afterwards each hit's ID is translated to its logical
  document without re-sorting. Deletions are still checked by logical ID.
  Collectors supplied by callers (`collect_segment_with_limit`) keep receiving
  logical IDs and the logical tie order.
- **Across segments:** the core merges compare hits of different segments
  only, and ties there are decided by segment, so each segment's order is kept.
- **Broker:** the partition merge sorts by score and then segment only; its
  stable sort keeps each shard's order within a segment, so a document's
  logical ID no longer overrides it.

## Validation

- `constant_score_queries_on_reordered_fields_rank_ties_in_physical_order`:
  on an RGB-reordered field, prefix, wildcard and regex top-k (alone and as
  the only clause of a Boolean query) return the first `k` matches in physical
  order, translated to logical IDs, sync and async; consecutive offset pages
  concatenate to a longer page; deletions are honoured; a Boolean mixing a
  term and a pattern keeps logical ties.
- `physical_ties_merge_across_segments_by_segment_then_physical_order`: two
  reordered segments merge by segment, each in physical order, including an
  offset page across the boundary.
- `expanded_term_counts_on_rgb_fields_traverse_physical_ids`: counts are
  unchanged and ranked expansions opt into physical traversal.
- Broker: `tied_hits_keep_each_shard_order_within_a_segment`.
  so for constant-score queries it now requires the oracle's score sequence
  and distinct matching documents instead of identical IDs; every other query
  keeps the exact ID check.

The mixed-Boolean check exposed an older bug on every build: a ranked
conjunction of a filter-like clause and one scored clause handed the scored
clause the top-k limit even when the filter had not been pushed into its
eligibility (the push-down only runs for multi-term or chunked clauses), so
`alp* AND beta` intersected the prefix with the term's own top-k and could
return nothing. Such clauses now get complete membership unless the push-down
succeeded (`ranked_must_filter_and_term_returns_filtered_top_k`, which also
covers a `MUST_NOT` filter).

