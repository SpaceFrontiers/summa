# Physical tie order for constant-score queries on reordered fields

September 30, 2026. Status: **implemented**.

## Problem

Prefix, wildcard and regex queries score every match 1.0, so their ranked
top-k is decided entirely by the tie order, which is the logical (stable)
document ID. On an RGB-reordered field postings follow physical order, so the
`k` smallest logical IDs are known only after every match has been mapped:
pattern top-k runs at 0.02–0.05× of an ordinary build on the 10M-document
benchmark ([RGB campaign](benchmark-results/rgb-2026-09-29/README.md),
[pairs campaign](benchmark-results/pairs-2026-09-29/README.md)). Lucene and
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
- Searchbench `audit`: its exhaustive rank oracle orders ties by logical ID,
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

## Measurements

1M documents of the benchmark corpus, RGB-reordered, `searchbench_http
diagnose` (`PROBE_LIMIT=10`, 20 iterations, best of two A B rounds, Apple M4,
single-threaded); the baseline binary is byte-identical to the previous
pairs build. Totals in µs per family pass:

| Family        | Default build |    RGB before |     RGB after |
| ------------- | ------------: | ------------: | ------------: |
| wildcard (11) |           474 | 2,820 (0.17×) |   511 (0.93×) |
| prefix3 (2)   |            28 |   249 (0.11×) |    43 (0.66×) |
| regex (4)     |         1,340 | 1,969 (0.68×) | 1,505 (0.89×) |

Per query 2.8–46× faster, except `[jkqxz][a-z]*ess`, whose cost is the
dictionary scan (1.0×). Exact counts are unchanged (same results, ±1%), and
316 scored queries (AND, OR, phrase) return identical IDs and score bits.

10M documents, the RGB + pairs and default + pairs builds of the
[pairs campaign](benchmark-results/pairs-2026-09-29/README.md), x86 benchmark
host, `searchbench_http diagnose` pinned to one core, previous binary (`wp3`)
against this tree, best of two A B rounds. Sums of per-query median search
time over the audited pattern queries, in ms:

| Family             | Limit | Default before | Default after | RGB before | RGB after | RGB speedup | RGB speed vs default |
| ------------------ | ----: | -------------: | ------------: | ---------: | --------: | ----------: | -------------------: |
| prefix3 (2)        |    10 |           0.17 |          0.18 |       5.42 |      0.17 |       32.8× |                1.07× |
| prefix3            |   100 |           0.19 |          0.19 |       5.49 |      0.18 |       31.0× |                1.06× |
| wildcard (11)      |    10 |           3.78 |          3.76 |      60.19 |      4.03 |       14.9× |                0.93× |
| wildcard           |   100 |           3.84 |          3.81 |      60.93 |      4.10 |       14.9× |                0.93× |
| regex (4)          |    10 |          10.96 |         10.77 |      25.17 |     11.65 |        2.2× |                0.92× |
| regex              |   100 |          11.02 |         10.68 |      24.86 |     11.82 |        2.1× |                0.90× |
| wildcard_scan (42) |    10 |          45.63 |         45.33 |      80.46 |     50.36 |        1.6× |                0.90× |
| wildcard_scan      |   100 |          45.93 |         45.65 |      80.72 |     50.76 |        1.6× |                0.90× |

Every query returns the same number of hits on both binaries; the default
build is unchanged (±2%). Pattern top-k on the RGB build is now within 10% of
the default build instead of 0.02–0.05× of Luxir; scaling the pairs
campaign's default-build Luxir ratios (prefix3 0.95×, regex 1.12×, wildcard
1.13×, wildcard_scan 5.33× at TOP_10) by the last column puts the RGB build
at roughly Luxir's level. The [throughput campaign](benchmark-results/ties-2026-09-30/README.md)
(126M requests) confirms it: RGB + pairs pattern top-10 runs at 0.98×
(prefix3), 1.14× (regex), 1.01× (wildcard) and 5.03× (wildcard_scan) of
Luxir, up from 0.02–0.05× (wildcard_scan 2.8×).
