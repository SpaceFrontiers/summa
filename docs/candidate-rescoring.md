# Candidate rescoring for linear L1 ranking

Status: opt-in Hermes implementation, 2026-09-05. Search API training and
activation are separate. Existing retrieval defaults remain unchanged.

## Objective and invariants

L0 retrieval cheaply nominates candidates from lexical, sparse, dense/binary,
and document-profile queries. L1 must evaluate the same feature queries for
every nominated item, regardless of which L0 branch found it. The resulting
bounded top-K document/passage set feeds an external cross-encoder (L2).
An absent L0 hit is not evidence of a zero score in another vertical.

Logical identities are `(segment, document)` and `(segment, document, ordinal)`.
Field-local physical IDs are not interchangeable: BMP and text BP reorder each
field independently. Document-profile ordinal zero is not body chunk zero.
Cross-field chunk alignment is an explicit caller contract, not inferred from
field names. Hermes preserves existing missing/multi-value semantics.

Filters and required quoted phrases define eligibility at L0 and remain hard
constraints. Feature weights cannot relax them. Exact candidate scores use
all retained scoring terms, without ANN nomination, LSP selection, heap-floor
pruning, or top-k truncation of the feature query. Scores remain exact with
respect to the stored representation (including sparse/vector quantization).

## Ranking modes

- RRF: existing rank-only fusion, kept for compatibility and paired baselines.
- Linear: bounded L0 union, complete raw feature backfill, fixed linear model,
  direct top-K. No RRF runs before or after the formula.
- Export: bounded L0 union plus raw features for caller-side model inference.

RRF and linear are mutually exclusive ranking policies. Formula weights are
not RRF branch weights. A model-bearing request cannot silently enter legacy
RRF when a backend lacks capabilities.

## Candidate and score model

Updated ownership decision: Hermes backfills raw feature scores, applies an
optional portable linear model to the full nominated union before truncation, and returns the raw features. Search API owns query intent,
training and model selection, and may apply a richer linear/CatBoost model
across separate query calls. The same versioned transforms and coefficients
can execute in Hermes and Search API. Learned weights never replace raw exports.

Each fusion branch has a unique `name`, an explicit `document`/`chunk` scope,
and the existing `Query` object. `l1.weights` references those branch names,
not schema field names or array indexes. The branch itself supplies the field,
tokenization, phrase offsets and vector; there is no separately maintained
scoring query that can drift from retrieval. A `score_only` branch computes a
feature without nominating candidates. A common `fusion.filters` applies hard
eligibility identically to every nomination branch; filters never become
features or independent votes. Initial L1 branches support one-field scoring
queries and SHOULD/Boost compositions; unsupported eligibility expressions
inside a scoring branch fail with guidance to use the common filter.

A branch omitted from weights still nominates and exports its feature, but
contributes zero. Unknown coefficient/transform names, duplicate or empty branch
names, all-zero models and non-finite values are errors. A schema field without
a query branch is neither searched nor scored. Missing candidate field data is
explicitly unavailable; failure to enter a vertical's top-K does not mean zero.
`l1` and legacy RRF coefficients are separate contracts. No fallback to RRF is
permitted for an invalid or unsupported linear request.

The bounded L0 union survives until Hermes L1 evaluation. Export-only requests
can return the whole union for external inference. Alternative reformulations
and multiple phrase spans use separately named branches, preserving raw scores.
Search API may share a logical coefficient across them by distributing it over
the branches (for example weight/N for the mean of N required phrase scores).
The artifact records that expansion policy; extra query calls cannot silently
add votes. Query/feature names and preprocessing form a versioned contract.

For a candidate passage c of document d:

```
passage_score(d,c) = bias + sum_i w_i * transform_i(raw_chunk_i(d,c))
                          + sum_j v_j * transform_j(raw_document_j(d))
document_score(d) = max over candidate passages c of passage_score(d,c)
```

A document-only candidate has an explicit document row, with no invented chunk
ordinal. Missing feature values have a presence bit and contribute zero; a
valid nonmatching lexical/sparse feature has score zero and is distinguished
from an unavailable field. Dense negatives remain valid values. Document
features are computed once and broadcast as context, not summed repeatedly
across chunks. Documents with many weak chunks cannot win by chunk count.
Returned ordinal scores are the same final passage scores used for selection.

Normalization is fixed in the model artifact, never local min/max over the
current shard or result page. Initial supported transforms should be identity,
signed log1p for unbounded lexical/sparse scores, and a configured affine
scale; trained parameters are derived from training data only. This permits
negative raw scores and yields comparable scores across shards. Validate
finite inputs, scales, weights, transformed values and final reductions.

## Passage nomination and diagnostics

Online L1 scores the union of passage ordinals actually nominated by the chunk
branches, plus document-scoped context once per candidate document. It does not
expand every stored passage of a book merely because one passage was retrieved.
A document-only candidate remains an explicit document row. Use score-only
document context branches in passage-search policy; document-discovery policy
uses document branches. Training labels this same nominated passage pool.
`score_export.all_passages` explicitly expands all stored ordinals for diagnostics.
Both paths retain the existing expansion/read/response budgets and never change
chunk extraction or AI context limits. Missing aligned fields remain unavailable.

## Exact feature execution and cost

Candidate rescoring belongs to core/query, with shared reader primitives for
address lookup. Adapters validate and translate; no second BM25 or vector
implementation belongs in the server, broker, or Search API. Reuse the existing
BM25/phrase frequency and length-normalization code and dense/binary SIMD
kernels. Sparse scoring probes the stored quantized impacts for every retained
query dimension, even when that dimension did not nominate the candidate.

Dense flat storage already supports logical document/ordinal lookup. BMP and
chunked text currently store primarily the reverse mapping (physical to
logical); a naive candidate pass would scan corpus-sized maps, or rerun full
retrieval, on every request. Neither is the intended steady-state cost.
Ordered maps are searched directly by logical ordinal. Reordered generations
carry an immutable `.lookup` sidecar with a versioned 32-byte header, 24-byte
field directory entries and sorted 12-byte `(document, ordinal, physical)` rows.
It stays file-backed and evictable. Ordinary merge streams/remaps prepared rows;
explicit Reorder constructs missing inverse maps under the existing reorder
memory budget. Publication, fsync and cleanup use canonical segment ownership.
Legacy reordered fields without the sidecar are reported by GetIndexInfo's
`unprepared_candidate_fields`; run Reorder before using L1 on them. MaxScore
sparse storage and ANN fields without stored flat vectors cannot be backfilled
by this version and fail explicitly. No extraction chunk limits change.

Batch candidates by segment, field and physical block. Reuse posting cursors
and vector buffers, prefetch only selected ranges, and keep scratch bounded by
candidate count and feature count. Bound feature query shape before opening
indexes or constructing scorers. No corpus-wide materialized scorer is an
acceptable substitute for candidate probing. The implementation must measure
legacy preparation cost separately from steady-state query cost.

## Distributed execution

For L1 requests, the broker preserves the candidate union until complete
feature scoring. A supplied model directly determines rank; RRF is an alternative, never an
implicit step after learned scoring. For
export-only requests the union survives until Search API scores it. An explicit per-branch nomination depth, separate
from the response limit, permits returning the whole bounded union. Each
retained hit carries its complete raw document/chunk feature scores. Local nomination includes every item
that could be in a global per-vertical top-K at the same depth; shard-local
union may be a superset. Search API sees all retained candidates before normalization/model inference.
Gather global text statistics for both nomination and scoring
queries, including phrase terms. Do not reuse shard-local phrase IDF.
Public pagination happens in Search API after model selection; Hermes feature
export addresses the complete requested candidate window.
Feature exports, ordinal scores, truncation and timing survive broker merging.

The core `Searcher::score_candidates` interface also supports diagnostics and
training over explicitly nominated candidate addresses. Candidate addresses refer to the
same immutable searcher snapshot; stale/foreign segment identities error
explicitly. It must never silently change the source set on a retry.

## Search API and training ownership

Hermes owns exact feature execution, validation, portable linear inference
instead of RRF, and raw exports. Search API owns query intent, original quoted
constraints, document versus passage profiles, training, model selection,
and any additional model inference across separate query calls. MCP, website and Cybrex inherit that policy. Ordinary Telegram keeps
its explicit document-discovery policy. Multiple API retrieval calls must
share a feature schema/model and deduplicate logical candidates before final
selection; where possible represent nominations in one Hermes request.

## Required validation

- Dense-only nominated chunk obtains independently verified BM25/sparse scores;
  lexical-only candidate gets dense/binary scores, including negatives/zero.
- Same doc, different chunk IDs; document context does not become chunk zero;
  missing fields, repeated query groups, sparse quantization, phrases and filters.
- Exact oracle comparison across segments, field reorder, merge and reopen;
  native/sync/async parity and portable compilation.
- Complete wire validation, global statistics, broker top-K/pagination, feature
  export limits, stale addresses, cancellation and resource exhaustion.
- Fixed-fixture scoring throughput and peak scratch on arm64/x86, plus live
  read-only paired quality/latency. Training/runtime formula parity and strict
  group separation. Required harness `check`, `full`, and WASM checks.

## Eligibility and bounded nomination

The common filter is represented separately from scoring. BM25 and BMP
collectors receive eligibility before their candidate heaps; filters contribute
neither scores nor ordinals. Vector ANN retains its existing bounded nomination
and is checked for eligibility before union, so selective filters may underfill
an ANN branch. Broader L0 depth is the recall control; raw backfill is exact over
the union that survives. No claim of exhaustive ANN recall is made.

Common filter bitmaps are capped at 16 MiB per segment and 64 clauses. Native
materializable text/phrase/fast-field filters use the existing bitset paths.
Other filters and portable builds use ordinary complete filter scorers only on
segments of at most 200,000 documents, failing explicitly above that bound.
This prevents an implicit corpus-sized scoring heap on a legacy backend.

## Client example

```python
result = await client.search("documents", query={"fusion": {
    "queries": [
        {"name": "body", "scope": "chunk", "query": {
            "match": {"field": "content", "text": "hemoglobin"}}},
        {"name": "title", "scope": "document", "score_only": True, "query": {
            "match": {"field": "title", "text": "hemoglobin"}}},
    ],
    "candidate_depth": 100,
}}, limit=100, l1={"weights": {"body": 1.0, "title": 0.2}}, score_export={})
```

Omit `l1` and keep `score_export={}` to collect the complete bounded union for
teacher labeling; set `limit` to cover all branch/shard candidates. Omitting
`score_export` avoids raw response maps while retaining the same linear rank.
An explicitly provided empty export object is meaningful. Omitted coefficients
are zero. Branch names and scopes survive both Python and TypeScript wrappers,
as do score zero, negative values, absent fields and the ranking-version marker.
