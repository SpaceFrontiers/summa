# Candidate rescoring for linear L1 ranking

Status: opt-in Hermes implementation, 2026-09-05. Search API training and
activation are separate. Existing retrieval defaults remain unchanged.

Logical addressing belongs to [BMP forward values](bmp-forward-index.md) and
the existing text chunk map. There are no `.lookup` sidecars. The historical
handoff describes the superseded draft implementation.

For reordered text, CHNK version 3 adds a kind-2 chunk section: the existing
physical document/ordinal/length columns followed by one `u32` physical slot
per logical-order row. This shares existing keys and lengths (4 additional
bytes/chunk) rather than duplicating logical keys in another file. Ordered
sections retain kind 0 with no extra bytes. Build/reorder constructs this
permutation; merge streams it with physical-slot base remapping. Legacy reordered
maps require explicit reorder, while ordered V1/V2 maps remain directly usable.
This is an addressing index of the existing chunk metadata, not a text scorer or
a separately published derived sidecar. Normal merge never sorts corpus-sized
metadata; all patch buffers are bounded. V3 rejects an unordered kind-0 section.
Pure legacy unordered maps retain V2 on ordinary merge; mixing them with
prepared maps fails with explicit Reorder guidance. Reorder upgrades legacy
chunk maps even when no BP permutation is planned, retaining token totals.
Its map columns, largest logical-slot permutation, length columns and retained
BP plans are admitted against the existing reorder memory budget.

## Objective and invariants

L0 retrieval cheaply nominates candidates from lexical, sparse, dense/binary,
and document-profile queries. L1 combines named branch scores for every nominated item, preserving organic
retrieval scores and optionally filling missing cells with the same queries. The resulting
bounded top-K document/passage set feeds an external cross-encoder (L2).
An absent L0 hit is not evidence of a zero score in another vertical.

Logical identities are `(segment, document)` and `(segment, document, ordinal)`.
Field-local physical IDs are not interchangeable: BMP and text BP reorder each
field independently. Document-profile ordinal zero is not body chunk zero.
Cross-field chunk alignment is an explicit caller contract, not inferred from
field names. Hermes preserves existing missing/multi-value semantics.

Filters and required quoted phrases define eligibility at L0 and remain hard
constraints. Feature weights cannot relax them. Backfilled candidate scores use
all retained scoring terms, without ANN nomination, LSP selection, heap-floor
pruning, or top-k truncation of the feature query. Scores remain exact with
respect to the stored representation (including sparse/vector quantization).

## Ranking modes

- RRF: existing rank-only fusion, kept for compatibility and paired baselines.
- Linear: bounded L0 union, optional missing-only backfill, fixed linear model,
  direct top-K. No RRF runs before or after the formula.
- Export: bounded L0 union plus raw features for caller-side model inference.

RRF and linear are mutually exclusive ranking policies. Formula weights are
not RRF branch weights. A model-bearing request cannot silently enter legacy
RRF when a backend lacks capabilities.

## Candidate and score model

L1 preserves scores returned organically
by each named retrieval branch. `l1.backfill` is optional and defaults to true;
it evaluates only missing branch/document or branch/passage cells. Setting it
to false performs no feature probes and leaves absent cells unavailable. Raw
maps omit unavailable keys, while actual zero and negative scores remain
present. `l1.missing_values` maps feature names to learned **raw** defaults.
Only missing cells use these defaults, before the feature transform and weight.
Without a configured default a missing cell contributes nothing, including no
transform offset. The exported raw map still omits an imputed feature; it never
pretends that a learned constant is an observed score. Defaults must be finite
and reference a feature with a coefficient. A real zero or negative value
always wins over its missing default.
Document-scoped retrieval scores retain their query's retrieval combiner;
backfilled document scores reduce all retained stored values. Approximate L0
scores are not relabeled as exhaustive scores or silently recomputed. Diagnostic
`all_passages` requires backfill. Training must bind the backfill policy along
with nomination settings because it changes the available feature population.

Hermes fills missing raw feature scores when requested, applies an
optional portable linear model to the full nominated union before truncation, and returns the raw features. Search API owns query intent,
training and model selection, and may apply a richer linear/CatBoost model
across separate query calls. The same versioned transforms and coefficients
can execute in Hermes and Search API. Learned weights never replace raw exports.

Each fusion branch has a unique `name`, an explicit `document`/`chunk` scope,
and the existing `Query` object. `l1.weights` references those branch names,
not schema field names or array indexes. The branch itself supplies the field,
tokenization, phrase offsets and vector; there is no separately maintained
scoring query that can drift from retrieval. A `score_only` branch supplies a
feature without nominating candidates; with backfill disabled its scores remain
missing and the model may substitute its learned default. A common `fusion.filters` applies hard
eligibility identically to every nomination branch; filters never become
features or independent votes. Initial L1 branches support one-field scoring
queries and SHOULD/Boost compositions; unsupported eligibility expressions
inside a scoring branch fail with guidance to use the common filter.

Exclusion-only common filters support both `Boolean(must_not: [...])` and
`Boolean(must: [AllQuery], must_not: [...])`. `AllQuery` includes documents
with missing metadata and uses a constant score of one when searched directly;
common filters never contribute scoring features. Its scorer uses constant
memory and shares the document-universe cursor with Boolean exclusion. Common
eligibility uses the existing bounded bitmap, with no storage or wire change.

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
effective_i = observed_i if present else learned_missing_i
contribution_i = w_i * transform_i(effective_i) if available else 0
passage_score(d,c) = bias + sum(chunk contributions at c)
                          + sum(document contributions at d)
document_score(d) = fusion.combiner({passage_score(d,c) for nominated c})
```

A document-only candidate has an explicit document row, with no invented chunk
ordinal. Missing feature values have a presence bit and use the configured
raw default (or contribute nothing if none is configured); a
valid nonmatching lexical/sparse feature has score zero and is distinguished
from an unavailable field. Dense negatives remain valid values. Document
features are computed once and broadcast as context, not summed repeatedly
across chunks. The default MAX reduction does not reward chunk count. SUM remains an explicit count-sensitive policy.
Returned ordinal scores are the same final passage scores used for selection.

Normalization is fixed in the model artifact, never local min/max over the
current shard or result page. Supported transforms are identity,
signed log1p for unbounded lexical/sparse scores, and a configured affine
scale; trained parameters are derived from training data only. This permits
negative raw scores and yields comparable scores across shards. Validate
finite inputs, scales, weights, transformed values and final reductions.

## Broker ownership of global selection

The broker is the coordinator for a logical index, including single-shard routes.
Each shard nominates at `candidate_depth` per branch; this depth is intentionally
not divided by partition count. Shards preserve organic scores, optionally backfill the bounded union and apply the
request's fixed L1 model. They retain at least `offset + limit` documents each,
which is sufficient for exact global top-K under the identical pointwise model
and document combiner. The broker reapplies the shared core formula, verifies
agreement and selects the global page. This reduces transfer without dropping
a possible global winner from the nominated pool.

MAX requires only the best passage feature row for each retained document;
weighted-top-k needs its top-k rows. AVG/SUM require all scored rows to reproduce
the document reduction. The shard export is widened to meet that requirement
before the broker reapplies the formula, then reduced to the caller's requested
export bound. Global BM25 statistics and fixed transforms remain mandatory.
No component normalizes against its own page or shard. Combined transport is
bounded to 64 MiB, divided across concurrently decoded shard responses, and
coordinator feature matrices are independently bounded to two million values.

Top-level RRF without the legacy vector reranker also runs at the broker: it globally merges each branch's nominated list
before calculating rank contributions. It does not merge shard-local RRF scores.
The transport uses `fusion.method = CANDIDATES` to return a bounded document
union and per-branch candidate scores/ordinals without fusion; `score_export`
additionally requests missing cross-vertical feature backfill. Raw feature-only
collection continues to return the complete union rather than ranking it.

The full per-shard nomination pools contribute to selection. A shared fixed
linear formula and exact shard-local top-K would be mathematically sufficient
for the same global top-K; moving the formula alone is not a quality claim.
Central ownership preserves the expanded pool for subsequent models and gives
RRF the correct global ranks. Broker CPU, combined rows, response bytes and
in-flight shard work must be bounded independently of each shard's limits.
Missing or incompatible shard exports fail the request instead of returning a
partially ranked pool. The same core inference and fusion implementations are used on each path.
Legacy nested fusion and fusion with the vector reranker retain their previous
shard execution; their results do not advertise `global_rrf_v1`.

## Combiners and learned fusion

Combiners retain their existing ownership. A vector query's combiner reduces
its stored field values to one document feature; a chunk feature is the raw
score of that ordinal and needs no within-field reduction. Text branches use
MAX over chunked values, or their one ordinary document value. Backfill evaluates
all stored values for a document-scoped feature, including present nonmatches
with zero scores. Retrieval's approximate or limited nomination may have seen
only a subset of those values; organic scores retain that retrieval behavior. They are never overwritten by
a backfill pass. Training must match this serving policy.
Boost and SHOULD composition preserve their expression order: boosting a reduced
document feature differs from reducing negatively boosted passage scores.

L1 combines named features at each passage. The existing `fusion.combiner` then
reduces those final passage predictions to a document score **before** top-K or
response passage truncation. Unset/0 retains fusion's existing MAX default;
MAX, AVG, SUM and normalized WEIGHTED_TOP_K retain their existing meanings and
parameters. The legacy fusion field cannot distinguish an explicit softmax enum
zero from unset; its existing MAX interpretation is unchanged. Vector branches
continue to support parameterized softmax and weighted-top-k normally.

The Search API answer profile chooses MAX because its document teacher is the
best answer passage. This is a profile choice, not an engine restriction. The
artifact must bind nomination combiners, document-feature combiners and the
final document reduction along with the feature queries. Training and serving
must use the same reductions and the same nominated-passage population. For
example, average over the exported top three is not average over the complete
scored union. Raw exports retain all required rows for offline evaluation.

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

Dense flat storage already supports logical document/ordinal lookup. Current BMP
uses quantized forward vectors inside `.sparse`; CHNK V3 uses addressed chunk
metadata inside `.chunks`. Ordered legacy maps can be searched directly;
Unordered BMP maps without forward storage and unordered text maps require
explicit Reorder before their missing cells can be backfilled. BMP storage is optional per field via
`bmp_forward_index` (default true). With it disabled, ordered BMP maps locate
candidates by binary search and score their query-term postings in selected
blocks. BP-reordered BMP maps need this setting enabled and explicit
reorder/rebuild to regain backfill capability. Disabling storage does not affect
full-text or sparse MaxScore backfill. See [optional forward storage](bmp-forward-index.md#configuration-and-scoring).
`GetIndexInfo.unprepared_candidate_fields`
reports these fields and ANN fields without stored flat vectors. Disabling
backfill, or supplying every needed organic cell, requires no address probes
and can use those fields without migration. No extraction limits change.

Candidate BM25 statistics read document frequencies from term-dictionary metadata
on native, async and WASM paths; gathering statistics never materializes posting
payloads ahead of scoring admission.

Full-text BM25, text MaxScore and phrase branches use the existing posting/skip
readers, phrase positions and shared global statistics. Sparse MaxScore fields
also support backfill: candidate-intersecting blocks in the existing skip index
establish stored field/ordinal presence, and query dimensions reuse the owning
quantized block scorer. A present nonmatch is zero; a missing value is absent.
Presence discovery may visit every active dimension's metadata; it does not
load complete posting lists. A request-wide budget caps sparse probes at two
million and encoded sparse reads at 256 MiB, checked before I/O. Exceeding a
budget is an explicit error. BMP forward lookup avoids this all-dimension
metadata work and reads only selected stored vectors.

Feature matrices and scored components are capped at two million values;
exact dense/binary reads are capped at 1 GiB. Nomination retains existing
candidate/ordinal caps, checked before cloning the union. On lazy remote
backends, text posting and position reads are admitted together against a
256 MiB request budget before materializing ranges. Mmap/RAM views do not
materialize those lists; their existing cursors decode selected blocks only. Raw response hydration
and broker transport have independent budgets.

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

The core `Searcher::score_candidates_with_retrieved` interface accepts borrowed
named branch lists to preserve organic scores. `Searcher::score_candidates`
supports diagnostics without organic inputs and
training over explicitly nominated candidate addresses. Candidate addresses refer to the
same immutable searcher snapshot; stale/foreign segment identities error
explicitly. It must never silently change the source set on a retry.

## Search API and training ownership

Hermes owns feature execution, validation, portable linear inference
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
result = await client.search(
    "documents",
    query={
        "fusion": {
            "queries": [
                {
                    "name": "body",
                    "scope": "chunk",
                    "query": {"match": {"field": "content", "text": "hemoglobin"}},
                },
                {
                    "name": "title",
                    "scope": "document",
                    "score_only": True,
                    "query": {"match": {"field": "title", "text": "hemoglobin"}},
                },
            ],
            "candidate_depth": 100,
        }
    },
    limit=100,
    l1={
        "weights": {"body": 1.0, "title": 0.2},
        "backfill": True,  # Default; only missing cells are scored.
        "missing_values": {"body": -0.1, "title": 0.25},  # Illustrative raw defaults.
    },
    score_export={},
)
```

Omit `l1` and keep `score_export={}` to collect the complete bounded union for
teacher labeling; set `limit` to cover all branch/shard candidates. Omitting
`score_export` avoids raw response maps while retaining the same linear rank.
An explicitly provided empty export object is meaningful. Omitted coefficients
are zero. Branch names and scopes survive both Python and TypeScript wrappers,
as do score zero, negative values, absent fields and the ranking-version marker.

Set `backfill=False` to rank from organic scores and learned defaults only.
TypeScript uses `backfill` and `missingValues` in `l1`. The protocol preserves
optional-boolean presence, so omitted and explicit false remain distinct.

The serving contract is `candidate_scoring_version = 2`, with response markers
`linear_v2` and `feature_export_v2`. Broker inference uses the same model,
including defaults, and rejects old or mixed ranking markers. RRF candidate
export retains `fusion_candidates_v1`; global RRF retains `global_rrf_v1`.
Deploy server and broker support together before activating L1; there is no
mixed-version fallback to rank fusion. Current BMP and CHNK V3 generations
require readers that understand these formats.

### Exclusion-only common filters

A nonempty Boolean `must_not` list with no positive clauses denotes every
segment document except the excluded matches. An entirely empty Boolean still
matches nothing. Fusion applies this eligibility before each branch selects
its candidates; exclusions create neither scores nor passage ordinals. Native
indexed terms materialize the complement in the bounded document bitmap,
including when an excluded term is absent. The async/portable scorer streams
the same complement without an all-document result heap.
