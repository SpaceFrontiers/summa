# Summa 2 namespace release — 2026-09-25

The [migration guide](summa-2-migration.md) records the breaking package, RPC,
configuration, metrics, browser-storage and training-artifact namespaces. This
change carries forward development main at `337db5d390572bf1653e17b9987850b28aecd481`.
Search-index format versions and binary fixtures are unchanged. The TQ
fingerprint regression initially detected a changed domain prefix; freezing its
precomputed FNV seed restores the existing persisted fingerprint without a
branding-dependent string. Both protocol schemas differ only in namespace text;
field numbers and types are unchanged. Server and broker packages obtain their
build inputs from the single `summa-proto` source crate.

The search `check` harness passes all five stages, including strict Clippy,
2,075 tests (25 ignored), native-without-sync and standalone broker compilation.
The WASM release build and 41 JavaScript tests pass; TypeScript has 17 passing
tests, Model Lab has seven, and the search UI has two. Both web applications
build. Documentation links and package source builds pass. Full RPC validation
is recorded below once complete. No performance improvement is claimed or
benchmark defaults changed.

Historical evidence labels were mechanically renamed. The retained split ZIP
has regenerated member/archive checksums and records its original capture hash;
its analysis still reproduces the reported numerical measurements. External
capture hashes continue to identify original artifacts. The chart plot pixels
are retained with replacement vector axis labels, avoiding regenerated curves.

Remaining review item: broader GPU/backend CI is required for the renamed ML
crates; local search checks do not establish accelerator support or measure
runtime performance. Package publication and Pages deployment must be verified
against their public destinations after the repository replacement.

# Core/server review — 2026-09-05

September 24 expanded-query follow-up: [the completed 826-query coverage and skipping campaign](benchmark-results/skipping-2026-09-24/README.md)
times 684 successful shared inputs across five deployments, plus 432 skipping
cells over two reversed-order rounds. All 2,558 exhaustive audits and 15,348
HTTP checks pass; all 702 timing cells have zero request errors. A bounded
score-priority pilot improves RGB high-frequency phrase top-10 by 15.4% with
13.7% less CPU/request, but leaves the overall phrase average nearly unchanged.
Disabling longer skips regresses high-frequency sloppy-phrase top-10 by 12.3%
plain / 21.3% RGB. Other aggregate gains are small relative to control drift;
no production default changes. Remaining priorities are single-term/Boolean
profiling, bounded pattern execution for 142 explicit budget failures, and
analysis compatibility: some conjunctions perform radically different work.
Four harness regressions, the full published-lock search check, Linux candidate
regressions and all six WASM builds/41 JavaScript tests pass. Artifacts were
checksum-verified and both machines independently confirmed stopped.

September 24 query compatibility: [whole-term regex queries and escaped literals](regex-query.md)
add the 13 missing benchmark regex expressions and 12 punctuation/escape cases.
Regex and wildcard share bounded dictionary matching and the existing union
scorer, including constant scores, logical RGB IDs and chunked-field rejection.
Pattern calls reject unindexed/nontext/unknown fields explicitly. The parser
unescapes literal terms once, preserving field analysis, and rejects malformed
regex calls and dangling escapes. No format, schema, worker or scoring default
changes; no index rebuild is needed for the query types.

The [reproducible HTTP probe](benchmark-results/query-support-2026-09-24/README.md)
accepts 826/826 expressions on the unchanged four-document fixture, up from 801.
Every previously accepted response is unchanged. Regression tests first failed
on missing regex behavior and punctuation parsing, then passed for all 13 regex
patterns with nonempty exact-ID/count oracles, Unicode/case/anchoring, Boolean
composition, duplicate terms, RGB mapping, invalid syntax and expansion errors.

Validation: the search `check` harness passes all five stages, 2,054 tests
(25 ignored), strict Clippy, native-without-sync and standalone broker checks.
All eight focused regex/wildcard tests also pass with async-only native features.
The WASM release build and all 41 JavaScript tests pass. Documentation and Python
checks pass. No lifecycle or wire protocol changed; the full RPC harness was not
rerun. Capability checks are not performance measurements. Remaining work is
full-corpus count/ranking agreement, standard-analyzer compatibility, sloppy-phrase
semantics and broader bounded expansion; the throughput gate remains 15/826.

September 24: the [ranked conjunction and phrase follow-up](ranked-pruning-followup.md)
retains fixed-block cursor seeks, locally guarded rare-term conjunction seeking,
certified phrase bounds/singleton probes, serial two-term native reader setup,
and a bounded score-floor pilot for expensive mapped ranked phrases. On the
same immutable 10M indexes, two reversed-order rounds improve RGB low-phrase
top-10 throughput 155.0% and medium-phrase top-100 44.0%; ordinary
conjunction top-10 changes +3.1% and still trails fresh Luxir. Exact audit and
response bytes agree. The report includes CPU/request, RSS and anonymous RSS,
count controls, rejected variants and measured ARM regressions, including the
noisy +17.8% RGB rare-conjunction/top-100 cell. This targeted 15-query run is not
the complete 826-query suite or an isolated RGB ablation. No persisted format,
schema or public configuration default changes; no reindex is needed.

Validation passes the search `check` harness (2,049 tests, 25 ignored), strict
Clippy, native-without-sync, focused async regressions, diagnostic-feature checks,
and the WASM build plus 40 JavaScript tests. Remaining work includes all-query,
cold, multi-segment and further ARM measurements. Evidence was verified and both
cloud machines were confirmed stopped. The separate [score-guided traversal research](score-guided-traversal.md)
describes existing bounded longer skips and proposes query-dependent region
ordering; no new persisted hierarchy is claimed as implemented.

September 23: issue #185 reproduced at the index boundary: with `en_stem` before
`lex(segmenter: unicode)`, `棕毛狐狸` returned no hits although the qualified
Chinese query matched. Both unqualified terms and the permissive plain-text
fallback now tokenize each default field independently. They retain OR semantics
across tokens and fields, ignore a field's empty token stream when another field
has tokens, and report the existing error when all fields yield no tokens.
Qualified queries and the single-field term planner decomposition are preserved.
The shared core parser serves native, async-only and WASM callers; persisted
index and wire formats are unchanged.

The cost is one tokenizer invocation per default field and one query clause per
emitted token. Temporary token storage covers one field at a time, with no new
cache or retained corpus data. This is a correctness fix; no latency or memory
improvement is claimed, and no performance benchmark was run.

Regression coverage includes both schema orders, Chinese segmentation, mixed
stemmed/unstemmed fields, stop-word-only input, strict parsing, and plain-text
fallback. The original regression failed before the fix; all 28 parser tests
pass afterward with both default features and native-without-sync. The WASM
release build and all 39 browser tests pass, including the new multilingual
regression. The initial `python3 scripts/check_search.py check` run passed
ownership and formatting checks but stopped at `stop_words::LANGUAGE`
deprecations in `tokenizer/mod.rs`, because it treats warnings as errors.

September 23 follow-up: the shared stop-word language mapping now uses
`stop_words::Language`, imported as `StopWordLanguage` to distinguish it from
Summa's own `Language`. Comparing the old and new APIs for all 18 supported
languages confirms byte-identical stop-word lists. Both APIs are generated from
the same upstream enum definition; tokenizer behavior, serialized formats and
allocation costs are unchanged. Strict search-stack and full-workspace Clippy
now pass without allowing deprecations. The WASM release build and all 39 browser
tests also pass.

September 19: [range bitset word materialization](range-word-materialization.md)
reduces measured warm filter-construction time by 39–66% on four Apple M4
fixtures (65,536 documents, one/sixteen copied blocks, 1%/50% selectivity).
The shared fast-field reader exposes bounded batches and the query packs exact
matches into words. Encoded bytes, scoring and planner/codec defaults do not
change. Output size is unchanged; 64 bytes of comparison scratch are added and
the compiled scan grows by 4,232 bytes. Process RSS measurements include setup
and do not establish memory savings. The report records source-comparison
findings, retained benchmark evidence and remaining cross-architecture/full-query
work; this is not an end-to-end latency claim.
Validation passes the search `check` harness (2,025 tests), both async-only
range regressions and all 38 WASM tests. A background-merge test timed out in
the initial run, then passed in isolation and in the complete two-thread rerun;
the linked report retains the failure and retry evidence.

September 19 follow-up: [range block scans](range-block-scans.md) retain two
format-preserving optimizations. Existing codec headers reject disjoint copied
blocks, and a constant-size decoder cursor removes repeated BlockwiseLinear
header walks. On the same M4, clustered 65K/sixteen-block filtering falls from
63.219 to 4.083 µs; million-document piecewise filtering falls from 5.172 to
2.863 ms. Pruning alone does not help BlockwiseLinear ordered columns. The
shuffled controls do not regress. Encoded sizes and output/metadata allocations
are unchanged; the cursor adds two `usize` fields, with no retained payload.
Process RSS includes setup and does not establish a residency reduction.
The linked report records isolated contributions, confidence intervals, noisy
and unsuccessful controls, and remaining cold/full-query/x86 measurements.
The native `check` harness passes 2,029 tests (25 normally ignored), strict
Clippy, native-without-sync and standalone broker compilation. All three
async-only range regressions and the WASM build plus 38 tests also pass.

Current release review: [module ownership and shared implementations](#release-review-module-ownership-and-shared-implementations).
The newest sparse storage results are in [compact Seismic summaries](seismic-compact-summaries.md);
[binary vector storage](binary-vector-storage.md) describes the single-copy layout.
The text benchmark overview below describes its September 16 measurement snapshot.

The September 20 [IResearch and Linux I/O audit](iresearch-optimization-audit.md)
finds existing batch/lazy evaluation, heap-based text collectors, partial adaptive
codec coverage, and a release-versus-benchmark SIMD verification gap. No io_uring
backend exists; a read-only check also found no rings in the three running index
server processes. The audit distinguishes the benchmark's buffered selection from
current IResearch's loser-tree collector and mmap reads from async writes. No
performance improvement or backend enablement is claimed.

The follow-up [collector experiment](collector-benchmark.md) compares the real
scoring heap with benchmark-only partial selection and a loser tree. Two M4
runs show partial selection reducing synthetic BM25 collection time by 22–28%
at k=100 and 60–62% at k=1000, with no meaningful top-10 win. It doubles retained
entry capacity and visits 30.6% more candidates in the k=1000 block-pruning
control. These are collector-only measurements with precomputed scores; desktop
noise and the absence of x86/end-to-end measurements limit conclusions. The
production heap remains unchanged. Oracle checks and the search harness pass.

September 22 [Searchbench preparation](searchbench-comparison.md) pins Yonik's
10M-document HTTP workload and probes all 826 selected queries. Of those, 585
parse with no known missing operator, 83 fail native syntax, and 158 require
wildcard/regex support. A four-document smoke confirms `th*e` is currently parsed
as prefix OR term, returning four matches instead of the wildcard's two. No
Searchbench timing has run; the serving adapter, count-agreement gate, corpus
builds, and execution-host decision remain outstanding.

The September 20 [topic-aware placement proposal](topic-aware-placement.md) traces
broker ingestion and proposes intra-shard topic cells, similarity-aware merges,
and local N→M redistribution. Primary-key shard routing stays unchanged; each
shard reuses its own ANN model, with optional broker-computed hints. Remaining work
is bounded builder scheduling, placement metadata, and local rewrite/publication
integration. No implementation or performance measurements are claimed.
The proposal's RGB reuse assessment identifies the existing shared partitioner
and field writers; whole-document mapping and byte-aware output planning remain
new work. Dense-only RGB would additionally need a measured graph adapter.

New [query-work diagnosis](search-work-diagnosis.md) separates the gap by family:
standalone top-10 decodes 8.29× Tantivy's document blocks; ranked unions and phrases
already avoid work, while intersection/complete phrase gap payloads remain larger.
The opt-in counters are absent from production builds. See the new report for
scope, per-query evidence and the norm-scoring control experiment.

Text measurement snapshot (September 16): opt-in compact text directories reduce
full-corpus official RSS from **1034.65 to 816.10 MiB**; byte norms reduce it to
**811.29 MiB**. Compact/exact top-10 and top-1000 are nearly flat, but top-100
plus count regresses 1.9%. Quantized norms with lookup scoring regress official
latency 2.3–5.1% and standalone top-1000 14.8%. **Tantivy parity remains unmet.**
Both new options remain disabled by default. The reader-only legacy-index
control also has remaining count overhead; all controls are reported in the
[current comparison](search-benchmark-current.md) and the
[compact-format review](#compact-text-directories-and-byte-norms--september-16).
Earlier sections preserve historical measurements; references there to “latest”
apply only to their frozen source snapshots.

Review base: `dc09bb3594424910f29f3854deaf1ac58c7fc0f1`.
This is a review of core/server entry points, merge representations, metadata
residency, and related broker behavior, followed by a focused alignment pass.
It is not a claim that every algorithm in the search stack has been audited.

## Implemented findings

| Priority    | Finding and trigger                                                                                                                                                       | Change and evidence                                                                                                                                                                                                                      |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1          | `GetTextStats` bypassed search shape validation and admission; deeply nested queries and concurrent statistics calls could reach expensive work outside the search limits | Reuse the query-shape walker and shared search permit before index open. Two RPC-level regressions first returned `NotFound` where `InvalidArgument`/`ResourceExhausted` were required; now they enforce the boundary and permit release |
| P1          | `SUMMA_PIN_METADATA_BUDGET_MB=u64::MAX` overflowed multiplication, panicking in debug or wrapping in release; malformed values silently disabled pinning                  | Checked conversion and actionable warnings; separate-process regression covers unset, zero, valid, malformed, negative, and overflowing values without mutating global test environment                                                  |
| P2          | A merge synthesized absent fast columns with document-sized vectors, ran codec estimation, serialized, and copied the result back out                                     | Emit the existing constant/empty codecs directly; one bounded missing payload per column and no second block-directory/payload-placeholder arrays                                                                                        |
| Maintenance | Search orchestration, shape policy, hydration accounting, and hundreds of tests lived in one 1,862-line file                                                              | Extract `search_service/validation.rs`, `response.rs`, and RPC/budget tests; preserve `SearchLimits` and `QueryShapeLimits` re-exports. The service retains the request orchestration                                                    |

The shared [contract](search-system-contract.md), root `AGENTS.md`, executable
`scripts/check_search.py`, harness self-tests, and CI ownership check now connect
the design requirements to a repeatable workflow. Existing documentation was
corrected where it equated `native` with `sync` or described advisory cold I/O
as a general cache-bypass guarantee.

Statistics validation preserves the broker's flattened text-term container:
the aggregate node/clause budgets still apply, while the per-Boolean scoring
fanout is specific to search. An additional regression first rejected 129
flattened terms, then passed after this distinction; an oversized aggregate
remains rejected. This avoids turning the new admission checks into a broker
compatibility regression.

The missing-column optimization preserves the format: 9 bytes for a nonempty
single-value absent column, 23 for a multi-value absent column. Tests compare
the complete encoded block against the existing builder for numeric/text,
single/multi-value, and several document counts. A complete three-source merge
reopens with missing sources on both sides of real numeric/text values. Another
test emits a billion-document absent block without a document-sized allocation.
Pinning coverage now compares exact doc IDs, score bits, and scored count before
and after moving metadata into heap-backed storage.

## Deployed indexing follow-up: position scratch and commit recovery

A userspace CPU-clock sample of release 1.8.121 on the social shard captured
4,850 indexing-worker samples, of which at least 4,676 (96.4%) were in the loop
clearing every retained position vector before each text field/chunk. The
scratch map accumulated the segment's vocabulary, including terms absent from
the current field; even unpositioned fields paid for that scan.

`SegmentBuilder` now records distinct terms with nonempty position scratch and
clears only those on the next field/chunk. Cleanup is O(previous field's unique
position terms), not O(segment vocabulary). Existing per-term vector capacities
are reused. Additional scratch is one `Spur` per distinct positioned term in
the largest field/chunk (four bytes per entry, plus Vec capacity slack); the
existing segment-wide map still retains its allocations. Regression coverage
checks growing vocabulary, repeated terms, empty/unpositioned fields, both
tokenizer paths, all three position modes, and isolation between chunks/docs.

The focused fixture is `summa-core/examples/indexing_scratch_benchmark.rs`:
1,000 eight-chunk documents after preloading 1K/10K/100K terms, excluding flush,
ANN and merge work. Exploratory before/after runs used Rust 1.98.0 and the same
release flags, but other CPU-heavy work ran on the shared Mac; these are not
controlled throughput estimates or a production speedup claim. The fixture's
large seed field also retains a large term-frequency table, so it includes
more than the position-cleanup cost. Repeat on an idle machine and measure
end-to-end ingest/maintenance separately before asserting sustained capacity.

The same production review found all three document shards paused after a
300-second flush timeout, despite their builds eventually finishing. Retrying
`Commit` recovered the retained generation (1,744,007 committed documents
across the three shards). Core now exposes `CommitFlushTimeout` as a distinct
error. The server owns an accepted commit through flush, publication, reader
reload and timeout retries even if its client disconnects. Shutdown waits for
that writer guard before closing segment-build admission. Build/publication
errors are not retried in a loop. See [the lifecycle contract](segment-lifecycle.md)
for recovery and forced-termination limits. This does not alter sparse reorder
policy, ANN construction, scoring, or persisted formats.

## Deployed BM25 latency review (1.8.122, 2026-09-05)

This follow-up is diagnostic only: no production settings, index data or search
implementation were changed. All four servers and the broker were healthy on
1.8.122. The document index contained 10,339,428 documents and 177,707,964 text
chunks; `machine` and `learning` occurred in 2,247,060 and 3,371,715 chunks.
Indexing/maintenance and other searches remained active. Segment count changed
from 20 to 15 during the observation window (approximately 11:50–12:00 UTC).

Sequential gRPC probes requested top 40, loaded only `id`, used the same query
text for BM25 and server-tokenized sparse search, and had a 30-second RPC limit.
These are small live samples, not controlled benchmarks, recall comparisons or
production percentile estimates. Times below are median `timings.search_us`
over three repetitions: the broker reports the maximum backend phase time,
excluding its statistics round trip and the client's SSH/network overhead.

| Query                | Plain content BM25 | BM25 + separate phrase bonus | Bounded proximity rescoring | Sparse |
| -------------------- | -----------------: | ---------------------------: | --------------------------: | -----: |
| `history of germany` |              85 ms |                       186 ms |                       68 ms |  53 ms |
| `machine learning`   |             140 ms |                       493 ms |                      116 ms |  21 ms |

For `machine learning`, plain BM25 ranged from 134–233 ms, phrase-bonus search
from 416–858 ms, and sparse from 16–118 ms. The phrase-bonus request exposed
217,093 intermediate hits versus 548 for plain BM25; these are executor results
seen by collection, **not** posting-visit counters. ID loading took 5–19 ms in
the plain case and 7–12 ms with the phrase bonus. A nine-term query did not show
a consistent BM25/sparse disadvantage: two plain-BM25 samples were 169/260 ms
versus sparse 173/281 ms. No equivalent dense query embedding was supplied, so
there is no matched dense latency or recall claim from this review.

Observed request shapes included `SHOULD(Match(content), Boost(Phrase(content)))`.
Plain text already selects the windowed block-max MaxScore executor; enabling
WAND/MaxScore from scratch is not the missing optimization. Production logs
also contain slow fusion requests, but currently do not identify each fused
subquery's cost, so their full latency cannot be attributed to BM25 alone.

### Confirmed issues and recommended order

1. **Fix the text pruning-factor contract.** The converter accepts
   `MatchQuery.heap_factor > 1`, and `BooleanQuery::with_text_heap_factor`
   preserves that value. `MaxScoreExecutor::new` then clamps it to `[0.01, 1]`:
   every supported approximate text value becomes exact `1`. On one document
   shard, factors 1 and 100 returned identical ordered IDs, scores and `seen`
   counts in both paired runs; scoring was 104–107 ms in all four requests.
   Normalize the public text convention to the executor's sparse convention
   explicitly and add an RPC-to-executor regression. Keep approximation opt-in
   and measure recall before selecting a default.

2. **Make budgets effective through scorer construction.** With a 1 ms text
   budget on the same shard, `machine learning` plain BM25 reported truncation
   after 10 ms of search, but the phrase-only path spent 385 ms and reported
   `truncated=false`. A single-term `machine` query spent 67 ms and also reported
   no truncation. The phrase-bonus request spent 282 ms before reporting
   truncation: its eager phrase construction had already run. Thread the budget
   through phrase verification and single-term chunked execution; do not present
   the RPC deadline as cancellation of already-started blocking CPU work.
   The existing fusion budget/statistics finding below also applies.

3. **Optimize phrase verification without changing match semantics.**
   `build_chunked_phrase_scorer` drains all matching chunks into a vector and
   folds all matching documents, even when the phrase only supplies an optional
   bonus. `PositionStream::read_into` re-decodes its position block on each
   access; its scratch retains capacity, not a keyed decoded-block cache.
   Phrase matching also restarts linear position scans for each starting
   position. Prototype per-term bounded position cursors/caches and monotone
   positional intersection, then a lazy, top-k-aware composition that preserves
   mandatory phrase constraints. The existing bounded `proximity_weight` stage
   is an optional ranking alternative, **not** an equivalent replacement: on
   `history of germany` its top 40 overlapped the separate-phrase version in
   only 25 results (35 for `machine learning`). Benchmark rank-safe kernel
   changes separately from any candidate-limited ranking change.

4. **Tighten existing chunk bounds before changing codecs or scoring.**
   `LengthLookup::length` writes raw chunk lengths into block minima, while
   scoring uses `ChunkMap::bm25_length`, floored at the nominal chunk length.
   List/block/group bounds use the raw minimum, so they can be conservatively
   loose. Applying the same floor when computing query-time upper bounds is a
   format-preserving, rank-safe candidate, subject to exact-top-k oracle tests.
   Cache bounds per active block/group rather than recomputing the same BM25
   divisions for successive windows; Lucene's
   [MaxScoreCache](https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/MaxScoreCache.html)
   uses this separation. Cross-segment pruning for chunked results additionally
   needs a floor backed by **distinct documents**, not a raw top-chunk heap.
   None of these proposed kernel speedups has been measured in this review.

The sampled document shard had zero CPU quota throttling, no swap or OOM events,
and zero recent memory-pressure averages, but some CPU contention. These checks
do not rule out page-cache misses or quantify per-query peak scratch. No
instruction-level CPU profile or controlled cold-cache/cross-architecture
benchmark was obtained. Preserve scoring, formats and production defaults until
the proposed changes pass the harness and a representative quality/performance
comparison. Local raw probes and the reproducible sequential client are retained
in `.context/bm25-remote-probes-20260905.json` and
`.context/bm25_remote_probe.py`.

Validation for this diagnostic/documentation follow-up:
`python3 scripts/check_search.py check` passed all four steps on Rust 1.98.1,
including 1,284 core tests, 56 server tests, broker/tool tests, Clippy, formatting
and the native-without-sync boundary. Evidence is in
`.context/search-harness/20260905T120227.417351Z-check/`. The additional `full`
harness, WASM and performance benchmarks were not rerun: no search implementation
was changed, and these existing tests do not establish coverage of the newly
identified bugs.

## BM25 execution fixes following the deployed review

Implemented against `5cecf189` with Rust 1.98.1. The preceding section records
the observation phase, before these fixes; publication/deployment are separate
release operations.

- Text and sparse now share the executor's reciprocal convention: factors below
  1 enable extra pruning (`threshold / factor`). RPC zero/unset or 1 selects
  exact search; negative, non-finite and >1 RPC values fail validation, without
  legacy translation. Single-token conversion no longer bypasses explicitly
  requested text pruning, and nested query flattening preserves tuning. The
  shared executor's existing 0.01 factor floor caps the effective multiplier at 100. Exact defaults and sparse tuning are unchanged.
- Nested scorers keep the deadline/truncation state while discarding the outer
  score floor. Term/phrase construction, phrase intersection, phrase bitsets,
  text executor entry and generic collection check it. Cancelled bitsets never
  escape as complete negative filters. An expired request can return no hits;
  cancelled chunked phrase construction does not start its all-hit sort/fold.
  Legacy non-phrase materializers have boundary checks, not preemption inside
  their work; outstanding I/O is still not interrupted.
- Phrase and proximity readers retain one decoded position block per term.
  Phrase slop/offset matching uses monotone position cursors rather than
  restarting each list scan for each starting position. Frequency, repeated
  terms, offset gaps, independent slop intervals and chunk boundaries stay intact.
- Text list/block/group score bounds apply the scoring chunk-length floor.
  Two one-entry bound caches per text cursor remove repeated BM25 divisions;
  no encoder, merge writer, persistent format or score formula changed.

The failing regressions reproduced equal exact/approximate rankings, ignored
expired construction budgets, and raw-length bounds before their respective
fixes. Coverage also includes converter-to-executor behavior for one/multiple
terms on plain/chunked fields, opposite text/sparse factor conventions, exact
top-k oracles, nested floor isolation, cancellation after an initial phrase
match, generous-budget score/ordinal/negative-filter equivalence, position
frequency oracles, and cached reads across copied short blocks and backward
seeks. Reading preserves the original encoded position bytes.

### Same-fixture local measurements

`summa-core/examples/bm25_execution_benchmark.rs`: 20,000 RAM documents,
80,000 positioned chunks, top 40 with ordinals, one indexing/search thread,
current-thread async entry, default release flags on the same Apple Silicon
Mac, Rust 1.98.1 (`48a229cea`). Each run has a warmup plus ten timed searches.
The corpus deliberately repeats terms 40–60 times in most chunks: it stresses
the positional kernel and is not representative of every production query.

| Query               | Before median | After medians, three runs |
| ------------------- | ------------: | ------------------------: |
| Plain BM25          |       1.43 ms |              1.30–1.40 ms |
| Phrase              |      36.48 ms |            11.53–11.66 ms |
| BM25 + phrase bonus |      38.53 ms |            13.57–13.96 ms |

The before/after ordered document IDs, score bits, ordinal score bits and seen
counts are identical in all runs. The final repeats ran after this task's
compilers/tests stopped; the shared Mac was not CPU-isolated. These are local
synthetic improvements (about 3.1x phrase and 2.8x phrase-bonus), not production
latency/recall estimates or evidence to change ranking defaults. Plain BM25 is
effectively unchanged on this tied-score fixture; no standalone speedup is
claimed for tighter bounds here.

Process peak RSS from `/usr/bin/time -l`, including fixture construction, was
109.5 MB before and 108.9–113.4 MB after. This is not isolated query scratch or
evidence of reduced memory. Each position cache holds at most 128 u32 values
(512 bytes) per term plus small metadata; the score-bound caches add 16 bytes
per text cursor. These measurements precede lazy folding (below). Per-document
position buffers remain. Fusion deadline/statistics propagation (see remaining
findings), and production/cold-cache/recall evaluation are still follow-ups,
not completed by this patch.

Raw local evidence: `.context/bm25-before*.log`, `.context/bm25-final-{1,2,3}*.log`.
Full validation passed in `.context/search-harness/20260905T124157.496034Z-full/`,
including both real-server broker tests; the final collector-boundary adjustments
passed `check` in `20260905T124513.498918Z-check/` (1,295 core tests passed,
12 ignored, plus server/broker/tool/integration tests). Final WASM build and all
four JS tests passed after `npm ci`. GPU/full-workspace and production performance
checks were not run.

### Incremental lazy-folding validation

Verified doc-ordered chunk maps now yield one folded document at a time, with
one matching-chunk lookahead. Reordered maps retain stable eager aggregation.
Both paths match the old fold's document IDs, score bits and ordinal encounter
order, including missing chunks and zero scores. A counted-cursor test proves
construction consumes only one document, and a seek skips directly to a late
document; the index-level test verifies matches beyond the requested top-k.
Expired iteration clears the current score/ordinals and marks truncation.

Three paired runs of the same 20K-document fixture above compared a preserved
pre-lazy binary with the final binary, both Rust 1.98.1 release, same Apple M4.
All this task's compilers/tests had stopped, but the shared Mac still had other
application/test activity, so these remain exploratory warm-cache measurements.

| Query               | Before lazy folding, medians | After lazy folding, medians |
| ------------------- | ---------------------------: | --------------------------: |
| Plain BM25          |                 1.29–1.37 ms |                1.33–1.42 ms |
| Phrase              |               11.73–12.15 ms |              11.11–11.12 ms |
| BM25 + phrase bonus |               13.77–13.87 ms |              12.87–12.90 ms |

Ordered IDs, exact score bits, ordinal score bits and seen counts match in all
three pairs. Lazy folding adds roughly 5–9% phrase and 6–7% phrase-bonus savings
on this fixture, not another multi-fold kernel speedup. Plain BM25 is not helped
by phrase folding. Process peak RSS (including ingestion) was 107.2–113.6 MB
before and 112.5–116.0 MB after: no process-memory reduction is established.
The structural query-scratch reduction is from all matching chunks/documents
to one document's ordinals; the opener adds a four-byte-per-chunk sequential
order check and one retained boolean. Reordered segments do not get this folding
benefit. No persisted bytes or merge writer changed.

Final `full` harness passed all eight steps in
`.context/search-harness/20260905T130936.230637Z-full/`: 1,299 core tests passed,
12 ignored, 58 server tests, broker/tool/integration tests, both real-server
broker tests, Clippy, portable/native boundaries and API docs. WASM release plus
four JS tests and regenerated TypeScript plus four client tests passed. The
documentation check passed via `uv run scripts/check_docs.py` (plain Python
lacked its declared Markdown dependency). npm reported two existing WASM dev
dependency audit findings; dependency upgrades are outside this patch.
Raw paired evidence: `.context/bm25-lazy-{before,after}-{1,2,3}*.log`.

## Rust and low-level continuation

The [Rust hot-path review](rust-hot-path-review.md) extends this pass with
optimized assembly, closure/virtual-call probes, decoder benchmarks, measured
native layouts, and ranked follow-up experiments. It implements batch range
materialization and safe byte-range validation that allows byte-aligned decoders
to vectorize. Range comparison semantics are shared by scorer, probes and scans;
no persisted format, ranking arithmetic, or architecture default changes.

On the same M4, 256-value 16/32/64-bit decode batches measured approximately
8.8×/3.8×/4.7× faster, with 696 additional bytes of decoder machine code and no
new heap scratch. Three same-binary range comparisons showed about 6.5–8.1×
lower materialization time; a fourth had an unstable control. The linked review
records the shared-machine noise, lost initial distributions, fresh baselines,
assembly, native/async/WASM validation, and the distinction between measured
fixes and proposed work on dispatch, scratch, metadata layout and header walks.

The harness now accepts `--bench rust_hot_paths` and a Criterion `--filter`.
New distributions default to `.context/search-harness/criterion` so compiler
output cleanup does not remove them. Earlier evidence paths below describe the
original runs, which used Cargo's target directory.

## Reordering continuation

The [reordering performance review](reordering-performance-review.md) traces
record/block BMP and chunked-text BP, removes repeated per-posting gain
arithmetic with an optional budgeted term cache, and records byte/permutation
and assembly evidence. It also records outstanding text memory/convergence
gaps, rank scratch, transpose sorting and scheduling experiments. The kernel
change preserves arithmetic order and existing formats; it does not resolve
the text-path findings or establish production performance. Synthetic BP
speedups are 1.11–2.33× on Apple M4 and 1.20–2.93× on an Intel Xeon 8581C
with default and AVX2 builds. Before/after permutations and sparse/query bytes
match within each tested platform; the original coarse permutation already
differs between macOS/ARM and Linux/x86. Both x86 builds passed 1,299 core
library tests, and peak process RSS stayed around 95–96 MiB on that fixture.

## Benchmark evidence

Fixture: `summa-core/benches/segment_merge.rs`; two RAM segments, each with
4,096 or 65,536 documents and one multi-value numeric fast field. The control
copies both source columns. The missing case simulates one older source without
the optional column. Source building/opening is outside timing. Each iteration
merges all segment components into the same unpublished output, preventing
unbounded retained RAM outputs. No ANN training or reordering is involved.

Host: Apple M4, 10 logical CPUs, aarch64 macOS 15.6.1; Rust 1.98.0 / LLVM 22.1.8;
Cargo release benchmark profile, default `sync` features, no custom RUSTFLAGS.
Criterion: 20 samples, 3-second warmup and approximately 5-second measurement
per case. A second baseline run replaced an initially noisy control measurement;
the 65,536-document missing case was stable at roughly 315–317 microseconds.
This is a shared development machine, so small control differences need caution.

| Case (documents per source) | Before, µs | After, µs | Interpretation                   |
| --------------------------- | ---------: | --------: | -------------------------------- |
| Copy both columns (4,096)   |      41.83 |     41.94 | No statistically detected change |
| One missing column (4,096)  |      47.20 |     31.48 | About 1.5× faster                |
| Copy both columns (65,536)  |      58.02 |     58.08 | No statistically detected change |
| One missing column (65,536) |     314.78 |     51.24 | About 6.1× faster                |

These are Criterion's reported central timing estimates. For the large missing
case, the before interval was 313.08–317.51 µs and after was 49.93–52.21 µs;
Criterion's separate comparison estimator reported a 83.90–84.98% reduction.
Both controls reported no significant change (`p=0.16` and `p=0.42`). The result
supports removing the document-sized synthesis cost; it is not a claim of a
6× speedup for ordinary merges or production search latency.

Commands used:

```sh
python3 scripts/check_search.py bench --save-baseline review-before
# After the implementation and correctness checks:
python3 scripts/check_search.py bench --baseline review-before
```

Raw commands, environment, dirty-source fingerprint, and timing output:
`.context/search-harness/20260904T204806.155345Z-bench/` (before) and
`.context/search-harness/20260904T210348.998886Z-bench/` (after). Criterion's
distributions are under `target/criterion/segment_merge/`. The fixture was added
before the baseline; the core implementation at baseline was still the review
base. Compare using the same added fixture when reproducing against that base.

Memory improvement follows directly from the removed allocation: the old
multi-value path retained at least `4*(N+1)` bytes of offsets and another
`8*(N+1)` bytes while encoding them, plus capacity slack/codec scratch. At
65,536 missing documents, those two arrays alone occupied about 768 KiB. The
replacement encodes 23 bytes using bounded temporary storage. This is a code-
derived allocation bound, not an RSS measurement. Existing source block views,
output bytes, store block metadata, and reader validation still have their own
costs; the entire merge is not constant-space.

## Remaining findings and proposed experiments

These items are not silently treated as compliant. Their changes need the
additional behavior/format or production-workload validation listed here.

1. **P1 — Fusion drops cross-shard statistics and the text deadline.** In
   [search_service.rs](../summa-server/src/search_service.rs), the fusion arm
   calls `search_fused_with_count`; only the ordinary-query arm constructs and
   passes `stats_override` and `deadline`. The broker deliberately extracts text
   leaves from fusion in `partition::text_stats_query` and sends merged stats.
   Unequal shard term distributions can therefore change lexical contributions
   inside fused ranking, and a fusion request's text budget is ineffective.
   Add a budget/statistics-aware fused entry point shared by native/async paths,
   aggregate truncation, and test two shards with unequal term distributions.
   Until supported, explicitly rejecting unsupported options is an alternative
   that requires a deliberate API compatibility decision. This is a code-path
   finding; no distributed fusion fix or performance claim is made here.

2. **P1 — Standalone document hydration remains outside search admission and
   response accounting.** `get_document` loads all stored fields and clones them
   into protobuf values without the permit/budget used by `search`. Transport
   byte limits are checked too late to bound this transient heap amplification.
   Define a document-fetch budget and admission policy, then reuse response
   accounting with endpoint-level tests for many concurrent large documents.
   Measure peak RSS and rejection/retry behavior. Applying the search hydration
   limit unchanged could reject documents that the standalone API currently
   serves, so that behavior needs an explicit decision.

3. **P2 — First text fast-field access builds an allocation-heavy global
   dictionary.** `FastFieldReader::build_text_state` in
   [fast_field/mod.rs](../summa-core/src/structures/fast_field/mod.rs) clones
   source strings into a `BTreeMap`, builds ordinal maps with another lookup
   pass, then serializes the global dictionary. Its “k-way/O(total_entries)”
   comment overstated the implementation and is now corrected. Prototype a heap of borrowed
   dictionary cursors, streaming unique terms and ordinal remaps directly into
   contiguous output. Benchmark first text access after multi-segment merges
   (1/4/16/64 blocks, high and low overlap), retained/peak heap, and warm lookups.
   Require byte-equivalent sorted dictionaries and identical multi-value
   ordinals, including empty/missing blocks. This is a stronger general workload
   candidate than optimizing rare absent columns alone.

4. **P2 — Extend bounded cold range copying to stores/fast fields if storage
   profiling warrants it.** BMP/ANN paths already use kernel-assisted range
   copying. `StoreMerger::append_store` reads each compressed block and writes
   its bytes; fast-field merge writes mapped data/dictionary slices directly.
   Prototype bounded local-file copies with short-copy/error handling and
   cancellation checks. Measure Linux page faults, read/write bytes, merge
   throughput and concurrent query p99 on data larger than RAM. The RAM benchmark
   here cannot justify such a change, and `copy_file_range` benefit depends on
   filesystem support. Dictionary-compressed stores still require recompression;
   removing it needs a versioned per-block dictionary-reference design.

5. **P2 — Audit physical-page ownership and aggregate pin accounting.** Pin
   budgets count logical section bytes and apply per segment/generation. Actual
   `mlock` rounds to pages, and the independent `HeapPinGuard` owners can cover
   allocations sharing a page. Linux memory locks do not stack: an overlapping
   `munlock` can release residency required by another live owner. The OS behavior
   is specified in [mlock(2)](https://man7.org/linux/man-pages/man2/mlock.2.html);
   the frequency of shared-page overlap here remains unmeasured. Add a Linux
   generation-overlap stress fixture, compare reported bytes to `VmLck`, and
   evaluate shared page-range ownership or dedicated page-aligned metadata
   arenas before promising a physical process-wide budget. Include old readers
   held across merges in residency sizing.

## Release gates

Validation completed on this host:

- `python3 scripts/check_search.py full`: all eight steps passed. This includes
  formatting, Clippy for core/server/broker/tool, 1,406 passing Rust tests,
  native-without-sync and minimal-core compile checks, API docs with warnings
  denied, and two additional broker tests against real server subprocesses.
- After the final statistics-container compatibility change: all 54 server
  tests, server Clippy, a fresh server build, and both real-server broker tests
  passed again. The new test adds one distinct regression to the full-run count.
- Five harness self-tests passed, including dependency aliases/target-specific
  boundaries, missing documentation, subprocess failure, and timeout cleanup.
  Python Ruff checks/formatting and `git diff --check` passed.
- WASM/browser, Linux mlock/cold-I/O, GPU, and x86 performance were not run in
  this pass. The new encoding helper is native-gated; public protocol formats
  and generated clients did not change.

Local raw evidence is retained in
`.context/search-harness/20260904T205148.716251Z-full/`, with the final server
checks in `.context/server-final-{clippy,test,build,e2e}.log`. The intentionally
failing regressions are in `.context/server-regression-before.log`,
`.context/pin-before.log`, and `.context/stats-flatten-before.log`.

Keep defaults unchanged until a representative production corpus and concurrent
ingest/merge workload show acceptable p95/p99, memory, recall, and throughput.
Run Linux cold-I/O/mlock measurements and x86 AVX2 comparison separately; this
Mac/RAM run cannot establish those results. The focused harness does not replace
the workspace's GPU, client generation, or WASM integration jobs.

## Candidate scoring and coordinator review (2026-09-05)

The post-implementation review traced the named query contract through native,
async and WASM execution, immutable ordinal lookup publication, branch nomination,
score backfill, document combiners, shard export, broker selection and clients.
Core owns the only linear scorer and fusion algorithm; RPC adapters validate and
translate. Existing extraction and chunk limits are unchanged.

Findings resolved in this change:

- A multi-field branch could cast two RRF votes for one logical passage, changing
  the winner. The failing `one_branch_cannot_vote_twice_for_the_same_passage_across_fields`
  regression now passes; core deduplicates by logical passage before assigning ranks.
- Averaging or summing exported top passages cannot reproduce a document's full
  reduction. Shards now export all scored rows for AVG/SUM and enough top rows
  for MAX/weighted-top-k. Both levels execute the same core formula and verify
  score agreement before the broker selects the global page.
- The broker rejects missing shard responses/statistics, incompatible response
  versions, duplicate document/branch identities, unexpected scopes, nonfinite
  features and incomplete combiner inputs. CPU work retains admission permits
  after client cancellation. Concurrent decode bytes and feature matrices are bounded.
- Plain-text fields missing required length metadata now advertise that they
  are unprepared instead of claiming readiness and failing only during backfill.
- Integration with the independent text-pruning change preserves lazy ordered
  phrase iteration and cancellation, while point backfill and ordinary phrase
  retrieval share positional verification, global BM25 statistics and field parameters.

The full harness passed all eight stages after integration with 1.8.123, including
1,309 core unit tests, 62 server tests, 49 broker unit tests, 13 mock-broker
integration tests and both real-server broker tests. Evidence is in
`.context/search-harness/20260905T133245.093943Z-full/`. Python client round trips
(8 tests) and TypeScript build/wire tests (6 tests) passed before the upstream
merge; generated clients were refreshed against the combined protocol afterward.

These are correctness results, not production recall or latency measurements.
The L1 model remains opt-in. Training must measure held-out teacher candidate
recall and passage survival with a frozen corpus, along with latency, response
bytes and concurrent indexing load. Legacy nested fusion and fusion with the
vector reranker retain their existing shard execution and do not advertise
`global_rrf_v1`. Ordered old segments remain readable without a new sidecar;
legacy reordered fields explicitly require preparation before L1 use.

The Linux CI broker-only build exposed an x86 feature-boundary regression:
`pack_group_bmi2` was compiled when its native/WASM writer caller was absent.
Its cfg now matches the caller, retaining runtime BMI2 dispatch in writer builds.
The failing CI run is `33969669241`; this does not change encoded bytes or scoring.
The first fix accidentally gated the read-side decoder; CI `33971140308` caught
that error. The corrected gate applies only to the writer. The broker's minimal
core now also passes an explicit `x86_64-apple-darwin` compile with Rust 1.98.1
and warnings denied (`.context/phrase-boost/l1-x86-broker-check.log` in the
Azeroth integration workspace).

The continuation caught lossy `GetIndexInfo` schema rendering: BMP sparse fields
were reported without `format: bmp`, dimensions, grid/block settings, mass
cropping or reordering. Re-parsing that report changed the apparent format to
MaxScore even though persisted production metadata was BMP. The behavior-named
round-trip regression fails before the fix and now preserves those settings.
This changes diagnostics and schema fingerprints, not persisted storage or
scoring. Production sparse format was verified against `metadata.json` on all
four shards; full-text continues to use its existing MaxScore execution.

Validation for the schema-reporting fix: all eight full-harness stages passed
with `RUST_TEST_THREADS=1` (evidence: `.context/search-harness/20260905T150826.682593Z-full/`).
Two earlier parallel runs timed out in different mock-broker discovery tests;
the recovery test passed in isolation and the complete serial suite passed.
A separate two-real-shard BMP fixture nominated only through dense search,
backfilled BM25/phrase/sparse/document features including negative and zero
values, and matched an independent full-union oracle for broker MAX/AVG/SUM
top-K with bounded raw passage export. Its script and results are retained in
the Azeroth integration workspace under `.context/l1-training/`.

## L1 completion and owned forward values (Quebec, 2026-09-05)

Reviewed and integrated `origin/handoff/l1-scoring-2026-09-05` at `2d21cb0e`.
Rebased onto `origin/main` at `1844d9af`, preserving its budgeted BP gain cache
and the earlier broker/text merge resolutions. This section supersedes the draft sidecar and complete-rescoring descriptions
above; historical evidence remains attributed to its original workspace.

Resolved findings:

- L1 recomputed organic scores and treated every missing value as an implicit
  zero contribution. It now preserves branch/document and branch/passage scores
  exactly, probes only missing cells, and supports optional `backfill` (default
  true). `missing_values` supplies learned raw defaults before transforms;
  raw exports retain absence. Actual zero/negative scores are never imputed.
  Core and broker share the same formula. The original coefficient contract was `linear_v2` /
  `feature_export_v2`, with candidate-scoring capability 2.
- Sparse MaxScore was unavailable for backfill. Bounded skip-index probes now
  establish presence and score selected documents/ordinals with the existing
  quantized block decoder. Full-text BM25/MaxScore and phrases already use the
  existing posting and position readers. Sparse probes/reads and lazy text
  materialization have request-wide admission budgets; no missing cells means
  no address/payload probes, including on legacy unprepared fields. Async BM25
  statistics now read only dictionary frequencies, fixing posting payload reads
  that bypassed candidate-scoring admission on lazy backends.
- Rejected `.lookup` sidecars introduced duplicate addressing and lifecycle
  state. BMP V20 stores quantized logical forward values inside `.sparse`;
  CHNK V3 adds only physical slots to the existing chunk metadata. Compatible
  merge copies payload and streams metadata remapping. Record and block BP copy
  forward bytes unchanged. Record BP graph construction and selected-record
  rewrite consume those forward values; block BP retains block-level inputs.
  V19 remains readable; all-legacy ordinary merges retain V19, mixed V19/V20
  merges reject, and explicit budgeted reorder migrates old values. No new
  publication, cleanup protocol or preparation RPC is introduced.
- Text merge could silently lose addressing by combining legacy unordered and
  prepared maps. It now rejects incompatible combinations before writing,
  retains the legacy version for pure legacy maps, and validates V3 ordering,
  permutations and section bounds. Explicit reorder upgrades even small legacy
  text segments with no BP plan, preserving original unsaturated token totals.
- Duplicate sparse dimensions were accepted by existing BMP ingestion but a
  new forward validator rejected them; the legacy point scorer also returned
  only one duplicate impact. Forward values preserve every retained impact,
  both point paths sum them, and accumulation overflow fails explicitly. Repeated
  query dimensions also retain all independently quantized contributions.
- The TypeScript wrapper discarded new L1 options despite correct generated
  bindings. A failing wrapper-level regression now verifies actual request
  forwarding of false and learned defaults. Python and TypeScript bindings
  preserve optional-boolean presence and absent raw feature keys. Both wrappers
  reject legacy linear responses rather than silently accepting ignored options.
- Nomination union construction cloned lists before checking retained budgets.
  The shared union builder accepts borrowed lists and checks each hit/ordinal
  before cloning. Original branch lists remain available for organic scores.

Behavior-named regressions reproduce V3 corruption acceptance, legacy duplicate
impact loss, repeated-query contribution loss, TypeScript option loss, legacy
response acceptance, premature lazy text payload reads, and invalid backfill/all-passage
policy acceptance before their fixes; the latter now fails before admission. Additional tests
cover quantized forward/inverted equality, real missing ordinals, byte identity
through copy merge and both BP modes, explicit V19 migration budgets, partial
writer failure, cancellation during payload output, and sparse MaxScore ordinals
spanning multiple blocks. Existing generation failure/cleanup tests exercise
these writers through their original lifecycle owner.

### Same-binary format/path comparison

Host: Apple M4, macOS 15.6.1 (24G90), aarch64, Rust 1.98.1 / LLVM 22.1.8,
default release flags. Fixed RAM fixture: 16,384 single-valued vectors, 4,096
dimensions, 64 retained entries/vector, 128-slot BMP blocks, 4-bit grids.
Legacy V19 is obtained by removing only the V20 forward section from the same
encoded fixture. This compares the retained old and new paths in one binary;
it is not a historical ingestion benchmark. Candidate scores agree exactly;
BP retains identical document, term and posting counts without budget truncation.
Existing BP rewrite tests separately compare all quantized values.

Run the ignored `measure_forward_candidate_scoring_and_bp_graph` core test in
release mode alone, with `--ignored --nocapture --test-threads=1`. It fixes BP
at four workers and a 128 MiB budget. Eleven samples measure 100 scoring calls
per sample (128 candidates, 64 query dimensions) and one BP graph construction.
No other CPU-heavy checks ran during this measurement.

| Operation             | V19 median (range)           | V20 median (range)        |
| --------------------- | ---------------------------- | ------------------------- |
| Candidate scoring     | 110.438 µs (110.162–115.661) | 12.427 µs (12.359–12.706) |
| BP graph construction | 2.759 ms (2.717–2.909)       | 2.180 ms (2.144–2.217)    |
| Encoded BMP bytes     | 6,944,624                    | 12,449,664                |

The forward section adds 5,505,040 bytes here (79.3% over V19). Its exact cost
is 5 bytes/retained posting + 16 bytes/vector + 16 bytes/field. Both forward
payload and directory remain evictable, with no new pinning allocation.
Graph CSR allocations remain 4,194,304 term bytes + 131,080 offset bytes on both
paths; existing physical maps and graph admission limits remain in force.
The entire measurement process peaked at 83,836,928 RSS bytes / 41,779,800 bytes
macOS memory footprint, including fixture construction and both source blobs.
This is not a per-operation or cold-mmap memory measurement. New-ingestion blob
construction took 121.580 ms, without a measured old-ingestion comparison.

Normal forward merge holds O(source count) plans, copies at most 4 MiB per
I/O call and streams fixed-size directory rows; it allocates no vector/posting
permutation. Explicit V19 migration separately admits 12 bytes/real vector of
permutation/offset scratch. New ingestion retains its original posting input
longer, adding only dimension cursors and 8-byte/vector offsets; it does not
materialize a second posting collection. Text merge patches doc IDs in 64 KiB
batches and streams slot remapping; explicit text rewrite admits its columns,
largest 4-byte/chunk permutation and retained BP plans before allocation.

These measurements support the selective scoring improvement on this fixture,
not a production latency, recall or universal BP speedup claim. Production
cold/warm p95/p99 with concurrent ingestion, reordered large corpora, Linux
kernel-copy/mlock behavior and x86 runtime measurements remain unmeasured.
No ranking, BP-granularity or SIMD defaults changed from these measurements.
Raw measurement output is `.context/l1-performance.log` in this workspace.

The follow-up [BMP forward-search research](bmp-forward-search.md) includes primary
literature, reproducible phase-two and whole-block kernel experiments, and explicit
safety conditions for selective completion, filters and threshold seeding. Current
Summa measurements favor very small survivor sets, not replacing whole-block
inverted evaluation. Retrieval defaults remain unchanged.

Follow-up measurements include production per-block term masks and already-parsed
phase-two blocks. They supersede the initial unmasked-header microbenchmark:
one-survivor completion is about 2.6 times faster, while full 32-slot forward
scoring is 11.3–11.6 times slower on that fixture. A separate record-BP mmap
experiment finds a 42–44% one-survivor forward-kernel reduction from a dense query table plus
fused validation, but a 423,516-byte table costs about 3.1 µs to allocate/prepare.
Only ignored experiments use that kernel; no new production cache or search
policy is enabled. See the linked research for setup, memory and locality limits.

### Validation of the rebased integration

All eight full-harness stages passed with `RUST_TEST_THREADS=1` after the rebase:
`.context/search-harness/20260905T174703.766254Z-full/`. This includes 1,324 core
unit tests (17 manual experiments ignored), 64 server tests, 50 broker unit tests,
13 broker integration tests, core integration/doctests, and both real-server
broker tests. The earlier required `check` mode also passed; `full` repeats and
extends all of its stages. The real L1 fixture tests disabled backfill, learned
missing defaults, retained organic values and shared broker/shard inference.

Supplementary checks passed:

- Native without sync: nine candidate-scoring/model tests, including MaxScore
  and legacy addressing; `.context/l1-rebased-native-async.log`.
- WASM release build and four runtime tests in two files;
  `.context/l1-rebased-wasm-tests.log` records the runtime results.
- Python and TypeScript client unit tests: nine each;
  `.context/l1-rebased-python-tests.log`, `.context/l1-rebased-ts-tests.log`.
- x86_64-apple-darwin minimal-core cross-check with warnings denied;
  `.context/l1-x86-minimal.log`. This is not an x86 performance measurement.
- Final format/diagnostic comment cleanup: core all-target Clippy with warnings
  denied, `git diff --check`, and search ownership/document contracts passed.
- All four manual measurement runs (candidate/BP, phase two, whole query, mmap
  query-table/BP locality) passed their integer/shape oracles after the rebase.

No required validation remains unrun and no environment failure blocked these
checks. Production cold-corpus latency, concurrent load, Linux residency/copy
behavior, and x86 runtime performance remain outside the measured evidence.

### Retired selective forward-search experiment

The prototype integrated per-slot forward completion into actual BMP traversal,
including bounded reads, exact score/ordinal comparisons and real RPC tests.
It has now been removed from production search, together with its query option,
adapters, counters and format-only uniqueness certificate. The measurements
below are historical evidence for that decision, not a current search feature.
The prototype sources/patch are archived in `.context/retired-forward-search/`.

The [whole-query experiment](bmp-forward-search.md#whole-query-traversal-and-record-bp-experiment)
now includes real traversal before and after record BP, exact score/ordinal
comparisons, p50/p95/p99 and memory/fault evidence. At 4K dimensions/depth 10/cap 2,
warm median latency improved from 169.46 to 162.92 µs (3.9%), with essentially
unchanged tails. After BP, small survivor sets were rare; forcing whole-block
completion at depth 10 regressed median latency by 1.6–1.7 times. Reclamation hints
also removed the warm benefit. Forward completion is no longer integrated with search;
dense query tables and fused validation remain isolated kernel experiments.

Historical prototype validation (before removing its search integration):

- Required `check` passed at `.context/search-harness/20260905T182911.605688Z-check/`.
- All eight `full` stages passed at
  `.context/search-harness/20260905T183614.969305Z-full/`: 1,332 core unit tests
  (18 manual experiments ignored), 65 server tests, 50 broker unit tests, 13 broker
  integration tests, core integration/doctests and three real-server broker tests.
  The new real RPC test checks the opt-in and rejects an oversized setting.
- Final dispatch guard/lookup-counter cleanup passed seven focused native
  regressions, seven native-without-sync regressions and core all-target Clippy
  with metrics and warnings denied. Evidence is in `.context/l1-traversal-final-*`
  and `.context/l1-traversal-native-async.log`.
- Final WASM release build and five runtime tests passed; Python and TypeScript
  client suites passed ten tests each. Logs use `.context/l1-traversal-*`.
- The whole-query benchmark passed all comparison oracles, including actual
  forward scoring after record BP. Peak process RSS / footprint were
  96,354,304 / 30,851,720 bytes, including setup and both index layouts.

All required checks ran successfully. Controlled cold-disk, concurrent production
load and x86 runtime performance remain unmeasured. The workspace remains rebased
on `origin/main` at `1844d9af`; defaults are not changed from this synthetic ARM
fixture.

### Optional BMP forward storage and final search policy

`bmp_forward_index` is a per-field schema boolean, default true. SDL and persisted
JSON retain explicit false; server schema export preserves it. Disabled ingestion
skips forward construction and releases input postings before grid output.
Ordinary merge and both BP output modes omit the forward section when disabled.
Mixed V19/V20 sources can then copy their compatible inverted blocks into V19;
enabled mixed-version merges still reject an implicit migration. A field excluded
from explicit reorder remains a byte-identical copy, preserving that contract.

BMP retrieval has no forward-index integration or query switch. BP's per-vector
graph and rewrite reads and L1 missing-cell backfill always use stored forward
values when available, without a crossover heuristic. Block BP continues to use
its compact block graph and copies unchanged payloads. Corrupt forward values
fail their L1/record-BP consumers; ordinary search never reads that payload.
Ordered V19 maps still support targeted L1 posting probes. Unordered V19 requires
enabling forward storage and explicit reorder/rebuild, or disabled backfill with
organic scores and learned missing defaults. Full-text/MaxScore are unaffected.

Eight focused regressions pass, including full inverted-byte identity with
storage disabled, mixed-version copy merge, standalone and merge-time BP,
missing/zero/organic L1 scores, schema persistence and the search/L1/BP read
boundary. Logs: `.context/l1-storage-optional-tests.log`. The initial ingestion
test failed before the option was wired (`.context/l1-storage-optional-red.log`).
The space saving is exactly the omitted forward section: 5 bytes per retained
posting, 16 bytes per vector and a 16-byte trailer. No new latency claim is made.

Final validation:

- Required `check` passed:
  `.context/search-harness/20260905T191742.125172Z-check/`.
- All eight `full` stages passed:
  `.context/search-harness/20260905T192107.952734Z-full/`, including 1,333 core unit
  tests (17 manual experiments ignored), 64 server tests, 50 broker unit tests,
  13 broker integration tests, core integration/doctests and two real-server RPC
  tests. This includes optional-storage schema round-tripping in server SDL.
- All eight storage/boundary regressions also passed on native without sync:
  `.context/l1-storage-native-async.log`.
- WASM release build and all five runtime tests passed, including equal search
  results with storage enabled/disabled. Python and TypeScript regenerated
  bindings and their nine client tests each passed. Logs use
  `.context/l1-storage-{wasm,python,ts}-*`.
- Documentation links, search ownership contracts, formatting and whitespace
  checks passed. No required check remains unrun or environmentally blocked.

The earlier production-corpus, controlled cold-storage, concurrent-load and x86
runtime performance limitations still apply; this storage option makes no new
performance claim or change to BMP search behavior.

## Current BMP format and exclusion filters (2026-09-05 follow-up)

BMP now has one accepted/emitted envelope. Optional forward storage is an
explicit section with enabled or disabled state; disabling it no longer writes
an older format. The published enabled representation is unchanged. Disabled
fields add a 16-byte marker. Older readers/writers and version-based merge
selection are removed. The earlier V19/V20 comparisons above are historical
measurements from the migration release, not current compatibility guarantees.
Deploy against rebuilt indexes or after every live BMP blob passes the current
format audit. The operator chose a fresh rebuild instead of waiting for the
one-time forward materialization and BP pass over existing production segments.

The common fusion filter exposed an exclusion-only Boolean bug: no positive
clause produced an empty scorer and an unsupported bitmap, so small segments
returned nothing and large segments could reject the materialization fallback.
The Boolean owner now supplies a neutral document universe, subtracts excluded
matches and preserves entirely empty Boolean semantics. Absent indexed exclusion
terms produce complete empty bitmaps instead of an unsupported result. Native
bitmap materialization is O(segment words + exclusion postings), with the same
fusion bitmap budget; async Boolean scoring streams the complement. Regressions
cover pre-selection exclusion in RRF, L1 and feature export, cross-partition
exclusions, absent terms, tail-bit bounds and exhausted scorers.

Validation:

- Required `check` passed: `.context/search-harness/20260905T204939.904681Z-check/`.
- All eight `full` stages passed:
  `.context/search-harness/20260905T205133.222273Z-full/`, including 1,339 core
  unit tests (17 manual experiments ignored), 65 server tests, 50 broker unit
  tests, 13 broker integration tests, and both real-server RPC tests.
- Native without sync passed the exclusion regression and all 11 selected
  forward-storage tests. WASM release build and all five runtime tests passed.
  Supporting logs are in the parent workspace's
  `.context/bmp-current-{async,wasm-build,wasm-test}.log`.
- The enabled fixture is byte-identical to release 1.8.125: 12,713 bytes,
  FNV-1a-64 `5ccfbcdae3690623`. Disabled storage preserves every inverted byte.
- The first broad run exposed an existing log-capture race with unrelated
  parallel tests. The capture now accepts only its owning test thread; both
  complete runs then passed. Production logging is unchanged.

No new search-latency or throughput claim is made by this cleanup.

## Deletion and maintenance admission (2026-09-06)

Production recreation exposed three lifecycle ordering bugs. Deletion evicted
the registry handle but did not stop the manager before waiting for issued
handles, allowing old Reorder work to retain maintenance capacity. A handler
that already held the index could then wait for its writer behind the delete
lease, forming a second handle-drain cycle. Registry open also swept alleged
orphans before acquiring the OS writer lock, which could delete another
process's unpublished output. Behavior-named regressions reproduced all three.

Deletion now stops manager admission and signals cancellation before the handle
drain; reopening checks the deletion marker before waiting for the lease. Open
uses the existing locked writer opener before loading metadata or cleaning up;
the returned index and writer share one segment manager. This also closes the
stale-snapshot window if another writer finishes during open. Actual blocking work, publication
and deferred deletion still drain before directory removal.

Reorder's shared-writer entry point commits admitted input, releases the writer
lock during maintenance, and uses the existing manager and primary-key refresh
path. Its retained writer Arc preserves the OS lock. Manual BP now uses the
configured background CPU pool. Tests hold all BP capacity while committing
new documents, preserve committed and pending primary keys across replacement,
and cancel queued maintenance without releasing the other index's permit.

The observed fresh-index stall was maintenance waiting, not multi-minute BMP
encoding: new-field BP/rewrite phases logged about 0.2–0.7 seconds while
publication retried 120-second timeouts. After recovery/recreation, all three
documents shards were committing again (94,442 documents at 21:09 UTC). These
are incident observations, not a controlled throughput benchmark.

Focused regression evidence is in the parent workspace's
`.context/lifecycle-{delete,reopen,writer-owner}-red.log`,
`.context/lifecycle-maintenance-tests.log` and
`.context/lifecycle-registry-tests.log`. The lifecycle-only `check` and all eight
`full` stages passed (1,341 core, 68 server, 50 broker unit, 13 broker integration
and two real-server tests), as did native-without-sync maintenance regressions.
One initial broad run hit a mock broker's ephemeral-port collision; rerunning
with `RUST_TEST_THREADS=1` passed without changing production or test behavior.

The API also uses an explicit `AllQuery` inside exclusion filters. The existing
wire variant now maps to a core query sharing the all-document cursor and
bounded bitmap. Regressions cover both common-filter spellings in all fusion
modes, missing metadata, bitmap tail bounds and forward-only cursor seeking.
The wire conversion regression first failed with the unimplemented-query error.
Final combined validation passed:

- `check`: `.context/search-harness/20260905T213650.334212Z-check/`.
- All eight `full` stages:
  `.context/search-harness/20260905T213942.454422Z-full/` (1,344 core tests,
  17 manual experiments ignored, 70 server, 50 broker unit, 13 broker integration
  and two real-server RPC tests).
- Native without sync: both match-all regressions and all three maintenance
  regressions passed. WASM release build and all five runtime tests passed.
  Logs: parent workspace `.context/final-engine2-{async,wasm-build,wasm-test}.log`.
- The lock-before-metadata regression failed before the final opener change;
  both registry ownership regressions then passed. Red evidence:
  `.context/lifecycle-open-snapshot-red.log` in the parent workspace.
  An interim full run was interrupted to make that final change; its process
  group cancellation reported an OS error, so only the complete final run above
  is used as validation evidence.

## Search correctness and execution review (2026-09-06)

Reviewed against `5286d8c5` (1.8.126), tracing the shared query-language parser,
Index/Searcher planning, common filters, Boolean/BMP/MaxScore execution and L1
candidate readers. The fixes stay in those owners; there are no index-format,
wire-format, scoring-formula or approximate-default changes.

### Confirmed and fixed

- `emb:sparse({...})` dropped the field's `query<lsp_gamma: N>` setting.
  Explicit zero therefore became the depth-derived approximate default. The
  core parser now carries `Some(0)` and positive caps through unchanged, while
  an omitted setting remains `None`. This shared parser serves native, async,
  tool and WASM callers. The structured RPC converter already inherited the
  setting. Regressions reproduce the lost zero and a nine-document query
  ignoring a one-superblock cap; parsed results now match an explicitly
  configured core query. The schema reference also now describes the existing
  3000/4000/depth gamma schedule correctly.
- Boolean optimization could flatten a common-filter wrapper, or a nested
  Boolean with required/excluded clauses, into unfiltered sparse terms.
  Scoring decomposition now keeps those constraints opaque. A separate BMP
  planning hook preserves query-global superblock selection for common-filter
  wrappers and Boolean queries with local pure filters. A two-segment test
  checks that a cap of one still admits only one superblock across the index.
- Common eligibility reached some nested BM25/BMP and sparse MaxScore plans
  only after their top-k heaps, allowing disallowed high scores to crowd out
  eligible hits. Local Boolean eligibility now intersects outer eligibility;
  sparse MaxScore receives both eligibility and the request deadline. Known-hit
  regressions cover direct, optimized Boolean and nested-filter plans.
- Common-filter materialization used fresh default options, losing deadlines,
  and continued into scoring even after the intersection became empty. It now
  shares the existing truncation state, discards incomplete bitmaps and stops
  before remaining filters or scoring payloads. The direct synchronous scorer
  also delegates to the implemented filtered path. Work remains bounded by the
  existing bitmap/fallback limits; emptiness checks scan existing bitmap words.
- BMP L1 backfill bounded candidate count but did not admit variable-size
  selected payloads against a byte budget. Forward offsets or selected inverted
  block ranges now reserve bytes before payload validation/scoring, sharing the
  existing 256 MiB lazy-text allowance across segments, features and components.
  The reader performs an O(selected values) metadata pass with constant scratch.
  A zero-budget regression failed before the fix for both forward-storage modes.
  Dead version-dependent logging in the current-format-only BMP reader was
  removed while checking this path.

The eligibility and admission invariants are recorded in
[candidate rescoring](candidate-rescoring.md#eligibility-and-bounded-nomination).
No writer or codec changed, so this pass does not claim a format rewrite or a
new byte-identity experiment.

### Local measurement

The empty-filter experiment uses the same RAM index of 16,384 documents, each
containing `alpha beta gamma` and a numeric eligibility value of zero. It asks
for ten `alpha` hits with eligibility equal to one and verifies an empty result
outside timing. Index construction is excluded. Each process measures eleven
batches of 100 end-to-end searches; three before/after pairs alternate order.
Both binaries use the same Apple M4, Rust 1.98.1 / LLVM 22.1.8 and Cargo release
flags, with no concurrent build during measurement.

| Measurement                                       |                   Before |                    After |
| ------------------------------------------------- | -----------------------: | -----------------------: |
| Median batch latency, three processes (µs/search) | 90.091 / 87.678 / 89.499 | 17.672 / 14.169 / 13.285 |
| Median of those medians (µs/search)               |                   89.499 |                   14.169 |
| Process peak RSS range (MiB)                      |              19.34–20.33 |              18.77–19.69 |
| Eligibility bitmap (bytes)                        |                     2048 |                     2048 |

This is a roughly 6.3× improvement for empty eligibility, not an overall search
speedup or a tail-latency claim. RSS includes the fixture and test process and
does not isolate per-query scratch. The first baseline process incurred 544
page faults; the other five reported zero. Queries use a warmed RAM fixture,
and ordinary desktop activity was present. Cold storage, loaded concurrency,
nonempty-filter latency and a controlled x86 before/after remain unmeasured.
Reproduction: run the ignored release test
`query::filtered::tests::empty_common_filter_benchmark` under `/usr/bin/time -l`.
Raw runs and environment metadata are in this workspace's
`.context/common-filter-benchmark-{before,after}-{1,2,3}.log` and
`.context/common-filter-benchmark-environment.json`.

### Deployed observations and remaining performance work

Read-only SSH/Kubernetes sampling observed the existing 1.8.126 broker and four
servers. No patch was deployed, requests replayed or settings changed. Between
approximately 04:52:42 and 04:59:36 UTC, broker counter deltas recorded 37
successful document searches averaging 448.1 ms and 12 social searches averaging
5.13 ms, with no new recorded errors. These are sparse, mixed live requests
(about 0.12 requests/second combined), not a throughput benchmark. The first
broker summary exported document p50/p95/p99 of 47.1/1303.0/1303.0 ms; these
rolling summary quantiles are not quantiles of the counter-delta interval.

On document shard `s2`, the same interval contained seven `sparse_vectors` LSP
plans averaging 90.0 ms and 56 segment BMP executions averaging 218.7 ms.
Mean execution components included 164.3 ms in block scoring, 49.3 ms in
prefetch, 3.64 ms in the D grid and 1.06 ms in document mapping. Four short-document
sparse plans averaged 25.3 ms; their 28 segment executions averaged 289.7 ms,
including 268.4 ms in block scoring. These component means identify profiling
targets; segment work can run concurrently and must not be summed as request
wall time. Sampled LSP counter deltas averaged gamma 3000, but metrics do not
identify the request syntax or prove those requests used a schema gamma of zero.

Two process/cgroup snapshots on `s2`, 337 seconds apart, showed RSS about
63.6 GiB (46.9–47.1 GiB anonymous and 16.5–16.7 GiB file-backed), 1.75 GiB locked
and no swap. Cgroup usage rose from 68.6 to 73.3 GiB; its historical peak was
206.0 GiB, while process peak RSS was 168.7 GiB. The interval added 7.20 million
major faults and 7.99 million file refaults, with CPU consumption averaging
5.53 cores and no quota throttling. Recent memory-pressure averages were low
and OOM counters were zero. These are whole-container observations, not
per-query allocations; neither faults nor the historical peaks can be assigned
to search versus background work from these samples alone.

Remaining work is to profile block scoring and selected-range I/O on a fixed
production-like corpus, correlating faults with per-request and maintenance
activity, then measure warm/cold latency and recall at the same explicit gamma.
The slow log includes candidate exports taking 3.1–6.5 seconds, some with empty
segment heaps and zero thresholds; selective-filter/candidate-depth behavior
deserves that controlled follow-up. The documented bounded ANN nomination can
still underfill a selective common filter. This pass preserves that policy;
changing it needs a recall/work comparison. No production default was tuned.

`kubectl top` failed because the cluster Metrics API is unavailable. Direct
Prometheus scrapes plus `/proc` and cgroup files supplied the observations
instead. Raw evidence is retained under `.context/review-prod-*`, with derived
tables in `.context/review-production-{analysis,resources-analysis,bmp-breakdown}.json`.

### Validation

- Required `python3 scripts/check_search.py check` passed: formatting, Clippy,
  1,356 core unit tests (18 manual experiments ignored), 70 server tests,
  50 broker unit tests, 13 broker integration tests, tool/integration/doc tests
  and the native-without-sync compile boundary. Run evidence:
  `.context/search-harness/20260906T045858.391501Z-check/`.
- Native without sync passed eight filter regressions, all 22 query-language
  tests and the BMP admission regression: `.context/review-native-async.log`.
- WASM release compilation and all seven runtime tests passed, including
  query-language searches with exhaustive and bounded schema gamma. Evidence:
  `.context/review-wasm-{npm-install,build,test}.log`.
- The extended `full` harness/real-server RPC tests, a Linux mlock experiment,
  controlled cold-cache and cross-architecture comparisons were not run in
  this pass. Lifecycle/RPC implementations and residency policy were unchanged.

## Search attribution, traces and symbolic L1 (Dushanbe, 2026-09-06)

This follow-up replaces the coefficient-only L1 API with required `l1.formula`
(capability 3, `formula_v1`). Removed coefficient fields have reserved protobuf
names/tags. `backfill` and raw missing defaults remain, with zero as the default
for a missing formula variable. Core owns the bounded compiled expression and
passage/document inference; server and broker translate and validate. Expressions
compile once per request, bind at most 17 indexed inputs, and have no global cache.
Python and TypeScript bindings and examples use the formula-only contract.

Resolved correctness findings and added observability:

- Native nomination without supplied text statistics bypassed the ordinary
  searcher's query-global BM25 statistics. A two-segment regression first
  reproduced different scores/ranking; nomination now uses the same statistics
  owner. Optional diagnostics preserve ordinary hit identities and score bits.
- `include_rrf_scores` returns a separate score and per-branch votes based on
  complete organic nominations, excluding backfill/score-only features. Broker
  attribution uses global ranks, including nominations absent from final hits.
- `tracing=false` preserves the default path. Opt-in traces retain bounded
  per-shard/per-branch nominations, raw scores and ordinals, query trees/common
  filters, nomination depth/counters and shard selection, across pagination.
  Candidate/ordinal and retained/encoded byte budgets apply before cloning.
- RRF in L1 must be evaluated inside the passage formula, before its document
  combiner. Global RRF can change both the best passage and winning document,
  so the broker obtains the whole bounded union and every scored passage.
  Shards use a constant formula for this export; the broker applies the actual
  formula with global votes. Regressions cover below-local-top-k winners,
  discarded passages, negative/zero multipliers, logarithms/division of RRF,
  incomplete exports, old backends and invalid formulas before admission.

### Fixed-fixture performance evidence

Apple M4 arm64, Rust 1.98.1, identical debug compiler flags and warm local gRPC.
The saved pre-formula server and new server used the same persisted 1,000-document
seed-7 fixture, two BM25 branches, nomination depth 40 and top 20, with raw feature
exports. Each mode started a fresh process, warmed up 20 times and measured 100
sequential calls; no concurrent builds ran. RSS is the maximum sampled whole
process RSS, not isolated scratch or an OS peak. These desktop debug samples are
smoke measurements, not production throughput or evidence to change defaults.

| L1 request                                            | p50 ms | p95 ms | Response bytes | Sampled RSS KiB |
| ----------------------------------------------------- | -----: | -----: | -------------: | --------------: |
| Previous coefficients, 0.2 title + 0.8 body           |  4.050 |  4.319 |          1,826 |          36,672 |
| Equivalent compiled formula                           |  4.020 |  4.136 |          1,825 |          36,784 |
| 0.2 title + 0.8 log1p(body) + 3 RRF, with attribution |  4.521 |  4.689 |          5,934 |          38,976 |

The equivalent formula returned identical document IDs and f32 score bits.
All three modes matched independently computed f64 arithmetic with checked f32
rounding. The tiny timing difference between coefficients and the equivalent
formula is inconclusive; the nonlinear/RRF mode performs additional ranking
and attribution work. Raw evidence and fixture script are in
`.context/formula-performance-samples.jsonl` and `.context/measure_l1_formulas.py`.

A separate run on the same fixture/configuration compared legacy fusion with
optional diagnostics using the new binary:

| Diagnostics     | p50 ms | p95 ms | Response bytes | Sampled RSS KiB |
| --------------- | -----: | -----: | -------------: | --------------: |
| Off             |  3.419 |  3.579 |          1,371 |          35,648 |
| RRF attribution |  3.710 |  4.773 |          5,517 |          36,880 |
| Trace           |  3.627 |  3.963 |          6,184 |          36,528 |
| Both            |  3.876 |  4.328 |          6,656 |          36,512 |

All four modes returned identical IDs and score bits. Traces retained all 80
organic branch nominations; RRF diagnostics matched the fusion scores exactly.
Evidence: `.context/formula-diagnostics-samples.jsonl` and
`.context/measure_search_diagnostics.py`. Earlier production observations above
remain applicable; this follow-up did not tune ANN/LSP defaults or deploy to the
sampled installation. Cold-cache, sustained-load, large-expression/many-passage
and x86 comparisons remain future performance work.

### Validation evidence

The final required `check` passed all four steps:
`.context/search-harness/20260906T073146.024481Z-check/`.
The full search harness passed all eight steps, including real-server broker RPCs:
`.context/search-harness/20260906T072445.352810Z-full/`. Eleven candidate-scoring
regressions also passed with native async execution (no sync feature), and the
WASM release build plus all eight runtime tests passed. Both regenerated client
suites passed eleven tests each. Evidence includes `.context/formula-native-async.log`,
`.context/formula-wasm-{build,tests}.log`, and
`.context/formula-{python,typescript}-tests.log`.

## Filtered body queries at small limits — 2026-09-06

Confirmed against `fa92ac62` (1.8.128). With ten higher-scoring `article`
documents and one lower-scoring `book`, a required chunked text term combined
with `kind:book` returned no hits at limit 1 and the correct book at limit 20.
The generic Boolean planner constructed the body's bounded scorer before
applying its document predicates. Required nested disjunctions and negative
fast-field filters had the same failure. The public query-language reproduction
is `body:machine AND kind:book` with a chunked `body` and raw fast `kind`.

The generic plans now push available predicates into shared eligibility before
constructing their text children. They reuse selective bitset construction
where supported and scan eligible fast-field values otherwise, retaining the
original scoring/verifier clauses and the existing 16 MiB bitmap ceiling.
The regression checks small/large-limit IDs, exact score bits and ordinals;
another covers required disjunctions on plain text. Fast text equality is used
only where it matches indexed-term semantics: single-valued raw text. Analyzed
and multi-valued indexed fields retain their posting-list semantics.

Actual WASM execution of this query also reproduced a MaxScore panic from an
unconditional `std::time::Instant::now()` used for diagnostic logging. Both
MaxScore loops now use the existing portable `observe::WallTimer`.
Red evidence: `.context/filtered-body-red.log`,
`.context/filtered-body-fast-text-red.log`, and
`.context/filtered-body-wasm-red.log`.

### Controlled fixture and limits of the measurements

Apple M4 arm64, Rust 1.98.1, identical debug server build commands and the same
persisted 10,000-document fixture (50 eligible books, one body chunk per
document), top 1. Each case opened the fixture in a fresh process, warmed 20
queries and measured 100 sequential RPCs. Index construction and the full-limit
correctness reference were outside timing; builds did not overlap sampling.

| Query shape                       | Before p50/p95 ms | After p50/p95 ms | Before/after sampled RSS KiB | Before/after correct top 1 |
| --------------------------------- | ----------------: | ---------------: | ---------------------------: | -------------------------- |
| Required body term + kind         |     1.090 / 1.232 |    1.475 / 2.005 |              34,640 / 36,352 | No / Yes                   |
| Required nested body OR + kind    |     1.679 / 2.816 |    1.422 / 1.737 |              35,328 / 35,456 | No / Yes                   |
| Required body term, excluded kind |     1.536 / 2.926 |    1.580 / 1.746 |              37,392 / 34,432 | No / Yes                   |
| Existing filtered SHOULD control  |     1.742 / 1.996 |    1.367 / 1.539 |              34,688 / 35,728 | Yes / Yes                  |

All fixed responses matched the full-limit reference IDs and score bits.
The control's movement makes the timing comparison inconclusive; this is a
correctness fix with bounded extra eligibility work, not a throughput claim.
RSS is sampled process residency, not a measurement of peak scratch. The
bitmap for this fixture occupies 1,256 bytes. Evidence and reproduction:
`.context/filtered-body-{before,after}.jsonl` and
`.context/measure_filtered_body.py`.

This fixes predicate pushdown, not every source of candidate loss. Filters
without a document predicate retain the existing verifier paths. The documented
two-times chunk nomination cap can still retain several chunks of one document
and underfill a document result window; this pass does not change that policy,
ANN/LSP defaults, stored formats, or production configuration. Cold-cache,
large-corpus, concurrency and x86 performance comparisons remain unmeasured.

### Validation and remaining async discrepancy

The final required harness passed all four steps:
`.context/search-harness/20260906T133139.703927Z-check/`. The WASM release build
and all nine runtime tests passed, as did documentation checks and all three
new native-without-sync regression tests. Logs:
`.context/filtered-body-{final-check,wasm-build,wasm-tests,docs}.log` and
`.context/filtered-body-native-async-regressions.log`. No lifecycle, RPC, or
wire definitions changed; the additional `full` lifecycle/RPC harness was not
rerun for this patch. The measured probes did exercise a real local server.

A broader native-without-sync chunked suite passed 16 of 17 tests and exposed
an existing discrepancy in `filters_and_phrases_push_into_chunked_text_maxscore`:
the async fallback includes the required phrase's ordinal 0, returning `[0, 1]`
where the sync bitset-filter path returns `[1]`. The saved 1.8.127 test binary
(`summa_core-14836e5caef7dcbc`, from the previous review) fails the same test
identically, confirming this is not introduced by the patch. Async phrase-filter
materialization/scoring parity remains a separate correctness follow-up.
Evidence: `.context/filtered-body-native-async.log` and
`.context/filtered-body-preexisting-async-phrase.log`.

## Missing one-word phrase filters — 2026-09-06

Confirmed against `56bcae64` (1.8.129): an indexed `PhraseQuery` with one
absent term returned `None` from native bitmap materialization. `None` means
unsupported, so common filtering attempted its scorer fallback and rejected
segments above 200,000 documents. A 200,001-document regression reproduced
the reported "common filter cannot be materialized" error. Structured phrase
requests preserve `PhraseQuery` even with one token; the query-language parser
normally lowers a single-token quote on one field to `TermQuery`, which does
not exercise this phrase path.

The phrase owner now distinguishes a successful posting lookup with no list
from an unavailable materialization. Missing indexed terms produce the empty
bitmap already allocated by this path. Read errors still return an unavailable
bitmap and fall back to explicit errors; they do not become empty matches.
Non-indexed fields retain the scorer fallback because fast-column values can
match even without postings. No parser, wire or persisted representation changed.

The regression covers direct async and sync scorers plus public index search,
plain/chunked text, unpopulated indexed fields, and positive, OR and negated
filters. A smaller fixture also checks present/absent terms against indexed and
fast-only fields across the native-without-sync boundary. Red and green evidence:
`.context/missing-phrase-{red,green}.log`.

The cost remains one document bitmap (25,008 payload bytes for the large
regression), plus the existing term lookup and matching-posting traversal.
The 16 MiB bitmap and 200,000-document scorer-fallback bounds are unchanged.
No latency or RSS benchmark was run for this correctness patch, and no
performance improvement is claimed. Portable builds still have the documented
fallback bound; the async phrase-ordinal discrepancy recorded above remains
outside this fix.

Validation passed: the four-step required harness at
`.context/search-harness/20260906T154504.235296Z-check/`, all ten active common-filter
tests with native async execution (one manual benchmark ignored), and the WASM
release build plus all nine runtime tests. Documentation checks also passed.
Logs: `.context/missing-phrase-{check,native-async,wasm-build,wasm-tests,docs}.log`.
The additional `full` lifecycle/RPC harness was not rerun because this patch
changes neither lifecycle nor RPC code.

## Long L1 phrase features — 2026-09-06

Confirmed against `cf170fbb` (1.8.130): a 64-token phrase succeeds in both
ordinary search and candidate scoring, while 65 tokens fail candidate-plan
validation with "invalid L1 phrase feature or missing positions". The server
converter admits up to 256 tokens by default. L1 reused the 64-term nomination
limit even though the shared phrase scorer uses dynamic positional cursors.
The same plan validation runs for learned ranking and raw feature collection.

Phrase feature validation now has its own 256-token bound, matching the default
server conversion budget. It preserves every term and existing offsets/slop.
An oversized core request fails with the feature name, actual term count and
maximum before statistics or candidate resolution, including an empty candidate
set. Field/position checks, nonphrase feature limits, nomination limits and
wire/storage formats are unchanged.

The core regression uses synthetic `term0` through `term255` on plain and
chunked fields. At 64, 65 and 256 tokens, raw exports and formula scores match
ordinary phrase search bit-for-bit. Documents with a wrong 65th term or missing
256th term score zero, so accepting the request cannot hide truncation. A
257-token candidate request still fails. The broker reproducer uses the same
four synthetic documents over two real server processes and checks formula
ranking and collection against ordinary distributed phrase scores.

Reproduction commands using synthetic data:

```sh
cargo test --locked -p summa-core --lib long_phrase_features_keep_every_term_in_ranking_and_collection
cargo build --locked -p summa-server --bin summa-server
cargo test --locked -p summa-broker --test e2e_real_server broker_ranks_and_exports_long_phrase_features_without_dropping_terms -- --ignored
```

The algorithm is unchanged: candidate phrase scoring keeps one posting cursor
and reusable position buffer per term, seeks only nominated physical targets,
and retains existing read and scored-value admission budgets. Increasing the
accepted phrase length increases that bounded work; this is a correctness fix,
not a throughput claim. No before/after latency or memory benchmark was run
for this patch. Red/green core evidence is in
`.context/long-phrase-{red,green}.log`. The previously recorded native-async
required-phrase ordinal discrepancy remains outside this change.

The required `check` passed all four steps with normal test concurrency:
`.context/search-harness/20260906T180558.238875Z-check/`. All eight full-harness
steps passed with `RUST_TEST_THREADS=1 python3 scripts/check_search.py full`:
`.context/search-harness/20260906T181122.939063Z-full/`, including all three
real-server broker tests. Eight candidate-scoring tests also passed with native
async execution. Logs: `.context/long-phrase-{final-check,full-serial,native-async,broker}.log`.

Parallel validation intermittently timed out in two existing broker integration
tests while waiting ten seconds for initial index discovery:
`client_deadline_propagates_and_absence_means_untimed` in the first `check`, and
`partitioned_stream_routes_each_flush_by_primary_key` in the parallel `full`.
Neither exercises candidate phrase scoring. The focused deadline retry and a
fresh default-concurrency `check` passed; the final `full` used one test thread.
No timeout or production setting was changed. The cause of these timeouts remains
unresolved; original failures are retained in `.context/long-phrase-{check,full}.log`
and the focused retry in `.context/long-phrase-broker-deadline-retry.log`.

The WASM release build and all nine existing runtime tests passed, along with
documentation checks and pre-commit hooks. Evidence:
`.context/long-phrase-{wasm-build,wasm-tests,docs,final-precommit}.log`.

## Index-configurable L1 phrase limits — 2026-09-06

Following the fixed 256-token limit in 1.8.131 (`2bf99929`), the creation schema
now accepts `max_l1_phrase_terms`. The requested default is restored to **64**;
indexes opt into longer features explicitly, for example
`index documents { max_l1_phrase_terms: 256 field body: text [indexed<token_position>] }`.
The optional positive-u32 setting is owned by core `Schema` and persisted under
`schema.max_l1_phrase_terms` in `metadata.json`. SDL, JSON creation, Rust builders,
server/broker creation and WASM creation share it. Index info returns the
effective value in SDL. No protobuf or segment format changes are involved.
Absent settings retain the prior metadata bytes and load as 64; explicit zero,
negative, fractional and out-of-range values fail creation/deserialization.

Before the change, the new default-limit regression incorrectly accepted a
65-term phrase with no nominated candidates (`.context/phrase-cap-red.log`).
It now checks the configured boundary in ranking and collection after reopen,
including custom limits 1, 65 and 300. The existing long-phrase fixture explicitly
sets 300 and compares complete plain/chunked phrase features and formula
predictions against ordinary search at 64, 65, 256 and 300 terms. Wrong 65th terms
and missing 256th terms still score zero. A two-shard RPC fixture checks both
default-64 and configured-256 indexes, reported schemas, and invalid creation.
WASM runtime coverage checks default/configured metadata across reopen and a
second commit, plus rejected zero limits.

The setting uses one inline `Option<NonZeroU32>` with no heap allocation;
validation reads it once per phrase component. Phrase probing is unchanged:
per-phrase cursor scratch and posting/position probes grow linearly with retained
terms. The shared 256 MiB candidate payload-read budget, scored-value budget,
nomination limits, and separate server token budget (default 256) still apply.
No latency/RSS benchmark was run for this configuration change and no performance
improvement is claimed. The previously recorded native-async phrase-ordinal
discrepancy and intermittent parallel broker startup timeouts remain separate
findings.

The initial required check exhausted local disk space while linking tests
(`.context/phrase-cap-check.log`). Removing only this workspace's rebuildable
incremental caches predating the task freed 45 GiB. The next normal-concurrency
check reached broker tests but timed out in the existing
`partitioned_create_and_commit_fan_out_to_every_partition` test, waiting ten
seconds for initial index discovery (`.context/phrase-cap-final-check.log`).
This is the same unresolved discovery failure recorded above; no production
behavior or test timeout was changed. Subsequent harness runs use one test thread.
The first serial full run passed all 1,369 core unit tests, then could not
execute an integration binary: the workspace's entire `target` directory
disappeared during the run, beyond the earlier bounded incremental-cache cleanup.
Its log is `.context/phrase-cap-full-serial.log`. Validation was restarted with
`CARGO_TARGET_DIR=$PWD/.context/l1-phrase-cap-build` to isolate build artifacts.

The isolated full run passed its first seven steps, including all search-stack
tests, native-without-sync and portable compilation, API docs, and the server
build (`.context/search-harness/20260906T184531.789002Z-full/`). Its standalone
broker step caught a native-gated helper used only by the new index-info test
assertion; replacing that helper with the portable SDL parser fixed the feature
boundary. Rerunning that exact final step passed all three real-server tests,
including both phrase-cap configurations (`.context/phrase-cap-broker.log`).
The WASM release build and all 12 runtime tests passed in the separate
`.context/l1-phrase-cap-wasm-build` directory; logs are
`.context/phrase-cap-wasm-{build,install,tests}.log`.
All nine candidate-scoring tests also passed with native async execution
(`cargo test --locked -p summa-core --no-default-features --features native --lib
query::candidate_scoring::tests`), including the 300-term exact-score checks:
`.context/phrase-cap-native-async.log`.

## Row deletion and compaction — 2026-09-07

Implementation base: `7a4b380c1dedfebac5be7a02b7924b93eadb9051`
(`origin/main`, 1.8.132). The [row-deletion contract](row-deletion.md) documents
format 7, immutable visibility, atomic full-document upserts, exact live-key
checks behind the existing Bloom filter, and single-segment compaction. The
implementation operates on encoded fields, so indexed-only values survive.
ANN codebooks, assignments, fingerprints, and surviving codes are retained.

### Correctness and lifecycle evidence

- `python3 scripts/check_search.py check` passed all four steps; evidence:
  `.context/search-harness/20260907T074632.727500Z-check/`.
- `python3 scripts/check_search.py full` passed all eight steps, including
  1,381 core tests, 77 server tests, broker/tool tests, portable/native-without-sync
  compilation, docs, and all three real-server broker E2E tests; evidence:
  `.context/search-harness/20260907T075229.998139Z-full/`.
- Follow-up regressions and final Clippy cover indexed-only numeric fields,
  binary vectors, JSON/byte storage, missing/multi-value fast fields, all-deleted
  dense/sparse segments, stacked PK dictionaries, stale visibility rejection,
  and global-before-local maintenance admission. The physical-copy merger
  rejects masked readers instead of silently discarding their visibility.
  Logs: `.context/deletion-final-{rows,pk,capacity,clippy}.log`.
  The five end-to-end deletion tests also pass with
  `--no-default-features --features native` (async search without `sync`);
  evidence: `.context/deletion-final-native-async.log`.
- Native tests preserve old searchers through commit and compaction, reopen
  cached Bloom filters, allow deleted keys to be reused, reject live/pending
  duplicates, and cover abort, publication failure, cancelled commits, and
  cancelled compactions with blocking writers. Shutdown drains the latter.
  PK reopen coverage includes Unicode keys and distinct case variants.
- ANN tests compare complete compacted bytes with canonical encoders for TQ,
  IVF-TQ, binary IVF, ScaNN AH, and ScaNN binary. Sparse block tests compare raw
  weights for Float32, Float16, UInt8, and UInt4. End-to-end BMP/MaxScore tests
  compare surviving scores and chunk/value ordinals.
- `summa-wasm/build.sh`, `npm ci`, and `npm test -- --run` passed; the final
  rebuild and 13 tests also passed after portable warning cleanup. A persisted
  indexed-only text index reopens with tombstones, hides deleted hits/hydration,
  preserves an old reader, and rejects a corrupt mask. Logs are under
  `.context/deletion-wasm-*.log`.
- A local CLI smoke test ran create/upsert/update/delete/compact/merge/reopen;
  only the replacement remained, with one physical row and no mask. Evidence:
  `.context/deletion-cli-smoke.log` and `.context/deletion-cli-smoke-verify.log`.
  The initial inspection script used the wrong metadata filename; the final
  verifier reads `metadata.json` and passes.

### Matched microbenchmarks

Before and after used the same unchanged `segment_merge` and `search_pipeline`
fixtures, Rust 1.98.1, release/bench flags, and Apple M4 / 32 GiB / macOS 15.6.1.
`RUSTFLAGS` was unset. Criterion used 20 samples, 0.5 s warmup and 1 s measurement.
The binaries were built separately and copied before execution; baseline source
was a detached checkout of the SHA above. No agent-owned compiler/test ran
during measurements, but other work on the shared machine was not controlled.

The first pass ran before then after. A second pass reversed order and repeated
all merge cases plus the three search cases below. Numbers are Criterion point
estimates in microseconds; large variation prevents production latency claims.
These fixtures measure clean segments with no deletion masks. They do not
measure large-corpus deletion scans, dirty compaction, cold disk I/O, or recall.

| Fixture                              | Before, pass 1 | After, pass 1 | Before, reverse pass | After, reverse pass |
| ------------------------------------ | -------------: | ------------: | -------------------: | ------------------: |
| Copy fast columns, 2 × 4,096 rows    |          26.46 |         37.15 |                30.39 |               40.46 |
| Missing fast column, 2 × 4,096 rows  |          27.03 |         28.16 |                27.40 |               28.93 |
| Copy fast columns, 2 × 65,536 rows   |          74.28 |         68.96 |                77.49 |               51.20 |
| Missing fast column, 2 × 65,536 rows |          52.62 |         52.32 |                58.85 |               55.65 |
| Dense top 10, multi-thread search    |          51.88 |         36.56 |                54.64 |               36.08 |
| Hybrid top 200, multi-thread search  |         561.52 |        497.69 |               498.47 |              340.07 |
| Dense top 200, current-thread search |         191.41 |        203.17 |               175.92 |              156.55 |

`/usr/bin/time -l` measured whole-process peak RSS, including fixture building
and Criterion, rather than allocator-only scratch. Merge RSS was 55.8 → 64.7 MiB
in pass 1 and 60.0 → 61.0 MiB in the reverse pass. Full-search RSS was
84.7 → 91.0 MiB; the matched three-case repeat was 74.8 → 76.2 MiB. These
measurements cannot isolate mask residency because the fixtures have no masks.
The exact mask file cost is `24 + 8 * ceil(rows / 64)` bytes; reader masks use
one bit per physical row, with separate PK live-ordinal masks and compressed
row-statistic columns accounted as described in the design.

Raw logs: `.context/deletion-bench-{before,after}-{segment_merge,search_pipeline}.log`,
`.context/deletion-bench-repeat-{before,after}-{segment_merge,search_pipeline}.log`,
and `.context/deletion-bench-environment.txt`. Criterion samples are under
`.context/deletion-bench-run/target/criterion/`. The reverse pass includes the
final physical-copy visibility guard; subsequent changes only add tests/docs.

### Remaining review findings and limits

- The small clean-copy merge fixture regressed in both passes (about 10 µs,
  33–40%). The larger copy fixture improved, and current-thread search changed
  direction between passes. This was unresolved at that stage; the follow-up
  below profiles the empty-dictionary cost and removes it. These microbenchmarks
  do not establish a production throughput change.
- The initial implementation compacted dirty merges automatically and wrote a
  scratch segment first. The follow-up below removes that default cost: ordinary
  merges retain masks, and explicit compaction rewrites each final output directly.
  No cold-storage, large-corpus throughput, or cross-architecture measurement was run.
- Compaction has explicit bounded scratch admission. Very large row maps or
  high-frequency positioned terms can exceed the supplied budget and return an
  error; there is no silent truncation or unbounded fallback.
- Bloom filters retain deleted keys as false positives. Heavy churn can reduce
  their selectivity; exact live-key checks preserve correctness. Adaptive Bloom
  rebuilding and long-running churn throughput were not measured here.
- Format 7 requires rebuilding older indexes. Updates replace full documents
  and permit one pending insertion/update per key per commit. RPC mutation
  endpoints/cross-shard atomic updates are outside this native-core/CLI change.
  BM25 statistics remain physical until compaction.

## Explicit compaction and performance review — 2026-09-07

The follow-up changes ordinary merge to retain tombstones. `ForceMerge.compact`
(and the Python/TypeScript helpers and CLI `merge --compact`) compacts each final
output once, including a singleton. The existing optimizer selects segments from
physical/deleted metadata counts at a configurable ratio (default 0.30; 0 disables).
It shares task slots, CPU pools, global/local merge capacity and the optimizer BP
gate. One automatic compaction is allowed globally, followed by a 60-second
completion cooldown. This requires the existing optimizer to be enabled; it does
not create another worker pool. Compaction scratch defaults to 256 MiB.

### Review findings and implemented improvements

- **Default merge cost:** encoded payloads remain on their ordinary copy/remap
  paths. Only mask words are remapped (`O(physical_rows / 64)`). Address-preserving
  single-source reorder/ANN rewrites reuse the exact immutable mask file. Explicit
  compaction writes directly from final sources; it does not compact intermediate
  merge outputs or reconstruct indexed-only values from the document store.
- **Search and primary keys:** clean searches borrow `None` without allocating a
  deletion bitmap; dirty searches borrow the generation's mask and apply it before
  top-k/pruning admission. A final visibility wrapper protects generic scorers.
  PK Bloom bits are never cleared. Exact dictionary membership is checked against
  a precomputed live-ordinal bitmap, keeping insert checks free of per-key row
  scans. The bitmap is rebuilt once per changed visibility generation.
- **Compaction CPU:** fast columns previously encoded every 4,096-row chunk twice
  to produce a leading directory, allocating a temporary value vector per row.
  They now retain a bounded prefix of encoded chunks and use existing value
  iterators. The uncached suffix uses the same deterministic second-pass encoder.
  Cache capacity (including vector capacities) is capped at a quarter of remaining
  scratch; directory entries reserve another quarter and chunk encoding half.
- **Compaction memory:** physical row maps now reserve `4 * (physical + live)`
  bytes instead of `8 * physical`, using validated deletion counts. They reject
  unexpected extra survivors before reallocating. At 50% deletion this saves 25%
  of row-map storage; at 90% it saves 45%. Chunk maps retain a conservative bound
  where the live virtual-record count is not known in advance.
- **Empty dictionary overhead:** a macOS `sample` profile found FST registry
  initialization dominating the small numeric-only merge fixture in both main
  and this branch (957/940 top-of-stack samples respectively in the two captures).
  The owning block-index encoder now caches only its canonical empty encoding,
  bounded by a test to 128 bytes. It retains no FST build registry and leaves the
  nonempty path unchanged. Tests compare complete bytes and empty lookup behavior.
- **Ordering and statistics:** compaction stably filters physical and per-field
  BMP order, rebuilds affected block/statistic metadata, and retains BP history.
  Previously reordered surviving BMP layouts lose convergence; compaction neither
  resets nor consumes the lineage's attempt budget. Index-info aggregates counts
  in one pass; the broker computes a weighted ratio from summed counts.
- **Lifecycle cost/correctness:** measurement found that CLI exit could interrupt
  retired-file cleanup. Row mutation/merge commands now stop workers, release
  writer snapshots, and drain core cleanup, including on maintenance errors.
  ForceMerge now uses Commit's admission rule: cancellation before obtaining the
  writer starts no task; admitted work owns the writer through reader refresh.
  Detached failures are logged. Both issues have failing-before/passing-after
  regression tests (`compaction-{cli-drain,rpc-admission}-{red,green}.log`).

### Matched measurements

All figures below use the same Apple M4 / 32 GiB Mac, Rust 1.98.1, default
sync/native features and unchanged fixtures between each pair. `RUSTFLAGS` and
`CARGO_ENCODED_RUSTFLAGS` were unset. No task-owned compiler or test ran during
measurements; other applications on this shared machine were not controlled.
These are synthetic CPU/allocator fixtures, not production latency or recall.

The `segment_merge` benchmark now includes `row_compaction/mixed_fast_columns`:
one primary-key text column plus eight numeric columns, missing values, one
multi-value column, 50% deleted rows, and a 32 MiB compaction budget. Fixture
building/deletion/validation is outside timing. Each iteration calls the actual
compactor and overwrites one unpublished RAM output. The baseline binary was
preserved before the chunk cache, iterator and row-map improvements; a final
binary also includes the empty-FST cache. Criterion used 20 samples, 0.5-second
warmup and 1-second requested measurement (extended for slow iterations).

| Physical rows |    Before | Chunk/map changes | Before, reverse repeat | Chunk/map changes, reverse repeat |     Final |
| ------------- | --------: | ----------------: | ---------------------: | --------------------------------: | --------: |
| 4,096         | 3.1107 ms |         1.8308 ms |              3.1188 ms |                         1.7829 ms | 1.8179 ms |
| 65,536        | 113.25 ms |         60.287 ms |              113.25 ms |                         62.288 ms | 60.138 ms |

This is about **42–47% less compaction time**, with the improvement surviving
reversed execution order. Complete `.fast` outputs match byte-for-byte at both
sizes: 153,906 and 2,477,059 bytes. Captures and SHA-256 digests are recorded in
`.context/compaction-perf-evidence.json` and `compaction-perf-{before,final}-*.fast`.
The fixture's first 65K setup attempt hit `QueueFull`; the harness was corrected
to wait for admission, and both compared binaries use that corrected setup.

Peak whole-process RSS **increased**: 145.2 → 163.6 MiB in the first pair and
133.6 → 146.1 MiB in the reverse pair; final was 161.4 MiB. RSS includes index
building, source readers, allocator retention and output buffers, so it does not
isolate scratch. The encoded cache trades bounded memory for CPU, within the
existing cap. Physical map allocation is separately bounded by the exact formula
above; a regression verifies a mostly-deleted map fits that smaller budget and
refuses growth beyond its admitted survivor count.

The earlier small clean-merge regression was investigated rather than dismissed.
With a 1-second warmup and 3-second measurement, the unchanged 2 × 4,096 numeric
fixture measured 27.801 µs on `origin/main` versus 34.104 µs before the empty-FST
fix. Final measured 5.328 µs; the reversed main repeat was 36.501 µs. The broad
spread demonstrates machine noise, but removal of repeated empty-registry work
is clear. A final all-case run measured 4.772/4.579 µs for 4K copy/missing columns
and 27.701/17.056 µs for 64K copy/missing columns. This benefit applies to empty
term dictionaries; it is not a claim that populated text merges improve equally.
The sampled profiles are attribution evidence, not the source of timing numbers.

A separate matched **debug CLI** fixture used two 4,096-row segments with a stored
primary key, numeric fast field, indexed-only 129-token text, and 50% deletion.
Three runs rotated execution order and checked all 4,096 surviving keys after
each command. After adding cleanup draining, median default merge was 0.10 s
(range 0.10–0.65), explicit compaction 0.39 s (0.39–0.40), and the initial implicit
compaction implementation 0.40 s (0.38–0.47). Median RSS was 29.8/31.7/30.8 MiB,
respectively. The old command could leave retired files, so its command time does
not include an equivalent cleanup guarantee. Both final modes left only owned
segment files. This run predates the chunk/FST optimizations; it establishes the
cost distinction of the API flag, not final release throughput.

Raw evidence: `.context/compaction-perf-{before,after,final}.log`,
`.context/compaction-perf-repeat-{before,after}.log`,
`.context/compaction-clean-{main,current,final,main-repeat}.log`,
`.context/compaction-clean-sample-{main,current}.txt`, and
`.context/compaction-cost/final/`. Build logs and binary hashes accompany the
fixture captures. Ignore Criterion's automatic cross-run percentage overlays;
the table above compares the recorded point estimates from the named binaries.

### Remaining limits and follow-up measurements

- No large-corpus ANN deletion/compaction, cold-storage throughput, x86/AVX2,
  sustained churn, or production p99 measurement was run. Defaults were not tuned
  from these fixtures. The 30% trigger and cooldown are configurable policy.
- Visibility-only refresh currently reopens the affected segment's compact
  metadata, including validation, while payload mappings remain evictable.
  Sharing more immutable decoded metadata across visibility generations needs a
  separate measured change to reader ownership; this review does not claim that
  a deletion costs only the mask write.
- Deletion still scans affected PK columns. Persisted Bloom filters can lose
  selectivity under heavy churn, although exact live-key checks remain correct.
  Neither adaptive Bloom rebuilding nor a new reverse PK-to-row index was added.
- Oversized compaction maps, terms, positions or column values fail before
  publication when the supplied scratch budget cannot cover them. Global pacing
  also conservatively cools down admission-skipped attempts; busy workloads can
  defer compaction until a later scan. Failure retries use existing capped
  exponential backoff, rather than an unbounded busy loop.

### Final validation

- The regular `check` harness passed in
  `.context/search-harness/20260907T091258.286326Z-check/`; a later parallel `full`
  also passed in `20260907T093125.569030Z-full/` before the performance refinements.
- Final `RUST_TEST_THREADS=1 python3 scripts/check_search.py full` passed all eight
  steps in `.context/search-harness/20260907T100254.254219Z-full/`: 1,389 core,
  80 server and 5 tool unit tests, broker tests, native-without-sync and portable
  compilation, docs, and all three real-server broker E2E tests. Individual
  concurrency tests retain their own worker/runtime concurrency.
- Two parallel final attempts hit the broker harness's 10-second discovery wait;
  the later failure log contains `Address already in use` from its bind/drop port
  probe. Isolated retry passed. Serial test scheduling avoided that harness race;
  production settings and the test timeouts were not changed. Failure evidence:
  `20260907T092846.698857Z-full/03-test.log` and
  `20260907T095930.504752Z-full/03-test.log`.
- Final WASM build and all 13 tests passed, including persisted visibility and
  corruption handling (`.context/compaction-review-wasm-{build,test}.log`).
  Python and TypeScript client unit tests each passed 12 tests, including flag
  serialization and deletion statistics; generated bindings were refreshed using
  the repository scripts (`.context/compaction-{python,ts}-*.log`).
- Final CLI smoke verified default merge retains 8,192 physical/4,096 live rows,
  then explicit singleton compaction leaves 4,096 physical/live rows and no
  tombstones. All 4,096 indexed-only text matches retain their exact primary keys,
  and no retired segment files remain (`.context/compaction-final-smoke.log`).
  Documentation/link checks and `git diff --check` passed.

## Second deletion review: identity, cancellation, and overlapping maintenance

This pass traced native mutation admission through manager publication, PK cache
refresh, ordinary merge, compaction, and retirement. It also checked the shared
fast-column decoder used by native async and WASM builds. Three additional bugs
were reproduced before fixing them:

- A document with two primary-key values reserved its first value but persisted
  its last value in the single-value fast column. Inserts and updates now share
  one validator that requires exactly one nonempty text key, capped at 65,536
  bytes. Validation precedes reservations and staged deletion; ordinary deduped
  inserts cannot admit a key too large for the deletion API. The regression
  verifies rejected input leaves no pending mutation, preserves the original
  row, and does not reserve either invalid insertion key.
- Cancellation could stop the ANN compactor's survivor iterator while the helper
  still returned success with a truncated run. The outer compactor already
  rejected publication after cancellation. The helper now checks cancellation
  before finishing its footer as well. The regression cancels during label
  writes, while complete-byte comparisons still cover all five ANN encodings.
- Optimizer retry records survived replacement of their source segments.
  Failure admission now checks current metadata under the publication lock;
  both ordinary and vector-generation replacement remove retired records under
  that lock. A late failure cannot recreate an obsolete entry. Cleanup hashes
  only retired IDs, rather than scanning the entire retry table on each merge.
  The regression reproduces a failed compaction followed by ordinary merge and
  verifies that a subsequent failure for the old ID leaves no retry record.

Failing-before evidence is retained in `.context/review2-pk-before.log`,
`.context/review2-ann-cancel-before.log`, and `.context/review2-retry-before.log`.

Additional correctness coverage includes:

- An ordinary merge paused during copying while another commit updates a row
  and deletes additional keys. The merge must carry the latest masks, retain
  all physical rows, preserve a still-pending insertion reservation, and keep
  pre-deletion and pre-merge readers valid. The fixture uses 65- and 67-row
  sources so source boundaries cross bitmap words. Subsequent explicit
  compaction preserves exactly the expected live keys and replacement value.
- An injected PK visibility-load failure after durable update/delete
  publication, followed by abort and a recovery commit. Reservations remain
  conservative, the replacement survives, and its published deletion is not
  replayed. Deleted keys become reusable after successful refresh.
- A deterministic reference-map test spanning 24 batches of mixed inserts,
  updates, deletes, aborts, ordinary merges, both compaction entry points, old
  snapshots, and reopen. It checks exact keys and versions through indexed-only
  text and numeric fast fields, including duplicate admission after Bloom reopen.
- All 128 bitmap offsets against all source lengths from 0 through 130,
  checking every output row, prior tombstones, neighboring live rows and padding.
- A 33-chunk fast column with missing values and a one-row final chunk. A 4 MiB
  budget forces cache overflow, while 16 MiB caches the entire encoded column;
  complete `.fast` bytes and every decoded value/presence bit match.
- Full deletion-file bytes compared against a scalar text-key reference scan
  over stacked dictionaries with different local ordinal orderings.

Deletion now resolves target dictionary ordinals once per segment, then uses
the existing batch column decoder to test integer membership. The target set is
bounded by the admitted key count; no corpus-sized reverse index is introduced.
Both dictionary resolution and scanning use the existing background CPU pool,
and the decoder propagates cancellation without processing the rest of a column.
This removes per-row text decoding, dictionary lookup and string hashing.

The publication lock still spans deletion preparation and persistence. Existing
searcher snapshots remain usable, but acquiring a new manager snapshot or
publishing maintenance can wait behind that commit. This pass reduces that work;
it does not claim a bounded production p99 or introduce optimistic rebase/retry
semantics. Large cold-storage, ANN-heavy and sustained-churn measurements remain
outstanding, as do x86 measurements. Defaults remain unchanged.

### Deletion commit measurements

The existing `segment_merge` benchmark now also measures
`row_deletion/commit_64_keys`: each iteration copies identical immutable RAM
index files, opens a fresh writer, initializes PK state, and stages 64 evenly
spaced keys. Only `writer.commit()` is timed, including publication and PK
visibility refresh. Setup, live/physical-count verification, and worker shutdown
remain outside timing. The saved binaries bracket the ordinal-scan change and
use the same benchmark source, Rust 1.98.1 release/default native+sync flags,
Apple M4 / 32 GiB Mac, and unset `RUSTFLAGS`/`CARGO_ENCODED_RUSTFLAGS`.
No task-owned build or test ran during measurement.

Initial 10-sample runs with 0.5-second warmup/1-second requested measurement
were noisy: 4K rows measured 389.74 → 400.86 µs, then 498.94 → 1,465.7 µs;
the latter optimized interval spanned 629–2,686 µs. The 65K case improved in
both pairs (12.625 → 4.411 ms and 15.695 → 5.549 ms). Longer runs used 1-second
warmup and 3-second requested measurement, retaining 10 samples and alternating
binary order between sizes:

| Physical rows / deleted keys | String scan | Ordinal scan | Time reduction | Peak process RSS, before → after |
| ---------------------------- | ----------: | -----------: | -------------: | -------------------------------: |
| 4,096 / 64                   |   339.08 µs |    278.97 µs |          17.7% |              123.25 → 126.33 MiB |
| 65,536 / 64                  |   10.835 ms |    3.0113 ms |          72.2% |              123.70 → 130.47 MiB |

Long-run timing intervals were 337.98–340.57 / 277.19–280.23 µs and
10.799–10.890 / 2.9817–3.0421 ms respectively. RSS includes all benchmark
fixtures, setup, worker pools, and allocator retention, so these figures do not
isolate the deletion target set. The integer set adds bounded temporary storage;
the optimization is primarily a CPU improvement. Short-run RSS changed in the
opposite direction (130.81 → 126.73 and 141.78 → 135.39 MiB), reinforcing the
need to avoid inferring an isolated allocation delta from process peaks.

Raw logs are `.context/review2-deletion-{before,after}.log`, their `-repeat`
variants, and `review2-deletion-{small,large}-{before,after}.log`.
`.context/review2-deletion-evidence.json` records fixtures, environment, binary
hashes and timing/RSS values. Criterion's automatic cross-run comparison lines
refer to its last result, not necessarily the intended pair; the table uses
the named binaries' recorded point estimates. These figures do not establish
production tail latency or cold-storage throughput.

### Validation of this pass

- `python3 scripts/check_search.py check` passed in
  `.context/search-harness/20260907T112132.747688Z-check/`.
- Final `RUST_TEST_THREADS=1 python3 scripts/check_search.py full` passed all
  eight stages in `20260907T113345.954895Z-full/`, including 1,397 core tests,
  80 server tests, broker/tool tests, native async and portable compilation,
  documentation, and three real-server E2E tests. Serial test scheduling avoids
  the previously observed broker port-probe race; concurrency regressions still
  exercise their own concurrent tasks and multithread runtimes.
- The new sequence test initially used a stable segment ID as an array index;
  this test-fixture error was corrected to use the searcher's segment map before
  the successful final run. The earlier full failure is recorded in
  `20260907T112643.811977Z-full/`.
- Native without sync passed all 18 selected deletion/visibility tests,
  including the mixed sequence and scalar-versus-batch mask-byte comparison
  (`.context/review2-native-async-final.log`). The shared-code WASM rebuild and
  all 13 JavaScript tests passed (`review2-wasm-{build,tests}.log`).
- Documentation/link checks passed (76 files, 281 links, 20 benchmark targets),
  and `git diff --check` passed. Client/protocol files were unchanged in this
  pass; their earlier validation is recorded above.

### Mutation surfaces and portable writer review (September 7, 2026)

Delete/upsert now reach every writable surface: native core/CLI, IndexService,
broker, Python/TypeScript clients, and WASM LocalIndex. The additive RPCs preserve
explicit commit and whole-document/chunk semantics. The server holds the existing
exclusive writer guard during staged mutations; four shared admission permits
bound conversion and staging. Started blocking workers own their permit/guard
through cancellation. Envelope limits are shared by the server and broker from
`summa-proto/mutations.rs`: 100,000 deletion keys / 8 MiB key bytes and 1,000
replacement documents / 32 MiB encoded bytes. Broker partition error mapping
validates total accounting, unique error positions, and bounds; it never invents
successful operations from an incomplete backend response. Cross-shard publication
remains non-atomic and mutations are not automatically retried.

Portable writes reuse the primary-key reservation/Bloom implementation, global
ordinal batch scanner, and mask encoder. Portable reopen builds its Bloom from
fast dictionaries; it does not yet persist a Bloom cache. Memory includes the
existing fast readers, Bloom bits (10 bits/key plus 100,000-key headroom), pending
key reservations, and dirty-segment live-key bitmaps. RAM data bytes remain shared
with the directory. Mask matching retains the bounded key-ordinal set and one
physical-row bitset. There is no corpus-sized reverse key map.

Review found quadratic metadata scans when refreshing many segment visibilities
or replacing many cached PK readers. Refresh now iterates visibility identities
once, looks up membership in metadata maps, and replaces readers with set
membership. Portable output protection/retirement also computes an ID set once
per operation. These sets are temporary and proportional to metadata, not rows.
Prepared live-key bitmaps are not decoded twice during portable initialization.

A failed/cancelled portable builder poisons the pending transaction until abort,
preventing an upsert from committing only its deletion. Metadata-save cancellation
is reconciled against the durable generation before cleanup or replay. All mask
and segment output IDs are claimed before writes. Portable open reclaims known
unreferenced segment artifacts; storage synchronization tracks attempted file
writes so retries can remove orphan files after successful metadata publication.
LocalIndex storage requires atomic per-file replacement and one writable instance
per namespace. RemoteIndex/IpfsIndex remain readers. Replacement batches preserve
the existing JS/serde conversion semantics and count JSON bytes without a second
serialized payload buffer.

New coverage includes RPC pre-admission size checks, stable partial-error positions,
concurrent same-key replacement, cancellation while awaiting the writer, two-server
partition routing and indexed-only chunks, malformed backend accounting, client
forwarding and single-item failures, portable build failure/cancellation, both
sides of metadata rename, storage failure before/after metadata replacement,
abort, reopen, and Bloom key reuse. Validation logs and measured evidence follow below.

The TypeScript transport test also reproduced loss of batch error details at
its default 4 MiB receive cap: 100,000 rejected deletions produced a 7,883,486-byte
response. TypeScript now uses the same bounded 50 MiB send/receive caps as Python.
The real gRPC regression receives every error and preserves index 99,999.

Validation for the mutation-surface extension:

- `python3 scripts/check_search.py check` passed all four stages in
  `.context/search-harness/20260907T121315.750336Z-check/`.
- `RUST_TEST_THREADS=1 python3 scripts/check_search.py full` passed all eight
  stages in `.context/search-harness/20260907T121523.104566Z-full/`: 1,397 core
  tests, 82 server tests, 57 broker unit tests, 13 broker integration tests,
  tool tests and four real-server broker tests. The expanded single/partitioned
  mutation E2E suite passed again in `.context/mutations-broker-e2e-final.log`.
- After the metadata-scan optimization, 52 deletion/PK tests passed in each of
  native sync and native async builds (`mutations-native-final-tests.log` and
  `mutations-native-async-tests.log`). These include complete deletion-mask byte
  comparisons, old snapshots, merge/compaction, and cancellation regressions.
- The portable writer's two fault-injection integration tests passed with
  `cargo test -p summa-core --no-default-features --features wasm --test portable_mutations`.
  They exercise failed/cancelled builds and failed/cancelled metadata rename.
- WASM release build and all 19 Vitest tests passed. Python's 13 unit tests and
  TypeScript's 14 tests passed, including the real gRPC maximum-deletion-batch
  transport regression. Bindings were regenerated through repository scripts.
- Native all-target Clippy and portable `--features wasm --lib` Clippy passed
  with `-D warnings`. Portable Clippy also exposed two existing conditional/
  nested-control-flow warnings; those small no-op control-flow cases were fixed.
  Documentation contracts, formatting and `git diff --check` passed.

The shared-core refactor was measured against the saved pre-extension ordinal-scan
binary using unchanged `row_deletion/commit_64_keys` fixtures, the same release
compiler/flags/machine, 10 samples, 1-second warmup and 3-second requested
measurement. No task-owned builds/tests ran during these measurements. Small and
large fixtures alternated binary order; peak RSS includes fixture setup and worker
pools. Host process activity and binary SHA-256 values are captured in
`.context/mutations-perf-evidence.json`.

| Physical rows / deleted keys |          Before extension |           After extension | Peak process RSS, before → after |
| ---------------------------- | ------------------------: | ------------------------: | -------------------------------: |
| 4,096 / 64                   | 593.14 µs (572.49–618.69) | 635.18 µs (582.17–670.62) |              125.92 → 114.69 MiB |
| 65,536 / 64                  | 6.2745 ms (5.7011–6.8714) | 5.7149 ms (5.4895–6.2066) |              121.84 → 123.56 MiB |

The timing intervals overlap in both cases. These runs do not establish a
speedup or a regression; earlier measurements in this document came from a
different period of host load and should not be used as the before value for
this change. Memory peaks also include allocator retention. The removal of
quadratic segment-metadata scans is an algorithmic improvement; this single-
segment fixture does not measure that scaling benefit. No scheduling or
compaction defaults were changed based on these measurements.

### Canonical upsert API naming

The unreleased replacement API is named `upsert` across native/portable writers,
CLI, gRPC server/broker, Python, TypeScript, and WASM LocalIndex. RPC bindings are
regenerated from `UpsertDocuments(UpsertDocumentsRequest)`. Error messages and
usage examples use the same terminology. This is a naming change: insertion of
missing keys, complete replacement of existing documents/chunks, commit semantics,
limits and encoded storage are unchanged. No performance algorithm or defaults
changed; the preceding measured evidence still applies.

Validation after the rename:

- `RUST_TEST_THREADS=1 python3 scripts/check_search.py full` passed all eight
  stages, including the four `check` stages and four real-server broker tests.
  Evidence: `.context/search-harness/20260907T125925.873829Z-full/`.
- Regenerated Python and TypeScript bindings; all 13 Python and 14 TypeScript
  tests passed, including the real gRPC transport regression.
- Both portable writer fault-injection tests passed. The WASM release build
  and all 19 Vitest tests passed with the generated `upsertDocument` and
  `upsertDocuments` exports.
- A CLI smoke test invoked `upsert --help`, inserted a missing key, replaced
  that key, and verified one live document with only its replacement searchable.
  Logs: `.context/upsert-cli-smoke.log`, `.context/upsert-portable-tests.log`,
  `.context/upsert-wasm-tests.log`, `.context/upsert-python-tests.log`, and
  `.context/upsert-ts-tests.log`.
- Source/binding scans found no remaining old replacement API identifiers;
  formatting and `git diff --check` passed. No new correctness or performance
  findings remain from this naming pass.

## Hybrid fast-only text filters — 2026-09-07

The reported `type = journal-article` failure is confirmed on 1.8.132. A read-only
probe of a document shard returned three hits for a standalone type filter,
three for unfiltered fusion, and zero for the same fusion with the type filter;
none of the responses was truncated. The production field is
`field type: text<raw_ci> [fast]`: it has a fast column and no inverted postings.
The probe used the public word `quantum` against `short_document`, top three,
and a 1.5-second server budget. These three live samples establish the mismatch,
not production latency percentiles. Local evidence is retained in
`.context/type-filter-production-{metadata,evidence}.json`.

Server fusion converts common filters once and wraps every nomination branch
in core `FilteredQuery`, including RRF, formula ranking, candidate export and
feature collection. That wrapper prefers a complete document bitmap. Previously,
`TermQuery::as_doc_bitset` treated absent inverted postings as an empty bitmap,
while the ordinary term scorer matched the fast column. This emptied every
branch before scoring, independent of embedding availability or the hyphen in
the type value. The new behavior-named regression failed on that exact mismatch
before the fix (`.context/type-filter-red.log`).

Core now materializes fast-only equality on native/sync and portable execution,
preserving global text ordinals, missing/empty values, and existing first-value
semantics for multi-value fast columns. Indexed terms keep posting-based
membership. Single-value materialization reuses the existing batch decoder,
adding caller-controlled early exit and 2 KiB of stack scratch; it does not
allocate a document-sized candidate heap or change persisted bytes. The existing
16 MiB bitmap and 64-filter limits still apply. Tests cover a 200,001-document
segment above the generic scorer-fallback cap, positive/negative Boolean filters,
date/type combinations, unchanged scores/ordinals across fusion modes, two-shard
broker requests and WASM branch filters.

Deadline-aware scans discard incomplete bitmaps. The ordinary fast-text scorer
also checks the deadline while skipping nonmatching values. Bitmap enumeration
captures each observed document ID once: a second `doc()` call could otherwise
turn into `TERMINATED` between the check and the bitmap write. A deterministic
regression covers that cancellation boundary, and batch-reader tests cover early
exit inside/across batches and remapped text dictionaries across merged blocks.

The first large-fixture attempt exposed repeated blockwise-linear header decoding
when materialization reused scalar fast-field reads. It was stopped after a CPU
sample identified that path, then single-value materialization was changed to
batch reads. The related general limitation remains: ordinary fast-only term
scans and multi-value first-value scans still use the existing scalar decoder;
their codec-dependent cost is not a linear-time guarantee. This change makes no
new performance claim for those paths and changes no compression/default policy.

Warm local RPC samples used the same persisted 10,000-document fixture, Apple M4,
Rust 1.98.1, default debug server build flags and 20 warmups plus 100 sequential
requests per mode. The first 9,950 documents are books; the last 50 are journal
articles. Two text branches request top three. The saved pre-fix server was built
at `7076b38d`; the intervening `7a4b380c` change only bumps release versions.
The fixture metadata SHA-256 stayed
`b3f3f04e9abf73e4f78cbf646f0d72eb37de19a2230e314fb69b26921a395c43`.

| Request                           |                  Before p50 / p95 | After p50 / p95 | Correctness                     |
| --------------------------------- | --------------------------------: | --------------: | ------------------------------- |
| Standalone fast-only type filter  |                    6.69 / 9.00 ms |  2.90 / 3.66 ms | Same IDs and score bits         |
| Unfiltered hybrid                 |                    6.01 / 6.99 ms |  2.78 / 3.12 ms | Same IDs and score bits         |
| Hybrid with indexed-type control  |                    5.18 / 6.38 ms |  2.55 / 2.93 ms | Same IDs and score bits         |
| Hybrid with fast-only type filter | 1.48 / 1.67 ms, incorrectly empty |  3.08 / 3.72 ms | Matches indexed control exactly |

Maximum sampled server RSS across the four modes was 40,912 KiB before and
37,520 KiB after; this includes mapped pages and heap, not a separate heap profile.
No builds from this workspace ran during sampling, but the shared host was not
controlled. Unchanged controls also became faster, so these before/after times
are not evidence of a general speedup. The corrected fast-only filter cost about
0.54 ms more than the indexed control in the after run. A fast-only field still
requires reading its column; indexing that field and rebuilding existing data
would remove that scan. This fix does not add a request cache: branches retain
the existing filter ownership and bounded memory lifetime. Script and raw samples:
`.context/measure_type_filter.py`, `.context/type-filter-{before,after}-samples.jsonl`.

All eight full-harness steps passed with `RUST_TEST_THREADS=1` and the isolated
native build directory: `.context/search-harness/20260907T135854.914377Z-full/`.
This includes 1,373 core unit tests, server/broker/tool checks, native-without-sync
and portable compilation, documentation and all three real-server broker tests.
The one-thread setting avoids the previously recorded broker discovery flake;
no production timeout or concurrency setting changed. Thirteen targeted common
filter tests also passed with native async execution, and the WASM release build
and all 13 runtime tests passed. Logs:
`.context/type-filter-{full-release,native-async,wasm-build,wasm-tests}.log`.

The earlier full run caught an invalid new RRF test request: `candidate_depth`
is only accepted for L1/exports. Correcting that test fixture in server and broker
tests made the regression and final full run pass. Its initial failure remains
in `.context/search-harness/20260907T133822.951694Z-full/`. An exploratory run
was intentionally stopped while replacing the scalar scan; the harness logged
an interrupt-cleanup `Operation not permitted` error in
`.context/search-harness/20260907T133111.044272Z-full/`. Neither is an unresolved
failure of the final checks. The earlier native-async phrase-ordinal discrepancy
and parallel broker discovery flake remain separate recorded findings.

## 2026-09-09: measuring retained segment candidates for L1

This is a design experiment, not a serving change. Retaining already-produced
segment candidates improves formula top-K recall on mixed segments, but it can
add substantial L1 work without improving recall on clustered segments. The
measurements do not support enabling it universally.

### Method and invariant

The isolated checkout at `.context/segment-l1-measure-worktree` is based on
`6989e797` (1.8.134). Test-only hooks observe completed segment result lists in
the existing searcher, **after** its shared threshold is raised and **before**
its shard merge discards results. They do not alter segment depth, shared vertical
thresholds, BMP's global LSP selection, search concurrency or scoring kernels.
L1 runs after nomination; its predictions never feed retrieval thresholds.
The normal nomination lists remain included, with addresses, scores and passage
rows checked against the captured copies on every extra-candidate request.

Each branch keeps a bounded heap ordered by its existing raw score and canonical
address tie-break. With normal depth `d` and extra allowance `e`, this heap keeps
at most `d + e` candidates from completed segment results. It does not capture
all visited documents or guarantee an independent per-segment top-d. A candidate
pruned inside segment execution remains unavailable. This is one specific extra
selection policy; the experiment does not establish its optimality.

The five compared policies use three nomination branches and final top-10:

| Policy     | Retrieval depth per branch | Maximum retained slots across branches |
| ---------- | -------------------------: | -------------------------------------: |
| Baseline   |                         20 |                                     60 |
| Extras 20  |                         20 |                                    120 |
| Extras 80  |                         20 |                                    300 |
| Deeper 40  |                         40 |                                    120 |
| Deeper 100 |                        100 |                                    300 |

Retained slots include duplicates across branches, so document unions are smaller.
The deeper controls have the same maximum retained allowance, not necessarily the
same actual union or read cost. They use the core interface; an RPC caller must
also satisfy the existing result-window/candidate oversubscription limits.

All policies use the existing core candidate scorer, preserve organic scores,
backfill missing cells, and apply the formula
`bm25 + 2 * ln(1 + sparse) + 8 * dense`. The features are four-term BM25,
12-dimension BMP sparse retrieval and 128-dimensional F32 flat dense retrieval.
Document features use MAX; the passage fixture aligns three body/sparse/dense
ordinals per document and applies MAX after passage L1. Some documents have no
sparse or dense field. The formula and generator were fixed before sampling.
BMP uses its normal block/heap pruning with gamma 0; a separate capped case uses
one global gamma 2. Each case keeps that setting fixed across all five policies.

The fixed seeded corpus has 32 topics and 16,384 documents, laid out as one,
four or 16 mixed segments, 16 topic-clustered segments, or 16 uneven segments
(one holds 80% of the corpus). Additional cases cover capped BMP and three
passages per document. A 65,536-document mixed fixture checks scale. A one-search-
thread repeat uses the **same persisted files** as the four-thread mixed fixture.
Fixture file hashes are retained with the raw results.

Exhaustive core feature scoring over every fixture document (every stored passage
for the passage case) defines the reference top-10. **Recall below is agreement
with that formula reference, not human relevance or teacher-labelled recall.**
There were no representative production requests or relevance labels available
in the workspace. The generator is synthetic; the binary IVF fields used by the
live installation are not represented by this F32 flat dense fixture.

### Results

Every cell below is **formula recall@10; median core latency**. Latency includes
nomination, pool assembly and L1, with compilation, fixture creation, expression
compilation, statistics preparation, reference scoring, assertions, JSON and RPC
outside the timed region. Each policy has 32 queries repeated four times, with
rotating policy order. All sampling uses the same release binary on Apple M4
arm64, rustc 1.98.1, two Tokio workers and four search threads unless noted.
All payloads fit in memory; these are warm, sequential-request measurements.

| Fixture                                           |       Baseline |      Extras 20 |       Extras 80 |      Deeper 40 |      Deeper 100 |
| ------------------------------------------------- | -------------: | -------------: | --------------: | -------------: | --------------: |
| 16k documents, 1 segment                          | 75.0%; 0.28 ms | 75.0%; 0.28 ms |  75.0%; 0.28 ms | 94.7%; 0.34 ms | 100.0%; 0.47 ms |
| 16k documents, 4 mixed segments                   | 75.0%; 0.28 ms | 94.7%; 0.32 ms |  99.7%; 0.40 ms | 94.7%; 0.35 ms | 100.0%; 0.50 ms |
| 16k documents, 16 mixed segments                  | 75.0%; 0.50 ms | 94.7%; 0.59 ms | 100.0%; 0.74 ms | 94.7%; 0.64 ms | 100.0%; 0.86 ms |
| 16k documents, 16 clustered segments              | 75.0%; 0.24 ms | 75.0%; 0.53 ms |  75.0%; 0.73 ms | 94.7%; 0.31 ms | 100.0%; 0.47 ms |
| 16k documents, 16 uneven segments                 | 75.0%; 0.38 ms | 85.1%; 0.50 ms |  85.3%; 0.58 ms | 94.7%; 0.49 ms | 100.0%; 0.70 ms |
| 16k documents, 16 mixed segments, gamma 2         | 71.6%; 0.45 ms | 91.9%; 0.56 ms | 100.0%; 0.70 ms | 91.9%; 0.59 ms | 100.0%; 0.83 ms |
| 16k documents, 3 passages, 16 segments            | 56.2%; 0.86 ms | 86.2%; 1.00 ms | 100.0%; 1.30 ms | 86.2%; 1.06 ms |  99.7%; 1.61 ms |
| 65k documents, 16 mixed segments                  | 48.4%; 0.99 ms | 71.2%; 1.07 ms |  98.1%; 1.32 ms | 71.2%; 1.19 ms |  98.1%; 1.55 ms |
| 16k documents, 16 mixed segments, 1 search thread | 75.0%; 0.75 ms | 94.7%; 0.83 ms | 100.0%; 0.95 ms | 94.7%; 0.96 ms | 100.0%; 1.31 ms |

For 65k documents, Extras 80 retained about 286 documents versus 60 in the
baseline and supplied 5.16 previously unnominated documents to the final top-10
on average. Its p95 was 1.77 ms versus 1.29 ms baseline and 1.90 ms Deeper 100.
On 16k mixed segments, Extras 20 supplied 2.16 new top-10 documents per request,
and Extras 80 supplied 2.50; 30 of the 32 queries benefited from extra winners.
On clustered segments both supplied **zero** new winning documents despite
retaining about 116 and 281 documents. Their respective median L1 times grew
from 0.06 ms baseline to 0.34 and 0.52 ms. Captured candidates from other segments
could not recover the stronger candidates discarded inside the topic's segment.

The uneven case gained approximately ten recall points from Extras 20, but
Deeper 40 gained approximately twenty points at a similar median cost. Raising
the extra allowance to 80 barely improved its recall. More retained rows alone
are therefore a poor reason to spend additional backfill budget. Segment merging
and corpus layout can materially change the benefit.

### Pruning, work and memory

Normal nomination document identities and order were unchanged in the paired
extra/baseline measurements. Native text scores sometimes varied by one float
ULP across repeated executions; the captured row was always byte-identical to
its originating normal row within that execution. Scoring schedules can vary
under the shared threshold without changing its policy.

BMP blocks scored per request confirm that extras did not obtain their gains by
requesting deeper BMP work: approximately 221 blocks for both baseline and
Extras 80 on the 16k mixed fixture, versus 490 for Deeper 100; approximately 377
versus 377 versus 1,132 on the 65k fixture. Small run-to-run block-count variation
is retained in the samples; copying captured results can affect scheduling even
though thresholds are published before capture. There is no claim that the
instruction trace or elapsed nomination time is identical.

Separate untimed allocator passes measured requested allocation bytes and the
maximum increase in live heap from request start. They exclude assertion/report
allocations and include temporary search/scoring buffers. These are not absolute
heap residency, mmap residency or process RSS; thread overlap affects the peak.
For the 16k mixed fixture:

| Policy     | Extra capture buffers | Median allocated bytes per request | Largest observed heap increase | Backfilled component values | Charged exact-vector / BMP bytes |
| ---------- | --------------------: | ---------------------------------: | -----------------------------: | --------------------------: | -------------------------------: |
| Baseline   |                 0 KiB |                          1.324 MiB |                      177.6 KiB |                         223 |                   18.3 / 8.4 KiB |
| Extras 20  |               8.8 KiB |                          1.413 MiB |                      193.3 KiB |                         424 |                  34.6 / 16.0 KiB |
| Extras 80  |              21.9 KiB |                          1.570 MiB |                      198.1 KiB |                         869 |                  70.7 / 32.6 KiB |
| Deeper 100 |                 0 KiB |                          2.190 MiB |                      205.6 KiB |                         869 |                  70.7 / 32.6 KiB |

Capture-buffer figures include the bounded heap's normal candidates as well as
its extras. Component counts include each query component, not just one value
per feature. Charged reads are the scorer's vector/BMP admission accounting;
text mmap probes and OS page faults are not represented by those byte totals.
The complete initial process peaked at 281.6 MiB RSS including fixture building
and exhaustive references; it cannot be attributed to any one policy. No builds
from this workspace ran during timed sampling. Other activity on the shared Mac
was uncontrolled, so small latency differences should not be generalized.

### Evidence and limits

There are 5,760 timed requests and 360 separate memory requests across the nine
fixture/thread cases. Raw rows include normal nomination signatures, pool size,
new winners, formula/pool recall, phase timings, BMP work and feature-read
accounting. Artifacts:

- `.context/segment-l1-measure-results/`: the seven 16k cases, environment,
  fixture hashes, raw JSONL, and CSV/Markdown/JSON summaries.
- `.context/segment-l1-measure-large/`: the 65k mixed case.
- `.context/segment-l1-measure-single-thread/`: one-thread repeat over the same
  16k mixed index files.
- `.context/analyze_segment_l1.py`: summary calculations.
- `.context/segment-l1-measurement.patch`: test-only instrumentation and harness
  for a detached checkout of `6989e797`; no public API is added.
- `.context/segment-l1-measure-{build,run,large,single-thread}.log`: commands'
  build/execution logs. Memory instrumentation is disabled during timed passes.

The ignored unit harness is invoked as follows, after applying the experiment
patch to its detached checkout:

```sh
CARGO_TARGET_DIR="$PWD/.context/segment-l1-measure-build" CARGO_BUILD_JOBS=4 \
  cargo test --locked --release \
  --manifest-path .context/segment-l1-measure-worktree/Cargo.toml \
  -p summa-core --lib segment_nomination_measurement::measure --no-run
SEGMENT_MEASURE_OUTPUT="$PWD/.context/segment-l1-measure-results" \
  SEGMENT_MEASURE_DOCS=16384 SEGMENT_MEASURE_QUERIES=32 \
  SEGMENT_MEASURE_REPEATS=4 SEGMENT_MEASURE_THREADS=4 \
  <test-binary> --exact segment_nomination_measurement::measure \
  --ignored --nocapture --test-threads=1
python3 .context/analyze_segment_l1.py
```

`SEGMENT_MEASURE_FIXTURE=doc_s16` selects the scale/thread repeats; their output
roots must differ. Use `SEGMENT_MEASURE_DOCS=65536` for the scale repeat and
`SEGMENT_MEASURE_THREADS=1` with the shared persisted fixture for the thread
repeat. Correctness smoke runs use 512 documents and two queries; their timings
are not included in the performance summaries.

Read-only live metadata collection found one existing shard with 40 segments,
26,291,811 physical documents, segment sizes from 557 to 4,187,252 and median
4,724. The smallest 24 segments contain 67,071 documents. This motivated the
uneven fixture; it is not a measurement of live recall or latency. No production
query workload, process, deployment, index or schema was changed.

Production relevance, x86 performance, binary IVF, cold payloads, concurrent
requests/ingestion, RRF-dependent formulas and phrase-feature costs remain
unmeasured. In particular, formulas containing RRF still require ranks at the
correct global scope before final selection. The experiment supports a bounded
opt-in trial on representative queries; it does not justify a default change or
predict a production speedup.

The experiment's 512-document smoke cases passed with both native sync and native
without sync; their logs are `.context/segment-l1-measure-smoke.log` and
`.context/segment-l1-measure-async-smoke.log`. The repository's required
`python3 scripts/check_search.py check` passed with `RUST_TEST_THREADS=1`, including
formatting, focused Clippy, core/server/broker/tool tests and native-async
compilation. Evidence is in
`.context/search-harness/20260909T052512.176694Z-check/`. Full RPC/WASM checks were
not rerun: the main checkout changes only this report, and the instrumentation
exists only in the isolated experiment checkout under `cfg(test)`.

`python3 .context/run_segment_l1.py --output <new-directory>` automates creating the
detached checkout, applying the patch, building, measuring and summarizing. The
source, raw measurements, summaries and instructions are also packaged in
`.context/segment-l1-measurement.tar.gz`; fixture index files are regenerated,
not included in the archive.

## 2026-09-09: feature reads and scoring review

This subsection records the initial design and measurement pass, before the
implementation follow-up below.
The invariant is identical candidates, organic scores, raw missing/zero values,
passage ordinals, formula outputs and request-wide work/read budgets. The cost
model separates segment/feature setup and scheduling, selected payload reads,
scoring kernels, feature assembly and formula evaluation.

### Measured opportunity: amortize CPU scheduling

The native server awaits `score_candidates_with_retrieved_and_rrf` directly in
`summa-server/src/search_service.rs`. Core groups candidates by segment and
processes those groups sequentially in
`summa-core/src/query/candidate_scoring/execution.rs`. Each BMP component and
each dense/binary scoring batch independently enters `install_search_cpu`.
Consequently, a small pool spread across 16 segments can incur approximately
30 separate synchronous CPU-pool handoffs. The text probes, feature assembly
and formula loop run on the calling thread. This differs from L0's coarser
native search dispatch.

The experiment freezes each candidate pool and its organic branch lists, then
compares the existing L1 call with the **same call** entered once through the
shared search CPU pool. Nested calls then stay on that worker. Both paths still
process segments sequentially; no parallel segment scorer, scoring algorithm,
candidate expansion or pruning change is involved. The experiment polls the
mmap future with `now_or_never()` and asserts that every read is immediately
ready. It does not block arbitrary async I/O on a Rayon worker. A serving design
would need bounded async admission/completion around core CPU work, preserving
lazy-directory, current-thread runtime, cancellation and WASM behavior.

Median L1 latency, including plan validation, feature backfill, prediction and
sorting, but excluding retrieval, expression parsing, statistics and RPC:

| Fixture / branch depth              | Mean pool | Current call | One worker entry | Median reduction |
| ----------------------------------- | --------: | -----------: | ---------------: | ---------------: |
| 1 segment / 20                      |      57.6 |      69.2 us |          64.4 us |               7% |
| 16 mixed segments / 20              |      57.6 |     320.9 us |         123.7 us |              61% |
| 16 mixed segments / 100             |     246.8 |     520.8 us |         263.8 us |              49% |
| 16 uneven segments / 20             |      57.6 |     180.4 us |          91.6 us |              49% |
| 16 clustered segments / 20          |      57.6 |      60.8 us |          56.9 us |               7% |
| 16 segments, passage features / 20  |      59.2 |     402.3 us |         179.9 us |              55% |
| 16 segments, passage features / 100 |     281.1 |     768.8 us |         502.5 us |              35% |

For the mixed 16-segment case, p95 changed from 478 to 154 us at depth 20,
and from 893 to 307 us at depth 100. The one-segment/clustered cases supply most
nominees from one segment, so consolidating dispatch has little to amortize;
their small median changes are inconclusive and their p95 did not improve.

A separate repeat with **one** search-pool thread over the same mixed index
changed medians from 242 to 115 us and from 403 to 243 us respectively. The gain
therefore does not require scoring segments in parallel. Worker locality and
execution scheduling both change; the entire latency difference should not be
described as a measurement of queue wait alone.

The native-without-sync control also passed the same byte comparisons. In that
build `install_search_cpu` is already inline, so both invocation styles should
perform the same work. Measured medians were 105.4/105.7 us at depth 20 and
233.8/235.3 us at depth 100, showing no useful difference. This is a control
within that build, not a recommendation to disable native search parallelism.

Separate instrumented passes explain the difference. For the small mixed pool,
BMP/vector dispatch spans totalled about 218 us per request, while their nested
kernel spans totalled about 21 us. With one worker entry those figures were
about 11.6 and 10.5 us. These spans overlap, contain instrumentation overhead,
and come from eight queries; they are attribution evidence, not an additive
latency breakdown or replacements for the unprofiled timing samples.

Every comparison checks the complete serialized results and feature rows,
including ordering, scores, missing values and passage ordinals, byte for byte.
Backfilled component counts and charged bytes also match: approximately
223 components, 18.25 KiB of exact vectors and 8.40 KiB of BMP payload at depth
20; 869 components, 70.72 KiB and 32.64 KiB at depth 100. No candidates are
discarded to obtain the improvement.

Memory is essentially unchanged by scheduling. The mixed depth-100 call
allocated a median 373.9 KiB through 3,407 allocations in either mode, with a
largest observed live-heap increase of 67.6 KiB. This shows a separate opportunity
to reduce temporary allocations. These numbers include only L1 and must not be
compared directly with the previous section's whole-search allocation totals.

### Further findings, in suggested implementation order

1. **Reduce text cursor setup and reuse bounded scratch.**
   `term::score_term_candidates` creates a posting iterator for every component
   in every occupied segment. `BlockPostingIterator::owned` allocates two
   128-element `u32` buffers and decodes block zero before the first target seek,
   even when the first candidate belongs to a later block. It also maintains
   position-frequency prefixes although ordinary BM25 probes need no positions.
   In the one-worker profile, text backfill took about 58 us for the small mixed
   pool and 71 us for the large one, versus approximately 10/21 us for BMP
   and less than 1 us for the dense arithmetic. This makes text setup/probing a
   more promising next target than changing the formula evaluator. A targeted
   first seek and reusable decode buffers belong in the existing posting reader;
   retain positional cursor semantics for phrase consumers. The fixture does
   not isolate how much time any one of these changes would save.

2. **Reduce feature assembly allocations.**
   `execution.rs` builds nested document feature vectors, tree maps for passage
   rows, and another document-to-location tree for every document-scope feature.
   `DocumentExpression::score` then allocates a `(ordinal, score)` vector for each
   component/document reduction. Chunk-field sets are rebuilt per segment and
   again per output document; model-bearing results clone old positions before
   replacing them. Reuse admitted scratch, hoist the field set, and evaluate
   contiguous row/location spans through the existing combiner. Preserve strict
   ordinal reduction order, missing versus zero, negative boosts, and the
   difference between `MAX(a) + MAX(b)` and `MAX(a + b)`. Flat internal storage
   should still produce the same owned export rows at the boundary.

3. **Prepare each scoring query once.**
   L0 already shares `PreparedBmpQuery` across segment scorers. L1's
   `score_bmp_candidates` rebuilds its quantized/sorted vectors, candidate mask
   and phase-one metadata for every segment/component, even though point scoring
   uses no LSP or block-pruning plan. `score_vector_candidates` similarly
   recomputes the query norm and F16 representation on every segment call;
   the F16 copy is unused by F32/UInt8 kernels. Reuse immutable preparation in
   the existing core plan, keeping segment-specific BMP scale and field/dimension
   validation at execution. In the one-worker mixed depth-100 profile, BMP
   preparation totalled about 4.5 us; vector preparation **including its output
   and raw-buffer allocations** totalled about 6.4 us. This is a smaller follow-up
   than scheduling or text work on this fixture, not a measured speedup yet.

4. **Share payload reads when several branches use the same field.**
   Each feature/component currently resolves locations and reads its selected
   vectors independently. Two dense queries over the same body field can read
   and copy the same flat rows twice; multiple sparse queries can validate and
   traverse the same forward vector repeatedly. A bounded field/segment batch
   could resolve the union of missing logical cells, read a row once, and apply
   the existing kernels for each required query before scattering scores back
   to branch slots. Preserve organic values even when another branch needs the
   same row, and charge both distinct payload bytes and component work. This
   needs a repeated-field/multi-query fixture: the measured three-field workload
   does not quantify its benefit.

5. **Treat lazy text reads and MaxScore presence discovery separately.**
   `reserve_candidate_text_reads` looks up term metadata before lazy reads;
   `get_postings`/`get_positions` then look it up again and request the complete
   term range. Mmap returns byte views, but a lazy backend can materialize those
   ranges for very few candidates. Reader-owned prepared term handles, selected
   block reads, and bounded I/O batching deserve a cold/remote benchmark.
   Separately, `maxscore_candidate_locations` invokes
   `SparseIndex::probe_candidates(..., None, ...)`, which probes **all retained
   dimensions** to discover field presence and ordinals; scoring then probes the
   query dimensions. Reuse discovery across compatible branches before
   considering a format change. Looking only at query dimensions would turn
   present-but-zero values into missing values and change formulas with missing
   defaults. The current production metadata uses BMP, and the local benchmark
   also uses BMP, so this is a conditional finding rather than its measured cost.

The dense path already sorts physical targets and coalesces adjacent flat-vector
reads. BMP already has an evictable forward representation and validates only
selected payloads. Those are useful existing mechanisms to build on. A
short-circuit in the BMP arithmetic loop after the query dimensions are exhausted
could avoid some work, but `BmpForward::vector` still validates the complete
selected vector first. Do not remove that validation or pin every vector to
make warm measurements look better. Gapped-read coalescing and prefetch should
be evaluated against bytes touched and cold page faults as well as elapsed time.

### Formula cost and correctness constraints

The compiled formula `bm25 + 2 * ln(1 + sparse) + 8 * dense` costs about
1.7 us per small document pool and 7.3 us per large document pool in a replay of
the existing model scorer, with **zero allocations** in those document-only
passes. Passage merge/reduction raises those figures to 5.9 and 28.6 us and
allocates scratch in `model.rs`. Replacing the expression package is therefore
low priority for these requests. Longer formulas, RRF contribution processing,
and documents with many passages need their own measurements.

L1 still backfills branches that are absent from the formula. An inference-only
request could potentially use a formula-dependency mask, but a mask alone would
break raw exports and broker RRF: the broker deliberately asks shards to run a
constant formula while exporting every raw branch. Passage nomination must also
remain intact. Treat pruning unused feature reads as an explicit execution-plan
optimization with export requirements, not a change to formula or missing-value
semantics.

Scoring more segment candidates should follow this cleanup, then be remeasured.
These results demonstrate savings without changing candidate recall; they do
not establish that a larger pool is free. Additional missing features still
require payload access, and the extra pool must remain bounded. Per-segment
parallelism is another proposal, not part of this experiment: it must share the
request's work/read admission and CPU capacity, account for simultaneous scratch,
and be tested under concurrent queries before a throughput claim.

### Reproduction and validation

Evidence is under `.context/feature-scoring-review/`, based on `6989e797`:

- `measurement.patch`: the complete test-only patch for a detached checkout.
  `original-segment-measurement.patch` retains the preceding experiment.
- `results/`: five fixture layouts, raw JSONL and summaries.
- `single-thread/`: the same mixed fixture with one search-pool thread.
- `async-control/`: native-without-sync byte-equivalence and scheduling control.
- `analyze.py`: summary calculations and paired work/byte-count audit.
- `build.log`, `run.log`, `single-thread.log`, `focused-tests.log`.

The experiment reuses the preceding section's 16,384-document mmap indexes,
compiler, release profile and machine (Rust 1.98.1, Apple M4/arm64). It uses
32 fixed-seed queries, depths 20/100, 12 repetitions and rotating variant order,
with allocation and phase instrumentation disabled during latency measurements.
There are **9,216 timed L1 calls**, 192 separate phase samples and 192 separate
allocation samples across the four-thread and one-thread runs. Formula replay
is separate: 384 timing estimates of 64 full-pool evaluations and 384 allocation
passes. It includes passage/context reduction and score checks, but excludes
copying the input rows. The native-without-sync control adds 512 timed L1 calls,
32 phase samples, 32 allocation samples, and 64 formula timing/allocation pairs;
its timings are kept separate. No builds from this workspace ran during timed sampling;
other activity on the shared Mac was uncontrolled.

After applying the patch and building the same optimized core unit-test binary:

```sh
FEATURE_MEASURE_DATA="$PWD/.context/segment-l1-measure-results" \
  FEATURE_MEASURE_OUTPUT="$PWD/.context/feature-scoring-review/results" \
  FEATURE_MEASURE_REPEATS=12 FEATURE_MEASURE_THREADS=4 \
  <test-binary> --exact \
  segment_nomination_measurement::feature_reads::measure_features \
  --ignored --nocapture --test-threads=1
python3 .context/feature-scoring-review/analyze.py \
  .context/feature-scoring-review/results
```

Set `FEATURE_MEASURE_THREADS=1`, `FEATURE_MEASURE_FIXTURE=doc_s16`, and a new output
directory for the control. The preceding section's runner regenerates the
fixture files when needed. Optimized focused candidate-scoring tests passed
in both native builds (13 passed, one existing manual benchmark ignored in each).
The main checkout only
changes this report; the earlier `check` result at the same source revision
remains applicable. RPC, WASM, x86, cold payloads, production binary vectors and
concurrent-query throughput are not validated by these measurements. No serving
change, publication or production mutation was made for this review.

The source, samples, analyzer and standalone reproduction instructions are
packaged in `.context/feature-scoring-review.tar.gz`. Fixture indexes and compiled
binaries are excluded; the included patch regenerates the fixtures. The report
passed Prettier, `git diff --check`, and the repository's documentation/ownership
contract check for this review.

## 2026-09-09: implemented feature-read and scoring improvements

The follow-up implements the measured scheduling improvement and reduces
repeated preparation and temporary allocations. It does **not** enable the
earlier retained-segment-candidate proposal. Candidate nomination, pruning,
scores, raw exports, formulas, storage/wire formats and request limits retain
their existing semantics.

### Implementation and ownership

- `Searcher::run_search_cpu` polls the existing borrowed scoring future on the
  shared search pool when called from a multithread Tokio runtime. A poll that
  encounters pending I/O returns to the original task, releasing the worker.
  The worker enters the caller's Tokio handle for async directory operations.
  There is no spawned/detached scoring task or second scorer: a request drop
  drops its future after any active scoped poll returns. Native current-thread
  and non-Tokio callers, native without sync, and WASM keep their existing path.
- The existing posting reader accepts reusable decode buffers and starts
  selected probes at their first requested block. Ordinary posting iteration
  and its position-prefix semantics are retained. Request-owned text scratch
  is reused across components and segments.
- BMP query quantization is cached per immutable component, with bounded
  replacement when segment dimensions differ. Segment dequantization remains
  local. Dense query norms are cached per component, and F16 query copies are
  built only for F16 segments. These caches never outlive the scoring request;
  they contain query preparation, not corpus payloads.
- Document feature reduction groups sorted location spans in one reusable
  vector, replacing per-document trees/vectors. Small reductions use the same
  combiner with inline storage. Passage/context inference uses a fixed feature
  array and inline score scratch, validates duplicate ordinals before inference,
  and preserves ordinal order for strict float reductions. Chunk-field sets are
  prepared once, and formula results construct their new positions directly.

The request still computes all required missing features, including unused
formula branches needed for raw export and broker RRF. Overlapping payload-read
sharing, lazy text range planning, MaxScore presence-discovery changes and
per-segment parallelism remain separate proposals. No new cache or candidate
policy is enabled for these purposes.

### Before/after measurements

Both binaries use the same compiler, release flags, host and persisted fixtures
from the review: Rust 1.98.1, Apple M4/arm64, warm mmap, 16,384 documents, 32
fixed-seed queries, three branches and the same nonlinear formula. The before
binary is based on `6989e797`; the after binary contains this implementation.

The before run saves the complete organic branch lists, candidate pools and
serialized expected results/features. The after run replays those saved inputs
and compares **every output byte**, rather than allowing retrieval to select a
different candidate pool. All comparisons passed, and the paired component,
vector-byte and BMP-payload charges are identical. The candidate counts below
are means; latency cells are p50/p95 in microseconds and cover L1 only.

| Fixture / branch depth                     | Pool docs | Before p50/p95 | After p50/p95 |
| ------------------------------------------ | --------: | -------------: | ------------: |
| 1 segment / 20                             |      57.6 |       72 / 122 |       51 / 61 |
| 1 segment / 100                            |     246.8 |      205 / 267 |     155 / 192 |
| 16 mixed segments / 20                     |      57.6 |      316 / 583 |      94 / 112 |
| 16 mixed segments / 100                    |     246.8 |      518 / 989 |     200 / 236 |
| 16 clustered segments / 20                 |      57.6 |        57 / 73 |       42 / 50 |
| 16 clustered segments / 100                |     246.7 |      174 / 211 |     130 / 168 |
| 16 uneven segments / 20                    |      57.6 |      165 / 239 |       68 / 81 |
| 16 uneven segments / 100                   |     246.8 |      404 / 523 |     183 / 205 |
| 16 segments, passage features / 20         |      59.2 |      392 / 675 |     147 / 182 |
| 16 segments, passage features / 100        |     281.1 |     754 / 1105 |     427 / 507 |
| 16 mixed segments, one search thread / 20  |      57.6 |      241 / 321 |      93 / 113 |
| 16 mixed segments, one search thread / 100 |     246.8 |      400 / 497 |     196 / 227 |

Thus the mixed four-thread case improves median L1 latency by approximately
70% at depth 20 and 61% at depth 100. Comparing with the before binary's
experimental one-worker mode isolates the remaining opportunity approximately:
121/256 us before versus 94/200 us after, another approximately 22% reduction
from preparation/allocation changes. This is not a separate kernel benchmark.

Separate allocator passes show lower allocation traffic, with a small increase
in the live heap from retaining reusable scratch until request completion:

| Case                        | Allocated KiB before/after | Allocations before/after | Largest heap increase KiB before/after |
| --------------------------- | -------------------------: | -----------------------: | -------------------------------------: |
| Mixed segments, depth 20    |                  199 / 104 |               1664 / 961 |                            17.6 / 19.8 |
| Mixed segments, depth 100   |                  374 / 234 |              3407 / 1382 |                            67.6 / 70.0 |
| Passage features, depth 20  |                  216 / 133 |              2123 / 1430 |                            32.8 / 35.5 |
| Passage features, depth 100 |                  548 / 414 |              5880 / 3600 |                          154.0 / 155.9 |

Across the fixtures, the largest observed per-request live-heap increase rises
by approximately 1.9–4.3 KiB. These are requested allocation bytes and heap
changes measured around L1; they do not describe absolute heap, mmap residency
or process RSS. Query-preparation residency scales with the admitted active
components and their vector dimensions, so the table does not establish a
universal scratch increment for arbitrarily large formulas/queries.

The small passage-formula replay drops from 5.9 to 4.6 us, and the large replay
from 28.9 to 22.4 us; both now allocate no scratch for these at-most-three-passage
fixtures. Larger reductions spill to bounded heap storage; 17/65-passage tests
cover that boundary. The formula library and arithmetic are unchanged.

There are 18,432 timed calls across the two binaries and six fixture/thread
cases, with phase and allocation accounting disabled during timing. Half use
the normal caller and half retain the earlier worker-entry control. Separate
passes contain 384 phase samples and 384 allocation samples, plus formula
replays. Compilation and full validation finished before the timed after run;
no builds from this workspace overlapped either timed comparison. Activity on
the shared Mac was otherwise uncontrolled. This is warm single-request evidence,
not a production throughput, x86, cold I/O or relevance benchmark. The source
changes no architecture-sensitive scoring or retrieval defaults.

### Validation and evidence

`python3 scripts/check_search.py full` passed all eight steps, including the
entire `check` sequence, native-without-sync and portable builds, API docs and
the broker's real-server tests. The core suite passed 1,407 tests with 18 existing
manual benchmarks ignored. Evidence:
`.context/search-harness/20260909T064235.450347Z-full/`.

The new regressions cover CPU-pool resumption after pending I/O, cancellation
and panic/error propagation, current-thread runtimes, posting seek/position
equivalence with recycled buffers, spill-sized passage reductions, and distinct
queries across segments using BMP, F32, F16, UInt8 and binary vectors. Existing
organic-score, missing-value, phrase, reorder and admission tests also passed.
The WASM release build, clean npm install and all 20 WASM tests passed.
The native-without-sync candidate-scoring run passed 15 tests, with one existing
manual benchmark ignored.

Evidence under `.context/feature-scoring-implementation/` includes:

- `before/`, `after/`, `before-single-thread/`, `after-single-thread/`: raw
  JSONL and summaries; `frozen/*.json` contains the persisted before inputs and
  expected output bytes.
- `before-measurement.patch`, `after-measurement.patch`: complete experiment
  source for detached checkouts of the base revision, including the production
  change in the latter. The fixture builder belongs to the before experiment;
  the after experiment reuses those index files and frozen pools.
- `instrument_after.py`, `compare.py`, `analyze.py`, compiler/runtime logs,
  validation logs and fixture/environment hashes.

The reproduction source and measurements are bundled as
`.context/feature-scoring-implementation.tar.gz`; compiled binaries and index
files are excluded. Recreating an index assigns new segment IDs, so regenerate
its frozen pools with the before binary before running the after comparison.

## Compaction copy-path audit — 2026-09-11

This is a code audit, not a new benchmark or implementation change. The reviewed
compaction, row-map, store, ANN and fast-column files match fetched `origin/main`.
Ordinary merge and explicit compaction have different costs: the latter does not
yet consistently bypass decoding/rebuilding of unaffected blocks.

Current behavior:

- `segment/merger/compact.rs::compact_columns` rebuilds every surviving fast
  column and row-statistic column into 4,096-row output chunks. It resolves text
  ordinals back to strings and uses scalar value access. There is no clean source
  block copy path. Its bounded encoded-prefix cache avoids a second encoding for
  cached chunks; overflow chunks are encoded twice.
- `compact_postings` reads each external term's complete postings and position
  streams, collects surviving entries/positions, then constructs and serializes
  a new posting list. It does not copy unaffected posting blocks. Scratch is
  admitted but still scales with the largest processed term, and large terms
  can exceed the budget. Text chunk maps and norms also retain arrays.
- `segment/store.rs::append_compacted` skips fully dead store blocks and copies
  intact dictionary-free compressed blocks with remapped block metadata. Mixed
  blocks and all retained dictionary-dependent blocks are decompressed and
  recompressed. Even the classification step currently scans row membership.
- Flat-vector compaction preserves encoded values but, for a field with any
  survivors, reads every payload batch and writes each surviving vector
  separately. It does not coalesce surviving intervals or skip all-dead batches.
  ANN compaction scans run labels repeatedly, copies byte-aligned codes per
  survivor, and repacks TQ/ScaNN packed lanes; it does not bypass repacking for
  clean runs. No ANN retraining occurs.
- BMP compaction skips graph construction but still builds record maps and an
  identity permutation, then reblocks surviving records and rebuilds pruning
  metadata. Forward payload bytes are copied per surviving record. MaxScore
  sparse compaction decodes/remaps address columns for every block while
  retaining weight codes and quantizer parameters.
- The physical `RowMap` retains two u32 arrays costing `4 * (physical + live)`
  bytes, before chunk maps and encoder scratch. For 100 million physical rows
  with 10% deleted, this alone is 760 MB (about 725 MiB). The default admission
  policy rejects such a map; it is not an allocation the 256 MiB cap permits.
- Force merge with compaction performs the ordinary merge hierarchy first, then
  compacts final outputs. It therefore writes merged payloads before rewriting
  their surviving data. Direct segment compaction avoids that initial merge.

Proposed improvements, not implemented or measured by this audit:

1. Replace dense physical row maps with the existing immutable visibility bitmap,
   a compact prefix-popcount index, and survivor iteration. Preserve efficient
   reverse lookup where needed or pass old/new IDs together through bounded
   encoder batches; repeated expensive select operations could undo CPU savings.
   Apply corresponding bounded mapping to chunk and BMP address spaces.
2. Classify source blocks/ranges as dead, intact or mixed. Drop dead payloads
   before I/O, copy intact compatible payloads, and decode/rebuild mixed blocks
   through the existing encoders. Fast columns already support variable-sized
   blocks with local dictionaries in the ordinary merger, so retaining clean
   source boundaries is format-compatible. Batch-decode dirty blocks instead
   of repeating scalar codec/header work.
3. Stream posting reconstruction and positions in bounded blocks, preserving
   compatible encoded streams where possible. A surviving posting block is not
   automatically copyable: deleting a nonmatching document between two postings
   changes their ID gap. For a live row, `new_id = old_id - deleted_before(old_id)`;
   constant shift over an encoded ID range can permit payload copying with base
   metadata adjustment, while changing shifts require address reconstruction.
   Position cursors, chunk IDs and pruning metadata have their own constraints.
4. Coalesce adjacent encoded vector/forward payloads into bounded range copies,
   retain clean ANN runs or suitably aligned packed blocks, and repack only
   affected groups. Removing packed lanes can change destination alignment, so
   clean source bytes alone do not prove a packed block can be copied.
5. Consider a later fused final merge/compaction pass while retaining ordinary
   merge's default behavior and the existing publication/ownership protocol.

Clean-block copying is distribution-dependent. At the default 30% deleted-row
threshold, independent random deletions touch essentially every moderately sized
block; clustered deletions can leave long copyable intervals. Large source fast
blocks make this distinction especially important. Benchmark both distributions,
multiple deletion ratios, dictionary/non-dictionary stores, frequent positioned
terms and vector formats before claiming an overall speedup. Include bytes read,
copied and rebuilt, peak scratch and process RSS, output size, and query/encoded
payload equivalence. Smaller maps and streaming term construction address memory
growth even when no whole block is copyable.

The earlier 42–47% mixed-column compaction improvement measured an encoded cache
and related changes; process RSS increased in that comparison. It is not evidence
that the proposed block/range paths exist or that lower memory has been measured.

## Compaction copy and streaming implementation — 2026-09-11

The preceding copy-path audit describes the baseline before this implementation.
The new paths preserve format 7 and survivor ordering. Ordinary merges, optimizer
thresholds/capacity and immutable publication remain unchanged.

Implemented in the owning components:

- Physical row maps share the visibility bitmap and retain a u32 rank prefix per
  64 rows. Chunk maps own bitmap/rank state instead of two document arrays.
  Norm and chunk-output allocations reserve their admitted sizes explicitly.
- Fast fields skip dead blocks, copy intact encoded blocks and local dictionaries,
  and batch-decode mixed ranges of at most 4096 source rows. Mixed text does not
  materialize a merged global dictionary.
- Postings retain bounded encoded directories, skip dead payload blocks, and
  rebase intact block headers while preserving encoded gaps/TFs. Mixed survivors
  share a 128-entry output buffer across source blocks; it flushes before a copied
  block. Current positions copy intact encoded ranges and decode one partial
  block at a time. The codec/footer helpers remain in the existing posting owner.
- Flat vectors skip all-dead batches before I/O and copy survivor intervals in
  bounded batches. ANN ordinals/binary codes and BMP forward payloads also copy
  intervals. Clean ANN runs and aligned TQ/ScaNN groups copy their encoded bytes;
  other groups use the owning repackers. ScaNN copy widths use the format's
  padding-aware helper, including odd sub-block counts.
- Zero-budget BMP reblocking no longer constructs the unused inverse map.

### Matched measurements

Apple M4 (Mac16,13), 32 GiB, aarch64 macOS, rustc 1.98.1 / LLVM 22.1.8,
Cargo's normal optimized bench profile, no RUSTFLAGS override. Both saved
executables use the same expanded `segment_merge` fixture; the baseline is
`6c4e1ded` with only the benchmark-fixture changes. Compaction uses a 32 MiB
scratch budget and a RAM directory. Setup/deletion/validation are outside timing.
Three runs per executable alternate before/after order (B/A, A/B, B/A), with
20 samples, 1 s warmup and 2 s measurement. No task-owned builds or tests overlap
measurement. Numbers below are medians of the three Criterion point estimates.

| Fixture                      | Physical rows |    Before |     After | Time reduction |
| ---------------------------- | ------------: | --------: | --------: | -------------: |
| Alternating, numeric columns |         4,096 |  1.935 ms |  1.492 ms |          22.9% |
| Alternating, numeric columns |        65,536 | 61.559 ms | 27.058 ms |          56.0% |
| Clustered, mixed fields      |         4,096 |  2.850 ms |  0.928 ms |          67.4% |
| Clustered, mixed fields      |        65,536 | 45.393 ms | 14.605 ms |          67.8% |
| Scattered, mixed fields      |         4,096 |  3.287 ms |  2.780 ms |          15.4% |
| Scattered, mixed fields      |        65,536 | 52.352 ms | 41.025 ms |          21.6% |

The alternating fixture deletes 50% of one segment's rows. Mixed fixtures first
merge 1024-row sources and then delete 25%, contiguously or every fourth row.
They include missing/multi-value numeric fields and indexed/stored text, but do
not measure ANN throughput or position-heavy queries.

Memory evidence distinguishes scratch from process residency. At 100 million
physical rows and 90 million survivors, the old physical map required 760,000,000
bytes; the new shared-bitmap rank table requires 6,250,004 bytes (about 122 times
smaller). This is a layout calculation, not a process-RSS benchmark. The existing
visibility bitmap is common input-reader state. Chunk bitmap/rank scratch is
about 0.1875 bytes per physical chunk, plus the retained output labels/norms.
A regression compacts an 8192-document, position-bearing term in 64 KiB of posting
scratch where the previous decoded-entry admission alone required 512 KiB.

`/usr/bin/time -l` records both RSS and macOS peak memory footprint. Equal-work
runs use the same executables with `--test row_compaction` (one benchmark
invocation plus validation per fixture), three times each in alternating order.
These figures include fixture construction, source readers and RAM output; they
are not isolated compactor heap measurements.

| Measurement               | Before median (range), MiB | After median (range), MiB |
| ------------------------- | -------------------------: | ------------------------: |
| Equal-work peak RSS       |     135.47 (130.17–138.08) |    141.75 (136.53–142.02) |
| Equal-work peak footprint |        54.94 (48.67–60.17) |       48.61 (45.00–51.53) |
| Criterion peak RSS        |     190.45 (188.31–203.81) |    199.64 (199.58–201.33) |
| Criterion peak footprint  |        50.98 (49.55–51.52) |       51.14 (50.52–51.86) |

Whole-process RSS did not improve: the equal-work median is approximately 4.6%
higher, while peak footprint is approximately 11.5% lower with overlapping run
ranges. Criterion also performs different iteration counts as throughput changes.
The evidence supports substantially smaller mapping/term scratch and lower CPU
cost; it does not establish a general process-RSS reduction.

Retaining source-local column boundaries has a small size cost in these fixtures.
At 65,536 physical rows, captured `.fast` sizes are:

| Fixture                      | Before bytes | After bytes | Change |
| ---------------------------- | -----------: | ----------: | -----: |
| Alternating, numeric columns |    2,477,059 |   2,477,374 | +0.01% |
| Clustered, mixed fields      |    3,712,690 |   3,719,999 | +0.20% |
| Scattered, mixed fields      |    3,717,152 |   3,835,464 | +3.18% |

Complete files can differ because legal block boundaries differ. Regressions
compare copied fast/posting/position payload bytes, all five complete ANN output
formats, source-local text/missing/multi-value semantics, and query/position
results. A heavy scattered-deletion test also verifies that 64 sparse posting
blocks coalesce into one output block. Failure coverage includes malformed
posting lengths/footer overflow, directory budgets, injected position-write
failure, cancellation during copies, concurrent visibility changes and ownership
drain. The review caught and fixed the odd-sub-block ScaNN padding copy error
before release.

Validation: `RUST_TEST_THREADS=1 python3 scripts/check_search.py check` passed
(1588 tests, 26 existing ignores, strict Clippy, ownership checks, native build
without sync). An earlier parallel broker run hit loopback port collisions
(`Address already in use`); the serial rerun passed. Native async compaction tests
passed (24, one existing ignored benchmark). `summa-wasm/build.sh`, `npm ci`
and `npm test -- --run` passed (20 tests). `full` was not run: this change does
not alter lifecycle/RPC protocols. The existing lifecycle/concurrency regressions
are included in the passing harness.

Raw logs, captures and summary are in `.context/compaction-copy-measurements/`;
compiler/executable/source hashes are in `.context/compaction-copy-build-metadata.json`.
Harness evidence is in `.context/search-harness/20260911T180857.540199Z-check/`.

### Remaining costs and review findings

- Final merge plus compaction remains two passes. Fusing them is separate work.
- Mixed fast chunks that overflow the bounded encoded cache are encoded twice
  because their directory precedes payload. More source-local blocks can enlarge
  files/directories; a future mixed-block packing policy needs size and query
  measurements as well as compaction time.
- Posting/position directories still scale with encoded block count and must fit
  admission. Legacy positions retain an explicitly bounded whole-list fallback.
  Current external postings remain external after shrinking; inline conversion
  is not part of the streaming path.
- BMP retains its forward record map and identity permutation and rebuilds block
  membership/pruning metadata. MaxScore sparse still visits and remaps blocks
  while preserving encoded weight bits. Row-statistic/norm output still scans
  surviving rows. These paths are not blanket raw-copy operations.
- The new posting payload reader issues block-sized lazy reads. Remote backends
  may need bounded read-ahead/coalescing; remote latency, cold mmap residency,
  ANN throughput, and x86 were not measured here. The local fixture results must
  not be generalized to those workloads or used to change defaults.

## Content-hash upserts (2026-09-13)

The schema's `content_hash` marker enables exact comparison against a committed
live row. The primary-key fast dictionary and column supply a compact inverse
row map; hashes remain stored-only. Equal hashes queue no indexing work and
stage no tombstones. Pending insertions still require commit; pending deletions
still require their replacement. The tests compare every persisted byte for
no-ops and cover abort, failed/cancelled reads, failed commit/retry, old readers,
reopen, merged dictionaries, compaction, JSON conversion and RPC accounting.

The initial measurement exposed a pre-existing bounded-decoder allocation bug:
a 16 KiB result retained 268,435,456 bytes of capacity. This exceeded store-cache
admission and caused repeated decompression. The shared limited Zstd decoder now
uses the frame's decoded-size hint (bounded by the caller's limit), with a 512 KiB
initial capacity and bounded streaming fallback for unknown sizes. Small blocks
retain their actual size and enter the existing byte-bounded cache. Tests verify
both dictionary modes, unknown-size frames and oversized-output rejection.
This also restores ordinary document-store cache admission under its existing
budgets; no compression bytes or cache-budget defaults changed.

### Fixture and measurements

Same executable/compiler/flags/host, schema marker off versus on: Rust 1.98.1,
release profile, Apple M4, macOS 15.6.1, RAM directory, one indexing worker,
NoMergePolicy, 2,000 distinct keys with 3,840-byte stored/indexed text bodies,
and three complete unchanged-upsert passes. Each phase includes commit; document
cloning is timed equally. Three alternating runs per mode use separate processes.
Timing begins after fixture document generation and writer initialization.
Correctness/lookup-memory checks are outside timing; process peak RSS includes
setup, validation and runtime overhead.

Medians (minimum–maximum) across three runs:

| Measurement                           |             Marker off |           Marker on |
| ------------------------------------- | ---------------------: | ------------------: |
| Initial insert + commit, ms           |    42.81 (41.25–58.46) | 41.76 (41.48–41.82) |
| 6,000 unchanged upserts + commits, ms | 125.38 (124.09–131.22) |    3.63 (3.56–3.63) |
| Process peak RSS, MiB                 | 108.55 (107.23–108.92) | 74.23 (67.33–77.33) |
| Physical / live rows afterward        |          8,000 / 2,000 |       2,000 / 2,000 |
| Inverse-map allocation, bytes         |                      0 |               8,000 |

Unchanged-upsert time improved about 34.5× on this fixture. Initial insertion
showed no apparent slowdown within the observed variation. Measurements waited
for concurrent builds to finish; this is a desktop RAM microbenchmark, not a
production latency or cross-architecture result. The earlier allocation-bug runs
had background macOS activity and are diagnostic only.

Reproduce the fixture (each mode in a separate process for RSS measurement):

```sh
SUMMA_CONTENT_HASH_BENCH_ENABLED=false cargo test -p summa-core --release content_hash_performance_fixture --lib -- --ignored --nocapture
SUMMA_CONTENT_HASH_BENCH_ENABLED=true cargo test -p summa-core --release content_hash_performance_fixture --lib -- --ignored --nocapture
```

### Validation and remaining costs

`CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0
RUST_TEST_THREADS=1 python3 scripts/check_search.py full` passed: 1,621 principal
suite tests, four real-server broker tests, strict Clippy, both compile boundaries
and API docs. The WASM build and all 21 JS tests passed. Portable transaction tests
also passed independently. Documentation links passed. Earlier builds exhausted
disk space; deleting this workspace's disposable debug artifacts and disabling
debug information/incremental compilation resolved that environmental failure.

- Native Rust upsert is now async to perform stored-field I/O safely. RPC/client
  wire formats are unchanged. Metadata format 8 protects the schema setting;
  versions 6 and 7 migrate without segment payload rewrites.
- The inverse map uses four bytes per key, capped at 64 MiB per segment.
  Larger dictionaries warn and use a constant-scratch column scan per matching
  key. This fallback preserves semantics but is not constant-time row resolution.
  A persisted evictable inverse column would need a separate format design and
  large-corpus measurement; it is not introduced here.
- Native hash readers share a 32 MiB decoded-store cache. Portable readers retain
  no decoded blocks. Large uncached documents still require whole-block
  decompression to read one stored hash. Store directories and overlapping old/
  new inverse maps add residency during refresh; no corpus hash cache is retained.
- Changed-hash upserts pay the fingerprint read followed by normal replacement.
  The mixed workload below measures this cost for a small RAM corpus. Neither
  fixture establishes cold-storage latency, ANN ingestion throughput,
  tail latency, huge-segment fallback cost, or x86 behavior. No architecture-
  sensitive, merge, indexing, or search defaults were changed.

Raw runs, executable/source hashes and machine metadata are in
`.context/content-hash-performance/`; the initial allocation-bug measurements are
preserved in its `pre-allocation-fix/` subdirectory. Full harness evidence is in
`.context/search-harness/20260913T051535.864903Z-full/`.

### One percent unchanged rows

The parameterized fixture also measures 1% unchanged and 99% changed upserts.
All primary keys already exist: this measures the hash-read cost before normal
replacement, not a workload dominated by unseen keys. Every hundredth row keeps
its initial hash and body; the other 1,980 rows update both on each of three
passes. Bodies remain 3,840 bytes. Thus 6,000 upserts include exactly 60 no-ops
when enabled. Document generation is outside timing; cloning, ingestion, and
each commit are timed. Process peak RSS includes fixture generation and
post-timing validation.

Same Apple M4, macOS 15.6.1, Rust 1.98.1, release profile, RAM directory, one
indexing worker, and NoMergePolicy as above. Both modes use the same executable.
Seven runs per mode alternate OFF/ON and ON/OFF pair order, each in a separate
process. Measurements ran after compiler and test activity finished; ordinary
desktop activity remained. Medians (minimum–maximum):

| Measurement                       |             Marker off |              Marker on |
| --------------------------------- | ---------------------: | ---------------------: |
| Initial insert + commit, ms       |    43.27 (42.48–55.47) |    43.37 (42.14–45.48) |
| 6,000 mixed upserts + commits, ms | 129.57 (127.12–139.77) | 130.79 (127.30–132.66) |
| Process peak RSS, MiB             | 115.28 (110.94–119.12) | 138.44 (135.19–147.34) |
| Physical / live rows afterward    |          8,000 / 2,000 |          7,940 / 2,000 |
| Inverse-map allocation, bytes     |                      0 |                 31,760 |

The enabled median was 0.94% slower, within the overlapping timing variation;
this does not establish a statistically significant throughput difference.
Exactly 60 replacements were avoided in every enabled run, with the same live
row count. Median process peak RSS increased by 23.16 MiB. The inverse map is
only 31,760 bytes here; stored-block caching and allocator/runtime retention
also contribute to process memory. RSS alone does not attribute that increase
to individual components. No cache or indexing defaults changed.

Reproduce (run separately per mode to measure RSS):

```sh
SUMMA_CONTENT_HASH_BENCH_DUPLICATE_PERCENT=1 SUMMA_CONTENT_HASH_BENCH_ENABLED=false cargo test -p summa-core --release content_hash_performance_fixture --lib -- --ignored --nocapture
SUMMA_CONTENT_HASH_BENCH_DUPLICATE_PERCENT=1 SUMMA_CONTENT_HASH_BENCH_ENABLED=true cargo test -p summa-core --release content_hash_performance_fixture --lib -- --ignored --nocapture
```

Omitting the percentage retains the all-unchanged workload. The new fixture
generates documents for each pass outside timing, so its RSS is not directly
comparable with the earlier all-unchanged fixture's RSS. This remains a small
RAM measurement without merges, cold I/O, ANN, or tail-latency coverage.

Raw runs, summaries, executable/source hashes, host configuration, process
snapshots, and runner scripts are in `.context/content-hash-performance-1pct/`.
`CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0
RUST_TEST_THREADS=1 python3 scripts/check_search.py check` passed after the
fixture change; evidence is in
`.context/search-harness/20260913T053616.945219Z-check/`. This follow-up changes
only the ignored benchmark and documentation; it does not change runtime code.

## Shared-resource logging — 2026-09-15

The search CPU pool, store cache, and BMP I/O gate registries keep weak
references. Once their last owner drops, reopening an index recreates these
resources and previously repeated all three process-wide INFO announcements.
Each resource kind now announces its first successful creation at INFO;
subsequent creations, including different settings, are available at DEBUG
(`RUST_LOG=info,summa_core::index=debug`). Reusing a live resource stays silent.
The logging state is one `OnceLock<()>` per resource kind, with no retained
resource ownership or growing configuration history. No storage formats,
execution policy, or resource lifetime changed.

`summa-core/tests/resource_logging.rs` exercises three open/drop cycles with
two overlapping index handles per cycle. Before the fix, each resource emitted
three INFO records; afterward it emits one INFO and two DEBUG records. The
regression passes with default features and native without sync. This is log
count evidence, not a latency or memory benchmark; no performance improvement
is claimed. The change is native-only, so no WASM build or lifecycle/RPC `full`
run is required.

`CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0
python3 scripts/check_search.py check` passed, including focused Clippy,
core/server/broker/tool tests with metrics, and the native-without-sync compile
boundary. Evidence: `.context/search-harness/20260915T104311.838948Z-check/`.
`uv run scripts/check_docs.py` also passed; direct system Python lacked
`markdown_it`, so the documented uv environment was used.

## Latest staged primary-key mutations — 2026-09-15

Upsert and delete now operate on the latest accepted version of an ID, including
queued documents, active builders, and flushed-but-unpublished segments. An equal
staged content hash is a no-op without stored-block I/O. Different IDs remain
independent. Ordinary add retains duplicate rejection. Accepted mutations do not
force a commit, and rejected queue/budget admission preserves the previous row.

The PK owner holds one row handle and optional exact hash per latest staged key.
A row handle serializes attachment to a builder row against cancellation. A
cancelled queued row can skip indexing; an encoded row is recorded in the owning
segment's cancellation list. The existing owned commit transaction materializes
one visibility bitmap at a time and publishes these masks with committed-row
deletions. No persisted or wire format changes. No replacement writer, hash
index, or corpus payload reconstruction is introduced.

Pending PK metadata is bounded at 64 MiB, with conservative accounting for hashes,
keys, handles, retained hash-table capacity, projected table growth, and sixteen
bytes per superseded row. This also bounds repeated replacements of one ID.
Commit/abort clears the pending sequence; table capacity may remain cached and
continues to count toward the budget. The budget excludes caller documents,
bounded channel payloads, builder memory, and immutable segment metadata. Existing
100,000-key / 8 MiB pending-deletion limits still apply. Exceeding a limit is an
explicit error requiring commit, not an implicit publication or dropped row.
Encoded superseded rows remain physical until compaction, and retain the existing
one-bit-per-physical-row visibility residency after publication.

### Diagnostic performance comparison

Compared base `d6b4109d` with the staged-mutation implementation using the same
fixture, Apple M4, macOS 15.6.1, Rust 1.98.1, and unoptimized test profile with
`CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0`.
The fixture uses RAM storage, one indexing worker, NoMergePolicy, 2,000 IDs,
3,840-byte stored/indexed bodies, and three complete upsert/commit passes. The
content-hash marker is enabled in both binaries. Document generation is outside
timing; cloning and commit are included. Each run performs 2,000 initial inserts
followed by exactly 6,000 upserts.

Three runs per binary and workload alternated before/after pair order. No builds
or other workspace benchmarks overlapped the retained paired runs; ordinary
macOS applications, including a busy VPN process, remained active. Medians
(minimum–maximum), with RSS measured in separate processes including setup and
validation:

| Workload / measurement                |                       Before |                        After |
| ------------------------------------- | ---------------------------: | ---------------------------: |
| 1% unchanged: insert + commit, ms     |       478.58 (478.56–494.95) |       484.54 (484.31–498.37) |
| 1% unchanged: upserts + commits, ms   | 1,448.52 (1,427.15–1,468.17) | 1,452.12 (1,447.46–1,455.67) |
| 1% unchanged: peak RSS, MiB           |       156.84 (154.44–157.69) |       157.97 (137.52–158.23) |
| 100% unchanged: insert + commit, ms   |       481.67 (480.11–484.68) |       484.83 (483.77–486.26) |
| 100% unchanged: upserts + commits, ms |          17.43 (17.28–17.76) |          17.26 (17.10–17.39) |
| 100% unchanged: peak RSS, MiB         |        105.70 (94.78–117.34) |       114.97 (111.22–117.16) |

The mixed-workload median differed by +0.25%, within the overlapping timing
variation. Both binaries produced 7,940 physical / 2,000 live rows at 1% unchanged,
and 2,000 physical / 2,000 live rows at 100% unchanged. RSS ranges overlap and do
not isolate individual allocations. This diagnostic does not establish release
throughput, cold-I/O behavior, x86 performance, or tail latency. It exercises the
existing committed-upsert path's overhead; correctness tests separately exercise
repeated staged mutations. Existing worker, merge, and search tuning defaults
were not changed.

Raw paired runs, process snapshots, executable hashes, environment, and scripts
are in `.context/staged-mutations/`. Initial non-paired measurements overlapped
other workspace benchmarks and are excluded from conclusions. Reproduce each
workload with the ignored `content_hash_performance_fixture`, setting
`SUMMA_CONTENT_HASH_BENCH_ENABLED=true` and
`SUMMA_CONTENT_HASH_BENCH_DUPLICATE_PERCENT=1` or `100`, with the same profile on
both revisions.

### Validation

The behavior regression first failed on the old pending-insertion rejection.
Coverage now includes same-hash no-ops without stored reads, reverting to the old
committed hash after a staged change, different IDs with equal hashes, queued and
flushed rows, queue-full and budget rejection, indexed-only fields, chunk removal,
abort, commit failure/retry, cancelled owned commits, portable cancellation after
rename, old readers, reopen, merge/compaction, and RPC accepted counts.

The final `python3 scripts/check_search.py full` passed with debug information and
incremental compilation disabled and `RUST_TEST_THREADS=1`: strict Clippy, native
and portable compile boundaries, core/server/broker/tool tests, API docs, and all
four real-server broker tests. Evidence is in
`.context/search-harness/20260915T175450.058137Z-full/`. The WASM release build and
all 22 JavaScript tests passed; four portable fault-injection tests also passed.
Initial WASM and real-server failures were stale assertions requiring staged
mutation rejection; they were updated to verify acceptance and final visibility.
Documentation links and formatting passed.

## Wikipedia search benchmark (2026-09-13)

The full 5,032,104-document corpus and all 962 official queries were measured on
a dedicated GCloud n2-highmem-8 (Intel Xeon, 64 GiB). Search was pinned to one CPU;
both Rust engines used rustc 1.98.1, native CPU flags and release LTO. Main
before/after samples reuse identical Summa index bytes, with 60-second warmup
and ten repetitions. Latencies below are geometric means of per-query medians,
in microseconds, including parsing and pipe transport.

| Command       | Summa before µs | Summa after µs | Tantivy µs | Lucene µs | Before/after speedup |
| ------------- | --------------: | -------------: | ---------: | --------: | -------------------: |
| TOP_10        |           2,110 |          1,485 |        552 |       558 |                1.42× |
| TOP_100       |           2,577 |          1,772 |        720 |       774 |                1.45× |
| TOP_1000      |           3,033 |          2,092 |        933 |     1,151 |                1.45× |
| TOP_100_COUNT |           5,196 |          3,097 |        862 |     1,382 |                1.68× |
| COUNT         |           5,051 |            987 |        426 |       485 |                5.12× |

All 962 exact counts agree with Tantivy 0.26 and Lucene 10.4.0. Pruned Summa
rankings agree with exhaustive scoring in score bits and ordered IDs at top-10,
top-100 and top-1000. `COUNT` is exact; `--exhaustive` also disables ranking
pruning. Summa remains slower than both references overall. Canonical score
arithmetic, saturated TF bounds, keyword boundaries and phrase termination
fixes are correctness requirements, not optional speed tradeoffs.

The implementation streams exhaustive text collection, batches score-free pure
unions, uses guarded exact term-frequency metadata, aligns selective required
clauses before phrase verification, and reuses existing posting decoders and
skip structures. Rejected experiments are recorded: batching conjunctions and
exclusions regressed them, and ordinary term streaming beat a new pruning route
on the complete 714-term supplement. No production codec/cache default changes.

See [the full results, remaining costs and evidence](search-benchmark-results.md),
[the reproducible protocol](search-benchmark-game.md), and
[the small results bundle](benchmark-results/search-game-2026-09-13/README.md).
Exhaustive ranking costs 1.96×, 1.72× and 1.52× the corresponding pruned runs.
The 1,024-block cache option improves official COUNT by 19% for roughly 5–7 MiB
more RSS. Existing packed/PFor codecs save about 10% of index bytes but regress
query latency, so rounded remains the default. A read-only position diagnostic
estimates another 10.7% of original whole-index bytes could be saved with
exact-width payloads; this remains an unimplemented format proposal.

Validation: the complete engineering check passes 1,630 tests (26 ignored),
formatting, Clippy and native-without-sync compilation; WASM passes all 20 tests;
the final x86 core suite passes 1,442 tests (17 ignored). All official and
supplemental exactness gates pass. Raw samples, source/binary hashes, profiles
and logs are preserved in the checksum-verified evidence archive.

The benchmark machine and boot disk were deleted after evidence verification.

## Score-bound follow-up — 2026-09-13

The [follow-up report](search-benchmark-ratio-results.md) records the new
same-index cloud comparison and separate 100,000-document ARM cross-check.
The implemented opt-in ratio extension addresses loose TF/length block bounds
without changing the canonical BM25 scorer or posting payload codecs. Compact
merges preserve the encoded extension and budget its metadata; defaults remain
unchanged. A single ratio-bearing text cursor now uses the existing block cursor
and heap without general window scratch. Standalone precomputed text results
reuse the existing ranked handoff, avoiding a second heap while retaining lazy
document-order traversal for Boolean composition.

The final local gate passes 1,640 tests, formatting, Clippy, native without sync,
and the WASM build plus 20 tests. ARM exact-ranking gates cover every official
query and all 714 supplemental terms on both the legacy and ratio-bearing
indexes. On the same legacy ARM index, the execution-only handoff improves
TOP_1000 by 1.075× overall and 1.200× for the 301 unions; TOP_10 and COUNT are
flat. The ratio-bearing single-term workload improves TOP_10 while TOP_1000
remains flat in the reversed-order repeat. These small-fixture results are not
full-corpus ARM claims.

On the full 5,032,104-document corpus, all 714 supplemental terms improve by
2.889× for TOP_10 and 1.648× for TOP_1000 on identical persisted index bytes.
TOP_10 improves for 713/714 terms and TOP_1000 for 705/714. The official 962-query
ranked commands improve by only 1.8–3.2%; exact-count commands remain flat, and
Summa still trails Tantivy by 2.12–3.37× across the five commands. Peak process
RSS increases by 0.38–1.27 MiB per official command. All 1,676 final x86 query
gates pass, including exact counts against Tantivy and ordered Summa IDs/score
bits against exhaustive scoring. These measurements use a new machine/index and
must not be multiplied into the original report's speedups.

The ratio arrays occupy 104,200,464 bytes (99.37 MiB), 1.92% of this index;
they remain mmap-backed and evictable. A separate same-binary cache comparison
raises capacity from 256 to 1,024 blocks: 1.36× faster supplemental TOP_10 and
4.37× faster supplemental exact COUNT, plus 1.11–1.19× across official commands,
for 5.18–5.73 MiB more peak RSS per official process. This is an optional budget
tradeoff, with its own controls and all exactness gates, not a default change.

Remaining measured priorities are stronger competitive impact bounds,
dictionary-cache residency, posting alignment and position decoding. Ratio
minima are still looser than a frontier of actual competitive frequency/norm
pairs. Any frontier extension must preserve configurable BM25/global statistics,
bounded resident metadata and compatible merge-copy behavior. No default or
unimplemented-format speed claim follows from this experiment. Full-corpus
latencies, memory, controls, rejected candidates and resource cleanup are
reported in the linked follow-up evidence.

### Bulk scoring, Lucene research and posting validation (2026-09-13)

See the [full measured report](search-benchmark-bulk-results.md),
[reproducible evidence](benchmark-results/bulk-scoring-2026-09-13/README.md) and
[pinned Lucene research](lucene-11-performance-research.md). Final corrected-build cloud timing and capture are complete; distinguish the
frozen prototypes from the retained source below.

**Retained implementation and invariants.** Complete Boolean unions use one
4096-document score window and membership bitset. Children contribute their
complete score in the original order; nested additions retain their parentheses.
Every matching document is still collected and counted exactly once in document
order. Term scoring gathers lengths and evaluates canonical BM25 over the owning
posting iterator's decoded runs, then scatters scores. Standalone terms with more
than one codec block use the same replacement-window path. Signed zero, boosts,
legacy length fallback and score bits remain unchanged. Position-aware collection,
unsupported query shapes and predicates keep their ordinary paths.

The posting iterator owns run advancement and TF-prefix/position-cursor accounting.
A cancelled callback leaves its run unconsumed; checks surround each window, run
and emitted hit. Scratch is a 16 KiB score array, 512-byte membership mask and two
128-float arrays, independent of hit count. No new scorer, format, query-answer
cache or index-specific branch is introduced. The x86 batch contains eight-lane
BM25 arithmetic; ARM uses four lanes. Separate profiles show work moving from
per-hit `doc`/`seek`/`score` dispatch into the batched accumulator and collector.

**Measured prototype result.** On the full 5,032,104-document corpus and all 962
official queries, frozen score-window v2 improves TOP100+COUNT
3076.521→2103.758 µs (1.462×), including 14112.282→4380.319 µs (3.222×) over all
301 unions. Its union COUNT control is flat. Summa still takes 2.338× Tantivy's
time overall for TOP100+COUNT. Small changes in unaffected ranked paths include
control drift and are not credited to batching. Matched process RSS is roughly
964 MiB versus Tantivy's 715 MiB; RSS includes resident mmap pages, not only heap.

The result includes regressions: `shih tzu`, 1,372 exact hits, changes
458.5→571.5 µs while COUNT remains flat. Fixed score-window initialization for
sparse matches is a plausible cause to profile. Across all union hit-count
quintiles, geomean gains are 1.69×, 2.32×, 3.01×, 4.36× and 6.66×. No query is
excluded and no special case is introduced for the slow query.

**Correctness follow-up.** An unknown posting codec was accepted by deserialization
and became an empty iterator. A failing regression was recorded before adding
structural validation. The structures owner now validates directory/header counts,
order and extents, codec widths, PFor exception tables and position cursors before
exposing an immutable posting view. Compaction reuses the same block validator.
Serialization and compatible merge payload bytes remain unchanged. Public sync
and async search regressions return an error. Validation uses constant scratch
and O(blocks + exceptions) work; it does not authenticate arbitrary in-range bit
changes or prove scoring bounds against original documents.

The isolated full-corpus validation comparison adds 6–9% overall:
TOP10 1407.595→1537.182; TOP1000 2043.366→2174.353;
TOP100+COUNT 2050.039→2178.410; COUNT 1007.733→1090.950 µs. The fix remains enabled.
Future validation reuse must be tied to immutable reader generations and byte
ranges, bound its metadata and preserve first-read and cache-miss errors.

**Rejected phrase prototype.** A competitive bound before position decoding was
correct but did not improve the overall workload. In the final alternating paired
repeat, phrase TOP10 improves 1103.082→977.068 µs (1.129×), but union TOP10 slows
1059.883→1178.364 µs (0.899×); overall TOP10 is flat and TOP1000 slightly slower.
Collector/trait/phrase changes were removed, preserving standalone term batching
and posting validation. The subsequent removal control did not recover the union
slowdown: the retained build is 4.8% slower on official TOP10 than the correct
phrase prototype, but 3.2% faster on supplemental TOP100+COUNT. The cross-workload
benefit remains inconclusive, and the union regression cannot be assigned solely
to the phrase hook. Prototype sources and tests remain in the evidence archive.
Any future phrase bound must use first-term frequency: under slop, multiple starts
can share a later position, so the minimum frequency across terms is unsafe.
Ties require strict bound comparison; exhaustive/composite collection must retain
complete membership, and unsupported wrappers must not forward pruning hints.

**Final retained source validation.** The full check passes 1,649 native tests
(26 ignored, 25 suites), formatting, Clippy and native without sync; the WASM
build and all 20 tests pass. All 1,676 final ARM gates agree with exact counts and
exhaustive ordered IDs/score bits. A clean paired ARM repeat versus v2 gives
23.051→20.547 µs (1.122×) on TOP100+COUNT for all 714 terms, with COUNT and TOP10
flat. Official commands are within about 1.1%, including their count control.
The first final-ARM pass overlapped a brief ZIP job and was preserved but excluded;
both complete workloads were repeated afterward. A source-archive timestamp
hazard was caught and repaired before cloud v1/v2 timing; distinct rebuilt
binaries and batch symbols were verified. No stale-binary samples are included.

**Remaining research.** The [BM25 impact-envelope proposal](posting-codecs.md#proposed-bm25-impact-envelope-research-not-implemented)
retains 3.600 points/block on average across 780,298 probed blocks, versus 8.506
for the full Pareto frontier. Its ideal final-TOP10 threshold rejects 99.138% of
blocks versus 69.009% for the ratio proxy. These are oracle-bound headroom numbers,
not actual skipping, latency, or index-wide storage estimates. The format remains
unimplemented. Dictionary decode/residency, posting alignment and position decoding
remain priorities; sparse performance has not been measured in this pass.

Lucene's 256-entry norm table also motivated an offline exact-factor lookup probe.
On ARM contiguous score arrays, its runtime-sized lookup/gather prototype was
slower than canonical vectorized division, despite identical score bits. It is
not in production and does not establish the result on x86 or a complete query.

Final cloud results for the retained build are TOP10 1589.342 µs (2.840×T),
TOP1000 2187.585 (2.298×T), TOP100+COUNT 2211.890 (2.505×T), COUNT1082.535
(2.538×T). All 1,676 final x86 exact-count/ordered-score gates pass. The final
archive has 228 hash-verified members; H/T index manifests match the initial
capture exactly. Ranked RSS is within 0.4 MiB of v2 in this same phase. Full
supplementary, family and variant tables remain in the linked report. The
ranking gap remains open; this pass does not establish leadership.

## Bounded posting validation reuse (2026-09-13)

The [validation reuse report](search-benchmark-validation-results.md) records a
new opt-in, byte-bounded table owned by each immutable posting-file reader. It
caches only successful range/footer validation, never payloads or query answers.
New file owners and lazy callbacks still validate; errors and cancellation remain
observable. The default is zero, maximum 64 MiB per segment, and heap accounting
includes the allocation. All public deserializers retain unconditional checks.

Full harness and WASM pass, with 1,656 native tests including two supplemental
mmap/admission tests, plus all 1,676 ARM exact-count/score-bit gates under both
budgets. Official ARM commands improve 1–3%; the supplemental terms instead slow
roughly 1–3%. Single-term COUNT varies even though it bypasses external postings.
Full-corpus x86 timing, RSS and profiles are complete below; the machine remains
active for the next dictionary experiment.

The validation-cache full-corpus follow-up is complete: unchanged index bytes,
all 1,676 gates at both budgets, and 5–10% official latency gains with 256 KiB per
segment. TOP100+exact count improves 1,943.889 → 1,844.662 µs, compared with
Tantivy's 810.085 µs. Supplemental term COUNT is flat; ARM terms regress 1–3%,
so the default stays disabled. Peak RSS increases at most 0.50 MiB versus the
new disabled build. [Full tables and evidence](search-benchmark-validation-results.md)
separate the measured frozen build from subsequent dictionary work.

## Dictionary block and allocation experiments (2026-09-13)

The [dictionary report](search-benchmark-dictionary-results.md) records opt-in
block targets and an actual decompressed-byte cache cap, preserving STB5/Zstd and
all posting bytes. The initial full-corpus 4 KiB layout improves matched-cap
TOP100+exact count 1,894.929 → 1,680.981 µs, versus Tantivy's 808.604 µs. Small ARM
changes and slower bounded prefix scans prevent a default change. Bounded
prefetch and failed-writer publication bugs are fixed with regressions. A
subsequent capacity regression proves that a tiny bounded Zstd decode reserves
64 MiB; a library-frame-size allocation fix is being measured independently.
The final writer-fix run confirms the layout gain: matched 16/4 KiB
TOP100+exact count is 1,961.198 / 1,749.443 µs, versus Tantivy 852.032 µs.
All 136 executable-archive members and unchanged index manifests are verified;
the report links a compact reproducible bundle. Allocation and conjunction
cloud runs remain separate and in progress, with the machine still active.

## Adaptive conjunction counting (2026-09-13, in progress)

Pure conjunctions can now intersect bounded 4,096-document membership windows.
Dense survivor masks use word intersections; sparse masks seek only surviving
candidates. The path preserves next-document score/position state, uses 1 KiB
mask scratch and reuses existing posting decoders. Optional/excluded clauses keep
their existing execution. The [design](search-benchmark-game.md#proposed-adaptive-conjunction-membership-windows)
records eligibility and the cost model. The verifier now checks score-free COUNT
independently against exhaustive VERIFY and Tantivy, so ranked correctness cannot
mask a count-path failure.

The focused harness passes 1,678 native tests, native-without-sync and portable
compilation, plus all 20 WASM tests. ARM gates pass all 1,676 queries for both
before and after builds. On the 100k fixture, conjunction COUNT improves
22.326 → 21.262 µs (5%); all official COUNT improves 23.663 → 23.331 µs and ranked
commands are flat. Full-corpus x86 measurements and retention decision are pending.
An additional deletion/multi-value conjunction assertion passes separately after
the measured source freeze; it is not part of that frozen overlay.

The [whole-vocabulary impact audit](posting-codecs.md#whole-vocabulary-storage-audit-and-narrower-first-experiment)
now separates storage costs from the earlier query-term bound-tightness sample.
On the 100k fixture, L0-only eight-point records for multi-block lists estimate
0.72% index overhead using existing vints or 1.70% with fixed f32 coordinates,
excluding any new footer/alignment. Applying coordinates to every list instead
estimates 7.10%. The warm ARM metadata kernel favors coordinates (4.539 versus
16.517 ns/block), but this is not a query-speed measurement. No impact format
has been implemented. All-vocabulary full-corpus costs and the paired public
workload remain necessary before selecting the representation.

## Systemic execution gap and rejected first conjunction policy (2026-09-14)

The captured first window candidate does **not** pass its performance gate.
Full-corpus conjunction COUNT changes 750.300 → 766.975 µs (2.2% slower), versus
Tantivy 285.112 µs. All official COUNT is flat (800.014 → 800.584 µs); TOP10 is
also flat (1,195.120 → 1,195.209 µs). TOP100+exact count is 1,682.446 → 1,661.517 µs
versus Tantivy 820.219 µs. Summa still has a systemic roughly twofold gap.
The 714-term control remains about 4.6× slower for TOP10. Full raw data and
unchanged manifests are in the [conjunction bundle](benchmark-results/conjunction-2026-09-13/README.md).
All 55 executable-archive members are verified (17,239,936 bytes, SHA-256
`4304750b4f13aeecd1972b6bd73bd7ad6e7501ca1264d813f5b323411043ff35`).

Complete per-query analysis separates the failure by lead density: the 43
conjunctions with a lead document frequency below 1,000 regress 20% geometrically;
the 113 between 1,000 and 10,000 regress 15%. Fifteen dense cases above 80,000
but below 500,000 improve about 2.1×; the two above 500,000 improve about 4.9×.
These bins describe the evidence, not execution thresholds. The next candidate
uses the existing candidate-versus-bitmap-word cost model for admission before
window setup; its performance remains unmeasured. The stronger verifier checks
Summa COUNT separately against exhaustive VERIFY and Tantivy for every query.

The next performance priority is paired full-workload profiling of both engines,
including pruned top-k, exhaustive scoring/counting, and score-free counting.
Parser/dictionary, posting traversal, position checks, BM25, collection, and
hardware-counter costs must explain the total gap. Small percentage gains do
not establish progress to engine leadership; no leadership claim is supported.

## Posting merge admission correctness (2026-09-14)

A format-path audit reproduced a corrupt single-source term being copied into
a successfully published merged segment. It also reproduced document rebasing
arithmetic overflow. Both concatenation APIs now check remapped ranges and sums;
streaming input uses the existing structural validator before any output,
including the zero-offset copy shortcut. No posting decode/re-encode is added.
All-codecs byte comparisons, malformed later-source/no-write checks, and the
public merge metadata-publication regression pass. Full eight-phase validation
passes 1,683 native tests (26 ignored), portable/native-without-sync compilation,
API docs and real-server broker tests; WASM build and all 20 tests pass.

Five alternating ARM process pairs over all 338,367 external lists show the cost:
a 55.05 MB single-source copy changes 8.519 → 11.779 ms, and a two-source merge
producing 91.18 MB changes 64.648 → 68.711 ms (6.3% slower). Output checksums agree
throughout; unit regressions compare actual bytes across all three codecs.
Median process peak RSS is 88.05 → 90.64 MiB. This measures the canonical posting
merge with a reused output buffer, excluding lifecycle/store work; it is a
correctness cost, not a query optimization. Frozen merge-admission-v2 excludes
a temporary helper that contaminated the initial full-check attempt; the final
clean harness passed. Subsequent conjunction-density work is a separate change.

### Systemic gap: paired complete-workload profiles (September 14)

The latest completed full-corpus comparison still leaves Summa slower than
Tantivy by 2.36× for TOP_10, 1.96× for TOP_1000, 2.03× for TOP_100_COUNT, and
1.94× for COUNT. The prior small improvements do not establish competitiveness.
The next acceptance criterion is closing these workload gaps with shared search
algorithms, while retaining independent exact-count and exhaustive-ranking gates.

Both engines were profiled sequentially on the same CPU and immutable indexes,
using all 962 official queries and all 714 supplemental terms. Each scope ran
complete warm-up passes for at least five seconds and sampled complete workload
passes for at least twenty seconds. The machine exposes no hardware performance
counters; results use `cpu-clock:u` at 997 Hz and software task-clock. Instrumented
throughput is diagnostic, not a replacement latency benchmark. Stack unwinding
is incomplete, so the following attribution uses self samples only.

| Summa self CPU           | Official TOP_10 | Official COUNT | Official TOP_100_COUNT |
| ------------------------ | --------------: | -------------: | ---------------------: |
| Posting iterator seek    |          24.84% |         21.02% |                 15.52% |
| Position stream read     |          15.90% |         17.92% |                  9.58% |
| Phrase position matching |           7.20% |          8.40% |                  4.24% |
| Term membership window   |               — |         21.17% |                      — |
| Term score accumulation  |               — |              — |                 26.38% |
| Top-k collect            |           0.43% |              — |                 11.50% |

For the 714 standalone terms, 44.93% of Summa CPU is deferred score computation
and 29.63% is the single-term executor. This has a different cause from exact
counting. Making the arithmetic cheaper cannot by itself close that much larger
ranking gap: tighter competitive block bounds must reduce how many documents
reach scoring. The researched impact envelope remains a proposal, not an
implemented performance claim. Likewise, eliminating even all seek overhead
would cap the official TOP_10 gain at about 1.33×. Traversal, position processing,
and competitive pruning need separate measured changes.

Seek disassembly confirms a document scan followed by frequency-prefix summation
on every movement, even without position reads. The candidate documented in
[posting codecs](posting-codecs.md#proposed-traversal-and-position-accounting-separation)
defers the frequency work until positions are requested and measures a
next-document probe plus binary lower bound. The public immutable position API
and all persisted bytes remain unchanged. This direction also matches the
separation of document movement and requested position offsets in
[Tantivy's posting reader](https://github.com/quickwit-oss/tantivy/blob/main/src/postings/segment_postings.rs).
The candidate has not yet passed the full-corpus performance gate.

Profile reports and reproducible source (local archive `benchmark-results/systemic-profile-2026-09-14/reports.zip`)
include query sets, software counters, disassembly, source overlays, locks,
hardware metadata and index manifests. The full 74,496,359-byte sample archive
was retrieved and all 83 manifest entries verified; its SHA-256 is recorded in
the [profile manifest](benchmark-results/systemic-profile-2026-09-14/manifest.json).

### Shared traversal and merged-position costs (September 14)

Two completed full-corpus passes improve the baseline but leave a systemic gap.
They use all 962 official queries, 714 supplemental terms, the same immutable
5,032,104-document indexes, Rust 1.98.1/native CPU/LTO, and one Cascade Lake core.
Each build passes independent COUNT versus exhaustive VERIFY versus Tantivy
counts, plus pruned versus exhaustive Summa top-k, on all 1,676 queries.
Numbers below are geometric means of per-query medians, in microseconds. The
rows are separate same-run comparisons; do not combine their absolute timings.

| Change / command                             |    Before |     After | Tantivy | After / Tantivy |
| -------------------------------------------- | --------: | --------: | ------: | --------------: |
| Deferred position accounting / TOP_10        | 1,179.717 | 1,010.058 | 502.314 |           2.01× |
| Deferred position accounting / TOP_100_COUNT | 1,652.055 | 1,484.176 | 810.999 |           1.83× |
| Deferred position accounting / COUNT         |   776.082 |   654.423 | 407.907 |           1.60× |
| Merged-position cursor / TOP_10              | 1,034.547 |   963.427 | 507.870 |           1.90× |
| Merged-position cursor / TOP_1000            | 1,493.736 | 1,448.519 | 869.066 |           1.67× |
| Merged-position cursor / TOP_100_COUNT       | 1,509.138 | 1,463.481 | 824.548 |           1.77× |
| Merged-position cursor / COUNT               |   669.388 |   632.656 | 410.692 |           1.54× |

The traversal pass removes frequency-prefix reduction from document movement
and uses a next-document probe plus binary lower bound for in-block seeks.
Its regression is real: an alternating 11-sample audit over all 962 queries
confirms union TOP_100_COUNT worsening 3,265.137 → 3,533.360 µs (8.2%). This
remains an unresolved finding; aggregate improvements do not waive it.

The position pass fixes a mismatch between the small fixture and real merged
indexes. All 714 sampled full-corpus position streams contain short interior
blocks preserved by ordinary copy merges; none of the original small fixture's
711 present streams did. A second ARM fixture built by the canonical writer
with a 16 MiB indexing budget has 706 such streams. The reader now checks its
cached logical range before searching the directory and uses a bounded forward
search for later misses. It still retains only one decoded block. All persisted
bytes are unchanged. The full-corpus phrase TOP_10 improves 1,061.700 → 926.963
µs, but remains 1.84× Tantivy. ARM results on both layouts are mostly flat.
A 1.9-second formatting command overlapped the merged ARM supplemental
TOP_1000 timing; that run's tiny differences are inconclusive.

Latest official TOP_10 peak RSS is 972.0 MiB for Summa versus 710.7 MiB for
Tantivy; the position change adds roughly 1 MiB to the measured process peak.
The 714-term TOP_10 still measures 203.243 versus 43.694 µs: **4.65× slower**.
These changes do not establish competitiveness. Software CPU samples now show
phrase candidate alignment at 12.05% of official TOP_10 and 13.77% of COUNT.
COUNT still spends 24.37% in term membership windows. The real-length BM25
scoring branch remains scalar in this frozen build, unlike its no-length
fallback. Batched length reads and a rarest-term phrase driver are separate
candidates undergoing measurement; neither is counted as a completed gain here.

The native/portable/WASM checks pass for both frozen passes. The position pass
also passes the full eight-phase lifecycle/RPC harness and 20 WASM tests.
Its new regressions cover sequential, forward, backward, spanning and failed
position reads, including byte comparisons. Raw samples, exact-count gates,
source snapshots, memory records, layout diagnostics and software profiles are
in the reproducible evidence bundle (local archive `benchmark-results/traversal-position-2026-09-14/results.zip`).
Both full cloud archives were retrieved and all 72/85 manifest entries verified;
the [manifest](benchmark-results/traversal-position-2026-09-14/manifest.json)
records their hashes and the omitted raw profiler/binary payloads.

### Batched lengths, phrase driver and corrected position admission (September 14)

Three further same-host full-corpus comparisons retain exact counts and exhaustive
Summa ranking gates for all 962 official queries plus 714 standalone terms.
Cross-engine counts agree; score bits and ordered IDs are checked between Summa
pruned and exhaustive execution, not between differing engine scoring models.
Each phase uses the same unchanged indexes, Rust 1.98.1/native CPU/LTO and one
Cascade Lake core. These are separate runs; do not combine their absolute times.

Batch length reads make the canonical real-length BM25 loop vectorize on x86
(`vdivps`) and ARM (`fdiv.4s`), without changing floating-point operation order.
Standalone-term TOP_10 improves 206.418 → 165.166 µs, but Tantivy is 43.031 µs.
Official TOP_10 improves 961.750 → 930.444 µs and TOP_100_COUNT 1,431.476 →
1,382.307 µs. The three-way alternating audit resolves the **earlier traversal**
union regression in this snapshot: 3,542.623 µs before traversal versus 3,204.137 µs
after batched lengths. This is distinct from the newer regression below.

A rarest-posting-list phrase driver keeps phrase arrays and position offsets in
query order while stopping alignment at the first rejected candidate. Full-corpus
phrase TOP_10 improves 928.596 → 796.953 µs, roughly 16%. All official TOP_10
improves 930.617 → 884.576 µs. These timings predate the position validation fix
and cannot be used as the corrected reader's performance.

The public phrase regression reproduced corrupt positions returning an empty
successful result. Position stream open now validates headers, physical extents
and logical directories before exposing infallible reads. Native and async search
reject the corrupt replacement, while an existing reader retains its immutable
valid generation. Successful proofs for document and position ranges share one
bounded per-segment cache under the existing setting, default zero. Lazy callbacks
always validate actual returned bytes; short reads and cancellation never publish
proofs. Encoded payloads and the BM25 formula are unchanged.

| Corrected build / official command | Preceding build, µs | Corrected build, µs | Tantivy, µs | Summa / Tantivy |
| ---------------------------------- | ------------------: | ------------------: | ----------: | --------------: |
| TOP_10                             |             900.090 |             888.590 |     500.381 |          1.776× |
| TOP_1000                           |           1,339.916 |           1,354.475 |     871.499 |          1.554× |
| TOP_100_COUNT                      |           1,293.741 |           1,333.589 |     801.917 |          1.663× |
| COUNT                              |             609.140 |             612.055 |     407.964 |          1.500× |

For the 714 standalone terms the corrected times are 164.559 / 603.562 /
705.030 / 10.404 µs for the same commands, versus Tantivy 44.364 / 408.937 /
243.036 / 6.594 µs. Competitive block pruning remains a large unresolved gap.

The newer union TOP_100_COUNT regression is confirmed by five fresh process
pairs, all 962 queries, and three alternating samples per query per pair. Pooled
union medians give 2,930.566 → 3,254.668 µs; the five per-run ratios range from
1.101 to 1.114 (median 1.112). All-workload time regresses about 2%. Repeating
processes therefore does not dismiss this as noise. Separate full-union CPU
profiles attribute about 29% to collection and 17% to its caller in both builds;
length gathering is about 7.5%, posting decoding about 7%. No causal explanation
for the regression has yet been established. The proposed bounded score collector
is a separately frozen candidate, not included in this table.

Corrected official TOP_10 peak RSS is 1,010.7 MiB versus 972.0 MiB for the preceding
build and 710.7 MiB for Tantivy. Separate Linux smaps measurements identify the
gap as mapped file residency: Summa 1,027,030 KiB file PSS and 5,808 KiB anonymous
PSS, versus Tantivy 724,758 and 776 KiB. These file pages remain evictable. The
strict admission pass touches more position block headers; validation correctness
is retained despite this cost. Neither RSS nor mmap residency is Rust `Pin`.

Both ARM fixtures (canonical and normally merged short position blocks) pass
all exact-count and exhaustive-ranking gates for both builds in every phase.
Batched lengths and the phrase driver improve official ARM commands by roughly
1–3%; the corrected admission build is essentially flat. These are 100k-document
fixtures, not substitutes for the full-corpus x86 measurements. Defaults remain
unchanged. Latest full eight-phase validation passes 1,697 non-doctest native
tests plus one doctest, including real-server broker tests and portable builds;
the WASM build and all 20 tests pass. The separate subsequent collection candidate
has its own validation and is not included in that count.

The 399-file reproducibility bundle (local archive `benchmark-results/execution-admission-2026-09-14/results.zip`)
contains source snapshots, raw samples, all gates, ARM evidence, compiler output,
software profiles, memory records and the five-process audit. All five executable
archives were retrieved and each external SHA-256 and every internal manifest
entry verified. The [manifest](benchmark-results/execution-admission-2026-09-14/manifest.json)
records exact archive hashes; the compact bundle excludes executable binaries
and raw perf samples. Summa has not met the performance objective.

The latency tables in this follow-up measure warm, single-client query execution.
Cold-cache tails, concurrent ingest/merge, and production concurrency were not
measured for these frozen changes. Software CPU sampling is diagnostic and does
not provide hardware cache-miss, branch-miss or memory-bandwidth counters. These
limits prevent extrapolating a benchmark improvement to general engine leadership.

### Bounded collection and audit controls (September 14)

The score-collection-v1 build uses the existing complete scored windows to send
64-document words to collectors that explicitly support batches. Top-k retains
its canonical float/doc-ID ordering and counts every matching bit, including
noncompetitive hits. It hoists heap representation/capacity checks and refreshes
the worst result only after replacement. CountCollector uses population counts.
Custom collector callback ordering, positions and deadline-bearing paths retain
their per-document behavior. No format, score formula, cache or default changes.

The full-corpus same-run official geometric means are:

| Command       | Before, µs | After, µs | Tantivy, µs | After / Tantivy |
| ------------- | ---------: | --------: | ----------: | --------------: |
| TOP_10        |    904.217 |   902.982 |     498.934 |          1.810× |
| TOP_1000      |  1,377.652 | 1,360.784 |     890.722 |          1.528× |
| TOP_100_COUNT |  1,329.599 | 1,250.694 |     810.271 |          1.544× |
| COUNT         |    618.511 |   609.336 |     409.398 |          1.488× |

Union TOP_100_COUNT improves 3,139.872 → 2,694.520 µs (14%), versus Tantivy
2,333.083 µs. AND TOP_10 remains 681.833 versus 330.145 µs, a systemic 2.07× gap.
The following required-clause driver candidate is not part of this frozen build.

The 714-term commands are 165.885 / 599.852 / 669.744 / 10.570 µs, versus Tantivy
43.334 / 409.454 / 238.467 / 6.746 µs. Standalone TOP_10 is 4.1% slower than the
159.280 µs control in this run, despite not using the new collector path. This
is recorded as an unresolved measurement, not claimed as a gain or explained
away. Complete-workload comparisons of the subsequent build remain necessary.

A three-build audit uses three fresh process sets and five alternating samples
per query per set, on all 962 official TOP_100_COUNT queries. The new collector
beats both the pre-admission and corrected-admission controls: union geometric
means are 2,947.837 / 2,955.379 / 2,551.189 µs. Its three ratios to pre-admission
range 0.865–0.872. However, the two unchanged control binaries are now essentially
flat rather than reproducing the prior two-build audit's 11% separation. That
sensitivity to execution context means the earlier separation is not an isolated
measurement of validation-cache cost. Both audits are retained; no specific
compiler, predictor or cache mechanism has been established. The collection
improvement itself is consistent across the sequential and alternating runs.

Both 100k ARM fixtures pass all 1,676 independent COUNT/VERIFY gates per build.
Union TOP_100_COUNT improves 56.796 → 52.349 µs on canonical positions and
57.301 → 53.372 µs on merged positions; all official commands improve about 3%
for TOP_100_COUNT and are otherwise flat. Official TOP_10 peak RSS is essentially
unchanged at 1,010.6 MiB, versus Tantivy 710.8 MiB. No new scratch allocation is
introduced. Separate full-workload software samples put the batch collector at
11.76% and its driver at 2.21% of TOP_100_COUNT CPU; the shared scoring loop now
accounts for 19.42%. These are self samples, not hardware performance counters.

The search harness passes 1,699 non-doctest native tests and one doctest, with
native-without-sync and portable builds. The WASM build and all 20 tests pass.
New tests compare batch and scalar results for signed zero, NaNs, infinities,
ties, partial/zero heaps, saturation, sparse masks and tuple callback ordering;
existing cancellation and exact public-query tests continue to pass. Full
lifecycle/RPC validation from the preceding admission build remains applicable;
this collection change does not alter those protocols.

The 144-file evidence bundle (local archive `benchmark-results/score-collection-2026-09-14/results.zip`)
and [manifest](benchmark-results/score-collection-2026-09-14/manifest.json) include
raw timings, all correctness gates, source, locks, ARM code generation, memory,
profiles and the three-build audit. Both full archives were retrieved and their
75/37 manifest entries verified. The goal of outperforming Tantivy remains unmet.

## Required-clause driver (September 14)

The public Boolean scorer still owns conjunction semantics. Its rarest required
clause now advances the candidate; each other required clause either agrees or
moves that driver forward immediately. The former loop inspected every clause's
current document to compute an initial maximum and sought the lead back to its
own candidate. The rewrite keeps clause/score order, optional clauses, exclusions,
two-phase confirmation and deadline checks. No format or new allocation is involved.

All 1,676 full-corpus queries passed exact COUNT and exhaustive-ranked VERIFY for
both builds. The official geometric means in microseconds (before / after /
Tantivy) are 899.137 / 839.351 / 499.157 for TOP_10, 1339.083 / 1286.436 /
869.968 for TOP_1000, 1240.157 / 1172.285 / 811.946 for TOP_100_COUNT, and
607.214 / 582.644 / 411.618 for COUNT. AND TOP_10 improves from 678.238 to
572.685, but Tantivy takes 330.276. Phrase and union gaps remain broad. Official
TOP_10 peak RSS stays approximately 1,011 MiB versus Tantivy's 711 MiB.

The 714 derived standalone terms take 156.640 / 599.095 / 667.479 / 10.676 µs
for the four commands; Tantivy takes 43.731 / 413.153 / 235.993 / 6.577.
Their before/after changes are not evidence of this conjunction optimization:
that query path is unaffected. Both ARM 100k fixtures pass all gates; AND
latency improves approximately 6–7%, while derived terms remain essentially flat.
The full corpus x86 run uses the same native LTO compiler, immutable indexes,
CPU 2, complete workloads, seven samples and ten-second warmups as its controls.
No builds, tests or profiles overlap timing.

The final check harness (`20260914T062232.379338Z-check`) passes all four phases,
1,700 non-doctest tests plus one doctest. Portable compilation and 20 WASM tests
pass. The initial Clippy failure in a test was fixed and retained in the evidence.
The frozen source preceded a formatting-only test line wrap; its exact addendum
is included, and release code is identical. The independent software profiles
still show distributed traversal, phrase-position and scoring costs, not an
isolated single slow function. Hardware counters were unavailable.

The corrected position-admission two-build audit and the later three-build
collector audit differ: the first showed an 11% union penalty; the latter's two
unchanged controls tie, with the collector consistently faster. This is evidence
of execution-context sensitivity, not proof that the cache caused the original
penalty or that its mechanism is solved. Both raw audits remain available.

Compact evidence (local archive `benchmark-results/required-driver-2026-09-14/results.zip`)
contains 122 files plus manifests, exact sources, raw samples, gates, ARM runs
and current profiles. ZIP SHA256:
`77e49a1926688988dc5701966ff0a06d33a613cac9b5b80b7b30762a4977dc5e`.
Full archive: 43,857,020 bytes, SHA256
`6983eca9abd3dd1206596834ed59a40ea8d9b92ebd0ea4f72f07230bf6748653`;
all 91 manifest entries and the compact archive were verified locally.

## Library SIMD codec (September 14)

**The codec swap does not solve the systemic gap.** Reusing `bitpacking` 0.9.3
(SSE3/NEON/scalar) reduces bytes, but the complete full-corpus x86 workload is
slower than the rebuilt rounded control. Default encoding remains rounded.
The [current comparison](search-benchmark-current.md) reports the latest paired
values; source snapshots, raw samples, correctness gates, builds, layout audits,
profiles and memory reports are in the
verified evidence bundle (local archive `benchmark-results/simd4x-2026-09-14/results.zip`).

V1 uses library SIMD for full 128-value document/TF/position blocks and exact
horizontal tails. Official rounded/SIMD geometric mean median microseconds are
841.106/912.257 (TOP_10), 1257.073/1328.757 (TOP_1000),
1226.787/1278.961 (TOP_100_COUNT), 578.001/627.534 (COUNT).
V2 keeps rounded encoding for every short block, while still accepting V1 bytes.
Its same-run rounded/SIMD values are 848.687/861.409, 1266.183/1319.779,
1214.732/1220.664, 585.263/611.340; Tantivy is 493.388, 875.670,
807.861, 410.399. All 962 official plus 714 supplementary queries pass exact
COUNT and exhaustive-ranked VERIFY across each reader/layout combination.
Separate old-binary/old-index and new-binary/old-index controls are retained.

All cloud rebuilds use four indexing threads, a configured 2 GB total indexing budget,
4 KiB dictionary blocks, ratio bounds, the same corpus/compiler/host and flags.
Their document-order hashes differ because scheduling affects flush/merge order;
these are **resulting-index** comparisons, not an isolated codec experiment.
Logical document/posting/position totals agree. Rounded/V1/V2 index bytes are
5,425,051,380 / 4,334,396,990 / 4,479,838,528. Corresponding indexing wall times
are 242.28 / 232.75 / 228.79 seconds, and peak RSS is 13,800,064 /
13,273,684 / 12,672,632 KiB. Scheduling/order differences also limit attribution
of build cost and peak memory changes. V2 official top-10 query RSS is
857,748 KiB versus rounded 1,063,264 KiB and Tantivy 727,728 KiB.

Both ARM 100k fixtures use one indexing thread and matching document-order
hashes. V1 is approximately flat for the canonical fixture but regresses
6–12% on normally merged short interior blocks. V2 recovers that regression:
canonical rounded/V2 TOP_10 33.655/33.291 µs and merged 34.500/34.217 µs;
merged standalone-term TOP_10 is 21.293/21.464 µs. V2 canonical/merged index
bytes are 92,847,338 / 138,831,203 versus rounded 106,324,232 / 145,695,257.
These small ARM differences do not establish a systemic query improvement.

The version-8 gate preserves native read-only compatibility with 6/7 while
refusing older readers on new indexes. Existing atomic writer-open migration
is reused. Payload-copy, malformed width/tag/extent, mixed codec, deletion,
reorder and sync/async checks pass. V1 full harness `20260914T070856.175675Z-full`
passes all eight phases, 1,706 non-doctest tests plus one doctest, portable core,
and 21 WASM tests. V2 check `20260914T072458.554794Z-check` passes all four phases,
portable core and 21 WASM tests; it changes only the tail encoding policy.
Big-endian word normalization has no runtime test on actual big-endian hardware.
A WASM invocation from the wrong working directory failed; the corrected command
passed. Both source versions and failure/correction logs are retained.

The full-vocabulary impact audit scans 3,964,752 external term lists and
586,083,745 postings. Final lists with multiple blocks contain 18,488,171 blocks;
only 322 need more than eight convex-envelope points. A hypothetical complete
vint representation on those final blocks costs 81,175,896 directory bytes,
82,320,119 pair bytes and 18,487,849 count bytes (about 173.55 MiB total).
This is a storage estimate, not a measured query gain or the actual new writer
layout: initial construction excludes single-block source lists, and copying
merges retain unknown records from those sources. The new format must be audited
after construction, including budgeted scratch and resident metadata costs.

Both full archives were downloaded and every manifest entry verified: V1
19,137,895 bytes, SHA-256
`95834e702eeb397178688510fd6a3fc7da259f5a0c04daf83c26fdefff0bcfd5`,
89 files; V2 73,118,337 bytes, SHA-256
`a19f929e8289f04e6a9f93bcf74fbf4055761df14cd7c9ffc062c68d13cbad43`,
144 files. The compact bundle verifies 362 entries and retains frozen source,
locks, exact commands, raw results and index/document-order manifests; raw
binaries/perf data remain in full archives, and document-order arrays on the machine.
Its SHA-256 is `7fa3a3b8e3d49475a9b235b8e7ea216a8750660fd4b9ef5d3a9ec15fbdcf126d`.

### Competitive-impact implementation, before timing

The first integer/vint envelope implementation now uses BPL2 flag 16, bounded
1–8 point records and borrowed directories. The canonical builder requires actual
effective scoring lengths; cold compaction keeps per-document lengths for changed
blocks, while concatenation and unchanged compaction blocks copy record bytes.
Native and WASM writer configuration share `posting_impact_bounds`; CLI
`--posting-impact-bounds` implies ratio bounds. Defaults remain off. The query
owner evaluates conservative f64 bounds with the canonical-f32 rounding guard;
score computation is unchanged. Diagnostics distinguish absent directories,
unknown entries and populated point counts. No query text or score threshold
enters the stored representation.

Regression tests reproduced three correctness failures before fixes: unsigned
vints could wrap on an overflowing tenth byte; posting footers admitted unexplained
trailing data; and short lazy metadata reads could panic in cold compaction.
The shared integer decoder, footer extent check and compaction source now reject
those cases. Two legacy fixtures were corrected to represent real old layouts
(no unflagged L1/ratio bytes, and f32 legacy TF words); permissive trailer handling
was not restored. Tests cover record/directory corruption, payload-copy identity,
copy/rebuild mixed sources, budgets before metadata I/O or output, cancellation,
write failure, parameter extremes, score ties, wide Boolean sums, global scoring,
length saturation, missing/multi/chunked fields, deletion and reorder.

Full harness `20260914T082009.535427Z-full` passes all eight phases: the main test
phase has 1,717 non-doctest tests and one doctest, and the separate real-server
phase has four broker tests. Portable core and native without sync pass. WASM
build and 22 browser/scalar tests pass, including native-written impact and
SIMD-v1 fixtures. Earlier attempts exposed a test-only API/import error and two
CLI test call sites missing the new argument; corrected runs pass. The short-I/O
regression and fix were followed by the final full run. These checks establish
correctness coverage, **not** a performance improvement. Full-corpus measurements
and actual index/storage/residency audits remain required.

## Competitive-impact results (September 14)

**The complete-workload performance gap remains systemic.** The initial L0-only
format changes official top-10 from 833.194 to 839.458 µs in the same binary,
while improving the separate 714-term workload from 156.392 to 86.053 µs.
Tantivy takes 503.951 and 44.395 µs respectively. The second format adds bounded
L1 group envelopes and coarse skips; the [current comparison](search-benchmark-current.md)
reports all four operations and query classes from its same-run controls.
Group top-10 regresses from rounded 846.194 to 917.003 µs, versus Tantivy's
505.370 µs. Union top-10 regresses from 1063.238 to 1302.814 µs. The unchanged
L0 index also regresses between old/new binaries (1028.730 to 1128.631 µs for
unions); preserve that control when attributing the regression. Group standalone
term top-10 improves to 70.668 µs, still behind Tantivy's 43.130 µs.
No new format becomes the default from these results.

Both cloud phases use matching document order and logical totals with one
indexing worker, a 2 GB writer buffer budget, no background merges, and a final
explicit merge. The group build also matches the L0 record coverage of its
control. All contain 5,032,104 documents, 586,083,745 postings and
1,330,791,236 positions. Rounded/L0/group index bytes are
5,067,578,899 / 5,126,280,959 / 5,135,200,732. Rounded/L0 build wall time is
683.50/688.63 seconds; the later group build takes 684.13 seconds. Peak build
RSS is 16,216,980 / 15,689,860 / 15,894,036 KiB. The configured writer buffer
budget is not a process RSS limit. Independent warm official top-10 profiles
report 1,056,896 / 1,067,744 / 1,071,308 KiB RSS, predominantly mapped file
pages, versus Tantivy's 727,788 KiB.

Each of four Summa reader/index controls per phase passes all 962 official
plus 714 supplementary COUNT comparisons with Tantivy and ordered ID/score-bit
VERIFY against its own exhaustive top-10/100/1000 oracle. Cross-engine ranking
identity is not claimed. Both ARM 100k fixtures pass the equivalent gates;
L0 and group latency changes are flat or mixed. In the ARM standalone traces,
groups skip more L1 ranges but score exactly the same number of posting blocks
as L0-only. This is reduced metadata traversal, not additional payload rejection.

The group format adds BPL2 flag 32, requires L0 impacts and L1 length metadata,
and retains the same bounded vint representation. Compatible L0 and aligned
L1 records copy byte-for-byte. Regrouped L1 metadata derives from at most eight
existing eight-point frontiers; unknown inputs yield unknown output. Cold
writers budget directories, records and 4 KiB scratch before output. Old
L0-only readers reject the new flag. Full harness
`20260914T085204.534082Z-full` passes all eight phases (1,720 non-doctest tests
plus one doctest, and four separate real-server broker tests). After gating a
native-only helper, final check `20260914T090655.316277Z-check`, portable
compilation and 23 WASM tests pass. The new 2,049-document native fixture tests
group metadata in the browser; the older L0 fixture remains unchanged.

Software profiles remain distributed across traversal, scoring and positions.
In the group official top-10 profile, scalar term score/seek are 9.03%/8.75%
self samples, window MaxScore 9.35%, phrase matching/alignment 7.99%/7.73%,
and cached position reads 6.19%. Impact bound evaluation becomes prominent in
the standalone workload but is not the dominant complete-workload cost. These
sample fractions cannot be multiplied by geometric mean latency to derive
causal time savings. No hardware-counter claim is made.

The compact evidence (local archive `benchmark-results/competitive-impacts-2026-09-14/results.zip`)
contains 521 verified entries, 8,208,173 bytes, SHA256
`25d7e957f69708203e5089cb5b14389bc5bc5b30f3a86e3410ab859e7da012c6`.
Full L0 archive: 18,527,275 bytes, SHA256
`f09782aef15e1c87c2d967a056852519e3f539ca0cfa5f62823952f22fb195fd`,
91 manifest entries. Full group archive: 116,253,814 bytes, SHA256
`5c3e196c300bc50845d3848412a2595fb5202222a835a072a0a9f8ab904374b0`,
211 entries. Full archives, every manifest entry, and the compact ZIP were
verified locally. Frozen sources, raw latency samples, gates, index manifests,
profiles, memory, build logs and commands are retained. Profiling, hashing,
transfers and builds did not overlap measured query loops.

## Rejected score-required union classification (September 14)

The full-corpus comparison rejects automatic score-required intersection for
unions in this implementation. Rounded before/after/Tantivy official geometric
mean median microseconds are 843.856/924.393/510.184 (TOP_10),
1297.641/1308.980/876.868 (TOP_1000), 1204.101/1183.079/818.484
(TOP_100_COUNT), and 576.544/569.029/413.421 (COUNT). Union TOP_10 regresses
1062.386 to 1385.058 µs; Tantivy takes 644.234 µs. Group-impact top-10 also
regresses: 930.306 to 977.419 µs overall and 1318.505 to 1615.754 µs for
unions. The small improvement in exhaustive collection does not justify this
ranked-query regression. The next candidate removes automatic classification.

All four Summa configurations pass 1,676 COUNT and exhaustive-ranked VERIFY
gates on immutable indexes; no rebuild or similarity change is involved. Both
ARM 100k fixtures pass their gates, with flat complete-workload latency.
Diagnostics confirm activation on ARM (145 queries and 1,455 windows in the
canonical group fixture), which does not establish a performance gain. Full
query profile RSS stays essentially unchanged: rounded before/after
1,057,204/1,056,212 KiB, groups 1,071,656/1,071,232 KiB, versus Tantivy
727,728 KiB. The dominant resident pages are file-backed, not heap.

In isolated union software profiles, seek preparation grows from 7.40% to
17.21% self samples on rounded bytes. The new candidate helper takes 7.97%;
length gathering and deferred scoring take 9.56% and 4.96%. Inlining changes
attribution between functions, so these percentages alone do not prove an
operation-count cause. Candidate probing still computes a complete block's
scores on the first surviving match. That specific cost warrants a separate
experiment; it is not an explanation established by these samples alone.

Final harness `20260914T092322.308216Z-check` passes all four phases, 1,723
non-doctest tests plus one doctest. Portable compilation and 23 WASM tests pass.
The first check failed two style lints, which were corrected before freezing.
The verified evidence bundle (local archive `benchmark-results/score-required-2026-09-14/results.zip`)
contains 326 entries, 4,204,146 bytes, SHA256
`1f49e4b93e7a7c00c87cbca412e7eec4c13087d49501ef7c353f116c1a69e8ce`.
The full archive contains 234 verified manifest entries, 146,090,211 bytes,
SHA256 `95dd996cfd4e9abeb167883726975970997ca72fa35157dfd25481d76069cfb8`.
Binaries, raw profiles and index-immutability checks are captured separately
from latency; the compact bundle retains source, locks, raw samples, gates,
text profiles, memory and exact commands.

## Rejected windowed semantic conjunction (September 14)

Plain unweighted same-field AND queries reused the shared text-window
executor in a separately frozen candidate. Every term is required from the
first window, and an exhausted term ends the intersection. Complete membership,
position collection, nested/boosted/mixed clauses, chunked fields and unsupported
settings retain the general scorer. There is no format change, and the rejected
automatic union classification is removed. Measurement uses the earlier
group-capable binary as the control, not the regressing score-required build.

The planner audit reproduced incorrect ranking and score bits when Boolean
grouping discarded explicit per-term statistics. `TermQueryInfo` now carries
those statistics and grouping retains original scorers where it cannot preserve
them. Terms remain scoring terms even when required; they are not converted
into filters. The first attempted opaque-decomposition fix failed the mixed
required/optional regression and was replaced. L1 candidate backfill rejects
unsupported term-local statistics instead of silently substituting parent data.
This adds a field to a public Rust planning struct; external literals need
`global_stats: None` for ordinary terms. No persisted or wire format changes.

Regression coverage includes term-local statistics, pure/mixed Boolean scores,
all-required windows with 64 terms, empty intersections, late winners, predicate
eligibility, exact count and score bits, positioned membership, and expired
budgets. Final check `20260914T101226.957816Z-check` passes all four phases:
1,725 non-doctest tests plus one doctest. Portable compilation and 23 WASM tests
pass, including AND queries on all three unchanged native golden fixtures.
The first focused invocation exposed two test-only type/private-field errors;
corrected focused tests and the complete harness pass.

The frozen candidate has 95 files, 877,935 bytes, SHA256
`4a2464cb34a0f5e3d5357ecba1e69480a829442285f75778640a05e742c1f852`.
An artifact-name collision was rejected before overwriting the historical
binary/archive. The new run uses `ranked-and` names. Six historical script/log
files overwritten during setup were restored byte-for-byte from their existing
verified evidence ZIP; the superseded setup files were retained separately.
The completed comparison rejects this dense-window AND implementation. On the
full corpus, rounded before/after/Tantivy geometric mean median microseconds
are 853.417/1087.211/495.598 for TOP_10, 1282.299/1674.846/881.234 for
TOP_1000, 1173.186/1189.554/811.799 for TOP_100_COUNT, and
572.976/581.418/411.382 for COUNT. AND TOP_10 regresses from 572.636 to
1369.351 µs, versus Tantivy's 327.833 µs. Group-impact AND similarly regresses
581.893 to 1441.958 µs. Union TOP_10 changes 1073.061 to 995.411 µs on
rounded bytes, but this does not justify the conjunction or overall regression.

Both 100k ARM layouts also reject the change: canonical rounded AND TOP_10
rises from 27.470 to 38.340 µs; merged rises from 27.315 to 38.681 µs.
All four cloud and ARM configurations pass the exact COUNT and exhaustive
ordered-ID/score-bit gates for all 1,676 queries on unchanged index bytes.
In isolated rounded AND profiles, the new dense executor accounts for 23.41%
self samples, seek preparation 16.62%, length gathering 11.85%, deferred block
scoring 8.42%, and candidate probing 7.59%. The original general path spends
33.11% in term seek and 20.81% in term scoring. Tantivy's corresponding profile
spends 29.42% in posting seek, 19.38% in intersection advance and 15.59% in
SIMD sorted decoding. These are sampling fractions, not causal time estimates.

The verified evidence (local archive `benchmark-results/ranked-conjunction-2026-09-14/results.zip`)
contains 322 entries, 4,250,359 bytes, SHA256
`5f615c0bfca083b6e16e13cd95b6c6a285a5ae94dc64abd3d4db7d40102d5c56`.
The full archive is 138,845,710 bytes, SHA256
`189f4d8672bc8c892567ff4f94b65dd9bdadc341107d20ec8e213586dc760776`,
with 234 verified manifest entries. Frozen source, raw samples, correctness,
profiles, memory and immutability evidence are retained. No format rebuild,
profiling, build or transfer overlapped latency measurement.

### Candidate-run and compact-conjunction follow-ups

A second frozen candidate scores only selected postings within each physical
block, sharing deferred TF readiness with full-block scoring. Its ARM canonical
rounded overall TOP_10 is 36.577 µs, versus the original 33.510 and first
windowed candidate's 37.435 µs; AND is 36.703 versus 26.954/38.230 µs.
Rounded union TOP_10 is 42.890 versus 44.126/44.127 µs. This recovers part of
the windowed regression, without establishing a competitive executor.

The completed cloud comparison confirms that limitation. Rounded
original/windowed/candidate-run/Tantivy microseconds are
850.355/1106.681/1040.562/503.842 (TOP_10),
1283.920/1682.766/1533.472/869.008 (TOP_1000),
1181.460/1208.586/1206.035/825.762 (TOP_100_COUNT), and
579.007/578.234/579.902/413.846 (COUNT). Union TOP_10 improves
1069.920/1013.102/950.393 µs across the three Summa binaries, versus
628.282 µs for Tantivy. AND remains regressed: 569.604/1404.613/1209.879
versus 334.471 µs. These selective-scoring gains do not rescue the dense AND
executor. All six Summa reader/index configurations pass the 1,676 COUNT and
exhaustive ordered-ID/score-bit gates; all index manifests remain unchanged.

Check `20260914T103231.933639Z-check`, portable core and 23 WASM tests pass.
The candidate-run evidence (local archive `benchmark-results/candidate-runs-2026-09-14/results.zip`)
contains 339 verified entries, 6,455,406 bytes, SHA256
`6ce8eebbe7a2cf5b29b8489d96924524c9ebdb4aef0fd11e6cbcd911fe9df64d`.
The full archive is 106,692,221 bytes, SHA256
`ae1c8872d374813473df1cddaf1429e43232a547d1a71c8d53243ba02f974c45`,
with 218 verified entries. Source, samples, tests, profiles, memory and manifests
are retained. The separate 714-term workload remains slower than Tantivy.

The current third prototype bypasses dense windows for semantic conjunctions.
It aligns cursors by increasing document frequency, then scores only eligible
intersections in bounded batches of 128 documents. It evaluates every eligible
matching document and retains canonical per-term accumulation order. Union
windows keep candidate-run scoring. A boundary regression caught the lead
cursor exposing next-block metadata before its deferred payload was loaded;
loading that payload before reading its TF fixes the failure. Focused and public
integration tests plus check `20260914T105614.770702Z-check`, portable compilation
and 23 WASM tests pass. The native ARM scoring kernel emits four-lane SIMD
divisions, verified in disassembly. Both matched 100k ARM fixtures recover the
windowed regression but remain essentially flat against the original executor:
canonical rounded TOP_10 is 33.551/36.964/33.376 µs for original/second/third
candidates, and AND is 27.240/37.437/26.915 µs. Merged overall is
34.672/38.355/34.671 µs, with AND 27.688/38.737/27.578 µs. All six controls
pass 1,676 COUNT and exhaustive VERIFY gates per fixture; index hashes are
unchanged. The completed cloud comparison leaves the systemic gap unresolved.
Rounded original/second/compact/Tantivy microseconds are
844.249/1023.705/839.632/499.850 (TOP_10),
1280.297/1530.622/1189.687/883.813 (TOP_1000),
1173.060/1218.596/1198.508/810.677 (TOP_100_COUNT), and
573.660/579.899/574.128/415.142 (COUNT). Compact AND improves the original
565.233 to 518.982 µs, versus Tantivy's 331.226 µs. Rounded union regresses
1061.052 to 1124.287 µs, versus 622.695 µs. Group-index overall TOP_10 is
829.194 µs, also well behind Tantivy. All six configurations pass every gate
and preserve all index files.

The compact conjunction's isolated AND profile places 34.55% of self samples
in cursor seek preparation, 13.21% in conjunction execution, 7.09% in 8-bit
SIMD delta decoding, and 7.02% in document block decoding. The shared BM25
kernel accounts for 0.66%. Its rounded union profile spends 54.69% in the
window executor and 6.63% in length gathering. These are sampling fractions,
not causal estimates. The union source retains candidate scoring, so changed
workload execution and code generation need separate investigation before
attributing that regression. The current evidence does not justify a claim
that scalar BM25 arithmetic explains the remaining overall gap.

The compact-conjunction evidence (local archive `benchmark-results/compact-conjunction-2026-09-14/results.zip`)
contains 340 entries, 6,418,344 bytes, SHA256
`cfc4db0f1b711566693c751a531d72e41cbc65ff45930be73d154d5fdce7fcab`.
The full archive is 109,821,586 bytes, SHA256
`6f234def0e1cdf4a227cdecea49a6fd4793c267f6601106c5948cbdc99b0cb45`,
with 218 verified entries. Source, profiles, memory, correctness, raw samples
and immutable-index manifests are retained.

### Deferred frequency experiment (mixed full-corpus result)

The shared ordinary posting iterator still decoded all term frequencies on
block load, including count-window traversal. A behavior-named regression first
fails at ID-only open, confirming that eager work. The new prototype defers
frequency decoding until a score, position-prefix read or scored visitor needs
it; membership windows use the same traversal with frequency access disabled.
The Vec and fixed-buffer callers share one extracted slice decoder. No encoded
bytes or score formulas change.

A boxed fixed-size `OnceLock` retains native Send+Sync for immutable frequency
access and avoids per-block allocation. Its scratch allocation is reused by
point probes. Forty-one focused posting tests pass, including all four codecs,
wide frequencies, borrowed/owned cursors, stopped visitors, resumption and
byte identity. Additional coverage exercises concurrent immutable first reads
and scratch-address reuse. Check `20260914T111601.289509Z-check` passes all
four phases, 1,727 non-doctest tests plus one doctest; portable compilation and
23 WASM tests pass. Both matched ARM fixtures pass their 1,676-query gates
for all six controls. Against the immediate compact-conjunction control,
rounded COUNT improves 26.467 to 26.073 µs on the canonical layout and 28.775
to 28.347 µs on the merged layout. TOP_100_COUNT regresses 32.187 to 32.563
and 36.143 to 36.936 µs respectively. The older control also runs slower in
this ARM session than earlier sessions; use paired controls, not cross-session
absolute differences. The full cloud result is also mixed. Rounded
original/compact-conjunction/deferred/Tantivy microseconds are
887.255/859.053/831.343/525.711 (TOP_10),
1334.375/1228.500/1249.446/887.944 (TOP_1000),
1219.921/1234.178/1296.959/825.161 (TOP_100_COUNT), and
579.230/577.423/554.271/412.600 (COUNT). COUNT improves about 4% against the
immediate predecessor, but scored exact-count collection regresses about 5%.
Phrase TOP_10 regresses 807.531 to 832.436 µs, versus Tantivy's 508.747 µs.
Group-index TOP_10 regresses 845.760 to 862.302 µs. This is not an established
systemic gain, and a count improvement does not justify scored regressions.

All six configurations pass the 1,676 exact COUNT/exhaustive top-k gates on
unchanged index files. The deferred-frequency evidence (local archive `benchmark-results/deferred-frequencies-2026-09-14/results.zip`)
contains 365 entries, 6,473,407 bytes, SHA256
`1bc09e5ae60c50f268ade0205aff7df08a29dd84a26509f8786a6aaac9166b4c`.
The full archive is 121,483,977 bytes, SHA256
`53f71a8bee3dc9dfffc77e7f4aca4b26f30ce1c237ac138e1cc08a9505c95c05`,
with 250 verified entries. Raw paired samples, immutable layouts, profiles,
memory snapshots, exact commands, frozen source and validation are retained.

The original rounded control in the windowed-conjunction comparison is slower
than Tantivy on 924 of 962 official TOP_10 queries. It wins on only 35 of 301
unions, one of 300 intersections and none of 300 phrases. Arithmetic mean
latency is also worse (2291.230 versus 1373.212 µs); the geometric-mean gap is
not solely an aggregation artifact. This distribution motivates revisiting
execution choice for sparse unions as well as reducing unnecessary frequency
work. A density-based compact-union path is proposed; no gain is yet measured.

### Compact sparse-union experiment (rejected at full scale)

The new metadata gate selects compact posting batches below one posting per
64 IDs across the combined term-list range. Dense unions keep their existing
window executor. All eligible matches are scored using the shared canonical
BM25 kernel and query reduction order, with absent terms omitted from scoring.
The same predicate, collector and budget protocol remain in use. This changes
execution choice, not the scoring formula or encoded representation.

The focused regression preserves exact ordered ID/score bits across all four
codecs, absent and duplicate terms, 64-clause reductions, missing norms, zero-k1,
late winners, predicates, k=0 and expired budgets. Native and async routing are
both exercised. Check `20260914T113344.861694Z-check` passes all four phases,
1,728 non-doctest tests plus one doctest. Portable compilation and all 24 browser
tests pass. An additional browser test compares sparse union scores with
independently executed single-term scores on a 4,097-document fixture. The
frozen candidate contains 96 files, 889,240 bytes, SHA256
`147c4f35e587bb92c863e780c0f266e93a20f5c84ced26c8484e0edcf067e7b6`.

Both ARM fixtures pass all 1,676 gates for six controls on unchanged bytes.
Canonical rounded original/deferred/compact overall TOP_10 is
32.995/33.085/32.926 µs, with union 44.983/44.732/43.632 µs. Merged overall
is 35.172/35.250/35.197 µs, with union 47.615/46.896/46.362 µs. Original
versus compact TOP_1000 is 48.534/47.801 µs canonical and 50.397/49.987 µs
merged, while TOP_100_COUNT regresses 32.891/33.254 and 33.502/34.148 µs.
The complete cloud comparison rejects the density gate. On the rounded index,
original/deferred/compact/Tantivy TOP_10 is
888.502/834.264/906.163/501.341 µs. Union TOP_10 is
1109.934/1007.578/1288.983/624.893 µs: the new path regresses its immediate
predecessor by 27.9% and the original by 16.1%. AND is
603.511/526.327/532.297/331.395 µs, and phrase is
811.012/835.297/840.648/498.741 µs. The same regression appears with group
impacts: overall 929.938/855.764/900.404/501.341 µs and union
1320.269/1097.928/1332.647/624.893 µs. Small ARM gains did not transfer.

Rounded TOP_1000 is 1295.926/1232.594/1220.445/881.007 µs;
TOP_100_COUNT is 1188.866/1232.823/1239.410/807.490 µs;
COUNT is 568.742/550.699/554.076/411.208 µs. The separate 714-term
TOP_10 is 155.970/160.405/159.231/43.776 µs. These isolated results do not
justify the TOP_10 regression. Remove the compact-union routing and kernel from
the next candidate, retaining its exhaustive sparse/duplicate correctness cases.
The already frozen short-WAND phase still contains this route and must be
interpreted accordingly. A following candidate must let short WAND cover these
two-term unions and retain score windows for wider unions.

Fresh candidate gates cover every 962+714 query on both unchanged indexes;
unchanged control gates are reused only after binary and manifest verification.
All controls are timed afresh. The verified full archive contains 251 files,
143,926,067 bytes, SHA256
`b9f993478949a3ab9cd40bd8333cce92a98a5eb92c979f54175277272b239c57`.
The compact evidence archive (local archive `benchmark-results/compact-unions-2026-09-14/results.zip`)
contains 368 files, 6,447,309 bytes, SHA256
`f8fa0a2810b2bbb5d8e36cb2abc7a55fe7b9a08e6a888842c82b3a62365f335c`.

Independent official TOP_10 profiles have original/deferred/compact/Tantivy RSS
of 1,056,596/1,055,900/1,057,140/727,864 KiB; anonymous residency is
5,772/5,792/5,792/768 KiB. The rest is predominantly file-backed. Software
union self samples put 44.43% in windows, 6.60% in block-bound lookup, 6.16%
in compact-union traversal, 5.85% in length gathering and 3.90% in compact
batch collection. These samples are separate from latency and are not causal
attributions; the branch does not solve the common window cost.

### Demand-driven phrase frequency (mixed full-corpus result)

Phrase confirmation previously counted every matching start even for exact
COUNT, which needs document membership. The prototype stops after the first
match and retains monotone position cursors. Scoring resumes that same scanner
for the remaining exact frequency, cached for repeated immutable score reads.
One scanner preserves first-term start counting, independent slop intervals,
repeated terms and overflow-safe offsets; the scoring formula and bytes remain
unchanged. The ordinary, async and point-backfill callers share this owner.

The nine focused phrase tests include an independent brute-force oracle across
position layouts and slop, concurrent immutable score reads, unchanged exact
score bits, membership without frequency completion, budgets and terminal seeks.
The first full check (`20260914T115515.987092Z-check`) failed the existing chunked
reorder/backfill test: a new document reused cached frequency from the previous
position buffers. The named regression
`point_phrase_backfill_invalidates_frequency_when_position_buffers_change`
reproduced the stale score before the fix. Invalidation now occurs at the
position-buffer write boundary, including direct candidate backfill. All nine
phrase tests and the original chunked regression pass after the fix. Check `20260914T121124.083859Z-check` passes all four phases, 1,730 non-doctest
tests plus one doctest; portable compilation passes. All 24 browser tests pass. The frozen source has 96 files, 893,571 bytes,
SHA256 `c0a8ebfd38d6342de98f0b53fc9e3d2662cdc6bd5a9e283a957406dff5be0b3b`;
every tar member matches its source manifest. Both ARM layouts pass all 1,676
gates for six controls on unchanged indexes. Canonical rounded original/prior/
phrase TOP_10 is 34.406/34.000/34.201 µs, and phrase-only is
30.072/30.247/30.418 µs. Merged overall is 36.119/36.048/36.245 µs; phrase-only
is 31.068/31.291/31.300 µs. Phrase COUNT is 23.772/23.927/24.028 µs canonical
and 24.866/24.870/24.808 µs merged. This does not establish a useful gain.
Full-corpus measurement is pending. The failed
check and before/after logs are retained; this prototype is not yet validated
for retention.

The complete cloud comparison does not establish a useful net gain. Rounded
original/compact-union/lazy-phrase/Tantivy TOP_10 is
850.866/878.431/894.401/497.417 µs. Phrase TOP_10 is
789.876/818.945/831.828/500.252 µs. TOP_1000 is
1273.885/1225.097/1226.926/896.090 µs; TOP_100_COUNT regresses
1197.197/1268.510/1308.048/831.392 µs. COUNT improves slightly to
575.713/559.902/551.125/414.456 µs. Group-index TOP_10 is
917.911/911.166/903.104/497.417 µs, but TOP_100_COUNT regresses
1251.855/1270.507/1294.315/831.392 µs. This mechanism alone is not a
retention result. The incremental exact-position candidate replaces its main
exact-phrase scan and must justify the resulting complete implementation.

Fresh candidates pass all 1,676 gates on both unchanged layouts; unchanged
controls reuse verified gates and all controls are timed afresh. The full archive
contains 253 files, 121,175,727 bytes, SHA256
`422d3cb3ec3b9cfae1fb189a389494056cfcd903f318376016d4ee15cd4a7c50`.
The compact evidence (local archive `benchmark-results/phrase-frequency-2026-09-14/results.zip`)
contains 377 files, 6,470,040 bytes, SHA256
`c2a8b711230f7699eb9db09b54bfa164665ae93c050da2a44803bf8414c4b834`.
Its first packaging attempt correctly failed verification because a downloader
log appended during copying. The rejected package is retained separately; the
packager now streams and hashes the same bounded input bytes, then verifies every
ZIP entry. The immutable full benchmark archive had already passed every hash.

Independent phrase TOP_10 RSS is 1,008,052/1,007,776/1,007,876/698,512 KiB;
anonymous residency is 4,832/4,832/4,832/640 KiB. After-change self samples are
19.35% phrase confirmation, 18.99% candidate alignment, 13.66% cached position
reads, 8.48% position-block lookup and 7.25% position-prefix accounting.
Phrase scoring is 1.84%. Tantivy spends 28.74% in position reads, 15.70% in
posting seek, 13.39% in phrase advancement and 12.07% in phrase matching.
Inlining moves attribution between names; these are sampled shares, not a
causal decomposition of the latency gap.

### Ranked-cursor lower-bound candidate (no overall gain)

The owning ranked cursor now probes the next document before standard-library
suffix binary search, and uses binary search after loading a new block. The
ordinary posting iterator already follows this policy; native and async ranked
cursors previously used a linear SIMD scan for long within-block jumps. The
compact-conjunction profile motivated the change by placing 34.55% of self
samples in seek preparation, not by an isolated synthetic speedup claim.

A behavior oracle passes before and after the rewrite. It covers all four
codecs, lengths around 128-posting boundaries, high document IDs, wide frequencies,
interleaved forward/equal/backward seeks and advances, exact score bits and
terminal states through sync and async methods. No I/O, representation, scorer,
ordinal policy or unsafe primitive changes. Check
`20260914T122350.655379Z-check` passes all four phases, portable core compiles,
and 24 WASM tests pass. The frozen source has 96 files, 897,590 bytes, SHA256
`8be21c1a0ad28b06839d3b106a778c527f5801fd16211e4bfac6bec87ff71556`.
Both ARM fixtures are essentially flat. The full-corpus result also fails to
establish an overall gain. Rounded original/predecessor/candidate/Tantivy TOP_10
is 850.390/896.385/898.717/504.500 µs. AND regresses 529.567 to 538.196 µs,
versus the original 570.548 and Tantivy 334.396. Union is 1279.837 to
1268.195 µs, versus 1068.074/626.239. Phrase is 834.337 to 835.779 µs,
versus 789.039/503.136. Overall TOP_1000 is 1282.343/1239.709/1238.880/
872.234 µs; TOP_100_COUNT is 1177.261/1251.392/1236.760/804.397 µs;
COUNT is 570.818/550.648/547.036/412.251 µs. Group-index TOP_10 regresses
905.133 to 917.899 µs. These are measured together; the predecessor already
contains rejected experiments, so recovery against it would not justify adoption.

The full capture is verified: 251 files, 153,910,595 bytes, SHA256
`c73be65a09e397a01ed65491c54197d43b2349ee9142267f271cbeff988025a6`.
The compact evidence (local archive `benchmark-results/ranked-seek-2026-09-14/results.zip`)
contains 375 files, 6,517,736 bytes, SHA256
`a0ce7adc18a65d31cc204a28aff034ee5d7e0569d4b09a1842fdc09cca0c5ee3`.
All new-candidate exact-count and exhaustive ordered-ranking gates pass; unchanged
controls reuse hash-verified gates and are timed afresh. Independent official
TOP_10 RSS is 1,057,252/1,056,564/1,057,096/727,820 KiB, with anonymous
residency 5,776/5,784/5,784/776 KiB. AND self samples still place 36.05% in
seek preparation and 8.41% in the sync seek boundary. Changing the within-block
search alone did not resolve the executor cost.

### Block occupancy does not explain the measured query gap

The full vocabulary has many short blocks after compatible copying merges, and
rounded index payloads are larger than Tantivy's. The current rounded file
manifest has 2,203,973,483 posting bytes and 2,752,821,603 position bytes; Tantivy
has 1,054,571,755 `.idx` bytes and 1,850,554,774 position bytes. These are different
representations, not directly comparable per-value codecs.

However, summing the 714 exact term-count gates gives 98,360,178 postings. The
retained historical frontier probe on the same corpus/query terms visited
780,298 physical blocks: 126.05 postings per block, versus a theoretical
768,791 blocks at complete per-term packing. That older layout differs from the
current matched-order rebuilds, so this is historical diagnostic evidence. It
shows why vocabulary-wide block counts cannot establish fragmentation as the
cause of the benchmark's systemic query gap. No default repacking policy or
query-time block coalescing is justified by that aggregate. Compatible merges
continue copying encoded blocks as required by the system contract.

### Query shape and the short-union hypothesis

The compact-conjunction full-corpus TOP_10 breakdown is:

| Regular query shape | Queries | Summa compact (µs) | Tantivy (µs) |
| ------------------- | ------: | -----------------: | -----------: |
| Two-term union      |     198 |            880.244 |      392.545 |
| Three-term union    |      83 |           1550.570 |     1223.131 |
| Longer union        |      18 |           2646.073 |     2744.842 |
| Two-term phrase     |     198 |            534.399 |      379.884 |
| Three-term phrase   |      83 |           1464.954 |      751.893 |
| Longer phrase       |      19 |           3158.904 |     1355.093 |

The complete mix also includes mixed Boolean and negated queries, plus special
stress queries; they remain in every official aggregate. Short unions suggest a
block-WAND experiment, while wider phrases suggest investigating position
alignment and unnecessary verification work. These are distinct hypotheses.
The existing window executor is not uniformly worse on all shapes, so wholesale
replacement is not justified. The current candidate now implements two-term
block-WAND after the low-density compact path, reusing the owning cursor,
block bounds and canonical BM25 scorer. A one-document case in the shared
scoring kernel avoids an unused gather buffer. Two focused exhaustive tests
pass, including all four codecs, norms/missing norms, gaps, absent/duplicate
clauses, reversed order, ties, seeded floors, high IDs, filters, late winners,
zero-k1, k=0/1/10/1000/all, and sync/async routing. A new browser regression
compares dense two-term and duplicate-term unions with independently executed
term scores. Check `20260914T124145.575176Z-check` passes all four phases,
1,733 non-doctest tests plus one doctest; portable compilation and all 25 browser
tests pass. Frozen source contains 97 files, 903,734 bytes, SHA256
`972f017a8fcfd4aeb88e9b2c6866b91b2ffd1d4cf49ae8c648db314fa344c78f`.
Every archived source entry matches its manifest. Matched timings are pending;
correctness alone does not establish a performance improvement.

The ranked-cursor lower-bound ARM comparison passes all 1,676 gates for all six
controls on each unchanged layout, but is essentially flat. Canonical rounded
original/phrase-control/lower-bound TOP_10 is 33.151/33.042/33.063 µs; AND is
26.834/26.709/26.888 µs. Merged overall is 35.183/35.542/35.378 µs, with AND
28.062/28.528/28.360 µs. The canonical/merged TOP_1000 predecessor/candidate
values are 47.430/47.596 and 50.155/49.960 µs. This does not establish a gain;
full x86 measurement is still pending.

From the compact-union cloud phase onward, unchanged controls reuse their
previous verified correctness evidence after checking source manifest entries,
identical binary hashes and immutable index manifests. Reused exact counts must
also equal the freshly verified candidate counts. Every changed candidate still
runs all 962 official plus 714 supplemental COUNT and exhaustive top-k gates.
All controls are timed afresh. `verify-reuse.json` records the source mapping;
this avoids rerunning expensive identical validation without changing acceptance
criteria or claiming those reused checks were newly executed.

The frozen short-WAND candidate's ARM runs have completed all gates on both
unchanged fixtures and regress union latency. Canonical rounded
original/ranked-seek/WAND TOP_10 is 34.040/33.914/35.045 µs, with union
44.878/42.995/48.866 µs. Merged TOP_10 is 35.275/35.315/36.527 µs, with
union 46.619/45.587/51.668 µs. TOP_1000 is
47.800/47.475/49.465 µs canonical and 53.510/53.003/55.432 µs merged.
COUNT is 24.356/24.158/23.968 and 24.758/24.476/24.261 µs, respectively.
This is not an adoption result. The frozen binary still includes the rejected
compact-union density route. Its ARM diagnostic routes 97 of 301 unions through
short WAND, 105 through compact union, 96 through windows and three through the
single-term path. Full-corpus measurement remains pending.

### Incremental exact-phrase position intersection (validation in progress)

The previous scorer reads every term's positions before checking whether a
prefix can match. A named regression reproduces that unnecessary later read.
The prototype retains the original first-term starts, visits other terms in
ascending document frequency, and compacts starts after each intermediate exact
offset intersection. Empty prefixes stop before later position reads. The last
pair confirms one occurrence and resumes only for exact scoring. A shared pair
cursor preserves first-start multiplicity and uses u64 offset comparisons;
nonzero slop retains the existing independent-interval scanner. Position reads,
frequency invalidation, native/async execution and point backfill remain in the
same scorer. No encoded representation or BM25 formula changes.

The baseline prefix-read regression fails as expected. All 12 phrase tests pass
after implementation, including independent occurrence/duplicate/overflow and
resumption oracles, multi-block positions on all four codecs, false candidates,
backfill and deadline boundaries. The first multi-block test run exposed a test
fixture error: it paired position streams with postings built without cumulative
position cursors, so document 128 read the wrong position range. The corrected
fixture uses the canonical writer with `with_positions=true`; production code
was not changed for that failure. Initial harness
`20260914T131923.543464Z-check` stops at a test-only Clippy range-loop warning;
the loop is corrected and the full check is running again.

This candidate also removes the rejected compact-union density gate and unused
kernel. Its sparse/duplicate/64-clause score-bit oracle is retained against the
routed executor and passes. Supported two-term unions now select short WAND;
wider unions retain score windows. These source changes must be reported
together, including effects outside their intended query families. A new browser
phrase test checks exact occurrence sets and top-k versus complete collection on
260 documents with repeated terms and failed prefixes. Full validation and
matched performance remain pending; neither change is an established gain.

Browser validation exposed an existing correctness problem beyond the position
intersection prototype: a quoted multi-token phrase on a field with no token
positions became a Boolean AND, accepting nonadjacent and reversed terms. The
new browser occurrence oracle failed, and
`phrases_require_token_positions_instead_of_accepting_unordered_terms` reproduced
the native failure before the fix. Ordinal-only fields also cannot establish
token adjacency. The shared phrase capability check now rejects both modes with
a field-specific error, and is used by parsing, native/async scoring and point
backfill. One analyzed token remains a term query. Fields with token/full
positions retain their existing matching and score semantics; no bytes change.

The parser also now keeps every default-field branch when their tokenizers yield
different token counts; previously a one-token first field returned early and
lost other fields. Additional routing propagates the original query's error.
Three native integration regressions cover unsupported modes, token/full modes,
all default-field branches, pre-I/O parsing and additional routing; all pass.
All 23 parser unit tests pass. The prior stemmed-phrase test is updated to require
an error for its unsupported field rather than pinning the incorrect fallback.
This is an intentional compatibility correction, documented in the
[phrase contract](dynamic-tokenizer-and-phrase.md#phrasequery-on-the-wire).
A fresh `full` harness, `20260914T133533.057936Z-full`, is running before the
candidate is frozen. Earlier pre-correction native binaries and logs are kept
separately and are not used for timing.

The corrected incremental-position candidate now passes full harness
`20260914T133533.057936Z-full`: all eight phases, 1,739 non-doctest tests plus
one doctest and four real-server broker tests, including native without sync,
portable compilation and API docs. All 28 browser tests pass across six files.
The frozen candidate contains 101 files, 928,552 bytes, SHA256
`7a4e33a0e41349ad5dd371173081850af9fe9dd7e25217fe38177732f521dd87`.
Both ARM fixtures now pass all gates and retain byte-identical indexes. Rounded
original/predecessor/candidate TOP_10 is 34.859/35.922/35.970 µs on canonical
and 34.861/36.459/36.164 µs on merged. Phrase improves only 30.237 to
29.748 µs and 31.230 to 30.745 µs versus the predecessor. Full-corpus
measurement is queued. This does not establish a net performance gain.

### Systemic cost isolation: latency strata and decoded work

The verified deferred-frequency run is still slower across phrase-query sizes:
Tantivy-latency quartiles give Summa/Tantivy TOP_10 ratios of
1.651/1.574/1.612/1.711. The AND quartiles are 1.608/1.520/1.443/1.443.
Unions differ: 2.419/1.733/1.331/0.936. Each family retains all its official
queries; quartiles only explain the aggregate, and do not define a query route.
These are geometric ratios of per-query medians from the same seven-sample run.
The growing absolute phrase cost rules out fixed query setup as the sole cause.
It does not establish which representation or traversal accounts for the gap.

A diagnostic-only experiment will count actual posting-ID, term-frequency and
position block decodes, decoded values, and encoded payload bytes for both
engines on the immutable full corpus. Instrumentation lives in isolated copies
of frozen Summa and the checksum-verified Tantivy 0.26.0 crate; it adds bounded
atomic counters at owning decoders and emits one record per adapter request.
No instrumentation enters production source or latency binaries. Recorded byte
counts are decoder payload input, not physical I/O or RSS. Position requests
and posting seek calls can additionally distinguish repeated work from decoding.
The original and instrumented adapters must preserve protocol/count results,
and Summa must still pass the exhaustive ranking oracle. Full traces run only
after the queued latency comparisons finish. This is a proposed diagnostic,
not a measured optimization or an explanation established by sample percentages.

## Selected phrase planner and native performance work (September 14)

The verified short-WAND experiment is rejected. Rounded union TOP_10 regresses
1268.903 to 1451.435 microseconds from the immediate predecessor, while the
original control is 1068.457 and Tantivy 638.869. Overall TOP_10 is
850.274 / 899.328 / 937.045 / 509.501 microseconds for original / predecessor /
WAND / Tantivy. All exact-count and exhaustive-ranking gates pass, so the
rejection is about performance. The complete verified evidence (local archive `benchmark-results/short-wand-2026-09-14/results.zip`)
retains the seven-sample full-corpus run, profiles, memory and immutable-index
manifests. Earlier statements that this experiment was pending describe its
then-current state.

The selected source restores the V4 ranked executor, including its original
within-block search. It removes the later compact-union and short-WAND routes
and their one-hit scorer specialization, but retains their independent behavior
oracles. It adds selectivity-ordered phrase document and position intersections
and retains the phrase correctness fixes. Its frozen source has 102 entries,
933838 compressed bytes and SHA256
`9e409e1cb42cfec4de6dd4e2cfac1004316904dee5bfbe0e0fe1153a94917f96`.
Native `check` run `20260914T145044.033860Z-check` passes all four stages;
portable compilation passes. The already queued browser build also completed
with all 29 tests passing. Following the user's instruction, native performance
iterations do not repeat the WASM build each time.

Both 100k ARM fixtures pass all 1676 count/ranking gates for all four controls,
with unchanged index bytes. Canonical original / best V4 / V9 / selected overall
TOP_10 is 34.554 / 34.531 / 36.014 / 33.902 microseconds; phrase is
29.436 / 29.704 / 29.044 / 28.303. The merged fixture gives
40.079 / 40.355 / 41.712 / 39.668 overall and
33.798 / 33.956 / 33.405 / 32.659 for phrases. These paired improvements
are small and do not establish competitive full-corpus performance. The
selected full-corpus timing and full decoded-work diagnostics remain pending.

A further execution investigation will compare semantic conjunctions using
the shared `BlockPostingIterator` against the ranked `TermCursor` traversal.
Both implement the same selectivity-ordered leapfrog intersection today, but
the ranked cursor additionally carries deferred score, sparse-I/O, ordinal and
block-bound state. The proposal is to move conjunction membership through the
existing posting reader while retaining the owning canonical batch scorer,
predicate, cancellation and exact-count semantics. Scratch remains bounded by
the number of query terms times the existing 128-posting block. No new scorer,
writer, format, query-text dispatch or approximation is proposed. Decode work
and whole-query timings must distinguish reduced cursor overhead from extra
TF decoding or allocation; an isolated prototype is not a selected default.

### Proposed direct accumulation of canonical window scores

The ranked union executor currently retains a full per-term score plane and
presence mask, then reconstructs each surviving document's canonical sum.
For a window with no nonessential terms, it can instead visit all cursors in
canonical query order and accumulate the final score directly. Two terms with
nonnegative, non-NaN scores also need only a single sum: exchanging two addends
does not reassociate a floating-point reduction. Unsupported BM25 parameters
retain the existing score planes. Wider windows with nonessential terms retain
the current canonical reconstruction.

This proposal keeps block bounds, candidate membership, ranking precision,
predicate evaluation, cancellation and collection unchanged. It makes score
accumulation explicit in the shared window writer and allocates per-term
scratch only when reconstruction is necessary. Worst-case scratch stays at the
existing term-count times 4096 slots; eligible windows avoid per-term score
writes, presence-mask writes and the final per-term reduction. This is an
isolated experiment, not a measured gain or a default format change.

## Full-corpus decoded work and matched execution experiments (September 14)

The verified decoded-work evidence (local archive `benchmark-results/decoded-work-2026-09-14/results.zip`)
covers all 962 official queries and 714 supplemental terms, every command,
two independent equal passes, and immutable full-corpus indexes. Instrumented
copies are excluded from latency measurements. The three Summa sources pass
cross-source ordered top-1000 ID and raw-score-bit equality on rounded and group
indexes, their pruned top-10/100/1000 agrees with exhaustive scoring, and exact
counts agree with Tantivy. Payload bytes below are encoded decoder inputs;
they exclude headers, skip metadata and inline postings and are not physical I/O.

| Official TOP_10 work          |    Summa V4 | New phrase planner | Tantivy 0.26 |
| ----------------------------- | ----------: | -----------------: | -----------: |
| AND decoded document IDs      |  96,438,925 |         96,438,925 |   93,314,862 |
| AND document payload bytes    | 110,617,532 |        110,617,532 |   65,064,884 |
| AND frequency payload bytes   |  94,734,254 |         94,734,254 |   66,494,517 |
| Union decoded document IDs    |  42,999,618 |         55,557,181 |  100,506,664 |
| Union decoded frequencies     |  23,423,882 |         40,256,936 |  100,506,664 |
| Phrase decoded document IDs   | 121,824,506 |         96,438,925 |   97,512,075 |
| Phrase positions requested    |  85,496,780 |         27,363,707 |   27,364,093 |
| Phrase decoded positions      | 238,953,816 |        106,114,929 |  106,867,899 |
| Phrase position payload bytes | 345,468,936 |        174,250,686 |  135,277,467 |

The diagnostic new-planner source retains V9's ranked executor; its union
column is not the selected production executor. The selected source combines
the new phrase planner with V4 ranking. The phrase planner removes most excess
position requests: full-corpus requests differ from Tantivy by 386, rather than
being exactly equal. V4 union pruning already decodes fewer than half as many
IDs and about a quarter as many frequencies as Tantivy, despite slower measured
latency. AND traversal work is close while its document payload is 1.70 times
larger. These findings make execution cost and representation the next measured
questions; decoder counts alone do not establish which causes the latency gap.

The first diagnostic build attempt was rejected before accepting any counters:
Cargo reused stale adapter artifacts across extracted source trees with shared
target directories. The retry touches crate and adapter entry points, builds
verbosely, embeds a distinct diagnostic source identity for every source, and
requires that identity plus nonzero counter records before full collection.
Only the rebuilt run is included in the verified results. Original logs and
binaries from the rejected attempt remain in the workspace evidence. The
36,835,614-byte compact ZIP has SHA256
`5f1c660160d4746a51f3f8fd3473b243b25cd290dde814369d94bf740fad2d86`.
Byte-identical source and oracle files are stored once, with a complete logical
manifest and a verified restoration script; no query or score record is omitted.

The incremental phrase-intersection experiment also has
complete verified latency evidence (local archive `benchmark-results/phrase-intersection-2026-09-14/results.zip`).
Original / preceding WAND / phrase-intersection / Tantivy official TOP_10 is
851.004 / 923.941 / 869.619 / 501.402 microseconds. Phrase TOP_10 is
789.233 / 828.064 / 783.673 / 497.892. Its improvement over a regressing
predecessor does not establish a better complete executor. The selected source
therefore retains the subsequent V4 rollback and selectivity planner described
above.

Two isolated native execution variants pass all scoring and integration tests
and all 1676 cross-source ranking/count oracles on both 100k ARM fixtures.
The shared conjunction reader retains the canonical batch scorer. The direct
window variant accumulates in canonical order when all terms are essential,
or exchanges only two supported nonnegative addends. Its numeric guard also
requires a positive BM25 denominator for a zero frequency, because the public
posting API permits zero. Other windows retain canonical score reconstruction.
Neither variant has been copied into the selected main source pending full
measurements.

A clean paired ARM repeat gives selected / reader / direct-window TOP_10 of
54.167 / 53.994 / 53.514 microseconds on the canonical fixture and
35.945 / 35.510 / 35.449 on the merged fixture. Absolute host timings varied
between runs; only paired comparisons are meaningful. A final native run of
the conservative numeric guard gives selected / guarded-window TOP_10 of
33.048 / 32.449 and 34.246 / 34.115 on the two fixtures, with union results
43.518 / 42.057 and 45.662 / 44.751. These modest gains do not close the
systemic gap. Both runs retain exact score bits and unchanged index bytes.

The queued full comparison includes original, V4, V9, selected, shared-reader,
guarded-window, selected with the existing Simd4x codec, and Tantivy. The new
Simd4x index uses the exact original rounded-index writer binary, one indexing
thread, no background merges, the same corpus and memory setting, and requires
identical document order and logical counts. Index preparation overlapped
untimed diagnostics; its elapsed time is provenance, not comparative indexing
performance. Query timings begin only after all preparation and correctness
gates finish. This run will distinguish source execution changes from changing
encoded representation without changing scoring or corpus order.

### Proposed intersection of overlapping decoded blocks

The next isolated semantic-AND experiment retains DF-ordered leapfrog seeks to
find the first common document, then intersects the remaining decoded slices
up to the minimum current block end. It records each surviving posting's
ordinal inside its owning reader buffer, fetches frequencies only after all
terms and the predicate match, and sends bounded batches to the existing
canonical scorer. It advances the lead past the processed interval and resumes
skip-based alignment. This avoids repeatedly entering the cursor API for every
candidate inside already decoded blocks. It neither expands to a dense document
universe nor scans through blocks that existing seeks can skip.

Invariant: the next result is the least unprocessed common document; every
candidate in the current interval is considered exactly once, and no reader
moves while its decoded ordinal is retained. Frequency rows remain in original
query order. Scratch is bounded by query terms times 128 byte-sized ordinals,
plus existing frequency rows and fixed 128-document buffers. Deadline checks
occur at alignment boundaries and per block/term, before frequency work and
canonical scoring. Query control stays in the executor; the posting owner
exposes only crate-private borrowed slices of its current decoded block. Native
and async text execution share this path, with no persisted-byte changes.

The first experiment uses bounded sorted-slice searches to isolate batching
from ISA-specific kernels. The [SIMD intersection research by Lemire, Boytsov
and Kurz](https://arxiv.org/abs/1401.6399) and its
[reference implementation](https://github.com/fast-pack/SIMDCompressionAndIntersection)
show why intersection, rather than decoding alone, deserves vectorization.
Their reported gains do not predict Summa performance. A later SIMD kernel
must beat this control, retain scalar behavior and validate x86 and ARM; none
is implied by the scalar prototype. Independent all-codec result/score oracles,
selective and dense lists, tails, high IDs, predicates and deadlines must pass
before whole-query measurement. This is proposed work, not a retained gain.

## Verified selected phrase source and codec comparison (September 14)

The eight-configuration evidence (local archive `benchmark-results/phrase-plan-selected-2026-09-14/results.zip`)
is complete and verified. Full archive: 206,602,204 bytes, 340 manifest entries,
SHA256 `46ca5ec700f59792ace554a03fb2b66fdbc57c2f3a897e2d28c5b89f4372f16c`.
Compact archive: 45,194,951 bytes, 593 entries, SHA256
`b7a90620ad24101474e22f6f97c165a811906ede08484dd7f862a881f75056e9`.
All official and supplemental exact-count, exhaustive-ranking and immutable
index gates pass, including ordered cross-index score bits for matched Simd4x.

| Official operation, geometric mean microseconds | Original |       V4 | Selected | Shared reader | Direct window | Selected Simd4x | Tantivy |
| ----------------------------------------------- | -------: | -------: | -------: | ------------: | ------------: | --------------: | ------: |
| TOP_10                                          |  868.436 |  819.568 |  803.338 |       810.856 |       823.140 |         824.067 | 516.614 |
| TOP_1000                                        | 1289.470 | 1222.007 | 1177.598 |      1144.803 |      1152.750 |        1219.905 | 890.228 |
| TOP_100_COUNT                                   | 1185.589 | 1219.299 | 1233.130 |      1232.886 |      1240.993 |        1283.243 | 841.675 |
| COUNT                                           |  572.636 |  559.319 |  540.990 |       540.643 |       541.937 |         557.645 | 415.010 |

Selected / reader / window / Simd4x / Tantivy TOP_10 is
1002.554 / 1129.146 / 1076.454 / 994.686 / 647.225 for unions,
524.453 / 484.000 / 523.692 / 542.731 / 343.946 for AND, and
755.534 / 750.865 / 757.511 / 793.697 / 507.057 for phrases.
Selected wins only 54 of 962 official queries: 46 unions, five AND and one
phrase. Neither new execution variant nor the codec improves complete TOP_10.
The selected main source remains 1.555 times slower than Tantivy; the goal is
not achieved. The V9 control, also retained in the archive, takes 867.999
microseconds overall and is not selected.

Matched Simd4x reduces index bytes from 5,067,578,899 to 4,044,864,324; Tantivy
uses 3,031,085,762. Their document-order arrays match exactly (40,256,832 bytes,
SHA256 `1b1419398758b57e960db9d0a2cca68ae29e4448e1b1888c3912d382d8113a2b`).
Separate warm official profiles give selected rounded / Simd4x / Tantivy
VmHWM of 1,056,876 / 851,072 / 676,520 KiB, with anonymous residency
5796 / 5784 / 764 KiB. Most residency is mapped file pages. The codec is a
useful space tradeoff but is slower in this matched run; its default stays
unchanged. Preparation timing was overlapped with untimed diagnostics and is
not an indexing performance comparison.

The shared-reader variant changes conjunction traversal, yet union TOP_10
regresses 12.6%. Its normalized x86 union-kernel disassembly is identical to the
selected build: 14,869 bytes and 2937 static instructions, including 105 YMM
instructions and no ZMM instructions. `score_text_run` and `seek_prepare` also
have identical normalized disassembly. Normalization removes relocation
addresses; this does not prove identical cache placement or runtime conditions.
It does rule out lost vectorization in these inspected kernels as an explanation.
The direct-window kernel grows to 15,509 bytes and 3089 static instructions.
Static instruction counts are not executed instruction counts or cycle estimates.
A rotated full-workload-pass repeat is queued to separate persistent differences
from sequential engine-order effects, using the same CPU, compiler, indexes and
all official/supplemental queries. Frozen original, V4, selected, reader,
window, selected Simd4x and Tantivy controls are retained, plus the refined
block-intersection source. This runner is additional evidence, not a silent
replacement of the upstream benchmark protocol.

The block-intersection prototype passes 29 native scoring tests, four focused
integration suites, Clippy, and all 1676 ordered cross-source ID/score/count
oracles on both ARM fixtures. Its first implementation regresses AND locally.
The refinement probes the current and next posting before binary-searching a
suffix, preserving the same block algorithm. A source search overlapped one
ARM timing run; that run is excluded and a clean same-binary repeat retained.
In the clean repeat, selected / shared-reader / first-block / refined-block
TOP_10 is 33.001 / 32.658 / 33.044 / 32.642 microseconds on the canonical
fixture and 34.569 / 34.162 / 34.502 / 34.151 on the merged fixture. AND is
27.116 / 26.708 / 27.693 / 26.841 and
27.996 / 27.451 / 28.406 / 27.517. The refined block variant does not beat
the shared reader locally; full measurements remain necessary. No block
prototype has been copied into the main source.

A separate trace of the public synchronous API confirms that single-segment
search still enters the shared Rayon pool. Nested query preparation also uses
parallel iteration, so bypassing that pool requires an explicit execution
policy to preserve ownership and avoid falling onto an unrelated global pool.
No inline-execution change or scheduling speedup is implemented or claimed.

### Verified rotated execution comparison

The rotated repeat (local archive `benchmark-results/block-executor-2026-09-14/results.zip`)
is complete: 135,476,723-byte full archive, 186 verified entries, SHA256
`b70c200e720fe25d7fc572eac062beb7a35c929d3c0b99445adcd8e6fa1e50ee`.
The compact ZIP is 26,884,167 bytes, 434 logical entries, SHA256
`923b1afb2ad3134b42d4194043229e07390afe2fb75b10ef8f58fcb1e82eab86`.
All fresh block-prototype ranking/count gates and all index byte checks pass.
Seven complete workload passes rotate engine order, with the driver and each
engine on CPU 2. This supplements the upstream sequential runner above.

| Official operation, geometric mean microseconds | Selected | Shared reader | Direct window | Refined block intersection | Tantivy |
| ----------------------------------------------- | -------: | ------------: | ------------: | -------------------------: | ------: |
| TOP_10                                          |  781.674 |       797.928 |       776.620 |                    768.576 | 503.673 |
| TOP_1000                                        | 1186.198 |      1149.523 |      1167.233 |                   1160.892 | 885.159 |
| TOP_100_COUNT                                   | 1187.665 |      1186.427 |      1175.677 |                   1154.313 | 814.953 |
| COUNT                                           |  518.756 |       520.476 |       516.417 |                    519.281 | 395.314 |

Selected / reader / window / block / Tantivy TOP_10 is
962.099 / 1120.601 / 941.001 / 972.873 / 618.928 for unions,
541.939 / 501.061 / 543.038 / 510.088 / 359.403 for AND, and
702.000 / 695.602 / 699.666 / 697.618 / 474.353 for phrases.
The reader's union regression persists after order rotation. Its shared-reader
AND traversal beats the more elaborate block algorithm by 1.8%; the block
algorithm also loses to that reader on both clean ARM fixtures. Its 1.7%
complete TOP_10 improvement against selected does not demonstrate an algorithmic
advance over the reader or close the gap. Neither is selected into main.
Selected wins 54/962 queries and four phrases in this repeat; even the fastest
prototype remains 1.526 times slower overall. Simd4x takes 830.893 microseconds,
remaining a space tradeoff rather than a warm-latency gain.

Separate official profiles give selected / block / Tantivy VmHWM of
1,056,444 / 1,057,092 / 676,628 KiB and anonymous residency of
5796 / 5784 / 764 KiB. Supplemental term TOP_10 is
164.422 / 162.879 / 50.619 microseconds: the separate single-term gap also
remains. All four operations and every control remain in the archive.

Validated `perf annotate` output shows approximately 13.9% of the selected
union kernel's local software samples in dense score clearing and 37.0% in
candidate extraction/filtering. That kernel accounts for about 55% of union
self samples. Two inlined posting-suffix binary-search loops together account
for about 39.9% of the phrase candidate function's local samples. These are
rounded software-sampling attributions, not hardware cycles or predictions of
achievable speedup. Reader and selected have similar local instruction profiles.
The remote symbol-filtered objdump files must not be treated as valid empty
kernels; the accepted static comparison uses nonempty local address ranges.

### Retained sparse candidates for a single essential cursor

Instruction-level software profiles of the selected union executor place
substantial samples in clearing its dense score array and extracting candidate
IDs/scores from membership words. The current block-max partition sometimes
leaves exactly one essential cursor. That cursor already produces unique,
ordered IDs with their scores, so converting its runs to a dense window and
back to ordered candidates is unnecessary.

The retained implementation appends those decoded runs directly to the existing
candidate vectors when exactly one cursor is essential. It retains the same
window boundaries, bound partition, pruning threshold, nonessential probes,
predicate, score arithmetic and canonical per-term reconstruction. Multiple
essential cursors retain the existing union algorithm. This differs from the
rejected density-based compact union, which changed which postings were visited
and scored; the proposed representation shortcut preserves those decisions.

The invariant is identical ordered candidate IDs and score bits at the boundary
before predicate filtering. Only slots with a current contribution-presence bit
may be read during canonical reconstruction. Both representations consume the
same owning cursor's decoded runs through one shared visitor; no second scorer
or decoder is introduced. Scratch remains bounded by query terms times the
existing 4096-ID window. For one essential cursor, candidate materialization
cost becomes proportional to matching postings rather than the document-ID span
and bitset words. Deadline and error boundaries remain unchanged.

The original upstream runner confirmation (local archive `benchmark-results/essential-upstream-2026-09-14/results.zip`)
uses the unmodified `make bench` client from benchmark-game revision
`a7c75473e91746280c5f01e69bf594ece5fca560`, all 962 official queries and 714
supplemental terms, seven samples after ten seconds of warmup, the same native
LTO binaries, CPU 2 and unchanged indexes. Previous selected / retained / Tantivy
geometric mean median microseconds are:

| Official operation | Previous selected | Retained | Tantivy |
| ------------------ | ----------------: | -------: | ------: |
| TOP_10             |           795.725 |  695.820 | 504.580 |
| TOP_1000           |          1177.202 | 1138.288 | 889.312 |
| TOP_100_COUNT      |          1171.875 | 1179.823 | 812.102 |
| COUNT              |           528.180 |  535.326 | 413.002 |

OR TOP_10 is 993.110 / 665.800 / 631.487: a 33.0% improvement against
the previous source and still 1.054× Tantivy. AND is
520.488 / 513.014 / 334.383; phrase is 751.583 / 744.745 / 500.328.
Complete TOP_10 improves 12.6%, remains 1.379× Tantivy, and wins 149/962
queries (141 OR, five AND, one phrase). This does not close the remaining
AND, phrase or single-term gaps. Supplemental TOP_10 is
156.641 / 159.788 / 46.680 microseconds. Measured regressions include
official COUNT +1.4%, TOP_100_COUNT +0.7%, supplemental TOP_10 +2.0%
and supplemental COUNT +3.3%; the report retains these alongside the OR gain.

The rotated complete-workload comparison (local archive `benchmark-results/essential-runs-2026-09-14/results.zip`)
independently measures previous / retained / Tantivy TOP_10 at
785.051 / 690.329 / 508.613 overall and 968.813 / 656.765 / 622.593
for OR. Its official TOP_1000 is 1179.516 / 1130.310 / 884.159,
TOP_100_COUNT 1207.123 / 1197.464 / 826.872, and COUNT
523.719 / 530.124 / 399.446. This additional runner rotates engine order
between full workload passes; it is explicitly separate from the upstream
confirmation. Separate official profiles show peak RSS of
1,057,232 / 1,057,068 / 676,580 KiB, with anonymous residency
5796 / 5792 / 764 KiB. The mapped-index size and memory disadvantage remain.
The union window kernel's self sample share falls from 54.6% to 38.2%; these
are software samples, not hardware cycle counts.

Both clean ARM fixtures show complete TOP_10 improvements of 2.2–2.8% and OR
improvements of 8.1–8.6%. All 1676 cross-source ordered top-1000 IDs, raw score
bits and counts agree on both ARM fixtures and the cloud rounded/grouped-impact
indexes. Fresh exhaustive ranking and Tantivy count gates pass. The upstream
confirmation reuses gates only after verifying exact binary/index/proof hashes,
and checks every timed COUNT/TOP_100_COUNT response again. All index bytes stay
unchanged. The selected main source passes `check_search.py check`, all four
stages and 1744 tests, run `20260914T174645.582411Z-check`; portable core
compilation also passes. No new WASM build was run, following the user's
native-performance instruction.

Both full captures and compact archives were checked entry by entry, including
the exact manifest file set. The rotated full capture is 92,563,160 bytes,
SHA256 `a748df4a8ebd3cfaccd4d4e7255019522cc3ad5effb675e9614c4bbe88b53277`;
its 21,503,226-byte compact ZIP is
`8bc734cb1cfdcea363852bd24ae02ce91f540be0c7743d04d04ddc360da5a9a2`.
The original-runner full capture is 17,100,564 bytes,
`c72a159c21729a33e14f3057b2509e0d43aab61267f0525a15da5951d046953c`;
its 1,600,415-byte compact ZIP is
`f3cdd57a40fc2f4d14f6b3ae07866f0b97cb66e338e975f724b0203f981b7e87`.

### Retained phrase rejection before position confirmation

This revisits the earlier rejected phrase-bound experiment against the newer
executor and selectivity planner. The earlier hook modified phrase confirmation
after receiving a collector threshold; the new capability leaves confirmation
exact and lets only the top-level ranked driver omit noncompetitive candidates.
The previous flat complete-workload result remains relevant: a new full-corpus
comparison is required, not a claim of a newly discovered technique.

The isolated phrase experiment applies the principle described in
[Lucene PR 15861](https://github.com/apache/lucene/pull/15861): prove that a
candidate cannot compete before initializing its positions. Summa's existing
candidate/confirmation protocol already separates document alignment from
phrase verification. The top-level ranked collector can use an optional final
score bound from that same scorer; complete and custom collectors must continue
ordinary exact traversal. Nested, filtered and chunk-folded scorers retain their
existing behavior unless their own final-score contract explicitly supports
this capability. Both public synchronous and asynchronous top-k search use the
shared top-k driver.

For a plain phrase with document lengths, its original-first term frequency
bounds the number of matching starts. The minimum frequency across terms is
not valid under Summa's duplicate-start multiplicity semantics. Use the first
frequency and the current document's actual scoring length with the existing
conservative BM25 envelope and floating-point guard. Unsupported numeric
parameters or length representations decline the optimization. No score model,
token-position format, or relevance semantics change.

The ranked driver checks the bound only after its retained heap is full, retains
ties using the existing document/score ordering, and confirms/scores candidates
that can compete. It checks cancellation before bounds, confirmation and
collection, retaining the existing observable deadline state. No second phrase
matcher or heap is introduced, and scratch is independent of hit count. Exact
COUNT and TOP_100_COUNT must still confirm every match; ranked `total_seen`
remains the number actually examined, not an exact-count promise. Tests must
cover false candidates, late winners, ties, duplicate starts, offsets/slop,
global statistics, cancellation, deletions, chunk/ordinal fallbacks and native
versus async equivalence before whole-query timing. The new source passes
231 query unit tests, seven integration
suites, Clippy and portable compilation. Both ARM fixtures preserve all 1676
ordered cross-source score/count oracles and immutable index bytes. Paired
TOP_10 is 31.086/30.951 microseconds overall and 26.527/26.478 for phrases on
the canonical fixture; merged is 31.135/31.137 and 26.083/26.100. These flat
local results alone do not justify selection.

The verified full-corpus comparison (local archive `benchmark-results/phrase-confirm-bound-2026-09-14/results.zip`)
measures previous OR / phrase-bound / Tantivy TOP_10 at
682.521 / 656.018 / 500.204 microseconds overall and
694.004 / 600.638 / 473.602 for phrases. Phrase improves 13.5%, overall 3.9%,
and phrase wins increase from 5/300 to 51/300; complete wins rise from 151/962
to 195/962. The source is retained on September 15. It still takes 1.311×
Tantivy overall. OR changes 646.229 → 654.302 (+1.2%) and AND
533.886 → 535.857 (+0.4%); these regressions remain in the record.
TOP_1000 is 1129.602 / 1123.565 / 876.069, TOP_100_COUNT
1184.191 / 1164.267 / 817.612, and COUNT
523.146 / 525.251 / 398.753. Supplemental TOP_10 is
168.042 / 167.819 / 50.903; supplemental COUNT is
12.575 / 12.967 / 9.095 (+3.1% against the control).

This is the additional rotated-pass protocol: all 962 official and 714
supplemental queries, four commands, seven samples after ten-second warmups,
the same compiler/native-LTO flags and CPU 2. The control is the hash-verified
retained OR binary. Fresh rounded/group-impact exact-count, exhaustive top-k
and cross-source ordered raw-score-bit gates pass; every index byte remains
unchanged. No builds or profiles overlap latency. Separate official profiles
give previous/current/Tantivy peak RSS of 1,057,120/1,057,040/676,628 KiB and
anonymous residency of 5796/5796/776 KiB. In phrase-only software samples,
position-cache reads fall from 9.7% to 6.0%, while candidate bounds and their
BM25 envelope now account for 8.4% and 5.1%. These fractions describe sampled
work, not hardware cycles or predicted speedups.

The full archive is 112,863,062 bytes with 172 verified entries, SHA256
`44782d7bb285970ac91a09f9dcadc990680526b4786250ee1316859efba5ca1f`.
The compact ZIP is 21,563,066 bytes with 231 logical entries, SHA256
`7af86f0dfc098745cfdd4c3f2d573bcc3bf05a56996a15c24f9a9276e374072f`.
Exact entry sets, bytes, aliases and external hashes were verified. The native
main-source harness is running; no new WASM build is scheduled. Original
upstream-runner confirmation is pending cloud reauthentication.

### Proposed block skipping for ranked phrases

The retained phrase bound rejects individual aligned candidates, so it still
pays document intersection costs before each rejection. Its phrase profile
attributes 28.4% of software samples to candidate alignment. The next isolated
experiment exposes the current original-first posting block's last document,
maximum frequency, minimum length and length/TF ratio from the structures owner.
The ranked driver caches a conservative phrase bound through that last document
and seeks past the block when the whole range cannot compete. Otherwise it
retains the existing per-document bound and exact position confirmation.

The first term bounds matching-start multiplicity; no minimum across phrase
terms is used. The query's BM25 owner evaluates the existing conservative
envelope from the block maximum TF and its length lower bounds. Stored min-length
and ratio metadata use `max(raw_length, 1)`, matching phrase length semantics.
Impact records use a different zero-length fallback (`TF`), so this prototype
does not use those records for phrase bounds. Unknown length metadata falls back
to the universally safe phrase length floor of one. A raw f32 term block bound
is not sufficient for this driver: its floating-point guard must be explicit.

Only the shared top-level ranked driver consumes a range hint. The hint covers
the current candidate and all later candidates through its inclusive end;
monotone document IDs and existing heap tie ordering make range skipping safe.
Complete/custom collectors, nested/filter/chunk wrappers and unsupported numeric
parameters retain exact traversal. Check cancellation before range lookup and
before the ensuing seek. Cache at most one range; no format, payload rewrite,
second scorer or corpus-sized state is introduced. Validate zero lengths,
duplicate starts, range-boundary ties, late winners, terminal seeks, all codecs,
global statistics and complete-count/native-async equivalence before timing.
This is an unmeasured proposal, separate from tiled seeking and direct impacts.

### Proposed bounded vector search within decoded posting blocks

Phrase instruction profiles put substantial samples in serial binary searches
of decoded posting suffixes. Summa already has SSE2/NEON lower-bound kernels;
reintroducing a full linear vector scan would repeat an earlier discarded path.
The next isolated experiment keeps current/next-document probes, binary-searches
16-value tile maxima, and applies the existing SIMD kernel only to the selected
tile. At most three tile comparisons precede one bounded vector scan in a
128-posting block. Short tails use the same kernel's bounded scalar remainder.

The structures owner retains cursor movement, deferred frequencies and position
prefixes. The result must equal scalar lower-bound on the remaining sorted IDs,
including tail blocks, high unsigned IDs, terminal seeks and backward probes.
No payload, skip metadata, buffer capacity, scoring or query planning changes.
Use the existing posting/phrase regressions, compare ordered score bits and index
bytes, and measure complete queries on both architectures before selecting it.
This remains a proposal and is independent of phrase score-bound pruning.
The prototype passes 227 query tests, 207 posting tests, six integration
suites, Clippy, portable compilation and both 1676-query cross-source oracles.
It changes the ordinary posting iterator, used by phrases and complete
intersections; ranked plain AND retains its separate typed cursor. The first
ARM run completed during an interruption and shows large control variation
between commands (including 23–67 microseconds on supplemental TOP_10 across
fixtures). Raw samples are retained, but they do not support a selection claim.

### Proposed direct use of complete impact bounds

The full-corpus decoder diagnostic records 40.2 million decoded postings for
the 714 ranked terms with ratio bounds, 3.1 million with group impacts, and
4.8 million for Tantivy. The impact representation already removes substantial
payload work, yet its measured standalone latency remains higher. The query
owner currently computes a raw min-length bound, a ratio bound and an impact
bound, then takes their minimum for every new block/group.

For a validated complete impact record and supported similarity parameters,
its conservative envelope alone bounds the actual score. An isolated rewrite
will return that bound directly and compute the existing simpler bounds only
when the impact record is absent, unknown or numerically unsupported. This
avoids redundant divisions and norm calculations without changing stored
records, query-global statistics, canonical scoring or default index policy.
Because conservative floating-point guards differ, it may admit a few blocks
that the previous minimum rejected; exact ranking remains the invariant,
not identical pruning decisions. Test both ratio-only and impact-bearing
indexes, unsupported parameters, raw score bits, unchanged bytes and complete
workload latency before selection. This is not yet a measured improvement.
The isolated source passes 227 query tests, six integration suites, Clippy and
portable compilation. All 1676 cross-source ordered ID/raw-score/count oracles
pass on both ARM fixtures with both rounded and group-impact metadata, and
index bytes remain unchanged. The interrupted ARM run has control TOP_10
varying from 37 to 234 microseconds between fixtures; it is not accepted as
evidence of a speedup. Native repeats and a full-corpus comparison remain needed.

The retained phrase source passed the complete native harness on September 15: all four stages, 1749 tests. The verified check archive (local archive `benchmark-results/phrase-confirm-bound-2026-09-14/native-check.zip`) is 122192 bytes, SHA-256 `ed335c9d3c9c60c495587260b9f1f304f187e8bfd9eae8536006dd4b6a95f47a`. The portable core check also passed; WASM was not rebuilt, following the user's instruction.

## Proposed competitive required/optional text windows (September 15)

The latest full rotated run leaves the 40 mixed required/optional official
queries at 5025.754 microseconds versus Tantivy 2313.232 (geometric mean of
per-query median TOP10). They are included in the 962-query total. Trace:
public parsed BooleanQuery -> shared async/sync Boolean planner -> general
BooleanScorer and complete child streams. Required clauses drive one-document
alignment; optional scorers are sought for every match. The specialized ranked
text window executor currently admits only pure OR, with a separate all-required
conjunction path. This is a query-shape execution gap, not evidence of a new
codec bottleneck.

Proposal: extend the existing text window owner with a compile-time required
mode. Keep its ordinary OR instantiation separate so the inner OR loop need not
branch on semantic requirements. A bounded u64 mask records required cursor
identities after sorting. The rarest required term supplies initial window
boundaries and candidates. Within that window, any optional term whose absence
provably makes the score noncompetitive can replace the candidate driver when
it is rarer. Existing run scoring and candidate-intersection helpers process
remaining terms; required membership is checked even for a zero score bound.
Existing per-term presence storage and query-order reduction preserve exact
scores, including duplicate clauses. All-required queries retain their current
conjunction path. TopKCollector and BM25 remain the single scoring owners.

Invariants: matches require every MUST term; SHOULD affects score only. A missing
MUST empties the query, a missing SHOULD does not. Candidate driving is independent
of canonical score addition (MUST then SHOULD, preserving order inside each).
Only a strict conservative bound can make an optional clause locally required;
equal-score document ties remain eligible. Exact COUNT, TOP100+COUNT, positioned,
chunked, nested, boosted, per-term statistics and unsupported compositions retain
their existing complete/semantic paths. Query-global statistics, eligibility and
deadlines must propagate identically. No persisted bytes, defaults, public scorer
API or cache policy change. Scratch remains bounded by MAX_QUERY_TERMS \* 4096;
no corpus-sized bitset or second posting representation is introduced.

Cost hypothesis: traverse the sparsest semantically or competitively required
posting run and seek other lists only to survivors, amortizing score dispatch and
reducing optional probes. Validate random and adversarial score-bit oracles,
zero-TF membership, duplicate/missing clauses, late winners, equal ties, seeded
thresholds, all codecs, chunks, global statistics, exclusions, offsets and
cancellation. Compare all 962+714 queries, counts and memory on fixed index bytes;
measure x86 and ARM before retaining. This remains an isolated proposal.

The phrase block-bound prototype passes 234 query unit tests, seven integration suites, Clippy and portable compilation. All four
ARM R/G layouts preserve all 1676 ordered top-1000 raw score/count oracles.
Fresh paired ARM timing (11 rotated samples, both fixtures, all queries and four
commands, no builds during timing) remains essentially flat: canonical official
TOP10 31.150/31.191 microseconds, phrase 26.563/26.461; merged official
31.835/31.671, phrase 27.155/26.890. No default or retained-source change follows
these local measurements. Full-corpus x86 validation awaits cloud authentication.

### Zero-frequency BM25 correctness finding

Required-window adversarial validation exposed an existing canonical arithmetic
edge case: frequency zero with k1=0 (or b=1 and length zero) evaluates 0/0 and
produces NaN. These settings and stored zero-frequency entries are supported.
A pruned score stream cannot conservatively bound this NaN as an ordinary positive
score, and the new exact oracle disagrees with exhaustive ranking. The initial
seeded test also incorrectly seeded a NaN threshold; that test issue was removed,
and the unseeded mismatch remains reproduced in the saved failure log.

Fix proposal: define a zero-frequency contribution as positive zero in the shared
Bm25Params owner; a zero field boost also contributes zero. Bounds for an all-zero
frequency block return zero (infinite conservative ratio/envelope fallback is
still safe). Preserve the exact expression and operation order for positive
frequencies/boosts. Default free scoring/bound helpers delegate to this existing
owner so behavior cannot drift. No format, membership, count or nonzero score
change is intended. A behavior-named test must fail before this correction; the
full fixed-corpus score-bit oracle and existing bounds tests must pass afterwards.

### Explicit Boolean statistics correctness finding

The new mixed-query global-statistics oracle also exposes an existing planner
inconsistency. BooleanQuery resolves its own statistics for grouped execution but
does not put that resolved value into child ScorerOptions. Complete/fallback child
streams therefore use local or inherited statistics instead. On the fixture,
ranked score 9.727402 disagrees with exhaustive 0.005412242 for the same document.
Flattening a nested pure SHOULD Boolean also drops that child's explicit
statistics. Both are violations of the documented precedence rule.

Correct the shared async/sync Boolean planner by passing its resolved statistics
through existing child options, and retain a Boolean boundary when that node has
explicit statistics. Explicit leaf or nested-query statistics still take
precedence over the inherited parent. A direct explicit-leaf reference pins
complete, ranked, positioned and nested scores before adopting either correction.
There is no new statistics representation or scorer; missing-statistics fallback
policy remains unchanged. Benchmark-generated global statistics are already
carried through options, so full-corpus ordinary scores are expected unchanged,
which must be verified rather than assumed.

Preserving the explicit nested-statistics boundary then exposes a second part of
that fallback bug: a summed parent receives only each child's top-k, omitting
joint winners outside individual child heaps. The same independent nested
reference fails ranked collection after complete collection passes. Generic
multi-SHOULD composition must request complete text child streams before summing,
as the existing multi-MUST path already does. The regression retains this nested
joint-winner case, and the saved intermediate failure distinguishes it from
statistics propagation. The fix remains in the shared Boolean planning owner.

### Required-window continuation: full-corpus result and selection

The zero-frequency and Boolean statistics/nested-ranking fixes are now retained
in the workspace. The behavior-named regressions fail against their saved
pre-fix sources and pass after correction. The native harness
`20260915T044129.810906Z-check` passes all four stages: 1751 tests, 26 ignored.
The preceding run stopped on a broker index-discovery timeout; the unchanged
serial repeat passed. Both logs are retained. No lifecycle or RPC code changed,
and WASM was not rebuilt, following the user's instruction.

The performance candidate and its control both carry those correctness fixes.
Both preserve all 1676 ordered top-1000 document IDs, raw score bits and exact
counts on canonical and merged 100k ARM fixtures, with rounded and group-impact
metadata. The candidate also passes 233 query unit tests, nine integration
targets, Clippy and portable compilation. It remains isolated pending the
complete x86 comparison; the correctness corrections do not depend on selecting
the optimization.

The fresh ARM run uses 11 rotated samples, 15-second warmup, all 962 official
queries and 714 supplemental terms, and four commands on unchanged index bytes.
Mixed required/optional top-10 improves from 57.686 to 45.810 microseconds on the
canonical fixture and 59.644 to 47.963 on the merged fixture. Overall official
top-10 changes from 31.338 to 30.940 and 32.045 to 31.982 respectively. These
small-fixture results do not establish a full-corpus or overall superiority claim.
Official maximum RSS is 54,214,656/54,444,032 bytes for control/candidate on the
canonical fixture and 55,279,616/54,640,640 on the merged fixture. RSS includes
mapped pages and is not a heap or scratch-capacity measurement.

Cloud authentication is restored. The full-corpus continuation builds both
frozen sources on the original Cascade Lake machine, with the same compiler, native
CPU flags, LTO and unchanged indexes. It independently gates both binaries
against exhaustive ranking, the prior cross-source score-bit oracle and Tantivy
counts before timing all queries. Compilation, validation, hashing and profiling
remain outside measured latency.

The full-corpus run is complete and its
verified evidence (local archive `benchmark-results/required-windows-2026-09-15/results.zip`)
is packaged. Both binaries carry identical correctness fixes; only the candidate
adds the required mode. All 962 official queries and 714 supplemental terms ran
under four commands, seven complete workload passes with rotated engine order,
on unchanged rounded index bytes. Both binaries pass exact counts against
Tantivy, pruned top-10/100/1000 against their own exhaustive scoring, and the
cross-source ordered top-1000 ID/raw score-bit oracle on the rounded and
grouped-impact indexes. Control/candidate/Tantivy geometric means, in
microseconds:

| Command       |  Control | Required windows | Tantivy | Candidate / control |
| ------------- | -------: | ---------------: | ------: | ------------------: |
| TOP_10        |  664.428 |          627.006 | 503.728 |               0.944 |
| TOP_1000      | 1131.650 |         1079.835 | 876.436 |               0.954 |
| TOP_100_COUNT | 1182.419 |         1142.753 | 812.016 |               0.966 |
| COUNT         |  525.464 |          526.410 | 402.372 |               1.002 |

The 40 mixed required/optional queries improve from 5348.734 to 1281.944
microseconds at TOP_10 (Tantivy 2333.201); every one of the 40 improves, by
1.13× to 21.7× with a median of 3.87×, and 39 now beat Tantivy. At TOP_1000
they improve 5565.3 to 2696.8 (Tantivy 2788.5). The 19 negated queries improve
727.130 to 645.203 because their optional part now uses the same executor.
Disclosed regressions: union TOP_10 659.286 to 670.099 (1.6%), union COUNT
691.0 to 698.9 (1.1%), phrase COUNT 689.0 to 695.4 (0.9%). The pure-OR
instantiation is unchanged in source, so these are run-to-run drift or code
layout effects, not a scored-path change; they remain within the rotated
runner's earlier control variation. Supplemental single-term TOP_10 is
167.173 versus 163.693 (2.1% better); its other commands are flat. Peak RSS
is 1,057,460 versus 1,057,712 KiB on the official workload and 369,568 versus
369,556 KiB on the supplemental workload; Tantivy uses 661,496 and 249,048 KiB.

Separate profiles show the mixed-query work moving from per-hit `TermScorer::score`
(39.7% of control self samples) and `seek` (14.3%) into batched
`score_candidates_sync` (28.0%), window appends (8.8%) and length gathering
(8.1%). Overall official TOP_10 samples remain dominated by phrase candidate
alignment (12.8%) and `seek_prepare` (10.1%).

The candidate is selected: `scoring.rs`, `boolean.rs` and the
`ranked_required_optional` integration test are copied into the main source
exactly as frozen for the cloud build (the test file is a later local revision
that also passes; timed binaries contain no tests). The main native harness
`20260915T075139.617117Z-check` then passes all four stages with 1754 tests
(26 ignored); the
captured check (local archive `benchmark-results/required-windows-2026-09-15/native-check.zip`)
pins the selected source hashes. WASM was not rebuilt, per the user's instruction. The remaining ranked gap is
now concentrated in intersections (537.637 versus 360.507), phrases (607.099
versus 474.557) and standalone terms, not in query-shape execution gaps.
Original upstream-runner confirmation for the phrase and required-window
changes has not been run. The benchmark machine is stopped, not deleted; its indexes
and evidence remain on its boot disk.

### Proposed score rejection before heap-key comparison

The retained phrase source's separate full-corpus supplemental top-10 profile
attributes 42.94% of self samples to `execute_single_text`, 18.19% to length
gathering and 6.57% to batched BM25 arithmetic. Inspection of its actual x86
binary finds that a full heap still converts both score bit patterns into total
order keys and evaluates document/ordinal tie breaks on every candidate. This
is instruction evidence, not a measurement of a replacement.

An isolated experiment adds a numerical less-than rejection in the existing
`ScoreCollector` only after its real heap is full. A score strictly below the
cached worst score cannot enter that heap. Equal scores, signed zeros and NaNs
still reach the existing total-order comparator; underfilled heaps and virtual
threshold sentinels retain their existing behavior. No second collector,
scoring formula, score quantization, new allocation or persisted format is
introduced. The hypothesis is fewer instructions on the common rejected-hit
path across existing text and sparse callers. Test against an independent sorted
reference with score bits, ordinals, non-finite values, late winners and repeated
seeds, then compare whole workloads before considering selection. This remains
a separate proposal from required/optional windows.

Measured outcome: the isolated source passes the new total-order reference test
(all k, seeded and exceptional values), the query unit tests, integration
suites, Clippy and portable compilation, and preserves all 1676 cross-source
ordered oracles and index bytes on both ARM fixtures. Paired ARM timing against
the zero-frequency control (11 rotated samples, all queries, four commands) is
flat: canonical official TOP_10 31.172 versus 31.025 microseconds, supplemental
TOP_10 19.583 versus 19.426; merged official TOP_10 31.467 versus 31.639,
supplemental 20.756 versus 20.594. Differences are within run-to-run
variation, so the change is not selected and no cloud run was scheduled. Its
evidence is retained under `.context/score-rejection*`.

### Cleanup continuation: reject detected content corruption at collection boundaries

The September 15 review fixes payload-induced cursor panics, but its infallible
posting iterator still logs an invalid block and terminates. A complete term or
phrase collector can consequently report successful partial results. Logging
alone does not satisfy the search error contract.

Implemented: the existing immutable `PostingListReader` owns one shared,
write-once record of the first rejected posting block. Lists borrowed from that
reader report content-decoding failures into that record. Segment collection
checks it before and after traversal, including nested scorers and candidate
backfill; a detected failure returns `Error::Corruption`. The record never resets
on that reader, so an old or concurrent query cannot clear another query's
failure. Reopening creates a separate owner. This is detection of visited corrupt
payloads, not authentication of arbitrary bit flips or an eager corpus scan.
Cost: one bounded record per segment, one Arc clone per acquired posting list,
and boundary checks; no new per-hit atomic operations or payload residency.
Persisted bytes, merge copying and successful-query scores remain unchanged.
Validate a payload-only mutation through public native/async ranked and complete
collectors, wrappers, and immutable reader replacement, then rerun score-bit
oracles and compare full workloads against the frozen cleanup binary.

A second remaining review finding concerns chunk compaction. Re-encoded partial
blocks used the source chunk map's BM25 length floor. Deleting long chunks can
lower the replacement map's floor, making that persisted bound nonconservative.
The rebuild must use raw surviving chunk lengths, as ingestion and merge do;
unchanged copied blocks already use those raw bounds. A regression removes the
long chunks and checks replacement bounds against the new one-token floor.

### Proposed byte-gap validation without a decoded-ID scan

The cleanup's two-fixture ARM comparison preserves all 1676 ordered ID/score-bit
oracles but slows official TOP_10 by 1.3–1.9% and COUNT by 3.0–3.2%. Reusing
scratch alone does not offset the new per-block content checks. The next isolated
experiment keeps those checks and reduces their read volume for Rounded 8-bit
gaps: nonzero raw gaps plus matching first/last decoded IDs imply strict order.
There are at most 127 gaps, so their sum is below 2^32; any wrap would end below
the first ID and fail the endpoint check. Other codecs keep the decoded-ID
ordering check. This checks at most 127 payload bytes instead of adjacent pairs
in 512 decoded bytes, with bounded scratch and unchanged persisted formats.
Compare against the corrected cleanup source, not an unchecked predecessor;
retain only after all-codec corruption tests, raw score oracles and paired ARM
and x86 full-workload measurements. This remains a proposal until measured.

### Cleanup validation recovered and resumed on September 15

Recovered all six review reports and four implementation reports from the
interrupted session. The cleanup's native check
`20260915T084353.359471Z-check` passed formatting, focused Clippy, tests and the
native-without-sync boundary. Its release build was interrupted and has now
been rebuilt from a source manifest. Four ARM fixtures (canonical/merged,
rounded/group impacts) each preserve all 1676 ordered top-1000 document IDs,
raw score bits and counts against the saved independent oracle.

The retained cleanup covers nested required-child completeness under summed
parents; nonpositive boost handoff; adjacent query modifiers; phrase statistics;
corrupt posting content; safe SIMD decode bounds and shared scalar tails;
codec/position admission; dictionary and validation budgets; merge bound
promotion/accounting; and bounded reuse of query scratch. Dead constructors,
duplicate codec helpers, unused decoder branches and outdated experiment names
were removed. No posting codec, cache or scoring approximation default changes.

The first paired ARM comparison (Apple M4, Rust 1.98.1, native CPU flags, LTO,
unchanged index files, all 962 official and 714 supplemental queries, 11 rotated
samples, 15-second warmup per command) measures pre-cleanup/cleanup, microseconds:

| Fixture                |          TOP_10 |        TOP_1000 |   TOP_100_COUNT |           COUNT |
| ---------------------- | --------------: | --------------: | --------------: | --------------: |
| Canonical official     | 30.860 / 31.439 | 44.823 / 45.309 | 30.970 / 31.807 | 22.583 / 23.268 |
| Merged official        | 31.785 / 32.194 | 46.294 / 46.687 | 31.940 / 32.657 | 22.676 / 23.402 |
| Canonical supplemental | 20.505 / 20.515 | 35.590 / 35.656 | 20.004 / 20.078 |   9.536 / 9.514 |
| Merged supplemental    | 20.963 / 21.036 | 36.541 / 36.400 | 20.833 / 21.161 |   9.458 / 9.450 |

These are correctness/maintenance changes with a measured small cost, not a
speedup. The corruption-propagation and compaction-floor regressions above both
fail before the continuation fixes and pass afterwards. The next paired
comparison isolates the byte-gap validation proposal against those corrected
sources; the old pre-cleanup binary is retained as a separate reference.

The continuation's full native harness `20260915T091108.787920Z-full` passes
all eight stages: 1786 tests (25 ignored in the standard run), four additional
real-server broker tests, focused Clippy, formatting, native-without-sync,
portable core and warnings-as-errors API documentation. WASM is intentionally
not rebuilt under the standing user instruction.

The byte-gap prototype preserves the eight corrected/candidate ARM raw score
oracles. It improves official TOP_10 by 1.4% on the canonical fixture and 0.4%
on the merged fixture, and TOP_1000 by 0.9% / 1.1%; supplemental results are
mostly flat. Separate four-command peak-RSS runs show corrected/candidate
48,119,808 / 47,874,048 bytes (canonical official) and
49,414,144 / 49,119,232 bytes (merged official). These are process RSS including
mapped pages, not isolated heap measurements. New integrity bookkeeping is
constant-sized per segment and accounted separately from validation-cache bytes.

ARM disassembly confirms the new raw-gap branch uses byte-wide NEON equality
checks; the fallback keeps the existing 32-bit ordering checks. The checked
decoder grows from 2996 to 3752 bytes of machine code, so this is a read-volume
tradeoff rather than a code-size reduction. Full-corpus x86 results determine
selection; these local gains alone do not establish an overall lead over Tantivy.

### Cleanup full-corpus result and selection

The full run and archive are complete and verified. The byte-gap source passes all x86 core unit/integration tests. Both corrected
and byte-gap builds pass the 1676-query ordered ID/raw-score/count oracle on
rounded and grouped-impact layouts. All three
index manifests, including Tantivy, are unchanged. The final source's ARM
release is byte-identical to the frozen timed byte-gap binary after a private
helper rename and strengthened test-only assertion.

Same-run pre-cleanup / corrected / byte-gap / Tantivy geometric means (µs):

| Workload / command         | Pre-cleanup | Corrected | Byte-gap | Tantivy |
| -------------------------- | ----------: | --------: | -------: | ------: |
| Official TOP_10            |     639.042 |   657.869 |  656.242 | 524.750 |
| Official TOP_1000          |    1172.666 |  1205.357 | 1198.499 | 976.103 |
| Official TOP_100_COUNT     |    1323.847 |  1386.651 | 1382.786 | 913.604 |
| Official COUNT             |     541.666 |   575.994 |  568.712 | 411.441 |
| Supplemental TOP_10        |     203.755 |   216.043 |  214.988 |  61.578 |
| Supplemental TOP_1000      |     703.733 |   725.518 |  722.426 | 458.030 |
| Supplemental TOP_100_COUNT |     865.366 |   896.808 |  887.841 | 273.643 |
| Supplemental COUNT         |      13.533 |    13.558 |   13.273 |   9.299 |

**Decision:** retain byte-gap validation for the modest official COUNT gain
(1.3%, faster in six of seven complete passes) and the small ARM improvements.
Treat x86 top-k differences as inconclusive: TOP_10 improves 0.25% by the
per-query-median aggregate, but individual pass ratios span 0.990–1.004;
TOP_1000 improves 0.57% but only three passes are faster. Negated TOP_10
regresses 2.5% versus the corrected control. Supplemental COUNT uses the
unchanged metadata shortcut; its 2.1% difference is not evidence of faster
posting decoding. No cache, codec or approximation default is changed.

The net safety/cleanup cost remains: byte-gap is 2.7% slower than pre-cleanup
at official TOP_10 and 5.0% slower at COUNT. The engine remains 1.25× slower
than Tantivy on official TOP_10 and 3.49× on supplemental terms. This iteration
does not establish ranked-search superiority or recover the entire safety cost.

Peak RSS over all four commands (KiB), pre-cleanup/corrected/byte-gap/Tantivy:
official 1,271,384/1,270,888/1,270,960/855,596; supplemental
491,400/491,356/490,952/348,528. Mapped pages dominate; no memory reduction is
claimed from these essentially flat Summa values. Byte-gap adds no scratch,
and the new shared integrity record is constant-sized per segment.

The verified archive (local archive `benchmark-results/cleanup-2026-09-15/results.zip`) and
full native check (local archive `benchmark-results/cleanup-2026-09-15/native-check.zip`)
retain source hashes, per-pass samples, errors reproduced before fixes,
correctness/format tests, memory and index manifests. See its
[reproduction notes](benchmark-results/cleanup-2026-09-15/README.md).

Remaining work recorded at cleanup: retain efficient batched collection under
deletion predicates (addressed for composites in the continuation below); reuse the remaining
16 KiB generic collector score scratch; study bounded reuse of content-validation
proofs without warming or retaining corpus payloads; and address the larger
intersection/phrase/single-term execution gap. These are proposals, not selected
changes. Original upstream-runner confirmation of post-OR changes, cold-cache,
concurrent-ingest and tail-latency measurements remain unrun. The preserved machine
is stopped after successful archive download and verification.

### Filtered collection windows — proposal, September 15

Trace: public segment collection (sync/async) wraps scorers through
`filtered::filtered`; Boolean filter push-down also constructs `PredicatedScorer`.
At the start of this continuation, the wrapper hid the driver's document/score-window capabilities,
so deletion masks and exclusions restore per-document dynamic dispatch.

Proposed change belongs in the existing wrapper: delegate bounded window
production, filter set bits in ascending document order, and add MUST verifier
scores in the same order as scalar `score`. Leave the cursor on the next eligible
match. Reuse the caller's fixed 4096-document score/membership buffers; introduce
no wrapper allocation, cache, format or alternate executor. Score-window
collection uses the existing collector's 16 KiB score buffer, including its
current per-query allocation; count-only filtering needs no additional scratch.
Positions retain the existing scalar path. Raw score bits, membership, forward-only consumption,
empty/tail windows and verifier position must match scalar traversal. Retention
requires measured paired results and regression checks; gains are not established.

The first cross-source ARM masked experiment shows why unconditional forwarding
is insufficient: official compound-heavy collection improves 15–20%, but the
714 standalone terms regress 6–12% on the canonical fixture (merged terms also
regress). Leaf window production and a second predicate pass can cost more than
scalar traversal. Revised proposal: a conservative scorer capability defaults
to false; eligible Boolean composites opt in because batching amortizes child
traversal. Predicate wrappers preserve that declaration. Leaves keep their
existing filtered scalar path, without frequency thresholds or new executors.
The ungated prototype and its measurements remain archived for comparison.

### Filtered collection windows — final ARM evidence

The refined candidate opts in only eligible Boolean composites; predicate
wrappers forward that capability, and leaves keep scalar filtering. The final
native `check` run `20260915T101236.184084Z-check` passes all four stages,
including 1789 tests (25 ignored), Clippy and native-without-sync compilation.
Separate portable-core compilation passes. No runtime source changed after the
release build. Its 331-file source manifest and all four ARM index manifests
are verified. All 1676 standard ordered-ID/raw-score/count oracles pass on
canonical/merged rounded and grouped-impact layouts; masked oracles also match
on both rounded fixtures. WASM was not rebuilt under the standing instruction.

Apple M4, Rust 1.98.1, native CPU, LTO, 100,000 documents, unchanged index bytes,
fixed `doc_id % 8 != 0` mask; seven rotated complete passes after ten-second
warmup. Before/after geometric means of per-query medians, microseconds:

| Fixture / workload           | Top 100 + exact count |     Exact count |
| ---------------------------- | --------------------: | --------------: |
| Canonical official           |       49.029 / 41.366 | 40.747 / 32.716 |
| Merged official              |       51.536 / 44.012 | 42.886 / 35.098 |
| Canonical supplemental terms |       31.447 / 31.224 | 19.823 / 19.271 |
| Merged supplemental terms    |       33.109 / 32.501 | 20.261 / 19.921 |

Official improvements are 15.6% / 14.6% for top-100 plus count and 19.7% / 18.2%
for count. All seven top-100-plus-count passes improve; count improves in seven
canonical and six merged passes (one merged pass is 21.5% slower). The prior
leaf regression is gone. Minor standalone-term differences do not establish a
new leaf optimization: that algorithm remains scalar.

The standard deletion-free ARM workload remains essentially flat. Canonical
before/after official TOP_10 is 31.766/31.653 µs and COUNT 23.227/23.169;
merged is 31.798/31.745 and 23.474/23.382. These differences are not claimed as
algorithmic gains. Standard four-command peak RSS is canonical official
48,070,656/48,365,568 bytes and merged 49,512,448/49,332,224 bytes; no resident
memory reduction is claimed.

The two old scalar window methods total 1524 bytes of ARM machine code. The
new delegated fill methods plus shared filtering routine total 1064 bytes,
with 144 additional bytes of wrapper capability checks. Driver dispatch happens
once per window; per-document predicate and verifier checks remain. This
measurement covers those methods, not the complete executable's code size.

The final full-corpus x86 comparison below completes selection. The rejected
unconditional-forwarding version improved masked official top-100 plus count
by 28.8% and count by 25.4%, but made masked standalone-term COUNT 77.0% slower
(408.242/722.527 µs). That regression reinforces keeping leaf filtering scalar.

### Filtered collection windows full-corpus result and selection

**Decision:** retain the conservative composite capability. The final source
passes x86 query/integration checks, both full-corpus standard 1676-query raw
score/count and pruned-ranking oracles, and the separate masked oracle. All
rounded, grouped-impact and Tantivy index hashes are unchanged. Three downloaded
archives have verified external hashes and exact entry manifests; each entry's
size and SHA-256 is checked before packaging.

Same preserved Cascade Lake machine, Rust 1.98.1, native CPU flags, LTO, CPU 2,
5,032,104 documents; seven rotated complete passes and ten-second warmup.
Fixed-mask before/after geometric means of per-query medians (µs):

| Workload                      | Top 100 + exact count |         Exact count |
| ----------------------------- | --------------------: | ------------------: |
| Official                      |   2242.510 / 1600.919 | 1446.428 / 1080.239 |
| Supplemental standalone terms |   1592.253 / 1587.173 |   415.478 / 413.316 |

Official gains are 28.6% and 25.3%; every complete pass improves. Per-pass
current/previous ratios span 0.707–0.726 and 0.738–0.755 respectively. Unions
supply most of the gain: 14344.029/5184.997 µs for top-100 plus count and
7072.212/2628.385 for count. Masked phrase COUNT regresses 6.7%
(714.946/762.673 µs); gains do not apply uniformly. Supplemental terms are
essentially flat, removing the rejected prototype's 77.0% COUNT regression.

Ordinary deletion-free official before/after/Tantivy (µs):

| Command       |   Before |    After | Tantivy |
| ------------- | -------: | -------: | ------: |
| TOP_10        |  681.258 |  688.527 | 541.047 |
| TOP_1000      | 1201.955 | 1214.734 | 978.355 |
| TOP_100_COUNT | 1347.536 | 1351.739 | 907.653 |
| COUNT         |  570.469 |  570.244 | 411.030 |

The 1.1% ordinary ranked regression is a measured tradeoff: six top-10 passes
and all seven top-1000 passes are slower. Intersection top-10 regresses 2.1%.
Top-100 plus count is 0.3% slower and ordinary COUNT is effectively unchanged.
Supplemental top-10 improves 4.0% (194.748/186.974 µs), but metadata COUNT is
2.4% slower (12.957/13.271); neither establishes a new leaf algorithm. This
change preserves composite batching under filters and does not close the
ordinary ranked performance gap. Previous independent gains are not compounded.

Ordinary official process high-water RSS is before/after/Tantivy
1,270,888/1,270,868/855,832 KiB; supplemental is
491,320/491,308/348,528 KiB. Masked official before/after is
1,270,324/1,270,476 KiB and supplemental 491,436/491,492 KiB. These include mapped
pages and heap. The wrapper adds no scratch; existing scored collection still
allocates its 16 KiB buffer. No memory reduction is claimed.

The verified archive (local archive `benchmark-results/filter-windows-2026-09-15/results.zip`),
native check (local archive `benchmark-results/filter-windows-2026-09-15/native-check.zip`) and
[reproduction notes](benchmark-results/filter-windows-2026-09-15/README.md)
preserve final and rejected source overlays, raw samples, exact oracles,
compiler/source identities and failure/recovery logs. Native validation passes
1789 tests with 25 ignored, all `check` stages and separate portable compilation.
Full RPC/browser tests were not rerun; WASM was not rebuilt under the standing
instruction. One stale cloud upload failed compilation before timing; the
reconstructed bundle matches all final input hashes. API connection failures
required retries during evidence retrieval; they did not restart timed runs.

Remaining work: examine wordwise deletion-mask intersection at its existing
owner to avoid per-hit predicate calls; investigate the masked phrase COUNT
and ordinary ranked regressions; reuse the remaining 16 KiB collector score
scratch; and address larger intersection/phrase/single-term execution costs.
These are proposals. Only a periodic 12.5% mask was timed; clustered/highly
selective filters, masked ranked pruning, cold-cache, concurrent-ingest and
tail-latency measurements remain unrun. Original upstream-runner confirmation
of post-OR changes is also outstanding.

The preserved benchmark machine was stopped after all jobs and verified downloads;
its final cloud status is `TERMINATED`.

### Parity and BMP-equivalent MaxScore reordering — September 15 proposal

The user requested continued work toward Tantivy parity, examination of IResearch
and PISA, and RGB support for MaxScore text following BMP's behavior. The
[reordering design](maxscore-text-reordering.md) records ownership, plain-versus-
chunked scoring invariants, format rejection and merge/publication constraints.
No new gain is established yet; the previous filtered-window build is the control.

Research snapshots: PISA `4af477f227fcbf8a9fbd35b3cee0ecb0e258b285` and IResearch
in SereneDB `6672dde0201981d4aa70447102d9029ccecce04f`. PISA's
[ranked conjunction](https://github.com/pisa-engine/pisa/blob/4af477f227fcbf8a9fbd35b3cee0ecb0e258b285/include/pisa/query/algorithm/block_max_ranked_and_query.hpp)
combines frequency-ordered probes and block bounds. IResearch's
[pruned conjunction](https://github.com/serenedb/serenedb/blob/6672dde0201981d4aa70447102d9029ccecce04f/iresearch/search/top/pruned_conjunction.hpp)
bounds a lead block and filters bounded candidate batches; its
[pruned phrase](https://github.com/serenedb/serenedb/blob/6672dde0201981d4aa70447102d9029ccecce04f/iresearch/search/top/pruned_phrase.hpp)
checks a frequency-derived score bound before positional matching. Summa already
has related paths; these sources guide targeted experiments, not a second executor
or permission to change canonical score arithmetic or equality pruning.

The x86 repeat of score rejection before heap-key comparison improves standalone
term TOP_10 by 8.5% (171.683/157.067 µs) and TOP_1000 by 5.2%
(641.520/608.018), faster in every one of seven passes. Official TOP_10 is
668.409/661.091 µs, but only four passes improve, so its 1.1% aggregate difference
is inconclusive. Supplemental TOP_100_COUNT regresses 1.9% (720.630/734.168),
while its metadata COUNT path varies by -2.8% despite unchanged implementation.
This remains a candidate pending final correctness/source gates and integration
with the RGB work. It does not establish Tantivy parity. Raw data and separate
profiles are retained under `.context/parity/` until final packaging.

### RGB-disabled execution experiments — September 15 continuation

The parity target explicitly uses **RGB disabled**. Physical text reordering is
an independent feature and cannot count toward that target. Current profiles on
unchanged full-corpus indexes attribute 29.1% of COUNT samples to posting bitmap
construction, and 35.0% of standalone top-10 samples to executor dispatch (which
includes heap admission). These are sample shares, not predicted speedups.

The next isolated experiments keep codecs, norms, scoring arithmetic and index
bytes fixed. Inspired by IResearch's bounded posting-batch admission and PISA's
threshold-first heap, screen eight scores before canonical heap admission;
equal/unordered scores still reach Summa's document/ordinal tie comparator.
Separately accumulate sorted document IDs into bitmap words in registers before
writing each word. Scratch remains bounded, no cache or second executor is added,
and a stale score threshold can only admit extra candidates. Both are proposals
until paired x86/ARM timings and exact ranking/position oracles pass.

IResearch's phrase path also prepares its scorer before candidate traversal.
Summa already rejects phrase candidates by frequency bounds before reading
positions, but recalculates numeric admissibility and invariant f64 factors for
every candidate. A separate prototype prepares those factors at the existing
BM25 owner and retains the identical per-frequency envelope and inflation. It
adds one fixed-size value to a phrase scorer, with no cache or allocation;
canonical BM25 scores and pruning-bound bits must remain identical.

A third independent experiment reuses the posting iterator's existing one-step
probe in `TermCursor::seek_prepare` before the SIMD search. The archived x86
call site (`.context/selected-codegen/summa-after-seek_prepare.asm`) sets up
vector comparisons even when the next decoded ID meets the target. The current
block bounds prove a successor exists before the new probe; distant seeks keep
the existing SIMD helper and no buffer or decoder is added. This targets the
9.6% executor-seek sample share, with union regressions explicitly part of the
acceptance decision. The earlier tiled/binary-search experiments remain rejected.

First combined x86 experiment: **reject bitmap word accumulation**. On the
same frozen full corpus and seven rotated passes, COUNT regresses
550.598→620.570 µs (12.7%), with all seven passes slower; unions regress 40.9%.
The added per-hit word-transition branch outweighs fewer bitmap writes.
Standalone TOP10 improves 166.705→150.059 µs (10.0%, all seven passes), and
TOP1000 improves 626.161→606.722 µs (3.1%, all seven). Official TOP10 is only
0.6% faster and remains 1.27× Tantivy. Score admission will be rerun alone;
this combined prototype is not selected. Both 1676-query raw/pruned oracles
match on two layouts, and all index hashes are unchanged. Verified raw samples,
RSS and sources are in `.context/parity/parity-batch-evidence/` pending packaging.

One larger reuse experiment routes all-required text through the newer required
window executor, eliminating the separate unpruned conjunction loop. This is
not the rejected dense-AND prototype: the current required mode always takes
candidates from one selective required posting run and applies existing block
bounds before scoring the rest. PISA's ranked conjunction provides the pruning
motivation. The experiment removes roughly 150 lines and preserves query-order
addition and required membership even at zero frequency. Its tradeoff is larger
bounded scratch: per-term contributions span 4096 IDs instead of a 128-document
conjunction batch. Both exact results and peak residency must be measured; no
selection or default change follows from code size alone.

Primary references for the new experiments: IResearch's
[posting batches](https://github.com/serenedb/serenedb/blob/6672dde0201981d4aa70447102d9029ccecce04f/iresearch/search/detail/posting_batch.hpp),
[admission](https://github.com/serenedb/serenedb/blob/6672dde0201981d4aa70447102d9029ccecce04f/iresearch/search/top/admit.hpp)
and [prepared phrase scoring](https://github.com/serenedb/serenedb/blob/6672dde0201981d4aa70447102d9029ccecce04f/iresearch/search/top/pruned_phrase.hpp);
PISA's [threshold-first heap](https://github.com/pisa-engine/pisa/blob/4af477f227fcbf8a9fbd35b3cee0ecb0e258b285/include/pisa/topk_queue.hpp)
and [ranked conjunction](https://github.com/pisa-engine/pisa/blob/4af477f227fcbf8a9fbd35b3cee0ecb0e258b285/include/pisa/query/algorithm/block_max_ranked_and_query.hpp).
The benchmark's [engine list](https://github.com/quickwit-oss/search-benchmark-game/blob/master/Makefile)
also includes Lucene; its already-adopted and rejected mechanisms are recorded in
[the pinned Lucene review](lucene-11-performance-research.md). Engine rankings alone
are not an explanation of Summa's costs.

The first score-batch x86 binary contains an eight-float vector comparison
(`vcmpngtps`) followed by scalar canonical admission for survivors. Compiler
output also includes a mask for the possibly short last chunk. The isolated
score-only repeat uses fixed eight-element chunks and a scalar tail to avoid
that mask. This uses portable Rust; no ISA-specific unsafe kernel was added.

Prepared phrase bounds are rejected: x86 official TOP10 changes
666.861→664.223 µs (0.4%, five of seven passes faster), with phrase TOP10
only 0.8% faster and phrase TOP1000 flat. ARM is essentially flat. The extra
prepared value is not justified by a repeatable material target-path gain.
The ARM one-step seek run is inconclusive; its small differences are about the
same size as variation in unchanged COUNT. No seek change is selected pending
a corrected x86 run.

**Provenance correction:** the first queued x86 `seek` and `required` attempts
preserved old source mtimes with `shutil.copy2`, so Cargo reused the previous
release binaries. Their build logs finish in 0.36/0.35 seconds. The seek binary
hash equals the bounds candidate; the required binary hash equals the mask
candidate. Those timings are invalid evidence for the named source changes,
even though their score/index oracles match. Provisional seek/AND conclusions
from them are withdrawn. They remain archived as failed experiment attempts.
The ARM candidates, x86 batch/bounds/mask, and final integrated sources did
compile afresh. Corrected x86 recipes rewrite the complete control overlay with
fresh timestamps, verify its source manifest, and require an actual core rebuild
before any oracle or timing run.

### Integrated score admission with RGB disabled — verified final source

The selected source uses the scalar threshold guard plus a fixed-eight score
mask for single-text runs. Canonical heap admission handles survivors, including
equal scores, signed zero and unordered scores. Predicate and document-mapped
runs keep their scalar boundary. It adds no scratch allocation, cache or codec.
The 332-file integrated source includes the RGB tie-order correction, but every
latency fixture has RGB disabled. Both 1676-query raw/pruned/count oracles match
on the full-corpus layouts; all four ARM layouts also match and all index hashes
are unchanged. The integrated core actually rebuilt before timing.

On the same 5,032,104-document x86 fixture, previous/current/Tantivy official
TOP10 is 668.198/659.236/530.903 µs (1.3% faster, six of seven passes), TOP1000
1178.144/1173.096/938.324 (0.4%, seven), TOP100_COUNT
1276.084/1285.205/860.477 (0.7% slower, five slower passes), and COUNT
558.439/558.654/404.941 (flat). Current/Tantivy is **1.24×, 1.25×, 1.49× and
1.38×** respectively. Parity remains unmet; ratios from earlier sessions must
not be used as the control for this change.

Supplemental standalone TOP10 is 179.182/142.964/50.939 µs (20.2% faster) and
TOP1000 is 669.933/600.089/432.319 (10.4% faster), with all seven passes faster.
Supplemental TOP100_COUNT regresses 2.7% (767.776/788.648, all seven slower);
metadata COUNT is 13.286/13.170, a 0.9% movement in the unchanged dictionary
shortcut. The gain is retained with this explicit scored-count tradeoff.
The official workload has only one standalone-term query, so these term gains
do not imply a comparable improvement in the official mix. The masked
prototype's isolated official scored-count regression is not compounded into
this final comparison.

ARM official TOP10 is 32.272/32.253 µs on the canonical fixture and
49.216/49.180 on the merged fixture: effectively flat. Supplemental TOP10 is
20.190/19.417 and 24.540/23.629, improvements of 3.8% and 3.7%. Other small
unchanged-path movements are inconclusive on the shared development machine.
The inspected x86 fixed-width mask partially vectorizes; ARM emits an unrolled
scalar comparison sequence. The integrated x86 dispatcher grows from 8,195 to
9,480 machine-code bytes, including the RGB mapping boundary; this is not a
mask-only code-size attribution. The implementation uses portable Rust, without
an ISA-specific unsafe kernel or changed dispatch default.

X86 peak RSS previous/current/Tantivy is 1,031,408/1,031,236/587,764 KiB on the
official workload and 349,820/350,316/235,520 KiB on the supplemental workload.
These are process mapped-plus-heap peaks across four warm commands, not heap
residency. No memory reduction is claimed. The full native harness passes 1,798
tests including real-server broker tests, with 25 ignored in the standard
stage, plus formatting, Clippy, native-without-sync, portable compilation and
API documentation. WASM is not rebuilt under the standing user instruction.

Remaining performance work centers on intersection/phrase execution, scored
counts and the scalar norm-gather cost. Additional bound preparation and bitmap
branching did not justify production complexity. Cold-cache, concurrent-ingest,
tail latency and original upstream-runner confirmation remain unrun. RGB
merge-time planning, legacy standalone migration and the full budget/corruption/
cancellation audit remain unfinished and separate from the RGB-disabled target.

The corrected x86 **seek** experiment rebuilt the core in 2m23s and produced a
different candidate hash (`bb70003399b96d233e7f79a9f58191fd50cd5b4b47b3f9c688081f63e160454f`).
All source hashes, both exact oracles and unchanged index manifests pass.
Official TOP10 is 656.162/661.658 µs (0.8% slower, six of seven passes slower),
TOP1000 1166.774/1174.201 (0.6%, all seven), and intersection TOP10 regresses
2.1% in every pass. The extra next-ID probe is rejected. COUNT also moves 2.3%
slower in the artifact comparison; this is not evidence of a changed COUNT
algorithm. This corrected run, not the stale first attempt, supports rejection.

The corrected x86 **all-required windows** experiment also rebuilt (2m25s),
with candidate hash `7fd939d719218cb5bf2dc4e8a4f25690912c013f4c6fc78a112c78e0b5c98a6f`.
Both exact oracles, its complete source manifest and immutable index checks pass.
Official TOP10 is 660.280/680.325 µs (3.0% slower) and TOP1000
1167.558/1306.868 (11.9% slower), with all seven passes slower. Intersection
TOP10 regresses 9.1% and TOP1000 43.0%, also in every pass. Together with the
ARM intersection regressions and larger per-term scratch, this rejects the
window-reuse prototype. No architecture or document-frequency cutoff is added
from these fixtures. Supplemental single-term TOP10 is essentially flat, as
expected for this source change. The original cached-binary attempt is retained
only as invalid provenance evidence; it does not support an AND speedup.

The verified parity archive (local archive `benchmark-results/parity-2026-09-15/results.zip`),
full native checks (local archive `benchmark-results/parity-2026-09-15/native-check.zip`) and
[reproduction notes](benchmark-results/parity-2026-09-15/README.md) preserve all
selected, rejected and invalid attempts, with explicit identities and corrected
rebuild recipes. Every transferred archive and packaged entry is hash-verified.

The owned benchmark machine is confirmed `TERMINATED` after all runs and verified
downloads. The stop command's status polling encountered a connection reset;
a subsequent direct status check confirmed shutdown. Changes remain uncommitted.

## AND block-bound experiment (September 15, follow-up)

Proposal, not selected: retain the semantic conjunction's rarest-first cursor
alignment and 128-hit canonical scoring batches, but shallow-seek existing text
block metadata before alignment once the heap is full. Sum existing conservative
BM25 block bounds and retain their minimum document endpoint. A strictly
noncompetitive sum permits advancing the lead beyond that endpoint; equality
still participates in the canonical score/document/ordinal ordering. Reuse the
executor's relative floating-point threshold margin and existing bound cache.
Pending hit batches may only make the heap threshold stale in the conservative
direction. Count collectors continue through their existing exhaustive path.

The public planner routes native and async in-memory text AND queries through
this same executor. Posting lists own the persisted block metadata and decoders;
no format, writer, scoring formula, query default, or RGB policy changes. Extra
state is one document endpoint and one score, with O(terms) metadata probes per
crossed overlapping block span, bounded by existing query-term limits. This
trades metadata work for avoided decoding, alignment and scoring. Whole-workload
ARM/x86 timing and exact ordered score-bit/count oracles determine selection.

This follows the shallow block-bound pruning in
[PISA ranked AND](https://github.com/pisa-engine/pisa/blob/4af477f227fcbf8a9fbd35b3cee0ecb0e258b285/include/pisa/query/algorithm/block_max_ranked_and_query.hpp)
and [Lucene BlockMaxConjunctionScorer](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/BlockMaxConjunctionScorer.java).
It is distinct from the rejected dense required-window replacement: the current
bounded 128-hit rows and canonical batch scorer remain in place.

Fresh family profiles use the preserved selected binary, RGB off, the same
5,032,104-document rounded fixture and CPU 2. AND TOP_10 self samples: cursor
seek preparation 33.42%, conjunction loop 11.84%, checked document decode 8.88%,
byte-gap SIMD decode 7.26%, seek wrapper 6.99%. Phrase TOP_10: candidate alignment
26.55%, candidate score bound 13.96%, confirmation 7.62%, checked document decode
6.59%, cached position reads 5.90%, position lookup 4.86%. These sample shares
identify where this binary spends CPU time, not predicted speedups or latency
measurements. DWARF call stacks contain unresolved frames; self symbols are used
for attribution. Hardware cycles, instruction, branch and cache counters are
unsupported on this machine, so the profiles do not establish a cache-miss rate or
cycles per decoded posting. No page faults occurred during the sampled AND run.

A separate proposed seek experiment narrows the iterator's decoded suffix by
exponential probes before its existing binary search. The current/next probes
remain. The search interval grows with the distance actually skipped, rather
than always searching the whole remaining block. This affects the owning
`BlockPostingIterator` (phrase and ordinary membership), not the unified
MaxScore cursor's SIMD seek; that cursor's added next-ID probe was already
rejected. No storage bytes, buffer sizes, frequency-prefix rules, backward RGB
probe semantics, or out-of-block directory seek change. Near gaps may require
fewer comparisons; distant gaps add probes. Both distributions and whole-query
cost must be measured before selection.

One batching interaction needs separate measurement: the initial conjunction
batch contains 128 hits even for top-10, so pruning has no threshold until that
batch is collected. A refined candidate fills the first batch at `min(k, 128)`
(and respects any existing collector entries), then resumes 128-hit batches.
The trigger is heap capacity, not a corpus/architecture cutoff. Canonical scoring
and tie order remain unchanged; a partially filled heap never permits pruning.

A further isolated phrase prototype borrows both posting cursors once for a
two-term phrase's alignment loop. The general loop currently follows a sorted
term-order vector and re-indexes the cursor vector on each probe and restart.
Tantivy's typed two-leading-cursor intersection motivates removing that
indirection for this common query shape. The same rarest-first leapfrog and
budget check apply; term identities, position streams and canonical scoring stay
in original query order. Longer phrases retain the existing loop. This is an
execution specialization, with no allocation or storage change, and must earn
its extra code through integrated measurements independently of galloping seeks.
The reference is Tantivy 0.26's
[typed intersection](https://github.com/quickwit-oss/tantivy/blob/0.26.0/src/query/intersection.rs).
Its `TermScorer` does not override `seek_danger`; the trait default delegates to
ordinary seeking. A membership-only seek shortcut is therefore not evidence for
this benchmark version and is not part of this prototype.

### Completed initial block-bound trial — rejected

Seven rotated x86 passes on the frozen full corpus (µs geometric means):

| Command       |   Before | Candidate | Tantivy | Candidate / before |
| ------------- | -------: | --------: | ------: | -----------------: |
| TOP_10        |  669.572 |   695.244 | 522.218 |             1.0383 |
| TOP_1000      | 1177.506 |  1201.754 | 933.929 |             1.0206 |
| TOP_100_COUNT | 1292.859 |  1281.244 | 868.158 |             0.9910 |
| COUNT         |  563.951 |   554.271 | 402.195 |             0.9828 |

AND TOP_10 regresses 12.1% and TOP_1000 6.6%, with all seven passes slower.
The conjunction gained no clear ARM improvement: canonical AND TOP_10 is flat
and merged is 1.2% slower. Small improvements in unchanged count/phrase paths
do not establish a pruning benefit. The baseline implementation is restored.
Both full-corpus layouts and all four ARM layouts preserve every ordered
document/score-bit/count oracle across 1,676 queries. The skip/late-winner
regression fails on baseline and passes on the prototype; all 42 focused scoring
tests pass on both architectures. Synthetic pruning is real, but the integrated
metadata/skip tradeoff is unfavorable.

A final isolated structural candidate replaces the duplicated ISA-specific
`find_first_ge_u32` scans with one portable integer reduction: count values below
the target in each eight-value sorted run, stop on a count below eight, then
binary-search the at-most-seven-value tail. In sorted order the count is exactly
the lower-bound offset. This removes the unsafe NEON/SSE search implementations
and feature dispatch without changing the helper's contract or memory budget.
Native compilers may vectorize the integer comparisons; portable builds remain
correct without SIMD. Existing exhaustive tests cover every length through 140,
duplicates, unsigned sign-bit crossings and terminal IDs against partition_point.

The current x86 `seek_prepare` assembly uses 128-bit compares/mask transfers
plus sign-bit XORs. The prototype's unsigned comparisons give the compiler a
simpler expression; ARM assembly confirms two four-lane comparisons per run.
This is code-generation evidence only. It must pass actual whole-query tests on
both architectures; the earlier rejected tile/next-probe/binary-search variants
are not evidence that this replacement is faster.

### Earlier first heap threshold — rejected

The first-batch refinement also loses. Seven-pass x86 official screening:

| Command  | Family       | Before (µs) | Candidate (µs) | Candidate / before |
| -------- | ------------ | ----------: | -------------: | -----------------: |
| TOP_10   | all          |     663.885 |        704.118 |             1.0606 |
| TOP_10   | intersection |     571.496 |        678.529 |             1.1873 |
| TOP_1000 | all          |    1202.249 |       1228.994 |             1.0222 |
| TOP_1000 | intersection |     666.396 |        707.271 |             1.0613 |

Both commands and their AND subsets are slower in every pass. ARM AND TOP_10
regresses 2.0% canonical and 3.75% merged. All six layout/corpus exact oracles
and focused scoring tests pass. Establishing the heap threshold earlier does
not rescue the block-bound cost; no conjunction-pruning code is retained.

### Distance-bounded iterator seek — completed screening

| Command | Family       | Before (µs) | Candidate (µs) | Candidate / before | Faster passes |
| ------- | ------------ | ----------: | -------------: | -----------------: | ------------: |
| TOP_10  | all          |     677.182 |        676.853 |             0.9995 |           3/7 |
| TOP_10  | union        |     721.652 |        725.576 |             1.0054 |           2/7 |
| TOP_10  | intersection |     581.149 |        581.489 |             1.0006 |           3/7 |
| TOP_10  | phrase       |     646.858 |        642.918 |             0.9939 |           5/7 |
| COUNT   | all          |     575.694 |        568.017 |             0.9867 |           7/7 |
| COUNT   | union        |     785.623 |        744.684 |             0.9479 |           7/7 |
| COUNT   | intersection |     510.571 |        515.573 |             1.0098 |           1/7 |
| COUNT   | phrase       |     734.608 |        730.830 |             0.9949 |           6/7 |

All four ARM and both full-corpus layout oracles pass for 1,676 queries; index bytes remain unchanged. These two-command screens do not establish performance for top-1000, top-100 plus count, or supplemental queries.

ARM overall count improves 1.5% on both fixtures; phrase top-10 improves 1.4% canonical and 2.8% merged. x86 top-10 is flat; union count improves 5.2%, while AND count regresses 1.0%. The 212 posting tests pass on both architectures.

### Borrowed two-cursor phrase alignment — completed screening

| Command | Family       | Before (µs) | Candidate (µs) | Candidate / before | Faster passes |
| ------- | ------------ | ----------: | -------------: | -----------------: | ------------: |
| TOP_10  | all          |     675.931 |        671.053 |             0.9928 |           4/7 |
| TOP_10  | union        |     720.965 |        718.990 |             0.9973 |           3/7 |
| TOP_10  | intersection |     580.480 |        581.565 |             1.0019 |           2/7 |
| TOP_10  | phrase       |     646.229 |        633.717 |             0.9806 |           7/7 |
| COUNT   | all          |     566.335 |        562.380 |             0.9930 |           6/7 |
| COUNT   | union        |     748.883 |        760.765 |             1.0159 |           0/7 |
| COUNT   | intersection |     505.572 |        504.906 |             0.9987 |           5/7 |
| COUNT   | phrase       |     734.868 |        720.751 |             0.9808 |           7/7 |

All four ARM and both full-corpus layout oracles pass for 1,676 queries; index bytes remain unchanged. These two-command screens do not establish performance for top-1000, top-100 plus count, or supplemental queries.

ARM phrase top-10/count changes are small and essentially flat. x86 phrase top-10 and count improve 1.9% in every pass, but overall top-10 is only 0.7% faster with four of seven passes faster. The 16 phrase tests pass on both architectures.

### Portable reduction rejected; explicit AVX2 follow-up

The portable count reduction regresses x86 overall TOP_10 4.7% and AND 8.4%,
with all seven passes slower. ARM mixed required/optional top-10 regresses
7–9%. All tests and exact oracles pass, but the shorter source is not selected.
Portable compilation passes with the existing no-native dead-code warning for
`ChunkMapBuilder::set_document_units` (native reordering writers use it).

Actual integrated x86 disassembly explains the reduction's cost: LLVM generates
an eight-lane unsigned comparison, then seven mask shifts, eight mask transfers,
per-lane scalar masking and an addition tree. It does not lower the integer sum
to the compact mask count seen in the standalone ARM probe. This is a concrete
code-generation reason to test an explicit packed-mask backend, not to assume
that the portable expression will optimize well.

Proposed AVX2 follow-up: scan eight sorted IDs with unsigned SIMD minimum,
equality and one mask extraction, returning the first set lane. The existing
SSE2 implementation serves the short tail and CPUs without AVX2. The existing
ARM NEON and portable fallback remain unchanged. Runtime feature dispatch uses
the module's existing AVX2 capability check; all loads stay inside the supplied
slice. No new executor, allocation, bound, cache or on-disk representation.
This candidate builds on the integrated phrase/iterator candidate and compares
against both that exact binary and the original frozen baseline, plus Tantivy,
across the full official/supplemental four-command matrix.

### Integrated phrase/galloping build — completed

The two retained portable traversal changes were rebuilt together before testing
AVX2. Their full four-command official/supplemental matrix is archived as
`and-phrase-final-evidence`. Against the frozen starting build, official
TOP_10 is 662.015 → 657.533 µs (-0.7%), TOP_1000 1169.489 → 1157.446 (-1.0%),
TOP_100_COUNT 1284.642 → 1281.412 (-0.3%), and COUNT 563.823 → 564.841 (+0.2%).
Phrase improves 1.9–2.9% in every pass, but AND count regresses 1.8% and union
count regresses 1.3%. The standalone galloping screen's 1.3% overall count gain
does not survive this integrated measurement and is not claimed as retained.
ARM top-10 improves 0.4–0.8% and count 1.8%. Exact oracles, source identities,
immutable index hashes and the 1,794-test native check all pass.

### Packed-mask AVX2 seek — selected after full comparison

The final four-engine run rotates the original frozen build, the exact combined
phrase/galloping binary, the AVX2 candidate and Tantivy. All four commands run
on both 962 official and 714 supplemental queries, seven complete passes each,
after ten seconds of warmup per engine/command. All source overlays contain
332 verified files and every candidate actually recompiles core. The main tree
matches the selected AVX2 source manifest exactly.

| Operation             | Starting build (µs) | Selected build (µs) | Tantivy (µs) | Change | Selected / Tantivy |
| --------------------- | ------------------: | ------------------: | -----------: | -----: | -----------------: |
| Top 10                |             668.690 |             647.571 |      529.436 |  -3.2% |          **1.22×** |
| Top 1000              |            1184.824 |            1154.399 |      939.181 |  -2.6% |          **1.23×** |
| Top 100 + exact count |            1273.917 |            1267.541 |      852.847 |  -0.5% |          **1.49×** |
| Exact count           |             564.696 |             562.863 |      404.698 |  -0.3% |          **1.39×** |

These changes are measured against the original binary in the same run, not
compounded from the separate screens. Incrementally against the two-change
build, AVX2 improves official TOP_10 and TOP_1000 1.9% in all seven passes,
including AND improvements of 5.4% and 5.1%. Incremental overall TOP_100_COUNT
(+0.2%) and COUNT (+0.3%) regress slightly. Against the original build, AND
TOP_100_COUNT regresses 1.6% and COUNT 2.4%, all seven passes slower; phrase
improves 2.4–2.9% across all commands, all seven passes faster.

Supplemental TOP_10 is 146.747 → 146.503 µs, TOP_1000 582.987 → 583.846,
TOP_100_COUNT 783.677 → 771.585 and COUNT 13.435 → 13.478. Most standalone
movements are flat. Against the intermediate build, TOP_10 regresses 1.3%
and COUNT 0.9%; do not attribute unchanged-path movements entirely to AVX2.

Actual selected `TermCursor::seek_prepare` disassembly uses an eight-lane
`vpcmpleud`, `kortestb`, then one mask transfer and `tzcnt`. Native Cascade Lake
flags let LLVM use AVX-512VL for the AVX2 source expression. Unlike the rejected
portable sum, it has no eight-element scalar reduction tree. This is measured
native-target code generation, not a claim about older AVX2-only CPUs.

Selected-source ARM official TOP_10 improves 1.0% on both rounded fixtures;
COUNT improves 2.3% canonical and 1.8% merged. ARM uses the existing NEON kernel.
Full-corpus original/selected/Tantivy peak RSS is 1,031,372 / 1,030,912 /
587,784 KiB official and 349,236 / 349,328 / 235,608 KiB supplemental. Memory
is effectively unchanged and the three changes introduce no allocations.

The final native check passes 1,794 tests (25 ignored), formatting, Clippy and
native without sync. The selected x86 source passes all 1,578 core tests
(16 ignored), all-target Clippy, both 1,676-query exact oracles and index-byte
checks. All four ARM oracles and 40 immutable index-file hashes pass. Portable
compilation passes with the existing no-native `set_document_units` warning.
WASM is skipped under the standing user instruction. Lifecycle/RPC checks were
not rerun because these changes do not alter lifecycle or wire semantics.

The final comparison remains 1.22× Tantivy for top-10 and 1.39× for count.
Parity and the separate RGB lifecycle/merge audit remain open. Future dispatch
work must retain shared CPU admission across concurrent sync callers; simply
bypassing the shared Rayon pool would change that policy and needs its own
lifecycle tests. Full profiles, all rejected trials, exact source overlays and
raw final results are in the
[verified follow-up evidence](benchmark-results/and-phrase-2026-09-15/README.md).

The first final-source native check hit two broker backend-registration timeouts
(`client_deadline_propagates_and_absence_means_untimed` and
`dead_backend_is_evicted_and_recovers`, both waiting ten seconds for mock index
registration). All 13 broker integration tests passed on retry; the complete
check was then repeated. Both attempts and the focused retry are archived.
The cloud stop command lost its polling connection, and a direct describe
confirmed `TERMINATED`; no shutdown success is inferred from that failed poll.

## Decoded-block conjunction intersection — proposal (2026-09-15)

The current ranked two-term AND loop repeatedly calls general cursor seek and
reads cursor/block state for each candidate. The earlier profile attributes
33.4% of starting-build AND self samples to seek preparation. The selected SIMD
backend reduces its kernel cost but retains this per-candidate protocol.

Proposed: intersect the two already-decoded posting slices within the existing
conjunction collector, using local offsets until either block or the 128-hit
batch ends. General checked seek/decode still owns cross-block navigation and
corrupt-payload errors. Decode TFs only on the first eligible hit of each block
overlap, then gather directly into the existing term-identity rows. Keep the
canonical score-order batch reduction, predicates, document mapping, cancellation
and fixed scratch budget unchanged. Larger conjunctions keep their existing
loop; no new scorer, writer, executor, cache or persisted format is introduced.

Invariant: both cursors resume at the first unconsumed posting, every eligible
intersection appears once, TF rows keep original cursor identity, and all raw
score bits and ordered top-k results equal exhaustive scoring. General native
and async semantics remain equivalent; portable code retains its current path.
Cost: compare loaded IDs directly, check block state only at block/batch
boundaries, with the same O(postings) upper bound and no extra allocation.
This is a hypothesis until paired ARM and full-corpus x86 measurements pass.

### Complete impact bound before loose fallback bounds — proposal

An independent candidate tests the existing complete impact envelope first in
`TermCursor::text_block_bound` and `text_group_bound`. A finite result is already
a conservative bound on canonical scoring under its validated parameters; in
that case there is no correctness need to also compute max-TF/min-length and
ratio bounds. Unknown envelopes and unsupported/nonfinite parameters retain the
existing fallback. The rounding guard and all actual score arithmetic remain
unchanged. The earlier standalone-term profile attributed 29.0% self time to
`text_block_bound` and 12.3% to envelope geometry; this identifies redundant
work worth measuring, not a guaranteed gain for the current source.

The selected bound may be slightly looser than the minimum of three individually
safe bounds; that can admit extra candidates, never discard a true top-k hit.
No format/default/cache/allocation change. Compare against the preserved AVX2
baseline independently of the block-intersection experiment before integration.

### Exact-candidate capability for Boolean confirmation — proposal

Count and complete Boolean collection align candidate cursors and then call
`confirm_candidate` on every required child. Plain term children already yield
exact postings: their default confirmation only calls `doc` again to reject
TERMINATED, although the parent has just established a nonterminal match.
This adds virtual calls per matched document, especially for count-heavy AND.

Proposed `Scorer::requires_candidate_confirmation` defaults to true, preserving
all custom/two-phase implementations. Only ordinary term, fast-field text and
Boolean scorers opt out; their candidate methods are exact and confirmation has
no side effects. `BooleanScorer::initialize` caches one boolean from its children
and skips the verification loop only when every required child opts out. Mixed
phrase/custom children retain the complete verification protocol. Consumers
still reject TERMINATED before using the capability. No additional allocation,
format, executor or lifecycle change; cancellation checks stay at their existing
owners. Native and async construction share the same initialization.

Measure independently against the preserved AVX2 baseline, including generic
Boolean/phrase regressions, full exact oracles and the four-command matrix.

The distinction is also explicit in [Lucene's Scorer API](<https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/search/Scorer.html#twoPhaseIterator()>): two-phase iteration is optional and intended for scorers with expensive confirmation. Summa keeps a conservative default so existing custom candidate methods still receive verification.

Pre-benchmark review found that term and fast-field `doc()` also observe timed
query cancellation. The first prototype was withdrawn before application or
latency measurement, despite its library tests passing. A new
`timed_term_candidates_keep_deadline_confirmation` regression pins this boundary:
only untimed leaf scorers may opt out; timed leaves keep confirmation enabled.
The corrected capability checks `budget.is_some()` once during parent setup.

### Compact membership batches for sparse AND counts — proposal

The existing Boolean count path uses document windows only when estimated
membership density amortizes a bitmap. Sparse AND falls back to virtual
per-document seek/advance/confirmation. Test a bounded `DocBatch` of 128 sorted
IDs in the existing DocSet/Boolean/collector owners: copy the lead's decoded
runs, retain candidates in each required child, then restore the parent's first
unconsumed exact match. Posting iteration owns the block-aware copy and retain
kernels; frequencies remain deferred. Dense windows keep their existing priority.

The capability is opt-in and defaults to scalar exact traversal. Untimed term
scorers and pure compatible conjunctions opt in. Timed or unsupported children,
filters, exclusions, scoring and positions retain their existing path. Count
collection checks its budget before and after a batch and discards a batch that
expires, matching the existing window contract. Additional scratch is one
128-ID array (512 bytes), with no cache or allocation. No representation,
scorer, executor or scoring semantics change. Tests must cover resume state,
frequency/position validity, empty/tail batches, all posting codecs and expiry
before count publication, plus the complete ARM/x86 exact oracles.

### Impact-first bound selection — rejected after measurement

All correctness gates passed, but the complete seven-pass x86 matrix does not
support retaining the simpler bound choice. Official TOP_10 changes
641.619 → 642.734 µs (+0.2%, only 4/7 passes faster), TOP_1000
1133.201 → 1138.231 (+0.4%, 2/7 faster), TOP_100_COUNT
1276.886 → 1281.436 (+0.4%, 2/7 faster), and COUNT
561.979 → 559.311 (-0.5%, 5/7 faster). Supplemental TOP_10 improves only
0.7%, while TOP_1000 regresses 0.8%. ARM changes are similarly small.
The original minimum of all supported bounds remains selected. Both complete
source manifests, all raw timings and exact oracles are preserved under the
`impact-first` experiment; the archive has 37 verified evidence members.

### Cached position span expansion — proposal

Phrase scorers reach `TermPositionCursor` through the same postings owner on
native, async and portable builds. `PositionStream::read_cached` can expand a
whole document directly from its existing decoded block when the complete TF
span is resident. Keep global checked range validation before this shortcut;
misses and spanning documents retain the validated block traversal. Both paths
share one delta-prefix helper. No extra cache, allocation, format or writer.

The cost hypothesis is avoiding a block lookup and general traversal setup per
cached phrase document. Existing tests cover short copied blocks, forward and
backward reads, spanning documents, failed cache replacement and encoded byte
identity; extend the cached final-block test with an invalid span followed by a
valid read. Measure this independently against the preserved AVX2 baseline on
both architectures before deciding whether to retain it.

### Packed rejection masks for exhaustive top-k collection — proposal

The ranked single-term executor already screens eight scores before heap
admission. Exhaustive `TopKCollector::collect_score_block` still performs total
float ordering and tie comparisons for every set bit. Test sharing the existing
eight-score mask kernel between both owning collectors: numerical scores below
the current full-heap floor can be excluded as a group, while equal and unordered
scores retain the canonical total-order comparison. A threshold that rises during
the block only admits extra candidates. Count publication uses the original
membership word, before screening; custom and position collectors keep their
existing path. No scoring arithmetic, format, extra scratch allocation or cache.

Use the existing exceptional-float, signed-zero, tie, late-winner and exact-count
reference test and complete score-bit oracles. The isolated candidate must improve
ARM before adding another full-corpus cloud experiment. Sparse words may make
packed screening more expensive, so retain a one-candidate path for each byte.

### Decoded two-cursor intersection — retained for combined measurement

The isolated seven-pass full-corpus comparison reduces official TOP_10
647.225 → 624.799 µs (-3.5%) and TOP_1000 1140.125 → 1109.534 (-2.7%),
with all passes faster. AND improves 543.119 → 485.841 (-10.5%) and
618.978 → 569.843 (-7.9%), also in every pass. TOP_100_COUNT is flat
1277.100 → 1279.687 (+0.2%); COUNT changes 559.805 → 550.553 (-1.7%),
though the count implementation is unchanged and AND count is 0.4% slower.
Do not attribute the unchanged-path movement entirely to this algorithm.

The disclosed tradeoff is supplemental TOP_1000 +2.0% in every pass and
TOP_10 +0.9%. TOP_100_COUNT improves 2.3%, COUNT is essentially flat.
Canonical/merged ARM ranked AND improves 2.5%/3.1%; official TOP_10
improves 1.7%/1.8%. All four ARM and both x86 1,676-query exact ordered/raw
score/count oracles pass, with identical index bytes. Full x86 core tests and
Clippy pass. Official peak RSS is 1,024,224 → 1,024,396 KiB; no allocation
is added. The candidate is retained for integration, not yet a combined result.
Its isolated TOP_10 remains **1.20× Tantivy**; the performance goal remains open.

### Exact-candidate confirmation capability — rejected after measurement

The corrected candidate passes all library, cancellation, score-bit, count and
immutable-index gates on both architectures. The full x86 comparison nevertheless
rejects it: official TOP_10 646.377 → 643.919 µs (-0.4%, 4/7 faster),
TOP_1000 1142.328 → 1138.590 (-0.3%, 4/7), TOP_100_COUNT
1276.818 → 1308.140 (+2.5%, 6/7 slower), COUNT 563.802 → 562.497
(-0.2%). AND COUNT improves 1.6% in every pass, but AND TOP_100_COUNT is flat.
Supplemental TOP_100_COUNT improves 1.5%; other movements are small.

This does not justify adding a capability and cached Boolean state. The existing
confirmation protocol stays in production. Preserve the withdrawn initial source,
the deadline regression's red/green logs and the corrected complete experiment;
no timings came from the withdrawn version. All 37 cloud evidence members were
downloaded and hash-verified before this decision.

### Exhaustive scored-block masks — rejected on ARM

Both complete ARM matrices and all four exact oracles pass. The target workload
fails the performance screen: supplemental TOP_100_COUNT regresses 1.7% on the
canonical layout and 0.8% on the copied layout. Official TOP_100_COUNT is flat
(-0.5%/+0.1%). The small ranked and count movements are on unchanged paths.
Do not add the shared mask helper or schedule a full-corpus run for this source.
The original block collector remains selected; all 1,577 core tests passed.

### Shared SIMD lower bounds in ordinary posting cursors — proposal

The new packed-mask AVX2 kernel improved the ranked `TermCursor`; phrase and
complete Boolean traversal still use `BlockPostingIterator`'s scalar binary
search. Test reusing the existing sorted-ID kernel at its three lower-bound
sites: inside the gallop-bounded suffix, after loading a later block, and during
a point probe. Keep nearby current/next probes, galloping, directory navigation,
TF deferral, position prefixes and corruption propagation unchanged. This is
three call substitutions, with no new kernel, state, allocation, API or format.

The kernel already has native SIMD and portable implementations with identical
unsigned lower-bound semantics. Existing all-codec mixed seek/advance/point,
frequency/position and boundary tests plus four ARM and two full-corpus exact
oracles gate the substitution. ARM and x86 measurements decide selection.

### Intersect decoded phrase posting blocks — proposal

The ranked two-cursor experiment establishes a measured benefit from doing
alignment inside loaded posting slices. Apply the same responsibility boundary
to two-term phrase alignment: `BlockPostingIterator` owns a bounded seek to the
next common document, compares local offsets inside both decoded blocks, and
commits valid cursor positions before returning a match. Only an exhausted
block or a target beyond its last ID re-enters general seek/decode. Phrase
confirmation and frequency remain in `PhraseScorer` in original term order.

The query supplies the existing stop check as a closure; the structure owner
knows no deadline policy. Stop checks remain inside alignment and return no hit
with valid cursor state. No frequencies or positions decode until their ordinary
consumer asks. No allocation, new scorer, result cache, representation or writer.
The first unconsumed intersection, resumed TF/position prefix, exhausted and
cancelled states must match scalar traversal on every codec. Existing phrase
multiplicity, backward point probe and deadline tests remain gates. Measure
independently from the SIMD call substitution and integrate only after exact
ARM/x86 oracles and whole-query timing pass.

### Compact sparse count batches — retained for combined measurement

The full-corpus x86 comparison improves official COUNT 566.095 → 527.791 µs
(-6.8%) and AND COUNT 518.010 → 410.291 (-20.8%), in all seven passes.
TOP_10 646.961 → 647.112, TOP_1000 1134.324 → 1135.247 and
TOP_100_COUNT 1282.811 → 1282.357 are essentially flat. The candidate's
COUNT remains **1.30× Tantivy**, whose same-run mean is 405.244 µs.

Supplemental single-term COUNT regresses 13.286 → 13.619 µs (+2.5%) in
all seven passes. That metadata-only shortcut does not use the new batches;
the measured regression remains disclosed. Supplemental ranked changes are
small (+0.7% TOP_10, +0.4% TOP_1000, +0.1% TOP_100_COUNT). All four ARM
and both x86 exact oracles and immutable index checks pass. ARM runs 1,580
core tests and x86 runs 1,581, with 16 ignored on each architecture; x86
all-target Clippy passes. Retain this source for a combined fresh measurement
with the ranked block intersection; do not multiply isolated percentages.

### Decoded phrase intersection helper — rejected on ARM

All 1,578 core tests, four exact oracles and both complete timing matrices pass.
Phrase TOP_10 is 0.6% slower canonical and 0.1% slower merged; phrase COUNT
is 0.6%/1.2% slower. TOP_1000 is flat to 0.7% faster. Other, unchanged query
families improve 1–5% in the same run, so the overall movement is not a phrase
algorithm gain. The binary has no out-of-line intersection helper symbol; merely
adding an inline hint would not address the measured result. Reject the larger
helper and retain the existing phrase alignment. No cloud run was scheduled.

### Compact scored conjunction batches — proposal

The retained compact count path removes 20.8% of x86 AND count time, but complete
ranked AND collection still scores each match through child virtual calls.
Extend the existing 128-ID batch protocol with exact score rows in `Scorer`.
The posting owner copies or retains matching frequencies using the same bounded
kernels as count (a compile-time frequency flag leaves count document-only).
Untimed unit-boost term scorers then call the existing canonical text-run scorer.

A pure compatible Boolean conjunction fills a selective lead batch and retains
memberships through its children. Preserve an original lead-row index while
compacting, and add each child's complete score in original query order. This
preserves floating sums when the lead is not the first clause and when a child
is nested. Existing dense score windows retain priority. Compact scoring is used
only for sufficiently large compatible conjunctions; unsupported, timed, boosted,
filtered and position consumers keep existing behavior. Defaults are conservative.
No new scorer, executor, BM25 formula, heap, cache or format is introduced.

Scratch is fixed to 128 candidates per active scorer: about 4.2 KiB for a flat
conjunction including its leaf and collector, plus about 1.2 KiB per active nested
conjunction. The server already caps query depth at 32. The collector checks
cancellation before and after each batch and before
per-hit publication. Tests cover score bits, missing lengths, zero frequencies,
lead-order cancellation in floating addition, nested sums, resumption and expired
batch discard. This source builds on the selected count-batches candidate and
is measured against that exact binary, not against the AVX2 starting build.
A final integrated comparison must still use the original AVX2 baseline/Tantivy.

### Whole-span position cache shortcut — rejected after x86 measurement

The full x86 comparison finds no material target-path gain: phrase TOP_10
620.749 → 621.947 µs (+0.2%), TOP_1000 742.569 → 743.810 (+0.2%),
TOP_100_COUNT 751.340 → 748.743 (-0.3%) and COUNT
711.844 → 704.984 (-1.0%, five of seven passes faster). Overall commands are
flat or slightly slower. Supplemental TOP_1000 regresses 1.0%; its TOP_100_COUNT
improves 1.8% on an unchanged path. Keep the original position reader.
All correctness gates pass and all 37 archive members are verified.

The automation used the preceding experiment's SHA-file name for this archive,
overwriting that marker and leaving the next stage queued. The preceding count
archive had already been downloaded and verified; neither archive's bytes nor
any timed samples changed. Before repairing the markers, the still-waiting next
process was stopped and its output directory confirmed absent. Both archive
hashes were recomputed, the count marker restored, the missing position marker
created, and the next stage restarted with its corrected runner. Original scripts,
repair script and before/after hashes are preserved. No latency runs overlapped.

### Two-term union cardinality from exact overlap — proposal

For two ordinary term sets, |A union B| = |A| + |B| - |A intersect B|. The current
full-corpus count-batches build spends 758.805 µs on OR counts versus 410.291 on
AND counts (same 300 query families; geometric means), so traversing only the
overlap may avoid substantial work. This is a hypothesis, not an estimated gain.

The existing complete `collect_segment` owner can offer this to count-only
collectors accepting `collect_count(0)` (an empty contribution, already legal in
window collection). Admit only a two-child plain union exposing count-equivalent
text terms, indexed physical fields, positive dictionary DFs and no deletions or
text mappings. Count the overlap through the existing Boolean query/scorer and
ordinary count driver. Reject inconsistent overlap/DF metadata. Missing terms,
fast-only fields, chunk/RGB mappings, tuning, proximity and other collectors keep
their current behavior. No approximate count estimates, new intersection kernel,
scorer, executor, persisted format or cache. Scratch and two cloned clause handles
are bounded independently of the corpus; score bits/ranked paths remain unchanged.

This optimizes the complete async collection entry point; limit-based sync and
async ranked collection retain their existing total-seen semantics. Portable and
native builds use the same helper. Preserve both raw and count-only behavior,
including duplicate clauses, cross-field overlap, missing terms, empty matches,
deletions, mapped multi-value fields and custom collectors declining aggregates.
Measure independently against the selected count-batches binary on ARM and x86.

The fallback test exposed an existing fast-only field bug: a standalone term
uses the fast-column scorer, but a pure union can be admitted to the postings-only
MaxScore/complete-text planner and return an empty result. A public behavior test
runs on the preserved count-batches source before repair. Fix admission in the
shared planner: both single-field preparation and per-field MaxScore eligibility
must require an indexed field. Existing fast-column scorers then handle these
clauses. The test covers complete counts and ranked async/sync results; this
correctness repair is retained independently of the union-count experiment.

### SIMD posting seek substitutions — rejected after x86 measurement

Replacing three scalar partition searches with the existing SIMD search does not
help the full-corpus workload. Official COUNT rises 560.975 → 566.812 µs (+1.0%),
AND COUNT 515.853 → 533.864 (+3.5%, all seven passes slower), and AND
TOP_100_COUNT 768.181 → 794.905 (+3.5%, all seven slower). Phrase TOP_10 regresses
0.9%; its COUNT improves only 0.6% in four of seven passes. ARM phrase
improvements do not generalize. Keep the original posting seek implementation.
Both architecture oracles, core tests, x86 Clippy and immutable bytes pass; all
37 archive members are verified.

### Exact-count collectors with separate ranked traversal — proposal

The complete collection API currently scores every match even when a top-k heap
and an exact counter are its only consumers. A single physical term already has
an exact dictionary DF; an eligible two-term union can obtain its exact count
through the existing overlap traversal. Test composing these counts with the
existing limit-based ranked collector, avoiding BM25 evaluation for discarded
matches. This is a collector capability, not a new scorer or executor.

The invariant is the same retained IDs, raw score bits, counts and top collector
`total_seen`, including pre-populated and nested tuple collectors. Add conservative
collector methods describing a finite retained rank limit and accepting counts
of omitted matches; only score-only TopK and Count collectors opt in. Tuples opt
in only when every child does and use the maximum child limit. Custom and position
collectors keep exhaustive callbacks. Admit only indexed, unmapped, undeleted
plain positive-weight text terms or same-field two-term unions, and only when
exact matches exceed the requested limit. All other queries retain existing
collection. Count and ranked traversal share the existing integrity checks.

The existing ranked executor adds its bounded top-k heap alongside the original
collector heap (up to twice the requested retained entries for the admitted plain
text plans). A stack counter measures published candidates, then collection adds
the exact number of omitted matches. Measure this memory tradeoff as well as
latency. Keep the
benchmark VERIFY/exhaustive branch explicitly exhaustive with a forwarding
collector that does not opt in, preserving its independent scoring oracle. The
cross-source oracle is also compared with immutable pre-optimization outputs.
This candidate builds on union-count and must be measured against that exact
binary independently before any integration. It adds no persisted format,
concurrency, cache or deadline protocol; bounded ranked-query APIs are unchanged.

### Compact scored conjunction batches — retained for combined measurement

Against its exact count-batches parent, full-corpus AND TOP_100_COUNT falls
802.463 → 685.130 µs (-14.6%) and the overall command falls
1341.296 → 1274.334 (-5.0%), both faster in all seven passes. Phrase
TOP_100_COUNT regresses 1.9% (758.895 → 773.231). Overall TOP_10 improves 1.0%,
TOP_1000 regresses 0.4%, and COUNT is flat (-0.2%); these are not the target
path. Supplemental commands move between -0.6% and +0.8%. ARM's small fixture
has many conjunction leads below the 128-document gate; its AND TOP_100_COUNT
is flat on canonical and +0.5% on merged. Full-corpus AND still takes 1.54×
Tantivy for TOP_100_COUNT. Retain pending an integrated comparison with the
original AVX2 starting binary; do not multiply the isolated improvements.
All four ARM and both x86 exact oracles, core tests, x86 Clippy and immutable
index bytes pass. All 37 archive members are downloaded and hash-verified.

The ungated ARM trial passes 1,584 core tests and every exact oracle but regresses
TOP_100_COUNT 4.8% overall on the canonical layout. Two-term unions with 101–1000
matches regress 32.6%, and those with 1001–10000 regress 33.6%; their second
traversal does not pay for itself. Standalone terms with 1001–10000 matches improve
7.6% and larger terms 31.2%, while small terms regress. Preserve the complete
ungated source/results and test a DF/limit gate: require at least
16 × max(128, k) matches for a single term and a term DF of at least
128 × max(128, k) before doing any union overlap traversal. These are experimental
admission thresholds, pending both-architecture measurement. Below them retain
the existing one-pass collection; no query's result limit or count is changed.

### Exact two-term union counts — retained for combined measurement

Compared with the count-batches parent, full-corpus COUNT falls
525.963 → 453.260 µs (-13.8%), faster in all seven passes. Tantivy takes
403.676 µs, leaving a 1.12× ratio. The changed OR family falls
766.177 → 505.701 µs (-34.0%, all seven faster), versus Tantivy 542.840 µs.
AND and phrase COUNT also improve 2.9% on
unchanged paths; do not attribute those movements to the union algorithm.
Other official commands move between -1.5% and -0.1%. Supplemental metadata
COUNT regresses 1.2% in all seven passes (13.106 → 13.268 µs); its path is
unchanged. ARM union COUNT improves 3.3% canonical and 3.2% merged; overall
COUNT improves 1.8%/1.5%. Retain for combined original-baseline measurement.

All 1,583 ARM core tests, four ARM and two full x86 exact oracles, x86 core
tests/Clippy and immutable index bytes pass. All 37 archive members are verified.
The included fast-only planner repair has a preserved failing baseline test and
a passing regression covering exact count and ranked async/sync behavior.

### Bounded AVX2 length gathering — proposal

A fresh isolated standalone TOP_10 profile of the original AVX2 baseline assigns
31.62% of Summa user CPU self samples to `DocLengths::gather_lengths`, 12.55%
to executor dispatch, 9.69% to block bounds, 8.32% to checked ID decoding and
8.27% to canonical scoring. The actual binary emits a four-way unrolled scalar
loop with an ID bounds branch, 16-bit load and output store per lane. Both engine
profiles, binary hashes and 17 archive members are verified. No hardware cycles,
branch or cache counters are exposed by this machine; perf reports them unsupported.
Sampling fractions are not a predicted latency reduction.

Test an owning-structures primitive that gathers eight little-endian u16 lengths
with AVX2. Gather complete four-byte words by `id >> 1`, then select the requested
halfword; mask every word whose full four bytes are unavailable. A final odd
halfword is read separately and selected only for its exact ID, so a vector load
never crosses the supplied slice. Missing IDs still return zero; trailing bytes
are ignored exactly as the scalar pairs view does. Output tails remain untouched.
Runtime AVX2 dispatch retains the existing scalar semantics on ARM and portable
builds. Strict chunk-length reporting/floors keep their existing path.

This adds no length cache, resident payload, scoring formula or persisted format.
Test arbitrary ID order, missing IDs, unsigned extremes, odd lengths, unaligned
byte views, every lane tail and a protected page immediately after the input.
Compare all four score-bit oracles, all index bytes, memory and complete workloads
on ARM and x86 against the original AVX2 baseline, separately from count changes.

### Gated separate ranking and exact counting — retained for combined measurement

Against its union-count parent, supplemental TOP_100_COUNT falls
739.015 → 249.711 µs (-66.2%, all seven passes faster); Tantivy takes 246.077 µs,
a 1.015× ratio. Official TOP_100_COUNT falls 1283.609 → 1198.149 µs (-6.7%,
all seven faster), versus Tantivy 851.246 µs. Official TOP_10 regresses 0.9%,
TOP_1000 0.7%, and COUNT 1.2%. AND TOP_100_COUNT regresses 2.9% and phrase
1.9% on unchanged traversal paths. The final combined measurement must include
these tradeoffs; the supplemental gain does not establish overall parity.

The gated ARM trial is essentially flat: official TOP_100_COUNT -0.2%/-0.5%,
supplemental -0.0%/+0.3% for canonical/merged layouts. Both architectures' tests,
exact ordered ID/score-bit/count oracles and immutable index bytes pass; all
37 archive members are verified. The final VERIFY adapter additionally checks
the optimized exact-count collector at each k=10/100/1000 against its explicitly
exhaustive reference, since admission depends on k.

### Memory difference investigation — protocol (completed below)

The user requested an explanation of the memory gap as well as further latency
work. A preserved standalone TOP_10 snapshot shows Summa RSS 350,416 KiB and
Tantivy 222,728 KiB. Proportional file-backed residency is 343,098 versus
219,895 KiB; anonymous residency is 5,032 versus 516 KiB. Neither process has
locked pages or swap in this snapshot. Anonymous memory includes stacks/runtime
allocations as well as heap; do not label all RSS as allocated heap. Most of this
measured gap is resident file-backed data. This is a standalone-workload snapshot,
not the peak of the four-command official workload.

Immutable full-corpus files: Summa postings 2,203,973,483 bytes versus Tantivy
1,054,571,755 bytes (2.09×); positions 2,752,821,603 versus 1,850,554,774 (1.49×).
Summa stores little-endian u16 lengths (10,064,248-byte chunks file), while
Tantivy's quantized field norms occupy 5,032,209 bytes. The current Summa Rounded
posting codec uses byte-rounded widths; Tantivy packs exact bit widths. These
format differences explain a reason for the larger footprint, but the fraction
of resident memory attributable to each must be measured rather than inferred
from file size alone. No cache/residency default or scoring precision is changed.

The audit was queued after final latency verification, on the original, integrated
and Tantivy binaries with identical RGB-off indexes. For each fresh engine and
workload, capture /proc smaps, smaps_rollup and status after one explicit opening
COUNT query and after each of three complete passes of each command. Aggregate
RSS/PSS/anonymous/locked bytes by mapped file, retain high-water RSS, and compare
pass stability. This is a memory attribution run with no latency claims. Verify
index hashes afterward and package every snapshot and script before stopping
the machine.

### Vector document-length gather — rejected

The AVX2 masked gather passes both architectures' tests, protected-page bounds
coverage and all exact oracles. All 37 cloud evidence members are verified.
It does not help the workload that motivated it: supplemental TOP_10 rises
143.794 → 146.758 µs (+2.1%), TOP_1000 +1.6% and TOP_100_COUNT +2.0%.
Official TOP_100_COUNT regresses 2.2% in all seven passes; TOP_10 is flat (-0.2%).
Official AND ranked queries improve 1.3–1.7%, but that does not justify retaining
the kernel. Keep the current scalar length representation and reads for this
traversal integration. The user's subsequent request to investigate quantized
lengths and compact postings/positions is a separate format/scoring experiment.

### Postings, positions and quantized norms — audit protocol

The user requested format investigation and permits evaluating quantized lengths.
The [format comparison and design](text-format-comparison.md) records the pinned
Tantivy implementation, Summa ownership/compatibility constraints, and the
proposed quantized-norm and compact-position experiments. The read-only audit
streams all term ranges and accounts for actual payload versus metadata bytes
in both immutable indexes. It also estimates narrower widths and reports the
corpus-wide precision loss from the 256-representative norm table. These are
size/precision measurements, not a new format or a latency prediction. The
separate process residency audit precedes these full-file scans.

### Completed memory attribution

All 401 audit members and input index hashes verify. After the third official
COUNT pass (following all ranked commands), selected Summa RSS is 1007.29 MiB
versus Tantivy 574.24 MiB. Position RSS is 664.19/343.77 MiB and posting RSS is
276.88/180.53 MiB: these account for 74.0% and 22.2% of the 433.05 MiB gap.
Anonymous residency is only 6.16/0.78 MiB; lengths/norms 9.60/4.80 MiB. Summa's
term dictionary is actually 6.70 MiB less resident. Original and selected builds
have the same mapped index residency. All three count-pass snapshots are stable;
all locked-byte observations are zero. Supplemental after/Tantivy RSS is
341.52/230.24 MiB, with no position-file residency. Keep this steady-state audit
separate from latency-run peak RSS and from the earlier standalone TOP_10 profile.

Source review identifies another layout cost: cache-miss admission checks every
interleaved block header, touching essentially every payload page in the term's
stream before pruning. The bounded proof cache avoids repeated validation but
cannot undo that initial page residency. A proposed compact directory should
carry sufficient structural metadata to validate without walking payload headers;
it must preserve rejection of corrupt data and block-owned content validation.
The observed RSS decomposition supports prioritizing postings/positions; it does
not assign a causal fraction of warm query latency to file size or cache misses.

### Integrated traversal result and validation

The selected 332-file source combines block-local ranked intersections, compact
count/score batches, eligible two-term union counts, and gated separate ranking
with exact counting, plus the reproduced fast-only planner fix. It changes nine
files from the preserved AVX2 baseline. Direct seven-pass full-corpus comparison:
TOP_10 648.391 → 633.884 µs versus Tantivy 523.427; TOP_1000
1138.350 → 1112.322 versus 918.857; TOP_100_COUNT 1269.097 → 1140.216 versus
847.893; COUNT 564.411 → 458.047 versus 402.800. All seven passes improve on
each operation. Ratios to Tantivy are 1.211/1.211/1.345/1.137; parity remains open.
Supplemental TOP_100_COUNT is 750.801 → 252.057 versus Tantivy 251.636 (1.002×).
Other supplemental operations are essentially flat.

Report the full-binary tradeoffs: phrase TOP_10 +1.8%, phrase TOP_100_COUNT +1.9%,
negated TOP_10 +7.6%, all seven slower. Negated COUNT +1.7% and mixed COUNT +2.0%.
Official peak RSS remains 1,031,540/1,031,444/587,744 KiB for before/after/Tantivy.
No memory reduction or universal per-query improvement is claimed.

ARM core 1,587 pass and x86 core 1,588 pass, each 16 ignored. Four ARM and two
x86 layouts preserve all 1,676 exact ordered ID/score-bit/count outputs and check
ranked and separately counted top-k at k=10/100/1000. All 40 ARM immutable files
and all full-corpus index manifests match. All 37 final cloud archive members
verify. Full native harness: 1,804 pass, 25 ignored, fmt/Clippy/native-no-sync
pass. Portable compilation passes with the preexisting set_document_units
dead-code warning. WASM is skipped per user instruction; lifecycle/RPC full,
cold-cache and concurrent ingest/merge tests are not run for this traversal-only
change. Main source hashes match every one of the 332 benchmark source files.

### Completed format and quantization accounting

All 13 format-audit members verify, including every component-sum identity and
immutable index hashes. Summa/Tantivy posting metadata is 652.434/50.166 MiB
(13.01×), explaining 54.9% of the posting-file gap. Position metadata is
339.989/13.429 MiB (25.32×), explaining 38.0% of the position-file gap. Posting
gap/frequency payloads are 887.465/660.291 and 561.974/295.261 MiB; position
payloads 2285.306/1751.398 MiB. The timed Rounded fixture has ratio/L1 bounds but
no impact records. Short blocks are 67.1% of postings and 46.4% of positions.

Exact/minus-one payload estimates save 577,974,979 posting bytes and exact-width
position estimates save 568,889,062 bytes, preserving existing block boundaries.
These are overlapping-format counterfactuals, not measured latency or a writer.
The documented analyzer difference remains: 177 terms, 220 document occurrences
and 247 position values more in Summa; timed-query counts match. This is tiny
beside the billion-byte format gap. See [full accounting and priorities](text-format-comparison.md).

Quantizing the existing norm column with the pinned 256-value table changes
58.28% of lengths, with 2.3456% mean and 11.0726% maximum downward relative loss
among nonzero lengths. It saves 4.80 MiB of norm payload. This is not a ranking
quality measurement. Quantized norms and lookup scoring, compact admission
metadata, and smaller payload/tail encodings remain versioned-format proposals;
none is silently enabled in the selected traversal source.

## September 16: bounded admission lookup experiment

Before changing the persisted block layout, measure repeated structural
admission caused by collisions in the existing direct-mapped proof cache.
The file-owning PostingListReader remains the sole proof owner for immutable
posting/position ranges. A proposed four-way set keeps the exact current entry
budget, read-lock-only hits, range plus proof-kind identity, and conservative
uncached lazy-file behavior. On a miss, validate completely before publishing;
under the existing write lock, reuse an identical entry or empty way, otherwise
replace one deterministically selected way. Probe at most four entries, use no
new allocation per request, and do not retain payloads or change cache defaults.

Invariant: every reused proof belongs to the same immutable file/kind/range;
corruption and short reads are rejected exactly as before. Reproduce repeated
collision validation with a behavior-named test before the change. Measure
against the September 15 integrated traversal build with the same immutable
indexes, budgets, compiler and flags; compare exact ordered score bits/counts,
RSS and both full workloads on ARM/x86. This remains an experiment until its
complete workload results pass; a hit-rate hypothesis alone is insufficient.

### Proposed batch admission for length gathers

The existing full-corpus standalone profile attributes 31.6% of sampled CPU to
`DocLengths::gather_lengths`. Its scalar loop checks every ID separately before
a two-byte read. The prior masked AVX2 gather was rejected on full workloads.
A separate portable experiment computes the maximum ID once for batches of at
least eight values. If it is within the existing two-byte column, all reads are
proven in range; otherwise the existing per-ID path retains missing-value and
strict chunk diagnostics. This introduces no sorted-ID assumption, allocation,
quantization, or new column representation. The structure owner remains
`segment/chunk_map`; `score_text_run` and canonical scoring are unchanged.

Cost: an O(n) maximum reduction over already-hot input IDs, followed by O(n)
length loads without per-ID conditional branches. This may cost more on short
or cache-cold runs, so selection requires the complete ARM and x86 workloads,
including supplemental terms and count-only controls. Compare exact raw score
oracles and all immutable bytes. Exercise arbitrary/repeated/unsigned-extreme
IDs, unaligned and odd-length columns, output tails and chunk floors before
claiming that the unchecked read is equivalent to the validated scalar path.

### Proposed reuse of admitted position directory lengths

Current sampled phrase-count CPU includes 6.38% in `PositionStream::locate_value`.
Trace: `TermPositionCursor::read_into` calls `read_cached`, whose noncanonical
(interior partial-block) lookup binary-searches the logical directory, then
re-reads and structurally checks the selected payload header to obtain its count.
`PositionStream::open` has already admitted every directory/header pair; proof
cache hits restore exactly that immutable admitted layout. Canonical streams
already trust the same admission proof for O(1) cursor addressing.

An isolated rewrite derives the located offset from the admitted logical start,
with the existing cursor/total and checked-subtraction guards. The upper-bound
search and admission invariant guarantee that a valid cursor precedes the next
logical start (or total). Do not change `validate_blocks`, decoding, file
ownership, serialized bytes, merge copying or the cache budget. Verify all
logical positions across mixed codecs and short copied interior blocks, backward
lookups and forward hints, corruption regressions, raw query oracles and both
architecture workloads. This is a format-preserving removal of repeated checks,
not permission to skip admission validation or defer malformed headers.

### Admission-cache result and rejected length experiment

The four-way cache is retained. On the full x86 corpus, unchanged budgets and
RGB off, before/cache/Tantivy official microseconds are TOP10
612.266/604.898/508.565, TOP1000 1089.938/1081.868/901.787, TOP100_COUNT
1116.342/1100.190/840.081, COUNT 453.954/446.120/400.934. The three ranked
commands improve in all seven rotated passes; COUNT improves in six. This is
only 0.7–1.7% overall. Union TOP10 and TOP1000 regress 1.6% and 1.8% in every
pass, while AND and phrase gains are larger. Supplemental TOP10 is effectively
flat at 142.837/142.671/52.974; scored count is 246.824/244.151/249.425. All 37
cloud members verify, all raw score/count oracles match, and the native harness
passes 1805 tests (25 ignored). ARM changes are small on the shared machine.

The separate memory audit verifies all 401 members. Official RSS after three
passes of each command is 1008.16/1007.60/574.26 MiB before/cache/Tantivy;
anonymous residency is 6.16/6.16/0.77 MiB. The index-file working set is
unchanged. Cache associativity saves repeated validation work, not file bytes
or the first admission's payload-page touches. Do not advertise an RSS fix.

The length-batch maximum reduction is **rejected**. All 37 x86 members and
correctness gates pass, but official TOP10 regresses 6.3% in every pass,
TOP1000 4.1% in every pass and TOP100_COUNT 2.0%; supplemental TOP10 and
TOP1000 regress 3.0% and 5.0% in every pass. The small ARM gains do not
transfer. The initial position-directory ARM prototype included this rejected
change; its queued cloud runner was cancelled before creating its output
folder. A clean position-only source is rebased on the retained cache source
and remeasured. No reported final gain may include the rejected gather.

The new cache-build CPU profiles verify all 45 archived members. Standalone
TOP10 still attributes 25.81% of sampled self CPU to `DocLengths::gather_lengths`,
10.86% to block bounds and 9.36% to checked doc-ID decoding. Phrase COUNT has
23.12% in posting seek, 10.53% in position reads and 6.38% in logical position
lookup. Hardware cycles/instructions/branch/cache counters are unsupported on
the machine; CPU-clock samples work. These samples select experiments, not substitute
for unprofiled latency or prove that file size alone causes the runtime gap.

### Final cache-plus-position source — verified September 16

The clean source removes the rejected length gather completely. Its 332-file
archive changes only the existing posting reader and position stream relative
to the September 15 traversal source, SHA-256
`722875a6e1a771a600e0edd61187c71cf6a1d0fac4bd3862366c84997feed921`.
Every main and cloud source hash matches. The original and cache-parent binary
hashes match their preserved selected builds. The final cloud archive verifies
42 members; the separate final memory audit verifies 401.

Direct original/cache-parent/final/Tantivy official geometric-mean microseconds:
TOP10 620.026/620.732/616.652/510.792; TOP1000
1115.694/1089.637/1071.549/929.304; TOP100_COUNT
1141.103/1105.286/1090.108/835.435; COUNT
458.181/447.773/441.880/401.152. Final versus original improves
0.5/4.0/4.5/3.6%, with 5/7/7/7 faster passes. The added position change improves
0.7/1.7/1.4/1.3% against the cache parent, with 5/7/6/7 faster passes. Top-10 is
nearly flat; parity is unmet at 1.207/1.153/1.305/1.102× Tantivy.

Supplemental original/final/Tantivy microseconds: TOP10
137.075/136.365/50.050, TOP1000 564.936/558.808/426.294, TOP100_COUNT
246.615/245.448/250.694, COUNT 12.945/13.160/8.938. Standalone top-10 remains
2.725× Tantivy. The metadata-only count regression is 1.7%, every pass slower;
union TOP10 regresses 1.2%, five slower passes. These tradeoffs are retained
explicitly; isolated gains are not multiplied into the direct final comparison.

Final position-only ARM official changes against the cache parent are
-0.5/-1.2/-0.8/-1.0% canonical and -0.3/-1.5/-0.8/-0.2% copy-merged, in the
same command order. These are small changes on a shared Mac, not architecture-
independent guarantees. All four ARM and two x86 1,676-query raw-score/count
oracles match, including independent exhaustive checks and optimized exact-count
TopK at 10/100/1000. Forty ARM files and all full-corpus index hashes are
unchanged. Final native harness: 1806 passed, 25 ignored; all four stages pass.
Portable core passes with the preexisting dead-code warning. Final ARM/x86 core
suites pass 1589/1590 tests, 16 ignored each. WASM is skipped as instructed;
cold/concurrent and full lifecycle/RPC checks are unrun.

Final official RSS original/final/Tantivy is 1008.24/1007.56/574.09 MiB;
anonymous residency 6.16/6.16/0.78 MiB. Supplemental RSS is
342.43/341.38/230.25 MiB. These are separate three-pass residency runs, not
latency-process peak RSS. Latency-run original/final/Tantivy peaks are
1031384/1031652/587800 KiB official and 349816/350352/235524 KiB supplemental.
No meaningful RSS reduction is claimed. The next larger work remains compact
posting/position metadata and payload experiments, plus versioned quantized
norms and lookup scoring with explicit ranking validation. None is silently
enabled here. [Complete evidence and reproduction](benchmark-results/admission-2026-09-16/README.md).

## Compact text directories and byte norms — September 16

### Ownership, format and cost model

The [format design](compact-text-format.md) preceded implementation. New writes
opt in with `compact_text` and `quantized_norms`; both defaults remain false.
BPL2 adds separate four-byte descriptors and optional four-byte position cursors.
POS4 stores two-byte descriptors and one checkpoint per eight blocks. Admission
reads these directories without touching packed payloads. Pfor exception framing
retains the old posting layout. Existing codecs and copied-block geometry remain
unchanged; ordinary merges remap small metadata and copy encoded arrays.

CHNK version 5 adds byte norm sections for ordinary fields, preserving exact
original token totals and physical chunk/reorder geometry. Pruning bounds use
the same rounded representatives as scoring. All-byte merges copy codes; mixed
old/new columns expand byte representatives to u16 with bounded scratch, keeping
source scores and copied bounds valid. The canonical BM25 owner supplies a
query-local 256-float lookup table, one KiB per active quantized cursor.
Metadata version 9 prevents older readers from silently accepting these layouts.

The first reader's profiles exposed out-of-line header/payload accessors. The
selected cleanup shares the descriptor/L0 borrow and reuses one header/payload
lookup during document decoding and content validation. It changes two files
from the first measured reader and no persisted bytes. Unit tests cover codec
combinations, short copied blocks, old/new/mixed merges, zero-width payloads,
malformed directories, cancellation and scratch budgets. Format-preserving
paths compare bytes as well as query results.

### Final direct measurements and tradeoffs

Same full corpus, compiler, machine, flags, caches and seven rotated passes;
RGB off. The final source is measured against the preserved September 16
baseline, a reader-only old-index control, compact/exact, compact/byte norms,
Tantivy and the first byte-norm reader in one run.

Official compact/exact changes versus the original baseline are
**-0.4/-0.5/+1.9/+0.5%** for top-10/top-1000/top-100 plus count/count.
Compact/byte changes are **+5.1/+4.0/+2.3/+3.4%**. The final reader cleanup helps
the byte-norm case **2.5/1.6/2.6/2.5%** against its first reader in the same run,
but standalone top-1000 regresses **5.5%** against that reader and **14.8%**
against the original baseline. These tradeoffs prevent a blanket speedup claim.
The old-index reader control changes **+0.7/+0.1/+3.2/+2.4%**; disabling the new
write options does not remove all reader-code overhead.

Compact/byte final Summa/Tantivy ratios are **1.254/1.242/1.323/1.140×**.
Compact/exact remains faster than byte norms overall, but still trails Tantivy.
ARM's 100,000-document check also gives mixed results: compact/byte official
changes **-0.3/+2.0/-0.4/+1.3%**. Shared Mac load limits interpretation of small
movements. Neither experiment supports changing defaults.

File savings: **96.589 MiB postings + 185.028 MiB positions**, plus **4.80 MiB**
for byte norms. Every term, encoded array, block boundary, codec, width and
document ID matches across layouts. Official RSS baseline/compact/byte/Tantivy
is **1034.65/816.10/811.29/574.09 MiB**. Anonymous RSS is only
**6.16/6.18/6.23/0.77 MiB**. Position mappings explain most of the improvement:
**671.06 → 458.94 MiB**; postings **286.94 → 280.38 MiB**. Norms fall from
**9.60 → 4.80 MiB**. The reduction is mainly mapped pages, not heap memory.

Quantization is a precision change. Full-corpus mean set overlap with exact
norms is **95.15/97.44/98.46%** at top-10/100/1000; it is not a relevance judgment.
All counts agree. Each representation independently matches exhaustive ordered
IDs/raw-score bits/counts for all 1,676 queries, including optimized exact-count
collectors at k=10/100/1000. Compact/exact matches the old score oracle exactly;
final reader outputs match the first reader for every layout on ARM and x86.

### Rejected probes and remaining work

The norm-code gather prototype did not improve ARM ranked search consistently
and was not promoted. A separate two-line decode-buffer reuse prototype passes
220 posting tests and all ARM oracles, but its small gains also appear in
metadata-only COUNT, which does not exercise that decoder. It is not evidence
of a decoder speedup and is absent from the selected source.

Remaining measured costs include seeks, decoding, phrase confirmation and
scoring. The current formats still retain Rounded payloads, ordinary tails and
large per-term footers; compact posting metadata is still 555.85 MiB and
position metadata 154.96 MiB. Lookup normalization is implemented but its CPU
benefit is unproven. Separate candidate/scored-block counts from arithmetic
cost before attributing the quantized regression; the current profiles alone
do not settle that question. Decoder payload-span lookup still reads neighboring
L0 metadata; further reuse of admitted offsets needs its own correctness and
performance checks. Additional payload/tail changes must preserve
copied merges and bounded seeks. [Upstream source findings](text-format-comparison.md#priorities-from-the-evidence)
remain research directions, not adopted defaults.

### Validation and evidence

Final full native harness: **1,816 passed, 25 ignored**, plus **4 real-server
checks passed**. Formatting, Clippy, native no-sync, portable core and docs build
pass. Portable core retains the existing `set_document_units` warning. Final
x86 core passes **1,599 tests, 16 ignored**, plus the compact/norm integration.
An initial parallel broker discovery timeout is retained alongside the passing
serial rerun and both passing full runs. WASM is skipped as instructed. Cold,
concurrent performance and judged relevance are unmeasured.

Final 335-file source SHA-256:
`fa9e4bf6c205058c84f35717015c4527e1aa3d19196e11b0dd18a79c52ff2260`.
Main, ARM and cloud source hashes agree; the immutable legacy/Tantivy indexes
are unchanged. The downloaded cloud archive verifies 1,503 members. The machine
stop command lost its connection, but a subsequent direct status query confirms
`TERMINATED`. [Verified source, scripts, raw measurements, checks and profiles](benchmark-results/compact-text-2026-09-16/README.md).

## Query work diagnosis — September 16

Implemented `query-diagnostics`: fixed-size per-scope counters in posting,
position, scorer and admission owners; async poll-scoped capture; explicit
synchronous segment-worker propagation with one merge per completion. Production
builds compile out counter calls and argument evaluation. The benchmark retains
its stdout protocol and emits structured stderr records; the reusable Python
collector checks responses, scope coverage and aligned comparisons.

The [diagnosis](search-work-diagnosis.md) records full-corpus and ARM evidence.
This changes no writer, index bytes, scoring formula, query plan or defaults.
Remaining review targets are single-term bound tightness/ties, COUNT norm-table
construction, scoring-loop cost, complete intersection traversal, and payload
density. Instrumented times are phase attribution only. Selected payload content
validation is still required; structural admission reuse is measured separately.

## Targeted pruning and setup fixes — September 16 proposal

The public collector builds ordinary `TermScorer`s for complete membership as
well as scored collection. Its byte-norm table is currently constructed when
lengths are attached, even if the collector never requests a score. Defer that
1 KiB allocation and its 256 normalizations until the first scoring call, cache
once per scorer, and invalidate the cache when parameters or lengths change.
Batch/window scoring resolves the cache once outside the document loop. Actual
score arithmetic, missing-length fallback, and async/sync paths stay shared.
Regression: quantized Boolean COUNT builds zero tables; subsequent ranked
collection still uses lookups and preserves exact score bits.

The selected ratio-only index cannot infer coupled frequency/length statistics
that were not persisted. Existing impact records supply those statistics, but
currently every bound evaluates all alternatives. Investigate threshold-aware
refinement: reject blocks with cheap bounds first and decode the bounded impact
record only when that can change the pruning decision. Preserve strict score
comparison and complete fallbacks for unsupported/unknown records; never change
query-global statistics or use a locally computed winning document as a global
bound. Measure impact-bearing and ratio-only fixtures separately.

Gap representation experiments must use existing codec tags or an explicit
format gate. Byte-rounded gaps consume eight bits even for dense runs; compact
directories do not change those arrays. Keep copy merges and content validation.
No codec/default change follows from work counters alone.

Before testing threshold refinement, cache the supported BM25 bound coefficients
once per text cursor: parameter checks, the IDF numerator, and `b / average`
are query constants. The existing ratio/impact formulas remain in the BM25
owner with their rounding guard. This adds bounded cursor-local scalars, no
index-sized cache, and applies to both ratio and impact indexes. The existing
public-internal bound helpers delegate to the same prepared implementation.

### Bounded strict-gap validation — implemented

Codec 3 stores gap-minus-one values. After validating its reserved first zero,
a block has at most 127 positive gaps. At widths at most 25 their sum is at
most `127 * 2^25 < 2^32`; a wrapped prefix necessarily has an endpoint below
its start. Matching ordered L0 endpoints therefore proves strict ordering,
without scanning the decoded u32 array again. Wider widths retain the generic
ordering reduction. This preserves rejection semantics and persisted bytes,
including old codec-3 tails. It makes the existing denser representation cheaper
to read; it does not relabel encoded bytes as physical I/O or change codecs.
Tests compare the shortcut with the general validator across every count, width,
unsigned wrap, sentinel endpoint and mismatched directory endpoint, plus existing
per-byte corruption and copy-merge tests.

The [follow-up results](search-pruning-fixes.md) also test a fresh combined
compact/Simd4x/impact index with exact norms, using the same corpus order, writer
budget and one indexing thread. This is a configuration experiment with existing
formats; no corpus reorder or writer implementation changes are involved. The
ideal-threshold probe distinguishes loose bounds from slow threshold discovery.

### Threshold-directed impact rejection — rejected experiment

The ideal-threshold probe on the full corpus leaves 17,524 decoded blocks for
`is` and 23,789 for `to` with ratio bounds, versus 26 and 16 with impact bounds.
Tight metadata is necessary; merely establishing a heap floor earlier is not
enough. Test a standalone-term rejection predicate over complete impact records:
invert the existing guarded bound once when the score threshold changes, then
compare every `(1 - b) + b * length / average` with `minimum * TF`. This avoids
per-point and final score divisions in groups that can be rejected immediately.
An inconclusive predicate retains the current complete bound calculation.

The inversion targets `threshold.next_down()` and rounds its quotient, cutoff
and comparison conservatively using the existing f32 error guard (much wider
than the new f64 operations' error). Zero/nonfinite floors and k1=0 retain the
existing path. A successful predicate caches `threshold.next_down()` as a valid
score bound, never a boolean that could be reused with a different threshold.
Scratch remains a threshold bit pattern and one optional f64 per executor.
Regression compares every successful shortcut with the existing inflated bound
and with canonical scores, including adjacent float thresholds and extremes.
Measure independently on the frozen packed candidate before retaining it.

### Collector-directed norm setup — selected revision

The full-corpus first timing pass shows a byte-norm ranked regression in the
combined reader candidate. Isolate lazy setup, and test avoiding its per-score
OnceLock access entirely: carry a private `skip_scoring_setup` hint from the
collector through existing `ScorerOptions`. Membership collectors set it; ranked
and positioned callers retain eager tables. Ordinary term scorers attach the
same represented lengths even when optional setup is skipped, so scoring remains
valid through canonical arithmetic if a nested/custom consumer requests it.
The hint is preserved by threshold/required-clause option transformations.
This reuses the existing collection intent boundary, with no new public query
mode, scorer implementation or hot-loop branch/atomic access.

### September 16 pruning follow-up: disposition

The [pruning fixes](search-pruning-fixes.md) retain collector-directed norm setup,
prepared BM25 bound constants, cheap-bound-first refinement, and the bounded
strict-gap ordering proof. The reader changes preserve encoded bytes and exact
results; tighter pruning and smaller gap payloads use the existing opt-in impact
and Simd4x formats in a newly built combined index. No default changes.

The lazy normalization cache was replaced by collector intent so ranked scoring
retains its original table lookup without a per-score initialization check.
The inverse-threshold impact predicate was removed: on the full-corpus paired
run it changed combined-index top-10 from 263.496 to 271.283 microseconds (+3.0%),
and impact-only from 264.428 to 265.108 (+0.3%). Its ARM gains did not justify
retaining the numerical shortcut. These are prototype comparisons; the final
selected-reader run is reported separately, without multiplying phase gains.

### Selected reader: final matched evidence

Full-corpus combined-index top-10 changes +4.0% on official queries and -26.6% across all queries. Official top-10 remains 1.208× Tantivy. The final COUNT pass constructs zero norm tables with all counts preserved. Native harness: 1,817 passed / 25 ignored; all five layouts match exhaustive references on both architectures. WASM was skipped per user instruction.

The [complete tables and evidence](search-pruning-fixes.md) separate code-only controls, the new combined configuration, byte norms, anonymous memory and resident index pages. Retained changes do not alter format defaults or silently transcode existing segments. Remaining Tantivy parity work is open.

Retained reader-only official top-10 is effectively flat (-0.5%), still 1.155× Tantivy. Byte-norm official COUNT improves 3.1%. The combined configuration regresses official top-10 4.0%, especially unions; keep it experimental. The largest common-category gaps on the original index are intersections (1.240× Tantivy) and phrases (1.263×), versus unions (1.035×). These workload timings locate remaining work without claiming a CPU-level cause.

## Official-workload parity — September 16 continuation

Acceptance target: compare the official 962-query workload with the pinned
Tantivy build, RGB disabled, on the same full corpus, CPU, compiler and flags.
Measure top-10, top-1000, top-100 plus exact count, and exact count separately;
supplemental standalone terms cannot compensate for an official regression.
Preserve exact result bits, membership, positions, cross-segment statistics,
corruption errors, bounded scratch and native/async equivalence. Keep final
latency separate from profiles, builds and memory audits.

The starting source is the retained pruning/count reader. Its original compact
index is the main fixture; the combined packed/impact index remains an
experiment because its official workload regressed. Refresh per-family CPU
profiles before choosing changes. Earlier samples point at intersection cursor
alignment and phrase candidate handling, but they predate several reader
changes. Proposed work must remove recurring work inside those existing owners,
not introduce a benchmark-specific executor or bypass validation.

### Current profile and bounded candidates

The refreshed exact-norm reader attributes 22.4% of phrase top-10 sampled CPU
to posting seek and 15.0% to per-candidate score bounds. Intersections attribute
25.2% to conjunction alignment and 18.6% to typed-cursor seek preparation.
These are sampled CPU shares, not latency improvements or exclusive causes.

Test two independent changes before combining them:

- Prepare the phrase bound's query-constant coefficients once, with the existing
  conservative arithmetic and first-term multiplicity semantics. This removes
  repeated parameter checks/divisions from each phrase candidate; candidate TF
  and represented document length still vary.
- Expose the fixed 128-document geometry to an in-block lower-bound search.
  Tantivy's [fixed-block search discussion](https://quickwit.io/blog/search-a-sorted-block)
  motivates comparing a fixed seven-step search with the current variable-slice
  gallop/SIMD loop. Retain monotone seek, exact exhaustion and tail behavior.
  This is a reader-only experiment; compare generated code and both architectures.

If these are insufficient, batch intersection matching inside loaded posting
blocks in the existing iterator/cursor owners, sharing the decoded membership
operation with phrase alignment. Scratch must remain bounded by the posting
block; no document-universe materialization or benchmark query classification.

The complete-count union profile attributes 17.8% of sampled CPU to scalar
heap admission over score windows. Test screening each 64-score block against
the current full-heap threshold before visiting its set bits. The screening
threshold may become stale only by admitting extra candidates; equal and
unordered scores must reach the existing total-order comparison. Count the
original membership mask before screening. This needs no new allocation, score
formula, posting format, or collector implementation.

Negated top-k queries spend 51.1% of sampled CPU in scalar term scoring.
Extend the existing Boolean score-batch capability to required-only queries
with exclusions. Score bounded positive batches in the existing child scorer,
then remove excluded IDs with exact child seeks and compact scores in place.
Keep optional clauses and non-batching positive scorers on their existing path;
resume at the next exact match and preserve positive-clause reduction order.

Byte-norm batches currently interleave the table lookup, missing-length fallback
and division in one loop, unlike exact-norm batches which first gather lengths
and then score contiguous inputs. Test the same two-phase shape for byte norms:
gather at most 128 canonical normalization values, then evaluate the existing
boosted formula over contiguous TF and normalization arrays. Preserve code-zero
TF-as-length fallback, zero TF/boost, strict floating arithmetic and per-query
table ownership. Compare exact-norm and quantized fixtures independently.

The L0 seek currently gathers up to 32 strided skip entries into a stack array
before searching them. Test lower-bound search directly over the validated
16-byte entries after the existing L1 gallop. It reads logarithmically many
entries, avoids the gather and leaves both the current-block shortcut and the
serialized representation unchanged. The existing near/distant seek regression
checks membership, position cursors and byte identity across codecs.

The fixed-128 seek screen is rejected: official x86 TOP_10 rises
610.583 → 668.926 µs (+9.6%); ARM rises 33.542 → 34.408 µs (+2.6%).
Keep the prior SIMD/galloping document searches. A subsequent reader candidate
separates loaded-block cursor seeking from block-directory lookup so the small
common case can inline without the cold sparse/text dispatch body. Phrase
bounds additionally test the equivalent direct TF/length envelope with one
f64 division instead of dividing normalization by TF and then dividing the
score. Retain the same conservative f32 rounding margin and test extreme
frequencies/lengths; scored values remain unchanged.

### Bounded decoded-block intersections — experiment

Use one structures primitive for sorted posting blocks: compare each candidate
from the sparser stream against eight IDs from the other block, emit matching
index pairs, and stop when the output or either block ends. The output is at
most 128 pairs of byte indices. SIMD equality avoids finding an exact lower
bound for every failed candidate inside an eight-ID group. Unsigned range
checks, scalar tails and a portable equality fallback preserve exact membership.

The existing two-cursor conjunction consumes the pairs to fetch TFs and keeps
query-order score reduction. The phrase iterator uses a one-pair output and
parks both posting cursors on the match before reading positions. Directory
skips remain outside the primitive; distant gaps must not turn into linear
block scans. Cancellation is checked at least once per bounded block operation.
No new query executor, document-wide bitmap, format, cache, or heap allocation
is introduced. Test suffixes, partial outputs, tails and unsigned extremes
against scalar intersection, then exact scores/counts/positions on both hosts.

The rejected global fixed-block search has a useful narrower signal: x86
intersection COUNT improves 397.40 → 367.41 µs and TOP_100_COUNT improves
668.30 → 615.42 µs; ARM intersection COUNT improves 24.64 → 23.66 µs.
Test that search only in sorted candidate-batch retention, keeping ordinary
posting seeks and ranked cursor alignment on their prior algorithms. This
separates membership probes from the harmful global substitution; the final
combined comparison must confirm the benefit.

ARM code inspection of the first intersection primitive confirms NEON equality
comparisons but also cursor-position stores inside the matching loop. Keep
positions in local integers and commit them once on return; no caller observes
intermediate block positions. Compare this refinement with the narrow batch
retention change against the frozen first-intersection binary.

### Validate byte views once — experiment

The directory owner resolves the Vec/mmap enum, follows its Arc and bounds-checks
the same range in each `OwnedBytes::as_slice`. Test a stored validated pointer/length
with the unchanged backing Arc, as specified in [owned byte views](owned-byte-views.md).
This targets recurring reader work across postings, positions and norms without
changing their formats, removing corruption checks or adding resident copies.
A slice-boundary regression must fail first; lifetime, mmap and threaded clone
tests precede performance selection. The direct view keeps the current struct
footprint, with a documented unsafe dereference and Send/Sync proof.

### Reuse exact required-term counts with ranked collection — experiment

The latest official comparison isolates another avoidable cost: the 40 queries
with one required text term plus optional terms take 5730.65 µs for top-100
with exact count, versus Tantivy's 2537.44 µs. Their count-only path already
uses the required term's exact dictionary cardinality. Reuse that cardinality
with the existing bounded ranked executor instead of scoring the full required
stream solely to count it.

Add a conservative query capability for a term-equivalent count alongside an
exact text rank plan. Ordinary positive text terms and untuned, same-field
positive text Booleans with one required term may expose it. A count-only hint
from an opaque/custom query is insufficient. The collector keeps its existing
physical, indexed, undeleted/unmapped/nonchunked checks and its existing
16 × max(128, k) cardinality gate. Optional clauses change scores, not membership;
all actual scoring and top-k selection remain in the existing query scorer.
Test duplicate and missing optional terms, ties, nested collectors, preexisting
hits, both sides of the cardinality gate and all posting codecs against the
exhaustive callback path before matched measurements.

### Share the existing conjunction executor with complete ranked collection — experiment

Pure text AND already enumerates every intersection in the typed ranked executor;
it does not prune matches. Complete top-k-plus-count currently uses the generic
Boolean scorer, which scores lead candidates before other terms reject them.
Allow a top-level score-only/count-compatible collector to request a bounded
ranked result plus the exact cardinality of this existing traversal. Keep the
same posting intersection, TF rows, canonical scoring and top-k owner. No second
executor, count pass or corpus-sized buffer is introduced.

The request is private scorer-construction metadata and is cleared at nested
clause boundaries. Only the existing untuned/unboosted same-field, nonchunked,
unmapped text conjunction plan may fulfill it; optional terms, positions,
deletions and unsupported plans retain complete streaming. The materialized
result scorer reports its exact count only for this explicit request. The
collector adds omitted hits through its existing exact-count capability. The
usual ranked APIs keep their existing total-seen semantics. Test all codecs,
empty/missing terms, duplicate terms, query order, ties, prepopulated/nested
collectors, k=0 and fallback plans against complete streaming before measurement.

### Reuse bounded heap screening for conjunction score batches — experiment

The conjunction scorer already produces contiguous exact score batches, but
inserts each hit individually. Reuse `ScoreCollector::insert_text_run`, already
used by single-term batches and tested for ties, unordered floats and seeded
heaps. For mapped results retain the existing identity conversion and scalar
insertion; ordinary conjunctions use the same bounded eight-score screen.
No scores, counts, traversal or heap policy change. Measure ranked AND and the
complete counted plan independently before selecting this dispatch.

### Reuse initialized block storage without a redundant zero pass — experiment

The optimized ARM `decode_block_doc_ids_checked` contains a vector zero-store
loop before decoding: `clear(); resize(count, 0)` discards the initialized
length on every block. All document, frequency and position kernels overwrite
every output lane, including zero-width streams. Keep the previous initialized
length and call `resize` alone: equal-sized blocks require no preliminary writes,
shrinking truncates, and growth initializes only the new suffix. No uninitialized
memory, unchecked indexing, allocation policy or decoder is introduced.

Apply this only to the owning decoders with full overwrite semantics. Preserve
output clearing on errors and document-position assembly (which uses push), and
preserve score-buffer clearing because it marks deferred computation state.
Test changing sizes, codecs, zero values and nonzero stale buffers; compare
immutable encoded bytes and full score/count oracles. Inspect the optimized
caller and measure both architectures before selection.

Counted-AND review found a wrapper boundary hazard before selection: a unit-boost
wrapper could pass the private construction request through, then hide its
child's exact count. The regression compares unit-boosted and filtered AND to
complete streaming. Only a query explicitly advertising the exact conjunction
handoff now receives that request; opaque/wrapped queries retain the existing
stream. Box forwarding preserves the underlying query's explicit capability.

### Extend block intersection to longer conjunctions — experiment

The refreshed counted-stage profile attributes 22.59% of ranked AND CPU samples
to `TermCursor::seek_sync`, 13.07% to the new block intersection, and only 3.46%
to length gathering. Of 300 official AND queries, 102 have three or more terms;
their existing path still aligns every document with generic seeks. Extend the
same pair-batch owner to the two rarest clauses, then verify the remaining
clauses against its bounded candidates. Keep TF rows in original cursor identity
using a 128-byte origin map and compact them once after membership is known.
Use later-clause cursor heads to skip impossible future candidates. No new
executor or scorer is introduced, and canonical reduction order remains intact.

Within the pair driver, seek through the directory only when an entire block
ends before the other head; the SIMD intersection already aligns overlapping
blocks. Longer phrases similarly intersect the two rarest posting lists before
probing remaining terms, while retaining their original position identities.
Test empty batches, late matches, duplicate clauses, every query order, budgets,
all codecs and exact score/count references. Measure the two-term controls too.

### Expose fixed intersection call-site sizes to the optimizer — experiment

The selected ARM phrase caller still calls `intersect_posting_blocks` out of
line, despite requesting exactly one pair; the generic callee retains dynamic
output-capacity and cursor-writeback handling. Add an ordinary `#[inline]` hint
to this shared bounded primitive so LLVM can specialize the fixed phrase output
and batch callers when profitable. No `inline(always)`, new kernel, unchecked
access or semantic change. Retain only with matching x86/ARM measurements and
inspect the final call sites rather than assuming the hint is honored.

### Compare packed score-only heap ordering — experiment

The remaining top-1000 gap may include collector ordering cost. Test encoding
`(reverse f32 total order, doc ID)` in the existing eight-byte private score-only
entry. This permits one integer comparison in heap and result ordering, while
recovering the exact score bits for output and the vector admission screen.
Keep the position-aware heap and canonical tie policy unchanged. The encoding
must round-trip signed zero, infinities and every NaN payload; compare ordering
against the current comparator and full collection results before measurement.
This is an isolated candidate; retain only after same-fixture timing.

### Cache immutable phrase frequency without an initialization lock — experiment

The counted-stage phrase profile includes OnceLock initialization/wakeup work on
successful phrase hits. The cached value is a pure count over immutable position
buffers; simultaneous shared score reads may compute and store the same value.
Test an AtomicU32 with zero meaning uncomputed and positive values meaning the
exact frequency. A confirmed phrase always has at least one occurrence. Reset
through exclusive mutable access at both cursor movement and backfill writes.
Relaxed atomics suffice because this cache publishes no other memory: shared
borrow/ownership already protects the input buffers. Preserve deferred counting
for membership-only consumers, thread-safe repeated score reads, duplicate-start
multiplicity and every existing invalidation boundary. This remains an isolated
candidate until correctness and matched timing support selection.

### Block-execution selection and remaining limits

The cumulative `packed` candidate retains the bounded intersections, direct
L0 search, validated byte view, complete-count handoffs, initialized-buffer reuse,
lookup gathering and packed eight-byte score-only heap. The global fixed-128
seek replacement is rejected. The wrapper/count regression discovered during
review is fixed in the selected source. No index-format defaults change.

The five-pass x86 screen of `inline` / `packed` / Tantivy measures, in µs:

| Command       |  Inline | Packed heap | Tantivy |
| ------------- | ------: | ----------: | ------: |
| TOP_10        | 518.550 |     519.512 | 546.600 |
| TOP_1000      | 947.152 |     919.891 | 935.636 |
| TOP_100_COUNT | 843.042 |     802.024 | 869.517 |
| COUNT         | 408.617 |     408.897 | 433.485 |

This is an experiment screen; use the independent final seven-pass confirmation
in the [block-execution report](search-block-execution.md) for the final claim.
The ARM packed-heap isolation screen changes top-1000 by +1.5%; it does not
establish a cross-architecture win for that individual change. The complete
before/after ARM comparison remains a separate acceptance check.

Actual ARM heap call-site assembly shows one 64-bit comparison for packed
entries, replacing score-key conversion and a separate document tie comparison.
All score bits round-trip; collector tests cover signed zero, infinities, NaN
payloads, ties and preexisting hits. Main source matches the recorded 205-file
packed source manifest. Final native checks pass 1,828 tests with 25 ignored.

Standalone terms, memory and per-family tails remain explicit review findings.
The initial confirmation's compact-exact RSS is 749.79 MiB versus Tantivy's
565.77 MiB, dominated by mapped positions and postings. Byte norms save about
4.8 MiB and do not establish latency parity across all commands. The optional
combined layout reduces payload residency but has workload-dependent latency.
No conclusion about cold-cache or concurrent ingest/merge performance follows.

The isolated `memo` phrase-frequency cache is rejected. Its x86 compact screen
changes TOP_10 502.233 → 504.498, TOP_1000 941.754 → 944.007,
TOP_100_COUNT 834.332 → 838.841 and COUNT 401.059 → 402.243 µs.
The small ARM gains did not reproduce on x86. Preserve the existing OnceLock;
its generic profile symbol also includes block-frequency initialization, so its
entire CPU share must not be attributed to the phrase-frequency cache. The
candidate passes correctness checks but provides no measured selection benefit.

## Final RGB-off confirmation and separate RGB experiment — September 16

The seven-pass confirmation selects the compact/exact `packed` reader. Official
962-query geometric means beat Tantivy by 4.0%, 1.8%, 8.6% and 5.8% for top-10,
top-1000, top-100 + count and count. Final fresh-process peak observed RSS is
760.03 MiB versus 574.24 MiB. All 1,676 exact references pass across five layouts
on both hosts; immutable compact and byte-norm files retain their hashes.
See [final results and evidence](search-block-execution.md). Supplemental terms
and memory remain open findings; the aggregate win does not erase those gaps.

The user requested a separate RGB benchmark. Before adapting the harness, the
invariant is stable document IDs, raw score bits, counts and positions across
reordering. The cost model includes the field-local map, graph construction,
postings/positions rewriting, mapped query execution and evictable payloads.
Use the existing standalone `IndexWriter::reorder` owner on a freshly built,
explicitly eligible schema. The benchmark adapter only exposes eligibility and
that existing operation; it must not edit persisted schema or implement a writer.
Compare the selected compact index, identity-mapped eligible control, RGB output
and Tantivy. Freeze the selected reader, corpus, compiler, flags and cache limits.
Measure official and supplemental workloads separately, construction/reorder
resources, query residency and file sizes. Require exact references before
latency runs. Preserve the RGB-off result independently. Merge-time RGB and the
remaining budget/migration audit are outside this benchmark's claim.

RGB setup audit found another format cost: the current standalone text reorder
writer preserves bound kinds but emits its existing noncompact postings and
positions representation. Passing `--compact-text` does not change that writer.
Thus RGB versus the winning compact index measures the complete current feature,
including format expansion; it is not a pure permutation experiment. Preserve
an identity-mapped control and report this limitation with file sizes. An initial
ARM attempt also inherited a 16 KiB dictionary target; it was stopped and excluded
before completing measurements, then rerun with the matched 4 KiB target.

### RGB execution diagnosis

The mapped plain-text complete path currently calls
`required_text::mapped_documents`: it first exhausts the physical scorer into a
logical-document bitmap, then `DocumentMappedScorer::position` walks logical IDs,
looks up each physical slot and calls `PostingIterator::seek_physical`. Backward
probes search from block zero and reload decoded buffers. Identity maps bypass
this wrapper, which explains why they do not expose its cost. This is a concrete
loss of RGB locality, not evidence that the partitioner produced no permutation.
The ARM partitioner reports convergence and a nonidentity permutation.

A future optimization must keep compatible same-field Boolean, phrase and exact
count evaluation in the shared physical order, translating predicates/results
at the appropriate boundary. Preserve stable-ID tie ordering, position identity,
exact scores, independent field permutations and the existing query owner; do
not replace this with a new executor or a global document reorder. Cross-field
compositions still need explicit mapping. This is a proposed fix, not part of
the frozen-reader RGB benchmark.

The published site snapshot fetched during this run contains Lucene 10.3.0,
Lucene 10.3.0-bp and Tantivy 0.25. Across its 962 queries, geometric means of
per-query median top-10 times give 1.33× for plain Lucene / Lucene-bp and 1.63×
for Tantivy / Lucene-bp. Larger speedups depend on query subset/aggregation.
These figures are from the public host, not comparable absolute times to our
Cascade Lake host. The original [Lucene analysis](https://jpountz.github.io/2025/05/12/analysis-of-Search-Benchmark-the-Game.html)
explains the locality/pruning benefit; [Lucene's reorder API](https://lucene.apache.org/core/10_4_0/misc/org/apache/lucene/misc/index/BPIndexReorderer.html)
describes physical doc-ID reassignment. Preserve the downloaded raw data and
calculation with the separate RGB evidence.

The ARM diagnostic pass confirms the mapping cost: official top-10 posting-block
decodes grow 42,105 → 339,303, while exact-count decodes grow 68,583 → 14,851,059.
Exact-count compressed gap bytes processed grow 9,460,850 → 2,640,608,494; these
are repeated decoder inputs, not disk reads. Top-10 exact scoring units fall
835,690 → 566,549, so improved pruning does exist. Ranked unions are 17% faster
and top-10 for the single official stop-word query `the` is 2.77× faster. The
feature regresses overall because other paths discard that locality.

During full-corpus eligible-index construction, a sampled RSS of 11,010,036 KiB
included 10,999,876 KiB anonymous residency despite the configured 2,000,000,000
byte indexing budget. This is measured process residency, not an estimate of
live allocations; allocator-retained memory may contribute. Treat that setting
as a configured budget rather than a demonstrated RSS cap. Record final resource
usage separately from search-process residency; a writer memory audit remains
outside this read-path benchmark.

The mapped disjunction fallback has another avoidable cost:
`complete_text_scorer` accepts `skip_scoring_setup`, but its unordered-map branch
calls `required_text::scorer` without that intent. `RequiredTextScorer::position`
probes every term and computes BM25 even for count-only collection. Those direct
BM25 calls are not included in the generic exact-score diagnostic counter;
zero reported units on COUNT is not proof of zero scoring work there. Carrying
count-only intent through the existing owner is another required correction.

On the same ARM COUNT pass, TF block decodes rise 6,725 → 5,194,781 and decoded
TF values rise 834,484 → 662,866,934, while normalization-table builds remain zero.
Phrase verification needs some TF data; unordered mapped complete collection
adds repeated frequency decoding and direct scoring beyond that baseline.

The completed seven-pass full-corpus ranked runs confirm the same split.
TOP_10 compact/identity/RGB/Tantivy are 499.001/675.705/2410.780/517.486 µs;
TOP_1000 are 909.733/1106.567/3916.042/925.268 µs. RGB makes ranked intersections
36.0× slower (300 queries) and ranked unions 9.4% faster (301 queries). For the
single official term query `the`, top-10 improves 5002.494 → 290.489 µs (17.2×);
Tantivy is 1764.875 µs on that query. Do not generalize that one-query gain to the
workload. It demonstrates that stable IDs do not inherently eliminate RGB gains;
physical-order execution already benefits on the same full corpus and binary.

Full-corpus file manifests show compact 4,550.15 MiB, identity 4,598.14 MiB and
RGB 4,481.59 MiB. The mapping adds 47.99 MiB, but reordering shrinks postings
2005.28 → 1904.10 MiB and positions 2440.27 → 2425.66 MiB. Thus the old encoder
selection does not imply a larger full-corpus index: clustering compensates here.
The large ranked slowdown is execution overhead, not an increase in index bytes.
Representation-only effects remain unisolated in this end-to-end RGB comparison.

### RGB screening protocol adjustment

Before any full-corpus counted-command result was completed, the observed RGB
runtime showed that seven repeats would spend hours measuring the already
identified fallback. Preserve the completed seven-pass ranked results and use
three matched passes for counted commands on x86 (including supplemental terms).
All queries, engines, indexes, binaries, flags, correctness checks and warmup
rules remain identical. ARM retains its completed seven/five-pass runs. Restart
and rewarm all engines for the remaining commands; discard unfinished samples
and retain the interrupted run's configuration/logs. This is a screening result
for a regressing feature, not a claim of improved latency or changed defaults.
The separate x86 residency comparison uses one full pass per command for every
engine; report observed residency without claiming a long-run memory plateau.

The reduced counted run was subsequently stopped during its second timed RGB
pass. No full-corpus counted command completed its repetitions, so none is
reported as a latency measurement. The final x86 scope is seven-pass official
and five-pass supplemental top-10/top-1000, plus one separate residency pass for
each ranked command. ARM retains completed measurements for all four commands;
both full-corpus layouts passed all 1,676 exact references. Interrupted logs and
both protocol changes are retained. Full-corpus counted latency/residency remain
unmeasured. This limits the report; it does not establish a counted-query win.

The [execution repair proposal](maxscore-text-reordering.md#proposed-execution-repair)
records the address-space invariant and the existing conjunction admission point.
In particular, `skip_scoring_setup` currently preserves valid scores for nested
consumers. Avoiding count-only probes requires lazy valid scoring or an explicit
membership-only contract, not substituting zero scores based on that hint.

Final separate ranked residency samples are compact 759.81 MiB, identity
1115.71 MiB, RGB 1019.41 MiB and Tantivy 574.46 MiB RSS. Anonymous residency is
5.76/5.75/5.80/0.73 MiB respectively. RGB's increase over compact is chiefly
214.50 MiB more resident position pages plus 47.43 MiB more map/length pages;
the query heap is not the main contributor. Formats and traversal differ, so
this does not isolate permutation's effect on page residency. These are
one-pass ranked snapshots, not a memory plateau or counted-search measurement.
The source/control/preserved-payload audit and all 145 downloaded evidence
members passed verification. Eligible-index construction peaked at 15,615 MiB
RSS; standalone reorder peaked at 8,964 MiB. Writer memory budgeting warrants
its own investigation.

## RGB physical traversal repair — September 16

Implemented same-field physical composition and late stable-ID translation,
with the existing ranked conjunction executor and dictionary count paths admitted
for plain document maps. The precomputed ranked handoff avoids a second heap;
the validated dense inverse column replaces O(log N) probes for retained hits.
Cross-field, chunked and opaque compositions keep the existing logical fallback.

On the frozen 5,032,104-document x86 fixture, official RGB top-10/top-1000 are
440.615/885.654 µs, versus 2444.065/3873.167 µs for the old RGB reader and
524.345/987.633 µs for contemporaneous Tantivy. Repaired top-100+count/count are
725.922/342.566 µs versus Tantivy 869.111/424.969 µs. Old full-corpus RGB counted
timing remains unmeasured. Supplemental single-term results still trail Tantivy;
these aggregate official wins do not imply every query family wins.

ARM work counters reduce count decoding from 14,851,059 to 60,157 blocks and
TF decoding from 5,194,781 to 5,197 blocks. Phrase counts still require positional
verification. No index payloads changed. The remaining resident map/position
working set is not fixed by eliminating replay; report it separately from heap.
ARM latency has substantial between-run host drift, exposed by unchanged controls.

All 1,676 exact references pass on ARM compact/identity/RGB and x86 compact/RGB.
The final native harness passes 1,830 tests, formatting, Clippy and native without
sync; four focused diagnostics tests and portable compilation pass. The latter
retains the existing unused-method warning. WASM was not rebuilt as instructed.

The updated user requirement is RGB top-10 faster than Lucene BP/RGB. The matched
60-second-warmup run is 444.841 µs for Summa RGB versus 392.235 µs for Lucene
10.4.0 BP, a 13.4% deficit. The goal remains open. Unions account for the main
family gap (499.622 versus 325.063 µs); phrases favor Summa (486.759 versus
505.129 µs). An isolated required-window candidate regresses ARM AND and is not
promoted. Block pruning and a proven two-term OR-to-AND transition are experiments
to evaluate, not delivered wins.
[Implementation, measurements and raw evidence](search-rgb-repair.md).

### Matched Lucene continuation: isolated reader experiments

The full-corpus Lucene BP comparison remains the acceptance target. The first
mapped two-term AND block-bound probe is rejected: TOP10 442.875 → 441.498 µs
is flat, while TOP1000 836.338 → 848.838 µs regresses. Reducing the synthetic
block count from 64 to 4 did not establish a workload benefit.

The two-term-only OR-to-AND tail is also rejected (x86 TOP10 487.156 → 483.137,
ARM top-10 regresses). A guarded, once-per-query proof for any arity performs
better: x86 TOP10 441.735 → 428.431 µs, with Lucene RGB at 396.400 µs. Union
queries improve 495.87 → 458.39 µs; Lucene is 332.69 µs. TOP1000 is essentially
flat (837.864 → 841.411). ARM TOP10 is 29.519 → 29.349 µs. All 1676 ordered
ID/raw-score-bit/count references pass on both hosts and both fixed Summa
indexes. This remains a candidate pending selection and confirmation, not a
claim that Summa has surpassed Lucene.

Adaptive window sizing, mapped batch admission and local score-required union
driving are separately frozen experiments. Their source and measurements stay
outside the main implementation until assessed. The latter revisits a previously
rejected approach only because candidate probing now scores matching postings
rather than the entire loaded block, and mapped execution retains RGB locality.

### Selected RGB reader and representation repairs

The general guarded OR-to-AND tail and mapped batch admission are selected.
The latter leaves official top-10 flat in isolation but reduces supplemental
single-term top-10 from 100.745 to 67.505 µs on x86 (Lucene RGB: 86.588 µs).
Supplemental top-1000 improves 839.725 to 744.854 µs. These are separate workload
results, not evidence of an official-workload Lucene win.

Adaptive windows and local required-term classification are rejected: x86
TOP10 is 444.663 → 445.214 and 442.069 → 440.835 µs respectively. The narrower
SIMD intersection within required OR windows also regresses (442.112 → 459.126).
Profiles explain why fewer candidates alone are insufficient: baseline unions
spend 16.04% of sampled CPU in window orchestration, 13.51% in candidate probing,
10.50% in scoring, and 7.82% in block bounds. Required-window classification adds
bookkeeping without reducing candidate cost materially. These samples are
attribution, not latency measurements.

The standalone writer now preserves compact input layouts. The regression first
failed on lost compact headers, then passed with exact score bits, counts,
positions and byte-identical untouched fields. On the full corpus, the rebuilt
index retains exactly the original RGB permutation while saving approximately
192 MiB in postings/positions. Query latency and RSS require separate measurement.
Graph-frequency and finer-partition variants remain isolated experiments.

The selected main source passes 1,833 native tests (25 ignored), formatting,
Clippy, native without sync and portable compilation. The existing portable
unused-method warning remains; WASM is skipped under the standing instruction.

A first admission-probe cloud build reused stale artifacts because extracted
source mtimes predated the preceding binary. Its identical binary hash exposed
the mistake; that run was stopped and quarantined. All accepted later cloud
builds touch extracted sources, record distinct binary hashes, and compile anew.
An ARM warmup overlapping a compile was likewise discarded and rerun. Neither
rejected run contributes to reported measurements.

### Full-corpus RGB layout comparison

The eight-engine follow-up confirms the selected reader at 424.943 µs official
top-10, versus the handoff reader's 441.718 and Lucene RGB's 393.534. The target
remains unmet by 8.0%. Top-1000/top-100+count/count are 839.102/724.421/349.461 µs,
versus Lucene 854.396/1078.629/378.291. Supplemental top-10 improves from 103.642
to 69.277 µs and beats Lucene's 86.756, but still trails Tantivy's 56.148.

A same-permutation compact rewrite is 429.560 µs top-10. Frequent-term and finer
partition graphs are 432.040/432.673 µs; both are rejected for query performance.
ARM ranked results are flat. The graph default remains unchanged.

Fresh top-10 RSS is 1019.39 MiB for the selected reader on the original RGB index,
1055.73 MiB for compact RGB, 745.95 MiB for RGB off, 573.92 MiB for Tantivy and
860.11 MiB for Lucene. Summa anonymous memory stays near 5.6 MiB. Compact output
saves 192.2 MiB on disk but reduces position residency by 95.88 MiB while increasing
posting/dictionary residency by 114.69/17.68 MiB. This is not a resident-memory win.
The larger mapped working set is established; its page-fault mechanism is not.

All eight-engine timing samples, memory snapshots and immutable index audits are
verified in the evidence. A PATH-resolution error in the Java binary inventory
was fixed before memory queries; forty timing files remained hash-identical
through recovery. The expanded mapped two-term regression is now in the main
source; the latest native harness again passes 1,833 tests, 25 ignored. It changes
only test code relative to the measured reader.

An existing-codec experiment measures Simd4x on the same RGB permutation, with
exact norms and existing ratio bounds. ARM official top-10 improves
30.043 → 28.549 µs; x86 is flat (compact 474.245, SIMD 473.595, Lucene 440.889 µs).
The target remains unmet. Full-corpus compact/SIMD index sizes are
4,289.08/3,117.26 MiB; fresh top-10 RSS is 1055.71/855.82 MiB, versus original
RGB 1018.90 and Lucene 859.91 MiB. Anonymous Summa memory remains about 5.5 MiB.
This is a measured storage/residency benefit, not an x86 latency win. The codec
remains opt-in. All exact references and permutation/unchanged-payload audits
pass; both fixture and comparison exports are locally hash-verified.

### Selected two-term contribution elision

Mapped two-term windows with validated finite, nonnegative scoring now reuse
their accumulated score, omitting duplicate per-term writes and canonical
reduction. Two-term addition retains exact score bits; longer/unsupported cases
keep the original reduction. All 1,676 references pass on RGB and RGB-off
indexes on ARM and x86. The expanded mapped regression remains in main.

Matched x86 official top-10 improves 421.535 → 414.659 µs, versus Lucene
392.170 µs; top-1000 improves 833.542 → 819.923 µs, versus Lucene 860.651 µs.
Every paired top-10 pass improves. ARM top-10 improves 30.120 → 29.580 µs.
Supplemental x86 top-10/top-1000 are essentially flat. The target remains unmet
by 5.7%. The selected source passes 1,833 native tests (25 ignored), formatting,
Clippy, native without sync and portable compilation. WASM remains skipped.
The delivered scorer is production-byte-identical to the measured pair-sums
source; only the existing mapped regression is expanded.

The separate SIMD candidate-join probe is not selected: x86 top-10 is
427.854 → 429.181 µs, although top-1000 improves 830.612 → 817.965 µs.
Local window boundaries likewise fail to improve x86 ranked latency
(top-10 440.348 → 441.180 µs; top-1000 847.646 → 849.651 µs) and are rejected.
The combined-window probe is not selected (x86 top-10 424.858 → 421.053 µs,
ARM top-10 flat). Canonical reduction with a term-major loop regresses x86
top-10 419.165 → 422.230 µs and top-1000 826.648 → 834.807 µs; it is rejected.
Their exact references pass and both 30-file exports are locally verified.

### Scheduling attribution experiment (not a production candidate)

The synchronous searcher also installs single-segment work on the shared Rayon
pool. An isolated caller-thread experiment retains the same segment closure,
statistics, IDs, offsets, errors and collectors, while async/multi-segment/BMP
execution stays pooled. A thread-affinity regression fails before and passes
after, and also checks async equivalence, offsets and error propagation.

This experiment is for attribution only: concurrent synchronous callers would
bypass the configured shared CPU bound. It is not eligible for selection even
if serial latency improves. Main retains pooled scheduling. Any production
follow-up must preserve bounded admission across sync, async, nested and parallel
searches; a serial benchmark cannot justify weakening that contract.

The caller-thread diagnostic measures x86 top-10 419.414 → 411.142 µs
(Lucene 393.957) and top-1000 817.116 → 802.778 µs. It still misses the target.
ARM official top-10 regresses 29.028 → 34.460 µs despite faster supplemental
terms. These timings include thread placement and scratch locality, not only
the cost of a queue operation. The source remains isolated and rejected.

A final local release build initially reused the same-name example artifact
from an isolated experiment sharing the target directory. Its binary hash
identified the mismatch before any results or timing used it. The source hashes
were checked, selected source mtimes refreshed, and a fresh build completed.
The resulting binary is byte-identical to the measured ARM pair-sums binary.
It also passes all 1,676 same-binary pruned-versus-exhaustive score/count checks
on compact RGB-off, original RGB and SIMD RGB fixtures.
This reinforces using immutable copied binaries for measurements and separate
target directories when switching checkouts. Native validation and the accepted
paired measurements did not use that stale final artifact.

### Coarse ratio group admission result

Enabling the existing group-skip proof for mapped ratio-bounded lists reduces
fine-window work in the regression fixture, preserving exact scores and stable
ties across all four codecs. It is rejected on actual workloads: ARM top-10
29.183 → 29.522 µs; x86 top-10 460.574 → 474.224 µs and top-1000
866.730 → 891.303 µs. All exact references pass. Fewer fine windows alone do
not pay for the added coarse-bound checks on this corpus. Main keeps the
existing impact-only admission and the selected two-term contribution repair.

### Final selected-reader confirmation

The four-engine final run confirms selected/original RGB at 416.881 µs top-10
versus predecessor 422.282 and Lucene RGB 390.625: a 1.3% reader improvement,
but the target remains unmet by 6.7%. Top-1000 is 824.472 versus Lucene 848.283.
Selected/SIMD top-10 is 425.249 µs, 2.0% slower than selected/original RGB; the
codec remains an opt-in space/residency tradeoff. Fresh top-10 RSS is 1019.28 MiB
for selected/original, 856.00 for selected/SIMD and 864.87 for Lucene. Summa
anonymous memory stays near 5.5 MiB.

Supplemental selected top-10 wins 70.099 versus Lucene 88.379 µs, while
top-1000 still loses 752.598 versus 475.309. These results cannot be pooled with
official queries to claim the requested top-10 win. All 145 exported files are
locally hash-verified; the four-engine query/pass matrix is independently checked.
The selected x86 SIMD combination passes all 1,676 canonical references after
timing. Native validation remains 1,833 passing tests, 25 ignored; portable
compilation passes, and WASM remains skipped.

A requested late control expansion was refused by an internal guard because
final latency had started; it stopped before any process or script mutation.
The final run remains the original four-engine protocol. Final pair-sums x86
RGB-off latency is unmeasured; paired ARM RGB-off controls and the earlier
eight-engine x86 reader results are separately identified.

## September 17: merge preparation review

See [the merge review](search-merge-review.md) for scope, corrections, BMP transfer
analysis and final validation. The branch now includes `origin/main` at
`b6212939`; content-hash deduplication and the search format stamp are reconciled.

Reproduced and corrected: untouched byte norms expanded by standalone RGB;
mixed exact/byte norm columns expanded by row compaction; insufficient text BP
scratch admitted before allocation; BMP-only merge status claiming text was
reordered; and legacy plain RGB silently skipping fields without document maps.
Text planning reuses the BMP frequency/budget helpers; term rewriting reuses
bounded compaction readers/writers. Raw benchmark archives remain unchanged.

WASM was explicitly re-enabled for this review. The stale native SIMD fixture
was regenerated with valid short-block tags, and a compact-directory/byte-norm
fixture was added. This review makes no new latency/parity claim. Integrated
merge-time text planning and the measured Lucene top-10 gap remain open.

Final review validation passes the eight-stage `full` harness: 1,856 workspace
tests, 26 ignored in that stage, plus four real server/broker tests. WASM passes
32 tests; targeted query diagnostics pass six. The final local reorder control
reduces peak process RSS from 119.2–122.0 to 35.5–36.3 MiB on 131,072 documents,
with identical payload bytes and exhaustive query checks. Timings are short and
variable (before 0.24/0.24 s, after 0.43/0.22 s); no speedup is claimed. This
measurement caught and corrected output fragmentation from copying position
blocks across permuted document boundaries. Ordinary compaction still copies
compatible blocks; explicit reorder repacks through the existing encoder.
Per user instruction, benchmark artifacts remain local and uncommitted.

## September 17 follow-up: integrated merge and remaining RGB gap

Integrated merge-time text RGB is complete. Opted-in text fields are planned
over the source segments and written once in their final physical order. The
implementation reuses BMP's BP selection/budget helpers and CPU gate, copy
merge's k-way dictionary traversal, standalone text rewriting, and the existing
posting/position/CHNK formats and writers. Ordinary merges still copy compatible
encoded payloads. No second output generation or publication protocol is added.

The [merge review](search-merge-review.md#follow-up-integrated-merge-time-text-rgb)
records byte-equivalence, query correctness, budget/cancellation coverage and
measurement limits. All eight `full` stages pass: 1,859 workspace tests,
26 ignored, plus four real server/broker tests. A fresh WASM build passes all
32 tests. The same-binary 131,072-document ARM lifecycle comparison takes
0.30/0.31 s for merge followed by reorder versus 0.22/0.22 s for integrated
merge-time RGB; peak RSS is 43.31/44.80 versus 38.19/31.56 MiB. Payloads match
byte-for-byte. This does not establish a query-latency improvement.

### Exact mapped normalization lookup: rejected

The frozen RGB index stores exact u16 document lengths in its physical map.
Its lengths cannot be passed through the existing byte-norm lookup without
changing scores. An isolated candidate instead caches one exact 4,096-entry
normalization table per worker, keyed by the raw BM25 parameter/average bits.
It costs 16 KiB per table and retains canonical arithmetic for larger lengths.
Active cursors retain their table; cache replacement cannot invalidate a query.
This experiment is confined to `.context/rgb-norm-lookup/candidate-src/` and is
**not included in the implementation**.

Both frozen RGB-on and RGB-off indexes pass all 1,676 canonical references on
ARM and x86 for the candidate (exact IDs, score bits, ranked limits and counts).
The candidate's edge/cache regression and 43 scoring unit tests also pass.
Correctness alone does not justify selection:

| Official workload, geometric mean of per-query medians | Current Summa | Lookup candidate | Lucene RGB |
| ------------------------------------------------------ | ------------- | ---------------- | ---------- |
| ARM top-10, µs                                         | 29.386        | 29.039           | unmeasured |
| ARM top-1000, µs                                       | 44.402        | 44.381           | unmeasured |
| x86 top-10, µs                                         | 426.918       | 470.017          | 400.954    |
| x86 top-1000, µs                                       | 867.289       | 931.035          | 882.210    |

The candidate regresses x86 top-10 by 10.1% and top-1000 by 7.4%, despite a
small ARM improvement. It is rejected without changing defaults or query code.
Current Summa remains **6.5% slower than Lucene RGB on official top-10** and
1.7% faster on top-1000. The official x86 category split isolates the remaining
deficit:

| Top-10 category, µs | Summa   | Lookup candidate | Lucene RGB |
| ------------------- | ------- | ---------------- | ---------- |
| AND                 | 308.930 | 333.310          | 315.764    |
| Phrase              | 497.401 | 537.676          | 506.094    |
| OR                  | 449.058 | 519.244          | 335.619    |

OR queries remain 33.8% slower; the wider exact-length lookup worsens that path.
These timings do not isolate whether lookup dependencies, cache pressure,
instruction layout or vectorization caused the regression. Hardware performance
counters are unavailable on this machine (`cycles` reports no supported events).

A separate 15-second userspace CPU-clock sample over official OR queries
(499 Hz, after warmup, no lost samples) attributes 14.59% of baseline CPU to
`run_text_windows`, 10.79% to `score_candidates_sync`, 10.15% to
`run_conjunction`, 10.15% to `score_text_run`, and 7.27% to block-bound evaluation.
The candidate's `score_text_run` share rises to 12.62%. These sampled shares
support investigating scoring, candidate probing and window/bound overhead;
they are not isolated per-function speed comparisons. Position decoding is not
a leading symbol in this OR workload. The OR-only resident snapshot is
353.25 MiB with 5.29 MiB anonymous memory for the baseline; candidate anonymous
memory is 5.32 MiB. Most residency is file-backed, not query heap. The full
official workload below additionally touches phrase position pages.

The protocol uses immutable executables built from separate source snapshots,
the same frozen index, Rust 1.98.1, native CPU flags and release LTO. It rotates
engine order for seven passes over all 962 official queries; x86 engines are
pinned to CPU 2. Builds, canonical verification and profiles are outside timing.
Peak RSS over the official top-10 plus top-1000 process lifetime is
1,012.19 MiB for Summa, 1,011.87 for the candidate and 906.41 for Lucene.
These are process maxima, not isolated heap sizes or a top-10-only comparison.

The separate 714-query supplemental workload uses five passes: x86 top-10
73.985/83.038/91.018 µs and top-1000 804.620/876.217/474.109 µs for
Summa/candidate/Lucene. It is not pooled with official queries to claim parity.
Source and binary hashes, complete query/pass matrices, canonical checks and
raw measurements remain local under `.context/rgb-norm-lookup/`; the downloaded
x86 archive's SHA256 and every query/pass matrix were verified. No benchmark
artifacts or implementation changes have been committed.

### September 17: optional probing and window work

The next investigation keeps the frozen RGB/RGB-off indexes and canonical raw
score contract. Candidate source snapshots, executable hashes, verifier logs,
per-query/per-pass samples and experiments are local under
`.context/rgb-or-cost/`; none are commit inputs. The production candidates reuse
`TermCursor`, `score_text_run`, the existing block-bound cache and window scratch.
They add no persistent format or cache allocation.

Initial matched screens (geometric mean of per-query medians, µs):

| Official workload | Baseline | Optional probe specialization | Combined bound division | Lucene RGB |
| ----------------- | -------: | ----------------------------: | ----------------------: | ---------: |
| ARM top-10        |   29.609 |                        29.243 |                  29.268 | unmeasured |
| ARM top-1000      |   45.262 |                        44.655 |                  44.715 | unmeasured |
| x86 top-10        |  441.006 |                       428.601 |                 444.719 |    396.455 |
| x86 top-1000      |  885.299 |                       878.999 |                 912.756 |    900.657 |

The probe specialization avoids optional candidate compaction, skips lower-bound
search when the cursor is already at or beyond the candidate, and stores the
128-entry block's posting slots as bytes. Required membership still compacts;
block-level and periodic candidate deadline checks remain. The bound-division
candidate passes correctness but regresses x86 and is rejected. An eight-lane
filter mask also fails selection: x86 top-10 is flat and top-1000 regresses 2.1%.
Neither rejected implementation is in production source.

All-essential windows can visit terms in canonical order and omit duplicate
contribution-plane writes and the final fold. An initial x86 screen suggested
an improvement, but a seven-pass confirmation of this plus probe specialization
measured **451.391/455.251/428.735 µs** for baseline/combined/Lucene top-10.
Top-1000 was 836.921/833.715/866.941 µs. This does **not** establish a win or
parity. Whole-pass timing drift motivates a separate stability run interleaving
engines every 32 queries and recording Linux process CPU time alongside wall
time. These protocols must not be pooled. A later ARM screen is excluded because
unrelated system/build load changed baseline timings substantially.

A separate proposal to replace globally optional window bounds with list bounds
is rejected for correctness: `plus size clothing` stops making progress when a
loose bound promotes an optional cursor that has already passed the window and
all remaining drivers are demoted. Unit tests alone missed this; the complete
1,676-query verifier exposed it. That candidate was isolated and never copied
into production. Any future version needs an explicit window-progress invariant.

The probe, combined-division, filter and canonical-accumulation candidates each
pass all 1,676 canonical reference queries on both frozen layouts on ARM and x86
(exact IDs, score bits, ranked limits and counts). This correctness evidence does
not override the rejected candidates' performance results.

The closer paired combined run uses seven official and five supplemental passes,
rotating engines every 32 queries on CPU 2. The process CPU clock is read through
libc's `clock_getcpuclockid`; an initial harness API error occurred before any
samples and is retained separately. Official wall times are
435.270/432.335/416.160 µs for baseline/combined/Lucene top-10 and
843.557/837.377/883.282 µs for top-1000. Corresponding top-10 process CPU times
are 424.598/421.427/389.788 µs. OR wall times remain
457.136/450.503/347.604 µs: a small improvement in this paired run, with the material OR
gap still open. Supplemental wall times are 115.386/110.777/131.411 µs top-10
and 808.715/801.428/521.707 µs top-1000; they remain a separate workload.

Memory is effectively unchanged by the combined change. After official top-10,
baseline/candidate RSS is 1,012.11/1,012.27 MiB, of which only 5.79/5.79 MiB is
anonymous. Lucene's corresponding RSS is 877.76 MiB with 334.42 MiB anonymous.
Summa' larger RSS here is file-backed residency, not a larger query heap.
These full-workload snapshots touch phrase positions too. PSS must not be used
as an engine comparison here: concurrent Summa processes share the same mapped
index pages. Peak RSS over both ranked limits is 1,012.24/1,012.33/901.91 MiB.
The downloaded stability archive and all query/pass matrices were verified;
SHA256 `9c2882482b08c8d4e9d753ff2bba055a0194c9c4311594e5e209dcd2385b56c1`.

#### Paired screens of remaining window costs

Each row below is a separate matched x86 run on the frozen 5,032,104-document
RGB fixture, Rust 1.98.1 with native CPU flags and release LTO, CPU 2, rotating
engine order every 32 queries. Values are geometric means of per-query median
microseconds (seven passes over 962 official queries). Do not compare absolute
values across rows: host timing drift is substantial. These are selection
screens, not independent confirmation of a final production build.

| Screen                               | Baseline top-10 | Candidate top-10 | Lucene top-10 | Baseline top-1000 | Candidate top-1000 | Lucene top-1000 |
| ------------------------------------ | --------------: | ---------------: | ------------: | ----------------: | -----------------: | --------------: |
| Current block covers window          |         435.686 |          428.155 |       417.308 |           857.471 |            843.187 |         892.914 |
| Adaptive mapped OR windows           |         434.766 |          426.666 |       420.097 |           838.355 |            824.190 |         886.037 |
| Cost-aware local partition           |         446.929 |          436.073 |       425.965 |           870.953 |            859.310 |         908.281 |
| Consecutive candidate TF slice       |         475.342 |          473.128 |       460.666 |           863.124 |            849.796 |         901.363 |
| Pruned two-term OR intersection tail |         443.890 |          434.088 |       424.740 |           857.399 |            847.400 |         891.653 |

The covered-window shortcut preserves the exact old bound, including the
whole-group substitution and zero clamp; it only avoids redundant navigation.
The adaptive policy amortizes tiny multi-driver windows, with scratch still
capped at 4,096 IDs. Cost ordering changes candidate drivers using the existing
prefix-bound proof. These candidates pass all 1,676 reference queries on both
RGB and RGB-off layouts on ARM and x86.

The TF-slice experiment is rejected: its own same-run adaptive parent measures
469.602/846.406 µs, better than its 473.128/849.796. The pruned tail's incremental
benefit is also inconclusive: its adaptive parent measures 435.993/844.151 µs,
so the tail improves top-10 slightly but regresses top-1000. It remains isolated.
Its behavior regression proves that a strong-prefix fixture needs only 256 of
8,192 matches scored, while equal-score ties still visit all matches and exact
count traversal always returns 8,192. The isolated tail passes 45 scoring tests.
The supplemental workload is retained separately in the raw evidence; it is
not pooled to claim a win over Lucene.

ARM uses its frozen approximately 100,000-document fixture, not the x86 corpus.
The adaptive screen measures 30.624/30.403 µs top-10 and 46.564/46.099 µs
top-1000 against baseline: near flat. Its supplemental top-1000 run is excluded
because severe external load changed baseline timing by an order of magnitude;
the raw samples and exclusion reason are retained. A later cost-ordering ARM
screen was also variable and does not establish a gain. Tail's ARM top-10 is
30.239 versus its adaptive parent's 30.214 µs; top-1000 is 46.061 versus 45.597.
No configuration or codec default changes are supported by these measurements.

Mandatory validation attempts before final selection encountered two distinct
failures: one native test process terminated with SIGTERM, and a later harness
run passed formatting/Clippy but timed out during broker fixture discovery.
The exact broker test immediately passed in isolation (0.31 s); the timeout's
cause is not established. These attempts do not count as completed checks.
The final selected source requires a fresh complete harness and WASM validation.

Bulk candidate probing through the shared intersection kernel passes 44 scoring
tests and all canonical references on ARM/x86 and both layouts, but is not
selected. Its ARM top-10/top-1000 screen improves 30.422/45.703 to
30.153/45.259 µs. In the x86 run, however, bulk measures 433.479/844.057 µs
against its adaptive parent's 430.171/842.523 µs (baseline
439.740/857.535, Lucene 414.675/887.927). A small ARM gain does not justify
regressing the target architecture. Its supplemental improvements are kept
separate from that decision.

The combined adaptive/cost policy passes 44 scoring tests and all ARM references.
Its ARM screen measures 28.987/44.594 µs top-10/top-1000 against baseline
29.485/45.151 and adaptive parent 29.084/44.731. Supplemental results are
16.197/36.740 versus baseline 16.553/37.020 µs. The cloud runner initially waited
on its own completion marker due to an orchestration typo; it was corrected
before any build or samples. The source snapshot and binary are unchanged by
that runner correction. Final cross-architecture selection is still pending.

The cost/window merge candidate completes `python3 scripts/check_search.py check`
with `RUST_TEST_THREADS=2`, `CARGO_BUILD_JOBS=2` and incremental compilation off:
formatting, Clippy with warnings denied, **1,860 tests passed** (26 ignored,
including the four separate real-server E2E tests), and native without sync.
Evidence is `.context/search-harness/20260917T102649.676426Z-check/`.
The final-source WASM build and **32 tests in seven files** pass. Existing
missing-LICENSE and newer-wasm-pack notices remain nonfatal. `full` is not rerun
for this scorer-only change; the preceding lifecycle review's full run remains
separate evidence. Any subsequently selected ranked-conjunction change requires
validation again on that source.

The combined adaptive/cost x86 screen completes all canonical references on both
layouts. Official top-10 is 472.087/467.834/463.928/453.361 µs for
baseline/adaptive/combined/Lucene; top-1000 is
905.303/889.949/895.359/930.728 µs. Combined improves baseline by 1.7% at top-10
but remains 2.3% behind Lucene, and trades a 0.6% top-1000 regression against
adaptive for its 0.8% top-10 improvement. Supplemental top-10 is
86.903/81.035/78.672/100.134 µs; top-1000 is
874.246/870.557/870.111/506.670 µs. This is still a screen, not the final
independent confirmation. The isolated ranked-conjunction extension passes 45
scoring tests; the complete public counted path remains exhaustive.

The ranked mapped-pair extension passes all 1,676 reference queries on ARM and
x86 for both frozen layouts. Its ARM screen is effectively flat: official
base/combined/ranked times are 30.144/29.868/29.835 µs at top-10 and
44.836/44.732/44.616 µs at top-1000. Supplemental times are
15.728/15.770/15.697 and 37.152/37.137/36.858 µs. The public Boolean planner
routes ranked counts through `execute_counted_conjunction`, whose compile-time
pruning flag remains false. Count-only collection rejects the ranked shortcut
at k = 0. Both caller contracts were traced before testing this extension.

#### Selected reader and independent confirmation

The adopted source combines optional membership specialization, canonical
all-essential accumulation, covered-block bounds, bounded adaptive mapped
windows, cost-aware local ordering, and ranked mapped-pair pruning. All changes
remain in the existing query/scoring owner. The bulk membership join, consecutive
TF slice, combined-bound arithmetic, vector filter, loose optional bounds and
large norm lookup remain isolated/rejected; none are copied into production.
No format, default, persisted bytes, scorer owner or retained cache is added.
The shared BP/reorder and BMP implementations are unchanged by these text-only
traversal changes; the typed text cursor and BM25 proofs do not apply to BMP's
separate sparse block scoring without a separate measured policy.

In the final selection screen, baseline/cost-window/ranked/Lucene x86 top-10 is
450.381/446.033/437.665/428.347 µs, with top-1000
891.691/882.465/884.199/923.907 µs. Ranked improves top-10 1.9% over its
parent while trading 0.2% at top-1000; it remains 2.2% behind Lucene at top-10.
The corresponding top-10 process CPU times are
431.881/426.893/419.773/395.156 µs, so wall-time proximity is not CPU parity.
OR wall times are 475.711/459.380/446.540/362.690 µs; AND is
328.086/330.915/320.567/343.144 µs. The remaining deficit is concentrated in OR.
Supplemental top-10 is 81.529/75.164/72.907/95.360 µs and top-1000
790.927/787.337/780.161/486.013 µs; these queries remain a separate workload.

The production scorer was adopted byte-for-byte from the `ranked-src` snapshot.
The independent confirmation reuses that frozen executable, avoiding a new
compiler/layout draw after selection. Its source identity is recorded in
`.context/rgb-or-cost/accepted-source.json`; all artifacts remain gitignored.
Confirmation covers all four commands on both layouts, with seven official and
five supplemental passes, process CPU time and resident-memory snapshots on
x86, and separate ARM measurements. Final-source harness and WASM checks are
rerun after local timing so our builds do not contaminate the ARM samples.

Independent ARM confirmation on its frozen smaller corpus (µs, seven official
passes; same binaries, flags and 32-query interleaving):

| Official command      | RGB before | RGB adopted | RGB-off before | RGB-off adopted |
| --------------------- | ---------: | ----------: | -------------: | --------------: |
| Top 10                |     30.418 |      29.995 |         32.651 |          32.049 |
| Top 1000              |     45.858 |      45.414 |         48.569 |          47.517 |
| Top 100 + exact count |     34.982 |      34.814 |         36.533 |          36.322 |
| Exact count           |     24.821 |      24.514 |         25.173 |          25.003 |

These are modest improvements, not a universal ARM speed claim. Count-only traversal remains
unchanged; sub-percent shifts do not establish a causal speedup. Ranked-plus-count
collection may reuse the improved ranked pass after an independent exact count.
The supplemental workload remains separate: RGB before/adopted is
16.428/16.143, 37.033/36.905, 23.723/23.117 and 9.685/9.396 µs in command
order; RGB-off is 18.901/18.749, 37.598/37.505, 27.483/26.932 and
9.781/9.388 µs. No compilation ran during these local measurements. Complete
query/pass matrices and `/usr/bin/time -l` process maxima are preserved under
`.context/rgb-or-cost/arm-confirm-*`.

Final adopted-source validation completes successfully:
`.context/search-harness/20260917T104951.078336Z-check/` contains formatting,
Clippy with warnings denied, **1,861 passing native tests** (26 ignored) and
native-without-sync checks. The rebuilt WASM package passes **32 tests in seven
files** (`accepted-wasm-build.log`, `accepted-wasm-tests.log`). The adopted
scorer SHA256 is
`d8e1bfc38bd068ce210401ddf3625c6f4dad58c3bad82a397e6fe9100e09a1ff`.
The full lifecycle/RPC mode is not rerun for these scorer-only changes.

#### Final independent x86 results

The Google Cloud session expired briefly during the run and was reauthenticated.
All four commands completed on both layouts; the full archive and its raw
query/pass matrices have now been downloaded and verified. Same frozen corpus,
compiler/flags, CPU 2 and cache budgets; seven official and five supplemental
passes with 32-query engine rotation. Values are geometric means of per-query
median microseconds.

| Official command      | RGB before | RGB adopted | Lucene RGB | RGB-off before | RGB-off adopted |
| --------------------- | ---------: | ----------: | ---------: | -------------: | --------------: |
| Top 10                |    441.487 |     432.761 |    420.303 |        534.265 |         529.659 |
| Top 1000              |    854.970 |     856.225 |    878.540 |        923.156 |         910.948 |
| Top 100 + exact count |    758.613 |     734.300 |   1111.100 |        857.318 |         846.729 |
| Exact count           |    365.429 |     364.564 |    399.772 |        415.177 |         415.922 |

RGB top-10 improves **2.0%**, but is still **3.0% slower than Lucene**. Top-1000
is flat against baseline (+0.15%) and 2.5% faster than Lucene. Top-100 plus exact
count improves 3.2%; count-only is effectively flat. RGB-off improves 0.9% at
top-10 and 1.3% at top-1000. These results do not establish Lucene parity.

The official RGB top-10 families before/adopted/Lucene are
465.569/441.818/351.736 µs for OR, 319.986/316.231/334.779 µs for AND and
514.607/520.426/526.713 µs for phrases. OR improves 5.1% but remains 25.6%
behind Lucene. Phrases regress 1.1% against baseline while remaining faster than
Lucene in this run. The aggregate win must not hide that tradeoff.

Process CPU means for the four RGB commands are
428.701/419.037/390.979, 830.980/831.209/833.961,
704.896/684.598/1047.784 and 324.054/323.885/359.301 µs for
before/adopted/Lucene. Top-10 CPU improves 2.3% but remains 7.2% behind Lucene;
wall-time proximity is not CPU parity. These clocks include process bookkeeping,
not just the scoring kernel. RGB-off CPU before/adopted is 516.716/513.008,
907.180/894.977, 799.132/788.341 and 375.023/375.718 µs.

Supplemental results remain separate:

| Supplemental command  | RGB before | RGB adopted | Lucene RGB | RGB-off before | RGB-off adopted |
| --------------------- | ---------: | ----------: | ---------: | -------------: | --------------: |
| Top 10                |     78.793 |      73.639 |     93.859 |        140.270 |         140.005 |
| Top 1000              |    787.764 |     779.449 |    474.634 |        609.264 |         618.607 |
| Top 100 + exact count |    240.691 |     233.196 |    460.460 |        295.658 |         298.957 |
| Exact count           |     19.176 |      19.057 |     18.857 |         18.998 |          19.037 |

RGB-off supplemental top-1000 regresses 1.5%, and ranked-plus-count regresses
1.1%. Supplemental RGB top-1000 remains substantially slower than Lucene; no
pooled official/supplemental result is used to claim a win.

#### Final memory and evidence

After official RGB top-10, baseline/adopted RSS is **1,012.215/1,011.707 MiB**;
anonymous residency is **5.789/5.793 MiB**. Lucene's RSS is 881.602 MiB, with
338.281 MiB anonymous. The adopted reader does not materially change memory.
Summa' larger RSS is predominantly mapped index pages, not query heap growth;
anonymous residency includes more than heap alone. Phrase queries also touch
position pages. These are the unchanged frozen fixtures, not a claim that every
RGB encoding has this residency. PSS is unsuitable for comparing these processes
because the two Summa executables share mapped index pages.

RGB-off baseline/adopted RSS is 736.605/736.496 MiB, with
5.773/5.770 MiB anonymous. Lifetime peak RSS across all four official commands is
1,037,152/1,036,500/927,536 KiB for RGB baseline/adopted/Lucene and
768,152/768,172 KiB for RGB-off baseline/adopted. Peaks are not isolated top-10
measurements and must not be presented as heap sizes.

The verified late archive contains **653 manifest-checked files**, **28 complete
query matrices**, and **1,237,968 timing samples** spanning the later screens and
final confirmation. SHA256:
`650af8ffed25acbdc919224cb3d7d9d1dd61adf4052eec670fd1e7d0aa76a80a`.
The final matrices contain all four commands on both layouts; their binary hashes
match the frozen adopted executable, and the archived production scorer matches
the workspace byte-for-byte. Verification and source identity are saved under
`.context/rgb-or-cost/late-evidence-verification.json` and `accepted-source.json`.
Benchmark artifacts remain gitignored. No commits or pushes were made.

The machine is confirmed **TERMINATED**. The stop command lost its connection while
polling the operation, but an independent `instances describe` completed with
status `TERMINATED`; the result is recorded in `machine-stop-confirmed.json`.

### September 17 — cleanup after acceptance of the measured results

Scope is source organization and reuse; the accepted algorithms and settings
remain fixed. `query/scoring.rs` keeps public construction/dispatch, shared
cursor state, heap, and scratch. Private `scoring/conjunction.rs` owns typed
intersection and counted/ranked admission; `scoring/windows.rs` owns bounded
windows, local partitions, and canonical reduction. Both are implementations of
the existing executor. `scoring/tests.rs` retains the existing regression suite
and module paths. The parent shrinks from 6,426 to 2,674 lines.

Ranked conjunction entry and a union's proven conjunction tail now call the
same `run_ranked_conjunction` helper. Exact counted traversal still explicitly
calls `run_conjunction::<false>`. Admission conditions, scoring arithmetic,
strict comparisons, cancellation points, scratch capacities, public visibility,
and persisted representations are preserved. No new writer, scorer, cache,
allocation, or dynamic dispatch is introduced. The different proofs for
nonnegative window accumulation and ranked-pair pruning remain separate.

Review of the surrounding owners confirmed that term/cursor/intersection scoring
already shares `score_text_run` and `NormTable`; compact codecs reuse the existing
horizontal tail codecs; text reorder uses the shared graph-bisection planner,
merged-term traversal, and posting/position writers. These owners do not need
another abstraction for this cleanup. BMP's different score semantics remain
outside the text-only pruning admission.

Movement evidence is under `.context/scoring-cleanup/`: the retained parent is
unchanged after accounting for the extracted regions, the test file matches the
original after dedenting/rustfmt, and reviewed traversal diffs contain only the
shared dispatch extraction plus paths/visibility/formatting. The before snapshot
matches the accepted frozen scorer. All cleanup benchmark scripts, binaries,
source snapshots, and raw measurements stay in this ignored directory.

Validation for this cleanup:

- `python3 scripts/check_search.py check`: passed, including formatting,
  Clippy with warnings denied, **1,861 native tests** (26 existing ignored), and
  the native-without-sync all-targets compile. Evidence:
  `.context/search-harness/20260917T111243.307467Z-check/`.
- Focused scoring suite: **45 passed**. Documentation checker: 118 Markdown
  files, 558 local links, and 20 benchmark targets passed.
- The lifecycle/RPC `full` harness is not repeated: this cleanup changes only
  query implementation organization. No lifecycle, RPC, encoding, or merge
  implementation changes are included.
- WASM release build and all **32 tests in 7 files** passed. The build emitted
  the existing nonfatal missing-LICENSE and newer-wasm-pack notices. Logs:
  `.context/scoring-cleanup/wasm-build.log` and `wasm-tests.log`.
- Release exact-reference checks passed for **1,676 queries on each local
  layout**, RGB on and off: ranked IDs/raw score bits at k=10/100/1000,
  complete top-100, and exact counts. Logs: `verify-rgb.log` and `verify-off.log`
  under `.context/scoring-cleanup/`.
- The cloud/x86 benchmark is not rerun for this source-only cleanup; the machine
  remains stopped. Local ARM measurements below compare the accepted frozen
  executable against the cleanup on identical fixtures/compiler/flags. They
  check for local regressions and do not replace the earlier full-corpus x86
  evidence or establish a new engine-parity claim.

Local ARM cleanup check (microseconds, geometric mean of per-query medians;
962 official queries, 7 interleaved passes, 10-second per-engine warmup for
each command, unchanged 100k-document fixtures). Rust 1.98.1, native CPU
code generation, release LTO; no builds ran during measurement. These are
process round-trip timings, including the benchmark protocol.

| Command               | RGB accepted | RGB cleanup | Change | Off accepted | Off cleanup | Change |
| --------------------- | -----------: | ----------: | -----: | -----------: | ----------: | -----: |
| Top 10                |       36.967 |      36.545 | -1.14% |       43.238 |      43.229 | -0.02% |
| Top 1000              |       51.162 |      50.974 | -0.37% |       55.439 |      55.393 | -0.08% |
| Top 100 + exact count |       44.893 |      44.756 | -0.31% |       40.639 |      40.466 | -0.42% |
| Exact count           |       25.331 |      25.257 | -0.29% |       25.924 |      26.018 | +0.36% |

Lifetime peak RSS (accepted/cleanup): rgb: 49.109/49.125 MiB; off: 44.922/44.891 MiB.
This is process residency, including mapped pages, not a heap measurement.
The observed timing differences are small; no new speedup claim is made.
All 20 fixture files remain byte-identical after both correctness and timing
checks. Raw samples, executable hashes, manifests, and commands are retained
under `.context/scoring-cleanup/`. No commits were made.

## Trusted text query reads and SIMD audit (2026-09-17)

Normal segment queries now trust Summa writers for posting directory, decoded
document order/range, pruning bounds and position-stream invariants. Full metadata
scans and decoded-document content checks are removed from that path, together
with the proof cache, lock, budget, CLI option and cache statistics. Explicit
deserialization and merge admission retain their checks. Format envelopes, I/O
failures and extents needed by unsafe decoders remain checked. An unsupported
block count is rejected before allocation or fixed-size kernel entry.

The position reader shares one view constructor; its obsolete cached proof type
is removed. A regression pins strict rejection of an overflowing checkpoint
before deriving its layout. Byte encodings and writer defaults are unchanged.
Footer flags remain necessary layout descriptors: their unknown-bit test is one
constant mask (`0xff`), not eight runtime validations. See
[posting codecs](posting-codecs.md#footer-flags-describe-stored-layout).

### Paired measurement

Same fixtures, Rust 1.98.1, release LTO, `target-cpu=native`, and machine per pair.
The ARM fixture has 100,000 documents; the x86 Xeon fixture has 5,032,104. Both
use 962 official queries, seven interleaved passes, ten seconds of warmup per
engine/command, the same dictionary budgets, and unchanged RGB-on/off indexes.
Reported latency is the geometric mean of per-query median wall time. The old
comparison uses its tuned 256 KiB validation cache; negative deltas are faster.

| Host / layout | Top 10 | Top 1000 | Top 100 + count |  Count |
| ------------- | -----: | -------: | --------------: | -----: |
| ARM / rgb     | -1.54% |   -1.44% |          -2.74% | -1.43% |
| ARM / off     | -1.48% |   -1.93% |          -1.85% | -1.90% |
| x86 / rgb     | -3.12% |   -3.00% |          -2.55% | -3.58% |
| x86 / off     | -3.54% |   -2.24% |          -4.63% | -3.53% |

Against the old zero-byte validation-cache setting (with the same dictionary
budgets), x86 RGB latency falls 21.8–26.7%. The retained RGB top-10 time is
433.79 µs versus 447.78 µs with the tuned cache, or 591.46 µs without it.
These are warm Summa comparisons; they do not establish cold-I/O, concurrent
ingestion, tail-latency or cross-engine parity. No BMP speedup is claimed.

The final owner-only cleanup was recompiled and compared against the selected
query implementation for top-10/count on both layouts, five interleaved passes:
ARM changes range from −1.18% to +0.53%; x86 from −1.37% to +0.10%.

### Memory

Peak process RSS (MiB), old tuned cache → trusted query implementation:

| Host |           RGB on |         RGB off |
| ---- | ---------------: | --------------: |
| ARM  |    48.94 → 45.53 |   44.70 → 44.77 |
| x86  | 1013.95 → 794.21 | 751.40 → 749.67 |

The deleted proof table saves its configured 256 KiB per segment. RSS additionally
includes file-backed pages touched by queries and scans; it is not a heap-size
measurement. The RGB RSS reduction is much larger than the removed allocation.
A separate one-pass replay with `/proc/PID/smaps` attributes it to position-file
pages: RGB `.pos` RSS falls from 635.02 to 417.77 MiB, while anonymous RSS falls
only from 5.52 to 5.28 MiB. Posting-file RSS is nearly unchanged (265.53 to
265.03 MiB). The removed position validation scan was touching pages that query
execution did not need. On RGB-off, `.pos` RSS is unchanged at 417.52 MiB. Inspected final-stream
footers are POS3 for RGB and POS4 for RGB-off: the former interleaves block
headers and payload, while the latter keeps a separate directory.

### SIMD selection and correctness

An AVX-512F lower-bound candidate used unsigned 16-lane comparisons and masked
tail loads, retaining AVX2 below 32 values. It passed exhaustive unsigned/tail/
alignment comparisons and a protected-page test on x86. Its isolated kernel was
11–25% faster for 32–256 values, but complete RGB top-10/top-1000 queries regressed
4.3–4.9% against trusted AVX2. The candidate was rejected. Existing AVX2/SSE2,
NEON and scalar text paths remain, along with supported AVX-512F dense-vector
and AVX-512 VPOPCNTDQ binary-vector paths. No experimental SIMD code is retained.

The full search harness passed: 1,854 native tests, four real-server broker tests,
strict Clippy, API docs, native without sync and portable compilation (26 existing
tests ignored in the ordinary native run). Final cleanup additionally passed
220 posting/position tests, strict Clippy, a fresh WASM release build and all
32 JavaScript tests. Final ARM and x86 executables each match 1,676 reference
queries on each layout: ranked IDs and score bits at k=10/100/1000, complete
top-100 and exact counts. Healthy serialization is byte-identical for all codecs.

Evidence, executable/source hashes, raw samples and rejected prototypes remain
local in `.context/validation-simd/`; full-harness evidence is in
`.context/search-harness/20260917T120701.487250Z-full/`. Benchmark programs and
raw results are excluded from the PR.

## Binary IVF comparison and scan optimization (2026-09-17)

### Scope and reference engines

The [Faiss binary benchmark](https://github.com/facebookresearch/faiss/wiki/Binary-hashing-index-benchmark)
uses 50 million 256-bit image descriptors for **range search**. It is relevant
algorithmically, but its timings are not comparable to Summa top-k search.
[Faiss BinaryIVF](https://github.com/facebookresearch/faiss/wiki/Binary-indexes)
is the measured reference here. Its
[merge implementation](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexBinaryIVF.cpp)
copies inverted-list contents and rebases IDs; the caller must supply compatible
centroids. Summa also checks the persisted quantizer generation.
[Milvus BIN_IVF_FLAT](https://milvus.io/docs/bin-ivf-flat.md) exposes a related
index, but this experiment does not measure Milvus service/storage overhead.

The fixture derives 256-bit and 2,560-bit codes from public SIFT1M vectors by
centering and seeded Gaussian sign projection: one million rows, 256 held-out
queries, 256 clusters, probes 1/4/16/64, and top-k 10/100. This is a derived
binary task, not the published SIFT Euclidean leaderboard. The main comparison
imports identical Faiss-trained centroids, list membership, codes, and probe
order into Summa's existing writer. It isolates the scanner and routing;
it does not establish independently trained model quality or build-speed parity.

Each engine uses one search thread, a warm-up pass, and five timed batch passes
per process. The main comparison has three interleaved process rounds. Rust
1.98.1 builds use release LTO and native CPU targeting; Faiss 1.15.1 uses official
CPU wheels and their shipped compiler/dispatch. No server, response hydration,
network, cold-I/O, concurrent ingest, or tail-latency claim follows from these
warm native timings. ARM is an Apple M4 shared desktop with matched thread QoS;
x86 is a dedicated eight-vCPU Intel Xeon machine using AVX2. The x86 CPU lacks
AVX-512 VPOPCNTDQ, so that Hamming path was not performance-tested.

### Retained changes

1. Binary leaf scoring reads the existing collector's conservative threshold
   once per 64 scores. Scores strictly below it skip document IDs, ordinals,
   visibility checks, and collector insertion. Every selected code is still
   scored. Equal-score candidates retain existing tie-breaking; Sum/Avg/
   LogSumExp/WeightedTopK retain every ordinal. The shared sink now owns the
   threshold method previously confined to ScaNN AH, avoiding a second trait.
2. The existing Hamming kernel receives a literal width for 256-bit code batches,
   allowing LLVM to unroll the same implementation. The four-row kernels are
   inline-eligible; CPU feature guards, bounds, row tails, and scalar fallback
   remain intact. No separate scoring implementation, cache, or format is added.

Representative preassigned-leaf top-10 times at 16 probes (microseconds/query):

| Host / bits |   Before | Threshold only | Threshold + width specialization |    Faiss | Tie-aware recall |
| ----------- | -------: | -------------: | -------------------------------: | -------: | ---------------: |
| ARM / 256   |   182.39 |          92.82 |                            80.11 |    41.32 |           97.62% |
| x86 / 256   |   523.28 |         334.57 |                           272.29 |   193.34 |           97.62% |
| x86 / 2,560 | 2,488.73 |       2,330.52 |                         2,309.28 | 2,741.84 |           98.52% |

The x86 specialization column is a separate two-round interleaved check against
threshold-only: 332.30 → 272.29 µs at 256 bits and 2,323.48 → 2,309.28 µs at
2,560 bits. Across probe budgets, specialization improves x86 256-bit top-10 by
16.8–18.4%, versus 13.0–16.6% on ARM. The x86 wide-code effects are approximately
−0.6% to +2.0%; ARM wide-code samples drift substantially on the shared desktop
and are inconclusive. Do not infer a wide-code ARM gain from them.

Summa remains slower than Faiss for 256-bit codes: approximately 1.41× on x86
and 1.94× on ARM at the tabulated point. Summa is approximately 16% faster for
2,560-bit x86 scanning on this fixture. Neither is a general engine-parity claim.

All main baseline/threshold results have identical document IDs, ordinals, and
score bits at every budget. Specialized scans match their controls too. With
identical preassigned leaves, Faiss's sorted Hamming distances match exactly;
recall is tie-aware against exhaustive BinaryFlat ground truth. At 64 probes,
top-10 recall is 100% for both widths. A separate quality-only check trained
Summa on the same 65,536 sampled rows for ten iterations: top-10 recall at 16
probes is 97.54% / 98.98% for 256 / 2,560 bits, versus the shared Faiss model's
97.62% / 98.52%. Those separately trained runs have different list populations
and are not substituted into the scanner timing comparison.

### Memory, storage, and merge

For one million single-value vectors, 256 clusters:

| Bits  | Summa ANN bytes | Summa flat bytes | Faiss serialized bytes |
| ----- | --------------: | ---------------: | ---------------------: |
| 256   |      38,012,368 |       38,000,016 |             40,010,355 |
| 2,560 |     326,012,368 |      326,000,016 |            328,084,083 |

Summa keeps two exact binary code copies, each with six bytes per vector for
IDs and ordinals. Together they use about 1.9–2.0× the standalone Faiss index's
disk space before outer container/shared model metadata. The measured Faiss index has no direct map (type 0), so its disk size does not
include an arbitrary-ID vector lookup structure. Flat storage currently supports
retrieval, training, rebuild, and multi-value completion; removing its writer
alone would break those operations. A single-value ANN query does not
read that second code payload for reranking.

Both Summa corpus payloads are evictable; disk size is not heap residency.
The isolated ANN reader's measured heap directory is 18,632 bytes at 256 runs,
plus existing bounded query scratch. A Linux warm-query snapshot reports Summa
process RSS of 39.36 / 313.94 MiB and anonymous RSS of 0.70 / 0.70 MiB at the two
widths. Faiss's Python process reports 80.88 / 356.04 MiB RSS and 58.40 / 333.54
MiB anonymous RSS. These include different language runtimes and exclude Summa's
flat representation/document fields; they illustrate mmap versus heap ownership,
not a fair whole-index memory ratio. The retained changes add no query or index
allocation and change no persisted bytes.

Two distinct one-million-row sources were merged. Summa's copied payload
columns remain byte-identical, with rebased IDs and 512 post-merge query checks
per width. Faiss's codes and rebased IDs match its sources too. Summa streams
the persisted output, while Faiss first merges heap lists and then serializes;
these are different timing/durability contracts, so no merge speedup is claimed.
The first wide merge ran out of machine disk space; after removing build-cache
artifacts, the retry and byte/query checks passed.

### Remaining storage work and validation

The selected [exact binary storage design](binary-vector-storage.md) preserves
retrieval, multi-value scoring, and rebuild/ALTER by retaining cluster-ordered
exact codes and adding a document lookup. At the end of the scanner experiment,
the indirect format was not implemented; the following section records its
subsequent implementation as the default, without a flag. Lossless XOR-residual compression saved 25–27% of code
bytes on this fixture, but compressing one of two copies only saves roughly
12–13% overall before metadata; one-copy storage offers greater potential savings.

The final implementation passed `python3 scripts/check_search.py check`: formatting,
strict Clippy, 1,857 native tests (26 existing tests ignored), and native compilation
without sync. A fresh WASM release build and all 32 JavaScript tests passed, as
did the documentation/link checker. The regression covers ties, deletions,
batch/run boundaries, serial/parallel collectors, and complete ordinal output.
The shared Hamming test explicitly exercises scalar and runtime-selected kernels
across widths and tail counts. Lifecycle/RPC code and persisted formats are
unchanged; the full lifecycle/RPC harness was not rerun for these scanner changes.
The benchmark machine was confirmed `TERMINATED` after results were collected.

All benchmark adapters, fixtures, raw samples, hashes, failed exploratory runs,
and experiment scripts stay ignored under `.context/binary-ivf/`; they are not
part of the production diff. Final harness logs are in
`.context/search-harness/20260917T133803.778018Z-check/`.

## 2026-09-17 — Single-copy binary ANN storage by default

Implemented the [exact binary storage format](binary-vector-storage.md) for
Binary IVF and binary ScaNN. New trained writes retain their exact codes in ANN
and replace the flat duplicate with TOC type 11: a versioned document/ordinal
lookup into copyable code spans. There is no flag. Flat-only and untrained
fields retain their sole flat representation; float formats are unchanged.
SOAR still has its intentional secondary ANN assignments, while exact retrieval
exposes each logical value once.

### Ownership and merge review

- The existing ANN writers report their actual code positions. A bounded
  external metadata sorter creates lookup blocks during build/rebuild and
  deletion compaction. Sort files are anonymous, owned before writing, and
  removed on drop, failure, cancellation, or unwind. Fan-in is capped at 16;
  the in-memory run holds at most 65,536 sixteen-byte records. Compaction
  splits its existing scratch budget between ANN output and lookup creation.
- Normal binary merge copies both ANN payloads and document lookup rows
  byte-for-byte. It relocates only ANN run directories, span byte bases, and
  block document bases. No vector reconstruction, assignment, retraining, or
  per-vector address rewriting occurs in this path. Retaining source extents
  means fragmentation can grow; this replaces automatic binary run coalescing.
  Float AH retains its existing packing policy.
- Legacy flat-plus-ANN sources have no lookup blocks. Their first merge logs
  the upgrade and sorts lookup metadata from existing ANN labels and spans;
  ANN code bytes remain unchanged. Subsequent merges copy the lookup. Existing
  BMP reorder clones the vectors file through the existing reorder owner.
- The exact-vector reader shares the ANN reader's immutable byte owner.
  This matters for HTTP/WASM as well as mmap: no duplicate corpus allocation
  or second range-fetch path is introduced. Contiguous reads retain shared
  byte views; scattered batches gather directly from the same owner.
- Existing readers, scorers, training, generation publication, and row deletion
  own their existing responsibilities. ALTER's deferred-flat path materializes
  exact codes through the common reader. Direct ALTER to a `flat` algorithm
  remains unsupported by the existing schema contract. Raw flat merge keeps
  its bounded byte windows even for vectors larger than one window.
- Diagnostics report `exact_storage` and `exact_lookup_bytes`; `flat_bytes`
  becomes zero for ANN-backed exact storage. Old readers reject the unknown
  type-11 entry. New readers retain compatibility with old flat-plus-ANN files.

### Measured storage and memory model

Same one-million-row projected-SIFT fixtures, 256 clusters, Apple M4, Rust 1.98.1,
release with LTO and `-C target-cpu=native`. The source ANN payload is byte-identical
to the earlier fixture; every exact vector was also checked against the original
base codes. Serialized field regions, excluding the common 58-byte outer TOC
and footer and the unchanged shared model:

| Bits  | Previous ANN + flat |  ANN + lookup | Reduction |
| ----- | ------------------: | ------------: | --------: |
| 256   |        76,012,384 B |  52,015,488 B |    31.57% |
| 2,560 |       652,012,384 B | 340,015,488 B |    47.85% |

The lookup is 14,003,120 bytes at either width: 14 bytes per logical vector,
256 twelve-byte spans, one sixteen-byte block, and a 32-byte header. The new
reader's owned block directory adds 40 heap bytes for this one-block fixture
(240 versus 200 bytes for the exact-vector view object plus owned directory).
Lookup rows/spans and ANN codes remain file-backed under mmap. The wider lookup
adds eight bytes per logical value to document lookup metadata, so a fully
pinned lookup budget can grow even though total disk storage falls. These sizes
are not whole-process RSS measurements. At dimensions below 64 bits, the map can
be larger than the removed duplicate codes and labels; the requested single-copy
default still applies.

### Correctness and verification

The new behavior test first failed because trained binary fields still emitted
flat payloads, then passed with type 11 present and the duplicate type 4 absent.
Tests cover repeated three-generation byte-copy merges for binary IVF and binary
ScaNN, missing documents, multi-value ordinals, SOAR deduplication, legacy lookup
construction, malformed addresses/versions, all five combiners, retraining,
ALTER through deferred flat and back, deletion compaction, and readers retained
across generation replacement. Native async and sync paths are exercised. The
ScaNN writer/compactor test also preserves a surviving nonzero ordinal while
removing a document with a secondary assignment.

The spill sorter matches in-memory output byte-for-byte with forced tiny runs.
Write failure and cancellation propagate as errors. Publication and cleanup
continue through the existing generation/lifecycle owners and their regression
suite. WASM's release build and 32 existing JavaScript tests pass; an additional
native-built single-copy fixture opens in WASM and returns exact multi-value
codes and missing-value results.

Benchmark adapters, source snapshots, fixtures, and logs remain ignored under
`.context/single-copy/`. No benchmark source was added to the production change.

Warm exact-access microbenchmarks on that shared desktop, median of five
iterations, same optimized executable for both layouts:

| Bits  | Flat / shared point read (ns/vector) | Flat / shared full scan (ms) |
| ----- | -----------------------------------: | ---------------------------: |
| 256   |                         6.66 / 43.00 |                0.526 / 6.624 |
| 2,560 |                        66.03 / 98.63 |               5.363 / 21.997 |

Point reads sample 10,000 deterministic rows. Full scans use 1,024-row batches
and checksum all bytes in both paths. These are warm mmap access costs, not
ANN-query latencies. Gathering makes document-order scans slower than the removed
contiguous copy. The initial per-row range-read implementation took 14.17 / 34.75
ms for the two full scans; sharing the ANN byte owner and gathering directly
reduced those samples to 6.62 / 22.00 ms. The runs were separate checks on a shared
desktop, so their ratio is directional evidence, not a controlled service-level
speedup. Normal ANN scan bytes and kernels are unchanged.

One observed two-million-row copy merge wrote ANN plus lookup in 6.75 / 44.66 ms
at the two widths, with outputs retained in RAM. Those single observations do
not measure durable filesystem throughput and are not compared with the previous
coalescing policy. The correctness tests separately pin payload byte identity
through repeated merges. Cold-query latency, whole-process RSS/peak scratch,
concurrent ingest/merge latency, and x86 storage measurements were not rerun for
this format change; no improvement in those quantities is claimed.

Full harness evidence: `.context/search-harness/20260917T142950.056452Z-full/`
passes 1,864 tests (26 normally ignored), then four real-server broker tests,
strict Clippy, native-without-sync and portable compilation, and API docs.
The final bounded-copy/budget cleanup also passed
`.context/search-harness/20260917T143624.567115Z-check/`. The initial broad run
caught a stale error-message assertion and another run hit an existing broker
10-second discovery timeout; the complete rerun passed with `RUST_TEST_THREADS=2`.
WASM evidence is `.context/single-copy/wasm-final.log`; all benchmark data and
source adapters remain under the ignored context directory.

A final failure-path regression reproduced acceptance of a short lookup read
when the returned prefix happened to form a valid smaller map. Opening now checks
the returned metadata length against the declared slice before parsing. The
focused regression (`exact_vector_lookup_rejects_short_metadata_reads`) passes,
as does fresh strict Clippy; this admission-only fix was checked separately after
the full harness. Final WASM verification includes the same guard. Evidence is
in `short-read-before.log`, `short-read-after.log`, and `clippy-final.log` under
`.context/single-copy/`.

## 2026-09-17 — Binary storage compatibility removal and standalone reorder

The earlier single-copy implementation retained an upgrade branch for duplicate
flat-plus-binary-ANN segments. Removed that branch and the normal ANN merge
writer's optional lookup-construction callback. Search and filtered training
admission now reject that duplicate layout explicitly; flat-only/untrained binary
fields still work. Old duplicate-layout indexes must be recreated.

Standalone `IndexWriter::reorder` now uses the existing encoded-run compactor
for fragmented binary IVF/ScaNN fields and the existing bounded exact-location
sorter. Codes and ordinals remain exact, document labels are rebased, and global
ANN artifacts are unchanged. The operation shares the existing output claim,
cancellation, cold writer, memory budget, and generation publication. Its budget
reserves compactor directories/span metadata and 256 KiB label scratch before
admitting the sorter. Text plan scratch is released before vector coalescing.

This is separate from ordinary merge and from configured text/BMP merge-time BP.
Normal binary merge copies ANN payloads and lookup rows unchanged and only rebases
run/span/block directories. It cannot allocate a lookup sorter. A copied cluster
can consequently span multiple source extents; standalone reorder joins those
extents. Already contiguous vector files are linked/copied unchanged. Explicit
reorder works on binary-only indexes; the server's automatic BP scheduler still
selects indexes through its existing text/BMP reorder-field policy.

The behavior regression failed before the change with fragmentation
`1.875 -> 1.875`; after wiring the existing compactor it reaches `1.0`. Coverage
also checks binary-only and mixed BMP indexes, exact codes/labels, old readers,
byte-identical vector files after a second reorder, repeated merges, SOAR
secondary assignments, and cancellation. Evidence lives in the ignored
`.context/single-copy/` directory; no benchmark harness is added to production.

Reader admission now resolves lookup spans to ANN label columns once, then
validates rows sequentially through the shared document-map validator. ANN
label checks follow physical payload order. Native metadata prefetch is bounded:
up to two 64K-row lookup windows and an 8 MiB page-rounded ordinal budget. Exact
code payloads remain untouched during this validation. This removes repeated
per-row lookup-block/run searches without dropping corruption checks.

The durable vector-only comparison uses two copies of the same one-million-row
projected-SIFT input, 256 clusters, production cold writers and fsync. The prior
layout is reconstructed with the existing ANN compactor and duplicate-flat copy
loop; the new layout uses the normal ANN/lookup copy writers. Both run in the
same release executable on the Apple M4 with native CPU flags. Six alternating
rounds discard the first; the following are medians of the remaining five from
`merge-qos-{256,2560}.json` under `.context/single-copy/`:

| Code width | Prior write + open | Single-copy write + open | Prior process CPU | Single-copy process CPU |
| ---------- | -----------------: | -----------------------: | ----------------: | ----------------------: |
| 256 bits   |          319.18 ms |                436.30 ms |          86.34 ms |                76.98 ms |
| 2560 bits  |         1623.96 ms |                774.19 ms |         444.49 ms |               235.33 ms |

These measurements overlapped validation builds/tests and are diagnostic, not
quiet-machine service benchmarks. Earlier repetitions also varied substantially.
Writing is cheaper in both cases, but opening the copied metadata remains more
expensive; the small-code case still regresses in elapsed time. Thus the requested
no-slower end-to-end merge condition is **not established**. No retraining,
re-encoding, or corpus-sized sort occurs in normal merge. This experiment omits
other segment files and atomic publication, and does not measure whole-process
peak RSS or x86 performance. The existing single-copy storage-size measurements
above remain applicable; bounded scratch accounting is not a measured RSS result.

The lifecycle implementation passed the full harness at
`.context/search-harness/20260917T155806.521805Z-full/`: 1,866 native tests,
26 ignored, four real-server broker tests, strict Clippy, portable/native
compilation and API documentation. The final `check` run at
`.context/search-harness/20260917T161756.487246Z-check/` also passes all four
stages against the subsequent reader refinements. Final WASM release build and all 32 tests pass, including a
native-built binary ANN fixture with exact multi-value and missing-value reads
(`.context/single-copy/reorder-wasm-final.log`).

## Experimental Seismic sparse retrieval (2026-09-17)

An isolated native prototype under `.context/seismic-experiment/` reuses upstream
Seismic clustering/summaries and Summa BMP forward storage, query preparation,
candidate scoring and collection. No production dependency, format, dispatch or
schema option was added. It supports one nonnegative vector per document only.
The source snapshot, patches, source/data hashes, commands and full results are
preserved there in `README.md`, `reproduction.json`, and `RESULTS.md`; benchmark
artifacts remain ignored and are not intended for commit.

On Apple M4, release LTO/native flags and the same nightly compiler, 1,000
eligible public NeurIPS SPLADE sparse-1M queries produced these fastest tested
results at >=99% recall against common quantized exhaustive truth. Times are
warm mean milliseconds, with one warm-up and two measured passes. Search budgets
were swept for BMP and Seismic; these are not held-out parameter selections.

| Engine                               | Top-10 ms / recall | Top-100 ms / recall |
| ------------------------------------ | -----------------: | ------------------: |
| BMP                                  |    7.596 / 99.860% |    11.651 / 99.101% |
| BMP + existing BP reorder            |    2.529 / 99.450% |     6.190 / 99.523% |
| Reference Seismic, fresh             |    0.297 / 99.320% |     1.084 / 99.218% |
| Summa Seismic prototype, fresh       |    0.814 / 99.320% |     3.163 / 99.218% |
| Summa prototype, 4 copied fragments  |    2.464 / 99.470% |  target not reached |
| Summa prototype, 16 copied fragments |    1.954 / 99.440% |     5.705 / 99.402% |

Fresh high-recall Seismic uses up to 4,096 postings/term; copy-fragment tests use
512 per term in each independent source. With 512, a fresh 1M index tops out near
93% top-10 recall in the tested sweep, so its fast low-recall timings are excluded
from the table. The 100K fixture, p95/p99, configurations and candidate counts are
also recorded in the ignored report.

The fresh 4,096-posting prototype stores 2.553 GB (1.916 GB nominations plus
0.637 GB forward values), versus BMP's 1.591 GB or BP BMP's 1.401 GB. At fixed
512-posting build settings, total prototype bytes grow from 1.274 GB fresh to
2.432 GB with four fragments and 4.700 GB with sixteen. The corresponding
vector-only copy+fsync times were 1.55, 2.43 and 6.61 seconds. These exclude reader
opening, other files and atomic publication; they do not resolve the separate
binary ANN end-to-end merge regression above. Build/RSS measurements are in the
report; summary objects still reside on heap and require an evictable format
before production integration.

Merge streams unchanged nomination payloads with a 64 KiB buffer and remaps run
headers; forward values use the existing copy writer. Repeated-copy bytes and
all copied forward vectors are checked. Missing IDs, duplicate nominations and
ties pass exact exhaustive-score comparisons on synthetic 1/4-fragment indexes;
cancellation, short writes and overflowing row offsets fail as expected. Normal
merge performs no sparse clustering, summary recomputation or model training.
It retains locally pruned candidates, so it is not equivalent to rebuilding one
globally pruned index. Bounded optimization outside normal merge is still needed
to control retained-list and summary growth.

Caveats: first 1,000 of 5,655 eligible <=64-term queries out of 6,980, no truncation;
quantization alone yields 99.37% (100K) and 99.03% (1M) original-float top-10 overlap.
Approximation can lose further original-float neighbors. No cold/RPC/concurrency,
filters, multi-value, deletion/purge, mixed quantization scales or prototype WASM
support is established. Reference Seismic uses f32 storage for the shared integer
values and no optional kNN graph; this is not a claim about its fastest encoding.
Future work should profile the shared forward candidate path, budget fragmentation,
and implement production lifecycle integration before considering any default.

Validation: the final native prototype checks pass. `python3 scripts/check_search.py
check` passes all four stages (format, strict Clippy, native tests and portable
native compilation), with evidence in
`.context/search-harness/20260917T174456.373217Z-check/`. Documentation checks pass
for 119 Markdown files, 568 local links and 20 benchmark targets. Prototype WASM
was not run; the new experimental dependency requires native nightly compilation.

### Seismic scorer and memory follow-up (2026-09-17)

The isolated prototype now reuses the existing BMP query-weight lookup experiment
and CPU prefetch helper. A reusable 120,436-byte query table replaces repeated
sparse-query walks; prefetch is bounded to 32 candidates and 512 bytes per vector.
At the user's request, the final experimental scorer skips per-candidate payload
corruption scans. Safe slice/table bounds and score-overflow checks remain;
explicit validation remains available for diagnostics. No scan was moved to
reader admission. Production BMP dispatch and validation are unchanged.

Three alternating independent process runs per engine on the same M4/compiler,
fixture and release flags give these median warm mean latencies:

| Top-k | Original Summa ms | Final Summa ms | Reference Seismic ms |  Recall |
| ----: | ----------------: | -------------: | -------------------: | ------: |
|    10 |             0.828 |          0.306 |                0.318 | 99.320% |
|   100 |             3.162 |          1.098 |                1.107 | 99.218% |

The improvement is 2.7–2.9x; this supports parity on this fixture, not a general
cross-platform lead. All 110,000 returned IDs, scores and positions are
byte-identical between Summa scorers in every paired run. Candidate counts and
recall against shared quantized truth are unchanged. A separate single-run
16-fragment top-100 comparison improves 5.550 to 2.635 ms with byte-identical
results and 99.402% recall. Formats, nomination budgets and merge logic did not
change. Reproduction metadata, source patches and full measurements are in
`.context/seismic-experiment/optimization/RESULTS.md` and `reproduction.json`.

Query-process peak RSS remains about 2.61 GB. Owned nomination structures account
for approximately 1.57 GB of summaries, 171 MB of row IDs and 31 MB of offsets;
the forward file maps another 637 MB. Reused visited/query scratch is only about
4.12 MB. Thus the peak chiefly reflects resident index state, not per-query
scratch. Across the 1,000 queries, the selected-term union covers only 523 MB at
cut=10 or 669 MB at cut=20, while the current loader owns all nomination objects.
A term-addressable file-backed representation or byte-budgeted cache remains
necessary before production integration; cold-I/O costs are not established.

The high-recall build took 103.4 seconds and peaked at 8.92 GB RSS, including BMP
preparation and export, so it is not an isolated constructor comparison. Its
2.553 GB disk footprint and the copy-merge measurements above are unchanged.
At fixed 512-posting settings, sixteen fragments occupy 3.69x the fresh bytes.
Copy-only merge preserves local pruning and clusters; an explicit bounded
optimization outside normal merge is required to control that growth. Merely
reordering forward vectors cannot remove duplicated retained lists/summaries.

Final native kernel and copy checks cover duplicate contributions, refreshed and
empty queries, score overflow, explicit corrupt-payload validation, missing IDs,
ties, repeated-copy bytes, cancellation and short writes. All four production
harness stages pass at `.context/search-harness/20260917T180804.078981Z-check/`;
that harness does not exercise the isolated CLI, which has separate checks.
Prototype WASM and x86 remain untested. No production Seismic integration or
benchmark-artifact commit is included.

### Seismic maintenance feasibility (proposal, 2026-09-17)

The existing optimizer/reorder lifecycle can host bounded term-level Seismic
maintenance while ordinary merges continue copying encoded fragments. Repacking
alone cannot remove retained nominations and cluster summaries; reducing that
debt requires re-pruning and rebuilding selected terms with the shared forward
vectors and existing Seismic builder. The prototype needs term-addressable
payloads and a bounded term builder before this can be an incremental pass;
its current whole-fragment bincode loader is unsuitable. No maintenance code or
new production defaults were added in this investigation.

For fixed top-L pruning, local retained lists suffice to compute global top-L
under matching capacity, weights and stable tie order. This does not establish
identical approximate results after reclustering. Increased capacity, deletions
and changed weights can require recovery from complete forward data. Maintenance
must preserve a measured recall target: the fixed-512 fresh index's approximately
93% top-10 recall makes its smaller bytes an invalid equal-recall compaction
claim. Budget scratch, clustering, output I/O and reader overlap; keep separate
Seismic debt and completion cooldown. Reuse source/output claims and atomic
replacement instead of creating a second publication protocol.

Code tracing also found that binary ANN coalescing runs inside standalone
reorder, but periodic fresh/deepening selection in the server optimizer is gated
on `has_reorder_fields()`. Seismic integration needs independent debt eligibility;
ANN-only scheduling should be reviewed alongside it. Detailed proposal and
acceptance criteria are in the ignored experiment's `DESIGN.md`. These are
unimplemented findings, not measured maintenance savings.

### Seismic term maintenance experiment (2026-09-17)

The isolated CLI now implements term-addressable nomination packs, explicit
bounded term maintenance and copy-only pack merging. It reuses upstream
Seismic's existing single-list clustering/summary builder, Summa BMP forward
values and cold output writers. Ordinary merge and untouched-term maintenance
share sequential range copying with row-directory remapping; adjacent payload
extents are coalesced without decoding. Directory parsing buffers reads; term
payload reads remain bounded to their own extents. No production dispatch,
scheduler, schema or public format was added.

On the same 1M M4 fixture, consolidating sixteen 512-posting source fragments
toward 4096 retained postings/term reduced nominations from 61,779,257 to
42,657,454. Nomination bytes fell from 4.076 to 1.917 GB; including unchanged
forward storage, total bytes fell from 4.713 to 2.554 GB (45.8%). It rebuilt
27,586 terms, with no remaining eligible terms or memory-admission skips.
Full maintenance took 209.05 seconds on one worker and peaked at 0.837 GB RSS,
including mapped forward pages. This full pass preceded the copy/parser
refinements; the term selection and clustering algorithms are unchanged.

Three alternating process runs per layout, one warm-up and two timed passes,
give these medians of warm mean latencies. The maintained layout uses larger
query budgets to meet or exceed the original layout's recall:

| Top-k | Before ms / recall | Maintained ms / recall | Query peak RSS before / after GB |
| ----: | -----------------: | ---------------------: | -------------------------------: |
|    10 |    1.107 / 99.440% |        0.422 / 99.610% |                    3.994 / 2.777 |
|   100 |    2.601 / 99.402% |        1.275 / 99.504% |                    4.550 / 2.778 |

Recall is against shared quantized exhaustive truth. Source-capacity shortages
and upstream tie policy prevent a fresh-global-top-L equivalence claim. Exact
candidate scores are unchanged; approximate candidate sets/results can change.
The query adapter still heap-loads summaries, so this does not implement
file-backed query summary access.

A 64-term pass required 1.17 seconds of term work. Sequential range copying and
buffered directory parsing reduced its wall time from 16.42 to 6.32 seconds,
with byte-identical payloads. Whole-pack rewriting remains significant: small
CPU budgets do not imply small output I/O. Copying the maintained 1.917 GB
nomination pack alone took 2.69 seconds including fsync; it copied no forward
file and excludes production segment publication. Another maintenance pass on
the converged layout rebuilt zero terms and retained byte-identical pack bytes.

Limits include terms/pass, a soft time window between non-preemptible term
builder calls, a conservative scratch admission estimate and a separate hard
output-byte cap. Skipped work/remaining debt are reported; scratch estimates are
not allocator-enforced RSS caps. Production must still integrate the shared
optimizer permits, cooldown, independent debt eligibility and SegmentManager
publication, and address repeated whole-file I/O. Deleted-candidate backfill,
filters, multi-value, prototype WASM and x86 remain unimplemented/untested.

Native checks cover exact small-fixture results, missing IDs, duplicate term
nominations, ties, partial progress, continuation, idempotence, untouched payload
bytes, row rebasing, scale incompatibility, overflow, short writes, cancellation,
panic cleanup and output ownership. Format conversion preserves all 110,000
result IDs/scores/positions on the 1M query set. Full ignored evidence and
commands are in `.context/seismic-experiment/maintenance/RESULTS.md` and
`README.md`. Benchmark artifacts remain uncommitted.

Final validation: `python3 scripts/check_search.py check` passes all four stages
at `.context/search-harness/20260917T185414.002295Z-check/`; native prototype
checks pass separately. Documentation links, snapshot formatting and whitespace
checks pass. No prototype WASM or x86 run is claimed.

## Seismic production integration — September 17–18

Seismic now owns all sparse build/read/search paths. Removed sparse BMP and sparse
MaxScore formats, writers, scorers, tuning flags and reorder kernels have no
compatibility dispatch. Text MaxScore/BP and dense ANN retain their owning code.
The shared weight codec preserves Float32, Float16, UInt8 and UInt4 forward
values, with signed scoring, empty/missing values and logical ordinals.

Normal merges copy encoded runs through the existing directory range-copy
operation and remap small metadata. They never cluster. Bounded maintenance and
explicit deletion compaction reuse SegmentManager ownership, immutable
publication, cancellation and deferred retirement. A new regression exposed and
fixed a deletion-generation lifetime race: maintenance now pins its exact source
deletion mask while concurrent visibility changes publish independently.

The broad tests also caught a distinction between complete ordinal aggregation
and matched positions. Zero contributions remain part of document scoring, while
fusion receives only genuinely matching ordinals, including cancelling matches.
Infinite cluster-summary proxies no longer reject valid finite document scores;
actual score overflow still returns an error. Sparse `reorder` is rejected as a
meaningless option; indexed plain/chunked text retains BP, while Seismic and binary
ANN maintenance follows actual persisted debt.

A final multi-value regression found native/WASM vector queries using temperature
0.7 while sparse-term queries and RPC used 1.5. Constructors and the RPC unset
case now reuse `MultiValueCombiner::default()` (1.5); explicit temperatures are
preserved. This intentionally changes omitted-combiner native/WASM multi-value
scores to match the shared default. Single-value benchmark scores are unchanged.

### Remaining production costs

- Maintenance currently rewrites the entire sparse file, even for a small term
  batch. It rebuilds at most 64 terms per pass, with shared scratch/time/cancel
  admission. The optimizer's finite follow-up limit can leave visible debt; it
  does not promise immediate convergence of a large copied vocabulary.
- Selection favors fragment count, then dimension ID. Emission order also sets
  rebuild order. This does not model query frequency or saved payload bytes.
- Format open visits forward directories and cluster headers. Payloads remain
  evictable mappings, but this traversal can fault pages containing cluster
  payloads. Process RSS therefore includes mapped residency as well as heap.
- Required sparse scoring clauses use two complete forward passes to accommodate
  the existing infallible scorer-advance interface. Selective bitmap filters use
  exact scoring of eligible IDs; predicate/deletion underfill can trigger a
  forward scan. These paths preserve membership but can cost more than nomination.
- Geometric build retains decoded rows and an inverted candidate map while
  clustering. Compaction charges 384 bytes per coordinate plus per-vector scratch;
  selected-term maintenance charges 192. These admission estimates are not a
  guarantee of an identical operating-system RSS ceiling.

### Production benchmark method

The final matrix uses the original Float32 SPLADE CSR fixtures (100,000 and
1,000,000 documents), Apple M4, macOS, Rust nightly 1.100.0
(`215a8af4b`, September 15), release thin LTO, one codegen unit and
`target-cpu=native`. Search/Rayon/Tokio have four workers; ingestion has one
builder and two compression workers. Runs are sequential, without overlapping
compilation. Input pages are warmed before building. Query means exclude one
warm-up pass and include two timed passes over the same first 1,000 eligible
queries (at most 64 dimensions). These workloads are unfiltered and each
document has one vector. Timers surround Summa's public search call;
stored-ID hydration is outside the latency timer. Merge and maintenance timers
start after writer opening; their process peaks include opening. The 1M merge
input runs are hashed outside the timer, warming those input pages. Peak RSS includes
opening, mappings and hydration. These are warm-query measurements, not a
controlled cold-storage benchmark.

Reference Seismic is upstream `3c267137e202748e69ada8cd093f4c0b7c04479c`
with vectorium `39e016caed0ed030b56ab0532d4dcf0c71556b6e`, built with the same
compiler/flags and four threads. Its standalone kernel has no Summa document,
filter, multi-value or lifecycle layer. It uses U16 dimensions and an owned
serialized index; Summa uses U32 forward dimensions and mmap plus stored IDs.
Both use 4,096 postings, strongest 15 coordinates for assignment, target cluster
size 64, summary energy 0.4, and query cut/factor 10/0.85. Sampling and minimum
cluster policies differ. Reference builds posting lists in parallel with Rayon;
Summa currently clusters terms serially within the single indexing builder.
The build comparison therefore measures complete implementations with different
active clustering concurrency, not equal-core kernel throughput. Reference is
a useful cost/quality comparison, not an identical implementation or
merge-capable replacement.

Previous Summa MaxScore provides the unquantized exact-result oracle. Previous
BMP requires UInt8 quantization (fixed global maximum weight 5.0), so its recall
also includes quantization effects. Recall below is strict document-ID recall;
no relevance judgments or tie credit are applied. Removed engines exist only in
ignored benchmark snapshots, not production dispatch.

### Format-preserving optimizations: 100,000 vectors

| Operation                            |   Before |    After |              Change |
| ------------------------------------ | -------: | -------: | ------------------: |
| Build                                | 40.214 s | 28.137 s |     30.0% less time |
| Top-10 mean                          | 3.316 ms | 2.223 ms |     33.0% less time |
| Top-100 mean                         | 5.546 ms | 3.558 ms |     35.8% less time |
| Exhaustive top-100 mean, 100 queries |  43.4 ms |  19.7 ms | about 55% less time |
| Peak build RSS                       | 595.7 MB | 644.1 MB |   +48.4 MB observed |

Build caches each row's strongest 15 coordinates once and uses partial top-L
selection before sorting retained nominees. Search reuses a bounded weight
lookup across summary and exact row scoring. The 888,326,149-byte `.sparse` file
is byte-identical before/after (SHA-256
`0f9c86f3cc6f37ace4a0e34b8167d3cc239e39b14729887ed3a4cbaa17acf443`).
All 330,000 top-10/top-100 result IDs and scores across fresh, copied-merge and
maintained layouts are identical to their corresponding pre-optimization
results. The cache's fixed allocation is 12 MB for this corpus; allocator and
other working-set differences mean that is not the full observed RSS delta.

The fresh optimized index gets 99.580% top-10 and 99.484% top-100 recall.
Reference Seismic gets 99.770% / 99.744% at 0.920 / 1.552 ms: Summa remains
about 2.4× / 2.3× slower on this fixture. Old BMP takes 1.757 / 2.990 ms at
99.080% / 99.271%; old exact MaxScore takes 4.269 / 6.085 ms. This is not a
claim of parity or a uniform speedup over the removed backends.

### Production matrix: one million vectors

All sizes below are decimal GB. Build wall times include construction and commit;
query times are warm means over the common 1,000-query fixture.

| Engine                  | Build s | Index GB | Peak build RSS GB |
| ----------------------- | ------: | -------: | ----------------: |
| Previous exact MaxScore |   1.575 |    0.763 |             3.235 |
| Previous BMP (UInt8)    |  27.082 |    1.552 |             3.902 |
| Reference Seismic       |  51.883 |    1.865 |             3.888 |
| Summa Seismic           | 122.856 |    3.702 |             5.437 |

| Engine                  | Top-10 ms / recall | Top-100 ms / recall | Peak query RSS GB |
| ----------------------- | -----------------: | ------------------: | ----------------: |
| Previous exact MaxScore |  33.424 / 100.000% |   46.143 / 100.000% |             0.690 |
| Previous BMP (UInt8)    |   10.544 / 98.760% |    16.559 / 99.175% |             0.852 |
| Reference Seismic       |    1.156 / 99.610% |     2.137 / 99.086% |             2.206 |
| Summa Seismic           |    2.911 / 99.370% |     5.152 / 98.947% |             3.743 |

Fresh Summa reduces warm top-10/top-100 time by 3.62× / 3.21× versus previous
BMP, with +0.610 / −0.228 percentage points of recall respectively. This is not
an equal-recall comparison. Reference remains 2.52× / 2.41× faster with slightly
higher recall and about half the disk size. Summa p95 is 5.269 / 7.826 ms;
reference p95 is 2.125 / 3.278 ms. The index-open and first-query measurements
vary substantially with cache state: the first fresh Summa run opened in
2.719 s and its first query took 514 ms; the following top-100 process opened
in 150 ms. These uncontrolled observations are not cold-start benchmarks.

Exhaustive Seismic top-10/top-100 took 194.3 / 195.8 ms on 20 sampled queries.
Those samples, merged exhaustive top-100 and maintained exhaustive top-100 all
matched the exact oracle's document IDs (maximum shared Float32 score difference
8e-6). The 100,000-vector exhaustive samples also matched. The old exact engine
is substantially faster for exhaustive requests; removing its sparse path is
an explicit architecture choice, not an exact-search performance improvement.

#### Four-segment build, copy merge and maintenance

Four sequential 250,000-document commits built in 216.910 s with 2.734 GB peak
RSS and 6.592 GB of index bytes. Copy merge took 16.965 s, peaked at 6.610 GB
process RSS (including admitted mapped input), and retained 6.592 GB. All four
encoded run lengths and SHA-256 hashes match before/after merge. No nomination
rebuild occurred. The peak process RSS is not the bounded copy-buffer size.

| Layout                        | Top-10 ms / recall | Top-100 ms / recall | Index GB |
| ----------------------------- | -----------------: | ------------------: | -------: |
| Four segments                 |    6.071 / 99.930% |    10.330 / 99.790% |    6.592 |
| One copied segment            |    8.331 / 99.830% |    14.775 / 99.671% |    6.592 |
| After four maintenance passes |    8.032 / 99.830% |    13.367 / 99.638% |    6.412 |

Segment topology changes approximate pruning and available segment parallelism;
copying encoded runs does not promise identical approximate top-k membership.
The 20-query exhaustive result JSON is byte-identical before merge, after merge
and after maintenance. Peak query RSS was 6.636,
6.639 and 6.459 GB for those layouts.

Each maintenance pass rebuilt 64 terms. The four passes took 14.441, 21.850,
15.701 and 20.697 s, peaking at 6.647 GB RSS, and produced about 25.95 GB of
replacement output cumulatively. Remaining debt fell from 27,575 to 27,319 terms:
only 0.93% of fragmented terms were cleared. Index size fell 2.72%; warm top-10
improved 3.59% and top-100 9.53%, with a 0.033 percentage-point top-100 recall
change. Explicitly running four passes exceeds the optimizer's default finite
follow-up allowance. Current bounded maintenance therefore does not quickly
restore fresh-global layout or cost. Whole-file rewriting and selection order
remain material production limitations.

### Why the production index and query RSS remain large

A byte census of the fresh 100,000-vector sparse file finds 101.84 MB of exact
forward values, 735.99 MB of cropped Float32 summaries, 45.47 MB of row
nominations and 5.03 MB of directories/headers. Summaries account for **82.9%**
of sparse bytes. Reference Seismic's serialized index is 413.52 MB versus
Summa's complete 888.44 MB index. Query peak RSS is approximately 905–909 MB
for Summa and 479–496 MB for reference. Summa does not heap-copy the whole
index; format opening touches dispersed cluster headers and query traversal
faults mapped payload pages. Compacting summaries and separating compact
admission metadata from payload pages remain distinct follow-up work.

On one million vectors, the fresh sparse file contains 1.011 GB of exact forward
values, 2.488 GB of summaries, 171.75 MB of nominations and 30.56 MB of metadata:
summaries are 67.2% of sparse bytes. Copy merging four inputs increases summary
bytes to 5.201 GB while forward values stay at 1.011 GB; four maintenance passes
reduce summaries to 5.032 GB. The growth comes from retained nomination/summary
fragments, not duplicate exact forward values.

The reference's `QuantizedSummary::from` quantizes each summary to UInt8 with
per-summary Float32 minimum/scale, transposes coordinates, uses Elias–Fano
dimension offsets, and bit-packs summary IDs (at most six bits for these build
settings). Its forward dimensions cost two bytes versus Summa's four; its
packed forward-offset nominations cost eight bytes versus Summa's four-byte
row IDs. The [reference summary codec](https://github.com/TusKANNy/seismic/blob/3c267137e202748e69ada8cd093f4c0b7c04479c/src/quantized_summary.rs)
therefore provides concrete compression work to evaluate. Summary counts also
differ with clustering/sampling and energy accumulation; these structural
observations do not attribute the entire disk gap to a single encoding choice.

### Separate partition experiment: maintenance locality versus merge cost

An isolated immutable-partition prototype was also evaluated with three
alternating warm-source process runs per layout. It is **not the production
layout** above. One 64-term maintenance pass over about 4 GB of nominations took
median 3.001 s and wrote 4,058,424,582 bytes in one pack. With 128 partitions it
took 1.148 s and wrote 10,796,262 bytes, reusing 126 of 128 packs: 2.61× faster
and 99.73% less new output. Peak RSS was 775 / 720 MB. Logical rows, nomination
payloads and all 110,000 top-10/top-100 result IDs/scores matched. The prototype
uses the earlier capped nomination packs and quantized forward values; matching
its outputs does not establish fresh full-corpus top-L coverage. Its query
summaries remain heap-loaded, unlike the production borrowed-byte views.

The same partition count made nomination-only copy merge 63.3% slower:
2.507 s for one pack versus 4.094 s for 128; peak RSS 20.2 / 142.1 MB.
Sixteen packs measured 2.300 s and 76.8 MB for copy merge, with overlapping timing
ranges versus one pack. Merge here excludes forward values and production
publication. This is evidence for further layout work, not a shipped
maintenance-locality improvement or a selected partition default.

A follow-up closed the 16-partition maintenance measurement gap using the same
frozen prototype binary and three alternating warm-source pairs. Its first
64-term pass took median 1.226 s, wrote 10,807,816 bytes, rewrote one pack and
reused fifteen; the paired monolithic median was 7.461 s for 4,058,424,582 bytes.
Peak RSS was 717 / 771 MB. Monolithic timings differed substantially from the
earlier 128-partition session, so these timings do not rank 16 versus 128.

Continuation exposed the first-pass bias: the next three 16-partition passes
wrote 792,337,076, 760,695,964 and 740,022,823 bytes. Four-pass output totals
**2.304 GB for 16 partitions versus 0.210 GB for 128**, or 10.97× more bytes.
Both reduced pending terms from 27,586 to 27,330, with identical final payload
fingerprints and all 110,000 top-10/top-100 result IDs/scores. Fewer partitions
reduce merge file overhead but can materially increase continued maintenance
writes. Held generations retain additional files; the measured converged
128-partition pass wrote no payloads. Raw results and ownership/cancellation
checks remain in ignored `.context/seismic-experiment/shards/RESULTS.md` and
`16-maintenance-summary.md` beside it.

### Final validation ledger

`CARGO_BUILD_JOBS=3 python3 scripts/check_search.py full` passed all eight stages:
formatting, strict Clippy, 1,755 native tests, native without sync, portable
no-default-feature compilation, documentation, server build, and four additional
real-server broker tests. The main test run reports 19 ignored cases; four of
those are exercised explicitly by the broker stage, while manual performance,
network-dependent tokenizer and ignored documentation fixtures remain excluded.
Evidence: `.context/search-harness/20260917T205906.711675Z-full/`.

The final WASM release build and all 33 WASM tests passed; Python passed 39 tests
against the rebuilt server, and TypeScript passed all 15. Focused diagnostics
feature tests additionally exercised lookup arithmetic, candidate backfill and
fusion. Cache/order tests compare encoded cluster bytes to the original
full-sort oracle, and large-fixture sparse bytes and query results were compared
as described above. The multi-value default mismatch was reproduced before its
fix and tested afterward. Documentation/link checks and `git diff --check` pass.

All new performance evidence is from Apple M4. No new x86/AVX performance run,
controlled cold-storage experiment, concurrent-ingestion workload or relevance
judgment evaluation is claimed. The default backend change follows the explicit
request to integrate Seismic unconditionally; these results do not establish
universal superiority over the removed exact and BMP backends. Benchmark
runners, snapshots, data, results and source fingerprints remain ignored under
`.context/seismic-production/`. Raw benchmark artifacts are not part of the
production change, and this work remains uncommitted.

## Three sparse algorithms and Seismic optimization — September 18

The user clarified that BMP stays the default; Seismic is a third backend beside
BMP and sparse MaxScore. This supersedes the sole-backend decision recorded
above. Restoration retains the shared query/lifecycle correctness fixes.

### Implementation and correctness ledger

- Restored BMP and sparse MaxScore in their existing builders, readers, query
  executors and lifecycle writers. BMP is the default; each sparse field can
  explicitly select BMP, MaxScore or Seismic. Mixed fields share the sparse-file
  envelope and segment publication owner. BMP LSP, BP reorder, pinning, heatmaps
  and dimension diagnostics remain available; Seismic diagnostics coexist.
- Seismic version 2 uses coordinate-transposed UInt8 nomination summaries and
  packed directories. Exact forward values remain one copy at the configured
  precision. Build and maintenance share the term writer and sparse weight
  codecs; MaxScore bulk weight decoding retains its SIMD implementation in the
  shared codec owner. Cached centroid-assignment coordinates preserve the
  previous assignment order; summary quantization requires new recall evidence.
- Ordinary merges copy compatible encoded runs/blocks and rebase metadata.
  Seismic term maintenance, text/BMP BP and binary ANN coalescing use the existing
  claims, concurrency permits, cancellation cleanup and atomic replacement.
  Persisted debt prevents fresh converged vector segments from being repeatedly
  replaced. The captured deletion-file owner remains pinned until maintenance
  publishes using the latest visibility mask. ANN single-copy storage is retained.
- Regressions cover mixed-backend merge/maintenance/deletion compaction, retained
  readers, cancellation and panic cleanup, Seismic encoded-copy identity, shared
  codec bytes, signed Seismic scoring, empty ordinals, filters and native/async
  equivalence. Empty-only Seismic fields retain their rows through public build
  and merge; empty BMP/MaxScore fields retain their previous no-payload behavior.
  BMP and Seismic pinning policies are tested separately. Exact MaxScore fixtures
  now select that backend explicitly instead of depending on the former default.

### Validation ledger

The successful full harness is
`.context/search-harness/20260918T052221.317312Z-full/`, run with
`RUST_TEST_THREADS=4` and `CARGO_BUILD_JOBS=3` on the Apple M4 host.

| Check                                                                                | Result                                    |
| ------------------------------------------------------------------------------------ | ----------------------------------------- |
| Full native core/server/broker/tool suite                                            | 1,945 passed, 24 ignored; no failures     |
| Separate real-server broker integration                                              | All 4 passed                              |
| Formatting, strict Clippy, native without sync, portable core, docs and server build | Passed in the full harness                |
| WASM build and JavaScript tests                                                      | Passed; 36 tests across 7 files           |
| TypeScript client tests                                                              | 16 passed                                 |
| Python client/integration suite                                                      | All 43 passed; Ruff and formatting passed |

The final `python3 scripts/check_search.py check` rerun also passed all four
stages; evidence is `.context/search-harness/20260918T060434.075944Z-check/`.
Documentation validation checked 120 Markdown files, 571 local links and
20 benchmark targets. `git diff --check` passed.

Earlier `full3` exposed two restoration regressions: BMP's implicit vocabulary
bound was enforced but omitted from the error text, and a loader mismatch fixture
still assumed MaxScore was the default. Both were fixed and pass in the full run.
`full4` hit the broker test's 10-second timeout under concurrent test load; the
bounded-parallelism `full5` run above passed that test and the complete harness.
The Python fixture now waits for an actual gRPC handshake instead of sleeping
for two seconds, captures output without undrained pipes, and always reaps its
child on setup failure. Its complete rerun passed. These failures are recorded
separately from benchmark outcomes.

### Remaining costs and measurement gate

Initial Seismic construction still holds decoded rows and inverted candidate
lists and clusters terms sequentially. Its 120-byte-per-vector assignment cache
is temporary; indexing input-memory limits do not impose a hard build RSS ceiling.
Build parallelism remains a separate measured follow-up.

Seismic maintenance bounds selected terms, clustering scratch and retries, but
still copies the whole sparse file for a pass. The prototype's partition-local
I/O reduction is not implemented in production. Insufficient scratch can leave
nomination debt; partial passes preserve that debt and the finite follow-up
count. These limits must be included in build/merge/maintenance comparisons.

Required measurement: current Summa BMP versus current Summa Seismic on the
same corpus, compiler, host, threads and ingestion batching. Report total live
index bytes and sparse payload bytes, initial build time, copy-merge time, peak
RSS, search latency and recall before/after merge, and maintenance separately.
Quantized BMP and Float32 Seismic have different precision; report it alongside
quality, rather than treating their scores as identical. Benchmark outputs stay
in ignored `.context/seismic-three-algorithms/`. Measured results follow.

### Current BMP versus Seismic: method

The ignored evaluator builds both backends from the same SPLADE CSR rows with
vocabulary 30,109, a stored numeric document ID, and identical writer settings:
one indexing builder, two compression threads, four search/Rayon/Tokio workers,
16 GiB indexing budget and no automatic merges. The release binary uses nightly
1.100.0 (215a8af4b), thin LTO, one codegen unit and `target-cpu=native` on Apple M4.
Validation uses the repository's pinned Rust 1.98.1. Input files, binary and
source fingerprints are retained in `reproduction.json` and `source-sha256.json`.

Each dataset has a fresh one-segment build and a four-segment build followed by
ordinary merge. Build time includes create, ingestion and commit; merge time
starts after writer open. Whole-process wall time and maximum RSS are recorded
separately. Disk size is the sum of active-generation files, including stored IDs
and metadata; sparse bytes and total directory bytes are also retained. RSS
includes mapped pages and must not be interpreted as persistent heap allocation.

The first 1,000 original query IDs with at most 64 coordinates are evaluated at
k=10 and k=100. One warmup precedes two measured passes; query latency excludes
stored-ID hydration, while process RSS includes it. The same binary's explicit
Float32 MaxScore backend supplies exact truth. BMP uses intrinsic UInt8 impacts
with scale 5; Seismic preserves Float32 forward values and quantizes nomination
summaries. Approximate recall is measured rather than assuming equal precision.
Exact Seismic samples compare all result IDs and scores with a 1e-4 absolute
score tolerance and require byte-identical results through merge/maintenance.

These are single lifecycle runs on a shared desktop, not isolated throughput
claims. Other workspaces compiled concurrently; per-operation process snapshots
record that load. The Python integration rerun also overlapped early 100K phases.
Disk/encoded-byte comparisons are deterministic; timing and RSS ratios should be
repeated on an otherwise idle host before making capacity commitments.

### 100K lifecycle comparison

Decimal MB (1,000,000 bytes); same corpus and four equal ingestion batches for
fragmented builds.

| Operation               | BMP seconds | Seismic seconds | BMP live MB | Seismic live MB | BMP peak RSS MB | Seismic peak RSS MB |
| ----------------------- | ----------: | --------------: | ----------: | --------------: | --------------: | ------------------: |
| Fresh one-segment build |      10.392 |          65.804 |      161.71 |          468.25 |          507.51 |              639.73 |
| Four-segment build      |       9.838 |          55.772 |      178.03 |          567.52 |          371.85 |              329.94 |
| Four-to-one copy merge  |       0.698 |           2.156 |      161.74 |          567.52 |          190.38 |              586.65 |

All 30 benchmark checks passed. Seismic copy merge preserved every encoded run;
100 exact queries retained identical serialized results through fresh build,
merge and four maintenance passes, with all IDs matching MaxScore and maximum
score difference 0.000007. The four maintenance passes took 17.267 seconds,
reduced pending terms 26,411→26,155 and live size 567.52→550.09 MB. This is bounded
progress, not convergence. Fresh Seismic sparse bytes fell from the prior v1
888,326,149 to v2 468,140,480 (47.3%), while forward storage remained unchanged.

### Remaining layout and build costs

The fresh 100K Seismic v2 sparse file contains:

| Component                                    |       Bytes | Share of sparse file |
| -------------------------------------------- | ----------: | -------------------: |
| Exact forward dimensions and Float32 weights | 101,839,632 |               21.75% |
| Nomination row IDs                           |  45,467,812 |                9.71% |
| Packed summary dimensions                    |  91,360,228 |               19.52% |
| Packed summary posting ends                  |  76,950,558 |               16.44% |
| Packed summary cluster IDs                   |  53,849,279 |               11.50% |
| UInt8 summary weights                        |  91,998,306 |               19.65% |
| Other directories, quantizers and envelopes  |   6,674,665 |                1.43% |

Summaries still occupy 314,158,371 bytes (67.11%). Dimension IDs and cumulative
posting ends alone occupy 168,310,786 bytes: they are packed to a fixed bit width
within each term and repeated across term summaries, without the reference's
Elias–Fano monotone encoding. The largest individual component is the unchanged
single forward copy, with equal bytes for U32 dimensions and Float32 weights.
The complete sparse file is 2.897 times BMP's 161,595,145 bytes; the comparison
also includes BMP's lower weight precision and different search structures.

The v1-to-v2 byte reduction is attributable exactly: summary payloads save
421,828,077 bytes, offset by 1,642,408 additional bytes of cluster quantizers and
term footers, for a net saving of 420,185,669 bytes. This is a storage comparison,
not a claim that quantized nomination results are unchanged.

Ordinary Seismic merge copies each source run, retaining its local top-L lists,
clusters and summary directories. Thus four-segment sparse storage remains
567,403,505 bytes after merge, 99,263,025 bytes above a fresh one-segment build;
forward bytes remain 101,839,632 in both. Bounded maintenance consolidates only
selected terms, explaining the modest four-pass size reduction and remaining
debt. BMP can repack its grid metadata during its streaming block-copy merge;
its sparse size falls from 177,913,899 to 161,626,801 bytes in this fixture.

Seismic term clustering remains serial. Version 2 additionally quantizes and
transposes summaries, sorts their occurrences and packs the integer arrays.
Its measured fresh build took 65.804 seconds versus BMP's 10.392 seconds here.
The earlier Seismic v1 28.1-second build was measured in a different shared-host
session and is not a controlled before/after timing result. No parallel build
or additional summary codec is claimed as implemented by these measurements.

### 1M lifecycle comparison

Decimal GB (1,000,000,000 bytes). These are the current backends from the same
release binary, not substituted historical BMP results.

| Operation               | BMP seconds | Seismic seconds | BMP live GB | Seismic live GB | BMP peak RSS GB | Seismic peak RSS GB |
| ----------------------- | ----------: | --------------: | ----------: | --------------: | --------------: | ------------------: |
| Fresh one-segment build |      27.440 |         129.677 |       1.543 |           2.213 |           3.910 |               5.299 |
| Four-segment build      |      26.539 |         240.011 |       1.557 |           3.537 |           2.606 |               3.007 |
| Four-to-one copy merge  |       3.850 |          11.971 |       1.543 |           3.537 |           1.566 |               3.557 |

Fresh Seismic is 1.434× BMP's size and takes 4.73× its build time. Four-batch
construction reduces build RSS but repeats clustering and summary construction:
Seismic takes 9.04× BMP's fragmented build time. Copy merge is 3.11× slower and
retains a 2.293× larger live index. Seismic's encoded-run fingerprints are
unchanged by merge: the operation copies encoded runs and remaps metadata.

### Query quality and memory

The following are mean warm query times and recall against the same binary's
Float32 MaxScore oracle. Peak RSS is the larger of the k=10/k=100 query processes.
Approximate recall changes across segment layouts; exact Seismic result identity
is checked separately below.

| Dataset / layout | Backend | Top-10 ms / recall | Top-100 ms / recall | Peak query RSS GB |
| ---------------- | ------- | -----------------: | ------------------: | ----------------: |
| 100K / f1        | bmp     |    6.793 / 99.080% |     8.502 / 99.271% |             0.113 |
| 100K / f1        | seismic |    4.260 / 99.580% |     4.613 / 99.484% |             0.478 |
| 1M / f1          | bmp     |   10.593 / 98.760% |    16.316 / 99.175% |             0.851 |
| 1M / f1          | seismic |    2.646 / 99.380% |     4.782 / 98.945% |             2.082 |
| 1M / f4          | bmp     |    6.541 / 98.770% |    11.531 / 99.185% |             0.936 |
| 1M / f4          | seismic |    5.122 / 99.960% |    10.771 / 99.857% |             3.389 |
| 1M / merged      | bmp     |   11.312 / 98.640% |    16.518 / 98.586% |             0.849 |
| 1M / merged      | seismic |    7.206 / 99.830% |    14.558 / 99.671% |             3.390 |
| 1M / maintained  | seismic |    6.884 / 99.830% |    12.293 / 99.638% |             3.322 |

At 1M, fresh Seismic top-10 is 4.00× faster than BMP; after copy merge its advantage
is 1.57×. Fresh top-100 is 3.41× faster, with slightly lower recall than BMP
(98.945% versus 99.175%). Four segments can execute in parallel; copying their
runs into one segment does not reproduce a freshly clustered one-segment index.
This is why the fresh-index result must not stand in for sustained merge behavior.

### 1M maintenance and correctness

Four bounded passes took 56.335 seconds and wrote
13.971 GB of sparse output in total. Live size fell
from 3.537 to 3.463 GB. Pending terms fell 27,575→27,319:
256 terms, or 0.93% of the initial debt. Peak maintenance RSS reached
3.589 GB. Full sparse-file copying dominates
maintenance I/O even though only 64 terms are rebuilt per pass. These four passes
do not establish convergence time or steady-state ingestion throughput.

Both datasets passed all 60 report checks, including identical encoded Seismic
runs through copy merge and byte-identical exact results across fresh, merged
and maintained indexes. The 1M exact sample contains 20 queries; all result IDs
match MaxScore, with maximum score difference 0.000008. The 100K sample contains
100 queries. Supplementary exhaustive BMP checks on 100 queries per dataset
returned identical IDs and scores before/after merge (maximum score error zero);
its approximate recall change is attributable to pruning, not lost stored scores.
Full 1,000-query approximate recall, latency percentiles, disk
components, peak RSS and per-pass debt remain in the ignored report artifacts.

The format improvement reduces the 1M fresh sparse payload from the saved v1
3,700,455,985 bytes to 2,211,397,011 bytes (40.24% smaller).
Its largest component is now exact forward data (1,010,560,752 bytes, 45.7%).
Summary dimensions, offsets, cluster IDs and UInt8 values together occupy
994,914,758 bytes (45.0%). Build scratch and mmap residency remain separate costs;
smaller payloads do not imply a proportionate reduction in peak build heap.

BMP remains the default. Seismic provides a measured query-latency tradeoff,
with larger indexes, slower initial construction and more expensive fragmented
maintenance. Serial clustering, repeated summary directories and whole-file
maintenance output remain open performance findings. No x86/AVX timing,
controlled cold-cache test, concurrent-ingestion benchmark or reference-Seismic
rerun is claimed for this current comparison. Benchmark artifacts remain ignored;
no commit was made.

## Seismic merged queries and partitioned maintenance — September 18

This supersedes the whole-file maintenance implementation measured above.
BMP remains the default, with MaxScore and Seismic as explicit alternatives.

Merged Seismic queries now order the first term's clusters across all copied
runs and score independent run summaries on at most four workers in the existing
search pool. One caller still owns candidate admission, filtering, exact scoring
and the result heap. Portable/asynchronous execution follows the same order
sequentially; exact query semantics are unchanged.

The first 100K run found no query improvement from summary parallelism alone.
A three-second sampling profile of maintained top-100 search attributed 1,859
of 2,247 primary-worker samples to exact forward scoring, including 1,372 in an
out-of-line coordinate iterator; summary orchestration accounted for 92 samples.
Inlining `next` alone did not improve the isolated 100K measurement
(4.410 versus 4.416 ms). The final iterator specializes its `fold` once per vector
precision, retaining the shared precision decoder and scoring order. Release
assembly confirms direct Float32 loads and multiply-adds without per-coordinate
decoder calls. On the same maintained index, top-100 latency fell from 4.410 to
3.614 ms (18.0%), with byte-identical hits across 1,000 queries, one warmup and two
timed passes. No new weight codec, representation or score accumulator was
introduced. Diagnostic sampling and assembly inspection ran separately from
latency measurements; the paired lifecycle matrix below is the final comparison.

Seismic version 3 retains exact forward values once in the sparse root and places
nominations into sixteen immutable partitions shared across fields. Ordinary
merges copy root and partition runs. Maintenance replaces one partition,
hard-links unchanged files through the existing lifecycle, and prioritizes
expensive fragmented terms before its continuation deadline. Its compact term
directory remains sorted even when payloads are emitted in work-priority order.
Earlier Seismic envelopes require rebuild; no legacy reader remains.

The shared cold writer supports bounded partition buffers: 64 KiB each, or 1 MiB
for all sixteen local writers. Maintenance and compaction charge these buffers
and retained TOCs against scratch allowances. Diagnostics include every partition
in sparse file/residency totals and distinguish forward-run count from term debt.

Validation: the `full` harness at
`.context/search-harness/20260918T083553.671224Z-full/` passed strict Clippy,
formatting, 1,967 native tests (24 ignored), native asynchronous/portable checks,
documentation, server build, and four additional real-server broker tests.
The final WASM build and all 36 WASM tests passed.
Python passed 43 tests and TypeScript passed 16. The no-feature core check retains
unused-builder warnings; the final WASM build has no new Rust warnings.

Cross-review also closed three boundary cases: nonempty term bases at the
forward-row end now fail admission; surviving partitions require a declared,
nonempty root; mixed BMP/MaxScore compaction reserves live Seismic writer scratch.
Regressions preserve valid empty fields and empty terms at the row boundary.

The paired benchmark and source/binary fingerprints are ignored under
`.context/seismic-merge-repair/`. Its checks compare exact result bytes between
versions, fresh forward/nomination payload bytes, and encoded-run identity through
ordinary merge. Maintenance inventories distinguish new output from hard-linked
files; output bytes are not a measurement of physical device traffic.

### Partition-selection heuristic review

Partition maintenance currently selects the largest fragmented-term count, then
its encoded nomination bytes, with a stable partition-ID tie-break. Within the
selected partition, terms with more runs and then more encoded bytes run first.
This policy retires term debt predictably on the approximately uniform modulo-16
benchmark partitions, but it is not an estimate of query frequency or guaranteed
byte savings. On a skewed vocabulary, many small fragmented terms can outrank a
partition containing fewer, much larger terms. A bytes-first partition policy is
a follow-up candidate and should be compared by bytes retired, maintenance time,
query latency, and recall before changing this first measured implementation.

Memory admission can also leave the selected terms unchanged. Existing bounded
maintenance follow-up counts limit repeated work; changing partition priority
alone does not resolve this case. The regression suite now exercises a partially
maintained generation merged with another segment: consolidated and untouched
nomination partitions coexist with multiple forward runs, while signed weights,
multi-value sums, missing rows, deletion visibility, and held readers remain
correct.

### Final paired v2/v3 measurements

The final matrix uses the same CSR corpus, Apple M4, nightly Rust
`1.100.0-nightly (215a8af4b)`, `-C target-cpu=native`, thin LTO and one codegen
unit for both versions. Seismic forward precision is Float32; four search
threads, one indexing builder and two compression threads are unchanged. Query
means cover the same 1,000-query sample, one warmup and two timed passes, excluding
hydration. No builds or test suites ran alongside timing. These are warm-process
measurements on a shared desktop, not controlled cold-cache or steady-state
concurrent-ingestion measurements.

| Measurement                      | 100K before → after | 1M before → after   |
| -------------------------------- | ------------------- | ------------------- |
| Fresh build                      | 30.151 → 30.994 s   | 130.401 → 130.769 s |
| Four-segment build               | 34.453 → 34.149 s   | 229.115 → 231.147 s |
| Copy merge                       | 2.064 → 2.480 s     | 11.590 → 13.672 s   |
| Fresh top-10                     | 1.879 → 1.684 ms    | 2.617 → 2.349 ms    |
| Fresh top-100                    | 3.266 → 2.731 ms    | 4.782 → 4.275 ms    |
| Merged top-10                    | 2.828 → 2.428 ms    | 7.322 → 6.277 ms    |
| Merged top-100                   | 4.718 → 3.938 ms    | 13.704 → 12.248 ms  |
| Maintained top-10                | 2.538 → 2.293 ms    | 6.677 → 6.095 ms    |
| Maintained top-100               | 4.207 → 3.749 ms    | 12.497 → 11.958 ms  |
| Four maintenance passes          | 11.252 → 7.160 s    | 58.851 → 15.757 s   |
| Sparse output across four passes | 2.227 → 0.101 GB    | 13.971 → 0.522 GB   |
| Maximum maintenance process RSS  | 0.624 → 0.603 GB    | 3.588 → 3.122 GB    |

GB means decimal bytes. Each pass retires 64 fragmented terms in both versions;
remaining debt after four passes is 26,155 terms at 100K and 27,319 at 1M. Every
v3 pass creates exactly one nomination partition, reuses fifteen partitions and
the exact-forward root, and emits roughly 2 KB of other metadata. The 1M passes
are 3.7× faster in aggregate and emit 26.8× less sparse payload. The slowest v2
pass took 23.376 s; single-run lifecycle timings include desktop/filesystem
variation and should not be treated as a tight throughput guarantee.

Fresh live index size changes only by 2,704 bytes per segment. At 1M, fresh live
size is 2.2125 GB, four independent segments total 3.5375 GB and their merged
index totals 3.5374 GB. Copy merge itself does not inflate the index: it retains
the separately selected nomination lists already present in those segments.
After four passes live size is 3.4460 GB, versus 3.4627 GB before this change.
Query-process RSS is essentially unchanged: about 2.08 GB fresh and 3.39 GB
merged. Lower maintenance RSS reflects a smaller working set, not a claim that
corpus-sized mmap payloads became resident heap.

Copy merge is 18–20% slower in this matrix despite identical copied run bytes.
The partitioned layout adds file/directory work; this measurement does not
isolate filesystem, cache and per-file costs. Reducing that overhead remains an
open optimization. It is still a streaming copy/remap operation with no
reclustering. Faster maintenance does not restore the fresh layout after four
passes: 1M maintained top-10 remains 6.095 ms versus 2.349 ms fresh. The bounded
sample does not establish convergence time or sustained maintenance capacity.

All 98 benchmark correctness/representation checks passed. Fresh forward and
nomination payloads are byte-identical across versions, ordinary merge preserves
encoded runs, and exact hit files are byte-identical across versions and across
fresh/merged/maintained states. Exact comparison against MaxScore covers 100
queries at 100K and 20 at 1M (absolute score tolerance 1e-4). Approximate recall
is measured over all 1,000 queries: 1M merged top-10 changes from 99.830% to
99.820%, merged top-100 from 99.671% to 99.661%, and maintained top-100 from
99.638% to 99.669%. At 100K the largest absolute recall change is 0.05 percentage
points. Fresh recall is unchanged. Exhaustive 1M top-100 also improves from
195.301 to 115.164 ms; no exhaustive correctness tradeoff is introduced.

The report, latency percentiles, run hashes, inode inventories and source/binary
fingerprints remain ignored under `.context/seismic-merge-repair/`. Final binary
SHA-256 is `de535e7893d63fd94b1e72c4e8466584500ee7366fe758058ca4dca3d7d8092e`.
No x86/AVX benchmark or reference-Seismic rerun is claimed. BMP remains the
default; earlier Seismic formats require rebuilding. No commit was made.

### Full Seismic consolidation endpoint — September 18

Continued an isolated, immutable-file-linked copy of the measured 1M index from
pass four until persisted nomination debt reached zero. The public explicit
`IndexWriter::reorder` operation and final measured binary/settings are unchanged:
64 terms maximum per pass, one partition per pass, four search threads, Float32
forward values. Each pass ran in a new process without optimizer cooldown.
The original four-pass index remains unchanged. This is manual completion;
the server's default three-pass replacement-lineage follow-up limit was bypassed
by explicit calls, not changed or tested as an automatic completion policy.

**Full consolidation does recover the fresh index's size on this fixture.**

| State                          |    Live bytes | Decimal GB | Pending terms |
| ------------------------------ | ------------: | ---------: | ------------: |
| Fresh single-segment build     | 2,212,514,503 |   2.212515 |             0 |
| Copy-merged four segments      | 3,537,446,009 |   3.537446 |        27,575 |
| After four passes              | 3,445,997,567 |   3.445998 |        27,319 |
| Fully consolidated, 433 passes | 2,212,434,943 |   2.212435 |             0 |

The final index is 79,560 bytes (0.0036%) smaller than fresh. Both layouts have
1,010,560,752 bytes of forward coordinates/weights, 24,000,000 bytes of row
directories, 171,750,264 bytes of nomination IDs, and 681,713 clusters. The
remaining size difference is predominantly encoded summaries: maintenance seeds
clustering with local row numbers, so rebuilt summaries need not be identical
to fresh-build summaries. Retaining the four forward-run envelopes adds only
192 bytes. A zero-debt layout is not a promise of byte-identical clustering.

Cumulative maintenance cost, including the first four measured passes:

- 433 successful passes, retiring all 27,575 fragmented terms; no stalled pass.
- 866.339 s (14.44 min) inside maintenance operations, or 869.510 s summed
  process wall time. The additional Python inventory/census work is excluded.
- 34,639,381,775 bytes (34.639 GB) of new sparse output; unchanged hard links
  are excluded. This is emitted file payload, not physical device traffic.
- Maximum per-process maintenance RSS: 3,121,823,744 bytes (3.122 GB).
- Four-segment build + copy merge + full maintenance: 1,111.158 s (18.52 min),
  versus 130.769 s (2.18 min) for the original fresh single-segment build.

Size converges well before all term debt is retired: pass 97 reached 2.314 GB
in 331.023 s, pass 188 reached 2.235 GB in 526.358 s, and pass 271 reached
2.214 GB in 660.639 s. Final consolidation continues rewriting mostly unchanged
partition bytes while handling smaller lists. These measurements identify
partition rewrite amplification and bounded-pass scheduling as remaining costs;
they do not justify changing defaults from this single corpus/architecture.
Individual term reclustering can increase summary size slightly, so live size
is not strictly monotonic between passes.

Search uses the same 1,000-query sample, one warmup and two measured passes,
excluding hydration. Fresh search was rerun after the consolidated search in
the same session, with no concurrent builds/tests. Original merged timings are
retained from the paired matrix, not rerun in this continuation.

| State                         | Top-10 mean | Top-10 recall | Top-100 mean | Top-100 recall |
| ----------------------------- | ----------: | ------------: | -----------: | -------------: |
| Fresh, same-session rerun     |    2.269 ms |       99.380% |     4.217 ms |        98.945% |
| Copy-merged, prior paired run |    6.277 ms |       99.820% |    12.248 ms |        99.661% |
| Fully consolidated            |    2.441 ms |       99.610% |     4.339 ms |        99.143% |

Final query RSS is about 2.08 GB, matching fresh and below the merged 3.39 GB.
Consolidated latency is 7.6% above fresh for top-10 and 2.9% above fresh for
top-100, with higher recall on this sample. Relative to the larger merged
nomination lists, consolidation lowers recall by 0.210 and 0.518 percentage
points respectively; that is the measured cost of returning to one global
nomination budget. No equal-recall latency comparison is claimed.

Every continuation pass checked nonincreasing term debt, at most 64 retired terms,
unchanged exact-forward root ownership and at most one new partition with at
least fifteen reused partitions. Final root bytes match the starting root by
SHA-256. Exact top-100 hit files for the same 20-query sample are byte-identical
to fresh; approximate recall is measured against Float32 MaxScore over all
1,000 queries. The source snapshot remained unchanged. No engine code changed,
so native/WASM suites were not rerun for this measurement; the final binary is
the same one covered by the validation above.

Per-pass results, memory, inode checks, component sizes, query output and the
report are ignored under `.context/seismic-full-consolidation/`. This establishes
full completion for this one 1M corpus, not automatic scheduler completion,
concurrent-ingestion capacity, cold-cache performance, or a general guarantee
that all corpora converge to the same fresh-build size. No commit was made.

## Seismic maintenance throughput and reuse follow-up (2026-09-18)

This follow-up implements requested items 1, 2, 4 and 5 from the full-consolidation
review. It leaves the existing run-count/byte priority order unchanged. BMP
remains the default sparse backend; merge still copies encoded Seismic runs.

### Changes and ownership

- Segment metadata now separates successful maintenance passes from consecutive
  no-progress passes. Productive published work resets the stall count and stays
  eligible beyond three passes. Three consecutive successful no-progress passes
  stop automatic follow-up under the existing default policy; failures retain
  their existing backoff and do not publish counter changes. Cooldown, shared
  concurrency, and per-pass admission remain in force.
- One selected nomination partition consumes the available time and scratch
  budget instead of stopping after 64 terms. An admitted term can finish beyond
  the deadline; remaining terms copy unchanged. Directory scratch is charged,
  and decoded term/assignment scratch is retained for one term at a time. A term that
  cannot fit is retained while later affordable terms may still progress.
- Automatic maintenance derives whether text/BMP reordering is still due under
  the claimed source metadata lock. Completed or retry-capped BP work reuses
  its existing files and counters during Seismic follow-up. Explicit manual
  reorder retains its meaning. Both operations share the same writer,
  cancellation, publication and retirement lifecycle.
- A cold-writer coalescing candidate was evaluated and rejected: the isolated
  copy loop improved, but paired whole merges were slower. The existing shared
  cold-writer implementation and buffer capacities remain unchanged.

### Reuse findings

RGB/BMP already warm-start from the published document/per-field ordering;
ordinary merge preserves compatible encoded ordered blocks. A new BP pass rebuilds its
temporary graph but begins from that existing order. The mixed-backend fix above
also avoids repeating completed BP work for unrelated Seismic debt.

Seismic reuses completed term payloads and untouched partitions. Its clustering
is sampled one-pass assignment, so there is no iterative optimizer checkpoint
to resume. Cropped quantized summaries cannot serve as complete centroid state.
Reusing an existing term when global top-L selection is unchanged is a possible
future optimization, subject to live-row coverage, deletion, ordinal, precision
and address-remapping proofs. That shortcut is not implemented or measured here.

### Copy-merge diagnosis and paired timing

An Apple M4 profile of the frozen baseline sampled approximately 6,808 stacks
under nomination copying (6,796 in writes), 1,678 under forward copying (1,616
in writes), and 188 across fsync sites out of 10,407 samples. The 12-second
sample covers part of a slower diagnostic run; it is not a complete attribution
of every merge. Output partition admission also appeared in about 859 samples.
The isolated copy probe reproduces actual ragged run sizes with `F_NOCACHE`
and fsync: old buffering took 5.462/4.074 s, coalesced buffering 2.455/2.678 s.
It does not include source/output admission or retain all seventeen writers
simultaneously, so those numbers are not end-to-end merge latency.

Whole-merge timing used disposable hard-link clones of the same unchanged four-
segment 1M input. No builds or other agent benchmarks ran concurrently. Each
run verified canonical encoded run hashes before/after merge. The original
baseline runs were 14.644, 16.518 and 13.559 s; initial changed-code runs were
28.164, 15.803 and 13.335 s. Follow-up interleaving of frozen/current binaries
produced 9.767/11.532 s and 10.622/10.901 s. Peak RSS remained approximately
3.44 GB. Thus the measurements do **not establish a whole-merge improvement**;
the interleaved current median is about 10% slower. The first post-build run is
retained in the evidence rather than discarded as an assumed cache effect.

The rejected candidate passed regression tests for exact bytes, retained scratch,
byte counts and an unchanged source offset on unsupported range copying. They
do not explain all elapsed-time variation. Seismic admission still scans each
term's cluster and summary directories when opening source/output readers;
those accesses live within evictable payload mappings. The candidate was reverted after these timings. No removal of admission checks
or change to cache policy is justified by this experiment.

After reverting the writer candidate, the retained release binary's repeated
copy merges took **27.258, 15.219 and 15.358 s**, with identical encoded runs and
maximum RSS **3,440,279,552 bytes**. The writer source matches the frozen baseline
byte-for-byte. User CPU time stayed near two seconds across baseline/current
runs, while recorded page faults ranged from about 125K in the interleaved
controls to 338K in the first post-build runs. This is consistent with a large
I/O/cache contribution, not proof of a controlled cold-cache comparison. The
copy-merge speed gap remains unresolved; no end-to-end gain is claimed.

### Repeated ingestion, merge and maintenance: paired 100K

One shared ignored runner was compiled against the frozen and retained sources
with identical nightly compiler, native CPU flags, thin LTO and four search
workers. Each run appended four 25K batches, merged after each append, performed
two maintenance passes between early cycles, then drained final debt. Maintenance
used a two-second term-admission budget. One concurrent top-10 query task used a
10 ms think time and held each phase's original reader snapshot. Writer shutdown,
reopen, exact checks and the final independent MaxScore oracle ran outside timing.

| Measurement                                    |        Frozen baseline |        Retained code |
| ---------------------------------------------- | ---------------------: | -------------------: |
| Append/commit time, all four cycles            |               34.783 s |             36.868 s |
| Copy-merge time, all four cycles               |                6.227 s |              6.814 s |
| Maintenance time, all cycles                   |              279.000 s |             85.185 s |
| Maintenance passes, all cycles                 |                    422 |                   37 |
| Cumulative sparse maintenance output           |        9,901,714,204 B |        836,149,783 B |
| Final-cycle drain                              | 273.695 s / 418 passes | 74.478 s / 33 passes |
| Whole-process wall time, including correctness |              329.350 s |            139.350 s |
| Whole-process peak RSS                         |        1,469,661,184 B |      1,429,946,368 B |
| Sampled peak unique-inode directory bytes      |        1,126,388,141 B |      1,120,869,073 B |
| Final live bytes                               |          468,378,078 B |        468,378,077 B |
| Final fragmented-term debt                     |                      0 |                    0 |

Maintenance is 3.28x faster overall and emits 91.6% fewer sparse bytes in this
workload. Build/merge timings do not show an improvement; total RSS and sampled
peak disk remain similar. The final one-byte size difference is metadata, not
a material storage reduction versus the old fully consolidated endpoint.

Final-drain queries: mean 2.874 -> 2.984 ms; p50 2.678 -> 2.761 ms;
p95 4.991 -> 5.002 ms; p99 7.153 -> 8.422 ms. Query count differs because the
maintenance interval is shorter (18,609 vs 5,017); percentiles use a bounded
1,024-sample reservoir. Earlier maintenance phases also have higher query tails
when more clustering is admitted per pass (cycle-two p95 3.045 -> 5.310 ms).
Thus this demonstrates a shorter maintenance interval, not universally lower
concurrent query latency. A smaller time budget trades completion rate for
shorter CPU bursts; defaults were not retuned from this one experiment.

Term debt after the two early passes was 24,486 -> 22,730 at cycle two and
25,818 -> 24,417 at cycle three (baseline -> retained). Later ingestion recreates
fragmentation, so debt is measured again after each merge rather than assumed
monotone across the full workload. The counter measures fragmented terms, not
all additional run multiplicity on already fragmented terms.

All four cross-version exact checkpoint files are byte-identical. Each run also
checks old held snapshots, post-merge/maintenance/reopen results, and final
20-query top-100 external IDs plus scores within absolute 1e-4 of MaxScore.
Approximate recall is not measured by this load test. Disk peaks are 100 ms
sampled lower bounds, deduplicated by inode; cumulative output excludes reused
hard links and is not physical SSD traffic. This is serial append/merge/
maintenance with overlapping queries, not continuous ingestion alongside
maintenance, saturation throughput, or automatic-scheduler wall time. The
automatic completion/cooldown policy is tested separately in native/server tests.

### Full 1M consolidation from the same checkpoint

Both versions start from the identical four-pass snapshot: 3,445,997,567 live
bytes and 27,319 fragmented terms. The comparison below excludes those four
common passes. Explicit manual maintenance has no deadline, admits one partition
per pass, and keeps the same scratch policy. This does not include automatic
scheduler cooldown. Final retained runner SHA-256:
`5926a766273100567c96982be2730e2e821dd302d67824b8eb3cb085a9ffaacf`.

| Remaining consolidation    |  Frozen baseline |   Retained code |
| -------------------------- | ---------------: | --------------: |
| Passes                     |              429 |              16 |
| Operation time             |        850.582 s |       245.242 s |
| Summed process wall time   |        853.721 s |       245.376 s |
| New sparse output          | 34,117,382,398 B | 1,176,758,848 B |
| Maximum per-process RSS    |  3,063,414,784 B | 3,064,119,296 B |
| Final live bytes           |  2,212,434,943 B | 2,212,434,942 B |
| Remaining fragmented terms |                0 |               0 |

This is **3.47x faster**, with **96.6% less emitted sparse output**, and essentially
unchanged peak RSS. Every retained pass produced one new partition, reused the
other fifteen, and wrote zero exact-forward bytes. Canonical hashes of **every
forward and nomination payload** match the old fully consolidated endpoint.
The one-byte final live-size difference is metadata: the generation counter is
21 instead of 434. Including the four common passes gives 20 total passes for
this continuation, versus the previously measured 433; this is not a fresh
merge-to-consolidated 16-pass timing.

Exact top-100 hits for 20 queries and approximate top-10/top-100 hits for 1,000
queries are byte-identical to the old consolidated endpoint. Thus the prior
measured recall remains 99.610% / 99.143%. Endpoint search was also compared with
the same retained binary in the same session:

| Endpoint                      | Top-10 mean | Top-100 mean | Query RSS range |
| ----------------------------- | ----------: | -----------: | --------------: |
| Old fully consolidated layout |    2.460 ms |     4.520 ms |  2.076–2.083 GB |
| New fully consolidated layout |    2.511 ms |     4.506 ms |  2.080–2.083 GB |

No query-speed improvement is claimed. Top-100 p95 was 6.891 -> 6.881 ms and
p99 8.043 -> 7.898 ms in that paired endpoint check. The same nominal nomination
budget and identical hits are retained; these changes primarily remove repeated
maintenance output and scheduling overhead.

An unrelated workspace compilation interrupted the first consolidation attempt
after four normal passes and inflated subsequent timings. That partial run is
preserved as `interfered-consolidation`; the final measurement restarted from
the original checkpoint after compilation ended. No compiler appeared in the
clean maintenance boundary snapshots. Another short compilation overlapped the
initial top-100/exact-query phase: its 10.501 ms top-100 result is retained as
contaminated evidence and excluded from the comparison above. The follow-up
endpoint runs had no compiler overlap in their boundary snapshots and verified
identical hit files. No other workspace process was stopped or modified.

All harness code, binary/source fingerprints, raw timing, phase distributions,
file inventories, payload checks and interrupted attempts remain ignored under
`.context/seismic-next/`. `retained-evaluation-summary.json` records the completed
comparisons. The initial `retained-matrix.log` ends with the intentional harness
interruption; the separately restarted consolidation completed successfully.

### Validation of the retained code

`python3 scripts/check_search.py full` passed after reverting the writer
candidate: **1,974 native tests**, 24 intentionally ignored, plus **4 real-server
broker tests**. Formatting, strict Clippy, native-without-sync and portable
compilation, API docs and server build passed. Evidence:
`.context/search-harness/20260918T103709.099807Z-full/`.

The WASM build and **36 WASM tests** passed for the scheduler/batching changes;
the subsequent revert only restores native cold I/O, outside the WASM build.
`uv run scripts/check_docs.py` checked 120 Markdown files, 571 local links and
20 benchmark targets. Direct system-Python invocation initially lacked
`markdown_it`; the documented `uv run` command succeeded. Nightly release builds
retain the pre-existing `fetch_update` deprecation warnings; pinned-toolchain
strict Clippy passes. Python/TypeScript suites were not rerun for these internal
metadata/lifecycle changes; their preceding 43/16-test results remain recorded
above. No wire/schema/default change is introduced here.

Behavior regressions cover more than three productive passes; persisted stalled
work and reset after progress; merge/replacement accounting; cancellation,
panic and failed publication; deadline, memory refusal and untouched payload
bytes; and automatic completion of all sixteen nomination partitions. Mixed
schemas test completed, capped and still-eligible BP, inode/byte reuse and held
reader results. Execution was on macOS/aarch64; no Linux/x86 rerun is claimed.

## 2026-09-18: copy-merge admission and bounded local copying

The isolated Linux four-to-one Seismic copy-merge is **2.25x faster**, including
source admission, output writing, output admission and publication. This keeps every
encoded run unchanged and preserves structural corruption checks. BMP remains
the default; no format, schema or query-budget setting changes.

### Paired whole-merge measurement

The fixture contains 1M documents in four source segments, with approximately
3.537 GB of live files. Both versions use the same source, Rust 1.98.1, release
thin LTO, one codegen unit and `-C target-cpu=native`, on a dedicated GCP
`n2-highmem-8` Linux/x86-64 machine with a 1 TB SSD persistent disk. Each mode has
three before and three after trials, alternating in the order before, after,
after, before, before, after. Inputs are warmed identically; outputs use the
unchanged cold writer. No compiler runs overlap timing. The fallback mode forces
`copy_file_range` to return unsupported before any output using an ignored
benchmark shim; the production binary handles the actual fallback.

| Whole merge                                      |          Before |           After |
| ------------------------------------------------ | --------------: | --------------: |
| Kernel-copy median                               |        25.928 s |        11.500 s |
| Kernel-copy maximum process RSS                  | 2,538,156,032 B | 2,538,037,248 B |
| Major page faults, each timed run in either mode |          13,112 |             126 |
| Fallback-copy median                             |        25.312 s |        11.548 s |
| Fallback-copy maximum process RSS                | 3,556,642,816 B | 2,549,891,072 B |

Times above measure `force_merge`; whole-process medians, including writer
opening and teardown, are 25.934 -> 11.506 seconds for kernel copying and
25.318 -> 11.556 seconds for fallback copying.

The kernel path takes **55.6% less time**. The fallback path is **2.19x faster**
and uses **28.3% less peak process RSS**. RSS includes resident mmap pages; it is
not a measurement of total system page cache or heap alone. The local-file
fallback uses at most 4 MiB of staging scratch instead of faulting the mapped
source as it writes. Kernel copying stays preferred. One shared helper serves
encoded ranges and temporary sparse sections, including BMP callers.

### Where the time went

A phase trace of the original code spent about 4.4 seconds writing output and
7.4 seconds through source admission plus merging, but 26.8 seconds through the
whole operation. The completed cold output must then be reopened for structural
admission and lifecycle statistics. That last scan accounted for about 19
seconds. Fsync itself took only about 0.03 seconds in the trace.

Seismic admission now checks extents first and visits terms in physical order,
reusing its existing extent scratch. Linux mmap admission hints the current
4 MiB window plus one lookahead, split into requests of at most 128 KiB. Other
platforms and heap-backed readers receive no new advice. All term checks remain;
no live metadata is repaired and no output admission is skipped.

Request size matters: an earlier candidate hinted whole 4 MiB windows and only
reduced kernel-copy median from 25.688 to 24.749 seconds. Linux 7.0
[`force_page_cache_ra`](https://github.com/torvalds/linux/blob/v7.0/mm/readahead.c#L325-L340)
caps an individual request using device read-ahead/I/O limits. This disk reports
128 KiB read-ahead and 256 KiB maximum I/O. Smaller requests cover the selected
window instead of assuming a successful large hint fetched its entire range.
A diagnostic screen reduced major faults from about 12,600 to 126 and output
admission to about 4.2 seconds; the table above measures the final implementation
without that diagnostic wrapper.

Staging alone did not improve whole-merge speed: its paired kernel medians were
26.288 -> 26.031 seconds, and fallback medians 25.948 -> 26.038 seconds. It did
remove about 1 GB of fallback process RSS. The final speedup comes from fixing
the admission read pattern, rather than attributing it to the copy buffer.

### Correctness, validation and limits

Every timed merge checks hashes of every copied encoded run against the same
canonical input. Output remains 3,537,446,009 bytes, including 3,536,330,892 sparse
bytes. Exact top-100 IDs and scores for 20 queries and approximate top-100
IDs and scores for 200 queries are byte-identical before/after in both copy
modes. The change copies and remaps metadata; it does not recluster or recompute
vectors. Copy regressions cover partial and interrupted I/O, truncation,
cancellation, abstract directories and scratch cleanup. Admission regressions
cover shuffled physical term order, unchanged scoring/debt, malformed term
metadata and bounded read-ahead, including overflow edges.

`python3 scripts/check_search.py check` passed: **1,988 native tests**, zero
failures and 24 intentionally ignored, plus formatting, strict Clippy and portable
compilation. Evidence is in `20260918T122848.743110Z-check`. The final WASM release
build and **36 WASM tests** passed. The first cloud harness attempt failed before
checks because the uploaded snapshot lacked Git metadata; a remote snapshot Git
index fixed the harness environment without creating a commit. That failed
attempt is retained alongside successful evidence. The full RPC/broker lifecycle
suite was not rerun for these internal copy/admission changes; the preceding full
run is recorded above. Local documentation/link and whitespace checks passed.

These are Linux cloud-disk results, not a macOS throughput claim or a comparison
with the reference Seismic engine. Local timings were excluded because competing
workspace compilers and low disk space prevented a clean measurement. No other
workspace jobs were stopped. This work does not claim improved query latency.
Raw binaries, source fingerprints, timing, phase logs, payload hashes and query
outputs remain ignored under `.context/seismic-copy-merge/`; no benchmark
artifacts were staged or committed.

### Memory-pressure investigation

The following measurements and gaps describe the control before the residency
and query-I/O changes recorded in the next section.

Seismic's local server/tool reader uses `MmapDirectory`: bulk encoded values are
file-backed and evictable, while only compact run objects/settings remain on the
heap. This is not a promise for every directory backend: `FsDirectory`, RAM and
HTTP return heap-backed whole-component buffers on this path. The control's pin
policy does not include Seismic's term or row directories. Admission read-ahead
warms pages but does not lock them.

A census of the exact 1M/four-source fixture measured these encoded byte sizes:

| Section                                          |         Bytes |
| ------------------------------------------------ | ------------: |
| Document/ordinal/forward-offset directory        |    24,000,000 |
| Term directory                                   |     4,394,800 |
| Cluster headers and term footers                 |    20,071,856 |
| Summary dimension/offset/occurrence/value arrays | 2,138,175,719 |
| Exact forward values                             | 1,010,560,752 |
| Nomination IDs                                   |   339,121,628 |

The first two total only 28.4 MB; all summaries total 2.14 GB. Integrating the
existing pin budget with compact directories is the first residency opportunity.
Summary caching should be selective and independently bounded, rather than
pinning entire nomination files. Copy-mode pinning must redirect reader views
to the retained copy; an unused duplicate allocation would not help. Scattered
header ranges may require more locked pages than their encoded byte count.
These were the opportunities identified before the implementation below.

Approximate query scratch is bounded by 262,144 nominated documents and aggregate
clusters per segment execution, a lookup table of at most 256 KiB, and at most
four summary tasks sharing a score buffer. Concurrent segment queries multiply
those bounds. Exhaustive queries scan without allocating corpus-sized score
arrays. Maintenance admits directory/per-term scratch against its allowance and
leaves debt when a term cannot fit. Initial construction still holds decoded
rows, inverted candidates and assignment scratch; a successful low-memory
merge/search does not establish a comparable initial-build memory bound.

The control also lacks query-time random-access advice and selected-range prefetch.
BMP and clustered ANN mark random-access payloads accordingly and prefetch bounded
selected extents; Seismic's scattered summary and forward-vector reads currently
use default mapping advice. Read-around amplification is therefore a candidate
cause of pressure-induced I/O, not a proven attribution. A follow-up should
compare faults and read bytes with random-access advice after admission, then
add bounded selected-range prefetch in the owning reader. Preserve scoring order,
avoid broad min-to-max spans and per-query policy toggles, and provide explicit
read-ahead for exhaustive/background scans before adopting a persistent policy.

#### Enforced memory limits on the merged fixture

A separate probe uses Linux cgroup v2 `MemoryMax` with swap disabled. This limit
covers process allocations and charged file cache, not just RSS. Each case
starts from the same four source segments after file-specific cache eviction,
merges into a new output, closes it, evicts that output's cached pages, and then
opens it for search. Approximate search uses the same 200 queries/top-100 over
three passes; reported mean/p95 exclude the first pass. Exact search uses three
queries/top-100 over three passes, with a separate cold start. Each cap has one
probe, so these are pressure diagnostics, not repeated throughput medians. They
must not be compared directly with the warm-source merge timings above.

| Memory cap | Cold-source merge | Approximate mean | Approximate p95 | Exact mean |
| ---------- | ----------------: | ---------------: | --------------: | ---------: |
| 4 GiB      |          20.635 s |        23.661 ms |       38.370 ms | 241.592 ms |
| 1 GiB      |          26.662 s |       916.083 ms |    1,654.452 ms | 253.017 ms |

Both completed cases reach their cache-inclusive cap with zero cgroup OOM events
or kills, preserve all copied-run hashes, and return identical approximate and
exact IDs/scores. At 1 GiB, approximate search is about 38.7x slower than at 4 GiB.
The three-query exhaustive result is a small functional/scan probe, not evidence
that exhaustive search generally outperforms nomination. Query mix and repeated
working set matter. Initial build and constrained-memory maintenance are not
measured here.

At 512 MiB, copy merge completed in 29.294 s with identical run hashes. The
200-query approximate probe exceeded its 1,200 s guard; no completed latency or
hit comparison is available, and exact search was not run for that case. The
last captured cgroup snapshot showed no OOM events, but it was taken before the
timeout and is not a final event count.

### Budgeted Seismic directories and query-I/O experiments

The initial candidate below preserved the encoded format, nomination order and
scoring. Its query-I/O policy was later removed after warm-query regressions;
only compact directory pinning and shared copy-pin read-ahead are retained:

- The existing per-segment pin policy now admits term directories before row
  directories, redirecting copy-mode lookups to the retained bytes. Original
  encoded owners remain available for streaming merge. Disabled-policy reports
  now retain intended/skipped sparse metadata bytes.
- Linux readers set random-access advice after admission and prefetch selected
  summary coordinates and forward rows with bounded requests. Full scans and
  maintenance retain rolling forward read-ahead across documents and ordinals.
  Cancellation is checked before I/O batches and ordinal expansion.
- The shared Linux copy-pin helper uses bounded rolling read-ahead when copying
  admitted metadata. The fixture has 28,394,800 bytes of term/row directories
  (27.08 MiB); summaries, nominations and exact vectors stay mapped and evictable.
  An additional 10,014,624-byte summary-routing table was tested and removed;
  its measured trade-offs are recorded below.
- Rust sparse-vector and sparse-term constructors now reuse the shared query
  defaults instead of repeating Seismic constants. No schema, query or pin
  default changed.

Initial construction is a separate remaining memory issue: the indexing memory
limit currently triggers flushing based on incoming sparse coordinates, rather
than establishing a hard bound on decoded rows, inverted nomination candidates,
assignment cache and encoder workspace. A hard construction cap needs a
calibrated peak estimate and pre-admission handling of an oversized document.
The reader changes do not claim to provide that cap.

The residency review also identified a preexisting limitation in generic ANN
heap locking: independently owned malloc allocations may share an OS page, and
`munlock` does not preserve another owner's lock on that page. The experimental routing
table used page-exclusive anonymous allocations with rounded budget admission;
that entire table implementation was subsequently removed after measurement. The
generic ANN heap-lock ownership issue remains a separate follow-up; it affects
residency guarantees rather than decoded values or search correctness.

#### Intermediate routing evaluation (not the retained performance claim)

The first 1 GiB / 64 MiB copy-pin probe preserved all copied-run hashes and
all 200 approximate plus three exact query outputs. Approximate mean fell from
916.083 to 719.062 ms, but cold open increased to 108.549 s and the merge operation
to 129.286 s. The query process read 28,689,376 filesystem input blocks (14.69 GB
at 512 bytes/block), versus the control's 62,096,224 blocks (31.79 GB). Its
360,263 major faults were lower than the control's 1,080,087, but that reduction
did not translate proportionally to latency.

The three-query exact probe regressed from 253.017 ms to 3,740.896 ms. The
forward payload alone is 1.011 GB, close to the enforced 1 GiB cap; additional
resident metadata can change whether repeated exhaustive scans fit. This is a
working-set hypothesis to distinguish with the directory-only and disabled-pin
cases, not an assertion that exact scoring became computationally slower.

These results exposed serial page faults during routing construction and
metadata copying under `MADV_RANDOM`. The next candidate batches eight footer
requests, bounds dimension-prefix advice to 256 KiB, and maintains rolling
256 KiB read-ahead while sampling. The shared copy-pin helper copies in 128 KiB
chunks with one-chunk lookahead on Linux. No mapping policy is toggled during
queries, and no extra corpus-sized buffer is allocated. These changes require
separate validation and measurements before accepting a performance claim.

The 1 GiB directory-only comparison completed with 703.581 ms approximate mean,
1,194.026 ms p95, 7.361 s cold open and 27.624 s merge. It preserved all run
hashes and both query output sets. Routing was slightly slower on the measured
query mix and much slower to initialize, so its arrays, lookup branch, allocator
and construction code were removed. That intermediate candidate retained compact directory pinning, bounded query
I/O and the shared copy-pin read-ahead fix. The subsequent warm-query screen
rejected query-time advice too; its code and batching helpers were removed.
The retained implementation keeps the original query traversal and OS mapping
policy, compact directory pinning, and the shared copy-pin read-ahead fix.
Directory-only exact mean was still 3,535.785 ms at the 1 GiB cap; the scan
regression therefore is not specific to routing.

#### Retained directory pinning: final validation and paired lifecycle checks

The final implementation removes the query-time random-access policy, explicit
summary/forward prefetch, routing arrays and associated batching helpers. It
retains compact term/row directory views, existing budgeted copy/mlock admission,
disabled-policy accounting, shared Linux copy-pin read-ahead and shared Rust
query defaults. Normal query traversal is restored to the copy-merge control.
Encoded formats, schema defaults, scoring and nomination budgets are unchanged.

The rejected query-I/O screen used six alternating warm trials and identical
outputs: median top-100 mean increased from 23.632 to 37.929 ms. An unmeasured
coalescing follow-up was removed as well. The earlier low-memory improvements
from that candidate are therefore not claims about the retained implementation.

Final measurements use the same 1M/four-source fixture, dedicated Linux/x86-64
machine, Rust 1.98.1, release settings and four search workers as above. The control
already includes the 2.25x copy-merge fix. Warm query and merge trials alternate
before/after three times each, with identical warming and no compiler overlap.

| Final comparison                              |      Control |     Retained |
| --------------------------------------------- | -----------: | -----------: |
| Warm top-100 median mean, 200 queries         |    23.499 ms |    23.712 ms |
| Maximum warm-query process RSS                | 3,406.20 MiB | 3,434.28 MiB |
| Whole-merge median                            |     11.462 s |     11.499 s |
| Maximum merge process RSS                     | 2,419.84 MiB | 2,420.42 MiB |
| 100K fresh build, single trial                |     64.944 s |     64.027 s |
| One bounded 1M maintenance pass, single trial |     28.593 s |     28.347 s |

Warm search uses a 64 MiB copy-pin budget. The control has no Seismic directory
pinning; the retained version copies 27.08 MiB. This is a 0.9% warm-latency cost,
not a query-speed win. Lifecycle comparisons use the unchanged zero pin budget;
the small single-run build/maintenance differences do not establish speedups.
Every merged run matches the canonical encoded input hashes. Build and
maintenance outputs, and their approximate/exact result files, match between
versions. All 200 warm approximate query outputs also match exactly.

The final Linux `full` harness passed **1,995 native tests**, with 24 intentionally
ignored, plus **4 real-server broker tests**, strict lint, native-without-sync,
portable compilation and API docs. The final WASM build and **36 tests** passed.
The local macOS/aarch64 `check` also passed **1,994 tests**. The Linux run validated
a frozen final source snapshot; the local run began during experiment cleanup.
Evidence: cloud `20260918T155554.081439Z-full` and local
`20260918T154949.155655Z-check`. Python/TypeScript suites were not rerun for these
internal reader/copy changes; no protocol or client schema changed in this step.

##### Final enforced-memory probes

Each row below is one fresh Linux cgroup-v2 probe with `MemoryMax` set to the
stated cap and swap disabled. Files are evicted before opening approximate
search, and again before exact search. Approximate results use three passes over
50 queries at 1/4 GiB and 10 queries at 512 MiB; means exclude the first pass.
Exact results use the same three-query sample over three passes. These are
capacity diagnostics, not repeated throughput medians. The 512 MiB sample must
not be compared directly with the 50-query rows or the 200-query warm table.

| Version  |     Cap | Copy-pin budget | Approximate mean |   Exact mean |
| -------- | ------: | --------------: | ---------------: | -----------: |
| Control  |   1 GiB |               0 |       862.130 ms |   249.947 ms |
| Retained |   1 GiB |               0 |       840.497 ms |   250.013 ms |
| Retained |   1 GiB |           8 MiB |       848.047 ms |   254.494 ms |
| Retained |   1 GiB |          64 MiB |       849.197 ms |   251.541 ms |
| Retained |   4 GiB |          64 MiB |        24.278 ms |   246.389 ms |
| Control  | 512 MiB |               0 |     8,630.722 ms | 3,435.391 ms |
| Retained | 512 MiB |          64 MiB |     7,213.452 ms | 2,945.760 ms |

The 8 MiB budget admits term directories; the 64 MiB budget admits term and row
directories (27.08 MiB total). Pinning provides no meaningful speed improvement
in this 1 GiB sample. At 512 MiB the retained implementation measured 16.4% lower
approximate latency than the control, but both are very slow and this single
probe does not establish a general speedup. At an identical 64 MiB pin budget,
the same 50-query sample is about 35x slower at 1 GiB than at 4 GiB. Compact
metadata cannot compensate for the 2.138 GB of evictable summary arrays and
1.011 GB of forward values. No pin, schema or query default is changed.

Restoring the original query I/O policy also restores roughly 250 ms exhaustive
scans at 1 GiB with full directory pinning. Thus the earlier multi-second scan
regression is not an inherent cost of retaining those directory copies; the
removed query-I/O experiment was responsible for that observed tradeoff. The
small exhaustive sample remains a scan/functional check, not evidence that exact
search is generally faster than approximate search.

The final 512 MiB / 64 MiB-pin case additionally performs a cold-source merge:
**30.833 s**, with all copied-run hashes preserved and the same 3,537,446,009-byte
live output. This is separate from the warm merge medians above. The benchmark
records process RSS, major faults, filesystem input blocks and cache-inclusive
cgroup memory. Process I/O counters include cold opening and all three query
passes, not only the reported warmed query means.

All approximate/exact query probes completed and match the control. Every
cgroup phase recorded zero OOM events and kills. The 1 GiB and 512 MiB cases
reached their enforced cache-inclusive caps; the final 4 GiB search probe peaked
at 3,614,760,960 bytes (3.37 GiB). Tested cloud Rust sources match the local
workspace, and the downloaded evidence archive SHA-256 was verified.
Construction still has no hard peak-memory cap; these results establish behavior for existing
index reading, merging and search, not constrained-memory initial ingestion.
All scripts, source/binary fingerprints, payload hashes, logs and complete query
outputs are retained locally under `.context/seismic-copy-merge/pinning-evidence/`.
Clean `npm ci` and a second **36-test WASM run** passed after benchmarking.

##### Compact-summary implementation and measurements

The lossless directory codec is implemented in the working tree. It encodes
128-entry monotone blocks using local bit packing or Elias–Fano, with an
array-level fallback to the previous packing when smaller. UInt8 weights,
quantizers, clustering, nomination order and query defaults are unchanged.
The [compact-summary design and measurements](seismic-compact-summaries.md)
describe the wire format and complete benchmark methodology. Seismic format 4
is incompatible with format 3: existing indexes require rebuilding. Both reader
versions explicitly reject the other's format; no live migration is provided.

The matched 1M/four-source Linux fixture uses Rust 1.98.1, release settings,
four workers and a 64 MiB copy-pin allowance. Warm measurements are medians of
three alternating runs per version, each with 200 queries/top-100 and three
passes, excluding the first pass. Memory-cap rows are single capacity probes
with swap disabled, using 50 queries at 1 GiB and ten at 512 MiB.

| Final comparison                      |        Before |       Compact |
| ------------------------------------- | ------------: | ------------: |
| Summary-array bytes                   | 2,138,175,719 | 1,414,544,853 |
| Complete live-index bytes             | 3,537,446,009 | 2,814,694,103 |
| Linux warm mean                       |     22.419 ms |     20.201 ms |
| Linux warm p95                        |     36.538 ms |     32.489 ms |
| Linux warm process RSS                |     3.353 GiB |     2.681 GiB |
| Linux index open                      |       2.892 s |       3.273 s |
| 1 GiB approximate mean                |    695.558 ms |    650.968 ms |
| 512 MiB approximate mean              |  7,063.370 ms |  6,524.871 ms |
| Copy-merge median, two trials/version |      11.776 s |      10.707 s |
| Maximum copy-merge RSS                | 2,509,152 KiB | 1,806,620 KiB |

Directory bytes fall 67.7%, summary arrays 33.8%, and the complete index 20.4%.
All 300,453,433 decoded directory entries match; hashes preserve 2,459,230,762
bytes across 219,744 untouched payload sections. Encoded source runs remain
byte-identical through copy merge. One fresh 100K build per version has similar
time and peak RSS, with a 23.8% smaller index and identical untouched payloads,
50 approximate query outputs and five exact query outputs. Construction still
has no hard memory cap.

The latency tradeoff is mixed. Linux warm means improve 9.9%, but open takes
13.2% longer. An initial 33.3-second open regression was fixed by sequential
block admission instead of repeated random Elias–Fano selection. The 1 GiB
mean improves 6.4%, with p95 slightly higher; at 512 MiB the mean improves 7.6%
but filesystem input drops only 2.6%. The separate 4 GiB probe is 3.2% slower.
All query probes preserve results and record zero OOM events/kills. I/O counters
include cold opening and all passes, rather than only warmed-query timing.
Compression does not eliminate the cache-thrashing cliff.

On the Apple M4 shared workstation, the same paired warm procedure gives
11.789 → 12.107 ms (+2.7%), RSS 3.016 → 2.343 GiB, and open 1.117 → 1.786 s.
One compact run has a large scheduling tail; it remains in the reported median.
Do not claim an ARM or architecture-independent speedup. The fixture has one
vector/document: 150M documents with 2–3B vectors, concurrent production QPS,
64/256-entry alternatives, payload colocation and UInt4 remain unmeasured or
unimplemented. No production RAM minimum follows from these file sizes.

Final validation passes **1,997 native tests** (24 intentionally ignored), strict
lint, native-without-sync, the WASM build and **36 WASM tests**. Evidence is
`.context/search-harness/20260918T173418.407024Z-check/` and
`.context/seismic-compact/`. The `full` lifecycle/RPC harness was not repeated
for this codec-only change; its previous pass is recorded above. No clients or
wire RPC schema changed. All 285 cloud source hashes match the local core
sources. Downloaded cloud logs, results, scripts, payload audits, compatibility
checks and executable/source fingerprints are retained in
`.context/seismic-compact/cloud-evidence/`. The verified evidence archive SHA-256
is `e27ba8433b1de8743461bcc8cd158f7b75ea22fec740397fcf82a9258719f91d`.
The benchmark machine was stopped after download and independently verified as
`TERMINATED`; a connection reset interrupted the stop command's status polling,
not the completed shutdown. Final documentation, ownership contracts and
`git diff --check` pass.

##### Cluster-ID compression and locality follow-up

The version-5 experiment preserves decoded summary cluster IDs, codes,
quantizers, nomination rows and forward vectors. Per-coordinate sorted ID sets
use combinatorial ranks when their complete encoded size beats fixed-width
packing. Every 128 coordinates has one U32 byte checkpoint. Terms with more
than 64 clusters or no size saving retain fixed-width IDs. Ordinary block delta
packing was screened out: it saved no complete terms after restart overhead on
this fixture. See the [codec design](seismic-compact-summaries.md) for wire layout,
writer bounds and benchmark evidence.

The 1M/four-source fixture has 650,174,079 occurrences. ID bytes shrink
418,818,287 → 332,618,612 (20.6%), summary arrays
1,414,544,853 → 1,328,345,178 (6.1%), and the complete index
2,814,694,103 → 2,728,494,428 (3.1%). Compression is selected for 54,648 terms.
The offline converter uses the production encoder and compares every decoded
ID/code; dimensions, ends, quantizers, nominations and forward bytes are copied.
The locality variant places each group's weight codes next to its ranks, with
exactly the same size as the separate-stream variant. It does not colocate the
dimension/end directories, quantizers or exact vectors.

At the user's request, normal summary reading now trusts immutable writer
output: no eager payload scans, term format/length checks or per-query ID/rank
validation. Writer/codec tests establish the invariant. The previous eager
prototype's timings are superseded and retained under `eager/`. A version-4
control was rebuilt with the identical no-scan/read policy, separating its
benefit from compression/locality. Existing outer-envelope compatibility and
directory ownership stay separate from summary contents. Version 5 requires
rebuilding older indexes; no live migration is provided.

Final warm runs use the same fixture, compiler, flags and four workers within
each architecture, a 64 MiB copy-pin allowance, 200 queries/top-100, three passes
with the first excluded, and three alternating trials per variant. Values are
medians of per-run statistics; no compiler overlap occurred.

| Warm query mean | Fixed IDs, no scans | Compressed, separate codes | Compressed, colocated codes |
| --------------- | ------------------: | -------------------------: | --------------------------: |
| Linux/x86-64    |           24.877 ms |                  25.186 ms |                   22.303 ms |
| Apple M4        |           12.200 ms |                  11.968 ms |                   12.085 ms |
| Linux open      |           26.362 ms |                  25.774 ms |                   25.438 ms |
| Apple M4 open   |           14.026 ms |                  12.814 ms |                   12.504 ms |

Removing eager scans lowers opening from seconds to milliseconds. Compression
alone has approximately neutral warm latency on these fixtures. Colocation
improves the Linux warm mean by 10.3% versus fixed IDs and 11.4% versus separated
compressed IDs; it has no meaningful warm advantage on Apple M4. Linux median
process RSS is 2,014,220 / 2,001,976 / 2,725,620 KiB for the three layouts:
colocation touches more resident mapped pages despite identical encoded size.
Apple M4 RSS is 1,453,293,568 / 1,438,220,288 / 1,436,860,416 bytes. These are
process residency observations, not minimum-memory requirements.

The single 50-query warm top-10 probes are 13.326 / 13.676 / 12.015 ms on Linux
and 6.360 / 6.466 / 6.917 ms on M4. They use a smaller query sample than the
200-query table and do not establish concurrent production QPS. All sampled
approximate and exact results match across layouts.

At a 1 GiB cgroup-v2 cap with swap disabled, the matched 50-query top-100
means are 667.399 / 642.730 / 628.797 ms (fixed/separate/colocated); p95 is
1,514.200 / 1,291.499 / 1,286.291 ms. Filesystem input is
11,451,456 / 11,111,880 / 11,119,832 blocks of 512 bytes, including cold opening,
all three passes and result collection. Colocation therefore does not
materially reduce measured disk volume versus separated compressed IDs, despite
its warm Linux latency improvement. File refaults are
1,116,176 / 1,075,761 / 1,076,610. These are single capacity probes, not repeated
throughput medians or per-component I/O attribution.

With the same cap and 50-query sample, requesting top-10 gives
492.745 / 468.405 / 462.614 ms. Returning fewer results helps, but still requires
summary discovery and candidate scoring; it does not make the read workload ten
small summary fetches. Compression/locality do not establish a low-memory
production configuration for 150M documents with 2–3B vectors. The fixture
has one vector per document and no concurrent production QPS is measured.

At 512 MiB, the ten-query top-100 probes average
6,093.037 / 6,469.505 / 6,506.735 ms; p95 is
10,523.400 / 11,520.801 / 11,307.942 ms. Filesystem input is
61,112,800 / 60,943,920 / 61,028,480 blocks, and file refaults are
7,367,733 / 7,347,504 / 7,358,068. Compression is 6.2% slower in this single
probe, while disk volume changes by less than 0.3%; colocation adds no useful
reduction. Do not present compression as a universal latency win. Every final
memory-cap case completed with zero OOM events and kills.

The retained writer uses separated compressed IDs/codes. Colocation remains an
experimental encoded layout: its Linux warm improvement does not establish a
broad memory-pressure or cross-architecture advantage. Summary format/length
checks and eager payload scans remain removed, as requested.

Two copy-merge trials per layout on the four-source 1M fixture give median
3.899 / 3.859 / 3.503 seconds, with maximum process RSS
47,840 / 64,816 / 64,896 KiB. Every encoded source run remains byte-identical
through merge. The separate-layout trials vary 3.490–4.228 seconds, so their
small median advantage is not a throughput claim. These low-residency results
use zero pin budget and the writer-trusting reader, unlike the earlier eager
admission measurements; the improvement is not solely ID compression.

A single fresh 100K build per version takes 64.960 → 66.103 seconds, with peak
RSS 648,420 → 652,628 KiB and index bytes 356,739,945 → 346,370,859 (2.9% smaller).
The candidate records 184 bounded queue retries versus zero for the control.
Converting the control through the production occurrence encoder yields exactly
the candidate's encoded sparse component bytes. All 50 approximate and five
exact fresh-build query outputs match. Construction still has no hard
peak-memory guarantee.

Final checks pass **1,997 native tests** (24 intentionally ignored), strict
lint/format, native-without-sync, the WASM build and **36 WASM tests**. Evidence:
`.context/search-harness/20260918T184035.497915Z-check/` and
`.context/seismic-occurrences/`. The `full` lifecycle/RPC harness was not repeated;
no RPC/client schema or publication protocol changed. Existing native lifecycle
tests and the measured build/copy-merge paths cover this reader/codec change.

Cloud evidence was downloaded and extracted under
`.context/seismic-occurrences/cloud-evidence/`. The archive SHA-256 is
`f4d7916e8ec4b180c5e623c58d161cea484c26ba7a2d6fa2d673a3a3ce9a31af`;
all 287 recorded source hashes match the local tested core sources.
The benchmark machine `benchmark-host` was stopped and independently
confirmed `TERMINATED` after the download.

## Release review: module ownership and shared implementations

The release review covers the complete branch: text scoring and formats,
RGB reordering, sparse backends and Seismic lifecycle, binary ANN exact-vector
ownership, copy I/O, schema/RPC/client adapters, diagnostics and benchmark tools.
The refactoring invariant is unchanged encoded bytes, scoring arithmetic,
query budgets and publication behavior. No summary-reader validation is added.

Seismic query preparation/exact scoring and complete-membership scoring move
into private modules; nomination remains in the executor. Unit and integration
tests keep their existing module paths. Sparse term decomposition and in-memory
backend dispatch share one implementation across entry points. Bloom hashing
and filesystem streaming-writer construction also have one owner. These
extractions do not add corpus-sized allocations or change the cost model.
Raw benchmark archives remain local evidence, outside the source release.

The benchmark adapter still accepted the removed posting-validation-cache option
and silently ignored its value. A regression test reproduced that behavior;
the obsolete option and help entry were removed, so ordinary unknown-option
handling reports it. Historical measurements remain labeled as historical.

Changing the sparse constructor default exposed a persisted-schema compatibility
bug: older MaxScore metadata omitted `format`, so it was reinterpreted as BMP
and failed opening its existing payload. The serialized field retains the
historical MaxScore omission default, while new writes emit the backend
explicitly. New SDL/programmatic schemas still default to BMP. The regression
opens format-6/7/8 metadata, appends, copy-merges, reopens, and compares results;
no payload migration or new reader validation is introduced.

Current-main integration keeps bounded segment opening and deletion-only reader
refreshes. The shared posting reader now clones its file handles and immutable
integrity state; it does not reopen or scan payloads. Seismic regressions cover
old/new visibility in native and async search, shared copied directory addresses,
and query/merge correctness after the original reader drops. Upstream ANN pin
ownership and singleton-upsert regressions are retained.

Final integrated validation passes `python3 scripts/check_search.py full`:
2,007 native tests (25 intentionally ignored in the ordinary run), five separately
run real-server broker tests, strict Clippy/format, native-without-sync and
portable builds, and API docs. A fresh WASM release build passes all 36 tests.
Python passes all 28 client tests, including real-server BMP/Seismic maintenance,
plus 16 stress-helper tests; TypeScript passes 17 tests. Regenerated Python and
TypeScript bindings match byte-for-byte. Python lint/format, shell checks,
documentation links/benchmark inventory and their checker tests also pass.

The first local fully parallel broker suite timed out in three mock-index
registration waits. All 13 tests passed serially; the full harness then passed
with `RUST_TEST_THREADS=4`. Normal Linux CI concurrency remains unchanged. This
local contention limitation is retained in the evidence rather than hidden by a
timeout increase. Final local evidence is under
`.context/search-harness/20260918T200126.221893Z-full/` and
`.context/release-review/`. No new throughput claim is made for the module cleanup.

Linux CI exposed a missing feature boundary: the standalone broker disables core
writers, but Seismic encoding helpers were still compiled and failed the strict
unused-code check. Writer-only modules, imports, and functions now use the same
native/WASM/test gates as their callers. No writer bytes or reader behavior
change. The local harness now checks the broker independently and treats compiler
warnings as errors, matching CI rather than relying on workspace feature unification.

Remaining repository dependency alerts are recorded in PR #191. Existing `lru`
and frontend/test/build-tooling advisories are outside this search-feature review;
the critical GitPython alert references a removed training lockfile. A passing
Cargo Audit job is not a claim that all repository dependency alerts are resolved.

## Seismic forward dimensions: DotVByte and U24 (2026-09-19)

The [forward-index paper](https://arxiv.org/pdf/2602.05445) applies to Seismic's
exact candidate-scoring owner. Implemented opt-in lossless dimension compression
in `structures/postings/sparse/dimensions.rs`, shared by forward iteration, scoring,
retrieval, and maintenance. Existing configured weight bytes and scorer remain
unchanged. The codec uses eight one/two-byte gaps per control byte, a U32 base
and prefix sums, independent row alignment, and raw U32 tails. ARM NEON and
x86 SSSE3 decode groups; WASM uses the portable decoder. Raw U16/U24/U32 fallbacks
cover short vectors and large gaps, including 100k vocabularies. The row tag
uses existing reserved directory space; persistent directory size stays 24 B.

Envelope version 6 rejects earlier versions, as requested. Copy merge preserves
encoded runs; compaction copies each surviving row's bytes and tag. No summary
or forward payload scan was added. Enable with
`seismic_forward_compression: true`; it was initially opt-in and is now the
default at the user's request (see the follow-up below). This is a physical
storage setting: schema, query, and returned dimension IDs retain U32 semantics.

The first prototype accidentally outlined the generic dimension fold. Assembly
confirmed a weight-decoder call per coordinate, and default raw exhaustive
search regressed. Explicitly inlining that fold restores constant precision
inside the shared hot loop; a single coordinate cursor also drives both
dimension and weight accesses. A compile-time raw specialization of the same
iterator keeps compressed-format dispatch out of the raw row-scoring loop.
The final comparison includes the unchanged main binary as well as
raw/compressed layouts in the candidate binary.

Validation: the full nine-step harness passed 2013 native tests (25 existing
ignored tests), strict Clippy, feature combinations, API docs, server build, and
five real-server tests. The final focused harness
(`20260919T055521.799404Z-check`) passed the same 2013 tests. Codec tests also passed as x86 binaries under
Rosetta; Linux query/codec tests and the rebuilt WASM suite provide independent
native-x86 and portable coverage. The final rebuilt WASM suite passes all 37
tests. Remaining limitations: no RGB dimension
permutation/global mapping, no separate packed SIMD weight scorer, and no measured claim for the user's actual
100k-token embedding model. A shifted real-vector sample exercises wide IDs
while preserving its original gap distribution.

Final measurements and reproduction artifacts are linked from
[the forward codec design](seismic-forward-compression.md) and
[the benchmark instructions](benchmark-results/seismic-forward/README.md).

On the 1M Float32 fixture, adaptive gaps save 64.8% of dimension bytes,
32.4% of forward payload, and 12.0% of the complete index. Warm top-100 means
are 27.99 ms raw, 31.50 ms U24, and 32.79 ms adaptive, with peak RSS 2.00,
1.89, and 1.70 GiB respectively. Original main averages 29.85 ms. Exhaustive
candidate scoring is 259.31 ms raw versus 378.16 ms adaptive, so this is a
measured CPU/storage tradeoff, not a blanket speed improvement.

Two reverse-order trials under a 1 GiB cgroup reproduce a cache threshold for
20 repeated queries: raw averages 984.41 ms, U24 35.77 ms, and adaptive gaps
33.97 ms. All IDs and scores match. Read traffic averages 2.44/0.96/0.77 GiB
respectively over all three passes. The compressed working set fits this
particular budget; this does not establish production RAM requirements or a
universal speedup. The shifted wide-ID sample averages 10.23/10.46/10.72 ms.
See [the measurements and limits](benchmark-results/seismic-forward/2026-09-19/README.md)
for full distributions, cgroup counters, environment, hashes, and validation.

The benchmark machine is confirmed `TERMINATED` after the evidence export and local
checksum/source verification. No benchmark process or cloud machine was left running.

### Default selection and BMP applicability

At the user's request, Seismic forward compression now defaults to true in
constructors, omitted serde fields and SDL. Explicit false is preserved through
JSON and server SDL round trips. This selects the existing measured codec; it
does not change score precision, the version-6 format, or already-written rows.
The benchmark snapshot above predates this default-only policy change; its
source manifest remains historical evidence, not a hash of the later config.
Cross-architecture latency and genuine 100k-token model results remain unmeasured.

The initial BMP investigation (superseded by the implementation below) found a
reusable dimension-codec opportunity: its forward
entries still use U32 dimensions plus U8 impacts, independently of `index_size`.
L1 backfill and BP consume them, while normal search remains inverted. At 126
entries, a proposed U24/count layout saves 18.9% including the existing logical
directory; gap savings require a BMP retained-row benchmark. See the
[current format and ownership](bmp-forward-index.md).
BMP storage and query execution were unchanged at that stage.

The default-on follow-up passed the focused harness
(`20260919T073653.759718Z-check`): 2015 native tests, 25 existing ignored tests,
strict Clippy and both feature checks. The rebuilt WASM suite passed all 37
tests, including default-compressed versus explicitly raw wide-ID fields.
Documentation links and formatting pass. No new BMP runtime or compression
benchmark was run; the BMP numbers above are layout estimates.

### BMPB gap packets and production distributions (2026-09-19)

Implemented bounded 128-entry forward packets using the same adaptive
U16/U24/U32/DotVByte dimension codec as Seismic. The codec now belongs to
`structures/postings/sparse/dimensions.rs`; Seismic's existing version-6 bytes
are unchanged. A single BMP row writer serves ingestion and materialization.
BMPB is an incompatible envelope requiring rebuilds; no BMPA runtime reader is
kept. Native copy merge, both BP modes, and deletion compaction preserve encoded
forward rows. U8 impacts, ordinals, duplicate dimensions and integer scoring
remain unchanged. Query scoring uses the writer-trusted view once; explicit
integrity/BP validation remains outside query loops.

Read-only sampling of three physical production shards found 101.49M documents
and 2.338B sparse vectors with a 105,879-dimension vocabulary. Exact retained
mean NNZ is 160.75 for passage vectors and 150.14 for short-document vectors.
Eight stratified windows per field/segment supplied 9,472 rows, requesting only
7.45 MB of metadata and payload. On those identical retained entries, BMPA
payload is 7,279,190 bytes and BMPB payload 4,024,684 bytes: 44.71% less, or
43.80% less including the unchanged directory. Segment-weighted extrapolation
estimates 846 GB less forward payload across the three shards (1.874 TB to
1.028 TB), reducing total sparse blobs about 21.7%. This is estimated disk
storage, not a resident-RAM requirement or measured production latency gain.

Same-fixture ARM64 resident-row scoring measured medians 164.66 ns raw and
263.84 ns compressed (1.60× CPU). A packet-level fused fold replaced the first
nested-iterator scoring loop, reducing its roughly 420 ns cost without changing
score units. The packet writer is 1,040 bytes plus a few KiB bounded temporary
scratch; decode needs eight U32 lanes and no candidate-sized allocation.
All sampled tuples and integer score checksums agree with the raw-row oracle.
Ordinary BMP retrieval remains inverted. The later
[query-latency follow-up](benchmark-results/bmp-forward/2026-09-19/query-latency.md)
measures public core L1 on x86 under warm and constrained memory. Full BP
build/rewrite timing, production-server latency and concurrent QPS remain unmeasured.

Additional read-only inspection covered 596 inverted BMP blocks and fast-field
metadata. Dimension arrays comprise about one third of sampled inverted block
bytes; U24 could reduce total sampled block bytes about 8.3% while preserving
indexed access. Gaps are smaller still but need a separate random-access design.
Language-column blocks covering 100.4M rows use 64-bit values despite small
dictionaries, exposing missing-sentinel width inflation; a validity bitmap is a
concrete follow-up experiment. Fast fields total only 5.35 GB, so prioritize the
much larger sparse payload. No other field format changed on the basis of
metadata alone. See the [measurement report](benchmark-results/bmp-forward/2026-09-19/README.md)
for raw aggregates, methodology, source/input provenance and limits.

Validation: the storage regression failed on BMPA and passed on BMPB. New
coverage exercises full-width IDs, duplicate dimensions, packet boundaries and
long rows, folded/iterated equality, writer failure, encoded-byte preservation
through merge/materialization/compaction, and rejection of BMPA. Existing BP,
missing-value, cancellation, score and lifecycle regressions are retained.
The first focused harness passed native tests but exposed featureless broker
build warnings after moving maintenance-only APIs; those APIs/imports now have
matching feature gates. Final full-harness and WASM results are recorded below.

Further inspection found 762.227 GB of duplicate flat binary vector sections
alongside 762.318 GB of IVF sections in the live files. No exact-location sections
were present. The repository already owns an ANN-code-plus-location-table writer;
using it on rebuilt generations suggests about 729 GB additional savings after
14-byte lookup rows, before small span/block metadata. This capability predates
this gap-encoding patch and was not deployed here. Bounded fast-value sampling
also confirmed 37/3,072 missing language ordinals among present IDs 0–39;
rare missing sentinels explain full-width blocks. Aggregate evidence is in the
same report; no extra runtime format changes were added for these findings.

Final validation: `20260919T080450.635887Z-full` passed formatting, strict Clippy,
2021 regular native tests (25 normally ignored), native-without-sync and broker
feature checks, featureless core, strict documentation, and the server build.
The parallel real-server stage passed four tests and hit `Address already in
use (os error 48)` during the fifth server's startup. Rerunning the entire
five-test integration suite with `--test-threads=1` passed all five in 5.21s
(`/tmp/summa-bmp-gap-e2e-serial.log`); no test assertion or timeout was weakened.
The WASM release build and all 38 JavaScript tests passed, including the new
wide-ID BMP packet/tail fixture. Moving the Seismic codec was additionally
byte-compared with the archived pre-extraction encoder for 9,472 production
rows in both modes and 1,932 boundary cases; every tag/payload matched.
`git diff --check` and documentation ownership checks pass. Production files
were only read. The later query-latency follow-up adds controlled x86 BMP
candidate measurements; full production and BP benchmarks remain future work.

## BMP packet forward values: measured query latency (2026-09-19)

The [query-latency report](benchmark-results/bmp-forward/2026-09-19/query-latency.md)
compares the public core retrieval/candidate APIs on one segment built from
900,000 genuine retained vectors (42,596 documents, 105,879 dimensions). A raw
comparison fixture copies the inverted prefix exactly. Both readers trust
writer-produced values. No production workload was available, so the benchmark
uses deterministic 16-term query templates and forces same-field all-passage
backfill; it does not model a production ranking plan or RPC latency.

On the common 64 queries, retrieval k=1,000 plus L1 has these mean/p95 times:

| Cache condition |         Raw mean / p95 |     Packets mean / p95 |
| --------------- | ---------------------: | ---------------------: |
| Warm            |       53.22 / 81.76 ms |     143.38 / 213.17 ms |
| 1 GiB           | 1,559.65 / 4,704.47 ms |     145.17 / 217.65 ms |
| 512 MiB         | 5,221.43 / 8,936.47 ms | 3,056.54 / 4,848.53 ms |

Forward payload shrinks 45.05%; full sparse bytes shrink 20.03%. The compressed
pipeline peaks at 892 MiB under the 1 GiB limit, while raw reaches the limit and
continues faulting. At 512 MiB both remain I/O-bound. Retrieval alone stays near
8 ms. The exact mean expansion is 476.09 unique documents to 64,653.36 vectors;
the largest query scores 124,040 vectors before retaining at most ten documents.

Remaining findings: warm backfill increases from 26.56 to 116.78 ms. A short CPU
profile and annotated assembly put the hotspot inside candidate scoring,
including repeated stack copies around the fallible fold accumulator. A smaller
primitive accumulator was a concrete experiment at that point; the follow-up
below measures the implemented fix. Forward
bytes also use `MADV_RANDOM`, with no explicit selected-range prefetch after
budget admission. At 512 MiB, nearby rather than scattered 1,000-document pools
reduce raw forward-only repeat latency only from 1,826.26 to 1,687.91 ms. These
pools differ in membership; page/vector footprints accompany the results.
Bounded coalesced forward prefetch is the next I/O experiment. Neither runtime
optimization was added during this measurement.

All 52 runs / 9,856 requests completed; all 26 format pairs match every returned
ID and score bit. Offline checking matched all 900,000 vectors / 146,460,740
entries against the source and byte-compared the 900,363,945-byte unchanged
inverted prefix. No pressure run recorded an OOM. Early 1 GiB I/O counters were
unavailable and remain null; the checked-in driver now enables I/O accounting
up front. The exact measured driver, binaries' hashes, timings, profiles and
input provenance are archived. The evidence was downloaded and hash-verified,
and the machine is confirmed stopped. This follow-up changes benchmark tooling and
documentation; runtime validation remains the preceding full native/WASM run.

## BMP rescoring accumulator optimization (2026-09-19)

The [accumulator follow-up](benchmark-results/bmp-forward/2026-09-19/accumulator-optimization.md)
replaces the full error-carrying fold state with `Option<u32>` and constructs the
existing error only at the row boundary. Checked integer arithmetic, duplicate
matches, exact score bits and serialized index bytes are preserved. The change
belongs solely to the shared query scorer; there is no new codec, allocation,
validation pass, cache policy or default.

On the same 900,000-vector fixture and Cascade Lake machine, paired warm k=1,000
pipeline backfill falls from 116.88 to 45.97 ms (2.54× faster). Total mean/p95
falls from 143.21/214.48 to 72.07/106.65 ms. Fixed 1,000-document scoring falls
from 38.94 to 16.06 ms; retrieval alone remains about 7.6 ms. Peak warm pipeline
RSS is effectively unchanged at 1,149 MiB. These are synthetic query templates
through the public core APIs, not production traffic or a concurrent QPS claim.

Validation: the full serial `check` harness passes 2,023 native tests (25 normally
ignored), strict Clippy and feature checks; WASM builds and passes all 38 tests.
The initial parallel harness hit a broker discovery timeout; the complete serial
rerun passed without weakening assertions. New regressions cover duplicate
matches across packets/encodings, no matches, the exact u32 limit, and sticky
overflow. Capped-memory measurements and profiles are recorded in the linked
report. Selected-range prefetch remains an unimplemented I/O follow-up.

All 24 before/after runs (4,096 timed requests) matched every result ID/score bit
across 12 pairs; index-file hashes were unchanged. The 1 GiB pipeline improves
from 144.72 to 73.62 ms at effectively unchanged 892 MiB peak cgroup memory;
all eight capped runs have zero OOM events. The annotated optimized loop removes
the large error-state copies; the hottest sampled instruction is now in SIMD
gap unpacking. Full evidence was downloaded and hash-verified, and the isolated
machine is confirmed `TERMINATED`.

Final pre-merge review: traced SDL/schema conversion, the shared dimension codec,
BMP and Seismic writers/readers, candidate scorers, copy merge, compaction and
WASM. No blocking correctness or duplicated runtime-codec findings remain.
Corrected a stale encoded-view comment that implied prefetch was implemented;
public benchmark metadata now redacts the internal host name while the private
archive retains exact provenance. The latest main cleanup changes comments and
benchmark labels only. The passing 2,023-test native harness, 38 WASM tests and
paired score/byte measurements above cover the unchanged runtime implementation.

### Searchbench restart and HTTP adapter — September 22, 2026

The authorized GCP benchmark machine is running; the verified 10M corpus is ready and the
four-engine campaign (including Luxir 0.1.0) started at 19:23 UTC. The native benchmark HTTP adapter uses canonical
writers/searchers/count collection and fast-column IDs. Native end-to-end smoke
passed on macOS and Linux. The repository check passed on a serial rerun after
three broker discovery timeouts in its first concurrent run; no broker code changed.

The 100k corpus probe found 162/826 queries with exact count agreement across
Summa/Elasticsearch/OpenSearch. The references agreed on all 826. Summa lexical
analysis differs from Lucene standard analysis, including punctuation and
contractions; even `the` matched 90,785 versus 90,758 documents. This coverage
limit must accompany any later QPS table. No full-corpus comparison numbers are
available yet. See [the campaign design and evidence](searchbench-comparison.md).

### Completed Searchbench evidence — September 23, 2026

All four engines completed 27 cells each without request errors, but only 15 of
826 queries have equal full-corpus counts. All three references agree on every
query. Summa has 216 errors (including unsupported operators) and 595 count
mismatches. Analyzer compatibility remains a prerequisite for broad claims.
At eight clients, Summa top-10 conjunction throughput is 9,253 QPS versus Luxir
10,848, Elasticsearch 5,049 and OpenSearch 5,742. Its phrase ranking trails all
references: 470 QPS for seven low-phrase queries and 189 for the single medium
phrase. Investigate phrase candidate/position work separately; no runtime changes
or default changes have been made from this experiment. See the
[full result table](benchmark-results/searchbench-2026-09-22.md).

## Non-RGB phrase follow-up (2026-09-23)

The retained changes move conservative score admission before expensive
conjunction/position work, batch posting intersections, reuse per-threshold
TF/length cutoffs, and streamline cached position reads and two-term exact
matching. Exact phrases may use the rarest term's TF bound only when the writer
certifies unique positions for the original first term. Sloppy phrases and
duplicate starts retain their existing multiplicity and conservative first-term
bound. The certificate costs one existing footer bit; it adds no payload bytes.

Position streams now use POS5/POS6. Old readers, whole-list migration branches,
unused position-list codecs and the obsolete codec benchmark were removed.
Old position formats require rebuilding. Copying merge combines certificates
with AND; doc-aligned deletion compaction and reordering preserve them. The
native-written WASM fixtures were rebuilt through the public writer, retaining
their corpus hashes and exact counts. Impact envelopes remain opt-in.

The native harness passed 2,021 tests, strict Clippy, native-without-sync and
standalone broker checks in `.context/search-harness/20260923T064244.392846Z-check`.
The WASM build and all 38 browser tests passed. Tests cover exact IDs/score bits,
position duplicates, offsets, zero/nonzero slop, cancellation, copied payload
bytes, reordering, compaction and rejection of obsolete position formats.
Full production-RPC testing was not run; no RPC or publication protocol changed.

The [final paired run](benchmark-results/searchbench-2026-09-23-phrases.md)
measured low-phrase top-10 at 2,605 QPS versus 677 for first-term admission and
1,510 for fresh Luxir. Medium-phrase top-10 reached 2,129 versus 1,099 and 3,524;
the 39.6% gap remains. Summa top-100 beat Luxir in both phrase families, while
phrase counting remained about 20–21% behind. Exact IDs, score bits and counts
matched between binaries on the same index. Query RSS remained about 1,137 MiB.
The current-format RGB build passed its separate exhaustive top-100 smoke audit.
Profiles identify candidate scanning/advancement (49.7% combined self samples)
as the main remaining medium-phrase top-10 cost; position range/read/matching
account for 55.9% of count samples. These profiles cover one query, not the
complete benchmark. Lazy cutoff construction
and block-max-one TF scanning were rejected because they regressed the common
phrase; no experimental assumption of unique positions survives in production.
Analyzer/count mismatches outside the 15-query agreement subset remain open.

## Phrase scan follow-up and 32-vCPU campaign (2026-09-23)

The [fresh eight-client paired comparison](benchmark-results/searchbench-2026-09-23-gap.md)
retains singleton/tie-aware admission, verified inverse-seeded length cutoffs,
AVX2/AVX-512 TF/norm scans, existing L1 group bounds, cost-aware intersections,
and bounded position-offset/membership caches. Encoded index bytes and defaults
are unchanged in this follow-up. The complete query screen supports the wider
x86 kernel; scalar and native/async semantics remain covered by tests.

On the same ordinary index, medium-phrase top-10 rises 2,121 → 3,130 QPS,
top-100 1,316 → 2,012, and count 99 → 138. Low-phrase top-10 rises
2,538 → 3,081 and count 321 → 383; top-100 regresses 1.3%. Conjunction count
regresses 3.2%. Fresh Luxir remains ahead for medium top-10 (3,540) and low
count (413). Optional impacts reach 6,352 medium top-10 QPS but regress low
phrase top-10/top-100 relative to ordinary Summa. Impacts stay disabled by default.
Only 15/826 queries qualify; no broad engine-parity conclusion is justified.

All 36 timing cells have zero request errors. Same-index before/after IDs,
score bits and counts agree on 45 HTTP responses and exhaustive top-100 audits.
RSS changes from 1,137 to 1,140 MiB. These are warm-cache process RSS figures,
not heap-only or cluster memory requirements. The software user-time profiles
still put candidate scanning/decoding at the center of medium top-10 cost and
posting seek/intersection at the center of the slowest low-phrase count case.
The linked report preserves regressions and sampling limitations.

The frozen phrase source passed the native harness at
`.context/search-harness/20260923T091012.641296Z-check`, and all 26 phrase tests
passed on the x86 host, exercising both vector kernels. Initial perf permission
failure was recovered by a temporary host setting change; prior settings were
restored. The archive was downloaded and SHA-256 verified before stopping the
8-vCPU machine; independent cloud status confirms `TERMINATED`.

The completed [32-client, 32-vCPU report](benchmark-results/searchbench-2026-09-23-32cpu.md)
uses 30 server hardware threads on 15 physical cores with SMT and reserves the
remaining physical core (two SMT threads) for replay. Prior/current Summa, RGB,
Elasticsearch, OpenSearch and Luxir run sequentially, followed by an optional
impacts variant. All 63 timing cells have zero errors. Ordinary medium-phrase
top-10 improves 10,248 → 14,896 QPS, still below Luxir's 17,580. Optional impacts
reach 27,085 but regress low-phrase top-100 relative to ordinary Summa. Impacts
remain off by default. RGB stays separate and has substantial regressions as well
as gains. Only 15/826 corpus-count-compatible queries are measured.

Before/after exact IDs, score bits and counts agree with exhaustive top-100;
RGB/impact builds preserve counts and ranked score bits with independent tie IDs.
Ordinary optimized peak process RSS is 1,284 MiB versus 1,283 before. The report
records all reference/memory results, CPU placement and the overlapping-client
caveat of the separate health control. Search measurements use disjoint CPUs.
An isolated-hostname lookup failure was fixed before any reference timing, then
references resumed in a fresh namespace with identical network mode and CPU
policy. Impacts use a third namespace after transferring the existing index.
No copying, indexing or compilation overlaps query timing.

## Wildcard term filters (separate from the timed binaries)

The [wildcard query](wildcard-query.md) adds a core `WildcardQuery` with whole-term
Unicode `*`, `?` and backslash escaping, plus named and bare-pattern QL forms.
Simple trailing-star patterns retain `PrefixQuery`. Other patterns no longer
silently split into prefix/term clauses. The benchmark adapter delegates its
three wildcard families to this core query; Lucene regex remains unsupported.
The general production protobuf interface has no dedicated wildcard variant yet.

Dictionary filtering remains in the SSTable owner, field-key expansion and posting
reads in the segment reader, and shared constant-score union/scoring in query.
Literal prefixes restrict dictionary scanning. Existing prefix term/posting limits
remain, with separate wildcard pattern/compiler/scan bounds. No new format,
second scorer, document scan, or index migration is introduced. Full 826-query
comparison still needs analyzer compatibility/rebuilt indexes, sloppy-phrase
semantics, regex and escaped-literal handling, and broader bounded expansion.

An end-to-end regression reproduced an existing prefix RGB defect: expanded
term filters returned physical rather than logical IDs. Shared union and prefix
bitset paths now translate through the existing document map. Tests preserve
unmapped/multi-value deduplication, RGB logical IDs, constant scores, exact counts,
Unicode/escaping, Boolean composition, limit errors and sync/async equality.
The WASM release build and all 39 browser tests pass, including both QL forms.
Ordinary/RGB benchmark HTTP smoke tests pass. Final native validation is recorded
below after completion; no new wildcard timing or full-corpus count claim is made.

Final combined-tree validation: `.context/search-harness/20260923T094049.124403Z-check`
passes formatting, strict Clippy, 2,029 native tests (25 normally ignored),
native-without-sync and standalone broker compilation. The final WASM release
build and 39 browser tests pass, including bare and function wildcard patterns.
Both ordinary and RGB HTTP smoke tests pass, including native pattern parsing.
Documentation links and `git diff --check` pass. An intermediate check caught the
new bare-pattern regression while parser edits were still in progress; the final
run uses the completed, unchanged Rust tree. Full production RPC/lifecycle tests
were not rerun: no protobuf, production RPC, or lifecycle protocol changed.
No ARM throughput or full-corpus wildcard-performance claim is made.

A post-change capability check submits all 826 published expressions to the real
HTTP adapter over a four-document fixture: **801 accepted, 25 explicit errors**
(13 regex and 12 escaped-query syntax cases). All 145 wildcard expressions are
accepted. This fixture deliberately does not establish 10M-corpus count agreement
or test full-vocabulary expansion budgets; the throughput gate remains 15/826.
Raw responses and binary/query hashes are retained in
`.context/yonik-benchmark/gap/wildcard-http-capabilities-826.json`.
The native-only preflight separately accepts 730 expressions; its 83 syntax
rejections include 71 sloppy phrases handled by the HTTP adapter's existing
`PhraseQuery` translation, plus those same 12 escaped expressions.

The final 32-client evidence archive is downloaded and SHA-256 verified. Both
`benchmark-host` and `benchmark-host` are independently
confirmed `TERMINATED`; temporary transfer keys are removed and changed host
restrictions restored. The completed report preserves the startup/transfer
failures and successful recovery; no benchmark work remains running.

## Conjunction HTTP scheduling and response encoding

The [conjunction follow-up](benchmark-results/searchbench-2026-09-23-conjunctions.md)
separates core work from the benchmark frontend. Two HTTP runtime threads were
limiting throughput while serializing/destroying response JSON and scheduling
requests. Encoding and temporary-tree destruction now remain inside the existing
bounded blocking worker and admission permit. Core scoring and the persisted
index are unchanged. A diagnostic-only counter records actual MaxScore heap
updates, including conjunctions; it compiles out of ordinary builds.

The benchmark frontend now defaults its HTTP workers from available logical
CPUs: one on a single CPU, otherwise `clamp(ceil(CPUs / 8), 2, 8)`. Detection
respects the measured Linux affinity and selected counts are logged; explicit
1–64 overrides remain available. Production gRPC/search-pool defaults and the
64-request admission limit are unchanged. This heuristic is not claimed optimal
across architectures or workloads.

On the same 10M-document ordinary index at 32 clients, top-100 rises from 21,036
to 33,246 QPS with the original two HTTP workers. Automatic sizing selects four
and yields 32,720 top-100, 41,722 top-10 and 68,524 count QPS. Fresh Luxir results
are 41,003, 53,073 and 64,756 respectively. Relative to baseline, automatic
sizing plus worker encoding improves top-100 by 55.5% and counts by 33.6%, while
top-10 is essentially unchanged. Single-client throughput regresses by
2.3–19.4%; retain that tradeoff and the explicit worker override.

All 24 cells and their repetitions are error-free. All 15 admitted queries have
identical before/after exhaustive top-100 IDs/score bits/counts; all 45 response
bodies are byte-identical. Only seven conjunctions are timed, with no claim of
full 826-query compatibility. Remaining Summa profile costs include posting
decode/seek/intersection and ID-column random reads. Luxir's stripped executable
prevents a comparable function-level attribution; do not infer its exact pruning
or codec strategy from the throughput gap. The report retains memory, CPU usage,
profiles, stage timings, work counts and the limited health-control comparison.

Final validation: `.context/search-harness/20260923T110257.277217Z-check` passes
all five stages, including 2,029 native tests (25 normally ignored). CPU-default
and feature-only heap-counter regressions pass; feature Clippy, WASM release and
39 browser tests, ordinary/RGB HTTP smokes, CLI bounds, Python compilation and
documentation/diff checks pass. Full production RPC/lifecycle tests were not
rerun; no production RPC or lifecycle protocol changed.

The final evidence archive (including raw perf) is downloaded and SHA-256
verified as `4b535604f4f5c8f9875d12e750bbb1e3d01da8de0b1d2358033979f6aab83188`
(65,645,432 bytes). Temporary transfer keys are removed and changed host settings
restored. Both cloud stop commands lost their polling connection, but independent
status confirms **both machines `TERMINATED`**. No benchmark work remains running.

## Borrowed-ID responses and shared-pool handoff measurements

The [response/handoff follow-up](benchmark-results/searchbench-2026-09-23-handoffs.md)
uses a fresh paired 32-vCPU campaign over the same 10M-document ordinary index.
The benchmark HTTP response now borrows external-ID strings into one bounded
vector and serializes them inside the existing admitted worker. It removes
per-hit JSON maps, string copies and the outer temporary-tree conversion.
Scoring, pruning, persisted bytes, shared search-pool ownership, CPU-based HTTP
defaults and concurrency bounds remain unchanged.

At 32 clients, conjunction top-100 improves **17.4%** (33,148 → 38,909 QPS),
reducing the gap to fresh Luxir from **20.3% to 6.4%**. Top-10 improves **6.7%**
but remains **15.0%** below Luxir. Count regresses **4.9%** while remaining
**2.9%** above Luxir. The shorter screen also shows a count regression; this
is a ranked-response improvement with an unresolved count tradeoff. Single-client
medians improve for ranked operations but regress for count; retain the raw
repetition ranges and do not infer a universal latency benefit.

Opt-in bounded timing diagnostics place response projection plus encoding/drop
at about 156 → 65 microseconds for 32-client conjunction top-100, with segment
work approximately unchanged. HTTP/blocking and shared-search-pool handoffs
remain material wall-time components. These nested, instrumented means include
warmup and are not production latency measurements. Normal builds compile out
the endpoint and timers. Completed pool installs count on the capturing caller;
direct async count collection and invalid-window rejection do not enter that pool.

Remaining work: attribute the count regression; investigate dispatch overhead
without adding an executor or bypassing owner-controlled search capacity; and
measure reader-owned sparse/batched ID lookup with a bounded metadata/scratch
budget. Production async entry differs from the benchmark's double offload, so
the measured handoff cost must not be projected onto gRPC. Existing unsuccessful
AND-bound-pruning experiments remain evidence against enabling more checks merely
to lower decoded-block counts. Luxir's stripped binary still prevents equivalent
function-level attribution of its internal strategy.

The saturation check at 64 clients reaches 47,107 conjunction top-100 QPS versus
Luxir's 44,167. At 32 clients, Summa uses 649 CPU µs/request versus Luxir's 703,
but only 25.25 CPU equivalents versus 29.22; utilization is now a larger part of
that gap than per-request CPU cost. These 64-client results remain separate from
the requested 32-client comparison. Peak anonymous RSS is 111.3 MiB versus
Luxir's 15.1 MiB, so the response change does not resolve the memory difference.
The separate count diagnostic shows cheaper encoding and nearly unchanged search
time, with higher parsing time; it does not establish the cause of the regression.

All 45 cells / 135 repetitions pass without request or memory-sampling errors.
The 15 exhaustive audits and 45 response bodies are identical before/after.
Coverage remains the original 15/826 count-compatible queries. Native harness
`.context/search-harness/20260923T113909.147943Z-check` passes all five stages,
including 2,029 tests (25 normally ignored). Example byte/default tests, feature
core diagnostics/Clippy, ordinary/RGB/diagnostic HTTP smoke checks, WASM release
and 39 WASM JavaScript tests pass. Full production RPC/lifecycle tests were not
rerun; no production protocol or lifecycle mechanism changed.

Final evidence is downloaded and checksum-verified as
`382ba48a577a49b034d6aee275606e6e497472ec0fd84dc52c978f4e05d7aadb`
(766,389 bytes), with the separate source/build archive recorded in the report.

Temporary transfer keys are removed and host restrictions are restored. Both
cloud stop commands lost their polling connection; independent cloud status
confirms **both machines `TERMINATED`** after evidence verification. No benchmark work
remains running. Final documentation links, ownership contracts, Python checks
and `git diff --check` pass.

## Worker configuration and alternating count comparison

The [worker study](benchmark-results/searchbench-2026-09-23-workers.md) varies
existing settings on the frozen borrowed-ID executable; it makes no Rust/runtime,
admission, format or default changes. It tests 8/15/30 coupled search/blocking
workers against 2/4/8 HTTP workers, with three stable 30/4 baseline anchors.
Smaller pools lose ranked throughput, and eight HTTP workers do not improve the
mix. This is not an isolated Rayon-pool experiment: `WORKERS` controls both pools.

The independent 32-client/all-15-query comparison finds a useful explicit
30-worker/two-HTTP-worker tradeoff: conjunction top-10 is **8.9% faster** and
top-100 **4.9% faster** than four HTTP workers, but count is **18.8% slower**.
Top-100 nearly matches fresh Luxir (42,720 versus 42,887 QPS). Four HTTP workers
retain the count advantage over Luxir (76,094 versus 66,996). Phrase ranked gains
are smaller or absent; phrase counts are effectively unchanged. Keep the
CPU-derived HTTP default and the explicit override, not a new universal policy.

The earlier 4.9% borrowed-response count-throughput regression at 32 clients does
not reproduce in before/borrowed/borrowed/before order: average session medians
are 76,639.99 versus 76,638.93 QPS. The original measurement remains recorded;
this does not establish single-client behavior. A **1.5% CPU-cost increase**
(315.4 versus 310.7 µs/count) remains unassigned. In the short screen, 15 workers
and two HTTP workers match count throughput with about **32% less CPU per count**,
but lose ranked throughput; confirm this count-only lead before any policy change.

Remaining ranked cost still includes utilization: two-HTTP-worker top-100 uses
599 CPU µs/request versus Luxir's 689, but 25.59 busy CPU equivalents versus
29.53. Pinned Tokio 1.53.1 source inspection shows `block_in_place` itself transfers
the runtime worker core through a blocking task. Replacing the benchmark's
outer dispatch is not automatically removal of all handoffs. Any experiment must
preserve Searcher ownership, bounded capacity, reader/permit retention,
panic-to-response behavior, cancellation and shutdown; no additional executor or
benchmark-only public core API was introduced here. Peak anonymous RSS remains
107.8–109.5 MiB for Summa versus Luxir's 12.4 MiB in this run.

All **64 cells / 192 repetitions** pass without request or memory-sampling errors.
Seventeen Summa instances preserve all 45 response bodies; worker widths preserve
the exhaustive IDs/score-bits/count audit. Coverage remains **15/826**. Harness
`.context/search-harness/20260923T160104.343227Z-check` passes all five stages and
2,029 native tests (25 normally ignored). Ruff/Python/report checks pass. WASM and
full production RPC tests were not rerun for this configuration-only follow-up;
the preceding WASM release/39 JavaScript tests validate the unchanged code.

Final evidence is downloaded, size-checked and SHA-256 verified as
`f8f8b4ffe0accb6a89547d16348de102cc1a688b4258ecf17b1caec000cffa2d`
(1,139,078 bytes). The build machine remained stopped. Initial cloud-start polling and
an early SSH connection failed; independent status and a successful retry
established the benchmark machine before timing. No timed sample was affected.

Host restrictions are restored and no temporary inter-machine transfer keys were
created. The benchmark stop command lost its polling connection; independent
cloud status confirms **both machines `TERMINATED`** after evidence verification.
No benchmark work remains running. Final documentation links, ownership
contracts, Python/Ruff and diff checks pass.

## Bounded ID lookup, dispatch policy, and envelope ownership

The [completed study](benchmark-results/searchbench-2026-09-23-dispatch-directory.md)
implements all three follow-ups. The fast-field owner now keeps at most 256
sparse header checkpoints (3 KiB heap payload per reader across all source
blocks), using the existing decoder and original encoded bytes. The benchmark
frontend exposes `blocking` / `in-place` dispatch while preserving 64-request
admission, reader/permit ownership through cancellation and panic, and shutdown
draining. Blocking remains the default; Searcher pool ownership and CPU-derived
HTTP-worker defaults are unchanged.

Envelope conversion moves both validated strings from the consumed JSON object.
A counting-allocator regression test caught two temporary key allocations in the
initial mutable-indexing prototype. Borrowed `get_mut` lookups remove those too:
successful conversion now allocates zero times. The corrected executable was
rebuilt and all 15 admitted queries were remeasured against fresh controls;
prototype measurements remain separate.

At 32 clients, corrected conjunction top-100 improves **4.1%** (37,927 to
39,493 QPS) and consumes **3.8% less CPU per request** (658.8 to 633.7 µs).
Top-10 improves 1.3%; count is effectively unchanged. The prototype ABBA repeat
also finds a 3.7% top-100 improvement and does not reproduce its first run's
2.8% top-10 loss. Separate stage probes show **about 43% less ID projection time**;
that is not an HTTP latency improvement of the same size. Two ARM decoder
microbenchmark passes support the reduced header-walk cost, with their small
control and noisy scalar measurement retained in the report.

In-place dispatch loses **2.5%** conjunction top-100 throughput and adds **7.8%**
CPU cost relative to updated blocking execution. Some phrase workloads improve,
but the evidence does not justify changing the default. Tokio still transfers
its runtime core, and the shared Searcher handoff remains. Updated Summa is
16.8% behind fresh Luxir on conjunction top-10 and 2.9% ahead on top-100 in the
corrected campaign. Luxir top-100 varies between campaigns; this is not a general
performance lead. Coverage remains **15/826**, with one medium-phrase query.

Peak anonymous RSS remains 109.1 MiB for updated blocking Summa versus 12.4 MiB
for Luxir; total RSS is 1,150.6 versus 159.0 MiB. Fast-field block metadata and
checkpoints now contribute to estimated heap accounting. Existing lazy dictionary
tables and ordinal maps remain outside that estimate. Remaining work includes
explaining the top-10 utilization/CPU-cost gap and memory difference, measuring
startup cost of the extra validated-header pass, and extending compatibility
coverage before claiming parity across the full query set.

All **114 cells / 342 repetitions** pass request and memory-sampling checks.
Fifteen Summa instances preserve all 45 response bodies and saved exhaustive
ranked-ID/score-bit/count audits agree. Full harness run
`20260923T170615.743556Z-full` passes all nine stages, including real-server
broker E2E. Final check `20260923T180425.771100Z-check` passes all five stages
and 2,031 native tests (25 normally ignored). One intermediate mock-broker
discovery timeout passes focused and full retries without code changes; the
failed log is retained. Final WASM release, 39 JavaScript tests (also rerun after
`npm ci`), diagnostic-feature Clippy, seven example tests and four real HTTP
smoke configurations pass. Documentation/contracts and diff checks pass.

Final combined evidence is downloaded and SHA-256 verified as
`6d9dffed84a78f3f3ad999dd930e70234729e699866bb5022e23371bbea35818`
(1,828,573 bytes), with source/build artifacts, raw Criterion data and validation
logs retained separately. The report records the pre-timing launch failures and
separate-boot repeats. Transfer credentials are removed and host restrictions
restored. Independent final cloud state confirms **both machines `TERMINATED`**.

## Range interpolation experiment (2026-09-19; rejected)

An exact quotient/remainder recurrence speeds warm BlockwiseLinear bitset scans
by about 2× on Apple M4 and 3.7× on Cascade Lake, but changes compiler decisions
in shared scan code. On a dedicated x86 machine, all four ordinary shuffled bitpacked
controls regress 8.1–8.3% in alternating runs. A final explicit kernel boundary
still regresses them 9.4–9.6%. No runtime change or new default is retained.

The [report and evidence](range-block-scans.md#incremental-interpolation-experiment-not-integrated)
record exact candidate patches, same-version/compiler/fixture comparisons,
confidence intervals, correctness and process-memory measurements. The main
prototype passed 2,031 native tests, three async-only range tests, and the WASM
build with 38 tests. Remote evidence was downloaded and hash-verified; the
isolated machine and boot disk were deleted. The lazy scorer follow-up below implements
bounded batching; accepting fully covered block spans through the shared reader
scan protocol remains unimplemented.

## Lazy range scorer batching (2026-09-19)

The shared native/async/WASM range scorer now batches sustained single-value
scans through a reader-owned cursor and retains a 64-bit membership mask. The
first eight probes and distant seeks remain scalar; multi-value fields retain
first-value semantics. Codec arithmetic, serialization, schema and planner
policies are unchanged. The existing boxed scorer grows from 40 to 96 bytes,
with 512 bytes of decoded stack scratch during refill and no added allocation.

On paired 65,536-document Cascade Lake runs, shuffled full scans improve from
829.109 to 330.960 microseconds (2.5×), and piecewise compressed scans improve
from 16.914 to 1.391 ms (12.2×). M4 measurements show 2.2× and 16.1× respectively,
with more background-load noise. Constant, missing-value and complete-miss scans
also improve; scalar multi-value scans remain effectively unchanged. The cost is
3.6–5.4 ns on x86 short first-hit queries and 0.06–0.15 microseconds across 65
cheap sparse seeks. Existing bitset controls do not regress on the isolated host.

The [report and reproducible evidence](range-block-scans.md#lazy-range-scorer-batching)
include per-run confidence intervals, exact patches, compiler/host information,
assembly findings, correctness and memory measurements. The complete check
harness passes 2,031 native tests, all four async-only range tests pass, and WASM
builds with all 38 tests passing. Evidence was hash-verified before deleting the
machine and boot disk. Cold/concurrent full-query, RPC and GPU checks were not run.

Remaining experiments: accept fully covered block spans without decoding, and
batch multi-value offsets/first values in their owning reader. Neither follows
from this result without separate measurements and semantic regressions.

## Plain-index ranked bound admission (September 24)

The expanded workload exposed unnecessary admission restrictions: ordinary
ranked terms required optional ratio bounds to enter MaxScore, and typed
ranked conjunction pruning required both a document map and ratio bounds.
Existing maximum-TF/minimum-length metadata is also conservative. Core now
admits it through the existing windowed term executor and typed conjunction
executor. The change preserves strict bound comparisons, supported finite
scoring parameters, exact COUNT traversal, positions, deletions, stable-ID ties,
and global statistics. No encoded bytes, schema or indexing defaults change.
The [paired report](benchmark-results/plain-bounds-2026-09-24/README.md) records
334-query ordinary/RGB comparisons, ARM controls, CPU, RSS and exact audits.

All 168 timing cells / 504 repetitions complete without errors, with 1,336
exhaustive query audits and 8,016 HTTP checks. Plain term top-k throughput rises
2.70–8.24× and CPU/request falls 64.3–89.1%; RGB high/high conjunction throughput
rises 31.8–94.3%. Plain high/low conjunctions regress 2.1–3.4%, plain high/medium
top-100 regresses 5.1%, and RGB high/low regresses 1.9–2.4%, consistently across
both rounds. COUNT/phrase controls stay within 1.5%. Peak anonymous RSS changes
123.6→124.8 MiB on plain and 130.5→130.7 MiB on RGB. Luxir was not rerun; the
earlier comparison still indicates substantial term and rare-conjunction gaps.
Final predicate helper extraction is correctness-tested but was not separately
timed; the report preserves the exact measured source patch.

The new work-counter regression fails on 1.9.1 because a plain ranked term
scores all 8,192 fixture documents without consulting block bounds. It passes
with the change; exhaustive IDs/score bits and counts agree. The existing
multi-segment term oracle now also covers indexes without optional ratio
metadata. Final native harness `20260924T195139.413963Z-check` passes 2,054 tests
(25 ignored), strict Clippy, async-only native and standalone broker checks.
Six focused diagnostic/ranking tests, 52 Linux release test executions, and
the WASM release build plus all 41 JavaScript tests pass, including a rerun
after the predicate helper cleanup. Four benchmark-script tests and the
documentation/link checks pass. An initial broker
discovery timeout passes its focused retry and the final full check. Full
lifecycle/RPC validation was not rerun because those protocols did not change.

Remaining findings: the benchmark audit's zero scorer limit does not establish
exhaustive OR enumeration; fix that diagnostic admission before using it as an
OR oracle. Also, `BlockPostingList::from_layout` still extracts two L1 arrays
from immutable bytes for each posting open. That work scales with posting-list
length even for a rare conjunction probe. A borrowed little-endian directory
view is a candidate follow-up, requiring byte/seek equivalence and measurements;
no additional cache or representation rewrite is introduced here.

## September 25: expanded workload, borrowed metadata and bounded term unions

The [closing-gap investigation](benchmark-results/closing-gap-2026-09-24/README.md)
records complete retrieved same-index matrices, the opt-in Unicode-word analyzer
comparison, concurrent expansion ownership, rejected kernels and ARM controls.
This work closes both findings from the preceding review: the benchmark OR audit
now explicitly collects exhaustive membership, and immutable L1 group directories
are borrowed as validated little-endian views instead of copied per query.
Encoded bytes, unaligned reads, seeks and scores remain covered by regressions.

The opt-in analyzer produces 677 successful full-corpus queries, 640 exact
reference counts, and all 677 within 5%; 149 requests retain explicit resource
errors. It changes index semantics and is reported separately from execution.
Compared with 684 successful queries previously, coverage drops by seven even
though count agreement improves. Four regex expressions now fit unchanged
budgets through finite literal-prefix ranges with aggregate accounting.

On the same 55-query index, query-local posting byte ownership raises concurrent
prefix top-k from about 5,800 to 16,000–18,000 QPS and reduces server CPU/request
from 4.85 ms to 1.5–1.65 ms; peak anonymous RSS falls from 203 to 187–190 MiB.
Subsequent query-local integrity ownership preserves the original first-error
observer; its newer matrix remains pending collection. Lazy iterator opening,
decoded-ID union accumulation and materialized membership batches retain exact
counts, deletion filtering, mapped IDs and stable tie behavior. One-to-three-doc
inline postings now stay in the canonical dictionary decoder's fixed storage
instead of being re-encoded for expanded-term queries. The paired ARM prefix
fixture improves 7.18–7.54× versus the preceding checkpoint, with regex/star
controls improving 7–12%; corpus retiming is outstanding.

The density-gated bitmap loop improves the common `+of +s` count probe about
1.47× in latency and CPU and the dense ARM fixtures 2.12–2.40×. Sparse ARM
controls remain within 1.1%. Unconditional grouping, alternative pair kernels,
a peeled variable-integer reader and rare-side pre-scoring were rejected after
corpus regressions. Fewer search workers help cheap queries but hurt conjunctions;
no worker, cache, norm-precision, impact or reorder defaults change.

The final inline implementation passes harness `20260924T223721.093990Z-check`:
2,069 native tests (25 ignored), formatting, strict Clippy, native without sync,
and standalone broker compilation. Two additional fixed-inline decoder tests
pass afterward in the nine-test inline selection; the final WASM release build
and all 41 JavaScript tests pass. Full lifecycle/RPC validation was not rerun
because those implementations did not change. Documentation checks pass.

Remaining work: several conjunction, broad-dictionary and cheap-term families
still trail Luxir. Final-source corpus retiming, collection of integrity/cache
and quantized-norm matrices, a blocked-intersection screen, and independent
shutdown verification are incomplete because remote authentication expired.
The immutable indexes and queued jobs remain available for continuation; neither
completion nor shutdown is inferred from the preconfigured shutdown schedule.

### September 25 follow-up: dense unions and dictionary setup

Matched-value projection removes the discarded key allocation from bounded term
expansion while sharing the existing SSTable parser, order, budgets and error
paths. The corpus prefix screen improves about 9%; ARM prefix fixtures improve
about 21–22%. Broad dictionary scans remain near parity.

Retaining the already-materialized union bitmap through the existing DocSet
window protocol avoids a bitmap-to-ID-vector conversion. The six-query ABBA
corpus screen gives 1.62–1.83× lower latency for four dense prefix/wildcard counts,
with corresponding CPU savings and exact counts. The broad scan and single-term
controls remain near parity. ARM dense/overlapping fixtures improve 1.35–1.50×;
the sparse control is within 1.1%. No resource limit or format changes.

Deferring optional posting byte views until a lazy union opens that list reduces
prefix search time by 1.70–1.76× in the seven-query corpus screen. All external
footers are still validated at expansion, and the canonical constructor/decoder
retain ownership of list interpretation. The count control is within 1.4%; ARM
prefix fixtures improve 6–7%. These are individual-query/fixture measurements;
the final concurrent matrix remains pending. Regression tests compare serialized
bytes, decoded blocks, reader-drop lifetimes and shared corruption observation.

Evidence: [projection](benchmark-results/closing-gap-2026-09-24/project-screen.json),
[dense count](benchmark-results/closing-gap-2026-09-24/dense-count.json),
[deferred views](benchmark-results/closing-gap-2026-09-24/deferred-view-screen.json),
[count control](benchmark-results/closing-gap-2026-09-24/deferred-count.json), and
[ARM dense unions](benchmark-results/closing-gap-2026-09-24/arm-dense.json).

### Photon research and format feasibility

[Photon](https://www.perplexity.ai/el/hub/blog/photon) describes density-adaptive
postings, selective frequency/position access, batch-oriented caching with CLOCK
eviction, and separate index construction. Its bounded candidate selection does
not guarantee exact top-k. These are workload-specific design choices, not
controlled comparisons with Summa.

For Summa, the actionable hypotheses are direct membership access for dense
postings and lower shared-cache contention. Existing exact score bounds,
corruption checks and count semantics remain requirements. A separate docblob
ranking representation or asynchronous disk reactor would need cold-I/O evidence;
the current warm wildcard profile instead concentrates on dictionary decoding.

The user explicitly authorized index-format experiments, including incompatible
changes if needed. A private feasibility probe builds a per-term FST from the
canonical dictionary iterator, evaluates single-star matching, and fetches only
matching values through canonical point lookup. It compares complete values with
the existing bounded scanner's matching kernel on the same corpus. This probe
writes a separate experimental artifact, changes no live index, and does not
establish a production format or new query-budget policy. Any retained format
must account for metadata size/residency, bounded construction, native/async/WASM
execution, corruption rejection, and compatible encoded merge behavior.
