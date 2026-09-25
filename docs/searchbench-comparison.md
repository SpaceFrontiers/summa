# Yonik Searchbench comparison and phrase optimization

September 24 expanded comparison: [all-query coverage and skipping experiments](benchmark-results/skipping-2026-09-24/README.md)
retain all **826** original query attempts. Summa executes **684** on the full
10M corpus; 142 hit explicit resource limits. The new [shared-input mode](#shared-input-comparison-mode)
allows count differences and exposes them alongside the completed 684-query timings. The older results
below use the stricter 15-query exact-count subset and remain historical results.
[Regex and escaped literals](regex-query.md) achieve 826/826 acceptance on the
four-document HTTP fixture; fixture acceptance does not establish full-corpus
execution within the same resource limits.

Status: September 23, 2026. **The four-engine 10M campaign completed successfully**
on September 22 at 21:06 UTC (September 23, 00:06 Moscow). All 108 measured cells
completed without request errors. Only **15/826 queries** passed the full-corpus
count gate: seven conjunctions, seven low-frequency phrases, and one medium-frequency
phrase. Elasticsearch, OpenSearch, and Luxir agreed on all 826 count probes;
Summa returned errors for 216 rows and different counts for another 595.
This is a severely restricted comparison, not a complete Searchbench result.

The [result table](benchmark-results/searchbench-2026-09-22.md) and
[CSV](benchmark-results/searchbench-2026-09-22.csv) retain every measured cell.
At eight clients, top-10 conjunction QPS was Summa 9,253, Elasticsearch 5,049,
OpenSearch 5,742, and Luxir 10,848. Phrase ranking was substantially slower in
Summa: 470 QPS for the seven low-phrase queries versus 984–1,503 for the references;
189 QPS for the single medium phrase versus 1,017–3,518. These results identify a
phrase-ranking gap but cannot establish an overall engine ranking.

The [first optimization/RGB follow-up](benchmark-results/searchbench-2026-09-23.md)
completed another 27 cells. It exposes a mapped-length phrase-pruning bug and
helps RGB, but does not close the ordinary-index phrase gap. Further non-RGB
work is described below. The [final non-RGB paired run](benchmark-results/searchbench-2026-09-23-phrases.md)
now measures 2,605 QPS for low-phrase top-10 and 2,129 for medium-phrase top-10,
versus fresh Luxir's 1,510 and 3,524. Summa leads both phrase top-100 cells;
medium-phrase top-10 and exact counting remain behind.

The monitoring process lost GCP authentication after 19:36 UTC. The remote job
continued and finished; result collection and machine shutdown were delayed until
access recovered on September 23. Evidence for that campaign is available under
`.context/yonik-benchmark/evidence/`; the machine was stopped after collection and
restarted for the non-RGB follow-up described below.

## Pinned workload

The requested [Elasticsearch/OpenSearch comparison](https://yonik.com/bench/elasticsearch-vs-opensearch/)
reports a September 20 campaign using Searchbench tag
[`fulltext-munin-host-20260920`](https://github.com/yonik/searchbench/tree/fulltext-munin-host-20260920),
resolved here to commit `81f67380ef2d1528b40b6770f5ea5d6b105bc88e`.
Its [benchmark definition](https://github.com/yonik/searchbench/blob/fulltext-munin-host-20260920/docs/full-text-benchmark.md)
and source presets define the workload; this is distinct from our existing
[Search Benchmark Game](search-benchmark-game.md) adapter.

- First 10,000,000 rows of the May 2012 Wikipedia line-document corpus; article
  chunks up to 1024 characters, preserving original text. The compressed source
  is 6,299,260,005 bytes, SHA-256
  `f3c18abca52f479ad29b427b6e7f7d453867dfc3e25546922fae855868d2936f`.
- The selected source file contains 826 queries in 20 families. Top-10, top-100,
  and exact count make 60 cells at each of 1/8/32 connections.
- Elasticsearch 9.5.4 and OpenSearch 3.8.0, official distributions with bundled
  JDKs and 8 GiB heaps. Query/request caches disabled, filesystem cache warm.
- One node/shard/force-merged segment, no replicas. Server and native HTTP load
  generator have disjoint CPU sets. Each engine is measured separately.
- Top-k projects only the corpus ID from a column, requests no total count,
  and reads no stored documents. Count returns an exact total and no hits.
- The article uses 30 seconds of session warmup, complete untimed response
  validation, one second of connection warmup, and three 10-second repetitions.
  Report median repetition QPS per family, operation, and concurrency; these
  short repetitions do not establish p99 latency.

The article's hardware is a Ryzen 9 9955HX with 64 GB RAM, 14 physical server
cores and two driver cores. Measurements on another host must rerun all engines
on that host; do not combine Summa numbers with the article's absolute QPS.
The existing GCP instance `benchmark-host` in `us-east1-b` is running (`n2-highmem-8`: 8 vCPUs, 62 GiB usable RAM). Server affinity is
`0-2,4-6` (three physical cores with SMT); driver affinity is `3,7` (one physical
core with SMT). This differs substantially from the article's host. New artifacts
live on the existing mounted 1 TB data disk under
`/mnt/summa-copy/searchbench-20260922`; the full boot disk is not used for new
build outputs or indexes.

The user authorized restarting this machine. The September 22 run uses a benchmark
HTTP example with a shared immutable searcher, bounded blocking execution, and
fast-column ID projection. The experiment will keep body analysis explicit
(`lex` with Unicode segmentation, no stemming/folding/variants, maximum token
length 255) and gate every comparison on corpus-specific count agreement.
Unsupported wildcard/regex families remain excluded from all paired timings.
The native string parser remains the owner for ordinary queries; the adapter
translates only the declared sloppy-phrase shape to existing `PhraseQuery`.

## Initial compatibility (September 22)

The native parser probe ran every selected query through the existing release
`search_benchmark_game validate-queries` entry point, without modifying the parser.

| Classification                        | Queries | Meaning                                                                    |
| ------------------------------------- | ------: | -------------------------------------------------------------------------- |
| Parses with no known missing operator |     585 | Still requires count agreement; parsing does not establish semantics       |
| Native syntax rejected                |      83 | 71 sloppy-phrase expressions and 12 punctuated/escaped Boolean expressions |
| Wildcard/regex operator unavailable   |     158 | 49 wildcard, 49 wildcard-scan, 47 leading-wildcard, 13 regex               |

All three term families, exact phrase families, and three-character prefixes have
native primitives. Boolean support exists, but the Lucene task file includes
literal terms such as `books.google.com` and escaped `user\:ed` that our query
grammar does not accept directly. `PhraseQuery::with_slop` exists in core, while
the native string grammar does not accept the task file's `"phrase"~4` spelling.
An adapter can translate that spelling to the existing phrase implementation;
it must then check matching semantics, not assume Lucene slop equivalence.

The 158 wildcard/regex rows must remain visibly unsupported unless core gains
those operators. **100 of them are accepted by the native grammar without
implementing the requested wildcard/regex operation** (98 wildcard rows and two
regex rows). On a one-segment fixture containing `there`, `them`, `the`, and `e`:

| Query  | Summa count | Intended wildcard/prefix count |
| ------ | ----------: | -----------------------------: |
| `th*e` |           4 |                              2 |
| `th*`  |           3 |                              3 |

The grammar interprets `th*e` as prefix `th*` OR term `e`. Therefore the benchmark
adapter must gate by query family before parsing, even when strict parsing
succeeds. This smoke check identifies a compatibility gap; no core behavior has
been changed or wildcard implementation added by this preparation.

Other boundaries:

- The production server exposes gRPC, while Searchbench drives persistent HTTP.
  A benchmark HTTP frontend over core would need to be explicitly named as such;
  it would not measure the production gRPC server's overhead.
- Summa's default tokenizer strips punctuation within whitespace-delimited
  tokens. Its lexical tokenizer also has normalization differences from Lucene's
  standard analyzer. The Game adapter's text transformation cannot be reused to
  claim the original-text Searchbench posture.
- The response's scored-document counter is not an exact count. Use the existing
  `CountCollector` path for COUNT, and the existing searcher for ranked retrieval.
- Project `id` through the existing fast-field reader. Reading the document store
  or merely returning internal IDs would change the requested workload.

## Implementation and measurement flow

1. Pin/download the source and reference distributions, verify their manifest
   identities, and generate the standard enriched corpus through the upstream
   transformer. Preserve all baseline fields; document native representation
   differences and avoid an ingest/size comparison on different field layouts.
2. Add an explicitly benchmark-only HTTP frontend in the server/examples owner,
   using the existing schema parser, writer, immutable searcher, exact-count
   collector, and fast-field projection. Bound requests, CPU admission, and
   response size; keep query execution out of the HTTP event loop. No alternate
   scorer, custom posting reader, or production transport replacement.
3. Add the corresponding request/topology/result adapter to the pinned
   Searchbench checkout. Use its native replay driver and record the exact
   emitted request bytes, index topology, engine/binary identities, and CPU sets.
4. Run a small smoke corpus across all compared engines. Translate only supported
   operators; report missing families and parser/analyzer/count disagreements.
   Never silently substitute an OR/prefix for a wildcard or use estimated totals.
5. Build and force-merge the full 10M indexes. Recheck the selected queries on the
   full corpus: agreement on a smoke prefix does not establish 10M agreement.
   Run each compared engine on the same resulting query subset and report every
   exclusion alongside the full published workload coverage.
6. Measure 1/8/32 clients sequentially per engine, with shared fixture, CPU sets,
   warmup and repetitions. Report per-family QPS and repetition spread, RSS,
   request failures, and topology. Keep HTTP health control, indexing, and any
   secondary query suite separate from full-text results.

The partial-selection collector prototypes remain benchmark-only and are not
enabled in this comparison. Comparing production Summa on a new workload is
separate from a collector A/B experiment.

## Reproducing the completed preflight

```sh
git clone --depth 1 --branch fulltext-munin-host-20260920 \
  https://github.com/yonik/searchbench.git .context/yonik-benchmark/searchbench
cargo build --locked --release -p summa-core --example search_benchmark_game
python3 scripts/searchbench/preflight.py \
  --binary target/release/examples/search_benchmark_game \
  --queries .context/yonik-benchmark/searchbench/queries/luceneutil/queries.txt \
  --output .context/yonik-benchmark/query-capabilities.json
```

The [probe](../scripts/searchbench/preflight.py) classifies every input and retains
native parse errors. It deliberately labels accepted queries as requiring count
agreement, not as benchmark-ready. Raw results and the four-document smoke output
are in `.context/yonik-benchmark/`. These checks produce no timing or QPS claim.

The native release build, all 826 syntax probes, the semantic smoke, Python lint,
documentation checks, and `python3 scripts/check_search.py check` completed. The
search-harness evidence is in
`.context/search-harness/20260922T185022.374433Z-check/`. No full RPC suite or WASM
rebuild was run: this preparation changes no runtime search code. The corpus
transfer is paused as an explicitly unverified `.part` file, with resumable
transfer metadata in `.context/yonik-benchmark/corpus-transfer.json`; it has not
been used to claim a verified 10M-document fixture.

## September 22 execution findings

The [HTTP frontend](../summa-server/examples/searchbench_http/main.rs) delegates
index construction to `IndexWriter`, ranked search to the synchronous `Searcher`,
exact counts to `collect_segment`/`CountCollector`, and ID hydration to fast text
columns. The HTTP frontend is an example target with an Axum dev dependency; it
does not alter the production gRPC service. A semaphore bounds queued/in-flight
requests to 64, with six blocking workers/search threads in the measured posture.
Core format and search feature defaults remain unchanged. The full enriched
corpus fields are retained, but dates use an exact text column rather than
Lucene's numeric date representation; this is a full-text query experiment,
not a physical index-size or ingestion comparison.

The [campaign adapter](../scripts/searchbench/campaign.py) reuses upstream REST
request construction, validation transport, binary workload serialization,
`bench_replay`, and `/proc` memory sampling. Its complete count probe retains all
826 rows. The gate selects one identical subset for all three engines. Untimed
ranked validation checks projected external IDs, uniqueness, and the expected
number of hits. The [runner](../scripts/searchbench/run.sh) uses the upstream
reference start/feed/merge machinery and runs engines sequentially.

The 100,000-row smoke prefix found exact count agreement for 162/826 queries (unchanged after adding Luxir):
15 terms, 56 conjunctions, 65 exact phrases, and 26 sloppy phrases. No union or
prefix query passed. Unsupported wildcard/regex families are explicitly excluded.
This is a serious coverage limit, not a performance result. Examples:

| Query |  Summa | Elasticsearch | OpenSearch |
| ----- | -----: | ------------: | ---------: |
| `the` | 90,785 |        90,758 |     90,758 |
| `is`  | 47,700 |        46,746 |     46,746 |
| `was` | 37,329 |        37,328 |     37,328 |

The lexical tokenizer always applies punctuation/contraction/elision handling and
NFKC normalization even with stemming, variants, folding, and stop words disabled.
Those operations differ from Lucene's standard analyzer. Equal counts on a corpus
are a benchmark gate, not proof that tokenization or relevance semantics match
for arbitrary documents. Full-corpus agreement must be measured again; this
smoke subset must not be reused as proof of 10M compatibility.

`python3 scripts/searchbench/smoke.py BINARY` exercises the actual HTTP frontend,
external IDs, exact counts, phrase/slop/prefix translation, unsupported-query
errors, concurrent requests, and graceful shutdown. It passed on both architectures.
The first repository `check` run passed formatting and Clippy but hit three
10-second broker discovery timeouts in the concurrent integration suite. The complete serial
rerun passed all five check stages (including the previously failing broker tests);
evidence is in `.context/search-harness/20260922T191320.317192Z-check/`.
Python lint and documentation checks also passed. The full production RPC suite
and WASM rebuild were not run: this adds only a native benchmark example.

### Luxir added to the campaign

At the user's request, the campaign now includes Luxir, using its official
[0.1.0 release](https://github.com/luxir-search/luxir/releases/tag/v0.1.0), published
September 21. The Linux x86-64-v4 binary passes the published SHA-256 check and
runs on the machine's AVX-512-capable Intel CPU. This is a pinned public release, not
the unpublished local release build used in the article. The upstream Luxir
adapter, feed script, and startup flags are reused, with the query cache disabled.
Its probe participates in the same four-engine count gate; its timings will use
the same resulting subset, corpus, CPU allocation, warmup, and repetitions.

## Run and evidence

The completed run executed sequentially on `benchmark-host`: reference
index construction, Summa construction, all four count probes, and the common
subset timing matrix. Repository changes are uncommitted.

Remote progress: `/mnt/summa-copy/searchbench-20260922/status.txt` and
`campaign.log`. The local detached supervisor records progress in
`.context/yonik-benchmark/supervisor.log`. On completion it downloads the raw
results, count exclusions, memory samples, and logs to
`.context/yonik-benchmark/evidence/`, then stops the dedicated machine. A 12-hour bound
prevents an unattended run from keeping the machine running indefinitely. A failed
run is preserved as a failure, without generating a completed comparison.

The [report generator](../scripts/searchbench/report.py) produces
`results/comparison.md` and `results/comparison.csv` only after all four engines
have completion markers and identical cell coverage. It retains repetition
spread and memory alongside median QPS. Inspect `completion.json` beside the
supervisor log for the benchmark and machine-shutdown exit statuses.

## Count and throughput investigation — September 23

The follow-up keeps the existing 10M indexes and production semantics. First,
classify every exclusion and compare canonical exact counts with exhaustive
scorer enumeration. Small punctuation and repeated/sloppy-phrase fixtures will
separate analysis differences from matching rules. Then profile the slow phrase
queries and compare any optimization against the unchanged binary/index on the
same machine and CPU allocation. Counts and ranked IDs/scores must remain equal;
byte formats, analyzer semantics and production defaults are outside the
performance rewrite. Diagnostic scratch retains only a bounded ID sample.

The initial profile of `"references reflist"` top-10 attributes 38.13% of sampled
CPU time to per-candidate phrase bounds, 13.16% to posting intersections, and
11.80% to the ranked collector. Proposed rewrite: expose conservative contiguous
candidate-block bounds to the existing ranked collector, using the original
first phrase term's max-TF/min-length metadata. Cache each block bound. A block
may be skipped only if even the smallest possible result ID cannot enter the
heap; mapped stable IDs and score ties remain safe. Frequency uses original-first
occurrences (including duplicates), so another term's smaller TF is not a valid
substitute. Complete counts and nested scorers retain exhaustive traversal.
No new encoded format or persisted metadata is introduced.

The RGB experiment builds a separate index from the same corpus with only the
body field's `reorder` attribute enabled. The canonical writer first merges to
one segment, then explicitly calls `IndexWriter::reorder()`. It retains the
ordinary index and measures both independently; build/reorder work never overlaps
query timing. The existing bounded RGB algorithm and stable logical-ID mapping
remain unchanged. Tables identify every timing column as throughput in QPS.

The first block-bound prototype added a virtual call per candidate and lost
throughput (quick 3×3s check: about 60 to 53 QPS for the medium phrase). The
revision checks each block once and adds a bounded 4 KiB table per phrase scorer
for TF=1 candidates with scoring lengths below 1,024. Each entry is the existing conservative
BM25 envelope for that length, preserving its rounding margin and fallback for
other frequencies. Lists with fewer than 1,024 possible candidates keep the
scalar path to avoid lookup-table setup dominating selective queries. The full
A/B results are linked above.

The [OpenSearch cancellation report](https://yonik.com/blog/opensearch-cancellation-wrapper/)
points to bulk-operation forwarding, iterator reuse, and allocation-free checks.
Summa already calls the posting owner's SIMD block intersection and reuses
position buffers; its phrase scorer drops deadline-free budget checks at setup.
The measured phrase hotspot is candidate scoring bounds, not callback allocation.
Cancellation remains enabled; no checks are removed for this experiment.

### Why the counts differ

All three reference engines agreed for all 826 input queries. Summa had 15 equal
counts, 595 different counts (583 higher, 12 lower), and 216 explicit errors.
The errors split into 158 unsupported wildcard/regex queries, 46 prefixes that
exceeded the existing 1,024-term expansion budget, and 12 query-syntax errors.
They are excluded, not assigned zero hits.

The canonical count collector and independent exhaustive scorer enumeration
agreed on all 156 audited full-corpus queries: every one of the 141 term queries
plus the 15 timed queries. This rules out an optimized-count discrepancy for
those queries; it does not prove all phrase semantics equivalent to Lucene.

Actual token streams from the benchmark Summa tokenizer and Elasticsearch's
standard analyzer explain large differences on Wikipedia markup:

| Input              | Summa terms              | Reference terms     |
| ------------------ | ------------------------ | ------------------- |
| `file:Example.jpg` | `file`, `example`, `jpg` | `file:example.jpg`  |
| `people’s world`   | `people`, `world`        | `people’s`, `world` |
| `the.com`          | `the`, `com`             | `the.com`           |
| `ＦＩＬＥ ﬁle`     | `file`, `file`           | `ｆｉｌｅ`, `ﬁle`   |

For example, `file` matches 536,622 Summa documents versus 82,102 in every
reference; `people` matches 928,191 versus 790,960. Disabling stemming, folding
and stop words does not make the two analyzers identical: Summa still splits
these boundaries and applies compatibility normalization.

A separate 22-document fixture exposes a sloppy-phrase semantic difference:
`"a a"~4` matches 13 Summa documents versus 2 reference documents. Summa's
existing anchored-window matcher may reuse one occurrence for repeated terms;
Lucene requires separate occurrences. Exact phrases agree on this fixture.
Changing these established semantics needs a separate matching/analyzer change;
the throughput optimization preserves current results and does not enlarge the
15-query comparison subset.

Raw fixture token streams, exact counts, and full-corpus audit output are retained
under `.context/yonik-benchmark/investigation/`. The benchmark's `audit` command
reproduces count-collector versus exhaustive-scorer comparisons without modifying
an index. Values in throughput tables are completed queries per second, not
matching documents per second or latency. `COUNT` requests an exact total;
`TOP_10` and `TOP_100` request ranked external IDs without an exact total.

RGB revealed an additional eligibility gap: plain reordered text uses a document
map for lengths, but phrase bounds previously accepted only an unmapped length
column. The revision permits document maps (one physical scoring unit per
logical document), retaining physical-length lookup and stable-ID heap ties.
Multi-value chunk maps remain excluded because their document folding requires
a different final-score bound. The regression first asserts this distinction;
full-corpus ranked audit compares optimized results to exhaustive scoring.

Validation for the runtime follow-up: the full serial `check` run passed 2,026
native tests (25 normally ignored), strict Clippy, native without sync, and
standalone broker compilation. WASM rebuilt and all 38 tests passed. The new
regressions cover every posting codec, long-length fallback, singleton lookup
score-bit identity, duplicate phrase starts, late winners, and document-map
versus multi-value-map eligibility. Existing deadline and ordinal tests remain
unchanged and pass. The RGB HTTP smoke test passes on Linux. The production RPC
`full` suite was not rerun; no RPC or lifecycle implementation changed.

The full RGB corpus audit matches all 15 exact counts and every top-100 score
bit against the original index. Shared external IDs have identical score bits;
all membership differences are ties at the cutoff. The fresh parallel build
assigns different internal document IDs, so equal-score tie order can differ
across the two independently built indexes. Within each index, optimized IDs
and score bits match exhaustive scoring exactly.

The initial diagnostic CPU profile overlapped compilation; its sample shares
identify candidate hot paths and are not an isolated speedup measurement. The
reported throughput runs exclude all indexing, reordering, compilation and
profiling work. RGB build elapsed time is informational: compilation overlapped
part of ingestion, and indexing paused briefly for the early A/B screen.

A tighter singleton-bound experiment was rejected. It used the exact canonical
TF=1 score and stronger unmapped ID ties, but an isolated 3×3s screen gave
186.8 → 186.7 QPS for the medium phrase and 475.6 → 483.1 QPS for low phrases.
That did not justify retaining the extra numerical special case. The final
source retains the conservative envelope described above.

### Completed first optimization and RGB follow-up

The same-host, eight-client, top-10 throughputs were:

| Family             | Original (QPS) | Optimized ordinary index (QPS) | Optimized RGB index (QPS) |
| ------------------ | -------------: | -----------------------------: | ------------------------: |
| Seven conjunctions |        9,173.5 |                        9,356.0 |                  15,596.1 |
| Seven low phrases  |          452.5 |                          478.4 |                     995.6 |
| One medium phrase  |          186.2 |                          188.6 |                     194.6 |

On the **same RGB index**, the prior binary delivered only 251.8 QPS for low
phrases and 51.2 QPS for the medium phrase. Restoring mapped-length bound
eligibility therefore recovers approximately 4× phrase throughput. All 45 HTTP
responses (15 queries × three operations) were identical between binaries on
each index. On the ordinary index the medium-phrase improvement is negligible
at eight clients; at one client it improved from 59.8 to 66.8 QPS. A subsequent
reverse-order baseline gave 442.4/184.8 QPS for low/medium phrases at eight clients.
The common phrase remains well behind the reference engines.

Ordinary/RGB index sizes are 14,532,580,307 / 12,924,541,349 bytes (RGB 11.1%
smaller). The fresh RGB build peaked at 23,898,960 KiB process RSS. Its 32m59s
elapsed time is not a controlled build comparison because some ingestion
coincided with compilation and a short pause. Query-process RSS is recorded per
cell in the linked CSV; it includes mapped pages and is not a heap-size estimate.

The retained implementation was checked in
`.context/search-harness/20260923T045954.798266Z-check`; all 2,026 native tests and
38 WASM tests passed. Raw evidence is saved locally in
`.context/yonik-benchmark/investigation-evidence.tar.gz`. The tighter-bound
prototype described above is excluded from these final numbers.

### Non-RGB competitive traversal experiment

The next experiment targets the remaining ordinary-index phrase gap. Currently
the ranked driver intersects postings before testing whether the candidate can
beat the heap. The proposed scorer operation receives only a conservative score
floor, scans original-first-term postings in bounded blocks, and aligns the
other terms only for competitive candidates. The original first term bounds
phrase frequency even with duplicate starts. Equal scores remain eligible; the
collector retains stable-ID tie decisions. Exact counts, nested scorers, score
arithmetic, position matching and persisted bytes remain unchanged. Deadline
checks occur at each bounded block and alignment boundary. The intended cost is
a contiguous frequency/length scan plus sparse conjunction probes, rather than
an intersection and dynamic dispatch per rejected candidate. This remains an
experiment until same-index timing and exact-result validation complete.

A separate non-RGB `index-impacts` build will use the existing canonical writer
with `posting_impact_bounds: true` (which includes ratio bounds), preserving
unquantized scoring lengths and ordinary document placement. This tests a
missing input to phrase block pruning without changing defaults or retrofitting
metadata into existing segments. Phrase traversal can consume the original
first term's frontier because phrase frequency cannot exceed that term's TF.
The first screen still uses the unchanged ordinary index to isolate traversal.

A second traversal experiment batches the two rarest posting lists' block
intersection into at most 128 index pairs. It retains both posting iterators on
the current match for canonical TF/position reads, invalidates the pair cache
on block changes, and skips cached pairs after monotone seeks. This amortizes
SIMD setup across a block and applies to exact phrase counts as well as ranking.
Scratch is 256 bytes of pairs plus constant cursor metadata per phrase.

The score admission prototype now caches, for TFs 1–32, the first length whose
existing conservative bound loses to the current heap score. Binary search
uses the unchanged bound calculation, and all equal-score candidates remain
eligible. Out-of-range TF/lengths use the scalar calculation. The table occupies
128 bytes plus its score key and is recomputed only on a threshold change.
Lists below 1,024 candidates retain the existing driver. For longer lists, a
first-term scan is tested only when it has at most four times the rarest term's
posting count; otherwise the scorer aligns the rarest pair and applies the same
length cutoff internally. Per-query timings motivated this cost comparison: an
unconditional first-term scan spent 21 ms on `"is preserved"` and 12 ms on
`"0 10px"` in the initial prototype. No result or count semantics depend on the
choice of traversal.

Exact phrase collection also advertises the existing 128-document batch
interface. Its shared scalar batch implementation invokes the concrete phrase
cursor and returns exact matches; count collection aggregates each batch rather
than dynamically dispatching once per hit. Scorer deadline checks remain inside
traversal and the collector checks cancellation before consuming a batch.

The next format-preserving experiment targets positional work in the CPU
profiles: reuse one TF decode/access for both a posting's position offset and
frequency, return reads wholly inside the cached position block before directory
lookup, and match two-term exact phrases directly in their original coordinates.
The latter must retain duplicate original-first starts, non-adjacent offsets,
`u32` overflow behavior, and deferred frequency counting. Rejected competitive
candidates need not reset phrase state until a candidate is returned. These
changes add no index metadata or unbounded scratch; existing bounded position
buffers and conservative score bounds remain authoritative. They require
same-index exact-result checks and paired timing before retention. Impact
metadata remains opt-in, per the user's decision.

A follow-up cutoff experiment initializes each TF entry on first use after a
threshold change, instead of running all 32 binary searches eagerly. It uses
the identical conservative predicate and bounds; no threshold update is delayed.
Zero marks an uncomputed entry because every computed cutoff is at least one.
The experiment trades one predictable branch per admitted TF check for avoiding
unused binary searches. It requires same-index timing before retention.

The lazy-cutoff screen was mixed: common-phrase top-10 fell from 2,141 to
1,928 QPS, while low-phrase top-10 rose from 2,606 to 2,927 QPS. The eager table
is retained for now. A further experiment uses an existing block maximum of
one as the score-admission TF bound for every posting in that block. This
avoids decoding TFs for blocks with no competitive documents and makes the
cutoff constant inside that scan. The callback consumes a conservative TF
bound, not an occurrence count; actual TFs and position cursors remain lazy and
unchanged for returned candidates. No new metadata or frequency assumption is
needed, including for zero-frequency postings.

The block-max-one experiment was rejected: common-phrase top-10 throughput
fell to 1,773 QPS and low-phrase top-10 to 2,478 QPS, from 2,141 and 2,606 with
the retained code. Its exhaustive top-100 audit and all 45 HTTP responses were
identical, but the extra block decision and changed scan did not improve this
workload. Neither this prototype nor lazy cutoff initialization is shipped.

### Final retained non-RGB implementation

The experiments above culminate in certified rare-term admission, bounded
intersection batches, eager TF cutoffs, cached position reads and direct exact
two-term matching. POS5/POS6 records the writer's per-document unique-position
certificate in one existing footer bit. Exact phrases with certified original
first positions may safely bound frequency by the rarest term. Other phrases
retain first-term bounds to preserve duplicate and slop semantics. Obsolete
position codecs and migration paths were removed; rebuild old indexes.

See the [final report and resource measurements](benchmark-results/searchbench-2026-09-23-phrases.md)
for all 27 fresh cells, paired correctness, remaining profiles and validation.
Impact metadata remains opt-in. The current-format RGB smoke passed; the earlier
full-corpus RGB throughput remains a distinct experiment.

Evidence collection is complete; the benchmark machine is confirmed `TERMINATED`.

### Remaining-gap experiments (September 23)

The next ranked-search experiment builds a bounded 128-bit candidate mask for
one decoded posting block using gathered scoring lengths and the existing exact
TF cutoff table. The mask is query-local, refreshed on block or threshold change;
monotone and physical seeks select bits at or after the current posting. It
introduces no persisted data or cross-query cache. Equal-score candidates remain
eligible and bounds/score arithmetic stay unchanged. The intended saving is to
hoist length-format selection and make admission a contiguous block operation.

For exact counting, investigate the repeated position-prefix reductions and
short-list copies in the retained profile. Any cursor optimization must retain
64-bit accumulated frequencies, zero-TF behavior, copied short-block boundaries,
backwards RGB probes, and lazy document-only decoding. These are experiments;
retain only improvements measured against the prior certified binary on the
same immutable index, with exhaustive top-100 and count comparisons.

A further ranked experiment uses the exact canonical score when phrase frequency
is bounded by one: a confirmed match then has exactly that frequency, so no
floating-point envelope inflation is needed. The ranked collector may reject
score equality only when there is no physical-to-logical mapping: remaining
monotone IDs cannot beat any equal-score ID already in its full local heap.
Mapped/RGB traversal keeps equality eligible. TFs above one retain inflated
bounds. The threshold's equality mode is part of the cutoff cache key. Tests
must cover mapped stable-ID ties, equal-score ordinary hits, and one-ULP floors.

The next screens reuse L1 group bounds in phrase admission, lazy per-block
64-bit position-prefix tables (129 entries per posting iterator), and seek-driven
intersection when term posting counts differ by more than fourfold. Group and
block pruning retain the canonical singleton/equality rule and existing
conservative ratio/impact envelopes. No persisted format or index default changes.
The impact-enabled corpus is a separate rebuilt variant; its physical IDs may
differ, so cross-index checks compare exact counts and score bits, and each
variant also checks its own IDs against exhaustive traversal.

A cutoff-construction screen seeds each integer search from the algebraic BM25
inverse. The estimate never decides admission: both adjacent integer lengths
are checked with the same canonical singleton or conservative bound predicate.
If they do not bracket the transition, bounded binary search completes it.
Zero length-normalization coefficients and non-finite estimates retain binary
search. This aims to reduce repeated table construction without changing a
single cutoff, threshold update, score bit, or persisted byte.

The final hardware screen compares a 16-lane AVX-512F admission kernel with
its 8-lane AVX2 equivalent on the same host. Both use identical integer TF/length
cutoffs, conservative uncommon-TF handling, and masked norm gathers. Runtime
feature detection retains the AVX2 and portable scalar paths; the tail reuses
AVX2. The existing every-lane/every-tail scalar comparison exercises both kernels.
Retain the wider kernel only if the complete query screen improves.

### Wildcard API follow-up (separate from the frozen phrase campaign)

The new [core wildcard query](wildcard-query.md) supports Unicode whole-term
`*`, `?` and escaping. The benchmark adapter now translates its three wildcard
families directly to this query type; regex remains unsupported. Existing prefix
expansion budgets remain in force, with a separate dictionary-scan budget for
leading wildcards. This feature is not present in the frozen binaries used by
the phrase/32-vCPU campaign, and it does not enlarge their 15-query subset.
Analyzer, sloppy-phrase and query-syntax compatibility remain separate work.

### Path to a complete 826-query comparison

The original full-corpus audit is immutable evidence: 15 agreeing queries,
595 count mismatches, and 216 errors (158 missing wildcard/regex operators,
46 over-budget prefixes, 12 syntax errors). New wildcard support makes the
145 wildcard-family expressions representable; it does **not** prove agreement
or remove expansion-limit failures. The 13 regex expressions remain a separate
query capability. The frozen phrase/scaling campaign still uses the original
15-query gate.

The next prerequisite is a named standard-analysis profile that reuses the
existing tokenizer machinery but preserves the reference token boundaries,
apostrophes and compatibility characters. Compare actual reference token streams,
positions, lowercasing and long-token behavior on the saved counterexamples before
rebuilding the ordinary and RGB indexes from the unchanged corpus. Do not claim
that disabling stemming/folding in today's lexical tokenizer is sufficient.

Then address sloppy phrase occurrence reuse, term transpositions and frequency
semantics; translate escaped literal terms without accidentally re-analyzing or
splitting them; add the required whole-term regexp language; and replace or expose
bounded multi-term execution capable of the benchmark's broader expansions.
Do not obtain coverage by silently truncating expansions or changing the corpus.
Reprobe all 826 exact counts after rebuilding, retaining explicit exclusions and
errors until every cell's semantics are established. Equal counts alone are not
a ranking oracle: preserve separate scorer/ID/score-bit checks within Summa.

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

### Completed 32-client comparison

The [32-vCPU host report](benchmark-results/searchbench-2026-09-23-32cpu.md)
records all seven variants, 63 error-free cells, memory, correctness and operational
recovery. Ordinary medium-phrase top-10 improves 45.4% over prior Summa; optional
impacts exceed Luxir on that cell but regress low-phrase top-100. RGB has separate
wins and regressions. Defaults remain unchanged, and coverage remains 15/826.

### Conjunction top-100 investigation

The completed [conjunction follow-up](benchmark-results/searchbench-2026-09-23-conjunctions.md)
traces the HTTP envelope through core planning, the shared search pool, ranked
conjunction execution, ID-column lookup and JSON encoding. It uses the unchanged
10M-document ordinary index and separate uninstrumented timing and instrumented
work-counter builds. Exact counts, ranked IDs, score bits and response bytes
are preserved. All 24 timing cells pass without errors.

At 32 clients, moving response encoding/destruction to the bounded worker raises
top-100 from 21,036 to 33,246 QPS with two HTTP workers. Automatic sizing (four
HTTP workers on 30 available CPUs) reaches 32,720 top-100 and 68,524 count QPS,
versus Luxir's 41,003 and 64,756 respectively. Automatic top-10 is roughly
unchanged from baseline. Single-client throughput regresses by 2.3–19.4% across
these three operations, so this is a throughput tradeoff, not a universal gain.
Only seven count-compatible conjunctions are timed; all 15 original admitted
queries are correctness-audited. The full 826-query prerequisites remain above.

The initial screen isolates an HTTP-reactor bottleneck: response JSON encoding
and destruction of its temporary tree previously ran after returning from the
CPU worker. Both now remain under the existing admission permit in that worker,
which returns only finished JSON bytes to the reactor. This preserves response
bytes and keeps search algorithms unchanged. An explicit bounded `HTTP_WORKERS` argument configures
the benchmark frontend (1–64). Its serving default scales
with available logical CPUs: one HTTP worker on one CPU, otherwise
`clamp(ceil(available_cpus / 8), 2, 8)`. `available_parallelism` accounts for Linux
affinity and supported CPU quotas. A detection failure
logs a warning and uses one CPU. Explicit HTTP worker counts override this
policy; indexing/audit diagnostics retain two runtime workers. Startup logs show
the detected CPU and selected worker counts. This is a benchmark frontend policy,
not a change to production gRPC or core search defaults. The report compares an
explicit two-worker configuration with the automatic default (four on the 30-CPU
server affinity); search-pool size and admission remain separately bounded. The diagnostic
command uses the existing parser, Searcher and shared response projection, with
100 samples after 20 warmups per top-k value. Work-counter builds are separate
from uninstrumented throughput. `maxscore_heap_updates` counts successful
ScoreCollector pushes/replacements, including conjunctions, under the existing
opt-in `query-diagnostics` feature; it is absent from normal builds.

Run stage diagnostics without instrumentation for timings, and separately with
work counters (the feature changes query overhead):

```sh
cargo run --release -p summa-server --example searchbench_http -- diagnose INDEX QUERIES.jsonl 30
cargo run --release -p summa-server --example searchbench_http --features query-diagnostics -- diagnose INDEX QUERIES.jsonl 30
# Automatic HTTP worker count, then explicit override:
cargo run --release -p summa-server --example searchbench_http -- serve INDEX 9401 30
cargo run --release -p summa-server --example searchbench_http -- serve INDEX 9401 30 2
```

Each input line uses the existing HTTP envelope (`query`, `class`); diagnostics
measure top-10 and top-100 through the same parser/searcher/projector, including
separate serialization and response-tree destruction. They do not include HTTP
transport, concurrent scheduling or request/collector destruction and are not an
end-to-end latency decomposition. `--families and_high_low` on the campaign
runner selects the already count-compatible conjunctions without widening the
original gate.

### Borrowed-ID responses and remaining scheduling costs

The completed [response and handoff follow-up](benchmark-results/searchbench-2026-09-23-handoffs.md)
replaces temporary per-hit JSON maps and owned ID strings with a typed response
borrowing the same fast-field values. One bounded vector retains the original
rank order until serialization finishes inside the admitted worker. Exact JSON
bytes, exhaustive ranked IDs/score bits/counts, index bytes and core algorithms
are preserved. CPU-based HTTP defaults and pool/admission limits are unchanged.

The fresh paired run improves conjunction top-100 from 33,148 to 38,909 QPS at
32 clients (+17.4%) and from 40,016 to 47,107 at 64 clients (+17.7%). Top-10
improves 6.7% and 2.9% respectively. Counts regress 4.9% and 3.7%; retain that
tradeoff. All 15 originally admitted queries are measured at 32 clients; the
seven conjunctions additionally run at one and 64 clients. Higher-concurrency
results are labeled separately and do not replace the 32-client comparison.
The report retains fresh Luxir results, memory, CPU cost and repetition ranges.

Instrumented top-100 probes show projection plus encoding/destruction falling
from about 156 to 65 microseconds at 32 clients, with segment-search time roughly
unchanged. Blocking-worker and shared-search-pool handoffs remain measurable.
These diagnostic wall times include instrumentation/warmup and are not production
latency measurements. Core-pool diagnostics belong to the Searcher; no benchmark-only
public execution API, pool bypass or second executor is introduced. The production
async path has a different dispatch boundary. A future scheduling change must
preserve bounded capacity, reader/permit ownership, cancellation, panic handling
and shutdown before its performance is compared.

With `summa-server/query-diagnostics` enabled, the benchmark-only
`GET /diagnostics` endpoint reports completed-handler/failed-worker counts and nine
fixed-size timing sums/maxima. It retains no per-request records or query text.
HTTP timings cover blocking entry/return, parse, search, projection and encoding
plus response destruction. Core timings cover shared-pool entry/return and summed
segment work; these are nested within search time and must not be added to it.
The instrumentation itself affects execution and aggregate wall times include
warmup in the diagnostic probes. Native Searcher capture records completed pool
installs on the caller, keeping worker-scope ownership and count semantics intact.
The endpoint is absent from normal builds and does not modify `/stats` or the
production protocol.

### Worker-count investigation

The completed [worker study](benchmark-results/searchbench-2026-09-23-workers.md)
uses the frozen borrowed-ID binary to cross 8/15/30 search/blocking workers with
2/4/8 HTTP workers. Three repeated 30/4 controls remain closely grouped. Smaller
worker widths hurt ranked throughput, and eight HTTP workers do not improve the
mix. The `WORKERS` argument couples both CPU-pool sizes, so the sweep does not
isolate Rayon from Tokio blocking capacity.

A longer all-15-query comparison confirms an explicit two-HTTP-worker tradeoff:
conjunction top-10 improves 8.9% and top-100 4.9%, but count drops 18.8%. Top-100
is nearly tied with fresh Luxir (42,720 versus 42,887 QPS). Phrase ranked gains
are smaller or absent, while phrase counts are effectively unchanged. Keep the
CPU-derived HTTP default and existing override; this is not a general policy
change based on one host. Core/runtime implementations, admission and persisted
formats are unchanged.

An alternating before/borrowed/borrowed/before count repeat does not reproduce
the previous 4.9% throughput loss at 32 clients: average session medians are both
76.64k QPS. A small 1.5% CPU-cost increase remains unassigned. The report retains
the original result, the new ranges, the count-only CPU-efficiency lead from
15 workers, memory, and the remaining dispatch/cancellation requirements.

### Dispatch, sparse ID directory, and envelope ownership experiment

The completed [dispatch and lookup study](benchmark-results/searchbench-2026-09-23-dispatch-directory.md)
preserves the existing search owner, scores, response bytes, 64-request admission,
and all persisted formats. It implements three changes:

- An explicit benchmark `serve ... [DISPATCH]` override (`blocking`, the
  unchanged default, or `in-place`). The latter uses Tokio `block_in_place`
  around the same admitted parse/search/project/encode closure. Tokio may still
  transfer its runtime core through the blocking pool; this is not zero-hop
  execution. Both modes retain the reader and permit until work finishes,
  translate panics into HTTP 500, and drain started work on runtime shutdown.
- Fast-field block-header discovery through a bounded sparse directory in
  the owning reader, without adding a benchmark-only public API. At most 256
  checkpoints (three u32 values each, 3 KiB) cover all single-value blockwise-linear
  streams in one reader. Existing validation precedes directory construction;
  encoded values and dictionaries stay file-backed. Heap metadata accounting
  includes the directory; segment heap totals also include fast-field block
  metadata (existing lazy text-dictionary tables remain outside that estimate).
  Binary search finds the preceding checkpoint, followed
  by at most roughly ceil(total codec blocks / 256) header advances. Other codecs
  and multivalue paths retain their existing semantics. No cache grows on queries.
- Validated query/class strings move out of the consumed JSON envelope, preserving
  validation order and errors while removing two string-buffer allocations.
  Borrowed `get_mut` lookups also avoid the two temporary key allocations caused
  by mutable JSON indexing; a counting-allocator regression test proves zero
  allocations for successful envelope conversion.

Validation covers malformed envelopes, missing and multivalue fields,
merged local dictionaries, block/tail boundaries, byte identity, cancellation,
panic, saturation and shutdown; native/full and WASM checks; paired throughput,
CPU and memory on the same fixture, machine, compiler and flags. The directory
is general reader metadata used by existing numeric/text accessors, not a second
ID decoder in the HTTP adapter.

The corrected paired run improves conjunction top-100 by **4.1%** (37,927 to
39,493 QPS) with **3.8% less CPU per request**. Separate projection probes show
about **43% less ID lookup time**. Top-10 improves 1.3% and count is effectively
unchanged. In-place dispatch loses 2.5% top-100 throughput and adds 7.8% CPU cost
relative to updated blocking execution; some phrase workloads improve. Keep it
opt-in and retain the CPU-derived HTTP-worker default. Coverage remains 15/826,
and the report preserves prototype controls, repetition ranges and memory.

## Shared-input comparison mode

`campaign.py gate --comparison shared-input` admits a query when every engine
executes it successfully, while retaining each engine's own count. The default
`exact-count` mode still requires identical counts. Neither mode discards failed
attempts from the gate report. A timing run validates counts and ranked hit
cardinality against the selected engine's own probe, and records the gate mode
in its context. Shared-input replay keeps a separate metrics bucket per query
so selectivity outliers remain inspectable.

After collecting all four `*-counts.json` files in the results directory:

```sh
python3 scripts/searchbench/campaign.py gate \
  --searchbench "$SEARCHBENCH" --out "$RESULTS" --comparison shared-input
```

Use that same `agreement.json` for each engine's `measure` invocation; the
measurement derives eligibility from the saved gate and validates against the
selected engine's count. Keep the query source, topology, affinity, limits and
timing settings identical across engines. The report labels shared-input
coverage separately from exact-count agreement.

This expands coverage without rewriting the query corpus, but different counts
mean different logical work. Report per-family query counts and count differences
alongside throughput; do not present this as a scoring-equivalence test. On the
September 24 full 10M probe, all three reference engines agree on all 826 counts.
Both Summa layouts execute 684 queries; 619 are within 5% of the reference count,
and the median relative difference is 0.34%. Only 15 match exactly. The other 142
Summa requests fail explicit dictionary-scan or term-expansion limits; they
remain failures in the 826-query coverage table and have no successful-query QPS.

A derived matched-workload suite could instead pair queries by operator shape,
term frequencies and result selectivity. Such queries must be retained and
labelled as a separate workload. Matching hit counts alone is insufficient for
phrases and Boolean intersections, whose candidate work can differ greatly.
Replacing query text cannot make a whole-field regex or leading-wildcard scan
fit a dictionary-scan limit smaller than that field's vocabulary; complete
successful coverage of those families also requires an execution improvement
or an explicitly documented resource-budget configuration.

The [September 25 closing-gap checkpoint](benchmark-results/closing-gap-2026-09-24/README.md)
adds same-index execution ablations and an explicitly separate Unicode-word
analyzer build. The latest full-corpus probe succeeds on 677/826 queries, with
640 exact reference counts and all successful counts within 5%; the other 149
remain visible resource-limit errors. Its 673-query timing matrix and four
separately timed newly admitted regex queries must not be combined into a
single throughput number. Final-source corpus retiming is still pending.
