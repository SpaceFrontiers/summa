# Same-host Searchbench throughput: borrowed-ID responses and search-pool handoffs

Count agreement: **15/826 queries**. This is a restricted workload, not the complete published benchmark.

10M Wikipedia chunks; one merged segment; 32-vCPU host (16 physical cores with SMT), 30 server hardware threads and two driver threads on the reserved physical core. Query and request caches disabled. Thirty-second session warmup, full untimed validation, one-second connection warmup, three ten-second repetitions per cell.

All 15 count-compatible queries are measured at 32 clients; the seven conjunctions are also measured at one and 64 clients. All three variants run sequentially in isolated loopback networking. Both Summa variants use the same ordinary index, automatic HTTP workers (four), 30 blocking/search workers and 64-request admission. RGB and impacts remain separate. Server CPUs: 0–14,16–30; driver CPUs: 15,31. No copying, compilation or indexing overlaps timing. Core search algorithms are unchanged.

Summa uses a benchmark HTTP frontend over core, not its production gRPC service. Text-analysis differences exclude many queries; equal corpus counts do not prove general analyzer or relevance equivalence. Luxir uses the official 0.1.0 x86-64-v4 release, not the article's local build. No p99 claim is made.

| Family             | Included | Published queries |
| ------------------ | -------: | ----------------: |
| and_high_high      |        0 |                47 |
| and_high_low       |        7 |                50 |
| and_high_med       |        0 |                50 |
| high_phrase        |        0 |                30 |
| high_sloppy_phrase |        0 |                 7 |
| high_term          |        0 |                45 |
| low_phrase         |        7 |                50 |
| low_sloppy_phrase  |        0 |                37 |
| low_term           |        0 |                47 |
| med_phrase         |        1 |                46 |
| med_sloppy_phrase  |        0 |                27 |
| med_term           |        0 |                49 |
| or_high_high       |        0 |                42 |
| or_high_low        |        0 |                46 |
| or_high_med        |        0 |                45 |
| prefix3            |        0 |                50 |
| regex              |        0 |                13 |
| wildcard           |        0 |                49 |
| wildcard_lead      |        0 |                47 |
| wildcard_scan      |        0 |                49 |

**Throughput (queries/second; higher is better).** Values are the median of three repetitions. The CSV retains repetition min/max and peak process RSS. Raw replay JSON and memory samples accompany each cell.

| Family       | Operation | Clients | Distinct queries | Summa before (QPS) | Summa borrowed IDs (QPS) | Luxir (QPS) |
| ------------ | --------- | ------: | ---------------: | -----------------: | -----------------------: | ----------: |
| and_high_low | COUNT     |       1 |                7 |            2,486.0 |                  2,344.0 |     3,032.6 |
| and_high_low | COUNT     |      32 |                7 |           70,461.6 |                 67,035.1 |    65,114.9 |
| and_high_low | COUNT     |      64 |                7 |           70,138.2 |                 67,562.1 |    69,671.9 |
| and_high_low | TOP_10    |       1 |                7 |            2,246.7 |                  2,463.2 |     2,465.2 |
| and_high_low | TOP_10    |      32 |                7 |           41,690.7 |                 44,494.0 |    52,338.7 |
| and_high_low | TOP_10    |      64 |                7 |           53,978.6 |                 55,548.8 |    55,967.4 |
| and_high_low | TOP_100   |       1 |                7 |            1,464.3 |                  1,941.7 |     1,510.5 |
| and_high_low | TOP_100   |      32 |                7 |           33,147.9 |                 38,908.8 |    41,582.1 |
| and_high_low | TOP_100   |      64 |                7 |           40,015.5 |                 47,106.9 |    44,166.8 |
| low_phrase   | COUNT     |      32 |                7 |            2,011.8 |                  1,999.3 |     2,054.6 |
| low_phrase   | TOP_10    |      32 |                7 |           15,694.0 |                 15,656.5 |     7,450.8 |
| low_phrase   | TOP_100   |      32 |                7 |            4,785.4 |                  4,882.1 |     3,779.9 |
| med_phrase   | COUNT     |      32 |                1 |              719.3 |                    714.3 |       605.6 |
| med_phrase   | TOP_10    |      32 |                1 |           15,026.5 |                 15,475.7 |    17,573.9 |
| med_phrase   | TOP_100   |      32 |                1 |            9,723.4 |                 10,261.7 |     5,972.0 |

## Measured result and remaining gap

At 32 clients, borrowed-ID responses improve conjunction top-100 **17.4%**
(33,148 → 38,909 QPS) and top-10 **6.7%** (41,691 → 44,494). Fresh Luxir
reaches 41,582 and 52,339 respectively: the top-100 gap falls from **20.3% to
6.4%**, while the top-10 gap remains **15.0%**. This isolates response allocation
as a material part of the apparent search-engine gap, without changing posting
decoding, scoring, pruning or the fixture.

Conjunction count throughput regresses **4.9%** at 32 clients (70,462 → 67,035),
although it remains **2.9%** above Luxir (65,115). The initial shorter screen also
showed a smaller count regression. Count responses do not perform ID projection,
so ranked improvements do not justify claiming a universal improvement. Preserve
this finding for further attribution; the response-shape change is retained for
its repeatable ranked-query gains.

Single-client medians improve **9.6%** for top-10 and **32.6%** for top-100;
count regresses **5.7%**. Absolute single-client throughput changes substantially
between the screen and final campaign. The CSV retains all repetition ranges;
these short sequential runs do not establish an equally precise latency ranking.

At 32 clients, low-frequency phrase top-10 remains roughly unchanged against
baseline (−0.2%), top-100 improves 2.0%, and count changes −0.6%. Medium-frequency
phrase top-10 improves 3.0%, top-100 5.5%, and count changes −0.7%. These smaller
changes do not establish a uniform phrase improvement; the response work matters
most where returning many IDs accounts for a larger share of request cost.

At 64 clients, conjunction top-100 reaches **47,107 QPS** versus Luxir's
**44,167** (+6.7%). Top-10 is approximately tied (55,549 versus 55,967), while
count remains below Luxir (67,562 versus 69,672, −3.0%). Moving from 32 to 64
clients raises borrowed-ID top-100 throughput 21.1% without another code change.
This separate saturation check shows that the 32-client gap includes utilization
and scheduling effects; it does not replace the requested 32-client comparison
or establish a general engine ranking.

## CPU and memory

CPU equivalents are server CPU seconds divided by elapsed wall seconds, not physical core counts. CPU cost is the median of three repetitions. RSS includes faulted mmap pages; anonymous RSS is not an allocator-exact heap measurement. Memory peaks cover all 15 cells per variant, so they are not directly comparable with the previous conjunction-only campaign.

| Conjunction operation | Clients | Variant | Busy CPU equivalents | CPU µs/request |
| --------------------- | ------: | ------- | -------------------: | -------------: |
| TOP_10                |      32 | before  |                24.78 |          594.3 |
| TOP_10                |      32 | typed   |                25.28 |          568.0 |
| TOP_10                |      32 | luxir   |                28.90 |          551.5 |
| TOP_10                |      64 | before  |                28.54 |          528.7 |
| TOP_10                |      64 | typed   |                28.63 |          515.0 |
| TOP_10                |      64 | luxir   |                29.99 |          534.8 |
| TOP_100               |      32 | before  |                24.81 |          748.5 |
| TOP_100               |      32 | typed   |                25.25 |          648.9 |
| TOP_100               |      32 | luxir   |                29.22 |          703.3 |
| TOP_100               |      64 | before  |                27.89 |          697.4 |
| TOP_100               |      64 | typed   |                28.17 |          597.5 |
| TOP_100               |      64 | luxir   |                30.00 |          679.2 |
| COUNT                 |      32 | before  |                23.68 |          335.5 |
| COUNT                 |      32 | typed   |                22.76 |          339.6 |
| COUNT                 |      32 | luxir   |                28.73 |          441.6 |
| COUNT                 |      64 | before  |                23.62 |          336.7 |
| COUNT                 |      64 | typed   |                22.95 |          339.7 |
| COUNT                 |      64 | luxir   |                29.99 |          430.4 |

| Variant | Peak process RSS MiB | Peak anonymous RSS MiB |
| ------- | -------------------: | ---------------------: |
| before  |               1155.2 |                  113.6 |
| typed   |               1153.3 |                  111.3 |
| luxir   |                161.0 |                   15.1 |

## Response and handoff measurements

Top-100 conjunction diagnostic means, in microseconds. Each column is a separate fresh instrumented server; these are wall-time observations including warmup, not throughput or production latency measurements. Shared-pool and segment values are nested within search.

| Stage            | Before, 1 client | Borrowed IDs, 1 client | Before, 32 clients | Borrowed IDs, 32 clients |
| ---------------- | ---------------: | ---------------------: | -----------------: | -----------------------: |
| blocking_queue   |            11.78 |                  15.21 |              53.77 |                    49.76 |
| parse            |            13.34 |                  12.52 |              22.99 |                    20.40 |
| search           |           325.61 |                 323.57 |             534.84 |                   521.86 |
| project          |            78.15 |                  50.74 |             130.24 |                    56.80 |
| encode_and_drop  |            12.23 |                   4.47 |              25.29 |                     8.40 |
| blocking_return  |            28.88 |                  28.06 |              60.63 |                    57.34 |
| core_pool_queue  |            18.51 |                  17.56 |              40.28 |                    33.58 |
| core_pool_return |            12.88 |                  13.98 |              64.66 |                    55.40 |
| segment_work     |           291.91 |                 289.69 |             425.37 |                   428.45 |

A separate uninstrumented stage command takes 100 samples after 20 warmups per query/top-k pair. Below are arithmetic means of the seven per-query medians, in microseconds; stages exclude concurrent HTTP transport and are not an end-to-end latency decomposition.

| Limit | Variant | Parse | Search | Project | Serialize | Response drop |
| ----: | ------- | ----: | -----: | ------: | --------: | ------------: |
|    10 | before  | 10.05 | 214.60 |    8.19 |      0.61 |          0.71 |
|    10 | typed   |  8.33 | 198.75 |    3.49 |      0.52 |          0.04 |
|   100 | before  | 10.36 | 229.75 |   61.60 |      4.98 |          6.55 |
|   100 | typed   |  8.30 | 215.91 |   30.76 |      2.91 |          0.05 |

For conjunction top-100 at 32 clients, borrowed-ID Summa now consumes **649 CPU
microseconds/request** versus Luxir's **703**, yet uses only **25.25** CPU
equivalents versus **29.22**. At 64 clients it reaches **28.17** equivalents and
**597 µs/request**, versus Luxir's **30.00** and **679 µs/request**. The remaining
32-client throughput deficit therefore cannot be described simply as more core
work per query. Higher concurrency improves utilization and amortizes scheduling
cost, while the measured entry/return delays make dispatch a concrete next
investigation. CPU microseconds include all server threads, not just scoring.

Memory remains a material difference: borrowed responses reduce peak anonymous
RSS only from 113.6 to 111.3 MiB in this campaign, versus Luxir's 15.1 MiB.
Summa' approximately 1.15 GiB process RSS includes mapped index pages. Removing
per-hit temporary allocations is not a solution to the broader residency gap.

A post-campaign count probe uses separate instrumented servers with the same
three-second warmup and ten-second measurement. At 32 clients, parse time rises
from 22.05 to 28.14 µs, while search stays near 258–259 µs and projection plus
encoding/drop decreases from 3.15 to 2.41 µs. No shared-pool install is recorded:
this adapter's count path calls the async segment collector directly. Its zero
segment-work counter is a capture-path limitation, not zero search work. The
probe does not attribute the regression to serialization or added Rayon work;
parsing/scheduling variance and code-generation effects remain to investigate.
The uninstrumented count regression remains the result used for comparison.

Luxir's same-process health control completes without errors at a median
338,865 requests/sec. It is a separate `GET /health`, c32/t4 measurement; its
client CPUs (14–15,30–31) overlap the server allocation on 14,30. Search replay
uses only the disjoint 15,31 pair. The health control is retained for transport
context, not used to normalize query throughput or make a p99 claim.

## Implementation and invariants

The benchmark HTTP adapter now serializes a typed response holding borrowed
external-ID strings. It still calls the same fast-field reader for each retained
hit, in the same rank order. A bounded ID vector replaces per-hit JSON maps,
owned string copies and the outer JSON-tree conversion. Count responses use a
typed `{docs, found}` shape. Serde performs escaping and number serialization.
The reader remains owned by the admitted blocking closure until the bytes are
finished; no borrowed data crosses the response boundary.

The old and new response encodings are byte-identical on all 45 admitted-query
HTTP cases and on unit cases covering Unicode, quotes, backslashes, control
characters, empty/duplicate IDs, empty results and extreme exact counts. Both
Summa variants preserve all 15 exhaustive top-100 document IDs, score bits and
counts. Every timed cell separately validates exact counts or ranked result
cardinality/unique IDs, with topology checked before and after each sweep.
Counts agreeing across engines do not establish ranking/analyzer equivalence.

Scoring, pruning, search-pool ownership, request admission, CPU-based HTTP worker
selection and persisted index formats are unchanged. Four HTTP workers, 30
blocking workers, 30 shared search workers and a 64-request admission limit apply
to both Summa variants. No indexing or corpus rewriting occurs. RGB and impact
indexes remain separate; impacts remain disabled. Normal builds do not enable
query diagnostics.

Before and after were built on the same 8-vCPU build machine during one boot, with
the pinned Rust 1.98.1 toolchain, release mode and `-C target-cpu=native`. All
transfers and compilation finished before timing on the 32-vCPU Cascade Lake
benchmark machine. The standard baseline includes the same diagnostic scaffolding
compiled out, isolating the response representation change. This is a fresh
paired comparison, not a comparison against selected numbers from an earlier
boot or session.

## What the diagnostic timings mean

The feature-enabled HTTP adapter captures wall time before dispatching to the
Tokio blocking worker and after the handler resumes. Inside that worker it
captures parsing, search, projection and encoding/destruction. The Searcher owns
separate shared-pool entry/return counters. Existing segment-work capture retains
its ownership and merge rules. Completed pool installs count on the calling
capture; direct async count collection does not create an install and invalid
windows fail before pool admission.

`GET /diagnostics` exists only with `summa-server/query-diagnostics`. Storage is
bounded to nine sums/maxima plus completion/error counters, without per-request
records or query text. Successful-handler totals exclude canceled handlers;
these probes complete without worker errors. Each diagnostic case uses a fresh
server, three seconds of warmup and ten seconds of replay; timing totals include
both periods. Core-pool and segment values are nested within search time and
must not be added to it a second time. Instrumentation affects execution, and
these aggregate means are not uninstrumented end-to-end latency measurements.

The returned-ID allocation reduction explains a concrete improvement without
changing decoded postings or BM25 evaluation. The handoff measurements identify
another cost to investigate; they do not prove that all waiting can be removed
or that it has the same magnitude in production gRPC. The production async
search path uses a different entry into the shared pool. Any future dispatch
change must preserve bounded worker capacity, reader/permit ownership through
cancellation, panic handling and shutdown; this follow-up adds no new executor
or benchmark-only public core execution API.

ID-column blockwise-header traversal remains in response projection. A future
batch or directory design must live in the reader/codec owner, account for
resident metadata or bounded scratch, and preserve missing/multi-value behavior.
Earlier AND-pruning experiments already found cases where fewer decoded blocks
lost to added bound-check overhead; this investigation does not enable broader
pruning from one throughput result. See
[the earlier pruning experiments](../maxscore-text-reordering.md#earlier-ranked-and-pruning-experiments).

## Scope and validation

Only the original 15/826 count-compatible queries qualify: seven conjunctions,
seven low-frequency phrases and one medium-frequency phrase. All are timed at
32 clients; only conjunctions are additionally timed at one and 64 clients.
Full 826-query coverage still needs the analyzer, sloppy-phrase, escaped-literal,
regex and bounded-expansion work described in
[the comparison design](../searchbench-comparison.md#path-to-a-complete-826-query-comparison).
No general relevance, cold-cache, concurrent ingestion/merge, ARM-throughput,
production-gRPC or p99 claim is made.

The final Rust tree passes `python3 scripts/check_search.py check` under
`.context/search-harness/20260923T113909.147943Z-check`: 2,029 native tests pass,
25 are normally ignored, and formatting, strict Clippy, native-without-sync and
standalone broker checks pass. Two example tests cover CPU defaults and JSON
byte equivalence; the feature-only core diagnostic integration and feature-enabled
strict Clippy pass. Diagnostic HTTP, ordinary HTTP and RGB/explicit-worker HTTP
smoke tests pass, including errors, concurrency and shutdown. The WASM release
build and all 39 WASM JavaScript tests pass. Full production RPC/lifecycle tests were not
rerun; no production protocol or lifecycle mechanism changed.

An initial local example compile failed because the new serializer required a
direct `serde` dev-dependency; the manifest/lock entry was corrected using the
already pinned workspace version. The subsequent tests and builds pass. The
build machine's cloud stop command lost its polling connection, but independent cloud
status confirmed shutdown; no timed sample was affected.

The source/build archive is downloaded and SHA-256 verified as
`84ef63658c8af5f43adc21e1a25bb13afeaeb392350220ff9a51a462472ea51b`.
Standard before executable:
`0938111c46ebf91d838c9857166a61a5aa646d4dcee1a02ed125ff993fb8f4be`.
Standard borrowed-ID executable:
`61f485a0ac1964389f4ec01b87b855fc8155911972f0c9f1b339d402b241c778`.
The separate instrumented binaries, raw timings, source snapshots and reproduction
scripts are retained under `.context/yonik-benchmark/handoffs/`.

All **45 cells / 135 repetitions** complete without request or memory-sampling
errors. The final evidence archive is downloaded, size-checked and SHA-256
verified as `382ba48a577a49b034d6aee275606e6e497472ec0fd84dc52c978f4e05d7aadb`
(766,389 bytes). It contains the raw replay JSON, memory timelines, exact audits,
response bytes, diagnostic snapshots, hardware/posture records and reproduction
scripts. The build/source archive is separate as recorded above. No new `perf`
profile was taken in this follow-up; earlier raw profiles remain in the
[conjunction report](searchbench-2026-09-23-conjunctions.md).

Temporary transfer keys are removed and host restrictions are restored. Both
cloud stop commands lost their polling connection; independent cloud status
confirms **both machines `TERMINATED`** after evidence verification. No benchmark work
remains running. Final documentation links, ownership contracts, Python checks
and `git diff --check` pass.

The subsequent [worker/count-repeat study](searchbench-2026-09-23-workers.md)
does not reproduce the 32-client count-throughput regression in an alternating
before/borrowed/borrowed/before run. Both average session medians are 76.64k QPS;
a roughly 1.5% CPU-cost increase remains. The earlier measurements above are
retained, and the follow-up does not establish single-client behavior.
