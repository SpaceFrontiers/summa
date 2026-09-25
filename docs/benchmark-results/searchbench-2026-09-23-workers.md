# Same-host Searchbench throughput: worker configuration and count repeatability

Count agreement: **15/826 queries**. This is a restricted workload, not the complete published benchmark.

10M Wikipedia chunks; one merged segment; 32-vCPU host (16 physical cores with SMT), 30 server hardware threads and two driver threads on the reserved physical core. Query and request caches disabled. Thirty-second session warmup, full untimed validation, one-second connection warmup, three ten-second repetitions per cell.

All 15 count-compatible queries are measured at 32 clients. Both Summa configurations use the same frozen borrowed-ID executable and ordinary index. Current: 30 search/blocking workers, four HTTP workers. Candidate: 30 search/blocking workers, 2 HTTP workers. Admission remains 64. The WORKERS argument couples search and blocking pool sizes; this is not an isolated ablation of either pool. No runtime implementation or default changes. Server CPUs: 0–14,16–30; driver CPUs: 15,31. No copying, compilation or indexing overlaps timing.

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

| Family       | Operation | Clients | Distinct queries | Summa current workers (QPS) | Summa candidate workers (QPS) | Luxir (QPS) |
| ------------ | --------- | ------: | ---------------: | --------------------------: | ----------------------------: | ----------: |
| and_high_low | COUNT     |      32 |                7 |                    76,093.9 |                      61,823.7 |    66,996.2 |
| and_high_low | TOP_10    |      32 |                7 |                    46,628.4 |                      50,785.2 |    53,968.9 |
| and_high_low | TOP_100   |      32 |                7 |                    40,727.8 |                      42,720.1 |    42,887.0 |
| low_phrase   | COUNT     |      32 |                7 |                     2,039.1 |                       2,038.2 |     2,078.0 |
| low_phrase   | TOP_10    |      32 |                7 |                    16,241.1 |                      16,977.5 |     7,617.6 |
| low_phrase   | TOP_100   |      32 |                7 |                     5,148.0 |                       5,165.1 |     3,855.9 |
| med_phrase   | COUNT     |      32 |                1 |                       727.5 |                         726.9 |       606.8 |
| med_phrase   | TOP_10    |      32 |                1 |                    15,759.5 |                      16,726.4 |    17,656.0 |
| med_phrase   | TOP_100   |      32 |                1 |                    10,511.7 |                      10,870.7 |     5,980.9 |

## Result and decision

Worker tuning exposes a workload tradeoff, not a general replacement for the
current default. In the independent final comparison, two HTTP workers improve
conjunction top-10 from 46,628 to 50,785 QPS (+8.9%) and top-100 from 40,728 to
42,720 (+4.9%). The count penalty must be weighed against those ranked gains.
The CPU-based default and existing explicit worker override remain unchanged.

The screen's three 30/4 anchors are closely grouped: top-10 spans
47,224–47,381 QPS, top-100 41,542–41,731, and count 76,931–78,004. Relative
configuration differences are much larger than this within-screen drift.
However, this fresh-boot baseline is faster than the preceding session with the
same binary, so cross-session absolute differences are not optimization gains.

Reducing the coupled search/blocking width to 15 loses roughly one fifth of
ranked throughput; reducing it to eight loses roughly two fifths. Counts respond
differently: 15 workers with two HTTP workers approximately match the 30/4 count
throughput in the short screen. Raising HTTP workers to eight with width 30
reduces all three operation rates. This rules out simply adding HTTP workers or
halving both CPU-worker limits as a broad solution on this workload.

The longer run confirms the count cost: conjunction count falls from 76,094 to
61,824 QPS (−18.8%) with two HTTP workers. Low-frequency phrase top-10 improves
4.5%, top-100 changes +0.3%, and count is effectively unchanged. Medium-frequency
phrase top-10 improves 6.1%, top-100 3.4%, and count is effectively unchanged.
Two HTTP workers are therefore an explicit ranked-workload option, not a general
speedup. Four remain the CPU-derived serving default on this 30-CPU allocation.

Against fresh Luxir, the two-HTTP-worker configuration is **0.4% below** on
conjunction top-100 (42,720 versus 42,887 QPS), effectively tied at this precision.
It is **5.9% below** on top-10, versus **13.6% below** with four workers. For
conjunction counts, four workers are **13.6% above** Luxir, while two are **7.7%
below**. These differences describe this admitted workload and do not select a
universal worker policy.

The only admitted medium-frequency phrase is `"references reflist"`. Two HTTP
workers narrow its top-10 deficit to **5.3%**, from **10.7%** with four workers;
Summa remains substantially ahead on top-100. This should not be generalized to
all medium-frequency phrases: 45 of the 46 published queries in that class are
outside the agreement gate. Luxir's stripped executable still prevents equivalent
function-level attribution of its pruning/codec choices.

The alternating count experiment does **not reproduce the previous 4.9%
throughput regression at 32 clients**. Averaging the two session medians gives
76,639.99 QPS before and 76,638.93 with borrowed responses. The before runs bracket
the borrowed runs, and their ranges show comparable drift. This does not erase
the earlier measurement or establish behavior at one client. Borrowed responses
still consume about **1.5% more CPU per count** in this repeat (315.4 versus
310.7 µs); its cause remains unassigned.

For conjunction top-100, two HTTP workers use 598.9 CPU µs/request versus Luxir's
688.7, but keep only 25.59 CPU equivalents busy versus 29.53. The medium phrase's
top-10 likewise uses less CPU per request than Luxir while leaving more capacity
idle. The measurements continue to point toward dispatch/utilization work, rather
than proving that Summa' scorer performs more CPU work per query. CPU cost here
includes all server threads, not scoring alone.

A separate count-only lead appears in the screen: 15 workers/two HTTP workers
match current count throughput while reducing CPU cost from about 315 to 214
µs/request (32% lower). That setting loses ranked throughput and has only the
short-screen evidence; it deserves count-only confirmation before any policy
change. A single worker-width setting does not optimize both execution paths.

## Worker sweep

Median QPS from three four-second repetitions; screen results only. The current 30/4 setting is repeated at positions 00, 05 and 10. CPU cost is median server CPU microseconds per request. Full repetition ranges, CPU and memory values are retained in the [screen CSV](searchbench-2026-09-23-workers-screen.csv).

| Order / workers / HTTP | Top-10 QPS | Top-100 QPS | Count QPS | Count CPU µs/request |
| ---------------------- | ---------: | ----------: | --------: | -------------------: |
| 00-w30-h4              |   47,380.7 |    41,580.7 |  77,332.8 |                315.0 |
| 01-w15-h2              |   41,820.3 |    33,893.0 |  77,243.5 |                214.4 |
| 02-w15-h4              |   39,518.9 |    33,012.3 |  75,757.9 |                232.5 |
| 03-w30-h2              |   49,845.8 |    41,732.6 |  60,842.8 |                291.0 |
| 04-w30-h8              |   45,135.5 |    39,641.2 |  61,446.0 |                340.1 |
| 05-w30-h4              |   47,326.3 |    41,730.7 |  76,931.4 |                314.6 |
| 06-w8-h8               |   28,612.7 |    23,327.7 |  50,469.3 |                188.7 |
| 07-w15-h8              |   39,131.8 |    32,674.5 |  72,565.5 |                246.0 |
| 08-w8-h2               |   28,105.3 |    24,082.7 |  49,274.8 |                185.8 |
| 09-w8-h4               |   28,464.8 |    22,905.6 |  47,808.2 |                199.6 |
| 10-w30-h4              |   47,224.2 |    41,542.0 |  78,004.3 |                315.6 |

## Final CPU and memory

CPU equivalents are server CPU seconds divided by elapsed wall seconds, not physical core counts. Memory peaks cover all nine final search cells per variant. RSS includes faulted mapped pages; anonymous RSS is not an allocator-exact heap measurement.

| Family / operation     | Variant   | Busy CPU equivalents | CPU µs/request |
| ---------------------- | --------- | -------------------: | -------------: |
| and_high_low / TOP_10  | current   |                25.21 |          540.7 |
| and_high_low / TOP_10  | candidate |                26.48 |          521.4 |
| and_high_low / TOP_10  | luxir     |                29.42 |          545.1 |
| and_high_low / TOP_100 | current   |                25.16 |          617.3 |
| and_high_low / TOP_100 | candidate |                25.59 |          598.9 |
| and_high_low / TOP_100 | luxir     |                29.53 |          688.7 |
| and_high_low / COUNT   | current   |                24.10 |          316.5 |
| and_high_low / COUNT   | candidate |                17.84 |          288.5 |
| and_high_low / COUNT   | luxir     |                29.20 |          435.9 |
| med_phrase / TOP_10    | current   |                25.57 |         1622.4 |
| med_phrase / TOP_10    | candidate |                27.04 |         1617.7 |
| med_phrase / TOP_10    | luxir     |                29.87 |         1691.9 |
| med_phrase / TOP_100   | current   |                25.91 |         2465.6 |
| med_phrase / TOP_100   | candidate |                26.79 |         2463.7 |
| med_phrase / TOP_100   | luxir     |                29.93 |         5003.3 |

| Variant   | Peak RSS MiB | Peak anonymous RSS MiB |
| --------- | -----------: | ---------------------: |
| current   |       1151.4 |                  109.5 |
| candidate |       1149.7 |                  107.8 |
| luxir     |        158.9 |                   12.4 |

## Alternating count comparison

Four fresh processes in before/borrowed/borrowed/before order. Every case uses 30 search/blocking workers and four HTTP workers, ten seconds of warmup and three ten-second count repetitions at 32 clients. The [count CSV](searchbench-2026-09-23-workers-counts.csv) retains all metrics.

| Order / executable | Median QPS | Repetition range  | Busy CPU equivalents | CPU µs/request |
| ------------------ | ---------: | ----------------- | -------------------: | -------------: |
| 00-before          |   77,084.0 | 76,830.2–77,302.7 |                23.95 |          310.7 |
| 01-typed           |   76,779.6 | 76,474.4–76,807.1 |                24.20 |          316.1 |
| 02-typed           |   76,498.3 | 76,439.0–76,763.5 |                24.08 |          314.7 |
| 03-before          |   76,195.9 | 76,180.3–76,934.4 |                23.73 |          310.8 |

## What was varied

This is a configuration study of the frozen borrowed-ID executable from the
[response/handoff follow-up](searchbench-2026-09-23-handoffs.md). No Rust search,
parser, response, executor, admission, format or default changes were made.
The explicit `WORKERS` argument sets both the shared Rayon search pool and the
maximum Tokio blocking workers (`max(WORKERS, 4)`). `HTTP_WORKERS` sets the HTTP
runtime workers separately. Admission remains 64. The count path collects on the
blocking worker directly; ranked search also enters the owner-controlled shared
search pool. This distinction matters when interpreting the different responses
to the same settings. It is not an isolated comparison of Rayon pool widths.

The screen crosses 8/15/30 workers with 2/4/8 HTTP workers, at 32 clients over the
seven admitted conjunctions. The current 30/4 setting anchors the beginning,
middle and end; the other eight settings use shuffled order with seed 42. Each
fresh process gets ten seconds of mixed-operation session warmup, full untimed
validation, one second of connection warmup and three four-second repetitions
per cell. Short screen results select the follow-up, not a new default.

The independent final run tests the selected configuration first, then the
current configuration, then fresh Luxir on all 15 admitted queries at 32 clients.
Each gets thirty seconds of session warmup and three ten-second repetitions per
cell. The separate count-repeatability run uses four fresh processes in
before/borrowed/borrowed/before order, with the same 30/4 settings, ten seconds of
warmup and three ten-second repetitions for conjunction counts at 32 clients.
All throughput comparisons use uninstrumented builds. No compile, copy, index
build or competing benchmark overlaps a timed session on the benchmark machine.

The before and borrowed-response binaries are the previously paired release
builds from the same 8-vCPU build-host boot, pinned Rust 1.98.1 and
`-C target-cpu=native`. They are reused without rebuilding. All runs here share
one fresh boot of the 32-vCPU Cascade Lake benchmark machine, one immutable ordinary
10M-document index, and the same disjoint server/driver CPU allocation. Prior
session throughput is context only; it is not the control for this experiment.
The build machine remains stopped throughout.

## Correctness, scope and validation

Exhaustive top-100 document IDs, score bits and counts match the prior reference
for each tested worker width. Every Summa instance checks all 45 response bodies
against the previous byte-exact reference before timing. The standard campaign
also validates exact counts/ranked cardinality and unique IDs, plus before/after
topology. Admission and response semantics are unchanged by worker settings.

Coverage remains **15/826** count-compatible queries: seven conjunctions, seven
low-frequency phrases and one medium-frequency phrase. Equal counts do not
establish cross-engine ranking/analyzer equivalence. No full-workload, cold-cache,
concurrent ingestion/merge, ARM-throughput, production-gRPC or p99 claim is made.
More efficient scheduling on a single benchmark host would not by itself justify
changing architecture-independent defaults.

The unchanged Rust tree passes `python3 scripts/check_search.py check` under
`.context/search-harness/20260923T160104.343227Z-check`: all five stages pass,
including 2,029 native tests (25 normally ignored), formatting, strict Clippy,
native-without-sync and standalone broker compilation. The report helper passes
Ruff 0.16.0, Python compilation and CLI parsing. WASM and full production RPC tests
were not rerun in this configuration-only follow-up; the preceding WASM release
build and 39 JavaScript tests remain the latest validation of the unchanged code.

The machine start command lost its cloud polling connection and the first SSH attempt
arrived before port 22 was ready. Independent status and a successful retry
established the running machine before timing; no timed cell was affected. A scheduled
shutdown was installed before the campaign. Raw failures and recovery are retained.

Borrowed-response binary SHA-256:
`61f485a0ac1964389f4ec01b87b855fc8155911972f0c9f1b339d402b241c778`.
Before-response binary SHA-256:
`0938111c46ebf91d838c9857166a61a5aa646d4dcee1a02ed125ff993fb8f4be`.
The verified source/build archive from the preceding run is
`84ef63658c8af5f43adc21e1a25bb13afeaeb392350220ff9a51a462472ea51b`.
This run's scripts, raw results and local logs are retained under
`.context/yonik-benchmark/scheduling/`.

## Remaining dispatch question

The pinned Tokio 1.53.1 implementation was inspected locally, alongside Summa'
public async and synchronous search entry points. In
`runtime/scheduler/multi_thread/worker.rs`, `block_in_place` transfers the runtime
worker core through `runtime::spawn_blocking`, then attempts to take it back on
return. The runtime builder supplies that shared Tokio pool with a cap of
`max_blocking_threads + worker_threads`. Substituting `block_in_place` for the
benchmark's explicit `spawn_blocking` therefore changes which work crosses a
handoff; it is not proof that handoff costs disappear.

A future dispatch experiment must keep the existing Searcher as execution owner,
retain bounded admission and reader/permit ownership, preserve panic-to-response
behavior, and test forward progress and shutdown at full capacity. It must also
separate count's direct async collector from ranked search's shared-pool install.
This investigation introduces no public benchmark-only core API, second executor,
or unbounded CPU work on the HTTP reactor. The count-only efficiency tradeoffs
of smaller worker widths do not establish a suitable mixed-workload default.

All **64 cells / 192 repetitions** complete without request or memory-sampling
errors: 33 screen cells, 27 final comparison cells and four alternating count
cells. All 17 Summa instances preserve the same 45 response bodies; all tested
worker widths preserve the exhaustive audit. Luxir's same-process health control
also completes without errors at a median 399,693 requests/sec. That separate
`GET /health` c32/t4 control uses client CPUs 14–15,30–31, overlapping the server
on 14,30. Query replay uses the disjoint 15,31 pair. Health is transport context,
not a normalization factor or p99 claim.

Final evidence is downloaded, size-checked and SHA-256 verified as
`f8f8b4ffe0accb6a89547d16348de102cc1a688b4258ecf17b1caec000cffa2d`
(1,139,078 bytes). It includes raw replay/memory data, response bodies, audits,
settings, binary hashes, source scripts and run logs. No new `perf` profile was
taken; the prior handoff diagnostics and conjunction perf profiles remain separate evidence.

Host restrictions are restored and no temporary inter-machine transfer keys were
created. The benchmark stop command lost its polling connection; independent
cloud status confirms **both machines `TERMINATED`** after evidence verification.
No benchmark work remains running. Final documentation links, ownership
contracts, Python/Ruff and diff checks pass.
