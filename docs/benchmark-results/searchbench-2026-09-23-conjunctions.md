# Same-host Searchbench throughput: conjunction response encoding and CPU-based HTTP workers

Count agreement: **15/826 queries**. This is a restricted workload, not the complete published benchmark.

10M Wikipedia chunks; one merged segment; 32-vCPU host (16 physical cores with SMT), 30 server hardware threads and two driver threads on the reserved physical core. Query and request caches disabled. Thirty-second session warmup, full untimed validation, one-second connection warmup, three ten-second repetitions per cell.

Only the seven agreeing `and_high_low` queries are timed here, at one and 32 clients. All four variants run sequentially in isolated loopback networking; all Summa variants use the same ordinary index, without RGB or impact metadata. Server CPUs: 0–14,16–30; driver CPUs: 15,31. No copying, compilation or indexing overlaps timed runs. Core search algorithms are unchanged by this follow-up.

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

| Family       | Operation | Clients | Distinct queries | Summa before (HTTP 2) (QPS) | Summa encoded (HTTP 2) (QPS) | Summa encoded (HTTP auto: 4) (QPS) | Luxir (QPS) |
| ------------ | --------- | ------: | ---------------: | --------------------------: | ---------------------------: | ---------------------------------: | ----------: |
| and_high_low | COUNT     |       1 |                7 |                     2,816.8 |                      2,599.6 |                            2,269.4 |     3,226.3 |
| and_high_low | COUNT     |      32 |                7 |                    51,285.9 |                     50,187.6 |                           68,523.8 |    64,756.2 |
| and_high_low | TOP_10    |       1 |                7 |                     2,143.1 |                      2,050.2 |                            2,093.8 |     2,717.1 |
| and_high_low | TOP_10    |      32 |                7 |                    41,427.8 |                     44,770.6 |                           41,721.5 |    53,073.0 |
| and_high_low | TOP_100   |       1 |                7 |                     1,507.2 |                      1,317.7 |                            1,312.1 |     1,589.2 |
| and_high_low | TOP_100   |      32 |                7 |                    21,036.2 |                     33,246.0 |                           32,719.8 |    41,003.0 |

## Measured result and tradeoffs

At 32 clients, the final automatic configuration improves top-100 **55.5%**
and count **33.6%** over baseline. Top-10 changes only **0.7%**, too little to
claim a robust improvement. Summa exceeds fresh Luxir count throughput by
**5.8%**, but remains **21.4% below** on top-10 and **20.2% below** on top-100.
The baseline top-100 gap was 48.7%: the frontend explains a substantial part of
it, while a material ranked-query gap remains.

Encoding with explicit HTTP 2 gives slightly higher ranked throughput than
HTTP auto/4 (top-10 44,771 versus 41,722; top-100 33,246 versus 32,720), but much
lower count throughput (50,188 versus 68,524). More HTTP threads are not uniformly
better. The override is useful when selecting a configuration for a known load.

Single-client medians regress against baseline: **2.3%** top-10, **12.9%** top-100,
and **19.4%** count with automatic sizing. Some repetition ranges overlap;
Luxir's single-client top-100 varies especially widely (1,343–2,064 QPS).
The CSV retains all ranges. These results do not support a universal latency
improvement or a precise single-client engine ranking from one short run.

## CPU and memory

The table gives top-100 at 32 clients. CPU values are medians of three repetitions;
RSS and anonymous RSS are peaks across all six search cells of each variant.
CPU equivalents are server CPU-seconds divided by wall seconds, not physical
core counts. Anonymous RSS is not an allocator-exact heap measurement.

| Variant              | Busy CPU equivalents | CPU µs/request | Peak RSS MiB | Peak anonymous RSS MiB |
| -------------------- | -------------------: | -------------: | -----------: | ---------------------: |
| Before, HTTP 2       |                17.39 |          826.8 |       1149.7 |                  108.0 |
| Encoded, HTTP 2      |                24.88 |          748.5 |       1147.2 |                  105.5 |
| Encoded, HTTP auto/4 |                24.88 |          760.2 |       1148.5 |                  106.6 |
| Luxir                |                29.00 |          707.3 |        113.7 |                   11.8 |

The remaining top-100 gap combines lower CPU utilization (24.9 versus 29.0 CPU
equivalents) with higher CPU cost per request (760 versus 707 µs). Measure time
waiting between HTTP,
blocking-worker and shared Rayon execution before redesigning pool ownership.
The original profile's work-stealing/epoch samples make scheduling a concrete
follow-up alongside posting and ID-column costs. Count CPU cost at 32 clients
is lower for automatic Summa (341 µs/request) than Luxir (442 µs/request).

These conjunction-only memory figures are not directly comparable with the
prior all-family report: a different set of queries faults different mmap pages.

## What changed and what this measures

Response JSON encoding and destruction of the temporary JSON tree now happen
inside the existing bounded blocking worker, before its admission permit drops.
The HTTP runtime receives the finished bytes with the same content type. The
search pool, scorer, ID projection, response representation and persisted index
are unchanged. The frontend still bounds admission at 64 and request bodies at
16 KiB; these runs use 30 search workers and at most 30 blocking workers.

The benchmark frontend's default HTTP worker count now
uses available logical CPUs: one on a single CPU, otherwise
`clamp(ceil(available_cpus / 8), 2, 8)`. The Linux server affinity exposes 30 CPUs,
so automatic selects four. Explicit `HTTP_WORKERS` values from 1 to 64 override
this choice. Startup logs show detection and selection; detection failure emits
a warning and uses one CPU. Other example commands retain two runtime workers.
Production gRPC and core search defaults are unchanged. This bounded heuristic
has been measured on this host; it is not claimed optimal across workloads or
architectures.

All final Summa variants were compiled on the same 8-vCPU Cascade Lake build machine
using Rust 1.98.1, release mode and `RUSTFLAGS='-C target-cpu=native'`, then run on
the same 32-vCPU Cascade Lake benchmark machine. Builds and transfers finished before
timing. All use the same immutable ordinary 10M-document index; RGB and impacts
remain separate and impacts remain disabled. The new wildcard implementation is
present in these rebuilt binaries but is not exercised by the seven conjunctions.
This does not expand the original 15/826 count-agreement gate.

## Diagnostics and attribution

The initial software user-time profile (`task-clock:u`, 99 Hz, DWARF) used the
frozen binary from the prior phrase campaign. During a 12-second top-100 replay,
Summa consumed about 215 server CPU-seconds (17.9 logical-CPU equivalents),
versus Luxir's 341 (28.4). Summa' two HTTP workers each accumulated about 11.7
CPU-seconds during the roughly 15 active seconds including warmup. These are
profiled diagnostics, not the uninstrumented throughput table above.

Summa' largest self samples include AVX2 posting-gap unpacking (9.19%), posting
intersection (8.92%), block seeking (7.56%), and the conjunction executor (4.85%).
ID-column blockwise reads account for 4.78%; malloc accounts for 4.34%. The
samples identify remaining Summa work, but do not prove Luxir's internal
algorithm is better. Luxir's official release is stripped: 97.4% of user samples
are attributed to its executable without sufficient function symbols for a
comparable kernel diagnosis. Kernel time is not included in these user profiles.

A separate three-by-five-second screen isolates the two frontend changes:

| Configuration                | Top-10 QPS | Top-100 QPS | Count QPS |
| ---------------------------- | ---------: | ----------: | --------: |
| Before, HTTP 2               |     38,469 |      20,531 |    49,305 |
| Only HTTP 4                  |     39,436 |      26,874 |    72,558 |
| Only worker encoding, HTTP 2 |     44,571 |      33,295 |    49,461 |

These preliminary measurements are separate from the final longer campaign.
They support two distinct effects: expensive response work congests the two
HTTP workers, and transport scheduling limits count throughput even though count
responses are tiny. Moving encoding alone is therefore not the whole fix.

Stage diagnostics call the existing parser, shared Searcher and response
projector. Timings use an uninstrumented binary, 20 warmups and 100 samples for
each query/limit. Aggregate values are means of seven per-query medians. Work
counts use a separate feature-enabled binary and are never used for throughput.
Stage timings exclude HTTP, concurrent scheduling and some request/collector
cleanup, so they are not a complete end-to-end latency decomposition.

| Limit | Parse µs | Core search µs | ID/JSON projection µs | Serialize µs | Response-tree drop µs |
| ----- | -------: | -------------: | --------------------: | -----------: | --------------------: |
| 10    |     9.38 |         212.61 |                  7.17 |         0.56 |                  0.67 |
| 100   |     9.46 |         228.85 |                 57.39 |         4.82 |                  6.54 |

Core search grows only about 7.6% between these limits, while projection grows
about eightfold. Decoded IDs, aligned candidates and score units are identical
at both limits; successful heap updates grow as expected. This rules out a larger
posting scan as the cause of the top-100 versus top-10 throughput cliff on these
queries. It does not eliminate heap/ordering costs or explain every scheduling
and allocator effect under concurrency.

| Query                   | Decoded IDs (both limits) | Candidates (both) | Score units (both) | Heap updates k=10 | Heap updates k=100 |
| ----------------------- | ------------------------: | ----------------: | -----------------: | ----------------: | -----------------: |
| `+2008 +extracting`     |                   191,100 |               123 |                246 |                43 |                120 |
| `+two +overthrowing`    |                    98,837 |                97 |                194 |                32 |                 97 |
| `+accessdate +wickets2` |                   193,060 |                84 |                168 |                29 |                 84 |
| `+is +86th`             |                   131,398 |               310 |                620 |                41 |                215 |
| `+also +mutated`        |                   204,902 |               500 |               1000 |                50 |                275 |
| `+of +ffebcd`           |                   119,301 |               166 |                332 |                37 |                149 |
| `+also +staring`        |                   171,882 |               320 |                640 |                41 |                212 |

## Correctness and remaining work

All three Summa variants preserve exact counts, ranked external IDs, score bits
and exhaustive top-100 results for all 15 admitted queries. Their 45 HTTP response
bodies (count, top-10 and top-100 for each query) are byte-identical. Every timing
cell also runs untimed count or ranked-cardinality/unique-ID validation; topology
is checked before and after each server session. Count agreement with Luxir is
not a cross-engine ranking oracle.

Next core candidates remain posting seek/intersection and ID-column random reads.
The latter currently walk preceding sub-block headers; any improvement needs a
bounded reader-owned design and unchanged missing/value/ordinal semantics. The
measurements do not justify changing BM25 scoring, enabling impacts or adopting
RGB by default. Phrase gaps from the prior report are not retimed here. Complete
826-query comparison still requires analyzer-compatible indexes, sloppy-phrase
semantics, regex/escaped-literal support and bounded broad term expansion.

No cold-cache, concurrent ingestion/merge, ARM-throughput, production-gRPC or
p99 claim is made. Three short repetitions on seven queries cannot establish a
general engine ranking. RSS includes mmap residency and is not heap-only memory
or total operating-system page cache.

## Validation and provenance

The combined Rust tree passes `python3 scripts/check_search.py check` under
`.context/search-harness/20260923T110257.277217Z-check`: 2,029 tests pass, 25 are
normally ignored, and formatting, strict Clippy, native-without-sync and standalone
broker checks pass. The HTTP default test and opt-in heap-counter regression pass;
feature-enabled strict Clippy passes. WASM release build and all 39 browser tests
pass. Ordinary automatic-worker and RGB explicit-four-worker HTTP smoke tests
cover query behavior, request failures, concurrent requests and shutdown. CLI
checks reject out-of-range worker counts and overrides on non-serving commands
before index I/O. Documentation links, Python compilation and diff whitespace
checks pass. Full production RPC/lifecycle tests were not rerun; no production
wire protocol or lifecycle protocol changed.

The first profile attempt omitted a required replay shuffle seed and failed
before timing. The corrected profile is retained separately as
`conjunction-profile-2`; the failed attempt is preserved. A direct system-Python
documentation check lacked `markdown_it`; the documented `uv run` check passes.
Both machine stop commands lost their cloud polling connections; independent status
checks are used to confirm actual shutdown. These operational failures are not timed samples.

Final uninstrumented executable SHA-256:
`552f0f6379a516e02da9d4052e9a7f2e84b9c29e264ebd0c0c4929cacf0d4aba`.
Before executable:
`17ff76f49b55306cd338cea2f625fd0b9ab5d912567ca566e668a4a7d7d0df0f`.
Diagnostic counter executable:
`0966779e9c4f13ff7e060d8c3421a119746d3380e09215cc5c136762abc722a3`.
The downloaded build/source archive is SHA-256 verified as
`e7413e9401b6e9b0ebfe121b08e9cf73cd849cc0fb4bdc6f0b9af155c4765490`.
Raw evidence and reproduction scripts are retained under
`.context/yonik-benchmark/conjunction/`.

Luxir's same-live-server `GET /health` control (c32/t4, three ten-second runs)
has zero errors and median 345,111 QPS (344,747–345,251). Its stock driver uses
CPUs 14–15,30–31 and overlaps server CPUs 14,30. The search comparison uses the
disjoint 30/2 split. This control is only a transport diagnostic; search QPS is
not normalized or corrected by it.

All **24 timing cells** and repetitions have zero request/memory-sampler errors.
The final campaign exits zero. The downloaded evidence archive includes raw perf,
repetition JSON, memory traces, topology, correctness audits, controls, scripts,
binary hashes and restored host settings. Remote/local SHA-256 matches:
`4b535604f4f5c8f9875d12e750bbb1e3d01da8de0b1d2358033979f6aab83188`
(65,645,432 bytes). Temporary transfer keys are removed, AppArmor user-namespace
restriction restored to 1 and perf paranoia restored to 4.

Independent cloud status confirms both `benchmark-host` and
`benchmark-host` are `TERMINATED` after evidence collection.
