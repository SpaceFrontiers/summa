# Same-host Searchbench throughput: 32-vCPU host, 32 clients, Summa/RGB and references

Count agreement: **15/826 queries**. This is a restricted workload, not the complete published benchmark.

10M Wikipedia chunks; one merged segment; 32-vCPU host (16 physical cores with SMT), 30 server hardware threads and two driver threads on the reserved physical core. Reference JVM heaps: 8 GiB. Query and request caches disabled. Thirty-second session warmup, full untimed validation, one-second connection warmup, three ten-second repetitions per cell.

All variants run sequentially with isolated loopback networking. Prior/current Summa use the same immutable ordinary index; RGB is a separate current-format build from the same corpus. Impacts are disabled except in the explicitly labeled optional-impact follow-up, when present. No indexing, copying or compilation overlaps a timed run. Server CPUs: 0–14,16–30; driver CPUs: 15,31. This measures 32 concurrent clients on 30 server hardware threads, not 32 physical cores.

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

| Family       | Operation | Clients | Distinct queries | Summa before (QPS) | Summa optimized (QPS) | Summa optimized + RGB (QPS) | Summa optimized + impacts (QPS) | Elasticsearch (QPS) | OpenSearch (QPS) | Luxir (QPS) |
| ------------ | --------- | ------: | ---------------: | -----------------: | --------------------: | --------------------------: | ------------------------------: | ------------------: | ---------------: | ----------: |
| and_high_low | COUNT     |      32 |                7 |           48,278.0 |              49,210.3 |                    53,420.9 |                        51,801.4 |            29,209.5 |         23,131.4 |    65,301.7 |
| and_high_low | TOP_10    |      32 |                7 |           38,637.2 |              39,915.5 |                    45,426.1 |                        40,253.5 |            22,199.4 |         25,040.5 |    52,814.0 |
| and_high_low | TOP_100   |      32 |                7 |           20,345.1 |              20,595.9 |                    23,746.3 |                        20,592.2 |            18,309.4 |         19,499.3 |    42,103.5 |
| low_phrase   | COUNT     |      32 |                7 |            1,620.0 |               1,960.9 |                     2,876.8 |                         1,964.0 |               344.4 |            348.7 |     2,066.7 |
| low_phrase   | TOP_10    |      32 |                7 |           12,635.3 |              15,273.6 |                     7,927.4 |                        14,602.5 |             2,899.9 |          3,020.0 |     7,565.6 |
| low_phrase   | TOP_100   |      32 |                7 |            4,619.8 |               4,644.4 |                     5,123.5 |                         3,978.5 |             1,174.1 |          1,124.6 |     3,788.7 |
| med_phrase   | COUNT     |      32 |                1 |              496.7 |                 701.6 |                       714.4 |                           699.5 |               428.1 |            268.0 |       608.5 |
| med_phrase   | TOP_10    |      32 |                1 |           10,248.2 |              14,896.4 |                    15,879.8 |                        27,084.9 |             5,211.2 |          4,493.2 |    17,579.5 |
| med_phrase   | TOP_100   |      32 |                1 |            6,290.1 |               9,053.7 |                     7,139.5 |                        10,651.1 |             1,531.0 |          1,196.8 |     5,972.5 |

## Interpretation and limits

This is a new same-host comparison, not a rescaling of the 8-vCPU report. The
n2-highmem-32 machine exposes 16 physical Cascade Lake cores with SMT across two
sockets. Fifteen physical cores serve requests; one physical core runs replay.
The 32-client load therefore measures concurrency on 30 server hardware threads.
The prior and optimized Summa binaries read exactly the same immutable ordinary
index. The optimized binary is the same frozen executable used in the
[8-client follow-up](searchbench-2026-09-23-gap.md), built with Rust 1.98.1 and
`-C target-cpu=native`; no compilation overlaps these measurements.

Without RGB or impacts, medium-phrase top-10 improves **45.4%**, top-100 **43.9%**,
and count **41.2%** over the prior certified implementation. Low-phrase top-10
improves **20.9%** and count **21.0%**; top-100 changes only **0.5%**, which is too
small to treat as a robust gain. Conjunction cells change by 1.2–3.3%.

Ordinary optimized Summa exceeds Luxir on low-phrase top-10/top-100 and
medium-phrase top-100/count. Remaining gaps are medium top-10 (**15.3% lower**),
low count (**5.1% lower**) and all three conjunction cells (**24.4–51.1% lower**).
This result does not support an across-the-board engine win. There is just **one**
qualifying medium phrase and seven each in the other two families; three short
repetitions do not establish general workload or tail-latency behavior.

RGB is a separate current-format build from the identical corpus. It improves
low-phrase counts by 46.7% and conjunction cells by 8.6–15.3%, but regresses
low-phrase top-10 by 48.1% and medium-phrase top-100 by 21.1% relative to ordinary
optimized Summa. RGB construction used 30 indexing workers while the earlier
ordinary index used six, so this is not a pure reorder-only ablation. Independent
builds can differ in physical IDs and fragmentation. Keep RGB separate; these
measurements do not justify changing defaults.

The optional impact index raises medium-phrase top-10 to **27,085 QPS**, 81.8%
over ordinary Summa and 54.1% over Luxir. Medium top-100 reaches **10,651 QPS**,
17.6% over ordinary Summa. It regresses low-phrase top-10 by 4.4% and top-100 by
14.3%; phrase counts are essentially unchanged. Peak process RSS is 1,284 MiB.
Impacts remain **disabled by default**. This independently built index agrees
with exhaustive top-100 and preserves ordinary-index counts and ranked score bits.

Peak process RSS across search cells is 1,283 MiB before, 1,284 MiB optimized,
1,381 MiB RGB, 9,275 MiB Elasticsearch, 9,364 MiB OpenSearch and 224 MiB Luxir.
RSS includes resident mmap pages and is not heap-only memory or total OS page
cache. JVM heaps are fixed at 8 GiB. The raw samples and CSV preserve per-cell
values; no cold-cache, ingest/merge-concurrent, ARM-throughput or production-gRPC
performance claim is made.

## Correctness, recovery and provenance

Before/after Summa audits preserve exact counts, ranked IDs and score bits and
agree with exhaustive top-100 traversal for every admitted query. RGB agrees with
its own exhaustive traversal and preserves ordinary-index counts and ranked score
bits; tied external IDs need not be identical across independent builds. Every
variant also runs untimed exact-count validation before each timed cell. Only
15/826 queries qualify under the original cross-engine count gate. The later
wildcard feature is absent from the frozen timed binaries and does not expand
this gate.

The original campaign completed all three Summa variants, then Elasticsearch
failed before any reference timing because the isolated namespace could not
resolve the host name. A local `/etc/hosts` entry fixed lookup. Reference engines
were resumed in a fresh namespace with the same isolated-loopback network mode,
CPU split, heaps and disabled caches. Both the original failure and resumed logs
are retained. The optional impacts follow-up uses a third isolated namespace;
its index transfer finished before its audit and timing. The source disk needed
a read-only mount after reboot; the unsuccessful transfer attempt copied no files.
No reference measurement overlaps indexing, bulk transfer or profiling.

Luxir's same-live-server c32/t4 `GET /health` control has zero errors and median
**343,031 QPS** (range 340,826–344,571). Its stock four-thread driver placement is
CPUs 14–15,30–31, overlapping server CPUs 14,30; search timing uses the explicit
disjoint 30/2-thread split. Treat this separately configured control only as a
transport diagnostic. Search QPS is not corrected or normalized by it.

The optimized executable SHA-256 is
`5fcfdb94592eb594f8b73f7fc9467bbc46d92c531dd66ca041bee1917397527c`.
Source hashes and the frozen archive are retained under
`.context/yonik-benchmark/gap/`. The full archive includes cell/repetition JSON,
memory samples, topology, CPU/namespace context, correctness audits, build logs,
controls, and original/resumed/impacts campaign scripts and exit status.

All **63 timing cells** and their repetitions have zero request and memory-sampler
errors. Final resumed-reference and impact campaign exit codes are zero. The
original startup failure remains recorded as exit 1. The final evidence archive
was downloaded and SHA-256 verified:
`6b88e6e428b7b946f779c1eaa4bf2294a96028e24319f2b75b30d04ce139b094`
(728,707 bytes), under `.context/yonik-benchmark/gap/scaling-final/`.

The unchanged final Rust tree passed `python3 scripts/check_search.py check`:
2,029 tests pass, 25 normally ignored, formatting/strict Clippy/feature builds
pass. WASM release build and 39 browser tests pass; ordinary and RGB HTTP smokes
pass. The 826-expression small-fixture wildcard capability check is separate:
801 accepted, 25 explicit errors; no full-corpus wildcard performance is claimed.
Full production RPC/lifecycle tests were not rerun; no production wire protocol
or lifecycle mechanism changed.

Evidence collection is complete. Temporary transfer keys were removed and
AppArmor user-namespace restriction restored to its prior setting. Independent
cloud status confirms both benchmark machines `TERMINATED` after collection.
