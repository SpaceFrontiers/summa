# Same-host Searchbench throughput: non-RGB phrase optimization and Luxir

Count agreement: **15/826 queries**. This is a restricted workload, not the complete published benchmark.

10M Wikipedia chunks; one merged segment; six server hardware threads and two driver threads on separate physical cores. Query and request caches disabled. Thirty-second session warmup, full untimed validation, one-second connection warmup, three ten-second repetitions per cell.

All three variants are measured sequentially in this run. Both Summa binaries use the same rebuilt non-RGB index, without impact metadata. The comparison binary disables certified rare-term admission; all other code and writer output are identical. Luxir is rerun on the same host. Indexing and compilation finish before query timing.

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

| Family       | Operation | Clients | Distinct queries | Summa first-term bounds (QPS) | Summa certified rare-term bounds (QPS) | Luxir (QPS) |
| ------------ | --------- | ------: | ---------------: | ----------------------------: | -------------------------------------: | ----------: |
| and_high_low | COUNT     |       8 |                7 |                      16,368.5 |                               16,150.8 |    13,545.5 |
| and_high_low | TOP_10    |       8 |                7 |                       9,380.8 |                                9,255.1 |    10,686.8 |
| and_high_low | TOP_100   |       8 |                7 |                       7,244.8 |                                7,187.1 |     8,381.1 |
| low_phrase   | COUNT     |       8 |                7 |                         320.9 |                                  322.3 |       409.6 |
| low_phrase   | TOP_10    |       8 |                7 |                         677.5 |                                2,605.5 |     1,509.8 |
| low_phrase   | TOP_100   |       8 |                7 |                         449.2 |                                  968.5 |       765.6 |
| med_phrase   | COUNT     |       8 |                1 |                          98.8 |                                   97.3 |       121.2 |
| med_phrase   | TOP_10    |       8 |                1 |                       1,099.5 |                                2,128.6 |     3,523.6 |
| med_phrase   | TOP_100   |       8 |                1 |                         821.0 |                                1,343.4 |     1,195.8 |

## Interpretation and remaining work

Certified rare-term admission raises low-phrase top-10 throughput 3.85× and
medium-phrase top-10 throughput 1.94× against the paired first-term control.
Summa exceeds fresh Luxir throughput for low-phrase top-10/top-100 and the
single medium phrase's top-100. Medium-phrase top-10 remains 39.6% below Luxir;
phrase counting remains about 20–21% below it. This is not general parity.
The control already includes earlier traversal and position-read improvements;
these ratios isolate the additional certified rare-term admission.

A separate, untimed single-client profile of `"references reflist"` attributes
34.8% of top-10 samples to `find_in_block`, 14.9% to competitive advancement,
and 10.7% to document delta decoding. Counting instead spends 21.8% in position
range calculation, 19.5% in cached position reads, 14.7% in phrase matching,
and 18.1% in posting intersection functions. These are sampled self costs on
one query, not an attribution of Luxir's implementation or eight-client latency.
The next opportunities are reducing competitive scanning and per-candidate
position traversal. Lazy cutoff initialization and block-max-one TF scanning
were tested and rejected because they regressed the common phrase.

## Correctness, formats and resources

- Both Summa binaries returned identical counts, document IDs and score bits
  on all 45 HTTP responses. Exhaustive top-100 audits also matched for all 15
  queries. Across the original and rebuilt index, counts and ranked score-bit
  sequences matched, but physical document order changed and no complete ID
  sequence was identical. Cross-engine admission checks counts, not ranking.
- Position streams now use POS5/POS6 with a writer-certified uniqueness bit in
  the existing footer. Old position readers, migration branches and obsolete
  position-list codecs were removed. Existing position indexes require a
  rebuild. Impact and ratio metadata remain disabled by default.
- The current ordinary index occupies 14,525,096,783 bytes. Its build peaked at
  23,511,972 KiB RSS. Build time (12m48.71s) is not a controlled comparison:
  compilation overlapped ingestion. The uniqueness certificate adds no bytes.
- Peak query-process RSS was approximately 1,137 MiB for either Summa binary
  and 152 MiB for Luxir. These include each process's mapped resident pages;
  they are not total machine page-cache usage or heap-only measurements.
- The native harness passed 2,021 tests, strict Clippy and feature checks. The
  WASM build and 38 browser tests passed with regenerated native-writer fixtures.
  A fresh RGB smoke index passed exhaustive top-100 audits for these 15 queries.
  Full 10M RGB throughput was measured in the
  [earlier separate campaign](searchbench-2026-09-23.md), not rerun for POS5/POS6.
  Full production RPC tests were not run; no RPC lifecycle changed.

Raw replay JSON, memory samples, audits, profiles, compiler/host records and
source identity are retained in
`.context/yonik-benchmark/nonrgb-final-evidence.tar.gz`. The source manifest is
`.context/yonik-benchmark/nonrgb/certified-source-identity.json`; the native
check is `.context/search-harness/20260923T064244.392846Z-check`.
[CSV with repetition ranges and RSS](searchbench-2026-09-23-phrases.csv).

Evidence collection is complete; the benchmark machine is confirmed `TERMINATED`.
