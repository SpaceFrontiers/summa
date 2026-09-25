# Closing the Searchbench gap: measured checkpoints

September 24–25, 2026. This is an ongoing investigation, not a claim of parity across all query families. Authentication was restored and the queued evidence has been collected. Follow-up execution and dictionary-format experiments are running; the gap is not yet closed.

## Workload and controls

The immutable corpus contains 10,000,000 documents (13,579,510,045 input bytes; SHA-256 `b15e60ad32a0e9f09f3be5335db86ccd885e1be1418086a906d68529e6ec1c9d`). Paired Linux runs use the same Intel Xeon 2.80 GHz machine with 32 logical CPUs / 16 physical cores, Rust 1.98.1, release optimization and `-C target-cpu=native`. Servers receive 30 logical CPUs; the driver receives the other two. There are 32 clients, four HTTP workers, 30 search workers, admission 64, and no result cache. Each cell has three three-second repetitions after a 40-second session warmup. Tables show median repetition QPS; JSON retains CPU/request, RSS, repetition ranges, hashes and query counts. Short runs do not establish tail-latency percentiles.

Candidate comparisons require exact within-index exhaustive counts and top-100 IDs/score bits before timing, followed by HTTP count and top-10/top-100 ID verification. Cross-engine tables use identical query text; hit-count differences remain visible and scores are not assumed equivalent. Broad queries that exceed resource limits are reported as failures, never as successful empty results.

## Coverage and analyzer configuration

The opt-in `unicode_word` tokenizer uses Unicode word boundaries and lowercase. Existing analyzer defaults are unchanged. On all 826 expressions, the latest probe succeeds on **677**, of which **640 have exactly the reference count** and all 677 are within 5%. The other **149 fail explicit scan/expansion budgets**. Before this analyzer change, 684 succeeded, only 15 matched exactly, and 619 were within 5%. Thus count agreement improves while successful coverage decreases by seven; this is not an execution-only speedup.

The new ordinary index is 18,693,711,229 bytes. Its build took about 17m40s with peak RSS 28,730,716 KiB; there is no controlled indexing-throughput comparison. Quantized norms are a separate index experiment because they can change ranking. Neither impacts, quantized norms, larger dictionary caches nor RGB are enabled by default by this work.

## Same-index execution changes

The `baseline-plain` and `patterns-plain` binaries use the same original plain index and 684 expressions. This checkpoint combines borrowed L1 metadata, existing single-term block traversal admission, lazy constant-score union, bounded regex prefix ranges and single-star matching. It predates the subsequent owner and inline optimizations.

| Family        | Operation | Queries | Before QPS | After QPS | Ratio | CPU µs/request before → after |
| ------------- | --------- | ------: | ---------: | --------: | ----: | ----------------------------: |
| and_high_high | COUNT     |      47 |      2,201 |     2,260 | 1.03× |           13,139.9 → 12,816.2 |
| and_high_high | TOP_10    |      47 |      1,261 |     1,286 | 1.02× |           23,157.6 → 22,615.9 |
| and_high_high | TOP_100   |      47 |      1,101 |     1,123 | 1.02× |           26,529.4 → 25,948.9 |
| and_high_low  | COUNT     |      50 |     12,694 |    12,872 | 1.01× |             2,143.1 → 2,094.6 |
| and_high_low  | TOP_10    |      50 |      2,987 |     2,952 | 0.99× |             9,577.4 → 9,689.7 |
| and_high_low  | TOP_100   |      50 |      2,949 |     2,927 | 0.99× |             9,735.6 → 9,747.9 |
| and_high_med  | COUNT     |      50 |      4,571 |     4,615 | 1.01× |             6,128.7 → 6,115.5 |
| and_high_med  | TOP_10    |      50 |      2,849 |     2,940 | 1.03× |             9,848.8 → 9,560.4 |
| and_high_med  | TOP_100   |      50 |      2,581 |     2,654 | 1.03× |           10,920.6 → 10,560.5 |
| high_term     | COUNT     |      45 |     80,970 |    82,284 | 1.02× |                   75.3 → 74.5 |
| high_term     | TOP_10    |      45 |      5,480 |     8,764 | 1.60× |             5,057.5 → 3,089.9 |
| high_term     | TOP_100   |      45 |      3,068 |     4,504 | 1.47× |             9,207.7 → 6,120.7 |
| low_term      | COUNT     |      47 |     80,197 |    81,215 | 1.01× |                   77.5 → 77.3 |
| low_term      | TOP_10    |      47 |     32,381 |    56,792 | 1.75× |                 766.0 → 450.6 |
| low_term      | TOP_100   |      47 |     20,199 |    29,923 | 1.48× |               1,230.5 → 829.9 |
| med_term      | COUNT     |      49 |     81,385 |    81,179 | 1.00× |                   75.3 → 77.1 |
| med_term      | TOP_10    |      49 |     18,867 |    30,561 | 1.62× |               1,345.4 → 831.2 |
| med_term      | TOP_100   |      49 |     10,088 |    13,919 | 1.38× |             2,590.0 → 1,859.2 |
| prefix3       | COUNT     |       4 |      2,546 |     2,415 | 0.95× |           11,353.0 → 12,034.5 |
| prefix3       | TOP_10    |       4 |      1,504 |     4,808 | 3.20× |            19,431.6 → 5,916.5 |
| prefix3       | TOP_100   |       4 |      1,492 |     4,941 | 3.31× |            19,579.6 → 5,814.5 |
| wildcard      | COUNT     |      15 |      1,206 |     1,218 | 1.01× |           24,325.8 → 24,189.5 |
| wildcard      | TOP_10    |      15 |        677 |     5,548 | 8.20× |            43,465.1 → 4,965.9 |
| wildcard      | TOP_100   |      15 |        677 |     5,470 | 8.08× |            43,468.9 → 5,011.6 |
| wildcard_scan | COUNT     |      47 |        321 |       449 | 1.40× |           89,936.7 → 64,395.1 |
| wildcard_scan | TOP_10    |      47 |        324 |       475 | 1.47× |           90,089.7 → 60,224.1 |
| wildcard_scan | TOP_100   |      47 |        323 |       475 | 1.47× |           90,066.5 → 59,924.6 |

## Unicode-word checkpoint versus Luxir

The following is the complete 673-query timing checkpoint before the four newly admitted regex expressions. Prefix/wildcard reference cells were rerun on exactly the same 55 selected expressions; the remaining cells retain the matching reference query sets. Exact count agreement does not establish equal candidate work or scoring semantics. Summa here uses the opt-in impacts index. Later owner/inline gains must not be multiplied into these measurements.

| Family             | Operation | Queries | Summa QPS | Luxir QPS | Summa / Luxir |
| ------------------ | --------- | ------: | --------: | --------: | ------------: |
| and_high_high      | COUNT     |      47 |     2,269 |     4,221 |         0.54× |
| and_high_high      | TOP_10    |      47 |     2,225 |     2,279 |         0.98× |
| and_high_high      | TOP_100   |      47 |     1,400 |     1,733 |         0.81× |
| and_high_low       | COUNT     |      50 |    44,627 |    47,970 |         0.93× |
| and_high_low       | TOP_10    |      50 |    26,090 |    43,628 |         0.60× |
| and_high_low       | TOP_100   |      50 |    24,451 |    30,337 |         0.81× |
| and_high_med       | COUNT     |      50 |     4,767 |     6,204 |         0.77× |
| and_high_med       | TOP_10    |      50 |     4,060 |     8,794 |         0.46× |
| and_high_med       | TOP_100   |      50 |     3,201 |     6,088 |         0.53× |
| high_phrase        | COUNT     |      30 |        60 |        63 |         0.96× |
| high_phrase        | TOP_10    |      30 |     2,036 |     1,144 |         1.78× |
| high_phrase        | TOP_100   |      30 |       506 |       319 |         1.59× |
| high_sloppy_phrase | COUNT     |       7 |       191 |       150 |         1.27× |
| high_sloppy_phrase | TOP_10    |       7 |     7,088 |       374 |        18.94× |
| high_sloppy_phrase | TOP_100   |       7 |     2,731 |       328 |         8.33× |
| high_term          | COUNT     |      45 |    83,284 |   122,838 |         0.68× |
| high_term          | TOP_10    |      45 |    60,500 |    58,989 |         1.03× |
| high_term          | TOP_100   |      45 |    25,484 |    15,833 |         1.61× |
| low_phrase         | COUNT     |      50 |       372 |       369 |         1.01× |
| low_phrase         | TOP_10    |      50 |     2,338 |     1,407 |         1.66× |
| low_phrase         | TOP_100   |      50 |       896 |       764 |         1.17× |
| low_sloppy_phrase  | COUNT     |      37 |       362 |       304 |         1.19× |
| low_sloppy_phrase  | TOP_10    |      37 |       930 |       360 |         2.58× |
| low_sloppy_phrase  | TOP_100   |      37 |       522 |       310 |         1.68× |
| low_term           | COUNT     |      47 |    82,830 |   123,633 |         0.67× |
| low_term           | TOP_10    |      47 |    67,399 |   110,381 |         0.61× |
| low_term           | TOP_100   |      47 |    39,112 |    33,502 |         1.17× |
| med_phrase         | COUNT     |      46 |       167 |       154 |         1.08× |
| med_phrase         | TOP_10    |      46 |     1,709 |     1,003 |         1.70× |
| med_phrase         | TOP_100   |      46 |       573 |       510 |         1.12× |
| med_sloppy_phrase  | COUNT     |      27 |       280 |       217 |         1.29× |
| med_sloppy_phrase  | TOP_10    |      27 |     1,265 |       358 |         3.53× |
| med_sloppy_phrase  | TOP_100   |      27 |       803 |       284 |         2.83× |
| med_term           | COUNT     |      49 |    82,747 |   125,919 |         0.66× |
| med_term           | TOP_10    |      49 |    66,582 |    88,022 |         0.76× |
| med_term           | TOP_100   |      49 |    30,836 |    21,776 |         1.42× |
| or_high_high       | COUNT     |      42 |     2,184 |     4,267 |         0.51× |
| or_high_high       | TOP_10    |      42 |     2,004 |     2,340 |         0.86× |
| or_high_high       | TOP_100   |      42 |     1,468 |     1,773 |         0.83× |
| or_high_low        | COUNT     |      46 |    37,650 |     9,229 |         4.08× |
| or_high_low        | TOP_10    |      46 |    17,878 |    33,938 |         0.53× |
| or_high_low        | TOP_100   |      46 |    11,955 |    15,274 |         0.78× |
| or_high_med        | COUNT     |      45 |     4,469 |     6,556 |         0.68× |
| or_high_med        | TOP_10    |      45 |     5,061 |     7,771 |         0.65× |
| or_high_med        | TOP_100   |      45 |     3,753 |     5,361 |         0.70× |
| prefix3            | COUNT     |       2 |     2,905 |    10,013 |         0.29× |
| prefix3            | TOP_10    |       2 |     5,804 |   117,904 |         0.05× |
| prefix3            | TOP_100   |       2 |     5,797 |   107,381 |         0.05× |
| wildcard           | COUNT     |      11 |     1,481 |     6,654 |         0.22× |
| wildcard           | TOP_10    |      11 |     5,659 |    29,427 |         0.19× |
| wildcard           | TOP_100   |      11 |     5,648 |    28,470 |         0.20× |
| wildcard_scan      | COUNT     |      42 |       292 |     1,740 |         0.17× |
| wildcard_scan      | TOP_10    |      42 |       303 |     1,815 |         0.17× |
| wildcard_scan      | TOP_100   |      42 |       302 |     1,808 |         0.17× |

Four additional regex expressions now complete with exact reference counts: `(www|http|https)`, the weekday alternation, `colou?r`, and `[jkqxz][a-z]*ess`. Their separately timed four-query cells are in [regex-matrix.json](regex-matrix.json). Bounded literal-prefix extraction retains all scan, match and posting limits.

## Concurrent expansion ownership

An ABBA comparison on the same 55-query index gives each expansion a local byte owner, preserving mmap lifetime and the original shared corruption observer. No payload is copied or pinned. The native handle remains 32 bytes; the portable handle grows from 24 to 32 bytes. The later query-local integrity wrapper shares the segment-wide write-once error state; its completed matrix is in [final-matrix.json](final-matrix.json).

| Family        | Operation | Before QPS, two rounds | After QPS, two rounds | CPU µs/request before → after | Anonymous RSS MiB before → after |
| ------------- | --------- | ---------------------: | --------------------: | ----------------------------: | -------------------------------: |
| prefix3       | COUNT     |           2,892, 2,879 |          3,370, 3,415 |                10,003 → 8,476 |                    296.9 → 283.0 |
| prefix3       | TOP_10    |           5,829, 5,856 |        16,270, 17,672 |                 4,846 → 1,584 |                    202.6 → 188.3 |
| prefix3       | TOP_100   |           5,801, 5,763 |        16,153, 17,570 |                 4,877 → 1,591 |                    202.9 → 188.9 |
| wildcard      | COUNT     |           1,487, 1,480 |          1,658, 1,647 |               19,818 → 17,704 |                    623.6 → 609.4 |
| wildcard      | TOP_10    |           5,670, 5,643 |          8,601, 8,784 |                 4,881 → 3,045 |                    350.7 → 337.4 |
| wildcard      | TOP_100   |           5,708, 5,587 |          8,508, 8,579 |                 4,861 → 3,039 |                    356.3 → 343.0 |
| wildcard_scan | COUNT     |               295, 294 |              295, 296 |               97,572 → 97,173 |                    628.3 → 616.0 |
| wildcard_scan | TOP_10    |               305, 305 |              305, 304 |               93,926 → 94,083 |                    626.5 → 613.4 |
| wildcard_scan | TOP_100   |               305, 305 |              305, 304 |               93,664 → 93,545 |                    628.1 → 615.7 |

## ARM and density controls

The Apple M4 checks use release Rust 1.98.1 with native CPU flags and ABBA order. Plain ranked-term fixtures improve 1.25–1.75×, constant-score filters 32–42×, and conjunction controls remain near parity. These are bounded synthetic fixtures, not evidence for a global default change. Single-star matching improves about 1.35×.

The retained list-density gate improves dense membership windows 2.12–2.40× on ARM; sparse controls stay within 1.1%. On the real corpus, `+of +s` COUNT improves about 1.47× in latency and CPU, with five sparse/common controls within about 3.5%. Unconditional grouping was rejected for 20–38% CPU regressions. [Count evidence](count-density.json) and [ARM evidence](arm-unions.json).

Avoiding re-encoding one-to-three-document dictionary postings improves the small prefix fixture 7.18–7.54× versus the immediately preceding union checkpoint. Regex and single-star controls improve 7–12%. Exact membership is checked outside the timed loop. [ARM inline evidence](arm-inline.json) includes binary hashes, confidence intervals and whole-process peak memory; its concurrent corpus matrix is being completed.

## Rejected experiments and remaining work

- Fewer search workers improve cheap term counts but materially hurt conjunction throughput; the 30-worker setting remains the reference. [Worker matrix](workers-matrix.json).
- Several alternative intersection kernels, wider sparse-seek admission and peeled variable-integer decoding failed matched corpus screens. They are absent from production code.
- Pre-scoring the rare conjunction side regressed representative top-100 probes by 16–29%; it is rejected. An incomplete ARM screen overlapped local compilation and is excluded from all performance claims.
- A 64 MiB dictionary-cache experiment roughly doubled two broad native wildcard probes, but increases memory and still needs its full concurrent matrix collected. No cache default changes.
- The matched 288-query quantized-norm experiment regresses high-term top-100 by about 13%, medium-term top-100 by about 13%, and high/low conjunction top-k by about 9%. It is rejected for the recommended benchmark configuration; defaults remain unchanged. [Matched norm matrix](norm-matrix-valid.json).
- A further blocked-intersection kernel has mixed ARM results and no consistent advantage on the important corpus controls; it remains outside production code.
- Broad dictionary scans and several conjunction/cheap-term cells remain below Luxir. Complete final-source retiming and independent machine shutdown verification remain outstanding.

## Validation and reproducibility

The final inline implementation passed the five-step search harness `20260924T223721.093990Z-check`: formatting, strict Clippy, native tests, native without synchronous execution, and standalone broker compilation. Two later decoder regression tests also pass in the nine-test inline selection. The WASM release build and all 41 JavaScript tests pass. No lifecycle or RPC implementation changed; the full lifecycle/RPC suite was not rerun.

New regressions cover borrowed encoded-group bytes and unaligned seeks; local-owner and mmap lifetimes; segment-wide corruption visibility after an expansion is dropped; complete mixed inline/external unions; malformed inline payloads in async/sync expansion; exact inline serialized bytes; and integers above `u32` rejected without truncation. Existing deletion, RGB, cross-segment, score-bit and batch-tail tests remain in the harness.

[matrix.json](matrix.json), [union-matrix.json](union-matrix.json), [coverage.json](coverage.json), and the other linked exports contain only completed retrieved evidence. `summarize.py` validates completion/error markers and HTTP/audit gates while exporting private artifacts. `traversal.patch` is the earlier measured checkpoint, not the final source; [current-source.patch](current-source.patch) records the reviewed working-tree changes at report generation.
