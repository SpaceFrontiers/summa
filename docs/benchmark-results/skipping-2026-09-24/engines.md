# Shared-input engine comparison — September 24, 2026

Same 684 successful query texts from the original 826-query corpus; 10M documents, 32 clients, 30 server CPU threads and two driver threads. These are successful-query throughput results, not a ranking-equivalence test. Summa counts differ from the reference engines; [coverage](coverage.json), [all counts](counts.json), and [substantial count outliers](count-outliers.json) remain visible.

Median QPS of three 3-second repetitions after 60 seconds of session warmup and one second of connection warmup. Runs are serial on the same machine, with query/request caches disabled and one merged segment per index. These short windows are screening measurements; no p99 claim is made. [CPU, memory, repetition ranges and errors](engines.json) and [per-query mean observed latencies](query-latencies.json) accompany the tables. Query latencies include queueing under their full family workload and are not isolated single-query latencies.

Plain and RGB use separate preserved builds of the same corpus, originally indexed with six and thirty workers respectively. Their comparison is not an isolated reorder-only ablation. The skipping experiments compare each algorithm on exactly the same index.

Luxir single-term count cells saturate the two driver threads, and low-term
top-10 approaches that ceiling. Their QPS is not an unconstrained server-capacity
estimate. Process memory and driver/CPU details are summarized in the
[campaign report](README.md#method-memory-and-validation).

## TOP_10

| Family             | Timed / original | Counts within 5% |  Summa | Summa RGB | Elasticsearch | OpenSearch |   Luxir |
| ------------------ | ---------------: | ---------------: | -----: | --------: | ------------: | ---------: | ------: |
| and_high_high      |            47/47 |            44/47 |  1,148 |     1,223 |         1,954 |      1,934 |   2,295 |
| and_high_low       |            50/50 |            32/50 |  3,100 |     3,851 |        22,644 |     26,012 |  53,707 |
| and_high_med       |            50/50 |            40/50 |  2,905 |     3,628 |         6,002 |      6,508 |   9,002 |
| high_phrase        |            30/30 |            30/30 |  1,607 |     1,204 |           283 |        296 |   1,162 |
| high_sloppy_phrase |              7/7 |              7/7 |  6,169 |     7,488 |           139 |        138 |     376 |
| high_term          |            45/45 |            44/45 |    829 |     9,805 |        25,689 |     23,435 |  65,013 |
| low_phrase         |            50/50 |            48/50 |  2,645 |     2,724 |           279 |        280 |   1,426 |
| low_sloppy_phrase  |            37/37 |            36/37 |    967 |       940 |            46 |         45 |     370 |
| low_term           |            47/47 |            42/47 |  8,262 |    44,292 |        40,301 |     40,077 | 116,455 |
| med_phrase         |            46/46 |            44/46 |  1,965 |     1,871 |           256 |        270 |   1,003 |
| med_sloppy_phrase  |            27/27 |            26/27 |  1,247 |     1,318 |            57 |         58 |     354 |
| med_term           |            49/49 |            39/49 |  2,557 |    24,530 |        33,662 |     33,638 |  92,852 |
| or_high_high       |            42/42 |            42/42 |  1,716 |     1,335 |         1,835 |      1,807 |   2,369 |
| or_high_low        |            46/46 |            42/46 | 17,813 |     4,424 |        14,423 |     25,282 |  39,590 |
| or_high_med        |            45/45 |            43/45 |  5,722 |     4,927 |         5,844 |      6,565 |   7,922 |
| prefix3            |             4/50 |              3/4 |  1,515 |     1,189 |        20,845 |      9,999 |  87,485 |
| wildcard           |            15/49 |            14/15 |    683 |       578 |         6,801 |      4,878 |  13,359 |
| wildcard_scan      |            47/49 |            43/47 |    324 |       324 |         1,028 |      1,000 |   1,704 |
| regex              |             0/13 |                — |      — |         — |             — |          — |       — |
| wildcard_lead      |             0/47 |                — |      — |         — |             — |          — |       — |

## TOP_100

| Family             | Timed / original | Counts within 5% |  Summa | Summa RGB | Elasticsearch | OpenSearch |  Luxir |
| ------------------ | ---------------: | ---------------: | -----: | --------: | ------------: | ---------: | -----: |
| and_high_high      |            47/47 |            44/47 |  1,141 |     1,213 |         1,441 |      1,444 |  1,755 |
| and_high_low       |            50/50 |            32/50 |  3,077 |     3,813 |        17,193 |     18,486 | 35,062 |
| and_high_med       |            50/50 |            40/50 |  2,808 |     3,536 |         4,466 |      4,738 |  6,136 |
| high_phrase        |            30/30 |            30/30 |    562 |       397 |            60 |         58 |    327 |
| high_sloppy_phrase |              7/7 |              7/7 |  2,564 |     2,929 |           112 |        104 |    331 |
| high_term          |            45/45 |            44/45 |    828 |     6,043 |         9,163 |      8,494 | 15,987 |
| low_phrase         |            50/50 |            48/50 |  1,034 |     1,116 |           117 |        113 |    780 |
| low_sloppy_phrase  |            37/37 |            36/37 |    560 |       599 |            44 |         44 |    322 |
| low_term           |            47/47 |            42/47 |  8,101 |    27,061 |        19,418 |     20,849 | 35,964 |
| med_phrase         |            46/46 |            44/46 |    714 |       713 |           109 |        106 |    519 |
| med_sloppy_phrase  |            27/27 |            26/27 |    837 |       882 |            51 |         51 |    289 |
| med_term           |            49/49 |            39/49 |  2,530 |    14,174 |        13,506 |     13,184 | 22,268 |
| or_high_high       |            42/42 |            42/42 |  1,394 |     1,400 |         1,424 |      1,428 |  1,800 |
| or_high_low        |            46/46 |            42/46 | 10,028 |     3,801 |         9,267 |     12,844 | 16,260 |
| or_high_med        |            45/45 |            43/45 |  4,311 |     5,179 |         4,212 |      4,524 |  5,516 |
| prefix3            |             4/50 |              3/4 |  1,514 |     1,194 |        18,005 |      9,512 | 77,814 |
| wildcard           |            15/49 |            14/15 |    679 |       579 |         6,464 |      4,775 | 13,081 |
| wildcard_scan      |            47/49 |            43/47 |    321 |       323 |         1,012 |        994 |  1,696 |
| regex              |             0/13 |                — |      — |         — |             — |          — |      — |
| wildcard_lead      |             0/47 |                — |      — |         — |             — |          — |      — |

## COUNT

| Family             | Timed / original | Counts within 5% |  Summa | Summa RGB | Elasticsearch | OpenSearch |   Luxir |
| ------------------ | ---------------: | ---------------: | -----: | --------: | ------------: | ---------: | ------: |
| and_high_high      |            47/47 |            44/47 |  2,275 |     2,314 |         3,923 |        586 |   4,249 |
| and_high_low       |            50/50 |            32/50 | 13,492 |    17,611 |        26,609 |     21,025 |  54,667 |
| and_high_med       |            50/50 |            40/50 |  4,688 |     5,567 |         5,481 |      1,193 |   6,284 |
| high_phrase        |            30/30 |            30/30 |     61 |        67 |            18 |         15 |      62 |
| high_sloppy_phrase |              7/7 |              7/7 |    188 |       198 |            83 |         66 |     153 |
| high_term          |            45/45 |            44/45 | 82,006 |    85,408 |        82,652 |     74,924 | 125,181 |
| low_phrase         |            50/50 |            48/50 |    350 |       408 |            38 |         38 |     346 |
| low_sloppy_phrase  |            37/37 |            36/37 |    355 |       410 |            45 |         40 |     290 |
| low_term           |            47/47 |            42/47 | 84,156 |    85,262 |        78,834 |     74,488 | 126,832 |
| med_phrase         |            46/46 |            44/46 |    165 |       183 |            31 |         28 |     158 |
| med_sloppy_phrase  |            27/27 |            26/27 |    278 |       311 |            45 |         43 |     224 |
| med_term           |            49/49 |            39/49 | 81,715 |    85,158 |        79,187 |     75,432 | 127,149 |
| or_high_high       |            42/42 |            42/42 |  2,209 |     2,265 |         4,042 |        581 |   4,319 |
| or_high_low        |            46/46 |            42/46 | 17,340 |    19,775 |         2,664 |      1,234 |   9,336 |
| or_high_med        |            45/45 |            43/45 |  4,520 |     5,571 |         5,827 |        996 |   6,656 |
| prefix3            |             4/50 |              3/4 |  2,636 |     1,644 |         4,347 |      2,568 |   7,709 |
| wildcard           |            15/49 |            14/15 |  1,219 |       856 |         3,343 |      1,148 |   5,311 |
| wildcard_scan      |            47/49 |            43/47 |    321 |       322 |           974 |        950 |   1,637 |
| regex              |             0/13 |                — |      — |         — |             — |          — |       — |
| wildcard_lead      |             0/47 |                — |      — |         — |             — |          — |       — |

The dashes mean no common successful-query timing exists. All three reference engines execute all 826 count probes; Summa fails 142 explicit scan/expansion budgets. Missing successful-query timings must not be treated as zero cost or as a fast error response.
