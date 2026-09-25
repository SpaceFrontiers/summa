# Same-host Searchbench throughput: Summa optimization and RGB

Count agreement: **15/826 queries**. This is a restricted workload, not the complete published benchmark.

10M Wikipedia chunks; one merged segment; six server hardware threads and two driver threads on separate physical cores. Query and request caches disabled. Thirty-second session warmup, full untimed validation, one-second connection warmup, three ten-second repetitions per cell.

The three Summa instances run sequentially: the unchanged binary and the optimized binary use the same original index; RGB uses a separately built index from the same corpus with the body field reordered. Reference-engine timings remain in the original report; no mixed-run speedup is claimed.

Summa uses a benchmark HTTP frontend over core, not its production gRPC service. Text-analysis differences exclude many queries; equal corpus counts do not prove general analyzer or relevance equivalence. No p99 claim is made.

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

| Family       | Operation | Clients | Distinct queries | Summa before (QPS) | Summa optimized (QPS) | Summa optimized + RGB (QPS) |
| ------------ | --------- | ------: | ---------------: | -----------------: | --------------------: | --------------------------: |
| and_high_low | COUNT     |       8 |                7 |           15,487.4 |              16,273.0 |                    26,648.2 |
| and_high_low | TOP_10    |       8 |                7 |            9,173.5 |               9,356.0 |                    15,596.1 |
| and_high_low | TOP_100   |       8 |                7 |            6,995.4 |               7,144.5 |                     9,586.8 |
| low_phrase   | COUNT     |       8 |                7 |              265.8 |                 258.2 |                       310.8 |
| low_phrase   | TOP_10    |       8 |                7 |              452.5 |                 478.4 |                       995.6 |
| low_phrase   | TOP_100   |       8 |                7 |              359.7 |                 364.2 |                       554.6 |
| med_phrase   | COUNT     |       8 |                1 |               69.5 |                  68.8 |                        65.8 |
| med_phrase   | TOP_10    |       8 |                1 |              186.2 |                 188.6 |                       194.6 |
| med_phrase   | TOP_100   |       8 |                1 |              181.9 |                 182.7 |                       186.4 |
