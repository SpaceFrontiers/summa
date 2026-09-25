# Same-host Searchbench throughput: four engines

Count agreement: **15/826 queries**. This is a restricted workload, not the complete published benchmark.

10M Wikipedia chunks; one merged segment; six server hardware threads and two driver threads on separate physical cores. Reference JVM heaps: 8 GiB. Query and request caches disabled. Thirty-second session warmup, full untimed validation, one-second connection warmup, three ten-second repetitions per cell.

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

| Family       | Operation | Clients | Distinct queries | Summa (QPS) | Elasticsearch (QPS) | OpenSearch (QPS) | Luxir (QPS) |
| ------------ | --------- | ------: | ---------------: | ----------: | ------------------: | ---------------: | ----------: |
| and_high_low | COUNT     |       1 |                7 |     3,582.0 |             1,485.5 |          1,247.5 |     3,200.5 |
| and_high_low | COUNT     |       8 |                7 |    15,632.7 |             6,653.1 |          5,165.0 |    13,543.9 |
| and_high_low | COUNT     |      32 |                7 |    19,025.6 |             8,223.9 |          5,902.9 |    14,049.3 |
| and_high_low | TOP_10    |       1 |                7 |     2,245.8 |             1,121.2 |          1,094.4 |     2,642.1 |
| and_high_low | TOP_10    |       8 |                7 |     9,253.1 |             5,048.5 |          5,741.8 |    10,848.1 |
| and_high_low | TOP_10    |      32 |                7 |    10,856.8 |             5,940.2 |          6,770.2 |    11,204.5 |
| and_high_low | TOP_100   |       1 |                7 |     1,781.1 |               924.9 |            913.9 |     1,969.0 |
| and_high_low | TOP_100   |       8 |                7 |     7,159.8 |             4,015.3 |          4,305.7 |     8,531.6 |
| and_high_low | TOP_100   |      32 |                7 |     7,968.5 |             4,645.2 |          4,929.0 |     8,808.5 |
| low_phrase   | COUNT     |       1 |                7 |        88.0 |                88.8 |             59.7 |       117.3 |
| low_phrase   | COUNT     |       8 |                7 |       266.9 |               325.2 |            204.5 |       415.4 |
| low_phrase   | COUNT     |      32 |                7 |       266.5 |               329.5 |            206.2 |       415.8 |
| low_phrase   | TOP_10    |       1 |                7 |       125.6 |               281.7 |            250.8 |       438.8 |
| low_phrase   | TOP_10    |       8 |                7 |       470.4 |             1,058.8 |            984.2 |     1,502.8 |
| low_phrase   | TOP_10    |      32 |                7 |       471.4 |             1,097.2 |          1,029.3 |     1,535.6 |
| low_phrase   | TOP_100   |       1 |                7 |       102.1 |               160.5 |            141.1 |       224.4 |
| low_phrase   | TOP_100   |       8 |                7 |       370.7 |               600.9 |            546.6 |       764.9 |
| low_phrase   | TOP_100   |      32 |                7 |       369.4 |               614.1 |            553.1 |       773.2 |
| med_phrase   | COUNT     |       1 |                1 |        26.6 |                33.0 |             17.2 |        36.8 |
| med_phrase   | COUNT     |       8 |                1 |        69.8 |               119.5 |             54.6 |       121.3 |
| med_phrase   | COUNT     |      32 |                1 |        69.9 |               120.0 |             54.8 |       121.5 |
| med_phrase   | TOP_10    |       1 |                1 |        62.6 |               319.6 |            278.5 |     1,152.5 |
| med_phrase   | TOP_10    |       8 |                1 |       188.9 |             1,195.5 |          1,016.9 |     3,517.7 |
| med_phrase   | TOP_10    |      32 |                1 |       188.4 |             1,237.1 |          1,044.6 |     3,598.6 |
| med_phrase   | TOP_100   |       1 |                1 |        60.1 |               141.5 |            109.6 |       395.7 |
| med_phrase   | TOP_100   |       8 |                1 |       185.5 |               533.3 |            404.1 |     1,193.3 |
| med_phrase   | TOP_100   |      32 |                1 |       184.9 |               542.0 |            412.3 |     1,210.7 |
