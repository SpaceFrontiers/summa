# Summa 1.9.1: latest search measurements

Warm throughput in queries/second (higher is better), 10 million Wikipedia
chunks, 32 clients, 30 server hardware threads on Intel Cascade Lake. Query
caches are disabled. Summa uses the benchmark HTTP adapter over the core engine.

| Workload                          |  Summa | Summa RGB |  Luxir | Elasticsearch\* | OpenSearch\* |
| --------------------------------- | -----: | --------: | -----: | --------------: | -----------: |
| Conjunction / top 10              | 47,939 |    75,533 | 53,361 |          22,199 |       25,041 |
| Conjunction / top 100             | 42,995 |    62,932 | 42,653 |          18,309 |       19,499 |
| Conjunction / count               | 75,576 |    93,064 | 66,758 |          29,210 |       23,131 |
| Low-frequency phrase / top 10     | 16,982 |    22,205 |  7,615 |           2,900 |        3,020 |
| Low-frequency phrase / top 100    |  5,147 |     6,012 |  3,869 |           1,174 |        1,125 |
| Low-frequency phrase / count      |  1,985 |     2,898 |  2,072 |             344 |          349 |
| Medium-frequency phrase / top 10  | 16,687 |    18,880 | 17,641 |           5,211 |        4,493 |
| Medium-frequency phrase / top 100 | 10,988 |    11,685 |  5,977 |           1,531 |        1,197 |
| Medium-frequency phrase / count   |    702 |       715 |    608 |             428 |          268 |

Summa, RGB and Luxir are from [September 24](ranked-pruning-followup.md).
\*Elasticsearch 9.5.4 and OpenSearch 3.8.0 are the latest available
[September 23 measurements](benchmark-results/searchbench-2026-09-23-32cpu.md)
on the same hardware and workload configuration; they were not rerun with this
release. Luxir is version 0.1.0. These columns combine separate sessions.

Coverage is **15 of 826 queries** that passed the original cross-engine count
gate: seven conjunctions, seven low-frequency phrases and one medium-frequency
phrase. This is a targeted comparison, not an overall engine ranking. Matching
counts do not establish equivalent analyzers or relevance. Ordinary and RGB
indexes used different indexing-worker counts, so their comparison does not
isolate reordering alone.

Against its own September 24 baseline, RGB improves low-frequency phrase top-10
throughput by **155%** and medium-frequency phrase top-100 by **44%**. Counts
and some controls regress slightly; ordinary conjunction top-10 remains behind
Luxir. Existing indexes benefit without schema changes or rebuilding.

These timings precede the regex/escaped-literal additions. The new query support
passes an [826-expression capability probe](benchmark-results/query-support-2026-09-24/README.md),
but the newly supported families have no full-corpus throughput measurements yet.
