# Expanded queries and skipping comparison — September 24, 2026

**Completed:** all 826 original queries were probed on five deployments; 684
successful shared inputs were timed across Summa, Summa RGB, Elasticsearch,
OpenSearch and Luxir. Six Summa configurations completed two skipping rounds.
All 702 timing cells finished without request errors. No production algorithm,
index format, schema or default changed.

## Coverage and comparison

| Deployment    | Successful count probes | Explicit failures |
| ------------- | ----------------------: | ----------------: |
| Summa         |                 684/826 |               142 |
| Summa RGB     |                 684/826 |               142 |
| Elasticsearch |                 826/826 |                 0 |
| OpenSearch    |                 826/826 |                 0 |
| Luxir         |                 826/826 |                 0 |

The three references agree on all 826 counts. The Summa layouts have identical
counts/errors. Only 15 queries match the reference counts exactly; **619 of the
684 successes are within 5%**, with a median relative difference of 0.34%.
The new `campaign.py gate --comparison shared-input` mode retains different
counts and validates each engine against its own observations. The default
`exact-count` mode remains available.

[Full top-10/top-100/count tables](engines.md), [per-cell CPU/memory and repetition
ranges](engines.json), [per-query observed latencies](query-latencies.json),
[all 826 counts/errors](counts.json), [coverage by family](coverage.json), and
[65 substantial count outliers](count-outliers.json) retain the complete picture.
These shared inputs do not establish equal logical work or ranking equivalence.

For example, `+in +user\:robert` matches 571,821 documents in Summa and 1,362 in
all references. Its recorded Summa plan requires `in` and an OR of `user` and
`robert`: punctuation is split during analysis, and the unqualified analyzed
term retains the parser's existing OR semantics. Conjunction timing gaps can
therefore include substantially different work. This experiment preserves those
semantics and exposes the discrepancy.

The 142 failures comprise 85 term-expansion limits and 57 dictionary-scan limits.
All 13 regex and 47 leading-wildcard inputs fail on this corpus. They remain in
the coverage report, with no successful-query QPS. Similar replacement text alone
cannot resolve a whole-field scan limit smaller than the field vocabulary.

## Selected top-10 results for a post

QPS, higher is better; 10M documents and 32 clients. These rows use the expanded
original-text workload, not the older 15-query exact-count subset. The full
report includes every family and operation.

| Family                  | Queries | Summa | Summa RGB | Elasticsearch | OpenSearch |  Luxir |
| ----------------------- | ------: | ----: | --------: | ------------: | ---------: | -----: |
| High/high conjunction   |      47 | 1,148 |     1,223 |         1,954 |      1,934 |  2,295 |
| High/low conjunction    |      50 | 3,100 |     3,851 |        22,644 |     26,012 | 53,707 |
| High-frequency phrase   |      30 | 1,607 |     1,204 |           283 |        296 |  1,162 |
| Medium-frequency phrase |      46 | 1,965 |     1,871 |           256 |        270 |  1,003 |
| Low-frequency phrase    |      50 | 2,645 |     2,724 |           279 |        280 |  1,426 |

Counts are within 5% for respectively 44/47, 32/50, 30/30, 44/46 and 48/50
queries in these rows. Phrase ranking is competitive on this broader workload;
single-term ranking, conjunctions and pattern execution remain important gaps.
RGB helps single-term top-10 substantially but regresses some OR/phrase families.
The preserved plain/RGB builds used different indexing worker counts, so this
is not an isolated reorder-only comparison.

## What the skipping experiments show

[All variant results and control drift](skipping.md) compare algorithms on the
same index. [Patches](patches/README.md) and the [design](../../score-guided-traversal.md#expanded-workload-experiment-september-24)
record the bounded work and exactness invariants.

- **Score-bound-prioritized pilot:** RGB high-frequency phrase top-10 improves
  from 1,196 to 1,380 QPS (**15.4%**), with **13.7% less CPU/request**. Both rounds
  improve (17.4% and 13.6%). The overall ranked phrase average is essentially
  unchanged: +0.3% plain and +0.1% RGB. This supports a targeted follow-up,
  not a universal replacement for the current pilot.
- **Longer skips matter:** disabling group jumps and losing-block coalescing
  reduces high-frequency sloppy-phrase top-10 QPS by **12.3% plain / 21.3% RGB**,
  with corresponding CPU increases. However, medium-frequency phrase top-100
  improves 6.6% / 5.0%. Measure skip lengths and metadata work before proposing
  a selective admission policy; do not disable longer skips globally.
- **No broad winner:** removing the pilot averages +2.0% plain / +0.9% RGB;
  eight short blocks average +1.0% / +0.3%; removing rare-term seeks averages
  +0.4% / +1.4%. Control cells drift by up to about 3.6%, so these small averages
  do not establish a default change. Rare-term seeks and pilots may still help
  individual queries hidden within a family.

The bound-priority experiment is a bounded pilot over existing phrase blocks.
It does not implement a general priority frontier over aligned document ranges
or a new persisted skip hierarchy. The larger remaining gaps warrant targeted
single-term/Boolean profiling and bounded pattern execution, alongside analysis
compatibility checks, before adding another generic skipping structure.

## Method, memory and validation

The [protocol](protocol.json) records compiler, binary/script hashes, CPU sets,
versions and timing settings. All runs use the preserved 10M single-segment
indexes, serial isolated-loopback sessions, 30 server CPU threads and two driver
threads. Query/request caches are disabled. Each cell has three 3-second
repetitions; skipping uses two reversed-order rounds, with 20-second session
warmup. Shared-input runs use 60-second session warmup. Connection warmup is one
second. These are screening measurements on one architecture, not p99 results.

Peak process residency across the shared-input cells:

| Deployment    | Total RSS (MiB) | Anonymous RSS (MiB) |
| ------------- | --------------: | ------------------: |
| Summa         |           3,740 |               1,196 |
| Summa RGB     |           3,937 |               1,203 |
| Elasticsearch |           9,813 |               9,138 |
| OpenSearch    |           9,978 |               9,305 |
| Luxir         |           1,357 |                 152 |

RSS includes resident mapped pages; anonymous RSS is not a precise heap-size
measurement. These are process peaks, not index sizes. Both JVM heaps are 8 GiB.
Luxir's single-term count cells use essentially both driver CPU threads; its
low-term top-10 cell approaches that ceiling too. Their throughput is not an
unconstrained server-capacity estimate. Per-cell CPU/request and driver usage
are retained in the JSON results.

All **2,558 exhaustive count/top-100 ID/score-bit audits** pass across selected
variants and layouts, together with **15,348 HTTP response checks** across both
rounds. The control covers all 344 conjunction/phrase queries on each layout.
The audits and 36,291,066 timed requests report no correctness or request
failures. The 142 rejected count probes remain explicit in coverage.

[Validation provenance](validation.json) records four harness regressions,
the full published-lock search check (**2,054 passed, 25 ignored**), strict
Clippy, native-without-sync and standalone broker checks. The five Linux
candidates pass 145 focused regression executions; all six configurations pass
WASM builds and **41 JavaScript tests each**. Supplemental earlier Apple ARM
checks used cached pre-release dependencies and are explicitly distinguished
from the published-lock checks. [Linux](linux-variants.log) and
[portable](portable-checks.log) logs are retained.

The complete raw archive was downloaded and SHA-256 verified. Both owned
machines were independently confirmed stopped after collection. Regenerate
compact result data from the extracted archive with:

```sh
python3 docs/benchmark-results/skipping-2026-09-24/summarize.py \
  --results "$RAW_RESULTS" --out "$SUMMARY_DIRECTORY"
```
