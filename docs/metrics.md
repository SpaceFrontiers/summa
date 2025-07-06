# Prometheus Metrics

Status: design (2026-07-09), implemented.

## Problem

Query-path performance work (BMP pruning quality, rerank IO, memory-bound
degradation) has relied on debug log lines (`slow BMP: ...ms, sbs=, blocks=`)
— unusable for dashboards, alerting, or regression tracking in production.

## Design

- **hermes-core** emits through the [`metrics`](https://docs.rs/metrics)
  facade behind a new **non-default `metrics` feature** (native-only; wasm
  never enables it). Without an installed recorder the macros are ~1ns no-ops,
  and every emission happens at _aggregation points_ (query end, phase end,
  IO call) — never per block/superblock inside scoring loops, so the hot-path
  allocation/atomics budget is untouched.
- **hermes-server** enables the feature, installs
  `metrics-exporter-prometheus`, and serves `GET /metrics` on
  `--metrics-addr` (default `0.0.0.0:9184`).
- Emission sites live behind tiny `observe::*` helper fns so call sites carry
  no `#[cfg]` noise; the helpers compile to nothing when the feature is off.

## Metric set

Histograms are seconds unless noted. `field` labels are **field names** and
every schema-scoped metric also carries an **`index` label** (the registry
index name, embedded in the schema at creation; "unknown" for pre-existing
indexes until recreated). Only `hermes_directory_read_*` and
`hermes_cold_write_bytes_total` lack the index label (no schema in scope).

| Metric                                                    | Type                | Labels                                                              | Meaning                                                                                                                                                                                                                                          |
| --------------------------------------------------------- | ------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `hermes_bmp_query_duration_seconds`                       | histogram           | `field`                                                             | full BMP executor wall time                                                                                                                                                                                                                      |
| `hermes_bmp_superblocks_visited_total` / `_skipped_total` | counter             | `field`                                                             | superblock iteration vs pruned — skip ratio via PromQL                                                                                                                                                                                           |
| `hermes_bmp_blocks_scored_total` / `_skipped_total`       | counter             | `field`                                                             | block-level skip ratio                                                                                                                                                                                                                           |
| `hermes_bmp_blocks_scored_per_query`                      | histogram           | `field`                                                             | per-query distribution (tail queries)                                                                                                                                                                                                            |
| `hermes_bmp_docmap_lookups_total` / `_per_query`          | counter / histogram | `field`                                                             | doc-map indirection: BMP reorder permutes only BMP-internal record order, so every scored candidate resolves through the doc-id map — this counts those scattered lookups (their latency lands in the query histogram; pin doc maps to bound it) |
| `hermes_sparse_maxscore_query_duration_seconds`           | histogram           | `field`                                                             | DAAT MaxScore executor wall time                                                                                                                                                                                                                 |
| `hermes_dense_l1_duration_seconds`                        | histogram           | `field`, `kind` (`rabitq`/`ivf_rabitq`/`scann`/`binary_ivf`/`flat`) | ANN / brute-force candidate generation                                                                                                                                                                                                           |
| `hermes_dense_rerank_duration_seconds`                    | histogram           | `field`                                                             | rerank total (resolve+read+score)                                                                                                                                                                                                                |
| `hermes_dense_rerank_resolve_duration_seconds`            | histogram           | `field`                                                             | doc→flat-slot indirection (flat store is not reordered)                                                                                                                                                                                          |
| `hermes_dense_rerank_read_duration_seconds`               | histogram           | `field`                                                             | raw-vector reads within rerank (page-fault sensitive)                                                                                                                                                                                            |
| `hermes_dense_rerank_vectors`                             | histogram           | `field`                                                             | candidates reranked per query                                                                                                                                                                                                                    |
| `hermes_directory_read_duration_seconds`                  | histogram           | `op` (`lazy_range`)                                                 | Directory-layer read latency — only Lazy handles (HTTP/custom `read_fn`) do real IO here; mmap slices are zero-copy and their fault latency lands inside the phase histograms                                                                    |
| `hermes_directory_read_bytes`                             | histogram           | `op`                                                                | read sizes                                                                                                                                                                                                                                       |
| `hermes_store_get_duration_seconds`                       | histogram           | —                                                                   | document store fetch (decompression + faults)                                                                                                                                                                                                    |
| `hermes_cold_write_bytes_total`                           | counter             | —                                                                   | merge/reorder bytes written via the page-cache-dropping cold path (`docs/cold-io.md`)                                                                                                                                                            |
| `hermes_search_duration_seconds`                          | histogram           | `index`, `status`                                                   | server: full search RPC                                                                                                                                                                                                                          |
| `hermes_search_requests_total`                            | counter             | `index`, `status`                                                   | server: RPC outcomes                                                                                                                                                                                                                             |

Skip-ratio note: ratios are derived in PromQL
(`rate(scored) / (rate(scored) + rate(skipped))`) rather than emitted, so
they aggregate correctly across instances and windows.

## Dashboard

`grafana/dashboard.json` — importable Grafana dashboard designed for
k8s-deployed Hermes (template variables: datasource / namespace / pod;
cross-pod `histogram_quantile` aggregation; cAdvisor pod-resource row for
page-fault correlation). The server configures explicit histogram buckets so
quantiles aggregate across pods.

## Non-goals / future

- Per-page-fault disk latency: mmap faults are invisible to userspace timers;
  they show up in the phase histograms that contain them (rerank read, BMP
  query). True per-read IO latency requires the Phase 2 DirectIO path.
- Exemplars, per-query tracing: out of scope; use the existing debug logs.