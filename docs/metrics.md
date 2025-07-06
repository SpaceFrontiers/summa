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
**every metric carries an `index` label** (the registry index name, embedded
in the schema at creation; "unknown" for pre-existing indexes until recreated
or patched — see below). Directory-layer metrics (`hermes_directory_read_*`,
`hermes_cold_write_bytes_total`) have no schema in scope, so the label is
attached late: `Index::open`/`create` call `Directory::set_index_label`
on the index's directory instance once the schema is loaded.

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
| `hermes_directory_read_duration_seconds`                  | histogram           | `index`, `op` (`lazy_range`)                                        | Directory-layer read latency — only Lazy handles (HTTP/custom `read_fn`) do real IO here; mmap slices are zero-copy and their fault latency lands inside the phase histograms                                                                    |
| `hermes_directory_read_bytes`                             | histogram           | `index`, `op`                                                       | read sizes                                                                                                                                                                                                                                       |
| `hermes_store_get_duration_seconds`                       | histogram           | `index`                                                             | document store fetch (decompression + faults), single- and multi-field paths                                                                                                                                                                     |
| `hermes_cold_write_bytes_total`                           | counter             | `index`                                                             | merge/reorder bytes written via the page-cache-dropping cold path (`docs/cold-io.md`)                                                                                                                                                            |
| `hermes_reorder_granularity_total`                        | counter             | `field`, `granularity` (`records`/`blocks`)                         | reorder passes by chosen granularity (`docs/block-level-reorder.md`)                                                                                                                                                                             |
| `hermes_reorder_coherence`                                | histogram           | `field`                                                             | raw block coherence d (avg records per block×dim pair) measured at each reorder decision                                                                                                                                                         |
| `hermes_reorder_coherence_norm`                           | histogram           | `field`                                                             | normalized coherence, 0 = random order, 1 = as coherent as the dim-frequency distribution allows — this drives the `Auto` decision (threshold 0.5)                                                                                               |
| `hermes_search_duration_seconds`                          | histogram           | `index`, `status`                                                   | server: full search RPC                                                                                                                                                                                                                          |
| `hermes_search_requests_total`                            | counter             | `index`, `status`                                                   | server: RPC outcomes                                                                                                                                                                                                                             |

Skip-ratio note: ratios are derived in PromQL
(`rate(scored) / (rate(scored) + rate(skipped))`) rather than emitted, so
they aggregate correctly across instances and windows.

### Pre-existing indexes and `index="unknown"`

Indexes created before the label shipped have no `index_name` in their
stored schema. Recreate them, or patch the stored metadata in place. Do it
with the server stopped (or the index not open for writing) — the server
rewrites `metadata.json` on commit/merge and would overwrite a live patch:

```bash
cd <data_dir>/<index_name>
jq --arg name "<index_name>" '.schema.index_name = $name' metadata.json \
  > metadata.json.patched && mv metadata.json.patched metadata.json
```

Note: `metadata.json.tmp` is reserved for crash recovery — never use it as
a scratch name.

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