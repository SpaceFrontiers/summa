# Prometheus Metrics

Status: design (2026-07-09), implemented.

## Problem

Query-path performance work (BMP pruning and Seismic nomination quality, rerank IO, memory-bound
degradation) also uses debug log lines and opt-in query work counters
— unusable for dashboards, alerting, or regression tracking in production.

## Design

- **summa-core** emits through the [`metrics`](https://docs.rs/metrics)
  facade behind a new **non-default `metrics` feature** (native-only; wasm
  never enables it). Without an installed recorder the macros are ~1ns no-ops,
  and every emission happens at _aggregation points_ (query end, phase end,
  IO call) — never per block/superblock inside scoring loops, so the hot-path
  allocation/atomics budget is untouched.
- **summa-server** enables the feature, installs
  `metrics-exporter-prometheus`, and serves `GET /metrics` on
  `--metrics-addr` (default `0.0.0.0:9184`).
- Emission sites live behind tiny `observe::*` helper fns so call sites carry
  no `#[cfg]` noise; the helpers compile to nothing when the feature is off.

## Metric set

Histograms are seconds unless noted. `field` labels are **field names** and
**every metric carries an `index` label** (the registry index name, embedded
in the schema at creation; "unknown" for pre-existing indexes until recreated
or patched — see below). Directory-layer metrics (`summa_directory_read_*`,
`summa_cold_write_bytes_total`) have no schema in scope, so the label is
attached late: `Index::open`/`create` call `Directory::set_index_label`
on the index's directory instance once the schema is loaded.

| Metric                                                                                                                  | Type                | Labels                                                      | Meaning                                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `summa_bmp_query_duration_seconds`                                                                                      | histogram           | `field`                                                     | full BMP executor wall time                                                                                                                                                   |
| `summa_bmp_query_prepare_duration_seconds`                                                                              | histogram           | `field`                                                     | shared query resolution/quantization; normally paid once at query-global planning and near-zero in each planned segment                                                       |
| `summa_bmp_d_grid_duration_seconds`                                                                                     | histogram           | `field`                                                     | D-grid decoding and per-block upper-bound calculation                                                                                                                         |
| `summa_bmp_prefetch_duration_seconds`                                                                                   | histogram           | `field`                                                     | userspace time preparing/coalescing D and block-payload `MADV_WILLNEED` ranges (not asynchronous page-fault completion)                                                       |
| `summa_bmp_block_score_duration_seconds`                                                                                | histogram           | `field`                                                     | visited block payload scoring                                                                                                                                                 |
| `summa_bmp_docmap_duration_seconds`                                                                                     | histogram           | `field`                                                     | virtual-record to document/ordinal resolution after threshold rejection                                                                                                       |
| `summa_bmp_prefetched_bytes_total` / `_ranges_total`                                                                    | counter             | `field`                                                     | bounded D/block-payload byte ranges submitted to the kernel                                                                                                                   |
| `summa_bmp_superblocks_visited_total` / `_skipped_total`                                                                | counter             | `field`                                                     | superblock iteration vs pruned — skip ratio via PromQL                                                                                                                        |
| `summa_bmp_blocks_scored_total` / `_skipped_total`                                                                      | counter             | `field`                                                     | block-level skip ratio                                                                                                                                                        |
| `summa_bmp_blocks_scored_per_query`                                                                                     | histogram           | `field`                                                     | per-query distribution (tail queries)                                                                                                                                         |
| `summa_bmp_docmap_lookups_total` / `_per_query`                                                                         | counter / histogram | `field`                                                     | scattered doc-map resolutions for candidates that survive the score floor (strictly lower candidates are rejected before this lookup)                                         |
| `summa_bmp_lsp_duration_seconds`                                                                                        | histogram           | `field`                                                     | query-global H/E hierarchy planning wall time                                                                                                                                 |
| `summa_bmp_lsp_prepare_duration_seconds`                                                                                | histogram           | `field`                                                     | one-time sparse query preparation before segment planning                                                                                                                     |
| `summa_bmp_lsp_h_scan_duration_seconds`                                                                                 | histogram           | `field`                                                     | scan of the pinned coarse H hierarchy across segments                                                                                                                         |
| `summa_bmp_lsp_select_duration_seconds`                                                                                 | histogram           | `field`                                                     | best-first H expansion, selected E-group decoding, and plan construction                                                                                                      |
| `summa_bmp_lsp_superblocks` / `_gamma`                                                                                  | histogram           | `field`                                                     | total physical E cells and configured global selection cap                                                                                                                    |
| `summa_bmp_lsp_coarse_groups` / `_expanded`                                                                             | histogram           | `field`                                                     | total H cells versus cells expanded into E; their ratio shows hierarchy effectiveness                                                                                         |
| `summa_bmp_lsp_superblocks_evaluated`                                                                                   | histogram           | `field`                                                     | E cells actually decoded while finding exact global top-gamma                                                                                                                 |
| `summa_bmp_integrity_fault_queries_total`                                                                               | counter             | `field`                                                     | BMP queries that hit at least one corrupt block, corrupt term range, out-of-range posting slot, or invalid doc-map entry (also `warn!`-logged with back-off)                  |
| `summa_bmp_corrupt_blocks_total` / `_corrupt_terms_total` / `_dropped_postings_total` / `_invalid_docmap_entries_total` | counter             | `field`                                                     | per-kind BMP integrity fault counts; a corrupt block is not counted as scored                                                                                                 |
| `summa_reorder_granularity_total`                                                                                       | counter             | `field`, `granularity` (`records`/`blocks`)                 | reorder passes by chosen granularity (`docs/block-level-reorder.md`)                                                                                                          |
| `summa_reorder_coherence`                                                                                               | histogram           | `field`                                                     | raw block coherence d (avg records per block×dim pair) measured at each reorder decision                                                                                      |
| `summa_reorder_coherence_norm`                                                                                          | histogram           | `field`                                                     | normalized coherence, 0 = random order, 1 = as coherent as the dim-frequency distribution allows — this drives the `Auto` decision (threshold 0.5)                            |
| `summa_seismic_query_duration_seconds`                                                                                  | histogram           | `field`                                                     | sparse candidate traversal and result ordinal hydration time                                                                                                                  |
| `summa_seismic_clusters_total`                                                                                          | counter             | `field`                                                     | clusters visited across nomination runs                                                                                                                                       |
| `summa_seismic_documents_total`                                                                                         | counter             | `field`                                                     | eligible candidate documents scored with complete ordinal aggregation                                                                                                         |
| `summa_seismic_budget_truncations_total`                                                                                | counter             | `field`                                                     | query executions stopped at bounded nomination expansion limits                                                                                                               |
| `summa_sparse_maxscore_query_duration_seconds`                                                                          | histogram           | `field`                                                     | sparse MaxScore executor wall time                                                                                                                                            |
| `summa_dense_l1_duration_seconds`                                                                                       | histogram           | `field`, `kind` (see below)                                 | ANN / brute-force candidate generation                                                                                                                                        |
| `summa_dense_rerank_duration_seconds`                                                                                   | histogram           | `field`                                                     | rerank total (resolve+read+score)                                                                                                                                             |
| `summa_dense_rerank_resolve_duration_seconds`                                                                           | histogram           | `field`                                                     | doc→flat-slot indirection (flat store is not reordered)                                                                                                                       |
| `summa_dense_rerank_read_duration_seconds`                                                                              | histogram           | `field`                                                     | raw-vector reads within rerank (page-fault sensitive)                                                                                                                         |
| `summa_dense_rerank_vectors`                                                                                            | histogram           | `field`                                                     | candidates reranked per query                                                                                                                                                 |
| `summa_directory_read_duration_seconds`                                                                                 | histogram           | `index`, `op` (`lazy_range`)                                | Directory-layer read latency — only Lazy handles (HTTP/custom `read_fn`) do real IO here; mmap slices are zero-copy and their fault latency lands inside the phase histograms |
| `summa_directory_read_bytes`                                                                                            | histogram           | `index`, `op`                                               | read sizes                                                                                                                                                                    |
| `summa_slice_cache_hits_total` / `_misses_total`                                                                        | counter             | `index`                                                     | `SliceCachingDirectory` range reads served from cache vs. forwarded to the inner directory (HTTP/WASM path); hits are flushed in batches of 1024 from an in-process counter   |
| `summa_slice_cache_hit_bytes` / `_miss_bytes`                                                                           | histogram           | `index`                                                     | sizes of cached / forwarded range reads (hit sizes sampled at each flush)                                                                                                     |
| `summa_slice_cache_evicted_slices_total` / `_bytes_total`                                                               | counter             | `index`                                                     | LRU eviction volume; a high rate against a flat hit rate means the cache budget is too small for the working set                                                              |
| `summa_rerank_candidates_skipped_total`                                                                                 | counter             | `index`, `kind` (`dense`/`binary`)                          | L2 rerank candidates dropped because their segment is gone or has no stored vectors for the field (also `warn!`-logged per request)                                           |
| `summa_dense_ann_scans_total`                                                                                           | counter             | `field`, `kind`, `regime` (`serial`/`parallel`)             | dense ANN segment scans by execution regime; the serial-vs-Rayon choice used to be invisible                                                                                  |
| `summa_dense_ann_blocks_scored` / `_blocks_pruned`                                                                      | histogram           | `field`, `kind`                                             | per-scan IVF-TQ/TQ blocks scored vs skipped by the scale upper bound (pruning ratio via PromQL)                                                                               |
| `summa_dense_ann_postings_probed`                                                                                       | histogram           | `field`, `kind`                                             | postings in the probed leaves; what a combined-document (non-Max) scan buffers before grouping                                                                                |
| `summa_dense_ann_non_finite_dropped_total`                                                                              | counter             | `field`, `kind`                                             | candidates dropped because their estimated score was NaN/±inf — a degenerate stored vector or codebook (also `warn!`-logged with back-off)                                    |
| `summa_ann_non_finite_scores_dropped_total`                                                                             | counter             | —                                                           | non-finite leaf scores dropped while combining multi-valued ANN candidates (no field in scope at that layer)                                                                  |
| `summa_scann_fast_scan_degenerate_queries_total`                                                                        | counter             | —                                                           | float ScaNN queries whose FastScan lookup table collapsed to a constant (all-zero AH scores); every leaf row scores as its centroid dot                                       |
| `summa_store_get_duration_seconds`                                                                                      | histogram           | `index`                                                     | document store fetch (decompression + faults), single- and multi-field paths                                                                                                  |
| `summa_cold_write_bytes_total`                                                                                          | counter             | `index`                                                     | merge/reorder bytes written via the page-cache-dropping cold path (`docs/cold-io.md`)                                                                                         |
| `summa_reorder_bp_passes_total`                                                                                         | counter             | `index`, `field`, `entity_kind`, `stop_reason`, `converged` | completed record/block BP graph passes and why refinement stopped (`complete`, `objective`, `time_budget`, or `memory_budget`)                                                |
| `summa_reorder_bp_duration_seconds`                                                                                     | histogram           | `index`, `field`, `entity_kind`                             | BP graph wall time (forward-index construction and blob rewrite are separate surrounding phases)                                                                              |
| `summa_reorder_bp_entities` / `_postings`                                                                               | histogram           | `index`, `field`, `entity_kind`                             | input graph size for each BP pass                                                                                                                                             |
| `summa_reorder_bp_partitions` / `_iterations_per_pass`                                                                  | histogram           | `index`, `field`, `entity_kind`                             | completed recursive partitions and refinement iterations                                                                                                                      |
| `summa_reorder_bp_entity_passes_per_pass` / `_swaps`                                                                    | histogram           | `index`, `field`, `entity_kind`                             | entities visited across gain iterations and entities moved                                                                                                                    |
| `summa_reorder_bp_active_passes`                                                                                        | gauge               | `index`, `field`, `entity_kind`                             | currently running BP graph passes; cancellation/panic-safe                                                                                                                    |
| `summa_search_duration_seconds`                                                                                         | histogram           | `index`, `status`                                           | server: full search RPC                                                                                                                                                       |
| `summa_search_requests_total`                                                                                           | counter             | `index`, `status`                                           | server: RPC outcomes                                                                                                                                                          |

Skip-ratio note: ratios are derived in PromQL
(`rate(scored) / (rate(scored) + rate(skipped))`) rather than emitted, so
they aggregate correctly across instances and windows.

Dense L1 `kind` labels follow the active reader path: `tq_flat`, `ivf_tq`,
`scann_ah`, `binary_ivf`, `scann_binary`, `global_binary_ivf`, `binary_scann`,
`binary_flat`, or `flat`. The binary names differ between the shared and
binary-specific entry points. IVF-PQ has been removed. The source of truth is
[`dense_ann_kind_label` and its call sites](../summa-core/src/segment/reader/mod.rs).

### Broker metrics (`summa-broker`)

Emitted by the broker process (see [broker.md](broker.md)), same conventions.
Every fallback path the broker can take is counted — nothing degrades
silently.

| Metric                                          | Type      | Labels                   | Meaning                                                                                                  |
| ----------------------------------------------- | --------- | ------------------------ | -------------------------------------------------------------------------------------------------------- |
| `summa_broker_search_duration_seconds`          | histogram | `index`, `status`        | full broker-side Search RPC                                                                              |
| `summa_broker_search_requests_total`            | counter   | `index`, `status`        | Search outcomes                                                                                          |
| `summa_broker_admission_rejected_total`         | counter   | `index`, `scope`         | try-acquire rejections before any backend RPC (`global` \| `backend`)                                    |
| `summa_broker_backend_request_duration_seconds` | histogram | `backend`, `rpc`         | outbound RPC latency per backend                                                                         |
| `summa_broker_backend_requests_total`           | counter   | `backend`, `rpc`, `code` | outbound outcomes by gRPC code                                                                           |
| `summa_broker_backend_healthy`                  | gauge     | `backend`, `shard`       | 1 healthy / 0.5 suspect / 0 evicted                                                                      |
| `summa_broker_backends`                         | gauge     | `state`                  | discovered backend counts (`healthy`/`suspect`/`evicted`/`unready`)                                      |
| `summa_broker_discovery_events_total`           | counter   | `type`                   | pod watcher churn (`added`/`removed`)                                                                    |
| `summa_broker_index_map_age_seconds`            | gauge     | `backend`                | staleness of the backend's learned index list                                                            |
| `summa_broker_ambiguous_index_total`            | counter   | `index`                  | reads routed to a deterministically-picked shard because the index exists on several shards with no rule |
| `summa_broker_stale_topology_serves_total`      | counter   | `backend`                | reads served by a Suspect backend off its last-known index map                                           |
| `summa_broker_write_rejected_total`             | counter   | `index`, `reason`        | write RPCs the broker itself refused (reason = gRPC code label)                                          |
| `summa_broker_stream_flushes_total`             | counter   | `index`                  | streaming IndexDocuments runs forwarded as BatchIndexDocuments                                           |

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
k8s-deployed Summa (template variables: datasource / namespace / pod;
cross-pod `histogram_quantile` aggregation; cAdvisor pod-resource row for
page-fault correlation). The server configures explicit histogram buckets so
quantiles aggregate across pods.

The **Search time breakdown by operation** panel (top of the overview row)
stacks the mean time each disjoint phase — sparse candidate search (Seismic), MaxScore
(DAAT), ANN L1 candidate gen, rerank, store fetch, directory read —
contributes per search request: `rate(<phase>_duration_seconds_sum) /
rate(summa_search_requests_total)`. Because segments/sub-queries run
concurrently, the stack is CPU-time-equivalent per request, not wall-clock
latency — read it for _where time goes_, and the p50/p95/p99 latency panels
for wall-clock.

## Non-goals / future

- Per-page-fault disk latency: mmap faults are invisible to userspace timers;
  they show up in the phase histograms that contain them (rerank read, Seismic
  query). True per-read IO latency requires the Phase 2 DirectIO path.
- Exemplars, per-query tracing: out of scope; use the existing debug logs.

Seismic opt-in query diagnostics also report forward rows and encoded forward bytes
scored, including exact backfill and ordinal hydration. The fixed-size counters
merge across segment workers; they allocate no per-dimension labels.
