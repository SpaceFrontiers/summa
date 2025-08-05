//! Broker metric names.
//!
//! Naming follows docs/metrics.md conventions: seconds histograms, `_total`
//! counters, and an `index` label wherever a request is index-scoped. Every
//! fallback path the broker can take (ambiguous routing, stale-topology
//! serving, admission rejection, write refusal) has a counter so it is
//! observable rather than silent.

/// Full broker-side Search RPC duration. Labels: `index`, `status`.
pub const SEARCH_DURATION: &str = "hermes_broker_search_duration_seconds";
/// Search RPC outcomes. Labels: `index`, `status`.
pub const SEARCH_REQUESTS: &str = "hermes_broker_search_requests_total";
/// Admission rejections before any backend RPC. Labels: `index`, `scope`
/// (`global` | `backend`).
pub const ADMISSION_REJECTED: &str = "hermes_broker_admission_rejected_total";
/// Outbound RPC latency. Labels: `backend`, `rpc`.
pub const BACKEND_DURATION: &str = "hermes_broker_backend_request_duration_seconds";
/// Outbound RPC outcomes by gRPC code. Labels: `backend`, `rpc`, `code`.
pub const BACKEND_REQUESTS: &str = "hermes_broker_backend_requests_total";
/// 1 healthy / 0.5 suspect / 0 evicted. Labels: `backend`, `shard`.
pub const BACKEND_HEALTHY: &str = "hermes_broker_backend_healthy";
/// Discovered backend counts. Labels: `state` (`healthy`|`suspect`|`evicted`|`unready`).
pub const BACKENDS: &str = "hermes_broker_backends";
/// Discovery churn. Labels: `type` (`added`|`removed`).
pub const DISCOVERY_EVENTS: &str = "hermes_broker_discovery_events_total";
/// Staleness of a backend's learned index list. Labels: `backend`.
pub const INDEX_MAP_AGE: &str = "hermes_broker_index_map_age_seconds";
/// Reads routed to a deterministically-picked shard because the index name was
/// seen on several shard ids without a placement rule. Labels: `index`.
pub const AMBIGUOUS_INDEX: &str = "hermes_broker_ambiguous_index_total";
/// Reads served by a Suspect backend off its last-known index map. Labels: `backend`.
pub const STALE_TOPOLOGY_SERVES: &str = "hermes_broker_stale_topology_serves_total";
/// Write RPCs refused by the broker itself. Labels: `index`, `reason`
/// (`ambiguous` | `no_master` | `master_unavailable` | `no_placement` | `unknown_index`).
pub const WRITE_REJECTED: &str = "hermes_broker_write_rejected_total";
/// Streaming IndexDocuments flushes forwarded as BatchIndexDocuments. Labels: `index`.
pub const STREAM_FLUSHES: &str = "hermes_broker_stream_flushes_total";
