//! Hermes gRPC Search Server

mod converters;
mod error;
mod index_service;
mod optimizer;
mod registry;
mod search_service;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use log::{info, warn};
use tonic::{codec::CompressionEncoding, transport::Server};

use hermes_core::IndexConfig;
use hermes_core::segment::pin::{PinMode, PinPolicy, set_pin_policy};

pub mod proto {
    tonic::include_proto!("hermes");
}

use proto::index_service_server::IndexServiceServer;
use proto::search_service_server::SearchServiceServer;

/// Hermes gRPC Search Server
#[derive(Parser, Debug)]
#[command(name = "hermes-server")]
#[command(about = "A high-performance async search server")]
struct Args {
    /// Address to bind to
    #[arg(short, long, default_value = "0.0.0.0:50051")]
    addr: String,

    /// Data directory for indexes
    #[arg(short, long, default_value = "./data")]
    data_dir: PathBuf,

    /// Cache directory for HuggingFace models/tokenizers
    #[arg(short, long)]
    cache_dir: Option<PathBuf>,

    /// Max indexing memory (MB) before auto-flush (global across all builders)
    #[arg(long, default_value = "16384")]
    max_indexing_memory_mb: usize,

    /// Process-wide budget (MB) for decompressed stored-document blocks.
    /// The cache is shared by every index and byte-bounded; blocks larger than
    /// 8 MiB bypass it so one large stored body cannot displace thousands of
    /// ordinary result blocks. Set to 0 to disable decompressed store caching.
    #[arg(long, default_value = "2048")]
    store_cache_budget_mb: usize,

    /// Maximum number of vectors sampled for one field's global ANN training.
    /// The memory limit below is enforced simultaneously.
    #[arg(long, default_value = "10000000")]
    vector_training_max_samples: usize,

    /// Maximum resident raw sample size (MB) for one ANN field during training.
    /// Fields are sampled and trained serially.
    #[arg(long, default_value = "4096")]
    vector_training_memory_mb: usize,

    /// Number of parallel indexing threads (defaults to CPU count)
    #[arg(long)]
    indexing_threads: Option<usize>,

    /// Reload interval in milliseconds for searcher to check for new segments
    /// Higher values reduce reload overhead during heavy indexing
    #[arg(long, default_value = "1000")]
    reload_interval_ms: u64,

    /// Maximum number of tokio worker threads (default: min(cpus, 16))
    #[arg(long)]
    worker_threads: Option<usize>,

    /// Maximum number of search RPCs executing concurrently across all
    /// connections (default: one per 8 CPUs, clamped to 1..=8)
    #[arg(long)]
    max_concurrent_searches: Option<usize>,

    /// Rayon threads shared by CPU-bound search work across every index
    /// (default: one per 4 CPUs, minimum 1)
    #[arg(long)]
    search_threads: Option<usize>,

    /// Maximum BMP segment scorers issuing random mmap reads concurrently
    /// across all indexes and queries.
    #[arg(long, default_value = "4")]
    bmp_io_concurrency: usize,

    /// Validate all indexes on startup, remove corrupt segments
    #[arg(long)]
    doctor: bool,

    /// Rayon worker threads shared by all BP reorder passes (0 = optimizer disabled;
    /// merge-time BP uses its default cores/2 pool)
    #[arg(long, default_value = "0")]
    optimizer_threads: usize,

    /// Maximum simultaneous whole-segment BP passes across optimizer,
    /// merge-time, and manual reorders. Clamped to 1..=2 because each pass can
    /// use all optimizer threads and up to the full BP memory budget. With two
    /// slots, automatic merges use at most one so optimizer work cannot starve.
    #[arg(long, default_value = "2")]
    optimizer_concurrent_passes: usize,

    /// Interval in seconds between optimizer scans for unreordered segments
    #[arg(long, default_value = "60")]
    optimizer_scan_interval_secs: u64,

    /// Segments with at least this many docs get budgeted (partial) BP passes
    #[arg(long, default_value = "5000000")]
    optimizer_large_segment_docs: u32,

    /// Wall-clock budget in seconds per BP pass on large segments
    #[arg(long, default_value = "600")]
    optimizer_time_budget_secs: u64,

    /// Depth cap for large-segment BP: stop at partitions of this many vectors
    /// (256 = one default 32-vector × 8-block LSP superblock)
    #[arg(long, default_value = "256")]
    optimizer_partial_min_partition_docs: usize,

    /// Cooldown in seconds between deepening passes on budget-truncated segments
    #[arg(long, default_value = "600")]
    optimizer_unconverged_cooldown_secs: u64,

    /// Maximum consecutive budget-exhausted rewrites that remain eligible for
    /// optimizer follow-up, including the initial partial pass. 0 disables
    /// follow-up deepening.
    #[arg(long, default_value = "3")]
    optimizer_max_unconverged_passes: u32,

    /// Wall-clock budget in seconds for merge-time BP reorder (0 = unbudgeted).
    /// A truncated pass still produces a valid, better-ordered segment; it is
    /// marked unconverged and the background optimizer deepens it later.
    #[arg(long, default_value = "600")]
    merge_bp_budget_secs: u64,

    /// Memory budget (MB) for the main BP/rewrite working set of one pass
    /// (merge-time and background). A cap, not an allocation. Over-budget
    /// record passes fall back to blockwise order and/or drop graph dimensions.
    #[arg(long, default_value = "24576")]
    bp_memory_budget_mb: usize,

    /// Budget (MB) for pinning hot per-segment metadata resident in RAM
    /// (BMP block offsets/coarse hierarchy, sparse skips, doc-id maps). 0 = off.
    /// Overrides the HERMES_PIN_METADATA_BUDGET_MB env var when set.
    #[arg(long)]
    pin_metadata_budget_mb: Option<u64>,

    /// How pinned metadata is held resident: `mlock` (lock mmap pages in place,
    /// needs RLIMIT_MEMLOCK headroom) or `copy` (heap copy, no permissions).
    /// Overrides the HERMES_PIN_MODE env var when set.
    #[arg(long, value_enum)]
    pin_mode: Option<PinModeArg>,

    /// Maximum background segment merges running at once (per index and the
    /// application-wide cap). Raise it when continuous ingestion outpaces
    /// merging and segment counts climb; keep it below the CPU/IO the box can
    /// spare from search. (large_scale default: 4)
    #[arg(long, default_value = "4")]
    max_concurrent_merges: usize,

    /// Tiered merge: segments allowed per tier before the tier is merged.
    /// Lower = fewer, larger segments (ideal count ~= num_tiers * this).
    /// (large_scale default: 10)
    #[arg(long, default_value = "10")]
    segments_per_tier: usize,

    /// Tiered merge: maximum segments merged in a single pass. Wider fan-in
    /// absorbs floods of small memtable flushes in fewer passes.
    /// (large_scale default: 24)
    #[arg(long, default_value = "24")]
    max_merge_at_once: usize,

    /// Tiered merge: maximum docs produced by one automatic merge. Also caps
    /// the working set of a single merge-time BP reorder — raising it past
    /// what the BP memory/time budgets can converge stalls the optimizer.
    /// (large_scale default: 5000000)
    #[arg(long, default_value = "5000000")]
    max_merged_docs: u32,

    /// Tiered merge: absolute cap on docs in one segment, honored by automatic
    /// merging and force-merge. Sets the floor on segment count
    /// (>= total_docs / this). (large_scale default: 5000000)
    #[arg(long, default_value = "5000000")]
    max_segment_docs: u32,

    /// Address for the Prometheus /metrics HTTP endpoint.
    /// Set to "off" to disable the exporter.
    #[arg(long, default_value = "0.0.0.0:9184")]
    metrics_addr: String,

    /// Maximum decoded (received) gRPC message size in MiB for SearchService
    /// requests. Queries are small; raise only for very large query vectors.
    #[arg(long, default_value = "4")]
    search_max_decode_mb: usize,

    /// Maximum encoded (sent) gRPC message size in MiB for SearchService
    /// responses. Must be at least --max-search-response-mb plus framing
    /// headroom; a hermes-broker in front must mirror it in
    /// --backend-max-decode-mb.
    #[arg(long, default_value = "256")]
    search_max_encode_mb: usize,

    /// Maximum decoded (received) gRPC message size in MiB for IndexService
    /// requests (bounds one indexing batch).
    #[arg(long, default_value = "256")]
    index_max_decode_mb: usize,

    /// Maximum encoded (sent) gRPC message size in MiB for IndexService
    /// responses.
    #[arg(long, default_value = "64")]
    index_max_encode_mb: usize,

    /// Hydration budget in MiB for one search response: caps estimated
    /// retained heap while the response is built and its encoded size.
    /// Must not exceed --search-max-encode-mb; downstream broker/client
    /// decode caps must stay above it.
    #[arg(long, default_value = "192")]
    max_search_response_mb: usize,

    /// Results returned when SearchRequest.limit is 0
    #[arg(long, default_value = "10")]
    default_search_limit: usize,

    /// Upper bound on SearchRequest.limit
    #[arg(long, default_value = "10000")]
    max_search_limit: usize,

    /// Upper bound on SearchRequest.offset + limit (pagination window)
    #[arg(long, default_value = "50000")]
    max_search_window: usize,

    /// Upper bound on the first-stage candidate pool
    /// (SearchRequest.candidate_limit)
    #[arg(long, default_value = "50000")]
    max_candidate_limit: usize,

    // Structural anti-DoS budgets for one decoded request. The transport's
    // decode limit bounds wire bytes, not decoded object count or downstream
    // expansion, so these are independent budgets.
    /// Maximum query tree nesting depth
    #[arg(long, default_value = "32")]
    max_query_depth: usize,

    /// Maximum query tree nodes per request
    #[arg(long, default_value = "256")]
    max_query_nodes: usize,

    /// Maximum aggregate clauses across the query tree
    #[arg(long, default_value = "512")]
    max_query_clauses: usize,

    /// Maximum clauses in a single BooleanQuery
    #[arg(long, default_value = "128")]
    max_boolean_clauses: usize,

    /// Maximum aggregate query text bytes (terms, match text, field names)
    #[arg(long, default_value = "65536")]
    max_query_text_bytes: usize,

    /// Maximum bytes in one field name
    #[arg(long, default_value = "255")]
    max_field_name_bytes: usize,

    /// Maximum bytes in the index name
    #[arg(long, default_value = "255")]
    max_index_name_bytes: usize,

    /// Maximum elements in one dense query/reranker vector
    #[arg(long, default_value = "65536")]
    max_dense_query_dims: usize,

    /// Maximum entries in one sparse query vector
    #[arg(long, default_value = "4096")]
    max_sparse_query_dims: usize,

    /// Maximum bytes in one binary query/reranker vector
    #[arg(long, default_value = "262144")]
    max_binary_query_bytes: usize,

    /// Maximum aggregate query vector bytes per request
    #[arg(long, default_value = "1048576")]
    max_total_query_vector_bytes: usize,

    /// Maximum names in SearchRequest.fields_to_load
    #[arg(long, default_value = "64")]
    max_fields_to_load: usize,

    /// Maximum aggregate bytes across fields_to_load names
    #[arg(long, default_value = "16384")]
    max_fields_to_load_name_bytes: usize,

    /// Maximum tokens a Term/Match query may expand to after tokenization
    #[arg(long, default_value = "256")]
    max_text_query_tokens: usize,

    /// Maximum dimensions a SparseVectorQuery text may tokenize into
    #[arg(long, default_value = "4096")]
    max_sparse_token_dimensions: usize,
}

/// gRPC transport envelope and request-facing search limits resolved from CLI
/// flags, cross-validated so an inconsistent configuration fails at startup
/// instead of rejecting every oversized response at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ServiceLimits {
    search_max_decode_bytes: usize,
    search_max_encode_bytes: usize,
    index_max_decode_bytes: usize,
    index_max_encode_bytes: usize,
    search: search_service::SearchLimits,
}

fn nonzero_mb_to_bytes(flag: &str, mb: usize) -> Result<usize> {
    if mb == 0 {
        return Err(anyhow::anyhow!("{flag} must be greater than zero"));
    }
    mb.checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("{flag} is too large"))
}

fn nonzero(flag: &str, value: usize) -> Result<usize> {
    if value == 0 {
        return Err(anyhow::anyhow!("{flag} must be greater than zero"));
    }
    Ok(value)
}

fn resolve_query_shape_limits(args: &Args) -> Result<search_service::QueryShapeLimits> {
    let shape = search_service::QueryShapeLimits {
        max_query_depth: nonzero("--max-query-depth", args.max_query_depth)?,
        max_query_nodes: nonzero("--max-query-nodes", args.max_query_nodes)?,
        max_query_clauses: nonzero("--max-query-clauses", args.max_query_clauses)?,
        max_boolean_clauses: nonzero("--max-boolean-clauses", args.max_boolean_clauses)?,
        max_query_text_bytes: nonzero("--max-query-text-bytes", args.max_query_text_bytes)?,
        max_field_name_bytes: nonzero("--max-field-name-bytes", args.max_field_name_bytes)?,
        max_index_name_bytes: nonzero("--max-index-name-bytes", args.max_index_name_bytes)?,
        max_dense_query_dims: nonzero("--max-dense-query-dims", args.max_dense_query_dims)?,
        max_sparse_query_dims: nonzero("--max-sparse-query-dims", args.max_sparse_query_dims)?,
        max_binary_query_bytes: nonzero("--max-binary-query-bytes", args.max_binary_query_bytes)?,
        max_total_query_vector_bytes: nonzero(
            "--max-total-query-vector-bytes",
            args.max_total_query_vector_bytes,
        )?,
        max_fields_to_load: nonzero("--max-fields-to-load", args.max_fields_to_load)?,
        max_fields_to_load_name_bytes: nonzero(
            "--max-fields-to-load-name-bytes",
            args.max_fields_to_load_name_bytes,
        )?,
        max_text_query_tokens: nonzero("--max-text-query-tokens", args.max_text_query_tokens)?,
        max_sparse_token_dimensions: nonzero(
            "--max-sparse-token-dimensions",
            args.max_sparse_token_dimensions,
        )?,
    };
    if shape.max_boolean_clauses > shape.max_query_clauses {
        return Err(anyhow::anyhow!(
            "--max-boolean-clauses ({}) must not exceed --max-query-clauses ({}): a single \
             BooleanQuery could never reach its own cap",
            shape.max_boolean_clauses,
            shape.max_query_clauses,
        ));
    }
    Ok(shape)
}

fn resolve_service_limits(args: &Args) -> Result<ServiceLimits> {
    let search_max_decode_bytes =
        nonzero_mb_to_bytes("--search-max-decode-mb", args.search_max_decode_mb)?;
    let search_max_encode_bytes =
        nonzero_mb_to_bytes("--search-max-encode-mb", args.search_max_encode_mb)?;
    let index_max_decode_bytes =
        nonzero_mb_to_bytes("--index-max-decode-mb", args.index_max_decode_mb)?;
    let index_max_encode_bytes =
        nonzero_mb_to_bytes("--index-max-encode-mb", args.index_max_encode_mb)?;
    let max_search_response_bytes =
        nonzero_mb_to_bytes("--max-search-response-mb", args.max_search_response_mb)?;

    if search_max_encode_bytes < max_search_response_bytes {
        return Err(anyhow::anyhow!(
            "--search-max-encode-mb ({}) must be at least --max-search-response-mb ({}) \
             or every response near the hydration budget fails to encode",
            args.search_max_encode_mb,
            args.max_search_response_mb,
        ));
    }
    if args.default_search_limit == 0 {
        return Err(anyhow::anyhow!(
            "--default-search-limit must be greater than zero"
        ));
    }
    if args.default_search_limit > args.max_search_limit {
        return Err(anyhow::anyhow!(
            "--default-search-limit ({}) must not exceed --max-search-limit ({})",
            args.default_search_limit,
            args.max_search_limit,
        ));
    }
    if args.max_search_window < args.max_search_limit {
        return Err(anyhow::anyhow!(
            "--max-search-window ({}) must be at least --max-search-limit ({})",
            args.max_search_window,
            args.max_search_limit,
        ));
    }
    if args.max_candidate_limit < args.max_search_window {
        return Err(anyhow::anyhow!(
            "--max-candidate-limit ({}) must be at least --max-search-window ({}) \
             so every allowed page can fill its candidate pool",
            args.max_candidate_limit,
            args.max_search_window,
        ));
    }

    Ok(ServiceLimits {
        search_max_decode_bytes,
        search_max_encode_bytes,
        index_max_decode_bytes,
        index_max_encode_bytes,
        search: search_service::SearchLimits {
            default_search_limit: args.default_search_limit,
            max_search_limit: args.max_search_limit,
            max_search_window: args.max_search_window,
            max_candidate_limit: args.max_candidate_limit,
            shape: resolve_query_shape_limits(args)?,
            max_search_response_bytes,
        },
    })
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum PinModeArg {
    Mlock,
    Copy,
}

impl From<PinModeArg> for PinMode {
    fn from(m: PinModeArg) -> Self {
        match m {
            PinModeArg::Mlock => PinMode::Mlock,
            PinModeArg::Copy => PinMode::Copy,
        }
    }
}

fn main() -> Result<()> {
    // Install panic hook that logs backtrace
    // This catches panics in spawned tasks that would otherwise be silent
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let bt = std::backtrace::Backtrace::force_capture();
        eprintln!("=== PANIC ===\n{info}\n{bt}");
        default_hook(info);
    }));

    // Initialize logging
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("hermes_server=info"),
    )
    .init();

    let args = Args::parse();

    let worker_threads = args
        .worker_threads
        .unwrap_or_else(|| num_cpus::get().min(16));

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .thread_name("hermes-worker")
        .thread_stack_size(4 * 1024 * 1024)
        .enable_all()
        .build()?;

    runtime.block_on(async_main(args, worker_threads))
}

async fn async_main(args: Args, worker_threads: usize) -> Result<()> {
    // Set HuggingFace cache directory if specified
    if let Some(cache_dir) = &args.cache_dir {
        std::fs::create_dir_all(cache_dir)?;
        // SAFETY: We set this env var before any threads are spawned
        unsafe { std::env::set_var("HF_HOME", cache_dir) };
        info!("HuggingFace cache directory: {:?}", cache_dir);
    }

    // Prometheus exporter: query-path metrics from hermes-core (BMP pruning,
    // rerank phases, doc-map indirection, ...) + RPC-level metrics. Fail loud:
    // a bad address or bind failure aborts startup rather than silently
    // serving without metrics.
    if args.metrics_addr != "off" {
        let metrics_addr: SocketAddr = args.metrics_addr.parse().map_err(|e| {
            anyhow::anyhow!("invalid --metrics-addr '{}': {}", args.metrics_addr, e)
        })?;
        metrics_exporter_prometheus::PrometheusBuilder::new()
            .with_http_listener(metrics_addr)
            .install()
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed to start metrics exporter on {}: {}",
                    metrics_addr,
                    e
                )
            })?;
        info!("Prometheus metrics on http://{}/metrics", metrics_addr);
    } else {
        warn!("Prometheus metrics exporter disabled (--metrics-addr off)");
    }

    // Create data directory if needed
    std::fs::create_dir_all(&args.data_dir)?;

    // Hot-metadata pinning policy. CLI flags take precedence; each unset flag
    // falls back to its HERMES_PIN_* environment value. Must be set
    // before the first segment is opened (registry cleanup / doctor below).
    let env_policy = PinPolicy::from_env();
    let pin_budget_bytes = match args.pin_metadata_budget_mb {
        Some(mb) => mb
            .checked_mul(1024 * 1024)
            .ok_or_else(|| anyhow::anyhow!("--pin-metadata-budget-mb is too large"))?,
        None => env_policy.budget_bytes,
    };
    let pin_policy = PinPolicy {
        budget_bytes: pin_budget_bytes,
        mode: args.pin_mode.map(PinMode::from).unwrap_or(env_policy.mode),
    };
    set_pin_policy(pin_policy);
    if pin_policy.is_enabled() {
        info!(
            "Hot-metadata pinning: {} MB budget, {:?} mode",
            pin_budget_bytes / (1024 * 1024),
            pin_policy.mode,
        );
    }

    let addr: SocketAddr = args.addr.parse()?;

    let service_limits = resolve_service_limits(&args)?;

    let num_indexing_threads = args
        .indexing_threads
        .unwrap_or_else(hermes_core::default_indexing_threads);
    let max_concurrent_searches = args
        .max_concurrent_searches
        .unwrap_or_else(|| (num_cpus::get() / 8).clamp(1, 8));
    if max_concurrent_searches == 0 {
        return Err(anyhow::anyhow!(
            "--max-concurrent-searches must be greater than zero"
        ));
    }
    let search_threads = args
        .search_threads
        .unwrap_or_else(hermes_core::default_search_threads);
    if search_threads == 0 {
        return Err(anyhow::anyhow!(
            "--search-threads must be greater than zero"
        ));
    }
    if args.bmp_io_concurrency == 0 {
        return Err(anyhow::anyhow!(
            "--bmp-io-concurrency must be greater than zero"
        ));
    }

    let max_indexing_memory_bytes = args
        .max_indexing_memory_mb
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--max-indexing-memory-mb is too large"))?;
    let store_cache_budget_bytes = args
        .store_cache_budget_mb
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--store-cache-budget-mb is too large"))?;
    if args.vector_training_max_samples == 0 || args.vector_training_memory_mb == 0 {
        return Err(anyhow::anyhow!(
            "--vector-training-max-samples and --vector-training-memory-mb must be greater than zero"
        ));
    }
    let vector_training_memory_bytes = args
        .vector_training_memory_mb
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--vector-training-memory-mb is too large"))?;
    let bp_memory_budget_bytes = args
        .bp_memory_budget_mb
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--bp-memory-budget-mb is too large"))?;
    let merge_bp_time_budget = merge_bp_time_budget(args.merge_bp_budget_secs);
    let concurrent_reorder_passes = args
        .optimizer_concurrent_passes
        .clamp(1, hermes_core::index::MAX_CONCURRENT_REORDER_PASSES);
    if concurrent_reorder_passes != args.optimizer_concurrent_passes {
        warn!(
            "--optimizer-concurrent-passes={} is outside the supported 1..={} range; using {}",
            args.optimizer_concurrent_passes,
            hermes_core::index::MAX_CONCURRENT_REORDER_PASSES,
            concurrent_reorder_passes,
        );
    }

    // One application-owned pool is cloned into every index. Optimizer and
    // merge-time BP therefore share a fixed number of OS threads instead of
    // multiplying pools per index and per subsystem.
    let background_reorder_pool = if args.optimizer_threads > 0 {
        Some(Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(args.optimizer_threads)
                .thread_name(|idx| format!("hermes-bp-{}", idx))
                .build()
                .map_err(|e| anyhow::anyhow!("failed to create BP thread pool: {}", e))?,
        ))
    } else {
        None
    };

    // Merge tuning: start from the large_scale preset and apply CLI overrides.
    // Fail loud on values that would disable merging or produce degenerate
    // tiers rather than silently clamping.
    if args.max_concurrent_merges == 0 {
        return Err(anyhow::anyhow!(
            "--max-concurrent-merges must be greater than zero"
        ));
    }
    if args.segments_per_tier < 2 {
        return Err(anyhow::anyhow!("--segments-per-tier must be at least 2"));
    }
    if args.max_merge_at_once < 2 {
        return Err(anyhow::anyhow!("--max-merge-at-once must be at least 2"));
    }
    if args.max_merged_docs == 0 || args.max_segment_docs == 0 {
        return Err(anyhow::anyhow!(
            "--max-merged-docs and --max-segment-docs must be greater than zero"
        ));
    }
    if args.max_merged_docs > args.max_segment_docs {
        warn!(
            "--max-merged-docs ({}) exceeds --max-segment-docs ({}); merges are capped by the smaller value",
            args.max_merged_docs, args.max_segment_docs,
        );
    }
    let mut merge_policy = hermes_core::merge::TieredMergePolicy::large_scale();
    merge_policy.segments_per_tier = args.segments_per_tier;
    merge_policy.max_merge_at_once = args.max_merge_at_once;
    merge_policy.max_merged_docs = args.max_merged_docs;
    merge_policy.max_segment_docs = args.max_segment_docs;

    let config = IndexConfig {
        num_threads: search_threads,
        bmp_io_concurrency: args.bmp_io_concurrency,
        store_cache_budget_bytes,
        max_indexing_memory_bytes,
        vector_training_max_samples: args.vector_training_max_samples,
        vector_training_memory_bytes,
        num_indexing_threads,
        reload_interval_ms: args.reload_interval_ms,
        merge_policy: Box::new(merge_policy),
        max_concurrent_merges: args.max_concurrent_merges,
        background_merge_permits: Arc::new(tokio::sync::Semaphore::new(args.max_concurrent_merges)),
        merge_bp_time_budget,
        bp_memory_budget_bytes,
        background_reorder_permits: Arc::new(hermes_core::index::ReorderConcurrencyGate::new(
            concurrent_reorder_passes,
        )),
        background_reorder_pool,
        ..Default::default()
    };

    let registry = Arc::new(registry::IndexRegistry::new(args.data_dir.clone(), config));

    // Clean up index directories from incomplete deletes (e.g. server crashed mid-delete)
    registry.cleanup_incomplete_deletes();

    if args.doctor {
        registry.doctor_all_indexes().await;
    }

    let search_service = search_service::SearchServiceImpl::new(
        Arc::clone(&registry),
        max_concurrent_searches,
        service_limits.search,
    );

    let index_service = index_service::IndexServiceImpl {
        registry: Arc::clone(&registry),
    };

    // One application-level signal coordinates tonic admission, optimizer
    // admission, and index lifecycle cancellation.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    // Spawn background optimizer
    let optimizer_handle = optimizer::spawn_optimizer(
        Arc::clone(&registry),
        optimizer::OptimizerConfig {
            threads: args.optimizer_threads,
            concurrent_passes: concurrent_reorder_passes,
            scan_interval: Duration::from_secs(args.optimizer_scan_interval_secs),
            large_segment_docs: args.optimizer_large_segment_docs,
            time_budget: Duration::from_secs(args.optimizer_time_budget_secs),
            partial_min_partition_docs: args.optimizer_partial_min_partition_docs,
            unconverged_cooldown: Duration::from_secs(args.optimizer_unconverged_cooldown_secs),
            max_unconverged_passes: args.optimizer_max_unconverged_passes,
        },
        shutdown_rx,
    );

    info!("Hermes server v{}", env!("CARGO_PKG_VERSION"));
    info!("Starting Hermes server on {}", addr);
    info!("Data directory: {:?}", args.data_dir);
    info!("Max indexing memory: {} MB", args.max_indexing_memory_mb);
    info!(
        "Shared document-store cache budget: {} MB",
        args.store_cache_budget_mb
    );
    info!(
        "Vector training sample: max {} vectors / {} MB per field",
        args.vector_training_max_samples, args.vector_training_memory_mb,
    );
    info!("Indexing threads: {}", num_indexing_threads);
    info!("Worker threads: {}", worker_threads);
    info!("Search CPU threads: {}", search_threads);
    info!("BMP random-I/O concurrency: {}", args.bmp_io_concurrency);
    info!("Maximum concurrent searches: {}", max_concurrent_searches);
    info!("Reload interval: {} ms", args.reload_interval_ms);
    info!(
        "Merge: {} concurrent, tiered(segments_per_tier={}, max_merge_at_once={}, max_merged_docs={}, max_segment_docs={})",
        args.max_concurrent_merges,
        args.segments_per_tier,
        args.max_merge_at_once,
        args.max_merged_docs,
        args.max_segment_docs,
    );
    match merge_bp_time_budget {
        Some(budget) => info!(
            "Merge BP time budget: {:.0}s per field",
            budget.as_secs_f64()
        ),
        None => info!("Merge BP time budget: unbudgeted"),
    }
    if args.optimizer_threads > 0 {
        info!(
            "Optimizer: {} shared BP threads, {} concurrent pass(es), {}s scan interval, {}-pass unconverged follow-up threshold",
            args.optimizer_threads,
            concurrent_reorder_passes,
            args.optimizer_scan_interval_secs,
            args.optimizer_max_unconverged_passes,
        );
    }

    info!(
        "gRPC message caps: search {}/{} MiB decode/encode, index {}/{} MiB decode/encode",
        args.search_max_decode_mb,
        args.search_max_encode_mb,
        args.index_max_decode_mb,
        args.index_max_encode_mb,
    );
    info!(
        "Search limits: default {}, max {}, window {}, candidates {}, response budget {} MiB",
        args.default_search_limit,
        args.max_search_limit,
        args.max_search_window,
        args.max_candidate_limit,
        args.max_search_response_mb,
    );
    info!(
        "Query shape budgets: depth {}, nodes {}, clauses {} (boolean {}), text {} B, \
         vectors {} B total, fields_to_load {}, token expansion {} (sparse {})",
        args.max_query_depth,
        args.max_query_nodes,
        args.max_query_clauses,
        args.max_boolean_clauses,
        args.max_query_text_bytes,
        args.max_total_query_vector_bytes,
        args.max_fields_to_load,
        args.max_text_query_tokens,
        args.max_sparse_token_dimensions,
    );

    let signal_registry = Arc::clone(&registry);
    let signal_shutdown = shutdown_tx.clone();
    let serve_result = Server::builder()
        // Connection management
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .http2_keepalive_interval(Some(Duration::from_secs(30)))
        .http2_keepalive_timeout(Some(Duration::from_secs(10)))
        // HTTP/2 flow control: adaptive window for large batch payloads
        .http2_adaptive_window(Some(true))
        .initial_connection_window_size(Some(4 * 1024 * 1024))
        .initial_stream_window_size(Some(2 * 1024 * 1024))
        // Concurrency limits: prevent single connection from monopolizing
        .max_concurrent_streams(Some(256))
        .concurrency_limit_per_connection(128)
        .add_service(
            SearchServiceServer::new(search_service)
                .max_decoding_message_size(service_limits.search_max_decode_bytes)
                .max_encoding_message_size(service_limits.search_max_encode_bytes)
                .accept_compressed(CompressionEncoding::Gzip)
                .accept_compressed(CompressionEncoding::Zstd)
                .send_compressed(CompressionEncoding::Zstd),
        )
        .add_service(
            IndexServiceServer::new(index_service)
                .max_decoding_message_size(service_limits.index_max_decode_bytes)
                .max_encoding_message_size(service_limits.index_max_encode_bytes)
                .accept_compressed(CompressionEncoding::Gzip)
                .accept_compressed(CompressionEncoding::Zstd)
                .send_compressed(CompressionEncoding::Zstd),
        )
        .serve_with_shutdown(addr, async move {
            shutdown_signal().await;
            // Stop core lifecycle admission immediately; BP observes this
            // cancellation while tonic drains requests.
            signal_registry.begin_shutdown();
            let _ = signal_shutdown.send(true);
        })
        .await;

    // A transport failure also takes the complete application through the
    // same ordered shutdown path.
    registry.begin_shutdown();
    let _ = shutdown_tx.send(true);

    info!("[shutdown] gRPC server drained; waiting for background work");
    let optimizer_result = match optimizer_handle {
        Some(handle) => handle.await,
        None => Ok(()),
    };
    let registry_result = registry.shutdown().await;

    serve_result?;
    optimizer_result?;
    registry_result?;
    info!("Hermes server shut down gracefully");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install ctrl+c handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            warn!("Received ctrl+c, starting graceful shutdown...");
        }
        _ = terminate => {
            warn!("Received SIGTERM, starting graceful shutdown...");
        }
    }
}

fn merge_bp_time_budget(seconds: u64) -> Option<Duration> {
    (seconds != 0).then(|| Duration::from_secs(seconds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_training_sample_cli_defaults_and_overrides() {
        let defaults = Args::try_parse_from(["hermes-server"]).unwrap();
        assert_eq!(defaults.vector_training_max_samples, 10_000_000);
        assert_eq!(defaults.vector_training_memory_mb, 4_096);

        let configured = Args::try_parse_from([
            "hermes-server",
            "--vector-training-max-samples",
            "20000000",
            "--vector-training-memory-mb",
            "3072",
        ])
        .unwrap();
        assert_eq!(configured.vector_training_max_samples, 20_000_000);
        assert_eq!(configured.vector_training_memory_mb, 3_072);
    }

    #[test]
    fn zero_merge_bp_budget_is_unbudgeted() {
        assert_eq!(merge_bp_time_budget(0), None);
        assert_eq!(merge_bp_time_budget(600), Some(Duration::from_secs(600)));
    }

    #[test]
    fn service_limit_defaults_match_historical_constants() {
        let args = Args::try_parse_from(["hermes-server"]).unwrap();
        let limits = resolve_service_limits(&args).unwrap();
        assert_eq!(limits.search_max_decode_bytes, 4 * 1024 * 1024);
        assert_eq!(limits.search_max_encode_bytes, 256 * 1024 * 1024);
        assert_eq!(limits.index_max_decode_bytes, 256 * 1024 * 1024);
        assert_eq!(limits.index_max_encode_bytes, 64 * 1024 * 1024);
        assert_eq!(limits.search, search_service::SearchLimits::default());
    }

    #[test]
    fn service_limit_flags_override_defaults() {
        let args = Args::try_parse_from([
            "hermes-server",
            "--search-max-encode-mb",
            "512",
            "--max-search-response-mb",
            "384",
            "--max-search-limit",
            "20000",
            "--max-search-window",
            "80000",
            "--max-candidate-limit",
            "80000",
        ])
        .unwrap();
        let limits = resolve_service_limits(&args).unwrap();
        assert_eq!(limits.search_max_encode_bytes, 512 * 1024 * 1024);
        assert_eq!(limits.search.max_search_response_bytes, 384 * 1024 * 1024);
        assert_eq!(limits.search.max_search_limit, 20_000);
        assert_eq!(limits.search.max_search_window, 80_000);
        assert_eq!(limits.search.max_candidate_limit, 80_000);
    }

    #[test]
    fn query_shape_flags_override_defaults_and_reject_inconsistency() {
        let args = Args::try_parse_from([
            "hermes-server",
            "--max-query-depth",
            "8",
            "--max-text-query-tokens",
            "64",
            "--max-sparse-token-dimensions",
            "1024",
        ])
        .unwrap();
        let shape = resolve_service_limits(&args).unwrap().search.shape;
        assert_eq!(shape.max_query_depth, 8);
        assert_eq!(shape.max_text_query_tokens, 64);
        assert_eq!(shape.max_sparse_token_dimensions, 1024);

        // Zero budgets are refused.
        let zero = Args::try_parse_from(["hermes-server", "--max-query-nodes", "0"]).unwrap();
        assert!(resolve_service_limits(&zero).is_err());

        // A per-boolean cap above the aggregate clause budget is unreachable.
        let unreachable = Args::try_parse_from([
            "hermes-server",
            "--max-boolean-clauses",
            "1024",
            "--max-query-clauses",
            "512",
        ])
        .unwrap();
        assert!(resolve_service_limits(&unreachable).is_err());
    }

    #[test]
    fn inconsistent_service_limits_fail_at_startup() {
        // Zero message caps are refused.
        let zero = Args::try_parse_from(["hermes-server", "--search-max-decode-mb", "0"]).unwrap();
        assert!(resolve_service_limits(&zero).is_err());

        // Hydration budget above the transport encode cap is refused.
        let over_encode = Args::try_parse_from([
            "hermes-server",
            "--search-max-encode-mb",
            "128",
            "--max-search-response-mb",
            "192",
        ])
        .unwrap();
        assert!(resolve_service_limits(&over_encode).is_err());

        // Default limit above the maximum is refused.
        let bad_default = Args::try_parse_from([
            "hermes-server",
            "--default-search-limit",
            "100",
            "--max-search-limit",
            "50",
        ])
        .unwrap();
        assert!(resolve_service_limits(&bad_default).is_err());

        // Window below the maximum limit is refused.
        let bad_window =
            Args::try_parse_from(["hermes-server", "--max-search-window", "5000"]).unwrap();
        assert!(resolve_service_limits(&bad_window).is_err());

        // Candidate cap below the window is refused.
        let bad_candidates =
            Args::try_parse_from(["hermes-server", "--max-candidate-limit", "10000"]).unwrap();
        assert!(resolve_service_limits(&bad_candidates).is_err());
    }
}
