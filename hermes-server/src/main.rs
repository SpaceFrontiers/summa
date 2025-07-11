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

    /// Validate all indexes on startup, remove corrupt segments
    #[arg(long)]
    doctor: bool,

    /// Number of background optimizer threads for BP reordering (0 = disabled)
    #[arg(long, default_value = "0")]
    optimizer_threads: usize,

    /// Interval in seconds between optimizer scans for unreordered segments
    #[arg(long, default_value = "60")]
    optimizer_scan_interval_secs: u64,

    /// Segments with at least this many docs get budgeted (partial) BP passes
    #[arg(long, default_value = "5000000")]
    optimizer_large_segment_docs: u32,

    /// Wall-clock budget in seconds per BP pass on large segments
    #[arg(long, default_value = "600")]
    optimizer_time_budget_secs: u64,

    /// Depth cap for large-segment BP: stop at partitions of this many docs
    /// (4096 = superblock granularity, keeps most of the pruning win)
    #[arg(long, default_value = "4096")]
    optimizer_partial_min_partition_docs: usize,

    /// Cooldown in seconds between deepening passes on budget-truncated segments
    #[arg(long, default_value = "600")]
    optimizer_unconverged_cooldown_secs: u64,

    /// Wall-clock budget in seconds for merge-time BP reorder (0 = unbudgeted).
    /// A truncated pass still produces a valid, better-ordered segment; it is
    /// marked unconverged and the background optimizer deepens it later.
    #[arg(long, default_value = "600")]
    merge_bp_budget_secs: u64,

    /// Memory budget (MB) for the BP forward index during reorder passes
    /// (merge-time and background). A cap, not an allocation — usage is
    /// proportional to the segment being reordered. Over-budget passes drop
    /// highest-df dims from BP's input with a loud warning; raise to 16384
    /// on hosts with headroom.
    #[arg(long, default_value = "8192")]
    bp_memory_budget_mb: usize,

    /// Address for the Prometheus /metrics HTTP endpoint.
    /// Set to "off" to disable the exporter.
    #[arg(long, default_value = "0.0.0.0:9184")]
    metrics_addr: String,
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

    let addr: SocketAddr = args.addr.parse()?;

    let num_indexing_threads = args
        .indexing_threads
        .unwrap_or_else(|| (num_cpus::get() / 4).max(1));

    let config = IndexConfig {
        max_indexing_memory_bytes: args.max_indexing_memory_mb * 1024 * 1024,
        num_indexing_threads,
        reload_interval_ms: args.reload_interval_ms,
        merge_policy: Box::new(hermes_core::merge::TieredMergePolicy::large_scale()),
        merge_bp_time_budget: (args.merge_bp_budget_secs > 0)
            .then(|| Duration::from_secs(args.merge_bp_budget_secs)),
        bp_memory_budget_bytes: args.bp_memory_budget_mb * 1024 * 1024,
        ..Default::default()
    };

    let registry = Arc::new(registry::IndexRegistry::new(args.data_dir.clone(), config));

    // Clean up index directories from incomplete deletes (e.g. server crashed mid-delete)
    registry.cleanup_incomplete_deletes();

    if args.doctor {
        registry.doctor_all_indexes().await;
    }

    let search_service = search_service::SearchServiceImpl {
        registry: Arc::clone(&registry),
    };

    let index_service = index_service::IndexServiceImpl {
        registry: Arc::clone(&registry),
    };

    // Spawn background optimizer
    let _optimizer_handle = optimizer::spawn_optimizer(
        Arc::clone(&registry),
        optimizer::OptimizerConfig {
            threads: args.optimizer_threads,
            scan_interval: Duration::from_secs(args.optimizer_scan_interval_secs),
            large_segment_docs: args.optimizer_large_segment_docs,
            time_budget: Duration::from_secs(args.optimizer_time_budget_secs),
            partial_min_partition_docs: args.optimizer_partial_min_partition_docs,
            unconverged_cooldown: Duration::from_secs(args.optimizer_unconverged_cooldown_secs),
        },
    );

    info!("Hermes server v{}", env!("CARGO_PKG_VERSION"));
    info!("Starting Hermes server on {}", addr);
    info!("Data directory: {:?}", args.data_dir);
    info!("Max indexing memory: {} MB", args.max_indexing_memory_mb);
    info!("Indexing threads: {}", num_indexing_threads);
    info!("Worker threads: {}", worker_threads);
    info!("Reload interval: {} ms", args.reload_interval_ms);
    if args.optimizer_threads > 0 {
        info!(
            "Optimizer: {} threads, {}s scan interval",
            args.optimizer_threads, args.optimizer_scan_interval_secs,
        );
    }

    // Separate message size limits for search vs index services
    const SEARCH_MAX_DECODE: usize = 4 * 1024 * 1024; // 4 MB (queries are small)
    const SEARCH_MAX_ENCODE: usize = 64 * 1024 * 1024; // 64 MB (large result sets)
    const INDEX_MAX_DECODE: usize = 256 * 1024 * 1024; // 256 MB (batch indexing)
    const INDEX_MAX_ENCODE: usize = 64 * 1024 * 1024; // 64 MB (responses are medium)

    Server::builder()
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
                .max_decoding_message_size(SEARCH_MAX_DECODE)
                .max_encoding_message_size(SEARCH_MAX_ENCODE)
                .accept_compressed(CompressionEncoding::Gzip)
                .accept_compressed(CompressionEncoding::Zstd)
                .send_compressed(CompressionEncoding::Zstd),
        )
        .add_service(
            IndexServiceServer::new(index_service)
                .max_decoding_message_size(INDEX_MAX_DECODE)
                .max_encoding_message_size(INDEX_MAX_ENCODE)
                .accept_compressed(CompressionEncoding::Gzip)
                .accept_compressed(CompressionEncoding::Zstd)
                .send_compressed(CompressionEncoding::Zstd),
        )
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;

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
