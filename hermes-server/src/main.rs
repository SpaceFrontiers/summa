//! Hermes gRPC Search Server

mod converters;
mod index_service;
mod registry;
mod search_service;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tonic::{codec::CompressionEncoding, transport::Server};
use tracing::info;

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
    #[arg(long, default_value = "4096")]
    max_indexing_memory_mb: usize,

    /// Number of parallel indexing threads (defaults to CPU count)
    #[arg(long)]
    indexing_threads: Option<usize>,

    /// Reload interval in milliseconds for searcher to check for new segments
    /// Higher values reduce reload overhead during heavy indexing
    #[arg(long, default_value = "1000")]
    reload_interval_ms: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Install panic hook that logs backtrace before aborting
    // This catches panics in spawned tasks that would otherwise be silent
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let bt = std::backtrace::Backtrace::force_capture();
        eprintln!("=== PANIC ===\n{info}\n{bt}");
        default_hook(info);
    }));

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("hermes_server=info".parse()?),
        )
        .init();

    let args = Args::parse();

    // Set HuggingFace cache directory if specified
    if let Some(cache_dir) = &args.cache_dir {
        std::fs::create_dir_all(cache_dir)?;
        // SAFETY: We set this env var before any threads are spawned
        unsafe { std::env::set_var("HF_HOME", cache_dir) };
        info!("HuggingFace cache directory: {:?}", cache_dir);
    }

    // Create data directory if needed
    std::fs::create_dir_all(&args.data_dir)?;

    let addr: SocketAddr = args.addr.parse()?;

    let num_indexing_threads = args
        .indexing_threads
        .unwrap_or_else(|| (num_cpus::get() / 2).max(1));

    let config = IndexConfig {
        max_indexing_memory_bytes: args.max_indexing_memory_mb * 1024 * 1024,
        num_indexing_threads,
        reload_interval_ms: args.reload_interval_ms,
        ..Default::default()
    };

    let registry = Arc::new(registry::IndexRegistry::new(args.data_dir.clone(), config));

    let search_service = search_service::SearchServiceImpl {
        registry: Arc::clone(&registry),
    };

    let index_service = index_service::IndexServiceImpl {
        registry: Arc::clone(&registry),
    };

    info!("Hermes server v{}", env!("CARGO_PKG_VERSION"));
    info!("Starting Hermes server on {}", addr);
    info!("Data directory: {:?}", args.data_dir);
    info!("Max indexing memory: {} MB", args.max_indexing_memory_mb);
    info!("Indexing threads: {}", num_indexing_threads);
    info!("Reload interval: {} ms", args.reload_interval_ms);

    // 256 MB limit for large batch index operations
    const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

    Server::builder()
        .add_service(
            SearchServiceServer::new(search_service)
                .max_decoding_message_size(MAX_MESSAGE_SIZE)
                .max_encoding_message_size(MAX_MESSAGE_SIZE)
                .accept_compressed(CompressionEncoding::Gzip)
                .send_compressed(CompressionEncoding::Gzip),
        )
        .add_service(
            IndexServiceServer::new(index_service)
                .max_decoding_message_size(MAX_MESSAGE_SIZE)
                .max_encoding_message_size(MAX_MESSAGE_SIZE)
                .accept_compressed(CompressionEncoding::Gzip)
                .send_compressed(CompressionEncoding::Gzip),
        )
        .serve(addr)
        .await?;

    Ok(())
}
