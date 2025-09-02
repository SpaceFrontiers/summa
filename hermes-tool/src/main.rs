//! Hermes Tool - CLI for index management and data processing
//!
//! # Overview
//!
//! This package provides command-line tools for creating, managing, and
//! preprocessing data for Hermes search indexes. See the package README for
//! complete lifecycle examples and current vector-tool limitations.
//!
//! # Index Commands
//!
//! - `create` - Create a new index from a schema file (JSON or SDL)
//! - `init` - Create a new index from an inline SDL schema string
//! - `index` - Index documents from a JSONL file or stdin
//! - `commit` - Commit any pending changes to the index
//! - `merge` - Force merge all segments into one
//! - `reorder` - Reorder BMP records for more effective pruning
//! - `info` - Display index information
//! - `diagnose` - Index health: ANN leaf skew/fragmentation, per-field disk
//!   usage, degenerate-vector scans, probe cost, page-cache residency
//! - `search` - Run a query against a local index
//! - `heatmap` - Visualize a BMP grid in the terminal
//! - `warmup` - Warm up slice cache and save to file
//!
//! # Data Processing Commands
//!
//! - `simhash` - Calculate SimHash for a text field and add it to each JSON object
//! - `sort` - Sort JSON objects by a specified field
//! - `term-stats` - Compute term statistics for WAND optimization
//! - `train-centroids` - Train IVF coarse centroids from JSONL vectors
//!
//! # Examples
//!
//! ## Create an index from SDL schema file
//! ```bash
//! hermes-tool create -i ./my_index -s schema.sdl
//! ```
//!
//! ## Index documents from stdin
//! ```bash
//! zstdcat dump.zst | hermes-tool index -i ./my_index --stdin
//! ```
//!
//! ## Calculate SimHash and sort
//! ```bash
//! zstdcat dump.zst | hermes-tool simhash -f title -o hash | hermes-tool sort -f hash
//! ```

mod data_processing;
mod diagnose;
mod index_ops;
mod vector_ops;

// Use jemalloc for better memory management (returns memory to OS)
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

/// Release unused memory back to the OS
#[cfg(not(target_env = "msvc"))]
fn release_memory_to_os() {
    use tikv_jemalloc_ctl::{epoch, stats};
    // Advance epoch to get fresh stats and trigger cleanup
    let _ = epoch::advance();
    // Request jemalloc to release unused pages back to OS
    // SAFETY: These are standard jemalloc control operations
    unsafe {
        if let Ok(purge) = tikv_jemalloc_ctl::raw::read::<bool>(b"opt.background_thread\0")
            && !purge
        {
            // If background threads are disabled, manually purge all arenas
            let _ = tikv_jemalloc_ctl::raw::write(b"arena.0.purge\0", ());
        }
    }
    // Log current memory stats
    if let (Ok(allocated), Ok(resident)) = (stats::allocated::read(), stats::resident::read()) {
        tracing::debug!(
            "Memory: allocated={}, resident={}",
            hermes_core::format_bytes(allocated as u64),
            hermes_core::format_bytes(resident as u64)
        );
    }
}

#[cfg(target_env = "msvc")]
fn release_memory_to_os() {
    // No-op on MSVC
}

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "hermes-tool")]
#[command(version, about = "CLI for Hermes index management and data processing")]
#[command(after_help = "Use 'hermes-tool <command> --help' for more information.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Index optimization mode for balancing compression ratio vs query speed
#[derive(Debug, Clone, Copy, ValueEnum)]
enum OptimizationMode {
    /// Rounded posting codec, term dictionary zstd level 9
    Adaptive,
    /// Patched-exception posting codec (pfor), term dictionary zstd level 22
    Size,
    /// Rounded posting codec, term dictionary zstd level 1
    Performance,
}

/// Posting block codec (see docs/posting-codecs.md)
#[derive(Debug, Clone, Copy, ValueEnum)]
enum PostingCodecArg {
    /// Widths rounded to 0/8/16/32 bits: fastest decode, largest
    Rounded,
    /// Exact bit widths (BP128): ~1.8x smaller, ~10% slower decode
    Packed,
    /// Exact widths with patched exceptions (OptP4D): smallest, ~30% slower decode
    Pfor,
}

impl PostingCodecArg {
    fn to_posting_codec(self) -> hermes_core::structures::PostingCodec {
        match self {
            Self::Rounded => hermes_core::structures::PostingCodec::Rounded,
            Self::Packed => hermes_core::structures::PostingCodec::Packed,
            Self::Pfor => hermes_core::structures::PostingCodec::Pfor,
        }
    }
}

impl OptimizationMode {
    fn to_index_optimization(self) -> hermes_core::structures::IndexOptimization {
        match self {
            Self::Adaptive => hermes_core::structures::IndexOptimization::Adaptive,
            Self::Size => hermes_core::structures::IndexOptimization::SizeOptimized,
            Self::Performance => hermes_core::structures::IndexOptimization::PerformanceOptimized,
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    // === Index Management Commands ===
    /// Create a new index from a schema file (JSON or SDL format)
    Create {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,

        /// Path to schema file (JSON or .sdl format)
        #[arg(short, long)]
        schema: PathBuf,
    },

    /// Initialize a new index from an SDL schema string
    Init {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,

        /// SDL schema definition inline
        #[arg(short, long)]
        sdl: String,
    },

    /// Index documents from a JSONL file or stdin
    Index {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,

        /// Path to JSONL documents file (omit if using --stdin)
        #[arg(short, long, required_unless_present = "stdin")]
        documents: Option<PathBuf>,

        /// Read documents from stdin instead of a file
        #[arg(long, default_value = "false")]
        stdin: bool,

        /// Log progress every N documents (0 to disable)
        #[arg(short, long, default_value = "100000")]
        progress: usize,

        /// Max indexing memory in MB before auto-flush (default: 3072)
        #[arg(short = 'm', long, default_value = "3072")]
        max_indexing_memory_mb: usize,

        /// Number of parallel segment builders (default: 1)
        #[arg(short = 'j', long)]
        indexing_threads: Option<usize>,

        /// Shared document-store compression pool width (default: CPUs / 4)
        #[arg(short = 'c', long)]
        compression_threads: Option<usize>,

        /// Index optimization mode
        /// - adaptive: rounded posting codec, term dictionary zstd 9
        /// - size: pfor posting codec, term dictionary zstd 22
        /// - performance: rounded posting codec, term dictionary zstd 1
        #[arg(short = 'O', long, default_value = "adaptive")]
        optimization: OptimizationMode,

        /// Posting block codec override (default: derived from --optimization)
        #[arg(long)]
        posting_codec: Option<PostingCodecArg>,
    },

    /// Commit pending changes
    Commit {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,
    },

    /// Force merge all segments
    Merge {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,
    },

    /// Reorder BMP blocks via Recursive Graph Bisection (BP) for better pruning
    Reorder {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,
    },

    /// Show index info
    Info {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,
    },

    /// Diagnose index health: ANN leaf skew/fragmentation, per-field disk
    /// usage, degenerate-vector scans, probe cost, page-cache residency
    Diagnose {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,

        /// Emit the report as JSON
        #[arg(long)]
        json: bool,

        /// Scan N evenly spaced flat vectors per field per segment for
        /// all-zero / non-finite content (reads payload bytes)
        #[arg(long, value_name = "N")]
        sample: Option<usize>,

        /// Estimate per-query bytes and extents touched at this nprobe
        #[arg(long, value_name = "NPROBE")]
        probe_cost: Option<usize>,

        /// Report page-cache residency of every segment file (mincore)
        #[arg(long)]
        residency: bool,

        /// Scan the whole term dictionary: doc-frequency distribution,
        /// inline-term ratio, and the top-N terms
        #[arg(long, value_name = "N")]
        terms: Option<usize>,

        /// Scan sparse BMP postings: per-dimension distribution and impact
        /// saturation (reads payload bytes)
        #[arg(long)]
        sparse_stats: bool,
    },

    /// Visualize BMP grid as a terminal heatmap
    Heatmap {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,

        /// Sparse field name (auto-detects first BMP field if omitted)
        #[arg(short, long)]
        field: Option<String>,

        /// Output width in columns (default: terminal width)
        #[arg(short = 'W', long)]
        width: Option<usize>,

        /// Output height in rows (default: terminal height - 6)
        #[arg(short = 'H', long)]
        height: Option<usize>,

        /// Segment index (0-based, default: 0 = first/largest segment)
        #[arg(short, long, default_value = "0")]
        segment: usize,
    },

    /// Search an index with a query string
    Search {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,

        /// Query string (e.g. "rust", "title:rust", "rust AND search")
        #[arg(
            short,
            long,
            required_unless_present = "queries_file",
            conflicts_with = "queries_file"
        )]
        query: Option<String>,

        /// File with one query per line; the index is opened once and one
        /// JSON result line is emitted per query
        #[arg(long)]
        queries_file: Option<PathBuf>,

        /// Concurrent queries for --queries-file (default: CPU count)
        #[arg(short = 'j', long)]
        concurrency: Option<usize>,

        /// Search pool threads (default: CPU count / 4)
        #[arg(short = 't', long)]
        search_threads: Option<usize>,

        /// Number of results to return
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Offset for pagination (single --query only)
        #[arg(short, long, default_value = "0")]
        offset: usize,
    },

    /// Warm up slice cache and save to file
    Warmup {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,

        /// Maximum cache size in bytes (default: 64MB)
        #[arg(short = 's', long, default_value = "67108864")]
        cache_size: usize,
    },

    // === Data Processing Commands ===
    /// Calculate SimHash for a text field and add to JSON
    Simhash {
        /// Field name to calculate SimHash from
        #[arg(short, long)]
        field: String,

        /// Output field name for the hash (default: {field}_simhash)
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Sort JSON objects by a field value using external merge sort
    Sort {
        /// Field name to sort by
        #[arg(short, long)]
        field: String,

        /// Sort in descending order
        #[arg(short = 'r', long, default_value = "false")]
        reverse: bool,

        /// Treat field as numeric (default: string comparison)
        #[arg(short = 'N', long, default_value = "false")]
        numeric: bool,

        /// Number of documents per chunk for external sort (default: 100000)
        #[arg(short = 'c', long, default_value = "100000")]
        chunk_size: usize,

        /// Temporary directory for chunk files (default: system temp)
        #[arg(short = 't', long)]
        temp_dir: Option<PathBuf>,
    },

    /// Compute term statistics for WAND optimization
    #[command(name = "term-stats")]
    TermStats {
        /// Field name(s) to analyze (can be specified multiple times)
        #[arg(short, long, required = true)]
        field: Vec<String>,

        /// Output format: json (default) or binary
        #[arg(short = 'F', long, default_value = "json")]
        format: String,

        /// Minimum document frequency to include a term (default: 1)
        #[arg(short = 'm', long, default_value = "1")]
        min_df: u32,

        /// BM25 k1 parameter (default: 1.2)
        #[arg(long, default_value = "1.2")]
        bm25_k1: f32,

        /// BM25 b parameter (default: 0.75)
        #[arg(long, default_value = "0.75")]
        bm25_b: f32,
    },

    // === Vector Index Commands ===
    /// Train a global IVF coarse codebook from sample vectors
    #[command(name = "train-centroids")]
    TrainCentroids {
        /// Path to input file with vectors (JSONL with field containing float arrays)
        #[arg(short, long)]
        input: PathBuf,

        /// Field name containing the vector
        #[arg(short, long)]
        field: String,

        /// Output path for centroids file
        #[arg(short, long)]
        output: PathBuf,

        /// Number of clusters (default: sqrt of sample size, max 65536)
        #[arg(short = 'k', long)]
        clusters: Option<usize>,

        /// Maximum number of k-means iterations (default: 20)
        #[arg(short = 'n', long, default_value = "20")]
        max_iters: usize,

        /// Maximum number of vectors to sample (default: all)
        #[arg(short = 's', long)]
        sample_size: Option<usize>,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Retrain centroids from an existing index and rebuild vector indexes
    #[command(name = "retrain-centroids")]
    RetrainCentroids {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,

        /// Dense vector field name to retrain
        #[arg(short, long)]
        field: String,

        /// Number of clusters (default: sqrt of vector count, max 65536)
        #[arg(short = 'k', long)]
        clusters: Option<usize>,

        /// Maximum number of k-means iterations (default: 20)
        #[arg(short = 'n', long, default_value = "20")]
        max_iters: usize,

        /// Sample size for training (default: min(1M, all vectors))
        #[arg(short = 's', long)]
        sample_size: Option<usize>,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("hermes_tool=info".parse()?),
        )
        // Logs go to stderr so machine-readable stdout (e.g. search
        // --queries-file JSONL) stays clean.
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        // Index management commands
        Commands::Create { index, schema } => {
            index_ops::create_index(index, schema).await?;
        }
        Commands::Init { index, sdl } => {
            index_ops::init_index_from_sdl(index, sdl).await?;
        }
        Commands::Index {
            index,
            documents,
            stdin,
            progress,
            max_indexing_memory_mb,
            indexing_threads,
            compression_threads,
            optimization,
            posting_codec,
        } => {
            index_ops::index_documents(
                index,
                documents,
                stdin,
                progress,
                max_indexing_memory_mb,
                indexing_threads,
                compression_threads,
                optimization.to_index_optimization(),
                posting_codec.map(PostingCodecArg::to_posting_codec),
            )
            .await?;
        }
        Commands::Commit { index } => {
            index_ops::commit_index(index).await?;
        }
        Commands::Merge { index } => {
            index_ops::merge_index(index).await?;
        }
        Commands::Reorder { index } => {
            index_ops::reorder_index(index).await?;
        }
        Commands::Info { index } => {
            index_ops::show_info(index).await?;
        }
        Commands::Diagnose {
            index,
            json,
            sample,
            probe_cost,
            residency,
            terms,
            sparse_stats,
        } => {
            diagnose::diagnose(diagnose::DiagnoseOptions {
                index,
                json,
                sample,
                probe_cost,
                residency,
                terms,
                sparse_stats,
            })
            .await?;
        }
        Commands::Heatmap {
            index,
            field,
            width,
            height,
            segment,
        } => {
            index_ops::heatmap_bmp_grid(index, field, width, height, segment).await?;
        }
        Commands::Search {
            index,
            query,
            queries_file,
            concurrency,
            search_threads,
            limit,
            offset,
        } => {
            if let Some(queries_file) = queries_file {
                anyhow::ensure!(offset == 0, "--offset is not supported with --queries-file");
                let concurrency = concurrency.unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(8)
                });
                index_ops::search_batch(index, queries_file, limit, concurrency, search_threads)
                    .await?;
            } else {
                anyhow::ensure!(
                    concurrency.is_none(),
                    "--concurrency requires --queries-file"
                );
                let query = query.expect("clap enforces --query when --queries-file is absent");
                index_ops::search_index(index, &query, limit, offset, search_threads).await?;
            }
        }
        Commands::Warmup { index, cache_size } => {
            index_ops::warmup_cache(index, cache_size).await?;
        }

        // Data processing commands
        Commands::Simhash { field, output } => {
            let output_field = output.unwrap_or_else(|| format!("{}_simhash", field));
            data_processing::run_simhash(&field, &output_field)
                .context("Failed to process simhash")?;
        }
        Commands::Sort {
            field,
            reverse,
            numeric,
            chunk_size,
            temp_dir,
        } => {
            data_processing::run_sort(&field, reverse, numeric, chunk_size, temp_dir)
                .context("Failed to sort documents")?;
        }
        Commands::TermStats {
            field,
            format,
            min_df,
            bm25_k1,
            bm25_b,
        } => {
            data_processing::run_term_stats(&field, &format, min_df, bm25_k1, bm25_b)
                .context("Failed to compute term statistics")?;
        }

        // Vector index commands
        Commands::TrainCentroids {
            input,
            field,
            output,
            clusters,
            max_iters,
            sample_size,
            seed,
        } => {
            vector_ops::train_centroids(
                input,
                field,
                output,
                clusters,
                max_iters,
                sample_size,
                seed,
            )
            .context("Failed to train centroids")?;
        }
        Commands::RetrainCentroids {
            index,
            field,
            clusters,
            max_iters,
            sample_size,
            seed,
        } => {
            vector_ops::retrain_centroids(index, field, clusters, max_iters, sample_size, seed)
                .await
                .context("Failed to retrain centroids")?;
        }
    }

    Ok(())
}
