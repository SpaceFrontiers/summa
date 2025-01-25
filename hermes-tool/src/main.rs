//! Hermes Tool - CLI for index management and data processing
//!
//! # Overview
//!
//! This package provides command-line tools for creating, managing, and
//! preprocessing data for Hermes search indexes.
//!
//! # Index Commands
//!
//! - `create` - Create a new index from a schema file (JSON or SDL)
//! - `init` - Create a new index from an inline SDL schema string
//! - `index` - Index documents from a JSONL file or stdin
//! - `commit` - Commit any pending changes to the index
//! - `merge` - Force merge all segments into one
//! - `info` - Display index information
//! - `warmup` - Warm up slice cache and save to file
//!
//! # Data Processing Commands
//!
//! - `simhash` - Calculate SimHash for a text field and add it to each JSON object
//! - `sort` - Sort JSON objects by a specified field
//! - `term-stats` - Compute term statistics for WAND optimization
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
            "Memory: allocated={:.1} MB, resident={:.1} MB",
            allocated as f64 / (1024.0 * 1024.0),
            resident as f64 / (1024.0 * 1024.0)
        );
    }
}

#[cfg(target_env = "msvc")]
fn release_memory_to_os() {
    // No-op on MSVC
}

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Cursor, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::info;

use hermes_core::{
    Document, FsDirectory, IndexConfig, IndexWriter, SLICE_CACHE_FILENAME, SliceCachingDirectory,
    parse_schema,
};

#[derive(Parser)]
#[command(name = "hermes-tool")]
#[command(version, about = "CLI for Hermes index management and data processing")]
#[command(after_help = "Use 'hermes-tool <command> --help' for more information.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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

        /// Max indexing memory in MB before auto-flush (default: 256)
        #[arg(short = 'm', long, default_value = "3072")]
        max_indexing_memory_mb: usize,

        /// Number of parallel segment builders (default: 1)
        #[arg(short = 'j', long)]
        indexing_threads: Option<usize>,

        /// Number of threads for parallel block compression (default: number of CPUs)
        #[arg(short = 'c', long)]
        compression_threads: Option<usize>,

        /// Index optimization mode: adaptive (default), size, performance
        /// - adaptive: balanced compression/speed (zstd level 7)
        /// - size: best compression ratio (OptP4D + zstd level 22)
        /// - performance: fastest queries (Roaring + zstd level 3)
        #[arg(short = 'O', long, default_value = "adaptive")]
        optimization: String,
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

    /// Show index info
    Info {
        /// Path to the index directory
        #[arg(short, long)]
        index: PathBuf,
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
    /// Train coarse centroids for IVF-RaBitQ from sample vectors
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

// ============================================================================
// Index Management Functions
// ============================================================================

async fn create_index(index_path: PathBuf, schema_path: PathBuf) -> Result<()> {
    let schema_content = fs::read_to_string(&schema_path)
        .with_context(|| format!("Failed to read schema file: {:?}", schema_path))?;

    let schema = parse_schema(&schema_content)
        .map_err(|e| anyhow::anyhow!("Failed to parse schema: {}", e))?;

    info!("Parsed schema with {} fields", schema.fields().count());

    std::fs::create_dir_all(&index_path)
        .with_context(|| format!("Failed to create index directory: {:?}", index_path))?;

    let dir = FsDirectory::new(&index_path);
    let config = IndexConfig::default();

    let _writer = IndexWriter::create(dir, schema, config).await?;

    info!("Created index at {:?}", index_path);

    Ok(())
}

async fn init_index_from_sdl(index_path: PathBuf, sdl: String) -> Result<()> {
    let schema =
        parse_schema(&sdl).map_err(|e| anyhow::anyhow!("Failed to parse schema: {}", e))?;

    std::fs::create_dir_all(&index_path)
        .with_context(|| format!("Failed to create index directory: {:?}", index_path))?;

    let dir = FsDirectory::new(&index_path);
    let config = IndexConfig::default();

    let _writer = IndexWriter::create(dir, schema.clone(), config).await?;

    info!("Created index at {:?}", index_path);
    info!("Schema has {} fields", schema.fields().count());

    Ok(())
}

async fn index_from_reader<R: BufRead>(
    writer: &IndexWriter<FsDirectory>,
    reader: R,
    progress_interval: usize,
) -> Result<usize> {
    let schema = writer.schema().clone();
    let mut count = 0usize;
    let mut errors = 0usize;
    let start_time = std::time::Instant::now();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value = match sonic_rs::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                if errors < 10 {
                    tracing::warn!("Failed to parse JSON at line {}: {}", count + 1, e);
                }
                errors += 1;
                continue;
            }
        };

        let doc = match Document::from_json(&json, &schema) {
            Some(d) => d,
            None => {
                if errors < 10 {
                    tracing::warn!("Failed to parse document at line {}", count + 1);
                }
                errors += 1;
                continue;
            }
        };

        writer.add_document(doc)?;
        count += 1;

        if progress_interval > 0 && count.is_multiple_of(progress_interval) {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = count as f64 / elapsed;
            info!(
                "Progress: {} documents indexed ({:.0} docs/sec)",
                count, rate
            );
        }
    }

    writer.commit().await?;

    // Wait for any in-flight background merges to complete before returning
    // Otherwise they will be cancelled when the IndexWriter/runtime is dropped
    if writer.pending_merge_count() > 0 {
        info!(
            "Waiting for {} background merges to complete...",
            writer.pending_merge_count()
        );
        writer.wait_for_merges().await;
    }

    release_memory_to_os();

    let elapsed = start_time.elapsed();
    let rate = count as f64 / elapsed.as_secs_f64();

    if errors > 0 {
        tracing::warn!("Skipped {} documents due to parse errors", errors);
    }

    info!(
        "Indexed {} documents in {:.2}s ({:.0} docs/sec)",
        count,
        elapsed.as_secs_f64(),
        rate
    );

    Ok(count)
}

#[allow(clippy::too_many_arguments)]
async fn index_documents(
    index_path: PathBuf,
    documents_path: Option<PathBuf>,
    use_stdin: bool,
    progress_interval: usize,
    max_indexing_memory_mb: usize,
    indexing_threads: Option<usize>,
    compression_threads: Option<usize>,
    optimization: String,
) -> Result<()> {
    use hermes_core::structures::IndexOptimization;

    let optimization_mode = IndexOptimization::parse(&optimization).ok_or_else(|| {
        anyhow::anyhow!(
            "Invalid optimization mode '{}'. Valid options: adaptive, size, performance",
            optimization
        )
    })?;

    let dir = FsDirectory::new(&index_path);
    let default_config = IndexConfig::default();
    let config = IndexConfig {
        max_indexing_memory_bytes: max_indexing_memory_mb * 1024 * 1024,
        num_indexing_threads: indexing_threads.unwrap_or(default_config.num_indexing_threads),
        num_compression_threads: compression_threads
            .unwrap_or(default_config.num_compression_threads),
        optimization: optimization_mode,
        ..default_config
    };
    let writer = IndexWriter::open(dir, config.clone()).await?;

    info!("Opened index at {:?}", index_path);
    info!(
        "Schema fields: {:?}",
        writer
            .schema()
            .fields()
            .map(|(_, e)| &e.name)
            .collect::<Vec<_>>()
    );
    info!(
        "Indexing threads: {}, Compression threads: {}, Optimization: {:?} (zstd level {})",
        config.num_indexing_threads,
        config.num_compression_threads,
        optimization_mode,
        optimization_mode.zstd_level()
    );

    let count = if use_stdin {
        info!("Reading documents from stdin...");
        let stdin = io::stdin();
        let reader = stdin.lock();
        index_from_reader(&writer, reader, progress_interval).await?
    } else if let Some(path) = documents_path {
        info!("Reading documents from {:?}", path);
        let file = File::open(&path)
            .with_context(|| format!("Failed to open documents file: {:?}", path))?;
        let reader = BufReader::new(file);
        index_from_reader(&writer, reader, progress_interval).await?
    } else {
        anyhow::bail!("Either --documents or --stdin must be specified");
    };

    info!("Successfully indexed {} documents", count);
    Ok(())
}

async fn commit_index(index_path: PathBuf) -> Result<()> {
    let dir = FsDirectory::new(&index_path);
    let config = IndexConfig::default();
    let writer = IndexWriter::open(dir, config).await?;

    writer.commit().await?;
    info!("Committed index at {:?}", index_path);

    Ok(())
}

async fn merge_index(index_path: PathBuf) -> Result<()> {
    let dir = FsDirectory::new(&index_path);
    let config = IndexConfig::default();
    let writer = IndexWriter::open(dir, config).await?;

    info!("Starting force merge...");
    writer.force_merge().await?;
    info!("Force merge completed");

    Ok(())
}

async fn show_info(index_path: PathBuf) -> Result<()> {
    let dir = FsDirectory::new(&index_path);
    let config = IndexConfig::default();
    let index = hermes_core::Index::open(dir, config).await?;

    println!("Index: {:?}", index_path);
    println!("Documents: {}", index.num_docs());
    println!("Segments: {}", index.segment_readers().len());
    println!();
    println!("Schema:");
    for (_field, entry) in index.schema().fields() {
        println!(
            "  {} ({:?}) - indexed: {}, stored: {}",
            entry.name, entry.field_type, entry.indexed, entry.stored
        );
    }

    Ok(())
}

async fn warmup_cache(index_path: PathBuf, cache_size: usize) -> Result<()> {
    info!(
        "Opening index with slice caching (max {} bytes)...",
        cache_size
    );

    let dir = FsDirectory::new(&index_path);
    let caching_dir = SliceCachingDirectory::new(dir, cache_size);
    let config = IndexConfig::default();

    let index = hermes_core::Index::open(caching_dir, config).await?;

    info!(
        "Index opened: {} documents, {} segments",
        index.num_docs(),
        index.segment_readers().len()
    );

    let stats = index.slice_cache_stats();
    info!(
        "Cache populated: {} bytes in {} slices across {} files",
        stats.total_bytes, stats.total_slices, stats.files_cached
    );

    index.save_slice_cache().await?;

    let cache_file = index_path.join(SLICE_CACHE_FILENAME);
    let cache_file_size = std::fs::metadata(&cache_file).map(|m| m.len()).unwrap_or(0);

    info!(
        "Slice cache saved to {:?} ({} bytes)",
        cache_file, cache_file_size
    );

    Ok(())
}

// ============================================================================
// Data Processing Functions
// ============================================================================

fn simhash(text: &str) -> u32 {
    let tokens: Vec<&str> = text.split_whitespace().collect();

    if tokens.is_empty() {
        return 0;
    }

    let mut bit_counts = [0i32; 32];

    for token in tokens {
        let hash = murmur3::murmur3_32(&mut Cursor::new(token.as_bytes()), 0).unwrap_or(0);

        for (i, count) in bit_counts.iter_mut().enumerate() {
            if (hash >> i) & 1 == 0 {
                *count += 1;
            } else {
                *count -= 1;
            }
        }
    }

    let mut fingerprint = 0u32;
    for (i, &count) in bit_counts.iter().enumerate() {
        if count > 0 {
            fingerprint |= 1 << i;
        }
    }

    fingerprint
}

fn run_simhash(field: &str, output_field: &str) -> Result<()> {
    use crossbeam_channel::{Receiver, Sender, bounded};
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    const BATCH_SIZE: usize = 10_000;
    const CHANNEL_CAPACITY: usize = 4;

    let field = field.to_string();
    let output_field = output_field.to_string();

    // Channels for pipeline: reader -> processor -> writer
    let (line_tx, line_rx): (Sender<Vec<String>>, Receiver<Vec<String>>) =
        bounded(CHANNEL_CAPACITY);
    let (result_tx, result_rx): (Sender<Vec<String>>, Receiver<Vec<String>>) =
        bounded(CHANNEL_CAPACITY);

    let count = std::sync::Arc::new(AtomicUsize::new(0));
    let errors = std::sync::Arc::new(AtomicUsize::new(0));

    let count_clone = count.clone();
    let errors_clone = errors.clone();

    // Reader thread
    let reader_handle = thread::spawn(move || -> Result<()> {
        let stdin = io::stdin();
        let reader = BufReader::new(stdin.lock());
        let mut batch = Vec::with_capacity(BATCH_SIZE);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            batch.push(line);

            if batch.len() >= BATCH_SIZE {
                if line_tx.send(std::mem::take(&mut batch)).is_err() {
                    break;
                }
                batch = Vec::with_capacity(BATCH_SIZE);
            }
        }

        if !batch.is_empty() {
            let _ = line_tx.send(batch);
        }

        Ok(())
    });

    // Processor thread (uses rayon for parallel simhash computation)
    let processor_handle = thread::spawn(move || {
        while let Ok(batch) = line_rx.recv() {
            let results: Vec<String> = batch
                .into_par_iter()
                .filter_map(|line| {
                    let mut json: serde_json::Value = match sonic_rs::from_str(&line) {
                        Ok(v) => v,
                        Err(_) => {
                            errors_clone.fetch_add(1, Ordering::Relaxed);
                            return None;
                        }
                    };

                    if let Some(obj) = json.as_object_mut() {
                        let text = obj.get(&field).and_then(|v| v.as_str()).unwrap_or("");
                        let hash = simhash(text);
                        obj.insert(output_field.clone(), serde_json::Value::from(hash));
                        count_clone.fetch_add(1, Ordering::Relaxed);
                        serde_json::to_string(&json).ok()
                    } else {
                        None
                    }
                })
                .collect();

            if result_tx.send(results).is_err() {
                break;
            }
        }
    });

    // Writer thread
    let writer_handle = thread::spawn(move || -> Result<()> {
        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());

        while let Ok(batch) = result_rx.recv() {
            for line in batch {
                writeln!(writer, "{}", line)?;
            }
        }

        writer.flush()?;
        Ok(())
    });

    reader_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Reader thread panicked"))??;
    processor_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Processor thread panicked"))?;
    writer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Writer thread panicked"))??;

    let final_count = count.load(Ordering::Relaxed);
    let final_errors = errors.load(Ordering::Relaxed);

    if final_errors > 0 {
        eprintln!(
            "Processed {} documents, {} errors",
            final_count, final_errors
        );
    } else {
        eprintln!("Processed {} documents", final_count);
    }

    Ok(())
}

fn compare_by_field(
    a: &serde_json::Value,
    b: &serde_json::Value,
    field: &str,
    numeric: bool,
    reverse: bool,
) -> std::cmp::Ordering {
    let val_a = a.get(field);
    let val_b = b.get(field);

    let cmp = if numeric {
        let num_a = val_a.and_then(|v| v.as_f64()).unwrap_or(0.0);
        let num_b = val_b.and_then(|v| v.as_f64()).unwrap_or(0.0);
        num_a
            .partial_cmp(&num_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    } else {
        let str_a = val_a.and_then(|v| v.as_str()).unwrap_or("");
        let str_b = val_b.and_then(|v| v.as_str()).unwrap_or("");
        str_a.cmp(str_b)
    };

    if reverse { cmp.reverse() } else { cmp }
}

fn write_chunk(
    mut chunk: Vec<serde_json::Value>,
    field: &str,
    numeric: bool,
    reverse: bool,
    temp_dir: &std::path::Path,
    chunk_num: usize,
) -> Result<PathBuf> {
    use rayon::prelude::*;

    // Use parallel sort
    chunk.par_sort_by(|a, b| compare_by_field(a, b, field, numeric, reverse));

    let chunk_path = temp_dir.join(format!("chunk_{:06}.jsonl", chunk_num));
    let file = std::fs::File::create(&chunk_path)?;
    let mut writer = BufWriter::new(file);

    for doc in chunk.iter() {
        serde_json::to_writer(&mut writer, doc)?;
        writeln!(writer)?;
    }
    writer.flush()?;

    Ok(chunk_path)
}

fn merge_chunks(
    chunk_paths: Vec<PathBuf>,
    field: &str,
    numeric: bool,
    reverse: bool,
) -> Result<()> {
    use std::collections::BinaryHeap;

    let mut readers: Vec<std::io::Lines<BufReader<std::fs::File>>> = Vec::new();
    for path in &chunk_paths {
        let file = std::fs::File::open(path)?;
        readers.push(BufReader::new(file).lines());
    }

    struct HeapItem {
        value: serde_json::Value,
        chunk_idx: usize,
        field: String,
        numeric: bool,
        reverse: bool,
    }

    impl PartialEq for HeapItem {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == std::cmp::Ordering::Equal
        }
    }
    impl Eq for HeapItem {}

    impl PartialOrd for HeapItem {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for HeapItem {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            compare_by_field(
                &other.value,
                &self.value,
                &self.field,
                self.numeric,
                self.reverse,
            )
        }
    }

    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    for (idx, reader) in readers.iter_mut().enumerate() {
        if let Some(Ok(line)) = reader.next()
            && let Ok(value) = sonic_rs::from_str(&line)
        {
            heap.push(HeapItem {
                value,
                chunk_idx: idx,
                field: field.to_string(),
                numeric,
                reverse,
            });
        }
    }

    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    let mut output_count = 0usize;

    while let Some(item) = heap.pop() {
        serde_json::to_writer(&mut writer, &item.value)?;
        writeln!(writer)?;
        output_count += 1;

        if let Some(Ok(line)) = readers[item.chunk_idx].next()
            && let Ok(value) = sonic_rs::from_str(&line)
        {
            heap.push(HeapItem {
                value,
                chunk_idx: item.chunk_idx,
                field: field.to_string(),
                numeric,
                reverse,
            });
        }
    }

    writer.flush()?;
    eprintln!(
        "Merged {} documents from {} chunks",
        output_count,
        chunk_paths.len()
    );

    for path in chunk_paths {
        let _ = std::fs::remove_file(path);
    }

    Ok(())
}

fn run_sort(
    field: &str,
    reverse: bool,
    numeric: bool,
    chunk_size: usize,
    temp_dir: Option<PathBuf>,
) -> Result<()> {
    use crossbeam_channel::{Sender, bounded};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    let stdin = io::stdin();
    let reader = BufReader::new(stdin.lock());

    let temp_dir = temp_dir.unwrap_or_else(|| std::env::temp_dir().join("hermes-sort"));
    std::fs::create_dir_all(&temp_dir)?;

    // Channel for sending chunks to be sorted/written in parallel
    #[allow(clippy::type_complexity)]
    let (chunk_tx, chunk_rx): (
        Sender<(Vec<serde_json::Value>, usize)>,
        crossbeam_channel::Receiver<(Vec<serde_json::Value>, usize)>,
    ) = bounded(4);

    // Channel for receiving completed chunk paths
    #[allow(clippy::type_complexity)]
    let (path_tx, path_rx): (
        Sender<(usize, PathBuf)>,
        crossbeam_channel::Receiver<(usize, PathBuf)>,
    ) = bounded(100);

    let field_clone = field.to_string();
    let temp_dir_clone = temp_dir.clone();
    let errors_count = Arc::new(AtomicUsize::new(0));

    // Spawn worker threads for parallel chunk sorting/writing
    let num_workers = rayon::current_num_threads().min(4);
    let mut worker_handles = Vec::new();

    for _ in 0..num_workers {
        let rx = chunk_rx.clone();
        let tx = path_tx.clone();
        let field = field_clone.clone();
        let temp_dir = temp_dir_clone.clone();

        let handle = thread::spawn(move || {
            while let Ok((chunk, chunk_num)) = rx.recv() {
                match write_chunk(chunk, &field, numeric, reverse, &temp_dir, chunk_num) {
                    Ok(path) => {
                        let _ = tx.send((chunk_num, path));
                    }
                    Err(e) => {
                        eprintln!("Error writing chunk {}: {}", chunk_num, e);
                    }
                }
            }
        });
        worker_handles.push(handle);
    }

    // Drop extra senders so workers can terminate
    drop(chunk_rx);
    drop(path_tx);

    let mut chunk: Vec<serde_json::Value> = Vec::with_capacity(chunk_size);
    let mut chunk_num = 0usize;
    let mut total_docs = 0usize;
    let mut errors = 0usize;

    eprintln!(
        "Reading and sorting chunks (chunk_size={}, workers={})...",
        chunk_size, num_workers
    );

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        match sonic_rs::from_str(&line) {
            Ok(v) => {
                chunk.push(v);
                total_docs += 1;

                if chunk.len() >= chunk_size {
                    eprintln!("  Sending chunk {} ({} docs)", chunk_num, chunk.len());
                    if chunk_tx
                        .send((std::mem::take(&mut chunk), chunk_num))
                        .is_err()
                    {
                        break;
                    }
                    chunk_num += 1;
                    chunk = Vec::with_capacity(chunk_size);
                }
            }
            Err(e) => {
                if errors < 5 {
                    eprintln!(
                        "Warning: Failed to parse JSON at line {}: {}",
                        line_num + 1,
                        e
                    );
                }
                errors += 1;
            }
        }
    }

    // Send final chunk
    if !chunk.is_empty() {
        eprintln!("  Sending chunk {} ({} docs)", chunk_num, chunk.len());
        let _ = chunk_tx.send((chunk, chunk_num));
    }

    // Close sender to signal workers to finish
    drop(chunk_tx);

    // Wait for all workers to complete
    for handle in worker_handles {
        let _ = handle.join();
    }

    // Collect all chunk paths and sort by chunk number to maintain order
    let mut chunk_paths: Vec<(usize, PathBuf)> = path_rx.iter().collect();
    chunk_paths.sort_by_key(|(num, _)| *num);
    let chunk_paths: Vec<PathBuf> = chunk_paths.into_iter().map(|(_, path)| path).collect();

    errors += errors_count.load(Ordering::Relaxed);

    eprintln!(
        "Read {} documents into {} chunks",
        total_docs,
        chunk_paths.len()
    );

    if chunk_paths.is_empty() {
        eprintln!("No documents to sort");
        return Ok(());
    }

    if chunk_paths.len() == 1 {
        let file = std::fs::File::open(&chunk_paths[0])?;
        let reader = BufReader::new(file);
        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());

        for line in reader.lines() {
            writeln!(writer, "{}", line?)?;
        }
        writer.flush()?;
        let _ = std::fs::remove_file(&chunk_paths[0]);
        eprintln!("Sorted {} documents (single chunk)", total_docs);
    } else {
        eprintln!("Merging {} chunks...", chunk_paths.len());
        merge_chunks(chunk_paths, field, numeric, reverse)?;
    }

    if errors > 0 {
        eprintln!("Skipped {} documents due to parse errors", errors);
    }

    let _ = std::fs::remove_dir(&temp_dir);

    Ok(())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
pub struct TermStats {
    pub term: String,
    pub df: u32,
    pub total_tf: u64,
    pub max_tf: u32,
    pub idf: f32,
    pub upper_bound: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
pub struct WandStats {
    pub total_docs: u64,
    pub total_tokens: u64,
    pub avg_doc_len: f32,
    pub bm25_k1: f32,
    pub bm25_b: f32,
    pub terms: Vec<TermStats>,
}

fn tokenize_simple(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

fn compute_idf(total_docs: u64, df: u32) -> f32 {
    let n = total_docs as f32;
    let df = df as f32;
    ((n - df + 0.5) / (df + 0.5)).ln()
}

fn compute_upper_bound(max_tf: u32, idf: f32, k1: f32, b: f32) -> f32 {
    let tf = max_tf as f32;
    let min_length_norm = 1.0 - b;
    let tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * min_length_norm);
    idf * tf_norm
}

fn run_term_stats(
    fields: &[String],
    format: &str,
    min_df: u32,
    bm25_k1: f32,
    bm25_b: f32,
) -> Result<()> {
    let stdin = io::stdin();
    let reader = BufReader::new(stdin.lock());

    let mut term_stats: HashMap<(String, String), (u32, u64, u32)> = HashMap::new();
    let mut total_docs = 0u64;
    let mut total_tokens = 0u64;
    let mut errors = 0usize;

    eprintln!("Computing term statistics for fields: {:?}", fields);

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value = match sonic_rs::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                if errors < 5 {
                    eprintln!("Warning: Failed to parse JSON: {}", e);
                }
                errors += 1;
                continue;
            }
        };

        total_docs += 1;

        if let Some(obj) = json.as_object() {
            let mut doc_terms: HashMap<(String, String), u32> = HashMap::new();

            for field_name in fields {
                let text = obj.get(field_name).and_then(|v| v.as_str()).unwrap_or("");
                let tokens = tokenize_simple(text);
                total_tokens += tokens.len() as u64;

                for token in tokens {
                    let key = (field_name.clone(), token);
                    *doc_terms.entry(key).or_insert(0) += 1;
                }
            }

            for ((field, term), tf) in doc_terms {
                let entry = term_stats.entry((field, term)).or_insert((0, 0, 0));
                entry.0 += 1;
                entry.1 += tf as u64;
                entry.2 = entry.2.max(tf);
            }
        }

        if total_docs.is_multiple_of(100_000) {
            eprintln!(
                "  Processed {} documents, {} unique terms...",
                total_docs,
                term_stats.len()
            );
        }
    }

    eprintln!(
        "Processed {} documents, {} unique terms, {} total tokens",
        total_docs,
        term_stats.len(),
        total_tokens
    );

    let avg_doc_len = if total_docs > 0 {
        total_tokens as f32 / total_docs as f32
    } else {
        0.0
    };

    let mut terms: Vec<TermStats> = term_stats
        .into_iter()
        .filter(|(_, (df, _, _))| *df >= min_df)
        .map(|((field, term), (df, total_tf, max_tf))| {
            let idf = compute_idf(total_docs, df);
            let upper_bound = compute_upper_bound(max_tf, idf, bm25_k1, bm25_b);
            TermStats {
                term: format!("{}:{}", field, term),
                df,
                total_tf,
                max_tf,
                idf,
                upper_bound,
            }
        })
        .collect();

    terms.sort_by(|a, b| b.df.cmp(&a.df));

    let wand_stats = WandStats {
        total_docs,
        total_tokens,
        avg_doc_len,
        bm25_k1,
        bm25_b,
        terms,
    };

    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());

    match format {
        "json" => {
            serde_json::to_writer_pretty(&mut writer, &wand_stats)?;
            writeln!(writer)?;
        }
        "binary" => {
            let encoded = bincode::encode_to_vec(&wand_stats, bincode::config::standard())
                .map_err(|e| anyhow::anyhow!("Bincode error: {}", e))?;
            writer.write_all(&encoded)?;
        }
        _ => {
            anyhow::bail!("Unknown format: {}. Use 'json' or 'binary'", format);
        }
    }

    writer.flush()?;

    eprintln!(
        "Output {} terms (min_df={}), avg_doc_len={:.2}",
        wand_stats.terms.len(),
        min_df,
        avg_doc_len
    );

    Ok(())
}

// ============================================================================
// Vector Index Functions
// ============================================================================

/// Train coarse centroids from a JSONL file containing vectors
fn train_centroids(
    input: PathBuf,
    field: String,
    output: PathBuf,
    clusters: Option<usize>,
    max_iters: usize,
    sample_size: Option<usize>,
    seed: u64,
) -> Result<()> {
    use hermes_core::structures::CoarseCentroids;

    info!("Loading vectors from {:?}, field '{}'", input, field);

    let file = File::open(&input).context("Failed to open input file")?;
    let reader = BufReader::new(file);

    let mut vectors: Vec<Vec<f32>> = Vec::new();
    let max_samples = sample_size.unwrap_or(usize::MAX);

    for line in reader.lines() {
        if vectors.len() >= max_samples {
            break;
        }

        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value =
            serde_json::from_str(&line).context("Failed to parse JSON line")?;

        if let Some(vec_value) = json.get(&field)
            && let Some(arr) = vec_value.as_array()
        {
            let vec: Vec<f32> = arr
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            if !vec.is_empty() {
                vectors.push(vec);
            }
        }
    }

    if vectors.is_empty() {
        anyhow::bail!("No vectors found in input file");
    }

    let dim = vectors[0].len();
    info!("Loaded {} vectors of dimension {}", vectors.len(), dim);

    // Determine number of clusters
    let num_clusters = clusters.unwrap_or_else(|| {
        let sqrt = (vectors.len() as f64).sqrt() as usize;
        sqrt.clamp(16, 65536)
    });

    info!(
        "Training {} clusters with {} iterations (seed={})",
        num_clusters, max_iters, seed
    );

    let start = std::time::Instant::now();
    let coarse_config = hermes_core::structures::CoarseConfig::new(dim, num_clusters)
        .with_max_iters(max_iters)
        .with_seed(seed);
    let centroids = CoarseCentroids::train(&coarse_config, &vectors);
    let elapsed = start.elapsed();

    info!(
        "Training complete in {:.2}s, saving to {:?}",
        elapsed.as_secs_f64(),
        output
    );

    centroids
        .save(&output)
        .context("Failed to save centroids")?;

    eprintln!(
        "Trained {} clusters from {} vectors ({} dims) in {:.2}s",
        num_clusters,
        vectors.len(),
        dim,
        elapsed.as_secs_f64()
    );
    eprintln!("Saved to: {:?}", output);

    Ok(())
}

/// Retrain centroids from an existing index and rebuild vector indexes
async fn retrain_centroids(
    index_path: PathBuf,
    _field: String,
    clusters: Option<usize>,
    max_iters: usize,
    sample_size: Option<usize>,
    seed: u64,
) -> Result<()> {
    use hermes_core::structures::CoarseCentroids;

    info!("Opening index at {:?}", index_path);

    let dir = FsDirectory::new(index_path.clone());
    let config = IndexConfig::default();

    let index = hermes_core::Index::open(dir, config)
        .await
        .context("Failed to open index")?;

    // Collect vectors from all segments
    info!("Collecting vectors from index segments...");
    let vectors: Vec<Vec<f32>> = Vec::new();
    let _max_samples = sample_size.unwrap_or(1_000_000);

    // TODO: Implement vector extraction from segments
    // This requires iterating through documents and extracting dense vector fields
    // For now, we'll show a placeholder message

    if vectors.is_empty() {
        eprintln!("Note: Vector extraction from existing index not yet implemented.");
        eprintln!("Please use 'train-centroids' with a JSONL file containing vectors.");
        eprintln!();
        eprintln!("Example workflow:");
        eprintln!(
            "  1. Export vectors: hermes-tool export-vectors -i {} -f {} > vectors.jsonl",
            index_path.display(),
            _field
        );
        eprintln!(
            "  2. Train centroids: hermes-tool train-centroids -i vectors.jsonl -f {} -o centroids.bin",
            _field
        );
        eprintln!("  3. Rebuild index with new centroids");
        return Ok(());
    }

    let dim = vectors[0].len();
    let num_clusters = clusters.unwrap_or_else(|| {
        let sqrt = (vectors.len() as f64).sqrt() as usize;
        sqrt.clamp(16, 65536)
    });

    info!(
        "Training {} clusters from {} vectors",
        num_clusters,
        vectors.len()
    );

    let coarse_config = hermes_core::structures::CoarseConfig::new(dim, num_clusters)
        .with_max_iters(max_iters)
        .with_seed(seed);
    let centroids = CoarseCentroids::train(&coarse_config, &vectors);

    // Save centroids to index directory
    let centroids_path = index_path.join("coarse_centroids.bin");
    centroids.save(&centroids_path)?;

    info!("Saved centroids to {:?}", centroids_path);

    // TODO: Rebuild vector indexes with new centroids
    eprintln!(
        "Trained {} clusters, saved to {:?}",
        num_clusters, centroids_path
    );
    eprintln!("Note: Index rebuild with new centroids not yet implemented.");

    drop(index);
    Ok(())
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("hermes_tool=info".parse()?),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        // Index management commands
        Commands::Create { index, schema } => {
            create_index(index, schema).await?;
        }
        Commands::Init { index, sdl } => {
            init_index_from_sdl(index, sdl).await?;
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
        } => {
            index_documents(
                index,
                documents,
                stdin,
                progress,
                max_indexing_memory_mb,
                indexing_threads,
                compression_threads,
                optimization,
            )
            .await?;
        }
        Commands::Commit { index } => {
            commit_index(index).await?;
        }
        Commands::Merge { index } => {
            merge_index(index).await?;
        }
        Commands::Info { index } => {
            show_info(index).await?;
        }
        Commands::Warmup { index, cache_size } => {
            warmup_cache(index, cache_size).await?;
        }

        // Data processing commands
        Commands::Simhash { field, output } => {
            let output_field = output.unwrap_or_else(|| format!("{}_simhash", field));
            run_simhash(&field, &output_field).context("Failed to process simhash")?;
        }
        Commands::Sort {
            field,
            reverse,
            numeric,
            chunk_size,
            temp_dir,
        } => {
            run_sort(&field, reverse, numeric, chunk_size, temp_dir)
                .context("Failed to sort documents")?;
        }
        Commands::TermStats {
            field,
            format,
            min_df,
            bm25_k1,
            bm25_b,
        } => {
            run_term_stats(&field, &format, min_df, bm25_k1, bm25_b)
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
            train_centroids(input, field, output, clusters, max_iters, sample_size, seed)
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
            retrain_centroids(index, field, clusters, max_iters, sample_size, seed)
                .await
                .context("Failed to retrain centroids")?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simhash_similar_strings() {
        let h1 = simhash("The quick brown fox jumps over the lazy dog");
        let h2 = simhash("The quick brown fox jumps over the lazy cat");
        let h3 = simhash("Something completely different here");

        let dist_similar = (h1 ^ h2).count_ones();
        let dist_different = (h1 ^ h3).count_ones();

        assert!(
            dist_similar < dist_different,
            "Similar strings should have smaller Hamming distance: {} vs {}",
            dist_similar,
            dist_different
        );
    }

    #[test]
    fn test_simhash_identical() {
        let h1 = simhash("Hello world");
        let h2 = simhash("Hello world");
        assert_eq!(h1, h2);
    }
}
