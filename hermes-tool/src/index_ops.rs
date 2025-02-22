//! Index management operations: create, index, commit, merge, info, warmup

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{Context, Result};
use tracing::info;

use hermes_core::{Document, FsDirectory, IndexConfig, IndexWriter, parse_schema};

use crate::release_memory_to_os;

pub async fn create_index(index_path: PathBuf, schema_path: PathBuf) -> Result<()> {
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

pub async fn init_index_from_sdl(index_path: PathBuf, sdl: String) -> Result<()> {
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
    writer: &mut IndexWriter<FsDirectory>,
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
    info!("Waiting for background merges to complete...");
    writer.wait_for_merging_thread().await;

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
pub async fn index_documents(
    index_path: PathBuf,
    documents_path: Option<PathBuf>,
    use_stdin: bool,
    progress_interval: usize,
    max_indexing_memory_mb: usize,
    indexing_threads: Option<usize>,
    compression_threads: Option<usize>,
    optimization: hermes_core::structures::IndexOptimization,
) -> Result<()> {
    let optimization_mode = optimization;

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
    let mut writer = IndexWriter::open(dir, config.clone()).await?;

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
        index_from_reader(&mut writer, reader, progress_interval).await?
    } else if let Some(path) = documents_path {
        info!("Reading documents from {:?}", path);
        let file = File::open(&path)
            .with_context(|| format!("Failed to open documents file: {:?}", path))?;
        let reader = BufReader::new(file);
        index_from_reader(&mut writer, reader, progress_interval).await?
    } else {
        anyhow::bail!("Either --documents or --stdin must be specified");
    };

    info!("Successfully indexed {} documents", count);
    Ok(())
}

pub async fn commit_index(index_path: PathBuf) -> Result<()> {
    let dir = FsDirectory::new(&index_path);
    let config = IndexConfig::default();
    let mut writer = IndexWriter::open(dir, config).await?;

    writer.commit().await?;
    info!("Committed index at {:?}", index_path);

    Ok(())
}

pub async fn merge_index(index_path: PathBuf) -> Result<()> {
    let dir = FsDirectory::new(&index_path);
    let config = IndexConfig::default();
    let mut writer = IndexWriter::open(dir, config).await?;

    info!("Starting force merge...");
    writer.force_merge().await?;
    info!("Force merge completed");

    Ok(())
}

pub async fn reorder_index(index_path: PathBuf) -> Result<()> {
    let dir = FsDirectory::new(&index_path);
    let config = IndexConfig::default();
    let mut writer = IndexWriter::open(dir, config).await?;

    info!("Starting SimHash reorder...");
    writer.reorder().await?;
    info!("Reorder completed");

    Ok(())
}

pub async fn search_index(
    index_path: PathBuf,
    query_str: &str,
    limit: usize,
    offset: usize,
) -> Result<()> {
    let dir = FsDirectory::new(&index_path);
    let config = IndexConfig::default();
    let index = hermes_core::Index::open(dir, config).await?;
    let schema = index.schema().clone();

    let response = index
        .query_offset(query_str, limit, offset)
        .await
        .with_context(|| format!("Search failed for query: {}", query_str))?;

    info!(
        "Found {} results (total: {})",
        response.hits.len(),
        response.total_hits
    );

    for (i, hit) in response.hits.iter().enumerate() {
        println!(
            "--- Result {} (score: {:.4}) ---",
            offset + i + 1,
            hit.score
        );
        if let Some(doc) = index.get_document(&hit.address).await? {
            let json = doc.to_json(&schema);
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
    }

    println!("---");
    println!(
        "Showing {}-{} of {} results",
        offset + 1,
        offset + response.hits.len(),
        response.total_hits
    );

    Ok(())
}

pub async fn show_info(index_path: PathBuf) -> Result<()> {
    let dir = FsDirectory::new(&index_path);
    let config = IndexConfig::default();
    let index = hermes_core::Index::open(dir, config).await?;

    println!("Index: {:?}", index_path);
    println!("Documents: {}", index.num_docs().await?);
    println!("Segments: {}", index.segment_readers().await?.len());
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

pub async fn warmup_cache(index_path: PathBuf, cache_size: usize) -> Result<()> {
    use hermes_core::{DirectoryWriter, SLICE_CACHE_FILENAME, SliceCachingDirectory};

    info!(
        "Opening index with slice caching (max {} bytes)...",
        cache_size
    );

    let dir = FsDirectory::new(&index_path);
    let caching_dir = SliceCachingDirectory::new(dir.clone(), cache_size);
    let config = IndexConfig::default();

    let index = hermes_core::Index::open(caching_dir, config).await?;

    info!(
        "Index opened: {} documents, {} segments",
        index.num_docs().await?,
        index.segment_readers().await?.len()
    );

    let stats = index.directory().stats();
    info!(
        "Cache populated: {} bytes in {} slices across {} files",
        stats.total_bytes, stats.total_slices, stats.files_cached
    );

    // Serialize cache data
    let cache_data = index.directory().serialize();
    let cache_file = index_path.join(SLICE_CACHE_FILENAME);
    dir.write(cache_file.as_path(), &cache_data).await?;

    let cache_file_size = std::fs::metadata(&cache_file).map(|m| m.len()).unwrap_or(0);

    info!(
        "Slice cache saved to {:?} ({} bytes)",
        cache_file, cache_file_size
    );

    Ok(())
}
