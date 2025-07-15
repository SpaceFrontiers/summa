//! Standalone segment reorder via Recursive Graph Bisection (BP).
//!
//! Copies unchanged segment files (postings, store, fast, dense vectors)
//! and rebuilds the sparse file with reordered BMP blocks. Non-BMP sparse
//! fields (MaxScore) are identity-copied.
//!
//! Background optimization is decoupled from the merge path. Merge-time
//! reordering is optional; when enabled, both paths share the same bounded
//! CPU pool and whole-pass concurrency gate.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::{FieldType, Schema};
use crate::segment::OffsetWriter;
use crate::segment::format::{SparseFieldToc, write_sparse_toc_and_footer};
use crate::segment::reader::SegmentReader;
use crate::segment::types::{SegmentFiles, SegmentId, SegmentMeta};
use crate::structures::SparseFormat;

/// Default memory budget for forward index during BP (24 GB). A cap, not an
/// allocation — usage is proportional to the segment being reordered
/// (~4 B/posting + ~32 B/doc). Sized from prod evidence: a 58M-doc /
/// 5B-posting pass estimated 20.1 GB, which smaller budgets trimmed by
/// dropping highest-df dims. Mirrored by
/// `IndexConfig::default().bp_memory_budget_bytes`.
pub const DEFAULT_MEMORY_BUDGET: usize = 24 * 1024 * 1024 * 1024;

/// Unique prefix for a pass's spilled grid-run temp files.
///
/// MUST be unique per pass, not per (process, field): in a container the
/// server is always PID 1 and every index's sparse field tends to share the
/// same field id, so the old `pid_field` scheme collided across concurrent
/// reorders (documents + social both spilling field 3) — passes overwrote
/// each other's runs and deleted them on completion, failing the other pass
/// with ENOENT after it had already copied tens of GB (orphaned outputs
/// filled the production disk overnight).
fn grid_run_prefix(field_id: u32) -> String {
    format!(
        "hermes_grid_run_{}_{}_{}",
        std::process::id(),
        field_id,
        SegmentId::new().to_hex(),
    )
}

/// Owns spill paths from before their first write. Drop removes complete and
/// partially-written files on success, `?`, and panic unwind. UUID prefixes
/// also avoid colliding with leftovers from a prior process using the same
/// PID (common for PID 1 in containers).
struct GridRunFiles {
    prefix: String,
    paths: Vec<PathBuf>,
}

impl GridRunFiles {
    fn new(field_id: u32) -> Self {
        Self {
            prefix: grid_run_prefix(field_id),
            paths: Vec::new(),
        }
    }

    fn allocate(&mut self) -> PathBuf {
        let path = std::env::temp_dir().join(format!("{}_{}.tmp", self.prefix, self.paths.len()));
        self.paths.push(path.clone());
        path
    }

    fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    fn len(&self) -> usize {
        self.paths.len()
    }

    fn iter(&self) -> impl Iterator<Item = &PathBuf> {
        self.paths.iter()
    }
}

impl Drop for GridRunFiles {
    fn drop(&mut self) {
        for path in &self.paths {
            if let Err(error) = std::fs::remove_file(path)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                log::warn!("[reorder_bmp] failed to remove spill {:?}: {}", path, error);
            }
        }
    }
}

/// Reorder granularity (see docs/block-level-reorder.md).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BpGranularity {
    /// Decide per field from footer stats: coherent blocks → `Blocks`,
    /// scrambled blocks → `Records`. The decision and its inputs are logged.
    #[default]
    Auto,
    /// Record-level BP: full quality (tight block UBs + superblock
    /// locality), scatter rewrite. Compacts interior padding.
    Records,
    /// Block-level BP: permute whole blocks to recover superblock locality
    /// at ~1/block_size of the BP cost and a memcpy-class rewrite. Block
    /// composition (and therefore block UBs) unchanged.
    Blocks,
}

/// `Auto` picks block-level reorder when normalized coherence — how far the
/// measured d sits between its random-order baseline and its perfect-packing
/// bound, 0..=1 — is at least this. Value-independent: unlike an absolute d
/// cutoff, it is not skewed by the corpus's dim-frequency distribution
/// (rare-dim corpora cap d low even when perfectly clustered; ubiquitous
/// dims inflate d even when scrambled). Measured values are logged per pass
/// for tuning.
pub const BLOCKWISE_NORM_COHERENCE_THRESHOLD: f32 = 0.5;

/// Scan cap for the coherence decision: above this many blocks the scan
/// samples every k-th block instead of all of them. 8192 blocks (~512k
/// docs at block_size 64) is statistically ample for a 0.5 threshold and
/// bounds the decision cost on multi-million-doc segments.
const MAX_COHERENCE_SCAN_BLOCKS: usize = 8192;

/// Coherence statistics driving the `Auto` granularity decision.
#[derive(Clone, Copy, Debug)]
struct CoherenceStats {
    /// Raw average records per (block, dim) pair: ≈1 when blocks are
    /// scrambled, grows toward block_size as blocks become coherent.
    d: f32,
    /// Expected `d` if the same records were assigned to blocks uniformly at
    /// random: `P / Σ_t B·(1 − (1 − 1/B)^df_t)`.
    d_rand: f32,
    /// Upper bound on achievable `d`: every dim packed into its minimal
    /// `ceil(df_t / block_size)` blocks. Ignores joint constraints across
    /// dims, so it overestimates — which biases `norm` down, toward the
    /// higher-quality record-level path.
    d_max: f32,
    /// `(d − d_rand) / (d_max − d_rand)`, clamped to 0..=1. When the data
    /// offers no clustering headroom (`d_max ≈ d_rand`, e.g. only ubiquitous
    /// or singleton dims), reordering records cannot help, so this is 1.0.
    norm: f32,
    /// Blocks actually scanned vs total — differ when the sampling cap hit.
    scanned_blocks: usize,
    total_blocks: usize,
}

/// Compute [`CoherenceStats`] from a streaming pass over block headers:
/// per-dim record frequencies come from posting-slice lengths, no weight
/// decode. Dim ids in block headers are raw (not bounded by the configured
/// grid dims), so counts live in a hash map; segments above
/// [`MAX_COHERENCE_SCAN_BLOCKS`] are stride-sampled and all aggregates
/// come from the sampled sub-population, which keeps the estimator
/// consistent at both the clustered and scrambled extremes.
///
/// Singleton dims (df=1) are excluded from every aggregate: they contribute
/// identically to the actual, random, and packed pair counts (one pair
/// each), so they carry zero ordering signal and would only compress the
/// three bounds together — on id-heavy corpora (every record has a unique
/// dim) enough to trip the no-headroom epsilon and mask real headroom in
/// the informative dims.
fn block_coherence(
    sources: &[(crate::segment::BmpIndex, u32)],
    block_size: usize,
) -> CoherenceStats {
    let total_blocks: usize = sources.iter().map(|(b, _)| b.num_blocks as usize).sum();
    if total_blocks == 0 {
        return CoherenceStats {
            d: 0.0,
            d_rand: 0.0,
            d_max: 0.0,
            norm: 0.0,
            scanned_blocks: 0,
            total_blocks,
        };
    }

    let stride = total_blocks.div_ceil(MAX_COHERENCE_SCAN_BLOCKS).max(1);
    // dim → (record count, block count) within the sampled blocks
    let mut df: rustc_hash::FxHashMap<u32, (u64, u64)> = rustc_hash::FxHashMap::default();
    let mut scanned_blocks = 0usize;
    let mut global_block = 0usize;
    for (bmp, _) in sources {
        for block_id in 0..bmp.num_blocks {
            if global_block.is_multiple_of(stride) {
                scanned_blocks += 1;
                for (dim_id, posts) in bmp.iter_block_terms(block_id) {
                    let e = df.entry(dim_id).or_insert((0, 0));
                    e.0 += posts.len() as u64;
                    e.1 += 1;
                }
            }
            global_block += 1;
        }
    }

    let b = scanned_blocks as f64;
    let keep = 1.0 - 1.0 / b;
    let mut postings: u64 = 0;
    let mut terms: u64 = 0;
    let mut expected_rand_pairs = 0.0f64;
    let mut min_pairs = 0u64;
    for &(records, blocks) in df.values() {
        if records < 2 {
            continue; // singleton: inert for clustering, see above
        }
        postings += records;
        terms += blocks;
        expected_rand_pairs += b * (1.0 - keep.powf(records as f64));
        min_pairs += records.div_ceil(block_size as u64);
    }
    if terms == 0 || postings == 0 {
        // No dim shared by ≥2 records: BP has no signal at any granularity.
        return CoherenceStats {
            d: 0.0,
            d_rand: 0.0,
            d_max: 0.0,
            norm: 0.0,
            scanned_blocks,
            total_blocks,
        };
    }

    let d = postings as f32 / terms as f32;
    let d_rand = if expected_rand_pairs > 0.0 {
        (postings as f64 / expected_rand_pairs) as f32
    } else {
        d
    };
    let d_max = postings as f32 / min_pairs.max(1) as f32;
    let headroom = d_max - d_rand;
    // Relative epsilon: below ~5% headroom the bounds are within estimation
    // noise of each other and no ordering can move d meaningfully.
    let norm = if headroom <= 0.05 * d_rand {
        1.0
    } else {
        ((d - d_rand) / headroom).clamp(0.0, 1.0)
    };
    CoherenceStats {
        d,
        d_rand,
        d_max,
        norm,
        scanned_blocks,
        total_blocks,
    }
}

/// Reorder a single segment's BMP data via Recursive Graph Bisection (BP).
///
/// Creates a new segment with reordered BMP blocks for better pruning.
/// Non-BMP fields are copied unchanged via streaming file copy.
///
/// `rayon_pool`: optional bounded thread pool for BP computation. When `Some`,
/// all rayon parallel work (gain computation, recursive bisection) runs on this
/// pool instead of the global rayon pool. This prevents the optimizer from
/// saturating all CPU cores.
///
/// Returns `(new_segment_hex_id, num_docs, bp_converged)`. `bp_converged` is
/// false iff the BP wall-clock budget ended a field's pass early (the output
/// is still valid and better-ordered than the input; a later pass warm-starts
/// from it and deepens).
///
/// `granularity`: callers deepening an unconverged segment must pass
/// `BpGranularity::Records` — `Auto` would measure the partial pass's
/// residual coherence, potentially take the blockwise path, report converged,
/// and end the deepening cascade at partial quality.
#[allow(clippy::too_many_arguments)]
pub async fn reorder_segment<D: Directory + DirectoryWriter>(
    dir: &D,
    schema: &Arc<Schema>,
    source_id: SegmentId,
    output_id: SegmentId,
    term_cache_blocks: usize,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    granularity: BpGranularity,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<(String, u32, bool)> {
    let reader = SegmentReader::open(dir, source_id, Arc::clone(schema), term_cache_blocks).await?;
    let num_docs = reader.num_docs();

    let src_files = SegmentFiles::new(source_id.0);
    let dst_files = SegmentFiles::new(output_id.0);

    log::info!(
        "[reorder] segment {} → {} ({} docs)",
        source_id.to_hex(),
        output_id.to_hex(),
        num_docs,
    );

    // Copy unchanged segment files
    let copy_start = std::time::Instant::now();
    for (src, dst, required) in [
        (&src_files.term_dict, &dst_files.term_dict, true),
        (&src_files.postings, &dst_files.postings, true),
        (
            &src_files.positions,
            &dst_files.positions,
            reader.has_positions_file(),
        ),
        (&src_files.store, &dst_files.store, true),
        (
            &src_files.fast,
            &dst_files.fast,
            !reader.fast_fields().is_empty(),
        ),
        (
            &src_files.vectors,
            &dst_files.vectors,
            !reader.vector_indexes().is_empty() || !reader.flat_vectors().is_empty(),
        ),
    ] {
        copy_segment_file(dir, src, dst, required).await?;
    }
    log::info!(
        "[reorder] copied files in {:.1}s",
        copy_start.elapsed().as_secs_f64(),
    );

    // Rebuild sparse file with reordered BMP data
    let bp_converged = reorder_sparse_file(
        dir,
        &reader,
        &dst_files,
        schema,
        memory_budget,
        bp_budget,
        granularity,
        rayon_pool,
    )
    .await?;

    // Write new meta with output segment ID
    let src_meta = reader.meta();
    let meta = SegmentMeta {
        id: output_id.0,
        num_docs: src_meta.num_docs,
        field_stats: src_meta.field_stats.clone(),
    };
    // Durable: the reordered segment replaces its fsynced source, so a
    // non-durable .meta could be the only copy across a power failure.
    dir.write_durable(&dst_files.meta, &meta.serialize()?)
        .await?;

    Ok((output_id.to_hex(), num_docs, bp_converged))
}

/// Copy a segment file via streaming I/O (4 MB chunks).
///
/// No-op if an optional source file does not exist. A file observed by the
/// source reader is required: losing it during the copy is corruption, not an
/// empty optional field.
/// Empty files are still created (SegmentReader requires their existence).
async fn copy_segment_file<D: Directory + DirectoryWriter>(
    dir: &D,
    src: &Path,
    dst: &Path,
    required: bool,
) -> Result<()> {
    let handle = match dir.open_read(src).await {
        Ok(h) => h,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !required => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(crate::Error::Corruption(format!(
                "required reorder source file {src:?} disappeared before copy"
            )));
        }
        Err(error) => return Err(crate::Error::Io(error)),
    };
    let mut writer = dir
        .streaming_writer_cold(dst)
        .await
        .map_err(crate::Error::Io)?;
    const CHUNK: usize = 4 * 1024 * 1024;
    let source_len = handle.len();
    let mut offset = 0u64;
    while offset < source_len {
        let end = (offset + CHUNK as u64).min(source_len);
        let data = handle
            .read_bytes_range(offset..end)
            .await
            .map_err(crate::Error::Io)?;
        writer = tokio::task::spawn_blocking(move || {
            writer.write_all(data.as_slice())?;
            // Drop source pages behind the copy cursor — a whole-file copy
            // must not leave the complete source resident in page cache.
            #[cfg(feature = "native")]
            data.madvise_range(0..data.len(), libc::MADV_DONTNEED);
            Ok::<_, std::io::Error>(writer)
        })
        .await
        .map_err(|error| {
            crate::Error::Internal(format!("reorder copy worker failed for {src:?}: {error}"))
        })?
        .map_err(crate::Error::Io)?;
        offset = end;
    }
    if writer.bytes_written() != source_len {
        return Err(crate::Error::Corruption(format!(
            "short reorder copy from {:?} to {:?}: wrote {} of {} bytes",
            src,
            dst,
            writer.bytes_written(),
            source_len,
        )));
    }
    tokio::task::spawn_blocking(move || writer.finish())
        .await
        .map_err(|error| {
            crate::Error::Internal(format!(
                "reorder copy finalizer failed for {dst:?}: {error}"
            ))
        })?
        .map_err(crate::Error::Io)?;

    Ok(())
}

/// Rebuild the sparse file with reordered BMP data.
///
/// - BMP fields: run BP reorder and write new blob
/// - MaxScore fields: identity-copy raw data from source
/// - No sparse fields: no-op
#[allow(clippy::too_many_arguments)]
async fn reorder_sparse_file<D: Directory + DirectoryWriter>(
    dir: &D,
    reader: &SegmentReader,
    dst_files: &SegmentFiles,
    schema: &Schema,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    granularity: BpGranularity,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<bool> {
    let sparse_fields: Vec<_> = schema
        .fields()
        .filter(|(_, entry)| matches!(entry.field_type, FieldType::SparseVector))
        .map(|(field, entry)| {
            (
                field,
                entry.sparse_vector_config.clone(),
                entry.reorder,
                entry.name.clone(),
            )
        })
        .collect();

    if sparse_fields.is_empty() {
        return Ok(true);
    }

    // Check if there's any BMP data to reorder. Per-field gate: only fields
    // with the `reorder` schema attribute get BP; others are copied unchanged.
    let has_bmp_data = sparse_fields.iter().any(|(field, config, reorder, _)| {
        *reorder
            && config.as_ref().map(|c| c.format) == Some(SparseFormat::Bmp)
            && reader.bmp_indexes().get(&field.0).is_some()
    });

    // Sparse files are legitimately absent when the schema has sparse fields
    // but this segment contains no sparse values. Do not turn that valid
    // optional-file case into a required-copy corruption error.
    if reader.sparse_indexes().is_empty() && reader.bmp_indexes().is_empty() {
        return Ok(true);
    }

    if !has_bmp_data {
        // No BMP field wants reordering — just copy the sparse file as-is
        log::info!(
            "[reorder] segment {:x}: no BMP field has the `reorder` schema attribute — sparse file copied unchanged",
            reader.meta().id,
        );
        let src_files = SegmentFiles::new(reader.meta().id);
        copy_segment_file(dir, &src_files.sparse, &dst_files.sparse, true).await?;
        return Ok(true);
    }

    let mut all_converged = true;
    let mut writer = OffsetWriter::new(
        dir.streaming_writer_cold(&dst_files.sparse)
            .await
            .map_err(crate::Error::Io)?,
    );
    let mut field_tocs: Vec<SparseFieldToc> = Vec::new();
    let mut all_skip_bytes: Vec<u8> = Vec::new();
    let mut skip_count: u32 = 0;

    for (field, sparse_config, reorder, field_name) in &sparse_fields {
        let format = sparse_config.as_ref().map(|c| c.format).unwrap_or_default();
        let quantization = sparse_config
            .as_ref()
            .map(|c| c.weight_quantization)
            .unwrap_or(crate::structures::WeightQuantization::Float32);

        if format == SparseFormat::Bmp {
            if let Some(bmp_idx) = reader.bmp_indexes().get(&field.0) {
                if !*reorder {
                    // Field opted out of BP: copy its blob byte-identically.
                    log::info!(
                        "[reorder] field {}: `reorder` attribute not set — blob copied unchanged",
                        field.0,
                    );
                    copy_bmp_blob(
                        bmp_idx,
                        field.0,
                        quantization,
                        bmp_idx.total_vectors,
                        &mut writer,
                        &mut field_tocs,
                    )?;
                    continue;
                }
                let effective_block_size = sparse_config
                    .as_ref()
                    .map(|c| c.bmp_block_size)
                    .unwrap_or(64)
                    .clamp(1, 256) as usize;
                let dims = sparse_config
                    .as_ref()
                    .and_then(|c| c.dims)
                    .unwrap_or_else(|| bmp_idx.dims());
                let grid_bits = sparse_config.as_ref().map(|c| c.bmp_grid_bits).unwrap_or(4);
                let max_weight_scale = sparse_config
                    .as_ref()
                    .and_then(|c| c.max_weight)
                    .unwrap_or(bmp_idx.max_weight_scale);
                let total_vectors = bmp_idx.total_vectors;

                // Clone BmpIndex (cheap: Arc ref bumps on OwnedBytes) and move
                // OffsetWriter + field_tocs into spawn_blocking so the entire
                // CPU-heavy reorder runs off tokio worker threads.
                let bmp_sources = vec![(bmp_idx.clone(), 0u32)];
                let fid = field.0;
                let fname = field_name.clone();
                let ilabel = schema.index_label().to_owned();
                let pool = rayon_pool.clone();
                let (w, ft, converged) = tokio::task::spawn_blocking(move || {
                    reorder_bmp_field(
                        &bmp_sources,
                        fid,
                        &ilabel,
                        &fname,
                        quantization,
                        dims,
                        effective_block_size,
                        grid_bits,
                        max_weight_scale,
                        total_vectors,
                        memory_budget,
                        bp_budget,
                        granularity,
                        writer,
                        field_tocs,
                        pool,
                    )
                })
                .await
                .map_err(|e| {
                    crate::Error::Internal(format!("reorder_bmp_field panicked: {}", e))
                })??;
                writer = w;
                field_tocs = ft;
                all_converged &= converged;
            }
        } else {
            // MaxScore format: identity-copy raw data from source
            if let Some(sparse_idx) = reader.sparse_indexes().get(&field.0) {
                copy_maxscore_field(
                    sparse_idx,
                    field.0,
                    quantization,
                    &mut writer,
                    &mut field_tocs,
                    &mut all_skip_bytes,
                    &mut skip_count,
                )
                .await?;
            }
        }
    }

    if field_tocs.is_empty() {
        drop(writer);
        let _ = dir.delete(&dst_files.sparse).await;
        return Ok(all_converged);
    }

    // Write skip section (MaxScore fields only)
    let skip_offset = writer.offset();
    if !all_skip_bytes.is_empty() {
        writer
            .write_all(&all_skip_bytes)
            .map_err(crate::Error::Io)?;
    }
    drop(all_skip_bytes);

    // Write TOC + footer
    let toc_offset = writer.offset();
    write_sparse_toc_and_footer(&mut writer, skip_offset, toc_offset, &field_tocs)
        .map_err(crate::Error::Io)?;

    writer.finish().map_err(crate::Error::Io)?;

    let total_dims: usize = field_tocs.iter().map(|f| f.dims.len()).sum();
    log::info!(
        "[reorder] sparse file written: {} fields, {} dims, {} skip entries (bp_converged={})",
        field_tocs.len(),
        total_dims,
        skip_count,
        all_converged,
    );

    Ok(all_converged)
}

/// Block-level reorder: permute whole blocks so similar blocks share
/// superblocks — a permuted block copy, no record unpacking. See
/// docs/block-level-reorder.md. Interior padding is preserved (blocks are
/// copied verbatim); block UBs are unchanged by construction.
#[allow(clippy::too_many_arguments)]
fn reorder_bmp_field_blockwise(
    sources: &[(crate::segment::BmpIndex, u32)],
    field_id: u32,
    quantization: crate::structures::WeightQuantization,
    dims: u32,
    effective_block_size: usize,
    grid_bits: u8,
    max_weight_scale: f32,
    total_vectors: u32,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    mut writer: OffsetWriter,
    mut field_tocs: Vec<SparseFieldToc>,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<(OffsetWriter, Vec<SparseFieldToc>, bool)> {
    use crate::segment::builder::bmp::write_bmp_footer;
    use crate::segment::builder::graph_bisection::{
        build_forward_index_from_blocks, graph_bisection,
    };
    use crate::segment::reader::bmp::BMP_SUPERBLOCK_SIZE;

    let bmp_refs: Vec<&crate::segment::BmpIndex> = sources.iter().map(|(b, _)| b).collect();
    let source_pages =
        crate::segment::reader::bmp::BmpScanPageGuard::new(sources.iter().map(|(bmp, _)| bmp));
    let num_blocks_total = bmp_refs
        .iter()
        .try_fold(0usize, |total, bmp| {
            total.checked_add(bmp.num_blocks as usize)
        })
        .ok_or_else(|| {
            crate::Error::Internal("reordered BMP block count overflows usize".into())
        })?;
    if num_blocks_total == 0 {
        return Ok((writer, field_tocs, true));
    }
    if num_blocks_total > u32::MAX as usize {
        return Err(crate::Error::Internal(
            "reordered BMP block count exceeds the V14 u32 format limit".into(),
        ));
    }
    let num_virtual_docs = num_blocks_total
        .checked_mul(effective_block_size)
        .filter(|&count| count <= u32::MAX as usize)
        .ok_or_else(|| {
            crate::Error::Internal(
                "reordered BMP virtual-document count exceeds the V14 u32 format limit".into(),
            )
        })?;

    // Global block idx → source: prefix sums.
    let block_base: Vec<usize> = std::iter::once(0)
        .chain(bmp_refs.iter().scan(0usize, |acc, b| {
            *acc += b.num_blocks as usize;
            Some(*acc)
        }))
        .collect();
    let resolve = |global: usize| -> (usize, usize) {
        let src = block_base.partition_point(|&b| b <= global) - 1;
        (src, global - block_base[src])
    };

    // ── BP over blocks, down to superblock granularity ──────────────────
    let bp_start = std::time::Instant::now();
    let sb = BMP_SUPERBLOCK_SIZE as usize;
    // The budget's depth cap is in DOCS but BP entities here are BLOCKS —
    // convert units, or the optimizer's partial cap (e.g. 4096 docs) would
    // be read as 4096 blocks and silently stop BP above superblock depth.
    let block_min_partition = bp_budget
        .min_partition_docs
        .map(|docs| (docs / effective_block_size).max(1));
    // Forward-index build is parallel too — keep it on the bounded pool.
    let run_bp = || {
        if bp_budget.time_budget.is_some_and(|budget| budget.is_zero()) {
            return ((0..num_blocks_total as u32).collect(), false);
        }
        let fwd = build_forward_index_from_blocks(&bmp_refs, memory_budget);
        if fwd.num_terms > 0 && num_blocks_total > sb {
            // The user-visible time budget covers the forward-index build as
            // well as graph refinement. Previously a multi-minute build ran
            // before a fresh full budget started, defeating the lifecycle cap.
            let block_budget = crate::segment::BpBudget {
                min_partition_docs: block_min_partition,
                time_budget: bp_budget
                    .time_budget
                    .map(|budget| budget.saturating_sub(bp_start.elapsed())),
            };
            graph_bisection(&fwd, sb, 20, block_budget)
        } else {
            (
                (0..num_blocks_total as u32).collect(),
                num_blocks_total <= sb || !fwd.budget_limited(),
            )
        }
    };
    let (perm, converged) = if let Some(ref pool) = rayon_pool {
        pool.install(run_bp)
    } else {
        run_bp()
    };
    // Resolving a global block through source prefix sums is cheap once, but
    // catastrophic inside the `dims × blocks` grid rewrite below (billions
    // of calls on production-sized fields). Cache the compact source/local
    // pair once per output block; this is 8 bytes/block.
    let permuted_blocks: Vec<(u32, u32)> = perm
        .iter()
        .map(|&old_global| {
            let (source, local_block) = resolve(old_global as usize);
            (source as u32, local_block as u32)
        })
        .collect();
    log::info!(
        "[reorder_bmp] field {}: blockwise BP over {} blocks in {:.1}ms (converged={})",
        field_id,
        num_blocks_total,
        bp_start.elapsed().as_secs_f64() * 1000.0,
        converged,
    );
    source_pages.switch_to_random();

    // ── Write blob: permuted block copy ─────────────────────────────────
    let blob_start = writer.offset();

    // Section B: block data verbatim, in permuted order
    let mut block_data_starts: Vec<u64> = Vec::with_capacity(num_blocks_total + 1);
    let mut cumulative: u64 = 0;
    let mut total_terms: u64 = 0;
    let mut total_postings: u64 = 0;
    for b in &bmp_refs {
        total_terms = total_terms.saturating_add(b.total_terms());
        total_postings = total_postings.saturating_add(b.total_postings());
    }
    for &(src, lb) in &permuted_blocks {
        let bmp = bmp_refs[src as usize];
        let start = bmp.block_data_start(lb) as usize;
        let end = if lb + 1 < bmp.num_blocks {
            bmp.block_data_start(lb + 1) as usize
        } else {
            bmp.block_data_sentinel() as usize
        };
        block_data_starts.push(cumulative);
        let bytes = &bmp.block_data_slice()[start..end];
        writer.write_all(bytes).map_err(crate::Error::Io)?;
        cumulative += bytes.len() as u64;
    }
    block_data_starts.push(cumulative);

    // Padding + Section A: block_data_starts
    let block_data_len = writer.offset() - blob_start;
    let padding = (8 - (block_data_len % 8) as usize) % 8;
    if padding > 0 {
        writer
            .write_all(&[0u8; 8][..padding])
            .map_err(crate::Error::Io)?;
    }
    for &v in &block_data_starts {
        writer
            .write_all(&v.to_le_bytes())
            .map_err(crate::Error::Io)?;
    }
    drop(block_data_starts);

    // Sections D + E: permuted grid rows + recomputed sb_grid, streamed per dim
    let grid_offset = writer.offset() - blob_start;
    let packed_row_size =
        crate::segment::builder::bmp::grid_packed_row_size(num_blocks_total, grid_bits);
    let num_superblocks = num_blocks_total.div_ceil(sb);
    let mut out_row = vec![0u8; packed_row_size];
    let mut sb_row = vec![0u8; num_superblocks];
    // Section E follows every Section D row on disk. Keeping all E rows until
    // then was O(dims × superblocks) RAM (tens of GB at production scale).
    // Spill E linearly while computing D; the RAII owner removes partial files
    // on every exit path.
    let mut sb_spill_files = GridRunFiles::new(field_id);
    let sb_spill_path = sb_spill_files.allocate();
    let sb_spill = std::fs::File::create(&sb_spill_path).map_err(crate::Error::Io)?;
    let mut sb_spill_writer = std::io::BufWriter::with_capacity(4 * 1024 * 1024, sb_spill);
    let dequant = crate::segment::builder::bmp::grid_dequant_scale(grid_bits);

    for dim in 0..dims as usize {
        out_row.fill(0);
        sb_row.fill(0);
        for (new_pos, &(src, lb)) in permuted_blocks.iter().enumerate() {
            let bmp = bmp_refs[src as usize];
            let prs = bmp.packed_row_size();
            let row = &bmp.grid_slice()[dim * prs..dim * prs + prs];
            let cell = crate::segment::builder::bmp::grid_get_cell(row, lb as usize, grid_bits);
            if cell == 0 {
                continue;
            }
            crate::segment::builder::bmp::grid_set_cell(&mut out_row, new_pos, cell, grid_bits);
            // sb_grid stores u8 impact scale (0-255) everywhere else (builder,
            // block-copy merge). Dequantize the cell to its safe u8 upper
            // bound — writing the raw cell here deflated superblock UBs ~17×
            // after blockwise passes (unsafe pruning once heaps fill).
            let ub = (cell as u32 * dequant).min(255) as u8;
            let slot = &mut sb_row[new_pos / sb];
            if ub > *slot {
                *slot = ub;
            }
        }
        writer.write_all(&out_row).map_err(crate::Error::Io)?;
        sb_spill_writer
            .write_all(&sb_row)
            .map_err(crate::Error::Io)?;
    }
    drop(out_row);
    drop(sb_row);
    sb_spill_writer.flush().map_err(crate::Error::Io)?;
    drop(sb_spill_writer);

    let sb_grid_offset = writer.offset() - blob_start;
    let mut sb_spill_reader = std::fs::File::open(&sb_spill_path).map_err(crate::Error::Io)?;
    let mut copy_buffer = vec![0u8; 4 * 1024 * 1024];
    loop {
        let read = sb_spill_reader
            .read(&mut copy_buffer)
            .map_err(crate::Error::Io)?;
        if read == 0 {
            break;
        }
        writer
            .write_all(&copy_buffer[..read])
            .map_err(crate::Error::Io)?;
    }
    drop(copy_buffer);
    drop(sb_spill_reader);

    // Sections F + G: doc maps copied per block chunk, ids offset-patched
    let doc_map_offset = writer.offset() - blob_start;
    let bs = effective_block_size;
    let mut num_real_docs: u32 = 0;
    for b in &bmp_refs {
        num_real_docs = num_real_docs
            .checked_add(b.num_real_docs())
            .ok_or_else(|| {
                crate::Error::Internal(
                    "reordered BMP real-document count exceeds the V14 u32 format limit".into(),
                )
            })?;
    }
    let mut id_chunk = vec![0u8; bs * 4];
    for &(src, lb) in &permuted_blocks {
        let src = src as usize;
        let lb = lb as usize;
        let (bmp, doc_offset) = (&sources[src].0, sources[src].1);
        let ids = bmp.doc_map_ids_slice();
        id_chunk.copy_from_slice(&ids[lb * bs * 4..(lb + 1) * bs * 4]);
        if doc_offset != 0 {
            let (chunks, _) = id_chunk.as_chunks_mut::<4>();
            for e in chunks {
                let doc_id = u32::from_le_bytes(*e);
                if doc_id != u32::MAX {
                    *e = doc_id
                        .checked_add(doc_offset)
                        .ok_or_else(|| {
                            crate::Error::Corruption(format!(
                                "BMP doc-id offset overflow: {} + {}",
                                doc_id, doc_offset
                            ))
                        })?
                        .to_le_bytes();
                }
            }
        }
        writer.write_all(&id_chunk).map_err(crate::Error::Io)?;
    }
    for &(src, lb) in &permuted_blocks {
        let src = src as usize;
        let lb = lb as usize;
        let ords = bmp_refs[src].doc_map_ordinals_slice();
        writer
            .write_all(&ords[lb * bs * 2..(lb + 1) * bs * 2])
            .map_err(crate::Error::Io)?;
    }

    // Footer
    write_bmp_footer(
        &mut writer,
        total_terms,
        total_postings,
        grid_offset,
        sb_grid_offset,
        num_blocks_total as u32,
        dims,
        effective_block_size as u32,
        num_virtual_docs as u32,
        max_weight_scale,
        doc_map_offset,
        num_real_docs,
        grid_bits,
    )
    .map_err(crate::Error::Io)?;

    let blob_len = writer.offset() - blob_start;
    let mut config_for_byte =
        crate::structures::SparseVectorConfig::from_byte(quantization as u8).unwrap_or_default();
    config_for_byte.format = SparseFormat::Bmp;
    config_for_byte.weight_quantization = quantization;
    field_tocs.push(SparseFieldToc {
        field_id,
        quantization: config_for_byte.to_byte(),
        total_vectors,
        dims: vec![crate::segment::format::SparseDimTocEntry {
            dim_id: 0xFFFFFFFF,
            block_data_offset: blob_start,
            skip_start: (blob_len & 0xFFFFFFFF) as u32,
            num_blocks: ((blob_len >> 32) & 0xFFFFFFFF) as u32,
            doc_count: 0,
            max_weight: 0.0,
        }],
    });

    log::info!(
        "[reorder_bmp] field {}: blockwise reorder done — {} blocks, {:.2} MB",
        field_id,
        num_blocks_total,
        blob_len as f64 / (1024.0 * 1024.0),
    );

    Ok((writer, field_tocs, converged))
}

/// Identity-copy a BMP field's raw blob (fields whose `reorder` schema
/// attribute is unset). Byte-identical: insertion order, padding, and footer
/// are all preserved.
fn copy_bmp_blob(
    bmp: &crate::segment::BmpIndex,
    field_id: u32,
    quantization: crate::structures::WeightQuantization,
    total_vectors: u32,
    writer: &mut OffsetWriter,
    field_tocs: &mut Vec<SparseFieldToc>,
) -> Result<()> {
    let blob = bmp.read_raw_blob().map_err(crate::Error::Io)?;
    let blob_start = writer.offset();
    const CHUNK: usize = 4 * 1024 * 1024;
    for chunk in blob.as_slice().chunks(CHUNK) {
        writer.write_all(chunk).map_err(crate::Error::Io)?;
    }
    let blob_len = writer.offset() - blob_start;

    let mut config_for_byte =
        crate::structures::SparseVectorConfig::from_byte(quantization as u8).unwrap_or_default();
    config_for_byte.format = SparseFormat::Bmp;
    config_for_byte.weight_quantization = quantization;

    field_tocs.push(SparseFieldToc {
        field_id,
        quantization: config_for_byte.to_byte(),
        total_vectors,
        dims: vec![crate::segment::format::SparseDimTocEntry {
            dim_id: 0xFFFFFFFF, // sentinel for BMP
            block_data_offset: blob_start,
            skip_start: (blob_len & 0xFFFFFFFF) as u32,
            num_blocks: ((blob_len >> 32) & 0xFFFFFFFF) as u32,
            doc_count: 0,
            max_weight: 0.0,
        }],
    });
    Ok(())
}

/// Identity-copy a MaxScore sparse field from source to destination.
#[allow(clippy::too_many_arguments)]
async fn copy_maxscore_field(
    sparse_idx: &crate::segment::SparseIndex,
    field_id: u32,
    quantization: crate::structures::WeightQuantization,
    writer: &mut OffsetWriter,
    field_tocs: &mut Vec<SparseFieldToc>,
    all_skip_bytes: &mut Vec<u8>,
    skip_count: &mut u32,
) -> Result<()> {
    let all_dims: Vec<u32> = sparse_idx.active_dimensions().collect();
    if all_dims.is_empty() {
        return Ok(());
    }

    let total_vectors = sparse_idx.total_vectors;
    let mut dim_toc_entries = Vec::with_capacity(all_dims.len());

    for &dim_id in &all_dims {
        let raw = match sparse_idx.read_dim_raw(dim_id).await? {
            Some(r) => r,
            None => continue,
        };

        if raw.raw_block_data.as_slice().is_empty() {
            continue;
        }

        let block_data_offset = writer.offset();
        let skip_start = *skip_count;
        let num_blocks = raw.skip_entries.len() as u32;

        // Write raw block data (identity copy)
        writer
            .write_all(raw.raw_block_data.as_slice())
            .map_err(crate::Error::Io)?;

        // Accumulate skip entries (offsets are already correct for single source)
        for entry in &raw.skip_entries {
            entry.write_to_vec(all_skip_bytes);
            *skip_count += 1;
        }

        dim_toc_entries.push(crate::segment::format::SparseDimTocEntry {
            dim_id,
            block_data_offset,
            skip_start,
            num_blocks,
            doc_count: raw.doc_count,
            max_weight: raw.global_max_weight,
        });
    }

    if !dim_toc_entries.is_empty() {
        field_tocs.push(SparseFieldToc {
            field_id,
            quantization: quantization as u8,
            total_vectors,
            dims: dim_toc_entries,
        });
    }

    Ok(())
}

/// Reorder one BMP field via Recursive Graph Bisection (BP), single- or
/// multi-source.
///
/// Reads block data from the source `BmpIndex`es (each paired with its doc-id
/// offset in the output segment), builds a combined forward index, runs BP to
/// compute a permutation over *real* (non-padding) records, then writes one
/// reordered blob. With a single `(bmp, 0)` source this is the standalone
/// segment reorder; with multiple sources it is the merge-time reorder path,
/// which replaces byte-level block stacking.
///
/// Realness is derived from each source's doc map (`build_vid_maps`), never
/// from `vid < num_real_docs` — block-copy merged sources carry interior
/// padding. Interior padding is compacted away in the output: the written
/// blob always has tail-only padding.
///
/// This is a synchronous function called from `spawn_blocking` so the
/// entire CPU-heavy reorder (forward index build, BP, blob write) runs
/// off tokio worker threads. The `OffsetWriter` streams directly to disk
/// — no in-memory buffering of the output blob.
///
/// When `rayon_pool` is `Some`, all rayon parallel work runs on that pool
/// instead of the global pool, bounding optimizer CPU usage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reorder_bmp_field(
    sources: &[(crate::segment::BmpIndex, u32)],
    field_id: u32,
    index_label: &str,
    field_name: &str,
    quantization: crate::structures::WeightQuantization,
    dims: u32,
    effective_block_size: usize,
    grid_bits: u8,
    max_weight_scale: f32,
    total_vectors: u32,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    granularity: BpGranularity,
    mut writer: OffsetWriter,
    mut field_tocs: Vec<SparseFieldToc>,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<(OffsetWriter, Vec<SparseFieldToc>, bool)> {
    if !(1..=256).contains(&effective_block_size) {
        return Err(crate::Error::Internal(format!(
            "invalid BMP reorder block size {} (expected 1..=256)",
            effective_block_size
        )));
    }
    use crate::segment::builder::bmp::{
        GridRunReader, stream_write_grids, stream_write_grids_merged, write_bmp_footer,
        write_grid_run,
    };
    use crate::segment::builder::graph_bisection::{
        build_forward_index_from_bmps_with_maps, build_vid_maps, graph_bisection,
    };

    if sources.is_empty() {
        return Ok((writer, field_tocs, true));
    }

    // ── Granularity decision (docs/block-level-reorder.md) ──────────────
    // The coherence scan runs only for `Auto`: explicit granularity (manual
    // override, or a deepening pass on an unconverged segment) must not pay
    // a header scan just to decide what is already decided.
    let effective_granularity = match granularity {
        BpGranularity::Auto => {
            let stats_start = std::time::Instant::now();
            let coherence = block_coherence(sources, effective_block_size);
            let chosen = if coherence.norm >= BLOCKWISE_NORM_COHERENCE_THRESHOLD {
                BpGranularity::Blocks
            } else {
                BpGranularity::Records
            };
            log::info!(
                "[reorder_bmp] field {}: coherence norm={:.3} (d={:.2}, rand={:.2}, max={:.2}, threshold {:.2}, {}/{} blocks scanned in {:.1}ms) → {:?} granularity",
                field_id,
                coherence.norm,
                coherence.d,
                coherence.d_rand,
                coherence.d_max,
                BLOCKWISE_NORM_COHERENCE_THRESHOLD,
                coherence.scanned_blocks,
                coherence.total_blocks,
                stats_start.elapsed().as_secs_f64() * 1000.0,
                chosen,
            );
            crate::observe::reorder_coherence(index_label, field_name, coherence.d, coherence.norm);
            chosen
        }
        explicit => {
            log::info!(
                "[reorder_bmp] field {}: {:?} granularity (explicit, coherence scan skipped)",
                field_id,
                explicit,
            );
            explicit
        }
    };
    crate::observe::reorder_granularity(
        index_label,
        field_name,
        match effective_granularity {
            BpGranularity::Blocks => "blocks",
            _ => "records",
        },
    );

    if effective_granularity == BpGranularity::Blocks {
        return reorder_bmp_field_blockwise(
            sources,
            field_id,
            quantization,
            dims,
            effective_block_size,
            grid_bits,
            max_weight_scale,
            total_vectors,
            memory_budget,
            bp_budget,
            writer,
            field_tocs,
            rayon_pool,
        );
    }

    // Record-level BP needs both directions of every document map before it
    // can even start building the forward graph. Reserve that non-negotiable
    // peak up front; previously these vectors were allocated outside the
    // advertised budget. Fall back to a valid blockwise rewrite when the
    // record representation itself cannot fit, but keep the result marked
    // unconverged so a larger-budget/manual pass may still deepen it later.
    let record_map_bytes = sources.iter().fold(0usize, |total, (bmp, _)| {
        total
            .saturating_add((bmp.num_virtual_docs as usize).saturating_mul(4))
            .saturating_add((bmp.num_real_docs() as usize).saturating_mul(4))
    });
    let record_rewrite_fixed_bytes = sources.iter().fold(0usize, |total, (bmp, _)| {
        total
            // retained real→virtual map + output permutation
            .saturating_add((bmp.num_real_docs() as usize).saturating_mul(8))
            // output block-offset table
            .saturating_add((bmp.num_blocks as usize).saturating_mul(8))
    });
    let record_fixed_peak = record_map_bytes.max(record_rewrite_fixed_bytes);
    const MIN_RECORD_WORKSPACE: usize = 16 * 1024 * 1024;
    if record_fixed_peak > memory_budget.saturating_sub(MIN_RECORD_WORKSPACE) {
        log::warn!(
            "[reorder_bmp] field {}: record maps need {:.0} MB of the {:.0} MB total budget; falling back to blockwise order",
            field_id,
            record_fixed_peak as f64 / (1024.0 * 1024.0),
            memory_budget as f64 / (1024.0 * 1024.0),
        );
        let (writer, field_tocs, _) = reorder_bmp_field_blockwise(
            sources,
            field_id,
            quantization,
            dims,
            effective_block_size,
            grid_bits,
            max_weight_scale,
            total_vectors,
            memory_budget,
            bp_budget,
            writer,
            field_tocs,
            rayon_pool,
        )?;
        return Ok((writer, field_tocs, false));
    }

    let source_pages =
        crate::segment::reader::bmp::BmpScanPageGuard::new(sources.iter().map(|(bmp, _)| bmp));

    // ── Phase 1: Build forward index and run BP ─────────────────────────
    let bp_start = std::time::Instant::now();

    let bmp_refs: Vec<&crate::segment::BmpIndex> = sources.iter().map(|(b, _)| b).collect();
    // Build padding-aware maps once. Forward-index construction needs both
    // directions; output encoding reuses real→virtual instead of rescanning
    // every source document map.
    let mut vid_maps: Vec<(Vec<u32>, Vec<u32>)> = bmp_refs
        .iter()
        .map(|bmp| build_vid_maps(bmp))
        .collect::<Result<_>>()?;
    // Prefix sums of real doc counts: source s owns global real ids
    // real_base[s]..real_base[s+1].
    let mut real_base = Vec::with_capacity(vid_maps.len() + 1);
    real_base.push(0usize);
    for (_, real_to_virtual) in &vid_maps {
        let next = real_base
            .last()
            .copied()
            .unwrap_or(0)
            .checked_add(real_to_virtual.len())
            .ok_or_else(|| {
                crate::Error::Internal("reordered BMP real-document count overflows usize".into())
            })?;
        real_base.push(next);
    }
    let num_real_docs = *real_base.last().unwrap();
    if num_real_docs == 0 {
        return Ok((writer, field_tocs, true));
    }
    if num_real_docs > u32::MAX as usize {
        return Err(crate::Error::Internal(
            "reordered BMP real-document count exceeds the V14 u32 format limit".into(),
        ));
    }

    log::info!(
        "[reorder_bmp] field {}: running BP on {} real docs from {} source(s)",
        field_id,
        num_real_docs,
        sources.len(),
    );

    // Zero time budget = identity permutation by construction — skip the
    // forward-index build entirely, it would be discarded unread. Reported
    // unconverged, matching budgeted-pass semantics (a follow-up pass
    // deepens).
    let zero_budget = bp_budget.time_budget.is_some_and(|d| d.is_zero());

    let (perm, converged) = if zero_budget {
        // No forward graph will consume virtual→real. Release it before
        // allocating the identity permutation; otherwise the zero-budget path
        // briefly holds both full map directions plus the permutation and can
        // exceed the same budget check that protects normal passes.
        for (virtual_to_real, _) in &mut vid_maps {
            *virtual_to_real = Vec::new();
        }
        log::info!(
            "[reorder_bmp] field {}: zero time budget — identity re-block, forward index skipped",
            field_id,
        );
        ((0..num_real_docs as u32).collect(), false)
    } else {
        let max_doc_freq = ((num_real_docs as f64) * 0.9) as usize;
        // Scale the df floor with segment size: a fixed 128 keeps only dims in
        // >=2.5% of docs on a 5k-doc segment — exactly the discriminative
        // mid-frequency dims BP needs — making reorder a near-no-op on small
        // segments. Rare dims are cheap (few postings), and the memory budget
        // already drops the highest-df dims when the forward index overflows.
        let min_doc_freq = (num_real_docs / 5000).clamp(2, 128);

        // Forward-index build is parallel too — keep it on the bounded pool.
        let forward_budget = memory_budget.saturating_sub(record_map_bytes);
        let build_forward = || {
            build_forward_index_from_bmps_with_maps(
                &bmp_refs,
                &vid_maps,
                min_doc_freq,
                max_doc_freq.max(1),
                forward_budget,
            )
        };
        let (fwd, _source_doc_counts) = if let Some(ref pool) = rayon_pool {
            pool.install(build_forward)
        } else {
            build_forward()
        };

        log::info!(
            "[reorder_bmp] field {}: forward index built in {:.1}ms ({} terms, {} postings)",
            field_id,
            bp_start.elapsed().as_secs_f64() * 1000.0,
            fwd.num_terms,
            fwd.total_postings(),
        );

        // Forward construction no longer needs virtual→real. Release those
        // potentially hundreds of MB before graph scratch reaches its peak;
        // retain real→virtual for output encoding.
        for (virtual_to_real, _) in &mut vid_maps {
            *virtual_to_real = Vec::new();
        }

        if fwd.num_terms > 0 && num_real_docs > effective_block_size {
            let graph_start = std::time::Instant::now();
            let graph_budget = crate::segment::BpBudget {
                min_partition_docs: bp_budget.min_partition_docs,
                time_budget: bp_budget
                    .time_budget
                    .map(|budget| budget.saturating_sub(bp_start.elapsed())),
            };
            let run_graph = || graph_bisection(&fwd, effective_block_size, 20, graph_budget);
            let (perm, converged) = if let Some(ref pool) = rayon_pool {
                pool.install(run_graph)
            } else {
                run_graph()
            };
            log::info!(
                "[reorder_bmp] field {}: BP completed in {:.1}ms (converged={})",
                field_id,
                graph_start.elapsed().as_secs_f64() * 1000.0,
                converged,
            );
            (perm, converged)
        } else {
            (
                (0..num_real_docs as u32).collect(),
                num_real_docs <= effective_block_size || !fwd.budget_limited(),
            )
        }
    };

    let real_to_virtual: Vec<Vec<u32>> = vid_maps.into_iter().map(|(_, real)| real).collect();

    log::info!(
        "[reorder_bmp] field {}: writing reordered blob ({} blocks)",
        field_id,
        num_real_docs.div_ceil(effective_block_size),
    );
    source_pages.switch_to_random();

    // ── Phase 2: Write new blob with records in permuted order ──────────
    let new_num_blocks = num_real_docs.div_ceil(effective_block_size);
    if new_num_blocks > u32::MAX as usize {
        return Err(crate::Error::Internal(
            "reordered BMP block count exceeds the V14 u32 format limit".into(),
        ));
    }
    let new_num_virtual_docs = new_num_blocks
        .checked_mul(effective_block_size)
        .filter(|&count| count <= u32::MAX as usize)
        .ok_or_else(|| {
            crate::Error::Internal(
                "reordered BMP virtual-document count exceeds the V14 u32 format limit".into(),
            )
        })?;

    let blob_start = writer.offset();
    let mut block_data_starts: Vec<u64> = Vec::with_capacity(new_num_blocks + 1);
    let mut total_terms: u64 = 0;
    let mut total_postings: u64 = 0;
    let mut cumulative_bytes: u64 = 0;

    // Grid entries and parallel encoded blocks coexist with the retained
    // real→virtual map and permutation. Divide the actual remaining pass
    // budget between them instead of adding fixed 512+256 MB allocations on
    // top of `--bp-memory-budget-mb`.
    let persistent_rewrite_bytes = perm
        .capacity()
        .saturating_mul(std::mem::size_of::<u32>())
        .saturating_add(real_to_virtual.iter().fold(0usize, |total, map| {
            total.saturating_add(map.capacity().saturating_mul(std::mem::size_of::<u32>()))
        }))
        .saturating_add(
            block_data_starts
                .capacity()
                .saturating_mul(std::mem::size_of::<u64>()),
        );
    let rewrite_workspace = memory_budget.saturating_sub(persistent_rewrite_bytes);
    let grid_entries_budget =
        (rewrite_workspace.saturating_mul(2) / 3).clamp(1024 * 1024, 512 * 1024 * 1024);
    let encode_window_budget = rewrite_workspace
        .saturating_sub(grid_entries_budget)
        .clamp(1024 * 1024, 256 * 1024 * 1024);

    // Grid entries with external merge sort: accumulate in memory up to budget,
    // spill sorted runs to temp files when exceeded. 12 bytes per entry in memory.
    const GRID_ENTRY_MEM_SIZE: usize = std::mem::size_of::<(u32, u32, u8)>(); // 12 bytes
    let max_entries_in_memory = grid_entries_budget / GRID_ENTRY_MEM_SIZE;

    let total_source_terms = bmp_refs.iter().fold(0usize, |total, bmp| {
        total.saturating_add(bmp.total_terms() as usize)
    });
    let est_entries = total_source_terms.min(max_entries_in_memory);
    let mut grid_entries: Vec<(u32, u32, u8)> = Vec::with_capacity(est_entries);
    let mut run_files = GridRunFiles::new(field_id);

    // Encode one output block: gather its records from source blocks and
    // serialize the V14 block layout. Pure function of `perm` + read-only
    // sources — output blocks encode independently.
    //
    // Global real ids resolve to a source via `real_base`, then to a
    // virtual vid (block/slot position) via that source's real→virtual map.
    // Old block/slot arithmetic uses each source's own stored block size
    // (uniform with the output in practice — merge validates it — but the
    // source footer is the ground truth for parsing source blocks).
    // (serialized block bytes, its (dim, out_block, max_impact) grid entries)
    type EncodedBlock = (Vec<u8>, Vec<(u32, u32, u8)>);
    let encode_block = |out_block: usize| -> Result<EncodedBlock> {
        let new_vid_start = out_block * effective_block_size;
        let new_vid_end = ((out_block + 1) * effective_block_size).min(num_real_docs);
        let slots_count = new_vid_end - new_vid_start;

        // Group records by (source, source block), scatter matching slots
        // All 256 u8 values are valid slots when bmp_block_size=256. The old
        // u8::MAX "unmapped" sentinel therefore dropped a posting whenever
        // source slot 255 mapped to output slot 255.
        let mut slot_map = [u16::MAX; 256];
        let mut block_mappings: rustc_hash::FxHashMap<(usize, usize), Vec<(u8, u8)>> =
            rustc_hash::FxHashMap::default();
        for new_local_slot in 0..slots_count {
            let global_real = perm[new_vid_start + new_local_slot] as usize;
            let src = real_base.partition_point(|&b| b <= global_real) - 1;
            let old_vid = real_to_virtual[src][global_real - real_base[src]] as usize;
            let src_block_size = sources[src].0.bmp_block_size.max(1) as usize;
            let old_block = old_vid / src_block_size;
            let old_slot = (old_vid % src_block_size) as u8;
            block_mappings
                .entry((src, old_block))
                .or_default()
                .push((old_slot, new_local_slot as u8));
        }

        let mut dim_postings: rustc_hash::FxHashMap<u32, Vec<(u8, u8)>> =
            rustc_hash::FxHashMap::default();
        for (&(src, old_block), mappings) in &block_mappings {
            for &(old_s, new_s) in mappings {
                slot_map[old_s as usize] = u16::from(new_s);
            }

            for (dim_id, postings) in sources[src].0.iter_block_terms(old_block as u32) {
                for p in postings {
                    let new_slot = slot_map[p.local_slot as usize];
                    if new_slot != u16::MAX {
                        dim_postings
                            .entry(dim_id)
                            .or_default()
                            .push((new_slot as u8, p.impact));
                    }
                }
            }

            for &(old_s, _) in mappings {
                slot_map[old_s as usize] = u16::MAX;
            }
        }

        if dim_postings.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut sorted_dims: Vec<u32> = dim_postings.keys().copied().collect();
        sorted_dims.sort_unstable();
        let nt = sorted_dims.len();

        let mut blk_buf: Vec<u8> = Vec::with_capacity(4 + nt * 8 + 4);
        let nt_u32 = u32::try_from(nt)
            .map_err(|_| crate::Error::Internal("BMP block term count exceeds u32::MAX".into()))?;
        blk_buf.extend_from_slice(&nt_u32.to_le_bytes());

        for &dim_id in &sorted_dims {
            blk_buf.extend_from_slice(&dim_id.to_le_bytes());
        }

        // u32 prefix sums (V14): u16 wrapped past 65,535 postings per block
        let mut cum: u32 = 0;
        for &dim_id in &sorted_dims {
            blk_buf.extend_from_slice(&cum.to_le_bytes());
            let posting_count = u32::try_from(dim_postings[&dim_id].len()).map_err(|_| {
                crate::Error::Internal("BMP block/dimension posting count exceeds u32::MAX".into())
            })?;
            cum = cum.checked_add(posting_count).ok_or_else(|| {
                crate::Error::Internal("BMP block posting prefix exceeds u32::MAX".into())
            })?;
        }
        blk_buf.extend_from_slice(&cum.to_le_bytes());

        let mut grid: Vec<(u32, u32, u8)> = Vec::with_capacity(nt);
        for &dim_id in &sorted_dims {
            let posts = &dim_postings[&dim_id];
            let mut max_impact: u8 = 0;
            for &(slot, impact) in posts {
                blk_buf.push(slot);
                blk_buf.push(impact);
                max_impact = max_impact.max(impact);
            }
            grid.push((dim_id, out_block as u32, max_impact));
        }
        Ok((blk_buf, grid))
    };

    // Encode blocks in parallel per bounded window (blob order is fixed, so
    // the write itself stays serial; the window caps buffered bytes).
    let source_block_bytes = bmp_refs
        .iter()
        .map(|bmp| bmp.block_data_slice().len())
        .fold(0usize, usize::saturating_add);
    let source_blocks = bmp_refs
        .iter()
        .map(|bmp| bmp.num_blocks as usize)
        .fold(0usize, usize::saturating_add);
    // Encoded bytes and grid tuples coexist. Twice the average source block
    // size is a conservative estimate; the byte budget prevents a pathological
    // high-dimensional corpus from buffering several GB merely because the
    // old fixed window admitted 4096 blocks.
    let estimated_bytes_per_block = source_block_bytes
        .checked_div(source_blocks.max(1))
        .unwrap_or(0)
        .saturating_mul(2)
        .max(1);
    // More queued blocks than workers cannot improve parallelism, but their
    // variable-size buffers all remain live until the serial write phase.
    // Two waves keep workers fed while imposing a hard small-count ceiling on
    // estimation error from heavily skewed blocks.
    let parallel_width = rayon_pool
        .as_ref()
        .map(|pool| pool.current_num_threads())
        .unwrap_or_else(rayon::current_num_threads)
        .max(1);
    let max_parallel_window = parallel_width.saturating_mul(2);
    let encode_window =
        (encode_window_budget / estimated_bytes_per_block).clamp(1, max_parallel_window);
    let mut encoded: Vec<Result<EncodedBlock>> = Vec::new();
    for window_start in (0..new_num_blocks).step_by(encode_window) {
        let window_end = (window_start + encode_window).min(new_num_blocks);
        encoded.clear();
        {
            use rayon::prelude::*;
            let run = |out: &mut Vec<Result<EncodedBlock>>| {
                (window_start..window_end)
                    .into_par_iter()
                    .map(encode_block)
                    .collect_into_vec(out);
            };
            if let Some(ref pool) = rayon_pool {
                pool.install(|| run(&mut encoded));
            } else {
                run(&mut encoded);
            }
        }

        for encoded_block in encoded.drain(..) {
            let (blk_buf, grid) = encoded_block?;
            block_data_starts.push(cumulative_bytes);
            if blk_buf.is_empty() {
                continue;
            }
            total_terms += grid.len() as u64;
            // layout: 4 (nt) + nt*4 (dims) + (nt+1)*4 (prefix) + postings*2
            total_postings += ((blk_buf.len() - 8 - grid.len() * 8) / 2) as u64;
            grid_entries.extend_from_slice(&grid);
            writer.write_all(&blk_buf).map_err(crate::Error::Io)?;
            cumulative_bytes += blk_buf.len() as u64;
        }

        // Spill grid entries to disk when memory budget exceeded
        if grid_entries.len() >= max_entries_in_memory {
            grid_entries.sort_unstable();
            let run_path = run_files.allocate();
            write_grid_run(&grid_entries, &run_path).map_err(crate::Error::Io)?;
            grid_entries.clear();
            log::debug!(
                "[reorder_bmp] field {}: spilled grid run {} to disk",
                field_id,
                run_files.len(),
            );
        }
    }
    drop(encoded);

    // Sentinel
    block_data_starts.push(cumulative_bytes);

    if total_terms == 0 {
        return Ok((writer, field_tocs, converged));
    }

    // Sort remaining in-memory entries
    grid_entries.sort_unstable();

    // ── Write remaining sections ────────────────────────────────────────

    let block_data_len = writer.offset() - blob_start;
    let padding = (8 - (block_data_len % 8) as usize) % 8;
    if padding > 0 {
        writer
            .write_all(&[0u8; 8][..padding])
            .map_err(crate::Error::Io)?;
    }

    // Section A: block_data_starts
    for &val in &block_data_starts {
        writer
            .write_all(&val.to_le_bytes())
            .map_err(crate::Error::Io)?;
    }
    drop(block_data_starts);

    // Sections D+E: packed grid + sb_grid
    let grid_offset = writer.offset() - blob_start;
    let (packed_bytes, _sb_bytes) = if run_files.is_empty() {
        // Fast path: all entries fit in memory, use existing function
        let result = stream_write_grids(
            &grid_entries,
            dims as usize,
            new_num_blocks,
            grid_bits,
            &mut writer,
        )
        .map_err(crate::Error::Io)?;
        drop(grid_entries);
        result
    } else {
        // Flush remaining in-memory entries as the final run if non-empty
        if !grid_entries.is_empty() {
            let run_path = run_files.allocate();
            write_grid_run(&grid_entries, &run_path).map_err(crate::Error::Io)?;
        }
        drop(grid_entries);

        // Open run readers for K-way merge
        let mut run_readers: Vec<GridRunReader> = Vec::with_capacity(run_files.len());
        for path in run_files.iter() {
            run_readers.push(GridRunReader::open(path).map_err(crate::Error::Io)?);
        }

        let result = stream_write_grids_merged(
            &mut run_readers,
            dims as usize,
            new_num_blocks,
            grid_bits,
            &mut writer,
        )
        .map_err(crate::Error::Io)?;

        // Readers must close before the RAII owner removes the run files.
        drop(run_readers);

        result
    };
    let sb_grid_offset = grid_offset + packed_bytes;

    // Sections F+G: doc_map [new_num_virtual_docs each]
    let doc_map_offset = writer.offset() - blob_start;

    // Resolve a global real id to (source, virtual vid within source).
    let resolve = |global_real: usize| -> (usize, usize) {
        let src = real_base.partition_point(|&b| b <= global_real) - 1;
        (
            src,
            real_to_virtual[src][global_real - real_base[src]] as usize,
        )
    };

    // Section F: doc_map_ids [u32-LE × new_num_virtual_docs], doc ids patched
    // with each source's offset in the output segment. Real slots are never
    // the u32::MAX sentinel (realness came from the doc map itself).
    for &global_real in perm.iter().take(num_real_docs) {
        let (src, vid) = resolve(global_real as usize);
        let ids = sources[src].0.doc_map_ids_slice();
        let off = vid * 4;
        let source_doc_id = u32::from_le_bytes(ids[off..off + 4].try_into().unwrap());
        let doc_id = source_doc_id.checked_add(sources[src].1).ok_or_else(|| {
            crate::Error::Corruption(format!(
                "BMP doc-id offset overflow: {} + {}",
                source_doc_id, sources[src].1
            ))
        })?;
        writer
            .write_all(&doc_id.to_le_bytes())
            .map_err(crate::Error::Io)?;
    }
    for _ in num_real_docs..new_num_virtual_docs {
        writer
            .write_all(&u32::MAX.to_le_bytes())
            .map_err(crate::Error::Io)?;
    }

    // Section G: doc_map_ordinals [u16-LE × new_num_virtual_docs]
    for &global_real in perm.iter().take(num_real_docs) {
        let (src, vid) = resolve(global_real as usize);
        let ords = sources[src].0.doc_map_ordinals_slice();
        let off = vid * 2;
        let ordinal = u16::from_le_bytes(ords[off..off + 2].try_into().unwrap());
        writer
            .write_all(&ordinal.to_le_bytes())
            .map_err(crate::Error::Io)?;
    }
    for _ in num_real_docs..new_num_virtual_docs {
        writer
            .write_all(&0u16.to_le_bytes())
            .map_err(crate::Error::Io)?;
    }

    // V14 footer (64 bytes)
    write_bmp_footer(
        &mut writer,
        total_terms,
        total_postings,
        grid_offset,
        sb_grid_offset,
        new_num_blocks as u32,
        dims,
        effective_block_size as u32,
        new_num_virtual_docs as u32,
        max_weight_scale,
        doc_map_offset,
        num_real_docs as u32,
        grid_bits,
    )
    .map_err(crate::Error::Io)?;

    let blob_len = writer.offset() - blob_start;

    // Push BMP sentinel TOC entry
    let mut config_for_byte =
        crate::structures::SparseVectorConfig::from_byte(quantization as u8).unwrap_or_default();
    config_for_byte.format = SparseFormat::Bmp;
    config_for_byte.weight_quantization = quantization;

    field_tocs.push(SparseFieldToc {
        field_id,
        quantization: config_for_byte.to_byte(),
        total_vectors,
        dims: vec![crate::segment::format::SparseDimTocEntry {
            dim_id: 0xFFFFFFFF, // sentinel for BMP
            block_data_offset: blob_start,
            skip_start: (blob_len & 0xFFFFFFFF) as u32,
            num_blocks: ((blob_len >> 32) & 0xFFFFFFFF) as u32,
            doc_count: 0,
            max_weight: 0.0,
        }],
    });

    log::info!(
        "[reorder_bmp] field {}: done — {} blocks, {} terms, {} postings, {:.2} MB",
        field_id,
        new_num_blocks,
        total_terms,
        total_postings,
        blob_len as f64 / (1024.0 * 1024.0),
    );

    Ok((writer, field_tocs, converged))
}

#[cfg(test)]
mod grid_run_prefix_tests {
    use crate::directories::{Directory, DirectoryWriter};

    #[tokio::test]
    async fn segment_copy_distinguishes_optional_and_required_missing_files() {
        let dir = crate::directories::RamDirectory::new();
        let missing = std::path::Path::new("missing");
        let output = std::path::Path::new("output");

        super::copy_segment_file(&dir, missing, output, false)
            .await
            .unwrap();
        assert!(!dir.exists(output).await.unwrap());

        let error = super::copy_segment_file(&dir, missing, output, true)
            .await
            .unwrap_err();
        assert!(matches!(error, crate::Error::Corruption(_)));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn segment_copy_preserves_all_bytes() {
        let dir = crate::directories::RamDirectory::new();
        let source = std::path::Path::new("source");
        let output = std::path::Path::new("output");
        let data = vec![0x5a; 4 * 1024 * 1024 + 17];
        dir.write(source, &data).await.unwrap();

        super::copy_segment_file(&dir, source, output, true)
            .await
            .unwrap();

        let copied = dir
            .open_read(output)
            .await
            .unwrap()
            .read_bytes()
            .await
            .unwrap();
        assert_eq!(copied.as_slice(), data);
    }

    /// Regression: run prefixes were `pid_field` — identical for every pass
    /// in a container (PID 1) reordering the same field id, so concurrent
    /// passes clobbered and deleted each other's spill files (prod ENOENT).
    #[test]
    fn test_grid_run_prefix_unique_per_pass() {
        let a = super::grid_run_prefix(3);
        let b = super::grid_run_prefix(3);
        assert_ne!(
            a, b,
            "two passes over the same field must not share spill files"
        );
    }

    #[test]
    fn grid_run_files_remove_partial_spills_on_drop() {
        let path = {
            let mut runs = super::GridRunFiles::new(7);
            let path = runs.allocate();
            std::fs::write(&path, b"partial run").unwrap();
            assert!(path.exists());
            path
        };
        assert!(
            !path.exists(),
            "spill owner must remove files on unwind/drop"
        );
    }
}
