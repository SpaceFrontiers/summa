//! Standalone segment reorder via Recursive Graph Bisection (BP).
//!
//! Copies unchanged segment files (postings, store, fast, dense vectors)
//! and rebuilds the sparse file with reordered BMP blocks. Non-BMP sparse
//! fields (MaxScore) are identity-copied.
//!
//! This module is decoupled from the merge path: merge never reorders,
//! and reorder never merges multiple segments.

use std::io::Write;
use std::path::Path;
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
/// (~4 B/posting + ~28 B/doc). Sized from prod evidence: a 58M-doc /
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
    static GRID_RUN_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = GRID_RUN_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!(
        "hermes_grid_run_{}_{}_{}",
        std::process::id(),
        field_id,
        seq
    )
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
    let mut df: rustc_hash::FxHashMap<u32, (u32, u32)> = rustc_hash::FxHashMap::default();
    let mut scanned_blocks = 0usize;
    let mut global_block = 0usize;
    for (bmp, _) in sources {
        for block_id in 0..bmp.num_blocks {
            if global_block.is_multiple_of(stride) {
                scanned_blocks += 1;
                for (dim_id, posts) in bmp.iter_block_terms(block_id) {
                    let e = df.entry(dim_id).or_insert((0, 0));
                    e.0 += posts.len() as u32;
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
        postings += records as u64;
        terms += blocks as u64;
        expected_rand_pairs += b * (1.0 - keep.powf(records as f64));
        min_pairs += (records as u64).div_ceil(block_size as u64);
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
    for (src, dst) in [
        (&src_files.term_dict, &dst_files.term_dict),
        (&src_files.postings, &dst_files.postings),
        (&src_files.positions, &dst_files.positions),
        (&src_files.store, &dst_files.store),
        (&src_files.fast, &dst_files.fast),
        (&src_files.vectors, &dst_files.vectors),
    ] {
        copy_segment_file(dir, src, dst).await?;
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
    dir.write(&dst_files.meta, &meta.serialize()?).await?;

    Ok((output_id.to_hex(), num_docs, bp_converged))
}

/// Copy a segment file via streaming I/O (4 MB chunks).
///
/// No-op if the source file does not exist.
/// Empty files are still created (SegmentReader requires their existence).
async fn copy_segment_file<D: Directory + DirectoryWriter>(
    dir: &D,
    src: &Path,
    dst: &Path,
) -> Result<()> {
    let handle = match dir.open_read(src).await {
        Ok(h) => h,
        Err(_) => return Ok(()), // file doesn't exist
    };
    let data = handle.read_bytes().await.map_err(crate::Error::Io)?;
    let slice = data.as_slice();

    let mut writer = dir
        .streaming_writer_cold(dst)
        .await
        .map_err(crate::Error::Io)?;
    const CHUNK: usize = 4 * 1024 * 1024;
    for (i, chunk) in slice.chunks(CHUNK).enumerate() {
        writer.write_all(chunk).map_err(crate::Error::Io)?;
        // Drop source pages behind the copy cursor — a whole-file copy must
        // not fault the entire source into the page cache and leave it
        // resident (see docs/cold-io.md).
        #[cfg(feature = "native")]
        data.madvise_range(i * CHUNK..i * CHUNK + chunk.len(), libc::MADV_DONTNEED);
    }
    writer.finish().map_err(crate::Error::Io)?;

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

    if !has_bmp_data {
        // No BMP field wants reordering — just copy the sparse file as-is
        log::info!(
            "[reorder] segment {:x}: no BMP field has the `reorder` schema attribute — sparse file copied unchanged",
            reader.meta().id,
        );
        let src_files = SegmentFiles::new(reader.meta().id);
        copy_segment_file(dir, &src_files.sparse, &dst_files.sparse).await?;
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
                    .min(256) as usize;
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
    let num_blocks_total: usize = bmp_refs.iter().map(|b| b.num_blocks as usize).sum();
    if num_blocks_total == 0 {
        return Ok((writer, field_tocs, true));
    }

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
    let block_budget = crate::segment::BpBudget {
        min_partition_docs: bp_budget
            .min_partition_docs
            .map(|docs| (docs / effective_block_size).max(1)),
        time_budget: bp_budget.time_budget,
    };
    // Forward-index build is parallel too — keep it on the bounded pool.
    let run_bp = || {
        let fwd = build_forward_index_from_blocks(&bmp_refs, memory_budget);
        if fwd.num_terms > 0 && num_blocks_total > sb {
            graph_bisection(&fwd, sb, 20, block_budget)
        } else {
            ((0..num_blocks_total as u32).collect(), true)
        }
    };
    let (perm, converged) = if let Some(ref pool) = rayon_pool {
        pool.install(run_bp)
    } else {
        run_bp()
    };
    log::info!(
        "[reorder_bmp] field {}: blockwise BP over {} blocks in {:.1}ms (converged={})",
        field_id,
        num_blocks_total,
        bp_start.elapsed().as_secs_f64() * 1000.0,
        converged,
    );

    // ── Write blob: permuted block copy ─────────────────────────────────
    let blob_start = writer.offset();

    // Section B: block data verbatim, in permuted order
    let mut block_data_starts: Vec<u64> = Vec::with_capacity(num_blocks_total + 1);
    let mut cumulative: u64 = 0;
    let mut total_terms: u64 = 0;
    let mut total_postings: u64 = 0;
    for b in &bmp_refs {
        total_terms += b.total_terms();
        total_postings += b.total_postings();
    }
    for &new_pos in perm.iter() {
        let (src, lb) = resolve(new_pos as usize);
        let bmp = bmp_refs[src];
        let start = bmp.block_data_start(lb as u32) as usize;
        let end = if (lb as u32) + 1 < bmp.num_blocks {
            bmp.block_data_start(lb as u32 + 1) as usize
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
    let mut sb_rows: Vec<Vec<u8>> = vec![vec![0u8; num_superblocks]; dims as usize];
    let dequant = crate::segment::builder::bmp::grid_dequant_scale(grid_bits);

    for (dim, sb_row) in sb_rows.iter_mut().enumerate() {
        out_row.fill(0);
        for (new_pos, &old_global) in perm.iter().enumerate() {
            let (src, lb) = resolve(old_global as usize);
            let bmp = bmp_refs[src];
            let prs = bmp.packed_row_size();
            let row = &bmp.grid_slice()[dim * prs..dim * prs + prs];
            let cell = crate::segment::builder::bmp::grid_get_cell(row, lb, grid_bits);
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
    }
    drop(out_row);

    let sb_grid_offset = writer.offset() - blob_start;
    for sb_row in &sb_rows {
        writer.write_all(sb_row).map_err(crate::Error::Io)?;
    }
    drop(sb_rows);

    // Sections F + G: doc maps copied per block chunk, ids offset-patched
    let doc_map_offset = writer.offset() - blob_start;
    let bs = effective_block_size;
    let mut num_real_docs: u32 = 0;
    for b in &bmp_refs {
        num_real_docs += b.num_real_docs();
    }
    let mut id_chunk = vec![0u8; bs * 4];
    for &old_global in perm.iter() {
        let (src, lb) = resolve(old_global as usize);
        let (bmp, doc_offset) = (&sources[src].0, sources[src].1);
        let ids = bmp.doc_map_ids_slice();
        id_chunk.copy_from_slice(&ids[lb * bs * 4..(lb + 1) * bs * 4]);
        if doc_offset != 0 {
            let (chunks, _) = id_chunk.as_chunks_mut::<4>();
            for e in chunks {
                let doc_id = u32::from_le_bytes(*e);
                if doc_id != u32::MAX {
                    *e = (doc_id + doc_offset).to_le_bytes();
                }
            }
        }
        writer.write_all(&id_chunk).map_err(crate::Error::Io)?;
    }
    for &old_global in perm.iter() {
        let (src, lb) = resolve(old_global as usize);
        let ords = bmp_refs[src].doc_map_ordinals_slice();
        writer
            .write_all(&ords[lb * bs * 2..(lb + 1) * bs * 2])
            .map_err(crate::Error::Io)?;
    }

    // Footer
    write_bmp_footer(
        &mut writer,
        total_terms as u32,
        total_postings as u32,
        grid_offset,
        sb_grid_offset,
        num_blocks_total as u32,
        dims,
        effective_block_size as u32,
        (num_blocks_total * bs) as u32,
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
    use crate::segment::builder::bmp::{
        GridRunReader, stream_write_grids, stream_write_grids_merged, write_bmp_footer,
        write_grid_run,
    };
    use crate::segment::builder::graph_bisection::{
        build_forward_index_from_bmps, build_vid_maps, graph_bisection,
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

    // ── Phase 1: Build forward index and run BP ─────────────────────────
    let bp_start = std::time::Instant::now();

    let bmp_refs: Vec<&crate::segment::BmpIndex> = sources.iter().map(|(b, _)| b).collect();
    // Real→virtual maps per source (padding-aware; doc map is ground truth).
    let real_to_virtual: Vec<Vec<u32>> = bmp_refs.iter().map(|b| build_vid_maps(b).1).collect();
    // Prefix sums of real doc counts: source s owns global real ids
    // real_base[s]..real_base[s+1].
    let real_base: Vec<usize> = std::iter::once(0)
        .chain(real_to_virtual.iter().scan(0usize, |acc, r2v| {
            *acc += r2v.len();
            Some(*acc)
        }))
        .collect();
    let num_real_docs = *real_base.last().unwrap();
    if num_real_docs == 0 {
        return Ok((writer, field_tocs, true));
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

        // Forward-index build is parallel too — run the whole phase on the
        // bounded rayon pool if provided, keeping background CPU off the
        // global (query) pool.
        let run_bp = || {
            let (fwd, _source_doc_counts) = build_forward_index_from_bmps(
                &bmp_refs,
                min_doc_freq,
                max_doc_freq.max(1),
                memory_budget,
            );

            log::info!(
                "[reorder_bmp] field {}: forward index built in {:.1}ms ({} terms, {} postings)",
                field_id,
                bp_start.elapsed().as_secs_f64() * 1000.0,
                fwd.num_terms,
                fwd.total_postings(),
            );

            if fwd.num_terms > 0 && num_real_docs > effective_block_size {
                let bp_start = std::time::Instant::now();
                let (perm, converged) = graph_bisection(&fwd, effective_block_size, 20, bp_budget);
                log::info!(
                    "[reorder_bmp] field {}: BP completed in {:.1}ms (converged={})",
                    field_id,
                    bp_start.elapsed().as_secs_f64() * 1000.0,
                    converged,
                );
                (perm, converged)
            } else {
                ((0..num_real_docs as u32).collect(), true)
            }
        };
        if let Some(ref pool) = rayon_pool {
            pool.install(run_bp)
        } else {
            run_bp()
        }
    };

    log::info!(
        "[reorder_bmp] field {}: writing reordered blob ({} blocks)",
        field_id,
        num_real_docs.div_ceil(effective_block_size),
    );

    // ── Phase 2: Write new blob with records in permuted order ──────────
    let new_num_blocks = num_real_docs.div_ceil(effective_block_size);
    let new_num_virtual_docs = new_num_blocks * effective_block_size;

    let blob_start = writer.offset();
    let mut block_data_starts: Vec<u64> = Vec::with_capacity(new_num_blocks + 1);
    let mut total_terms: u32 = 0;
    let mut total_postings: u32 = 0;
    let mut cumulative_bytes: u64 = 0;

    // Grid entries with external merge sort: accumulate in memory up to budget,
    // spill sorted runs to temp files when exceeded. 12 bytes per entry in memory.
    const GRID_ENTRIES_BUDGET: usize = 512 * 1024 * 1024; // 512 MB
    const GRID_ENTRY_MEM_SIZE: usize = std::mem::size_of::<(u32, u32, u8)>(); // 12 bytes
    let max_entries_in_memory = GRID_ENTRIES_BUDGET / GRID_ENTRY_MEM_SIZE;

    let total_source_terms: usize = bmp_refs.iter().map(|b| b.total_terms() as usize).sum();
    let est_entries = total_source_terms.min(max_entries_in_memory);
    let mut grid_entries: Vec<(u32, u32, u8)> = Vec::with_capacity(est_entries);
    let mut run_files: Vec<std::path::PathBuf> = Vec::new();
    let run_prefix = grid_run_prefix(field_id);

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
    let encode_block = |out_block: usize| -> EncodedBlock {
        let new_vid_start = out_block * effective_block_size;
        let new_vid_end = ((out_block + 1) * effective_block_size).min(num_real_docs);
        let slots_count = new_vid_end - new_vid_start;

        // Group records by (source, source block), scatter matching slots
        let mut slot_map = [u8::MAX; 256];
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
                slot_map[old_s as usize] = new_s;
            }

            for (dim_id, postings) in sources[src].0.iter_block_terms(old_block as u32) {
                for p in postings {
                    let new_slot = slot_map[p.local_slot as usize];
                    if new_slot != u8::MAX {
                        dim_postings
                            .entry(dim_id)
                            .or_default()
                            .push((new_slot, p.impact));
                    }
                }
            }

            for &(old_s, _) in mappings {
                slot_map[old_s as usize] = u8::MAX;
            }
        }

        if dim_postings.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let mut sorted_dims: Vec<u32> = dim_postings.keys().copied().collect();
        sorted_dims.sort_unstable();
        let nt = sorted_dims.len();

        let mut blk_buf: Vec<u8> = Vec::with_capacity(4 + nt * 8 + 4);
        blk_buf.extend_from_slice(&(nt as u32).to_le_bytes());

        for &dim_id in &sorted_dims {
            blk_buf.extend_from_slice(&dim_id.to_le_bytes());
        }

        // u32 prefix sums (V14): u16 wrapped past 65,535 postings per block
        let mut cum: u32 = 0;
        for &dim_id in &sorted_dims {
            blk_buf.extend_from_slice(&cum.to_le_bytes());
            cum += dim_postings[&dim_id].len() as u32;
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
        (blk_buf, grid)
    };

    // Encode blocks in parallel per bounded window (blob order is fixed, so
    // the write itself stays serial; the window caps buffered bytes).
    const ENCODE_WINDOW: usize = 4096;
    let mut encoded: Vec<EncodedBlock> = Vec::new();
    for window_start in (0..new_num_blocks).step_by(ENCODE_WINDOW) {
        let window_end = (window_start + ENCODE_WINDOW).min(new_num_blocks);
        encoded.clear();
        {
            use rayon::prelude::*;
            let run = |out: &mut Vec<EncodedBlock>| {
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

        for (blk_buf, grid) in &encoded {
            block_data_starts.push(cumulative_bytes);
            if blk_buf.is_empty() {
                continue;
            }
            total_terms += grid.len() as u32;
            // layout: 4 (nt) + nt*4 (dims) + (nt+1)*4 (prefix) + postings*2
            total_postings += ((blk_buf.len() - 8 - grid.len() * 8) / 2) as u32;
            grid_entries.extend_from_slice(grid);
            writer.write_all(blk_buf).map_err(crate::Error::Io)?;
            cumulative_bytes += blk_buf.len() as u64;
        }

        // Spill grid entries to disk when memory budget exceeded
        if grid_entries.len() >= max_entries_in_memory {
            grid_entries.sort_unstable();
            let run_path =
                std::env::temp_dir().join(format!("{}_{}.tmp", run_prefix, run_files.len()));
            write_grid_run(&grid_entries, &run_path).map_err(crate::Error::Io)?;
            run_files.push(run_path);
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
        // Clean up any run files
        for path in &run_files {
            let _ = std::fs::remove_file(path);
        }
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
            let run_path =
                std::env::temp_dir().join(format!("{}_{}.tmp", run_prefix, run_files.len()));
            write_grid_run(&grid_entries, &run_path).map_err(crate::Error::Io)?;
            run_files.push(run_path);
        }
        drop(grid_entries);

        // Open run readers for K-way merge
        let mut run_readers: Vec<GridRunReader> = Vec::with_capacity(run_files.len());
        for path in &run_files {
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

        // Clean up run files
        drop(run_readers);
        for path in &run_files {
            let _ = std::fs::remove_file(path);
        }

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
        let doc_id = u32::from_le_bytes(ids[off..off + 4].try_into().unwrap()) + sources[src].1;
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
}
