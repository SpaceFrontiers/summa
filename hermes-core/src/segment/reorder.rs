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

/// Default memory budget for forward index during BP (2 GB).
pub const DEFAULT_MEMORY_BUDGET: usize = 2 * 1024 * 1024 * 1024;

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
#[allow(clippy::too_many_arguments)]
pub async fn reorder_segment<D: Directory + DirectoryWriter>(
    dir: &D,
    schema: &Arc<Schema>,
    source_id: SegmentId,
    output_id: SegmentId,
    term_cache_blocks: usize,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
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
async fn reorder_sparse_file<D: Directory + DirectoryWriter>(
    dir: &D,
    reader: &SegmentReader,
    dst_files: &SegmentFiles,
    schema: &Schema,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<bool> {
    let sparse_fields: Vec<_> = schema
        .fields()
        .filter(|(_, entry)| matches!(entry.field_type, FieldType::SparseVector))
        .map(|(field, entry)| (field, entry.sparse_vector_config.clone(), entry.reorder))
        .collect();

    if sparse_fields.is_empty() {
        return Ok(true);
    }

    // Check if there's any BMP data to reorder. Per-field gate: only fields
    // with the `reorder` schema attribute get BP; others are copied unchanged.
    let has_bmp_data = sparse_fields.iter().any(|(field, config, reorder)| {
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

    for (field, sparse_config, reorder) in &sparse_fields {
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
                let pool = rayon_pool.clone();
                let (w, ft, converged) = tokio::task::spawn_blocking(move || {
                    reorder_bmp_field(
                        &bmp_sources,
                        fid,
                        quantization,
                        dims,
                        effective_block_size,
                        max_weight_scale,
                        total_vectors,
                        memory_budget,
                        bp_budget,
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
    quantization: crate::structures::WeightQuantization,
    dims: u32,
    effective_block_size: usize,
    max_weight_scale: f32,
    total_vectors: u32,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    mut writer: OffsetWriter,
    mut field_tocs: Vec<SparseFieldToc>,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<(OffsetWriter, Vec<SparseFieldToc>, bool)> {
    use crate::segment::builder::bmp::{
        GridRunReader, stream_write_grids, stream_write_grids_merged, write_grid_run,
        write_v13_footer,
    };
    use crate::segment::builder::graph_bisection::{
        build_forward_index_from_bmps, build_vid_maps, graph_bisection,
    };

    if sources.is_empty() {
        return Ok((writer, field_tocs, true));
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

    let max_doc_freq = ((num_real_docs as f64) * 0.9) as usize;
    let min_doc_freq = 128.min(num_real_docs);

    let (fwd, _source_doc_counts) =
        build_forward_index_from_bmps(&bmp_refs, min_doc_freq, max_doc_freq.max(1), memory_budget);

    log::info!(
        "[reorder_bmp] field {}: forward index built in {:.1}ms ({} terms, {} postings)",
        field_id,
        bp_start.elapsed().as_secs_f64() * 1000.0,
        fwd.num_terms,
        fwd.total_postings(),
    );

    let (perm, converged) = if fwd.num_terms > 0 && num_real_docs > effective_block_size {
        let bp_start = std::time::Instant::now();
        // Run BP on the bounded rayon pool if provided, otherwise global pool.
        let (perm, converged) = if let Some(ref pool) = rayon_pool {
            pool.install(|| graph_bisection(&fwd, effective_block_size, 20, bp_budget))
        } else {
            graph_bisection(&fwd, effective_block_size, 20, bp_budget)
        };
        log::info!(
            "[reorder_bmp] field {}: BP completed in {:.1}ms (converged={})",
            field_id,
            bp_start.elapsed().as_secs_f64() * 1000.0,
            converged,
        );
        (perm, converged)
    } else {
        drop(fwd);
        ((0..num_real_docs as u32).collect(), true)
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
    let run_prefix = format!("hermes_grid_run_{}_{}", std::process::id(), field_id);

    // Per-block scratch buffers
    let mut blk_buf: Vec<u8> = Vec::with_capacity(4096);
    let mut dim_postings: rustc_hash::FxHashMap<u32, Vec<(u8, u8)>> =
        rustc_hash::FxHashMap::default();

    for out_block in 0..new_num_blocks {
        block_data_starts.push(cumulative_bytes);

        let new_vid_start = out_block * effective_block_size;
        let new_vid_end = ((out_block + 1) * effective_block_size).min(num_real_docs);
        let slots_count = new_vid_end - new_vid_start;

        dim_postings.clear();

        // Group records by (source, source block), scatter matching slots
        let mut slot_map = [u8::MAX; 256];

        // Collect which (source, old_block)s we need and the slot mappings.
        // Global real ids resolve to a source via `real_base`, then to a
        // virtual vid (block/slot position) via that source's real→virtual map.
        let mut block_mappings: rustc_hash::FxHashMap<(usize, usize), Vec<(u8, u8)>> =
            rustc_hash::FxHashMap::default();
        for new_local_slot in 0..slots_count {
            let global_real = perm[new_vid_start + new_local_slot] as usize;
            let src = real_base.partition_point(|&b| b <= global_real) - 1;
            let old_vid = real_to_virtual[src][global_real - real_base[src]] as usize;
            let old_block = old_vid / effective_block_size;
            let old_slot = (old_vid % effective_block_size) as u8;
            block_mappings
                .entry((src, old_block))
                .or_default()
                .push((old_slot, new_local_slot as u8));
        }

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

        // Write this block's data
        if !dim_postings.is_empty() {
            let mut sorted_dims: Vec<u32> = dim_postings.keys().copied().collect();
            sorted_dims.sort_unstable();

            blk_buf.clear();
            let nt = sorted_dims.len();

            blk_buf.extend_from_slice(&(nt as u16).to_le_bytes());

            for &dim_id in &sorted_dims {
                blk_buf.extend_from_slice(&dim_id.to_le_bytes());
            }

            let mut cum: u16 = 0;
            for &dim_id in &sorted_dims {
                blk_buf.extend_from_slice(&cum.to_le_bytes());
                cum += dim_postings[&dim_id].len() as u16;
            }
            blk_buf.extend_from_slice(&cum.to_le_bytes());

            for &dim_id in &sorted_dims {
                let posts = &dim_postings[&dim_id];
                let mut max_impact: u8 = 0;
                for &(slot, impact) in posts {
                    blk_buf.push(slot);
                    blk_buf.push(impact);
                    max_impact = max_impact.max(impact);
                }
                total_postings += posts.len() as u32;
                grid_entries.push((dim_id, out_block as u32, max_impact));
            }
            total_terms += nt as u32;

            writer.write_all(&blk_buf).map_err(crate::Error::Io)?;
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
        let result = stream_write_grids(&grid_entries, dims as usize, new_num_blocks, &mut writer)
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

        let result =
            stream_write_grids_merged(&mut run_readers, dims as usize, new_num_blocks, &mut writer)
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

    // V13 footer (64 bytes)
    write_v13_footer(
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
