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
/// Returns `(new_segment_hex_id, num_docs)`.
pub async fn reorder_segment<D: Directory + DirectoryWriter>(
    dir: &D,
    schema: &Arc<Schema>,
    source_id: SegmentId,
    output_id: SegmentId,
    term_cache_blocks: usize,
    memory_budget: usize,
) -> Result<(String, u32)> {
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
    reorder_sparse_file(dir, &reader, &dst_files, schema, memory_budget).await?;

    // Write new meta with output segment ID
    let src_meta = reader.meta();
    let meta = SegmentMeta {
        id: output_id.0,
        num_docs: src_meta.num_docs,
        field_stats: src_meta.field_stats.clone(),
    };
    dir.write(&dst_files.meta, &meta.serialize()?).await?;

    Ok((output_id.to_hex(), num_docs))
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

    let mut writer = dir.streaming_writer(dst).await.map_err(crate::Error::Io)?;
    const CHUNK: usize = 4 * 1024 * 1024;
    for chunk in slice.chunks(CHUNK) {
        writer.write_all(chunk).map_err(crate::Error::Io)?;
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
) -> Result<()> {
    let sparse_fields: Vec<_> = schema
        .fields()
        .filter(|(_, entry)| matches!(entry.field_type, FieldType::SparseVector))
        .map(|(field, entry)| (field, entry.sparse_vector_config.clone()))
        .collect();

    if sparse_fields.is_empty() {
        return Ok(());
    }

    // Check if there's any BMP data to reorder
    let has_bmp_data = sparse_fields.iter().any(|(field, config)| {
        config.as_ref().map(|c| c.format) == Some(SparseFormat::Bmp)
            && reader.bmp_indexes().get(&field.0).is_some()
    });

    if !has_bmp_data {
        // No BMP data — just copy the sparse file as-is
        let src_files = SegmentFiles::new(reader.meta().id);
        copy_segment_file(dir, &src_files.sparse, &dst_files.sparse).await?;
        return Ok(());
    }

    let mut writer = OffsetWriter::new(
        dir.streaming_writer(&dst_files.sparse)
            .await
            .map_err(crate::Error::Io)?,
    );
    let mut field_tocs: Vec<SparseFieldToc> = Vec::new();
    let mut all_skip_bytes: Vec<u8> = Vec::new();
    let mut skip_count: u32 = 0;

    for (field, sparse_config) in &sparse_fields {
        let format = sparse_config.as_ref().map(|c| c.format).unwrap_or_default();
        let quantization = sparse_config
            .as_ref()
            .map(|c| c.weight_quantization)
            .unwrap_or(crate::structures::WeightQuantization::Float32);

        if format == SparseFormat::Bmp {
            if let Some(bmp_idx) = reader.bmp_indexes().get(&field.0) {
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

                reorder_bmp_field(
                    bmp_idx,
                    field.0,
                    quantization,
                    dims,
                    effective_block_size,
                    max_weight_scale,
                    total_vectors,
                    memory_budget,
                    &mut writer,
                    &mut field_tocs,
                )?;
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
        return Ok(());
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
        "[reorder] sparse file written: {} fields, {} dims, {} skip entries",
        field_tocs.len(),
        total_dims,
        skip_count,
    );

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

/// Reorder a single BMP field via Recursive Graph Bisection (BP).
///
/// Reads block data from the source BmpIndex, builds a forward index,
/// runs BP to compute a permutation, then writes the reordered blob.
#[allow(clippy::too_many_arguments)]
fn reorder_bmp_field(
    bmp: &crate::segment::BmpIndex,
    field_id: u32,
    quantization: crate::structures::WeightQuantization,
    dims: u32,
    effective_block_size: usize,
    max_weight_scale: f32,
    total_vectors: u32,
    memory_budget: usize,
    writer: &mut OffsetWriter,
    field_tocs: &mut Vec<SparseFieldToc>,
) -> Result<()> {
    use crate::segment::builder::bmp::{stream_write_grids, write_v13_footer};
    use crate::segment::builder::graph_bisection::{
        build_forward_index_from_bmps, graph_bisection,
    };

    let num_real_docs = bmp.num_real_docs() as usize;
    if num_real_docs == 0 {
        return Ok(());
    }

    log::info!(
        "[reorder_bmp] field {}: running BP on {} real docs",
        field_id,
        num_real_docs,
    );

    // ── Phase 1: Build forward index and run BP ─────────────────────────
    let bp_start = std::time::Instant::now();
    let max_doc_freq = ((num_real_docs as f64) * 0.9) as usize;
    let min_doc_freq = 128.min(num_real_docs);

    let bmp_refs = [bmp];
    let (fwd, _source_doc_counts) =
        build_forward_index_from_bmps(&bmp_refs, min_doc_freq, max_doc_freq.max(1), memory_budget);

    log::info!(
        "[reorder_bmp] field {}: forward index built in {:.1}ms ({} terms, {} postings)",
        field_id,
        bp_start.elapsed().as_secs_f64() * 1000.0,
        fwd.num_terms,
        fwd.total_postings(),
    );

    let perm = if fwd.num_terms > 0 && num_real_docs > effective_block_size {
        let bp_start = std::time::Instant::now();
        let perm = graph_bisection(&fwd, effective_block_size, 20);
        log::info!(
            "[reorder_bmp] field {}: BP completed in {:.1}ms",
            field_id,
            bp_start.elapsed().as_secs_f64() * 1000.0,
        );
        perm
    } else {
        (0..num_real_docs as u32).collect()
    };
    drop(fwd);

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
    let est_terms: u32 = bmp.total_terms() as u32;
    let mut grid_entries: Vec<(u32, u32, u8)> = Vec::with_capacity(est_terms as usize);
    let mut total_terms: u32 = 0;
    let mut total_postings: u32 = 0;
    let mut cumulative_bytes: u64 = 0;

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

        // Group records by source block, scatter matching slots
        let mut slot_map = [u8::MAX; 256];

        // Collect which old_blocks we need and the slot mappings
        let mut block_mappings: rustc_hash::FxHashMap<usize, Vec<(u8, u8)>> =
            rustc_hash::FxHashMap::default();
        for new_local_slot in 0..slots_count {
            let old_vid = perm[new_vid_start + new_local_slot] as usize;
            let old_block = old_vid / effective_block_size;
            let old_slot = (old_vid % effective_block_size) as u8;
            block_mappings
                .entry(old_block)
                .or_default()
                .push((old_slot, new_local_slot as u8));
        }

        for (&old_block, mappings) in &block_mappings {
            for &(old_s, new_s) in mappings {
                slot_map[old_s as usize] = new_s;
            }

            for (dim_id, postings) in bmp.iter_block_terms(old_block as u32) {
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
    }

    // Sentinel
    block_data_starts.push(cumulative_bytes);

    if total_terms == 0 {
        return Ok(());
    }

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
    let (packed_bytes, _sb_bytes) =
        stream_write_grids(&grid_entries, dims as usize, new_num_blocks, writer)
            .map_err(crate::Error::Io)?;
    let sb_grid_offset = grid_offset + packed_bytes;
    drop(grid_entries);

    // Sections F+G: doc_map [new_num_virtual_docs each]
    let doc_map_offset = writer.offset() - blob_start;

    let src_ids = bmp.doc_map_ids_slice();
    let src_ords = bmp.doc_map_ordinals_slice();

    // Section F: doc_map_ids [u32-LE × new_num_virtual_docs]
    for &vid in perm.iter().take(num_real_docs) {
        let off = vid as usize * 4;
        let doc_id = u32::from_le_bytes(src_ids[off..off + 4].try_into().unwrap());
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
    for &vid in perm.iter().take(num_real_docs) {
        let off = vid as usize * 2;
        let ordinal = u16::from_le_bytes(src_ords[off..off + 2].try_into().unwrap());
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
        writer,
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

    Ok(())
}
