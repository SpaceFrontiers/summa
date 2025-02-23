//! Sparse vector merge via byte-level block stacking (V3 format).
//!
//! V3 file layout (footer-based, data-first):
//! ```text
//! [block data for all dims across all fields]
//! [skip section: SparseSkipEntry × total (24B each), contiguous]
//! [TOC: per-field header(13B) + per-dim entries(28B each)]
//! [footer: skip_offset(8) + toc_offset(8) + num_fields(4) + magic(4) = 24B]
//! ```
//!
//! Each dimension is merged by stacking raw block bytes from source segments.
//! No deserialization or re-serialization of block data — only the small
//! skip entries (24 bytes per block) are written fresh in a separate section.
//! The raw block bytes are copied directly from mmap.

use std::io::Write;

use super::OffsetWriter;
use super::SegmentMerger;
use super::doc_offsets;
use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::FieldType;
use crate::segment::BmpIndex;
use crate::segment::SparseIndex;
use crate::segment::format::{SparseDimTocEntry, SparseFieldToc, write_sparse_toc_and_footer};
use crate::segment::reader::SegmentReader;
use crate::segment::types::SegmentFiles;
use crate::structures::{SparseFormat, SparseSkipEntry};

impl SegmentMerger {
    /// Merge sparse vector indexes via byte-level block stacking (V3 format).
    ///
    /// V3 separates block data from skip entries:
    ///   Phase 1: Write raw block data for all dims (copied from mmap)
    ///   Phase 2: Write skip section (all skip entries contiguous)
    ///   Phase 3: Write TOC (per-field header + per-dim entries with embedded metadata)
    ///   Phase 4: Write 24-byte footer
    pub(super) async fn merge_sparse_vectors<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
    ) -> Result<usize> {
        let doc_offs = doc_offsets(segments)?;

        // Collect all sparse vector fields from schema
        let sparse_fields: Vec<_> = self
            .schema
            .fields()
            .filter(|(_, entry)| matches!(entry.field_type, FieldType::SparseVector))
            .map(|(field, entry)| (field, entry.sparse_vector_config.clone()))
            .collect();

        if sparse_fields.is_empty() {
            return Ok(0);
        }

        let mut writer = OffsetWriter::new(dir.streaming_writer(&files.sparse).await?);

        // Accumulated per-field data for TOC
        let mut field_tocs: Vec<SparseFieldToc> = Vec::new();
        // Skip entries serialized as contiguous bytes (24B each)
        let mut skip_bytes: Vec<u8> = Vec::new();
        let mut skip_count: u32 = 0;

        for (field, sparse_config) in &sparse_fields {
            let format = sparse_config.as_ref().map(|c| c.format).unwrap_or_default();
            let quantization = sparse_config
                .as_ref()
                .map(|c| c.weight_quantization)
                .unwrap_or(crate::structures::WeightQuantization::Float32);

            // BMP format: merge BMP indexes if any source segments have them
            if format == SparseFormat::Bmp {
                let bmp_indexes: Vec<Option<&BmpIndex>> = segments
                    .iter()
                    .map(|seg| seg.bmp_indexes().get(&field.0))
                    .collect();
                let has_bmp_data = bmp_indexes.iter().any(|bi| bi.is_some());
                if has_bmp_data {
                    let total_vectors_bmp: u32 = bmp_indexes
                        .iter()
                        .filter_map(|bi| bi.map(|idx| idx.total_vectors))
                        .sum();
                    let bmp_block_size = sparse_config
                        .as_ref()
                        .map(|c| c.bmp_block_size)
                        .unwrap_or(64);
                    let dims = sparse_config
                        .as_ref()
                        .and_then(|c| c.dims)
                        .unwrap_or_else(|| {
                            // Derive from first available source
                            bmp_indexes
                                .iter()
                                .find_map(|bi| bi.map(|idx| idx.dims()))
                                .unwrap_or(105879)
                        });
                    let max_weight_scale = sparse_config
                        .as_ref()
                        .and_then(|c| c.max_weight)
                        .unwrap_or_else(|| {
                            bmp_indexes
                                .iter()
                                .find_map(|bi| bi.map(|idx| idx.max_weight_scale))
                                .unwrap_or(5.0)
                        });
                    // Record-level reorder: only when force_reorder is explicitly
                    // requested. The `reorder` SDL attribute controls build-time
                    // BP reorder but does NOT trigger reorder during merge, as
                    // merge-time BP is expensive and block-copy merge is sufficient.
                    if self.force_reorder {
                        let sources: Vec<(&BmpIndex, u32)> = bmp_indexes
                            .iter()
                            .copied()
                            .zip(doc_offs.iter().copied())
                            .filter_map(|(opt, doc_off)| opt.map(|bmp| (bmp, doc_off)))
                            .collect();
                        if !sources.is_empty() {
                            reorder_bmp_blob(
                                &sources,
                                field.0,
                                quantization,
                                dims,
                                bmp_block_size.min(256) as usize,
                                max_weight_scale,
                                total_vectors_bmp,
                                &mut writer,
                                &mut field_tocs,
                            )?;
                            continue;
                        }
                    }
                    merge_bmp_field(
                        &bmp_indexes,
                        segments,
                        &doc_offs,
                        field.0,
                        quantization,
                        dims,
                        bmp_block_size,
                        max_weight_scale,
                        total_vectors_bmp,
                        &mut writer,
                        &mut field_tocs,
                    )?;
                    continue;
                }
                // No BMP data — fall through to MaxScore path
                // (handles legacy segments that used MaxScore format)
            }

            // MaxScore format: byte-level block stacking
            // Collect all unique dimension IDs across segments (sorted for determinism)
            let all_dims: Vec<u32> = {
                let mut set = rustc_hash::FxHashSet::default();
                for segment in segments {
                    if let Some(si) = segment.sparse_indexes().get(&field.0) {
                        for dim_id in si.active_dimensions() {
                            set.insert(dim_id);
                        }
                    }
                }
                let mut v: Vec<u32> = set.into_iter().collect();
                v.sort_unstable();
                v
            };

            if all_dims.is_empty() {
                continue;
            }

            let sparse_indexes: Vec<Option<&SparseIndex>> = segments
                .iter()
                .map(|seg| seg.sparse_indexes().get(&field.0))
                .collect();

            // Sum total_vectors across segments for this field
            let total_vectors: u32 = sparse_indexes
                .iter()
                .filter_map(|si| si.map(|idx| idx.total_vectors))
                .sum();

            log::debug!(
                "[merge] sparse field {}: {} unique dims across {} segments, total_vectors={}",
                field.0,
                all_dims.len(),
                segments.len(),
                total_vectors,
            );

            let mut dim_toc_entries: Vec<SparseDimTocEntry> = Vec::with_capacity(all_dims.len());

            for &dim_id in &all_dims {
                // Collect raw data from each segment (skip entries + raw block bytes)
                let mut sources = Vec::with_capacity(segments.len());
                for (seg_idx, sparse_idx) in sparse_indexes.iter().enumerate() {
                    if let Some(idx) = sparse_idx
                        && let Some(raw) = idx.read_dim_raw(dim_id).await?
                    {
                        sources.push((raw, doc_offs[seg_idx]));
                    }
                }

                if sources.is_empty() {
                    continue;
                }

                // Compute merged metadata
                let total_docs: u32 = sources.iter().map(|(r, _)| r.doc_count).sum();
                let global_max: f32 = sources
                    .iter()
                    .map(|(r, _)| r.global_max_weight)
                    .fold(f32::NEG_INFINITY, f32::max);
                let total_blocks: u32 = sources
                    .iter()
                    .map(|(r, _)| r.skip_entries.len() as u32)
                    .sum();

                // Phase 1: Write block data only (no header, no skip entries)
                let block_data_offset = writer.offset();

                // Serialize adjusted skip entries directly to byte buffer
                let skip_start = skip_count;
                let mut cumulative_block_offset = 0u64;

                // Block header layout: count(2) + doc_id_bits(1) + ordinal_bits(1)
                //   + weight_quant(1) + pad(1) + pad(2) + first_doc_id(4, LE) + max_weight(4)
                const FIRST_DOC_ID_OFFSET: usize = 8;
                for (src_idx, (raw, doc_offset)) in sources.iter().enumerate() {
                    let _ = src_idx; // used by diagnostics feature
                    let data = raw.raw_block_data.as_slice();

                    #[cfg(feature = "diagnostics")]
                    super::diagnostics::validate_merge_source(dim_id, src_idx, raw)?;

                    // Serialize adjusted skip entries to byte buffer
                    for entry in &raw.skip_entries {
                        SparseSkipEntry::new(
                            entry.first_doc + doc_offset,
                            entry.last_doc + doc_offset,
                            cumulative_block_offset + entry.offset,
                            entry.length,
                            entry.max_weight,
                        )
                        .write_to_vec(&mut skip_bytes);
                        skip_count += 1;
                    }
                    // Advance cumulative offset by this source's total block data size
                    if let Some(last) = raw.skip_entries.last() {
                        cumulative_block_offset += last.offset + last.length as u64;
                    }

                    // Write raw block data, patching first_doc_id when doc_offset > 0
                    if *doc_offset == 0 {
                        writer.write_all(data)?;
                    } else {
                        for (i, entry) in raw.skip_entries.iter().enumerate() {
                            let start = entry.offset as usize;
                            let end = if i + 1 < raw.skip_entries.len() {
                                raw.skip_entries[i + 1].offset as usize
                            } else {
                                data.len()
                            };
                            let block = &data[start..end];
                            writer.write_all(&block[..FIRST_DOC_ID_OFFSET])?;
                            let old = u32::from_le_bytes(
                                block[FIRST_DOC_ID_OFFSET..FIRST_DOC_ID_OFFSET + 4]
                                    .try_into()
                                    .unwrap(),
                            );
                            writer.write_all(&(old + doc_offset).to_le_bytes())?;
                            writer.write_all(&block[FIRST_DOC_ID_OFFSET + 4..])?;
                        }
                    }
                }

                dim_toc_entries.push(SparseDimTocEntry {
                    dim_id,
                    block_data_offset,
                    skip_start,
                    num_blocks: total_blocks,
                    doc_count: total_docs,
                    max_weight: global_max,
                });
            }

            if !dim_toc_entries.is_empty() {
                field_tocs.push(SparseFieldToc {
                    field_id: field.0,
                    quantization: quantization as u8,
                    total_vectors,
                    dims: dim_toc_entries,
                });
            }
        }

        if field_tocs.is_empty() {
            drop(writer);
            let _ = dir.delete(&files.sparse).await;
            return Ok(0);
        }

        // Phase 2: Write skip section (dump serialized bytes)
        let skip_offset = writer.offset();
        writer.write_all(&skip_bytes).map_err(crate::Error::Io)?;
        drop(skip_bytes);

        // Phase 3 + 4: Write TOC + footer
        let toc_offset = writer.offset();
        write_sparse_toc_and_footer(&mut writer, skip_offset, toc_offset, &field_tocs)
            .map_err(crate::Error::Io)?;

        let output_size = writer.offset() as usize;
        writer.finish().map_err(crate::Error::Io)?;

        let total_dims: usize = field_tocs.iter().map(|f| f.dims.len()).sum();
        log::info!(
            "[merge_sparse] file written: {:.2} MB ({} fields, {} dims, {} skip entries)",
            output_size as f64 / (1024.0 * 1024.0),
            field_tocs.len(),
            total_dims,
            skip_count,
        );

        Ok(output_size)
    }
}

/// Merge BMP fields with **streaming block-copy V13 format**.
///
/// V13 block-copy merge: all segments share the same `dims`, `bmp_block_size`,
/// and `max_weight_scale`, so blocks are self-contained and can be copied directly.
///
/// Phases:
/// 1. Stream Section B (block data) — sequential chunked copy
/// 2. Write padding + Section A (block_data_starts) — recomputed on-the-fly
/// 3. Stream Section D (packed_grid) — one row at a time
/// 4. Write Section E (sb_grid) — one row at a time
/// 5. Stream Section F+G (doc_map) — bulk copy with offset patching
/// 6. Write V13 footer (64 bytes)
///
/// Peak memory: row_buf (~4 MB) + sb_row (~120 KB) + id_buf (256 KB) + tiny Vecs.
#[allow(clippy::too_many_arguments)]
fn merge_bmp_field(
    bmp_indexes: &[Option<&BmpIndex>],
    _segments: &[SegmentReader],
    doc_offs: &[u32],
    field_id: u32,
    quantization: crate::structures::WeightQuantization,
    dims: u32,
    bmp_block_size: u32,
    max_weight_scale: f32,
    total_vectors: u32,
    writer: &mut OffsetWriter,
    field_tocs: &mut Vec<SparseFieldToc>,
) -> Result<()> {
    use crate::segment::builder::bmp::write_v13_footer;
    use crate::segment::reader::bmp::BMP_SUPERBLOCK_SIZE;

    let effective_block_size = bmp_block_size.min(256);

    // Pre-build source list: (&BmpIndex, doc_offset). Filters out None segments.
    let sources: Vec<(&BmpIndex, u32)> = bmp_indexes
        .iter()
        .copied()
        .zip(doc_offs.iter().copied())
        .filter_map(|(opt, doc_off)| opt.map(|bmp| (bmp, doc_off)))
        .collect();

    if sources.is_empty() {
        return Ok(());
    }

    // ── Phase 0: Validate all sources share dims, block_size, max_weight_scale ──
    let mut total_source_blocks: u32 = 0;
    let mut num_real_docs_total: u32 = 0;

    for &(bmp, _) in &sources {
        if bmp.dims() != dims {
            return Err(crate::Error::Corruption(format!(
                "BMP merge: source dims={} != expected dims={}",
                bmp.dims(),
                dims
            )));
        }
        if bmp.bmp_block_size != effective_block_size {
            return Err(crate::Error::Corruption(format!(
                "BMP merge: source block_size={} != expected {}",
                bmp.bmp_block_size, effective_block_size
            )));
        }
        if (bmp.max_weight_scale - max_weight_scale).abs() > f32::EPSILON {
            return Err(crate::Error::Corruption(format!(
                "BMP merge: source max_weight_scale={:.4} != expected {:.4}",
                bmp.max_weight_scale, max_weight_scale
            )));
        }
        total_source_blocks += bmp.num_blocks;
        num_real_docs_total += bmp.num_real_docs();
    }

    if total_source_blocks == 0 {
        return Ok(());
    }

    let num_blocks = total_source_blocks as usize;
    let num_virtual_docs = num_blocks * effective_block_size as usize;
    let packed_row_size = num_blocks.div_ceil(2);
    let num_superblocks = num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE as usize);

    // Pre-compute aggregate stats (needed for footer, independent of block order)
    let mut total_terms: u32 = 0;
    let mut total_postings: u32 = 0;
    for &(bmp, _) in &sources {
        total_terms += bmp.total_terms() as u32;
        total_postings += bmp.total_postings() as u32;
    }
    if total_terms == 0 {
        return Ok(());
    }

    // Pre-compute per-source block offsets (first global block id for each source)
    let mut block_offsets: Vec<u32> = Vec::with_capacity(sources.len());
    {
        let mut cumulative: u32 = 0;
        for &(bmp, _) in &sources {
            block_offsets.push(cumulative);
            cumulative += bmp.num_blocks;
        }
    }

    log::debug!(
        "[merge_bmp_v13] field {}: dims={}, {} sources, {} total_blocks, \
         block_size={}, max_weight_scale={:.4}",
        field_id,
        dims,
        sources.len(),
        num_blocks,
        effective_block_size,
        max_weight_scale,
    );

    // Hint sequential access on all source mmaps before reading
    #[cfg(feature = "native")]
    for &(bmp, _) in &sources {
        bmp.madvise_sequential();
    }

    let blob_start = writer.offset();

    const CHUNK_SIZE: usize = 4 * 1024 * 1024; // 4 MB

    // ═══ Sequential block-copy merge ════════════════════════════════════
    // Blocks are self-contained — copy directly from sources in order.
    // For record-level BP reorder, use `reorder_bmp_blob` (handles 1+ sources).

    // ── Phase 1: Stream Section B (block data) — chunked copy ───
    let mut source_byte_offsets: Vec<u64> = Vec::with_capacity(sources.len());
    let mut cumulative_bytes: u64 = 0;

    for &(bmp, _) in &sources {
        source_byte_offsets.push(cumulative_bytes);
        let sentinel = bmp.block_data_sentinel() as usize;
        let src_data = &bmp.block_data_slice()[..sentinel];
        for chunk in src_data.chunks(CHUNK_SIZE) {
            writer.write_all(chunk).map_err(crate::Error::Io)?;
        }
        cumulative_bytes += sentinel as u64;
    }

    // Release block data pages
    #[cfg(feature = "native")]
    for &(bmp, _) in &sources {
        bmp.madvise_dontneed_block_data();
    }

    // ── Phase 2: Write padding + Section A (block_data_starts) ──
    let block_data_len = writer.offset() - blob_start;
    let padding = (8 - (block_data_len % 8) as usize) % 8;
    if padding > 0 {
        writer
            .write_all(&[0u8; 8][..padding])
            .map_err(crate::Error::Io)?;
    }

    for (src_idx, &(bmp, _)) in sources.iter().enumerate() {
        let base = source_byte_offsets[src_idx];
        for b in 0..bmp.num_blocks {
            let val = base + bmp.block_data_start(b);
            writer
                .write_all(&val.to_le_bytes())
                .map_err(crate::Error::Io)?;
        }
    }
    // Sentinel
    writer
        .write_all(&cumulative_bytes.to_le_bytes())
        .map_err(crate::Error::Io)?;

    // ── Phase 3: Stream Section D (packed_grid) — one row at a time
    let grid_offset = writer.offset() - blob_start;
    let mut row_buf = vec![0u8; packed_row_size];

    for dim_id in 0..dims {
        row_buf.fill(0);
        for (src_idx, &(bmp, _)) in sources.iter().enumerate() {
            let col_offset = block_offsets[src_idx] as usize;
            let src_prs = bmp.packed_row_size();
            let src_num_blocks = bmp.num_blocks as usize;
            let src_row_start = dim_id as usize * src_prs;
            let src_row_end = src_row_start + src_prs;
            let src_grid = bmp.grid_slice();
            if src_row_end > src_grid.len() {
                continue;
            }
            let src_row = &src_grid[src_row_start..src_row_end];
            copy_nibbles(src_row, src_num_blocks, &mut row_buf, col_offset);
        }
        writer.write_all(&row_buf).map_err(crate::Error::Io)?;
    }
    drop(row_buf);

    // ── Phase 4: Stream Section E (sb_grid) — one row at a time ─
    let sb_grid_offset = writer.offset() - blob_start;
    let mut sb_row = vec![0u8; num_superblocks];
    let sb_size = BMP_SUPERBLOCK_SIZE as usize;

    for dim_id in 0..dims {
        sb_row.fill(0);
        for (src_idx, &(bmp, _)) in sources.iter().enumerate() {
            let col_offset = block_offsets[src_idx] as usize;
            let src_num_blocks = bmp.num_blocks as usize;
            let src_num_sbs = bmp.num_superblocks as usize;
            let src_sb_grid = bmp.sb_grid_slice();
            let src_sb_row_start = dim_id as usize * src_num_sbs;
            let src_sb_row_end = src_sb_row_start + src_num_sbs;
            if src_sb_row_end > src_sb_grid.len() {
                continue;
            }
            let src_sb_row = &src_sb_grid[src_sb_row_start..src_sb_row_end];

            for (sb_src, &val) in src_sb_row.iter().enumerate() {
                if val == 0 {
                    continue;
                }
                let first_block = col_offset + sb_src * sb_size;
                let last_block = (first_block + sb_size).min(col_offset + src_num_blocks) - 1;
                let first_out_sb = first_block / sb_size;
                let last_out_sb = last_block / sb_size;
                for slot in &mut sb_row[first_out_sb..=last_out_sb] {
                    if val > *slot {
                        *slot = val;
                    }
                }
            }
        }
        writer.write_all(&sb_row).map_err(crate::Error::Io)?;
    }
    drop(sb_row);

    // Release grid pages
    #[cfg(feature = "native")]
    for &(bmp, _) in &sources {
        bmp.madvise_dontneed_grids();
    }

    // ── Phase 5: Stream Section F+G (doc_map) — bulk copy ───────
    let doc_map_offset = writer.offset() - blob_start;

    const DOC_MAP_CHUNK: usize = 64 * 1024;
    let mut id_buf = vec![0u8; DOC_MAP_CHUNK * 4];

    for &(bmp, doc_offset) in &sources {
        let src_ids = bmp.doc_map_ids_slice();
        let n = bmp.num_virtual_docs as usize;

        if doc_offset == 0 {
            for chunk in src_ids[..n * 4].chunks(CHUNK_SIZE) {
                writer.write_all(chunk).map_err(crate::Error::Io)?;
            }
        } else {
            for chunk_start in (0..n).step_by(DOC_MAP_CHUNK) {
                let chunk_end = (chunk_start + DOC_MAP_CHUNK).min(n);
                let chunk_len = chunk_end - chunk_start;
                let src = &src_ids[chunk_start * 4..chunk_end * 4];
                let dst = &mut id_buf[..chunk_len * 4];
                dst.copy_from_slice(src);

                for i in 0..chunk_len {
                    let off = i * 4;
                    let doc_id = u32::from_le_bytes(dst[off..off + 4].try_into().unwrap());
                    if doc_id != u32::MAX {
                        let adjusted = doc_id + doc_offset;
                        dst[off..off + 4].copy_from_slice(&adjusted.to_le_bytes());
                    }
                }
                writer.write_all(dst).map_err(crate::Error::Io)?;
            }
        }
    }
    drop(id_buf);

    for &(bmp, _) in &sources {
        let src_ords = bmp.doc_map_ordinals_slice();
        let n = bmp.num_virtual_docs as usize;
        for chunk in src_ords[..n * 2].chunks(CHUNK_SIZE) {
            writer.write_all(chunk).map_err(crate::Error::Io)?;
        }
    }

    // ── Phase 6: Write V13 footer (64 bytes) ────────────────────────────
    write_v13_footer(
        writer,
        total_terms,
        total_postings,
        grid_offset,
        sb_grid_offset,
        num_blocks as u32,
        dims,
        effective_block_size,
        num_virtual_docs as u32,
        max_weight_scale,
        doc_map_offset,
        num_real_docs_total,
    )
    .map_err(crate::Error::Io)?;

    let blob_len = writer.offset() - blob_start;
    push_bmp_field_toc(
        field_tocs,
        field_id,
        quantization,
        total_vectors,
        blob_start,
        blob_len,
    );

    Ok(())
}

/// Push a BMP sentinel field TOC entry after writing a blob.
fn push_bmp_field_toc(
    field_tocs: &mut Vec<SparseFieldToc>,
    field_id: u32,
    quantization: crate::structures::WeightQuantization,
    total_vectors: u32,
    blob_start: u64,
    blob_len: u64,
) {
    if blob_len == 0 {
        return;
    }
    let mut config_for_byte =
        crate::structures::SparseVectorConfig::from_byte(quantization as u8).unwrap_or_default();
    config_for_byte.format = SparseFormat::Bmp;
    config_for_byte.weight_quantization = quantization;

    field_tocs.push(SparseFieldToc {
        field_id,
        quantization: config_for_byte.to_byte(),
        total_vectors,
        dims: vec![SparseDimTocEntry {
            dim_id: 0xFFFFFFFF, // sentinel for BMP
            block_data_offset: blob_start,
            skip_start: (blob_len & 0xFFFFFFFF) as u32,
            num_blocks: ((blob_len >> 32) & 0xFFFFFFFF) as u32,
            doc_count: 0,
            max_weight: 0.0,
        }],
    });
}

/// Record-level BMP reorder via Recursive Graph Bisection (BP).
///
/// Supports single-source (re-reorder) and multi-source (merge + reorder) modes.
/// When merging multiple already-reordered segments, the concatenated order
/// preserves within-segment locality, giving BP a warm start that converges
/// faster than starting from random order.
///
/// **Two-phase streaming design:**
///
/// Phase 1 — Build forward index from block data (all sources), run BP to get permutation.
/// Phase 2 — Write new blob with records in permuted order (random-read from sources).
///
/// Memory: forward index ~200 bytes/doc + permutation, plus one output block
/// scratch buffer (~4 KB) and grid_entries.
#[allow(clippy::too_many_arguments)]
fn reorder_bmp_blob(
    bmps: &[(&BmpIndex, u32)],
    field_id: u32,
    quantization: crate::structures::WeightQuantization,
    dims: u32,
    effective_block_size: usize,
    max_weight_scale: f32,
    total_vectors: u32,
    writer: &mut OffsetWriter,
    field_tocs: &mut Vec<SparseFieldToc>,
) -> Result<()> {
    use crate::segment::builder::bmp::{stream_write_grids, write_v13_footer};
    use crate::segment::builder::graph_bisection::{
        build_forward_index_from_bmps, graph_bisection,
    };

    let num_sources = bmps.len();
    let bmp_refs: Vec<&BmpIndex> = bmps.iter().map(|&(b, _)| b).collect();
    let num_real_docs: usize = bmp_refs.iter().map(|b| b.num_real_docs() as usize).sum();

    if num_real_docs == 0 {
        return Ok(());
    }

    log::info!(
        "[reorder_bmp] field {}: Phase 1 — running BP on {} sources, {} real docs",
        field_id,
        num_sources,
        num_real_docs,
    );

    // ── Phase 1: Build forward index and run BP ─────────────────────────
    let bp_start = std::time::Instant::now();
    let max_doc_freq = ((num_real_docs as f64) * 0.9) as usize;
    let min_doc_freq = 128.min(num_real_docs);

    let (fwd, source_doc_counts) =
        build_forward_index_from_bmps(&bmp_refs, min_doc_freq, max_doc_freq.max(1));

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

    // Pre-compute cumulative doc offsets for multi-source vid resolution
    let mut cumulative_docs: Vec<usize> = Vec::with_capacity(num_sources + 1);
    cumulative_docs.push(0);
    for &count in &source_doc_counts {
        cumulative_docs.push(cumulative_docs.last().unwrap() + count);
    }

    log::info!(
        "[reorder_bmp] field {}: Phase 2 — writing reordered blob ({} blocks)",
        field_id,
        num_real_docs.div_ceil(effective_block_size),
    );

    // ── Phase 2: Write new blob with records in permuted order ──────────
    let new_num_blocks = num_real_docs.div_ceil(effective_block_size);
    let new_num_virtual_docs = new_num_blocks * effective_block_size;

    let blob_start = writer.offset();
    let mut block_data_starts: Vec<u64> = Vec::with_capacity(new_num_blocks + 1);
    let est_terms: u32 = bmp_refs.iter().map(|b| b.total_terms() as u32).sum();
    let mut grid_entries: Vec<(u32, u32, u8)> = Vec::with_capacity(est_terms as usize);
    let mut total_terms: u32 = 0;
    let mut total_postings: u32 = 0;
    let mut cumulative_bytes: u64 = 0;

    // Per-block scratch buffers
    let mut blk_buf: Vec<u8> = Vec::with_capacity(4096);
    let mut dim_postings: rustc_hash::FxHashMap<u32, Vec<(u8, u8)>> =
        rustc_hash::FxHashMap::default();
    // Group source slots by (source_idx, old_block) -> Vec<(old_slot, new_local_slot)>
    let mut source_slots: rustc_hash::FxHashMap<(usize, usize), Vec<(u8, u8)>> =
        rustc_hash::FxHashMap::default();
    let mut slot_map = [u8::MAX; 256];

    for out_block in 0..new_num_blocks {
        block_data_starts.push(cumulative_bytes);

        let new_vid_start = out_block * effective_block_size;
        let new_vid_end = ((out_block + 1) * effective_block_size).min(num_real_docs);
        let slots_count = new_vid_end - new_vid_start;

        // Group records by (source, source_block)
        source_slots.clear();
        for new_local_slot in 0..slots_count {
            let combined_vid = perm[new_vid_start + new_local_slot] as usize;
            let (src_idx, local_vid) = resolve_source(combined_vid, &cumulative_docs);
            let old_block = local_vid / effective_block_size;
            let old_slot = (local_vid % effective_block_size) as u8;
            source_slots
                .entry((src_idx, old_block))
                .or_default()
                .push((old_slot, new_local_slot as u8));
        }

        dim_postings.clear();

        // Iterate each source block once, scatter matching slots
        for (&(src_idx, old_block), mappings) in &source_slots {
            for &(old_s, new_s) in mappings {
                slot_map[old_s as usize] = new_s;
            }

            let bmp = bmp_refs[src_idx];
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

    // Sections F+G: doc_map_ids + doc_map_ordinals [new_num_virtual_docs each]
    let doc_map_offset = writer.offset() - blob_start;

    // Pre-resolve all permuted vids → (source_idx, local_vid) to avoid
    // repeated binary searches across Sections F and G.
    let resolved: Vec<(usize, usize)> = perm
        .iter()
        .take(num_real_docs)
        .map(|&v| resolve_source(v as usize, &cumulative_docs))
        .collect();

    // Section F: doc_map_ids [u32-LE × new_num_virtual_docs]
    for &(src_idx, local_vid) in &resolved {
        let bmp = bmp_refs[src_idx];
        let doc_offset = bmps[src_idx].1;
        let src_ids = bmp.doc_map_ids_slice();
        let off = local_vid * 4;
        let doc_id = u32::from_le_bytes(src_ids[off..off + 4].try_into().unwrap());
        let adjusted = if doc_id != u32::MAX {
            doc_id + doc_offset
        } else {
            doc_id
        };
        writer
            .write_all(&adjusted.to_le_bytes())
            .map_err(crate::Error::Io)?;
    }
    for _ in num_real_docs..new_num_virtual_docs {
        writer
            .write_all(&u32::MAX.to_le_bytes())
            .map_err(crate::Error::Io)?;
    }

    // Section G: doc_map_ordinals [u16-LE × new_num_virtual_docs]
    for &(src_idx, local_vid) in &resolved {
        let bmp = bmp_refs[src_idx];
        let src_ords = bmp.doc_map_ordinals_slice();
        let off = local_vid * 2;
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
    push_bmp_field_toc(
        field_tocs,
        field_id,
        quantization,
        total_vectors,
        blob_start,
        blob_len,
    );

    log::info!(
        "[reorder_bmp] field {}: done — {} sources, {} blocks, {} terms, {} postings, {:.2} MB",
        field_id,
        num_sources,
        new_num_blocks,
        total_terms,
        total_postings,
        blob_len as f64 / (1024.0 * 1024.0),
    );

    Ok(())
}

/// Resolve a combined virtual ID to (source_index, local_vid) using cumulative offsets.
///
/// `cumulative_docs` is `[0, n0, n0+n1, ...]` where n_i is the doc count of source i.
#[inline]
fn resolve_source(combined_vid: usize, cumulative_docs: &[usize]) -> (usize, usize) {
    let src_idx = cumulative_docs
        .partition_point(|&c| c <= combined_vid)
        .saturating_sub(1);
    (src_idx, combined_vid - cumulative_docs[src_idx])
}

/// Copy 4-bit nibbles from a source grid row to a destination row at a column offset.
///
/// Hot inner loop for grid merge (~8 KB per row).
/// Source nibbles are packed: low nibble = even block, high nibble = odd block.
#[inline]
fn copy_nibbles(src_row: &[u8], src_blocks: usize, dst_row: &mut [u8], offset: usize) {
    for b in 0..src_blocks {
        let val = if b.is_multiple_of(2) {
            src_row[b / 2] & 0x0F
        } else {
            src_row[b / 2] >> 4
        };
        if val == 0 {
            continue;
        }
        let out_b = offset + b;
        if out_b.is_multiple_of(2) {
            dst_row[out_b / 2] |= val;
        } else {
            dst_row[out_b / 2] |= val << 4;
        }
    }
}
