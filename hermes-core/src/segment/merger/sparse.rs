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
                    merge_bmp_field(
                        &bmp_indexes,
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

/// Merge BMP fields with **streaming block-copy V11 format**.
///
/// V11 block-copy merge: all segments share the same `dims`, `bmp_block_size`,
/// and `max_weight_scale`, so blocks are self-contained and can be copied directly.
///
/// Phases:
/// 1. Stream Section B (block data) — chunked file-to-file copy from source mmaps
/// 2. Write padding + Section A (block_data_starts) — recomputed on-the-fly
/// 3. Stream Section D (packed_grid) — one row at a time
/// 4. Write Section E (sb_grid) — one row at a time
/// 5. Stream Section F+G (doc_map) — bulk copy with offset patching
/// 6. Write V11 footer (64 bytes)
///
/// Peak memory: row_buf (~4 MB) + sb_row (~120 KB) + id_buf (256 KB) + tiny Vecs.
#[allow(clippy::too_many_arguments)]
fn merge_bmp_field(
    bmp_indexes: &[Option<&BmpIndex>],
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
    use crate::segment::builder::bmp::write_v11_footer;
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

    log::debug!(
        "[merge_bmp_v11] field {}: dims={}, {} sources, {} total_blocks, \
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

    // ── Phase 1: Stream Section B (block data) — chunked copy ─────────
    // Copy in 4 MB chunks to let the kernel evict pages between chunks,
    // avoiding multi-GB RSS spikes from a single write_all.
    const CHUNK_SIZE: usize = 4 * 1024 * 1024; // 4 MB

    let mut total_terms: u32 = 0;
    let mut total_postings: u32 = 0;

    // Per-source cumulative byte/block offsets for Phase 2 recomputation
    // and Phase 3/4 grid column positioning.
    let mut source_byte_offsets: Vec<u64> = Vec::with_capacity(sources.len());
    let mut block_offsets: Vec<u32> = Vec::with_capacity(sources.len());
    let mut cumulative_bytes: u64 = 0;
    let mut cumulative_blocks: u32 = 0;

    for &(bmp, _) in &sources {
        source_byte_offsets.push(cumulative_bytes);
        block_offsets.push(cumulative_blocks);

        let sentinel = bmp.block_data_sentinel() as usize;
        let src_data = &bmp.block_data_slice()[..sentinel];
        for chunk in src_data.chunks(CHUNK_SIZE) {
            writer.write_all(chunk).map_err(crate::Error::Io)?;
        }
        cumulative_bytes += sentinel as u64;
        cumulative_blocks += bmp.num_blocks;

        total_terms += bmp.total_terms() as u32;
        total_postings += bmp.total_postings() as u32;
    }

    if total_terms == 0 {
        return Ok(());
    }

    // Release block data pages — no longer needed after Phase 1.
    // Keep block_data_starts (needed for Phase 2 recomputation).
    #[cfg(feature = "native")]
    for &(bmp, _) in &sources {
        bmp.madvise_dontneed_block_data();
    }

    // ── Phase 2: Write padding + Section A (block_data_starts) ──────────
    // Recompute from source metadata on-the-fly (no buffering).
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

    // ── Phase 3: Stream Section D (packed_grid) — one row at a time ─────
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

    // ── Phase 4: Stream Section E (sb_grid) — one row at a time ─────────
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
                // Map source SB to output SB(s) — may span boundary
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

    // Release grid pages — no longer needed after Phase 3+4
    #[cfg(feature = "native")]
    for &(bmp, _) in &sources {
        bmp.madvise_dontneed_grids();
    }

    // ── Phase 5: Stream Section F+G (doc_map) — bulk copy from mmaps ──
    let doc_map_offset = writer.offset() - blob_start;

    // Section F: doc_map_ids [u32-LE × num_virtual_docs]
    const DOC_MAP_CHUNK: usize = 64 * 1024; // 64K entries × 4 bytes = 256 KB
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

    // Section G: doc_map_ordinals [u16-LE × num_virtual_docs]
    for &(bmp, _) in &sources {
        let src_ords = bmp.doc_map_ordinals_slice();
        let n = bmp.num_virtual_docs as usize;
        for chunk in src_ords[..n * 2].chunks(CHUNK_SIZE) {
            writer.write_all(chunk).map_err(crate::Error::Io)?;
        }
    }

    // ── Phase 6: Write V11 footer (64 bytes) ────────────────────────────
    write_v11_footer(
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
    if blob_len > 0 {
        let mut config_for_byte =
            crate::structures::SparseVectorConfig::from_byte(quantization as u8)
                .unwrap_or_default();
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

    Ok(())
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
