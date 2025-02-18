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
                    merge_bmp_field(
                        &bmp_indexes,
                        &doc_offs,
                        field.0,
                        quantization,
                        bmp_block_size,
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

/// Emit one output block: serialize V10 interleaved data to writer, update grids.
///
/// `block_buf` contains `(dim_idx, local_slot, impact)` tuples for this block.
/// The caller is responsible for pushing the current cumulative offset to
/// `block_data_starts` before calling this function.
///
/// Returns the number of bytes written to the writer for this block.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn emit_v10_block(
    block_buf: &mut Vec<(u32, u8, u8)>,
    block_id: u32,
    writer: &mut OffsetWriter,
    blk_scratch: &mut Vec<u8>,
    packed_grid: &mut [u8],
    sb_grid: &mut [u8],
    packed_row_size: usize,
    num_superblocks: usize,
    dim_id_width: u8,
    total_terms: &mut u32,
    total_postings: &mut u32,
) -> Result<u64> {
    use crate::segment::builder::bmp::quantize_u8_to_u4_ceil;
    use crate::segment::reader::bmp::BMP_SUPERBLOCK_SIZE;

    if block_buf.is_empty() {
        // Empty block — no data
        return Ok(0);
    }

    // Sort by dim_idx to group postings by dimension
    block_buf.sort_unstable_by_key(|&(di, _, _)| di);

    blk_scratch.clear();

    // Pass 1: count terms (= distinct dim_idx values in sorted block_buf)
    let mut num_block_terms: u16 = 0;
    {
        let mut i = 0;
        while i < block_buf.len() {
            num_block_terms += 1;
            let dim = block_buf[i].0;
            while i < block_buf.len() && block_buf[i].0 == dim {
                i += 1;
            }
        }
    }

    // Write num_terms (u16)
    blk_scratch.extend_from_slice(&num_block_terms.to_le_bytes());

    // Write term_dim_indices
    {
        let mut i = 0;
        while i < block_buf.len() {
            let dim_idx = block_buf[i].0;
            if dim_id_width == 2 {
                blk_scratch.extend_from_slice(&(dim_idx as u16).to_le_bytes());
            } else {
                blk_scratch.extend_from_slice(&dim_idx.to_le_bytes());
            }
            while i < block_buf.len() && block_buf[i].0 == dim_idx {
                i += 1;
            }
        }
    }

    // Write posting_starts (u16, relative) + update grids
    {
        let mut i = 0;
        let mut cumul: u16 = 0;
        let b = block_id as usize;
        while i < block_buf.len() {
            let dim_idx = block_buf[i].0;
            blk_scratch.extend_from_slice(&cumul.to_le_bytes());
            let start = i;
            let mut max_imp = 0u8;
            while i < block_buf.len() && block_buf[i].0 == dim_idx {
                max_imp = max_imp.max(block_buf[i].2);
                i += 1;
            }
            cumul += (i - start) as u16;

            // Update packed grid (4-bit ceiling quantized)
            let q4 = quantize_u8_to_u4_ceil(max_imp);
            let row = dim_idx as usize * packed_row_size;
            if b.is_multiple_of(2) {
                packed_grid[row + b / 2] |= q4;
            } else {
                packed_grid[row + b / 2] |= q4 << 4;
            }
            // Update superblock grid (8-bit max)
            let sb = b / BMP_SUPERBLOCK_SIZE as usize;
            let sr = dim_idx as usize * num_superblocks + sb;
            if max_imp > sb_grid[sr] {
                sb_grid[sr] = max_imp;
            }
        }
        // Sentinel posting_start
        blk_scratch.extend_from_slice(&cumul.to_le_bytes());
        *total_postings += cumul as u32;
    }

    // Write postings [(u8, u8)] — already grouped by dim_idx after sort
    for &(_, slot, imp) in block_buf.iter() {
        blk_scratch.push(slot);
        blk_scratch.push(imp);
    }

    *total_terms += num_block_terms as u32;
    block_buf.clear();

    // Write block data directly to writer
    let block_bytes = blk_scratch.len() as u64;
    writer.write_all(blk_scratch).map_err(crate::Error::Io)?;

    Ok(block_bytes)
}

/// Merge BMP fields with **fully streaming V10 format**.
///
/// V10 data-first layout eliminates the ~1.4 GB `block_data_buf`:
///
/// 1. **No vid_pairs allocation**: `vid_bases[seg] + src_vid` for O(1) vid remapping;
///    doc_map sections (F+G) re-derived from source BmpIndex at write time.
/// 2. **In-place grids**: packed_grid + sb_grid arrays (~40 MB for 100 dims × 400K blocks)
///    replace grid_entries Vec (~1.2 GB for 100M terms × 12B).
/// 3. **Streaming block writes**: each block is written directly to the output file
///    via a small reusable scratch buffer (~4 KB). Only `block_data_starts` (~6 MB)
///    is accumulated in memory.
///
/// Peak memory: grids (~40 MB) + block_data_starts (~6 MB) + scratch (~4 KB).
#[allow(clippy::too_many_arguments)]
fn merge_bmp_field(
    bmp_indexes: &[Option<&BmpIndex>],
    doc_offs: &[u32],
    field_id: u32,
    quantization: crate::structures::WeightQuantization,
    bmp_block_size: u32,
    total_vectors: u32,
    writer: &mut OffsetWriter,
    field_tocs: &mut Vec<SparseFieldToc>,
) -> Result<()> {
    use crate::segment::builder::bmp::{write_u32_slice_le, write_u64_slice_le};
    use crate::segment::reader::bmp::BMP_SUPERBLOCK_SIZE;
    use byteorder::{LittleEndian, WriteBytesExt};

    // ── Phase 0: Compute merged parameters from source segments ──────────
    let mut new_max_weight_scale: f32 = 0.0;
    let mut dim_set = rustc_hash::FxHashSet::default();

    for bmp_opt in bmp_indexes.iter() {
        let bmp = match bmp_opt {
            Some(b) => b,
            None => continue,
        };
        if bmp.max_weight_scale > new_max_weight_scale {
            new_max_weight_scale = bmp.max_weight_scale;
        }
        for dim_id in bmp.dim_ids() {
            dim_set.insert(dim_id);
        }
    }

    if new_max_weight_scale == 0.0 || dim_set.is_empty() {
        return Ok(());
    }

    // Check if all segments share the same max_weight_scale (e.g. fixed quantization_factor).
    // When true, rescale_impact is a no-op (rescale=1.0) for all segments.
    let all_same_scale = bmp_indexes.iter().all(|bi| {
        bi.is_none_or(|b| (b.max_weight_scale - new_max_weight_scale).abs() < f32::EPSILON)
    });
    if all_same_scale {
        log::debug!(
            "[merge_bmp] field {}: all segments share max_weight_scale={:.4}, no rescaling needed",
            field_id,
            new_max_weight_scale,
        );
    }

    let mut dim_ids: Vec<u32> = dim_set.into_iter().collect();
    dim_ids.sort_unstable();
    let num_dims = dim_ids.len();

    // FxHashMap for O(1) dim_id → dim_idx lookup
    let dim_to_idx: rustc_hash::FxHashMap<u32, usize> =
        dim_ids.iter().enumerate().map(|(i, &d)| (d, i)).collect();

    // Safety: local_slot is u8, so block_size must not exceed 256
    let effective_block_size = bmp_block_size.min(256);
    let ebs = effective_block_size as u64;

    // ── Compute vid_bases and num_virtual_docs (no vid_pairs allocation) ─
    // doc_map sections (F+G) will be re-derived from source BmpIndex at write time,
    // saving ~300 MB for 50M vdocs.
    let mut num_virtual_docs: usize = 0;
    let mut vid_bases: Vec<u64> = Vec::with_capacity(bmp_indexes.len());
    for bmp_opt in bmp_indexes.iter() {
        vid_bases.push(num_virtual_docs as u64);
        if let Some(bmp) = bmp_opt {
            num_virtual_docs += bmp.num_virtual_docs as usize;
        }
    }

    if num_virtual_docs == 0 {
        return Ok(());
    }

    let num_blocks = num_virtual_docs.div_ceil(effective_block_size as usize);
    let dim_id_width: u8 = if num_dims <= 65536 { 2 } else { 4 };
    let packed_row_size = num_blocks.div_ceil(2);
    let num_superblocks = num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE as usize);

    // ── In-place grids (replaces grid_entries Vec) ───────────────────────
    // packed_grid: 4-bit packed, ~40 MB for 100 dims × 400K blocks
    // sb_grid:     8-bit max,    ~1.2 MB for 100 dims × 12.5K superblocks
    let mut packed_grid: Vec<u8> = vec![0u8; num_dims * packed_row_size];
    let mut sb_grid_arr: Vec<u8> = vec![0u8; num_dims * num_superblocks];

    // ── V10 streaming: block_data_starts accumulated, blocks written directly ─
    let mut block_data_starts: Vec<u64> = Vec::with_capacity(num_blocks + 1);
    let mut blk_scratch: Vec<u8> = Vec::with_capacity(4096);
    let mut block_bytes_written: u64 = 0;

    // Buffer for current output block being assembled
    let mut block_buf: Vec<(u32, u8, u8)> = Vec::with_capacity(256); // (dim_idx, local_slot, impact)
    let mut current_block: i64 = -1;
    let mut total_terms: u32 = 0;
    let mut total_postings: u32 = 0;

    // Temporary buffer for one source block's remapped postings (reused across iterations)
    let mut src_block_buf: Vec<(u32, u32, u8, u8)> =
        Vec::with_capacity(effective_block_size as usize * 256);

    // Process segments sequentially — they occupy contiguous vid ranges
    for (seg_idx, bmp_opt) in bmp_indexes.iter().enumerate() {
        let bmp = match bmp_opt {
            Some(b) => b,
            None => continue,
        };
        let vid_base = vid_bases[seg_idx];
        let rescale = bmp.max_weight_scale / new_max_weight_scale;

        for src_block in 0..bmp.num_blocks {
            src_block_buf.clear();

            for (dim_id, postings) in bmp.iter_block_terms(src_block) {
                let dim_idx = dim_to_idx[&dim_id] as u32;

                for p in postings {
                    let src_vid = src_block * bmp.bmp_block_size + p.local_slot as u32;
                    let new_vid = vid_base + src_vid as u64;
                    let new_block = (new_vid / ebs) as u32;
                    let new_slot = (new_vid % ebs) as u8;
                    let imp = rescale_impact(p.impact, rescale);
                    if imp == 0 {
                        continue;
                    }
                    src_block_buf.push((new_block, dim_idx, new_slot, imp));
                }
            }

            // Sort source block's remapped postings by (new_block, dim_idx).
            // A source block spans at most 2 output blocks, so this is tiny (~64K entries max).
            src_block_buf.sort_unstable_by_key(|&(nb, di, _, _)| ((nb as u64) << 32) | di as u64);

            // Distribute to output block buffer, flushing completed blocks
            for &(new_block, dim_idx, new_slot, imp) in &src_block_buf {
                while current_block < new_block as i64 {
                    if current_block >= 0 {
                        block_data_starts.push(block_bytes_written);
                        block_bytes_written += emit_v10_block(
                            &mut block_buf,
                            current_block as u32,
                            writer,
                            &mut blk_scratch,
                            &mut packed_grid,
                            &mut sb_grid_arr,
                            packed_row_size,
                            num_superblocks,
                            dim_id_width,
                            &mut total_terms,
                            &mut total_postings,
                        )?;
                    }
                    current_block += 1;
                }
                block_buf.push((dim_idx, new_slot, imp));
            }
        }
    }

    // Flush remaining blocks up to num_blocks
    while current_block < num_blocks as i64 {
        if current_block >= 0 {
            block_data_starts.push(block_bytes_written);
            block_bytes_written += emit_v10_block(
                &mut block_buf,
                current_block as u32,
                writer,
                &mut blk_scratch,
                &mut packed_grid,
                &mut sb_grid_arr,
                packed_row_size,
                num_superblocks,
                dim_id_width,
                &mut total_terms,
                &mut total_postings,
            )?;
        }
        current_block += 1;
    }
    // Sentinel
    block_data_starts.push(block_bytes_written);

    if total_terms == 0 {
        return Ok(());
    }

    log::debug!(
        "[merge_bmp] field {}: {} dims, {} terms, {} postings, {} blocks (block_size={})",
        field_id,
        num_dims,
        total_terms,
        total_postings,
        num_blocks,
        effective_block_size,
    );

    // ── Write remaining V10 sections ───────────────────────────────────────
    // Section B (block data) already written above via emit_v10_block.
    let mut bytes_written: u64 = block_bytes_written;

    // Padding to 8-byte boundary (for u64 alignment of Section A)
    let padding = (8 - (bytes_written % 8) as usize) % 8;
    if padding > 0 {
        writer
            .write_all(&[0u8; 8][..padding])
            .map_err(crate::Error::Io)?;
        bytes_written += padding as u64;
    }

    // Section A: block_data_starts [u64-LE × (num_blocks + 1)]
    bytes_written += write_u64_slice_le(writer, &block_data_starts).map_err(crate::Error::Io)?;
    drop(block_data_starts);

    // Section C: dim_ids [u32-LE × num_dims]
    let dim_ids_offset = bytes_written;
    bytes_written += write_u32_slice_le(writer, &dim_ids).map_err(crate::Error::Io)?;

    // Section D: packed grid [u8 × (num_dims × packed_row_size)]
    let grid_offset = bytes_written;
    writer.write_all(&packed_grid).map_err(crate::Error::Io)?;
    bytes_written += packed_grid.len() as u64;
    drop(packed_grid);

    // Section E: superblock grid [u8 × (num_dims × num_superblocks)]
    let sb_grid_offset = bytes_written;
    writer.write_all(&sb_grid_arr).map_err(crate::Error::Io)?;
    bytes_written += sb_grid_arr.len() as u64;
    drop(sb_grid_arr);

    // Section F: doc_map_ids [u32-LE × num_virtual_docs]
    // Re-derived from source BmpIndex — no vid_pairs allocation needed.
    let doc_map_offset = bytes_written;
    for (seg_idx, bmp_opt) in bmp_indexes.iter().enumerate() {
        let bmp = match bmp_opt {
            Some(b) => b,
            None => continue,
        };
        let doc_offset = doc_offs[seg_idx];
        for vid in 0..bmp.num_virtual_docs {
            let (doc_id, _ordinal) = bmp.virtual_to_doc(vid);
            writer
                .write_u32::<LittleEndian>(doc_id + doc_offset)
                .map_err(crate::Error::Io)?;
        }
    }
    bytes_written += num_virtual_docs as u64 * 4;

    // Section G: doc_map_ordinals [u16-LE × num_virtual_docs]
    for bmp_opt in bmp_indexes.iter() {
        let bmp = match bmp_opt {
            Some(b) => b,
            None => continue,
        };
        for vid in 0..bmp.num_virtual_docs {
            let (_doc_id, ordinal) = bmp.virtual_to_doc(vid);
            writer
                .write_u16::<LittleEndian>(ordinal)
                .map_err(crate::Error::Io)?;
        }
    }
    bytes_written += num_virtual_docs as u64 * 2;

    // BMP V10 Footer (64 bytes)
    use crate::segment::builder::bmp::write_v10_footer;
    write_v10_footer(
        writer,
        total_terms,
        total_postings,
        dim_ids_offset,
        grid_offset,
        num_blocks as u32,
        num_dims as u32,
        effective_block_size,
        num_virtual_docs as u32,
        new_max_weight_scale,
        sb_grid_offset,
        doc_map_offset,
    )
    .map_err(crate::Error::Io)?;
    bytes_written += 64;

    let blob_len = bytes_written;
    if blob_len > 0 {
        let current_offset = writer.offset() - blob_len;

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
                block_data_offset: current_offset,
                skip_start: (blob_len & 0xFFFFFFFF) as u32,
                num_blocks: ((blob_len >> 32) & 0xFFFFFFFF) as u32,
                doc_count: 0,
                max_weight: 0.0,
            }],
        });
    }

    Ok(())
}

/// Rescale an impact value when merging segments with different max_weight_scale.
#[inline]
fn rescale_impact(impact: u8, rescale: f32) -> u8 {
    if rescale >= 1.0 {
        impact
    } else {
        let v = (impact as f32 * rescale).round();
        v.min(255.0) as u8
    }
}
