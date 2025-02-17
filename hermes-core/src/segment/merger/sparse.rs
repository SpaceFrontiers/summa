//! Sparse vector merge via byte-level block stacking (V3 format).
//!
//! V3 file layout (footer-based, data-first):
//! ```text
//! [block data for all dims across all fields]
//! [skip section: SparseSkipEntry × total (24B each), contiguous]
//! [TOC: per-field header(9B) + per-dim entries(24B each)]
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

/// Merge BMP fields using a **block-at-a-time** approach.
///
/// Eliminates the O(total_terms) FxHashMap counting structure and O(total_terms)
/// sorted key vector that caused high memory usage on large merges. Instead:
///
/// **Fused counting pass:** Iterates output blocks sequentially. For each block:
///   1. Find contributing source blocks via `source_blocks_for_output()`
///   2. Count postings per dim_id using a small reusable scratch Vec
///   3. Update grid and append to metadata arrays incrementally
///
/// **Writing pass (same as before):** Re-iterates source blocks in output order,
/// streaming postings directly to the V3 blob writer.
///
/// Memory: `O(grid + metadata_arrays)` instead of `O(grid + counts_hashmap + term_keys)`.
/// For typical data: ~128MB grid + ~16MB metadata vs ~256MB grid + ~88MB HashMap/keys.
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
    use byteorder::{LittleEndian, WriteBytesExt};

    // ── Phase 0: Compute merged parameters from source segments ──────────
    let mut new_max_weight_scale: f32 = 0.0;
    let mut new_num_ordinals: u32 = 0;
    let mut max_merged_virtual: u64 = 0;
    let mut dim_set = rustc_hash::FxHashSet::default();

    for (seg_idx, bmp_opt) in bmp_indexes.iter().enumerate() {
        let bmp = match bmp_opt {
            Some(b) => b,
            None => continue,
        };
        if bmp.max_weight_scale > new_max_weight_scale {
            new_max_weight_scale = bmp.max_weight_scale;
        }
        if bmp.num_ordinals > new_num_ordinals {
            new_num_ordinals = bmp.num_ordinals;
        }
        for dim_id in bmp.dim_ids() {
            dim_set.insert(dim_id);
        }

        if bmp.num_blocks > 0 {
            let src_max_virtual = bmp.num_blocks as u64 * bmp.bmp_block_size as u64 - 1;
            let src_max_doc = src_max_virtual / bmp.num_ordinals as u64;
            let remapped_doc = src_max_doc + doc_offs[seg_idx] as u64;
            let merged_virtual = remapped_doc * new_num_ordinals.max(1) as u64
                + (new_num_ordinals.max(1) - 1) as u64;
            if merged_virtual > max_merged_virtual {
                max_merged_virtual = merged_virtual;
            }
        }
    }

    if new_max_weight_scale == 0.0 || dim_set.is_empty() {
        return Ok(());
    }

    let mut dim_ids: Vec<u32> = dim_set.into_iter().collect();
    dim_ids.sort_unstable();
    let num_dims = dim_ids.len();

    // Safety: local_slot is u8, so block_size must not exceed 256
    let effective_block_size = bmp_block_size.min(256);
    let num_blocks = if max_merged_virtual == 0 {
        1
    } else {
        (max_merged_virtual / effective_block_size as u64 + 1) as usize
    };

    let ebs = effective_block_size as u64;

    // ── Fused counting pass: block-at-a-time ─────────────────────────────
    // Build grid + metadata arrays incrementally, one output block at a time.
    // Per-block scratch (dim_id, posting_count, max_impact) is reused each block.
    // Eliminates global FxHashMap<u64, u32> counts + Vec<u64> term_keys.

    let mut grid = vec![0u8; num_dims * num_blocks];
    let mut block_term_starts: Vec<u32> = Vec::with_capacity(num_blocks + 1);
    let mut term_dim_ids: Vec<u32> = Vec::new();
    let mut term_posting_starts: Vec<u32> = vec![0]; // prefix sums
    let mut cumulative_postings: u32 = 0;

    // Per-block scratch: (dim_id, posting_count, max_impact) — reused each iteration
    let mut block_scratch: Vec<(u32, u32, u8)> = Vec::new();

    for block in 0..num_blocks as u32 {
        block_term_starts.push(term_dim_ids.len() as u32);
        block_scratch.clear();

        for (seg_idx, bmp_opt) in bmp_indexes.iter().enumerate() {
            let bmp = match bmp_opt {
                Some(b) => b,
                None => continue,
            };
            let doc_offset = doc_offs[seg_idx];
            let rescale = bmp.max_weight_scale / new_max_weight_scale;

            let (src_start, src_end) =
                source_blocks_for_output(block, ebs, bmp, doc_offset, new_num_ordinals);
            for src_block in src_start..src_end {
                let (st, et) = bmp.block_term_range(src_block);
                for ti in st..et {
                    let dim_id = bmp.block_term_dim_id(ti);
                    let mut count = 0u32;
                    let mut max_imp = 0u8;
                    for p in bmp.term_postings(ti) {
                        let src_virtual = src_block * bmp.bmp_block_size + p.local_slot as u32;
                        let (doc_id, ordinal) = bmp.virtual_to_doc(src_virtual);
                        let new_doc_id = doc_id + doc_offset;
                        let new_virtual =
                            new_doc_id as u64 * new_num_ordinals as u64 + ordinal as u64;
                        let new_block_id = (new_virtual / ebs) as u32;
                        if new_block_id != block {
                            continue;
                        }
                        let imp = rescale_impact(p.impact, rescale);
                        if imp == 0 {
                            continue;
                        }
                        count += 1;
                        if imp > max_imp {
                            max_imp = imp;
                        }
                    }
                    if count > 0 {
                        block_scratch.push((dim_id, count, max_imp));
                    }
                }
            }
        }

        if block_scratch.is_empty() {
            continue;
        }

        // Sort by dim_id, merge duplicates
        block_scratch.sort_unstable_by_key(|&(d, _, _)| d);

        let mut i = 0;
        while i < block_scratch.len() {
            let dim_id = block_scratch[i].0;
            let mut total_count = 0u32;
            let mut max_imp = 0u8;
            while i < block_scratch.len() && block_scratch[i].0 == dim_id {
                total_count += block_scratch[i].1;
                if block_scratch[i].2 > max_imp {
                    max_imp = block_scratch[i].2;
                }
                i += 1;
            }

            // Update grid (binary search instead of HashMap lookup)
            let dim_idx = dim_ids.binary_search(&dim_id).unwrap();
            let grid_idx = dim_idx * num_blocks + block as usize;
            if max_imp > grid[grid_idx] {
                grid[grid_idx] = max_imp;
            }

            term_dim_ids.push(dim_id);
            cumulative_postings += total_count;
            term_posting_starts.push(cumulative_postings);
        }
    }
    block_term_starts.push(term_dim_ids.len() as u32);

    let total_terms = term_dim_ids.len();
    let total_postings = cumulative_postings;

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

    // ── Write V3 blob — sections streamed in order ───────────────────────
    let blob_start = writer.offset();
    let mut bytes_written: u64 = 0;

    // Section 1: block_term_starts [u32-LE × (num_blocks + 1)]
    for &v in &block_term_starts {
        writer
            .write_u32::<LittleEndian>(v)
            .map_err(crate::Error::Io)?;
    }
    bytes_written += block_term_starts.len() as u64 * 4;

    // Section 2: term_dim_ids [u32-LE × total_terms]
    for &v in &term_dim_ids {
        writer
            .write_u32::<LittleEndian>(v)
            .map_err(crate::Error::Io)?;
    }
    bytes_written += total_terms as u64 * 4;

    // Section 3: term_posting_starts [u32-LE × (total_terms + 1)]
    for &v in &term_posting_starts {
        writer
            .write_u32::<LittleEndian>(v)
            .map_err(crate::Error::Io)?;
    }
    bytes_written += term_posting_starts.len() as u64 * 4;

    // Section 4: postings — re-iterate source segments in output order
    // For each (output_block, dim_id), find source postings and stream them.
    let mut postings_written = 0u32;
    for block in 0..num_blocks as u32 {
        let ts = block_term_starts[block as usize] as usize;
        let te = block_term_starts[block as usize + 1] as usize;
        if ts == te {
            continue;
        }

        for &dim_id in &term_dim_ids[ts..te] {
            for (seg_idx, bmp_opt) in bmp_indexes.iter().enumerate() {
                let bmp = match bmp_opt {
                    Some(b) => b,
                    None => continue,
                };
                let doc_offset = doc_offs[seg_idx];
                let rescale = bmp.max_weight_scale / new_max_weight_scale;

                // Determine which source blocks contribute to this output block
                let (src_start, src_end) =
                    source_blocks_for_output(block, ebs, bmp, doc_offset, new_num_ordinals);
                if src_start >= src_end {
                    continue;
                }

                for src_block in src_start..src_end {
                    let (st, et) = bmp.block_term_range(src_block);
                    if let Some(src_ti) = bmp.find_dim_in_block(st, et, dim_id) {
                        for p in bmp.term_postings(src_ti) {
                            let src_virtual = src_block * bmp.bmp_block_size + p.local_slot as u32;
                            let (doc_id, ordinal) = bmp.virtual_to_doc(src_virtual);
                            let new_doc_id = doc_id + doc_offset;
                            let new_virtual =
                                new_doc_id as u64 * new_num_ordinals as u64 + ordinal as u64;
                            let new_block_id = (new_virtual / ebs) as u32;
                            if new_block_id != block {
                                continue;
                            }

                            let new_local_slot = (new_virtual % ebs) as u8;
                            let new_impact = rescale_impact(p.impact, rescale);
                            if new_impact == 0 {
                                continue;
                            }

                            writer
                                .write_all(&[new_local_slot, new_impact])
                                .map_err(crate::Error::Io)?;
                            postings_written += 1;
                        }
                    }
                }
            }
        }
    }
    bytes_written += postings_written as u64 * 2;

    debug_assert_eq!(
        postings_written, total_postings,
        "BMP merge posting count mismatch: wrote {} but expected {}",
        postings_written, total_postings
    );

    // Section 5: padding to 4-byte boundary
    let padding = (4 - (bytes_written % 4) as usize) % 4;
    if padding > 0 {
        writer
            .write_all(&[0u8; 4][..padding])
            .map_err(crate::Error::Io)?;
        bytes_written += padding as u64;
    }

    // Section 6: dim_ids [u32-LE × num_dims]
    let dim_ids_offset = bytes_written as u32;
    for &d in &dim_ids {
        writer
            .write_u32::<LittleEndian>(d)
            .map_err(crate::Error::Io)?;
    }
    bytes_written += num_dims as u64 * 4;

    // Section 7: grid [u8 × (num_dims × num_blocks)]
    let grid_offset = bytes_written as u32;
    writer.write_all(&grid).map_err(crate::Error::Io)?;
    bytes_written += grid.len() as u64;

    // BMP V3 Footer (48 bytes)
    use crate::segment::format::BMP_BLOB_MAGIC_V3;
    writer
        .write_u32::<LittleEndian>(total_terms as u32)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(total_postings)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(dim_ids_offset)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(grid_offset)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(num_blocks as u32)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(num_dims as u32)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(effective_block_size)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(new_num_ordinals)
        .map_err(crate::Error::Io)?;
    writer
        .write_f32::<LittleEndian>(new_max_weight_scale)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(0)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(0)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(BMP_BLOB_MAGIC_V3)
        .map_err(crate::Error::Io)?;
    bytes_written += 48;

    let blob_len = bytes_written;
    let _ = blob_start; // used for offset computation verification

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

/// Determine which source blocks from a segment contribute to an output block.
///
/// Returns `(first_src_block, exclusive_end_src_block)`.
/// Uses a fast path when source and output ordinals match (simple shift),
/// and a conservative bound otherwise.
#[inline]
fn source_blocks_for_output(
    out_block: u32,
    effective_bs: u64,
    bmp: &BmpIndex,
    doc_offset: u32,
    new_num_ordinals: u32,
) -> (u32, u32) {
    let src_bs = bmp.bmp_block_size as u64;
    let new_ord = new_num_ordinals as u64;
    let out_v_lo = out_block as u64 * effective_bs;
    let out_v_hi = out_v_lo + effective_bs;
    let max_src_virt = bmp.num_blocks as u64 * src_bs;

    if bmp.num_ordinals as u64 == new_ord {
        // Fast path: simple virtual ID shift
        let shift = doc_offset as u64 * new_ord;
        if out_v_hi <= shift {
            return (0, 0);
        }
        let src_lo = out_v_lo.saturating_sub(shift);
        let src_hi = out_v_hi - shift;
        if src_lo >= max_src_virt {
            return (0, 0);
        }
        let first = (src_lo / src_bs) as u32;
        let last = (src_hi.min(max_src_virt).div_ceil(src_bs)).min(bmp.num_blocks as u64) as u32;
        (first, last)
    } else {
        // Conservative bound for mixed ordinals
        let min_doc = (out_v_lo / new_ord).saturating_sub(doc_offset as u64);
        let max_doc = out_v_hi.div_ceil(new_ord).saturating_sub(doc_offset as u64) + 1;
        let min_sv = min_doc * bmp.num_ordinals as u64;
        let max_sv = (max_doc + 1) * bmp.num_ordinals as u64;
        if min_sv >= max_src_virt {
            return (0, 0);
        }
        let first = (min_sv / src_bs) as u32;
        let last = (max_sv.min(max_src_virt).div_ceil(src_bs)).min(bmp.num_blocks as u64) as u32;
        (first, last)
    }
}
