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

/// Merge BMP fields using a **single-pass block-at-a-time** approach.
///
/// V6 format with compact virtual IDs:
/// - **Compact virtual IDs**: sequential assignment eliminates sparse ID space blowup.
/// - **Single-pass collection**: all postings collected with compact IDs, sorted, emitted.
/// - **FxHashMap dim lookup**: O(1) dim_id → dim_idx instead of O(log n) binary search.
/// - **sb_grid on disk**: superblock grid is computed during merge and persisted in V6 format.
/// - **Bulk writes**: `write_u32_slice_le()` for sections 1-3, 6.
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
    use crate::segment::builder::bmp::{
        stream_write_grids, write_dim_indices_section, write_u32_slice_le,
    };
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

    // FxHashMap for O(1) dim_id → dim_idx lookup (replaces binary search)
    let dim_to_idx: rustc_hash::FxHashMap<u32, usize> =
        dim_ids.iter().enumerate().map(|(i, &d)| (d, i)).collect();

    // Safety: local_slot is u8, so block_size must not exceed 256
    let effective_block_size = bmp_block_size.min(256);

    // Pre-scan: collect all unique (new_doc_id, ordinal) pairs for compact virtual IDs.
    // Iterates doc_maps directly — O(total_virtual_docs) instead of O(total_postings).
    let mut vid_pairs: Vec<(u32, u16)> = Vec::new();
    for (seg_idx, bmp_opt) in bmp_indexes.iter().enumerate() {
        let bmp = match bmp_opt {
            Some(b) => b,
            None => continue,
        };
        let doc_offset = doc_offs[seg_idx];
        for vid in 0..bmp.num_virtual_docs {
            let (doc_id, ordinal) = bmp.virtual_to_doc(vid);
            vid_pairs.push((doc_id + doc_offset, ordinal));
        }
    }
    vid_pairs.sort_unstable();
    vid_pairs.dedup();
    let num_virtual_docs = vid_pairs.len();

    if num_virtual_docs == 0 {
        return Ok(());
    }

    // Lookup: binary search on sorted vid_pairs (O(log N) per lookup, saves ~50-80 bytes/entry)

    let num_blocks = num_virtual_docs.div_ceil(effective_block_size as usize);
    let ebs = effective_block_size as u64;

    // ── Collect all postings, sort by (block, dim), emit ───────────────────
    // Grid entries are collected sparsely — no dense grid allocation.
    // They are sorted and streamed dim-by-dim at write time.
    let mut grid_entries: Vec<(u32, u32, u8)> = Vec::new(); // (dim_idx, block_id, max_impact)
    let mut block_term_starts: Vec<u32> = Vec::with_capacity(num_blocks + 1);
    let mut term_dim_ids: Vec<u32> = Vec::new();
    let mut term_posting_starts: Vec<u32> = vec![0]; // prefix sums
    let mut cumulative_postings: u32 = 0;

    // Single pass over all source segments: collect all postings with compact virtual IDs.
    // Sort by (block_id, dim_idx) for ordered output. No source_blocks_for_output needed.
    let total_source_postings: usize = bmp_indexes
        .iter()
        .filter_map(|bi| bi.map(|b| b.total_postings() as usize))
        .sum();

    // Flat buffer for all postings across all blocks (2 bytes per posting)
    let mut postings_buf: Vec<u8> = Vec::with_capacity(total_source_postings * 2);

    struct MergedPosting {
        block_id: u32,
        dim_idx: u32,
        local_slot: u8,
        impact: u8,
    }

    let mut all_postings: Vec<MergedPosting> = Vec::with_capacity(total_source_postings);
    for (seg_idx, bmp_opt) in bmp_indexes.iter().enumerate() {
        let bmp = match bmp_opt {
            Some(b) => b,
            None => continue,
        };
        let doc_offset = doc_offs[seg_idx];
        let rescale = bmp.max_weight_scale / new_max_weight_scale;

        for src_block in 0..bmp.num_blocks {
            let (st, et) = bmp.block_term_range(src_block);
            for ti in st..et {
                let dim_id = bmp.block_term_dim_id(ti);
                let dim_idx = dim_to_idx[&dim_id];

                for p in bmp.term_postings(ti) {
                    let src_virtual = src_block * bmp.bmp_block_size + p.local_slot as u32;
                    let (doc_id, ordinal) = bmp.virtual_to_doc(src_virtual);
                    let new_doc_id = doc_id + doc_offset;
                    let new_virtual =
                        crate::segment::builder::bmp::vid_lookup(&vid_pairs, (new_doc_id, ordinal))
                            as u64;
                    let new_block_id = (new_virtual / ebs) as u32;
                    let new_local_slot = (new_virtual % ebs) as u8;
                    let imp = rescale_impact(p.impact, rescale);
                    if imp == 0 {
                        continue;
                    }
                    all_postings.push(MergedPosting {
                        block_id: new_block_id,
                        dim_idx: dim_idx as u32,
                        local_slot: new_local_slot,
                        impact: imp,
                    });
                }
            }
        }
    }

    // Sort by (block_id, dim_idx) for block-ordered, dim-sorted output.
    // Pack into u64 for single-comparison sort key.
    all_postings.sort_unstable_by_key(|p| (p.block_id as u64) << 32 | p.dim_idx as u64);

    // Build output arrays from sorted postings
    let mut pi = 0; // cursor into all_postings
    for block in 0..num_blocks as u32 {
        block_term_starts.push(term_dim_ids.len() as u32);

        // Process all postings for this block, grouped by dim
        while pi < all_postings.len() && all_postings[pi].block_id == block {
            let dim_idx = all_postings[pi].dim_idx as usize;
            let dim_start = pi;
            let mut max_imp = 0u8;

            // Consume all postings for this (block, dim)
            while pi < all_postings.len()
                && all_postings[pi].block_id == block
                && all_postings[pi].dim_idx as usize == dim_idx
            {
                let p = &all_postings[pi];
                postings_buf.push(p.local_slot);
                postings_buf.push(p.impact);
                if p.impact > max_imp {
                    max_imp = p.impact;
                }
                pi += 1;
            }

            let count = (pi - dim_start) as u32;
            term_dim_ids.push(dim_idx as u32);
            cumulative_postings += count;
            term_posting_starts.push(cumulative_postings);
            grid_entries.push((dim_idx as u32, block, max_imp));
        }
    }
    block_term_starts.push(term_dim_ids.len() as u32);
    drop(all_postings); // Free 8 × N bytes before write phase

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

    // ── Write V7 blob — no second pass ───────────────────────────────────
    let blob_start = writer.offset();
    let mut bytes_written: u64 = 0;

    // Section 1: block_term_starts [u32-LE × (num_blocks + 1)]
    bytes_written += write_u32_slice_le(writer, &block_term_starts).map_err(crate::Error::Io)?;

    // Section 2: term_dim_indices — u16 when num_dims ≤ 65536, else u32
    bytes_written +=
        write_dim_indices_section(writer, &term_dim_ids, num_dims).map_err(crate::Error::Io)?;

    // Section 3: term_posting_starts [u32-LE × (total_terms + 1)]
    bytes_written += write_u32_slice_le(writer, &term_posting_starts).map_err(crate::Error::Io)?;

    // Section 4: postings — single write from buffer (no second pass!)
    writer.write_all(&postings_buf).map_err(crate::Error::Io)?;
    bytes_written += postings_buf.len() as u64;

    debug_assert_eq!(
        postings_buf.len() as u32,
        total_postings * 2,
        "BMP merge posting count mismatch: buffered {} bytes but expected {} postings × 2",
        postings_buf.len(),
        total_postings
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
    bytes_written += write_u32_slice_le(writer, &dim_ids).map_err(crate::Error::Io)?;

    // Sections 7+8: packed grid + sb_grid (streaming from sparse grid entries)
    grid_entries.sort_unstable();
    let grid_offset = bytes_written as u32;
    let (packed_bytes, sb_bytes) = stream_write_grids(&grid_entries, num_dims, num_blocks, writer)
        .map_err(crate::Error::Io)?;
    let sb_grid_offset = (bytes_written + packed_bytes) as u32;
    bytes_written += packed_bytes + sb_bytes;

    // Section 9a: doc_map_ids [u32-LE × num_virtual_docs]
    let doc_map_offset = bytes_written as u32;
    for &(doc_id, _) in &vid_pairs {
        writer
            .write_u32::<LittleEndian>(doc_id)
            .map_err(crate::Error::Io)?;
    }
    bytes_written += num_virtual_docs as u64 * 4;

    // Section 9b: doc_map_ordinals [u16-LE × num_virtual_docs]
    for &(_, ord) in &vid_pairs {
        writer
            .write_u16::<LittleEndian>(ord)
            .map_err(crate::Error::Io)?;
    }
    bytes_written += num_virtual_docs as u64 * 2;

    // BMP V7 Footer (48 bytes)
    use crate::segment::format::BMP_BLOB_MAGIC_V7;
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
        .write_u32::<LittleEndian>(num_virtual_docs as u32)
        .map_err(crate::Error::Io)?;
    writer
        .write_f32::<LittleEndian>(new_max_weight_scale)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(sb_grid_offset)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(doc_map_offset)
        .map_err(crate::Error::Io)?;
    writer
        .write_u32::<LittleEndian>(BMP_BLOB_MAGIC_V7)
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
