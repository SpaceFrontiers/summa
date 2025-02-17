//! Sparse vector merge via byte-level block stacking (V3 format).
//!
//! V3 file layout (footer-based, data-first):
//! ```text
//! [block data for all dims across all fields]
//! [skip section: SparseSkipEntry × total (20B each), contiguous]
//! [TOC: per-field header(9B) + per-dim entries(24B each)]
//! [footer: skip_offset(8) + toc_offset(8) + num_fields(4) + magic(4) = 24B]
//! ```
//!
//! Each dimension is merged by stacking raw block bytes from source segments.
//! No deserialization or re-serialization of block data — only the small
//! skip entries (20 bytes per block) are written fresh in a separate section.
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
        // Accumulated skip entries (written in Phase 2)
        let mut all_skip_entries: Vec<SparseSkipEntry> = Vec::new();

        for (field, sparse_config) in &sparse_fields {
            let format = sparse_config.as_ref().map(|c| c.format).unwrap_or_default();
            let quantization = sparse_config
                .as_ref()
                .map(|c| c.weight_quantization)
                .unwrap_or(crate::structures::WeightQuantization::Float32);

            // BMP format: iterate source blocks, remap doc_ids, stream to output
            if format == SparseFormat::Bmp {
                let bmp_indexes: Vec<Option<&BmpIndex>> = segments
                    .iter()
                    .map(|seg| seg.bmp_indexes().get(&field.0))
                    .collect();
                let total_vectors: u32 = bmp_indexes
                    .iter()
                    .filter_map(|bi| bi.map(|idx| idx.total_vectors))
                    .sum();
                let has_data = bmp_indexes.iter().any(|bi| bi.is_some());
                if has_data {
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
                        total_vectors,
                        &mut writer,
                        &mut field_tocs,
                    )?;
                }
                continue;
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

                // Accumulate adjusted skip entries for Phase 2
                let skip_start = all_skip_entries.len() as u32;
                let mut cumulative_block_offset = 0u64;

                // Block header layout: count(2) + doc_id_bits(1) + ordinal_bits(1)
                //   + weight_quant(1) + pad(1) + pad(2) + first_doc_id(4, LE) + max_weight(4)
                const FIRST_DOC_ID_OFFSET: usize = 8;
                for (src_idx, (raw, doc_offset)) in sources.iter().enumerate() {
                    let _ = src_idx; // used by diagnostics feature
                    let data = raw.raw_block_data.as_slice();

                    #[cfg(feature = "diagnostics")]
                    super::diagnostics::validate_merge_source(dim_id, src_idx, raw)?;

                    // Adjust skip entries (doc offsets + block data offsets)
                    for entry in &raw.skip_entries {
                        all_skip_entries.push(SparseSkipEntry::new(
                            entry.first_doc + doc_offset,
                            entry.last_doc + doc_offset,
                            cumulative_block_offset + entry.offset,
                            entry.length,
                            entry.max_weight,
                        ));
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

        // Phase 2: Write skip section (all skip entries contiguous)
        let skip_offset = writer.offset();
        for entry in &all_skip_entries {
            entry.write(&mut writer).map_err(crate::Error::Io)?;
        }

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
            all_skip_entries.len(),
        );

        Ok(output_size)
    }
}

/// Merge BMP fields by iterating source blocks directly, remapping doc_ids,
/// and streaming the rebuilt blob to the writer.
///
/// Memory-bounded: uses a single flat Vec of compact entries (12 bytes each)
/// plus the block-max grid (num_dims x num_blocks bytes). No HashMap, no
/// per-block Vec allocations, no intermediate blob buffer.
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
    use rustc_hash::FxHashMap;

    // Phase 1: Compute merged parameters from source segments
    let mut new_max_weight_scale: f32 = 0.0;
    let mut new_num_ordinals: u32 = 0;
    let mut max_merged_virtual: u64 = 0;
    let mut dim_set = rustc_hash::FxHashSet::default();
    let mut total_postings_est: usize = 0;

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
        total_postings_est += bmp.total_postings() as usize;

        // Compute max virtual_id this segment can contribute in merged space
        if bmp.num_blocks > 0 {
            let src_max_virtual = bmp.num_blocks as u64 * bmp.bmp_block_size as u64 - 1;
            let src_max_doc = src_max_virtual / bmp.num_ordinals as u64;
            let remapped_doc = src_max_doc + doc_offs[seg_idx] as u64;
            // In merged space with new_num_ordinals (use max possible)
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
    let dim_to_idx: FxHashMap<u32, usize> =
        dim_ids.iter().enumerate().map(|(i, &d)| (d, i)).collect();

    let num_blocks = (max_merged_virtual / bmp_block_size as u64 + 1) as usize;

    // Phase 2: Iterate source segments, collect flat entries + build grid
    let mut entries: Vec<(u32, u32, u8, u8)> = Vec::with_capacity(total_postings_est);
    let mut grid = vec![0u8; num_dims * num_blocks];

    for (seg_idx, bmp_opt) in bmp_indexes.iter().enumerate() {
        let bmp = match bmp_opt {
            Some(b) => b,
            None => continue,
        };
        let doc_offset = doc_offs[seg_idx];
        let rescale = bmp.max_weight_scale / new_max_weight_scale;

        for block_id in 0..bmp.num_blocks {
            let (term_start, term_end) = bmp.block_term_range(block_id);

            for rel_idx in 0..(term_end - term_start) {
                let ti = term_start + rel_idx;
                let dim_id = bmp.block_term_dim_id(ti);
                let dim_idx = dim_to_idx[&dim_id];

                for p in bmp.term_postings(ti) {
                    // Decode virtual_id in source space
                    let src_virtual = block_id * bmp.bmp_block_size + p.local_slot as u32;
                    let (doc_id, ordinal) = bmp.virtual_to_doc(src_virtual);

                    // Remap to merged space
                    let new_doc_id = doc_id + doc_offset;
                    let new_virtual = new_doc_id as u64 * new_num_ordinals as u64 + ordinal as u64;
                    let new_block_id = (new_virtual / bmp_block_size as u64) as u32;
                    let new_local_slot = (new_virtual % bmp_block_size as u64) as u8;

                    // Rescale impact (u8 → u8, no f32 roundtrip)
                    let new_impact = if rescale >= 1.0 {
                        p.impact
                    } else {
                        let v = (p.impact as f32 * rescale).round();
                        v.min(255.0) as u8
                    };
                    if new_impact == 0 {
                        continue;
                    }

                    // Update grid
                    let grid_idx = dim_idx * num_blocks + new_block_id as usize;
                    if new_impact > grid[grid_idx] {
                        grid[grid_idx] = new_impact;
                    }

                    entries.push((new_block_id, dim_id, new_local_slot, new_impact));
                }
            }
        }
    }

    if entries.is_empty() {
        return Ok(());
    }

    log::debug!(
        "[merge_bmp] field {}: {} dims, {} entries, {} blocks (bmp_block_size={})",
        field_id,
        num_dims,
        entries.len(),
        num_blocks,
        bmp_block_size,
    );

    // Phase 3: Sort entries by (block_id, dim_id, local_slot)
    entries.sort_unstable();

    // Phase 4: Stream BMP blob to writer
    let current_offset = writer.offset();
    let blob_len = super::super::builder::bmp::write_bmp_blob_v3(
        &entries,
        &dim_ids,
        &grid,
        num_blocks,
        bmp_block_size,
        new_num_ordinals,
        new_max_weight_scale,
        writer,
    )
    .map_err(crate::Error::Io)?;

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
