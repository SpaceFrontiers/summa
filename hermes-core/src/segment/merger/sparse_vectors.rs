//! Sparse vector merge via byte-level block stacking.
//!
//! File format (footer-based, data-first):
//! ```text
//! [posting data for all dims across all fields]
//! [TOC: per-field header + per-dim entries]
//! [footer: toc_offset(u64) + num_fields(u32) + magic(u32)]
//! ```
//!
//! Each dimension is merged by stacking raw block bytes from source segments.
//! No deserialization or re-serialization of block data — only the small
//! header and skip entries (20 bytes per block) are written fresh.
//! The raw block bytes are copied directly from mmap.

use std::io::Write;

use super::OffsetWriter;
use super::SegmentMerger;
use super::doc_offsets;
use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::FieldType;
use crate::segment::SparseIndex;
use crate::segment::format::{SparseFieldToc, write_sparse_toc_and_footer};
use crate::segment::reader::SegmentReader;
use crate::segment::types::SegmentFiles;
use crate::structures::SparseSkipEntry;

impl SegmentMerger {
    /// Merge sparse vector indexes via byte-level block stacking.
    ///
    /// For each dimension, reads raw block data bytes from each source segment
    /// (single mmap read per segment, zero deserialization) and writes:
    ///   1. A new header (doc_count + global_max_weight + num_blocks)
    ///   2. Adjusted skip entries (first_doc/last_doc += doc_offset, block offsets remapped)
    ///   3. Raw block bytes copied directly from source segments
    ///
    /// Memory per dimension: only skip entries (~20 bytes × num_blocks) + mmap views.
    /// TOC (~16 bytes per dim) is the only thing accumulated across dimensions.
    pub(super) async fn merge_sparse_vectors<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
    ) -> Result<usize> {
        use byteorder::{LittleEndian, WriteBytesExt};

        let doc_offs = doc_offsets(segments);

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

        let mut field_tocs: Vec<SparseFieldToc> = Vec::new();

        for (field, sparse_config) in &sparse_fields {
            let quantization = sparse_config
                .as_ref()
                .map(|c| c.weight_quantization)
                .unwrap_or(crate::structures::WeightQuantization::Float32);

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

            log::debug!(
                "[merge] sparse field {}: {} unique dims across {} segments",
                field.0,
                all_dims.len(),
                segments.len()
            );

            let mut dim_entries: Vec<(u32, u64, u32)> = Vec::with_capacity(all_dims.len());

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

                // Compute merged header values
                let total_docs: u32 = sources.iter().map(|(r, _)| r.doc_count).sum();
                let global_max: f32 = sources
                    .iter()
                    .map(|(r, _)| r.global_max_weight)
                    .fold(f32::NEG_INFINITY, f32::max);
                let total_blocks: u32 = sources
                    .iter()
                    .map(|(r, _)| r.skip_entries.len() as u32)
                    .sum();

                let data_offset = writer.offset();

                // Write posting list header: doc_count(4) + global_max_weight(4) + num_blocks(4)
                writer.write_u32::<LittleEndian>(total_docs)?;
                writer.write_f32::<LittleEndian>(global_max)?;
                writer.write_u32::<LittleEndian>(total_blocks)?;

                // Write adjusted skip entries (block offsets remapped to merged layout)
                let mut block_data_offset = 0u32;
                for (raw, doc_offset) in &sources {
                    for entry in raw.skip_entries {
                        let adjusted = SparseSkipEntry::new(
                            entry.first_doc + doc_offset,
                            entry.last_doc + doc_offset,
                            block_data_offset + entry.offset,
                            entry.length,
                            entry.max_weight,
                        );
                        adjusted.write(&mut writer).map_err(crate::Error::Io)?;
                    }
                    // Advance cumulative offset by this source's total block data size
                    if let Some(last) = raw.skip_entries.last() {
                        block_data_offset += last.offset + last.length;
                    }
                }

                // Copy raw block data from each source, patching first_doc_id in each
                // block header when doc_offset > 0.
                // Block header layout: count(2) + doc_id_bits(1) + ordinal_bits(1)
                //   + weight_quant(1) + pad(1) + pad(2) + first_doc_id(4, LE) + max_weight(4)
                // So first_doc_id is at byte offset 8 within each block.
                const FIRST_DOC_ID_OFFSET: usize = 8;
                const BLOCK_HEADER_SIZE: usize = 16;
                for (raw, doc_offset) in &sources {
                    let data = raw.raw_block_data.as_slice();
                    if *doc_offset == 0 {
                        writer.write_all(data)?;
                    } else {
                        // Write block-by-block, patching 4-byte first_doc_id inline.
                        // Avoids cloning the entire raw_block_data buffer.
                        for (i, entry) in raw.skip_entries.iter().enumerate() {
                            let start = entry.offset as usize;
                            let end = if i + 1 < raw.skip_entries.len() {
                                raw.skip_entries[i + 1].offset as usize
                            } else {
                                data.len()
                            };
                            let block = &data[start..end];
                            if block.len() >= BLOCK_HEADER_SIZE {
                                // Write header prefix (before first_doc_id)
                                writer.write_all(&block[..FIRST_DOC_ID_OFFSET])?;
                                // Write patched first_doc_id
                                let old = u32::from_le_bytes(
                                    block[FIRST_DOC_ID_OFFSET..FIRST_DOC_ID_OFFSET + 4]
                                        .try_into()
                                        .unwrap(),
                                );
                                writer.write_all(&(old + doc_offset).to_le_bytes())?;
                                // Write remainder of block
                                writer.write_all(&block[FIRST_DOC_ID_OFFSET + 4..])?;
                            } else {
                                writer.write_all(block)?;
                            }
                        }
                    }
                }

                let posting_len = (writer.offset() - data_offset) as u32;
                dim_entries.push((dim_id, data_offset, posting_len));
                // sources (mmap views + skip entry refs) dropped here
            }

            if !dim_entries.is_empty() {
                field_tocs.push(SparseFieldToc {
                    field_id: field.0,
                    quantization: quantization as u8,
                    dims: dim_entries,
                });
            }
        }

        if field_tocs.is_empty() {
            drop(writer);
            let _ = dir.delete(&files.sparse).await;
            return Ok(0);
        }

        let toc_offset = writer.offset();
        write_sparse_toc_and_footer(&mut writer, toc_offset, &field_tocs)
            .map_err(crate::Error::Io)?;

        let output_size = writer.offset() as usize;
        writer.finish()?;

        let total_dims: usize = field_tocs.iter().map(|f| f.dims.len()).sum();
        log::info!(
            "[merge] sparse done: {} fields, {} dims, {}",
            field_tocs.len(),
            total_dims,
            super::format_bytes(output_size),
        );

        Ok(output_size)
    }
}
