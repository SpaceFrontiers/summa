//! Sparse vector merge via per-dimension streaming and block stacking.
//!
//! File format (footer-based, data-first):
//! ```text
//! [posting data for all dims across all fields]
//! [TOC: per-field header + per-dim entries]
//! [footer: toc_offset(u64) + num_fields(u32) + magic(u32)]
//! ```
//! Data is streamed one dimension at a time, then dropped immediately.
//! Only the small TOC entries (~16 bytes per dim) are kept in memory.

use std::io::Write;
use std::sync::Arc;

use super::OffsetWriter;
use super::SegmentMerger;
use super::doc_offsets;
use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::FieldType;
use crate::segment::SparseIndex;
use crate::segment::reader::SegmentReader;
use crate::segment::sparse_format::SPARSE_FOOTER_MAGIC;
use crate::segment::types::SegmentFiles;

impl SegmentMerger {
    /// Merge sparse vector indexes using block stacking (streaming, O(1) mem per dim)
    ///
    /// Data is written directly to disk one dimension at a time. Only the small
    /// TOC (dim_id + offset + length per dim, ~16 bytes each) is accumulated in memory.
    /// After all data, the TOC and footer are appended.
    pub(super) async fn merge_sparse_vectors<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
    ) -> Result<usize> {
        use crate::structures::BlockSparsePostingList;
        use byteorder::{LittleEndian, WriteBytesExt};

        let doc_offs = doc_offsets(segments);
        for (i, seg) in segments.iter().enumerate() {
            log::debug!(
                "Sparse merge: segment {} has {} docs, doc_offset={}",
                i,
                seg.num_docs(),
                doc_offs[i]
            );
        }

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

        // Per-field TOC data accumulated in memory (~16 bytes per dim, tiny)
        struct FieldToc {
            field_id: u32,
            quantization: u8,
            dims: Vec<(u32, u64, u32)>, // (dim_id, data_offset, data_length)
        }
        let mut field_tocs: Vec<FieldToc> = Vec::new();

        for (field, sparse_config) in &sparse_fields {
            let quantization = sparse_config
                .as_ref()
                .map(|c| c.weight_quantization)
                .unwrap_or(crate::structures::WeightQuantization::Float32);

            // Collect all dimensions across all segments for this field
            let all_dims: Vec<u32> = {
                let mut set = rustc_hash::FxHashSet::default();
                for segment in segments {
                    if let Some(sparse_index) = segment.sparse_indexes().get(&field.0) {
                        for dim_id in sparse_index.active_dimensions() {
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
                "Sparse merge field {}: {} unique dims across {} segments",
                field.0,
                all_dims.len(),
                segments.len()
            );

            // Stream one dimension at a time: merge → serialize → write → drop
            let mut dim_entries: Vec<(u32, u64, u32)> = Vec::with_capacity(all_dims.len());
            let mut serialize_buf = Vec::new();

            for dim_id in &all_dims {
                let mut posting_arcs: Vec<(Arc<BlockSparsePostingList>, u32)> = Vec::new();

                for (seg_idx, sparse_idx) in sparse_indexes.iter().enumerate() {
                    if let Some(idx) = sparse_idx
                        && let Some(pl) = idx.get_posting(*dim_id).await?
                    {
                        posting_arcs.push((pl, doc_offs[seg_idx]));
                    }
                }

                if posting_arcs.is_empty() {
                    continue;
                }

                let lists_with_offsets: Vec<(&BlockSparsePostingList, u32)> = posting_arcs
                    .iter()
                    .map(|(pl, offset)| (pl.as_ref(), *offset))
                    .collect();

                let merged = BlockSparsePostingList::merge_with_offsets(&lists_with_offsets);

                let data_offset = writer.offset();
                serialize_buf.clear();
                merged
                    .serialize(&mut serialize_buf)
                    .map_err(crate::Error::Io)?;
                writer.write_all(&serialize_buf)?;

                dim_entries.push((*dim_id, data_offset, serialize_buf.len() as u32));
                // merged + posting_arcs dropped here — one dim at a time
            }

            if !dim_entries.is_empty() {
                field_tocs.push(FieldToc {
                    field_id: field.0,
                    quantization: quantization as u8,
                    dims: dim_entries,
                });
            }
        }

        if field_tocs.is_empty() {
            // No data written, clean up empty file
            drop(writer);
            let _ = dir.delete(&files.sparse).await;
            return Ok(0);
        }

        // Write TOC at end of file
        let toc_offset = writer.offset();
        for ftoc in &field_tocs {
            writer.write_u32::<LittleEndian>(ftoc.field_id)?;
            writer.write_u8(ftoc.quantization)?;
            writer.write_u32::<LittleEndian>(ftoc.dims.len() as u32)?;
            for &(dim_id, offset, length) in &ftoc.dims {
                writer.write_u32::<LittleEndian>(dim_id)?;
                writer.write_u64::<LittleEndian>(offset)?;
                writer.write_u32::<LittleEndian>(length)?;
            }
        }

        // Write footer: toc_offset(8) + num_fields(4) + magic(4) = 16 bytes
        writer.write_u64::<LittleEndian>(toc_offset)?;
        writer.write_u32::<LittleEndian>(field_tocs.len() as u32)?;
        writer.write_u32::<LittleEndian>(SPARSE_FOOTER_MAGIC)?;

        let output_size = writer.offset() as usize;
        writer.finish()?;

        let total_dims: usize = field_tocs.iter().map(|f| f.dims.len()).sum();
        log::info!(
            "Sparse vector merge complete: {} fields, {} dims, {}",
            field_tocs.len(),
            total_dims,
            super::format_bytes(output_size),
        );

        Ok(output_size)
    }
}
