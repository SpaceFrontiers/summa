//! Sparse vector merge using block stacking
//!
//! Merges sparse vector indexes by stacking compressed posting list blocks
//! directly — O(blocks) instead of O(postings). Only adjusts first_doc_id
//! in each block header by the doc offset; deltas within blocks stay unchanged.

use std::io::Write;
use std::mem::size_of;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::OffsetWriter;
use super::SegmentMerger;
use super::doc_offsets;
use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::FieldType;
use crate::segment::reader::SegmentReader;
use crate::segment::types::SegmentFiles;

impl SegmentMerger {
    /// Merge sparse vector indexes using block stacking
    ///
    /// This is O(blocks) instead of O(postings) - we stack blocks directly
    /// and only adjust the first_doc_id in each block header by the doc offset.
    /// Deltas within blocks remain unchanged since they're relative.
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

        // Collect field data: (field_id, quantization, max_dim_id, dim_id -> merged_posting_list)
        type SparseFieldData = (
            u32,
            crate::structures::WeightQuantization,
            u32,
            FxHashMap<u32, Vec<u8>>,
        );
        let mut field_data: Vec<SparseFieldData> = Vec::new();

        for (field, sparse_config) in &sparse_fields {
            // Get quantization from config
            let quantization = sparse_config
                .as_ref()
                .map(|c| c.weight_quantization)
                .unwrap_or(crate::structures::WeightQuantization::Float32);

            // Collect all dimensions across all segments for this field
            let mut all_dims: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
            for segment in segments {
                if let Some(sparse_index) = segment.sparse_indexes().get(&field.0) {
                    for dim_id in sparse_index.active_dimensions() {
                        all_dims.insert(dim_id);
                    }
                }
            }

            if all_dims.is_empty() {
                continue;
            }

            // Bulk-read ALL posting lists per segment in one I/O call each.
            // This replaces 80K+ individual get_posting() calls with ~4 bulk reads.
            let mut segment_postings: Vec<FxHashMap<u32, Arc<BlockSparsePostingList>>> =
                Vec::with_capacity(segments.len());
            for (seg_idx, segment) in segments.iter().enumerate() {
                if let Some(sparse_index) = segment.sparse_indexes().get(&field.0) {
                    log::debug!(
                        "Sparse merge field {}: bulk-reading {} dims from segment {}",
                        field.0,
                        sparse_index.num_dimensions(),
                        seg_idx
                    );
                    let postings = sparse_index.read_all_postings_bulk().await?;
                    segment_postings.push(postings);
                } else {
                    segment_postings.push(FxHashMap::default());
                }
            }

            let mut dim_bytes: FxHashMap<u32, Vec<u8>> = FxHashMap::default();

            // Merge from in-memory data — no I/O in this loop
            for dim_id in all_dims {
                let mut posting_arcs: Vec<(Arc<BlockSparsePostingList>, u32)> = Vec::new();

                for (seg_idx, postings) in segment_postings.iter().enumerate() {
                    if let Some(posting_list) = postings.get(&dim_id) {
                        posting_arcs.push((Arc::clone(posting_list), doc_offs[seg_idx]));
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

                let mut bytes = Vec::new();
                merged.serialize(&mut bytes).map_err(crate::Error::Io)?;
                dim_bytes.insert(dim_id, bytes);
            }

            // Drop bulk data before accumulating output
            drop(segment_postings);

            // Store num_dims (active count) instead of max_dim_id
            field_data.push((field.0, quantization, dim_bytes.len() as u32, dim_bytes));
        }

        if field_data.is_empty() {
            return Ok(0);
        }

        // Sort by field_id
        field_data.sort_by_key(|(id, _, _, _)| *id);

        // Compute header size and per-dimension offsets before writing
        // num_fields(u32) + per-field: field_id(u32) + quant(u8) + num_dims(u32)
        // per-dim: dim_id(u32) + offset(u64) + length(u32)
        let per_dim_entry = size_of::<u32>() + size_of::<u64>() + size_of::<u32>();
        let per_field_header = size_of::<u32>() + size_of::<u8>() + size_of::<u32>();
        let mut header_size = size_of::<u32>() as u64;
        for (_, _, num_dims, _) in &field_data {
            header_size += per_field_header as u64;
            header_size += (*num_dims as u64) * per_dim_entry as u64;
        }

        // Pre-compute offset tables (small — just dim_id + offset + length per dim)
        let mut current_offset = header_size;
        let mut field_tables: Vec<Vec<(u32, u64, u32)>> = Vec::new();
        for (_, _, _, dim_bytes) in &field_data {
            let mut table: Vec<(u32, u64, u32)> = Vec::with_capacity(dim_bytes.len());
            let mut dims: Vec<_> = dim_bytes.keys().copied().collect();
            dims.sort();
            for dim_id in dims {
                let bytes = &dim_bytes[&dim_id];
                table.push((dim_id, current_offset, bytes.len() as u32));
                current_offset += bytes.len() as u64;
            }
            field_tables.push(table);
        }

        // Stream header + tables + data directly to disk
        let mut writer = OffsetWriter::new(dir.streaming_writer(&files.sparse).await?);

        writer.write_u32::<LittleEndian>(field_data.len() as u32)?;
        for (i, (field_id, quantization, num_dims, _)) in field_data.iter().enumerate() {
            writer.write_u32::<LittleEndian>(*field_id)?;
            writer.write_u8(*quantization as u8)?;
            writer.write_u32::<LittleEndian>(*num_dims)?;
            for &(dim_id, offset, length) in &field_tables[i] {
                writer.write_u32::<LittleEndian>(dim_id)?;
                writer.write_u64::<LittleEndian>(offset)?;
                writer.write_u32::<LittleEndian>(length)?;
            }
        }

        // Stream posting data per-field, per-dimension (drop each after writing)
        for (_, _, _, dim_bytes) in field_data {
            let mut dims: Vec<_> = dim_bytes.keys().copied().collect();
            dims.sort();
            for dim_id in dims {
                writer.write_all(&dim_bytes[&dim_id])?;
            }
        }

        let output_size = writer.offset() as usize;
        writer.finish()?;

        log::info!(
            "Sparse vector merge complete: {} fields, {} bytes",
            field_tables.len(),
            output_size
        );

        Ok(output_size)
    }
}
