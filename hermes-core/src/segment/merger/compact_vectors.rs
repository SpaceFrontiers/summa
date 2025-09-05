//! Compaction of vector payloads preserves code bytes and value ordinals.
use super::*;
use crate::segment::format::{
    DenseVectorTocEntry, SparseDimTocEntry, SparseFieldToc, write_dense_toc_and_footer,
    write_sparse_toc_and_footer,
};
use crate::segment::row_map::RowMap;
use crate::segment::{FlatVectorData, VectorIndex};
use std::io::Write;

impl SegmentMerger {
    pub(super) async fn compact_vectors<D: DirectoryWriter>(
        &self,
        dir: &D,
        source: &SegmentReader,
        rows: &RowMap,
        files: &SegmentFiles,
        budget: usize,
    ) -> Result<usize> {
        if source.flat_vectors().is_empty() {
            return Ok(0);
        }
        let mut writer = OffsetWriter::new(dir.streaming_writer_cold(&files.vectors).await?);
        let mut entries = Vec::new();
        let mut fields: Vec<_> = source.flat_vectors().keys().copied().collect();
        fields.sort_unstable();
        for field in fields {
            self.ensure_not_cancelled()?;
            let flat = &source.flat_vectors()[&field];
            let count = (0..flat.num_vectors)
                .filter(|&i| rows.get(flat.get_doc_id(i).0).is_some())
                .count();
            if count == 0 {
                continue;
            }
            if let Some(index) = source.vector_indexes().get(&field) {
                use crate::segment::ann_build::*;
                let ann = match index {
                    VectorIndex::BinaryIvf(index) => Some((index, BINARY_IVF_TYPE)),
                    VectorIndex::Tq { index, .. } => Some((index, TQ_FLAT_TYPE)),
                    VectorIndex::IvfTq { index, .. } => Some((index, IVF_TQ_TYPE)),
                    VectorIndex::ScannAh(index) => Some((index, SCANN_AH_TYPE)),
                    VectorIndex::ScannBinary(index) => Some((index, SCANN_BINARY_TYPE)),
                };
                if let Some((index, index_type)) = ann {
                    let offset = writer.offset();
                    let size = crate::segment::ann_disk::write_live_ann(
                        index.get(),
                        rows,
                        &mut writer,
                        budget,
                        self.cancellation.as_deref(),
                    )?;
                    if size > 0 {
                        entries.push(DenseVectorTocEntry {
                            field_id: field,
                            index_type,
                            offset,
                            size,
                        });
                    }
                }
            }
            let offset = writer.offset();
            FlatVectorData::write_binary_header(flat.dim, count, flat.quantization, &mut writer)?;
            let width = flat.vector_byte_size();
            if width > budget / 4 {
                return Err(crate::Error::Schema(
                    "flat vector exceeds compaction scratch budget".into(),
                ));
            }
            let batch_rows = ((budget / 4).min(4 * 1024 * 1024) / width.max(1)).max(1);
            for start in (0..flat.num_vectors).step_by(batch_rows) {
                self.ensure_not_cancelled()?;
                let end = (start + batch_rows).min(flat.num_vectors);
                let base = flat.vectors_byte_offset();
                let bytes = flat
                    .handle()
                    .read_bytes_range(base + (start * width) as u64..base + (end * width) as u64)
                    .await?;
                for i in start..end {
                    if rows.get(flat.get_doc_id(i).0).is_some() {
                        let local = (i - start) * width;
                        writer.write_all(&bytes.as_slice()[local..local + width])?;
                    }
                }
            }
            for i in 0..flat.num_vectors {
                if i.is_multiple_of(4096) {
                    self.ensure_not_cancelled()?;
                }
                let (doc, ordinal) = flat.get_doc_id(i);
                if let Some(new) = rows.get(doc) {
                    writer.write_all(&new.to_le_bytes())?;
                    writer.write_all(&ordinal.to_le_bytes())?;
                }
            }
            entries.push(DenseVectorTocEntry {
                field_id: field,
                index_type: crate::segment::ann_build::FLAT_TYPE,
                offset,
                size: writer.offset() - offset,
            });
        }
        let offset = writer.offset();
        write_dense_toc_and_footer(&mut writer, offset, &entries)?;
        let bytes = writer.offset() as usize;
        writer.finish()?;
        Ok(bytes)
    }

    pub(super) async fn compact_sparse<D: DirectoryWriter>(
        &self,
        dir: &D,
        source: &SegmentReader,
        rows: &RowMap,
        files: &SegmentFiles,
        budget: usize,
    ) -> Result<usize> {
        if source.sparse_indexes().is_empty() && source.bmp_indexes().is_empty() {
            return Ok(0);
        }
        let mut writer = OffsetWriter::new(dir.streaming_writer_cold(&files.sparse).await?);
        let mut field_tocs = Vec::new();
        let mut skips = Vec::new();
        for (field, entry) in self.schema.fields() {
            if entry.field_type != FieldType::SparseVector {
                continue;
            }
            self.ensure_not_cancelled()?;
            let total_vectors = rows.new_to_old.iter().try_fold(0u32, |total, &old| {
                let count = source.row_stats()[&field.0].get_u64(old);
                u32::try_from(count)
                    .ok()
                    .and_then(|count| total.checked_add(count))
                    .ok_or_else(|| {
                        crate::Error::Corruption("compacted sparse vector count overflow".into())
                    })
            })?;
            if let Some(bmp) = source.bmp_indexes().get(&field.0) {
                let remaining = budget.saturating_sub(
                    skips.len() * std::mem::size_of::<crate::structures::SparseSkipEntry>(),
                );
                let scratch = dir
                    .local_path(&files.sparse)
                    .unwrap_or_else(|| std::env::temp_dir().join(&files.sparse));
                let (next, tocs, _) = crate::segment::reorder::rewrite_bmp_field(
                    &[(bmp.clone(), 0)],
                    field.0,
                    self.schema.index_label(),
                    &entry.name,
                    bmp.dims(),
                    bmp.bmp_block_size as usize,
                    entry
                        .sparse_vector_config
                        .as_ref()
                        .map_or(4, |config| config.bmp_grid_bits),
                    bmp.max_weight_scale,
                    total_vectors,
                    remaining,
                    crate::segment::BpBudget {
                        time_budget: Some(std::time::Duration::ZERO),
                        ..crate::segment::BpBudget::full()
                    },
                    self.cancellation.clone(),
                    crate::segment::reorder::BpGranularity::Records,
                    scratch,
                    bmp.forward().is_some(),
                    writer,
                    field_tocs,
                    self.background_pool.clone(),
                    Some(rows),
                )?;
                writer = next;
                field_tocs = tocs;
                continue;
            }
            let Some(index) = source.sparse_indexes().get(&field.0) else {
                continue;
            };
            let mut dims = Vec::new();
            for (dim, _, blocks) in index.dimension_counts() {
                let start = writer.offset();
                let skip_start = skips.len() as u32;
                let mut doc_count = 0u32;
                let mut previous_doc = None;
                let mut max_weight = 0.0f32;
                for i in 0..blocks {
                    self.ensure_not_cancelled()?;
                    if skips.len().saturating_mul(64) >= budget / 2 {
                        return Err(crate::Error::Schema(
                            "sparse compaction skip directory exceeds scratch budget".into(),
                        ));
                    }
                    let block = index.get_block(dim, i as usize).await?.ok_or_else(|| {
                        crate::Error::Corruption("missing sparse compaction block".into())
                    })?;
                    if let Some(block) = block.remap_documents(|doc| rows.get(doc))? {
                        let at = writer.offset();
                        block.write(&mut writer)?;
                        skips.push(crate::structures::SparseSkipEntry::new(
                            block.header.first_doc_id,
                            block.last_doc_id(),
                            at - start,
                            (writer.offset() - at) as u32,
                            block.header.max_weight,
                        ));
                        for doc in block.decode_doc_ids() {
                            if previous_doc != Some(doc) {
                                doc_count += 1;
                                previous_doc = Some(doc);
                            }
                        }
                        max_weight = max_weight.max(block.header.max_weight);
                    }
                }
                let num_blocks = skips.len() as u32 - skip_start;
                if num_blocks > 0 {
                    dims.push(SparseDimTocEntry {
                        dim_id: dim,
                        block_data_offset: start,
                        skip_start,
                        num_blocks,
                        doc_count,
                        max_weight,
                    });
                }
            }
            if !dims.is_empty() {
                field_tocs.push(SparseFieldToc {
                    field_id: field.0,
                    quantization: entry
                        .sparse_vector_config
                        .as_ref()
                        .map_or(crate::structures::WeightQuantization::Float32, |config| {
                            config.weight_quantization
                        }) as u8,
                    total_vectors,
                    dims,
                });
            }
        }
        let skip_offset = writer.offset();
        for skip in skips {
            skip.write(&mut writer)?;
        }
        let toc_offset = writer.offset();
        write_sparse_toc_and_footer(&mut writer, skip_offset, toc_offset, &field_tocs)?;
        let bytes = writer.offset() as usize;
        writer.finish()?;
        Ok(bytes)
    }
}
