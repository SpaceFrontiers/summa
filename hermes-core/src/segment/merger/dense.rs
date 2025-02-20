//! Dense vector merge strategies
//!
//! Every field always gets a Flat entry (raw vectors for reranking/merge).
//! Optionally, an ANN entry is also written alongside:
//! 1. O(1) cluster merge for homogeneous ANN types (IVF/ScaNN)
//! 2. ANN rebuild with trained structures
//!
//! Raw vectors are read from source segments' lazy flat data (mmap-backed),
//! never from the document store.
//!
//! **Streaming**: ANN blobs are written immediately after serialization and
//! freed, so peak memory is bounded by one ANN blob at a time (not all blobs
//! simultaneously). Doc_id maps are streamed in chunks (384 KB per chunk).

use std::io::Write;

use super::OffsetWriter;
use super::SegmentMerger;
use super::TrainedVectorStructures;
use super::doc_offsets;
use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::{DenseVectorQuantization, FieldType, VectorIndexType};
use crate::segment::format::{DenseVectorTocEntry, write_dense_toc_and_footer};
use crate::segment::reader::SegmentReader;
use crate::segment::types::SegmentFiles;
use crate::segment::vector_data::{FlatVectorData, dequantize_raw};

/// Batch size for streaming vector reads (1024 vectors at a time)
const VECTOR_BATCH_SIZE: usize = 1024;

/// Chunk size for streaming flat vector bytes during merge (8 MB)
const FLAT_VECTOR_CHUNK: u64 = 8 * 1024 * 1024;

/// Chunk size for streaming doc_id+ordinal entries during merge.
/// 64K entries × 6 bytes = 384 KB per chunk (vs 60 MB unbounded).
const DOC_ID_CHUNK: usize = 64 * 1024;

/// Streams vectors from a segment's lazy flat data into an add_fn callback.
///
/// Reads vectors in batches of VECTOR_BATCH_SIZE to bound memory usage.
/// Each batch is a single range read via LazyFileSlice. Vectors are
/// dequantized to f32 regardless of storage quantization (f16, u8, f32).
///
/// Returns number of vectors added.
async fn feed_segment(
    segment: &SegmentReader,
    field: crate::dsl::Field,
    doc_id_offset: u32,
    mut add_fn: impl FnMut(u32, u16, &[f32]),
) -> crate::Result<usize> {
    let lazy_flat = match segment.flat_vectors().get(&field.0) {
        Some(f) => f,
        None => return Ok(0),
    };
    let n = lazy_flat.num_vectors;
    if n == 0 {
        return Ok(0);
    }
    let dim = lazy_flat.dim;
    let quant = lazy_flat.quantization;
    let mut count = 0;
    // Only allocate dequantize buffer for non-f32 quantizations
    let needs_dequant = quant != DenseVectorQuantization::F32;
    let mut f32_buf: Vec<f32> = Vec::new();

    for batch_start in (0..n).step_by(VECTOR_BATCH_SIZE) {
        let batch_count = VECTOR_BATCH_SIZE.min(n - batch_start);
        let batch_bytes = lazy_flat
            .read_vectors_batch(batch_start, batch_count)
            .await
            .map_err(crate::Error::Io)?;
        let raw = batch_bytes.as_slice();
        let batch_floats = batch_count * dim;

        // For f32: reinterpret mmap bytes directly (zero-copy).
        // For f16/u8: dequantize into buffer.
        let vectors: &[f32] = if needs_dequant {
            f32_buf.resize(batch_floats, 0.0);
            dequantize_raw(raw, quant, batch_floats, &mut f32_buf);
            &f32_buf
        } else {
            assert!(
                (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()),
                "f32 vector data not 4-byte aligned"
            );
            // Safety: mmap-backed vector data is page-aligned. Assertion above
            // guards against unexpected misalignment.
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, batch_floats) }
        };

        for i in 0..batch_count {
            let (doc_id, ordinal) = lazy_flat.get_doc_id(batch_start + i);
            add_fn(
                doc_id_offset + doc_id,
                ordinal,
                &vectors[i * dim..(i + 1) * dim],
            );
            count += 1;
        }
    }
    Ok(count)
}

/// Write a Flat entry: header + raw vectors (chunked) + doc_ids (chunked).
///
/// Doc_id map is streamed in DOC_ID_CHUNK entries (384 KB) instead of
/// buffering all at once (was 60 MB for 10M vectors).
async fn write_flat_entry(
    field_id: u32,
    dim: usize,
    total_vectors: usize,
    quantization: DenseVectorQuantization,
    segments: &[SegmentReader],
    doc_offs: &[u32],
    writer: &mut OffsetWriter,
) -> Result<()> {
    FlatVectorData::write_binary_header(dim, total_vectors, quantization, writer)?;

    // Pass 1: stream raw vector bytes in chunks
    for segment in segments {
        if let Some(lazy_flat) = segment.flat_vectors().get(&field_id) {
            let total_bytes = lazy_flat.vector_bytes_len();
            let base_offset = lazy_flat.vectors_byte_offset();
            let handle = lazy_flat.handle();
            for chunk_start in (0..total_bytes).step_by(FLAT_VECTOR_CHUNK as usize) {
                let chunk_end = (chunk_start + FLAT_VECTOR_CHUNK).min(total_bytes);
                let bytes = handle
                    .read_bytes_range(base_offset + chunk_start..base_offset + chunk_end)
                    .await
                    .map_err(crate::Error::Io)?;
                writer.write_all(bytes.as_slice())?;
            }
        }
    }

    // Pass 2: stream doc_ids with offset adjustment (chunked, 384 KB per chunk)
    let mut buf = Vec::with_capacity(DOC_ID_CHUNK * 6);
    for (seg_idx, segment) in segments.iter().enumerate() {
        if let Some(lazy_flat) = segment.flat_vectors().get(&field_id) {
            let offset = doc_offs[seg_idx];
            let count = lazy_flat.num_vectors;
            for chunk_start in (0..count).step_by(DOC_ID_CHUNK) {
                buf.clear();
                let chunk_end = (chunk_start + DOC_ID_CHUNK).min(count);
                for i in chunk_start..chunk_end {
                    let (doc_id, ordinal) = lazy_flat.get_doc_id(i);
                    buf.extend_from_slice(&(offset + doc_id).to_le_bytes());
                    buf.extend_from_slice(&ordinal.to_le_bytes());
                }
                writer.write_all(&buf)?;
            }
        }
    }

    Ok(())
}

impl SegmentMerger {
    /// Merge dense vector indexes - returns output size in bytes
    ///
    /// Single-pass streaming: for each field, the ANN blob (if any) is built,
    /// serialized, written to disk, and freed immediately. Then the Flat entry
    /// is streamed. Peak memory = one ANN blob at a time, not all simultaneously.
    ///
    /// No document store reads — all vector data comes from .vectors files.
    pub(super) async fn merge_dense_vectors<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
        trained: Option<&TrainedVectorStructures>,
    ) -> Result<usize> {
        let doc_offs = doc_offsets(segments)?;

        // Collect fields that need writing (in schema field_id order)
        struct FieldInfo {
            field: crate::dsl::Field,
            dim: usize,
            total_vectors: usize,
            quantization: DenseVectorQuantization,
        }
        let mut fields_to_write: Vec<FieldInfo> = Vec::new();

        for (field, entry) in self.schema.fields() {
            if !matches!(entry.field_type, FieldType::DenseVector)
                || !(entry.indexed || entry.stored)
            {
                continue;
            }

            let dim: usize = segments
                .iter()
                .filter_map(|s| s.flat_vectors().get(&field.0).map(|f| f.dim))
                .find(|&d| d > 0)
                .unwrap_or(0);
            if dim == 0 {
                continue;
            }

            let total_vectors: usize = segments
                .iter()
                .filter_map(|s| s.flat_vectors().get(&field.0).map(|f| f.num_vectors))
                .sum();
            if total_vectors == 0 {
                continue;
            }

            let quantization = entry
                .dense_vector_config
                .as_ref()
                .map(|c| c.quantization)
                .or_else(|| {
                    segments
                        .iter()
                        .find_map(|s| s.flat_vectors().get(&field.0).map(|f| f.quantization))
                })
                .unwrap_or(DenseVectorQuantization::F32);

            fields_to_write.push(FieldInfo {
                field,
                dim,
                total_vectors,
                quantization,
            });
        }

        if fields_to_write.is_empty() {
            return Ok(0);
        }

        let write_start = std::time::Instant::now();
        let mut writer = OffsetWriter::new(dir.streaming_writer(&files.vectors).await?);
        let mut toc: Vec<DenseVectorTocEntry> = Vec::new();

        for fi in &fields_to_write {
            let field = fi.field;
            let entry = self.schema.get_field_entry(field).unwrap();
            let config = entry.dense_vector_config.as_ref();

            // ── ANN entry (written first, index_type < FLAT_TYPE) ────────
            let ann_blob = self
                .try_build_ann(field, config, segments, &doc_offs, trained)
                .await?;

            if let Some((index_type, bytes)) = ann_blob {
                let data_offset = writer.offset();
                writer.write_all(&bytes)?;
                let data_size = writer.offset() - data_offset;
                toc.push(DenseVectorTocEntry {
                    field_id: field.0,
                    index_type,
                    offset: data_offset,
                    size: data_size,
                });
                // Pad to 8-byte boundary
                let pad = (8 - (writer.offset() % 8)) % 8;
                if pad > 0 {
                    writer.write_all(&[0u8; 8][..pad as usize])?;
                }
                // `bytes` dropped here — frees the ANN blob immediately
            }

            // ── Flat entry (always written, index_type = FLAT_TYPE) ──────
            let data_offset = writer.offset();
            write_flat_entry(
                field.0,
                fi.dim,
                fi.total_vectors,
                fi.quantization,
                segments,
                &doc_offs,
                &mut writer,
            )
            .await?;
            let data_size = writer.offset() - data_offset;
            toc.push(DenseVectorTocEntry {
                field_id: field.0,
                index_type: crate::segment::ann_build::FLAT_TYPE,
                offset: data_offset,
                size: data_size,
            });
            // Pad to 8-byte boundary
            let pad = (8 - (writer.offset() % 8)) % 8;
            if pad > 0 {
                writer.write_all(&[0u8; 8][..pad as usize])?;
            }
        }

        // Write TOC + footer
        let toc_offset = writer.offset();
        write_dense_toc_and_footer(&mut writer, toc_offset, &toc)?;

        let output_size = writer.offset() as usize;
        writer.finish()?;
        log::info!(
            "[merge_vectors] file written: {} ({} entries) in {:.1}s",
            super::format_bytes(output_size),
            toc.len(),
            write_start.elapsed().as_secs_f64()
        );
        Ok(output_size)
    }

    /// Try to build an ANN index for a field. Returns (index_type, serialized_bytes)
    /// or None if no ANN path is available.
    ///
    /// Tries in order: O(1) ScaNN merge → O(1) IVF merge → rebuild from trained.
    async fn try_build_ann(
        &self,
        field: crate::dsl::Field,
        config: Option<&crate::dsl::DenseVectorConfig>,
        segments: &[SegmentReader],
        doc_offs: &[u32],
        trained: Option<&TrainedVectorStructures>,
    ) -> Result<Option<(u8, Vec<u8>)>> {
        let segments_with_flat = segments
            .iter()
            .filter(|s| s.flat_vectors().contains_key(&field.0))
            .count();

        // --- Try O(1) cluster merge for homogeneous ScaNN ---
        let scann_indexes: Vec<_> = segments
            .iter()
            .filter_map(|s| s.get_scann_vector_index(field))
            .collect();

        if scann_indexes.len() == segments_with_flat && !scann_indexes.is_empty() {
            let refs: Vec<&crate::structures::IVFPQIndex> =
                scann_indexes.iter().map(|(idx, _)| idx.as_ref()).collect();
            let codebook = scann_indexes.first().map(|(_, cb)| cb);

            match (
                crate::structures::IVFPQIndex::merge(&refs, doc_offs),
                codebook,
            ) {
                (Ok(merged), Some(codebook)) => {
                    let index_data = crate::segment::ScaNNIndexData {
                        codebook: (**codebook).clone(),
                        index: merged,
                    };
                    let bytes = index_data
                        .to_bytes()
                        .map_err(|e| crate::Error::Serialization(e.to_string()))?;
                    return Ok(Some((crate::segment::ann_build::SCANN_TYPE, bytes)));
                }
                (Err(e), _) => {
                    log::warn!("ScaNN merge failed: {}, falling back to rebuild", e);
                }
                (_, None) => {
                    log::warn!("ScaNN merge: missing codebook, falling back to rebuild");
                }
            }
        }

        // --- Try O(1) cluster merge for homogeneous IVF-RaBitQ ---
        let ivf_indexes: Vec<_> = segments
            .iter()
            .filter_map(|s| s.get_ivf_vector_index(field))
            .collect();

        if ivf_indexes.len() == segments_with_flat && !ivf_indexes.is_empty() {
            let refs: Vec<&crate::structures::IVFRaBitQIndex> =
                ivf_indexes.iter().map(|(idx, _)| idx.as_ref()).collect();
            let codebook = ivf_indexes.first().map(|(_, cb)| cb);

            match (
                crate::structures::IVFRaBitQIndex::merge(&refs, doc_offs),
                codebook,
            ) {
                (Ok(merged), Some(codebook)) => {
                    let index_data = crate::segment::IVFRaBitQIndexData {
                        codebook: (**codebook).clone(),
                        index: merged,
                    };
                    let bytes = index_data
                        .to_bytes()
                        .map_err(|e| crate::Error::Serialization(e.to_string()))?;
                    return Ok(Some((crate::segment::ann_build::IVF_RABITQ_TYPE, bytes)));
                }
                (Err(e), _) => {
                    log::warn!("IVF merge failed: {}, falling back to rebuild", e);
                }
                (_, None) => {
                    log::warn!("IVF merge: missing codebook, falling back to rebuild");
                }
            }
        }

        // --- Try ANN rebuild from lazy flat vectors using trained structures ---
        let ann_type = trained
            .zip(config)
            .and_then(|(trained, config)| match config.index_type {
                VectorIndexType::IvfRaBitQ if trained.centroids.contains_key(&field.0) => {
                    Some(VectorIndexType::IvfRaBitQ)
                }
                VectorIndexType::ScaNN
                    if trained.centroids.contains_key(&field.0)
                        && trained.codebooks.contains_key(&field.0) =>
                {
                    Some(VectorIndexType::ScaNN)
                }
                _ => None,
            });

        let ann = match ann_type {
            Some(ann) => ann,
            None => {
                log::debug!(
                    "[merge_vectors] field {}: no ANN path available (trained={}, config={}, ivf={}/{}, scann={}/{})",
                    field.0,
                    trained.is_some(),
                    config
                        .map(|c| format!("{:?}", c.index_type))
                        .unwrap_or_else(|| "None".into()),
                    ivf_indexes.len(),
                    segments_with_flat,
                    scann_indexes.len(),
                    segments_with_flat,
                );
                return Ok(None);
            }
        };

        let trained = trained.unwrap();
        let dim = segments
            .iter()
            .filter_map(|s| s.flat_vectors().get(&field.0).map(|f| f.dim))
            .find(|&d| d > 0)
            .unwrap();
        let mut total_fed = 0usize;
        let ann_start = std::time::Instant::now();

        let (index_type, bytes) = match ann {
            VectorIndexType::Flat | VectorIndexType::RaBitQ => unreachable!(),
            VectorIndexType::IvfRaBitQ => {
                let centroids = &trained.centroids[&field.0];
                let (mut index, codebook) =
                    crate::segment::ann_build::new_ivf_rabitq(dim, centroids);

                for (seg_idx, segment) in segments.iter().enumerate() {
                    let offset = doc_offs[seg_idx];
                    let fed = feed_segment(segment, field, offset, |doc_id, ordinal, vec| {
                        index.add_vector(centroids, &codebook, doc_id, ordinal, vec);
                    })
                    .await?;
                    total_fed += fed;
                    if fed > 0 {
                        log::debug!(
                            "[merge_vectors] field {} IVF: fed {} vectors from segment {} ({} total, {:.1}s)",
                            field.0,
                            fed,
                            seg_idx,
                            total_fed,
                            ann_start.elapsed().as_secs_f64()
                        );
                    }
                }

                log::info!(
                    "[merge_vectors] field {} IVF: serializing index ({} vectors, {:.1}s elapsed)",
                    field.0,
                    total_fed,
                    ann_start.elapsed().as_secs_f64()
                );
                let bytes = crate::segment::ann_build::serialize_ivf_rabitq(index, codebook)?;
                (crate::segment::ann_build::IVF_RABITQ_TYPE, bytes)
            }
            VectorIndexType::ScaNN => {
                let centroids = &trained.centroids[&field.0];
                let codebook = &trained.codebooks[&field.0];
                let mut index = crate::segment::ann_build::new_scann(dim, centroids, codebook);

                for (seg_idx, segment) in segments.iter().enumerate() {
                    let offset = doc_offs[seg_idx];
                    let fed = feed_segment(segment, field, offset, |doc_id, ordinal, vec| {
                        index.add_vector(centroids, codebook, doc_id, ordinal, vec);
                    })
                    .await?;
                    total_fed += fed;
                    if fed > 0 {
                        log::debug!(
                            "[merge_vectors] field {} ScaNN: fed {} vectors from segment {} ({} total, {:.1}s)",
                            field.0,
                            fed,
                            seg_idx,
                            total_fed,
                            ann_start.elapsed().as_secs_f64()
                        );
                    }
                }

                log::info!(
                    "[merge_vectors] field {} ScaNN: serializing index ({} vectors, {:.1}s elapsed)",
                    field.0,
                    total_fed,
                    ann_start.elapsed().as_secs_f64()
                );
                let bytes = crate::segment::ann_build::serialize_scann(index, codebook)?;
                (crate::segment::ann_build::SCANN_TYPE, bytes)
            }
        };

        log::info!(
            "[merge_vectors] field {} ANN(type={}) rebuilt: {} vectors, blob={}, {:.1}s",
            field.0,
            index_type,
            total_fed,
            super::format_bytes(bytes.len()),
            ann_start.elapsed().as_secs_f64()
        );
        Ok(Some((index_type, bytes)))
    }
}
