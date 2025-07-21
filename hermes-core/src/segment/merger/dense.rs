//! Dense vector merge strategies
//!
//! Every field always gets a Flat entry (raw vectors for reranking/merge).
//! Optionally, an ANN entry is also written alongside. Ordinary merges copy
//! immutable ANN runs byte-for-byte; vector-generation rewrites explicitly
//! rebuild against newly trained global artifacts.
//!
//! Raw vectors are read from source segments' lazy flat data (mmap-backed),
//! never from the document store.
//!
//! **Streaming**: normal ANN merge payloads and flat vectors are copied in
//! bounded chunks. Rebuilds retain only one field's compressed ANN output;
//! doc-ID maps are streamed in chunks (384 KB per chunk).

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AnnWriteMode {
    /// Ordinary segment merge: require compatible ANN payloads and copy their
    /// immutable run columns. Missing/mismatched data is corruption.
    Copy,
    /// Vector-generation rewrite: rebuild payloads from flat vectors against
    /// the explicitly supplied new global artifacts.
    Rebuild,
}

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
    mut add_batch: impl FnMut(&[(u32, u16)], &[f32]) -> Result<()>,
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
    let mut labels = Vec::with_capacity(VECTOR_BATCH_SIZE);

    for batch_start in (0..n).step_by(VECTOR_BATCH_SIZE) {
        let batch_count = VECTOR_BATCH_SIZE.min(n - batch_start);
        let batch_bytes = lazy_flat
            .read_vectors_batch(batch_start, batch_count)
            .await
            .map_err(crate::Error::Io)?;
        let raw = batch_bytes.as_slice();
        let batch_floats = batch_count.checked_mul(dim).ok_or_else(|| {
            crate::Error::Corruption("dense merge batch size overflows usize".into())
        })?;

        // For f32: reinterpret mmap bytes directly (zero-copy).
        // For f16/u8: dequantize into buffer.
        let vectors: &[f32] = if needs_dequant {
            f32_buf.resize(batch_floats, 0.0);
            dequantize_raw(raw, quant, batch_floats, &mut f32_buf).map_err(crate::Error::Io)?;
            &f32_buf
        } else {
            if !(raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
                return Err(crate::Error::Corruption(
                    "f32 flat vector data is not 4-byte aligned".into(),
                ));
            }
            // Safety: mmap-backed vector data is page-aligned. Assertion above
            // guards against unexpected misalignment.
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, batch_floats) }
        };

        labels.clear();
        for i in 0..batch_count {
            let (doc_id, ordinal) = lazy_flat.get_doc_id(batch_start + i);
            labels.push((doc_id_offset + doc_id, ordinal));
        }
        add_batch(&labels, vectors)?;
        count += batch_count;
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
                let chunk_end = chunk_start
                    .saturating_add(FLAT_VECTOR_CHUNK)
                    .min(total_bytes);
                let range_start = base_offset.checked_add(chunk_start).ok_or_else(|| {
                    crate::Error::Corruption("flat vector source offset exceeds u64".into())
                })?;
                let range_end = base_offset.checked_add(chunk_end).ok_or_else(|| {
                    crate::Error::Corruption("flat vector source range exceeds u64".into())
                })?;
                let bytes = handle
                    .read_bytes_range(range_start..range_end)
                    .await
                    .map_err(crate::Error::Io)?;
                let expected_len = usize::try_from(chunk_end - chunk_start).map_err(|_| {
                    crate::Error::Corruption("flat vector merge chunk exceeds usize".into())
                })?;
                if bytes.len() != expected_len {
                    return Err(crate::Error::Corruption(format!(
                        "flat vector merge read returned {} bytes, expected {expected_len}",
                        bytes.len()
                    )));
                }
                super::block_in_place_if_multithread(|| writer.write_all(bytes.as_slice()))?;
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
                super::block_in_place_if_multithread(|| writer.write_all(&buf))?;
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
    pub(crate) async fn merge_dense_vectors<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
        trained: Option<&TrainedVectorStructures>,
        ann_mode: AnnWriteMode,
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
            if !matches!(
                entry.field_type,
                FieldType::DenseVector | FieldType::BinaryDenseVector
            ) || !(entry.indexed || entry.stored)
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

            let total_vectors = segments
                .iter()
                .filter_map(|s| s.flat_vectors().get(&field.0).map(|f| f.num_vectors))
                .try_fold(0usize, |total, count| total.checked_add(count))
                .ok_or_else(|| {
                    crate::Error::Corruption(format!(
                        "flat vector count overflows usize for field {}",
                        field.0
                    ))
                })?;
            if total_vectors == 0 {
                continue;
            }

            let quantization = if entry.field_type == FieldType::BinaryDenseVector {
                DenseVectorQuantization::Binary
            } else {
                entry
                    .dense_vector_config
                    .as_ref()
                    .map(|c| c.quantization)
                    .or_else(|| {
                        segments
                            .iter()
                            .find_map(|s| s.flat_vectors().get(&field.0).map(|f| f.quantization))
                    })
                    .unwrap_or(DenseVectorQuantization::F32)
            };

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
        let mut writer = OffsetWriter::new(dir.streaming_writer_cold(&files.vectors).await?);
        let mut toc: Vec<DenseVectorTocEntry> = Vec::new();

        for fi in &fields_to_write {
            let field = fi.field;
            let entry = self.schema.get_field_entry(field).unwrap();
            let config = entry.dense_vector_config.as_ref();

            // ── ANN entry (written first, index_type != FLAT_TYPE) ───────
            let tq_config = config.filter(|config| {
                entry.field_type == FieldType::DenseVector
                    && config.index_type == VectorIndexType::Tq
            });
            let ann_entry = if let Some(tq_config) = tq_config {
                // TQ payloads carry no trained generation: every merge mode
                // byte-copies compatible runs; only sources without a payload
                // (pre-TQ builds or a schema switch to `tq`) are upgraded by
                // re-encoding from their flat vectors.
                self.write_tq_entry(field, tq_config, segments, &doc_offs, &mut writer)
                    .await?
            } else {
                match ann_mode {
                    AnnWriteMode::Copy => {
                        self.copy_ann_runs(field, entry, segments, &doc_offs, trained, &mut writer)?
                    }
                    AnnWriteMode::Rebuild if entry.field_type == FieldType::BinaryDenseVector => {
                        if let Some(index) = self
                            .rebuild_binary_ivf(field, entry, segments, &doc_offs, trained)
                            .await?
                        {
                            let data_offset = writer.offset();
                            let routing = entry
                                .binary_dense_vector_config
                                .as_ref()
                                .expect("binary field configuration validated")
                                .ivf_routing;
                            super::block_in_place_if_multithread(|| {
                                crate::segment::ann_disk::write_built_binary_ivf(
                                    &index,
                                    routing,
                                    &mut writer,
                                )
                            })
                            .map_err(crate::Error::Io)?;
                            Some((
                                crate::segment::ann_build::BINARY_IVF_TYPE,
                                data_offset,
                                writer.offset() - data_offset,
                            ))
                        } else {
                            None
                        }
                    }
                    AnnWriteMode::Rebuild => {
                        if let Some((index, num_clusters)) = self
                            .rebuild_float_ivf(field, config, segments, &doc_offs, trained)
                            .await?
                        {
                            let data_offset = writer.offset();
                            super::block_in_place_if_multithread(|| {
                                crate::segment::ann_disk::write_built_ivf_pq(
                                    &index,
                                    num_clusters,
                                    &mut writer,
                                )
                            })
                            .map_err(crate::Error::Io)?;
                            Some((
                                crate::segment::ann_build::IVF_PQ_TYPE,
                                data_offset,
                                writer.offset() - data_offset,
                            ))
                        } else {
                            None
                        }
                    }
                }
            };
            if let Some((index_type, data_offset, data_size)) = ann_entry {
                toc.push(DenseVectorTocEntry {
                    field_id: field.0,
                    index_type,
                    offset: data_offset,
                    size: data_size,
                });
                let pad = (8 - (writer.offset() % 8)) % 8;
                if pad > 0 {
                    super::block_in_place_if_multithread(|| {
                        writer.write_all(&[0u8; 8][..pad as usize])
                    })?;
                }
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
                super::block_in_place_if_multithread(|| {
                    writer.write_all(&[0u8; 8][..pad as usize])
                })?;
            }
        }

        // Write TOC + footer
        let toc_offset = writer.offset();
        write_dense_toc_and_footer(&mut writer, toc_offset, &toc)?;

        let output_size = writer.offset() as usize;
        super::block_in_place_if_multithread(move || writer.finish())?;
        log::info!(
            "[dense_vector_merge] file written: {} ({} fields) in {:.1}s",
            crate::format_bytes(output_size as u64),
            toc.len(),
            write_start.elapsed().as_secs_f64()
        );
        Ok(output_size)
    }

    /// Copy compatible ANN run columns directly from source mmaps. This is the
    /// only ANN path used by ordinary segment merges.
    fn copy_ann_runs(
        &self,
        field: crate::dsl::Field,
        entry: &crate::dsl::FieldEntry,
        segments: &[SegmentReader],
        doc_offs: &[u32],
        trained: Option<&TrainedVectorStructures>,
        writer: &mut OffsetWriter,
    ) -> Result<Option<(u8, u64, u64)>> {
        let has_source_ann = segments
            .iter()
            .any(|segment| segment.vector_indexes().contains_key(&field.0));
        let Some(trained) = trained else {
            if has_source_ann {
                return Err(crate::Error::Corruption(format!(
                    "ANN field {} has segment payloads but no loaded global artifacts",
                    field.0,
                )));
            }
            return Ok(None);
        };

        let mut sources = Vec::new();
        let index_type = match entry.field_type {
            FieldType::DenseVector => {
                let Some(config) = entry
                    .dense_vector_config
                    .as_ref()
                    .filter(|config| config.index_type == VectorIndexType::IvfPq)
                else {
                    if has_source_ann {
                        return Err(crate::Error::Corruption(format!(
                            "flat dense field {} unexpectedly contains ANN payloads",
                            field.0,
                        )));
                    }
                    return Ok(None);
                };
                let Some(centroids) = trained.centroids.get(&field.0) else {
                    if has_source_ann {
                        return Err(crate::Error::Corruption(format!(
                            "IVF-PQ field {} has payloads but no global centroids",
                            field.0,
                        )));
                    }
                    return Ok(None);
                };
                let Some(codebook) = trained.codebooks.get(&field.0) else {
                    if has_source_ann {
                        return Err(crate::Error::Corruption(format!(
                            "IVF-PQ field {} has payloads but no global codebook",
                            field.0,
                        )));
                    }
                    return Ok(None);
                };
                let max_assignments = centroids
                    .soar_config
                    .as_ref()
                    .map_or(1, |soar| 1usize.saturating_add(soar.num_secondary));
                for (segment_index, segment) in segments.iter().enumerate() {
                    let Some(flat) = segment.flat_vectors().get(&field.0) else {
                        continue;
                    };
                    let Some(crate::segment::VectorIndex::IvfPq(index)) =
                        segment.vector_indexes().get(&field.0)
                    else {
                        return Err(crate::Error::Corruption(format!(
                            "ordinary merge source {:032x} field {} is missing its IVF-PQ payload",
                            segment.meta().id,
                            field.0,
                        )));
                    };
                    let disk = index.get();
                    let header = disk.header();
                    let max_vectors =
                        flat.num_vectors
                            .checked_mul(max_assignments)
                            .ok_or_else(|| {
                                crate::Error::Corruption(format!(
                                    "IVF-PQ assignment count overflows for field {}",
                                    field.0,
                                ))
                            })?;
                    if header.dim != config.dim
                        || header.code_size != codebook.config.num_subspaces
                        || header.num_clusters != centroids.num_clusters
                        || header.quantizer_version != centroids.version
                        || header.codebook_version != codebook.version
                        || header.routing != config.ivf_routing
                        || !(flat.num_vectors..=max_vectors).contains(&header.vector_count)
                    {
                        return Err(crate::Error::Corruption(format!(
                            "ordinary merge source {:032x} field {} uses an incompatible IVF-PQ generation",
                            segment.meta().id,
                            field.0,
                        )));
                    }
                    sources.push((disk, doc_offs[segment_index]));
                }
                crate::segment::ann_build::IVF_PQ_TYPE
            }
            FieldType::BinaryDenseVector => {
                let Some(config) = entry
                    .binary_dense_vector_config
                    .as_ref()
                    .filter(|config| config.index_type == crate::dsl::BinaryIndexType::Ivf)
                else {
                    if has_source_ann {
                        return Err(crate::Error::Corruption(format!(
                            "flat binary field {} unexpectedly contains ANN payloads",
                            field.0,
                        )));
                    }
                    return Ok(None);
                };
                let Some(quantizer) = trained.binary_quantizers.get(&field.0) else {
                    if has_source_ann {
                        return Err(crate::Error::Corruption(format!(
                            "binary IVF field {} has payloads but no global quantizer",
                            field.0,
                        )));
                    }
                    return Ok(None);
                };
                for (segment_index, segment) in segments.iter().enumerate() {
                    let Some(flat) = segment.flat_vectors().get(&field.0) else {
                        continue;
                    };
                    let Some(crate::segment::VectorIndex::BinaryIvf(index)) =
                        segment.vector_indexes().get(&field.0)
                    else {
                        return Err(crate::Error::Corruption(format!(
                            "ordinary merge source {:032x} field {} is missing its binary IVF payload",
                            segment.meta().id,
                            field.0,
                        )));
                    };
                    let disk = index.get();
                    let header = disk.header();
                    if header.dim != config.dim
                        || header.code_size != config.byte_len()
                        || header.num_clusters != quantizer.num_clusters
                        || header.quantizer_version != quantizer.version
                        || header.codebook_version != 0
                        || header.routing != config.ivf_routing
                        || header.vector_count != flat.num_vectors
                    {
                        return Err(crate::Error::Corruption(format!(
                            "ordinary merge source {:032x} field {} uses an incompatible binary IVF generation",
                            segment.meta().id,
                            field.0,
                        )));
                    }
                    sources.push((disk, doc_offs[segment_index]));
                }
                crate::segment::ann_build::BINARY_IVF_TYPE
            }
            _ => return Ok(None),
        };

        if sources.is_empty() {
            return Ok(None);
        }
        let data_offset = writer.offset();
        super::block_in_place_if_multithread(|| {
            crate::segment::ann_disk::write_merged_ann(&sources, writer)
        })
        .map_err(crate::Error::Io)?;
        let data_size = writer.offset() - data_offset;
        log::debug!(
            "[dense_vector_merge] field {}: copied {} compatible ANN run source(s), {}",
            field.0,
            sources.len(),
            crate::format_bytes(data_size),
        );
        Ok(Some((index_type, data_offset, data_size)))
    }

    /// Write the merged TQ payload for one field.
    ///
    /// The normal path is the same pure byte-copy as every other ANN merge:
    /// TQ payloads have no trained generation, so compatibility is only the
    /// codec fingerprint. Sources without a payload (segments built before
    /// the field used `tq`) are upgraded by re-encoding from flat storage —
    /// loudly, and only for this field.
    async fn write_tq_entry(
        &self,
        field: crate::dsl::Field,
        config: &crate::dsl::DenseVectorConfig,
        segments: &[SegmentReader],
        doc_offs: &[u32],
        writer: &mut OffsetWriter,
    ) -> Result<Option<(u8, u64, u64)>> {
        let expected_fingerprint =
            crate::structures::vector::quantization::tq_expected_fingerprint(config.dim);
        let mut sources = Vec::new();
        let mut segments_with_vectors = 0usize;
        let mut missing_payloads = 0usize;
        for (segment_index, segment) in segments.iter().enumerate() {
            let Some(flat) = segment.flat_vectors().get(&field.0) else {
                continue;
            };
            segments_with_vectors += 1;
            match segment.vector_indexes().get(&field.0) {
                Some(crate::segment::VectorIndex::Tq { index, .. }) => {
                    let header = index.get().header();
                    if header.dim != config.dim
                        || header.quantizer_version != expected_fingerprint
                        || header.vector_count != flat.num_vectors
                    {
                        return Err(crate::Error::Corruption(format!(
                            "merge source {:032x} field {} carries an incompatible TQ payload \
                             (dim {} fingerprint {:#x} count {}, expected dim {} fingerprint \
                             {:#x} count {})",
                            segment.meta().id,
                            field.0,
                            header.dim,
                            header.quantizer_version,
                            header.vector_count,
                            config.dim,
                            expected_fingerprint,
                            flat.num_vectors,
                        )));
                    }
                    sources.push((index.get(), doc_offs[segment_index]));
                }
                Some(_) => {
                    return Err(crate::Error::Corruption(format!(
                        "merge source {:032x} field {} has a non-TQ ANN payload for a tq field",
                        segment.meta().id,
                        field.0,
                    )));
                }
                None => missing_payloads += 1,
            }
        }
        if segments_with_vectors == 0 {
            return Ok(None);
        }

        if missing_payloads == 0 {
            let data_offset = writer.offset();
            super::block_in_place_if_multithread(|| {
                crate::segment::ann_disk::write_merged_ann(&sources, writer)
            })
            .map_err(crate::Error::Io)?;
            let data_size = writer.offset() - data_offset;
            log::debug!(
                "[merge_vectors] field {}: copied {} compatible TQ run source(s), {} bytes",
                field.0,
                sources.len(),
                data_size,
            );
            return Ok(Some((
                crate::segment::ann_build::TQ_FLAT_TYPE,
                data_offset,
                data_size,
            )));
        }

        log::info!(
            "[merge_vectors] field {}: {}/{} TQ source(s) have no payload; re-encoding from \
             flat vectors (training-free)",
            field.0,
            missing_payloads,
            segments_with_vectors,
        );
        drop(sources);
        let codec = std::sync::Arc::new(crate::structures::TqCodec::new(config.dim));
        let mut builder = crate::structures::TqFlatBuilder::new(codec);
        let encode_start = std::time::Instant::now();
        for (segment_index, segment) in segments.iter().enumerate() {
            feed_segment(
                segment,
                field,
                doc_offs[segment_index],
                |labels, vectors| {
                    super::block_in_place_if_multithread(|| {
                        let mut add = || builder.add_batch(labels, vectors);
                        if let Some(pool) = &self.background_pool {
                            pool.install(add)
                        } else {
                            add()
                        }
                    })
                    .map_err(|error| {
                        crate::Error::Internal(format!(
                            "TQ re-encode failed for field {}: {error}",
                            field.0,
                        ))
                    })
                },
            )
            .await?;
        }
        if builder.is_empty() {
            log::warn!(
                "[merge_vectors] field {}: TQ re-encode found no vectors in any source; \
                 the merged segment will carry no TQ payload and fall back to exact scan",
                field.0,
            );
            return Ok(None);
        }
        builder.finish();
        let vector_count = builder.len();
        let data_offset = writer.offset();
        super::block_in_place_if_multithread(|| {
            crate::segment::ann_disk::write_built_tq_flat(&builder, writer)
        })
        .map_err(crate::Error::Io)?;
        log::info!(
            "[merge_vectors] field {}: TQ re-encoded {} vectors in {:.1}s",
            field.0,
            vector_count,
            encode_start.elapsed().as_secs_f64(),
        );
        Ok(Some((
            crate::segment::ann_build::TQ_FLAT_TYPE,
            data_offset,
            writer.offset() - data_offset,
        )))
    }

    /// Rebuild the binary IVF index for a vector-generation rewrite.
    ///
    /// Streams packed codes from source flat storage and assigns every vector
    /// to the already-trained global quantizer generation. Codes are exact, so
    /// rebuilding the per-segment run payload is lossless.
    async fn rebuild_binary_ivf(
        &self,
        field: crate::dsl::Field,
        entry: &crate::dsl::FieldEntry,
        segments: &[SegmentReader],
        doc_offs: &[u32],
        trained: Option<&TrainedVectorStructures>,
    ) -> Result<Option<crate::structures::BinaryIvfIndex>> {
        let Some(cfg) = entry.binary_dense_vector_config.as_ref() else {
            return Ok(None);
        };
        if cfg.index_type != crate::dsl::BinaryIndexType::Ivf {
            return Ok(None);
        }
        let Some(quantizer) = trained.and_then(|trained| trained.binary_quantizers.get(&field.0))
        else {
            return Ok(None);
        };

        const CODE_BATCH: usize = 65536;
        let mut builder =
            crate::structures::vector::index::BinaryIvfBuilder::new(quantizer, cfg.ivf_routing)
                .map_err(crate::Error::Io)?;
        let mut labels = Vec::with_capacity(CODE_BATCH);
        let mut vector_count = 0usize;
        for (seg_idx, segment) in segments.iter().enumerate() {
            let Some(lazy_flat) = segment.flat_vectors().get(&field.0) else {
                continue;
            };
            let offset = doc_offs[seg_idx];
            let n = lazy_flat.num_vectors;
            for batch_start in (0..n).step_by(CODE_BATCH) {
                let batch_count = CODE_BATCH.min(n - batch_start);
                let bytes = lazy_flat
                    .read_vectors_batch(batch_start, batch_count)
                    .await
                    .map_err(crate::Error::Io)?;
                labels.clear();
                for i in 0..batch_count {
                    let (doc_id, ordinal) = lazy_flat.get_doc_id(batch_start + i);
                    let merged_doc_id = doc_id.checked_add(offset).ok_or_else(|| {
                        crate::Error::Corruption(format!(
                            "binary IVF doc-id offset overflow: {doc_id} + {offset}"
                        ))
                    })?;
                    labels.push((merged_doc_id, ordinal));
                }
                super::block_in_place_if_multithread(|| {
                    let mut add = || builder.add_batch(quantizer, bytes.as_slice(), &labels);
                    if let Some(pool) = &self.background_pool {
                        pool.install(add)
                    } else {
                        add()
                    }
                })
                .map_err(crate::Error::Io)?;
                vector_count = vector_count.checked_add(batch_count).ok_or_else(|| {
                    crate::Error::Corruption(format!(
                        "binary IVF rebuild vector count overflows for field {}",
                        field.0,
                    ))
                })?;
            }
        }

        if vector_count == 0 {
            return Ok(None);
        }
        let num_clusters = quantizer.num_clusters;
        let index = builder.finish().map_err(crate::Error::Io)?;
        log::debug!(
            "[dense_vector_merge] field {}: binary IVF rebuilt ({} vectors, {} clusters, estimated {})",
            field.0,
            vector_count,
            num_clusters,
            crate::format_bytes(index.estimated_memory_bytes() as u64),
        );
        Ok(Some(index))
    }

    async fn rebuild_float_ivf(
        &self,
        field: crate::dsl::Field,
        config: Option<&crate::dsl::DenseVectorConfig>,
        segments: &[SegmentReader],
        doc_offs: &[u32],
        trained: Option<&TrainedVectorStructures>,
    ) -> Result<Option<(crate::structures::IVFPQIndex, u32)>> {
        let Some(config) = config.filter(|config| config.index_type == VectorIndexType::IvfPq)
        else {
            return Ok(None);
        };
        let Some(trained) = trained else {
            return Ok(None);
        };
        let Some(centroids) = trained.centroids.get(&field.0) else {
            return Ok(None);
        };
        let Some(codebook) = trained.codebooks.get(&field.0) else {
            return Ok(None);
        };

        let mut total_fed = 0usize;
        let ann_start = std::time::Instant::now();
        let mut index = crate::segment::ann_build::new_ivf_pq(
            config.dim,
            config.ivf_routing,
            centroids,
            codebook,
        );

        for (segment_index, segment) in segments.iter().enumerate() {
            let offset = doc_offs[segment_index];
            let fed = feed_segment(segment, field, offset, |labels, vectors| {
                super::block_in_place_if_multithread(|| {
                    let mut add =
                        || index.add_vectors_parallel(centroids, codebook, labels, vectors);
                    if let Some(pool) = &self.background_pool {
                        pool.install(add)
                    } else {
                        add()
                    }
                })
                .map_err(|error| {
                    crate::Error::Internal(format!(
                        "parallel IVF-PQ rebuild failed for field {}: {error}",
                        field.0,
                    ))
                })
            })
            .await?;
            total_fed = total_fed.checked_add(fed).ok_or_else(|| {
                crate::Error::Corruption(format!(
                    "IVF-PQ rebuild vector count overflows for field {}",
                    field.0,
                ))
            })?;
        }

        if total_fed == 0 {
            return Ok(None);
        }

        log::info!(
            "[dense_vector_merge] field {} IVF-PQ rebuilt from {} vectors into {} assignments (estimated {}, {:.1}s)",
            field.0,
            total_fed,
            index.len(),
            crate::format_bytes(index.estimated_memory_bytes() as u64),
            ann_start.elapsed().as_secs_f64()
        );
        Ok(Some((index, centroids.num_clusters)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::SegmentMerger;
    use crate::directories::RamDirectory;
    use crate::dsl::{DenseVectorConfig, Document, SchemaBuilder};
    use crate::index::{IndexConfig, IndexWriter};
    use crate::segment::reader::SegmentReader;
    use crate::segment::types::SegmentId;

    async fn committed_segment_ids(dir: &RamDirectory) -> Vec<String> {
        crate::index::IndexMetadata::load(dir)
            .await
            .unwrap()
            .segment_ids()
    }

    async fn newly_committed_segment(dir: &RamDirectory, known: &mut Vec<String>) -> String {
        let ids = committed_segment_ids(dir).await;
        let new: Vec<String> = ids
            .iter()
            .filter(|id| !known.contains(id))
            .cloned()
            .collect();
        assert_eq!(
            new.len(),
            1,
            "expected exactly one new segment, got {new:?}"
        );
        *known = ids;
        new.into_iter().next().unwrap()
    }

    /// Regression: ANN merge sources must retain their matching position in
    /// the full per-segment doc-offset array. With sources [A(has field),
    /// B(no field), C(has field)], C must not inherit B's offset.
    #[tokio::test]
    async fn ann_run_copy_skips_offsets_of_segments_without_the_field() {
        let dim = 8;
        let mut sb = SchemaBuilder::default();
        let title = sb.add_text_field("title", true, true);
        let embedding = sb.add_dense_vector_field_with_config(
            "embedding",
            true,
            true,
            DenseVectorConfig::with_ivf_pq(dim, Some(1), 1),
        );
        let schema = sb.build();

        let dir = RamDirectory::new();
        let config = IndexConfig {
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            num_indexing_threads: 1,
            ..Default::default()
        };
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config)
            .await
            .unwrap();

        // Training segment: provides the sample, stays out of the merge.
        for i in 0..4 {
            let mut doc = Document::new();
            doc.add_text(title, format!("train {i}"));
            doc.add_dense_vector(embedding, vec![i as f32 + 1.0; dim]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        writer.build_vector_index().await.unwrap();
        let mut known = committed_segment_ids(&dir).await;

        // Segment A: 2 docs WITH the field (gets a per-segment IVF at flush).
        for i in 0..2 {
            let mut doc = Document::new();
            doc.add_text(title, format!("a {i}"));
            doc.add_dense_vector(embedding, vec![0.5; dim]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        let seg_a = newly_committed_segment(&dir, &mut known).await;

        // Segment B: 3 docs WITHOUT the field (no flat entry, no ANN entry).
        for i in 0..3 {
            let mut doc = Document::new();
            doc.add_text(title, format!("b {i}"));
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        let seg_b = newly_committed_segment(&dir, &mut known).await;

        // Segment C: 2 docs WITH the field.
        for i in 0..2 {
            let mut doc = Document::new();
            doc.add_text(title, format!("c {i}"));
            doc.add_dense_vector(embedding, vec![0.25; dim]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        let seg_c = newly_committed_segment(&dir, &mut known).await;

        // Open the merge sources in a fixed order: [A, B, C].
        let schema = Arc::new(schema);
        let mut readers = Vec::new();
        for id in [&seg_a, &seg_b, &seg_c] {
            readers.push(
                SegmentReader::open(
                    &dir,
                    SegmentId::from_hex(id).unwrap(),
                    Arc::clone(&schema),
                    16,
                )
                .await
                .unwrap(),
            );
        }
        assert!(matches!(
            readers[0].get_vector_index(embedding),
            Some(crate::segment::VectorIndex::IvfPq(_))
        ));
        assert!(
            readers[1].get_vector_index(embedding).is_none(),
            "segment B must not carry the dense field"
        );
        assert!(matches!(
            readers[2].get_vector_index(embedding),
            Some(crate::segment::VectorIndex::IvfPq(_))
        ));

        let merged_id = SegmentId::new();
        let trained = writer.segment_manager().trained().unwrap();
        SegmentMerger::new(Arc::clone(&schema))
            .merge(&dir, &readers, merged_id, Some(trained.as_ref()))
            .await
            .unwrap();

        let mut merged = SegmentReader::open(&dir, merged_id, Arc::clone(&schema), 16)
            .await
            .unwrap();
        merged.set_trained_vectors(Arc::clone(&trained));
        let mut doc_ids: Vec<u32> = merged
            .search_dense_vector(
                embedding,
                &[0.5; 8],
                4,
                1,
                1.0,
                crate::query::MultiValueCombiner::Max,
            )
            .await
            .unwrap()
            .into_iter()
            .map(|result| result.doc_id)
            .collect();
        doc_ids.sort_unstable();
        // A occupies merged docs 0..2, B (no vectors) 2..5, C 5..7. C's
        // vectors must be remapped with C's own offset (5), not B's (2).
        assert_eq!(
            doc_ids,
            vec![0, 1, 5, 6],
            "merged ANN doc ids must use each field-bearing segment's own offset"
        );
    }
}
