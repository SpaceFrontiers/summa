//! Dense vector merge strategies
//!
//! Every field always gets a Flat entry (raw vectors for reranking/merge).
//! Optionally, an ANN entry is also written alongside:
//! 1. O(1) cluster merge for homogeneous ANN types (IVF/ScaNN)
//! 2. ANN rebuild with trained structures
//!
//! Raw vectors are read from source segments' lazy flat data (mmap-backed),
//! never from the document store.

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

/// Pre-serialized field data (ANN indexes, cluster merges)
struct BlobField {
    field_id: u32,
    index_type: u8,
    data: Vec<u8>,
}

/// Flat field to be streamed from source segments' lazy flat vectors
struct FlatStreamField {
    field_id: u32,
    dim: usize,
    total_vectors: usize,
    quantization: DenseVectorQuantization,
}

/// Batch size for streaming vector reads (1024 vectors at a time)
const VECTOR_BATCH_SIZE: usize = 1024;

/// Chunk size for streaming flat vector bytes during merge (8 MB)
const FLAT_VECTOR_CHUNK: u64 = 8 * 1024 * 1024;

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

impl SegmentMerger {
    /// Merge dense vector indexes - returns output size in bytes
    ///
    /// Every field always gets a Flat entry (raw vectors streamed from source
    /// segments' lazy flat data via mmap). Optionally an ANN entry is also
    /// written alongside via O(1) cluster merge or ANN rebuild.
    ///
    /// No document store reads — all vector data comes from .vectors files.
    pub(super) async fn merge_dense_vectors<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
        trained: Option<&TrainedVectorStructures>,
    ) -> Result<usize> {
        let mut blob_fields: Vec<BlobField> = Vec::new();
        let mut flat_fields: Vec<FlatStreamField> = Vec::new();
        let doc_offs = doc_offsets(segments)?;

        for (field, entry) in self.schema.fields() {
            if !matches!(entry.field_type, FieldType::DenseVector)
                || !(entry.indexed || entry.stored)
            {
                continue;
            }

            let config = entry.dense_vector_config.as_ref();

            // Full dimension from lazy flat (for Flat entries and reranking)
            let dim: usize = segments
                .iter()
                .filter_map(|s| s.flat_vectors().get(&field.0).map(|f| f.dim))
                .find(|&d| d > 0)
                .unwrap_or(0);
            if dim == 0 {
                continue;
            }

            // Count total vectors across all segments (from lazy flat)
            let total_vectors: usize = segments
                .iter()
                .filter_map(|s| s.flat_vectors().get(&field.0).map(|f| f.num_vectors))
                .sum();

            // Resolve quantization: prefer schema config, fall back to source segment
            let quantization = config
                .map(|c| c.quantization)
                .or_else(|| {
                    segments
                        .iter()
                        .find_map(|s| s.flat_vectors().get(&field.0).map(|f| f.quantization))
                })
                .unwrap_or(DenseVectorQuantization::F32);

            // 1. ALWAYS write Flat entry (raw vectors for reranking/merge)
            if total_vectors > 0 {
                flat_fields.push(FlatStreamField {
                    field_id: field.0,
                    dim,
                    total_vectors,
                    quantization,
                });
            }

            // 2. Optionally write ANN entry alongside Flat

            // Count segments with each ANN type
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
                    crate::structures::IVFPQIndex::merge(&refs, &doc_offs),
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
                        blob_fields.push(BlobField {
                            field_id: field.0,
                            index_type: crate::segment::ann_build::SCANN_TYPE,
                            data: bytes,
                        });
                        continue; // ANN done, flat already queued
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
                    crate::structures::IVFRaBitQIndex::merge(&refs, &doc_offs),
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
                        blob_fields.push(BlobField {
                            field_id: field.0,
                            index_type: crate::segment::ann_build::IVF_RABITQ_TYPE,
                            data: bytes,
                        });
                        continue; // ANN done, flat already queued
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
            let ann_type =
                trained
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

            if ann_type.is_none() {
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
            }

            if let Some(ann) = ann_type {
                let trained = trained.unwrap();
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
                            let fed =
                                feed_segment(segment, field, offset, |doc_id, ordinal, vec| {
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
                        let bytes =
                            crate::segment::ann_build::serialize_ivf_rabitq(index, codebook)?;
                        (crate::segment::ann_build::IVF_RABITQ_TYPE, bytes)
                    }
                    VectorIndexType::ScaNN => {
                        let centroids = &trained.centroids[&field.0];
                        let codebook = &trained.codebooks[&field.0];
                        let mut index =
                            crate::segment::ann_build::new_scann(dim, centroids, codebook);

                        for (seg_idx, segment) in segments.iter().enumerate() {
                            let offset = doc_offs[seg_idx];
                            let fed =
                                feed_segment(segment, field, offset, |doc_id, ordinal, vec| {
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
                blob_fields.push(BlobField {
                    field_id: field.0,
                    index_type,
                    data: bytes,
                });
            }
        }

        // --- Write vectors file (data-first, TOC at end) ---
        //
        // Format: [field data...] [TOC entries] [footer]
        // Data starts at file offset 0 → mmap page-aligned, no alignment copies needed.
        // Footer (last 16 bytes): toc_offset(u64) + num_fields(u32) + magic(u32)
        let total_entries = blob_fields.len() + flat_fields.len();
        if total_entries == 0 {
            return Ok(0);
        }

        // Build write order: sort by (field_id, index_type)
        struct PendingEntry {
            field_id: u32,
            index_type: u8,
            blob_idx: Option<usize>,
            flat_idx: Option<usize>,
        }
        let mut pending: Vec<PendingEntry> = Vec::with_capacity(total_entries);

        for (i, flat) in flat_fields.iter().enumerate() {
            pending.push(PendingEntry {
                field_id: flat.field_id,
                index_type: crate::segment::ann_build::FLAT_TYPE,
                blob_idx: None,
                flat_idx: Some(i),
            });
        }
        for (i, blob) in blob_fields.iter().enumerate() {
            pending.push(PendingEntry {
                field_id: blob.field_id,
                index_type: blob.index_type,
                blob_idx: Some(i),
                flat_idx: None,
            });
        }
        pending.sort_by(|a, b| {
            a.field_id
                .cmp(&b.field_id)
                .then(a.index_type.cmp(&b.index_type))
        });

        // Stream data first — track offsets as we go
        let write_start = std::time::Instant::now();
        let mut writer = OffsetWriter::new(dir.streaming_writer(&files.vectors).await?);
        let mut toc: Vec<DenseVectorTocEntry> = Vec::with_capacity(total_entries);

        for entry in &pending {
            let data_offset = writer.offset();

            if let Some(blob_idx) = entry.blob_idx {
                writer.write_all(&blob_fields[blob_idx].data)?;
            } else if let Some(flat_idx) = entry.flat_idx {
                let flat = &flat_fields[flat_idx];

                FlatVectorData::write_binary_header(
                    flat.dim,
                    flat.total_vectors,
                    flat.quantization,
                    &mut writer,
                )?;

                // Pass 1: stream raw vector bytes in chunks
                for segment in segments {
                    if let Some(lazy_flat) = segment.flat_vectors().get(&entry.field_id) {
                        let total_bytes = lazy_flat.vector_bytes_len();
                        let base_offset = lazy_flat.vectors_byte_offset();
                        let handle = lazy_flat.handle();
                        for chunk_start in (0..total_bytes).step_by(FLAT_VECTOR_CHUNK as usize) {
                            let chunk_end = (chunk_start + FLAT_VECTOR_CHUNK).min(total_bytes);
                            let bytes = handle
                                .read_bytes_range(
                                    base_offset + chunk_start..base_offset + chunk_end,
                                )
                                .await
                                .map_err(crate::Error::Io)?;
                            writer.write_all(bytes.as_slice())?;
                        }
                    }
                }

                // Pass 2: stream doc_ids with offset adjustment (buffered per segment)
                for (seg_idx, segment) in segments.iter().enumerate() {
                    if let Some(lazy_flat) = segment.flat_vectors().get(&entry.field_id) {
                        let offset = doc_offs[seg_idx];
                        let count = lazy_flat.num_vectors;
                        let mut buf = Vec::with_capacity(count * 6);
                        for i in 0..count {
                            let (doc_id, ordinal) = lazy_flat.get_doc_id(i);
                            buf.extend_from_slice(&(offset + doc_id).to_le_bytes());
                            buf.extend_from_slice(&ordinal.to_le_bytes());
                        }
                        writer.write_all(&buf)?;
                    }
                }
            }

            let data_size = writer.offset() - data_offset;
            toc.push(DenseVectorTocEntry {
                field_id: entry.field_id,
                index_type: entry.index_type,
                offset: data_offset,
                size: data_size,
            });

            // Pad to 8-byte boundary so next field's mmap slice is aligned
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
}
