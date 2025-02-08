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
use std::mem::size_of;

use super::OffsetWriter;
use super::SegmentMerger;
use super::TrainedVectorStructures;
use super::doc_offsets;
use crate::Result;
use crate::directories::{AsyncFileRead, Directory, DirectoryWriter};
use crate::dsl::{DenseVectorQuantization, FieldType, VectorIndexType};
use crate::segment::reader::SegmentReader;
use crate::segment::types::SegmentFiles;
use crate::segment::vector_data::FlatVectorData;

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

/// Batch size for streaming vector reads (1024 vectors × 1024 dims × 4 bytes ≈ 4MB)
const VECTOR_BATCH_SIZE: usize = 1024;

/// Streams vectors from a segment's lazy flat data into an add_fn callback.
///
/// Reads vectors in batches of VECTOR_BATCH_SIZE to bound memory usage.
/// Each batch is a single range read via LazyFileSlice.
///
/// Returns number of vectors added.
async fn feed_segment(
    segment: &SegmentReader,
    field: crate::dsl::Field,
    doc_id_offset: u32,
    mut add_fn: impl FnMut(u32, u16, &[f32]),
) -> usize {
    let lazy_flat = match segment.flat_vectors().get(&field.0) {
        Some(f) => f,
        None => return 0,
    };
    let n = lazy_flat.num_vectors;
    if n == 0 {
        return 0;
    }
    let dim = lazy_flat.dim;
    let mut count = 0;

    for batch_start in (0..n).step_by(VECTOR_BATCH_SIZE) {
        let batch_count = VECTOR_BATCH_SIZE.min(n - batch_start);
        let batch_bytes = match lazy_flat.read_vectors_batch(batch_start, batch_count).await {
            Ok(b) => b,
            Err(_) => continue,
        };
        let raw = batch_bytes.as_slice();
        let batch_floats = batch_count * dim;

        // Use mmap bytes directly if f32-aligned, otherwise copy once
        let mut aligned_buf: Vec<f32> = Vec::new();
        let vectors: &[f32] = if (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
        {
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, batch_floats) }
        } else {
            aligned_buf.resize(batch_floats, 0.0);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    raw.as_ptr(),
                    aligned_buf.as_mut_ptr() as *mut u8,
                    batch_floats * std::mem::size_of::<f32>(),
                );
            }
            &aligned_buf
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
    count
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
        let doc_offs = doc_offsets(segments);

        for (field, entry) in self.schema.fields() {
            if !matches!(entry.field_type, FieldType::DenseVector)
                || !(entry.indexed || entry.stored)
            {
                continue;
            }

            let config = entry.dense_vector_config.as_ref();

            // Full dimension from lazy flat (for Flat entries and reranking)
            let full_dim: usize = segments
                .iter()
                .filter_map(|s| s.flat_vectors().get(&field.0).map(|f| f.dim))
                .find(|&d| d > 0)
                .unwrap_or(0);
            if full_dim == 0 {
                continue;
            }
            let dim = full_dim;

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
                    dim: full_dim,
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

                let centroids_and_codebook = segments.iter().find_map(|s| {
                    let centroids = s.coarse_centroids()?.clone();
                    let (_, codebook) = s.get_scann_vector_index(field)?;
                    Some((centroids, codebook))
                });

                match (
                    crate::structures::IVFPQIndex::merge(&refs, &doc_offs),
                    centroids_and_codebook,
                ) {
                    (Ok(merged), Some((centroids, codebook))) => {
                        let index_data = crate::segment::ScaNNIndexData {
                            centroids: (*centroids).clone(),
                            codebook: (*codebook).clone(),
                            index: merged,
                        };
                        let bytes = index_data
                            .to_bytes()
                            .map_err(|e| crate::Error::Serialization(e.to_string()))?;
                        blob_fields.push(BlobField {
                            field_id: field.0,
                            index_type: 2,
                            data: bytes,
                        });
                        continue; // ANN done, flat already queued
                    }
                    (Err(e), _) => {
                        log::warn!("ScaNN merge failed: {}, falling back to rebuild", e);
                    }
                    (_, None) => {
                        log::warn!(
                            "ScaNN merge: missing centroids/codebook, falling back to rebuild"
                        );
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

                let centroids_and_codebook = segments.iter().find_map(|s| {
                    let centroids = s.coarse_centroids()?.clone();
                    let (_, codebook) = s.get_ivf_vector_index(field)?;
                    Some((centroids, codebook))
                });

                match (
                    crate::structures::IVFRaBitQIndex::merge(&refs, &doc_offs),
                    centroids_and_codebook,
                ) {
                    (Ok(merged), Some((centroids, codebook))) => {
                        let index_data = crate::segment::IVFRaBitQIndexData {
                            centroids: (*centroids).clone(),
                            codebook: (*codebook).clone(),
                            index: merged,
                        };
                        let bytes = index_data
                            .to_bytes()
                            .map_err(|e| crate::Error::Serialization(e.to_string()))?;
                        blob_fields.push(BlobField {
                            field_id: field.0,
                            index_type: 1,
                            data: bytes,
                        });
                        continue; // ANN done, flat already queued
                    }
                    (Err(e), _) => {
                        log::warn!("IVF merge failed: {}, falling back to rebuild", e);
                    }
                    (_, None) => {
                        log::warn!(
                            "IVF merge: missing centroids/codebook, falling back to rebuild"
                        );
                    }
                }
            }

            // --- Try ANN rebuild from lazy flat vectors ---
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

            if let Some(ann) = ann_type {
                let trained = trained.unwrap();
                let mut total_fed = 0usize;

                match ann {
                    VectorIndexType::IvfRaBitQ => {
                        let centroids = &trained.centroids[&field.0];
                        let rabitq_config = crate::structures::RaBitQConfig::new(dim);
                        let codebook = crate::structures::RaBitQCodebook::new(rabitq_config);
                        let ivf_config = crate::structures::IVFRaBitQConfig::new(dim);
                        let mut ivf_index = crate::structures::IVFRaBitQIndex::new(
                            ivf_config,
                            centroids.version,
                            codebook.version,
                        );

                        for (seg_idx, segment) in segments.iter().enumerate() {
                            let offset = doc_offs[seg_idx];
                            total_fed +=
                                feed_segment(segment, field, offset, |doc_id, ordinal, vec| {
                                    ivf_index
                                        .add_vector(centroids, &codebook, doc_id, ordinal, vec);
                                })
                                .await;
                        }

                        let index_data = crate::segment::IVFRaBitQIndexData {
                            centroids: (**centroids).clone(),
                            codebook,
                            index: ivf_index,
                        };
                        let bytes = index_data
                            .to_bytes()
                            .map_err(|e| crate::Error::Serialization(e.to_string()))?;
                        blob_fields.push(BlobField {
                            field_id: field.0,
                            index_type: 1,
                            data: bytes,
                        });
                        log::info!(
                            "Rebuilt IVF-RaBitQ for field {} ({} vectors)",
                            field.0,
                            total_fed
                        );
                    }
                    VectorIndexType::ScaNN => {
                        let centroids = &trained.centroids[&field.0];
                        let codebook = &trained.codebooks[&field.0];
                        let ivf_pq_config = crate::structures::IVFPQConfig::new(dim);
                        let mut ivf_pq_index = crate::structures::IVFPQIndex::new(
                            ivf_pq_config,
                            centroids.version,
                            codebook.version,
                        );

                        for (seg_idx, segment) in segments.iter().enumerate() {
                            let offset = doc_offs[seg_idx];
                            total_fed +=
                                feed_segment(segment, field, offset, |doc_id, ordinal, vec| {
                                    ivf_pq_index
                                        .add_vector(centroids, codebook, doc_id, ordinal, vec);
                                })
                                .await;
                        }

                        let index_data = crate::segment::ScaNNIndexData {
                            centroids: (**centroids).clone(),
                            codebook: (**codebook).clone(),
                            index: ivf_pq_index,
                        };
                        let bytes = index_data
                            .to_bytes()
                            .map_err(|e| crate::Error::Serialization(e.to_string()))?;
                        blob_fields.push(BlobField {
                            field_id: field.0,
                            index_type: 2,
                            data: bytes,
                        });
                        log::info!(
                            "Rebuilt ScaNN for field {} ({} vectors)",
                            field.0,
                            total_fed
                        );
                    }
                    _ => {}
                }
            }
        }

        // --- Write vectors file ---
        let total_entries = blob_fields.len() + flat_fields.len();
        if total_entries == 0 {
            return Ok(0);
        }

        struct FieldEntry {
            field_id: u32,
            index_type: u8,
            data_size: u64,
            blob_idx: Option<usize>,
            flat_idx: Option<usize>,
        }

        // Flat entries first (sorted by field_id), then ANN blobs
        // Loader processes sequentially: ANN overwrites Flat in vector_indexes,
        // but both are stored — Flat goes to flat_vectors, ANN goes to vector_indexes.
        let mut entries: Vec<FieldEntry> = Vec::with_capacity(total_entries);

        for (i, flat) in flat_fields.iter().enumerate() {
            entries.push(FieldEntry {
                field_id: flat.field_id,
                index_type: 4,
                data_size: FlatVectorData::serialized_binary_size(
                    flat.dim,
                    flat.total_vectors,
                    flat.quantization,
                ) as u64,
                blob_idx: None,
                flat_idx: Some(i),
            });
        }

        for (i, blob) in blob_fields.iter().enumerate() {
            entries.push(FieldEntry {
                field_id: blob.field_id,
                index_type: blob.index_type,
                data_size: blob.data.len() as u64,
                blob_idx: Some(i),
                flat_idx: None,
            });
        }

        // Sort by (field_id, index_type) so Flat (4) comes after ANN (1,2) for same field
        entries.sort_by(|a, b| {
            a.field_id
                .cmp(&b.field_id)
                .then(a.index_type.cmp(&b.index_type))
        });

        // Write header + data
        use byteorder::{LittleEndian, WriteBytesExt};
        let mut writer = OffsetWriter::new(dir.streaming_writer(&files.vectors).await?);

        let per_field_entry =
            size_of::<u32>() + size_of::<u8>() + size_of::<u64>() + size_of::<u64>();
        let header_size = size_of::<u32>() + entries.len() * per_field_entry;

        writer.write_u32::<LittleEndian>(entries.len() as u32)?;

        let mut current_offset = header_size as u64;
        for entry in &entries {
            writer.write_u32::<LittleEndian>(entry.field_id)?;
            writer.write_u8(entry.index_type)?;
            writer.write_u64::<LittleEndian>(current_offset)?;
            writer.write_u64::<LittleEndian>(entry.data_size)?;
            current_offset += entry.data_size;
        }

        // Write field data
        for entry in &entries {
            if let Some(blob_idx) = entry.blob_idx {
                writer.write_all(&blob_fields[blob_idx].data)?;
            } else if let Some(flat_idx) = entry.flat_idx {
                let flat = &flat_fields[flat_idx];

                // Write binary header with quantization type
                FlatVectorData::write_binary_header(
                    flat.dim,
                    flat.total_vectors,
                    flat.quantization,
                    &mut writer,
                )?;

                // Pass 1: stream raw vector bytes in chunks from each segment's lazy flat
                for segment in segments {
                    if let Some(lazy_flat) = segment.flat_vectors().get(&entry.field_id) {
                        let total_bytes = lazy_flat.vector_bytes_len();
                        let base_offset = lazy_flat.vectors_byte_offset();
                        let handle = lazy_flat.handle();
                        const CHUNK: u64 = 1024 * 1024; // 1MB
                        for chunk_start in (0..total_bytes).step_by(CHUNK as usize) {
                            let chunk_end = (chunk_start + CHUNK).min(total_bytes);
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

                // Pass 2: stream doc_ids with offset adjustment
                for (seg_idx, segment) in segments.iter().enumerate() {
                    if let Some(lazy_flat) = segment.flat_vectors().get(&entry.field_id) {
                        let offset = doc_offs[seg_idx];
                        for &(doc_id, ordinal) in &lazy_flat.doc_ids {
                            writer.write_all(&(offset + doc_id).to_le_bytes())?;
                            writer.write_all(&ordinal.to_le_bytes())?;
                        }
                    }
                }
            }
        }

        let output_size = writer.offset() as usize;
        writer.finish()?;
        Ok(output_size)
    }
}
