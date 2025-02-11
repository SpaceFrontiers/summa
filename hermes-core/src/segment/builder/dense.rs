//! Dense vector streaming build (footer-based format).
//!
//! Streams each field's flat data directly to disk, then writes TOC + footer.
//! Supports parallel ANN index building (IvfRaBitQ, ScaNN).

use std::io::Write;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::Result;
use crate::dsl::{DenseVectorQuantization, Field, Schema, VectorIndexType};
use crate::segment::format::{DenseVectorTocEntry, write_dense_toc_and_footer};
use crate::segment::vector_data::FlatVectorData;

use crate::DocId;

/// Builder for dense vector index
///
/// Collects vectors with ordinal tracking for multi-valued fields.
pub(super) struct DenseVectorBuilder {
    /// Dimension of vectors
    pub dim: usize,
    /// Document IDs with ordinals: (doc_id, ordinal)
    pub doc_ids: Vec<(DocId, u16)>,
    /// Flat vector storage (doc_ids.len() * dim floats)
    pub vectors: Vec<f32>,
}

impl DenseVectorBuilder {
    pub fn new(dim: usize) -> Self {
        // Pre-allocate for ~16 vectors to avoid early reallocation chains
        Self {
            dim,
            doc_ids: Vec::with_capacity(16),
            vectors: Vec::with_capacity(16 * dim),
        }
    }

    pub fn add(&mut self, doc_id: DocId, ordinal: u16, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim, "Vector dimension mismatch");
        self.doc_ids.push((doc_id, ordinal));
        self.vectors.extend_from_slice(vector);
    }

    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }
}

/// Stream dense vectors directly to disk (zero-buffer for vector data).
///
/// Computes sizes deterministically (no trial serialization needed), writes
/// a small header, then streams each field's raw f32 data directly to the writer.
pub(super) fn build_vectors_streaming(
    dense_vectors: FxHashMap<u32, DenseVectorBuilder>,
    schema: &Schema,
    trained: Option<&super::super::TrainedVectorStructures>,
    writer: &mut dyn Write,
) -> Result<()> {
    let mut fields: Vec<(u32, DenseVectorBuilder)> = dense_vectors
        .into_iter()
        .filter(|(_, b)| b.len() > 0)
        .collect();
    fields.sort_by_key(|(id, _)| *id);

    if fields.is_empty() {
        return Ok(());
    }

    // Resolve quantization config per field from schema
    let quants: Vec<DenseVectorQuantization> = fields
        .iter()
        .map(|(field_id, _)| {
            schema
                .get_field_entry(Field(*field_id))
                .and_then(|e| e.dense_vector_config.as_ref())
                .map(|c| c.quantization)
                .unwrap_or(DenseVectorQuantization::F32)
        })
        .collect();

    // Compute sizes using deterministic formula (no serialization needed)
    let mut field_sizes: Vec<usize> = Vec::with_capacity(fields.len());
    for (i, (_field_id, builder)) in fields.iter().enumerate() {
        field_sizes.push(FlatVectorData::serialized_binary_size(
            builder.dim,
            builder.len(),
            quants[i],
        ));
    }

    // Data-first format: stream field data, then write TOC + footer at end.
    // Data starts at file offset 0 → mmap page-aligned, no alignment copies.
    let mut toc: Vec<DenseVectorTocEntry> = Vec::with_capacity(fields.len() * 2);
    let mut current_offset = 0u64;

    // Pre-build ANN indexes in parallel across fields.
    // Each field's ANN build is independent (different vectors, different centroids).
    let ann_blobs: Vec<(u32, u8, Vec<u8>)> = if let Some(trained) = trained {
        fields
            .par_iter()
            .filter_map(|(field_id, builder)| {
                let config = schema
                    .get_field_entry(Field(*field_id))
                    .and_then(|e| e.dense_vector_config.as_ref())?;

                let dim = builder.dim;
                let blob = match config.index_type {
                    VectorIndexType::IvfRaBitQ if trained.centroids.contains_key(field_id) => {
                        let centroids = &trained.centroids[field_id];
                        let (mut index, codebook) =
                            super::super::ann_build::new_ivf_rabitq(dim, centroids);
                        for (i, (doc_id, ordinal)) in builder.doc_ids.iter().enumerate() {
                            let v = &builder.vectors[i * dim..(i + 1) * dim];
                            index.add_vector(centroids, &codebook, *doc_id, *ordinal, v);
                        }
                        super::super::ann_build::serialize_ivf_rabitq(index, codebook)
                            .map(|b| (super::super::ann_build::IVF_RABITQ_TYPE, b))
                    }
                    VectorIndexType::ScaNN
                        if trained.centroids.contains_key(field_id)
                            && trained.codebooks.contains_key(field_id) =>
                    {
                        let centroids = &trained.centroids[field_id];
                        let codebook = &trained.codebooks[field_id];
                        let mut index =
                            super::super::ann_build::new_scann(dim, centroids, codebook);
                        for (i, (doc_id, ordinal)) in builder.doc_ids.iter().enumerate() {
                            let v = &builder.vectors[i * dim..(i + 1) * dim];
                            index.add_vector(centroids, codebook, *doc_id, *ordinal, v);
                        }
                        super::super::ann_build::serialize_scann(index, codebook)
                            .map(|b| (super::super::ann_build::SCANN_TYPE, b))
                    }
                    _ => return None,
                };
                match blob {
                    Ok((index_type, bytes)) => {
                        log::info!(
                            "[segment_build] built ANN(type={}) for field {} ({} vectors, {} bytes)",
                            index_type,
                            field_id,
                            builder.doc_ids.len(),
                            bytes.len()
                        );
                        Some((*field_id, index_type, bytes))
                    }
                    Err(e) => {
                        log::warn!(
                            "[segment_build] ANN serialize failed for field {}: {}",
                            field_id,
                            e
                        );
                        None
                    }
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // Stream each field's flat data directly (builder → disk, no intermediate buffer)
    for (i, (_field_id, builder)) in fields.into_iter().enumerate() {
        let data_offset = current_offset;
        FlatVectorData::serialize_binary_from_flat_streaming(
            builder.dim,
            &builder.vectors,
            &builder.doc_ids,
            quants[i],
            writer,
        )
        .map_err(crate::Error::Io)?;
        current_offset += field_sizes[i] as u64;
        toc.push(DenseVectorTocEntry {
            field_id: _field_id,
            index_type: super::super::ann_build::FLAT_TYPE,
            offset: data_offset,
            size: field_sizes[i] as u64,
        });
        // Pad to 8-byte boundary so next field's mmap slice is aligned
        let pad = (8 - (current_offset % 8)) % 8;
        if pad > 0 {
            writer.write_all(&[0u8; 8][..pad as usize])?;
            current_offset += pad;
        }
        // builder dropped here, freeing vector memory before next field
    }

    // Write ANN blob entries after flat entries
    for (field_id, index_type, blob) in ann_blobs {
        let data_offset = current_offset;
        let blob_len = blob.len() as u64;
        writer.write_all(&blob)?;
        current_offset += blob_len;
        toc.push(DenseVectorTocEntry {
            field_id,
            index_type,
            offset: data_offset,
            size: blob_len,
        });
        let pad = (8 - (current_offset % 8)) % 8;
        if pad > 0 {
            writer.write_all(&[0u8; 8][..pad as usize])?;
            current_offset += pad;
        }
    }

    // Write TOC + footer
    write_dense_toc_and_footer(writer, current_offset, &toc)?;

    Ok(())
}
