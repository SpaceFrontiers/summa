//! Dense vector streaming build (footer-based format).
//!
//! Streams each field's flat data directly to disk, then writes TOC + footer.
//! Supports parallel ANN index building (IvfRaBitQ, ScaNN).

use std::io::Write;

#[cfg(feature = "native")]
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::Result;
#[cfg(feature = "native")]
use crate::dsl::VectorIndexType;
use crate::dsl::{DenseVectorQuantization, Field, FieldType, Schema};
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

/// Builder for binary dense vector index
///
/// Collects packed-bit vectors with ordinal tracking for multi-valued fields.
pub(super) struct BinaryDenseVectorBuilder {
    /// Number of bits (dimensions)
    pub dim_bits: usize,
    /// Bytes per vector: ceil(dim_bits/8)
    pub byte_len: usize,
    /// Document IDs with ordinals: (doc_id, ordinal)
    pub doc_ids: Vec<(DocId, u16)>,
    /// Flat packed-bit storage (doc_ids.len() * byte_len bytes)
    pub vectors: Vec<u8>,
}

impl BinaryDenseVectorBuilder {
    pub fn new(dim_bits: usize) -> Self {
        let byte_len = dim_bits.div_ceil(8);
        Self {
            dim_bits,
            byte_len,
            doc_ids: Vec::with_capacity(16),
            vectors: Vec::with_capacity(16 * byte_len),
        }
    }

    pub fn add(&mut self, doc_id: DocId, ordinal: u16, packed_bytes: &[u8]) {
        debug_assert_eq!(
            packed_bytes.len(),
            self.byte_len,
            "Binary vector byte length mismatch: expected {}, got {}",
            self.byte_len,
            packed_bytes.len()
        );
        self.doc_ids.push((doc_id, ordinal));
        self.vectors.extend_from_slice(packed_bytes);
    }

    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }
}

/// Stream dense and binary dense vectors directly to disk (zero-buffer for vector data).
///
/// Computes sizes deterministically (no trial serialization needed), writes
/// a small header, then streams each field's raw data directly to the writer.
/// Both dense (f32/f16/u8) and binary dense (packed bits) vectors share a single
/// TOC + footer to avoid the double-footer bug.
pub(super) fn build_vectors_streaming(
    dense_vectors: FxHashMap<u32, DenseVectorBuilder>,
    binary_vectors: FxHashMap<u32, BinaryDenseVectorBuilder>,
    schema: &Schema,
    trained: Option<&super::super::TrainedVectorStructures>,
    writer: &mut dyn Write,
) -> Result<()> {
    let mut fields: Vec<(u32, DenseVectorBuilder)> = dense_vectors
        .into_iter()
        .filter(|(_, b)| b.len() > 0)
        .collect();
    fields.sort_by_key(|(id, _)| *id);

    let mut binary_fields: Vec<(u32, BinaryDenseVectorBuilder)> = binary_vectors
        .into_iter()
        .filter(|(_, b)| b.len() > 0)
        .collect();
    binary_fields.sort_by_key(|(id, _)| *id);

    if fields.is_empty() && binary_fields.is_empty() {
        return Ok(());
    }

    // Resolve quantization config per field from schema
    let quants: Vec<DenseVectorQuantization> = fields
        .iter()
        .map(|(field_id, builder)| {
            let entry = schema.get_field_entry(Field(*field_id)).ok_or_else(|| {
                crate::Error::Schema(format!(
                    "dense vector builder references unknown field {field_id}"
                ))
            })?;
            let config = entry
                .dense_vector_config
                .as_ref()
                .filter(|_| entry.field_type == FieldType::DenseVector)
                .ok_or_else(|| {
                    crate::Error::Schema(format!(
                        "dense vector builder field {field_id} does not match its schema type"
                    ))
                })?;
            if builder.dim != config.dim {
                return Err(crate::Error::Schema(format!(
                    "dense vector builder field {field_id} has dimension {}, schema expects {}",
                    builder.dim, config.dim
                )));
            }
            Ok(config.quantization)
        })
        .collect::<Result<_>>()?;

    for (field_id, builder) in &binary_fields {
        let entry = schema.get_field_entry(Field(*field_id)).ok_or_else(|| {
            crate::Error::Schema(format!(
                "binary vector builder references unknown field {field_id}"
            ))
        })?;
        let config = entry
            .binary_dense_vector_config
            .as_ref()
            .filter(|_| entry.field_type == FieldType::BinaryDenseVector)
            .ok_or_else(|| {
                crate::Error::Schema(format!(
                    "binary vector builder field {field_id} does not match its schema type"
                ))
            })?;
        if builder.dim_bits != config.dim {
            return Err(crate::Error::Schema(format!(
                "binary vector builder field {field_id} has dimension {}, schema expects {}",
                builder.dim_bits, config.dim
            )));
        }
    }

    // Compute sizes using deterministic formula (no serialization needed)
    let mut field_sizes: Vec<usize> = Vec::with_capacity(fields.len());
    for (i, (_field_id, builder)) in fields.iter().enumerate() {
        field_sizes.push(FlatVectorData::validate_dense_input(
            builder.dim,
            &builder.vectors,
            &builder.doc_ids,
            quants[i],
        )?);
    }
    let binary_field_sizes: Vec<usize> = binary_fields
        .iter()
        .map(|(_, builder)| {
            FlatVectorData::validate_binary_input(
                builder.dim_bits,
                &builder.vectors,
                &builder.doc_ids,
            )
        })
        .collect::<std::io::Result<_>>()?;

    // Data-first format: stream field data, then write TOC + footer at end.
    // Data starts at file offset 0 → mmap page-aligned, no alignment copies.
    let toc_capacity = fields
        .len()
        .checked_add(binary_fields.len())
        .and_then(|field_count| field_count.checked_mul(2))
        .ok_or_else(|| {
            crate::Error::Internal("dense-vector TOC capacity overflows usize".into())
        })?;
    let mut toc: Vec<DenseVectorTocEntry> = Vec::with_capacity(toc_capacity);
    let mut current_offset = 0u64;

    // Pre-build ANN indexes across fields (native only — requires trained structures).
    #[cfg(feature = "native")]
    let ann_blobs: Vec<(u32, u8, Vec<u8>)> = if let Some(trained) = trained {
        let ann_blob_fn = |(field_id, builder): &(u32, DenseVectorBuilder)|
         -> Result<Option<(u32, u8, Vec<u8>)>> {
                let Some(config) = schema
                    .get_field_entry(Field(*field_id))
                    .and_then(|e| e.dense_vector_config.as_ref())
                else {
                    return Ok(None);
                };

                let dim = builder.dim;
                let blob = match config.index_type {
                    VectorIndexType::IvfRaBitQ if trained.centroids.contains_key(field_id) => {
                        let centroids = &trained.centroids[field_id];
                        let bits = config.rabitq_bits.unwrap_or(1);
                        let (mut index, codebook) =
                            super::super::ann_build::new_ivf_rabitq(dim, centroids, bits);
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
                    _ => return Ok(None),
                };
                let (index_type, bytes) = blob?;
                log::info!(
                    "[segment_build] built ANN(type={}) for field {} ({} vectors, {} bytes)",
                    index_type,
                    field_id,
                    builder.doc_ids.len(),
                    bytes.len()
                );
                Ok(Some((*field_id, index_type, bytes)))
            };

        fields
            .par_iter()
            .map(ann_blob_fn)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect()
    } else {
        Vec::new()
    };
    // WASM: no ANN index building (requires trained structures from SegmentManager)
    #[cfg(not(feature = "native"))]
    let ann_blobs: Vec<(u32, u8, Vec<u8>)> = {
        let _ = trained; // suppress unused warning
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
        let field_size = u64::try_from(field_sizes[i])
            .map_err(|_| crate::Error::Internal("flat vector size exceeds u64".into()))?;
        current_offset = current_offset
            .checked_add(field_size)
            .ok_or_else(|| crate::Error::Internal("vector output offset exceeds u64".into()))?;
        toc.push(DenseVectorTocEntry {
            field_id: _field_id,
            index_type: super::super::ann_build::FLAT_TYPE,
            offset: data_offset,
            size: field_size,
        });
        // Pad to 8-byte boundary so next field's mmap slice is aligned
        let pad = (8 - (current_offset % 8)) % 8;
        if pad > 0 {
            writer.write_all(&[0u8; 8][..pad as usize])?;
            current_offset = current_offset.checked_add(pad).ok_or_else(|| {
                crate::Error::Internal("vector output padding exceeds u64".into())
            })?;
        }
        // builder dropped here, freeing vector memory before next field
    }

    // Write ANN blob entries after flat entries
    for (field_id, index_type, blob) in ann_blobs {
        let data_offset = current_offset;
        let blob_len = u64::try_from(blob.len())
            .map_err(|_| crate::Error::Internal("ANN blob size exceeds u64".into()))?;
        writer.write_all(&blob)?;
        current_offset = current_offset
            .checked_add(blob_len)
            .ok_or_else(|| crate::Error::Internal("vector output offset exceeds u64".into()))?;
        toc.push(DenseVectorTocEntry {
            field_id,
            index_type,
            offset: data_offset,
            size: blob_len,
        });
        let pad = (8 - (current_offset % 8)) % 8;
        if pad > 0 {
            writer.write_all(&[0u8; 8][..pad as usize])?;
            current_offset = current_offset.checked_add(pad).ok_or_else(|| {
                crate::Error::Internal("vector output padding exceeds u64".into())
            })?;
        }
    }

    // Stream binary dense vector fields (packed bits, Hamming distance)
    for ((field_id, builder), data_size) in binary_fields.into_iter().zip(binary_field_sizes) {
        let data_offset = current_offset;
        #[cfg(feature = "native")]
        let num_vectors = builder.len();

        FlatVectorData::serialize_binary_from_bits_streaming(
            builder.dim_bits,
            &builder.vectors,
            &builder.doc_ids,
            writer,
        )
        .map_err(crate::Error::Io)?;

        let data_size = u64::try_from(data_size)
            .map_err(|_| crate::Error::Internal("binary flat vector size exceeds u64".into()))?;
        current_offset = current_offset
            .checked_add(data_size)
            .ok_or_else(|| crate::Error::Internal("vector output offset exceeds u64".into()))?;
        toc.push(DenseVectorTocEntry {
            field_id,
            index_type: super::super::ann_build::FLAT_TYPE,
            offset: data_offset,
            size: data_size,
        });

        let pad = (8 - (current_offset % 8)) % 8;
        if pad > 0 {
            writer.write_all(&[0u8; 8][..pad as usize])?;
            current_offset = current_offset.checked_add(pad).ok_or_else(|| {
                crate::Error::Internal("vector output padding exceeds u64".into())
            })?;
        }

        // Binary IVF index (native only): built at commit when configured
        // and the segment is large enough for probing to beat brute force.
        #[cfg(feature = "native")]
        {
            let binary_config = schema
                .get_field_entry(Field(field_id))
                .and_then(|e| e.binary_dense_vector_config.as_ref());
            if let Some(cfg) = binary_config
                && cfg.index_type == crate::dsl::BinaryIndexType::Ivf
                && num_vectors >= cfg.default_build_threshold()
            {
                let num_clusters = cfg.optimal_num_clusters(num_vectors);
                let ivf_config =
                    crate::structures::BinaryIvfConfig::new(builder.dim_bits, num_clusters);
                let index = crate::structures::BinaryIvfIndex::build(
                    ivf_config,
                    &builder.vectors,
                    &builder.doc_ids,
                )
                .map_err(crate::Error::Io)?;
                let blob_offset = current_offset;
                let mut output = &mut *writer;
                let blob_len = u64::try_from(
                    index.write_to(&mut output).map_err(crate::Error::Io)?,
                )
                .map_err(|_| crate::Error::Internal("binary IVF blob size exceeds u64".into()))?;
                current_offset = current_offset.checked_add(blob_len).ok_or_else(|| {
                    crate::Error::Internal("binary IVF output offset exceeds u64".into())
                })?;
                toc.push(DenseVectorTocEntry {
                    field_id,
                    index_type: super::super::ann_build::BINARY_IVF_TYPE,
                    offset: blob_offset,
                    size: blob_len,
                });
                drop(index);
                let pad = (8 - (current_offset % 8)) % 8;
                if pad > 0 {
                    writer.write_all(&[0u8; 8][..pad as usize])?;
                    current_offset = current_offset.checked_add(pad).ok_or_else(|| {
                        crate::Error::Internal("vector output padding exceeds u64".into())
                    })?;
                }
                log::debug!(
                    "[build_vectors] field {}: binary IVF built ({} vectors, {} clusters, {} bytes)",
                    field_id,
                    num_vectors,
                    num_clusters,
                    blob_len,
                );
            }
        }
    }

    // Write TOC + footer
    write_dense_toc_and_footer(writer, current_offset, &toc)?;

    Ok(())
}
