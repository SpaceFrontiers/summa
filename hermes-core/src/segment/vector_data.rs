//! Vector index data structures shared between builder and reader

use std::io;
use std::mem::size_of;

use serde::{Deserialize, Serialize};

use crate::directories::{AsyncFileRead, LazyFileSlice, OwnedBytes};
use crate::dsl::DenseVectorQuantization;
use crate::structures::simd::{batch_f32_to_f16, batch_f32_to_u8, f16_to_f32, u8_to_f32};

/// Dequantize raw bytes to f32 based on storage quantization.
///
/// `raw` is the quantized byte slice, `out` receives the f32 values.
/// `num_floats` is the number of f32 values to produce (= num_vectors × dim).
/// Data-first file layout guarantees alignment for f32/f16 access.
#[inline]
pub fn dequantize_raw(
    raw: &[u8],
    quant: DenseVectorQuantization,
    num_floats: usize,
    out: &mut [f32],
) {
    debug_assert!(out.len() >= num_floats);
    match quant {
        DenseVectorQuantization::F32 => {
            debug_assert!(
                (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()),
                "f32 vector data not 4-byte aligned"
            );
            out[..num_floats].copy_from_slice(unsafe {
                std::slice::from_raw_parts(raw.as_ptr() as *const f32, num_floats)
            });
        }
        DenseVectorQuantization::F16 => {
            debug_assert!(
                (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>()),
                "f16 vector data not 2-byte aligned"
            );
            let f16_slice =
                unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u16, num_floats) };
            for (i, &h) in f16_slice.iter().enumerate() {
                out[i] = f16_to_f32(h);
            }
        }
        DenseVectorQuantization::UInt8 => {
            for (i, &b) in raw.iter().enumerate().take(num_floats) {
                out[i] = u8_to_f32(b);
            }
        }
    }
}

/// Magic number for binary flat vector format v3 ("FVD3" in little-endian)
const FLAT_BINARY_MAGIC: u32 = 0x46564433;

/// Binary header: magic(u32) + dim(u32) + num_vectors(u32) + quant_type(u8) + padding(3)
const FLAT_BINARY_HEADER_SIZE: usize = 16;
/// Per-doc_id entry: doc_id(u32) + ordinal(u16)
const DOC_ID_ENTRY_SIZE: usize = size_of::<u32>() + size_of::<u16>();

/// Flat vector binary format helpers for writing.
///
/// Binary format v3:
/// ```text
/// [magic(u32)][dim(u32)][num_vectors(u32)][quant_type(u8)][padding(3)]
/// [vectors: N×dim×element_size]
/// [doc_ids: N×(u32+u16)]
/// ```
///
/// `element_size` is determined by `quant_type`: f32=4, f16=2, uint8=1.
/// Reading is handled by [`LazyFlatVectorData`] which loads only doc_ids into memory
/// and accesses vector data lazily via mmap-backed range reads.
pub struct FlatVectorData;

impl FlatVectorData {
    /// Write the binary header to a writer.
    pub fn write_binary_header(
        dim: usize,
        num_vectors: usize,
        quant: DenseVectorQuantization,
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        writer.write_all(&FLAT_BINARY_MAGIC.to_le_bytes())?;
        writer.write_all(&(dim as u32).to_le_bytes())?;
        writer.write_all(&(num_vectors as u32).to_le_bytes())?;
        writer.write_all(&[quant.tag(), 0, 0, 0])?; // quant_type + 3 bytes padding
        Ok(())
    }

    /// Compute the serialized size without actually serializing.
    pub fn serialized_binary_size(
        dim: usize,
        num_vectors: usize,
        quant: DenseVectorQuantization,
    ) -> usize {
        FLAT_BINARY_HEADER_SIZE
            + num_vectors * dim * quant.element_size()
            + num_vectors * DOC_ID_ENTRY_SIZE
    }

    /// Stream from flat f32 storage to a writer, quantizing on write.
    ///
    /// `flat_vectors` is contiguous storage of dim*n f32 floats.
    /// Vectors are quantized to the specified format before writing.
    pub fn serialize_binary_from_flat_streaming(
        dim: usize,
        flat_vectors: &[f32],
        doc_ids: &[(u32, u16)],
        quant: DenseVectorQuantization,
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        let num_vectors = doc_ids.len();
        Self::write_binary_header(dim, num_vectors, quant, writer)?;

        match quant {
            DenseVectorQuantization::F32 => {
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        flat_vectors.as_ptr() as *const u8,
                        std::mem::size_of_val(flat_vectors),
                    )
                };
                writer.write_all(bytes)?;
            }
            DenseVectorQuantization::F16 => {
                let mut buf = vec![0u16; dim];
                for v in flat_vectors.chunks_exact(dim) {
                    batch_f32_to_f16(v, &mut buf);
                    let bytes: &[u8] =
                        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, dim * 2) };
                    writer.write_all(bytes)?;
                }
            }
            DenseVectorQuantization::UInt8 => {
                let mut buf = vec![0u8; dim];
                for v in flat_vectors.chunks_exact(dim) {
                    batch_f32_to_u8(v, &mut buf);
                    writer.write_all(&buf)?;
                }
            }
        }

        for &(doc_id, ordinal) in doc_ids {
            writer.write_all(&doc_id.to_le_bytes())?;
            writer.write_all(&ordinal.to_le_bytes())?;
        }

        Ok(())
    }

    /// Write raw pre-quantized vector bytes to a writer (for merger streaming).
    ///
    /// `raw_bytes` is already in the target quantized format.
    pub fn write_raw_vector_bytes(
        raw_bytes: &[u8],
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        writer.write_all(raw_bytes)
    }
}

/// Lazy flat vector data — doc_ids in memory, vectors accessed via range reads.
///
/// Only the doc_id index (~6 bytes/vector) is loaded into memory.
/// Vector data stays on disk and is accessed via mmap-backed range reads.
/// Element size depends on quantization: f32=4, f16=2, uint8=1 bytes/dim.
///
/// Used for:
/// - Brute-force search (batched scoring with native-precision SIMD)
/// - Reranking (read individual vectors by doc_id via binary search)
/// - doc() hydration (dequantize to f32 for stored documents)
/// - Merge streaming (chunked raw vector bytes + doc_id iteration)
#[derive(Debug, Clone)]
pub struct LazyFlatVectorData {
    /// Vector dimension
    pub dim: usize,
    /// Total number of vectors
    pub num_vectors: usize,
    /// Storage quantization type
    pub quantization: DenseVectorQuantization,
    /// In-memory doc_id index: (doc_id, ordinal) per vector
    pub doc_ids: Vec<(u32, u16)>,
    /// Lazy handle to this field's flat data region in the .vectors file
    handle: LazyFileSlice,
    /// Byte offset within handle where raw vector data starts (after header)
    vectors_offset: u64,
    /// Bytes per vector element (cached from quantization.element_size())
    element_size: usize,
}

impl LazyFlatVectorData {
    /// Open from a lazy file slice pointing to the flat binary data region.
    ///
    /// Reads header (16 bytes) + doc_ids (~6 bytes/vector) into memory.
    /// Vector data stays lazy on disk.
    pub async fn open(handle: LazyFileSlice) -> io::Result<Self> {
        // Read header: magic(4) + dim(4) + num_vectors(4) + quant_type(1) + pad(3) = 16 bytes
        let header = handle
            .read_bytes_range(0..FLAT_BINARY_HEADER_SIZE as u64)
            .await?;
        let hdr = header.as_slice();

        let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
        if magic != FLAT_BINARY_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid FlatVectorData binary magic",
            ));
        }

        let dim = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
        let num_vectors = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]) as usize;
        let quantization = DenseVectorQuantization::from_tag(hdr[12]).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown quantization tag: {}", hdr[12]),
            )
        })?;
        let element_size = quantization.element_size();

        // Read doc_ids section (small: 6 bytes per vector)
        let vectors_byte_len = num_vectors * dim * element_size;
        let doc_ids_start = (FLAT_BINARY_HEADER_SIZE + vectors_byte_len) as u64;
        let doc_ids_byte_len = (num_vectors * DOC_ID_ENTRY_SIZE) as u64;

        let doc_ids_bytes = handle
            .read_bytes_range(doc_ids_start..doc_ids_start + doc_ids_byte_len)
            .await?;
        let d = doc_ids_bytes.as_slice();

        let mut doc_ids = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let off = i * DOC_ID_ENTRY_SIZE;
            let doc_id = u32::from_le_bytes([d[off], d[off + 1], d[off + 2], d[off + 3]]);
            let ordinal = u16::from_le_bytes([d[off + 4], d[off + 5]]);
            doc_ids.push((doc_id, ordinal));
        }

        Ok(Self {
            dim,
            num_vectors,
            quantization,
            doc_ids,
            handle,
            vectors_offset: FLAT_BINARY_HEADER_SIZE as u64,
            element_size,
        })
    }

    /// Read a single vector by index, dequantized to f32.
    ///
    /// `out` must have length >= `self.dim`. Returns `Ok(())` on success.
    /// Used for ANN training and doc() hydration where f32 is needed.
    pub async fn read_vector_into(&self, idx: usize, out: &mut [f32]) -> io::Result<()> {
        debug_assert!(out.len() >= self.dim);
        let vec_byte_len = self.dim * self.element_size;
        let byte_offset = self.vectors_offset + (idx * vec_byte_len) as u64;
        let bytes = self
            .handle
            .read_bytes_range(byte_offset..byte_offset + vec_byte_len as u64)
            .await?;
        let raw = bytes.as_slice();

        dequantize_raw(raw, self.quantization, self.dim, out);
        Ok(())
    }

    /// Read a single vector by index, dequantized to f32 (allocates a new Vec<f32>).
    pub async fn get_vector(&self, idx: usize) -> io::Result<Vec<f32>> {
        let mut vector = vec![0f32; self.dim];
        self.read_vector_into(idx, &mut vector).await?;
        Ok(vector)
    }

    /// Read a single vector's raw bytes (no dequantization) into a caller-provided buffer.
    ///
    /// `out` must have length >= `self.vector_byte_size()`.
    /// Used for native-precision reranking where raw quantized bytes are scored directly.
    pub async fn read_vector_raw_into(&self, idx: usize, out: &mut [u8]) -> io::Result<()> {
        let vbs = self.vector_byte_size();
        debug_assert!(out.len() >= vbs);
        let byte_offset = self.vectors_offset + (idx * vbs) as u64;
        let bytes = self
            .handle
            .read_bytes_range(byte_offset..byte_offset + vbs as u64)
            .await?;
        out[..vbs].copy_from_slice(bytes.as_slice());
        Ok(())
    }

    /// Read a contiguous batch of raw quantized bytes by index range.
    ///
    /// Returns raw bytes for vectors `[start_idx..start_idx+count)`.
    /// Bytes are in native quantized format — pass to `batch_cosine_scores_f16/u8`
    /// or `batch_cosine_scores` (for f32) for scoring.
    pub async fn read_vectors_batch(
        &self,
        start_idx: usize,
        count: usize,
    ) -> io::Result<OwnedBytes> {
        debug_assert!(start_idx + count <= self.num_vectors);
        let vec_byte_len = self.dim * self.element_size;
        let byte_offset = self.vectors_offset + (start_idx * vec_byte_len) as u64;
        let byte_len = (count * vec_byte_len) as u64;
        self.handle
            .read_bytes_range(byte_offset..byte_offset + byte_len)
            .await
    }

    /// Find flat indexes for a given doc_id via binary search on sorted doc_ids.
    ///
    /// doc_ids are sorted by (doc_id, ordinal) — segment builder adds docs
    /// sequentially. Returns a slice of (doc_id, ordinal) entries; the position
    /// of each entry in `self.doc_ids` is its flat vector index.
    ///
    /// Returns `(start_index, slice)` where start_index is the position in doc_ids.
    pub fn flat_indexes_for_doc(&self, doc_id: u32) -> (usize, &[(u32, u16)]) {
        let start = self.doc_ids.partition_point(|&(id, _)| id < doc_id);
        let end = start + self.doc_ids[start..].partition_point(|&(id, _)| id == doc_id);
        (start, &self.doc_ids[start..end])
    }

    /// Get doc_id and ordinal at index (from in-memory index).
    #[inline]
    pub fn get_doc_id(&self, idx: usize) -> (u32, u16) {
        self.doc_ids[idx]
    }

    /// Bytes per vector in storage.
    #[inline]
    pub fn vector_byte_size(&self) -> usize {
        self.dim * self.element_size
    }

    /// Total byte length of raw vector data (for chunked merger streaming).
    pub fn vector_bytes_len(&self) -> u64 {
        (self.num_vectors as u64) * (self.vector_byte_size() as u64)
    }

    /// Byte offset where vector data starts (for direct handle access in merger).
    pub fn vectors_byte_offset(&self) -> u64 {
        self.vectors_offset
    }

    /// Access the underlying lazy file handle (for chunked byte-range reads in merger).
    pub fn handle(&self) -> &LazyFileSlice {
        &self.handle
    }

    /// Estimated memory usage (only doc_ids are in memory).
    pub fn estimated_memory_bytes(&self) -> usize {
        self.doc_ids.capacity() * size_of::<(u32, u16)>() + size_of::<Self>()
    }
}

/// IVF-RaBitQ index data with embedded centroids and codebook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFRaBitQIndexData {
    pub index: crate::structures::IVFRaBitQIndex,
    pub centroids: crate::structures::CoarseCentroids,
    pub codebook: crate::structures::RaBitQCodebook,
}

impl IVFRaBitQIndexData {
    pub fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        bincode::serde::decode_from_slice(data, bincode::config::standard())
            .map(|(v, _)| v)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// ScaNN index data with embedded centroids and codebook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaNNIndexData {
    pub index: crate::structures::IVFPQIndex,
    pub centroids: crate::structures::CoarseCentroids,
    pub codebook: crate::structures::PQCodebook,
}

impl ScaNNIndexData {
    pub fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        bincode::serde::decode_from_slice(data, bincode::config::standard())
            .map(|(v, _)| v)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
