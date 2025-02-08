//! Vector index data structures shared between builder and reader

use std::io;
use std::mem::size_of;

use serde::{Deserialize, Serialize};

use crate::directories::{AsyncFileRead, LazyFileSlice, OwnedBytes};

/// Magic number for binary flat vector format ("FVD2" in little-endian)
const FLAT_BINARY_MAGIC: u32 = 0x46564432;

/// Binary header: magic(u32) + dim(u32) + num_vectors(u32)
const FLAT_BINARY_HEADER_SIZE: usize = 3 * size_of::<u32>();
/// Per-vector element size
const FLOAT_SIZE: usize = size_of::<f32>();
/// Per-doc_id entry: doc_id(u32) + ordinal(u16)
const DOC_ID_ENTRY_SIZE: usize = size_of::<u32>() + size_of::<u16>();

/// Flat vector binary format helpers for writing.
///
/// Binary format: `[magic(u32)][dim(u32)][num_vectors(u32)][vectors: N×dim×f32][doc_ids: N×(u32+u16)]`
///
/// Reading is handled by [`LazyFlatVectorData`] which loads only doc_ids into memory
/// and accesses vector data lazily via mmap-backed range reads.
pub struct FlatVectorData;

impl FlatVectorData {
    /// Write the binary header (magic + dim + num_vectors) to a writer.
    pub fn write_binary_header(
        dim: usize,
        num_vectors: usize,
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        writer.write_all(&FLAT_BINARY_MAGIC.to_le_bytes())?;
        writer.write_all(&(dim as u32).to_le_bytes())?;
        writer.write_all(&(num_vectors as u32).to_le_bytes())?;
        Ok(())
    }

    /// Compute the serialized size without actually serializing.
    pub fn serialized_binary_size(dim: usize, num_vectors: usize) -> usize {
        FLAT_BINARY_HEADER_SIZE + num_vectors * dim * FLOAT_SIZE + num_vectors * DOC_ID_ENTRY_SIZE
    }

    /// Stream directly from flat f32 storage to a writer (zero-buffer serialization).
    ///
    /// `flat_vectors` is contiguous storage of dim*n floats.
    /// `original_dim` is the dimension in flat_vectors (may differ from `dim` for MRL).
    pub fn serialize_binary_from_flat_streaming(
        dim: usize,
        flat_vectors: &[f32],
        original_dim: usize,
        doc_ids: &[(u32, u16)],
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        let num_vectors = doc_ids.len();

        writer.write_all(&FLAT_BINARY_MAGIC.to_le_bytes())?;
        writer.write_all(&(dim as u32).to_le_bytes())?;
        writer.write_all(&(num_vectors as u32).to_le_bytes())?;

        if dim == original_dim {
            // No trimming — write all floats directly
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    flat_vectors.as_ptr() as *const u8,
                    flat_vectors.len() * FLOAT_SIZE,
                )
            };
            writer.write_all(bytes)?;
        } else {
            // Trim each vector to dim (matryoshka/MRL)
            for i in 0..num_vectors {
                let start = i * original_dim;
                let slice = &flat_vectors[start..start + dim];
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const u8, dim * FLOAT_SIZE)
                };
                writer.write_all(bytes)?;
            }
        }

        for &(doc_id, ordinal) in doc_ids {
            writer.write_all(&doc_id.to_le_bytes())?;
            writer.write_all(&ordinal.to_le_bytes())?;
        }

        Ok(())
    }
}

/// Lazy flat vector data — doc_ids in memory, vectors accessed via range reads.
///
/// Only the doc_id index (~6 bytes/vector) is loaded into memory.
/// Vector data (~dim×4 bytes/vector) stays on disk and is accessed via
/// mmap-backed range reads on demand. For 768-dim vectors this is
/// ~3KB per vector that stays lazy vs 6 bytes loaded.
///
/// Used for:
/// - Reranking (read individual vectors by doc_id)
/// - Merge streaming (bulk-read vector bytes in chunks)
#[derive(Debug, Clone)]
pub struct LazyFlatVectorData {
    /// Vector dimension
    pub dim: usize,
    /// Total number of vectors
    pub num_vectors: usize,
    /// In-memory doc_id index: (doc_id, ordinal) per vector
    pub doc_ids: Vec<(u32, u16)>,
    /// Lazy handle to this field's flat data region in the .vectors file
    handle: LazyFileSlice,
    /// Byte offset within handle where raw vector f32 data starts (after header)
    vectors_offset: u64,
}

impl LazyFlatVectorData {
    /// Open from a lazy file slice pointing to the flat binary data region.
    ///
    /// Reads header (12 bytes) + doc_ids (~6 bytes/vector) into memory.
    /// Vector data stays lazy on disk.
    pub async fn open(handle: LazyFileSlice) -> io::Result<Self> {
        // Read header: magic(4) + dim(4) + num_vectors(4) = 12 bytes
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

        // Read doc_ids section (small: 6 bytes per vector)
        let vectors_byte_len = num_vectors * dim * FLOAT_SIZE;
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
            doc_ids,
            handle,
            vectors_offset: FLAT_BINARY_HEADER_SIZE as u64,
        })
    }

    /// Read a single vector by index into a caller-provided slice (zero allocation).
    ///
    /// `out` must have length >= `self.dim`. Returns `Ok(())` on success.
    pub async fn read_vector_into(&self, idx: usize, out: &mut [f32]) -> io::Result<()> {
        debug_assert!(out.len() >= self.dim);
        let byte_offset = self.vectors_offset + (idx * self.dim * FLOAT_SIZE) as u64;
        let byte_len = (self.dim * FLOAT_SIZE) as u64;
        let bytes = self
            .handle
            .read_bytes_range(byte_offset..byte_offset + byte_len)
            .await?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_slice().as_ptr(),
                out.as_mut_ptr() as *mut u8,
                self.dim * FLOAT_SIZE,
            );
        }
        Ok(())
    }

    /// Read a single vector by index (allocates a new Vec<f32>).
    pub async fn get_vector(&self, idx: usize) -> io::Result<Vec<f32>> {
        let mut vector = vec![0f32; self.dim];
        self.read_vector_into(idx, &mut vector).await?;
        Ok(vector)
    }

    /// Read all raw vector bytes at once (for bulk streaming in merger).
    ///
    /// Returns the contiguous f32 data as owned bytes. Caller writes it
    /// directly to the output writer — no f32 parsing needed.
    pub async fn read_all_vector_bytes(&self) -> io::Result<OwnedBytes> {
        let byte_len = (self.num_vectors * self.dim * FLOAT_SIZE) as u64;
        self.handle
            .read_bytes_range(self.vectors_offset..self.vectors_offset + byte_len)
            .await
    }

    /// Get doc_id and ordinal at index (from in-memory index).
    #[inline]
    pub fn get_doc_id(&self, idx: usize) -> (u32, u16) {
        self.doc_ids[idx]
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
        serde_json::to_vec(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        serde_json::from_slice(data)
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
        serde_json::to_vec(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        serde_json::from_slice(data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
