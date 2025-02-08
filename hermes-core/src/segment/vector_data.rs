//! Vector index data structures shared between builder and reader

use std::mem::size_of;

use serde::{Deserialize, Serialize};

/// Magic number for binary flat vector format ("FVD2" in little-endian)
const FLAT_BINARY_MAGIC: u32 = 0x46564432;

/// Binary header: magic(u32) + dim(u32) + num_vectors(u32)
const FLAT_BINARY_HEADER_SIZE: usize = 3 * size_of::<u32>();
/// Per-vector element size
const FLOAT_SIZE: usize = size_of::<f32>();
/// Per-doc_id entry: doc_id(u32) + ordinal(u16)
const DOC_ID_ENTRY_SIZE: usize = size_of::<u32>() + size_of::<u16>();

/// Flat vector data for brute-force search.
///
/// Uses a single contiguous `Vec<f32>` instead of `Vec<Vec<f32>>`.
/// Loading is a single bulk memcpy (1 allocation) instead of N separate
/// allocations with float-by-float parsing. For 3.3M vectors at 768 dims
/// this reduces load time from ~36s to ~1s.
#[derive(Debug, Clone)]
pub struct FlatVectorData {
    pub dim: usize,
    /// Flat contiguous vector storage: num_vectors * dim f32 values.
    /// Access vector i via `vectors[i*dim .. (i+1)*dim]`.
    vectors: Vec<f32>,
    /// Document IDs with ordinals: (doc_id, ordinal) pairs
    /// Ordinal tracks which vector in a multi-valued field
    pub doc_ids: Vec<(u32, u16)>,
}

impl FlatVectorData {
    /// Number of vectors
    #[inline]
    pub fn num_vectors(&self) -> usize {
        self.vectors.len().checked_div(self.dim).unwrap_or(0)
    }

    /// Get vector at index as a &[f32] slice.
    ///
    /// # Panics
    /// Panics if `idx >= num_vectors`.
    #[inline]
    pub fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        &self.vectors[start..start + self.dim]
    }

    /// Get doc_id and ordinal at index.
    #[inline]
    pub fn get_doc_id(&self, idx: usize) -> (u32, u16) {
        self.doc_ids[idx]
    }

    /// Raw flat vector storage as byte slice for bulk streaming.
    /// Returns the contiguous f32 data reinterpreted as bytes.
    #[inline]
    pub fn vectors_as_bytes(&self) -> &[u8] {
        // SAFETY: reinterpret &[f32] as &[u8] — same layout, just wider view
        unsafe {
            std::slice::from_raw_parts(
                self.vectors.as_ptr() as *const u8,
                self.vectors.len() * FLOAT_SIZE,
            )
        }
    }

    /// Stream vectors and doc_ids from this FlatVectorData to a writer.
    /// `doc_id_offset` is added to each doc_id for multi-segment merges.
    pub fn stream_to_writer(
        &self,
        writer: &mut dyn std::io::Write,
        doc_id_offset: u32,
    ) -> std::io::Result<()> {
        // Bulk write all vector bytes
        writer.write_all(self.vectors_as_bytes())?;
        // Write doc_ids with offset
        for &(doc_id, ordinal) in &self.doc_ids {
            writer.write_all(&(doc_id_offset + doc_id).to_le_bytes())?;
            writer.write_all(&ordinal.to_le_bytes())?;
        }
        Ok(())
    }

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

    /// Estimate memory usage
    pub fn estimated_memory_bytes(&self) -> usize {
        let vectors_bytes = self.vectors.capacity() * FLOAT_SIZE;
        let doc_ids_bytes = self.doc_ids.capacity() * size_of::<(u32, u16)>();
        vectors_bytes + doc_ids_bytes + size_of::<Self>()
    }

    /// Deserialize from binary format. Single bulk memcpy for vectors.
    ///
    /// Parses the 12-byte header, bulk-copies vectors as one contiguous Vec<f32>,
    /// and parses doc_ids. For 3.3M vectors at 768 dims this is ~1s vs ~36s
    /// with the old Vec<Vec<f32>> approach (1 allocation vs 3.3M allocations).
    pub fn from_binary_bytes(data: &[u8]) -> std::io::Result<Self> {
        if data.len() < FLAT_BINARY_HEADER_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "FlatVectorData binary too short",
            ));
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != FLAT_BINARY_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid FlatVectorData binary magic",
            ));
        }

        let dim = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let num_vectors = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

        let vectors_byte_len = num_vectors * dim * FLOAT_SIZE;
        let doc_ids_start = FLAT_BINARY_HEADER_SIZE + vectors_byte_len;
        let doc_ids_byte_len = num_vectors * DOC_ID_ENTRY_SIZE;

        if data.len() < doc_ids_start + doc_ids_byte_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "FlatVectorData binary truncated",
            ));
        }

        // Bulk memcpy: one allocation of num_vectors*dim floats
        let total_floats = num_vectors * dim;
        let mut vectors = vec![0f32; total_floats];
        // SAFETY: copy raw bytes into properly-aligned Vec<f32>
        unsafe {
            std::ptr::copy_nonoverlapping(
                data[FLAT_BINARY_HEADER_SIZE..].as_ptr(),
                vectors.as_mut_ptr() as *mut u8,
                vectors_byte_len,
            );
        }

        // Parse doc_ids (small: ~6 bytes per vector vs ~3072 bytes per 768-dim vector)
        let mut doc_ids = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let off = doc_ids_start + i * DOC_ID_ENTRY_SIZE;
            let doc_id =
                u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            let ordinal = u16::from_le_bytes([data[off + 4], data[off + 5]]);
            doc_ids.push((doc_id, ordinal));
        }

        Ok(FlatVectorData {
            dim,
            vectors,
            doc_ids,
        })
    }

    /// Compute the serialized size without actually serializing.
    pub fn serialized_binary_size(index_dim: usize, num_vectors: usize) -> usize {
        FLAT_BINARY_HEADER_SIZE
            + num_vectors * index_dim * FLOAT_SIZE
            + num_vectors * DOC_ID_ENTRY_SIZE
    }

    /// Stream directly from flat f32 storage to a writer (zero-buffer serialization).
    ///
    /// `flat_vectors` is contiguous storage of dim*n floats.
    /// `original_dim` is the dimension in flat_vectors (may differ from index_dim for MRL).
    pub fn serialize_binary_from_flat_streaming(
        index_dim: usize,
        flat_vectors: &[f32],
        original_dim: usize,
        doc_ids: &[(u32, u16)],
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        let num_vectors = doc_ids.len();

        writer.write_all(&FLAT_BINARY_MAGIC.to_le_bytes())?;
        writer.write_all(&(index_dim as u32).to_le_bytes())?;
        writer.write_all(&(num_vectors as u32).to_le_bytes())?;

        if index_dim == original_dim {
            // No trimming — write all floats directly
            // SAFETY: reinterpret f32 slice as bytes for efficient bulk write
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    flat_vectors.as_ptr() as *const u8,
                    flat_vectors.len() * FLOAT_SIZE,
                )
            };
            writer.write_all(bytes)?;
        } else {
            // Trim each vector to index_dim (matryoshka/MRL)
            for i in 0..num_vectors {
                let start = i * original_dim;
                let slice = &flat_vectors[start..start + index_dim];
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const u8, index_dim * FLOAT_SIZE)
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
