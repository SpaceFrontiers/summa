//! Vector index data structures shared between builder and reader

use serde::{Deserialize, Serialize};

/// Magic number for binary flat vector format ("FVD2" in little-endian)
const FLAT_BINARY_MAGIC: u32 = 0x46564432;

/// Flat vector data for brute-force search (accumulating state)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatVectorData {
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
    /// Document IDs with ordinals: (doc_id, ordinal) pairs
    /// Ordinal tracks which vector in a multi-valued field
    pub doc_ids: Vec<(u32, u16)>,
}

impl FlatVectorData {
    /// Estimate memory usage
    pub fn estimated_memory_bytes(&self) -> usize {
        // Vec<Vec<f32>>: each inner vec has capacity * 4 bytes + Vec overhead
        let vec_overhead = std::mem::size_of::<Vec<f32>>();
        let vectors_bytes: usize = self
            .vectors
            .iter()
            .map(|v| v.capacity() * 4 + vec_overhead)
            .sum();
        // doc_ids: (u32, u16) = 6 bytes + padding = 8 bytes each
        let doc_ids_bytes = self.doc_ids.capacity() * 8;
        vectors_bytes + doc_ids_bytes + vec_overhead * 2
    }

    /// Serialize to compact binary format.
    ///
    /// Format: magic(4) + dim(4) + num_vectors(4) + vectors(dim*n*4) + doc_ids(n*6)
    /// Much more compact than JSON and avoids intermediate allocations.
    pub fn to_binary_bytes(&self) -> Vec<u8> {
        let num_vectors = self.doc_ids.len();
        let total = 12 + num_vectors * self.dim * 4 + num_vectors * 6;
        let mut buf = Vec::with_capacity(total);

        buf.extend_from_slice(&FLAT_BINARY_MAGIC.to_le_bytes());
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
        buf.extend_from_slice(&(num_vectors as u32).to_le_bytes());

        for vec in &self.vectors {
            for &val in vec {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        for &(doc_id, ordinal) in &self.doc_ids {
            buf.extend_from_slice(&doc_id.to_le_bytes());
            buf.extend_from_slice(&ordinal.to_le_bytes());
        }

        buf
    }

    /// Serialize directly from flat f32 storage (avoids Vec<Vec<f32>> intermediate).
    ///
    /// `flat_vectors` is contiguous storage of dim*n floats.
    /// `original_dim` is the dimension in flat_vectors (may differ from index_dim for MRL).
    pub fn serialize_binary_from_flat(
        index_dim: usize,
        flat_vectors: &[f32],
        original_dim: usize,
        doc_ids: &[(u32, u16)],
    ) -> Vec<u8> {
        let num_vectors = doc_ids.len();
        let total = 12 + num_vectors * index_dim * 4 + num_vectors * 6;
        let mut buf = Vec::with_capacity(total);

        buf.extend_from_slice(&FLAT_BINARY_MAGIC.to_le_bytes());
        buf.extend_from_slice(&(index_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(num_vectors as u32).to_le_bytes());

        if index_dim == original_dim {
            // No trimming — write all floats directly
            for &val in flat_vectors {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        } else {
            // Trim each vector to index_dim (matryoshka/MRL)
            for i in 0..num_vectors {
                let start = i * original_dim;
                for j in 0..index_dim {
                    buf.extend_from_slice(&flat_vectors[start + j].to_le_bytes());
                }
            }
        }

        for &(doc_id, ordinal) in doc_ids {
            buf.extend_from_slice(&doc_id.to_le_bytes());
            buf.extend_from_slice(&ordinal.to_le_bytes());
        }

        buf
    }

    /// Deserialize from binary format.
    pub fn from_binary_bytes(data: &[u8]) -> std::io::Result<Self> {
        if data.len() < 12 {
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

        let vectors_start = 12;
        let vectors_byte_len = num_vectors * dim * 4;
        let doc_ids_start = vectors_start + vectors_byte_len;
        let doc_ids_byte_len = num_vectors * 6;

        if data.len() < doc_ids_start + doc_ids_byte_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "FlatVectorData binary truncated",
            ));
        }

        let mut vectors = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let mut vec = Vec::with_capacity(dim);
            let base = vectors_start + i * dim * 4;
            for j in 0..dim {
                let off = base + j * 4;
                vec.push(f32::from_le_bytes([
                    data[off],
                    data[off + 1],
                    data[off + 2],
                    data[off + 3],
                ]));
            }
            vectors.push(vec);
        }

        let mut doc_ids = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let off = doc_ids_start + i * 6;
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
