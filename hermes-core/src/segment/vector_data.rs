//! Vector index data structures shared between builder and reader

use serde::{Deserialize, Serialize};

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
