//! Types for segment reader

use std::io::Cursor;
use std::sync::Arc;

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::DocId;
use crate::directories::{AsyncFileRead, LazyFileHandle};
use crate::structures::{
    BlockSparsePostingList, IVFPQIndex, IVFRaBitQIndex, PQCodebook, RaBitQCodebook, RaBitQIndex,
};

use super::super::vector_data::FlatVectorData;

/// Vector index type - Flat, RaBitQ, IVF-RaBitQ, or ScaNN (IVF-PQ)
#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
pub enum VectorIndex {
    /// Flat - brute-force search over raw vectors (accumulating state)
    Flat(Arc<FlatVectorData>),
    /// RaBitQ - binary quantization, good for small datasets
    RaBitQ(Arc<RaBitQIndex>),
    /// IVF-RaBitQ - inverted file with RaBitQ, good for medium datasets
    IVF {
        index: Arc<IVFRaBitQIndex>,
        codebook: Arc<RaBitQCodebook>,
    },
    /// ScaNN (IVF-PQ) - product quantization with OPQ, best for large datasets
    ScaNN {
        index: Arc<IVFPQIndex>,
        codebook: Arc<PQCodebook>,
    },
}

/// Sparse vector index for a field: lazy loading with mmap
///
/// Only stores the offset table in memory. Posting lists are loaded
/// on-demand during queries using mmap for memory efficiency.
#[derive(Clone)]
pub struct SparseIndex {
    /// Lazy file handle for on-demand loading
    handle: LazyFileHandle,
    /// Offset table: (offset, length) indexed by dimension ID
    /// (0, 0) means dimension not present
    offsets: Arc<Vec<(u64, u32)>>,
    /// Cache of loaded posting lists (LRU-style, dimension_id -> posting_list)
    cache: Arc<RwLock<FxHashMap<u32, Arc<BlockSparsePostingList>>>>,
    /// Total document count in this segment (for IDF computation)
    pub total_docs: u32,
    /// Total sparse vectors in this segment (for multi-valued IDF)
    pub total_vectors: u32,
}

impl SparseIndex {
    /// Create a new lazy sparse index
    pub fn new(
        handle: LazyFileHandle,
        offsets: Vec<(u64, u32)>,
        total_docs: u32,
        total_vectors: u32,
    ) -> Self {
        Self {
            handle,
            offsets: Arc::new(offsets),
            cache: Arc::new(RwLock::new(FxHashMap::default())),
            total_docs,
            total_vectors,
        }
    }

    /// Get offset and length for a dimension, or None if not present
    #[inline]
    fn get_offset(&self, dim_id: u32) -> Option<(u64, u32)> {
        self.offsets
            .get(dim_id as usize)
            .filter(|(_, l)| *l > 0)
            .copied()
    }

    /// Deserialize and cache a posting list from raw bytes
    fn deserialize_and_cache(
        &self,
        dim_id: u32,
        data: &[u8],
    ) -> crate::Result<Arc<BlockSparsePostingList>> {
        let posting_list = BlockSparsePostingList::deserialize(&mut Cursor::new(data))?;
        let pl = Arc::new(posting_list);

        // Cache it
        {
            let mut cache = self.cache.write();
            cache.insert(dim_id, Arc::clone(&pl));
        }

        Ok(pl)
    }

    /// Get posting list for a dimension (async, loads on-demand)
    pub async fn get_posting(
        &self,
        dim_id: u32,
    ) -> crate::Result<Option<Arc<BlockSparsePostingList>>> {
        // Check bounds
        let (offset, length) = match self.get_offset(dim_id) {
            Some(v) => v,
            None => return Ok(None),
        };

        // Check cache first
        if let Some(pl) = self.get_cached(dim_id) {
            return Ok(Some(pl));
        }

        // Load from file (mmap range read)
        let data = self
            .handle
            .read_bytes_range(offset..offset + length as u64)
            .await
            .map_err(crate::Error::Io)?;

        Ok(Some(self.deserialize_and_cache(dim_id, data.as_slice())?))
    }

    /// Check if dimension exists (without loading)
    #[inline]
    pub fn has_dimension(&self, dim_id: u32) -> bool {
        self.get_offset(dim_id).is_some()
    }

    /// Get the number of dimensions in the index
    #[inline]
    pub fn num_dimensions(&self) -> usize {
        self.offsets.len()
    }

    /// Clear the cache (useful for memory pressure)
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    /// Get cache size (number of cached posting lists)
    pub fn cache_size(&self) -> usize {
        self.cache.read().len()
    }

    /// Iterate over all active dimension IDs (dimensions that have posting lists)
    /// This is synchronous and doesn't load the posting lists
    pub fn active_dimensions(&self) -> impl Iterator<Item = u32> + '_ {
        self.offsets
            .iter()
            .enumerate()
            .filter(|(_, (_, l))| *l > 0)
            .map(|(i, _)| i as u32)
    }

    /// Get cached posting list if available (for synchronous access)
    /// Returns None if not cached (doesn't load from disk)
    pub fn get_cached(&self, dim_id: u32) -> Option<Arc<BlockSparsePostingList>> {
        self.cache.read().get(&dim_id).cloned()
    }

    /// Compute IDF (inverse document frequency) for a dimension
    /// Uses cached posting if available, otherwise returns 0
    #[inline]
    pub fn idf(&self, dim_id: u32) -> f32 {
        if let Some(pl) = self.get_cached(dim_id) {
            let df = pl.doc_count() as f32;
            if df > 0.0 {
                let n = self.total_vectors.max(self.total_docs) as f32;
                (n / df).ln().max(0.0)
            } else {
                0.0
            }
        } else {
            // Not cached - return 0 (caller should use GlobalStats for accurate IDF)
            0.0
        }
    }

    /// Get IDF weights for multiple dimensions (uses cache only)
    pub fn idf_weights(&self, dim_ids: &[u32]) -> Vec<f32> {
        dim_ids.iter().map(|&d| self.idf(d)).collect()
    }

    /// Get posting list synchronously (blocking) - for merger use only
    /// Prefer async get_posting() for query-time access
    #[cfg(feature = "native")]
    pub fn get_posting_blocking(
        &self,
        dim_id: u32,
    ) -> crate::Result<Option<Arc<BlockSparsePostingList>>> {
        // Check bounds
        let (offset, length) = match self.get_offset(dim_id) {
            Some(v) => v,
            None => return Ok(None),
        };

        // Check cache first
        if let Some(pl) = self.get_cached(dim_id) {
            return Ok(Some(pl));
        }

        // Block on async read (only for merger, not query path)
        let handle = self.handle.clone();
        let data = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                handle
                    .read_bytes_range(offset..offset + length as u64)
                    .await
            })
        })
        .map_err(crate::Error::Io)?;

        Ok(Some(self.deserialize_and_cache(dim_id, data.as_slice())?))
    }
}

/// Vector search result with ordinal tracking for multi-value fields
///
/// Each result contains the combined score and individual contributions
/// from each ordinal (for multi-valued vector fields)
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// Document ID
    pub doc_id: DocId,
    /// Combined score (after applying combiner: Sum/Max/Avg)
    pub score: f32,
    /// Individual ordinal contributions: (ordinal, score)
    /// For single-value fields, this will have one entry with ordinal 0
    pub ordinals: Vec<(u32, f32)>,
}

impl VectorSearchResult {
    pub fn new(doc_id: DocId, score: f32, ordinals: Vec<(u32, f32)>) -> Self {
        Self {
            doc_id,
            score,
            ordinals,
        }
    }
}
