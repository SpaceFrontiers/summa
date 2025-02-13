//! Types for segment reader

use std::sync::{Arc, OnceLock};

use crate::DocId;
use crate::directories::{FileHandle, OwnedBytes};
use crate::structures::{
    BlockSparsePostingList, IVFPQIndex, IVFRaBitQIndex, PQCodebook, RaBitQCodebook, RaBitQIndex,
    SparseBlock, SparseSkipEntry,
};

/// Vector index type - RaBitQ, IVF-RaBitQ, or ScaNN (IVF-PQ)
///
/// Raw flat vectors are stored separately in [`LazyFlatVectorData`] and accessed
/// via mmap for reranking and merge. This enum only holds ANN indexes.
///
/// All variants are **lazy**: `OwnedBytes` (zero-copy mmap ref) are stored on
/// construction, deserialization is deferred to first search access via `OnceLock`.
/// No heap copies during segment load.
#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
pub enum VectorIndex {
    /// RaBitQ - lazy JSON deserialization on first access
    RaBitQ(Arc<LazyRaBitQ>),
    /// IVF-RaBitQ - lazy deserialization on first access
    IVF(Arc<LazyIVF>),
    /// ScaNN (IVF-PQ) - lazy deserialization on first access
    ScaNN(Arc<LazyScaNN>),
}

/// Lazy RaBitQ index — defers serde_json deserialization to first access
///
/// Stores `OwnedBytes` which for mmap directories is a zero-copy reference.
/// The mmap pages are only paged into physical RAM when deserialization happens.
pub struct LazyRaBitQ {
    raw: OwnedBytes,
    resolved: OnceLock<Option<Arc<RaBitQIndex>>>,
}

impl LazyRaBitQ {
    pub fn new(raw: OwnedBytes) -> Self {
        Self {
            raw,
            resolved: OnceLock::new(),
        }
    }

    pub fn get(&self) -> Option<&Arc<RaBitQIndex>> {
        self.resolved
            .get_or_init(
                || match serde_json::from_slice::<RaBitQIndex>(self.raw.as_slice()) {
                    Ok(idx) => Some(Arc::new(idx)),
                    Err(e) => {
                        log::warn!("[lazy_rabitq] deserialization failed: {}", e);
                        None
                    }
                },
            )
            .as_ref()
    }

    pub fn estimated_memory_bytes(&self) -> usize {
        match self.resolved.get() {
            Some(Some(idx)) => idx.estimated_memory_bytes(),
            _ => self.raw.len(),
        }
    }
}

/// Lazy IVF-RaBitQ index — defers bincode deserialization to first access
///
/// Stores `OwnedBytes` which for mmap directories is a zero-copy reference.
/// The mmap pages are only paged into physical RAM when deserialization happens.
pub struct LazyIVF {
    raw: OwnedBytes,
    resolved: OnceLock<Option<(Arc<IVFRaBitQIndex>, Arc<RaBitQCodebook>)>>,
}

impl LazyIVF {
    pub fn new(raw: OwnedBytes) -> Self {
        Self {
            raw,
            resolved: OnceLock::new(),
        }
    }

    pub fn get(&self) -> Option<(&Arc<IVFRaBitQIndex>, &Arc<RaBitQCodebook>)> {
        self.resolved
            .get_or_init(|| {
                match super::super::vector_data::IVFRaBitQIndexData::from_bytes(&self.raw) {
                    Ok(data) => Some((Arc::new(data.index), Arc::new(data.codebook))),
                    Err(e) => {
                        log::warn!("[lazy_ivf] deserialization failed: {}", e);
                        None
                    }
                }
            })
            .as_ref()
            .map(|(i, c)| (i, c))
    }

    pub fn estimated_memory_bytes(&self) -> usize {
        match self.resolved.get() {
            Some(Some((index, codebook))) => {
                index.estimated_memory_bytes() + codebook.estimated_memory_bytes()
            }
            _ => self.raw.len(),
        }
    }
}

/// Lazy ScaNN (IVF-PQ) index — defers bincode deserialization to first access
///
/// Stores `OwnedBytes` which for mmap directories is a zero-copy reference.
/// The mmap pages are only paged into physical RAM when deserialization happens.
pub struct LazyScaNN {
    raw: OwnedBytes,
    resolved: OnceLock<Option<(Arc<IVFPQIndex>, Arc<PQCodebook>)>>,
}

impl LazyScaNN {
    pub fn new(raw: OwnedBytes) -> Self {
        Self {
            raw,
            resolved: OnceLock::new(),
        }
    }

    pub fn get(&self) -> Option<(&Arc<IVFPQIndex>, &Arc<PQCodebook>)> {
        self.resolved
            .get_or_init(|| {
                match super::super::vector_data::ScaNNIndexData::from_bytes(&self.raw) {
                    Ok(data) => Some((Arc::new(data.index), Arc::new(data.codebook))),
                    Err(e) => {
                        log::warn!("[lazy_scann] deserialization failed: {}", e);
                        None
                    }
                }
            })
            .as_ref()
            .map(|(i, c)| (i, c))
    }

    pub fn estimated_memory_bytes(&self) -> usize {
        match self.resolved.get() {
            Some(Some((index, codebook))) => {
                index.estimated_memory_bytes() + codebook.estimated_memory_bytes()
            }
            _ => self.raw.len(),
        }
    }
}

impl VectorIndex {
    /// Estimate memory usage of this vector index
    pub fn estimated_memory_bytes(&self) -> usize {
        match self {
            VectorIndex::RaBitQ(lazy) => lazy.estimated_memory_bytes(),
            VectorIndex::IVF(lazy) => lazy.estimated_memory_bytes(),
            VectorIndex::ScaNN(lazy) => lazy.estimated_memory_bytes(),
        }
    }
}

/// Raw dimension data for zero-copy merge (no block deserialization).
///
/// Contains skip entries parsed on-demand from the mmap-backed skip section,
/// metadata, and raw block data bytes read directly from mmap.
pub struct DimRawData {
    /// Skip entries for this dimension (parsed from skip section)
    pub skip_entries: Vec<SparseSkipEntry>,
    /// Total document count in this dimension's posting list
    pub doc_count: u32,
    /// Global max weight across all blocks
    pub global_max_weight: f32,
    /// Raw block data bytes (single contiguous mmap read)
    pub raw_block_data: crate::directories::OwnedBytes,
}

/// SoA (Struct-of-Arrays) dimension table for cache-friendly access.
///
/// Binary search only touches `dim_ids` (352KB for 88K dims, fits L2 cache).
/// IDF computation only touches `doc_counts`. Scoring only touches `max_weights`.
/// All arrays are parallel — same index accesses the same dimension.
#[derive(Clone)]
pub struct DimensionTable {
    /// Dimension IDs, sorted for binary search
    pub dim_ids: Vec<u32>,
    /// Byte offset in file where block data starts (relative to file start)
    pub block_offsets: Vec<u64>,
    /// Index into the skip section (entry index, multiply by 20 for byte offset)
    pub skip_starts: Vec<u32>,
    /// Number of skip entries (= number of blocks) per dimension
    pub skip_counts: Vec<u32>,
    /// Total documents per dimension's posting list
    pub doc_counts: Vec<u32>,
    /// Global max weight per dimension
    pub max_weights: Vec<f32>,
}

impl DimensionTable {
    /// Create an empty table
    pub fn new() -> Self {
        Self {
            dim_ids: Vec::new(),
            block_offsets: Vec::new(),
            skip_starts: Vec::new(),
            skip_counts: Vec::new(),
            doc_counts: Vec::new(),
            max_weights: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            dim_ids: Vec::with_capacity(cap),
            block_offsets: Vec::with_capacity(cap),
            skip_starts: Vec::with_capacity(cap),
            skip_counts: Vec::with_capacity(cap),
            doc_counts: Vec::with_capacity(cap),
            max_weights: Vec::with_capacity(cap),
        }
    }

    /// Push a dimension entry
    pub fn push(
        &mut self,
        dim_id: u32,
        block_offset: u64,
        skip_start: u32,
        skip_count: u32,
        doc_count: u32,
        max_weight: f32,
    ) {
        self.dim_ids.push(dim_id);
        self.block_offsets.push(block_offset);
        self.skip_starts.push(skip_start);
        self.skip_counts.push(skip_count);
        self.doc_counts.push(doc_count);
        self.max_weights.push(max_weight);
    }

    /// Sort all arrays by dim_id (for binary search).
    /// Skips allocation when already sorted (the common case from builder/merger).
    pub fn sort_by_dim_id(&mut self) {
        if self.dim_ids.windows(2).all(|w| w[0] <= w[1]) {
            return;
        }
        let mut indices: Vec<usize> = (0..self.dim_ids.len()).collect();
        indices.sort_unstable_by_key(|&i| self.dim_ids[i]);
        self.dim_ids = indices.iter().map(|&i| self.dim_ids[i]).collect();
        self.block_offsets = indices.iter().map(|&i| self.block_offsets[i]).collect();
        self.skip_starts = indices.iter().map(|&i| self.skip_starts[i]).collect();
        self.skip_counts = indices.iter().map(|&i| self.skip_counts[i]).collect();
        self.doc_counts = indices.iter().map(|&i| self.doc_counts[i]).collect();
        self.max_weights = indices.iter().map(|&i| self.max_weights[i]).collect();
    }

    /// Number of dimensions
    #[inline]
    pub fn len(&self) -> usize {
        self.dim_ids.len()
    }

    /// Binary search for dim_id, returns index
    #[inline]
    pub fn find(&self, dim_id: u32) -> Option<usize> {
        self.dim_ids.binary_search(&dim_id).ok()
    }

    /// Estimated heap memory in bytes (6 Vecs × capacity × element_size)
    pub fn estimated_memory_bytes(&self) -> usize {
        let n = self.dim_ids.capacity();
        n * (4 + 8 + 4 + 4 + 4 + 4) // u32×4 + u64×1 + f32×1 = 28 bytes per entry
    }
}

/// Sparse vector index for a field: lazy block loading via mmap.
///
/// V3 layout optimizations:
/// - **SoA dimension table**: binary search only touches `dim_ids` array (352KB
///   for 88K dims, fits L2 cache) instead of 2.8MB AoS.
/// - **OwnedBytes skip section**: zero-copy mmap reference to the contiguous
///   skip-entry region at the tail of the file. No heap allocation or parsing
///   during segment load.
/// - **Single tail read**: only footer + TOC + skip section are read during
///   load; block data at the front of the file is never touched.
#[derive(Clone)]
pub struct SparseIndex {
    /// File handle for block data reads (Inline for mmap, Lazy for HTTP)
    handle: FileHandle,
    /// SoA dimension table (sorted by dim_id)
    dims: Arc<DimensionTable>,
    /// Zero-copy skip section: all SparseSkipEntry structs contiguous (20B each)
    skip_bytes: Arc<OwnedBytes>,
    /// Total document count in this segment (for IDF computation)
    pub total_docs: u32,
    /// Total sparse vectors in this segment (for multi-valued IDF)
    pub total_vectors: u32,
}

impl SparseIndex {
    /// Create a new V3 sparse index with SoA dimension table and zero-copy skip section
    pub fn new(
        handle: FileHandle,
        dims: DimensionTable,
        skip_bytes: OwnedBytes,
        total_docs: u32,
        total_vectors: u32,
    ) -> Self {
        Self {
            handle,
            dims: Arc::new(dims),
            skip_bytes: Arc::new(skip_bytes),
            total_docs,
            total_vectors,
        }
    }

    /// Total number of skip entries across all dimensions
    #[inline]
    fn skip_entry_count(&self) -> usize {
        self.skip_bytes.len() / SparseSkipEntry::SIZE
    }

    /// Parse a single skip entry from the zero-copy skip section
    #[inline]
    pub fn read_skip_entry(&self, entry_idx: usize) -> SparseSkipEntry {
        SparseSkipEntry::read_at(&self.skip_bytes, entry_idx)
    }

    /// Get skip entries for a dimension as a Vec (parsed from skip section)
    fn dim_skip_entries_vec(&self, idx: usize) -> Vec<SparseSkipEntry> {
        let start = self.dims.skip_starts[idx] as usize;
        let count = self.dims.skip_counts[idx] as usize;
        (0..count)
            .map(|i| self.read_skip_entry(start + i))
            .collect()
    }

    /// Load a single block via mmap (OS page cache handles caching)
    async fn load_block_at(&self, dim_idx: usize, block_idx: usize) -> crate::Result<SparseBlock> {
        let entry = self.read_skip_entry(self.dims.skip_starts[dim_idx] as usize + block_idx);
        let base = self.dims.block_offsets[dim_idx];
        let abs_offset = base + entry.offset;
        let data = self
            .handle
            .read_bytes_range(abs_offset..abs_offset + entry.length as u64)
            .await
            .map_err(crate::Error::Io)?;

        SparseBlock::from_owned_bytes(data).map_err(|e| {
            crate::Error::Corruption(format!(
                "dim_id={} block_idx={} offset={} length={} base={}: {e}",
                self.dims.dim_ids[dim_idx], block_idx, entry.offset, entry.length, base
            ))
        })
    }

    /// Get posting list for a dimension (loads all blocks via mmap)
    pub async fn get_posting(
        &self,
        dim_id: u32,
    ) -> crate::Result<Option<Arc<BlockSparsePostingList>>> {
        let idx = match self.dims.find(dim_id) {
            Some(i) => i,
            None => return Ok(None),
        };

        let count = self.dims.skip_counts[idx] as usize;
        let mut blocks = Vec::with_capacity(count);
        for i in 0..count {
            blocks.push(self.load_block_at(idx, i).await?);
        }

        Ok(Some(Arc::new(BlockSparsePostingList {
            doc_count: self.dims.doc_counts[idx],
            blocks,
        })))
    }

    /// Get skip list for a dimension (for block-max iteration without loading blocks).
    /// Returns owned Vec since entries are parsed from zero-copy mmap bytes.
    pub fn get_skip_list(&self, dim_id: u32) -> Option<(Vec<SparseSkipEntry>, f32)> {
        let idx = self.dims.find(dim_id)?;
        Some((self.dim_skip_entries_vec(idx), self.dims.max_weights[idx]))
    }

    /// Get skip range for a dimension — zero-alloc alternative to `get_skip_list`.
    ///
    /// Returns `(skip_start, skip_count, max_weight)` where `skip_start` is the
    /// index into the skip section. Access individual entries via `read_skip_entry(skip_start + i)`.
    #[inline]
    pub fn get_skip_range(&self, dim_id: u32) -> Option<(usize, usize, f32)> {
        let idx = self.dims.find(dim_id)?;
        Some((
            self.dims.skip_starts[idx] as usize,
            self.dims.skip_counts[idx] as usize,
            self.dims.max_weights[idx],
        ))
    }

    /// Like `get_skip_range` but also returns `block_data_offset` for use with
    /// `load_block_direct`. Avoids needing a second dim_id lookup later.
    ///
    /// Returns `(skip_start, skip_count, max_weight, block_data_offset)`.
    #[inline]
    pub fn get_skip_range_full(&self, dim_id: u32) -> Option<(usize, usize, f32, u64)> {
        let idx = self.dims.find(dim_id)?;
        Some((
            self.dims.skip_starts[idx] as usize,
            self.dims.skip_counts[idx] as usize,
            self.dims.max_weights[idx],
            self.dims.block_offsets[idx],
        ))
    }

    /// Load a contiguous range of blocks for a dimension in a single mmap read.
    ///
    /// Returns individual `SparseBlock`s parsed from the coalesced byte range.
    /// This reduces mmap syscalls when processing superblocks (up to 8 blocks).
    pub async fn get_blocks_range(
        &self,
        dim_id: u32,
        block_start: usize,
        block_count: usize,
    ) -> crate::Result<Vec<SparseBlock>> {
        let idx = match self.dims.find(dim_id) {
            Some(i) => i,
            None => return Ok(Vec::new()),
        };
        let skip_start = self.dims.skip_starts[idx] as usize;
        let total_blocks = self.dims.skip_counts[idx] as usize;
        if block_start >= total_blocks || block_count == 0 {
            return Ok(Vec::new());
        }
        let end = (block_start + block_count).min(total_blocks);
        let base = self.dims.block_offsets[idx];

        // Compute the byte range covering all blocks [block_start..end)
        let first_entry = self.read_skip_entry(skip_start + block_start);
        let last_entry = self.read_skip_entry(skip_start + end - 1);
        let range_start = base + first_entry.offset;
        let range_end = base + last_entry.offset + last_entry.length as u64;

        // Single coalesced mmap read
        let range_data = self
            .handle
            .read_bytes_range(range_start..range_end)
            .await
            .map_err(crate::Error::Io)?;

        // Slice into individual blocks
        let mut blocks = Vec::with_capacity(end - block_start);
        for bi in block_start..end {
            let entry = self.read_skip_entry(skip_start + bi);
            let rel_offset = entry.offset - first_entry.offset;
            let block_bytes = range_data
                .slice(rel_offset as usize..(rel_offset as usize + entry.length as usize));
            blocks.push(SparseBlock::from_owned_bytes(block_bytes).map_err(|e| {
                crate::Error::Corruption(format!(
                    "dim_id={} block={}/{} skip_entry(offset={},length={}) base={}: {e}",
                    dim_id, bi, total_blocks, entry.offset, entry.length, base
                ))
            })?);
        }

        Ok(blocks)
    }

    /// Load specific block for a dimension
    pub async fn get_block(
        &self,
        dim_id: u32,
        block_idx: usize,
    ) -> crate::Result<Option<SparseBlock>> {
        let idx = match self.dims.find(dim_id) {
            Some(i) => i,
            None => return Ok(None),
        };
        if block_idx >= self.dims.skip_counts[idx] as usize {
            return Ok(None);
        }
        Ok(Some(self.load_block_at(idx, block_idx).await?))
    }

    /// Load a block using pre-resolved skip_start and block_data_offset.
    ///
    /// Avoids the dim_id binary search in `get_block` — intended for cursors
    /// that resolved the dimension index once at construction time.
    pub async fn load_block_direct(
        &self,
        skip_start: usize,
        block_data_offset: u64,
        block_idx: usize,
    ) -> crate::Result<Option<SparseBlock>> {
        if skip_start + block_idx >= self.skip_entry_count() {
            return Ok(None);
        }
        let entry = self.read_skip_entry(skip_start + block_idx);
        let base = block_data_offset;
        let abs_offset = base + entry.offset;
        let data = self
            .handle
            .read_bytes_range(abs_offset..abs_offset + entry.length as u64)
            .await
            .map_err(crate::Error::Io)?;
        Ok(Some(SparseBlock::from_owned_bytes(data).map_err(|e| {
            crate::Error::Corruption(format!(
                "direct block load skip_start={} block_idx={} offset={} length={} base={}: {e}",
                skip_start, block_idx, entry.offset, entry.length, base
            ))
        })?))
    }

    /// Check if dimension exists
    #[inline]
    pub fn has_dimension(&self, dim_id: u32) -> bool {
        self.dims.find(dim_id).is_some()
    }

    /// Get the number of dimensions in the index
    #[inline]
    pub fn num_dimensions(&self) -> usize {
        self.dims.len()
    }

    /// Iterate over all active dimension IDs
    pub fn active_dimensions(&self) -> impl Iterator<Item = u32> + '_ {
        self.dims.dim_ids.iter().copied()
    }

    /// Get doc count for dimension (from SoA table, no I/O)
    pub fn doc_count(&self, dim_id: u32) -> u32 {
        self.dims
            .find(dim_id)
            .map(|i| self.dims.doc_counts[i])
            .unwrap_or(0)
    }

    /// Compute IDF using doc_count from SoA table
    #[inline]
    pub fn idf(&self, dim_id: u32) -> f32 {
        let df = self.doc_count(dim_id) as f32;
        if df > 0.0 {
            let n = self.total_vectors.max(self.total_docs) as f32;
            (n / df).ln().max(0.0)
        } else {
            0.0
        }
    }

    /// Get IDF weights for multiple dimensions
    pub fn idf_weights(&self, dim_ids: &[u32]) -> Vec<f32> {
        dim_ids.iter().map(|&d| self.idf(d)).collect()
    }

    /// Estimated memory usage in bytes
    ///
    /// - DimensionTable: SoA arrays (6 × ndims × 4 bytes)
    /// - Skip section: OwnedBytes (Arc overhead only; data is mmap-backed)
    /// - Handle: LazyFileHandle overhead
    pub fn estimated_memory_bytes(&self) -> usize {
        let dim_table = self.dims.estimated_memory_bytes();
        // OwnedBytes: just Arc + range (no heap copy for mmap)
        let skip_overhead = std::mem::size_of::<OwnedBytes>() + self.skip_bytes.len();
        dim_table + skip_overhead
    }

    /// Whether the handle supports synchronous reads (mmap/RAM-backed).
    #[inline]
    pub fn is_sync(&self) -> bool {
        self.handle.is_sync()
    }

    /// Synchronous block load using pre-resolved skip_start and block_data_offset.
    ///
    /// Only works when the handle is Inline (mmap/RAM). Panics or errors on Lazy handles.
    /// This bypasses all async overhead for mmap-backed sparse indexes.
    #[inline]
    pub fn load_block_direct_sync(
        &self,
        skip_start: usize,
        block_data_offset: u64,
        block_idx: usize,
    ) -> crate::Result<Option<SparseBlock>> {
        if skip_start + block_idx >= self.skip_entry_count() {
            return Ok(None);
        }
        let entry = self.read_skip_entry(skip_start + block_idx);
        let base = block_data_offset;
        let abs_offset = base + entry.offset;
        let data = self
            .handle
            .read_bytes_range_sync(abs_offset..abs_offset + entry.length as u64)
            .map_err(crate::Error::Io)?;
        Ok(Some(SparseBlock::from_owned_bytes(data).map_err(|e| {
            crate::Error::Corruption(format!(
                "sync direct block load skip_start={} block_idx={} offset={} length={} base={}: {e}",
                skip_start, block_idx, entry.offset, entry.length, base
            ))
        })?))
    }

    /// Synchronous contiguous range block load.
    ///
    /// Only works when the handle is Inline (mmap/RAM). Single zero-copy slice
    /// into the mmap, then split into individual blocks.
    pub fn get_blocks_range_sync(
        &self,
        dim_id: u32,
        block_start: usize,
        block_count: usize,
    ) -> crate::Result<Vec<SparseBlock>> {
        let idx = match self.dims.find(dim_id) {
            Some(i) => i,
            None => return Ok(Vec::new()),
        };
        let skip_start = self.dims.skip_starts[idx] as usize;
        let total_blocks = self.dims.skip_counts[idx] as usize;
        if block_start >= total_blocks || block_count == 0 {
            return Ok(Vec::new());
        }
        let end = (block_start + block_count).min(total_blocks);
        let base = self.dims.block_offsets[idx];

        let first_entry = self.read_skip_entry(skip_start + block_start);
        let last_entry = self.read_skip_entry(skip_start + end - 1);
        let range_start = base + first_entry.offset;
        let range_end = base + last_entry.offset + last_entry.length as u64;

        let range_data = self
            .handle
            .read_bytes_range_sync(range_start..range_end)
            .map_err(crate::Error::Io)?;

        let mut blocks = Vec::with_capacity(end - block_start);
        for bi in block_start..end {
            let entry = self.read_skip_entry(skip_start + bi);
            let rel_offset = entry.offset - first_entry.offset;
            let block_bytes = range_data
                .slice(rel_offset as usize..(rel_offset as usize + entry.length as usize));
            blocks.push(SparseBlock::from_owned_bytes(block_bytes).map_err(|e| {
                crate::Error::Corruption(format!(
                    "sync dim_id={} block={}/{} skip_entry(offset={},length={}) base={}: {e}",
                    dim_id, bi, total_blocks, entry.offset, entry.length, base
                ))
            })?);
        }

        Ok(blocks)
    }

    /// Get merge-level info for a dimension: skip entries, doc_count, global_max_weight,
    /// and raw block data bytes (single mmap read, zero deserialization).
    pub async fn read_dim_raw(&self, dim_id: u32) -> crate::Result<Option<DimRawData>> {
        let idx = match self.dims.find(dim_id) {
            Some(i) => i,
            None => return Ok(None),
        };
        let skip = self.dim_skip_entries_vec(idx);
        if skip.is_empty() {
            return Ok(None);
        }
        // Total block data size: last entry's (offset + length)
        let last = &skip[skip.len() - 1];
        let total_bytes = last.offset + last.length as u64;
        let base = self.dims.block_offsets[idx];
        let raw = self
            .handle
            .read_bytes_range(base..base + total_bytes)
            .await
            .map_err(crate::Error::Io)?;
        Ok(Some(DimRawData {
            skip_entries: skip,
            doc_count: self.dims.doc_counts[idx],
            global_max_weight: self.dims.max_weights[idx],
            raw_block_data: raw,
        }))
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
