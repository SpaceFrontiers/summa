//! Index loading functions for segment reader

use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};
use rustc_hash::FxHashMap;
use std::io::Cursor;

use crate::Result;
use crate::directories::{AsyncFileRead, Directory, LazyFileHandle};
use crate::dsl::Schema;
use crate::structures::{BlockSparsePostingList, CoarseCentroids, RaBitQIndex};

use super::super::types::SegmentFiles;
use super::super::vector_data::{FlatVectorData, IVFRaBitQIndexData, ScaNNIndexData};
use super::{SparseIndex, VectorIndex};

/// Load dense vector indexes from unified .vectors file
///
/// Supports RaBitQ (type 0), IVF-RaBitQ (type 1), and ScaNN (type 2).
/// Also loads coarse centroids and PQ codebook as needed.
///
/// Memory optimization: Uses lazy range reads to load each index separately,
/// avoiding loading the entire vectors file into memory at once.
pub async fn load_vectors_file<D: Directory>(
    dir: &D,
    files: &SegmentFiles,
    schema: &Schema,
) -> Result<(FxHashMap<u32, VectorIndex>, Option<Arc<CoarseCentroids>>)> {
    let mut indexes = FxHashMap::default();
    let mut coarse_centroids: Option<Arc<CoarseCentroids>> = None;

    // Skip loading vectors file if schema has no dense vector fields
    let has_dense_vectors = schema
        .fields()
        .any(|(_, entry)| entry.dense_vector_config.is_some());
    if !has_dense_vectors {
        return Ok((indexes, None));
    }

    // Try to open vectors file (may not exist if no vectors were indexed)
    let handle = match dir.open_lazy(&files.vectors).await {
        Ok(h) => h,
        Err(_) => return Ok((indexes, None)),
    };

    // Read only the header first (4 bytes for num_fields)
    let header_bytes = match handle.read_bytes_range(0..4).await {
        Ok(b) => b,
        Err(_) => return Ok((indexes, None)),
    };

    if header_bytes.is_empty() {
        return Ok((indexes, None));
    }

    let mut cursor = Cursor::new(header_bytes.as_slice());
    let num_fields = cursor.read_u32::<LittleEndian>()?;

    if num_fields == 0 {
        return Ok((indexes, None));
    }

    // Read field entries header: (field_id: 4, index_type: 1, offset: 8, length: 8) = 21 bytes per field
    let entries_size = num_fields as u64 * 21;
    let entries_bytes = handle.read_bytes_range(4..4 + entries_size).await?;
    let mut cursor = Cursor::new(entries_bytes.as_slice());

    // Read field entries (field_id, index_type, offset, length)
    let mut entries = Vec::with_capacity(num_fields as usize);
    for _ in 0..num_fields {
        let field_id = cursor.read_u32::<LittleEndian>()?;
        // Try to read index_type - if this fails, assume old format without type
        let index_type = cursor.read_u8().unwrap_or(255); // 255 = unknown/legacy
        let offset = cursor.read_u64::<LittleEndian>()?;
        let length = cursor.read_u64::<LittleEndian>()?;
        entries.push((field_id, index_type, offset, length));
    }

    // Load each index on-demand using range reads (memory efficient)
    for (field_id, index_type, offset, length) in entries {
        // Read only this index's data
        let data = handle.read_bytes_range(offset..offset + length).await?;
        let _field = crate::dsl::Field(field_id);

        match index_type {
            3 => {
                // Flat (brute-force) - raw vectors for accumulating state
                if let Ok(flat_data) = serde_json::from_slice::<FlatVectorData>(data.as_slice()) {
                    indexes.insert(field_id, VectorIndex::Flat(Arc::new(flat_data)));
                }
            }
            2 => {
                // ScaNN (IVF-PQ) with embedded centroids and codebook
                if let Ok(scann_data) = ScaNNIndexData::from_bytes(data.as_slice()) {
                    coarse_centroids = Some(Arc::new(scann_data.centroids));
                    indexes.insert(
                        field_id,
                        VectorIndex::ScaNN {
                            index: Arc::new(scann_data.index),
                            codebook: Arc::new(scann_data.codebook),
                        },
                    );
                }
            }
            1 => {
                // IVF-RaBitQ with embedded centroids and codebook
                if let Ok(ivf_data) = IVFRaBitQIndexData::from_bytes(data.as_slice()) {
                    coarse_centroids = Some(Arc::new(ivf_data.centroids));
                    indexes.insert(
                        field_id,
                        VectorIndex::IVF {
                            index: Arc::new(ivf_data.index),
                            codebook: Arc::new(ivf_data.codebook),
                        },
                    );
                }
            }
            0 => {
                // RaBitQ (standalone)
                if let Ok(rabitq_index) = serde_json::from_slice::<RaBitQIndex>(data.as_slice()) {
                    indexes.insert(field_id, VectorIndex::RaBitQ(Arc::new(rabitq_index)));
                }
            }
            _ => {
                // Unknown type - try Flat first (most common in new indexes)
                if let Ok(flat_data) = serde_json::from_slice::<FlatVectorData>(data.as_slice()) {
                    indexes.insert(field_id, VectorIndex::Flat(Arc::new(flat_data)));
                } else if let Ok(rabitq_index) =
                    serde_json::from_slice::<RaBitQIndex>(data.as_slice())
                {
                    indexes.insert(field_id, VectorIndex::RaBitQ(Arc::new(rabitq_index)));
                }
            }
        }
    }

    Ok((indexes, coarse_centroids))
}

/// Load sparse vector indexes from .sparse file
///
/// File format (direct-indexed table for O(1) dimension lookup):
/// - Header: num_fields (u32)
/// - For each field:
///   - field_id (u32)
///   - quantization (u8)
///   - max_dim_id (u32)          ← table size
///   - table: [(offset: u64, length: u32)] × max_dim_id  ← direct indexed
///     (offset=0, length=0 means dimension not present)
/// - Data: concatenated serialized BlockSparsePostingList
pub async fn load_sparse_file<D: Directory>(
    dir: &D,
    files: &SegmentFiles,
    total_docs: u32,
    schema: &Schema,
) -> Result<FxHashMap<u32, SparseIndex>> {
    let mut indexes = FxHashMap::default();

    // Skip loading sparse file if schema has no sparse vector fields
    let has_sparse_vectors = schema
        .fields()
        .any(|(_, entry)| entry.sparse_vector_config.is_some());
    if !has_sparse_vectors {
        return Ok(indexes);
    }

    // Try to open sparse file (may not exist if no sparse vectors were indexed)
    let handle = match dir.open_lazy(&files.sparse).await {
        Ok(h) => h,
        Err(e) => {
            log::debug!("No sparse file found ({}): {:?}", files.sparse.display(), e);
            return Ok(indexes);
        }
    };

    // Read the entire file (sparse files are typically small enough)
    let data = match handle.read_bytes().await {
        Ok(d) => d,
        Err(_) => return Ok(indexes),
    };

    if data.len() < 4 {
        return Ok(indexes);
    }

    let mut cursor = Cursor::new(data.as_slice());
    let num_fields = cursor.read_u32::<LittleEndian>()?;

    log::debug!(
        "Loading sparse file: size={} bytes, num_fields={}",
        data.len(),
        num_fields
    );

    if num_fields == 0 {
        return Ok(indexes);
    }

    // Read field entries and build indexes
    for _ in 0..num_fields {
        let field_id = cursor.read_u32::<LittleEndian>()?;
        let _quantization = cursor.read_u8()?; // Already stored in each BlockSparsePostingList
        let max_dim_id = cursor.read_u32::<LittleEndian>()?;

        // Read direct-indexed table
        let mut postings: Vec<Option<Arc<BlockSparsePostingList>>> =
            vec![None; max_dim_id as usize];

        for dim_id in 0..max_dim_id {
            let offset = cursor.read_u64::<LittleEndian>()?;
            let length = cursor.read_u32::<LittleEndian>()?;

            // offset=0, length=0 means dimension not present
            if length > 0 {
                let start = offset as usize;
                let end = start + length as usize;
                if end <= data.len() {
                    let posting_data = &data.as_slice()[start..end];
                    if let Ok(posting_list) =
                        BlockSparsePostingList::deserialize(&mut Cursor::new(posting_data))
                    {
                        postings[dim_id as usize] = Some(Arc::new(posting_list));
                    }
                }
            }
        }

        let num_postings = postings.iter().filter(|p| p.is_some()).count();
        log::debug!(
            "Loaded sparse index for field {}: max_dim={}, active_postings={}",
            field_id,
            max_dim_id,
            num_postings
        );

        indexes.insert(
            field_id,
            SparseIndex {
                postings,
                total_docs,
            },
        );
    }

    log::debug!(
        "Sparse file loaded: fields={:?}",
        indexes.keys().collect::<Vec<_>>()
    );

    Ok(indexes)
}

/// Open positions file handle (no header parsing needed - offsets are in TermInfo)
pub async fn open_positions_file<D: Directory>(
    dir: &D,
    files: &SegmentFiles,
    schema: &Schema,
) -> Result<Option<LazyFileHandle>> {
    // Skip loading positions file if schema has no fields with position tracking
    let has_positions = schema.fields().any(|(_, entry)| entry.positions.is_some());
    if !has_positions {
        return Ok(None);
    }

    // Try to open positions file (may not exist if no positions were indexed)
    match dir.open_lazy(&files.positions).await {
        Ok(h) => Ok(Some(h)),
        Err(_) => Ok(None),
    }
}
