//! Index loading functions for segment reader

use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};
use rustc_hash::FxHashMap;
use std::io::Cursor;

use crate::Result;
use crate::directories::{AsyncFileRead, Directory, LazyFileHandle};
use crate::dsl::Schema;
use crate::structures::{CoarseCentroids, RaBitQIndex};

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
            4 => {
                // Flat binary format (compact, no JSON overhead)
                if let Ok(flat_data) = FlatVectorData::from_binary_bytes(data.as_slice()) {
                    indexes.insert(field_id, VectorIndex::Flat(Arc::new(flat_data)));
                }
            }
            3 => {
                // Flat (brute-force) - JSON format (legacy)
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

/// Load sparse vector indexes from .sparse file (lazy loading)
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
///
/// Memory optimization: Only loads the offset table, not the posting lists.
/// Posting lists are loaded on-demand during queries via mmap range reads.
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

    // Try to open sparse file lazily (may not exist if no sparse vectors were indexed)
    let handle = match dir.open_lazy(&files.sparse).await {
        Ok(h) => h,
        Err(e) => {
            log::debug!("No sparse file found ({}): {:?}", files.sparse.display(), e);
            return Ok(indexes);
        }
    };

    let file_size = handle.len();
    if file_size < 4 {
        return Ok(indexes);
    }

    // Read ENTIRE sparse file in one I/O call - much faster than multiple small reads
    let all_bytes = match handle.read_bytes_range(0..file_size).await {
        Ok(d) => d,
        Err(_) => return Ok(indexes),
    };
    let data = all_bytes.as_slice();

    let num_fields = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);

    log::debug!(
        "Loading sparse file (lazy): size={} bytes, num_fields={}",
        file_size,
        num_fields
    );

    if num_fields == 0 {
        return Ok(indexes);
    }

    // Parse from memory - no more I/O calls
    let mut pos: usize = 4; // After num_fields

    for _ in 0..num_fields {
        // Read field header: field_id(4) + quant(1) + num_dims(4) = 9 bytes
        let field_id = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        let _quantization = data[pos + 4];
        let num_dims =
            u32::from_le_bytes([data[pos + 5], data[pos + 6], data[pos + 7], data[pos + 8]]);
        pos += 9;

        // Parse dimension entries with skip lists
        // Format per dimension:
        // - dim_id: u32
        // - data_offset: u64 (absolute offset to posting list data)
        // - posting_list header: doc_count(4) + global_max_weight(4) + num_blocks(4)
        // - skip_entries: [SparseSkipEntry] × num_blocks (20 bytes each)
        let mut dimensions: Vec<super::types::DimensionEntry> =
            Vec::with_capacity(num_dims as usize);

        for _ in 0..num_dims {
            let dim_id =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            let data_offset = u64::from_le_bytes([
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
                data[pos + 8],
                data[pos + 9],
                data[pos + 10],
                data[pos + 11],
            ]);
            let posting_length = u32::from_le_bytes([
                data[pos + 12],
                data[pos + 13],
                data[pos + 14],
                data[pos + 15],
            ]);
            pos += 16;

            // Read posting list header from data_offset to get skip list
            let pl_data =
                &data[data_offset as usize..(data_offset + posting_length as u64) as usize];
            let doc_count = u32::from_le_bytes([pl_data[0], pl_data[1], pl_data[2], pl_data[3]]);
            let global_max_weight =
                f32::from_le_bytes([pl_data[4], pl_data[5], pl_data[6], pl_data[7]]);
            let num_blocks =
                u32::from_le_bytes([pl_data[8], pl_data[9], pl_data[10], pl_data[11]]) as usize;

            // Parse skip entries (20 bytes each: first_doc + last_doc + offset + length + max_weight)
            let mut skip_entries = Vec::with_capacity(num_blocks);
            let mut skip_pos = 12; // After header
            for _ in 0..num_blocks {
                let first_doc = u32::from_le_bytes([
                    pl_data[skip_pos],
                    pl_data[skip_pos + 1],
                    pl_data[skip_pos + 2],
                    pl_data[skip_pos + 3],
                ]);
                let last_doc = u32::from_le_bytes([
                    pl_data[skip_pos + 4],
                    pl_data[skip_pos + 5],
                    pl_data[skip_pos + 6],
                    pl_data[skip_pos + 7],
                ]);
                let offset = u32::from_le_bytes([
                    pl_data[skip_pos + 8],
                    pl_data[skip_pos + 9],
                    pl_data[skip_pos + 10],
                    pl_data[skip_pos + 11],
                ]);
                let length = u32::from_le_bytes([
                    pl_data[skip_pos + 12],
                    pl_data[skip_pos + 13],
                    pl_data[skip_pos + 14],
                    pl_data[skip_pos + 15],
                ]);
                let max_weight = f32::from_le_bytes([
                    pl_data[skip_pos + 16],
                    pl_data[skip_pos + 17],
                    pl_data[skip_pos + 18],
                    pl_data[skip_pos + 19],
                ]);
                skip_entries.push(crate::structures::SparseSkipEntry::new(
                    first_doc, last_doc, offset, length, max_weight,
                ));
                skip_pos += 20;
            }

            // data_offset points to start of posting list, block data starts after header + skip list
            let header_size = 12 + num_blocks * 20;
            dimensions.push(super::types::DimensionEntry {
                dim_id,
                data_offset: data_offset + header_size as u64,
                doc_count,
                global_max_weight,
                skip_entries,
            });
        }
        // Ensure sorted by dim_id for binary search
        dimensions.sort_by_key(|d| d.dim_id);

        // total_vectors equals total_docs because doc_count per dimension
        // already counts unique documents (not ordinals). The max(total_vectors,
        // total_docs) in IDF computation is just a safety invariant.
        let total_vectors = total_docs;

        log::debug!(
            "Loaded sparse index for field {} (lazy): num_dims={}, total_docs={}",
            field_id,
            num_dims,
            total_docs
        );

        indexes.insert(
            field_id,
            SparseIndex::new(handle.clone(), dimensions, total_docs, total_vectors),
        );
    }

    log::debug!(
        "Sparse file loaded (lazy): fields={:?}",
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
