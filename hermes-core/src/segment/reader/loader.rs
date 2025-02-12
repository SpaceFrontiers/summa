//! Index loading functions for segment reader

use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};
use rustc_hash::FxHashMap;
use std::io::Cursor;

use super::super::types::SegmentFiles;
use super::super::vector_data::LazyFlatVectorData;
use super::{SparseIndex, VectorIndex};
use crate::Result;
use crate::directories::{Directory, FileHandle};
use crate::dsl::Schema;

/// Vectors file loading result
pub struct VectorsFileData {
    /// ANN indexes per field (IVF, ScaNN, RaBitQ) — loaded into memory for search
    pub indexes: FxHashMap<u32, VectorIndex>,
    /// Lazy flat vectors per field — doc_ids in memory, vectors via mmap for reranking/merge
    pub flat_vectors: FxHashMap<u32, LazyFlatVectorData>,
}

use crate::segment::format::{
    DENSE_TOC_ENTRY_SIZE, DenseVectorTocEntry, FOOTER_SIZE, VECTORS_FOOTER_MAGIC, read_dense_toc,
};

/// Load dense vector indexes from unified .vectors file
///
/// File format (data-first, TOC at end):
/// - [field data...]  — starts at offset 0 (mmap page-aligned)
/// - [TOC entries]    — field_id(4) + index_type(1) + offset(8) + size(8) per field
/// - [footer 16B]     — toc_offset(8) + num_fields(4) + magic(4)
///
/// Also supports legacy header-first format (no magic) for backwards compatibility.
pub async fn load_vectors_file<D: Directory>(
    dir: &D,
    files: &SegmentFiles,
    schema: &Schema,
) -> Result<VectorsFileData> {
    let mut indexes = FxHashMap::default();
    let mut flat_vectors = FxHashMap::default();
    let empty = || VectorsFileData {
        indexes: FxHashMap::default(),
        flat_vectors: FxHashMap::default(),
    };

    // Skip loading vectors file if schema has no dense vector fields
    let has_dense_vectors = schema
        .fields()
        .any(|(_, entry)| entry.dense_vector_config.is_some());
    if !has_dense_vectors {
        return Ok(empty());
    }

    // Try to open vectors file (may not exist if no vectors were indexed)
    let handle = match dir.open_lazy(&files.vectors).await {
        Ok(h) => h,
        Err(_) => return Ok(empty()),
    };

    let file_size = handle.len();
    if file_size < FOOTER_SIZE {
        return Ok(empty());
    }

    // Try new format: read footer (last 16 bytes)
    let footer_bytes = handle
        .read_bytes_range(file_size - FOOTER_SIZE..file_size)
        .await?;
    let mut cursor = Cursor::new(footer_bytes.as_slice());
    let toc_offset = cursor.read_u64::<LittleEndian>()?;
    let num_fields = cursor.read_u32::<LittleEndian>()?;
    let magic = cursor.read_u32::<LittleEndian>()?;

    let entries: Vec<DenseVectorTocEntry> =
        if magic == VECTORS_FOOTER_MAGIC && toc_offset < file_size - FOOTER_SIZE {
            // New format: TOC at end
            let toc_size = num_fields as u64 * DENSE_TOC_ENTRY_SIZE;
            let toc_bytes = handle
                .read_bytes_range(toc_offset..toc_offset + toc_size)
                .await?;
            read_dense_toc(toc_bytes.as_slice(), num_fields)?
        } else {
            // Legacy format: header at start (num_fields(4) + entries)
            let header_bytes = handle.read_bytes_range(0..4).await?;
            let mut cursor = Cursor::new(header_bytes.as_slice());
            let num_fields = cursor.read_u32::<LittleEndian>()?;
            if num_fields == 0 {
                return Ok(empty());
            }
            let entries_size = num_fields as u64 * DENSE_TOC_ENTRY_SIZE;
            let entries_bytes = handle.read_bytes_range(4..4 + entries_size).await?;
            read_dense_toc(entries_bytes.as_slice(), num_fields)?
        };

    if entries.is_empty() {
        return Ok(empty());
    }

    // Load each entry — a field can have both Flat (lazy) and ANN (in-memory)
    use crate::segment::ann_build;
    for DenseVectorTocEntry {
        field_id,
        index_type,
        offset,
        size: length,
    } in entries
    {
        match index_type {
            ann_build::FLAT_TYPE => {
                // Flat binary — load lazily (only doc_ids in memory, vectors via mmap)
                let slice = handle.slice(offset..offset + length);
                match LazyFlatVectorData::open(slice).await {
                    Ok(lazy_flat) => {
                        flat_vectors.insert(field_id, lazy_flat);
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to load lazy flat vectors for field {}: {}",
                            field_id,
                            e
                        );
                    }
                }
            }
            ann_build::SCANN_TYPE => {
                // ScaNN (IVF-PQ) — lazy: OwnedBytes stored (zero-copy mmap ref),
                // deserialized on first search. No heap copy during segment load.
                let data = handle.read_bytes_range(offset..offset + length).await?;
                indexes.insert(
                    field_id,
                    VectorIndex::ScaNN(Arc::new(super::types::LazyScaNN::new(data))),
                );
            }
            ann_build::IVF_RABITQ_TYPE => {
                // IVF-RaBitQ — lazy: OwnedBytes stored (zero-copy mmap ref),
                // deserialized on first search. No heap copy during segment load.
                let data = handle.read_bytes_range(offset..offset + length).await?;
                indexes.insert(
                    field_id,
                    VectorIndex::IVF(Arc::new(super::types::LazyIVF::new(data))),
                );
            }
            ann_build::RABITQ_TYPE => {
                // RaBitQ — lazy: OwnedBytes stored (zero-copy mmap ref),
                // deserialized on first search. No heap copy during segment load.
                let data = handle.read_bytes_range(offset..offset + length).await?;
                indexes.insert(
                    field_id,
                    VectorIndex::RaBitQ(Arc::new(super::types::LazyRaBitQ::new(data))),
                );
            }
            _ => {
                log::warn!(
                    "Unknown vector index type {} for field {}",
                    index_type,
                    field_id
                );
            }
        }
    }

    Ok(VectorsFileData {
        indexes,
        flat_vectors,
    })
}

/// Load sparse vector indexes from .sparse file (lazy loading)
///
/// Footer-based format (data-first):
/// ```text
/// [posting data for all dims across all fields]
/// [TOC: per-field header + per-dim entries]
/// [footer: toc_offset(u64) + num_fields(u32) + magic(u32)]
/// ```
///
/// Memory optimization: Only loads the offset table + skip lists, not the block data.
/// Block data is loaded on-demand during queries via mmap range reads.
pub async fn load_sparse_file<D: Directory>(
    dir: &D,
    files: &SegmentFiles,
    total_docs: u32,
    schema: &Schema,
) -> Result<FxHashMap<u32, SparseIndex>> {
    use crate::segment::format::{SPARSE_FOOTER_MAGIC, SPARSE_FOOTER_SIZE};

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
    if file_size < SPARSE_FOOTER_SIZE {
        return Ok(indexes);
    }

    // Read V3 footer (24 bytes): skip_offset(8) + toc_offset(8) + num_fields(4) + magic(4)
    let footer_bytes = match handle
        .read_bytes_range(file_size - SPARSE_FOOTER_SIZE..file_size)
        .await
    {
        Ok(d) => d,
        Err(_) => return Ok(indexes),
    };
    let fb = footer_bytes.as_slice();

    let skip_offset = u64::from_le_bytes(fb[0..8].try_into().unwrap());
    let toc_offset = u64::from_le_bytes(fb[8..16].try_into().unwrap());
    let num_fields = u32::from_le_bytes(fb[16..20].try_into().unwrap());
    let magic = u32::from_le_bytes(fb[20..24].try_into().unwrap());

    if magic != SPARSE_FOOTER_MAGIC {
        return Err(crate::Error::Corruption(format!(
            "Invalid sparse footer magic: {:#x} (expected {:#x})",
            magic, SPARSE_FOOTER_MAGIC
        )));
    }

    log::debug!(
        "Loading sparse V3: size={} bytes, num_fields={}, skip_offset={}, toc_offset={}",
        file_size,
        num_fields,
        skip_offset,
        toc_offset,
    );

    if num_fields == 0 {
        return Ok(indexes);
    }

    // Single tail read: skip section + TOC (skip_offset .. footer_start)
    // For mmap this is zero-copy. Block data at the front is never touched.
    let tail_bytes = handle
        .read_bytes_range(skip_offset..file_size - SPARSE_FOOTER_SIZE)
        .await?;
    let tail = tail_bytes.as_slice();

    // skip section is at tail[0 .. toc_offset - skip_offset]
    let skip_section_len = (toc_offset - skip_offset) as usize;
    let skip_section = tail_bytes.slice(0..skip_section_len);
    let toc_data = &tail[skip_section_len..];

    // Parse TOC: per-field header(9B) + per-dim entries(28B each)
    let mut pos = 0usize;

    for _ in 0..num_fields {
        // Field header: field_id(4) + quant(1) + num_dims(4) = 9 bytes
        let field_id = u32::from_le_bytes(toc_data[pos..pos + 4].try_into().unwrap());
        let _quantization = toc_data[pos + 4];
        let ndims = u32::from_le_bytes(toc_data[pos + 5..pos + 9].try_into().unwrap()) as usize;
        pos += 9;

        // Parse V3 per-dim entries directly into SoA DimensionTable
        let mut dims = super::types::DimensionTable::with_capacity(ndims);
        for _ in 0..ndims {
            let d = &toc_data[pos..pos + 28];
            let dim_id = u32::from_le_bytes(d[0..4].try_into().unwrap());
            let block_data_offset = u64::from_le_bytes(d[4..12].try_into().unwrap());
            let skip_start = u32::from_le_bytes(d[12..16].try_into().unwrap());
            let num_blocks = u32::from_le_bytes(d[16..20].try_into().unwrap());
            let doc_count = u32::from_le_bytes(d[20..24].try_into().unwrap());
            let max_weight = f32::from_le_bytes(d[24..28].try_into().unwrap());
            dims.push(
                dim_id,
                block_data_offset,
                skip_start,
                num_blocks,
                doc_count,
                max_weight,
            );
            pos += 28;
        }
        // Ensure sorted by dim_id for binary search
        dims.sort_by_dim_id();

        // total_vectors equals total_docs (doc_count per dimension counts unique docs)
        let total_vectors = total_docs;

        log::debug!(
            "Loaded sparse V3 index for field {}: num_dims={}, total_docs={}, skip_bytes={}",
            field_id,
            dims.len(),
            total_docs,
            skip_section.len(),
        );

        indexes.insert(
            field_id,
            SparseIndex::new(
                handle.clone(),
                dims,
                skip_section.clone(),
                total_docs,
                total_vectors,
            ),
        );
    }

    log::debug!(
        "Sparse V3 file loaded: fields={:?}",
        indexes.keys().collect::<Vec<_>>()
    );

    Ok(indexes)
}

/// Open positions file handle (no header parsing needed - offsets are in TermInfo)
pub async fn open_positions_file<D: Directory>(
    dir: &D,
    files: &SegmentFiles,
    schema: &Schema,
) -> Result<Option<FileHandle>> {
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
