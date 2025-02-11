//! Index loading functions for segment reader

use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};
use rustc_hash::FxHashMap;
use std::io::Cursor;

use crate::Result;
use crate::directories::{AsyncFileRead, Directory, LazyFileHandle, LazyFileSlice};
use crate::dsl::Schema;
use crate::structures::RaBitQIndex;

use super::super::types::SegmentFiles;
use super::super::vector_data::LazyFlatVectorData;
use super::{SparseIndex, VectorIndex};

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
                let slice = LazyFileSlice::from_handle_range(&handle, offset, length);
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
                // RaBitQ (standalone)
                let data = handle.read_bytes_range(offset..offset + length).await?;
                if let Ok(rabitq_index) = serde_json::from_slice::<RaBitQIndex>(data.as_slice()) {
                    indexes.insert(field_id, VectorIndex::RaBitQ(Arc::new(rabitq_index)));
                }
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
    use crate::segment::format::{FOOTER_SIZE, SPARSE_FOOTER_MAGIC};

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
    if file_size < FOOTER_SIZE {
        return Ok(indexes);
    }

    // Read only footer (16 bytes) — not the entire file
    let footer_bytes = match handle
        .read_bytes_range(file_size - FOOTER_SIZE..file_size)
        .await
    {
        Ok(d) => d,
        Err(_) => return Ok(indexes),
    };
    let fb = footer_bytes.as_slice();

    // Parse footer: toc_offset(8) + num_fields(4) + magic(4)
    let toc_offset = u64::from_le_bytes(fb[0..8].try_into().unwrap());
    let num_fields = u32::from_le_bytes(fb[8..12].try_into().unwrap());
    let magic = u32::from_le_bytes(fb[12..16].try_into().unwrap());

    if magic != SPARSE_FOOTER_MAGIC {
        return Err(crate::Error::Corruption(format!(
            "Invalid sparse footer magic: {:#x} (expected {:#x})",
            magic, SPARSE_FOOTER_MAGIC
        )));
    }

    log::debug!(
        "Loading sparse file (lazy): size={} bytes, num_fields={}, toc_offset={}",
        file_size,
        num_fields,
        toc_offset,
    );

    if num_fields == 0 {
        return Ok(indexes);
    }

    // Read only the TOC section (toc_offset .. footer_start) — skip all posting block data
    let toc_end = file_size - FOOTER_SIZE;
    let toc_bytes = handle.read_bytes_range(toc_offset..toc_end).await?;
    let toc_data = toc_bytes.as_slice();
    let mut pos = 0usize;

    // First pass: parse TOC to collect dimension metadata (dim_id, data_offset, posting_length)
    // without reading any posting data yet.
    struct DimTocEntry {
        dim_id: u32,
        data_offset: u64,
        posting_length: u32,
    }
    struct FieldToc {
        field_id: u32,
        dims: Vec<DimTocEntry>,
    }
    let mut field_tocs = Vec::with_capacity(num_fields as usize);
    for _ in 0..num_fields {
        // Field header: field_id(4) + quant(1) + num_dims(4) = 9 bytes
        let field_id = u32::from_le_bytes([
            toc_data[pos],
            toc_data[pos + 1],
            toc_data[pos + 2],
            toc_data[pos + 3],
        ]);
        let _quantization = toc_data[pos + 4];
        let ndims = u32::from_le_bytes([
            toc_data[pos + 5],
            toc_data[pos + 6],
            toc_data[pos + 7],
            toc_data[pos + 8],
        ]);
        pos += 9;

        let mut dims = Vec::with_capacity(ndims as usize);
        for _ in 0..ndims {
            let dim_id = u32::from_le_bytes([
                toc_data[pos],
                toc_data[pos + 1],
                toc_data[pos + 2],
                toc_data[pos + 3],
            ]);
            let data_offset = u64::from_le_bytes([
                toc_data[pos + 4],
                toc_data[pos + 5],
                toc_data[pos + 6],
                toc_data[pos + 7],
                toc_data[pos + 8],
                toc_data[pos + 9],
                toc_data[pos + 10],
                toc_data[pos + 11],
            ]);
            let posting_length = u32::from_le_bytes([
                toc_data[pos + 12],
                toc_data[pos + 13],
                toc_data[pos + 14],
                toc_data[pos + 15],
            ]);
            pos += 16;
            dims.push(DimTocEntry {
                dim_id,
                data_offset,
                posting_length,
            });
        }
        field_tocs.push(FieldToc { field_id, dims });
    }

    // Second pass: for each field, compute the contiguous header+skip region and
    // read it in a single I/O call (avoids per-dimension reads).
    for ftoc in &field_tocs {
        if ftoc.dims.is_empty() {
            continue;
        }

        // Posting data for dimensions within a field is written sequentially, so
        // the headers+skip entries form a prefix of each posting, laid out one
        // after another. We read the entire posting region and parse only the
        // header+skip portion of each dimension, ignoring block data bytes.
        let region_start = ftoc.dims.first().unwrap().data_offset;
        let last = ftoc.dims.last().unwrap();
        let region_end = last.data_offset + last.posting_length as u64;
        let region_bytes = handle.read_bytes_range(region_start..region_end).await?;
        let region = region_bytes.as_slice();

        let mut dimensions: Vec<super::types::DimensionEntry> = Vec::with_capacity(ftoc.dims.len());
        let mut all_skip_entries: Vec<crate::structures::SparseSkipEntry> = Vec::new();

        for dim_entry in &ftoc.dims {
            // Offset within region
            let local = (dim_entry.data_offset - region_start) as usize;
            let pl_data = &region[local..local + dim_entry.posting_length as usize];

            // Parse posting header: doc_count(4) + global_max_weight(4) + num_blocks(4)
            let doc_count = u32::from_le_bytes([pl_data[0], pl_data[1], pl_data[2], pl_data[3]]);
            let global_max_weight =
                f32::from_le_bytes([pl_data[4], pl_data[5], pl_data[6], pl_data[7]]);
            let num_blocks =
                u32::from_le_bytes([pl_data[8], pl_data[9], pl_data[10], pl_data[11]]) as usize;

            // Parse skip entries (20 bytes each)
            let skip_start = all_skip_entries.len() as u32;
            let mut skip_pos = 12;
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
                all_skip_entries.push(crate::structures::SparseSkipEntry::new(
                    first_doc, last_doc, offset, length, max_weight,
                ));
                skip_pos += 20;
            }

            // data_offset points to posting start; block data starts after header + skip list
            let header_size = 12 + num_blocks * 20;
            dimensions.push(super::types::DimensionEntry {
                dim_id: dim_entry.dim_id,
                data_offset: dim_entry.data_offset + header_size as u64,
                doc_count,
                global_max_weight,
                skip_start,
                skip_count: num_blocks as u32,
            });
        }
        // Ensure sorted by dim_id for binary search
        dimensions.sort_by_key(|d| d.dim_id);

        // total_vectors equals total_docs because doc_count per dimension
        // already counts unique documents (not ordinals). The max(total_vectors,
        // total_docs) in IDF computation is just a safety invariant.
        let total_vectors = total_docs;

        log::debug!(
            "Loaded sparse index for field {} (lazy): num_dims={}, total_docs={}, skip_entries={}",
            ftoc.field_id,
            ftoc.dims.len(),
            total_docs,
            all_skip_entries.len()
        );

        indexes.insert(
            ftoc.field_id,
            SparseIndex::new(
                handle.clone(),
                dimensions,
                all_skip_entries,
                total_docs,
                total_vectors,
            ),
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
