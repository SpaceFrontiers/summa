//! Index loading functions for segment reader

use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};
use rustc_hash::{FxHashMap, FxHashSet};
use std::io::Cursor;

use super::super::types::SegmentFiles;
use super::super::vector_data::LazyFlatVectorData;
use super::bmp::BmpIndex;
use super::{SparseIndex, VectorIndex};
use crate::Result;
use crate::directories::{Directory, FileHandle};
use crate::dsl::{
    BinaryIndexType, DenseVectorQuantization, Field, FieldType, Schema, VectorIndexType,
};

/// Result of loading the `.sparse` file — may contain MaxScore and/or BMP indexes.
pub struct SparseFileData {
    pub maxscore_indexes: FxHashMap<u32, SparseIndex>,
    pub bmp_indexes: FxHashMap<u32, BmpIndex>,
}

/// Vectors file loading result
pub struct VectorsFileData {
    /// Segment-level IVF payloads per field, loaded lazily for search.
    pub indexes: FxHashMap<u32, VectorIndex>,
    /// Lazy flat vectors per field — doc_ids in memory, vectors via mmap for reranking/merge
    pub flat_vectors: FxHashMap<u32, LazyFlatVectorData>,
    /// ANN field IDs declared by the validated TOC. Training-only callers use
    /// this without opening corpus-sized ANN payloads.
    #[cfg(feature = "native")]
    pub ann_fields: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
fn validate_sparse_dimension_skip_entries(
    skip_section: &[u8],
    skip_start: usize,
    num_blocks: usize,
    block_data_offset: u64,
    data_end: u64,
    total_docs: u32,
    global_max_weight: f32,
    field_id: u32,
    dim_id: u32,
) -> Result<std::ops::Range<u64>> {
    if num_blocks == 0 {
        return Err(crate::Error::Corruption(format!(
            "sparse field {field_id} dimension {dim_id} has no blocks"
        )));
    }
    if !global_max_weight.is_finite() || global_max_weight < 0.0 {
        return Err(crate::Error::Corruption(format!(
            "sparse field {field_id} dimension {dim_id} has invalid global max weight {global_max_weight}"
        )));
    }

    let mut previous_end = 0u64;
    let mut previous_first_doc = 0u32;
    let mut previous_last_doc = 0u32;
    let mut range_start = None;
    let mut range_end = 0u64;
    for block_index in 0..num_blocks {
        let entry =
            crate::structures::SparseSkipEntry::read_at(skip_section, skip_start + block_index);
        if entry.length == 0 {
            return Err(crate::Error::Corruption(format!(
                "sparse field {field_id} dimension {dim_id} block {block_index} is empty"
            )));
        }
        if entry.first_doc > entry.last_doc || entry.last_doc >= total_docs {
            return Err(crate::Error::Corruption(format!(
                "sparse field {field_id} dimension {dim_id} block {block_index} has invalid document range {}..={} for {total_docs} documents",
                entry.first_doc, entry.last_doc
            )));
        }
        if !entry.max_weight.is_finite()
            || entry.max_weight < 0.0
            || entry.max_weight > global_max_weight
        {
            return Err(crate::Error::Corruption(format!(
                "sparse field {field_id} dimension {dim_id} block {block_index} has invalid max weight {} (global {global_max_weight})",
                entry.max_weight
            )));
        }

        let relative_end = entry
            .offset
            .checked_add(u64::from(entry.length))
            .ok_or_else(|| {
                crate::Error::Corruption(format!(
                    "sparse field {field_id} dimension {dim_id} block {block_index} range overflows u64"
                ))
            })?;
        let absolute_start = block_data_offset
            .checked_add(entry.offset)
            .ok_or_else(|| {
                crate::Error::Corruption(format!(
                    "sparse field {field_id} dimension {dim_id} block {block_index} file start overflows u64"
                ))
            })?;
        let absolute_end = block_data_offset
            .checked_add(relative_end)
            .ok_or_else(|| {
                crate::Error::Corruption(format!(
                    "sparse field {field_id} dimension {dim_id} block {block_index} file range overflows u64"
                ))
            })?;
        if absolute_end > data_end || entry.offset < previous_end {
            return Err(crate::Error::Corruption(format!(
                "sparse field {field_id} dimension {dim_id} block {block_index} has overlapping or out-of-bounds byte range"
            )));
        }
        if block_index > 0
            && (entry.first_doc < previous_first_doc || entry.last_doc < previous_last_doc)
        {
            return Err(crate::Error::Corruption(format!(
                "sparse field {field_id} dimension {dim_id} block {block_index} document ranges are not monotonic"
            )));
        }
        previous_end = relative_end;
        previous_first_doc = entry.first_doc;
        previous_last_doc = entry.last_doc;
        range_start.get_or_insert(absolute_start);
        range_end = absolute_end;
    }
    Ok(range_start.expect("num_blocks was validated above")..range_end)
}

fn validate_ann_schema(schema: &Schema, field_id: u32, index_type: u8) -> Result<()> {
    use crate::segment::ann_build;

    let field = schema.get_field_entry(Field(field_id)).ok_or_else(|| {
        crate::Error::Corruption(format!(
            "ANN vectors reference unknown schema field {field_id}"
        ))
    })?;

    let matches_schema = match index_type {
        ann_build::IVF_PQ_TYPE => {
            field.field_type == FieldType::DenseVector
                && field
                    .dense_vector_config
                    .as_ref()
                    .is_some_and(|config| config.index_type == VectorIndexType::IvfPq)
        }
        ann_build::BINARY_IVF_TYPE => {
            field.field_type == FieldType::BinaryDenseVector
                && field
                    .binary_dense_vector_config
                    .as_ref()
                    .is_some_and(|config| config.index_type == BinaryIndexType::Ivf)
        }
        _ => false,
    };

    if !matches_schema {
        return Err(crate::Error::Corruption(format!(
            "ANN vector type {index_type} for field {field_id} does not match schema field '{}' ({:?})",
            field.name, field.field_type
        )));
    }
    Ok(())
}

fn take_toc_bytes<'a>(
    data: &'a [u8],
    position: &mut usize,
    length: usize,
    description: &str,
) -> Result<&'a [u8]> {
    let end = position
        .checked_add(length)
        .ok_or_else(|| crate::Error::Corruption(format!("{description} range overflows usize")))?;
    let bytes = data.get(*position..end).ok_or_else(|| {
        crate::Error::Corruption(format!(
            "truncated {description}: need bytes {}..{}, TOC has {}",
            *position,
            end,
            data.len(),
        ))
    })?;
    *position = end;
    Ok(bytes)
}

use crate::segment::format::{
    DENSE_TOC_ENTRY_SIZE, DenseVectorTocEntry, FOOTER_SIZE, VECTORS_FOOTER_MAGIC, read_dense_toc,
};

fn is_ann_vector_type(index_type: u8) -> bool {
    use crate::segment::ann_build;

    matches!(
        index_type,
        ann_build::IVF_PQ_TYPE | ann_build::BINARY_IVF_TYPE
    )
}

fn checked_vector_payload_end(
    entry: &DenseVectorTocEntry,
    data_start: u64,
    data_end: u64,
) -> Result<u64> {
    entry
        .offset
        .checked_add(entry.size)
        .filter(|&end| entry.offset >= data_start && end <= data_end)
        .ok_or_else(|| {
            crate::Error::Corruption(format!(
                "vector field {} has invalid data range {}..{}+{}; valid payload is {data_start}..{data_end}",
                entry.field_id, entry.offset, entry.offset, entry.size
            ))
        })
}

fn validate_vector_toc_ranges(
    entries: &[DenseVectorTocEntry],
    data_start: u64,
    data_end: u64,
) -> Result<()> {
    let mut ranges = Vec::with_capacity(entries.len());
    for entry in entries {
        if is_ann_vector_type(entry.index_type) && entry.size == 0 {
            return Err(crate::Error::Corruption(format!(
                "ANN vectors for field {} have an empty payload",
                entry.field_id
            )));
        }
        let end = checked_vector_payload_end(entry, data_start, data_end)?;
        ranges.push((entry.offset, end, entry.field_id, entry.index_type));
    }

    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if current.0 < previous.1 {
            return Err(crate::Error::Corruption(format!(
                "vector TOC payloads overlap: field {} type {} uses {}..{}, field {} type {} uses {}..{}",
                previous.2,
                previous.3,
                previous.0,
                previous.1,
                current.2,
                current.3,
                current.0,
                current.1
            )));
        }
    }
    Ok(())
}

/// Load dense vector indexes from unified .vectors file
///
/// File format (data-first, TOC at end):
/// - [field data...]  — starts at offset 0 (mmap page-aligned)
/// - [TOC entries]    — field_id(4) + index_type(1) + offset(8) + size(8) per field
/// - [footer 16B]     — toc_offset(8) + num_fields(4) + magic(4)
pub async fn load_vectors_file<D: Directory>(
    dir: &D,
    files: &SegmentFiles,
    schema: &Schema,
    total_docs: u32,
) -> Result<VectorsFileData> {
    load_vectors_file_impl(dir, files, schema, total_docs, true, None).await
}

/// Open only selected flat-vector fields for training. The TOC and flat
/// payloads receive the same validation as search loading, but ANN run columns
/// are not mapped or parsed.
#[cfg(feature = "native")]
pub(crate) async fn load_flat_vectors_file<D: Directory>(
    dir: &D,
    files: &SegmentFiles,
    schema: &Schema,
    total_docs: u32,
    field_ids: &[u32],
) -> Result<VectorsFileData> {
    load_vectors_file_impl(dir, files, schema, total_docs, false, Some(field_ids)).await
}

async fn load_vectors_file_impl<D: Directory>(
    dir: &D,
    files: &SegmentFiles,
    schema: &Schema,
    total_docs: u32,
    load_ann: bool,
    flat_field_filter: Option<&[u32]>,
) -> Result<VectorsFileData> {
    let mut indexes = FxHashMap::default();
    let mut flat_vectors = FxHashMap::default();
    let empty = || VectorsFileData {
        indexes: FxHashMap::default(),
        flat_vectors: FxHashMap::default(),
        #[cfg(feature = "native")]
        ann_fields: Vec::new(),
    };

    // Skip loading vectors file if schema has no dense/binary dense vector fields
    let has_dense_vectors = schema.fields().any(|(_, entry)| {
        entry.dense_vector_config.is_some() || entry.binary_dense_vector_config.is_some()
    });
    if !has_dense_vectors {
        return Ok(empty());
    }

    // Try to open vectors file (may not exist if no vectors were indexed)
    let handle = match dir.open_lazy(&files.vectors).await {
        Ok(h) => h,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(empty()),
        Err(error) => return Err(crate::Error::Io(error)),
    };

    let file_size = handle.len();
    if file_size == 0 {
        return Ok(empty());
    }
    if file_size < FOOTER_SIZE {
        return Err(crate::Error::Corruption(format!(
            "vector file is {file_size} bytes, shorter than the required footer"
        )));
    }

    let footer_bytes = handle
        .read_bytes_range(file_size - FOOTER_SIZE..file_size)
        .await?;
    let mut cursor = Cursor::new(footer_bytes.as_slice());
    let toc_offset = cursor.read_u64::<LittleEndian>()?;
    let num_fields = cursor.read_u32::<LittleEndian>()?;
    let magic = cursor.read_u32::<LittleEndian>()?;
    if magic != VECTORS_FOOTER_MAGIC {
        return Err(crate::Error::Corruption(
            "dense-vector file has no supported TOC footer".into(),
        ));
    }
    let footer_start = file_size - FOOTER_SIZE;
    let toc_size = u64::from(num_fields)
        .checked_mul(DENSE_TOC_ENTRY_SIZE)
        .ok_or_else(|| crate::Error::Corruption("dense-vector TOC size overflows u64".into()))?;
    let toc_end = toc_offset
        .checked_add(toc_size)
        .ok_or_else(|| crate::Error::Corruption("dense-vector TOC range overflows u64".into()))?;
    if toc_offset > footer_start || toc_end > footer_start {
        return Err(crate::Error::Corruption(format!(
            "dense-vector TOC range {toc_offset}..{toc_end} exceeds footer start {footer_start}"
        )));
    }
    let toc_bytes = handle.read_bytes_range(toc_offset..toc_end).await?;
    let entries = read_dense_toc(toc_bytes.as_slice(), num_fields)?;
    let data_start = 0;
    let data_end = toc_offset;

    if entries.is_empty() {
        return Ok(empty());
    }
    validate_vector_toc_ranges(&entries, data_start, data_end)?;

    // Validate field-level TOC structure before opening any payload. ANN
    // always depends on the same field's exact flat data, and duplicate
    // entries are corruption regardless of which load mode the caller uses.
    use crate::segment::ann_build;
    let mut flat_toc_fields = FxHashSet::default();
    let mut ann_toc_fields = FxHashSet::default();
    for entry in &entries {
        let inserted = match entry.index_type {
            ann_build::FLAT_TYPE => flat_toc_fields.insert(entry.field_id),
            ann_build::IVF_PQ_TYPE | ann_build::BINARY_IVF_TYPE => {
                ann_toc_fields.insert(entry.field_id)
            }
            _ => continue,
        };
        if !inserted {
            return Err(crate::Error::Corruption(format!(
                "duplicate vector entry type {} for field {}",
                entry.index_type, entry.field_id,
            )));
        }
    }
    for &field_id in &ann_toc_fields {
        if !flat_toc_fields.contains(&field_id) {
            return Err(crate::Error::Corruption(format!(
                "ANN vectors for field {field_id} are missing matching flat vector storage",
            )));
        }
    }
    #[cfg(feature = "native")]
    let mut ann_fields: Vec<_> = ann_toc_fields.into_iter().collect();
    #[cfg(feature = "native")]
    ann_fields.sort_unstable();

    // Load each entry — a field can have both flat and mmap-backed ANN data.
    for DenseVectorTocEntry {
        field_id,
        index_type,
        offset,
        size: length,
    } in entries
    {
        // The complete TOC was bounds/overlap checked before any payload read.
        let end = offset + length;
        match index_type {
            ann_build::FLAT_TYPE => {
                // Search readers validate the zero-copy document map. Training
                // readers need only the header and sampled raw-vector ranges.
                let field = schema.get_field_entry(Field(field_id)).ok_or_else(|| {
                    crate::Error::Corruption(format!(
                        "flat vectors reference unknown schema field {field_id}"
                    ))
                })?;
                if flat_field_filter.is_some_and(|filter| !filter.contains(&field_id)) {
                    continue;
                }
                let slice = handle.slice(offset..end);
                let lazy_flat = if load_ann {
                    LazyFlatVectorData::open_with_doc_limit(slice, Some(total_docs)).await
                } else {
                    LazyFlatVectorData::open_for_training(slice).await
                }
                .map_err(|error| {
                    crate::Error::Corruption(format!(
                        "invalid flat vectors for field {field_id}: {error}"
                    ))
                })?;
                match lazy_flat.quantization {
                    DenseVectorQuantization::Binary => {
                        let config = field
                            .binary_dense_vector_config
                            .as_ref()
                            .filter(|_| field.field_type == FieldType::BinaryDenseVector)
                            .ok_or_else(|| {
                                crate::Error::Corruption(format!(
                                    "binary flat vectors for field {field_id} do not match its schema type"
                                ))
                            })?;
                        if lazy_flat.dim != config.dim {
                            return Err(crate::Error::Corruption(format!(
                                "binary flat vectors for field {field_id} have dimension {}, schema expects {}",
                                lazy_flat.dim, config.dim
                            )));
                        }
                    }
                    stored_quantization => {
                        let config = field
                            .dense_vector_config
                            .as_ref()
                            .filter(|_| field.field_type == FieldType::DenseVector)
                            .ok_or_else(|| {
                                crate::Error::Corruption(format!(
                                    "flat vectors for field {field_id} do not match its schema type"
                                ))
                            })?;
                        if lazy_flat.dim != config.dim || stored_quantization != config.quantization
                        {
                            return Err(crate::Error::Corruption(format!(
                                "flat vectors for field {field_id} have dimension {} and {:?} storage, schema expects {} and {:?}",
                                lazy_flat.dim, stored_quantization, config.dim, config.quantization
                            )));
                        }
                    }
                }
                if flat_vectors.insert(field_id, lazy_flat).is_some() {
                    return Err(crate::Error::Corruption(format!(
                        "duplicate flat-vector entry for field {field_id}"
                    )));
                }
            }
            ann_build::IVF_PQ_TYPE | ann_build::BINARY_IVF_TYPE => {
                validate_ann_schema(schema, field_id, index_type)?;
                if !load_ann {
                    continue;
                }
                // ANN corpus columns stay mmap-backed. Opening validates the
                // compact header/run directory without decoding payloads.
                let data = handle.read_bytes_range(offset..end).await?;
                let index = match index_type {
                    ann_build::IVF_PQ_TYPE => VectorIndex::IvfPq(Arc::new(
                        super::types::MmapAnnIndex::open(
                            data,
                            crate::segment::ann_disk::AnnKind::IvfPq,
                            total_docs,
                        )
                        .map_err(|error| {
                            crate::Error::Corruption(format!(
                                "invalid IVF-PQ payload for field {field_id}: {error}"
                            ))
                        })?,
                    )),
                    ann_build::BINARY_IVF_TYPE => VectorIndex::BinaryIvf(Arc::new(
                        super::types::MmapAnnIndex::open(
                            data,
                            crate::segment::ann_disk::AnnKind::BinaryIvf,
                            total_docs,
                        )
                        .map_err(|error| {
                            crate::Error::Corruption(format!(
                                "invalid binary IVF payload for field {field_id}: {error}"
                            ))
                        })?,
                    )),
                    _ => {
                        return Err(crate::Error::Corruption(format!(
                            "unknown vector index type {index_type} for field {field_id}"
                        )));
                    }
                };
                if indexes.insert(field_id, index).is_some() {
                    return Err(crate::Error::Corruption(format!(
                        "multiple ANN entries for vector field {field_id}"
                    )));
                }
            }
            _ => {
                return Err(crate::Error::Corruption(format!(
                    "unknown vector index type {index_type} for field {field_id}"
                )));
            }
        }
    }

    Ok(VectorsFileData {
        indexes,
        flat_vectors,
        #[cfg(feature = "native")]
        ann_fields,
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
) -> Result<SparseFileData> {
    use crate::segment::format::{SPARSE_FOOTER_MAGIC, SPARSE_FOOTER_SIZE};
    use crate::structures::{SparseSkipEntry, SparseVectorConfig};

    let empty = || SparseFileData {
        maxscore_indexes: FxHashMap::default(),
        bmp_indexes: FxHashMap::default(),
    };

    let mut maxscore_indexes = FxHashMap::default();
    let mut bmp_indexes = FxHashMap::default();

    // Skip loading sparse file if schema has no sparse vector fields
    let has_sparse_vectors = schema
        .fields()
        .any(|(_, entry)| entry.sparse_vector_config.is_some());
    if !has_sparse_vectors {
        return Ok(empty());
    }

    // Try to open sparse file lazily (may not exist if no sparse vectors were indexed)
    let handle = match dir.open_lazy(&files.sparse).await {
        Ok(h) => h,
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                log::debug!("No sparse file found ({}): {:?}", files.sparse.display(), e);
                return Ok(empty());
            }
            return Err(crate::Error::Io(e));
        }
    };

    let file_size = handle.len();
    if file_size < SPARSE_FOOTER_SIZE {
        return if file_size == 0 {
            Ok(empty())
        } else {
            Err(crate::Error::Corruption(format!(
                "sparse file is {file_size} bytes, shorter than its {SPARSE_FOOTER_SIZE}-byte footer"
            )))
        };
    }

    // Read footer (24 bytes): skip_offset(8) + toc_offset(8) + num_fields(4) + magic(4)
    let footer_bytes = handle
        .read_bytes_range(file_size - SPARSE_FOOTER_SIZE..file_size)
        .await?;
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

    let footer_start = file_size - SPARSE_FOOTER_SIZE;
    if skip_offset > toc_offset || toc_offset > footer_start {
        return Err(crate::Error::Corruption(format!(
            "invalid sparse section offsets: skip={skip_offset}, toc={toc_offset}, footer={footer_start}"
        )));
    }

    log::debug!(
        "Loading sparse vectors: size={}, num_fields={}, skip_offset={}, toc_offset={}",
        crate::format_bytes(file_size),
        num_fields,
        skip_offset,
        toc_offset,
    );

    if num_fields == 0 {
        if skip_offset != footer_start || toc_offset != footer_start {
            return Err(crate::Error::Corruption(format!(
                "empty sparse TOC leaves unowned bytes before footer: skip={skip_offset}, toc={toc_offset}, footer={footer_start}"
            )));
        }
        return Ok(empty());
    }

    // Single tail read: skip section + TOC (skip_offset .. footer_start)
    // For mmap this is zero-copy. Block data at the front is never touched.
    let tail_bytes = handle.read_bytes_range(skip_offset..footer_start).await?;
    let tail = tail_bytes.as_slice();

    // skip section is at tail[0 .. toc_offset - skip_offset]
    let skip_section_len = usize::try_from(toc_offset - skip_offset).map_err(|_| {
        crate::Error::Corruption("sparse skip section does not fit in address space".into())
    })?;
    if skip_section_len % SparseSkipEntry::SIZE != 0 {
        return Err(crate::Error::Corruption(format!(
            "sparse skip section is {skip_section_len} bytes, not a multiple of {}",
            SparseSkipEntry::SIZE,
        )));
    }
    let skip_section = tail_bytes.slice(0..skip_section_len);
    let toc_data = &tail[skip_section_len..];
    let skip_entry_count = skip_section_len / SparseSkipEntry::SIZE;

    // Parse TOC: per-field header(13B) + per-dim entries(28B each)
    let mut pos = 0usize;
    let mut payload_ranges: Vec<(u64, u64, u32, u32)> = Vec::new();
    let mut skip_ranges: Vec<(usize, usize, u32, u32)> = Vec::new();

    for _ in 0..num_fields {
        // Field header: field_id(4) + quant(1) + num_dims(4) + total_vectors(4) = 13 bytes
        let header = take_toc_bytes(toc_data, &mut pos, 13, "sparse field header")?;
        let field_id = u32::from_le_bytes(header[0..4].try_into().unwrap());
        let quantization = header[4];
        let ndims = u32::from_le_bytes(header[5..9].try_into().unwrap()) as usize;
        let total_vectors = u32::from_le_bytes(header[9..13].try_into().unwrap());
        let entries_len = ndims.checked_mul(28).ok_or_else(|| {
            crate::Error::Corruption(format!(
                "sparse field {field_id} dimension TOC size overflows usize"
            ))
        })?;
        let entries = take_toc_bytes(toc_data, &mut pos, entries_len, "sparse dimension entries")?;

        if maxscore_indexes.contains_key(&field_id) || bmp_indexes.contains_key(&field_id) {
            return Err(crate::Error::Corruption(format!(
                "duplicate sparse field {field_id} in TOC"
            )));
        }

        // Detect BMP format from the quant byte (bit 3 = format flag)
        let stored_config = SparseVectorConfig::from_byte(quantization).ok_or_else(|| {
            crate::Error::Corruption(format!(
                "invalid sparse configuration byte {quantization:#04x} for field {field_id}"
            ))
        })?;
        let is_bmp = stored_config.format == crate::structures::SparseFormat::Bmp;

        if is_bmp && ndims != 1 {
            return Err(crate::Error::Corruption(format!(
                "BMP field {field_id} has {ndims} TOC entries, expected one blob marker"
            )));
        }

        if is_bmp {
            // BMP field: single sentinel entry with blob location
            let d = &entries[..28];
            let dim_id = u32::from_le_bytes(d[0..4].try_into().unwrap());
            let blob_offset = u64::from_le_bytes(d[4..12].try_into().unwrap());
            let blob_len_low = u32::from_le_bytes(d[12..16].try_into().unwrap());
            let blob_len_high = u32::from_le_bytes(d[16..20].try_into().unwrap());

            if dim_id != 0xFFFFFFFF {
                return Err(crate::Error::Corruption(format!(
                    "BMP field {field_id} has dimension marker {dim_id:#x}, expected 0xffffffff"
                )));
            }

            let blob_len = (blob_len_high as u64) << 32 | blob_len_low as u64;
            let blob_end = blob_offset.checked_add(blob_len).ok_or_else(|| {
                crate::Error::Corruption(format!("BMP field {field_id} blob range overflows u64"))
            })?;
            if blob_end > skip_offset {
                return Err(crate::Error::Corruption(format!(
                    "BMP field {field_id} blob {blob_offset}..{blob_end} overlaps sparse metadata at {skip_offset}"
                )));
            }
            payload_ranges.push((blob_offset, blob_end, field_id, dim_id));

            match BmpIndex::parse(
                handle.clone(),
                blob_offset,
                blob_len,
                total_docs,
                total_vectors,
            ) {
                Ok(idx) => {
                    log::debug!(
                        "Loaded BMP sparse vector index for field {}: dims={}, num_blocks={}, total_vectors={}",
                        field_id,
                        idx.dims(),
                        idx.num_blocks,
                        total_vectors,
                    );
                    bmp_indexes.insert(field_id, idx);
                }
                Err(e) => {
                    return Err(e);
                }
            }
        } else {
            // MaxScore field: standard per-dimension entries
            let mut dims = super::types::DimensionTable::with_capacity(ndims);
            for d in entries.chunks_exact(28) {
                let dim_id = u32::from_le_bytes(d[0..4].try_into().unwrap());
                let block_data_offset = u64::from_le_bytes(d[4..12].try_into().unwrap());
                let skip_start = u32::from_le_bytes(d[12..16].try_into().unwrap());
                let num_blocks = u32::from_le_bytes(d[16..20].try_into().unwrap());
                let doc_count = u32::from_le_bytes(d[20..24].try_into().unwrap());
                let max_weight = f32::from_le_bytes(d[24..28].try_into().unwrap());
                let skip_end = (skip_start as usize)
                    .checked_add(num_blocks as usize)
                    .filter(|&end| end <= skip_entry_count)
                    .ok_or_else(|| {
                        crate::Error::Corruption(format!(
                            "sparse field {field_id} dimension {dim_id} references skip entries {skip_start}+{num_blocks}, but only {skip_entry_count} exist"
                        ))
                    })?;
                skip_ranges.push((skip_start as usize, skip_end, field_id, dim_id));
                if block_data_offset > skip_offset {
                    return Err(crate::Error::Corruption(format!(
                        "sparse field {field_id} dimension {dim_id} block offset {block_data_offset} exceeds data section {skip_offset}"
                    )));
                }
                if doc_count > total_docs {
                    return Err(crate::Error::Corruption(format!(
                        "sparse field {field_id} dimension {dim_id} has {doc_count} docs, segment has {total_docs}"
                    )));
                }
                if doc_count == 0 {
                    return Err(crate::Error::Corruption(format!(
                        "sparse field {field_id} dimension {dim_id} has blocks but no documents"
                    )));
                }
                let payload_range = validate_sparse_dimension_skip_entries(
                    skip_section.as_slice(),
                    skip_start as usize,
                    num_blocks as usize,
                    block_data_offset,
                    skip_offset,
                    total_docs,
                    max_weight,
                    field_id,
                    dim_id,
                )?;
                payload_ranges.push((payload_range.start, payload_range.end, field_id, dim_id));
                dims.push(
                    dim_id,
                    block_data_offset,
                    skip_start,
                    num_blocks,
                    doc_count,
                    max_weight,
                );
            }
            dims.sort_by_dim_id();
            if dims.dim_ids.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(crate::Error::Corruption(format!(
                    "sparse field {field_id} contains duplicate dimension IDs"
                )));
            }

            log::debug!(
                "Loaded sparse vector index for field {}: num_dims={}, total_vectors={}, skip={}",
                field_id,
                dims.len(),
                total_vectors,
                crate::format_bytes(skip_section.len() as u64),
            );

            maxscore_indexes.insert(
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
    }

    payload_ranges.sort_unstable_by_key(|range| (range.0, range.1, range.2, range.3));
    for pair in payload_ranges.windows(2) {
        let (left_start, left_end, left_field, left_dim) = pair[0];
        let (right_start, right_end, right_field, right_dim) = pair[1];
        if right_start < left_end {
            return Err(crate::Error::Corruption(format!(
                "sparse payloads overlap: field {left_field} dimension {left_dim} uses \
                 {left_start}..{left_end}, field {right_field} dimension {right_dim} uses \
                 {right_start}..{right_end}"
            )));
        }
    }
    skip_ranges.sort_unstable_by_key(|range| (range.0, range.1, range.2, range.3));
    let mut owned_skip_entries = 0usize;
    for &(start, end, field_id, dim_id) in &skip_ranges {
        if start != owned_skip_entries {
            return Err(crate::Error::Corruption(format!(
                "sparse skip-entry ownership has a gap or overlap before field {field_id} \
                 dimension {dim_id}: expected start {owned_skip_entries}, got {start}"
            )));
        }
        owned_skip_entries = end;
    }
    if owned_skip_entries != skip_entry_count {
        return Err(crate::Error::Corruption(format!(
            "sparse skip section contains {} unowned entries",
            skip_entry_count - owned_skip_entries
        )));
    }

    if pos != toc_data.len() {
        return Err(crate::Error::Corruption(format!(
            "sparse TOC has {} trailing bytes after {num_fields} fields",
            toc_data.len() - pos,
        )));
    }

    log::debug!(
        "Sparse file loaded: maxscore_fields={:?}, bmp_fields={:?}",
        maxscore_indexes.keys().collect::<Vec<_>>(),
        bmp_indexes.keys().collect::<Vec<_>>()
    );

    Ok(SparseFileData {
        maxscore_indexes,
        bmp_indexes,
    })
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
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(crate::Error::Io(error)),
    }
}

/// Load fast-field columns from `.fast` file.
/// Returns a map of field_id → FastFieldReader.
pub async fn load_fast_fields_file<D: Directory>(
    dir: &D,
    files: &SegmentFiles,
    schema: &Schema,
) -> Result<FxHashMap<u32, crate::structures::fast_field::FastFieldReader>> {
    use crate::structures::fast_field::{
        FastFieldReader, read_fast_field_footer, read_fast_field_toc,
    };

    // Skip if no fast fields in schema
    let has_fast = schema.fields().any(|(_, entry)| entry.fast);
    if !has_fast {
        return Ok(FxHashMap::default());
    }

    // Try to open the .fast file (may not exist for old segments)
    let handle = match dir.open_read(&files.fast).await {
        Ok(h) => h,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            log::debug!("[fast-fields] .fast file not found ({}), skipping", e);
            return Ok(FxHashMap::default());
        }
        Err(e) => return Err(crate::Error::Io(e)),
    };

    let file_data = handle.read_bytes().await?;
    if file_data.is_empty() {
        return Ok(FxHashMap::default());
    }

    let (toc_offset, num_columns) = read_fast_field_footer(&file_data).map_err(crate::Error::Io)?;

    let mut readers = FxHashMap::default();

    let toc_entries =
        read_fast_field_toc(&file_data, toc_offset, num_columns).map_err(crate::Error::Io)?;
    for toc in &toc_entries {
        let reader = FastFieldReader::open(&file_data, toc).map_err(crate::Error::Io)?;
        readers.insert(toc.field_id, reader);
    }

    log::debug!(
        "[fast-fields] loaded {} columns from .fast file",
        readers.len(),
    );

    Ok(readers)
}

#[cfg(test)]
mod tests {
    use crate::directories::{DirectoryWriter, RamDirectory};
    use crate::dsl::{DenseVectorQuantization, SchemaBuilder};
    use crate::segment::format::{DenseVectorTocEntry, write_dense_toc_and_footer};
    use crate::segment::{FlatVectorData, ann_build};
    use crate::structures::{SparseSkipEntry, SparseVectorConfig};

    use super::{
        SegmentFiles, load_flat_vectors_file, load_sparse_file, load_vectors_file,
        validate_sparse_dimension_skip_entries,
    };

    fn encoded_skip_entries(entries: &[SparseSkipEntry]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(entries.len() * SparseSkipEntry::SIZE);
        for entry in entries {
            entry.write_to_vec(&mut bytes);
        }
        bytes
    }

    fn vectors_file_with_toc(mut payload: Vec<u8>, entries: &[DenseVectorTocEntry]) -> Vec<u8> {
        let toc_offset = payload.len() as u64;
        write_dense_toc_and_footer(&mut payload, toc_offset, entries).unwrap();
        payload
    }

    fn vectors_file_with_payloads(payloads: Vec<(u32, u8, Vec<u8>)>) -> Vec<u8> {
        let mut file = Vec::new();
        let mut entries = Vec::with_capacity(payloads.len());
        for (field_id, index_type, payload) in payloads {
            let offset = file.len() as u64;
            let size = payload.len() as u64;
            file.extend_from_slice(&payload);
            entries.push(DenseVectorTocEntry {
                field_id,
                index_type,
                offset,
                size,
            });
        }
        vectors_file_with_toc(file, &entries)
    }

    fn vectors_file_with_flat_payload(payload: Vec<u8>) -> Vec<u8> {
        vectors_file_with_payloads(vec![(0, ann_build::FLAT_TYPE, payload)])
    }

    fn one_dense_flat_payload() -> Vec<u8> {
        let mut payload = Vec::new();
        FlatVectorData::serialize_binary_from_flat_streaming(
            2,
            &[1.0, 2.0],
            &[(0, 0)],
            DenseVectorQuantization::F32,
            &mut payload,
        )
        .unwrap();
        payload
    }

    #[tokio::test]
    async fn existing_truncated_sparse_file_is_corruption() {
        let mut schema = SchemaBuilder::default();
        schema.add_sparse_vector_field_with_config(
            "sparse",
            true,
            true,
            SparseVectorConfig::default(),
        );
        let schema = schema.build();
        let files = SegmentFiles::new(7);
        let dir = RamDirectory::new();
        dir.write(&files.sparse, &[1, 2, 3]).await.unwrap();

        let result = load_sparse_file(&dir, &files, 1, &schema).await;
        assert!(matches!(result, Err(crate::Error::Corruption(_))));
    }

    #[tokio::test]
    async fn unknown_dense_vector_type_is_corruption() {
        let mut schema = SchemaBuilder::default();
        schema.add_binary_dense_vector_field("binary", 8, true, true);
        let schema = schema.build();
        let files = SegmentFiles::new(8);
        let dir = RamDirectory::new();

        let mut bytes = vec![0];
        write_dense_toc_and_footer(
            &mut bytes,
            1,
            &[DenseVectorTocEntry {
                field_id: 0,
                index_type: u8::MAX,
                offset: 0,
                size: 1,
            }],
        )
        .unwrap();
        dir.write(&files.vectors, &bytes).await.unwrap();

        let result = load_vectors_file(&dir, &files, &schema, 1).await;
        assert!(matches!(result, Err(crate::Error::Corruption(_))));
    }

    #[tokio::test]
    async fn existing_truncated_dense_vector_file_is_corruption() {
        let mut schema = SchemaBuilder::default();
        schema.add_binary_dense_vector_field("binary", 8, true, true);
        let schema = schema.build();
        let files = SegmentFiles::new(9);
        let dir = RamDirectory::new();
        dir.write(&files.vectors, &[1, 2, 3]).await.unwrap();

        let result = load_vectors_file(&dir, &files, &schema, 1).await;
        assert!(matches!(result, Err(crate::Error::Corruption(_))));
    }

    #[tokio::test]
    async fn flat_vectors_must_match_schema_storage() {
        let mut schema = SchemaBuilder::default();
        schema.add_dense_vector_field("dense", 2, true, true);
        let schema = schema.build();
        let files = SegmentFiles::new(10);
        let dir = RamDirectory::new();

        let mut payload = Vec::new();
        FlatVectorData::serialize_binary_from_flat_streaming(
            2,
            &[1.0, 2.0],
            &[(0, 0)],
            DenseVectorQuantization::F16,
            &mut payload,
        )
        .unwrap();
        dir.write(&files.vectors, &vectors_file_with_flat_payload(payload))
            .await
            .unwrap();

        let result = load_vectors_file(&dir, &files, &schema, 1).await;
        assert!(matches!(result, Err(crate::Error::Corruption(_))));
    }

    #[tokio::test]
    async fn flat_vector_doc_ids_must_fit_segment_metadata() {
        let mut schema = SchemaBuilder::default();
        schema.add_dense_vector_field("dense", 2, true, true);
        let schema = schema.build();
        let files = SegmentFiles::new(11);
        let dir = RamDirectory::new();

        let mut payload = Vec::new();
        FlatVectorData::serialize_binary_from_flat_streaming(
            2,
            &[1.0, 2.0],
            &[(1, 0)],
            DenseVectorQuantization::F32,
            &mut payload,
        )
        .unwrap();
        dir.write(&files.vectors, &vectors_file_with_flat_payload(payload))
            .await
            .unwrap();

        let result = load_vectors_file(&dir, &files, &schema, 1).await;
        assert!(matches!(result, Err(crate::Error::Corruption(_))));
    }

    #[tokio::test]
    async fn ann_vectors_require_same_field_flat_storage() {
        let mut schema = SchemaBuilder::default();
        schema.add_dense_vector_field("dense_0", 2, true, true);
        schema.add_dense_vector_field("dense_1", 2, true, true);
        let schema = schema.build();
        let files = SegmentFiles::new(12);
        let dir = RamDirectory::new();

        let mut flat = Vec::new();
        FlatVectorData::serialize_binary_from_flat_streaming(
            2,
            &[1.0, 2.0],
            &[(0, 0)],
            DenseVectorQuantization::F32,
            &mut flat,
        )
        .unwrap();
        let bytes = vectors_file_with_payloads(vec![
            (0, ann_build::FLAT_TYPE, flat),
            (1, ann_build::IVF_PQ_TYPE, vec![0]),
        ]);
        dir.write(&files.vectors, &bytes).await.unwrap();

        let result = load_vectors_file(&dir, &files, &schema, 1).await;
        let Err(crate::Error::Corruption(message)) = result else {
            panic!("ANN storage without a same-field flat payload must be rejected");
        };
        assert!(message.contains("field 1"));
        assert!(message.contains("missing matching flat"));
    }

    #[tokio::test]
    async fn training_loads_flat_samples_without_opening_ann_columns() {
        let mut schema = SchemaBuilder::default();
        schema.add_dense_vector_field("dense", 2, true, true);
        let schema = schema.build();
        let files = SegmentFiles::new(120);
        let dir = RamDirectory::new();
        let bytes = vectors_file_with_payloads(vec![
            (0, ann_build::FLAT_TYPE, one_dense_flat_payload()),
            (0, ann_build::IVF_PQ_TYPE, vec![0]),
        ]);
        dir.write(&files.vectors, &bytes).await.unwrap();

        let training = load_flat_vectors_file(&dir, &files, &schema, 1, &[0])
            .await
            .unwrap();
        assert_eq!(training.ann_fields, [0]);
        assert_eq!(training.flat_vectors[&0].num_vectors, 1);
        assert_eq!(training.flat_vectors[&0].num_docs_with_vectors(), 0);
        assert!(load_vectors_file(&dir, &files, &schema, 1).await.is_err());
    }

    #[tokio::test]
    async fn ann_vector_type_must_match_schema_index_type() {
        let mut schema = SchemaBuilder::default();
        schema.add_dense_vector_field_with_config(
            "dense",
            true,
            true,
            crate::dsl::DenseVectorConfig::flat(2),
        );
        let schema = schema.build();
        let files = SegmentFiles::new(13);
        let dir = RamDirectory::new();

        let mut flat = Vec::new();
        FlatVectorData::serialize_binary_from_flat_streaming(
            2,
            &[1.0, 2.0],
            &[(0, 0)],
            DenseVectorQuantization::F32,
            &mut flat,
        )
        .unwrap();
        let bytes = vectors_file_with_payloads(vec![
            (0, ann_build::FLAT_TYPE, flat),
            (0, ann_build::IVF_PQ_TYPE, vec![0]),
        ]);
        dir.write(&files.vectors, &bytes).await.unwrap();

        let result = load_vectors_file(&dir, &files, &schema, 1).await;
        let Err(crate::Error::Corruption(message)) = result else {
            panic!("ANN storage that disagrees with the schema must be rejected");
        };
        assert!(message.contains("does not match schema"));
    }

    #[tokio::test]
    async fn flat_only_vectors_are_valid_before_ann_build() {
        let mut schema = SchemaBuilder::default();
        schema.add_dense_vector_field("dense", 2, true, true);
        let schema = schema.build();
        let files = SegmentFiles::new(14);
        let dir = RamDirectory::new();

        let mut flat = Vec::new();
        FlatVectorData::serialize_binary_from_flat_streaming(
            2,
            &[1.0, 2.0],
            &[(0, 0)],
            DenseVectorQuantization::F32,
            &mut flat,
        )
        .unwrap();
        dir.write(&files.vectors, &vectors_file_with_flat_payload(flat))
            .await
            .unwrap();

        let loaded = load_vectors_file(&dir, &files, &schema, 1)
            .await
            .expect("flat-only pre-threshold segment must remain readable");
        assert!(loaded.indexes.is_empty());
        assert!(loaded.flat_vectors.contains_key(&0));
    }

    #[tokio::test]
    async fn ann_vector_payload_must_not_be_empty() {
        let mut schema = SchemaBuilder::default();
        schema.add_dense_vector_field("dense", 2, true, true);
        let schema = schema.build();
        let files = SegmentFiles::new(15);
        let dir = RamDirectory::new();

        let bytes = vectors_file_with_payloads(vec![
            (0, ann_build::FLAT_TYPE, one_dense_flat_payload()),
            (0, ann_build::IVF_PQ_TYPE, Vec::new()),
        ]);
        dir.write(&files.vectors, &bytes).await.unwrap();

        let result = load_vectors_file(&dir, &files, &schema, 1).await;
        let Err(crate::Error::Corruption(message)) = result else {
            panic!("an empty ANN payload must be rejected");
        };
        assert!(message.contains("empty payload"));
    }

    #[tokio::test]
    async fn vector_toc_rejects_overlapping_and_aliased_payloads() {
        let mut schema = SchemaBuilder::default();
        schema.add_dense_vector_field("dense", 2, true, true);
        let schema = schema.build();
        let flat = one_dense_flat_payload();
        let flat_len = flat.len() as u64;

        // Exercise both exact aliasing and a one-byte partial overlap.
        for (segment_id, ann_offset, ann_size) in [(16, 0, flat_len), (17, flat_len - 1, 2)] {
            let files = SegmentFiles::new(segment_id);
            let dir = RamDirectory::new();
            let mut payload = flat.clone();
            let required_len = (ann_offset + ann_size) as usize;
            payload.resize(payload.len().max(required_len), 0);
            let entries = [
                DenseVectorTocEntry {
                    field_id: 0,
                    index_type: ann_build::FLAT_TYPE,
                    offset: 0,
                    size: flat_len,
                },
                DenseVectorTocEntry {
                    field_id: 0,
                    index_type: ann_build::IVF_PQ_TYPE,
                    offset: ann_offset,
                    size: ann_size,
                },
            ];
            let bytes = vectors_file_with_toc(payload, &entries);
            dir.write(&files.vectors, &bytes).await.unwrap();

            let result = load_vectors_file(&dir, &files, &schema, 1).await;
            let Err(crate::Error::Corruption(message)) = result else {
                panic!("overlapping vector payloads must be rejected");
            };
            assert!(message.contains("overlap"));
        }
    }

    #[tokio::test]
    async fn vector_toc_allows_unreferenced_padding_gaps() {
        let mut schema = SchemaBuilder::default();
        schema.add_dense_vector_field("dense_0", 2, true, true);
        schema.add_dense_vector_field("dense_1", 2, true, true);
        let schema = schema.build();
        let files = SegmentFiles::new(18);
        let dir = RamDirectory::new();

        let first = one_dense_flat_payload();
        let second = one_dense_flat_payload();
        let first_size = first.len() as u64;
        let padding = 8 - first.len() % 8;
        let second_offset = first_size + padding as u64;
        let second_size = second.len() as u64;
        let mut payload = first;
        payload.resize(payload.len() + padding, 0);
        payload.extend_from_slice(&second);
        let entries = [
            DenseVectorTocEntry {
                field_id: 0,
                index_type: ann_build::FLAT_TYPE,
                offset: 0,
                size: first_size,
            },
            DenseVectorTocEntry {
                field_id: 1,
                index_type: ann_build::FLAT_TYPE,
                offset: second_offset,
                size: second_size,
            },
        ];
        let bytes = vectors_file_with_toc(payload, &entries);
        dir.write(&files.vectors, &bytes).await.unwrap();

        let loaded = load_vectors_file(&dir, &files, &schema, 1)
            .await
            .expect("padding between disjoint vector payloads must be allowed");
        assert_eq!(loaded.flat_vectors.len(), 2);
    }

    #[test]
    fn sparse_skip_metadata_rejects_unsafe_ranges_and_pruning_bounds() {
        let valid = [
            SparseSkipEntry::new(0, 4, 0, 8, 2.0),
            SparseSkipEntry::new(4, 9, 8, 12, 3.0),
        ];
        let valid_bytes = encoded_skip_entries(&valid);
        validate_sparse_dimension_skip_entries(&valid_bytes, 0, valid.len(), 10, 30, 10, 3.0, 0, 7)
            .unwrap();

        let overlapping = encoded_skip_entries(&[
            SparseSkipEntry::new(0, 4, 0, 8, 2.0),
            SparseSkipEntry::new(5, 9, 7, 4, 2.0),
        ]);
        assert!(
            validate_sparse_dimension_skip_entries(&overlapping, 0, 2, 10, 30, 10, 2.0, 0, 7,)
                .is_err()
        );

        let unsafe_bound = encoded_skip_entries(&[SparseSkipEntry::new(0, 9, 0, 8, 4.0)]);
        assert!(
            validate_sparse_dimension_skip_entries(&unsafe_bound, 0, 1, 10, 30, 10, 3.0, 0, 7,)
                .is_err()
        );
    }
}
