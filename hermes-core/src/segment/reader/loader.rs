//! Index loading functions for segment reader

use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};
use rustc_hash::FxHashMap;
use std::io::Cursor;

use super::super::types::SegmentFiles;
use super::super::vector_data::LazyFlatVectorData;
use super::bmp::BmpIndex;
use super::{SparseIndex, VectorIndex};
use crate::Result;
use crate::directories::{Directory, FileHandle};
use crate::dsl::Schema;

/// Result of loading the `.sparse` file — may contain MaxScore and/or BMP indexes.
pub struct SparseFileData {
    pub maxscore_indexes: FxHashMap<u32, SparseIndex>,
    pub bmp_indexes: FxHashMap<u32, BmpIndex>,
}

/// Vectors file loading result
pub struct VectorsFileData {
    /// ANN indexes per field (IVF, ScaNN, RaBitQ) — loaded into memory for search
    pub indexes: FxHashMap<u32, VectorIndex>,
    /// Lazy flat vectors per field — doc_ids in memory, vectors via mmap for reranking/merge
    pub flat_vectors: FxHashMap<u32, LazyFlatVectorData>,
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
    if file_size < 4 {
        return Err(crate::Error::Corruption(format!(
            "vector file is {file_size} bytes, shorter than a legacy header"
        )));
    }

    // Try new format when a complete footer is present; otherwise validate
    // the legacy header instead of silently accepting a truncated file.
    let footer = if file_size >= FOOTER_SIZE {
        let footer_bytes = handle
            .read_bytes_range(file_size - FOOTER_SIZE..file_size)
            .await?;
        let mut cursor = Cursor::new(footer_bytes.as_slice());
        Some((
            cursor.read_u64::<LittleEndian>()?,
            cursor.read_u32::<LittleEndian>()?,
            cursor.read_u32::<LittleEndian>()?,
        ))
    } else {
        None
    };

    let (entries, data_start, data_end): (Vec<DenseVectorTocEntry>, u64, u64) = if let Some((
        toc_offset,
        num_fields,
        _,
    )) =
        footer.filter(|(_, _, magic)| *magic == VECTORS_FOOTER_MAGIC)
    {
        // New format: TOC at end
        let footer_start = file_size - FOOTER_SIZE;
        let toc_size = u64::from(num_fields)
            .checked_mul(DENSE_TOC_ENTRY_SIZE)
            .ok_or_else(|| {
                crate::Error::Corruption("dense-vector TOC size overflows u64".into())
            })?;
        let toc_end = toc_offset.checked_add(toc_size).ok_or_else(|| {
            crate::Error::Corruption("dense-vector TOC range overflows u64".into())
        })?;
        if toc_offset > footer_start || toc_end > footer_start {
            return Err(crate::Error::Corruption(format!(
                "dense-vector TOC range {toc_offset}..{toc_end} exceeds footer start {footer_start}"
            )));
        }
        let toc_bytes = handle.read_bytes_range(toc_offset..toc_end).await?;
        (
            read_dense_toc(toc_bytes.as_slice(), num_fields)?,
            0,
            toc_offset,
        )
    } else {
        // Legacy format: header at start (num_fields(4) + entries)
        let header_bytes = handle.read_bytes_range(0..4).await?;
        let mut cursor = Cursor::new(header_bytes.as_slice());
        let num_fields = cursor.read_u32::<LittleEndian>()?;
        if num_fields == 0 {
            return Ok(empty());
        }
        let entries_size = u64::from(num_fields)
            .checked_mul(DENSE_TOC_ENTRY_SIZE)
            .ok_or_else(|| {
                crate::Error::Corruption("legacy dense-vector TOC size overflows u64".into())
            })?;
        let entries_end = 4u64.checked_add(entries_size).ok_or_else(|| {
            crate::Error::Corruption("legacy dense-vector TOC range overflows u64".into())
        })?;
        if entries_end > file_size {
            return Err(crate::Error::Corruption(format!(
                "legacy dense-vector TOC ends at {entries_end}, beyond file size {file_size}"
            )));
        }
        let entries_bytes = handle.read_bytes_range(4..entries_end).await?;
        (
            read_dense_toc(entries_bytes.as_slice(), num_fields)?,
            entries_end,
            file_size,
        )
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
        let end = offset
            .checked_add(length)
            .filter(|&end| offset >= data_start && end <= data_end)
            .ok_or_else(|| {
                crate::Error::Corruption(format!(
                    "vector field {field_id} has invalid data range {offset}..{offset}+{length}; valid payload is {data_start}..{data_end}"
                ))
            })?;
        match index_type {
            ann_build::FLAT_TYPE => {
                // Flat binary — load lazily (only doc_ids in memory, vectors via mmap)
                let slice = handle.slice(offset..end);
                let lazy_flat = LazyFlatVectorData::open(slice).await.map_err(|error| {
                    crate::Error::Corruption(format!(
                        "invalid flat vectors for field {field_id}: {error}"
                    ))
                })?;
                if flat_vectors.insert(field_id, lazy_flat).is_some() {
                    return Err(crate::Error::Corruption(format!(
                        "duplicate flat-vector entry for field {field_id}"
                    )));
                }
            }
            ann_build::SCANN_TYPE
            | ann_build::IVF_RABITQ_TYPE
            | ann_build::BINARY_IVF_TYPE
            | ann_build::RABITQ_TYPE => {
                // ANN payloads stay lazy and zero-copy; only their validated
                // byte range is retained until first search.
                let data = handle.read_bytes_range(offset..end).await?;
                let index = match index_type {
                    ann_build::SCANN_TYPE => {
                        VectorIndex::ScaNN(Arc::new(super::types::LazyScaNN::new(data)))
                    }
                    ann_build::IVF_RABITQ_TYPE => {
                        VectorIndex::IVF(Arc::new(super::types::LazyIVF::new(data)))
                    }
                    ann_build::BINARY_IVF_TYPE => {
                        VectorIndex::BinaryIvf(Arc::new(super::types::LazyBinaryIvf::new(data)))
                    }
                    ann_build::RABITQ_TYPE => {
                        VectorIndex::RaBitQ(Arc::new(super::types::LazyRaBitQ::new(data)))
                    }
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
        "Loading sparse: size={} bytes, num_fields={}, skip_offset={}, toc_offset={}",
        file_size,
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

            match BmpIndex::parse(
                handle.clone(),
                blob_offset,
                blob_len,
                total_docs,
                total_vectors,
            ) {
                Ok(idx) => {
                    log::debug!(
                        "Loaded BMP index for field {}: dims={}, num_blocks={}, total_vectors={}",
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
                let _skip_end = (skip_start as usize)
                    .checked_add(num_blocks as usize)
                    .filter(|&end| end <= skip_entry_count)
                    .ok_or_else(|| {
                        crate::Error::Corruption(format!(
                            "sparse field {field_id} dimension {dim_id} references skip entries {skip_start}+{num_blocks}, but only {skip_entry_count} exist"
                        ))
                    })?;
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
                "Loaded sparse index for field {}: num_dims={}, total_vectors={}, skip_bytes={}",
                field_id,
                dims.len(),
                total_vectors,
                skip_section.len(),
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
    use crate::dsl::SchemaBuilder;
    use crate::segment::format::{DenseVectorTocEntry, write_dense_toc_and_footer};
    use crate::structures::SparseVectorConfig;

    use super::{SegmentFiles, load_sparse_file, load_vectors_file};

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

        let result = load_vectors_file(&dir, &files, &schema).await;
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

        let result = load_vectors_file(&dir, &files, &schema).await;
        assert!(matches!(result, Err(crate::Error::Corruption(_))));
    }
}
