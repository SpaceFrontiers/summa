//! Shared format constants for segment binary files.
//!
//! `.vectors` uses a 16-byte footer-based (data-first) layout.
//! `.sparse` V3 uses a 24-byte footer with separate skip section.
//!
//! Data starts at offset 0 (mmap page-aligned). The footer points back
//! to the TOC via `toc_offset`.

// ── Shared ──────────────────────────────────────────────────────────────────

/// Footer size for `.vectors` file.
/// Layout: toc_offset(8) + num_entries(4) + magic(4) = 16 bytes.
pub const FOOTER_SIZE: u64 = 16;

// ── Dense vectors (.vectors) ────────────────────────────────────────────────

/// Magic number for `.vectors` file footer ("VEC2" in LE)
pub const VECTORS_FOOTER_MAGIC: u32 = 0x32434556;

/// Magic number for flat vector binary header ("FVD3" in LE)
pub const FLAT_BINARY_MAGIC: u32 = 0x46564433;

/// Flat vector binary header size: magic(4) + dim(4) + num_vectors(4) + quant(1) + pad(3)
pub const FLAT_BINARY_HEADER_SIZE: usize = 16;

/// Per-doc_id entry size: doc_id(u32) + ordinal(u16)
pub const DOC_ID_ENTRY_SIZE: usize = std::mem::size_of::<u32>() + std::mem::size_of::<u16>();

/// Per-field TOC entry for `.vectors` file.
/// Shared by builder, merger, and reader.
/// Wire format: field_id(4) + index_type(1) + offset(8) + size(8) = 21 bytes.
pub struct DenseVectorTocEntry {
    pub field_id: u32,
    pub index_type: u8,
    pub offset: u64,
    pub size: u64,
}

/// Size in bytes of a single dense vector TOC entry on disk.
pub const DENSE_TOC_ENTRY_SIZE: u64 = 4 + 1 + 8 + 8; // 21

/// Write dense vector TOC entries + footer to writer.
///
/// Called after all field data has been written. `toc_offset` is the
/// current file position (byte offset where the TOC starts).
#[cfg(feature = "native")]
pub fn write_dense_toc_and_footer(
    writer: &mut (impl std::io::Write + ?Sized),
    toc_offset: u64,
    entries: &[DenseVectorTocEntry],
) -> std::io::Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};

    for e in entries {
        writer.write_u32::<LittleEndian>(e.field_id)?;
        writer.write_u8(e.index_type)?;
        writer.write_u64::<LittleEndian>(e.offset)?;
        writer.write_u64::<LittleEndian>(e.size)?;
    }
    writer.write_u64::<LittleEndian>(toc_offset)?;
    writer.write_u32::<LittleEndian>(entries.len() as u32)?;
    writer.write_u32::<LittleEndian>(VECTORS_FOOTER_MAGIC)?;
    Ok(())
}

/// Read dense vector TOC entries from raw bytes (already loaded from file).
pub fn read_dense_toc(
    toc_bytes: &[u8],
    num_fields: u32,
) -> std::io::Result<Vec<DenseVectorTocEntry>> {
    use byteorder::{LittleEndian, ReadBytesExt};
    let mut cursor = std::io::Cursor::new(toc_bytes);
    let mut entries = Vec::with_capacity(num_fields as usize);
    for _ in 0..num_fields {
        let field_id = cursor.read_u32::<LittleEndian>()?;
        let index_type = cursor.read_u8().unwrap_or(255);
        let offset = cursor.read_u64::<LittleEndian>()?;
        let size = cursor.read_u64::<LittleEndian>()?;
        entries.push(DenseVectorTocEntry {
            field_id,
            index_type,
            offset,
            size,
        });
    }
    Ok(entries)
}

// ── Sparse vectors (.sparse) ────────────────────────────────────────────────

/// Magic number for `.sparse` file footer ("SPR4" in LE)
/// Field header: field_id(4) + quant(1) + num_dims(4) + total_vectors(4) = 13B
pub const SPARSE_FOOTER_MAGIC: u32 = 0x34525053;

/// Magic number for BMP V5 blob footer within `.sparse` file ("BMP5" in LE)
pub const BMP_BLOB_MAGIC_V5: u32 = 0x35504D42;

/// BMP V5 blob footer size (48 bytes):
/// total_terms(4) + total_postings(4) + dim_ids_offset(4) + grid_offset(4) +
/// num_blocks(4) + num_dims(4) + bmp_block_size(4) + num_ordinals(4) +
/// max_weight_scale(4) + sb_grid_offset(4) + _reserved(4) + magic(4)
pub const BMP_BLOB_FOOTER_SIZE_V5: usize = 48;

/// V3 footer size: skip_offset(8) + toc_offset(8) + num_fields(4) + magic(4) = 24
pub const SPARSE_FOOTER_SIZE: u64 = 24;

/// Per-dim TOC entry accumulated during V3 sparse build/merge.
#[cfg(feature = "native")]
pub struct SparseDimTocEntry {
    pub dim_id: u32,
    /// Absolute byte offset in file where block data starts for this dim
    pub block_data_offset: u64,
    /// Index into the global skip-entry array (entry index, not byte offset)
    pub skip_start: u32,
    /// Number of skip entries (= number of blocks)
    pub num_blocks: u32,
    /// Total document count in this dimension's posting list
    pub doc_count: u32,
    /// Global max weight across all blocks
    pub max_weight: f32,
}

/// Per-field TOC entry accumulated during V3 sparse build/merge.
#[cfg(feature = "native")]
pub struct SparseFieldToc {
    pub field_id: u32,
    pub quantization: u8,
    /// Total number of documents that have at least one sparse vector in this field.
    /// Not the same as segment total_docs when some docs lack the field.
    pub total_vectors: u32,
    pub dims: Vec<SparseDimTocEntry>,
}

/// Write V3 sparse TOC + footer.
///
/// V3 layout:
/// ```text
/// [block data ...]
/// [skip section: SparseSkipEntry × total_skips (20B each)]
/// [TOC: per-field header(13B) + per-dim entries(28B each)]
/// [footer: skip_offset(8) + toc_offset(8) + num_fields(4) + magic(4)]
/// ```
#[cfg(feature = "native")]
pub fn write_sparse_toc_and_footer(
    writer: &mut (impl std::io::Write + ?Sized),
    skip_offset: u64,
    toc_offset: u64,
    field_tocs: &[SparseFieldToc],
) -> std::io::Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};

    for ftoc in field_tocs {
        writer.write_u32::<LittleEndian>(ftoc.field_id)?;
        writer.write_u8(ftoc.quantization)?;
        writer.write_u32::<LittleEndian>(ftoc.dims.len() as u32)?;
        writer.write_u32::<LittleEndian>(ftoc.total_vectors)?;
        for d in &ftoc.dims {
            writer.write_u32::<LittleEndian>(d.dim_id)?;
            writer.write_u64::<LittleEndian>(d.block_data_offset)?;
            writer.write_u32::<LittleEndian>(d.skip_start)?;
            writer.write_u32::<LittleEndian>(d.num_blocks)?;
            writer.write_u32::<LittleEndian>(d.doc_count)?;
            writer.write_f32::<LittleEndian>(d.max_weight)?;
        }
    }
    writer.write_u64::<LittleEndian>(skip_offset)?;
    writer.write_u64::<LittleEndian>(toc_offset)?;
    writer.write_u32::<LittleEndian>(field_tocs.len() as u32)?;
    writer.write_u32::<LittleEndian>(SPARSE_FOOTER_MAGIC)?;
    Ok(())
}
