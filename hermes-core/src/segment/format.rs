//! Shared format constants for segment binary files.
//!
//! Both `.vectors` and `.sparse` files use a footer-based (data-first) layout:
//! ```text
//! [field data...]
//! [TOC entries]
//! [footer: toc_offset(u64) + num_entries(u32) + magic(u32)]
//! ```
//! Data starts at offset 0 (mmap page-aligned). The footer (last 16 bytes)
//! points back to the TOC via `toc_offset`.

// ── Shared ──────────────────────────────────────────────────────────────────

/// Footer size shared by `.vectors` and `.sparse` files.
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

/// Magic number for `.sparse` file footer ("SPR2" in LE)
pub const SPARSE_FOOTER_MAGIC: u32 = 0x32525053;

/// Per-field TOC entry accumulated during sparse build/merge.
pub struct SparseFieldToc {
    pub field_id: u32,
    pub quantization: u8,
    /// (dim_id, data_offset, data_length) per dimension
    pub dims: Vec<(u32, u64, u32)>,
}

/// Write sparse TOC + footer to writer.
///
/// Called after all posting data has been written. `toc_offset` is the
/// current file position (byte offset where the TOC starts).
pub fn write_sparse_toc_and_footer(
    writer: &mut (impl std::io::Write + ?Sized),
    toc_offset: u64,
    field_tocs: &[SparseFieldToc],
) -> std::io::Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};

    for ftoc in field_tocs {
        writer.write_u32::<LittleEndian>(ftoc.field_id)?;
        writer.write_u8(ftoc.quantization)?;
        writer.write_u32::<LittleEndian>(ftoc.dims.len() as u32)?;
        for &(dim_id, offset, length) in &ftoc.dims {
            writer.write_u32::<LittleEndian>(dim_id)?;
            writer.write_u64::<LittleEndian>(offset)?;
            writer.write_u32::<LittleEndian>(length)?;
        }
    }
    writer.write_u64::<LittleEndian>(toc_offset)?;
    writer.write_u32::<LittleEndian>(field_tocs.len() as u32)?;
    writer.write_u32::<LittleEndian>(SPARSE_FOOTER_MAGIC)?;
    Ok(())
}
