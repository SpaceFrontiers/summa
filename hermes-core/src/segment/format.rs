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
