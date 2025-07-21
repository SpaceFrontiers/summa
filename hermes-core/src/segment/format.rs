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

/// Magic number for the current BMP V18 blob footer ("BMP8" in LE).
///
/// V18 uses LSP/0's 256-vector superblocks and ceil-u4 maximum weights at all
/// pruning levels. In addition to the block and superblock grids, it persists
/// an exact coarse grid over groups of 256 superblocks. The coarse grid lets
/// query planning find the global top-gamma superblocks without sweeping the
/// data-sized superblock grid.
///
/// Every grid retains independently addressable, locally bit-packed 256-cell
/// groups and exact per-term block-header maxima.
/// This is the only accepted BMP representation; indexes are rebuilt on format
/// changes rather than carrying compatibility parsers.
pub const BMP_BLOB_MAGIC: u32 = 0x38504D42;

/// Current BMP V18 blob footer size.
pub const BMP_BLOB_FOOTER_SIZE: usize = 80;

/// V3 footer size: skip_offset(8) + toc_offset(8) + num_fields(4) + magic(4) = 24
pub const SPARSE_FOOTER_SIZE: u64 = 24;

/// Per-dim TOC entry accumulated during V3 sparse build/merge.
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
pub struct SparseFieldToc {
    pub field_id: u32,
    pub quantization: u8,
    /// Total sparse vectors in this field. Multi-valued fields can have more
    /// vectors than segment documents.
    pub total_vectors: u32,
    pub dims: Vec<SparseDimTocEntry>,
}

impl SparseFieldToc {
    /// Build the sentinel TOC entry used by the self-contained BMP blob.
    pub(crate) fn bmp(field_id: u32, total_vectors: u32, blob_offset: u64, blob_len: u64) -> Self {
        // V18 always stores u32 dimensions and u8 impacts regardless of the
        // generic sparse schema knobs. Persist the physical representation,
        // not a misleading caller-supplied MaxScore descriptor.
        let config = crate::structures::SparseVectorConfig {
            format: crate::structures::SparseFormat::Bmp,
            index_size: crate::structures::IndexSize::U32,
            weight_quantization: crate::structures::WeightQuantization::UInt8,
            ..Default::default()
        };

        Self {
            field_id,
            quantization: config.to_byte(),
            total_vectors,
            dims: vec![SparseDimTocEntry {
                dim_id: u32::MAX,
                block_data_offset: blob_offset,
                skip_start: blob_len as u32,
                num_blocks: (blob_len >> 32) as u32,
                doc_count: 0,
                max_weight: 0.0,
            }],
        }
    }
}

/// Write V3 sparse TOC + footer.
///
/// V3 layout:
/// ```text
/// [block data ...]
/// [skip section: SparseSkipEntry × total_skips (24B each)]
/// [TOC: per-field header(13B) + per-dim entries(28B each)]
/// [footer: skip_offset(8) + toc_offset(8) + num_fields(4) + magic(4)]
/// ```
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

#[cfg(test)]
mod tests {
    use super::SparseFieldToc;
    use crate::structures::{IndexSize, SparseFormat, SparseVectorConfig, WeightQuantization};

    #[test]
    fn bmp_toc_describes_the_physical_v18_encoding() {
        let blob_len = u64::from(u32::MAX) + 17;
        let toc = SparseFieldToc::bmp(7, 11, 23, blob_len);
        let config = SparseVectorConfig::from_byte(toc.quantization).unwrap();

        assert_eq!(config.format, SparseFormat::Bmp);
        assert_eq!(config.index_size, IndexSize::U32);
        assert_eq!(config.weight_quantization, WeightQuantization::UInt8);
        assert_eq!(toc.total_vectors, 11);
        assert_eq!(toc.dims[0].dim_id, u32::MAX);
        assert_eq!(toc.dims[0].block_data_offset, 23);
        assert_eq!(toc.dims[0].skip_start, 16);
        assert_eq!(toc.dims[0].num_blocks, 1);
    }
}
