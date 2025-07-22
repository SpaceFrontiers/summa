//! BMP (Block-Max Pruning) index reader for sparse vectors — **V18 zero-copy**.
//!
//! V18 uses fixed `dims` (vocabulary size) and dim_id directly in per-block data.
//! Grid is indexed by dim_id as row index (no Section C dim_ids array).
//! Data-first layout: block data (Section B) appears before block_data_starts
//! (Section A). The reader derives the Section A offset from
//! `grid_offset - (num_blocks + 1) * 8`.
//!
//! Block-interleaved format: all data needed to score one block is contiguous
//! (~200-2000 bytes, fits in 1-2 pages). Reduces cold-query page faults to 1.
//!
//! At load time the entire blob is acquired as a single `OwnedBytes` (mmap-backed
//! or Arc-Vec) and sliced into sections. No heap allocation — all data including
//! the superblock grid is mmap-backed.
//!
//! Uses **compact virtual coordinates**: sequential IDs assigned to unique
//! `(doc_id, ordinal)` pairs. A doc_map lookup table maps virtual IDs back
//! to original coordinates at query time.
//!
//! Based on Mallia, Suel & Tonellotto (SIGIR 2024).

use crate::directories::{FileHandle, OwnedBytes};
use crate::segment::bmp_grid::CompressedGrid;

/// Number of BMP blocks grouped into one LSP/0 superblock.
///
/// Carlson et al. recommend `block_size × blocks_per_superblock <= 256`.
/// Hermes keeps the requested 32-vector blocks, so eight blocks form one
/// 256-vector superblock. Eight divides the 256-cell compressed-grid group,
/// ensuring a selected superblock never crosses a codec group.
pub const BMP_SUPERBLOCK_SIZE: u32 = 8;

/// Number of LSP/0 superblocks summarized by one cell in the coarse grid.
///
/// This deliberately matches the compressed-grid addressing group. Expanding
/// one promising coarse cell therefore reads one independently addressable
/// 256-superblock group from E for each query dimension.
pub const BMP_COARSE_SUPERBLOCKS: u32 = 256;

/// A single posting in BMP's block-local flat inverted index.
///
/// Exactly 2 bytes: `[local_slot: u8, impact: u8]`.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BmpPosting {
    pub local_slot: u8,
    pub impact: u8,
}

// ── u32 read helpers ─────────────────────────────────────────────────────────

/// Read a little-endian u32 from a raw pointer at element index.
/// No bounds check — used in the hot scoring loop where bounds are
/// validated once at the method boundary via debug_assert.
///
/// Uses `read_unaligned` for portability (handles any alignment).
/// On x86/ARM this compiles to a single `ldr`/`mov` instruction.
///
/// # Safety
/// Caller must ensure `base.add(idx * 4 + 3)` is within the allocation.
#[inline(always)]
unsafe fn read_u32_unchecked(base: *const u8, idx: usize) -> u32 {
    unsafe {
        let p = base.add(idx * 4);
        u32::from_le((p as *const u32).read_unaligned())
    }
}

/// Read a little-endian u64 from a raw pointer at element index.
/// No bounds check — used in the hot scoring loop for block_data_starts.
///
/// # Safety
/// Caller must ensure `base.add(idx * 8 + 7)` is within the allocation.
#[inline(always)]
unsafe fn read_u64_unchecked(base: *const u8, idx: usize) -> u64 {
    unsafe {
        let p = base.add(idx * 8);
        u64::from_le((p as *const u64).read_unaligned())
    }
}

/// BMP V18 index for a single sparse field — fully zero-copy mmap-backed.
///
/// V18 format with Recursive Graph Bisection (BP) document ordering.
///
/// All data sections are `OwnedBytes` slices into the same underlying mmap Arc.
/// No heap allocation — the superblock grid is persisted on disk and loaded as
/// a zero-copy OwnedBytes slice.
///
/// Uses a three-level pruning hierarchy:
/// 1. **Coarse grid**: upper bounds over groups of `BMP_COARSE_SUPERBLOCKS`
///    superblocks, used to find the exact global top-gamma without sweeping E
/// 2. **Superblock grid**: upper bounds over `BMP_SUPERBLOCK_SIZE` blocks
/// 3. **Block grid**: fine-grained upper bounds per individual block
#[derive(Clone)]
pub struct BmpIndex {
    /// BMP block size (number of consecutive virtual_ids per block)
    pub bmp_block_size: u32,
    /// Number of blocks
    pub num_blocks: u32,
    /// Number of compact virtual documents (= num_blocks × bmp_block_size, padded)
    pub num_virtual_docs: u32,
    /// Global max weight scale factor (for dequantizing u8 impacts back to f32)
    pub max_weight_scale: f32,
    /// Total sparse vectors (from TOC entry)
    pub total_vectors: u32,
    /// Number of documents in the containing segment. Document-map entries
    /// must be either padding or strictly below this bound.
    segment_num_docs: u32,

    // ── Section metadata ──────────────────────────────────────────────
    /// Fixed vocabulary size — grid has `dims` rows
    dims: u32,
    total_terms: u64,
    total_postings: u64,
    /// Bits per block-grid cell (4 or 2); dequant scale is 17 or 85.
    grid_bits: u8,
    /// Actual vector count before padding
    num_real_docs: u32,
    /// True when every stored vector is ordinal zero. This is derived from
    /// the physical document map rather than trusting schema declarations.
    single_valued: bool,

    // ── Zero-copy OwnedBytes sections (keeps backing store alive) ────
    /// Section A: block_data_starts[block_id] = byte offset into block_data_bytes
    block_data_starts_bytes: OwnedBytes,
    /// Section B: interleaved per-block data (all scoring data contiguous per block)
    block_data_bytes: OwnedBytes,
    /// Locally bit-packed block maxima. Stored values retain their configured
    /// ceil-u4/u2 semantics exactly.
    block_grid: CompressedGrid,
    /// Locally bit-packed ceil-u4 superblock maxima.
    superblock_grid: CompressedGrid,
    /// Number of superblocks
    pub num_superblocks: u32,
    /// Locally bit-packed ceil-u4 maxima over 256-superblock groups.
    coarse_grid: CompressedGrid,
    /// Number of coarse superblock groups.
    pub num_coarse_groups: u32,
    /// doc_map_ids[virtual_id] = original doc_id — zero-copy OwnedBytes
    doc_map_ids_bytes: OwnedBytes,
    /// doc_map_ordinals[virtual_id] = original ordinal — zero-copy OwnedBytes
    doc_map_ordinals_bytes: OwnedBytes,

    // ── Raw blob source (identity copies) ─────────────────────────────
    /// Source file handle + blob range, kept so reorder can copy the blob
    /// byte-identically for fields whose `reorder` schema attribute is unset.
    /// Reorder is native-only, so these are dead on wasm.
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    source: FileHandle,
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    blob_offset: u64,
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    blob_len: u64,
    /// Offset of Section F within the blob. Retained so local block-copy
    /// merges can pass byte-identical document-map ranges directly to
    /// `copy_file_range` without faulting their mmap pages into userspace.
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    doc_map_offset: u64,
}

// SAFETY: All raw pointer access is derived from OwnedBytes which are Send+Sync
// (backed by Arc<Vec<u8>> or Arc<Mmap>). The pointers are never mutated.
// BmpIndex already stores OwnedBytes (which is Send+Sync), so the struct
// inherits Send+Sync automatically through its fields.

impl BmpIndex {
    /// Parse a BMP V18 blob from the given file handle.
    ///
    /// Reads the footer, then acquires the entire blob as a single
    /// `OwnedBytes` and slices it into zero-copy sections.
    ///
    /// V18 data-first layout: Section B (per-block interleaved data) first,
    /// then Section A (block_data_starts with u64 entries), grids, doc_map.
    pub fn parse(
        handle: FileHandle,
        blob_offset: u64,
        blob_len: u64,
        total_docs: u32,
        total_vectors: u32,
    ) -> crate::Result<Self> {
        use crate::segment::format::{BMP_BLOB_FOOTER_SIZE, BMP_BLOB_MAGIC};

        if blob_len < BMP_BLOB_FOOTER_SIZE as u64 {
            return Err(crate::Error::Corruption(
                "BMP blob too small for V18 footer".into(),
            ));
        }

        // Read the footer.
        let blob_end = blob_offset
            .checked_add(blob_len)
            .ok_or_else(|| crate::Error::Corruption("BMP blob range overflows u64".into()))?;
        let footer_start = blob_end - BMP_BLOB_FOOTER_SIZE as u64;
        let footer_bytes = handle
            .read_bytes_range_sync(footer_start..blob_end)
            .map_err(crate::Error::Io)?;
        let fb = footer_bytes.as_slice();

        let total_terms = u64::from_le_bytes(fb[0..8].try_into().unwrap());
        let total_postings = u64::from_le_bytes(fb[8..16].try_into().unwrap());
        let grid_offset = u64::from_le_bytes(fb[16..24].try_into().unwrap());
        let sb_grid_offset = u64::from_le_bytes(fb[24..32].try_into().unwrap());
        let coarse_grid_offset = u64::from_le_bytes(fb[32..40].try_into().unwrap());
        let num_blocks = u32::from_le_bytes(fb[40..44].try_into().unwrap());
        let dims = u32::from_le_bytes(fb[44..48].try_into().unwrap());
        let bmp_block_size = u32::from_le_bytes(fb[48..52].try_into().unwrap());
        let num_virtual_docs = u32::from_le_bytes(fb[52..56].try_into().unwrap());
        let max_weight_scale = f32::from_le_bytes(fb[56..60].try_into().unwrap());
        let doc_map_offset = u64::from_le_bytes(fb[60..68].try_into().unwrap());
        let num_real_docs = u32::from_le_bytes(fb[68..72].try_into().unwrap());
        let grid_bits_raw = u32::from_le_bytes(fb[72..76].try_into().unwrap());
        let magic = u32::from_le_bytes(fb[76..80].try_into().unwrap());

        if magic != BMP_BLOB_MAGIC {
            return Err(crate::Error::Corruption(format!(
                "Invalid BMP blob magic: {:#x} (expected BMP8 {:#x}); rebuild \
                 the index with this version.",
                magic, BMP_BLOB_MAGIC
            )));
        }
        let grid_bits: u8 = match grid_bits_raw {
            4 => 4,
            2 => 2,
            other => {
                return Err(crate::Error::Corruption(format!(
                    "Unsupported BMP grid_bits {} (expected 2 or 4) — data too new to read?",
                    other
                )));
            }
        };

        // Handle empty index
        if num_blocks == 0 {
            if num_virtual_docs != 0 || num_real_docs != 0 {
                return Err(crate::Error::Corruption(format!(
                    "empty BMP index has non-zero document counts (virtual={}, real={})",
                    num_virtual_docs, num_real_docs
                )));
            }
            return Ok(Self {
                bmp_block_size,
                num_blocks,
                num_virtual_docs,
                max_weight_scale,
                total_vectors,
                segment_num_docs: total_docs,
                dims,
                total_terms: 0,
                total_postings: 0,
                grid_bits,
                num_real_docs,
                single_valued: true,
                block_data_starts_bytes: OwnedBytes::empty(),
                block_data_bytes: OwnedBytes::empty(),
                block_grid: CompressedGrid::empty(),
                superblock_grid: CompressedGrid::empty(),
                num_superblocks: 0,
                coarse_grid: CompressedGrid::empty(),
                num_coarse_groups: 0,
                doc_map_ids_bytes: OwnedBytes::empty(),
                doc_map_ordinals_bytes: OwnedBytes::empty(),
                source: handle,
                blob_offset,
                blob_len,
                doc_map_offset,
            });
        }

        if !(1..=256).contains(&bmp_block_size) {
            return Err(crate::Error::Corruption(format!(
                "invalid BMP block size {} (expected 1..=256)",
                bmp_block_size
            )));
        }
        let expected_virtual_docs = u64::from(num_blocks) * u64::from(bmp_block_size);
        if expected_virtual_docs != u64::from(num_virtual_docs) {
            return Err(crate::Error::Corruption(format!(
                "BMP block/document mismatch: {} blocks × {} != {} virtual docs",
                num_blocks, bmp_block_size, num_virtual_docs
            )));
        }
        if num_real_docs > num_virtual_docs {
            return Err(crate::Error::Corruption(format!(
                "BMP real document count {} exceeds virtual count {}",
                num_real_docs, num_virtual_docs
            )));
        }
        if !max_weight_scale.is_finite() || max_weight_scale <= 0.0 {
            return Err(crate::Error::Corruption(format!(
                "invalid BMP max-weight scale {}",
                max_weight_scale
            )));
        }

        // Read entire blob (excluding footer) as one OwnedBytes — zero-copy mmap slice
        let data_len = blob_len - BMP_BLOB_FOOTER_SIZE as u64;
        let data_len_usize = usize::try_from(data_len).map_err(|_| {
            crate::Error::Corruption("BMP blob is too large for this platform".into())
        })?;
        let blob = handle
            .read_bytes_range_sync(blob_offset..footer_start)
            .map_err(crate::Error::Io)?;

        // Layout: Section B (block_data) at offset 0, Section A (block_data_starts)
        // immediately before grid. Derive Section A position from grid_offset.
        let num_blocks_usize = num_blocks as usize;
        let section_a_size = num_blocks_usize
            .checked_add(1)
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| {
                crate::Error::Corruption("BMP block-offset table size overflows usize".into())
            })?;
        let grid_start = usize::try_from(grid_offset).map_err(|_| {
            crate::Error::Corruption("BMP grid offset is too large for this platform".into())
        })?;
        let bds_start = grid_start.checked_sub(section_a_size).ok_or_else(|| {
            crate::Error::Corruption(format!(
                "BMP grid offset {} precedes {}-byte block-offset table",
                grid_offset, section_a_size
            ))
        })?;
        if grid_start > data_len_usize {
            return Err(crate::Error::Corruption(format!(
                "BMP grid offset {} exceeds data length {}",
                grid_start, data_len_usize
            )));
        }

        // Section B: block_data [0..bds_start) (includes padding before Section A)
        let block_data_bytes = blob.slice(0..bds_start);
        // Section A: block_data_starts [bds_start..grid_offset)
        let block_data_starts_bytes = blob.slice(bds_start..grid_start);

        // Sections D+E+H: compressed ceil-u4 block, superblock, and coarse
        // grids, then document maps. Their byte lengths are carried
        // by the footer's section offsets; each grid validates its own row
        // table before exposing random group access.
        let num_superblocks = num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE);
        let num_coarse_groups = num_superblocks.div_ceil(BMP_COARSE_SUPERBLOCKS);
        let sb_grid_start = usize::try_from(sb_grid_offset).map_err(|_| {
            crate::Error::Corruption("BMP superblock-grid offset is too large".into())
        })?;
        if sb_grid_start < grid_start || sb_grid_start > data_len_usize {
            return Err(crate::Error::Corruption(format!(
                "BMP section order mismatch: block grid starts at {}, superblock grid at {}, data ends at {}",
                grid_start, sb_grid_start, data_len_usize
            )));
        }
        let coarse_grid_start = usize::try_from(coarse_grid_offset)
            .map_err(|_| crate::Error::Corruption("BMP coarse-grid offset is too large".into()))?;
        if coarse_grid_start < sb_grid_start || coarse_grid_start > data_len_usize {
            return Err(crate::Error::Corruption(format!(
                "BMP section order mismatch: superblock grid starts at {}, coarse grid at {}, data ends at {}",
                sb_grid_start, coarse_grid_start, data_len_usize
            )));
        }

        let dm_start = usize::try_from(doc_map_offset)
            .map_err(|_| crate::Error::Corruption("BMP document-map offset is too large".into()))?;
        if dm_start < coarse_grid_start || dm_start > data_len_usize {
            return Err(crate::Error::Corruption(format!(
                "BMP section order mismatch: coarse grid starts at {}, document map at {}, data ends at {}",
                coarse_grid_start, dm_start, data_len_usize
            )));
        }
        let dm_ids_len = (num_virtual_docs as usize).checked_mul(4).ok_or_else(|| {
            crate::Error::Corruption("BMP document-id map size overflows usize".into())
        })?;
        let dm_ords_len = (num_virtual_docs as usize).checked_mul(2).ok_or_else(|| {
            crate::Error::Corruption("BMP ordinal map size overflows usize".into())
        })?;
        let dm_ids_end = dm_start.checked_add(dm_ids_len).ok_or_else(|| {
            crate::Error::Corruption("BMP document-id map end overflows usize".into())
        })?;
        let dm_ords_end = dm_ids_end.checked_add(dm_ords_len).ok_or_else(|| {
            crate::Error::Corruption("BMP ordinal map end overflows usize".into())
        })?;
        if dm_ords_end != data_len_usize {
            return Err(crate::Error::Corruption(format!(
                "BMP data length mismatch: sections end at {}, blob data ends at {}",
                dm_ords_end, data_len_usize
            )));
        }

        // Slice into sections (all zero-copy — just offset adjustments on same Arc)
        let block_grid = CompressedGrid::parse(
            blob.slice(grid_start..sb_grid_start),
            dims as usize,
            num_blocks as usize,
            grid_bits,
            "BMP block grid",
        )?;
        let superblock_grid = CompressedGrid::parse(
            blob.slice(sb_grid_start..coarse_grid_start),
            dims as usize,
            num_superblocks as usize,
            4,
            "BMP superblock grid",
        )?;
        let coarse_grid = CompressedGrid::parse(
            blob.slice(coarse_grid_start..dm_start),
            dims as usize,
            num_coarse_groups as usize,
            4,
            "BMP coarse grid",
        )?;
        let doc_map_ids_bytes = blob.slice(dm_start..dm_ids_end);
        let doc_map_ordinals_bytes = blob.slice(dm_ids_end..dm_ords_end);
        let single_valued = doc_map_ordinals_bytes
            .as_slice()
            .chunks_exact(2)
            .all(|ordinal| ordinal == [0, 0]);

        // This compact table is cheap to validate in full and is the trust
        // boundary for every later raw-pointer block access.
        let starts = block_data_starts_bytes.as_slice();
        let mut previous = 0u64;
        for index in 0..=num_blocks_usize {
            let offset = index * 8;
            let current = u64::from_le_bytes(starts[offset..offset + 8].try_into().unwrap());
            if (index == 0 && current != 0) || current < previous || current > bds_start as u64 {
                return Err(crate::Error::Corruption(format!(
                    "invalid BMP block offset at {}: {} (previous={}, data_limit={})",
                    index, current, previous, bds_start
                )));
            }
            if current > previous && current - previous < 8 {
                return Err(crate::Error::Corruption(format!(
                    "BMP block {} is too small for a header ({} bytes)",
                    index - 1,
                    current - previous
                )));
            }
            previous = current;
        }

        // Query-time access to block data, the doc map, AND the block grid is
        // scattered. Default kernel readahead pulls in 128KB per fault around
        // each touched location, which evicts hot pages under memory pressure.
        //
        // The block grid especially: queries read one eight-cell range per
        // (query dim, surviving superblock) at UB-priority, i.e. effectively
        // random offsets. Default readahead can amplify each tiny probe into
        // 128KB of page cache and march a data-sized grid into memory.
        //
        // E is now accessed only for selected 256-superblock groups and is
        // random. H is tiny, swept contiguously, and pinnable (priority 4).
        #[cfg(feature = "native")]
        {
            block_data_bytes.madvise(libc::MADV_RANDOM);
            doc_map_ids_bytes.madvise(libc::MADV_RANDOM);
            doc_map_ordinals_bytes.madvise(libc::MADV_RANDOM);
            block_grid.madvise_rows(libc::MADV_RANDOM);
            superblock_grid.madvise_rows(libc::MADV_RANDOM);
            coarse_grid.madvise_rows(libc::MADV_SEQUENTIAL);
        }

        log::debug!(
            "BMP V18 index loaded: num_blocks={}, num_superblocks={}, coarse_groups={}, dims={}, bmp_block_size={}, \
             num_virtual_docs={}, num_real_docs={}, max_weight_scale={:.4}, postings={}, \
             block_grid={}, superblock_grid={}, coarse_grid={}, single_valued={}, block_data={}, doc_map={}",
            num_blocks,
            num_superblocks,
            num_coarse_groups,
            dims,
            bmp_block_size,
            num_virtual_docs,
            num_real_docs,
            max_weight_scale,
            total_postings,
            crate::format_bytes(block_grid.encoded_bytes() as u64),
            crate::format_bytes(superblock_grid.encoded_bytes() as u64),
            crate::format_bytes(coarse_grid.encoded_bytes() as u64),
            single_valued,
            crate::format_bytes(bds_start as u64),
            crate::format_bytes(u64::from(num_virtual_docs) * 6),
        );

        Ok(Self {
            bmp_block_size,
            num_blocks,
            num_virtual_docs,
            max_weight_scale,
            total_vectors,
            segment_num_docs: total_docs,
            dims,
            total_terms,
            total_postings,
            grid_bits,
            num_real_docs,
            single_valued,
            block_data_starts_bytes,
            block_data_bytes,
            block_grid,
            superblock_grid,
            num_superblocks,
            coarse_grid,
            num_coarse_groups,
            doc_map_ids_bytes,
            doc_map_ordinals_bytes,
            source: handle,
            blob_offset,
            blob_len,
            doc_map_offset,
        })
    }

    /// Read the entire raw V18 blob (including footer) from the source file.
    ///
    /// Used by reorder paths (native-only) to copy a field byte-identically
    /// when its `reorder` schema attribute is unset.
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub(crate) fn read_raw_blob(&self) -> std::io::Result<OwnedBytes> {
        self.source
            .read_bytes_range_sync(self.blob_offset..self.blob_offset + self.blob_len)
    }

    /// Convert a compact virtual_id to (doc_id, ordinal) via table lookup.
    ///
    /// Uses unchecked reads — virtual_id is validated by the caller
    /// (only called for top-k results which are valid compact virtual IDs).
    #[inline(always)]
    pub fn virtual_to_doc(&self, virtual_id: u32) -> (u32, u16) {
        if virtual_id >= self.num_virtual_docs {
            return (u32::MAX, 0);
        }
        let ids = self.doc_map_ids_bytes.as_slice();
        let ords = self.doc_map_ordinals_bytes.as_slice();
        debug_assert!((virtual_id as usize + 1) * 4 <= ids.len());
        debug_assert!((virtual_id as usize + 1) * 2 <= ords.len());
        unsafe {
            let doc_id = read_u32_unchecked(ids.as_ptr(), virtual_id as usize);
            if doc_id >= self.segment_num_docs {
                return (u32::MAX, 0);
            }
            let p = ords.as_ptr().add(virtual_id as usize * 2);
            let ordinal = u16::from_le((p as *const u16).read_unaligned());
            (doc_id, ordinal)
        }
    }

    /// Get the original doc_id for a compact virtual_id (no ordinal needed).
    /// Used in the predicate filter path — hot loop, unchecked reads.
    #[inline(always)]
    pub fn doc_id_for_virtual(&self, virtual_id: u32) -> u32 {
        if virtual_id >= self.num_virtual_docs {
            return u32::MAX;
        }
        let d = self.doc_map_ids_bytes.as_slice();
        debug_assert!((virtual_id as usize + 1) * 4 <= d.len());
        let doc_id = unsafe { read_u32_unchecked(d.as_ptr(), virtual_id as usize) };
        if doc_id < self.segment_num_docs {
            doc_id
        } else {
            u32::MAX
        }
    }

    // ── Hot-path block-data accessors ────────────────────────────────

    /// Byte offset range in block_data_bytes for a block (u64 entries).
    #[inline(always)]
    pub(crate) fn block_data_range(&self, block_id: u32) -> (u64, u64) {
        let d = self.block_data_starts_bytes.as_slice();
        debug_assert!((block_id as usize + 2) * 8 <= d.len());
        unsafe {
            let start = read_u64_unchecked(d.as_ptr(), block_id as usize);
            let end = read_u64_unchecked(d.as_ptr(), block_id as usize + 1);
            (start, end)
        }
    }

    /// Pin the block-offset table (priority 1: every scored block does an
    /// offset lookup through it).
    #[cfg(feature = "native")]
    pub(crate) fn pin_block_starts(
        &mut self,
        mode: crate::segment::pin::PinMode,
        remaining: &mut u64,
        report: &mut crate::segment::pin::PinReport,
    ) {
        crate::segment::pin::pin_section(
            &mut self.block_data_starts_bytes,
            "bmp block_data_starts",
            mode,
            remaining,
            report,
        );
        self.block_grid
            .pin_offsets("bmp block_grid row_offsets", mode, remaining, report);
    }

    /// Pin the virtual-doc → (doc_id, ordinal) maps (priority 3: every
    /// top-k resolution touches them).
    #[cfg(feature = "native")]
    pub(crate) fn pin_doc_maps(
        &mut self,
        mode: crate::segment::pin::PinMode,
        remaining: &mut u64,
        report: &mut crate::segment::pin::PinReport,
    ) {
        crate::segment::pin::pin_section(
            &mut self.doc_map_ids_bytes,
            "bmp doc_map_ids",
            mode,
            remaining,
            report,
        );
        crate::segment::pin::pin_section(
            &mut self.doc_map_ordinals_bytes,
            "bmp doc_map_ordinals",
            mode,
            remaining,
            report,
        );
    }

    /// Pin the sparse planning hierarchy (priority 4).
    ///
    /// E is data-sized and only accessed for selected coarse groups, so only
    /// its row offsets are pinned. H is roughly 256x smaller and is swept for
    /// every BMP query, so both its offsets and rows are pinned. The block-grid
    /// payload is deliberately never pinned; its row offsets are priority 1.
    #[cfg(feature = "native")]
    pub(crate) fn pin_query_hierarchy(
        &mut self,
        mode: crate::segment::pin::PinMode,
        remaining: &mut u64,
        report: &mut crate::segment::pin::PinReport,
    ) {
        self.superblock_grid
            .pin_offsets("bmp sb_grid row_offsets", mode, remaining, report);
        self.coarse_grid.pin_all(
            "bmp coarse_grid row_offsets",
            "bmp coarse_grid rows",
            mode,
            remaining,
            report,
        );
    }

    /// Page-level prefetch (`MADV_WILLNEED`) of a block-data byte range.
    ///
    /// Used by the BMP executor to batch-prefetch the surviving blocks of a
    /// superblock before scoring: on memory-bound hosts the kernel clusters
    /// the page-ins into large sequential reads instead of taking one
    /// synchronous major fault per scored block (~265µs each on cold NVMe).
    /// No-op for non-mmap (RAM/HTTP) backing.
    #[cfg(feature = "native")]
    #[inline]
    pub(crate) fn prefetch_block_data(&self, byte_start: u64, byte_end: u64) {
        self.block_data_bytes
            .madvise_range(byte_start as usize..byte_end as usize, libc::MADV_WILLNEED);
    }

    /// Coalesce page-near block payload ranges before issuing WILLNEED.
    ///
    /// Selected LSP superblocks are score-ordered rather than file-ordered, so
    /// one giant min..max advice span can pull gigabytes of unvisited data.
    /// This keeps distant extents independent while collapsing ranges that the
    /// kernel would round onto the same/adjacent pages anyway.
    #[cfg(feature = "native")]
    pub(crate) fn prefetch_block_data_ranges(
        &self,
        ranges: &mut Vec<std::ops::Range<u64>>,
    ) -> (usize, usize) {
        if ranges.is_empty() {
            return (0, 0);
        }
        const PAGE_NEAR_BYTES: u64 = 4096;
        ranges.sort_unstable_by_key(|range| (range.start, range.end));
        let mut advised_bytes = 0usize;
        let mut calls = 0usize;
        let mut current = ranges[0].clone();
        for range in &ranges[1..] {
            if range.start <= current.end.saturating_add(PAGE_NEAR_BYTES) {
                current.end = current.end.max(range.end);
                continue;
            }
            advised_bytes = advised_bytes.saturating_add((current.end - current.start) as usize);
            calls += 1;
            self.prefetch_block_data(current.start, current.end);
            current = range.clone();
        }
        advised_bytes = advised_bytes.saturating_add((current.end - current.start) as usize);
        calls += 1;
        self.prefetch_block_data(current.start, current.end);
        ranges.clear();
        (advised_bytes, calls)
    }

    /// Get a raw pointer to the start of a block's contiguous data.
    /// Used for software prefetching — 1 prefetch loads all block scoring data.
    #[inline(always)]
    pub(crate) fn block_data_ptr(&self, block_id: u32) -> *const u8 {
        let (start, _) = self.block_data_range(block_id);
        unsafe {
            self.block_data_bytes
                .as_slice()
                .as_ptr()
                .add(start as usize)
        }
    }

    /// Parse a block header: returns
    /// (num_terms, dim_ptr, ps_ptr, max_ptr, post_ptr, total_block_postings).
    /// All pointers are within block_data_bytes — guaranteed contiguous.
    ///
    /// Always 4-byte (u32) dim IDs.
    ///
    /// Returns zero terms and null section pointers for empty blocks.
    #[inline(always)]
    pub(crate) fn parse_block(
        &self,
        block_id: u32,
    ) -> (u32, *const u8, *const u8, *const u8, *const u8, u32) {
        if block_id >= self.num_blocks {
            return (
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
            );
        }
        let (start, end) = self.block_data_range(block_id);
        if start == end {
            return (
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
            );
        }
        let invalid = || {
            (
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
            )
        };
        let Ok(block_len) = usize::try_from(end - start) else {
            return invalid();
        };
        if block_len < 8 {
            return invalid();
        }
        let base = unsafe {
            self.block_data_bytes
                .as_slice()
                .as_ptr()
                .add(start as usize)
        };
        let num_terms = unsafe { u32::from_le((base as *const u32).read_unaligned()) };
        let Some(header_len) = (num_terms as usize)
            .checked_mul(9)
            .and_then(|bytes| bytes.checked_add(8))
        else {
            return invalid();
        };
        if header_len > block_len || !(block_len - header_len).is_multiple_of(2) {
            return invalid();
        }
        let total_block_postings = (block_len - header_len) / 2;
        let Ok(total_block_postings_u32) = u32::try_from(total_block_postings) else {
            return invalid();
        };
        let dim_ptr = unsafe { base.add(4) };
        // Always u32 dim IDs (4 bytes each)
        let ps_ptr = unsafe { dim_ptr.add(num_terms as usize * 4) };
        let max_ptr = unsafe { ps_ptr.add((num_terms as usize + 1) * 4) };
        let post_ptr = unsafe { max_ptr.add(num_terms as usize) };
        let first = unsafe { u32::from_le((ps_ptr as *const u32).read_unaligned()) };
        let last = unsafe {
            u32::from_le((ps_ptr.add(num_terms as usize * 4) as *const u32).read_unaligned())
        };
        if first != 0 || last != total_block_postings_u32 {
            return invalid();
        }
        (
            num_terms,
            dim_ptr,
            ps_ptr,
            max_ptr,
            post_ptr,
            total_block_postings_u32,
        )
    }

    /// Get a raw pointer to block_data_starts at the given block.
    /// Used for prefetching the N+2 block's offset during scoring.
    /// Each entry is 8 bytes (u64).
    #[inline(always)]
    pub(crate) fn block_data_starts_ptr(&self, block_id: u32) -> *const u8 {
        unsafe {
            self.block_data_starts_bytes
                .as_slice()
                .as_ptr()
                .add(block_id as usize * 8)
        }
    }

    /// Iterate terms in a block (for merger). Returns (dim_id, &[BmpPosting]) per term.
    ///
    /// Reads u32 dim_id directly from block data (no dim_idx→dim_id lookup).
    pub fn iter_block_terms(&self, block_id: u32) -> BlockTermIter<'_> {
        let (num_terms, dim_ptr, ps_ptr, max_ptr, post_ptr, total_postings) =
            self.parse_block(block_id);
        BlockTermIter {
            dim_ptr,
            ps_ptr,
            max_ptr,
            post_ptr,
            num_terms,
            total_postings,
            current: 0,
            _marker: std::marker::PhantomData,
        }
    }

    // ── Non-hot-path accessors ───────────────────────────────────────

    /// Fixed vocabulary size (number of grid rows).
    pub fn dims(&self) -> u32 {
        self.dims
    }

    /// Validate the persisted layout before a merge or reorder interprets it
    /// using schema-derived output parameters.
    ///
    /// The footer is the source of truth for reading this blob. Rewriting with
    /// a different block width, grid width, vocabulary, or impact scale would
    /// otherwise make block slicing or copied upper bounds invalid.
    #[cfg(any(feature = "native", test))]
    pub(crate) fn validate_rewrite_layout(
        &self,
        context: &str,
        expected_dims: u32,
        expected_block_size: u32,
        expected_grid_bits: u8,
        expected_max_weight_scale: f32,
    ) -> crate::Result<()> {
        if expected_dims == 0 {
            return Err(crate::Error::Corruption(format!(
                "{context}: expected vocabulary is empty",
            )));
        }
        if self.dims != expected_dims {
            return Err(crate::Error::Corruption(format!(
                "{context}: source dims={} != expected {expected_dims}",
                self.dims,
            )));
        }
        if self.bmp_block_size != expected_block_size {
            return Err(crate::Error::Corruption(format!(
                "{context}: source block_size={} != expected {expected_block_size}",
                self.bmp_block_size,
            )));
        }
        if self.grid_bits != expected_grid_bits {
            return Err(crate::Error::Corruption(format!(
                "{context}: source grid_bits={} != expected {expected_grid_bits}",
                self.grid_bits,
            )));
        }
        if !expected_max_weight_scale.is_finite() || expected_max_weight_scale <= 0.0 {
            return Err(crate::Error::Corruption(format!(
                "{context}: invalid expected max_weight_scale={expected_max_weight_scale}",
            )));
        }
        if self.max_weight_scale.to_bits() != expected_max_weight_scale.to_bits() {
            return Err(crate::Error::Corruption(format!(
                "{context}: source max_weight_scale={:.4} != expected {:.4}",
                self.max_weight_scale, expected_max_weight_scale,
            )));
        }
        Ok(())
    }

    /// Validate the document map and visit each non-padding virtual slot.
    ///
    /// All rewrite paths share this scan so block-copy cannot offset corrupt
    /// source IDs into another segment while record reorder rejects them.
    #[cfg(any(feature = "native", feature = "wasm", test))]
    pub(crate) fn visit_real_slots_for_rewrite(
        &self,
        mut visitor: impl FnMut(usize),
    ) -> crate::Result<()> {
        let expected_real = self.num_real_docs as usize;
        let mut real_slots = 0usize;
        for (virtual_id, chunk) in self
            .doc_map_ids_bytes
            .as_slice()
            .chunks_exact(4)
            .enumerate()
        {
            let doc_id = u32::from_le_bytes(chunk.try_into().unwrap());
            if doc_id == u32::MAX {
                continue;
            }
            if doc_id >= self.segment_num_docs {
                return Err(crate::Error::Corruption(format!(
                    "BMP document map contains doc id {doc_id} outside segment bound {}",
                    self.segment_num_docs,
                )));
            }
            if real_slots == expected_real {
                return Err(crate::Error::Corruption(format!(
                    "BMP document map contains more than the footer's {expected_real} real slots"
                )));
            }
            visitor(virtual_id);
            real_slots += 1;
        }
        if real_slots != expected_real {
            return Err(crate::Error::Corruption(format!(
                "BMP document map has {real_slots} real slots but footer declares {expected_real}",
            )));
        }
        Ok(())
    }

    /// Validate one block before a rewrite feeds it into infallible hot-path
    /// iterators. Query parsing deliberately degrades malformed blocks to
    /// empty for availability; a rewrite must instead fail loudly so it never
    /// publishes silent data loss or indexes an invalid local slot.
    #[cfg(any(feature = "native", test))]
    pub(crate) fn validate_block_for_rewrite(&self, block_id: u32) -> crate::Result<()> {
        if block_id >= self.num_blocks {
            return Err(crate::Error::Corruption(format!(
                "BMP rewrite block {block_id} exceeds block count {}",
                self.num_blocks,
            )));
        }
        let (start, end) = self.block_data_range(block_id);
        let start = usize::try_from(start)
            .map_err(|_| crate::Error::Corruption("BMP block start exceeds usize".into()))?;
        let end = usize::try_from(end)
            .map_err(|_| crate::Error::Corruption("BMP block end exceeds usize".into()))?;
        let block = self
            .block_data_bytes
            .as_slice()
            .get(start..end)
            .ok_or_else(|| {
                crate::Error::Corruption(format!(
                    "BMP block {block_id} range {start}..{end} exceeds block data",
                ))
            })?;
        if block.is_empty() {
            return Ok(());
        }
        if block.len() < 8 {
            return Err(crate::Error::Corruption(format!(
                "BMP block {block_id} is too short: {} bytes",
                block.len(),
            )));
        }

        let num_terms = u32::from_le_bytes(block[..4].try_into().unwrap()) as usize;
        let header_len = num_terms
            .checked_mul(9)
            .and_then(|bytes| bytes.checked_add(8))
            .ok_or_else(|| {
                crate::Error::Corruption(format!(
                    "BMP block {block_id} header length overflows usize",
                ))
            })?;
        if header_len > block.len() || !(block.len() - header_len).is_multiple_of(2) {
            return Err(crate::Error::Corruption(format!(
                "BMP block {block_id} has invalid header/data lengths: header={header_len}, total={}",
                block.len(),
            )));
        }

        let dims_start = 4;
        let prefixes_start = dims_start + num_terms * 4;
        let maxima_start = prefixes_start + (num_terms + 1) * 4;
        let postings_start = maxima_start + num_terms;
        debug_assert_eq!(postings_start, header_len);
        let total_postings = (block.len() - postings_start) / 2;
        let read_prefix = |index: usize| {
            let offset = prefixes_start + index * 4;
            u32::from_le_bytes(block[offset..offset + 4].try_into().unwrap()) as usize
        };
        if read_prefix(0) != 0 || read_prefix(num_terms) != total_postings {
            return Err(crate::Error::Corruption(format!(
                "BMP block {block_id} posting prefixes do not span 0..{total_postings}",
            )));
        }

        let mut previous_dim = None;
        for term in 0..num_terms {
            let dim_offset = dims_start + term * 4;
            let dimension =
                u32::from_le_bytes(block[dim_offset..dim_offset + 4].try_into().unwrap());
            if dimension >= self.dims || previous_dim.is_some_and(|previous| dimension <= previous)
            {
                return Err(crate::Error::Corruption(format!(
                    "BMP block {block_id} has invalid/non-increasing dimension {dimension} at term {term}",
                )));
            }
            previous_dim = Some(dimension);

            let posting_start = read_prefix(term);
            let posting_end = read_prefix(term + 1);
            if posting_start >= posting_end || posting_end > total_postings {
                return Err(crate::Error::Corruption(format!(
                    "BMP block {block_id} term {term} has invalid posting range {posting_start}..{posting_end}",
                )));
            }
            let postings =
                &block[postings_start + posting_start * 2..postings_start + posting_end * 2];
            let mut observed_max = 0u8;
            for posting in postings.chunks_exact(2) {
                if u32::from(posting[0]) >= self.bmp_block_size {
                    return Err(crate::Error::Corruption(format!(
                        "BMP block {block_id} term {term} has local slot {} outside block size {}",
                        posting[0], self.bmp_block_size,
                    )));
                }
                observed_max = observed_max.max(posting[1]);
            }
            let stored_max = block[maxima_start + term];
            if stored_max != observed_max {
                return Err(crate::Error::Corruption(format!(
                    "BMP block {block_id} term {term} maximum {stored_max} != observed {observed_max}",
                )));
            }
        }
        Ok(())
    }

    /// Total number of terms (unique dim×block pairs) stored in the index.
    pub fn total_terms(&self) -> u64 {
        self.total_terms
    }

    /// Total number of postings stored in the index.
    pub fn total_postings(&self) -> u64 {
        self.total_postings
    }

    /// Actual vector count before block-alignment padding.
    pub fn num_real_docs(&self) -> u32 {
        self.num_real_docs
    }

    /// Number of documents in the containing segment.
    /// Whether this segment physically contains at most one vector per
    /// document. Unlike the schema's `multi` flag, this remains reliable for
    /// old or externally-created segments with inaccurate metadata.
    pub fn is_single_valued(&self) -> bool {
        self.single_valued
    }

    /// Estimated heap retained by this index. All corpus-sized sections are
    /// file-backed `OwnedBytes` slices and therefore excluded.
    pub fn estimated_heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    /// Bits per block-grid cell (4 or 2).
    pub fn grid_bits(&self) -> u8 {
        self.grid_bits
    }

    /// Direct random-group access to the compressed block grid.
    #[inline]
    pub(crate) fn block_grid(&self) -> &CompressedGrid {
        &self.block_grid
    }

    /// Direct random-group access to the compressed ceil-u4 superblock grid.
    #[inline]
    pub(crate) fn superblock_grid(&self) -> &CompressedGrid {
        &self.superblock_grid
    }

    /// Direct access to the ceil-u4 grid over 256-superblock groups.
    #[inline]
    pub(crate) fn coarse_grid(&self) -> &CompressedGrid {
        &self.coarse_grid
    }

    /// Visit independently decoded chunks of one block-grid row.
    ///
    /// This is intended for diagnostics such as the CLI heatmap. `None`
    /// represents an all-zero chunk and avoids materializing it; non-zero
    /// values are valid only for the duration of the callback.
    pub fn for_each_block_grid_chunk(
        &self,
        dimension: u32,
        mut visitor: impl FnMut(usize, usize, Option<&[u8]>),
    ) -> crate::Result<()> {
        let dimension = dimension as usize;
        if dimension >= self.block_grid.dims() {
            return Err(crate::Error::Query(format!(
                "BMP block-grid dimension {dimension} exceeds {}",
                self.block_grid.dims()
            )));
        }
        let mut decoded = [0u8; crate::segment::bmp_grid::GRID_GROUP_CELLS];
        self.block_grid
            .try_for_each_row_group(dimension, |group_id, group| {
                let start = group_id * crate::segment::bmp_grid::GRID_GROUP_CELLS;
                let count =
                    crate::segment::bmp_grid::GRID_GROUP_CELLS.min(self.block_grid.cells() - start);
                if group.width() == 0 {
                    visitor(start, count, None);
                } else {
                    group.decode(0, count, &mut decoded);
                    visitor(start, count, Some(&decoded[..count]));
                }
                Ok(())
            })
    }

    // ── Streaming merge accessors (block-copy) ────────────────────────

    /// Raw block data bytes (Section B). For block-copy merge.
    #[inline]
    pub fn block_data_slice(&self) -> &[u8] {
        self.block_data_bytes.as_slice()
    }

    /// Byte offset of block `block_id` in block data (from block_data_starts).
    #[inline]
    pub fn block_data_start(&self, block_id: u32) -> u64 {
        let d = self.block_data_starts_bytes.as_slice();
        let off = block_id as usize * 8;
        u64::from_le_bytes(d[off..off + 8].try_into().unwrap())
    }

    /// Sentinel value = total bytes in Section B (block_data_starts[num_blocks]).
    #[inline]
    pub fn block_data_sentinel(&self) -> u64 {
        self.block_data_start(self.num_blocks)
    }

    /// Raw doc_map_ids bytes (Section F). For bulk merge copy.
    /// Layout: `[u32-LE × num_virtual_docs]`.
    #[inline]
    pub fn doc_map_ids_slice(&self) -> &[u8] {
        self.doc_map_ids_bytes.as_slice()
    }

    /// Raw doc_map_ordinals bytes (Section G). For bulk merge copy.
    /// Layout: `[u16-LE × num_virtual_docs]`.
    #[inline]
    pub fn doc_map_ordinals_slice(&self) -> &[u8] {
        self.doc_map_ordinals_bytes.as_slice()
    }

    /// Native source-file range containing Section B (block payload).
    #[cfg(feature = "native")]
    pub(crate) fn block_data_file_range(&self) -> std::ops::Range<u64> {
        self.blob_offset..self.blob_offset + self.block_data_sentinel()
    }

    /// Native source-file range containing Section F (document IDs).
    #[cfg(feature = "native")]
    pub(crate) fn doc_map_ids_file_range(&self) -> std::ops::Range<u64> {
        let start = self.blob_offset + self.doc_map_offset;
        start..start + u64::from(self.num_virtual_docs) * 4
    }

    /// Native source-file range containing Section G (ordinals).
    #[cfg(feature = "native")]
    pub(crate) fn doc_map_ordinals_file_range(&self) -> std::ops::Range<u64> {
        let start = self.blob_offset + self.doc_map_offset + u64::from(self.num_virtual_docs) * 4;
        start..start + u64::from(self.num_virtual_docs) * 2
    }

    /// Advise the kernel about sequential access patterns for merge.
    ///
    /// Only effective on mmap-backed data. No-op for heap (Vec) or non-native.
    #[cfg(feature = "native")]
    pub fn madvise_sequential(&self) {
        Self::madvise_owned(&self.block_data_bytes, libc::MADV_SEQUENTIAL);
        Self::madvise_owned(&self.block_data_starts_bytes, libc::MADV_SEQUENTIAL);
        self.block_grid.madvise_rows(libc::MADV_SEQUENTIAL);
        self.superblock_grid.madvise_rows(libc::MADV_SEQUENTIAL);
        self.coarse_grid.madvise_rows(libc::MADV_SEQUENTIAL);
        Self::madvise_owned(&self.doc_map_ids_bytes, libc::MADV_SEQUENTIAL);
        Self::madvise_owned(&self.doc_map_ordinals_bytes, libc::MADV_SEQUENTIAL);
    }

    /// Release block data pages after Phase 1 completes.
    /// Keeps block_data_starts — needed for Phase 2 recomputation.
    #[cfg(feature = "native")]
    pub fn madvise_dontneed_block_data(&self) {
        Self::madvise_owned(&self.block_data_bytes, libc::MADV_DONTNEED);
    }

    /// Restore query-pattern advice (same as set at `parse`) after a merge
    /// flipped these regions to `MADV_SEQUENTIAL`. Source segments keep
    /// serving queries while and after being merged, until swapped out.
    #[cfg(feature = "native")]
    pub fn madvise_random_query(&self) {
        Self::madvise_owned(&self.block_data_bytes, libc::MADV_RANDOM);
        self.block_grid.madvise_rows(libc::MADV_RANDOM);
        self.superblock_grid.madvise_rows(libc::MADV_RANDOM);
        self.coarse_grid.madvise_rows(libc::MADV_SEQUENTIAL);
        Self::madvise_owned(&self.doc_map_ids_bytes, libc::MADV_RANDOM);
        Self::madvise_owned(&self.doc_map_ordinals_bytes, libc::MADV_RANDOM);
    }

    /// Release grid pages after Phase 3+4 complete.
    #[cfg(feature = "native")]
    pub fn madvise_dontneed_grids(&self) {
        self.block_grid.madvise_rows(libc::MADV_DONTNEED);
        self.superblock_grid.madvise_rows(libc::MADV_DONTNEED);
        self.coarse_grid.madvise_rows(libc::MADV_DONTNEED);
    }

    /// Release document-map pages faulted by a full reorder scan. They remain
    /// mmap-backed and refault on demand for any reader that still references
    /// the source segment during publication.
    #[cfg(feature = "native")]
    pub fn madvise_dontneed_doc_maps(&self) {
        Self::madvise_owned(&self.doc_map_ids_bytes, libc::MADV_DONTNEED);
        Self::madvise_owned(&self.doc_map_ordinals_bytes, libc::MADV_DONTNEED);
    }

    /// Call `madvise` only when the backing store is mmap.
    ///
    /// `MADV_DONTNEED` on heap (Vec) memory zeroes pages on Linux and can
    /// corrupt allocator metadata (the page-aligned pointer may reach into
    /// malloc headers before the allocation). This caused `free(): invalid
    /// pointer` crashes in CI where tests use RamDirectory (Vec-backed).
    #[cfg(feature = "native")]
    fn madvise_owned(bytes: &crate::directories::OwnedBytes, advice: i32) {
        bytes.madvise(advice);
    }
}

/// Kernel page-advice lifecycle for exhaustive background scans.
///
/// Construction marks source mappings sequential. Drop releases the large
/// block/grid/doc-map regions and restores random query advice, including on
/// `?` and panic unwind. The mappings stay valid and refault for readers that
/// still reference a source during publication.
#[cfg(feature = "native")]
pub(crate) struct BmpScanPageGuard<'a> {
    indexes: Vec<&'a BmpIndex>,
}

#[cfg(feature = "native")]
impl<'a> BmpScanPageGuard<'a> {
    pub(crate) fn new(indexes: impl IntoIterator<Item = &'a BmpIndex>) -> Self {
        let indexes: Vec<_> = indexes.into_iter().collect();
        for index in &indexes {
            index.madvise_sequential();
        }
        Self { indexes }
    }

    pub(crate) fn switch_to_random(&self) {
        for index in &self.indexes {
            index.madvise_random_query();
        }
    }
}

#[cfg(feature = "native")]
impl Drop for BmpScanPageGuard<'_> {
    fn drop(&mut self) {
        for index in &self.indexes {
            index.madvise_dontneed_block_data();
            index.madvise_dontneed_grids();
            index.madvise_dontneed_doc_maps();
            index.madvise_random_query();
        }
    }
}

/// Iterator over terms in a block. Returns `(dim_id, &[BmpPosting])` per term.
///
/// Reads u32 dim_id directly from block data (no dim_idx→dim_id lookup).
pub struct BlockTermIter<'a> {
    dim_ptr: *const u8,
    ps_ptr: *const u8,
    max_ptr: *const u8,
    post_ptr: *const u8,
    num_terms: u32,
    total_postings: u32,
    current: u32,
    // lifetime marker for the underlying BmpIndex data
    _marker: std::marker::PhantomData<&'a ()>,
}

// Manually implement Send+Sync. The raw pointers are derived from OwnedBytes
// (which is Send+Sync) and are never mutated. The iterator borrows &BmpIndex.
unsafe impl<'a> Send for BlockTermIter<'a> {}
unsafe impl<'a> Sync for BlockTermIter<'a> {}

impl<'a> Iterator for BlockTermIter<'a> {
    type Item = (u32, u8, &'a [BmpPosting]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.num_terms {
            return None;
        }
        let i = self.current;
        self.current += 1;

        // Read u32 dim_id directly from block data
        let dim_id = unsafe { read_u32_unchecked(self.dim_ptr, i as usize) };
        let max_impact = unsafe { *self.max_ptr.add(i as usize) };

        // Get postings from block-local ps_ptr/post_ptr
        let postings =
            unsafe { block_term_postings(self.ps_ptr, self.post_ptr, i, self.total_postings) };
        Some((dim_id, max_impact, postings))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = (self.num_terms - self.current) as usize;
        (rem, Some(rem))
    }
}

impl<'a> ExactSizeIterator for BlockTermIter<'a> {}

// ============================================================================
// Block-data free functions (used by query and merger)
// ============================================================================

/// Binary search for a dimension ID in a block's term_dim_ids.
///
/// Always u32 dim_ids. `dim_ptr` points to the block's term_dim_ids array.
/// Returns the local term index (0..num_terms) if found.
///
/// # Safety
/// `dim_ptr` must be valid for `num_terms * 4` bytes.
#[inline(always)]
pub(crate) fn find_dim_in_block_data(
    dim_ptr: *const u8,
    num_terms: u32,
    dim_id: u32,
) -> Option<u32> {
    let count = num_terms as usize;
    if count == 0 {
        return None;
    }

    let mut lo = 0usize;
    let mut hi = count;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let val = unsafe { read_u32_unchecked(dim_ptr, mid) };
        match val.cmp(&dim_id) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Equal => return Some(mid as u32),
            std::cmp::Ordering::Greater => hi = mid,
        }
    }
    None
}

/// Get postings for a local term index within a parsed block.
///
/// `ps_ptr` points to the block's posting_starts array [u32 × (num_terms + 1)].
/// `post_ptr` points to the block's postings array [(u8, u8) × total].
///
/// # Safety
/// Pointers must be valid and derived from a BmpIndex's block_data_bytes.
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn block_term_postings<'a>(
    ps_ptr: *const u8,
    post_ptr: *const u8,
    local_term: u32,
    total_block_postings: u32,
) -> &'a [BmpPosting] {
    let start_p = ps_ptr.add(local_term as usize * 4);
    let end_p = ps_ptr.add((local_term as usize + 1) * 4);
    let start = u32::from_le((start_p as *const u32).read_unaligned()) as usize;
    let end = u32::from_le((end_p as *const u32).read_unaligned()) as usize;
    // Prefix sums are cumulative — a non-monotonic pair means the block is
    // corrupt. Never build a wild slice from it (`end - start` can underflow).
    if end <= start || end > total_block_postings as usize {
        return &[];
    }
    let count = end - start;
    // SAFETY: BmpPosting is #[repr(C)] with align=1 (two u8 fields).
    let ptr = post_ptr.add(start * 2) as *const BmpPosting;
    std::slice::from_raw_parts(ptr, count)
}

#[cfg(test)]
mod safety_tests {
    use super::BmpIndex;
    use crate::directories::{FileHandle, OwnedBytes};
    use crate::segment::format::BMP_BLOB_FOOTER_SIZE;
    use rustc_hash::FxHashMap;

    fn test_blob() -> Vec<u8> {
        let mut postings = FxHashMap::default();
        postings.insert(3, vec![(0, 0, 1.0), (1, 0, 0.5)]);
        let mut blob = Vec::new();
        crate::segment::builder::bmp::build_bmp_blob(
            postings, 64, 4, 0.0, None, 16, 5.0, 0, &mut blob,
        )
        .unwrap();
        blob
    }

    fn parse(blob: Vec<u8>) -> crate::Result<BmpIndex> {
        let len = blob.len() as u64;
        BmpIndex::parse(FileHandle::from_bytes(OwnedBytes::new(blob)), 0, len, 2, 2)
    }

    #[test]
    fn parse_rejects_footer_section_underflow_without_panicking() {
        let mut blob = test_blob();
        let footer = blob.len() - BMP_BLOB_FOOTER_SIZE;
        blob[footer + 16..footer + 24].copy_from_slice(&0u64.to_le_bytes());
        assert!(matches!(parse(blob), Err(crate::Error::Corruption(_))));
    }

    #[test]
    fn parse_rejects_nonzero_first_block_offset() {
        let mut blob = test_blob();
        let footer = blob.len() - BMP_BLOB_FOOTER_SIZE;
        let grid_offset =
            u64::from_le_bytes(blob[footer + 16..footer + 24].try_into().unwrap()) as usize;
        let num_blocks =
            u32::from_le_bytes(blob[footer + 40..footer + 44].try_into().unwrap()) as usize;
        let starts = grid_offset - (num_blocks + 1) * 8;
        blob[starts..starts + 8].copy_from_slice(&1u64.to_le_bytes());
        assert!(matches!(parse(blob), Err(crate::Error::Corruption(_))));
    }

    #[test]
    fn physical_single_value_detection_uses_ordinal_map() {
        let single = parse(test_blob()).unwrap();
        assert!(single.is_single_valued());

        let mut postings = FxHashMap::default();
        postings.insert(3, vec![(0, 0, 1.0), (0, 1, 0.8), (1, 0, 0.5)]);
        let mut blob = Vec::new();
        crate::segment::builder::bmp::build_bmp_blob(
            postings, 64, 4, 0.0, None, 16, 5.0, 0, &mut blob,
        )
        .unwrap();
        let multi = parse(blob).unwrap();
        assert!(!multi.is_single_valued());
    }

    #[test]
    fn rewrite_validation_rejects_out_of_range_local_slot() {
        let mut blob = test_blob();
        // One-term block header is 8 + 9 bytes; postings follow immediately.
        blob[17] = 64;
        let index = parse(blob).unwrap();
        let error = index.validate_block_for_rewrite(0).unwrap_err();
        assert!(matches!(error, crate::Error::Corruption(_)));
    }

    #[test]
    fn rewrite_validation_rejects_bad_dimension_and_maximum() {
        let mut bad_dimension = test_blob();
        bad_dimension[4..8].copy_from_slice(&16u32.to_le_bytes());
        let index = parse(bad_dimension).unwrap();
        assert!(matches!(
            index.validate_block_for_rewrite(0),
            Err(crate::Error::Corruption(_))
        ));

        let mut bad_maximum = test_blob();
        bad_maximum[16] = 0;
        let index = parse(bad_maximum).unwrap();
        assert!(matches!(
            index.validate_block_for_rewrite(0),
            Err(crate::Error::Corruption(_))
        ));
    }

    #[test]
    fn invalid_doc_map_id_is_bounded_and_rewrite_rejects_it() {
        let mut blob = test_blob();
        let footer = blob.len() - BMP_BLOB_FOOTER_SIZE;
        let doc_map =
            u64::from_le_bytes(blob[footer + 60..footer + 68].try_into().unwrap()) as usize;
        blob[doc_map..doc_map + 4].copy_from_slice(&2u32.to_le_bytes());
        let index = parse(blob).unwrap();

        assert_eq!(index.doc_id_for_virtual(0), u32::MAX);
        assert!(matches!(
            crate::segment::builder::graph_bisection::build_vid_maps(&index),
            Err(crate::Error::Corruption(_))
        ));
    }

    #[test]
    fn rewrite_layout_requires_exact_finite_scale() {
        let index = parse(test_blob()).unwrap();
        let adjacent_scale = f32::from_bits(index.max_weight_scale.to_bits() + 1);
        assert!(matches!(
            index.validate_rewrite_layout("test", 16, 64, 4, adjacent_scale),
            Err(crate::Error::Corruption(_))
        ));
        assert!(matches!(
            index.validate_rewrite_layout("test", 16, 64, 4, f32::NAN),
            Err(crate::Error::Corruption(_))
        ));
    }
}
