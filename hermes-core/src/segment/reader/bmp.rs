//! BMP (Block-Max Pruning) index reader for sparse vectors — **V15 zero-copy**.
//!
//! V15 uses fixed `dims` (vocabulary size) and dim_id directly in per-block data.
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

/// Number of BMP blocks grouped into one superblock for hierarchical pruning.
///
/// Based on Carlson et al. (SIGIR 2025): "Dynamic Superblock Pruning for Fast
/// Learned Sparse Retrieval". Superblocks precompute per-superblock upper bounds,
/// enabling entire groups of blocks to be pruned before computing block-level UBs.
///
/// At 1M×5 ordinals (78K blocks), this reduces UB computation from 78K entries
/// to ~1.2K superblock entries, pruning 25-75% of blocks before detailed scoring.
pub const BMP_SUPERBLOCK_SIZE: u32 = 64;

/// A single posting in the BMP forward index: (local_slot, impact).
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

/// BMP V15 index for a single sparse field — fully zero-copy mmap-backed.
///
/// V15 format with Recursive Graph Bisection (BP) document ordering.
///
/// All data sections are `OwnedBytes` slices into the same underlying mmap Arc.
/// No heap allocation — the superblock grid is persisted on disk and loaded as
/// a zero-copy OwnedBytes slice.
///
/// Uses two-level pruning hierarchy (Carlson et al., SIGIR 2025):
/// 1. **Superblock grid**: coarse upper bounds over groups of `BMP_SUPERBLOCK_SIZE` blocks
/// 2. **Block grid**: fine-grained upper bounds per individual block (4-bit packed)
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

    // ── Section metadata ──────────────────────────────────────────────
    /// Fixed vocabulary size — grid has `dims` rows
    dims: u32,
    total_terms: u64,
    total_postings: u64,
    /// Packed row size for 4-bit grid: `(num_blocks + 1) / 2`
    packed_row_size: u32,
    /// Bits per block-grid cell (4 or 2); dequant scale is 17 or 85.
    grid_bits: u8,
    /// Actual vector count before padding
    num_real_docs: u32,
    /// True when every stored vector is ordinal zero. This is derived from
    /// the physical document map rather than the schema so legacy segments
    /// whose `multi` flag was inaccurate still get correct query planning.
    single_valued: bool,

    // ── Zero-copy OwnedBytes sections (keeps backing store alive) ────
    /// Section A: block_data_starts[block_id] = byte offset into block_data_bytes
    block_data_starts_bytes: OwnedBytes,
    /// Section B: interleaved per-block data (all scoring data contiguous per block)
    block_data_bytes: OwnedBytes,
    /// 4-bit packed block grid: `grid[dim_id * packed_row_size + block_id/2]`
    grid_bytes: OwnedBytes,
    /// sb_grid[dim_id * num_superblocks + sb_id] = max impact across all blocks in superblock
    sb_grid_bytes: OwnedBytes,
    /// Number of superblocks
    pub num_superblocks: u32,
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
}

// SAFETY: All raw pointer access is derived from OwnedBytes which are Send+Sync
// (backed by Arc<Vec<u8>> or Arc<Mmap>). The pointers are never mutated.
// BmpIndex already stores OwnedBytes (which is Send+Sync), so the struct
// inherits Send+Sync automatically through its fields.

impl BmpIndex {
    /// Parse a BMP V15 blob from the given file handle.
    ///
    /// Reads the footer, then acquires the entire blob as a single
    /// `OwnedBytes` and slices it into zero-copy sections.
    ///
    /// V15 data-first layout: Section B (per-block interleaved data) first,
    /// then Section A (block_data_starts with u64 entries), grids, doc_map.
    pub fn parse(
        handle: FileHandle,
        blob_offset: u64,
        blob_len: u64,
        _total_docs: u32,
        total_vectors: u32,
    ) -> crate::Result<Self> {
        use crate::segment::format::{BMP_BLOB_FOOTER_SIZE, BMP_BLOB_MAGIC};

        if blob_len < BMP_BLOB_FOOTER_SIZE as u64 {
            return Err(crate::Error::Corruption(
                "BMP blob too small for V15 footer".into(),
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
        let num_blocks = u32::from_le_bytes(fb[32..36].try_into().unwrap());
        let dims = u32::from_le_bytes(fb[36..40].try_into().unwrap());
        let bmp_block_size = u32::from_le_bytes(fb[40..44].try_into().unwrap());
        let num_virtual_docs = u32::from_le_bytes(fb[44..48].try_into().unwrap());
        let max_weight_scale = f32::from_le_bytes(fb[48..52].try_into().unwrap());
        let doc_map_offset = u64::from_le_bytes(fb[52..60].try_into().unwrap());
        let num_real_docs = u32::from_le_bytes(fb[60..64].try_into().unwrap());
        let grid_bits_raw = u32::from_le_bytes(fb[64..68].try_into().unwrap());
        let magic = u32::from_le_bytes(fb[68..72].try_into().unwrap());

        if magic != BMP_BLOB_MAGIC {
            return Err(crate::Error::Corruption(format!(
                "Invalid BMP blob magic: {:#x} (expected BMP5 {:#x}); rebuild \
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
                dims,
                total_terms: 0,
                total_postings: 0,
                packed_row_size: 0,
                grid_bits,
                num_real_docs,
                single_valued: true,
                block_data_starts_bytes: OwnedBytes::empty(),
                block_data_bytes: OwnedBytes::empty(),
                grid_bytes: OwnedBytes::empty(),
                sb_grid_bytes: OwnedBytes::empty(),
                num_superblocks: 0,
                doc_map_ids_bytes: OwnedBytes::empty(),
                doc_map_ordinals_bytes: OwnedBytes::empty(),
                source: handle,
                blob_offset,
                blob_len,
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

        // Sections D+E: grid, sb_grid, doc_map
        let packed_row_size =
            crate::segment::builder::bmp::grid_packed_row_size(num_blocks as usize, grid_bits)
                as u32;
        let grid_size = (dims as usize)
            .checked_mul(packed_row_size as usize)
            .ok_or_else(|| crate::Error::Corruption("BMP grid size overflows usize".into()))?;
        let grid_end = grid_start.checked_add(grid_size).ok_or_else(|| {
            crate::Error::Corruption("BMP grid end offset overflows usize".into())
        })?;

        let num_superblocks = num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE);
        let sb_grid_start = usize::try_from(sb_grid_offset).map_err(|_| {
            crate::Error::Corruption("BMP superblock-grid offset is too large".into())
        })?;
        if sb_grid_start != grid_end {
            return Err(crate::Error::Corruption(format!(
                "BMP section order mismatch: superblock grid starts at {}, expected {}",
                sb_grid_start, grid_end
            )));
        }
        let sb_grid_size = (dims as usize)
            .checked_mul(num_superblocks as usize)
            .ok_or_else(|| {
                crate::Error::Corruption("BMP superblock-grid size overflows usize".into())
            })?;
        let sb_grid_end = sb_grid_start.checked_add(sb_grid_size).ok_or_else(|| {
            crate::Error::Corruption("BMP superblock-grid end overflows usize".into())
        })?;

        let dm_start = usize::try_from(doc_map_offset)
            .map_err(|_| crate::Error::Corruption("BMP document-map offset is too large".into()))?;
        if dm_start != sb_grid_end {
            return Err(crate::Error::Corruption(format!(
                "BMP section order mismatch: document map starts at {}, expected {}",
                dm_start, sb_grid_end
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
        let grid_bytes = blob.slice(grid_start..grid_end);
        let sb_grid_bytes = blob.slice(sb_grid_start..sb_grid_end);
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
        // The grid especially: at production scale queries take the direct
        // (non-compact) path and read 32 bytes per (query dim, surviving
        // superblock) at UB-priority — i.e. effectively random — offsets
        // within each ~100KB+ row. Default readahead amplifies each 32-byte
        // probe into 128KB of page cache (~4000×), marching the entire
        // data-sized grid into memory and OOMing cgroup-limited deployments.
        // The compact path (whole-row copies) only runs when a query's rows
        // total ≤128KB, where losing readahead costs a couple of pages.
        //
        // sb_grid keeps default readahead: its rows are swept contiguously
        // per query dim, and it is pinnable (priority 4).
        #[cfg(feature = "native")]
        {
            block_data_bytes.madvise(libc::MADV_RANDOM);
            doc_map_ids_bytes.madvise(libc::MADV_RANDOM);
            doc_map_ordinals_bytes.madvise(libc::MADV_RANDOM);
            grid_bytes.madvise(libc::MADV_RANDOM);
        }

        log::debug!(
            "BMP V15 index loaded: num_blocks={}, num_superblocks={}, dims={}, bmp_block_size={}, \
             num_virtual_docs={}, num_real_docs={}, max_weight_scale={:.4}, postings={}, \
             packed_row_size={}, single_valued={}, block_data={}B, doc_map={}B",
            num_blocks,
            num_superblocks,
            dims,
            bmp_block_size,
            num_virtual_docs,
            num_real_docs,
            max_weight_scale,
            total_postings,
            packed_row_size,
            single_valued,
            bds_start,
            num_virtual_docs as usize * 6,
        );

        Ok(Self {
            bmp_block_size,
            num_blocks,
            num_virtual_docs,
            max_weight_scale,
            total_vectors,
            dims,
            total_terms,
            total_postings,
            packed_row_size,
            grid_bits,
            num_real_docs,
            single_valued,
            block_data_starts_bytes,
            block_data_bytes,
            grid_bytes,
            sb_grid_bytes,
            num_superblocks,
            doc_map_ids_bytes,
            doc_map_ordinals_bytes,
            source: handle,
            blob_offset,
            blob_len,
        })
    }

    /// Read the entire raw V15 blob (including footer) from the source file.
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
        unsafe { read_u32_unchecked(d.as_ptr(), virtual_id as usize) }
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

    /// Pin the superblock grid (priority 4: every query dim reads a row).
    /// The 4-bit block grid is deliberately never pinned (data-sized).
    #[cfg(feature = "native")]
    pub(crate) fn pin_sb_grid(
        &mut self,
        mode: crate::segment::pin::PinMode,
        remaining: &mut u64,
        report: &mut crate::segment::pin::PinReport,
    ) {
        crate::segment::pin::pin_section(
            &mut self.sb_grid_bytes,
            "bmp sb_grid",
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
    /// (num_terms, dim_ptr, ps_ptr, post_ptr, total_block_postings).
    /// All pointers are within block_data_bytes — guaranteed contiguous.
    ///
    /// Always 4-byte (u32) dim IDs.
    ///
    /// Returns zero terms and null section pointers for empty blocks.
    #[inline(always)]
    pub(crate) fn parse_block(&self, block_id: u32) -> (u32, *const u8, *const u8, *const u8, u32) {
        if block_id >= self.num_blocks {
            return (0, std::ptr::null(), std::ptr::null(), std::ptr::null(), 0);
        }
        let (start, end) = self.block_data_range(block_id);
        if start == end {
            return (0, std::ptr::null(), std::ptr::null(), std::ptr::null(), 0);
        }
        let invalid = || (0, std::ptr::null(), std::ptr::null(), std::ptr::null(), 0);
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
            .checked_mul(8)
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
        // u32 prefix sums (V14): u16 wrapped past 65,535 postings per block
        let post_ptr = unsafe { ps_ptr.add((num_terms as usize + 1) * 4) };
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
        let (num_terms, dim_ptr, ps_ptr, post_ptr, total_postings) = self.parse_block(block_id);
        BlockTermIter {
            dim_ptr,
            ps_ptr,
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

    /// Whether this segment physically contains at most one vector per
    /// document. Unlike the schema's `multi` flag, this remains reliable for
    /// old or externally-created segments with inaccurate metadata.
    pub fn is_single_valued(&self) -> bool {
        self.single_valued
    }

    /// Estimated memory usage in bytes (mmap-backed region sizes).
    ///
    /// Fully zero-copy: all data is mmap-backed OwnedBytes, but the
    /// mapped regions still consume RSS when paged in by the OS.
    pub fn estimated_memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.block_data_starts_bytes.len()
            + self.block_data_bytes.len()
            + self.grid_bytes.len()
            + self.sb_grid_bytes.len()
            + self.doc_map_ids_bytes.len()
            + self.doc_map_ordinals_bytes.len()
    }

    /// Extract compact grid data for query-relevant dims into caller-provided buffers.
    ///
    /// `dim_indices` are dim_id values (used directly as grid row indices).
    ///
    /// Copies only the rows corresponding to `dim_indices`, creating a contiguous
    /// layout that fits in L1/L2 cache. For ~20 query dims with 1500 blocks:
    /// sb_grid ~480B + grid ~15KB = ~16KB — comfortably in L1 (32-64KB).
    ///
    /// After extraction, local dim index `i` maps to `compact_sb_grid[i * nsb..]`
    /// and `compact_grid[i * prs..]`.
    pub(crate) fn extract_compact_grids(
        &self,
        dim_indices: &[usize],
        compact_sb_grid: &mut Vec<u8>,
        compact_grid: &mut Vec<u8>,
    ) {
        let nsb = self.num_superblocks as usize;
        let prs = self.packed_row_size as usize;
        let nqd = dim_indices.len();

        compact_sb_grid.resize(nqd * nsb, 0);
        compact_grid.resize(nqd * prs, 0);

        let sb_grid = self.sb_grid_bytes.as_slice();
        let grid = self.grid_bytes.as_slice();

        for (local, &dim_idx) in dim_indices.iter().enumerate() {
            compact_sb_grid[local * nsb..(local + 1) * nsb]
                .copy_from_slice(&sb_grid[dim_idx * nsb..(dim_idx + 1) * nsb]);
            compact_grid[local * prs..(local + 1) * prs]
                .copy_from_slice(&grid[dim_idx * prs..(dim_idx + 1) * prs]);
        }
    }

    /// Packed row size (bytes per dim row in 4-bit grid).
    #[inline]
    pub fn packed_row_size(&self) -> usize {
        self.packed_row_size as usize
    }

    /// Bits per block-grid cell (4 or 2).
    pub fn grid_bits(&self) -> u8 {
        self.grid_bits
    }

    /// Direct access to mmap-backed superblock grid (zero-copy, zero allocation).
    /// Used for large segments where compact grid extraction would be too expensive.
    #[inline]
    pub(crate) fn sb_grid_slice(&self) -> &[u8] {
        self.sb_grid_bytes.as_slice()
    }

    /// Direct access to mmap-backed block grid (zero-copy, zero allocation).
    #[inline]
    pub fn grid_slice(&self) -> &[u8] {
        self.grid_bytes.as_slice()
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

    /// Advise the kernel about sequential access patterns for merge.
    ///
    /// Only effective on mmap-backed data. No-op for heap (Vec) or non-native.
    #[cfg(feature = "native")]
    pub fn madvise_sequential(&self) {
        Self::madvise_owned(&self.block_data_bytes, libc::MADV_SEQUENTIAL);
        Self::madvise_owned(&self.block_data_starts_bytes, libc::MADV_SEQUENTIAL);
        Self::madvise_owned(&self.grid_bytes, libc::MADV_SEQUENTIAL);
        Self::madvise_owned(&self.sb_grid_bytes, libc::MADV_SEQUENTIAL);
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
        Self::madvise_owned(&self.grid_bytes, libc::MADV_RANDOM);
        Self::madvise_owned(&self.sb_grid_bytes, libc::MADV_RANDOM);
        Self::madvise_owned(&self.doc_map_ids_bytes, libc::MADV_RANDOM);
        Self::madvise_owned(&self.doc_map_ordinals_bytes, libc::MADV_RANDOM);
    }

    /// Release grid pages after Phase 3+4 complete.
    #[cfg(feature = "native")]
    pub fn madvise_dontneed_grids(&self) {
        Self::madvise_owned(&self.grid_bytes, libc::MADV_DONTNEED);
        Self::madvise_owned(&self.sb_grid_bytes, libc::MADV_DONTNEED);
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
    type Item = (u32, &'a [BmpPosting]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.num_terms {
            return None;
        }
        let i = self.current;
        self.current += 1;

        // Read u32 dim_id directly from block data
        let dim_id = unsafe { read_u32_unchecked(self.dim_ptr, i as usize) };

        // Get postings from block-local ps_ptr/post_ptr
        let postings =
            unsafe { block_term_postings(self.ps_ptr, self.post_ptr, i, self.total_postings) };
        Some((dim_id, postings))
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
    // corrupt. Never build a wild slice from it (`end - start` underflow on
    // wrapped V13 data was a production SIGSEGV).
    if end <= start || end > total_block_postings as usize {
        return &[];
    }
    let count = end - start;
    // SAFETY: BmpPosting is #[repr(C)] with align=1 (two u8 fields).
    let ptr = post_ptr.add(start * 2) as *const BmpPosting;
    std::slice::from_raw_parts(ptr, count)
}

// ============================================================================
// SIMD-accelerated helpers for BMP scoring
// ============================================================================
// Packed grid accumulation
// ============================================================================
/// Add one packed grid row to exact integer block-bound accumulators.
///
/// The query quantizer proves that the sum over all rows fits `u32`. Keeping
/// this loop in integer units makes the final f32 conversion monotonic with
/// document scoring. Rows are unpacked a byte at a time to avoid per-cell
/// division and repeated packed-byte loads.
#[inline]
pub(crate) fn accumulate_grid_u32(
    packed: &[u8],
    grid_bits: u8,
    elem_offset: usize,
    count: usize,
    weight: u32,
    out: &mut [u32],
) {
    debug_assert!(out.len() >= count);
    if grid_bits == 2 {
        let scaled_weight = weight * 85;
        let mut index = 0usize;
        while index < count && !(elem_offset + index).is_multiple_of(4) {
            let absolute = elem_offset + index;
            let cell =
                (unsafe { *packed.get_unchecked(absolute / 4) } >> ((absolute % 4) * 2)) & 0x03;
            unsafe { *out.get_unchecked_mut(index) += u32::from(cell) * scaled_weight };
            index += 1;
        }
        while index + 4 <= count {
            let byte = unsafe { *packed.get_unchecked((elem_offset + index) / 4) };
            unsafe {
                *out.get_unchecked_mut(index) += u32::from(byte & 0x03) * scaled_weight;
                *out.get_unchecked_mut(index + 1) += u32::from((byte >> 2) & 0x03) * scaled_weight;
                *out.get_unchecked_mut(index + 2) += u32::from((byte >> 4) & 0x03) * scaled_weight;
                *out.get_unchecked_mut(index + 3) += u32::from(byte >> 6) * scaled_weight;
            }
            index += 4;
        }
        while index < count {
            let absolute = elem_offset + index;
            let cell =
                (unsafe { *packed.get_unchecked(absolute / 4) } >> ((absolute % 4) * 2)) & 0x03;
            unsafe { *out.get_unchecked_mut(index) += u32::from(cell) * scaled_weight };
            index += 1;
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if elem_offset.is_multiple_of(2) {
        unsafe { accumulate_u4_u32_neon(packed, elem_offset, count, weight, out) };
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if elem_offset.is_multiple_of(2) && is_x86_feature_detected!("sse4.1") {
        unsafe { accumulate_u4_u32_sse41(packed, elem_offset, count, weight, out) };
        return;
    }

    let scaled_weight = weight * 17;
    let mut index = 0usize;
    if !elem_offset.is_multiple_of(2) && count > 0 {
        let byte = unsafe { *packed.get_unchecked(elem_offset / 2) };
        unsafe { *out.get_unchecked_mut(0) += u32::from(byte >> 4) * scaled_weight };
        index = 1;
    }
    while index + 2 <= count {
        let byte = unsafe { *packed.get_unchecked((elem_offset + index) / 2) };
        unsafe {
            *out.get_unchecked_mut(index) += u32::from(byte & 0x0f) * scaled_weight;
            *out.get_unchecked_mut(index + 1) += u32::from(byte >> 4) * scaled_weight;
        }
        index += 2;
    }
    if index < count {
        let byte = unsafe { *packed.get_unchecked((elem_offset + index) / 2) };
        unsafe { *out.get_unchecked_mut(index) += u32::from(byte & 0x0f) * scaled_weight };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn accumulate_u4_u32_neon(
    packed: &[u8],
    elem_offset: usize,
    count: usize,
    weight: u32,
    out: &mut [u32],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0f);
    let scale = vdupq_n_u8(17);
    let packed_ptr = packed.as_ptr().add(elem_offset / 2);
    let out_ptr = out.as_mut_ptr();
    let chunks = count / 32;

    for chunk in 0..chunks {
        let bytes = vld1q_u8(packed_ptr.add(chunk * 16));
        let low = vmulq_u8(vandq_u8(bytes, mask), scale);
        let high = vmulq_u8(vshrq_n_u8::<4>(bytes), scale);
        let first = vzip1q_u8(low, high);
        let second = vzip2q_u8(low, high);
        let output = out_ptr.add(chunk * 32);

        let first_low = vmovl_u8(vget_low_u8(first));
        let first_high = vmovl_u8(vget_high_u8(first));
        let second_low = vmovl_u8(vget_low_u8(second));
        let second_high = vmovl_u8(vget_high_u8(second));
        let vectors = [
            vmovl_u16(vget_low_u16(first_low)),
            vmovl_u16(vget_high_u16(first_low)),
            vmovl_u16(vget_low_u16(first_high)),
            vmovl_u16(vget_high_u16(first_high)),
            vmovl_u16(vget_low_u16(second_low)),
            vmovl_u16(vget_high_u16(second_low)),
            vmovl_u16(vget_low_u16(second_high)),
            vmovl_u16(vget_high_u16(second_high)),
        ];
        for (vector_index, values) in vectors.into_iter().enumerate() {
            let destination = output.add(vector_index * 4);
            let accumulated = vld1q_u32(destination);
            vst1q_u32(destination, vmlaq_n_u32(accumulated, values, weight));
        }
    }

    let base = chunks * 32;
    let scaled_weight = weight * 17;
    for index in base..count {
        let absolute = elem_offset + index;
        let byte = *packed.get_unchecked(absolute / 2);
        let cell = if absolute.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        };
        *out.get_unchecked_mut(index) += u32::from(cell) * scaled_weight;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn accumulate_u4_u32_sse41(
    packed: &[u8],
    elem_offset: usize,
    count: usize,
    weight: u32,
    out: &mut [u32],
) {
    use std::arch::x86_64::*;

    let mask = _mm_set1_epi8(0x0f);
    let zero = _mm_setzero_si128();
    let weight_vector = _mm_set1_epi32(weight as i32);
    let packed_ptr = packed.as_ptr().add(elem_offset / 2);
    let out_ptr = out.as_mut_ptr();
    let chunks = count / 32;

    for chunk in 0..chunks {
        let bytes = _mm_loadu_si128(packed_ptr.add(chunk * 16) as *const __m128i);
        let low = _mm_and_si128(bytes, mask);
        let high = _mm_and_si128(_mm_srli_epi16::<4>(bytes), mask);
        let low = _mm_add_epi8(_mm_slli_epi16::<4>(low), low);
        let high = _mm_add_epi8(_mm_slli_epi16::<4>(high), high);
        let first = _mm_unpacklo_epi8(low, high);
        let second = _mm_unpackhi_epi8(low, high);
        let output = out_ptr.add(chunk * 32);

        let first_low = _mm_unpacklo_epi8(first, zero);
        let first_high = _mm_unpackhi_epi8(first, zero);
        let second_low = _mm_unpacklo_epi8(second, zero);
        let second_high = _mm_unpackhi_epi8(second, zero);
        let vectors = [
            _mm_unpacklo_epi16(first_low, zero),
            _mm_unpackhi_epi16(first_low, zero),
            _mm_unpacklo_epi16(first_high, zero),
            _mm_unpackhi_epi16(first_high, zero),
            _mm_unpacklo_epi16(second_low, zero),
            _mm_unpackhi_epi16(second_low, zero),
            _mm_unpacklo_epi16(second_high, zero),
            _mm_unpackhi_epi16(second_high, zero),
        ];
        for (vector_index, values) in vectors.into_iter().enumerate() {
            let destination = output.add(vector_index * 4) as *mut __m128i;
            let accumulated = _mm_loadu_si128(destination);
            let contribution = _mm_mullo_epi32(values, weight_vector);
            _mm_storeu_si128(destination, _mm_add_epi32(accumulated, contribution));
        }
    }

    let base = chunks * 32;
    let scaled_weight = weight * 17;
    for index in base..count {
        let absolute = elem_offset + index;
        let byte = *packed.get_unchecked(absolute / 2);
        let cell = if absolute.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        };
        *out.get_unchecked_mut(index) += u32::from(cell) * scaled_weight;
    }
}

/// Legacy f32 SIMD kernel retained only by the grid-layout microbenchmark.
/// Production pruning accumulates integer bound units in `query::bmp`; f32
/// term-by-term addition can round below the matching integer document score.
#[cfg(test)]
pub(crate) fn accumulate_u4_weighted(
    packed: &[u8],
    elem_offset: usize,
    count: usize,
    weight: f32,
    out: &mut [f32],
) {
    if count == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if elem_offset.is_multiple_of(2) {
            unsafe { accumulate_u4_weighted_neon(packed, elem_offset, count, weight, out) };
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if elem_offset.is_multiple_of(2) && is_x86_feature_detected!("sse4.1") {
            unsafe { accumulate_u4_weighted_sse41(packed, elem_offset, count, weight, out) };
            return;
        }
    }

    // Scalar fallback (also used for odd elem_offset)
    for i in 0..count {
        let abs_idx = elem_offset + i;
        let byte_val = unsafe { *packed.get_unchecked(abs_idx / 2) };
        let val = if abs_idx.is_multiple_of(2) {
            byte_val & 0x0F
        } else {
            byte_val >> 4
        };
        unsafe {
            *out.get_unchecked_mut(i) += (val as u32 * 17) as f32 * weight;
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn accumulate_u4_weighted_neon(
    packed: &[u8],
    elem_offset: usize,
    count: usize,
    weight: f32,
    out: &mut [f32],
) {
    use std::arch::aarch64::*;

    debug_assert!(elem_offset.is_multiple_of(2));

    let weight_v = vdupq_n_f32(weight);
    let mask_lo = vdupq_n_u8(0x0F);
    let scale17 = vdupq_n_u8(17);

    let byte_offset = elem_offset / 2;
    let packed_ptr = packed.as_ptr().add(byte_offset);
    let out_ptr = out.as_mut_ptr();

    let chunks = count / 32;
    let remainder = count % 32;

    for chunk in 0..chunks {
        let pb = packed_ptr.add(chunk * 16);
        let ob = out_ptr.add(chunk * 32);

        let bytes = vld1q_u8(pb);

        let low = vandq_u8(bytes, mask_lo);
        let high = vshrq_n_u8::<4>(bytes);

        let low_scaled = vmulq_u8(low, scale17);
        let high_scaled = vmulq_u8(high, scale17);

        let elems_0_15 = vzip1q_u8(low_scaled, high_scaled);
        let elems_16_31 = vzip2q_u8(low_scaled, high_scaled);

        {
            let lo8 = vget_low_u8(elems_0_15);
            let hi8 = vget_high_u8(elems_0_15);
            let lo16 = vmovl_u8(lo8);
            let hi16 = vmovl_u8(hi8);

            let u32_0 = vmovl_u16(vget_low_u16(lo16));
            let f32_0 = vcvtq_f32_u32(u32_0);
            let acc_0 = vld1q_f32(ob);
            vst1q_f32(ob, vfmaq_f32(acc_0, f32_0, weight_v));

            let u32_1 = vmovl_u16(vget_high_u16(lo16));
            let f32_1 = vcvtq_f32_u32(u32_1);
            let acc_1 = vld1q_f32(ob.add(4));
            vst1q_f32(ob.add(4), vfmaq_f32(acc_1, f32_1, weight_v));

            let u32_2 = vmovl_u16(vget_low_u16(hi16));
            let f32_2 = vcvtq_f32_u32(u32_2);
            let acc_2 = vld1q_f32(ob.add(8));
            vst1q_f32(ob.add(8), vfmaq_f32(acc_2, f32_2, weight_v));

            let u32_3 = vmovl_u16(vget_high_u16(hi16));
            let f32_3 = vcvtq_f32_u32(u32_3);
            let acc_3 = vld1q_f32(ob.add(12));
            vst1q_f32(ob.add(12), vfmaq_f32(acc_3, f32_3, weight_v));
        }

        {
            let lo8 = vget_low_u8(elems_16_31);
            let hi8 = vget_high_u8(elems_16_31);
            let lo16 = vmovl_u8(lo8);
            let hi16 = vmovl_u8(hi8);

            let u32_0 = vmovl_u16(vget_low_u16(lo16));
            let f32_0 = vcvtq_f32_u32(u32_0);
            let acc_0 = vld1q_f32(ob.add(16));
            vst1q_f32(ob.add(16), vfmaq_f32(acc_0, f32_0, weight_v));

            let u32_1 = vmovl_u16(vget_high_u16(lo16));
            let f32_1 = vcvtq_f32_u32(u32_1);
            let acc_1 = vld1q_f32(ob.add(20));
            vst1q_f32(ob.add(20), vfmaq_f32(acc_1, f32_1, weight_v));

            let u32_2 = vmovl_u16(vget_low_u16(hi16));
            let f32_2 = vcvtq_f32_u32(u32_2);
            let acc_2 = vld1q_f32(ob.add(24));
            vst1q_f32(ob.add(24), vfmaq_f32(acc_2, f32_2, weight_v));

            let u32_3 = vmovl_u16(vget_high_u16(hi16));
            let f32_3 = vcvtq_f32_u32(u32_3);
            let acc_3 = vld1q_f32(ob.add(28));
            vst1q_f32(ob.add(28), vfmaq_f32(acc_3, f32_3, weight_v));
        }
    }

    let base_elem = chunks * 32;
    for i in 0..remainder {
        let abs_idx = elem_offset + base_elem + i;
        let byte_val = *packed.get_unchecked(abs_idx / 2);
        let val = if abs_idx.is_multiple_of(2) {
            byte_val & 0x0F
        } else {
            byte_val >> 4
        };
        *out.get_unchecked_mut(base_elem + i) += (val as u32 * 17) as f32 * weight;
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn accumulate_u4_weighted_sse41(
    packed: &[u8],
    elem_offset: usize,
    count: usize,
    weight: f32,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    debug_assert!(elem_offset.is_multiple_of(2));

    let weight_v = _mm_set1_ps(weight);
    let mask_lo = _mm_set1_epi8(0x0F);
    let zero = _mm_setzero_si128();

    let byte_offset = elem_offset / 2;
    let packed_ptr = packed.as_ptr().add(byte_offset);
    let out_ptr = out.as_mut_ptr();

    let chunks = count / 32;
    let remainder = count % 32;

    for chunk in 0..chunks {
        let pb = packed_ptr.add(chunk * 16);
        let ob = out_ptr.add(chunk * 32);

        let bytes = _mm_loadu_si128(pb as *const __m128i);

        let low = _mm_and_si128(bytes, mask_lo);
        let high = _mm_srli_epi16::<4>(bytes);
        let high = _mm_and_si128(high, mask_lo);

        let low_scaled = _mm_add_epi8(_mm_slli_epi16::<4>(_mm_and_si128(low, mask_lo)), low);
        let high_scaled = _mm_add_epi8(_mm_slli_epi16::<4>(_mm_and_si128(high, mask_lo)), high);

        let elems_0_15 = _mm_unpacklo_epi8(low_scaled, high_scaled);
        let elems_16_31 = _mm_unpackhi_epi8(low_scaled, high_scaled);

        {
            let lo8 = _mm_unpacklo_epi8(elems_0_15, zero);
            let hi8 = _mm_unpackhi_epi8(elems_0_15, zero);

            let u32_0 = _mm_unpacklo_epi16(lo8, zero);
            let f32_0 = _mm_cvtepi32_ps(u32_0);
            let acc_0 = _mm_loadu_ps(ob);
            _mm_storeu_ps(ob, _mm_add_ps(acc_0, _mm_mul_ps(f32_0, weight_v)));

            let u32_1 = _mm_unpackhi_epi16(lo8, zero);
            let f32_1 = _mm_cvtepi32_ps(u32_1);
            let acc_1 = _mm_loadu_ps(ob.add(4));
            _mm_storeu_ps(ob.add(4), _mm_add_ps(acc_1, _mm_mul_ps(f32_1, weight_v)));

            let u32_2 = _mm_unpacklo_epi16(hi8, zero);
            let f32_2 = _mm_cvtepi32_ps(u32_2);
            let acc_2 = _mm_loadu_ps(ob.add(8));
            _mm_storeu_ps(ob.add(8), _mm_add_ps(acc_2, _mm_mul_ps(f32_2, weight_v)));

            let u32_3 = _mm_unpackhi_epi16(hi8, zero);
            let f32_3 = _mm_cvtepi32_ps(u32_3);
            let acc_3 = _mm_loadu_ps(ob.add(12));
            _mm_storeu_ps(ob.add(12), _mm_add_ps(acc_3, _mm_mul_ps(f32_3, weight_v)));
        }

        {
            let lo8 = _mm_unpacklo_epi8(elems_16_31, zero);
            let hi8 = _mm_unpackhi_epi8(elems_16_31, zero);

            let u32_0 = _mm_unpacklo_epi16(lo8, zero);
            let f32_0 = _mm_cvtepi32_ps(u32_0);
            let acc_0 = _mm_loadu_ps(ob.add(16));
            _mm_storeu_ps(ob.add(16), _mm_add_ps(acc_0, _mm_mul_ps(f32_0, weight_v)));

            let u32_1 = _mm_unpackhi_epi16(lo8, zero);
            let f32_1 = _mm_cvtepi32_ps(u32_1);
            let acc_1 = _mm_loadu_ps(ob.add(20));
            _mm_storeu_ps(ob.add(20), _mm_add_ps(acc_1, _mm_mul_ps(f32_1, weight_v)));

            let u32_2 = _mm_unpacklo_epi16(hi8, zero);
            let f32_2 = _mm_cvtepi32_ps(u32_2);
            let acc_2 = _mm_loadu_ps(ob.add(24));
            _mm_storeu_ps(ob.add(24), _mm_add_ps(acc_2, _mm_mul_ps(f32_2, weight_v)));

            let u32_3 = _mm_unpackhi_epi16(hi8, zero);
            let f32_3 = _mm_cvtepi32_ps(u32_3);
            let acc_3 = _mm_loadu_ps(ob.add(28));
            _mm_storeu_ps(ob.add(28), _mm_add_ps(acc_3, _mm_mul_ps(f32_3, weight_v)));
        }
    }

    let base_elem = chunks * 32;
    for i in 0..remainder {
        let abs_idx = elem_offset + base_elem + i;
        let byte_val = *packed.get_unchecked(abs_idx / 2);
        let val = if abs_idx.is_multiple_of(2) {
            byte_val & 0x0F
        } else {
            byte_val >> 4
        };
        *out.get_unchecked_mut(base_elem + i) += (val as u32 * 17) as f32 * weight;
    }
}

// ============================================================================
// Block mask computation: standalone function with SIMD dispatch
// ============================================================================
/// Compute per-block query-dim presence masks from 4-bit packed grid data.
///
/// Standalone function that works with any grid slice (full index grid or
/// compact query-local grid). Uses SIMD when available.
///
/// `grid` layout: `grid[dim_idx * prs + byte_idx]` where each byte packs
/// two 4-bit values (low nibble = even block, high nibble = odd block).
///
/// `query_dims` entries: `(dim_idx, weight)` where dim_idx indexes into grid rows.
/// Bits-dispatching mask computation (which query dims are present per block).
pub(crate) fn compute_block_masks(
    grid: &[u8],
    grid_bits: u8,
    prs: usize,
    query_dims: &[(usize, f32)],
    block_start: usize,
    count: usize,
    masks: &mut [u64],
) {
    // Presence masks have 64 bits. For wider queries, disable this optional
    // shortcut (all blocks appear present) and let binary search decide per
    // dimension; shifting by q>=64 used to panic in debug and wrap in release.
    if query_dims.len() > 64 {
        masks[..count].fill(u64::MAX);
        return;
    }
    if grid_bits == 2 {
        compute_block_masks_2bit(grid, prs, query_dims, block_start, count, masks);
    } else {
        compute_block_masks_4bit(grid, prs, query_dims, block_start, count, masks);
    }
}

/// Scalar 2-bit presence masks.
pub(crate) fn compute_block_masks_2bit(
    grid: &[u8],
    prs: usize,
    query_dims: &[(usize, f32)],
    block_start: usize,
    count: usize,
    masks: &mut [u64],
) {
    debug_assert!(masks.len() >= count);
    if query_dims.len() > 64 {
        masks[..count].fill(u64::MAX);
        return;
    }
    masks[..count].fill(0);
    for (q, &(dim_idx, _)) in query_dims.iter().enumerate() {
        let row = &grid[dim_idx * prs..(dim_idx + 1) * prs];
        let bit = 1u64 << q;
        for b in 0..count {
            let abs = block_start + b;
            let cell = (unsafe { *row.get_unchecked(abs / 4) } >> ((abs % 4) * 2)) & 0x03;
            if cell != 0 {
                unsafe { *masks.get_unchecked_mut(b) |= bit };
            }
        }
    }
}

pub(crate) fn compute_block_masks_4bit(
    grid: &[u8],
    prs: usize,
    query_dims: &[(usize, f32)],
    block_start: usize,
    count: usize,
    masks: &mut [u64],
) {
    debug_assert!(masks.len() >= count);
    if query_dims.len() > 64 {
        masks[..count].fill(u64::MAX);
        return;
    }
    masks[..count].fill(0);

    #[cfg(target_arch = "aarch64")]
    {
        if block_start.is_multiple_of(2) {
            unsafe {
                compute_block_masks_range_neon(grid, prs, query_dims, block_start, count, masks)
            };
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if block_start.is_multiple_of(2) && is_x86_feature_detected!("sse4.1") {
            unsafe {
                compute_block_masks_range_sse41(grid, prs, query_dims, block_start, count, masks)
            };
            return;
        }
    }

    for (q, &(dim_idx, _)) in query_dims.iter().enumerate() {
        let row = &grid[dim_idx * prs..(dim_idx + 1) * prs];
        let bit = 1u64 << q;
        for b in 0..count {
            let abs_b = block_start + b;
            let byte_val = unsafe { *row.get_unchecked(abs_b / 2) };
            let val = if abs_b.is_multiple_of(2) {
                byte_val & 0x0F
            } else {
                byte_val >> 4
            };
            if val > 0 {
                unsafe { *masks.get_unchecked_mut(b) |= bit };
            }
        }
    }
}

// ============================================================================
// Block mask SIMD kernels
// ============================================================================
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn compute_block_masks_range_neon(
    grid: &[u8],
    prs: usize,
    query_dims: &[(usize, f32)],
    block_start: usize,
    count: usize,
    masks: &mut [u64],
) {
    use std::arch::aarch64::*;

    debug_assert!(block_start.is_multiple_of(2));
    let byte_offset = block_start / 2;
    let zero = vdupq_n_u8(0);
    let mask_lo = vdupq_n_u8(0x0F);

    for (q, &(dim_idx, _)) in query_dims.iter().enumerate() {
        let row_ptr = grid.as_ptr().add(dim_idx * prs + byte_offset);
        let bit = 1u64 << q;

        let chunks = count / 32;
        let remainder = count % 32;

        for chunk in 0..chunks {
            let pb = row_ptr.add(chunk * 16);
            let base = chunk * 32;

            let bytes = vld1q_u8(pb);

            let low = vandq_u8(bytes, mask_lo);
            let high = vshrq_n_u8::<4>(bytes);

            let elems_lo = vzip1q_u8(low, high);
            let elems_hi = vzip2q_u8(low, high);

            let nz_lo = vcgtq_u8(elems_lo, zero);
            let nz_hi = vcgtq_u8(elems_hi, zero);

            let mut lo_arr = [0u8; 16];
            let mut hi_arr = [0u8; 16];
            vst1q_u8(lo_arr.as_mut_ptr(), nz_lo);
            vst1q_u8(hi_arr.as_mut_ptr(), nz_hi);

            for (i, &v) in lo_arr.iter().enumerate() {
                if v != 0 {
                    *masks.get_unchecked_mut(base + i) |= bit;
                }
            }
            for (i, &v) in hi_arr.iter().enumerate() {
                if v != 0 {
                    *masks.get_unchecked_mut(base + 16 + i) |= bit;
                }
            }
        }

        let base = chunks * 32;
        for i in 0..remainder {
            let abs_b = block_start + base + i;
            let byte_val = *grid.get_unchecked(dim_idx * prs + abs_b / 2);
            let val = if abs_b.is_multiple_of(2) {
                byte_val & 0x0F
            } else {
                byte_val >> 4
            };
            if val > 0 {
                *masks.get_unchecked_mut(base + i) |= bit;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn compute_block_masks_range_sse41(
    grid: &[u8],
    prs: usize,
    query_dims: &[(usize, f32)],
    block_start: usize,
    count: usize,
    masks: &mut [u64],
) {
    use std::arch::x86_64::*;

    debug_assert!(block_start.is_multiple_of(2));
    let byte_offset = block_start / 2;
    let zero = _mm_setzero_si128();
    let mask_lo_v = _mm_set1_epi8(0x0F);

    for (q, &(dim_idx, _)) in query_dims.iter().enumerate() {
        let row_ptr = grid.as_ptr().add(dim_idx * prs + byte_offset);
        let bit = 1u64 << q;

        let chunks = count / 32;
        let remainder = count % 32;

        for chunk in 0..chunks {
            let pb = row_ptr.add(chunk * 16);
            let base = chunk * 32;

            let bytes = _mm_loadu_si128(pb as *const __m128i);

            let low = _mm_and_si128(bytes, mask_lo_v);
            let high = _mm_and_si128(_mm_srli_epi16::<4>(bytes), mask_lo_v);

            let elems_lo = _mm_unpacklo_epi8(low, high);
            let elems_hi = _mm_unpackhi_epi8(low, high);

            let nz_lo = _mm_cmpgt_epi8(elems_lo, zero);
            let nz_hi = _mm_cmpgt_epi8(elems_hi, zero);

            let mut m = _mm_movemask_epi8(nz_lo) as u32;
            while m != 0 {
                let i = m.trailing_zeros() as usize;
                m &= m - 1;
                *masks.get_unchecked_mut(base + i) |= bit;
            }
            let mut m = _mm_movemask_epi8(nz_hi) as u32;
            while m != 0 {
                let i = m.trailing_zeros() as usize;
                m &= m - 1;
                *masks.get_unchecked_mut(base + 16 + i) |= bit;
            }
        }

        let base = chunks * 32;
        for i in 0..remainder {
            let abs_b = block_start + base + i;
            let byte_val = *grid.get_unchecked(dim_idx * prs + abs_b / 2);
            let val = if abs_b.is_multiple_of(2) {
                byte_val & 0x0F
            } else {
                byte_val >> 4
            };
            if val > 0 {
                *masks.get_unchecked_mut(base + i) |= bit;
            }
        }
    }
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
            u32::from_le_bytes(blob[footer + 32..footer + 36].try_into().unwrap()) as usize;
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
}

/// Microbenchmark: dense 4-bit grid (SIMD nibble sweep, the current layout)
/// vs a CSR superblock-run grid, on Zipfian SPLADE-like block occupancy.
/// Run: cargo test --release -p hermes-core --features native --lib -- \
///        --ignored bench_grid_dense_vs_csr --nocapture
/// See docs/bmp-grid-compression.md for measured results.
#[cfg(test)]
mod grid_bench {
    use super::{accumulate_u4_weighted, compute_block_masks};
    use std::hint::black_box;
    use std::time::Instant;

    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_f64(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn wide_query_disables_u64_presence_mask_without_shifting() {
        let query_dims: Vec<(usize, f32)> = (0..65).map(|dim| (dim, 1.0)).collect();
        let grid = vec![0x11; query_dims.len()];
        let mut masks = [0u64; 2];
        compute_block_masks(&grid, 4, 1, &query_dims, 0, 2, &mut masks);
        assert_eq!(masks, [u64::MAX; 2]);
    }

    /// CSR grid: per dim, runs of present superblocks with (slot, nibble)
    /// entries. Flattened per dim for cache-friendly probing.
    struct CsrGrid {
        /// per dim: [start..end) into run_sbs / run_entry_offsets
        dim_run_offsets: Vec<u32>,
        /// superblock id of each run
        run_sbs: Vec<u16>,
        /// per run: [start..end) into entries
        run_entry_offsets: Vec<u32>,
        /// (block_in_sb, nibble)
        entries: Vec<(u8, u8)>,
    }

    impl CsrGrid {
        /// Accumulate dim's contribution to the 64 block UBs of superblock
        /// `sb` — the CSR replacement for one dense accumulate_u4_weighted
        /// call. Returns false if the dim has no blocks in this superblock.
        #[inline]
        fn accumulate(&self, dim: usize, sb: u16, weight: f32, out: &mut [f32]) -> bool {
            let rs = self.dim_run_offsets[dim] as usize;
            let re = self.dim_run_offsets[dim + 1] as usize;
            let runs = &self.run_sbs[rs..re];
            match runs.binary_search(&sb) {
                Ok(pos) => {
                    let es = self.run_entry_offsets[rs + pos] as usize;
                    let ee = self.run_entry_offsets[rs + pos + 1] as usize;
                    for &(slot, nib) in &self.entries[es..ee] {
                        unsafe {
                            *out.get_unchecked_mut(slot as usize) +=
                                (nib as u32 * 17) as f32 * weight;
                        }
                    }
                    true
                }
                Err(_) => false,
            }
        }
    }

    fn run_config(num_docs: usize, block_size: usize, dims: usize, nnz_per_doc: usize) {
        const SB: usize = 64; // blocks per superblock
        let num_blocks = num_docs.div_ceil(block_size);
        let num_sbs = num_blocks.div_ceil(SB);
        let prs = num_blocks.div_ceil(2);

        // Zipf(1.0) over dims: cumulative distribution for binary-search sampling
        let mut cum = Vec::with_capacity(dims);
        let mut acc = 0.0f64;
        for i in 0..dims {
            acc += 1.0 / (i + 1) as f64;
            cum.push(acc);
        }
        let total = acc;

        // Per-block occupancy: block_size × nnz draws, dedup per block.
        // (dim, block) → nibble, filling dense + collecting CSR entries.
        let mut rng = XorShift(0x9E3779B97F4A7C15);
        let mut dense = vec![0u8; dims * prs];
        // per-dim collected (block, nibble), built block-major then sorted
        let mut per_dim: Vec<Vec<(u32, u8)>> = vec![Vec::new(); dims];
        let mut seen = vec![u32::MAX; dims];
        let mut total_entries = 0u64;
        let gen_start = Instant::now();
        for b in 0..num_blocks {
            let draws = block_size * nnz_per_doc;
            for _ in 0..draws {
                let r = rng.next_f64() * total;
                let dim = cum.partition_point(|&c| c < r).min(dims - 1);
                if seen[dim] != b as u32 {
                    seen[dim] = b as u32;
                    let nib = (rng.next() % 15 + 1) as u8;
                    dense[dim * prs + b / 2] |= if b % 2 == 0 { nib } else { nib << 4 };
                    per_dim[dim].push((b as u32, nib));
                    total_entries += 1;
                }
            }
        }
        // Build CSR (per-dim vectors are already block-sorted by construction)
        let mut csr = CsrGrid {
            dim_run_offsets: Vec::with_capacity(dims + 1),
            run_sbs: Vec::new(),
            run_entry_offsets: Vec::new(),
            entries: Vec::with_capacity(total_entries as usize),
        };
        csr.dim_run_offsets.push(0);
        for blocks in &per_dim {
            let mut cur_sb = u32::MAX;
            for &(b, nib) in blocks {
                let sb = b as usize / SB;
                if sb as u32 != cur_sb {
                    cur_sb = sb as u32;
                    csr.run_sbs.push(sb as u16);
                    csr.run_entry_offsets.push(csr.entries.len() as u32);
                }
                csr.entries.push(((b as usize % SB) as u8, nib));
            }
            csr.dim_run_offsets.push(csr.run_sbs.len() as u32);
        }
        csr.run_entry_offsets.push(csr.entries.len() as u32);
        let num_runs = csr.run_sbs.len() as u64;

        // ── Memory accounting ────────────────────────────────────────────
        let dense_bytes = dense.len() as u64;
        // sb-run encoding: 10 bits/entry (6-bit slot + 4-bit nibble) + 3B/run
        // (u16 sb + u8 count) + 5B/dim row offset
        let csr_bytes = total_entries * 10 / 8 + num_runs * 3 + dims as u64 * 5;
        // flat encoding: u16 block delta + 4-bit nibble + 4B/dim offset
        let flat_bytes = total_entries * 5 / 2 + dims as u64 * 4;
        let density = total_entries as f64 / (dims as f64 * num_blocks as f64);
        println!(
            "\n=== docs={num_docs} block={block_size} dims={dims} blocks={num_blocks} sbs={num_sbs} (gen {:.1}s) ===",
            gen_start.elapsed().as_secs_f64()
        );
        println!(
            "occupancy: {total_entries} (dim,block) pairs, density {:.2}% | runs {num_runs}",
            density * 100.0
        );
        println!(
            "memory: dense {:.1} MB | csr sb-run {:.1} MB ({:.1}x) | csr flat {:.1} MB ({:.1}x)",
            dense_bytes as f64 / 1e6,
            csr_bytes as f64 / 1e6,
            dense_bytes as f64 / csr_bytes as f64,
            flat_bytes as f64 / 1e6,
            dense_bytes as f64 / flat_bytes as f64,
        );

        // ── Correctness: identical block UBs from both layouts ──────────
        let mut out_d = vec![0.0f32; SB];
        let mut out_c = vec![0.0f32; SB];
        for probe in 0..200 {
            let dim = (rng.next() % dims as u64) as usize;
            let sb = (rng.next() % num_sbs as u64) as usize;
            let count = SB.min(num_blocks - sb * SB);
            out_d[..count].fill(0.0);
            out_c[..count].fill(0.0);
            accumulate_u4_weighted(
                &dense[dim * prs..(dim + 1) * prs],
                sb * SB,
                count,
                1.5,
                &mut out_d[..count],
            );
            csr.accumulate(dim, sb as u16, 1.5, &mut out_c[..count]);
            assert_eq!(out_d, out_c, "layout mismatch at probe {probe}");
        }

        // ── Timing: Q queries × 16 dims × 30% surviving superblocks ─────
        const QUERIES: usize = 200;
        const QDIMS: usize = 16;
        let surviving = (num_sbs * 3 / 10).max(1);
        type Query = (Vec<(usize, f32)>, Vec<u16>);
        let queries: Vec<Query> = (0..QUERIES)
            .map(|_| {
                let qdims: Vec<(usize, f32)> = (0..QDIMS)
                    .map(|_| {
                        let r = rng.next_f64() * total;
                        let dim = cum.partition_point(|&c| c < r).min(dims - 1);
                        (dim, rng.next_f64() as f32 + 0.1)
                    })
                    .collect();
                let sbs: Vec<u16> = (0..surviving)
                    .map(|_| (rng.next() % num_sbs as u64) as u16)
                    .collect();
                (qdims, sbs)
            })
            .collect();

        let probes = (QUERIES * QDIMS * surviving) as f64;

        let t = Instant::now();
        let mut sink = 0.0f32;
        for (qdims, sbs) in &queries {
            for &sb in sbs {
                out_d.fill(0.0);
                for &(dim, w) in qdims {
                    let count = SB.min(num_blocks - (sb as usize) * SB);
                    accumulate_u4_weighted(
                        &dense[dim * prs..(dim + 1) * prs],
                        sb as usize * SB,
                        count,
                        w,
                        &mut out_d[..count],
                    );
                }
                sink += out_d[0];
            }
        }
        let dense_t = t.elapsed();
        black_box(sink);

        let t = Instant::now();
        let mut sink = 0.0f32;
        let mut hits = 0u64;
        for (qdims, sbs) in &queries {
            for &sb in sbs {
                out_c.fill(0.0);
                for &(dim, w) in qdims {
                    if csr.accumulate(dim, sb, w, &mut out_c) {
                        hits += 1;
                    }
                }
                sink += out_c[0];
            }
        }
        let csr_t = t.elapsed();
        black_box(sink);

        println!(
            "block-UB compute: dense {:.0} ns/probe ({:.2} ms/query) | csr {:.0} ns/probe ({:.2} ms/query, {:.0}% probes hit) | csr/dense {:.2}x",
            dense_t.as_nanos() as f64 / probes,
            dense_t.as_secs_f64() * 1000.0 / QUERIES as f64,
            csr_t.as_nanos() as f64 / probes,
            csr_t.as_secs_f64() * 1000.0 / QUERIES as f64,
            hits as f64 / probes * 100.0,
            csr_t.as_secs_f64() / dense_t.as_secs_f64(),
        );
    }
}
