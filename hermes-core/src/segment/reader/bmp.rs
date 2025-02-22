//! BMP (Block-Max Pruning) index reader for sparse vectors — **V12 zero-copy**.
//!
//! V12 uses fixed `dims` (vocabulary size) and dim_id directly in per-block data.
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

/// BMP V12 index for a single sparse field — fully zero-copy mmap-backed.
///
/// V12 format with Section H (per-block SimHash) for merge-time reordering.
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
    total_terms: u32,
    total_postings: u32,
    /// Packed row size for 4-bit grid: `(num_blocks + 1) / 2`
    packed_row_size: u32,
    /// Actual vector count before padding
    num_real_docs: u32,

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
    /// Section H: per-block majority SimHash [u64-LE × num_blocks].
    /// Empty when SimHash is disabled.
    block_simhash_bytes: OwnedBytes,
}

// SAFETY: All raw pointer access is derived from OwnedBytes which are Send+Sync
// (backed by Arc<Vec<u8>> or Arc<Mmap>). The pointers are never mutated.
// BmpIndex already stores OwnedBytes (which is Send+Sync), so the struct
// inherits Send+Sync automatically through its fields.

impl BmpIndex {
    /// Parse a BMP V12 blob from the given file handle.
    ///
    /// Reads the 72-byte footer, then acquires the entire blob as a single
    /// `OwnedBytes` and slices it into zero-copy sections.
    ///
    /// V12 data-first layout: Section B (per-block interleaved data) first,
    /// then Section A (block_data_starts with u64 entries), grids, doc_map,
    /// and Section H (per-block SimHash).
    pub fn parse(
        handle: FileHandle,
        blob_offset: u64,
        blob_len: u64,
        _total_docs: u32,
        total_vectors: u32,
    ) -> crate::Result<Self> {
        use crate::segment::format::{BMP_BLOB_FOOTER_SIZE_V12, BMP_BLOB_MAGIC_V12};

        if blob_len < BMP_BLOB_FOOTER_SIZE_V12 as u64 {
            return Err(crate::Error::Corruption(
                "BMP blob too small for V12 footer".into(),
            ));
        }

        // Read the footer (last 72 bytes of the blob)
        let footer_start = blob_offset + blob_len - BMP_BLOB_FOOTER_SIZE_V12 as u64;
        let footer_bytes = handle
            .read_bytes_range_sync(footer_start..footer_start + BMP_BLOB_FOOTER_SIZE_V12 as u64)
            .map_err(crate::Error::Io)?;
        let fb = footer_bytes.as_slice();

        let total_terms = u32::from_le_bytes(fb[0..4].try_into().unwrap());
        let total_postings = u32::from_le_bytes(fb[4..8].try_into().unwrap());
        let grid_offset = u64::from_le_bytes(fb[8..16].try_into().unwrap());
        let sb_grid_offset = u64::from_le_bytes(fb[16..24].try_into().unwrap());
        let num_blocks = u32::from_le_bytes(fb[24..28].try_into().unwrap());
        let dims = u32::from_le_bytes(fb[28..32].try_into().unwrap());
        let bmp_block_size = u32::from_le_bytes(fb[32..36].try_into().unwrap());
        let num_virtual_docs = u32::from_le_bytes(fb[36..40].try_into().unwrap());
        let max_weight_scale = f32::from_le_bytes(fb[40..44].try_into().unwrap());
        let doc_map_offset = u64::from_le_bytes(fb[44..52].try_into().unwrap());
        let num_real_docs = u32::from_le_bytes(fb[52..56].try_into().unwrap());
        let block_simhash_offset = u64::from_le_bytes(fb[56..64].try_into().unwrap());
        // fb[64..68] = reserved
        let magic = u32::from_le_bytes(fb[68..72].try_into().unwrap());

        if magic != BMP_BLOB_MAGIC_V12 {
            return Err(crate::Error::Corruption(format!(
                "Invalid BMP blob magic: {:#x} (expected BMP2 {:#x})",
                magic, BMP_BLOB_MAGIC_V12
            )));
        }

        // Handle empty index
        if num_blocks == 0 {
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
                num_real_docs,
                block_data_starts_bytes: OwnedBytes::empty(),
                block_data_bytes: OwnedBytes::empty(),
                grid_bytes: OwnedBytes::empty(),
                sb_grid_bytes: OwnedBytes::empty(),
                num_superblocks: 0,
                doc_map_ids_bytes: OwnedBytes::empty(),
                doc_map_ordinals_bytes: OwnedBytes::empty(),
                block_simhash_bytes: OwnedBytes::empty(),
            });
        }

        // Read entire blob (excluding footer) as one OwnedBytes — zero-copy mmap slice
        let data_len = blob_len - BMP_BLOB_FOOTER_SIZE_V12 as u64;
        let blob = handle
            .read_bytes_range_sync(blob_offset..blob_offset + data_len)
            .map_err(crate::Error::Io)?;

        // Layout: Section B (block_data) at offset 0, Section A (block_data_starts)
        // immediately before grid. Derive Section A position from grid_offset.
        let section_a_size = (num_blocks as usize + 1) * 8;
        let bds_start = grid_offset as usize - section_a_size;

        // Section B: block_data [0..bds_start) (includes padding before Section A)
        let block_data_bytes = blob.slice(0..bds_start);
        // Section A: block_data_starts [bds_start..grid_offset)
        let block_data_starts_bytes = blob.slice(bds_start..grid_offset as usize);

        // Sections D+E: grid, sb_grid, doc_map
        let packed_row_size = (num_blocks as usize).div_ceil(2) as u32;
        let grid_start = grid_offset as usize;
        let grid_end = grid_start + dims as usize * packed_row_size as usize;

        let num_superblocks = num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE);
        let sb_grid_start = sb_grid_offset as usize;
        let sb_grid_end = sb_grid_start + dims as usize * num_superblocks as usize;

        let dm_start = doc_map_offset as usize;
        let dm_ids_end = dm_start + num_virtual_docs as usize * 4;
        let dm_ords_end = dm_ids_end + num_virtual_docs as usize * 2;

        // Section H: per-block SimHash (V12)
        let block_simhash_bytes = if block_simhash_offset > 0 {
            let sh_start = block_simhash_offset as usize;
            let sh_end = sh_start + num_blocks as usize * 8;
            blob.slice(sh_start..sh_end)
        } else {
            OwnedBytes::empty()
        };

        // Slice into sections (all zero-copy — just offset adjustments on same Arc)
        let grid_bytes = blob.slice(grid_start..grid_end);
        let sb_grid_bytes = blob.slice(sb_grid_start..sb_grid_end);
        let doc_map_ids_bytes = blob.slice(dm_start..dm_ids_end);
        let doc_map_ordinals_bytes = blob.slice(dm_ids_end..dm_ords_end);

        log::debug!(
            "BMP V12 index loaded: num_blocks={}, num_superblocks={}, dims={}, bmp_block_size={}, \
             num_virtual_docs={}, num_real_docs={}, max_weight_scale={:.4}, postings={}, \
             packed_row_size={}, block_data={}B, doc_map={}B, simhash={}",
            num_blocks,
            num_superblocks,
            dims,
            bmp_block_size,
            num_virtual_docs,
            num_real_docs,
            max_weight_scale,
            total_postings,
            packed_row_size,
            bds_start,
            num_virtual_docs as usize * 6,
            !block_simhash_bytes.is_empty(),
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
            num_real_docs,
            block_data_starts_bytes,
            block_data_bytes,
            grid_bytes,
            sb_grid_bytes,
            num_superblocks,
            doc_map_ids_bytes,
            doc_map_ordinals_bytes,
            block_simhash_bytes,
        })
    }

    /// Convert a compact virtual_id to (doc_id, ordinal) via table lookup.
    ///
    /// Uses unchecked reads — virtual_id is validated by the caller
    /// (only called for top-k results which are valid compact virtual IDs).
    #[inline(always)]
    pub fn virtual_to_doc(&self, virtual_id: u32) -> (u32, u16) {
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

    /// Parse a block header: returns (num_terms, dim_ptr, ps_ptr, post_ptr).
    /// All pointers are within block_data_bytes — guaranteed contiguous.
    ///
    /// Always 4-byte (u32) dim IDs.
    ///
    /// Returns `(0, null, null, null)` for empty blocks.
    #[inline(always)]
    pub(crate) fn parse_block(&self, block_id: u32) -> (u16, *const u8, *const u8, *const u8) {
        let (start, end) = self.block_data_range(block_id);
        if start == end {
            return (0, std::ptr::null(), std::ptr::null(), std::ptr::null());
        }
        let base = unsafe {
            self.block_data_bytes
                .as_slice()
                .as_ptr()
                .add(start as usize)
        };
        let num_terms = unsafe { u16::from_le((base as *const u16).read_unaligned()) };
        let dim_ptr = unsafe { base.add(2) };
        // Always u32 dim IDs (4 bytes each)
        let ps_ptr = unsafe { dim_ptr.add(num_terms as usize * 4) };
        let post_ptr = unsafe { ps_ptr.add((num_terms as usize + 1) * 2) };
        (num_terms, dim_ptr, ps_ptr, post_ptr)
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
        let (num_terms, dim_ptr, ps_ptr, post_ptr) = self.parse_block(block_id);
        BlockTermIter {
            dim_ptr,
            ps_ptr,
            post_ptr,
            num_terms,
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
        self.total_terms as u64
    }

    /// Total number of postings stored in the index.
    pub fn total_postings(&self) -> u64 {
        self.total_postings as u64
    }

    /// Actual vector count before block-alignment padding.
    pub fn num_real_docs(&self) -> u32 {
        self.num_real_docs
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
            + self.block_simhash_bytes.len()
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

    /// Per-block majority SimHash for block `block_id` (V12 Section H).
    /// Returns 0 if no SimHash data is present.
    #[inline]
    pub fn block_simhash(&self, block_id: u32) -> u64 {
        let d = self.block_simhash_bytes.as_slice();
        if d.is_empty() {
            return 0;
        }
        let off = block_id as usize * 8;
        u64::from_le_bytes(d[off..off + 8].try_into().unwrap())
    }

    /// Raw per-block SimHash bytes (Section H). For merge copy.
    /// Layout: `[u64-LE × num_blocks]`. Empty if no SimHash data.
    #[inline]
    pub fn block_simhash_slice(&self) -> &[u8] {
        self.block_simhash_bytes.as_slice()
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
        Self::madvise_owned(&self.block_simhash_bytes, libc::MADV_SEQUENTIAL);
    }

    /// Release block data pages after Phase 1 completes.
    /// Keeps block_data_starts — needed for Phase 2 recomputation.
    #[cfg(feature = "native")]
    pub fn madvise_dontneed_block_data(&self) {
        Self::madvise_owned(&self.block_data_bytes, libc::MADV_DONTNEED);
    }

    /// Release grid pages after Phase 3+4 complete.
    #[cfg(feature = "native")]
    pub fn madvise_dontneed_grids(&self) {
        Self::madvise_owned(&self.grid_bytes, libc::MADV_DONTNEED);
        Self::madvise_owned(&self.sb_grid_bytes, libc::MADV_DONTNEED);
    }

    /// Call `madvise` only when the backing store is mmap.
    ///
    /// `MADV_DONTNEED` on heap (Vec) memory zeroes pages on Linux and can
    /// corrupt allocator metadata (the page-aligned pointer may reach into
    /// malloc headers before the allocation). This caused `free(): invalid
    /// pointer` crashes in CI where tests use RamDirectory (Vec-backed).
    #[cfg(feature = "native")]
    fn madvise_owned(bytes: &crate::directories::OwnedBytes, advice: i32) {
        if !bytes.is_mmap() {
            return;
        }
        let slice = bytes.as_slice();
        if slice.is_empty() {
            return;
        }
        let ptr = slice.as_ptr();
        let len = slice.len();
        // Align down to page boundary
        let page_size = 4096usize;
        let aligned_ptr = (ptr as usize) & !(page_size - 1);
        let aligned_len = len + (ptr as usize - aligned_ptr);
        unsafe {
            libc::madvise(aligned_ptr as *mut libc::c_void, aligned_len, advice);
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
    num_terms: u16,
    current: u16,
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
        let postings = unsafe { block_term_postings(self.ps_ptr, self.post_ptr, i) };
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
    num_terms: u16,
    dim_id: u32,
) -> Option<u16> {
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
            std::cmp::Ordering::Equal => return Some(mid as u16),
            std::cmp::Ordering::Greater => hi = mid,
        }
    }
    None
}

/// Get postings for a local term index within a parsed block.
///
/// `ps_ptr` points to the block's posting_starts array [u16 × (num_terms + 1)].
/// `post_ptr` points to the block's postings array [(u8, u8) × total].
///
/// # Safety
/// Pointers must be valid and derived from a BmpIndex's block_data_bytes.
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn block_term_postings<'a>(
    ps_ptr: *const u8,
    post_ptr: *const u8,
    local_term: u16,
) -> &'a [BmpPosting] {
    let start_p = ps_ptr.add(local_term as usize * 2);
    let end_p = ps_ptr.add((local_term as usize + 1) * 2);
    let start = u16::from_le((start_p as *const u16).read_unaligned()) as usize;
    let end = u16::from_le((end_p as *const u16).read_unaligned()) as usize;
    let count = end - start;
    if count == 0 {
        return &[];
    }
    // SAFETY: BmpPosting is #[repr(C)] with align=1 (two u8 fields).
    let ptr = post_ptr.add(start * 2) as *const BmpPosting;
    std::slice::from_raw_parts(ptr, count)
}

// ============================================================================
// SIMD-accelerated helpers for BMP scoring
// ============================================================================

/// Accumulate `out[i] += input[i] as f32 * weight` for all i.
///
/// Uses NEON on aarch64, SSE4.1 on x86_64, scalar fallback on other platforms.
#[inline]
pub(crate) fn accumulate_u8_weighted(input: &[u8], weight: f32, out: &mut [f32]) {
    debug_assert_eq!(input.len(), out.len());
    let n = input.len();

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64
        unsafe { accumulate_u8_weighted_neon(input, weight, out, n) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            unsafe { accumulate_u8_weighted_sse41(input, weight, out, n) };
            return;
        }
        for i in 0..n {
            out[i] += input[i] as f32 * weight;
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..n {
            out[i] += input[i] as f32 * weight;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn accumulate_u8_weighted_neon(input: &[u8], weight: f32, out: &mut [f32], n: usize) {
    use std::arch::aarch64::*;

    let weight_v = vdupq_n_f32(weight);
    let chunks = n / 16;
    let remainder = n % 16;

    for chunk in 0..chunks {
        let base = chunk * 16;
        let in_ptr = input.as_ptr().add(base);
        let out_ptr = out.as_mut_ptr().add(base);

        let bytes = vld1q_u8(in_ptr);

        let low8 = vget_low_u8(bytes);
        let high8 = vget_high_u8(bytes);
        let low16 = vmovl_u8(low8);
        let high16 = vmovl_u8(high8);

        let u32_0 = vmovl_u16(vget_low_u16(low16));
        let f32_0 = vcvtq_f32_u32(u32_0);
        let acc_0 = vld1q_f32(out_ptr);
        vst1q_f32(out_ptr, vfmaq_f32(acc_0, f32_0, weight_v));

        let u32_1 = vmovl_u16(vget_high_u16(low16));
        let f32_1 = vcvtq_f32_u32(u32_1);
        let acc_1 = vld1q_f32(out_ptr.add(4));
        vst1q_f32(out_ptr.add(4), vfmaq_f32(acc_1, f32_1, weight_v));

        let u32_2 = vmovl_u16(vget_low_u16(high16));
        let f32_2 = vcvtq_f32_u32(u32_2);
        let acc_2 = vld1q_f32(out_ptr.add(8));
        vst1q_f32(out_ptr.add(8), vfmaq_f32(acc_2, f32_2, weight_v));

        let u32_3 = vmovl_u16(vget_high_u16(high16));
        let f32_3 = vcvtq_f32_u32(u32_3);
        let acc_3 = vld1q_f32(out_ptr.add(12));
        vst1q_f32(out_ptr.add(12), vfmaq_f32(acc_3, f32_3, weight_v));
    }

    let base = chunks * 16;
    for i in 0..remainder {
        *out.get_unchecked_mut(base + i) += *input.get_unchecked(base + i) as f32 * weight;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn accumulate_u8_weighted_sse41(input: &[u8], weight: f32, out: &mut [f32], n: usize) {
    use std::arch::x86_64::*;

    let weight_v = _mm_set1_ps(weight);
    let zero = _mm_setzero_si128();
    let chunks = n / 16;
    let remainder = n % 16;

    for chunk in 0..chunks {
        let base = chunk * 16;
        let in_ptr = input.as_ptr().add(base);
        let out_ptr = out.as_mut_ptr().add(base);

        let bytes = _mm_loadu_si128(in_ptr as *const __m128i);

        let lo8 = _mm_unpacklo_epi8(bytes, zero);
        let hi8 = _mm_unpackhi_epi8(bytes, zero);

        let u32_0 = _mm_unpacklo_epi16(lo8, zero);
        let f32_0 = _mm_cvtepi32_ps(u32_0);
        let acc_0 = _mm_loadu_ps(out_ptr);
        _mm_storeu_ps(out_ptr, _mm_add_ps(acc_0, _mm_mul_ps(f32_0, weight_v)));

        let u32_1 = _mm_unpackhi_epi16(lo8, zero);
        let f32_1 = _mm_cvtepi32_ps(u32_1);
        let acc_1 = _mm_loadu_ps(out_ptr.add(4));
        _mm_storeu_ps(
            out_ptr.add(4),
            _mm_add_ps(acc_1, _mm_mul_ps(f32_1, weight_v)),
        );

        let u32_2 = _mm_unpacklo_epi16(hi8, zero);
        let f32_2 = _mm_cvtepi32_ps(u32_2);
        let acc_2 = _mm_loadu_ps(out_ptr.add(8));
        _mm_storeu_ps(
            out_ptr.add(8),
            _mm_add_ps(acc_2, _mm_mul_ps(f32_2, weight_v)),
        );

        let u32_3 = _mm_unpackhi_epi16(hi8, zero);
        let f32_3 = _mm_cvtepi32_ps(u32_3);
        let acc_3 = _mm_loadu_ps(out_ptr.add(12));
        _mm_storeu_ps(
            out_ptr.add(12),
            _mm_add_ps(acc_3, _mm_mul_ps(f32_3, weight_v)),
        );
    }

    let base = chunks * 16;
    for i in 0..remainder {
        *out.get_unchecked_mut(base + i) += *input.get_unchecked(base + i) as f32 * weight;
    }
}

// ============================================================================
// 4-bit grid accumulation
// ============================================================================

/// Accumulate 4-bit packed grid values into f32 output.
///
/// Internally unpacks u4 → u8 (×17) then does FMA, so the caller can use the
/// same `/255.0` weight scale as for u8 grids.
///
/// `packed[i/2]`: low nibble = even element, high nibble = odd element.
#[inline]
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

#[cfg(target_arch = "aarch64")]
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

#[cfg(target_arch = "x86_64")]
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
pub(crate) fn compute_block_masks_4bit(
    grid: &[u8],
    prs: usize,
    query_dims: &[(usize, f32)],
    block_start: usize,
    count: usize,
    masks: &mut [u64],
) {
    debug_assert!(masks.len() >= count);
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
