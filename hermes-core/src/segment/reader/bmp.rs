//! BMP (Block-Max Pruning) index reader for sparse vectors — **V6 zero-copy**.
//!
//! V6 stores contiguous arrays on disk so that at load time the entire blob
//! is acquired as a single `OwnedBytes` (mmap-backed or Arc-Vec) and sliced
//! into typed sections. No heap allocation — all data including the superblock
//! grid is mmap-backed.
//!
//! V6 packs the block grid to 4-bit (50% grid memory reduction) while keeping
//! the superblock grid at full 8-bit precision for safe hierarchical pruning.
//!
//! Uses **compact virtual coordinates**: sequential IDs assigned to unique
//! `(doc_id, ordinal)` pairs. A doc_map lookup table (Section 9) maps virtual
//! IDs back to original coordinates at query time.
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

/// Read a little-endian u32 from a byte slice at element index (bounds-checked).
/// Used for non-hot-path reads (load time, merger iteration).
#[inline(always)]
fn read_u32_le(bytes: &[u8], idx: usize) -> u32 {
    let off = idx * 4;
    u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
}

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

/// BMP V7 index for a single sparse field — fully zero-copy mmap-backed.
///
/// All data sections are `OwnedBytes` slices into the same underlying mmap Arc.
/// No heap allocation — the superblock grid is persisted on disk and loaded as
/// a zero-copy OwnedBytes slice.
///
/// V7 packs the block grid to 4-bit (50% grid memory) while keeping sb_grid at
/// full 8-bit. The reader unpacks 4-bit values via ×17 to get u8-equivalent
/// upper bounds, so the same `/255.0` weight scale works for both levels.
///
/// V7 uses compact virtual IDs with a doc_map lookup table (Section 9) instead
/// of the sparse `doc_id * num_ordinals + ordinal` scheme.
///
/// V7 stores dim_indices in Section 2 instead of dim_ids, using u16 entries
/// when `num_dims ≤ 65536` (halving section 2 size for typical SPLADE models).
///
/// Hot-path accessors cache raw pointers from the OwnedBytes slices and use
/// unchecked reads with `ptr::read_unaligned` to eliminate per-iteration bounds
/// checks in the binary search and posting lookup loops.
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
    /// Number of compact virtual documents (= actual vector count)
    pub num_virtual_docs: u32,
    /// Global max weight scale factor (for dequantizing u8 impacts back to f32)
    pub max_weight_scale: f32,
    /// Total sparse vectors (from TOC entry)
    pub total_vectors: u32,

    // ── V7 section metadata ──────────────────────────────────────────
    num_dims: u32,
    total_postings: u32,
    /// Packed row size for 4-bit grid: `(num_blocks + 1) / 2`
    packed_row_size: u32,
    /// Bytes per element in term_dim_ids section: 2 (u16) or 4 (u32)
    term_dim_id_width: u8,

    // ── Zero-copy OwnedBytes sections (keeps backing store alive) ────
    block_term_starts_bytes: OwnedBytes,
    term_dim_ids_bytes: OwnedBytes,
    term_posting_starts_bytes: OwnedBytes,
    postings_bytes: OwnedBytes,
    dim_ids_bytes: OwnedBytes,
    /// 4-bit packed block grid: `grid[dim_idx * packed_row_size + block_id/2]`
    grid_bytes: OwnedBytes,
    /// sb_grid[dim_idx * num_superblocks + sb_id] = max impact across all blocks in superblock
    sb_grid_bytes: OwnedBytes,
    /// Number of superblocks
    pub num_superblocks: u32,
    /// doc_map_ids[virtual_id] = original doc_id — zero-copy OwnedBytes
    doc_map_ids_bytes: OwnedBytes,
    /// doc_map_ordinals[virtual_id] = original ordinal — zero-copy OwnedBytes
    doc_map_ordinals_bytes: OwnedBytes,
}

// SAFETY: All raw pointer access is derived from OwnedBytes which are Send+Sync
// (backed by Arc<Vec<u8>> or Arc<Mmap>). The pointers are never mutated.
// BmpIndex already stores OwnedBytes (which is Send+Sync), so the struct
// inherits Send+Sync automatically through its fields.

impl BmpIndex {
    /// Parse a BMP V7 blob from the given file handle.
    ///
    /// Reads the 48-byte footer, then acquires the entire blob as a single
    /// `OwnedBytes` and slices it into zero-copy sections. The superblock
    /// grid is persisted on disk and loaded as a zero-copy slice — no heap
    /// allocation at load time.
    ///
    /// V7 stores dim_indices in Section 2 (u16 when num_dims ≤ 65536, else u32).
    /// Uses compact virtual IDs with a doc_map lookup table (Section 9).
    pub fn parse(
        handle: FileHandle,
        blob_offset: u64,
        blob_len: u64,
        _total_docs: u32,
        total_vectors: u32,
    ) -> crate::Result<Self> {
        use crate::segment::format::{BMP_BLOB_FOOTER_SIZE_V7, BMP_BLOB_MAGIC_V7};

        if blob_len < BMP_BLOB_FOOTER_SIZE_V7 as u64 {
            return Err(crate::Error::Corruption(
                "BMP blob too small for footer".into(),
            ));
        }

        // Read the footer (last 48 bytes of the blob)
        let footer_start = blob_offset + blob_len - BMP_BLOB_FOOTER_SIZE_V7 as u64;
        let footer_bytes = handle
            .read_bytes_range_sync(footer_start..footer_start + BMP_BLOB_FOOTER_SIZE_V7 as u64)
            .map_err(crate::Error::Io)?;
        let fb = footer_bytes.as_slice();

        let total_terms = u32::from_le_bytes(fb[0..4].try_into().unwrap());
        let total_postings = u32::from_le_bytes(fb[4..8].try_into().unwrap());
        let dim_ids_offset = u32::from_le_bytes(fb[8..12].try_into().unwrap());
        let _grid_offset = u32::from_le_bytes(fb[12..16].try_into().unwrap());
        let num_blocks = u32::from_le_bytes(fb[16..20].try_into().unwrap());
        let num_dims = u32::from_le_bytes(fb[20..24].try_into().unwrap());
        let bmp_block_size = u32::from_le_bytes(fb[24..28].try_into().unwrap());
        let num_virtual_docs = u32::from_le_bytes(fb[28..32].try_into().unwrap());
        let max_weight_scale = f32::from_le_bytes(fb[32..36].try_into().unwrap());
        let sb_grid_offset = u32::from_le_bytes(fb[36..40].try_into().unwrap());
        let doc_map_offset = u32::from_le_bytes(fb[40..44].try_into().unwrap());
        let magic = u32::from_le_bytes(fb[44..48].try_into().unwrap());

        if magic != BMP_BLOB_MAGIC_V7 {
            return Err(crate::Error::Corruption(format!(
                "Invalid BMP blob magic: {:#x} (expected BMP7 {:#x})",
                magic, BMP_BLOB_MAGIC_V7
            )));
        }

        // Handle empty index
        if num_blocks == 0 || total_terms == 0 {
            return Ok(Self {
                bmp_block_size,
                num_blocks,
                num_virtual_docs,
                max_weight_scale,
                total_vectors,
                num_dims,
                total_postings: 0,
                packed_row_size: 0,
                term_dim_id_width: if num_dims <= 65536 { 2 } else { 4 },
                block_term_starts_bytes: OwnedBytes::empty(),
                term_dim_ids_bytes: OwnedBytes::empty(),
                term_posting_starts_bytes: OwnedBytes::empty(),
                postings_bytes: OwnedBytes::empty(),
                dim_ids_bytes: OwnedBytes::empty(),
                grid_bytes: OwnedBytes::empty(),
                sb_grid_bytes: OwnedBytes::empty(),
                num_superblocks: 0,
                doc_map_ids_bytes: OwnedBytes::empty(),
                doc_map_ordinals_bytes: OwnedBytes::empty(),
            });
        }

        // Read entire blob (excluding footer) as one OwnedBytes — zero-copy mmap slice
        let data_len = blob_len - BMP_BLOB_FOOTER_SIZE_V7 as u64;
        let blob = handle
            .read_bytes_range_sync(blob_offset..blob_offset + data_len)
            .map_err(crate::Error::Io)?;

        // V7: Section 2 uses u16 when num_dims ≤ 65536, else u32
        let term_dim_id_width: u8 = if num_dims <= 65536 { 2 } else { 4 };

        // Compute section boundaries
        let bts_len = (num_blocks as usize + 1) * 4;
        let tdi_len = total_terms as usize * term_dim_id_width as usize;
        let tps_len = (total_terms as usize + 1) * 4;
        let post_len = total_postings as usize * 2;

        let bts_end = bts_len;
        let tdi_end = bts_end + tdi_len;
        let tps_end = tdi_end + tps_len;
        let post_end = tps_end + post_len;

        // dim_ids and grid offsets come from footer
        let dim_ids_start = dim_ids_offset as usize;
        let dim_ids_end = dim_ids_start + num_dims as usize * 4;

        // Grid is 4-bit packed — packed_row_size = ceil(num_blocks / 2)
        let packed_row_size = (num_blocks as usize).div_ceil(2) as u32;
        let grid_start = dim_ids_end;
        let grid_end = grid_start + num_dims as usize * packed_row_size as usize;

        // sb_grid from footer — full 8-bit precision
        let num_superblocks = num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE);
        let sb_grid_start = sb_grid_offset as usize;
        let sb_grid_end = sb_grid_start + num_dims as usize * num_superblocks as usize;

        // Section 9: doc_map (compact virtual_id → (doc_id, ordinal) lookup)
        let dm_start = doc_map_offset as usize;
        let dm_ids_end = dm_start + num_virtual_docs as usize * 4;
        let dm_ords_end = dm_ids_end + num_virtual_docs as usize * 2;
        let doc_map_ids_bytes = blob.slice(dm_start..dm_ids_end);
        let doc_map_ordinals_bytes = blob.slice(dm_ids_end..dm_ords_end);

        // Slice into sections (all zero-copy — just offset adjustments on same Arc)
        let block_term_starts_bytes = blob.slice(0..bts_end);
        let term_dim_ids_bytes = blob.slice(bts_end..tdi_end);
        let term_posting_starts_bytes = blob.slice(tdi_end..tps_end);
        let postings_bytes = blob.slice(tps_end..post_end);
        let dim_ids_bytes = blob.slice(dim_ids_start..dim_ids_end);
        let grid_bytes = blob.slice(grid_start..grid_end);
        let sb_grid_bytes = blob.slice(sb_grid_start..sb_grid_end);

        log::debug!(
            "BMP V7 index loaded: num_blocks={}, num_superblocks={}, num_dims={}, bmp_block_size={}, \
             num_virtual_docs={}, max_weight_scale={:.4}, terms={}, postings={}, packed_row_size={}, \
             term_dim_id_width={}, doc_map={}B",
            num_blocks,
            num_superblocks,
            num_dims,
            bmp_block_size,
            num_virtual_docs,
            max_weight_scale,
            total_terms,
            total_postings,
            packed_row_size,
            term_dim_id_width,
            num_virtual_docs as usize * 6,
        );

        Ok(Self {
            bmp_block_size,
            num_blocks,
            num_virtual_docs,
            max_weight_scale,
            total_vectors,
            num_dims,
            total_postings,
            packed_row_size,
            term_dim_id_width,
            block_term_starts_bytes,
            term_dim_ids_bytes,
            term_posting_starts_bytes,
            postings_bytes,
            dim_ids_bytes,
            grid_bytes,
            sb_grid_bytes,
            num_superblocks,
            doc_map_ids_bytes,
            doc_map_ordinals_bytes,
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

    /// Binary search for a dimension ID in the global dim_ids array.
    #[inline]
    pub fn find_dim_idx(&self, dim_id: u32) -> Option<usize> {
        let d = self.dim_ids_bytes.as_slice();
        let n = self.num_dims as usize;
        // SAFETY: dim_ids_bytes has exactly num_dims × 4 bytes (validated at parse time)
        debug_assert!(n * 4 <= d.len());
        let base = d.as_ptr();
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let val = unsafe { read_u32_unchecked(base, mid) };
            match val.cmp(&dim_id) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Equal => return Some(mid),
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }

    /// Compute upper bound scores for ALL superblocks in a single vectorized pass.
    pub fn compute_superblock_ubs(&self, query_dims: &[(usize, f32)], out: &mut [f32]) {
        let nsb = self.num_superblocks as usize;
        debug_assert!(out.len() >= nsb);

        out[..nsb].fill(0.0);

        let sb_grid = self.sb_grid_bytes.as_slice();
        for &(dim_idx, weight) in query_dims {
            let row = &sb_grid[dim_idx * nsb..dim_idx * nsb + nsb];
            accumulate_u8_weighted(row, weight, &mut out[..nsb]);
        }
    }

    /// Compute block UBs for blocks `[block_start..block_end)` within one superblock.
    ///
    /// Reads 4-bit packed grid, unpacks via ×17 to get u8-equivalent UBs.
    pub fn compute_block_ubs_range(
        &self,
        query_dims: &[(usize, f32)],
        block_start: usize,
        block_end: usize,
        out: &mut [f32],
    ) {
        let count = block_end - block_start;
        debug_assert!(out.len() >= count);
        let prs = self.packed_row_size as usize;
        let grid = self.grid_bytes.as_slice();

        out[..count].fill(0.0);

        for &(dim_idx, weight) in query_dims {
            let row = &grid[dim_idx * prs..(dim_idx + 1) * prs];
            accumulate_u4_weighted(row, block_start, count, weight, &mut out[..count]);
        }
    }

    /// Build per-block query-dim presence masks for blocks `[block_start..block_end)`.
    ///
    /// Reads 4-bit packed grid, unpacks inline to check presence.
    /// Uses SIMD when available to process 32 blocks at a time per query dim.
    pub fn compute_block_masks_range(
        &self,
        query_dims: &[(usize, f32)],
        block_start: usize,
        block_end: usize,
        masks: &mut [u64],
    ) {
        let prs = self.packed_row_size as usize;
        let grid = self.grid_bytes.as_slice();
        compute_block_masks_4bit(
            grid,
            prs,
            query_dims,
            block_start,
            block_end - block_start,
            masks,
        );
    }

    // ── Hot-path query-time accessors (unchecked reads) ─────────────

    /// Get the term range for a block: [start..end) into term arrays.
    ///
    /// Uses unchecked reads — block_id is validated by the caller
    /// (superblock loop bounds guarantee block_id < num_blocks).
    #[inline(always)]
    pub fn block_term_range(&self, block_id: u32) -> (u32, u32) {
        let d = self.block_term_starts_bytes.as_slice();
        debug_assert!((block_id as usize + 2) * 4 <= d.len());
        let base = d.as_ptr();
        unsafe {
            let start = read_u32_unchecked(base, block_id as usize);
            let end = read_u32_unchecked(base, block_id as usize + 1);
            (start, end)
        }
    }

    /// Binary search for a dimension index in a block's term range.
    ///
    /// V7 stores dim_indices (not dim_ids) in Section 2, using u16 or u32
    /// depending on `term_dim_id_width`. The `dim_idx` parameter is the
    /// position in the dim_ids array.
    ///
    /// Returns the absolute term index if found, or None.
    ///
    /// Uses unchecked pointer reads in the inner loop — bounds are validated
    /// once via debug_assert at entry, not per-iteration.
    #[inline(always)]
    pub fn find_dim_in_block(&self, term_start: u32, term_end: u32, dim_idx: u32) -> Option<u32> {
        let count = (term_end - term_start) as usize;
        if count == 0 {
            return None;
        }
        let d = self.term_dim_ids_bytes.as_slice();
        let base_idx = term_start as usize;
        let base = d.as_ptr();

        if self.term_dim_id_width == 2 {
            // u16 path
            debug_assert!((base_idx + count) * 2 <= d.len());
            let mut lo = 0usize;
            let mut hi = count;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let val = unsafe {
                    let p = base.add((base_idx + mid) * 2);
                    u16::from_le((p as *const u16).read_unaligned()) as u32
                };
                match val.cmp(&dim_idx) {
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Equal => return Some(term_start + mid as u32),
                    std::cmp::Ordering::Greater => hi = mid,
                }
            }
        } else {
            // u32 path
            debug_assert!((base_idx + count) * 4 <= d.len());
            let mut lo = 0usize;
            let mut hi = count;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let val = unsafe { read_u32_unchecked(base, base_idx + mid) };
                match val.cmp(&dim_idx) {
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Equal => return Some(term_start + mid as u32),
                    std::cmp::Ordering::Greater => hi = mid,
                }
            }
        }
        None
    }

    /// Get the dim_idx at the given term index (bounds-checked, for merger).
    ///
    /// V7 stores dim_indices in Section 2. To get the actual dim_id,
    /// callers use `dim_id_at(dim_idx)`.
    #[inline]
    pub fn block_term_dim_idx(&self, term_idx: u32) -> u32 {
        let d = self.term_dim_ids_bytes.as_slice();
        if self.term_dim_id_width == 2 {
            let off = term_idx as usize * 2;
            u16::from_le_bytes(d[off..off + 2].try_into().unwrap()) as u32
        } else {
            read_u32_le(d, term_idx as usize)
        }
    }

    /// Get the dimension ID at the given term index (for merger compatibility).
    ///
    /// Reads dim_idx from Section 2, then looks up actual dim_id from Section 6.
    #[inline]
    pub fn block_term_dim_id(&self, term_idx: u32) -> u32 {
        let dim_idx = self.block_term_dim_idx(term_idx);
        self.dim_id_at(dim_idx as usize)
    }

    /// Get postings for a term.
    ///
    /// Uses unchecked reads for the prefix-sum lookup. BmpPosting has align=1
    /// (#[repr(C)], two u8 fields), so the pointer cast is always valid.
    #[inline(always)]
    pub fn term_postings(&self, term_idx: u32) -> &[BmpPosting] {
        let tps = self.term_posting_starts_bytes.as_slice();
        debug_assert!((term_idx as usize + 2) * 4 <= tps.len());
        let tps_base = tps.as_ptr();
        let start = unsafe { read_u32_unchecked(tps_base, term_idx as usize) } as usize;
        let end = unsafe { read_u32_unchecked(tps_base, term_idx as usize + 1) } as usize;
        let count = end - start;
        if count == 0 {
            return &[];
        }
        let pb = self.postings_bytes.as_slice();
        debug_assert!(end * 2 <= pb.len());
        // SAFETY: BmpPosting is #[repr(C)] with align=1 (two u8 fields).
        // Size is exactly 2 bytes. Any byte pointer is valid for this type.
        unsafe {
            let ptr = pb.as_ptr().add(start * 2) as *const BmpPosting;
            std::slice::from_raw_parts(ptr, count)
        }
    }

    // ── Prefetch support ─────────────────────────────────────────────

    /// Get a raw pointer to the first posting of the given block's first term.
    /// Returns `None` if the block has no terms. Used for software prefetching.
    ///
    /// Uses unchecked reads — this is only called for prefetch hints where
    /// correctness of the pointer value is not critical (worst case: prefetch
    /// a wrong cache line, which is harmless).
    #[inline(always)]
    pub fn first_posting_ptr(&self, block_id: u32) -> Option<*const u8> {
        let bts = self.block_term_starts_bytes.as_slice();
        let bts_base = bts.as_ptr();
        let term_start = unsafe { read_u32_unchecked(bts_base, block_id as usize) };
        let term_end = unsafe { read_u32_unchecked(bts_base, block_id as usize + 1) };
        if term_start >= term_end {
            return None;
        }
        let tps_base = self.term_posting_starts_bytes.as_slice().as_ptr();
        let posting_start = unsafe { read_u32_unchecked(tps_base, term_start as usize) } as usize;
        let posting_end = unsafe { read_u32_unchecked(tps_base, term_start as usize + 1) } as usize;
        if posting_start >= posting_end {
            return None;
        }
        Some(unsafe {
            self.postings_bytes
                .as_slice()
                .as_ptr()
                .add(posting_start * 2)
        })
    }

    /// Get a raw pointer to the term_dim_ids at the given term index.
    #[inline(always)]
    pub fn term_dim_ids_ptr(&self, term_start: u32) -> *const u8 {
        unsafe {
            self.term_dim_ids_bytes
                .as_slice()
                .as_ptr()
                .add(term_start as usize * self.term_dim_id_width as usize)
        }
    }

    /// Get a raw pointer to the block_term_starts at the given block.
    #[inline(always)]
    pub fn block_term_starts_ptr(&self, block_id: u32) -> *const u8 {
        unsafe {
            self.block_term_starts_bytes
                .as_slice()
                .as_ptr()
                .add(block_id as usize * 4)
        }
    }

    /// Get a raw pointer to the term_posting_starts at the given term index.
    /// Used for prefetching the posting starts for the next block's terms.
    #[inline(always)]
    pub fn term_posting_starts_ptr(&self, term_start: u32) -> *const u8 {
        unsafe {
            self.term_posting_starts_bytes
                .as_slice()
                .as_ptr()
                .add(term_start as usize * 4)
        }
    }

    // ── Non-hot-path accessors ───────────────────────────────────────

    /// Number of unique dimensions.
    pub fn num_dimensions(&self) -> usize {
        self.num_dims as usize
    }

    /// Get the dim_id at the given dimension index.
    pub fn dim_id_at(&self, idx: usize) -> u32 {
        read_u32_le(self.dim_ids_bytes.as_slice(), idx)
    }

    /// Iterate over all dimension IDs.
    pub fn dim_ids(&self) -> DimIdIter<'_> {
        DimIdIter {
            bytes: self.dim_ids_bytes.as_slice(),
            pos: 0,
            count: self.num_dims as usize,
        }
    }

    /// Total number of postings stored in the index.
    pub fn total_postings(&self) -> u64 {
        self.total_postings as u64
    }

    /// Estimated memory usage in bytes (mmap-backed region sizes).
    ///
    /// V7 fully zero-copy: all data is mmap-backed OwnedBytes, but the
    /// mapped regions still consume RSS when paged in by the OS.
    pub fn estimated_memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.block_term_starts_bytes.len()
            + self.term_dim_ids_bytes.len()
            + self.term_posting_starts_bytes.len()
            + self.postings_bytes.len()
            + self.dim_ids_bytes.len()
            + self.grid_bytes.len()
            + self.sb_grid_bytes.len()
            + self.doc_map_ids_bytes.len()
            + self.doc_map_ordinals_bytes.len()
    }

    /// Prefault all mmap-backed sections into the OS page cache.
    ///
    /// Eliminates 20-50ms cold-start penalty on first query by forcing page
    /// faults at a predictable time (segment load) rather than during query
    /// execution. On Unix, also issues `madvise(MADV_WILLNEED)` for readahead.
    ///
    /// Sections are prefaulted in query access order:
    /// 1. sb_grid (superblock UB computation)
    /// 2. grid (block UB + mask computation)
    /// 3. block_term_starts + term_dim_ids + term_posting_starts (block scoring)
    /// 4. postings (actual posting data)
    /// 5. doc_map (virtual-to-doc mapping for result output)
    pub fn warmup(&self) {
        self.sb_grid_bytes.prefault();
        self.grid_bytes.prefault();
        self.block_term_starts_bytes.prefault();
        self.term_dim_ids_bytes.prefault();
        self.term_posting_starts_bytes.prefault();
        self.postings_bytes.prefault();
        self.dim_ids_bytes.prefault();
        self.doc_map_ids_bytes.prefault();
        self.doc_map_ordinals_bytes.prefault();
    }

    /// Extract compact grid data for query-relevant dims into caller-provided buffers.
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
    pub(crate) fn packed_row_size(&self) -> usize {
        self.packed_row_size as usize
    }
}

/// Iterator over dimension IDs stored in zero-copy OwnedBytes.
pub struct DimIdIter<'a> {
    bytes: &'a [u8],
    pos: usize,
    count: usize,
}

impl<'a> Iterator for DimIdIter<'a> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.pos >= self.count {
            return None;
        }
        let val = read_u32_le(self.bytes, self.pos);
        self.pos += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.count - self.pos;
        (rem, Some(rem))
    }
}

impl<'a> ExactSizeIterator for DimIdIter<'a> {}

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

        // Load 16 u8 values
        let bytes = vld1q_u8(in_ptr);

        // Widen u8 → u16
        let low8 = vget_low_u8(bytes);
        let high8 = vget_high_u8(bytes);
        let low16 = vmovl_u8(low8);
        let high16 = vmovl_u8(high8);

        // Widen u16 → u32 → f32, FMA into output
        // Group 0: elements 0-3
        let u32_0 = vmovl_u16(vget_low_u16(low16));
        let f32_0 = vcvtq_f32_u32(u32_0);
        let acc_0 = vld1q_f32(out_ptr);
        vst1q_f32(out_ptr, vfmaq_f32(acc_0, f32_0, weight_v));

        // Group 1: elements 4-7
        let u32_1 = vmovl_u16(vget_high_u16(low16));
        let f32_1 = vcvtq_f32_u32(u32_1);
        let acc_1 = vld1q_f32(out_ptr.add(4));
        vst1q_f32(out_ptr.add(4), vfmaq_f32(acc_1, f32_1, weight_v));

        // Group 2: elements 8-11
        let u32_2 = vmovl_u16(vget_low_u16(high16));
        let f32_2 = vcvtq_f32_u32(u32_2);
        let acc_2 = vld1q_f32(out_ptr.add(8));
        vst1q_f32(out_ptr.add(8), vfmaq_f32(acc_2, f32_2, weight_v));

        // Group 3: elements 12-15
        let u32_3 = vmovl_u16(vget_high_u16(high16));
        let f32_3 = vcvtq_f32_u32(u32_3);
        let acc_3 = vld1q_f32(out_ptr.add(12));
        vst1q_f32(out_ptr.add(12), vfmaq_f32(acc_3, f32_3, weight_v));
    }

    // Scalar remainder
    let base = chunks * 16;
    for i in 0..remainder {
        *out.get_unchecked_mut(base + i) += *input.get_unchecked(base + i) as f32 * weight;
    }
}

/// SSE4.1 implementation for u8-weighted accumulation.
///
/// Processes 16 u8 elements per iteration: widen u8→u32→f32, multiply, add.
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

        // Load 16 u8 values
        let bytes = _mm_loadu_si128(in_ptr as *const __m128i);

        // Unpack u8→u16
        let lo8 = _mm_unpacklo_epi8(bytes, zero);
        let hi8 = _mm_unpackhi_epi8(bytes, zero);

        // Group 0: elements 0-3
        let u32_0 = _mm_unpacklo_epi16(lo8, zero);
        let f32_0 = _mm_cvtepi32_ps(u32_0);
        let acc_0 = _mm_loadu_ps(out_ptr);
        _mm_storeu_ps(out_ptr, _mm_add_ps(acc_0, _mm_mul_ps(f32_0, weight_v)));

        // Group 1: elements 4-7
        let u32_1 = _mm_unpackhi_epi16(lo8, zero);
        let f32_1 = _mm_cvtepi32_ps(u32_1);
        let acc_1 = _mm_loadu_ps(out_ptr.add(4));
        _mm_storeu_ps(
            out_ptr.add(4),
            _mm_add_ps(acc_1, _mm_mul_ps(f32_1, weight_v)),
        );

        // Group 2: elements 8-11
        let u32_2 = _mm_unpacklo_epi16(hi8, zero);
        let f32_2 = _mm_cvtepi32_ps(u32_2);
        let acc_2 = _mm_loadu_ps(out_ptr.add(8));
        _mm_storeu_ps(
            out_ptr.add(8),
            _mm_add_ps(acc_2, _mm_mul_ps(f32_2, weight_v)),
        );

        // Group 3: elements 12-15
        let u32_3 = _mm_unpackhi_epi16(hi8, zero);
        let f32_3 = _mm_cvtepi32_ps(u32_3);
        let acc_3 = _mm_loadu_ps(out_ptr.add(12));
        _mm_storeu_ps(
            out_ptr.add(12),
            _mm_add_ps(acc_3, _mm_mul_ps(f32_3, weight_v)),
        );
    }

    // Scalar remainder
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
            // SAFETY: NEON is always available on aarch64. Even-aligned fast path.
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

/// NEON implementation for 4-bit grid accumulation.
///
/// Processes 32 elements (16 packed bytes) per iteration:
/// 1. Load 16 packed bytes
/// 2. Extract low/high nibbles
/// 3. Scale ×17 to get u8-equivalent values
/// 4. Interleave to element order
/// 5. Widen to f32 and FMA into output
///
/// Requires `elem_offset` to be even (always true since block_start is
/// a multiple of BMP_SUPERBLOCK_SIZE=64).
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

        // Load 16 packed bytes (= 32 elements)
        let bytes = vld1q_u8(pb);

        // Extract nibbles
        let low = vandq_u8(bytes, mask_lo);
        let high = vshrq_n_u8::<4>(bytes);

        // Scale to u8 range: val * 17 (15*17=255, fits in u8)
        let low_scaled = vmulq_u8(low, scale17);
        let high_scaled = vmulq_u8(high, scale17);

        // Interleave to element order: (low[0],high[0],low[1],high[1],...)
        let elems_0_15 = vzip1q_u8(low_scaled, high_scaled);
        let elems_16_31 = vzip2q_u8(low_scaled, high_scaled);

        // Process first 16 elements through u8→f32 widen+FMA pipeline
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

        // Process second 16 elements
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

    // Scalar remainder
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

/// SSE4.1 implementation for 4-bit packed grid accumulation.
///
/// Processes 32 elements (16 packed bytes) per iteration:
/// 1. Load 16 packed bytes
/// 2. AND with 0x0F for low nibbles, SHR 4 for high nibbles
/// 3. Scale ×17 via `(v << 4) + v` (shift+add replaces multiply)
/// 4. Interleave to element order
/// 5. Widen to f32 and FMA into output
///
/// Requires `elem_offset` to be even.
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

        // Load 16 packed bytes (= 32 elements)
        let bytes = _mm_loadu_si128(pb as *const __m128i);

        // Extract nibbles
        let low = _mm_and_si128(bytes, mask_lo);
        let high = _mm_srli_epi16::<4>(bytes);
        let high = _mm_and_si128(high, mask_lo); // clean high bits after shift

        // Scale ×17 using (v << 4) + v
        let low_scaled = _mm_add_epi8(_mm_slli_epi16::<4>(_mm_and_si128(low, mask_lo)), low);
        let high_scaled = _mm_add_epi8(_mm_slli_epi16::<4>(_mm_and_si128(high, mask_lo)), high);

        // Interleave to element order: (low[0],high[0],low[1],high[1],...)
        let elems_0_15 = _mm_unpacklo_epi8(low_scaled, high_scaled);
        let elems_16_31 = _mm_unpackhi_epi8(low_scaled, high_scaled);

        // Process first 16 elements
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

        // Process second 16 elements
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

    // Scalar remainder
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

/// NEON kernel: compute block masks for 4-bit grid.
///
/// Processes 32 blocks (16 packed bytes) per iteration per query dim.
/// For each packed byte, extracts both nibbles, checks non-zero, and ORs
/// the query dim's bit into the corresponding mask.
///
/// Requires `block_start` to be even (always true: BMP_SUPERBLOCK_SIZE=64).
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

            // Load 16 packed bytes = 32 elements
            let bytes = vld1q_u8(pb);

            // Extract nibbles
            let low = vandq_u8(bytes, mask_lo);
            let high = vshrq_n_u8::<4>(bytes);

            // Interleave to element order
            let elems_lo = vzip1q_u8(low, high); // elements 0-15
            let elems_hi = vzip2q_u8(low, high); // elements 16-31

            // Compare > 0: result bytes are 0xFF or 0x00
            let nz_lo = vcgtq_u8(elems_lo, zero);
            let nz_hi = vcgtq_u8(elems_hi, zero);

            // Extract per-byte results — store to temp array
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

        // Scalar remainder
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

/// SSE4.1 kernel: compute block masks for 4-bit grid.
///
/// Uses `_mm_movemask_epi8` to extract 16-bit masks efficiently, then
/// scatters bits using `trailing_zeros` loop.
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

            // Load 16 packed bytes = 32 elements
            let bytes = _mm_loadu_si128(pb as *const __m128i);

            // Extract nibbles
            let low = _mm_and_si128(bytes, mask_lo_v);
            let high = _mm_and_si128(_mm_srli_epi16::<4>(bytes), mask_lo_v);

            // Interleave to element order
            let elems_lo = _mm_unpacklo_epi8(low, high); // elements 0-15
            let elems_hi = _mm_unpackhi_epi8(low, high); // elements 16-31

            // Compare > 0
            let nz_lo = _mm_cmpgt_epi8(elems_lo, zero);
            let nz_hi = _mm_cmpgt_epi8(elems_hi, zero);

            // Extract 16-bit masks
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

        // Scalar remainder
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
