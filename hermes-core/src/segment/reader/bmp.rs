//! BMP (Block-Max Pruning) index reader for sparse vectors.
//!
//! Reads the BMP blob from a `.sparse` file at load time and keeps all block
//! data pre-decoded in contiguous flat arrays for zero-allocation query execution.
//!
//! Uses **virtual coordinates**: `virtual_id = doc_id * num_ordinals + ordinal`.
//! Blocks are over virtual_ids. Postings are 2 bytes: `(local_slot, impact)`.
//!
//! Based on Mallia, Suel & Tonellotto (SIGIR 2024).

use crate::directories::FileHandle;

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

/// BMP index for a single sparse field.
///
/// All block data is pre-decoded at load time into flat contiguous arrays.
/// Query execution touches only these arrays — no file I/O or parsing per query.
///
/// Uses two-level pruning hierarchy (Carlson et al., SIGIR 2025):
/// 1. **Superblock grid**: coarse upper bounds over groups of `BMP_SUPERBLOCK_SIZE` blocks
/// 2. **Block grid**: fine-grained upper bounds per individual block
#[derive(Clone)]
pub struct BmpIndex {
    /// BMP block size (number of consecutive virtual_ids per block)
    pub bmp_block_size: u32,
    /// Number of blocks
    pub num_blocks: u32,
    /// Number of ordinals (virtual_id = doc_id * num_ordinals + ordinal)
    pub num_ordinals: u32,
    /// Global max weight scale factor (for dequantizing u8 impacts back to f32)
    pub max_weight_scale: f32,
    /// Total sparse vectors (from TOC entry)
    pub total_vectors: u32,

    // ── Pre-decoded block data (flat arrays) ──────────────────────────
    /// Per-block: index into `term_dim_ids` where this block's terms start.
    /// Length = num_blocks + 1 (sentinel at end).
    block_term_starts: Vec<u32>,
    /// Dimension ID for each term across all blocks (sorted per-block).
    term_dim_ids: Vec<u32>,
    /// Per-term: (posting_start, posting_count) into `postings` array.
    term_posting_ranges: Vec<(u32, u16)>,
    /// All postings across all blocks, contiguous.
    postings: Vec<BmpPosting>,

    // ── Block-max grid ────────────────────────────────────────────────
    /// Dimension IDs (sorted, for binary search)
    dim_ids: Vec<u32>,
    /// Block-max grid: grid[dim_idx * num_blocks + block_id] = max quantized impact (u8)
    grid: Vec<u8>,

    // ── Superblock grid (computed at load time) ───────────────────────
    /// sb_grid[dim_idx * num_superblocks + sb_id] = max impact across all blocks in superblock
    sb_grid: Vec<u8>,
    /// Number of superblocks
    pub num_superblocks: u32,
}

impl BmpIndex {
    /// Parse a BMP blob from the given file handle.
    ///
    /// Reads footer, block offset table, block-max grid, and **all block
    /// forward index data** into memory. After this call, query execution
    /// requires zero file I/O.
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
                "BMP blob too small for footer".into(),
            ));
        }

        // Read the footer (last 32 bytes of the blob)
        let footer_start = blob_offset + blob_len - BMP_BLOB_FOOTER_SIZE as u64;
        let footer_bytes = handle
            .read_bytes_range_sync(footer_start..footer_start + BMP_BLOB_FOOTER_SIZE as u64)
            .map_err(crate::Error::Io)?;
        let fb = footer_bytes.as_slice();

        let grid_offset = u32::from_le_bytes(fb[0..4].try_into().unwrap());
        let offsets_table_offset = u32::from_le_bytes(fb[4..8].try_into().unwrap());
        let bmp_block_size = u32::from_le_bytes(fb[8..12].try_into().unwrap());
        let num_blocks = u32::from_le_bytes(fb[12..16].try_into().unwrap());
        let num_dims = u32::from_le_bytes(fb[16..20].try_into().unwrap());
        let num_ordinals = u32::from_le_bytes(fb[20..24].try_into().unwrap());
        let max_weight_scale = f32::from_le_bytes(fb[24..28].try_into().unwrap());
        let magic = u32::from_le_bytes(fb[28..32].try_into().unwrap());

        if magic != BMP_BLOB_MAGIC {
            return Err(crate::Error::Corruption(format!(
                "Invalid BMP blob magic: {:#x} (expected {:#x})",
                magic, BMP_BLOB_MAGIC
            )));
        }

        // Read block offset table: [u32; num_blocks]
        let offsets_abs = blob_offset + offsets_table_offset as u64;
        let offsets_size = num_blocks as u64 * 4;
        let offsets_bytes = handle
            .read_bytes_range_sync(offsets_abs..offsets_abs + offsets_size)
            .map_err(crate::Error::Io)?;
        let mut block_offsets = Vec::with_capacity(num_blocks as usize);
        for i in 0..num_blocks as usize {
            block_offsets.push(u32::from_le_bytes(
                offsets_bytes.as_slice()[i * 4..(i + 1) * 4]
                    .try_into()
                    .unwrap(),
            ));
        }

        // Read block-max grid: dim_ids[u32; num_dims] + grid_data[u8; num_dims * num_blocks]
        let grid_abs = blob_offset + grid_offset as u64;
        let grid_header_size = num_dims as u64 * 4;
        let grid_data_size = num_dims as u64 * num_blocks as u64;
        let grid_total = grid_header_size + grid_data_size;
        let grid_bytes = handle
            .read_bytes_range_sync(grid_abs..grid_abs + grid_total)
            .map_err(crate::Error::Io)?;
        let gb = grid_bytes.as_slice();

        let mut dim_ids = Vec::with_capacity(num_dims as usize);
        for i in 0..num_dims as usize {
            dim_ids.push(u32::from_le_bytes(
                gb[i * 4..(i + 1) * 4].try_into().unwrap(),
            ));
        }

        let grid_start = num_dims as usize * 4;
        let grid = gb[grid_start..].to_vec();

        // ── Pre-decode ALL block forward index data ──────────────────
        // Read the entire block data section at once (from blob start to offsets_table_offset)
        if offsets_table_offset == 0 || num_blocks == 0 {
            return Ok(Self {
                bmp_block_size,
                num_blocks,
                num_ordinals,
                max_weight_scale,
                total_vectors,
                block_term_starts: vec![0],
                term_dim_ids: Vec::new(),
                term_posting_ranges: Vec::new(),
                postings: Vec::new(),
                dim_ids,
                grid,
                sb_grid: Vec::new(),
                num_superblocks: 0,
            });
        }

        let block_data_bytes = handle
            .read_bytes_range_sync(blob_offset..blob_offset + offsets_table_offset as u64)
            .map_err(crate::Error::Io)?;
        let block_data = block_data_bytes.as_slice();

        let mut block_term_starts: Vec<u32> = Vec::with_capacity(num_blocks as usize + 1);
        let mut term_dim_ids: Vec<u32> = Vec::new();
        let mut term_posting_ranges: Vec<(u32, u16)> = Vec::new();
        let mut all_postings: Vec<BmpPosting> = Vec::new();

        for block_id in 0..num_blocks as usize {
            block_term_starts.push(term_dim_ids.len() as u32);

            let start = block_offsets[block_id] as usize;
            let end = if block_id + 1 < num_blocks as usize {
                block_offsets[block_id + 1] as usize
            } else {
                offsets_table_offset as usize
            };

            if end <= start + 2 {
                continue;
            }

            let bdata = &block_data[start..end];
            let num_terms = u16::from_le_bytes(bdata[0..2].try_into().unwrap()) as usize;
            let mut pos = 2;

            for _ in 0..num_terms {
                let dim_id = u32::from_le_bytes(bdata[pos..pos + 4].try_into().unwrap());
                pos += 4;
                let num_postings = u16::from_le_bytes(bdata[pos..pos + 2].try_into().unwrap());
                pos += 2;

                let posting_start = all_postings.len() as u32;
                term_dim_ids.push(dim_id);
                term_posting_ranges.push((posting_start, num_postings));

                for _ in 0..num_postings {
                    let local_slot = bdata[pos];
                    let impact = bdata[pos + 1];
                    pos += 2;

                    all_postings.push(BmpPosting { local_slot, impact });
                }
            }
        }
        // Sentinel
        block_term_starts.push(term_dim_ids.len() as u32);

        // ── Compute superblock grid from block grid ──────────────────
        let num_superblocks = num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE);
        let mut sb_grid = vec![0u8; num_dims as usize * num_superblocks as usize];

        for dim_idx in 0..num_dims as usize {
            let row_start = dim_idx * num_blocks as usize;
            for sb in 0..num_superblocks as usize {
                let start = sb * BMP_SUPERBLOCK_SIZE as usize;
                let end = (start + BMP_SUPERBLOCK_SIZE as usize).min(num_blocks as usize);
                let mut max_val = 0u8;
                for b in start..end {
                    let v = grid[row_start + b];
                    if v > max_val {
                        max_val = v;
                    }
                }
                sb_grid[dim_idx * num_superblocks as usize + sb] = max_val;
            }
        }

        log::debug!(
            "BMP index loaded: num_blocks={}, num_superblocks={}, num_dims={}, bmp_block_size={}, \
             num_ordinals={}, max_weight_scale={:.4}, terms={}, postings={}",
            num_blocks,
            num_superblocks,
            num_dims,
            bmp_block_size,
            num_ordinals,
            max_weight_scale,
            term_dim_ids.len(),
            all_postings.len(),
        );

        Ok(Self {
            bmp_block_size,
            num_blocks,
            num_ordinals,
            max_weight_scale,
            total_vectors,
            block_term_starts,
            term_dim_ids,
            term_posting_ranges,
            postings: all_postings,
            dim_ids,
            grid,
            sb_grid,
            num_superblocks,
        })
    }

    /// Convert a virtual_id from the flat accumulator to (doc_id, ordinal).
    #[inline]
    pub fn virtual_to_doc(&self, virtual_id: u32) -> (u32, u16) {
        let doc_id = virtual_id / self.num_ordinals;
        let ordinal = (virtual_id % self.num_ordinals) as u16;
        (doc_id, ordinal)
    }

    /// Binary search for a dimension ID, returns its index in dim_ids.
    #[inline]
    pub fn find_dim_idx(&self, dim_id: u32) -> Option<usize> {
        self.dim_ids.binary_search(&dim_id).ok()
    }

    /// Compute upper bound scores for ALL superblocks in a single vectorized pass.
    ///
    /// For each resolved query dimension, streams through the contiguous sb_grid row
    /// (num_superblocks bytes) and accumulates weighted impacts into `out`.
    ///
    /// At 1M×5 ordinals: ~1.2K superblocks vs 78K blocks → ~65× fewer entries.
    ///
    /// `query_dims`: `&[(dim_idx, pre_scaled_weight)]` — weights must already include
    /// the `max_weight_scale / 255.0` factor.
    pub fn compute_superblock_ubs(&self, query_dims: &[(usize, f32)], out: &mut [f32]) {
        let nsb = self.num_superblocks as usize;
        debug_assert!(out.len() >= nsb);

        out[..nsb].fill(0.0);

        for &(dim_idx, weight) in query_dims {
            let row = &self.sb_grid[dim_idx * nsb..dim_idx * nsb + nsb];
            accumulate_u8_weighted(row, weight, &mut out[..nsb]);
        }
    }

    /// Compute block UBs for blocks `[block_start..block_end)` within one superblock.
    ///
    /// Only reads the grid slice for this superblock's blocks — fits in L1 cache
    /// (~1.9KB per dim for c=64 blocks). Called only for surviving superblocks
    /// after superblock-level pruning.
    pub fn compute_block_ubs_range(
        &self,
        query_dims: &[(usize, f32)],
        block_start: usize,
        block_end: usize,
        out: &mut [f32],
    ) {
        let count = block_end - block_start;
        debug_assert!(out.len() >= count);
        let nb = self.num_blocks as usize;

        out[..count].fill(0.0);

        for &(dim_idx, weight) in query_dims {
            let row_offset = dim_idx * nb + block_start;
            let row = &self.grid[row_offset..row_offset + count];
            accumulate_u8_weighted(row, weight, &mut out[..count]);
        }
    }

    /// Build per-block query-dim presence masks for blocks `[block_start..block_end)`.
    ///
    /// Same as `compute_block_masks` but scoped to a single superblock's blocks.
    /// L1-cache friendly: reads only the grid slice for this superblock.
    pub fn compute_block_masks_range(
        &self,
        query_dims: &[(usize, f32)],
        block_start: usize,
        block_end: usize,
        masks: &mut [u32],
    ) {
        let count = block_end - block_start;
        debug_assert!(masks.len() >= count);
        let nb = self.num_blocks as usize;

        masks[..count].fill(0);

        for (q, &(dim_idx, _weight)) in query_dims.iter().enumerate() {
            let row_offset = dim_idx * nb + block_start;
            let row = &self.grid[row_offset..row_offset + count];
            let bit = 1u32 << q;
            for b in 0..count {
                if unsafe { *row.get_unchecked(b) } > 0 {
                    unsafe { *masks.get_unchecked_mut(b) |= bit };
                }
            }
        }
    }

    // ── Query-time accessors for pre-decoded data ─────────────────────

    /// Get the term range for a block: [start..end) into `term_dim_ids` / `term_posting_ranges`.
    #[inline]
    pub fn block_term_range(&self, block_id: u32) -> (u32, u32) {
        let start = self.block_term_starts[block_id as usize];
        let end = self.block_term_starts[block_id as usize + 1];
        (start, end)
    }

    /// Get the sorted dim_id slice for a block's term range.
    /// Used for binary search during query scoring.
    #[inline]
    pub fn block_term_dim_ids(&self, term_start: u32, term_end: u32) -> &[u32] {
        &self.term_dim_ids[term_start as usize..term_end as usize]
    }

    /// Get the posting range for a term: (start, count) into `postings`.
    #[inline]
    pub fn term_postings(&self, term_idx: u32) -> &[BmpPosting] {
        let (start, count) = self.term_posting_ranges[term_idx as usize];
        &self.postings[start as usize..(start as usize + count as usize)]
    }

    // ── Prefetch support ─────────────────────────────────────────────

    /// Get a raw pointer to the first posting of the given block's first term.
    /// Returns `None` if the block has no terms. Used for software prefetching.
    #[inline]
    pub fn first_posting_ptr(&self, block_id: u32) -> Option<*const BmpPosting> {
        let (term_start, term_end) = self.block_term_range(block_id);
        if term_start >= term_end {
            return None;
        }
        let (posting_start, count) = self.term_posting_ranges[term_start as usize];
        if count == 0 {
            return None;
        }
        Some(unsafe { self.postings.as_ptr().add(posting_start as usize) })
    }

    // ── Extended prefetch support ───────────────────────────────────────

    /// Get a raw pointer to the term_dim_ids array at the given term index.
    /// Used for prefetching the next block's sorted dim list ahead of time.
    #[inline]
    pub fn term_dim_ids_ptr(&self, term_start: u32) -> *const u32 {
        unsafe { self.term_dim_ids.as_ptr().add(term_start as usize) }
    }

    /// Get a raw pointer to the block_term_starts array at the given block.
    /// Used for prefetching block metadata 2 blocks ahead.
    #[inline]
    pub fn block_term_starts_ptr(&self, block_id: u32) -> *const u32 {
        unsafe { self.block_term_starts.as_ptr().add(block_id as usize) }
    }

    /// Get dimension IDs.
    pub fn dim_ids(&self) -> &[u32] {
        &self.dim_ids
    }

    /// Number of unique dimensions.
    pub fn num_dimensions(&self) -> usize {
        self.dim_ids.len()
    }

    /// Total number of postings stored in the index.
    pub fn total_postings(&self) -> u64 {
        self.postings.len() as u64
    }

    /// Estimated memory usage in bytes.
    pub fn estimated_memory_bytes(&self) -> usize {
        let block_starts = self.block_term_starts.capacity() * 4;
        let term_dims = self.term_dim_ids.capacity() * 4;
        let term_ranges = self.term_posting_ranges.capacity() * std::mem::size_of::<(u32, u16)>();
        let postings = self.postings.capacity() * std::mem::size_of::<BmpPosting>();
        let dims = self.dim_ids.capacity() * 4;
        let grid = self.grid.capacity();
        let sb_grid = self.sb_grid.capacity();
        block_starts
            + term_dims
            + term_ranges
            + postings
            + dims
            + grid
            + sb_grid
            + std::mem::size_of::<Self>()
    }
}

// ============================================================================
// SIMD-accelerated helpers for BMP scoring
// ============================================================================

/// Accumulate `out[i] += input[i] as f32 * weight` for all i.
///
/// Uses NEON on aarch64, auto-vectorization hint on other platforms.
#[inline]
fn accumulate_u8_weighted(input: &[u8], weight: f32, out: &mut [f32]) {
    debug_assert_eq!(input.len(), out.len());
    let n = input.len();

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64
        unsafe { accumulate_u8_weighted_neon(input, weight, out, n) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Scalar fallback — auto-vectorizes well with -O2
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
