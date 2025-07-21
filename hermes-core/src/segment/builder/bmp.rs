//! BMP (Block-Max Pruning) index builder for sparse vectors — **V18 format**.
//!
//! Builds a block-at-a-time (BAAT) index using **compact virtual coordinates**:
//! sequential IDs are assigned to unique `(doc_id, ordinal)` pairs. A lookup
//! table enables query-time recovery of the original coordinates.
//!
//! Postings are 2 bytes each: `(local_slot: u8, impact: u8)`.
//!
//! Based on Mallia, Suel & Tonellotto (SIGIR 2024).
//!
//! ## Memory efficiency
//!
//! Uses a K-way merge over per-dim cursors with **streaming block writes**.
//! Each dim's postings are already sorted by `(doc_id, ordinal)` = sorted by
//! `compact_virtual_id` = sorted by `block_id`, so a min-heap merges them in
//! block order. Each block's data is serialized and written immediately —
//! no intermediate arrays for postings, dim_ids, or posting_starts.
//!
//! Peak memory: `input + grid_entries + O(num_blocks) block_data_starts`.
//!
//! ## BMP V18 Blob Layout (data-first, block-interleaved)
//!
//! ```text
//! Section B:  block_data         [per-block interleaved data]   variable-length
//!             padding            [0-7 bytes to 8-byte boundary]
//! Section A:  block_data_starts  [u64-LE × (num_blocks + 1)]   byte offsets into Section B
//! Section D:  compressed block grid       [random-access 256-cell groups]
//! Section E:  compressed superblock grid  [ceil-u4, random-access 256-cell groups]
//! Section H:  compressed coarse grid      [ceil-u4, one cell per 256 superblocks]
//! Section F:  doc_map_ids        [u32-LE × num_virtual_docs]
//! Section G:  doc_map_ordinals   [u16-LE × num_virtual_docs]
//!
//! Per-block data layout (for non-empty blocks):
//!   num_terms: u32                                    offset 0
//!   term_dim_ids: [u32-LE × num_terms]                offset 4
//!   posting_starts: [u32-LE × (num_terms + 1)]        relative cumulative counts
//!   term_max_impacts: [u8 × num_terms]                exact per-block maxima
//!   postings: [(u8, u8) × total_block_postings]       BmpPosting pairs
//!
//! BMP V18 Footer (80 bytes):
//!   total_terms: u64              //  0- 7  (stats only)
//!   total_postings: u64           //  8-15  (stats only)
//!   grid_offset: u64              // 16-23  (byte offset of Section D)
//!   sb_grid_offset: u64           // 24-31  (byte offset of Section E)
//!   coarse_grid_offset: u64       // 32-39  (byte offset of Section H)
//!   num_blocks: u32               // 40-43
//!   dims: u32                     // 44-47  (fixed vocabulary size)
//!   bmp_block_size: u32           // 48-51
//!   num_virtual_docs: u32         // 52-55  (= num_blocks × bmp_block_size, padded)
//!   max_weight_scale: f32         // 56-59
//!   doc_map_offset: u64           // 60-67  (byte offset of Section F)
//!   num_real_docs: u32            // 68-71  (actual vector count before padding)
//!   grid_bits: u32                // 72-75  (2 or 4)
//!   magic: u32                    // 76-79  (BMP8 = 0x38504D42)
//! ```

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use rustc_hash::FxHashMap;

/// O(1) virtual-ID lookup table, built from sorted `(doc_id, ordinal)` pairs.
///
/// Uses FxHashMap for O(1) amortised lookup instead of O(log N) binary search.
/// Memory: ~18 bytes per entry (~48 MB for 2.7M entries) — a speed/memory
/// trade-off that eliminates billions of binary search comparisons during
/// the K-way merge (267M postings × 21 comparisons = 5.6B comparisons).
pub(crate) struct VidLookup {
    map: FxHashMap<(crate::DocId, u16), u32>,
}

impl VidLookup {
    /// Build from sorted `(doc_id, ordinal)` pairs. The index in the sorted
    /// order becomes the virtual ID.
    pub fn from_sorted_pairs(vid_pairs: &[(crate::DocId, u16)]) -> Self {
        let mut map = FxHashMap::with_capacity_and_hasher(vid_pairs.len(), Default::default());
        for (vid, &pair) in vid_pairs.iter().enumerate() {
            map.insert(pair, vid as u32);
        }
        Self { map }
    }

    #[inline]
    pub fn get(&self, key: (crate::DocId, u16)) -> u32 {
        self.map[&key]
    }
}

use crate::DocId;
use crate::segment::bmp_grid::{
    CompressedGridLayout, GRID_GROUP_CELLS, LSP_SUPERBLOCK_GRID_BITS, bit_width, pack_group,
    quantize_block_maximum,
};
use crate::segment::format::{BMP_BLOB_FOOTER_SIZE, BMP_BLOB_MAGIC};
use crate::segment::reader::bmp::BMP_SUPERBLOCK_SIZE;

/// Build a BMP V18 blob from per-dimension postings.
///
/// **Takes ownership** of the postings HashMap. All per-dim Vecs are moved
/// out of the HashMap into `dim_vecs` before the K-way merge starts, and
/// the HashMap shell is dropped immediately. After the merge completes,
/// `dim_vecs` and `vid_lookup` are explicitly dropped before grid/doc_map write.
///
/// Uses compact virtual IDs: sequential IDs assigned to unique `(doc_id, ordinal)`
/// pairs, eliminating the sparse `doc_id * num_ordinals + ordinal` space.
///
/// **Streaming block writes**: the K-way merge writes each block's data directly
/// to the writer as it's produced, instead of collecting all postings into
/// intermediate arrays. This eliminates `postings_flat` (O(total_postings × 2)),
/// `term_dim_ids` (O(total_terms × 4)), and `term_posting_starts` (O(total_terms × 4))
/// — saving ~740 MB for 7M-doc segments with 50 dims.
///
/// Only `grid_entries` (O(total_terms × 12)) and `block_data_starts` (O(num_blocks × 8))
/// are buffered for later sections.
///
/// Block data uses u32 dim_id directly (not dim_idx). Grid has `dims` rows.
/// num_virtual_docs is padded to block_size alignment.
///
/// `bmp_block_size` is clamped to 1..=256 (u8 local_slot).
///
/// BP reorder is NOT done at build time — it's handled by the background
/// optimizer or explicit reorder command.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_bmp_blob(
    mut postings: FxHashMap<u32, Vec<(DocId, u16, f32)>>,
    bmp_block_size: u32,
    grid_bits: u8,
    weight_threshold: f32,
    pruning_fraction: Option<f32>,
    dims: u32,
    max_weight: f32,
    min_terms: usize,
    writer: &mut dyn Write,
) -> std::io::Result<u64> {
    if postings.is_empty() {
        return Ok(0);
    }

    // Phase 0: Prune per-dimension (skip dims with fewer than min_terms postings)
    for dim_postings in postings.values_mut() {
        if let Some(fraction) = pruning_fraction
            && dim_postings.len() >= min_terms
            && fraction < 1.0
        {
            dim_postings.sort_unstable_by(|a, b| {
                b.2.abs()
                    .partial_cmp(&a.2.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let keep = ((dim_postings.len() as f64 * fraction as f64).ceil() as usize).max(1);
            dim_postings.truncate(keep);
            dim_postings.sort_unstable_by_key(|(doc_id, ordinal, _)| (*doc_id, *ordinal));
        }
    }

    // Phase 1: Collect unique (doc_id, ordinal) pairs and track dominant dimension.
    // Single pass over all postings — dedup by FxHashMap keyed on (doc_id, ordinal).
    // Each entry also stores (max_impact, argmax_dim) for sorting.
    //
    // Weight threshold is skipped for dims with fewer than min_terms postings
    // to protect small dimensions from losing signal.
    //
    // Capacity: use the LARGEST single dim's posting count as a lower bound on
    // unique pairs (each doc appears at least once).
    let max_dim_postings: usize = postings.values().map(|v| v.len()).max().unwrap_or(0);
    // Collect unique (doc_id, ordinal) pairs with non-zero quantized impact.
    let mut vid_set: rustc_hash::FxHashSet<(DocId, u16)> =
        rustc_hash::FxHashSet::with_capacity_and_hasher(max_dim_postings, Default::default());

    for dim_postings in postings.values() {
        let skip_threshold = dim_postings.len() < min_terms;
        for &(doc_id, ordinal, weight) in dim_postings {
            let abs_w = weight.abs();
            if !skip_threshold && abs_w < weight_threshold {
                continue;
            }
            if quantize_weight(abs_w, max_weight) > 0 {
                vid_set.insert((doc_id, ordinal));
            }
        }
    }

    if vid_set.is_empty() {
        return Ok(0);
    }

    // max_weight_scale is fixed (= max_weight parameter)
    let max_weight_scale = max_weight;

    // Assign compact virtual IDs: sequential IDs for unique (doc_id, ordinal) pairs.
    let mut vid_pairs: Vec<(DocId, u16)> = vid_set.into_iter().collect();
    vid_pairs.sort_unstable();
    let num_real_docs = vid_pairs.len();
    if num_real_docs > u32::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "BMP real document count exceeds the V18 u32 format limit",
        ));
    }

    // Build O(1) lookup table from (doc_id, ordinal) → virtual_id.
    let vid_lookup = VidLookup::from_sorted_pairs(&vid_pairs);

    // Safety: local_slot is u8, so block_size must not exceed 256
    let effective_block_size = bmp_block_size.clamp(1, 256);

    // Pad only the final document block. Compressed-grid groups are
    // independently decodable, so merge can repack the at-most-two groups
    // crossing each source boundary while copying aligned interiors verbatim.
    let num_blocks = num_real_docs.div_ceil(effective_block_size as usize);
    let num_virtual_docs = num_blocks
        .checked_mul(effective_block_size as usize)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "BMP padded document count overflows usize",
            )
        })?;
    if num_virtual_docs > u32::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "BMP padded document count exceeds the V18 u32 format limit",
        ));
    }

    let mut dim_ids: Vec<u32> = postings.keys().copied().collect();
    dim_ids.sort_unstable();

    // The block-max grid only has rows for dim_id < dims; postings beyond
    // that would be written into block data but never into the grid, making
    // those dimensions silently unsearchable. Fail loud instead.
    if let Some(&max_dim) = dim_ids.last()
        && max_dim >= dims
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "BMP postings contain dim_id {max_dim} out of range for the configured \
                 dims={dims}: dimensions >= dims have no block-max grid row and can never \
                 match a query; raise `dims` in the field's sparse_vector config"
            ),
        ));
    }

    // Phase 2: K-way merge over per-dim cursors
    //
    // Take ownership of per-dim posting Vecs. This drains the HashMap so
    // its memory can be reclaimed by the allocator during the merge.
    let dim_vecs: Vec<Vec<(DocId, u16, f32)>> = dim_ids
        .iter()
        .map(|&d| postings.remove(&d).unwrap_or_default())
        .collect();
    drop(postings); // Free HashMap shell now

    // Borrow slices for the merge loop
    let dim_slices: Vec<&[(DocId, u16, f32)]> = dim_vecs.iter().map(|v| v.as_slice()).collect();

    // Per-dim flag: true means this dim has fewer than min_terms postings,
    // so weight_threshold should not be applied.
    let dim_skip_threshold: Vec<bool> = dim_slices.iter().map(|s| s.len() < min_terms).collect();

    // Per-dim cursor positions
    let num_dims = dim_ids.len();
    let mut cursors: Vec<usize> = vec![0; num_dims];

    // Min-heap: (block_id, dim_id, dim_idx)
    let mut heap: BinaryHeap<Reverse<(u32, u32, usize)>> = BinaryHeap::with_capacity(num_dims);

    let bs64 = effective_block_size as u64;

    // Initialize heap with first valid posting from each dim
    for (dim_idx, &dim_id) in dim_ids.iter().enumerate() {
        let posts = dim_slices[dim_idx];
        let skip_wt = dim_skip_threshold[dim_idx];
        for (pos, &(doc_id, ordinal, weight)) in posts.iter().enumerate() {
            let abs_w = weight.abs();
            if !skip_wt && abs_w < weight_threshold {
                continue;
            }
            let impact = quantize_weight(abs_w, max_weight_scale);
            if impact == 0 {
                continue;
            }
            let virtual_id = vid_lookup.get((doc_id, ordinal)) as u64;
            let block_id = (virtual_id / bs64) as u32;
            cursors[dim_idx] = pos;
            heap.push(Reverse((block_id, dim_id, dim_idx)));
            break;
        }
    }

    if heap.is_empty() {
        return Ok(0);
    }

    // ── Phase 2: K-way merge + streaming block write ──────────────────
    //
    // Write each block's data directly during the merge instead of collecting
    // into intermediate arrays. Eliminates postings_flat (O(total_postings × 2)),
    // term_dim_ids (O(total_terms × 4)), and term_posting_starts (O(total_terms × 4)).
    //
    // Only block_data_starts (O(num_blocks) = ~880 KB) and grid_entries
    // (O(total_terms) = ~66 MB) are buffered.
    let mut block_data_starts: Vec<u64> = Vec::with_capacity(num_blocks + 1);
    let mut grid_entries: Vec<(u32, u32, u8)> = Vec::new();
    let mut total_terms: u64 = 0;
    let mut total_postings: u64 = 0;
    let mut cumulative_bytes: u64 = 0;
    let mut last_block_filled: i64 = -1;

    // Per-block scratch (reused per block, bounded by one block's data ~4 KB)
    let mut blk_buf: Vec<u8> = Vec::with_capacity(4096);
    let mut blk_dim_ids: Vec<u32> = Vec::new();
    let mut blk_posting_counts: Vec<u32> = Vec::new();
    let mut blk_max_impacts: Vec<u8> = Vec::new();
    let mut blk_postings: Vec<u8> = Vec::new();

    while let Some(&Reverse((block_id, _, _))) = heap.peek() {
        // Fill block_data_starts for empty blocks before this one
        for _ in (last_block_filled + 1) as u32..block_id {
            block_data_starts.push(cumulative_bytes);
        }
        // This block's start offset
        block_data_starts.push(cumulative_bytes);
        last_block_filled = block_id as i64;

        // Clear per-block scratch
        blk_dim_ids.clear();
        blk_posting_counts.clear();
        blk_max_impacts.clear();
        blk_postings.clear();

        // Process all dims with postings in this block
        while let Some(&Reverse((bid, dim_id, dim_idx))) = heap.peek() {
            if bid != block_id {
                break;
            }
            heap.pop();

            let posts = dim_slices[dim_idx];
            let skip_wt = dim_skip_threshold[dim_idx];
            let mut pos = cursors[dim_idx];
            let mut max_impact = 0u8;
            let mut next_block: Option<u32> = None;
            let mut term_posting_count: u32 = 0;

            blk_dim_ids.push(dim_id);

            // Process all postings for this dim in this block
            while pos < posts.len() {
                let (doc_id, ordinal, weight) = posts[pos];
                let abs_w = weight.abs();
                if !skip_wt && abs_w < weight_threshold {
                    pos += 1;
                    continue;
                }
                let impact = quantize_weight(abs_w, max_weight_scale);
                if impact == 0 {
                    pos += 1;
                    continue;
                }

                let virtual_id = vid_lookup.get((doc_id, ordinal)) as u64;
                let bid2 = (virtual_id / bs64) as u32;
                if bid2 != block_id {
                    next_block = Some(bid2);
                    break;
                }

                let local_slot = (virtual_id % bs64) as u8;
                blk_postings.push(local_slot);
                blk_postings.push(impact);
                term_posting_count = term_posting_count.checked_add(1).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "BMP postings for one block/dimension exceed u32::MAX",
                    )
                })?;
                max_impact = max_impact.max(impact);
                pos += 1;
            }

            blk_posting_counts.push(term_posting_count);
            blk_max_impacts.push(max_impact);
            total_postings = total_postings.saturating_add(u64::from(term_posting_count));
            total_terms = total_terms.saturating_add(1);

            // Grid entry — indexed by dim_id directly
            grid_entries.push((dim_id, block_id, max_impact));

            // Advance cursor
            cursors[dim_idx] = pos;
            if let Some(nb) = next_block {
                heap.push(Reverse((nb, dim_id, dim_idx)));
            }
        }

        // Serialize and write this block's data directly to writer
        if !blk_dim_ids.is_empty() {
            blk_buf.clear();
            let nt = blk_dim_ids.len();

            // num_terms (u32)
            let nt_u32 = u32::try_from(nt).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "BMP block term count exceeds the V18 u32 format limit",
                )
            })?;
            blk_buf.extend_from_slice(&nt_u32.to_le_bytes());

            // term_dim_ids [u32 × nt]
            for &did in &blk_dim_ids {
                blk_buf.extend_from_slice(&did.to_le_bytes());
            }

            // posting_starts [u32 × (nt + 1)] — relative cumulative.
            // A 256-vector block with hundreds of dimensions can exceed
            // 65,535 postings, so both counts and prefixes are u32.
            let mut cum: u32 = 0;
            for &count in &blk_posting_counts {
                blk_buf.extend_from_slice(&cum.to_le_bytes());
                cum = cum.checked_add(count).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "BMP block posting prefix exceeds u32::MAX",
                    )
                })?;
            }
            blk_buf.extend_from_slice(&cum.to_le_bytes());

            // Exact per-term block maxima. The compressed D grid keeps its
            // existing ceil-u4 semantics; these exact values let reorder
            // rebuild tight ceil-u4 superblock maxima without rescanning postings.
            blk_buf.extend_from_slice(&blk_max_impacts);

            // postings [(u8, u8) × total_block_postings]
            blk_buf.extend_from_slice(&blk_postings);

            writer.write_all(&blk_buf)?;
            cumulative_bytes += blk_buf.len() as u64;
        }
    }

    // Fill remaining empty blocks
    for _ in (last_block_filled + 1) as u32..num_blocks as u32 {
        block_data_starts.push(cumulative_bytes);
    }
    // Sentinel
    block_data_starts.push(cumulative_bytes);

    // Sort grid entries by (dim_id, block_id) for streaming write
    grid_entries.sort_unstable();

    log::info!(
        "[bmp_build] V18 vectors={} padded={} blocks={} dims={} \
         terms={} postings={} grid_entries={}",
        num_real_docs,
        num_virtual_docs,
        num_blocks,
        dims,
        total_terms,
        total_postings,
        grid_entries.len(),
    );

    // Free K-way merge inputs — reclaims per-dim posting Vecs and vid lookup.
    drop(dim_slices); // borrows dim_vecs, must drop first
    drop(dim_vecs);
    drop(vid_lookup);

    // ── Write remaining sections ──────────────────────────────────────────
    let mut bytes_written: u64 = cumulative_bytes;

    // Padding to 8-byte boundary (for u64 alignment of Section A)
    let padding = (8 - (bytes_written % 8) as usize) % 8;
    if padding > 0 {
        writer.write_all(&[0u8; 8][..padding])?;
        bytes_written += padding as u64;
    }

    // Section A: block_data_starts [u64 × (num_blocks + 1)]
    bytes_written += write_u64_slice_le(writer, &block_data_starts)?;
    drop(block_data_starts);

    // Sections D+E+H: block, superblock, and coarse-superblock grids.
    // `dims` rows (not num_dims)
    let grid_offset = bytes_written;
    let (packed_bytes, sb_bytes, coarse_bytes) =
        stream_write_grids(&grid_entries, dims as usize, num_blocks, grid_bits, writer)?;
    let sb_grid_offset = bytes_written + packed_bytes;
    let coarse_grid_offset = sb_grid_offset + sb_bytes;
    bytes_written += packed_bytes + sb_bytes + coarse_bytes;
    drop(grid_entries); // Free grid entries before doc_map write

    // Section F: doc_map_ids [u32-LE × num_virtual_docs]
    // Real entries from vid_pairs, then padding entries (u32::MAX sentinel)
    let doc_map_offset = bytes_written;
    for &(doc_id, _) in &vid_pairs {
        writer.write_u32::<LittleEndian>(doc_id)?;
    }
    // Padding entries for block alignment
    for _ in num_real_docs..num_virtual_docs {
        writer.write_u32::<LittleEndian>(u32::MAX)?;
    }
    bytes_written += num_virtual_docs as u64 * 4;

    // Section G: doc_map_ordinals [u16-LE × num_virtual_docs]
    for &(_, ord) in &vid_pairs {
        writer.write_u16::<LittleEndian>(ord)?;
    }
    // Padding entries
    for _ in num_real_docs..num_virtual_docs {
        writer.write_u16::<LittleEndian>(0)?;
    }
    bytes_written += num_virtual_docs as u64 * 2;

    drop(vid_pairs); // Free after last use (~6 bytes × num_real_docs)

    // BMP V18 footer.
    write_bmp_footer(
        writer,
        total_terms,
        total_postings,
        grid_offset,
        sb_grid_offset,
        coarse_grid_offset,
        num_blocks as u32,
        dims,
        effective_block_size,
        num_virtual_docs as u32,
        max_weight_scale,
        doc_map_offset,
        num_real_docs as u32,
        grid_bits,
    )?;
    bytes_written += BMP_BLOB_FOOTER_SIZE as u64;

    Ok(bytes_written)
}

/// Write the BMP V18 footer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_bmp_footer(
    writer: &mut dyn Write,
    total_terms: u64,
    total_postings: u64,
    grid_offset: u64,
    sb_grid_offset: u64,
    coarse_grid_offset: u64,
    num_blocks: u32,
    dims: u32,
    bmp_block_size: u32,
    num_virtual_docs: u32,
    max_weight_scale: f32,
    doc_map_offset: u64,
    num_real_docs: u32,
    grid_bits: u8,
) -> std::io::Result<()> {
    writer.write_u64::<LittleEndian>(total_terms)?; //  0- 7
    writer.write_u64::<LittleEndian>(total_postings)?; //  8-15
    writer.write_u64::<LittleEndian>(grid_offset)?; // 16-23
    writer.write_u64::<LittleEndian>(sb_grid_offset)?; // 24-31
    writer.write_u64::<LittleEndian>(coarse_grid_offset)?; // 32-39
    writer.write_u32::<LittleEndian>(num_blocks)?; // 40-43
    writer.write_u32::<LittleEndian>(dims)?; // 44-47
    writer.write_u32::<LittleEndian>(bmp_block_size)?; // 48-51
    writer.write_u32::<LittleEndian>(num_virtual_docs)?; // 52-55
    writer.write_f32::<LittleEndian>(max_weight_scale)?; // 56-59
    writer.write_u64::<LittleEndian>(doc_map_offset)?; // 60-67
    writer.write_u32::<LittleEndian>(num_real_docs)?; // 68-71
    writer.write_u32::<LittleEndian>(grid_bits as u32)?; // 72-75
    writer.write_u32::<LittleEndian>(BMP_BLOB_MAGIC)?; // 76-79
    Ok(())
}

/// Bulk-write a `&[u64]` slice as little-endian bytes.
///
/// On little-endian platforms this is a single `write_all` (zero-copy cast).
/// On big-endian platforms, falls back to per-element byte-swap.
pub(crate) fn write_u64_slice_le(writer: &mut dyn Write, data: &[u64]) -> std::io::Result<u64> {
    if data.is_empty() {
        return Ok(0);
    }
    #[cfg(target_endian = "little")]
    {
        // SAFETY: u64 has no padding bytes, LE matches wire format.
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
        writer.write_all(bytes)?;
    }
    #[cfg(target_endian = "big")]
    {
        for &v in data {
            writer.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(data.len() as u64 * 8)
}

#[derive(Clone, Copy)]
enum GridProjection {
    Block { bits: u8 },
    Superblock,
    CoarseSuperblock,
}

impl GridProjection {
    #[inline]
    fn cells(self, num_blocks: usize) -> usize {
        match self {
            Self::Block { .. } => num_blocks,
            Self::Superblock => num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE as usize),
            Self::CoarseSuperblock => num_blocks.div_ceil(
                BMP_SUPERBLOCK_SIZE as usize
                    * crate::segment::reader::bmp::BMP_COARSE_SUPERBLOCKS as usize,
            ),
        }
    }

    #[inline]
    fn max_width(self) -> u8 {
        match self {
            Self::Block { bits } => bits,
            Self::Superblock | Self::CoarseSuperblock => LSP_SUPERBLOCK_GRID_BITS,
        }
    }

    #[inline]
    fn project(self, block: u32, impact: u8) -> (usize, u8) {
        match self {
            Self::Block { bits } => (block as usize, quantize_block_maximum(impact, bits)),
            Self::Superblock => (
                block as usize / BMP_SUPERBLOCK_SIZE as usize,
                quantize_block_maximum(impact, LSP_SUPERBLOCK_GRID_BITS),
            ),
            Self::CoarseSuperblock => (
                block as usize
                    / (BMP_SUPERBLOCK_SIZE as usize
                        * crate::segment::reader::bmp::BMP_COARSE_SUPERBLOCKS as usize),
                quantize_block_maximum(impact, LSP_SUPERBLOCK_GRID_BITS),
            ),
        }
    }
}

fn fill_row_widths(
    entries: &[(u32, u32, u8)],
    projection: GridProjection,
    widths: &mut [u8],
    cells: usize,
) -> std::io::Result<()> {
    widths.fill(0);
    for &(_, block, impact) in entries {
        let (cell, value) = projection.project(block, impact);
        if cell >= cells {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("BMP grid cell {cell} exceeds configured cell count {cells}"),
            ));
        }
        let group = cell / GRID_GROUP_CELLS;
        widths[group] = widths[group].max(bit_width(value));
    }
    Ok(())
}

fn write_compressed_row_payload(
    entries: &[(u32, u32, u8)],
    projection: GridProjection,
    widths: &[u8],
    cells: usize,
    writer: &mut dyn Write,
) -> std::io::Result<()> {
    let mut values = [0u8; GRID_GROUP_CELLS];
    let mut packed = [0u8; GRID_GROUP_CELLS];
    let mut entry = 0usize;
    for (group, &width) in widths.iter().enumerate() {
        values.fill(0);
        while entry < entries.len() {
            let (_, block, impact) = entries[entry];
            let (cell, value) = projection.project(block, impact);
            let entry_group = cell / GRID_GROUP_CELLS;
            if entry_group > group {
                break;
            }
            if entry_group < group || cell >= cells {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "BMP grid entries are not sorted by block within a dimension",
                ));
            }
            let slot = &mut values[cell % GRID_GROUP_CELLS];
            *slot = (*slot).max(value);
            entry += 1;
        }
        let payload_len = pack_group(&values, width, &mut packed)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        writer.write_all(&packed[..payload_len])?;
    }
    if entry != entries.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "BMP grid row contains entries beyond the final group",
        ));
    }
    Ok(())
}

fn write_compressed_grid_section(
    grid_entries: &[(u32, u32, u8)],
    num_dims: usize,
    num_blocks: usize,
    projection: GridProjection,
    writer: &mut dyn Write,
) -> std::io::Result<u64> {
    let cells = projection.cells(num_blocks);
    let layout = CompressedGridLayout::new(num_dims, cells);
    let mut widths = vec![0u8; layout.groups()];
    let mut row_sizes = Vec::with_capacity(num_dims);

    let mut entry = 0usize;
    for dim in 0..num_dims as u32 {
        let start = entry;
        while entry < grid_entries.len() && grid_entries[entry].0 == dim {
            entry += 1;
        }
        fill_row_widths(&grid_entries[start..entry], projection, &mut widths, cells)?;
        row_sizes.push(
            layout
                .row_bytes(&widths)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?,
        );
    }
    if entry != grid_entries.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "BMP grid entry dim_id {} exceeds configured dims={num_dims}",
                grid_entries[entry].0
            ),
        ));
    }

    let table_bytes = layout.write_row_offsets(&row_sizes, writer)?;
    entry = 0;
    for dim in 0..num_dims as u32 {
        let start = entry;
        while entry < grid_entries.len() && grid_entries[entry].0 == dim {
            entry += 1;
        }
        let row_entries = &grid_entries[start..entry];
        fill_row_widths(row_entries, projection, &mut widths, cells)?;
        layout.write_row_header(&widths, projection.max_width(), writer)?;
        write_compressed_row_payload(row_entries, projection, &widths, cells, writer)?;
    }
    Ok(table_bytes + row_sizes.into_iter().sum::<u64>())
}

/// Stream-write compressed ceil-quantized block (D), LSP/0 superblock (E),
/// and coarse-superblock (H) grids. All pruning levels use four bits by
/// default; an explicitly configured two-bit block grid does not reduce
/// either safe upper hierarchy.
///
/// `grid_entries` sorted by `(dim_id, block_id)`. `num_dims` is the fixed
/// vocabulary size (dims), so the grid has `num_dims` rows.
/// Each entry is `(dim_id, block_id, max_impact_u8)`.
///
/// Memory is bounded by one selector row plus two 256-byte group buffers.
/// Returns `(block_grid_bytes, superblock_grid_bytes, coarse_grid_bytes)`.
pub(crate) fn stream_write_grids(
    grid_entries: &[(u32, u32, u8)],
    num_dims: usize,
    num_blocks: usize,
    grid_bits: u8,
    writer: &mut dyn Write,
) -> std::io::Result<(u64, u64, u64)> {
    let block_bytes = write_compressed_grid_section(
        grid_entries,
        num_dims,
        num_blocks,
        GridProjection::Block { bits: grid_bits },
        writer,
    )?;
    let superblock_bytes = write_compressed_grid_section(
        grid_entries,
        num_dims,
        num_blocks,
        GridProjection::Superblock,
        writer,
    )?;
    let coarse_bytes = write_compressed_grid_section(
        grid_entries,
        num_dims,
        num_blocks,
        GridProjection::CoarseSuperblock,
        writer,
    )?;
    Ok((block_bytes, superblock_bytes, coarse_bytes))
}

/// Size of a single grid entry on disk: dim_id(4) + block_id(4) + impact(1) = 9 bytes.
const GRID_ENTRY_DISK_SIZE: usize = 9;

/// Reader for a sorted grid entry run file.
///
/// Each run file contains `(dim_id, block_id, max_impact)` triples sorted by
/// `(dim_id, block_id)`. Entries are 9 bytes each on disk.
#[cfg(feature = "native")]
pub(crate) struct GridRunReader {
    reader: std::io::BufReader<std::fs::File>,
    /// Peeked current entry (None when exhausted).
    pub current: Option<(u32, u32, u8)>,
}

#[cfg(feature = "native")]
impl GridRunReader {
    /// Open a run file and read the first entry.
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::with_capacity(256 * 1024, file);
        let current = Self::read_entry(&mut reader)?;
        Ok(Self { reader, current })
    }

    /// Read the next entry from the underlying file.
    fn read_entry(
        reader: &mut std::io::BufReader<std::fs::File>,
    ) -> std::io::Result<Option<(u32, u32, u8)>> {
        use std::io::Read;
        let mut buf = [0u8; GRID_ENTRY_DISK_SIZE];
        // EOF is valid only between complete records. `read_exact` alone
        // cannot distinguish an empty read from a truncated 1..8-byte tail.
        if reader.read(&mut buf[..1])? == 0 {
            return Ok(None);
        }
        reader.read_exact(&mut buf[1..])?;
        let dim_id = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let block_id = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let impact = buf[8];
        Ok(Some((dim_id, block_id, impact)))
    }

    /// Advance to the next entry.
    pub fn advance(&mut self) -> std::io::Result<()> {
        self.current = Self::read_entry(&mut self.reader)?;
        Ok(())
    }

    /// Seek back to the beginning and re-read the first entry.
    pub fn reset(&mut self) -> std::io::Result<()> {
        use std::io::Seek;
        self.reader.seek(std::io::SeekFrom::Start(0))?;
        self.current = Self::read_entry(&mut self.reader)?;
        Ok(())
    }
}

/// Write sorted grid entries to a run file on disk.
///
/// Entries must already be sorted by `(dim_id, block_id)`.
#[cfg(feature = "native")]
pub(crate) fn write_grid_run(
    entries: &[(u32, u32, u8)],
    path: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::BufWriter;
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::with_capacity(256 * 1024, file);
    let mut buf = [0u8; GRID_ENTRY_DISK_SIZE];
    for &(dim_id, block_id, impact) in entries {
        buf[0..4].copy_from_slice(&dim_id.to_le_bytes());
        buf[4..8].copy_from_slice(&block_id.to_le_bytes());
        buf[8] = impact;
        w.write_all(&buf)?;
    }
    w.flush()?;
    Ok(())
}

/// Merge sorted grid runs into one sorted run without materializing entries.
#[cfg(feature = "native")]
pub(crate) fn merge_grid_runs(
    input_paths: &[std::path::PathBuf],
    output_path: &std::path::Path,
) -> std::io::Result<()> {
    if let [input] = input_paths {
        let mut reader = std::io::BufReader::with_capacity(256 * 1024, std::fs::File::open(input)?);
        let mut writer =
            std::io::BufWriter::with_capacity(256 * 1024, std::fs::File::create(output_path)?);
        std::io::copy(&mut reader, &mut writer)?;
        return writer.flush();
    }
    let mut readers: Vec<GridRunReader> = input_paths
        .iter()
        .map(|path| GridRunReader::open(path))
        .collect::<std::io::Result<_>>()?;
    let output = std::fs::File::create(output_path)?;
    let mut writer = std::io::BufWriter::with_capacity(256 * 1024, output);
    let mut heap: BinaryHeap<Reverse<(u32, u32, u8, usize)>> =
        BinaryHeap::with_capacity(readers.len());
    for (run, reader) in readers.iter().enumerate() {
        if let Some((dimension, block, impact)) = reader.current {
            heap.push(Reverse((dimension, block, impact, run)));
        }
    }
    let mut buffer = [0u8; GRID_ENTRY_DISK_SIZE];
    while let Some(mut head) = heap.peek_mut() {
        let Reverse((dimension, block, impact, run)) = *head;
        buffer[0..4].copy_from_slice(&dimension.to_le_bytes());
        buffer[4..8].copy_from_slice(&block.to_le_bytes());
        buffer[8] = impact;
        writer.write_all(&buffer)?;
        let reader = &mut readers[run];
        reader.advance()?;
        if let Some((next_dimension, next_block, next_impact)) = reader.current {
            *head = Reverse((next_dimension, next_block, next_impact, run));
        } else {
            std::collections::binary_heap::PeekMut::pop(head);
        }
    }
    writer.flush()
}

#[cfg(feature = "native")]
struct MergedGridCursor<'a> {
    readers: &'a mut [GridRunReader],
    heap: BinaryHeap<Reverse<(u32, u32, u8, usize)>>,
}

#[cfg(feature = "native")]
impl<'a> MergedGridCursor<'a> {
    fn new(readers: &'a mut [GridRunReader]) -> Self {
        let mut heap = BinaryHeap::with_capacity(readers.len());
        for (run, reader) in readers.iter().enumerate() {
            if let Some((dimension, block, impact)) = reader.current {
                heap.push(Reverse((dimension, block, impact, run)));
            }
        }
        Self { readers, heap }
    }

    fn visit_dimension(
        &mut self,
        dimension: u32,
        mut visitor: impl FnMut(u32, u8) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        if self
            .heap
            .peek()
            .is_some_and(|Reverse((next, _, _, _))| *next < dimension)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "BMP external grid runs are not sorted by dimension",
            ));
        }
        while let Some(mut head) = self.heap.peek_mut() {
            let Reverse((next_dimension, block, impact, run)) = *head;
            if next_dimension != dimension {
                break;
            }
            visitor(block, impact)?;
            let reader = &mut self.readers[run];
            reader.advance()?;
            if let Some((next_dimension, next_block, next_impact)) = reader.current {
                *head = Reverse((next_dimension, next_block, next_impact, run));
            } else {
                std::collections::binary_heap::PeekMut::pop(head);
            }
        }
        Ok(())
    }

    fn finish(self, num_dims: usize) -> std::io::Result<()> {
        if let Some(Reverse((dimension, _, _, _))) = self.heap.peek() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("BMP grid run dimension {dimension} exceeds configured dims={num_dims}"),
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "native")]
struct ProjectedRowEncoder {
    projection: GridProjection,
    cells: usize,
    widths: Vec<u8>,
    touched_groups: Vec<usize>,
    payload: Vec<u8>,
    values: [u8; GRID_GROUP_CELLS],
    packed: [u8; GRID_GROUP_CELLS],
    current_group: Option<usize>,
    previous_cell: Option<usize>,
}

#[cfg(feature = "native")]
impl ProjectedRowEncoder {
    fn new(
        projection: GridProjection,
        cells: usize,
        groups: usize,
        payload_capacity: usize,
    ) -> Self {
        Self {
            projection,
            cells,
            widths: vec![0; groups],
            touched_groups: Vec::new(),
            payload: Vec::with_capacity(payload_capacity),
            values: [0; GRID_GROUP_CELLS],
            packed: [0; GRID_GROUP_CELLS],
            current_group: None,
            previous_cell: None,
        }
    }

    fn push(&mut self, block: u32, impact: u8) -> std::io::Result<()> {
        let (cell, value) = self.projection.project(block, impact);
        if cell >= self.cells {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "BMP grid cell {cell} exceeds configured cell count {}",
                    self.cells
                ),
            ));
        }
        if self.previous_cell.is_some_and(|previous| cell < previous) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "BMP external grid runs are not sorted by block",
            ));
        }
        let group = cell / GRID_GROUP_CELLS;
        if self.current_group != Some(group) {
            self.finish_current_group()?;
            self.values.fill(0);
            self.current_group = Some(group);
        }
        let slot = &mut self.values[cell % GRID_GROUP_CELLS];
        *slot = (*slot).max(value);
        self.previous_cell = Some(cell);
        Ok(())
    }

    fn finish_current_group(&mut self) -> std::io::Result<()> {
        let Some(group) = self.current_group else {
            return Ok(());
        };
        let maximum = self.values.iter().copied().max().unwrap_or(0);
        let width = bit_width(maximum);
        self.widths[group] = width;
        if width > 0 {
            self.touched_groups.push(group);
        }
        let payload_len = pack_group(&self.values, width, &mut self.packed)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        self.payload.extend_from_slice(&self.packed[..payload_len]);
        Ok(())
    }

    fn finish_row(&mut self) -> std::io::Result<()> {
        self.finish_current_group()?;
        Ok(())
    }

    fn reset(&mut self) {
        for group in self.touched_groups.drain(..) {
            self.widths[group] = 0;
        }
        self.payload.clear();
        self.current_group = None;
        self.previous_cell = None;
    }
}

#[cfg(feature = "native")]
struct MergedGridPlan {
    projection: GridProjection,
    cells: usize,
    layout: CompressedGridLayout,
    widths: Vec<u8>,
    touched_groups: Vec<usize>,
    payload_units: u64,
    row_sizes: Vec<u64>,
}

#[cfg(feature = "native")]
impl MergedGridPlan {
    fn new(projection: GridProjection, num_dims: usize, num_blocks: usize) -> Self {
        let cells = projection.cells(num_blocks);
        let layout = CompressedGridLayout::new(num_dims, cells);
        Self {
            projection,
            cells,
            layout,
            widths: vec![0; layout.groups()],
            touched_groups: Vec::new(),
            payload_units: 0,
            row_sizes: Vec::with_capacity(num_dims),
        }
    }

    fn observe(&mut self, block: u32, impact: u8) -> std::io::Result<()> {
        let (cell, value) = self.projection.project(block, impact);
        if cell >= self.cells {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "BMP grid cell {cell} exceeds configured cell count {}",
                    self.cells
                ),
            ));
        }
        let group = cell / GRID_GROUP_CELLS;
        let old_width = self.widths[group];
        let new_width = old_width.max(bit_width(value));
        if new_width != old_width {
            if old_width == 0 {
                self.touched_groups.push(group);
            }
            self.widths[group] = new_width;
            self.payload_units = self
                .payload_units
                .checked_add(u64::from(new_width - old_width))
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "BMP compressed-grid row size exceeds u64",
                    )
                })?;
        }
        Ok(())
    }

    fn finish_sizing_row(&mut self) -> std::io::Result<()> {
        let payload_bytes = self
            .payload_units
            .checked_mul((GRID_GROUP_CELLS / 8) as u64)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "BMP compressed-grid row size exceeds u64",
                )
            })?;
        let row_size = (self.layout.row_header_bytes() as u64)
            .checked_add(payload_bytes)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "BMP compressed-grid row size exceeds u64",
                )
            })?;
        self.row_sizes.push(row_size);
        for group in self.touched_groups.drain(..) {
            self.widths[group] = 0;
        }
        self.payload_units = 0;
        Ok(())
    }

    fn encoder(&self) -> std::io::Result<ProjectedRowEncoder> {
        let largest_row = self.row_sizes.iter().copied().max().unwrap_or(0);
        let largest_row = usize::try_from(largest_row).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "BMP compressed-grid row exceeds addressable memory",
            )
        })?;
        let payload_capacity = largest_row
            .checked_sub(self.layout.row_header_bytes())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "BMP compressed-grid row is shorter than its header",
                )
            })?;
        Ok(ProjectedRowEncoder::new(
            self.projection,
            self.cells,
            self.layout.groups(),
            payload_capacity,
        ))
    }

    fn write_row(
        &self,
        dimension: usize,
        row: &mut ProjectedRowEncoder,
        writer: &mut dyn Write,
    ) -> std::io::Result<()> {
        let expected_row_size = usize::try_from(self.row_sizes[dimension]).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "BMP compressed-grid row exceeds addressable memory",
            )
        })?;
        let expected_payload = expected_row_size
            .checked_sub(self.layout.row_header_bytes())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "BMP compressed-grid row is shorter than its header",
                )
            })?;
        row.finish_row()?;
        if row.payload.len() != expected_payload {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "BMP compressed-grid row size changed between sizing and encoding",
            ));
        }
        self.layout
            .write_row_header(&row.widths, self.projection.max_width(), writer)?;
        writer.write_all(&row.payload)?;
        row.reset();
        Ok(())
    }

    fn rows_bytes(&self) -> std::io::Result<u64> {
        self.row_sizes.iter().try_fold(0u64, |total, &size| {
            total.checked_add(size).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "BMP compressed-grid section exceeds u64",
                )
            })
        })
    }
}

#[cfg(feature = "native")]
fn copy_grid_spool(
    path: &std::path::Path,
    expected_bytes: u64,
    writer: &mut dyn Write,
) -> std::io::Result<()> {
    use std::io::Read;

    let file = std::fs::File::open(path)?;
    let actual_bytes = file.metadata()?.len();
    if actual_bytes != expected_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "BMP compressed-grid spool size changed: expected {expected_bytes}, got {actual_bytes}"
            ),
        ));
    }
    let mut reader = std::io::BufReader::with_capacity(1024 * 1024, file);
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut copied = 0u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        writer.write_all(&buffer[..read])?;
        copied = copied.checked_add(read as u64).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "BMP compressed-grid spool copy exceeds u64",
            )
        })?;
    }
    if copied != expected_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "short BMP compressed-grid spool copy: expected {expected_bytes}, copied {copied}"
            ),
        ));
    }
    Ok(())
}

/// Stream-write compressed grids from multiple sorted run files.
///
/// The complete D/E/H hierarchy uses one sizing pass and one encoding pass
/// over the external runs. D rows stream directly to the output; E/H rows are
/// temporarily spooled because their offset tables follow the complete D
/// section. Encoding buffers at most one compressed row per projection.
#[cfg(feature = "native")]
pub(crate) fn stream_write_grids_merged(
    run_readers: &mut [GridRunReader],
    num_dims: usize,
    num_blocks: usize,
    grid_bits: u8,
    superblock_spool: &std::path::Path,
    coarse_spool: &std::path::Path,
    writer: &mut dyn Write,
) -> std::io::Result<(u64, u64, u64)> {
    let num_dims_u32 = u32::try_from(num_dims).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "BMP grid dimensions exceed u32::MAX",
        )
    })?;
    let mut plans = [
        MergedGridPlan::new(
            GridProjection::Block { bits: grid_bits },
            num_dims,
            num_blocks,
        ),
        MergedGridPlan::new(GridProjection::Superblock, num_dims, num_blocks),
        MergedGridPlan::new(GridProjection::CoarseSuperblock, num_dims, num_blocks),
    ];

    {
        let mut merged = MergedGridCursor::new(run_readers);
        for dimension in 0..num_dims_u32 {
            merged.visit_dimension(dimension, |block, impact| {
                for plan in &mut plans {
                    plan.observe(block, impact)?;
                }
                Ok(())
            })?;
            for plan in &mut plans {
                plan.finish_sizing_row()?;
            }
        }
        merged.finish(num_dims)?;
    }
    for reader in run_readers.iter_mut() {
        reader.reset()?;
    }

    let block_table_bytes = plans[0]
        .layout
        .write_row_offsets(&plans[0].row_sizes, writer)?;
    let superblock_file = std::fs::File::create(superblock_spool)?;
    let coarse_file = std::fs::File::create(coarse_spool)?;
    let mut superblock_writer = std::io::BufWriter::with_capacity(1024 * 1024, superblock_file);
    let mut coarse_writer = std::io::BufWriter::with_capacity(1024 * 1024, coarse_file);
    let mut rows = plans
        .iter()
        .map(MergedGridPlan::encoder)
        .collect::<std::io::Result<Vec<_>>>()?;

    {
        let mut merged = MergedGridCursor::new(run_readers);
        for dimension in 0..num_dims_u32 {
            let dimension_index = dimension as usize;
            merged.visit_dimension(dimension, |block, impact| {
                for row in &mut rows {
                    row.push(block, impact)?;
                }
                Ok(())
            })?;
            plans[0].write_row(dimension_index, &mut rows[0], writer)?;
            plans[1].write_row(dimension_index, &mut rows[1], &mut superblock_writer)?;
            plans[2].write_row(dimension_index, &mut rows[2], &mut coarse_writer)?;
        }
        merged.finish(num_dims)?;
    }
    superblock_writer.flush()?;
    coarse_writer.flush()?;
    drop(superblock_writer);
    drop(coarse_writer);

    let block_rows_bytes = plans[0].rows_bytes()?;
    let superblock_rows_bytes = plans[1].rows_bytes()?;
    let coarse_rows_bytes = plans[2].rows_bytes()?;
    let superblock_table_bytes = plans[1]
        .layout
        .write_row_offsets(&plans[1].row_sizes, writer)?;
    copy_grid_spool(superblock_spool, superblock_rows_bytes, writer)?;
    let coarse_table_bytes = plans[2]
        .layout
        .write_row_offsets(&plans[2].row_sizes, writer)?;
    copy_grid_spool(coarse_spool, coarse_rows_bytes, writer)?;

    let block_bytes = block_table_bytes
        .checked_add(block_rows_bytes)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "BMP block-grid section exceeds u64",
            )
        })?;
    let superblock_bytes = superblock_table_bytes
        .checked_add(superblock_rows_bytes)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "BMP superblock-grid section exceeds u64",
            )
        })?;
    let coarse_bytes = coarse_table_bytes
        .checked_add(coarse_rows_bytes)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "BMP coarse-grid section exceeds u64",
            )
        })?;
    Ok((block_bytes, superblock_bytes, coarse_bytes))
}

/// Quantize a weight to u8 (0-255) given the global max scale.
#[inline]
fn quantize_weight(weight: f32, max_scale: f32) -> u8 {
    if max_scale <= 0.0 {
        return 0;
    }
    let normalized = (weight / max_scale * 255.0).round();
    normalized.clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_weight() {
        assert_eq!(quantize_weight(1.0, 1.0), 255);
        assert_eq!(quantize_weight(0.5, 1.0), 128);
        assert_eq!(quantize_weight(0.0, 1.0), 0);
        assert_eq!(quantize_weight(1.0, 2.0), 128);
    }

    #[test]
    fn coarse_grid_is_exact_max_projection_of_superblock_grid() {
        use crate::directories::OwnedBytes;
        use crate::segment::bmp_grid::{CompressedGrid, GRID_GROUP_CELLS};

        let num_blocks = BMP_SUPERBLOCK_SIZE as usize * GRID_GROUP_CELLS + 1;
        let entries = vec![
            (0, 0, 100),
            (0, num_blocks as u32 - 1, 200),
            (1, num_blocks as u32 - 2, 150),
        ];
        let mut encoded = Vec::new();
        let (block_bytes, superblock_bytes, coarse_bytes) =
            stream_write_grids(&entries, 2, num_blocks, 4, &mut encoded).unwrap();
        assert_eq!(
            encoded.len() as u64,
            block_bytes + superblock_bytes + coarse_bytes
        );

        let bytes = OwnedBytes::new(encoded);
        let superblocks = num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE as usize);
        let coarse_cells = superblocks.div_ceil(GRID_GROUP_CELLS);
        let superblock = CompressedGrid::parse(
            bytes.slice(block_bytes as usize..(block_bytes + superblock_bytes) as usize),
            2,
            superblocks,
            4,
            "test E",
        )
        .unwrap();
        let coarse = CompressedGrid::parse(
            bytes.slice(
                (block_bytes + superblock_bytes) as usize
                    ..(block_bytes + superblock_bytes + coarse_bytes) as usize,
            ),
            2,
            coarse_cells,
            4,
            "test H",
        )
        .unwrap();

        for dimension in 0..2 {
            let mut e = vec![0u8; superblocks];
            let mut decoded = [0u8; GRID_GROUP_CELLS];
            superblock
                .try_for_each_row_group(dimension, |group, packed| {
                    let start = group * GRID_GROUP_CELLS;
                    let count = GRID_GROUP_CELLS.min(superblocks - start);
                    packed.decode(0, count, &mut decoded);
                    e[start..start + count].copy_from_slice(&decoded[..count]);
                    Ok(())
                })
                .unwrap();
            let mut h = vec![0u8; coarse_cells];
            coarse
                .group(dimension, 0)
                .unwrap()
                .decode(0, coarse_cells, &mut h);
            let expected: Vec<_> = e
                .chunks(GRID_GROUP_CELLS)
                .map(|group| group.iter().copied().max().unwrap_or(0))
                .collect();
            assert_eq!(h, expected);
        }
    }

    #[test]
    fn bmp_footer_preserves_u64_statistics() {
        let total_terms = u32::MAX as u64 + 17;
        let total_postings = u32::MAX as u64 + 29;
        let mut footer = Vec::new();

        write_bmp_footer(
            &mut footer,
            total_terms,
            total_postings,
            11,
            22,
            25,
            33,
            44,
            32,
            55,
            6.0,
            66,
            77,
            4,
        )
        .unwrap();

        assert_eq!(footer.len(), BMP_BLOB_FOOTER_SIZE);
        assert_eq!(
            u64::from_le_bytes(footer[0..8].try_into().unwrap()),
            total_terms
        );
        assert_eq!(
            u64::from_le_bytes(footer[8..16].try_into().unwrap()),
            total_postings
        );
    }

    #[test]
    fn test_build_bmp_blob_empty() {
        let postings = FxHashMap::default();
        let mut buf = Vec::new();
        let size = build_bmp_blob(postings, 64, 4, 0.0, None, 105879, 5.0, 4, &mut buf).unwrap();
        assert_eq!(size, 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_build_bmp_blob_basic() {
        let mut postings = FxHashMap::default();
        // dim 0: doc 0 with weight 1.0, doc 1 with weight 0.5
        postings.insert(0u32, vec![(0u32, 0u16, 1.0f32), (1, 0, 0.5)]);
        // dim 1: doc 0 with weight 0.8
        postings.insert(1, vec![(0, 0, 0.8)]);

        let mut buf = Vec::new();
        let size = build_bmp_blob(postings, 64, 4, 0.0, None, 105879, 5.0, 4, &mut buf).unwrap();
        assert!(size > 0);
        assert_eq!(buf.len(), size as usize);

        // Verify the current footer magic.
        let footer_start = buf.len() - 4;
        let magic = u32::from_le_bytes(buf[footer_start..].try_into().unwrap());
        assert_eq!(magic, BMP_BLOB_MAGIC);
    }

    #[test]
    fn test_build_bmp_blob_rejects_dim_id_out_of_range() {
        // The grid only has rows for dim_id < dims; postings beyond that were
        // silently dropped from the grid (unsearchable). Must fail loud.
        let mut postings = FxHashMap::default();
        postings.insert(2u32, vec![(0u32, 0u16, 1.0f32)]);
        postings.insert(7u32, vec![(1u32, 0u16, 0.5f32)]); // >= dims (4)

        let mut buf = Vec::new();
        let err = build_bmp_blob(postings, 64, 4, 0.0, None, 4, 5.0, 4, &mut buf)
            .expect_err("dim_id >= dims must be rejected at build time");
        let msg = err.to_string();
        assert!(msg.contains('7'), "error must name the dim_id: {msg}");
        assert!(
            msg.contains('4'),
            "error must name the configured dims: {msg}"
        );
    }

    #[test]
    fn test_build_bmp_blob_multi_ordinal() {
        let mut postings = FxHashMap::default();
        // dim 0: doc 0 ord 0, doc 0 ord 1, doc 1 ord 0
        postings.insert(0u32, vec![(0u32, 0u16, 1.0f32), (0, 1, 0.8), (1, 0, 0.5)]);

        let mut buf = Vec::new();
        let size = build_bmp_blob(postings, 64, 4, 0.0, None, 105879, 5.0, 4, &mut buf).unwrap();
        assert!(size > 0);

        // num_virtual_docs should be 64 (padded to block_size)
        // 3 real docs padded to 64
        let footer_start = buf.len() - BMP_BLOB_FOOTER_SIZE;
        let fb = &buf[footer_start..];
        let num_virtual_docs = u32::from_le_bytes(fb[52..56].try_into().unwrap());
        assert_eq!(num_virtual_docs, 64); // padded to block_size

        // num_real_docs should be 3
        let num_real_docs = u32::from_le_bytes(fb[68..72].try_into().unwrap());
        assert_eq!(num_real_docs, 3);
    }

    #[test]
    fn test_build_bmp_blob_fixed_scale() {
        let mut postings = FxHashMap::default();
        // dim 0: doc 0 with weight 2.0, doc 1 with weight 1.0
        postings.insert(0u32, vec![(0u32, 0u16, 2.0f32), (1, 0, 1.0)]);

        // max_weight=5.0: max_weight_scale = 5.0
        // impact(2.0) = round(2.0/5.0*255) = round(102) = 102
        // impact(1.0) = round(1.0/5.0*255) = round(51) = 51
        let mut buf = Vec::new();
        let size = build_bmp_blob(postings, 64, 4, 0.0, None, 105879, 5.0, 4, &mut buf).unwrap();
        assert!(size > 0);

        // Verify max_weight_scale in the footer.
        let footer_start = buf.len() - BMP_BLOB_FOOTER_SIZE;
        let fb = &buf[footer_start..];
        let scale = f32::from_le_bytes(fb[56..60].try_into().unwrap());
        assert!((scale - 5.0).abs() < 0.001, "scale={}, expected 5.0", scale);
    }

    #[cfg(feature = "native")]
    #[test]
    fn grid_run_reader_rejects_a_truncated_final_record() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("grid-run");
        std::fs::write(&path, [0u8; GRID_ENTRY_DISK_SIZE + 1]).unwrap();

        let mut reader = GridRunReader::open(&path).unwrap();
        let error = reader
            .advance()
            .expect_err("a partial grid record must not be treated as clean EOF");
        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[cfg(feature = "native")]
    #[test]
    fn merged_grid_runs_are_byte_identical_to_in_memory_encoding() {
        let directory = tempfile::tempdir().unwrap();
        let mut entries = vec![
            (0, 0, 1),
            (0, 63, 17),
            (0, 64, 31),
            (0, 64, 200), // duplicate cell across runs: maximum must win
            (0, 16_383, 127),
            (0, 16_384, 255),
            // Dimension 1 intentionally empty.
            (2, 1, 9),
            (2, 64, 66),
            (2, 16_384, 129),
            (3, 16_383, 254),
        ];
        entries.sort_unstable();
        let mut runs = [Vec::new(), Vec::new(), Vec::new()];
        for (index, &entry) in entries.iter().enumerate() {
            runs[index % runs.len()].push(entry);
        }
        let run_paths: Vec<_> = runs
            .iter()
            .enumerate()
            .map(|(index, run)| {
                let path = directory.path().join(format!("run-{index}"));
                write_grid_run(run, &path).unwrap();
                path
            })
            .collect();

        for grid_bits in [2, 4] {
            let mut expected = Vec::new();
            let expected_sizes =
                stream_write_grids(&entries, 4, 16_385, grid_bits, &mut expected).unwrap();
            let mut readers: Vec<_> = run_paths
                .iter()
                .map(|path| GridRunReader::open(path).unwrap())
                .collect();
            let superblock_spool = directory.path().join(format!("sb-{grid_bits}"));
            let coarse_spool = directory.path().join(format!("coarse-{grid_bits}"));
            let mut actual = Vec::new();
            let actual_sizes = stream_write_grids_merged(
                &mut readers,
                4,
                16_385,
                grid_bits,
                &superblock_spool,
                &coarse_spool,
                &mut actual,
            )
            .unwrap();

            assert_eq!(actual_sizes, expected_sizes);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_fixed_scale_across_segments() {
        // Two segments with different max weights should produce the same
        // max_weight_scale with fixed max_weight.

        // Segment A: max weight = 3.0
        let mut postings_a = FxHashMap::default();
        postings_a.insert(0u32, vec![(0u32, 0u16, 3.0f32), (1, 0, 1.5)]);

        // Segment B: max weight = 1.0
        let mut postings_b = FxHashMap::default();
        postings_b.insert(0u32, vec![(0u32, 0u16, 1.0f32), (1, 0, 0.5)]);

        // Fixed max_weight=5.0: same scale
        let mut buf_a = Vec::new();
        build_bmp_blob(postings_a, 64, 4, 0.0, None, 105879, 5.0, 4, &mut buf_a).unwrap();
        let footer_a = buf_a.len() - BMP_BLOB_FOOTER_SIZE;
        let scale_a = f32::from_le_bytes(buf_a[footer_a + 56..footer_a + 60].try_into().unwrap());

        let mut buf_b = Vec::new();
        build_bmp_blob(postings_b, 64, 4, 0.0, None, 105879, 5.0, 4, &mut buf_b).unwrap();
        let footer_b = buf_b.len() - BMP_BLOB_FOOTER_SIZE;
        let scale_b = f32::from_le_bytes(buf_b[footer_b + 56..footer_b + 60].try_into().unwrap());
        assert_eq!(
            scale_a, scale_b,
            "Fixed max_weight scales must be identical"
        );
        assert!((scale_a - 5.0).abs() < 0.001);
    }
}
