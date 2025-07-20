//! BMP (Block-Max Pruning) index builder for sparse vectors — **V15 format**.
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
//! ## V13 changes from V12
//!
//! - **Recursive Graph Bisection (BP)** replaces SimHash for document ordering
//! - **Section H removed**: no per-block SimHash (BP needs no stored metadata)
//! - **64-byte footer** (was 72): `block_simhash_offset` field removed
//! - V12 segments are incompatible — must rebuild
//!
//! ## BMP V15 Blob Layout (data-first, block-interleaved)
//!
//! ```text
//! Section B:  block_data         [per-block interleaved data]   variable-length
//!             padding            [0-7 bytes to 8-byte boundary]
//! Section A:  block_data_starts  [u64-LE × (num_blocks + 1)]   byte offsets into Section B
//! Section D:  grid_packed4       [u8 × (dims × packed_row_size)]  ← indexed by dim_id directly
//! Section E:  sb_grid            [u8 × (dims × num_superblocks)]  ← indexed by dim_id directly
//! Section F:  doc_map_ids        [u32-LE × num_virtual_docs]
//! Section G:  doc_map_ordinals   [u16-LE × num_virtual_docs]
//!
//! Per-block data layout (for non-empty blocks):
//!   num_terms: u32                                    offset 0
//!   term_dim_ids: [u32-LE × num_terms]                offset 4
//!   posting_starts: [u32-LE × (num_terms + 1)]        relative cumulative counts
//!   postings: [(u8, u8) × total_block_postings]       BmpPosting pairs
//!
//! BMP V15 Footer (72 bytes):
//!   total_terms: u64              //  0- 7  (stats only)
//!   total_postings: u64           //  8-15  (stats only)
//!   grid_offset: u64              // 16-23  (byte offset of Section D)
//!   sb_grid_offset: u64           // 24-31  (byte offset of Section E)
//!   num_blocks: u32               // 32-35
//!   dims: u32                     // 36-39  (fixed vocabulary size)
//!   bmp_block_size: u32           // 40-43
//!   num_virtual_docs: u32         // 44-47  (= num_blocks × bmp_block_size, padded)
//!   max_weight_scale: f32         // 48-51
//!   doc_map_offset: u64           // 52-59  (byte offset of Section F)
//!   num_real_docs: u32            // 60-63  (actual vector count before padding)
//!   grid_bits: u32                // 64-67  (2 or 4)
//!   magic: u32                    // 68-71  (BMP5 = 0x35504D42)
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
use crate::segment::format::{BMP_BLOB_FOOTER_SIZE, BMP_BLOB_MAGIC};
use crate::segment::reader::bmp::BMP_SUPERBLOCK_SIZE;

/// Build a BMP V15 blob from per-dimension postings.
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
            "BMP real document count exceeds the V15 u32 format limit",
        ));
    }

    // Build O(1) lookup table from (doc_id, ordinal) → virtual_id.
    let vid_lookup = VidLookup::from_sorted_pairs(&vid_pairs);

    // Safety: local_slot is u8, so block_size must not exceed 256
    let effective_block_size = bmp_block_size.clamp(1, 256);

    // Pad num_virtual_docs to block_size alignment
    let num_virtual_docs = num_real_docs
        .div_ceil(effective_block_size as usize)
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
            "BMP padded document count exceeds the V15 u32 format limit",
        ));
    }
    let num_blocks = num_virtual_docs / effective_block_size as usize;

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
            total_postings = total_postings.saturating_add(u64::from(term_posting_count));
            total_terms = total_terms.saturating_add(1);

            // Grid entry — indexed by dim_id directly
            grid_entries.push((dim_id, block_id, max_impact));

            // Advance cursor
            if let Some(nb) = next_block {
                cursors[dim_idx] = pos;
                heap.push(Reverse((nb, dim_id, dim_idx)));
            } else {
                cursors[dim_idx] = pos;
                while pos < posts.len() {
                    let (doc_id, ordinal, weight) = posts[pos];
                    let abs_w = weight.abs();
                    if skip_wt || abs_w >= weight_threshold {
                        let impact = quantize_weight(abs_w, max_weight_scale);
                        if impact > 0 {
                            let virtual_id = vid_lookup.get((doc_id, ordinal)) as u64;
                            let nb = (virtual_id / bs64) as u32;
                            cursors[dim_idx] = pos;
                            heap.push(Reverse((nb, dim_id, dim_idx)));
                            break;
                        }
                    }
                    pos += 1;
                }
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
                    "BMP block term count exceeds the V15 u32 format limit",
                )
            })?;
            blk_buf.extend_from_slice(&nt_u32.to_le_bytes());

            // term_dim_ids [u32 × nt]
            for &did in &blk_dim_ids {
                blk_buf.extend_from_slice(&did.to_le_bytes());
            }

            // posting_starts [u32 × (nt + 1)] — relative cumulative.
            // u32, not u16: a 256-doc block of ~300-dim docs exceeds 65,535
            // postings, and the old u16 sums wrapped silently (V13 → V14).
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
        "[bmp_build] V15 vectors={} padded={} blocks={} dims={} \
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

    // Sections D+E: packed grid + sb_grid (streaming from sparse grid entries)
    // `dims` rows (not num_dims)
    let grid_offset = bytes_written;
    let (packed_bytes, sb_bytes) =
        stream_write_grids(&grid_entries, dims as usize, num_blocks, grid_bits, writer)?;
    let sb_grid_offset = bytes_written + packed_bytes;
    bytes_written += packed_bytes + sb_bytes;
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

    // BMP V15 footer.
    write_bmp_footer(
        writer,
        total_terms,
        total_postings,
        grid_offset,
        sb_grid_offset,
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

/// Write the BMP V15 footer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_bmp_footer(
    writer: &mut dyn Write,
    total_terms: u64,
    total_postings: u64,
    grid_offset: u64,
    sb_grid_offset: u64,
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
    writer.write_u32::<LittleEndian>(num_blocks)?; // 32-35
    writer.write_u32::<LittleEndian>(dims)?; // 36-39
    writer.write_u32::<LittleEndian>(bmp_block_size)?; // 40-43
    writer.write_u32::<LittleEndian>(num_virtual_docs)?; // 44-47
    writer.write_f32::<LittleEndian>(max_weight_scale)?; // 48-51
    writer.write_u64::<LittleEndian>(doc_map_offset)?; // 52-59
    writer.write_u32::<LittleEndian>(num_real_docs)?; // 60-63
    writer.write_u32::<LittleEndian>(grid_bits as u32)?; // 64-67
    writer.write_u32::<LittleEndian>(BMP_BLOB_MAGIC)?; // 68-71
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

/// Bytes per grid row for `num_blocks` cells at `bits` per cell (2 or 4).
#[inline]
pub(crate) fn grid_packed_row_size(num_blocks: usize, grid_bits: u8) -> usize {
    match grid_bits {
        2 => num_blocks.div_ceil(4),
        _ => num_blocks.div_ceil(2),
    }
}

/// Dequantization multiplier for a grid cell: `cell × scale` recovers a safe
/// (ceil-quantized) u8 upper bound. 4-bit → 17 (15×17=255); 2-bit → 85.
#[inline]
pub(crate) fn grid_dequant_scale(grid_bits: u8) -> u32 {
    match grid_bits {
        2 => 85,
        _ => 17,
    }
}

/// Ceiling quantize u8 to a `bits`-wide grid cell. Guarantees
/// `cell × grid_dequant_scale(bits) >= original` (bounds never underestimate).
#[inline]
pub(crate) fn grid_quantize_ceil(val: u8, grid_bits: u8) -> u8 {
    if val == 0 {
        return 0;
    }
    match grid_bits {
        2 => (val as u16 * 3).div_ceil(255) as u8,
        _ => (val as u16 * 15).div_ceil(255) as u8,
    }
}

/// OR a quantized cell into a packed grid row at cell index `idx`.
#[inline]
pub(crate) fn grid_set_cell(row: &mut [u8], idx: usize, cell: u8, grid_bits: u8) {
    match grid_bits {
        2 => row[idx / 4] |= cell << ((idx % 4) * 2),
        _ => {
            if idx.is_multiple_of(2) {
                row[idx / 2] |= cell;
            } else {
                row[idx / 2] |= cell << 4;
            }
        }
    }
}

/// Read a cell from a packed grid row at cell index `idx`.
#[inline]
pub(crate) fn grid_get_cell(row: &[u8], idx: usize, grid_bits: u8) -> u8 {
    match grid_bits {
        2 => (row[idx / 4] >> ((idx % 4) * 2)) & 0x03,
        _ => {
            if idx.is_multiple_of(2) {
                row[idx / 2] & 0x0F
            } else {
                row[idx / 2] >> 4
            }
        }
    }
}

/// Stream-write 4-bit packed grid (Section D) and 8-bit superblock grid (Section E).
///
/// `grid_entries` sorted by `(dim_id, block_id)`. `num_dims` is the fixed
/// vocabulary size (dims), so the grid has `num_dims` rows.
/// Each entry is `(dim_id, block_id, max_impact_u8)`.
///
/// Memory: O(packed_row_size + num_superblocks) — one row buffer each.
/// Returns `(packed_grid_bytes, sb_grid_bytes)`.
pub(crate) fn stream_write_grids(
    grid_entries: &[(u32, u32, u8)],
    num_dims: usize,
    num_blocks: usize,
    grid_bits: u8,
    writer: &mut dyn Write,
) -> std::io::Result<(u64, u64)> {
    let packed_row_size = grid_packed_row_size(num_blocks, grid_bits);
    let num_superblocks = num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE as usize);

    let mut row_buf = vec![0u8; packed_row_size];
    let mut sb_row = vec![0u8; num_superblocks];

    // Section D: packed 4-bit grid, one dim row at a time
    let mut gi = 0;
    for dim_id in 0..num_dims as u32 {
        row_buf.fill(0);
        while gi < grid_entries.len() && grid_entries[gi].0 == dim_id {
            let b = grid_entries[gi].1 as usize;
            let cell = grid_quantize_ceil(grid_entries[gi].2, grid_bits);
            grid_set_cell(&mut row_buf, b, cell, grid_bits);
            gi += 1;
        }
        writer.write_all(&row_buf)?;
    }
    let packed_bytes = (num_dims * packed_row_size) as u64;

    // Entries with dim_id >= num_dims sort past the cursor and have no grid
    // row: they would be dropped silently, leaving those dimensions
    // unsearchable. The segment builder rejects them at add/build time, so
    // leftovers here mean a legacy blob or a caller bug — say so loudly.
    if gi < grid_entries.len() {
        log::warn!(
            "[bmp] {} grid entries with dim_id >= dims={} dropped from the block-max grid \
             (first dim_id={}); these dimensions are unsearchable — raise `dims` in the \
             field's sparse_vector config and rebuild",
            grid_entries.len() - gi,
            num_dims,
            grid_entries[gi].0,
        );
    }

    // Section E: 8-bit superblock grid, one dim row at a time
    gi = 0;
    for dim_id in 0..num_dims as u32 {
        sb_row.fill(0);
        while gi < grid_entries.len() && grid_entries[gi].0 == dim_id {
            let b = grid_entries[gi].1 as usize;
            let sb = b / BMP_SUPERBLOCK_SIZE as usize;
            if grid_entries[gi].2 > sb_row[sb] {
                sb_row[sb] = grid_entries[gi].2;
            }
            gi += 1;
        }
        writer.write_all(&sb_row)?;
    }
    let sb_bytes = (num_dims * num_superblocks) as u64;

    Ok((packed_bytes, sb_bytes))
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
        match reader.read_exact(&mut buf) {
            Ok(()) => {
                let dim_id = u32::from_le_bytes(buf[0..4].try_into().unwrap());
                let block_id = u32::from_le_bytes(buf[4..8].try_into().unwrap());
                let impact = buf[8];
                Ok(Some((dim_id, block_id, impact)))
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e),
        }
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

/// Stream-write grids from multiple sorted run files (external merge sort).
///
/// Two-pass approach: first pass writes Section D (packed 4-bit grid),
/// then resets all readers and second pass writes Section E (superblock grid).
///
/// Within each run, entries are sorted by `(dim_id, block_id)`, so for each dim
/// we drain matching entries from all readers. No heap needed.
///
/// Returns `(packed_grid_bytes, sb_grid_bytes)`.
#[cfg(feature = "native")]
pub(crate) fn stream_write_grids_merged(
    run_readers: &mut [GridRunReader],
    num_dims: usize,
    num_blocks: usize,
    grid_bits: u8,
    writer: &mut dyn Write,
) -> std::io::Result<(u64, u64)> {
    let packed_row_size = grid_packed_row_size(num_blocks, grid_bits);
    let num_superblocks = num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE as usize);

    let mut row_buf = vec![0u8; packed_row_size];
    let mut sb_row = vec![0u8; num_superblocks];

    // Pass 1: Section D — packed 4-bit grid, one dim row at a time
    for dim_id in 0..num_dims as u32 {
        row_buf.fill(0);
        for reader in run_readers.iter_mut() {
            while let Some((d, block_id, impact)) = reader.current {
                if d != dim_id {
                    break;
                }
                let b = block_id as usize;
                let cell = grid_quantize_ceil(impact, grid_bits);
                grid_set_cell(&mut row_buf, b, cell, grid_bits);
                reader.advance()?;
            }
        }
        writer.write_all(&row_buf)?;
    }
    let packed_bytes = (num_dims * packed_row_size) as u64;

    // Same silent-drop hazard as `stream_write_grids`: any entry still
    // pending after the dim sweep has dim_id >= dims and no grid row.
    for reader in run_readers.iter() {
        if let Some((dim_id, _, _)) = reader.current {
            log::warn!(
                "[bmp] grid run contains entries with dim_id >= dims={num_dims} \
                 (first dim_id={dim_id}) that were dropped from the block-max grid; \
                 these dimensions are unsearchable — raise `dims` in the field's \
                 sparse_vector config and rebuild",
            );
            break;
        }
    }

    // Reset all readers for pass 2
    for reader in run_readers.iter_mut() {
        reader.reset()?;
    }

    // Pass 2: Section E — 8-bit superblock grid, one dim row at a time
    for dim_id in 0..num_dims as u32 {
        sb_row.fill(0);
        for reader in run_readers.iter_mut() {
            while let Some((d, block_id, impact)) = reader.current {
                if d != dim_id {
                    break;
                }
                let sb = block_id as usize / BMP_SUPERBLOCK_SIZE as usize;
                if impact > sb_row[sb] {
                    sb_row[sb] = impact;
                }
                reader.advance()?;
            }
        }
        writer.write_all(&sb_row)?;
    }
    let sb_bytes = (num_dims * num_superblocks) as u64;

    Ok((packed_bytes, sb_bytes))
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

        assert_eq!(footer.len(), 72);
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
        let num_virtual_docs = u32::from_le_bytes(fb[44..48].try_into().unwrap());
        assert_eq!(num_virtual_docs, 64); // padded to block_size

        // num_real_docs should be 3
        let num_real_docs = u32::from_le_bytes(fb[60..64].try_into().unwrap());
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
        let scale = f32::from_le_bytes(fb[48..52].try_into().unwrap());
        assert!((scale - 5.0).abs() < 0.001, "scale={}, expected 5.0", scale);
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
        let scale_a = f32::from_le_bytes(buf_a[footer_a + 48..footer_a + 52].try_into().unwrap());

        let mut buf_b = Vec::new();
        build_bmp_blob(postings_b, 64, 4, 0.0, None, 105879, 5.0, 4, &mut buf_b).unwrap();
        let footer_b = buf_b.len() - BMP_BLOB_FOOTER_SIZE;
        let scale_b = f32::from_le_bytes(buf_b[footer_b + 48..footer_b + 52].try_into().unwrap());
        assert_eq!(
            scale_a, scale_b,
            "Fixed max_weight scales must be identical"
        );
        assert!((scale_a - 5.0).abs() < 0.001);
    }
}
