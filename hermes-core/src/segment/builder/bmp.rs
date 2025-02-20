//! BMP (Block-Max Pruning) index builder for sparse vectors — **V11 format**.
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
//! Uses a K-way merge over per-dim cursors — **zero intermediate allocation**.
//! Each dim's postings are already sorted by `(doc_id, ordinal)` = sorted by
//! `compact_virtual_id` = sorted by `block_id`, so a min-heap merges them in
//! block order directly into the output arrays.
//!
//! Peak memory: `output arrays only + O(num_dims) heap`.
//!
//! ## V11 changes from V10
//!
//! - **Fixed `dims`**: vocabulary size stored in footer, grid indexed by dim_id directly
//! - **dim_id in per-block data**: u32 dim_id replaces dim_idx, blocks are portable
//! - **Block-aligned segments**: padded to block_size for block-copy merge
//! - **Fixed `max_weight_scale`**: no impact rescaling during merge
//! - **Section C (dim_ids) removed**: grid uses dim_id as row index
//! - **num_real_docs** footer field: actual vector count before padding
//!
//! ## BMP V11 Blob Layout (data-first, block-interleaved)
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
//!   num_terms: u16                                    offset 0
//!   term_dim_ids: [u32-LE × num_terms]                offset 2
//!   posting_starts: [u16-LE × (num_terms + 1)]        relative cumulative counts
//!   postings: [(u8, u8) × total_block_postings]       BmpPosting pairs
//!
//! BMP V11 Footer (64 bytes):
//!   total_terms: u32              //  0- 3  (stats only)
//!   total_postings: u32           //  4- 7  (stats only)
//!   grid_offset: u64              //  8-15  (byte offset of Section D)
//!   sb_grid_offset: u64           // 16-23  (byte offset of Section E)
//!   num_blocks: u32               // 24-27
//!   dims: u32                     // 28-31  (fixed vocabulary size)
//!   bmp_block_size: u32           // 32-35
//!   num_virtual_docs: u32         // 36-39  (= num_blocks × bmp_block_size, padded)
//!   max_weight_scale: f32         // 40-43
//!   doc_map_offset: u64           // 44-51  (byte offset of Section F)
//!   num_real_docs: u32            // 52-55  (actual vector count before padding)
//!   reserved: u32                 // 56-59
//!   magic: u32                    // 60-63  (BMP1 = 0x31504D42)
//! ```

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use rustc_hash::{FxHashMap, FxHashSet};

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
use crate::segment::format::BMP_BLOB_MAGIC_V11;
use crate::segment::reader::bmp::BMP_SUPERBLOCK_SIZE;

/// Build a BMP V11 blob from per-dimension postings.
///
/// **Takes ownership** of the postings HashMap. All per-dim Vecs are moved
/// out of the HashMap into `dim_vecs` before the K-way merge starts, and
/// the HashMap shell is dropped immediately. After the merge completes,
/// `dim_vecs` and `vid_lookup` are explicitly dropped before output write.
///
/// Uses compact virtual IDs: sequential IDs assigned to unique `(doc_id, ordinal)`
/// pairs, eliminating the sparse `doc_id * num_ordinals + ordinal` space.
///
/// Uses a K-way merge over per-dim cursors to avoid flattening all postings
/// into a single Vec + global sort. Each dim's postings are already sorted by
/// `(doc_id, ordinal)` = sorted by `compact_virtual_id` = sorted by `block_id`.
/// A min-heap merges them in `(block_id, dim_id)` order.
///
/// V11: block data uses u32 dim_id directly (not dim_idx). Grid has `dims` rows.
/// num_virtual_docs is padded to block_size alignment.
///
/// `bmp_block_size` is clamped to 256 max (u8 local_slot).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_bmp_blob(
    mut postings: FxHashMap<u32, Vec<(DocId, u16, f32)>>,
    bmp_block_size: u32,
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
    for (_dim_id, dim_postings) in postings.iter_mut() {
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

    // Phase 1: Find global max weight AND collect unique (doc_id, ordinal) pairs.
    // Single pass over all postings — avoids double iteration.
    // Uses FxHashSet for dedup to bound memory by unique pairs (not total postings).
    //
    // Weight threshold is skipped for dims with fewer than min_terms postings
    // to protect small dimensions from losing signal.
    //
    // Capacity: use the LARGEST single dim's posting count as a lower bound on
    // unique pairs (each doc appears at least once). This avoids the old bug
    // of pre-allocating for total_postings (50× larger than unique pairs when
    // docs appear in ~50 dims), which wasted GBs of heap.
    let max_dim_postings: usize = postings.values().map(|v| v.len()).max().unwrap_or(0);
    let mut vid_set: FxHashSet<(DocId, u16)> =
        FxHashSet::with_capacity_and_hasher(max_dim_postings, Default::default());

    for dim_postings in postings.values() {
        let skip_threshold = dim_postings.len() < min_terms;
        for &(doc_id, ordinal, weight) in dim_postings {
            let abs_w = weight.abs();
            if !skip_threshold && abs_w < weight_threshold {
                continue;
            }
            let impact = quantize_weight(abs_w, max_weight);
            if impact == 0 {
                continue;
            }
            vid_set.insert((doc_id, ordinal));
        }
    }

    if vid_set.is_empty() {
        return Ok(0);
    }

    // V11: max_weight_scale is fixed (= max_weight parameter)
    let max_weight_scale = max_weight;

    // Assign compact virtual IDs: sequential IDs for unique (doc_id, ordinal) pairs.
    let mut vid_pairs: Vec<(DocId, u16)> = vid_set.into_iter().collect();
    vid_pairs.sort_unstable();
    let num_real_docs = vid_pairs.len();

    // Build O(1) lookup table from (doc_id, ordinal) → virtual_id.
    // Replaces O(log N) binary search per-posting in the K-way merge.
    let vid_lookup = VidLookup::from_sorted_pairs(&vid_pairs);

    // Safety: local_slot is u8, so block_size must not exceed 256
    let effective_block_size = bmp_block_size.min(256);

    // V11: Pad num_virtual_docs to block_size alignment
    let num_virtual_docs =
        num_real_docs.div_ceil(effective_block_size as usize) * effective_block_size as usize;
    let num_blocks = num_virtual_docs / effective_block_size as usize;

    let mut dim_ids: Vec<u32> = postings.keys().copied().collect();
    dim_ids.sort_unstable();

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

    // Output arrays — V11: term_dim_ids stores dim_id (not dim_idx)
    let mut block_term_starts: Vec<u32> = Vec::with_capacity(num_blocks + 1);
    let mut term_dim_ids: Vec<u32> = Vec::new();
    let mut term_posting_starts: Vec<u32> = Vec::new();
    let mut postings_flat: Vec<u8> = Vec::new();
    let mut grid_entries: Vec<(u32, u32, u8)> = Vec::new(); // (dim_id, block_id, max_impact)
    let mut posting_count: u32 = 0;
    let mut last_block_filled: i64 = -1; // tracks which blocks have block_term_starts entries

    while let Some(&Reverse((block_id, _, _))) = heap.peek() {
        // Fill block_term_starts for empty blocks up to and including this one
        let fill_from = (last_block_filled + 1) as u32;
        for _ in fill_from..=block_id {
            block_term_starts.push(term_dim_ids.len() as u32);
        }
        last_block_filled = block_id as i64;

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

            // V11: store dim_id directly (not dim_idx)
            term_dim_ids.push(dim_id);
            term_posting_starts.push(posting_count);

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
                    // This posting belongs to a later block — record and stop
                    next_block = Some(bid2);
                    break;
                }

                let local_slot = (virtual_id % bs64) as u8;
                postings_flat.push(local_slot);
                postings_flat.push(impact);
                posting_count += 1;
                max_impact = max_impact.max(impact);
                pos += 1;
            }

            // Grid entry — V11: indexed by dim_id directly
            grid_entries.push((dim_id, block_id, max_impact));

            // Advance cursor
            if let Some(nb) = next_block {
                // Already found the next valid posting at `pos`
                cursors[dim_idx] = pos;
                heap.push(Reverse((nb, dim_id, dim_idx)));
            } else {
                // Scan for next valid posting (skipping invalids past this block)
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
    }

    // Fill remaining empty blocks
    for _ in (last_block_filled + 1) as u32..num_blocks as u32 {
        block_term_starts.push(term_dim_ids.len() as u32);
    }
    // Sentinels
    block_term_starts.push(term_dim_ids.len() as u32);
    term_posting_starts.push(posting_count);

    let total_terms = term_dim_ids.len() as u32;
    let total_postings = posting_count;

    // Sort grid entries by (dim_id, block_id) for streaming write
    grid_entries.sort_unstable();

    log::info!(
        "[bmp_build] V11 num_real_docs={} num_virtual_docs={} num_blocks={} dims={} \
         total_terms={} total_postings={} grid_entries={}",
        num_real_docs,
        num_virtual_docs,
        num_blocks,
        dims,
        total_terms,
        total_postings,
        grid_entries.len(),
    );

    // Free K-way merge inputs before writing output — reclaims per-dim
    // posting Vecs and the O(1) vid lookup table.
    drop(dim_slices); // borrows dim_vecs, must drop first
    drop(dim_vecs);
    drop(vid_lookup);

    // ── Write V11 interleaved sections (data-first) ──────────────────────
    let mut bytes_written: u64 = 0;

    // Sections B+A: block data first, then block_data_starts
    // V11: always 4-byte dim IDs
    bytes_written += write_v11_interleaved_sections(
        writer,
        block_term_starts,
        term_dim_ids,
        term_posting_starts,
        postings_flat,
        num_blocks,
    )?;

    // Sections D+E: packed grid + sb_grid (streaming from sparse grid entries)
    // V11: `dims` rows (not num_dims)
    let grid_offset = bytes_written;
    let (packed_bytes, sb_bytes) =
        stream_write_grids(&grid_entries, dims as usize, num_blocks, writer)?;
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

    // BMP V11 Footer (64 bytes)
    write_v11_footer(
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
    )?;
    bytes_written += 64;

    Ok(bytes_written)
}

/// Write the BMP V11 footer (64 bytes).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_v11_footer(
    writer: &mut dyn Write,
    total_terms: u32,
    total_postings: u32,
    grid_offset: u64,
    sb_grid_offset: u64,
    num_blocks: u32,
    dims: u32,
    bmp_block_size: u32,
    num_virtual_docs: u32,
    max_weight_scale: f32,
    doc_map_offset: u64,
    num_real_docs: u32,
) -> std::io::Result<()> {
    writer.write_u32::<LittleEndian>(total_terms)?; //  0- 3
    writer.write_u32::<LittleEndian>(total_postings)?; //  4- 7
    writer.write_u64::<LittleEndian>(grid_offset)?; //  8-15
    writer.write_u64::<LittleEndian>(sb_grid_offset)?; // 16-23
    writer.write_u32::<LittleEndian>(num_blocks)?; // 24-27
    writer.write_u32::<LittleEndian>(dims)?; // 28-31
    writer.write_u32::<LittleEndian>(bmp_block_size)?; // 32-35
    writer.write_u32::<LittleEndian>(num_virtual_docs)?; // 36-39
    writer.write_f32::<LittleEndian>(max_weight_scale)?; // 40-43
    writer.write_u64::<LittleEndian>(doc_map_offset)?; // 44-51
    writer.write_u32::<LittleEndian>(num_real_docs)?; // 52-55
    writer.write_u32::<LittleEndian>(0)?; // 56-59 reserved
    writer.write_u32::<LittleEndian>(BMP_BLOB_MAGIC_V11)?; // 60-63
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

/// Write V11 interleaved block data: Section B (per-block data) + padding + Section A (block_data_starts).
///
/// V11: always uses u32 dim_ids (no dim_id_width concept).
///
/// **Consumes** the intermediate arrays to free memory before writing.
/// Uses a two-pass approach:
///   Pass 1: compute per-block data sizes → fill `block_data_starts`
///   Pass 2: stream each block's data directly to writer using a small reusable buffer
///   Finally: write padding + block_data_starts
///
/// Returns total bytes written (sections B + padding + A).
pub(crate) fn write_v11_interleaved_sections(
    writer: &mut dyn Write,
    block_term_starts: Vec<u32>,
    term_dim_ids: Vec<u32>,
    term_posting_starts: Vec<u32>,
    postings_flat: Vec<u8>,
    num_blocks: usize,
) -> std::io::Result<u64> {
    // Pass 1: compute per-block byte sizes for block_data_starts
    let mut block_data_starts: Vec<u64> = Vec::with_capacity(num_blocks + 1);
    let mut cumulative: u64 = 0;
    for b in 0..num_blocks {
        block_data_starts.push(cumulative);
        let ts = block_term_starts[b] as usize;
        let te = block_term_starts[b + 1] as usize;
        let nt = te - ts;
        if nt == 0 {
            continue;
        }
        let posting_bytes = (term_posting_starts[te] - term_posting_starts[ts]) as usize * 2;
        // num_terms(2) + dim_ids(nt × 4) + posting_starts((nt+1) × 2) + postings
        let block_size = 2 + nt * 4 + (nt + 1) * 2 + posting_bytes;
        cumulative += block_size as u64;
    }
    block_data_starts.push(cumulative);

    let mut bytes_written: u64 = 0;

    // Section B: stream each block's interleaved data directly to writer.
    let mut blk_buf: Vec<u8> = Vec::with_capacity(4096);

    for b in 0..num_blocks {
        let ts = block_term_starts[b] as usize;
        let te = block_term_starts[b + 1] as usize;
        let nt = te - ts;
        if nt == 0 {
            continue;
        }
        blk_buf.clear();

        // num_terms (u16)
        blk_buf.extend_from_slice(&(nt as u16).to_le_bytes());

        // V11: term_dim_ids — always u32
        for &dim_id in &term_dim_ids[ts..te] {
            blk_buf.extend_from_slice(&dim_id.to_le_bytes());
        }

        // posting_starts (u16, relative to this block)
        let first_posting = term_posting_starts[ts];
        for &ps in &term_posting_starts[ts..=te] {
            let relative = (ps - first_posting) as u16;
            blk_buf.extend_from_slice(&relative.to_le_bytes());
        }

        // postings [(u8, u8) × block_posting_count]
        let p_start = term_posting_starts[ts] as usize * 2;
        let p_end = term_posting_starts[te] as usize * 2;
        blk_buf.extend_from_slice(&postings_flat[p_start..p_end]);

        writer.write_all(&blk_buf)?;
    }
    bytes_written += cumulative;

    // Inputs consumed — freed here
    drop(block_term_starts);
    drop(term_dim_ids);
    drop(term_posting_starts);
    drop(postings_flat);

    // Padding to 8-byte boundary (for u64 alignment of Section A)
    let padding = (8 - (bytes_written % 8) as usize) % 8;
    if padding > 0 {
        writer.write_all(&[0u8; 8][..padding])?;
        bytes_written += padding as u64;
    }

    // Section A: block_data_starts [u64 × (num_blocks + 1)]
    bytes_written += write_u64_slice_le(writer, &block_data_starts)?;

    Ok(bytes_written)
}

/// Ceiling quantize u8 → u4. Guarantees `u4 * 17 >= original`.
///
/// This ensures that 4-bit grid upper bounds are always safe (never
/// underestimate the true u8 value when unpacked via ×17).
#[inline]
pub(crate) fn quantize_u8_to_u4_ceil(val: u8) -> u8 {
    if val == 0 {
        return 0;
    }
    (val as u16 * 15).div_ceil(255) as u8
}

/// Stream-write 4-bit packed grid (Section D) and 8-bit superblock grid (Section E).
///
/// V11: `grid_entries` sorted by `(dim_id, block_id)`. `num_dims` is the fixed
/// vocabulary size (dims), so the grid has `num_dims` rows.
/// Each entry is `(dim_id, block_id, max_impact_u8)`.
///
/// Memory: O(packed_row_size + num_superblocks) — one row buffer each.
/// Returns `(packed_grid_bytes, sb_grid_bytes)`.
pub(crate) fn stream_write_grids(
    grid_entries: &[(u32, u32, u8)],
    num_dims: usize,
    num_blocks: usize,
    writer: &mut dyn Write,
) -> std::io::Result<(u64, u64)> {
    let packed_row_size = num_blocks.div_ceil(2);
    let num_superblocks = num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE as usize);

    let mut row_buf = vec![0u8; packed_row_size];
    let mut sb_row = vec![0u8; num_superblocks];

    // Section D: packed 4-bit grid, one dim row at a time
    let mut gi = 0;
    for dim_id in 0..num_dims as u32 {
        row_buf.fill(0);
        while gi < grid_entries.len() && grid_entries[gi].0 == dim_id {
            let b = grid_entries[gi].1 as usize;
            let q4 = quantize_u8_to_u4_ceil(grid_entries[gi].2);
            if b.is_multiple_of(2) {
                row_buf[b / 2] |= q4;
            } else {
                row_buf[b / 2] |= q4 << 4;
            }
            gi += 1;
        }
        writer.write_all(&row_buf)?;
    }
    let packed_bytes = (num_dims * packed_row_size) as u64;

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
    fn test_build_bmp_blob_empty() {
        let postings = FxHashMap::default();
        let mut buf = Vec::new();
        let size = build_bmp_blob(postings, 64, 0.0, None, 105879, 5.0, 4, &mut buf).unwrap();
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
        let size = build_bmp_blob(postings, 64, 0.0, None, 105879, 5.0, 4, &mut buf).unwrap();
        assert!(size > 0);
        assert_eq!(buf.len(), size as usize);

        // Verify footer magic
        let footer_start = buf.len() - 4;
        let magic = u32::from_le_bytes(buf[footer_start..].try_into().unwrap());
        assert_eq!(magic, BMP_BLOB_MAGIC_V11);
    }

    #[test]
    fn test_build_bmp_blob_multi_ordinal() {
        let mut postings = FxHashMap::default();
        // dim 0: doc 0 ord 0, doc 0 ord 1, doc 1 ord 0
        postings.insert(0u32, vec![(0u32, 0u16, 1.0f32), (0, 1, 0.8), (1, 0, 0.5)]);

        let mut buf = Vec::new();
        let size = build_bmp_blob(postings, 64, 0.0, None, 105879, 5.0, 4, &mut buf).unwrap();
        assert!(size > 0);

        // Verify footer: num_virtual_docs should be 64 (padded to block_size)
        // 3 real docs padded to 64
        let footer_start = buf.len() - 64;
        let fb = &buf[footer_start..];
        let num_virtual_docs = u32::from_le_bytes(fb[36..40].try_into().unwrap());
        assert_eq!(num_virtual_docs, 64); // padded to block_size

        // num_real_docs should be 3
        let num_real_docs = u32::from_le_bytes(fb[52..56].try_into().unwrap());
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
        let size = build_bmp_blob(postings, 64, 0.0, None, 105879, 5.0, 4, &mut buf).unwrap();
        assert!(size > 0);

        // Verify max_weight_scale in V11 footer (bytes 40-43)
        let footer_start = buf.len() - 64;
        let fb = &buf[footer_start..];
        let scale = f32::from_le_bytes(fb[40..44].try_into().unwrap());
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
        build_bmp_blob(postings_a, 64, 0.0, None, 105879, 5.0, 4, &mut buf_a).unwrap();
        let scale_a = f32::from_le_bytes(
            buf_a[buf_a.len() - 64 + 40..buf_a.len() - 64 + 44]
                .try_into()
                .unwrap(),
        );

        let mut buf_b = Vec::new();
        build_bmp_blob(postings_b, 64, 0.0, None, 105879, 5.0, 4, &mut buf_b).unwrap();
        let scale_b = f32::from_le_bytes(
            buf_b[buf_b.len() - 64 + 40..buf_b.len() - 64 + 44]
                .try_into()
                .unwrap(),
        );
        assert_eq!(
            scale_a, scale_b,
            "Fixed max_weight scales must be identical"
        );
        assert!((scale_a - 5.0).abs() < 0.001);
    }
}
