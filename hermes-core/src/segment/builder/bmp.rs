//! BMP (Block-Max Pruning) index builder for sparse vectors — **V7 format**.
//!
//! Builds a block-at-a-time (BAAT) index using **compact virtual coordinates**:
//! sequential IDs are assigned to unique `(doc_id, ordinal)` pairs. A lookup
//! table (Section 9) enables query-time recovery of the original coordinates.
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
//! V7 replaces the `FxHashMap vid_map` with binary search on sorted `vid_pairs`,
//! saving ~50-80 bytes per entry during build.
//!
//! Peak memory: `output arrays only + O(num_dims) heap`.
//!
//! ## BMP V7 Blob Layout
//!
//! V7 uses compact virtual IDs (sequential assignment) instead of the sparse
//! `doc_id * num_ordinals + ordinal` scheme, eliminating catastrophic space
//! blowup when ordinal distribution is skewed.
//!
//! V7 stores dim_indices (position in dim_ids) instead of raw dim_ids in
//! Section 2, using u16 entries when num_dims ≤ 65536 (halving section size).
//!
//! ```text
//! Section 1:  block_term_starts    [u32-LE × (num_blocks + 1)]
//! Section 2:  term_dim_indices     [u16-LE or u32-LE × total_terms]  ← u16 when num_dims ≤ 65536
//! Section 3:  term_posting_starts  [u32-LE × (total_terms + 1)]      ← prefix sums
//! Section 4:  postings             [(u8, u8) × total_postings]       ← BmpPosting pairs
//! Section 5:  padding              [0-3 bytes to next 4-byte boundary]
//! Section 6:  dim_ids              [u32-LE × num_dims]
//! Section 7:  grid_packed4         [u8 × (num_dims × packed_row_size)]  ← 4-bit packed
//! Section 8:  sb_grid              [u8 × (num_dims × num_superblocks)]  ← 8-bit
//! Section 9a: doc_map_ids          [u32-LE × num_virtual_docs]
//! Section 9b: doc_map_ordinals     [u16-LE × num_virtual_docs]
//!
//! BMP7 Footer (48 bytes):
//!   total_terms: u32              // 0-3
//!   total_postings: u32           // 4-7
//!   dim_ids_offset: u32           // 8-11   (byte offset of section 6)
//!   grid_offset: u32              // 12-15  (byte offset of section 7)
//!   num_blocks: u32               // 16-19
//!   num_dims: u32                 // 20-23
//!   bmp_block_size: u32           // 24-27
//!   num_virtual_docs: u32         // 28-31  (= actual vector count)
//!   max_weight_scale: f32         // 32-35
//!   sb_grid_offset: u32           // 36-39  (byte offset of section 8)
//!   doc_map_offset: u32           // 40-43  (byte offset of section 9)
//!   magic: u32                    // 44-47  (BMP7 = 0x37504D42)
//! ```

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use rustc_hash::{FxHashMap, FxHashSet};

/// Look up a compact virtual ID by binary search on sorted `vid_pairs`.
///
/// `vid_pairs` must be sorted. Returns the index (= compact virtual ID).
/// Panics if the key is not found (all valid postings must have a vid).
#[inline]
pub(crate) fn vid_lookup(vid_pairs: &[(crate::DocId, u16)], key: (crate::DocId, u16)) -> u32 {
    vid_pairs.binary_search(&key).unwrap() as u32
}

use crate::DocId;
use crate::segment::format::BMP_BLOB_MAGIC_V7;
use crate::segment::reader::bmp::BMP_SUPERBLOCK_SIZE;

/// Build a BMP V7 blob from per-dimension postings.
///
/// Uses compact virtual IDs: sequential IDs assigned to unique `(doc_id, ordinal)`
/// pairs, eliminating the sparse `doc_id * num_ordinals + ordinal` space.
///
/// Uses a K-way merge over per-dim cursors to avoid flattening all postings
/// into a single Vec + global sort. Each dim's postings are already sorted by
/// `(doc_id, ordinal)` = sorted by `compact_virtual_id` = sorted by `block_id`.
/// A min-heap merges them in `(block_id, dim_id)` order.
///
/// `bmp_block_size` is clamped to 256 max (u8 local_slot).
pub(crate) fn build_bmp_blob(
    postings: &mut FxHashMap<u32, Vec<(DocId, u16, f32)>>,
    bmp_block_size: u32,
    weight_threshold: f32,
    pruning_fraction: Option<f32>,
    quantization_factor: Option<f32>,
    writer: &mut dyn Write,
) -> std::io::Result<u64> {
    if postings.is_empty() {
        return Ok(0);
    }

    // Phase 0: Prune per-dimension
    for (_dim_id, dim_postings) in postings.iter_mut() {
        if let Some(fraction) = pruning_fraction
            && dim_postings.len() > 1
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
    let mut global_max_weight: f32 = 0.0;
    let mut vid_set: FxHashSet<(DocId, u16)> = FxHashSet::default();

    for dim_postings in postings.values() {
        for &(doc_id, ordinal, weight) in dim_postings {
            let abs_w = weight.abs();
            if abs_w < weight_threshold {
                continue;
            }
            if abs_w > global_max_weight {
                global_max_weight = abs_w;
            }
            vid_set.insert((doc_id, ordinal));
        }
    }

    if global_max_weight == 0.0 || vid_set.is_empty() {
        return Ok(0);
    }

    // Fixed quantization: impact = min(255, round(weight * factor))
    // Derive max_weight_scale = 255 / factor for backward-compatible query math.
    // Dynamic quantization: impact = round(weight / max * 255)
    let max_weight_scale = if let Some(factor) = quantization_factor {
        255.0 / factor
    } else {
        global_max_weight
    };

    // Assign compact virtual IDs: sequential IDs for unique (doc_id, ordinal) pairs.
    // This eliminates the sparse `doc_id * num_ordinals + ordinal` space that causes
    // catastrophic grid blowup when ordinal distribution is skewed.
    let mut vid_pairs: Vec<(DocId, u16)> = vid_set.into_iter().collect();
    vid_pairs.sort_unstable();
    let num_virtual_docs = vid_pairs.len();

    // Lookup: binary search on sorted vid_pairs (O(log N) per lookup, saves ~50-80 bytes/entry)

    // Safety: local_slot is u8, so block_size must not exceed 256
    let effective_block_size = bmp_block_size.min(256);
    let num_blocks = num_virtual_docs.div_ceil(effective_block_size as usize);

    let mut dim_ids: Vec<u32> = postings.keys().copied().collect();
    dim_ids.sort_unstable();
    let num_dims = dim_ids.len();

    // Phase 2: K-way merge over per-dim cursors
    //
    // Collect dim slices (avoid repeated HashMap lookups in hot loop)
    let dim_slices: Vec<&[(DocId, u16, f32)]> = dim_ids
        .iter()
        .map(|&d| postings.get(&d).map(|v| v.as_slice()).unwrap_or(&[]))
        .collect();

    // Per-dim cursor positions
    let mut cursors: Vec<usize> = vec![0; num_dims];

    // Min-heap: (block_id, dim_id, dim_idx)
    let mut heap: BinaryHeap<Reverse<(u32, u32, usize)>> = BinaryHeap::with_capacity(num_dims);

    let bs64 = effective_block_size as u64;

    // Initialize heap with first valid posting from each dim
    for (dim_idx, &dim_id) in dim_ids.iter().enumerate() {
        let posts = dim_slices[dim_idx];
        for (pos, &(doc_id, ordinal, weight)) in posts.iter().enumerate() {
            let abs_w = weight.abs();
            if abs_w < weight_threshold {
                continue;
            }
            let impact = quantize_weight(abs_w, max_weight_scale);
            if impact == 0 {
                continue;
            }
            let virtual_id = vid_lookup(&vid_pairs, (doc_id, ordinal)) as u64;
            let block_id = (virtual_id / bs64) as u32;
            cursors[dim_idx] = pos;
            heap.push(Reverse((block_id, dim_id, dim_idx)));
            break;
        }
    }

    if heap.is_empty() {
        return Ok(0);
    }

    // Output arrays
    let mut block_term_starts: Vec<u32> = Vec::with_capacity(num_blocks + 1);
    let mut term_dim_ids: Vec<u32> = Vec::new();
    let mut term_posting_starts: Vec<u32> = Vec::new();
    let mut postings_flat: Vec<u8> = Vec::new();
    let mut grid_entries: Vec<(u32, u32, u8)> = Vec::new(); // (dim_idx, block_id, max_impact)
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
            let mut pos = cursors[dim_idx];
            let mut max_impact = 0u8;
            let mut next_block: Option<u32> = None;

            term_dim_ids.push(dim_idx as u32);
            term_posting_starts.push(posting_count);

            // Process all postings for this dim in this block
            while pos < posts.len() {
                let (doc_id, ordinal, weight) = posts[pos];
                let abs_w = weight.abs();
                if abs_w < weight_threshold {
                    pos += 1;
                    continue;
                }
                let impact = quantize_weight(abs_w, max_weight_scale);
                if impact == 0 {
                    pos += 1;
                    continue;
                }

                let virtual_id = vid_lookup(&vid_pairs, (doc_id, ordinal)) as u64;
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

            // Grid entry
            grid_entries.push((dim_idx as u32, block_id, max_impact));

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
                    if abs_w >= weight_threshold {
                        let impact = quantize_weight(abs_w, max_weight_scale);
                        if impact > 0 {
                            let virtual_id = vid_lookup(&vid_pairs, (doc_id, ordinal)) as u64;
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

    // Sort grid entries by (dim_idx, block_id) for streaming write
    grid_entries.sort_unstable();

    let packed_row_size = num_blocks.div_ceil(2);
    let num_superblocks = num_blocks.div_ceil(BMP_SUPERBLOCK_SIZE as usize);
    let tdi_elem_size = if num_dims <= 65536 { 2 } else { 4 };
    log::info!(
        "[bmp_build] num_virtual_docs={} num_blocks={} num_dims={} \
         total_terms={} total_postings={} grid_entries={} \
         block_term_starts={}B term_dim_indices={}B(u{}) term_posting_starts={}B \
         postings_flat={}B grid={}B sb_grid={}B doc_map={}B",
        num_virtual_docs,
        num_blocks,
        num_dims,
        total_terms,
        total_postings,
        grid_entries.len(),
        (num_blocks + 1) * 4,
        total_terms as usize * tdi_elem_size,
        tdi_elem_size * 8,
        (total_terms as usize + 1) * 4,
        total_postings as usize * 2,
        num_dims * packed_row_size,
        num_dims * num_superblocks,
        num_virtual_docs * 6,
    );

    // ── Write V7 sections ───────────────────────────────────────────────
    let mut bytes_written: u64 = 0;

    // Section 1: block_term_starts [u32-LE × (num_blocks + 1)]
    bytes_written += write_u32_slice_le(writer, &block_term_starts)?;

    // Section 2: term_dim_indices — u16 when num_dims ≤ 65536, else u32
    bytes_written += write_dim_indices_section(writer, &term_dim_ids, num_dims)?;

    // Section 3: term_posting_starts [u32-LE × (total_terms + 1)]
    bytes_written += write_u32_slice_le(writer, &term_posting_starts)?;

    // Section 4: postings [(u8, u8) × total_postings]
    writer.write_all(&postings_flat)?;
    bytes_written += postings_flat.len() as u64;

    // Section 5: padding to 4-byte boundary
    let padding = (4 - (bytes_written % 4) as usize) % 4;
    if padding > 0 {
        writer.write_all(&[0u8; 4][..padding])?;
        bytes_written += padding as u64;
    }

    // Section 6: dim_ids [u32-LE × num_dims]
    let dim_ids_offset = bytes_written as u32;
    bytes_written += write_u32_slice_le(writer, &dim_ids)?;

    // Sections 7+8: packed grid + sb_grid (streaming from sparse grid entries)
    let grid_offset = bytes_written as u32;
    let (packed_bytes, sb_bytes) = stream_write_grids(&grid_entries, num_dims, num_blocks, writer)?;
    let sb_grid_offset = (bytes_written + packed_bytes) as u32;
    bytes_written += packed_bytes + sb_bytes;

    // Section 9a: doc_map_ids [u32-LE × num_virtual_docs]
    let doc_map_offset = bytes_written as u32;
    for &(doc_id, _) in &vid_pairs {
        writer.write_u32::<LittleEndian>(doc_id)?;
    }
    bytes_written += num_virtual_docs as u64 * 4;

    // Section 9b: doc_map_ordinals [u16-LE × num_virtual_docs]
    for &(_, ord) in &vid_pairs {
        writer.write_u16::<LittleEndian>(ord)?;
    }
    bytes_written += num_virtual_docs as u64 * 2;

    // BMP V7 Footer (48 bytes)
    writer.write_u32::<LittleEndian>(total_terms)?;
    writer.write_u32::<LittleEndian>(total_postings)?;
    writer.write_u32::<LittleEndian>(dim_ids_offset)?;
    writer.write_u32::<LittleEndian>(grid_offset)?;
    writer.write_u32::<LittleEndian>(num_blocks as u32)?;
    writer.write_u32::<LittleEndian>(num_dims as u32)?;
    writer.write_u32::<LittleEndian>(effective_block_size)?;
    writer.write_u32::<LittleEndian>(num_virtual_docs as u32)?;
    writer.write_f32::<LittleEndian>(max_weight_scale)?;
    writer.write_u32::<LittleEndian>(sb_grid_offset)?;
    writer.write_u32::<LittleEndian>(doc_map_offset)?;
    writer.write_u32::<LittleEndian>(BMP_BLOB_MAGIC_V7)?;
    bytes_written += 48;

    Ok(bytes_written)
}

/// Bulk-write a `&[u32]` slice as little-endian bytes.
///
/// On little-endian platforms this is a single `write_all` (zero-copy cast).
/// On big-endian platforms, falls back to per-element byte-swap.
pub(crate) fn write_u32_slice_le(writer: &mut dyn Write, data: &[u32]) -> std::io::Result<u64> {
    if data.is_empty() {
        return Ok(0);
    }
    #[cfg(target_endian = "little")]
    {
        // SAFETY: u32 has no padding bytes, LE matches wire format.
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        writer.write_all(bytes)?;
    }
    #[cfg(target_endian = "big")]
    {
        for &v in data {
            writer.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(data.len() as u64 * 4)
}

/// Write V7 Section 2 (term_dim_indices): u16 when num_dims ≤ 65536, else u32.
///
/// `indices` contains dim_idx values (positions in dim_ids array, not raw dim_ids).
pub(crate) fn write_dim_indices_section(
    writer: &mut dyn Write,
    indices: &[u32],
    num_dims: usize,
) -> std::io::Result<u64> {
    if indices.is_empty() {
        return Ok(0);
    }
    if num_dims <= 65536 {
        let u16_vals: Vec<u16> = indices.iter().map(|&v| v as u16).collect();
        write_u16_slice_le(writer, &u16_vals)
    } else {
        write_u32_slice_le(writer, indices)
    }
}

/// Bulk-write a `&[u16]` slice as little-endian bytes.
pub(crate) fn write_u16_slice_le(writer: &mut dyn Write, data: &[u16]) -> std::io::Result<u64> {
    if data.is_empty() {
        return Ok(0);
    }
    #[cfg(target_endian = "little")]
    {
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
        writer.write_all(bytes)?;
    }
    #[cfg(target_endian = "big")]
    {
        for &v in data {
            writer.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(data.len() as u64 * 2)
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

/// Stream-write 4-bit packed grid (section 7) and 8-bit superblock grid (section 8).
///
/// `grid_entries` must be sorted by `(dim_idx, block_id)`.
/// Each entry is `(dim_idx, block_id, max_impact_u8)`.
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

    // Section 7: packed 4-bit grid, one dim row at a time
    let mut gi = 0;
    for dim_idx in 0..num_dims as u32 {
        row_buf.fill(0);
        while gi < grid_entries.len() && grid_entries[gi].0 == dim_idx {
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

    // Section 8: 8-bit superblock grid, one dim row at a time
    gi = 0;
    for dim_idx in 0..num_dims as u32 {
        sb_row.fill(0);
        while gi < grid_entries.len() && grid_entries[gi].0 == dim_idx {
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
        let mut postings = FxHashMap::default();
        let mut buf = Vec::new();
        let size = build_bmp_blob(&mut postings, 64, 0.0, None, None, &mut buf).unwrap();
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
        let size = build_bmp_blob(&mut postings, 64, 0.0, None, None, &mut buf).unwrap();
        assert!(size > 0);
        assert_eq!(buf.len(), size as usize);

        // Verify footer magic
        let footer_start = buf.len() - 4;
        let magic = u32::from_le_bytes(buf[footer_start..].try_into().unwrap());
        assert_eq!(magic, BMP_BLOB_MAGIC_V7);
    }

    #[test]
    fn test_build_bmp_blob_multi_ordinal() {
        let mut postings = FxHashMap::default();
        // dim 0: doc 0 ord 0, doc 0 ord 1, doc 1 ord 0
        postings.insert(0u32, vec![(0u32, 0u16, 1.0f32), (0, 1, 0.8), (1, 0, 0.5)]);

        let mut buf = Vec::new();
        let size = build_bmp_blob(&mut postings, 64, 0.0, None, None, &mut buf).unwrap();
        assert!(size > 0);

        // Verify footer: num_virtual_docs should be 3 (at offset 28-31 in 48-byte footer)
        // 3 unique (doc_id, ordinal) pairs: (0,0), (0,1), (1,0)
        let footer_start = buf.len() - 48;
        let fb = &buf[footer_start..];
        let num_virtual_docs = u32::from_le_bytes(fb[28..32].try_into().unwrap());
        assert_eq!(num_virtual_docs, 3);
    }

    #[test]
    fn test_build_bmp_blob_quantization_factor() {
        let mut postings = FxHashMap::default();
        // dim 0: doc 0 with weight 2.0, doc 1 with weight 1.0
        postings.insert(0u32, vec![(0u32, 0u16, 2.0f32), (1, 0, 1.0)]);

        // With factor=100: impact(2.0) = min(255, round(2.0 * 100)) = 200
        //                   impact(1.0) = min(255, round(1.0 * 100)) = 100
        // max_weight_scale = 255/100 = 2.55
        let mut buf = Vec::new();
        let size = build_bmp_blob(&mut postings, 64, 0.0, None, Some(100.0), &mut buf).unwrap();
        assert!(size > 0);

        // Verify max_weight_scale in footer (bytes 32-35)
        let footer_start = buf.len() - 48;
        let fb = &buf[footer_start..];
        let scale = f32::from_le_bytes(fb[32..36].try_into().unwrap());
        assert!(
            (scale - 2.55).abs() < 0.001,
            "scale={}, expected 2.55",
            scale
        );
    }

    #[test]
    fn test_quantization_factor_vs_dynamic() {
        // Two segments with different max weights should produce different
        // max_weight_scale with dynamic quantization but same with fixed factor.

        // Segment A: max weight = 3.0
        let mut postings_a = FxHashMap::default();
        postings_a.insert(0u32, vec![(0u32, 0u16, 3.0f32), (1, 0, 1.5)]);

        // Segment B: max weight = 1.0
        let mut postings_b = FxHashMap::default();
        postings_b.insert(0u32, vec![(0u32, 0u16, 1.0f32), (1, 0, 0.5)]);

        // Dynamic: different scales
        let mut buf_a = Vec::new();
        build_bmp_blob(&mut postings_a.clone(), 64, 0.0, None, None, &mut buf_a).unwrap();
        let scale_a = f32::from_le_bytes(
            buf_a[buf_a.len() - 48 + 32..buf_a.len() - 48 + 36]
                .try_into()
                .unwrap(),
        );

        let mut buf_b = Vec::new();
        build_bmp_blob(&mut postings_b.clone(), 64, 0.0, None, None, &mut buf_b).unwrap();
        let scale_b = f32::from_le_bytes(
            buf_b[buf_b.len() - 48 + 32..buf_b.len() - 48 + 36]
                .try_into()
                .unwrap(),
        );
        assert_ne!(scale_a, scale_b, "Dynamic scales should differ");

        // Fixed factor=50: same scale = 255/50 = 5.1
        let mut buf_af = Vec::new();
        build_bmp_blob(&mut postings_a, 64, 0.0, None, Some(50.0), &mut buf_af).unwrap();
        let scale_af = f32::from_le_bytes(
            buf_af[buf_af.len() - 48 + 32..buf_af.len() - 48 + 36]
                .try_into()
                .unwrap(),
        );

        let mut buf_bf = Vec::new();
        build_bmp_blob(&mut postings_b, 64, 0.0, None, Some(50.0), &mut buf_bf).unwrap();
        let scale_bf = f32::from_le_bytes(
            buf_bf[buf_bf.len() - 48 + 32..buf_bf.len() - 48 + 36]
                .try_into()
                .unwrap(),
        );
        assert_eq!(scale_af, scale_bf, "Fixed factor scales must be identical");
        assert!((scale_af - 5.1).abs() < 0.001);
    }
}
