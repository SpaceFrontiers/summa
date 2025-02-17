//! BMP (Block-Max Pruning) index builder for sparse vectors — **V5 format**.
//!
//! Builds a block-at-a-time (BAAT) index using **virtual coordinates**:
//! `virtual_id = doc_id * num_ordinals + ordinal`, so each (doc_id, ordinal)
//! pair is a unique "virtual document". Blocks are over virtual_ids.
//!
//! Postings are 2 bytes each: `(local_slot: u8, impact: u8)`.
//!
//! Based on Mallia, Suel & Tonellotto (SIGIR 2024).
//!
//! ## Memory efficiency
//!
//! All postings are collected into a single flat Vec, sorted once, then written
//! as contiguous arrays. No per-block Vec allocations, no intermediate buffer.
//!
//! Peak memory: `entries (12B × total_postings) + grid_entries (9B × total_terms)`.
//!
//! ## BMP V5 Blob Layout
//!
//! V5 packs the block grid to 4-bit (50% memory reduction, ≤0.05 recall loss).
//!
//! ```text
//! Section 1: block_term_starts    [u32-LE × (num_blocks + 1)]
//! Section 2: term_dim_ids         [u32-LE × total_terms]
//! Section 3: term_posting_starts  [u32-LE × (total_terms + 1)]    ← prefix sums
//! Section 4: postings             [(u8, u8) × total_postings]     ← BmpPosting pairs
//! Section 5: padding              [0-3 bytes to next 4-byte boundary]
//! Section 6: dim_ids              [u32-LE × num_dims]
//! Section 7: grid_packed4         [u8 × (num_dims × packed_row_size)]  ← 4-bit packed
//! Section 8: sb_grid              [u8 × (num_dims × num_superblocks)]  ← 8-bit
//!
//! BMP5 Footer (48 bytes):
//!   total_terms: u32              // 0-3
//!   total_postings: u32           // 4-7
//!   dim_ids_offset: u32           // 8-11   (byte offset of section 6)
//!   grid_offset: u32              // 12-15  (byte offset of section 7)
//!   num_blocks: u32               // 16-19
//!   num_dims: u32                 // 20-23
//!   bmp_block_size: u32           // 24-27
//!   num_ordinals: u32             // 28-31
//!   max_weight_scale: f32         // 32-35
//!   sb_grid_offset: u32           // 36-39  (byte offset of section 8)
//!   _reserved: u32                // 40-43
//!   magic: u32                    // 44-47  (BMP5 = 0x35504D42)
//! ```

use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use rustc_hash::FxHashMap;

use crate::DocId;
use crate::segment::format::BMP_BLOB_MAGIC_V5;
use crate::segment::reader::bmp::BMP_SUPERBLOCK_SIZE;

/// Build a BMP V5 blob from per-dimension postings.
///
/// Grid entries `(dim_idx, block_id, max_impact)` are collected sparsely
/// during the posting scan — no dense grid allocation. The 4-bit packed grid
/// and 8-bit superblock grid are streamed dim-by-dim during write.
/// `bmp_block_size` is clamped to 256 max (u8 local_slot).
pub(crate) fn build_bmp_blob(
    postings: &mut FxHashMap<u32, Vec<(DocId, u16, f32)>>,
    bmp_block_size: u32,
    weight_threshold: f32,
    pruning_fraction: Option<f32>,
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

    // Phase 1: Find global max weight, max doc_id, and num_ordinals
    let mut global_max_weight: f32 = 0.0;
    let mut max_doc_id: DocId = 0;
    let mut max_ordinal: u16 = 0;
    let mut has_any = false;

    for dim_postings in postings.values() {
        for &(doc_id, ordinal, weight) in dim_postings {
            let abs_w = weight.abs();
            if abs_w < weight_threshold {
                continue;
            }
            has_any = true;
            if abs_w > global_max_weight {
                global_max_weight = abs_w;
            }
            if doc_id > max_doc_id {
                max_doc_id = doc_id;
            }
            if ordinal > max_ordinal {
                max_ordinal = ordinal;
            }
        }
    }

    if global_max_weight == 0.0 || !has_any {
        return Ok(0);
    }

    let max_weight_scale = global_max_weight;
    let num_ordinals = max_ordinal as u32 + 1;

    // Virtual ID space
    let max_virtual_id = max_doc_id as u64 * num_ordinals as u64 + max_ordinal as u64;
    // Safety: local_slot is u8, so block_size must not exceed 256
    let effective_block_size = bmp_block_size.min(256);
    let num_blocks = (max_virtual_id / effective_block_size as u64 + 1) as usize;

    let mut dim_ids: Vec<u32> = postings.keys().copied().collect();
    dim_ids.sort_unstable();

    // Phase 2: Flatten all postings into a single sorted Vec
    let dim_to_idx: FxHashMap<u32, usize> =
        dim_ids.iter().enumerate().map(|(i, &d)| (d, i)).collect();

    let total_postings_est: usize = postings.values().map(|v| v.len()).sum();
    let mut entries: Vec<(u32, u32, u8, u8)> = Vec::with_capacity(total_postings_est);

    for (&dim_id, dim_posts) in postings.iter() {
        for &(doc_id, ordinal, weight) in dim_posts {
            let abs_w = weight.abs();
            if abs_w < weight_threshold {
                continue;
            }
            let virtual_id = doc_id as u64 * num_ordinals as u64 + ordinal as u64;
            let block_id = (virtual_id / effective_block_size as u64) as u32;
            let local_slot = (virtual_id % effective_block_size as u64) as u8;
            let impact_u8 = quantize_weight(abs_w, max_weight_scale);

            if impact_u8 == 0 {
                continue;
            }

            entries.push((block_id, dim_id, local_slot, impact_u8));
        }
    }

    if entries.is_empty() {
        return Ok(0);
    }

    // Phase 3: Sort once by (block_id, dim_id, local_slot)
    entries.sort_unstable();

    // Phase 4: Write V5 blob
    write_bmp_blob_v5(
        &entries,
        &dim_ids,
        &dim_to_idx,
        num_blocks,
        effective_block_size,
        num_ordinals,
        max_weight_scale,
        writer,
    )
}

/// Write a BMP V5 blob from pre-sorted entries.
///
/// `entries` must be sorted by (block_id, dim_id, local_slot).
/// Grid entries are collected sparsely during the block iteration and
/// streamed dim-by-dim — no dense grid allocation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_bmp_blob_v5(
    entries: &[(u32, u32, u8, u8)],
    dim_ids: &[u32],
    dim_to_idx: &FxHashMap<u32, usize>,
    num_blocks: usize,
    bmp_block_size: u32,
    num_ordinals: u32,
    max_weight_scale: f32,
    writer: &mut dyn Write,
) -> std::io::Result<u64> {
    let num_dims = dim_ids.len();

    // Single pass: build flat arrays from sorted entries + collect grid entries
    let mut block_term_starts: Vec<u32> = Vec::with_capacity(num_blocks + 1);
    let mut term_dim_ids: Vec<u32> = Vec::new();
    let mut term_posting_starts: Vec<u32> = Vec::new();
    let mut postings_flat: Vec<u8> = Vec::new(); // (local_slot, impact) pairs
    let mut grid_entries: Vec<(u32, u32, u8)> = Vec::new(); // (dim_idx, block_id, max_impact)

    let mut entry_idx = 0;
    let mut posting_count: u32 = 0;

    for block_id in 0..num_blocks as u32 {
        block_term_starts.push(term_dim_ids.len() as u32);

        // Find all entries for this block
        let block_start = entry_idx;
        while entry_idx < entries.len() && entries[entry_idx].0 == block_id {
            entry_idx += 1;
        }
        let block_entries = &entries[block_start..entry_idx];

        if block_entries.is_empty() {
            continue;
        }

        // Group by dim_id (entries are sorted by dim_id within block)
        let mut i = 0;
        while i < block_entries.len() {
            let dim_id = block_entries[i].1;
            let group_start = i;
            let mut max_impact = 0u8;
            while i < block_entries.len() && block_entries[i].1 == dim_id {
                if block_entries[i].3 > max_impact {
                    max_impact = block_entries[i].3;
                }
                i += 1;
            }

            term_dim_ids.push(dim_id);
            term_posting_starts.push(posting_count);

            for e in &block_entries[group_start..i] {
                postings_flat.push(e.2); // local_slot
                postings_flat.push(e.3); // impact
                posting_count += 1;
            }

            // Collect sparse grid entry (no dense grid allocation)
            let dim_idx = dim_to_idx[&dim_id];
            grid_entries.push((dim_idx as u32, block_id, max_impact));
        }
    }
    // Sentinel for block_term_starts
    block_term_starts.push(term_dim_ids.len() as u32);
    // Sentinel for term_posting_starts
    term_posting_starts.push(posting_count);

    let total_terms = term_dim_ids.len() as u32;
    let total_postings = posting_count;

    // Sort grid entries by (dim_idx, block_id) for streaming write
    grid_entries.sort_unstable();

    // ── Write sections ──────────────────────────────────────────────────
    let mut bytes_written: u64 = 0;

    // Section 1: block_term_starts [u32-LE × (num_blocks + 1)]
    bytes_written += write_u32_slice_le(writer, &block_term_starts)?;

    // Section 2: term_dim_ids [u32-LE × total_terms]
    bytes_written += write_u32_slice_le(writer, &term_dim_ids)?;

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
    bytes_written += write_u32_slice_le(writer, dim_ids)?;

    // Sections 7+8: packed grid + sb_grid (streaming from sparse grid entries)
    let grid_offset = bytes_written as u32;
    let (packed_bytes, sb_bytes) = stream_write_grids(&grid_entries, num_dims, num_blocks, writer)?;
    let sb_grid_offset = (bytes_written + packed_bytes) as u32;
    bytes_written += packed_bytes + sb_bytes;

    // BMP V5 Footer (48 bytes)
    writer.write_u32::<LittleEndian>(total_terms)?; // 0-3
    writer.write_u32::<LittleEndian>(total_postings)?; // 4-7
    writer.write_u32::<LittleEndian>(dim_ids_offset)?; // 8-11
    writer.write_u32::<LittleEndian>(grid_offset)?; // 12-15
    writer.write_u32::<LittleEndian>(num_blocks as u32)?; // 16-19
    writer.write_u32::<LittleEndian>(num_dims as u32)?; // 20-23
    writer.write_u32::<LittleEndian>(bmp_block_size)?; // 24-27
    writer.write_u32::<LittleEndian>(num_ordinals)?; // 28-31
    writer.write_f32::<LittleEndian>(max_weight_scale)?; // 32-35
    writer.write_u32::<LittleEndian>(sb_grid_offset)?; // 36-39 sb_grid_offset
    writer.write_u32::<LittleEndian>(0)?; // 40-43 reserved
    writer.write_u32::<LittleEndian>(BMP_BLOB_MAGIC_V5)?; // 44-47
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
        let size = build_bmp_blob(&mut postings, 64, 0.0, None, &mut buf).unwrap();
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
        let size = build_bmp_blob(&mut postings, 64, 0.0, None, &mut buf).unwrap();
        assert!(size > 0);
        assert_eq!(buf.len(), size as usize);

        // Verify footer magic
        let footer_start = buf.len() - 4;
        let magic = u32::from_le_bytes(buf[footer_start..].try_into().unwrap());
        assert_eq!(magic, BMP_BLOB_MAGIC_V5);
    }

    #[test]
    fn test_build_bmp_blob_multi_ordinal() {
        let mut postings = FxHashMap::default();
        // dim 0: doc 0 ord 0, doc 0 ord 1, doc 1 ord 0
        postings.insert(0u32, vec![(0u32, 0u16, 1.0f32), (0, 1, 0.8), (1, 0, 0.5)]);

        let mut buf = Vec::new();
        let size = build_bmp_blob(&mut postings, 64, 0.0, None, &mut buf).unwrap();
        assert!(size > 0);

        // Verify footer: num_ordinals should be 2 (at offset 28-31 in 48-byte footer)
        let footer_start = buf.len() - 48;
        let fb = &buf[footer_start..];
        let num_ordinals = u32::from_le_bytes(fb[28..32].try_into().unwrap());
        assert_eq!(num_ordinals, 2);
    }
}
