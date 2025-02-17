//! BMP (Block-Max Pruning) index builder for sparse vectors.
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
//! All postings are collected into a single flat Vec, sorted once, then streamed
//! directly to the output writer block-by-block. No per-block Vec allocations,
//! no intermediate buffer for the entire blob.
//!
//! Peak memory: `entries (12B × total_postings) + grid (num_dims × num_blocks)`.
//!
//! ## Blob Layout
//!
//! ```text
//! [Block Forward Index Data]
//!   For each block:
//!     num_terms: u16
//!     For each term (sorted by dim_id):
//!       dim_id: u32
//!       num_postings: u16
//!       postings: [(local_slot: u8, impact: u8)] × num_postings
//!
//! [Block Offset Table]
//!   offsets: [u32; num_blocks]  // relative to blob start
//!
//! [Block-Max Grid]
//!   dim_ids: [u32; num_dims]
//!   grid_data: [u8; num_dims * num_blocks]  // row-major
//!
//! [BMP Footer] (32 bytes)
//!   grid_offset: u32
//!   offsets_table_offset: u32
//!   bmp_block_size: u32
//!   num_blocks: u32
//!   num_dims: u32
//!   num_ordinals: u32
//!   max_weight_scale: f32
//!   magic: u32 (BMP2)
//! ```

use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use rustc_hash::FxHashMap;

use crate::DocId;
use crate::segment::format::BMP_BLOB_MAGIC;

/// Build a BMP blob from per-dimension postings (convenience wrapper for tests).
#[cfg(test)]
fn build_bmp_blob(
    postings: &mut FxHashMap<u32, Vec<(DocId, u16, f32)>>,
    bmp_block_size: u32,
    weight_threshold: f32,
    pruning_fraction: Option<f32>,
    writer: &mut dyn Write,
) -> std::io::Result<u64> {
    build_bmp_blob_with_grid_cap(
        postings,
        bmp_block_size,
        weight_threshold,
        pruning_fraction,
        0,
        writer,
    )
}

/// Build a BMP blob with configurable grid memory cap.
///
/// If the grid (num_dims × num_blocks bytes) would exceed `max_grid_bytes`,
/// `bmp_block_size` is automatically increased (to next power of 2) to reduce
/// num_blocks until the grid fits. Set `max_grid_bytes = 0` to disable the cap.
pub(crate) fn build_bmp_blob_with_grid_cap(
    postings: &mut FxHashMap<u32, Vec<(DocId, u16, f32)>>,
    bmp_block_size: u32,
    weight_threshold: f32,
    pruning_fraction: Option<f32>,
    max_grid_bytes: u64,
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
    let mut effective_block_size = bmp_block_size;
    let mut num_blocks = (max_virtual_id / effective_block_size as u64 + 1) as usize;

    let mut dim_ids: Vec<u32> = postings.keys().copied().collect();
    dim_ids.sort_unstable();
    let num_dims = dim_ids.len();

    // Auto-scale block_size if grid would exceed memory cap
    if max_grid_bytes > 0 {
        let grid_bytes = num_dims as u64 * num_blocks as u64;
        if grid_bytes > max_grid_bytes && num_dims > 0 {
            let max_blocks = max_grid_bytes / num_dims as u64;
            if let Some(ratio) = max_virtual_id.checked_div(max_blocks) {
                effective_block_size = ((ratio) + 1).next_power_of_two() as u32;
                effective_block_size = effective_block_size.max(bmp_block_size);
                num_blocks = (max_virtual_id / effective_block_size as u64 + 1) as usize;
                log::info!(
                    "BMP grid would be {:.0}MB ({} dims × {} blocks), \
                     auto-scaled block_size {} → {} (num_blocks {})",
                    grid_bytes as f64 / (1024.0 * 1024.0),
                    num_dims,
                    grid_bytes / num_dims as u64,
                    bmp_block_size,
                    effective_block_size,
                    num_blocks,
                );
            }
        }
    }

    // Phase 2: Flatten all postings into a single sorted Vec + build grid
    let dim_to_idx: FxHashMap<u32, usize> =
        dim_ids.iter().enumerate().map(|(i, &d)| (d, i)).collect();

    let total_postings_est: usize = postings.values().map(|v| v.len()).sum();
    let mut entries: Vec<(u32, u32, u8, u8)> = Vec::with_capacity(total_postings_est);
    let mut grid = vec![0u8; num_dims * num_blocks];

    for (&dim_id, dim_posts) in postings.iter() {
        let dim_idx = dim_to_idx[&dim_id];
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

            let grid_idx = dim_idx * num_blocks + block_id as usize;
            if impact_u8 > grid[grid_idx] {
                grid[grid_idx] = impact_u8;
            }

            entries.push((block_id, dim_id, local_slot, impact_u8));
        }
    }

    if entries.is_empty() {
        return Ok(0);
    }

    // Phase 3: Sort once by (block_id, dim_id, local_slot)
    entries.sort_unstable();

    // Phase 4: Stream to writer
    write_bmp_blob_streaming(
        &entries,
        &dim_ids,
        &grid,
        num_blocks,
        effective_block_size,
        num_ordinals,
        max_weight_scale,
        writer,
    )
}

/// Stream a BMP blob from pre-sorted entries directly to the writer.
///
/// `entries` must be sorted by (block_id, dim_id, local_slot).
/// `grid` is row-major: grid[dim_idx * num_blocks + block_id] = max impact.
///
/// This is the shared core for both initial segment building and merging.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_bmp_blob_streaming(
    entries: &[(u32, u32, u8, u8)],
    dim_ids: &[u32],
    grid: &[u8],
    num_blocks: usize,
    bmp_block_size: u32,
    num_ordinals: u32,
    max_weight_scale: f32,
    writer: &mut dyn Write,
) -> std::io::Result<u64> {
    let num_dims = dim_ids.len();
    let mut bytes_written: u64 = 0;
    let mut block_offsets: Vec<u32> = Vec::with_capacity(num_blocks);
    let mut entry_idx = 0;

    for block_id in 0..num_blocks as u32 {
        block_offsets.push(bytes_written as u32);

        // Find all entries for this block (entries are sorted by block_id)
        let block_start = entry_idx;
        while entry_idx < entries.len() && entries[entry_idx].0 == block_id {
            entry_idx += 1;
        }

        let block_entries = &entries[block_start..entry_idx];

        if block_entries.is_empty() {
            writer.write_u16::<LittleEndian>(0)?;
            bytes_written += 2;
            continue;
        }

        // Count unique dims (entries are sorted by dim_id within block)
        let num_terms = count_unique_dims(block_entries);
        writer.write_u16::<LittleEndian>(num_terms as u16)?;
        bytes_written += 2;

        // Write term groups (already grouped by dim_id due to sort order)
        let mut i = 0;
        while i < block_entries.len() {
            let dim_id = block_entries[i].1;
            let group_start = i;
            while i < block_entries.len() && block_entries[i].1 == dim_id {
                i += 1;
            }
            let count = (i - group_start) as u16;
            writer.write_u32::<LittleEndian>(dim_id)?;
            writer.write_u16::<LittleEndian>(count)?;
            bytes_written += 6;
            for e in &block_entries[group_start..i] {
                writer.write_all(&[e.2, e.3])?;
                bytes_written += 2;
            }
        }
    }

    // Block offset table
    let offsets_table_offset = bytes_written as u32;
    for &offset in &block_offsets {
        writer.write_u32::<LittleEndian>(offset)?;
    }
    bytes_written += num_blocks as u64 * 4;

    // Block-max grid: dim_ids then row-major grid data
    let grid_offset = bytes_written as u32;
    for &dim_id in dim_ids {
        writer.write_u32::<LittleEndian>(dim_id)?;
    }
    bytes_written += num_dims as u64 * 4;
    writer.write_all(grid)?;
    bytes_written += grid.len() as u64;

    // BMP footer (32 bytes)
    writer.write_u32::<LittleEndian>(grid_offset)?;
    writer.write_u32::<LittleEndian>(offsets_table_offset)?;
    writer.write_u32::<LittleEndian>(bmp_block_size)?;
    writer.write_u32::<LittleEndian>(num_blocks as u32)?;
    writer.write_u32::<LittleEndian>(num_dims as u32)?;
    writer.write_u32::<LittleEndian>(num_ordinals)?;
    writer.write_f32::<LittleEndian>(max_weight_scale)?;
    writer.write_u32::<LittleEndian>(BMP_BLOB_MAGIC)?;
    bytes_written += 32;

    Ok(bytes_written)
}

/// Count unique dim_ids in a slice sorted by (block_id, dim_id, local_slot).
fn count_unique_dims(entries: &[(u32, u32, u8, u8)]) -> usize {
    if entries.is_empty() {
        return 0;
    }
    let mut count = 1;
    let mut prev = entries[0].1;
    for e in &entries[1..] {
        if e.1 != prev {
            count += 1;
            prev = e.1;
        }
    }
    count
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
        assert_eq!(magic, BMP_BLOB_MAGIC);
    }

    #[test]
    fn test_build_bmp_blob_multi_ordinal() {
        let mut postings = FxHashMap::default();
        // dim 0: doc 0 ord 0, doc 0 ord 1, doc 1 ord 0
        postings.insert(0u32, vec![(0u32, 0u16, 1.0f32), (0, 1, 0.8), (1, 0, 0.5)]);

        let mut buf = Vec::new();
        let size = build_bmp_blob(&mut postings, 64, 0.0, None, &mut buf).unwrap();
        assert!(size > 0);

        // Verify footer: num_ordinals should be 2
        let footer_start = buf.len() - 32;
        let fb = &buf[footer_start..];
        let num_ordinals = u32::from_le_bytes(fb[20..24].try_into().unwrap());
        assert_eq!(num_ordinals, 2);
    }
}
