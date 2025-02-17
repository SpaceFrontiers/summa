//! BMP (Block-Max Pruning) index builder for sparse vectors — **V3 format**.
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
//! Peak memory: `entries (12B × total_postings) + grid (num_dims × num_blocks)`.
//!
//! ## BMP V3 Blob Layout
//!
//! Contiguous arrays for zero-copy mmap at read time:
//!
//! ```text
//! Section 1: block_term_starts    [u32-LE × (num_blocks + 1)]
//! Section 2: term_dim_ids         [u32-LE × total_terms]
//! Section 3: term_posting_starts  [u32-LE × (total_terms + 1)]    ← prefix sums
//! Section 4: postings             [(u8, u8) × total_postings]     ← BmpPosting pairs
//! Section 5: padding              [0-3 bytes to next 4-byte boundary]
//! Section 6: dim_ids              [u32-LE × num_dims]
//! Section 7: grid                 [u8 × (num_dims × num_blocks)]
//!
//! BMP3 Footer (48 bytes):
//!   total_terms: u32              // 0-3
//!   total_postings: u32           // 4-7
//!   dim_ids_offset: u32           // 8-11   (byte offset of section 6)
//!   grid_offset: u32              // 12-15  (byte offset of section 7)
//!   num_blocks: u32               // 16-19
//!   num_dims: u32                 // 20-23
//!   bmp_block_size: u32           // 24-27
//!   num_ordinals: u32             // 28-31
//!   max_weight_scale: f32         // 32-35
//!   _reserved: u32                // 36-39
//!   _reserved: u32                // 40-43
//!   magic: u32                    // 44-47  (BMP3 = 0x33504D42)
//! ```

use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use rustc_hash::FxHashMap;

use crate::DocId;
use crate::segment::format::BMP_BLOB_MAGIC_V3;

/// Build a BMP V3 blob from per-dimension postings.
///
/// The grid (num_dims × num_blocks bytes) is allocated as a transient Vec
/// and freed after writing. `bmp_block_size` is clamped to 256 max (u8 local_slot).
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
    let num_dims = dim_ids.len();

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

    // Phase 4: Write V3 blob
    write_bmp_blob_v3(
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

/// Write a BMP V3 blob from pre-sorted entries.
///
/// `entries` must be sorted by (block_id, dim_id, local_slot).
/// `grid` is row-major: grid[dim_idx * num_blocks + block_id] = max impact.
///
/// V3 writes contiguous arrays for zero-copy mmap at read time.
/// This is the shared core for both initial segment building and merging.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_bmp_blob_v3(
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

    // Single pass: build flat arrays from sorted entries
    let mut block_term_starts: Vec<u32> = Vec::with_capacity(num_blocks + 1);
    let mut term_dim_ids: Vec<u32> = Vec::new();
    let mut term_posting_starts: Vec<u32> = Vec::new();
    let mut postings_flat: Vec<u8> = Vec::new(); // (local_slot, impact) pairs

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
            while i < block_entries.len() && block_entries[i].1 == dim_id {
                i += 1;
            }

            term_dim_ids.push(dim_id);
            term_posting_starts.push(posting_count);

            for e in &block_entries[group_start..i] {
                postings_flat.push(e.2); // local_slot
                postings_flat.push(e.3); // impact
                posting_count += 1;
            }
        }
    }
    // Sentinel for block_term_starts
    block_term_starts.push(term_dim_ids.len() as u32);
    // Sentinel for term_posting_starts
    term_posting_starts.push(posting_count);

    let total_terms = term_dim_ids.len() as u32;
    let total_postings = posting_count;

    // ── Write sections ──────────────────────────────────────────────────
    let mut bytes_written: u64 = 0;

    // Section 1: block_term_starts [u32-LE × (num_blocks + 1)]
    for &v in &block_term_starts {
        writer.write_u32::<LittleEndian>(v)?;
    }
    bytes_written += block_term_starts.len() as u64 * 4;

    // Section 2: term_dim_ids [u32-LE × total_terms]
    for &v in &term_dim_ids {
        writer.write_u32::<LittleEndian>(v)?;
    }
    bytes_written += total_terms as u64 * 4;

    // Section 3: term_posting_starts [u32-LE × (total_terms + 1)]
    for &v in &term_posting_starts {
        writer.write_u32::<LittleEndian>(v)?;
    }
    bytes_written += term_posting_starts.len() as u64 * 4;

    // Section 4: postings [(u8, u8) × total_postings]
    writer.write_all(&postings_flat)?;
    bytes_written += postings_flat.len() as u64;

    // Section 5: padding to 4-byte boundary
    let padding = (4 - (bytes_written % 4) as usize) % 4;
    if padding > 0 {
        writer.write_all(&vec![0u8; padding])?;
        bytes_written += padding as u64;
    }

    // Section 6: dim_ids [u32-LE × num_dims]
    let dim_ids_offset = bytes_written as u32;
    for &dim_id in dim_ids {
        writer.write_u32::<LittleEndian>(dim_id)?;
    }
    bytes_written += num_dims as u64 * 4;

    // Section 7: grid [u8 × (num_dims × num_blocks)]
    let grid_offset = bytes_written as u32;
    writer.write_all(grid)?;
    bytes_written += grid.len() as u64;

    // BMP V3 Footer (48 bytes)
    writer.write_u32::<LittleEndian>(total_terms)?; // 0-3
    writer.write_u32::<LittleEndian>(total_postings)?; // 4-7
    writer.write_u32::<LittleEndian>(dim_ids_offset)?; // 8-11
    writer.write_u32::<LittleEndian>(grid_offset)?; // 12-15
    writer.write_u32::<LittleEndian>(num_blocks as u32)?; // 16-19
    writer.write_u32::<LittleEndian>(num_dims as u32)?; // 20-23
    writer.write_u32::<LittleEndian>(bmp_block_size)?; // 24-27
    writer.write_u32::<LittleEndian>(num_ordinals)?; // 28-31
    writer.write_f32::<LittleEndian>(max_weight_scale)?; // 32-35
    writer.write_u32::<LittleEndian>(0)?; // 36-39 reserved
    writer.write_u32::<LittleEndian>(0)?; // 40-43 reserved
    writer.write_u32::<LittleEndian>(BMP_BLOB_MAGIC_V3)?; // 44-47
    bytes_written += 48;

    Ok(bytes_written)
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
        assert_eq!(magic, BMP_BLOB_MAGIC_V3);
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
