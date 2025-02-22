//! BMP (Block-Max Pruning) query executor for sparse vectors.
//!
//! Superblock-at-a-time (SaaT) processor that groups BMP blocks into superblocks
//! and uses hierarchical pruning for faster query execution.
//!
//! Uses **compact virtual coordinates**: sequential IDs assigned to unique
//! `(doc_id, ordinal)` pairs. A doc_map lookup table maps virtual IDs back
//! to original coordinates at query time.
//!
//! Based on:
//! - Mallia, Suel & Tonellotto (SIGIR 2024): BMP block-at-a-time processing
//! - Carlson et al. (SIGIR 2025): Superblock pruning for learned sparse retrieval
//! - Carlson et al. (arXiv 2602.02883, 2026): LSP/0 gamma cap, integer scoring
//!
//! ## Three-level pruning hierarchy
//!
//! 1. **LSP/0 gamma cap**: Hard limit on superblock visits (prevents tail waste)
//! 2. **Superblock UBs** (~1.2K entries at 1M×5): cheap to compute, prune 25-75%
//! 3. **Block UBs** (only for surviving superblocks): L1-cache friendly per-SB
//!
//! ## Performance
//!
//! All block data is pre-decoded at index load time. Query execution touches
//! only flat contiguous arrays — no file I/O, no parsing, no heap allocation
//! in the hot path.
//!
//! - **LSP/0 gamma cap**: Hard limit on superblock visits for predictable latency
//! - **Integer scoring**: u32 accumulators with u16 quantized query weights (~20% faster)
//! - **Superblock pruning**: Skip entire groups of blocks via coarse UBs
//! - **L1 cache locality**: SaaT loop keeps c=64 blocks' grid data in L1
//! - **SIMD UB computation**: NEON-accelerated for both superblock and block levels
//! - **Pre-scaled weights**: `weight * scale` computed once, not per-block
//! - **Bitmask skip**: Register-level mask check replaces grid DRAM lookups
//! - **Bucket sort**: O(n) superblock ordering by UB descending
//! - **Binary search scoring**: O(|query| × log|block_terms|) per block
//! - **Multi-level prefetch**: SB offset warming → pre-loop burst → N+1/N+2 data pipeline
//! - **Thread-local scratch**: Zero per-query allocation for large buffers
//! - **Early termination**: stop when superblock/block UB < top-k threshold

use super::scoring::{ScoreCollector, ScoredDoc};
use crate::segment::{
    BMP_SUPERBLOCK_SIZE, BmpIndex, accumulate_u4_weighted, block_term_postings,
    compute_block_masks_4bit, find_dim_in_block_data,
};

// dim_id is used directly as grid row index. No dim_idx indirection.

// ============================================================================
// Software prefetch: hint the CPU to load data into cache ahead of time
// ============================================================================

/// Prefetch a memory location for reading with temporal locality.
///
/// This is a no-op on unsupported platforms. On aarch64/x86_64 it issues
/// a hardware prefetch hint that has zero cost if the data is already cached.
#[inline(always)]
fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) ptr,
            options(nostack, preserves_flags)
        );
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

// ============================================================================
// Thread-local scratch buffers: zero per-query allocation
// ============================================================================

/// Reusable scratch buffers for BMP query execution.
///
/// Two-level hierarchy:
/// - **Superblock-level**: sized to num_superblocks, for SB ordering + early termination
/// - **Local block-level**: sized to BMP_SUPERBLOCK_SIZE (64), for per-SB block computation
/// - **Accumulator**: u32 sized to bmp_block_size, reused per block (integer scoring)
#[derive(Default)]
struct BmpScratch {
    // Superblock-level (reused across queries, sized to num_superblocks)
    sb_ubs: Vec<f32>,
    sb_order: Vec<u32>,
    // Block-level (reused per superblock, sized to BMP_SUPERBLOCK_SIZE)
    local_block_ubs: Vec<f32>,
    local_block_masks: Vec<u64>,
    local_block_order: Vec<u32>,
    // Per-slot accumulator (sized to block_size) — u32 for integer scoring
    acc: Vec<u32>,
    // Compact grid buffers: query-relevant dim rows copied contiguously for L1 locality.
    // Resized per query to num_query_dims × row_size. Typical: 20 dims → ~16KB total.
    compact_sb_grid: Vec<u8>,
    compact_grid: Vec<u8>,
}

impl BmpScratch {
    /// Ensure superblock + local buffers have sufficient capacity.
    fn ensure_capacity_sb(&mut self, num_superblocks: usize, sb_size: usize, block_size: usize) {
        if self.sb_ubs.len() < num_superblocks {
            self.sb_ubs.resize(num_superblocks, 0.0);
        }
        if self.sb_order.capacity() < num_superblocks {
            self.sb_order.reserve(num_superblocks - self.sb_order.len());
        }
        if self.local_block_ubs.len() < sb_size {
            self.local_block_ubs.resize(sb_size, 0.0);
        }
        if self.local_block_masks.len() < sb_size {
            self.local_block_masks.resize(sb_size, 0u64);
        }
        if self.local_block_order.len() < sb_size {
            self.local_block_order.resize(sb_size, 0);
        }
        if self.acc.len() < block_size {
            self.acc.resize(block_size, 0);
        }
    }
}

thread_local! {
    static BMP_SCRATCH: std::cell::RefCell<BmpScratch> =
        std::cell::RefCell::new(BmpScratch::default());
}

/// Execute a BMP query against the given index.
///
/// Returns top-k results sorted by score descending.
/// Scores are computed over compact virtual documents and resolved to real
/// (doc_id, ordinal) pairs at collection time. The caller's `combine_ordinal_results`
/// handles multi-ordinal grouping via the configured combiner.
///
/// Uses superblock pruning: computes coarse UBs over groups of 64 blocks,
/// prunes entire superblocks, then scores only surviving blocks.
///
/// `heap_factor` controls approximate retrieval (BMP alpha parameter):
/// - **1.0**: exact/safe retrieval (default)
/// - **0.8**: prune when `UB * 0.8 <= threshold` → ~20% more aggressive
/// - **0.6**: prune when `UB * 0.6 <= threshold` → ~40% more aggressive
///
/// `max_superblocks` (LSP/0 gamma cap): hard limit on superblock visits.
/// - **0**: unlimited (default, existing behavior)
/// - **>0**: stop after visiting this many superblocks
///
/// Based on Mallia et al. (SIGIR 2024) and Carlson et al. (arXiv 2602.02883).
pub fn execute_bmp(
    index: &BmpIndex,
    query_terms: &[(u32, f32)],
    k: usize,
    heap_factor: f32,
    max_superblocks: usize,
) -> crate::Result<Vec<ScoredDoc>> {
    execute_bmp_inner(index, query_terms, k, heap_factor, max_superblocks, None)
}

/// Execute a BMP query with a document predicate filter.
///
/// Same as [`execute_bmp`] but only collects documents that pass the predicate.
/// The predicate is checked during scoring (not post-filter), so the collector
/// only contains valid documents and the threshold evolves correctly.
pub fn execute_bmp_filtered(
    index: &BmpIndex,
    query_terms: &[(u32, f32)],
    k: usize,
    heap_factor: f32,
    max_superblocks: usize,
    predicate: &dyn Fn(crate::DocId) -> bool,
) -> crate::Result<Vec<ScoredDoc>> {
    execute_bmp_inner(
        index,
        query_terms,
        k,
        heap_factor,
        max_superblocks,
        Some(predicate),
    )
}

fn execute_bmp_inner(
    index: &BmpIndex,
    query_terms: &[(u32, f32)],
    k: usize,
    heap_factor: f32,
    max_superblocks: usize,
    predicate: Option<&dyn Fn(crate::DocId) -> bool>,
) -> crate::Result<Vec<ScoredDoc>> {
    if query_terms.is_empty() || index.num_blocks == 0 {
        return Ok(Vec::new());
    }

    // Alpha parameter for approximate retrieval:
    // UB * alpha <= threshold → prune when UB < threshold / alpha
    // alpha < 1.0 means more aggressive pruning (approximate but faster)
    let alpha = heap_factor.clamp(0.01, 1.0);
    let scale = index.max_weight_scale / 255.0;
    let num_blocks = index.num_blocks as usize;
    let block_size = index.bmp_block_size as usize;
    let num_superblocks_total = index.num_superblocks as usize;

    // ── Phase 1: Resolve query dims and pre-scale weights ──────────────
    // dim_id is used directly as grid row index (no dim_idx indirection).
    // Build combined info, sort by dim_id, then split into resolved + query_by_dim_u16.
    // Both arrays MUST have the same ordering so mask bit `q` corresponds to the
    // same dimension in both compute_block_masks (uses resolved) and
    // score_block_bsearch_int (uses query_by_dim_u16).
    let dims = index.dims();
    let mut query_info: Vec<(u32, usize, f32)> = Vec::with_capacity(query_terms.len());

    for &(dim_id, weight) in query_terms {
        // dim_id IS the grid row index — just check bounds
        if dim_id < dims {
            let scaled = weight * scale;
            query_info.push((dim_id, dim_id as usize, scaled));
        }
    }

    if query_info.is_empty() {
        return Ok(Vec::new());
    }

    // Sort by dim_id for binary search within blocks (blocks store dim_id directly).
    query_info.sort_unstable_by_key(|&(dim_id, _, _)| dim_id);

    // Split into parallel arrays with matching order
    // resolved: (dim_id_as_grid_row, f32_weight) — for grid UB computation
    let resolved: Vec<(usize, f32)> = query_info.iter().map(|&(_, idx, w)| (idx, w)).collect();

    // Integer scoring: quantize query weights to u16 for u32 accumulator path
    // Max accumulator = 16383 × 255 × num_dims. At 1024 dims: 4.27B < u32::MAX.
    let max_scaled = query_info.iter().map(|q| q.2.abs()).fold(0.0f32, f32::max);
    let (quant_scale, dequant) = if max_scaled > 0.0 {
        (16383.0 / max_scaled, max_scaled / 16383.0)
    } else {
        (0.0, 0.0)
    };
    // query_by_dim_u16 stores (dim_id, weight) — matches per-block dim_ids
    let query_by_dim_u16: Vec<(u32, u16)> = query_info
        .iter()
        .map(|&(dim_id, _, w)| {
            (
                dim_id,
                (w.abs() * quant_scale).round().clamp(0.0, 16383.0) as u16,
            )
        })
        .collect();

    // Over-fetch for multi-ordinal: each real doc may occupy multiple collector
    // slots (one per ordinal). Inflate collector so combine_ordinal_results has
    // enough unique docs after grouping. Cap at 10× to avoid degenerate cases
    // (e.g., 200 ordinals/doc). For single-ordinal, collector_k == k (no overhead).
    let ordinals_per_doc = if index.num_real_docs() > 0 {
        (index.num_virtual_docs as f32 / index.num_real_docs() as f32).ceil() as usize
    } else {
        1
    };
    let collector_k = (k * ordinals_per_doc).min(k * 10);

    let t_start = std::time::Instant::now();

    let result = BMP_SCRATCH.with(|cell| {
        let scratch = &mut *cell.borrow_mut();

        // ── Superblock-at-a-time scoring ─────────────────────────────
        scratch.ensure_capacity_sb(
            num_superblocks_total,
            BMP_SUPERBLOCK_SIZE as usize,
            block_size,
        );

        let prs = index.packed_row_size();

        // ── Grid access strategy ─────────────────────────────────────
        // For small segments: copy query-relevant grid rows into compact
        // buffers that fit L1 cache (~16KB for 20 dims × 750 blocks).
        // For large segments: use mmap-backed grid directly (zero alloc)
        // to avoid multi-MB thread-local scratch that never shrinks.
        const COMPACT_GRID_MAX: usize = 128 * 1024; // 128KB — fits L2 comfortably
        let compact_sb_size = resolved.len() * num_superblocks_total;
        let compact_grid_size = resolved.len() * prs;
        let use_compact = compact_sb_size + compact_grid_size <= COMPACT_GRID_MAX;

        // grid_dims: (dim_index_into_grid, weight) — local for compact, global for direct
        let grid_dims: Vec<(usize, f32)>;
        // sb_int_weights: (dim_index_into_sb_grid, u16_weight) — for integer-consistent SB UBs
        let sb_int_weights: Vec<(usize, u16)>;
        let sb_grid_slice: &[u8];
        let grid_slice: &[u8];

        if use_compact {
            let dim_indices: Vec<usize> = resolved.iter().map(|&(idx, _)| idx).collect();
            index.extract_compact_grids(
                &dim_indices,
                &mut scratch.compact_sb_grid,
                &mut scratch.compact_grid,
            );
            grid_dims = (0..resolved.len()).map(|i| (i, resolved[i].1)).collect();
            sb_int_weights = (0..resolved.len())
                .map(|i| (i, query_by_dim_u16[i].1))
                .collect();
            sb_grid_slice = &scratch.compact_sb_grid;
            grid_slice = &scratch.compact_grid;
        } else {
            // Direct mmap access — zero allocation. Grid layout uses global dim indices.
            grid_dims = resolved.clone();
            sb_int_weights = resolved
                .iter()
                .enumerate()
                .map(|(i, &(idx, _))| (idx, query_by_dim_u16[i].1))
                .collect();
            sb_grid_slice = index.sb_grid_slice();
            grid_slice = index.grid_slice();
            // Shrink oversized compact buffers from previous queries
            if scratch.compact_grid.capacity() > COMPACT_GRID_MAX {
                scratch.compact_sb_grid = Vec::new();
                scratch.compact_grid = Vec::new();
            }
        }

        // Phase 2: Compute SUPERBLOCK UBs using integer weights (matching scoring path)
        compute_sb_ubs_int(
            sb_grid_slice,
            num_superblocks_total,
            &sb_int_weights,
            dequant,
            &mut scratch.sb_ubs,
        );
        sort_sb_desc_into(
            &scratch.sb_ubs[..num_superblocks_total],
            &mut scratch.sb_order,
        );

        if scratch.sb_order.is_empty() {
            return Vec::new();
        }

        // Phase 3: Score superblocks in UB-descending order
        let mut blocks_scored = 0u32;
        let mut sbs_scored = 0u32;
        let mut collector = ScoreCollector::new(collector_k);

        for (idx, &sb_id) in scratch.sb_order.iter().enumerate() {
            // LSP/0: hard cap on superblock visits
            if max_superblocks > 0 && idx >= max_superblocks {
                break;
            }

            let sb_ub = scratch.sb_ubs[sb_id as usize];
            // Threshold pruning still applies within the cap
            if collector.len() >= collector_k && sb_ub * alpha <= collector.threshold() {
                break;
            }

            let block_start = sb_id as usize * BMP_SUPERBLOCK_SIZE as usize;
            let block_end = (block_start + BMP_SUPERBLOCK_SIZE as usize).min(num_blocks);
            let count = block_end - block_start;

            // Level 1: Warm block_data_starts for this superblock (~520 bytes, 8-9 cache lines).
            // Ensures block_data_ptr() never stalls on offset lookups during scoring.
            // Range includes the sentinel at block_end (needed by block_data_range).
            {
                let bds_base = index.block_data_starts_ptr(0);
                // 8 blocks × 8 bytes = 64 bytes = 1 cache line per prefetch
                for b in (block_start..block_end + 1).step_by(8) {
                    prefetch_read(unsafe { bds_base.add(b * 8) });
                }
            }

            // Compute block UBs + masks from grid
            compute_block_ubs_compact(
                grid_slice,
                prs,
                &grid_dims,
                block_start,
                block_end,
                &mut scratch.local_block_ubs,
            );
            compute_block_masks_4bit(
                grid_slice,
                prs,
                &grid_dims,
                block_start,
                block_end - block_start,
                &mut scratch.local_block_masks,
            );

            sort_local_blocks_desc(
                &scratch.local_block_ubs[..count],
                &mut scratch.local_block_order,
            );

            score_superblock_blocks(
                index,
                block_start,
                count,
                &scratch.local_block_order,
                &scratch.local_block_ubs,
                &scratch.local_block_masks,
                &query_by_dim_u16,
                dequant,
                alpha,
                collector_k,
                &predicate,
                &mut collector,
                &mut blocks_scored,
                &mut scratch.acc,
            );

            // Cross-superblock lookahead: prefetch next superblock's block_data_starts.
            // Gives offsets time to arrive during pruning check + UB/mask computation.
            // Range includes the sentinel at next_end (needed by block_data_range).
            if let Some(&next_sb) = scratch.sb_order.get(idx + 1) {
                let next_start = next_sb as usize * BMP_SUPERBLOCK_SIZE as usize;
                let next_end = (next_start + BMP_SUPERBLOCK_SIZE as usize).min(num_blocks);
                let bds_base = index.block_data_starts_ptr(0);
                for b in (next_start..next_end + 1).step_by(8) {
                    prefetch_read(unsafe { bds_base.add(b * 8) });
                }
            }

            sbs_scored += 1;
        }

        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        let threshold = collector.threshold();
        if elapsed_ms > 500.0 {
            log::warn!(
                "slow BMP: {:.1}ms, sbs={}/{}, blocks={}/{}, returned={}, threshold={:.4}, alpha={:.2}",
                elapsed_ms,
                sbs_scored,
                num_superblocks_total,
                blocks_scored,
                num_blocks,
                collector.len(),
                threshold,
                alpha,
            );
        } else {
            log::debug!(
                "BMP execute: {:.1}ms, sbs={}/{}, blocks={}/{}, returned={}, threshold={:.4}, alpha={:.2}",
                elapsed_ms,
                sbs_scored,
                num_superblocks_total,
                blocks_scored,
                num_blocks,
                collector.len(),
                threshold,
                alpha,
            );
        }

        // Diagnostic: check if any pruned superblock has UB above effective
        // pruning threshold (threshold / alpha). Only a real bug if alpha=1.0
        // and SBs with UB > threshold were skipped.
        if log::log_enabled!(log::Level::Debug) && sbs_scored < num_superblocks_total as u32 {
            let effective_threshold = if alpha > 0.0 { threshold / alpha } else { threshold };
            let mut pruned_above = 0u32;
            let mut max_pruned_ub = 0.0f32;
            for sb in 0..num_superblocks_total {
                let ub = scratch.sb_ubs[sb];
                if ub > effective_threshold && ub > 0.0 {
                    // Check if this SB was actually visited
                    let was_visited = scratch.sb_order[..sbs_scored as usize]
                        .iter()
                        .any(|&id| id as usize == sb);
                    if !was_visited {
                        pruned_above += 1;
                        max_pruned_ub = max_pruned_ub.max(ub);
                    }
                }
            }
            if pruned_above > 0 {
                log::warn!(
                    "BMP PRUNING BUG: {} superblocks with UB > {:.4} (threshold/alpha) pruned, max_pruned_ub={:.4}",
                    pruned_above,
                    effective_threshold,
                    max_pruned_ub,
                );
            }
        }

        collector_to_results(collector)
    });

    // Verify: compare pruned results with exhaustive (no-pruning) scoring.
    // Only runs at DEBUG level. Detects any pruning-induced result divergence.
    if log::log_enabled!(log::Level::Debug) {
        let exhaustive = execute_bmp_exhaustive(index, query_terms, k)?;
        let mut pruned_ids: Vec<(u32, u16)> =
            result.iter().map(|r| (r.doc_id, r.ordinal)).collect();
        let mut exhaust_ids: Vec<(u32, u16)> =
            exhaustive.iter().map(|r| (r.doc_id, r.ordinal)).collect();
        pruned_ids.sort_unstable();
        exhaust_ids.sort_unstable();
        if pruned_ids != exhaust_ids {
            let missing: Vec<_> = exhaust_ids
                .iter()
                .filter(|id| !pruned_ids.contains(id))
                .collect();
            let extra: Vec<_> = pruned_ids
                .iter()
                .filter(|id| !exhaust_ids.contains(id))
                .collect();
            log::warn!(
                "BMP VERIFY MISMATCH: pruned returned {} results, exhaustive returned {}. \
                 missing={} (in exhaustive but not pruned), extra={} (in pruned but not exhaustive). \
                 Top exhaustive scores: {:?}",
                result.len(),
                exhaustive.len(),
                missing.len(),
                extra.len(),
                exhaustive
                    .iter()
                    .take(5)
                    .map(|r| (r.doc_id, r.score))
                    .collect::<Vec<_>>(),
            );
        }
    }

    Ok(result)
}

/// Exhaustive BMP execution: score ALL blocks without any pruning.
///
/// For debugging: bypasses superblock/block pruning entirely.
/// Scores every non-empty block and returns top-k results.
/// Compare with `execute_bmp` to identify pruning-related issues.
pub fn execute_bmp_exhaustive(
    index: &BmpIndex,
    query_terms: &[(u32, f32)],
    k: usize,
) -> crate::Result<Vec<ScoredDoc>> {
    if query_terms.is_empty() || index.num_blocks == 0 {
        return Ok(Vec::new());
    }

    let scale = index.max_weight_scale / 255.0;
    let num_blocks = index.num_blocks as usize;
    let block_size = index.bmp_block_size as usize;
    let dims = index.dims();

    let mut query_info: Vec<(u32, f32)> = Vec::with_capacity(query_terms.len());
    for &(dim_id, weight) in query_terms {
        if dim_id < dims {
            query_info.push((dim_id, weight * scale));
        }
    }
    if query_info.is_empty() {
        return Ok(Vec::new());
    }
    query_info.sort_unstable_by_key(|&(dim_id, _)| dim_id);

    let max_scaled = query_info.iter().map(|q| q.1.abs()).fold(0.0f32, f32::max);
    let (quant_scale, dequant) = if max_scaled > 0.0 {
        (16383.0 / max_scaled, max_scaled / 16383.0)
    } else {
        (0.0, 0.0)
    };
    let query_by_dim_u16: Vec<(u32, u16)> = query_info
        .iter()
        .map(|&(dim_id, w)| {
            (
                dim_id,
                (w.abs() * quant_scale).round().clamp(0.0, 16383.0) as u16,
            )
        })
        .collect();

    let ordinals_per_doc = if index.num_real_docs() > 0 {
        (index.num_virtual_docs as f32 / index.num_real_docs() as f32).ceil() as usize
    } else {
        1
    };
    let collector_k = (k * ordinals_per_doc).min(k * 10).max(k);
    let mut collector = ScoreCollector::new(collector_k);
    let mut acc = vec![0u32; block_size];
    let num_vdocs = index.num_virtual_docs as usize;
    let mut blocks_scored = 0u32;

    for block_id in 0..num_blocks as u32 {
        let (num_terms, dim_ptr, ps_ptr, post_ptr) = index.parse_block(block_id);
        if num_terms == 0 {
            continue;
        }

        let mut touched = [0u64; 4];
        score_block_bsearch_int(
            num_terms,
            dim_ptr,
            ps_ptr,
            post_ptr,
            &query_by_dim_u16,
            u64::MAX, // no mask filtering
            &mut acc,
            &mut touched,
        );

        let base = block_id as usize * block_size;
        for (word, &touch_word) in touched.iter().enumerate() {
            let mut scan = touch_word;
            while scan != 0 {
                let bit = scan.trailing_zeros() as usize;
                scan &= scan - 1;
                let i = word * 64 + bit;
                let score_u32 = acc[i];
                acc[i] = 0;
                if score_u32 == 0 {
                    continue;
                }
                let virtual_id = base + i;
                if virtual_id >= num_vdocs {
                    continue;
                }
                let (doc_id, ordinal) = index.virtual_to_doc(virtual_id as u32);
                if doc_id == u32::MAX {
                    continue;
                }
                let score = score_u32 as f32 * dequant;
                if collector.would_enter(score) {
                    collector.insert_with_ordinal(doc_id, score, ordinal);
                }
            }
        }
        blocks_scored += 1;
    }

    log::debug!(
        "BMP exhaustive: blocks={}/{}, returned={}",
        blocks_scored,
        num_blocks,
        collector.len(),
    );

    Ok(collector_to_results(collector))
}

// ============================================================================
// Integer scoring: u32 accumulators with u16 quantized query weights
// ============================================================================

/// Score a block using integer arithmetic (u32 accumulators, u16 weights).
///
/// Uses **bitmask skip**: checks `block_mask & (1 << q) != 0` before binary search.
/// Accumulates `w_u16 * impact_u8` into u32 — eliminates u8→f32 conversion per posting.
///
/// Blocks store u32 dim_id directly. Binary search on dim_id (always 4 bytes).
///
/// Block data is contiguous — `dim_ptr`, `ps_ptr`, `post_ptr` point into
/// the same ~200-2000 byte region (1-2 pages). Binary search and posting reads
/// touch only this contiguous region.
///
/// Tracks touched slots via a `[u64; 4]` bitmask (works for block_size ≤ 256).
/// Caller uses the bitmask for lazy accumulator zeroing.
///
/// Complexity: O(|present_query_dims| × log|block_terms|) per block.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn score_block_bsearch_int(
    num_terms: u16,
    dim_ptr: *const u8,
    ps_ptr: *const u8,
    post_ptr: *const u8,
    query_by_dim_u16: &[(u32, u16)],
    block_mask: u64,
    acc: &mut [u32],
    touched: &mut [u64; 4],
) {
    for (q, &(dim_id, w)) in query_by_dim_u16.iter().enumerate() {
        // Bitmask skip: if this query dim has zero max in this block, skip
        if block_mask & (1u64 << q) == 0 {
            continue;
        }
        // find_dim_in_block_data always uses u32 dim_ids
        if let Some(local_term) = find_dim_in_block_data(dim_ptr, num_terms, dim_id) {
            let postings = unsafe { block_term_postings(ps_ptr, post_ptr, local_term) };
            for p in postings {
                let slot = p.local_slot as usize;
                // SAFETY: local_slot < block_size = acc.len()
                unsafe {
                    *acc.get_unchecked_mut(slot) += w as u32 * p.impact as u32;
                }
                touched[slot / 64] |= 1u64 << (slot % 64);
            }
        }
    }
}

// ============================================================================
// Superblock-at-a-time scoring
// ============================================================================

/// Score blocks within a single superblock using integer scoring.
///
/// `block_start` is the global block ID of the first block in this superblock.
/// `local_order`, `local_ubs`, `local_masks` are indexed by local offset (0..count).
///
/// Integer scoring: accumulates u16×u8 products into u32, then dequantizes to f32
/// for collector comparison.
#[allow(clippy::too_many_arguments)]
fn score_superblock_blocks(
    index: &BmpIndex,
    block_start: usize,
    count: usize,
    local_order: &[u32],
    local_ubs: &[f32],
    local_masks: &[u64],
    query_by_dim_u16: &[(u32, u16)],
    dequant: f32,
    alpha: f32,
    k: usize,
    predicate: &Option<&dyn Fn(crate::DocId) -> bool>,
    collector: &mut ScoreCollector,
    blocks_scored: &mut u32,
    acc: &mut [u32],
) {
    let block_size = index.bmp_block_size as usize;
    let num_vdocs_total = index.num_virtual_docs as usize;

    // Level 2: Pre-warm first few blocks' data (eliminates cold-start for first block).
    // block_data_starts offsets are already in cache from superblock-level prefetch.
    for &li in local_order.iter().take(4) {
        let li = li as usize;
        if li >= count {
            break;
        }
        prefetch_read(index.block_data_ptr((block_start + li) as u32));
    }

    for (order_idx, &local_idx) in local_order.iter().enumerate() {
        if local_idx as usize >= count {
            break;
        }

        let ub = local_ubs[local_idx as usize];
        // Block early termination: UB * alpha <= threshold (BMP alpha parameter)
        if collector.len() >= k && ub * alpha <= collector.threshold() {
            break;
        }

        let block_id = (block_start + local_idx as usize) as u32;

        // Lazy per-block predicate bitmask: only evaluated for blocks that survive
        // UB pruning. For each slot, check the predicate.
        // If all words are 0, skip the entire block (no scoring needed).
        // Cost: block_size predicate evaluations per block.
        let pred_mask: [u64; 4] = if let Some(pred) = predicate {
            let base = block_id as usize * block_size;
            let end = (base + block_size).min(num_vdocs_total);
            let mut mask = [0u64; 4];
            for slot in 0..(end - base) {
                let doc_id = index.doc_id_for_virtual((base + slot) as u32);
                if doc_id != u32::MAX && pred(doc_id) {
                    mask[slot / 64] |= 1u64 << (slot % 64);
                }
            }
            if mask == [0u64; 4] {
                continue; // skip scoring entirely — no matching docs
            }
            mask
        } else {
            [u64::MAX; 4]
        };

        // Level 3: Two-deep data prefetch (N+1 and N+2 block data).
        // block_data_starts offsets are warm from superblock-level prefetch,
        // so block_data_ptr() reads hit L1/L2 cache (no stall on offset lookup).
        if order_idx + 1 < local_order.len() {
            let next_local = local_order[order_idx + 1] as usize;
            if next_local < count {
                prefetch_read(index.block_data_ptr((block_start + next_local) as u32));
                if order_idx + 2 < local_order.len() {
                    let next2_local = local_order[order_idx + 2] as usize;
                    if next2_local < count {
                        prefetch_read(index.block_data_ptr((block_start + next2_local) as u32));
                    }
                }
            }
        }

        let (num_terms, dim_ptr, ps_ptr, post_ptr) = index.parse_block(block_id);
        let mask = local_masks[local_idx as usize];
        let mut touched = [0u64; 4];
        if num_terms > 0 {
            score_block_bsearch_int(
                num_terms,
                dim_ptr,
                ps_ptr,
                post_ptr,
                query_by_dim_u16,
                mask,
                acc,
                &mut touched,
            );
        }

        // Collect results + lazy zeroing.
        // Split into collect (touched & pred_mask) and reject (touched & !pred_mask).
        // When no predicate, pred_mask = [u64::MAX; 4] so reject is all zeros.
        //
        // Resolve virtual → real (doc_id, ordinal) inline and insert with real
        // doc_id. The combine_ordinal_results layer handles multi-ordinal grouping.
        let base = block_id as usize * block_size;
        let num_vdocs = index.num_virtual_docs as usize;

        for word in 0..4 {
            // Zero rejected slots (touched but filtered out by predicate)
            let mut reject = touched[word] & !pred_mask[word];
            while reject != 0 {
                let bit = reject.trailing_zeros() as usize;
                reject &= reject - 1;
                acc[word * 64 + bit] = 0;
            }

            // Collect matching slots
            let mut scan = touched[word] & pred_mask[word];
            while scan != 0 {
                let bit = scan.trailing_zeros() as usize;
                scan &= scan - 1;
                let i = word * 64 + bit;
                let score_u32 = acc[i];
                acc[i] = 0;
                if score_u32 == 0 {
                    continue;
                }

                let virtual_id = base + i;
                if virtual_id >= num_vdocs {
                    continue;
                }
                let (doc_id, ordinal) = index.virtual_to_doc(virtual_id as u32);
                if doc_id == u32::MAX {
                    continue;
                }

                let score = score_u32 as f32 * dequant;
                if collector.would_enter(score) {
                    collector.insert_with_ordinal(doc_id, score, ordinal);
                }
            }
        }

        *blocks_scored += 1;
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Sort local block indices by their UBs in descending order.
///
/// For c=64 blocks, a simple sort on non-zero UB indices is optimal.
/// Reuses `out` Vec to avoid allocation.
fn sort_local_blocks_desc(local_ubs: &[f32], out: &mut Vec<u32>) {
    out.clear();
    for (i, &ub) in local_ubs.iter().enumerate() {
        if ub > 0.0 {
            out.push(i as u32);
        }
    }
    out.sort_unstable_by(|&a, &b| {
        local_ubs[b as usize]
            .partial_cmp(&local_ubs[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Sort superblock IDs by their upper bounds in strictly descending order, into `out`.
///
/// Uses comparison sort for correctness: the `break` in the superblock loop relies
/// on strict descending order — if a SB with UB <= threshold appears, ALL subsequent
/// SBs must also have UB <= threshold. Approximate sorts (bucket sort) can violate
/// this within a bucket, causing the `break` to skip SBs that should be processed.
///
/// For ~2K superblocks, comparison sort takes ~30μs — negligible vs BMP query time.
/// Reuses `out` Vec to avoid allocation.
fn sort_sb_desc_into(block_ubs: &[f32], out: &mut Vec<u32>) {
    out.clear();
    for (i, &ub) in block_ubs.iter().enumerate() {
        if ub > 0.0 {
            out.push(i as u32);
        }
    }
    out.sort_unstable_by(|&a, &b| {
        block_ubs[b as usize]
            .partial_cmp(&block_ubs[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn collector_to_results(collector: ScoreCollector) -> Vec<ScoredDoc> {
    collector
        .into_sorted_results()
        .into_iter()
        .map(|(doc_id, score, ordinal)| ScoredDoc {
            doc_id,
            score,
            ordinal,
        })
        .collect()
}

// ============================================================================
// Compact grid helpers: operate on query-local contiguous grid buffers
// ============================================================================

/// Compute superblock UBs using integer weights for consistency with integer scoring.
///
/// Uses the same u16 query weights and dequantization factor as `score_block_bsearch_int`,
/// ensuring `sb_ub >= dequantized_score` for any document in the superblock. This avoids
/// a subtle correctness issue where f32-weighted UBs can be slightly LOWER than
/// integer-scored thresholds due to u16 quantization rounding.
///
/// `compact_sb_grid` layout: `compact_sb_grid[local_dim * nsb + sb_id]`
/// `int_weights[i] = (local_dim_index, u16_weight)` — parallel to `compact_dims`.
#[inline]
fn compute_sb_ubs_int(
    compact_sb_grid: &[u8],
    nsb: usize,
    int_weights: &[(usize, u16)],
    dequant: f32,
    out: &mut [f32],
) {
    debug_assert!(out.len() >= nsb);
    // Accumulate as u32 to match integer scoring path exactly
    for sb in 0..nsb {
        let mut acc: u32 = 0;
        for &(local_idx, w) in int_weights {
            let val = compact_sb_grid[local_idx * nsb + sb];
            acc += w as u32 * val as u32;
        }
        out[sb] = acc as f32 * dequant;
    }
}

/// Compute block UBs from compact 4-bit grid for blocks `[block_start..block_end)`.
///
/// Operates on compact (query-local) grid where dim rows are contiguous
/// in memory (L1-cache friendly).
#[inline]
fn compute_block_ubs_compact(
    compact_grid: &[u8],
    prs: usize,
    compact_dims: &[(usize, f32)],
    block_start: usize,
    block_end: usize,
    out: &mut [f32],
) {
    let count = block_end - block_start;
    debug_assert!(out.len() >= count);
    out[..count].fill(0.0);
    for &(local_idx, weight) in compact_dims {
        let row = &compact_grid[local_idx * prs..(local_idx + 1) * prs];
        accumulate_u4_weighted(row, block_start, count, weight, &mut out[..count]);
    }
}
