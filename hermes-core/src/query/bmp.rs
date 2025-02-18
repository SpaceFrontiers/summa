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
//! - **Extended prefetch**: N+1 block metadata + N+2 block term starts prefetched
//! - **Thread-local scratch**: Zero per-query allocation for large buffers
//! - **Early termination**: stop when superblock/block UB < top-k threshold

use super::scoring::{ScoreCollector, ScoredDoc};
use crate::segment::{BMP_SUPERBLOCK_SIZE, BmpIndex};

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
/// Scores are computed over compact virtual documents, then mapped back to
/// real (doc_id, ordinal) pairs via the doc_map lookup table.
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
    // Build combined info, sort by dim_id, then split into resolved + query_by_dim_u16.
    // Both arrays MUST have the same ordering so mask bit `q` corresponds to the
    // same dimension in both compute_block_masks (uses resolved) and
    // score_block_bsearch_int (uses query_by_dim_u16).
    let mut query_info: Vec<(u32, usize, f32)> = Vec::with_capacity(query_terms.len());

    for &(dim_id, weight) in query_terms {
        if let Some(idx) = index.find_dim_idx(dim_id) {
            let scaled = weight * scale;
            query_info.push((dim_id, idx, scaled));
        }
    }

    if query_info.is_empty() {
        return Ok(Vec::new());
    }

    // Sort by dim_idx for binary search within blocks (V7 stores dim_idx, not dim_id).
    // Since dim_ids is sorted and dim_idx preserves order, sorting by dim_idx gives
    // the same relative ordering as sorting by dim_id.
    query_info.sort_unstable_by_key(|&(_, idx, _)| idx);

    // Split into parallel arrays with matching order
    // resolved: (dim_idx, f32_weight) — for grid UB computation (sb_grid u8 + block grid u4×17)
    let resolved: Vec<(usize, f32)> = query_info.iter().map(|&(_, idx, w)| (idx, w)).collect();

    // Integer scoring: quantize query weights to u16 for u32 accumulator path
    // Max accumulator = 16383 × 255 × num_dims. At 1024 dims: 4.27B < u32::MAX.
    let max_scaled = query_info.iter().map(|q| q.2.abs()).fold(0.0f32, f32::max);
    let (quant_scale, dequant) = if max_scaled > 0.0 {
        (16383.0 / max_scaled, max_scaled / 16383.0)
    } else {
        (0.0, 0.0)
    };
    // V7: query_by_dim_u16 stores (dim_idx, weight) — matches Section 2 dim_indices
    let query_by_dim_u16: Vec<(u32, u16)> = query_info
        .iter()
        .map(|&(_, idx, w)| {
            (
                idx as u32,
                (w.abs() * quant_scale).round().clamp(0.0, 16383.0) as u16,
            )
        })
        .collect();

    let _start = std::time::Instant::now();

    let result = BMP_SCRATCH.with(|cell| {
        let scratch = &mut *cell.borrow_mut();

        // ── Superblock-at-a-time scoring ─────────────────────────────
        scratch.ensure_capacity_sb(
            num_superblocks_total,
            BMP_SUPERBLOCK_SIZE as usize,
            block_size,
        );

        // Phase 2: Compute SUPERBLOCK UBs
        index.compute_superblock_ubs(&resolved, &mut scratch.sb_ubs);
        bucket_sort_blocks_desc_into(
            &scratch.sb_ubs[..num_superblocks_total],
            &mut scratch.sb_order,
        );

        if scratch.sb_order.is_empty() {
            return Vec::new();
        }

        // Phase 3: Score superblocks in UB-descending order
        let mut blocks_scored = 0u32;
        let mut sbs_scored = 0u32;
        let mut collector = ScoreCollector::new(k);

        for (idx, &sb_id) in scratch.sb_order.iter().enumerate() {
            // LSP/0: hard cap on superblock visits
            if max_superblocks > 0 && idx >= max_superblocks {
                break;
            }

            let sb_ub = scratch.sb_ubs[sb_id as usize];
            // Threshold pruning still applies within the cap
            if collector.len() >= k && sb_ub * alpha <= collector.threshold() {
                break;
            }

            let block_start = sb_id as usize * BMP_SUPERBLOCK_SIZE as usize;
            let block_end = (block_start + BMP_SUPERBLOCK_SIZE as usize).min(num_blocks);
            let count = block_end - block_start;

            // Compute block UBs + masks for ONLY this superblock's blocks
            index.compute_block_ubs_range(
                &resolved,
                block_start,
                block_end,
                &mut scratch.local_block_ubs,
            );
            index.compute_block_masks_range(
                &resolved,
                block_start,
                block_end,
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
                block_size,
                alpha,
                k,
                &predicate,
                &mut collector,
                &mut blocks_scored,
                &mut scratch.acc,
            );

            sbs_scored += 1;
        }

        let elapsed_ms = _start.elapsed().as_secs_f64() * 1000.0;
        if elapsed_ms > 500.0 {
            log::warn!(
                "slow BMP: {:.1}ms, sbs={}/{}, blocks={}/{}, returned={}",
                elapsed_ms,
                sbs_scored,
                num_superblocks_total,
                blocks_scored,
                num_blocks,
                collector.len()
            );
        } else {
            log::debug!(
                "BMP execute: {:.1}ms, sbs={}/{}, blocks={}/{}, returned={}",
                elapsed_ms,
                sbs_scored,
                num_superblocks_total,
                blocks_scored,
                num_blocks,
                collector.len()
            );
        }

        collector_to_results(collector)
    });

    let mut results = result;

    // ── Phase 4: Map compact virtual_ids back to (doc_id, ordinal) ────
    for r in &mut results {
        let (doc_id, ordinal) = index.virtual_to_doc(r.doc_id);
        r.doc_id = doc_id;
        r.ordinal = ordinal;
    }

    Ok(results)
}

// ============================================================================
// Integer scoring: u32 accumulators with u16 quantized query weights
// ============================================================================

/// Score a block using integer arithmetic (u32 accumulators, u16 weights).
///
/// Uses **bitmask skip**: checks `block_mask & (1 << q) != 0` before binary search.
/// Accumulates `w_u16 * impact_u8` into u32 — eliminates u8→f32 conversion per posting.
///
/// V7: searches by dim_idx (position in dim_ids array), not dim_id.
///
/// Tracks touched slots via a u64 bitmask (works for block_size ≤ 64).
/// Caller uses the bitmask for lazy accumulator zeroing.
///
/// Complexity: O(|present_query_dims| × log|block_terms|) per block.
#[inline(always)]
fn score_block_bsearch_int(
    index: &BmpIndex,
    term_start: u32,
    term_end: u32,
    query_by_dim_u16: &[(u32, u16)],
    block_mask: u64,
    acc: &mut [u32],
    touched: &mut u64,
) {
    for (q, &(dim_idx, w)) in query_by_dim_u16.iter().enumerate() {
        // Bitmask skip: if this query dim has zero max in this block, skip
        if block_mask & (1u64 << q) == 0 {
            continue;
        }
        if let Some(ti) = index.find_dim_in_block(term_start, term_end, dim_idx) {
            for p in index.term_postings(ti) {
                let slot = p.local_slot as usize;
                // SAFETY: local_slot < block_size = acc.len()
                unsafe {
                    *acc.get_unchecked_mut(slot) += w as u32 * p.impact as u32;
                }
                *touched |= 1u64 << slot;
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
    _block_size: usize,
    alpha: f32,
    k: usize,
    predicate: &Option<&dyn Fn(crate::DocId) -> bool>,
    collector: &mut ScoreCollector,
    blocks_scored: &mut u32,
    acc: &mut [u32],
) {
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

        // Extended prefetch: next block in local order
        if order_idx + 1 < local_order.len() {
            let next_local = local_order[order_idx + 1] as usize;
            if next_local < count {
                let next_block = (block_start + next_local) as u32;
                let (ts, te) = index.block_term_range(next_block);
                if ts < te {
                    prefetch_read(index.term_dim_ids_ptr(ts));
                    prefetch_read(index.term_posting_starts_ptr(ts));
                    if let Some(ptr) = index.first_posting_ptr(next_block) {
                        prefetch_read(ptr);
                    }
                }
                if order_idx + 2 < local_order.len() {
                    let next2_local = local_order[order_idx + 2] as usize;
                    if next2_local < count {
                        prefetch_read(
                            index.block_term_starts_ptr((block_start + next2_local) as u32),
                        );
                    }
                }
            }
        }

        let (term_start, term_end) = index.block_term_range(block_id);
        let mask = local_masks[local_idx as usize];
        let mut touched: u64 = 0;
        score_block_bsearch_int(
            index,
            term_start,
            term_end,
            query_by_dim_u16,
            mask,
            acc,
            &mut touched,
        );

        // Collect results: dequantize u32 → f32, only visit touched slots
        let base = block_id * index.bmp_block_size;
        let num_vdocs = index.num_virtual_docs;
        let mut scan = touched;
        while scan != 0 {
            let i = scan.trailing_zeros() as usize;
            scan &= scan - 1; // clear lowest set bit
            let score_u32 = acc[i];
            if score_u32 > 0 {
                let virtual_id = base + i as u32;
                // Guard against phantom slots in the last block
                if virtual_id < num_vdocs {
                    let score = score_u32 as f32 * dequant;
                    if collector.would_enter(score) {
                        if let Some(pred) = predicate {
                            let doc_id = index.doc_id_for_virtual(virtual_id);
                            if !pred(doc_id) {
                                continue;
                            }
                        }
                        collector.insert_with_ordinal(virtual_id, score, 0);
                    }
                }
            }
        }

        // Lazy zeroing: only clear touched slots instead of acc[..block_size].fill(0)
        let mut clear = touched;
        while clear != 0 {
            let bit = clear.trailing_zeros() as usize;
            acc[bit] = 0;
            clear &= clear - 1;
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

/// Bucket sort block IDs by their upper bounds in descending order, into `out`.
///
/// Uses 256 buckets. O(n) time vs O(n log n) for comparison sort.
/// Reuses `out` Vec to avoid allocation.
fn bucket_sort_blocks_desc_into(block_ubs: &[f32], out: &mut Vec<u32>) {
    out.clear();

    // Find max UB for normalization
    let max_ub = block_ubs.iter().cloned().fold(0.0f32, f32::max);
    if max_ub <= 0.0 {
        return;
    }

    const NUM_BUCKETS: usize = 256;
    let inv_max = (NUM_BUCKETS - 1) as f32 / max_ub;

    // Count pass: how many blocks per bucket
    let mut counts = [0u32; NUM_BUCKETS];
    for &ub in block_ubs {
        if ub > 0.0 {
            let bucket = (ub * inv_max) as usize;
            counts[bucket.min(NUM_BUCKETS - 1)] += 1;
        }
    }

    // Prefix sum (from high to low for descending order)
    let total: u32 = counts.iter().sum();
    let mut offsets = [0u32; NUM_BUCKETS];
    let mut running = 0u32;
    for i in (0..NUM_BUCKETS).rev() {
        offsets[i] = running;
        running += counts[i];
    }

    // Scatter pass
    out.resize(total as usize, 0);
    for (b, &ub) in block_ubs.iter().enumerate() {
        if ub > 0.0 {
            let bucket = (ub * inv_max) as usize;
            let bucket = bucket.min(NUM_BUCKETS - 1);
            let pos = offsets[bucket] as usize;
            out[pos] = b as u32;
            offsets[bucket] += 1;
        }
    }
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
