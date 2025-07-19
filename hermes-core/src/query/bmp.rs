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
//! - **Integer-consistent UBs**: bounds and document scores share exact accumulator units
//! - **Pre-scaled weights**: `weight * scale` computed once, not per-block
//! - **Bitmask skip**: Register-level mask check replaces grid DRAM lookups
//! - **Bucket sort**: O(n) superblock ordering by UB descending
//! - **Binary search scoring**: O(|query| × log|block_terms|) per block
//! - **Multi-level prefetch**: SB offset warming → pre-loop burst → N+1/N+2 data pipeline
//! - **Thread-local scratch**: Zero per-query allocation for large buffers
//! - **Early termination**: stop when superblock/block UB < top-k threshold

use super::scoring::{ScoreCollector, ScoredDoc, SharedThreshold};
use crate::segment::{
    BMP_SUPERBLOCK_SIZE, BmpIndex, accumulate_grid_u32, block_term_postings, compute_block_masks,
    find_dim_in_block_data,
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
    sb_priorities: Vec<f32>,
    sb_suffix_max: Vec<f32>,
    // Block-level (reused per superblock, sized to BMP_SUPERBLOCK_SIZE)
    local_block_ubs: Vec<f32>,
    local_block_ub_units: Vec<u32>,
    local_block_masks: Vec<u64>,
    local_block_order: Vec<u32>,
    // Two-phase block scoring: phase1 block UBs (sized to BMP_SUPERBLOCK_SIZE)
    phase1_local_block_ubs: Vec<f32>,
    phase1_local_block_ub_units: Vec<u32>,
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
        if self.sb_priorities.len() < num_superblocks {
            self.sb_priorities.resize(num_superblocks, 0.0);
        }
        // suffix_max needs num_superblocks + 1 for sentinel
        if self.sb_suffix_max.len() < num_superblocks + 1 {
            self.sb_suffix_max.resize(num_superblocks + 1, 0.0);
        }
        if self.local_block_ubs.len() < sb_size {
            self.local_block_ubs.resize(sb_size, 0.0);
        }
        if self.local_block_ub_units.len() < sb_size {
            self.local_block_ub_units.resize(sb_size, 0);
        }
        if self.local_block_masks.len() < sb_size {
            self.local_block_masks.resize(sb_size, 0u64);
        }
        if self.local_block_order.len() < sb_size {
            self.local_block_order.resize(sb_size, 0);
        }
        if self.phase1_local_block_ubs.len() < sb_size {
            self.phase1_local_block_ubs.resize(sb_size, 0.0);
        }
        if self.phase1_local_block_ub_units.len() < sb_size {
            self.phase1_local_block_ub_units.resize(sb_size, 0);
        }
        // On-disk local slots are u8. Keep all 256 addresses valid even if a
        // corrupt block contains a slot beyond its declared logical size;
        // the scorer still rejects such a posting explicitly.
        let safe_acc_size = block_size.max(256);
        if self.acc.len() < safe_acc_size {
            self.acc.resize(safe_acc_size, 0);
        }
    }
}

thread_local! {
    static BMP_SCRATCH: std::cell::RefCell<BmpScratch> =
        std::cell::RefCell::new(BmpScratch::default());
}

/// A correctness-approved cross-segment score floor for BMP. The planner only
/// supplies this when raw ordinal scores are also final document scores
/// (physically single-valued data or the `Max` combiner).
#[derive(Clone, Copy, Default)]
pub(crate) struct BmpThreshold<'a> {
    pub initial: f32,
    pub shared: Option<&'a SharedThreshold>,
    /// A raw BMP heap can publish its k-th score only when its entries are
    /// guaranteed to represent distinct documents.
    pub publish: bool,
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
/// - **0.8**: prune when `UB * 0.8 < threshold` → ~20% more aggressive
/// - **0.6**: prune when `UB * 0.6 < threshold` → ~40% more aggressive
///
/// `max_superblocks` (LSP/0 gamma cap): hard limit on superblock visits.
/// - **0**: unlimited (default, existing behavior)
/// - **>0**: stop after visiting this many superblocks
///
/// Based on Mallia et al. (SIGIR 2024) and Carlson et al. (arXiv 2602.02883).
#[allow(dead_code)]
pub fn execute_bmp(
    index: &BmpIndex,
    index_label: &str,
    field_label: &str,
    query_terms: &[(u32, f32)],
    k: usize,
    heap_factor: f32,
    max_superblocks: usize,
) -> crate::Result<Vec<ScoredDoc>> {
    execute_bmp_inner(
        index,
        index_label,
        field_label,
        query_terms,
        k,
        heap_factor,
        max_superblocks,
        None,
        BmpThreshold::default(),
    )
}

/// BMP execution with a planner-validated live cross-segment threshold.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_bmp_with_threshold(
    index: &BmpIndex,
    index_label: &str,
    field_label: &str,
    query_terms: &[(u32, f32)],
    k: usize,
    heap_factor: f32,
    max_superblocks: usize,
    threshold: BmpThreshold<'_>,
) -> crate::Result<Vec<ScoredDoc>> {
    execute_bmp_inner(
        index,
        index_label,
        field_label,
        query_terms,
        k,
        heap_factor,
        max_superblocks,
        None,
        threshold,
    )
}

/// Execute a BMP query with a document predicate filter.
///
/// Same as [`execute_bmp`] but only collects documents that pass the predicate.
/// The predicate is checked during scoring (not post-filter), so the collector
/// only contains valid documents and the threshold evolves correctly.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn execute_bmp_filtered(
    index: &BmpIndex,
    index_label: &str,
    field_label: &str,
    query_terms: &[(u32, f32)],
    k: usize,
    heap_factor: f32,
    max_superblocks: usize,
    predicate: &dyn Fn(crate::DocId) -> bool,
) -> crate::Result<Vec<ScoredDoc>> {
    execute_bmp_inner(
        index,
        index_label,
        field_label,
        query_terms,
        k,
        heap_factor,
        max_superblocks,
        Some(predicate),
        BmpThreshold::default(),
    )
}

/// Filtered BMP execution with a planner-validated live threshold.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_bmp_filtered_with_threshold(
    index: &BmpIndex,
    index_label: &str,
    field_label: &str,
    query_terms: &[(u32, f32)],
    k: usize,
    heap_factor: f32,
    max_superblocks: usize,
    predicate: &dyn Fn(crate::DocId) -> bool,
    threshold: BmpThreshold<'_>,
) -> crate::Result<Vec<ScoredDoc>> {
    execute_bmp_inner(
        index,
        index_label,
        field_label,
        query_terms,
        k,
        heap_factor,
        max_superblocks,
        Some(predicate),
        threshold,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_bmp_inner(
    index: &BmpIndex,
    index_label: &str,
    field_label: &str,
    query_terms: &[(u32, f32)],
    k: usize,
    heap_factor: f32,
    max_superblocks: usize,
    predicate: Option<&dyn Fn(crate::DocId) -> bool>,
    threshold_source: BmpThreshold<'_>,
) -> crate::Result<Vec<ScoredDoc>> {
    if query_terms.is_empty() || index.num_blocks == 0 || k == 0 {
        return Ok(Vec::new());
    }

    // Alpha parameter for approximate retrieval:
    // UB * alpha < threshold → prune when UB < threshold / alpha
    // alpha < 1.0 means more aggressive pruning (approximate but faster)
    let alpha = heap_factor.clamp(0.01, 1.0);
    let scale = index.max_weight_scale / 255.0;
    let num_blocks = index.num_blocks as usize;
    let block_size = index.bmp_block_size as usize;
    let num_superblocks_total = index.num_superblocks as usize;

    // ── Phase 1: Resolve query dimensions and quantize weights ────────
    // dim_id is used directly as grid row index (no dim_idx indirection).
    // Build combined info, sort by dim_id, then split into resolved + query_by_dim_u16.
    // Both arrays MUST have the same ordering so mask bit `q` corresponds to the
    // same dimension in both compute_block_masks (uses resolved) and
    // score_block_bsearch_int (uses query_by_dim_u16).
    let dims = index.dims();
    let mut query_info: Vec<(u32, usize, f32)> = Vec::with_capacity(query_terms.len());

    for &(dim_id, weight) in query_terms {
        // dim_id IS the grid row index — just check bounds
        if dim_id < dims && weight.is_finite() && weight != 0.0 {
            // BMP stores absolute document impacts. Upper bounds and integer
            // scoring must use the same absolute query weight; keeping the
            // sign only in the grid path could under-estimate a block and
            // prune a result that the scorer later treated as positive.
            query_info.push((dim_id, dim_id as usize, weight.abs()));
        }
    }

    if query_info.is_empty() {
        return Ok(Vec::new());
    }

    // Block masks and the two-phase scorer use one u64 bit per query term.
    // SparseVectorQuery enforces this already; reject oversized direct calls
    // rather than silently producing an incomplete mask.
    if query_info.len() > crate::query::MAX_QUERY_TERMS {
        return Err(crate::Error::Query(format!(
            "BMP query has {} resolved dimensions; maximum is {}",
            query_info.len(),
            crate::query::MAX_QUERY_TERMS
        )));
    }

    // Sort by dim_id for binary search within blocks (blocks store dim_id directly).
    query_info.sort_unstable_by_key(|&(dim_id, _, _)| dim_id);

    // Integer scoring: adapt the quantizer to query width so the documented
    // u32 accumulator bound remains true for queries wider than 1024 dims.
    let max_query_weight = query_info.iter().map(|q| q.2).fold(0.0f32, f32::max);
    let accumulator_denominator = 255u64.saturating_mul(query_info.len() as u64);
    let max_quantized_weight = ((u32::MAX as u64) / accumulator_denominator).min(16_383);
    if max_quantized_weight == 0 {
        return Err(crate::Error::Query(format!(
            "BMP query has too many resolved dimensions ({})",
            query_info.len()
        )));
    }
    let (quant_scale, dequant) = if max_query_weight > 0.0 {
        (
            max_quantized_weight as f32 / max_query_weight,
            (max_query_weight / max_quantized_weight as f32) * scale,
        )
    } else {
        (0.0, 0.0)
    };
    // query_by_dim_u16 stores (dim_id, weight) — matches per-block dim_ids
    let query_by_dim_u16: Vec<(u32, u16)> = query_info
        .iter()
        .map(|&(dim_id, _, w)| {
            (
                dim_id,
                (w.abs() * quant_scale)
                    .round()
                    .clamp(0.0, max_quantized_weight as f32) as u16,
            )
        })
        .collect();
    if !dequant.is_finite() {
        return Err(crate::Error::Query(
            "BMP query score scale exceeds the finite f32 range".into(),
        ));
    }

    // Block and superblock bounds use the exact integer-scoring weight
    // (quantized query × common dequantizer), not the original f32 weight.
    // Otherwise rounding a query weight upward could make a scored document
    // slightly exceed its block bound and be pruned incorrectly.
    let resolved: Vec<(usize, f32)> = query_info
        .iter()
        .zip(&query_by_dim_u16)
        .map(|(&(.., grid_index, _), &(_, weight))| (grid_index, weight as f32 * dequant))
        .collect();

    // ── Two-phase lazy block scoring setup ────────────────────────────
    // For queries with >5 dims, score only the top-3 heaviest dims first (phase1).
    // If max_partial_score + remaining_block_ub < threshold, skip phase2.
    // For ≤5 dims: full scoring directly (zero overhead).
    const PHASE1_DIMS: usize = 3;
    const MIN_DIMS_FOR_TWO_PHASE: usize = 6;
    let two_phase_active = (MIN_DIMS_FOR_TWO_PHASE..=64).contains(&query_by_dim_u16.len());
    // phase1_mask: bitmask of which query dim indices are in phase1
    let phase1_mask: u64 = if two_phase_active {
        let mut weight_indices: Vec<(u16, usize)> = query_by_dim_u16
            .iter()
            .enumerate()
            .map(|(i, &(_, w))| (w, i))
            .collect();
        weight_indices.sort_unstable_by_key(|b| std::cmp::Reverse(b.0));
        weight_indices[..PHASE1_DIMS]
            .iter()
            .fold(0u64, |m, &(_, i)| m | (1u64 << i))
    } else {
        u64::MAX
    };
    // phase1 grid dims for UB computation (indices into grid_dims/resolved)
    let phase1_grid_indices: Vec<usize> = if two_phase_active {
        (0..query_by_dim_u16.len())
            .filter(|&i| phase1_mask & (1u64 << i) != 0)
            .collect()
    } else {
        Vec::new()
    };

    // The planner has already applied the configured ordinal over-fetch factor
    // to `k`.  Do not derive another factor from num_virtual_docs: that ratio
    // includes block-alignment padding and used to multiply the heap depth a
    // second time (usually turning the default 2x into 4x).
    let collector_k = k;

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
        let compact_sb_size = resolved.len().saturating_mul(num_superblocks_total);
        let compact_grid_size = resolved.len().saturating_mul(prs);
        let use_compact = compact_sb_size.saturating_add(compact_grid_size) <= COMPACT_GRID_MAX;

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

        let phase1_int_weights: Vec<(usize, u16)> = if two_phase_active {
            phase1_grid_indices
                .iter()
                .map(|&index| sb_int_weights[index])
                .collect()
        } else {
            Vec::new()
        };

        // Phase 2: Compute SUPERBLOCK UBs using integer weights (matching scoring path)
        compute_sb_ubs_int(
            sb_grid_slice,
            num_superblocks_total,
            &sb_int_weights,
            dequant,
            &mut scratch.sb_ubs,
        );

        // Coverage-biased SB ordering: boost SBs with higher query-dim coverage.
        // Count how many query dims are active in each SB (non-zero sb_grid value).
        let nqd = sb_int_weights.len();
        for sb in 0..num_superblocks_total {
            let base_ub = scratch.sb_ubs[sb];
            if base_ub == 0.0 {
                scratch.sb_priorities[sb] = 0.0;
                continue;
            }
            let mut coverage = 0u32;
            for &(local_idx, _) in &sb_int_weights {
                if sb_grid_slice[local_idx * num_superblocks_total + sb] > 0 {
                    coverage += 1;
                }
            }
            // Boost: high-coverage SBs get +5% priority boost (breaks ties, doesn't
            // significantly reorder SBs with very different UBs).
            let cf = coverage as f32 / nqd as f32;
            scratch.sb_priorities[sb] = base_ub * (1.0 + cf * 0.05);
        }

        sort_sb_desc_into(
            &scratch.sb_priorities[..num_superblocks_total],
            &mut scratch.sb_order,
        );

        if scratch.sb_order.is_empty() {
            return Vec::new();
        }

        // Pre-compute suffix-max of ACTUAL UBs for safe early termination.
        // Because coverage-biased ordering is non-monotonic in UB, we can't `break`
        // on a single low-UB SB. Instead we `break` when the max UB of ALL remaining
        // SBs < threshold — guaranteed correct while retaining canonical ties.
        compute_suffix_max_ubs(
            &scratch.sb_ubs,
            &scratch.sb_order,
            &mut scratch.sb_suffix_max,
        );

        // Phase 3: Score superblocks in priority-descending order
        let mut blocks_scored = 0u32;
        let mut docmap_lookups = 0u32;
        let mut sbs_scored = 0u32;
        let mut collector = ScoreCollector::new(collector_k);
        let initial_threshold = threshold_source
            .shared
            .map(SharedThreshold::get)
            .unwrap_or(0.0)
            .max(threshold_source.initial);
        collector.seed_threshold(initial_threshold);

        for (idx, &sb_id) in scratch.sb_order.iter().enumerate() {
            if let Some(shared) = threshold_source.shared {
                collector.seed_threshold(shared.get());
            }
            // LSP/0: hard cap on superblock visits
            if max_superblocks > 0 && idx >= max_superblocks {
                break;
            }

            // Safe early termination: max UB of ALL remaining SBs < threshold
            if collector.len() >= collector_k
                && scratch.sb_suffix_max[idx] * alpha < collector.threshold()
            {
                break;
            }

            // Individual SB skip (continue, not break — ordering is non-monotonic in UB)
            let sb_ub = scratch.sb_ubs[sb_id as usize];
            if collector.len() >= collector_k && sb_ub * alpha < collector.threshold() {
                continue;
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
            compute_block_ubs_int(
                grid_slice,
                index.grid_bits(),
                prs,
                &sb_int_weights,
                block_start,
                block_end,
                dequant,
                &mut scratch.local_block_ub_units,
                &mut scratch.local_block_ubs,
            );
            compute_block_masks(
                grid_slice,
                index.grid_bits(),
                prs,
                &grid_dims,
                block_start,
                block_end - block_start,
                &mut scratch.local_block_masks,
            );

            // Two-phase: compute phase1 block UBs (only top-3 dims)
            if two_phase_active {
                compute_block_ubs_int(
                    grid_slice,
                    index.grid_bits(),
                    prs,
                    &phase1_int_weights,
                    block_start,
                    block_end,
                    dequant,
                    &mut scratch.phase1_local_block_ub_units,
                    &mut scratch.phase1_local_block_ubs,
                );
            }

            sort_local_blocks_desc(
                &scratch.local_block_ubs[..count],
                &mut scratch.local_block_order,
            );

            // Level 3: page-level prefetch of the surviving blocks' data.
            // Mirrors the scoring loop's skip conditions (UB-descending order,
            // break on ub*alpha < threshold, skip zero-mask blocks) to find
            // the byte span that will actually be read, then issues a single
            // MADV_WILLNEED so cold pages are clustered into sequential reads
            // instead of one major fault per block (memory-bound hosts).
            #[cfg(feature = "native")]
            {
                let thr = collector.threshold();
                let heap_full = collector.len() >= collector_k;
                let mut lo = u64::MAX;
                let mut hi = 0u64;
                for &li in scratch.local_block_order.iter().take(count) {
                    let li = li as usize;
                    if li >= count {
                        break;
                    }
                    if heap_full && scratch.local_block_ubs[li] * alpha < thr {
                        break;
                    }
                    if scratch.local_block_masks[li] == 0 {
                        continue;
                    }
                    let (s, e) = index.block_data_range((block_start + li) as u32);
                    lo = lo.min(s);
                    hi = hi.max(e);
                }
                if lo < hi {
                    index.prefetch_block_data(lo, hi);
                }
            }

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
                &mut docmap_lookups,
                &mut scratch.acc,
                phase1_mask,
                if two_phase_active {
                    Some(&scratch.phase1_local_block_ubs)
                } else {
                    None
                },
            );

            if threshold_source.publish
                && collector.real_len() >= collector_k
                && let Some(shared) = threshold_source.shared
            {
                shared.raise(collector.threshold());
            }

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
        let returned = collector.real_len();
        crate::observe::bmp_query(
            index_label,
            field_label,
            t_start.elapsed().as_secs_f64(),
            sbs_scored as usize,
            num_superblocks_total,
            blocks_scored as usize,
            num_blocks,
            docmap_lookups as usize,
        );
        if elapsed_ms > 500.0 {
            log::warn!(
                "slow BMP: {:.1}ms, sbs={}/{}, blocks={}/{}, k={}, returned={}, seed={:.4}, threshold={:.4}, alpha={:.2}",
                elapsed_ms,
                sbs_scored,
                num_superblocks_total,
                blocks_scored,
                num_blocks,
                collector_k,
                returned,
                initial_threshold,
                threshold,
                alpha,
            );
        } else {
            log::debug!(
                "BMP execute: {:.1}ms, sbs={}/{}, blocks={}/{}, k={}, returned={}, seed={:.4}, threshold={:.4}, alpha={:.2}",
                elapsed_ms,
                sbs_scored,
                num_superblocks_total,
                blocks_scored,
                num_blocks,
                collector_k,
                returned,
                initial_threshold,
                threshold,
                alpha,
            );
        }

        collector_to_results(collector)
    });

    Ok(result)
}

// ============================================================================
// Integer scoring: u32 accumulators with u16 quantized query weights
// ============================================================================

/// Find the maximum u32 value across touched accumulator slots.
///
/// Uses the touched bitmask for O(|touched_slots|) — typically 5-20 slots per block.
#[inline(always)]
fn max_touched_acc(acc: &[u32], touched: &[u64; 4]) -> u32 {
    let mut max_val = 0u32;
    for word in 0..4 {
        let mut bits = touched[word];
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            max_val = max_val.max(acc[word * 64 + bit]);
        }
    }
    max_val
}

/// Zero all touched accumulator slots.
#[inline(always)]
fn zero_touched_acc(acc: &mut [u32], touched: &[u64; 4]) {
    for word in 0..4 {
        let mut bits = touched[word];
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            acc[word * 64 + bit] = 0;
        }
    }
}

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
    num_terms: u32,
    dim_ptr: *const u8,
    ps_ptr: *const u8,
    post_ptr: *const u8,
    query_by_dim_u16: &[(u32, u16)],
    block_mask: u64,
    acc: &mut [u32],
    touched: &mut [u64; 4],
    block_size: usize,
    total_block_postings: u32,
) {
    for (q, &(dim_id, w)) in query_by_dim_u16.iter().enumerate() {
        // Bitmask skip: if this query dim has zero max in this block, skip
        if q < 64 && block_mask & (1u64 << q) == 0 {
            continue;
        }
        // find_dim_in_block_data always uses u32 dim_ids
        if let Some(local_term) = find_dim_in_block_data(dim_ptr, num_terms, dim_id) {
            let postings =
                unsafe { block_term_postings(ps_ptr, post_ptr, local_term, total_block_postings) };
            for p in postings {
                let slot = p.local_slot as usize;
                if slot >= block_size {
                    continue;
                }
                acc[slot] = acc[slot].saturating_add(w as u32 * p.impact as u32);
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
/// **Two-phase lazy scoring**: When `phase1_mask != u64::MAX` and `phase1_local_ubs`
/// is `Some`, scores only phase1 dims first. If the best possible score from phase1 +
/// remaining block UB < threshold, skips phase2 dims entirely (~40-60% of scoring work).
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
    docmap_lookups: &mut u32,
    acc: &mut [u32],
    phase1_mask: u64,
    phase1_local_ubs: Option<&[f32]>,
) {
    let block_size = index.bmp_block_size as usize;
    let two_phase = phase1_mask != u64::MAX && phase1_local_ubs.is_some();

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
        // Block early termination: UB * alpha < threshold (BMP alpha parameter)
        if collector.len() >= k && ub * alpha < collector.threshold() {
            break;
        }

        let block_id = (block_start + local_idx as usize) as u32;

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

        let (num_terms, dim_ptr, ps_ptr, post_ptr, total_block_postings) =
            index.parse_block(block_id);
        let mask = local_masks[local_idx as usize];
        let mut touched = [0u64; 4];
        if num_terms > 0 {
            if two_phase && collector.len() >= k {
                // Phase 1: Score only the heaviest dims
                score_block_bsearch_int(
                    num_terms,
                    dim_ptr,
                    ps_ptr,
                    post_ptr,
                    query_by_dim_u16,
                    mask & phase1_mask,
                    acc,
                    &mut touched,
                    block_size,
                    total_block_postings,
                );

                // Check if phase2 can be skipped:
                // max_partial_score + remaining_ub < threshold?
                // remaining_ub = full_block_ub - phase1_block_ub
                let max_partial = max_touched_acc(acc, &touched) as f32 * dequant;
                let phase1_ub = phase1_local_ubs.unwrap()[local_idx as usize];
                let remaining_ub = (ub - phase1_ub).max(0.0);
                if (max_partial + remaining_ub) * alpha < collector.threshold() {
                    // Skip phase2 — zero touched slots and continue
                    zero_touched_acc(acc, &touched);
                    *blocks_scored += 1;
                    continue;
                }

                // Phase 2: Score remaining dims
                score_block_bsearch_int(
                    num_terms,
                    dim_ptr,
                    ps_ptr,
                    post_ptr,
                    query_by_dim_u16,
                    mask & !phase1_mask,
                    acc,
                    &mut touched,
                    block_size,
                    total_block_postings,
                );
            } else {
                // Single-phase: score all dims at once
                score_block_bsearch_int(
                    num_terms,
                    dim_ptr,
                    ps_ptr,
                    post_ptr,
                    query_by_dim_u16,
                    mask,
                    acc,
                    &mut touched,
                    block_size,
                    total_block_postings,
                );
            }
        }

        // Collect results + lazy zeroing. Apply an ordinary predicate only to
        // slots that scoring actually touched. Sparse queries commonly touch a
        // small fraction of a block, so evaluating every slot before scoring
        // was needlessly expensive for broad/non-bitset predicates.
        //
        // Resolve virtual → real (doc_id, ordinal) inline and insert with real
        // doc_id. The combine_ordinal_results layer handles multi-ordinal grouping.
        let base = block_id as usize * block_size;
        let num_vdocs = index.num_virtual_docs as usize;

        for (word, &touched_word) in touched.iter().enumerate() {
            let mut scan = touched_word;
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
                // Doc-map indirection: BMP reorder permutes only BMP-internal
                // record order, so every candidate pays a scattered lookup
                // into the doc-id map here. Counted per query (metered as
                // hermes_bmp_docmap_lookups_*).
                *docmap_lookups += 1;
                let (doc_id, ordinal) = index.virtual_to_doc(virtual_id as u32);
                if doc_id == u32::MAX {
                    continue;
                }
                if let Some(pred) = predicate
                    && !pred(doc_id)
                {
                    continue;
                }

                let score = score_u32 as f32 * dequant;
                if collector.would_enter_candidate(doc_id, score, ordinal) {
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

/// Sort superblock IDs by values (priorities or UBs) in descending order, into `out`.
///
/// For ~2K superblocks, comparison sort takes ~30μs — negligible vs BMP query time.
/// Reuses `out` Vec to avoid allocation.
fn sort_sb_desc_into(values: &[f32], out: &mut Vec<u32>) {
    out.clear();
    for (i, &v) in values.iter().enumerate() {
        if v > 0.0 {
            out.push(i as u32);
        }
    }
    out.sort_unstable_by(|&a, &b| {
        values[b as usize]
            .partial_cmp(&values[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Pre-compute suffix-max of actual UBs over the sorted SB order.
///
/// `suffix_max[i]` = max UB among all SBs at positions `i..order.len()`.
/// This enables safe early termination with non-monotonic ordering:
/// if `suffix_max[i] * alpha < threshold`, ALL remaining SBs can be skipped.
///
/// O(num_superblocks) pre-computation, then O(1) check per SB in the loop.
fn compute_suffix_max_ubs(sb_ubs: &[f32], order: &[u32], out: &mut [f32]) {
    let n = order.len();
    // Sentinel: suffix_max[n] = 0.0 (no remaining SBs)
    out[n] = 0.0;
    for i in (0..n).rev() {
        let ub = sb_ubs[order[i] as usize];
        out[i] = ub.max(out[i + 1]);
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

/// Compute block UBs in the same integer units as document scoring.
///
/// Grid cells are ceiling-quantized upper bounds. Accumulating `u16 × u8` in
/// `u32`, then applying the common dequantizer once, preserves the monotonic
/// relationship `block_ub >= document_score` under f32 conversion. Summing
/// already-dequantized f32 terms can round downward (measurably even at 64
/// dimensions) and made exact-mode pruning unsafe near the heap threshold.
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_block_ubs_int(
    compact_grid: &[u8],
    grid_bits: u8,
    prs: usize,
    int_weights: &[(usize, u16)],
    block_start: usize,
    block_end: usize,
    dequant: f32,
    units: &mut [u32],
    out: &mut [f32],
) {
    let count = block_end - block_start;
    debug_assert!(units.len() >= count);
    debug_assert!(out.len() >= count);
    units[..count].fill(0);
    for &(local_idx, weight) in int_weights {
        let row = &compact_grid[local_idx * prs..(local_idx + 1) * prs];
        accumulate_grid_u32(
            row,
            grid_bits,
            block_start,
            count,
            u32::from(weight),
            &mut units[..count],
        );
    }
    for (bound, &integer_units) in out[..count].iter_mut().zip(&units[..count]) {
        *bound = integer_units as f32 * dequant;
    }
}

#[cfg(test)]
mod tests {
    use super::compute_block_ubs_int;

    #[test]
    fn integer_block_bound_cannot_round_below_integer_score() {
        // Repeated f32 accumulation underestimates this 64-dimension sum on
        // common targets. The block bound and document score now convert the
        // same integer units exactly once.
        let dimensions = 64usize;
        let grid = vec![0x0f; dimensions]; // one 4-bit cell (= 255) per row
        let weights: Vec<(usize, u16)> =
            (0..dimensions).map(|dimension| (dimension, 160)).collect();
        let dequant = 0.123_456_7f32;
        let mut units = [0u32; 1];
        let mut bounds = [0.0f32; 1];

        compute_block_ubs_int(
            &grid,
            4,
            1,
            &weights,
            0,
            1,
            dequant,
            &mut units,
            &mut bounds,
        );

        let score_units = 160u32 * 255 * dimensions as u32;
        let score = score_units as f32 * dequant;
        assert!(bounds[0] >= score);
        assert_eq!(units[0], score_units);
    }

    #[test]
    fn packed_integer_bound_kernel_matches_every_4bit_cell() {
        let blocks = 64usize;
        let grid: Vec<u8> = (0..blocks / 2)
            .map(|pair| {
                let low = (pair * 2 % 16) as u8;
                let high = ((pair * 2 + 1) % 16) as u8;
                low | (high << 4)
            })
            .collect();
        let mut units = [0u32; 64];
        let mut bounds = [0.0f32; 64];

        compute_block_ubs_int(
            &grid,
            4,
            grid.len(),
            &[(0, 123)],
            0,
            blocks,
            1.0,
            &mut units,
            &mut bounds,
        );

        for block in 0..blocks {
            assert_eq!(units[block], (block % 16) as u32 * 17 * 123);
            assert_eq!(bounds[block], units[block] as f32);
        }
    }
}
