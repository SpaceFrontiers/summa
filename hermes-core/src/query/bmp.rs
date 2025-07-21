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
//! 1. **LSP/0 top-γ**: Visit only the highest-SBMax superblocks
//! 2. **Superblock UBs** (~1.2K entries at 1M×5): cheap to compute, prune 25-75%
//! 3. **Block UBs** (only for surviving superblocks): L1-cache friendly per-SB
//!
//! ## Performance
//!
//! All block data is pre-decoded at index load time. Query execution touches
//! only flat contiguous arrays — no file I/O, no parsing, no heap allocation
//! in the hot path.
//!
//! - **LSP/0 top-γ**: strict SBMax order with safe superblock-level pruning
//! - **Integer scoring**: u32 accumulators with u16 quantized query weights (~20% faster)
//! - **Superblock pruning**: Skip entire groups of blocks via coarse UBs
//! - **L1 cache locality**: SaaT loop keeps eight blocks' grid data in registers/L1
//! - **Integer-consistent UBs**: bounds and document scores share exact accumulator units
//! - **Pre-scaled weights**: `weight * scale` computed once, not per-block
//! - **Bitmask skip**: Register-level mask check replaces grid DRAM lookups
//! - **Bucket sort**: O(n) superblock ordering by UB descending
//! - **Binary search scoring**: O(|query| × log|block_terms|) per block
//! - **Multi-level prefetch**: SB offset warming → pre-loop burst → N+1/N+2 data pipeline
//! - **Thread-local scratch**: Zero per-query allocation for large buffers
//! - **Early termination**: stop when superblock/block UB < top-k threshold

use super::scoring::{ScoreCollector, ScoredDoc, SharedThreshold};
use crate::segment::bmp_grid::{
    CompressedGrid, GRID_GROUP_CELLS, LSP_SUPERBLOCK_GRID_BITS, accumulate_packed_u4,
    accumulate_u8, block_grid_scale,
};
use crate::segment::{BMP_SUPERBLOCK_SIZE, BmpIndex, block_term_postings, find_dim_in_block_data};

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
/// - **Local block-level**: sized to BMP_SUPERBLOCK_SIZE (8), for per-SB block computation
/// - **Accumulator**: u32 sized to bmp_block_size, reused per block (integer scoring)
#[derive(Default)]
struct BmpScratch {
    // Superblock-level (reused across queries, sized to num_superblocks)
    sb_ubs: Vec<f32>,
    sb_ub_units: Vec<u32>,
    sb_order: Vec<u32>,
    // Block-level (reused per superblock, sized to BMP_SUPERBLOCK_SIZE)
    local_block_ubs: Vec<f32>,
    local_block_ub_units: Vec<u32>,
    query_presence: Vec<u64>,
    local_block_order: Vec<u32>,
    // Two-phase block scoring: phase1 block UBs (sized to BMP_SUPERBLOCK_SIZE)
    phase1_local_block_ub_units: Vec<u32>,
    // Per-slot accumulator (sized to block_size) — u32 for integer scoring
    acc: Vec<u32>,
    // One decoded random-access group. D uses at most eight values per visit; E
    // uses the full 256-cell group.
    decoded_grid_group: Vec<u8>,
}

impl BmpScratch {
    /// Ensure superblock + local buffers have sufficient capacity.
    fn ensure_capacity_sb(&mut self, num_superblocks: usize, sb_size: usize, block_size: usize) {
        if self.sb_ubs.len() < num_superblocks {
            self.sb_ubs.resize(num_superblocks, 0.0);
        }
        if self.sb_ub_units.len() < num_superblocks {
            self.sb_ub_units.resize(num_superblocks, 0);
        }
        if self.sb_order.capacity() < num_superblocks {
            self.sb_order.reserve(num_superblocks - self.sb_order.len());
        }
        if self.local_block_ubs.len() < sb_size {
            self.local_block_ubs.resize(sb_size, 0.0);
        }
        if self.local_block_ub_units.len() < sb_size {
            self.local_block_ub_units.resize(sb_size, 0);
        }
        if self.local_block_order.len() < sb_size {
            self.local_block_order.resize(sb_size, 0);
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
        if self.decoded_grid_group.len() < GRID_GROUP_CELLS {
            self.decoded_grid_group.resize(GRID_GROUP_CELLS, 0);
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

struct BmpQueryWeights {
    query_by_dim_u16: Vec<(u32, u16)>,
    candidate_grid_weights: Vec<BmpGridWeight>,
    candidate_mask: u64,
    dequant: f32,
}

#[derive(Clone, Copy)]
struct BmpGridWeight {
    dimension: usize,
    weight: u16,
    query_index: usize,
}

/// Globally selected LSP/0 superblocks for one segment.
///
/// The multi-segment searcher computes all segment SBMax values first and
/// partitions the single query-level top-γ set into these segment plans. This
/// prevents a 50-segment index from silently turning γ into `50 × γ`.
#[derive(Debug)]
pub(crate) struct LspSegmentPlan {
    /// Upper bounds aligned one-for-one with `sb_order`. Keeping only selected
    /// values makes plan residency O(gamma), not O(all superblocks).
    pub(crate) sb_ubs: Vec<f32>,
    pub(crate) sb_order: Vec<u32>,
}

/// Robust zero-shot γ schedule reported for LSP/0.
///
/// The paper evaluates 250 for k=10, 500 for k=100, and 1000 for k=1000.
/// Beyond the evaluated range we avoid inventing a sublinear cap: γ grows
/// with the requested depth and is never smaller than 2000.
pub(crate) const fn recommended_lsp_gamma(retrieval_depth: usize) -> usize {
    match retrieval_depth {
        0..=10 => 250,
        11..=100 => 500,
        101..=1000 => 1000,
        _ => {
            if retrieval_depth < 2000 {
                2000
            } else {
                retrieval_depth
            }
        }
    }
}

fn prepare_bmp_query(
    index: &BmpIndex,
    candidate_terms: &[(u32, f32)],
    scoring_terms: &[(u32, f32)],
) -> crate::Result<Option<BmpQueryWeights>> {
    let dims = index.dims();
    let mut query_info: Vec<(u32, f32)> = scoring_terms
        .iter()
        .filter_map(|&(dimension, weight)| {
            (dimension < dims && weight.is_finite() && weight != 0.0)
                .then_some((dimension, weight.abs()))
        })
        .collect();
    if query_info.is_empty() {
        return Ok(None);
    }
    if query_info.len() > crate::query::MAX_QUERY_TERMS {
        return Err(crate::Error::Query(format!(
            "BMP query has {} resolved dimensions; maximum is {}",
            query_info.len(),
            crate::query::MAX_QUERY_TERMS
        )));
    }
    query_info.sort_unstable_by_key(|&(dimension, _)| dimension);

    let max_query_weight = query_info
        .iter()
        .map(|&(_, weight)| weight)
        .fold(0.0f32, f32::max);
    let accumulator_denominator = 255u64.saturating_mul(query_info.len() as u64);
    let max_quantized_weight = ((u32::MAX as u64) / accumulator_denominator).min(16_383);
    if max_quantized_weight == 0 {
        return Err(crate::Error::Query(format!(
            "BMP query has too many resolved dimensions ({})",
            query_info.len()
        )));
    }
    let quant_scale = max_quantized_weight as f32 / max_query_weight;
    let dequant =
        (max_query_weight / max_quantized_weight as f32) * (index.max_weight_scale / 255.0);
    if !dequant.is_finite() {
        return Err(crate::Error::Query(
            "BMP query score scale exceeds the finite f32 range".into(),
        ));
    }

    let query_by_dim_u16: Vec<(u32, u16)> = query_info
        .iter()
        .map(|&(dimension, weight)| {
            (
                dimension,
                (weight * quant_scale)
                    .round()
                    .clamp(0.0, max_quantized_weight as f32) as u16,
            )
        })
        .collect();
    let mut candidate_dimensions: Vec<u32> = candidate_terms
        .iter()
        .filter_map(|&(dimension, weight)| {
            (dimension < dims && weight.is_finite() && weight != 0.0).then_some(dimension)
        })
        .collect();
    candidate_dimensions.sort_unstable();
    candidate_dimensions.dedup();
    let mut candidate_mask = 0u64;
    let mut candidate_grid_weights = Vec::with_capacity(candidate_dimensions.len());
    let mut candidate_index = 0usize;
    for (query_index, &(dimension, weight)) in query_by_dim_u16.iter().enumerate() {
        while candidate_dimensions
            .get(candidate_index)
            .is_some_and(|&candidate| candidate < dimension)
        {
            candidate_index += 1;
        }
        if candidate_dimensions.get(candidate_index) == Some(&dimension) {
            candidate_mask |= 1u64 << query_index;
            candidate_grid_weights.push(BmpGridWeight {
                dimension: dimension as usize,
                weight,
                query_index,
            });
        }
    }
    if candidate_grid_weights.is_empty() {
        return Ok(None);
    }
    Ok(Some(BmpQueryWeights {
        query_by_dim_u16,
        candidate_grid_weights,
        candidate_mask,
        dequant,
    }))
}

/// Compute one segment's SBMax vector for the global LSP/0 selection pass.
pub(crate) fn prepare_lsp_superblock_ubs(
    index: &BmpIndex,
    candidate_terms: &[(u32, f32)],
    scoring_terms: &[(u32, f32)],
) -> crate::Result<Vec<f32>> {
    let Some(weights) = prepare_bmp_query(index, candidate_terms, scoring_terms)? else {
        return Ok(vec![0.0; index.num_superblocks as usize]);
    };
    let count = index.num_superblocks as usize;
    let mut units = vec![0u32; count];
    let mut bounds = vec![0.0f32; count];
    let mut decoded = [0u8; GRID_GROUP_CELLS];
    compute_sb_ubs_int(
        index.superblock_grid(),
        &weights.candidate_grid_weights,
        weights.dequant,
        &mut units,
        &mut bounds,
        &mut decoded,
    )?;
    Ok(bounds)
}

/// Execute a BMP query against the given index.
///
/// Returns top-k results sorted by score descending.
/// Scores are computed over compact virtual documents and resolved to real
/// (doc_id, ordinal) pairs at collection time. The caller's `combine_ordinal_results`
/// handles multi-ordinal grouping via the configured combiner.
///
/// Uses LSP/0 pruning: computes coarse UBs over groups of eight blocks,
/// prunes entire superblocks, then scores only surviving blocks.
///
/// `heap_factor` controls approximate retrieval (BMP alpha parameter):
/// - **1.0**: exact/safe retrieval (default)
/// - **0.8**: prune when `UB * 0.8 < threshold` → ~20% more aggressive
/// - **0.6**: prune when `UB * 0.6 < threshold` → ~40% more aggressive
///
/// `lsp_gamma` is the paper's γ: the maximum number of highest-SBMax
/// superblocks selected for block traversal. Zero means exhaustive top-down
/// traversal while retaining the same safe superblock stopping condition.
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
    lsp_gamma: usize,
) -> crate::Result<Vec<ScoredDoc>> {
    execute_bmp_inner(
        index,
        index_label,
        field_label,
        query_terms,
        query_terms,
        k,
        heap_factor,
        lsp_gamma,
        None,
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
    candidate_terms: &[(u32, f32)],
    scoring_terms: &[(u32, f32)],
    k: usize,
    heap_factor: f32,
    lsp_gamma: usize,
    lsp_plan: Option<&LspSegmentPlan>,
    threshold: BmpThreshold<'_>,
) -> crate::Result<Vec<ScoredDoc>> {
    execute_bmp_inner(
        index,
        index_label,
        field_label,
        candidate_terms,
        scoring_terms,
        k,
        heap_factor,
        lsp_gamma,
        lsp_plan,
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
    lsp_gamma: usize,
    predicate: &dyn Fn(crate::DocId) -> bool,
) -> crate::Result<Vec<ScoredDoc>> {
    execute_bmp_inner(
        index,
        index_label,
        field_label,
        query_terms,
        query_terms,
        k,
        heap_factor,
        lsp_gamma,
        None,
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
    candidate_terms: &[(u32, f32)],
    scoring_terms: &[(u32, f32)],
    k: usize,
    heap_factor: f32,
    lsp_gamma: usize,
    lsp_plan: Option<&LspSegmentPlan>,
    predicate: &dyn Fn(crate::DocId) -> bool,
    threshold: BmpThreshold<'_>,
) -> crate::Result<Vec<ScoredDoc>> {
    execute_bmp_inner(
        index,
        index_label,
        field_label,
        candidate_terms,
        scoring_terms,
        k,
        heap_factor,
        lsp_gamma,
        lsp_plan,
        Some(predicate),
        threshold,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_bmp_inner(
    index: &BmpIndex,
    index_label: &str,
    field_label: &str,
    candidate_terms: &[(u32, f32)],
    scoring_terms: &[(u32, f32)],
    k: usize,
    heap_factor: f32,
    lsp_gamma: usize,
    lsp_plan: Option<&LspSegmentPlan>,
    predicate: Option<&dyn Fn(crate::DocId) -> bool>,
    threshold_source: BmpThreshold<'_>,
) -> crate::Result<Vec<ScoredDoc>> {
    if candidate_terms.is_empty() || scoring_terms.is_empty() || index.num_blocks == 0 || k == 0 {
        return Ok(Vec::new());
    }

    // Alpha parameter for approximate retrieval:
    // UB * alpha < threshold → prune when UB < threshold / alpha
    // alpha < 1.0 means more aggressive pruning (approximate but faster)
    let alpha = heap_factor.clamp(0.01, 1.0);
    let num_blocks = index.num_blocks as usize;
    let block_size = index.bmp_block_size as usize;
    let num_superblocks_total = index.num_superblocks as usize;

    // ── Phase 1: Resolve query dimensions and quantize weights ────────
    let Some(prepared_query) = prepare_bmp_query(index, candidate_terms, scoring_terms)? else {
        return Ok(Vec::new());
    };
    let BmpQueryWeights {
        query_by_dim_u16,
        candidate_grid_weights,
        candidate_mask,
        dequant,
    } = prepared_query;

    // ── Two-phase lazy block scoring setup ────────────────────────────
    // For queries with >5 dims, score only the top-3 heaviest dims first (phase1).
    // If max_partial_score + remaining_block_ub < threshold, skip phase2.
    // For ≤5 dims: full scoring directly (zero overhead).
    const PHASE1_DIMS: usize = 3;
    const MIN_DIMS_FOR_TWO_PHASE: usize = 6;
    let two_phase_active = candidate_mask.count_ones() as usize == query_by_dim_u16.len()
        && (MIN_DIMS_FOR_TWO_PHASE..=64).contains(&query_by_dim_u16.len());
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
    // The planner has already applied the configured ordinal over-fetch factor
    // to `k`.  Do not derive another factor from num_virtual_docs: that ratio
    // includes block-alignment padding and used to multiply the heap depth a
    // second time (usually turning the default 2x into 4x).
    let collector_k = k;

    let t_start = std::time::Instant::now();

    let result = BMP_SCRATCH.with(|cell| -> crate::Result<Vec<ScoredDoc>> {
        let scratch = &mut *cell.borrow_mut();

        // ── Superblock-at-a-time scoring ─────────────────────────────
        scratch.ensure_capacity_sb(
            if lsp_plan.is_some() {
                0
            } else {
                num_superblocks_total
            },
            BMP_SUPERBLOCK_SIZE as usize,
            block_size,
        );

        let (sb_ubs, sb_order, superblock_visit_limit, bounds_follow_order) =
            match lsp_plan {
                Some(plan) => {
                    if plan.sb_ubs.len() != plan.sb_order.len() {
                        return Err(crate::Error::Internal(format!(
                            "global LSP/0 plan has {} bounds for {} selected superblocks",
                            plan.sb_ubs.len(),
                            plan.sb_order.len()
                        )));
                    }
                    if plan
                        .sb_order
                        .iter()
                        .any(|&superblock| superblock as usize >= num_superblocks_total)
                    {
                        return Err(crate::Error::Internal(
                            "global LSP/0 plan contains an invalid superblock id".into(),
                        ));
                    }
                    (
                        plan.sb_ubs.as_slice(),
                        plan.sb_order.as_slice(),
                        plan.sb_order.len(),
                        true,
                    )
                }
                None => {
                    // Single-segment/direct execution computes its own SBMax order.
                    compute_sb_ubs_int(
                        index.superblock_grid(),
                        &candidate_grid_weights,
                        dequant,
                        &mut scratch.sb_ub_units,
                        &mut scratch.sb_ubs,
                        &mut scratch.decoded_grid_group,
                    )?;
                    // LSP/0 requires strict non-increasing SBMax order. A secondary
                    // coverage heuristic would change membership in top-γ.
                    sort_sb_desc_into(
                        &scratch.sb_ubs[..num_superblocks_total],
                        &mut scratch.sb_order,
                    );
                    let visit_limit = if lsp_gamma == 0 {
                        scratch.sb_order.len()
                    } else {
                        lsp_gamma.min(scratch.sb_order.len())
                    };
                    (
                        &scratch.sb_ubs[..num_superblocks_total],
                        scratch.sb_order.as_slice(),
                        visit_limit,
                        false,
                    )
                }
            };

        if sb_order.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 3: score the selected superblocks in SBMax-descending order.
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

        for (idx, &sb_id) in sb_order.iter().take(superblock_visit_limit).enumerate() {
            if let Some(shared) = threshold_source.shared {
                collector.seed_threshold(shared.get());
            }
            let sb_ub = if bounds_follow_order {
                sb_ubs[idx]
            } else {
                sb_ubs[sb_id as usize]
            };

            // LSP/0's superblock condition is SBMax >= theta. Eta/alpha is
            // deliberately NOT applied here; it belongs only to block
            // pruning. With an unpruned query this is rank-safe; candidate
            // query pruning is the paper's intentional approximation while
            // final document scores still use the full query.
            let threshold = collector.threshold();
            if threshold > 0.0 && sb_ub < threshold {
                break;
            }

            let block_start = sb_id as usize * BMP_SUPERBLOCK_SIZE as usize;
            let block_end = (block_start + BMP_SUPERBLOCK_SIZE as usize).min(num_blocks);
            let count = block_end - block_start;

            // Warm one cache line of block offsets for this eight-block superblock.
            // Ensures block_data_ptr() never stalls on offset lookups during scoring.
            // Range includes the sentinel at block_end (needed by block_data_range).
            {
                let bds_base = index.block_data_starts_ptr(0);
                // 8 blocks × 8 bytes = 64 bytes = 1 cache line per prefetch
                for b in (block_start..block_end + 1).step_by(8) {
                    prefetch_read(unsafe { bds_base.add(b * 8) });
                }
            }

            // Decode every candidate-generation dimension once for this
            // eight-block range. The same pass accumulates full/phase-one
            // candidate bounds and produces a block-presence bitset. Final
            // payload scoring below still evaluates all bounded query terms.
            let blocks_with_query_terms = compute_block_ubs_and_presence(
                index.block_grid(),
                index.grid_bits(),
                &candidate_grid_weights,
                query_by_dim_u16.len(),
                phase1_mask,
                block_start,
                count,
                &mut scratch.local_block_ub_units,
                &mut scratch.phase1_local_block_ub_units,
                &mut scratch.local_block_ubs,
                &mut scratch.query_presence,
                &mut scratch.decoded_grid_group,
                dequant,
            )?;
            #[cfg(not(feature = "native"))]
            let _ = blocks_with_query_terms;

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
                    if blocks_with_query_terms & (1u64 << li) == 0 {
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
                &scratch.local_block_ub_units,
                &scratch.query_presence,
                &query_by_dim_u16,
                candidate_mask,
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
                    Some(&scratch.phase1_local_block_ub_units)
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
            if idx + 1 < superblock_visit_limit
                && let Some(&next_sb) = sb_order.get(idx + 1)
            {
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
                "slow BMP: {:.1}ms, sbs={}/{}, gamma={}, blocks={}/{}, dims={}/{}, k={}, returned={}, seed={:.4}, threshold={:.4}, eta={:.2}",
                elapsed_ms,
                sbs_scored,
                num_superblocks_total,
                superblock_visit_limit,
                blocks_scored,
                num_blocks,
                candidate_mask.count_ones(),
                query_by_dim_u16.len(),
                collector_k,
                returned,
                initial_threshold,
                threshold,
                alpha,
            );
        } else {
            log::debug!(
                "BMP execute: {:.1}ms, sbs={}/{}, gamma={}, blocks={}/{}, dims={}/{}, k={}, returned={}, seed={:.4}, threshold={:.4}, eta={:.2}",
                elapsed_ms,
                sbs_scored,
                num_superblocks_total,
                superblock_visit_limit,
                blocks_scored,
                num_blocks,
                candidate_mask.count_ones(),
                query_by_dim_u16.len(),
                collector_k,
                returned,
                initial_threshold,
                threshold,
                alpha,
            );
        }

        Ok(collector_to_results(collector))
    })?;

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
/// Uses one per-query-dimension block-presence bitset before binary search.
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
    query_mask: u64,
    candidate_mask: u64,
    query_presence: &[u64],
    local_block: usize,
    acc: &mut [u32],
    touched: &mut [u64; 4],
    block_size: usize,
    total_block_postings: u32,
) {
    for (q, &(dim_id, w)) in query_by_dim_u16.iter().enumerate() {
        // The fused grid decoder produces one block-presence bitset per query
        // dimension. Avoid transposing those bits into 64 per-block masks.
        let query_bit = 1u64 << q;
        if query_mask & query_bit == 0
            || candidate_mask & query_bit != 0 && query_presence[q] & (1u64 << local_block) == 0
        {
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
/// `local_order` and `local_ubs` are indexed by local offset (0..count);
/// `query_presence` stores one local-block bitset per query dimension.
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
    local_ub_units: &[u32],
    query_presence: &[u64],
    query_by_dim_u16: &[(u32, u16)],
    candidate_mask: u64,
    dequant: f32,
    alpha: f32,
    k: usize,
    predicate: &Option<&dyn Fn(crate::DocId) -> bool>,
    collector: &mut ScoreCollector,
    blocks_scored: &mut u32,
    docmap_lookups: &mut u32,
    acc: &mut [u32],
    phase1_mask: u64,
    phase1_local_ub_units: Option<&[u32]>,
) {
    let block_size = index.bmp_block_size as usize;
    let two_phase = phase1_mask != u64::MAX && phase1_local_ub_units.is_some();

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

        let (num_terms, dim_ptr, ps_ptr, _, post_ptr, total_block_postings) =
            index.parse_block(block_id);
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
                    phase1_mask,
                    candidate_mask,
                    query_presence,
                    local_idx as usize,
                    acc,
                    &mut touched,
                    block_size,
                    total_block_postings,
                );

                // Keep the subtraction and addition in integer accumulator
                // units. Subtracting independently rounded f32 bounds can
                // underestimate the remaining score by one ULP near 2^24.
                let max_possible_units = two_phase_upper_bound_units(
                    max_touched_acc(acc, &touched),
                    local_ub_units[local_idx as usize],
                    phase1_local_ub_units.unwrap()[local_idx as usize],
                );
                let max_possible = max_possible_units as f32 * dequant;
                if max_possible * alpha < collector.threshold() {
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
                    !phase1_mask,
                    candidate_mask,
                    query_presence,
                    local_idx as usize,
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
                    u64::MAX,
                    candidate_mask,
                    query_presence,
                    local_idx as usize,
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

#[inline(always)]
fn two_phase_upper_bound_units(
    max_partial_units: u32,
    full_bound_units: u32,
    phase1_bound_units: u32,
) -> u32 {
    max_partial_units.saturating_add(full_bound_units.saturating_sub(phase1_bound_units))
}

// ============================================================================
// Helpers
// ============================================================================

/// Sort local block indices by their UBs in descending order.
///
/// For eight blocks, a simple sort on non-zero UB indices is optimal.
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
            .total_cmp(&local_ubs[a as usize])
            .then_with(|| a.cmp(&b))
    });
}

/// Sort superblock IDs by SBMax in descending order, into `out`.
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
            .total_cmp(&values[a as usize])
            .then_with(|| a.cmp(&b))
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

/// Compute superblock UBs using integer weights for consistency with integer scoring.
///
/// Uses the same u16 query weights and dequantization factor as `score_block_bsearch_int`,
/// ensuring `sb_ub >= dequantized_score` for any document in the superblock. This avoids
/// a subtle correctness issue where f32-weighted UBs can be slightly LOWER than
/// integer-scored thresholds due to u16 quantization rounding.
///
#[inline]
fn compute_sb_ubs_int(
    grid: &CompressedGrid,
    int_weights: &[BmpGridWeight],
    dequant: f32,
    units: &mut [u32],
    out: &mut [f32],
    decoded: &mut [u8],
) -> crate::Result<()> {
    let nsb = grid.cells();
    debug_assert!(out.len() >= nsb);
    debug_assert!(units.len() >= nsb);
    debug_assert!(decoded.len() >= GRID_GROUP_CELLS);
    units[..nsb].fill(0);

    // E is always consumed as a complete row. Walk its selectors and payload
    // once without allocating descriptors or paying a checkpoint-prefix
    // lookup for every group.
    let cell_scale = block_grid_scale(LSP_SUPERBLOCK_GRID_BITS);
    for &BmpGridWeight {
        dimension, weight, ..
    } in int_weights
    {
        grid.try_for_each_row_group(dimension, |group_id, group| {
            let start = group_id * GRID_GROUP_CELLS;
            let count = GRID_GROUP_CELLS.min(nsb - start);
            if group.width() == 0 {
                return Ok(());
            }
            group.decode(0, count, decoded);
            accumulate_u8(
                decoded,
                count,
                cell_scale * u32::from(weight),
                &mut units[start..start + count],
            );
            Ok(())
        })?;
    }
    for (bound, &integer_units) in out[..nsb].iter_mut().zip(&units[..nsb]) {
        *bound = integer_units as f32 * dequant;
    }
    Ok(())
}

/// Decode one eight-block superblock and compute every grid-derived value in one
/// pass: full bounds, phase-one bounds, and per-query-dimension presence.
///
/// Grid cells are ceiling-quantized upper bounds. Accumulating `u16 × u8` in
/// `u32`, then applying the common dequantizer once, preserves
/// `block_ub >= candidate-query document score` under f32 conversion.
/// Summing already-dequantized terms can round downward and made unpruned
/// exact-mode pruning unsafe near the heap threshold.
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_block_ubs_and_presence(
    grid: &CompressedGrid,
    grid_bits: u8,
    int_weights: &[BmpGridWeight],
    query_term_count: usize,
    phase1_mask: u64,
    block_start: usize,
    count: usize,
    units: &mut [u32],
    phase1_units: &mut [u32],
    out: &mut [f32],
    query_presence: &mut Vec<u64>,
    decoded: &mut [u8],
    dequant: f32,
) -> crate::Result<u64> {
    debug_assert!(units.len() >= count);
    debug_assert!(phase1_units.len() >= count);
    debug_assert!(out.len() >= count);
    debug_assert!(decoded.len() >= count);
    debug_assert!(count <= BMP_SUPERBLOCK_SIZE as usize);
    let group_id = block_start / GRID_GROUP_CELLS;
    let within = block_start % GRID_GROUP_CELLS;
    if within + count > GRID_GROUP_CELLS {
        return Err(crate::Error::Corruption(
            "BMP superblock crosses a compressed-grid group boundary".into(),
        ));
    }

    units[..count].fill(0);
    phase1_units[..count].fill(0);
    if query_presence.len() < query_term_count {
        query_presence.resize(query_term_count, 0);
    }
    query_presence[..query_term_count].fill(0);

    let cell_scale = crate::segment::bmp_grid::block_grid_scale(grid_bits);
    let use_phase1 = phase1_mask != u64::MAX;
    let mut blocks_with_query_terms = 0u64;
    for &BmpGridWeight {
        dimension,
        weight,
        query_index,
    } in int_weights
    {
        let group = grid.group(dimension, group_id)?;
        if group.width() == 0 {
            continue;
        }
        group.decode_u4_packed(within, count, decoded);
        let multiplier = cell_scale * u32::from(weight);
        let phase1 = (use_phase1 && phase1_mask & (1u64 << query_index) != 0)
            .then_some(&mut phase1_units[..count]);
        let presence = accumulate_packed_u4(
            &decoded[..count.div_ceil(2)],
            count,
            multiplier,
            &mut units[..count],
            phase1,
        );
        query_presence[query_index] = presence;
        blocks_with_query_terms |= presence;
    }
    for (bound, &integer_units) in out[..count].iter_mut().zip(&units[..count]) {
        *bound = integer_units as f32 * dequant;
    }
    Ok(blocks_with_query_terms)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::{
        BmpGridWeight, compute_block_ubs_and_presence, recommended_lsp_gamma, sort_sb_desc_into,
        two_phase_upper_bound_units,
    };
    use crate::directories::OwnedBytes;
    use crate::segment::BMP_SUPERBLOCK_SIZE;
    use crate::segment::bmp_grid::{
        CompressedGrid, CompressedGridLayout, GRID_GROUP_CELLS, bit_width, pack_group,
    };

    fn test_grid(rows: &[Vec<u8>], cells: usize) -> CompressedGrid {
        let layout = CompressedGridLayout::new(rows.len(), cells);
        let widths_by_row: Vec<Vec<u8>> = rows
            .iter()
            .map(|row| {
                row.chunks(GRID_GROUP_CELLS)
                    .map(|group| bit_width(group.iter().copied().max().unwrap_or(0)))
                    .collect()
            })
            .collect();
        let row_sizes: Vec<u64> = widths_by_row
            .iter()
            .map(|widths| layout.row_bytes(widths).unwrap())
            .collect();
        let mut bytes = Vec::new();
        layout.write_row_offsets(&row_sizes, &mut bytes).unwrap();
        let mut values = [0u8; GRID_GROUP_CELLS];
        let mut packed = [0u8; GRID_GROUP_CELLS];
        for (row, widths) in rows.iter().zip(&widths_by_row) {
            layout.write_row_header(widths, 4, &mut bytes).unwrap();
            for (group, &width) in widths.iter().enumerate() {
                values.fill(0);
                let start = group * GRID_GROUP_CELLS;
                let count = GRID_GROUP_CELLS.min(cells - start);
                values[..count].copy_from_slice(&row[start..start + count]);
                let len = pack_group(&values, width, &mut packed).unwrap();
                bytes.write_all(&packed[..len]).unwrap();
            }
        }
        CompressedGrid::parse(OwnedBytes::new(bytes), rows.len(), cells, 4, "test grid").unwrap()
    }

    #[test]
    fn integer_block_bound_cannot_round_below_integer_score() {
        // Repeated f32 accumulation underestimates this 64-dimension sum on
        // common targets. The block bound and document score now convert the
        // same integer units exactly once.
        let dimensions = 64usize;
        let rows = vec![vec![15]; dimensions]; // one 4-bit cell (= 255) per row
        let grid = test_grid(&rows, 1);
        let weights: Vec<BmpGridWeight> = (0..dimensions)
            .map(|dimension| BmpGridWeight {
                dimension,
                weight: 160,
                query_index: dimension,
            })
            .collect();
        let dequant = 0.123_456_7f32;
        let mut units = [0u32; 1];
        let mut bounds = [0.0f32; 1];

        let mut phase1_units = [0u32; 1];
        let mut presence = Vec::new();
        let mut decoded = [0u8; GRID_GROUP_CELLS];
        compute_block_ubs_and_presence(
            &grid,
            4,
            &weights,
            dimensions,
            u64::MAX,
            0,
            1,
            &mut units,
            &mut phase1_units,
            &mut bounds,
            &mut presence,
            &mut decoded,
            dequant,
        )
        .unwrap();

        let score_units = 160u32 * 255 * dimensions as u32;
        let score = score_units as f32 * dequant;
        assert!(bounds[0] >= score);
        assert_eq!(units[0], score_units);
    }

    #[test]
    fn two_phase_bound_is_combined_before_f32_rounding() {
        // Around 2^24, independently converting the full and phase-one
        // bounds to f32 loses a unit in their difference:
        // f32(max_partial) + (f32(full) - f32(phase1)) = 16_777_215,
        // while the integer-domain upper bound is 16_777_216.
        let max_partial = 16_777_199u32;
        let full = 16_777_217u32;
        let phase1 = 16_777_200u32;
        let rounded_subtraction = max_partial as f32 + (full as f32 - phase1 as f32).max(0.0);
        let integer_bound = two_phase_upper_bound_units(max_partial, full, phase1) as f32;

        assert!(rounded_subtraction < integer_bound);
        assert_eq!(integer_bound, 16_777_216.0);
    }

    #[test]
    fn packed_integer_bound_kernel_matches_every_4bit_cell() {
        let blocks = 64usize;
        let grid = test_grid(
            &[(0..blocks).map(|block| (block % 16) as u8).collect()],
            blocks,
        );
        for block_start in (0..blocks).step_by(BMP_SUPERBLOCK_SIZE as usize) {
            let mut units = [0u32; BMP_SUPERBLOCK_SIZE as usize];
            let mut phase1_units = [0u32; BMP_SUPERBLOCK_SIZE as usize];
            let mut bounds = [0.0f32; BMP_SUPERBLOCK_SIZE as usize];
            let mut presence = Vec::new();
            let mut decoded = [0u8; GRID_GROUP_CELLS];

            compute_block_ubs_and_presence(
                &grid,
                4,
                &[BmpGridWeight {
                    dimension: 0,
                    weight: 123,
                    query_index: 0,
                }],
                1,
                u64::MAX,
                block_start,
                BMP_SUPERBLOCK_SIZE as usize,
                &mut units,
                &mut phase1_units,
                &mut bounds,
                &mut presence,
                &mut decoded,
                1.0,
            )
            .unwrap();

            for local in 0..BMP_SUPERBLOCK_SIZE as usize {
                let block = block_start + local;
                assert_eq!(units[local], (block % 16) as u32 * 17 * 123);
                assert_eq!(bounds[local], units[local] as f32);
            }
        }
    }

    #[test]
    fn lsp_zero_shot_gamma_schedule_is_depth_aware() {
        assert_eq!(recommended_lsp_gamma(0), 250);
        assert_eq!(recommended_lsp_gamma(10), 250);
        assert_eq!(recommended_lsp_gamma(11), 500);
        assert_eq!(recommended_lsp_gamma(100), 500);
        assert_eq!(recommended_lsp_gamma(101), 1000);
        assert_eq!(recommended_lsp_gamma(1000), 1000);
        assert_eq!(recommended_lsp_gamma(1001), 2000);
        assert_eq!(recommended_lsp_gamma(4800), 4800);
    }

    #[test]
    fn lsp_orders_strictly_by_superblock_maximum() {
        let bounds = [4.0, 9.0, 9.0, 1.0, 7.0];
        let mut order = Vec::new();
        sort_sb_desc_into(&bounds, &mut order);
        assert_eq!(order, vec![1, 2, 4, 0, 3]);
    }
}
