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
//! Compact hierarchy metadata is available at index open. Selected block
//! payloads remain evictable and are decoded on demand. Native execution reuses
//! bounded thread-local scratch; lazy backends admit bounded read windows.
//!
//! - **LSP/0 top-γ**: strict SBMax order with safe superblock-level pruning
//! - **Integer scoring**: u32 accumulators with u16 quantized query weights (~20% faster)
//! - **Superblock pruning**: Skip entire groups of blocks via coarse UBs
//! - **L1 cache locality**: SaaT loop keeps eight blocks' grid data in registers/L1
//! - **Integer-consistent UBs**: bounds and document scores share exact accumulator units
//! - **Pre-scaled weights**: `weight * scale` computed once, not per-block
//! - **Bitmask skip**: Register-level mask check replaces grid DRAM lookups
//! - **Integer-key superblock sort**: packed `(bound bits, id)` u64 keys; only the
//!   top-γ prefix is ordered when a visit cap applies
//! - **Per-block term masks**: presence is transposed once per superblock so a block
//!   probes only the query dimensions it contains
//! - **Binary search scoring**: O(|present_query_dims| × log|block_terms|) per block
//! - **Integer threshold compare**: the heap floor is mapped to accumulator units once
//!   per change, so rejected candidates never convert to f32
//! - **Every drop is counted**: unparseable blocks, out-of-range slots and invalid
//!   doc-map entries surface in `BmpExecutionStats` and a rate-limited warning
//! - **Multi-level prefetch**: SB offset warming → pre-loop burst → N+1/N+2 data pipeline
//! - **Thread-local scratch**: Zero per-query allocation for large buffers
//! - **Early termination**: stop when superblock/block UB < top-k threshold

use super::scoring::{ScoreCollector, ScoredDoc, SharedThreshold};
use crate::segment::bmp_adaptive::{AdaptiveBlock, AdaptivePostings};
use crate::segment::bmp_grid::{
    CompressedGrid, GRID_GROUP_CELLS, GridKernels, LSP_SUPERBLOCK_GRID_BITS, ResolvedGridGroup,
    accumulate_packed_u4_with, accumulate_u8_with, block_grid_scale,
};
use crate::segment::{BMP_SUPERBLOCK_SIZE, BmpIndex};

// dim_id is used directly as grid row index. No dim_idx indirection.

// ============================================================================
// Software prefetch: hint the CPU to load data into cache ahead of time
// ============================================================================

/// Prefetch a memory location for reading with temporal locality.
///
/// This is a no-op on unsupported platforms. On aarch64/x86_64 it issues
/// a hardware prefetch hint. Its cost and benefit depend on the processor and
/// workload, including when the requested line is already cached.
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
/// Segment-execution hierarchy:
/// - **Superblock-level**: sized to num_superblocks, for SB ordering + early termination
/// - **Window-level**: one bounded D/payload lookahead window of `BMP_PREFETCH_SUPERBLOCKS`
/// - **Accumulator**: 256 u32 slots (the on-disk slot type is `u8`), reused per block
struct BmpScratch {
    // Superblock-level (reused across queries, sized to num_superblocks)
    sb_ubs: Vec<f32>,
    sb_ub_units: Vec<u32>,
    sb_order: Vec<u32>,
    /// Packed `(inverted bound bits, superblock id)` sort keys.
    sb_sort_keys: Vec<u64>,
    /// Fixed-length window; `prepared_count` entries are live per window.
    prepared_superblocks: Vec<PreparedSuperblock>,
    /// Distinct D-grid groups covered by the current window (at most the
    /// window size, usually one or two).
    window_groups: Vec<usize>,
    /// Index into `window_groups` for each window position.
    window_group_slot: [u8; BMP_PREFETCH_SUPERBLOCKS],
    /// Resolved `(candidate dimension, window group)` payload locations,
    /// laid out `[weight_index * window_groups.len() + group_slot]`.
    resolved_groups: Vec<ResolvedGridGroup>,
    #[cfg(feature = "native")]
    next_window_groups: Vec<usize>,
    #[cfg(feature = "native")]
    grid_prefetch_ranges: Vec<std::ops::Range<usize>>,
    #[cfg(feature = "native")]
    block_prefetch_ranges: Vec<std::ops::Range<u64>>,
    /// Per-slot accumulator. Fixed at 256 so indexing by a `u8` slot needs
    /// no bounds check; a corrupt slot beyond the block's logical size is
    /// still rejected (and counted) by the scorer before accumulation.
    acc: Box<[u32; 256]>,
    /// One decoded random-access group. D uses at most eight values per
    /// visit; E uses the full 256-cell group.
    decoded_grid_group: Box<[u8; GRID_GROUP_CELLS]>,
}

impl Default for BmpScratch {
    fn default() -> Self {
        Self {
            sb_ubs: Vec::new(),
            sb_ub_units: Vec::new(),
            sb_order: Vec::new(),
            sb_sort_keys: Vec::new(),
            prepared_superblocks: (0..BMP_PREFETCH_SUPERBLOCKS)
                .map(|_| PreparedSuperblock::default())
                .collect(),
            window_groups: Vec::with_capacity(BMP_PREFETCH_SUPERBLOCKS),
            window_group_slot: [0; BMP_PREFETCH_SUPERBLOCKS],
            resolved_groups: Vec::new(),
            #[cfg(feature = "native")]
            next_window_groups: Vec::with_capacity(BMP_PREFETCH_SUPERBLOCKS),
            #[cfg(feature = "native")]
            grid_prefetch_ranges: Vec::new(),
            #[cfg(feature = "native")]
            block_prefetch_ranges: Vec::new(),
            acc: Box::new([0; 256]),
            decoded_grid_group: Box::new([0; GRID_GROUP_CELLS]),
        }
    }
}

impl BmpScratch {
    /// Ensure superblock + window buffers have sufficient capacity.
    fn ensure_capacity_sb(&mut self, num_superblocks: usize, candidate_dims: usize) {
        if self.sb_ubs.len() < num_superblocks {
            self.sb_ubs.resize(num_superblocks, 0.0);
        }
        if self.sb_ub_units.len() < num_superblocks {
            self.sb_ub_units.resize(num_superblocks, 0);
        }
        if self.sb_order.capacity() < num_superblocks {
            self.sb_order.reserve(num_superblocks - self.sb_order.len());
        }
        if self.sb_sort_keys.capacity() < num_superblocks {
            self.sb_sort_keys
                .reserve(num_superblocks - self.sb_sort_keys.len());
        }
        let resolved = candidate_dims * BMP_PREFETCH_SUPERBLOCKS;
        if self.resolved_groups.len() < resolved {
            self.resolved_groups
                .resize(resolved, ResolvedGridGroup::default());
        }
        debug_assert_eq!(self.prepared_superblocks.len(), BMP_PREFETCH_SUPERBLOCKS);
    }
}

const BMP_PREFETCH_SUPERBLOCKS: usize = 8;
const PHASE1_DIMS: usize = 3;
const MIN_DIMS_FOR_TWO_PHASE: usize = 6;

/// One superblock's grid-derived state for the current window. Entries are
/// written in place for each window (every field the block loop reads is
/// rewritten), so no per-superblock zeroing is needed.
#[derive(Default)]
struct PreparedSuperblock {
    sb_ub: f32,
    block_start: usize,
    count: usize,
    block_ubs: [f32; BMP_SUPERBLOCK_SIZE as usize],
    block_ub_units: [u32; BMP_SUPERBLOCK_SIZE as usize],
    phase1_block_ub_units: [u32; BMP_SUPERBLOCK_SIZE as usize],
    /// Per local block: the query-term bitset to probe — candidate
    /// dimensions the D grid marks present plus every non-candidate
    /// scoring dimension (those have no grid row to consult).
    block_terms: [u64; BMP_SUPERBLOCK_SIZE as usize],
    local_order: [u32; BMP_SUPERBLOCK_SIZE as usize],
    local_order_len: usize,
    blocks_with_query_terms: u64,
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

#[derive(Debug)]
pub(crate) struct PreparedBmpQuery {
    dims: u32,
    query_by_dim_u16: Vec<(u32, u16)>,
    candidate_grid_weights: Vec<BmpGridWeight>,
    candidate_mask: u64,
    /// The three heaviest dimensions used by lazy block scoring. Computed
    /// once per query, rather than allocated and sorted again in every segment.
    phase1_mask: u64,
    /// Query-only half of dequantization. Segment max-weight scale is applied
    /// at execution because it is persisted with each BMP index.
    dequant_per_max_weight: f32,
}

#[derive(Clone, Copy, Debug)]
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
    /// Decomposition shared by every segment scorer. Without this, a
    /// multi-segment query rebuilt and allocated the same term-info vector
    /// once per segment even though global LSP had already decomposed it.
    pub(crate) infos: std::sync::Arc<[crate::query::SparseTermQueryInfo]>,
    /// Query resolution/quantization is identical across immutable segments.
    /// Retaining it here prevents the LSP pass and every segment scorer from
    /// independently sorting and allocating the same vectors.
    pub(crate) prepared_query: std::sync::Arc<PreparedBmpQuery>,
    /// `None` means local/exhaustive SB traversal. `Some`, including an empty
    /// selection, is the single global top-gamma projection for this segment.
    pub(crate) selection: Option<LspSelection>,
}

#[derive(Debug)]
pub(crate) struct LspSelection {
    /// Upper bounds aligned one-for-one with `sb_order`. Keeping only selected
    /// values makes plan residency O(gamma), not O(all superblocks).
    pub(crate) sb_ubs: Vec<f32>,
    pub(crate) sb_order: Vec<u32>,
}

impl LspSegmentPlan {
    #[inline]
    pub(crate) fn has_work(&self) -> bool {
        self.selection
            .as_ref()
            .is_none_or(|selection| !selection.sb_order.is_empty())
    }

    #[inline]
    pub(crate) fn priority(&self) -> f32 {
        match &self.selection {
            // Exhaustive/local traversal has no precomputed E bound. It still
            // needs a pilot and is ordered by segment size by the caller.
            None => f32::INFINITY,
            Some(selection) => selection
                .sb_ubs
                .first()
                .copied()
                .unwrap_or(f32::NEG_INFINITY),
        }
    }
}

/// γ schedule for LSP/0 superblock selection.
///
/// The paper's zero-shot values (250 for k=10, 500 for k=100, 1000 for
/// k=1000) are too aggressive for a large, topically dense corpus: superblock
/// upper bounds are sums of per-block dimension maxima, so a query whose
/// terms are common in the corpus inflates hundreds of blocks above the true
/// best document's *actual* score, pushing its block out of the top-γ
/// projection entirely. Measured on the 69M-doc production index
/// (2026-08-23): an exact-title query's true top-1 document (actual score
/// 24.3 vs 21.5 for the approximate head) was dropped at γ=500 and only
/// recovered at γ≥1000; a broad topical query needed γ≥2000. Selection cost
/// is nearly flat in γ up to ~5000 (< 50ms end-to-end delta), so the
/// schedule buys reliability with margin. It also removes the paper
/// schedule's rank instability across result windows (the γ step at depth
/// 101 made shallow and deep requests disagree on the head).
pub(crate) const fn recommended_lsp_gamma(retrieval_depth: usize) -> usize {
    match retrieval_depth {
        0..=100 => 3000,
        101..=1000 => 4000,
        _ => {
            if retrieval_depth < 4000 {
                4000
            } else {
                retrieval_depth
            }
        }
    }
}

pub(crate) fn prepare_bmp_query(
    dims: u32,
    candidate_terms: &[(u32, f32)],
    scoring_terms: &[(u32, f32)],
) -> crate::Result<Option<PreparedBmpQuery>> {
    prepare_bmp_query_iter(
        dims,
        candidate_terms.iter().copied(),
        scoring_terms.iter().copied(),
    )
}

pub(crate) fn prepare_bmp_query_infos(
    dims: u32,
    infos: &[crate::query::SparseTermQueryInfo],
) -> crate::Result<Option<PreparedBmpQuery>> {
    prepare_bmp_query_iter(
        dims,
        infos
            .iter()
            .filter(|info| info.candidate)
            .map(|info| (info.dim_id, info.weight)),
        infos.iter().map(|info| (info.dim_id, info.weight)),
    )
}

fn prepare_bmp_query_iter(
    dims: u32,
    candidate_terms: impl IntoIterator<Item = (u32, f32)>,
    scoring_terms: impl IntoIterator<Item = (u32, f32)>,
) -> crate::Result<Option<PreparedBmpQuery>> {
    let mut query_info: Vec<(u32, f32)> = scoring_terms
        .into_iter()
        .filter_map(|(dimension, weight)| {
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
    let dequant_per_max_weight = (max_query_weight / max_quantized_weight as f32) / 255.0;
    if !dequant_per_max_weight.is_finite() {
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
        .into_iter()
        .filter_map(|(dimension, weight)| {
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
    let phase1_mask = compute_phase1_mask(&query_by_dim_u16, candidate_mask);
    Ok(Some(PreparedBmpQuery {
        dims,
        query_by_dim_u16,
        candidate_grid_weights,
        candidate_mask,
        phase1_mask,
        dequant_per_max_weight,
    }))
}

fn compute_phase1_mask(query: &[(u32, u16)], candidate_mask: u64) -> u64 {
    if candidate_mask.count_ones() as usize != query.len()
        || !(MIN_DIMS_FOR_TWO_PHASE..=crate::query::MAX_QUERY_TERMS).contains(&query.len())
    {
        return u64::MAX;
    }

    let mut heaviest = [(0u16, usize::MAX); PHASE1_DIMS];
    for (index, &(_, weight)) in query.iter().enumerate() {
        for position in 0..PHASE1_DIMS {
            if heaviest[position].1 == usize::MAX || weight > heaviest[position].0 {
                heaviest[position..].rotate_right(1);
                heaviest[position] = (weight, index);
                break;
            }
        }
    }
    heaviest
        .iter()
        .fold(0u64, |mask, &(_, index)| mask | (1u64 << index))
}

impl PreparedBmpQuery {
    fn dequant_for(&self, index: &BmpIndex) -> crate::Result<f32> {
        if index.dims() != self.dims {
            return Err(crate::Error::Corruption(format!(
                "BMP query was prepared for {} dimensions but segment has {}",
                self.dims,
                index.dims()
            )));
        }
        let dequant = self.dequant_per_max_weight * index.max_weight_scale;
        if !dequant.is_finite() {
            return Err(crate::Error::Query(
                "BMP query score scale exceeds the finite f32 range".into(),
            ));
        }
        Ok(dequant)
    }
}

/// Compute the small H-grid bounds used to drive exact hierarchical LSP
/// selection. Each cell safely bounds 256 E-grid superblocks.
pub(crate) fn prepare_lsp_coarse_ubs(
    index: &BmpIndex,
    prepared: &PreparedBmpQuery,
) -> crate::Result<Vec<f32>> {
    let dequant = prepared.dequant_for(index)?;
    let count = index.num_coarse_groups as usize;
    let mut units = vec![0u32; count];
    let mut bounds = vec![0.0f32; count];
    let mut decoded = [0u8; GRID_GROUP_CELLS];
    compute_grid_ubs_int(
        index.coarse_grid(),
        GridKernels::detect(),
        &prepared.candidate_grid_weights,
        dequant,
        &mut units,
        &mut bounds,
        &mut decoded,
    )?;
    Ok(bounds)
}

/// Expand one H cell into its exact E-grid superblock bounds.
///
/// H uses the same 256-cell grouping as E's random-access codec, so expansion
/// performs one group lookup per query dimension and never scans unrelated E
/// payload. `output` is cleared and receives only positive bounds.
pub(crate) fn expand_lsp_coarse_group(
    index: &BmpIndex,
    prepared: &PreparedBmpQuery,
    coarse_group: u32,
    output: &mut Vec<(u32, f32)>,
) -> crate::Result<()> {
    let coarse_group = coarse_group as usize;
    if coarse_group >= index.num_coarse_groups as usize {
        return Err(crate::Error::Corruption(format!(
            "BMP coarse group {coarse_group} exceeds {}",
            index.num_coarse_groups
        )));
    }
    let start = coarse_group * GRID_GROUP_CELLS;
    let count = GRID_GROUP_CELLS.min(index.num_superblocks as usize - start);
    let dequant = prepared.dequant_for(index)?;
    let mut units = [0u32; GRID_GROUP_CELLS];
    let mut decoded = [0u8; GRID_GROUP_CELLS];
    let multiplier_scale = block_grid_scale(LSP_SUPERBLOCK_GRID_BITS);
    let kernels = GridKernels::detect();
    for &BmpGridWeight {
        dimension, weight, ..
    } in &prepared.candidate_grid_weights
    {
        let group = index.superblock_grid().group(dimension, coarse_group)?;
        if group.width() == 0 {
            continue;
        }
        group.decode_with(kernels, 0, count, &mut decoded);
        accumulate_u8_with(
            kernels,
            &decoded,
            count,
            multiplier_scale * u32::from(weight),
            &mut units,
        );
    }
    output.clear();
    output.reserve(count);
    output.extend(
        units[..count]
            .iter()
            .enumerate()
            .filter(|&(_, &integer_units)| integer_units != 0)
            .map(|(within, &integer_units)| {
                ((start + within) as u32, integer_units as f32 * dequant)
            }),
    );
    Ok(())
}

/// BMP execution with a planner-validated live cross-segment threshold.
///
/// Scores compact virtual documents and resolves `(doc_id, ordinal)` only
/// after score-floor pruning. The caller combines ordinals. A zero
/// `lsp_gamma` is exhaustive; positive gamma and `heap_factor < 1.0` are
/// explicit approximations.
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
    Ok(execute_bmp_with_threshold_stats(
        index,
        index_label,
        field_label,
        candidate_terms,
        scoring_terms,
        k,
        heap_factor,
        lsp_gamma,
        lsp_plan,
        threshold,
    )?
    .0)
}

/// [`execute_bmp_with_threshold`] that also returns the execution counters,
/// including the integrity counters that regression tests pin.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_bmp_with_threshold_stats(
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
) -> crate::Result<(Vec<ScoredDoc>, BmpExecutionStats)> {
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
    Ok(execute_bmp_inner(
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
    )?
    .0)
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
) -> crate::Result<(Vec<ScoredDoc>, BmpExecutionStats)> {
    // The only always-on clock: it feeds the slow-query log line. Phase
    // timers below are `observe::Timer`, which folds away without `metrics`.
    let total_start = crate::observe::WallTimer::start();
    if index.num_blocks == 0
        || k == 0
        || lsp_plan.is_none() && (candidate_terms.is_empty() || scoring_terms.is_empty())
    {
        return Ok((Vec::new(), BmpExecutionStats::default()));
    }

    // Alpha parameter for approximate retrieval:
    // UB * alpha < threshold → prune when UB < threshold / alpha
    // alpha < 1.0 means more aggressive pruning (approximate but faster)
    let alpha = heap_factor.clamp(0.01, 1.0);
    let num_blocks = index.num_blocks as usize;
    let num_superblocks_total = index.num_superblocks as usize;
    let kernels = GridKernels::detect();

    // ── Phase 1: Resolve query dimensions and quantize weights ────────
    let prepare_start = crate::observe::Timer::start();
    let local_prepared;
    let prepared_query = if let Some(plan) = lsp_plan {
        plan.prepared_query.as_ref()
    } else {
        let Some(prepared) = prepare_bmp_query(index.dims(), candidate_terms, scoring_terms)?
        else {
            return Ok((Vec::new(), BmpExecutionStats::default()));
        };
        local_prepared = prepared;
        &local_prepared
    };
    let prepare_secs = prepare_start.secs();
    let query_by_dim_u16 = prepared_query.query_by_dim_u16.as_slice();
    let candidate_grid_weights = prepared_query.candidate_grid_weights.as_slice();
    let candidate_mask = prepared_query.candidate_mask;
    let dequant = prepared_query.dequant_for(index)?;
    // Scoring dimensions pruned from the candidate set have no grid row, so
    // every block probes them.
    let non_candidate_mask = !candidate_mask & valid_query_bits(query_by_dim_u16.len());

    // ── Two-phase lazy block scoring setup ────────────────────────────
    // For queries with >5 dims, score only the top-3 heaviest dims first (phase1).
    // If max_partial_score + remaining_block_ub < threshold, skip phase2.
    // For ≤5 dims: full scoring directly (zero overhead).
    let phase1_mask = prepared_query.phase1_mask;
    let two_phase_active = phase1_mask != u64::MAX;
    // The planner has already applied the configured ordinal over-fetch factor
    // to `k`.  Do not derive another factor from num_virtual_docs: that ratio
    // includes block-alignment padding and used to multiply the heap depth a
    // second time (usually turning the default 2x into 4x).
    let collector_k = k;

    let result = BMP_SCRATCH.with(|cell| -> crate::Result<(Vec<ScoredDoc>, BmpExecutionStats)> {
        let scratch = &mut *cell.borrow_mut();

        // ── Superblock-at-a-time scoring ─────────────────────────────
        scratch.ensure_capacity_sb(
            if lsp_plan
                .and_then(|plan| plan.selection.as_ref())
                .is_some()
            {
                0
            } else {
                num_superblocks_total
            },
            candidate_grid_weights.len(),
        );

        let (sb_ubs, sb_order, superblock_visit_limit, bounds_follow_order) =
            match lsp_plan.and_then(|plan| plan.selection.as_ref()) {
                Some(selection) => {
                    if selection.sb_ubs.len() != selection.sb_order.len() {
                        return Err(crate::Error::Internal(format!(
                            "global LSP/0 plan has {} bounds for {} selected superblocks",
                            selection.sb_ubs.len(),
                            selection.sb_order.len()
                        )));
                    }
                    if selection
                        .sb_order
                        .iter()
                        .any(|&superblock| superblock as usize >= num_superblocks_total)
                    {
                        return Err(crate::Error::Internal(
                            "global LSP/0 plan contains an invalid superblock id".into(),
                        ));
                    }
                    (
                        selection.sb_ubs.as_slice(),
                        selection.sb_order.as_slice(),
                        selection.sb_order.len(),
                        true,
                    )
                }
                None => {
                    // Single-segment/direct execution computes its own SBMax order.
                    compute_grid_ubs_int(
                        index.superblock_grid(),
                        kernels,
                        candidate_grid_weights,
                        dequant,
                        &mut scratch.sb_ub_units,
                        &mut scratch.sb_ubs,
                        &mut scratch.decoded_grid_group[..],
                    )?;
                    // LSP/0 requires strict non-increasing SBMax order. A secondary
                    // coverage heuristic would change membership in top-γ.
                    let visit_cap = if lsp_gamma == 0 { usize::MAX } else { lsp_gamma };
                    sort_sb_desc_into(
                        &scratch.sb_ubs[..num_superblocks_total],
                        visit_cap,
                        &mut scratch.sb_sort_keys,
                        &mut scratch.sb_order,
                    );
                    let visit_limit = visit_cap.min(scratch.sb_order.len());
                    (
                        &scratch.sb_ubs[..num_superblocks_total],
                        scratch.sb_order.as_slice(),
                        visit_limit,
                        false,
                    )
                }
            };

        if sb_order.is_empty() {
            return Ok((Vec::new(), BmpExecutionStats::default()));
        }

        // Phase 3: score the selected superblocks in SBMax-descending order.
        let mut stats = BmpExecutionStats::default();
        let mut phases = crate::observe::BmpQueryPhases {
            prepare_secs,
            ..Default::default()
        };
        let mut collector = ScoreCollector::new(collector_k);
        let initial_threshold = threshold_source
            .shared
            .map(SharedThreshold::get)
            .unwrap_or(0.0)
            .max(threshold_source.initial);
        collector.seed_threshold(initial_threshold);
        let mut threshold_units = ThresholdUnits::new(dequant, alpha);
        let mut cursor = 0usize;

        // Stage one of the first window's D-grid pipeline: only deterministic
        // selector/checkpoint ranges are needed, so no cold row metadata is
        // dereferenced here.
        #[cfg(feature = "native")]
        {
            let prefetch_start = crate::observe::Timer::start();
            let (bytes, calls) = prefetch_grid_metadata_window(
                index.block_grid(),
                candidate_grid_weights,
                sb_order,
                cursor,
                superblock_visit_limit,
                &mut scratch.next_window_groups,
                &mut scratch.grid_prefetch_ranges,
            )?;
            phases.prefetch_secs += prefetch_start.secs();
            phases.prefetched_bytes = phases.prefetched_bytes.saturating_add(bytes);
            phases.prefetch_ranges = phases.prefetch_ranges.saturating_add(calls);
        }

        'windows: while cursor < superblock_visit_limit {
            if let Some(shared) = threshold_source.shared {
                collector.seed_threshold(shared.get());
            }
            let first_sb = sb_order[cursor];
            let first_ub = if bounds_follow_order {
                sb_ubs[cursor]
            } else {
                sb_ubs[first_sb as usize]
            };
            // LSP/0's superblock condition is SBMax >= theta. Eta/alpha is
            // deliberately NOT applied here; it belongs only to block
            // pruning. With an unpruned query this is rank-safe; candidate
            // query pruning is the paper's intentional approximation while
            // final document scores still use the full query.
            let threshold = collector.threshold();
            if threshold > 0.0 && first_ub < threshold {
                break;
            }

            let window_start = cursor;
            let window_end =
                (cursor + BMP_PREFETCH_SUPERBLOCKS).min(superblock_visit_limit);

            // Locate every (candidate dimension, D-grid group) payload of this
            // window exactly once. Native builds advise those payload pages
            // (stage two of the D pipeline) and the D pass below decodes from
            // the same resolved locations instead of walking the metadata again.
            let d_grid_start = crate::observe::Timer::start();
            resolve_grid_window(
                index.block_grid(),
                candidate_grid_weights,
                &sb_order[window_start..window_end],
                &mut scratch.window_groups,
                &mut scratch.window_group_slot,
                &mut scratch.resolved_groups,
            )?;
            phases.d_grid_secs += d_grid_start.secs();
            let window_group_count = scratch.window_groups.len();
            let resolved_len = candidate_grid_weights.len() * window_group_count;
            #[cfg(feature = "native")]
            {
                let prefetch_start = crate::observe::Timer::start();
                let (bytes, calls) = prefetch_resolved_grid_payloads(
                    index.block_grid(),
                    &scratch.resolved_groups[..resolved_len],
                    &mut scratch.grid_prefetch_ranges,
                );
                phases.prefetch_secs += prefetch_start.secs();
                phases.prefetched_bytes = phases.prefetched_bytes.saturating_add(bytes);
                phases.prefetch_ranges = phases.prefetch_ranges.saturating_add(calls);
            }

            let d_grid_start = crate::observe::Timer::start();
            let window_threshold = collector.threshold();
            let mut stop_after_window = false;
            let mut prepared_count = 0usize;
            // Disjoint field borrows: `sb_ubs`/`sb_order` may alias
            // `scratch.sb_ubs`/`scratch.sb_order` on the local path.
            {
                let resolved_groups = &scratch.resolved_groups[..resolved_len];
                for position in window_start..window_end {
                    let sb_id = sb_order[position];
                    let sb_ub = if bounds_follow_order {
                        sb_ubs[position]
                    } else {
                        sb_ubs[sb_id as usize]
                    };
                    if window_threshold > 0.0 && sb_ub < window_threshold {
                        stop_after_window = true;
                        break;
                    }

                    let block_start = sb_id as usize * BMP_SUPERBLOCK_SIZE as usize;
                    let block_end =
                        (block_start + BMP_SUPERBLOCK_SIZE as usize).min(num_blocks);
                    let count = block_end - block_start;
                    let bds_base = index.block_data_starts_ptr(0);
                    for block in (block_start..block_end + 1).step_by(8) {
                        prefetch_read(unsafe { bds_base.add(block * 8) });
                    }

                    let prepared = &mut scratch.prepared_superblocks[prepared_count];
                    prepared.sb_ub = sb_ub;
                    prepared.block_start = block_start;
                    prepared.count = count;
                    let group_slot =
                        usize::from(scratch.window_group_slot[position - window_start]);
                    prepared.blocks_with_query_terms = compute_block_ubs_and_presence(
                        index.block_grid(),
                        kernels,
                        index.grid_bits(),
                        candidate_grid_weights,
                        resolved_groups,
                        window_group_count,
                        group_slot,
                        non_candidate_mask,
                        phase1_mask,
                        block_start,
                        count,
                        &mut prepared.block_ub_units,
                        &mut prepared.phase1_block_ub_units,
                        &mut prepared.block_ubs,
                        &mut prepared.block_terms,
                        &mut scratch.decoded_grid_group[..],
                        dequant,
                    )?;
                    prepared.local_order_len = sort_local_blocks_desc_fixed(
                        &prepared.block_ubs,
                        count,
                        &mut prepared.local_order,
                    );
                    prepared_count += 1;
                }
            }
            cursor = window_start.saturating_add(prepared_count);
            phases.d_grid_secs += d_grid_start.secs();
            if prepared_count == 0 {
                break;
            }

            #[cfg(feature = "native")]
            {
                let prefetch_start = crate::observe::Timer::start();
                // Advise the next D selector window now; current payload
                // scoring supplies the actual I/O lead time.
                let (grid_bytes, grid_calls) = prefetch_grid_metadata_window(
                    index.block_grid(),
                    candidate_grid_weights,
                    sb_order,
                    cursor,
                    superblock_visit_limit,
                    &mut scratch.next_window_groups,
                    &mut scratch.grid_prefetch_ranges,
                )?;

                let (block_bytes, block_calls) = if index.block_data_resident() {
                    (0, 0)
                } else {
                    scratch.block_prefetch_ranges.clear();
                    let heap_full = collector.len() >= collector_k;
                    let threshold = collector.threshold();
                    for prepared in &scratch.prepared_superblocks[..prepared_count] {
                        for &local in &prepared.local_order[..prepared.local_order_len] {
                            let local = local as usize;
                            if heap_full && prepared.block_ubs[local] * alpha < threshold {
                                break;
                            }
                            if prepared.blocks_with_query_terms & (1u64 << local) == 0 {
                                continue;
                            }
                            let (start, end) =
                                index.block_data_range((prepared.block_start + local) as u32);
                            if start < end {
                                scratch.block_prefetch_ranges.push(start..end);
                            }
                        }
                    }
                    index.prefetch_block_data_ranges(&mut scratch.block_prefetch_ranges)
                };
                phases.prefetch_secs += prefetch_start.secs();
                phases.prefetched_bytes = phases
                    .prefetched_bytes
                    .saturating_add(grid_bytes)
                    .saturating_add(block_bytes);
                phases.prefetch_ranges = phases
                    .prefetch_ranges
                    .saturating_add(grid_calls)
                    .saturating_add(block_calls);
            }

            let (prepared_superblocks, acc) = (
                &scratch.prepared_superblocks[..prepared_count],
                &mut scratch.acc,
            );
            for prepared in prepared_superblocks {
                if let Some(shared) = threshold_source.shared {
                    collector.seed_threshold(shared.get());
                }
                if collector.threshold() > 0.0
                    && prepared.sb_ub < collector.threshold()
                {
                    stop_after_window = true;
                    break;
                }
                let score_start = crate::observe::Timer::start();
                score_superblock_blocks(
                    index,
                    prepared,
                    query_by_dim_u16,
                    dequant,
                    alpha,
                    collector_k,
                    &predicate,
                    &mut collector,
                    &mut threshold_units,
                    &mut stats,
                    acc,
                    phase1_mask,
                    two_phase_active,
                    &mut phases.docmap_secs,
                );
                phases.block_score_secs += score_start.secs();
                stats.superblocks_scored += 1;

                if threshold_source.publish
                    && collector.real_len() >= collector_k
                    && let Some(shared) = threshold_source.shared
                {
                    shared.raise(collector.threshold());
                }
            }

            if stop_after_window {
                break 'windows;
            }
            // A partially filled window means the descending SB list crossed
            // the current threshold; every later superblock is also prunable.
            if prepared_count < window_end - window_start {
                break;
            }
        }

        let elapsed_ms = total_start.secs() * 1000.0;
        let threshold = collector.threshold();
        let returned = collector.real_len();
        crate::observe::bmp_query(
            index_label,
            field_label,
            total_start.secs(),
            phases,
            stats.superblocks_scored as usize,
            num_superblocks_total,
            stats.blocks_scored as usize,
            num_blocks,
            stats.docmap_lookups as usize,
        );
        if stats.has_integrity_faults() {
            report_integrity_faults(index_label, field_label, index, &stats);
        }
        if elapsed_ms > 500.0 {
            log::warn!(
                "slow BMP: {:.1}ms, sbs={}/{}, gamma={}, blocks={}/{}, dims={}/{}, k={}, returned={}, seed={:.4}, threshold={:.4}, eta={:.2}",
                elapsed_ms,
                stats.superblocks_scored,
                num_superblocks_total,
                superblock_visit_limit,
                stats.blocks_scored,
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
                stats.superblocks_scored,
                num_superblocks_total,
                superblock_visit_limit,
                stats.blocks_scored,
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

        Ok((collector_to_results(collector), stats))
    })?;

    Ok(result)
}

// ============================================================================
// Execution counters and integrity reporting
// ============================================================================

/// Per-execution counters for one BMP segment query.
///
/// The `corrupt_*`, `dropped_*` and `invalid_*` counters are integrity
/// faults: every place the executor used to skip data silently now counts
/// it here, and any non-zero fault triggers a rate-limited warning naming the
/// index/field. They are zero for a well-formed segment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BmpExecutionStats {
    pub(crate) superblocks_scored: u32,
    /// Blocks whose payload was parsed and scored. An unparseable block is
    /// counted in `corrupt_blocks` instead, never here.
    pub(crate) blocks_scored: u32,
    pub(crate) docmap_lookups: u32,
    /// Blocks the grid selected for scoring whose payload failed to parse.
    pub(crate) corrupt_blocks: u32,
    /// Query dimensions found in a block's term table whose posting range
    /// was invalid.
    pub(crate) corrupt_terms: u32,
    /// Sparse postings whose slot lies outside the block's logical size.
    pub(crate) dropped_postings: u32,
    /// Scored slots that mapped past the virtual-document count or to a
    /// document-map entry outside the segment.
    pub(crate) invalid_docmap_entries: u32,
}

impl BmpExecutionStats {
    #[inline(always)]
    fn absorb(&mut self, faults: BlockScoreFaults) {
        self.dropped_postings += faults.dropped_postings;
        self.corrupt_terms += faults.corrupt_terms;
    }

    pub(crate) fn has_integrity_faults(&self) -> bool {
        (self.corrupt_blocks
            | self.corrupt_terms
            | self.dropped_postings
            | self.invalid_docmap_entries)
            != 0
    }
}

/// Faults observed while scoring one block; folded into `BmpExecutionStats`.
#[derive(Clone, Copy, Default)]
struct BlockScoreFaults {
    dropped_postings: u32,
    corrupt_terms: u32,
}

/// Number of queries (process-wide) that hit at least one integrity fault.
static INTEGRITY_FAULT_QUERIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Warn about silently unusable data with exponential back-off: the 1st,
/// 2nd, 4th, 8th, ... faulting query is logged, so a corrupt segment under
/// load cannot flood the log but also never goes quiet. Per-query counts are
/// always available through `BmpExecutionStats` and exported as the
/// `summa_bmp_corrupt_blocks_total` family of counters on every faulting
/// query, not only the logged ones.
fn report_integrity_faults(
    index_label: &str,
    field_label: &str,
    index: &BmpIndex,
    stats: &BmpExecutionStats,
) {
    crate::observe::bmp_integrity_faults(
        index_label,
        field_label,
        u64::from(stats.corrupt_blocks),
        u64::from(stats.corrupt_terms),
        u64::from(stats.dropped_postings),
        u64::from(stats.invalid_docmap_entries),
    );
    let nth = INTEGRITY_FAULT_QUERIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if nth.is_power_of_two() {
        log::warn!(
            "BMP integrity faults on index={} field={} (segment with {} blocks, {} virtual docs, block_size={}): \
             corrupt_blocks={} corrupt_terms={} dropped_postings={} invalid_docmap_entries={} \
             — faulting query #{} in this process; next report at #{}",
            index_label,
            field_label,
            index.num_blocks,
            index.num_virtual_docs,
            index.bmp_block_size,
            stats.corrupt_blocks,
            stats.corrupt_terms,
            stats.dropped_postings,
            stats.invalid_docmap_entries,
            nth,
            nth.saturating_mul(2),
        );
    }
}

/// Bit mask of the valid query-term indices for a query of `query_len` terms.
#[inline(always)]
fn valid_query_bits(query_len: usize) -> u64 {
    if query_len >= u64::BITS as usize {
        u64::MAX
    } else {
        (1u64 << query_len) - 1
    }
}

// ============================================================================
// Integer threshold image
// ============================================================================

/// Exact integer image of the collector threshold for the two per-candidate
/// f32 comparisons in block scoring.
///
/// Scores are `units as f32 * dequant`; the two-phase bound is
/// `(units as f32 * dequant) * alpha`. Both maps `f` are monotone
/// non-decreasing in `units`: `u32 -> f32` conversion and multiplication by
/// a positive finite constant each preserve `<=` under IEEE-754
/// round-to-nearest. Hence for a threshold `t` there is a smallest `m` with
/// `f(m) >= t`, and for every `u`: `f(u) < t  <=>  u < m`. Comparing
/// `u < m` rejects *exactly* the candidates the f32 comparison rejected — it
/// is equivalent, not merely conservative — so results are bit-identical
/// while rejected candidates skip the conversion and multiply. `m` is kept
/// as `u64` so "no u32 reaches `t`" is representable as `2^32`.
///
/// `refresh` runs where the threshold can change (block start, after a heap
/// insertion) and short-circuits when the value is unchanged.
struct ThresholdUnits {
    dequant: f32,
    alpha: f32,
    threshold: f32,
    score_min: u64,
    phase_min: u64,
}

impl ThresholdUnits {
    fn new(dequant: f32, alpha: f32) -> Self {
        Self {
            dequant,
            alpha,
            // NaN never compares equal, so the first refresh always computes.
            threshold: f32::NAN,
            score_min: 0,
            phase_min: 0,
        }
    }

    #[inline]
    fn refresh(&mut self, threshold: f32) {
        if threshold == self.threshold {
            return;
        }
        self.threshold = threshold;
        let dequant = self.dequant;
        let alpha = self.alpha;
        self.score_min = min_units_reaching(threshold, f64::from(dequant), |units| {
            units as f32 * dequant
        });
        self.phase_min =
            min_units_reaching(threshold, f64::from(dequant) * f64::from(alpha), |units| {
                (units as f32 * dequant) * alpha
            });
    }

    /// `units as f32 * dequant < threshold`, in integer units.
    #[inline(always)]
    fn rejects_score(&self, units: u32) -> bool {
        u64::from(units) < self.score_min
    }

    /// `(units as f32 * dequant) * alpha < threshold`, in integer units.
    #[inline(always)]
    fn rejects_phase(&self, units: u32) -> bool {
        u64::from(units) < self.phase_min
    }
}

/// Smallest `m` in `0..2^32` with `f(m) >= threshold`, or `2^32` when no
/// `u32` reaches it. `f` must be monotone non-decreasing and `scale`
/// approximates `f(u) / u`, which only seeds the search bracket: the
/// bracket is verified against `f` and widened to the full range if the
/// estimate is off, so the result never depends on the estimate's accuracy.
fn min_units_reaching(threshold: f32, scale: f64, f: impl Fn(u32) -> f32) -> u64 {
    const LIMIT: u64 = 1 << 32;
    if threshold.is_nan() {
        // `x < NaN` is false for every x: the f32 compare rejects nothing.
        return 0;
    }
    if f(u32::MAX) < threshold {
        return LIMIT;
    }
    let estimate = f64::from(threshold) / scale;
    let (mut lo, mut hi) = if estimate.is_finite() && estimate >= 0.0 {
        let margin = 2f64.powi(-16);
        let lo = (estimate * (1.0 - margin)).floor().min(u32::MAX as f64);
        let hi = (estimate * (1.0 + margin)).ceil().min(u32::MAX as f64);
        (lo as u64, hi as u64)
    } else {
        (0, u64::from(u32::MAX))
    };
    // Search invariants: `lo == 0 || f(lo - 1) < threshold` and
    // `f(hi) >= threshold`; the answer therefore lies in `lo..=hi`.
    if lo > 0 && f((lo - 1) as u32) >= threshold {
        lo = 0;
    }
    if f(hi as u32) < threshold {
        hi = u64::from(u32::MAX);
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if f(mid as u32) >= threshold {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

// ============================================================================
// Integer scoring: u32 accumulators with u16 quantized query weights
// ============================================================================

/// Find the maximum u32 value across touched accumulator slots.
///
/// Uses the touched bitmask for O(|touched_slots|) — typically 5-20 slots per block.
#[inline(always)]
fn max_touched_acc(acc: &[u32; 256], touched: &[u64; 4]) -> u32 {
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
fn zero_touched_acc(acc: &mut [u32; 256], touched: &[u64; 4]) {
    for word in 0..4 {
        let mut bits = touched[word];
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            acc[word * 64 + bit] = 0;
        }
    }
}

/// Dense rows deliberately mark their whole block. At the density crossover
/// at least half of those slots already contain postings, and a full-word mask
/// lets the accumulation loop remain branch-free and vectorizable.
#[inline(always)]
fn touch_dense_block(touched: &mut [u64; 4], block_size: usize) {
    let full_words = block_size / 64;
    touched[..full_words].fill(u64::MAX);
    let remainder = block_size % 64;
    if remainder != 0 {
        touched[full_words] |= (1u64 << remainder) - 1;
    }
}

/// Score a block using integer arithmetic (u32 accumulators, u16 weights).
///
/// `term_mask` names the query dimensions to probe: the caller has already
/// intersected the phase mask with the block's per-block term mask (grid
/// presence transposed once per superblock, plus every non-candidate
/// scoring dimension), so this loop visits only set bits instead of testing
/// all query terms. Accumulates `w_u16 * impact_u8` into u32 — eliminates
/// u8→f32 conversion per posting.
///
/// Blocks store u32 dim_id directly. Binary search on dim_id (always 4 bytes).
///
/// Block data is contiguous — `dim_ptr`, `ps_ptr`, `post_ptr` point into
/// the same ~200-2000 byte region (1-2 pages). Binary search and posting reads
/// touch only this contiguous region.
///
/// Tracks touched slots via a `[u64; 4]` bitmask (works for block_size ≤ 256).
/// Caller uses the bitmask for lazy accumulator zeroing. The accumulator is a
/// fixed 256-slot array so a `u8` slot indexes it without a bounds check.
///
/// Complexity: O(|present_query_dims| × log|block_terms|) per block.
#[inline(always)]
fn score_block_bsearch_int(
    block: AdaptiveBlock<'_>,
    query_by_dim_u16: &[(u32, u16)],
    term_mask: u64,
    narrow_query: bool,
    acc: &mut [u32; 256],
    touched: &mut [u64; 4],
    block_size: usize,
) -> BlockScoreFaults {
    let mut faults = BlockScoreFaults::default();
    let mut remaining = term_mask;
    while remaining != 0 {
        let q = remaining.trailing_zeros() as usize;
        remaining &= remaining - 1;
        let (dim_id, w) = query_by_dim_u16[q];
        let local_term = if narrow_query {
            block.find_dimension_branching(dim_id)
        } else {
            block.find_dimension(dim_id)
        };
        let Some(local_term) = local_term else {
            continue;
        };
        let Some(postings) = block.postings(local_term) else {
            // The term table names this dimension but its posting range is
            // invalid: corrupt block, counted rather than skipped silently.
            faults.corrupt_terms += 1;
            continue;
        };
        match postings {
            AdaptivePostings::Sparse(postings) => {
                for p in postings {
                    let slot = usize::from(p.local_slot);
                    if slot >= block_size {
                        faults.dropped_postings += 1;
                        continue;
                    }
                    // At most MAX_QUERY_TERMS=64 distinct u16×u8
                    // contributions reach one slot, whose worst-case sum
                    // is below u32::MAX. Plain addition keeps this loop
                    // vector/scalar friendly without changing overflow
                    // semantics for any valid query.
                    acc[slot] += w as u32 * p.impact as u32;
                    touched[slot / 64] |= 1u64 << (slot % 64);
                }
            }
            AdaptivePostings::Dense(impacts) => {
                let weight = w as u32;
                for (score, &impact) in acc[..block_size].iter_mut().zip(impacts) {
                    // Same bounded-sum invariant as the sparse loop.
                    *score += weight * impact as u32;
                }
                touch_dense_block(touched, block_size);
            }
        }
    }
    faults
}

// ============================================================================
// Superblock-at-a-time scoring
// ============================================================================

/// Score blocks within a single superblock using integer scoring.
///
/// `prepared` carries the superblock's grid-derived state: `block_start` is
/// the global block ID of its first block, `local_order`/`block_ubs` are
/// indexed by local offset (0..count), and `block_terms` holds one query-term
/// bitset per local block.
///
/// **Two-phase lazy scoring**: When `two_phase_active`, scores only phase1
/// dims first. If the best possible score from phase1 + remaining block UB
/// < threshold, skips phase2 dims entirely (~40-60% of scoring work).
///
/// Integer scoring: accumulates u16×u8 products into u32; the heap threshold
/// is compared in the same integer units (`ThresholdUnits`) and only
/// surviving candidates are dequantized to f32 for the collector.
#[allow(clippy::too_many_arguments)]
fn score_superblock_blocks(
    index: &BmpIndex,
    prepared: &PreparedSuperblock,
    query_by_dim_u16: &[(u32, u16)],
    dequant: f32,
    alpha: f32,
    k: usize,
    predicate: &Option<&dyn Fn(crate::DocId) -> bool>,
    collector: &mut ScoreCollector,
    threshold_units: &mut ThresholdUnits,
    stats: &mut BmpExecutionStats,
    acc: &mut [u32; 256],
    phase1_mask: u64,
    two_phase_active: bool,
    docmap_secs: &mut f64,
) {
    let block_size = index.bmp_block_size as usize;
    let block_start = prepared.block_start;
    let count = prepared.count;
    let local_order = &prepared.local_order[..prepared.local_order_len];
    let local_ubs = &prepared.block_ubs;
    let local_ub_units = &prepared.block_ub_units;
    let phase1_local_ub_units = &prepared.phase1_block_ub_units;
    // The branching binary search wins for narrow probes; decided per phase
    // from the query masks exactly as before the per-block term masks.
    let valid_query_bits = valid_query_bits(query_by_dim_u16.len());
    let narrow_all = query_by_dim_u16.len() <= 8;
    let narrow_phase1 = (phase1_mask & valid_query_bits).count_ones() <= 8;
    let narrow_phase2 = (!phase1_mask & valid_query_bits).count_ones() <= 8;

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
        let local = local_idx as usize;
        if local >= count {
            break;
        }

        let ub = local_ubs[local];
        // Block early termination: UB * alpha < threshold (BMP alpha parameter)
        if collector.len() >= k && ub * alpha < collector.threshold() {
            break;
        }

        let block_id = (block_start + local) as u32;

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

        let Some(block) = index.parse_block(block_id) else {
            // The grid says this block holds query terms but its payload does
            // not parse: corrupt. Counted, and NOT counted as scored.
            stats.corrupt_blocks += 1;
            continue;
        };
        let block_terms = prepared.block_terms[local];
        threshold_units.refresh(collector.threshold());

        let mut touched = [0u64; 4];
        if two_phase_active && collector.len() >= k {
            // Phase 1: Score only the heaviest dims
            stats.absorb(score_block_bsearch_int(
                block,
                query_by_dim_u16,
                block_terms & phase1_mask,
                narrow_phase1,
                acc,
                &mut touched,
                block_size,
            ));

            // Keep the subtraction and addition in integer accumulator
            // units. Subtracting independently rounded f32 bounds can
            // underestimate the remaining score by one ULP near 2^24.
            let max_possible_units = two_phase_upper_bound_units(
                max_touched_acc(acc, &touched),
                local_ub_units[local],
                phase1_local_ub_units[local],
            );
            // Integer image of `max_possible * alpha < threshold`.
            if threshold_units.rejects_phase(max_possible_units) {
                // Skip phase2 — zero touched slots and continue
                zero_touched_acc(acc, &touched);
                stats.blocks_scored += 1;
                continue;
            }

            // Phase 2: Score remaining dims
            stats.absorb(score_block_bsearch_int(
                block,
                query_by_dim_u16,
                block_terms & !phase1_mask,
                narrow_phase2,
                acc,
                &mut touched,
                block_size,
            ));
        } else {
            // Single-phase: score all dims at once
            stats.absorb(score_block_bsearch_int(
                block,
                query_by_dim_u16,
                block_terms,
                narrow_all,
                acc,
                &mut touched,
                block_size,
            ));
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
        let docmap_start = crate::observe::Timer::start();

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
                    stats.invalid_docmap_entries += 1;
                    continue;
                }
                // Integer image of `score < threshold` (exact, see
                // `ThresholdUnits`): a strictly lower score cannot enter
                // regardless of the doc-id/ordinal tie break, so it pays
                // neither the f32 conversion nor the scattered map read.
                // Equal scores still resolve because doc id decides ordering.
                if collector.len() >= k && threshold_units.rejects_score(score_u32) {
                    continue;
                }
                let score = score_u32 as f32 * dequant;
                // Doc-map indirection: BMP reorder permutes only BMP-internal
                // record order, so every candidate pays a scattered lookup
                // into the doc-id map here. Counted per query (metered as
                // summa_bmp_docmap_lookups_*).
                stats.docmap_lookups += 1;
                let (doc_id, ordinal) = index.virtual_to_doc(virtual_id as u32);
                if doc_id == u32::MAX {
                    // A scored slot whose map entry is padding or outside the
                    // segment: the posting exists but its document does not.
                    stats.invalid_docmap_entries += 1;
                    continue;
                }
                if let Some(pred) = predicate
                    && !pred(doc_id)
                {
                    continue;
                }

                collector.insert_with_ordinal(doc_id, score, ordinal);
                threshold_units.refresh(collector.threshold());
            }
        }
        *docmap_secs += docmap_start.secs();

        stats.blocks_scored += 1;
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

#[cfg(feature = "native")]
fn collect_grid_group_ids(sb_order: &[u32], start: usize, limit: usize, groups: &mut Vec<usize>) {
    groups.clear();
    let end = (start + BMP_PREFETCH_SUPERBLOCKS).min(limit);
    for &superblock in &sb_order[start..end] {
        let block = superblock as usize * BMP_SUPERBLOCK_SIZE as usize;
        groups.push(block / GRID_GROUP_CELLS);
    }
    groups.sort_unstable();
    groups.dedup();
}

#[cfg(feature = "native")]
fn prefetch_grid_metadata_window(
    grid: &CompressedGrid,
    weights: &[BmpGridWeight],
    sb_order: &[u32],
    start: usize,
    limit: usize,
    groups: &mut Vec<usize>,
    ranges: &mut Vec<std::ops::Range<usize>>,
) -> crate::Result<(usize, usize)> {
    // Heap/RAM-backed rows (RAM directory, heap pin copy) are always
    // resident; madvise would be a no-op, so skip the range bookkeeping.
    if start >= limit || grid.rows_resident() {
        return Ok((0, 0));
    }
    collect_grid_group_ids(sb_order, start, limit, groups);
    ranges.clear();
    for weight in weights {
        for &group in groups.iter() {
            ranges.push(grid.group_metadata_range(weight.dimension, group)?);
        }
    }
    Ok(grid.prefetch_ranges(ranges))
}

/// Advise the payload pages of every resolved group in the current window.
#[cfg(feature = "native")]
fn prefetch_resolved_grid_payloads(
    grid: &CompressedGrid,
    resolved: &[ResolvedGridGroup],
    ranges: &mut Vec<std::ops::Range<usize>>,
) -> (usize, usize) {
    if grid.rows_resident() {
        return (0, 0);
    }
    ranges.clear();
    ranges.extend(resolved.iter().filter_map(|group| group.payload_range()));
    grid.prefetch_ranges(ranges)
}

/// Resolve every `(candidate dimension, D-grid group)` pair covered by one
/// window of superblocks exactly once.
///
/// Eight consecutive superblocks span at most eight distinct 256-block
/// groups (usually one or two), so `groups` stays tiny and a linear
/// membership scan beats sort+dedup. `resolved` is laid out
/// `[weight_index * groups.len() + slot]`; `slots[position]` gives the
/// group slot of each window position.
fn resolve_grid_window(
    grid: &CompressedGrid,
    weights: &[BmpGridWeight],
    window: &[u32],
    groups: &mut Vec<usize>,
    slots: &mut [u8; BMP_PREFETCH_SUPERBLOCKS],
    resolved: &mut [ResolvedGridGroup],
) -> crate::Result<()> {
    debug_assert!(window.len() <= BMP_PREFETCH_SUPERBLOCKS);
    groups.clear();
    for (position, &superblock) in window.iter().enumerate() {
        let group = superblock as usize * BMP_SUPERBLOCK_SIZE as usize / GRID_GROUP_CELLS;
        let slot = match groups.iter().position(|&known| known == group) {
            Some(slot) => slot,
            None => {
                groups.push(group);
                groups.len() - 1
            }
        };
        slots[position] = slot as u8;
    }
    let stride = groups.len();
    debug_assert!(resolved.len() >= weights.len() * stride);
    for (weight_index, weight) in weights.iter().enumerate() {
        let row = &mut resolved[weight_index * stride..(weight_index + 1) * stride];
        for (entry, &group) in row.iter_mut().zip(groups.iter()) {
            *entry = grid.resolve_group(weight.dimension, group)?;
        }
    }
    Ok(())
}

/// Sort at most eight local block indices by UB without allocating.
fn sort_local_blocks_desc_fixed(
    local_ubs: &[f32],
    count: usize,
    out: &mut [u32; BMP_SUPERBLOCK_SIZE as usize],
) -> usize {
    let mut len = 0usize;
    for (i, &ub) in local_ubs[..count].iter().enumerate() {
        if ub > 0.0 {
            out[len] = i as u32;
            len += 1;
        }
    }
    out[..len].sort_unstable_by(|&a, &b| {
        local_ubs[b as usize]
            .total_cmp(&local_ubs[a as usize])
            .then_with(|| a.cmp(&b))
    });
    len
}

/// Order the superblocks with a positive bound by SBMax descending (ties:
/// lower id first) into `out`.
///
/// Each superblock becomes one `u64` key: the inverted bit pattern of its
/// positive f32 bound (bit order equals numeric order for positive floats,
/// so inverting yields a descending sort) in the high half and the id in
/// the low half. The sort runs on a contiguous integer array instead of
/// comparing through indirect f32 loads and yields exactly the
/// `(bound desc, id asc)` order the LSP/0 selection requires.
///
/// When `visit_limit` is below the number of positive superblocks (the
/// γ-capped local path), only the top `visit_limit` keys are ordered —
/// select, then sort the prefix — because the executor never reads past
/// its cap. `out` still lists every positive superblock so callers can size
/// the visit limit. At production scale (tens of thousands of superblocks
/// per segment) this pass is measurable, so it must not allocate.
fn sort_sb_desc_into(values: &[f32], visit_limit: usize, keys: &mut Vec<u64>, out: &mut Vec<u32>) {
    keys.clear();
    for (i, &v) in values.iter().enumerate() {
        if v > 0.0 {
            keys.push((u64::from(!v.to_bits()) << 32) | i as u64);
        }
    }
    if visit_limit > 0 && visit_limit < keys.len() {
        keys.select_nth_unstable(visit_limit - 1);
        keys[..visit_limit].sort_unstable();
    } else {
        keys.sort_unstable();
    }
    out.clear();
    out.extend(keys.iter().map(|&key| key as u32));
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
fn compute_grid_ubs_int(
    grid: &CompressedGrid,
    kernels: GridKernels,
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
            group.decode_with(kernels, 0, count, decoded);
            accumulate_u8_with(
                kernels,
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
/// pass: full bounds, phase-one bounds, and per-block query-term masks.
///
/// Grid cells are ceiling-quantized upper bounds. Accumulating `u16 × u8` in
/// `u32`, then applying the common dequantizer once, preserves
/// `block_ub >= candidate-query document score` under f32 conversion.
/// Summing already-dequantized terms can round downward and made unpruned
/// exact-mode pruning unsafe near the heap threshold.
///
/// Group payloads come pre-resolved (`resolved[weight_index * group_stride +
/// group_slot]`, see `resolve_grid_window`), so this pass never re-walks the
/// grid metadata. Returns the bitset of local blocks containing at least one
/// candidate dimension.
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_block_ubs_and_presence(
    grid: &CompressedGrid,
    kernels: GridKernels,
    grid_bits: u8,
    int_weights: &[BmpGridWeight],
    resolved: &[ResolvedGridGroup],
    group_stride: usize,
    group_slot: usize,
    non_candidate_mask: u64,
    phase1_mask: u64,
    block_start: usize,
    count: usize,
    units: &mut [u32],
    phase1_units: &mut [u32],
    out: &mut [f32],
    block_terms: &mut [u64; BMP_SUPERBLOCK_SIZE as usize],
    decoded: &mut [u8],
    dequant: f32,
) -> crate::Result<u64> {
    debug_assert!(units.len() >= count);
    debug_assert!(phase1_units.len() >= count);
    debug_assert!(out.len() >= count);
    debug_assert!(decoded.len() >= count);
    debug_assert!(count <= BMP_SUPERBLOCK_SIZE as usize);
    debug_assert!(group_slot < group_stride);
    debug_assert!(resolved.len() >= int_weights.len() * group_stride);
    let within = block_start % GRID_GROUP_CELLS;
    if within + count > GRID_GROUP_CELLS {
        return Err(crate::Error::Corruption(
            "BMP superblock crosses a compressed-grid group boundary".into(),
        ));
    }

    units[..count].fill(0);
    phase1_units[..count].fill(0);
    // Non-candidate scoring dimensions have no grid row to consult, so every
    // block probes them; grid-present candidate dimensions are OR-ed in below.
    block_terms.fill(non_candidate_mask);

    let cell_scale = crate::segment::bmp_grid::block_grid_scale(grid_bits);
    let use_phase1 = phase1_mask != u64::MAX;
    let mut blocks_with_query_terms = 0u64;
    for (
        weight_index,
        &BmpGridWeight {
            weight,
            query_index,
            ..
        },
    ) in int_weights.iter().enumerate()
    {
        let group = grid.resolved(resolved[weight_index * group_stride + group_slot]);
        if group.width() == 0 {
            continue;
        }
        group.decode_u4_packed_with(kernels, within, count, decoded);
        let multiplier = cell_scale * u32::from(weight);
        let phase1 = (use_phase1 && phase1_mask & (1u64 << query_index) != 0)
            .then_some(&mut phase1_units[..count]);
        let presence = accumulate_packed_u4_with(
            kernels,
            &decoded[..count.div_ceil(2)],
            count,
            multiplier,
            &mut units[..count],
            phase1,
        );
        blocks_with_query_terms |= presence;
        // Transpose the per-dimension presence bitset into per-block term
        // masks so block scoring iterates only the terms a block contains.
        let term_bit = 1u64 << query_index;
        let mut bits = presence;
        while bits != 0 {
            let local = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            block_terms[local] |= term_bit;
        }
    }
    for (bound, &integer_units) in out[..count].iter_mut().zip(&units[..count]) {
        *bound = integer_units as f32 * dequant;
    }
    Ok(blocks_with_query_terms)
}

/// Quantized candidate scoring over selected forward vectors, preserving the
/// full query weights and additive duplicate dimensions.
fn score_forward_units(
    vector: crate::segment::bmp_forward::ForwardVector<'_>,
    query_by_dim_u16: &[(u32, u16)],
) -> crate::Result<u32> {
    let mut query = query_by_dim_u16;
    // Keep the fallible accumulator compact in the per-coordinate fold. Carrying
    // the full Error enum here generates stack copies on every decoded lane.
    vector
        .fold(Some(0u32), |units, dimension, impact| {
            let mut units = units?;
            while query.first().is_some_and(|&(dim, _)| dim < dimension) {
                query = &query[1..];
            }
            // Retain every already-quantized query contribution, including repeated
            // dimensions. Keep the cursor here for duplicate document entries too.
            for &(_, weight) in query.iter().take_while(|&&(dim, _)| dim == dimension) {
                units = units.checked_add(u32::from(impact) * u32::from(weight))?;
            }
            Some(units)
        })
        .ok_or_else(|| crate::Error::Query("BMP candidate score overflow".into()))
}

/// Request-owned preparation for one immutable scoring component. Retain only
/// one dimension configuration; a differing segment replaces the preparation.
#[derive(Default)]
pub(super) struct CandidateBmpPreparation {
    prepared: Option<(u32, Option<PreparedBmpQuery>)>,
}

/// Exact probes share retrieval's integer quantization/dequantization. Targets
/// are sorted forward rows or real physical slots from the candidate reader.
#[cfg(test)]
pub(crate) fn score_bmp_candidates(
    index: &BmpIndex,
    terms: &[(u32, f32)],
    targets: &[u32],
) -> crate::Result<Vec<f32>> {
    CandidateBmpPreparation::default().score(index, terms, targets)
}

impl CandidateBmpPreparation {
    pub(super) fn score(
        &mut self,
        index: &BmpIndex,
        terms: &[(u32, f32)],
        targets: &[u32],
    ) -> crate::Result<Vec<f32>> {
        use crate::segment::bmp_adaptive::AdaptivePostings;
        let mut scores = vec![0.0; targets.len()];
        if self
            .prepared
            .as_ref()
            .is_none_or(|(dims, _)| *dims != index.dims())
        {
            self.prepared = Some((index.dims(), prepare_bmp_query(index.dims(), terms, terms)?));
        }
        let Some(prepared) = self.prepared.as_ref().and_then(|(_, query)| query.as_ref()) else {
            return Ok(scores);
        };
        let dequant = prepared.dequant_for(index)?;
        if let Some(forward) = index.forward() {
            for (score, &target) in scores.iter_mut().zip(targets) {
                let vector = forward.vector_for_scoring(target)?;
                let units = score_forward_units(vector, &prepared.query_by_dim_u16)?;
                *score = units as f32 * dequant;
            }
            return Ok(scores);
        }
        let mut start = 0;
        while start < targets.len() {
            let block_id = targets[start] / index.bmp_block_size;
            let mut end = start + 1;
            while end < targets.len() && targets[end] / index.bmp_block_size == block_id {
                end += 1;
            }
            // Missing/empty blocks contain no nonzero terms. Real-slot identity is
            // validated by the reader before this call.
            let (byte_start, byte_end) = index.block_data_range(block_id);
            if byte_start == byte_end {
                start = end;
                continue;
            }
            let block = index.parse_block(block_id).ok_or_else(|| {
                crate::Error::Corruption(format!("cannot decode candidate BMP block {block_id}"))
            })?;
            let mut units = [0u32; 256];
            for &(dimension, weight) in &prepared.query_by_dim_u16 {
                let Some(term) = block.find_dimension(dimension) else {
                    continue;
                };
                let postings = block.postings(term).ok_or_else(|| {
                    crate::Error::Corruption(format!(
                        "cannot decode candidate BMP dimension {dimension}"
                    ))
                })?;
                match postings {
                    AdaptivePostings::Dense(impacts) => {
                        for &target in &targets[start..end] {
                            let slot = (target % index.bmp_block_size) as usize;
                            units[slot] = units[slot]
                                .checked_add(u32::from(impacts[slot]) * u32::from(weight))
                                .ok_or_else(|| {
                                    crate::Error::Query("BMP candidate score overflow".into())
                                })?;
                        }
                    }
                    AdaptivePostings::Sparse(postings) => {
                        for &target in &targets[start..end] {
                            let slot = (target % index.bmp_block_size) as u8;
                            let start = postings.partition_point(|p| p.local_slot < slot);
                            for posting in postings[start..]
                                .iter()
                                .take_while(|p| p.local_slot == slot)
                            {
                                units[slot as usize] = units[slot as usize]
                                    .checked_add(u32::from(posting.impact) * u32::from(weight))
                                    .ok_or_else(|| {
                                        crate::Error::Query("BMP candidate score overflow".into())
                                    })?;
                            }
                        }
                    }
                }
            }
            for i in start..end {
                scores[i] = units[(targets[i] % index.bmp_block_size) as usize] as f32 * dequant;
            }
            start = end;
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::{
        BmpGridWeight, compute_block_ubs_and_presence, min_units_reaching, recommended_lsp_gamma,
        sort_sb_desc_into, two_phase_upper_bound_units,
    };
    use crate::directories::OwnedBytes;
    use crate::segment::BMP_SUPERBLOCK_SIZE;
    use crate::segment::bmp_grid::{
        CompressedGrid, CompressedGridLayout, GRID_GROUP_CELLS, GridKernels, ResolvedGridGroup,
        bit_width, pack_group,
    };

    fn forward_score(entries: &[(u32, u8)], query: &[(u32, u16)]) -> crate::Result<u32> {
        use crate::segment::bmp_forward::{BmpForward, RowWriter, write_directory};
        use crate::segment::logical_address::LogicalUnit;

        let mut bytes = Vec::new();
        let mut writer = RowWriter::default();
        for &(dimension, impact) in entries {
            writer.push(dimension, impact, &mut bytes).unwrap();
        }
        writer.finish(&mut bytes).unwrap();
        let payload_bytes = bytes.len() as u64;
        write_directory(
            &mut bytes,
            [Ok((LogicalUnit { doc: 0, ordinal: 0 }, 0))],
            1,
            payload_bytes,
        )
        .unwrap();
        let forward = BmpForward::parse(OwnedBytes::new(bytes), 1, 1, u32::MAX).unwrap();
        super::score_forward_units(forward.vector_for_scoring(0).unwrap(), query)
    }

    #[test]
    fn forward_score_preserves_duplicate_matches_across_packets_and_encodings() {
        // Exercise fixed-width packets, gap groups, packet boundaries and tails.
        for (base, gap) in [(0, 1), (65_530, 3), (100_000, 70_000), (1 << 25, 30_000)] {
            let entries: Vec<_> = (0..273)
                .map(|i| (base + (i / 3) * gap, (i % 255 + 1) as u8))
                .collect();
            let query = [
                (base, 1),
                (base + 42 * gap, 127),
                (base + 42 * gap, 256),
                (base + 90 * gap, u16::MAX),
            ];
            let expected: u32 = entries
                .iter()
                .map(|&(dim, impact)| {
                    query
                        .iter()
                        .filter(|&&(q, _)| q == dim)
                        .map(|&(_, weight)| u32::from(impact) * u32::from(weight))
                        .sum::<u32>()
                })
                .sum();
            assert_eq!(forward_score(&entries, &query).unwrap(), expected);
            assert_eq!(forward_score(&entries, &[]).unwrap(), 0);
            assert_eq!(forward_score(&entries, &[(u32::MAX, 1)]).unwrap(), 0);
        }
    }

    #[test]
    fn forward_score_accepts_u32_max_and_keeps_overflow_sticky_across_packets() {
        // (257 * 255 + 2) * 65535 is u32::MAX; the next match must error.
        let mut entries = vec![(7, 255); 257];
        entries.push((7, 2));
        assert_eq!(forward_score(&entries, &[(7, u16::MAX)]).unwrap(), u32::MAX);
        entries.push((7, 1));
        // Later unmatched packets must not turn an overflow into a valid score.
        entries.extend([(8, 255); 257]);
        assert!(matches!(
            forward_score(&entries, &[(7, u16::MAX)]),
            Err(crate::Error::Query(message)) if message == "BMP candidate score overflow"
        ));
    }

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

    /// Resolve group 0 of every weight's row, as `resolve_grid_window` does
    /// for a one-group window.
    fn resolve_group_zero(
        grid: &CompressedGrid,
        weights: &[BmpGridWeight],
    ) -> Vec<ResolvedGridGroup> {
        weights
            .iter()
            .map(|weight| grid.resolve_group(weight.dimension, 0).unwrap())
            .collect()
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
        let resolved = resolve_group_zero(&grid, &weights);
        let dequant = 0.123_456_7f32;
        let mut units = [0u32; 1];
        let mut bounds = [0.0f32; 1];

        let mut phase1_units = [0u32; 1];
        let mut block_terms = [0u64; BMP_SUPERBLOCK_SIZE as usize];
        let mut decoded = [0u8; GRID_GROUP_CELLS];
        compute_block_ubs_and_presence(
            &grid,
            GridKernels::detect(),
            4,
            &weights,
            &resolved,
            1,
            0,
            0,
            u64::MAX,
            0,
            1,
            &mut units,
            &mut phase1_units,
            &mut bounds,
            &mut block_terms,
            &mut decoded,
            dequant,
        )
        .unwrap();

        let score_units = 160u32 * 255 * dimensions as u32;
        let score = score_units as f32 * dequant;
        assert!(bounds[0] >= score);
        assert_eq!(units[0], score_units);
        assert_eq!(
            block_terms[0],
            u64::MAX,
            "every dimension is present in block 0"
        );
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
        let weights = [BmpGridWeight {
            dimension: 0,
            weight: 123,
            query_index: 0,
        }];
        let resolved = resolve_group_zero(&grid, &weights);
        for block_start in (0..blocks).step_by(BMP_SUPERBLOCK_SIZE as usize) {
            let mut units = [0u32; BMP_SUPERBLOCK_SIZE as usize];
            let mut phase1_units = [0u32; BMP_SUPERBLOCK_SIZE as usize];
            let mut bounds = [0.0f32; BMP_SUPERBLOCK_SIZE as usize];
            let mut block_terms = [0u64; BMP_SUPERBLOCK_SIZE as usize];
            let mut decoded = [0u8; GRID_GROUP_CELLS];

            let present = compute_block_ubs_and_presence(
                &grid,
                GridKernels::detect(),
                4,
                &weights,
                &resolved,
                1,
                0,
                0,
                u64::MAX,
                block_start,
                BMP_SUPERBLOCK_SIZE as usize,
                &mut units,
                &mut phase1_units,
                &mut bounds,
                &mut block_terms,
                &mut decoded,
                1.0,
            )
            .unwrap();

            for local in 0..BMP_SUPERBLOCK_SIZE as usize {
                let block = block_start + local;
                assert_eq!(units[local], (block % 16) as u32 * 17 * 123);
                assert_eq!(bounds[local], units[local] as f32);
                let expected_present = block % 16 != 0;
                assert_eq!((present >> local) & 1 == 1, expected_present);
                assert_eq!(block_terms[local], u64::from(expected_present));
            }
        }
    }

    #[test]
    fn block_term_masks_transpose_presence_and_keep_non_candidate_terms() {
        // Three candidate rows over eight blocks with distinct patterns; query
        // indices are deliberately non-contiguous (1, 3, 5) and index 4 is a
        // non-candidate scoring term that every block must probe.
        let rows = vec![
            vec![1, 0, 1, 0, 1, 0, 1, 0],
            vec![0, 0, 0, 0, 2, 2, 2, 2],
            vec![0, 0, 0, 0, 0, 0, 0, 3],
        ];
        let grid = test_grid(&rows, 8);
        let weights: Vec<BmpGridWeight> = [(0usize, 1usize), (1, 3), (2, 5)]
            .into_iter()
            .map(|(dimension, query_index)| BmpGridWeight {
                dimension,
                weight: 10,
                query_index,
            })
            .collect();
        let resolved = resolve_group_zero(&grid, &weights);
        let non_candidate_mask = 1u64 << 4;
        let mut units = [0u32; 8];
        let mut phase1_units = [0u32; 8];
        let mut bounds = [0.0f32; 8];
        let mut block_terms = [u64::MAX; 8];
        let mut decoded = [0u8; GRID_GROUP_CELLS];
        let present = compute_block_ubs_and_presence(
            &grid,
            GridKernels::detect(),
            4,
            &weights,
            &resolved,
            1,
            0,
            non_candidate_mask,
            u64::MAX,
            0,
            8,
            &mut units,
            &mut phase1_units,
            &mut bounds,
            &mut block_terms,
            &mut decoded,
            1.0,
        )
        .unwrap();

        assert_eq!(present, 0b1111_0101);
        let expected = [
            1u64 << 1,
            0,
            1 << 1,
            0,
            (1 << 1) | (1 << 3),
            1 << 3,
            (1 << 1) | (1 << 3),
            (1 << 3) | (1 << 5),
        ];
        for (local, &mask) in expected.iter().enumerate() {
            assert_eq!(
                block_terms[local],
                mask | non_candidate_mask,
                "block {local}: stale mask bits must be overwritten, non-candidate kept"
            );
        }
    }

    #[test]
    fn integer_threshold_matches_f32_compare_at_rounding_boundaries() {
        // `min_units_reaching` must return the exact smallest m such that
        // `u < m  <=>  f(u) < threshold`, including where u32 -> f32 rounds
        // (above 2^24) and where the estimate `threshold / scale` is off.
        fn check(threshold: f32, dequant: f32, alpha: f32) {
            let f_score = |u: u32| u as f32 * dequant;
            let f_phase = |u: u32| (u as f32 * dequant) * alpha;
            for (f, scale) in [
                (&f_score as &dyn Fn(u32) -> f32, f64::from(dequant)),
                (&f_phase, f64::from(dequant) * f64::from(alpha)),
            ] {
                let m = min_units_reaching(threshold, scale, f);
                assert!(m <= 1 << 32);
                if m < 1 << 32 {
                    assert!(f(m as u32) >= threshold, "f({m}) < {threshold}");
                }
                if m > 0 {
                    assert!(f((m - 1) as u32) < threshold, "f({}) >= {threshold}", m - 1);
                }
                let lo = m.saturating_sub(4);
                let hi = (m + 4).min(u64::from(u32::MAX));
                for u in lo..=hi {
                    assert_eq!(
                        u < m,
                        f(u as u32) < threshold,
                        "u={u} m={m} threshold={threshold} dequant={dequant} alpha={alpha}"
                    );
                }
            }
        }

        let dequants = [1.0f32, 0.123_456_7, 3.7e-5, 1e-9, 250.0];
        let alphas = [1.0f32, 0.85, 0.5, 0.01];
        // Boundaries: exact images of representable and non-representable
        // unit counts around 2^24, tiny/huge thresholds, zero and negative.
        let unit_probes = [
            0u32,
            1,
            2,
            255,
            16_777_215,
            16_777_216,
            16_777_217,
            16_777_219,
            33_554_433,
            u32::MAX / 2,
            u32::MAX - 1,
            u32::MAX,
        ];
        for &dequant in &dequants {
            for &alpha in &alphas {
                for &units in &unit_probes {
                    let image = units as f32 * dequant;
                    check(image, dequant, alpha);
                    check(image * alpha, dequant, alpha);
                    check(f32::from_bits(image.to_bits() + 1), dequant, alpha);
                    if image > 0.0 {
                        check(f32::from_bits(image.to_bits() - 1), dequant, alpha);
                    }
                }
                check(0.0, dequant, alpha);
                check(-1.0, dequant, alpha);
                check(f32::MAX, dequant, alpha);
                check(f32::INFINITY, dequant, alpha);
            }
        }
        // NaN rejects nothing, exactly like `x < NaN`.
        assert_eq!(min_units_reaching(f32::NAN, 1.0, |u| u as f32), 0);
        // A pseudo-random sweep with a deliberately wrong scale estimate.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..2000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let units = (state >> 32) as u32;
            let dequant = dequants[(state >> 8) as usize % dequants.len()];
            let threshold = units as f32 * dequant;
            let f = |u: u32| u as f32 * dequant;
            let m = min_units_reaching(threshold, f64::from(dequant) * 1.5, f);
            assert!(f(m as u32) >= threshold);
            assert!(m == 0 || f((m - 1) as u32) < threshold);
        }
    }

    #[test]
    fn lsp_gamma_schedule_is_depth_aware_with_reliability_floors() {
        // Floors chosen from the 2026-08-23 production incident: γ=500
        // dropped an exact-title query's true top-1 document; γ=2000 was the
        // measured fix for broad topical queries; floors carry 1.5-2x margin
        // and selection cost is ~flat in γ up to ~5000.
        assert_eq!(recommended_lsp_gamma(0), 3000);
        assert_eq!(recommended_lsp_gamma(10), 3000);
        assert_eq!(recommended_lsp_gamma(100), 3000);
        assert_eq!(recommended_lsp_gamma(101), 4000);
        assert_eq!(recommended_lsp_gamma(1000), 4000);
        assert_eq!(recommended_lsp_gamma(1001), 4000);
        assert_eq!(recommended_lsp_gamma(4800), 4800);
    }

    #[test]
    fn lsp_orders_strictly_by_superblock_maximum() {
        let bounds = [4.0, 9.0, 9.0, 1.0, 7.0];
        let mut keys = Vec::new();
        let mut order = Vec::new();
        sort_sb_desc_into(&bounds, usize::MAX, &mut keys, &mut order);
        assert_eq!(order, vec![1, 2, 4, 0, 3]);
    }

    /// The reference order: comparison sort on f32 with id tie-break.
    fn reference_sb_order(values: &[f32]) -> Vec<u32> {
        let mut out: Vec<u32> = (0..values.len() as u32)
            .filter(|&i| values[i as usize] > 0.0)
            .collect();
        out.sort_unstable_by(|&a, &b| {
            values[b as usize]
                .total_cmp(&values[a as usize])
                .then_with(|| a.cmp(&b))
        });
        out
    }

    #[test]
    fn integer_key_superblock_sort_matches_f32_comparison_sort_with_gamma_prefix() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for &count in &[1usize, 7, 64, 1000, 5000] {
            // Heavy ties (values drawn from a small set), zeros, and large
            // magnitudes so bit-pattern ordering is exercised end to end.
            let values: Vec<f32> = (0..count)
                .map(|_| match next() % 5 {
                    0 => 0.0,
                    1 => (next() % 7) as f32,
                    2 => (next() % 100) as f32 * 1e-6,
                    3 => (next() as f32) * 1e3,
                    _ => f32::from_bits(0x3f80_0000 + next() % 16),
                })
                .collect();
            let expected = reference_sb_order(&values);
            let mut keys = Vec::new();
            let mut order = Vec::new();
            sort_sb_desc_into(&values, usize::MAX, &mut keys, &mut order);
            assert_eq!(order, expected, "full order, n={count}");
            for gamma in [
                1usize,
                2,
                3,
                expected.len() / 2,
                expected.len(),
                expected.len() + 5,
            ] {
                sort_sb_desc_into(&values, gamma, &mut keys, &mut order);
                assert_eq!(
                    order.len(),
                    expected.len(),
                    "gamma={gamma} keeps every positive SB"
                );
                let prefix = gamma.min(expected.len());
                assert_eq!(
                    &order[..prefix],
                    &expected[..prefix],
                    "gamma={gamma} prefix, n={count}"
                );
            }
        }
    }

    /// `cargo test -p summa-core --release -- --ignored --nocapture sort_sb_desc_micro_bench`
    ///
    /// The bench harness cannot reach this private function, and the 200k-doc
    /// hot-path benchmark only has 98 superblocks; production segments have
    /// tens of thousands. Prints old (indirect f32 comparison sort) vs new
    /// (packed integer keys; γ-capped prefix) timings.
    #[test]
    #[ignore]
    fn sort_sb_desc_micro_bench() {
        let count = 40_000usize;
        let mut state = 42u64;
        let values: Vec<f32> = (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) % 100_000) as f32 * 0.001
            })
            .collect();
        let rounds = 200;
        let mut keys = Vec::new();
        let mut order = Vec::new();
        let start = std::time::Instant::now();
        for _ in 0..rounds {
            order = reference_sb_order(&values);
        }
        let old = start.elapsed().as_secs_f64() / rounds as f64;
        let start = std::time::Instant::now();
        for _ in 0..rounds {
            sort_sb_desc_into(&values, usize::MAX, &mut keys, &mut order);
        }
        let full = start.elapsed().as_secs_f64() / rounds as f64;
        let start = std::time::Instant::now();
        for _ in 0..rounds {
            sort_sb_desc_into(&values, 3000, &mut keys, &mut order);
        }
        let capped = start.elapsed().as_secs_f64() / rounds as f64;
        println!(
            "sort_sb_desc n={count}: f32-compare {:.1}us, integer-key full {:.1}us, integer-key gamma=3000 {:.1}us",
            old * 1e6,
            full * 1e6,
            capped * 1e6
        );
        assert_eq!(order.len(), reference_sb_order(&values).len());
    }
}
