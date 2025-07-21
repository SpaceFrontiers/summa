//! Recursive Graph Bisection (BP) for BMP document ordering.
//!
//! Based on Dhulipala et al. (KDD 2016) and Mackenzie et al. — the same
//! algorithm used in Lucene and PISA for document reordering.
//!
//! Directly optimizes log-gap cost: docs sharing dimensions end up in the
//! same BMP blocks, producing tight upper bounds and effective pruning.
//!
//! Memory is budgeted: CSR terms plus roughly 32 bytes/document of graph
//! scratch, with lazily initialized direct-index term-degree arrays.

#[cfg(feature = "native")]
use rayon::prelude::*;
const TERM_DEGREE_VALUE_BYTES: usize = std::mem::size_of::<[u32; 2]>();
const CANDIDATE_ENTRY_BYTES: usize = std::mem::size_of::<(usize, u32)>();
// Radix selection scans the gain array four times and parallel degree updates
// require private vocabulary arrays. Quickselect wins decisively below this
// scale; above it, bounded memory and worker utilization dominate.
const PARALLEL_BP_MIN_ENTITIES: usize = 1_048_576;
const MIN_RELATIVE_OBJECTIVE_IMPROVEMENT: f64 = 1e-6;
const MIN_OBJECTIVE_ITERATIONS: usize = 4;
const OBJECTIVE_STALL_ITERATIONS: usize = 2;

fn term_degree_bytes(num_terms: usize) -> usize {
    let bitmap_words = num_terms.div_ceil(64);
    num_terms
        .saturating_mul(TERM_DEGREE_VALUE_BYTES)
        .saturating_add(bitmap_words.saturating_mul(std::mem::size_of::<u64>()))
        // A reusable lane remembers only bitmap words touched by its previous
        // partition, so reset is proportional to active terms rather than the
        // complete vocabulary.
        .saturating_add(bitmap_words.saturating_mul(std::mem::size_of::<u32>()))
}

/// Count a dense vocabulary without a cache-line-contended atomic increment
/// for every posting. Workers own private tables, and the number of tables is
/// capped by the caller's remaining memory budget. A single plain table is
/// used when the budget cannot afford useful parallelism.
#[cfg(feature = "native")]
trait FrequencyParallelSafe: Sync {}
#[cfg(feature = "native")]
impl<T: Sync + ?Sized> FrequencyParallelSafe for T {}

#[cfg(not(feature = "native"))]
trait FrequencyParallelSafe {}
#[cfg(not(feature = "native"))]
impl<T: ?Sized> FrequencyParallelSafe for T {}

fn count_frequencies_bounded<T: FrequencyParallelSafe>(
    items: &[T],
    num_terms: usize,
    available_bytes: usize,
    count_item: impl Fn(&T, &mut [u32]) + FrequencyParallelSafe,
) -> Option<Vec<u32>> {
    if num_terms == 0 {
        return Some(Vec::new());
    }
    let table_bytes = num_terms
        .checked_mul(std::mem::size_of::<u32>())?
        .checked_add(std::mem::size_of::<Vec<u32>>())?;
    let affordable_tables = available_bytes.checked_div(table_bytes)?;
    if affordable_tables == 0 {
        return None;
    }

    #[cfg(feature = "native")]
    {
        let lanes = affordable_tables
            .min(rayon::current_num_threads().max(1))
            .min(items.len().max(1));
        if lanes > 1 {
            let chunk_len = items.len().div_ceil(lanes);
            return items
                .par_chunks(chunk_len)
                .map(|chunk| {
                    let mut counts = vec![0u32; num_terms];
                    for item in chunk {
                        count_item(item, &mut counts);
                    }
                    counts
                })
                .reduce_with(|mut left, right| {
                    for (total, count) in left.iter_mut().zip(right) {
                        *total = total.saturating_add(count);
                    }
                    left
                });
        }
    }

    let mut counts = vec![0u32; num_terms];
    for item in items {
        count_item(item, &mut counts);
    }
    Some(counts)
}

/// Retain the lowest-frequency eligible dimensions while the frequency table
/// is live. Both record- and block-level builders use the same bounded policy.
fn select_frequency_candidates(
    frequencies: &[u32],
    min_frequency: usize,
    max_frequency: usize,
    candidate_budget_bytes: usize,
) -> (Vec<(u32, usize)>, bool) {
    let eligible_count = frequencies
        .iter()
        .filter(|&&frequency| {
            let frequency = frequency as usize;
            frequency >= min_frequency && frequency <= max_frequency
        })
        .count();
    let capacity = candidate_budget_bytes
        .checked_div(CANDIDATE_ENTRY_BYTES)
        .unwrap_or(0)
        .min(eligible_count);
    let mut candidates = std::collections::BinaryHeap::with_capacity(capacity);
    for (term_id, &frequency) in frequencies.iter().enumerate() {
        let frequency = frequency as usize;
        if frequency < min_frequency || frequency > max_frequency {
            continue;
        }
        let candidate = (frequency, term_id as u32);
        if candidates.len() < capacity {
            candidates.push(candidate);
        } else if capacity > 0 && candidate < *candidates.peek().unwrap() {
            candidates.pop();
            candidates.push(candidate);
        }
    }
    let selected = candidates
        .into_vec()
        .into_iter()
        .map(|(frequency, term_id)| (term_id, frequency))
        .collect();
    (selected, capacity < eligible_count)
}

struct CandidateFit {
    estimated_bytes: usize,
    retained_postings: usize,
    dropped: usize,
}

/// Apply the shared forward-index memory model, preferring low-frequency
/// dimensions because they add the least CSR storage and the strongest
/// clustering signal.
fn fit_candidates_to_budget(
    candidates: &mut Vec<(u32, usize)>,
    fixed_bytes: usize,
    memory_budget_bytes: usize,
) -> CandidateFit {
    let total_postings = candidates.iter().fold(0usize, |total, (_, frequency)| {
        total.saturating_add(*frequency)
    });
    let estimated_bytes = total_postings
        .saturating_mul(std::mem::size_of::<u32>())
        .saturating_add(fixed_bytes)
        .saturating_add(candidates.len().saturating_mul(CANDIDATE_ENTRY_BYTES))
        .saturating_add(term_degree_bytes(candidates.len()));
    if estimated_bytes <= memory_budget_bytes || candidates.is_empty() {
        return CandidateFit {
            estimated_bytes,
            retained_postings: total_postings,
            dropped: 0,
        };
    }

    candidates.sort_by_key(|&(_, frequency)| frequency);
    let mut used_bytes = fixed_bytes;
    let mut retained_postings = 0usize;
    let mut keep = 0usize;
    for &(_, frequency) in candidates.iter() {
        let term_bytes = frequency
            .saturating_mul(std::mem::size_of::<u32>())
            .saturating_add(TERM_DEGREE_VALUE_BYTES + 1)
            .saturating_add(CANDIDATE_ENTRY_BYTES);
        if term_bytes > memory_budget_bytes.saturating_sub(used_bytes) {
            break;
        }
        used_bytes = used_bytes.saturating_add(term_bytes);
        retained_postings = retained_postings.saturating_add(frequency);
        keep += 1;
    }
    let dropped = candidates.len() - keep;
    candidates.truncate(keep);
    // BinaryHeap::into_vec retains its original capacity. Release dropped
    // candidates before allocating the remap, CSR, and graph scratch.
    candidates.shrink_to_fit();
    CandidateFit {
        estimated_bytes,
        retained_postings,
        dropped,
    }
}

fn parallel_bisect_lanes(
    memory_budget_bytes: usize,
    non_degree_bytes: usize,
    num_terms: usize,
) -> usize {
    let per_node = term_degree_bytes(num_terms).max(1);
    let affordable_nodes = memory_budget_bytes
        .saturating_sub(non_degree_bytes)
        .checked_div(per_node)
        .unwrap_or(0)
        .max(1);
    #[cfg(feature = "native")]
    let worker_limit = rayon::current_num_threads().max(1);
    #[cfg(not(feature = "native"))]
    let worker_limit = 1usize;
    affordable_nodes.min(worker_limit)
}

/// Per-partition left/right term degrees with direct compact-term indexing.
///
/// Recursive BP used to zero two `num_terms`-long vectors at every node. At
/// 100k vocabulary terms and hundreds of thousands of fine partitions, that
/// turns into a large amount of memory traffic unrelated to actual postings.
/// A one-bit initialization map lets us retain array-speed lookups while only
/// touching degree slots present in the current partition.
struct TermDegrees {
    values: Vec<std::mem::MaybeUninit<[u32; 2]>>,
    initialized: Vec<u64>,
    touched_words: Vec<u32>,
}

impl TermDegrees {
    fn new(num_terms: usize) -> Self {
        let bitmap_words = num_terms.div_ceil(64);
        let mut values = Vec::with_capacity(num_terms);
        values.resize_with(num_terms, std::mem::MaybeUninit::uninit);
        Self {
            values,
            initialized: vec![0; bitmap_words],
            touched_words: Vec::with_capacity(bitmap_words),
        }
    }

    /// Reuse this vocabulary-sized lane for another partition without
    /// clearing the complete initialization bitmap.
    fn reset(&mut self) {
        for word in self.touched_words.drain(..) {
            self.initialized[word as usize] = 0;
        }
    }

    fn sort_touched_words(&mut self) {
        self.touched_words.sort_unstable();
    }

    #[inline]
    fn entry_mut(&mut self, term: usize) -> &mut [u32; 2] {
        let word = term / 64;
        let mask = 1u64 << (term % 64);
        if self.initialized[word] & mask == 0 {
            if self.initialized[word] == 0 {
                self.touched_words.push(word as u32);
            }
            self.values[term].write([0, 0]);
            self.initialized[word] |= mask;
        }
        // SAFETY: the bit above is set only after writing this exact slot.
        unsafe { self.values[term].assume_init_mut() }
    }

    #[inline]
    fn get(&self, term: usize) -> [u32; 2] {
        let word = term / 64;
        let mask = 1u64 << (term % 64);
        if self.initialized[word] & mask == 0 {
            return [0, 0];
        }
        // SAFETY: an initialized bit is published only after the slot write;
        // scoring reads degrees after construction, with no concurrent writes.
        unsafe { *self.values[term].assume_init_ref() }
    }

    fn merge_from(&mut self, other: &Self) {
        for &word_idx in &other.touched_words {
            let word_idx = word_idx as usize;
            let mut pending = other.initialized[word_idx];
            while pending != 0 {
                let bit = pending.trailing_zeros() as usize;
                let term = word_idx * 64 + bit;
                // SAFETY: `pending` is derived from the initialized bitmap.
                let [left, right] = unsafe { *other.values[term].assume_init_ref() };
                let entry = self.entry_mut(term);
                entry[0] += left;
                entry[1] += right;
                pending &= pending - 1;
            }
        }
    }

    /// Apply directional movement counts accumulated in a reusable lane.
    ///
    /// Each entry is `[right_to_left, left_to_right]`. Keeping two unsigned
    /// counters lets partition workers reuse the exact same storage as degree
    /// construction instead of allocating a separate signed delta table.
    fn apply_moves_to(&self, degrees: &mut Self) {
        for &word_idx in &self.touched_words {
            let word_idx = word_idx as usize;
            let mut pending = self.initialized[word_idx];
            while pending != 0 {
                let bit = pending.trailing_zeros() as usize;
                let term = word_idx * 64 + bit;
                // SAFETY: `pending` is derived from the initialized bitmap.
                let [right_to_left, left_to_right] =
                    unsafe { *self.values[term].assume_init_ref() };
                debug_assert!(
                    degrees.initialized[word_idx] & (1u64 << bit) != 0,
                    "a moved term must already exist in the partition degrees"
                );
                let degree = degrees.entry_mut(term);
                let new_left =
                    i64::from(degree[0]) + i64::from(right_to_left) - i64::from(left_to_right);
                let new_right =
                    i64::from(degree[1]) + i64::from(left_to_right) - i64::from(right_to_left);
                debug_assert!(new_left >= 0 && new_right >= 0);
                debug_assert!(new_left <= i64::from(u32::MAX));
                debug_assert!(new_right <= i64::from(u32::MAX));
                degree[0] = new_left as u32;
                degree[1] = new_right as u32;
                pending &= pending - 1;
            }
        }
    }

    /// Exact bisection objective optimized by the BP gain approximation.
    ///
    /// This returns the negative assignment-dependent BiMLogA bisection cost
    /// from Dhulipala et al., so a larger value means lower cost. Keeping the
    /// partition-size term matters for odd-sized partitions, whose halves
    /// differ by one entity.
    fn bisection_objective(&self, left_size: usize, right_size: usize, log_table: &[f32]) -> f64 {
        let mut objective = 0.0f64;
        let side_log = [
            fast_log2_lookup(left_size, log_table) as f64,
            fast_log2_lookup(right_size, log_table) as f64,
        ];
        for &word_idx in &self.touched_words {
            let word_idx = word_idx as usize;
            let mut pending = self.initialized[word_idx];
            while pending != 0 {
                let bit = pending.trailing_zeros() as usize;
                let term = word_idx * 64 + bit;
                // SAFETY: `pending` is derived from the initialized bitmap.
                let [left, right] = unsafe { *self.values[term].assume_init_ref() };
                for (side, count) in [left, right].into_iter().enumerate() {
                    if count > 0 {
                        objective += count as f64
                            * (fast_log2_lookup(count as usize + 1, log_table) as f64
                                - side_log[side]);
                    }
                }
                pending &= pending - 1;
            }
        }
        objective
    }
}

// ── Forward index (CSR) ──────────────────────────────────────────────────

/// Forward index in CSR format: doc `d`'s terms are `terms[offsets[d]..offsets[d+1]]`.
///
/// Term IDs are remapped to compact range `0..num_terms` for flat-array degree tracking.
pub(crate) struct ForwardIndex {
    terms: Vec<u32>,
    /// u64, not u32: a 58M-doc / ~85-dims-per-doc reorder pass carries ~4.9B
    /// postings — u32 prefix sums wrapped and the CSR carving panicked
    /// (prod 2026-07-14, "mid > len"). The old 8 GB memory budget masked it
    /// by dropping dims below the u32 limit.
    offsets: Vec<u64>,
    pub num_terms: usize,
    /// Exact number of simultaneous vocabulary-sized degree-array lanes
    /// allowed by the memory budget. Recursion divides this allowance between
    /// children without rounding non-power-of-two worker pools down.
    parallel_bisect_lanes: usize,
    /// True when the configured memory limit forced graph signal to be
    /// discarded. Callers must not report the resulting order as fully
    /// converged: a later pass with a larger budget may still improve it.
    budget_limited: bool,
}

/// Build CSR offsets (prefix sums) from per-entity counts. u64 output — the
/// sum of counts legitimately exceeds u32::MAX on large reorder passes.
fn build_csr_offsets(counts: &[u32]) -> Vec<u64> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    offsets.push(0u64);
    for &c in counts {
        offsets.push(offsets.last().unwrap() + c as u64);
    }
    offsets
}

impl ForwardIndex {
    #[inline]
    pub fn num_docs(&self) -> usize {
        if self.offsets.is_empty() {
            0
        } else {
            self.offsets.len() - 1
        }
    }

    #[inline]
    fn doc_terms(&self, doc: usize) -> &[u32] {
        let start = self.offsets[doc] as usize;
        let end = self.offsets[doc + 1] as usize;
        &self.terms[start..end]
    }

    /// Total postings in the forward index.
    pub fn total_postings(&self) -> u64 {
        self.offsets.last().copied().unwrap_or(0)
    }

    #[inline]
    pub fn budget_limited(&self) -> bool {
        self.budget_limited
    }
}

/// Build virtual→real and real→virtual vid maps from a BMP doc map.
///
/// A virtual slot is real iff its doc-map entry is not the `u32::MAX` padding
/// sentinel. Realness must come from the doc map itself: block-copy merged
/// segments carry each source's tail padding as *interior* padding, so
/// `vid < num_real_docs` does NOT identify real docs there.
///
/// Returns `(virtual_to_real, real_to_virtual)` where `virtual_to_real[vid]`
/// is the dense real index or `u32::MAX` for padding.
pub(crate) fn build_vid_maps(
    bmp: &crate::segment::reader::bmp::BmpIndex,
) -> crate::Result<(Vec<u32>, Vec<u32>)> {
    let ids = bmp.doc_map_ids_slice();
    let num_virtual = bmp.num_virtual_docs as usize;
    let expected_real = bmp.num_real_docs() as usize;
    let mut virtual_to_real = vec![u32::MAX; num_virtual];
    let mut real_to_virtual = Vec::with_capacity(expected_real);
    debug_assert_eq!(ids.len(), virtual_to_real.len() * 4);
    bmp.visit_real_slots_for_rewrite(|vid| {
        virtual_to_real[vid] = real_to_virtual.len() as u32;
        real_to_virtual.push(vid as u32);
    })?;
    Ok((virtual_to_real, real_to_virtual))
}

/// One (source, block) unit of forward-index construction. Because
/// [`build_vid_maps`] assigns real ids in ascending vid order, a block's real
/// docs form the contiguous per-source range
/// `real_start..real_start + real_len` — blocks can be processed in parallel
/// with disjoint output slices.
struct BlockJob {
    src: u32,
    block_id: u32,
    /// Per-source real index of the block's first real doc.
    real_start: u32,
    /// Number of real (non-padding) docs in the block.
    real_len: u32,
}

/// Enumerate jobs in (source, block) order — cumulative `real_len` tiles the
/// global real-id space `0..total_docs` exactly.
fn build_block_jobs(
    bmps: &[&crate::segment::reader::bmp::BmpIndex],
    vid_maps: &[(Vec<u32>, Vec<u32>)],
) -> Vec<BlockJob> {
    let total_blocks: usize = bmps.iter().map(|b| b.num_blocks as usize).sum();
    let mut jobs = Vec::with_capacity(total_blocks);
    for (src, (bmp, (v2r, _))) in bmps.iter().zip(vid_maps).enumerate() {
        let block_size = bmp.bmp_block_size as usize;
        let mut real_cursor = 0u32;
        for block_id in 0..bmp.num_blocks as usize {
            let vid_start = block_id * block_size;
            let vid_end = ((block_id + 1) * block_size).min(v2r.len());
            let real_len = v2r[vid_start..vid_end]
                .iter()
                .filter(|&&r| r != u32::MAX)
                .count() as u32;
            jobs.push(BlockJob {
                src: src as u32,
                block_id: block_id as u32,
                real_start: real_cursor,
                real_len,
            });
            real_cursor += real_len;
        }
    }
    jobs
}

/// Build forward index from BmpIndex sources (single or multi-source).
///
/// Documents are identified by dense *real* indices assigned sequentially
/// across sources: source 0 gets 0..n0, source 1 gets n0..n0+n1, etc., where
/// each n is the source's real (non-padding) doc count derived from its doc
/// map via [`build_vid_maps`].
///
/// Filters dims with doc_freq outside `[min_doc_freq, max_doc_freq]`.
/// If the estimated forward index memory exceeds `memory_budget_bytes`, the
/// highest-frequency dims are dropped to stay within budget. This prevents OOM
/// for huge segments at the cost of slightly reduced reorder quality.
///
/// Remaps term IDs to compact range for flat-array degree tracking.
#[cfg(test)]
pub(crate) fn build_forward_index_from_bmps(
    bmps: &[&crate::segment::reader::bmp::BmpIndex],
    min_doc_freq: usize,
    max_doc_freq: usize,
    memory_budget_bytes: usize,
) -> crate::Result<ForwardIndex> {
    let vid_maps: Vec<(Vec<u32>, Vec<u32>)> = bmps
        .iter()
        .map(|bmp| build_vid_maps(bmp))
        .collect::<crate::Result<_>>()?;
    Ok(build_forward_index_from_bmps_with_maps(
        bmps,
        &vid_maps,
        min_doc_freq,
        max_doc_freq,
        memory_budget_bytes,
    ))
}

/// Variant for reorder callers that already need the virtual/real maps during
/// output encoding. Reusing them avoids a second full document-map scan and a
/// duplicate real-to-virtual allocation on very large segments.
pub(crate) fn build_forward_index_from_bmps_with_maps(
    bmps: &[&crate::segment::reader::bmp::BmpIndex],
    vid_maps: &[(Vec<u32>, Vec<u32>)],
    min_doc_freq: usize,
    max_doc_freq: usize,
    memory_budget_bytes: usize,
) -> ForwardIndex {
    debug_assert_eq!(bmps.len(), vid_maps.len());
    let total_docs: usize = vid_maps.iter().map(|(_, r2v)| r2v.len()).sum();

    if total_docs == 0 {
        return ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
            parallel_bisect_lanes: 1,
            budget_limited: false,
        };
    }

    // Job list: one entry per (source, block). Real ids are assigned in
    // ascending vid order (see build_vid_maps), so each block owns a
    // contiguous real-id range — every phase below can process blocks in
    // parallel, writing disjoint slices.
    let jobs = build_block_jobs(bmps, vid_maps);

    // Phase 1: count document frequencies in bounded worker-local tables.
    // Shared atomics made popular dimensions a cache-coherence bottleneck;
    // private dense tables avoid that while retaining an exact memory cap.
    let max_dims = bmps
        .iter()
        .map(|bmp| bmp.dims() as usize)
        .max()
        .unwrap_or(0);
    let jobs_bytes = jobs
        .len()
        .saturating_mul(std::mem::size_of::<BlockJob>().saturating_add(40));
    let frequency_bytes = max_dims.saturating_mul(std::mem::size_of::<u32>());
    if frequency_bytes > memory_budget_bytes.saturating_sub(jobs_bytes) {
        log::warn!(
            "[reorder] memory budget {} cannot hold the {} dimension-frequency table; using identity order",
            crate::format_bytes(memory_budget_bytes as u64),
            crate::format_bytes(frequency_bytes as u64),
        );
        return ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
            parallel_bisect_lanes: 1,
            budget_limited: true,
        };
    }
    let Some(dim_df) = count_frequencies_bounded(
        &jobs,
        max_dims,
        memory_budget_bytes.saturating_sub(jobs_bytes),
        |job, counts| {
            let bmp = bmps[job.src as usize];
            let (v2r, _) = &vid_maps[job.src as usize];
            let block_size = bmp.bmp_block_size as usize;
            for (dim_id, _, postings) in bmp.iter_block_terms(job.block_id) {
                let mut frequency = 0u32;
                for posting in postings {
                    let vid = job.block_id as usize * block_size + posting.local_slot as usize;
                    if v2r.get(vid).is_some_and(|&real| real != u32::MAX) && posting.impact > 0 {
                        frequency = frequency.saturating_add(1);
                    }
                }
                if frequency > 0
                    && let Some(total) = counts.get_mut(dim_id as usize)
                {
                    *total = total.saturating_add(frequency);
                }
            }
        },
    ) else {
        log::warn!(
            "[reorder] memory budget {} cannot hold a bounded dimension-frequency table; using identity order",
            crate::format_bytes(memory_budget_bytes as u64),
        );
        return ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
            parallel_bisect_lanes: 1,
            budget_limited: true,
        };
    };

    // Retain the lowest-frequency candidates in a bounded heap while the
    // frequency table is live. This makes candidate discovery itself obey the
    // configured limit even for extremely large vocabularies.
    let (mut eligible, mut budget_limited) = select_frequency_candidates(
        &dim_df,
        min_doc_freq,
        max_doc_freq,
        memory_budget_bytes
            .saturating_sub(jobs_bytes)
            .saturating_sub(frequency_bytes),
    );
    drop(dim_df);

    // Memory budget: estimate forward index + bisection scratch.
    // Includes jobs/slice descriptors, dense remap, all per-document scratch,
    // and at least one exact TermDegrees allocation.
    let entity_scratch_bytes = total_docs.saturating_mul(32);
    let remap_bytes = max_dims.saturating_mul(4);
    let fixed_bytes = entity_scratch_bytes
        .saturating_add(remap_bytes)
        .saturating_add(jobs_bytes);
    let fit = fit_candidates_to_budget(&mut eligible, fixed_bytes, memory_budget_bytes);
    if fit.dropped > 0 {
        budget_limited = true;
        log::warn!(
            "[reorder] memory budget {}: estimated {}, dropped {} highest-df dims, keeping {} ({} postings)",
            crate::format_bytes(memory_budget_bytes as u64),
            crate::format_bytes(fit.estimated_bytes as u64),
            fit.dropped,
            eligible.len(),
            fit.retained_postings,
        );
    }

    if eligible.is_empty() {
        // The caller emits an identity permutation when there is no graph
        // signal. Avoid allocating per-document counts and u64 offsets only
        // to discover that the terms array is empty, especially when the
        // configured budget is below the fixed document scratch cost.
        return ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
            parallel_bisect_lanes: 1,
            budget_limited,
        };
    }

    let mut term_remap = vec![u32::MAX; max_dims];
    for (compact_id, &(dim_id, _)) in eligible.iter().enumerate() {
        term_remap[dim_id as usize] = compact_id as u32;
    }
    let num_active_terms = eligible.len();
    // Jobs, candidate metadata, and the dense remap are construction scratch
    // and are gone before graph recursion. Admit lanes against graph-resident
    // CSR/entity scratch so non-power-of-two pools are not needlessly rounded
    // down under a tight budget.
    let non_degree_bytes = entity_scratch_bytes.saturating_add(
        fit.retained_postings
            .saturating_mul(std::mem::size_of::<u32>()),
    );
    let parallel_bisect_lanes =
        parallel_bisect_lanes(memory_budget_bytes, non_degree_bytes, num_active_terms);
    drop(eligible);

    // Phase 2: count terms per doc (filtered) — per-block disjoint slices
    let mut counts = vec![0u32; total_docs];
    let fill_block_counts = |job: &BlockJob, out: &mut [u32]| {
        let bmp = bmps[job.src as usize];
        let (v2r, _) = &vid_maps[job.src as usize];
        let block_size = bmp.bmp_block_size as usize;
        for (dim_id, _, postings) in bmp.iter_block_terms(job.block_id) {
            if term_remap.get(dim_id as usize).copied().unwrap_or(u32::MAX) == u32::MAX {
                continue;
            }
            for p in postings {
                let vid = job.block_id as usize * block_size + p.local_slot as usize;
                let Some(&real) = v2r.get(vid) else {
                    continue;
                };
                if real != u32::MAX && p.impact > 0 {
                    out[(real - job.real_start) as usize] += 1;
                }
            }
        }
    };
    {
        let mut slices: Vec<(&BlockJob, &mut [u32])> = Vec::with_capacity(jobs.len());
        let mut rest: &mut [u32] = &mut counts;
        for job in &jobs {
            let (head, tail) = rest.split_at_mut(job.real_len as usize);
            slices.push((job, head));
            rest = tail;
        }
        #[cfg(feature = "native")]
        slices
            .into_par_iter()
            .for_each(|(job, out)| fill_block_counts(job, out));
        #[cfg(not(feature = "native"))]
        for (job, out) in slices {
            fill_block_counts(job, out);
        }
    }

    // Phase 3: build CSR offsets (u64 — sums exceed u32::MAX at scale)
    let offsets = build_csr_offsets(&counts);
    let total = *offsets.last().unwrap() as usize;
    drop(counts);

    // Phase 4: fill terms (compact IDs) — each block writes the contiguous
    // terms range covering its real docs; per-doc write cursors are local.
    let mut terms = vec![0u32; total];
    let fill_block_terms = |job: &BlockJob, global_real_start: usize, out: &mut [u32]| {
        let bmp = bmps[job.src as usize];
        let (v2r, _) = &vid_maps[job.src as usize];
        let block_size = bmp.bmp_block_size as usize;
        // local_slot is u8, so a block never holds more than 256 real docs
        assert!(job.real_len as usize <= 256, "BMP block exceeds 256 docs");
        let mut cursor = [0u32; 256];
        let base = offsets[global_real_start] as usize;
        for (dim_id, _, postings) in bmp.iter_block_terms(job.block_id) {
            let compact = term_remap.get(dim_id as usize).copied().unwrap_or(u32::MAX);
            if compact == u32::MAX {
                continue;
            }
            for p in postings {
                let vid = job.block_id as usize * block_size + p.local_slot as usize;
                let Some(&real) = v2r.get(vid) else {
                    continue;
                };
                if real != u32::MAX && p.impact > 0 {
                    let local = (real - job.real_start) as usize;
                    let pos =
                        offsets[global_real_start + local] as usize - base + cursor[local] as usize;
                    out[pos] = compact;
                    cursor[local] += 1;
                }
            }
        }
    };
    {
        let mut slices: Vec<(&BlockJob, usize, &mut [u32])> = Vec::with_capacity(jobs.len());
        let mut rest: &mut [u32] = &mut terms;
        let mut global_real = 0usize;
        for job in &jobs {
            let len =
                (offsets[global_real + job.real_len as usize] - offsets[global_real]) as usize;
            let (head, tail) = rest.split_at_mut(len);
            slices.push((job, global_real, head));
            rest = tail;
            global_real += job.real_len as usize;
        }
        #[cfg(feature = "native")]
        slices
            .into_par_iter()
            .for_each(|(job, g, out)| fill_block_terms(job, g, out));
        #[cfg(not(feature = "native"))]
        for (job, g, out) in slices {
            fill_block_terms(job, g, out);
        }
    }

    ForwardIndex {
        terms,
        offsets,
        num_terms: num_active_terms,
        parallel_bisect_lanes,
        budget_limited,
    }
}

/// Build a forward index over BLOCKS (one entity per block, its terms = the
/// block's header dim list). Used by block-level reorder: BP over blocks is
/// ~block_size× cheaper than over records and only needs to decide superblock
/// assignment. Blocks are numbered globally across sources in source order.
///
/// Dims appearing in fewer than 2 blocks carry no clustering signal and are
/// dropped; the memory budget applies as in the record-level builder.
pub(crate) fn build_forward_index_from_blocks(
    bmps: &[&crate::segment::reader::bmp::BmpIndex],
    memory_budget_bytes: usize,
) -> ForwardIndex {
    let total_blocks: usize = bmps.iter().map(|b| b.num_blocks as usize).sum();
    if total_blocks == 0 {
        return ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
            parallel_bisect_lanes: 1,
            budget_limited: false,
        };
    }

    // (source, block) pairs in global block order — the parallel unit.
    let blocks: Vec<(u32, u32)> = bmps
        .iter()
        .enumerate()
        .flat_map(|(src, bmp)| (0..bmp.num_blocks).map(move |b| (src as u32, b)))
        .collect();

    // Phase 1: bounded worker-local frequency tables. This is the same policy
    // as record-level BP and avoids hot atomic increments on common terms.
    let max_dims = bmps
        .iter()
        .map(|bmp| bmp.dims() as usize)
        .max()
        .unwrap_or(0);
    let blocks_bytes = blocks
        .len()
        .saturating_mul(std::mem::size_of::<(u32, u32)>().saturating_add(32));
    let frequency_bytes = max_dims.saturating_mul(std::mem::size_of::<u32>());
    if frequency_bytes > memory_budget_bytes.saturating_sub(blocks_bytes) {
        log::warn!(
            "[reorder] block-level frequency table exceeds memory budget; using identity order"
        );
        return ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
            parallel_bisect_lanes: 1,
            budget_limited: true,
        };
    }
    let Some(dim_bf) = count_frequencies_bounded(
        &blocks,
        max_dims,
        memory_budget_bytes.saturating_sub(blocks_bytes),
        |&(src, block_id), counts| {
            for (dim_id, _, _) in bmps[src as usize].iter_block_terms(block_id) {
                if let Some(count) = counts.get_mut(dim_id as usize) {
                    *count = count.saturating_add(1);
                }
            }
        },
    ) else {
        log::warn!(
            "[reorder] block-level frequency table cannot fit its bounded allocation; using identity order"
        );
        return ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
            parallel_bisect_lanes: 1,
            budget_limited: true,
        };
    };

    let max_bf = (total_blocks as f64 * 0.9) as usize;
    let (mut eligible, mut budget_limited) = select_frequency_candidates(
        &dim_bf,
        2,
        max_bf.max(2),
        memory_budget_bytes
            .saturating_sub(blocks_bytes)
            .saturating_sub(frequency_bytes),
    );
    drop(dim_bf);

    let entity_scratch_bytes = total_blocks.saturating_mul(32);
    let remap_bytes = max_dims.saturating_mul(4);
    let fixed_bytes = entity_scratch_bytes
        .saturating_add(remap_bytes)
        .saturating_add(blocks_bytes);
    let fit = fit_candidates_to_budget(&mut eligible, fixed_bytes, memory_budget_bytes);
    if fit.dropped > 0 {
        budget_limited = true;
        log::warn!(
            "[reorder] block-level fwd index over budget — dropped {} highest-bf dims",
            fit.dropped,
        );
    }

    if eligible.is_empty() {
        return ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
            parallel_bisect_lanes: 1,
            budget_limited,
        };
    }

    let mut term_remap = vec![u32::MAX; max_dims];
    for (compact, &(dim_id, _)) in eligible.iter().enumerate() {
        term_remap[dim_id as usize] = compact as u32;
    }
    let num_terms = eligible.len();
    let non_degree_bytes = entity_scratch_bytes.saturating_add(
        fit.retained_postings
            .saturating_mul(std::mem::size_of::<u32>()),
    );
    let parallel_bisect_lanes =
        parallel_bisect_lanes(memory_budget_bytes, non_degree_bytes, num_terms);
    drop(eligible);

    // Phase 2+3: counts and CSR fill — one entity per block, so each block
    // maps to a single count cell and a contiguous terms range.
    let count_remapped = |&(src, block_id): &(u32, u32)| -> u32 {
        bmps[src as usize]
            .iter_block_terms(block_id)
            .filter(|(dim_id, _, _)| {
                term_remap
                    .get(*dim_id as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
                    != u32::MAX
            })
            .count() as u32
    };
    #[cfg(feature = "native")]
    let counts: Vec<u32> = blocks.par_iter().map(count_remapped).collect();
    #[cfg(not(feature = "native"))]
    let counts: Vec<u32> = blocks.iter().map(count_remapped).collect();

    let offsets = build_csr_offsets(&counts);
    let total = *offsets.last().unwrap() as usize;
    drop(counts);

    let mut terms = vec![0u32; total];
    let fill_block = |&(src, block_id): &(u32, u32), out: &mut [u32]| {
        let mut n = 0usize;
        for (dim_id, _, _) in bmps[src as usize].iter_block_terms(block_id) {
            let compact = term_remap.get(dim_id as usize).copied().unwrap_or(u32::MAX);
            if compact != u32::MAX {
                out[n] = compact;
                n += 1;
            }
        }
    };
    {
        let mut slices: Vec<(&(u32, u32), &mut [u32])> = Vec::with_capacity(blocks.len());
        let mut rest: &mut [u32] = &mut terms;
        for (gb, b) in blocks.iter().enumerate() {
            let len = (offsets[gb + 1] - offsets[gb]) as usize;
            let (head, tail) = rest.split_at_mut(len);
            slices.push((b, head));
            rest = tail;
        }
        #[cfg(feature = "native")]
        slices
            .into_par_iter()
            .for_each(|(b, out)| fill_block(b, out));
        #[cfg(not(feature = "native"))]
        for (b, out) in slices {
            fill_block(b, out);
        }
    }

    ForwardIndex {
        terms,
        offsets,
        num_terms,
        parallel_bisect_lanes,
        budget_limited,
    }
}

// ── Recursive Graph Bisection ────────────────────────────────────────────

/// CPU/depth budget for a BP pass. BP is an anytime algorithm: stopping at
/// any depth or deadline still yields a valid permutation, and because the
/// output layout becomes the next pass's input order, repeated budgeted
/// passes warm-start and deepen (top levels converge in ~0 swaps, the budget
/// flows to deeper levels).
#[derive(Clone, Copy, Debug, Default)]
pub struct BpBudget {
    /// Stop recursion at partitions of at most this many docs instead of
    /// descending to block granularity. `None` = full depth. Capping at
    /// superblock granularity (superblock_size × block_size docs) keeps most
    /// of the superblock-pruning win at ~⅓ less depth.
    pub min_partition_docs: Option<usize>,
    /// Wall-clock cap for the whole BP computation. The pass ends cleanly at
    /// the deadline with whatever depth it reached (`converged = false`).
    /// Ignored on wasm (no monotonic clock).
    pub time_budget: Option<std::time::Duration>,
}

impl BpBudget {
    /// Unbudgeted: full depth, no deadline.
    pub fn full() -> Self {
        Self::default()
    }
}

/// Build one partition's term degrees in preallocated lanes.
///
/// At coarse levels each worker scans a contiguous document range into a
/// private lane, then the lanes are reduced into `workspaces[0]`. Recursive
/// siblings receive disjoint workspace slices, so the complete pass performs
/// exactly the vocabulary-sized allocations admitted before BP starts.
fn build_term_degrees(
    docs: &[u32],
    mid: usize,
    fwd: &ForwardIndex,
    workspaces: &mut [TermDegrees],
) {
    debug_assert!(!workspaces.is_empty());
    let build_range = |degrees: &mut TermDegrees, start: usize, chunk: &[u32]| {
        degrees.reset();
        for (offset, &doc) in chunk.iter().enumerate() {
            let side = usize::from(start + offset >= mid);
            for &term in fwd.doc_terms(doc as usize) {
                degrees.entry_mut(term as usize)[side] += 1;
            }
        }
    };

    #[cfg(feature = "native")]
    {
        let workers = workspaces
            .len()
            .min(docs.len().div_ceil(PARALLEL_BP_MIN_ENTITIES).max(1));
        if workers > 1 {
            let chunk_len = docs.len().div_ceil(workers);
            workspaces[..workers]
                .par_iter_mut()
                .enumerate()
                .for_each(|(worker, degrees)| {
                    let start = worker * chunk_len;
                    let end = (start + chunk_len).min(docs.len());
                    build_range(degrees, start, &docs[start..end]);
                });
            let (degrees, partials) = workspaces[..workers]
                .split_first_mut()
                .expect("at least one BP degree workspace");
            for partial in partials {
                degrees.merge_from(partial);
            }
            return;
        }
        build_range(&mut workspaces[0], 0, docs);
    }

    #[cfg(not(feature = "native"))]
    build_range(&mut workspaces[0], 0, docs);
}

/// Convert `f32::total_cmp` ordering into an unsigned radix key.
///
/// BP gains are finite, but preserving the complete IEEE total order makes
/// the parallel selector exactly equivalent to the former comparator even if
/// malformed input ever produces a signed zero, infinity, or NaN.
#[inline]
fn gain_order_key(gain: f32) -> u32 {
    let bits = gain.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

/// Find the gain key of the last entity admitted to the left half and the
/// number of strictly lower keys. Four byte-histogram passes replace the old
/// serial `select_nth_unstable_by` over an 8-byte index per entity.
fn select_gain_threshold(gains: &[f32], left_count: usize) -> (u32, usize) {
    debug_assert!(left_count > 0 && left_count <= gains.len());
    let mut rank_within_prefix = left_count - 1;
    let mut strictly_lower = 0usize;
    let mut prefix = 0u32;
    let mut prefix_mask = 0u32;

    for shift in [24u32, 16, 8, 0] {
        let histogram = {
            #[cfg(feature = "native")]
            {
                if gains.len() >= PARALLEL_BP_MIN_ENTITIES {
                    gains
                        .par_iter()
                        .fold(
                            || Box::new([0usize; 256]),
                            |mut counts, &gain| {
                                let key = gain_order_key(gain);
                                if key & prefix_mask == prefix {
                                    counts[((key >> shift) & 0xff) as usize] += 1;
                                }
                                counts
                            },
                        )
                        .reduce(
                            || Box::new([0usize; 256]),
                            |mut left, right| {
                                for (dst, &count) in left.iter_mut().zip(right.iter()) {
                                    *dst += count;
                                }
                                left
                            },
                        )
                } else {
                    let mut counts = [0usize; 256];
                    for &gain in gains {
                        let key = gain_order_key(gain);
                        if key & prefix_mask == prefix {
                            counts[((key >> shift) & 0xff) as usize] += 1;
                        }
                    }
                    Box::new(counts)
                }
            }
            #[cfg(not(feature = "native"))]
            {
                let mut counts = [0usize; 256];
                for &gain in gains {
                    let key = gain_order_key(gain);
                    if key & prefix_mask == prefix {
                        counts[((key >> shift) & 0xff) as usize] += 1;
                    }
                }
                counts
            }
        };

        let mut before_bucket = 0usize;
        let mut selected_bucket = None;
        for (bucket, count) in histogram.iter().copied().enumerate() {
            if rank_within_prefix < before_bucket + count {
                selected_bucket = Some(bucket as u32);
                rank_within_prefix -= before_bucket;
                strictly_lower += before_bucket;
                break;
            }
            before_bucket += count;
        }
        let selected_bucket = selected_bucket.expect("BP radix selection lost the requested rank");
        prefix |= selected_bucket << shift;
        prefix_mask |= 0xffu32 << shift;
    }

    (prefix, strictly_lower)
}

enum PartitionDegreeUpdate {
    /// Parallel workers accumulated directional movement counts in the first
    /// reusable movement workspace.
    Moves,
    /// Small partitions use the former unstable selection order. Keep its
    /// reusable rank map long enough to update only records that crossed the
    /// cut.
    Ranked,
    /// A large partition with only one affordable degree lane still uses the
    /// bounded-memory radix selector, then applies deltas serially.
    Threshold {
        threshold_key: u32,
        ties_left: usize,
    },
}

struct PartitionOutcome {
    swap_count: usize,
    degree_update: PartitionDegreeUpdate,
}

#[derive(Clone, Copy)]
struct PartitionChunk {
    start: usize,
    end: usize,
    strictly_lower: usize,
    equal: usize,
    ties_left: usize,
}

#[inline]
fn select_left(key: u32, threshold_key: u32, equal_seen: &mut usize, ties_left: usize) -> bool {
    if key < threshold_key {
        true
    } else if key == threshold_key {
        let selected = *equal_seen < ties_left;
        *equal_seen += 1;
        selected
    } else {
        false
    }
}

/// Exact, deterministic parallel partition by `(gain.total_cmp(), old_index)`.
///
/// Output is stable within each half. Parallel workers also accumulate
/// per-term degree deltas for moved entities, eliminating the former serial
/// postings update without rescanning every posting after each iteration.
/// Small partitions retain the old unstable selector in their preallocated
/// rank slice: it is faster than four radix scans, and its within-half
/// permutation supplies the graph algorithm's established symmetry breaking.
fn partition_by_gain(
    docs: &[u32],
    gains: &[f32],
    mid: usize,
    fwd: &ForwardIndex,
    movement_workspaces: &mut [TermDegrees],
    output: &mut [u32],
    ranked_scratch: &mut [usize],
) -> PartitionOutcome {
    #[cfg(not(feature = "native"))]
    let _ = (fwd, &movement_workspaces);

    #[cfg(feature = "native")]
    if !movement_workspaces.is_empty() && docs.len() >= PARALLEL_BP_MIN_ENTITIES {
        let (threshold_key, strictly_lower) = select_gain_threshold(gains, mid);
        let ties_left = mid - strictly_lower;

        // `degrees` remains live while these deltas are built. Reserving one
        // lane for it keeps total vocabulary arrays within the admitted set.
        let chunk_count = movement_workspaces
            .len()
            .min(docs.len().div_ceil(PARALLEL_BP_MIN_ENTITIES).max(1));
        let chunk_len = docs.len().div_ceil(chunk_count);
        let mut chunks: Vec<PartitionChunk> = gains
            .par_chunks(chunk_len)
            .enumerate()
            .map(|(chunk_id, chunk)| {
                let mut lower = 0usize;
                let mut equal = 0usize;
                for &gain in chunk {
                    match gain_order_key(gain).cmp(&threshold_key) {
                        std::cmp::Ordering::Less => lower += 1,
                        std::cmp::Ordering::Equal => equal += 1,
                        std::cmp::Ordering::Greater => {}
                    }
                }
                let start = chunk_id * chunk_len;
                PartitionChunk {
                    start,
                    end: start + chunk.len(),
                    strictly_lower: lower,
                    equal,
                    ties_left: 0,
                }
            })
            .collect();

        let mut remaining_ties = ties_left;
        for chunk in &mut chunks {
            chunk.ties_left = remaining_ties.min(chunk.equal);
            remaining_ties -= chunk.ties_left;
        }
        debug_assert_eq!(remaining_ties, 0);

        let (mut left_rest, mut right_rest) = output.split_at_mut(mid);
        let mut jobs = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let left_len = chunk.strictly_lower + chunk.ties_left;
            let right_len = chunk.end - chunk.start - left_len;
            let (left_out, next_left) = left_rest.split_at_mut(left_len);
            let (right_out, next_right) = right_rest.split_at_mut(right_len);
            jobs.push((
                chunk.start,
                &docs[chunk.start..chunk.end],
                &gains[chunk.start..chunk.end],
                chunk.ties_left,
                left_out,
                right_out,
            ));
            left_rest = next_left;
            right_rest = next_right;
        }
        debug_assert!(left_rest.is_empty() && right_rest.is_empty());

        let swap_count = movement_workspaces[..chunk_count]
            .par_iter_mut()
            .zip(jobs.into_par_iter())
            .map(
                |(moves, (start, docs, gains, ties_for_chunk, left_out, right_out))| {
                    moves.reset();
                    let mut equal_seen = 0usize;
                    let mut left_cursor = 0usize;
                    let mut right_cursor = 0usize;
                    let mut swaps = 0usize;

                    for (offset, (&doc, &gain)) in docs.iter().zip(gains).enumerate() {
                        let key = gain_order_key(gain);
                        let now_left =
                            select_left(key, threshold_key, &mut equal_seen, ties_for_chunk);
                        if now_left {
                            left_out[left_cursor] = doc;
                            left_cursor += 1;
                        } else {
                            right_out[right_cursor] = doc;
                            right_cursor += 1;
                        }

                        let was_left = start + offset < mid;
                        if was_left != now_left {
                            swaps += 1;
                            // [right→left, left→right]
                            let direction = usize::from(was_left);
                            for &term in fwd.doc_terms(doc as usize) {
                                moves.entry_mut(term as usize)[direction] += 1;
                            }
                        }
                    }
                    debug_assert_eq!(left_cursor, left_out.len());
                    debug_assert_eq!(right_cursor, right_out.len());
                    swaps
                },
            )
            .sum();

        let (moves, partials) = movement_workspaces[..chunk_count]
            .split_first_mut()
            .expect("parallel BP partition must use at least one movement workspace");
        for partial in partials {
            moves.merge_from(partial);
        }

        return PartitionOutcome {
            swap_count,
            degree_update: PartitionDegreeUpdate::Moves,
        };
    }

    if docs.len() < PARALLEL_BP_MIN_ENTITIES {
        debug_assert_eq!(ranked_scratch.len(), docs.len());
        for (index, rank) in ranked_scratch.iter_mut().enumerate() {
            *rank = index;
        }
        ranked_scratch.select_nth_unstable_by(mid, |&left, &right| {
            gains[left]
                .total_cmp(&gains[right])
                .then_with(|| left.cmp(&right))
        });

        let mut swaps = 0usize;
        for (rank, &old_index) in ranked_scratch.iter().enumerate() {
            output[rank] = docs[old_index];
            swaps += usize::from((old_index < mid) != (rank < mid));
        }
        return PartitionOutcome {
            swap_count: swaps,
            degree_update: PartitionDegreeUpdate::Ranked,
        };
    }

    let (threshold_key, strictly_lower) = select_gain_threshold(gains, mid);
    let ties_left = mid - strictly_lower;
    let mut equal_seen = 0usize;
    let mut left_cursor = 0usize;
    let mut right_cursor = mid;
    let mut swaps = 0usize;
    for (idx, (&doc, &gain)) in docs.iter().zip(gains).enumerate() {
        let key = gain_order_key(gain);
        let now_left = select_left(key, threshold_key, &mut equal_seen, ties_left);
        if now_left {
            output[left_cursor] = doc;
            left_cursor += 1;
        } else {
            output[right_cursor] = doc;
            right_cursor += 1;
        }
        swaps += usize::from((idx < mid) != now_left);
    }
    debug_assert_eq!(left_cursor, mid);
    debug_assert_eq!(right_cursor, docs.len());

    PartitionOutcome {
        swap_count: swaps,
        degree_update: PartitionDegreeUpdate::Threshold {
            threshold_key,
            ties_left,
        },
    }
}

/// Apply degree changes for a bounded-memory serial radix partition.
fn update_degrees_for_threshold_partition(
    docs: &[u32],
    gains: &[f32],
    mid: usize,
    threshold_key: u32,
    ties_left: usize,
    fwd: &ForwardIndex,
    degrees: &mut TermDegrees,
) {
    let mut equal_seen = 0usize;
    for (idx, (&doc, &gain)) in docs.iter().zip(gains).enumerate() {
        let key = gain_order_key(gain);
        let now_left = select_left(key, threshold_key, &mut equal_seen, ties_left);
        let was_left = idx < mid;
        if was_left == now_left {
            continue;
        }
        let left_delta = if was_left { -1i64 } else { 1i64 };
        for &term in fwd.doc_terms(doc as usize) {
            let degree = degrees.entry_mut(term as usize);
            let new_left = degree[0] as i64 + left_delta;
            let new_right = degree[1] as i64 - left_delta;
            debug_assert!(new_left >= 0 && new_right >= 0);
            degree[0] = new_left as u32;
            degree[1] = new_right as u32;
        }
    }
}

/// Apply degree changes using the exact unstable rank order selected for a
/// small partition.
fn update_degrees_for_ranked_partition(
    docs: &[u32],
    ranked: &[usize],
    mid: usize,
    fwd: &ForwardIndex,
    degrees: &mut TermDegrees,
) {
    for (rank, &old_index) in ranked.iter().enumerate() {
        let was_left = old_index < mid;
        let now_left = rank < mid;
        if was_left == now_left {
            continue;
        }
        let left_delta = if was_left { -1i64 } else { 1i64 };
        for &term in fwd.doc_terms(docs[old_index] as usize) {
            let degree = degrees.entry_mut(term as usize);
            let new_left = degree[0] as i64 + left_delta;
            let new_right = degree[1] as i64 - left_delta;
            debug_assert!(new_left >= 0 && new_right >= 0);
            degree[0] = new_left as u32;
            degree[1] = new_right as u32;
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BpProgressLabel<'a> {
    pub index: &'a str,
    pub field: &'a str,
    pub entity_kind: &'static str,
}

#[cfg(test)]
impl BpProgressLabel<'static> {
    fn anonymous() -> Self {
        Self {
            index: "unknown",
            field: "unknown",
            entity_kind: "entities",
        }
    }
}

#[cfg(feature = "native")]
struct BpProgress<'a> {
    label: BpProgressLabel<'a>,
    start: std::time::Instant,
    total_entities: usize,
    total_postings: u64,
    expected_depth: usize,
    next_log_ms: std::sync::atomic::AtomicU64,
    active_partitions: std::sync::atomic::AtomicU64,
    partitions_started: std::sync::atomic::AtomicU64,
    partitions_completed: std::sync::atomic::AtomicU64,
    iterations: std::sync::atomic::AtomicU64,
    entity_passes: std::sync::atomic::AtomicU64,
    swaps: std::sync::atomic::AtomicU64,
    deepest_level: std::sync::atomic::AtomicU64,
    objective_stops: std::sync::atomic::AtomicU64,
    last_objective_delta_bits: std::sync::atomic::AtomicU64,
    last_relative_delta_bits: std::sync::atomic::AtomicU64,
    active_metric_released: std::sync::atomic::AtomicBool,
}

#[cfg(feature = "native")]
impl<'a> BpProgress<'a> {
    fn new(
        label: BpProgressLabel<'a>,
        total_entities: usize,
        total_postings: u64,
        expected_depth: usize,
    ) -> Self {
        log::info!(
            "[reorder][bp] started: index={} field={} entity_kind={} entities={} postings={} expected_depth={} objective_stall_threshold={:.1e}x{} min_objective_iterations={}",
            label.index,
            label.field,
            label.entity_kind,
            total_entities,
            total_postings,
            expected_depth,
            MIN_RELATIVE_OBJECTIVE_IMPROVEMENT,
            OBJECTIVE_STALL_ITERATIONS,
            MIN_OBJECTIVE_ITERATIONS,
        );
        crate::observe::reorder_bp_started(label.index, label.field, label.entity_kind);
        Self {
            label,
            start: std::time::Instant::now(),
            total_entities,
            total_postings,
            expected_depth,
            next_log_ms: std::sync::atomic::AtomicU64::new(30_000),
            active_partitions: std::sync::atomic::AtomicU64::new(0),
            partitions_started: std::sync::atomic::AtomicU64::new(0),
            partitions_completed: std::sync::atomic::AtomicU64::new(0),
            iterations: std::sync::atomic::AtomicU64::new(0),
            entity_passes: std::sync::atomic::AtomicU64::new(0),
            swaps: std::sync::atomic::AtomicU64::new(0),
            deepest_level: std::sync::atomic::AtomicU64::new(0),
            objective_stops: std::sync::atomic::AtomicU64::new(0),
            last_objective_delta_bits: std::sync::atomic::AtomicU64::new(0f64.to_bits()),
            last_relative_delta_bits: std::sync::atomic::AtomicU64::new(0f64.to_bits()),
            active_metric_released: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn partition_started(&self, level: usize) {
        self.active_partitions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.partitions_started
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.deepest_level
            .fetch_max(level as u64, std::sync::atomic::Ordering::Relaxed);
    }

    fn partition_finished(&self) {
        self.partitions_completed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.active_partitions
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn iteration(&self, entities: usize, swaps: usize, objective_delta: f64, relative_delta: f64) {
        self.iterations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.entity_passes
            .fetch_add(entities as u64, std::sync::atomic::Ordering::Relaxed);
        self.swaps
            .fetch_add(swaps as u64, std::sync::atomic::Ordering::Relaxed);
        self.last_objective_delta_bits.store(
            objective_delta.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.last_relative_delta_bits.store(
            relative_delta.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.maybe_log();
    }

    fn objective_stop(&self) {
        self.objective_stops
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn maybe_log(&self) {
        let elapsed_ms = self.start.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let next = self.next_log_ms.load(std::sync::atomic::Ordering::Relaxed);
        if elapsed_ms < next
            || self
                .next_log_ms
                .compare_exchange(
                    next,
                    elapsed_ms.saturating_add(30_000),
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_err()
        {
            return;
        }

        let active = self
            .active_partitions
            .load(std::sync::atomic::Ordering::Relaxed);
        let started = self
            .partitions_started
            .load(std::sync::atomic::Ordering::Relaxed);
        let completed = self
            .partitions_completed
            .load(std::sync::atomic::Ordering::Relaxed);
        let iterations = self.iterations.load(std::sync::atomic::Ordering::Relaxed);
        let entity_passes = self
            .entity_passes
            .load(std::sync::atomic::Ordering::Relaxed);
        let swaps = self.swaps.load(std::sync::atomic::Ordering::Relaxed);
        let deepest = self
            .deepest_level
            .load(std::sync::atomic::Ordering::Relaxed);
        let objective_delta = f64::from_bits(
            self.last_objective_delta_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        let relative_delta = f64::from_bits(
            self.last_relative_delta_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        log::info!(
            "[reorder][bp] progress: index={} field={} entity_kind={} elapsed={:.1}s depth={}/{} partitions={}/{} active={} iterations={} entity_passes={} swaps={} last_objective_delta={:.3} relative={:.3e}",
            self.label.index,
            self.label.field,
            self.label.entity_kind,
            self.start.elapsed().as_secs_f64(),
            deepest,
            self.expected_depth,
            completed,
            started,
            active,
            iterations,
            entity_passes,
            swaps,
            objective_delta,
            relative_delta,
        );
    }

    fn finish(&self, converged: bool, memory_limited: bool, deadline_exhausted: bool) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let partitions = self
            .partitions_completed
            .load(std::sync::atomic::Ordering::Relaxed);
        let iterations = self.iterations.load(std::sync::atomic::Ordering::Relaxed);
        let entity_passes = self
            .entity_passes
            .load(std::sync::atomic::Ordering::Relaxed);
        let swaps = self.swaps.load(std::sync::atomic::Ordering::Relaxed);
        let deepest = self
            .deepest_level
            .load(std::sync::atomic::Ordering::Relaxed);
        let objective_stops = self
            .objective_stops
            .load(std::sync::atomic::Ordering::Relaxed);
        let stop_reason = if memory_limited {
            "memory_budget"
        } else if deadline_exhausted {
            "time_budget"
        } else if objective_stops > 0 {
            "objective"
        } else {
            "complete"
        };
        log::info!(
            "[reorder][bp] completed: index={} field={} entity_kind={} entities={} postings={} elapsed={:.1}s depth={}/{} partitions={} iterations={} entity_passes={} swaps={} objective_stops={} converged={} stop_reason={}",
            self.label.index,
            self.label.field,
            self.label.entity_kind,
            self.total_entities,
            self.total_postings,
            elapsed,
            deepest,
            self.expected_depth,
            partitions,
            iterations,
            entity_passes,
            swaps,
            objective_stops,
            converged,
            stop_reason,
        );
        crate::observe::reorder_bp_pass(
            self.label.index,
            self.label.field,
            self.label.entity_kind,
            stop_reason,
            elapsed,
            self.total_entities,
            self.total_postings,
            partitions,
            iterations,
            entity_passes,
            swaps,
            converged,
        );
        self.release_active_metric();
    }

    fn release_active_metric(&self) {
        if !self
            .active_metric_released
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            crate::observe::reorder_bp_finished(
                self.label.index,
                self.label.field,
                self.label.entity_kind,
            );
        }
    }
}

#[cfg(feature = "native")]
impl Drop for BpProgress<'_> {
    fn drop(&mut self) {
        self.release_active_metric();
    }
}

#[cfg(not(feature = "native"))]
struct BpProgress<'a>(std::marker::PhantomData<&'a ()>);

#[cfg(not(feature = "native"))]
impl BpProgress<'_> {
    fn new(_: BpProgressLabel<'_>, _: usize, _: u64, _: usize) -> Self {
        Self(std::marker::PhantomData)
    }
    fn partition_started(&self, _: usize) {}
    fn partition_finished(&self) {}
    fn iteration(&self, _: usize, _: usize, _: f64, _: f64) {}
    fn objective_stop(&self) {}
    fn finish(&self, _: bool, _: bool, _: bool) {}
}

/// Recursive graph bisection. Returns `(perm, converged)` where
/// `perm[new_pos] = old_index`. Convergence is false when the wall-clock or
/// memory budget prevents the requested work from finishing; a configured
/// depth cap is a chosen target and reports converged.
///
/// `min_partition_size` should be the configured BMP block size.
/// `max_iters` controls convergence (20 is standard).
///
/// Term IDs in the forward index must be compact (0..num_terms) so we can
/// use flat arrays for O(1) degree lookups instead of hash maps.
#[cfg(test)]
pub(crate) fn graph_bisection(
    fwd: &ForwardIndex,
    min_partition_size: usize,
    max_iters: usize,
    budget: BpBudget,
) -> (Vec<u32>, bool) {
    graph_bisection_with_progress(
        fwd,
        min_partition_size,
        max_iters,
        budget,
        BpProgressLabel::anonymous(),
    )
}

pub(crate) fn graph_bisection_with_progress(
    fwd: &ForwardIndex,
    min_partition_size: usize,
    max_iters: usize,
    budget: BpBudget,
    progress_label: BpProgressLabel<'_>,
) -> (Vec<u32>, bool) {
    let n = fwd.num_docs();
    if n == 0 {
        return (Vec::new(), !fwd.budget_limited);
    }

    let effective_min_partition = budget
        .min_partition_docs
        .unwrap_or(0)
        .max(min_partition_size);

    let mut docs: Vec<u32> = (0..n as u32).collect();
    let depth = if effective_min_partition > 0 {
        ((n as f64) / (effective_min_partition as f64))
            .log2()
            .ceil() as usize
    } else {
        0
    };
    let log_table = build_log_table(4096);
    let progress = BpProgress::new(progress_label, n, fwd.total_postings(), depth);

    log::debug!(
        "BP graph_bisection: n={}, min_partition={}, max_iters={}, depth=~{}, time_budget={:?}",
        n,
        effective_min_partition,
        max_iters,
        depth,
        budget.time_budget,
    );

    #[cfg(feature = "native")]
    let deadline = budget.time_budget.map(|duration| {
        let now = std::time::Instant::now();
        now.checked_add(duration).unwrap_or(now)
    });
    #[cfg(not(feature = "native"))]
    let deadline: Option<()> = None;

    let exhausted = std::sync::atomic::AtomicBool::new(false);
    let context = BisectContext {
        fwd,
        min_partition_size: effective_min_partition,
        max_iters,
        log_table: &log_table,
        #[cfg(feature = "native")]
        deadline,
        #[cfg(not(feature = "native"))]
        deadline,
        exhausted: &exhausted,
        progress: &progress,
    };
    let immediately_exhausted = {
        #[cfg(feature = "native")]
        {
            deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline)
        }
        #[cfg(not(feature = "native"))]
        {
            false
        }
    };
    if immediately_exhausted {
        exhausted.store(true, std::sync::atomic::Ordering::Relaxed);
    } else if n > effective_min_partition {
        // Entity scratch is allocated once for the pass and carved into
        // disjoint slices down the recursion tree. Degree lanes are likewise
        // allocated exactly once from the memory-budgeted lane count.
        let mut gains = vec![0.0f32; n];
        let mut partitioned = vec![0u32; n];
        let mut ranked = vec![0usize; n];
        #[cfg(feature = "native")]
        let degree_lanes = fwd.parallel_bisect_lanes.max(1);
        #[cfg(not(feature = "native"))]
        let degree_lanes = 1usize;
        let mut degree_workspaces: Vec<TermDegrees> = (0..degree_lanes)
            .map(|_| TermDegrees::new(fwd.num_terms))
            .collect();
        bisect(
            &mut docs,
            &mut gains,
            &mut partitioned,
            &mut ranked,
            &mut degree_workspaces,
            0,
            &context,
        );
    }

    let deadline_exhausted = exhausted.load(std::sync::atomic::Ordering::Relaxed);
    let converged = !fwd.budget_limited && !deadline_exhausted;
    progress.finish(converged, fwd.budget_limited, deadline_exhausted);
    if !converged {
        log::info!(
            "BP graph_bisection: budget incomplete at n={} (time={:?}, memory_limited={}) — emitting partial (still valid) permutation",
            n,
            budget.time_budget,
            fwd.budget_limited,
        );
    }
    (docs, converged)
}

/// Recursive bisection of a document slice.
///
/// Uses flat `Vec<u32>` degree arrays indexed by compact term_id for cache-friendly
/// O(1) lookups (vs FxHashMap which has poor cache locality at scale).
///
/// Gain computation is parallelized via rayon for large partitions (n > 4096).
/// Adaptive iteration count reduces work at top levels where coarse splits
/// converge faster and dominate total runtime.
struct BisectContext<'a> {
    fwd: &'a ForwardIndex,
    min_partition_size: usize,
    max_iters: usize,
    log_table: &'a [f32],
    #[cfg(feature = "native")]
    deadline: Option<std::time::Instant>,
    #[cfg(not(feature = "native"))]
    deadline: Option<()>,
    exhausted: &'a std::sync::atomic::AtomicBool,
    progress: &'a BpProgress<'a>,
}

#[allow(clippy::too_many_arguments)]
fn bisect(
    docs: &mut [u32],
    gains: &mut [f32],
    partitioned: &mut [u32],
    ranked_scratch: &mut [usize],
    degree_workspaces: &mut [TermDegrees],
    level: usize,
    context: &BisectContext<'_>,
) {
    let n = docs.len();
    debug_assert_eq!(gains.len(), n);
    debug_assert_eq!(partitioned.len(), n);
    debug_assert_eq!(ranked_scratch.len(), n);
    debug_assert!(!degree_workspaces.is_empty());
    if n <= context.min_partition_size {
        return;
    }
    // Anytime cutoff: leave this subtree in its current (valid) order.
    if context.exhausted.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    #[cfg(feature = "native")]
    if let Some(dl) = context.deadline
        && std::time::Instant::now() >= dl
    {
        context
            .exhausted
            .store(true, std::sync::atomic::Ordering::Relaxed);
        return;
    }
    #[cfg(not(feature = "native"))]
    let _ = context.deadline;

    context.progress.partition_started(level);
    let mid = n / 2;

    // Adaptive iteration count: large partitions converge faster with
    // coarse splits, so fewer refinement passes suffice. The fine-grained
    // clustering is handled by deeper recursion levels with full iterations.
    let effective_iters = if n > 100_000 {
        context.max_iters.min(12)
    } else {
        context.max_iters
    };

    // Compact term IDs permit direct indexing. Slots are initialized lazily so
    // deep partitions touch only their active terms. Coarse partitions use
    // multiple preallocated lanes; recursion splits the exact same workspace
    // set between children.
    build_term_degrees(docs, mid, context.fwd, degree_workspaces);
    // A full-vocabulary objective scan pays off only on coarse partitions,
    // where one avoided refinement skips millions of postings. At fine levels
    // it cost more than the work it could save, so retain the cheap gain
    // cooling condition there.
    let track_objective = n >= PARALLEL_BP_MIN_ENTITIES;
    if track_objective {
        // The old bitmap scan accumulated terms in ascending order. Sorting
        // once preserves that exact floating-point order while subsequent
        // objective evaluations visit only words active in this partition.
        degree_workspaces[0].sort_touched_words();
    }
    let mut previous_objective = if track_objective {
        degree_workspaces[0].bisection_objective(mid, n - mid, context.log_table)
    } else {
        0.0
    };
    let mut best_objective = previous_objective;
    let mut objective_stalls = 0usize;

    for iter in 0..effective_iters {
        // Anytime cutoff between refinement passes: keep the current split.
        #[cfg(feature = "native")]
        if let Some(dl) = context.deadline
            && std::time::Instant::now() >= dl
        {
            context
                .exhausted
                .store(true, std::sync::atomic::Ordering::Relaxed);
            break;
        }
        // Compute gain for each document (approx_1 from Dhulipala et al.)
        // Parallelized for large partitions where per-doc work dominates.
        compute_gains(
            docs,
            context.fwd,
            mid,
            &degree_workspaces[0],
            context.log_table,
            gains,
        );

        // Exact median selection and stable partition. The old path allocated
        // `Vec<usize>` (8 bytes/entity) and selected/applied it serially; the
        // radix path is O(4n), parallel, and accumulates moved-term deltas in
        // the same bounded worker lanes used by degree construction.
        let partition = partition_by_gain(
            docs,
            gains,
            mid,
            context.fwd,
            &mut degree_workspaces[1..],
            partitioned,
            ranked_scratch,
        );

        if partition.swap_count == 0 {
            context.progress.iteration(n, 0, 0.0, 0.0);
            // Preserve the selector's within-half order before recursing.
            // Small partitions intentionally retain quickselect's established
            // symmetry-breaking permutation.
            docs.copy_from_slice(partitioned);
            break;
        }

        match &partition.degree_update {
            PartitionDegreeUpdate::Moves => {
                let (degrees, movement_workspaces) = degree_workspaces
                    .split_first_mut()
                    .expect("BP always has one degree workspace");
                movement_workspaces
                    .first()
                    .expect("parallel BP movement update requires a spare workspace")
                    .apply_moves_to(degrees);
            }
            PartitionDegreeUpdate::Ranked => update_degrees_for_ranked_partition(
                docs,
                ranked_scratch,
                mid,
                context.fwd,
                &mut degree_workspaces[0],
            ),
            PartitionDegreeUpdate::Threshold {
                threshold_key,
                ties_left,
            } => update_degrees_for_threshold_partition(
                docs,
                gains,
                mid,
                *threshold_key,
                *ties_left,
                context.fwd,
                &mut degree_workspaces[0],
            ),
        }

        let (new_objective, objective_improvement, relative_improvement) = if track_objective {
            let new_objective =
                degree_workspaces[0].bisection_objective(mid, n - mid, context.log_table);
            let objective_improvement = new_objective - previous_objective;
            let relative_improvement = objective_improvement / previous_objective.abs().max(1.0);
            (new_objective, objective_improvement, relative_improvement)
        } else {
            (0.0, 0.0, 0.0)
        };
        context.progress.iteration(
            n,
            partition.swap_count,
            objective_improvement,
            relative_improvement,
        );

        // Keep the existing approximate-BP semantics: one median refinement
        // can temporarily reduce the exact objective before the reciprocal
        // move settles. Rejecting that first step made valid clustered inputs
        // no-op. Instead, accept refinements and stop only after a warm-up plus
        // consecutive iterations that fail to improve the best exact
        // objective by a meaningful relative amount.
        docs.copy_from_slice(partitioned);
        if track_objective {
            previous_objective = new_objective;
            let relative_best_improvement =
                (new_objective - best_objective) / best_objective.abs().max(1.0);
            if relative_best_improvement >= MIN_RELATIVE_OBJECTIVE_IMPROVEMENT {
                best_objective = new_objective;
                objective_stalls = 0;
            } else if iter + 1 >= MIN_OBJECTIVE_ITERATIONS {
                objective_stalls += 1;
            }
            if objective_stalls >= OBJECTIVE_STALL_ITERATIONS {
                context.progress.objective_stop();
                break;
            }
        }

        // Early termination: if < 0.5% of docs swapped, partition is stable
        if iter > 2 && partition.swap_count < n / 200 {
            break;
        }

        // Fine partitions retain the previous cheap cooling rule. Computing
        // an exact vocabulary objective here costs more than the posting work
        // it can avoid.
        if !track_objective && iter > 5 {
            let max_abs_gain = gains
                .iter()
                .copied()
                .fold(0.0f32, |max_gain, gain| max_gain.max(gain.abs()));
            if max_abs_gain < 0.001 {
                break;
            }
        }
    }

    context.progress.partition_finished();

    let (left, right) = docs.split_at_mut(mid);
    let (left_gains, right_gains) = gains.split_at_mut(mid);
    let (left_partitioned, right_partitioned) = partitioned.split_at_mut(mid);
    let (left_ranked, right_ranked) = ranked_scratch.split_at_mut(mid);
    #[cfg(feature = "native")]
    if degree_workspaces.len() > 1 {
        let left_lanes = degree_workspaces.len() / 2;
        let (left_workspaces, right_workspaces) = degree_workspaces.split_at_mut(left_lanes);
        rayon::join(
            || {
                bisect(
                    left,
                    left_gains,
                    left_partitioned,
                    left_ranked,
                    left_workspaces,
                    level + 1,
                    context,
                )
            },
            || {
                bisect(
                    right,
                    right_gains,
                    right_partitioned,
                    right_ranked,
                    right_workspaces,
                    level + 1,
                    context,
                )
            },
        );
    } else {
        // Gain computation inside each node remains parallel, so serializing
        // recursion here bounds vocabulary-sized degree arrays without leaving
        // the Rayon pool idle.
        bisect(
            left,
            left_gains,
            left_partitioned,
            left_ranked,
            degree_workspaces,
            level + 1,
            context,
        );
        bisect(
            right,
            right_gains,
            right_partitioned,
            right_ranked,
            degree_workspaces,
            level + 1,
            context,
        );
    }
    #[cfg(not(feature = "native"))]
    {
        bisect(
            left,
            left_gains,
            left_partitioned,
            left_ranked,
            degree_workspaces,
            level + 1,
            context,
        );
        bisect(
            right,
            right_gains,
            right_partitioned,
            right_ranked,
            degree_workspaces,
            level + 1,
            context,
        );
    }
}

/// Compute gains for all documents, parallelized via rayon for large partitions.
///
/// Each doc's gain is independent: iterate its terms, accumulate the log-gap
/// cost delta of moving it to the other side. Read-only access to degree arrays
/// makes this embarrassingly parallel.
#[inline(never)]
fn compute_gains(
    docs: &[u32],
    fwd: &ForwardIndex,
    mid: usize,
    degrees: &TermDegrees,
    log_table: &[f32],
    gains: &mut [f32],
) {
    // Single coherent key: HIGH = belongs in the RIGHT half.
    // Left docs get +approx_one(from=left, to=right) — a misplaced left doc
    // (terms concentrated right) scores high. Right docs get
    // -approx_one(from=right, to=left) — a misplaced right doc scores low.
    // This matches the reference two-sided formulation (compute_gains_left /
    // compute_gains_right with negation); ranking both halves by raw
    // "move gain" instead made both sides' misplaced docs rank identically,
    // so the partition step could never exchange them.
    let gain_for_doc = |i: usize| -> f32 {
        let doc = docs[i] as usize;
        let in_left = i < mid;
        let mut g = 0.0f32;
        for &term in fwd.doc_terms(doc) {
            let [left, right] = degrees.get(term as usize);
            let (from, to) = if in_left {
                (left, right)
            } else {
                (right, left)
            };
            let move_gain = fast_log2_lookup(to as usize + 2, log_table)
                - fast_log2_lookup(from as usize, log_table)
                - std::f32::consts::LOG2_E / (1.0 + to as f32);
            g += if in_left { move_gain } else { -move_gain };
        }
        g
    };

    #[cfg(feature = "native")]
    {
        if docs.len() > 4096 {
            gains
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, gain)| *gain = gain_for_doc(i));
        } else {
            for (i, gain) in gains.iter_mut().enumerate().take(docs.len()) {
                *gain = gain_for_doc(i);
            }
        }
    }
    #[cfg(not(feature = "native"))]
    {
        for (i, gain) in gains.iter_mut().enumerate().take(docs.len()) {
            *gain = gain_for_doc(i);
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Build precomputed log2 table for values 0..size.
fn build_log_table(size: usize) -> Vec<f32> {
    let mut table = vec![0.0f32; size];
    // log2(0) is undefined; use a large negative value
    table[0] = -10.0;
    for (i, entry) in table.iter_mut().enumerate().skip(1) {
        *entry = (i as f32).log2();
    }
    table
}

/// Fast log2 with precomputed table lookup.
#[inline]
fn fast_log2_lookup(val: usize, table: &[f32]) -> f32 {
    if val < table.len() {
        table[val]
    } else {
        (val as f32).log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_frequency_count_and_candidate_selection_are_exact() {
        let items: Vec<u32> = (0..10_000).collect();
        let frequencies = count_frequencies_bounded(&items, 7, 16 * 1024, |item, counts| {
            let term = *item as usize % counts.len();
            counts[term] += 1;
        })
        .unwrap();
        assert_eq!(frequencies.iter().sum::<u32>(), items.len() as u32);
        for (term, &frequency) in frequencies.iter().enumerate() {
            let expected = (term..items.len()).step_by(frequencies.len()).count() as u32;
            assert_eq!(frequency, expected);
        }

        let (selected, limited) =
            select_frequency_candidates(&[8, 3, 0, 5, 3], 1, 10, 2 * CANDIDATE_ENTRY_BYTES);
        assert!(limited);
        let mut selected = selected;
        selected.sort_unstable();
        assert_eq!(selected, vec![(1, 3), (4, 3)]);
    }

    #[test]
    fn bounded_frequency_count_rejects_an_undersized_budget() {
        assert!(
            count_frequencies_bounded(&[0u32], 32, 32, |_, _| {}).is_none(),
            "one complete dense frequency table must fit before counting"
        );
    }

    #[test]
    fn lazy_term_degrees_initialize_only_on_first_write() {
        let mut degrees = TermDegrees::new(130);
        let values_ptr = degrees.values.as_ptr();
        let bitmap_ptr = degrees.initialized.as_ptr();
        assert_eq!(degrees.get(65), [0, 0]);
        degrees.entry_mut(65)[0] += 3;
        degrees.entry_mut(65)[1] += 2;
        assert_eq!(degrees.get(65), [3, 2]);
        assert_eq!(degrees.get(64), [0, 0]);
        assert_eq!(
            degrees
                .initialized
                .iter()
                .map(|w| w.count_ones())
                .sum::<u32>(),
            1
        );

        degrees.reset();
        assert_eq!(degrees.values.as_ptr(), values_ptr);
        assert_eq!(degrees.initialized.as_ptr(), bitmap_ptr);
        assert_eq!(degrees.get(65), [0, 0]);
        assert!(degrees.touched_words.is_empty());
        degrees.entry_mut(129)[1] = 7;
        assert_eq!(degrees.get(129), [0, 7]);
        assert_eq!(degrees.get(65), [0, 0]);
    }

    #[test]
    fn sparse_objective_keeps_the_original_ascending_term_order() {
        let mut degrees = TermDegrees::new(130);
        *degrees.entry_mut(129) = [5, 2];
        *degrees.entry_mut(1) = [3, 7];
        *degrees.entry_mut(65) = [11, 13];
        assert_eq!(degrees.touched_words, vec![2, 0, 1]);
        degrees.sort_touched_words();
        assert_eq!(degrees.touched_words, vec![0, 1, 2]);

        let log_table = build_log_table(4096);
        let actual = degrees.bisection_objective(31, 32, &log_table);
        let side_log = [
            fast_log2_lookup(31, &log_table) as f64,
            fast_log2_lookup(32, &log_table) as f64,
        ];
        let mut original_bitmap_scan = 0.0f64;
        for (word_idx, &initialized) in degrees.initialized.iter().enumerate() {
            let mut pending = initialized;
            while pending != 0 {
                let bit = pending.trailing_zeros() as usize;
                let term = word_idx * 64 + bit;
                let [left, right] = degrees.get(term);
                for (side, count) in [left, right].into_iter().enumerate() {
                    if count > 0 {
                        original_bitmap_scan += count as f64
                            * (fast_log2_lookup(count as usize + 1, &log_table) as f64
                                - side_log[side]);
                    }
                }
                pending &= pending - 1;
            }
        }
        assert_eq!(actual.to_bits(), original_bitmap_scan.to_bits());
    }

    #[test]
    fn gain_radix_key_matches_total_cmp() {
        let values = [
            f32::from_bits(0xffc0_0001),
            f32::NEG_INFINITY,
            -42.0,
            -0.0,
            0.0,
            42.0,
            f32::INFINITY,
            f32::from_bits(0x7fc0_0001),
        ];
        let mut by_cmp = values;
        by_cmp.sort_by(f32::total_cmp);
        let mut by_key = values;
        by_key.sort_by_key(|value| gain_order_key(*value));
        assert_eq!(
            by_cmp.map(f32::to_bits),
            by_key.map(f32::to_bits),
            "radix selection must preserve the former total_cmp order"
        );
    }

    #[test]
    fn radix_threshold_matches_exact_rank_with_ties() {
        let gains = [3.0, -1.0, 7.0, -1.0, 0.0, -0.0, 3.0, 9.0, 3.0, 2.0, 2.0];
        let mut sorted: Vec<(u32, usize)> = gains
            .iter()
            .enumerate()
            .map(|(idx, &gain)| (gain_order_key(gain), idx))
            .collect();
        sorted.sort_unstable();

        for left_count in 1..=gains.len() {
            let (threshold, lower) = select_gain_threshold(&gains, left_count);
            assert_eq!(threshold, sorted[left_count - 1].0);
            assert_eq!(lower, sorted.partition_point(|&(key, _)| key < threshold),);
        }
    }

    #[cfg(feature = "native")]
    #[test]
    fn parallel_partition_matches_exact_selection_and_degree_rebuild() {
        const N: usize = PARALLEL_BP_MIN_ENTITIES + 1;
        const TERMS: usize = 101;
        let mut terms = Vec::with_capacity(N * 3);
        let mut offsets = Vec::with_capacity(N + 1);
        offsets.push(0);
        for doc in 0..N {
            terms.extend_from_slice(&[
                (doc % TERMS) as u32,
                ((doc / 7) % TERMS) as u32,
                ((doc * 13) % TERMS) as u32,
            ]);
            offsets.push(terms.len() as u64);
        }
        let fwd = ForwardIndex {
            terms,
            offsets,
            num_terms: TERMS,
            parallel_bisect_lanes: 4,
            budget_limited: false,
        };
        let docs: Vec<u32> = (0..N as u32)
            .map(|idx| ((idx as usize * 7_919) % N) as u32)
            .collect();
        let gains: Vec<f32> = docs
            .iter()
            .map(|&doc| ((doc as usize * 37) % 257) as f32 - 128.0)
            .collect();
        let mid = N / 2;

        let mut ranked: Vec<usize> = (0..N).collect();
        ranked.sort_unstable_by(|&left, &right| {
            gains[left]
                .total_cmp(&gains[right])
                .then_with(|| left.cmp(&right))
        });
        let mut selected_left = vec![false; N];
        for &idx in &ranked[..mid] {
            selected_left[idx] = true;
        }
        let expected: Vec<u32> = docs
            .iter()
            .enumerate()
            .filter(|(idx, _)| selected_left[*idx])
            .chain(
                docs.iter()
                    .enumerate()
                    .filter(|(idx, _)| !selected_left[*idx]),
            )
            .map(|(_, &doc)| doc)
            .collect();

        let mut output = vec![0; N];
        let mut ranked_scratch = Vec::new();
        let mut movement_workspaces: Vec<_> = (0..3).map(|_| TermDegrees::new(TERMS)).collect();
        let outcome = partition_by_gain(
            &docs,
            &gains,
            mid,
            &fwd,
            &mut movement_workspaces,
            &mut output,
            &mut ranked_scratch,
        );
        assert_eq!(output, expected);
        assert!(
            matches!(&outcome.degree_update, PartitionDegreeUpdate::Moves),
            "test must exercise parallel movement counts"
        );

        let mut updated_workspaces: Vec<_> = (0..4).map(|_| TermDegrees::new(TERMS)).collect();
        build_term_degrees(&docs, mid, &fwd, &mut updated_workspaces);
        movement_workspaces[0].apply_moves_to(&mut updated_workspaces[0]);

        let mut rebuilt_workspaces: Vec<_> = (0..4).map(|_| TermDegrees::new(TERMS)).collect();
        build_term_degrees(&output, mid, &fwd, &mut rebuilt_workspaces);
        for term in 0..TERMS {
            assert_eq!(
                updated_workspaces[0].get(term),
                rebuilt_workspaces[0].get(term),
                "parallel movement-count mismatch for term {term}"
            );
        }
    }

    /// Regression: CSR offsets were u32 and wrapped past 4.29B postings —
    /// a 58M-doc / ~85-dims-per-doc prod reorder pass (~4.9B postings)
    /// panicked with "mid > len" in the terms carving. The old 8 GB memory
    /// budget masked the overflow by dropping dims; raising the budget
    /// exposed it. Offsets must be u64.
    #[test]
    fn test_csr_offsets_do_not_wrap_past_u32() {
        let counts = [1_500_000_000u32; 3]; // 4.5B total > u32::MAX
        let offsets = build_csr_offsets(&counts);
        assert_eq!(
            offsets,
            vec![0, 1_500_000_000, 3_000_000_000, 4_500_000_000]
        );
        assert!(*offsets.last().unwrap() > u32::MAX as u64);
    }

    /// Build a simple forward index from (doc_id, terms) pairs.
    fn make_fwd(docs: &[&[u32]], num_terms: usize) -> ForwardIndex {
        let mut terms = Vec::new();
        let mut offsets = vec![0u64];
        for doc_terms in docs {
            terms.extend_from_slice(doc_terms);
            offsets.push(terms.len() as u64);
        }
        ForwardIndex {
            terms,
            offsets,
            num_terms,
            parallel_bisect_lanes: 1,
            budget_limited: false,
        }
    }

    #[test]
    fn test_bp_empty() {
        let fwd = ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
            parallel_bisect_lanes: 1,
            budget_limited: false,
        };
        let (perm, _) = graph_bisection(&fwd, 4, 20, BpBudget::full());
        assert!(perm.is_empty());
    }

    #[test]
    fn test_bp_small() {
        // 4 docs, min_partition_size=4 → no bisection, identity
        let fwd = make_fwd(&[&[0, 1], &[0, 2], &[1, 3], &[2, 3]], 4);
        let (perm, _) = graph_bisection(&fwd, 4, 20, BpBudget::full());
        assert_eq!(perm.len(), 4);
        // All docs present
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_bp_clusters() {
        // 8 docs in 2 clear clusters:
        // Cluster A (docs 0-3): share terms 0, 1
        // Cluster B (docs 4-7): share terms 2, 3
        let fwd = make_fwd(
            &[
                &[0, 1],
                &[0, 1],
                &[0, 1],
                &[0, 1],
                &[2, 3],
                &[2, 3],
                &[2, 3],
                &[2, 3],
            ],
            4,
        );
        let (perm, _) = graph_bisection(&fwd, 4, 20, BpBudget::full());
        assert_eq!(perm.len(), 8);

        // After bisection, docs from same cluster should be in same half
        let left: Vec<u32> = perm[..4].to_vec();

        // Either all of cluster A is in left and B in right, or vice versa
        let a_in_left = left.iter().filter(|&&d| d < 4).count();
        let b_in_left = left.iter().filter(|&&d| d >= 4).count();
        assert!(
            (a_in_left == 4 && b_in_left == 0) || (a_in_left == 0 && b_in_left == 4),
            "Clusters should be separated: a_left={}, b_left={}",
            a_in_left,
            b_in_left,
        );
    }

    #[test]
    fn test_bp_permutation_valid() {
        // 16 docs with mixed terms: terms range from 0..4 and 10..18
        let docs: Vec<Vec<u32>> = (0..16).map(|i| vec![i / 4, 10 + i / 2]).collect();
        let doc_refs: Vec<&[u32]> = docs.iter().map(|v| v.as_slice()).collect();
        let fwd = make_fwd(&doc_refs, 18); // max term = 10 + 15/2 = 17, so need 18
        let (perm, _) = graph_bisection(&fwd, 4, 20, BpBudget::full());

        assert_eq!(perm.len(), 16);
        // Must be a valid permutation
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<u32> = (0..16).collect();
        assert_eq!(sorted, expected);
    }

    /// Depth-capped BP: with min_partition_docs above the cluster size, only
    /// the top-level split happens — clusters still separate (coarse
    /// clustering), the permutation stays valid, and the pass converges
    /// (a depth cap is a chosen target, not an interruption).
    #[test]
    fn test_bp_depth_cap_separates_clusters_and_converges() {
        // Mostly-separated clusters with one misplaced doc per half — the
        // top-level swap pass must exchange docs 3 and 4.
        let fwd = make_fwd(
            &[
                &[0, 1],
                &[0, 1],
                &[0, 1],
                &[2, 3],
                &[0, 1],
                &[2, 3],
                &[2, 3],
                &[2, 3],
            ],
            4,
        );
        let budget = BpBudget {
            min_partition_docs: Some(4),
            time_budget: None,
        };
        let (perm, converged) = graph_bisection(&fwd, 2, 20, budget);
        assert!(converged, "depth cap must report converged");
        assert_eq!(perm.len(), 8);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            (0..8).collect::<Vec<u32>>(),
            "must stay a valid permutation"
        );
        // Top-level split separates the clusters (docs {0,1,2,4} share terms
        // 0/1; docs {3,5,6,7} share terms 2/3)
        let cluster_a = [0u32, 1, 2, 4];
        let a_in_left = perm[..4].iter().filter(|d| cluster_a.contains(d)).count();
        assert!(
            a_in_left == 4 || a_in_left == 0,
            "clusters should separate at the top level: {:?}",
            perm
        );
    }

    /// Zero wall-clock budget: the pass ends immediately, reports
    /// converged=false, and still emits a valid (identity) permutation.
    #[test]
    fn test_bp_zero_time_budget_emits_valid_partial_permutation() {
        let docs: Vec<Vec<u32>> = (0..64).map(|i| vec![i % 4]).collect();
        let doc_refs: Vec<&[u32]> = docs.iter().map(|v| v.as_slice()).collect();
        let fwd = make_fwd(&doc_refs, 4);
        let budget = BpBudget {
            min_partition_docs: None,
            time_budget: Some(std::time::Duration::ZERO),
        };
        let (perm, converged) = graph_bisection(&fwd, 4, 20, budget);
        assert!(!converged, "zero budget must report unconverged");
        assert_eq!(perm.len(), 64);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..64).collect::<Vec<u32>>());
    }

    #[test]
    fn test_memory_limited_graph_never_reports_converged() {
        let mut fwd = make_fwd(&[&[0], &[0], &[1], &[1]], 2);
        fwd.budget_limited = true;

        let (perm, converged) = graph_bisection(&fwd, 2, 20, BpBudget::full());

        assert!(!converged);
        let mut sorted = perm;
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_fast_log2() {
        let table = build_log_table(4096);
        assert!((table[1] - 0.0).abs() < 0.001);
        assert!((table[2] - 1.0).abs() < 0.001);
        assert!((table[4] - 2.0).abs() < 0.001);
        assert!((table[1024] - 10.0).abs() < 0.001);
        // Fallback for values beyond table
        let val = fast_log2_lookup(8192, &table);
        assert!((val - 13.0).abs() < 0.001);
    }
}
