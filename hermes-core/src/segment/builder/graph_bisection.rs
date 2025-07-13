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

fn term_degree_bytes(num_terms: usize) -> usize {
    num_terms
        .saturating_mul(TERM_DEGREE_VALUE_BYTES)
        .saturating_add(num_terms.div_ceil(64).saturating_mul(8))
}

fn parallel_bisect_depth(
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
    affordable_nodes.min(worker_limit).ilog2() as usize
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
}

impl TermDegrees {
    fn new(num_terms: usize) -> Self {
        let mut values = Vec::with_capacity(num_terms);
        values.resize_with(num_terms, std::mem::MaybeUninit::uninit);
        Self {
            values,
            initialized: vec![0; num_terms.div_ceil(64)],
        }
    }

    #[inline]
    fn entry_mut(&mut self, term: usize) -> &mut [u32; 2] {
        let word = term / 64;
        let mask = 1u64 << (term % 64);
        if self.initialized[word] & mask == 0 {
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
    /// Maximum recursion depth at which both children may own a vocabulary-
    /// sized degree array concurrently. Deeper partitions still use Rayon for
    /// gain computation, but recurse serially to honor the memory budget.
    parallel_bisect_depth: usize,
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
    for (vid, (slot, chunk)) in virtual_to_real
        .iter_mut()
        .zip(ids.as_chunks::<4>().0)
        .enumerate()
    {
        let doc_id = u32::from_le_bytes(*chunk);
        if doc_id != u32::MAX {
            if real_to_virtual.len() == expected_real {
                return Err(crate::Error::Corruption(format!(
                    "BMP document map contains more than the footer's {expected_real} real slots"
                )));
            }
            *slot = real_to_virtual.len() as u32;
            real_to_virtual.push(vid as u32);
        }
    }
    if real_to_virtual.len() != expected_real {
        return Err(crate::Error::Corruption(format!(
            "BMP document map has {} real slots but footer declares {expected_real}",
            real_to_virtual.len(),
        )));
    }
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
/// map via [`build_vid_maps`]. Returns `(forward_index, per_source_real_doc_counts)`.
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
) -> crate::Result<(ForwardIndex, Vec<usize>)> {
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
) -> (ForwardIndex, Vec<usize>) {
    debug_assert_eq!(bmps.len(), vid_maps.len());
    let source_doc_counts: Vec<usize> = vid_maps.iter().map(|(_, r2v)| r2v.len()).collect();
    let total_docs: usize = source_doc_counts.iter().sum();

    if total_docs == 0 {
        return (
            ForwardIndex {
                terms: Vec::new(),
                offsets: Vec::new(),
                num_terms: 0,
                parallel_bisect_depth: 0,
                budget_limited: false,
            },
            source_doc_counts,
        );
    }

    // Job list: one entry per (source, block). Real ids are assigned in
    // ascending vid order (see build_vid_maps), so each block owns a
    // contiguous real-id range — every phase below can process blocks in
    // parallel, writing disjoint slices.
    let jobs = build_block_jobs(bmps, vid_maps);

    // Phase 1: count doc frequency in one dense atomic table. The previous
    // Rayon fold built a vocabulary-sized hash map per worker before the
    // budget check, multiplying peak memory by the CPU count.
    let max_dims = bmps
        .iter()
        .map(|bmp| bmp.dims() as usize)
        .max()
        .unwrap_or(0);
    let jobs_bytes = jobs
        .len()
        .saturating_mul(std::mem::size_of::<BlockJob>().saturating_add(40));
    let frequency_bytes =
        max_dims.saturating_mul(std::mem::size_of::<std::sync::atomic::AtomicU32>());
    if frequency_bytes > memory_budget_bytes.saturating_sub(jobs_bytes) {
        log::warn!(
            "[reorder] memory budget {:.0} MB cannot hold the {:.0} MB dimension-frequency table; using identity order",
            memory_budget_bytes as f64 / (1024.0 * 1024.0),
            frequency_bytes as f64 / (1024.0 * 1024.0),
        );
        return (
            ForwardIndex {
                terms: Vec::new(),
                offsets: Vec::new(),
                num_terms: 0,
                parallel_bisect_depth: 0,
                budget_limited: true,
            },
            source_doc_counts,
        );
    }
    let dim_df: Vec<std::sync::atomic::AtomicU32> = (0..max_dims)
        .map(|_| std::sync::atomic::AtomicU32::new(0))
        .collect();
    let count_block_df = |job: &BlockJob| {
        let bmp = bmps[job.src as usize];
        let (v2r, _) = &vid_maps[job.src as usize];
        let block_size = bmp.bmp_block_size as usize;
        for (dim_id, postings) in bmp.iter_block_terms(job.block_id) {
            let mut n = 0usize;
            for p in postings {
                let vid = job.block_id as usize * block_size + p.local_slot as usize;
                if v2r[vid] != u32::MAX && p.impact > 0 {
                    n += 1;
                }
            }
            if n > 0
                && let Some(count) = dim_df.get(dim_id as usize)
            {
                count.fetch_add(n as u32, std::sync::atomic::Ordering::Relaxed);
            }
        }
    };
    #[cfg(feature = "native")]
    jobs.par_iter().for_each(count_block_df);
    #[cfg(not(feature = "native"))]
    jobs.iter().for_each(count_block_df);

    // Retain the lowest-frequency candidates in a bounded heap while the
    // frequency table is live. This makes candidate discovery itself obey the
    // configured limit even for extremely large vocabularies.
    let eligible_candidate_count = dim_df
        .iter()
        .filter(|df| {
            let df = df.load(std::sync::atomic::Ordering::Relaxed) as usize;
            df >= min_doc_freq && df <= max_doc_freq
        })
        .count();
    let candidate_capacity = memory_budget_bytes
        .saturating_sub(jobs_bytes)
        .saturating_sub(frequency_bytes)
        .checked_div(std::mem::size_of::<(usize, u32)>())
        .unwrap_or(0)
        .min(eligible_candidate_count);
    let mut candidate_heap = std::collections::BinaryHeap::with_capacity(candidate_capacity);
    for (dim_id, df) in dim_df.iter().enumerate() {
        let df = df.load(std::sync::atomic::Ordering::Relaxed) as usize;
        if df < min_doc_freq || df > max_doc_freq {
            continue;
        }
        let candidate = (df, dim_id as u32);
        if candidate_heap.len() < candidate_capacity {
            candidate_heap.push(candidate);
        } else if candidate_capacity > 0 && candidate < *candidate_heap.peek().unwrap() {
            candidate_heap.pop();
            candidate_heap.push(candidate);
        }
    }
    drop(dim_df);
    let mut eligible: Vec<(u32, usize)> = candidate_heap
        .into_vec()
        .into_iter()
        .map(|(df, dim_id)| (dim_id, df))
        .collect();
    let mut budget_limited = eligible.len() < eligible_candidate_count;

    // Memory budget: estimate forward index + bisection scratch.
    // Includes jobs/slice descriptors, dense remap, all per-document scratch,
    // and at least one exact TermDegrees allocation.
    let total_postings_est = eligible
        .iter()
        .fold(0usize, |total, (_, df)| total.saturating_add(*df));
    let entity_scratch_bytes = total_docs.saturating_mul(32);
    let remap_bytes = max_dims.saturating_mul(4);
    let fixed_bytes = entity_scratch_bytes
        .saturating_add(remap_bytes)
        .saturating_add(jobs_bytes);
    let estimated_bytes = total_postings_est
        .saturating_mul(4)
        .saturating_add(fixed_bytes)
        // Candidate metadata coexists with the dense remap until construction
        // starts; omitting it let a huge rare-term vocabulary exceed the cap.
        .saturating_add(eligible.len().saturating_mul(CANDIDATE_ENTRY_BYTES))
        .saturating_add(term_degree_bytes(eligible.len()));

    if estimated_bytes > memory_budget_bytes && !eligible.is_empty() {
        // Sort by df ascending — keep discriminative low-df dims first,
        // drop highest-df dims which contribute the most postings.
        eligible.sort_by_key(|&(_, df)| df);

        // Account for each retained term together with its postings. The old
        // calculation charged the eight-byte degree slot for every candidate
        // before deciding how many to retain; a large rare-term vocabulary
        // could therefore make the target zero even when a useful subset fit.
        let mut used_bytes = fixed_bytes;
        let mut cum = 0usize;
        let mut keep_count = 0;
        for &(_, df) in &eligible {
            let term_bytes = df
                .saturating_mul(4)
                .saturating_add(TERM_DEGREE_VALUE_BYTES + 1)
                .saturating_add(CANDIDATE_ENTRY_BYTES);
            if term_bytes > memory_budget_bytes.saturating_sub(used_bytes) {
                break;
            }
            used_bytes = used_bytes.saturating_add(term_bytes);
            cum = cum.saturating_add(df);
            keep_count += 1;
        }

        let dropped = eligible.len() - keep_count;
        eligible.truncate(keep_count);
        budget_limited |= dropped > 0;

        log::warn!(
            "[reorder] memory budget {:.0} MB: estimated {:.0} MB, dropped {} highest-df dims, keeping {} ({} postings)",
            memory_budget_bytes as f64 / (1024.0 * 1024.0),
            estimated_bytes as f64 / (1024.0 * 1024.0),
            dropped,
            keep_count,
            cum,
        );
    }

    if eligible.is_empty() {
        // The caller emits an identity permutation when there is no graph
        // signal. Avoid allocating per-document counts and u64 offsets only
        // to discover that the terms array is empty, especially when the
        // configured budget is below the fixed document scratch cost.
        return (
            ForwardIndex {
                terms: Vec::new(),
                offsets: Vec::new(),
                num_terms: 0,
                parallel_bisect_depth: 0,
                budget_limited,
            },
            source_doc_counts,
        );
    }

    let mut term_remap = vec![u32::MAX; max_dims];
    for (compact_id, &(dim_id, _)) in eligible.iter().enumerate() {
        term_remap[dim_id as usize] = compact_id as u32;
    }
    let num_active_terms = eligible.len();
    let retained_postings = eligible
        .iter()
        .fold(0usize, |total, (_, df)| total.saturating_add(*df));
    let non_degree_bytes = fixed_bytes.saturating_add(retained_postings.saturating_mul(4));
    let parallel_bisect_depth =
        parallel_bisect_depth(memory_budget_bytes, non_degree_bytes, num_active_terms);
    drop(eligible);

    // Phase 2: count terms per doc (filtered) — per-block disjoint slices
    let mut counts = vec![0u32; total_docs];
    let fill_block_counts = |job: &BlockJob, out: &mut [u32]| {
        let bmp = bmps[job.src as usize];
        let (v2r, _) = &vid_maps[job.src as usize];
        let block_size = bmp.bmp_block_size as usize;
        for (dim_id, postings) in bmp.iter_block_terms(job.block_id) {
            if term_remap.get(dim_id as usize).copied().unwrap_or(u32::MAX) == u32::MAX {
                continue;
            }
            for p in postings {
                let vid = job.block_id as usize * block_size + p.local_slot as usize;
                let real = v2r[vid];
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
        for (dim_id, postings) in bmp.iter_block_terms(job.block_id) {
            let compact = term_remap.get(dim_id as usize).copied().unwrap_or(u32::MAX);
            if compact == u32::MAX {
                continue;
            }
            for p in postings {
                let vid = job.block_id as usize * block_size + p.local_slot as usize;
                let real = v2r[vid];
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

    (
        ForwardIndex {
            terms,
            offsets,
            num_terms: num_active_terms,
            parallel_bisect_depth,
            budget_limited,
        },
        source_doc_counts,
    )
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
            parallel_bisect_depth: 0,
            budget_limited: false,
        };
    }

    // (source, block) pairs in global block order — the parallel unit.
    let blocks: Vec<(u32, u32)> = bmps
        .iter()
        .enumerate()
        .flat_map(|(src, bmp)| (0..bmp.num_blocks).map(move |b| (src as u32, b)))
        .collect();

    // Phase 1: one bounded dense frequency table, shared by every worker.
    let max_dims = bmps
        .iter()
        .map(|bmp| bmp.dims() as usize)
        .max()
        .unwrap_or(0);
    let blocks_bytes = blocks
        .len()
        .saturating_mul(std::mem::size_of::<(u32, u32)>().saturating_add(32));
    let frequency_bytes =
        max_dims.saturating_mul(std::mem::size_of::<std::sync::atomic::AtomicU32>());
    if frequency_bytes > memory_budget_bytes.saturating_sub(blocks_bytes) {
        log::warn!(
            "[reorder] block-level frequency table exceeds memory budget; using identity order"
        );
        return ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
            parallel_bisect_depth: 0,
            budget_limited: true,
        };
    }
    let dim_bf: Vec<std::sync::atomic::AtomicU32> = (0..max_dims)
        .map(|_| std::sync::atomic::AtomicU32::new(0))
        .collect();
    let count_block_bf = |&(src, block_id): &(u32, u32)| {
        for (dim_id, _) in bmps[src as usize].iter_block_terms(block_id) {
            if let Some(count) = dim_bf.get(dim_id as usize) {
                count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    };
    #[cfg(feature = "native")]
    blocks.par_iter().for_each(count_block_bf);
    #[cfg(not(feature = "native"))]
    blocks.iter().for_each(count_block_bf);

    let max_bf = (total_blocks as f64 * 0.9) as usize;
    let eligible_candidate_count = dim_bf
        .iter()
        .filter(|bf| {
            let bf = bf.load(std::sync::atomic::Ordering::Relaxed) as usize;
            bf >= 2 && bf <= max_bf.max(2)
        })
        .count();
    let candidate_capacity = memory_budget_bytes
        .saturating_sub(blocks_bytes)
        .saturating_sub(frequency_bytes)
        .checked_div(std::mem::size_of::<(usize, u32)>())
        .unwrap_or(0)
        .min(eligible_candidate_count);
    let mut candidate_heap = std::collections::BinaryHeap::with_capacity(candidate_capacity);
    for (dim_id, bf) in dim_bf.iter().enumerate() {
        let bf = bf.load(std::sync::atomic::Ordering::Relaxed) as usize;
        if bf < 2 || bf > max_bf.max(2) {
            continue;
        }
        let candidate = (bf, dim_id as u32);
        if candidate_heap.len() < candidate_capacity {
            candidate_heap.push(candidate);
        } else if candidate_capacity > 0 && candidate < *candidate_heap.peek().unwrap() {
            candidate_heap.pop();
            candidate_heap.push(candidate);
        }
    }
    drop(dim_bf);
    let mut eligible: Vec<(u32, usize)> = candidate_heap
        .into_vec()
        .into_iter()
        .map(|(bf, dim_id)| (dim_id, bf))
        .collect();
    let mut budget_limited = eligible.len() < eligible_candidate_count;

    let total_postings_est = eligible
        .iter()
        .fold(0usize, |total, (_, bf)| total.saturating_add(*bf));
    let entity_scratch_bytes = total_blocks.saturating_mul(32);
    let remap_bytes = max_dims.saturating_mul(4);
    let fixed_bytes = entity_scratch_bytes
        .saturating_add(remap_bytes)
        .saturating_add(blocks_bytes);
    let estimated_bytes = total_postings_est
        .saturating_mul(4)
        .saturating_add(fixed_bytes)
        .saturating_add(eligible.len().saturating_mul(CANDIDATE_ENTRY_BYTES))
        .saturating_add(term_degree_bytes(eligible.len()));
    if estimated_bytes > memory_budget_bytes && !eligible.is_empty() {
        eligible.sort_by_key(|&(_, bf)| bf);
        let mut used_bytes = fixed_bytes;
        let mut cum = 0usize;
        let mut keep = 0;
        for &(_, bf) in &eligible {
            let term_bytes = bf
                .saturating_mul(4)
                .saturating_add(TERM_DEGREE_VALUE_BYTES + 1)
                .saturating_add(CANDIDATE_ENTRY_BYTES);
            if term_bytes > memory_budget_bytes.saturating_sub(used_bytes) {
                break;
            }
            used_bytes = used_bytes.saturating_add(term_bytes);
            cum = cum.saturating_add(bf);
            keep += 1;
        }
        let dropped = eligible.len() - keep;
        budget_limited |= dropped > 0;
        log::warn!(
            "[reorder] block-level fwd index over budget — dropped {} highest-bf dims",
            dropped,
        );
        eligible.truncate(keep);
    }

    if eligible.is_empty() {
        return ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
            parallel_bisect_depth: 0,
            budget_limited,
        };
    }

    let mut term_remap = vec![u32::MAX; max_dims];
    for (compact, &(dim_id, _)) in eligible.iter().enumerate() {
        term_remap[dim_id as usize] = compact as u32;
    }
    let num_terms = eligible.len();
    let retained_postings = eligible
        .iter()
        .fold(0usize, |total, (_, bf)| total.saturating_add(*bf));
    let non_degree_bytes = fixed_bytes.saturating_add(retained_postings.saturating_mul(4));
    let parallel_bisect_depth =
        parallel_bisect_depth(memory_budget_bytes, non_degree_bytes, num_terms);
    drop(eligible);

    // Phase 2+3: counts and CSR fill — one entity per block, so each block
    // maps to a single count cell and a contiguous terms range.
    let count_remapped = |&(src, block_id): &(u32, u32)| -> u32 {
        bmps[src as usize]
            .iter_block_terms(block_id)
            .filter(|(dim_id, _)| {
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
        for (dim_id, _) in bmps[src as usize].iter_block_terms(block_id) {
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
        parallel_bisect_depth,
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

/// Recursive graph bisection. Returns `(perm, converged)` where
/// `perm[new_pos] = old_index` and `converged` is false iff the wall-clock
/// budget ended the pass before it finished (a depth cap alone is a chosen
/// target, not an interruption — it reports converged).
///
/// `min_partition_size` should be the BMP block_size (64).
/// `max_iters` controls convergence (20 is standard).
///
/// Term IDs in the forward index must be compact (0..num_terms) so we can
/// use flat arrays for O(1) degree lookups instead of hash maps.
pub(crate) fn graph_bisection(
    fwd: &ForwardIndex,
    min_partition_size: usize,
    max_iters: usize,
    budget: BpBudget,
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
    };
    #[cfg(feature = "native")]
    bisect(&mut docs, fwd.parallel_bisect_depth, &context);
    #[cfg(not(feature = "native"))]
    bisect(&mut docs, 0, &context);

    let converged = !fwd.budget_limited && !exhausted.load(std::sync::atomic::Ordering::Relaxed);
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
}

fn bisect(docs: &mut [u32], parallel_depth: usize, context: &BisectContext<'_>) {
    #[cfg(not(feature = "native"))]
    let _ = parallel_depth;
    let n = docs.len();
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

    let mid = n / 2;
    let nt = context.fwd.num_terms;

    // Adaptive iteration count: large partitions converge faster with
    // coarse splits, so fewer refinement passes suffice. The fine-grained
    // clustering is handled by deeper recursion levels with full iterations.
    let effective_iters = if n > 100_000 {
        context.max_iters.min(12)
    } else {
        context.max_iters
    };

    // Compact term IDs permit direct indexing. Slots are initialized lazily so
    // deep partitions do not zero the full vocabulary on every recursive node.
    let mut degrees = TermDegrees::new(nt);

    for (i, &doc) in docs.iter().enumerate() {
        let side = usize::from(i >= mid);
        for &term in context.fwd.doc_terms(doc as usize) {
            degrees.entry_mut(term as usize)[side] += 1;
        }
    }

    // Scratch buffers reused across iterations
    let mut gains: Vec<f32> = vec![0.0; n];
    let mut indices: Vec<usize> = (0..n).collect();
    let mut new_left: Vec<u32> = Vec::with_capacity(mid);
    let mut new_right: Vec<u32> = Vec::with_capacity(n - mid);

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
            &degrees,
            context.log_table,
            &mut gains,
        );

        // Partition: the `mid` LOWEST keys (strongest left affinity) go left
        indices.clear();
        indices.extend(0..n);
        indices.select_nth_unstable_by(mid, |&a, &b| {
            gains[a].total_cmp(&gains[b]).then_with(|| a.cmp(&b))
        });

        // Apply partition, update degree arrays for swapped docs
        new_left.clear();
        new_right.clear();
        let mut swap_count: usize = 0;

        for (rank, &idx) in indices.iter().enumerate() {
            let doc = docs[idx];
            let was_left = idx < mid;
            let now_left = rank < mid;

            if now_left {
                new_left.push(doc);
            } else {
                new_right.push(doc);
            }

            if was_left != now_left {
                swap_count += 1;
                for &term in context.fwd.doc_terms(doc as usize) {
                    let degree = degrees.entry_mut(term as usize);
                    if was_left {
                        degree[0] -= 1;
                        degree[1] += 1;
                    } else {
                        degree[1] -= 1;
                        degree[0] += 1;
                    }
                }
            }
        }

        docs[..mid].copy_from_slice(&new_left);
        docs[mid..].copy_from_slice(&new_right);

        if swap_count == 0 {
            break;
        }

        // Early termination: if < 0.5% of docs swapped, partition is stable
        if iter > 2 && swap_count < n / 200 {
            break;
        }

        // Cooling: break early if gains are negligible
        if iter > 5 {
            let max_abs_gain = gains
                .iter()
                .copied()
                .fold(0.0f32, |max_gain, gain| max_gain.max(gain.abs()));
            if max_abs_gain < 0.001 {
                break;
            }
        }
    }

    // Drop scratch before recursion to free memory for sub-problems
    drop(degrees);
    drop(gains);
    drop(indices);
    drop(new_left);
    drop(new_right);

    let (left, right) = docs.split_at_mut(mid);
    #[cfg(feature = "native")]
    if parallel_depth > 0 {
        rayon::join(
            || bisect(left, parallel_depth - 1, context),
            || bisect(right, parallel_depth - 1, context),
        );
    } else {
        // Gain computation inside each node remains parallel, so serializing
        // recursion here bounds vocabulary-sized degree arrays without leaving
        // the Rayon pool idle.
        bisect(left, 0, context);
        bisect(right, 0, context);
    }
    #[cfg(not(feature = "native"))]
    {
        bisect(left, 0, context);
        bisect(right, 0, context);
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
    fn lazy_term_degrees_initialize_only_on_first_write() {
        let mut degrees = TermDegrees::new(130);
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
            parallel_bisect_depth: 0,
            budget_limited: false,
        }
    }

    #[test]
    fn test_bp_empty() {
        let fwd = ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
            parallel_bisect_depth: 0,
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
