//! Recursive Graph Bisection (BP) for BMP document ordering.
//!
//! Based on Dhulipala et al. (KDD 2016) and Mackenzie et al. — the same
//! algorithm used in Lucene and PISA for document reordering.
//!
//! Directly optimizes log-gap cost: docs sharing dimensions end up in the
//! same BMP blocks, producing tight upper bounds and effective pruning.
//!
//! Memory: forward index ~200 bytes/doc (temporary) + degree arrays ~840 KB.

#[cfg(feature = "native")]
use rayon::prelude::*;
use rustc_hash::FxHashMap;

// ── Forward index (CSR) ──────────────────────────────────────────────────

/// Forward index in CSR format: doc `d`'s terms are `terms[offsets[d]..offsets[d+1]]`.
///
/// Term IDs are remapped to compact range `0..num_terms` for flat-array degree tracking.
pub(crate) struct ForwardIndex {
    terms: Vec<u32>,
    offsets: Vec<u32>,
    pub num_terms: usize,
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
    pub fn total_postings(&self) -> u32 {
        self.offsets.last().copied().unwrap_or(0)
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
pub(crate) fn build_vid_maps(bmp: &crate::segment::reader::bmp::BmpIndex) -> (Vec<u32>, Vec<u32>) {
    let ids = bmp.doc_map_ids_slice();
    let num_virtual = bmp.num_virtual_docs as usize;
    let mut virtual_to_real = vec![u32::MAX; num_virtual];
    let mut real_to_virtual = Vec::with_capacity(bmp.num_real_docs() as usize);
    for (vid, (slot, chunk)) in virtual_to_real
        .iter_mut()
        .zip(ids.as_chunks::<4>().0)
        .enumerate()
    {
        let doc_id = u32::from_le_bytes(*chunk);
        if doc_id != u32::MAX {
            *slot = real_to_virtual.len() as u32;
            real_to_virtual.push(vid as u32);
        }
    }
    if real_to_virtual.len() != bmp.num_real_docs() as usize {
        log::warn!(
            "[reorder] BMP doc map has {} real slots but footer says num_real_docs={} — trusting the doc map",
            real_to_virtual.len(),
            bmp.num_real_docs(),
        );
    }
    (virtual_to_real, real_to_virtual)
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
pub(crate) fn build_forward_index_from_bmps(
    bmps: &[&crate::segment::reader::bmp::BmpIndex],
    min_doc_freq: usize,
    max_doc_freq: usize,
    memory_budget_bytes: usize,
) -> (ForwardIndex, Vec<usize>) {
    // Per-source virtual→real maps (padding-aware realness from the doc map).
    let vid_maps: Vec<(Vec<u32>, Vec<u32>)> = bmps.iter().map(|b| build_vid_maps(b)).collect();
    let source_doc_counts: Vec<usize> = vid_maps.iter().map(|(_, r2v)| r2v.len()).collect();
    let total_docs: usize = source_doc_counts.iter().sum();

    if total_docs == 0 {
        return (
            ForwardIndex {
                terms: Vec::new(),
                offsets: Vec::new(),
                num_terms: 0,
            },
            source_doc_counts,
        );
    }

    // Phase 1: count doc freq per dimension across all sources + assign compact IDs
    let mut dim_df: FxHashMap<u32, usize> = FxHashMap::default();
    for (bmp, (v2r, _)) in bmps.iter().zip(&vid_maps) {
        let num_blocks = bmp.num_blocks as usize;
        let block_size = bmp.bmp_block_size as usize;
        for block_id in 0..num_blocks {
            for (dim_id, postings) in bmp.iter_block_terms(block_id as u32) {
                for p in postings {
                    let vid = block_id * block_size + p.local_slot as usize;
                    if v2r[vid] != u32::MAX && p.impact > 0 {
                        *dim_df.entry(dim_id).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Filter dims by [min_doc_freq, max_doc_freq] range
    let mut eligible: Vec<(u32, usize)> = dim_df
        .iter()
        .filter(|&(_, df)| *df >= min_doc_freq && *df <= max_doc_freq)
        .map(|(&dim_id, &df)| (dim_id, df))
        .collect();
    drop(dim_df);

    // Memory budget: estimate forward index + bisection scratch.
    // Peak ≈ 4*total_postings (terms array) + 28*total_docs (offsets, counts,
    //   docs, gains, indices, new_left, new_right) + 8*num_terms (degree arrays).
    let total_postings_est: usize = eligible.iter().map(|(_, df)| *df).sum();
    let estimated_bytes = total_postings_est * 4 + total_docs * 28 + eligible.len() * 8;

    if estimated_bytes > memory_budget_bytes && !eligible.is_empty() {
        // Sort by df ascending — keep discriminative low-df dims first,
        // drop highest-df dims which contribute the most postings.
        eligible.sort_by_key(|&(_, df)| df);

        let target_postings =
            memory_budget_bytes.saturating_sub(total_docs * 28 + eligible.len() * 8) / 4;
        let mut cum = 0usize;
        let mut keep_count = 0;
        for &(_, df) in &eligible {
            if cum + df > target_postings {
                break;
            }
            cum += df;
            keep_count += 1;
        }

        let dropped = eligible.len() - keep_count;
        eligible.truncate(keep_count);

        log::warn!(
            "[reorder] memory budget {:.0} MB: estimated {:.0} MB, dropped {} highest-df dims, keeping {} ({} postings)",
            memory_budget_bytes as f64 / (1024.0 * 1024.0),
            estimated_bytes as f64 / (1024.0 * 1024.0),
            dropped,
            keep_count,
            cum,
        );
    }

    let mut term_remap: FxHashMap<u32, u32> = FxHashMap::default();
    for &(dim_id, _) in &eligible {
        let compact_id = term_remap.len() as u32;
        term_remap.insert(dim_id, compact_id);
    }
    let num_active_terms = term_remap.len();
    drop(eligible);

    // Phase 2: count terms per doc (filtered)
    let mut counts = vec![0u32; total_docs];
    let mut real_offset = 0usize;

    for (src_idx, bmp) in bmps.iter().enumerate() {
        let (v2r, _) = &vid_maps[src_idx];
        let num_blocks = bmp.num_blocks as usize;
        let block_size = bmp.bmp_block_size as usize;
        for block_id in 0..num_blocks {
            for (dim_id, postings) in bmp.iter_block_terms(block_id as u32) {
                if !term_remap.contains_key(&dim_id) {
                    continue;
                }
                for p in postings {
                    let vid = block_id * block_size + p.local_slot as usize;
                    let real = v2r[vid];
                    if real != u32::MAX && p.impact > 0 {
                        counts[real_offset + real as usize] += 1;
                    }
                }
            }
        }
        real_offset += source_doc_counts[src_idx];
    }

    // Phase 3: build CSR offsets
    let mut offsets = Vec::with_capacity(total_docs + 1);
    offsets.push(0u32);
    for &c in &counts {
        offsets.push(offsets.last().unwrap() + c);
    }
    let total = *offsets.last().unwrap() as usize;

    // Phase 4: fill terms (compact IDs)
    let mut terms = vec![0u32; total];
    counts.fill(0);
    real_offset = 0;

    for (src_idx, bmp) in bmps.iter().enumerate() {
        let (v2r, _) = &vid_maps[src_idx];
        let num_blocks = bmp.num_blocks as usize;
        let block_size = bmp.bmp_block_size as usize;
        for block_id in 0..num_blocks {
            for (dim_id, postings) in bmp.iter_block_terms(block_id as u32) {
                let Some(&compact) = term_remap.get(&dim_id) else {
                    continue;
                };
                for p in postings {
                    let vid = block_id * block_size + p.local_slot as usize;
                    let real = v2r[vid];
                    if real != u32::MAX && p.impact > 0 {
                        let global_real = real_offset + real as usize;
                        let pos = offsets[global_real] as usize + counts[global_real] as usize;
                        terms[pos] = compact;
                        counts[global_real] += 1;
                    }
                }
            }
        }
        real_offset += source_doc_counts[src_idx];
    }

    (
        ForwardIndex {
            terms,
            offsets,
            num_terms: num_active_terms,
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
        };
    }

    // Phase 1: dim → number of blocks containing it (block-level df)
    let mut dim_bf: FxHashMap<u32, usize> = FxHashMap::default();
    for bmp in bmps {
        for block_id in 0..bmp.num_blocks {
            for (dim_id, _) in bmp.iter_block_terms(block_id) {
                *dim_bf.entry(dim_id).or_insert(0) += 1;
            }
        }
    }

    let max_bf = (total_blocks as f64 * 0.9) as usize;
    let mut eligible: Vec<(u32, usize)> = dim_bf
        .iter()
        .filter(|&(_, bf)| *bf >= 2 && *bf <= max_bf.max(2))
        .map(|(&dim_id, &bf)| (dim_id, bf))
        .collect();
    drop(dim_bf);

    let total_postings_est: usize = eligible.iter().map(|(_, bf)| *bf).sum();
    let estimated_bytes = total_postings_est * 4 + total_blocks * 28 + eligible.len() * 8;
    if estimated_bytes > memory_budget_bytes && !eligible.is_empty() {
        eligible.sort_by_key(|&(_, bf)| bf);
        let target = memory_budget_bytes.saturating_sub(total_blocks * 28 + eligible.len() * 8) / 4;
        let mut cum = 0usize;
        let mut keep = 0;
        for &(_, bf) in &eligible {
            if cum + bf > target {
                break;
            }
            cum += bf;
            keep += 1;
        }
        log::warn!(
            "[reorder] block-level fwd index over budget — dropped {} highest-bf dims",
            eligible.len() - keep,
        );
        eligible.truncate(keep);
    }

    let mut term_remap: FxHashMap<u32, u32> = FxHashMap::default();
    for &(dim_id, _) in &eligible {
        let compact = term_remap.len() as u32;
        term_remap.insert(dim_id, compact);
    }
    let num_terms = term_remap.len();
    drop(eligible);

    // Phase 2+3: counts and CSR fill
    let mut counts = vec![0u32; total_blocks];
    let mut gb = 0usize;
    for bmp in bmps {
        for block_id in 0..bmp.num_blocks {
            for (dim_id, _) in bmp.iter_block_terms(block_id) {
                if term_remap.contains_key(&dim_id) {
                    counts[gb] += 1;
                }
            }
            gb += 1;
        }
    }
    let mut offsets = Vec::with_capacity(total_blocks + 1);
    offsets.push(0u32);
    for &c in &counts {
        offsets.push(offsets.last().unwrap() + c);
    }
    let total = *offsets.last().unwrap() as usize;
    let mut terms = vec![0u32; total];
    counts.fill(0);
    gb = 0;
    for bmp in bmps {
        for block_id in 0..bmp.num_blocks {
            for (dim_id, _) in bmp.iter_block_terms(block_id) {
                if let Some(&compact) = term_remap.get(&dim_id) {
                    let pos = offsets[gb] as usize + counts[gb] as usize;
                    terms[pos] = compact;
                    counts[gb] += 1;
                }
            }
            gb += 1;
        }
    }

    ForwardIndex {
        terms,
        offsets,
        num_terms,
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
        return (Vec::new(), true);
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
    let deadline = budget.time_budget.map(|d| std::time::Instant::now() + d);
    #[cfg(not(feature = "native"))]
    let deadline: Option<()> = None;
    #[cfg(not(feature = "native"))]
    let _ = deadline;

    let exhausted = std::sync::atomic::AtomicBool::new(false);
    #[cfg(feature = "native")]
    bisect(
        &mut docs,
        fwd,
        effective_min_partition,
        max_iters,
        &log_table,
        deadline,
        &exhausted,
    );
    #[cfg(not(feature = "native"))]
    bisect(
        &mut docs,
        fwd,
        effective_min_partition,
        max_iters,
        &log_table,
        None,
        &exhausted,
    );

    let converged = !exhausted.load(std::sync::atomic::Ordering::Relaxed);
    if !converged {
        log::info!(
            "BP graph_bisection: wall-clock budget {:?} exhausted at n={} — emitting partial (still valid) permutation",
            budget.time_budget,
            n,
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
fn bisect(
    docs: &mut [u32],
    fwd: &ForwardIndex,
    min_partition_size: usize,
    max_iters: usize,
    log_table: &[f32],
    #[cfg(feature = "native")] deadline: Option<std::time::Instant>,
    #[cfg(not(feature = "native"))] deadline: Option<()>,
    exhausted: &std::sync::atomic::AtomicBool,
) {
    let n = docs.len();
    if n <= min_partition_size {
        return;
    }
    // Anytime cutoff: leave this subtree in its current (valid) order.
    if exhausted.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    #[cfg(feature = "native")]
    if let Some(dl) = deadline
        && std::time::Instant::now() >= dl
    {
        exhausted.store(true, std::sync::atomic::Ordering::Relaxed);
        return;
    }
    #[cfg(not(feature = "native"))]
    let _ = deadline;

    let mid = n / 2;
    let nt = fwd.num_terms;

    // Adaptive iteration count: large partitions converge faster with
    // coarse splits, so fewer refinement passes suffice. The fine-grained
    // clustering is handled by deeper recursion levels with full iterations.
    let effective_iters = if n > 100_000 {
        max_iters.min(12)
    } else {
        max_iters
    };

    // Flat degree arrays: left_deg[term] and right_deg[term].
    // Allocated per bisection level; freed on return before recursion uses the
    // memory for sub-problems. Peak = O(num_terms × log2(n/min_partition_size)).
    let mut left_deg = vec![0u32; nt];
    let mut right_deg = vec![0u32; nt];

    for (i, &doc) in docs.iter().enumerate() {
        let target = if i < mid {
            &mut left_deg
        } else {
            &mut right_deg
        };
        for &term in fwd.doc_terms(doc as usize) {
            target[term as usize] += 1;
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
        if let Some(dl) = deadline
            && std::time::Instant::now() >= dl
        {
            exhausted.store(true, std::sync::atomic::Ordering::Relaxed);
            break;
        }
        // Compute gain for each document (approx_1 from Dhulipala et al.)
        // Parallelized for large partitions where per-doc work dominates.
        compute_gains(docs, fwd, mid, &left_deg, &right_deg, log_table, &mut gains);

        // Partition: the `mid` LOWEST keys (strongest left affinity) go left
        indices.clear();
        indices.extend(0..n);
        indices.select_nth_unstable_by(mid, |&a, &b| {
            gains[a]
                .partial_cmp(&gains[b])
                .unwrap_or(std::cmp::Ordering::Equal)
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
                for &term in fwd.doc_terms(doc as usize) {
                    let t = term as usize;
                    if was_left {
                        left_deg[t] -= 1;
                        right_deg[t] += 1;
                    } else {
                        right_deg[t] -= 1;
                        left_deg[t] += 1;
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
            let max_gain = gains.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max_gain.abs() < 0.001 {
                break;
            }
        }
    }

    // Drop scratch before recursion to free memory for sub-problems
    drop(left_deg);
    drop(right_deg);
    drop(gains);
    drop(indices);
    drop(new_left);
    drop(new_right);

    let (left, right) = docs.split_at_mut(mid);
    #[cfg(feature = "native")]
    rayon::join(
        || {
            bisect(
                left,
                fwd,
                min_partition_size,
                max_iters,
                log_table,
                deadline,
                exhausted,
            )
        },
        || {
            bisect(
                right,
                fwd,
                min_partition_size,
                max_iters,
                log_table,
                deadline,
                exhausted,
            )
        },
    );
    #[cfg(not(feature = "native"))]
    {
        bisect(
            left,
            fwd,
            min_partition_size,
            max_iters,
            log_table,
            deadline,
            exhausted,
        );
        bisect(
            right,
            fwd,
            min_partition_size,
            max_iters,
            log_table,
            deadline,
            exhausted,
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
    left_deg: &[u32],
    right_deg: &[u32],
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
            let t = term as usize;
            let (from, to) = if in_left {
                (left_deg[t], right_deg[t])
            } else {
                (right_deg[t], left_deg[t])
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

    /// Build a simple forward index from (doc_id, terms) pairs.
    fn make_fwd(docs: &[&[u32]], num_terms: usize) -> ForwardIndex {
        let mut terms = Vec::new();
        let mut offsets = vec![0u32];
        for doc_terms in docs {
            terms.extend_from_slice(doc_terms);
            offsets.push(terms.len() as u32);
        }
        ForwardIndex {
            terms,
            offsets,
            num_terms,
        }
    }

    #[test]
    fn test_bp_empty() {
        let fwd = ForwardIndex {
            terms: Vec::new(),
            offsets: Vec::new(),
            num_terms: 0,
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
