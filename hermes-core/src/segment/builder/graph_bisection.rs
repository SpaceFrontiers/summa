//! Recursive Graph Bisection (BP) for BMP document ordering.
//!
//! Based on Dhulipala et al. (KDD 2016) and Mackenzie et al. — the same
//! algorithm used in Lucene and PISA for document reordering.
//!
//! Directly optimizes log-gap cost: docs sharing dimensions end up in the
//! same BMP blocks, producing tight upper bounds and effective pruning.
//!
//! Memory: forward index ~200 bytes/doc (temporary) + degree arrays ~840 KB.

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

/// Build forward index from BmpIndex sources (single or multi-source merge reorder).
///
/// Virtual IDs are assigned sequentially across sources: source 0 gets 0..n0,
/// source 1 gets n0..n0+n1, etc. Returns `(forward_index, per_source_doc_counts)`.
///
/// Filters dims with doc_freq outside `[min_doc_freq, max_doc_freq]`.
/// Remaps term IDs to compact range for flat-array degree tracking.
pub(crate) fn build_forward_index_from_bmps(
    bmps: &[&crate::segment::reader::bmp::BmpIndex],
    min_doc_freq: usize,
    max_doc_freq: usize,
) -> (ForwardIndex, Vec<usize>) {
    let source_doc_counts: Vec<usize> = bmps.iter().map(|b| b.num_real_docs() as usize).collect();
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
    for bmp in bmps {
        let num_blocks = bmp.num_blocks as usize;
        let block_size = bmp.bmp_block_size as usize;
        let num_docs = bmp.num_real_docs() as usize;
        for block_id in 0..num_blocks {
            for (dim_id, postings) in bmp.iter_block_terms(block_id as u32) {
                for p in postings {
                    let vid = block_id * block_size + p.local_slot as usize;
                    if vid < num_docs && p.impact > 0 {
                        *dim_df.entry(dim_id).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    let mut term_remap: FxHashMap<u32, u32> = FxHashMap::default();
    for (&dim_id, &df) in &dim_df {
        if df >= min_doc_freq && df <= max_doc_freq {
            let compact_id = term_remap.len() as u32;
            term_remap.insert(dim_id, compact_id);
        }
    }
    let num_active_terms = term_remap.len();

    // Phase 2: count terms per doc (filtered)
    let mut counts = vec![0u32; total_docs];
    let mut vid_offset = 0usize;

    for bmp in bmps {
        let num_blocks = bmp.num_blocks as usize;
        let block_size = bmp.bmp_block_size as usize;
        let num_docs = bmp.num_real_docs() as usize;
        for block_id in 0..num_blocks {
            for (dim_id, postings) in bmp.iter_block_terms(block_id as u32) {
                if !term_remap.contains_key(&dim_id) {
                    continue;
                }
                for p in postings {
                    let vid = block_id * block_size + p.local_slot as usize;
                    if vid < num_docs && p.impact > 0 {
                        counts[vid_offset + vid] += 1;
                    }
                }
            }
        }
        vid_offset += num_docs;
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
    vid_offset = 0;

    for bmp in bmps {
        let num_blocks = bmp.num_blocks as usize;
        let block_size = bmp.bmp_block_size as usize;
        let num_docs = bmp.num_real_docs() as usize;
        for block_id in 0..num_blocks {
            for (dim_id, postings) in bmp.iter_block_terms(block_id as u32) {
                let Some(&compact) = term_remap.get(&dim_id) else {
                    continue;
                };
                for p in postings {
                    let vid = block_id * block_size + p.local_slot as usize;
                    if vid < num_docs && p.impact > 0 {
                        let global_vid = vid_offset + vid;
                        let pos = offsets[global_vid] as usize + counts[global_vid] as usize;
                        terms[pos] = compact;
                        counts[global_vid] += 1;
                    }
                }
            }
        }
        vid_offset += num_docs;
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

// ── Recursive Graph Bisection ────────────────────────────────────────────

/// Recursive graph bisection. Returns permutation: `perm[new_pos] = old_index`.
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
) -> Vec<u32> {
    let n = fwd.num_docs();
    if n == 0 {
        return Vec::new();
    }

    let mut docs: Vec<u32> = (0..n as u32).collect();
    let depth = if min_partition_size > 0 {
        ((n as f64) / (min_partition_size as f64)).log2().ceil() as usize
    } else {
        0
    };
    let log_table = build_log_table(4096);

    log::debug!(
        "BP graph_bisection: n={}, min_partition={}, max_iters={}, depth=~{}",
        n,
        min_partition_size,
        max_iters,
        depth,
    );

    bisect(&mut docs, fwd, min_partition_size, max_iters, &log_table);

    docs
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
) {
    let n = docs.len();
    if n <= min_partition_size {
        return;
    }

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
        // Compute gain for each document (approx_1 from Dhulipala et al.)
        // Parallelized for large partitions where per-doc work dominates.
        compute_gains(docs, fwd, mid, &left_deg, &right_deg, log_table, &mut gains);

        // Partition: top `mid` by gain go to left
        indices.clear();
        indices.extend(0..n);
        indices.select_nth_unstable_by(mid, |&a, &b| {
            gains[b]
                .partial_cmp(&gains[a])
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
    rayon::join(
        || bisect(left, fwd, min_partition_size, max_iters, log_table),
        || bisect(right, fwd, min_partition_size, max_iters, log_table),
    );
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
    let gain_for_doc = |i: usize| -> f32 {
        let doc = docs[i] as usize;
        let in_left = i < mid;
        let mut g = 0.0f32;
        for &term in fwd.doc_terms(doc) {
            let t = term as usize;
            let (same, other) = if in_left {
                (left_deg[t], right_deg[t])
            } else {
                (right_deg[t], left_deg[t])
            };
            g += fast_log2_lookup(other as usize + 2, log_table)
                - fast_log2_lookup(same as usize, log_table)
                - std::f32::consts::LOG2_E / (1.0 + other as f32);
        }
        g
    };

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
        let perm = graph_bisection(&fwd, 4, 20);
        assert!(perm.is_empty());
    }

    #[test]
    fn test_bp_small() {
        // 4 docs, min_partition_size=4 → no bisection, identity
        let fwd = make_fwd(&[&[0, 1], &[0, 2], &[1, 3], &[2, 3]], 4);
        let perm = graph_bisection(&fwd, 4, 20);
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
        let perm = graph_bisection(&fwd, 4, 20);
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
        let perm = graph_bisection(&fwd, 4, 20);

        assert_eq!(perm.len(), 16);
        // Must be a valid permutation
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<u32> = (0..16).collect();
        assert_eq!(sorted, expected);
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
