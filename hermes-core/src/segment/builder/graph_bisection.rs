//! Recursive Graph Bisection (BP) for BMP document ordering.
//!
//! Based on Dhulipala et al. (KDD 2016) and Mackenzie et al. — the same
//! algorithm used in Lucene and PISA for document reordering.
//!
//! Directly optimizes log-gap cost: docs sharing dimensions end up in the
//! same BMP blocks, producing tight upper bounds and effective pruning.
//!
//! Memory: forward index ~200 bytes/doc (temporary) + degree arrays ~840 KB.

use rustc_hash::FxHashMap;

use crate::DocId;

use super::bmp::VidLookup;

// ── Forward index (CSR) ──────────────────────────────────────────────────

/// Forward index in CSR format: doc `d`'s terms are `terms[offsets[d]..offsets[d+1]]`.
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
}

// ── BP configuration ─────────────────────────────────────────────────────

/// Parameters for building a forward index from inverted postings.
pub(crate) struct BpParams {
    pub weight_threshold: f32,
    pub max_weight: f32,
    pub min_terms: usize,
    pub min_doc_freq: usize,
    pub max_doc_freq: usize,
}

// ── Build forward index from inverted postings (builder path) ────────────

/// Build forward index from BMP postings (inverted → forward).
///
/// Filters dims with doc_freq outside `[min_doc_freq, max_doc_freq]`.
/// `vid_lookup` maps `(doc_id, ordinal)` → virtual_id.
pub(crate) fn build_forward_index(
    num_docs: usize,
    postings: &FxHashMap<u32, Vec<(DocId, u16, f32)>>,
    vid_lookup: &VidLookup,
    params: &BpParams,
) -> ForwardIndex {
    let BpParams {
        weight_threshold,
        max_weight,
        min_terms,
        min_doc_freq,
        max_doc_freq,
    } = *params;
    // Phase 1: count terms per doc
    let mut counts = vec![0u32; num_docs];
    let mut num_active_terms = 0usize;

    for (&_dim_id, dim_posts) in postings {
        // Filter by doc frequency
        let df = dim_posts.len();
        if df < min_doc_freq || df > max_doc_freq {
            continue;
        }
        num_active_terms += 1;
        let skip_wt = df < min_terms;
        for &(doc_id, ordinal, weight) in dim_posts {
            let abs_w = weight.abs();
            if !skip_wt && abs_w < weight_threshold {
                continue;
            }
            let impact = quantize_weight_for_bp(abs_w, max_weight);
            if impact == 0 {
                continue;
            }
            if let Some(vid) = vid_lookup
                .try_get((doc_id, ordinal))
                .filter(|&v| (v as usize) < num_docs)
            {
                counts[vid as usize] += 1;
            }
        }
    }

    // Phase 2: build CSR offsets
    let mut offsets = Vec::with_capacity(num_docs + 1);
    offsets.push(0u32);
    for &c in &counts {
        offsets.push(offsets.last().unwrap() + c);
    }
    let total = *offsets.last().unwrap() as usize;

    // Phase 3: fill terms
    let mut terms = vec![0u32; total];
    counts.fill(0);

    for (&dim_id, dim_posts) in postings {
        let df = dim_posts.len();
        if df < min_doc_freq || df > max_doc_freq {
            continue;
        }
        let skip_wt = df < min_terms;
        for &(doc_id, ordinal, weight) in dim_posts {
            let abs_w = weight.abs();
            if !skip_wt && abs_w < weight_threshold {
                continue;
            }
            let impact = quantize_weight_for_bp(abs_w, max_weight);
            if impact == 0 {
                continue;
            }
            if let Some(vid) = vid_lookup.try_get((doc_id, ordinal)) {
                let vid = vid as usize;
                if vid < num_docs {
                    let pos = offsets[vid] as usize + counts[vid] as usize;
                    terms[pos] = dim_id;
                    counts[vid] += 1;
                }
            }
        }
    }

    ForwardIndex {
        terms,
        offsets,
        num_terms: num_active_terms,
    }
}

/// Build forward index from an existing BmpIndex (for reorder path).
///
/// Iterates all blocks and postings to invert the BMP block data into a
/// per-virtual-doc forward index. Filters by doc frequency bounds.
pub(crate) fn build_forward_index_from_bmp(
    bmp: &crate::segment::reader::bmp::BmpIndex,
    min_doc_freq: usize,
    max_doc_freq: usize,
) -> ForwardIndex {
    let num_docs = bmp.num_real_docs() as usize;
    let num_blocks = bmp.num_blocks as usize;
    let block_size = bmp.bmp_block_size as usize;

    // Phase 1: count doc freq per dimension
    let mut dim_df: FxHashMap<u32, usize> = FxHashMap::default();
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

    // Phase 2: count terms per doc (filtered)
    let mut counts = vec![0u32; num_docs];
    let mut num_active_terms = 0usize;

    for block_id in 0..num_blocks {
        for (dim_id, postings) in bmp.iter_block_terms(block_id as u32) {
            let df = dim_df.get(&dim_id).copied().unwrap_or(0);
            if df < min_doc_freq || df > max_doc_freq {
                continue;
            }
            // Count active terms only once
            // (we'll deduplicate after)
            for p in postings {
                let vid = block_id * block_size + p.local_slot as usize;
                if vid < num_docs && p.impact > 0 {
                    counts[vid] += 1;
                }
            }
        }
    }

    // Count distinct active terms
    for &df in dim_df.values() {
        if df >= min_doc_freq && df <= max_doc_freq {
            num_active_terms += 1;
        }
    }

    // Phase 3: build CSR offsets
    let mut offsets = Vec::with_capacity(num_docs + 1);
    offsets.push(0u32);
    for &c in &counts {
        offsets.push(offsets.last().unwrap() + c);
    }
    let total = *offsets.last().unwrap() as usize;

    // Phase 4: fill terms
    let mut terms = vec![0u32; total];
    counts.fill(0);

    for block_id in 0..num_blocks {
        for (dim_id, postings) in bmp.iter_block_terms(block_id as u32) {
            let df = dim_df.get(&dim_id).copied().unwrap_or(0);
            if df < min_doc_freq || df > max_doc_freq {
                continue;
            }
            for p in postings {
                let vid = block_id * block_size + p.local_slot as usize;
                if vid < num_docs && p.impact > 0 {
                    let pos = offsets[vid] as usize + counts[vid] as usize;
                    terms[pos] = dim_id;
                    counts[vid] += 1;
                }
            }
        }
    }

    ForwardIndex {
        terms,
        offsets,
        num_terms: num_active_terms,
    }
}

// ── Recursive Graph Bisection ────────────────────────────────────────────

/// Recursive graph bisection. Returns permutation: `perm[new_pos] = old_index`.
///
/// `min_partition_size` should be the BMP block_size (64).
/// `max_iters` controls convergence (20 is standard).
pub(crate) fn graph_bisection(
    fwd: &ForwardIndex,
    min_partition_size: usize,
    max_iters: usize,
) -> Vec<u32> {
    let n = fwd.num_docs();
    if n == 0 {
        return Vec::new();
    }

    // Initialize document order as identity permutation
    let mut docs: Vec<u32> = (0..n as u32).collect();

    // Precompute fast_log2 table
    let log_table = build_log_table(4096);

    // Allocate degree arrays — shared across recursive calls via slicing.
    // Two arrays: left_deg[term], right_deg[term] for the current partition.
    // We need term_id → degree, using the max dim_id as upper bound won't work
    // efficiently. Instead, use a FxHashMap approach within each bisection call.
    // For efficiency, allocate scratch buffers once.
    bisect(&mut docs, fwd, min_partition_size, max_iters, &log_table);

    docs
}

/// Recursive bisection of a document slice.
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

    // Build term degree arrays for left and right halves.
    // left_deg[term_id] = count of docs in left half containing term_id
    // right_deg[term_id] = count of docs in right half containing term_id
    let mut left_deg: FxHashMap<u32, u32> = FxHashMap::default();
    let mut right_deg: FxHashMap<u32, u32> = FxHashMap::default();

    for (i, &doc) in docs.iter().enumerate() {
        let target = if i < mid {
            &mut left_deg
        } else {
            &mut right_deg
        };
        for &term in fwd.doc_terms(doc as usize) {
            *target.entry(term).or_insert(0) += 1;
        }
    }

    // Iterative refinement
    let mut gains: Vec<f32> = vec![0.0; n];

    for iter in 0..max_iters {
        // Compute gain for each document
        // gain(d) = sum over terms t in d:
        //   if d is in left:  log2(right_deg[t] + 2) - log2(left_deg[t]) - 1/(1 + right_deg[t])
        //   if d is in right: log2(left_deg[t] + 2) - log2(right_deg[t]) - 1/(1 + left_deg[t])
        // (approx_1 from the paper: gain for moving d from its current side to the other)
        for (i, &doc) in docs.iter().enumerate() {
            let mut gain = 0.0f32;
            let in_left = i < mid;
            for &term in fwd.doc_terms(doc as usize) {
                let ld = *left_deg.get(&term).unwrap_or(&0);
                let rd = *right_deg.get(&term).unwrap_or(&0);
                if in_left {
                    // Moving from left to right
                    // from_deg = ld, to_deg = rd
                    gain += fast_log2_lookup(rd as usize + 2, log_table)
                        - fast_log2_lookup(ld as usize, log_table)
                        - std::f32::consts::LOG2_E / (1.0 + rd as f32);
                } else {
                    // Moving from right to left
                    gain += fast_log2_lookup(ld as usize + 2, log_table)
                        - fast_log2_lookup(rd as usize, log_table)
                        - std::f32::consts::LOG2_E / (1.0 + ld as f32);
                }
            }
            gains[i] = gain;
        }

        // Partition: highest-gain docs go to left, lowest to right.
        // Use select_nth_unstable_by to partition around midpoint.
        // We want the top `mid` docs by gain in left, rest in right.
        // So partition by descending gain — docs[0..mid] = highest gain (want to be in left).

        // Create index array sorted by gain
        let mut indices: Vec<usize> = (0..n).collect();
        indices.select_nth_unstable_by(mid, |&a, &b| {
            gains[b]
                .partial_cmp(&gains[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Determine which docs are swapping sides
        let mut new_left: Vec<u32> = Vec::with_capacity(mid);
        let mut new_right: Vec<u32> = Vec::with_capacity(n - mid);
        let mut any_swap = false;

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
                any_swap = true;
                // Update degree arrays
                for &term in fwd.doc_terms(doc as usize) {
                    if was_left {
                        // Moved left → right
                        *left_deg.entry(term).or_insert(0) -= 1;
                        *right_deg.entry(term).or_insert(0) += 1;
                    } else {
                        // Moved right → left
                        *right_deg.entry(term).or_insert(0) -= 1;
                        *left_deg.entry(term).or_insert(0) += 1;
                    }
                }
            }
        }

        // Apply new ordering
        docs[..mid].copy_from_slice(&new_left);
        docs[mid..].copy_from_slice(&new_right);

        if !any_swap {
            break;
        }

        // Cooling: if gains are very small in later iterations, break early
        if iter > 5 {
            let max_gain = gains.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max_gain.abs() < 0.001 {
                break;
            }
        }
    }

    // Recurse on left and right halves
    let (left, right) = docs.split_at_mut(mid);
    rayon::join(
        || bisect(left, fwd, min_partition_size, max_iters, log_table),
        || bisect(right, fwd, min_partition_size, max_iters, log_table),
    );
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Quantize weight to u8 for BP forward index (same scale as BMP builder).
#[inline]
fn quantize_weight_for_bp(weight: f32, max_scale: f32) -> u8 {
    if max_scale <= 0.0 {
        return 0;
    }
    let normalized = (weight / max_scale * 255.0).round();
    normalized.clamp(0.0, 255.0) as u8
}

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
        // 16 docs with mixed terms
        let docs: Vec<Vec<u32>> = (0..16).map(|i| vec![i / 4, 10 + i / 2]).collect();
        let doc_refs: Vec<&[u32]> = docs.iter().map(|v| v.as_slice()).collect();
        let fwd = make_fwd(&doc_refs, 16);
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
