use super::*;

// Preserve the original posting-by-posting arithmetic as a test oracle. Its
// operation order matters: gain bits determine ties and the persisted order.
fn reference_gains(
    docs: &[u32],
    fwd: &ForwardIndex,
    mid: usize,
    degrees: &TermDegrees,
    log_table: &[f32],
) -> Vec<f32> {
    docs.iter()
        .enumerate()
        .map(|(i, &doc)| {
            let in_left = i < mid;
            let mut gain = 0.0f32;
            for &term in fwd.doc_terms(doc as usize) {
                let [left, right] = degrees.get(term as usize);
                let (from, to) = if in_left {
                    (left, right)
                } else {
                    (right, left)
                };
                let movement = fast_log2_lookup(to as usize + 2, log_table)
                    - fast_log2_lookup(from as usize, log_table)
                    - std::f32::consts::LOG2_E / (1.0 + to as f32);
                gain += if in_left { movement } else { -movement };
            }
            gain
        })
        .collect()
}

fn fixture(n: usize, vocabulary: usize, width: usize, lanes: usize) -> ForwardIndex {
    let mut terms = Vec::with_capacity(n * width);
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0);
    let mut random = 0xc0ffee1234567890u64;
    let mut row = Vec::with_capacity(width);
    for doc in 0..n {
        // Mixed popular and topic-local terms in a shuffled arrival order.
        // Empty rows and uneven term counts also exercise the CSR boundaries.
        if doc % 97 != 0 {
            row.clear();
            let topic = (doc * 7919) % 64;
            for i in 0..width {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let term = if i % 2 == 0 {
                    random as usize % (vocabulary / 16).max(1)
                } else {
                    (topic * (vocabulary / 64) + random as usize % 64) % vocabulary
                };
                row.push(term as u32);
            }
            row.sort_unstable();
            row.dedup();
            terms.extend_from_slice(&row);
        }
        offsets.push(terms.len() as u64);
    }
    ForwardIndex {
        terms,
        offsets,
        num_terms: vocabulary,
        parallel_bisect_lanes: lanes,
        cache_gains: true,
        budget_limited: false,
    }
}

#[test]
fn gain_bits_preserve_half_direction_empty_rows_and_reused_degrees() {
    let fwd = fixture(10_003, 257, 24, 1);
    let mut docs: Vec<u32> = (0..fwd.num_docs() as u32).collect();
    let log_table = build_log_table(4096);
    let mut workspaces = [TermDegrees::new(fwd.num_terms)];
    let mut gains = vec![0.0; docs.len()];
    for mid in [1, 5001, 10_002] {
        docs.rotate_left(31);
        build_term_degrees(&docs, mid, &fwd, &mut workspaces, None);
        let expected = reference_gains(&docs, &fwd, mid, &workspaces[0], &log_table);
        for cached in [false, true] {
            if cached {
                workspaces[0]
                    .gain_cache
                    .resize(fwd.num_terms, [f32::NAN; 2]);
            } else {
                workspaces[0].gain_cache.clear();
            }
            for parallel in [false, true] {
                compute_gains(
                    &docs,
                    &fwd,
                    mid,
                    &mut workspaces[0],
                    &log_table,
                    &mut gains,
                    parallel,
                );
                assert_eq!(
                    gains.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "mid={mid}, parallel={parallel}, cached={cached}",
                );
            }
        }
    }
}

#[test]
fn gain_cache_requires_spare_memory_for_every_existing_lane() {
    let terms = 130;
    let lanes = 3;
    let fixed = 20_000;
    let required = fixed
        + LOG_TABLE_SIZE * std::mem::size_of::<f32>()
        + lanes * (term_degree_bytes(terms) + terms * 8 + std::mem::size_of::<TermDegrees>());
    assert!(!gain_cache_fits(required - 1, fixed, terms, lanes));
    assert!(gain_cache_fits(required, fixed, terms, lanes));
    assert!(!gain_cache_fits(usize::MAX, fixed, usize::MAX, lanes));
    assert!(!gain_cache_fits(usize::MAX, fixed, terms, usize::MAX));
    assert!(!gain_cache_fits(usize::MAX, fixed, 0, lanes));
}

#[test]
fn cached_gains_preserve_full_permutations_and_reuse_lane_storage() {
    let mut fwd = fixture(2051, 1024, 24, 1);
    fwd.cache_gains = false;
    let direct = graph_bisection(&fwd, 32, 12, BpBudget::full());
    fwd.cache_gains = true;
    assert_eq!(graph_bisection(&fwd, 32, 12, BpBudget::full()), direct);

    let docs: Vec<u32> = (0..fwd.num_docs() as u32).collect();
    let mut degrees = [TermDegrees::new(fwd.num_terms)];
    degrees[0].gain_cache = vec![[f32::NAN; 2]; fwd.num_terms];
    let cache_ptr = degrees[0].gain_cache.as_ptr();
    let logs = build_log_table(4096);
    // Deep partitions omit most of the preceding partition's terms. A reused
    // cache must read only freshly initialized degrees and current terms.
    for subset in [&docs[..], &docs[10..43], &docs[900..911]] {
        let mid = subset.len() / 2;
        build_term_degrees(subset, mid, &fwd, &mut degrees, None);
        let expected = reference_gains(subset, &fwd, mid, &degrees[0], &logs);
        let mut actual = vec![0.0; subset.len()];
        compute_gains(
            subset,
            &fwd,
            mid,
            &mut degrees[0],
            &logs,
            &mut actual,
            false,
        );
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(degrees[0].gain_cache.as_ptr(), cache_ptr);
    }
}
