use super::scoring::MAX_LOOKUP_DIMENSIONS;
use super::*;
fn merged_fixture(values: &[&[f32]]) -> SeismicIndex {
    use crate::segment::seismic::{MemoryWriter, build_blob, copy_sources_for_test};
    use crate::structures::{SparseFormat, SparseVectorConfig};
    let config = SparseVectorConfig {
        format: SparseFormat::Seismic,
        ..Default::default()
    };
    let indexes: Vec<_> = values
        .iter()
        .map(|weights| {
            let mut postings = rustc_hash::FxHashMap::default();
            postings.insert(
                1,
                weights
                    .iter()
                    .enumerate()
                    .map(|(ordinal, &weight)| (0, ordinal as u16, weight))
                    .collect(),
            );
            let mut bytes = MemoryWriter::default();
            build_blob(postings, &config, &mut bytes).unwrap();
            bytes.parse(1, weights.len() as u32).unwrap()
        })
        .collect();
    let sources: Vec<_> = indexes
        .iter()
        .enumerate()
        .map(|(doc, index)| (index, doc as u32))
        .collect();
    let mut bytes = MemoryWriter::default();
    copy_sources_for_test(&sources, &mut bytes, &|| Ok(())).unwrap();
    bytes
        .parse(
            values.len() as u32,
            values.iter().map(|weights| weights.len() as u32).sum(),
        )
        .unwrap()
}

fn infos(combiner: MultiValueCombiner, exhaustive: bool) -> Vec<SparseTermQueryInfo> {
    use crate::query::{Query, QueryDecomposition, SparseVectorQuery};
    let query = SparseVectorQuery::new(crate::Field(0), vec![(1, 1.0)])
        .with_combiner(combiner)
        .with_seismic_factor(1.0)
        .with_exhaustive(exhaustive);
    let QueryDecomposition::SparseTerms(infos) = query.decompose() else {
        panic!("expected sparse query");
    };
    infos
}

#[test]
fn merged_first_term_visits_strongest_run_before_scoring_weak_candidates() {
    let index = merged_fixture(&[&[1.0], &[100.0]]);
    let visited = std::cell::RefCell::new(Vec::new());
    let hits = execute(
        &index,
        &infos(MultiValueCombiner::Max, false),
        1,
        &|doc| {
            visited.borrow_mut().push(doc);
            true
        },
        &ScorerOptions::default(),
        false,
        ("test", "sparse"),
    )
    .unwrap();
    assert_eq!(*visited.borrow(), [1]);
    assert_eq!((hits[0].doc_id, hits[0].score), (1, 100.0));
}

#[test]
fn merged_ordering_keeps_filtered_multivalue_and_exact_scores() {
    let index = merged_fixture(&[&[60.0, 60.0], &[100.0], &[-20.0]]);
    for combiner in [
        MultiValueCombiner::Max,
        MultiValueCombiner::Sum,
        MultiValueCombiner::Avg,
    ] {
        for filtered in [false, true] {
            let mut results = Vec::new();
            for exhaustive in [false, true] {
                let hits = execute(
                    &index,
                    &infos(combiner, exhaustive),
                    1,
                    &|doc| !filtered || doc != 1,
                    &ScorerOptions::with_positions(),
                    filtered,
                    ("test", "sparse"),
                )
                .unwrap();
                results.push(
                    hits.into_iter()
                        .map(|hit| (hit.doc_id, hit.score, hit.ordinals))
                        .collect::<Vec<_>>(),
                );
            }
            assert_eq!(results[0], results[1]);
        }
    }
}

#[cfg(feature = "sync")]
#[test]
fn merged_parallel_summaries_preserve_signed_duplicate_arithmetic_and_cancellation() {
    let index = merged_fixture(&[&[1.0], &[100.0], &[-20.0], &[2.0, -3.0], &[8.0]]);
    let runs: Vec<_> = index.term_runs(1).collect();
    let count = runs.iter().map(|run| run.cluster_count()).sum();
    let query = [(1, -2.0), (1, 0.5), (1, 0.0)];
    let mut expected = vec![0.0; count];
    assert!(score_term_runs_sequential(
        &runs,
        &query,
        &mut expected,
        &ScorerOptions::default()
    ));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut actual = vec![f32::NAN; count];
    assert!(pool.install(|| score_term_runs(
        &runs,
        &query,
        &mut actual,
        &ScorerOptions::default()
    )));
    assert_eq!(
        actual.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|s| s.to_bits()).collect::<Vec<_>>()
    );
    let budget =
        super::super::SharedThreshold::new().with_deadline(Some(std::time::Instant::now()));
    let options = ScorerOptions {
        shared_threshold: Some(budget.clone()),
        ..Default::default()
    };
    assert!(!pool.install(|| score_term_runs(&runs, &query, &mut actual, &options)));
    assert!(budget.truncated());
}

#[test]
fn forward_iterator_scoring_preserves_signed_duplicate_and_cancelling_arithmetic() {
    use crate::segment::seismic::{MemoryWriter, build_blob};
    use crate::structures::{SparseFormat, SparseVectorConfig, WeightQuantization};
    for precision in [
        WeightQuantization::Float32,
        WeightQuantization::Float16,
        WeightQuantization::UInt8,
        WeightQuantization::UInt4,
    ] {
        let config = SparseVectorConfig {
            format: SparseFormat::Seismic,
            weight_quantization: precision,
            dims: Some(129),
            ..Default::default()
        };
        let postings = (0..129)
            .map(|dimension| (dimension, vec![(0, 0, (dimension % 17) as f32 - 8.0)]))
            .collect();
        let mut output = MemoryWriter::default();
        build_blob(postings, &config, &mut output).unwrap();
        let index = output.parse(1, 1).unwrap();
        let vector = index.vector(0);
        let decoded: Vec<_> = vector.iter().collect();
        let unique: Vec<_> = (0..129)
            .map(|dimension| (dimension, (dimension % 7) as f32 - 3.0))
            .collect();
        let duplicates: Vec<_> = unique
            .iter()
            .flat_map(|&(dim, weight)| [(dim, weight), (dim, -weight)])
            .collect();
        let mut wide = unique.clone();
        wide.push((90_000, 1.0));
        for terms in [
            unique,
            duplicates,
            wide,
            vec![(0, f32::MAX), (0, -f32::MAX)],
        ] {
            let query = PreparedQuery::new(terms);
            let expected = query.score_vector(decoded.iter().copied());
            let actual = query.score_vector(vector.iter());
            match (expected, actual) {
                (Ok(expected), Ok(actual)) => assert_eq!(
                    expected.map(f32::to_bits),
                    actual.map(f32::to_bits),
                    "{precision:?}"
                ),
                (Err(_), Err(_)) => {}
                (expected, actual) => {
                    panic!("{precision:?}: forward score differs: {expected:?} vs {actual:?}")
                }
            }
        }
    }
}

#[test]
fn disabling_exhaustive_preserves_explicit_bmp_query_controls() {
    use crate::query::{Query, QueryDecomposition, SparseTermQuery, SparseVectorQuery};
    let field = crate::Field(0);
    let vector = SparseVectorQuery::new(field, vec![(0, 1.0)])
        .with_heap_factor(0.75)
        .with_lsp_gamma(0)
        .with_exhaustive(true)
        .with_exhaustive(false);
    let term = SparseTermQuery::new(field, 0, 1.0)
        .with_heap_factor(0.75)
        .with_lsp_gamma(0)
        .with_exhaustive(true)
        .with_exhaustive(false);
    for query in [&vector as &dyn Query, &term as &dyn Query] {
        let QueryDecomposition::SparseTerms(infos) = query.decompose() else {
            panic!("expected sparse query");
        };
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].heap_factor, 0.75);
        assert_eq!(infos[0].lsp_gamma, Some(0));
        assert!(!infos[0].exhaustive);
    }
}

#[test]
fn signed_forward_scoring_preserves_duplicates_and_large_dimensions() {
    let query = PreparedQuery::new(vec![(2, 1.0), (2, 0.5), (90_000, -2.0)]);
    assert_eq!(
        query
            .score_vector([(2, -4.0), (90_000, -3.0)].into_iter())
            .unwrap(),
        Some(0.0)
    );
    assert_eq!(query.score_vector([(3, 1.0)].into_iter()).unwrap(), None);
}
#[test]
fn geometric_proxy_overflow_does_not_hide_exact_score_overflow() {
    let values = [(1, f32::MAX), (2, f32::MAX)];
    let query = PreparedQuery::new(vec![(1, 1.0), (2, 1.0)]);
    assert_eq!(query.summary_score(values.into_iter()), f32::INFINITY);
    assert!(query.score_vector(values.into_iter()).is_err());
}

#[test]
fn lookup_matches_sorted_scoring_for_signed_cancelling_and_stored_zero_values() {
    let terms = vec![(1, -2.0), (2, 0.0), (4, 3.0), (65_535, 0.5)];
    let lookup = PreparedQuery::new(terms.clone());
    assert!(
        matches!(&lookup, PreparedQuery::Lookup { weights, .. } if weights.len() == MAX_LOOKUP_DIMENSIONS)
    );
    let sorted = PreparedQuery::Sorted(terms.into_iter().filter(|&(_, w)| w != 0.0).collect());
    let vectors = [
        vec![],
        vec![(0, 8.0), (u32::MAX, 7.0)],
        vec![(2, 7.0)],
        vec![(4, -0.0)],
        vec![(1, 3.0), (4, 2.0)],
        vec![
            (1, -4.0),
            (3, 9.0),
            (4, 2.0),
            (65_535, 5.0),
            (u32::MAX, 7.0),
        ],
    ];
    for vector in vectors {
        assert_eq!(
            lookup
                .score_vector(vector.iter().copied())
                .unwrap()
                .map(f32::to_bits),
            sorted
                .score_vector(vector.iter().copied())
                .unwrap()
                .map(f32::to_bits),
            "{vector:?}",
        );
        assert_eq!(
            lookup
                .summary_score(vector.iter().map(|&(dim, weight)| (dim, weight.abs())))
                .to_bits(),
            sorted
                .summary_score(vector.iter().map(|&(dim, weight)| (dim, weight.abs())))
                .to_bits(),
        );
    }
    assert_eq!(
        lookup.score_vector([(4, 0.0)].into_iter()).unwrap(),
        Some(0.0)
    );
    assert_eq!(
        lookup
            .score_vector([(1, 3.0), (4, 2.0)].into_iter())
            .unwrap(),
        Some(0.0)
    );
    assert_eq!(lookup.score_vector([(2, 7.0)].into_iter()).unwrap(), None);
}

#[test]
fn duplicate_and_wide_queries_keep_individual_sorted_operations() {
    let duplicate = PreparedQuery::new(vec![(2, f32::MAX), (2, -f32::MAX)]);
    assert!(matches!(duplicate, PreparedQuery::Sorted(_)));
    // Combining these query weights first would hide the original
    // overflowing products. The fallback must preserve the error.
    assert!(duplicate.score_vector([(2, 2.0)].into_iter()).is_err());
    assert_eq!(
        duplicate.summary_score([(2, 2.0)].into_iter()),
        f32::INFINITY
    );
    for dimension in [65_536, 90_000, u32::MAX] {
        let query = PreparedQuery::new(vec![(dimension, -2.0)]);
        assert!(matches!(query, PreparedQuery::Sorted(_)));
        assert_eq!(
            query.score_vector([(dimension, -3.0)].into_iter()).unwrap(),
            Some(6.0)
        );
        assert_eq!(query.summary_score([(dimension, 3.0)].into_iter()), 6.0);
    }
}

#[test]
fn query_budgets_reject_invalid_values() {
    for (cut, factor) in [(0, 0.85), (65, 0.85), (10, f32::NAN), (10, -0.1), (10, 1.1)] {
        assert!(validate_options(cut, factor).is_err());
    }
    assert!(validate_options(64, 0.0).is_ok());
}
