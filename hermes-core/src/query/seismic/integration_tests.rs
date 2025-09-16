use super::*;
use crate::query::{FilteredQuery, Query, SparseVectorQuery, TermQuery};
use crate::structures::{SparseFormat, SparseVectorConfig, WeightQuantization};
use crate::{Document, Index, IndexConfig, IndexWriter, RamDirectory, Schema};
use std::sync::Arc;

async fn fixture() -> (Index<RamDirectory>, crate::Field, crate::Field) {
    let mut schema = Schema::builder();
    let field = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::Seismic,
            weight_quantization: WeightQuantization::Float32,
            dims: Some(100_000),
            ..Default::default()
        },
    );
    let tag = schema.add_text_field("tag", true, false);
    let directory = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(directory.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    let vectors = [
        vec![vec![(2, -4.0), (90_000, -3.0)], vec![(2, 6.0)]],
        vec![vec![(2, 4.0)]],
        vec![vec![(3, 100.0)]],
        vec![],
        vec![vec![(2, -2.0)]],
    ];
    for (doc_id, vectors) in vectors.into_iter().enumerate() {
        let mut doc = Document::new();
        doc.add_text(tag, if doc_id == 4 { "allowed" } else { "excluded" });
        for vector in vectors {
            doc.add_sparse_vector(field, vector);
        }
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    (Index::open(directory, config).await.unwrap(), field, tag)
}

#[tokio::test]
async fn default_bmp_and_explicit_sparse_backends_dispatch_and_filter_consistently() {
    let mut schema = Schema::builder();
    let default = schema.add_sparse_vector_field("default", true, false);
    let maxscore = schema.add_sparse_vector_field_with_config(
        "maxscore",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::MaxScore,
            ..Default::default()
        },
    );
    let seismic = schema.add_sparse_vector_field_with_config(
        "seismic",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::Seismic,
            ..Default::default()
        },
    );
    let tag = schema.add_text_field("tag", true, false);
    let fields = [default, maxscore, seismic];
    let directory = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(directory.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for (doc_id, weight) in [0.5, 1.0, 2.0].into_iter().enumerate() {
        let mut doc = Document::new();
        doc.add_text(tag, if doc_id < 2 { "allowed" } else { "excluded" });
        for field in fields {
            doc.add_sparse_vector(field, vec![(1, weight), (2, weight)]);
        }
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(directory, config).await.unwrap();
    let segment = index.segment_readers().await.unwrap().pop().unwrap();
    assert_eq!(SparseVectorConfig::default().format, SparseFormat::Bmp);
    assert!(segment.bmp_index(default).is_some());
    assert!(segment.bmp_index(maxscore).is_none());
    assert!(segment.sparse_index(maxscore).is_some());
    assert!(segment.seismic_index(seismic).is_some());
    for field in fields {
        let query = SparseVectorQuery::new(field, vec![(1, 1.0), (2, 0.5)])
            .with_combiner(MultiValueCombiner::Max)
            .with_exhaustive(true);
        let hits = index.search(&query, 2).await.unwrap().hits;
        assert_eq!(
            hits.iter()
                .map(|hit| hit.address.doc_id)
                .collect::<Vec<_>>(),
            [2, 1]
        );
        let filtered = FilteredQuery::new(
            Arc::new(query),
            vec![Arc::new(TermQuery::text(tag, "allowed"))],
        );
        let hits = index.search(&filtered, 1).await.unwrap().hits;
        assert_eq!(
            hits.iter()
                .map(|hit| hit.address.doc_id)
                .collect::<Vec<_>>(),
            [1]
        );
        let term = crate::query::SparseTermQuery::new(field, 1, 1.0).with_exhaustive(true);
        let hits = index.search(&term, 2).await.unwrap().hits;
        assert_eq!(
            hits.iter()
                .map(|hit| hit.address.doc_id)
                .collect::<Vec<_>>(),
            [2, 1]
        );
    }
}

#[tokio::test]
async fn sparse_vector_and_term_queries_share_default_ordinal_combination() {
    let (index, field, _) = fixture().await;
    let vector = SparseVectorQuery::new(field, vec![(2, 1.0)]).with_exhaustive(true);
    let term = crate::query::SparseTermQuery::new(field, 2, 1.0).with_exhaustive(true);
    let vector_hits = index.search(&vector, 10).await.unwrap().hits;
    let term_hits = index.search(&term, 10).await.unwrap().hits;
    let scores = |hits: &[crate::query::SearchHit]| {
        hits.iter()
            .map(|hit| (hit.address.clone(), hit.score))
            .collect::<Vec<_>>()
    };
    assert_eq!(scores(&vector_hits), scores(&term_hits));
    let multi = vector_hits
        .iter()
        .find(|hit| hit.address.doc_id == 0)
        .unwrap();
    assert_eq!(
        multi.score,
        MultiValueCombiner::default().combine(&[(0, -4.0), (1, 6.0)])
    );
}

#[tokio::test]
async fn signed_multivalue_queries_match_full_forward_scores_for_every_combiner() {
    let (index, field, _) = fixture().await;
    let reader = index.segment_readers().await.unwrap().pop().unwrap();
    for combiner in [
        MultiValueCombiner::Sum,
        MultiValueCombiner::Max,
        MultiValueCombiner::Avg,
        MultiValueCombiner::LogSumExp { temperature: 0.7 },
        MultiValueCombiner::WeightedTopK { k: 2, decay: 0.7 },
    ] {
        let expected = combiner.combine(&[(0, 0.0), (1, 9.0)]);
        for exhaustive in [false, true] {
            let query = SparseVectorQuery::new(field, vec![(2, 1.0), (2, 0.5), (90_000, -2.0)])
                .with_combiner(combiner)
                .with_seismic_factor(0.0)
                .with_exhaustive(exhaustive);
            let mut scorer = query.scorer(&reader, 10).await.unwrap();
            let (hits, _) = scorer.precomputed_top_k(10, true).unwrap();
            assert_eq!(hits.len(), 3);
            let doc = hits.iter().find(|h| h.doc_id == 0).unwrap();
            assert!(
                (doc.score - expected).abs() < 1e-6,
                "{combiner:?}: {} != {expected}",
                doc.score
            );
            assert_eq!(doc.positions[0].1.len(), 2);
            assert_eq!(hits.iter().find(|h| h.doc_id == 4).unwrap().score, -3.0);
            #[cfg(feature = "sync")]
            {
                let mut synchronous = query.scorer_sync(&reader, 10).unwrap();
                let (sync_hits, _) = synchronous.precomputed_top_k(10, true).unwrap();
                assert_eq!(
                    hits.iter().map(|h| (h.doc_id, h.score)).collect::<Vec<_>>(),
                    sync_hits
                        .iter()
                        .map(|h| (h.doc_id, h.score))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[tokio::test]
async fn sparse_filters_apply_before_heap_and_keep_negative_matches() {
    let (index, field, tag) = fixture().await;
    let query = FilteredQuery::new(
        Arc::new(SparseVectorQuery::new(field, vec![(2, 1.0)]).with_exhaustive(true)),
        vec![Arc::new(TermQuery::text(tag, "allowed"))],
    );
    let result = index.search(&query, 1).await.unwrap();
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].address.doc_id, 4);
    assert_eq!(result.hits[0].score, -2.0);
}

#[tokio::test]
async fn negative_scores_and_ties_preserve_cross_segment_order() {
    let mut schema = Schema::builder();
    let field = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::Seismic,
            ..Default::default()
        },
    );
    let directory = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(directory.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for weight in [-2.0, -1.0, -1.0, -3.0] {
        let mut doc = Document::new();
        doc.add_sparse_vector(field, vec![(1, weight)]);
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
    }
    let index = Index::open(directory, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    for exhaustive in [false, true] {
        let query = SparseVectorQuery::new(field, vec![(1, 1.0)]).with_exhaustive(exhaustive);
        let hits = searcher.search(&query, 4).await.unwrap();
        assert_eq!(
            hits.iter().map(|h| h.score).collect::<Vec<_>>(),
            [-1.0, -1.0, -2.0, -3.0]
        );
        assert!(hits[0].segment_id < hits[1].segment_id);
        let top = searcher.search(&query, 1).await.unwrap();
        assert_eq!(
            (top[0].segment_id, top[0].doc_id),
            (hits[0].segment_id, hits[0].doc_id)
        );
        #[cfg(feature = "sync")]
        {
            let sync = searcher
                .search_with_offset_and_count_sync(&query, 4, 0)
                .unwrap()
                .0;
            assert_eq!(hits, sync);
        }
    }
}

#[tokio::test]
async fn overflowing_cluster_proxy_preserves_finite_document_scores() {
    let mut schema = Schema::builder();
    let field = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::Seismic,
            seismic: crate::structures::SeismicConfig {
                postings: 2,
                cluster_size: 2,
                summary_energy: 1.0,
            },
            ..Default::default()
        },
    );
    let directory = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(directory.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    // The shared dimension puts both rows in one nomination cluster;
    // their large coordinates occur in different documents.
    for dimension in [1, 2] {
        let mut doc = Document::new();
        doc.add_sparse_vector(field, vec![(0, 1.0), (dimension, f32::MAX)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(directory, config).await.unwrap();
    for exhaustive in [false, true] {
        let query = SparseVectorQuery::new(field, vec![(0, 1.0), (1, 1.0), (2, 1.0)])
            .with_combiner(MultiValueCombiner::Max)
            .with_exhaustive(exhaustive);
        let hits = index.search(&query, 2).await.unwrap().hits;
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|hit| hit.score == f32::MAX));
    }
}

#[tokio::test]
async fn selective_filters_find_matches_outside_retained_top_l_nominations() {
    let mut schema = Schema::builder();
    let field = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::Seismic,
            seismic: crate::structures::SeismicConfig {
                postings: 1,
                cluster_size: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let tag = schema.add_text_field("tag", true, false);
    let directory = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(directory.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for (tag_value, weight) in [("excluded", 100.0), ("allowed", 0.1)] {
        let mut doc = Document::new();
        doc.add_text(tag, tag_value);
        doc.add_sparse_vector(field, vec![(1, weight)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(directory, config).await.unwrap();
    let sparse = SparseVectorQuery::new(field, vec![(1, 1.0)]);
    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(FilteredQuery::new(
            Arc::new(sparse.clone()),
            vec![Arc::new(TermQuery::text(tag, "allowed"))],
        )),
        Box::new(
            crate::query::BooleanQuery::new()
                .must(TermQuery::text(tag, "allowed"))
                .should(sparse),
        ),
    ];
    for query in queries {
        let hits = index.search(query.as_ref(), 1).await.unwrap().hits;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].address.doc_id, 1);
        assert_eq!(hits[0].score, 0.1);
    }
}

#[tokio::test]
async fn complete_ordinal_combination_includes_empty_and_nonmatching_values() {
    let mut schema = Schema::builder();
    let field = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::Seismic,
            ..Default::default()
        },
    );
    let directory = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(directory.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    let mut doc = Document::new();
    doc.add_sparse_vector(field, vec![(2, -3.0)]);
    doc.add_sparse_vector(field, Vec::new());
    doc.add_sparse_vector(field, vec![(9, 4.0)]);
    writer.add_document(doc).unwrap();
    writer.commit().await.unwrap();
    let index = Index::open(directory, config).await.unwrap();
    let segment = index.segment_readers().await.unwrap().pop().unwrap();
    for (combiner, expected) in [
        (MultiValueCombiner::Max, 0.0),
        (MultiValueCombiner::Avg, -1.0),
        (MultiValueCombiner::Sum, -3.0),
    ] {
        let query = SparseVectorQuery::new(field, vec![(2, 1.0)])
            .with_combiner(combiner)
            .with_exhaustive(true);
        let mut scorer = query.scorer(&segment, 1).await.unwrap();
        let (hits, _) = scorer.precomputed_top_k(1, true).unwrap();
        assert_eq!(hits[0].score, expected);
        assert_eq!(hits[0].positions[0].1.len(), 1);
        assert_eq!(hits[0].positions[0].1[0].position, 0);
        let required = query
            .scorer_with_options(
                &segment,
                1,
                ScorerOptions {
                    complete_text_matches: true,
                    ..ScorerOptions::with_positions()
                },
            )
            .await
            .unwrap();
        assert_eq!(required.score(), expected);
        let positions = required.matched_positions().unwrap();
        assert_eq!(positions[0].1.len(), 1);
        assert_eq!(positions[0].1[0].position, 0);
    }
}

#[tokio::test]
async fn segment_heap_shallower_than_result_window_never_publishes_floor() {
    let (index, field, _) = fixture().await;
    let segment = index.segment_readers().await.unwrap().pop().unwrap();
    let query = SparseVectorQuery::new(field, vec![(2, 1.0)])
        .with_combiner(MultiValueCombiner::Sum)
        .with_exhaustive(true);
    let shared = super::super::SharedThreshold::for_limit(3);
    let mut scorer = query
        .scorer_with_options(
            &segment,
            1,
            ScorerOptions {
                shared_threshold: Some(shared.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(scorer.precomputed_top_k(1, false).unwrap().0.len(), 1);
    assert_eq!(shared.get(), 0.0);
}

#[tokio::test]
async fn required_sparse_scorer_preserves_complete_membership_at_limit_one() {
    let (index, field, _) = fixture().await;
    let reader = index.segment_readers().await.unwrap().pop().unwrap();
    let query = SparseVectorQuery::new(field, vec![(2, 1.0)]);
    let mut scorer = query
        .scorer_with_options(
            &reader,
            1,
            ScorerOptions::with_positions().for_required_clause(),
        )
        .await
        .unwrap();
    let mut docs = Vec::new();
    while scorer.doc() != crate::TERMINATED {
        docs.push(scorer.doc());
        scorer.advance();
    }
    assert_eq!(docs, [0, 1, 4]);
}

#[tokio::test]
async fn sparse_membership_is_complete_even_when_nomination_dimensions_are_pruned() {
    let (index, field, _) = fixture().await;
    let reader = index.segment_readers().await.unwrap().pop().unwrap();
    let query = SparseVectorQuery::new(field, vec![(2, 1.0), (3, 0.1)]).with_max_query_dims(1);
    let bits = query
        .as_doc_bitset_with_options(&reader, &ScorerOptions::default())
        .unwrap();
    assert!(bits.contains(2));
    assert!(!bits.contains(3));
}
