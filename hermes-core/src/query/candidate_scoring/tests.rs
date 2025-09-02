use super::*;
use crate::dsl::{DenseVectorConfig, DenseVectorQuantization, PositionMode, VectorIndexType};
use crate::query::{
    DenseVectorQuery, MultiValueCombiner, PhraseQuery, Query, SparseVectorQuery, TermQuery,
};
use crate::structures::{SparseFormat, SparseVectorConfig, WeightQuantization};
use crate::{Document, Index, IndexConfig, IndexWriter, RamDirectory, Schema};

#[tokio::test]
async fn dense_only_candidate_gets_exact_bm25_phrase_sparse_and_negative_dense_features() {
    let mut schema = Schema::builder();
    let text = schema.add_text_field_with_tokenizer("body", true, true, "simple");
    schema.set_chunked(text, true);
    schema.set_positions(text, PositionMode::TokenPosition);
    let profile = schema.add_text_field_with_tokenizer("profile", true, true, "simple");
    let sparse = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::Bmp,
            dims: Some(32),
            weight_quantization: WeightQuantization::UInt8,
            ..Default::default()
        },
    );
    let dense = schema.add_dense_vector_field_with_config(
        "dense",
        true,
        false,
        DenseVectorConfig {
            dim: 2,
            index_type: VectorIndexType::Flat,
            quantization: DenseVectorQuantization::F32,
            num_clusters: None,
            target_vectors: None,
            tree_levels: None,
            ivf_routing: crate::dsl::IvfRoutingMode::Auto,
            nprobe: 1,
            unit_norm: false,
            soar: None,
        },
    );
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    let mut document = Document::new();
    document.add_text(text, "irrelevant topic");
    document.add_text(text, "hemoglobin carries oxygen hemoglobin carries oxygen");
    document.add_text(profile, "medical reference");
    document.add_sparse_vector(sparse, vec![(2, 0.5)]);
    document.add_sparse_vector(sparse, vec![(1, 0.7), (2, 0.9)]);
    document.add_dense_vector(dense, vec![-1.0, 0.0]);
    document.add_dense_vector(dense, vec![0.5, 0.5]);
    writer.add_document(document).unwrap();
    let mut missing = Document::new();
    missing.add_text(profile, "medical reference");
    writer.add_document(missing).unwrap();
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    let dense_query = DenseVectorQuery::new(dense, vec![1.0, 0.0]);
    let candidates = searcher
        .search_with_positions(&dense_query, 1)
        .await
        .unwrap()
        .0;
    assert_eq!(candidates.len(), 1);
    let term = TermQuery::text(text, "hemoglobin");
    let phrase = PhraseQuery::text(text, "carries oxygen");
    let sparse_query = SparseVectorQuery::new(sparse, vec![(1, 0.7), (2, 0.3)])
        .with_combiner(MultiValueCombiner::Max);
    let profile_query = TermQuery::text(profile, "medical");
    let plan = CandidateScoringPlan {
        features: vec![
            CandidateFeature {
                name: "bm25".into(),
                scope: ScoreScope::Chunk,
                query: term.candidate_query().unwrap(),
            },
            CandidateFeature {
                name: "phrase".into(),
                scope: ScoreScope::Chunk,
                query: phrase.candidate_query().unwrap(),
            },
            CandidateFeature {
                name: "sparse".into(),
                scope: ScoreScope::Chunk,
                query: sparse_query.candidate_query().unwrap(),
            },
            CandidateFeature {
                name: "dense".into(),
                scope: ScoreScope::Chunk,
                query: dense_query.candidate_query().unwrap(),
            },
            CandidateFeature {
                name: "profile".into(),
                scope: ScoreScope::Document,
                query: profile_query.candidate_query().unwrap(),
            },
        ],
        model: Some(LinearModel {
            weights: std::collections::BTreeMap::from([
                ("bm25".into(), 1.0),
                ("dense".into(), 1.0),
                ("profile".into(), 0.2),
            ]),
            ..Default::default()
        }),
        export_passages: 10,
        all_passages: true,
    };
    let scored = searcher
        .score_candidates(&candidates, &plan, None)
        .await
        .unwrap();
    let rows = &scored[0].features.passages;
    let matching = rows.iter().find(|p| p.ordinal == 1).unwrap();
    let irrelevant = rows.iter().find(|p| p.ordinal == 0).unwrap();
    assert_eq!(irrelevant.values[0], Some(0.0));
    assert_eq!(irrelevant.values[1], Some(0.0));
    assert!(irrelevant.values[3].unwrap() < -0.99);
    assert!(matching.values[0].unwrap() > 0.0);
    assert!(matching.values[1].unwrap() > 0.0);
    assert!(scored[0].features.document[4].unwrap() > 0.0);
    assert!(
        matching.values[4].is_none(),
        "document context is not a body chunk feature"
    );
    for (feature, query) in [(0, &term as &dyn Query), (1, &phrase), (2, &sparse_query)] {
        let exhaustive = searcher.search_with_positions(query, 10).await.unwrap().0;
        let expected = exhaustive[0]
            .positions
            .iter()
            .flat_map(|(_, p)| p)
            .find(|p| p.position == 1)
            .unwrap()
            .score;
        assert!(
            (matching.values[feature].unwrap() - expected).abs() < 1e-5,
            "feature {feature}"
        );
    }
    assert_eq!(scored[0].result.score, matching.score);
    let mut nominated = candidates[0].clone();
    nominated.positions = vec![(dense.0, vec![crate::query::ScoredPosition::new(0, -1.0)])];
    let mut passage_plan = plan.clone();
    passage_plan.all_passages = false;
    let passage_scores = searcher
        .score_candidates(&[nominated], &passage_plan, None)
        .await
        .unwrap();
    assert_eq!(passage_scores[0].features.scored_passages, 1);
    assert_eq!(passage_scores[0].features.passages[0].ordinal, 0);
    assert_eq!(
        passage_scores[0].features.passages[0].values,
        irrelevant.values
    );
    assert_eq!(passage_scores[0].result.score, irrelevant.score);

    // A document-only candidate must remain an explicit document row.
    let missing_candidate = crate::query::SearchResult {
        doc_id: 1,
        segment_id: candidates[0].segment_id,
        score: 100.0,
        positions: vec![],
    };
    let scored_missing = searcher
        .score_candidates(&[missing_candidate], &plan, None)
        .await
        .unwrap();
    assert!(scored_missing[0].features.passages.is_empty());
    assert!(
        scored_missing[0]
            .result
            .positions
            .iter()
            .all(|(_, p)| p.is_empty())
    );
    assert!(scored_missing[0].features.document[4].is_some());
    let mut stale = candidates[0].clone();
    stale.segment_id = 123;
    assert!(
        searcher
            .score_candidates(&[stale], &plan, None)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn absent_text_in_an_entire_segment_is_missing_not_zero_or_unsupported() {
    let mut schema = Schema::builder();
    let title = schema.add_text_field_with_tokenizer("title", true, false, "simple");
    let profile = schema.add_text_field_with_tokenizer("profile", true, false, "simple");
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    let mut present = Document::new();
    present.add_text(title, "candidate");
    present.add_text(profile, "medicine");
    writer.add_document(present).unwrap();
    writer.commit().await.unwrap();
    let mut missing = Document::new();
    missing.add_text(title, "candidate");
    writer.add_document(missing).unwrap();
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    let candidates = searcher
        .search(&TermQuery::text(title, "candidate"), 10)
        .await
        .unwrap();
    let plan = CandidateScoringPlan {
        features: vec![CandidateFeature {
            name: "profile".into(),
            scope: ScoreScope::Document,
            query: TermQuery::text(profile, "hemoglobin")
                .candidate_query()
                .unwrap(),
        }],
        model: Some(LinearModel {
            weights: std::collections::BTreeMap::from([("profile".into(), 1.0)]),
            transforms: std::collections::BTreeMap::from([(
                "profile".into(),
                FeatureTransform {
                    offset: 1.0,
                    ..Default::default()
                },
            )]),
            ..Default::default()
        }),
        export_passages: 1,
        all_passages: false,
    };
    let scored = searcher
        .score_candidates(&candidates, &plan, None)
        .await
        .unwrap();
    assert_eq!(scored.len(), 2);
    assert_eq!(scored[0].features.document, vec![Some(0.0)]);
    assert_eq!(scored[0].result.score, 1.0);
    assert_eq!(scored[1].features.document, vec![None]);
    assert_eq!(scored[1].result.score, 0.0);
}
