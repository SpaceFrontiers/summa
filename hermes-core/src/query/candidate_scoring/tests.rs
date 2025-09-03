use super::*;
use crate::dsl::{DenseVectorConfig, DenseVectorQuantization, PositionMode, VectorIndexType};
use crate::query::{
    DenseVectorQuery, MultiValueCombiner, PhraseQuery, Query, SparseVectorQuery, TermQuery,
};
use crate::structures::{SparseFormat, SparseVectorConfig, WeightQuantization};
use crate::{Document, Index, IndexConfig, IndexWriter, RamDirectory, Schema};

#[tokio::test]
async fn l1_preserves_organic_zero_and_negative_scores_and_backfills_only_missing_cells() {
    let mut schema = Schema::builder();
    let field = schema.add_dense_vector_field_with_config(
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
    for vector in [vec![1.0, 0.0], vec![-1.0, 0.0]] {
        let mut doc = Document::new();
        doc.add_dense_vector(field, vector);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    let segment = searcher.segment_readers()[0].meta().id;
    let hit = |doc_id, score| crate::query::SearchResult {
        segment_id: segment,
        doc_id,
        score,
        positions: vec![(field.0, vec![crate::query::ScoredPosition::new(0, score)])],
    };
    let first = vec![hit(0, 0.0)];
    let second = vec![hit(1, -0.4)];
    let candidates = searcher
        .merge_candidate_lists([first.clone(), second.clone()])
        .unwrap();
    let mut plan = CandidateScoringPlan {
        backfill: false,
        features: ["x", "y"]
            .into_iter()
            .zip([vec![1.0, 0.0], vec![0.0, 1.0]])
            .map(|(name, vector)| CandidateFeature {
                name: name.into(),
                scope: ScoreScope::Chunk,
                query: DenseVectorQuery::new(field, vector)
                    .candidate_query()
                    .unwrap(),
            })
            .collect(),
        model: Some(LinearModel {
            weights: std::collections::BTreeMap::from([("x".into(), 2.0), ("y".into(), 1.0)]),
            missing_values: std::collections::BTreeMap::from([
                ("x".into(), 0.25),
                ("y".into(), 0.75),
            ]),
            transforms: std::collections::BTreeMap::from([(
                "x".into(),
                FeatureTransform {
                    scale: 3.0,
                    offset: 1.0,
                    ..Default::default()
                },
            )]),
            ..Default::default()
        }),
        export_passages: 1,
        all_passages: false,
        document_combiner: MultiValueCombiner::Max,
    };
    let raw = searcher
        .score_candidates_with_retrieved(&candidates, &plan, None, &[(0, &first), (1, &second)])
        .await
        .unwrap();
    let doc = |id| raw.iter().find(|s| s.result.doc_id == id).unwrap();
    assert_eq!(doc(0).features.passages[0].values, vec![Some(0.0), None]);
    assert_eq!(doc(1).features.passages[0].values, vec![None, Some(-0.4)]);
    assert_eq!(doc(0).result.score, 2.75);
    assert!((doc(1).result.score - 3.1).abs() < 1e-6);
    plan.backfill = true;
    let filled = searcher
        .score_candidates_with_retrieved(&candidates, &plan, None, &[(0, &first), (1, &second)])
        .await
        .unwrap();
    let doc = |id| filled.iter().find(|s| s.result.doc_id == id).unwrap();
    assert_eq!(
        doc(0).features.passages[0].values,
        vec![Some(0.0), Some(0.0)]
    );
    assert!((doc(1).features.passages[0].values[0].unwrap() + 1.0).abs() < 2e-5);
    assert_eq!(doc(1).features.passages[0].values[1], Some(-0.4));
}

#[tokio::test]
async fn dense_only_candidate_gets_exact_bm25_phrase_sparse_and_negative_dense_features() {
    cross_vertical_backfill(SparseFormat::Bmp).await;
}

#[tokio::test]
async fn dense_only_candidate_backfills_maxscore_sparse_fields() {
    cross_vertical_backfill(SparseFormat::MaxScore).await;
}

async fn cross_vertical_backfill(sparse_format: SparseFormat) {
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
            format: sparse_format,
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
        backfill: true,
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
        document_combiner: crate::query::MultiValueCombiner::Max,
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

    // Document feature reduction belongs to the query, while final passage
    // reduction belongs to fusion. Neither may be replaced with MAX or run
    // after the response truncates its passage rows.
    let raw_dense = vec![
        (0, irrelevant.values[3].unwrap()),
        (1, matching.values[3].unwrap()),
    ];
    for combiner in [
        MultiValueCombiner::Max,
        MultiValueCombiner::Avg,
        MultiValueCombiner::Sum,
        MultiValueCombiner::LogSumExp { temperature: 0.7 },
        MultiValueCombiner::WeightedTopK { k: 2, decay: 0.4 },
    ] {
        let mut document_plan = plan.clone();
        document_plan.features = vec![CandidateFeature {
            name: "dense".into(),
            scope: ScoreScope::Document,
            query: DenseVectorQuery::new(dense, vec![1.0, 0.0])
                .with_combiner(combiner)
                .candidate_query()
                .unwrap()
                .boosted(-2.0)
                .unwrap(),
        }];
        document_plan.model = Some(LinearModel {
            weights: std::collections::BTreeMap::from([("dense".into(), 1.0)]),
            ..Default::default()
        });
        let actual = searcher
            .score_candidates(&candidates, &document_plan, None)
            .await
            .unwrap();
        let expected = -2.0 * combiner.combine(&raw_dense);
        assert!(
            (actual[0].features.document[0].unwrap() - expected).abs() < 1e-6,
            "{combiner:?}"
        );
        assert!((actual[0].result.score - expected).abs() < 1e-6);
        assert!(actual[0].features.passages.is_empty());

        let mut passage_plan = plan.clone();
        passage_plan.model = Some(LinearModel {
            weights: std::collections::BTreeMap::from([("dense".into(), 1.0)]),
            bias: -2.0,
            ..Default::default()
        });
        passage_plan.document_combiner = combiner;
        passage_plan.export_passages = 1;
        let actual = searcher
            .score_candidates(&candidates, &passage_plan, None)
            .await
            .unwrap();
        let predicted: Vec<_> = raw_dense
            .iter()
            .map(|&(ordinal, score)| (ordinal, score - 2.0))
            .collect();
        assert!(
            (actual[0].result.score - combiner.combine(&predicted)).abs() < 1e-6,
            "{combiner:?}"
        );
        assert_eq!(actual[0].features.scored_passages, 2);
        assert_eq!(actual[0].features.passages.len(), 1);
    }
    let mut composition = plan.clone();
    composition.features = vec![CandidateFeature {
        name: "dense".into(),
        scope: ScoreScope::Document,
        query: CandidateQuery::sum([
            DenseVectorQuery::new(dense, vec![1.0, 0.0])
                .with_combiner(MultiValueCombiner::Max)
                .candidate_query(),
            DenseVectorQuery::new(dense, vec![-1.0, 0.0])
                .with_combiner(MultiValueCombiner::Max)
                .candidate_query(),
        ])
        .unwrap(),
    }];
    composition.model = Some(LinearModel {
        weights: std::collections::BTreeMap::from([("dense".into(), 1.0)]),
        ..Default::default()
    });
    let actual = searcher
        .score_candidates(&candidates, &composition, None)
        .await
        .unwrap();
    assert!(
        (actual[0].result.score - (raw_dense[1].1 - raw_dense[0].1)).abs() < 1e-6,
        "sum of separately reduced vector queries must preserve expression order"
    );

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
        backfill: true,
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
        document_combiner: crate::query::MultiValueCombiner::Max,
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

#[tokio::test]
async fn maxscore_backfill_preserves_ordinals_across_block_boundaries_and_distinguishes_missing() {
    let mut schema = Schema::builder();
    let field = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::MaxScore,
            dims: Some(8),
            weight_quantization: WeightQuantization::UInt8,
            ..Default::default()
        },
    );
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for doc in 0..3 {
        let mut document = Document::new();
        if doc == 0 {
            for _ in 0..513 {
                document.add_sparse_vector(field, vec![(1, 0.8)]);
            }
        } else if doc == 1 {
            document.add_sparse_vector(field, vec![(7, 0.5)]);
        }
        writer.add_document(document).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    let segment_id = searcher.segment_readers()[0].meta().id;
    let candidates: Vec<_> = (0..3)
        .map(|doc_id| crate::query::SearchResult {
            segment_id,
            doc_id,
            score: 0.0,
            positions: Vec::new(),
        })
        .collect();
    let plan = CandidateScoringPlan {
        features: vec![CandidateFeature {
            name: "sparse".into(),
            scope: ScoreScope::Chunk,
            query: SparseVectorQuery::new(field, vec![(1, 0.25)])
                .candidate_query()
                .unwrap(),
        }],
        backfill: true,
        model: None,
        export_passages: 1024,
        all_passages: true,
        document_combiner: MultiValueCombiner::Max,
    };
    let result = searcher
        .score_candidates(&candidates, &plan, None)
        .await
        .unwrap();
    let scored = result.iter().find(|c| c.result.doc_id == 0).unwrap();
    assert_eq!(scored.features.scored_passages, 513);
    for row in &scored.features.passages {
        assert!((row.values[0].unwrap() - 0.2).abs() < 0.002);
    }
    let nonmatch = result.iter().find(|c| c.result.doc_id == 1).unwrap();
    assert_eq!(nonmatch.features.passages[0].values, vec![Some(0.0)]);
    let missing = result.iter().find(|c| c.result.doc_id == 2).unwrap();
    assert!(missing.features.passages.is_empty());
    assert_eq!(missing.features.document, vec![None]);
}

#[tokio::test]
async fn complete_organic_scores_skip_legacy_addressing_and_reorder_upgrades_small_text_segments() {
    use crate::directories::{Directory, DirectoryWriter};
    let mut schema = Schema::builder();
    let field = schema.add_text_field_with_tokenizer("body", true, false, "simple");
    schema.set_chunked(field, true);
    schema.set_reorder(field, true);
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for _ in 0..2 {
        let mut doc = Document::new();
        doc.add_text(field, "shared text");
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    let files = crate::segment::SegmentFiles::new(searcher.segment_readers()[0].meta().id);
    let mut bytes = dir
        .open_read(&files.chunks)
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap()
        .to_vec();
    // Model a valid legacy BP permutation: both chunks contain identical text.
    let offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
    bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
    bytes[offset..offset + 4].copy_from_slice(&1u32.to_le_bytes());
    bytes[offset + 4..offset + 8].copy_from_slice(&0u32.to_le_bytes());
    dir.write(&files.chunks, &bytes).await.unwrap();
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    assert_eq!(
        searcher.segment_readers()[0].unprepared_candidate_fields(),
        vec!["body"]
    );
    let query = TermQuery::text(field, "shared");
    let hits = searcher.search_with_positions(&query, 2).await.unwrap().0;
    let plan = CandidateScoringPlan {
        features: vec![CandidateFeature {
            name: "body".into(),
            scope: ScoreScope::Document,
            query: query.candidate_query().unwrap(),
        }],
        backfill: true,
        model: None,
        export_passages: 2,
        all_passages: false,
        document_combiner: MultiValueCombiner::Max,
    };
    let scored = searcher
        .score_candidates_with_retrieved(&hits, &plan, None, &[(0, &hits)])
        .await
        .unwrap();
    for candidate in scored {
        let original = hits
            .iter()
            .find(|h| h.doc_id == candidate.result.doc_id)
            .unwrap();
        assert_eq!(candidate.features.document, vec![Some(original.score)]);
    }
    assert!(
        searcher.score_candidates(&hits, &plan, None).await.is_err(),
        "missing cells still require addressing"
    );
    writer.reorder().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    assert!(
        searcher.segment_readers()[0]
            .unprepared_candidate_fields()
            .is_empty()
    );
    let hits = searcher.search_with_positions(&query, 2).await.unwrap().0;
    assert_eq!(
        searcher
            .score_candidates(&hits, &plan, None)
            .await
            .unwrap()
            .len(),
        2
    );
}

#[tokio::test]
async fn bmp_backfill_without_forward_storage_preserves_missing_zero_and_organic_scores() {
    let mut schema = Schema::builder();
    let title = schema.add_text_field("title", true, false);
    let field = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::Bmp,
            dims: Some(8),
            max_weight: Some(5.0),
            bmp_forward_index: false,
            ..Default::default()
        },
    );
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for id in 0..3 {
        let mut doc = Document::new();
        doc.add_text(title, "candidate");
        if id == 0 {
            doc.add_sparse_vector(field, vec![(1, 0.8)]);
            doc.add_sparse_vector(field, vec![(1, 0.2)]);
        } else if id == 1 {
            doc.add_sparse_vector(field, vec![(3, 1.0)]);
        }
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    assert!(
        searcher.segment_readers()[0]
            .bmp_index(field)
            .unwrap()
            .forward()
            .is_none()
    );
    let candidates = searcher
        .search(&TermQuery::text(title, "candidate"), 10)
        .await
        .unwrap();
    let sparse_query =
        SparseVectorQuery::new(field, vec![(1, 1.0)]).with_combiner(MultiValueCombiner::Max);
    let organic = searcher.search(&sparse_query, 10).await.unwrap();
    let mut plan = CandidateScoringPlan {
        backfill: true,
        features: vec![CandidateFeature {
            name: "sparse".into(),
            scope: ScoreScope::Document,
            query: sparse_query.candidate_query().unwrap(),
        }],
        model: Some(LinearModel {
            weights: std::collections::BTreeMap::from([("sparse".into(), 2.0)]),
            missing_values: std::collections::BTreeMap::from([("sparse".into(), -0.5)]),
            ..Default::default()
        }),
        export_passages: 1,
        all_passages: false,
        document_combiner: MultiValueCombiner::Max,
    };
    let filled = searcher
        .score_candidates(&candidates, &plan, None)
        .await
        .unwrap();
    let doc = |id| filled.iter().find(|s| s.result.doc_id == id).unwrap();
    assert_eq!(
        doc(0).features.document[0].unwrap().to_bits(),
        organic[0].score.to_bits()
    );
    assert_eq!(doc(1).features.document, vec![Some(0.0)]);
    assert_eq!(doc(2).features.document, vec![None]);
    assert_eq!(doc(2).result.score, -1.0);
    let mut known = organic;
    known[0].score = 17.0;
    let reused = searcher
        .score_candidates_with_retrieved(&candidates, &plan, None, &[(0, &known)])
        .await
        .unwrap();
    assert_eq!(
        reused
            .iter()
            .find(|s| s.result.doc_id == 0)
            .unwrap()
            .features
            .document,
        vec![Some(17.0)]
    );
    plan.backfill = false;
    let unfilled = searcher
        .score_candidates_with_retrieved(&candidates, &plan, None, &[(0, &known)])
        .await
        .unwrap();
    assert_eq!(
        unfilled
            .iter()
            .find(|s| s.result.doc_id == 1)
            .unwrap()
            .features
            .document,
        vec![None]
    );
    assert_eq!(
        unfilled
            .iter()
            .find(|s| s.result.doc_id == 1)
            .unwrap()
            .result
            .score,
        -1.0
    );
}
