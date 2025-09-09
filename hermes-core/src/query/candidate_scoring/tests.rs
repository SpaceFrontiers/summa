use super::*;
use crate::dsl::{DenseVectorConfig, DenseVectorQuantization, PositionMode, VectorIndexType};
use crate::query::{
    DenseVectorQuery, MultiValueCombiner, PhraseQuery, Query, SparseVectorQuery, TermQuery,
};
use crate::structures::{SparseFormat, SparseVectorConfig, WeightQuantization};
use crate::{Document, Index, IndexConfig, IndexWriter, RamDirectory, Schema};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_backfill_keeps_distinct_queries_and_quantizations_correct_across_segments() {
    let mut schema = Schema::builder();
    let dense: Vec<_> = [
        DenseVectorQuantization::F32,
        DenseVectorQuantization::F16,
        DenseVectorQuantization::UInt8,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, quantization)| {
        schema.add_dense_vector_field_with_config(
            &format!("dense{i}"),
            true,
            false,
            DenseVectorConfig {
                dim: 4,
                index_type: VectorIndexType::Flat,
                quantization,
                num_clusters: None,
                target_vectors: None,
                tree_levels: None,
                ivf_routing: crate::dsl::IvfRoutingMode::Auto,
                nprobe: 1,
                unit_norm: false,
                soar: None,
            },
        )
    })
    .collect();
    let sparse = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::Bmp,
            dims: Some(16),
            max_weight: Some(2.0),
            ..Default::default()
        },
    );
    let binary = schema.add_binary_dense_vector_field("binary", 16, true, false);
    let directory = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(directory.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for segment in 0..2 {
        for doc in 0..3 {
            let mut document = Document::new();
            // A missing row, one value, and a value set exceeding inline scratch.
            for ordinal in 0..[0, 1, 20][doc] {
                let x = (segment + ordinal + 1) as f32 / 25.0;
                for &field in &dense {
                    document.add_dense_vector(field, vec![x, 1.0 - x, 0.2, -0.3]);
                }
                document.add_sparse_vector(sparse, vec![(0, x), (1, 1.0 - x)]);
                document.add_binary_dense_vector(binary, vec![ordinal as u8, segment as u8]);
            }
            writer.add_document(document).unwrap();
        }
        writer.commit().await.unwrap();
    }
    let index = Index::open(directory, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    assert_eq!(searcher.segment_readers().len(), 2);
    let candidates: Vec<_> = searcher
        .segment_readers()
        .iter()
        .flat_map(|reader| {
            (0..reader.num_docs()).map(move |doc_id| crate::query::SearchResult {
                doc_id,
                score: 0.0,
                segment_id: reader.meta().id,
                positions: Vec::new(),
            })
        })
        .collect();
    let mut features = Vec::new();
    for (i, &field) in dense.iter().enumerate() {
        for (j, vector) in [vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 2.0, 0.0, 0.0]]
            .into_iter()
            .enumerate()
        {
            features.push(CandidateFeature {
                name: format!("dense_{i}_{j}"),
                scope: ScoreScope::Document,
                query: DenseVectorQuery::new(field, vector)
                    .with_combiner(MultiValueCombiner::Avg)
                    .candidate_query()
                    .unwrap(),
            });
        }
    }
    for (i, terms) in [vec![(0, 0.3), (0, 0.1)], vec![(1, 0.8)]]
        .into_iter()
        .enumerate()
    {
        features.push(CandidateFeature {
            name: format!("sparse_{i}"),
            scope: ScoreScope::Document,
            query: SparseVectorQuery::new(sparse, terms)
                .with_combiner(MultiValueCombiner::Sum)
                .candidate_query()
                .unwrap(),
        });
    }
    for (i, vector) in [vec![0, 0], vec![255, 0]].into_iter().enumerate() {
        features.push(CandidateFeature {
            name: format!("binary_{i}"),
            scope: ScoreScope::Document,
            query: crate::query::BinaryDenseVectorQuery::new(binary, vector)
                .candidate_query()
                .unwrap(),
        });
    }
    let plan = CandidateScoringPlan {
        features,
        backfill: true,
        model: None,
        export_passages: 1,
        all_passages: false,
        seed_document_passages: false,
        document_combiner: MultiValueCombiner::Max,
    };
    let combined = searcher
        .score_candidates(&candidates, &plan, None)
        .await
        .unwrap();
    for (i, feature) in plan.features.iter().enumerate() {
        let isolated = CandidateScoringPlan {
            features: vec![feature.clone()],
            ..plan.clone()
        };
        for candidate in &candidates {
            // Fresh request and one document: no cross-segment/component cache reuse.
            let expected = searcher
                .score_candidates(std::slice::from_ref(candidate), &isolated, None)
                .await
                .unwrap();
            let actual = combined
                .iter()
                .find(|hit| {
                    hit.result.segment_id == candidate.segment_id
                        && hit.result.doc_id == candidate.doc_id
                })
                .unwrap();
            assert_eq!(
                actual.features.document[i].map(f32::to_bits),
                expected[0].features.document[0].map(f32::to_bits),
                "{}",
                feature.name
            );
        }
    }
}

#[tokio::test]
async fn index_creation_configures_phrase_limits_for_ranking_and_collection_after_reopen() {
    for configured in [None, Some(1), Some(65), Some(300)] {
        let option = configured
            .map(|limit| format!("max_l1_phrase_terms: {limit}"))
            .unwrap_or_default();
        let schema = crate::parse_schema(&format!(
            "index documents {{ {option} field body: text<simple> [indexed<token_position>] }}"
        ))
        .unwrap();
        let field = schema.get_field("body").unwrap();
        let directory = RamDirectory::new();
        let config = IndexConfig::default();
        let created = Index::create(directory.clone(), schema, config.clone())
            .await
            .unwrap();
        drop(created);
        let reopened = Index::open(directory.clone(), config).await.unwrap();
        let searcher = reopened.reader().await.unwrap().searcher().await.unwrap();
        let limit = configured.unwrap_or(64);
        for ranked in [false, true] {
            for count in [limit, limit + 1] {
                let plan = CandidateScoringPlan {
                    features: vec![CandidateFeature {
                        name: "phrase".into(),
                        scope: ScoreScope::Document,
                        query: PhraseQuery::new(field, vec![b"term".to_vec(); count])
                            .candidate_query()
                            .unwrap(),
                    }],
                    backfill: true,
                    model: ranked.then(|| {
                        RankingModel::compile("phrase", &["phrase"], &Default::default()).unwrap()
                    }),
                    export_passages: 1,
                    all_passages: !ranked,
                    seed_document_passages: false,
                    document_combiner: MultiValueCombiner::Max,
                };
                let result = searcher.score_candidates(&[], &plan, None).await;
                if count == limit {
                    assert!(result.unwrap().is_empty());
                } else {
                    let error =
                        result.expect_err("reject oversized phrases even without candidates");
                    assert!(
                        error
                            .to_string()
                            .contains(&format!("{count} terms; maximum is {limit}")),
                        "{error}"
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn long_phrase_features_keep_every_term_in_ranking_and_collection() {
    let mut schema = Schema::builder();
    schema.set_max_l1_phrase_terms(std::num::NonZeroU32::new(300).unwrap());
    let plain = schema.add_text_field_with_tokenizer("plain", true, false, "simple");
    let chunked = schema.add_text_field_with_tokenizer("chunked", true, false, "simple");
    schema.set_chunked(chunked, true);
    for field in [plain, chunked] {
        schema.set_positions(field, PositionMode::TokenPosition);
    }
    let terms: Vec<_> = (0..300).map(|i| format!("term{i}")).collect();
    let mut wrong_word = terms.clone();
    wrong_word[64] = "different".into();
    let directory = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(directory.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for text in [
        terms.join(" "),
        terms[..64].join(" "),
        wrong_word.join(" "),
        terms[..255].join(" "),
    ] {
        let mut document = Document::new();
        for field in [plain, chunked] {
            document.add_text(field, &text);
        }
        writer.add_document(document).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(directory, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    for (field, scope) in [(plain, ScoreScope::Document), (chunked, ScoreScope::Chunk)] {
        let candidates = searcher
            .search_with_positions(&TermQuery::text(field, "term0"), 4)
            .await
            .unwrap()
            .0;
        assert_eq!(candidates.len(), 4);
        for count in [64, 65, 256, 300] {
            let phrase = PhraseQuery::text(field, &terms[..count].join(" "));
            let reference = searcher.search_with_positions(&phrase, 4).await.unwrap().0;
            let mut matching: Vec<_> = reference.iter().map(|hit| hit.doc_id).collect();
            matching.sort_unstable();
            assert_eq!(
                matching,
                match count {
                    64 => vec![0, 1, 2, 3],
                    65 => vec![0, 3],
                    _ => vec![0],
                }
            );
            for ranked in [false, true] {
                let plan = CandidateScoringPlan {
                    features: vec![CandidateFeature {
                        name: "phrase".into(),
                        scope,
                        query: phrase.candidate_query().unwrap(),
                    }],
                    backfill: true,
                    model: ranked.then(|| {
                        RankingModel::compile("phrase", &["phrase"], &Default::default()).unwrap()
                    }),
                    export_passages: 1,
                    all_passages: !ranked,
                    seed_document_passages: false,
                    document_combiner: MultiValueCombiner::Max,
                };
                let scored = searcher
                    .score_candidates(&candidates, &plan, None)
                    .await
                    .unwrap_or_else(|error| {
                        panic!("{count}-term phrase, {scope:?}, ranked={ranked}: {error}")
                    });
                for candidate in scored {
                    let expected = reference
                        .iter()
                        .find(|hit| hit.doc_id == candidate.result.doc_id)
                        .map_or(0.0, |hit| hit.score);
                    let actual = match scope {
                        ScoreScope::Document => candidate.features.document[0].unwrap(),
                        ScoreScope::Chunk => {
                            assert_eq!(candidate.features.passages.len(), 1);
                            assert_eq!(candidate.features.passages[0].ordinal, 0);
                            candidate.features.passages[0].values[0].unwrap()
                        }
                    };
                    assert_eq!(actual.to_bits(), expected.to_bits(), "{count}-term phrase");
                    if ranked {
                        assert_eq!(candidate.result.score.to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }
    let oversized = CandidateScoringPlan {
        features: vec![CandidateFeature {
            name: "phrase".into(),
            scope: ScoreScope::Document,
            query: PhraseQuery::new(plain, vec![b"term0".to_vec(); 301])
                .candidate_query()
                .unwrap(),
        }],
        backfill: true,
        model: None,
        export_passages: 1,
        all_passages: false,
        seed_document_passages: false,
        document_combiner: MultiValueCombiner::Max,
    };
    let error = searcher
        .score_candidates(&[], &oversized, None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("301 terms; maximum is 300"));
}

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
        model: Some(
            RankingModel::compile(
                "2 * (3 * x + 1) + y",
                &["x", "y"],
                &std::collections::BTreeMap::from([("x".into(), 0.25), ("y".into(), 0.75)]),
            )
            .unwrap(),
        ),
        export_passages: 1,
        all_passages: false,
        seed_document_passages: false,
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
        model: Some(
            RankingModel::compile(
                "bm25 + dense + 0.2 * profile",
                &["bm25", "phrase", "sparse", "dense", "profile"],
                &Default::default(),
            )
            .unwrap(),
        ),
        export_passages: 10,
        all_passages: true,
        seed_document_passages: false,
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
    passage_plan.seed_document_passages = true;
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

    let document_candidates = searcher
        .search_with_positions(&profile_query, 10)
        .await
        .unwrap()
        .0;
    let mut document_plan = passage_plan.clone();
    document_plan.seed_document_passages = false;
    let unseeded = searcher
        .score_candidates(&document_candidates, &document_plan, None)
        .await
        .unwrap();
    assert!(unseeded.iter().all(|row| row.features.passages.is_empty()));
    document_plan.seed_document_passages = true;
    let mut invalid = document_plan.clone();
    invalid.backfill = false;
    assert!(
        searcher
            .score_candidates(&document_candidates, &invalid, None)
            .await
            .is_err()
    );
    invalid.backfill = true;
    invalid
        .features
        .retain(|feature| feature.scope == ScoreScope::Document);
    invalid.model = None;
    assert!(
        searcher
            .score_candidates(&document_candidates, &invalid, None)
            .await
            .is_err()
    );
    let mut raw_plan = document_plan.clone();
    raw_plan.model = None;
    let raw_seeded = searcher
        .score_candidates(&document_candidates, &raw_plan, None)
        .await
        .unwrap();
    let raw_body = raw_seeded
        .iter()
        .find(|row| row.result.doc_id == candidates[0].doc_id)
        .unwrap();
    assert_eq!(raw_body.features.scored_passages, 2);
    assert_eq!(
        raw_body
            .features
            .passages
            .iter()
            .find(|row| row.ordinal == 1)
            .unwrap()
            .values,
        matching.values
    );
    document_plan.export_passages = 1;
    let seeded = searcher
        .score_candidates(&document_candidates, &document_plan, None)
        .await
        .unwrap();
    let with_body = seeded
        .iter()
        .find(|row| row.result.doc_id == candidates[0].doc_id)
        .unwrap();
    assert_eq!(with_body.features.scored_passages, 2);
    assert_eq!(with_body.features.passages.len(), 1);
    assert_eq!(with_body.features.passages[0].ordinal, 1);
    assert_eq!(with_body.features.passages[0].values, matching.values);
    assert_eq!(with_body.result.score, matching.score);
    let without_body = seeded
        .iter()
        .find(|row| row.result.doc_id != candidates[0].doc_id)
        .unwrap();
    assert!(without_body.features.passages.is_empty());

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
        document_plan.model =
            Some(RankingModel::compile("dense", &["dense"], &Default::default()).unwrap());
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
        passage_plan.model = Some(
            RankingModel::compile(
                "dense - 2",
                &["bm25", "phrase", "sparse", "dense", "profile"],
                &Default::default(),
            )
            .unwrap(),
        );
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
    composition.model =
        Some(RankingModel::compile("dense", &["dense"], &Default::default()).unwrap());
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
        model: Some(
            RankingModel::compile(
                "profile + 1",
                &["profile"],
                &std::collections::BTreeMap::from([("profile".into(), -1.0)]),
            )
            .unwrap(),
        ),
        export_passages: 1,
        all_passages: false,
        seed_document_passages: false,
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
        seed_document_passages: false,
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
        seed_document_passages: false,
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
        model: Some(
            RankingModel::compile(
                "2 * sparse",
                &["sparse"],
                &std::collections::BTreeMap::from([("sparse".into(), -0.5)]),
            )
            .unwrap(),
        ),
        export_passages: 1,
        all_passages: false,
        seed_document_passages: false,
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
