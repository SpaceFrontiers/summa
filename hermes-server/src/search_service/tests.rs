//! Request budget and RPC boundary regressions.

use super::validation::MAX_FUSION_SUB_QUERIES;
use super::*;
use tonic::Code;

#[test]
fn text_stats_accepts_broker_flattened_terms_but_keeps_aggregate_limit() {
    let shape = QueryShapeLimits::default();
    // The broker flattens text leaves from valid nested/fused searches into
    // one Boolean container. Statistics extraction never scores that Boolean.
    let request = |count| GetTextStatsRequest {
        index_name: "test-index".into(),
        query: Some(Query {
            query: Some(query::Query::Boolean(BooleanQuery {
                should: (0..count)
                    .map(|_| Query {
                        query: Some(query::Query::Term(TermQuery {
                            field: "title".into(),
                            term: "hermes".into(),
                            ..Default::default()
                        })),
                    })
                    .collect(),
                ..Default::default()
            })),
        }),
    };
    validate_text_stats_request(&request(shape.max_boolean_clauses + 1), &shape).unwrap();
    assert_eq!(
        validate_text_stats_request(&request(shape.max_query_nodes), &shape)
            .unwrap_err()
            .code(),
        Code::InvalidArgument,
    );
}

#[tokio::test]
async fn text_stats_validates_shape_before_opening_index() {
    let temp = tempfile::tempdir().unwrap();
    let service = SearchServiceImpl::new(
        Arc::new(IndexRegistry::new(temp.path().into(), Default::default())),
        1,
        SearchLimits {
            shape: QueryShapeLimits {
                max_query_depth: 1,
                ..QueryShapeLimits::default()
            },
            ..SearchLimits::default()
        },
    );
    let error = service
        .get_text_stats(Request::new(GetTextStatsRequest {
            index_name: "absent".into(),
            query: Some(Query {
                query: Some(query::Query::Boolean(BooleanQuery {
                    must: vec![all_query()],
                    ..Default::default()
                })),
            }),
        }))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("nesting depth"));
}

#[tokio::test]
async fn text_stats_shares_search_admission_and_releases_permits_on_error() {
    let temp = tempfile::tempdir().unwrap();
    let service = SearchServiceImpl::new(
        Arc::new(IndexRegistry::new(temp.path().into(), Default::default())),
        1,
        SearchLimits::default(),
    );
    let request = || {
        Request::new(GetTextStatsRequest {
            index_name: "absent".into(),
            query: Some(all_query()),
        })
    };
    let permit = try_acquire_search_permit(&service.search_permits).unwrap();
    assert_eq!(
        service.get_text_stats(request()).await.unwrap_err().code(),
        Code::ResourceExhausted,
    );
    drop(permit);
    assert_eq!(
        service.get_text_stats(request()).await.unwrap_err().code(),
        Code::NotFound,
    );
    assert_eq!(service.search_permits.available_permits(), 1);
}

fn all_query() -> Query {
    Query {
        query: Some(query::Query::All(AllQuery::default())),
    }
}

fn ordinary_request() -> SearchRequest {
    SearchRequest {
        index_name: "test-index".to_string(),
        query: Some(all_query()),
        ..Default::default()
    }
}

fn fusion_request(sub_queries: usize, candidate_limit: u32) -> SearchRequest {
    SearchRequest {
        index_name: "test-index".to_string(),
        candidate_limit,
        query: Some(Query {
            query: Some(query::Query::Fusion(FusionQuery {
                queries: (0..sub_queries)
                    .map(|_| WeightedQuery {
                        query: Some(all_query()),
                        weight: 1.0,

                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            })),
        }),
        ..Default::default()
    }
}

fn limits() -> SearchLimits {
    SearchLimits::default()
}

#[test]
fn search_budget_applies_defaults() {
    let budget = validate_search_budget(&ordinary_request(), &limits()).unwrap();

    assert_eq!(budget.final_limit, limits().default_search_limit);
    assert_eq!(budget.offset, 0);
    assert_eq!(budget.search_limit, limits().default_search_limit);
    assert_eq!(budget.candidate_limit, limits().default_search_limit);
}

#[test]
fn search_budget_honors_configured_limits() {
    let tight = SearchLimits {
        default_search_limit: 5,
        max_search_limit: 20,
        max_search_window: 30,
        max_candidate_limit: 30,
        ..SearchLimits::default()
    };

    let budget = validate_search_budget(&ordinary_request(), &tight).unwrap();
    assert_eq!(budget.final_limit, 5);

    let mut over_limit = ordinary_request();
    over_limit.limit = 21;
    assert_eq!(
        validate_search_budget(&over_limit, &tight)
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );

    let mut over_window = ordinary_request();
    over_window.limit = 20;
    over_window.offset = 11;
    assert_eq!(
        validate_search_budget(&over_window, &tight)
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );

    let mut over_candidates = ordinary_request();
    over_candidates.candidate_limit = 31;
    assert_eq!(
        validate_search_budget(&over_candidates, &tight)
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
}

#[test]
fn query_shape_honors_configured_limits() {
    let tight = SearchLimits {
        shape: QueryShapeLimits {
            max_query_depth: 2,
            max_fields_to_load: 1,
            ..QueryShapeLimits::default()
        },
        ..SearchLimits::default()
    };

    // Depth 3 (two boolean wrappers around the leaf) exceeds the
    // configured depth of 2 but is fine under the defaults.
    let mut nested = all_query();
    for _ in 0..2 {
        nested = Query {
            query: Some(query::Query::Boolean(BooleanQuery {
                must: vec![nested],
                ..Default::default()
            })),
        };
    }
    let mut deep_req = ordinary_request();
    deep_req.query = Some(nested);
    assert_eq!(
        validate_search_budget(&deep_req, &tight)
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    assert!(validate_search_budget(&deep_req, &limits()).is_ok());

    // Two selected fields exceed the configured cap of 1.
    let mut wide_req = ordinary_request();
    wide_req.fields_to_load = vec!["a".to_owned(), "b".to_owned()];
    assert_eq!(
        validate_search_budget(&wide_req, &tight)
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    assert!(validate_search_budget(&wide_req, &limits()).is_ok());
}

#[test]
fn search_budget_rejects_missing_or_oversized_index_name() {
    let mut req = ordinary_request();
    req.index_name.clear();
    assert_eq!(
        validate_search_budget(&req, &limits()).unwrap_err().code(),
        Code::InvalidArgument
    );

    req.index_name = "x".repeat(limits().shape.max_index_name_bytes + 1);
    assert_eq!(
        validate_search_budget(&req, &limits()).unwrap_err().code(),
        Code::InvalidArgument
    );
}

#[test]
fn search_budget_rejects_excessive_final_limit() {
    let mut req = ordinary_request();
    req.limit = (limits().max_search_limit + 1) as u32;

    let err = validate_search_budget(&req, &limits()).unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
}

#[test]
fn search_budget_accounts_for_offset_and_bounds_the_result_window() {
    let mut req = ordinary_request();
    req.limit = 100;
    req.offset = 400;
    let budget = validate_search_budget(&req, &limits()).unwrap();
    assert_eq!(budget.final_limit, 100);
    assert_eq!(budget.offset, 400);
    assert_eq!(budget.search_limit, 500);

    req.limit = limits().default_search_limit as u32;
    req.offset = (limits().max_search_window - limits().default_search_limit) as u32;
    assert!(validate_search_budget(&req, &limits()).is_ok());

    req.offset += 1;
    let err = validate_search_budget(&req, &limits()).unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
}

#[test]
fn search_budget_caps_candidate_depth_at_two_x_the_result_window() {
    let mut ordinary_default = ordinary_request();
    ordinary_default.limit = 300;
    assert_eq!(
        validate_search_budget(&ordinary_default, &limits())
            .unwrap()
            .candidate_limit,
        300
    );

    let mut default_req = ordinary_request();
    default_req.limit = limits().max_search_limit as u32;
    assert_eq!(
        validate_search_budget(&default_req, &limits())
            .unwrap()
            .candidate_limit,
        limits().max_search_limit
    );

    let mut excessive_req = ordinary_request();
    excessive_req.limit = 100;
    excessive_req.candidate_limit = 201;
    let err = validate_search_budget(&excessive_req, &limits()).unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);

    let mut impossible_page = ordinary_request();
    impossible_page.limit = 10;
    impossible_page.offset = 1_000;
    impossible_page.candidate_limit = 100;
    let err = validate_search_budget(&impossible_page, &limits()).unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
}

#[test]
fn explicit_candidate_budget_is_not_expanded_again() {
    let mut request = ordinary_request();
    request.limit = 100;
    request.candidate_limit = 100;
    let budget = validate_search_budget(&request, &limits()).unwrap();
    assert_eq!(budget.search_limit, 100);
    assert_eq!(budget.candidate_limit, 100);
}

#[test]
fn search_admission_rejects_overload_without_queueing() {
    let permits = Arc::new(Semaphore::new(1));
    let permit = try_acquire_search_permit(&permits).unwrap();

    let err = try_acquire_search_permit(&permits).unwrap_err();
    assert_eq!(err.code(), Code::ResourceExhausted);

    drop(permit);
    assert!(try_acquire_search_permit(&permits).is_ok());
}

#[test]
fn fusion_budget_rejects_more_than_two_x_the_result_window() {
    let mut req = fusion_request(2, 201);
    req.limit = 100;

    let err = validate_search_budget(&req, &limits()).unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
}

#[test]
fn fusion_and_reranker_share_one_candidate_pool() {
    let mut req = fusion_request(2, 0);
    req.limit = 100;
    req.reranker = Some(Reranker::default());

    let budget = validate_search_budget(&req, &limits()).unwrap();
    assert_eq!(budget.candidate_limit, 100);
}

#[test]
fn fusion_budget_rejects_too_many_sub_queries_before_conversion() {
    let req = fusion_request(MAX_FUSION_SUB_QUERIES + 1, 50);

    let err = validate_search_budget(&req, &limits()).unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
}

#[test]
fn fusion_budget_rejects_excessive_fetch_and_aggregate_work() {
    let excessive_fetch = fusion_request(2, (limits().max_candidate_limit + 1) as u32);
    assert_eq!(
        validate_search_budget(&excessive_fetch, &limits())
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );

    let mut excessive_aggregate = fusion_request(5, limits().max_candidate_limit as u32);
    excessive_aggregate.limit = limits().max_search_limit as u32;
    excessive_aggregate.offset = (limits().max_search_window - limits().max_search_limit) as u32;
    assert_eq!(
        validate_search_budget(&excessive_aggregate, &limits())
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
}

#[test]
fn fusion_default_candidate_pool_is_checked_and_capped() {
    let mut req = fusion_request(2, 0);
    req.limit = limits().max_search_limit as u32;
    req.reranker = Some(Reranker::default());

    let budget = validate_search_budget(&req, &limits()).unwrap();
    assert_eq!(budget.candidate_limit, limits().max_search_limit);
}

#[test]
fn fusion_budget_rejects_non_finite_or_negative_scoring_parameters() {
    for rrf_k in [f32::NAN, f32::INFINITY, -1.0] {
        let mut req = fusion_request(2, 50);
        let Some(query::Query::Fusion(fusion)) =
            req.query.as_mut().and_then(|query| query.query.as_mut())
        else {
            unreachable!();
        };
        fusion.rrf_k = rrf_k;
        assert_eq!(
            validate_search_budget(&req, &limits()).unwrap_err().code(),
            Code::InvalidArgument
        );
    }

    for weight in [f32::NAN, f32::INFINITY, -1.0] {
        let mut req = fusion_request(2, 50);
        let Some(query::Query::Fusion(fusion)) =
            req.query.as_mut().and_then(|query| query.query.as_mut())
        else {
            unreachable!();
        };
        fusion.queries[0].weight = weight;
        assert_eq!(
            validate_search_budget(&req, &limits()).unwrap_err().code(),
            Code::InvalidArgument
        );
    }
}

#[test]
fn request_shape_rejects_field_and_boolean_amplification() {
    let mut too_many_fields = ordinary_request();
    too_many_fields.fields_to_load = vec![String::new(); limits().shape.max_fields_to_load + 1];
    assert_eq!(
        validate_search_budget(&too_many_fields, &limits())
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );

    let mut too_many_clauses = ordinary_request();
    too_many_clauses.query = Some(Query {
        query: Some(query::Query::Boolean(BooleanQuery {
            must: (0..=limits().shape.max_boolean_clauses)
                .map(|_| all_query())
                .collect(),
            ..Default::default()
        })),
    });
    assert_eq!(
        validate_search_budget(&too_many_clauses, &limits())
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
}

#[test]
fn request_shape_rejects_depth_node_text_and_vector_expansion() {
    let mut nested = all_query();
    for _ in 0..limits().shape.max_query_depth {
        nested = Query {
            query: Some(query::Query::Boost(Box::new(BoostQuery {
                query: Some(Box::new(nested)),
                boost: 1.0,
            }))),
        };
    }
    let mut excessive_depth = ordinary_request();
    excessive_depth.query = Some(nested);
    assert_eq!(
        validate_search_budget(&excessive_depth, &limits())
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );

    // 1 root + 128 Boolean children + 128 leaves exceeds the node budget,
    // while each Boolean and the aggregate clause count remain legal.
    let branches = (0..limits().shape.max_boolean_clauses)
        .map(|_| Query {
            query: Some(query::Query::Boolean(BooleanQuery {
                must: vec![all_query()],
                ..Default::default()
            })),
        })
        .collect();
    let mut excessive_nodes = ordinary_request();
    excessive_nodes.query = Some(Query {
        query: Some(query::Query::Boolean(BooleanQuery {
            should: branches,
            ..Default::default()
        })),
    });
    assert_eq!(
        validate_search_budget(&excessive_nodes, &limits())
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );

    let mut excessive_text = ordinary_request();
    excessive_text.query = Some(Query {
        query: Some(query::Query::Match(MatchQuery {
            field: "body".to_owned(),
            text: "x".repeat(limits().shape.max_query_text_bytes + 1),
            tokenizer_hint: String::new(),
            proximity_weight: 0.0,
            proximity_window: 0,
            heap_factor: 0.0,
            max_terms: 0,
        })),
    });
    assert_eq!(
        validate_search_budget(&excessive_text, &limits())
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );

    let mut excessive_phrase = ordinary_request();
    excessive_phrase.query = Some(Query {
        query: Some(query::Query::Phrase(crate::proto::PhraseQuery {
            field: "body".to_owned(),
            text: "x".repeat(limits().shape.max_query_text_bytes + 1),
            slop: 0,
            tokenizer_hint: String::new(),
        })),
    });
    assert_eq!(
        validate_search_budget(&excessive_phrase, &limits())
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );

    let mut oversized_hint = ordinary_request();
    oversized_hint.query = Some(Query {
        query: Some(query::Query::Match(MatchQuery {
            field: "body".to_owned(),
            text: "x".to_owned(),
            tokenizer_hint: "en,".repeat(limits().shape.max_query_text_bytes),
            proximity_weight: 0.0,
            proximity_window: 0,
            heap_factor: 0.0,
            max_terms: 0,
        })),
    });
    assert_eq!(
        validate_search_budget(&oversized_hint, &limits())
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );

    let mut excessive_vector = ordinary_request();
    excessive_vector.query = Some(Query {
        query: Some(query::Query::DenseVector(DenseVectorQuery {
            field: "embedding".to_owned(),
            vector: vec![0.0; limits().shape.max_dense_query_dims + 1],
            ..Default::default()
        })),
    });
    assert_eq!(
        validate_search_budget(&excessive_vector, &limits())
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
}

#[test]
fn metrics_use_only_canonical_schema_labels() {
    let mut builder = hermes_core::SchemaBuilder::default();
    builder.add_text_field("title", true, true);
    let mut schema = builder.build();

    assert_eq!(UNKNOWN_INDEX_LABEL, "unknown");
    assert_eq!(canonical_metric_index_label(&schema), UNKNOWN_INDEX_LABEL);
    schema.set_index_name("known-index");
    assert_eq!(canonical_metric_index_label(&schema), "known-index");
}

fn named_l1_request() -> SearchRequest {
    SearchRequest {
        index_name: "l1-test".into(),
        limit: 1,
        query: Some(Query {
            query: Some(query::Query::Fusion(FusionQuery {
                queries: vec![
                    WeightedQuery {
                        name: "title".into(),
                        scope: ScoreScope::Document as i32,
                        query: Some(Query {
                            query: Some(query::Query::Match(MatchQuery {
                                field: "title".into(),
                                text: "candidate".into(),
                                ..Default::default()
                            })),
                        }),
                        ..Default::default()
                    },
                    WeightedQuery {
                        name: "body".into(),
                        scope: ScoreScope::Chunk as i32,
                        query: Some(Query {
                            query: Some(query::Query::Match(MatchQuery {
                                field: "body".into(),
                                text: "hemoglobin".into(),
                                ..Default::default()
                            })),
                        }),
                        ..Default::default()
                    },
                ],
                filters: vec![Query {
                    query: Some(query::Query::Phrase(crate::proto::PhraseQuery {
                        field: "body".into(),
                        text: "red blood cells".into(),
                        ..Default::default()
                    })),
                }],
                candidate_depth: 1,
                ..Default::default()
            })),
        }),
        l1: Some(L1Ranking {
            formula: "body".into(),
            ..Default::default()
        }),
        score_export: Some(ScoreExport::default()),
        fields_to_load: vec!["title".into()],
        ..Default::default()
    }
}

#[tokio::test]
async fn l1_invalid_formula_fails_before_opening_index_or_acquiring_capacity() {
    let temp = tempfile::tempdir().unwrap();
    let service = SearchServiceImpl::new(
        Arc::new(IndexRegistry::new(temp.path().into(), Default::default())),
        1,
        limits(),
    );
    let _permit = try_acquire_search_permit(&service.search_permits).unwrap();
    let mut request = named_l1_request();
    for formula in [
        "body + 0.2 * missing_branch".to_owned(),
        String::new(),
        "1/0".to_owned(),
        format!("{}body{}", "(".repeat(33), ")".repeat(33)),
    ] {
        request.l1.as_mut().unwrap().formula = formula;
        let error = service
            .search(Request::new(request.clone()))
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::InvalidArgument);
    }
}

#[test]
fn l1_disabled_backfill_rejects_all_passages_before_admission() {
    let mut request = named_l1_request();
    request.l1.as_mut().unwrap().backfill = Some(false);
    request.score_export.as_mut().unwrap().all_passages = true;
    assert!(
        validate_search_budget(&request, &limits())
            .unwrap_err()
            .message()
            .contains("all_passages diagnostics require backfill")
    );
}

#[test]
fn l1_requires_a_formula_and_rejects_ambiguous_branches_and_rank_fusion_options() {
    let request = named_l1_request();
    validate_search_budget(&request, &limits()).unwrap();
    let mut bad = request.clone();
    bad.l1.as_mut().unwrap().formula.clear();
    assert!(validate_search_budget(&bad, &limits()).is_err());
    let mut bad = request.clone();
    bad.l1.as_mut().unwrap().formula = "sqrt(-1)".into();
    assert!(validate_search_budget(&bad, &limits()).is_err());
    let mut bad = request.clone();
    let Some(query::Query::Fusion(fusion)) = bad.query.as_mut().unwrap().query.as_mut() else {
        unreachable!()
    };
    fusion.queries[0].name = "body".into();
    assert!(validate_search_budget(&bad, &limits()).is_err());
    let mut bad = request.clone();
    let Some(query::Query::Fusion(fusion)) = bad.query.as_mut().unwrap().query.as_mut() else {
        unreachable!()
    };
    fusion.rrf_k = 60.0;
    assert!(validate_search_budget(&bad, &limits()).is_err());
    let mut bad = request;
    let Some(query::Query::Fusion(fusion)) = bad.query.as_mut().unwrap().query.as_mut() else {
        unreachable!()
    };
    for branch in &mut fusion.queries {
        branch.score_only = true;
    }
    assert!(validate_search_budget(&bad, &limits()).is_err());
}

#[test]
fn l1_keeps_existing_fusion_combiners_and_rejects_unknown_values() {
    for combiner in 0..=4 {
        let mut request = named_l1_request();
        let Some(query::Query::Fusion(fusion)) = request.query.as_mut().unwrap().query.as_mut()
        else {
            unreachable!()
        };
        fusion.combiner = combiner;
        validate_search_budget(&request, &limits()).unwrap();
    }
    let mut request = named_l1_request();
    let Some(query::Query::Fusion(fusion)) = request.query.as_mut().unwrap().query.as_mut() else {
        unreachable!()
    };
    fusion.combiner = 900;
    assert!(
        validate_search_budget(&request, &limits())
            .unwrap_err()
            .message()
            .contains("unknown fusion combiner")
    );
}

async fn l1_service_fixture() -> (tempfile::TempDir, Arc<IndexRegistry>, SearchServiceImpl) {
    let temp = tempfile::tempdir().unwrap();
    let registry = Arc::new(IndexRegistry::new(temp.path().into(), Default::default()));
    let mut schema = hermes_core::Schema::builder();
    let title = schema.add_text_field_with_tokenizer("title", true, true, "simple");
    let body = schema.add_text_field_with_tokenizer("body", true, false, "simple");
    let rerank = schema.add_dense_vector_field("rerank", 2, true, false);
    schema.set_chunked(body, true);
    schema.set_positions(body, hermes_core::dsl::PositionMode::TokenPosition);
    registry
        .create_index("l1-test", schema.build())
        .await
        .unwrap();
    let writer = registry.get_writer("l1-test").await.unwrap();
    {
        let mut writer = writer.write().await;
        for (heading, text) in [
            (
                "candidate candidate candidate",
                "red blood cells general medicine",
            ),
            ("candidate", "red blood cells hemoglobin hemoglobin"),
            ("candidate", "hemoglobin hemoglobin hemoglobin hemoglobin"),
        ] {
            let mut document = hermes_core::Document::new();
            document.add_text(title, heading);
            document.add_text(body, text);
            document.add_dense_vector(
                rerank,
                if heading.contains("candidate candidate") {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                },
            );
            writer.add_document(document).unwrap();
        }
        writer.commit().await.unwrap();
    }
    let index = registry.get_or_open_index("l1-test").await.unwrap();
    index.reader().await.unwrap().reload().await.unwrap();
    let service = SearchServiceImpl::new(registry.clone(), 1, limits());
    (temp, registry, service)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l1_rpc_backfills_the_union_before_top_k_and_preserves_required_phrases() {
    let (_temp, registry, service) = l1_service_fixture().await;
    let ranked = service
        .search(Request::new(named_l1_request()))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(ranked.ranking_method, "formula_v1");
    assert_eq!(ranked.hits.len(), 1);
    assert_eq!(
        ranked.hits[0].address.as_ref().unwrap().doc_id,
        1,
        "the model selects from the whole union; the quote-less document remains excluded"
    );
    let raw = ranked.hits[0].candidate_scores.as_ref().unwrap();
    assert!(
        raw.document["title"] > 0.0,
        "title score is backfilled although title top-1 nominated another doc"
    );
    assert_eq!(raw.passages[0].scores["body"], ranked.hits[0].score);
    assert_eq!(raw.passages[0].l1_score, Some(ranked.hits[0].score));
    let mut export = named_l1_request();
    export.l1 = None;
    export.limit = 10;
    export.score_export.as_mut().unwrap().all_passages = true;
    let all = service
        .search(Request::new(export))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(all.ranking_method, "feature_export_v2");
    assert_eq!(all.hits.len(), 2);
    let other = all
        .hits
        .iter()
        .find(|hit| hit.address.as_ref().unwrap().doc_id == 0)
        .unwrap();
    assert_eq!(
        other.candidate_scores.as_ref().unwrap().passages[0].scores["body"],
        0.0,
        "a valid nonmatch is exported as zero"
    );
    let mut too_small = named_l1_request();
    too_small.l1 = None;
    assert_eq!(
        service
            .search(Request::new(too_small))
            .await
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    let info = service
        .get_index_info(Request::new(GetIndexInfoRequest {
            index_name: "l1-test".into(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.candidate_scoring_version, 3);
    assert!(info.unprepared_candidate_fields.is_empty());
    registry.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fusion_fast_only_type_filter_preserves_rrf_formula_and_export_matches() {
    let temp = tempfile::tempdir().unwrap();
    let registry = Arc::new(IndexRegistry::new(temp.path().into(), Default::default()));
    let mut schema = hermes_core::Schema::builder();
    let title = schema.add_text_field_with_tokenizer("title", true, false, "simple");
    let body = schema.add_text_field_with_tokenizer("body", true, false, "simple");
    schema.set_chunked(body, true);
    let kind = schema.add_text_field_with_tokenizer("type", false, false, "raw_ci");
    schema.set_fast(kind, true);
    let issued = schema.add_i64_field("issued_at", false, false);
    schema.set_fast(issued, true);
    registry
        .create_index("l1-test", schema.build())
        .await
        .unwrap();
    let writer = registry.get_writer("l1-test").await.unwrap();
    {
        let mut writer = writer.write().await;
        for value in [
            "journal-article",
            "journal-article",
            "journal-article",
            "book",
        ] {
            let mut document = hermes_core::Document::new();
            document.add_text(kind, value);
            document.add_i64(issued, 2026);
            document.add_text(
                title,
                if value == "book" {
                    "other"
                } else {
                    "candidate"
                },
            );
            document.add_text(
                body,
                if value == "book" {
                    "other"
                } else {
                    "hemoglobin"
                },
            );
            writer.add_document(document).unwrap();
        }
        writer.commit().await.unwrap();
    }
    let index = registry.get_or_open_index("l1-test").await.unwrap();
    index.reader().await.unwrap().reload().await.unwrap();
    let service = SearchServiceImpl::new(registry.clone(), 1, limits());
    let kind = Query {
        query: Some(query::Query::Term(crate::proto::TermQuery {
            field: "type".into(),
            term: "journal-article".into(),
            ..Default::default()
        })),
    };
    let date = Query {
        query: Some(query::Query::Range(crate::proto::RangeQuery {
            field: "issued_at".into(),
            min_i64: Some(2025),
            ..Default::default()
        })),
    };
    let plain = service
        .search(Request::new(SearchRequest {
            index_name: "l1-test".into(),
            query: Some(kind.clone()),
            limit: 3,
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(plain.hits.len(), 3);
    for mode in ["rrf", "formula", "export", "candidates"] {
        let mut baseline = None;
        for filters in [
            vec![],
            vec![date.clone()],
            vec![kind.clone()],
            vec![date.clone(), kind.clone()],
        ] {
            let mut request = named_l1_request();
            request.limit = 3;
            request.tracing = true;
            request.l1.as_mut().unwrap().formula = "title + body".into();
            if mode != "formula" {
                request.l1 = None;
            }
            if mode == "rrf" || mode == "candidates" {
                request.score_export = None;
            }
            let Some(query::Query::Fusion(fusion)) = request.query.as_mut().unwrap().query.as_mut()
            else {
                unreachable!()
            };
            fusion.filters = filters;
            fusion.candidate_depth = if mode == "rrf" { 0 } else { 3 };
            if mode == "candidates" {
                fusion.method = FusionMethod::FusionCandidates as i32;
            }
            let response = service
                .search(Request::new(request))
                .await
                .unwrap()
                .into_inner();
            assert_eq!(response.hits.len(), 3, "{mode}");
            if let Some(expected) = &baseline {
                assert_eq!(
                    &response.hits, expected,
                    "{mode}: an eligibility-only filter preserves scores and ordinals"
                );
            } else {
                baseline = Some(response.hits.clone());
            }
            let trace = response.trace.unwrap();
            assert!(
                trace.shards[0]
                    .queries
                    .iter()
                    .all(|q| q.candidates.len() == 3),
                "{mode}: all branches retain matching candidates"
            );
        }
    }
    registry.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fusion_exclusion_only_filters_remove_self_before_candidate_selection() {
    let temp = tempfile::tempdir().unwrap();
    let registry = Arc::new(IndexRegistry::new(temp.path().into(), Default::default()));
    let mut schema = hermes_core::Schema::builder();
    let title = schema.add_text_field_with_tokenizer("title", true, false, "simple");
    let id = schema.add_text_field_with_tokenizer("id", true, true, "simple");
    registry
        .create_index("l1-test", schema.build())
        .await
        .unwrap();
    let writer = registry.get_writer("l1-test").await.unwrap();
    {
        let mut writer = writer.write().await;
        for (identity, text) in [
            ("self", "candidate candidate candidate"),
            ("allowed", "candidate candidate"),
            ("other", "candidate"),
        ] {
            let mut document = hermes_core::Document::new();
            document.add_text(id, identity);
            document.add_text(title, text);
            writer.add_document(document).unwrap();
        }
        writer.commit().await.unwrap();
    }
    let index = registry.get_or_open_index("l1-test").await.unwrap();
    index.reader().await.unwrap().reload().await.unwrap();
    let service = SearchServiceImpl::new(registry, 1, limits());
    for explicit_all in [false, true] {
        for mode in ["rrf", "linear", "export"] {
            for (excluded, expected) in [
                (vec!["self"], Some(1)),
                (vec!["absent"], Some(0)),
                (vec!["self", "allowed", "other"], None),
            ] {
                let mut request = named_l1_request();
                request.fields_to_load = vec!["id".into()];
                request.l1.as_mut().unwrap().formula = "title".into();
                if mode != "linear" {
                    request.l1 = None;
                }
                if mode == "rrf" {
                    request.score_export = None;
                }
                let Some(query::Query::Fusion(fusion)) =
                    request.query.as_mut().unwrap().query.as_mut()
                else {
                    unreachable!()
                };
                fusion.queries.truncate(1);
                fusion.candidate_depth = if mode == "rrf" { 0 } else { 1 };
                fusion.filters = vec![Query {
                    query: Some(query::Query::Boolean(crate::proto::BooleanQuery {
                        must: if explicit_all {
                            vec![Query {
                                query: Some(query::Query::All(crate::proto::AllQuery {})),
                            }]
                        } else {
                            vec![]
                        },
                        must_not: excluded
                            .iter()
                            .map(|value| Query {
                                query: Some(query::Query::Term(crate::proto::TermQuery {
                                    field: "id".into(),
                                    term: (*value).into(),
                                    ..Default::default()
                                })),
                            })
                            .collect(),
                        ..Default::default()
                    })),
                }];
                let result = service
                    .search(Request::new(request))
                    .await
                    .unwrap()
                    .into_inner();
                assert_eq!(
                    result.hits.len(),
                    usize::from(expected.is_some()),
                    "{mode}, {excluded:?}"
                );
                if let Some(expected) = expected {
                    assert_eq!(
                        result.hits[0].address.as_ref().unwrap().doc_id,
                        expected,
                        "{mode}, {excluded:?}"
                    );
                    if mode != "rrf" {
                        let scores = result.hits[0].candidate_scores.as_ref().unwrap();
                        assert_eq!(
                            scores.document.len(),
                            1,
                            "filters must not become scoring features"
                        );
                        assert!(scores.document["title"] > 0.0);
                    }
                }
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rrf_diagnostics_preserve_linear_scores_and_exclude_backfilled_votes() {
    let (_temp, registry, service) = l1_service_fixture().await;
    let plain = service
        .search(Request::new(named_l1_request()))
        .await
        .unwrap()
        .into_inner();
    assert!(plain.hits[0].rrf_score.is_none());
    assert!(plain.hits[0].rrf_contributions.is_empty());
    let mut request = named_l1_request();
    request.include_rrf_scores = true;
    let explained = service
        .search(Request::new(request))
        .await
        .unwrap()
        .into_inner();
    let hit = &explained.hits[0];
    assert_eq!(hit.address, plain.hits[0].address);
    assert_eq!(hit.score.to_bits(), plain.hits[0].score.to_bits());
    assert_eq!(hit.candidate_scores, plain.hits[0].candidate_scores);
    assert_eq!(hit.rrf_score, Some(1.0 / 61.0));
    assert_eq!(hit.rrf_contributions.len(), 1);
    let vote = &hit.rrf_contributions[0];
    assert_eq!(
        (
            vote.query_index,
            vote.query_name.as_str(),
            vote.ordinal,
            vote.rank
        ),
        (1, "body", Some(0), 1)
    );
    assert_eq!(vote.score, 1.0 / 61.0);
    registry.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tracing_keeps_l1_discarded_candidates_and_query_provenance() {
    let (_temp, registry, service) = l1_service_fixture().await;
    let plain = service
        .search(Request::new(named_l1_request()))
        .await
        .unwrap()
        .into_inner();
    assert!(plain.trace.is_none());
    let mut request = named_l1_request();
    request.tracing = true;
    let traced = service
        .search(Request::new(request.clone()))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(traced.hits, plain.hits);
    let trace = traced.trace.unwrap();
    assert_eq!(trace.shards.len(), 1);
    let shard = &trace.shards[0];
    assert_eq!(shard.index_name, "l1-test");
    assert_eq!(shard.queries.len(), 2);
    let Some(query::Query::Fusion(fusion)) = request.query.as_ref().unwrap().query.as_ref() else {
        unreachable!()
    };
    assert_eq!(shard.filters, fusion.filters);
    for (i, branch) in shard.queries.iter().enumerate() {
        assert_eq!(branch.query, fusion.queries[i].query);
        assert_eq!(branch.candidate_depth, 1);
        assert_eq!(branch.candidates.len(), 1);
        assert_eq!(
            branch.candidates[0].address.as_ref().unwrap().doc_id,
            i as u32
        );
    }
    assert_eq!(shard.selected.len(), 1);
    assert_eq!(shard.selected[0].address, plain.hits[0].address);
    assert!(
        traced.fusion_candidates.is_empty(),
        "trace holds nomination data once"
    );
    request.include_rrf_scores = true;
    let both = service
        .search(Request::new(request))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(both.trace, Some(trace));
    assert!(both.fusion_candidates.is_empty());
    assert_eq!(both.hits[0].rrf_contributions.len(), 1);
    registry.shutdown().await.unwrap();
}

#[test]
fn rrf_request_rejects_partial_scoring_and_nonfusion_before_admission() {
    let mut request = named_l1_request();
    request.include_rrf_scores = true;
    request.time_budget_ms = 1;
    assert!(
        validate_search_budget(&request, &limits())
            .unwrap_err()
            .message()
            .contains("complete")
    );
    request.time_budget_ms = 0;
    request.l1 = None;
    request.score_export = None;
    request.query = Some(Query {
        query: Some(query::Query::All(crate::proto::AllQuery {})),
    });
    assert!(
        validate_search_budget(&request, &limits())
            .unwrap_err()
            .message()
            .contains("fusion")
    );
}

#[test]
fn trace_budget_counts_discarded_candidates_and_encoded_query_metadata() {
    let mut budget = response::SearchResponseBudget::with_maximum(512);
    let oversized = vec![
        hermes_core::query::SearchResult {
            segment_id: 1,
            doc_id: 0,
            score: 1.0,
            positions: Vec::new()
        };
        hermes_core::query::MAX_FUSION_CANDIDATE_SLOTS + 1
    ];
    assert!(
        candidate_scoring::export_candidates(&oversized, &mut budget)
            .unwrap_err()
            .message()
            .contains("candidate")
    );
    let response = SearchResponse {
        trace: Some(SearchTrace {
            shards: vec![ShardSearchTrace {
                index_name: "x".repeat(513),
                ..Default::default()
            }],
        }),
        ..Default::default()
    };
    assert!(
        response::SearchResponseBudget::with_maximum(512)
            .check_response(&response)
            .is_err()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rrf_and_tracing_preserve_reranker_selection_and_pre_rerank_votes() {
    let (_temp, registry, service) = l1_service_fixture().await;
    let mut request = named_l1_request();
    request.l1 = None;
    request.score_export = None;
    request.candidate_limit = 2;
    let Some(query::Query::Fusion(fusion)) = request.query.as_mut().unwrap().query.as_mut() else {
        unreachable!()
    };
    fusion.candidate_depth = 0;
    request.reranker = Some(Reranker {
        field: "rerank".into(),
        vector: vec![1.0, 0.0],
        ..Default::default()
    });
    let plain = service
        .search(Request::new(request.clone()))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(plain.hits[0].address.as_ref().unwrap().doc_id, 0);
    request.include_rrf_scores = true;
    request.tracing = true;
    let traced = service
        .search(Request::new(request))
        .await
        .unwrap()
        .into_inner();
    let hit = &traced.hits[0];
    assert_eq!(hit.address, plain.hits[0].address);
    assert_eq!(hit.score.to_bits(), plain.hits[0].score.to_bits());
    assert_eq!(hit.rrf_score, Some(1.0 / 61.0));
    assert_eq!(hit.rrf_contributions.len(), 1);
    assert_eq!(hit.rrf_contributions[0].query_name, "title");
    let shard = &traced.trace.unwrap().shards[0];
    assert_eq!(shard.queries[0].candidates.len(), 2);
    assert_eq!(
        shard.queries[1].candidates[0]
            .address
            .as_ref()
            .unwrap()
            .doc_id,
        1
    );
    assert_eq!(shard.selected[0].address, hit.address);
    registry.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rrf_in_symbolic_formula_participates_before_passage_selection() {
    let (_temp, registry, service) = l1_service_fixture().await;
    let mut request = named_l1_request();
    request.limit = 2;
    let Some(query::Query::Fusion(fusion)) = request.query.as_mut().unwrap().query.as_mut() else {
        unreachable!()
    };
    fusion.candidate_depth = 2;
    let base = service
        .search(Request::new(request.clone()))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(base.hits[0].address.as_ref().unwrap().doc_id, 1);
    request.include_rrf_scores = true;
    request.tracing = true;
    for weight in [0.0, 3.0, -1000.0] {
        request.l1.as_mut().unwrap().formula = format!("body + {weight} * rrf");
        let result = service
            .search(Request::new(request.clone()))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(result.ranking_method, "formula_v1");
        assert_eq!(
            result.hits[0].address.as_ref().unwrap().doc_id,
            if weight < 0.0 { 0 } else { 1 }
        );
        for hit in &result.hits {
            let original = base
                .hits
                .iter()
                .find(|old| old.address == hit.address)
                .unwrap();
            let expected_rrf: f32 = if hit.address.as_ref().unwrap().doc_id == 0 {
                1.0 / 61.0
            } else {
                1.0 / 61.0 + 1.0 / 62.0
            };
            assert_eq!(hit.rrf_score, Some(expected_rrf));
            assert_eq!(
                hit.score,
                (f64::from(original.score) + weight * f64::from(expected_rrf)) as f32
            );
            let raw = hit.candidate_scores.as_ref().unwrap();
            let before = original.candidate_scores.as_ref().unwrap();
            assert_eq!(raw.document, before.document);
            for row in &raw.passages {
                let old = before
                    .passages
                    .iter()
                    .find(|old| old.ordinal == row.ordinal)
                    .unwrap();
                assert_eq!(row.scores, old.scores);
                let rrf: f32 = hit
                    .rrf_contributions
                    .iter()
                    .filter(|vote| vote.ordinal.is_none() || vote.ordinal == Some(row.ordinal))
                    .map(|vote| vote.score)
                    .sum();
                assert_eq!(
                    row.l1_score.unwrap(),
                    (f64::from(old.l1_score.unwrap()) + weight * f64::from(rrf)) as f32
                );
                assert_eq!(
                    hit.ordinal_scores
                        .iter()
                        .find(|score| score.ordinal == row.ordinal)
                        .unwrap()
                        .score,
                    row.l1_score.unwrap()
                );
            }
        }
    }
    request.l1.as_mut().unwrap().formula = "1 / 0".into();
    assert_eq!(
        service
            .search(Request::new(request))
            .await
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    registry.shutdown().await.unwrap();
}
