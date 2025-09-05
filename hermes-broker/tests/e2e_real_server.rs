//! End-to-end: the real hermes-broker binary in front of two real
//! hermes-server subprocesses, one index family per shard.
//!
//! Ignored by default because it needs the hermes-server binary built first:
//!
//! ```sh
//! cargo build -p hermes-server --bin hermes-server
//! cargo test -p hermes-broker --test e2e_real_server -- --ignored
//! ```
//!
//! Set HERMES_SERVER_BIN to point at a binary outside the local target dir.

mod support;

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use support::proto::*;
use support::{broker_index_client, broker_search_client, spawn_broker, wait_for_indexes};

const SCHEMA: &str = r#"
index e2e {
    field id: text<raw> [primary, indexed, stored]
    field title: text<default> [indexed, stored]
}
"#;

fn hermes_server_bin() -> PathBuf {
    if let Ok(path) = std::env::var("HERMES_SERVER_BIN") {
        return PathBuf::from(path);
    }
    // target/{profile}/deps/e2e_real_server-* -> target/{profile}/hermes-server
    let exe = std::env::current_exe().expect("current test exe");
    let target_dir = exe
        .parent()
        .and_then(|p| p.parent())
        .expect("target profile dir");
    let candidate = target_dir.join("hermes-server");
    assert!(
        candidate.exists(),
        "hermes-server binary not found at {candidate:?}; \
         run `cargo build -p hermes-server --bin hermes-server` first \
         or set HERMES_SERVER_BIN"
    );
    candidate
}

struct ServerProc {
    child: Child,
    addr: String,
    _data_dir: tempfile::TempDir,
}

impl Drop for ServerProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn spawn_server() -> ServerProc {
    let data_dir = tempfile::tempdir().expect("tempdir");
    let addr = format!("127.0.0.1:{}", free_port());
    let child = Command::new(hermes_server_bin())
        .args([
            "--addr",
            &addr,
            "--data-dir",
            data_dir.path().to_str().unwrap(),
            "--metrics-addr",
            "off",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .env("RUST_LOG", "hermes_server=warn")
        .spawn()
        .expect("spawn hermes-server");
    ServerProc {
        child,
        addr,
        _data_dir: data_dir,
    }
}

async fn direct_index_client(
    addr: &str,
) -> index_service_client::IndexServiceClient<tonic::transport::Channel> {
    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{addr}")).unwrap();
    index_service_client::IndexServiceClient::new(endpoint.connect_lazy())
}

async fn wait_server_ready(addr: &str) {
    let mut client = direct_index_client(addr).await;
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        if client.list_indexes(ListIndexesRequest {}).await.is_ok() {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "hermes-server at {addr} never became ready"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

fn doc(id: &str, title: &str) -> NamedDocument {
    let text = |s: &str| FieldValue {
        value: Some(field_value::Value::Text(s.to_string())),
    };
    NamedDocument {
        fields: vec![
            FieldEntry {
                name: "id".to_string(),
                value: Some(text(id)),
            },
            FieldEntry {
                name: "title".to_string(),
                value: Some(text(title)),
            },
        ],
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the hermes-server binary; see module docs"]
async fn broker_routes_real_hermes_servers() {
    let server_a = spawn_server();
    let server_b = spawn_server();
    wait_server_ready(&server_a.addr).await;
    wait_server_ready(&server_b.addr).await;

    let broker = spawn_broker(
        &[
            format!("id=a,addr={},shard=0", server_a.addr),
            format!("id=b,addr={},shard=1", server_b.addr),
        ],
        &["--placement", "docs*=0", "--placement", "social*=1"],
    );
    // Broker is up and has published its first topology snapshot (both
    // servers start with zero indexes, so wait for an empty-but-served list).
    wait_for_indexes(&broker, &[], Duration::from_secs(10)).await;

    // Placement lands each new index on its ruled shard.
    let mut index = broker_index_client(&broker).await;
    for name in ["docs_e2e", "social_e2e"] {
        let schema = SCHEMA.replace("index e2e", &format!("index {name}"));
        let created = index
            .create_index(CreateIndexRequest {
                index_name: name.to_string(),
                schema,
            })
            .await
            .unwrap_or_else(|e| panic!("create_index {name}: {e}"))
            .into_inner();
        assert!(created.success);
    }
    let on_a = direct_index_client(&server_a.addr)
        .await
        .list_indexes(ListIndexesRequest {})
        .await
        .unwrap()
        .into_inner()
        .index_names;
    let on_b = direct_index_client(&server_b.addr)
        .await
        .list_indexes(ListIndexesRequest {})
        .await
        .unwrap()
        .into_inner()
        .index_names;
    assert_eq!(on_a, vec!["docs_e2e"]);
    assert_eq!(on_b, vec!["social_e2e"]);
    wait_for_indexes(
        &broker,
        &["docs_e2e", "social_e2e"],
        Duration::from_secs(10),
    )
    .await;

    // Write + commit through the broker.
    let batch = index
        .batch_index_documents(BatchIndexDocumentsRequest {
            index_name: "docs_e2e".to_string(),
            documents: vec![
                doc("doc-1", "hermes broker end to end"),
                doc("doc-2", "second document"),
            ],
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(batch.indexed_count, 2);
    assert_eq!(batch.error_count, 0);

    let committed = index
        .commit(CommitRequest {
            index_name: "docs_e2e".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(committed.success);
    assert_eq!(committed.num_docs, 2);

    // Duplicate primary keys surface as per-document errors through the
    // broker — index-builder's retry idempotency depends on this.
    let dup = index
        .batch_index_documents(BatchIndexDocumentsRequest {
            index_name: "docs_e2e".to_string(),
            documents: vec![doc("doc-1", "resent")],
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(dup.indexed_count, 0);
    assert_eq!(dup.error_count, 1);
    let message = dup.errors[0].error.to_lowercase();
    assert!(
        message.contains("duplicate") || message.contains("primary") || message.contains("exists"),
        "unexpected duplicate-pk error text: {}",
        dup.errors[0].error
    );

    // Search through the broker and fetch the hit back by address.
    let mut search = broker_search_client(&broker).await;
    let found = search
        .search(SearchRequest {
            time_budget_ms: 0,
            text_stats: None,
            index_name: "docs_e2e".to_string(),
            query: Some(Query {
                query: Some(query::Query::Term(TermQuery {
                    field: "id".to_string(),
                    term: "doc-1".to_string(),
                    tokenizer_hint: String::new(),
                })),
            }),
            limit: 10,
            offset: 0,
            fields_to_load: vec!["id".to_string(), "title".to_string()],
            reranker: None,
            candidate_limit: 0,

            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(found.hits.len(), 1);
    let hit = &found.hits[0];
    let id_values = &hit.fields["id"].values;
    assert!(matches!(
        id_values[0].value.as_ref().unwrap(),
        field_value::Value::Text(t) if t == "doc-1"
    ));

    let fetched = search
        .get_document(GetDocumentRequest {
            index_name: "docs_e2e".to_string(),
            address: hit.address.clone(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(fetched.fields.contains_key("title"));

    // Cross-shard isolation: social_e2e searches hit shard 1 and see nothing
    // of shard 0's corpus.
    let empty = search
        .search(SearchRequest {
            time_budget_ms: 0,
            text_stats: None,
            index_name: "social_e2e".to_string(),
            query: Some(Query {
                query: Some(query::Query::Term(TermQuery {
                    field: "id".to_string(),
                    term: "doc-1".to_string(),
                    tokenizer_hint: String::new(),
                })),
            }),
            limit: 10,
            offset: 0,
            fields_to_load: vec![],
            reranker: None,
            candidate_limit: 0,

            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(empty.hits.len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the hermes-server binary; see module docs"]
async fn broker_ranks_and_exports_long_phrase_features_without_dropping_terms() {
    let server_a = spawn_server();
    let server_b = spawn_server();
    wait_server_ready(&server_a.addr).await;
    wait_server_ready(&server_b.addr).await;
    let broker = spawn_broker(
        &[
            format!("id=a,addr={},shard=0", server_a.addr),
            format!("id=b,addr={},shard=1", server_b.addr),
        ],
        &["--placement", "docs*=0,1"],
    );
    wait_for_indexes(&broker, &[], Duration::from_secs(10)).await;
    let mut index = broker_index_client(&broker).await;
    let invalid = index
        .create_index(CreateIndexRequest {
            index_name: "docs_invalid_phrase_limit".into(),
            schema: "index documents { max_l1_phrase_terms: 0 field title: text }".into(),
        })
        .await
        .expect_err("invalid limits must fail index creation");
    assert_eq!(invalid.code(), tonic::Code::InvalidArgument);
    for configured in [false, true] {
        let index_name = if configured {
            "docs_long_phrase"
        } else {
            "docs_default_phrase"
        };
        let option = if configured {
            "max_l1_phrase_terms: 256"
        } else {
            ""
        };
        index
            .create_index(CreateIndexRequest {
                index_name: index_name.into(),
                schema: format!(
                    "index {index_name} {{
                    {option}
                    field id: text<raw> [primary, indexed, stored]
                    field title: text<simple> [indexed<chunked, token_position>]
                }}"
                ),
            })
            .await
            .unwrap();
        wait_for_indexes(&broker, &[index_name], Duration::from_secs(10)).await;
        let terms: Vec<_> = (0..256).map(|i| format!("term{i}")).collect();
        let mut wrong_word = terms.clone();
        wrong_word[64] = "different".into();
        index
            .batch_index_documents(BatchIndexDocumentsRequest {
                index_name: index_name.into(),
                documents: vec![
                    doc("doc0", &terms.join(" ")),
                    doc("doc1", &terms[..64].join(" ")),
                    doc("doc2", &wrong_word.join(" ")),
                    doc("doc3", &terms[..255].join(" ")),
                ],
            })
            .await
            .unwrap();
        index
            .commit(CommitRequest {
                index_name: index_name.into(),
            })
            .await
            .unwrap();
        let mut search = broker_search_client(&broker).await;
        let info = search
            .get_index_info(GetIndexInfoRequest {
                index_name: index_name.into(),
            })
            .await
            .unwrap()
            .into_inner();
        let reported_schema = hermes_core::parse_single_index(&info.schema)
            .unwrap()
            .to_schema();
        assert_eq!(
            reported_schema.max_l1_phrase_terms(),
            if configured { 256 } else { 64 }
        );
        fn id(hit: &SearchHit) -> &str {
            match hit.fields["id"].values[0].value.as_ref().unwrap() {
                field_value::Value::Text(id) => id,
                _ => panic!("expected a text primary key"),
            }
        }
        for count in [64, 65, 256] {
            let phrase = Query {
                query: Some(query::Query::Phrase(PhraseQuery {
                    field: "title".into(),
                    text: terms[..count].join(" "),
                    ..Default::default()
                })),
            };
            let reference = search
                .search(SearchRequest {
                    index_name: index_name.into(),
                    query: Some(phrase.clone()),
                    limit: 4,
                    fields_to_load: vec!["id".into()],
                    ..Default::default()
                })
                .await
                .unwrap()
                .into_inner();
            let mut matching: Vec<_> = reference.hits.iter().map(id).collect();
            matching.sort_unstable();
            assert_eq!(
                matching,
                match count {
                    64 => vec!["doc0", "doc1", "doc2", "doc3"],
                    65 => vec!["doc0", "doc3"],
                    _ => vec!["doc0"],
                }
            );
            for ranked in [false, true] {
                let response = search
                    .search(SearchRequest {
                        index_name: index_name.into(),
                        limit: 4,
                        fields_to_load: vec!["id".into()],
                        query: Some(Query {
                            query: Some(query::Query::Fusion(FusionQuery {
                                queries: vec![
                                    WeightedQuery {
                                        name: "nomination".into(),
                                        scope: ScoreScope::Chunk as i32,
                                        query: Some(Query {
                                            query: Some(query::Query::Term(TermQuery {
                                                field: "title".into(),
                                                term: "term0".into(),
                                                ..Default::default()
                                            })),
                                        }),
                                        ..Default::default()
                                    },
                                    WeightedQuery {
                                        name: "phrase".into(),
                                        scope: ScoreScope::Chunk as i32,
                                        query: Some(phrase.clone()),
                                        score_only: true,
                                        ..Default::default()
                                    },
                                ],
                                candidate_depth: 4,
                                ..Default::default()
                            })),
                        }),
                        l1: ranked.then(|| L1Ranking {
                            formula: "phrase".into(),
                            ..Default::default()
                        }),
                        score_export: Some(ScoreExport {
                            passages_per_document: 1,
                            all_passages: !ranked,
                        }),
                        ..Default::default()
                    })
                    .await;
                if !configured && count > 64 {
                    let error =
                        response.expect_err("default index cap must reject long L1 phrases");
                    assert_eq!(error.code(), tonic::Code::InvalidArgument, "{error}");
                    assert!(
                        error
                            .message()
                            .contains(&format!("{count} terms; maximum is 64")),
                        "{error}"
                    );
                    continue;
                }
                let response = response
                    .unwrap_or_else(|error| panic!("{count} terms, ranked={ranked}: {error}"))
                    .into_inner();
                assert_eq!(
                    response.ranking_method,
                    if ranked {
                        "formula_v1"
                    } else {
                        "feature_export_v2"
                    }
                );
                assert_eq!(response.hits.len(), 4);
                for hit in &response.hits {
                    let expected = reference
                        .hits
                        .iter()
                        .find(|other| id(other) == id(hit))
                        .map_or(0.0, |hit| hit.score);
                    let passages = &hit.candidate_scores.as_ref().unwrap().passages;
                    assert_eq!(passages.len(), 1);
                    assert_eq!(passages[0].ordinal, 0);
                    assert_eq!(passages[0].scores["phrase"].to_bits(), expected.to_bits());
                    if ranked {
                        assert_eq!(hit.score.to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the hermes-server binary; see module docs"]
async fn partitioned_fusion_uses_global_text_stats_and_exclusion_filters() {
    let server_a = spawn_server();
    let server_b = spawn_server();
    wait_server_ready(&server_a.addr).await;
    wait_server_ready(&server_b.addr).await;

    let broker = spawn_broker(
        &[
            format!("id=a,addr={},shard=0", server_a.addr),
            format!("id=b,addr={},shard=1", server_b.addr),
        ],
        &["--placement", "docs*=0,1"],
    );
    wait_for_indexes(&broker, &[], Duration::from_secs(10)).await;

    let index_name = "docs_fusion_e2e";
    let schema = SCHEMA
        .replace("index e2e", &format!("index {index_name}"))
        .replace('}', "field type: text<raw_ci> [fast]\n}");
    let mut index = broker_index_client(&broker).await;
    index
        .create_index(CreateIndexRequest {
            index_name: index_name.to_string(),
            schema,
        })
        .await
        .unwrap();
    wait_for_indexes(&broker, &[index_name], Duration::from_secs(10)).await;
    let typed_doc = |id: &str, title: &str, kind: &str| {
        let mut document = doc(id, title);
        document.fields.push(FieldEntry {
            name: "type".into(),
            value: Some(FieldValue {
                value: Some(field_value::Value::Text(kind.into())),
            }),
        });
        document
    };
    index
        .batch_index_documents(BatchIndexDocumentsRequest {
            index_name: index_name.to_string(),
            documents: vec![
                typed_doc("doc-1", "quantum field theory", "journal-article"),
                typed_doc("doc-2", "another quantum document", "book"),
            ],
        })
        .await
        .unwrap();
    index
        .commit(CommitRequest {
            index_name: index_name.to_string(),
        })
        .await
        .unwrap();

    let match_query = |text: &str| Query {
        query: Some(query::Query::Match(MatchQuery {
            field: "title".to_string(),
            text: text.to_string(),
            ..Default::default()
        })),
    };
    let response = broker_search_client(&broker)
        .await
        .search(SearchRequest {
            index_name: index_name.to_string(),
            query: Some(Query {
                query: Some(query::Query::Fusion(FusionQuery {
                    queries: vec![
                        WeightedQuery {
                            query: Some(match_query("quantum")),
                            weight: 1.0,

                            ..Default::default()
                        },
                        WeightedQuery {
                            query: Some(match_query("document")),
                            weight: 1.0,

                            ..Default::default()
                        },
                    ],
                    rrf_k: 60.0,
                    ..Default::default()
                })),
            }),
            limit: 10,
            include_rrf_scores: true,
            tracing: true,
            fields_to_load: vec!["id".to_string(), "title".to_string()],
            ..Default::default()
        })
        .await
        .expect("partitioned fusion with BM25 terms should collect shared stats")
        .into_inner();
    // Fusion total_hits sums the contributing ranked-list counts; hits are
    // de-duplicated by document address in the fused result.
    assert_eq!(response.ranking_method, "global_rrf_v1");
    assert_eq!(response.total_hits, 3);
    assert_eq!(response.hits.len(), 2);
    assert!(response.fusion_candidates.is_empty());
    for hit in &response.hits {
        assert_eq!(hit.rrf_score.unwrap().to_bits(), hit.score.to_bits());
        assert!(!hit.rrf_contributions.is_empty());
    }
    let trace = response.trace.unwrap();
    assert_eq!(trace.shards.len(), 2);
    assert_eq!(
        trace
            .shards
            .iter()
            .map(|s| (s.shard_id.as_str(), s.backend_id.as_str()))
            .collect::<Vec<_>>(),
        vec![("0", "a"), ("1", "b")]
    );
    assert!(
        trace
            .shards
            .iter()
            .all(|s| s.index_name == index_name && s.queries.len() == 2)
    );
    assert_eq!(
        trace
            .shards
            .iter()
            .flat_map(|s| &s.queries)
            .map(|q| q.candidates.len())
            .sum::<usize>(),
        3
    );

    let request = SearchRequest {
        index_name: index_name.into(),
        limit: 10,
        query: Some(Query {
            query: Some(query::Query::Fusion(FusionQuery {
                queries: vec![
                    WeightedQuery {
                        name: "topic".into(),
                        scope: ScoreScope::Document as i32,
                        query: Some(match_query("quantum")),
                        ..Default::default()
                    },
                    WeightedQuery {
                        name: "specific".into(),
                        scope: ScoreScope::Document as i32,
                        query: Some(match_query("document")),
                        score_only: true,
                        ..Default::default()
                    },
                ],
                candidate_depth: 10,
                ..Default::default()
            })),
        }),
        l1: Some(L1Ranking {
            formula: "topic + 10 * specific".into(),
            ..Default::default()
        }),
        score_export: Some(ScoreExport::default()),
        fields_to_load: vec!["id".into()],
        ..Default::default()
    };
    let ranked = broker_search_client(&broker)
        .await
        .search(request.clone())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(ranked.ranking_method, "formula_v1");
    assert_eq!(ranked.hits.len(), 2);
    let id = &ranked.hits[0].fields["id"].values[0].value;
    assert_eq!(id, &Some(field_value::Value::Text("doc-2".into())));
    for hit in &ranked.hits {
        let raw = hit.candidate_scores.as_ref().unwrap();
        assert!(
            (hit.score - (raw.document["topic"] + 10.0 * raw.document["specific"])).abs() < 1e-5
        );
        assert!(raw.passages.is_empty());
    }
    let a = &ranked.hits[0].candidate_scores.as_ref().unwrap().document;
    let b = &ranked.hits[1].candidate_scores.as_ref().unwrap().document;
    assert!(
        (a["topic"] - b["topic"]).abs() < 1e-6,
        "the same term and length use shared cross-shard statistics"
    );
    assert_eq!(
        b["specific"], 0.0,
        "score-only backfill distinguishes a valid nonmatch"
    );
    for mode in ["rrf", "formula", "export"] {
        let mut filtered = request.clone();
        if mode != "formula" {
            filtered.l1 = None;
        }
        if mode == "rrf" {
            filtered.score_export = None;
        }
        let Some(query::Query::Fusion(fusion)) = filtered.query.as_mut().unwrap().query.as_mut()
        else {
            unreachable!()
        };
        if mode == "rrf" {
            fusion.queries.truncate(1);
            fusion.candidate_depth = 0;
        }
        fusion.filters = vec![Query {
            query: Some(query::Query::Term(TermQuery {
                field: "type".into(),
                term: "journal-article".into(),
                ..Default::default()
            })),
        }];
        let response = broker_search_client(&broker)
            .await
            .search(filtered)
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.hits.len(), 1, "{mode}: fast-only type filter");
        assert_eq!(
            response.hits[0].fields["id"].values[0].value,
            Some(field_value::Value::Text("doc-1".into())),
            "{mode}: fast-only type filter"
        );
    }
    let mut traced_request = request.clone();
    traced_request.tracing = true;
    traced_request.include_rrf_scores = true;
    let traced = broker_search_client(&broker)
        .await
        .search(traced_request)
        .await
        .unwrap()
        .into_inner();
    for (plain, diagnostic) in ranked.hits.iter().zip(&traced.hits) {
        let mut stripped = diagnostic.clone();
        stripped.rrf_score = None;
        stripped.rrf_contributions.clear();
        assert_eq!(*plain, stripped);
        assert_eq!(diagnostic.rrf_contributions.len(), 1);
        assert_eq!(diagnostic.rrf_contributions[0].query_name, "topic");
        assert_eq!(diagnostic.rrf_contributions[0].ordinal, None);
    }
    let trace = traced.trace.unwrap();
    assert_eq!(trace.shards.len(), 2);
    for shard in trace.shards {
        assert_eq!(shard.queries.len(), 2);
        assert_eq!(shard.queries[0].candidates.len(), 1);
        assert!(shard.queries[1].score_only);
        assert!(shard.queries[1].candidates.is_empty());
    }
    let mut rrf_formula = request.clone();
    rrf_formula.include_rrf_scores = true;
    rrf_formula.tracing = true;
    rrf_formula.l1.as_mut().unwrap().formula = "topic + 10 * specific - 1000 * rrf".into();
    let reranked = broker_search_client(&broker)
        .await
        .search(rrf_formula)
        .await
        .unwrap()
        .into_inner();
    assert_eq!(reranked.ranking_method, "formula_v1");
    for hit in &reranked.hits {
        let original = ranked
            .hits
            .iter()
            .find(|original| original.address == hit.address)
            .unwrap();
        let rrf = hit.rrf_score.unwrap();
        let raw = &original.candidate_scores.as_ref().unwrap().document;
        assert_eq!(
            hit.score,
            (f64::from(raw["topic"]) + 10.0 * f64::from(raw["specific"]) - 1000.0 * f64::from(rrf))
                as f32
        );
        assert_eq!(hit.candidate_scores, original.candidate_scores);
        assert_eq!(hit.rrf_contributions[0].query_name, "topic");
    }
    assert_eq!(reranked.trace.unwrap().shards.len(), 2);
    // Nonlinear formulas use the same raw features at shards and broker. In
    // particular log/division of global RRF must never be evaluated with zero
    // substituted while shards prepare their complete feature export.
    for formula in [
        "sqrt(abs(topic)) + log1p(specific)",
        "ln(rrf) + specific / rrf",
    ] {
        let mut symbolic = request.clone();
        symbolic.l1.as_mut().unwrap().formula = formula.into();
        symbolic.include_rrf_scores = true;
        let response = broker_search_client(&broker)
            .await
            .search(symbolic)
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.ranking_method, "formula_v1");
        for hit in &response.hits {
            let raw = &hit.candidate_scores.as_ref().unwrap().document;
            let expected = if formula.starts_with("sqrt") {
                f64::from(raw["topic"]).abs().sqrt() + f64::from(raw["specific"]).ln_1p()
            } else {
                let rrf = f64::from(hit.rrf_score.unwrap());
                rrf.ln() + f64::from(raw["specific"]) / rrf
            };
            assert_eq!(hit.score, expected as f32, "{formula}");
        }
    }
    let mut invalid_formula = request.clone();
    invalid_formula.l1.as_mut().unwrap().formula = "ln(-rrf)".into();
    assert_eq!(
        broker_search_client(&broker)
            .await
            .search(invalid_formula)
            .await
            .unwrap_err()
            .code(),
        tonic::Code::InvalidArgument
    );
    let mut no_backfill = request.clone();
    let model = no_backfill.l1.as_mut().unwrap();
    model.backfill = Some(false);
    model.missing_values.insert("specific".into(), -0.75);
    let missing = broker_search_client(&broker)
        .await
        .search(no_backfill)
        .await
        .unwrap()
        .into_inner();
    assert_eq!(missing.ranking_method, "formula_v1");
    assert_eq!(missing.hits.len(), ranked.hits.len());
    for hit in &missing.hits {
        let raw = hit.candidate_scores.as_ref().unwrap();
        assert!(
            !raw.document.contains_key("specific"),
            "imputation must not hide raw missing values"
        );
        assert_eq!(hit.score, (f64::from(raw.document["topic"]) - 7.5) as f32);
        let original = ranked
            .hits
            .iter()
            .find(|h| h.address == hit.address)
            .unwrap();
        assert_eq!(
            raw.document["topic"],
            original.candidate_scores.as_ref().unwrap().document["topic"],
            "organic scores are preserved across backfill settings"
        );
    }
    for (excluded, expected) in [("doc-1", "doc-2"), ("doc-2", "doc-1"), ("absent", "doc-2")] {
        let mut filtered = request.clone();
        filtered.limit = 1;
        let Some(query::Query::Fusion(fusion)) = filtered.query.as_mut().unwrap().query.as_mut()
        else {
            unreachable!()
        };
        fusion.candidate_depth = 1;
        fusion.filters = vec![Query {
            query: Some(query::Query::Boolean(BooleanQuery {
                must_not: vec![Query {
                    query: Some(query::Query::Term(TermQuery {
                        field: "id".into(),
                        term: excluded.into(),
                        ..Default::default()
                    })),
                }],
                ..Default::default()
            })),
        }];
        let result = broker_search_client(&broker)
            .await
            .search(filtered)
            .await
            .unwrap()
            .into_inner();
        assert_eq!(result.hits.len(), 1);
        assert_eq!(
            result.hits[0].fields["id"].values[0].value,
            Some(field_value::Value::Text(expected.into()))
        );
        assert_eq!(
            result.hits[0]
                .candidate_scores
                .as_ref()
                .unwrap()
                .document
                .len(),
            2
        );
    }
    let mut export = request;
    export.l1 = None;
    let raw = broker_search_client(&broker)
        .await
        .search(export)
        .await
        .unwrap()
        .into_inner();
    assert_eq!(raw.ranking_method, "feature_export_v2");
    assert_eq!(raw.hits.len(), 2);
}
