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
async fn partitioned_fusion_collects_text_stats_from_real_servers() {
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
    let schema = SCHEMA.replace("index e2e", &format!("index {index_name}"));
    let mut index = broker_index_client(&broker).await;
    index
        .create_index(CreateIndexRequest {
            index_name: index_name.to_string(),
            schema,
        })
        .await
        .unwrap();
    wait_for_indexes(&broker, &[index_name], Duration::from_secs(10)).await;
    index
        .batch_index_documents(BatchIndexDocumentsRequest {
            index_name: index_name.to_string(),
            documents: vec![
                doc("doc-1", "quantum field theory"),
                doc("doc-2", "another quantum document"),
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
            fields_to_load: vec!["id".to_string(), "title".to_string()],
            ..Default::default()
        })
        .await
        .expect("partitioned fusion with BM25 terms should collect shared stats")
        .into_inner();
    // Fusion total_hits sums the contributing ranked-list counts; hits are
    // de-duplicated by document address in the fused result.
    assert_eq!(response.total_hits, 3);
    assert_eq!(response.hits.len(), 2);

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
            weights: std::collections::HashMap::from([
                ("topic".into(), 1.0),
                ("specific".into(), 10.0),
            ]),
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
    assert_eq!(ranked.ranking_method, "linear_v1");
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
    let mut export = request;
    export.l1 = None;
    let raw = broker_search_client(&broker)
        .await
        .search(export)
        .await
        .unwrap()
        .into_inner();
    assert_eq!(raw.ranking_method, "feature_export_v1");
    assert_eq!(raw.hits.len(), 2);
}
