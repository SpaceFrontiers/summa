//! Integration tests: the real hermes-broker binary (static discovery)
//! against in-process mock hermes backends.

mod support;

use std::time::Duration;

use prost::Message;
use support::proto::*;
use support::*;

fn backend_spec(id: &str, addr: &str, shard: &str) -> String {
    format!("id={id},addr={addr},shard={shard}")
}

#[tokio::test(flavor = "multi_thread")]
async fn pass_through_is_identical() {
    let mock = MockBackend::new(&["documents"]);
    mock.state.lock().search_response = rich_search_response();
    let addr = mock.spawn().await;
    let broker = spawn_broker(&[backend_spec("a", &addr, "0")], &[]);
    wait_for_indexes(&broker, &["documents"], Duration::from_secs(10)).await;

    let direct = direct_search_client(&addr)
        .await
        .search(simple_search_request("documents"))
        .await
        .unwrap()
        .into_inner();
    let brokered = broker_search_client(&broker)
        .await
        .search(simple_search_request("documents"))
        .await
        .unwrap()
        .into_inner();

    // Proto-equal, not byte-equal: SearchHit.fields is a protobuf map, and a
    // decode/re-encode may serialize map entries in a different order. That
    // ordering carries no meaning in gRPC; everything observable matches.
    assert_eq!(direct, brokered);
    assert_eq!(brokered.total_hits, 1234);
    assert_eq!(brokered.hits.len(), 2);

    // For map-free payloads the forwarding really is byte-identical.
    let hit_free_of_maps = SearchResponse {
        truncated: brokered.truncated,
        hits: vec![],
        total_hits: brokered.total_hits,
        took_ms: brokered.took_ms,
        timings: brokered.timings,

        ..Default::default()
    };
    assert_eq!(
        hit_free_of_maps.encode_to_vec(),
        SearchResponse {
            hits: vec![],
            ..direct
        }
        .encode_to_vec()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn routes_by_index_to_the_right_backend() {
    let mock_a = MockBackend::new(&["documents"]);
    let mock_b = MockBackend::new(&["social"]);
    let addr_a = mock_a.spawn().await;
    let addr_b = mock_b.spawn().await;
    let broker = spawn_broker(
        &[
            backend_spec("a", &addr_a, "0"),
            backend_spec("b", &addr_b, "1"),
        ],
        &[],
    );
    wait_for_indexes(&broker, &["documents", "social"], Duration::from_secs(10)).await;

    let mut search = broker_search_client(&broker).await;
    search
        .search(simple_search_request("documents"))
        .await
        .unwrap();
    search
        .search(simple_search_request("social"))
        .await
        .unwrap();
    assert_eq!(mock_a.state.lock().searches, vec!["documents"]);
    assert_eq!(mock_b.state.lock().searches, vec!["social"]);

    let mut index = broker_index_client(&broker).await;
    index
        .commit(CommitRequest {
            index_name: "social".to_string(),
        })
        .await
        .unwrap();
    assert!(mock_a.state.lock().commits.is_empty());
    assert_eq!(mock_b.state.lock().commits, vec!["social"]);

    // Unknown index is NOT_FOUND at the broker, with its own message.
    let err = search
        .search(simple_search_request("missing"))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
    assert!(err.message().contains("not present on any healthy backend"));
}

#[tokio::test(flavor = "multi_thread")]
async fn ambiguous_index_reads_are_deterministic_and_writes_refused() {
    let mock_a = MockBackend::new(&["social"]);
    let mock_b = MockBackend::new(&["social"]);
    let addr_a = mock_a.spawn().await;
    let addr_b = mock_b.spawn().await;
    let backends = [
        backend_spec("a", &addr_a, "0"),
        backend_spec("b", &addr_b, "1"),
    ];

    // No placement rule: reads go to the lexicographically-first shard,
    // writes are refused.
    let broker = spawn_broker(&backends, &[]);
    wait_for_indexes(&broker, &["social"], Duration::from_secs(10)).await;
    broker_search_client(&broker)
        .await
        .search(simple_search_request("social"))
        .await
        .unwrap();
    assert_eq!(mock_a.state.lock().searches.len(), 1);
    assert_eq!(mock_b.state.lock().searches.len(), 0);
    let err = broker_index_client(&broker)
        .await
        .commit(CommitRequest {
            index_name: "social".to_string(),
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    assert!(err.message().contains("placement"));
    drop(broker);

    // A placement rule pins both reads and writes to shard 1.
    let broker = spawn_broker(&backends, &["--placement", "social*=1"]);
    wait_for_indexes(&broker, &["social"], Duration::from_secs(10)).await;
    broker_search_client(&broker)
        .await
        .search(simple_search_request("social"))
        .await
        .unwrap();
    broker_index_client(&broker)
        .await
        .commit(CommitRequest {
            index_name: "social".to_string(),
        })
        .await
        .unwrap();
    assert_eq!(mock_b.state.lock().searches.len(), 1);
    assert_eq!(mock_b.state.lock().commits, vec!["social"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn create_index_follows_placement_rules() {
    let mock_a = MockBackend::new(&["documents"]);
    let mock_b = MockBackend::new(&[]);
    let addr_a = mock_a.spawn().await;
    let addr_b = mock_b.spawn().await;
    let broker = spawn_broker(
        &[
            backend_spec("a", &addr_a, "0"),
            backend_spec("b", &addr_b, "1"),
        ],
        &["--placement", "documents*=0", "--placement", "social*=1"],
    );
    wait_for_indexes(&broker, &["documents"], Duration::from_secs(10)).await;

    let mut index = broker_index_client(&broker).await;
    index
        .create_index(CreateIndexRequest {
            index_name: "documents_20260808".to_string(),
            schema: "index documents_20260808 {}".to_string(),
        })
        .await
        .unwrap();
    index
        .create_index(CreateIndexRequest {
            index_name: "social_20260808".to_string(),
            schema: "index social_20260808 {}".to_string(),
        })
        .await
        .unwrap();
    assert_eq!(mock_a.state.lock().creates, vec!["documents_20260808"]);
    assert_eq!(mock_b.state.lock().creates, vec!["social_20260808"]);

    // Unmatched name with default 'single': lands on the shard hosting the
    // fewest indexes (shard 1: one index vs shard 0's two).
    index
        .create_index(CreateIndexRequest {
            index_name: "fresh".to_string(),
            schema: "index fresh {}".to_string(),
        })
        .await
        .unwrap();
    assert_eq!(
        mock_b.state.lock().creates,
        vec!["social_20260808", "fresh"]
    );

    // The broker refreshes its topology right after CreateIndex: the new
    // index becomes routable well before the steady-state poll interval.
    wait_for_indexes(&broker, &["fresh"], Duration::from_secs(5)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_index_documents_regroups_by_index() {
    let mock_a = MockBackend::new(&["documents"]);
    let mock_b = MockBackend::new(&["social"]);
    let addr_a = mock_a.spawn().await;
    let addr_b = mock_b.spawn().await;
    let broker = spawn_broker(
        &[
            backend_spec("a", &addr_a, "0"),
            backend_spec("b", &addr_b, "1"),
        ],
        &[],
    );
    wait_for_indexes(&broker, &["documents", "social"], Duration::from_secs(10)).await;

    let doc = |index: &str| IndexDocumentRequest {
        index_name: index.to_string(),
        fields: vec![FieldEntry {
            name: "id".to_string(),
            value: Some(FieldValue {
                value: Some(field_value::Value::Text("x".to_string())),
            }),
        }],
    };
    // documents ×2 → social ×1 → documents ×1: three flushes, re-routed
    // mid-stream exactly like hermes-server switches indexes mid-stream.
    let stream = tokio_stream::iter(vec![
        doc("documents"),
        doc("documents"),
        doc("social"),
        doc("documents"),
    ]);
    let response = broker_index_client(&broker)
        .await
        .index_documents(stream)
        .await
        .unwrap()
        .into_inner();

    assert_eq!(response.indexed_count, 4);
    assert_eq!(
        mock_a.state.lock().batches,
        vec![("documents".to_string(), 2), ("documents".to_string(), 1)]
    );
    assert_eq!(mock_b.state.lock().batches, vec![("social".to_string(), 1)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn client_deadline_propagates_and_absence_means_untimed() {
    let mock = MockBackend::new(&["documents"]);
    let addr = mock.spawn().await;
    let broker = spawn_broker(&[backend_spec("a", &addr, "0")], &[]);
    wait_for_indexes(&broker, &["documents"], Duration::from_secs(10)).await;

    let mut client = broker_search_client(&broker).await;

    // With a client deadline: the backend sees a grpc-timeout slightly under it.
    let mut request = tonic::Request::new(simple_search_request("documents"));
    request.set_timeout(Duration::from_secs(30));
    client.search(request).await.unwrap();

    // Without one: the backend sees no grpc-timeout at all (24h admin calls
    // and untimed index-builder channels rely on this).
    client
        .search(simple_search_request("documents"))
        .await
        .unwrap();

    let timeouts = mock.state.lock().search_timeouts.clone();
    assert_eq!(timeouts.len(), 2);
    let with_deadline = timeouts[0].as_ref().expect("first call carries a deadline");
    let micros: u64 = with_deadline
        .strip_suffix(['u', 'm', 'S', 'n', 'M', 'H'])
        .unwrap()
        .parse()
        .unwrap();
    let forwarded = match with_deadline.chars().last().unwrap() {
        'u' => Duration::from_micros(micros),
        'm' => Duration::from_millis(micros),
        'S' => Duration::from_secs(micros),
        'n' => Duration::from_nanos(micros),
        other => panic!("unexpected grpc-timeout unit {other}"),
    };
    assert!(
        forwarded <= Duration::from_secs(30) && forwarded > Duration::from_secs(25),
        "forwarded deadline {forwarded:?} should be shaved slightly below 30s"
    );
    assert!(timeouts[1].is_none(), "second call must carry no deadline");
}

#[tokio::test(flavor = "multi_thread")]
async fn dead_backend_is_evicted_and_recovers() {
    let mock = MockBackend::new(&["documents"]);
    let addr = mock.spawn().await;
    let broker = spawn_broker(
        &[backend_spec("a", &addr, "0")],
        &["--backend-unreachable-grace-secs", "2"],
    );
    wait_for_indexes(&broker, &["documents"], Duration::from_secs(10)).await;

    // Take the backend down: polls fail, grace (2s) elapses, the route drops.
    mock.state.lock().unavailable = true;
    let mut search = broker_search_client(&broker).await;
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    loop {
        let err = search
            .search(simple_search_request("documents"))
            .await
            .err();
        match err {
            // Suspect window: the broker still routes (stale grace) and the
            // backend itself answers UNAVAILABLE.
            Some(status) if status.code() == tonic::Code::Unavailable => {}
            // Evicted: the route is gone entirely.
            Some(status) if status.code() == tonic::Code::NotFound => break,
            other => panic!("unexpected search outcome while backend down: {other:?}"),
        }
        assert!(
            std::time::Instant::now() < deadline,
            "backend was never evicted"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Recovery needs two consecutive successful probes.
    mock.state.lock().unavailable = false;
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    loop {
        if search
            .search(simple_search_request("documents"))
            .await
            .is_ok()
        {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "backend never recovered"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn get_index_info_and_get_document_pass_through() {
    let mock = MockBackend::new(&["documents"]);
    let addr = mock.spawn().await;
    let broker = spawn_broker(&[backend_spec("a", &addr, "0")], &[]);
    wait_for_indexes(&broker, &["documents"], Duration::from_secs(10)).await;

    let mut search = broker_search_client(&broker).await;
    let info = search
        .get_index_info(GetIndexInfoRequest {
            index_name: "documents".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.num_docs, 42);
    assert_eq!(info.num_segments, 3);

    search
        .get_document(GetDocumentRequest {
            index_name: "documents".to_string(),
            address: Some(DocAddress {
                segment_id: String::new(),
                doc_id: 1,
            }),
        })
        .await
        .unwrap();
}

/// Three partitions of `documents` on shards 2, 3 and 4 (a multi-shard
/// placement rule): writes hash by primary key, reads fan out and merge.
async fn partitioned_fixture() -> (Vec<MockBackend>, BrokerProc) {
    let mocks: Vec<MockBackend> = (0..3).map(|_| MockBackend::new(&["documents"])).collect();
    let mut specs = Vec::new();
    for (i, mock) in mocks.iter().enumerate() {
        mock.state.lock().schema =
            "index documents {\n  field id: text<raw_ci> [indexed, stored, fast, primary]\n  field title: text [indexed]\n}"
                .to_string();
        let addr = mock.spawn().await;
        specs.push(backend_spec(&format!("m{i}"), &addr, &(i + 2).to_string()));
    }
    let broker = spawn_broker(&specs, &["--placement", "documents*=2,3,4"]);
    wait_for_indexes(&broker, &["documents"], Duration::from_secs(10)).await;
    (mocks, broker)
}

fn text_doc(id: &str) -> NamedDocument {
    NamedDocument {
        fields: vec![FieldEntry {
            name: "id".to_string(),
            value: Some(FieldValue {
                value: Some(field_value::Value::Text(id.to_string())),
            }),
        }],
    }
}

fn scored_hit(id: &str, score: f32) -> SearchHit {
    let mut fields = std::collections::HashMap::new();
    fields.insert(
        "id".to_string(),
        FieldValueList {
            values: vec![FieldValue {
                value: Some(field_value::Value::Text(id.to_string())),
            }],
        },
    );
    SearchHit {
        score,
        fields,
        ..Default::default()
    }
}

fn hit_id(hit: &SearchHit) -> String {
    match &hit.fields["id"].values[0].value {
        Some(field_value::Value::Text(t)) => t.clone(),
        other => panic!("unexpected id value {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_create_and_commit_fan_out_to_every_partition() {
    let (mocks, broker) = partitioned_fixture().await;
    let mut index = broker_index_client(&broker).await;
    index
        .create_index(CreateIndexRequest {
            index_name: "documents_20260904".to_string(),
            schema: "index documents_20260904 {}".to_string(),
        })
        .await
        .unwrap();
    let commit = index
        .commit(CommitRequest {
            index_name: "documents".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(commit.success);
    // Each mock reports 7 docs on commit.
    assert_eq!(commit.num_docs, 21);
    for mock in &mocks {
        let state = mock.state.lock();
        assert_eq!(state.creates, vec!["documents_20260904"]);
        assert_eq!(state.commits, vec!["documents"]);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_batches_are_hash_routed_by_primary_key() {
    let (mocks, broker) = partitioned_fixture().await;
    let ids: Vec<String> = (0..300).map(|i| format!("doc-{i}")).collect();
    let documents: Vec<NamedDocument> = ids.iter().map(|id| text_doc(id)).collect();
    let response = broker_index_client(&broker)
        .await
        .batch_index_documents(BatchIndexDocumentsRequest {
            index_name: "documents".to_string(),
            documents,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.indexed_count, 300);
    assert_eq!(response.error_count, 0);

    let counts: Vec<usize> = mocks
        .iter()
        .map(|mock| {
            let state = mock.state.lock();
            assert_eq!(state.batches.len(), 1, "one batch per partition");
            state.batches[0].1
        })
        .collect();
    assert_eq!(counts.iter().sum::<usize>(), 300);
    assert!(
        counts.iter().all(|c| *c > 50),
        "FNV-1a should spread 300 keys across 3 partitions, got {counts:?}"
    );

    // Same keys again land on the same partitions: routing is a pure
    // function of the key.
    let documents: Vec<NamedDocument> = ids.iter().map(|id| text_doc(id)).collect();
    broker_index_client(&broker)
        .await
        .batch_index_documents(BatchIndexDocumentsRequest {
            index_name: "documents".to_string(),
            documents,
        })
        .await
        .unwrap();
    for (mock, count) in mocks.iter().zip(&counts) {
        let state = mock.state.lock();
        assert_eq!(state.batches[1].1, *count);
    }

    // A document without the primary key is refused at the broker with its
    // request position, and never reaches a backend.
    let response = broker_index_client(&broker)
        .await
        .batch_index_documents(BatchIndexDocumentsRequest {
            index_name: "documents".to_string(),
            documents: vec![
                text_doc("doc-a"),
                NamedDocument { fields: vec![] },
                text_doc("doc-b"),
            ],
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.indexed_count, 2);
    assert_eq!(response.error_count, 1);
    assert_eq!(response.errors.len(), 1);
    assert_eq!(response.errors[0].index, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_stream_routes_each_flush_by_primary_key() {
    let (mocks, broker) = partitioned_fixture().await;
    let stream = tokio_stream::iter((0..100).map(|i| IndexDocumentRequest {
        index_name: "documents".to_string(),
        fields: text_doc(&format!("doc-{i}")).fields,
    }));
    let response = broker_index_client(&broker)
        .await
        .index_documents(stream)
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.indexed_count, 100);
    let total: usize = mocks
        .iter()
        .map(|m| m.state.lock().batches.iter().map(|b| b.1).sum::<usize>())
        .sum();
    assert_eq!(total, 100);
    assert!(mocks.iter().all(|m| !m.state.lock().batches.is_empty()));
}

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_search_merges_by_score_with_shared_stats() {
    let (mocks, broker) = partitioned_fixture().await;
    let canned = [
        vec![scored_hit("p0-a", 9.0), scored_hit("p0-b", 3.0)],
        vec![
            scored_hit("p1-a", 7.0),
            scored_hit("p1-b", 6.0),
            scored_hit("p1-c", 1.0),
        ],
        vec![scored_hit("p2-a", 8.0)],
    ];
    for (mock, hits) in mocks.iter().zip(canned) {
        let mut state = mock.state.lock();
        state.search_response = SearchResponse {
            hits,
            total_hits: 100,
            took_ms: 5,
            timings: None,
            truncated: false,

            ..Default::default()
        };
    }

    let mut request = simple_search_request("documents");
    request.query = Some(Query {
        query: Some(query::Query::Match(MatchQuery {
            field: "title".to_string(),
            text: "quantum".to_string(),
            ..Default::default()
        })),
    });
    request.offset = 1;
    request.limit = 3;
    let response = broker_search_client(&broker)
        .await
        .search(request)
        .await
        .unwrap()
        .into_inner();
    let ids: Vec<String> = response.hits.iter().map(hit_id).collect();
    // Global order: p0-a 9, p2-a 8, p1-a 7, p1-b 6, p1-c 1, p0-b 3 → offset 1, limit 3.
    assert_eq!(ids, vec!["p2-a", "p1-a", "p1-b"]);
    assert_eq!(response.total_hits, 300);

    for mock in &mocks {
        let state = mock.state.lock();
        assert_eq!(state.search_requests.len(), 1);
        let sent = &state.search_requests[0];
        // Every partition is asked for the full window from the top.
        assert_eq!(sent.offset, 0);
        assert_eq!(sent.limit, 4);
        // ...with the merged corpus statistics attached.
        assert_eq!(sent.text_stats.as_ref().map(|s| s.total_docs), Some(126));
    }

    // A query without text terms skips the statistics round trip.
    broker_search_client(&broker)
        .await
        .search(simple_search_request("documents"))
        .await
        .unwrap();
    for mock in &mocks {
        let state = mock.state.lock();
        assert_eq!(state.search_requests.len(), 2);
        assert!(state.search_requests[1].text_stats.is_none());
        assert_eq!(state.text_stats_requests.len(), 1);
    }

    // Hybrid retrieval is a top-level fusion. GetTextStats cannot execute a
    // FusionQuery, so the broker must send only the BM25 leaf for statistics
    // while preserving the fusion in the actual Search request.
    let mut hybrid = simple_search_request("documents");
    hybrid.query = Some(Query {
        query: Some(query::Query::Fusion(FusionQuery {
            queries: vec![
                WeightedQuery {
                    query: Some(Query {
                        query: Some(query::Query::SparseVector(SparseVectorQuery {
                            field: "sparse_vectors".to_string(),
                            ..Default::default()
                        })),
                    }),
                    weight: 1.0,

                    ..Default::default()
                },
                WeightedQuery {
                    query: Some(Query {
                        query: Some(query::Query::Match(MatchQuery {
                            field: "content".to_string(),
                            text: "quantum".to_string(),
                            ..Default::default()
                        })),
                    }),
                    weight: 1.0,

                    ..Default::default()
                },
            ],
            ..Default::default()
        })),
    });
    broker_search_client(&broker)
        .await
        .search(hybrid)
        .await
        .unwrap();
    for mock in &mocks {
        let state = mock.state.lock();
        assert_eq!(state.search_requests.len(), 3);
        assert!(matches!(
            state.search_requests[2]
                .query
                .as_ref()
                .and_then(|query| query.query.as_ref()),
            Some(query::Query::Fusion(_))
        ));
        assert_eq!(state.text_stats_requests.len(), 2);
        assert!(matches!(
            state.text_stats_requests[1]
                .query
                .as_ref()
                .and_then(|query| query.query.as_ref()),
            Some(query::Query::Match(_))
        ));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_reads_aggregate_and_fail_on_any_partition() {
    let (mocks, broker) = partitioned_fixture().await;
    let mut search = broker_search_client(&broker).await;
    let info = search
        .get_index_info(GetIndexInfoRequest {
            index_name: "documents".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.num_docs, 126);
    assert_eq!(info.num_segments, 9);
    assert!(info.schema.contains("primary"));

    // GetDocument: the partition holding the segment answers.
    mocks[0].state.lock().document_missing = true;
    mocks[1].state.lock().document_missing = true;
    search
        .get_document(GetDocumentRequest {
            index_name: "documents".to_string(),
            address: Some(DocAddress {
                segment_id: "seg".to_string(),
                doc_id: 1,
            }),
        })
        .await
        .unwrap();
    mocks[2].state.lock().document_missing = true;
    let missing = search
        .get_document(GetDocumentRequest {
            index_name: "documents".to_string(),
            address: Some(DocAddress {
                segment_id: "seg".to_string(),
                doc_id: 1,
            }),
        })
        .await
        .unwrap_err();
    assert_eq!(missing.code(), tonic::Code::NotFound);

    // A dead partition makes the whole read fail rather than return a
    // partial answer.
    mocks[1].state.lock().unavailable = true;
    let failed = search
        .search(simple_search_request("documents"))
        .await
        .unwrap_err();
    assert_eq!(failed.code(), tonic::Code::Unavailable);
    assert!(
        failed.message().contains("partition"),
        "{}",
        failed.message()
    );
}
