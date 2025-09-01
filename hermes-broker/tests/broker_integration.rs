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
