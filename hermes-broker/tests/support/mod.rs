//! Shared integration-test support: an in-process mock hermes-server and a
//! guard that runs the real hermes-broker binary against it.

// Each tests/*.rs crate includes this module and uses a different subset.
#![allow(dead_code)]

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::codec::CompressionEncoding;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

pub mod proto {
    #![allow(clippy::enum_variant_names)]
    tonic::include_proto!("hermes");
}

use proto::index_service_client::IndexServiceClient;
use proto::index_service_server::{IndexService, IndexServiceServer};
use proto::search_service_client::SearchServiceClient;
use proto::search_service_server::{SearchService, SearchServiceServer};
use proto::*;

/// Scripted state of one mock backend, mutable from the test body while the
/// broker (a subprocess) talks to it.
#[derive(Default)]
pub struct MockState {
    /// ListIndexes response.
    pub indexes: Vec<String>,
    /// Canned Search response.
    pub search_response: SearchResponse,
    /// When set, every RPC (including ListIndexes) returns UNAVAILABLE —
    /// simulates a dead/draining backend without tearing the socket down.
    pub unavailable: bool,
    /// Recorded `grpc-timeout` header per Search call (None = absent).
    pub search_timeouts: Vec<Option<String>>,
    /// Recorded (index_name, num_docs) per BatchIndexDocuments call.
    pub batches: Vec<(String, usize)>,
    /// Recorded index names per Commit / CreateIndex / DeleteIndex call.
    pub commits: Vec<String>,
    pub creates: Vec<String>,
    pub deletes: Vec<String>,
    /// Recorded index names per Search call.
    pub searches: Vec<String>,
}

#[derive(Clone)]
pub struct MockBackend {
    pub state: Arc<Mutex<MockState>>,
}

impl MockBackend {
    pub fn new(indexes: &[&str]) -> Self {
        Self {
            state: Arc::new(Mutex::new(MockState {
                indexes: indexes.iter().map(|s| s.to_string()).collect(),
                ..Default::default()
            })),
        }
    }

    fn check_available(&self) -> Result<(), Status> {
        if self.state.lock().unavailable {
            return Err(Status::unavailable("mock backend is down"));
        }
        Ok(())
    }

    /// Serve on an ephemeral local port; returns `host:port`.
    pub async fn spawn(&self) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let incoming = TcpListenerStream::new(listener);
        let search = self.clone();
        let index = self.clone();
        tokio::spawn(async move {
            Server::builder()
                .add_service(
                    SearchServiceServer::new(search)
                        .accept_compressed(CompressionEncoding::Zstd)
                        .accept_compressed(CompressionEncoding::Gzip),
                )
                .add_service(
                    IndexServiceServer::new(index)
                        .accept_compressed(CompressionEncoding::Zstd)
                        .accept_compressed(CompressionEncoding::Gzip),
                )
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        format!("127.0.0.1:{}", addr.port())
    }
}

fn recorded_timeout(metadata: &tonic::metadata::MetadataMap) -> Option<String> {
    metadata
        .get("grpc-timeout")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

#[tonic::async_trait]
impl SearchService for MockBackend {
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        self.check_available()?;
        let timeout = recorded_timeout(request.metadata());
        let req = request.into_inner();
        let mut state = self.state.lock();
        state.search_timeouts.push(timeout);
        state.searches.push(req.index_name);
        Ok(Response::new(state.search_response.clone()))
    }

    async fn get_document(
        &self,
        _request: Request<GetDocumentRequest>,
    ) -> Result<Response<GetDocumentResponse>, Status> {
        self.check_available()?;
        Ok(Response::new(GetDocumentResponse {
            fields: Default::default(),
        }))
    }

    async fn get_index_info(
        &self,
        request: Request<GetIndexInfoRequest>,
    ) -> Result<Response<GetIndexInfoResponse>, Status> {
        self.check_available()?;
        let req = request.into_inner();
        Ok(Response::new(GetIndexInfoResponse {
            index_name: req.index_name,
            num_docs: 42,
            num_segments: 3,
            schema: "index mock {}".to_string(),
            memory_stats: None,
            vector_stats: vec![],
        }))
    }
}

#[tonic::async_trait]
impl IndexService for MockBackend {
    async fn create_index(
        &self,
        request: Request<CreateIndexRequest>,
    ) -> Result<Response<CreateIndexResponse>, Status> {
        self.check_available()?;
        let req = request.into_inner();
        let mut state = self.state.lock();
        state.creates.push(req.index_name.clone());
        state.indexes.push(req.index_name);
        Ok(Response::new(CreateIndexResponse { success: true }))
    }

    async fn index_documents(
        &self,
        request: Request<Streaming<IndexDocumentRequest>>,
    ) -> Result<Response<IndexDocumentsResponse>, Status> {
        self.check_available()?;
        let mut stream = request.into_inner();
        let mut count = 0u32;
        while stream.message().await?.is_some() {
            count += 1;
        }
        Ok(Response::new(IndexDocumentsResponse {
            indexed_count: count,
            errors: vec![],
        }))
    }

    async fn batch_index_documents(
        &self,
        request: Request<BatchIndexDocumentsRequest>,
    ) -> Result<Response<BatchIndexDocumentsResponse>, Status> {
        self.check_available()?;
        let req = request.into_inner();
        let count = req.documents.len();
        self.state.lock().batches.push((req.index_name, count));
        Ok(Response::new(BatchIndexDocumentsResponse {
            indexed_count: count as u32,
            error_count: 0,
            errors: vec![],
        }))
    }

    async fn commit(
        &self,
        request: Request<CommitRequest>,
    ) -> Result<Response<CommitResponse>, Status> {
        self.check_available()?;
        let req = request.into_inner();
        self.state.lock().commits.push(req.index_name);
        Ok(Response::new(CommitResponse {
            success: true,
            num_docs: 7,
        }))
    }

    async fn force_merge(
        &self,
        _request: Request<ForceMergeRequest>,
    ) -> Result<Response<ForceMergeResponse>, Status> {
        self.check_available()?;
        Ok(Response::new(ForceMergeResponse {
            success: true,
            num_segments: 1,
        }))
    }

    async fn reorder(
        &self,
        _request: Request<ReorderRequest>,
    ) -> Result<Response<ReorderResponse>, Status> {
        self.check_available()?;
        Ok(Response::new(ReorderResponse {
            success: true,
            num_segments: 1,
        }))
    }

    async fn delete_index(
        &self,
        request: Request<DeleteIndexRequest>,
    ) -> Result<Response<DeleteIndexResponse>, Status> {
        self.check_available()?;
        let req = request.into_inner();
        let mut state = self.state.lock();
        state.deletes.push(req.index_name.clone());
        state.indexes.retain(|name| name != &req.index_name);
        Ok(Response::new(DeleteIndexResponse { success: true }))
    }

    async fn list_indexes(
        &self,
        _request: Request<ListIndexesRequest>,
    ) -> Result<Response<ListIndexesResponse>, Status> {
        self.check_available()?;
        Ok(Response::new(ListIndexesResponse {
            index_names: self.state.lock().indexes.clone(),
        }))
    }

    async fn retrain_vector_index(
        &self,
        _request: Request<RetrainVectorIndexRequest>,
    ) -> Result<Response<RetrainVectorIndexResponse>, Status> {
        self.check_available()?;
        Ok(Response::new(RetrainVectorIndexResponse { success: true }))
    }
}

/// The real hermes-broker binary running against test backends. Killed on drop.
pub struct BrokerProc {
    child: Child,
    pub addr: String,
}

impl Drop for BrokerProc {
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

/// Spawn the broker with static discovery. `backends` are
/// "id=..,addr=..,shard=.." specs; `extra_args` appends e.g. --placement.
///
/// Ports come from a bind-then-drop probe, so two tests spawning brokers
/// concurrently can race for the same port; the loser exits at bind time.
/// Detect an early exit and retry with a fresh port instead of letting the
/// test time out against a dead process.
pub fn spawn_broker(backends: &[String], extra_args: &[&str]) -> BrokerProc {
    for attempt in 0..5 {
        let port = free_port();
        let addr = format!("127.0.0.1:{port}");
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_hermes-broker"));
        cmd.args([
            "--addr",
            &addr,
            "--metrics-addr",
            "off",
            "--discovery",
            "static",
            "--index-poll-interval-secs",
            "1",
            "--probe-interval-secs",
            "1",
        ]);
        for backend in backends {
            cmd.args(["--backend", backend]);
        }
        cmd.args(extra_args);
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::inherit());
        cmd.env("RUST_LOG", "hermes_broker=info");
        let mut child = cmd.spawn().expect("spawn hermes-broker");
        std::thread::sleep(Duration::from_millis(300));
        match child.try_wait().expect("query broker child") {
            None => return BrokerProc { child, addr },
            Some(status) => {
                eprintln!("broker attempt {attempt} exited early ({status}); retrying");
            }
        }
    }
    panic!("hermes-broker failed to start after 5 attempts");
}

pub async fn broker_search_client(
    broker: &BrokerProc,
) -> SearchServiceClient<tonic::transport::Channel> {
    let endpoint =
        tonic::transport::Endpoint::from_shared(format!("http://{}", broker.addr)).unwrap();
    SearchServiceClient::new(endpoint.connect_lazy())
}

pub async fn broker_index_client(
    broker: &BrokerProc,
) -> IndexServiceClient<tonic::transport::Channel> {
    let endpoint =
        tonic::transport::Endpoint::from_shared(format!("http://{}", broker.addr)).unwrap();
    IndexServiceClient::new(endpoint.connect_lazy())
}

pub async fn direct_search_client(addr: &str) -> SearchServiceClient<tonic::transport::Channel> {
    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{addr}")).unwrap();
    SearchServiceClient::new(endpoint.connect_lazy())
}

/// Wait until the broker's cached ListIndexes contains every expected name.
pub async fn wait_for_indexes(broker: &BrokerProc, expected: &[&str], timeout: Duration) {
    let deadline = std::time::Instant::now() + timeout;
    let mut client = broker_index_client(broker).await;
    loop {
        if let Ok(response) = client.list_indexes(ListIndexesRequest {}).await {
            let names = response.into_inner().index_names;
            if expected.iter().all(|e| names.iter().any(|n| n == e)) {
                return;
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "broker did not learn indexes {expected:?} in {timeout:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// A canned search response exercising every field shape the proto carries,
/// so the pass-through byte-identity test is meaningful.
pub fn rich_search_response() -> SearchResponse {
    let mut fields = std::collections::HashMap::new();
    fields.insert(
        "id".to_string(),
        FieldValueList {
            values: vec![FieldValue {
                value: Some(field_value::Value::Text("doc-1".to_string())),
            }],
        },
    );
    fields.insert(
        "span".to_string(),
        FieldValueList {
            values: vec![
                FieldValue {
                    value: Some(field_value::Value::JsonValue("[0,10]".to_string())),
                },
                FieldValue {
                    value: Some(field_value::Value::U64(17)),
                },
            ],
        },
    );
    fields.insert(
        "binary".to_string(),
        FieldValueList {
            values: vec![FieldValue {
                value: Some(field_value::Value::BinaryDenseVector(vec![0xAB, 0xCD])),
            }],
        },
    );
    SearchResponse {
        hits: vec![
            SearchHit {
                address: Some(DocAddress {
                    segment_id: "0123456789abcdef0123456789abcdef".to_string(),
                    doc_id: 5,
                }),
                score: 1.25,
                fields,
                ordinal_scores: vec![
                    OrdinalScore {
                        ordinal: 0,
                        score: 0.5,
                    },
                    OrdinalScore {
                        ordinal: 3,
                        score: 0.25,
                    },
                ],
            },
            SearchHit {
                address: Some(DocAddress {
                    segment_id: "fedcba9876543210fedcba9876543210".to_string(),
                    doc_id: 0,
                }),
                score: 0.75,
                fields: Default::default(),
                ordinal_scores: vec![],
            },
        ],
        total_hits: 1234,
        took_ms: 7,
        timings: Some(SearchTimings {
            search_us: 100,
            rerank_us: 200,
            load_us: 300,
            total_us: 700,
        }),
    }
}

pub fn simple_search_request(index_name: &str) -> SearchRequest {
    SearchRequest {
        index_name: index_name.to_string(),
        query: Some(Query {
            query: Some(query::Query::All(AllQuery {})),
        }),
        limit: 10,
        offset: 0,
        fields_to_load: vec!["id".to_string()],
        reranker: None,
        candidate_limit: 0,
    }
}
