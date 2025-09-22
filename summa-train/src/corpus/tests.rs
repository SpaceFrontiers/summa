use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::*;

struct MockSearchTransport {
    responses: Mutex<VecDeque<Value>>,
    requests: Arc<Mutex<Vec<Value>>>,
}

impl MockSearchTransport {
    fn new(responses: Vec<Value>) -> (Self, Arc<Mutex<Vec<Value>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                responses: Mutex::new(responses.into()),
                requests: Arc::clone(&requests),
            },
            requests,
        )
    }
}

impl SearchApiTransport for MockSearchTransport {
    fn post_json(&self, _endpoint: &str, body: &Value) -> Result<Value> {
        self.requests.lock().unwrap().push(body.clone());
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("unexpected search request"))
    }
}

fn search_api_config() -> SearchApiConfig {
    SearchApiConfig {
        endpoint: "https://search.invalid/v2/search".to_owned(),
        request_template: json!({
            "payload": {
                "words": "",
                "start": 0,
                "count": 1,
                "kind": "sparse_dense_fusion",
                "types": [],
                "fusion": {
                    "sparse": {"field": "template_sparse", "operator": "bm25"},
                    "dense": {"field": "template_dense", "operator": "cosine"}
                },
                "reranker": { "enabled": true },
                "cross_rerank": true,
                "hydrate": true,
                "snapshot_revision": ""
            }
        }),
        request_mapping: SearchApiRequestMapping {
            query_pointer: "/payload/words".to_owned(),
            offset_pointer: "/payload/start".to_owned(),
            limit_pointer: "/payload/count".to_owned(),
            parameter_pointers: BTreeMap::from([("types".to_owned(), "/payload/types".to_owned())]),
            disabled_reranker_pointers: vec![
                "/payload/reranker/enabled".to_owned(),
                "/payload/cross_rerank".to_owned(),
            ],
            return_documents_pointer: Some("/payload/hydrate".to_owned()),
        },
        response_mapping: SearchApiResponseMapping {
            hits_pointer: "/results/items".to_owned(),
            total_hits_pointer: Some("/results/count".to_owned()),
            record_key_pointer: "/primary_key".to_owned(),
            score_pointer: Some("/rank".to_owned()),
            uris_pointer: Some("/opaque_locations".to_owned()),
            inline_text_pointer: None,
            metadata_pointers: BTreeMap::from([(
                "level".to_owned(),
                "/attributes/level".to_owned(),
            )]),
        },
        fusion: SearchApiFusionContract {
            marker_pointer: "/payload/kind".to_owned(),
            marker_value: json!("sparse_dense_fusion"),
            sparse: SearchApiFusionBranch {
                clause_pointer: "/payload/fusion/sparse".to_owned(),
                vector_field_pointer: "/payload/fusion/sparse/field".to_owned(),
                vector_field: "schema_specific_sparse".to_owned(),
            },
            dense: SearchApiFusionBranch {
                clause_pointer: "/payload/fusion/dense".to_owned(),
                vector_field_pointer: "/payload/fusion/dense/field".to_owned(),
                vector_field: "schema_specific_dense".to_owned(),
            },
        },
        snapshot: SearchApiSnapshotContract {
            provider: "search-test".to_owned(),
            revision: "index-42".to_owned(),
            request_revision_pointer: "/payload/snapshot_revision".to_owned(),
            response_provider_pointer: "/snapshot/provider".to_owned(),
            response_revision_pointer: "/snapshot/revision".to_owned(),
        },
        page_size: 50,
        timeout_seconds: 5,
        max_retries: 0,
        retry_initial_ms: 1,
        retry_max_ms: 1,
        max_response_bytes: 1024 * 1024,
        auth: None,
    }
}

fn discover_search_response(config: SearchApiConfig, response: Value) -> Result<DiscoveryPage> {
    let (transport, _) = MockSearchTransport::new(vec![response]);
    SearchApiClient::with_transport(config, transport)?.discover(
        &DiscoveryQuery {
            name: "probe".to_owned(),
            text: "probe".to_owned(),
            limit: 1,
            parameters: BTreeMap::new(),
        },
        0,
        1,
    )
}

fn valid_search_response() -> Value {
    json!({
        "snapshot": {"provider": "search-test", "revision": "index-42"},
        "results": {
            "count": 1,
            "items": [{
                "primary_key": "record-1",
                "rank": 0.75
            }]
        }
    })
}

#[test]
fn search_api_uses_configured_fields_and_forces_rerankers_off() {
    let (transport, requests) = MockSearchTransport::new(vec![json!({
        "snapshot": {"provider": "search-test", "revision": "index-42"},
        "results": {
            "count": 1,
            "items": [{
                "primary_key": 17,
                "rank": 0.75,
                "opaque_locations": ["urn:book:17", "custom+store://bucket/key"],
                "attributes": { "level": "school" }
            }]
        }
    })]);
    let client = SearchApiClient::with_transport(search_api_config(), transport).unwrap();
    let query = DiscoveryQuery {
        name: "school".to_owned(),
        text: "basic geometry".to_owned(),
        limit: 10,
        parameters: BTreeMap::from([("types".to_owned(), json!(["manual"]))]),
    };
    let page = client.discover(&query, 7, 10).unwrap();
    assert_eq!(page.total_hits, Some(1));
    assert_eq!(page.hits[0].record_key, "17");
    assert_eq!(
        page.hits[0].uris,
        ["urn:book:17", "custom+store://bucket/key"]
    );
    assert_eq!(page.hits[0].metadata["level"], "school");

    let requests = requests.lock().unwrap();
    let request = &requests[0];
    assert_eq!(
        request.pointer("/payload/words"),
        Some(&json!("basic geometry"))
    );
    assert_eq!(request.pointer("/payload/start"), Some(&json!(7)));
    assert_eq!(request.pointer("/payload/count"), Some(&json!(10)));
    assert_eq!(request.pointer("/payload/types"), Some(&json!(["manual"])));
    assert_eq!(
        request.pointer("/payload/reranker/enabled"),
        Some(&json!(false))
    );
    assert_eq!(
        request.pointer("/payload/cross_rerank"),
        Some(&json!(false))
    );
    assert_eq!(request.pointer("/payload/hydrate"), Some(&json!(false)));
    assert_eq!(
        request.pointer("/payload/snapshot_revision"),
        Some(&json!("index-42"))
    );
    assert_eq!(
        request.pointer("/payload/fusion/sparse/field"),
        Some(&json!("schema_specific_sparse"))
    );
    assert_eq!(
        request.pointer("/payload/fusion/dense/field"),
        Some(&json!("schema_specific_dense"))
    );
    assert_eq!(
        request.pointer("/payload/fusion/sparse/operator"),
        Some(&json!("bm25"))
    );
    assert_eq!(
        request.pointer("/payload/fusion/dense/operator"),
        Some(&json!("cosine"))
    );
}

#[test]
fn search_api_rejects_a_vector_field_outside_its_declared_fusion_clause() {
    let mut config = search_api_config();
    config.fusion.sparse.vector_field_pointer = "/payload/types".to_owned();
    let (transport, _) = MockSearchTransport::new(vec![]);
    let error = SearchApiClient::with_transport(config, transport)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("inside its fusion clause"), "{error}");
}

#[test]
fn search_api_rejects_overlapping_request_write_targets() {
    let cases = [
        {
            let mut config = search_api_config();
            config.request_mapping.disabled_reranker_pointers[0] =
                config.fusion.marker_pointer.clone();
            ("reranker and fusion marker", config)
        },
        {
            let mut config = search_api_config();
            config.request_mapping.offset_pointer = config.request_mapping.query_pointer.clone();
            ("query and pagination", config)
        },
        {
            let mut config = search_api_config();
            config.request_mapping.parameter_pointers.insert(
                "types".to_owned(),
                config.request_mapping.limit_pointer.clone(),
            );
            ("parameter and pagination", config)
        },
        {
            let mut config = search_api_config();
            config.snapshot.request_revision_pointer = config.fusion.marker_pointer.clone();
            ("snapshot and fusion marker", config)
        },
        {
            let mut config = search_api_config();
            config.request_mapping.return_documents_pointer =
                Some(config.fusion.sparse.clause_pointer.clone());
            ("ancestor and descendant", config)
        },
    ];

    for (name, config) in cases {
        let (transport, _) = MockSearchTransport::new(vec![]);
        let error = SearchApiClient::with_transport(config, transport)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("write targets"), "{name}: {error}");
        assert!(error.contains("overlap"), "{name}: {error}");
    }
}

#[test]
fn search_api_rejects_missing_or_drifted_remote_snapshot_proof() {
    for response in [
        json!({"results": {"count": 0, "items": []}}),
        json!({
            "snapshot": {"provider": "search-test", "revision": "index-43"},
            "results": {"count": 0, "items": []}
        }),
    ] {
        let (transport, _) = MockSearchTransport::new(vec![response]);
        let client = SearchApiClient::with_transport(search_api_config(), transport).unwrap();
        let error = client
            .discover(
                &DiscoveryQuery {
                    name: "probe".to_owned(),
                    text: "probe".to_owned(),
                    limit: 1,
                    parameters: BTreeMap::new(),
                },
                0,
                1,
            )
            .unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("snapshot"), "{error}");
    }
}

#[test]
fn search_api_rejects_missing_or_malformed_configured_numeric_response_fields() {
    let cases = [
        {
            let mut response = valid_search_response();
            response["results"]["items"][0]
                .as_object_mut()
                .unwrap()
                .remove("rank");
            ("missing score", response, "configured score")
        },
        {
            let mut response = valid_search_response();
            response["results"]["items"][0]["rank"] = json!("high");
            ("wrong score type", response, "finite number")
        },
        {
            let mut response = valid_search_response();
            response["results"].as_object_mut().unwrap().remove("count");
            ("missing total", response, "configured total_hits")
        },
        {
            let mut response = valid_search_response();
            response["results"]["count"] = json!("one");
            (
                "wrong total type",
                response,
                "nonnegative integer fitting u64",
            )
        },
        {
            let mut response = valid_search_response();
            response["results"]["count"] = json!(-1);
            (
                "negative total",
                response,
                "nonnegative integer fitting u64",
            )
        },
    ];

    for (name, response, expected) in cases {
        let error = format!(
            "{:#}",
            discover_search_response(search_api_config(), response).unwrap_err()
        );
        assert!(error.contains(expected), "{name}: {error}");
    }
}

#[test]
fn absent_numeric_response_mappings_keep_explicit_defaults() {
    let mut config = search_api_config();
    config.response_mapping.score_pointer = None;
    config.response_mapping.total_hits_pointer = None;
    let response = json!({
        "snapshot": {"provider": "search-test", "revision": "index-42"},
        "results": {"items": [{"primary_key": "record-1"}]}
    });

    let page = discover_search_response(config, response).unwrap();
    assert_eq!(page.total_hits, None);
    assert_eq!(page.hits[0].score, 0.0);
}

#[test]
fn search_api_enforces_configured_page_and_resource_limits() {
    let mut response = valid_search_response();
    response["results"]["count"] = json!(2);
    response["results"]["items"] = json!([
        {"primary_key": "record-1", "rank": 0.75},
        {"primary_key": "record-2", "rank": 0.50}
    ]);
    let error = discover_search_response(search_api_config(), response)
        .unwrap_err()
        .to_string();
    assert!(error.contains("request limit of 1"), "{error}");

    for (name, mutate) in [
        (
            "page size",
            (|config: &mut SearchApiConfig| config.page_size = usize::MAX)
                as fn(&mut SearchApiConfig),
        ),
        (
            "response bytes",
            (|config: &mut SearchApiConfig| config.max_response_bytes = usize::MAX)
                as fn(&mut SearchApiConfig),
        ),
        (
            "timeout",
            (|config: &mut SearchApiConfig| config.timeout_seconds = u64::MAX)
                as fn(&mut SearchApiConfig),
        ),
        (
            "retry delay",
            (|config: &mut SearchApiConfig| config.retry_max_ms = u64::MAX)
                as fn(&mut SearchApiConfig),
        ),
    ] {
        let mut config = search_api_config();
        mutate(&mut config);
        assert!(config.validate().is_err(), "accepted hostile {name}");
    }
}

#[test]
fn search_api_rejects_a_malformed_configured_inline_text() {
    let mut config = search_api_config();
    config.response_mapping.inline_text_pointer = Some("/body".to_owned());
    let mut response = valid_search_response();
    response["results"]["items"][0]["body"] = json!(["not", "text"]);

    let error = format!(
        "{:#}",
        discover_search_response(config, response).unwrap_err()
    );
    assert!(error.contains("inline text"), "{error}");
    assert!(error.contains("must be a string"), "{error}");
}

struct MockPostgresExecutor {
    rows: Vec<PostgresRecordRow>,
}

impl PostgresExecutor for MockPostgresExecutor {
    fn snapshot_revision(&self) -> Result<String> {
        Ok("10:20:".to_owned())
    }

    fn fetch(
        &self,
        statement: &str,
        columns: &PostgresColumnMapping,
        record_keys: &[String],
    ) -> Result<Vec<PostgresRecordRow>> {
        ensure!(statement == "SELECT configured");
        ensure!(columns.text == "unusual_text_alias");
        Ok(self
            .rows
            .iter()
            .filter(|row| record_keys.contains(&row.record_key))
            .cloned()
            .collect())
    }
}

fn postgres_config() -> PostgresRecordMaterializerConfig {
    PostgresRecordMaterializerConfig {
        connection_environment: "TEST_POSTGRES_DSN".to_owned(),
        statement: "SELECT configured".to_owned(),
        columns: PostgresColumnMapping {
            record_key: "unusual_key_alias".to_owned(),
            text: "unusual_text_alias".to_owned(),
            uris: Some("unusual_uri_alias".to_owned()),
            metadata: Some("unusual_metadata_alias".to_owned()),
        },
        snapshot_statement: "SELECT snapshot".to_owned(),
        transport_security: PostgresTransportSecurity::VerifiedTls {
            server_names: vec!["database.example.invalid".to_owned()],
            trust: PostgresTlsTrust::System,
        },
        require_every_record: true,
    }
}

#[test]
fn postgres_materializer_uses_canonical_rows_and_opaque_uris() {
    let executor = MockPostgresExecutor {
        rows: vec![PostgresRecordRow {
            record_key: "a".to_owned(),
            text: "canonical database text".to_owned(),
            uris: vec!["not-a-standard-prefix!value".to_owned()],
            metadata: BTreeMap::from([("kind".to_owned(), json!("book"))]),
        }],
    };
    let materializer =
        PostgresRecordMaterializer::with_executor(postgres_config(), executor).unwrap();
    let records = materializer
        .materialize(&[DiscoveryHit {
            record_key: "a".to_owned(),
            score: 1.0,
            uris: vec!["search:uri".to_owned()],
            metadata: BTreeMap::from([("query".to_owned(), json!("grammar"))]),
            inline_text: Some("search snippets are not canonical".to_owned()),
        }])
        .unwrap();
    assert_eq!(records[0].text, "canonical database text");
    assert_eq!(records[0].uris, ["not-a-standard-prefix!value"]);
    assert_eq!(records[0].metadata["kind"], "book");
    assert_eq!(records[0].metadata["query"], "grammar");
    assert_eq!(materializer.snapshot().unwrap().revision, "10:20:");
    let serialized = materializer.configuration().unwrap().to_string();
    assert!(serialized.contains("TEST_POSTGRES_DSN"));
    assert!(serialized.contains("verified_tls"));
    assert!(serialized.contains("database.example.invalid"));
    assert!(!serialized.contains("password"));
}

#[test]
fn postgres_materializer_requires_a_stable_snapshot_statement() {
    let mut config = serde_json::to_value(postgres_config()).unwrap();
    config.as_object_mut().unwrap().remove("snapshot_statement");
    let error = serde_json::from_value::<PostgresRecordMaterializerConfig>(config)
        .unwrap_err()
        .to_string();
    assert!(error.contains("snapshot_statement"), "{error}");
}

#[test]
fn postgres_transport_policy_is_required_and_strict() {
    let mut config = serde_json::to_value(postgres_config()).unwrap();
    config.as_object_mut().unwrap().remove("transport_security");
    let error = serde_json::from_value::<PostgresRecordMaterializerConfig>(config)
        .unwrap_err()
        .to_string();
    assert!(error.contains("transport_security"), "{error}");

    let mut config = serde_json::to_value(postgres_config()).unwrap();
    config["transport_security"]["danger_accept_invalid_certificates"] = json!(true);
    let error = serde_json::from_value::<PostgresRecordMaterializerConfig>(config)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("danger_accept_invalid_certificates"),
        "{error}"
    );
}

#[test]
fn postgres_transport_validation_rejects_unsafe_or_unpinned_settings() {
    let cases = [
        PostgresTransportSecurity::PlaintextLocalProxy {
            acknowledge_plaintext: false,
        },
        PostgresTransportSecurity::VerifiedTls {
            server_names: Vec::new(),
            trust: PostgresTlsTrust::System,
        },
        PostgresTransportSecurity::VerifiedTls {
            server_names: vec!["*.example.invalid".to_owned()],
            trust: PostgresTlsTrust::System,
        },
        PostgresTransportSecurity::VerifiedTls {
            server_names: vec!["db.example.invalid".to_owned()],
            trust: PostgresTlsTrust::PinnedPem {
                certificate_pem_environment: "POSTGRES_ROOT_CERT".to_owned(),
                certificate_sha256: "0".repeat(64),
            },
        },
    ];
    for transport_security in cases {
        let mut config = postgres_config();
        config.transport_security = transport_security;
        assert!(config.validate().is_err());
    }
}

struct MockSearchBackend {
    calls: Mutex<Vec<(usize, usize)>>,
    fail_once_at: Mutex<Option<usize>>,
}

impl SearchBackend for MockSearchBackend {
    fn name(&self) -> &str {
        "mock_search"
    }

    fn configuration(&self) -> Result<Value> {
        Ok(json!({"fields": {"key": "pk-z", "vectors": ["sv-z", "dv-z"]}}))
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "mock".to_owned(),
            revision: "stable".to_owned(),
        })
    }

    fn page_size(&self) -> usize {
        2
    }

    fn discover(
        &self,
        _query: &DiscoveryQuery,
        offset: usize,
        limit: usize,
    ) -> Result<DiscoveryPage> {
        self.calls.lock().unwrap().push((offset, limit));
        let mut fail_once_at = self.fail_once_at.lock().unwrap();
        if *fail_once_at == Some(offset) {
            *fail_once_at = None;
            return Err(anyhow::anyhow!(
                "injected search interruption at offset {offset}"
            ));
        }
        drop(fail_once_at);
        let hit = |record_key: &str| DiscoveryHit {
            record_key: record_key.to_owned(),
            score: 1.0,
            uris: vec![format!("opaque::{record_key}")],
            metadata: BTreeMap::new(),
            inline_text: None,
        };
        let hits = match offset {
            0 => vec![hit("a"), hit("b")],
            2 => vec![hit("b"), hit("c")],
            _ => vec![],
        };
        Ok(DiscoveryPage {
            hits,
            total_hits: Some(4),
            snapshot: SourceSnapshot {
                provider: "mock".to_owned(),
                revision: "stable".to_owned(),
            },
        })
    }
}

struct AdversarialSearchBackend {
    page_size: usize,
    pages: Mutex<VecDeque<DiscoveryPage>>,
}

struct SnapshotOnlySearchBackend {
    snapshot: SourceSnapshot,
}

impl SearchBackend for SnapshotOnlySearchBackend {
    fn name(&self) -> &str {
        "snapshot_only_search"
    }

    fn configuration(&self) -> Result<Value> {
        Ok(json!({"type": "snapshot_only"}))
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(self.snapshot.clone())
    }

    fn page_size(&self) -> usize {
        1
    }

    fn discover(
        &self,
        _query: &DiscoveryQuery,
        _offset: usize,
        _limit: usize,
    ) -> Result<DiscoveryPage> {
        unreachable!("invalid initial snapshots fail before discovery")
    }
}

impl SearchBackend for AdversarialSearchBackend {
    fn name(&self) -> &str {
        "adversarial_search"
    }

    fn configuration(&self) -> Result<Value> {
        Ok(json!({"type": "adversarial"}))
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "mock".to_owned(),
            revision: "stable".to_owned(),
        })
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn discover(
        &self,
        _query: &DiscoveryQuery,
        _offset: usize,
        _limit: usize,
    ) -> Result<DiscoveryPage> {
        self.pages
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("unexpected adversarial search request"))
    }
}

fn discovery_hit(record_key: &str) -> DiscoveryHit {
    DiscoveryHit {
        record_key: record_key.to_owned(),
        score: 1.0,
        uris: vec![format!("opaque::{record_key}")],
        metadata: BTreeMap::new(),
        inline_text: None,
    }
}

fn discovery_page(record_keys: &[&str], total_hits: Option<u64>) -> DiscoveryPage {
    DiscoveryPage {
        hits: record_keys
            .iter()
            .map(|record_key| discovery_hit(record_key))
            .collect(),
        total_hits,
        snapshot: SourceSnapshot {
            provider: "mock".to_owned(),
            revision: "stable".to_owned(),
        },
    }
}

struct MockMaterializer;

impl RecordMaterializer for MockMaterializer {
    fn name(&self) -> &str {
        "mock_store"
    }

    fn configuration(&self) -> Result<Value> {
        Ok(json!({"text_column": "canonical-z"}))
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "mock_store".to_owned(),
            revision: "stable".to_owned(),
        })
    }

    fn materialize(&self, hits: &[DiscoveryHit]) -> Result<Vec<MaterializedRecord>> {
        // Return reverse order to verify that pipeline output is re-aligned to
        // deterministic discovery order.
        Ok(hits
            .iter()
            .rev()
            .map(|hit| {
                let (text, level) = match hit.record_key.as_str() {
                    "a" => ("Alpha   grammar\r\nlesson", "foundation"),
                    "b" => ("Alpha grammar\nlesson", "foundation"),
                    "c" => ("Advanced mathematics guide", "university"),
                    _ => unreachable!(),
                };
                MaterializedRecord {
                    record_key: hit.record_key.clone(),
                    text: text.to_owned(),
                    uris: hit.uris.clone(),
                    metadata: BTreeMap::from([("level".to_owned(), json!(level))]),
                }
            })
            .collect())
    }
}

struct SnapshotMaterializer {
    revision: &'static str,
}

struct DuplicateKeyMaterializer;

impl RecordMaterializer for DuplicateKeyMaterializer {
    fn name(&self) -> &str {
        "duplicate_key_store"
    }

    fn configuration(&self) -> Result<Value> {
        Ok(json!({"type": "adversarial_duplicates"}))
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "duplicate_key_store".to_owned(),
            revision: "stable".to_owned(),
        })
    }

    fn materialize(&self, hits: &[DiscoveryHit]) -> Result<Vec<MaterializedRecord>> {
        let mut records = MockMaterializer.materialize(hits)?;
        records.push(
            records
                .first()
                .expect("pipeline sends a non-empty batch")
                .clone(),
        );
        Ok(records)
    }
}

struct OmittingMaterializer;

impl RecordMaterializer for OmittingMaterializer {
    fn name(&self) -> &str {
        "omitting_store"
    }

    fn configuration(&self) -> Result<Value> {
        Ok(json!({"type": "omits_deleted_rows"}))
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        MockMaterializer.snapshot()
    }

    fn materialize(&self, hits: &[DiscoveryHit]) -> Result<Vec<MaterializedRecord>> {
        // Model a row deleted between discovery and materialization.
        let mut records = MockMaterializer.materialize(hits)?;
        records.retain(|record| record.record_key != "b");
        Ok(records)
    }
}

impl RecordMaterializer for SnapshotMaterializer {
    fn name(&self) -> &str {
        "mock_store"
    }

    fn configuration(&self) -> Result<Value> {
        MockMaterializer.configuration()
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "mock_store".to_owned(),
            revision: self.revision.to_owned(),
        })
    }

    fn materialize(&self, hits: &[DiscoveryHit]) -> Result<Vec<MaterializedRecord>> {
        MockMaterializer.materialize(hits)
    }
}

struct MockTokenizer;

impl CorpusTokenizer for MockTokenizer {
    fn snapshot(&self) -> TokenizerSnapshot {
        TokenizerSnapshot {
            implementation: "mock".to_owned(),
            revision: "1".to_owned(),
            vocabulary_size: 100,
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text
            .split_whitespace()
            .enumerate()
            .map(|(index, _)| index as u32 + 1)
            .collect())
    }
}

struct FixedTokenizer {
    snapshot: TokenizerSnapshot,
    tokens: Vec<u32>,
}

impl CorpusTokenizer for FixedTokenizer {
    fn snapshot(&self) -> TokenizerSnapshot {
        self.snapshot.clone()
    }

    fn encode(&self, _text: &str) -> Result<Vec<u32>> {
        Ok(self.tokens.clone())
    }
}

struct DriftingTokenizer {
    encoded: Mutex<bool>,
}

impl CorpusTokenizer for DriftingTokenizer {
    fn snapshot(&self) -> TokenizerSnapshot {
        TokenizerSnapshot {
            implementation: "drifting".to_owned(),
            revision: if *self.encoded.lock().unwrap() {
                "2".to_owned()
            } else {
                "1".to_owned()
            },
            vocabulary_size: 100,
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        *self.encoded.lock().unwrap() = true;
        MockTokenizer.encode(text)
    }
}

struct CountingTokenizer {
    calls: Mutex<usize>,
}

impl CorpusTokenizer for CountingTokenizer {
    fn snapshot(&self) -> TokenizerSnapshot {
        MockTokenizer.snapshot()
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        *self.calls.lock().unwrap() += 1;
        MockTokenizer.encode(text)
    }
}

struct FailOnceTokenizer {
    fail_on: Mutex<Option<String>>,
}

impl CorpusTokenizer for FailOnceTokenizer {
    fn snapshot(&self) -> TokenizerSnapshot {
        MockTokenizer.snapshot()
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut fail_on = self.fail_on.lock().unwrap();
        if fail_on
            .as_deref()
            .is_some_and(|needle| text.contains(needle))
        {
            *fail_on = None;
            return Err(anyhow::anyhow!("injected tokenizer interruption"));
        }
        drop(fail_on);
        MockTokenizer.encode(text)
    }
}

fn pipeline_config() -> CorpusBuildConfig {
    CorpusBuildConfig {
        version: CORPUS_SCHEMA_VERSION,
        build_id: "test-build".to_owned(),
        discovery: DiscoveryConfig {
            queries: vec![DiscoveryQuery {
                name: "all".to_owned(),
                text: "all records".to_owned(),
                limit: 4,
                parameters: BTreeMap::new(),
            }],
            materialization_batch_size: 2,
        },
        normalization: NormalizationConfig::default(),
        deduplication: DeduplicationConfig::default(),
        classification: ClassificationConfig {
            topic_rules: vec![
                ClassificationRule {
                    label: "language".to_owned(),
                    priority: 0,
                    any_terms: vec!["grammar".to_owned()],
                    all_terms: Vec::new(),
                    none_terms: Vec::new(),
                    metadata_equals: BTreeMap::new(),
                },
                ClassificationRule {
                    label: "math".to_owned(),
                    priority: 0,
                    any_terms: vec!["mathematics".to_owned()],
                    all_terms: Vec::new(),
                    none_terms: Vec::new(),
                    metadata_equals: BTreeMap::new(),
                },
            ],
            difficulty_rules: vec![ClassificationRule {
                label: "advanced".to_owned(),
                priority: 0,
                any_terms: Vec::new(),
                all_terms: Vec::new(),
                none_terms: Vec::new(),
                metadata_equals: BTreeMap::from([("level".to_owned(), json!("university"))]),
            }],
            default_topic: Some("general".to_owned()),
            default_difficulty: Some("basic".to_owned()),
        },
        transformations: vec![TransformationConfig {
            name: "study".to_owned(),
            template: "Study ${topic}: ${text}".to_owned(),
            copies: 1,
            when: RecordPredicate::default(),
        }],
        repetition: RepetitionConfig {
            base_copies: 1,
            max_copies_per_record: 2,
            topic_copies: BTreeMap::new(),
            difficulty_copies: BTreeMap::new(),
        },
        token_target: TokenTarget {
            minimum: 6,
            desired: 6,
            maximum: 6,
        },
        sharding: ShardingConfig {
            max_tokens_per_shard: 7,
        },
    }
}

#[test]
fn corpus_config_bounds_work_amplification_before_pipeline_execution() {
    let mut config = pipeline_config();
    config.discovery.materialization_batch_size = usize::MAX;
    assert!(config.validate().is_err());

    let mut config = pipeline_config();
    config.discovery.queries[0].limit = usize::MAX;
    assert!(config.validate().is_err());

    let mut config = pipeline_config();
    config.repetition.max_copies_per_record = usize::MAX;
    assert!(config.validate().is_err());

    let mut config = pipeline_config();
    config.transformations = (0..1_025)
        .map(|index| TransformationConfig {
            name: format!("transform-{index}"),
            template: "${text}".to_owned(),
            copies: 1,
            when: RecordPredicate::default(),
        })
        .collect();
    assert!(config.validate().is_err());

    let mut config = pipeline_config();
    config.transformations[0].name = "source".to_owned();
    assert!(config.validate().is_err());

    let mut config = pipeline_config();
    config.classification.topic_rules = vec![ClassificationRule {
        label: "non-mathematics".to_owned(),
        priority: 0,
        any_terms: Vec::new(),
        all_terms: Vec::new(),
        none_terms: vec!["mathematics".to_owned()],
        metadata_equals: BTreeMap::new(),
    }];
    config.validate().unwrap();
}

fn target_boundary_config(
    build_id: &str,
    minimum: u64,
    desired: u64,
    maximum: u64,
) -> CorpusBuildConfig {
    let mut config = pipeline_config();
    config.build_id = build_id.to_owned();
    config.token_target = TokenTarget {
        minimum,
        desired,
        maximum,
    };
    config
}

fn build_test_corpus(
    config: CorpusBuildConfig,
    tokenizer: &dyn CorpusTokenizer,
) -> (tempfile::TempDir, std::path::PathBuf, CorpusManifest) {
    let root = tempfile::tempdir().unwrap();
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let (path, manifest) =
        CorpusPipeline::new(config, &backend, &MockMaterializer, tokenizer, &mut dedup)
            .unwrap()
            .run(root.path())
            .unwrap();
    (root, path, manifest)
}

fn rehash_manifest_body(manifest: &mut CorpusManifest) {
    manifest.manifest_sha256 =
        super::pipeline::canonical_json_sha256(&serde_json::to_value(&manifest.build).unwrap())
            .unwrap();
}

fn rehash_manifest_configuration_and_body(manifest: &mut CorpusManifest) {
    manifest.build.config_sha256 =
        super::pipeline::corpus_manifest_configuration_sha256(&manifest.build).unwrap();
    rehash_manifest_body(manifest);
}

#[test]
fn pipeline_rejects_invalid_generic_source_snapshots_before_io() {
    for (label, snapshot) in [
        (
            "provider",
            SourceSnapshot {
                provider: " ".to_owned(),
                revision: "stable".to_owned(),
            },
        ),
        (
            "revision",
            SourceSnapshot {
                provider: "mock".to_owned(),
                revision: String::new(),
            },
        ),
    ] {
        let root = tempfile::tempdir().unwrap();
        let backend = SnapshotOnlySearchBackend { snapshot };
        let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
        let error = CorpusPipeline::new(
            pipeline_config(),
            &backend,
            &MockMaterializer,
            &MockTokenizer,
            &mut dedup,
        )
        .unwrap()
        .run(root.path())
        .unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("source snapshot"), "{label}: {error}");
        assert!(error.contains(label), "{label}: {error}");
    }

    let root = tempfile::tempdir().unwrap();
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let materializer = SnapshotMaterializer { revision: "" };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let error = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &materializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap_err();
    let error = format!("{error:#}");
    assert!(error.contains("materializer"), "{error}");
    assert!(error.contains("revision"), "{error}");
}

#[test]
fn pipeline_rejects_empty_generic_discovery_keys() {
    let root = tempfile::tempdir().unwrap();
    let backend = AdversarialSearchBackend {
        page_size: 2,
        pages: Mutex::new([discovery_page(&[""], Some(1))].into_iter().collect()),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let error = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap_err();
    let error = format!("{error:#}");
    assert!(error.contains("invalid hit"), "{error}");
    assert!(error.contains("record key must not be empty"), "{error}");
}

#[test]
fn pipeline_rejects_non_finite_generic_discovery_scores() {
    for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let root = tempfile::tempdir().unwrap();
        let mut hit = discovery_hit("invalid-score");
        hit.score = score;
        let backend = AdversarialSearchBackend {
            page_size: 1,
            pages: Mutex::new(
                [DiscoveryPage {
                    hits: vec![hit],
                    total_hits: Some(1),
                    snapshot: SourceSnapshot {
                        provider: "mock".to_owned(),
                        revision: "stable".to_owned(),
                    },
                }]
                .into_iter()
                .collect(),
            ),
        };
        let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
        let error = CorpusPipeline::new(
            pipeline_config(),
            &backend,
            &MockMaterializer,
            &MockTokenizer,
            &mut dedup,
        )
        .unwrap()
        .run(root.path())
        .unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("invalid hit"), "{score}: {error}");
        assert!(error.contains("score must be finite"), "{score}: {error}");
    }
}

#[test]
fn pipeline_counts_records_the_materializer_omitted() {
    let root = tempfile::tempdir().unwrap();
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let (_, manifest) = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &OmittingMaterializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap();
    assert_eq!(manifest.build.stats.unmaterialized_records, 1);
    assert_eq!(manifest.build.stats.materialized_records, 2);
}

#[test]
fn pipeline_rejects_duplicate_keys_from_generic_materializers() {
    let root = tempfile::tempdir().unwrap();
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let error = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &DuplicateKeyMaterializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap_err()
    .to_string();
    assert!(error.contains("duplicate record key"), "{error}");
}

#[test]
fn pipeline_rejects_a_search_page_larger_than_the_requested_limit() {
    let root = tempfile::tempdir().unwrap();
    let backend = AdversarialSearchBackend {
        page_size: 2,
        pages: Mutex::new(
            [discovery_page(&["a", "b", "c"], Some(3))]
                .into_iter()
                .collect(),
        ),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let error = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap_err()
    .to_string();
    assert!(error.contains("request limit of 2"), "{error}");
}

#[test]
fn pipeline_rejects_drifting_or_impossible_total_hits() {
    let scenarios = [
        (
            vec![
                discovery_page(&["a", "b"], Some(4)),
                discovery_page(&["b", "c"], Some(5)),
            ],
            "changed total_hits",
        ),
        (
            vec![discovery_page(&["a", "b"], Some(1))],
            "beyond total_hits 1",
        ),
        (vec![discovery_page(&["a"], Some(4))], "short page"),
    ];
    for (index, (pages, expected)) in scenarios.into_iter().enumerate() {
        let root = tempfile::tempdir().unwrap();
        let backend = AdversarialSearchBackend {
            page_size: 2,
            pages: Mutex::new(pages.into_iter().collect()),
        };
        let mut config = pipeline_config();
        config.build_id = format!("invalid-total-{index}");
        let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
        let error = CorpusPipeline::new(
            config,
            &backend,
            &MockMaterializer,
            &MockTokenizer,
            &mut dedup,
        )
        .unwrap()
        .run(root.path())
        .unwrap_err()
        .to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn pipeline_runs_all_stages_and_publishes_immutable_manifest() {
    let root = tempfile::tempdir().unwrap();
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let mut pipeline = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap();
    let (path, manifest) = pipeline.run(root.path()).unwrap();

    assert_eq!(*backend.calls.lock().unwrap(), [(0, 2), (2, 2)]);
    assert_eq!(manifest.build.stats.discovered_hits, 4);
    assert_eq!(manifest.build.stats.duplicate_discovery_keys, 1);
    assert_eq!(manifest.build.stats.materialized_records, 3);
    assert_eq!(manifest.build.stats.duplicate_records, 1);
    assert_eq!(manifest.build.stats.unique_records, 2);
    assert_eq!(manifest.build.stats.unique_tokens, 6);
    assert_eq!(manifest.build.stats.emitted_views, 4);
    assert_eq!(manifest.build.topic_counts["language"], 1);
    assert_eq!(manifest.build.topic_counts["math"], 1);
    assert_eq!(manifest.build.difficulty_counts["advanced"], 1);
    assert!(manifest.build.desired_token_target_reached);
    assert!(path.join("manifest.json").is_file());
    assert!(!root.path().join(".test-build.building").exists());
    manifest.verify(&path).unwrap();

    let lines: Vec<Value> = manifest
        .build
        .shards
        .iter()
        .flat_map(|shard| {
            fs::read_to_string(path.join(&shard.path))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[0]["record_key"], "a");
    assert_eq!(lines[0]["uris"], json!(["opaque::a"]));
    assert_eq!(lines[2]["record_key"], "c");

    let mut second_dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let (existing_path, existing_manifest) = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut second_dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap();
    assert_eq!(existing_path, path);
    assert_eq!(existing_manifest.manifest_sha256, manifest.manifest_sha256);

    let second_root = tempfile::tempdir().unwrap();
    let second_backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut reproducible_dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let (reproduced_path, reproduced) = CorpusPipeline::new(
        pipeline_config(),
        &second_backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut reproducible_dedup,
    )
    .unwrap()
    .run(second_root.path())
    .unwrap();
    assert_eq!(reproduced.manifest_sha256, manifest.manifest_sha256);
    assert_eq!(reproduced.build.shards, manifest.build.shards);

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        let shard = reproduced_path.join(&reproduced.build.shards[0].path);
        let target = reproduced_path.join("relocated-shard.jsonl");
        fs::rename(&shard, &target).unwrap();
        symlink("relocated-shard.jsonl", &shard).unwrap();
        assert!(reproduced.verify(&reproduced_path).is_err());
    }

    let first_shard = path.join(&manifest.build.shards[0].path);
    fs::OpenOptions::new()
        .append(true)
        .open(&first_shard)
        .unwrap()
        .write_all(b"tampered")
        .unwrap();
    assert!(manifest.verify(&path).is_err());
}

#[test]
fn authenticated_corpus_reuses_the_verified_shard_generation() {
    let (_root, path, manifest) = build_test_corpus(pipeline_config(), &MockTokenizer);
    let corpus = AuthenticatedCorpus::open_data_path(&path)
        .unwrap()
        .expect("a corpus directory must produce an authenticated binding");

    assert_eq!(corpus.manifest().manifest_sha256, manifest.manifest_sha256);
    assert_eq!(corpus.shard_count(), manifest.build.shards.len());
    for index in 0..corpus.shard_count() {
        let expected = fs::read(path.join(&manifest.build.shards[index].path)).unwrap();
        let mut actual = Vec::new();
        corpus
            .with_shard(index, |_path, shard| {
                shard.read_to_end(&mut actual)?;
                Ok(())
            })
            .unwrap();
        assert_eq!(actual, expected);
    }
    corpus.ensure_still_published().unwrap();

    let standalone = path.join("ordinary.jsonl");
    fs::write(&standalone, b"{\"text\":\"ordinary\"}\n").unwrap();
    assert!(
        AuthenticatedCorpus::open_data_path(&standalone)
            .unwrap()
            .is_none()
    );
}

#[cfg(unix)]
#[test]
fn verified_corpus_reader_rejects_in_place_mutation_before_returning_more_bytes() {
    let (_root, path, manifest) = build_test_corpus(pipeline_config(), &MockTokenizer);
    let corpus = AuthenticatedCorpus::open_data_path(&path)
        .unwrap()
        .expect("a corpus directory must produce an authenticated binding");
    let published = path.join(&manifest.build.shards[0].path);
    let expected = fs::read(&published).unwrap();

    let mut rest = Vec::new();
    let error = corpus
        .with_verified_shard(0, |_path, shard| {
            let mut first = [0_u8; 1];
            shard.read_exact(&mut first)?;
            fs::OpenOptions::new()
                .append(true)
                .open(&published)?
                .write_all(b"tampered")?;
            shard.read_to_end(&mut rest)?;
            Ok(())
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("changed"), "{error}");
    assert_eq!(
        rest,
        expected[1..],
        "only bytes captured and authenticated before the mutation may escape the verified reader"
    );
}

#[cfg(unix)]
#[test]
fn authenticated_corpus_rejects_persistent_manifest_and_shard_replacements() {
    fn replace_with_same_bytes(path: &std::path::Path) {
        let replacement = path.with_extension("replacement");
        fs::write(&replacement, fs::read(path).unwrap()).unwrap();
        fs::rename(&replacement, path).unwrap();
    }

    let (_manifest_root, manifest_path, _manifest) =
        build_test_corpus(pipeline_config(), &MockTokenizer);
    let manifest_binding = AuthenticatedCorpus::open_data_path(&manifest_path)
        .unwrap()
        .unwrap();
    replace_with_same_bytes(&manifest_path.join("manifest.json"));
    let error = manifest_binding
        .ensure_still_published()
        .unwrap_err()
        .to_string();
    assert!(error.contains("corpus manifest"), "{error}");

    let mut config = pipeline_config();
    config.build_id = "persistent-shard-replacement".to_owned();
    let (_shard_root, shard_path, manifest) = build_test_corpus(config, &MockTokenizer);
    let shard_binding = AuthenticatedCorpus::open_data_path(&shard_path)
        .unwrap()
        .unwrap();
    replace_with_same_bytes(&shard_path.join(&manifest.build.shards[0].path));
    let error = shard_binding
        .ensure_still_published()
        .unwrap_err()
        .to_string();
    assert!(error.contains("corpus shard"), "{error}");

    let mut config = pipeline_config();
    config.build_id = "replacement-during-shard-read".to_owned();
    let (_during_root, during_path, manifest) = build_test_corpus(config, &MockTokenizer);
    let during_binding = AuthenticatedCorpus::open_data_path(&during_path)
        .unwrap()
        .unwrap();
    let published = during_path.join(&manifest.build.shards[0].path);
    let parked = during_path.join("parked-original.jsonl");
    let expected = fs::read(&published).unwrap();
    let error = during_binding
        .with_shard(0, |_path, opened| {
            fs::rename(&published, &parked)?;
            fs::write(&published, &expected)?;
            let mut observed = Vec::new();
            opened.read_to_end(&mut observed)?;
            assert_eq!(
                observed, expected,
                "the opened handle must remain generation A"
            );
            Ok(())
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("corpus shard"), "{error}");
}

#[cfg(unix)]
#[test]
fn authenticated_corpus_rejects_symlinked_root_manifest_and_shard() {
    use std::os::unix::fs::symlink;

    let (root, path, manifest) = build_test_corpus(pipeline_config(), &MockTokenizer);
    let root_alias = root.path().join("root-alias");
    symlink(&path, &root_alias).unwrap();
    assert!(AuthenticatedCorpus::open_data_path(&root_alias).is_err());

    let manifest_path = path.join("manifest.json");
    let real_manifest = path.join("real-manifest.json");
    fs::rename(&manifest_path, &real_manifest).unwrap();
    symlink("real-manifest.json", &manifest_path).unwrap();
    assert!(AuthenticatedCorpus::open_data_path(&manifest_path).is_err());
    fs::remove_file(&manifest_path).unwrap();
    fs::rename(&real_manifest, &manifest_path).unwrap();

    let shard_path = path.join(&manifest.build.shards[0].path);
    let real_shard = path.join("real-shard.jsonl");
    fs::rename(&shard_path, &real_shard).unwrap();
    symlink("real-shard.jsonl", &shard_path).unwrap();
    assert!(AuthenticatedCorpus::open_data_path(&path).is_err());
}

#[test]
fn manifest_verification_recounts_actual_shard_rows_and_tokens() {
    let root = tempfile::tempdir().unwrap();
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let (path, manifest) = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap();

    let mut forged_records = manifest.clone();
    forged_records.build.shards[0].records += 1;
    forged_records.build.stats.emitted_views += 1;
    forged_records.manifest_sha256 = super::pipeline::canonical_json_sha256(
        &serde_json::to_value(&forged_records.build).unwrap(),
    )
    .unwrap();
    let error = forged_records.verify(&path).unwrap_err().to_string();
    assert!(
        error.contains("records but its manifest declares"),
        "{error}"
    );

    let mut forged_tokens = manifest;
    forged_tokens.build.shards[0].tokens += 1;
    forged_tokens.build.stats.exposure_tokens += 1;
    forged_tokens.manifest_sha256 = super::pipeline::canonical_json_sha256(
        &serde_json::to_value(&forged_tokens.build).unwrap(),
    )
    .unwrap();
    let error = forged_tokens.verify(&path).unwrap_err().to_string();
    assert!(
        error.contains("tokens but its manifest declares"),
        "{error}"
    );
}

#[test]
fn manifest_verification_rejects_rehashed_cross_field_forgery() {
    let (_root, path, manifest) = build_test_corpus(pipeline_config(), &MockTokenizer);

    let mut wrong_version = manifest.clone();
    wrong_version.build.version += 1;
    rehash_manifest_body(&mut wrong_version);
    let error = wrong_version.verify(&path).unwrap_err().to_string();
    assert!(error.contains("version differs"), "{error}");

    let mut wrong_build = manifest.clone();
    wrong_build.build.build_id.push_str("-forged");
    rehash_manifest_body(&mut wrong_build);
    let error = wrong_build.verify(&path).unwrap_err().to_string();
    assert!(error.contains("build_id differs"), "{error}");

    let mut changed_component = manifest.clone();
    changed_component.build.discovery.configuration["forged"] = json!(true);
    rehash_manifest_body(&mut changed_component);
    let error = changed_component.verify(&path).unwrap_err().to_string();
    assert!(error.contains("configuration hash"), "{error}");

    let mut changed_hash = manifest.clone();
    changed_hash.build.config_sha256 = "0".repeat(64);
    rehash_manifest_body(&mut changed_hash);
    let error = changed_hash.verify(&path).unwrap_err().to_string();
    assert!(error.contains("configuration hash"), "{error}");

    let mut impossible_stats = manifest.clone();
    impossible_stats.build.stats.unique_records = impossible_stats.build.stats.emitted_views + 1;
    rehash_manifest_body(&mut impossible_stats);
    let error = impossible_stats.verify(&path).unwrap_err().to_string();
    assert!(error.contains("unique-record count"), "{error}");

    let mut impossible_topics = manifest.clone();
    impossible_topics.build.topic_counts = BTreeMap::from([(
        "forged".to_owned(),
        impossible_topics.build.stats.unique_records + 1,
    )]);
    rehash_manifest_body(&mut impossible_topics);
    let error = impossible_topics.verify(&path).unwrap_err().to_string();
    assert!(error.contains("topic counts"), "{error}");

    let mut empty_component_name = manifest;
    empty_component_name.build.materializer.name = " ".to_owned();
    rehash_manifest_body(&mut empty_component_name);
    let error = empty_component_name.verify(&path).unwrap_err().to_string();
    assert!(error.contains("component name"), "{error}");
}

#[test]
fn manifest_verification_rejects_noncanonical_paths_and_symlinked_roots() {
    let (root, path, manifest) = build_test_corpus(pipeline_config(), &MockTokenizer);

    let mut noncanonical = manifest.clone();
    noncanonical.build.shards[0].path = format!("./{}", noncanonical.build.shards[0].path);
    rehash_manifest_body(&mut noncanonical);
    let error = noncanonical.verify(&path).unwrap_err().to_string();
    assert!(error.contains("non-canonical"), "{error}");

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        let alias = root.path().join("corpus-alias");
        symlink(&path, &alias).unwrap();
        let error = manifest.verify(&alias).unwrap_err().to_string();
        assert!(error.contains("real directory"), "{error}");
    }
}

#[test]
fn manifest_verification_rejects_out_of_vocabulary_tokens_and_shard_limit_forgery() {
    let (_root, path, manifest) = build_test_corpus(pipeline_config(), &MockTokenizer);

    let mut out_of_vocabulary = manifest.clone();
    let shard_path = path.join(&out_of_vocabulary.build.shards[0].path);
    let mut rows = fs::read_to_string(&shard_path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    rows[0]["tokens"][0] = json!(out_of_vocabulary.build.tokenizer.vocabulary_size);
    let mut bytes = Vec::new();
    for row in rows {
        bytes.extend(serde_json::to_vec(&row).unwrap());
        bytes.push(b'\n');
    }
    fs::write(&shard_path, &bytes).unwrap();
    out_of_vocabulary.build.shards[0].sha256 = format!("{:x}", Sha256::digest(&bytes));
    rehash_manifest_body(&mut out_of_vocabulary);
    let error = out_of_vocabulary.verify(&path).unwrap_err().to_string();
    assert!(error.contains("outside vocabulary size"), "{error}");

    // Restore the authenticated shard before exercising an independent
    // manifest/configuration forgery against the same immutable build.
    let (_other_root, other_path, mut too_small_shards) =
        build_test_corpus(pipeline_config(), &MockTokenizer);
    too_small_shards.build.config.sharding.max_tokens_per_shard = 1;
    rehash_manifest_configuration_and_body(&mut too_small_shards);
    let error = too_small_shards
        .verify(&other_path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("exceeds configured"), "{error}");
}

#[test]
fn pipeline_rejects_invalid_drifting_and_out_of_vocabulary_tokenizers() {
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let root = tempfile::tempdir().unwrap();
    let invalid = FixedTokenizer {
        snapshot: TokenizerSnapshot {
            implementation: "invalid".to_owned(),
            revision: "1".to_owned(),
            vocabulary_size: 0,
        },
        tokens: vec![0],
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let error = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &MockMaterializer,
        &invalid,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap_err();
    assert!(format!("{error:#}").contains("vocabulary_size must be positive"));
    assert!(backend.calls.lock().unwrap().is_empty());

    let root = tempfile::tempdir().unwrap();
    let out_of_vocabulary = FixedTokenizer {
        snapshot: TokenizerSnapshot {
            implementation: "invalid-output".to_owned(),
            revision: "1".to_owned(),
            vocabulary_size: 2,
        },
        tokens: vec![2],
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let error = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &MockMaterializer,
        &out_of_vocabulary,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap_err();
    assert!(format!("{error:#}").contains("outside vocabulary size"));

    let root = tempfile::tempdir().unwrap();
    let drifting = DriftingTokenizer {
        encoded: Mutex::new(false),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let error = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &MockMaterializer,
        &drifting,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap_err();
    assert!(format!("{error:#}").contains("tokenizer snapshot changed"));
}

#[test]
fn pipeline_rejects_a_single_view_larger_than_its_shard_limit() {
    let mut config = pipeline_config();
    config.build_id = "oversized-view".to_owned();
    config.sharding.max_tokens_per_shard = 2;
    let root = tempfile::tempdir().unwrap();
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let error = CorpusPipeline::new(
        config,
        &backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap_err();
    let error = format!("{error:#}");
    assert!(
        error.contains("one corpus view contains 3 tokens"),
        "{error}"
    );
    assert!(error.contains("max_tokens_per_shard 2"), "{error}");
}

#[cfg(unix)]
#[test]
fn pipeline_rejects_a_symlinked_output_root() {
    use std::os::unix::fs::symlink;

    let parent = tempfile::tempdir().unwrap();
    let target = parent.path().join("real-output");
    fs::create_dir(&target).unwrap();
    let alias = parent.path().join("output-alias");
    symlink(&target, &alias).unwrap();
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();

    let error = CorpusPipeline::new(
        pipeline_config(),
        &backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap()
    .run(&alias)
    .unwrap_err()
    .to_string();
    assert!(error.contains("output root"), "{error}");
    assert!(error.contains("real directory"), "{error}");
    assert!(backend.calls.lock().unwrap().is_empty());
}

#[test]
fn pipeline_tokenizes_identical_repeated_views_once_per_record() {
    let mut config = pipeline_config();
    config.build_id = "tokenization-cache".to_owned();
    config.repetition.base_copies = 2;
    config.repetition.max_copies_per_record = 4;
    config.transformations[0].copies = 2;
    let tokenizer = CountingTokenizer {
        calls: Mutex::new(0),
    };

    let (_root, _path, manifest) = build_test_corpus(config, &tokenizer);

    assert_eq!(manifest.build.stats.unique_records, 2);
    assert_eq!(manifest.build.stats.emitted_views, 8);
    assert_eq!(
        *tokenizer.calls.lock().unwrap(),
        4,
        "each record has only source and transformed token content"
    );
}

#[test]
fn pipeline_stops_oversized_source_at_first_desired_record_boundary() {
    let root = tempfile::tempdir().unwrap();
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let (path, manifest) = CorpusPipeline::new(
        target_boundary_config("oversized-target", 3, 3, 4),
        &backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap();

    // Discovery found a later unique three-token record, but the canonical
    // prefix reached the desired target at `a`, so no later batch was used.
    assert_eq!(manifest.build.stats.discovered_hits, 4);
    assert_eq!(manifest.build.stats.materialized_records, 2);
    assert_eq!(manifest.build.stats.unique_records, 1);
    assert_eq!(manifest.build.stats.unique_tokens, 3);
    assert_eq!(manifest.build.stats.duplicate_records, 0);
    assert!(manifest.build.desired_token_target_reached);
    for shard in &manifest.build.shards {
        for line in fs::read_to_string(path.join(&shard.path)).unwrap().lines() {
            let record: Value = serde_json::from_str(line).unwrap();
            assert_eq!(record["record_key"], "a");
        }
    }
    manifest.verify(&path).unwrap();
}

#[test]
fn pipeline_publishes_legal_prefix_before_maximum_overshoot() {
    let root = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let database = work.path().join("dedup.sqlite");
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut dedup = SqliteDeduplicator::open(&database, DeduplicationConfig::default()).unwrap();
    let (path, manifest) = CorpusPipeline::new(
        target_boundary_config("maximum-boundary", 3, 4, 5),
        &backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap();
    drop(dedup);

    // `a` contributes three unique tokens. `b` is a duplicate, and adding
    // canonical successor `c` would produce six tokens, beyond the maximum.
    assert_eq!(manifest.build.stats.materialized_records, 3);
    assert_eq!(manifest.build.stats.duplicate_records, 1);
    assert_eq!(manifest.build.stats.unique_records, 1);
    assert_eq!(manifest.build.stats.unique_tokens, 3);
    assert!(!manifest.build.desired_token_target_reached);
    let connection = rusqlite::Connection::open(&database).unwrap();
    let committed_keys: i64 = connection
        .query_row("SELECT count(*) FROM corpus_keys", [], |row| row.get(0))
        .unwrap();
    assert_eq!(committed_keys, 1, "overshooting record entered dedup state");
    manifest.verify(&path).unwrap();
}

#[test]
fn pipeline_fails_when_no_in_range_canonical_prefix_exists() {
    let root = tempfile::tempdir().unwrap();
    let backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let error = CorpusPipeline::new(
        target_boundary_config("impossible-boundary", 4, 4, 5),
        &backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut dedup,
    )
    .unwrap()
    .run(root.path())
    .unwrap_err();
    let error = format!("{error:#}");
    assert!(error.contains("no legal canonical prefix"), "{error}");
    assert!(error.contains("below minimum 4"), "{error}");
}

#[test]
fn target_boundary_decision_resumes_identically_after_interruption() {
    let config = target_boundary_config("resumed-target-boundary", 3, 4, 5);
    let baseline_root = tempfile::tempdir().unwrap();
    let baseline_work = tempfile::tempdir().unwrap();
    let baseline_backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut baseline_dedup = SqliteDeduplicator::open(
        &baseline_work.path().join("dedup.sqlite"),
        DeduplicationConfig::default(),
    )
    .unwrap();
    let (_, baseline) = CorpusPipeline::new(
        config.clone(),
        &baseline_backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut baseline_dedup,
    )
    .unwrap()
    .run(baseline_root.path())
    .unwrap();
    drop(baseline_dedup);

    let resumed_root = tempfile::tempdir().unwrap();
    let resumed_work = tempfile::tempdir().unwrap();
    let database = resumed_work.path().join("dedup.sqlite");
    let interrupted_backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let failing_tokenizer = FailOnceTokenizer {
        fail_on: Mutex::new(Some("Advanced mathematics".to_owned())),
    };
    let mut interrupted_dedup =
        SqliteDeduplicator::open(&database, DeduplicationConfig::default()).unwrap();
    let error = CorpusPipeline::new(
        config.clone(),
        &interrupted_backend,
        &MockMaterializer,
        &failing_tokenizer,
        &mut interrupted_dedup,
    )
    .unwrap()
    .run(resumed_root.path())
    .unwrap_err();
    assert!(format!("{error:#}").contains("injected tokenizer interruption"));
    drop(interrupted_dedup);

    let resume_backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut resumed_dedup =
        SqliteDeduplicator::open(&database, DeduplicationConfig::default()).unwrap();
    let (resumed_path, resumed) = CorpusPipeline::new(
        config,
        &resume_backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut resumed_dedup,
    )
    .unwrap()
    .run(resumed_root.path())
    .unwrap();

    assert!(resume_backend.calls.lock().unwrap().is_empty());
    assert_eq!(resumed.manifest_sha256, baseline.manifest_sha256);
    assert_eq!(resumed.build.shards, baseline.build.shards);
    resumed.verify(&resumed_path).unwrap();
}

#[test]
fn pipeline_resumes_discovery_and_uncommitted_shard_tail_exactly() {
    let baseline_root = tempfile::tempdir().unwrap();
    let baseline_work = tempfile::tempdir().unwrap();
    let baseline_backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut baseline_dedup = SqliteDeduplicator::open(
        &baseline_work.path().join("dedup.sqlite"),
        DeduplicationConfig::default(),
    )
    .unwrap();
    let (_, baseline) = CorpusPipeline::new(
        pipeline_config(),
        &baseline_backend,
        &MockMaterializer,
        &FailOnceTokenizer {
            fail_on: Mutex::new(None),
        },
        &mut baseline_dedup,
    )
    .unwrap()
    .run(baseline_root.path())
    .unwrap();
    drop(baseline_dedup);

    let resumed_root = tempfile::tempdir().unwrap();
    let resumed_work = tempfile::tempdir().unwrap();
    let interrupted_backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(Some(2)),
    };
    let mut interrupted_dedup = SqliteDeduplicator::open(
        &resumed_work.path().join("dedup.sqlite"),
        DeduplicationConfig::default(),
    )
    .unwrap();
    let error = CorpusPipeline::new(
        pipeline_config(),
        &interrupted_backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut interrupted_dedup,
    )
    .unwrap()
    .run(resumed_root.path())
    .unwrap_err();
    let error = format!("{error:#}");
    assert!(error.contains("injected search interruption"), "{error}");
    drop(interrupted_dedup);

    let resume_backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let failing_tokenizer = FailOnceTokenizer {
        fail_on: Mutex::new(Some("Study math".to_owned())),
    };
    let mut resumed_dedup = SqliteDeduplicator::open(
        &resumed_work.path().join("dedup.sqlite"),
        DeduplicationConfig::default(),
    )
    .unwrap();
    let error = CorpusPipeline::new(
        pipeline_config(),
        &resume_backend,
        &MockMaterializer,
        &failing_tokenizer,
        &mut resumed_dedup,
    )
    .unwrap()
    .run(resumed_root.path())
    .unwrap_err();
    let error = format!("{error:#}");
    assert!(error.contains("injected tokenizer interruption"), "{error}");
    assert_eq!(*resume_backend.calls.lock().unwrap(), [(2, 2)]);
    let staging = resumed_root.path().join(".test-build.building");
    assert!(staging.join("progress.json").is_file());
    assert!(staging.join("shard-00002.tokens.jsonl").is_file());
    drop(resumed_dedup);

    let final_backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut final_dedup = SqliteDeduplicator::open(
        &resumed_work.path().join("dedup.sqlite"),
        DeduplicationConfig::default(),
    )
    .unwrap();
    let (resumed_path, resumed) = CorpusPipeline::new(
        pipeline_config(),
        &final_backend,
        &MockMaterializer,
        &MockTokenizer,
        &mut final_dedup,
    )
    .unwrap()
    .run(resumed_root.path())
    .unwrap();
    assert!(final_backend.calls.lock().unwrap().is_empty());
    assert_eq!(resumed.manifest_sha256, baseline.manifest_sha256);
    assert_eq!(resumed.build.shards, baseline.build.shards);
    resumed.verify(&resumed_path).unwrap();
}

#[test]
fn pipeline_refuses_resume_when_materializer_snapshot_changes() {
    let output_root = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let database = work.path().join("dedup.sqlite");
    let interrupted_backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(Some(2)),
    };
    let mut interrupted_dedup =
        SqliteDeduplicator::open(&database, DeduplicationConfig::default()).unwrap();
    CorpusPipeline::new(
        pipeline_config(),
        &interrupted_backend,
        &SnapshotMaterializer {
            revision: "generation-1",
        },
        &MockTokenizer,
        &mut interrupted_dedup,
    )
    .unwrap()
    .run(output_root.path())
    .unwrap_err();
    drop(interrupted_dedup);

    let resumed_backend = MockSearchBackend {
        calls: Mutex::new(Vec::new()),
        fail_once_at: Mutex::new(None),
    };
    let mut resumed_dedup =
        SqliteDeduplicator::open(&database, DeduplicationConfig::default()).unwrap();
    let error = CorpusPipeline::new(
        pipeline_config(),
        &resumed_backend,
        &SnapshotMaterializer {
            revision: "generation-2",
        },
        &MockTokenizer,
        &mut resumed_dedup,
    )
    .unwrap()
    .run(output_root.path())
    .unwrap_err();
    let error = format!("{error:#}");
    assert!(error.contains("resume identity changed"), "{error}");
    assert!(resumed_backend.calls.lock().unwrap().is_empty());
}

struct OverlapSearchBackend;

impl SearchBackend for OverlapSearchBackend {
    fn name(&self) -> &str {
        "overlap_search"
    }

    fn configuration(&self) -> Result<Value> {
        Ok(json!({"provider": "overlap-test"}))
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "overlap".to_owned(),
            revision: "1".to_owned(),
        })
    }

    fn page_size(&self) -> usize {
        10
    }

    fn discover(
        &self,
        query: &DiscoveryQuery,
        offset: usize,
        _limit: usize,
    ) -> Result<DiscoveryPage> {
        assert_eq!(offset, 0);
        let hit = |record_key: &str, source: &str| DiscoveryHit {
            record_key: record_key.to_owned(),
            score: if source == "alpha" { 0.8 } else { 0.9 },
            uris: vec![format!("opaque:{source}:{record_key}")],
            metadata: BTreeMap::from([("search_source".to_owned(), json!(source))]),
            inline_text: None,
        };
        let hits = match query.name.as_str() {
            "alpha" => vec![hit("x", "alpha")],
            "omega" => vec![hit("x", "omega"), hit("y", "omega")],
            _ => unreachable!(),
        };
        Ok(DiscoveryPage {
            total_hits: Some(hits.len() as u64),
            hits,
            snapshot: self.snapshot()?,
        })
    }
}

struct OverlapMaterializer;

impl RecordMaterializer for OverlapMaterializer {
    fn name(&self) -> &str {
        "canonical_overlap_store"
    }

    fn configuration(&self) -> Result<Value> {
        Ok(json!({"canonical_metadata": true}))
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "canonical_overlap_store".to_owned(),
            revision: "1".to_owned(),
        })
    }

    fn materialize(&self, hits: &[DiscoveryHit]) -> Result<Vec<MaterializedRecord>> {
        Ok(hits
            .iter()
            .map(|hit| MaterializedRecord {
                record_key: hit.record_key.clone(),
                text: match hit.record_key.as_str() {
                    "x" => "University textbook abstract theorem references".to_owned(),
                    "y" => "Alphabet phonics primary reader".to_owned(),
                    _ => unreachable!(),
                },
                uris: vec![format!("canonical:{}", hit.record_key)],
                metadata: BTreeMap::from([("canonical".to_owned(), json!(true))]),
            })
            .collect())
    }
}

fn overlap_config(reverse_queries_and_rules: bool) -> CorpusBuildConfig {
    let mut queries = vec![
        DiscoveryQuery {
            name: "alpha".to_owned(),
            text: "first".to_owned(),
            limit: 2,
            parameters: BTreeMap::new(),
        },
        DiscoveryQuery {
            name: "omega".to_owned(),
            text: "second".to_owned(),
            limit: 2,
            parameters: BTreeMap::new(),
        },
    ];
    let mut difficulty_rules = vec![
        ClassificationRule {
            label: "university".to_owned(),
            priority: 10,
            any_terms: vec!["university textbook".to_owned()],
            all_terms: Vec::new(),
            none_terms: Vec::new(),
            metadata_equals: BTreeMap::new(),
        },
        ClassificationRule {
            label: "scholarly".to_owned(),
            priority: 20,
            any_terms: vec!["abstract".to_owned(), "theorem".to_owned()],
            all_terms: Vec::new(),
            none_terms: Vec::new(),
            metadata_equals: BTreeMap::new(),
        },
        ClassificationRule {
            label: "foundation".to_owned(),
            priority: 30,
            any_terms: vec!["alphabet".to_owned(), "phonics".to_owned()],
            all_terms: Vec::new(),
            none_terms: Vec::new(),
            metadata_equals: BTreeMap::new(),
        },
    ];
    if reverse_queries_and_rules {
        queries.reverse();
        difficulty_rules.reverse();
    }
    CorpusBuildConfig {
        version: CORPUS_SCHEMA_VERSION,
        build_id: "overlap-build".to_owned(),
        discovery: DiscoveryConfig {
            queries,
            materialization_batch_size: 1,
        },
        normalization: NormalizationConfig::default(),
        deduplication: DeduplicationConfig::default(),
        classification: ClassificationConfig {
            topic_rules: Vec::new(),
            difficulty_rules,
            default_topic: Some("general".to_owned()),
            default_difficulty: Some("unclassified".to_owned()),
        },
        transformations: Vec::new(),
        repetition: RepetitionConfig::default(),
        token_target: TokenTarget {
            minimum: 1,
            desired: 9,
            maximum: 100,
        },
        sharding: ShardingConfig {
            max_tokens_per_shard: 100,
        },
    }
}

#[test]
fn overlapping_queries_and_rules_are_order_independent() {
    let first_root = tempfile::tempdir().unwrap();
    let second_root = tempfile::tempdir().unwrap();
    let mut first_dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let (_, first) = CorpusPipeline::new(
        overlap_config(false),
        &OverlapSearchBackend,
        &OverlapMaterializer,
        &MockTokenizer,
        &mut first_dedup,
    )
    .unwrap()
    .run(first_root.path())
    .unwrap();
    let mut second_dedup = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
    let (_, second) = CorpusPipeline::new(
        overlap_config(true),
        &OverlapSearchBackend,
        &OverlapMaterializer,
        &MockTokenizer,
        &mut second_dedup,
    )
    .unwrap()
    .run(second_root.path())
    .unwrap();

    assert_eq!(first.build.stats.duplicate_discovery_keys, 1);
    assert_eq!(second.build.stats.duplicate_discovery_keys, 1);
    assert_eq!(
        first.build.difficulty_counts,
        second.build.difficulty_counts
    );
    assert_eq!(first.build.difficulty_counts["scholarly"], 1);
    assert_eq!(first.build.difficulty_counts["foundation"], 1);
    assert_eq!(first.build.shards, second.build.shards);
}

#[test]
fn sqlite_deduplicator_detects_key_and_text_duplicates() {
    let directory = tempfile::tempdir().unwrap();
    let mut dedup = SqliteDeduplicator::open(
        &directory.path().join("dedup.sqlite"),
        DeduplicationConfig::default(),
    )
    .unwrap();
    dedup.begin_checkpoint().unwrap();
    let hit = DiscoveryHit {
        record_key: "a".to_owned(),
        score: 1.0,
        uris: Vec::new(),
        metadata: BTreeMap::new(),
        inline_text: None,
    };
    assert!(!dedup.stage_discovery_hit(&hit).unwrap());
    assert!(dedup.stage_discovery_hit(&hit).unwrap());
    assert!(!dedup.seen_or_insert("a", "first").unwrap());
    assert!(dedup.seen_or_insert("a", "different").unwrap());
    assert!(dedup.seen_or_insert("b", "first").unwrap());
    assert!(!dedup.seen_or_insert("b", "second").unwrap());
    dedup.abort_checkpoint().unwrap();
}

#[test]
fn sqlite_deduplicator_rejects_legacy_catalogs() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dedup.sqlite");
    let connection = rusqlite::Connection::open(&path).unwrap();
    connection
        .execute_batch("CREATE TABLE discovery_keys(record_key TEXT PRIMARY KEY NOT NULL)")
        .unwrap();
    drop(connection);

    let error = SqliteDeduplicator::open(&path, DeduplicationConfig::default())
        .err()
        .unwrap();
    let error = format!("{error:#}");
    assert!(error.contains("legacy deduplication catalog"), "{error}");
}

#[test]
fn production_recipe_is_strict_and_validates_without_live_services() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus.production.example.json");
    let recipe = SearchApiPostgresCorpusRecipe::load(&path).unwrap();
    assert_eq!(recipe.corpus.token_target.minimum, 10_000_000_000);
    assert_eq!(recipe.corpus.token_target.maximum, 20_000_000_000);
    assert_eq!(recipe.search_api.fusion.marker_value, "hybrid");
    assert_eq!(
        recipe.search_api.fusion.sparse.vector_field,
        "content_sparse_vectors"
    );
    assert_eq!(recipe.search_api.fusion.dense.vector_field, "dense_vectors");
    assert_eq!(
        recipe.search_api.snapshot.request_revision_pointer,
        "/snapshot_revision"
    );
    assert_eq!(
        recipe.search_api.snapshot.response_revision_pointer,
        "/snapshot/revision"
    );
    assert_eq!(
        recipe
            .search_api
            .request_template
            .pointer(&recipe.search_api.fusion.sparse.clause_pointer)
            .and_then(|clause| clause.get("operator")),
        Some(&json!("sparse"))
    );
    assert_eq!(
        recipe.postgres.connection_environment,
        "CORPUS_POSTGRES_DSN"
    );
    assert!(matches!(
        &recipe.postgres.transport_security,
        PostgresTransportSecurity::VerifiedTls {
            server_names,
            trust: PostgresTlsTrust::PinnedPem { .. },
        } if server_names == &["postgres.example.invalid"]
    ));
    assert!(
        recipe
            .postgres
            .snapshot_statement
            .contains("corpus_generations")
    );
}
