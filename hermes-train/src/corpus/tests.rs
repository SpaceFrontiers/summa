use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::io::Write;
use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use serde_json::{Value, json};

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
        auth: None,
    }
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
