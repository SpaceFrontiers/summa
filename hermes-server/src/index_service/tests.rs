use super::*;
use hermes_core::tokenizer::{SimpleTokenizer, Token, Tokenizer};
use hermes_core::{Document, IndexConfig, SchemaBuilder};
use std::sync::{Mutex, mpsc};
use std::time::Duration;

#[tokio::test]
async fn cancelled_force_merge_waiting_for_writer_starts_no_compaction() {
    let root = tempfile::tempdir().unwrap();
    let registry = Arc::new(IndexRegistry::new(
        root.path().to_owned(),
        IndexConfig::default(),
    ));
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    registry.create_index("test", schema.build()).await.unwrap();
    let writer = registry.get_writer("test").await.unwrap();
    let mut held = writer.write().await;
    held.init_primary_key_dedup().await.unwrap();
    for key in ["dead", "live"] {
        let mut doc = Document::new();
        doc.add_text(id, key);
        held.add_document(doc).unwrap();
    }
    held.commit().await.unwrap();
    held.delete_primary_key("dead").unwrap();
    held.commit().await.unwrap();
    let service = IndexServiceImpl {
        registry: registry.clone(),
    };
    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            service.force_merge(Request::new(ForceMergeRequest {
                index_name: "test".into(),
                compact: true,
            }))
        )
        .await
        .is_err()
    );
    drop(held);
    let writer = writer.write().await;
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.path().join("test/metadata.json")).unwrap())
            .unwrap();
    let physical: u64 = metadata["segment_metas"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v["num_docs"].as_u64().unwrap())
        .sum();
    assert_eq!(
        physical, 2,
        "cancelled admission must not start a detached compaction"
    );
    drop(writer);
    registry.shutdown().await.unwrap();
}

#[tokio::test]
async fn force_merge_rpc_compacts_only_when_requested_and_reports_deleted_share() {
    use crate::proto::{
        index_service_client::IndexServiceClient, index_service_server::IndexServiceServer,
        search_service_client::SearchServiceClient, search_service_server::SearchServiceServer,
    };
    let root = tempfile::tempdir().unwrap();
    let registry = Arc::new(IndexRegistry::new(
        root.path().to_owned(),
        IndexConfig {
            merge_policy: Box::new(hermes_core::NoMergePolicy),
            ..Default::default()
        },
    ));
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    registry.create_index("test", schema.build()).await.unwrap();
    let writer = registry.get_writer("test").await.unwrap();
    {
        let mut writer = writer.write().await;
        writer.init_primary_key_dedup().await.unwrap();
        for keys in [["a", "b"], ["c", "d"]] {
            for key in keys {
                let mut doc = Document::new();
                doc.add_text(id, key);
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
        }
        writer.delete_primary_key("a").unwrap();
        writer.commit().await.unwrap();
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown, stop) = tokio::sync::oneshot::channel();
    let index_service = IndexServiceImpl {
        registry: registry.clone(),
    };
    let search_service =
        crate::search_service::SearchServiceImpl::new(registry.clone(), 2, Default::default());
    let server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(IndexServiceServer::new(index_service))
            .add_service(SearchServiceServer::new(search_service))
            .serve_with_incoming_shutdown(
                tonic::codegen::tokio_stream::wrappers::TcpListenerStream::new(listener),
                async {
                    let _ = stop.await;
                },
            ),
    );
    let endpoint = format!("http://{address}");
    let mut client = IndexServiceClient::connect(endpoint.clone()).await.unwrap();
    let mut search = SearchServiceClient::connect(endpoint).await.unwrap();
    for (compact, physical, deleted) in [(false, 4, 1), (true, 3, 0)] {
        client
            .force_merge(ForceMergeRequest {
                index_name: "test".into(),
                compact,
            })
            .await
            .unwrap();
        let info = search
            .get_index_info(GetIndexInfoRequest {
                index_name: "test".into(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(info.num_docs, 3);
        assert_eq!(info.num_segments, 1);
        assert_eq!(info.physical_num_docs, physical);
        assert_eq!(info.num_deleted_docs, deleted);
        assert_eq!(info.deleted_ratio, deleted as f64 / physical as f64);
    }
    shutdown.send(()).unwrap();
    server.await.unwrap().unwrap();
    registry.shutdown().await.unwrap();
}

#[derive(Clone)]
struct GatedTokenizer {
    started: Arc<tokio::sync::Notify>,
    release: Arc<Mutex<mpsc::Receiver<()>>>,
}

impl Tokenizer for GatedTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        if text == "slow" {
            self.started.notify_one();
            let _ = self
                .release
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10));
        }
        SimpleTokenizer.tokenize(text)
    }
}

async fn cancelled_commit_completes(shutdown: bool) {
    let root = std::env::temp_dir().join(format!(
        "hermes_owned_commit_{}",
        hermes_core::segment::SegmentId::new().to_hex()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let registry = Arc::new(IndexRegistry::new(
        root.clone(),
        IndexConfig {
            num_indexing_threads: 1,
            merge_policy: Box::new(hermes_core::merge::NoMergePolicy),
            ..Default::default()
        },
    ));
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("body", true, false);
    registry.create_index("test", schema.build()).await.unwrap();
    let index = registry.get_or_open_index("test").await.unwrap();
    let reader = index.reader().await.unwrap();
    assert_eq!(reader.searcher().await.unwrap().num_docs(), 0);
    let writer = registry.get_writer("test").await.unwrap();
    let (release, receiver) = mpsc::channel();
    let started = Arc::new(tokio::sync::Notify::new());
    writer.write().await.set_tokenizer(
        body,
        GatedTokenizer {
            started: started.clone(),
            release: Arc::new(Mutex::new(receiver)),
        },
    );
    let make_doc = |text: &str| {
        let mut doc = Document::new();
        doc.add_text(body, text);
        doc
    };
    writer.read().await.add_document(make_doc("slow")).unwrap();
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .unwrap();
    let service = IndexServiceImpl {
        registry: registry.clone(),
    };
    let request = tokio::spawn(async move {
        service
            .commit(Request::new(CommitRequest {
                index_name: "test".into(),
            }))
            .await
    });
    // The RPC has transferred exclusive ownership to its completion task.
    tokio::time::timeout(Duration::from_secs(5), async {
        while writer.try_read().is_ok() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    request.abort();
    assert!(request.await.unwrap_err().is_cancelled());
    assert!(
        writer.try_read().is_err(),
        "dropping the RPC released the paused writer"
    );

    if shutdown {
        registry.begin_shutdown();
        let draining = {
            let registry = registry.clone();
            tokio::spawn(async move { registry.shutdown().await })
        };
        tokio::task::yield_now().await;
        assert!(!draining.is_finished());
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), draining)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(reader.searcher().await.unwrap().num_docs(), 1);
    } else {
        release.send(()).unwrap();
        let guard = tokio::time::timeout(Duration::from_secs(5), writer.read())
            .await
            .unwrap();
        assert_eq!(
            reader.searcher().await.unwrap().num_docs(),
            1,
            "detached commit must reload the cached reader"
        );
        guard.add_document(make_doc("next")).unwrap();
        drop(guard);
        let service = IndexServiceImpl {
            registry: registry.clone(),
        };
        let response = service
            .commit(Request::new(CommitRequest {
                index_name: "test".into(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(response.success);
        assert_eq!(response.num_docs, 2);
        registry.shutdown().await.unwrap();
    }

    drop(writer);
    drop(index);
    drop(registry);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn cancelled_commit_rpc_finishes_and_resumes_ingestion() {
    cancelled_commit_completes(false).await;
}

#[tokio::test]
async fn shutdown_waits_for_cancelled_commit_rpc_to_flush_and_publish() {
    cancelled_commit_completes(true).await;
}

#[tokio::test]
async fn mutation_batches_account_for_every_input_and_publish_only_on_commit() {
    let root = tempfile::tempdir().unwrap();
    let registry = Arc::new(IndexRegistry::new(
        root.path().to_owned(),
        IndexConfig::default(),
    ));
    let service = IndexServiceImpl {
        registry: registry.clone(),
    };
    // Envelope limits precede even the missing-index lookup.
    let error = service
        .delete_documents(Request::new(DeleteDocumentsRequest {
            index_name: "missing".into(),
            primary_keys: vec![String::new(); 100_001],
        }))
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    let error = service
        .upsert_documents(Request::new(UpsertDocumentsRequest {
            index_name: "missing".into(),
            documents: vec![NamedDocument::default(); 1_001],
        }))
        .await
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    service.create_index(Request::new(CreateIndexRequest {
        index_name: "docs".into(), schema: "index docs { field id: text<raw> [primary, indexed, stored] field body: text<simple> [indexed<chunked>] }".into(),
    })).await.unwrap();
    fn doc(id: Option<&str>, chunks: &[&str]) -> NamedDocument {
        let mut fields = vec![];
        for (name, text) in id
            .into_iter()
            .map(|id| ("id", id))
            .chain(chunks.iter().map(|text| ("body", *text)))
        {
            fields.push(FieldEntry {
                name: name.into(),
                value: Some(FieldValue {
                    value: Some(field_value::Value::Text(text.into())),
                }),
            });
        }
        NamedDocument { fields }
    }
    let accepted = service
        .upsert_documents(Request::new(UpsertDocumentsRequest {
            index_name: "docs".into(),
            documents: vec![
                doc(None, &["invalid"]),
                doc(Some("a"), &["old", "tail"]),
                doc(Some("b"), &["keep"]),
            ],
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(accepted.accepted_count, 2);
    assert_eq!(
        accepted
            .errors
            .iter()
            .map(|error| error.index)
            .collect::<Vec<_>>(),
        [0]
    );
    let index = registry.get_or_open_index("docs").await.unwrap();
    assert_eq!(
        index
            .reader()
            .await
            .unwrap()
            .searcher()
            .await
            .unwrap()
            .num_docs(),
        0
    );
    service
        .commit(Request::new(CommitRequest {
            index_name: "docs".into(),
        }))
        .await
        .unwrap();
    let old = index.reader().await.unwrap().searcher().await.unwrap();
    assert_eq!(old.num_docs(), 2);
    let response = service
        .upsert_documents(Request::new(UpsertDocumentsRequest {
            index_name: "docs".into(),
            documents: vec![
                doc(Some("a"), &["replacement"]),
                doc(Some("a"), &["conflict"]),
                doc(None, &["missing"]),
            ],
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.accepted_count, 2);
    assert_eq!(
        response.errors.iter().map(|e| e.index).collect::<Vec<_>>(),
        [2]
    );
    let response = service
        .delete_documents(Request::new(DeleteDocumentsRequest {
            index_name: "docs".into(),
            primary_keys: vec!["b".into(), "absent".into(), "".into(), "a".into()],
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.accepted_count, 3);
    assert_eq!(
        response.errors.iter().map(|e| e.index).collect::<Vec<_>>(),
        [2]
    );
    assert_eq!(
        index
            .reader()
            .await
            .unwrap()
            .searcher()
            .await
            .unwrap()
            .num_docs(),
        2
    );
    service
        .commit(Request::new(CommitRequest {
            index_name: "docs".into(),
        }))
        .await
        .unwrap();
    assert_eq!(
        index
            .reader()
            .await
            .unwrap()
            .searcher()
            .await
            .unwrap()
            .num_docs(),
        0
    );
    assert_eq!(old.num_docs(), 2);
    let make_upsert = || {
        Request::new(UpsertDocumentsRequest {
            index_name: "docs".into(),
            documents: vec![doc(Some("concurrent"), &["one replacement"])],
        })
    };
    let (left, right) = tokio::join!(
        service.upsert_documents(make_upsert()),
        service.upsert_documents(make_upsert())
    );
    let (left, right) = (left.unwrap().into_inner(), right.unwrap().into_inner());
    assert_eq!(left.accepted_count + right.accepted_count, 2);
    assert_eq!(left.errors.len() + right.errors.len(), 0);
    service
        .commit(Request::new(CommitRequest {
            index_name: "docs".into(),
        }))
        .await
        .unwrap();
    assert_eq!(
        index
            .reader()
            .await
            .unwrap()
            .searcher()
            .await
            .unwrap()
            .num_docs(),
        1
    );
    drop(old);
    drop(index);
    registry.shutdown().await.unwrap();
}

#[tokio::test]
async fn cancelled_mutations_waiting_for_writer_do_not_stage_work() {
    let root = tempfile::tempdir().unwrap();
    let registry = Arc::new(IndexRegistry::new(
        root.path().to_owned(),
        IndexConfig::default(),
    ));
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    registry.create_index("test", schema.build()).await.unwrap();
    let writer = registry.get_writer("test").await.unwrap();
    let mut held = writer.write().await;
    let mut doc = Document::new();
    doc.add_text(id, "a");
    held.add_document(doc).unwrap();
    held.commit().await.unwrap();
    let service = IndexServiceImpl {
        registry: registry.clone(),
    };
    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            service.delete_documents(Request::new(DeleteDocumentsRequest {
                index_name: "test".into(),
                primary_keys: vec!["a".into()],
            }))
        )
        .await
        .is_err()
    );
    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            service.upsert_documents(Request::new(UpsertDocumentsRequest {
                index_name: "test".into(),
                documents: vec![NamedDocument {
                    fields: vec![FieldEntry {
                        name: "id".into(),
                        value: Some(FieldValue {
                            value: Some(field_value::Value::Text("a".into()))
                        })
                    }]
                }],
            }))
        )
        .await
        .is_err()
    );
    assert!(!held.commit().await.unwrap());
    drop(held);
    registry.shutdown().await.unwrap();
}

#[tokio::test]
async fn content_hash_noops_are_accepted_without_new_physical_rows() {
    let root = tempfile::tempdir().unwrap();
    let registry = Arc::new(IndexRegistry::new(
        root.path().to_owned(),
        IndexConfig::default(),
    ));
    let service = IndexServiceImpl {
        registry: registry.clone(),
    };
    service.create_index(Request::new(CreateIndexRequest {
        index_name: "hashes".into(),
        schema: "index hashes { field id: text<raw> [primary, stored] field digest: text [stored, content_hash] }".into(),
    })).await.unwrap();
    let document = |digest: &str| NamedDocument {
        fields: [("id", "key"), ("digest", digest)]
            .into_iter()
            .map(|(name, text)| FieldEntry {
                name: name.into(),
                value: Some(FieldValue {
                    value: Some(field_value::Value::Text(text.into())),
                }),
            })
            .collect(),
    };
    for (documents, accepted, errors) in [
        (vec![document("v1")], 1, 0),
        (
            vec![
                document("v1"),
                document("v1"),
                document("v2"),
                document("v2"),
            ],
            4,
            0,
        ),
        (vec![document("v2"), document("v2")], 2, 0),
    ] {
        let response = service
            .upsert_documents(Request::new(UpsertDocumentsRequest {
                index_name: "hashes".into(),
                documents,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.accepted_count, accepted);
        assert_eq!(response.errors.len(), errors);
        if errors > 0 {
            assert_eq!(response.errors[0].index, 3);
        }
        service
            .commit(Request::new(CommitRequest {
                index_name: "hashes".into(),
            }))
            .await
            .unwrap();
    }
    let dir = hermes_core::directories::MmapDirectory::new(root.path().join("hashes"));
    let metadata = hermes_core::index::IndexMetadata::load(&dir).await.unwrap();
    assert_eq!(
        metadata
            .segment_metas
            .values()
            .map(|info| info.num_docs)
            .sum::<u32>(),
        2
    );
    assert_eq!(
        metadata
            .segment_metas
            .values()
            .map(|info| info.num_live_docs())
            .sum::<u32>(),
        1
    );
}

#[tokio::test]
async fn large_singleton_upsert_rpc_preserves_complete_document_and_replacement() {
    use crate::proto::{
        index_service_client::IndexServiceClient, index_service_server::IndexServiceServer,
    };
    let root = tempfile::tempdir().unwrap();
    let registry = Arc::new(IndexRegistry::new(
        root.path().to_owned(),
        IndexConfig {
            merge_policy: Box::new(hermes_core::NoMergePolicy),
            ..Default::default()
        },
    ));
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    let body = schema.add_text_field("body", false, true);
    registry
        .create_index("large", schema.build())
        .await
        .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown, stop) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(
                IndexServiceServer::new(IndexServiceImpl {
                    registry: registry.clone(),
                })
                .max_decoding_message_size(256 * 1024 * 1024),
            )
            .serve_with_incoming_shutdown(
                tonic::codegen::tokio_stream::wrappers::TcpListenerStream::new(listener),
                async {
                    let _ = stop.await;
                },
            ),
    );
    let mut client = IndexServiceClient::connect(format!("http://{address}"))
        .await
        .unwrap()
        .max_encoding_message_size(200 * 1024 * 1024);
    let index = registry.get_or_open_index("large").await.unwrap();
    let reader = index.reader().await.unwrap();
    for text in ["x".repeat(34 * 1024 * 1024), "replacement".into()] {
        let response = client
            .upsert_documents(UpsertDocumentsRequest {
                index_name: "large".into(),
                documents: vec![NamedDocument {
                    fields: vec![
                        FieldEntry {
                            name: "id".into(),
                            value: Some(FieldValue {
                                value: Some(field_value::Value::Text("book".into())),
                            }),
                        },
                        FieldEntry {
                            name: "body".into(),
                            value: Some(FieldValue {
                                value: Some(field_value::Value::Text(text.clone())),
                            }),
                        },
                    ],
                }],
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.accepted_count, 1);
        assert!(response.errors.is_empty());
        client
            .commit(CommitRequest {
                index_name: "large".into(),
            })
            .await
            .unwrap();
        reader.reload().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        assert_eq!(searcher.num_docs(), 1);
        let hits = searcher
            .search(&hermes_core::query::AllQuery, 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        let stored = searcher
            .doc(hits[0].segment_id, hits[0].doc_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            stored.get_first(body).unwrap().as_text(),
            Some(text.as_str())
        );
    }
    shutdown.send(()).unwrap();
    server.await.unwrap().unwrap();
    registry.shutdown().await.unwrap();
}
