use super::*;
use hermes_core::tokenizer::{SimpleTokenizer, Token, Tokenizer};
use hermes_core::{Document, IndexConfig, SchemaBuilder};
use std::sync::{Mutex, mpsc};
use std::time::Duration;

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
