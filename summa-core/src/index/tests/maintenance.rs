use crate::directories::RamDirectory;
use crate::index::{Index, IndexConfig, IndexWriter, ReorderConcurrencyGate, ReorderPriority};
use crate::{Document, Error, SchemaBuilder};
use std::sync::Arc;
use std::time::Duration;

async fn reorder_waiting_for_capacity(cancel: bool) {
    let gate = Arc::new(ReorderConcurrencyGate::new(1));
    let occupied = gate.acquire(ReorderPriority::Optimizer).await.unwrap();
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_fast(id, true);
    schema.set_primary_key(id);
    let index = Index::create(
        RamDirectory::new(),
        schema.build(),
        IndexConfig {
            num_indexing_threads: 1,
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            background_reorder_permits: gate,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut initial_writer = index.writer();
    initial_writer.init_primary_key_dedup().await.unwrap();
    let document = |key: &str| {
        let mut doc = Document::new();
        doc.add_text(id, key);
        doc
    };
    initial_writer.add_document(document("first")).unwrap();
    let writer = Arc::new(tokio::sync::RwLock::new(initial_writer));
    let snapshot_ready = Arc::new(tokio::sync::Notify::new());
    let operation = {
        let writer = Arc::clone(&writer);
        let ready = Arc::clone(&snapshot_ready);
        tokio::spawn(async move {
            IndexWriter::reorder_with_shared_writer(&writer, move || {
                ready.notify_one();
                std::future::ready(Ok(()))
            })
            .await
        })
    };
    tokio::time::timeout(Duration::from_secs(5), snapshot_ready.notified())
        .await
        .expect("reorder did not commit and capture its snapshot");
    assert!(!operation.is_finished(), "maintenance gate was bypassed");

    if cancel {
        // Deletion signals this before waiting for the RPC's issued handles.
        // Cancellation must wake the queued operation even while the other
        // index retains all shared maintenance capacity.
        index.segment_manager().begin_shutdown();
        let result = tokio::time::timeout(Duration::from_secs(2), operation)
            .await
            .expect("maintenance cancellation waited for another index's BP pass")
            .unwrap();
        assert!(matches!(result, Err(Error::IndexClosed)));
        drop(occupied);
    } else {
        let mut live_writer = tokio::time::timeout(Duration::from_secs(2), writer.write())
            .await
            .expect("queued maintenance retained the ingestion writer lock");
        live_writer.add_document(document("second")).unwrap();
        live_writer.commit().await.unwrap();
        live_writer.add_document(document("pending")).unwrap();
        drop(live_writer);
        assert_eq!(index.num_docs().await.unwrap(), 2);
        assert!(!operation.is_finished());

        drop(occupied);
        tokio::time::timeout(Duration::from_secs(5), operation)
            .await
            .expect("reorder did not finish after capacity became available")
            .unwrap()
            .unwrap();
        let mut live_writer = writer.write().await;
        for key in ["first", "second", "pending"] {
            assert!(matches!(
                live_writer.add_document(document(key)),
                Err(Error::DuplicatePrimaryKey(_))
            ));
        }
        live_writer.commit().await.unwrap();
        index.reader().await.unwrap().reload().await.unwrap();
        assert_eq!(index.num_docs().await.unwrap(), 3);
    }
    writer.write().await.shutdown().await.unwrap();
    drop(writer);
    index.segment_manager().wait_for_shutdown().await;
}

#[tokio::test]
async fn queued_reorder_allows_ingestion_and_preserves_primary_key_state() {
    reorder_waiting_for_capacity(false).await;
}

#[tokio::test]
async fn deleting_index_cancels_reorder_without_waiting_for_other_indexes() {
    reorder_waiting_for_capacity(true).await;
}

#[tokio::test]
async fn opening_index_with_writer_shares_commits_and_lifecycle_ownership() {
    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut initial = IndexWriter::create(
        dir.clone(),
        SchemaBuilder::default().build(),
        config.clone(),
    )
    .await
    .unwrap();
    initial.add_document(Document::new()).unwrap();
    initial.commit().await.unwrap();
    initial.shutdown().await.unwrap();
    drop(initial);

    let (index, mut writer) = Index::open_with_writer(dir, config).await.unwrap();
    assert!(Arc::ptr_eq(
        index.segment_manager(),
        writer.segment_manager()
    ));
    assert_eq!(index.num_docs().await.unwrap(), 1);
    writer.add_document(Document::new()).unwrap();
    writer.commit().await.unwrap();
    index.reader().await.unwrap().reload().await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 2);
    writer.shutdown().await.unwrap();
    drop(writer);
    index.segment_manager().wait_for_shutdown().await;
}
