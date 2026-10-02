#![cfg(feature = "native")]

use summa_core::index::{Index, IndexConfig};
use summa_core::query::DocAddress;
use summa_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test]
async fn document_batches_preserve_fields_duplicates_missing_rows_and_reader_generations() {
    check_batches(RamDirectory::new(), false).await;
    let temp = tempfile::tempdir().unwrap();
    check_batches(summa_core::directories::FsDirectory::new(temp.path()), true).await;
}

#[tokio::test]
async fn shared_payload_services_use_the_canonical_document_batch_decoder() {
    use summa_core::directories::{MmapDirectory, PayloadReadBackend, PayloadReadService};
    let mut backends = vec![PayloadReadBackend::Pool];
    if cfg!(all(target_os = "linux", feature = "io-uring")) {
        backends.push(PayloadReadBackend::IoUring);
    }
    for backend in backends {
        let temp = tempfile::tempdir().unwrap();
        let service = PayloadReadService::new(backend).unwrap();
        check_batches(
            MmapDirectory::new(temp.path()).with_payload_reads(service.clone()),
            true,
        )
        .await;
        service.shutdown().await.unwrap();
        assert_eq!(service.stats().quarantined_bytes, 0);
    }
}

async fn check_batches<
    D: summa_core::directories::Directory + summa_core::directories::DirectoryWriter + Clone,
>(
    directory: D,
    lazy: bool,
) {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_fast(id, true);
    schema.set_primary_key(id);
    let title = schema.add_text_field("title", true, true);
    let vector = schema.add_dense_vector_field("vector", 4, false, true);
    let index = Index::create(
        directory,
        schema.build(),
        IndexConfig {
            num_threads: 1,
            num_indexing_threads: 1,
            store_cache_budget_bytes: 0,
            merge_policy: Box::new(summa_core::merge::NoMergePolicy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for batch in 0..2 {
        for n in 0..3 {
            let mut doc = Document::new();
            doc.add_text(id, format!("{batch}-{n}"));
            doc.add_text(title, "first value");
            doc.add_text(title, format!("second value {n}"));
            doc.add_dense_vector(vector, vec![1.0, 2.0, 3.0, n as f32]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    assert_eq!(searcher.num_segments(), 2);
    assert_eq!(searcher.has_lazy_document_reads(), lazy);
    let segments: Vec<_> = searcher
        .segment_readers()
        .iter()
        .map(|s| s.meta().id)
        .collect();
    let addresses = vec![
        DocAddress::new(segments[1], 2),
        DocAddress::new(segments[0], 0),
        DocAddress::new(segments[1], 2),
        DocAddress::new(segments[0], 99),
        DocAddress::new(0, 0),
    ];
    assert!(
        searcher
            .prepare_document_reads(&vec![addresses[0].clone(); 33])
            .await
            .is_err()
    );
    let prepared = searcher.prepare_document_reads(&addresses).await.unwrap();
    for fields in [
        None,
        Some([title.0].into_iter().collect()),
        Some([vector.0].into_iter().collect()),
    ] {
        for (position, address) in addresses.iter().enumerate() {
            let expected = searcher
                .get_document_with_fields(address, fields.as_ref())
                .await
                .unwrap();
            let actual = prepared.get(position, fields.as_ref()).await.unwrap();
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                serde_json::to_value(expected).unwrap()
            );
        }
    }
    assert!(prepared.get(addresses.len(), None).await.is_err());
    let old_document = prepared.get(1, None).await.unwrap().unwrap();
    let deleted_key = old_document.get_first(id).unwrap().as_text().unwrap();
    writer.delete_primary_key(deleted_key).unwrap();
    writer.commit().await.unwrap();
    reader.reload().await.unwrap();
    let current = reader.searcher().await.unwrap();
    let current_batch = current.prepare_document_reads(&addresses).await.unwrap();
    assert!(current_batch.get(1, None).await.unwrap().is_none());
    for (position, address) in addresses.iter().enumerate() {
        assert_eq!(
            serde_json::to_value(current_batch.get(position, None).await.unwrap()).unwrap(),
            serde_json::to_value(current.get_document(address).await.unwrap()).unwrap()
        );
    }
    // The prepared old reader still owns its generation across publication.
    assert!(prepared.get(1, None).await.unwrap().is_some());
    writer.shutdown().await.unwrap();
}

/// A test backend selecting positional reads only at the explicit payload boundary.
struct MappedMetadata {
    mapped: summa_core::directories::MmapDirectory,
    payloads: summa_core::directories::FsDirectory,
    payload_opens: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    fail_payload_open: bool,
}

#[async_trait::async_trait]
impl summa_core::directories::Directory for MappedMetadata {
    async fn exists(&self, path: &std::path::Path) -> std::io::Result<bool> {
        self.mapped.exists(path).await
    }
    async fn file_size(&self, path: &std::path::Path) -> std::io::Result<u64> {
        self.mapped.file_size(path).await
    }
    async fn open_read(
        &self,
        path: &std::path::Path,
    ) -> std::io::Result<summa_core::directories::FileHandle> {
        self.mapped.open_read(path).await
    }
    async fn read_range(
        &self,
        path: &std::path::Path,
        range: std::ops::Range<u64>,
    ) -> std::io::Result<summa_core::directories::OwnedBytes> {
        self.mapped.read_range(path, range).await
    }
    async fn list_files(
        &self,
        prefix: &std::path::Path,
    ) -> std::io::Result<Vec<std::path::PathBuf>> {
        self.mapped.list_files(prefix).await
    }
    async fn open_lazy(
        &self,
        path: &std::path::Path,
    ) -> std::io::Result<summa_core::directories::FileHandle> {
        self.mapped.open_lazy(path).await
    }
    async fn open_payload(
        &self,
        path: &std::path::Path,
    ) -> std::io::Result<summa_core::directories::FileHandle> {
        self.payload_opens
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if self.fail_payload_open {
            return Err(std::io::Error::other("injected payload-open failure"));
        }
        self.payloads.open_lazy(path).await
    }
}

#[tokio::test]
async fn stored_payload_selection_preserves_scoring_and_survives_cache_wrappers() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use summa_core::directories::{
        CachingDirectory, Directory, MmapDirectory, SliceCachingDirectory,
    };
    use summa_core::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentReader};
    let temp = tempfile::tempdir().unwrap();
    let directory = MmapDirectory::new(temp.path());
    let mut schema = SchemaBuilder::default();
    let title = schema.add_text_field("title", true, true);
    let schema = Arc::new(schema.build());
    let mut builder = SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
    for n in 0..3 {
        let mut doc = Document::new();
        doc.add_text(title, format!("payload document {n}"));
        builder.add_document(doc).unwrap();
    }
    let id = SegmentId::new();
    builder.build(&directory, id, None).await.unwrap();
    let baseline = SegmentReader::open(&directory, id, schema.clone(), 0)
        .await
        .unwrap();
    let opens = Arc::new(AtomicUsize::new(0));
    let mixed = || MappedMetadata {
        mapped: directory.clone(),
        payloads: summa_core::directories::FsDirectory::new(temp.path()),
        payload_opens: opens.clone(),
        fail_payload_open: false,
    };
    async fn check<D: Directory>(
        directory: D,
        baseline: &SegmentReader,
        id: SegmentId,
        schema: Arc<summa_core::Schema>,
        title: summa_core::dsl::Field,
        sync: bool,
    ) {
        use summa_core::query::{CountCollector, TermQuery, collect_segment};
        let reader = SegmentReader::open(&directory, id, schema, 0)
            .await
            .unwrap();
        assert!(!reader.store_data_slice().is_sync());
        let query = TermQuery::text(title, "payload");
        let mut count = CountCollector::new();
        collect_segment(&reader, &query, &mut count).await.unwrap();
        assert_eq!(count.count(), 3);
        #[cfg(feature = "sync")]
        if sync {
            let mut count = CountCollector::new();
            summa_core::query::collect_segment_with_limit_sync(&reader, &query, &mut count, 0)
                .unwrap();
            assert_eq!(count.count(), 3);
        }
        #[cfg(not(feature = "sync"))]
        let _ = sync;
        for doc in 0..3 {
            assert_eq!(
                serde_json::to_value(reader.doc(doc).await.unwrap()).unwrap(),
                serde_json::to_value(baseline.doc(doc).await.unwrap()).unwrap()
            );
        }
    }
    check(mixed(), &baseline, id, schema.clone(), title, true).await;
    assert_eq!(opens.load(Ordering::Relaxed), 1);
    check(
        CachingDirectory::new(mixed(), 1024 * 1024),
        &baseline,
        id,
        schema.clone(),
        title,
        true,
    )
    .await;
    assert_eq!(opens.load(Ordering::Relaxed), 2);
    check(
        SliceCachingDirectory::new(mixed(), 1024 * 1024),
        &baseline,
        id,
        schema.clone(),
        title,
        false,
    )
    .await;
    assert_eq!(opens.load(Ordering::Relaxed), 3);
    let mut failed = mixed();
    failed.fail_payload_open = true;
    let error = SegmentReader::open(&failed, id, schema, 0)
        .await
        .err()
        .expect("payload open must fail");
    assert!(error.to_string().contains("injected payload-open failure"));
}

#[cfg(unix)]
#[tokio::test]
async fn cached_payload_handle_keeps_its_open_file_after_unlink() {
    use summa_core::directories::{Directory, FsDirectory, SliceCachingDirectory};
    let temp = tempfile::tempdir().unwrap();
    let path = std::path::Path::new("payload");
    let full_path = temp.path().join(path);
    std::fs::write(&full_path, b"original").unwrap();
    let directory = SliceCachingDirectory::new(FsDirectory::new(temp.path()), 64);
    let handle = directory.open_payload(path).await.unwrap();
    std::fs::remove_file(&full_path).unwrap();
    // A later generation can reuse the pathname; this handle owns the old file.
    std::fs::write(&full_path, b"replaced").unwrap();
    assert_eq!(
        handle.read_bytes_range(0..4).await.unwrap().as_slice(),
        b"orig"
    );
    assert_eq!(
        handle.read_bytes_range(4..8).await.unwrap().as_slice(),
        b"inal"
    );
    assert_eq!(
        handle.read_bytes_range(0..4).await.unwrap().as_slice(),
        b"orig"
    );
    assert!(directory.stats().hits > 0);
    drop(directory);
    assert_eq!(
        handle.read_bytes_range(0..8).await.unwrap().as_slice(),
        b"original"
    );
    assert!(handle.read_bytes_range(0..9).await.is_err());
}
