use crate::directories::{Directory, RamDirectory};
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig};
use crate::query::SparseVectorQuery;
use crate::segment::SegmentFiles;
use crate::structures::SparseVectorConfig;

#[tokio::test]
async fn seismic_copy_merge_preserves_payloads_and_bounded_maintenance_retires_debt() {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    let vector = schema.add_sparse_vector_field_with_config(
        "vector",
        true,
        false,
        SparseVectorConfig {
            format: crate::structures::SparseFormat::Seismic,
            dims: Some(120),
            ..Default::default()
        },
    );
    let dir = RamDirectory::new();
    let index = Index::create(
        dir.clone(),
        schema.build(),
        IndexConfig {
            merge_policy: Box::new(crate::NoMergePolicy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for batch in 0..2 {
        for row in 0..3 {
            let mut doc = Document::new();
            doc.add_text(id, format!("{batch}-{row}"));
            if row != 1 {
                doc.add_sparse_vector(
                    vector,
                    (0..120)
                        .map(|dim| (dim, (batch * 3 + row + 1) as f32))
                        .collect(),
                );
                if row == 0 {
                    doc.add_sparse_vector(vector, (0..120).map(|dim| (dim, 0.5)).collect());
                }
            }
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    let reader = index.reader().await.unwrap();
    let old_searcher = reader.searcher().await.unwrap();
    let mut copied_payload = Vec::new();
    for segment in old_searcher.segment_readers() {
        let seismic = segment.seismic_index(vector).unwrap();
        let raw = dir
            .open_read(&SegmentFiles::new(segment.meta().id).sparse)
            .await
            .unwrap()
            .read_bytes()
            .await
            .unwrap();
        for (offset, len) in seismic.run_ranges() {
            copied_payload.extend_from_slice(&raw[offset as usize..(offset + len) as usize]);
        }
    }
    writer.force_merge().await.unwrap();
    reader.reload().await.unwrap();
    let merged_searcher = reader.searcher().await.unwrap();
    let merged = &merged_searcher.segment_readers()[0];
    let seismic = merged.seismic_index(vector).unwrap();
    assert_eq!(seismic.total_vectors(), 6);
    assert_eq!(seismic.pending_terms(), 120);
    let raw = dir
        .open_read(&SegmentFiles::new(merged.meta().id).sparse)
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap();
    assert_eq!(
        &raw[..copied_payload.len()],
        copied_payload.as_slice(),
        "normal merge must copy encoded runs byte for byte"
    );
    let keys: Vec<_> = (0..seismic.len())
        .map(|row| {
            (
                seismic.key(row),
                seismic.vector(row).iter().collect::<Vec<_>>(),
            )
        })
        .collect();
    let query = SparseVectorQuery::new(vector, vec![(0, 1.0)]);
    let before: Vec<_> = merged_searcher
        .search(&query, 10)
        .await
        .unwrap()
        .into_iter()
        .map(|hit| (hit.doc_id, hit.score))
        .collect();
    let root_bytes = raw.as_slice().to_vec();
    let mut previous_files = SegmentFiles::new(merged.meta().id);
    let mut expected_debt = 120;
    for pass in 0..crate::segment::seismic::PARTITIONS {
        expected_debt -= if pass < 8 { 8 } else { 7 };
        let mut old_partitions = Vec::new();
        for path in previous_files.seismic_partitions() {
            old_partitions.push(
                dir.open_read(&path)
                    .await
                    .unwrap()
                    .read_bytes()
                    .await
                    .unwrap(),
            );
        }
        writer.reorder().await.unwrap();
        reader.reload().await.unwrap();
        let current = reader.searcher().await.unwrap();
        let seismic = current.segment_readers()[0].seismic_index(vector).unwrap();
        let current_files = SegmentFiles::new(current.segment_readers()[0].meta().id);
        assert_eq!(
            dir.open_read(&current_files.sparse)
                .await
                .unwrap()
                .read_bytes()
                .await
                .unwrap()
                .as_slice(),
            root_bytes,
            "nomination maintenance must retain exact-forward bytes",
        );
        let mut changed = 0;
        for (partition, path) in current_files.seismic_partitions().enumerate() {
            let bytes = dir
                .open_read(&path)
                .await
                .unwrap()
                .read_bytes()
                .await
                .unwrap();
            changed += usize::from(bytes.as_slice() != old_partitions[partition].as_slice());
        }
        assert_eq!(
            changed, 1,
            "one maintenance pass rewrites one shared partition"
        );
        previous_files = current_files;
        assert_eq!(seismic.pending_terms(), expected_debt);
        assert_eq!(
            keys,
            (0..seismic.len())
                .map(|row| {
                    (
                        seismic.key(row),
                        seismic.vector(row).iter().collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        );
        let after: Vec<_> = current
            .search(&query, 10)
            .await
            .unwrap()
            .into_iter()
            .map(|hit| (hit.doc_id, hit.score))
            .collect();
        assert_eq!(before, after);
        let metadata = crate::index::IndexMetadata::load(&dir).await.unwrap();
        let info = metadata.segment_metas.values().next().unwrap();
        assert_eq!(info.seismic_no_progress_passes, 0);
        assert_eq!(
            index
                .segment_manager()
                .unconverged_segments_below(3)
                .await
                .len(),
            usize::from(expected_debt > 0),
            "productive maintenance must stay eligible beyond three passes"
        );
        assert_eq!(
            metadata
                .segment_metas
                .values()
                .next()
                .unwrap()
                .seismic_pending_terms,
            expected_debt
        );
    }
    assert_eq!(old_searcher.num_docs(), 6);
    assert_eq!(
        merged_searcher.search(&query, 10).await.unwrap().len(),
        before.len(),
        "retired reader must retain forward values and nominations"
    );
    writer.delete_primary_key("1-2").unwrap();
    writer.commit().await.unwrap();
    writer.compact(16 * 1024 * 1024).await.unwrap();
    reader.reload().await.unwrap();
    let compacted = reader.searcher().await.unwrap();
    assert_eq!(compacted.num_docs(), 5);
    assert_eq!(
        compacted.segment_readers()[0]
            .seismic_index(vector)
            .unwrap()
            .total_vectors(),
        5
    );
    assert_eq!(
        merged_searcher.num_docs(),
        6,
        "old visibility survives compaction"
    );
}

#[derive(Clone, Default)]
struct MaintenanceFaultDirectory {
    ram: RamDirectory,
    mode: std::sync::Arc<std::sync::atomic::AtomicU8>,
    entered: std::sync::Arc<tokio::sync::Notify>,
    resume: std::sync::Arc<tokio::sync::Notify>,
}
#[async_trait::async_trait]
impl crate::directories::Directory for MaintenanceFaultDirectory {
    async fn exists(&self, path: &std::path::Path) -> std::io::Result<bool> {
        self.ram.exists(path).await
    }
    async fn file_size(&self, path: &std::path::Path) -> std::io::Result<u64> {
        self.ram.file_size(path).await
    }
    async fn open_read(
        &self,
        path: &std::path::Path,
    ) -> std::io::Result<crate::directories::FileHandle> {
        if self.mode.load(std::sync::atomic::Ordering::Acquire) == 4
            && path.extension().is_some_and(|extension| extension == "del")
        {
            self.entered.notify_one();
            self.resume.notified().await;
        }
        self.ram.open_read(path).await
    }
    async fn open_lazy(
        &self,
        path: &std::path::Path,
    ) -> std::io::Result<crate::directories::FileHandle> {
        self.ram.open_lazy(path).await
    }
    async fn read_range(
        &self,
        path: &std::path::Path,
        range: std::ops::Range<u64>,
    ) -> std::io::Result<crate::directories::OwnedBytes> {
        self.ram.read_range(path, range).await
    }
    async fn list_files(&self, path: &std::path::Path) -> std::io::Result<Vec<std::path::PathBuf>> {
        self.ram.list_files(path).await
    }
}
#[async_trait::async_trait]
impl crate::directories::DirectoryWriter for MaintenanceFaultDirectory {
    async fn write(&self, path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
        self.ram.write(path, bytes).await
    }
    async fn delete(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.ram.delete(path).await
    }
    async fn rename(&self, from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
        self.ram.rename(from, to).await
    }
    async fn sync(&self) -> std::io::Result<()> {
        self.ram.sync().await
    }
    async fn streaming_writer(
        &self,
        path: &std::path::Path,
    ) -> std::io::Result<Box<dyn crate::directories::StreamingWriter>> {
        let mode = self.mode.load(std::sync::atomic::Ordering::Acquire);
        if mode == 8 && path == std::path::Path::new("metadata.json.tmp") {
            return Err(std::io::Error::other(
                "injected metadata publication failure",
            ));
        }
        let injected = if mode >= 5 && path.to_string_lossy().ends_with(".seismic.03") {
            mode - 4
        } else if path
            .extension()
            .is_some_and(|extension| extension == "sparse")
        {
            mode
        } else {
            0
        };
        match injected {
            1 => return Err(std::io::Error::other("injected sparse writer failure")),
            2 => {
                self.entered.notify_one();
                std::future::pending::<()>().await;
            }
            3 => panic!("injected sparse writer panic"),
            _ => {}
        }
        self.ram.streaming_writer(path).await
    }
}

#[tokio::test]
async fn failed_cancelled_or_panicked_sparse_maintenance_keeps_source_and_cleans_claimed_output() {
    use std::sync::atomic::Ordering;
    for mode in [1, 2, 3, 5, 6, 7, 8] {
        let dir = MaintenanceFaultDirectory::default();
        let mut schema = SchemaBuilder::default();
        let field = schema.add_sparse_vector_field_with_config(
            "sparse",
            true,
            false,
            SparseVectorConfig {
                format: crate::structures::SparseFormat::Seismic,
                dims: Some(8),
                ..Default::default()
            },
        );
        let index = Index::create(
            dir.clone(),
            schema.build(),
            IndexConfig {
                merge_policy: Box::new(crate::NoMergePolicy),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let mut writer = index.writer();
        for _ in 0..2 {
            let mut doc = Document::new();
            doc.add_sparse_vector(field, vec![(0, 2.0)]);
            writer.add_document(doc).unwrap();
            writer.commit().await.unwrap();
        }
        writer.force_merge().await.unwrap();
        let manager = std::sync::Arc::clone(index.segment_manager());
        let sources = manager.get_segment_ids().await;
        let metadata_before =
            serde_json::to_value(crate::index::IndexMetadata::load(&dir).await.unwrap()).unwrap();
        let mut before = dir.list_files(std::path::Path::new("")).await.unwrap();
        before.sort();
        dir.mode.store(mode, Ordering::Release);
        let operation = {
            let manager = manager.clone();
            let source = sources[0].clone();
            tokio::spawn(async move {
                manager
                    .reorder_single_segment(&source, None, crate::segment::BpBudget::full())
                    .await
            })
        };
        if matches!(mode, 2 | 6) {
            tokio::time::timeout(std::time::Duration::from_secs(5), dir.entered.notified())
                .await
                .unwrap();
            operation.abort();
            assert!(operation.await.unwrap_err().is_cancelled());
        } else if matches!(mode, 3 | 7) {
            assert!(operation.await.unwrap_err().is_panic());
        } else {
            assert!(operation.await.unwrap().is_err());
        }
        dir.mode.store(0, Ordering::Release);
        assert_eq!(
            manager.get_segment_ids().await,
            sources,
            "failed maintenance must not publish replacement metadata"
        );
        assert_eq!(
            serde_json::to_value(crate::index::IndexMetadata::load(&dir).await.unwrap()).unwrap(),
            metadata_before,
            "failed or cancelled publication must not count as maintenance progress"
        );
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                let mut files = dir.list_files(std::path::Path::new("")).await.unwrap();
                files.sort();
                if files == before {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("claimed replacement files leaked after failed maintenance");
        let searcher = index.reader().await.unwrap().searcher().await.unwrap();
        assert_eq!(
            searcher
                .search(&SparseVectorQuery::new(field, vec![(0, 1.0)]), 1)
                .await
                .unwrap()
                .len(),
            1
        );
        writer.shutdown().await.unwrap();
        manager.wait_for_shutdown().await;
    }
}

#[tokio::test]
async fn maintenance_pins_source_visibility_until_replacement_uses_the_latest_mask() {
    use std::sync::atomic::Ordering;
    let dir = MaintenanceFaultDirectory::default();
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    let sparse = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: crate::structures::SparseFormat::Seismic,
            dims: Some(8),
            ..Default::default()
        },
    );
    let index = Index::create(
        dir.clone(),
        schema.build(),
        IndexConfig {
            merge_policy: Box::new(crate::NoMergePolicy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for row in 0..4 {
        let mut doc = Document::new();
        doc.add_text(id, row.to_string());
        doc.add_sparse_vector(sparse, vec![(0, row as f32 + 1.0)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.delete_primary_key("0").unwrap();
    writer.commit().await.unwrap();
    let manager = std::sync::Arc::clone(index.segment_manager());
    let source = manager.get_segment_ids().await[0].clone();
    let original = manager
        .read_metadata(|metadata| metadata.segment_metas[&source].deletions.clone().unwrap())
        .await;
    let refs = manager.tracker().ref_count(&original.id);
    dir.mode.store(4, Ordering::Release);
    let operation = {
        let manager = manager.clone();
        let source = source.clone();
        tokio::spawn(async move {
            manager
                .reorder_single_segment(&source, None, crate::segment::BpBudget::full())
                .await
        })
    };
    tokio::time::timeout(std::time::Duration::from_secs(5), dir.entered.notified())
        .await
        .unwrap();
    assert_eq!(
        manager.tracker().ref_count(&original.id),
        refs + 1,
        "maintenance must own the exact deletion generation before starting its asynchronous read"
    );
    dir.mode.store(0, Ordering::Release);
    writer.delete_primary_key("1").unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), writer.commit())
        .await
        .expect("row deletion must not wait for an address-preserving maintenance read")
        .unwrap();
    let latest = manager
        .read_metadata(|metadata| metadata.segment_metas[&source].deletions.clone().unwrap())
        .await;
    assert_ne!(latest.id, original.id);
    assert!(
        dir.exists(&original.path().unwrap()).await.unwrap(),
        "concurrent delete must retain the mask used by maintenance"
    );
    dir.resume.notify_one();
    assert!(operation.await.unwrap().unwrap());
    let published = manager
        .read_metadata(|metadata| {
            metadata
                .segment_metas
                .values()
                .next()
                .unwrap()
                .deletions
                .clone()
                .unwrap()
        })
        .await;
    assert_eq!(
        published, latest,
        "replacement publication must preserve the latest committed visibility"
    );
    let reader = index.reader().await.unwrap();
    reader.reload().await.unwrap();
    assert_eq!(reader.searcher().await.unwrap().num_docs(), 2);
    writer.shutdown().await.unwrap();
    manager.wait_for_shutdown().await;
}

#[tokio::test]
async fn mixed_sparse_backends_share_copy_merge_maintenance_and_deletion_publication() {
    use crate::structures::{SparseFormat, WeightQuantization};
    let mut schema = SchemaBuilder::default();
    let key = schema.add_text_field("id", true, true);
    schema.set_primary_key(key);
    let fields: Vec<_> = [
        SparseFormat::Bmp,
        SparseFormat::MaxScore,
        SparseFormat::Seismic,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, format)| {
        let field = schema.add_sparse_vector_field_with_config(
            &format!("sparse_{i}"),
            true,
            false,
            SparseVectorConfig {
                format,
                dims: Some(8),
                max_weight: Some(5.0),
                weight_quantization: WeightQuantization::Float32,
                ..Default::default()
            },
        );
        schema.set_multi(field, true);
        if format == SparseFormat::Bmp {
            schema.set_reorder(field, true);
        }
        field
    })
    .collect();
    let dir = RamDirectory::new();
    let index = Index::create(
        dir.clone(),
        schema.build(),
        IndexConfig {
            merge_policy: Box::new(crate::NoMergePolicy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for batch in 0..2 {
        for row in 0..4 {
            let mut doc = Document::new();
            doc.add_text(key, format!("{batch}-{row}"));
            for &field in &fields {
                if row != 1 {
                    doc.add_sparse_vector(field, vec![(0, 1.0 + row as f32 / 4.0), (3, 0.5)]);
                    if row == 0 {
                        doc.add_sparse_vector(field, vec![(0, 0.25)]);
                    }
                }
            }
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    let reader = index.reader().await.unwrap();
    let held = reader.searcher().await.unwrap();
    let mut expected = Vec::new();
    for &field in &fields {
        let query = SparseVectorQuery::new(field, vec![(0, 1.0)])
            .with_combiner(crate::query::MultiValueCombiner::Sum)
            .with_lsp_gamma(0);
        let mut scores = std::collections::BTreeMap::new();
        for hit in held.search(&query, 20).await.unwrap() {
            let doc = held.doc(hit.segment_id, hit.doc_id).await.unwrap().unwrap();
            scores.insert(
                doc.get_first(key).unwrap().as_text().unwrap().to_owned(),
                hit.score,
            );
        }
        assert_eq!(scores.len(), 6);
        expected.push((query, scores));
    }
    writer.force_merge().await.unwrap();
    reader.reload().await.unwrap();
    let merged = reader.searcher().await.unwrap();
    let segment = &merged.segment_readers()[0];
    assert!(segment.bmp_index(fields[0]).is_some());
    assert!(segment.sparse_index(fields[1]).is_some());
    assert_eq!(segment.seismic_index(fields[2]).unwrap().run_count(), 2);
    assert_eq!(segment.seismic_index(fields[2]).unwrap().pending_terms(), 2);
    for phase in 0..3 {
        match phase {
            0 => writer.reorder().await.unwrap(),
            1 => {
                writer.delete_primary_key("1-2").unwrap();
                writer.commit().await.unwrap();
                writer.compact(32 * 1024 * 1024).await.unwrap();
            }
            _ => writer.reorder().await.unwrap(),
        }
        reader.reload().await.unwrap();
        let current = reader.searcher().await.unwrap();
        for (query, scores) in &expected {
            let hits = current.search(query, 20).await.unwrap();
            assert_eq!(hits.len(), if phase == 0 { 6 } else { 5 });
            for hit in hits {
                let doc = current
                    .doc(hit.segment_id, hit.doc_id)
                    .await
                    .unwrap()
                    .unwrap();
                let name = doc.get_first(key).unwrap().as_text().unwrap();
                if phase > 0 {
                    assert_ne!(name, "1-2");
                }
                assert!(
                    (hit.score - scores[name]).abs() < 1e-5,
                    "{name}: {} != {}",
                    hit.score,
                    scores[name]
                );
            }
        }
        assert_eq!(
            current.segment_readers()[0]
                .seismic_index(fields[2])
                .unwrap()
                .pending_terms(),
            if phase == 0 { 1 } else { 0 }
        );
    }
    for (query, _) in &expected {
        assert_eq!(held.search(query, 20).await.unwrap().len(), 6);
    }
}

#[tokio::test]
async fn empty_only_seismic_fields_preserve_ordinals_through_public_build_and_merge() {
    let mut schema = SchemaBuilder::default();
    let vector = schema.add_sparse_vector_field_with_config(
        "vector",
        true,
        false,
        SparseVectorConfig {
            format: crate::structures::SparseFormat::Seismic,
            ..Default::default()
        },
    );
    schema.set_multi(vector, true);
    let index = Index::create(
        RamDirectory::new(),
        schema.build(),
        IndexConfig {
            merge_policy: Box::new(crate::NoMergePolicy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    for _ in 0..2 {
        let mut doc = Document::new();
        doc.add_sparse_vector(vector, vec![]);
        doc.add_sparse_vector(vector, vec![]);
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
    }
    let reader = index.reader().await.unwrap();
    let before = reader.searcher().await.unwrap();
    assert_eq!(before.segment_readers().len(), 2);
    for segment in before.segment_readers() {
        let field = segment
            .seismic_index(vector)
            .expect("empty values retain their field owner");
        assert_eq!(field.total_vectors(), 2);
        assert_eq!((field.key(0).ordinal, field.key(1).ordinal), (0, 1));
        assert_eq!((field.vector(0).len(), field.vector(1).len()), (0, 0));
    }
    writer.force_merge().await.unwrap();
    reader.reload().await.unwrap();
    let after = reader.searcher().await.unwrap();
    let field = after.segment_readers()[0].seismic_index(vector).unwrap();
    assert_eq!(field.total_vectors(), 4);
    assert_eq!(
        (0..field.len())
            .map(|row| (field.key(row).doc, field.key(row).ordinal))
            .collect::<Vec<_>>(),
        vec![(0, 0), (0, 1), (1, 0), (1, 1)]
    );
}

#[tokio::test]
async fn missing_nomination_partition_is_corruption_even_when_its_terms_are_empty() {
    use crate::directories::DirectoryWriter;
    let mut schema = SchemaBuilder::default();
    let field = schema.add_sparse_vector_field_with_config(
        "vector",
        true,
        false,
        SparseVectorConfig {
            format: crate::structures::SparseFormat::Seismic,
            ..Default::default()
        },
    );
    let dir = RamDirectory::new();
    let index = Index::create(dir.clone(), schema.build(), IndexConfig::default())
        .await
        .unwrap();
    let mut writer = index.writer();
    let mut doc = Document::new();
    doc.add_sparse_vector(field, vec![(0, 2.0)]);
    writer.add_document(doc).unwrap();
    writer.commit().await.unwrap();
    let id = index.segment_readers().await.unwrap()[0].meta().id;
    dir.delete(&SegmentFiles::new(id).seismic_partition(7))
        .await
        .unwrap();
    let result =
        crate::segment::SegmentReader::open(&dir, crate::segment::SegmentId(id), index.schema(), 0)
            .await;
    assert!(matches!(result, Err(crate::Error::Corruption(_))));
    writer.shutdown().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn partition_maintenance_hardlinks_forward_and_untouched_nominations() {
    use std::os::unix::fs::MetadataExt;
    let temp = tempfile::tempdir().unwrap();
    let dir = crate::directories::MmapDirectory::new(temp.path());
    let mut schema = SchemaBuilder::default();
    let field = schema.add_sparse_vector_field_with_config(
        "vector",
        true,
        false,
        SparseVectorConfig {
            format: crate::structures::SparseFormat::Seismic,
            ..Default::default()
        },
    );
    let index = Index::create(
        dir,
        schema.build(),
        IndexConfig {
            merge_policy: Box::new(crate::NoMergePolicy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    for _ in 0..2 {
        let mut doc = Document::new();
        doc.add_sparse_vector(field, vec![(0, 2.0), (1, 1.0)]);
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
    }
    writer.force_merge().await.unwrap();
    let reader = index.reader().await.unwrap();
    let held = reader.searcher().await.unwrap();
    let source = SegmentFiles::new(held.segment_readers()[0].meta().id);
    writer.reorder().await.unwrap();
    reader.reload().await.unwrap();
    let current = reader.searcher().await.unwrap();
    let output = SegmentFiles::new(current.segment_readers()[0].meta().id);
    let inode = |path: &std::path::Path| std::fs::metadata(temp.path().join(path)).unwrap().ino();
    assert_eq!(inode(&source.sparse), inode(&output.sparse));
    for partition in 0..crate::segment::seismic::PARTITIONS {
        if partition == 0 {
            assert_ne!(
                inode(&source.seismic_partition(partition)),
                inode(&output.seismic_partition(partition))
            );
        } else {
            assert_eq!(
                inode(&source.seismic_partition(partition)),
                inode(&output.seismic_partition(partition))
            );
        }
    }
    let query = SparseVectorQuery::new(field, vec![(0, 1.0)]);
    assert_eq!(held.search(&query, 10).await.unwrap().len(), 2);
    assert_eq!(current.search(&query, 10).await.unwrap().len(), 2);
    writer.shutdown().await.unwrap();
}

#[tokio::test]
async fn maintained_partition_merges_again_with_correct_rows_ordinals_and_deletions() {
    use std::collections::BTreeMap;

    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    let vector = schema.add_sparse_vector_field_with_config(
        "vector",
        true,
        false,
        SparseVectorConfig {
            format: crate::structures::SparseFormat::Seismic,
            dims: Some(17),
            ..Default::default()
        },
    );
    schema.set_multi(vector, true);
    let dir = RamDirectory::new();
    let index = Index::create(
        dir,
        schema.build(),
        IndexConfig {
            merge_policy: Box::new(crate::NoMergePolicy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    let mut expected = BTreeMap::new();
    for batch in 0..2 {
        for row in 0..3 {
            let name = format!("{batch}-{row}");
            let mut doc = Document::new();
            doc.add_text(id, &name);
            if row != 1 {
                let value = (batch * 3 + row + 1) as f32;
                doc.add_sparse_vector(vector, vec![(0, value), (1, value / 2.0), (16, -value)]);
                let mut score = value * 1.5;
                if row == 0 {
                    doc.add_sparse_vector(vector, vec![(0, 2.0), (1, 0.25)]);
                    score += 2.5;
                }
                expected.insert(name, score);
            }
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.force_merge().await.unwrap();
    writer.delete_primary_key("0-2").unwrap();
    writer.commit().await.unwrap();
    expected.remove("0-2");
    writer.reorder().await.unwrap();
    let reader = index.reader().await.unwrap();
    let held = reader.searcher().await.unwrap();
    let maintained = held.segment_readers()[0].seismic_index(vector).unwrap();
    assert_eq!(maintained.run_count(), 2, "forward runs stay copied");
    assert_eq!(
        maintained.term_runs(0).count(),
        1,
        "partition zero was consolidated"
    );
    assert_eq!(
        maintained.term_runs(1).count(),
        2,
        "partition one remains copied"
    );
    assert_eq!(maintained.pending_terms(), 1);

    for value in [7.0, 8.0] {
        let name = format!("new-{value}");
        let mut doc = Document::new();
        doc.add_text(id, &name);
        doc.add_sparse_vector(vector, vec![(0, value), (1, value / 2.0), (16, -value)]);
        doc.add_sparse_vector(vector, vec![(0, 2.0), (1, 0.25)]);
        expected.insert(name, value * 1.5 + 2.5);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    for phase in 0..3 {
        match phase {
            0 => {}
            1 => writer.force_merge().await.unwrap(),
            _ => writer.reorder().await.unwrap(),
        }
        reader.reload().await.unwrap();
        let current = reader.searcher().await.unwrap();
        for exhaustive in [false, true] {
            let query = SparseVectorQuery::new(vector, vec![(0, 1.0), (1, 2.0), (16, 0.5)])
                .with_combiner(crate::query::MultiValueCombiner::Sum)
                .with_exhaustive(exhaustive);
            let mut actual = BTreeMap::new();
            for hit in current.search(&query, 20).await.unwrap() {
                let doc = current
                    .doc(hit.segment_id, hit.doc_id)
                    .await
                    .unwrap()
                    .unwrap();
                actual.insert(
                    doc.get_first(id).unwrap().as_text().unwrap().to_owned(),
                    hit.score,
                );
            }
            assert_eq!(actual, expected, "phase={phase}, exhaustive={exhaustive}");
            assert_eq!(
                held.search(&query, 20).await.unwrap().len(),
                3,
                "the held maintained generation retains its original visibility and nominations"
            );
        }
    }
    writer.shutdown().await.unwrap();
}

#[tokio::test]
async fn surviving_nomination_partitions_require_a_nonempty_declared_root() {
    use crate::directories::DirectoryWriter;
    for damage in ["missing", "zero", "undeclared"] {
        let mut schema = SchemaBuilder::default();
        let field = schema.add_sparse_vector_field_with_config(
            "vector",
            true,
            false,
            SparseVectorConfig {
                format: crate::structures::SparseFormat::Seismic,
                ..Default::default()
            },
        );
        let dir = RamDirectory::new();
        let index = Index::create(dir.clone(), schema.build(), IndexConfig::default())
            .await
            .unwrap();
        let mut writer = index.writer();
        let mut doc = Document::new();
        doc.add_sparse_vector(field, vec![(0, 2.0)]);
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
        let id = index.segment_readers().await.unwrap()[0].meta().id;
        let files = SegmentFiles::new(id);
        match damage {
            "missing" => dir.delete(&files.sparse).await.unwrap(),
            "zero" => dir.write(&files.sparse, &[]).await.unwrap(),
            _ => {
                let mut empty_root = Vec::new();
                crate::segment::format::write_sparse_toc_and_footer(&mut empty_root, 0, 0, &[])
                    .unwrap();
                dir.write(&files.sparse, &empty_root).await.unwrap();
            }
        }
        let result = crate::segment::SegmentReader::open(
            &dir,
            crate::segment::SegmentId(id),
            index.schema(),
            0,
        )
        .await;
        assert!(
            matches!(result, Err(crate::Error::Corruption(_))),
            "{damage} root was accepted"
        );
        writer.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn seismic_schema_without_values_needs_neither_root_nor_partitions() {
    use crate::directories::DirectoryWriter;
    let mut schema = SchemaBuilder::default();
    let title = schema.add_text_field("title", true, true);
    schema.add_sparse_vector_field_with_config(
        "vector",
        true,
        false,
        SparseVectorConfig {
            format: crate::structures::SparseFormat::Seismic,
            ..Default::default()
        },
    );
    let dir = RamDirectory::new();
    let index = Index::create(dir.clone(), schema.build(), IndexConfig::default())
        .await
        .unwrap();
    let mut writer = index.writer();
    let mut doc = Document::new();
    doc.add_text(title, "no sparse values");
    writer.add_document(doc).unwrap();
    writer.commit().await.unwrap();
    let id = index.segment_readers().await.unwrap()[0].meta().id;
    let files = SegmentFiles::new(id);
    assert!(!dir.exists(&files.sparse).await.unwrap());
    for path in files.seismic_partitions() {
        assert!(!dir.exists(&path).await.unwrap());
    }
    for empty_root in [false, true] {
        if empty_root {
            dir.write(&files.sparse, &[]).await.unwrap();
        }
        let reopened = crate::segment::SegmentReader::open(
            &dir,
            crate::segment::SegmentId(id),
            index.schema(),
            0,
        )
        .await
        .unwrap();
        assert!(reopened.seismic_indexes().is_empty());
        assert_eq!(reopened.num_docs(), 1);
    }
    writer.shutdown().await.unwrap();
}

#[tokio::test]
async fn seismic_stall_limit_survives_restart_and_progress_reopens_followups() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let field = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: crate::structures::SparseFormat::Seismic,
            dims: Some(16),
            ..Default::default()
        },
    );
    let config = || IndexConfig {
        merge_policy: Box::new(crate::NoMergePolicy),
        ..Default::default()
    };
    let index = Index::create(dir.clone(), schema.build(), config())
        .await
        .unwrap();
    let mut writer = index.writer();
    for _ in 0..2 {
        let mut doc = Document::new();
        doc.add_sparse_vector(field, (0..16).map(|dim| (dim, 1.0)).collect());
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
    }
    writer.force_merge().await.unwrap();
    for stalls in 1..=3 {
        let manager = index.segment_manager();
        let source = manager.get_segment_ids().await.remove(0);
        assert!(
            manager
                .reorder_single_segment(
                    &source,
                    None,
                    crate::segment::BpBudget {
                        time_budget: Some(std::time::Duration::ZERO),
                        ..Default::default()
                    },
                )
                .await
                .unwrap()
        );
        let metadata = crate::index::IndexMetadata::load(&dir).await.unwrap();
        let info = metadata.segment_metas.values().next().unwrap();
        assert_eq!(info.seismic_pending_terms, 16);
        assert_eq!(info.seismic_no_progress_passes, stalls);
        assert_eq!(
            manager.unconverged_segments_below(3).await.len(),
            usize::from(stalls < 3)
        );
        assert!(manager.unreordered_segments().await.is_empty());
    }
    writer.shutdown().await.unwrap();
    drop(writer);
    drop(index);
    let reopened = Index::open(dir.clone(), config()).await.unwrap();
    assert!(
        reopened
            .segment_manager()
            .unconverged_segments_below(3)
            .await
            .is_empty()
    );
    let source = reopened.segment_manager().get_segment_ids().await.remove(0);
    reopened
        .segment_manager()
        .reorder_single_segment(&source, None, crate::segment::BpBudget::full())
        .await
        .unwrap();
    let metadata = crate::index::IndexMetadata::load(&dir).await.unwrap();
    let info = metadata.segment_metas.values().next().unwrap();
    assert_eq!(info.seismic_pending_terms, 15);
    assert_eq!(info.seismic_maintenance_passes, 4);
    assert_eq!(info.seismic_no_progress_passes, 0);
    assert_eq!(
        reopened
            .segment_manager()
            .unconverged_segments_below(3)
            .await
            .len(),
        1
    );
    assert!(
        reopened
            .segment_manager()
            .unreordered_segments()
            .await
            .is_empty()
    );
}

#[cfg(unix)]
#[tokio::test]
async fn automatic_seismic_followups_reuse_finished_or_capped_bp_files() {
    use crate::segment::BpBudget;
    use std::os::unix::fs::MetadataExt;
    for (seed_passes, partial) in [(1, false), (3, true), (1, true)] {
        let temp = tempfile::tempdir().unwrap();
        let dir = crate::directories::MmapDirectory::new(temp.path());
        let mut schema = SchemaBuilder::default();
        let text = schema.add_text_field("text", true, true);
        schema.set_reorder(text, true);
        schema.set_positions(text, crate::dsl::PositionMode::TokenPosition);
        let bmp = schema.add_sparse_vector_field_with_config(
            "bmp",
            true,
            false,
            SparseVectorConfig {
                dims: Some(16),
                max_weight: Some(5.0),
                ..Default::default()
            },
        );
        schema.set_reorder(bmp, true);
        let seismic = schema.add_sparse_vector_field_with_config(
            "seismic",
            true,
            false,
            SparseVectorConfig {
                dims: Some(16),
                format: crate::structures::SparseFormat::Seismic,
                ..Default::default()
            },
        );
        let index = Index::create(
            dir.clone(),
            schema.build(),
            IndexConfig {
                merge_policy: Box::new(crate::NoMergePolicy),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let mut writer = index.writer();
        for batch in 0..2 {
            for row in 0..4 {
                let mut document = Document::new();
                document.add_text(
                    text,
                    format!("common topic{} document{}", row % 2, batch * 4 + row),
                );
                document.add_sparse_vector(bmp, vec![(row, 1.0), (15, 0.5)]);
                document.add_sparse_vector(seismic, (0..16).map(|dim| (dim, 1.0)).collect());
                writer.add_document(document).unwrap();
            }
            writer.commit().await.unwrap();
        }
        writer.force_merge().await.unwrap();
        let manager = index.segment_manager();
        for _ in 0..seed_passes {
            let source = manager.get_segment_ids().await.remove(0);
            manager
                .reorder_single_segment(
                    &source,
                    None,
                    BpBudget {
                        min_partition_docs: partial.then_some(256),
                        time_budget: None,
                    },
                )
                .await
                .unwrap();
        }
        let reader = index.reader().await.unwrap();
        reader.reload().await.unwrap();
        let held = reader.searcher().await.unwrap();
        let old_files = SegmentFiles::new(held.segment_readers()[0].meta().id);
        let before = crate::index::IndexMetadata::load(&dir).await.unwrap();
        let info = before.segment_metas.values().next().unwrap();
        assert_eq!(info.bp_converged, !partial);
        let source = manager.get_segment_ids().await.remove(0);
        assert!(
            manager
                .optimize_single_segment(&source, None, BpBudget::full(), 3)
                .await
                .unwrap()
        );
        reader.reload().await.unwrap();
        let current = reader.searcher().await.unwrap();
        let new_files = SegmentFiles::new(current.segment_readers()[0].meta().id);
        let after = crate::index::IndexMetadata::load(&dir).await.unwrap();
        let next = after.segment_metas.values().next().unwrap();
        assert_eq!(next.seismic_pending_terms + 1, info.seismic_pending_terms);
        assert_eq!(
            next.seismic_maintenance_passes,
            info.seismic_maintenance_passes + 1
        );
        assert_eq!(next.generation, info.generation + 1);
        for field in [bmp, seismic] {
            let query = SparseVectorQuery::new(field, vec![(0, 1.0)]);
            let before_hits: Vec<_> = held
                .search(&query, 8)
                .await
                .unwrap()
                .into_iter()
                .map(|hit| (hit.doc_id, hit.score))
                .collect();
            let after_hits: Vec<_> = current
                .search(&query, 8)
                .await
                .unwrap()
                .into_iter()
                .map(|hit| (hit.doc_id, hit.score))
                .collect();
            assert_eq!(before_hits, after_hits);
        }

        let inode =
            |path: &std::path::Path| std::fs::metadata(temp.path().join(path)).unwrap().ino();
        if !partial || seed_passes == 3 {
            assert_eq!(
                (
                    next.reordered,
                    next.bp_converged,
                    next.bp_unconverged_passes
                ),
                (
                    info.reordered,
                    info.bp_converged,
                    info.bp_unconverged_passes
                )
            );
            for (old, new) in [
                (&old_files.sparse, &new_files.sparse),
                (&old_files.term_dict, &new_files.term_dict),
                (&old_files.postings, &new_files.postings),
                (&old_files.positions, &new_files.positions),
                (&old_files.chunks, &new_files.chunks),
            ] {
                assert_eq!(
                    inode(old),
                    inode(new),
                    "completed/capped BP payload was rewritten: {old:?}"
                );
                assert_eq!(
                    std::fs::read(temp.path().join(old)).unwrap(),
                    std::fs::read(temp.path().join(new)).unwrap()
                );
            }
        } else {
            assert!(next.bp_converged, "unfinished BP below the cap must deepen");
            assert_eq!(next.bp_unconverged_passes, 0);
            assert_ne!(inode(&old_files.sparse), inode(&new_files.sparse));
        }
        writer.shutdown().await.unwrap();
    }
}
