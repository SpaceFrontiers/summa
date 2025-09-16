//! Regressions for the compatible metadata format 6/7 -> 8 upgrade path.
//!
//! Format 7 only added the optional per-segment `deletions` entry, so a format
//! 6 `metadata.json` written by 1.8.125..=1.8.133 describes segments this build
//! reads unchanged. Opening it must upgrade the stamp loudly instead of
//! refusing, and a writer must persist the upgrade so older builds cannot
//! later drop deletion generations from the same file.

use std::path::Path;

use crate::directories::{Directory, DirectoryWriter, RamDirectory};
use crate::dsl::{Document, SchemaBuilder};
use crate::index::metadata::{INDEX_META_FILENAME, INDEX_META_FORMAT_VERSION};
use crate::index::{Index, IndexConfig, IndexMetadata, IndexWriter};
use crate::query::TermQuery;

async fn on_disk_version(dir: &RamDirectory) -> u64 {
    let bytes = dir
        .open_read(Path::new(INDEX_META_FILENAME))
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap();
    let raw: serde_json::Value = serde_json::from_slice(bytes.as_slice()).unwrap();
    raw["version"].as_u64().unwrap()
}

/// Build a committed index, then rewrite its metadata stamp to the previous
/// format exactly as a 1.8.133 build would have left it.
async fn compatible_fixture(version: u32) -> (RamDirectory, IndexConfig, crate::dsl::Field) {
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("body", true, true);
    let schema = schema.build();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..IndexConfig::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    for i in 0..3 {
        let mut doc = Document::new();
        doc.add_text(body, format!("needle value{i}"));
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
    }
    drop(writer);

    let bytes = dir
        .open_read(Path::new(INDEX_META_FILENAME))
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap();
    let mut raw: serde_json::Value = serde_json::from_slice(bytes.as_slice()).unwrap();
    assert_eq!(
        raw["version"].as_u64().unwrap(),
        u64::from(INDEX_META_FORMAT_VERSION)
    );
    raw["version"] = serde_json::Value::from(version);
    dir.write(
        Path::new(INDEX_META_FILENAME),
        &serde_json::to_vec(&raw).unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(on_disk_version(&dir).await, u64::from(version));
    (dir, config, body)
}

async fn count_hits(index: &Index<RamDirectory>, body: crate::dsl::Field) -> usize {
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = TermQuery::text(body, "needle");
    searcher.search(&query, 10).await.unwrap().len()
}

#[tokio::test]
async fn writer_open_migrates_compatible_metadata_and_persists_current_format() {
    for version in [6, 7] {
        let (dir, config, body) = compatible_fixture(version).await;

        let (index, _writer) = Index::open_with_writer(dir.clone(), config)
            .await
            .expect("a compatible metadata must open");

        assert_eq!(
            on_disk_version(&dir).await,
            u64::from(INDEX_META_FORMAT_VERSION),
            "the writer must persist the upgraded stamp before serving"
        );
        let metadata = IndexMetadata::load(&dir).await.unwrap();
        assert_eq!(metadata.version, INDEX_META_FORMAT_VERSION);
        assert_eq!(
            metadata.segment_metas.len(),
            3,
            "migration must keep every segment"
        );
        assert!(
            metadata
                .segment_metas
                .values()
                .all(|m| m.deletions.is_none()),
            "this fixture carries no deletion generation"
        );
        assert_eq!(count_hits(&index, body).await, 3);
    }
}

#[tokio::test]
async fn read_only_open_migrates_compatible_metadata_in_memory() {
    for version in [6, 7] {
        let (dir, config, body) = compatible_fixture(version).await;

        let index = Index::open(dir.clone(), config)
            .await
            .expect("a read-only open must not refuse compatible metadata");
        assert_eq!(count_hits(&index, body).await, 3);
        assert_eq!(
            on_disk_version(&dir).await,
            u64::from(version),
            "a search-only open never rewrites metadata"
        );
    }
}

#[tokio::test]
async fn legacy_implicit_maxscore_metadata_reopens_and_keeps_its_backend_on_append() {
    use crate::query::SparseVectorQuery;
    use crate::structures::{SparseFormat, SparseVectorConfig};

    for version in [6u32, 7, 8] {
        let mut schema = SchemaBuilder::default();
        let field = schema.add_sparse_vector_field_with_config(
            "sparse",
            true,
            false,
            SparseVectorConfig {
                format: SparseFormat::MaxScore,
                ..Default::default()
            },
        );
        let dir = RamDirectory::new();
        let config = IndexConfig {
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            ..Default::default()
        };
        let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
            .await
            .unwrap();
        let mut doc = Document::new();
        doc.add_sparse_vector(field, vec![(1, 2.0)]);
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
        writer.shutdown().await.unwrap();

        let path = Path::new(INDEX_META_FILENAME);
        let bytes = dir
            .open_read(path)
            .await
            .unwrap()
            .read_bytes()
            .await
            .unwrap();
        let mut raw: serde_json::Value = serde_json::from_slice(bytes.as_slice()).unwrap();
        raw["version"] = version.into();
        raw["schema"]["fields"][0]["sparse_vector_config"]
            .as_object_mut()
            .unwrap()
            .remove("format");
        dir.write(path, &serde_json::to_vec(&raw).unwrap())
            .await
            .unwrap();

        let query = SparseVectorQuery::new(field, vec![(1, 1.0)]).with_exhaustive(true);
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert_eq!(index.search(&query, 10).await.unwrap().hits.len(), 1);
        assert_eq!(on_disk_version(&dir).await, u64::from(version));
        let mut writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        let mut doc = Document::new();
        doc.add_sparse_vector(field, vec![(1, 3.0)]);
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
        writer.force_merge().await.unwrap();
        writer.shutdown().await.unwrap();
        let index = Index::open(dir.clone(), config).await.unwrap();
        let metadata = IndexMetadata::load(&dir).await.unwrap();
        assert_eq!(
            metadata
                .schema
                .get_field_entry(field)
                .unwrap()
                .sparse_vector_config
                .as_ref()
                .unwrap()
                .format,
            SparseFormat::MaxScore
        );
        let hits = index.search(&query, 10).await.unwrap().hits;
        assert_eq!(
            hits.iter().map(|hit| hit.score).collect::<Vec<_>>(),
            [3.0, 2.0]
        );
        let serialized: serde_json::Value =
            serde_json::from_slice(&metadata.serialize_to_bytes().unwrap()).unwrap();
        assert_eq!(
            serialized["schema"]["fields"][0]["sparse_vector_config"]["format"],
            "MaxScore"
        );
    }
    let current = SparseVectorConfig::default();
    assert_eq!(current.format, SparseFormat::Bmp);
    let encoded = serde_json::to_value(&current).unwrap();
    assert_eq!(encoded["format"], "Bmp");
    assert_eq!(
        serde_json::from_value::<SparseVectorConfig>(encoded).unwrap(),
        current
    );
}
