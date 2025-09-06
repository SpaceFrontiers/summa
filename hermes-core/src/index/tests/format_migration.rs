//! Regressions for the metadata format 6 -> 7 upgrade path.
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

const PREVIOUS_FORMAT_VERSION: u32 = INDEX_META_FORMAT_VERSION - 1;

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
async fn format_6_fixture() -> (RamDirectory, IndexConfig, crate::dsl::Field) {
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
    raw["version"] = serde_json::Value::from(PREVIOUS_FORMAT_VERSION);
    dir.write(
        Path::new(INDEX_META_FILENAME),
        &serde_json::to_vec(&raw).unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        on_disk_version(&dir).await,
        u64::from(PREVIOUS_FORMAT_VERSION)
    );
    (dir, config, body)
}

async fn count_hits(index: &Index<RamDirectory>, body: crate::dsl::Field) -> usize {
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = TermQuery::text(body, "needle");
    searcher.search(&query, 10).await.unwrap().len()
}

#[tokio::test]
async fn writer_open_migrates_format_6_metadata_and_persists_format_7() {
    let (dir, config, body) = format_6_fixture().await;

    let (index, _writer) = Index::open_with_writer(dir.clone(), config)
        .await
        .expect("a format 6 index written after 1.8.125 must open");

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
        "format 6 segments carry no deletion generation"
    );
    assert_eq!(count_hits(&index, body).await, 3);
}

#[tokio::test]
async fn read_only_open_migrates_format_6_metadata_in_memory() {
    let (dir, config, body) = format_6_fixture().await;

    let index = Index::open(dir.clone(), config)
        .await
        .expect("a read-only open must not refuse format 6");
    assert_eq!(count_hits(&index, body).await, 3);
    assert_eq!(
        on_disk_version(&dir).await,
        u64::from(PREVIOUS_FORMAT_VERSION),
        "a search-only open never rewrites metadata"
    );
}
