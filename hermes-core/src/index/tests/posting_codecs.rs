//! End-to-end coverage for posting-codec selection and compatibility stamps.

use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexMetadata, IndexWriter};
use crate::structures::{IndexOptimization, PostingCodec};

#[tokio::test]
async fn inline_source_is_promoted_without_reencoding_external_blocks() {
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("body", true, false);
    let schema = schema.build();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        optimization: IndexOptimization::SizeOptimized,
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..IndexConfig::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    // One inline source plus one external source must become two copied blocks;
    // combining all nine postings into one block would expose re-encoding.
    let mut first = Document::new();
    first.add_text(body, "needle");
    writer.add_document(first).unwrap();
    writer.commit().await.unwrap();
    for i in 0..8 {
        let mut doc = Document::new();
        doc.add_text(body, format!("needle value{i}"));
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    let metadata = IndexMetadata::load(&dir).await.unwrap();
    assert_eq!(
        metadata.version,
        crate::index::metadata::INDEX_META_FORMAT_VERSION,
        "new text layouts use the current index format"
    );

    writer.force_merge().await.unwrap();
    drop(writer);
    let index = Index::open(dir.clone(), config).await.unwrap();
    let segments = index.segment_readers().await.unwrap();
    assert_eq!(segments.len(), 1);
    let postings = segments[0]
        .get_postings(body, b"needle")
        .await
        .unwrap()
        .unwrap();
    assert!(postings.num_blocks() > 0);
    assert_eq!(
        postings.num_blocks(),
        2,
        "inline promotion must not collapse/re-encode the external block"
    );
    assert!(
        (0..postings.num_blocks())
            .all(|block| { postings.block_codec(block) == Some(PostingCodec::Pfor) }),
        "external and promoted inline blocks retain the configured size codec"
    );
}
