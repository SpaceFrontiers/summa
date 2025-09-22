#![cfg(feature = "native")]

use summa_core::dsl::{PositionMode, parse_single_index};
use summa_core::index::{Index, IndexConfig, IndexWriter};
use summa_core::query::{CountCollector, collect_segment};
use summa_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn unicode_word_queries_keep_punctuated_terms_distinct_after_reopen() {
    parse_single_index("index sample { field body: text<unicode_word> [indexed] }").unwrap();
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field_with_tokenizer("body", true, false, "unicode_word");
    schema.set_positions(body, PositionMode::TokenPosition);
    schema.set_default_fields(vec!["body".into()]);
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for text in [
        "new term_start5",
        "new term start5",
        "new unrelated",
        "user:robert",
        "user robert",
    ] {
        let mut doc = Document::new();
        doc.add_text(body, text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    for source in [
        "+new +term_start5",
        "user\\:robert",
        "\"term start5\"",
        "\"user robert\"",
    ] {
        let query = searcher.query_parser().parse_strict(source).unwrap();
        let mut count = CountCollector::new();
        collect_segment(&searcher.segment_readers()[0], query.as_ref(), &mut count)
            .await
            .unwrap();
        assert_eq!(count.count(), 1, "{source}");
    }
}
