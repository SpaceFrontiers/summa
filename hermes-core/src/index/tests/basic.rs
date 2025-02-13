use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};

#[tokio::test]
async fn test_index_create_and_search() {
    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let body = schema_builder.add_text_field("body", true, true);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    // Create index and add documents
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    let mut doc1 = Document::new();
    doc1.add_text(title, "Hello World");
    doc1.add_text(body, "This is the first document");
    writer.add_document(doc1).unwrap();

    let mut doc2 = Document::new();
    doc2.add_text(title, "Goodbye World");
    doc2.add_text(body, "This is the second document");
    writer.add_document(doc2).unwrap();

    writer.commit().await.unwrap();

    // Open for reading
    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 2);

    // Check postings
    let postings = index.get_postings(title, b"world").await.unwrap();
    assert_eq!(postings.len(), 1); // One segment
    assert_eq!(postings[0].1.doc_count(), 2); // Two docs with "world"

    // Retrieve document via searcher snapshot
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let seg_id = searcher.segment_readers()[0].meta().id;
    let doc = searcher.doc(seg_id, 0).await.unwrap().unwrap();
    assert_eq!(doc.get_first(title).unwrap().as_text(), Some("Hello World"));
}

#[tokio::test]
async fn test_multiple_segments() {
    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig {
        max_indexing_memory_bytes: 1024, // Very small to trigger frequent flushes
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Add documents in batches to create multiple segments
    for batch in 0..3 {
        for i in 0..5 {
            let mut doc = Document::new();
            doc.add_text(title, format!("Document {} batch {}", i, batch));
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    // Open and check
    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 15);
    // With queue-based indexing, exact segment count varies
    assert!(
        index.segment_readers().await.unwrap().len() >= 2,
        "Expected multiple segments"
    );
}

#[tokio::test]
async fn test_segment_merge() {
    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig {
        max_indexing_memory_bytes: 512, // Very small to trigger frequent flushes
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Create multiple segments by flushing between batches
    for batch in 0..3 {
        for i in 0..3 {
            let mut doc = Document::new();
            doc.add_text(title, format!("Document {} batch {}", i, batch));
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    // Should have multiple segments (at least 2, one per flush with docs)
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert!(
        index.segment_readers().await.unwrap().len() >= 2,
        "Expected multiple segments"
    );

    // Force merge
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    // Should have 1 segment now
    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(index.num_docs().await.unwrap(), 9);

    // Verify all documents accessible (order may vary with queue-based indexing)
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let seg_id = searcher.segment_readers()[0].meta().id;
    let mut found_docs = 0;
    for i in 0..9 {
        if searcher.doc(seg_id, i).await.unwrap().is_some() {
            found_docs += 1;
        }
    }
    assert_eq!(found_docs, 9);
}

#[tokio::test]
async fn test_match_query() {
    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let body = schema_builder.add_text_field("body", true, true);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    let mut doc1 = Document::new();
    doc1.add_text(title, "rust programming");
    doc1.add_text(body, "Learn rust language");
    writer.add_document(doc1).unwrap();

    let mut doc2 = Document::new();
    doc2.add_text(title, "python programming");
    doc2.add_text(body, "Learn python language");
    writer.add_document(doc2).unwrap();

    writer.commit().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();

    // Test match query with multiple default fields
    let results = index.query("rust", 10).await.unwrap();
    assert_eq!(results.hits.len(), 1);

    // Test match query with multiple tokens
    let results = index.query("rust programming", 10).await.unwrap();
    assert!(!results.hits.is_empty());

    // Verify hit has address (segment_id + doc_id)
    let hit = &results.hits[0];
    assert!(!hit.address.segment_id.is_empty(), "Should have segment_id");

    // Verify document retrieval by address
    let doc = index.get_document(&hit.address).await.unwrap().unwrap();
    assert!(
        !doc.field_values().is_empty(),
        "Doc should have field values"
    );

    // Also verify doc retrieval via searcher snapshot
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let seg_id = searcher.segment_readers()[0].meta().id;
    let doc = searcher.doc(seg_id, 0).await.unwrap().unwrap();
    assert!(
        !doc.field_values().is_empty(),
        "Doc should have field values"
    );
}

#[tokio::test]
#[cfg(not(feature = "sync"))]
async fn test_slice_cache_warmup_and_load() {
    use crate::directories::SliceCachingDirectory;

    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let body = schema_builder.add_text_field("body", true, true);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    // Create index with some documents
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    for i in 0..10 {
        let mut doc = Document::new();
        doc.add_text(title, format!("Document {} about rust", i));
        doc.add_text(body, format!("This is body text number {}", i));
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Open with slice caching and perform some operations to warm up cache
    let caching_dir = SliceCachingDirectory::new(dir.clone(), 1024 * 1024);
    let index = Index::open(caching_dir, config.clone()).await.unwrap();

    // Perform a search to warm up the cache
    let results = index.query("rust", 10).await.unwrap();
    assert!(!results.hits.is_empty());

    // Check cache stats - should have cached some data
    let stats = index.directory.stats();
    assert!(stats.total_bytes > 0, "Cache should have data after search");
}

#[tokio::test]
async fn test_multivalue_field_indexing_and_search() {
    let mut schema_builder = SchemaBuilder::default();
    let uris = schema_builder.add_text_field("uris", true, true);
    let title = schema_builder.add_text_field("title", true, true);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    // Create index and add document with multi-value field
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    let mut doc = Document::new();
    doc.add_text(uris, "one");
    doc.add_text(uris, "two");
    doc.add_text(title, "Test Document");
    writer.add_document(doc).unwrap();

    // Add another document with different uris
    let mut doc2 = Document::new();
    doc2.add_text(uris, "three");
    doc2.add_text(title, "Another Document");
    writer.add_document(doc2).unwrap();

    writer.commit().await.unwrap();

    // Open for reading
    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 2);

    // Verify document retrieval preserves all values
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let seg_id = searcher.segment_readers()[0].meta().id;
    let doc = searcher.doc(seg_id, 0).await.unwrap().unwrap();
    let all_uris: Vec<_> = doc.get_all(uris).collect();
    assert_eq!(all_uris.len(), 2, "Should have 2 uris values");
    assert_eq!(all_uris[0].as_text(), Some("one"));
    assert_eq!(all_uris[1].as_text(), Some("two"));

    // Verify to_json returns array for multi-value field
    let json = doc.to_json(index.schema());
    let uris_json = json.get("uris").unwrap();
    assert!(uris_json.is_array(), "Multi-value field should be an array");
    let uris_arr = uris_json.as_array().unwrap();
    assert_eq!(uris_arr.len(), 2);
    assert_eq!(uris_arr[0].as_str(), Some("one"));
    assert_eq!(uris_arr[1].as_str(), Some("two"));

    // Verify both values are searchable
    let results = index.query("uris:one", 10).await.unwrap();
    assert_eq!(results.hits.len(), 1, "Should find doc with 'one'");
    assert_eq!(results.hits[0].address.doc_id, 0);

    let results = index.query("uris:two", 10).await.unwrap();
    assert_eq!(results.hits.len(), 1, "Should find doc with 'two'");
    assert_eq!(results.hits[0].address.doc_id, 0);

    let results = index.query("uris:three", 10).await.unwrap();
    assert_eq!(results.hits.len(), 1, "Should find doc with 'three'");
    assert_eq!(results.hits[0].address.doc_id, 1);

    // Verify searching for non-existent value returns no results
    let results = index.query("uris:nonexistent", 10).await.unwrap();
    assert_eq!(results.hits.len(), 0, "Should not find non-existent value");
}
