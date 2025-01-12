#[cfg(feature = "native")]
mod builder;
mod merger;
mod reader;
mod store;
mod types;

#[cfg(feature = "native")]
pub use builder::{SegmentBuilder, SegmentBuilderConfig};
pub use merger::{SegmentMerger, delete_segment};
pub use reader::{AsyncSegmentReader, SegmentReader};
pub use store::*;
pub use types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};

#[cfg(test)]
#[cfg(feature = "native")]
mod tests {
    use super::*;
    use crate::directories::RamDirectory;
    use crate::dsl::SchemaBuilder;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_async_segment_reader() {
        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        let schema = Arc::new(schema_builder.build());

        let dir = RamDirectory::new();
        let segment_id = SegmentId::new();

        // Build segment using sync builder
        let config = SegmentBuilderConfig::default();
        let mut builder = SegmentBuilder::new((*schema).clone(), config).unwrap();

        let mut doc = crate::dsl::Document::new();
        doc.add_text(title, "Hello World");
        builder.add_document(doc).unwrap();

        let mut doc = crate::dsl::Document::new();
        doc.add_text(title, "Goodbye World");
        builder.add_document(doc).unwrap();

        builder.build(&dir, segment_id).await.unwrap();

        // Open with async reader
        let reader = AsyncSegmentReader::open(&dir, segment_id, schema.clone(), 0, 16)
            .await
            .unwrap();

        assert_eq!(reader.num_docs(), 2);

        // Test postings lookup
        let postings = reader.get_postings(title, b"hello").await.unwrap();
        assert!(postings.is_some());
        assert_eq!(postings.unwrap().doc_count(), 1);

        let postings = reader.get_postings(title, b"world").await.unwrap();
        assert!(postings.is_some());
        assert_eq!(postings.unwrap().doc_count(), 2);

        // Test document retrieval
        let doc = reader.doc(0).await.unwrap().unwrap();
        assert_eq!(doc.get_first(title).unwrap().as_text(), Some("Hello World"));
    }
}
