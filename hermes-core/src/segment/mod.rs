pub(crate) mod ann_build;
#[cfg(feature = "native")]
mod builder;
pub(crate) mod format;
#[cfg(feature = "native")]
mod merger;
mod reader;
mod store;
#[cfg(feature = "native")]
mod tracker;
mod types;
mod vector_data;

#[cfg(feature = "native")]
pub use builder::{MemoryBreakdown, SegmentBuilder, SegmentBuilderConfig, SegmentBuilderStats};
#[cfg(feature = "native")]
pub use merger::{MergeStats, SegmentMerger, delete_segment};
pub use reader::{SegmentReader, SparseIndex, VectorIndex, VectorSearchResult};
pub use store::*;
#[cfg(feature = "native")]
pub use tracker::{SegmentSnapshot, SegmentTracker};
pub use types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta, TrainedVectorStructures};
pub use vector_data::{
    FlatVectorData, IVFRaBitQIndexData, LazyFlatVectorData, ScaNNIndexData, dequantize_raw,
};

/// Format byte count as human-readable string
#[cfg(feature = "native")]
pub(crate) fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Write adapter that tracks bytes written.
///
/// Concrete type so it works with generic `serialize<W: Write>` functions
/// (unlike `dyn StreamingWriter` which isn't `Sized`).
#[cfg(feature = "native")]
pub(crate) struct OffsetWriter {
    inner: Box<dyn crate::directories::StreamingWriter>,
    offset: u64,
}

#[cfg(feature = "native")]
impl OffsetWriter {
    pub(crate) fn new(inner: Box<dyn crate::directories::StreamingWriter>) -> Self {
        Self { inner, offset: 0 }
    }

    /// Current write position (total bytes written so far).
    pub(crate) fn offset(&self) -> u64 {
        self.offset
    }

    /// Finalize the underlying streaming writer.
    pub(crate) fn finish(self) -> std::io::Result<()> {
        self.inner.finish()
    }
}

#[cfg(feature = "native")]
impl std::io::Write for OffsetWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.offset += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

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
        let mut builder = SegmentBuilder::new(Arc::clone(&schema), config).unwrap();

        let mut doc = crate::dsl::Document::new();
        doc.add_text(title, "Hello World");
        builder.add_document(doc).unwrap();

        let mut doc = crate::dsl::Document::new();
        doc.add_text(title, "Goodbye World");
        builder.add_document(doc).unwrap();

        builder.build(&dir, segment_id, None).await.unwrap();

        // Open with async reader
        let reader = SegmentReader::open(&dir, segment_id, schema.clone(), 16)
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

    #[tokio::test]
    async fn test_dense_vector_ordinal_tracking() {
        use crate::query::MultiValueCombiner;

        let mut schema_builder = SchemaBuilder::default();
        // Use simple add method - defaults to Flat index
        let embedding = schema_builder.add_dense_vector_field("embedding", 4, true, true);
        let schema = Arc::new(schema_builder.build());

        let dir = RamDirectory::new();
        let segment_id = SegmentId::new();

        let config = SegmentBuilderConfig::default();
        let mut builder = SegmentBuilder::new(Arc::clone(&schema), config).unwrap();

        // Doc 0: single vector
        let mut doc = crate::dsl::Document::new();
        doc.add_dense_vector(embedding, vec![1.0, 0.0, 0.0, 0.0]);
        builder.add_document(doc).unwrap();

        // Doc 1: multi-valued vectors (2 vectors)
        let mut doc = crate::dsl::Document::new();
        doc.add_dense_vector(embedding, vec![0.0, 1.0, 0.0, 0.0]);
        doc.add_dense_vector(embedding, vec![0.0, 0.0, 1.0, 0.0]);
        builder.add_document(doc).unwrap();

        // Doc 2: single vector
        let mut doc = crate::dsl::Document::new();
        doc.add_dense_vector(embedding, vec![0.0, 0.0, 0.0, 1.0]);
        builder.add_document(doc).unwrap();

        builder.build(&dir, segment_id, None).await.unwrap();

        let reader = SegmentReader::open(&dir, segment_id, schema.clone(), 16)
            .await
            .unwrap();

        // Query close to doc 1's first vector
        let query = vec![0.0, 0.9, 0.1, 0.0];
        let results = reader
            .search_dense_vector(embedding, &query, 10, 0, 1.0, MultiValueCombiner::Max)
            .await
            .unwrap();

        // Doc 1 should be in results with ordinal tracking
        let doc1_result = results.iter().find(|r| r.doc_id == 1);
        assert!(doc1_result.is_some(), "Doc 1 should be in results");

        let doc1 = doc1_result.unwrap();
        // Should have 2 ordinals (0 and 1) for the two vectors
        assert!(
            doc1.ordinals.len() <= 2,
            "Doc 1 should have at most 2 ordinals, got {}",
            doc1.ordinals.len()
        );

        // Check ordinals are valid (0 or 1)
        for (ordinal, _score) in &doc1.ordinals {
            assert!(*ordinal <= 1, "Ordinal should be 0 or 1, got {}", ordinal);
        }
    }

    #[tokio::test]
    async fn test_sparse_vector_ordinal_tracking() {
        use crate::query::MultiValueCombiner;

        let mut schema_builder = SchemaBuilder::default();
        let sparse = schema_builder.add_sparse_vector_field("sparse", true, true);
        let schema = Arc::new(schema_builder.build());

        let dir = RamDirectory::new();
        let segment_id = SegmentId::new();

        let config = SegmentBuilderConfig::default();
        let mut builder = SegmentBuilder::new(Arc::clone(&schema), config).unwrap();

        // Doc 0: single sparse vector
        let mut doc = crate::dsl::Document::new();
        doc.add_sparse_vector(sparse, vec![(0, 1.0), (1, 0.5)]);
        builder.add_document(doc).unwrap();

        // Doc 1: multi-valued sparse vectors (2 vectors)
        let mut doc = crate::dsl::Document::new();
        doc.add_sparse_vector(sparse, vec![(0, 0.8), (2, 0.3)]);
        doc.add_sparse_vector(sparse, vec![(1, 0.9), (3, 0.4)]);
        builder.add_document(doc).unwrap();

        // Doc 2: single sparse vector
        let mut doc = crate::dsl::Document::new();
        doc.add_sparse_vector(sparse, vec![(2, 1.0), (3, 0.5)]);
        builder.add_document(doc).unwrap();

        builder.build(&dir, segment_id, None).await.unwrap();

        let reader = SegmentReader::open(&dir, segment_id, schema.clone(), 16)
            .await
            .unwrap();

        // Query matching dimension 0
        let query = vec![(0u32, 1.0f32)];
        let results = reader
            .search_sparse_vector(sparse, &query, 10, MultiValueCombiner::Sum, 1.0)
            .await
            .unwrap();

        // Both doc 0 and doc 1 have dimension 0
        assert!(results.len() >= 2, "Should have at least 2 results");

        // Check doc 1 has ordinal tracking
        let doc1_result = results.iter().find(|r| r.doc_id == 1);
        assert!(doc1_result.is_some(), "Doc 1 should be in results");

        let doc1 = doc1_result.unwrap();
        // Doc 1's first sparse vector has dim 0, so ordinal should be 0
        assert!(
            !doc1.ordinals.is_empty(),
            "Doc 1 should have ordinal information"
        );

        // Check ordinals are valid (0 or 1)
        for (ordinal, _score) in &doc1.ordinals {
            assert!(*ordinal <= 1, "Ordinal should be 0 or 1, got {}", ordinal);
        }
    }
}
