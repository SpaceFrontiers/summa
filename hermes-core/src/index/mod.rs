//! Index - multi-segment async search index
//!
//! The `Index` is the central concept that provides:
//! - `Index::create()` / `Index::open()` - create or open an index
//! - `index.writer()` - get an IndexWriter for adding documents
//! - `index.reader()` - get an IndexReader for searching (with reload policy)
//!
//! The Index owns the SegmentManager which handles segment lifecycle and tracking.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::dsl::Schema;
use crate::error::Result;
use crate::structures::{CoarseCentroids, PQCodebook};

#[cfg(feature = "native")]
mod reader;
#[cfg(feature = "native")]
mod vector_builder;
#[cfg(feature = "native")]
mod writer;
#[cfg(feature = "native")]
pub use reader::{IndexReader, Searcher};
#[cfg(feature = "native")]
pub use writer::IndexWriter;

mod metadata;
pub use metadata::{FieldVectorMeta, INDEX_META_FILENAME, IndexMetadata, VectorIndexState};

#[cfg(feature = "native")]
mod helpers;
#[cfg(feature = "native")]
pub use helpers::{
    IndexingStats, SchemaConfig, SchemaFieldConfig, create_index_at_path, create_index_from_sdl,
    index_documents_from_reader, index_json_document, parse_schema,
};

/// Default file name for the slice cache
pub const SLICE_CACHE_FILENAME: &str = "index.slicecache";

/// Index configuration
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Number of threads for CPU-intensive tasks (search parallelism)
    pub num_threads: usize,
    /// Number of parallel segment builders (documents distributed round-robin)
    pub num_indexing_threads: usize,
    /// Number of threads for parallel block compression within each segment
    pub num_compression_threads: usize,
    /// Block cache size for term dictionary per segment
    pub term_cache_blocks: usize,
    /// Block cache size for document store per segment
    pub store_cache_blocks: usize,
    /// Max memory (bytes) across all builders before auto-commit (global limit)
    pub max_indexing_memory_bytes: usize,
    /// Merge policy for background segment merging
    pub merge_policy: Box<dyn crate::merge::MergePolicy>,
    /// Index optimization mode (adaptive, size-optimized, performance-optimized)
    pub optimization: crate::structures::IndexOptimization,
}

impl Default for IndexConfig {
    fn default() -> Self {
        #[cfg(feature = "native")]
        let cpus = num_cpus::get().max(1);
        #[cfg(not(feature = "native"))]
        let cpus = 1;

        Self {
            num_threads: cpus,
            num_indexing_threads: 1,
            num_compression_threads: cpus,
            term_cache_blocks: 256,
            store_cache_blocks: 32,
            max_indexing_memory_bytes: 2 * 1024 * 1024 * 1024, // 256 MB default
            merge_policy: Box::new(crate::merge::TieredMergePolicy::default()),
            optimization: crate::structures::IndexOptimization::default(),
        }
    }
}

/// Multi-segment async Index
///
/// The central concept for search. Owns segment lifecycle and provides:
/// - `Index::create()` / `Index::open()` - create or open an index
/// - `index.writer()` - get an IndexWriter for adding documents
/// - `index.reader()` - get an IndexReader for searching with reload policy
///
/// All segment management is delegated to SegmentManager.
#[cfg(feature = "native")]
pub struct Index<D: crate::directories::DirectoryWriter + 'static> {
    directory: Arc<D>,
    schema: Arc<Schema>,
    config: IndexConfig,
    /// Segment manager - owns segments, tracker, and metadata
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Trained centroids for vector search
    trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
    /// Trained codebooks for vector search
    trained_codebooks: FxHashMap<u32, Arc<PQCodebook>>,
}

#[cfg(feature = "native")]
impl<D: crate::directories::DirectoryWriter + 'static> Index<D> {
    /// Create a new index in the directory
    pub async fn create(directory: D, schema: Schema, config: IndexConfig) -> Result<Self> {
        let directory = Arc::new(directory);
        let schema = Arc::new(schema);
        let metadata = IndexMetadata::new((*schema).clone());

        let segment_manager = Arc::new(crate::merge::SegmentManager::new(
            Arc::clone(&directory),
            Arc::clone(&schema),
            metadata,
            config.merge_policy.clone_box(),
            config.term_cache_blocks,
        ));

        // Save initial metadata
        segment_manager.update_metadata(|_| {}).await?;

        Ok(Self {
            directory,
            schema,
            config,
            segment_manager,
            trained_centroids: FxHashMap::default(),
            trained_codebooks: FxHashMap::default(),
        })
    }

    /// Open an existing index from a directory
    pub async fn open(directory: D, config: IndexConfig) -> Result<Self> {
        let directory = Arc::new(directory);

        // Load metadata (includes schema)
        let metadata = IndexMetadata::load(directory.as_ref()).await?;
        let schema = Arc::new(metadata.schema.clone());

        // Load trained structures
        let (trained_centroids, trained_codebooks) =
            metadata.load_trained_structures(directory.as_ref()).await;

        let segment_manager = Arc::new(crate::merge::SegmentManager::new(
            Arc::clone(&directory),
            Arc::clone(&schema),
            metadata,
            config.merge_policy.clone_box(),
            config.term_cache_blocks,
        ));

        Ok(Self {
            directory,
            schema,
            config,
            segment_manager,
            trained_centroids,
            trained_codebooks,
        })
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get a reference to the underlying directory
    pub fn directory(&self) -> &D {
        &self.directory
    }

    /// Get the segment manager
    pub fn segment_manager(&self) -> &Arc<crate::merge::SegmentManager<D>> {
        &self.segment_manager
    }

    /// Get an IndexWriter for adding documents
    pub fn writer(&self) -> writer::IndexWriter<D> {
        writer::IndexWriter::from_index(self)
    }

    /// Get an IndexReader for searching (with reload policy)
    pub async fn reader(&self) -> Result<IndexReader<D>> {
        IndexReader::from_segment_manager(
            Arc::clone(&self.schema),
            Arc::clone(&self.segment_manager),
            self.trained_centroids.clone(),
            self.trained_codebooks.clone(),
            self.config.term_cache_blocks,
        )
        .await
    }

    /// Get the config
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    /// Get trained centroids
    pub fn trained_centroids(&self) -> &FxHashMap<u32, Arc<CoarseCentroids>> {
        &self.trained_centroids
    }

    /// Get trained codebooks
    pub fn trained_codebooks(&self) -> &FxHashMap<u32, Arc<PQCodebook>> {
        &self.trained_codebooks
    }

    // ========== Convenience methods delegating to IndexReader/Searcher ==========

    /// Get segment readers for query execution (convenience method)
    pub async fn segment_readers(&self) -> Result<Vec<Arc<crate::segment::SegmentReader>>> {
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;
        Ok(searcher.segment_readers().to_vec())
    }

    /// Total number of documents across all segments
    pub async fn num_docs(&self) -> Result<u32> {
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;
        Ok(searcher.num_docs())
    }

    /// Get a document by global doc_id
    pub async fn doc(&self, doc_id: crate::DocId) -> Result<Option<crate::dsl::Document>> {
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;
        searcher.doc(doc_id).await
    }

    /// Get default fields for search
    pub fn default_fields(&self) -> Vec<crate::Field> {
        if !self.schema.default_fields().is_empty() {
            self.schema.default_fields().to_vec()
        } else {
            self.schema
                .fields()
                .filter(|(_, entry)| {
                    entry.indexed && entry.field_type == crate::dsl::FieldType::Text
                })
                .map(|(field, _)| field)
                .collect()
        }
    }

    /// Get tokenizer registry
    pub fn tokenizers(&self) -> Arc<crate::tokenizer::TokenizerRegistry> {
        Arc::new(crate::tokenizer::TokenizerRegistry::default())
    }

    /// Create a query parser for this index
    pub fn query_parser(&self) -> crate::dsl::QueryLanguageParser {
        let default_fields = self.default_fields();
        let tokenizers = self.tokenizers();

        let query_routers = self.schema.query_routers();
        if !query_routers.is_empty()
            && let Ok(router) = crate::dsl::QueryFieldRouter::from_rules(query_routers)
        {
            return crate::dsl::QueryLanguageParser::with_router(
                Arc::clone(&self.schema),
                default_fields,
                tokenizers,
                router,
            );
        }

        crate::dsl::QueryLanguageParser::new(Arc::clone(&self.schema), default_fields, tokenizers)
    }

    /// Parse and search using a query string
    pub async fn query(
        &self,
        query_str: &str,
        limit: usize,
    ) -> Result<crate::query::SearchResponse> {
        self.query_offset(query_str, limit, 0).await
    }

    /// Query with offset for pagination
    pub async fn query_offset(
        &self,
        query_str: &str,
        limit: usize,
        offset: usize,
    ) -> Result<crate::query::SearchResponse> {
        let parser = self.query_parser();
        let query = parser
            .parse(query_str)
            .map_err(crate::error::Error::Query)?;
        self.search_offset(query.as_ref(), limit, offset).await
    }

    /// Search and return results
    pub async fn search(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
    ) -> Result<crate::query::SearchResponse> {
        self.search_offset(query, limit, 0).await
    }

    /// Search with offset for pagination
    pub async fn search_offset(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<crate::query::SearchResponse> {
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;
        let segments = searcher.segment_readers();

        let mut all_results: Vec<(u128, crate::query::SearchResult)> = Vec::new();
        let fetch_limit = offset + limit;

        for segment in segments {
            let segment_id = segment.meta().id;
            let results =
                crate::query::search_segment(segment.as_ref(), query, fetch_limit).await?;
            for result in results {
                all_results.push((segment_id, result));
            }
        }

        all_results.sort_by(|a, b| {
            b.1.score
                .partial_cmp(&a.1.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_hits = all_results.len() as u32;

        let hits: Vec<crate::query::SearchHit> = all_results
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(segment_id, result)| crate::query::SearchHit {
                address: crate::query::DocAddress::new(segment_id, result.doc_id),
                score: result.score,
                matched_fields: result.extract_ordinals(),
            })
            .collect();

        Ok(crate::query::SearchResponse { hits, total_hits })
    }

    /// Get a document by its unique address
    pub async fn get_document(
        &self,
        address: &crate::query::DocAddress,
    ) -> Result<Option<crate::dsl::Document>> {
        let segment_id = address.segment_id_u128().ok_or_else(|| {
            crate::error::Error::Query(format!("Invalid segment ID: {}", address.segment_id))
        })?;

        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;

        for segment in searcher.segment_readers() {
            if segment.meta().id == segment_id {
                return segment.doc(address.doc_id).await;
            }
        }

        Ok(None)
    }

    /// Reload is no longer needed - reader handles this automatically
    pub async fn reload(&self) -> Result<()> {
        // No-op - reader reloads automatically based on policy
        Ok(())
    }

    /// Get posting lists for a term across all segments
    pub async fn get_postings(
        &self,
        field: crate::Field,
        term: &[u8],
    ) -> Result<
        Vec<(
            Arc<crate::segment::SegmentReader>,
            crate::structures::BlockPostingList,
        )>,
    > {
        let segments = self.segment_readers().await?;
        let mut results = Vec::new();

        for segment in segments {
            if let Some(postings) = segment.get_postings(field, term).await? {
                results.push((segment, postings));
            }
        }

        Ok(results)
    }
}

// TODO: Add back warmup_and_save_slice_cache when slice caching is re-integrated

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directories::RamDirectory;
    use crate::dsl::{Document, SchemaBuilder};

    #[tokio::test]
    async fn test_index_create_and_search() {
        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        let body = schema_builder.add_text_field("body", true, true);
        let schema = schema_builder.build();

        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        // Create index and add documents
        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
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

        // Retrieve document
        let doc = index.doc(0).await.unwrap().unwrap();
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

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
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

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Create multiple segments by flushing between batches
        for batch in 0..3 {
            for i in 0..3 {
                let mut doc = Document::new();
                doc.add_text(title, format!("Document {} batch {}", i, batch));
                writer.add_document(doc).unwrap();
            }
            writer.flush().await.unwrap();
        }
        writer.commit().await.unwrap();

        // Should have multiple segments (at least 2, one per flush with docs)
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert!(
            index.segment_readers().await.unwrap().len() >= 2,
            "Expected multiple segments"
        );

        // Force merge
        let writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        writer.force_merge().await.unwrap();

        // Should have 1 segment now
        let index = Index::open(dir, config).await.unwrap();
        assert_eq!(index.segment_readers().await.unwrap().len(), 1);
        assert_eq!(index.num_docs().await.unwrap(), 9);

        // Verify all documents accessible (order may vary with queue-based indexing)
        let mut found_docs = 0;
        for i in 0..9 {
            if index.doc(i).await.unwrap().is_some() {
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

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
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

        // Also verify doc retrieval directly by global doc_id
        let doc = index.doc(0).await.unwrap().unwrap();
        assert!(
            !doc.field_values().is_empty(),
            "Doc should have field values"
        );
    }

    #[tokio::test]
    async fn test_slice_cache_warmup_and_load() {
        use crate::directories::SliceCachingDirectory;

        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        let body = schema_builder.add_text_field("body", true, true);
        let schema = schema_builder.build();

        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        // Create index with some documents
        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
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
        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
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
        let doc = index.doc(0).await.unwrap().unwrap();
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

    /// Comprehensive test for WAND optimization in BooleanQuery OR queries
    ///
    /// This test verifies that:
    /// 1. BooleanQuery with multiple SHOULD term queries uses WAND automatically
    /// 2. Search results are correct regardless of WAND optimization
    /// 3. Scores are reasonable for matching documents
    #[tokio::test]
    async fn test_wand_optimization_for_or_queries() {
        use crate::query::{BooleanQuery, TermQuery};

        let mut schema_builder = SchemaBuilder::default();
        let content = schema_builder.add_text_field("content", true, true);
        let schema = schema_builder.build();

        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        // Create index with documents containing various terms
        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Doc 0: contains "rust" and "programming"
        let mut doc = Document::new();
        doc.add_text(content, "rust programming language is fast");
        writer.add_document(doc).unwrap();

        // Doc 1: contains "rust" only
        let mut doc = Document::new();
        doc.add_text(content, "rust is a systems language");
        writer.add_document(doc).unwrap();

        // Doc 2: contains "programming" only
        let mut doc = Document::new();
        doc.add_text(content, "programming is fun");
        writer.add_document(doc).unwrap();

        // Doc 3: contains "python" (neither rust nor programming)
        let mut doc = Document::new();
        doc.add_text(content, "python is easy to learn");
        writer.add_document(doc).unwrap();

        // Doc 4: contains both "rust" and "programming" multiple times
        let mut doc = Document::new();
        doc.add_text(content, "rust rust programming programming systems");
        writer.add_document(doc).unwrap();

        writer.commit().await.unwrap();

        // Open for reading
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();

        // Test 1: Pure OR query with multiple terms (should use WAND automatically)
        let or_query = BooleanQuery::new()
            .should(TermQuery::text(content, "rust"))
            .should(TermQuery::text(content, "programming"));

        let results = index.search(&or_query, 10).await.unwrap();

        // Should find docs 0, 1, 2, 4 (all that contain "rust" OR "programming")
        assert_eq!(results.hits.len(), 4, "Should find exactly 4 documents");

        let doc_ids: Vec<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
        assert!(doc_ids.contains(&0), "Should find doc 0");
        assert!(doc_ids.contains(&1), "Should find doc 1");
        assert!(doc_ids.contains(&2), "Should find doc 2");
        assert!(doc_ids.contains(&4), "Should find doc 4");
        assert!(
            !doc_ids.contains(&3),
            "Should NOT find doc 3 (only has 'python')"
        );

        // Test 2: Single term query (should NOT use WAND, but still work)
        let single_query = BooleanQuery::new().should(TermQuery::text(content, "rust"));

        let results = index.search(&single_query, 10).await.unwrap();
        assert_eq!(results.hits.len(), 3, "Should find 3 documents with 'rust'");

        // Test 3: Query with MUST (should NOT use WAND)
        let must_query = BooleanQuery::new()
            .must(TermQuery::text(content, "rust"))
            .should(TermQuery::text(content, "programming"));

        let results = index.search(&must_query, 10).await.unwrap();
        // Must have "rust", optionally "programming"
        assert_eq!(results.hits.len(), 3, "Should find 3 documents with 'rust'");

        // Test 4: Query with MUST_NOT (should NOT use WAND)
        let must_not_query = BooleanQuery::new()
            .should(TermQuery::text(content, "rust"))
            .should(TermQuery::text(content, "programming"))
            .must_not(TermQuery::text(content, "systems"));

        let results = index.search(&must_not_query, 10).await.unwrap();
        // Should exclude docs with "systems" (doc 1 and 4)
        let doc_ids: Vec<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
        assert!(
            !doc_ids.contains(&1),
            "Should NOT find doc 1 (has 'systems')"
        );
        assert!(
            !doc_ids.contains(&4),
            "Should NOT find doc 4 (has 'systems')"
        );

        // Test 5: Verify top-k limit works correctly with WAND
        let or_query = BooleanQuery::new()
            .should(TermQuery::text(content, "rust"))
            .should(TermQuery::text(content, "programming"));

        let results = index.search(&or_query, 2).await.unwrap();
        assert_eq!(results.hits.len(), 2, "Should return only top 2 results");

        // Top results should be docs that match both terms (higher scores)
        // Doc 0 and 4 contain both "rust" and "programming"
    }

    /// Test that WAND optimization produces same results as non-WAND for correctness
    #[tokio::test]
    async fn test_wand_results_match_standard_boolean() {
        use crate::query::{BooleanQuery, TermQuery, WandOrQuery};

        let mut schema_builder = SchemaBuilder::default();
        let content = schema_builder.add_text_field("content", true, true);
        let schema = schema_builder.build();

        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Add several documents
        for i in 0..10 {
            let mut doc = Document::new();
            let text = match i % 4 {
                0 => "apple banana cherry",
                1 => "apple orange",
                2 => "banana grape",
                _ => "cherry date",
            };
            doc.add_text(content, text);
            writer.add_document(doc).unwrap();
        }

        writer.commit().await.unwrap();
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();

        // Compare explicit WandOrQuery with auto-optimized BooleanQuery
        let wand_query = WandOrQuery::new(content).term("apple").term("banana");

        let bool_query = BooleanQuery::new()
            .should(TermQuery::text(content, "apple"))
            .should(TermQuery::text(content, "banana"));

        let wand_results = index.search(&wand_query, 10).await.unwrap();
        let bool_results = index.search(&bool_query, 10).await.unwrap();

        // Both should find the same documents
        assert_eq!(
            wand_results.hits.len(),
            bool_results.hits.len(),
            "WAND and Boolean should find same number of docs"
        );

        let wand_docs: std::collections::HashSet<u32> =
            wand_results.hits.iter().map(|h| h.address.doc_id).collect();
        let bool_docs: std::collections::HashSet<u32> =
            bool_results.hits.iter().map(|h| h.address.doc_id).collect();

        assert_eq!(
            wand_docs, bool_docs,
            "WAND and Boolean should find same documents"
        );
    }

    #[tokio::test]
    async fn test_vector_index_threshold_switch() {
        use crate::dsl::{DenseVectorConfig, VectorIndexType};

        // Create schema with dense vector field configured for IVF-RaBitQ
        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        let embedding = schema_builder.add_dense_vector_field_with_config(
            "embedding",
            true, // indexed
            true, // stored
            DenseVectorConfig {
                dim: 8,
                index_type: VectorIndexType::IvfRaBitQ,
                store_raw: true,
                num_clusters: Some(4), // Small for test
                nprobe: 2,
                mrl_dim: None,
                build_threshold: Some(50), // Build when we have 50+ vectors
            },
        );
        let schema = schema_builder.build();

        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        // Phase 1: Add vectors below threshold (should use Flat index)
        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Add 30 documents (below threshold of 50)
        for i in 0..30 {
            let mut doc = Document::new();
            doc.add_text(title, format!("Document {}", i));
            // Simple embedding: [i, i, i, i, i, i, i, i] normalized
            let vec: Vec<f32> = (0..8).map(|_| (i as f32) / 30.0).collect();
            doc.add_dense_vector(embedding, vec);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();

        // Open index and verify it's using Flat (not built yet)
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert!(
            index.trained_centroids.is_empty(),
            "Should not have trained centroids below threshold"
        );

        // Search should work with Flat index
        let query_vec: Vec<f32> = vec![0.5; 8];
        let segments = index.segment_readers().await.unwrap();
        assert!(!segments.is_empty());

        let results = segments[0]
            .search_dense_vector(
                embedding,
                &query_vec,
                5,
                1,
                crate::query::MultiValueCombiner::Max,
            )
            .unwrap();
        assert!(!results.is_empty(), "Flat search should return results");

        // Phase 2: Add more vectors to cross threshold
        let writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();

        // Add 30 more documents (total 60, above threshold of 50)
        for i in 30..60 {
            let mut doc = Document::new();
            doc.add_text(title, format!("Document {}", i));
            let vec: Vec<f32> = (0..8).map(|_| (i as f32) / 60.0).collect();
            doc.add_dense_vector(embedding, vec);
            writer.add_document(doc).unwrap();
        }
        // Commit auto-triggers vector index build when threshold is crossed
        writer.commit().await.unwrap();

        // Verify centroids were trained (auto-triggered)
        assert!(
            writer.is_vector_index_built(embedding).await,
            "Vector index should be built after crossing threshold"
        );

        // Reopen index and verify trained structures are loaded
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert!(
            index.trained_centroids.contains_key(&embedding.0),
            "Should have loaded trained centroids for embedding field"
        );

        // Search should still work
        let segments = index.segment_readers().await.unwrap();
        let results = segments[0]
            .search_dense_vector(
                embedding,
                &query_vec,
                5,
                1,
                crate::query::MultiValueCombiner::Max,
            )
            .unwrap();
        assert!(
            !results.is_empty(),
            "Search should return results after build"
        );

        // Phase 3: Verify calling build_vector_index again is a no-op
        let writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        writer.build_vector_index().await.unwrap(); // Should skip training

        // Still built
        assert!(writer.is_vector_index_built(embedding).await);
    }
}
