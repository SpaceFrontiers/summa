//! Index - multi-segment async search index
//!
//! The `Index` is the central concept that provides:
//! - `Index::create()` / `Index::open()` - create or open an index
//! - `index.writer()` - get an IndexWriter for adding documents
//! - `index.reader()` - get an IndexReader for searching (with reload policy)
//!
//! The Index owns the SegmentManager which handles segment lifecycle and tracking.

#[cfg(feature = "native")]
use crate::dsl::Schema;
#[cfg(feature = "native")]
use crate::error::Result;
#[cfg(feature = "native")]
use std::sync::Arc;

mod searcher;
pub use searcher::Searcher;

#[cfg(feature = "native")]
mod reader;
#[cfg(feature = "native")]
mod vector_builder;
#[cfg(feature = "native")]
mod writer;
#[cfg(feature = "native")]
pub use reader::IndexReader;
#[cfg(feature = "native")]
pub use writer::{IndexWriter, PreparedCommit};

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
    /// Reload interval in milliseconds for IndexReader (how often to check for new segments)
    pub reload_interval_ms: u64,
    /// Maximum number of concurrent background merges (default: 4)
    pub max_concurrent_merges: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        #[cfg(feature = "native")]
        let indexing_threads = crate::default_indexing_threads();
        #[cfg(not(feature = "native"))]
        let indexing_threads = 1;

        #[cfg(feature = "native")]
        let compression_threads = crate::default_compression_threads();
        #[cfg(not(feature = "native"))]
        let compression_threads = 1;

        Self {
            num_threads: indexing_threads,
            num_indexing_threads: 1, // Increase to 2+ for production to avoid stalls during segment build
            num_compression_threads: compression_threads,
            term_cache_blocks: 256,
            store_cache_blocks: 32,
            max_indexing_memory_bytes: 256 * 1024 * 1024, // 256 MB default
            merge_policy: Box::new(crate::merge::TieredMergePolicy::default()),
            optimization: crate::structures::IndexOptimization::default(),
            reload_interval_ms: 1000, // 1 second default
            max_concurrent_merges: 4,
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
    /// Segment manager - owns segments, tracker, metadata, and trained structures
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Cached reader (created lazily, reused across calls)
    cached_reader: tokio::sync::OnceCell<IndexReader<D>>,
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
            config.max_concurrent_merges,
        ));

        // Save initial metadata
        segment_manager.update_metadata(|_| {}).await?;

        Ok(Self {
            directory,
            schema,
            config,
            segment_manager,
            cached_reader: tokio::sync::OnceCell::new(),
        })
    }

    /// Open an existing index from a directory
    pub async fn open(directory: D, config: IndexConfig) -> Result<Self> {
        let directory = Arc::new(directory);

        // Load metadata (includes schema)
        let metadata = IndexMetadata::load(directory.as_ref()).await?;
        let schema = Arc::new(metadata.schema.clone());

        let segment_manager = Arc::new(crate::merge::SegmentManager::new(
            Arc::clone(&directory),
            Arc::clone(&schema),
            metadata,
            config.merge_policy.clone_box(),
            config.term_cache_blocks,
            config.max_concurrent_merges,
        ));

        // Load trained structures into SegmentManager's ArcSwap
        segment_manager.load_and_publish_trained().await;

        Ok(Self {
            directory,
            schema,
            config,
            segment_manager,
            cached_reader: tokio::sync::OnceCell::new(),
        })
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get the schema as an Arc reference (avoids clone when Arc is needed)
    pub fn schema_arc(&self) -> &Arc<Schema> {
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

    /// Get an IndexReader for searching (with reload policy)
    ///
    /// The reader is cached and reused across calls. The reader's internal
    /// searcher will reload segments based on its reload interval (configurable via IndexConfig).
    pub async fn reader(&self) -> Result<&IndexReader<D>> {
        self.cached_reader
            .get_or_try_init(|| async {
                IndexReader::from_segment_manager(
                    Arc::clone(&self.schema),
                    Arc::clone(&self.segment_manager),
                    self.config.term_cache_blocks,
                    self.config.reload_interval_ms,
                )
                .await
            })
            .await
    }

    /// Get the config
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

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

        let fetch_limit = offset + limit;

        let futures: Vec<_> = segments
            .iter()
            .map(|segment| {
                let sid = segment.meta().id;
                async move {
                    let results =
                        crate::query::search_segment(segment.as_ref(), query, fetch_limit).await?;
                    Ok::<_, crate::error::Error>(
                        results
                            .into_iter()
                            .map(move |r| (sid, r))
                            .collect::<Vec<_>>(),
                    )
                }
            })
            .collect();

        let batches = futures::future::try_join_all(futures).await?;
        let mut all_results: Vec<(u128, crate::query::SearchResult)> =
            Vec::with_capacity(batches.iter().map(|b| b.len()).sum());
        for batch in batches {
            all_results.extend(batch);
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
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;
        searcher.get_document(address).await
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

/// Native-only methods for Index
#[cfg(feature = "native")]
impl<D: crate::directories::DirectoryWriter + 'static> Index<D> {
    /// Get an IndexWriter for adding documents
    pub fn writer(&self) -> writer::IndexWriter<D> {
        writer::IndexWriter::from_index(self)
    }
}

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
        let doc = searcher.doc(0).await.unwrap().unwrap();
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
        let mut found_docs = 0;
        for i in 0..9 {
            if searcher.doc(i).await.unwrap().is_some() {
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
        let doc = searcher.doc(0).await.unwrap().unwrap();
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
        let doc = searcher.doc(0).await.unwrap().unwrap();
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
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
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

        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
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
        use crate::dsl::{DenseVectorConfig, DenseVectorQuantization, VectorIndexType};

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
                quantization: DenseVectorQuantization::F32,
                num_clusters: Some(4), // Small for test
                nprobe: 2,
                build_threshold: Some(50), // Build when we have 50+ vectors
            },
        );
        let schema = schema_builder.build();

        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        // Phase 1: Add vectors below threshold (should use Flat index)
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
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
            index.segment_manager.trained().is_none(),
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
                0,
                1,
                crate::query::MultiValueCombiner::Max,
            )
            .await
            .unwrap();
        assert!(!results.is_empty(), "Flat search should return results");

        // Phase 2: Add more vectors to cross threshold
        let mut writer = IndexWriter::open(dir.clone(), config.clone())
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
        writer.commit().await.unwrap();

        // Manually trigger vector index build (no longer auto-triggered by commit)
        writer.build_vector_index().await.unwrap();

        // Reopen index and verify trained structures are loaded
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert!(
            index.segment_manager.trained().is_some(),
            "Should have loaded trained centroids for embedding field"
        );

        // Search should still work
        let segments = index.segment_readers().await.unwrap();
        let results = segments[0]
            .search_dense_vector(
                embedding,
                &query_vec,
                5,
                0,
                1,
                crate::query::MultiValueCombiner::Max,
            )
            .await
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

        // Still built (trained structures present in ArcSwap)
        assert!(writer.segment_manager.trained().is_some());
    }

    /// Multi-round merge: flush many small segments, merge, add more, merge again.
    /// Verifies search correctness (term + phrase queries) through multiple merge rounds.
    #[tokio::test]
    async fn test_multi_round_merge_with_search() {
        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        let body = schema_builder.add_text_field("body", true, true);
        let schema = schema_builder.build();

        let dir = RamDirectory::new();
        let config = IndexConfig {
            max_indexing_memory_bytes: 512,
            ..Default::default()
        };

        // --- Round 1: 5 segments × 10 docs = 50 docs ---
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        for batch in 0..5 {
            for i in 0..10 {
                let mut doc = Document::new();
                doc.add_text(
                    title,
                    format!("alpha bravo charlie batch{} doc{}", batch, i),
                );
                doc.add_text(
                    body,
                    format!("the quick brown fox jumps over the lazy dog number {}", i),
                );
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
        }

        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        let pre_merge_segments = index.segment_readers().await.unwrap().len();
        assert!(
            pre_merge_segments >= 3,
            "Expected >=3 segments, got {}",
            pre_merge_segments
        );
        assert_eq!(index.num_docs().await.unwrap(), 50);

        // Search before merge
        let results = index.query("alpha", 100).await.unwrap();
        assert_eq!(results.hits.len(), 50, "all 50 docs should match 'alpha'");

        let results = index.query("fox", 100).await.unwrap();
        assert_eq!(results.hits.len(), 50, "all 50 docs should match 'fox'");

        // --- Merge round 1 ---
        let mut writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        writer.force_merge().await.unwrap();

        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert_eq!(index.segment_readers().await.unwrap().len(), 1);
        assert_eq!(index.num_docs().await.unwrap(), 50);

        // Search after first merge
        let results = index.query("alpha", 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            50,
            "all 50 docs should match 'alpha' after merge 1"
        );

        let results = index.query("fox", 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            50,
            "all 50 docs should match 'fox' after merge 1"
        );

        // Verify all docs retrievable
        let reader1 = index.reader().await.unwrap();
        let searcher1 = reader1.searcher().await.unwrap();
        for i in 0..50 {
            let doc = searcher1.doc(i).await.unwrap();
            assert!(doc.is_some(), "doc {} should exist after merge 1", i);
        }

        // --- Round 2: add 30 more docs in 3 segments ---
        let mut writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        for batch in 0..3 {
            for i in 0..10 {
                let mut doc = Document::new();
                doc.add_text(
                    title,
                    format!("delta echo foxtrot round2_batch{} doc{}", batch, i),
                );
                doc.add_text(
                    body,
                    format!("the quick brown fox jumps again number {}", i),
                );
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
        }

        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert_eq!(index.num_docs().await.unwrap(), 80);
        assert!(
            index.segment_readers().await.unwrap().len() >= 2,
            "Should have >=2 segments after round 2 ingestion"
        );

        // Search spans both old merged segment and new segments
        let results = index.query("fox", 100).await.unwrap();
        assert_eq!(results.hits.len(), 80, "all 80 docs should match 'fox'");

        let results = index.query("alpha", 100).await.unwrap();
        assert_eq!(results.hits.len(), 50, "only round 1 docs match 'alpha'");

        let results = index.query("delta", 100).await.unwrap();
        assert_eq!(results.hits.len(), 30, "only round 2 docs match 'delta'");

        // --- Merge round 2 ---
        let mut writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        writer.force_merge().await.unwrap();

        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert_eq!(index.segment_readers().await.unwrap().len(), 1);
        assert_eq!(index.num_docs().await.unwrap(), 80);

        // All searches still correct after second merge
        let results = index.query("fox", 100).await.unwrap();
        assert_eq!(results.hits.len(), 80, "all 80 docs after merge 2");

        let results = index.query("alpha", 100).await.unwrap();
        assert_eq!(results.hits.len(), 50, "round 1 docs after merge 2");

        let results = index.query("delta", 100).await.unwrap();
        assert_eq!(results.hits.len(), 30, "round 2 docs after merge 2");

        // Verify all 80 docs retrievable
        let reader2 = index.reader().await.unwrap();
        let searcher2 = reader2.searcher().await.unwrap();
        for i in 0..80 {
            let doc = searcher2.doc(i).await.unwrap();
            assert!(doc.is_some(), "doc {} should exist after merge 2", i);
        }
    }

    /// Large-scale merge: many segments with overlapping terms, verifying
    /// BM25 scoring and doc retrieval after merge.
    #[tokio::test]
    async fn test_large_scale_merge_correctness() {
        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        let schema = schema_builder.build();

        let dir = RamDirectory::new();
        let config = IndexConfig {
            max_indexing_memory_bytes: 512,
            ..Default::default()
        };

        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // 8 batches × 25 docs = 200 docs total
        // Terms: "common" appears in all, "unique_N" appears in batch N only
        let total_docs = 200u32;
        for batch in 0..8 {
            for i in 0..25 {
                let mut doc = Document::new();
                doc.add_text(
                    title,
                    format!("common shared term unique_{} item{}", batch, i),
                );
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
        }

        // Verify pre-merge
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert_eq!(index.num_docs().await.unwrap(), total_docs);

        let results = index.query("common", 300).await.unwrap();
        assert_eq!(
            results.hits.len(),
            total_docs as usize,
            "all docs should match 'common'"
        );

        // Each unique_N matches exactly 25 docs
        for batch in 0..8 {
            let q = format!("unique_{}", batch);
            let results = index.query(&q, 100).await.unwrap();
            assert_eq!(results.hits.len(), 25, "'{}' should match 25 docs", q);
        }

        // Force merge
        let mut writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        writer.force_merge().await.unwrap();

        // Verify post-merge: single segment, same results
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert_eq!(index.segment_readers().await.unwrap().len(), 1);
        assert_eq!(index.num_docs().await.unwrap(), total_docs);

        let results = index.query("common", 300).await.unwrap();
        assert_eq!(results.hits.len(), total_docs as usize);

        for batch in 0..8 {
            let q = format!("unique_{}", batch);
            let results = index.query(&q, 100).await.unwrap();
            assert_eq!(results.hits.len(), 25, "'{}' after merge", q);
        }

        // Verify doc retrieval for every doc
        let reader = index.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        for i in 0..total_docs {
            let doc = searcher.doc(i).await.unwrap();
            assert!(doc.is_some(), "doc {} missing after merge", i);
        }
    }

    /// Test that auto-merge is triggered by the merge policy during commit,
    /// without calling force_merge. Uses MmapDirectory and higher parallelism
    /// to reproduce production conditions.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_auto_merge_triggered() {
        use crate::directories::MmapDirectory;
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::new(tmp_dir.path());

        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        let body = schema_builder.add_text_field("body", true, true);
        let schema = schema_builder.build();

        // Aggressive policy: merge when 3 segments in same tier
        let config = IndexConfig {
            max_indexing_memory_bytes: 4096,
            num_indexing_threads: 4,
            merge_policy: Box::new(crate::merge::TieredMergePolicy::aggressive()),
            ..Default::default()
        };

        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Create 12 segments with ~50 docs each (4x the aggressive threshold of 3)
        for batch in 0..12 {
            for i in 0..50 {
                let mut doc = Document::new();
                doc.add_text(title, format!("document_{} batch_{} alpha bravo", i, batch));
                doc.add_text(
                    body,
                    format!(
                        "the quick brown fox jumps over lazy dog number {} round {}",
                        i, batch
                    ),
                );
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
        }

        let pre_merge = writer.segment_manager.get_segment_ids().await.len();

        // wait_for_merging_thread waits for the single in-flight merge. After it completes,
        // re-evaluate since segments accumulated while the merge was running.
        writer.wait_for_merging_thread().await;
        writer.maybe_merge().await;
        writer.wait_for_merging_thread().await;

        // After commit + auto-merge, segment count should be reduced
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        let segment_count = index.segment_readers().await.unwrap().len();
        eprintln!(
            "Segments: {} before merge, {} after auto-merge",
            pre_merge, segment_count
        );
        assert!(
            segment_count < pre_merge,
            "Expected auto-merge to reduce segments from {}, got {}",
            pre_merge,
            segment_count
        );
    }

    /// Regression test: commit with dense vector fields + aggressive merge policy.
    /// Exercises the race where background merge deletes segment files while
    /// maybe_build_vector_index → collect_vectors_for_training tries to open them.
    /// Before the fix, this would fail with "IO error: No such file or directory".
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_commit_with_vectors_and_background_merge() {
        use crate::directories::MmapDirectory;
        use crate::dsl::DenseVectorConfig;

        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::new(tmp_dir.path());

        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        // RaBitQ with very low build_threshold so vector index building triggers during commit
        let vec_config = DenseVectorConfig::new(8).with_build_threshold(10);
        let embedding =
            schema_builder.add_dense_vector_field_with_config("embedding", true, true, vec_config);
        let schema = schema_builder.build();

        // Aggressive merge: triggers background merges at 3 segments per tier
        let config = IndexConfig {
            max_indexing_memory_bytes: 4096,
            num_indexing_threads: 4,
            merge_policy: Box::new(crate::merge::TieredMergePolicy::aggressive()),
            ..Default::default()
        };

        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Create 12 segments with vectors — enough to trigger both
        // background merges (aggressive policy) and vector index building (threshold=10)
        for batch in 0..12 {
            for i in 0..5 {
                let mut doc = Document::new();
                doc.add_text(title, format!("doc_{}_batch_{}", i, batch));
                // 8-dim random vector
                let vec: Vec<f32> = (0..8).map(|j| (i * 8 + j + batch) as f32 * 0.1).collect();
                doc.add_dense_vector(embedding, vec);
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
        }
        writer.wait_for_merging_thread().await;

        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        let num_docs = index.num_docs().await.unwrap();
        assert_eq!(num_docs, 60, "Expected 60 docs, got {}", num_docs);
    }

    /// Stress test: force_merge with many segments (iterative batching).
    /// Verifies that merging 50 segments doesn't OOM or exhaust file descriptors.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_force_merge_many_segments() {
        use crate::directories::MmapDirectory;
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::new(tmp_dir.path());

        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        let schema = schema_builder.build();

        let config = IndexConfig {
            max_indexing_memory_bytes: 512,
            ..Default::default()
        };

        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Create 50 tiny segments
        for batch in 0..50 {
            for i in 0..3 {
                let mut doc = Document::new();
                doc.add_text(title, format!("term_{} batch_{}", i, batch));
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
        }
        // Wait for background merges before reading segment count
        writer.wait_for_merging_thread().await;

        let seg_ids = writer.segment_manager.get_segment_ids().await;
        let pre = seg_ids.len();
        eprintln!("Segments before force_merge: {}", pre);
        assert!(pre >= 2, "Expected multiple segments, got {}", pre);

        // Force merge all into one — should iterate in batches, not OOM
        writer.force_merge().await.unwrap();

        let index2 = Index::open(dir, config).await.unwrap();
        let post = index2.segment_readers().await.unwrap().len();
        eprintln!("Segments after force_merge: {}", post);
        assert_eq!(post, 1);
        assert_eq!(index2.num_docs().await.unwrap(), 150);
    }

    /// Test that background merges produce correct generation metadata.
    /// Creates many segments with aggressive policy, commits, waits for merges,
    /// and verifies that merged segments have generation >= 1 with correct ancestors.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_background_merge_generation() {
        use crate::directories::MmapDirectory;
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::new(tmp_dir.path());

        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        let schema = schema_builder.build();

        let config = IndexConfig {
            max_indexing_memory_bytes: 4096,
            num_indexing_threads: 2,
            merge_policy: Box::new(crate::merge::TieredMergePolicy::aggressive()),
            ..Default::default()
        };

        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Create 15 small segments — enough for aggressive policy to trigger merges
        for batch in 0..15 {
            for i in 0..5 {
                let mut doc = Document::new();
                doc.add_text(title, format!("doc_{}_batch_{}", i, batch));
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
        }
        writer.wait_for_merging_thread().await;

        // Read metadata and verify generation tracking
        let metas = writer
            .segment_manager
            .read_metadata(|m| m.segment_metas.clone())
            .await;

        let max_gen = metas.values().map(|m| m.generation).max().unwrap_or(0);
        eprintln!(
            "Segments after merge: {}, max generation: {}",
            metas.len(),
            max_gen
        );

        // Background merges should have produced at least one merged segment (gen >= 1)
        assert!(
            max_gen >= 1,
            "Expected at least one merged segment (gen >= 1), got max_gen={}",
            max_gen
        );

        // Every merged segment (gen > 0) must have non-empty ancestors
        for (id, info) in &metas {
            if info.generation > 0 {
                assert!(
                    !info.ancestors.is_empty(),
                    "Segment {} has gen={} but no ancestors",
                    id,
                    info.generation
                );
            } else {
                assert!(
                    info.ancestors.is_empty(),
                    "Fresh segment {} has gen=0 but has ancestors",
                    id
                );
            }
        }
    }

    /// Test that merging preserves every single document.
    /// Indexes 1000+ unique documents across many segments, force-merges,
    /// and verifies exact doc count and that every unique term is searchable.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_merge_preserves_all_documents() {
        use crate::directories::MmapDirectory;
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::new(tmp_dir.path());

        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        let schema = schema_builder.build();

        let config = IndexConfig {
            max_indexing_memory_bytes: 4096,
            ..Default::default()
        };

        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        let total_docs = 1200;
        let docs_per_batch = 60;
        let batches = total_docs / docs_per_batch;

        // Each doc has a unique term "uid_N" for verification
        for batch in 0..batches {
            for i in 0..docs_per_batch {
                let doc_num = batch * docs_per_batch + i;
                let mut doc = Document::new();
                doc.add_text(
                    title,
                    format!("uid_{} common_term batch_{}", doc_num, batch),
                );
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
        }

        let pre_segments = writer.segment_manager.get_segment_ids().await.len();
        assert!(
            pre_segments >= 2,
            "Need multiple segments, got {}",
            pre_segments
        );

        // Force merge to single segment
        writer.force_merge().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();
        assert_eq!(index.segment_readers().await.unwrap().len(), 1);
        assert_eq!(
            index.num_docs().await.unwrap(),
            total_docs as u32,
            "Doc count mismatch after force_merge"
        );

        // Verify every unique document is searchable
        let results = index.query("common_term", total_docs + 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            total_docs,
            "common_term should match all docs"
        );

        // Spot-check unique IDs across the range
        for check in [0, 1, total_docs / 2, total_docs - 1] {
            let q = format!("uid_{}", check);
            let results = index.query(&q, 10).await.unwrap();
            assert_eq!(results.hits.len(), 1, "'{}' should match exactly 1 doc", q);
        }
    }

    /// Multi-round commit+merge: verify doc count grows correctly
    /// and no documents are lost across multiple merge cycles.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_multi_round_merge_doc_integrity() {
        use crate::directories::MmapDirectory;
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::new(tmp_dir.path());

        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        let schema = schema_builder.build();

        let config = IndexConfig {
            max_indexing_memory_bytes: 4096,
            num_indexing_threads: 2,
            merge_policy: Box::new(crate::merge::TieredMergePolicy::aggressive()),
            ..Default::default()
        };

        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        let mut expected_total = 0u64;

        // 4 rounds of: add docs → commit → wait for merges → verify count
        for round in 0..4 {
            let docs_this_round = 50 + round * 25; // 50, 75, 100, 125
            for batch in 0..5 {
                for i in 0..docs_this_round / 5 {
                    let mut doc = Document::new();
                    doc.add_text(
                        title,
                        format!("round_{}_batch_{}_doc_{} searchable", round, batch, i),
                    );
                    writer.add_document(doc).unwrap();
                }
                writer.commit().await.unwrap();
            }
            writer.wait_for_merging_thread().await;

            expected_total += docs_this_round as u64;

            let actual = writer
                .segment_manager
                .read_metadata(|m| {
                    m.segment_metas
                        .values()
                        .map(|s| s.num_docs as u64)
                        .sum::<u64>()
                })
                .await;

            assert_eq!(
                actual, expected_total,
                "Round {}: expected {} docs, metadata reports {}",
                round, expected_total, actual
            );
        }

        // Final verify: open fresh and query
        let index = Index::open(dir, config).await.unwrap();
        assert_eq!(index.num_docs().await.unwrap(), expected_total as u32);

        let results = index
            .query("searchable", expected_total as usize + 100)
            .await
            .unwrap();
        assert_eq!(
            results.hits.len(),
            expected_total as usize,
            "All docs should match 'searchable'"
        );

        // Check generation grew across rounds
        let metas = index
            .segment_manager()
            .read_metadata(|m| m.segment_metas.clone())
            .await;
        let max_gen = metas.values().map(|m| m.generation).max().unwrap_or(0);
        eprintln!(
            "Final: {} segments, {} docs, max generation={}",
            metas.len(),
            expected_total,
            max_gen
        );
        assert!(
            max_gen >= 1,
            "Multiple merge rounds should produce gen >= 1"
        );
    }

    /// Sustained indexing: verify segment count stays O(logN) bounded.
    ///
    /// Indexes many small batches with aggressive merge policy and checks that
    /// the segment count never grows unbounded. With tiered merging the count
    /// should stay roughly O(segments_per_tier * num_tiers) ≈ O(log(N)).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_segment_count_bounded_during_sustained_indexing() {
        use crate::directories::MmapDirectory;
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::new(tmp_dir.path());

        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, false);
        let schema = schema_builder.build();

        let policy = crate::merge::TieredMergePolicy {
            segments_per_tier: 3,
            max_merge_at_once: 5,
            tier_factor: 10.0,
            tier_floor: 50,
            max_merged_docs: 1_000_000,
        };

        let config = IndexConfig {
            max_indexing_memory_bytes: 4096, // tiny budget → frequent flushes
            num_indexing_threads: 1,
            merge_policy: Box::new(policy),
            max_concurrent_merges: 4,
            ..Default::default()
        };

        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        let num_commits = 40;
        let docs_per_commit = 30;
        let total_docs = num_commits * docs_per_commit;
        let mut max_segments_seen = 0usize;

        for commit_idx in 0..num_commits {
            for i in 0..docs_per_commit {
                let mut doc = Document::new();
                doc.add_text(
                    title,
                    format!("doc_{} text", commit_idx * docs_per_commit + i),
                );
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();

            // Give background merges a moment to run
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;

            let seg_count = writer.segment_manager.get_segment_ids().await.len();
            max_segments_seen = max_segments_seen.max(seg_count);
        }

        // Wait for all merges to finish
        writer.wait_for_all_merges().await;

        let final_segments = writer.segment_manager.get_segment_ids().await.len();
        let final_docs: u64 = writer
            .segment_manager
            .read_metadata(|m| {
                m.segment_metas
                    .values()
                    .map(|s| s.num_docs as u64)
                    .sum::<u64>()
            })
            .await;

        eprintln!(
            "Sustained indexing: {} commits, {} total docs, final segments={}, max segments seen={}",
            num_commits, total_docs, final_segments, max_segments_seen
        );

        // With 1200 docs and segments_per_tier=3, tier_floor=50:
        // tier 0: ≤50 docs, tier 1: 50-500, tier 2: 500-5000
        // We should have at most ~3 segments per tier * ~3 tiers ≈ 9-12 segments at peak.
        // The key invariant: segment count must NOT grow linearly with commits.
        // 40 commits should NOT produce 40 segments.
        let max_allowed = num_commits / 2; // generous: at most half the commits as segments
        assert!(
            max_segments_seen <= max_allowed,
            "Segment count grew too fast: max seen {} > allowed {} (out of {} commits). \
             Merging is not keeping up.",
            max_segments_seen,
            max_allowed,
            num_commits
        );

        // After all merges complete, should be well under the limit
        assert!(
            final_segments <= 10,
            "After all merges, expected ≤10 segments, got {}",
            final_segments
        );

        // No data loss
        assert_eq!(
            final_docs, total_docs as u64,
            "Expected {} docs, metadata reports {}",
            total_docs, final_docs
        );
    }
}
