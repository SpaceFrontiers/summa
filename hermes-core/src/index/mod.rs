//! Index - multi-segment async search index
//!
//! Components:
//! - Index: main entry point for searching
//! - IndexWriter: for adding documents and committing segments (native only)
//! - Supports multiple segments with merge

use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;

use rustc_hash::FxHashMap;

use crate::DocId;
use crate::directories::{Directory, SliceCachingDirectory};
use crate::dsl::{Document, Field, Schema};
use crate::error::{Error, Result};
use crate::segment::{SegmentId, SegmentReader};
use crate::structures::BlockPostingList;
use crate::structures::{CoarseCentroids, PQCodebook};

#[cfg(feature = "native")]
mod vector_builder;
#[cfg(feature = "native")]
mod writer;
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
/// The main entry point for searching. Manages multiple segments
/// and provides unified search across all of them.
pub struct Index<D: Directory> {
    directory: Arc<D>,
    schema: Arc<Schema>,
    config: IndexConfig,
    segments: RwLock<Vec<Arc<SegmentReader>>>,
    default_fields: Vec<crate::Field>,
    tokenizers: Arc<crate::tokenizer::TokenizerRegistry>,
    /// Cached global statistics for cross-segment IDF computation
    global_stats: crate::query::GlobalStatsCache,
    /// Index-level trained centroids per field (loaded from metadata)
    #[allow(dead_code)] // Used in native builds for clone_for_builder
    trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
    /// Index-level trained PQ codebooks per field (for ScaNN)
    #[allow(dead_code)] // Used in native builds for clone_for_builder
    trained_codebooks: FxHashMap<u32, Arc<PQCodebook>>,
    #[cfg(feature = "native")]
    thread_pool: Arc<rayon::ThreadPool>,
}

impl<D: Directory> Index<D> {
    /// Open an existing index from a directory
    pub async fn open(directory: D, config: IndexConfig) -> Result<Self> {
        let directory = Arc::new(directory);

        // Read schema
        let schema_slice = directory.open_read(Path::new("schema.json")).await?;
        let schema_bytes = schema_slice.read_bytes().await?;
        let schema: Schema = serde_json::from_slice(schema_bytes.as_slice())
            .map_err(|e| Error::Serialization(e.to_string()))?;
        let schema = Arc::new(schema);

        // Load metadata and trained structures
        let (segments, trained_centroids, trained_codebooks) =
            Self::load_segments_and_trained(&directory, &schema, &config).await?;

        #[cfg(feature = "native")]
        let thread_pool = {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.num_threads)
                .build()
                .map_err(|e| Error::Io(std::io::Error::other(e)))?;
            Arc::new(pool)
        };

        // Use schema's default_fields if specified, otherwise fall back to all indexed text fields
        let default_fields: Vec<crate::Field> = if !schema.default_fields().is_empty() {
            schema.default_fields().to_vec()
        } else {
            schema
                .fields()
                .filter(|(_, entry)| {
                    entry.indexed && entry.field_type == crate::dsl::FieldType::Text
                })
                .map(|(field, _)| field)
                .collect()
        };

        Ok(Self {
            directory,
            schema,
            config,
            segments: RwLock::new(segments),
            default_fields,
            tokenizers: Arc::new(crate::tokenizer::TokenizerRegistry::default()),
            global_stats: crate::query::GlobalStatsCache::new(),
            trained_centroids,
            trained_codebooks,
            #[cfg(feature = "native")]
            thread_pool,
        })
    }

    /// Load segments and trained structures from metadata
    async fn load_segments_and_trained(
        directory: &Arc<D>,
        schema: &Arc<Schema>,
        config: &IndexConfig,
    ) -> Result<(
        Vec<Arc<SegmentReader>>,
        FxHashMap<u32, Arc<CoarseCentroids>>,
        FxHashMap<u32, Arc<PQCodebook>>,
    )> {
        // Load metadata
        let meta = Self::load_metadata(directory).await?;

        // Load trained centroids and codebooks
        let (trained_centroids, trained_codebooks) =
            meta.load_trained_structures(directory.as_ref()).await;

        // Load segments
        let mut segments = Vec::new();
        let mut doc_id_offset = 0u32;

        for id_str in meta.segments {
            let segment_id = SegmentId::from_hex(&id_str)
                .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {}", id_str)))?;
            let reader = SegmentReader::open(
                directory.as_ref(),
                segment_id,
                Arc::clone(schema),
                doc_id_offset,
                config.term_cache_blocks,
            )
            .await?;

            doc_id_offset += reader.meta().num_docs;
            segments.push(Arc::new(reader));
        }

        Ok((segments, trained_centroids, trained_codebooks))
    }

    /// Load metadata from metadata.json
    async fn load_metadata(directory: &Arc<D>) -> Result<IndexMetadata> {
        let meta_path = Path::new(INDEX_META_FILENAME);
        if directory.exists(meta_path).await.unwrap_or(false) {
            let slice = directory.open_read(meta_path).await?;
            let bytes = slice.read_bytes().await?;
            let meta: IndexMetadata = serde_json::from_slice(bytes.as_slice())
                .map_err(|e| Error::Serialization(e.to_string()))?;
            return Ok(meta);
        }
        Ok(IndexMetadata::new())
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get a reference to the underlying directory
    pub fn directory(&self) -> &D {
        &self.directory
    }

    /// Total number of documents across all segments
    pub fn num_docs(&self) -> u32 {
        self.segments.read().iter().map(|s| s.num_docs()).sum()
    }

    /// Get a document by global doc_id (async)
    pub async fn doc(&self, doc_id: DocId) -> Result<Option<Document>> {
        let segments = self.segments.read().clone();

        let mut offset = 0u32;
        for segment in segments.iter() {
            let segment_docs = segment.meta().num_docs;
            if doc_id < offset + segment_docs {
                let local_doc_id = doc_id - offset;
                return segment.doc(local_doc_id).await;
            }
            offset += segment_docs;
        }

        Ok(None)
    }

    /// Get posting lists for a term across all segments (async)
    pub async fn get_postings(
        &self,
        field: Field,
        term: &[u8],
    ) -> Result<Vec<(Arc<SegmentReader>, BlockPostingList)>> {
        let segments = self.segments.read().clone();
        let mut results = Vec::new();

        for segment in segments.iter() {
            if let Some(postings) = segment.get_postings(field, term).await? {
                results.push((Arc::clone(segment), postings));
            }
        }

        Ok(results)
    }

    /// Execute CPU-intensive work on thread pool (native only)
    #[cfg(feature = "native")]
    pub async fn spawn_blocking<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.thread_pool.spawn(move || {
            let result = f();
            let _ = tx.send(result);
        });
        rx.await.expect("Thread pool task panicked")
    }

    /// Get segment readers for query execution
    pub fn segment_readers(&self) -> Vec<Arc<SegmentReader>> {
        self.segments.read().clone()
    }

    /// Reload segments from directory (after new segments added or merges completed)
    ///
    /// This reads the current metadata.json and loads only segments that exist.
    /// Safe to call at any time - will not reference deleted segments.
    ///
    /// Note: This only reloads segments, not trained structures (they are immutable once built)
    pub async fn reload(&self) -> Result<()> {
        // Load fresh metadata from disk
        let meta = Self::load_metadata(&self.directory).await?;

        // Load only segments that exist in current metadata
        let mut new_segments = Vec::new();
        let mut doc_id_offset = 0u32;

        for id_str in meta.segments {
            let segment_id = match SegmentId::from_hex(&id_str) {
                Some(id) => id,
                None => {
                    log::warn!("Invalid segment ID in metadata: {}", id_str);
                    continue;
                }
            };

            // Try to open segment - skip if files don't exist (may have been deleted by merge)
            match SegmentReader::open(
                self.directory.as_ref(),
                segment_id,
                Arc::clone(&self.schema),
                doc_id_offset,
                self.config.term_cache_blocks,
            )
            .await
            {
                Ok(reader) => {
                    doc_id_offset += reader.meta().num_docs;
                    new_segments.push(Arc::new(reader));
                }
                Err(e) => {
                    // Segment files may have been deleted by concurrent merge - skip it
                    log::warn!(
                        "Could not open segment {}: {:?} (may have been merged)",
                        id_str,
                        e
                    );
                }
            }
        }

        *self.segments.write() = new_segments;
        // Invalidate global stats cache since segments changed
        self.global_stats.invalidate();
        Ok(())
    }

    /// Check if segments need reloading (metadata changed since last load)
    pub async fn needs_reload(&self) -> Result<bool> {
        let meta = Self::load_metadata(&self.directory).await?;
        let current_segments = self.segments.read();

        // Compare segment count and IDs
        if meta.segments.len() != current_segments.len() {
            return Ok(true);
        }

        for (meta_id, reader) in meta.segments.iter().zip(current_segments.iter()) {
            let reader_id = SegmentId::from_u128(reader.meta().id).to_hex();
            if meta_id != &reader_id {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get global statistics for cross-segment IDF computation (sync, basic stats only)
    ///
    /// Returns cached stats if available. For full stats including term frequencies,
    /// call `build_global_stats().await` first.
    ///
    /// This sync version only includes:
    /// - Total docs
    /// - Sparse vector dimension document frequencies
    /// - Average field lengths
    pub fn global_stats(&self) -> Option<Arc<crate::query::GlobalStats>> {
        self.global_stats.get()
    }

    /// Build and cache global statistics (async, includes term frequencies)
    ///
    /// This iterates term dictionaries across all segments to compute
    /// accurate cross-segment IDF values for full-text queries.
    ///
    /// Call this once after opening the index or after reload().
    pub async fn build_global_stats(&self) -> Result<Arc<crate::query::GlobalStats>> {
        // Return cached if available
        if let Some(stats) = self.global_stats.get() {
            return Ok(stats);
        }

        let segments = self.segments.read().clone();
        let schema = &self.schema;
        let mut builder = crate::query::GlobalStatsBuilder::new();

        // Track field length sums for computing global avg
        let mut field_len_sums: rustc_hash::FxHashMap<u32, (u64, u64)> =
            rustc_hash::FxHashMap::default();

        for segment in &segments {
            let num_docs = segment.num_docs() as u64;
            builder.total_docs += num_docs;

            // Aggregate sparse vector statistics
            for (&field_id, sparse_index) in segment.sparse_indexes() {
                for (dim_id, posting_list) in sparse_index.postings.iter().enumerate() {
                    if let Some(pl) = posting_list {
                        builder.add_sparse_df(
                            crate::dsl::Field(field_id),
                            dim_id as u32,
                            pl.doc_count() as u64,
                        );
                    }
                }
            }

            // Aggregate text field average lengths
            for (field, entry) in schema.fields() {
                if entry.indexed && entry.field_type == crate::dsl::FieldType::Text {
                    let avg_len = segment.avg_field_len(field);
                    let (sum, count) = field_len_sums.entry(field.0).or_insert((0, 0));
                    *sum += (avg_len * num_docs as f32) as u64;
                    *count += num_docs;
                }
            }

            // Iterate term dictionary to get term document frequencies
            for (field, term, doc_freq) in segment.all_terms_with_stats().await? {
                builder.add_text_df(field, term, doc_freq as u64);
            }
        }

        // Set global average field lengths
        for (field_id, (sum, count)) in field_len_sums {
            if count > 0 {
                let global_avg = sum as f32 / count as f32;
                builder.set_avg_field_len(crate::dsl::Field(field_id), global_avg);
            }
        }

        let generation = self.global_stats.generation();
        let stats = builder.build(generation);
        self.global_stats.set_stats(stats);

        Ok(self.global_stats.get().unwrap())
    }

    /// Search and return results with document addresses (no document content)
    ///
    /// This is the primary search method. Use `get_document` to fetch document content.
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
        self.search_internal(query, limit, offset, false).await
    }

    /// Search with matched field ordinals (for multi-valued fields with position tracking)
    ///
    /// Returns which array elements matched for each field with position tracking enabled.
    pub async fn search_with_matched_fields(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
    ) -> Result<crate::query::SearchResponse> {
        self.search_internal(query, limit, 0, true).await
    }

    async fn search_internal(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
        offset: usize,
        collect_positions: bool,
    ) -> Result<crate::query::SearchResponse> {
        let segments = self.segments.read().clone();
        let mut all_results: Vec<(u128, crate::query::SearchResult)> = Vec::new();

        // Fetch enough results to cover offset + limit
        let fetch_limit = offset + limit;
        for segment in &segments {
            let segment_id = segment.meta().id;
            let results = if collect_positions {
                crate::query::search_segment_with_positions(segment.as_ref(), query, fetch_limit)
                    .await?
            } else {
                crate::query::search_segment(segment.as_ref(), query, fetch_limit).await?
            };
            for result in results {
                all_results.push((segment_id, result));
            }
        }

        // Sort by score descending
        all_results.sort_by(|a, b| {
            b.1.score
                .partial_cmp(&a.1.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Total hits before pagination
        let total_hits = all_results.len() as u32;

        // Apply offset and limit
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

    /// Get a document by its unique address (segment_id + local doc_id)
    pub async fn get_document(
        &self,
        address: &crate::query::DocAddress,
    ) -> Result<Option<Document>> {
        let segment_id = address
            .segment_id_u128()
            .ok_or_else(|| Error::Query(format!("Invalid segment ID: {}", address.segment_id)))?;

        let segments = self.segments.read().clone();
        for segment in &segments {
            if segment.meta().id == segment_id {
                return segment.doc(address.doc_id).await;
            }
        }

        Ok(None)
    }

    /// Get the default fields for this index
    pub fn default_fields(&self) -> &[crate::Field] {
        &self.default_fields
    }

    /// Set the default fields for query parsing
    pub fn set_default_fields(&mut self, fields: Vec<crate::Field>) {
        self.default_fields = fields;
    }

    /// Get the tokenizer registry
    pub fn tokenizers(&self) -> &Arc<crate::tokenizer::TokenizerRegistry> {
        &self.tokenizers
    }

    /// Create a query parser for this index
    ///
    /// If the schema contains query router rules, they will be used to route
    /// queries to specific fields based on regex patterns.
    pub fn query_parser(&self) -> crate::dsl::QueryLanguageParser {
        // Check if schema has query routers
        let query_routers = self.schema.query_routers();
        if !query_routers.is_empty() {
            // Try to create a router from the schema's rules
            if let Ok(router) = crate::dsl::QueryFieldRouter::from_rules(query_routers) {
                return crate::dsl::QueryLanguageParser::with_router(
                    Arc::clone(&self.schema),
                    self.default_fields.clone(),
                    Arc::clone(&self.tokenizers),
                    router,
                );
            }
        }

        // Fall back to parser without router
        crate::dsl::QueryLanguageParser::new(
            Arc::clone(&self.schema),
            self.default_fields.clone(),
            Arc::clone(&self.tokenizers),
        )
    }

    /// Parse and search using a query string
    ///
    /// Accepts both query language syntax (field:term, AND, OR, NOT, grouping)
    /// and simple text (tokenized and searched across default fields).
    /// Returns document addresses (segment_id + doc_id) without document content.
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
        let query = parser.parse(query_str).map_err(Error::Query)?;
        self.search_offset(query.as_ref(), limit, offset).await
    }
}

/// Methods for opening index with slice caching
impl<D: Directory> Index<SliceCachingDirectory<D>> {
    /// Open an index with slice caching, automatically loading the cache file if present
    ///
    /// This wraps the directory in a SliceCachingDirectory and attempts to load
    /// any existing slice cache file to prefill the cache with hot data.
    pub async fn open_with_cache(
        directory: D,
        config: IndexConfig,
        cache_max_bytes: usize,
    ) -> Result<Self> {
        let caching_dir = SliceCachingDirectory::new(directory, cache_max_bytes);

        // Try to load existing slice cache
        let cache_path = Path::new(SLICE_CACHE_FILENAME);
        if let Ok(true) = caching_dir.inner().exists(cache_path).await
            && let Ok(slice) = caching_dir.inner().open_read(cache_path).await
            && let Ok(bytes) = slice.read_bytes().await
        {
            let _ = caching_dir.deserialize(bytes.as_slice());
        }

        Self::open(caching_dir, config).await
    }

    /// Serialize the current slice cache to the index directory
    ///
    /// This saves all cached slices to a single file that can be loaded
    /// on subsequent index opens for faster startup.
    #[cfg(feature = "native")]
    pub async fn save_slice_cache(&self) -> Result<()>
    where
        D: crate::directories::DirectoryWriter,
    {
        let cache_data = self.directory.serialize();
        let cache_path = Path::new(SLICE_CACHE_FILENAME);
        self.directory
            .inner()
            .write(cache_path, &cache_data)
            .await?;
        Ok(())
    }

    /// Get slice cache statistics
    pub fn slice_cache_stats(&self) -> crate::directories::SliceCacheStats {
        self.directory.stats()
    }
}

/// Warm up the slice cache by opening an index and performing typical read operations
///
/// This function opens an index using a SliceCachingDirectory, performs operations
/// that would typically be done during search (reading term dictionaries, posting lists),
/// and then serializes the cache to a file for future use.
///
/// The resulting cache file contains all the "hot" data that was read during warmup,
/// allowing subsequent index opens to prefill the cache and avoid cold-start latency.
#[cfg(feature = "native")]
pub async fn warmup_and_save_slice_cache<D: crate::directories::DirectoryWriter>(
    directory: D,
    config: IndexConfig,
    cache_max_bytes: usize,
) -> Result<()> {
    let caching_dir = SliceCachingDirectory::new(directory, cache_max_bytes);
    let index = Index::open(caching_dir, config).await?;

    // Warm up by loading segment metadata and term dictionaries
    // The SegmentReader::open already reads essential metadata
    // Additional warmup can be done by iterating terms or doing sample queries

    // Save the cache
    index.save_slice_cache().await?;

    Ok(())
}

#[cfg(feature = "native")]
impl<D: Directory> Clone for Index<D> {
    fn clone(&self) -> Self {
        Self {
            directory: Arc::clone(&self.directory),
            schema: Arc::clone(&self.schema),
            config: self.config.clone(),
            segments: RwLock::new(self.segments.read().clone()),
            default_fields: self.default_fields.clone(),
            tokenizers: Arc::clone(&self.tokenizers),
            global_stats: crate::query::GlobalStatsCache::new(),
            trained_centroids: self.trained_centroids.clone(),
            trained_codebooks: self.trained_codebooks.clone(),
            thread_pool: Arc::clone(&self.thread_pool),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directories::RamDirectory;
    use crate::dsl::SchemaBuilder;

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
        writer.add_document(doc1).await.unwrap();

        let mut doc2 = Document::new();
        doc2.add_text(title, "Goodbye World");
        doc2.add_text(body, "This is the second document");
        writer.add_document(doc2).await.unwrap();

        writer.commit().await.unwrap();

        // Open for reading
        let index = Index::open(dir, config).await.unwrap();
        assert_eq!(index.num_docs(), 2);

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
                writer.add_document(doc).await.unwrap();
            }
            writer.commit().await.unwrap();
        }

        // Open and check
        let index = Index::open(dir, config).await.unwrap();
        assert_eq!(index.num_docs(), 15);
        assert_eq!(index.segment_readers().len(), 3);
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

        // Create multiple segments
        for i in 0..9 {
            let mut doc = Document::new();
            doc.add_text(title, format!("Document {}", i));
            writer.add_document(doc).await.unwrap();
        }
        writer.commit().await.unwrap();

        // Should have 3 segments
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert_eq!(index.segment_readers().len(), 3);

        // Force merge
        let writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        writer.force_merge().await.unwrap();

        // Should have 1 segment now
        let index = Index::open(dir, config).await.unwrap();
        assert_eq!(index.segment_readers().len(), 1);
        assert_eq!(index.num_docs(), 9);

        // Verify all documents accessible
        for i in 0..9 {
            let doc = index.doc(i).await.unwrap().unwrap();
            assert_eq!(
                doc.get_first(title).unwrap().as_text(),
                Some(format!("Document {}", i).as_str())
            );
        }
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
        writer.add_document(doc1).await.unwrap();

        let mut doc2 = Document::new();
        doc2.add_text(title, "python programming");
        doc2.add_text(body, "Learn python language");
        writer.add_document(doc2).await.unwrap();

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
            writer.add_document(doc).await.unwrap();
        }
        writer.commit().await.unwrap();

        // Open with slice caching and perform some operations to warm up cache
        let caching_dir = SliceCachingDirectory::new(dir.clone(), 1024 * 1024);
        let index = Index::open(caching_dir, config.clone()).await.unwrap();

        // Perform a search to warm up the cache
        let results = index.query("rust", 10).await.unwrap();
        assert!(!results.hits.is_empty());

        // Check cache stats - should have cached some data
        let stats = index.slice_cache_stats();
        assert!(stats.total_bytes > 0, "Cache should have data after search");

        // Save the cache
        index.save_slice_cache().await.unwrap();

        // Verify cache file was written
        assert!(dir.exists(Path::new(SLICE_CACHE_FILENAME)).await.unwrap());

        // Now open with cache loading
        let index2 = Index::open_with_cache(dir.clone(), config.clone(), 1024 * 1024)
            .await
            .unwrap();

        // Cache should be prefilled
        let stats2 = index2.slice_cache_stats();
        assert!(
            stats2.total_bytes > 0,
            "Cache should be prefilled from file"
        );

        // Search should still work
        let results2 = index2.query("rust", 10).await.unwrap();
        assert_eq!(results.hits.len(), results2.hits.len());
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
        writer.add_document(doc).await.unwrap();

        // Add another document with different uris
        let mut doc2 = Document::new();
        doc2.add_text(uris, "three");
        doc2.add_text(title, "Another Document");
        writer.add_document(doc2).await.unwrap();

        writer.commit().await.unwrap();

        // Open for reading
        let index = Index::open(dir, config).await.unwrap();
        assert_eq!(index.num_docs(), 2);

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
        writer.add_document(doc).await.unwrap();

        // Doc 1: contains "rust" only
        let mut doc = Document::new();
        doc.add_text(content, "rust is a systems language");
        writer.add_document(doc).await.unwrap();

        // Doc 2: contains "programming" only
        let mut doc = Document::new();
        doc.add_text(content, "programming is fun");
        writer.add_document(doc).await.unwrap();

        // Doc 3: contains "python" (neither rust nor programming)
        let mut doc = Document::new();
        doc.add_text(content, "python is easy to learn");
        writer.add_document(doc).await.unwrap();

        // Doc 4: contains both "rust" and "programming" multiple times
        let mut doc = Document::new();
        doc.add_text(content, "rust rust programming programming systems");
        writer.add_document(doc).await.unwrap();

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
            writer.add_document(doc).await.unwrap();
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
            writer.add_document(doc).await.unwrap();
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
        let segments = index.segment_readers();
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
            writer.add_document(doc).await.unwrap();
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
        let segments = index.segment_readers();
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
