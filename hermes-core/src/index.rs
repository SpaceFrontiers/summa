//! Index - multi-segment async search index
//!
//! Components:
//! - Index: main entry point for searching
//! - IndexWriter: for adding documents and committing segments
//! - Supports multiple segments with merge

use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
#[cfg(feature = "native")]
use rustc_hash::FxHashMap;

use crate::DocId;
#[cfg(feature = "native")]
use crate::directories::DirectoryWriter;
use crate::directories::{Directory, SliceCachingDirectory};
use crate::dsl::{Document, Field, Schema};
use crate::error::{Error, Result};
#[cfg(feature = "native")]
use crate::segment::{SegmentBuilder, SegmentMerger};
use crate::segment::{SegmentId, SegmentReader};
use crate::structures::BlockPostingList;
#[cfg(feature = "native")]
use crate::tokenizer::BoxedTokenizer;

#[cfg(feature = "native")]
use tokio::sync::Mutex as AsyncMutex;

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
    /// Max documents per segment before auto-commit
    pub max_docs_per_segment: u32,
    /// Min segments to trigger merge
    pub merge_threshold: usize,
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
            max_docs_per_segment: 100_000,
            merge_threshold: 5,
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

        // Read segment list
        let segments = Self::load_segments(&directory, &schema, &config).await?;

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
            #[cfg(feature = "native")]
            thread_pool,
        })
    }

    async fn load_segments(
        directory: &Arc<D>,
        schema: &Arc<Schema>,
        config: &IndexConfig,
    ) -> Result<Vec<Arc<SegmentReader>>> {
        // Read segments.json which lists all segment IDs
        let segments_path = Path::new("segments.json");
        if !directory.exists(segments_path).await? {
            return Ok(Vec::new());
        }

        let segments_slice = directory.open_read(segments_path).await?;
        let segments_bytes = segments_slice.read_bytes().await?;
        let segment_ids: Vec<String> = serde_json::from_slice(segments_bytes.as_slice())
            .map_err(|e| Error::Serialization(e.to_string()))?;

        let mut segments = Vec::new();
        let mut doc_id_offset = 0u32;

        for id_str in segment_ids {
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

        Ok(segments)
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

    /// Reload segments from directory (after new segments added)
    pub async fn reload(&self) -> Result<()> {
        let new_segments = Self::load_segments(&self.directory, &self.schema, &self.config).await?;
        *self.segments.write() = new_segments;
        Ok(())
    }

    /// Search across all segments
    pub async fn search(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
    ) -> Result<Vec<crate::query::SearchResult>> {
        let segments = self.segments.read().clone();
        let mut all_results = Vec::new();

        for segment in &segments {
            let results = crate::query::search_segment(segment.as_ref(), query, limit).await?;
            all_results.extend(results);
        }

        // Sort by score descending
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(limit);

        Ok(all_results)
    }

    /// Search and return results with document addresses (no document content)
    pub async fn search_with_addresses(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
    ) -> Result<crate::query::SearchResponse> {
        let segments = self.segments.read().clone();
        let mut all_results: Vec<(u128, crate::query::SearchResult)> = Vec::new();

        for segment in &segments {
            let segment_id = segment.meta().id;
            let results = crate::query::search_segment(segment.as_ref(), query, limit).await?;
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
        all_results.truncate(limit);

        let total_hits = all_results.len() as u32;
        let hits: Vec<crate::query::SearchHit> = all_results
            .into_iter()
            .map(|(segment_id, result)| crate::query::SearchHit {
                address: crate::query::DocAddress::new(segment_id, result.doc_id),
                score: result.score,
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
        let parser = self.query_parser();
        let query = parser.parse(query_str).map_err(Error::Query)?;
        self.search_with_addresses(query.as_ref(), limit).await
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
        D: DirectoryWriter,
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
pub async fn warmup_and_save_slice_cache<D: DirectoryWriter>(
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
            thread_pool: Arc::clone(&self.thread_pool),
        }
    }
}

/// Async IndexWriter for adding documents and committing segments
///
/// Internally manages parallel segment building based on `num_indexing_threads` config.
/// Documents are distributed across multiple segment builders round-robin style.
#[cfg(feature = "native")]
pub struct IndexWriter<D: DirectoryWriter> {
    directory: Arc<D>,
    schema: Arc<Schema>,
    config: IndexConfig,
    tokenizers: FxHashMap<Field, BoxedTokenizer>,
    /// Multiple segment builders for parallel indexing
    segment_builders: AsyncMutex<Vec<SegmentBuilder>>,
    /// Round-robin counter for distributing documents
    next_builder: std::sync::atomic::AtomicUsize,
    /// List of committed segment IDs (hex strings)
    segment_ids: AsyncMutex<Vec<String>>,
}

#[cfg(feature = "native")]
impl<D: DirectoryWriter> IndexWriter<D> {
    /// Create a new index in the directory
    pub async fn create(directory: D, schema: Schema, config: IndexConfig) -> Result<Self> {
        let directory = Arc::new(directory);
        let schema = Arc::new(schema);

        // Write schema
        let schema_bytes =
            serde_json::to_vec(&*schema).map_err(|e| Error::Serialization(e.to_string()))?;
        directory
            .write(Path::new("schema.json"), &schema_bytes)
            .await?;

        // Write empty segments list
        let segments_bytes = serde_json::to_vec(&Vec::<String>::new())
            .map_err(|e| Error::Serialization(e.to_string()))?;
        directory
            .write(Path::new("segments.json"), &segments_bytes)
            .await?;

        Ok(Self {
            directory,
            schema,
            config,
            tokenizers: FxHashMap::default(),
            segment_builders: AsyncMutex::new(Vec::new()),
            next_builder: std::sync::atomic::AtomicUsize::new(0),
            segment_ids: AsyncMutex::new(Vec::new()),
        })
    }

    /// Open an existing index for writing
    pub async fn open(directory: D, config: IndexConfig) -> Result<Self> {
        let directory = Arc::new(directory);

        // Read schema
        let schema_slice = directory.open_read(Path::new("schema.json")).await?;
        let schema_bytes = schema_slice.read_bytes().await?;
        let schema: Schema = serde_json::from_slice(schema_bytes.as_slice())
            .map_err(|e| Error::Serialization(e.to_string()))?;
        let schema = Arc::new(schema);

        // Read existing segment IDs (hex strings)
        let segments_slice = directory.open_read(Path::new("segments.json")).await?;
        let segments_bytes = segments_slice.read_bytes().await?;
        let segment_ids: Vec<String> = serde_json::from_slice(segments_bytes.as_slice())
            .map_err(|e| Error::Serialization(e.to_string()))?;

        Ok(Self {
            directory,
            schema,
            config,
            tokenizers: FxHashMap::default(),
            segment_builders: AsyncMutex::new(Vec::new()),
            next_builder: std::sync::atomic::AtomicUsize::new(0),
            segment_ids: AsyncMutex::new(segment_ids),
        })
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Set tokenizer for a field
    pub fn set_tokenizer<T: crate::tokenizer::Tokenizer>(&mut self, field: Field, tokenizer: T) {
        self.tokenizers.insert(field, Box::new(tokenizer));
    }

    /// Add a document
    ///
    /// Documents are distributed across internal segment builders based on
    /// `num_indexing_threads` config. When any builder reaches `max_docs_per_segment`,
    /// it is committed to disk in the background.
    pub async fn add_document(&self, doc: Document) -> Result<DocId> {
        let num_builders = self.config.num_indexing_threads.max(1);
        let mut builders = self.segment_builders.lock().await;

        // Initialize builders if empty
        if builders.is_empty() {
            for _ in 0..num_builders {
                let mut builder = SegmentBuilder::new((*self.schema).clone());
                for (field, tokenizer) in &self.tokenizers {
                    builder.set_tokenizer(*field, tokenizer.clone_box());
                }
                builders.push(builder);
            }
        }

        // Round-robin distribution
        let idx = self
            .next_builder
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % num_builders;
        let builder = &mut builders[idx];
        let doc_id = builder.add_document(doc)?;

        // Check if this builder needs to be committed
        if builder.num_docs() >= self.config.max_docs_per_segment {
            // Take the full builder and replace with a new one
            let full_builder = std::mem::replace(builder, {
                let mut new_builder = SegmentBuilder::new((*self.schema).clone());
                for (field, tokenizer) in &self.tokenizers {
                    new_builder.set_tokenizer(*field, tokenizer.clone_box());
                }
                new_builder
            });
            drop(builders); // Release lock before async operation
            self.commit_segment(full_builder).await?;
        }

        Ok(doc_id)
    }

    /// Commit all pending segments to disk
    ///
    /// Commits all internal segment builders in parallel.
    pub async fn commit(&self) -> Result<()> {
        let mut builders = self.segment_builders.lock().await;

        if builders.is_empty() {
            return Ok(());
        }

        // Take all builders and replace with empty vec
        let builders_to_commit: Vec<SegmentBuilder> = std::mem::take(&mut *builders);
        drop(builders); // Release lock before async operations

        // Commit all non-empty builders in parallel
        let mut handles = Vec::new();

        for builder in builders_to_commit {
            if builder.num_docs() == 0 {
                continue;
            }

            let directory = Arc::clone(&self.directory);
            let compression_threads = self.config.num_compression_threads;
            let handle = tokio::spawn(async move {
                let segment_id = SegmentId::new();
                builder
                    .build_with_threads(directory.as_ref(), segment_id, compression_threads)
                    .await?;
                Ok::<String, Error>(segment_id.to_hex())
            });
            handles.push(handle);
        }

        // Collect all new segment IDs
        let mut new_segment_ids = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(id)) => new_segment_ids.push(id),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(Error::Internal(format!("Task join error: {}", e))),
            }
        }

        // Update segment list atomically
        if !new_segment_ids.is_empty() {
            let mut segment_ids = self.segment_ids.lock().await;
            segment_ids.extend(new_segment_ids);

            let segments_bytes = serde_json::to_vec(&*segment_ids)
                .map_err(|e| Error::Serialization(e.to_string()))?;
            self.directory
                .write(Path::new("segments.json"), &segments_bytes)
                .await?;
        }

        Ok(())
    }

    async fn commit_segment(&self, builder: SegmentBuilder) -> Result<()> {
        if builder.num_docs() == 0 {
            return Ok(());
        }

        let segment_id = SegmentId::new();
        builder
            .build_with_threads(
                self.directory.as_ref(),
                segment_id,
                self.config.num_compression_threads,
            )
            .await?;

        // Update segment list
        let mut segment_ids = self.segment_ids.lock().await;
        segment_ids.push(segment_id.to_hex());

        let segments_bytes =
            serde_json::to_vec(&*segment_ids).map_err(|e| Error::Serialization(e.to_string()))?;
        self.directory
            .write(Path::new("segments.json"), &segments_bytes)
            .await?;

        // Check if merge is needed
        if segment_ids.len() >= self.config.merge_threshold {
            drop(segment_ids); // Release lock
            self.maybe_merge().await?;
        }

        Ok(())
    }

    /// Merge segments if threshold is reached
    async fn maybe_merge(&self) -> Result<()> {
        let segment_ids = self.segment_ids.lock().await;

        if segment_ids.len() < self.config.merge_threshold {
            return Ok(());
        }

        let ids_to_merge: Vec<String> = segment_ids.clone();
        drop(segment_ids);

        // Load segment readers
        let mut readers = Vec::new();
        let mut doc_offset = 0u32;

        for id_str in &ids_to_merge {
            let segment_id = SegmentId::from_hex(id_str)
                .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {}", id_str)))?;
            let reader = SegmentReader::open(
                self.directory.as_ref(),
                segment_id,
                Arc::clone(&self.schema),
                doc_offset,
                self.config.term_cache_blocks,
            )
            .await?;
            doc_offset += reader.meta().num_docs;
            readers.push(reader);
        }

        // Merge into new segment
        let merger = SegmentMerger::new(Arc::clone(&self.schema));
        let new_segment_id = SegmentId::new();
        merger
            .merge(self.directory.as_ref(), &readers, new_segment_id)
            .await?;

        // Update segment list
        let mut segment_ids = self.segment_ids.lock().await;
        segment_ids.clear();
        segment_ids.push(new_segment_id.to_hex());

        let segments_bytes =
            serde_json::to_vec(&*segment_ids).map_err(|e| Error::Serialization(e.to_string()))?;
        self.directory
            .write(Path::new("segments.json"), &segments_bytes)
            .await?;

        // Delete old segments
        for id_str in ids_to_merge {
            if let Some(segment_id) = SegmentId::from_hex(&id_str) {
                let _ = crate::segment::delete_segment(self.directory.as_ref(), segment_id).await;
            }
        }

        Ok(())
    }

    /// Force merge all segments into one
    pub async fn force_merge(&self) -> Result<()> {
        // First commit any pending documents
        self.commit().await?;

        let segment_ids = self.segment_ids.lock().await;
        if segment_ids.len() <= 1 {
            return Ok(());
        }

        let ids_to_merge: Vec<String> = segment_ids.clone();
        drop(segment_ids);

        // Load all segments
        let mut readers = Vec::new();
        let mut doc_offset = 0u32;

        for id_str in &ids_to_merge {
            let segment_id = SegmentId::from_hex(id_str)
                .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {}", id_str)))?;
            let reader = SegmentReader::open(
                self.directory.as_ref(),
                segment_id,
                Arc::clone(&self.schema),
                doc_offset,
                self.config.term_cache_blocks,
            )
            .await?;
            doc_offset += reader.meta().num_docs;
            readers.push(reader);
        }

        // Merge
        let merger = SegmentMerger::new(Arc::clone(&self.schema));
        let new_segment_id = SegmentId::new();
        merger
            .merge(self.directory.as_ref(), &readers, new_segment_id)
            .await?;

        // Update segment list
        let mut segment_ids = self.segment_ids.lock().await;
        segment_ids.clear();
        segment_ids.push(new_segment_id.to_hex());

        let segments_bytes =
            serde_json::to_vec(&*segment_ids).map_err(|e| Error::Serialization(e.to_string()))?;
        self.directory
            .write(Path::new("segments.json"), &segments_bytes)
            .await?;

        // Delete old segments
        for id_str in ids_to_merge {
            if let Some(segment_id) = SegmentId::from_hex(&id_str) {
                let _ = crate::segment::delete_segment(self.directory.as_ref(), segment_id).await;
            }
        }

        Ok(())
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
        let mut config = IndexConfig::default();
        config.max_docs_per_segment = 5; // Small segments for testing
        config.merge_threshold = 10; // Don't auto-merge

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
        let mut config = IndexConfig::default();
        config.max_docs_per_segment = 3;
        config.merge_threshold = 100; // Don't auto-merge

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
        assert!(results.hits.len() >= 1);

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
        assert!(
            dir.exists(Path::new(super::SLICE_CACHE_FILENAME))
                .await
                .unwrap()
        );

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
}
