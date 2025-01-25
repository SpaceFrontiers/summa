//! IndexReader and Searcher - two-tier search abstraction
//!
//! - `IndexReader`: Manages reload policy, holds current Searcher
//! - `Searcher`: Holds segment snapshot, provides search/doc access
//!
//! Usage pattern:
//! 1. Get IndexReader from Index via `index.reader()`
//! 2. Get Searcher from IndexReader via `reader.searcher()`
//! 3. Use Searcher for search/doc operations
//! 4. Searcher dropped → segment refs released → deferred deletions proceed

use std::sync::Arc;

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::dsl::Schema;
use crate::error::Result;
use crate::query::LazyGlobalStats;
use crate::segment::{SegmentId, SegmentReader, SegmentSnapshot};
use crate::structures::{CoarseCentroids, PQCodebook};

#[cfg(feature = "native")]
use crate::directories::DirectoryWriter;

/// Searcher - holds segment snapshot for safe searching
///
/// Segments referenced by this reader won't be deleted until the reader is dropped.
#[cfg(feature = "native")]
pub struct Searcher<D: DirectoryWriter + 'static> {
    /// Segment snapshot holding refs (prevents deletion)
    _snapshot: SegmentSnapshot<D>,
    /// Loaded segment readers
    segments: Vec<Arc<SegmentReader>>,
    /// Schema
    schema: Arc<Schema>,
    /// Default fields for search
    default_fields: Vec<crate::Field>,
    /// Tokenizers
    tokenizers: Arc<crate::tokenizer::TokenizerRegistry>,
    /// Trained centroids per field
    trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
    /// Trained codebooks per field
    trained_codebooks: FxHashMap<u32, Arc<PQCodebook>>,
    /// Lazy global statistics for cross-segment IDF computation
    /// Bound to this Searcher's segment set - stats are computed lazily and cached
    global_stats: Arc<LazyGlobalStats>,
}

#[cfg(feature = "native")]
impl<D: DirectoryWriter + 'static> Searcher<D> {
    /// Create a new reader from a snapshot
    pub(crate) async fn from_snapshot(
        directory: Arc<D>,
        schema: Arc<Schema>,
        snapshot: SegmentSnapshot<D>,
        trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
        trained_codebooks: FxHashMap<u32, Arc<PQCodebook>>,
        term_cache_blocks: usize,
    ) -> Result<Self> {
        // Load segment readers for the snapshot
        let mut segments = Vec::new();
        let mut doc_id_offset = 0u32;

        for id_str in snapshot.segment_ids() {
            let Some(segment_id) = SegmentId::from_hex(id_str) else {
                continue;
            };

            match SegmentReader::open(
                directory.as_ref(),
                segment_id,
                Arc::clone(&schema),
                doc_id_offset,
                term_cache_blocks,
            )
            .await
            {
                Ok(reader) => {
                    doc_id_offset += reader.meta().num_docs;
                    segments.push(Arc::new(reader));
                }
                Err(e) => {
                    log::warn!("Failed to open segment {}: {:?}", id_str, e);
                }
            }
        }

        // Build default fields
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

        // Create lazy global statistics bound to this segment set
        // IDF values will be computed on-demand and cached
        let global_stats = Arc::new(LazyGlobalStats::new(segments.clone()));

        Ok(Self {
            _snapshot: snapshot,
            segments,
            schema,
            default_fields,
            tokenizers: Arc::new(crate::tokenizer::TokenizerRegistry::default()),
            trained_centroids,
            trained_codebooks,
            global_stats,
        })
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get segment readers
    pub fn segment_readers(&self) -> &[Arc<SegmentReader>] {
        &self.segments
    }

    /// Get default fields for search
    pub fn default_fields(&self) -> &[crate::Field] {
        &self.default_fields
    }

    /// Get tokenizer registry
    pub fn tokenizers(&self) -> &crate::tokenizer::TokenizerRegistry {
        &self.tokenizers
    }

    /// Get trained centroids
    pub fn trained_centroids(&self) -> &FxHashMap<u32, Arc<CoarseCentroids>> {
        &self.trained_centroids
    }

    /// Get trained codebooks
    pub fn trained_codebooks(&self) -> &FxHashMap<u32, Arc<PQCodebook>> {
        &self.trained_codebooks
    }

    /// Get lazy global statistics for cross-segment IDF computation
    ///
    /// Statistics are computed lazily on first access and cached per term/dimension.
    /// The stats are bound to this Searcher's segment set.
    pub fn global_stats(&self) -> &Arc<LazyGlobalStats> {
        &self.global_stats
    }

    /// Get total document count across all segments
    pub fn num_docs(&self) -> u32 {
        self.segments.iter().map(|s| s.meta().num_docs).sum()
    }

    /// Get number of segments
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Get a document by global doc_id
    pub async fn doc(&self, doc_id: u32) -> Result<Option<crate::dsl::Document>> {
        let mut offset = 0u32;
        for segment in &self.segments {
            let segment_docs = segment.meta().num_docs;
            if doc_id < offset + segment_docs {
                let local_doc_id = doc_id - offset;
                return segment.doc(local_doc_id).await;
            }
            offset += segment_docs;
        }
        Ok(None)
    }

    /// Search across all segments and return aggregated results
    pub async fn search(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
    ) -> Result<Vec<crate::query::SearchResult>> {
        self.search_with_offset(query, limit, 0).await
    }

    /// Search with offset for pagination
    pub async fn search_with_offset(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<crate::query::SearchResult>> {
        let fetch_limit = offset + limit;
        let mut all_results: Vec<(u128, crate::query::SearchResult)> = Vec::new();

        for segment in &self.segments {
            let segment_id = segment.meta().id;
            let results =
                crate::query::search_segment(segment.as_ref(), query, fetch_limit).await?;
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

        // Apply offset and limit, return just the results
        Ok(all_results
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(_, result)| result)
            .collect())
    }
}

/// IndexReader - manages Searcher with reload policy
///
/// The IndexReader periodically reloads its Searcher to pick up new segments.
/// Uses SegmentManager as authoritative source for segment state (avoids race conditions).
#[cfg(feature = "native")]
pub struct IndexReader<D: DirectoryWriter + 'static> {
    /// Schema
    schema: Arc<Schema>,
    /// Segment manager - authoritative source for segments
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Current searcher
    searcher: RwLock<Arc<Searcher<D>>>,
    /// Trained centroids
    trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
    /// Trained codebooks
    trained_codebooks: FxHashMap<u32, Arc<PQCodebook>>,
    /// Term cache blocks
    term_cache_blocks: usize,
    /// Last reload time
    last_reload: RwLock<std::time::Instant>,
    /// Reload interval (default 1 second)
    reload_interval: std::time::Duration,
}

#[cfg(feature = "native")]
impl<D: DirectoryWriter + 'static> IndexReader<D> {
    /// Create a new searcher from a segment manager
    pub async fn from_segment_manager(
        schema: Arc<Schema>,
        segment_manager: Arc<crate::merge::SegmentManager<D>>,
        trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
        trained_codebooks: FxHashMap<u32, Arc<PQCodebook>>,
        term_cache_blocks: usize,
    ) -> Result<Self> {
        let reader = Self::create_reader(
            &schema,
            &segment_manager,
            &trained_centroids,
            &trained_codebooks,
            term_cache_blocks,
        )
        .await?;

        Ok(Self {
            schema,
            segment_manager,
            searcher: RwLock::new(Arc::new(reader)),
            trained_centroids,
            trained_codebooks,
            term_cache_blocks,
            last_reload: RwLock::new(std::time::Instant::now()),
            reload_interval: std::time::Duration::from_secs(1),
        })
    }

    /// Create a new reader with fresh snapshot from segment manager
    /// This avoids race conditions by using SegmentManager's locked metadata
    async fn create_reader(
        schema: &Arc<Schema>,
        segment_manager: &Arc<crate::merge::SegmentManager<D>>,
        trained_centroids: &FxHashMap<u32, Arc<CoarseCentroids>>,
        trained_codebooks: &FxHashMap<u32, Arc<PQCodebook>>,
        term_cache_blocks: usize,
    ) -> Result<Searcher<D>> {
        // Use SegmentManager's acquire_snapshot - this locks metadata and tracker together
        let snapshot = segment_manager.acquire_snapshot().await;

        Searcher::from_snapshot(
            segment_manager.directory(),
            Arc::clone(schema),
            snapshot,
            trained_centroids.clone(),
            trained_codebooks.clone(),
            term_cache_blocks,
        )
        .await
    }

    /// Set reload interval
    pub fn set_reload_interval(&mut self, interval: std::time::Duration) {
        self.reload_interval = interval;
    }

    /// Get current searcher (reloads if interval exceeded)
    pub async fn searcher(&self) -> Result<Arc<Searcher<D>>> {
        // Check if reload needed
        let should_reload = {
            let last = self.last_reload.read();
            last.elapsed() >= self.reload_interval
        };

        if should_reload {
            self.reload().await?;
        }

        Ok(Arc::clone(&*self.searcher.read()))
    }

    /// Force reload reader with fresh snapshot
    pub async fn reload(&self) -> Result<()> {
        let new_reader = Self::create_reader(
            &self.schema,
            &self.segment_manager,
            &self.trained_centroids,
            &self.trained_codebooks,
            self.term_cache_blocks,
        )
        .await?;

        // Swap in new searcher (old one will be dropped when last ref released)
        *self.searcher.write() = Arc::new(new_reader);
        *self.last_reload.write() = std::time::Instant::now();

        Ok(())
    }

    /// Get schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}
