//! IndexWriter - async document indexing with parallel segment building
//!
//! This module is only compiled with the "native" feature.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashMap;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::mpsc;

use crate::DocId;
use crate::directories::DirectoryWriter;
use crate::dsl::{Document, Field, Schema};
use crate::error::{Error, Result};
use crate::segment::{
    SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentMerger, SegmentReader,
};
use crate::tokenizer::BoxedTokenizer;

use super::IndexConfig;

/// Async IndexWriter for adding documents and committing segments
///
/// Features:
/// - Parallel indexing with multiple segment builders
/// - Streams documents to disk immediately (no in-memory document storage)
/// - Uses string interning for terms (reduced allocations)
/// - Uses hashbrown HashMap (faster than BTreeMap)
///
/// **State management:**
/// - Building segments: Managed here (pending_builds)
/// - Committed segments + metadata: Managed by SegmentManager (sole owner of metadata.json)
pub struct IndexWriter<D: DirectoryWriter + 'static> {
    pub(super) directory: Arc<D>,
    pub(super) schema: Arc<Schema>,
    pub(super) config: IndexConfig,
    builder_config: SegmentBuilderConfig,
    tokenizers: FxHashMap<Field, BoxedTokenizer>,
    /// Multiple segment builders for parallel indexing
    builders: Vec<AsyncMutex<Option<SegmentBuilder>>>,
    /// Segment manager - owns metadata.json, handles segments and background merging
    pub(super) segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Channel sender for completed segment IDs from background builds
    segment_id_sender: mpsc::UnboundedSender<String>,
    /// Channel receiver for completed segment IDs
    segment_id_receiver: AsyncMutex<mpsc::UnboundedReceiver<String>>,
    /// Count of in-flight background builds
    pending_builds: Arc<AtomicUsize>,
}

impl<D: DirectoryWriter + 'static> IndexWriter<D> {
    /// Create a new index in the directory
    pub async fn create(directory: D, schema: Schema, config: IndexConfig) -> Result<Self> {
        Self::create_with_config(directory, schema, config, SegmentBuilderConfig::default()).await
    }

    /// Create a new index with custom builder config
    pub async fn create_with_config(
        directory: D,
        schema: Schema,
        config: IndexConfig,
        builder_config: SegmentBuilderConfig,
    ) -> Result<Self> {
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

        // Create multiple builders for parallel indexing
        let num_builders = config.num_indexing_threads.max(1);
        let mut builders = Vec::with_capacity(num_builders);
        for _ in 0..num_builders {
            builders.push(AsyncMutex::new(None));
        }

        // Create channel for background builds to report completed segment IDs
        let (segment_id_sender, segment_id_receiver) = mpsc::unbounded_channel();

        // Initialize empty metadata for new index
        let metadata = super::IndexMetadata::new();

        // Create segment manager - owns metadata.json
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
            builder_config,
            tokenizers: FxHashMap::default(),
            builders,
            segment_manager,
            segment_id_sender,
            segment_id_receiver: AsyncMutex::new(segment_id_receiver),
            pending_builds: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Open an existing index for writing
    pub async fn open(directory: D, config: IndexConfig) -> Result<Self> {
        Self::open_with_config(directory, config, SegmentBuilderConfig::default()).await
    }

    /// Open an existing index with custom builder config
    pub async fn open_with_config(
        directory: D,
        config: IndexConfig,
        builder_config: SegmentBuilderConfig,
    ) -> Result<Self> {
        let directory = Arc::new(directory);

        // Read schema
        let schema_slice = directory.open_read(Path::new("schema.json")).await?;
        let schema_bytes = schema_slice.read_bytes().await?;
        let schema: Schema = serde_json::from_slice(schema_bytes.as_slice())
            .map_err(|e| Error::Serialization(e.to_string()))?;
        let schema = Arc::new(schema);

        // Load unified metadata
        let metadata = super::IndexMetadata::load(directory.as_ref()).await?;

        // Create multiple builders for parallel indexing
        let num_builders = config.num_indexing_threads.max(1);
        let mut builders = Vec::with_capacity(num_builders);
        for _ in 0..num_builders {
            builders.push(AsyncMutex::new(None));
        }

        // Create channel for background builds to report completed segment IDs
        let (segment_id_sender, segment_id_receiver) = mpsc::unbounded_channel();

        // Create segment manager - owns metadata.json
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
            builder_config,
            tokenizers: FxHashMap::default(),
            builders,
            segment_manager,
            segment_id_sender,
            segment_id_receiver: AsyncMutex::new(segment_id_receiver),
            pending_builds: Arc::new(AtomicUsize::new(0)),
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
    /// Documents are distributed randomly across multiple builders for parallel indexing.
    /// Random distribution avoids atomic contention and provides better load balancing.
    /// When a builder reaches `max_docs_per_segment`, it is committed and a new one starts.
    pub async fn add_document(&self, doc: Document) -> Result<DocId> {
        use rand::Rng;

        // Random selection of builder - avoids atomic contention
        let builder_idx = rand::rng().random_range(0..self.builders.len());

        let mut builder_guard = self.builders[builder_idx].lock().await;

        // Initialize builder if needed
        if builder_guard.is_none() {
            let mut builder =
                SegmentBuilder::new((*self.schema).clone(), self.builder_config.clone())?;
            for (field, tokenizer) in &self.tokenizers {
                builder.set_tokenizer(*field, tokenizer.clone_box());
            }
            *builder_guard = Some(builder);
        }

        let builder = builder_guard.as_mut().unwrap();
        let doc_id = builder.add_document(doc)?;

        // Check if we need to commit
        if builder.num_docs() >= self.config.max_docs_per_segment {
            let full_builder = builder_guard.take().unwrap();
            drop(builder_guard); // Release lock before spawning background task
            self.spawn_background_build(full_builder);
        }

        Ok(doc_id)
    }

    /// Spawn a background task to build a segment without blocking document ingestion
    ///
    /// The background task will send its segment ID through the channel when complete,
    /// allowing indexing to continue immediately.
    fn spawn_background_build(&self, builder: SegmentBuilder) {
        let directory = Arc::clone(&self.directory);
        let segment_id = SegmentId::new();
        let segment_hex = segment_id.to_hex();
        let sender = self.segment_id_sender.clone();
        let segment_manager = Arc::clone(&self.segment_manager);

        self.pending_builds.fetch_add(1, Ordering::SeqCst);

        // Spawn a fully independent task that registers its own segment ID
        tokio::spawn(async move {
            match builder.build(directory.as_ref(), segment_id).await {
                Ok(_) => {
                    // Register segment via SegmentManager (also triggers merge check)
                    let _ = segment_manager.register_segment(segment_hex.clone()).await;
                    // Also send through channel for flush() to know when all are done
                    let _ = sender.send(segment_hex);
                }
                Err(e) => {
                    // Log error but don't crash - segment just won't be registered
                    eprintln!("Background segment build failed: {:?}", e);
                }
            }
        });
    }

    /// Collect any completed segment IDs from the channel (non-blocking)
    ///
    /// Merge checking is now handled by SegmentManager.register_segment().
    async fn collect_completed_segments(&self) {
        let mut receiver = self.segment_id_receiver.lock().await;
        while let Ok(_segment_hex) = receiver.try_recv() {
            // Segment ID already registered by the background task via SegmentManager
            self.pending_builds.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// Get the number of pending background builds
    pub fn pending_build_count(&self) -> usize {
        self.pending_builds.load(Ordering::SeqCst)
    }

    /// Get the number of pending background merges
    pub fn pending_merge_count(&self) -> usize {
        self.segment_manager.pending_merge_count()
    }

    /// Check merge policy and spawn background merges if needed
    ///
    /// This is called automatically after segment builds complete via SegmentManager.
    /// Can also be called manually to trigger merge checking.
    pub async fn maybe_merge(&self) {
        self.segment_manager.maybe_merge().await;
    }

    /// Wait for all pending merges to complete
    pub async fn wait_for_merges(&self) {
        self.segment_manager.wait_for_merges().await;
    }

    /// Clean up orphan segment files that are not registered
    ///
    /// This can happen if the process halts after segment files are written
    /// but before they are registered in segments.json. Call this after opening
    /// an index to reclaim disk space from incomplete operations.
    ///
    /// Returns the number of orphan segments deleted.
    pub async fn cleanup_orphan_segments(&self) -> Result<usize> {
        self.segment_manager.cleanup_orphan_segments().await
    }

    /// Get current builder statistics for debugging (aggregated from all builders)
    pub async fn get_builder_stats(&self) -> Option<crate::segment::SegmentBuilderStats> {
        let mut total_stats: Option<crate::segment::SegmentBuilderStats> = None;

        for builder_mutex in &self.builders {
            let guard = builder_mutex.lock().await;
            if let Some(builder) = guard.as_ref() {
                let stats = builder.stats();
                if let Some(ref mut total) = total_stats {
                    total.num_docs += stats.num_docs;
                    total.unique_terms += stats.unique_terms;
                    total.postings_in_memory += stats.postings_in_memory;
                    total.interned_strings += stats.interned_strings;
                    total.doc_field_lengths_size += stats.doc_field_lengths_size;
                } else {
                    total_stats = Some(stats);
                }
            }
        }

        total_stats
    }

    /// Flush current builders to background processing (non-blocking)
    ///
    /// This takes all current builders with documents and spawns background tasks
    /// to build them. Returns immediately - use `commit()` for durability.
    /// New documents can continue to be added while segments are being built.
    pub async fn flush(&self) -> Result<()> {
        // Collect any already-completed segments
        self.collect_completed_segments().await;

        // Take all builders that have documents and spawn background builds
        for builder_mutex in &self.builders {
            let mut guard = builder_mutex.lock().await;
            if let Some(builder) = guard.take()
                && builder.num_docs() > 0
            {
                self.spawn_background_build(builder);
            }
        }

        Ok(())
    }

    /// Commit all pending segments to disk and wait for completion
    ///
    /// This flushes any current builders and waits for ALL background builds
    /// and merges to complete. Provides durability guarantees - all data is persisted.
    ///
    /// **Auto-triggers vector index build** when threshold is crossed for any field.
    pub async fn commit(&self) -> Result<()> {
        // First flush any current builders
        self.flush().await?;

        // Wait for all pending builds to complete
        let mut receiver = self.segment_id_receiver.lock().await;
        while self.pending_builds.load(Ordering::SeqCst) > 0 {
            match receiver.recv().await {
                Some(_segment_hex) => {
                    self.pending_builds.fetch_sub(1, Ordering::SeqCst);
                }
                None => break, // Channel closed
            }
        }
        drop(receiver);

        // Auto-trigger vector index build if threshold crossed
        self.maybe_build_vector_index().await?;

        Ok(())
    }

    // Vector index building methods are in vector_builder.rs

    /// Merge all segments into one (called explicitly via force_merge)
    async fn do_merge(&self) -> Result<()> {
        let segment_ids = self.segment_manager.get_segment_ids().await;

        if segment_ids.len() < 2 {
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

        // Atomically update segments and delete old ones via SegmentManager
        self.segment_manager
            .replace_segments(vec![new_segment_id.to_hex()], ids_to_merge)
            .await?;

        Ok(())
    }

    /// Force merge all segments into one
    pub async fn force_merge(&self) -> Result<()> {
        // First commit all pending documents (waits for completion)
        self.commit().await?;
        // Then merge all segments
        self.do_merge().await
    }

    // Vector index methods (build_vector_index, rebuild_vector_index, etc.)
    // are implemented in vector_builder.rs
}
