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
pub struct IndexWriter<D: DirectoryWriter + 'static> {
    directory: Arc<D>,
    schema: Arc<Schema>,
    config: IndexConfig,
    builder_config: SegmentBuilderConfig,
    tokenizers: FxHashMap<Field, BoxedTokenizer>,
    /// Multiple segment builders for parallel indexing
    builders: Vec<AsyncMutex<Option<SegmentBuilder>>>,
    /// Segment manager - handles segment registration and background merging
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Channel sender for completed segment IDs from background builds
    segment_id_sender: mpsc::UnboundedSender<String>,
    /// Channel receiver for completed segment IDs
    segment_id_receiver: AsyncMutex<mpsc::UnboundedReceiver<String>>,
    /// Count of in-flight background builds
    pending_builds: Arc<AtomicUsize>,
    /// Unified index metadata - segments + vector index state
    metadata: AsyncMutex<super::IndexMetadata>,
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

        // Create segment manager with the configured merge policy
        let segment_manager = Arc::new(crate::merge::SegmentManager::new(
            Arc::clone(&directory),
            Arc::clone(&schema),
            Vec::new(),
            config.merge_policy.clone_box(),
            config.term_cache_blocks,
        ));

        // Initialize empty metadata for new index and save it
        let metadata = super::IndexMetadata::new();
        metadata.save(directory.as_ref()).await?;

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
            metadata: AsyncMutex::new(metadata),
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

        // Load unified metadata (with migration from old segments.json)
        let metadata = super::IndexMetadata::load(directory.as_ref()).await?;
        let segment_ids = metadata.segments.clone();

        // Create multiple builders for parallel indexing
        let num_builders = config.num_indexing_threads.max(1);
        let mut builders = Vec::with_capacity(num_builders);
        for _ in 0..num_builders {
            builders.push(AsyncMutex::new(None));
        }

        // Create channel for background builds to report completed segment IDs
        let (segment_id_sender, segment_id_receiver) = mpsc::unbounded_channel();

        // Create segment manager with the configured merge policy
        let segment_manager = Arc::new(crate::merge::SegmentManager::new(
            Arc::clone(&directory),
            Arc::clone(&schema),
            segment_ids,
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
            metadata: AsyncMutex::new(metadata),
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
                    segment_manager.register_segment(segment_hex.clone()).await;
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
    /// to complete. Provides durability guarantees - all data is persisted.
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

        // Update and save unified metadata
        let segment_ids = self.segment_manager.get_segment_ids().await;
        {
            let mut meta = self.metadata.lock().await;
            meta.segments = segment_ids;
            meta.save(self.directory.as_ref()).await?;
        }

        // Auto-trigger vector index build if threshold crossed
        self.maybe_build_vector_index().await?;

        Ok(())
    }

    /// Check if any dense vector field should be built and trigger training
    async fn maybe_build_vector_index(&self) -> Result<()> {
        use crate::dsl::FieldType;

        // Find dense vector fields that need ANN indexes
        let dense_fields: Vec<(Field, crate::dsl::DenseVectorConfig)> = self
            .schema
            .fields()
            .filter_map(|(field, entry)| {
                if entry.field_type == FieldType::DenseVector && entry.indexed {
                    entry
                        .dense_vector_config
                        .as_ref()
                        .filter(|c| !c.is_flat())
                        .map(|c| (field, c.clone()))
                } else {
                    None
                }
            })
            .collect();

        if dense_fields.is_empty() {
            return Ok(());
        }

        // Count total vectors across all segments
        let segment_ids = self.segment_manager.get_segment_ids().await;
        let mut total_vectors = 0usize;
        let mut doc_offset = 0u32;

        for id_str in &segment_ids {
            if let Some(segment_id) = SegmentId::from_hex(id_str)
                && let Ok(reader) = SegmentReader::open(
                    self.directory.as_ref(),
                    segment_id,
                    Arc::clone(&self.schema),
                    doc_offset,
                    self.config.term_cache_blocks,
                )
                .await
            {
                // Count vectors from Flat indexes
                for index in reader.vector_indexes().values() {
                    if let crate::segment::VectorIndex::Flat(flat_data) = index {
                        total_vectors += flat_data.vectors.len();
                    }
                }
                doc_offset += reader.meta().num_docs;
            }
        }

        // Update total in metadata
        {
            let mut meta = self.metadata.lock().await;
            meta.total_vectors = total_vectors;
        }

        // Check if any field should be built
        let should_build = {
            let meta = self.metadata.lock().await;
            dense_fields.iter().any(|(field, config)| {
                let threshold = config.build_threshold.unwrap_or(1000);
                meta.should_build_field(field.0, threshold)
            })
        };

        if should_build {
            log::info!(
                "Threshold crossed ({} vectors), auto-triggering vector index build",
                total_vectors
            );
            self.build_vector_index().await?;
        }

        Ok(())
    }

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

        // Update segment list via segment manager
        {
            let segment_ids_arc = self.segment_manager.segment_ids();
            let mut segment_ids = segment_ids_arc.lock().await;
            segment_ids.clear();
            segment_ids.push(new_segment_id.to_hex());
        }

        // Update and save metadata
        let segment_ids = self.segment_manager.get_segment_ids().await;
        {
            let mut meta = self.metadata.lock().await;
            meta.segments = segment_ids;
            meta.save(self.directory.as_ref()).await?;
        }

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
        // First commit all pending documents (waits for completion)
        self.commit().await?;
        // Then merge all segments
        self.do_merge().await
    }

    /// Build vector index from accumulated Flat vectors (trains ONCE)
    ///
    /// This trains centroids/codebooks from ALL vectors across all segments.
    /// Training happens only ONCE - subsequent calls are no-ops if already built.
    ///
    /// **Note:** This is auto-triggered by `commit()` when threshold is crossed.
    /// You typically don't need to call this manually.
    ///
    /// The process:
    /// 1. Check if already built (skip if so)
    /// 2. Collect all vectors from all segments
    /// 3. Train centroids/codebooks based on schema's index_type
    /// 4. Update metadata to mark as built (prevents re-training)
    pub async fn build_vector_index(&self) -> Result<()> {
        use crate::dsl::{FieldType, VectorIndexType};

        // Find dense vector fields that need ANN indexes
        let dense_fields: Vec<(Field, crate::dsl::DenseVectorConfig)> = self
            .schema
            .fields()
            .filter_map(|(field, entry)| {
                if entry.field_type == FieldType::DenseVector && entry.indexed {
                    entry
                        .dense_vector_config
                        .as_ref()
                        .filter(|c| !c.is_flat())
                        .map(|c| (field, c.clone()))
                } else {
                    None
                }
            })
            .collect();

        if dense_fields.is_empty() {
            log::info!("No dense vector fields configured for ANN indexing");
            return Ok(());
        }

        // Check which fields need building (skip already built)
        let fields_to_build: Vec<_> = {
            let meta = self.metadata.lock().await;
            dense_fields
                .iter()
                .filter(|(field, _)| !meta.is_field_built(field.0))
                .cloned()
                .collect()
        };

        if fields_to_build.is_empty() {
            log::info!("All vector fields already built, skipping training");
            return Ok(());
        }

        let segment_ids = self.segment_manager.get_segment_ids().await;
        if segment_ids.is_empty() {
            return Ok(());
        }

        // Collect all vectors from all segments for fields that need building
        let mut all_vectors: rustc_hash::FxHashMap<u32, Vec<Vec<f32>>> =
            rustc_hash::FxHashMap::default();
        let mut doc_offset = 0u32;

        for id_str in &segment_ids {
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

            // Extract vectors from each Flat index
            for (field_id, index) in reader.vector_indexes() {
                // Only collect for fields we need to build
                if fields_to_build.iter().any(|(f, _)| f.0 == *field_id)
                    && let crate::segment::VectorIndex::Flat(flat_data) = index
                {
                    all_vectors
                        .entry(*field_id)
                        .or_default()
                        .extend(flat_data.vectors.iter().cloned());
                }
            }

            doc_offset += reader.meta().num_docs;
        }

        // Train centroids/codebooks for each field and store at index level
        for (field, config) in &fields_to_build {
            let field_id = field.0;
            if let Some(vectors) = all_vectors.get(&field_id) {
                if vectors.is_empty() {
                    continue;
                }

                let index_dim = config.index_dim();
                let num_vectors = vectors.len();
                let num_clusters = config.optimal_num_clusters(num_vectors);

                log::info!(
                    "Training vector index for field {} with {} vectors, {} clusters",
                    field_id,
                    num_vectors,
                    num_clusters
                );

                let centroids_filename = format!("field_{}_centroids.bin", field_id);
                let mut codebook_filename: Option<String> = None;

                match config.index_type {
                    VectorIndexType::IvfRaBitQ => {
                        // Train coarse centroids
                        let coarse_config =
                            crate::structures::CoarseConfig::new(index_dim, num_clusters);
                        let centroids =
                            crate::structures::CoarseCentroids::train(&coarse_config, vectors);

                        // Save centroids to index-level file
                        let centroids_path = std::path::Path::new(&centroids_filename);
                        let centroids_bytes = serde_json::to_vec(&centroids)
                            .map_err(|e| Error::Serialization(e.to_string()))?;
                        self.directory
                            .write(centroids_path, &centroids_bytes)
                            .await?;

                        log::info!(
                            "Saved IVF-RaBitQ centroids for field {} ({} clusters)",
                            field_id,
                            centroids.num_clusters
                        );
                    }
                    VectorIndexType::ScaNN => {
                        // Train coarse centroids
                        let coarse_config =
                            crate::structures::CoarseConfig::new(index_dim, num_clusters);
                        let centroids =
                            crate::structures::CoarseCentroids::train(&coarse_config, vectors);

                        // Train PQ codebook
                        let pq_config = crate::structures::PQConfig::new(index_dim);
                        let codebook = crate::structures::PQCodebook::train(pq_config, vectors, 10);

                        // Save centroids and codebook to index-level files
                        let centroids_path = std::path::Path::new(&centroids_filename);
                        let centroids_bytes = serde_json::to_vec(&centroids)
                            .map_err(|e| Error::Serialization(e.to_string()))?;
                        self.directory
                            .write(centroids_path, &centroids_bytes)
                            .await?;

                        codebook_filename = Some(format!("field_{}_codebook.bin", field_id));
                        let codebook_path =
                            std::path::Path::new(codebook_filename.as_ref().unwrap());
                        let codebook_bytes = serde_json::to_vec(&codebook)
                            .map_err(|e| Error::Serialization(e.to_string()))?;
                        self.directory.write(codebook_path, &codebook_bytes).await?;

                        log::info!(
                            "Saved ScaNN centroids and codebook for field {} ({} clusters)",
                            field_id,
                            centroids.num_clusters
                        );
                    }
                    _ => {
                        // RaBitQ or Flat - no pre-training needed
                        continue;
                    }
                }

                // Update metadata to mark this field as built (prevents re-training)
                {
                    let mut meta = self.metadata.lock().await;
                    meta.init_field(field_id, config.index_type);
                    meta.total_vectors = num_vectors;
                    meta.mark_field_built(
                        field_id,
                        num_vectors,
                        num_clusters,
                        centroids_filename,
                        codebook_filename,
                    );
                    meta.save(self.directory.as_ref()).await?;
                }
            }
        }

        log::info!("Vector index training complete. Rebuilding segments with ANN indexes...");

        // Rebuild segments with ANN indexes using trained structures
        self.rebuild_segments_with_ann().await?;

        Ok(())
    }

    /// Rebuild all segments with ANN indexes using trained centroids/codebooks
    async fn rebuild_segments_with_ann(&self) -> Result<()> {
        use crate::segment::{SegmentMerger, TrainedVectorStructures};

        let segment_ids = self.segment_manager.get_segment_ids().await;
        if segment_ids.is_empty() {
            return Ok(());
        }

        // Load trained structures from metadata
        let (trained_centroids, trained_codebooks) = {
            let meta = self.metadata.lock().await;
            meta.load_trained_structures(self.directory.as_ref()).await
        };

        if trained_centroids.is_empty() {
            log::info!("No trained structures to rebuild with");
            return Ok(());
        }

        let trained = TrainedVectorStructures {
            centroids: trained_centroids,
            codebooks: trained_codebooks,
        };

        // Load all segment readers
        let mut readers = Vec::new();
        let mut doc_offset = 0u32;

        for id_str in &segment_ids {
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

        // Merge all segments into one with ANN indexes
        let merger = SegmentMerger::new(Arc::clone(&self.schema));
        let new_segment_id = SegmentId::new();
        merger
            .merge_with_ann(self.directory.as_ref(), &readers, new_segment_id, &trained)
            .await?;

        // Update segment list
        {
            let segment_ids_arc = self.segment_manager.segment_ids();
            let mut segment_ids = segment_ids_arc.lock().await;
            let old_ids: Vec<String> = segment_ids.clone();
            segment_ids.clear();
            segment_ids.push(new_segment_id.to_hex());

            // Save metadata
            let mut meta = self.metadata.lock().await;
            meta.segments = segment_ids.clone();
            meta.save(self.directory.as_ref()).await?;

            // Delete old segments
            for id_str in old_ids {
                if let Some(segment_id) = SegmentId::from_hex(&id_str) {
                    let _ =
                        crate::segment::delete_segment(self.directory.as_ref(), segment_id).await;
                }
            }
        }

        log::info!("Segments rebuilt with ANN indexes");
        Ok(())
    }

    /// Get total vector count across all segments (for threshold checking)
    pub async fn total_vector_count(&self) -> usize {
        self.metadata.lock().await.total_vectors
    }

    /// Check if vector index has been built for a field
    pub async fn is_vector_index_built(&self, field: Field) -> bool {
        self.metadata.lock().await.is_field_built(field.0)
    }

    /// Rebuild vector index by retraining centroids/codebooks
    ///
    /// Use this when:
    /// - Significant new data has been added and you want better centroids
    /// - You want to change the number of clusters
    /// - The vector distribution has changed significantly
    ///
    /// This resets the Built state to Flat, then triggers a fresh training.
    pub async fn rebuild_vector_index(&self) -> Result<()> {
        use crate::dsl::FieldType;

        // Find all dense vector fields
        let dense_fields: Vec<Field> = self
            .schema
            .fields()
            .filter_map(|(field, entry)| {
                if entry.field_type == FieldType::DenseVector && entry.indexed {
                    Some(field)
                } else {
                    None
                }
            })
            .collect();

        if dense_fields.is_empty() {
            return Ok(());
        }

        // Reset all fields to Flat state (forces rebuild)
        {
            let mut meta = self.metadata.lock().await;
            for field in &dense_fields {
                if let Some(field_meta) = meta.vector_fields.get_mut(&field.0) {
                    field_meta.state = super::VectorIndexState::Flat;
                    // Delete old centroids/codebook files
                    if let Some(ref centroids_file) = field_meta.centroids_file {
                        let _ = self
                            .directory
                            .delete(std::path::Path::new(centroids_file))
                            .await;
                    }
                    if let Some(ref codebook_file) = field_meta.codebook_file {
                        let _ = self
                            .directory
                            .delete(std::path::Path::new(codebook_file))
                            .await;
                    }
                    field_meta.centroids_file = None;
                    field_meta.codebook_file = None;
                }
            }
            meta.save(self.directory.as_ref()).await?;
        }

        log::info!("Reset vector index state to Flat, triggering rebuild...");

        // Now build fresh
        self.build_vector_index().await
    }
}
