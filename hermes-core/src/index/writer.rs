//! IndexWriter - async document indexing with parallel segment building
//!
//! This module is only compiled with the "native" feature.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashMap;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::DocId;
use crate::directories::DirectoryWriter;
use crate::dsl::{Document, Field, Schema};
use crate::error::{Error, Result};
use crate::segment::{
    SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentMerger, SegmentReader,
};
use crate::tokenizer::BoxedTokenizer;

use super::IndexConfig;

/// Message sent to worker tasks
enum WorkerMessage {
    /// A document to index
    Document(Document),
    /// Signal to flush current builder and respond when done
    Flush(oneshot::Sender<()>),
}

/// Async IndexWriter for adding documents and committing segments
///
/// Features:
/// - Queue-based parallel indexing with worker tasks
/// - Streams documents to disk immediately (no in-memory document storage)
/// - Uses string interning for terms (reduced allocations)
/// - Uses hashbrown HashMap (faster than BTreeMap)
///
/// **Architecture:**
/// - `add_document()` sends to per-worker unbounded channels (non-blocking)
/// - Round-robin distribution across workers - no mutex contention
/// - Each worker owns a SegmentBuilder and flushes when memory threshold is reached
///
/// **State management:**
/// - Building segments: Managed here (pending_builds)
/// - Committed segments + metadata: Managed by SegmentManager (sole owner of metadata.json)
pub struct IndexWriter<D: DirectoryWriter + 'static> {
    pub(super) directory: Arc<D>,
    pub(super) schema: Arc<Schema>,
    pub(super) config: IndexConfig,
    #[allow(dead_code)] // Used for creating new builders in worker_state
    builder_config: SegmentBuilderConfig,
    tokenizers: FxHashMap<Field, BoxedTokenizer>,
    /// Per-worker channel senders - round-robin distribution
    worker_senders: Vec<mpsc::UnboundedSender<WorkerMessage>>,
    /// Round-robin counter for worker selection
    next_worker: AtomicUsize,
    /// Worker task handles - kept alive to prevent premature shutdown
    #[allow(dead_code)]
    workers: Vec<JoinHandle<()>>,
    /// Shared state for workers
    #[allow(dead_code)]
    worker_state: Arc<WorkerState<D>>,
    /// Segment manager - owns metadata.json, handles segments and background merging
    pub(super) segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Channel receiver for completed segment IDs
    segment_id_receiver: AsyncMutex<mpsc::UnboundedReceiver<String>>,
    /// Count of in-flight background builds
    pending_builds: Arc<AtomicUsize>,
    /// Global memory usage across all builders (bytes)
    #[allow(dead_code)]
    global_memory_bytes: Arc<AtomicUsize>,
}

/// Shared state for worker tasks
struct WorkerState<D: DirectoryWriter + 'static> {
    directory: Arc<D>,
    schema: Arc<Schema>,
    config: IndexConfig,
    builder_config: SegmentBuilderConfig,
    tokenizers: FxHashMap<Field, BoxedTokenizer>,
    segment_id_sender: mpsc::UnboundedSender<String>,
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
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

        // Create channel for background builds to report completed segment IDs
        let (segment_id_sender, segment_id_receiver) = mpsc::unbounded_channel();

        // Initialize metadata with schema
        let metadata = super::IndexMetadata::new((*schema).clone());

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

        let pending_builds = Arc::new(AtomicUsize::new(0));
        let global_memory_bytes = Arc::new(AtomicUsize::new(0));

        // Create shared worker state
        let worker_state = Arc::new(WorkerState {
            directory: Arc::clone(&directory),
            schema: Arc::clone(&schema),
            config: config.clone(),
            builder_config: builder_config.clone(),
            tokenizers: FxHashMap::default(),
            segment_id_sender,
            segment_manager: Arc::clone(&segment_manager),
            pending_builds: Arc::clone(&pending_builds),
        });

        // Create per-worker unbounded channels and spawn workers
        let num_workers = config.num_indexing_threads.max(1);
        let mut worker_senders = Vec::with_capacity(num_workers);
        let mut workers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let (tx, rx) = mpsc::unbounded_channel::<WorkerMessage>();
            worker_senders.push(tx);

            let state = Arc::clone(&worker_state);
            let handle = tokio::spawn(async move {
                Self::worker_loop(state, rx).await;
            });
            workers.push(handle);
        }

        Ok(Self {
            directory,
            schema,
            config,
            builder_config,
            tokenizers: FxHashMap::default(),
            worker_senders,
            next_worker: AtomicUsize::new(0),
            workers,
            worker_state,
            segment_manager,
            segment_id_receiver: AsyncMutex::new(segment_id_receiver),
            pending_builds,
            global_memory_bytes,
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

        // Load unified metadata (includes schema)
        let metadata = super::IndexMetadata::load(directory.as_ref()).await?;
        let schema = Arc::new(metadata.schema.clone());

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

        let pending_builds = Arc::new(AtomicUsize::new(0));
        let global_memory_bytes = Arc::new(AtomicUsize::new(0));

        // Create shared worker state
        let worker_state = Arc::new(WorkerState {
            directory: Arc::clone(&directory),
            schema: Arc::clone(&schema),
            config: config.clone(),
            builder_config: builder_config.clone(),
            tokenizers: FxHashMap::default(),
            segment_id_sender,
            segment_manager: Arc::clone(&segment_manager),
            pending_builds: Arc::clone(&pending_builds),
        });

        // Create per-worker unbounded channels and spawn workers
        let num_workers = config.num_indexing_threads.max(1);
        let mut worker_senders = Vec::with_capacity(num_workers);
        let mut workers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let (tx, rx) = mpsc::unbounded_channel::<WorkerMessage>();
            worker_senders.push(tx);

            let state = Arc::clone(&worker_state);
            let handle = tokio::spawn(async move {
                Self::worker_loop(state, rx).await;
            });
            workers.push(handle);
        }

        Ok(Self {
            directory,
            schema,
            config,
            builder_config,
            tokenizers: FxHashMap::default(),
            worker_senders,
            next_worker: AtomicUsize::new(0),
            workers,
            worker_state,
            segment_manager,
            segment_id_receiver: AsyncMutex::new(segment_id_receiver),
            pending_builds,
            global_memory_bytes,
        })
    }

    /// Create an IndexWriter from an existing Index
    ///
    /// This shares the SegmentManager with the Index, ensuring consistent
    /// segment lifecycle management.
    pub fn from_index(index: &super::Index<D>) -> Self {
        let segment_manager = Arc::clone(&index.segment_manager);
        let directory = Arc::clone(&index.directory);
        let schema = Arc::clone(&index.schema);
        let config = index.config.clone();
        let builder_config = crate::segment::SegmentBuilderConfig::default();

        // Create channel for background builds
        let (segment_id_sender, segment_id_receiver) = tokio::sync::mpsc::unbounded_channel();

        let pending_builds = Arc::new(AtomicUsize::new(0));
        let global_memory_bytes = Arc::new(AtomicUsize::new(0));

        // Create shared worker state
        let worker_state = Arc::new(WorkerState {
            directory: Arc::clone(&directory),
            schema: Arc::clone(&schema),
            config: config.clone(),
            builder_config: builder_config.clone(),
            tokenizers: FxHashMap::default(),
            segment_id_sender,
            segment_manager: Arc::clone(&segment_manager),
            pending_builds: Arc::clone(&pending_builds),
        });

        // Create per-worker channels and spawn workers
        let num_workers = config.num_indexing_threads.max(1);
        let mut worker_senders = Vec::with_capacity(num_workers);
        let mut workers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<WorkerMessage>();
            worker_senders.push(tx);

            let state = Arc::clone(&worker_state);
            let handle = tokio::spawn(async move {
                Self::worker_loop(state, rx).await;
            });
            workers.push(handle);
        }

        Self {
            directory,
            schema,
            config,
            builder_config,
            tokenizers: FxHashMap::default(),
            worker_senders,
            next_worker: AtomicUsize::new(0),
            workers,
            worker_state,
            segment_manager,
            segment_id_receiver: AsyncMutex::new(segment_id_receiver),
            pending_builds,
            global_memory_bytes,
        }
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Set tokenizer for a field
    pub fn set_tokenizer<T: crate::tokenizer::Tokenizer>(&mut self, field: Field, tokenizer: T) {
        self.tokenizers.insert(field, Box::new(tokenizer));
    }

    /// Add a document to the indexing queue
    ///
    /// Documents are sent to per-worker unbounded channels.
    /// This is O(1) and never blocks - returns immediately.
    /// Workers handle the actual indexing in parallel.
    pub fn add_document(&self, doc: Document) -> Result<DocId> {
        // Round-robin select worker
        let idx = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.worker_senders.len();
        self.worker_senders[idx]
            .send(WorkerMessage::Document(doc))
            .map_err(|_| Error::Internal("Document channel closed".into()))?;
        Ok(0)
    }

    /// Add multiple documents to the indexing queue
    ///
    /// Documents are distributed round-robin to workers.
    /// Returns immediately - never blocks.
    pub fn add_documents(&self, documents: Vec<Document>) -> Result<usize> {
        let num_workers = self.worker_senders.len();
        let count = documents.len();
        let base = self.next_worker.fetch_add(count, Ordering::Relaxed);
        for (i, doc) in documents.into_iter().enumerate() {
            let idx = (base + i) % num_workers;
            let _ = self.worker_senders[idx].send(WorkerMessage::Document(doc));
        }
        Ok(count)
    }

    /// Worker loop - polls messages from its own channel and indexes documents
    async fn worker_loop(
        state: Arc<WorkerState<D>>,
        mut receiver: mpsc::UnboundedReceiver<WorkerMessage>,
    ) {
        let mut builder: Option<SegmentBuilder> = None;
        let mut doc_count = 0u32;

        loop {
            // Receive from own channel - no mutex contention
            let msg = receiver.recv().await;

            let Some(msg) = msg else {
                // Channel closed - flush remaining docs and exit
                if let Some(b) = builder.take()
                    && b.num_docs() > 0
                {
                    Self::spawn_segment_build(&state, b);
                }
                return;
            };

            match msg {
                WorkerMessage::Document(doc) => {
                    // Initialize builder if needed
                    if builder.is_none() {
                        match SegmentBuilder::new(
                            (*state.schema).clone(),
                            state.builder_config.clone(),
                        ) {
                            Ok(mut b) => {
                                for (field, tokenizer) in &state.tokenizers {
                                    b.set_tokenizer(*field, tokenizer.clone_box());
                                }
                                builder = Some(b);
                            }
                            Err(e) => {
                                eprintln!("Failed to create segment builder: {:?}", e);
                                continue;
                            }
                        }
                    }

                    // Index the document
                    let b = builder.as_mut().unwrap();
                    if let Err(e) = b.add_document(doc) {
                        eprintln!("Failed to index document: {:?}", e);
                        continue;
                    }

                    doc_count += 1;

                    // Check memory periodically
                    // Use smaller interval for small memory limits (for testing)
                    let per_worker_limit = state.config.max_indexing_memory_bytes
                        / state.config.num_indexing_threads.max(1);
                    let check_interval = if per_worker_limit < 1024 * 1024 {
                        1
                    } else {
                        100
                    };

                    if doc_count.is_multiple_of(check_interval) {
                        let builder_memory = b.stats().estimated_memory_bytes;

                        if builder_memory >= per_worker_limit {
                            let full_builder = builder.take().unwrap();
                            Self::spawn_segment_build(&state, full_builder);
                            doc_count = 0;
                        }
                    }
                }
                WorkerMessage::Flush(respond) => {
                    // Flush current builder if it has documents
                    if let Some(b) = builder.take()
                        && b.num_docs() > 0
                    {
                        Self::spawn_segment_build(&state, b);
                    }
                    doc_count = 0;
                    // Signal that flush is complete for this worker
                    let _ = respond.send(());
                }
            }
        }
    }
    fn spawn_segment_build(state: &Arc<WorkerState<D>>, builder: SegmentBuilder) {
        let directory = Arc::clone(&state.directory);
        let segment_id = SegmentId::new();
        let segment_hex = segment_id.to_hex();
        let sender = state.segment_id_sender.clone();
        let segment_manager = Arc::clone(&state.segment_manager);
        let pending_builds = Arc::clone(&state.pending_builds);

        pending_builds.fetch_add(1, Ordering::SeqCst);

        tokio::spawn(async move {
            match builder.build(directory.as_ref(), segment_id).await {
                Ok(_) => {
                    let _ = segment_manager.register_segment(segment_hex.clone()).await;
                }
                Err(e) => {
                    eprintln!("Background segment build failed: {:?}", e);
                }
            }
            // Always send to channel and decrement - even on failure
            // This ensures commit() doesn't hang waiting for messages
            let _ = sender.send(segment_hex);
            pending_builds.fetch_sub(1, Ordering::SeqCst);
        });
    }

    /// Collect any completed segment IDs from the channel (non-blocking)
    async fn collect_completed_segments(&self) {
        let mut receiver = self.segment_id_receiver.lock().await;
        while receiver.try_recv().is_ok() {
            // Segment already registered by spawn_segment_build
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

    /// Get the segment tracker for sharing with readers
    /// This allows readers to acquire snapshots that prevent segment deletion
    pub fn tracker(&self) -> std::sync::Arc<crate::segment::SegmentTracker> {
        self.segment_manager.tracker()
    }

    /// Acquire a snapshot of current segments for reading
    /// The snapshot holds references - segments won't be deleted while snapshot exists
    pub async fn acquire_snapshot(&self) -> crate::segment::SegmentSnapshot<D> {
        self.segment_manager.acquire_snapshot().await
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

    /// Flush all workers - signals them to build their current segments
    ///
    /// Sends flush signals to all workers and waits for them to acknowledge.
    /// Workers continue running and can accept new documents after flush.
    pub async fn flush(&self) -> Result<()> {
        // Send flush signal to each worker's channel
        let mut responses = Vec::with_capacity(self.worker_senders.len());

        for sender in &self.worker_senders {
            let (tx, rx) = oneshot::channel();
            if sender.send(WorkerMessage::Flush(tx)).is_err() {
                // Channel closed, worker may have exited
                continue;
            }
            responses.push(rx);
        }

        // Wait for all workers to acknowledge flush
        for rx in responses {
            let _ = rx.await;
        }

        // Collect any completed segments
        self.collect_completed_segments().await;

        Ok(())
    }

    /// Commit all pending segments to disk and wait for completion
    ///
    /// This flushes workers and waits for ALL background builds to complete.
    /// Provides durability guarantees - all data is persisted.
    ///
    /// **Auto-triggers vector index build** when threshold is crossed for any field.
    pub async fn commit(&self) -> Result<()> {
        // Flush all workers first
        self.flush().await?;

        // Wait for all pending builds to complete
        let mut receiver = self.segment_id_receiver.lock().await;
        while self.pending_builds.load(Ordering::SeqCst) > 0 {
            if receiver.recv().await.is_none() {
                break; // Channel closed
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
        // Wait for any background merges to complete (avoid race with segment deletion)
        self.wait_for_merges().await;
        // Then merge all segments
        self.do_merge().await
    }

    // Vector index methods (build_vector_index, rebuild_vector_index, etc.)
    // are implemented in vector_builder.rs
}
