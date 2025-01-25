//! IndexWriter - async document indexing with parallel segment building
//!
//! This module is only compiled with the "native" feature.

use std::path::Path;
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
/// - `add_document()` sends to a bounded channel (fast, non-blocking)
/// - Worker tasks poll the channel and index documents in parallel
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
    /// Channel sender for worker messages - add_document() sends here
    msg_sender: mpsc::Sender<WorkerMessage>,
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

        // Create message channel - bounded to apply backpressure
        let num_workers = config.num_indexing_threads.max(1);
        let (msg_sender, msg_receiver) = mpsc::channel::<WorkerMessage>(num_workers * 1000);
        let msg_receiver = Arc::new(AsyncMutex::new(msg_receiver));

        // Spawn worker tasks
        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let state = Arc::clone(&worker_state);
            let receiver = Arc::clone(&msg_receiver);
            let handle = tokio::spawn(async move {
                Self::worker_loop(state, receiver).await;
            });
            workers.push(handle);
        }

        Ok(Self {
            directory,
            schema,
            config,
            builder_config,
            tokenizers: FxHashMap::default(),
            msg_sender,
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

        // Read schema
        let schema_slice = directory.open_read(Path::new("schema.json")).await?;
        let schema_bytes = schema_slice.read_bytes().await?;
        let schema: Schema = serde_json::from_slice(schema_bytes.as_slice())
            .map_err(|e| Error::Serialization(e.to_string()))?;
        let schema = Arc::new(schema);

        // Load unified metadata
        let metadata = super::IndexMetadata::load(directory.as_ref()).await?;

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

        // Create message channel - bounded to apply backpressure
        let num_workers = config.num_indexing_threads.max(1);
        let (msg_sender, msg_receiver) = mpsc::channel::<WorkerMessage>(num_workers * 1000);
        let msg_receiver = Arc::new(AsyncMutex::new(msg_receiver));

        // Spawn worker tasks
        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let state = Arc::clone(&worker_state);
            let receiver = Arc::clone(&msg_receiver);
            let handle = tokio::spawn(async move {
                Self::worker_loop(state, receiver).await;
            });
            workers.push(handle);
        }

        Ok(Self {
            directory,
            schema,
            config,
            builder_config,
            tokenizers: FxHashMap::default(),
            msg_sender,
            workers,
            worker_state,
            segment_manager,
            segment_id_receiver: AsyncMutex::new(segment_id_receiver),
            pending_builds,
            global_memory_bytes,
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

    /// Add a document to the indexing queue
    ///
    /// Documents are sent to worker tasks via a bounded channel.
    /// This is fast and non-blocking (unless backpressure kicks in).
    /// Workers handle the actual indexing in parallel.
    pub async fn add_document(&self, doc: Document) -> Result<DocId> {
        // Send to worker queue - applies backpressure if queue is full
        self.msg_sender
            .send(WorkerMessage::Document(doc))
            .await
            .map_err(|_| Error::Internal("Document channel closed".into()))?;

        // Note: We don't have the actual DocId here since workers assign it
        // Return a placeholder - callers typically don't need the exact ID
        Ok(0)
    }

    /// Documents are sent to worker tasks for parallel processing.
    /// Returns the count of documents successfully queued.
    pub async fn add_documents(&self, documents: Vec<Document>) -> Result<usize> {
        let mut queued = 0;
        for doc in documents {
            if self
                .msg_sender
                .send(WorkerMessage::Document(doc))
                .await
                .is_ok()
            {
                queued += 1;
            }
        }
        Ok(queued)
    }

    /// Worker loop - polls messages from queue and indexes documents
    async fn worker_loop(
        state: Arc<WorkerState<D>>,
        receiver: Arc<AsyncMutex<mpsc::Receiver<WorkerMessage>>>,
    ) {
        let mut builder: Option<SegmentBuilder> = None;
        let mut doc_count = 0u32;

        loop {
            // Try to receive a message
            let msg = {
                let mut rx = receiver.lock().await;
                rx.recv().await
            };

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
        // Send flush signal to each worker and collect response channels
        let num_workers = self.config.num_indexing_threads.max(1);
        let mut responses = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let (tx, rx) = oneshot::channel();
            if self
                .msg_sender
                .send(WorkerMessage::Flush(tx))
                .await
                .is_err()
            {
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
        // Then merge all segments
        self.do_merge().await
    }

    // Vector index methods (build_vector_index, rebuild_vector_index, etc.)
    // are implemented in vector_builder.rs
}
