//! IndexWriter — async document indexing with parallel segment building.
//!
//! This module is only compiled with the "native" feature.
//!
//! # Architecture
//!
//! ```text
//! add_document() ──try_send──► [shared bounded MPMC] ◄──recv── worker 0
//!                                                     ◄──recv── worker 1
//!                                                     ◄──recv── worker N
//! ```
//!
//! - **Shared MPMC queue** (`async_channel`): all workers compete for documents.
//!   Busy workers (building segments) naturally stop pulling; free workers pick up slack.
//! - **Zero-copy pipeline**: `Document` is moved (never cloned) through every stage:
//!   `add_document()` → channel → `recv_blocking()` → `SegmentBuilder::add_document()`.
//! - `add_document` returns `QueueFull` when the queue is at capacity.
//! - **Workers are OS threads**: CPU-intensive work (tokenization, posting list building)
//!   runs on dedicated threads, never blocking the tokio async runtime.
//!   Async I/O (segment file writes) is bridged via `Handle::block_on()`.
//! - **Fixed per-worker memory budget**: `max_indexing_memory_bytes / num_workers`.
//! - **Two-phase commit**:
//!   1. `prepare_commit()` — closes queue, drains workers, segments written to disk.
//!      Returns a `PreparedCommit` guard. No new documents accepted until resolved.
//!   2. `PreparedCommit::commit()` — registers segments in metadata, respawns workers.
//!   3. `PreparedCommit::abort()` — discards prepared segments, respawns workers.
//!   4. `commit()` — convenience: `prepare_commit().await?.commit().await`.
//!
//! Since `prepare_commit`/`commit` take `&mut self`, Rust’s borrow checker
//! guarantees no concurrent `add_document` calls during the commit window.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::directories::DirectoryWriter;
use crate::dsl::{Document, Field, Schema};
use crate::error::{Error, Result};
use crate::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId};
use crate::tokenizer::BoxedTokenizer;

use super::IndexConfig;

/// Total pipeline capacity (in documents).
const PIPELINE_MAX_SIZE_IN_DOCS: usize = 10_000;

/// Async IndexWriter for adding documents and committing segments.
///
/// **Backpressure:** `add_document()` is sync, O(1). Returns `Error::QueueFull`
/// when the shared queue is at capacity — caller must back off.
///
/// **Two-phase commit:**
/// - `prepare_commit()` → `PreparedCommit::commit()` or `PreparedCommit::abort()`
/// - `commit()` is a convenience that does both phases.
/// - Between prepare and commit, the caller can do external work (WAL, sync, etc.)
///   knowing that abort is possible if something fails.
/// - Dropping `PreparedCommit` without calling commit/abort auto-aborts.
pub struct IndexWriter<D: DirectoryWriter + 'static> {
    pub(super) directory: Arc<D>,
    pub(super) schema: Arc<Schema>,
    pub(super) config: IndexConfig,
    /// MPMC sender — `try_send(&self)` is thread-safe, no lock needed.
    /// Replaced on each commit cycle.
    doc_sender: async_channel::Sender<Document>,
    /// Worker OS thread handles — replaced on each commit cycle.
    workers: Vec<std::thread::JoinHandle<()>>,
    /// Shared worker state (immutable config + mutable segment output)
    worker_state: Arc<WorkerState<D>>,
    /// Segment manager — owns metadata.json, handles segments and background merging
    pub(super) segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Segments flushed to disk but not yet registered in metadata
    flushed_segments: Vec<(String, u32)>,
}

/// Shared state for worker threads.
struct WorkerState<D: DirectoryWriter + 'static> {
    directory: Arc<D>,
    schema: Arc<Schema>,
    builder_config: SegmentBuilderConfig,
    tokenizers: parking_lot::RwLock<FxHashMap<Field, BoxedTokenizer>>,
    /// Fixed per-worker memory budget (bytes). When a builder exceeds this, segment is built.
    memory_budget_per_worker: usize,
    /// Segment manager — workers read trained structures from its ArcSwap (lock-free).
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Segments built by workers, collected by `prepare_commit()`. Sync mutex for sub-μs push.
    built_segments: parking_lot::Mutex<Vec<(String, u32)>>,
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
        let metadata = super::IndexMetadata::new((*schema).clone());

        let segment_manager = Arc::new(crate::merge::SegmentManager::new(
            Arc::clone(&directory),
            Arc::clone(&schema),
            metadata,
            config.merge_policy.clone_box(),
            config.term_cache_blocks,
            config.max_concurrent_merges,
        ));
        segment_manager.update_metadata(|_| {}).await?;

        Ok(Self::new_with_parts(
            directory,
            schema,
            config,
            builder_config,
            segment_manager,
        ))
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
        let metadata = super::IndexMetadata::load(directory.as_ref()).await?;
        let schema = Arc::new(metadata.schema.clone());

        let segment_manager = Arc::new(crate::merge::SegmentManager::new(
            Arc::clone(&directory),
            Arc::clone(&schema),
            metadata,
            config.merge_policy.clone_box(),
            config.term_cache_blocks,
            config.max_concurrent_merges,
        ));
        segment_manager.load_and_publish_trained().await;

        Ok(Self::new_with_parts(
            directory,
            schema,
            config,
            builder_config,
            segment_manager,
        ))
    }

    /// Create an IndexWriter from an existing Index.
    /// Shares the SegmentManager for consistent segment lifecycle management.
    pub fn from_index(index: &super::Index<D>) -> Self {
        Self::new_with_parts(
            Arc::clone(&index.directory),
            Arc::clone(&index.schema),
            index.config.clone(),
            SegmentBuilderConfig::default(),
            Arc::clone(&index.segment_manager),
        )
    }

    // ========================================================================
    // Construction + pipeline management
    // ========================================================================

    /// Common construction: creates worker state, spawns workers, assembles `Self`.
    fn new_with_parts(
        directory: Arc<D>,
        schema: Arc<Schema>,
        config: IndexConfig,
        builder_config: SegmentBuilderConfig,
        segment_manager: Arc<crate::merge::SegmentManager<D>>,
    ) -> Self {
        let num_workers = config.num_indexing_threads.max(1);
        let worker_state = Arc::new(WorkerState {
            directory: Arc::clone(&directory),
            schema: Arc::clone(&schema),
            builder_config,
            tokenizers: parking_lot::RwLock::new(FxHashMap::default()),
            memory_budget_per_worker: config.max_indexing_memory_bytes / num_workers,
            segment_manager: Arc::clone(&segment_manager),
            built_segments: parking_lot::Mutex::new(Vec::new()),
        });
        let (doc_sender, workers) = Self::spawn_workers(&worker_state, num_workers);

        Self {
            directory,
            schema,
            config,
            doc_sender,
            workers,
            worker_state,
            segment_manager,
            flushed_segments: Vec::new(),
        }
    }

    fn spawn_workers(
        worker_state: &Arc<WorkerState<D>>,
        num_workers: usize,
    ) -> (
        async_channel::Sender<Document>,
        Vec<std::thread::JoinHandle<()>>,
    ) {
        let (sender, receiver) = async_channel::bounded(PIPELINE_MAX_SIZE_IN_DOCS);
        let handle = tokio::runtime::Handle::current();
        let mut workers = Vec::with_capacity(num_workers);
        for i in 0..num_workers {
            let state = Arc::clone(worker_state);
            let rx = receiver.clone();
            let rt = handle.clone();
            workers.push(
                std::thread::Builder::new()
                    .name(format!("index-worker-{}", i))
                    .spawn(move || Self::worker_loop(state, rx, rt))
                    .expect("failed to spawn index worker thread"),
            );
        }
        (sender, workers)
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Set tokenizer for a field.
    /// Propagated to worker threads — takes effect for the next SegmentBuilder they create.
    pub fn set_tokenizer<T: crate::tokenizer::Tokenizer>(&mut self, field: Field, tokenizer: T) {
        self.worker_state
            .tokenizers
            .write()
            .insert(field, Box::new(tokenizer));
    }

    /// Add a document to the indexing queue (sync, O(1), lock-free).
    ///
    /// `Document` is moved into the channel (zero-copy). Workers compete to pull it.
    /// Returns `Error::QueueFull` when the queue is at capacity — caller must back off.
    pub fn add_document(&self, doc: Document) -> Result<()> {
        self.doc_sender.try_send(doc).map_err(|e| match e {
            async_channel::TrySendError::Full(_) => Error::QueueFull,
            async_channel::TrySendError::Closed(_) => {
                Error::Internal("Document channel closed".into())
            }
        })
    }

    /// Add multiple documents to the indexing queue.
    ///
    /// Returns the number of documents successfully queued. Stops at the first
    /// `QueueFull` and returns the count queued so far.
    pub fn add_documents(&self, documents: Vec<Document>) -> Result<usize> {
        let total = documents.len();
        for (i, doc) in documents.into_iter().enumerate() {
            match self.add_document(doc) {
                Ok(()) => {}
                Err(Error::QueueFull) => return Ok(i),
                Err(e) => return Err(e),
            }
        }
        Ok(total)
    }

    // ========================================================================
    // Worker loop
    // ========================================================================

    /// Worker loop — runs on a dedicated OS thread.
    ///
    /// Pulls documents from the shared MPMC queue (blocking recv), indexes them
    /// (CPU-intensive: tokenization, posting list updates), and builds segments
    /// inline when the memory budget is exceeded.
    ///
    /// Async I/O (segment file writes) is bridged via `Handle::block_on()`.
    /// Exits when the channel is closed (prepare_commit closes the sender).
    fn worker_loop(
        state: Arc<WorkerState<D>>,
        receiver: async_channel::Receiver<Document>,
        handle: tokio::runtime::Handle,
    ) {
        let mut builder: Option<SegmentBuilder> = None;

        while let Ok(doc) = receiver.recv_blocking() {
            // Initialize builder if needed
            if builder.is_none() {
                match SegmentBuilder::new(Arc::clone(&state.schema), state.builder_config.clone()) {
                    Ok(mut b) => {
                        for (field, tokenizer) in state.tokenizers.read().iter() {
                            b.set_tokenizer(*field, tokenizer.clone_box());
                        }
                        builder = Some(b);
                    }
                    Err(e) => {
                        log::error!("Failed to create segment builder: {:?}", e);
                        continue;
                    }
                }
            }

            let b = builder.as_mut().unwrap();
            if let Err(e) = b.add_document(doc) {
                log::error!("Failed to index document: {:?}", e);
                continue;
            }

            let builder_memory = b.estimated_memory_bytes();

            if b.num_docs() & 0x3FFF == 0 {
                log::debug!(
                    "[indexing] docs={}, memory={:.2} MB, budget={:.2} MB",
                    b.num_docs(),
                    builder_memory as f64 / (1024.0 * 1024.0),
                    state.memory_budget_per_worker as f64 / (1024.0 * 1024.0)
                );
            }

            // Require minimum 100 docs before flushing to avoid tiny segments
            const MIN_DOCS_BEFORE_FLUSH: u32 = 100;

            if builder_memory >= state.memory_budget_per_worker
                && b.num_docs() >= MIN_DOCS_BEFORE_FLUSH
            {
                log::info!(
                    "[indexing] memory budget reached, building segment: \
                     docs={}, memory={:.2} MB, budget={:.2} MB",
                    b.num_docs(),
                    builder_memory as f64 / (1024.0 * 1024.0),
                    state.memory_budget_per_worker as f64 / (1024.0 * 1024.0),
                );
                let full_builder = builder.take().unwrap();
                Self::build_segment_inline(&state, full_builder, &handle);
            }
        }

        // Channel closed — build remaining docs
        if let Some(b) = builder.take()
            && b.num_docs() > 0
        {
            Self::build_segment_inline(&state, b, &handle);
        }
    }

    /// Build a segment on the worker thread. Uses `Handle::block_on()` to bridge
    /// into async context for I/O (streaming writers). CPU work (rayon) stays on
    /// the worker thread / rayon pool.
    fn build_segment_inline(
        state: &WorkerState<D>,
        builder: SegmentBuilder,
        handle: &tokio::runtime::Handle,
    ) {
        let segment_id = SegmentId::new();
        let segment_hex = segment_id.to_hex();
        let trained = state.segment_manager.trained();
        let doc_count = builder.num_docs();
        let build_start = std::time::Instant::now();

        log::info!(
            "[segment_build] segment_id={} doc_count={} ann={}",
            segment_hex,
            doc_count,
            trained.is_some()
        );

        match handle.block_on(builder.build(
            state.directory.as_ref(),
            segment_id,
            trained.as_deref(),
        )) {
            Ok(meta) if meta.num_docs > 0 => {
                let duration_ms = build_start.elapsed().as_millis() as u64;
                log::info!(
                    "[segment_build_done] segment_id={} doc_count={} duration_ms={}",
                    segment_hex,
                    meta.num_docs,
                    duration_ms,
                );
                state
                    .built_segments
                    .lock()
                    .push((segment_hex, meta.num_docs));
            }
            Ok(_) => {}
            Err(e) => {
                log::error!(
                    "[segment_build_failed] segment_id={} error={:?}",
                    segment_hex,
                    e
                );
            }
        }
    }

    // ========================================================================
    // Public API — commit, merge, etc.
    // ========================================================================

    /// Check merge policy and spawn a background merge if needed.
    pub async fn maybe_merge(&self) {
        self.segment_manager.maybe_merge().await;
    }

    /// Wait for the in-flight background merge to complete (if any).
    pub async fn wait_for_merging_thread(&self) {
        self.segment_manager.wait_for_merging_thread().await;
    }

    /// Wait for all eligible merges to complete, including cascading merges.
    pub async fn wait_for_all_merges(&self) {
        self.segment_manager.wait_for_all_merges().await;
    }

    /// Get the segment tracker for sharing with readers.
    pub fn tracker(&self) -> std::sync::Arc<crate::segment::SegmentTracker> {
        self.segment_manager.tracker()
    }

    /// Acquire a snapshot of current segments for reading.
    pub async fn acquire_snapshot(&self) -> crate::segment::SegmentSnapshot {
        self.segment_manager.acquire_snapshot().await
    }

    /// Clean up orphan segment files not registered in metadata.
    pub async fn cleanup_orphan_segments(&self) -> Result<usize> {
        self.segment_manager.cleanup_orphan_segments().await
    }

    /// Prepare commit — close queue, drain workers, collect built segments.
    ///
    /// All documents sent via `add_document` before this call are guaranteed
    /// to be written to segment files on disk. Segments are NOT yet registered
    /// in metadata — call `PreparedCommit::commit()` for that.
    ///
    /// Between prepare and commit, the caller can do external work (WAL sync,
    /// replication, etc.) knowing that `abort()` is possible if something fails.
    ///
    /// `add_document` will return `Closed` error until commit/abort respawns workers.
    pub async fn prepare_commit(&mut self) -> Result<PreparedCommit<'_, D>> {
        // 1. Close channel → workers drain remaining docs and exit
        self.doc_sender.close();

        // 2. Join worker OS threads (via spawn_blocking to avoid blocking tokio)
        let workers = std::mem::take(&mut self.workers);
        tokio::task::spawn_blocking(move || {
            for w in workers {
                let _ = w.join();
            }
        })
        .await
        .map_err(|e| Error::Internal(format!("Failed to join workers: {}", e)))?;

        // 3. Collect built segments
        let built = std::mem::take(&mut *self.worker_state.built_segments.lock());
        self.flushed_segments.extend(built);

        Ok(PreparedCommit {
            writer: self,
            is_resolved: false,
        })
    }

    /// Commit (convenience): prepare_commit + commit in one call.
    ///
    /// Guarantees all prior `add_document` calls are committed.
    /// Vector training is decoupled — call `build_vector_index()` manually.
    pub async fn commit(&mut self) -> Result<()> {
        self.prepare_commit().await?.commit().await
    }

    /// Force merge all segments into one.
    pub async fn force_merge(&mut self) -> Result<()> {
        self.prepare_commit().await?.commit().await?;
        self.segment_manager.force_merge().await
    }

    /// Respawn workers with a fresh channel. Called after commit or abort.
    ///
    /// If the tokio runtime has shut down (e.g., program exit), this is a no-op
    /// to avoid panicking in `Handle::current()`. The writer is left in a
    /// degraded state (closed channel, no workers) — `add_document` will return
    /// `Closed` errors, which is acceptable during shutdown.
    fn respawn_workers(&mut self) {
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let num_workers = self.config.num_indexing_threads.max(1);
        let (sender, workers) = Self::spawn_workers(&self.worker_state, num_workers);
        self.doc_sender = sender;
        self.workers = workers;
    }

    // Vector index methods (build_vector_index, etc.) are in vector_builder.rs
}

impl<D: DirectoryWriter + 'static> Drop for IndexWriter<D> {
    fn drop(&mut self) {
        self.doc_sender.close();
        for w in std::mem::take(&mut self.workers) {
            let _ = w.join();
        }
    }
}

/// A prepared commit that can be finalized or aborted.
///
/// Two-phase commit guard. Between `prepare_commit()` and
/// `commit()`/`abort()`, segments are on disk but NOT in metadata.
/// Dropping without calling either will auto-abort (discard segments,
/// respawn workers).
pub struct PreparedCommit<'a, D: DirectoryWriter + 'static> {
    writer: &'a mut IndexWriter<D>,
    is_resolved: bool,
}

impl<'a, D: DirectoryWriter + 'static> PreparedCommit<'a, D> {
    /// Finalize: register segments in metadata, evaluate merge policy, respawn workers.
    pub async fn commit(mut self) -> Result<()> {
        self.is_resolved = true;
        let segments = std::mem::take(&mut self.writer.flushed_segments);
        self.writer.segment_manager.commit(segments).await?;
        self.writer.segment_manager.maybe_merge().await;
        self.writer.respawn_workers();
        Ok(())
    }

    /// Abort: discard prepared segments, respawn workers.
    /// Segment files become orphans (cleaned up by `cleanup_orphan_segments`).
    pub fn abort(mut self) {
        self.is_resolved = true;
        self.writer.flushed_segments.clear();
        self.writer.respawn_workers();
    }
}

impl<D: DirectoryWriter + 'static> Drop for PreparedCommit<'_, D> {
    fn drop(&mut self) {
        if !self.is_resolved {
            log::warn!("PreparedCommit dropped without commit/abort — auto-aborting");
            self.writer.flushed_segments.clear();
            self.writer.respawn_workers();
        }
    }
}
