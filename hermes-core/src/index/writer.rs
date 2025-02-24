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
//!   1. `prepare_commit()` — closes queue, workers flush builders to disk.
//!      Returns a `PreparedCommit` guard. No new documents accepted until resolved.
//!   2. `PreparedCommit::commit()` — registers segments in metadata, resumes workers.
//!   3. `PreparedCommit::abort()` — discards prepared segments, resumes workers.
//!   4. `commit()` — convenience: `prepare_commit().await?.commit().await`.
//!
//! Since `prepare_commit`/`commit` take `&mut self`, Rust’s borrow checker
//! guarantees no concurrent `add_document` calls during the commit window.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
    /// Replaced on each commit cycle (workers get new receiver via resume).
    doc_sender: async_channel::Sender<Document>,
    /// Worker OS thread handles — long-lived, survive across commits.
    workers: Vec<std::thread::JoinHandle<()>>,
    /// Shared worker state (immutable config + mutable segment output + sync)
    worker_state: Arc<WorkerState<D>>,
    /// Segment manager — owns metadata.json, handles segments and background merging
    pub(super) segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Segments flushed to disk but not yet registered in metadata
    flushed_segments: Vec<(String, u32)>,
    /// Primary key dedup index (None if schema has no primary field)
    primary_key_index: Option<super::primary_key::PrimaryKeyIndex>,
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

    // === Worker lifecycle synchronization ===
    // Workers survive across commits. On prepare_commit the channel is closed;
    // workers flush their builders, increment flush_count, then wait on
    // resume_cvar for a new receiver. commit/abort creates a fresh channel
    // and wakes them.
    /// Number of workers that have completed their flush.
    flush_count: AtomicUsize,
    /// Mutex + condvar for prepare_commit to wait on all workers flushed.
    flush_mutex: parking_lot::Mutex<()>,
    flush_cvar: parking_lot::Condvar,
    /// Holds the new channel receiver after commit/abort. Workers clone from this.
    resume_receiver: parking_lot::Mutex<Option<async_channel::Receiver<Document>>>,
    /// Monotonically increasing epoch, bumped by each resume_workers call.
    /// Workers compare against their local epoch to avoid re-cloning a stale receiver.
    resume_epoch: AtomicUsize,
    /// Condvar for workers to wait for resume (new channel) or shutdown.
    resume_cvar: parking_lot::Condvar,
    /// When true, workers should exit permanently (IndexWriter dropped).
    shutdown: AtomicBool,
    /// Total number of worker threads.
    num_workers: usize,
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
        // Auto-configure tokenizers from schema for all text fields
        let registry = crate::tokenizer::TokenizerRegistry::new();
        let mut tokenizers = FxHashMap::default();
        for (field, entry) in schema.fields() {
            if matches!(entry.field_type, crate::dsl::FieldType::Text)
                && let Some(ref tok_name) = entry.tokenizer
                && let Some(tok) = registry.get(tok_name)
            {
                tokenizers.insert(field, tok);
            }
        }

        let num_workers = config.num_indexing_threads.max(1);
        let worker_state = Arc::new(WorkerState {
            directory: Arc::clone(&directory),
            schema: Arc::clone(&schema),
            builder_config,
            tokenizers: parking_lot::RwLock::new(tokenizers),
            memory_budget_per_worker: config.max_indexing_memory_bytes / num_workers,
            segment_manager: Arc::clone(&segment_manager),
            built_segments: parking_lot::Mutex::new(Vec::new()),
            flush_count: AtomicUsize::new(0),
            flush_mutex: parking_lot::Mutex::new(()),
            flush_cvar: parking_lot::Condvar::new(),
            resume_receiver: parking_lot::Mutex::new(None),
            resume_epoch: AtomicUsize::new(0),
            resume_cvar: parking_lot::Condvar::new(),
            shutdown: AtomicBool::new(false),
            num_workers,
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
            primary_key_index: None,
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

    /// Initialize primary key deduplication from committed segments.
    ///
    /// Tries to load a cached bloom filter from `pk_bloom.bin` first. If the
    /// cache covers all current segments, the bloom is reused directly (fast
    /// path). If new segments appeared since the cache was written, only their
    /// keys are iterated (incremental). Falls back to a full rebuild when no
    /// cache exists.
    ///
    /// The CPU-intensive bloom build is offloaded via `spawn_blocking` so it
    /// does not block the tokio runtime.
    ///
    /// No-op if schema has no primary field.
    pub async fn init_primary_key_dedup(&mut self) -> Result<()> {
        use super::primary_key::{PK_BLOOM_FILE, deserialize_pk_bloom};

        let field = match self.schema.primary_field() {
            Some(f) => f,
            None => return Ok(()),
        };

        let snapshot = self.segment_manager.acquire_snapshot().await;
        let current_seg_ids: Vec<String> = snapshot.segment_ids().to_vec();

        // Try to load persisted bloom filter.
        let cached = match self
            .directory
            .open_read(std::path::Path::new(PK_BLOOM_FILE))
            .await
        {
            Ok(handle) => {
                let data = handle.read_bytes_range(0..handle.len()).await;
                match data {
                    Ok(bytes) => deserialize_pk_bloom(bytes.as_slice()),
                    Err(_) => None,
                }
            }
            Err(_) => None,
        };

        // Open all segment readers concurrently (needed for committed-key lookups
        // regardless of whether we use the cached bloom).
        let open_futures: Vec<_> = current_seg_ids
            .iter()
            .map(|seg_id_str| {
                let seg_id_str = seg_id_str.clone();
                let dir = self.directory.as_ref();
                let schema = Arc::clone(&self.schema);
                let cache_blocks = self.config.term_cache_blocks;
                async move {
                    let seg_id =
                        crate::segment::SegmentId::from_hex(&seg_id_str).ok_or_else(|| {
                            Error::Internal(format!("Invalid segment id: {}", seg_id_str))
                        })?;
                    let reader =
                        crate::segment::SegmentReader::open(dir, seg_id, schema, cache_blocks)
                            .await?;
                    Ok::<_, Error>(Arc::new(reader))
                }
            })
            .collect();
        let readers = futures::future::try_join_all(open_futures).await?;

        if let Some((persisted_seg_ids, bloom)) = cached {
            // Find readers for segments not covered by the persisted bloom.
            let new_readers: Vec<Arc<crate::segment::SegmentReader>> = current_seg_ids
                .iter()
                .zip(readers.iter())
                .filter(|(id, _)| !persisted_seg_ids.contains(*id))
                .map(|(_, r)| Arc::clone(r))
                .collect();

            let needs_persist = !new_readers.is_empty();
            let pk_index = if new_readers.is_empty() {
                // Fast path: all segments covered by cache.
                super::primary_key::PrimaryKeyIndex::from_persisted(
                    field,
                    bloom,
                    readers,
                    &[],
                    snapshot,
                )
            } else {
                // Incremental: only iterate new segments' keys.
                tokio::task::spawn_blocking(move || {
                    super::primary_key::PrimaryKeyIndex::from_persisted(
                        field,
                        bloom,
                        readers,
                        &new_readers,
                        snapshot,
                    )
                })
                .await
                .map_err(|e| Error::Internal(format!("spawn_blocking failed: {}", e)))?
            };

            if needs_persist {
                self.persist_pk_bloom(&pk_index, &current_seg_ids).await;
            }

            self.primary_key_index = Some(pk_index);
        } else {
            // No cache — full rebuild, offloaded to blocking thread.
            let pk_index = tokio::task::spawn_blocking(move || {
                super::primary_key::PrimaryKeyIndex::new(field, readers, snapshot)
            })
            .await
            .map_err(|e| Error::Internal(format!("spawn_blocking failed: {}", e)))?;

            self.persist_pk_bloom(&pk_index, &current_seg_ids).await;
            self.primary_key_index = Some(pk_index);
        }

        Ok(())
    }

    /// Persist the primary-key bloom filter to `pk_bloom.bin`.
    /// Best-effort: errors are logged but not propagated.
    async fn persist_pk_bloom(
        &self,
        pk_index: &super::primary_key::PrimaryKeyIndex,
        segment_ids: &[String],
    ) {
        use super::primary_key::{PK_BLOOM_FILE, serialize_pk_bloom};

        let bloom_bytes = pk_index.bloom_to_bytes();
        let data = serialize_pk_bloom(segment_ids, &bloom_bytes);
        if let Err(e) = self
            .directory
            .write(std::path::Path::new(PK_BLOOM_FILE), &data)
            .await
        {
            log::warn!("[primary_key] failed to persist bloom cache: {}", e);
        }
    }

    /// Add a document to the indexing queue (sync, O(1), lock-free).
    ///
    /// `Document` is moved into the channel (zero-copy). Workers compete to pull it.
    /// Returns `Error::QueueFull` when the queue is at capacity — caller must back off.
    pub fn add_document(&self, doc: Document) -> Result<()> {
        if let Some(ref pk_index) = self.primary_key_index {
            pk_index.check_and_insert(&doc)?;
        }
        match self.doc_sender.try_send(doc) {
            Ok(()) => Ok(()),
            Err(async_channel::TrySendError::Full(doc)) => {
                // Roll back PK registration so the caller can retry later
                if let Some(ref pk_index) = self.primary_key_index {
                    pk_index.rollback_uncommitted_key(&doc);
                }
                Err(Error::QueueFull)
            }
            Err(async_channel::TrySendError::Closed(doc)) => {
                // Roll back PK registration for defense-in-depth
                if let Some(ref pk_index) = self.primary_key_index {
                    pk_index.rollback_uncommitted_key(&doc);
                }
                Err(Error::Internal("Document channel closed".into()))
            }
        }
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

    /// Worker loop — runs on a dedicated OS thread, survives across commits.
    ///
    /// Outer loop: each iteration processes one commit cycle.
    ///   Inner loop: pull documents from MPMC queue, index them, build segments
    ///   when memory budget is exceeded.
    ///   On channel close (prepare_commit): flush current builder, signal
    ///   flush_count, wait for resume with new receiver.
    ///   On shutdown (Drop): exit permanently.
    fn worker_loop(
        state: Arc<WorkerState<D>>,
        initial_receiver: async_channel::Receiver<Document>,
        handle: tokio::runtime::Handle,
    ) {
        let mut receiver = initial_receiver;
        let mut my_epoch = 0usize;

        loop {
            // Wrap the recv+build phase in catch_unwind so a panic doesn't
            // prevent flush_count from being signaled (which would hang
            // prepare_commit forever).
            let build_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut builder: Option<SegmentBuilder> = None;

                while let Ok(doc) = receiver.recv_blocking() {
                    // Initialize builder if needed
                    if builder.is_none() {
                        match SegmentBuilder::new(
                            Arc::clone(&state.schema),
                            state.builder_config.clone(),
                        ) {
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

                    // Reserve 20% headroom for segment build overhead (vid_set,
                    // VidLookup, postings_flat, grid_entries). These temporary
                    // allocations exist alongside the builder's data during build.
                    let effective_budget = state.memory_budget_per_worker * 4 / 5;

                    if builder_memory >= effective_budget && b.num_docs() >= MIN_DOCS_BEFORE_FLUSH {
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

                // Channel closed — flush current builder
                if let Some(b) = builder.take()
                    && b.num_docs() > 0
                {
                    Self::build_segment_inline(&state, b, &handle);
                }
            }));

            if build_result.is_err() {
                log::error!(
                    "[worker] panic during indexing cycle — documents in this cycle may be lost"
                );
            }

            // Signal flush completion (always, even after panic — prevents
            // prepare_commit from hanging)
            let prev = state.flush_count.fetch_add(1, Ordering::Release);
            if prev + 1 == state.num_workers {
                // Last worker — wake prepare_commit
                let _lock = state.flush_mutex.lock();
                state.flush_cvar.notify_one();
            }

            // Wait for resume (new channel) or shutdown.
            // Check resume_epoch to avoid re-cloning a stale receiver from
            // a previous cycle.
            {
                let mut lock = state.resume_receiver.lock();
                loop {
                    if state.shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    let current_epoch = state.resume_epoch.load(Ordering::Acquire);
                    if current_epoch > my_epoch
                        && let Some(rx) = lock.as_ref()
                    {
                        receiver = rx.clone();
                        my_epoch = current_epoch;
                        break;
                    }
                    state.resume_cvar.wait(&mut lock);
                }
            }
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

    /// Abort all in-flight merge tasks without waiting for completion.
    pub async fn abort_merges(&self) {
        self.segment_manager.abort_merges().await;
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

    /// Prepare commit — signal workers to flush, wait for completion, collect segments.
    ///
    /// All documents sent via `add_document` before this call are guaranteed
    /// to be written to segment files on disk. Segments are NOT yet registered
    /// in metadata — call `PreparedCommit::commit()` for that.
    ///
    /// Workers are NOT destroyed — they flush their builders and wait for
    /// `resume_workers()` to give them a new channel.
    ///
    /// `add_document` will return `Closed` error until commit/abort resumes workers.
    pub async fn prepare_commit(&mut self) -> Result<PreparedCommit<'_, D>> {
        // 1. Close channel → workers drain remaining docs and flush builders
        self.doc_sender.close();

        // Wake any workers still waiting on resume_cvar from previous cycle.
        // They'll clone the stale receiver, enter recv_blocking, get Err
        // immediately (sender already closed), flush, and signal completion.
        self.worker_state.resume_cvar.notify_all();

        // 2. Wait for all workers to complete their flush (via spawn_blocking
        //    to avoid blocking the tokio runtime)
        let state = Arc::clone(&self.worker_state);
        let all_flushed = tokio::task::spawn_blocking(move || {
            let mut lock = state.flush_mutex.lock();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
            while state.flush_count.load(Ordering::Acquire) < state.num_workers {
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                if remaining.is_zero() {
                    log::error!(
                        "[prepare_commit] timed out waiting for workers: {}/{} flushed",
                        state.flush_count.load(Ordering::Acquire),
                        state.num_workers
                    );
                    return false;
                }
                state.flush_cvar.wait_for(&mut lock, remaining);
            }
            true
        })
        .await
        .map_err(|e| Error::Internal(format!("Failed to wait for workers: {}", e)))?;

        if !all_flushed {
            // Resume workers so the system isn't stuck, then return error
            self.resume_workers();
            return Err(Error::Internal(format!(
                "prepare_commit timed out: {}/{} workers flushed",
                self.worker_state.flush_count.load(Ordering::Acquire),
                self.worker_state.num_workers
            )));
        }

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
    pub async fn commit(&mut self) -> Result<bool> {
        self.prepare_commit().await?.commit().await
    }

    /// Force merge all segments into one.
    pub async fn force_merge(&mut self) -> Result<()> {
        self.prepare_commit().await?.commit().await?;
        self.segment_manager.force_merge().await
    }

    /// Reorder all segments via Recursive Graph Bisection (BP) for better BMP pruning.
    ///
    /// Each segment is individually rebuilt with record-level BP reordering:
    /// ordinals are shuffled across blocks so that similar content clusters tightly.
    pub async fn reorder(&mut self) -> Result<()> {
        self.prepare_commit().await?.commit().await?;
        self.segment_manager.reorder_segments().await
    }

    /// Get the segment manager (for background optimizer access).
    pub fn segment_manager(&self) -> &Arc<crate::merge::SegmentManager<D>> {
        &self.segment_manager
    }

    /// Resume workers with a fresh channel. Called after commit or abort.
    ///
    /// Workers are already alive — just give them a new channel and wake them.
    /// If the tokio runtime has shut down (e.g., program exit), this is a no-op.
    fn resume_workers(&mut self) {
        if tokio::runtime::Handle::try_current().is_err() {
            // Runtime is gone — signal permanent shutdown so workers don't
            // hang forever on resume_cvar.
            self.worker_state.shutdown.store(true, Ordering::Release);
            self.worker_state.resume_cvar.notify_all();
            return;
        }

        // Reset flush count for next cycle
        self.worker_state.flush_count.store(0, Ordering::Release);

        // Create new channel
        let (sender, receiver) = async_channel::bounded(PIPELINE_MAX_SIZE_IN_DOCS);
        self.doc_sender = sender;

        // Set new receiver, bump epoch, and wake all workers
        {
            let mut lock = self.worker_state.resume_receiver.lock();
            *lock = Some(receiver);
        }
        self.worker_state
            .resume_epoch
            .fetch_add(1, Ordering::Release);
        self.worker_state.resume_cvar.notify_all();
    }

    // Vector index methods (build_vector_index, etc.) are in vector_builder.rs
}

impl<D: DirectoryWriter + 'static> Drop for IndexWriter<D> {
    fn drop(&mut self) {
        // 1. Signal permanent shutdown
        self.worker_state.shutdown.store(true, Ordering::Release);
        // 2. Close channel to wake workers blocked on recv_blocking
        self.doc_sender.close();
        // 3. Wake workers that might be waiting on resume_cvar
        self.worker_state.resume_cvar.notify_all();
        // 4. Join worker threads
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
    /// Finalize: register segments in metadata, evaluate merge policy, resume workers.
    ///
    /// Returns `true` if new segments were committed, `false` if nothing changed.
    pub async fn commit(mut self) -> Result<bool> {
        self.is_resolved = true;
        let segments = std::mem::take(&mut self.writer.flushed_segments);

        // Fast path: nothing to commit
        if segments.is_empty() {
            log::debug!("[commit] no segments to commit, skipping");
            self.writer.resume_workers();
            return Ok(false);
        }

        self.writer.segment_manager.commit(segments).await?;

        // Refresh primary key index with new committed readers (parallel open)
        if let Some(ref mut pk_index) = self.writer.primary_key_index {
            let snapshot = self.writer.segment_manager.acquire_snapshot().await;
            let open_futures: Vec<_> = snapshot
                .segment_ids()
                .iter()
                .filter_map(|seg_id_str| {
                    let seg_id = crate::segment::SegmentId::from_hex(seg_id_str)?;
                    let dir = self.writer.directory.as_ref();
                    let schema = Arc::clone(&self.writer.schema);
                    let cache_blocks = self.writer.config.term_cache_blocks;
                    Some(async move {
                        crate::segment::SegmentReader::open(dir, seg_id, schema, cache_blocks)
                            .await
                            .map(Arc::new)
                    })
                })
                .collect();
            let readers = futures::future::try_join_all(open_futures).await?;
            let seg_ids: Vec<String> = snapshot.segment_ids().to_vec();
            pk_index.refresh(readers, snapshot);

            // Persist bloom cache (extract bytes to avoid borrow conflict).
            let bloom_bytes = pk_index.bloom_to_bytes();
            let data = super::primary_key::serialize_pk_bloom(&seg_ids, &bloom_bytes);
            if let Err(e) = self
                .writer
                .directory
                .write(
                    std::path::Path::new(super::primary_key::PK_BLOOM_FILE),
                    &data,
                )
                .await
            {
                log::warn!("[primary_key] failed to persist bloom cache: {}", e);
            }
        }

        self.writer.segment_manager.maybe_merge().await;
        self.writer.resume_workers();
        Ok(true)
    }

    /// Abort: discard prepared segments, resume workers.
    /// Segment files become orphans (cleaned up by `cleanup_orphan_segments`).
    pub fn abort(mut self) {
        self.is_resolved = true;
        self.writer.flushed_segments.clear();
        if let Some(ref mut pk_index) = self.writer.primary_key_index {
            pk_index.clear_uncommitted();
        }
        self.writer.resume_workers();
    }
}

impl<D: DirectoryWriter + 'static> Drop for PreparedCommit<'_, D> {
    fn drop(&mut self) {
        if !self.is_resolved {
            log::warn!("PreparedCommit dropped without commit/abort — auto-aborting");
            self.writer.flushed_segments.clear();
            if let Some(ref mut pk_index) = self.writer.primary_key_index {
                pk_index.clear_uncommitted();
            }
            self.writer.resume_workers();
        }
    }
}
