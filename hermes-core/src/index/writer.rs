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

use futures::FutureExt;
use rustc_hash::FxHashMap;

use crate::directories::DirectoryWriter;
use crate::dsl::{Document, Field, Schema};
use crate::error::{Error, Result};
use crate::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId};
use crate::tokenizer::BoxedTokenizer;

use super::IndexConfig;

/// Total pipeline capacity (in documents).
const PIPELINE_MAX_SIZE_IN_DOCS: usize = 10_000;

/// File name of the advisory single-writer lock inside the index directory.
pub const WRITER_LOCK_FILENAME: &str = ".hermes_writer.lock";

/// Advisory single-writer lock state.
///
/// Two independent writers on one index directory silently destroy each
/// other's data: the orphan sweep at writer open deletes the other process's
/// unpublished segment files, and metadata saves are last-writer-wins. For
/// directories rooted on a local filesystem the writer therefore holds an OS
/// advisory lock for its whole lifetime; the kernel releases it automatically
/// when the process dies.
enum WriterLock {
    /// Lock acquired. Closing the file (writer drop) releases it.
    Held { _file: std::fs::File },
    /// The directory has no lockable local filesystem root (e.g. RAM or
    /// remote directories) — cross-process locking is not applicable.
    NotApplicable,
    /// Another writer holds the lock. Every mutating operation fails loudly
    /// with this message instead of silently double-writing.
    Unavailable { reason: String },
}

/// Local filesystem root of the index directory, when the directory type
/// exposes one.
fn writer_lock_root<D: DirectoryWriter + 'static>(directory: &D) -> Option<std::path::PathBuf> {
    let any: &dyn std::any::Any = directory;
    if let Some(mmap) = any.downcast_ref::<crate::directories::MmapDirectory>() {
        return Some(mmap.root().to_path_buf());
    }
    // FsDirectory does not expose its root path, so the single-writer lock
    // cannot be enforced for it yet. Say so loudly instead of silently
    // skipping protection for a filesystem-backed writer.
    if any
        .downcast_ref::<crate::directories::FsDirectory>()
        .is_some()
    {
        log::warn!(
            "[writer_lock] FsDirectory exposes no root path; single-writer locking \
             is not enforced for this writer — do not open a second writer for the \
             same index directory"
        );
    }
    None
}

/// Try to take the exclusive single-writer lock for `directory`.
///
/// Returns `WriterLock::Unavailable` (not `Err`) on conflict so infallible
/// constructors can defer the failure to their first mutating operation.
fn try_acquire_writer_lock<D: DirectoryWriter + 'static>(directory: &D) -> Result<WriterLock> {
    let Some(root) = writer_lock_root(directory) else {
        return Ok(WriterLock::NotApplicable);
    };
    std::fs::create_dir_all(&root)?;
    let lock_path = root.join(WRITER_LOCK_FILENAME);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)?;
    match file.try_lock() {
        Ok(()) => Ok(WriterLock::Held { _file: file }),
        Err(std::fs::TryLockError::WouldBlock) => Ok(WriterLock::Unavailable {
            reason: format!(
                "another IndexWriter already holds the single-writer lock for this \
                 index ({}); Hermes supports one writer per index directory — stop \
                 the other writer (e.g. a running hermes-server or hermes-tool) \
                 before opening this one",
                lock_path.display()
            ),
        }),
        Err(std::fs::TryLockError::Error(error)) => Err(Error::Io(error)),
    }
}

/// Async IndexWriter for adding documents and committing segments.
///
/// **Backpressure:** `add_document()` is sync and O(1). It returns
/// `Error::QueueFull` when the shared queue is full and
/// `Error::CommitInProgress` while a generation is publishing or awaiting
/// retry; callers must back off.
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
    /// MPMC sender, replaced under a brief lock on each commit cycle (workers
    /// get the corresponding new receiver via resume).
    doc_sender: Arc<parking_lot::RwLock<async_channel::Sender<Document>>>,
    /// Worker OS thread handles — long-lived, survive across commits.
    workers: Vec<std::thread::JoinHandle<()>>,
    /// Shared worker state (immutable config + mutable segment output + sync)
    worker_state: Arc<WorkerState<D>>,
    /// Segment manager — owns metadata.json, handles segments and background merging
    pub(super) segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Segments flushed to disk but not yet registered in metadata. Each item
    /// owns an active-operation guard, so orphan sweeping cannot delete it.
    flushed_segments: Arc<parking_lot::Mutex<Vec<PreparedSegment<D>>>>,
    /// Primary key dedup index (None if schema has no primary field)
    primary_key_index: Arc<parking_lot::RwLock<Option<super::primary_key::PrimaryKeyIndex>>>,
    /// Tracks the owned finalizer spawned by `PreparedCommit::commit`. The
    /// requesting future may disappear, but a second commit generation must
    /// not start until this one has made publication and worker state agree.
    commit_finalization: Arc<CommitFinalizationState>,
    /// True while a failed post-commit PK refresh has left the uncommitted
    /// reservations as the ONLY record of already-committed keys (fail-closed,
    /// see `finalize_prepared_commit`). While set, abort paths must NOT clear
    /// the reservations or duplicate primary keys could be admitted.
    pk_reservations_retained: Arc<AtomicBool>,
    /// Advisory single-writer lock, held for the writer's lifetime.
    /// `Unavailable` is retryable: the conflicting holder may exit at any
    /// time (the kernel then releases its lock), so `ensure_writer_lock`
    /// re-attempts acquisition instead of caching the conflict forever.
    writer_lock: parking_lot::RwLock<WriterLock>,
}

#[derive(Default)]
struct CommitFinalizationState {
    in_progress: AtomicBool,
    idle: tokio::sync::Notify,
}

impl CommitFinalizationState {
    fn begin(&self) -> bool {
        self.in_progress
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    fn finish(&self) {
        self.in_progress.store(false, Ordering::Release);
        self.idle.notify_waiters();
    }

    async fn wait_until_idle(&self) {
        while self.in_progress.load(Ordering::Acquire) {
            let notified = self.idle.notified();
            if !self.in_progress.load(Ordering::Acquire) {
                break;
            }
            notified.await;
        }
    }
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
    /// Segments built by workers, collected by `prepare_commit()`. Their RAII
    /// guards protect both in-progress and completed-uncommitted files.
    built_segments: parking_lot::Mutex<Vec<PreparedSegment<D>>>,
    /// First failure in the current flush generation. Worker-side indexing is
    /// asynchronous, so `prepare_commit` is the only sound place to surface
    /// it to the caller. A failed generation is aborted as a unit; publishing
    /// only its successful segments would silently lose documents.
    cycle_error: parking_lot::Mutex<Option<String>>,
    cycle_failed: AtomicBool,

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

/// A completed indexing segment that has not been published in metadata yet.
///
/// `operation` is intentionally data, not a side-channel set update: moving
/// this value through worker → prepared commit → commit/abort moves lifecycle
/// ownership with it, and every unwind/drop path releases ownership safely.
struct PreparedSegment<D: DirectoryWriter + 'static> {
    id: String,
    segment_id: SegmentId,
    num_docs: u32,
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
    operation: Option<crate::merge::SegmentOperationGuard>,
    runtime: tokio::runtime::Handle,
    needs_vector_upgrade: bool,
    published: bool,
}

impl<D: DirectoryWriter + 'static> PreparedSegment<D> {
    fn metadata_entry(&self) -> (String, u32) {
        (self.id.clone(), self.num_docs)
    }

    fn mark_published(&mut self) {
        self.published = true;
        // Metadata + SegmentTracker are now the durable lifecycle owners.
        drop(self.operation.take());
    }
}

impl<D: DirectoryWriter + 'static> WorkerState<D> {
    fn record_cycle_error(&self, error: impl Into<String>) {
        let mut first_error = self.cycle_error.lock();
        if first_error.is_none() {
            *first_error = Some(error.into());
        }
        drop(first_error);
        self.cycle_failed.store(true, Ordering::Release);
    }
}

impl<D: DirectoryWriter + 'static> Drop for PreparedSegment<D> {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        let Some(operation) = self.operation.take() else {
            return;
        };
        self.segment_manager.schedule_unpublished_segment_cleanup(
            self.segment_id,
            operation,
            self.runtime.clone(),
        );
    }
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
        // Directory-layer metrics (cold writes, lazy reads) carry the index label
        directory.set_index_label(schema.index_label());

        // Refuse a second writer before touching any index state.
        let writer_lock = try_acquire_writer_lock(directory.as_ref())?;
        if let WriterLock::Unavailable { reason } = &writer_lock {
            return Err(Error::Internal(reason.clone()));
        }
        // Refuse to clobber an existing index: persisting a fresh empty
        // metadata.json would orphan every committed segment, and the next
        // writer open's orphan sweep would permanently delete them.
        if directory
            .exists(std::path::Path::new(super::INDEX_META_FILENAME))
            .await?
        {
            return Err(Error::Internal(format!(
                "refusing to create index: {} already exists in this directory; \
                 use IndexWriter::open to open the existing index, or delete the \
                 directory first if you really want to start over",
                super::INDEX_META_FILENAME
            )));
        }

        let metadata = super::IndexMetadata::new((*schema).clone());

        let segment_manager = Arc::new(crate::merge::SegmentManager::new(
            Arc::clone(&directory),
            Arc::clone(&schema),
            metadata,
            config.merge_policy.clone_box(),
            config.term_cache_blocks,
            config.max_concurrent_merges,
            Arc::clone(&config.background_merge_permits),
            config.merge_bp_time_budget,
            config.bp_memory_budget_bytes,
            Arc::clone(&config.background_reorder_permits),
            config.background_reorder_pool.clone(),
        ));
        segment_manager.update_metadata(|_| {}).await?;

        Ok(Self::new_with_parts(
            directory,
            schema,
            config,
            builder_config,
            segment_manager,
            writer_lock,
        ))
    }

    /// Open an existing index for exclusive writing.
    ///
    /// Multiple independent writers for the same directory are unsupported;
    /// for filesystem-rooted directories this is enforced with an advisory
    /// single-writer lock ([`WRITER_LOCK_FILENAME`]) held for the writer's
    /// lifetime. This path removes crash-leftover outputs before starting its
    /// workers. Use [`Index::writer`](super::Index::writer) to share lifecycle
    /// state with an already-open search index.
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

        // The lock must be held before the orphan sweep below: sweeping while
        // another process's writer is live deletes its in-flight outputs.
        let writer_lock = try_acquire_writer_lock(directory.as_ref())?;
        if let WriterLock::Unavailable { reason } = &writer_lock {
            return Err(Error::Internal(reason.clone()));
        }

        let metadata = super::IndexMetadata::load(directory.as_ref()).await?;
        let schema = Arc::new(metadata.schema.clone());
        // Directory-layer metrics (cold writes, lazy reads) carry the index label
        directory.set_index_label(schema.index_label());

        let segment_manager = Arc::new(crate::merge::SegmentManager::new(
            Arc::clone(&directory),
            Arc::clone(&schema),
            metadata,
            config.merge_policy.clone_box(),
            config.term_cache_blocks,
            config.max_concurrent_merges,
            Arc::clone(&config.background_merge_permits),
            config.merge_bp_time_budget,
            config.bp_memory_budget_bytes,
            Arc::clone(&config.background_reorder_permits),
            config.background_reorder_pool.clone(),
        ));
        let swept = segment_manager.cleanup_orphan_segments().await?;
        if swept > 0 {
            log::warn!(
                "[segment_cleanup] swept {} orphan segment(s) while opening writer",
                swept
            );
        }
        segment_manager.try_load_and_publish_trained().await?;

        Ok(Self::new_with_parts(
            directory,
            schema,
            config,
            builder_config,
            segment_manager,
            writer_lock,
        ))
    }

    /// Create an IndexWriter from an existing Index.
    /// Shares the SegmentManager for consistent segment lifecycle management.
    ///
    /// This constructor is infallible, so a single-writer lock conflict is
    /// deferred: the returned writer fails loudly on its first mutating
    /// operation instead of silently double-writing next to another writer.
    pub fn from_index(index: &super::Index<D>) -> Self {
        let writer_lock = match try_acquire_writer_lock(index.directory.as_ref()) {
            Ok(lock) => lock,
            Err(error) => WriterLock::Unavailable {
                reason: format!("failed to acquire the single-writer lock: {error}"),
            },
        };
        if let WriterLock::Unavailable { reason } = &writer_lock {
            log::error!("[writer_lock] {reason}");
        }
        Self::new_with_parts(
            Arc::clone(&index.directory),
            Arc::clone(&index.schema),
            index.config.clone(),
            SegmentBuilderConfig::default(),
            Arc::clone(&index.segment_manager),
            writer_lock,
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
        writer_lock: WriterLock,
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
            cycle_error: parking_lot::Mutex::new(None),
            cycle_failed: AtomicBool::new(false),
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
            doc_sender: Arc::new(parking_lot::RwLock::new(doc_sender)),
            workers,
            worker_state,
            segment_manager,
            flushed_segments: Arc::new(parking_lot::Mutex::new(Vec::new())),
            primary_key_index: Arc::new(parking_lot::RwLock::new(None)),
            commit_finalization: Arc::new(CommitFinalizationState::default()),
            pk_reservations_retained: Arc::new(AtomicBool::new(false)),
            writer_lock: parking_lot::RwLock::new(writer_lock),
        }
    }

    /// Fail loudly when another writer owns the single-writer lock.
    ///
    /// A deferred conflict (`from_index` during a writer handover, e.g. a
    /// rolling pod restart) is not permanent: the holder exits and the kernel
    /// releases its advisory lock. Re-attempt acquisition on every call in
    /// the `Unavailable` state so the writer recovers as soon as the lock
    /// frees, instead of rejecting all writes for its lifetime.
    fn ensure_writer_lock(&self) -> Result<()> {
        // Fast path: uncontended read on the healthy states.
        if !matches!(&*self.writer_lock.read(), WriterLock::Unavailable { .. }) {
            return Ok(());
        }

        let mut lock = self.writer_lock.write();
        // Another thread may have recovered while we waited for the write lock.
        if !matches!(&*lock, WriterLock::Unavailable { .. }) {
            return Ok(());
        }
        match try_acquire_writer_lock(self.directory.as_ref())? {
            acquired @ (WriterLock::Held { .. } | WriterLock::NotApplicable) => {
                log::info!(
                    "[writer_lock] single-writer lock acquired after retry; \
                     the previous holder has released it — resuming writes"
                );
                *lock = acquired;
                Ok(())
            }
            WriterLock::Unavailable { reason } => {
                let err = Error::Internal(reason.clone());
                *lock = WriterLock::Unavailable { reason };
                Err(err)
            }
        }
    }

    /// Clear primary-key reservations after an aborted or failed generation.
    ///
    /// Skipped while a failed post-commit PK refresh has left the uncommitted
    /// reservations as the ONLY record of already-committed keys (fail-closed,
    /// see `finalize_prepared_commit`): wiping them would admit duplicate
    /// primary keys. Retaining the aborted generation's keys as well is
    /// deliberately conservative — they clear on the next successful commit's
    /// refresh.
    fn clear_uncommitted_pk_reservations(&self) {
        if self.pk_reservations_retained.load(Ordering::Acquire) {
            log::warn!(
                "[primary_key] keeping uncommitted reservations through abort: a \
                 failed post-commit refresh left them as the only record of \
                 committed keys; they are cleared by the next successful commit"
            );
            return;
        }
        if let Some(pk_index) = self.primary_key_index.write().as_mut() {
            pk_index.clear_uncommitted();
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
    /// Only loads fast-field data (text dictionaries) per segment — NOT full
    /// `SegmentReader`s — to avoid duplicating dense/sparse index memory.
    ///
    /// The CPU-intensive bloom build is offloaded via `spawn_blocking` so it
    /// does not block the tokio runtime.
    ///
    /// No-op if schema has no primary field.
    pub async fn init_primary_key_dedup(&mut self) -> Result<()> {
        use super::primary_key::{PK_BLOOM_FILE, deserialize_pk_bloom};

        self.commit_finalization.wait_until_idle().await;

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

        // Load lightweight fast-field data for all segments concurrently.
        let load_futures: Vec<_> = current_seg_ids
            .iter()
            .map(|seg_id_str| {
                let seg_id_str = seg_id_str.clone();
                let dir = self.directory.as_ref();
                let schema = Arc::clone(&self.schema);
                async move { load_pk_segment_data(dir, &seg_id_str, &schema).await }
            })
            .collect();
        let all_data = futures::future::try_join_all(load_futures).await?;

        if let Some((persisted_seg_ids, bloom)) = cached {
            // Partition: old segments (covered by bloom) first, new segments at end.
            let mut pk_data = Vec::with_capacity(all_data.len());
            let mut new_data = Vec::new();
            for d in all_data {
                if persisted_seg_ids.contains(&d.segment_id) {
                    pk_data.push(d);
                } else {
                    new_data.push(d);
                }
            }
            let needs_persist = !new_data.is_empty();
            let new_start = pk_data.len();
            pk_data.extend(new_data);

            let pk_index = if new_start == pk_data.len() {
                // Fast path: all segments covered by cache.
                super::primary_key::PrimaryKeyIndex::from_persisted(
                    field,
                    bloom,
                    pk_data,
                    &[],
                    snapshot,
                )
            } else {
                // Incremental: only iterate new segments' keys.
                tokio::task::spawn_blocking(move || {
                    // Insert new segments' keys into the bloom, then construct
                    // PrimaryKeyIndex with the pre-populated bloom.
                    let mut bloom = bloom;
                    let mut added = 0usize;
                    let num_new = pk_data.len() - new_start;
                    for data in &pk_data[new_start..] {
                        if let Some(ff) = data.fast_fields.get(&field.0)
                            && let Some(dict) = ff.text_dict()
                        {
                            for key in dict.iter() {
                                bloom.insert(key.as_bytes());
                                added += 1;
                            }
                        }
                    }
                    if added > 0 {
                        log::info!(
                            "[primary_key] bloom: added {} keys from {} new segment(s)",
                            added,
                            num_new,
                        );
                    }
                    super::primary_key::PrimaryKeyIndex::from_persisted(
                        field,
                        bloom,
                        pk_data,
                        &[],
                        snapshot,
                    )
                })
                .await
                .map_err(|e| Error::Internal(format!("spawn_blocking failed: {}", e)))?
            };

            if needs_persist {
                self.persist_pk_bloom(&pk_index, &current_seg_ids).await;
            }

            *self.primary_key_index.write() = Some(pk_index);
        } else {
            // No cache — full rebuild, offloaded to blocking thread.
            let pk_index = tokio::task::spawn_blocking(move || {
                super::primary_key::PrimaryKeyIndex::new(field, all_data, snapshot)
            })
            .await
            .map_err(|e| Error::Internal(format!("spawn_blocking failed: {}", e)))?;

            self.persist_pk_bloom(&pk_index, &current_seg_ids).await;
            *self.primary_key_index.write() = Some(pk_index);
        }

        // The freshly built index covers every committed segment, so any
        // reservations retained after a failed post-commit refresh are
        // superseded by committed_data.
        self.pk_reservations_retained
            .store(false, Ordering::Release);

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

    /// Add a document to the indexing queue (sync, O(1)).
    ///
    /// `Document` is moved into the channel (zero-copy). Workers compete to pull it.
    /// Returns an explicit backpressure error when the queue is at capacity or
    /// a prepared commit generation is not yet resolved.
    pub fn add_document(&self, doc: Document) -> Result<()> {
        self.ensure_writer_lock()?;
        if self.worker_state.shutdown.load(Ordering::Acquire) {
            return Err(Error::IndexClosed);
        }
        if self.commit_finalization.in_progress.load(Ordering::Acquire) {
            return Err(Error::CommitInProgress);
        }
        let sender = self.doc_sender.read().clone();
        // A publication error deliberately leaves the prepared generation and
        // its workers paused for a lossless retry. Report this as backpressure
        // instead of inserting/rolling back a PK key against a closed channel.
        if sender.is_closed() {
            return Err(Error::CommitInProgress);
        }
        let primary_key_index = self.primary_key_index.read();
        if let Some(ref pk_index) = *primary_key_index {
            pk_index.check_and_insert(&doc)?;
        }
        match sender.try_send(doc) {
            Ok(()) => Ok(()),
            Err(async_channel::TrySendError::Full(doc)) => {
                // Roll back PK registration so the caller can retry later
                if let Some(ref pk_index) = *primary_key_index {
                    pk_index.rollback_uncommitted_key(&doc);
                }
                Err(Error::QueueFull)
            }
            Err(async_channel::TrySendError::Closed(doc)) => {
                // Roll back PK registration for defense-in-depth
                if let Some(ref pk_index) = *primary_key_index {
                    pk_index.rollback_uncommitted_key(&doc);
                }
                Err(Error::CommitInProgress)
            }
        }
    }

    /// Add multiple documents to the indexing queue.
    ///
    /// Returns the number of documents successfully queued. Stops at the first
    /// backpressure error and returns the count queued so far.
    pub fn add_documents(&self, documents: Vec<Document>) -> Result<usize> {
        let total = documents.len();
        for (i, doc) in documents.into_iter().enumerate() {
            match self.add_document(doc) {
                Ok(()) => {}
                Err(Error::QueueFull | Error::CommitInProgress) => return Ok(i),
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
                    if state.shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    // Another worker already invalidated this generation.
                    // Drain the shared queue so prepare_commit can complete,
                    // but do not spend CPU/RAM building outputs that must be
                    // discarded transactionally.
                    if state.cycle_failed.load(Ordering::Acquire) {
                        continue;
                    }
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
                                state.record_cycle_error(format!(
                                    "failed to create segment builder: {e}"
                                ));
                                continue;
                            }
                        }
                    }

                    let b = builder.as_mut().unwrap();
                    if let Err(e) = b.add_document(doc) {
                        log::error!("Failed to index document: {:?}", e);
                        state.record_cycle_error(format!("failed to index document: {e}"));
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
                if !state.cycle_failed.load(Ordering::Acquire)
                    && let Some(b) = builder.take()
                    && b.num_docs() > 0
                {
                    Self::build_segment_inline(&state, b, &handle);
                }
            }));

            if build_result.is_err() {
                log::error!(
                    "[worker] panic during indexing cycle — documents in this cycle may be lost"
                );
                state.record_cycle_error("indexing worker panicked while building the batch");
            }

            // Signal flush completion (always, even after panic — prevents
            // prepare_commit from hanging)
            let prev = state.flush_count.fetch_add(1, Ordering::Release);
            if prev + 1 == state.num_workers {
                // Last worker — wake prepare_commit. notify_all, not
                // notify_one: a cancelled commit leaves its detached
                // spawn_blocking waiter parked on this condvar, and with a
                // single notification that dead waiter would consume the
                // only wakeup, stalling a retried prepare_commit for its
                // full deadline.
                let _lock = state.flush_mutex.lock();
                state.flush_cvar.notify_all();
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
        // Claim the ID before the first file write. The guard is moved into
        // `PreparedSegment` on success and otherwise releases automatically.
        let operation = match state
            .segment_manager
            .protect_new_segment(segment_hex.clone())
        {
            Ok(operation) => operation,
            Err(e) => {
                log::error!(
                    "[segment_build_failed] segment_id={} lifecycle_error={}",
                    segment_hex,
                    e,
                );
                state.record_cycle_error(format!(
                    "failed to claim segment {segment_hex} for building: {e}"
                ));
                return;
            }
        };
        let trained = state.segment_manager.trained_for_segment_build();
        let doc_count = builder.num_docs();
        let build_start = std::time::Instant::now();

        log::info!(
            "[segment_build] segment_id={} doc_count={} ann={}",
            segment_hex,
            doc_count,
            trained.is_some()
        );

        // Construct the cleanup owner before building. It keeps lifecycle
        // ownership through async deletion on ordinary error, abort, and
        // panic unwind; crash recovery is the only path left to the sweeper.
        let mut prepared = PreparedSegment {
            id: segment_hex.clone(),
            segment_id,
            num_docs: doc_count,
            segment_manager: Arc::clone(&state.segment_manager),
            operation: Some(operation),
            runtime: handle.clone(),
            needs_vector_upgrade: trained.is_none(),
            published: false,
        };

        match handle.block_on(builder.build(
            state.directory.as_ref(),
            segment_id,
            trained.as_deref(),
        )) {
            Ok(meta) if meta.num_docs == doc_count && meta.num_docs > 0 => {
                let duration_ms = build_start.elapsed().as_millis() as u64;
                log::info!(
                    "[segment_build_done] segment_id={} doc_count={} duration_ms={}",
                    segment_hex,
                    meta.num_docs,
                    duration_ms,
                );
                prepared.num_docs = meta.num_docs;
                state.built_segments.lock().push(prepared);
            }
            Ok(meta) => {
                let error = format!(
                    "segment {segment_hex} built {} docs from a {doc_count}-document builder",
                    meta.num_docs
                );
                log::error!("[segment_build_failed] {error}");
                state.record_cycle_error(error);
            }
            Err(e) => {
                log::error!(
                    "[segment_build_failed] segment_id={} error={:?}",
                    segment_hex,
                    e
                );
                // `prepared` owns the lifecycle claim and schedules one
                // tracked, idempotent cleanup pass when this scope ends.
                state.record_cycle_error(format!("failed to build segment {segment_hex}: {e}"));
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

    /// Drain all in-flight merge tasks.
    /// Blocking merge phases cannot be cancelled safely once started.
    pub async fn abort_merges(&self) {
        self.segment_manager.abort_merges().await;
    }

    /// Stop accepting lifecycle work, stop and join indexing workers, and
    /// discard unpublished segments. Index deletion calls this while holding
    /// the registry writer lock so in-flight requests finish first and stale
    /// writer Arcs cannot restart work afterward.
    pub async fn shutdown(&mut self) -> Result<()> {
        self.segment_manager.begin_shutdown();
        self.signal_worker_shutdown();

        // A cancelled commit request leaves its owned finalizer running. Do not
        // clear shared PK/prepared state while that task may still publish or
        // refresh it. Worker shutdown is signalled first, so a successful
        // finalizer cannot restart ingestion while deletion is waiting.
        self.commit_finalization.wait_until_idle().await;

        let workers = std::mem::take(&mut self.workers);
        let panicked = tokio::task::spawn_blocking(move || {
            workers
                .into_iter()
                .map(|worker| worker.join().is_err())
                .filter(|panicked| *panicked)
                .count()
        })
        .await
        .map_err(|error| Error::Internal(format!("failed to join index workers: {}", error)))?;
        if panicked > 0 {
            log::error!("[index_shutdown] {} indexing worker(s) panicked", panicked);
        }

        // No commit is possible after shutdown. Dropping these RAII values
        // releases their lifecycle ownership before directory deletion.
        self.flushed_segments.lock().clear();
        self.worker_state.built_segments.lock().clear();
        if let Some(pk_index) = self.primary_key_index.write().as_mut() {
            pk_index.clear_uncommitted();
        }
        Ok(())
    }

    /// Wait for the in-flight background merge to complete (if any).
    pub async fn wait_for_merging_thread(&self) {
        self.segment_manager.wait_for_merging_thread().await;
    }

    /// Wait for all eligible merges to complete, including cascading merges.
    pub async fn wait_for_all_merges(&self) {
        self.segment_manager.wait_for_all_merges().await;
    }

    /// Wait until an owned commit finalizer has reconciled durable metadata,
    /// primary-key state, and worker availability. Normally callers need not
    /// use this: it exists for orderly shutdown and request supervisors that
    /// want to observe completion after cancelling their original waiter.
    pub async fn wait_for_commit_finalization(&self) {
        self.commit_finalization.wait_until_idle().await;
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
    ///
    /// Requires the single-writer lock: sweeping while another process's
    /// writer is live would delete its in-flight segment outputs.
    pub async fn cleanup_orphan_segments(&self) -> Result<usize> {
        self.ensure_writer_lock()?;
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
    /// `add_document` returns `CommitInProgress` until commit/abort resumes workers.
    pub async fn prepare_commit(&mut self) -> Result<PreparedCommit<'_, D>> {
        self.ensure_writer_lock()?;
        if self.worker_state.shutdown.load(Ordering::Acquire) {
            return Err(Error::IndexClosed);
        }
        if self.commit_finalization.in_progress.load(Ordering::Acquire) {
            return Err(Error::CommitInProgress);
        }
        // 1. Close channel → workers drain remaining docs and flush builders
        self.doc_sender.read().close();

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
            // Keep this commit cycle paused. Resetting flush_count and handing
            // out a new receiver while an old worker is still building lets
            // that late worker increment the *next* cycle's counter. A later
            // prepare can then return before all of its workers flushed and
            // publish an incomplete set of segments. The caller may retry
            // prepare_commit; it will observe the same generation and collect
            // every completed output once the lagging worker finishes.
            return Err(Error::Internal(format!(
                "prepare_commit timed out: {}/{} workers flushed; writer remains paused, retry commit",
                self.worker_state.flush_count.load(Ordering::Acquire),
                self.worker_state.num_workers
            )));
        }

        let cycle_error = { self.worker_state.cycle_error.lock().take() };
        if let Some(error) = cycle_error {
            // No partial publication: some documents in this generation no
            // longer exist in a worker builder, so successful sibling outputs
            // cannot be committed without violating commit's all-prior-docs
            // guarantee. Their RAII drops retain ownership through deletion.
            self.flushed_segments.lock().clear();
            self.worker_state.built_segments.lock().clear();
            self.clear_uncommitted_pk_reservations();
            self.resume_workers();
            return Err(Error::Internal(format!(
                "indexing generation failed; no documents from this batch were committed: {error}"
            )));
        }

        // 3. Collect built segments
        let built = std::mem::take(&mut *self.worker_state.built_segments.lock());
        self.flushed_segments.lock().extend(built);

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
        Self::resume_workers_shared(&self.worker_state, &self.doc_sender);
    }

    fn resume_workers_shared(
        worker_state: &Arc<WorkerState<D>>,
        doc_sender: &Arc<parking_lot::RwLock<async_channel::Sender<Document>>>,
    ) {
        if worker_state.shutdown.load(Ordering::Acquire) {
            return;
        }
        if tokio::runtime::Handle::try_current().is_err() {
            // Runtime is gone — signal permanent shutdown so workers don't
            // hang forever on resume_cvar.
            worker_state.shutdown.store(true, Ordering::Release);
            worker_state.resume_cvar.notify_all();
            return;
        }

        // Reset flush count for next cycle
        worker_state.flush_count.store(0, Ordering::Release);
        *worker_state.cycle_error.lock() = None;
        worker_state.cycle_failed.store(false, Ordering::Release);

        // Create new channel
        let (sender, receiver) = async_channel::bounded(PIPELINE_MAX_SIZE_IN_DOCS);
        *doc_sender.write() = sender;

        // Set new receiver, bump epoch, and wake all workers
        {
            let mut lock = worker_state.resume_receiver.lock();
            *lock = Some(receiver);
        }
        worker_state.resume_epoch.fetch_add(1, Ordering::Release);
        worker_state.resume_cvar.notify_all();
    }

    fn signal_worker_shutdown(&self) {
        self.worker_state.shutdown.store(true, Ordering::Release);
        self.doc_sender.read().close();
        self.worker_state.resume_cvar.notify_all();
    }

    // Vector index methods (build_vector_index, etc.) are in vector_builder.rs
}

impl<D: DirectoryWriter + 'static> Drop for IndexWriter<D> {
    fn drop(&mut self) {
        self.signal_worker_shutdown();
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

/// Returns prepared segments to the writer if an owned commit finalizer fails
/// or unwinds before it can establish that metadata owns them. Retrying commit
/// is safe even when publication actually won the race: `SegmentManager::commit`
/// is idempotent and the operation guards keep the files protected meanwhile.
struct PreparedSegmentsGuard<D: DirectoryWriter + 'static> {
    segments: Option<Vec<PreparedSegment<D>>>,
    retry_slot: Arc<parking_lot::Mutex<Vec<PreparedSegment<D>>>>,
}

impl<D: DirectoryWriter + 'static> PreparedSegmentsGuard<D> {
    fn metadata_entries(&self) -> Vec<(String, u32)> {
        self.segments
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(PreparedSegment::metadata_entry)
            .collect()
    }

    fn take_published(&mut self) -> Vec<PreparedSegment<D>> {
        self.segments.take().unwrap_or_default()
    }

    fn vector_upgrade_segment_ids(&self) -> Vec<String> {
        self.segments
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter(|segment| segment.needs_vector_upgrade)
            .map(|segment| segment.id.clone())
            .collect()
    }
}

impl<D: DirectoryWriter + 'static> Drop for PreparedSegmentsGuard<D> {
    fn drop(&mut self) {
        if let Some(segments) = self.segments.take() {
            self.retry_slot.lock().extend(segments);
        }
    }
}

/// Couples completion of the owned commit task to writer availability. The
/// default is deliberately fail-closed: a pre-publication error or panic keeps
/// workers paused so the retained prepared generation can be retried. Only the
/// normal published path arms resumption.
struct CommitFinalizationGuard<D: DirectoryWriter + 'static> {
    state: Arc<CommitFinalizationState>,
    worker_state: Arc<WorkerState<D>>,
    doc_sender: Arc<parking_lot::RwLock<async_channel::Sender<Document>>>,
    resume_workers: bool,
}

impl<D: DirectoryWriter + 'static> CommitFinalizationGuard<D> {
    fn resume_on_drop(&mut self) {
        self.resume_workers = true;
    }
}

impl<D: DirectoryWriter + 'static> Drop for CommitFinalizationGuard<D> {
    fn drop(&mut self) {
        if self.resume_workers {
            IndexWriter::<D>::resume_workers_shared(&self.worker_state, &self.doc_sender);
        }
        self.state.finish();
    }
}

/// Everything needed to finish one prepared generation is moved into this
/// value before spawning. Its two guards therefore reconcile segment
/// ownership and writer availability even if Tokio drops the task before its
/// first poll.
struct OwnedCommitFinalization<D: DirectoryWriter + 'static> {
    directory: Arc<D>,
    schema: Arc<Schema>,
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
    primary_key_index: Arc<parking_lot::RwLock<Option<super::primary_key::PrimaryKeyIndex>>>,
    prepared: PreparedSegmentsGuard<D>,
    finalization: Option<CommitFinalizationGuard<D>>,
    publication_observed: Arc<AtomicBool>,
    pk_reservations_retained: Arc<AtomicBool>,
}

async fn refresh_primary_key_after_commit<D: DirectoryWriter + 'static>(
    directory: &Arc<D>,
    schema: &Arc<Schema>,
    segment_manager: &Arc<crate::merge::SegmentManager<D>>,
    primary_key_index: &Arc<parking_lot::RwLock<Option<super::primary_key::PrimaryKeyIndex>>>,
) -> Result<()> {
    let existing_ids: std::collections::HashSet<String> = {
        let guard = primary_key_index.read();
        let Some(pk_index) = guard.as_ref() else {
            return Ok(());
        };
        pk_index
            .committed_segment_ids()
            .map(ToOwned::to_owned)
            .collect()
    };

    let snapshot = segment_manager.acquire_snapshot().await;
    let load_futures: Vec<_> = snapshot
        .segment_ids()
        .iter()
        .filter(|id| !existing_ids.contains(id.as_str()))
        .map(|seg_id_str| {
            let seg_id_str = seg_id_str.clone();
            let dir = directory.as_ref();
            let schema = Arc::clone(schema);
            async move { load_pk_segment_data(dir, &seg_id_str, &schema).await }
        })
        .collect();
    let new_data = futures::future::try_join_all(load_futures).await?;
    let seg_ids: Vec<String> = snapshot.segment_ids().to_vec();

    let bloom_file = {
        let mut guard = primary_key_index.write();
        let Some(pk_index) = guard.as_mut() else {
            return Ok(());
        };
        pk_index.refresh_incremental(new_data, snapshot);
        let bloom_bytes = pk_index.bloom_to_bytes();
        super::primary_key::serialize_pk_bloom(&seg_ids, &bloom_bytes)
    };

    if let Err(error) = directory
        .write(
            std::path::Path::new(super::primary_key::PK_BLOOM_FILE),
            &bloom_file,
        )
        .await
    {
        log::warn!("[primary_key] failed to persist bloom cache: {}", error);
    }
    Ok(())
}

async fn finalize_prepared_commit<D: DirectoryWriter + 'static>(
    mut commit: OwnedCommitFinalization<D>,
) -> Result<bool> {
    let metadata_entries = commit.prepared.metadata_entries();
    let published_segment_ids = commit.prepared.vector_upgrade_segment_ids();

    // This entire future is owned by a Tokio task. Cancelling the RPC only
    // drops its JoinHandle; it cannot split durable metadata publication from
    // PK reservations or worker resumption.
    commit.segment_manager.commit(&metadata_entries).await?;
    commit.publication_observed.store(true, Ordering::Release);

    let mut published = commit.prepared.take_published();
    for segment in &mut published {
        segment.mark_published();
    }
    drop(published);
    commit
        .segment_manager
        .schedule_vector_segment_upgrades(published_segment_ids);
    // Publication is irreversible. From here onward every exit path, including
    // panic unwind, must make the writer available again while PK reservations
    // remain fail-closed until refresh succeeds.
    if let Some(finalization) = commit.finalization.as_mut() {
        finalization.resume_on_drop();
    } else {
        log::error!("owned commit finalization guard was already released after publication");
    }

    // Metadata publication is the commit point. Cache refresh is fail-closed:
    // retaining the generation's uncommitted keys may cause conservative
    // duplicate rejections, but can never admit a duplicate or turn a durable
    // commit into an API error.
    match refresh_primary_key_after_commit(
        &commit.directory,
        &commit.schema,
        &commit.segment_manager,
        &commit.primary_key_index,
    )
    .await
    {
        // A successful refresh folded every committed key into committed_data
        // and cleared the reservations — nothing retained anymore.
        Ok(()) => commit
            .pk_reservations_retained
            .store(false, Ordering::Release),
        Err(error) => {
            // The retained reservations are now the ONLY record of the
            // published segments' keys. Abort paths must not clear them
            // (see clear_uncommitted_pk_reservations) or duplicates would
            // be admitted.
            commit
                .pk_reservations_retained
                .store(true, Ordering::Release);
            log::error!(
                "[primary_key] committed metadata but failed to refresh dedup state; \
                 retaining reservations until a later successful commit: {}",
                error,
            );
        }
    }

    // Merge scheduling is optional post-commit work and may briefly wait on
    // manager state. Reconcile worker availability first so it cannot extend
    // ingestion backpressure after metadata and PK state already agree.
    drop(commit.finalization.take());
    commit.segment_manager.maybe_merge().await;
    Ok(true)
}

impl<'a, D: DirectoryWriter + 'static> PreparedCommit<'a, D> {
    /// Finalize: register segments in metadata, evaluate merge policy, resume workers.
    ///
    /// Returns `true` if new segments were committed, `false` if nothing changed.
    pub async fn commit(mut self) -> Result<bool> {
        let segments = std::mem::take(&mut *self.writer.flushed_segments.lock());

        // Fast path: nothing to commit
        if segments.is_empty() {
            log::debug!("[commit] no segments to commit, skipping");
            self.is_resolved = true;
            self.writer.resume_workers();
            return Ok(false);
        }

        if !self.writer.commit_finalization.begin() {
            self.writer.flushed_segments.lock().extend(segments);
            // Keep the prepared generation paused. Letting `Drop` auto-abort
            // here would delete the retryable segments owned by another
            // finalization state transition.
            self.is_resolved = true;
            return Err(Error::CommitInProgress);
        }

        let publication_observed = Arc::new(AtomicBool::new(false));
        let owned = OwnedCommitFinalization {
            directory: Arc::clone(&self.writer.directory),
            schema: Arc::clone(&self.writer.schema),
            segment_manager: Arc::clone(&self.writer.segment_manager),
            primary_key_index: Arc::clone(&self.writer.primary_key_index),
            prepared: PreparedSegmentsGuard {
                segments: Some(segments),
                retry_slot: Arc::clone(&self.writer.flushed_segments),
            },
            finalization: Some(CommitFinalizationGuard {
                state: Arc::clone(&self.writer.commit_finalization),
                worker_state: Arc::clone(&self.writer.worker_state),
                doc_sender: Arc::clone(&self.writer.doc_sender),
                resume_workers: false,
            }),
            publication_observed: Arc::clone(&publication_observed),
            pk_reservations_retained: Arc::clone(&self.writer.pk_reservations_retained),
        };

        // From this point the owned value, not this cancel-sensitive guard,
        // controls every segment and the paused worker generation. Resolve the
        // local guard before spawning so even a runtime-spawn panic cannot
        // auto-abort the retryable generation during unwind.
        self.is_resolved = true;
        let task_publication = Arc::clone(&publication_observed);
        let task = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tokio::spawn(async move {
                match std::panic::AssertUnwindSafe(finalize_prepared_commit(owned))
                    .catch_unwind()
                    .await
                {
                    Ok(result) => result,
                    Err(_) if task_publication.load(Ordering::Acquire) => {
                        log::error!(
                            "owned commit finalizer panicked after metadata publication; \
                             treating the durable generation as committed"
                        );
                        Ok(true)
                    }
                    Err(_) => Err(Error::Internal(
                        "owned commit finalizer panicked before metadata publication".into(),
                    )),
                }
            })
        }))
        .map_err(|_| Error::Internal("runtime rejected owned commit finalizer".into()))?;

        match task.await {
            Ok(result) => result,
            Err(error) if publication_observed.load(Ordering::Acquire) => {
                log::error!(
                    "owned commit finalizer terminated after metadata publication: {}; \
                     treating the durable generation as committed",
                    error,
                );
                Ok(true)
            }
            Err(error) => Err(Error::Internal(format!(
                "owned commit finalizer terminated unexpectedly: {error}"
            ))),
        }
    }

    /// Abort: discard prepared segments, delete their files asynchronously,
    /// and resume workers. Lifecycle ownership is held until deletion ends.
    pub fn abort(mut self) {
        self.is_resolved = true;
        self.writer.flushed_segments.lock().clear();
        self.writer.clear_uncommitted_pk_reservations();
        self.writer.resume_workers();
    }
}

impl<D: DirectoryWriter + 'static> Drop for PreparedCommit<'_, D> {
    fn drop(&mut self) {
        if !self.is_resolved {
            log::warn!("PreparedCommit dropped without commit/abort — auto-aborting");
            self.writer.flushed_segments.lock().clear();
            self.writer.clear_uncommitted_pk_reservations();
            self.writer.resume_workers();
        }
    }
}

/// Load only fast-field data for a segment (lightweight alternative to full SegmentReader).
async fn load_pk_segment_data<D: crate::directories::Directory>(
    dir: &D,
    seg_id_str: &str,
    schema: &Arc<crate::dsl::Schema>,
) -> Result<super::primary_key::PkSegmentData> {
    let seg_id = crate::segment::SegmentId::from_hex(seg_id_str)
        .ok_or_else(|| Error::Internal(format!("Invalid segment id: {}", seg_id_str)))?;
    let files = crate::segment::SegmentFiles::new(seg_id.0);
    let fast_fields =
        crate::segment::reader::loader::load_fast_fields_file(dir, &files, schema).await?;
    Ok(super::primary_key::PkSegmentData {
        segment_id: seg_id_str.to_string(),
        fast_fields,
    })
}
