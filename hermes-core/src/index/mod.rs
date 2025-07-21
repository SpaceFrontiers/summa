//! Index - multi-segment async search index
//!
//! The `Index` is the central concept that provides:
//! - `Index::create()` / `Index::open()` - create or open an index
//! - `index.writer()` - get an IndexWriter for adding documents
//! - `index.reader()` - get an IndexReader for searching (with reload policy)
//!
//! The Index owns the SegmentManager which handles segment lifecycle and tracking.

#[cfg(feature = "native")]
use crate::dsl::Schema;
#[cfg(feature = "native")]
use crate::error::Result;
#[cfg(feature = "sync")]
use std::collections::HashMap;
#[cfg(feature = "native")]
use std::sync::Arc;
#[cfg(feature = "native")]
use std::sync::{OnceLock, Weak};

mod searcher;
pub use searcher::Searcher;

#[cfg(feature = "native")]
mod primary_key;
#[cfg(feature = "native")]
mod reader;
#[cfg(feature = "native")]
mod vector_builder;
#[cfg(all(feature = "wasm", not(feature = "native")))]
mod wasm_writer;
#[cfg(feature = "native")]
mod writer;
#[cfg(feature = "native")]
pub use primary_key::PrimaryKeyIndex;
#[cfg(feature = "native")]
pub use reader::IndexReader;
#[cfg(all(feature = "wasm", not(feature = "native")))]
pub use wasm_writer::IndexWriter as WasmIndexWriter;
#[cfg(feature = "native")]
pub use writer::{IndexWriter, PreparedCommit, WRITER_LOCK_FILENAME};

mod metadata;
pub use metadata::{
    FieldVectorMeta, INDEX_META_FILENAME, IndexMetadata, SegmentMetaInfo, VectorIndexState,
};

#[cfg(feature = "native")]
mod helpers;
#[cfg(feature = "native")]
pub use helpers::{
    IndexingStats, SchemaConfig, SchemaFieldConfig, create_index_at_path, create_index_from_sdl,
    index_documents_from_reader, index_json_document, parse_schema,
};

/// Default file name for the slice cache
pub const SLICE_CACHE_FILENAME: &str = "index.slicecache";

/// A BP pass can consume every background CPU worker and the complete
/// per-pass memory allowance. More than two simultaneous passes only
/// oversubscribe the same pool and multiply memory-bandwidth pressure.
#[cfg(feature = "native")]
pub const MAX_CONCURRENT_REORDER_PASSES: usize = 2;

#[cfg(feature = "native")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReorderPriority {
    /// Periodic optimizer and explicit standalone reorder work. This class
    /// must retain capacity while automatic merges continuously arrive,
    /// otherwise fresh segments stay unordered for minutes behind giant BP
    /// passes and query pruning recovers slowly after ingestion.
    Optimizer,
    /// BP performed while producing an automatic merge output.
    AutomaticMerge,
    Foreground,
}

/// Application-wide gate shared by optimizer, merge-time, and manual BP.
///
/// Besides enforcing the hard two-pass ceiling, the gate lets an explicit
/// force merge reserve all but one slot. Background passes already running
/// finish normally; new ones wait until the force merge releases its guard.
#[cfg(feature = "native")]
#[derive(Debug)]
pub struct ReorderConcurrencyGate {
    permits: Arc<tokio::sync::Semaphore>,
    /// Automatic merges may consume all but one whole-pass slot. The reserved
    /// slot lets short optimizer passes continuously retire fresh segments.
    /// With a one-pass configuration both classes share the only slot.
    automatic_merge_permits: Arc<tokio::sync::Semaphore>,
    limit: usize,
    foreground_lock: Arc<tokio::sync::Mutex<()>>,
    foreground_active: std::sync::atomic::AtomicBool,
    foreground_finished: tokio::sync::Notify,
}

/// Process-wide cap on simultaneously active BMP segment scorers.
///
/// Each scorer performs random mmap reads. Letting every segment of every
/// concurrent query run at once multiplies page faults without increasing
/// useful NVMe throughput, so this gate is independent from the CPU pool.
#[cfg(feature = "native")]
#[derive(Debug)]
pub(crate) struct BmpIoGate {
    limit: usize,
    active: parking_lot::Mutex<usize>,
    available: parking_lot::Condvar,
    async_available: tokio::sync::Notify,
}

#[cfg(feature = "native")]
impl BmpIoGate {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            active: parking_lot::Mutex::new(0),
            available: parking_lot::Condvar::new(),
            async_available: tokio::sync::Notify::new(),
        }
    }

    fn acquire(&self) -> BmpIoPermit<'_> {
        let mut active = self.active.lock();
        while *active >= self.limit {
            self.available.wait(&mut active);
        }
        *active += 1;
        BmpIoPermit { gate: self }
    }

    async fn acquire_async(&self) -> BmpIoPermit<'_> {
        loop {
            // Register before checking the counter, so a release between the
            // check and await cannot be lost.
            let notified = self.async_available.notified();
            {
                let mut active = self.active.lock();
                if *active < self.limit {
                    *active += 1;
                    return BmpIoPermit { gate: self };
                }
            }
            notified.await;
        }
    }
}

#[cfg(feature = "native")]
struct BmpIoPermit<'a> {
    gate: &'a BmpIoGate,
}

#[cfg(feature = "native")]
impl Drop for BmpIoPermit<'_> {
    fn drop(&mut self) {
        let mut active = self.gate.active.lock();
        *active -= 1;
        self.gate.available.notify_one();
        self.gate.async_available.notify_one();
    }
}

#[cfg(feature = "native")]
impl ReorderConcurrencyGate {
    pub fn new(requested_limit: usize) -> Self {
        let limit = requested_limit.clamp(1, MAX_CONCURRENT_REORDER_PASSES);
        let automatic_merge_limit = limit.saturating_sub(1).max(1);
        Self {
            permits: Arc::new(tokio::sync::Semaphore::new(limit)),
            automatic_merge_permits: Arc::new(tokio::sync::Semaphore::new(automatic_merge_limit)),
            limit,
            foreground_lock: Arc::new(tokio::sync::Mutex::new(())),
            foreground_active: std::sync::atomic::AtomicBool::new(false),
            foreground_finished: tokio::sync::Notify::new(),
        }
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    pub(crate) async fn acquire(
        self: &Arc<Self>,
        priority: ReorderPriority,
    ) -> std::result::Result<ReorderPermit, tokio::sync::AcquireError> {
        match priority {
            ReorderPriority::Optimizer => self.acquire_background(None).await,
            ReorderPriority::AutomaticMerge => {
                let merge_permit = Arc::clone(&self.automatic_merge_permits)
                    .acquire_owned()
                    .await?;
                self.acquire_background(Some(merge_permit)).await
            }
            ReorderPriority::Foreground => self.acquire_foreground().await,
        }
    }

    /// Acquire capacity for periodic optimizer or automatic merge work.
    async fn acquire_background(
        self: &Arc<Self>,
        automatic_merge: Option<tokio::sync::OwnedSemaphorePermit>,
    ) -> std::result::Result<ReorderPermit, tokio::sync::AcquireError> {
        loop {
            if self
                .foreground_active
                .load(std::sync::atomic::Ordering::Acquire)
            {
                let notified = self.foreground_finished.notified();
                if self
                    .foreground_active
                    .load(std::sync::atomic::Ordering::Acquire)
                {
                    notified.await;
                    continue;
                }
            }

            let permit = Arc::clone(&self.permits).acquire_owned().await?;
            if !self
                .foreground_active
                .load(std::sync::atomic::Ordering::Acquire)
            {
                return Ok(ReorderPermit {
                    _permit: permit,
                    _automatic_merge: automatic_merge,
                });
            }
            // A foreground operation started between the check and permit
            // acquisition. Yield the slot instead of extending its queue.
            drop(permit);
        }
    }

    /// Acquire the one BP slot left available to a foreground force merge.
    async fn acquire_foreground(
        self: &Arc<Self>,
    ) -> std::result::Result<ReorderPermit, tokio::sync::AcquireError> {
        let permit = Arc::clone(&self.permits).acquire_owned().await?;
        Ok(ReorderPermit {
            _permit: permit,
            _automatic_merge: None,
        })
    }

    /// Prioritize one explicit force merge across all indexes using this gate.
    ///
    /// Foreground operations are serialized to avoid two force merges each
    /// reserving one slot and then waiting for the other. The guard is
    /// cancellation-safe and releases reservations on drop.
    pub(crate) async fn begin_foreground(
        self: &Arc<Self>,
    ) -> std::result::Result<ForegroundReorderGuard, tokio::sync::AcquireError> {
        let exclusive = Arc::clone(&self.foreground_lock).lock_owned().await;
        self.foreground_active
            .store(true, std::sync::atomic::Ordering::Release);

        // Construct the guard before awaiting capacity. If this future is
        // cancelled while existing background work drains, Drop clears the
        // active flag and releases the foreground mutex.
        let mut guard = ForegroundReorderGuard {
            gate: Arc::clone(self),
            reserved: None,
            _exclusive: exclusive,
        };
        if self.limit > 1 {
            guard.reserved = Some(
                Arc::clone(&self.permits)
                    .acquire_many_owned((self.limit - 1) as u32)
                    .await?,
            );
        }
        Ok(guard)
    }
}

#[cfg(feature = "native")]
pub(crate) struct ReorderPermit {
    _permit: tokio::sync::OwnedSemaphorePermit,
    _automatic_merge: Option<tokio::sync::OwnedSemaphorePermit>,
}

#[cfg(feature = "native")]
pub(crate) struct ForegroundReorderGuard {
    gate: Arc<ReorderConcurrencyGate>,
    reserved: Option<tokio::sync::OwnedSemaphorePermit>,
    _exclusive: tokio::sync::OwnedMutexGuard<()>,
}

#[cfg(feature = "native")]
impl Drop for ForegroundReorderGuard {
    fn drop(&mut self) {
        // Make capacity visible before waking background waiters.
        drop(self.reserved.take());
        self.gate
            .foreground_active
            .store(false, std::sync::atomic::Ordering::Release);
        self.gate.foreground_finished.notify_waiters();
    }
}

/// Index configuration
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Number of threads shared by CPU-intensive search work.
    ///
    /// Indexes in the same process that request the same width reuse one Rayon
    /// pool. A value of zero is invalid and is rejected by `Index::create` and
    /// `Index::open`.
    pub num_threads: usize,
    /// Maximum BMP segment scorers issuing random mmap reads concurrently
    /// across the process. CPU parallelism remains controlled by
    /// `num_threads`; this separate cap protects the page cache and storage
    /// queue from segment/query fan-out.
    pub bmp_io_concurrency: usize,
    /// Number of parallel segment builders (documents distributed round-robin)
    pub num_indexing_threads: usize,
    /// Number of threads for parallel block compression within each segment
    pub num_compression_threads: usize,
    /// Block cache size for term dictionary per segment
    pub term_cache_blocks: usize,
    /// Process-wide byte budget for decompressed document-store blocks.
    ///
    /// Indexes opened with the same budget share one read-concurrent,
    /// byte-bounded cache. This is a byte limit rather than a block count
    /// because a stored document can legitimately make one decompressed block
    /// tens of MiB.
    pub store_cache_budget_bytes: usize,
    /// Max memory (bytes) across all builders before auto-commit (global limit)
    pub max_indexing_memory_bytes: usize,
    /// Maximum vectors retained for one field's global ANN training sample.
    /// The byte budget below is applied at the same time; the smaller bound
    /// wins. Fields are sampled and trained serially.
    pub vector_training_max_samples: usize,
    /// Maximum raw vector bytes retained for one field's ANN training sample.
    pub vector_training_memory_bytes: usize,
    /// Merge policy for background segment merging
    pub merge_policy: Box<dyn crate::merge::MergePolicy>,
    /// Index optimization mode (adaptive, size-optimized, performance-optimized)
    pub optimization: crate::structures::IndexOptimization,
    /// Reload interval in milliseconds for IndexReader (how often to check for new segments)
    pub reload_interval_ms: u64,
    /// Maximum number of concurrent background merges per index (default: 4)
    pub max_concurrent_merges: usize,
    /// Application-wide background merge gate shared by clones of this
    /// config. The per-index limit alone multiplied large merge working sets
    /// by the number of active indexes.
    #[cfg(feature = "native")]
    pub background_merge_permits: Arc<tokio::sync::Semaphore>,
    /// Wall-clock budget for merge-time BP reorder per field (only applies
    /// when the index has `reorder_on_merge`). A truncated pass still writes
    /// a valid, better-ordered segment; it is marked `bp_converged = false`
    /// and the background optimizer deepens it later (warm-started).
    /// `None` = unbudgeted (BP runs to full depth inside the merge, which can
    /// hold a merge slot for 10-30+ minutes on 10M+ doc outputs).
    pub merge_bp_time_budget: Option<std::time::Duration>,
    /// Memory budget (bytes) for the BP forward index during reorder passes
    /// (merge-time and background). When a large segment's forward index
    /// would exceed this, the highest-df dims are dropped from BP's input
    /// (logged loudly) — clustering quality degrades gracefully. Production
    /// evidence: 18M-doc merges exceeded the former 2 GB default and dropped
    /// ~10% of eligible dims; hosts with less headroom may lower this.
    pub bp_memory_budget_bytes: usize,
    /// Hard limit on simultaneous whole-segment BP rewrites. This is shared
    /// by all indexes opened from clones of this config and applies to
    /// optimizer, merge-time, and manual reorder passes. It is deliberately
    /// separate from the Rayon pool width: one pass can already use every
    /// background CPU thread and consume the full BP memory budget.
    #[cfg(feature = "native")]
    pub background_reorder_permits: Arc<ReorderConcurrencyGate>,
    /// Optional process/application-owned Rayon pool for BP work. Supplying
    /// one lets every index and the optimizer share the same worker threads;
    /// `None` lazily uses one process-wide cores/2 fallback pool.
    #[cfg(feature = "native")]
    pub background_reorder_pool: Option<Arc<rayon::ThreadPool>>,
}

/// Search pools are shared process-wide by width. This avoids multiplying OS
/// threads by the number of open indexes while still allowing applications to
/// deliberately isolate indexes that need different CPU budgets.
#[cfg(feature = "sync")]
static SEARCH_CPU_POOLS: OnceLock<parking_lot::Mutex<HashMap<usize, Weak<rayon::ThreadPool>>>> =
    OnceLock::new();

/// Store caches are shared process-wide by configured byte budget, just like
/// search CPU pools are shared by width. `IndexRegistry` clones one config for
/// every index, but standalone callers with the same policy also converge on
/// the same bounded cache.
#[cfg(feature = "native")]
static STORE_CACHE_POOLS: OnceLock<
    parking_lot::Mutex<std::collections::HashMap<usize, Weak<crate::segment::SharedStoreCache>>>,
> = OnceLock::new();

#[cfg(feature = "native")]
static BMP_IO_GATES: OnceLock<
    parking_lot::Mutex<std::collections::HashMap<usize, Weak<BmpIoGate>>>,
> = OnceLock::new();

#[cfg(feature = "native")]
pub(crate) fn shared_bmp_io_gate(limit: usize) -> Arc<BmpIoGate> {
    let mut gates = BMP_IO_GATES
        .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()))
        .lock();
    if let Some(gate) = gates.get(&limit).and_then(Weak::upgrade) {
        return gate;
    }
    let gate = Arc::new(BmpIoGate::new(limit));
    gates.retain(|_, gate| gate.strong_count() > 0);
    gates.insert(limit, Arc::downgrade(&gate));
    log::info!("[bmp] process-wide random-I/O concurrency={limit}");
    gate
}

#[cfg(feature = "native")]
pub(crate) fn shared_store_cache(budget_bytes: usize) -> Arc<crate::segment::SharedStoreCache> {
    let mut caches = STORE_CACHE_POOLS
        .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()))
        .lock();
    if let Some(cache) = caches.get(&budget_bytes).and_then(Weak::upgrade) {
        return cache;
    }
    let cache = Arc::new(crate::segment::SharedStoreCache::new(budget_bytes));
    caches.retain(|_, cache| cache.strong_count() > 0);
    caches.insert(budget_bytes, Arc::downgrade(&cache));
    log::info!(
        "[store_cache] process-wide budget={}",
        crate::format_bytes(budget_bytes as u64)
    );
    cache
}

#[cfg(feature = "sync")]
fn shared_search_pool(num_threads: usize) -> Result<Arc<rayon::ThreadPool>> {
    if num_threads == 0 {
        return Err(crate::Error::Internal(
            "IndexConfig.num_threads must be greater than zero".into(),
        ));
    }

    let mut pools = SEARCH_CPU_POOLS
        .get_or_init(|| parking_lot::Mutex::new(HashMap::new()))
        .lock();
    if let Some(pool) = pools.get(&num_threads).and_then(Weak::upgrade) {
        return Ok(pool);
    }

    // Build while holding the registry lock. Index construction is cold-path
    // work, and serialization here prevents two concurrent opens from creating
    // duplicate pools for the same width.
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(move |idx| format!("hermes-search-{}-{}", num_threads, idx))
            .build()
            .map_err(|error| {
                crate::Error::Internal(format!(
                    "failed to create {num_threads}-thread search pool: {error}"
                ))
            })?,
    );
    pools.retain(|_, pool| pool.strong_count() > 0);
    pools.insert(num_threads, Arc::downgrade(&pool));
    log::info!("[search] process-wide CPU pool: {} thread(s)", num_threads);
    Ok(pool)
}

impl Default for IndexConfig {
    fn default() -> Self {
        #[cfg(feature = "native")]
        let compression_threads = crate::default_compression_threads();
        #[cfg(not(feature = "native"))]
        let compression_threads = 1;

        #[cfg(feature = "native")]
        let search_threads = crate::default_search_threads();
        #[cfg(not(feature = "native"))]
        let search_threads = 1;

        Self {
            num_threads: search_threads,
            bmp_io_concurrency: 4,
            num_indexing_threads: 1, // Increase to 2+ for production to avoid stalls during segment build
            num_compression_threads: compression_threads,
            term_cache_blocks: 256,
            // Stored bodies can be much larger than the writer's nominal
            // 16-KiB block target. Keep this process-wide and byte bounded so
            // segment fan-out cannot multiply it into tens of GiB.
            #[cfg(target_pointer_width = "64")]
            store_cache_budget_bytes: 2 * 1024 * 1024 * 1024,
            #[cfg(not(target_pointer_width = "64"))]
            store_cache_budget_bytes: 32 * 1024 * 1024,
            max_indexing_memory_bytes: 256 * 1024 * 1024, // 256 MB default
            vector_training_max_samples: 10_000_000,
            #[cfg(target_pointer_width = "64")]
            vector_training_memory_bytes: 4 * 1024 * 1024 * 1024,
            #[cfg(not(target_pointer_width = "64"))]
            vector_training_memory_bytes: usize::MAX,
            // large_scale: wide fan-in + budget/scored selection. Safe for
            // small indexes too (tier floors only shape *when* segments
            // merge); merge-time BP is wall-clock budgeted, so giant merges
            // cannot hold slots indefinitely.
            merge_policy: Box::new(crate::merge::TieredMergePolicy::large_scale()),
            optimization: crate::structures::IndexOptimization::default(),
            reload_interval_ms: 1000, // 1 second default
            max_concurrent_merges: 4,
            #[cfg(feature = "native")]
            background_merge_permits: Arc::new(tokio::sync::Semaphore::new(4)),
            merge_bp_time_budget: Some(std::time::Duration::from_secs(600)),
            // 24 GB — mirrors segment::reorder::DEFAULT_MEMORY_BUDGET (that
            // module is native-only; IndexConfig also compiles for wasm).
            // A cap, not an allocation: usage is proportional to the segment
            // being reordered (~4 B/posting + ~32 B/doc). Sized from prod
            // evidence: a 58M-doc/5B-posting pass estimated 20.1 GB, which
            // 8/16 GB budgets trimmed by dropping highest-df dims.
            // 24 GB overflows 32-bit usize (wasm32) — reorder never runs
            // there, so any large value works; use usize::MAX.
            #[cfg(target_pointer_width = "64")]
            bp_memory_budget_bytes: 24 * 1024 * 1024 * 1024,
            #[cfg(not(target_pointer_width = "64"))]
            bp_memory_budget_bytes: usize::MAX,
            #[cfg(feature = "native")]
            background_reorder_permits: Arc::new(ReorderConcurrencyGate::new(2)),
            #[cfg(feature = "native")]
            background_reorder_pool: None,
        }
    }
}

/// Multi-segment async Index
///
/// The central concept for search. Owns segment lifecycle and provides:
/// - `Index::create()` / `Index::open()` - create or open an index
/// - `index.writer()` - get an IndexWriter for adding documents
/// - `index.reader()` - get an IndexReader for searching with reload policy
///
/// All segment management is delegated to SegmentManager.
#[cfg(feature = "native")]
pub struct Index<D: crate::directories::DirectoryWriter + 'static> {
    directory: Arc<D>,
    schema: Arc<Schema>,
    config: IndexConfig,
    /// Cache and CPU policy used by every searcher reload.
    search_resources: searcher::SearcherResources,
    /// Segment manager - owns segments, tracker, metadata, and trained structures
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Cached reader (created lazily, reused across calls)
    cached_reader: tokio::sync::OnceCell<IndexReader<D>>,
}

#[cfg(feature = "native")]
impl<D: crate::directories::DirectoryWriter + 'static> Index<D> {
    /// Create a new index in the directory
    pub async fn create(directory: D, schema: Schema, config: IndexConfig) -> Result<Self> {
        let search_resources = searcher::SearcherResources::new(
            config.term_cache_blocks,
            config.store_cache_budget_bytes,
            config.num_threads,
            config.bmp_io_concurrency,
        )?;
        let directory = Arc::new(directory);
        let schema = Arc::new(schema);
        // Directory-layer metrics (cold writes, lazy reads) carry the index label
        directory.set_index_label(schema.index_label());

        // Refuse to clobber an existing index: persisting a fresh empty
        // metadata.json would orphan every committed segment, and the next
        // writer open's orphan sweep would permanently delete them.
        if directory
            .exists(std::path::Path::new(INDEX_META_FILENAME))
            .await?
        {
            return Err(crate::Error::Internal(format!(
                "refusing to create index: {} already exists in this directory; \
                 use Index::open to open the existing index, or delete the \
                 directory first if you really want to start over",
                INDEX_META_FILENAME
            )));
        }

        let metadata = IndexMetadata::new((*schema).clone());

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

        // Save initial metadata
        segment_manager.update_metadata(|_| {}).await?;

        Ok(Self {
            directory,
            schema,
            config,
            search_resources,
            segment_manager,
            cached_reader: tokio::sync::OnceCell::new(),
        })
    }

    /// Open an existing index from a directory
    pub async fn open(directory: D, config: IndexConfig) -> Result<Self> {
        let search_resources = searcher::SearcherResources::new(
            config.term_cache_blocks,
            config.store_cache_budget_bytes,
            config.num_threads,
            config.bmp_io_concurrency,
        )?;
        let directory = Arc::new(directory);

        // Load metadata (includes schema)
        let metadata = IndexMetadata::load(directory.as_ref()).await?;
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

        // Load trained structures into SegmentManager's ArcSwap
        segment_manager.try_load_and_publish_trained().await?;

        Ok(Self {
            directory,
            schema,
            config,
            search_resources,
            segment_manager,
            cached_reader: tokio::sync::OnceCell::new(),
        })
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get the schema as an Arc reference (avoids clone when Arc is needed)
    pub fn schema_arc(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Get a reference to the underlying directory
    pub fn directory(&self) -> &D {
        &self.directory
    }

    /// Get the segment manager
    pub fn segment_manager(&self) -> &Arc<crate::merge::SegmentManager<D>> {
        &self.segment_manager
    }

    /// Get an IndexReader for searching (with reload policy)
    ///
    /// The reader is cached and reused across calls. The reader's internal
    /// searcher will reload segments based on its reload interval (configurable via IndexConfig).
    pub async fn reader(&self) -> Result<&IndexReader<D>> {
        self.cached_reader
            .get_or_try_init(|| async {
                IndexReader::from_segment_manager_with_resources(
                    Arc::clone(&self.schema),
                    Arc::clone(&self.segment_manager),
                    self.config.reload_interval_ms,
                    self.search_resources.clone(),
                )
                .await
            })
            .await
    }

    /// Get the config
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    /// Get segment readers for query execution (convenience method)
    pub async fn segment_readers(&self) -> Result<Vec<Arc<crate::segment::SegmentReader>>> {
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;
        Ok(searcher.segment_readers().to_vec())
    }

    /// Total number of documents across all segments
    pub async fn num_docs(&self) -> Result<u32> {
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;
        Ok(searcher.num_docs())
    }

    /// Get default fields for search
    pub fn default_fields(&self) -> Vec<crate::Field> {
        if !self.schema.default_fields().is_empty() {
            self.schema.default_fields().to_vec()
        } else {
            self.schema
                .fields()
                .filter(|(_, entry)| {
                    entry.indexed && entry.field_type == crate::dsl::FieldType::Text
                })
                .map(|(field, _)| field)
                .collect()
        }
    }

    /// Get tokenizer registry
    pub fn tokenizers(&self) -> Arc<crate::tokenizer::TokenizerRegistry> {
        Arc::new(crate::tokenizer::TokenizerRegistry::default())
    }

    /// Create a query parser for this index
    pub fn query_parser(&self) -> crate::dsl::QueryLanguageParser {
        let default_fields = self.default_fields();
        let tokenizers = self.tokenizers();

        let query_routers = self.schema.query_routers();
        if !query_routers.is_empty()
            && let Ok(router) = crate::dsl::QueryFieldRouter::from_rules(query_routers)
        {
            return crate::dsl::QueryLanguageParser::with_router(
                Arc::clone(&self.schema),
                default_fields,
                tokenizers,
                router,
            );
        }

        crate::dsl::QueryLanguageParser::new(Arc::clone(&self.schema), default_fields, tokenizers)
    }

    /// Parse and search using a query string
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
        let query = parser
            .parse(query_str)
            .map_err(crate::error::Error::Query)?;
        self.search_offset(query.as_ref(), limit, offset).await
    }

    /// Search and return results
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
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;

        #[cfg(feature = "sync")]
        let (results, total_seen) = {
            // Sync search: rayon handles segment parallelism internally.
            // On multi-threaded tokio, use block_in_place to yield the worker;
            // on single-threaded (tests), call directly.
            let runtime_flavor = tokio::runtime::Handle::current().runtime_flavor();
            if runtime_flavor == tokio::runtime::RuntimeFlavor::MultiThread {
                tokio::task::block_in_place(|| {
                    searcher.search_with_offset_and_count_sync(query, limit, offset)
                })?
            } else {
                searcher.search_with_offset_and_count_sync(query, limit, offset)?
            }
        };

        #[cfg(not(feature = "sync"))]
        let (results, total_seen) = {
            searcher
                .search_with_offset_and_count(query, limit, offset)
                .await?
        };

        let total_hits = total_seen;
        let hits: Vec<crate::query::SearchHit> = results
            .into_iter()
            .map(|result| crate::query::SearchHit {
                address: crate::query::DocAddress::new(result.segment_id, result.doc_id),
                score: result.score,
                matched_fields: result.extract_ordinals(),
            })
            .collect();

        Ok(crate::query::SearchResponse { hits, total_hits })
    }

    /// Get a document by its unique address
    pub async fn get_document(
        &self,
        address: &crate::query::DocAddress,
    ) -> Result<Option<crate::dsl::Document>> {
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;
        searcher.get_document(address).await
    }

    /// Get posting lists for a term across all segments
    pub async fn get_postings(
        &self,
        field: crate::Field,
        term: &[u8],
    ) -> Result<
        Vec<(
            Arc<crate::segment::SegmentReader>,
            crate::structures::BlockPostingList,
        )>,
    > {
        let segments = self.segment_readers().await?;
        let mut results = Vec::new();

        for segment in segments {
            if let Some(postings) = segment.get_postings(field, term).await? {
                results.push((segment, postings));
            }
        }

        Ok(results)
    }
}

/// Native-only methods for Index
#[cfg(feature = "native")]
impl<D: crate::directories::DirectoryWriter + 'static> Index<D> {
    /// Get an IndexWriter for adding documents
    pub fn writer(&self) -> writer::IndexWriter<D> {
        writer::IndexWriter::from_index(self)
    }
}

#[cfg(test)]
mod tests;

// (tests moved to index/tests/ module)
