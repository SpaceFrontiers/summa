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

#[cfg(any(feature = "native", feature = "wasm"))]
mod content_hash;
#[cfg(any(feature = "native", feature = "wasm"))]
mod primary_key;
#[cfg(feature = "native")]
mod reader;
#[cfg(any(feature = "native", feature = "wasm"))]
pub(crate) mod staged_row;
#[cfg(feature = "native")]
mod vector_builder;
#[cfg(all(feature = "wasm", not(feature = "native")))]
mod wasm_writer;
#[cfg(feature = "native")]
mod writer;
#[cfg(any(feature = "native", feature = "wasm"))]
pub use primary_key::PrimaryKeyIndex;
#[cfg(feature = "native")]
pub use reader::IndexReader;
#[cfg(feature = "native")]
pub use vector_builder::{AlterVectorIndexOutcome, AlterVectorIndexState};
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
    /// Standalone optimizer passes rewrite complete sparse blobs and can each
    /// consume the full per-pass memory budget. Serializing this class avoids
    /// multiplying disk traffic and retained source/output pages when a scan
    /// discovers candidates in multiple indexes at once.
    optimizer_permits: Arc<tokio::sync::Semaphore>,
    /// Automatic merges may consume all but one whole-pass slot. The reserved
    /// slot lets short optimizer passes continuously retire fresh segments.
    /// With a one-pass configuration both classes share the only slot.
    automatic_merge_permits: Arc<tokio::sync::Semaphore>,
    limit: usize,
    foreground_lock: Arc<tokio::sync::Mutex<()>>,
    foreground_active: std::sync::atomic::AtomicBool,
    foreground_finished: tokio::sync::Notify,
}

/// Process-wide cap on simultaneously active sparse segment scorers.
///
/// Each scorer performs random mmap reads. Letting every segment of every
/// concurrent query run at once multiplies page faults without increasing
/// useful NVMe throughput, so this gate is independent from the CPU pool.
#[cfg(feature = "native")]
#[derive(Debug)]
pub(crate) struct SparseIoGate {
    limit: usize,
    active: parking_lot::Mutex<usize>,
    available: parking_lot::Condvar,
    async_available: tokio::sync::Notify,
}

#[cfg(feature = "native")]
impl SparseIoGate {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            active: parking_lot::Mutex::new(0),
            available: parking_lot::Condvar::new(),
            async_available: tokio::sync::Notify::new(),
        }
    }

    #[cfg(feature = "sync")]
    fn acquire(&self) -> SparseIoPermit<'_> {
        let mut active = self.active.lock();
        while *active >= self.limit {
            self.available.wait(&mut active);
        }
        *active += 1;
        SparseIoPermit { gate: self }
    }

    async fn acquire_async(&self) -> SparseIoPermit<'_> {
        loop {
            // Register before checking the counter, so a release between the
            // check and await cannot be lost.
            let notified = self.async_available.notified();
            {
                let mut active = self.active.lock();
                if *active < self.limit {
                    *active += 1;
                    return SparseIoPermit { gate: self };
                }
            }
            notified.await;
        }
    }
}

#[cfg(feature = "native")]
struct SparseIoPermit<'a> {
    gate: &'a SparseIoGate,
}

#[cfg(feature = "native")]
impl Drop for SparseIoPermit<'_> {
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
            optimizer_permits: Arc::new(tokio::sync::Semaphore::new(1)),
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

    /// Periodic maintenance must not occupy a queued task while foreground
    /// work or another full sparse rewrite owns this shared capacity.
    pub(crate) fn try_acquire_optimizer(
        self: &Arc<Self>,
    ) -> std::result::Result<ReorderPermit, tokio::sync::TryAcquireError> {
        use std::sync::atomic::Ordering;
        use tokio::sync::TryAcquireError;
        if self.foreground_active.load(Ordering::Acquire) {
            return Err(TryAcquireError::NoPermits);
        }
        let optimizer = Arc::clone(&self.optimizer_permits).try_acquire_owned()?;
        let permit = Arc::clone(&self.permits).try_acquire_owned()?;
        if self.foreground_active.load(Ordering::Acquire) {
            return Err(TryAcquireError::NoPermits);
        }
        Ok(ReorderPermit {
            _permit: permit,
            _optimizer: Some(optimizer),
            _automatic_merge: None,
        })
    }

    pub(crate) async fn acquire(
        self: &Arc<Self>,
        priority: ReorderPriority,
    ) -> std::result::Result<ReorderPermit, tokio::sync::AcquireError> {
        match priority {
            ReorderPriority::Optimizer => {
                let optimizer_permit = Arc::clone(&self.optimizer_permits).acquire_owned().await?;
                self.acquire_background(Some(optimizer_permit), None).await
            }
            ReorderPriority::AutomaticMerge => {
                let merge_permit = Arc::clone(&self.automatic_merge_permits)
                    .acquire_owned()
                    .await?;
                self.acquire_background(None, Some(merge_permit)).await
            }
            ReorderPriority::Foreground => self.acquire_foreground().await,
        }
    }

    /// Acquire capacity for periodic optimizer or automatic merge work.
    async fn acquire_background(
        self: &Arc<Self>,
        optimizer: Option<tokio::sync::OwnedSemaphorePermit>,
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
                    _optimizer: optimizer,
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
            _optimizer: None,
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
    _optimizer: Option<tokio::sync::OwnedSemaphorePermit>,
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
    /// Maximum sparse segment scorers issuing random mmap reads concurrently
    /// across the process. CPU parallelism remains controlled by
    /// `num_threads`; this separate cap protects the page cache and storage
    /// queue from segment/query fan-out.
    pub sparse_io_concurrency: usize,
    /// Number of parallel segment builders (documents distributed round-robin)
    pub num_indexing_threads: usize,
    /// Width of the document-store compression pool. Concurrent segment
    /// builders requesting the same width share one process-wide pool, so
    /// indexing-worker fan-out does not multiply this thread count.
    pub num_compression_threads: usize,
    /// Block cache size for term dictionary per segment
    pub term_cache_blocks: usize,
    /// Optional per-segment cap on retained decompressed dictionary-block bytes.
    /// None preserves the block-count policy; zero disables retention.
    pub term_cache_budget_bytes: Option<usize>,
    /// Flush target for newly written term dictionaries; default 16 KiB.
    pub term_dict_block_size: crate::structures::SSTableBlockSize,
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
    /// Index optimization mode (adaptive, size-optimized, performance-optimized).
    /// Selects the term-dictionary compression level and, unless
    /// `posting_codec` overrides it, the posting block codec
    /// (`docs/posting-codecs.md`).
    pub optimization: crate::structures::IndexOptimization,
    /// Explicit posting block codec; `None` derives it from `optimization`
    /// (`size` → `Pfor`, everything else → `Rounded`).
    pub posting_codec: Option<crate::structures::PostingCodec>,
    /// New plain-text columns use versioned byte4 norms. Existing segments retain their scores.
    pub quantized_norms: bool,
    /// New position streams use a compact directory separate from payload pages.
    pub compact_text: bool,
    /// Opt in to compact, score-independent length/TF block bounds.
    ///
    /// Applies to new segments only. Merges, compaction, and reorder copy or
    /// re-encode each list in the representation its sources already have:
    /// existing blocks keep their layout and are never upgraded, not even by
    /// `force_merge`. Rebuild (re-index) to add bounds to old data. Opening
    /// an index whose segments lack the enabled bounds logs this once.
    pub posting_ratio_bounds: bool,
    /// Opt in to bounded competitive frequency/length envelopes. Implies ratio
    /// bounds (`effective_posting_bounds`). Same new-segments-only policy as
    /// `posting_ratio_bounds`.
    pub posting_impact_bounds: bool,
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
    /// Scratch limit for explicit or background physical row compaction.
    pub compaction_memory_budget_bytes: usize,
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
static SPARSE_IO_GATES: OnceLock<
    parking_lot::Mutex<std::collections::HashMap<usize, Weak<SparseIoGate>>>,
> = OnceLock::new();

/// Announce each resource kind once per process. Weak registry entries can
/// expire between index opens; subsequent creations (including different
/// settings) remain visible at DEBUG without retaining resources or an
/// unbounded history of configurations just for logging.
#[cfg(feature = "native")]
fn shared_resource_log_level(announced: &OnceLock<()>) -> log::Level {
    if announced.set(()).is_ok() {
        log::Level::Info
    } else {
        log::Level::Debug
    }
}

#[cfg(feature = "native")]
pub(crate) fn shared_sparse_io_gate(limit: usize) -> Arc<SparseIoGate> {
    let mut gates = SPARSE_IO_GATES
        .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()))
        .lock();
    if let Some(gate) = gates.get(&limit).and_then(Weak::upgrade) {
        return gate;
    }
    let gate = Arc::new(SparseIoGate::new(limit));
    gates.retain(|_, gate| gate.strong_count() > 0);
    gates.insert(limit, Arc::downgrade(&gate));
    static ANNOUNCED: OnceLock<()> = OnceLock::new();
    log::log!(
        shared_resource_log_level(&ANNOUNCED),
        "[sparse] process-wide random-I/O concurrency={limit}"
    );
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
    static ANNOUNCED: OnceLock<()> = OnceLock::new();
    log::log!(
        shared_resource_log_level(&ANNOUNCED),
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
    static ANNOUNCED: OnceLock<()> = OnceLock::new();
    log::log!(
        shared_resource_log_level(&ANNOUNCED),
        "[search] process-wide CPU pool: {} thread(s)",
        num_threads
    );
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
            sparse_io_concurrency: 4,
            num_indexing_threads: 1, // Increase to 2+ for production to avoid stalls during segment build
            num_compression_threads: compression_threads,
            term_cache_blocks: 256,
            term_cache_budget_bytes: None,
            term_dict_block_size: crate::structures::SSTableBlockSize::default(),
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
            posting_codec: None,
            quantized_norms: false,
            compact_text: false,
            posting_ratio_bounds: false,
            posting_impact_bounds: false,
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
            compaction_memory_budget_bytes: 256 * 1024 * 1024,
            #[cfg(feature = "native")]
            background_reorder_permits: Arc::new(ReorderConcurrencyGate::new(2)),
            #[cfg(feature = "native")]
            background_reorder_pool: None,
        }
    }
}

/// Largest `IndexConfig::term_cache_blocks`; the per-segment dictionary block
/// cache is sized by count and this keeps a typo from pinning a whole
/// dictionary per segment.
pub const MAX_TERM_CACHE_BLOCKS: usize = 65_536;

/// Reject an out-of-range dictionary block cap before any segment is opened.
#[cfg(feature = "native")]
pub(crate) fn validate_term_cache_blocks(blocks: usize) -> crate::Result<()> {
    if blocks > MAX_TERM_CACHE_BLOCKS {
        return Err(crate::Error::Internal(format!(
            "IndexConfig.term_cache_blocks must be at most {MAX_TERM_CACHE_BLOCKS} (got {blocks})"
        )));
    }
    Ok(())
}

/// Block-bound metadata layout new posting lists are written with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PostingBounds {
    /// Score-independent length/TF ratio minima per block and L1 group.
    pub ratio: bool,
    /// Competitive frequency/length envelopes (always together with ratios).
    pub impact: bool,
}

impl PostingBounds {
    pub(crate) fn new(ratio: bool, impact: bool) -> Self {
        Self {
            ratio: ratio || impact,
            impact,
        }
    }
}

impl IndexConfig {
    /// Posting block codec new segments and merges are written with.
    pub fn effective_posting_codec(&self) -> crate::structures::PostingCodec {
        self.posting_codec
            .unwrap_or_else(|| self.optimization.default_posting_codec())
    }

    /// Block-bound metadata new segments are written with. Impact bounds
    /// imply ratio bounds; this is the single place that rule is applied.
    pub fn effective_posting_bounds(&self) -> PostingBounds {
        PostingBounds::new(self.posting_ratio_bounds, self.posting_impact_bounds)
    }
}

/// Segments probed by [`segments_missing_posting_bounds`] and dictionary
/// entries scanned per segment before the probe gives up as inconclusive.
#[cfg(feature = "native")]
const POSTING_BOUNDS_PROBE_SEGMENTS: usize = 32;
#[cfg(feature = "native")]
const POSTING_BOUNDS_PROBE_TERMS: usize = 4096;

/// Existing segments whose posting lists lack the block-bound metadata
/// `config` enables (`posting_ratio_bounds` / `posting_impact_bounds`).
///
/// Bounds apply to new segments only; nothing upgrades old blocks. This
/// probe reads one term dictionary prefix and one external posting list per
/// segment (bounded by the constants above) so an operator learns at open
/// time that the option is not retroactive. Returns `(missing, probed)`.
/// Impact envelopes exist only on multi-block lists, so the probe checks
/// ratio metadata, which both options write.
#[cfg(feature = "native")]
pub(crate) async fn segments_missing_posting_bounds<D: crate::directories::Directory>(
    directory: &D,
    metadata: &IndexMetadata,
    config: &IndexConfig,
) -> Result<(Vec<String>, usize)> {
    use crate::segment::{SegmentFiles, SegmentId};
    use crate::structures::{AsyncSSTableReader, BlockPostingList, TermInfo};

    let mut missing = Vec::new();
    let mut probed = 0usize;
    if !config.effective_posting_bounds().ratio {
        return Ok((missing, probed));
    }
    for id in metadata
        .segment_ids()
        .into_iter()
        .take(POSTING_BOUNDS_PROBE_SEGMENTS)
    {
        let Some(segment_id) = SegmentId::from_hex(&id) else {
            continue;
        };
        let files = SegmentFiles::new(segment_id.0);
        if !directory.exists(&files.term_dict).await? {
            continue;
        }
        let term_dict = AsyncSSTableReader::<TermInfo>::open_with_cache_budget(
            directory.open_lazy(&files.term_dict).await?,
            1,
            None,
        )
        .await?;
        let mut terms = term_dict.iter();
        let mut external = None;
        for _ in 0..POSTING_BOUNDS_PROBE_TERMS {
            match terms.next().await? {
                Some((_, info)) => {
                    if let Some(range) = info.external_info() {
                        external = Some(range);
                        break;
                    }
                }
                None => break,
            }
        }
        let Some((offset, len)) = external else {
            continue;
        };
        let postings = directory.open_lazy(&files.postings).await?;
        let end = offset.checked_add(len).ok_or_else(|| {
            crate::Error::Corruption("posting range overflow while probing bounds".into())
        })?;
        let list =
            BlockPostingList::deserialize_zero_copy(postings.read_bytes_range(offset..end).await?)?;
        probed += 1;
        if !list.has_ratio_bounds() {
            missing.push(id);
        }
    }
    Ok((missing, probed))
}

/// Log once per open when enabled posting bounds do not cover existing
/// segments. Probe failures are logged, never fatal: a corrupt segment fails
/// loudly when it is actually opened.
#[cfg(feature = "native")]
async fn log_posting_bounds_policy<D: crate::directories::Directory>(
    directory: &D,
    metadata: &IndexMetadata,
    config: &IndexConfig,
) {
    match segments_missing_posting_bounds(directory, metadata, config).await {
        Ok((missing, probed)) if !missing.is_empty() => log::info!(
            "[index] {}: posting_ratio_bounds/posting_impact_bounds are enabled but {} of {} \
             probed existing segments carry no block-bound metadata (e.g. {}). Bounds apply \
             to new segments only; merges and compaction keep existing block layouts. \
             Re-index to add bounds to old data.",
            metadata.schema.index_label(),
            missing.len(),
            probed,
            missing[0]
        ),
        Ok(_) => {}
        Err(error) => log::warn!(
            "[index] {}: could not probe existing segments for posting bounds: {}",
            metadata.schema.index_label(),
            error
        ),
    }
}

/// Build the segment-lifecycle owner from the corresponding index policy.
///
/// `Index` and `IndexWriter` both support create/open entry points. Routing
/// their shared configuration through this helper prevents a new
/// `SegmentManager` option from being wired into only some constructors.
#[cfg(feature = "native")]
fn segment_manager_from_config<D: crate::directories::DirectoryWriter + 'static>(
    directory: &Arc<D>,
    schema: &Arc<Schema>,
    metadata: IndexMetadata,
    config: &IndexConfig,
) -> Result<Arc<crate::merge::SegmentManager<D>>> {
    // Writer-only opens never build `SearcherResources`; lifecycle readers
    // still size their dictionary caches from this value.
    validate_term_cache_blocks(config.term_cache_blocks)?;
    Ok(Arc::new(
        crate::merge::SegmentManager::new(
            Arc::clone(directory),
            Arc::clone(schema),
            metadata,
            config.merge_policy.clone_box(),
            config.term_cache_blocks,
            config.max_concurrent_merges,
            Arc::clone(&config.background_merge_permits),
            config.merge_bp_time_budget,
            config.bp_memory_budget_bytes,
            Arc::clone(&config.background_reorder_permits),
            config.background_reorder_pool.clone(),
        )
        .with_posting_config(config.optimization, config.effective_posting_codec())
        .with_term_dict_block_size(config.term_dict_block_size)
        .with_term_cache_budget(config.term_cache_budget_bytes),
    ))
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
        schema.validate()?;
        let search_resources = searcher::SearcherResources::from_config(&config)?;
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

        let segment_manager = segment_manager_from_config(&directory, &schema, metadata, &config)?;

        // Save initial metadata
        segment_manager.update_metadata(|_| {}).await?;

        Ok(Self {
            directory,
            config,
            search_resources,
            segment_manager,
            cached_reader: tokio::sync::OnceCell::new(),
        })
    }

    /// Open an existing index from a directory
    pub async fn open(directory: D, config: IndexConfig) -> Result<Self> {
        let search_resources = searcher::SearcherResources::from_config(&config)?;
        let directory = Arc::new(directory);

        // Load metadata (includes schema)
        let metadata = IndexMetadata::load(directory.as_ref()).await?;
        let schema = Arc::new(metadata.schema.clone());
        // Directory-layer metrics (cold writes, lazy reads) carry the index label
        directory.set_index_label(schema.index_label());
        log_posting_bounds_policy(directory.as_ref(), &metadata, &config).await;

        let segment_manager = segment_manager_from_config(&directory, &schema, metadata, &config)?;

        // Load trained structures into SegmentManager's ArcSwap
        segment_manager.try_load_and_publish_trained().await?;

        Ok(Self {
            directory,
            config,
            search_resources,
            segment_manager,
            cached_reader: tokio::sync::OnceCell::new(),
        })
    }

    /// Open a search index and its sole writer under one lifecycle owner.
    ///
    /// The writer lock precedes metadata loading and crash cleanup. Use this
    /// when opening for mutation; `open` remains a read-only operation.
    pub async fn open_with_writer(
        directory: D,
        config: IndexConfig,
    ) -> Result<(Self, IndexWriter<D>)> {
        let search_resources = searcher::SearcherResources::from_config(&config)?;
        let writer = IndexWriter::open(directory, config.clone()).await?;
        let index = Self {
            directory: Arc::clone(&writer.directory),
            config,
            search_resources,
            segment_manager: Arc::clone(writer.segment_manager()),
            cached_reader: tokio::sync::OnceCell::new(),
        };
        Ok((index, writer))
    }

    /// Get the schema
    pub fn schema(&self) -> Arc<Schema> {
        self.schema_arc()
    }

    /// Clone the schema handle from the currently published generation.
    pub fn schema_arc(&self) -> Arc<Schema> {
        self.segment_manager.published_generation().schema.clone()
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
                    self.schema_arc(),
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
        let schema = self.schema_arc();
        if !schema.default_fields().is_empty() {
            schema.default_fields().to_vec()
        } else {
            schema
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
        let schema = self.schema_arc();

        let query_routers = schema.query_routers();
        if !query_routers.is_empty()
            && let Ok(router) = crate::dsl::QueryFieldRouter::from_rules(query_routers)
        {
            return crate::dsl::QueryLanguageParser::with_router(
                Arc::clone(&schema),
                default_fields,
                tokenizers,
                router,
            );
        }

        crate::dsl::QueryLanguageParser::new(schema, default_fields, tokenizers)
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
