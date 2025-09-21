//! Shared scoring abstractions for text and sparse vector search
//!
//! Provides common types and executors for efficient top-k retrieval:
//! - `TermCursor`: Unified cursor for both BM25 text and sparse vector posting lists
//! - `ScoreCollector`: Efficient min-heap for maintaining top-k results
//! - `MaxScoreExecutor`: Unified Block-Max MaxScore with conjunction optimization
//! - `ScoredDoc`: Result type with doc_id, score, and ordinal

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use log::{debug, warn};

use crate::DocId;

mod conjunction;
mod windows;

/// Avoid eagerly reserving an arbitrarily large top-k heap. Most searches
/// return far fewer hits than a very large requested limit, so let the heap
/// grow on demand beyond this point.
const MAX_INITIAL_SCORE_COLLECTOR_CAPACITY: usize = 8 * 1024;

/// Entry for top-k min-heap
#[derive(Clone, Copy)]
pub struct HeapEntry {
    pub doc_id: DocId,
    pub score: f32,
    pub ordinal: u16,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits()
            && self.doc_id == other.doc_id
            && self.ordinal == other.ordinal
    }
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower scores come first (to be evicted).
        // Keep a total float order and deterministic doc/ordinal tie breaks.
        // The generic then_with closures inline; no callback allocation is needed.
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
            .then_with(|| self.ordinal.cmp(&other.ordinal))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Efficient top-k collector using min-heap (internal, scoring-layer)
///
/// Maintains the k highest-scoring documents using a min-heap where the
/// lowest score is at the top for O(1) threshold lookup and O(log k) eviction.
/// No deduplication — caller must ensure each doc_id is inserted only once.
///
/// This is intentionally separate from `TopKCollector` in `collector.rs`:
/// `ScoreCollector` is used inside `MaxScoreExecutor` where only `(doc_id,
/// score, ordinal)` tuples exist — no `Scorer` trait, no position tracking,
/// and the threshold must be inlined for tight block-max loops.
/// `TopKCollector` wraps a `Scorer` and drives the full `DocSet`/`Scorer`
/// protocol, collecting positions on demand.
pub struct ScoreCollector {
    /// Min-heap of top-k entries (lowest score at top for eviction)
    heap: BinaryHeap<HeapEntry>,
    pub k: usize,
    /// Cached threshold: avoids repeated heap.peek() in hot loops.
    /// Updated only when the heap changes (insert/pop).
    cached_threshold: f32,
    /// Score of the logical sentinel filling every unused top-k slot after
    /// threshold seeding. Keeping one score here instead of `k - heap.len()`
    /// entries makes filling those unused slots O(1) time and memory.
    virtual_threshold: Option<f32>,
}

impl ScoreCollector {
    /// Create a new collector for top-k results
    pub fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k.min(MAX_INITIAL_SCORE_COLLECTOR_CAPACITY)),
            k,
            cached_threshold: 0.0,
            virtual_threshold: None,
        }
    }

    /// Current score threshold (minimum score to enter top-k)
    #[inline]
    pub fn threshold(&self) -> f32 {
        self.cached_threshold
    }

    /// Recompute cached threshold from heap state
    #[inline]
    fn update_threshold(&mut self) {
        self.cached_threshold = if let Some(threshold) = self.virtual_threshold {
            threshold
        } else if self.heap.len() >= self.k {
            self.heap.peek().map(|e| e.score).unwrap_or(0.0)
        } else {
            0.0
        };
    }

    /// Insert a document score. Returns true if inserted in top-k.
    /// Caller must ensure each doc_id is inserted only once.
    #[inline]
    pub fn insert(&mut self, doc_id: DocId, score: f32) -> bool {
        self.insert_with_ordinal(doc_id, score, 0)
    }

    /// Insert a document score with ordinal. Returns true if inserted in top-k.
    /// Caller must ensure each doc_id is inserted only once.
    #[inline]
    pub fn insert_with_ordinal(&mut self, doc_id: DocId, score: f32, ordinal: u16) -> bool {
        if self.k == 0 {
            return false;
        }
        let entry = HeapEntry {
            doc_id,
            score,
            ordinal,
        };
        if self.heap.len() < self.k {
            if let Some(threshold) = self.virtual_threshold {
                let sentinel = HeapEntry {
                    doc_id: u32::MAX,
                    score: threshold,
                    ordinal: 0,
                };
                if entry >= sentinel {
                    return false;
                }
            }

            self.heap.push(entry);
            crate::observe::search_work!(maxscore_heap_updates += 1);
            // The final real entry displaces the last virtual sentinel.
            if self.heap.len() == self.k {
                self.virtual_threshold = None;
                self.update_threshold();
            }
            true
        } else if score < self.cached_threshold {
            false
        } else if self.heap.peek().is_some_and(|worst| entry < *worst) {
            {
                let mut worst = self.heap.peek_mut().expect("full heap has a root");
                *worst = entry;
            }
            self.update_threshold();
            crate::observe::search_work!(maxscore_heap_updates += 1);
            true
        } else {
            false
        }
    }

    /// Screen eight scores together, then use the canonical heap admission.
    /// A stale threshold only admits extra candidates. Equal and unordered
    /// scores still reach the total-order comparison, including seeded heaps.
    fn insert_text_run(&mut self, docs: &[DocId], scores: &[f32]) {
        self.insert_text_run_with_mapping(docs, scores, |doc| doc);
    }

    /// Resolve stable IDs only after the score screen, before heap tie-breaking.
    fn insert_text_run_with_mapping(
        &mut self,
        docs: &[DocId],
        scores: &[f32],
        resolve: impl Fn(DocId) -> DocId,
    ) {
        debug_assert_eq!(docs.len(), scores.len());
        crate::observe::search_work!(score_batches += 1);
        let (blocks, tail) = scores.as_chunks::<8>();
        for (docs, scores) in docs.chunks_exact(8).zip(blocks) {
            let threshold = if self.heap.len() >= self.k {
                self.cached_threshold
            } else {
                f32::NEG_INFINITY
            };
            let mut candidates = 0u8;
            for (i, &score) in scores.iter().enumerate() {
                let eligible = if score < threshold { 0 } else { 1 };
                candidates |= eligible << i;
            }
            while candidates != 0 {
                let i = candidates.trailing_zeros() as usize;
                self.insert(resolve(docs[i]), 0.0 + scores[i]);
                candidates &= candidates - 1;
            }
        }
        for (&doc, &score) in docs[blocks.len() * 8..].iter().zip(tail) {
            if self.heap.len() >= self.k && score < self.cached_threshold {
                continue;
            }
            self.insert(resolve(doc), 0.0 + score);
        }
    }

    /// Check if a score could potentially enter top-k
    #[cfg(test)]
    pub fn would_enter(&self, score: f32) -> bool {
        self.len() < self.k || score > self.cached_threshold
    }

    /// Check whether this fully identified candidate ranks ahead of the current
    /// worst retained entry, including deterministic tie breaks.
    #[cfg(test)]
    pub fn would_enter_candidate(&self, doc_id: DocId, score: f32, ordinal: u16) -> bool {
        if self.k == 0 {
            return false;
        }
        let entry = HeapEntry {
            doc_id,
            score,
            ordinal,
        };
        if let Some(threshold) = self.virtual_threshold {
            let sentinel = HeapEntry {
                doc_id: u32::MAX,
                score: threshold,
                ordinal: 0,
            };
            entry < sentinel
        } else {
            self.heap.len() < self.k || self.heap.peek().is_some_and(|worst| entry < *worst)
        }
    }

    /// Get the conceptual heap length, including virtual threshold sentinels.
    #[inline]
    pub fn len(&self) -> usize {
        if self.virtual_threshold.is_some() {
            self.k
        } else {
            self.heap.len()
        }
    }

    /// Number of real results retained, excluding threshold sentinels.
    #[inline]
    pub fn real_len(&self) -> usize {
        self.heap.len()
    }

    /// Check if collector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Seed the threshold from a cross-segment shared value.
    ///
    /// Logically fills unused slots and replaces retained entries below the new
    /// floor with virtual dummy entries. This can be called repeatedly while
    /// another segment raises the shared threshold; equal-scoring real
    /// candidates win the deterministic doc-id tie break over sentinels.
    pub fn seed_threshold(&mut self, initial_threshold: f32) {
        if initial_threshold <= 0.0
            || self.k == 0
            || (self.len() >= self.k && initial_threshold <= self.cached_threshold)
        {
            return;
        }

        let sentinel = HeapEntry {
            doc_id: u32::MAX,
            score: initial_threshold,
            ordinal: 0,
        };

        // When unused slots are already represented by a virtual sentinel, a
        // new seed only changes the heap if it outranks the old floor. With an
        // all-real full heap, it must similarly outrank the current root.
        if let Some(current_threshold) = self.virtual_threshold {
            let current = HeapEntry {
                doc_id: u32::MAX,
                score: current_threshold,
                ordinal: 0,
            };
            if sentinel >= current {
                return;
            }
        } else if self.heap.len() >= self.k
            && !self.heap.peek().is_some_and(|worst| sentinel < *worst)
        {
            return;
        }

        self.virtual_threshold = Some(initial_threshold);
        while self.heap.peek().is_some_and(|worst| sentinel < *worst) {
            self.heap.pop();
        }
        self.update_threshold();
    }

    /// Convert to sorted top-k results (descending by score).
    /// Filters out sentinel entries (doc_id == u32::MAX) from threshold seeding.
    pub fn into_sorted_results(self) -> Vec<(DocId, f32, u16)> {
        let mut results: Vec<(DocId, f32, u16)> = self
            .heap
            .into_vec()
            .into_iter()
            .filter(|e| e.doc_id != u32::MAX)
            .map(|e| (e.doc_id, e.score, e.ordinal))
            .collect();

        // Sort by score descending, then doc_id ascending
        results.sort_unstable_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.2.cmp(&b.2))
        });

        results
    }
}

/// Cross-segment top-k score floor, shared across the parallel/concurrent
/// per-segment searches of a single query.
///
/// Stores an `f32` as raw bits in an atomic so it can be read and monotonically
/// raised from many threads without a lock. Each segment reads the current
/// floor as its initial pruning threshold (`ScorerOptions::initial_threshold`)
/// and, once it has collected a *full* top-k of its own, raises the floor to
/// its k-th score.
///
/// Safety of seeding: a segment only raises the floor after filling its own
/// heap, so a floor value `v` is always backed by at least `k` real documents
/// scoring `>= v`. The final merged k-th score is therefore `>= v`, and seeding
/// any other segment with `v` can never drop a document that belongs in the
/// final top-k. Completion order is arbitrary, so the floor is best-effort — it
/// only changes how aggressively later segments prune, never correctness.
///
/// The floor carries the query's result-window depth `k` (`for_limit`).
/// Publishing from a heap shallower than `k` is invalid — a segment with
/// fewer documents than the window fills its clamped heap early, and its
/// heap threshold says nothing about the query-global k-th score. Executors
/// must check `SharedThreshold::covers` before raising the floor with a
/// full-heap threshold.
#[derive(Clone, Debug)]
pub struct SharedThreshold {
    floor: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Result-window depth the floor is valid for. `usize::MAX` means the
    /// depth is unknown; reading stays safe, publishing is disabled.
    k: usize,
    /// Wall-clock budget of the whole query (anytime mode): executors that
    /// honour it stop scoring once it passes and flag the result truncated.
    deadline: Option<std::time::Instant>,
    /// Set by any executor that stopped early because of `deadline`.
    truncated: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Default for SharedThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedThreshold {
    /// A fresh floor of 0.0 (no pruning seed) with an unknown window depth.
    /// Executors can read and manually raise it, but never publish their own
    /// full-heap thresholds into it.
    pub fn new() -> Self {
        Self::with_depth(usize::MAX)
    }

    /// A fresh floor valid for a query fetching `limit` results.
    pub fn for_limit(limit: usize) -> Self {
        Self::with_depth(limit)
    }

    fn with_depth(k: usize) -> Self {
        Self {
            // 0.0_f32.to_bits() == 0, matching AtomicU32::default().
            floor: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            k,
            deadline: None,
            truncated: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Attach a wall-clock budget (`None` = unbounded).
    pub fn with_deadline(mut self, deadline: Option<std::time::Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    /// The query's deadline, if any.
    pub fn deadline(&self) -> Option<std::time::Instant> {
        self.deadline
    }

    /// Keep cancellation/observability, but isolate a component's score space.
    pub(crate) fn budget_only(&self) -> Self {
        Self {
            floor: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            k: usize::MAX,
            deadline: self.deadline,
            truncated: self.truncated.clone(),
        }
    }

    #[inline]
    pub(crate) fn stop_if_expired(&self) -> bool {
        if self.expired() {
            self.mark_truncated();
            true
        } else {
            false
        }
    }

    /// Whether the deadline has passed.
    #[inline]
    pub fn expired(&self) -> bool {
        self.deadline
            .is_some_and(|deadline| std::time::Instant::now() >= deadline)
    }

    /// Record that an executor stopped early because the deadline passed.
    pub fn mark_truncated(&self) {
        self.truncated
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether any executor of this query stopped early.
    pub fn truncated(&self) -> bool {
        self.truncated.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// True when a full heap of `heap_depth` distinct documents backs a valid
    /// query-global floor for this threshold's result window.
    #[inline]
    pub(crate) fn covers(&self, heap_depth: usize) -> bool {
        heap_depth >= self.k
    }

    /// Current floor.
    #[inline]
    pub fn get(&self) -> f32 {
        f32::from_bits(self.floor.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Raise the floor to `score` if it is strictly higher. Monotonic; a lower
    /// or non-positive `score` is ignored. Scores here are BM25/sparse and thus
    /// non-negative, but the comparison is done on `f32` values (not raw bits)
    /// so it stays correct regardless.
    pub fn raise(&self, score: f32) {
        // Ignore non-positive scores; a NaN falls through harmlessly (the CAS
        // loop condition below is false for NaN, so nothing is stored).
        if score <= 0.0 {
            return;
        }
        use std::sync::atomic::Ordering::Relaxed;
        let bits = score.to_bits();
        let mut cur = self.floor.load(Relaxed);
        while f32::from_bits(cur) < score {
            match self
                .floor
                .compare_exchange_weak(cur, bits, Relaxed, Relaxed)
            {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }
}

/// Search result from MaxScore execution
#[derive(Debug, Clone, Copy)]
pub struct ScoredDoc {
    pub doc_id: DocId,
    pub score: f32,
    /// Ordinal for multi-valued fields (which vector in the field matched)
    pub ordinal: u16,
}

/// Unified Block-Max MaxScore executor for top-k retrieval
///
/// Works with both full-text (BM25) and sparse vector (dot product) queries
/// through the polymorphic `TermCursor`. Combines three optimizations:
/// 1. **MaxScore partitioning** (Turtle & Flood 1995): terms split into essential
///    (must check) and non-essential (only scored if candidate is promising)
/// 2. **Block-max pruning** (Ding & Suel 2011): skip blocks where per-block
///    upper bounds can't beat the current threshold
/// 3. **Conjunction optimization** (Lucene/Grand 2023): progressively intersect
///    essential terms as threshold rises, skipping docs that lack enough terms
pub struct MaxScoreExecutor<'a> {
    /// Metric labels (index, field) for the samples this executor emits.
    /// Default to `"unknown"`; callers set real schema names through
    /// [`Self::with_metric_labels`].
    metric_index: &'a str,
    metric_field: &'a str,
    cursors: Vec<TermCursor<'a>>,
    prefix_sums: Vec<f32>,
    /// Cursor indices in input term order, independent of pruning order.
    score_order: Vec<usize>,
    /// Semantic conjunction: batch only fully aligned eligible hits.
    all_required: bool,
    /// Semantic requirements in sorted cursor order; zero for ordinary unions.
    required_mask: u64,
    collector: ScoreCollector,
    /// Plain mapped fields rank ties by stable document ID, never physical slot.
    document_map: Option<&'a crate::segment::chunk_map::ChunkMap>,
    inv_heap_factor: f32,
    predicate: Option<super::DocPredicate<'a>>,
    /// Query-global budget: checked every few thousand loop iterations;
    /// an expired deadline ends traversal with the results so far.
    budget: Option<SharedThreshold>,
    /// Cursors dropped by the constructor at `MAX_QUERY_TERMS`; once any were
    /// dropped the input order is lost and required-term semantics are
    /// refused.
    dropped_cursors: usize,
    /// A rejected `require_*` configuration; surfaced as an error by
    /// `execute`/`execute_sync` instead of running with wrong semantics.
    configuration_error: Option<String>,
    /// Counters of the last run (summary log line, tests).
    stats: ExecutorStats,
}

/// Counters of an executor's last run, reported in its summary log line and
/// inspected by tests. Each path fills only the counters it tracks.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ExecutorStats {
    /// Windowed path: windows visited, windows skipped whole by their bounds,
    /// L1 groups skipped by impact bounds (also counted by the single path).
    pub windows: u64,
    pub windows_skipped: u64,
    pub groups_skipped: u64,
    /// Windowed path: candidates that survived to full scoring, documents
    /// that entered the heap.
    pub candidates: u64,
    pub docs_scored: u64,
    /// REQUIRED windows driven by a cheaper optional term because the
    /// required terms alone could not reach the threshold.
    pub optional_leads: u64,
    /// Single-cursor path: blocks decoded and scored / skipped by bounds.
    pub blocks_scored: u64,
    pub blocks_skipped: u64,
}

/// Where a text cursor reads the length of a scoring unit: chunk lengths of a
/// chunked field, or the persisted per-document field lengths (norms) of a
/// plain field. Without either, `tf` stands in for the length.
#[derive(Clone, Copy)]
pub enum LengthSource<'a> {
    Chunks(&'a crate::segment::chunk_map::ChunkMap),
    Docs(&'a crate::segment::chunk_map::DocLengths),
}

impl LengthSource<'_> {
    #[inline]
    pub fn length(&self, id: u32) -> u32 {
        match self {
            LengthSource::Chunks(map) => map.bm25_length(id),
            LengthSource::Docs(lengths) => lengths.length(id),
        }
    }

    pub(crate) fn gather_lengths(&self, ids: &[u32], out: &mut [u32]) {
        match self {
            LengthSource::Chunks(map) => map.gather_bm25_lengths(ids, out),
            LengthSource::Docs(lengths) => lengths.gather_lengths(ids, out),
        }
    }
}

/// Unified term cursor for Block-Max MaxScore execution.
///
/// All per-position decode buffers (`doc_ids`, `scores`, `ordinals`) live in
/// the struct directly and are filled by `ensure_block_loaded`.
///
/// Skip-list metadata is **not** materialized — it is read lazily from the
/// underlying source (`BlockPostingList` for text, `SparseIndex` for sparse),
/// both backed by zero-copy mmap'd `OwnedBytes`.
pub(crate) struct TermCursor<'a> {
    pub max_score: f32,
    num_blocks: usize,
    // ── Per-position state (filled by ensure_block_loaded) ──────────
    block_idx: usize,
    /// Decoded ids of the loaded block. Taken from the per-thread
    /// [`CursorBuffers`] pool and returned on drop.
    doc_ids: Vec<u32>,
    /// Scores of the loaded block (empty while a text block's TF decode is
    /// still deferred).
    scores: Vec<f32>,
    ordinals: Vec<u16>,
    /// Decoded term frequencies of the loaded text block.
    tfs: Vec<u32>,
    pos: usize,
    block_loaded: bool,
    exhausted: bool,
    // ── Lazy ordinal decode (sparse only) ───────────────────────────
    /// When true, ordinal decode is deferred until ordinal_mut() is called.
    /// Set to true for MaxScoreExecutor cursors (most blocks never need ordinals).
    lazy_ordinals: bool,
    /// Whether ordinals have been decoded for the current block.
    ordinals_loaded: bool,
    /// Stored sparse block for deferred ordinal decode (cheap Arc clone of mmap data).
    current_sparse_block: Option<crate::structures::SparseBlock>,
    // ── Block decode + skip access source ───────────────────────────
    variant: CursorVariant<'a>,
}

// One cursor per query term; the text variant carries the decoded-block
// state inline on purpose (no indirection on the scoring path).
#[allow(clippy::large_enum_variant)]
enum CursorVariant<'a> {
    /// Full-text BM25 — in-memory BlockPostingList (skip list + block data)
    Text {
        list: crate::structures::BlockPostingList,
        idf: f32,
        /// Real per-posting lengths (chunk lengths or document norms).
        /// `None` keeps the historic `tf`-as-length approximation.
        lengths: Option<LengthSource<'a>>,
        /// Block bounds may use the block's minimum length: only when the
        /// list stores one and scoring uses real lengths (a `tf`-as-length
        /// score is not bounded by a real-length bound).
        length_bounds: bool,
        length_floor: u32,
        block_bound: CachedScoreBound,
        group_bound: CachedScoreBound,
        prepared_bounds: Option<super::bm25::PreparedBounds>,
        /// Average length used by the bounds (matches the scoring average).
        avg_len: f32,
        /// Per-field k1/b, used by the block and group bounds.
        params: super::Bm25Params,
        normalization: Option<Box<super::bm25::NormTable>>,
        /// Deferred TF decode state: (block_offset, tf_start, count).
        /// Set when doc_ids are decoded but TFs are not. Candidate runs may
        /// decode TFs while leaving the full score vector empty.
        deferred_tf: Option<(usize, usize, usize)>,
    },
    /// Sparse vector — mmap'd SparseIndex (skip entries + block data)
    Sparse {
        si: &'a crate::segment::SparseIndex,
        query_weight: f32,
        skip_start: usize,
        block_data_offset: u64,
    },
}

/// Decode buffers of one cursor. A query builds one cursor per term and
/// each cursor needs three or four block-sized vectors, so they are pooled
/// per thread (the `BmpScratch` pattern) instead of being allocated per
/// query: [`TermCursor`] takes a set on construction and returns it on drop.
#[derive(Default)]
struct CursorBuffers {
    doc_ids: Vec<u32>,
    scores: Vec<f32>,
    ordinals: Vec<u16>,
    tfs: Vec<u32>,
}

/// Bound of the per-thread cursor-buffer pool: two full queries' worth.
const CURSOR_BUFFER_POOL_LIMIT: usize = 2 * super::MAX_QUERY_TERMS;

thread_local! {
    static CURSOR_BUFFERS: std::cell::RefCell<Vec<CursorBuffers>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

impl CursorBuffers {
    /// A pooled set, or a fresh one sized for a posting block.
    fn take() -> Self {
        CURSOR_BUFFERS
            .with(|pool| pool.borrow_mut().pop())
            .unwrap_or_else(|| {
                // The largest block either variant decodes: text blocks hold
                // `POSTING_BLOCK_SIZE` postings, sparse blocks up to 256
                // (`sparse::block::MAX_BLOCK_SIZE`), so a fresh set never
                // regrows and stays that size once pooled.
                const BLOCK: usize = if crate::structures::postings::POSTING_BLOCK_SIZE > 256 {
                    crate::structures::postings::POSTING_BLOCK_SIZE
                } else {
                    256
                };
                Self {
                    doc_ids: Vec::with_capacity(BLOCK),
                    scores: Vec::with_capacity(BLOCK),
                    ordinals: Vec::new(),
                    tfs: Vec::with_capacity(BLOCK),
                }
            })
    }

    /// Clear and return the set to the pool (dropped once the pool is full).
    fn recycle(mut self) {
        self.doc_ids.clear();
        self.scores.clear();
        self.ordinals.clear();
        self.tfs.clear();
        CURSOR_BUFFERS.with(|pool| {
            let mut pool = pool.borrow_mut();
            if pool.len() < CURSOR_BUFFER_POOL_LIMIT {
                pool.push(self);
            }
        });
    }
}

impl Drop for TermCursor<'_> {
    fn drop(&mut self) {
        CursorBuffers {
            doc_ids: std::mem::take(&mut self.doc_ids),
            scores: std::mem::take(&mut self.scores),
            ordinals: std::mem::take(&mut self.ordinals),
            tfs: std::mem::take(&mut self.tfs),
        }
        .recycle();
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn score_text_run(
    params: super::Bm25Params,
    idf: f32,
    avg_len: f32,
    lengths: Option<LengthSource<'_>>,
    normalization: Option<&super::bm25::NormTable>,
    docs: &[DocId],
    tfs: &[u32],
    scores: &mut [f32],
) {
    debug_assert_eq!(docs.len(), tfs.len());
    debug_assert_eq!(docs.len(), scores.len());
    crate::observe::search_work!(score_batches += 1);
    if let (Some(LengthSource::Docs(lengths)), Some(table)) = (lengths, normalization) {
        crate::observe::search_work!(lookup_score_units += docs.len());
        table.score_batch(
            params,
            idf,
            avg_len,
            1.0,
            docs.iter().map(|&doc| lengths.norm_code(doc)),
            tfs,
            scores,
        );
        return;
    }
    let mut gathered = [0; crate::structures::postings::POSTING_BLOCK_SIZE];
    crate::observe::search_work!(exact_score_units += docs.len());
    if let Some(source) = lengths {
        source.gather_lengths(docs, &mut gathered[..docs.len()]);
    }
    for ((&tf, &len), score) in tfs.iter().zip(&gathered[..docs.len()]).zip(scores) {
        let tf = tf as f32;
        let len = if len == 0 { tf } else { len as f32 };
        *score = params.score(tf, idf, len, avg_len);
    }
}

/// One query-local memoized bound, not a corpus-sized table. Atomic packing
/// keeps shared read access Sync without a lock or separate key/value races.
struct CachedScoreBound(std::sync::atomic::AtomicU64);

impl CachedScoreBound {
    fn new() -> Self {
        Self(std::sync::atomic::AtomicU64::new(u64::MAX))
    }

    fn get_or_compute(&self, key: usize, compute: impl FnOnce() -> f32) -> f32 {
        use std::sync::atomic::Ordering::Relaxed;
        // Posting block counts are bounded by the u32 posting-id space.
        let key = key as u64;
        let entry = self.0.load(Relaxed);
        if entry >> 32 == key {
            return f32::from_bits(entry as u32);
        }
        let score = compute();
        self.0
            .store((key << 32) | u64::from(score.to_bits()), Relaxed);
        score
    }
}

// ── TermCursor async/sync macros ──────────────────────────────────────────
//
// Parameterised on:
//   $load_block_fn – load_block_direct | load_block_direct_sync  (sparse I/O)
//   $ensure_fn     – ensure_block_loaded | ensure_block_loaded_sync
//   $($aw)*        – .await  (present for async, absent for sync)

macro_rules! cursor_ensure_block {
    ($self:ident, $load_block_fn:ident, $($aw:tt)*) => {{
        if $self.exhausted || $self.block_loaded {
            return Ok(!$self.exhausted);
        }
        match &mut $self.variant {
            CursorVariant::Text {
                list,
                deferred_tf,
                ..
            } => {
                if let Some(state) = list.decode_block_doc_ids_only($self.block_idx, &mut $self.doc_ids) {
                    *deferred_tf = Some(state);
                    $self.scores.clear();
                    $self.pos = 0;
                    $self.block_loaded = true;
                    Ok(true)
                } else {
                    // `block_idx < num_blocks` here, so a block that fails to
                    // decode is corrupt, not the end of the list. Surface it:
                    // treating it as exhaustion would silently truncate the
                    // top-k.
                    $self.exhausted = true;
                    Err(crate::Error::Corruption(format!(
                        "text posting block {} of {} failed to decode",
                        $self.block_idx, $self.num_blocks
                    )))
                }
            }
            CursorVariant::Sparse {
                si,
                query_weight,
                skip_start,
                block_data_offset,
                ..
            } => {
                let block = si
                    .$load_block_fn(*skip_start, *block_data_offset, $self.block_idx)
                    $($aw)* ?;
                match block {
                    Some(b) => {
                        b.decode_doc_ids_into(&mut $self.doc_ids);
                        b.decode_scored_weights_into(*query_weight, &mut $self.scores);
                        if $self.lazy_ordinals {
                            // Defer ordinal decode until ordinal_mut() is called.
                            // Stores cheap Arc-backed mmap slice, no copy.
                            $self.current_sparse_block = Some(b);
                            $self.ordinals_loaded = false;
                        } else {
                            b.decode_ordinals_into(&mut $self.ordinals);
                            $self.ordinals_loaded = true;
                            $self.current_sparse_block = None;
                        }
                        $self.pos = 0;
                        $self.block_loaded = true;
                        Ok(true)
                    }
                    None => {
                        $self.exhausted = true;
                        Ok(false)
                    }
                }
            }
        }
    }};
}

macro_rules! cursor_advance {
    ($self:ident, $ensure_fn:ident, $($aw:tt)*) => {{
        if $self.exhausted {
            return Ok(u32::MAX);
        }
        $self.$ensure_fn() $($aw)* ?;
        if $self.exhausted {
            return Ok(u32::MAX);
        }
        Ok($self.advance_pos())
    }};
}

macro_rules! cursor_seek {
    ($self:ident, $ensure_fn:ident, $target:expr, $($aw:tt)*) => {{
        if let Some(doc) = $self.seek_prepare($target) {
            return Ok(doc);
        }
        $self.$ensure_fn() $($aw)* ?;
        if $self.seek_finish($target) {
            $self.$ensure_fn() $($aw)* ?;
        }
        Ok($self.doc())
    }};
}

impl<'a> TermCursor<'a> {
    /// Full-text BM25 cursor with explicit per-field parameters. `lengths`
    /// supplies real scoring-unit lengths (chunk lengths or document norms);
    /// without it `tf` stands in for the length.
    pub fn text_with_params(
        posting_list: crate::structures::BlockPostingList,
        idf: f32,
        avg_field_len: f32,
        lengths: Option<LengthSource<'a>>,
        params: super::Bm25Params,
    ) -> Self {
        let posting_max_tf = posting_list.max_tf();
        let max_tf = posting_max_tf as f32;
        let safe_avg = avg_field_len.max(1.0);
        let length_bounds = lengths.is_some() && posting_list.min_len().is_some();
        let length_floor = match lengths {
            Some(LengthSource::Chunks(map)) => map.length_floor(),
            _ => 0,
        };
        let max_score = match posting_list.min_len() {
            Some(min_len) if length_bounds => params.upper_bound_with_len(
                max_tf.max(1.0),
                idf,
                min_len.max(length_floor) as f32,
                safe_avg,
            ),
            _ => params.upper_bound(max_tf.max(1.0), idf),
        };
        let num_blocks = posting_list.num_blocks();
        let buffers = CursorBuffers::take();
        Self {
            max_score,
            num_blocks,
            block_idx: 0,
            doc_ids: buffers.doc_ids,
            scores: buffers.scores,
            ordinals: buffers.ordinals,
            tfs: buffers.tfs,
            pos: 0,
            block_loaded: false,
            exhausted: num_blocks == 0,
            lazy_ordinals: false,
            ordinals_loaded: true, // text cursors never have ordinals
            current_sparse_block: None,
            variant: CursorVariant::Text {
                list: posting_list,
                idf,
                lengths,
                length_bounds,
                length_floor,
                block_bound: CachedScoreBound::new(),
                group_bound: CachedScoreBound::new(),
                prepared_bounds: super::bm25::PreparedBounds::new(
                    params,
                    posting_max_tf,
                    idf,
                    safe_avg,
                ),
                avg_len: safe_avg,
                params,
                normalization: match lengths {
                    Some(LengthSource::Docs(lengths)) if lengths.is_quantized() => {
                        Some(Box::new(super::bm25::NormTable::new(params, safe_avg)))
                    }
                    _ => None,
                },
                deferred_tf: None,
            },
        }
    }

    /// Create a sparse vector cursor with lazy block loading.
    /// Skip entries are **not** copied — they are read from `SparseIndex` mmap on demand.
    pub fn sparse(
        si: &'a crate::segment::SparseIndex,
        query_weight: f32,
        skip_start: usize,
        skip_count: usize,
        global_max_weight: f32,
        block_data_offset: u64,
    ) -> Self {
        let buffers = CursorBuffers::take();
        Self {
            max_score: query_weight.abs() * global_max_weight,
            num_blocks: skip_count,
            block_idx: 0,
            doc_ids: buffers.doc_ids,
            scores: buffers.scores,
            ordinals: buffers.ordinals,
            tfs: buffers.tfs,
            pos: 0,
            block_loaded: false,
            exhausted: skip_count == 0,
            lazy_ordinals: false,
            ordinals_loaded: true,
            current_sparse_block: None,
            variant: CursorVariant::Sparse {
                si,
                query_weight,
                skip_start,
                block_data_offset,
            },
        }
    }

    // ── Skip-entry access (lazy, zero-copy for sparse) ──────────────────

    #[inline]
    fn block_first_doc(&self, idx: usize) -> DocId {
        match &self.variant {
            CursorVariant::Text { list, .. } => list.block_first_doc(idx).unwrap_or(u32::MAX),
            CursorVariant::Sparse { si, skip_start, .. } => {
                si.read_skip_entry(*skip_start + idx).first_doc
            }
        }
    }

    #[inline]
    fn block_last_doc(&self, idx: usize) -> DocId {
        match &self.variant {
            CursorVariant::Text { list, .. } => list.block_last_doc(idx).unwrap_or(0),
            CursorVariant::Sparse { si, skip_start, .. } => {
                si.read_skip_entry(*skip_start + idx).last_doc
            }
        }
    }

    // ── Read-only accessors ─────────────────────────────────────────────

    #[inline]
    pub fn doc(&self) -> DocId {
        if self.exhausted {
            return u32::MAX;
        }
        if self.block_loaded {
            debug_assert!(self.pos < self.doc_ids.len());
            // SAFETY: pos < doc_ids.len() is maintained by advance_pos/ensure_block_loaded.
            unsafe { *self.doc_ids.get_unchecked(self.pos) }
        } else {
            self.block_first_doc(self.block_idx)
        }
    }

    #[inline]
    pub fn ordinal(&self) -> u16 {
        if !self.block_loaded || self.ordinals.is_empty() {
            return 0;
        }
        debug_assert!(self.pos < self.ordinals.len());
        // SAFETY: pos < ordinals.len() is maintained by advance_pos/ensure_block_loaded.
        unsafe { *self.ordinals.get_unchecked(self.pos) }
    }

    /// Lazily-decoded ordinal accessor for MaxScore executor.
    ///
    /// When `lazy_ordinals=true`, ordinals are not decoded during block loading.
    /// This method triggers the deferred decode on first access, amortized over
    /// the block. Subsequent calls within the same block are free.
    #[inline]
    fn ordinal_mut(&mut self) -> u16 {
        if !self.block_loaded {
            return 0;
        }
        if !self.ordinals_loaded {
            if let Some(ref block) = self.current_sparse_block {
                block.decode_ordinals_into(&mut self.ordinals);
            }
            self.ordinals_loaded = true;
        }
        if self.ordinals.is_empty() {
            return 0;
        }
        debug_assert!(self.pos < self.ordinals.len());
        unsafe { *self.ordinals.get_unchecked(self.pos) }
    }

    #[inline]
    pub fn score(&self) -> f32 {
        if !self.block_loaded {
            return 0.0;
        }
        debug_assert!(self.pos < self.scores.len());
        // SAFETY: pos < scores.len() is maintained by advance_pos/ensure_block_loaded.
        unsafe { *self.scores.get_unchecked(self.pos) }
    }

    /// Ensure BM25 scores are computed for the current block (lazy TF decode).
    ///
    /// For text cursors, TF unpacking and BM25 scoring are deferred from block
    /// loading until this method is called, saving work for blocks skipped by
    /// block-max or conjunction pruning. No-op for sparse cursors.
    #[inline]
    fn ensure_scores(&mut self) {
        if self.block_loaded && self.scores.is_empty() {
            self.compute_deferred_scores();
        }
    }

    #[inline]
    fn current_block_max_score(&self) -> f32 {
        if self.exhausted {
            return 0.0;
        }
        match &self.variant {
            CursorVariant::Text { .. } => self.text_block_bound(self.block_idx),
            CursorVariant::Sparse {
                si,
                query_weight,
                skip_start,
                ..
            } => query_weight.abs() * si.read_skip_entry(*skip_start + self.block_idx).max_weight,
        }
    }

    /// Upper bound over the L1 group (eight blocks) containing the current
    /// block, for text lists that store superblock bounds; `None` when the
    /// cursor cannot bound a whole group (sparse, legacy lists).
    #[inline]
    fn current_group_max_score(&self) -> Option<f32> {
        if self.exhausted {
            return Some(0.0);
        }
        match &self.variant {
            CursorVariant::Text { .. } => self.text_group_bound(self.block_idx),
            CursorVariant::Sparse { .. } => None,
        }
    }

    /// Whether this cursor reads an in-memory text posting list (all of its
    /// I/O is synchronous, so the windowed executor can drive it).
    #[inline]
    fn is_text(&self) -> bool {
        matches!(self.variant, CursorVariant::Text { .. })
    }

    /// Upper bound of text block `idx` from its `(max_tf, min_len)` word.
    fn text_block_bound(&self, idx: usize) -> f32 {
        self.text_block_bound_for_threshold(idx, f32::NEG_INFINITY)
    }

    /// A loose bound that already loses cannot benefit from impact refinement.
    /// Cached loose bounds remain conservative for every subsequent caller.
    fn text_block_bound_for_threshold(&self, idx: usize, threshold: f32) -> f32 {
        crate::observe::search_work!(block_bound_calls += 1);
        match &self.variant {
            CursorVariant::Text {
                list,
                idf,
                length_bounds,
                length_floor,
                prepared_bounds,
                block_bound,
                avg_len,
                params,
                ..
            } => block_bound.get_or_compute(idx, || {
                let (max_tf, min_len) = list.block_bounds(idx).unwrap_or((0, None));
                let bound = match min_len {
                    Some(min_len) if *length_bounds => params.upper_bound_with_len(
                        (max_tf as f32).max(1.0),
                        *idf,
                        min_len.max(*length_floor) as f32,
                        *avg_len,
                    ),
                    _ => params.upper_bound((max_tf as f32).max(1.0), *idf),
                };
                if *length_bounds {
                    let bound =
                        bound.min(prepared_bounds.as_ref().map_or(f32::INFINITY, |bounds| {
                            bounds.ratio(max_tf, list.block_length_ratio(idx))
                        }));
                    if bound >= threshold && list.has_impact_bounds() {
                        bound.min(prepared_bounds.as_ref().map_or(f32::INFINITY, |bounds| {
                            bounds.impacts(|a, b| list.block_impact_minimum(idx, a, b))
                        }))
                    } else {
                        bound
                    }
                } else {
                    bound
                }
            }),
            CursorVariant::Sparse { .. } => self.max_score,
        }
    }

    /// Upper bound of the L1 group containing text block `idx`.
    fn text_group_bound(&self, idx: usize) -> Option<f32> {
        self.text_group_bound_for_threshold(idx, f32::NEG_INFINITY)
    }

    fn text_group_bound_for_threshold(&self, idx: usize, threshold: f32) -> Option<f32> {
        crate::observe::search_work!(group_bound_calls += 1);
        match &self.variant {
            CursorVariant::Text {
                list,
                idf,
                length_bounds,
                length_floor,
                prepared_bounds,
                group_bound,
                avg_len,
                params,
                ..
            } => {
                let (max_tf, min_len) = list.group_bounds(idx)?;
                Some(group_bound.get_or_compute(list.next_group_block(idx), || {
                    if *length_bounds {
                        let bound = params
                            .upper_bound_with_len(
                                (max_tf as f32).max(1.0),
                                *idf,
                                min_len.max(*length_floor) as f32,
                                *avg_len,
                            )
                            .min(prepared_bounds.as_ref().map_or(f32::INFINITY, |bounds| {
                                bounds.ratio(max_tf, list.group_length_ratio(idx))
                            }));
                        if bound >= threshold && list.has_group_impact_bounds() {
                            bound.min(prepared_bounds.as_ref().map_or(f32::INFINITY, |bounds| {
                                bounds.impacts(|a, b| list.group_impact_minimum(idx, a, b))
                            }))
                        } else {
                            bound
                        }
                    } else {
                        params.upper_bound((max_tf as f32).max(1.0), *idf)
                    }
                }))
            }
            CursorVariant::Sparse { .. } => None,
        }
    }

    fn has_group_impacts(&self) -> bool {
        matches!(&self.variant, CursorVariant::Text { list, .. } if list.has_group_impact_bounds())
    }

    /// A remaining posting's enclosing group. Passed documents only loosen
    /// the bound; no payload or scoring length is read by this probe.
    fn text_group_span_from(&self, from: DocId) -> Option<(DocId, DocId, f32)> {
        if self.exhausted {
            return None;
        }
        let CursorVariant::Text { list, .. } = &self.variant else {
            return None;
        };
        let start = from.max(self.doc());
        let idx = list.seek_block(start, self.block_idx)?;
        let first = start.max(list.block_first_doc(idx)?);
        let (last, bound) = if let Some(bound) = self.text_group_bound(idx) {
            (list.group_last_doc(idx)?, bound)
        } else {
            (list.block_last_doc(idx)?, self.text_block_bound(idx))
        };
        Some((first, last, bound))
    }

    /// Upper bound of this cursor's contribution to any id in `[from, to]`:
    /// the largest block bound over the blocks intersecting the range, with
    /// one L1 word standing in for a group that lies inside it. Reads skip
    /// entries only; no block is decoded (Lucene `advanceShallow` +
    /// `getMaxScore(upTo)`).
    fn window_upper_bound(&self, from: DocId, to: DocId) -> f32 {
        if self.exhausted {
            return 0.0;
        }
        let CursorVariant::Text { list, .. } = &self.variant else {
            return self.max_score;
        };
        // Postings the cursor has already passed cannot score again: the
        // bound starts at its current id, not at the window start.
        let start = from.max(self.doc());
        if start > to {
            return 0.0;
        }
        // The loaded block often covers the entire window. Preserve the
        // existing whole-group substitution when a complete group fits.
        if self.block_last_doc(self.block_idx) >= to
            && !(list.is_group_start(self.block_idx)
                && list
                    .group_last_doc(self.block_idx)
                    .is_some_and(|last| last <= to))
        {
            return 0.0f32.max(self.text_block_bound(self.block_idx));
        }
        let Some(mut idx) = list.seek_block(start, self.block_idx) else {
            return 0.0;
        };
        let mut bound = 0.0f32;
        while idx < self.num_blocks {
            if list.block_first_doc(idx).unwrap_or(u32::MAX) > to {
                break;
            }
            if list.is_group_start(idx)
                && list.group_last_doc(idx).is_some_and(|last| last <= to)
                && let Some(group_bound) = self.text_group_bound(idx)
            {
                bound = bound.max(group_bound);
                idx = list.next_group_block(idx);
                continue;
            }
            bound = bound.max(self.text_block_bound(idx));
            idx += 1;
        }
        bound
    }

    /// Add this cursor's scores for every id in `[from, to]` to the window
    /// buffers (`scores[id - from]`, bit `id - from` of `mask`) and leave the
    /// cursor on its first id after `to`. Whole runs of a block are
    /// processed in one pass over its decoded arrays. Text cursors only.
    /// The per-term `contributions` (values, presence bits) feed the
    /// canonical query-order reduction; a lone essential cursor uses
    /// [`Self::append_scored_window_sync`] instead.
    fn score_window_sync(
        &mut self,
        from: DocId,
        to: DocId,
        scores: &mut [f32],
        mask: &mut [u64],
        mut contributions: Option<(&mut [f32], &mut [u64])>,
    ) -> crate::Result<u32> {
        self.visit_scored_window_sync(to, |docs, block_scores| {
            for (doc, score) in docs.iter().zip(block_scores) {
                let slot = (doc - from) as usize;
                scores[slot] += score;
                if let Some((values, present)) = contributions.as_mut() {
                    values[slot] = *score;
                    present[slot >> 6] |= 1u64 << (slot & 63);
                }
                mask[slot >> 6] |= 1u64 << (slot & 63);
            }
        })
    }

    /// A sole essential cursor already yields sorted, unique candidates.
    /// Retain per-term contributions for the later canonical reduction without
    /// expanding the candidate sequence into a dense document-ID window.
    ///
    /// `0.0 + score`: a single-term score is emitted as-is instead of through
    /// the query-order fold, which starts at `0.0`. Adding `0.0` here
    /// normalises a `-0.0` score to `+0.0` exactly like that fold does, so
    /// every path stays bit-identical (all other bits are unchanged). The
    /// same idiom appears in [`MaxScoreExecutor::execute_single_text`].
    fn append_scored_window_sync(
        &mut self,
        from: DocId,
        to: DocId,
        docs: &mut Vec<DocId>,
        scores: &mut Vec<f32>,
        mut contributions: Option<(&mut [f32], &mut [u64])>,
    ) -> crate::Result<u32> {
        self.visit_scored_window_sync(to, |run_docs, run_scores| {
            docs.extend_from_slice(run_docs);
            scores.extend(run_scores.iter().map(|score| 0.0 + score));
            if let Some((values, present)) = contributions.as_mut() {
                for (&doc, &score) in run_docs.iter().zip(run_scores) {
                    let slot = (doc - from) as usize;
                    values[slot] = score;
                    present[slot >> 6] |= 1u64 << (slot & 63);
                }
            }
        })
    }

    /// Visit the same bounded decoded runs for dense and sparse materialization.
    /// The caller has already positioned this cursor at the window's start.
    fn visit_scored_window_sync(
        &mut self,
        to: DocId,
        mut visit: impl FnMut(&[DocId], &[f32]),
    ) -> crate::Result<u32> {
        let mut matched = 0u32;
        loop {
            if self.exhausted {
                return Ok(matched);
            }
            if !self.block_loaded {
                if self.block_first_doc(self.block_idx) > to {
                    return Ok(matched);
                }
                self.ensure_block_loaded_sync()?;
                if self.exhausted {
                    return Ok(matched);
                }
            }
            if self.doc_ids[self.pos] > to {
                return Ok(matched);
            }
            self.ensure_scores();
            let remaining = &self.doc_ids[self.pos..];
            let end = if to == u32::MAX {
                remaining.len()
            } else {
                crate::structures::simd::find_first_ge_u32(remaining, to + 1)
            };
            let block_scores = &self.scores[self.pos..self.pos + end];
            visit(&remaining[..end], block_scores);
            matched += end as u32;
            self.pos += end;
            if self.pos >= self.doc_ids.len() {
                self.block_idx += 1;
                self.block_loaded = false;
                if self.block_idx >= self.num_blocks {
                    self.exhausted = true;
                    return Ok(matched);
                }
            } else {
                return Ok(matched);
            }
        }
    }

    /// Move past every id `<= to`, skipping whole blocks that end before it
    /// without decoding them.
    fn skip_past_sync(&mut self, to: DocId) -> crate::Result<()> {
        if to == u32::MAX {
            self.exhausted = true;
            return Ok(());
        }
        while !self.exhausted && self.block_last_doc(self.block_idx) <= to {
            if self.current_group_last_doc() <= to {
                self.skip_to_next_group();
            } else {
                self.skip_to_next_block();
            }
        }
        if !self.exhausted && self.doc() <= to {
            self.seek_sync(to + 1)?;
        }
        Ok(())
    }

    /// Probe sorted candidate IDs, optionally intersecting their membership.
    /// Contributions stay in the caller's canonical per-term window buffers.
    /// False reports deadline truncation before the window is collected.
    fn score_candidates_sync(
        &mut self,
        from: DocId,
        docs: &mut Vec<DocId>,
        scores: &mut Vec<f32>,
        required: bool,
        contributions: Option<(&mut [f32], &mut [u64])>,
        budget: Option<&SharedThreshold>,
    ) -> crate::Result<bool> {
        if required {
            self.score_candidate_membership::<true>(from, docs, scores, contributions, budget)
        } else {
            self.score_candidate_membership::<false>(from, docs, scores, contributions, budget)
        }
    }

    fn score_candidate_membership<const REQUIRED: bool>(
        &mut self,
        from: DocId,
        docs: &mut Vec<DocId>,
        scores: &mut Vec<f32>,
        mut contributions: Option<(&mut [f32], &mut [u64])>,
        budget: Option<&SharedThreshold>,
    ) -> crate::Result<bool> {
        const RUN: usize = crate::structures::postings::POSTING_BLOCK_SIZE;
        let mut matched_docs = [0; RUN];
        // A loaded text block has at most 128 entries.
        const { assert!(RUN <= u8::MAX as usize + 1) };
        let mut posting_slots = [0u8; RUN];
        let mut output_slots = [0; RUN];
        let mut values = [0.0; RUN];
        let mut input = 0;
        let mut kept = 0;
        while input < docs.len() {
            if budget.is_some_and(SharedThreshold::stop_if_expired) {
                return Ok(false);
            }
            // One metadata-aware seek per block. All further probes in this
            // run stay inside the loaded document slice and cannot load I/O.
            if self.seek_sync(docs[input])? == u32::MAX {
                break;
            }
            let block_last = *self.doc_ids.last().expect("loaded posting block");
            let mut matched = 0;
            while input < docs.len() && docs[input] <= block_last && matched < RUN {
                if input.is_multiple_of(64) && budget.is_some_and(SharedThreshold::stop_if_expired)
                {
                    return Ok(false);
                }
                let doc = docs[input];
                if self.doc_ids[self.pos] < doc {
                    self.pos +=
                        crate::structures::simd::find_first_ge_u32(&self.doc_ids[self.pos..], doc);
                }
                let present = self.doc_ids[self.pos] == doc;
                if present || !REQUIRED {
                    if REQUIRED {
                        docs[kept] = doc;
                        scores[kept] = scores[input];
                    }
                    if present {
                        matched_docs[matched] = doc;
                        posting_slots[matched] = self.pos as u8;
                        output_slots[matched] = kept;
                        matched += 1;
                    }
                    kept += 1;
                }
                input += 1;
            }
            if matched > 0 {
                self.score_candidate_block(
                    &matched_docs[..matched],
                    &posting_slots[..matched],
                    &mut values[..matched],
                );
                for i in 0..matched {
                    scores[output_slots[i]] += values[i];
                    if let Some((stored, present)) = contributions.as_mut() {
                        let slot = (matched_docs[i] - from) as usize;
                        stored[slot] = values[i];
                        present[slot >> 6] |= 1u64 << (slot & 63);
                    }
                }
            }
        }
        if REQUIRED {
            docs.truncate(kept);
            scores.truncate(kept);
        }
        Ok(true)
    }

    /// Score just the matching postings from one loaded block. TF unpacking
    /// remains shared with full-window scoring; score readiness is independent.
    fn score_candidate_block(&mut self, docs: &[DocId], slots: &[u8], values: &mut [f32]) {
        if !self.scores.is_empty() {
            for (&slot, value) in slots.iter().zip(values) {
                *value = self.scores[usize::from(slot)];
            }
            return;
        }
        self.decode_deferred_tfs();
        let CursorVariant::Text {
            idf,
            avg_len,
            params,
            lengths,
            normalization,
            ..
        } = &self.variant
        else {
            unreachable!("loaded sparse blocks already contain scores");
        };
        let mut frequencies = [0; crate::structures::postings::POSTING_BLOCK_SIZE];
        for (tf, &slot) in frequencies.iter_mut().zip(slots) {
            *tf = self.tfs[usize::from(slot)];
        }
        score_text_run(
            *params,
            *idf,
            *avg_len,
            *lengths,
            normalization.as_deref(),
            docs,
            &frequencies[..docs.len()],
            values,
        );
    }

    /// Last doc of the L1 group containing the current block (text only).
    #[inline]
    fn current_group_last_doc(&self) -> DocId {
        match &self.variant {
            CursorVariant::Text { list, .. } => list.group_last_doc(self.block_idx).unwrap_or(0),
            CursorVariant::Sparse { .. } => self.block_last_doc(self.block_idx),
        }
    }

    /// Jump past the current L1 group (text) or block (sparse).
    fn skip_to_next_group(&mut self) -> DocId {
        if self.exhausted {
            return u32::MAX;
        }
        let next = match &self.variant {
            CursorVariant::Text { list, .. } => list.next_group_block(self.block_idx),
            CursorVariant::Sparse { .. } => self.block_idx + 1,
        };
        self.block_idx = next;
        self.block_loaded = false;
        if self.block_idx >= self.num_blocks {
            self.exhausted = true;
            return u32::MAX;
        }
        self.block_first_doc(self.block_idx)
    }

    // ── Block navigation ────────────────────────────────────────────────

    fn skip_to_next_block(&mut self) -> DocId {
        if self.exhausted {
            return u32::MAX;
        }
        self.block_idx += 1;
        self.block_loaded = false;
        if self.block_idx >= self.num_blocks {
            self.exhausted = true;
            return u32::MAX;
        }
        self.block_first_doc(self.block_idx)
    }

    #[inline]
    fn advance_pos(&mut self) -> DocId {
        self.pos += 1;
        if self.pos >= self.doc_ids.len() {
            self.block_idx += 1;
            self.block_loaded = false;
            if self.block_idx >= self.num_blocks {
                self.exhausted = true;
                return u32::MAX;
            }
        }
        self.doc()
    }

    /// Compute BM25 scores from deferred TF data (lazy decode for text cursors).
    #[inline(never)]
    fn decode_deferred_tfs(&mut self) {
        if let CursorVariant::Text {
            list, deferred_tf, ..
        } = &mut self.variant
            && let Some((block_offset, tf_start, count)) = deferred_tf.take()
        {
            list.decode_block_tfs_deferred(block_offset, tf_start, count, &mut self.tfs);
        }
    }

    fn compute_deferred_scores(&mut self) {
        self.decode_deferred_tfs();
        if let CursorVariant::Text {
            idf,
            avg_len,
            params,
            lengths,
            normalization,
            ..
        } = &self.variant
        {
            self.scores.resize(self.doc_ids.len(), 0.0);
            score_text_run(
                *params,
                *idf,
                *avg_len,
                *lengths,
                normalization.as_deref(),
                &self.doc_ids,
                &self.tfs,
                &mut self.scores,
            );
        }
    }

    // ── Block loading / advance / seek ─────────────────────────────────
    //
    // Macros parameterised on sparse I/O method + optional .await to
    // stamp out both async and sync variants without duplication.

    pub async fn ensure_block_loaded(&mut self) -> crate::Result<bool> {
        cursor_ensure_block!(self, load_block_direct, .await)
    }

    pub fn ensure_block_loaded_sync(&mut self) -> crate::Result<bool> {
        cursor_ensure_block!(self, load_block_direct_sync,)
    }

    pub async fn advance(&mut self) -> crate::Result<DocId> {
        cursor_advance!(self, ensure_block_loaded, .await)
    }

    pub fn advance_sync(&mut self) -> crate::Result<DocId> {
        cursor_advance!(self, ensure_block_loaded_sync,)
    }

    pub async fn seek(&mut self, target: DocId) -> crate::Result<DocId> {
        cursor_seek!(self, ensure_block_loaded, target, .await)
    }

    pub fn seek_sync(&mut self, target: DocId) -> crate::Result<DocId> {
        cursor_seek!(self, ensure_block_loaded_sync, target,)
    }

    #[inline]
    fn seek_prepare(&mut self, target: DocId) -> Option<DocId> {
        crate::observe::search_work!(posting_seeks += 1);
        if self.exhausted {
            return Some(u32::MAX);
        }

        // Fast path: target is within the currently loaded block
        if self.block_loaded
            && let Some(&last) = self.doc_ids.last()
        {
            if last >= target && self.doc_ids[self.pos] < target {
                let remaining = &self.doc_ids[self.pos..];
                self.pos += crate::structures::simd::find_first_ge_u32(remaining, target);
                if self.pos >= self.doc_ids.len() {
                    self.block_idx += 1;
                    self.block_loaded = false;
                    if self.block_idx >= self.num_blocks {
                        self.exhausted = true;
                        return Some(u32::MAX);
                    }
                }
                return Some(self.doc());
            }
            if self.doc_ids[self.pos] >= target {
                return Some(self.doc());
            }
        }

        self.seek_directory(target)
    }

    /// Keep the loaded-block seek small; directory traversal is needed only
    /// after the current block cannot answer the target.
    #[inline(never)]
    fn seek_directory(&mut self, target: DocId) -> Option<DocId> {
        let lo = match &self.variant {
            // Text: SIMD-accelerated 2-level seek (L1 + L0)
            CursorVariant::Text { list, .. } => match list.seek_block(target, self.block_idx) {
                Some(idx) => idx,
                None => {
                    self.exhausted = true;
                    return Some(u32::MAX);
                }
            },
            // Sparse: binary search on skip entries (lazy mmap reads)
            CursorVariant::Sparse { .. } => {
                let mut lo = self.block_idx;
                let mut hi = self.num_blocks;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    if self.block_last_doc(mid) < target {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                lo
            }
        };
        if lo >= self.num_blocks {
            self.exhausted = true;
            return Some(u32::MAX);
        }
        if lo != self.block_idx || !self.block_loaded {
            self.block_idx = lo;
            self.block_loaded = false;
        }
        None
    }

    #[inline]
    fn seek_finish(&mut self, target: DocId) -> bool {
        if self.exhausted {
            return false;
        }
        self.pos = crate::structures::simd::find_first_ge_u32(&self.doc_ids, target);
        if self.pos >= self.doc_ids.len() {
            self.block_idx += 1;
            self.block_loaded = false;
            if self.block_idx >= self.num_blocks {
                self.exhausted = true;
                return false;
            }
            return true;
        }
        false
    }
}

/// Macro to stamp out the Block-Max MaxScore loop for both async and sync paths.
///
/// `$ensure`, `$advance`, `$seek` are cursor method idents (async or _sync variants).
/// `$($aw:tt)*` captures `.await` for async or nothing for sync.
macro_rules! bms_execute_loop {
    ($self:ident, $ensure:ident, $advance:ident, $seek:ident, $($aw:tt)*) => {{
        let n = $self.cursors.len();

        // Load first block for each cursor (ensures doc() returns real values)
        for cursor in &mut $self.cursors {
            cursor.$ensure() $($aw)* ?;
        }

        let mut docs_scored = 0u64;
        let mut docs_skipped = 0u64;
        let mut blocks_skipped = 0u64;
        let mut groups_skipped = 0u64;
        let mut conjunction_skipped = 0u64;
        let mut ordinal_scores: Vec<(u16, f32)> = Vec::with_capacity(n * 2);
        let started = crate::observe::WallTimer::start();

        // The same rounding margin as every other path (`pruning_threshold`):
        // bound sums and partial scores accumulate in traversal order, so a
        // fixed absolute epsilon is not enough once scores exceed a few units.
        let mut adjusted_threshold = $self.pruning_threshold();
        let mut iterations: u64 = 0;

        loop {
            // Anytime budget: a coarse deadline check (one clock read per
            // 4096 iterations); the results collected so far are returned
            // and the query is flagged truncated.
            iterations += 1;
            if iterations & 0xFFF == 0
                && let Some(budget) = &$self.budget
                && budget.expired()
            {
                budget.mark_truncated();
                log::debug!(
                    "MaxScoreExecutor: deadline reached after {} iterations, {} scored",
                    iterations,
                    docs_scored
                );
                break;
            }
            let partition = $self.find_partition();
            if partition >= n {
                break;
            }

            // Find minimum doc_id across essential cursors and collect
            // which cursors are at min_doc (avoids redundant re-checks in
            // conjunction, block-max, predicate, and scoring passes).
            let mut min_doc = u32::MAX;
            // Smallest essential doc after min_doc: the first doc where a
            // cursor not at min_doc can contribute, hence the farthest a
            // block skip may safely go.
            let mut next_other = u32::MAX;
            let mut at_min_mask = 0u64; // bitset of cursor indices at min_doc
            for i in partition..n {
                let doc = $self.cursors[i].doc();
                match doc.cmp(&min_doc) {
                    std::cmp::Ordering::Less => {
                        next_other = min_doc;
                        min_doc = doc;
                        at_min_mask = 1u64 << (i as u32);
                    }
                    std::cmp::Ordering::Equal => {
                        at_min_mask |= 1u64 << (i as u32);
                    }
                    std::cmp::Ordering::Greater => {
                        if doc < next_other {
                            next_other = doc;
                        }
                    }
                }
            }
            if min_doc == u32::MAX {
                break;
            }

            let non_essential_upper = if partition > 0 {
                $self.prefix_sums[partition - 1]
            } else {
                0.0
            };

            // --- Conjunction optimization ---
            if $self.collector.len() >= $self.collector.k {
                let mut present_upper: f32 = 0.0;
                let mut mask = at_min_mask;
                while mask != 0 {
                    let i = mask.trailing_zeros() as usize;
                    present_upper += $self.cursors[i].max_score;
                    mask &= mask - 1;
                }

                if present_upper + non_essential_upper < adjusted_threshold {
                    let mut mask = at_min_mask;
                    while mask != 0 {
                        let i = mask.trailing_zeros() as usize;
                        $self.cursors[i].$ensure() $($aw)* ?;
                        $self.cursors[i].$advance() $($aw)* ?;
                        mask &= mask - 1;
                    }
                    conjunction_skipped += 1;
                    continue;
                }
            }

            // --- Block-max pruning ---
            if $self.collector.len() >= $self.collector.k {
                let mut block_max_sum: f32 = 0.0;
                let mut mask = at_min_mask;
                while mask != 0 {
                    let i = mask.trailing_zeros() as usize;
                    block_max_sum += $self.cursors[i].current_block_max_score();
                    mask &= mask - 1;
                }

                if block_max_sum + non_essential_upper < adjusted_threshold {
                    // Block-Max MaxScore skip: every document before
                    // `next_other` is covered only by the cursors at min_doc
                    // (plus non-essential ones), whose block bounds cannot
                    // reach the threshold. A document at or after
                    // `next_other` may also receive another essential
                    // cursor's score, so no cursor jumps past it: skip the
                    // block when it ends before `next_other`, otherwise seek
                    // to `next_other` inside the block.
                    //
                    // Superblocks: when the cursors' L1 group bounds cannot
                    // reach the threshold either, the same argument covers
                    // the whole group of eight blocks, so a cursor may jump
                    // to its next group instead (bounded by `next_other` in
                    // the same way). A cursor without group bounds counts
                    // with its block bound and still skips one block.
                    let mut group_sum: f32 = 0.0;
                    let mut mask = at_min_mask;
                    while mask != 0 {
                        let i = mask.trailing_zeros() as usize;
                        group_sum += $self.cursors[i]
                            .current_group_max_score()
                            .unwrap_or_else(|| $self.cursors[i].current_block_max_score());
                        mask &= mask - 1;
                    }
                    let group_prunable = group_sum + non_essential_upper < adjusted_threshold;
                    let mut mask = at_min_mask;
                    while mask != 0 {
                        let i = mask.trailing_zeros() as usize;
                        let by_group =
                            group_prunable && $self.cursors[i].current_group_max_score().is_some();
                        let boundary = if by_group {
                            $self.cursors[i].current_group_last_doc()
                        } else {
                            $self.cursors[i].block_last_doc($self.cursors[i].block_idx)
                        };
                        if next_other > boundary {
                            if by_group {
                                $self.cursors[i].skip_to_next_group();
                                groups_skipped += 1;
                            } else {
                                $self.cursors[i].skip_to_next_block();
                            }
                            $self.cursors[i].$ensure() $($aw)* ?;
                        } else {
                            $self.cursors[i].$seek(next_other) $($aw)* ?;
                        }
                        mask &= mask - 1;
                    }
                    blocks_skipped += 1;
                    continue;
                }
            }

            // --- Predicate filter (after block-max, before scoring) ---
            if let Some(ref pred) = $self.predicate {
                if !pred(min_doc) {
                    let mut mask = at_min_mask;
                    while mask != 0 {
                        let i = mask.trailing_zeros() as usize;
                        $self.cursors[i].$ensure() $($aw)* ?;
                        $self.cursors[i].$advance() $($aw)* ?;
                        mask &= mask - 1;
                    }
                    continue;
                }
            }

            // --- Score essential cursors ---
            ordinal_scores.clear();
            {
                let mut mask = at_min_mask;
                while mask != 0 {
                    let i = mask.trailing_zeros() as usize;
                    $self.cursors[i].$ensure() $($aw)* ?;
                    $self.cursors[i].ensure_scores();
                    while $self.cursors[i].doc() == min_doc {
                        let ord = $self.cursors[i].ordinal_mut();
                        let sc = $self.cursors[i].score();
                        ordinal_scores.push((ord, sc));
                        $self.cursors[i].$advance() $($aw)* ?;
                    }
                    mask &= mask - 1;
                }
            }

            let essential_total: f32 = ordinal_scores.iter().map(|(_, s)| *s).sum();
            if $self.collector.len() >= $self.collector.k
                && essential_total + non_essential_upper < adjusted_threshold
            {
                docs_skipped += 1;
                continue;
            }

            // --- Score non-essential cursors (highest max_score first for early exit) ---
            let mut running_total = essential_total;
            for i in (0..partition).rev() {
                if $self.collector.len() >= $self.collector.k
                    && running_total + $self.prefix_sums[i] < adjusted_threshold
                {
                    break;
                }

                let doc = $self.cursors[i].$seek(min_doc) $($aw)* ?;
                if doc == min_doc {
                    $self.cursors[i].ensure_scores();
                    while $self.cursors[i].doc() == min_doc {
                        let s = $self.cursors[i].score();
                        running_total += s;
                        let ord = $self.cursors[i].ordinal_mut();
                        ordinal_scores.push((ord, s));
                        $self.cursors[i].$advance() $($aw)* ?;
                    }
                }
            }

            // --- Group by ordinal and insert ---
            // Fast path: single entry (common for single-valued fields) — skip sort + grouping
            if ordinal_scores.len() == 1 {
                let (ord, score) = ordinal_scores[0];
                if $self.collector.insert_with_ordinal(min_doc, score, ord) {
                    docs_scored += 1;
                    adjusted_threshold = $self.pruning_threshold();
                } else {
                    docs_skipped += 1;
                }
            } else if !ordinal_scores.is_empty() {
                if ordinal_scores.len() > 2 {
                    ordinal_scores.sort_unstable_by_key(|(ord, _)| *ord);
                } else if ordinal_scores.len() == 2 && ordinal_scores[0].0 > ordinal_scores[1].0 {
                    ordinal_scores.swap(0, 1);
                }
                let mut j = 0;
                while j < ordinal_scores.len() {
                    let current_ord = ordinal_scores[j].0;
                    let mut score = 0.0f32;
                    while j < ordinal_scores.len() && ordinal_scores[j].0 == current_ord {
                        score += ordinal_scores[j].1;
                        j += 1;
                    }
                    if $self
                        .collector
                        .insert_with_ordinal(min_doc, score, current_ord)
                    {
                        docs_scored += 1;
                        adjusted_threshold = $self.pruning_threshold();
                    } else {
                        docs_skipped += 1;
                    }
                }
            }
        }

        let results = $self.finish();

        let elapsed_ms = (started.secs() * 1000.0) as u64;
        if elapsed_ms > 500 {
            warn!(
                "slow MaxScore: {}ms, cursors={}, scored={}, skipped={}, blocks_skipped={}, groups_skipped={}, conjunction_skipped={}, returned={}, top_score={:.4}",
                elapsed_ms,
                n,
                docs_scored,
                docs_skipped,
                blocks_skipped,
                groups_skipped,
                conjunction_skipped,
                results.len(),
                results.first().map(|r| r.score).unwrap_or(0.0)
            );
        } else {
            debug!(
                "MaxScoreExecutor: {}ms, scored={}, skipped={}, blocks_skipped={}, groups_skipped={}, conjunction_skipped={}, returned={}, top_score={:.4}",
                elapsed_ms,
                docs_scored,
                docs_skipped,
                blocks_skipped,
                groups_skipped,
                conjunction_skipped,
                results.len(),
                results.first().map(|r| r.score).unwrap_or(0.0)
            );
        }

        Ok(results)
    }};
}

impl<'a> MaxScoreExecutor<'a> {
    /// Create a new executor from pre-built cursors.
    ///
    /// Cursors are sorted by max_score ascending (non-essential first) and
    /// prefix sums are computed for the MaxScore partitioning.
    pub(crate) fn new(mut cursors: Vec<TermCursor<'a>>, k: usize, heap_factor: f32) -> Self {
        // The execution loop tracks cursors at the current document in a u64.
        // Query construction normally enforces this bound, but keep this
        // boundary defensive for direct/internal executor users as well.
        // Dropping cursors changes the query (and loses the input order the
        // `require_*` builders rely on), so it is logged with counts and
        // remembered.
        let dropped_cursors = cursors.len().saturating_sub(super::MAX_QUERY_TERMS);
        if dropped_cursors > 0 {
            log::warn!(
                "MaxScoreExecutor: {} cursors exceed the {}-term limit; dropping the {} with the lowest upper bounds (input order is lost, required-term semantics will be refused)",
                cursors.len(),
                super::MAX_QUERY_TERMS,
                dropped_cursors
            );
            cursors.sort_unstable_by(|a, b| b.max_score.total_cmp(&a.max_score));
            cursors.truncate(super::MAX_QUERY_TERMS);
        }

        // Enable lazy ordinal decode — ordinals are only decoded when a doc
        // actually reaches the scoring phase (saves ~100ns per skipped block).
        for c in &mut cursors {
            c.lazy_ordinals = true;
        }

        // Sort by max_score ascending (non-essential first)
        let mut numbered: Vec<_> = cursors.into_iter().enumerate().collect();
        numbered.sort_by(|a, b| a.1.max_score.total_cmp(&b.1.max_score));
        let mut score_order: Vec<_> = (0..numbered.len()).collect();
        score_order.sort_unstable_by_key(|&i| numbered[i].0);
        let cursors: Vec<_> = numbered.into_iter().map(|(_, cursor)| cursor).collect();

        let mut prefix_sums = Vec::with_capacity(cursors.len());
        let mut cumsum = 0.0f32;
        for c in &cursors {
            cumsum += c.max_score;
            prefix_sums.push(cumsum);
        }

        let clamped_heap_factor = heap_factor.clamp(0.01, 1.0);

        log::trace!(
            "Creating MaxScoreExecutor: num_cursors={}, k={}, total_upper={:.4}, heap_factor={:.2}",
            cursors.len(),
            k,
            cumsum,
            clamped_heap_factor
        );

        Self {
            cursors,
            prefix_sums,
            score_order,
            all_required: false,
            required_mask: 0,
            collector: ScoreCollector::new(k),
            document_map: None,
            inv_heap_factor: 1.0 / clamped_heap_factor,
            predicate: None,
            budget: None,
            metric_index: "unknown",
            metric_field: "unknown",
            dropped_cursors,
            configuration_error: None,
            stats: ExecutorStats::default(),
        }
    }

    /// Record a `require_*` misuse. Logged at error level and turned into an
    /// `Error::Query` by `execute`/`execute_sync`: running anyway would
    /// silently rank with the wrong semantics.
    fn reject_configuration(&mut self, message: String) {
        log::error!("MaxScoreExecutor: {message}; the query fails instead of mis-ranking");
        self.configuration_error.get_or_insert(message);
    }

    /// Attach the query's wall-clock budget (anytime mode).
    pub fn with_budget(mut self, budget: Option<SharedThreshold>) -> Self {
        self.budget = budget.filter(|b| b.deadline().is_some());
        self
    }

    /// Use compact hit batches for a pure conjunction. The planner retains complete
    /// membership callers and unsupported compositions in the general scorer.
    ///
    /// Needs at least two text cursors and no cursor dropped at the term
    /// limit; otherwise the executor is marked invalid and `execute` fails.
    pub(crate) fn require_all_terms(mut self) -> Self {
        if self.dropped_cursors > 0 {
            self.reject_configuration(format!(
                "require_all_terms after {} cursors were dropped at the {}-term limit",
                self.dropped_cursors,
                super::MAX_QUERY_TERMS
            ));
        } else if !self.all_text() || self.cursors.len() < 2 {
            self.reject_configuration(format!(
                "require_all_terms needs at least two text cursors (got {} cursors, all_text={})",
                self.cursors.len(),
                self.all_text()
            ));
        } else {
            self.all_required = true;
        }
        self
    }

    /// The first input cursors are semantic MUST terms; later ones are optional.
    /// Record identities after the constructor's bound-based cursor sort.
    ///
    /// Needs `1..=len` text cursors in their original input order; a cursor
    /// dropped at the term limit destroys that order, so the executor is
    /// then marked invalid and `execute` fails.
    pub(crate) fn require_prefix_terms(mut self, count: usize) -> Self {
        if self.dropped_cursors > 0 {
            self.reject_configuration(format!(
                "require_prefix_terms({count}) after {} cursors were dropped at the {}-term limit",
                self.dropped_cursors,
                super::MAX_QUERY_TERMS
            ));
        } else if !self.all_text() || count == 0 || count > self.cursors.len() {
            self.reject_configuration(format!(
                "require_prefix_terms({count}) needs 1..={} text cursors (all_text={})",
                self.cursors.len(),
                self.all_text()
            ));
        } else {
            for &index in &self.score_order[..count] {
                self.required_mask |= 1u64 << index;
            }
        }
        self
    }

    /// Attach (index, field) labels for the metrics this executor emits.
    pub fn with_metric_labels(mut self, index: &'a str, field: &'a str) -> Self {
        self.metric_index = index;
        self.metric_field = field;
        self
    }

    /// Create an executor for sparse vector queries.
    ///
    /// Builds `TermCursor::Sparse` for each matched dimension.
    pub fn sparse(
        sparse_index: &'a crate::segment::SparseIndex,
        query_terms: Vec<(u32, f32)>,
        k: usize,
        heap_factor: f32,
    ) -> Self {
        let cursors: Vec<TermCursor<'a>> = query_terms
            .iter()
            .filter_map(|&(dim_id, qw)| {
                let (skip_start, skip_count, global_max, block_data_offset) =
                    sparse_index.get_skip_range_full(dim_id)?;
                Some(TermCursor::sparse(
                    sparse_index,
                    qw,
                    skip_start,
                    skip_count,
                    global_max,
                    block_data_offset,
                ))
            })
            .collect();
        Self::new(cursors, k, heap_factor)
    }

    /// Executor for full-text BM25 over `posting_lists` (`(list, idf)`
    /// pairs). Every posting is scored with the length `lengths` supplies
    /// (chunk lengths or document norms; `None` uses `tf` as the length).
    pub fn text_with_lengths(
        posting_lists: Vec<(crate::structures::BlockPostingList, f32)>,
        avg_len: f32,
        k: usize,
        lengths: Option<LengthSource<'a>>,
        params: super::Bm25Params,
        heap_factor: f32,
    ) -> Self {
        let cursors: Vec<TermCursor<'a>> = posting_lists
            .into_iter()
            .map(|(pl, idf)| TermCursor::text_with_params(pl, idf, avg_len, lengths, params))
            .collect();
        Self::new(cursors, k, heap_factor)
    }

    /// Executor for full-text BM25 over a plain field, scored with the
    /// persisted per-document lengths when available.
    pub fn text(
        posting_lists: Vec<(crate::structures::BlockPostingList, f32)>,
        avg_field_len: f32,
        k: usize,
        lengths: Option<&'a crate::segment::chunk_map::DocLengths>,
        params: super::Bm25Params,
        heap_factor: f32,
    ) -> Self {
        Self::text_with_lengths(
            posting_lists,
            avg_field_len,
            k,
            lengths.map(LengthSource::Docs),
            params,
            heap_factor,
        )
    }

    /// Executor for BM25 over a chunked text field: posting ids are virtual
    /// chunk ids, scored with each chunk's real length. Results carry the
    /// virtual id in `doc_id`; the caller resolves it through `lengths`.
    pub fn text_chunked(
        posting_lists: Vec<(crate::structures::BlockPostingList, f32)>,
        avg_chunk_len: f32,
        k: usize,
        lengths: &'a crate::segment::chunk_map::ChunkMap,
        params: super::Bm25Params,
        heap_factor: f32,
    ) -> Self {
        Self::text_with_lengths(
            posting_lists,
            avg_chunk_len,
            k,
            Some(LengthSource::Chunks(lengths)),
            params,
            heap_factor,
        )
    }

    #[inline]
    fn find_partition(&self) -> usize {
        // Alpha < 1.0 raises the effective threshold → more terms become
        // non-essential → more aggressive pruning (approximate retrieval).
        // Use multiplication by reciprocal (cheaper than division).
        let threshold = self.pruning_threshold();
        // Keep an equal-score candidate essential: it can still displace the
        // current worst hit through the deterministic doc/ordinal tie-break.
        self.prefix_sums.partition_point(|&sum| sum < threshold)
    }

    /// The threshold every pruning decision compares bounds against (all
    /// execution paths). Bound sums and partial scores use traversal order,
    /// whereas final text scores use input order. Cover both accumulation
    /// errors and subtraction of remaining bounds with a margin relative to
    /// the score and the term count; an absolute epsilon alone fails once
    /// scores exceed a few units.
    fn pruning_threshold(&self) -> f32 {
        let threshold = self.collector.threshold() * self.inv_heap_factor;
        threshold - threshold.abs() * (4.0 * self.cursors.len() as f32 * f32::EPSILON) - 1e-6
    }

    pub(crate) fn with_document_map(
        mut self,
        map: &'a crate::segment::chunk_map::ChunkMap,
    ) -> Self {
        self.document_map = Some(map);
        self
    }

    fn result_doc(&self, physical: DocId) -> DocId {
        self.document_map
            .map_or(physical, |map| map.doc_id(physical))
    }

    /// Attach a per-doc predicate filter to this executor.
    ///
    /// Docs failing the predicate are skipped after block-max pruning but
    /// before scoring. The predicate does not affect thresholds or block-max
    /// comparisons — the heap stores pure sparse/text scores.
    pub fn with_predicate(mut self, predicate: super::DocPredicate<'a>) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Seed the collector with an initial threshold for tighter early pruning.
    pub fn seed_threshold(&mut self, initial_threshold: f32) {
        self.collector.seed_threshold(initial_threshold);
    }

    /// Execute Block-Max MaxScore and return top-k results (async).
    ///
    /// Text cursors (in-memory posting lists) run the synchronous dispatch
    /// (`dispatch_sync`); sparse cursors, whose blocks may need asynchronous
    /// I/O, run the document-at-a-time loop.
    pub async fn execute(mut self) -> crate::Result<Vec<ScoredDoc>> {
        let Some(timer) = self.begin()? else {
            return Ok(Vec::new());
        };
        let results = if self.all_text() {
            self.dispatch_sync()
        } else {
            bms_execute_loop!(self, ensure_block_loaded, advance, seek, .await)
        };
        self.record(timer, &results);
        results
    }

    /// Synchronous execution — works when all cursors are text or mmap-backed sparse.
    pub fn execute_sync(mut self) -> crate::Result<Vec<ScoredDoc>> {
        let Some(timer) = self.begin()? else {
            return Ok(Vec::new());
        };
        let results = self.dispatch_sync();
        self.record(timer, &results);
        results
    }

    /// Shared entry checks: a rejected `require_*` configuration is an
    /// error, an empty query returns nothing; otherwise the metrics timer
    /// starts.
    fn begin(&mut self) -> crate::Result<Option<crate::observe::Timer>> {
        if let Some(error) = self.configuration_error.take() {
            return Err(crate::Error::Query(error));
        }
        if self.cursors.is_empty() {
            return Ok(None);
        }
        Ok(Some(crate::observe::Timer::start()))
    }

    /// Pick the synchronous path: semantic conjunction, required-prefix
    /// windows, the single ratio-bounded text cursor, text windows, or the
    /// document-at-a-time loop for sparse cursors.
    fn dispatch_sync(&mut self) -> crate::Result<Vec<ScoredDoc>> {
        if self.all_required {
            self.execute_conjunction()
        } else if self.required_mask != 0 {
            self.execute_text_windows::<true>()
        } else if self.single_text_with_ratio_bounds() {
            self.execute_single_text()
        } else if self.all_text() {
            self.execute_windowed()
        } else {
            bms_execute_loop!(self, ensure_block_loaded_sync, advance_sync, seek_sync,)
        }
    }

    fn record(&self, timer: crate::observe::Timer, results: &crate::Result<Vec<ScoredDoc>>) {
        crate::observe::search_work!(
            executor_windows += self.stats.windows,
            executor_windows_skipped += self.stats.windows_skipped,
            executor_groups_skipped += self.stats.groups_skipped,
            executor_candidates += self.stats.candidates,
            executor_heap_admissions += self.stats.docs_scored,
            single_blocks_scored += self.stats.blocks_scored,
            single_blocks_skipped += self.stats.blocks_skipped
        );
        if let Ok(r) = results {
            crate::observe::maxscore_query(
                self.metric_index,
                self.metric_field,
                timer.secs(),
                r.len(),
            );
        }
    }

    /// Drain the collector into results sorted by score (every path).
    fn finish(&mut self) -> Vec<ScoredDoc> {
        let collector = std::mem::replace(&mut self.collector, ScoreCollector::new(0));
        collector
            .into_sorted_results()
            .into_iter()
            .map(|(doc_id, score, ordinal)| ScoredDoc {
                doc_id,
                score,
                ordinal,
            })
            .collect()
    }

    /// The document-at-a-time loop on any cursors (the reference the
    /// windowed executor is checked against in tests).
    #[cfg(test)]
    pub(crate) fn execute_doc_at_a_time_sync(mut self) -> crate::Result<Vec<ScoredDoc>> {
        if self.cursors.is_empty() {
            return Ok(Vec::new());
        }
        bms_execute_loop!(self, ensure_block_loaded_sync, advance_sync, seek_sync,)
    }

    /// Route a lone text cursor to [`Self::execute_single_text`] only when
    /// its list stores ratio bounds. That path decides per block and group
    /// from skip metadata alone and needs the tight ratio bounds to skip
    /// anything; without them its bounds admit nearly every block, and the
    /// measured cost of the windowed path at n = 1 (one sorted run per
    /// block, no reduction) is no higher, so plain lists stay on the general
    /// path.
    fn single_text_with_ratio_bounds(&self) -> bool {
        matches!(self.cursors.as_slice(), [TermCursor { variant: CursorVariant::Text { list, .. }, .. }] if list.has_ratio_bounds())
    }

    /// A single text cursor needs no dense ID-window scratch or score reduction.
    /// Keep skip decisions shallow: decode only after both levels admit a block.
    fn execute_single_text(&mut self) -> crate::Result<Vec<ScoredDoc>> {
        if self.collector.k == 0 {
            return Ok(Vec::new());
        }
        let mut blocks_scored = 0u64;
        let mut blocks_skipped = 0u64;
        let mut groups_skipped = 0u64;
        let mut decisions = 0u64;
        let started = crate::observe::WallTimer::start();
        while !self.cursors[0].exhausted {
            // One clock read per 64 block decisions, like the other loops.
            if decisions & 0x3F == 0
                && self
                    .budget
                    .as_ref()
                    .is_some_and(SharedThreshold::stop_if_expired)
            {
                break;
            }
            decisions += 1;
            let threshold = self.pruning_threshold();
            let full = self.collector.len() >= self.collector.k;
            let cursor = &mut self.cursors[0];
            if full
                && cursor
                    .text_group_bound_for_threshold(cursor.block_idx, threshold)
                    .is_some_and(|bound| bound < threshold)
            {
                let before = cursor.block_idx;
                cursor.skip_to_next_group();
                blocks_skipped += (cursor.block_idx - before) as u64;
                groups_skipped += 1;
                continue;
            }
            if full
                && cursor.text_block_bound_for_threshold(cursor.block_idx, threshold) < threshold
            {
                cursor.skip_to_next_block();
                blocks_skipped += 1;
                continue;
            }
            if !cursor.ensure_block_loaded_sync()? {
                break;
            }
            cursor.ensure_scores();
            blocks_scored += 1;
            if self.predicate.is_none() {
                let docs = &cursor.doc_ids[cursor.pos..];
                let scores = &cursor.scores[cursor.pos..];
                if let Some(map) = self.document_map {
                    self.collector
                        .insert_text_run_with_mapping(docs, scores, |doc| map.doc_id(doc));
                } else {
                    self.collector.insert_text_run(docs, scores);
                }
            } else {
                for (&doc, &score) in cursor.doc_ids[cursor.pos..]
                    .iter()
                    .zip(&cursor.scores[cursor.pos..])
                {
                    if self
                        .predicate
                        .as_ref()
                        .is_none_or(|predicate| predicate(doc))
                    {
                        // `0.0 + score`: see `TermCursor::append_scored_window_sync`.
                        self.collector.insert_with_ordinal(
                            self.document_map.map_or(doc, |map| map.doc_id(doc)),
                            0.0 + score,
                            0,
                        );
                    }
                }
            }
            cursor.skip_to_next_block();
        }
        let results = self.finish();
        self.stats = ExecutorStats {
            blocks_scored,
            blocks_skipped,
            groups_skipped,
            ..ExecutorStats::default()
        };
        debug!(
            "MaxScoreExecutor(single): {}ms, blocks_scored={}, blocks_skipped={}, groups_skipped={}, returned={}",
            (started.secs() * 1000.0) as u64,
            self.stats.blocks_scored,
            self.stats.blocks_skipped,
            self.stats.groups_skipped,
            results.len()
        );
        Ok(results)
    }

    fn all_text(&self) -> bool {
        self.cursors.iter().all(TermCursor::is_text)
    }
}

/// Ids per window of the windowed executor (Lucene's `INNER_WINDOW_SIZE`).
const WINDOW_IDS: usize = 4096;

/// Per-thread scratch of the windowed and conjunction executors (the
/// `BmpScratch` pattern): buffers are cleared or resized per run, never
/// reallocated once grown, and bounded by `MAX_QUERY_TERMS` × `WINDOW_IDS`
/// (about 1 MiB of contributions per thread at the term limit).
#[derive(Default)]
struct WindowScratch {
    /// Dense per-window score accumulator (`WINDOW_IDS` slots).
    window_scores: Vec<f32>,
    /// Match bitset of the window (`WINDOW_IDS / 64` words).
    window_mask: Vec<u64>,
    /// Per-term contributions, `n × WINDOW_IDS`, read only through the
    /// presence bits in `contribution_masks` (`n × WINDOW_IDS / 64`), so
    /// stale values from earlier windows or queries are never summed.
    contributions: Vec<f32>,
    contribution_masks: Vec<u64>,
    /// Surviving candidates of the current window, in id order.
    cand_docs: Vec<u32>,
    cand_scores: Vec<f32>,
    /// Per-cursor window bounds, bound-sorted cursor order, prefix sums.
    wmax: Vec<f32>,
    order: Vec<usize>,
    wprefix: Vec<f32>,
    /// Conjunction: `n × POSTING_BLOCK_SIZE` term frequencies of one batch.
    conjunction_tfs: Vec<u32>,
}

impl WindowScratch {
    /// Size every window buffer for `n` cursors; grows only past the largest
    /// query seen so far on this thread.
    fn prepare_windows(&mut self, n: usize) {
        debug_assert!(n <= super::MAX_QUERY_TERMS);
        grow(&mut self.window_scores, WINDOW_IDS, 0.0);
        grow(&mut self.window_mask, WINDOW_IDS / 64, 0);
        if n > 1 {
            grow(&mut self.contributions, n * WINDOW_IDS, 0.0);
            grow(&mut self.contribution_masks, n * (WINDOW_IDS / 64), 0);
        }
        self.cand_docs.clear();
        self.cand_scores.clear();
        self.cand_docs.reserve(WINDOW_IDS);
        self.cand_scores.reserve(WINDOW_IDS);
        self.wmax.clear();
        self.wmax.resize(n, 0.0);
        self.wprefix.clear();
        self.wprefix.resize(n, 0.0);
        self.order.clear();
        self.order.extend(0..n);
    }

    fn prepare_conjunction(&mut self, n: usize) {
        debug_assert!(n <= super::MAX_QUERY_TERMS);
        grow(
            &mut self.conjunction_tfs,
            n * crate::structures::postings::POSTING_BLOCK_SIZE,
            0,
        );
    }
}

/// Extend `buffer` to at least `len` elements (no-op once large enough).
fn grow<T: Copy>(buffer: &mut Vec<T>, len: usize, fill: T) {
    if buffer.len() < len {
        buffer.resize(len, fill);
    }
}

thread_local! {
    static WINDOW_SCRATCH: std::cell::RefCell<WindowScratch> =
        std::cell::RefCell::new(WindowScratch::default());
}

/// Borrow this thread's scratch for one run. A nested run on the same thread
/// (an executor driven from inside another executor's predicate) cannot
/// share it and gets a private, freshly allocated scratch instead; that is
/// logged because it defeats the reuse the scratch exists for.
fn with_window_scratch<R>(run: impl FnOnce(&mut WindowScratch) -> R) -> R {
    WINDOW_SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => run(&mut scratch),
        Err(_) => {
            log::warn!(
                "MaxScoreExecutor: window scratch already in use on this thread (nested execution); allocating a private scratch"
            );
            run(&mut WindowScratch::default())
        }
    })
}

/// Keep the candidates that can still reach `threshold` once `remaining`
/// (the bounds of the cursors not yet applied) is added. Written without a
/// data-dependent branch, like Lucene's `VectorUtil.filterByScore`.
fn filter_competitive(docs: &mut Vec<u32>, scores: &mut Vec<f32>, remaining: f32, threshold: f32) {
    let mut kept = 0usize;
    for j in 0..docs.len() {
        let doc = docs[j];
        let score = scores[j];
        docs[kept] = doc;
        scores[kept] = score;
        kept += (score + remaining >= threshold) as usize;
    }
    docs.truncate(kept);
    scores.truncate(kept);
}

#[cfg(test)]
mod tests;
