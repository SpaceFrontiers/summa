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
            // The final real entry displaces the last virtual sentinel.
            if self.heap.len() == self.k {
                self.virtual_threshold = None;
                self.update_threshold();
            }
            true
        } else if self.heap.peek().is_some_and(|worst| entry < *worst) {
            {
                let mut worst = self.heap.peek_mut().expect("full heap has a root");
                *worst = entry;
            }
            self.update_threshold();
            true
        } else {
            false
        }
    }

    /// Check if a score could potentially enter top-k
    #[inline]
    pub fn would_enter(&self, score: f32) -> bool {
        self.len() < self.k || score > self.cached_threshold
    }

    /// Check whether this fully identified candidate ranks ahead of the current
    /// worst retained entry, including deterministic tie breaks.
    #[inline]
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
    /// Metric labels (index, field) — set via `with_metric_labels`; empty
    /// strings render as "unknown"/"?" is avoided by callers passing real
    /// names from the schema.
    metric_index: &'a str,
    metric_field: &'a str,
    cursors: Vec<TermCursor<'a>>,
    prefix_sums: Vec<f32>,
    collector: ScoreCollector,
    inv_heap_factor: f32,
    predicate: Option<super::DocPredicate<'a>>,
    /// Query-global budget: checked every few thousand loop iterations;
    /// an expired deadline ends traversal with the results so far.
    budget: Option<SharedThreshold>,
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
    doc_ids: Vec<u32>,
    scores: Vec<f32>,
    ordinals: Vec<u16>,
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
        /// Precomputed: idf * (BM25_K1 + 1.0) — numerator scale factor
        idf_times_k1_plus_1: f32,
        /// Precomputed: 1.0 + BM25_K1 * (BM25_B / avg_field_len) — denominator tf coefficient
        denom_tf_coeff: f32,
        /// Precomputed: BM25_K1 * (1.0 - BM25_B) — denominator constant
        denom_const: f32,
        /// Precomputed: BM25_K1 * BM25_B / avg_len — per-token length
        /// coefficient, used when `lengths` supplies real chunk lengths.
        denom_len_coeff: f32,
        /// Real per-posting lengths (chunk lengths or document norms).
        /// `None` keeps the historic `tf`-as-length approximation.
        lengths: Option<LengthSource<'a>>,
        /// Block bounds may use the block's minimum length: only when the
        /// list stores one and scoring uses real lengths (a `tf`-as-length
        /// score is not bounded by a real-length bound).
        length_bounds: bool,
        /// Average length used by the bounds (matches the scoring average).
        avg_len: f32,
        /// Per-field k1/b, used by the block and group bounds.
        params: super::Bm25Params,
        tfs: Vec<u32>,
        /// Deferred TF decode state: (block_offset, tf_start, count).
        /// Set when doc_ids are decoded but TFs/scores are not yet computed.
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
                    $self.exhausted = true;
                    Ok(false)
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
    /// Full-text BM25 cursor with explicit per-field parameters.
    pub fn text_with_params(
        posting_list: crate::structures::BlockPostingList,
        idf: f32,
        avg_field_len: f32,
        lengths: Option<LengthSource<'a>>,
        params: super::Bm25Params,
    ) -> Self {
        Self::text_with_lengths(posting_list, idf, avg_field_len, lengths, params)
    }

    fn text_with_lengths(
        posting_list: crate::structures::BlockPostingList,
        idf: f32,
        avg_field_len: f32,
        lengths: Option<LengthSource<'a>>,
        params: super::Bm25Params,
    ) -> Self {
        let max_tf = posting_list.max_tf() as f32;
        let safe_avg = avg_field_len.max(1.0);
        let length_bounds = lengths.is_some() && posting_list.min_len().is_some();
        let max_score = match posting_list.min_len() {
            Some(min_len) if length_bounds => {
                params.upper_bound_with_len(max_tf.max(1.0), idf, min_len as f32, safe_avg)
            }
            _ => params.upper_bound(max_tf.max(1.0), idf),
        };
        let num_blocks = posting_list.num_blocks();
        Self {
            max_score,
            num_blocks,
            block_idx: 0,
            doc_ids: Vec::with_capacity(128),
            scores: Vec::with_capacity(128),
            ordinals: Vec::new(),
            pos: 0,
            block_loaded: false,
            exhausted: num_blocks == 0,
            lazy_ordinals: false,
            ordinals_loaded: true, // text cursors never have ordinals
            current_sparse_block: None,
            variant: CursorVariant::Text {
                list: posting_list,
                idf,
                idf_times_k1_plus_1: idf * (params.k1 + 1.0),
                denom_tf_coeff: 1.0 + params.k1 * (params.b / safe_avg),
                denom_const: params.k1 * (1.0 - params.b),
                denom_len_coeff: params.k1 * params.b / safe_avg,
                lengths,
                length_bounds,
                avg_len: safe_avg,
                params,
                tfs: Vec::with_capacity(128),
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
        Self {
            max_score: query_weight.abs() * global_max_weight,
            num_blocks: skip_count,
            block_idx: 0,
            doc_ids: Vec::with_capacity(256),
            scores: Vec::with_capacity(256),
            ordinals: Vec::with_capacity(256),
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
    pub fn ordinal_mut(&mut self) -> u16 {
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
    pub fn ensure_scores(&mut self) {
        if self.block_loaded && self.scores.is_empty() {
            self.compute_deferred_scores();
        }
    }

    #[inline]
    pub fn current_block_max_score(&self) -> f32 {
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
    pub fn current_group_max_score(&self) -> Option<f32> {
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
    pub(crate) fn is_text(&self) -> bool {
        matches!(self.variant, CursorVariant::Text { .. })
    }

    /// Upper bound of text block `idx` from its `(max_tf, min_len)` word.
    fn text_block_bound(&self, idx: usize) -> f32 {
        match &self.variant {
            CursorVariant::Text {
                list,
                idf,
                length_bounds,
                avg_len,
                params,
                ..
            } => {
                let (max_tf, min_len) = list.block_bounds(idx).unwrap_or((0, None));
                match min_len {
                    Some(min_len) if *length_bounds => params.upper_bound_with_len(
                        (max_tf as f32).max(1.0),
                        *idf,
                        min_len as f32,
                        *avg_len,
                    ),
                    _ => params.upper_bound((max_tf as f32).max(1.0), *idf),
                }
            }
            CursorVariant::Sparse { .. } => self.max_score,
        }
    }

    /// Upper bound of the L1 group containing text block `idx`.
    fn text_group_bound(&self, idx: usize) -> Option<f32> {
        match &self.variant {
            CursorVariant::Text {
                list,
                idf,
                length_bounds,
                avg_len,
                params,
                ..
            } => {
                let (max_tf, min_len) = list.group_bounds(idx)?;
                Some(if *length_bounds {
                    params.upper_bound_with_len(
                        (max_tf as f32).max(1.0),
                        *idf,
                        min_len as f32,
                        *avg_len,
                    )
                } else {
                    params.upper_bound((max_tf as f32).max(1.0), *idf)
                })
            }
            CursorVariant::Sparse { .. } => None,
        }
    }

    /// Upper bound of this cursor's contribution to any id in `[from, to]`:
    /// the largest block bound over the blocks intersecting the range, with
    /// one L1 word standing in for a group that lies inside it. Reads skip
    /// entries only; no block is decoded (Lucene `advanceShallow` +
    /// `getMaxScore(upTo)`).
    pub(crate) fn window_upper_bound(&self, from: DocId, to: DocId) -> f32 {
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
    pub(crate) fn score_window_sync(
        &mut self,
        from: DocId,
        to: DocId,
        scores: &mut [f32],
        mask: &mut [u64],
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
            for (doc, score) in remaining[..end].iter().zip(block_scores) {
                let slot = (doc - from) as usize;
                scores[slot] += score;
                mask[slot >> 6] |= 1u64 << (slot & 63);
            }
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
    pub(crate) fn skip_past_sync(&mut self, to: DocId) -> crate::Result<()> {
        if to == u32::MAX {
            self.exhausted = true;
            return Ok(());
        }
        while !self.exhausted && self.block_last_doc(self.block_idx) <= to {
            self.skip_to_next_block();
        }
        if !self.exhausted && self.doc() <= to {
            self.seek_sync(to + 1)?;
        }
        Ok(())
    }

    /// Last doc of the L1 group containing the current block (text only).
    #[inline]
    pub fn current_group_last_doc(&self) -> DocId {
        match &self.variant {
            CursorVariant::Text { list, .. } => list.group_last_doc(self.block_idx).unwrap_or(0),
            CursorVariant::Sparse { .. } => self.block_last_doc(self.block_idx),
        }
    }

    /// Jump past the current L1 group (text) or block (sparse).
    pub fn skip_to_next_group(&mut self) -> DocId {
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

    pub fn skip_to_next_block(&mut self) -> DocId {
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
    fn compute_deferred_scores(&mut self) {
        if let CursorVariant::Text {
            list,
            idf_times_k1_plus_1,
            denom_tf_coeff,
            denom_const,
            denom_len_coeff,
            lengths,
            tfs,
            deferred_tf,
            ..
        } = &mut self.variant
            && let Some((block_offset, tf_start, count)) = deferred_tf.take()
        {
            list.decode_block_tfs_deferred(block_offset, tf_start, count, tfs);
            let num_scale = *idf_times_k1_plus_1;
            let d_tf = *denom_tf_coeff;
            let d_const = *denom_const;
            let d_len = *denom_len_coeff;
            self.scores.clear();
            self.scores.resize(count, 0.0);
            match lengths {
                // Real BM25 length normalisation per chunk or document.
                Some(source) => {
                    for i in 0..count {
                        let tf = unsafe { *tfs.get_unchecked(i) } as f32;
                        let vid = unsafe { *self.doc_ids.get_unchecked(i) };
                        let len = source.length(vid) as f32;
                        let score = (num_scale * tf) / (tf + d_const + d_len * len);
                        unsafe {
                            *self.scores.get_unchecked_mut(i) = score;
                        }
                    }
                }
                None => {
                    for i in 0..count {
                        let tf = unsafe { *tfs.get_unchecked(i) } as f32;
                        let score = (num_scale * tf) / (d_tf * tf + d_const);
                        unsafe {
                            *self.scores.get_unchecked_mut(i) = score;
                        }
                    }
                }
            }
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

    fn seek_prepare(&mut self, target: DocId) -> Option<DocId> {
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

        // Seek to the block containing target
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
        let _bms_start = std::time::Instant::now();

        let inv_heap_factor = $self.inv_heap_factor;
        let mut adjusted_threshold = $self.collector.threshold() * inv_heap_factor - 1e-6;
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
                    adjusted_threshold = $self.collector.threshold() * inv_heap_factor - 1e-6;
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
                        adjusted_threshold = $self.collector.threshold() * inv_heap_factor - 1e-6;
                    } else {
                        docs_skipped += 1;
                    }
                }
            }
        }

        let results: Vec<ScoredDoc> = $self
            .collector
            .into_sorted_results()
            .into_iter()
            .map(|(doc_id, score, ordinal)| ScoredDoc {
                doc_id,
                score,
                ordinal,
            })
            .collect();

        let _bms_elapsed_ms = _bms_start.elapsed().as_millis() as u64;
        if _bms_elapsed_ms > 500 {
            warn!(
                "slow MaxScore: {}ms, cursors={}, scored={}, skipped={}, blocks_skipped={}, groups_skipped={}, conjunction_skipped={}, returned={}, top_score={:.4}",
                _bms_elapsed_ms,
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
                _bms_elapsed_ms,
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
        if cursors.len() > super::MAX_QUERY_TERMS {
            cursors.sort_unstable_by(|a, b| b.max_score.total_cmp(&a.max_score));
            cursors.truncate(super::MAX_QUERY_TERMS);
            log::warn!(
                "MaxScore cursor count exceeded {}; retaining the strongest cursors",
                super::MAX_QUERY_TERMS
            );
        }

        // Enable lazy ordinal decode — ordinals are only decoded when a doc
        // actually reaches the scoring phase (saves ~100ns per skipped block).
        for c in &mut cursors {
            c.lazy_ordinals = true;
        }

        // Sort by max_score ascending (non-essential first)
        cursors.sort_by(|a, b| {
            a.max_score
                .partial_cmp(&b.max_score)
                .unwrap_or(Ordering::Equal)
        });

        let mut prefix_sums = Vec::with_capacity(cursors.len());
        let mut cumsum = 0.0f32;
        for c in &cursors {
            cumsum += c.max_score;
            prefix_sums.push(cumsum);
        }

        let clamped_heap_factor = heap_factor.clamp(0.01, 1.0);

        debug!(
            "Creating MaxScoreExecutor: num_cursors={}, k={}, total_upper={:.4}, heap_factor={:.2}",
            cursors.len(),
            k,
            cumsum,
            clamped_heap_factor
        );

        Self {
            cursors,
            prefix_sums,
            collector: ScoreCollector::new(k),
            inv_heap_factor: 1.0 / clamped_heap_factor,
            predicate: None,
            budget: None,
            metric_index: "unknown",
            metric_field: "unknown",
        }
    }

    /// Attach the query's wall-clock budget (anytime mode).
    pub fn with_budget(mut self, budget: Option<SharedThreshold>) -> Self {
        self.budget = budget.filter(|b| b.deadline().is_some());
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

    /// Create an executor for full-text BM25 queries.
    ///
    /// Builds `TermCursor::Text` for each posting list.
    pub fn text(
        posting_lists: Vec<(crate::structures::BlockPostingList, f32)>,
        avg_field_len: f32,
        k: usize,
        lengths: Option<&'a crate::segment::chunk_map::DocLengths>,
        params: super::Bm25Params,
        heap_factor: f32,
    ) -> Self {
        let cursors: Vec<TermCursor<'a>> = posting_lists
            .into_iter()
            .map(|(pl, idf)| {
                TermCursor::text_with_params(
                    pl,
                    idf,
                    avg_field_len,
                    lengths.map(LengthSource::Docs),
                    params,
                )
            })
            .collect();
        Self::new(cursors, k, heap_factor)
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
        let cursors: Vec<TermCursor<'a>> = posting_lists
            .into_iter()
            .map(|(pl, idf)| {
                TermCursor::text_with_params(
                    pl,
                    idf,
                    avg_chunk_len,
                    Some(LengthSource::Chunks(lengths)),
                    params,
                )
            })
            .collect();
        Self::new(cursors, k, heap_factor)
    }

    #[inline]
    fn find_partition(&self) -> usize {
        // Alpha < 1.0 raises the effective threshold → more terms become
        // non-essential → more aggressive pruning (approximate retrieval).
        // Use multiplication by reciprocal (cheaper than division).
        let threshold = self.collector.threshold() * self.inv_heap_factor;
        // Keep an equal-score candidate essential: it can still displace the
        // current worst hit through the deterministic doc/ordinal tie-break.
        self.prefix_sums.partition_point(|&sum| sum < threshold)
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
    /// Text cursors (in-memory posting lists) run the windowed executor;
    /// sparse cursors, whose blocks may need asynchronous I/O, run the
    /// document-at-a-time loop.
    pub async fn execute(mut self) -> crate::Result<Vec<ScoredDoc>> {
        if self.cursors.is_empty() {
            return Ok(Vec::new());
        }
        let t = crate::observe::Timer::start();
        let results = if self.all_text() {
            self.execute_windowed()
        } else {
            bms_execute_loop!(self, ensure_block_loaded, advance, seek, .await)
        };
        if let Ok(r) = &results {
            crate::observe::maxscore_query(self.metric_index, self.metric_field, t.secs(), r.len());
        }
        results
    }

    /// Synchronous execution — works when all cursors are text or mmap-backed sparse.
    pub fn execute_sync(mut self) -> crate::Result<Vec<ScoredDoc>> {
        if self.cursors.is_empty() {
            return Ok(Vec::new());
        }
        let t = crate::observe::Timer::start();
        let results = if self.all_text() {
            self.execute_windowed()
        } else {
            bms_execute_loop!(self, ensure_block_loaded_sync, advance_sync, seek_sync,)
        };
        if let Ok(r) = &results {
            crate::observe::maxscore_query(self.metric_index, self.metric_field, t.secs(), r.len());
        }
        results
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

    fn all_text(&self) -> bool {
        self.cursors.iter().all(TermCursor::is_text)
    }

    /// Window-at-a-time Block-Max MaxScore for text cursors.
    ///
    /// The id space is walked in windows of at most [`WINDOW_IDS`] ids that
    /// start at the first id a globally essential cursor can still reach and
    /// end at the smallest current block end among those cursors. Per window
    /// (Lucene `MaxScoreBulkScorer`; turbopuffer "batched iterator
    /// advancement"):
    ///
    /// 1. every cursor's bound over the window is read from its skip entries
    ///    (`window_upper_bound`), and the cursors are re-partitioned into
    ///    essential and non-essential by those bounds, so the partition is
    ///    the block-max one, not the list-max one;
    /// 2. a window whose summed bounds cannot reach the threshold is skipped
    ///    by every cursor without decoding a block;
    /// 3. each essential cursor scores all of its postings in the window in
    ///    one pass over its decoded block (dense `scores[id - from]` buffer
    ///    plus a match bitset), so the same iterator advances many times in
    ///    a row instead of alternating with the others;
    /// 4. the candidates are filtered branch-free against the threshold
    ///    minus what the remaining cursors could still add, and each
    ///    non-essential cursor is then sought to the survivors in id order
    ///    (again one iterator at a time), strongest bound first.
    ///
    /// Rank-safe: only documents whose window-essential score plus the
    /// non-essential bounds cannot reach the threshold are dropped. The
    /// approximate `heap_factor` mode scales the threshold as in the
    /// document-at-a-time loop.
    pub(crate) fn execute_windowed(&mut self) -> crate::Result<Vec<ScoredDoc>> {
        let n = self.cursors.len();
        for cursor in &mut self.cursors {
            cursor.ensure_block_loaded_sync()?;
        }
        let inv_heap_factor = self.inv_heap_factor;
        let mut window_scores = vec![0.0f32; WINDOW_IDS];
        let mut window_mask = vec![0u64; WINDOW_IDS / 64];
        let mut cand_docs: Vec<u32> = Vec::with_capacity(WINDOW_IDS);
        let mut cand_scores: Vec<f32> = Vec::with_capacity(WINDOW_IDS);
        let mut wmax = vec![0.0f32; n];
        let mut order: Vec<usize> = (0..n).collect();
        let mut wprefix = vec![0.0f32; n];
        let mut windows = 0u64;
        let mut windows_skipped = 0u64;
        let mut candidates = 0u64;
        let mut docs_scored = 0u64;
        let started = std::time::Instant::now();

        loop {
            windows += 1;
            if windows & 0x3F == 0
                && let Some(budget) = &self.budget
                && budget.expired()
            {
                budget.mark_truncated();
                log::debug!(
                    "MaxScoreExecutor(windowed): deadline reached after {} windows, {} scored",
                    windows,
                    docs_scored
                );
                break;
            }
            let partition = self.find_partition();
            if partition >= n {
                break;
            }
            // Window: from the first id a globally essential cursor can
            // still reach to the smallest current block end among them.
            let mut from = u32::MAX;
            let mut to = u32::MAX;
            for cursor in &self.cursors[partition..] {
                if cursor.exhausted {
                    continue;
                }
                from = from.min(cursor.doc());
                to = to.min(cursor.block_last_doc(cursor.block_idx));
            }
            if from == u32::MAX {
                break;
            }
            let to = to.max(from).min(from.saturating_add(WINDOW_IDS as u32 - 1));
            let width = (to - from) as usize + 1;
            let words = width.div_ceil(64);

            // Block-max partition over the window.
            let heap_full = self.collector.len() >= self.collector.k;
            let threshold = if heap_full {
                self.collector.threshold() * inv_heap_factor - 1e-6
            } else {
                0.0
            };
            for (i, bound) in wmax.iter_mut().enumerate() {
                *bound = self.cursors[i].window_upper_bound(from, to);
            }
            order.sort_unstable_by(|&a, &b| wmax[a].total_cmp(&wmax[b]));
            let mut sum = 0.0f32;
            for (rank, &i) in order.iter().enumerate() {
                sum += wmax[i];
                wprefix[rank] = sum;
            }
            let wpartition = if heap_full {
                wprefix.partition_point(|&s| s < threshold)
            } else {
                0
            };
            if wpartition >= n {
                // Nothing in the window can compete: every cursor jumps past it.
                for cursor in &mut self.cursors {
                    if !cursor.exhausted && cursor.doc() <= to {
                        cursor.skip_past_sync(to)?;
                    }
                }
                windows_skipped += 1;
                continue;
            }

            // Essential cursors: bulk-score into the window buffers.
            window_scores[..width].fill(0.0);
            window_mask[..words].fill(0);
            for &i in &order[wpartition..] {
                let cursor = &mut self.cursors[i];
                if cursor.exhausted {
                    continue;
                }
                if cursor.doc() < from {
                    cursor.seek_sync(from)?;
                }
                if cursor.exhausted || cursor.doc() > to {
                    continue;
                }
                cursor.score_window_sync(
                    from,
                    to,
                    &mut window_scores[..width],
                    &mut window_mask[..words],
                )?;
            }

            // Candidates in id order.
            cand_docs.clear();
            cand_scores.clear();
            for (word_idx, word) in window_mask[..words].iter().enumerate() {
                let mut bits = *word;
                while bits != 0 {
                    let slot = (word_idx << 6) | bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    cand_docs.push(from + slot as u32);
                    cand_scores.push(window_scores[slot]);
                }
            }
            if let Some(pred) = &self.predicate {
                let mut kept = 0usize;
                for j in 0..cand_docs.len() {
                    let doc = cand_docs[j];
                    cand_docs[kept] = doc;
                    cand_scores[kept] = cand_scores[j];
                    kept += pred(doc) as usize;
                }
                cand_docs.truncate(kept);
                cand_scores.truncate(kept);
            }

            // Non-essential cursors on the survivors, strongest bound first.
            let mut remaining = if wpartition > 0 {
                wprefix[wpartition - 1]
            } else {
                0.0
            };
            for rank in (0..wpartition).rev() {
                let i = order[rank];
                if heap_full {
                    filter_competitive(&mut cand_docs, &mut cand_scores, remaining, threshold);
                }
                if cand_docs.is_empty() {
                    break;
                }
                if wmax[i] > 0.0 {
                    let cursor = &mut self.cursors[i];
                    for (doc, score) in cand_docs.iter().zip(cand_scores.iter_mut()) {
                        if cursor.seek_sync(*doc)? == *doc {
                            cursor.ensure_scores();
                            *score += cursor.score();
                        }
                    }
                }
                remaining -= wmax[i];
            }
            if heap_full {
                filter_competitive(&mut cand_docs, &mut cand_scores, 0.0, threshold);
            }
            candidates += cand_docs.len() as u64;
            for (doc, score) in cand_docs.iter().zip(&cand_scores) {
                if self.collector.insert_with_ordinal(*doc, *score, 0) {
                    docs_scored += 1;
                }
            }
        }

        let collector = std::mem::replace(&mut self.collector, ScoreCollector::new(0));
        let results: Vec<ScoredDoc> = collector
            .into_sorted_results()
            .into_iter()
            .map(|(doc_id, score, ordinal)| ScoredDoc {
                doc_id,
                score,
                ordinal,
            })
            .collect();
        let elapsed_ms = started.elapsed().as_millis() as u64;
        if elapsed_ms > 500 {
            warn!(
                "slow windowed MaxScore: {}ms, cursors={}, windows={}, windows_skipped={}, candidates={}, scored={}, returned={}, top_score={:.4}",
                elapsed_ms,
                n,
                windows,
                windows_skipped,
                candidates,
                docs_scored,
                results.len(),
                results.first().map(|r| r.score).unwrap_or(0.0)
            );
        } else {
            debug!(
                "MaxScoreExecutor(windowed): {}ms, cursors={}, windows={}, windows_skipped={}, candidates={}, scored={}, returned={}, top_score={:.4}",
                elapsed_ms,
                n,
                windows,
                windows_skipped,
                candidates,
                docs_scored,
                results.len(),
                results.first().map(|r| r.score).unwrap_or(0.0)
            );
        }
        Ok(results)
    }
}

/// Ids per window of the windowed executor (Lucene's `INNER_WINDOW_SIZE`).
const WINDOW_IDS: usize = 4096;

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
mod tests {
    use super::*;

    // ── Windowed executor parity ─────────────────────────────────────────

    struct Corpus {
        /// Per term: sorted `(doc, tf)` postings.
        postings: Vec<Vec<(u32, u32)>>,
        lengths: Vec<u16>,
        n_docs: u32,
    }

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Terms with very different densities (from 0.5% to 60% of the
    /// documents), skewed term frequencies, and pseudo-random lengths.
    fn random_corpus(seed: u64, n_docs: u32, n_terms: usize) -> Corpus {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let lengths: Vec<u16> = (0..n_docs)
            .map(|_| 1 + (xorshift(&mut state) % 400) as u16)
            .collect();
        let densities = [0.6, 0.25, 0.1, 0.03, 0.005];
        let postings = (0..n_terms)
            .map(|t| {
                let density = densities[t % densities.len()];
                let cutoff = (density * u32::MAX as f64) as u64;
                let mut postings = Vec::new();
                for doc in 0..n_docs {
                    if (xorshift(&mut state) & 0xFFFF_FFFF) >= cutoff {
                        continue;
                    }
                    let r = xorshift(&mut state) % 100;
                    let tf = if r < 70 {
                        1
                    } else if r < 90 {
                        2
                    } else {
                        3 + (r % 6) as u32
                    };
                    postings.push((doc, tf));
                }
                postings
            })
            .collect();
        Corpus {
            postings,
            lengths,
            n_docs,
        }
    }

    fn build_lists(
        corpus: &Corpus,
        lengths: Option<&crate::segment::chunk_map::DocLengths>,
    ) -> Vec<(crate::structures::BlockPostingList, f32)> {
        corpus
            .postings
            .iter()
            .map(|postings| {
                let mut list = crate::structures::PostingList::new();
                for &(doc, tf) in postings {
                    list.push(doc, tf);
                }
                let length_of = lengths.map(|l| move |doc: DocId| l.length(doc));
                let block_list = crate::structures::BlockPostingList::from_posting_list_with(
                    &list,
                    false,
                    length_of.as_ref().map(|f| f as &dyn Fn(DocId) -> u32),
                )
                .unwrap();
                let idf = super::super::bm25_idf(postings.len() as f32, corpus.n_docs as f32);
                (block_list, idf)
            })
            .collect()
    }

    /// Exhaustive per-document scores with the same formula the cursors use.
    fn exhaustive(
        corpus: &Corpus,
        lists: &[(crate::structures::BlockPostingList, f32)],
        real_lengths: bool,
        avg: f32,
        params: super::super::Bm25Params,
    ) -> std::collections::HashMap<u32, f32> {
        let mut scores: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        for (postings, (_, idf)) in corpus.postings.iter().zip(lists) {
            for &(doc, tf) in postings {
                let len = if real_lengths {
                    corpus.lengths[doc as usize] as f32
                } else {
                    tf as f32
                };
                *scores.entry(doc).or_insert(0.0) += params.score(tf as f32, *idf, len, avg);
            }
        }
        scores
    }

    fn check_top_k(
        label: &str,
        results: &[ScoredDoc],
        exhaustive: &std::collections::HashMap<u32, f32>,
        k: usize,
        predicate: Option<&dyn Fn(u32) -> bool>,
    ) {
        let mut expected: Vec<(u32, f32)> = exhaustive
            .iter()
            .filter(|(doc, _)| predicate.is_none_or(|p| p(**doc)))
            .map(|(doc, score)| (*doc, *score))
            .collect();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let want = k.min(expected.len());
        assert_eq!(results.len(), want, "{label}: result count");
        for (rank, (got, exp)) in results.iter().zip(&expected).enumerate() {
            let tolerance = 1e-4 * exp.1.abs().max(1.0);
            assert!(
                (got.score - exp.1).abs() <= tolerance,
                "{label}: rank {rank} score {} vs exhaustive {} (doc {} vs {})",
                got.score,
                exp.1,
                got.doc_id,
                exp.0
            );
            let own = exhaustive[&got.doc_id];
            assert!(
                (got.score - own).abs() <= tolerance,
                "{label}: doc {} scored {} but exhaustive says {}",
                got.doc_id,
                got.score,
                own
            );
            if let Some(p) = predicate {
                assert!(
                    p(got.doc_id),
                    "{label}: doc {} fails the predicate",
                    got.doc_id
                );
            }
        }
        for pair in results.windows(2) {
            assert!(
                pair[0].score >= pair[1].score,
                "{label}: results not sorted"
            );
        }
    }

    /// The windowed executor returns the exact top-k (against an exhaustive
    /// scorer and against the document-at-a-time loop) over corpora of
    /// different sizes and term mixes, with and without real lengths,
    /// predicates, and a seeded threshold.
    #[test]
    fn windowed_text_maxscore_matches_exhaustive_and_doc_at_a_time() {
        let params = super::super::Bm25Params::default();
        let predicate_fn = |doc: u32| !doc.is_multiple_of(3);
        let mut cases = 0usize;
        for seed in 1..=6u64 {
            for &n_docs in &[300u32, 2_500, 12_000] {
                for &n_terms in &[1usize, 2, 4, 9] {
                    let corpus = random_corpus(seed, n_docs, n_terms);
                    let doc_lengths =
                        crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
                    for real_lengths in [true, false] {
                        let lengths = real_lengths.then_some(&doc_lengths);
                        let lists = build_lists(&corpus, lengths);
                        let avg = if real_lengths {
                            doc_lengths.avg_len()
                        } else {
                            1.0
                        };
                        let truth = exhaustive(&corpus, &lists, real_lengths, avg, params);
                        for &k in &[1usize, 10, 100] {
                            for with_predicate in [false, true] {
                                let label = format!(
                                    "seed={seed} docs={n_docs} terms={n_terms} lengths={real_lengths} k={k} pred={with_predicate}"
                                );
                                let pred: Option<&dyn Fn(u32) -> bool> =
                                    with_predicate.then_some(&predicate_fn);
                                let make = |seeded: f32| {
                                    let mut executor = MaxScoreExecutor::text(
                                        lists.clone(),
                                        avg,
                                        k,
                                        lengths,
                                        params,
                                        1.0,
                                    );
                                    if with_predicate {
                                        executor = executor.with_predicate(Box::new(predicate_fn));
                                    }
                                    if seeded > 0.0 {
                                        executor.seed_threshold(seeded);
                                    }
                                    executor
                                };
                                let windowed = make(0.0).execute_windowed().unwrap();
                                check_top_k(
                                    &format!("windowed {label}"),
                                    &windowed,
                                    &truth,
                                    k,
                                    pred,
                                );
                                let reference = make(0.0).execute_doc_at_a_time_sync().unwrap();
                                check_top_k(
                                    &format!("reference {label}"),
                                    &reference,
                                    &truth,
                                    k,
                                    pred,
                                );
                                // A floor below the k-th score keeps the exact top-k.
                                if let Some(kth) = windowed.last().map(|r| r.score)
                                    && windowed.len() == k
                                {
                                    let seeded = make(kth * 0.9).execute_windowed().unwrap();
                                    check_top_k(
                                        &format!("seeded {label}"),
                                        &seeded,
                                        &truth,
                                        k,
                                        pred,
                                    );
                                    // A floor above every score returns nothing.
                                    let above =
                                        make(windowed[0].score * 1.5).execute_windowed().unwrap();
                                    assert!(above.is_empty(), "{label}: floor above all scores");
                                }
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(cases > 400);
    }

    /// The approximate mode returns a subset of the exact top-k with exact
    /// scores.
    #[test]
    fn windowed_text_maxscore_heap_factor_is_a_subset_with_exact_scores() {
        let params = super::super::Bm25Params::default();
        let corpus = random_corpus(7, 20_000, 6);
        let doc_lengths = crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
        let lists = build_lists(&corpus, Some(&doc_lengths));
        let avg = doc_lengths.avg_len();
        let truth = exhaustive(&corpus, &lists, true, avg, params);
        let exact = MaxScoreExecutor::text(lists.clone(), avg, 50, Some(&doc_lengths), params, 1.0)
            .execute_windowed()
            .unwrap();
        check_top_k("exact", &exact, &truth, 50, None);
        let approx = MaxScoreExecutor::text(lists, avg, 50, Some(&doc_lengths), params, 0.6)
            .execute_windowed()
            .unwrap();
        assert_eq!(approx.len(), 50);
        for hit in &approx {
            let own = truth[&hit.doc_id];
            assert!((hit.score - own).abs() <= 1e-4 * own.max(1.0));
        }
        // The usual heap-factor guarantee: every returned score is within the
        // factor of the exact k-th score, and the best document is exact.
        let exact_kth = exact.last().unwrap().score;
        assert!(approx.iter().all(|hit| hit.score >= exact_kth * 0.6 - 1e-4));
        assert_eq!(approx[0].doc_id, exact[0].doc_id);
        let overlap = approx
            .iter()
            .filter(|hit| exact.iter().any(|e| e.doc_id == hit.doc_id))
            .count();
        assert!(overlap >= 25, "overlap {overlap} of 50");
    }

    #[test]
    fn test_shared_threshold_monotonic_raise() {
        let shared = SharedThreshold::new();
        assert_eq!(shared.get(), 0.0);

        shared.raise(2.5);
        assert_eq!(shared.get(), 2.5);

        // Lower values never lower the floor.
        shared.raise(1.0);
        assert_eq!(shared.get(), 2.5);

        // Higher values raise it.
        shared.raise(4.0);
        assert_eq!(shared.get(), 4.0);

        // Non-positive and NaN are ignored.
        shared.raise(0.0);
        shared.raise(-3.0);
        shared.raise(f32::NAN);
        assert_eq!(shared.get(), 4.0);

        // Clones share the same atomic cell.
        let clone = shared.clone();
        clone.raise(9.0);
        assert_eq!(shared.get(), 9.0);
    }

    #[test]
    fn test_shared_threshold_seed_matches_manual() {
        // A collector seeded with a floor prunes anything at/below it, matching
        // the threshold a fully-populated heap would have produced.
        let mut seeded = ScoreCollector::new(2);
        seeded.seed_threshold(3.0);
        assert_eq!(seeded.threshold(), 3.0);
        // A score at/below the floor cannot enter.
        assert!(!seeded.would_enter(3.0));
        assert!(seeded.would_enter(3.5));
        // Real inserts above the floor evict the sentinels; results contain no
        // sentinel (doc_id == u32::MAX) entries.
        seeded.insert(1, 5.0);
        seeded.insert(2, 4.0);
        let results = seeded.into_sorted_results();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);
        assert_eq!(results[1].0, 2);
    }

    #[test]
    fn test_shared_threshold_can_raise_after_real_inserts() {
        let mut collector = ScoreCollector::new(3);
        collector.insert(1, 10.0);
        collector.insert(2, 4.0);
        assert_eq!(collector.real_len(), 2);

        // Raising the floor after traversal has started removes retained work
        // that can no longer reach the global top-k.
        collector.seed_threshold(6.0);
        assert_eq!(collector.threshold(), 6.0);
        assert_eq!(collector.real_len(), 1);

        // A real candidate tied with the floor displaces the sentinel because
        // its doc id wins the canonical tie break.
        assert!(collector.would_enter_candidate(3, 6.0, 0));
        assert!(collector.insert(3, 6.0));
        assert_eq!(collector.real_len(), 2);
        let results = collector.into_sorted_results();
        assert_eq!(results, vec![(1, 10.0, 0), (3, 6.0, 0)]);
    }

    #[test]
    fn test_large_seed_uses_virtual_sentinels() {
        let k = 1_000_000_000;
        let mut collector = ScoreCollector::new(k);
        assert!(collector.heap.capacity() <= MAX_INITIAL_SCORE_COLLECTOR_CAPACITY);

        collector.seed_threshold(42.0);

        // Seeding a huge top-k is constant-time and does not materialize any
        // of its conceptual sentinel entries.
        assert_eq!(collector.heap.len(), 0);
        assert!(collector.heap.capacity() <= MAX_INITIAL_SCORE_COLLECTOR_CAPACITY);
        assert_eq!(collector.len(), k);
        assert_eq!(collector.real_len(), 0);
        assert_eq!(collector.threshold(), 42.0);
        assert!(!collector.is_empty());

        // A real result tied with the floor beats the sentinel by doc-id, while
        // a lower score remains below the conceptual threshold.
        assert!(collector.insert_with_ordinal(9, 42.0, 7));
        assert!(!collector.insert(10, 41.0));
        assert!(collector.insert(11, 43.0));
        assert_eq!(collector.len(), k);
        assert_eq!(collector.real_len(), 2);
        assert_eq!(
            collector.into_sorted_results(),
            vec![(11, 43.0, 0), (9, 42.0, 7)]
        );
    }

    #[test]
    fn test_virtual_sentinels_preserve_tie_order_when_filled() {
        let mut collector = ScoreCollector::new(3);
        collector.seed_threshold(5.0);

        assert!(collector.insert_with_ordinal(3, 5.0, 2));
        assert!(collector.insert_with_ordinal(2, 5.0, 8));
        assert!(collector.insert_with_ordinal(1, 5.0, 4));
        assert_eq!(collector.real_len(), 3);
        assert!(collector.virtual_threshold.is_none());

        // Once all virtual slots have been displaced, canonical doc/ordinal
        // ordering still controls root replacement at an equal score.
        assert!(collector.insert_with_ordinal(2, 5.0, 1));
        assert!(!collector.insert_with_ordinal(4, 5.0, 0));
        assert_eq!(
            collector.into_sorted_results(),
            vec![(1, 5.0, 4), (2, 5.0, 1), (2, 5.0, 8)]
        );
    }

    #[test]
    fn test_score_collector_basic() {
        let mut collector = ScoreCollector::new(3);

        collector.insert(1, 1.0);
        collector.insert(2, 2.0);
        collector.insert(3, 3.0);
        assert_eq!(collector.threshold(), 1.0);

        collector.insert(4, 4.0);
        assert_eq!(collector.threshold(), 2.0);

        let results = collector.into_sorted_results();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 4); // Highest score
        assert_eq!(results[1].0, 3);
        assert_eq!(results[2].0, 2);
    }

    #[test]
    fn test_score_collector_threshold() {
        let mut collector = ScoreCollector::new(2);

        collector.insert(1, 5.0);
        collector.insert(2, 3.0);
        assert_eq!(collector.threshold(), 3.0);

        // Should not enter (score too low)
        assert!(!collector.would_enter(2.0));
        assert!(!collector.insert(3, 2.0));

        // Should enter (score high enough)
        assert!(collector.would_enter(4.0));
        assert!(collector.insert(4, 4.0));
        assert_eq!(collector.threshold(), 4.0);
    }

    #[test]
    fn test_heap_entry_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(HeapEntry {
            doc_id: 1,
            score: 3.0,
            ordinal: 0,
        });
        heap.push(HeapEntry {
            doc_id: 2,
            score: 1.0,
            ordinal: 0,
        });
        heap.push(HeapEntry {
            doc_id: 3,
            score: 2.0,
            ordinal: 0,
        });

        // Min-heap: lowest score should come out first
        assert_eq!(heap.pop().unwrap().score, 1.0);
        assert_eq!(heap.pop().unwrap().score, 2.0);
        assert_eq!(heap.pop().unwrap().score, 3.0);
    }
}
