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

/// Entry for top-k min-heap
#[derive(Clone, Copy)]
pub struct HeapEntry {
    pub doc_id: DocId,
    pub score: f32,
    pub ordinal: u16,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.doc_id == other.doc_id
    }
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower scores come first (to be evicted).
        // total_cmp is branchless (compiles to a single comparison instruction).
        other
            .score
            .total_cmp(&self.score)
            .then(self.doc_id.cmp(&other.doc_id))
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
}

impl ScoreCollector {
    /// Create a new collector for top-k results
    pub fn new(k: usize) -> Self {
        // Cap capacity to avoid allocation overflow for very large k
        let capacity = k.saturating_add(1).min(1_000_000);
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            k,
            cached_threshold: 0.0,
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
        self.cached_threshold = if self.heap.len() >= self.k {
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
        if self.heap.len() < self.k {
            self.heap.push(HeapEntry {
                doc_id,
                score,
                ordinal,
            });
            // Only recompute threshold when heap just became full
            if self.heap.len() == self.k {
                self.update_threshold();
            }
            true
        } else if score > self.cached_threshold {
            self.heap.push(HeapEntry {
                doc_id,
                score,
                ordinal,
            });
            self.heap.pop(); // Remove lowest
            self.update_threshold();
            true
        } else {
            false
        }
    }

    /// Check if a score could potentially enter top-k
    #[inline]
    pub fn would_enter(&self, score: f32) -> bool {
        self.heap.len() < self.k || score > self.cached_threshold
    }

    /// Get number of documents collected so far
    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if collector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Convert to sorted top-k results (descending by score)
    pub fn into_sorted_results(self) -> Vec<(DocId, f32, u16)> {
        let mut results: Vec<(DocId, f32, u16)> = self
            .heap
            .into_vec()
            .into_iter()
            .map(|e| (e.doc_id, e.score, e.ordinal))
            .collect();

        // Sort by score descending, then doc_id ascending
        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        results
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
    cursors: Vec<TermCursor<'a>>,
    prefix_sums: Vec<f32>,
    collector: ScoreCollector,
    heap_factor: f32,
    predicate: Option<super::DocPredicate<'a>>,
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

enum CursorVariant<'a> {
    /// Full-text BM25 — in-memory BlockPostingList (skip list + block data)
    Text {
        list: crate::structures::BlockPostingList,
        idf: f32,
        /// Precomputed: BM25_B / max(avg_field_len, 1.0)
        b_over_avgfl: f32,
        /// Precomputed: 1.0 - BM25_B
        one_minus_b: f32,
        tfs: Vec<u32>, // temp decode buffer, converted to scores
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
                idf,
                b_over_avgfl,
                one_minus_b,
                tfs,
            } => {
                if list.decode_block_into($self.block_idx, &mut $self.doc_ids, tfs) {
                    let idf_val = *idf;
                    let b_avg = *b_over_avgfl;
                    let one_b = *one_minus_b;
                    $self.scores.clear();
                    $self.scores.reserve(tfs.len());
                    // Precomputed BM25: length_norm = one_minus_b + b_over_avgfl * tf
                    // (tf is used as both term frequency and doc length — a known approx)
                    for &tf in tfs.iter() {
                        let tf = tf as f32;
                        let length_norm = one_b + b_avg * tf;
                        let tf_norm = (tf * (super::BM25_K1 + 1.0))
                            / (tf + super::BM25_K1 * length_norm);
                        $self.scores.push(idf_val * tf_norm);
                    }
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
    /// Create a full-text BM25 cursor (lazy — no blocks decoded yet).
    pub fn text(
        posting_list: crate::structures::BlockPostingList,
        idf: f32,
        avg_field_len: f32,
    ) -> Self {
        let max_tf = posting_list.max_tf() as f32;
        let max_score = super::bm25_upper_bound(max_tf.max(1.0), idf);
        let num_blocks = posting_list.num_blocks();
        let safe_avg = avg_field_len.max(1.0);
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
                b_over_avgfl: super::BM25_B / safe_avg,
                one_minus_b: 1.0 - super::BM25_B,
                tfs: Vec::with_capacity(128),
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

    #[inline]
    pub fn current_block_max_score(&self) -> f32 {
        if self.exhausted {
            return 0.0;
        }
        match &self.variant {
            CursorVariant::Text { list, idf, .. } => {
                let block_max_tf = list.block_max_tf(self.block_idx).unwrap_or(0) as f32;
                super::bm25_upper_bound(block_max_tf.max(1.0), *idf)
            }
            CursorVariant::Sparse {
                si,
                query_weight,
                skip_start,
                ..
            } => query_weight.abs() * si.read_skip_entry(*skip_start + self.block_idx).max_weight,
        }
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
        let mut conjunction_skipped = 0u64;
        let mut ordinal_scores: Vec<(u16, f32)> = Vec::with_capacity(n * 2);
        let _bms_start = std::time::Instant::now();

        loop {
            let partition = $self.find_partition();
            if partition >= n {
                break;
            }

            // Find minimum doc_id across essential cursors and collect
            // which cursors are at min_doc (avoids redundant re-checks in
            // conjunction, block-max, predicate, and scoring passes).
            let mut min_doc = u32::MAX;
            let mut at_min_mask = 0u64; // bitset of cursor indices at min_doc
            for i in partition..n {
                let doc = $self.cursors[i].doc();
                match doc.cmp(&min_doc) {
                    std::cmp::Ordering::Less => {
                        min_doc = doc;
                        at_min_mask = 1u64 << (i as u32);
                    }
                    std::cmp::Ordering::Equal => {
                        at_min_mask |= 1u64 << (i as u32);
                    }
                    _ => {}
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
            // Small epsilon to guard against FP rounding in score accumulation.
            // Without this, a document whose true score equals the threshold can
            // be incorrectly pruned due to rounding in the heap_factor multiply
            // or in the prefix_sum additions.
            let adjusted_threshold = $self.collector.threshold() * $self.heap_factor - 1e-6;

            // --- Conjunction optimization ---
            if $self.collector.len() >= $self.collector.k {
                let mut present_upper: f32 = 0.0;
                let mut mask = at_min_mask;
                while mask != 0 {
                    let i = mask.trailing_zeros() as usize;
                    present_upper += $self.cursors[i].max_score;
                    mask &= mask - 1;
                }

                if present_upper + non_essential_upper <= adjusted_threshold {
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

                if block_max_sum + non_essential_upper <= adjusted_threshold {
                    let mut mask = at_min_mask;
                    while mask != 0 {
                        let i = mask.trailing_zeros() as usize;
                        $self.cursors[i].skip_to_next_block();
                        $self.cursors[i].$ensure() $($aw)* ?;
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
                && essential_total + non_essential_upper <= adjusted_threshold
            {
                docs_skipped += 1;
                continue;
            }

            // --- Score non-essential cursors (highest max_score first for early exit) ---
            let mut running_total = essential_total;
            for i in (0..partition).rev() {
                if $self.collector.len() >= $self.collector.k
                    && running_total + $self.prefix_sums[i] <= adjusted_threshold
                {
                    break;
                }

                let doc = $self.cursors[i].$seek(min_doc) $($aw)* ?;
                if doc == min_doc {
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
                "slow MaxScore: {}ms, cursors={}, scored={}, skipped={}, blocks_skipped={}, conjunction_skipped={}, returned={}, top_score={:.4}",
                _bms_elapsed_ms,
                n,
                docs_scored,
                docs_skipped,
                blocks_skipped,
                conjunction_skipped,
                results.len(),
                results.first().map(|r| r.score).unwrap_or(0.0)
            );
        } else {
            debug!(
                "MaxScoreExecutor: {}ms, scored={}, skipped={}, blocks_skipped={}, conjunction_skipped={}, returned={}, top_score={:.4}",
                _bms_elapsed_ms,
                docs_scored,
                docs_skipped,
                blocks_skipped,
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

        debug!(
            "Creating MaxScoreExecutor: num_cursors={}, k={}, total_upper={:.4}, heap_factor={:.2}",
            cursors.len(),
            k,
            cumsum,
            heap_factor
        );

        Self {
            cursors,
            prefix_sums,
            collector: ScoreCollector::new(k),
            heap_factor: heap_factor.clamp(0.0, 1.0),
            predicate: None,
        }
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
    ) -> Self {
        let cursors: Vec<TermCursor<'a>> = posting_lists
            .into_iter()
            .map(|(pl, idf)| TermCursor::text(pl, idf, avg_field_len))
            .collect();
        Self::new(cursors, k, 1.0)
    }

    #[inline]
    fn find_partition(&self) -> usize {
        let threshold = self.collector.threshold() * self.heap_factor;
        self.prefix_sums.partition_point(|&sum| sum <= threshold)
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

    /// Execute Block-Max MaxScore and return top-k results (async).
    pub async fn execute(mut self) -> crate::Result<Vec<ScoredDoc>> {
        if self.cursors.is_empty() {
            return Ok(Vec::new());
        }
        bms_execute_loop!(self, ensure_block_loaded, advance, seek, .await)
    }

    /// Synchronous execution — works when all cursors are text or mmap-backed sparse.
    pub fn execute_sync(mut self) -> crate::Result<Vec<ScoredDoc>> {
        if self.cursors.is_empty() {
            return Ok(Vec::new());
        }
        bms_execute_loop!(self, ensure_block_loaded_sync, advance_sync, seek_sync,)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
