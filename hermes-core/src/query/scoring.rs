//! Shared scoring abstractions for text and sparse vector search
//!
//! Provides common types and executors for efficient top-k retrieval:
//! - `TermCursor`: Unified cursor for both BM25 text and sparse vector posting lists
//! - `ScoreCollector`: Efficient min-heap for maintaining top-k results
//! - `MaxScoreExecutor`: Unified Block-Max MaxScore with conjunction optimization
//! - `BmpExecutor`: Block-at-a-time executor for learned sparse retrieval (12+ terms)

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use log::debug;

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
        // Min-heap: lower scores come first (to be evicted)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Efficient top-k collector using min-heap
///
/// Maintains the k highest-scoring documents using a min-heap where the
/// lowest score is at the top for O(1) threshold lookup and O(log k) eviction.
/// No deduplication - caller must ensure each doc_id is inserted only once.
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
            self.update_threshold();
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
        let heap_vec = self.heap.into_vec();
        let mut results: Vec<(DocId, f32, u16)> = Vec::with_capacity(heap_vec.len());
        for e in heap_vec {
            results.push((e.doc_id, e.score, e.ordinal));
        }

        // Sort by score descending, then doc_id ascending
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

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

/// Block-Max Pruning (BMP) executor for learned sparse retrieval
///
/// Processes blocks in score-descending order using a priority queue.
/// Best for queries with many terms (20+), like SPLADE expansions.
/// Uses document accumulators (FxHashMap) instead of per-term iterators.
///
/// **Memory-efficient**: Only skip entries (block metadata) are kept in memory.
/// Actual block data is loaded on-demand via mmap range reads during execution.
///
/// Reference: Mallia et al., "Faster Learned Sparse Retrieval with
/// Block-Max Pruning" (SIGIR 2024)
pub struct BmpExecutor<'a> {
    /// Sparse index for on-demand block loading
    sparse_index: &'a crate::segment::SparseIndex,
    /// Query terms: (dim_id, query_weight) for each matched dimension
    query_terms: Vec<(u32, f32)>,
    /// Number of results to return
    k: usize,
    /// Heap factor for approximate search
    heap_factor: f32,
    /// Optional filter predicate (checked at final collection)
    predicate: Option<super::DocPredicate<'a>>,
}

/// Superblock size: group S consecutive blocks into one priority queue entry.
/// Reduces heap operations by S× (e.g. 8× fewer push/pop for S=8).
const BMP_SUPERBLOCK_SIZE: usize = 8;

/// Megablock size: group M superblocks into one outer priority queue entry.
/// Two-level pruning: megablock-level (coarse) → superblock-level (fine).
/// Reduces outer heap operations by M× compared to single-level superblocks.
const BMP_MEGABLOCK_SIZE: usize = 16;

/// Superblock entry (stored per-term, not in the heap directly)
struct BmpSuperBlock {
    /// Upper bound contribution of this superblock (sum of constituent blocks)
    contribution: f32,
    /// First block index in this superblock
    block_start: usize,
    /// Number of blocks in this superblock (1..=BMP_SUPERBLOCK_SIZE)
    block_count: usize,
}

/// Entry in the BMP outer priority queue: represents a megablock (group of superblocks)
struct BmpMegaBlockEntry {
    /// Upper bound contribution of this megablock (sum of constituent superblocks)
    contribution: f32,
    /// Index into query_terms
    term_idx: usize,
    /// First superblock index within term_superblocks[term_idx]
    sb_start: usize,
    /// Number of superblocks in this megablock (1..=BMP_MEGABLOCK_SIZE)
    sb_count: usize,
}

impl PartialEq for BmpMegaBlockEntry {
    fn eq(&self, other: &Self) -> bool {
        self.contribution == other.contribution
    }
}

impl Eq for BmpMegaBlockEntry {}

impl Ord for BmpMegaBlockEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: higher contributions come first
        self.contribution
            .partial_cmp(&other.contribution)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for BmpMegaBlockEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Macro to stamp out the BMP execution loop for both async and sync paths.
///
/// `$get_blocks:ident` is the SparseIndex method (get_blocks_range or get_blocks_range_sync).
/// `$($aw:tt)*` captures `.await` for async or nothing for sync.
macro_rules! bmp_execute_loop {
    ($self:ident, $get_blocks:ident, $($aw:tt)*) => {{
        use rustc_hash::FxHashMap;

        let num_terms = $self.query_terms.len();
        let si = $self.sparse_index;

        // Two-level queue construction:
        // 1. Build superblocks per term (flat Vecs)
        // 2. Group superblocks into megablocks, push to outer BinaryHeap
        let mut term_superblocks: Vec<Vec<BmpSuperBlock>> = Vec::with_capacity(num_terms);
        let mut term_skip_starts: Vec<usize> = Vec::with_capacity(num_terms);
        let mut global_min_doc = u32::MAX;
        let mut global_max_doc = 0u32;
        let mut total_remaining = 0.0f32;

        for &(dim_id, qw) in &$self.query_terms {
            let mut term_skip_start = 0usize;
            let mut superblocks = Vec::new();

            let abs_qw = qw.abs();
            if let Some((skip_start, skip_count, _global_max)) = si.get_skip_range(dim_id) {
                term_skip_start = skip_start;
                let mut sb_start = 0;
                while sb_start < skip_count {
                    let sb_count = (skip_count - sb_start).min(BMP_SUPERBLOCK_SIZE);
                    let mut sb_contribution = 0.0f32;
                    for j in 0..sb_count {
                        let skip = si.read_skip_entry(skip_start + sb_start + j);
                        sb_contribution += abs_qw * skip.max_weight;
                        global_min_doc = global_min_doc.min(skip.first_doc);
                        global_max_doc = global_max_doc.max(skip.last_doc);
                    }
                    total_remaining += sb_contribution;
                    superblocks.push(BmpSuperBlock {
                        contribution: sb_contribution,
                        block_start: sb_start,
                        block_count: sb_count,
                    });
                    sb_start += sb_count;
                }
            }
            term_skip_starts.push(term_skip_start);
            term_superblocks.push(superblocks);
        }

        // Step 2: Group superblocks into megablocks and build outer priority queue
        let mut mega_queue: BinaryHeap<BmpMegaBlockEntry> = BinaryHeap::new();
        for (term_idx, superblocks) in term_superblocks.iter().enumerate() {
            let mut mb_start = 0;
            while mb_start < superblocks.len() {
                let mb_count = (superblocks.len() - mb_start).min(BMP_MEGABLOCK_SIZE);
                let mb_contribution: f32 = superblocks[mb_start..mb_start + mb_count]
                    .iter()
                    .map(|sb| sb.contribution)
                    .sum();
                mega_queue.push(BmpMegaBlockEntry {
                    contribution: mb_contribution,
                    term_idx,
                    sb_start: mb_start,
                    sb_count: mb_count,
                });
                mb_start += mb_count;
            }
        }

        // Hybrid accumulator: flat array for ordinal=0, FxHashMap for multi-ordinal
        let doc_range = if global_max_doc >= global_min_doc {
            (global_max_doc - global_min_doc + 1) as usize
        } else {
            0
        };
        let use_flat = doc_range > 0 && doc_range <= 256 * 1024;
        let mut flat_scores: Vec<f32> = if use_flat {
            vec![0.0; doc_range]
        } else {
            Vec::new()
        };
        let mut dirty: Vec<u32> = if use_flat {
            Vec::with_capacity(4096)
        } else {
            Vec::new()
        };
        let mut multi_ord_accumulators: FxHashMap<u64, f32> = FxHashMap::default();

        let mut blocks_processed = 0u64;
        let mut blocks_skipped = 0u64;

        let mut top_k = ScoreCollector::new($self.k);

        let mut doc_ids_buf: Vec<u32> = Vec::with_capacity(256);
        let mut weights_buf: Vec<f32> = Vec::with_capacity(256);
        let mut ordinals_buf: Vec<u16> = Vec::with_capacity(256);

        let mut terms_warmed = vec![false; num_terms];
        let mut warmup_remaining = $self.k.min(num_terms);

        while let Some(mega) = mega_queue.pop() {
            total_remaining -= mega.contribution;

            if !terms_warmed[mega.term_idx] {
                terms_warmed[mega.term_idx] = true;
                warmup_remaining = warmup_remaining.saturating_sub(1);
            }

            if warmup_remaining == 0 {
                let adjusted_threshold = top_k.threshold() * $self.heap_factor;
                if top_k.len() >= $self.k && total_remaining <= adjusted_threshold {
                    let remaining_blocks: u64 = mega_queue
                        .iter()
                        .map(|m| {
                            let sbs =
                                &term_superblocks[m.term_idx][m.sb_start..m.sb_start + m.sb_count];
                            sbs.iter().map(|sb| sb.block_count as u64).sum::<u64>()
                        })
                        .sum();
                    blocks_skipped += remaining_blocks;
                    debug!(
                        "BMP early termination after {} blocks: remaining={:.4} <= threshold={:.4}",
                        blocks_processed, total_remaining, adjusted_threshold
                    );
                    break;
                }
            }

            let dim_id = $self.query_terms[mega.term_idx].0;
            let qw = $self.query_terms[mega.term_idx].1;
            let abs_qw = qw.abs();
            let skip_start = term_skip_starts[mega.term_idx];

            for sb in term_superblocks[mega.term_idx]
                .iter()
                .skip(mega.sb_start)
                .take(mega.sb_count)
            {
                if top_k.len() >= $self.k {
                    let adjusted_threshold = top_k.threshold() * $self.heap_factor;
                    if sb.contribution + total_remaining <= adjusted_threshold {
                        blocks_skipped += sb.block_count as u64;
                        continue;
                    }
                }

                // Coalesced superblock loading — async or sync dispatch point
                let sb_blocks = si
                    .$get_blocks(dim_id, sb.block_start, sb.block_count)
                    $($aw)*?;

                let adjusted_threshold2 = top_k.threshold() * $self.heap_factor;
                let dirty_start = dirty.len();

                for (blk_offset, block) in sb_blocks.into_iter().enumerate() {
                    let blk_idx = sb.block_start + blk_offset;

                    if top_k.len() >= $self.k {
                        let skip = si.read_skip_entry(skip_start + blk_idx);
                        let blk_contrib = abs_qw * skip.max_weight;
                        if blk_contrib + total_remaining <= adjusted_threshold2 {
                            blocks_skipped += 1;
                            continue;
                        }
                    }

                    block.decode_doc_ids_into(&mut doc_ids_buf);

                    if block.header.ordinal_bits == 0 && use_flat {
                        block.accumulate_scored_weights(
                            qw,
                            &doc_ids_buf,
                            &mut flat_scores,
                            global_min_doc,
                            &mut dirty,
                        );
                    } else {
                        block.decode_scored_weights_into(qw, &mut weights_buf);
                        let count = block.header.count as usize;

                        block.decode_ordinals_into(&mut ordinals_buf);
                        if use_flat {
                            for i in 0..count {
                                let doc_id = doc_ids_buf[i];
                                let ordinal = ordinals_buf[i];
                                let score_contribution = weights_buf[i];

                                if ordinal == 0 {
                                    let off = (doc_id - global_min_doc) as usize;
                                    if flat_scores[off] == 0.0 {
                                        dirty.push(doc_id);
                                    }
                                    flat_scores[off] += score_contribution;
                                } else {
                                    let key = (doc_id as u64) << 16 | ordinal as u64;
                                    let acc = multi_ord_accumulators.entry(key).or_insert(0.0);
                                    *acc += score_contribution;
                                    top_k.insert_with_ordinal(doc_id, *acc, ordinal);
                                }
                            }
                        } else {
                            for i in 0..count {
                                let key = (doc_ids_buf[i] as u64) << 16 | ordinals_buf[i] as u64;
                                let acc = multi_ord_accumulators.entry(key).or_insert(0.0);
                                *acc += weights_buf[i];
                                top_k.insert_with_ordinal(doc_ids_buf[i], *acc, ordinals_buf[i]);
                            }
                        }
                    }

                    blocks_processed += 1;
                }

                for &doc_id in &dirty[dirty_start..] {
                    let off = (doc_id - global_min_doc) as usize;
                    top_k.insert_with_ordinal(doc_id, flat_scores[off], 0);
                }
            }
        }

        // Collect final top-k with predicate filtering
        let mut final_top_k = ScoreCollector::new($self.k);

        let num_accumulators = if use_flat {
            for &doc_id in &dirty {
                if let Some(ref pred) = $self.predicate
                    && !pred(doc_id)
                {
                    continue;
                }
                let off = (doc_id - global_min_doc) as usize;
                let score = flat_scores[off];
                if score > 0.0 {
                    final_top_k.insert_with_ordinal(doc_id, score, 0);
                }
            }
            dirty.len() + multi_ord_accumulators.len()
        } else {
            multi_ord_accumulators.len()
        };

        for (key, score) in &multi_ord_accumulators {
            let doc_id = (*key >> 16) as crate::DocId;
            if let Some(ref pred) = $self.predicate
                && !pred(doc_id)
            {
                continue;
            }
            final_top_k.insert_with_ordinal(doc_id, *score, (*key & 0xFFFF) as u16);
        }

        let results: Vec<ScoredDoc> = final_top_k
            .into_sorted_results()
            .into_iter()
            .map(|(doc_id, score, ordinal)| ScoredDoc {
                doc_id,
                score,
                ordinal,
            })
            .collect();

        debug!(
            "BmpExecutor completed: blocks_processed={}, blocks_skipped={}, accumulators={}, flat={}, returned={}, top_score={:.4}",
            blocks_processed,
            blocks_skipped,
            num_accumulators,
            use_flat,
            results.len(),
            results.first().map(|r| r.score).unwrap_or(0.0)
        );

        Ok(results)
    }};
}

impl<'a> BmpExecutor<'a> {
    /// Create a new BMP executor with lazy block loading
    ///
    /// `query_terms` should contain only dimensions that exist in the index.
    /// Block metadata (skip entries) is read from the sparse index directly.
    pub fn new(
        sparse_index: &'a crate::segment::SparseIndex,
        query_terms: Vec<(u32, f32)>,
        k: usize,
        heap_factor: f32,
    ) -> Self {
        Self {
            sparse_index,
            query_terms,
            k,
            heap_factor: heap_factor.clamp(0.0, 1.0),
            predicate: None,
        }
    }

    /// Set a filter predicate that rejects documents at final collection.
    pub fn set_predicate(&mut self, predicate: Option<super::DocPredicate<'a>>) {
        self.predicate = predicate;
    }

    /// Execute BMP and return top-k results (async).
    pub async fn execute(self) -> crate::Result<Vec<ScoredDoc>> {
        if self.query_terms.is_empty() {
            return Ok(Vec::new());
        }
        bmp_execute_loop!(self, get_blocks_range, .await)
    }

    /// Synchronous BMP execution — works when sparse index is mmap-backed.
    #[cfg(feature = "sync")]
    pub fn execute_sync(self) -> crate::Result<Vec<ScoredDoc>> {
        if self.query_terms.is_empty() {
            return Ok(Vec::new());
        }
        bmp_execute_loop!(self, get_blocks_range_sync,)
    }
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
    // ── Block decode + skip access source ───────────────────────────
    variant: CursorVariant<'a>,
}

enum CursorVariant<'a> {
    /// Full-text BM25 — in-memory BlockPostingList (skip list + block data)
    Text {
        list: crate::structures::BlockPostingList,
        idf: f32,
        avg_field_len: f32,
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
            variant: CursorVariant::Text {
                list: posting_list,
                idf,
                avg_field_len,
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
            self.doc_ids.get(self.pos).copied().unwrap_or(u32::MAX)
        } else {
            self.block_first_doc(self.block_idx)
        }
    }

    #[inline]
    pub fn ordinal(&self) -> u16 {
        if !self.block_loaded || self.ordinals.is_empty() {
            return 0;
        }
        self.ordinals.get(self.pos).copied().unwrap_or(0)
    }

    #[inline]
    pub fn score(&self) -> f32 {
        if !self.block_loaded {
            return 0.0;
        }
        self.scores.get(self.pos).copied().unwrap_or(0.0)
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

    // ── Block loading (dispatch: decode format + I/O differ) ────────────

    pub async fn ensure_block_loaded(&mut self) -> crate::Result<bool> {
        if self.exhausted || self.block_loaded {
            return Ok(!self.exhausted);
        }
        match &mut self.variant {
            CursorVariant::Text {
                list,
                idf,
                avg_field_len,
                tfs,
            } => {
                if list.decode_block_into(self.block_idx, &mut self.doc_ids, tfs) {
                    self.scores.clear();
                    self.scores.reserve(tfs.len());
                    for &tf in tfs.iter() {
                        let tf = tf as f32;
                        self.scores
                            .push(super::bm25_score(tf, *idf, tf, *avg_field_len));
                    }
                    self.pos = 0;
                    self.block_loaded = true;
                    Ok(true)
                } else {
                    self.exhausted = true;
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
                    .load_block_direct(*skip_start, *block_data_offset, self.block_idx)
                    .await?;
                match block {
                    Some(b) => {
                        b.decode_doc_ids_into(&mut self.doc_ids);
                        b.decode_ordinals_into(&mut self.ordinals);
                        b.decode_scored_weights_into(*query_weight, &mut self.scores);
                        self.pos = 0;
                        self.block_loaded = true;
                        Ok(true)
                    }
                    None => {
                        self.exhausted = true;
                        Ok(false)
                    }
                }
            }
        }
    }

    pub fn ensure_block_loaded_sync(&mut self) -> crate::Result<bool> {
        if self.exhausted || self.block_loaded {
            return Ok(!self.exhausted);
        }
        match &mut self.variant {
            CursorVariant::Text {
                list,
                idf,
                avg_field_len,
                tfs,
            } => {
                if list.decode_block_into(self.block_idx, &mut self.doc_ids, tfs) {
                    self.scores.clear();
                    self.scores.reserve(tfs.len());
                    for &tf in tfs.iter() {
                        let tf = tf as f32;
                        self.scores
                            .push(super::bm25_score(tf, *idf, tf, *avg_field_len));
                    }
                    self.pos = 0;
                    self.block_loaded = true;
                    Ok(true)
                } else {
                    self.exhausted = true;
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
                let block =
                    si.load_block_direct_sync(*skip_start, *block_data_offset, self.block_idx)?;
                match block {
                    Some(b) => {
                        b.decode_doc_ids_into(&mut self.doc_ids);
                        b.decode_ordinals_into(&mut self.ordinals);
                        b.decode_scored_weights_into(*query_weight, &mut self.scores);
                        self.pos = 0;
                        self.block_loaded = true;
                        Ok(true)
                    }
                    None => {
                        self.exhausted = true;
                        Ok(false)
                    }
                }
            }
        }
    }

    // ── Advance / Seek ──────────────────────────────────────────────────

    pub async fn advance(&mut self) -> crate::Result<DocId> {
        if self.exhausted {
            return Ok(u32::MAX);
        }
        self.ensure_block_loaded().await?;
        if self.exhausted {
            return Ok(u32::MAX);
        }
        Ok(self.advance_pos())
    }

    pub fn advance_sync(&mut self) -> crate::Result<DocId> {
        if self.exhausted {
            return Ok(u32::MAX);
        }
        self.ensure_block_loaded_sync()?;
        if self.exhausted {
            return Ok(u32::MAX);
        }
        Ok(self.advance_pos())
    }

    pub async fn seek(&mut self, target: DocId) -> crate::Result<DocId> {
        if let Some(doc) = self.seek_prepare(target) {
            return Ok(doc);
        }
        self.ensure_block_loaded().await?;
        if self.seek_finish(target) {
            self.ensure_block_loaded().await?;
        }
        Ok(self.doc())
    }

    pub fn seek_sync(&mut self, target: DocId) -> crate::Result<DocId> {
        if let Some(doc) = self.seek_prepare(target) {
            return Ok(doc);
        }
        self.ensure_block_loaded_sync()?;
        if self.seek_finish(target) {
            self.ensure_block_loaded_sync()?;
        }
        Ok(self.doc())
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

        loop {
            let partition = $self.find_partition();
            if partition >= n {
                break;
            }

            // Find minimum doc_id across essential cursors
            let mut min_doc = u32::MAX;
            for i in partition..n {
                let doc = $self.cursors[i].doc();
                if doc < min_doc {
                    min_doc = doc;
                }
            }
            if min_doc == u32::MAX {
                break;
            }

            // --- Filter predicate check (before any scoring) ---
            if let Some(ref pred) = $self.predicate {
                if !pred(min_doc) {
                    // Advance essential cursors past this doc
                    for i in partition..n {
                        if $self.cursors[i].doc() == min_doc {
                            $self.cursors[i].$ensure() $($aw)* ?;
                            $self.cursors[i].$advance() $($aw)* ?;
                        }
                    }
                    docs_skipped += 1;
                    continue;
                }
            }

            let non_essential_upper = if partition > 0 {
                $self.prefix_sums[partition - 1]
            } else {
                0.0
            };
            let adjusted_threshold = $self.collector.threshold() * $self.heap_factor;

            // --- Conjunction optimization ---
            if $self.collector.len() >= $self.collector.k {
                let present_upper: f32 = (partition..n)
                    .filter(|&i| $self.cursors[i].doc() == min_doc)
                    .map(|i| $self.cursors[i].max_score)
                    .sum();

                if present_upper + non_essential_upper <= adjusted_threshold {
                    for i in partition..n {
                        if $self.cursors[i].doc() == min_doc {
                            $self.cursors[i].$ensure() $($aw)* ?;
                            $self.cursors[i].$advance() $($aw)* ?;
                        }
                    }
                    conjunction_skipped += 1;
                    continue;
                }
            }

            // --- Block-max pruning ---
            if $self.collector.len() >= $self.collector.k {
                let block_max_sum: f32 = (partition..n)
                    .filter(|&i| $self.cursors[i].doc() == min_doc)
                    .map(|i| $self.cursors[i].current_block_max_score())
                    .sum();

                if block_max_sum + non_essential_upper <= adjusted_threshold {
                    for i in partition..n {
                        if $self.cursors[i].doc() == min_doc {
                            $self.cursors[i].skip_to_next_block();
                            $self.cursors[i].$ensure() $($aw)* ?;
                        }
                    }
                    blocks_skipped += 1;
                    continue;
                }
            }

            // --- Score essential cursors ---
            ordinal_scores.clear();
            for i in partition..n {
                if $self.cursors[i].doc() == min_doc {
                    $self.cursors[i].$ensure() $($aw)* ?;
                    while $self.cursors[i].doc() == min_doc {
                        ordinal_scores.push(($self.cursors[i].ordinal(), $self.cursors[i].score()));
                        $self.cursors[i].$advance() $($aw)* ?;
                    }
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
                        ordinal_scores.push(($self.cursors[i].ordinal(), s));
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
                } else if ordinal_scores[0].0 > ordinal_scores[1].0 {
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

        debug!(
            "MaxScoreExecutor: scored={}, skipped={}, blocks_skipped={}, conjunction_skipped={}, returned={}, top_score={:.4}",
            docs_scored,
            docs_skipped,
            blocks_skipped,
            conjunction_skipped,
            results.len(),
            results.first().map(|r| r.score).unwrap_or(0.0)
        );

        Ok(results)
    }};
}

impl<'a> MaxScoreExecutor<'a> {
    /// Create a new executor from pre-built cursors.
    ///
    /// Cursors are sorted by max_score ascending (non-essential first) and
    /// prefix sums are computed for the MaxScore partitioning.
    pub(crate) fn new(mut cursors: Vec<TermCursor<'a>>, k: usize, heap_factor: f32) -> Self {
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

    /// Set a filter predicate that rejects documents before scoring.
    pub fn set_predicate(&mut self, predicate: Option<super::DocPredicate<'a>>) {
        self.predicate = predicate;
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
