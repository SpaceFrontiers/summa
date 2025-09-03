//! Query and Scorer traits with async support
//!
//! Provides the core abstractions for search queries and document scoring.

use std::future::Future;
use std::pin::Pin;

use crate::segment::SegmentReader;
use crate::{DocId, Result, Score};

/// Future type for scorer creation
#[cfg(not(target_arch = "wasm32"))]
pub type ScorerFuture<'a> = Pin<Box<dyn Future<Output = Result<Box<dyn Scorer + 'a>>> + Send + 'a>>;
#[cfg(target_arch = "wasm32")]
pub type ScorerFuture<'a> = Pin<Box<dyn Future<Output = Result<Box<dyn Scorer + 'a>>> + 'a>>;

/// Options that affect scorer construction rather than scoring semantics.
///
/// Position postings can be much larger than the top-k result itself. Keeping
/// this explicit lets ID/score-only collectors avoid loading them while query
/// types that need positions for matching (for example phrases) remain free to
/// load their own internal data.
#[derive(Debug, Clone, Default)]
pub struct ScorerOptions {
    /// Eligibility pushed into candidate collectors; never a scoring feature.
    pub(crate) eligibility: Option<std::sync::Arc<DocBitset>>,
    pub collect_positions: bool,
    /// Initial top-k score floor to seed into MaxScore/BMP pruning. Used to
    /// carry the running k-th score across the segments of one query so later
    /// segments prune from a nonzero threshold (see `SharedThreshold`). 0.0 =
    /// no seed. Only honored on exact, final-score executor paths.
    pub initial_threshold: f32,
    /// Live form of `initial_threshold`. Exact final-score executors may read
    /// it during traversal so concurrently searched segments benefit as soon
    /// as another segment establishes a stronger global floor.
    pub shared_threshold: Option<super::scoring::SharedThreshold>,
    /// Query-global LSP/0 selection projected onto this segment.
    pub(crate) lsp_plan: Option<std::sync::Arc<super::bmp::LspSegmentPlan>>,
    /// Query-global text statistics (document frequencies, corpus sizes,
    /// average lengths aggregated over every segment of the searcher, or
    /// supplied by a broker across shards). Text scorers use them for IDF
    /// and length normalisation so a term scores the same in every segment;
    /// a query's own `with_global_stats` takes precedence.
    pub global_stats: Option<std::sync::Arc<super::GlobalStats>>,
}

impl ScorerOptions {
    pub const fn with_positions() -> Self {
        Self {
            eligibility: None,
            collect_positions: true,
            initial_threshold: 0.0,
            shared_threshold: None,
            lsp_plan: None,
            global_stats: None,
        }
    }

    /// Preserve collection behavior while preventing a nested/component
    /// scorer from applying a floor expressed in the outer query's score
    /// space.
    pub fn without_threshold(&self) -> Self {
        Self {
            eligibility: self.eligibility.clone(),
            collect_positions: self.collect_positions,
            initial_threshold: 0.0,
            shared_threshold: self
                .shared_threshold
                .as_ref()
                .filter(|shared| shared.deadline().is_some())
                .map(super::SharedThreshold::budget_only),
            lsp_plan: None,
            global_stats: self.global_stats.clone(),
        }
    }

    pub(crate) fn stop_if_expired(&self) -> bool {
        self.shared_threshold
            .as_ref()
            .is_some_and(super::SharedThreshold::stop_if_expired)
    }

    /// Materialization uses the same budget as scoring. Implementations must
    /// never return a partial bitset (especially for MUST_NOT).
    pub(crate) fn doc_bitset(
        &self,
        query: &dyn Query,
        reader: &SegmentReader,
    ) -> Option<DocBitset> {
        query.as_doc_bitset_with_options(reader, self)
    }
}

/// Future type for count estimation
#[cfg(not(target_arch = "wasm32"))]
pub type CountFuture<'a> = Pin<Box<dyn Future<Output = Result<u32>> + Send + 'a>>;
#[cfg(target_arch = "wasm32")]
pub type CountFuture<'a> = Pin<Box<dyn Future<Output = Result<u32>> + 'a>>;

/// Per-document predicate closure type (platform-aware Send+Sync bounds)
#[cfg(not(target_arch = "wasm32"))]
pub type DocPredicate<'a> = Box<dyn Fn(DocId) -> bool + Send + Sync + 'a>;
#[cfg(target_arch = "wasm32")]
pub type DocPredicate<'a> = Box<dyn Fn(DocId) -> bool + 'a>;

/// Compact bitset indexed by doc_id. O(1) lookup, ~2.25 MB for 18M docs.
///
/// Built from posting lists or predicate scans. Used by BMP filtered queries
/// to avoid repeated fast-field decoding during per-slot predicate evaluation.
/// Lookup cost depends on residency and the caller's dispatch, not just this type.
#[derive(Debug)]
pub struct DocBitset {
    pub(crate) bits: Vec<u64>,
}

impl DocBitset {
    /// Create an empty bitset for `num_docs` documents.
    pub fn new(num_docs: u32) -> Self {
        let num_words = (num_docs as usize).div_ceil(64);
        Self {
            bits: vec![0u64; num_words],
        }
    }

    /// The segment's document universe, with padding bits left clear.
    pub(crate) fn all(num_docs: u32) -> Self {
        let mut result = Self::new(num_docs);
        result.bits.fill(u64::MAX);
        if !num_docs.is_multiple_of(64)
            && let Some(last) = result.bits.last_mut()
        {
            *last = (1u64 << (num_docs % 64)) - 1;
        }
        result
    }

    /// Set bit for `doc_id`.
    #[inline]
    pub fn set(&mut self, doc_id: u32) {
        let word = doc_id as usize / 64;
        let bit = doc_id as usize % 64;
        if word < self.bits.len() {
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// First set bit at or after `from`, if any.
    pub fn next_set_bit(&self, from: DocId) -> Option<DocId> {
        let mut word = from as usize / 64;
        if word >= self.bits.len() {
            return None;
        }
        let mut bits = self.bits[word] & (u64::MAX << (from % 64));
        loop {
            if bits != 0 {
                return Some((word * 64 + bits.trailing_zeros() as usize) as DocId);
            }
            word += 1;
            if word >= self.bits.len() {
                return None;
            }
            bits = self.bits[word];
        }
    }

    /// Test if `doc_id` is in the bitset.
    #[inline(always)]
    pub fn contains(&self, doc_id: u32) -> bool {
        let word = doc_id as usize / 64;
        let bit = doc_id as usize % 64;
        word < self.bits.len() && self.bits[word] & (1u64 << bit) != 0
    }

    /// Number of set bits (matching docs).
    pub fn count(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    /// Build bitset from a predicate by scanning all docs. O(N).
    pub fn from_predicate(num_docs: u32, pred: &dyn Fn(DocId) -> bool) -> Self {
        let mut bs = Self::new(num_docs);
        for doc_id in 0..num_docs {
            if pred(doc_id) {
                bs.set(doc_id);
            }
        }
        bs
    }

    /// In-place OR (union): `self |= other`.
    pub fn union_with(&mut self, other: &DocBitset) {
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= *b;
        }
    }

    /// In-place AND (intersection): `self &= other`.
    pub fn intersect_with(&mut self, other: &DocBitset) {
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a &= *b;
        }
        // Zero out any words beyond `other`'s length
        for a in self.bits.iter_mut().skip(other.bits.len()) {
            *a = 0;
        }
    }

    /// In-place ANDNOT (subtract): `self &= !other`.
    pub fn subtract(&mut self, other: &DocBitset) {
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a &= !*b;
        }
    }

    /// Keep only the set docs for which `pred` returns true. O(count) probes —
    /// the planner uses this to refine a small accumulator against a wide
    /// clause instead of materializing that clause's full bitset.
    pub fn retain(&mut self, pred: &dyn Fn(DocId) -> bool) {
        for (w, word) in self.bits.iter_mut().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                let b = bits.trailing_zeros();
                let doc = (w * 64) as u32 + b;
                if !pred(doc) {
                    *word &= !(1u64 << b);
                }
                bits &= bits - 1;
            }
        }
    }
}

/// Info for MaxScore-optimizable term queries
#[derive(Debug, Clone)]
pub struct TermQueryInfo {
    /// Field being searched
    pub field: crate::dsl::Field,
    /// Term bytes (lowercase)
    pub term: Vec<u8>,
    /// Query-side weight of the term (a boost, or the query term frequency
    /// of a de-duplicated match); scales the term's idf, hence its scores
    /// and bounds alike. 1.0 = plain.
    pub weight: f32,
}

/// Info for MaxScore-optimizable sparse term queries
#[derive(Debug, Clone, Copy)]
pub struct SparseTermQueryInfo {
    /// Sparse vector field
    pub field: crate::dsl::Field,
    /// Dimension ID in the sparse vector
    pub dim_id: u32,
    /// Query weight for this dimension
    pub weight: f32,
    /// Whether this term participates in candidate generation. BMP/LSP uses
    /// the pruned subset for maximum-grid traversal, then scores candidates
    /// with every term retained in this decomposition.
    pub candidate: bool,
    /// MaxScore heap factor (1.0 = exact, lower = approximate)
    pub heap_factor: f32,
    /// Multi-value combiner for ordinal deduplication
    pub combiner: super::MultiValueCombiner,
    /// Multiplier on executor limit to compensate for ordinal deduplication
    /// (1.0 = exact, 2.0 = fetch 2x then combine down)
    pub over_fetch_factor: f32,
    /// LSP/0 γ. None is depth-derived; Some(0) is exhaustive.
    pub lsp_gamma: Option<usize>,
}

/// Decomposition of a query for MaxScore optimization.
///
/// The planner inspects this to decide whether to use text MaxScore,
/// sparse MaxScore, or standard BooleanScorer execution.
#[derive(Debug, Clone)]
pub enum QueryDecomposition {
    /// Single text term — eligible for text MaxScore grouping
    TextTerm(TermQueryInfo),
    /// One or more sparse dimensions — eligible for sparse MaxScore
    SparseTerms(Vec<SparseTermQueryInfo>),
    /// Not decomposable — falls back to standard execution
    Opaque,
}

/// Matched positions for a field (field_id, list of scored positions)
/// Each position includes its individual score contribution
pub type MatchedPositions = Vec<(u32, Vec<super::ScoredPosition>)>;

macro_rules! define_query_traits {
    ($($send_bounds:tt)*) => {
        /// A search query (async)
        ///
        /// Note: `scorer` takes `&self` (not `&'a self`) so that scorers don't borrow the query.
        /// This enables query composition - queries can create sub-queries locally and get their scorers.
        /// Implementations must clone/capture any data they need during scorer creation.
        pub trait Query: std::fmt::Display + $($send_bounds)* {
            /// Create a scorer for this query against a single segment (async)
            ///
            /// The `limit` parameter specifies the maximum number of results to return.
            /// This is passed from the top-level search limit.
            ///
            /// Note: The scorer borrows only the reader, not the query. Implementations
            /// should capture any needed query data (field, terms, etc.) during creation.
            fn scorer<'a>(
                &self,
                reader: &'a SegmentReader,
                limit: usize,
            ) -> ScorerFuture<'a>;

            /// Create a scorer with collector-specific construction options.
            /// Query implementations that can avoid optional position data
            /// should override this; the default preserves existing behavior.
            fn scorer_with_options<'a>(
                &self,
                reader: &'a SegmentReader,
                limit: usize,
                options: ScorerOptions,
            ) -> ScorerFuture<'a> {
                let _ = options;
                self.scorer(reader, limit)
            }

            /// Estimated number of matching documents in a segment (async)
            fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a>;

            /// Create a scorer synchronously (mmap/RAM only).
            ///
            /// Available when the `sync` feature is enabled.
            /// Default implementation returns an error.
            #[cfg(feature = "sync")]
            fn scorer_sync<'a>(
                &self,
                reader: &'a SegmentReader,
                limit: usize,
            ) -> Result<Box<dyn Scorer + 'a>> {
                let _ = (reader, limit);
                Err(crate::error::Error::Query(
                    "sync scorer not supported for this query type".into(),
                ))
            }

            /// Synchronous counterpart to [`Query::scorer_with_options`].
            #[cfg(feature = "sync")]
            fn scorer_sync_with_options<'a>(
                &self,
                reader: &'a SegmentReader,
                limit: usize,
                options: ScorerOptions,
            ) -> Result<Box<dyn Scorer + 'a>> {
                let _ = options;
                self.scorer_sync(reader, limit)
            }

            /// Decompose this query for MaxScore optimization.
            ///
            /// Returns `TextTerm` for simple term queries, `SparseTerms` for
            /// sparse vector queries (single or multi-dim), or `Opaque` if
            /// the query cannot be decomposed.
            fn decompose(&self) -> QueryDecomposition {
                QueryDecomposition::Opaque
            }

            /// Exact scoring plan for a named L1 branch. Unsupported queries
            /// reject explicitly rather than returning truncated retrieval scores.
            fn candidate_query(&self) -> Result<super::CandidateQuery> {
                super::CandidateQuery::from_decomposition(self.decompose())
            }

            /// Append every `(field, term)` this query scores with BM25 to
            /// `out`. The searcher aggregates their document frequencies
            /// across segments before scoring (see `ScorerOptions::global_stats`).
            fn text_terms(&self, out: &mut Vec<(crate::dsl::Field, Vec<u8>)>) {
                let _ = out;
            }

            /// True if this query is a pure filter (always scores 1.0, no positions).
            /// Used by the planner to convert non-selective MUST filters into predicates.
            fn is_filter(&self) -> bool {
                false
            }

            /// For filter queries: return a cheap per-doc predicate against a segment.
            /// The predicate does O(1) work per doc (e.g., fast-field lookup).
            fn as_doc_predicate<'a>(
                &self,
                _reader: &'a SegmentReader,
            ) -> Option<DocPredicate<'a>> {
                None
            }

            /// Build a compact bitset of matching doc_ids for this query.
            ///
            /// Preferred over `as_doc_predicate` for BMP filtered queries because
            /// bitset lookup is ~2ns vs ~30-40ns for a fast-field closure.
            /// Default returns None; TermQuery overrides this to build from its
            /// posting list in O(M) time.
            fn as_doc_bitset(
                &self,
                _reader: &SegmentReader,
            ) -> Option<DocBitset> {
                None
            }

            /// Budget-aware materialization. `None` means unsupported or
            /// cancelled; cancellation must flag the shared budget, and a
            /// partial bitset must never escape as a complete filter.
            fn as_doc_bitset_with_options(&self, reader: &SegmentReader, options: &ScorerOptions) -> Option<DocBitset> {
                if options.stop_if_expired() { return None; }
                let bitset = self.as_doc_bitset(reader);
                if options.stop_if_expired() { None } else { bitset }
            }

            /// Cheap estimate of how many docs this filter clause matches in
            /// the segment. Used by the boolean planner to order MUST/MUST_NOT
            /// evaluation: the narrowest clause is materialized first and wider
            /// clauses refine it with per-doc probes instead of being fully
            /// materialized. `None` = unknown (treated as matching everything).
            fn bitset_cardinality_estimate(&self, _reader: &SegmentReader) -> Option<u64> {
                None
            }

            /// For a query that is a pure disjunction of sub-queries (a Boolean
            /// query with only SHOULD clauses and no boost), the clauses.
            ///
            /// The boolean planner flattens these into the enclosing SHOULD
            /// list: `OR(OR(a, b), c)` scores exactly like `OR(a, b, c)`, and
            /// the flat form is eligible for MaxScore and filter push-down
            /// where the nested form would be an opaque, top-k-truncated
            /// sub-scorer.
            fn should_children(&self) -> Option<&[std::sync::Arc<dyn Query>]> {
                None
            }
        }

        /// Scored document stream: a DocSet that also provides scores.
        pub trait Scorer: super::docset::DocSet + $($send_bounds)* {
            /// Score for current document
            fn score(&self) -> Score;

            /// Get matched positions for the current document (if available)
            /// Returns (field_id, positions) pairs where positions are encoded as per PositionMode
            fn matched_positions(&self) -> Option<MatchedPositions> {
                None
            }

            /// Standalone fast path for scorers that wrap an already ranked
            /// top-k list (vector executors). When this query is the top-level
            /// query of a segment search, the caller may take the ranked list
            /// directly instead of walking the DocSet and re-collecting it:
            /// the result must be exactly what a `TopKCollector` of size
            /// `limit` would produce (score desc, doc id asc, `total_seen`).
            ///
            /// Only valid before the first `advance`/`seek`. Default: `None`
            /// (the scorer must be driven).
            fn precomputed_top_k(
                &mut self,
                limit: usize,
                collect_positions: bool,
            ) -> Option<(Vec<super::SearchResult>, u32)> {
                let _ = (limit, collect_positions);
                None
            }
        }
    };
}

#[cfg(not(target_arch = "wasm32"))]
define_query_traits!(Send + Sync);

#[cfg(target_arch = "wasm32")]
define_query_traits!();

impl Query for Box<dyn Query> {
    fn as_doc_bitset_with_options(
        &self,
        reader: &SegmentReader,
        options: &ScorerOptions,
    ) -> Option<DocBitset> {
        (**self).as_doc_bitset_with_options(reader, options)
    }
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        (**self).scorer(reader, limit)
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        (**self).count_estimate(reader)
    }

    fn scorer_with_options<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
        options: ScorerOptions,
    ) -> ScorerFuture<'a> {
        (**self).scorer_with_options(reader, limit, options)
    }

    fn candidate_query(&self) -> Result<super::CandidateQuery> {
        (**self).candidate_query()
    }

    fn text_terms(&self, out: &mut Vec<(crate::dsl::Field, Vec<u8>)>) {
        (**self).text_terms(out)
    }

    fn decompose(&self) -> QueryDecomposition {
        (**self).decompose()
    }

    fn is_filter(&self) -> bool {
        (**self).is_filter()
    }

    fn as_doc_predicate<'a>(&self, reader: &'a SegmentReader) -> Option<DocPredicate<'a>> {
        (**self).as_doc_predicate(reader)
    }

    fn as_doc_bitset(&self, reader: &SegmentReader) -> Option<DocBitset> {
        (**self).as_doc_bitset(reader)
    }

    fn should_children(&self) -> Option<&[std::sync::Arc<dyn Query>]> {
        (**self).should_children()
    }

    fn bitset_cardinality_estimate(&self, reader: &SegmentReader) -> Option<u64> {
        (**self).bitset_cardinality_estimate(reader)
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
    ) -> Result<Box<dyn Scorer + 'a>> {
        (**self).scorer_sync(reader, limit)
    }

    #[cfg(feature = "sync")]
    fn scorer_sync_with_options<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
        options: ScorerOptions,
    ) -> Result<Box<dyn Scorer + 'a>> {
        (**self).scorer_sync_with_options(reader, limit, options)
    }
}

/// Empty scorer for terms that don't exist
pub struct EmptyScorer;

// A document universe is a neutral required cursor for exclusion-only Boolean
// queries. It adds neither a relevance score nor matched positions.
impl Scorer for super::AllDocSet {
    fn score(&self) -> Score {
        0.0
    }
}

impl super::docset::DocSet for EmptyScorer {
    fn doc(&self) -> DocId {
        crate::structures::TERMINATED
    }

    fn advance(&mut self) -> DocId {
        crate::structures::TERMINATED
    }

    fn seek(&mut self, _target: DocId) -> DocId {
        crate::structures::TERMINATED
    }

    fn size_hint(&self) -> u32 {
        0
    }
}

impl Scorer for EmptyScorer {
    fn score(&self) -> Score {
        0.0
    }
}
