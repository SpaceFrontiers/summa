//! Boolean query planner — shared helpers for MaxScore and filter push-down optimisation
//!
//! Extracted from `boolean.rs` to keep the planner logic separate from the
//! BooleanQuery struct, builder, and scorer types.

use std::sync::Arc;

use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::{DocId, Score};

use super::{
    DocPredicate, EmptyScorer, GlobalStats, MatchedPositions, MaxScoreExecutor, MultiValueCombiner,
    Query, ScoredDoc, ScoredPosition, Scorer, SparseTermQueryInfo, TermQueryInfo,
};

// ── IDF ──────────────────────────────────────────────────────────────────

/// Compute IDF for a posting list, preferring global stats.
pub(super) fn compute_idf(
    posting_list: &crate::structures::BlockPostingList,
    field: crate::Field,
    term: &[u8],
    num_docs: f32,
    global_stats: Option<&Arc<GlobalStats>>,
) -> f32 {
    if let Some(stats) = global_stats {
        let global_idf = stats.text_idf(field, &String::from_utf8_lossy(term));
        if global_idf > 0.0 {
            return global_idf;
        }
    }
    let doc_freq = posting_list.doc_count() as f32;
    super::bm25_idf(doc_freq, num_docs)
}

// ── Text MaxScore helpers ────────────────────────────────────────────────

/// Shared pre-check for text MaxScore: extract term infos, field, avg_field_len, num_docs.
/// Returns None if not all SHOULD clauses are single-field term queries.
pub(super) fn prepare_text_maxscore(
    should: &[Arc<dyn Query>],
    reader: &SegmentReader,
    global_stats: Option<&Arc<GlobalStats>>,
) -> Option<(Vec<TermQueryInfo>, crate::Field, f32, f32)> {
    let infos: Vec<_> = should
        .iter()
        .filter_map(|q| match q.decompose() {
            super::QueryDecomposition::TextTerm(info) => Some(info),
            _ => None,
        })
        .collect();
    if infos.len() != should.len() {
        return None;
    }
    let field = infos[0].field;
    if !infos.iter().all(|t| t.field == field) {
        return None;
    }
    let avg_field_len = global_stats
        .map(|s| s.avg_field_len(field))
        .unwrap_or_else(|| reader.avg_field_len(field));
    let num_docs = reader.num_docs() as f32;
    Some((infos, field, avg_field_len, num_docs))
}

/// Build a TopK scorer from fetched posting lists via text MaxScore.
pub(super) fn finish_text_maxscore<'a>(
    posting_lists: Vec<(crate::structures::BlockPostingList, f32)>,
    avg_field_len: f32,
    limit: usize,
) -> crate::Result<Box<dyn Scorer + 'a>> {
    if posting_lists.is_empty() {
        return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + 'a>);
    }
    let results = MaxScoreExecutor::text(posting_lists, avg_field_len, limit).execute_sync()?;
    Ok(Box::new(TopKResultScorer::new(results)) as Box<dyn Scorer + 'a>)
}

// ── Per-field grouping ───────────────────────────────────────────────────

/// Shared grouping result for per-field MaxScore.
pub(super) struct PerFieldGrouping {
    /// (field, avg_field_len, term_infos) for groups with 2+ terms
    pub multi_term_groups: Vec<(crate::Field, f32, Vec<TermQueryInfo>)>,
    /// Original indices of single-term and non-term SHOULD clauses (fallback scorers)
    pub fallback_indices: Vec<usize>,
    /// Limit per field group (over-fetched to compensate for cross-field scoring)
    pub per_field_limit: usize,
    pub num_docs: f32,
}

/// Group SHOULD clauses by field for per-field MaxScore.
/// Returns None if no group has 2+ terms (no optimization benefit).
pub(super) fn prepare_per_field_grouping(
    should: &[Arc<dyn Query>],
    reader: &SegmentReader,
    limit: usize,
    global_stats: Option<&Arc<GlobalStats>>,
) -> Option<PerFieldGrouping> {
    let mut field_groups: rustc_hash::FxHashMap<crate::Field, Vec<(usize, TermQueryInfo)>> =
        rustc_hash::FxHashMap::default();
    let mut non_term_indices: Vec<usize> = Vec::new();

    for (i, q) in should.iter().enumerate() {
        if let super::QueryDecomposition::TextTerm(info) = q.decompose() {
            field_groups.entry(info.field).or_default().push((i, info));
        } else {
            non_term_indices.push(i);
        }
    }

    if !field_groups.values().any(|g| g.len() >= 2) {
        return None;
    }

    let num_groups = field_groups.len() + non_term_indices.len();
    let per_field_limit = limit * num_groups;
    let num_docs = reader.num_docs() as f32;

    let mut multi_term_groups = Vec::new();
    let mut fallback_indices = non_term_indices;

    for group in field_groups.into_values() {
        if group.len() >= 2 {
            let field = group[0].1.field;
            let avg_field_len = global_stats
                .map(|s| s.avg_field_len(field))
                .unwrap_or_else(|| reader.avg_field_len(field));
            let infos: Vec<_> = group.into_iter().map(|(_, info)| info).collect();
            multi_term_groups.push((field, avg_field_len, infos));
        } else {
            fallback_indices.push(group[0].0);
        }
    }

    Some(PerFieldGrouping {
        multi_term_groups,
        fallback_indices,
        per_field_limit,
        num_docs,
    })
}

// ── Sparse MaxScore helpers ──────────────────────────────────────────────

/// Build a sparse MaxScoreExecutor from decomposed sparse infos.
///
/// Returns the executor + representative info (for combiner/field), or None
/// if the sparse index doesn't exist or no query dims match.
pub(crate) fn build_sparse_maxscore_executor<'a>(
    infos: &[SparseTermQueryInfo],
    reader: &'a SegmentReader,
    limit: usize,
    predicate: Option<DocPredicate<'a>>,
) -> Option<(MaxScoreExecutor<'a>, SparseTermQueryInfo)> {
    let field = infos[0].field;
    let si = reader.sparse_index(field)?;
    let query_terms: Vec<(u32, f32)> = infos
        .iter()
        .filter(|info| si.has_dimension(info.dim_id))
        .map(|info| (info.dim_id, info.weight))
        .collect();
    if query_terms.is_empty() {
        return None;
    }
    let executor_limit = (limit as f32 * infos[0].over_fetch_factor).ceil() as usize;
    let mut executor =
        MaxScoreExecutor::sparse(si, query_terms, executor_limit, infos[0].heap_factor);
    if let Some(pred) = predicate {
        executor = executor.with_predicate(pred);
    }
    Some((executor, infos[0]))
}

/// Build a sparse BMP executor from decomposed sparse infos.
///
/// Auto-detected: called when the field has a BMP index. Returns scored
/// results directly (BMP is always synchronous), or None if no BMP index
/// exists for the field.
pub(crate) fn build_sparse_bmp_results(
    infos: &[SparseTermQueryInfo],
    reader: &SegmentReader,
    limit: usize,
) -> Option<(Vec<ScoredDoc>, SparseTermQueryInfo)> {
    let field = infos[0].field;
    let bmp = reader.bmp_index(field)?;
    let query_terms: Vec<(u32, f32)> = infos
        .iter()
        .map(|info| (info.dim_id, info.weight))
        .collect();
    if query_terms.is_empty() {
        return None;
    }
    let executor_limit = (limit as f32 * infos[0].over_fetch_factor).ceil() as usize;
    match super::bmp::execute_bmp(bmp, &query_terms, executor_limit, infos[0].heap_factor, 0) {
        Ok(results) => Some((results, infos[0])),
        Err(e) => {
            log::warn!("BMP execution failed for field {}: {}", field.0, e);
            None
        }
    }
}

/// Build a sparse BMP executor with a document predicate filter.
///
/// The predicate is applied during BMP scoring (not post-filter), ensuring
/// the collector only contains valid documents and the threshold evolves correctly.
pub(crate) fn build_sparse_bmp_results_filtered(
    infos: &[SparseTermQueryInfo],
    reader: &SegmentReader,
    limit: usize,
    predicate: &dyn Fn(crate::DocId) -> bool,
) -> Option<(Vec<ScoredDoc>, SparseTermQueryInfo)> {
    let field = infos[0].field;
    let bmp = reader.bmp_index(field)?;
    let query_terms: Vec<(u32, f32)> = infos
        .iter()
        .map(|info| (info.dim_id, info.weight))
        .collect();
    if query_terms.is_empty() {
        return None;
    }
    let executor_limit = (limit as f32 * infos[0].over_fetch_factor).ceil() as usize;
    match super::bmp::execute_bmp_filtered(
        bmp,
        &query_terms,
        executor_limit,
        infos[0].heap_factor,
        0,
        predicate,
    ) {
        Ok(results) => Some((results, infos[0])),
        Err(e) => {
            log::warn!("BMP filtered execution failed for field {}: {}", field.0, e);
            None
        }
    }
}

/// Combine raw MaxScore results with ordinal deduplication into a scorer.
pub(crate) fn combine_sparse_results<'a>(
    raw: Vec<ScoredDoc>,
    combiner: MultiValueCombiner,
    field: crate::Field,
    limit: usize,
) -> Box<dyn Scorer + 'a> {
    let combined = crate::segment::combine_ordinal_results(
        raw.into_iter().map(|r| (r.doc_id, r.ordinal, r.score)),
        combiner,
        limit,
    );
    Box::new(VectorTopKResultScorer::new(combined, field.0))
}

/// Extract all sparse term infos from SHOULD clauses, flattening SparseVectorQuery.
///
/// Returns `None` if any SHOULD clause is not decomposable into sparse term queries
/// or if the resulting infos span multiple fields.
pub(super) fn extract_all_sparse_infos(
    should: &[Arc<dyn Query>],
) -> Option<Vec<SparseTermQueryInfo>> {
    let mut all = Vec::new();
    for q in should {
        match q.decompose() {
            super::QueryDecomposition::SparseTerms(infos) => all.extend(infos),
            _ => return None,
        }
    }
    if all.is_empty() {
        return None;
    }
    let field = all[0].field;
    if !all.iter().all(|i| i.field == field) {
        return None;
    }
    Some(all)
}

// ── Predicate helpers ────────────────────────────────────────────────────

/// Chain multiple predicates into a single combined predicate.
pub(super) fn chain_predicates<'a>(predicates: Vec<DocPredicate<'a>>) -> DocPredicate<'a> {
    if predicates.len() == 1 {
        return predicates.into_iter().next().unwrap();
    }
    Box::new(move |doc_id| predicates.iter().all(|p| p(doc_id)))
}

/// Build a combined DocBitset from MUST and MUST_NOT clause bitsets.
///
/// Returns None if any clause doesn't support bitset creation.
/// For term queries this is O(M) (posting list iteration); for range queries O(N).
/// The resulting bitset enables ~2ns per-doc lookups in BMP (vs ~30-40ns for closures).
pub(super) fn build_combined_bitset(
    must: &[std::sync::Arc<dyn super::Query>],
    must_not: &[std::sync::Arc<dyn super::Query>],
    reader: &crate::segment::SegmentReader,
) -> Option<super::DocBitset> {
    if must.is_empty() && must_not.is_empty() {
        return None;
    }

    let num_docs = reader.num_docs();
    let mut result: Option<super::DocBitset> = None;

    // Intersect MUST bitsets
    for q in must {
        let bs = q.as_doc_bitset(reader)?;
        match result {
            None => result = Some(bs),
            Some(ref mut acc) => acc.intersect_with(&bs),
        }
    }

    // Subtract MUST_NOT bitsets
    for q in must_not {
        let bs = q.as_doc_bitset(reader)?;
        match result {
            None => {
                // No MUST clauses — start with all-ones, then subtract
                let mut all = super::DocBitset::new(num_docs);
                for w in &mut all.bits {
                    *w = u64::MAX;
                }
                // Clear bits beyond num_docs
                let tail_bits = num_docs as usize % 64;
                if tail_bits > 0 && !all.bits.is_empty() {
                    let last = all.bits.len() - 1;
                    all.bits[last] &= (1u64 << tail_bits) - 1;
                }
                all.subtract(&bs);
                result = Some(all);
            }
            Some(ref mut acc) => acc.subtract(&bs),
        }
    }

    result
}

// ── Result scorers ───────────────────────────────────────────────────────

/// Scorer that iterates over pre-computed top-k results
pub(super) struct TopKResultScorer {
    results: Vec<ScoredDoc>,
    position: usize,
}

impl TopKResultScorer {
    pub(super) fn new(mut results: Vec<ScoredDoc>) -> Self {
        // Sort by doc_id ascending — required for DocSet seek() correctness
        results.sort_unstable_by_key(|r| r.doc_id);
        Self {
            results,
            position: 0,
        }
    }
}

impl super::docset::DocSet for TopKResultScorer {
    fn doc(&self) -> DocId {
        if self.position < self.results.len() {
            self.results[self.position].doc_id
        } else {
            TERMINATED
        }
    }

    fn advance(&mut self) -> DocId {
        self.position += 1;
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let remaining = &self.results[self.position..];
        self.position += remaining.partition_point(|r| r.doc_id < target);
        self.doc()
    }

    fn size_hint(&self) -> u32 {
        (self.results.len() - self.position) as u32
    }
}

impl Scorer for TopKResultScorer {
    fn score(&self) -> Score {
        if self.position < self.results.len() {
            self.results[self.position].score
        } else {
            0.0
        }
    }
}

/// Scorer that iterates over pre-computed vector results with ordinal information.
/// Used by sparse MaxScore path to preserve per-ordinal scores for matched_positions().
pub(crate) struct VectorTopKResultScorer {
    results: Vec<crate::segment::VectorSearchResult>,
    position: usize,
    field_id: u32,
}

impl VectorTopKResultScorer {
    pub(crate) fn new(mut results: Vec<crate::segment::VectorSearchResult>, field_id: u32) -> Self {
        results.sort_unstable_by_key(|r| r.doc_id);
        Self {
            results,
            position: 0,
            field_id,
        }
    }
}

impl super::docset::DocSet for VectorTopKResultScorer {
    fn doc(&self) -> DocId {
        if self.position < self.results.len() {
            self.results[self.position].doc_id
        } else {
            TERMINATED
        }
    }

    fn advance(&mut self) -> DocId {
        self.position += 1;
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let remaining = &self.results[self.position..];
        self.position += remaining.partition_point(|r| r.doc_id < target);
        self.doc()
    }

    fn size_hint(&self) -> u32 {
        (self.results.len() - self.position) as u32
    }
}

impl Scorer for VectorTopKResultScorer {
    fn score(&self) -> Score {
        if self.position < self.results.len() {
            self.results[self.position].score
        } else {
            0.0
        }
    }

    fn matched_positions(&self) -> Option<MatchedPositions> {
        if self.position >= self.results.len() {
            return None;
        }
        let result = &self.results[self.position];
        let scored_positions: Vec<ScoredPosition> = result
            .ordinals
            .iter()
            .map(|&(ordinal, score)| ScoredPosition::new(ordinal, score))
            .collect();
        Some(vec![(self.field_id, scored_positions)])
    }
}
