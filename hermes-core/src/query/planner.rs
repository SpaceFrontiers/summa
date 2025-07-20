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
///
/// `shared_threshold` is a QUERY-EXECUTION-local cell: when one query has
/// multiple per-field MaxScore groups (path 2c), field A's result seeds
/// field B's pruning. It must never be shared across queries — a
/// per-segment cell here caused cross-query threshold leaks under
/// concurrent searches (one query's threshold wrongly pruning another's
/// results).
pub(super) fn finish_text_maxscore<'a>(
    posting_lists: Vec<(crate::structures::BlockPostingList, f32)>,
    avg_field_len: f32,
    limit: usize,
    shared_threshold: &std::cell::Cell<f32>,
    index_label: &str,
    field_label: &str,
) -> crate::Result<Box<dyn Scorer + 'a>> {
    if posting_lists.is_empty() {
        return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + 'a>);
    }
    let mut executor = MaxScoreExecutor::text(posting_lists, avg_field_len, limit)
        .with_metric_labels(index_label, field_label);
    let initial = shared_threshold.get();
    if initial > 0.0 {
        executor.seed_threshold(initial);
    }
    let results = executor.execute_sync()?;
    if results.len() >= limit
        && let Some(last) = results.last()
        && last.score > shared_threshold.get()
    {
        shared_threshold.set(last.score);
    }
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

    let per_field_limit = super::max_candidate_limit(limit).min(reader.num_docs() as usize);
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

const MAX_SPARSE_EXECUTOR_RESULTS: usize = 200_000;

pub(super) fn bounded_sparse_executor_limit(limit: usize, over_fetch_factor: f32) -> usize {
    let factor = if over_fetch_factor.is_finite() && over_fetch_factor >= 1.0 {
        over_fetch_factor.min(super::MAX_CANDIDATE_OVERSUBSCRIPTION as f32) as f64
    } else {
        1.0
    };
    let derived = (limit as f64 * factor).ceil();
    if !derived.is_finite() || derived >= usize::MAX as f64 {
        return MAX_SPARSE_EXECUTOR_RESULTS;
    }
    (derived as usize).min(MAX_SPARSE_EXECUTOR_RESULTS)
}

/// Physical single-value BMP segments need no ordinal over-fetch: every raw
/// hit is already a distinct final document. Genuine multi-value segments
/// retain the configured budget because aggregation can collapse many raw
/// ordinals into one document.
fn bmp_executor_limit(
    limit: usize,
    over_fetch_factor: f32,
    bmp: &crate::segment::reader::bmp::BmpIndex,
) -> usize {
    bmp_executor_limit_for_counts(
        limit,
        over_fetch_factor,
        bmp.is_single_valued(),
        bmp.num_real_docs() as usize,
        bmp.num_virtual_docs as usize,
    )
}

fn bmp_executor_limit_for_counts(
    limit: usize,
    over_fetch_factor: f32,
    single_valued: bool,
    num_real_docs: usize,
    num_virtual_docs: usize,
) -> usize {
    if single_valued {
        limit.min(num_real_docs)
    } else {
        bounded_sparse_executor_limit(limit, over_fetch_factor).min(num_virtual_docs)
    }
}

fn bmp_threshold<'a>(
    options: &'a super::ScorerOptions,
    combiner: MultiValueCombiner,
    single_valued: bool,
) -> super::bmp::BmpThreshold<'a> {
    if !single_valued && combiner != MultiValueCombiner::Max {
        return super::bmp::BmpThreshold::default();
    }
    super::bmp::BmpThreshold {
        initial: options.initial_threshold,
        shared: options.shared_threshold.as_ref(),
        publish: single_valued,
    }
}

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
    // Sparse postings contain one entry per stored ordinal, so the raw heap
    // may legitimately need to exceed the real document count before ordinal
    // scores can be combined back into documents.
    let executor_limit = bounded_sparse_executor_limit(limit, infos[0].over_fetch_factor)
        .min(si.total_vectors as usize);
    let mut executor =
        MaxScoreExecutor::sparse(si, query_terms, executor_limit, infos[0].heap_factor)
            .with_metric_labels(
                reader.schema().index_label(),
                reader.schema().get_field_name(field).unwrap_or("?"),
            );
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
    options: &super::ScorerOptions,
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
    let executor_limit = bmp_executor_limit(limit, infos[0].over_fetch_factor, bmp);
    let max_sb = infos[0].max_superblocks;
    let field_label = reader.schema().get_field_name(field).unwrap_or("?");
    match super::bmp::execute_bmp_with_threshold(
        bmp,
        reader.schema().index_label(),
        field_label,
        &query_terms,
        executor_limit,
        infos[0].heap_factor,
        max_sb,
        bmp_threshold(options, infos[0].combiner, bmp.is_single_valued()),
    ) {
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
    options: &super::ScorerOptions,
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
    let executor_limit = bmp_executor_limit(limit, infos[0].over_fetch_factor, bmp);
    let max_sb = infos[0].max_superblocks;
    let field_label = reader.schema().get_field_name(field).unwrap_or("?");
    match super::bmp::execute_bmp_filtered_with_threshold(
        bmp,
        reader.schema().index_label(),
        field_label,
        &query_terms,
        executor_limit,
        infos[0].heap_factor,
        max_sb,
        predicate,
        bmp_threshold(options, infos[0].combiner, bmp.is_single_valued()),
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

/// Refining a small accumulator beats materializing a clause's full bitset
/// when the clause is estimated to match at least this many times more docs
/// than the accumulator holds. Probes cost ~30-40ns (fast-field closure) vs
/// ~2-5ns per materialized entry, hence the margin.
const PROBE_ADVANTAGE: u64 = 8;

/// Build a combined DocBitset from MUST and MUST_NOT clause bitsets.
///
/// Selectivity-aware: MUST clauses are evaluated narrowest-first (posting
/// doc counts are exact, fast-field ranges are sampled), and once the
/// accumulator is much smaller than a remaining clause's estimate, that
/// clause is applied as O(|acc|) per-doc predicate probes instead of being
/// materialized. A `type = X` filter matching millions of docs is then never
/// iterated when a recent-dates range keeps only a few thousand candidates —
/// each surviving doc just probes the `type` fast field once.
///
/// Returns None if a clause that must be materialized doesn't support bitset
/// creation (probed clauses only need `as_doc_predicate`).
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

    // Order MUST clauses by estimated match count, narrowest first. Unknown
    // estimates sort last (pessimistically treated as matching everything).
    let mut order: Vec<(usize, u64)> = must
        .iter()
        .enumerate()
        .map(|(i, q)| {
            (
                i,
                q.bitset_cardinality_estimate(reader)
                    .unwrap_or(num_docs as u64),
            )
        })
        .collect();
    order.sort_unstable_by_key(|&(_, est)| est);

    let mut result: Option<super::DocBitset> = None;
    let mut acc_count: u64 = 0;

    for (idx, est) in order {
        let q = &must[idx];
        match result {
            None => {
                // Seed: materialize the narrowest clause.
                let bs = q.as_doc_bitset(reader)?;
                acc_count = bs.count() as u64;
                result = Some(bs);
            }
            Some(ref mut acc) => {
                let mut probed = false;
                if acc_count.saturating_mul(PROBE_ADVANTAGE) <= est
                    && let Some(pred) = q.as_doc_predicate(reader)
                {
                    acc.retain(&*pred);
                    probed = true;
                }
                if !probed {
                    let bs = q.as_doc_bitset(reader)?;
                    acc.intersect_with(&bs);
                }
                acc_count = acc.count() as u64;
                log::debug!(
                    "[planner] MUST clause {}: est={} probed={} acc={}",
                    idx,
                    est,
                    probed,
                    acc_count,
                );
            }
        }
        if acc_count == 0 {
            // AND is already empty — nothing can revive it.
            break;
        }
    }

    // Subtract MUST_NOT bitsets (probe-refined when the accumulator is small)
    for q in must_not {
        match result {
            None => {
                // No MUST clauses — start with all-ones, then subtract
                let bs = q.as_doc_bitset(reader)?;
                let mut all = super::DocBitset::new(num_docs);
                all.bits.fill(u64::MAX);
                // Clear bits beyond num_docs
                let tail_bits = num_docs as usize % 64;
                if tail_bits > 0 && !all.bits.is_empty() {
                    let last = all.bits.len() - 1;
                    all.bits[last] &= (1u64 << tail_bits) - 1;
                }
                all.subtract(&bs);
                acc_count = all.count() as u64;
                result = Some(all);
            }
            Some(ref mut acc) => {
                let est = q
                    .bitset_cardinality_estimate(reader)
                    .unwrap_or(num_docs as u64);
                let mut probed = false;
                if acc_count.saturating_mul(PROBE_ADVANTAGE) <= est
                    && let Some(pred) = q.as_doc_predicate(reader)
                {
                    acc.retain(&|doc| !pred(doc));
                    probed = true;
                }
                if !probed {
                    let bs = q.as_doc_bitset(reader)?;
                    acc.subtract(&bs);
                }
                acc_count = acc.count() as u64;
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bmp_single_value_limit_does_not_overfetch() {
        assert_eq!(bounded_sparse_executor_limit(320, 99.0), 640);
        assert_eq!(
            bmp_executor_limit_for_counts(320, 2.0, true, 10_000, 10_048),
            320
        );
        assert_eq!(
            bmp_executor_limit_for_counts(320, 2.0, false, 10_000, 10_048),
            640
        );
    }

    #[test]
    fn bmp_threshold_is_only_used_in_final_score_space() {
        let shared = super::super::SharedThreshold::new();
        shared.raise(7.0);
        let options = super::super::ScorerOptions {
            collect_positions: false,
            initial_threshold: 5.0,
            shared_threshold: Some(shared),
        };

        let single_sum = bmp_threshold(&options, MultiValueCombiner::Sum, true);
        assert_eq!(single_sum.initial, 5.0);
        assert!(single_sum.shared.is_some());
        assert!(single_sum.publish);

        let multi_max = bmp_threshold(&options, MultiValueCombiner::Max, false);
        assert!(multi_max.shared.is_some());
        assert!(!multi_max.publish);

        let multi_sum = bmp_threshold(&options, MultiValueCombiner::Sum, false);
        assert_eq!(multi_sum.initial, 0.0);
        assert!(multi_sum.shared.is_none());
        assert!(!multi_sum.publish);
    }
}
