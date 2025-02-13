//! Boolean query with MUST, SHOULD, and MUST_NOT clauses

use std::sync::Arc;

use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::{DocId, Score};

use super::{
    CountFuture, GlobalStats, MaxScoreExecutor, Query, ScoredDoc, Scorer, ScorerFuture,
    SparseTermQueryInfo,
};

/// Boolean query with MUST, SHOULD, and MUST_NOT clauses
///
/// When all clauses are SHOULD term queries on the same field, automatically
/// uses MaxScore optimization for efficient top-k retrieval.
#[derive(Default, Clone)]
pub struct BooleanQuery {
    pub must: Vec<Arc<dyn Query>>,
    pub should: Vec<Arc<dyn Query>>,
    pub must_not: Vec<Arc<dyn Query>>,
    /// Optional global statistics for cross-segment IDF
    global_stats: Option<Arc<GlobalStats>>,
}

impl std::fmt::Debug for BooleanQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BooleanQuery")
            .field("must_count", &self.must.len())
            .field("should_count", &self.should.len())
            .field("must_not_count", &self.must_not.len())
            .field("has_global_stats", &self.global_stats.is_some())
            .finish()
    }
}

impl std::fmt::Display for BooleanQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Boolean(")?;
        let mut first = true;
        for q in &self.must {
            if !first {
                write!(f, " ")?;
            }
            write!(f, "+{}", q)?;
            first = false;
        }
        for q in &self.should {
            if !first {
                write!(f, " ")?;
            }
            write!(f, "{}", q)?;
            first = false;
        }
        for q in &self.must_not {
            if !first {
                write!(f, " ")?;
            }
            write!(f, "-{}", q)?;
            first = false;
        }
        write!(f, ")")
    }
}

impl BooleanQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn must(mut self, query: impl Query + 'static) -> Self {
        self.must.push(Arc::new(query));
        self
    }

    pub fn should(mut self, query: impl Query + 'static) -> Self {
        self.should.push(Arc::new(query));
        self
    }

    pub fn must_not(mut self, query: impl Query + 'static) -> Self {
        self.must_not.push(Arc::new(query));
        self
    }

    /// Set global statistics for cross-segment IDF
    pub fn with_global_stats(mut self, stats: Arc<GlobalStats>) -> Self {
        self.global_stats = Some(stats);
        self
    }
}

/// Compute IDF for a posting list, preferring global stats.
fn compute_idf(
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

/// Shared pre-check for text MaxScore: extract term infos, field, avg_field_len, num_docs.
/// Returns None if not all SHOULD clauses are single-field term queries.
fn prepare_text_maxscore(
    should: &[Arc<dyn Query>],
    reader: &SegmentReader,
    global_stats: Option<&Arc<GlobalStats>>,
) -> Option<(Vec<super::TermQueryInfo>, crate::Field, f32, f32)> {
    let infos: Vec<_> = should
        .iter()
        .filter_map(|q| q.as_term_query_info())
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
fn finish_text_maxscore<'a>(
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

/// Shared grouping result for per-field MaxScore.
struct PerFieldGrouping {
    /// (field, avg_field_len, term_infos) for groups with 2+ terms
    multi_term_groups: Vec<(crate::Field, f32, Vec<super::TermQueryInfo>)>,
    /// Original indices of single-term and non-term SHOULD clauses (fallback scorers)
    fallback_indices: Vec<usize>,
    /// Limit per field group (over-fetched to compensate for cross-field scoring)
    per_field_limit: usize,
    num_docs: f32,
}

/// Group SHOULD clauses by field for per-field MaxScore.
/// Returns None if no group has 2+ terms (no optimization benefit).
fn prepare_per_field_grouping(
    should: &[Arc<dyn Query>],
    reader: &SegmentReader,
    limit: usize,
    global_stats: Option<&Arc<GlobalStats>>,
) -> Option<PerFieldGrouping> {
    let mut field_groups: rustc_hash::FxHashMap<crate::Field, Vec<(usize, super::TermQueryInfo)>> =
        rustc_hash::FxHashMap::default();
    let mut non_term_indices: Vec<usize> = Vec::new();

    for (i, q) in should.iter().enumerate() {
        if let Some(info) = q.as_term_query_info() {
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

/// Build a SHOULD-only scorer from a vec of optimized scorers.
fn build_should_scorer<'a>(scorers: Vec<Box<dyn Scorer + 'a>>) -> Box<dyn Scorer + 'a> {
    if scorers.is_empty() {
        return Box::new(EmptyScorer);
    }
    if scorers.len() == 1 {
        return scorers.into_iter().next().unwrap();
    }
    let mut scorer = BooleanScorer {
        must: vec![],
        should: scorers,
        must_not: vec![],
        current_doc: 0,
    };
    scorer.current_doc = scorer.find_next_match();
    Box::new(scorer)
}

/// Build a sparse MaxScoreExecutor from decomposed sparse infos.
///
/// Returns the executor + representative info (for combiner/field), or None
/// if the sparse index doesn't exist or no query dims match.
fn build_sparse_maxscore_executor<'a>(
    infos: &[SparseTermQueryInfo],
    reader: &'a SegmentReader,
    limit: usize,
    predicate: Option<super::DocPredicate<'a>>,
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

/// Combine raw MaxScore results with ordinal deduplication into a scorer.
fn combine_sparse_results<'a>(
    raw: Vec<ScoredDoc>,
    combiner: super::MultiValueCombiner,
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
fn extract_all_sparse_infos(should: &[Arc<dyn Query>]) -> Option<Vec<SparseTermQueryInfo>> {
    let mut all = Vec::new();
    for q in should {
        if let Some(info) = q.as_sparse_term_query_info() {
            all.push(info);
        } else if let Some(infos) = q.as_sparse_term_queries() {
            all.extend(infos);
        } else {
            return None;
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

/// Chain multiple predicates into a single combined predicate.
fn chain_predicates<'a>(predicates: Vec<super::DocPredicate<'a>>) -> super::DocPredicate<'a> {
    if predicates.len() == 1 {
        return predicates.into_iter().next().unwrap();
    }
    Box::new(move |doc_id| predicates.iter().all(|p| p(doc_id)))
}

// ── Planner macro ────────────────────────────────────────────────────────
//
// Unified planner for both async and sync paths.  Parameterised on:
//   $scorer_fn      – scorer | scorer_sync
//   $get_postings_fn – get_postings | get_postings_sync
//   $execute_fn     – execute | execute_sync
//   $($aw)*         – .await  (present for async, absent for sync)
//
// Decision order:
//   1. Single-clause unwrap
//   2. Pure OR → text MaxScore | sparse MaxScore | per-field MaxScore
//   3. Filter push-down → predicate-aware sparse MaxScore | PredicatedScorer
//   4. Standard BooleanScorer fallback
macro_rules! boolean_plan {
    ($must:expr, $should:expr, $must_not:expr, $global_stats:expr,
     $reader:expr, $limit:expr,
     $scorer_fn:ident, $get_postings_fn:ident, $execute_fn:ident
     $(, $aw:tt)*) => {{
        let must: &[Arc<dyn Query>] = &$must;
        let should: &[Arc<dyn Query>] = &$should;
        let must_not: &[Arc<dyn Query>] = &$must_not;
        let global_stats: Option<&Arc<GlobalStats>> = $global_stats;
        let reader: &SegmentReader = $reader;
        let limit: usize = $limit;

        // ── 1. Single-clause optimisation ────────────────────────────────
        if must_not.is_empty() {
            if must.len() == 1 && should.is_empty() {
                return must[0].$scorer_fn(reader, limit) $(.  $aw)* ;
            }
            if should.len() == 1 && must.is_empty() {
                return should[0].$scorer_fn(reader, limit) $(. $aw)* ;
            }
        }

        // ── 2. Pure OR → MaxScore optimisations ──────────────────────────
        if must.is_empty() && must_not.is_empty() && should.len() >= 2 {
            // 2a. Text MaxScore (single-field, all term queries)
            if let Some((mut infos, _field, avg_field_len, num_docs)) =
                prepare_text_maxscore(should, reader, global_stats)
            {
                let mut posting_lists = Vec::with_capacity(infos.len());
                for info in infos.drain(..) {
                    if let Some(pl) = reader.$get_postings_fn(info.field, &info.term)
                        $(. $aw)* ?
                    {
                        let idf = compute_idf(&pl, info.field, &info.term, num_docs, global_stats);
                        posting_lists.push((pl, idf));
                    }
                }
                return finish_text_maxscore(posting_lists, avg_field_len, limit);
            }

            // 2b. Sparse MaxScore (single-field, all sparse term queries)
            if let Some(infos) = extract_all_sparse_infos(should) {
                if let Some((executor, info)) =
                    build_sparse_maxscore_executor(&infos, reader, limit, None)
                {
                    let raw = executor.$execute_fn() $(. $aw)* ?;
                    return Ok(combine_sparse_results(raw, info.combiner, info.field, limit));
                }
            }

            // 2c. Per-field text MaxScore (multi-field term grouping)
            if let Some(grouping) = prepare_per_field_grouping(should, reader, limit, global_stats)
            {
                let mut scorers: Vec<Box<dyn Scorer + '_>> = Vec::new();
                for (field, avg_field_len, infos) in &grouping.multi_term_groups {
                    let mut posting_lists = Vec::with_capacity(infos.len());
                    for info in infos {
                        if let Some(pl) = reader.$get_postings_fn(info.field, &info.term)
                            $(. $aw)* ?
                        {
                            let idf = compute_idf(
                                &pl, *field, &info.term, grouping.num_docs, global_stats,
                            );
                            posting_lists.push((pl, idf));
                        }
                    }
                    if !posting_lists.is_empty() {
                        scorers.push(finish_text_maxscore(
                            posting_lists,
                            *avg_field_len,
                            grouping.per_field_limit,
                        )?);
                    }
                }
                for &idx in &grouping.fallback_indices {
                    scorers.push(should[idx].$scorer_fn(reader, limit) $(. $aw)* ?);
                }
                return Ok(build_should_scorer(scorers));
            }
        }

        // ── 3. Filter push-down (MUST + SHOULD) ─────────────────────────
        if !should.is_empty() && !must.is_empty() && limit < usize::MAX / 4 {
            // 3a. Compile MUST → predicates (O(1)) vs verifier scorers (seek)
            let mut predicates: Vec<super::DocPredicate<'_>> = Vec::new();
            let mut must_verifiers: Vec<Box<dyn super::Scorer + '_>> = Vec::new();
            for q in must {
                if let Some(pred) = q.as_doc_predicate(reader) {
                    predicates.push(pred);
                } else {
                    must_verifiers.push(q.$scorer_fn(reader, limit) $(. $aw)* ?);
                }
            }
            // Compile MUST_NOT → negated predicates vs verifier scorers
            let mut must_not_verifiers: Vec<Box<dyn super::Scorer + '_>> = Vec::new();
            for q in must_not {
                if let Some(pred) = q.as_doc_predicate(reader) {
                    let negated: super::DocPredicate<'_> =
                        Box::new(move |doc_id| !pred(doc_id));
                    predicates.push(negated);
                } else {
                    must_not_verifiers.push(q.$scorer_fn(reader, limit) $(. $aw)* ?);
                }
            }

            // 3b. Fast path: pure predicates + sparse SHOULD → MaxScore w/ predicate
            if must_verifiers.is_empty()
                && must_not_verifiers.is_empty()
                && !predicates.is_empty()
            {
                if let Some(infos) = extract_all_sparse_infos(should) {
                    let combined = chain_predicates(predicates);
                    if let Some((executor, info)) =
                        build_sparse_maxscore_executor(&infos, reader, limit, Some(combined))
                    {
                        log::debug!(
                            "BooleanQuery planner: predicate-aware sparse MaxScore, {} dims",
                            infos.len()
                        );
                        let raw = executor.$execute_fn() $(. $aw)* ?;
                        return Ok(combine_sparse_results(raw, info.combiner, info.field, limit));
                    }
                    // predicates consumed — cannot fall through; rebuild them
                    // (this path only triggers if sparse index is absent)
                    predicates = Vec::new();
                    for q in must {
                        if let Some(pred) = q.as_doc_predicate(reader) {
                            predicates.push(pred);
                        }
                    }
                    for q in must_not {
                        if let Some(pred) = q.as_doc_predicate(reader) {
                            let negated: super::DocPredicate<'_> =
                                Box::new(move |doc_id| !pred(doc_id));
                            predicates.push(negated);
                        }
                    }
                }
            }

            // 3c. PredicatedScorer fallback (over-fetch 4x when predicates present)
            let should_limit = if !predicates.is_empty() { limit * 4 } else { limit };
            let should_scorer = if should.len() == 1 {
                should[0].$scorer_fn(reader, should_limit) $(. $aw)* ?
            } else {
                let sub = BooleanQuery {
                    must: Vec::new(),
                    should: should.to_vec(),
                    must_not: Vec::new(),
                    global_stats: global_stats.cloned(),
                };
                sub.$scorer_fn(reader, should_limit) $(. $aw)* ?
            };

            let use_predicated =
                must_verifiers.is_empty() || should_scorer.size_hint() >= limit as u32;

            if use_predicated {
                log::debug!(
                    "BooleanQuery planner: PredicatedScorer {} preds + {} must_v + {} must_not_v, \
                     SHOULD size_hint={}, over_fetch={}",
                    predicates.len(), must_verifiers.len(), must_not_verifiers.len(),
                    should_scorer.size_hint(), should_limit
                );
                return Ok(Box::new(super::PredicatedScorer::new(
                    should_scorer, predicates, must_verifiers, must_not_verifiers,
                )));
            }

            // size_hint < limit with verifiers → BooleanScorer
            let mut scorer = BooleanScorer {
                must: must_verifiers,
                should: vec![should_scorer],
                must_not: must_not_verifiers,
                current_doc: 0,
            };
            scorer.current_doc = scorer.find_next_match();
            return Ok(Box::new(scorer));
        }

        // ── 4. Standard BooleanScorer fallback ───────────────────────────
        let mut must_scorers = Vec::with_capacity(must.len());
        for q in must {
            must_scorers.push(q.$scorer_fn(reader, limit) $(. $aw)* ?);
        }
        let mut should_scorers = Vec::with_capacity(should.len());
        for q in should {
            should_scorers.push(q.$scorer_fn(reader, limit) $(. $aw)* ?);
        }
        let mut must_not_scorers = Vec::with_capacity(must_not.len());
        for q in must_not {
            must_not_scorers.push(q.$scorer_fn(reader, limit) $(. $aw)* ?);
        }
        let mut scorer = BooleanScorer {
            must: must_scorers,
            should: should_scorers,
            must_not: must_not_scorers,
            current_doc: 0,
        };
        scorer.current_doc = scorer.find_next_match();
        Ok(Box::new(scorer) as Box<dyn Scorer + '_>)
    }};
}

impl Query for BooleanQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        let must = self.must.clone();
        let should = self.should.clone();
        let must_not = self.must_not.clone();
        let global_stats = self.global_stats.clone();
        Box::pin(async move {
            boolean_plan!(
                must,
                should,
                must_not,
                global_stats.as_ref(),
                reader,
                limit,
                scorer,
                get_postings,
                execute,
                await
            )
        })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        boolean_plan!(
            self.must,
            self.should,
            self.must_not,
            self.global_stats.as_ref(),
            reader,
            limit,
            scorer_sync,
            get_postings_sync,
            execute_sync
        )
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let must = self.must.clone();
        let should = self.should.clone();

        Box::pin(async move {
            if !must.is_empty() {
                let mut estimates = Vec::with_capacity(must.len());
                for q in &must {
                    estimates.push(q.count_estimate(reader).await?);
                }
                estimates
                    .into_iter()
                    .min()
                    .ok_or_else(|| crate::Error::Corruption("Empty must clause".to_string()))
            } else if !should.is_empty() {
                let mut sum = 0u32;
                for q in &should {
                    sum = sum.saturating_add(q.count_estimate(reader).await?);
                }
                Ok(sum)
            } else {
                Ok(0)
            }
        })
    }
}

struct BooleanScorer<'a> {
    must: Vec<Box<dyn Scorer + 'a>>,
    should: Vec<Box<dyn Scorer + 'a>>,
    must_not: Vec<Box<dyn Scorer + 'a>>,
    current_doc: DocId,
}

impl BooleanScorer<'_> {
    fn find_next_match(&mut self) -> DocId {
        if self.must.is_empty() && self.should.is_empty() {
            return TERMINATED;
        }

        loop {
            let candidate = if !self.must.is_empty() {
                let mut max_doc = self
                    .must
                    .iter()
                    .map(|s| s.doc())
                    .max()
                    .unwrap_or(TERMINATED);

                if max_doc == TERMINATED {
                    return TERMINATED;
                }

                loop {
                    let mut all_match = true;
                    for scorer in &mut self.must {
                        let doc = scorer.seek(max_doc);
                        if doc == TERMINATED {
                            return TERMINATED;
                        }
                        if doc > max_doc {
                            max_doc = doc;
                            all_match = false;
                            break;
                        }
                    }
                    if all_match {
                        break;
                    }
                }
                max_doc
            } else {
                self.should
                    .iter()
                    .map(|s| s.doc())
                    .filter(|&d| d != TERMINATED)
                    .min()
                    .unwrap_or(TERMINATED)
            };

            if candidate == TERMINATED {
                return TERMINATED;
            }

            let excluded = self.must_not.iter_mut().any(|scorer| {
                let doc = scorer.seek(candidate);
                doc == candidate
            });

            if !excluded {
                // Seek SHOULD scorers to candidate so score() can see their contributions
                for scorer in &mut self.should {
                    scorer.seek(candidate);
                }
                self.current_doc = candidate;
                return candidate;
            }

            // Advance past excluded candidate
            if !self.must.is_empty() {
                for scorer in &mut self.must {
                    scorer.advance();
                }
            } else {
                // For SHOULD-only: seek all scorers past the excluded candidate
                for scorer in &mut self.should {
                    if scorer.doc() <= candidate && scorer.doc() != TERMINATED {
                        scorer.seek(candidate + 1);
                    }
                }
            }
        }
    }
}

impl super::docset::DocSet for BooleanScorer<'_> {
    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn advance(&mut self) -> DocId {
        if !self.must.is_empty() {
            for scorer in &mut self.must {
                scorer.advance();
            }
        } else {
            for scorer in &mut self.should {
                if scorer.doc() == self.current_doc {
                    scorer.advance();
                }
            }
        }

        self.current_doc = self.find_next_match();
        self.current_doc
    }

    fn seek(&mut self, target: DocId) -> DocId {
        for scorer in &mut self.must {
            scorer.seek(target);
        }

        for scorer in &mut self.should {
            scorer.seek(target);
        }

        self.current_doc = self.find_next_match();
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        if !self.must.is_empty() {
            self.must.iter().map(|s| s.size_hint()).min().unwrap_or(0)
        } else {
            self.should.iter().map(|s| s.size_hint()).sum()
        }
    }
}

impl Scorer for BooleanScorer<'_> {
    fn score(&self) -> Score {
        let mut total = 0.0;

        for scorer in &self.must {
            if scorer.doc() == self.current_doc {
                total += scorer.score();
            }
        }

        for scorer in &self.should {
            if scorer.doc() == self.current_doc {
                total += scorer.score();
            }
        }

        total
    }

    fn matched_positions(&self) -> Option<super::MatchedPositions> {
        let mut all_positions: super::MatchedPositions = Vec::new();

        for scorer in &self.must {
            if scorer.doc() == self.current_doc
                && let Some(positions) = scorer.matched_positions()
            {
                all_positions.extend(positions);
            }
        }

        for scorer in &self.should {
            if scorer.doc() == self.current_doc
                && let Some(positions) = scorer.matched_positions()
            {
                all_positions.extend(positions);
            }
        }

        if all_positions.is_empty() {
            None
        } else {
            Some(all_positions)
        }
    }
}

/// Scorer that iterates over pre-computed top-k results
struct TopKResultScorer {
    results: Vec<ScoredDoc>,
    position: usize,
}

impl TopKResultScorer {
    fn new(mut results: Vec<ScoredDoc>) -> Self {
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
struct VectorTopKResultScorer {
    results: Vec<crate::segment::VectorSearchResult>,
    position: usize,
    field_id: u32,
}

impl VectorTopKResultScorer {
    fn new(mut results: Vec<crate::segment::VectorSearchResult>, field_id: u32) -> Self {
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

    fn matched_positions(&self) -> Option<super::MatchedPositions> {
        if self.position >= self.results.len() {
            return None;
        }
        let result = &self.results[self.position];
        let scored_positions: Vec<super::ScoredPosition> = result
            .ordinals
            .iter()
            .map(|&(ordinal, score)| super::ScoredPosition::new(ordinal, score))
            .collect();
        Some(vec![(self.field_id, scored_positions)])
    }
}

/// Empty scorer for when no terms match
struct EmptyScorer;

impl super::docset::DocSet for EmptyScorer {
    fn doc(&self) -> DocId {
        TERMINATED
    }

    fn advance(&mut self) -> DocId {
        TERMINATED
    }

    fn seek(&mut self, _target: DocId) -> DocId {
        TERMINATED
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::Field;
    use crate::query::TermQuery;

    #[test]
    fn test_maxscore_eligible_pure_or_same_field() {
        // Pure OR query with multiple terms in same field should be MaxScore-eligible
        let query = BooleanQuery::new()
            .should(TermQuery::text(Field(0), "hello"))
            .should(TermQuery::text(Field(0), "world"))
            .should(TermQuery::text(Field(0), "foo"));

        // All clauses should return term info
        assert!(
            query
                .should
                .iter()
                .all(|q| q.as_term_query_info().is_some())
        );

        // All should be same field
        let infos: Vec<_> = query
            .should
            .iter()
            .filter_map(|q| q.as_term_query_info())
            .collect();
        assert_eq!(infos.len(), 3);
        assert!(infos.iter().all(|i| i.field == Field(0)));
    }

    #[test]
    fn test_maxscore_not_eligible_different_fields() {
        // OR query with terms in different fields should NOT use MaxScore
        let query = BooleanQuery::new()
            .should(TermQuery::text(Field(0), "hello"))
            .should(TermQuery::text(Field(1), "world")); // Different field!

        let infos: Vec<_> = query
            .should
            .iter()
            .filter_map(|q| q.as_term_query_info())
            .collect();
        assert_eq!(infos.len(), 2);
        // Fields are different, MaxScore should not be used
        assert!(infos[0].field != infos[1].field);
    }

    #[test]
    fn test_maxscore_not_eligible_with_must() {
        // Query with MUST clause should NOT use MaxScore optimization
        let query = BooleanQuery::new()
            .must(TermQuery::text(Field(0), "required"))
            .should(TermQuery::text(Field(0), "hello"))
            .should(TermQuery::text(Field(0), "world"));

        // Has MUST clause, so MaxScore optimization should not kick in
        assert!(!query.must.is_empty());
    }

    #[test]
    fn test_maxscore_not_eligible_with_must_not() {
        // Query with MUST_NOT clause should NOT use MaxScore optimization
        let query = BooleanQuery::new()
            .should(TermQuery::text(Field(0), "hello"))
            .should(TermQuery::text(Field(0), "world"))
            .must_not(TermQuery::text(Field(0), "excluded"));

        // Has MUST_NOT clause, so MaxScore optimization should not kick in
        assert!(!query.must_not.is_empty());
    }

    #[test]
    fn test_maxscore_not_eligible_single_term() {
        // Single SHOULD clause should NOT use MaxScore (no benefit)
        let query = BooleanQuery::new().should(TermQuery::text(Field(0), "hello"));

        // Only one term, MaxScore not beneficial
        assert_eq!(query.should.len(), 1);
    }

    #[test]
    fn test_term_query_info_extraction() {
        let term_query = TermQuery::text(Field(42), "test");
        let info = term_query.as_term_query_info();

        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.field, Field(42));
        assert_eq!(info.term, b"test");
    }

    #[test]
    fn test_boolean_query_no_term_info() {
        // BooleanQuery itself should not return term info
        let query = BooleanQuery::new().should(TermQuery::text(Field(0), "hello"));

        assert!(query.as_term_query_info().is_none());
    }
}
