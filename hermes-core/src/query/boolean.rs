//! Boolean query with MUST, SHOULD, and MUST_NOT clauses

use std::sync::Arc;

use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::{DocId, Score};

use super::planner::{
    build_sparse_bmp_results, build_sparse_bmp_results_filtered, build_sparse_maxscore_executor,
    chain_predicates, combine_sparse_results, compute_idf, extract_all_sparse_infos,
    finish_text_maxscore, prepare_per_field_grouping, prepare_text_maxscore,
};
use super::{CountFuture, EmptyScorer, GlobalStats, Query, Scorer, ScorerFuture};

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
        let should_all: &[Arc<dyn Query>] = &$should;
        let must_not: &[Arc<dyn Query>] = &$must_not;
        let global_stats: Option<&Arc<GlobalStats>> = $global_stats;
        let reader: &SegmentReader = $reader;
        let limit: usize = $limit;

        // Cap SHOULD clauses to MAX_QUERY_TERMS, but only count queries that need
        // posting-list cursors. Fast-field predicates (O(1) per doc) are exempt.
        let should_capped: Vec<Arc<dyn Query>>;
        let should: &[Arc<dyn Query>] = if should_all.len() > super::MAX_QUERY_TERMS {
            let is_predicate: Vec<bool> = should_all
                .iter()
                .map(|q| q.is_filter() || q.as_doc_predicate(reader).is_some())
                .collect();
            let cursor_count = is_predicate.iter().filter(|&&p| !p).count();

            if cursor_count > super::MAX_QUERY_TERMS {
                let mut kept = Vec::with_capacity(should_all.len());
                let mut cursor_kept = 0usize;
                for (q, &is_pred) in should_all.iter().zip(is_predicate.iter()) {
                    if is_pred {
                        kept.push(q.clone());
                    } else if cursor_kept < super::MAX_QUERY_TERMS {
                        kept.push(q.clone());
                        cursor_kept += 1;
                    }
                }
                log::debug!(
                    "BooleanQuery: capping cursor SHOULD from {} to {} ({} fast-field predicates exempt)",
                    cursor_count,
                    super::MAX_QUERY_TERMS,
                    kept.len() - cursor_kept,
                );
                should_capped = kept;
                &should_capped
            } else {
                log::debug!(
                    "BooleanQuery: {} SHOULD clauses OK ({} need cursors, {} fast-field predicates)",
                    should_all.len(),
                    cursor_count,
                    should_all.len() - cursor_count,
                );
                should_all
            }
        } else {
            should_all
        };

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

            // 2b. Sparse (single-field, all sparse term queries)
            // Auto-detect: BMP executor if field has BMP index, else MaxScore
            if let Some(infos) = extract_all_sparse_infos(should) {
                if let Some((raw, info)) =
                    build_sparse_bmp_results(&infos, reader, limit)
                {
                    return Ok(combine_sparse_results(raw, info.combiner, info.field, limit));
                }
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
                    log::debug!("BooleanQuery planner 3a: MUST clause → predicate ({})", q);
                    predicates.push(pred);
                } else {
                    log::debug!("BooleanQuery planner 3a: MUST clause → verifier scorer ({})", q);
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

            // 3b. Fast path: pure predicates + sparse SHOULD → BMP or MaxScore w/ predicate
            if must_verifiers.is_empty()
                && must_not_verifiers.is_empty()
                && !predicates.is_empty()
            {
                if let Some(infos) = extract_all_sparse_infos(should) {
                    // Try BMP with predicate first (BMP is the default format)
                    let combined = chain_predicates(predicates);
                    if let Some((raw, info)) =
                        build_sparse_bmp_results_filtered(&infos, reader, limit, &*combined)
                    {
                        log::debug!(
                            "BooleanQuery planner: predicate-aware sparse BMP, {} dims",
                            infos.len()
                        );
                        return Ok(combine_sparse_results(raw, info.combiner, info.field, limit));
                    }
                    // Try MaxScore with predicate
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
                    // (this path only triggers if neither sparse index exists)
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
            log::debug!(
                "BooleanQuery planner: BooleanScorer fallback, size_hint={} < limit={}, \
                 {} must_v + {} must_not_v",
                should_scorer.size_hint(), limit,
                must_verifiers.len(), must_not_verifiers.len()
            );
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

    fn as_doc_predicate<'a>(&self, reader: &'a SegmentReader) -> Option<super::DocPredicate<'a>> {
        // Need at least some clauses
        if self.must.is_empty() && self.should.is_empty() {
            return None;
        }

        // Try converting all clauses to predicates; bail if any child can't
        let must_preds: Vec<_> = self
            .must
            .iter()
            .map(|q| q.as_doc_predicate(reader))
            .collect::<Option<Vec<_>>>()?;
        let should_preds: Vec<_> = self
            .should
            .iter()
            .map(|q| q.as_doc_predicate(reader))
            .collect::<Option<Vec<_>>>()?;
        let must_not_preds: Vec<_> = self
            .must_not
            .iter()
            .map(|q| q.as_doc_predicate(reader))
            .collect::<Option<Vec<_>>>()?;

        let has_must = !must_preds.is_empty();

        Some(Box::new(move |doc_id| {
            // All MUST predicates must pass
            if !must_preds.iter().all(|p| p(doc_id)) {
                return false;
            }
            // When there are no MUST clauses, at least one SHOULD must pass
            if !has_must && !should_preds.is_empty() && !should_preds.iter().any(|p| p(doc_id)) {
                return false;
            }
            // No MUST_NOT predicate should pass
            must_not_preds.iter().all(|p| !p(doc_id))
        }))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::Field;
    use crate::query::{QueryDecomposition, TermQuery};

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
                .all(|q| matches!(q.decompose(), QueryDecomposition::TextTerm(_)))
        );

        // All should be same field
        let infos: Vec<_> = query
            .should
            .iter()
            .filter_map(|q| match q.decompose() {
                QueryDecomposition::TextTerm(info) => Some(info),
                _ => None,
            })
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
            .filter_map(|q| match q.decompose() {
                QueryDecomposition::TextTerm(info) => Some(info),
                _ => None,
            })
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
        match term_query.decompose() {
            QueryDecomposition::TextTerm(info) => {
                assert_eq!(info.field, Field(42));
                assert_eq!(info.term, b"test");
            }
            _ => panic!("Expected TextTerm decomposition"),
        }
    }

    #[test]
    fn test_boolean_query_no_term_info() {
        // BooleanQuery itself should not return term info
        let query = BooleanQuery::new().should(TermQuery::text(Field(0), "hello"));

        assert!(matches!(query.decompose(), QueryDecomposition::Opaque));
    }
}
