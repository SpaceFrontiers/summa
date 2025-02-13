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

/// Try text MaxScore for pure OR queries (async).
async fn try_maxscore_scorer<'a>(
    should: &[Arc<dyn Query>],
    reader: &'a SegmentReader,
    limit: usize,
    global_stats: Option<&Arc<GlobalStats>>,
) -> crate::Result<Option<Box<dyn Scorer + 'a>>> {
    let (mut infos, _field, avg_field_len, num_docs) =
        match prepare_text_maxscore(should, reader, global_stats) {
            Some(v) => v,
            None => return Ok(None),
        };
    let mut posting_lists = Vec::with_capacity(infos.len());
    for info in infos.drain(..) {
        if let Some(pl) = reader.get_postings(info.field, &info.term).await? {
            let idf = compute_idf(&pl, info.field, &info.term, num_docs, global_stats);
            posting_lists.push((pl, idf));
        }
    }
    Ok(Some(finish_text_maxscore(
        posting_lists,
        avg_field_len,
        limit,
    )?))
}

/// Try text MaxScore for pure OR queries (sync).
#[cfg(feature = "sync")]
fn try_maxscore_scorer_sync<'a>(
    should: &[Arc<dyn Query>],
    reader: &'a SegmentReader,
    limit: usize,
    global_stats: Option<&Arc<GlobalStats>>,
) -> crate::Result<Option<Box<dyn Scorer + 'a>>> {
    let (mut infos, _field, avg_field_len, num_docs) =
        match prepare_text_maxscore(should, reader, global_stats) {
            Some(v) => v,
            None => return Ok(None),
        };
    let mut posting_lists = Vec::with_capacity(infos.len());
    for info in infos.drain(..) {
        if let Some(pl) = reader.get_postings_sync(info.field, &info.term)? {
            let idf = compute_idf(&pl, info.field, &info.term, num_docs, global_stats);
            posting_lists.push((pl, idf));
        }
    }
    Ok(Some(finish_text_maxscore(
        posting_lists,
        avg_field_len,
        limit,
    )?))
}

/// Try to build a sparse MaxScoreExecutor from SHOULD clauses.
/// Returns None if not eligible, Some(Err) for empty segment, Some(Ok) otherwise.
fn prepare_sparse_maxscore<'a>(
    should: &[Arc<dyn Query>],
    reader: &'a SegmentReader,
    limit: usize,
) -> Option<Result<MaxScoreExecutor<'a>, Box<dyn Scorer + 'a>>> {
    let infos: Vec<SparseTermQueryInfo> = should
        .iter()
        .filter_map(|q| q.as_sparse_term_query_info())
        .collect();
    if infos.len() != should.len() {
        return None;
    }
    let field = infos[0].field;
    if !infos.iter().all(|t| t.field == field) {
        return None;
    }
    let si = match reader.sparse_index(field) {
        Some(si) => si,
        None => return Some(Err(Box::new(EmptyScorer))),
    };
    let query_terms: Vec<(u32, f32)> = infos
        .iter()
        .filter(|info| si.has_dimension(info.dim_id))
        .map(|info| (info.dim_id, info.weight))
        .collect();
    if query_terms.is_empty() {
        return Some(Err(Box::new(EmptyScorer)));
    }
    let executor_limit = (limit as f32 * infos[0].over_fetch_factor).ceil() as usize;
    Some(Ok(MaxScoreExecutor::sparse(
        si,
        query_terms,
        executor_limit,
        infos[0].heap_factor,
    )))
}

/// Combine raw MaxScore results with ordinal deduplication into a scorer.
fn combine_sparse_results<'a>(
    raw: Vec<ScoredDoc>,
    combiner: super::MultiValueCombiner,
    limit: usize,
) -> Box<dyn Scorer + 'a> {
    let combined = crate::segment::combine_ordinal_results(
        raw.into_iter().map(|r| (r.doc_id, r.ordinal, r.score)),
        combiner,
        limit,
    );
    let scored: Vec<ScoredDoc> = combined
        .into_iter()
        .map(|r| ScoredDoc {
            doc_id: r.doc_id,
            score: r.score,
            ordinal: 0,
        })
        .collect();
    Box::new(TopKResultScorer::new(scored))
}

/// Build MaxScore scorer from sparse term infos (async).
async fn try_sparse_maxscore_scorer<'a>(
    should: &[Arc<dyn Query>],
    reader: &'a SegmentReader,
    limit: usize,
) -> crate::Result<Option<Box<dyn Scorer + 'a>>> {
    let executor = match prepare_sparse_maxscore(should, reader, limit) {
        None => return Ok(None),
        Some(Err(empty)) => return Ok(Some(empty)),
        Some(Ok(e)) => e,
    };
    let combiner = should[0].as_sparse_term_query_info().unwrap().combiner;
    let raw = executor.execute().await?;
    Ok(Some(combine_sparse_results(raw, combiner, limit)))
}

/// Build MaxScore scorer from sparse term infos (sync).
#[cfg(feature = "sync")]
fn try_sparse_maxscore_scorer_sync<'a>(
    should: &[Arc<dyn Query>],
    reader: &'a SegmentReader,
    limit: usize,
) -> crate::Result<Option<Box<dyn Scorer + 'a>>> {
    let executor = match prepare_sparse_maxscore(should, reader, limit) {
        None => return Ok(None),
        Some(Err(empty)) => return Ok(Some(empty)),
        Some(Ok(e)) => e,
    };
    let combiner = should[0].as_sparse_term_query_info().unwrap().combiner;
    let raw = executor.execute_sync()?;
    Ok(Some(combine_sparse_results(raw, combiner, limit)))
}

impl Query for BooleanQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        // Clone Arc vectors - cheap reference counting
        let must = self.must.clone();
        let should = self.should.clone();
        let must_not = self.must_not.clone();
        let global_stats = self.global_stats.clone();

        Box::pin(async move {
            // Single-clause optimization: unwrap to inner scorer directly
            if must_not.is_empty() {
                if must.len() == 1 && should.is_empty() {
                    return must[0].scorer(reader, limit).await;
                }
                if should.len() == 1 && must.is_empty() {
                    return should[0].scorer(reader, limit).await;
                }
            }

            // Check if this is a pure OR query eligible for MaxScore optimization
            // Conditions: no MUST, no MUST_NOT, multiple SHOULD clauses, all same field
            if must.is_empty() && must_not.is_empty() && should.len() >= 2 {
                // Try text MaxScore first
                if let Some(scorer) =
                    try_maxscore_scorer(&should, reader, limit, global_stats.as_ref()).await?
                {
                    return Ok(scorer);
                }
                // Try sparse MaxScore
                if let Some(scorer) = try_sparse_maxscore_scorer(&should, reader, limit).await? {
                    return Ok(scorer);
                }
            }

            // Fall back to standard boolean scoring
            let mut must_scorers = Vec::with_capacity(must.len());
            for q in &must {
                must_scorers.push(q.scorer(reader, limit).await?);
            }

            let mut should_scorers = Vec::with_capacity(should.len());
            for q in &should {
                should_scorers.push(q.scorer(reader, limit).await?);
            }

            let mut must_not_scorers = Vec::with_capacity(must_not.len());
            for q in &must_not {
                must_not_scorers.push(q.scorer(reader, limit).await?);
            }

            let mut scorer = BooleanScorer {
                must: must_scorers,
                should: should_scorers,
                must_not: must_not_scorers,
                current_doc: 0,
            };
            // Initialize to first match
            scorer.current_doc = scorer.find_next_match();
            Ok(Box::new(scorer) as Box<dyn Scorer + 'a>)
        })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        // Single-clause optimization: unwrap to inner scorer directly
        if self.must_not.is_empty() {
            if self.must.len() == 1 && self.should.is_empty() {
                return self.must[0].scorer_sync(reader, limit);
            }
            if self.should.len() == 1 && self.must.is_empty() {
                return self.should[0].scorer_sync(reader, limit);
            }
        }

        // MaxScore optimization for pure OR queries
        if self.must.is_empty() && self.must_not.is_empty() && self.should.len() >= 2 {
            if let Some(scorer) =
                try_maxscore_scorer_sync(&self.should, reader, limit, self.global_stats.as_ref())?
            {
                return Ok(scorer);
            }
            if let Some(scorer) = try_sparse_maxscore_scorer_sync(&self.should, reader, limit)? {
                return Ok(scorer);
            }
        }

        // Fall back to standard boolean scoring
        let mut must_scorers = Vec::with_capacity(self.must.len());
        for q in &self.must {
            must_scorers.push(q.scorer_sync(reader, limit)?);
        }

        let mut should_scorers = Vec::with_capacity(self.should.len());
        for q in &self.should {
            should_scorers.push(q.scorer_sync(reader, limit)?);
        }

        let mut must_not_scorers = Vec::with_capacity(self.must_not.len());
        for q in &self.must_not {
            must_not_scorers.push(q.scorer_sync(reader, limit)?);
        }

        let mut scorer = BooleanScorer {
            must: must_scorers,
            should: should_scorers,
            must_not: must_not_scorers,
            current_doc: 0,
        };
        scorer.current_doc = scorer.find_next_match();
        Ok(Box::new(scorer) as Box<dyn Scorer + 'a>)
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
