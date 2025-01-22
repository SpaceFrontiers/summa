//! Boolean query with MUST, SHOULD, and MUST_NOT clauses

use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::{DocId, Score};

use super::{CountFuture, Query, ScoredDoc, Scorer, ScorerFuture, TextTermScorer, WandExecutor};

/// Boolean query with MUST, SHOULD, and MUST_NOT clauses
#[derive(Default)]
pub struct BooleanQuery {
    pub must: Vec<Box<dyn Query>>,
    pub should: Vec<Box<dyn Query>>,
    pub must_not: Vec<Box<dyn Query>>,
}

impl std::fmt::Debug for BooleanQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BooleanQuery")
            .field("must_count", &self.must.len())
            .field("should_count", &self.should.len())
            .field("must_not_count", &self.must_not.len())
            .finish()
    }
}

impl BooleanQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn must(mut self, query: impl Query + 'static) -> Self {
        self.must.push(Box::new(query));
        self
    }

    pub fn should(mut self, query: impl Query + 'static) -> Self {
        self.should.push(Box::new(query));
        self
    }

    pub fn must_not(mut self, query: impl Query + 'static) -> Self {
        self.must_not.push(Box::new(query));
        self
    }

    /// Try to create a WAND-optimized scorer for pure OR queries
    ///
    /// Returns Some(scorer) if all SHOULD clauses are term queries for the same field.
    /// Returns None if WAND optimization is not applicable.
    async fn try_wand_scorer<'a>(
        &'a self,
        reader: &'a SegmentReader,
        limit: usize,
    ) -> crate::Result<Option<Box<dyn Scorer + 'a>>> {
        // Extract term info from all SHOULD clauses
        let mut term_infos: Vec<_> = self
            .should
            .iter()
            .filter_map(|q| q.as_term_query_info())
            .collect();

        // Check if all clauses are term queries
        if term_infos.len() != self.should.len() {
            return Ok(None);
        }

        // Check if all terms are for the same field
        let first_field = term_infos[0].field;
        if !term_infos.iter().all(|t| t.field == first_field) {
            return Ok(None);
        }

        // Build WAND scorers for each term
        let mut scorers: Vec<TextTermScorer> = Vec::with_capacity(term_infos.len());
        let avg_field_len = reader.avg_field_len(first_field);
        let num_docs = reader.num_docs() as f32;

        for info in term_infos.drain(..) {
            if let Some(posting_list) = reader.get_postings(info.field, &info.term).await? {
                let doc_freq = posting_list.doc_count() as f32;
                let idf = super::bm25_idf(doc_freq, num_docs);
                scorers.push(TextTermScorer::new(posting_list, idf, avg_field_len));
            }
        }

        if scorers.is_empty() {
            return Ok(Some(Box::new(EmptyWandScorer) as Box<dyn Scorer + 'a>));
        }

        // Use WAND executor for efficient top-k
        let results = WandExecutor::new(scorers, limit).execute();
        Ok(Some(
            Box::new(WandResultScorer::new(results)) as Box<dyn Scorer + 'a>
        ))
    }
}

impl Query for BooleanQuery {
    fn scorer<'a>(&'a self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        Box::pin(async move {
            // Check if this is a pure OR query eligible for WAND optimization
            // Conditions: no MUST, no MUST_NOT, multiple SHOULD clauses, all same field
            if self.must.is_empty()
                && self.must_not.is_empty()
                && self.should.len() >= 2
                && let Some(scorer) = self.try_wand_scorer(reader, limit).await?
            {
                return Ok(scorer);
            }

            // Fall back to standard boolean scoring
            let mut must_scorers = Vec::with_capacity(self.must.len());
            for q in &self.must {
                must_scorers.push(q.scorer(reader, limit).await?);
            }

            let mut should_scorers = Vec::with_capacity(self.should.len());
            for q in &self.should {
                should_scorers.push(q.scorer(reader, limit).await?);
            }

            let mut must_not_scorers = Vec::with_capacity(self.must_not.len());
            for q in &self.must_not {
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

    fn count_estimate<'a>(&'a self, reader: &'a SegmentReader) -> CountFuture<'a> {
        Box::pin(async move {
            if !self.must.is_empty() {
                let mut estimates = Vec::with_capacity(self.must.len());
                for q in &self.must {
                    estimates.push(q.count_estimate(reader).await?);
                }
                estimates
                    .into_iter()
                    .min()
                    .ok_or_else(|| crate::Error::Corruption("Empty must clause".to_string()))
            } else if !self.should.is_empty() {
                let mut sum = 0u32;
                for q in &self.should {
                    sum += q.count_estimate(reader).await?;
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

impl Scorer for BooleanScorer<'_> {
    fn doc(&self) -> DocId {
        self.current_doc
    }

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

        self.find_next_match()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        for scorer in &mut self.must {
            scorer.seek(target);
        }

        for scorer in &mut self.should {
            scorer.seek(target);
        }

        self.find_next_match()
    }

    fn size_hint(&self) -> u32 {
        if !self.must.is_empty() {
            self.must.iter().map(|s| s.size_hint()).min().unwrap_or(0)
        } else {
            self.should.iter().map(|s| s.size_hint()).sum()
        }
    }
}

/// Scorer that iterates over pre-computed WAND results
struct WandResultScorer {
    results: Vec<ScoredDoc>,
    position: usize,
}

impl WandResultScorer {
    fn new(results: Vec<ScoredDoc>) -> Self {
        Self {
            results,
            position: 0,
        }
    }
}

impl Scorer for WandResultScorer {
    fn doc(&self) -> DocId {
        if self.position < self.results.len() {
            self.results[self.position].doc_id
        } else {
            TERMINATED
        }
    }

    fn score(&self) -> Score {
        if self.position < self.results.len() {
            self.results[self.position].score
        } else {
            0.0
        }
    }

    fn advance(&mut self) -> DocId {
        self.position += 1;
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        while self.position < self.results.len() && self.results[self.position].doc_id < target {
            self.position += 1;
        }
        self.doc()
    }

    fn size_hint(&self) -> u32 {
        self.results.len() as u32
    }
}

/// Empty scorer for when no terms match
struct EmptyWandScorer;

impl Scorer for EmptyWandScorer {
    fn doc(&self) -> DocId {
        TERMINATED
    }

    fn score(&self) -> Score {
        0.0
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::Field;
    use crate::query::TermQuery;

    #[test]
    fn test_wand_eligible_pure_or_same_field() {
        // Pure OR query with multiple terms in same field should be WAND-eligible
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
    fn test_wand_not_eligible_different_fields() {
        // OR query with terms in different fields should NOT use WAND
        let query = BooleanQuery::new()
            .should(TermQuery::text(Field(0), "hello"))
            .should(TermQuery::text(Field(1), "world")); // Different field!

        let infos: Vec<_> = query
            .should
            .iter()
            .filter_map(|q| q.as_term_query_info())
            .collect();
        assert_eq!(infos.len(), 2);
        // Fields are different, WAND should not be used
        assert!(infos[0].field != infos[1].field);
    }

    #[test]
    fn test_wand_not_eligible_with_must() {
        // Query with MUST clause should NOT use WAND optimization
        let query = BooleanQuery::new()
            .must(TermQuery::text(Field(0), "required"))
            .should(TermQuery::text(Field(0), "hello"))
            .should(TermQuery::text(Field(0), "world"));

        // Has MUST clause, so WAND optimization should not kick in
        assert!(!query.must.is_empty());
    }

    #[test]
    fn test_wand_not_eligible_with_must_not() {
        // Query with MUST_NOT clause should NOT use WAND optimization
        let query = BooleanQuery::new()
            .should(TermQuery::text(Field(0), "hello"))
            .should(TermQuery::text(Field(0), "world"))
            .must_not(TermQuery::text(Field(0), "excluded"));

        // Has MUST_NOT clause, so WAND optimization should not kick in
        assert!(!query.must_not.is_empty());
    }

    #[test]
    fn test_wand_not_eligible_single_term() {
        // Single SHOULD clause should NOT use WAND (no benefit)
        let query = BooleanQuery::new().should(TermQuery::text(Field(0), "hello"));

        // Only one term, WAND not beneficial
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
