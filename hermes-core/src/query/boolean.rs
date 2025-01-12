//! Boolean query with MUST, SHOULD, and MUST_NOT clauses

use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::{DocId, Score};

use super::{CountFuture, Query, Scorer, ScorerFuture};

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
}

impl Query for BooleanQuery {
    fn scorer<'a>(&'a self, reader: &'a SegmentReader) -> ScorerFuture<'a> {
        Box::pin(async move {
            let mut must_scorers = Vec::with_capacity(self.must.len());
            for q in &self.must {
                must_scorers.push(q.scorer(reader).await?);
            }

            let mut should_scorers = Vec::with_capacity(self.should.len());
            for q in &self.should {
                should_scorers.push(q.scorer(reader).await?);
            }

            let mut must_not_scorers = Vec::with_capacity(self.must_not.len());
            for q in &self.must_not {
                must_not_scorers.push(q.scorer(reader).await?);
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
