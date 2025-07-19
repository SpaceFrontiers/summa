//! Boost query - multiplies the score of the inner query

use std::sync::Arc;

use crate::segment::SegmentReader;
use crate::{DocId, Score};

use super::{CountFuture, Query, Scorer, ScorerFuture};

/// Boost query - multiplies the score of the inner query
#[derive(Clone)]
pub struct BoostQuery {
    pub inner: Arc<dyn Query>,
    pub boost: f32,
}

impl std::fmt::Debug for BoostQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoostQuery")
            .field("boost", &self.boost)
            .finish()
    }
}

impl std::fmt::Display for BoostQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}^{}", self.inner, self.boost)
    }
}

impl BoostQuery {
    pub fn new(query: impl Query + 'static, boost: f32) -> Self {
        Self {
            inner: Arc::new(query),
            boost,
        }
    }
}

impl Query for BoostQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        self.scorer_with_options(reader, limit, super::ScorerOptions::with_positions())
    }

    fn scorer_with_options<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
        options: super::ScorerOptions,
    ) -> ScorerFuture<'a> {
        let inner = self.inner.clone();
        let boost = self.boost;
        Box::pin(async move {
            if !boost.is_finite() {
                return Err(crate::Error::Query(
                    "boost must be a finite number".to_string(),
                ));
            }
            let inner_options = if boost == 1.0 {
                options
            } else {
                options.without_threshold()
            };
            let inner_scorer = inner
                .scorer_with_options(reader, limit, inner_options)
                .await?;
            Ok(Box::new(BoostScorer {
                inner: inner_scorer,
                boost,
            }) as Box<dyn Scorer + 'a>)
        })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        self.scorer_sync_with_options(reader, limit, super::ScorerOptions::with_positions())
    }

    #[cfg(feature = "sync")]
    fn scorer_sync_with_options<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
        options: super::ScorerOptions,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        if !self.boost.is_finite() {
            return Err(crate::Error::Query(
                "boost must be a finite number".to_string(),
            ));
        }
        let inner_options = if self.boost == 1.0 {
            options
        } else {
            options.without_threshold()
        };
        let inner_scorer = self
            .inner
            .scorer_sync_with_options(reader, limit, inner_options)?;
        Ok(Box::new(BoostScorer {
            inner: inner_scorer,
            boost: self.boost,
        }) as Box<dyn Scorer + 'a>)
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let inner = self.inner.clone();
        Box::pin(async move { inner.count_estimate(reader).await })
    }

    fn is_filter(&self) -> bool {
        self.boost == 1.0 && self.inner.is_filter()
    }

    fn as_doc_predicate<'a>(&self, reader: &'a SegmentReader) -> Option<super::DocPredicate<'a>> {
        (self.boost == 1.0)
            .then(|| self.inner.as_doc_predicate(reader))
            .flatten()
    }

    fn decompose(&self) -> super::QueryDecomposition {
        if self.boost == 1.0 {
            self.inner.decompose()
        } else {
            super::QueryDecomposition::Opaque
        }
    }
}

struct BoostScorer<'a> {
    inner: Box<dyn Scorer + 'a>,
    boost: f32,
}

impl super::docset::DocSet for BoostScorer<'_> {
    fn doc(&self) -> DocId {
        self.inner.doc()
    }

    fn advance(&mut self) -> DocId {
        self.inner.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.inner.seek(target)
    }

    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }
}

impl Scorer for BoostScorer<'_> {
    fn score(&self) -> Score {
        self.inner.score() * self.boost
    }

    fn matched_positions(&self) -> Option<super::MatchedPositions> {
        let mut positions = self.inner.matched_positions()?;
        for (_, scored_positions) in &mut positions {
            for position in scored_positions {
                position.score *= self.boost;
            }
        }
        Some(positions)
    }
}
