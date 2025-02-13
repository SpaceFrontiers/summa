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

impl BoostQuery {
    pub fn new(query: impl Query + 'static, boost: f32) -> Self {
        Self {
            inner: Arc::new(query),
            boost,
        }
    }
}

impl Query for BoostQuery {
    fn scorer<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
        predicate: Option<super::DocPredicate<'a>>,
    ) -> ScorerFuture<'a> {
        let inner = self.inner.clone();
        let boost = self.boost;
        Box::pin(async move {
            let inner_scorer = inner.scorer(reader, limit, predicate).await?;
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
        predicate: Option<super::DocPredicate<'a>>,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        let inner_scorer = self.inner.scorer_sync(reader, limit, predicate)?;
        Ok(Box::new(BoostScorer {
            inner: inner_scorer,
            boost: self.boost,
        }) as Box<dyn Scorer + 'a>)
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let inner = self.inner.clone();
        Box::pin(async move { inner.count_estimate(reader).await })
    }
}

struct BoostScorer<'a> {
    inner: Box<dyn Scorer + 'a>,
    boost: f32,
}

impl Scorer for BoostScorer<'_> {
    fn doc(&self) -> DocId {
        self.inner.doc()
    }

    fn score(&self) -> Score {
        self.inner.score() * self.boost
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
