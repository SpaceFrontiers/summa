//! WAND-optimized OR query for efficient multi-term full-text search
//!
//! Uses MaxScore WAND algorithm for efficient top-k retrieval when
//! searching for documents matching any of multiple terms.

use std::sync::Arc;

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::{DocId, Score};

use super::{
    CountFuture, GlobalStats, Query, ScoredDoc, Scorer, ScorerFuture, TextTermScorer, WandExecutor,
};

/// WAND-optimized OR query for multiple terms
///
/// More efficient than `BooleanQuery` with SHOULD clauses for top-k retrieval
/// because it uses MaxScore pruning to skip low-scoring documents.
///
/// # Example
/// ```ignore
/// let query = WandOrQuery::new(field)
///     .term("hello")
///     .term("world");
/// let results = index.search(&query, 10).await?;
/// ```
#[derive(Clone)]
pub struct WandOrQuery {
    /// Field to search
    pub field: Field,
    /// Terms to search for (OR semantics)
    pub terms: Vec<String>,
    /// Optional global statistics for cross-segment IDF
    global_stats: Option<Arc<GlobalStats>>,
}

impl std::fmt::Debug for WandOrQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WandOrQuery")
            .field("field", &self.field)
            .field("terms", &self.terms)
            .field("has_global_stats", &self.global_stats.is_some())
            .finish()
    }
}

impl WandOrQuery {
    /// Create a new WAND OR query for a field
    pub fn new(field: Field) -> Self {
        Self {
            field,
            terms: Vec::new(),
            global_stats: None,
        }
    }

    /// Add a term to the OR query
    pub fn term(mut self, term: impl Into<String>) -> Self {
        self.terms.push(term.into().to_lowercase());
        self
    }

    /// Add multiple terms
    pub fn terms(mut self, terms: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for t in terms {
            self.terms.push(t.into().to_lowercase());
        }
        self
    }

    /// Set global statistics for cross-segment IDF
    pub fn with_global_stats(mut self, stats: Arc<GlobalStats>) -> Self {
        self.global_stats = Some(stats);
        self
    }
}

impl Query for WandOrQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        let field = self.field;
        let terms = self.terms.clone();
        let global_stats = self.global_stats.clone();

        Box::pin(async move {
            let mut scorers: Vec<TextTermScorer> = Vec::with_capacity(terms.len());

            // Get avg field length (from global stats or segment)
            let avg_field_len = global_stats
                .as_ref()
                .map(|s| s.avg_field_len(field))
                .unwrap_or_else(|| reader.avg_field_len(field));

            let num_docs = reader.num_docs() as f32;

            for term in &terms {
                let term_bytes = term.as_bytes();

                if let Some(posting_list) = reader.get_postings(field, term_bytes).await? {
                    // Compute IDF
                    let doc_freq = posting_list.doc_count() as f32;
                    let idf = if let Some(ref stats) = global_stats {
                        let global_idf = stats.text_idf(field, term);
                        if global_idf > 0.0 {
                            global_idf
                        } else {
                            super::bm25_idf(doc_freq, num_docs)
                        }
                    } else {
                        super::bm25_idf(doc_freq, num_docs)
                    };

                    scorers.push(TextTermScorer::new(posting_list, idf, avg_field_len));
                }
            }

            if scorers.is_empty() {
                return Ok(Box::new(EmptyWandScorer) as Box<dyn Scorer + 'a>);
            }

            // Use WAND executor for efficient top-k
            let results = WandExecutor::new(scorers, limit).execute();

            Ok(Box::new(WandResultScorer::new(results)) as Box<dyn Scorer + 'a>)
        })
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let field = self.field;
        let terms = self.terms.clone();

        Box::pin(async move {
            let mut sum = 0u32;
            for term in &terms {
                if let Some(posting_list) = reader.get_postings(field, term.as_bytes()).await? {
                    sum += posting_list.doc_count();
                }
            }
            Ok(sum)
        })
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
            crate::structures::TERMINATED
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
        crate::structures::TERMINATED
    }

    fn score(&self) -> Score {
        0.0
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wand_or_query_builder() {
        let query = WandOrQuery::new(Field(0))
            .term("hello")
            .term("world")
            .terms(vec!["foo", "bar"]);

        assert_eq!(query.terms.len(), 4);
        assert_eq!(query.terms[0], "hello");
        assert_eq!(query.terms[1], "world");
        assert_eq!(query.terms[2], "foo");
        assert_eq!(query.terms[3], "bar");
    }
}
