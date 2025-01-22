//! Term query - matches documents containing a specific term

use std::sync::Arc;

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::structures::BlockPostingList;
use crate::{DocId, Score};

use super::{
    Bm25Params, CountFuture, EmptyScorer, GlobalStats, Query, Scorer, ScorerFuture, TermQueryInfo,
};

/// Term query - matches documents containing a specific term
#[derive(Clone)]
pub struct TermQuery {
    pub field: Field,
    pub term: Vec<u8>,
    /// Optional global statistics for cross-segment IDF
    global_stats: Option<Arc<GlobalStats>>,
}

impl std::fmt::Debug for TermQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TermQuery")
            .field("field", &self.field)
            .field("term", &String::from_utf8_lossy(&self.term))
            .field("has_global_stats", &self.global_stats.is_some())
            .finish()
    }
}

impl TermQuery {
    pub fn new(field: Field, term: impl Into<Vec<u8>>) -> Self {
        Self {
            field,
            term: term.into(),
            global_stats: None,
        }
    }

    pub fn text(field: Field, text: &str) -> Self {
        Self {
            field,
            term: text.to_lowercase().into_bytes(),
            global_stats: None,
        }
    }

    /// Create with global statistics for cross-segment IDF
    pub fn with_global_stats(field: Field, text: &str, stats: Arc<GlobalStats>) -> Self {
        Self {
            field,
            term: text.to_lowercase().into_bytes(),
            global_stats: Some(stats),
        }
    }

    /// Set global statistics for cross-segment IDF
    pub fn set_global_stats(&mut self, stats: Arc<GlobalStats>) {
        self.global_stats = Some(stats);
    }
}

impl Query for TermQuery {
    fn scorer<'a>(&'a self, reader: &'a SegmentReader, _limit: usize) -> ScorerFuture<'a> {
        Box::pin(async move {
            let postings = reader.get_postings(self.field, &self.term).await?;

            match postings {
                Some(posting_list) => {
                    // Use global stats IDF if available, otherwise segment-local
                    let (idf, avg_field_len) = if let Some(ref stats) = self.global_stats {
                        let term_str = String::from_utf8_lossy(&self.term);
                        let global_idf = stats.text_idf(self.field, &term_str);

                        // If global stats has this term, use global IDF
                        // Otherwise fall back to segment-local
                        if global_idf > 0.0 {
                            (global_idf, stats.avg_field_len(self.field))
                        } else {
                            // Fall back to segment-local IDF
                            let num_docs = reader.num_docs() as f32;
                            let doc_freq = posting_list.doc_count() as f32;
                            let idf = super::bm25_idf(doc_freq, num_docs);
                            (idf, reader.avg_field_len(self.field))
                        }
                    } else {
                        // Compute IDF from segment statistics
                        let num_docs = reader.num_docs() as f32;
                        let doc_freq = posting_list.doc_count() as f32;
                        let idf = super::bm25_idf(doc_freq, num_docs);
                        (idf, reader.avg_field_len(self.field))
                    };

                    Ok(Box::new(TermScorer::new(
                        posting_list,
                        idf,
                        avg_field_len,
                        Bm25Params::default(),
                        1.0, // default field boost
                    )) as Box<dyn Scorer + 'a>)
                }
                None => Ok(Box::new(EmptyScorer) as Box<dyn Scorer + 'a>),
            }
        })
    }

    fn count_estimate<'a>(&'a self, reader: &'a SegmentReader) -> CountFuture<'a> {
        Box::pin(async move {
            match reader.get_postings(self.field, &self.term).await? {
                Some(list) => Ok(list.doc_count()),
                None => Ok(0),
            }
        })
    }

    fn as_term_query_info(&self) -> Option<TermQueryInfo> {
        Some(TermQueryInfo {
            field: self.field,
            term: self.term.clone(),
        })
    }
}

struct TermScorer {
    iterator: crate::structures::BlockPostingIterator<'static>,
    idf: f32,
    /// BM25 parameters
    params: Bm25Params,
    /// Average field length for this field
    avg_field_len: f32,
    /// Field boost/weight for BM25F
    field_boost: f32,
}

impl TermScorer {
    pub fn new(
        posting_list: BlockPostingList,
        idf: f32,
        avg_field_len: f32,
        params: Bm25Params,
        field_boost: f32,
    ) -> Self {
        Self {
            iterator: posting_list.into_iterator(),
            idf,
            params,
            avg_field_len,
            field_boost,
        }
    }
}

impl Scorer for TermScorer {
    fn doc(&self) -> DocId {
        self.iterator.doc()
    }

    fn score(&self) -> Score {
        let tf = self.iterator.term_freq() as f32;
        let k1 = self.params.k1;
        let b = self.params.b;

        // BM25F: apply field boost and length normalization
        let length_norm = 1.0 - b + b * (tf / self.avg_field_len.max(1.0));
        let tf_norm =
            (tf * self.field_boost * (k1 + 1.0)) / (tf * self.field_boost + k1 * length_norm);

        self.idf * tf_norm
    }

    fn advance(&mut self) -> DocId {
        self.iterator.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.iterator.seek(target)
    }

    fn size_hint(&self) -> u32 {
        0
    }
}
