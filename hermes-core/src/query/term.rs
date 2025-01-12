//! Term query - matches documents containing a specific term

use std::sync::Arc;

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::structures::BlockPostingList;
use crate::wand::WandData;
use crate::{DocId, Score};

use super::{Bm25Params, CountFuture, EmptyScorer, Query, Scorer, ScorerFuture};

/// Term query - matches documents containing a specific term
#[derive(Debug, Clone)]
pub struct TermQuery {
    pub field: Field,
    pub term: Vec<u8>,
    /// Optional pre-computed WAND data for collection-wide IDF
    wand_data: Option<Arc<WandData>>,
    /// Field name for WAND data lookup
    field_name: Option<String>,
}

impl TermQuery {
    pub fn new(field: Field, term: impl Into<Vec<u8>>) -> Self {
        Self {
            field,
            term: term.into(),
            wand_data: None,
            field_name: None,
        }
    }

    pub fn text(field: Field, text: &str) -> Self {
        Self {
            field,
            term: text.to_lowercase().into_bytes(),
            wand_data: None,
            field_name: None,
        }
    }

    /// Create a term query with pre-computed WAND data for collection-wide IDF
    ///
    /// This enables more accurate scoring when querying across multiple segments,
    /// as the IDF values are computed from the entire collection rather than
    /// per-segment.
    pub fn with_wand_data(
        field: Field,
        field_name: &str,
        term: &str,
        wand_data: Arc<WandData>,
    ) -> Self {
        Self {
            field,
            term: term.to_lowercase().into_bytes(),
            wand_data: Some(wand_data),
            field_name: Some(field_name.to_string()),
        }
    }

    /// Set WAND data for this query
    pub fn set_wand_data(&mut self, field_name: &str, wand_data: Arc<WandData>) {
        self.wand_data = Some(wand_data);
        self.field_name = Some(field_name.to_string());
    }
}

impl Query for TermQuery {
    fn scorer<'a>(&'a self, reader: &'a SegmentReader) -> ScorerFuture<'a> {
        Box::pin(async move {
            let postings = reader.get_postings(self.field, &self.term).await?;

            match postings {
                Some(posting_list) => {
                    // Try to get IDF from pre-computed WAND data first
                    let idf = if let (Some(wand_data), Some(field_name)) =
                        (&self.wand_data, &self.field_name)
                    {
                        let term_str = String::from_utf8_lossy(&self.term);
                        wand_data.get_idf(field_name, &term_str).unwrap_or_else(|| {
                            // Fall back to segment-level IDF if term not in WAND data
                            let num_docs = reader.num_docs() as f32;
                            let doc_freq = posting_list.doc_count() as f32;
                            ((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln()
                        })
                    } else {
                        // Compute IDF from segment statistics
                        let num_docs = reader.num_docs() as f32;
                        let doc_freq = posting_list.doc_count() as f32;
                        ((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln()
                    };

                    // Get average field length for BM25F length normalization
                    // Use WAND data avg_doc_len if available, otherwise segment-level
                    let avg_field_len = self
                        .wand_data
                        .as_ref()
                        .map(|w| w.avg_doc_len)
                        .unwrap_or_else(|| reader.avg_field_len(self.field));

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
