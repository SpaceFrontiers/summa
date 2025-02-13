//! Term query - matches documents containing a specific term

use std::sync::Arc;

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::structures::BlockPostingList;
use crate::structures::TERMINATED;
use crate::{DocId, Score};

use super::{CountFuture, EmptyScorer, GlobalStats, Query, Scorer, ScorerFuture, TermQueryInfo};

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

impl std::fmt::Display for TermQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Term({}:\"{}\")",
            self.field.0,
            String::from_utf8_lossy(&self.term)
        )
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
    fn scorer<'a>(&self, reader: &'a SegmentReader, _limit: usize) -> ScorerFuture<'a> {
        let field = self.field;
        let term = self.term.clone();
        let global_stats = self.global_stats.clone();
        Box::pin(async move {
            let postings = reader.get_postings(field, &term).await?;

            match postings {
                Some(posting_list) => {
                    // Use global stats IDF if available, otherwise segment-local
                    let (idf, avg_field_len) = if let Some(ref stats) = global_stats {
                        let term_str = String::from_utf8_lossy(&term);
                        let global_idf = stats.text_idf(field, &term_str);

                        // If global stats has this term, use global IDF
                        // Otherwise fall back to segment-local
                        if global_idf > 0.0 {
                            (global_idf, stats.avg_field_len(field))
                        } else {
                            // Fall back to segment-local IDF
                            let num_docs = reader.num_docs() as f32;
                            let doc_freq = posting_list.doc_count() as f32;
                            let idf = super::bm25_idf(doc_freq, num_docs);
                            (idf, reader.avg_field_len(field))
                        }
                    } else {
                        // Compute IDF from segment statistics
                        let num_docs = reader.num_docs() as f32;
                        let doc_freq = posting_list.doc_count() as f32;
                        let idf = super::bm25_idf(doc_freq, num_docs);
                        (idf, reader.avg_field_len(field))
                    };

                    // Try to load positions if available
                    let positions = reader.get_positions(field, &term).await.ok().flatten();

                    let mut scorer = TermScorer::new(
                        posting_list,
                        idf,
                        avg_field_len,
                        1.0, // default field boost
                    );

                    if let Some(pos) = positions {
                        scorer = scorer.with_positions(field.0, pos);
                    }

                    Ok(Box::new(scorer) as Box<dyn Scorer + 'a>)
                }
                None => {
                    // Fall back to fast field scanning for fast-only text fields
                    let term_str = String::from_utf8_lossy(&term);
                    if let Some(scorer) = FastFieldTextScorer::try_new(reader, field, &term_str) {
                        Ok(Box::new(scorer) as Box<dyn Scorer + 'a>)
                    } else {
                        Ok(Box::new(EmptyScorer) as Box<dyn Scorer + 'a>)
                    }
                }
            }
        })
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let field = self.field;
        let term = self.term.clone();
        Box::pin(async move {
            match reader.get_postings(field, &term).await? {
                Some(list) => Ok(list.doc_count()),
                None => Ok(0),
            }
        })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        _limit: usize,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        let postings = reader.get_postings_sync(self.field, &self.term)?;

        match postings {
            Some(posting_list) => {
                let (idf, avg_field_len) = if let Some(ref stats) = self.global_stats {
                    let term_str = String::from_utf8_lossy(&self.term);
                    let global_idf = stats.text_idf(self.field, &term_str);
                    if global_idf > 0.0 {
                        (global_idf, stats.avg_field_len(self.field))
                    } else {
                        let num_docs = reader.num_docs() as f32;
                        let doc_freq = posting_list.doc_count() as f32;
                        (
                            super::bm25_idf(doc_freq, num_docs),
                            reader.avg_field_len(self.field),
                        )
                    }
                } else {
                    let num_docs = reader.num_docs() as f32;
                    let doc_freq = posting_list.doc_count() as f32;
                    (
                        super::bm25_idf(doc_freq, num_docs),
                        reader.avg_field_len(self.field),
                    )
                };

                let positions = reader
                    .get_positions_sync(self.field, &self.term)
                    .ok()
                    .flatten();

                let mut scorer = TermScorer::new(posting_list, idf, avg_field_len, 1.0);
                if let Some(pos) = positions {
                    scorer = scorer.with_positions(self.field.0, pos);
                }

                Ok(Box::new(scorer) as Box<dyn Scorer + 'a>)
            }
            None => {
                let term_str = String::from_utf8_lossy(&self.term);
                if let Some(scorer) = FastFieldTextScorer::try_new(reader, self.field, &term_str) {
                    Ok(Box::new(scorer) as Box<dyn Scorer + 'a>)
                } else {
                    Ok(Box::new(EmptyScorer) as Box<dyn Scorer + 'a>)
                }
            }
        }
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
    /// Average field length for this field
    avg_field_len: f32,
    /// Field boost/weight for BM25F
    field_boost: f32,
    /// Field ID for position reporting
    field_id: u32,
    /// Position posting list (if positions are enabled)
    positions: Option<crate::structures::PositionPostingList>,
}

impl TermScorer {
    pub fn new(
        posting_list: BlockPostingList,
        idf: f32,
        avg_field_len: f32,
        field_boost: f32,
    ) -> Self {
        Self {
            iterator: posting_list.into_iterator(),
            idf,
            avg_field_len,
            field_boost,
            field_id: 0,
            positions: None,
        }
    }

    pub fn with_positions(
        mut self,
        field_id: u32,
        positions: crate::structures::PositionPostingList,
    ) -> Self {
        self.field_id = field_id;
        self.positions = Some(positions);
        self
    }
}

impl super::docset::DocSet for TermScorer {
    fn doc(&self) -> DocId {
        self.iterator.doc()
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

// ── Fast field text equality scorer ──────────────────────────────────────

/// Scorer that scans a text fast field for exact string equality.
/// Used as fallback when a TermQuery targets a fast-only text field (no inverted index).
/// Returns score 1.0 for matching docs (filter-style, like RangeScorer).
struct FastFieldTextScorer<'a> {
    fast_field: &'a crate::structures::fast_field::FastFieldReader,
    target_ordinal: u64,
    current: u32,
    num_docs: u32,
}

impl<'a> FastFieldTextScorer<'a> {
    fn try_new(reader: &'a SegmentReader, field: Field, text: &str) -> Option<Self> {
        let fast_field = reader.fast_field(field.0)?;
        let target_ordinal = fast_field.text_ordinal(text)?;
        let num_docs = reader.num_docs();
        let mut scorer = Self {
            fast_field,
            target_ordinal,
            current: 0,
            num_docs,
        };
        // Position on first matching doc
        if num_docs > 0 && fast_field.get_u64(0) != target_ordinal {
            scorer.scan_forward();
        }
        Some(scorer)
    }

    fn scan_forward(&mut self) {
        loop {
            self.current += 1;
            if self.current >= self.num_docs {
                self.current = self.num_docs;
                return;
            }
            if self.fast_field.get_u64(self.current) == self.target_ordinal {
                return;
            }
        }
    }
}

impl super::docset::DocSet for FastFieldTextScorer<'_> {
    fn doc(&self) -> DocId {
        if self.current >= self.num_docs {
            TERMINATED
        } else {
            self.current
        }
    }

    fn advance(&mut self) -> DocId {
        self.scan_forward();
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if target > self.current {
            self.current = target;
            if self.current < self.num_docs
                && self.fast_field.get_u64(self.current) != self.target_ordinal
            {
                self.scan_forward();
            }
        }
        self.doc()
    }

    fn size_hint(&self) -> u32 {
        0
    }
}

impl Scorer for FastFieldTextScorer<'_> {
    fn score(&self) -> Score {
        1.0
    }
}

impl Scorer for TermScorer {
    fn score(&self) -> Score {
        let tf = self.iterator.term_freq() as f32;
        // Note: Using tf as doc_len proxy since we don't store per-doc field lengths.
        // This is a common approximation - longer docs tend to have higher TF.
        super::bm25f_score(tf, self.idf, tf, self.avg_field_len, self.field_boost)
    }

    fn matched_positions(&self) -> Option<super::MatchedPositions> {
        let positions = self.positions.as_ref()?;
        let doc_id = self.iterator.doc();
        let pos = positions.get_positions(doc_id)?;
        let score = self.score();
        // Each position contributes equally to the term score
        let per_position_score = if pos.is_empty() {
            0.0
        } else {
            score / pos.len() as f32
        };
        let scored_positions: Vec<super::ScoredPosition> = pos
            .iter()
            .map(|&p| super::ScoredPosition::new(p, per_position_score))
            .collect();
        Some(vec![(self.field_id, scored_positions)])
    }
}
