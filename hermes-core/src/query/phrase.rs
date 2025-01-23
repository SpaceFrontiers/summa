//! Phrase query - matches documents containing terms in consecutive positions

use std::sync::Arc;

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::structures::{BlockPostingIterator, BlockPostingList, PositionPostingList, TERMINATED};
use crate::{DocId, Score};

use super::{CountFuture, EmptyScorer, GlobalStats, Query, Scorer, ScorerFuture};

/// Phrase query - matches documents containing terms in consecutive positions
///
/// Example: "quick brown fox" matches only if all three terms appear
/// consecutively in the document.
#[derive(Clone)]
pub struct PhraseQuery {
    pub field: Field,
    /// Terms in the phrase, in order
    pub terms: Vec<Vec<u8>>,
    /// Optional slop (max distance between terms, 0 = exact phrase)
    pub slop: u32,
    /// Optional global statistics for cross-segment IDF
    global_stats: Option<Arc<GlobalStats>>,
}

impl std::fmt::Debug for PhraseQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let terms: Vec<_> = self
            .terms
            .iter()
            .map(|t| String::from_utf8_lossy(t).to_string())
            .collect();
        f.debug_struct("PhraseQuery")
            .field("field", &self.field)
            .field("terms", &terms)
            .field("slop", &self.slop)
            .finish()
    }
}

impl PhraseQuery {
    /// Create a new exact phrase query
    pub fn new(field: Field, terms: Vec<Vec<u8>>) -> Self {
        Self {
            field,
            terms,
            slop: 0,
            global_stats: None,
        }
    }

    /// Create from text (splits on whitespace and lowercases)
    pub fn text(field: Field, phrase: &str) -> Self {
        let terms: Vec<Vec<u8>> = phrase
            .split_whitespace()
            .map(|w| w.to_lowercase().into_bytes())
            .collect();
        Self {
            field,
            terms,
            slop: 0,
            global_stats: None,
        }
    }

    /// Set slop (max distance between terms)
    pub fn with_slop(mut self, slop: u32) -> Self {
        self.slop = slop;
        self
    }

    /// Set global statistics for cross-segment IDF
    pub fn with_global_stats(mut self, stats: Arc<GlobalStats>) -> Self {
        self.global_stats = Some(stats);
        self
    }
}

impl Query for PhraseQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        let field = self.field;
        let terms = self.terms.clone();
        let slop = self.slop;
        let _global_stats = self.global_stats.clone();

        Box::pin(async move {
            if terms.is_empty() {
                return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + 'a>);
            }

            // Single term - delegate to TermQuery
            if terms.len() == 1 {
                let term_query = super::TermQuery::new(field, terms[0].clone());
                return term_query.scorer(reader, limit).await;
            }

            // Check if positions are available
            if !reader.has_positions(field) {
                // Fall back to AND query (BooleanQuery with MUST clauses)
                let mut bool_query = super::BooleanQuery::new();
                for term in &terms {
                    bool_query = bool_query.must(super::TermQuery::new(field, term.clone()));
                }
                return bool_query.scorer(reader, limit).await;
            }

            // Load postings and positions for all terms (parallel per term)
            let mut term_postings: Vec<BlockPostingList> = Vec::with_capacity(terms.len());
            let mut term_positions: Vec<PositionPostingList> = Vec::with_capacity(terms.len());

            for term in &terms {
                // Fetch postings and positions in parallel
                let (postings, positions) = futures::join!(
                    reader.get_postings(field, term),
                    reader.get_positions(field, term)
                );

                match (postings?, positions?) {
                    (Some(p), Some(pos)) => {
                        term_postings.push(p);
                        term_positions.push(pos);
                    }
                    _ => {
                        // If any term is missing, no documents can match
                        return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + 'a>);
                    }
                }
            }

            // Compute combined IDF (sum of individual IDFs)
            let idf: f32 = term_postings
                .iter()
                .map(|p| {
                    let num_docs = reader.num_docs() as f32;
                    let doc_freq = p.doc_count() as f32;
                    super::bm25_idf(doc_freq, num_docs)
                })
                .sum();

            let avg_field_len = reader.avg_field_len(field);

            Ok(Box::new(PhraseScorer::new(
                term_postings,
                term_positions,
                slop,
                idf,
                avg_field_len,
            )) as Box<dyn Scorer + 'a>)
        })
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let field = self.field;
        let terms = self.terms.clone();

        Box::pin(async move {
            if terms.is_empty() {
                return Ok(0);
            }

            // Estimate based on minimum posting list size
            let mut min_count = u32::MAX;
            for term in &terms {
                match reader.get_postings(field, term).await? {
                    Some(list) => min_count = min_count.min(list.doc_count()),
                    None => return Ok(0),
                }
            }

            // Phrase matching will typically match fewer docs than the minimum
            // Estimate ~10% of the smallest posting list
            Ok((min_count / 10).max(1))
        })
    }
}

/// Scorer that checks phrase positions
struct PhraseScorer {
    /// Posting iterators for each term
    posting_iters: Vec<BlockPostingIterator<'static>>,
    /// Position iterators for each term
    position_lists: Vec<PositionPostingList>,
    /// Max slop between terms
    slop: u32,
    /// Current matching document
    current_doc: DocId,
    /// Combined IDF
    idf: f32,
    /// Average field length
    avg_field_len: f32,
}

impl PhraseScorer {
    fn new(
        posting_lists: Vec<BlockPostingList>,
        position_lists: Vec<PositionPostingList>,
        slop: u32,
        idf: f32,
        avg_field_len: f32,
    ) -> Self {
        let posting_iters: Vec<_> = posting_lists
            .into_iter()
            .map(|p| p.into_iterator())
            .collect();

        let mut scorer = Self {
            posting_iters,
            position_lists,
            slop,
            current_doc: 0,
            idf,
            avg_field_len,
        };

        scorer.find_next_phrase_match();
        scorer
    }

    /// Find next document where all terms appear as a phrase
    fn find_next_phrase_match(&mut self) {
        loop {
            // First, find a document where all terms appear (AND semantics)
            let doc = self.find_next_and_match();
            if doc == TERMINATED {
                self.current_doc = TERMINATED;
                return;
            }

            // Check if positions form a valid phrase
            if self.check_phrase_positions(doc) {
                self.current_doc = doc;
                return;
            }

            // Advance and try again
            self.posting_iters[0].advance();
        }
    }

    /// Find next document where all terms appear
    fn find_next_and_match(&mut self) -> DocId {
        if self.posting_iters.is_empty() {
            return TERMINATED;
        }

        loop {
            let max_doc = self.posting_iters.iter().map(|it| it.doc()).max().unwrap();

            if max_doc == TERMINATED {
                return TERMINATED;
            }

            let mut all_match = true;
            for it in &mut self.posting_iters {
                let doc = it.seek(max_doc);
                if doc != max_doc {
                    all_match = false;
                    if doc == TERMINATED {
                        return TERMINATED;
                    }
                }
            }

            if all_match {
                return max_doc;
            }
        }
    }

    /// Check if positions form a valid phrase for the given document
    fn check_phrase_positions(&self, doc_id: DocId) -> bool {
        // Get positions for each term in this document
        let mut term_positions: Vec<Vec<u32>> = Vec::with_capacity(self.position_lists.len());

        for pos_list in &self.position_lists {
            match pos_list.get_positions(doc_id) {
                Some(positions) => term_positions.push(positions.to_vec()),
                None => return false,
            }
        }

        // Check for consecutive positions
        // For exact phrase (slop=0), position[i+1] = position[i] + 1
        self.find_phrase_match(&term_positions)
    }

    /// Find if there's a valid phrase match among the positions
    fn find_phrase_match(&self, term_positions: &[Vec<u32>]) -> bool {
        if term_positions.is_empty() {
            return false;
        }

        // For each position of the first term, check if subsequent terms
        // have positions that form a phrase
        for &first_pos in &term_positions[0] {
            if self.check_phrase_from_position(first_pos, term_positions) {
                return true;
            }
        }

        false
    }

    /// Check if a phrase exists starting from the given position
    fn check_phrase_from_position(&self, start_pos: u32, term_positions: &[Vec<u32>]) -> bool {
        let mut expected_pos = start_pos;

        for (i, positions) in term_positions.iter().enumerate() {
            if i == 0 {
                continue; // Skip first term, already matched
            }

            expected_pos += 1;

            // Find a position within slop distance
            let found = positions.iter().any(|&pos| {
                if self.slop == 0 {
                    pos == expected_pos
                } else {
                    let diff = pos.abs_diff(expected_pos);
                    diff <= self.slop
                }
            });

            if !found {
                return false;
            }
        }

        true
    }
}

impl Scorer for PhraseScorer {
    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn score(&self) -> Score {
        if self.current_doc == TERMINATED {
            return 0.0;
        }

        // Sum term frequencies for BM25 scoring
        let tf: f32 = self
            .posting_iters
            .iter()
            .map(|it| it.term_freq() as f32)
            .sum();

        // Phrase matches get a boost since they're more precise
        super::bm25_score(tf, self.idf, tf, self.avg_field_len) * 1.5
    }

    fn advance(&mut self) -> DocId {
        if self.current_doc == TERMINATED {
            return TERMINATED;
        }

        self.posting_iters[0].advance();
        self.find_next_phrase_match();
        self.current_doc
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if target == TERMINATED {
            self.current_doc = TERMINATED;
            return TERMINATED;
        }

        self.posting_iters[0].seek(target);
        self.find_next_phrase_match();
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        0
    }
}
