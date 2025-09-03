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

/// Compute (idf, avg_field_len) from a posting list, using global stats when available.
pub(super) fn compute_term_idf(
    posting_list: &BlockPostingList,
    field: Field,
    reader: &SegmentReader,
    global_stats: Option<&Arc<GlobalStats>>,
    term: &[u8],
) -> (f32, f32) {
    if let Some(stats) = global_stats {
        let term_str = String::from_utf8_lossy(term);
        let global_idf = stats.text_idf(field, &term_str);
        if global_idf > 0.0 {
            return (global_idf, stats.avg_field_len(field));
        }
    }
    let num_docs = reader.text_corpus_size(field);
    let doc_freq = posting_list.doc_count() as f32;
    (
        super::bm25_idf(doc_freq, num_docs),
        reader.avg_field_len(field),
    )
}

// ── Unified term scorer macro ────────────────────────────────────────────
//
// Parameterised on:
//   $get_postings_fn – get_postings | get_postings_sync
//   $get_positions_fn – get_positions | get_positions_sync
//   $($aw)*          – .await  (present for async, absent for sync)
macro_rules! term_plan {
    ($field:expr, $term:expr, $global_stats:expr, $reader:expr, $limit:expr,
     $load_positions:expr, $eligibility:expr, $budget:expr, $get_postings_fn:ident, $get_positions_fn:ident
     $(, $aw:tt)*) => {{
        let field: Field = $field;
        let term: &[u8] = $term;
        let global_stats: Option<&Arc<GlobalStats>> = $global_stats;
        let reader: &SegmentReader = $reader;
        let limit: usize = $limit;
        let budget: Option<&super::SharedThreshold> = $budget;
        if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
            return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>);
        }

        // Non-indexed fields → fast-field-only path
        let is_indexed = reader.schema().get_field_entry(field).is_none_or(|e| e.indexed);
        if !is_indexed {
            let term_str = String::from_utf8_lossy(term);
            if let Some(scorer) = FastFieldTextScorer::try_new(reader, field, &term_str) {
                return Ok(Box::new(scorer) as Box<dyn Scorer + '_>);
            }
            return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>);
        }

        let postings = reader.$get_postings_fn(field, term) $(. $aw)* ?;

        match postings {
            // Chunked field: postings are keyed by virtual chunk id. Score the
            // chunks, fold them back to documents and report the ordinals.
            Some(posting_list) if reader.is_chunked_field(field) => {
                let (idf, avg_field_len) =
                    compute_term_idf(&posting_list, field, reader, global_stats, term);
                super::planner::finish_chunked_text_maxscore(
                    vec![(posting_list, idf)],
                    avg_field_len,
                    limit,
                    reader,
                    field,
                    $eligibility.as_ref().map(|filter| {
                        let filter = filter.clone();
                        Box::new(move |doc| filter.contains(doc)) as super::DocPredicate<'_>
                    }),
                    None,
                    1.0,
                    budget,
                )
            }
            Some(posting_list) => {
                let (idf, avg_field_len) =
                    compute_term_idf(&posting_list, field, reader, global_stats, term);

                let positions = if $load_positions {
                    reader.$get_positions_fn(field, term) $(. $aw)* ?
                } else {
                    None
                };

                let mut scorer = TermScorer::new(posting_list, idf, avg_field_len, 1.0)
                    .with_params(super::Bm25Params::for_field(reader.schema(), field));
                scorer.budget = budget.filter(|b| b.deadline().is_some()).cloned();
                if let Some(lengths) = reader.doc_lengths(field) {
                    scorer = scorer.with_doc_lengths(lengths.clone());
                }
                if let Some(pos) = positions {
                    scorer = scorer.with_positions(field.0, pos);
                }
                Ok(Box::new(scorer) as Box<dyn Scorer + '_>)
            }
            None => {
                let term_str = String::from_utf8_lossy(term);
                if let Some(scorer) = FastFieldTextScorer::try_new(reader, field, &term_str) {
                    Ok(Box::new(scorer) as Box<dyn Scorer + '_>)
                } else {
                    Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>)
                }
            }
        }
    }};
}

impl Query for TermQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        self.scorer_with_options(reader, limit, super::ScorerOptions::with_positions())
    }

    fn scorer_with_options<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
        options: super::ScorerOptions,
    ) -> ScorerFuture<'a> {
        let field = self.field;
        let term = self.term.clone();
        let global_stats = self
            .global_stats
            .clone()
            .or_else(|| options.global_stats.clone());
        let load_positions = options.collect_positions;
        Box::pin(async move {
            term_plan!(
                field,
                &term,
                global_stats.as_ref(),
                reader,
                limit,
                load_positions,
                options.eligibility,
                options.shared_threshold.as_ref(),
                get_postings,
                get_positions,
                await
            )
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
        let global_stats = self
            .global_stats
            .clone()
            .or_else(|| options.global_stats.clone());
        term_plan!(
            self.field,
            &self.term,
            global_stats.as_ref(),
            reader,
            limit,
            options.collect_positions,
            options.eligibility,
            options.shared_threshold.as_ref(),
            get_postings_sync,
            get_positions_sync
        )
    }

    fn as_doc_predicate<'a>(&self, reader: &'a SegmentReader) -> Option<super::DocPredicate<'a>> {
        let fast_field = reader.fast_field(self.field.0)?;
        let term_str = String::from_utf8_lossy(&self.term);
        match fast_field.text_ordinal(&term_str) {
            Some(target_ordinal) => Some(Box::new(move |doc_id: DocId| -> bool {
                fast_field.get_u64(doc_id) == target_ordinal
            })),
            // Term doesn't exist in this segment — no doc can match.
            None => Some(Box::new(|_| false)),
        }
    }

    #[cfg(feature = "sync")]
    fn bitset_cardinality_estimate(&self, reader: &SegmentReader) -> Option<u64> {
        // Chunked postings count chunks, not documents.
        if reader.is_chunked_field(self.field) {
            return None;
        }
        // Exact: the posting list header carries the doc count.
        let pl = reader.get_postings_sync(self.field, &self.term).ok()??;
        Some(pl.doc_count() as u64)
    }

    #[cfg(feature = "sync")]
    fn as_doc_bitset(&self, reader: &SegmentReader) -> Option<super::DocBitset> {
        // Chunked postings are keyed by virtual chunk ids, not document ids;
        // a bitset over them would filter the wrong documents.
        if reader.is_chunked_field(self.field) {
            return None;
        }
        // Build bitset from posting list: O(M) where M = matching doc count.
        // Much faster than O(N) fast-field scan for selective terms.
        let mut bitset = super::DocBitset::new(reader.num_docs());
        let Some(pl) = reader.get_postings_sync(self.field, &self.term).ok()? else {
            return Some(bitset);
        };
        let mut iter = pl.iterator();
        loop {
            let doc = iter.doc();
            if doc == crate::structures::TERMINATED {
                break;
            }
            bitset.set(doc);
            iter.advance();
        }
        Some(bitset)
    }

    fn text_terms(&self, out: &mut Vec<(Field, Vec<u8>)>) {
        out.push((self.field, self.term.clone()));
    }

    fn decompose(&self) -> super::QueryDecomposition {
        super::QueryDecomposition::TextTerm(TermQueryInfo {
            weight: 1.0,
            field: self.field,
            term: self.term.clone(),
        })
    }
}

struct TermScorer {
    budget: Option<super::SharedThreshold>,
    iterator: crate::structures::BlockPostingIterator<'static>,
    idf: f32,
    /// Average field length for this field
    avg_field_len: f32,
    /// Field boost/weight for BM25F
    field_boost: f32,
    /// Field ID for position reporting
    field_id: u32,
    /// Positions of the term (if positions are enabled)
    positions: Option<crate::structures::TermPositions>,
    /// Persisted per-document field lengths; `None` keeps `tf` as the length.
    lengths: Option<crate::segment::chunk_map::DocLengths>,
    /// Per-field k1/b.
    params: super::Bm25Params,
}

impl TermScorer {
    pub fn new(
        posting_list: BlockPostingList,
        idf: f32,
        avg_field_len: f32,
        field_boost: f32,
    ) -> Self {
        Self {
            budget: None,
            iterator: posting_list.into_iterator(),
            idf,
            avg_field_len,
            field_boost,
            field_id: 0,
            positions: None,
            lengths: None,
            params: super::Bm25Params::default(),
        }
    }

    /// Score with the field's BM25 parameters.
    pub fn with_params(mut self, params: super::Bm25Params) -> Self {
        self.params = params;
        self
    }

    /// Score with the field's persisted per-document lengths.
    pub fn with_doc_lengths(mut self, lengths: crate::segment::chunk_map::DocLengths) -> Self {
        self.lengths = Some(lengths);
        self
    }

    pub fn with_positions(
        mut self,
        field_id: u32,
        positions: crate::structures::TermPositions,
    ) -> Self {
        self.field_id = field_id;
        self.positions = Some(positions);
        self
    }
}

impl super::docset::DocSet for TermScorer {
    fn doc(&self) -> DocId {
        if self
            .budget
            .as_ref()
            .is_some_and(super::SharedThreshold::stop_if_expired)
        {
            return TERMINATED;
        }
        self.iterator.doc()
    }

    fn advance(&mut self) -> DocId {
        if self.doc() == TERMINATED {
            return TERMINATED;
        }
        self.iterator.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.doc() == TERMINATED {
            return TERMINATED;
        }
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
        // Persisted field length when the segment has norms; otherwise `tf`
        // stands in for the length (legacy segments).
        let doc_len = self
            .lengths
            .as_ref()
            .map(|lengths| lengths.length(self.iterator.doc()) as f32)
            .filter(|len| *len > 0.0)
            .unwrap_or(tf);
        self.params
            .score_boosted(tf, self.idf, doc_len, self.avg_field_len, self.field_boost)
    }

    fn matched_positions(&self) -> Option<super::MatchedPositions> {
        let positions = self.positions.as_ref()?;
        let doc_id = self.iterator.doc();
        let pos = positions.positions(
            doc_id,
            self.iterator.position_cursor(),
            self.iterator.term_freq(),
        )?;
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

/// Point BM25 probes. Targets are sorted physical IDs, never a retrieval top-k.
pub(super) async fn score_term_candidates(
    reader: &SegmentReader,
    field: Field,
    terms: &[(Vec<u8>, f32)],
    targets: &[u32],
    stats: Option<&Arc<GlobalStats>>,
) -> crate::Result<Vec<f32>> {
    let mut scores = vec![0.0; targets.len()];
    let params = super::Bm25Params::for_field(reader.schema(), field);
    for (term, weight) in terms {
        let Some(postings) = reader.get_postings(field, term).await? else {
            continue;
        };
        let (idf, avg_len) = compute_term_idf(&postings, field, reader, stats, term);
        let mut cursor = postings.into_iterator();
        for (index, &target) in targets.iter().enumerate() {
            if cursor.seek(target) != target {
                continue;
            }
            let tf = cursor.term_freq() as f32;
            let length = if let Some(map) = reader.chunk_map(field) {
                map.bm25_length(target) as f32
            } else if let Some(lengths) = reader.doc_lengths(field) {
                lengths.length(target) as f32
            } else {
                tf
            };
            scores[index] += params.score(tf, idf * weight, length, avg_len);
        }
    }
    Ok(scores)
}
