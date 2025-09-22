//! Term query - matches documents containing a specific term

use std::sync::Arc;

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::structures::BlockPostingList;
use crate::structures::TERMINATED;
use crate::{DocId, Score};

use super::docset::DocSet;
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

    fn fast_field_bitset(
        &self,
        reader: &SegmentReader,
        options: &super::ScorerOptions,
    ) -> Option<super::DocBitset> {
        let mut bits = super::DocBitset::new(reader.num_docs());
        let Some(fast_field) = reader.fast_field(self.field.0) else {
            return Some(bits);
        };
        let term = String::from_utf8_lossy(&self.term);
        let Some(target_ordinal) = fast_field.text_ordinal(&term) else {
            return (!options.stop_if_expired()).then_some(bits);
        };
        if !fast_field.multi {
            // Avoid decoding column codec headers separately for every doc.
            let scanned = fast_field.try_scan_single_values(|doc, ordinal| {
                if doc.is_multiple_of(1024) && options.stop_if_expired() {
                    return Err(());
                }
                if ordinal == target_ordinal {
                    bits.set(doc);
                }
                Ok(())
            });
            return (scanned.is_ok() && !options.stop_if_expired()).then_some(bits);
        }
        // Multi-value fast equality retains the ordinary scorer's first-value
        // semantics. The single-value batch API cannot represent those offsets.
        if let Some(mut scorer) = FastFieldTextScorer::try_new(
            reader,
            self.field,
            &term,
            options.shared_threshold.as_ref(),
        ) {
            let mut doc = scorer.doc();
            while doc != TERMINATED {
                bits.set(doc);
                doc = scorer.advance();
            }
        }
        // A cancelled scan is never a complete filter, especially under NOT.
        (!options.stop_if_expired()).then_some(bits)
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

/// Complete membership and positioned callers need a cursor. Ranked callers
/// benefit from block traversal only when bounds can avoid part of the list.
fn can_rank_term(
    postings: &BlockPostingList,
    limit: usize,
    complete: bool,
    collect_positions: bool,
) -> bool {
    if complete || collect_positions {
        return false;
    }
    let has_block_bounds = postings.min_len().is_some() && postings.num_blocks() > 1;
    let needs_top_k = limit < postings.doc_count() as usize;
    has_block_bounds && needs_top_k
}

// ── Unified term scorer macro ────────────────────────────────────────────
//
// Parameterised on:
//   $get_postings_fn – get_postings | get_postings_sync
//   $get_positions_fn – get_positions | get_positions_sync
//   $($aw)*          – .await  (present for async, absent for sync)
macro_rules! term_plan {
    ($field:expr, $term:expr, $global_stats:expr, $reader:expr, $limit:expr,
     $load_positions:expr, $eligibility:expr, $budget:expr, $complete:expr, $skip_scoring_setup:expr, $initial_threshold:expr, $physical_field:expr, $get_postings_fn:ident, $get_positions_fn:ident
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
            if let Some(scorer) = FastFieldTextScorer::try_new(reader, field, &term_str, budget) {
                return Ok(Box::new(scorer) as Box<dyn Scorer + '_>);
            }
            return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>);
        }

        let postings = reader.$get_postings_fn(field, term) $(. $aw)* ?;

        match postings {
            Some(posting_list) if reader.chunk_map(field).is_some_and(|map| map.is_document_map())
                && ($complete || $load_positions || $physical_field == Some(field)) => {
                let map = reader.chunk_map(field).unwrap();
                let (idf, avg_field_len) = compute_term_idf(&posting_list, field, reader, global_stats, term);
                let mut scorer = TermScorer::new(posting_list, idf, avg_field_len, 1.0)
                    .with_params(super::Bm25Params::for_field(reader.schema(), field));
                scorer.chunk_lengths = Some(map.clone());
                scorer.budget = budget.cloned();
                if $load_positions && let Some(positions) = reader.$get_positions_fn(field, term) $(. $aw)* ? {
                    scorer = scorer.with_positions(field.0, positions);
                }
                if $physical_field == Some(field) {
                    return Ok(Box::new(scorer) as Box<dyn Scorer + '_>);
                }
                let scorer = super::required_text::mapped_documents(
                    scorer, map.clone(), reader.num_docs(), budget.cloned(),
                    |scorer, slot| scorer.iterator.seek_physical(slot),
                )?;
                Ok(super::filtered::filtered(scorer, $eligibility.clone()))
            }
            // Chunked field: postings are keyed by virtual chunk id. Score the
            // chunks, fold them back to documents and report the ordinals.
            Some(posting_list) if reader.has_text_mapping(field) => {
                let (idf, avg_field_len) =
                    compute_term_idf(&posting_list, field, reader, global_stats, term);
                if $complete {
                    return complete_text_scorer(vec![(posting_list, idf)], avg_field_len, reader, field, &super::ScorerOptions {
                        shared_threshold: budget.cloned(), eligibility: $eligibility.clone(),
                        skip_scoring_setup: $skip_scoring_setup, ..Default::default()
                    });
                }
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

                // Ranked term requests can use the same block bounds as a text
                // union. Complete membership and positioned callers keep a cursor.
                if can_rank_term(&posting_list, limit, $complete, $load_positions) {
                    return super::planner::finish_text_maxscore(
                        vec![(posting_list, idf)],
                        avg_field_len,
                        reader.doc_lengths(field),
                        limit,
                        &std::cell::Cell::new($initial_threshold),
                        reader,
                        field,
                        $eligibility.as_ref().map(|filter| {
                            let filter = filter.clone();
                            Box::new(move |doc| filter.contains(doc)) as super::DocPredicate<'_>
                        }),
                        super::Bm25Params::for_field(reader.schema(), field),
                        None,
                        1.0,
                        budget,
                    );
                }

                let positions = if $load_positions {
                    reader.$get_positions_fn(field, term) $(. $aw)* ?
                } else {
                    None
                };

                let mut scorer = TermScorer::new(posting_list, idf, avg_field_len, 1.0)
                    .with_params(super::Bm25Params::for_field(reader.schema(), field));
                scorer.budget = budget.filter(|b| b.deadline().is_some()).cloned();
                if let Some(lengths) = reader.doc_lengths(field) {
                    scorer = scorer.with_doc_lengths(lengths.clone(), $skip_scoring_setup);
                }
                if let Some(pos) = positions {
                    scorer = scorer.with_positions(field.0, pos);
                }
                Ok(Box::new(scorer) as Box<dyn Scorer + '_>)
            }
            None => {
                let term_str = String::from_utf8_lossy(term);
                if let Some(scorer) = FastFieldTextScorer::try_new(reader, field, &term_str, budget) {
                    Ok(Box::new(scorer) as Box<dyn Scorer + '_>)
                } else {
                    Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>)
                }
            }
        }
    }};
}

impl Query for TermQuery {
    fn physical_text_field(&self, reader: &SegmentReader, complete: bool) -> Option<Field> {
        let entry = reader.schema().get_field_entry(self.field)?;
        (complete
            && entry.indexed
            && !entry.fast
            && reader
                .chunk_map(self.field)
                .is_some_and(|map| map.is_document_map()))
        .then_some(self.field)
    }
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
                options.complete_text_matches,
                options.skip_scoring_setup,
                options.initial_threshold,
                options.physical_text_field,
                get_postings,
                get_positions,
                await
            )
        })
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let field = self.field;
        let term = self.term.clone();
        Box::pin(async move { reader.text_doc_freq(field, &term).await })
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
            options.complete_text_matches,
            options.skip_scoring_setup,
            options.initial_threshold,
            options.physical_text_field,
            get_postings_sync,
            get_positions_sync
        )
    }

    fn as_doc_predicate<'a>(&self, reader: &'a SegmentReader) -> Option<super::DocPredicate<'a>> {
        let entry = reader.schema().get_field_entry(self.field)?;
        // A fast column exposes the first complete value. Indexed term
        // membership is equivalent only for single-valued raw text; analyzed
        // text and later values must keep their posting-list semantics.
        if entry.indexed
            && (entry.multi || !matches!(entry.tokenizer.as_deref(), Some("raw" | "raw_ci")))
        {
            return None;
        }
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
        if reader.has_text_mapping(self.field) {
            return None;
        }
        // Exact: the posting list header carries the doc count.
        let pl = reader.get_postings_sync(self.field, &self.term).ok()??;
        Some(pl.doc_count() as u64)
    }

    fn as_doc_bitset(&self, reader: &SegmentReader) -> Option<super::DocBitset> {
        self.as_doc_bitset_with_options(reader, &super::ScorerOptions::default())
    }

    fn as_doc_bitset_with_options(
        &self,
        reader: &SegmentReader,
        options: &super::ScorerOptions,
    ) -> Option<super::DocBitset> {
        if options.stop_if_expired() {
            return None;
        }
        if reader
            .schema()
            .get_field_entry(self.field)
            .is_some_and(|entry| !entry.indexed)
        {
            // Fast-only text has no posting list. Match exactly as its ordinary
            // scorer does, without the bounded candidate-heap fallback.
            return self.fast_field_bitset(reader, options);
        }
        #[cfg(feature = "sync")]
        {
            // Chunked postings use chunk ids, not document ids.
            if reader.has_text_mapping(self.field) {
                return None;
            }
            let Some(pl) = reader.get_postings_sync(self.field, &self.term).ok()? else {
                // Preserve the ordinary term scorer's fast-column fallback.
                return self.fast_field_bitset(reader, options);
            };
            // Indexed membership remains O(matches), not a full-column scan.
            let mut bitset = super::DocBitset::new(reader.num_docs());
            let mut iter = pl.iterator();
            let mut visited = 0usize;
            while iter.doc() != TERMINATED {
                if visited.is_multiple_of(1024) && options.stop_if_expired() {
                    return None;
                }
                bitset.set(iter.doc());
                iter.advance();
                visited += 1;
            }
            (!options.stop_if_expired()).then_some(bitset)
        }
        #[cfg(not(feature = "sync"))]
        {
            None
        }
    }

    fn text_terms(&self, out: &mut Vec<(Field, Vec<u8>)>) {
        out.push((self.field, self.term.clone()));
    }

    fn decompose(&self) -> super::QueryDecomposition {
        super::QueryDecomposition::TextTerm(TermQueryInfo {
            weight: 1.0,
            field: self.field,
            term: self.term.clone(),
            global_stats: self.global_stats.clone(),
        })
    }
}

struct TermScorer {
    budget: Option<super::SharedThreshold>,
    iterator: crate::structures::BlockPostingIterator<'static>,
    /// Physical posting cardinality for conjunction planning, not a live hit count.
    doc_count: u32,
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
    chunk_lengths: Option<crate::segment::chunk_map::ChunkMap>,
    /// Per-field k1/b.
    params: super::Bm25Params,
    normalization: Option<Box<super::bm25::NormTable>>,
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
            doc_count: posting_list.doc_count(),
            iterator: posting_list.into_iterator(),
            idf,
            avg_field_len,
            field_boost,
            field_id: 0,
            positions: None,
            lengths: None,
            chunk_lengths: None,
            params: super::Bm25Params::default(),
            normalization: None,
        }
    }

    /// Score with the field's BM25 parameters.
    pub fn with_params(mut self, params: super::Bm25Params) -> Self {
        self.params = params;
        if self
            .lengths
            .as_ref()
            .is_some_and(|lengths| lengths.is_quantized())
        {
            self.normalization = Some(Box::new(super::bm25::NormTable::new(
                params,
                self.avg_field_len,
            )));
        }
        self
    }

    /// Score with the field's persisted per-document lengths.
    fn with_doc_lengths(
        mut self,
        lengths: crate::segment::chunk_map::DocLengths,
        skip_scoring_setup: bool,
    ) -> Self {
        if lengths.is_quantized() && !skip_scoring_setup {
            self.normalization = Some(Box::new(super::bm25::NormTable::new(
                self.params,
                self.avg_field_len,
            )));
        } else {
            self.normalization = None;
        }
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

    fn score_batch_values(&self, docs: &[DocId], tfs: &[u32], scores: &mut [Score]) {
        let lengths = self
            .chunk_lengths
            .as_ref()
            .map(super::scoring::LengthSource::Chunks)
            .or_else(|| {
                self.lengths
                    .as_ref()
                    .map(super::scoring::LengthSource::Docs)
            });
        super::scoring::score_text_run(
            self.params,
            self.idf,
            self.avg_field_len,
            lengths,
            self.normalization.as_deref(),
            docs,
            tfs,
            scores,
        );
    }

    fn score_window<const ACCUMULATE: bool>(
        &mut self,
        base: DocId,
        scores: &mut [Score; super::docset::DOC_WINDOW_SIZE as usize],
        bits: &mut super::docset::DocWindow,
    ) {
        use super::docset::DocSet;
        if self.seek(base) == TERMINATED {
            return;
        }
        let end = base.saturating_add(super::docset::DOC_WINDOW_SIZE);
        let mut lengths = [0u32; crate::structures::postings::POSTING_BLOCK_SIZE];
        let mut values = [0.0; crate::structures::postings::POSTING_BLOCK_SIZE];
        let params = self.params;
        let idf = self.idf;
        let avg_len = self.avg_field_len;
        let boost = self.field_boost;
        let length_source = self
            .chunk_lengths
            .as_ref()
            .map(super::scoring::LengthSource::Chunks)
            .or_else(|| {
                self.lengths
                    .as_ref()
                    .map(super::scoring::LengthSource::Docs)
            });
        let budget = &self.budget;
        let normalization = self.normalization.as_deref();
        self.iterator.visit_postings_until(end, |docs, tfs| {
            crate::observe::search_work!(score_batches += 1);
            if budget
                .as_ref()
                .is_some_and(super::SharedThreshold::stop_if_expired)
            {
                return false;
            }
            if let (Some(super::scoring::LengthSource::Docs(norms)), Some(table)) =
                (length_source, normalization)
            {
                crate::observe::search_work!(lookup_score_units += docs.len());
                table.score_batch(
                    params,
                    idf,
                    avg_len,
                    boost,
                    docs.iter().map(|&doc| norms.norm_code(doc)),
                    tfs,
                    &mut values[..docs.len()],
                );
            } else {
                crate::observe::search_work!(exact_score_units += docs.len());
                if let Some(source) = length_source {
                    source.gather_lengths(docs, &mut lengths[..docs.len()]);
                } else {
                    lengths[..docs.len()].copy_from_slice(tfs);
                }
                // Independent canonical scores over contiguous inputs. Missing
                // lengths retain the scalar scorer's TF fallback.
                for i in 0..docs.len() {
                    let tf = tfs[i] as f32;
                    let len = if lengths[i] == 0 {
                        tf
                    } else {
                        lengths[i] as f32
                    };
                    values[i] = params.score_boosted(tf, idf, len, avg_len, boost);
                }
            }
            for (i, &doc) in docs.iter().enumerate() {
                let offset = (doc - base) as usize;
                if ACCUMULATE {
                    scores[offset] += values[i];
                } else {
                    scores[offset] = values[i];
                }
                bits[offset / 64] |= 1u64 << (offset % 64);
            }
            true
        });
    }
}

impl super::docset::DocSet for TermScorer {
    fn supports_doc_batches(&self) -> bool {
        self.budget.is_none()
    }

    fn fill_doc_batch(&mut self, docs: &mut super::docset::DocBatch) -> usize {
        if self.budget.is_some() {
            return super::docset::fill_batch(self, docs);
        }
        self.iterator.fill_doc_batch(docs)
    }

    fn retain_doc_batch(&mut self, docs: &mut super::docset::DocBatch, len: usize) -> usize {
        assert!(len <= docs.len());
        if self.budget.is_some() {
            return super::docset::retain_batch(self, docs, len);
        }
        self.iterator.retain_doc_batch(&mut docs[..len])
    }

    fn supports_doc_windows(&self) -> bool {
        true
    }

    fn fill_doc_window(&mut self, base: DocId, bits: &mut super::docset::DocWindow) {
        if self.doc() == TERMINATED {
            bits.fill(0);
            return;
        }
        self.iterator.fill_doc_window(base, bits);
    }

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
        let doc = self.iterator.advance();
        if self.positions.is_some() {
            // Commit the position prefix now so the immutable
            // `position_cursor()` used by `matched_positions` is O(1).
            self.iterator.position_cursor_mut();
        }
        doc
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.doc() == TERMINATED {
            return TERMINATED;
        }
        let doc = self.iterator.seek(target);
        if self.positions.is_some() {
            // See `advance`: commit the position prefix eagerly.
            self.iterator.position_cursor_mut();
        }
        doc
    }

    fn size_hint(&self) -> u32 {
        self.doc_count
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
    budget: Option<super::SharedThreshold>,
}

impl<'a> FastFieldTextScorer<'a> {
    fn try_new(
        reader: &'a SegmentReader,
        field: Field,
        text: &str,
        budget: Option<&super::SharedThreshold>,
    ) -> Option<Self> {
        let fast_field = reader.fast_field(field.0)?;
        let target_ordinal = fast_field.text_ordinal(text)?;
        let num_docs = reader.num_docs();
        let mut scorer = Self {
            fast_field,
            target_ordinal,
            current: 0,
            num_docs,
            budget: budget.filter(|budget| budget.deadline().is_some()).cloned(),
        };
        // Position on first matching doc
        if scorer.doc() != TERMINATED && fast_field.get_u64(0) != target_ordinal {
            scorer.scan_forward();
        }
        Some(scorer)
    }

    fn scan_forward(&mut self) {
        loop {
            self.current += 1;
            if self.current >= self.num_docs
                || (self.current.is_multiple_of(1024)
                    && self
                        .budget
                        .as_ref()
                        .is_some_and(super::SharedThreshold::stop_if_expired))
            {
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
        if self.current >= self.num_docs
            || self
                .budget
                .as_ref()
                .is_some_and(super::SharedThreshold::stop_if_expired)
        {
            TERMINATED
        } else {
            self.current
        }
    }

    fn advance(&mut self) -> DocId {
        if self.doc() == TERMINATED {
            return TERMINATED;
        }
        self.scan_forward();
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.doc() == TERMINATED {
            return TERMINATED;
        }
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
    fn supports_score_batches(&self) -> bool {
        self.budget.is_none() && self.field_boost == 1.0
    }

    fn fill_score_batch(
        &mut self,
        docs: &mut super::docset::DocBatch,
        scores: &mut super::ScoreBatch,
    ) -> usize {
        if !self.supports_score_batches() {
            return super::traits::fill_score_batch_scalar(self, docs, scores);
        }
        let mut tfs = [0; super::docset::DOC_BATCH_SIZE];
        let count = self.iterator.fill_scored_doc_batch(docs, &mut tfs);
        self.score_batch_values(&docs[..count], &tfs[..count], &mut scores[..count]);
        count
    }

    fn score_batch_matches(
        &mut self,
        docs: &super::docset::DocBatch,
        len: usize,
        scores: &mut super::ScoreBatch,
        matches: &mut super::ScoreBatchMask,
    ) {
        assert!(len <= docs.len());
        if !self.supports_score_batches() {
            return super::traits::score_batch_matches_scalar(self, docs, len, scores, matches);
        }
        matches.fill(0);
        let mut retained = *docs;
        let mut tfs = [0; super::docset::DOC_BATCH_SIZE];
        let count = self
            .iterator
            .retain_scored_doc_batch(&mut retained[..len], &mut tfs[..len]);
        let mut values = [0.0; super::docset::DOC_BATCH_SIZE];
        self.score_batch_values(&retained[..count], &tfs[..count], &mut values[..count]);
        let mut input = 0;
        for i in 0..count {
            while docs[input] < retained[i] {
                input += 1;
            }
            scores[input] = values[i];
            matches[input / 64] |= 1 << (input % 64);
        }
    }

    fn supports_score_windows(&self) -> bool {
        self.doc_count > crate::structures::postings::POSTING_BLOCK_SIZE as u32
    }

    fn fill_score_window(
        &mut self,
        base: DocId,
        scores: &mut [Score; super::docset::DOC_WINDOW_SIZE as usize],
        bits: &mut super::docset::DocWindow,
    ) {
        bits.fill(0);
        self.score_window::<false>(base, scores, bits);
    }

    fn accumulate_score_window(
        &mut self,
        base: DocId,
        scores: &mut [Score; super::docset::DOC_WINDOW_SIZE as usize],
        bits: &mut super::docset::DocWindow,
    ) {
        self.score_window::<true>(base, scores, bits);
    }

    fn score(&self) -> Score {
        let tf = self.iterator.term_freq() as f32;
        if let (Some(lengths), Some(table)) = (&self.lengths, &self.normalization)
            && self.chunk_lengths.is_none()
        {
            crate::observe::search_work!(lookup_score_units += 1);
            return table.score_boosted(
                self.params,
                tf,
                self.idf,
                lengths.norm_code(self.iterator.doc()),
                self.avg_field_len,
                self.field_boost,
            );
        }
        // Persisted field length when the segment has norms; otherwise `tf`
        crate::observe::search_work!(exact_score_units += 1);
        // stands in for the length (legacy segments).
        let doc_len = self
            .chunk_lengths
            .as_ref()
            .map(|map| map.length(self.iterator.doc()).max(map.length_floor()) as f32)
            .or_else(|| {
                self.lengths
                    .as_ref()
                    .map(|lengths| lengths.length(self.iterator.doc()) as f32)
            })
            .filter(|len| *len > 0.0)
            .unwrap_or(tf);
        self.params
            .score_boosted(tf, self.idf, doc_len, self.avg_field_len, self.field_boost)
    }

    fn matched_positions(&self) -> Option<super::MatchedPositions> {
        let positions = self.positions.as_ref()?;
        let pos =
            positions.positions(self.iterator.position_cursor(), self.iterator.term_freq())?;
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

pub(super) fn complete_text_scorer<'a>(
    postings: Vec<(BlockPostingList, f32)>,
    avg_field_len: f32,
    reader: &'a SegmentReader,
    field: Field,
    options: &super::ScorerOptions,
) -> crate::Result<Box<dyn Scorer + 'a>> {
    let budget = options.shared_threshold.clone();
    let eligibility = options.eligibility.clone();
    let skip_scoring_setup = options.skip_scoring_setup;
    let physical = options.physical_text_field == Some(field);
    if postings.is_empty()
        || budget
            .as_ref()
            .is_some_and(super::SharedThreshold::stop_if_expired)
    {
        return Ok(Box::new(EmptyScorer));
    }
    let map = reader.chunk_map(field);
    if reader.has_text_mapping(field) && map.is_none() {
        return Err(crate::Error::Corruption(
            "chunked text has postings without a chunk map".into(),
        ));
    }
    if !physical && map.is_some_and(|map| !map.is_doc_ordered()) {
        return super::required_text::scorer(
            postings,
            avg_field_len,
            reader,
            field,
            budget,
            eligibility,
        );
    }
    let mut terms: Vec<Box<dyn Scorer>> = Vec::with_capacity(postings.len());
    for (posting, idf) in postings {
        let mut scorer = TermScorer::new(posting, idf, avg_field_len, 1.0)
            .with_params(super::Bm25Params::for_field(reader.schema(), field));
        scorer.chunk_lengths = map.cloned();
        if let Some(lengths) = reader.doc_lengths(field) {
            scorer = scorer.with_doc_lengths(lengths.clone(), skip_scoring_setup);
        }
        scorer.budget = budget.clone();
        terms.push(Box::new(scorer));
    }
    let scorer = super::boolean::BooleanScorer::disjunction(terms);
    let scorer: Box<dyn Scorer + 'a> = match map {
        Some(map) if !map.is_document_map() => {
            super::phrase::fold_chunked_phrase_scorer(scorer, map.clone(), field.0, budget)
        }
        Some(_) => Box::new(scorer),
        None => Box::new(scorer),
    };
    Ok(super::filtered::filtered(scorer, eligibility))
}

/// Point BM25 probes. Targets are sorted physical IDs, never a retrieval top-k.
pub(super) async fn score_term_candidates(
    reader: &SegmentReader,
    field: Field,
    terms: &[(Vec<u8>, f32)],
    targets: &[u32],
    stats: Option<&Arc<GlobalStats>>,
    scratch: &mut crate::structures::postings::PostingDecodeScratch,
) -> crate::Result<Vec<f32>> {
    reader.check_posting_integrity()?;
    let mut scores = vec![0.0; targets.len()];
    let Some(&first_target) = targets.first() else {
        return Ok(scores);
    };
    let params = super::Bm25Params::for_field(reader.schema(), field);
    for (term, weight) in terms {
        let Some(postings) = reader.get_postings(field, term).await? else {
            continue;
        };
        let (idf, avg_len) = compute_term_idf(&postings, field, reader, stats, term);
        let mut cursor = postings.into_candidate_iterator(first_target, scratch);
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
        cursor.recycle(scratch);
    }
    reader.check_posting_integrity()?;
    Ok(scores)
}

#[cfg(test)]
mod score_window_tests {
    use super::*;
    use crate::query::DocSet;
    use crate::segment::chunk_map::DocLengths;
    use crate::structures::postings::{PostingCodec, PostingList};

    #[test]
    fn compact_term_scores_preserve_scalar_bits_membership_and_resume() {
        let mut list = PostingList::new();
        for i in 0..701u32 {
            list.push(i * 11 + 1, [0, 1, 3, 65536, u32::MAX][i as usize % 5]);
        }
        let lengths = DocLengths::from_lengths(
            &(0..7800)
                .map(|i| [0, 1, 97, 65535][i % 4])
                .collect::<Vec<_>>(),
        );
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let postings = BlockPostingList::from_posting_list_with_codec(&list, codec).unwrap();
            for with_lengths in [false, true] {
                for boost in [0.0, 0.5, 1.0, 2.5] {
                    for params in [
                        super::super::Bm25Params { k1: 0.0, b: 0.0 },
                        super::super::Bm25Params::default(),
                        super::super::Bm25Params { k1: 14.3, b: 1.0 },
                    ] {
                        let build = || {
                            let mut scorer =
                                TermScorer::new(postings.clone(), 2.713, 503.17, boost)
                                    .with_params(params);
                            if with_lengths {
                                scorer.lengths = Some(lengths.clone());
                            }
                            scorer
                        };
                        let mut scalar = build();
                        let mut batch = build();
                        assert_eq!(batch.supports_score_batches(), boost == 1.0);
                        let mut docs = [0; super::super::docset::DOC_BATCH_SIZE];
                        let mut scores = [f32::NAN; super::super::docset::DOC_BATCH_SIZE];
                        scalar.seek(121);
                        batch.seek(121);
                        while batch.doc() != TERMINATED {
                            let len = batch.fill_score_batch(&mut docs, &mut scores);
                            assert!(len > 0);
                            for i in 0..len {
                                assert_eq!(docs[i], scalar.doc());
                                assert_eq!(
                                    scores[i].to_bits(),
                                    scalar.score().to_bits(),
                                    "{codec:?} boost={boost} doc={}",
                                    docs[i]
                                );
                                scalar.advance();
                            }
                            assert_eq!(batch.doc(), scalar.doc());
                        }
                        let mut scalar = build();
                        let mut batch = build();
                        for start in (0..8000u32).step_by(128) {
                            let docs = std::array::from_fn(|i| start + i as u32);
                            let mut bits = [u64::MAX; 2];
                            batch.score_batch_matches(&docs, docs.len(), &mut scores, &mut bits);
                            for (i, &doc) in docs.iter().enumerate() {
                                let found = scalar.seek(doc) == doc;
                                assert_eq!(bits[i / 64] & (1 << (i % 64)) != 0, found);
                                if found {
                                    assert_eq!(scores[i].to_bits(), scalar.score().to_bits());
                                }
                            }
                            assert_eq!(batch.doc(), scalar.doc());
                        }
                    }
                }
            }
        }
        let mut timed = TermScorer::new(
            BlockPostingList::from_posting_list(&list).unwrap(),
            1.0,
            1.0,
            1.0,
        );
        timed.budget = Some(super::super::SharedThreshold::new());
        assert!(!timed.supports_score_batches());
    }

    #[test]
    fn replacement_term_windows_clear_stale_hits_and_stop_at_deadlines() {
        let mut list = PostingList::new();
        for doc in [0, 4095, 4096, 8192, TERMINATED - 1] {
            list.push(doc, 1);
        }
        let postings = BlockPostingList::from_posting_list(&list).unwrap();
        let mut scalar = TermScorer::new(postings.clone(), -0.0, 10.0, 1.0);
        let mut batched = TermScorer::new(postings.clone(), -0.0, 10.0, 1.0);
        let mut scores = Box::new([f32::NAN; super::super::docset::DOC_WINDOW_SIZE as usize]);
        let mut bits = [u64::MAX; super::super::docset::DOC_WINDOW_WORDS];
        for base in [
            0,
            4096,
            8192,
            TERMINATED - super::super::docset::DOC_WINDOW_SIZE,
        ] {
            bits.fill(u64::MAX);
            scores.fill(f32::NAN);
            batched.fill_score_window(base, &mut scores, &mut bits);
            scalar.seek(base);
            let end = base.saturating_add(super::super::docset::DOC_WINDOW_SIZE);
            while scalar.doc() < end {
                let offset = (scalar.doc() - base) as usize;
                assert_ne!(bits[offset / 64] & (1u64 << (offset % 64)), 0);
                assert_eq!(scores[offset].to_bits(), scalar.score().to_bits());
                bits[offset / 64] &= !(1u64 << (offset % 64));
                scalar.advance();
            }
            assert!(bits.iter().all(|&word| word == 0));
            assert_eq!(batched.doc(), scalar.doc());
        }
        assert_eq!(batched.doc(), TERMINATED);
        let mut expired = TermScorer::new(postings, 1.0, 10.0, 1.0);
        let budget = super::super::SharedThreshold::for_limit(1)
            .with_deadline(Some(std::time::Instant::now()));
        expired.budget = Some(budget.clone());
        bits.fill(u64::MAX);
        expired.fill_score_window(0, &mut scores, &mut bits);
        assert!(bits.iter().all(|&word| word == 0));
        assert_eq!(expired.doc(), TERMINATED);
        assert!(budget.truncated());
    }

    #[test]
    fn term_score_runs_preserve_scalar_bits_with_boosts_lengths_and_legacy_fallback() {
        let mut list = PostingList::new();
        for i in 0..327u32 {
            list.push(i * 11, [1, 3, 127, 65535, 65536, u32::MAX][i as usize % 6]);
        }
        let lengths = DocLengths::from_lengths(
            &(0..3600)
                .map(|i| [0, 1, 97, 65535][i % 4])
                .collect::<Vec<_>>(),
        );
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let postings = BlockPostingList::from_posting_list_with_codec(&list, codec).unwrap();
            for with_lengths in [false, true] {
                for params in [
                    super::super::Bm25Params { k1: 0.0, b: 0.0 },
                    super::super::Bm25Params::default(),
                    super::super::Bm25Params { k1: 14.3, b: 1.0 },
                ] {
                    for boost in [0.0, 0.5, 1.0, 2.5] {
                        for avg in [0.0, 1.0, 503.17] {
                            for idf in [-0.0, 0.001, 10.0] {
                                let build = || {
                                    let mut scorer =
                                        TermScorer::new(postings.clone(), idf, avg, boost)
                                            .with_params(params);
                                    if with_lengths {
                                        scorer.lengths = Some(lengths.clone());
                                    }
                                    scorer
                                };
                                for replace in [false, true] {
                                    let mut scalar = build();
                                    let mut batched = build();
                                    let base = 121;
                                    scalar.seek(base);
                                    batched.seek(base);
                                    let mut scores = Box::new(
                                        [0.0; super::super::docset::DOC_WINDOW_SIZE as usize],
                                    );
                                    let mut bits = [if replace { u64::MAX } else { 0 };
                                        super::super::docset::DOC_WINDOW_WORDS];
                                    if replace {
                                        scores.fill(f32::NAN);
                                        batched.fill_score_window(base, &mut scores, &mut bits);
                                    } else {
                                        batched.accumulate_score_window(
                                            base,
                                            &mut scores,
                                            &mut bits,
                                        );
                                    }
                                    while scalar.doc() != TERMINATED {
                                        let offset = (scalar.doc() - base) as usize;
                                        assert_ne!(bits[offset / 64] & (1u64 << (offset % 64)), 0);
                                        let expected = if replace {
                                            scalar.score()
                                        } else {
                                            0.0 + scalar.score()
                                        };
                                        assert_eq!(
                                            scores[offset].to_bits(),
                                            expected.to_bits(),
                                            "codec={codec:?} lengths={with_lengths} params={params:?} boost={boost} avg={avg} idf={idf} doc={}",
                                            scalar.doc()
                                        );
                                        bits[offset / 64] &= !(1u64 << (offset % 64));
                                        scalar.advance();
                                    }
                                    assert!(bits.iter().all(|word| *word == 0));
                                    assert_eq!(batched.doc(), TERMINATED);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
