//! Phrase query - matches documents containing terms in consecutive positions

use std::sync::Arc;

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::structures::{BlockPostingIterator, BlockPostingList, TERMINATED, TermPositions};
use crate::{DocId, Score};

use super::docset::DocSet;
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
    /// Token offset of each term inside the phrase, ascending, one per term.
    /// `offsets[i + 1] - offsets[i]` is the required distance between two
    /// consecutive terms: 1 for adjacent words, more when index-time stop
    /// words were dropped between them (`quantum@0 art@3`). [`PhraseQuery::new`]
    /// makes every term adjacent.
    pub offsets: Vec<u32>,
    /// Optional slop (max distance between terms, 0 = exact phrase)
    pub slop: u32,
    /// Optional global statistics for cross-segment IDF
    global_stats: Option<Arc<GlobalStats>>,
}

impl std::fmt::Display for PhraseQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let terms: Vec<String> = self
            .terms
            .iter()
            .zip(&self.offsets)
            .map(|(term, offset)| {
                if self.is_adjacent() {
                    String::from_utf8_lossy(term).into_owned()
                } else {
                    format!("{}@{offset}", String::from_utf8_lossy(term))
                }
            })
            .collect();
        write!(f, "Phrase({}:\"{}\"", self.field.0, terms.join(" "))?;
        if self.slop > 0 {
            write!(f, "~{}", self.slop)?;
        }
        write!(f, ")")
    }
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
            .field("offsets", &self.offsets)
            .field("slop", &self.slop)
            .finish()
    }
}

impl PhraseQuery {
    /// Multi-token phrases need token positions; element ordinals cannot prove
    /// adjacency. Shared by schema-aware parsing, both scorers and backfill.
    pub(crate) fn validate_positions(schema: &crate::Schema, field: Field) -> crate::Result<()> {
        let entry = schema
            .get_field_entry(field)
            .ok_or_else(|| crate::Error::FieldNotFound(field.0.to_string()))?;
        if entry.indexed
            && entry.field_type == crate::dsl::FieldType::Text
            && entry
                .positions
                .is_some_and(|mode| mode.tracks_token_position())
        {
            return Ok(());
        }
        Err(crate::Error::Query(format!(
            "phrase queries on field {:?} require token positions; rebuild with indexed<token_position> or use AND for unordered terms",
            entry.name
        )))
    }

    /// Create a new exact phrase query of adjacent terms.
    pub fn new(field: Field, terms: Vec<Vec<u8>>) -> Self {
        let offsets = (0..terms.len() as u32).collect();
        Self {
            field,
            terms,
            offsets,
            slop: 0,
            global_stats: None,
        }
    }

    /// Create a phrase whose terms carry their token offsets, as produced by
    /// a tokenizer that drops stop words without renumbering (`(0, quantum)`,
    /// `(3, art)` for "quantum of the art"). Offsets must be ascending.
    pub fn with_offsets(field: Field, terms: Vec<(u32, Vec<u8>)>) -> Self {
        debug_assert!(
            terms.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "phrase offsets must be strictly ascending"
        );
        let (offsets, terms): (Vec<u32>, Vec<Vec<u8>>) = terms.into_iter().unzip();
        Self {
            field,
            terms,
            offsets,
            slop: 0,
            global_stats: None,
        }
    }

    /// Create from text using the simple tokenizer (whitespace split,
    /// punctuation stripped, lowercased). Fields with a stemming tokenizer
    /// should tokenize the phrase themselves and call
    /// [`PhraseQuery::with_offsets`] so the query terms match the indexed
    /// stems and keep the gaps of dropped stop words.
    pub fn text(field: Field, phrase: &str) -> Self {
        use crate::tokenizer::Tokenizer;
        let terms: Vec<(u32, Vec<u8>)> = crate::tokenizer::SimpleTokenizer
            .tokenize(phrase)
            .into_iter()
            .map(|token| (token.position, token.text.into_bytes()))
            .collect();
        Self::with_offsets(field, terms)
    }

    /// Whether every term must directly follow the previous one.
    fn is_adjacent(&self) -> bool {
        self.offsets.windows(2).all(|pair| pair[1] == pair[0] + 1)
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

/// Ordered maps need only one document's ordinals and one matching-chunk
/// lookahead. Reordered maps retain the stable, all-document aggregation.
pub(super) fn fold_chunked_phrase_scorer<'a, S: Scorer + 'a>(
    mut scorer: S,
    chunk_map: crate::segment::chunk_map::ChunkMap,
    field_id: u32,
    budget: Option<super::SharedThreshold>,
) -> Box<dyn Scorer + 'a> {
    if chunk_map.is_doc_ordered() {
        let mut folded = ChunkedPhraseScorer {
            inner: scorer,
            chunk_map,
            field_id,
            budget,
            current_doc: TERMINATED,
            score: 0.0,
            ordinals: crate::segment::VectorOrdinals::new(),
        };
        folded.fold_next_document();
        return Box::new(folded);
    }
    let mut raw: Vec<(u32, u16, f32)> = Vec::new();
    while scorer.doc() != TERMINATED {
        if budget
            .as_ref()
            .is_some_and(super::SharedThreshold::stop_if_expired)
        {
            return Box::new(EmptyScorer);
        }
        let (doc_id, ordinal) = chunk_map.resolve(scorer.doc());
        raw.push((doc_id, ordinal, scorer.score()));
        scorer.advance();
    }
    if budget
        .as_ref()
        .is_some_and(super::SharedThreshold::stop_if_expired)
    {
        // Do not start an all-hit sort/fold after cancellation.
        return Box::new(EmptyScorer);
    }
    // Every matching document is kept: a phrase is also used as a MUST
    // constraint (verifier or bitset), where truncating to `limit` would
    // silently reject documents that do contain the phrase.
    let combined =
        crate::segment::combine_ordinal_results(raw, super::MultiValueCombiner::Max, usize::MAX);
    Box::new(super::vector::VectorResultScorer::new(combined, field_id))
}

struct ChunkedPhraseScorer<S> {
    inner: S,
    chunk_map: crate::segment::chunk_map::ChunkMap,
    field_id: u32,
    budget: Option<super::SharedThreshold>,
    current_doc: DocId,
    score: Score,
    ordinals: crate::segment::VectorOrdinals,
}

impl<S: Scorer> ChunkedPhraseScorer<S> {
    fn finish(&mut self) -> DocId {
        self.current_doc = TERMINATED;
        self.score = 0.0;
        self.ordinals.clear();
        TERMINATED
    }

    fn expired(&self) -> bool {
        self.budget
            .as_ref()
            .is_some_and(super::SharedThreshold::stop_if_expired)
    }

    fn fold_next_document(&mut self) -> DocId {
        if self.expired() || self.inner.doc() == TERMINATED {
            return self.finish();
        }
        let doc = self.chunk_map.doc_id(self.inner.doc());
        self.ordinals.clear();
        loop {
            self.ordinals.push((
                u32::from(self.chunk_map.ordinal(self.inner.doc())),
                self.inner.score(),
            ));
            self.inner.advance();
            // Never expose a partial document's max score or ordinal list.
            if self.expired() {
                return self.finish();
            }
            if self.inner.doc() == TERMINATED || self.chunk_map.doc_id(self.inner.doc()) != doc {
                break;
            }
        }
        self.current_doc = doc;
        self.score = super::MultiValueCombiner::Max.combine(&self.ordinals);
        doc
    }
}

impl<S: Scorer> DocSet for ChunkedPhraseScorer<S> {
    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn advance(&mut self) -> DocId {
        if self.current_doc == TERMINATED {
            return TERMINATED;
        }
        self.fold_next_document()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current_doc >= target {
            return self.current_doc;
        }
        if target == TERMINATED || self.expired() {
            return self.finish();
        }
        let vid = self.chunk_map.lower_bound_doc(target);
        if vid == self.chunk_map.num_chunks() {
            return self.finish();
        }
        self.inner.seek(vid);
        self.fold_next_document()
    }

    fn size_hint(&self) -> u32 {
        if self.current_doc == TERMINATED {
            0
        } else {
            self.inner.size_hint().saturating_add(1)
        }
    }
}

impl<S: Scorer> Scorer for ChunkedPhraseScorer<S> {
    fn score(&self) -> Score {
        self.score
    }

    fn matched_positions(&self) -> Option<super::MatchedPositions> {
        (self.current_doc != TERMINATED).then(|| {
            vec![(
                self.field_id,
                self.ordinals
                    .iter()
                    .map(|&(ordinal, score)| super::ScoredPosition::new(ordinal, score))
                    .collect(),
            )]
        })
    }
}

/// Build the shared positional scorer without walking beyond the candidate set.
#[allow(clippy::too_many_arguments)]
fn prepare_phrase_scorer(
    reader: &SegmentReader,
    field: Field,
    terms: &[Vec<u8>],
    term_data: Vec<(BlockPostingList, TermPositions)>,
    offsets: &[u32],
    slop: u32,
    stats: Option<&Arc<GlobalStats>>,
    budget: Option<super::SharedThreshold>,
) -> crate::Result<PhraseScorer> {
    let mut avg_len = reader.avg_field_len(field);
    let mut idf = 0.0;
    for ((postings, _), term) in term_data.iter().zip(terms) {
        let (term_idf, length) =
            super::term::compute_term_idf(postings, field, reader, stats, term);
        idf += term_idf;
        avg_len = length;
    }
    let (postings, positions) = term_data.into_iter().unzip();
    let mut scorer =
        PhraseScorer::unpositioned(postings, positions, offsets, slop, idf, avg_len, budget)
            .with_params(super::Bm25Params::for_field(reader.schema(), field));
    if reader.has_text_mapping(field) {
        let map = reader.chunk_map(field).ok_or_else(|| {
            crate::Error::Corruption("chunked phrase has postings without a chunk map".into())
        })?;
        scorer = scorer.with_lengths(Lengths::Chunks(map.clone()));
    } else if let Some(lengths) = reader.doc_lengths(field) {
        scorer = scorer.with_lengths(Lengths::Docs(lengths.clone()));
    }
    Ok(scorer)
}

/// Ordinary retrieval enumerates phrase matches; point scoring below parks
/// these same cursors only on nominated targets and shares frequency/scoring.
fn finish_phrase_scorer<'a>(
    mut scorer: PhraseScorer,
    reader: &SegmentReader,
    field: Field,
    budget: Option<super::SharedThreshold>,
    physical: bool,
) -> crate::Result<Box<dyn Scorer + 'a>> {
    scorer.find_next_phrase_match();
    if physical {
        return Ok(Box::new(scorer));
    }
    if let Some(map) = reader.chunk_map(field) {
        if map.is_document_map() {
            return super::required_text::mapped_documents(
                scorer,
                map.clone(),
                reader.num_docs(),
                budget,
                |scorer, slot| {
                    for cursor in &mut scorer.posting_iters {
                        cursor.seek_physical(slot);
                    }
                    scorer.park(TERMINATED, None);
                    scorer.find_next_phrase_match();
                    scorer.doc()
                },
            );
        }
        Ok(fold_chunked_phrase_scorer(
            scorer,
            map.clone(),
            field.0,
            budget,
        ))
    } else {
        Ok(Box::new(scorer))
    }
}

pub(super) async fn score_phrase_candidates(
    reader: &SegmentReader,
    query: &PhraseQuery,
    targets: &[u32],
    stats: Option<&Arc<GlobalStats>>,
) -> crate::Result<Vec<f32>> {
    reader.check_posting_integrity()?;
    let stats = query.global_stats.as_ref().or(stats);
    if query.terms.len() == 1 {
        return super::term::score_term_candidates(
            reader,
            query.field,
            &[(query.terms[0].clone(), 1.0)],
            targets,
            stats,
            &mut Default::default(),
        )
        .await;
    }
    let mut scores = vec![0.0; targets.len()];
    if query.terms.is_empty() {
        return Ok(scores);
    }
    PhraseQuery::validate_positions(reader.schema(), query.field)?;
    let mut data = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let (p, pos) = futures::join!(
            reader.get_postings(query.field, term),
            reader.get_positions(query.field, term)
        );
        match (p?, pos?) {
            (Some(p), Some(pos)) => data.push((p, pos)),
            _ => return Ok(scores),
        }
    }
    let mut scorer = prepare_phrase_scorer(
        reader,
        query.field,
        &query.terms,
        data,
        &query.offsets,
        query.slop,
        stats,
        None,
    )?;
    for (index, &target) in targets.iter().enumerate() {
        let mut matches = true;
        for cursor in &mut scorer.posting_iters {
            matches &= cursor.seek(target) == target;
        }
        if matches && scorer.check_phrase_positions(target) {
            scorer.current_doc = target;
            scores[index] = scorer.score();
        }
    }
    reader.check_posting_integrity()?;
    Ok(scores)
}

// ── Shared early-return checks for phrase scorer ─────────────────────────
//
// Handles: empty terms, single-term delegation, token-position requirements.
// Parameterised on the option-aware scorer function plus async/sync awaiting.
macro_rules! phrase_early_returns {
    ($field:expr, $terms:expr, $reader:expr, $limit:expr,
     $scorer_fn:ident, $options:expr $(, $aw:tt)*) => {
        if $options.stop_if_expired() || $terms.is_empty() {
            return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>);
        }
        if $terms.len() == 1 {
            let tq = super::TermQuery::new($field, $terms[0].clone());
            return tq.$scorer_fn($reader, $limit, $options) $(. $aw)* ;
        }
        PhraseQuery::validate_positions($reader.schema(), $field)?;
    };
}

impl Query for PhraseQuery {
    fn physical_text_field(&self, reader: &SegmentReader, _complete: bool) -> Option<Field> {
        let entry = reader.schema().get_field_entry(self.field)?;
        (entry.indexed
            && !entry.fast
            && reader
                .chunk_map(self.field)
                .is_some_and(|map| map.is_document_map()))
        .then_some(self.field)
    }
    fn candidate_query(&self) -> crate::Result<crate::query::CandidateQuery> {
        Ok(super::CandidateQuery::new(
            self.field,
            super::candidate_scoring::ScoreComponent::Phrase(self.clone()),
        ))
    }

    fn text_terms(&self, out: &mut Vec<(Field, Vec<u8>)>) {
        for term in &self.terms {
            out.push((self.field, term.clone()));
        }
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
        let terms = self.terms.clone();
        let offsets = self.offsets.clone();
        let slop = self.slop;
        let stats = self
            .global_stats
            .clone()
            .or_else(|| options.global_stats.clone());

        Box::pin(async move {
            phrase_early_returns!(
                field,
                terms,
                reader,
                limit,
                scorer_with_options,
                options,
                await
            );

            // Fetch postings + positions in parallel per term via futures::join!
            let mut term_data = Vec::with_capacity(terms.len());
            for term in &terms {
                let (postings, positions) = futures::join!(
                    reader.get_postings(field, term),
                    reader.get_positions(field, term)
                );
                match (postings?, positions?) {
                    (Some(p), Some(pos)) => term_data.push((p, pos)),
                    _ => return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + 'a>),
                }
            }

            let scorer = prepare_phrase_scorer(
                reader,
                field,
                &terms,
                term_data,
                &offsets,
                slop,
                stats.as_ref(),
                options.shared_threshold.clone(),
            )?;
            finish_phrase_scorer(
                scorer,
                reader,
                field,
                options.shared_threshold,
                options.physical_text_field == Some(field),
            )
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
        phrase_early_returns!(
            self.field,
            self.terms,
            reader,
            limit,
            scorer_sync_with_options,
            options
        );

        // Parallel fetch across all terms via rayon
        use rayon::prelude::*;
        let pairs: crate::Result<Vec<Option<(BlockPostingList, TermPositions)>>> = self
            .terms
            .par_iter()
            .map(|term| {
                let postings = reader.get_postings_sync(self.field, term)?;
                let positions = reader.get_positions_sync(self.field, term)?;
                Ok(match (postings, positions) {
                    (Some(p), Some(pos)) => Some((p, pos)),
                    _ => None,
                })
            })
            .collect();
        let mut term_data = Vec::with_capacity(self.terms.len());
        for entry in pairs? {
            match entry {
                Some(pair) => term_data.push(pair),
                None => return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + 'a>),
            }
        }

        let scorer = prepare_phrase_scorer(
            reader,
            self.field,
            &self.terms,
            term_data,
            &self.offsets,
            self.slop,
            self.global_stats.as_ref().or(options.global_stats.as_ref()),
            options.shared_threshold.clone(),
        )?;
        finish_phrase_scorer(
            scorer,
            reader,
            self.field,
            options.shared_threshold,
            options.physical_text_field == Some(self.field),
        )
    }

    /// Every document containing the phrase, as a bitset (documents, also
    /// for chunked fields). Lets the planner push a quoted span into the
    /// MaxScore executors as an O(1) predicate instead of a verifier.
    #[cfg(feature = "sync")]
    fn as_doc_bitset(&self, reader: &SegmentReader) -> Option<super::DocBitset> {
        self.as_doc_bitset_with_options(reader, &super::ScorerOptions::default())
    }

    #[cfg(feature = "sync")]
    fn as_doc_bitset_with_options(
        &self,
        reader: &SegmentReader,
        options: &super::ScorerOptions,
    ) -> Option<super::DocBitset> {
        if options.stop_if_expired() || self.terms.is_empty() {
            return None;
        }
        let mut bitset = super::DocBitset::new(reader.num_docs());
        if self.terms.len() == 1 {
            // Non-indexed terms may match through a fast column. Preserve the
            // scorer fallback; absent postings do not prove an empty filter.
            if reader
                .schema()
                .get_field_entry(self.field)
                .is_some_and(|entry| !entry.indexed)
            {
                return None;
            }
            // A one-term phrase is the term itself; walk its postings and
            // resolve chunk ids to documents where needed.
            let Some(list) = reader.get_postings_sync(self.field, &self.terms[0]).ok()? else {
                // A supported filter with no matches is not an unsupported
                // filter: the latter would invoke the bounded scorer fallback.
                return Some(bitset);
            };
            let chunk_map = reader.chunk_map(self.field);
            let mut it = list.iterator();
            while it.doc() != TERMINATED {
                if options.stop_if_expired() {
                    return None;
                }
                let doc = chunk_map.map_or(it.doc(), |map| map.doc_id(it.doc()));
                bitset.set(doc);
                it.advance();
            }
            return Some(bitset);
        }
        let mut scorer = self
            .scorer_sync_with_options(reader, usize::MAX, options.without_threshold())
            .ok()?;
        while scorer.doc() != TERMINATED {
            if options.stop_if_expired() {
                return None;
            }
            bitset.set(scorer.doc());
            scorer.advance();
        }
        if options.stop_if_expired() {
            None
        } else {
            Some(bitset)
        }
    }

    /// Matches are at most the rarest term's postings; the planner only
    /// needs the order of magnitude to pick which clause to materialize.
    #[cfg(feature = "sync")]
    fn bitset_cardinality_estimate(&self, reader: &SegmentReader) -> Option<u64> {
        let mut min = u64::MAX;
        for term in &self.terms {
            let list = reader.get_postings_sync(self.field, term).ok()??;
            min = min.min(u64::from(list.doc_count()));
        }
        Some((min / 10).max(1))
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

/// Real lengths of the scoring units of a phrase: chunk lengths of a chunked
/// field or persisted document lengths of a plain field.
enum Lengths {
    Chunks(crate::segment::chunk_map::ChunkMap),
    Docs(crate::segment::chunk_map::DocLengths),
}

impl Lengths {
    fn length(&self, id: u32) -> u32 {
        match self {
            Lengths::Chunks(map) => map.bm25_length(id),
            Lengths::Docs(lengths) => lengths.length(id),
        }
    }
}

/// Scorer that checks phrase positions
struct PhraseScorer {
    /// Upper bound on matches, used to choose conjunction drivers.
    cost: u32,
    /// Rarest posting list; position arrays retain their original phrase order.
    lead: usize,
    budget: Option<super::SharedThreshold>,
    /// Posting iterators for each term
    posting_iters: Vec<BlockPostingIterator<'static>>,
    /// Positions of each term (legacy list or cursor-addressed stream)
    position_lists: Vec<crate::structures::postings::TermPositionCursor>,
    /// Position-list cursors advance monotonically within a matching unit.
    position_indices: Vec<usize>,
    /// Traversal order is independent of original phrase coordinates and score
    /// multiplicity. Every posting probe starts with the most selective terms.
    term_order: smallvec::SmallVec<[usize; 8]>,
    /// Required distance of each term from the first one (`offsets[i] -
    /// offsets[0]`); `deltas[0]` is 0.
    deltas: Vec<u32>,
    /// Max slop between terms
    slop: u32,
    /// Current matching document
    current_doc: DocId,
    /// Membership is confirmed by one occurrence. Scoring finishes the saved
    /// monotone scan once; immutable score reads remain safe to share.
    first_match: bool,
    frequency: std::sync::OnceLock<u32>,
    next_start: usize,
    /// Exact confirmation for the current candidate; reset on cursor movement.
    confirmed: Option<bool>,
    /// Combined IDF
    idf: f32,
    /// Per-field k1/b.
    params: super::Bm25Params,
    /// Average field length
    avg_field_len: f32,
    /// Real lengths of the scoring units. `None` keeps the historic
    /// `tf`-as-length approximation.
    lengths: Option<Lengths>,
    /// Whether per-candidate score bounds are numerically safe; computed once
    /// from the lengths and parameters, not per candidate.
    prepared_bounds: Option<super::bm25::PreparedBounds>,
    /// Reusable position buffers (one per term, avoids per-document allocation)
    position_bufs: Vec<Vec<u32>>,
}

impl PhraseScorer {
    #[allow(clippy::too_many_arguments)]
    fn unpositioned(
        posting_lists: Vec<BlockPostingList>,
        position_lists: Vec<TermPositions>,
        offsets: &[u32],
        slop: u32,
        idf: f32,
        avg_field_len: f32,
        budget: Option<super::SharedThreshold>,
    ) -> Self {
        let (lead, cost) = posting_lists
            .iter()
            .enumerate()
            .map(|(index, list)| (index, list.doc_count()))
            .min_by_key(|&(_, count)| count)
            .unwrap_or((0, 0));
        let mut term_order: smallvec::SmallVec<[usize; 8]> = (0..posting_lists.len()).collect();
        term_order.sort_unstable_by_key(|&index| (posting_lists[index].doc_count(), index));
        let posting_iters: Vec<_> = posting_lists
            .into_iter()
            .map(|p| p.into_iterator())
            .collect();

        let num_terms = position_lists.len();
        // Offsets are optional for callers that built the query term by
        // term; missing entries mean adjacency.
        let first = offsets.first().copied().unwrap_or(0);
        let deltas: Vec<u32> = (0..num_terms)
            .map(|i| offsets.get(i).map_or(i as u32, |o| o - first))
            .collect();
        Self {
            cost,
            lead,
            budget: budget.filter(|b| b.deadline().is_some()),
            posting_iters,
            position_lists: position_lists
                .into_iter()
                .map(TermPositions::into_cursor)
                .collect(),
            position_indices: vec![0; num_terms],
            term_order,
            deltas,
            slop,
            current_doc: 0,
            first_match: false,
            frequency: std::sync::OnceLock::new(),
            next_start: 0,
            confirmed: None,
            params: super::Bm25Params::default(),
            idf,
            avg_field_len,
            lengths: None,
            prepared_bounds: None,
            position_bufs: (0..num_terms).map(|_| Vec::new()).collect(),
        }
    }

    /// Score with the real length of each scoring unit.
    fn with_lengths(mut self, lengths: Lengths) -> Self {
        self.lengths = Some(lengths);
        self.prepared_bounds = self.prepare_bounds();
        self
    }

    /// Score with the field's BM25 parameters.
    fn with_params(mut self, params: super::Bm25Params) -> Self {
        self.params = params;
        self.prepared_bounds = self.prepare_bounds();
        self
    }

    /// Conservative bounds need document lengths and parameters under which
    /// the canonical score is finite and monotone in term frequency.
    fn prepare_bounds(&self) -> Option<super::bm25::PreparedBounds> {
        if !matches!(self.lengths, Some(Lengths::Docs(_))) {
            return None;
        }
        super::bm25::PreparedBounds::new(self.params, u32::MAX, self.idf, self.avg_field_len)
    }

    /// Park on a posting intersection without reading positions.
    fn find_next_candidate(&mut self) -> DocId {
        let doc = self.find_next_and_match();
        if doc != self.current_doc {
            self.park(doc, None);
        }
        doc
    }

    /// Move to `doc` and invalidate per-document phrase state. `confirmed`
    /// records whether the candidate's positions were already checked.
    fn park(&mut self, doc: DocId, confirmed: Option<bool>) {
        self.current_doc = doc;
        self.first_match = false;
        self.frequency.take();
        self.confirmed = confirmed;
    }

    /// Find next document where all terms appear as a phrase.
    fn find_next_phrase_match(&mut self) {
        while self.find_next_candidate() != TERMINATED {
            if self.confirm_candidate() {
                return;
            }
            if self.current_doc == TERMINATED {
                return;
            }
            self.posting_iters[self.lead].advance();
        }
    }

    /// Find next document where all terms appear
    fn find_next_and_match(&mut self) -> DocId {
        if self.posting_iters.is_empty() {
            return TERMINATED;
        }

        if self.posting_iters.len() == 1 {
            return self.posting_iters[0].doc();
        }

        // Keep the two common cursors borrowed through alignment. Their
        // query identities and position streams remain in their original order.
        if let [first, second] = self.posting_iters.as_mut_slice() {
            let (lead, other) = if self.lead == 0 {
                (first, second)
            } else {
                (second, first)
            };
            loop {
                if self
                    .budget
                    .as_ref()
                    .is_some_and(super::SharedThreshold::stop_if_expired)
                {
                    return TERMINATED;
                }
                if let Some(doc) = lead.intersect_block(other) {
                    return doc;
                }
            }
        }

        'align: loop {
            if self
                .budget
                .as_ref()
                .is_some_and(super::SharedThreshold::stop_if_expired)
            {
                return TERMINATED;
            }
            let [lead, second] = self
                .posting_iters
                .get_disjoint_mut([self.lead, self.term_order[1]])
                .expect("distinct phrase cursors");
            let Some(candidate) = lead.intersect_block(second) else {
                continue;
            };
            if candidate == TERMINATED {
                return TERMINATED;
            }
            for &index in self.term_order.iter().skip(2) {
                let doc = self.posting_iters[index].seek(candidate);
                if doc != candidate {
                    self.posting_iters[self.lead].seek(doc);
                    continue 'align;
                }
            }
            return candidate;
        }
    }

    /// Check if positions form a valid phrase for the given document
    fn check_phrase_positions(&mut self, doc_id: DocId) -> bool {
        crate::observe::search_work!(phrase_confirmations += 1);
        // Backfill also calls this method directly after parking term cursors.
        // Frequency belongs to these position buffers, so invalidate it at
        // their write boundary as well as on ordinary candidate movement.
        self.first_match = false;
        self.frequency.take();
        if self.slop == 0 {
            return self.check_exact_phrase_positions(doc_id);
        }
        // Get positions for each term into reusable buffers (zero allocation).
        // The doc-posting iterator of every term is parked on `doc_id`, so
        // its cursor and term frequency address the term's position stream.
        for i in 0..self.position_lists.len() {
            if !self.read_term_positions(i, doc_id) {
                return false;
            }
        }

        self.position_indices.fill(0);
        self.next_start = 0;
        self.first_match = scan_phrase_matches(
            &self.position_bufs,
            &self.deltas,
            self.slop,
            &mut self.position_indices,
            &mut self.next_start,
            1,
        ) != 0;
        self.first_match
    }

    fn read_term_positions(&mut self, term: usize, doc_id: DocId) -> bool {
        let cursor = self.posting_iters[term].position_cursor_mut();
        let tf = self.posting_iters[term].term_freq();
        self.position_lists[term].read_into(doc_id, cursor, tf, &mut self.position_bufs[term])
    }

    fn check_exact_phrase_positions(&mut self, doc_id: DocId) -> bool {
        let anchor = self.lead;
        let last = if anchor == 0 {
            *self.term_order.last().unwrap()
        } else {
            0
        };
        if !self.read_term_positions(anchor, doc_id) {
            return false;
        }
        if anchor != 0 {
            // Filtering coordinates are possible original-first starts. Values
            // before the anchor offset cannot participate in an exact phrase.
            let delta = self.deltas[anchor];
            self.position_bufs[anchor].retain_mut(|position| {
                if let Some(start) = position.checked_sub(delta) {
                    *position = start;
                    true
                } else {
                    false
                }
            });
        }
        if self.position_bufs[anchor].is_empty() {
            return false;
        }
        for rank in 1..self.term_order.len() {
            let term = self.term_order[rank];
            if term == last && (anchor == 0 || rank + 1 == self.term_order.len()) {
                continue;
            }
            if !self.read_term_positions(term, doc_id) {
                return false;
            }
            let (starts, positions) = if anchor < term {
                let (left, right) = self.position_bufs.split_at_mut(term);
                (&mut left[anchor], &right[0])
            } else {
                let (left, right) = self.position_bufs.split_at_mut(anchor);
                (&mut right[0], &left[term])
            };
            retain_exact_phrase_starts(starts, positions, self.deltas[term]);
            if starts.is_empty() {
                return false;
            }
        }
        if (anchor == 0 || self.term_order.last() == Some(&0))
            && !self.read_term_positions(last, doc_id)
        {
            return false;
        }
        // Enumerate the original first term even when a rarer term supplied
        // the filter, so duplicate first starts retain their multiplicity.
        let (other, delta) = if anchor == 0 {
            (last, self.deltas[last])
        } else {
            (anchor, 0)
        };
        self.next_start = 0;
        self.position_indices[other] = 0;
        self.first_match = next_exact_phrase_match(
            &self.position_bufs[0],
            &self.position_bufs[other],
            delta,
            &mut self.next_start,
            &mut self.position_indices[other],
        )
        .is_some();
        self.first_match
    }

    /// Resume after the confirmed first occurrence, without scanning its
    /// prefix again. Only scoring consumers initialize this per-document value.
    fn exact_phrase_frequency(&self) -> u32 {
        if !self.first_match {
            return 0;
        }
        *self.frequency.get_or_init(|| {
            if self.slop == 0 {
                let (last, delta) = if self.lead == 0 {
                    let last = *self.term_order.last().unwrap();
                    (last, self.deltas[last])
                } else {
                    (self.lead, 0)
                };
                let mut next_start = self.next_start;
                let mut position_index = self.position_indices[last];
                let mut matches = 1;
                while next_exact_phrase_match(
                    &self.position_bufs[0],
                    &self.position_bufs[last],
                    delta,
                    &mut next_start,
                    &mut position_index,
                )
                .is_some()
                {
                    matches += 1;
                }
                return matches;
            }
            let mut indices = smallvec::SmallVec::<[usize; 8]>::from_slice(&self.position_indices);
            let mut next_start = self.next_start;
            1 + scan_phrase_matches(
                &self.position_bufs,
                &self.deltas,
                self.slop,
                &mut indices,
                &mut next_start,
                u32::MAX,
            )
        })
    }
}

/// Yield original first-term starts that match an exact offset. Retaining the
/// position cursor on equality preserves duplicate first-start multiplicity.
#[inline]
fn next_exact_phrase_match(
    starts: &[u32],
    positions: &[u32],
    delta: u32,
    next_start: &mut usize,
    position_index: &mut usize,
) -> Option<usize> {
    while let (Some(&start), Some(&position)) =
        (starts.get(*next_start), positions.get(*position_index))
    {
        let expected = u64::from(start) + u64::from(delta);
        match expected.cmp(&u64::from(position)) {
            std::cmp::Ordering::Less => *next_start += 1,
            std::cmp::Ordering::Greater => *position_index += 1,
            std::cmp::Ordering::Equal => {
                let matched = *next_start;
                *next_start += 1;
                return Some(matched);
            }
        }
    }
    None
}

fn retain_exact_phrase_starts(starts: &mut Vec<u32>, positions: &[u32], delta: u32) {
    let mut next_start = 0;
    let mut position_index = 0;
    let mut retained = 0;
    while let Some(matched) = next_exact_phrase_match(
        starts,
        positions,
        delta,
        &mut next_start,
        &mut position_index,
    ) {
        // `retained <= matched < next_start`: writes cannot change unread starts.
        starts[retained] = starts[matched];
        retained += 1;
    }
    starts.truncate(retained);
}

/// Count starts with a match in every term's independent slop interval.
/// Ascending starts make each interval monotone: O(terms * starts + positions),
/// preserving repeated terms and the existing (not edit-distance) slop rule.
/// A stopped scan retains its next start and all monotone term cursors.
fn scan_phrase_matches(
    bufs: &[Vec<u32>],
    deltas: &[u32],
    slop: u32,
    indices: &mut [usize],
    next_start: &mut usize,
    max_matches: u32,
) -> u32 {
    if max_matches == 0 {
        return 0;
    }
    let Some(first) = bufs.first() else {
        return 0;
    };
    let mut matches = 0;
    'starts: while let Some(&start) = first.get(*next_start) {
        *next_start += 1;
        for i in 1..bufs.len() {
            let expected = u64::from(start) + u64::from(deltas[i]);
            let low = expected.saturating_sub(u64::from(slop));
            let high = expected + u64::from(slop);
            while indices[i] < bufs[i].len() && u64::from(bufs[i][indices[i]]) < low {
                indices[i] += 1;
            }
            let Some(&position) = bufs[i].get(indices[i]) else {
                return matches;
            };
            if u64::from(position) > high {
                continue 'starts;
            }
        }
        matches += 1;
        if matches == max_matches {
            return matches;
        }
    }
    matches
}

#[cfg(test)]
fn count_phrase_matches(
    bufs: &[Vec<u32>],
    deltas: &[u32],
    slop: u32,
    indices: &mut [usize],
) -> u32 {
    indices.fill(0);
    scan_phrase_matches(bufs, deltas, slop, indices, &mut 0, u32::MAX)
}

impl super::docset::DocSet for PhraseScorer {
    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn advance(&mut self) -> DocId {
        if self.current_doc == TERMINATED {
            return TERMINATED;
        }

        self.posting_iters[self.lead].advance();
        self.find_next_phrase_match();
        self.current_doc
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if target == TERMINATED || self.current_doc == TERMINATED {
            self.park(TERMINATED, self.confirmed);
            return TERMINATED;
        }

        self.posting_iters[self.lead].seek(target);
        self.find_next_phrase_match();
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        self.cost
    }
}

impl Scorer for PhraseScorer {
    fn supports_candidate_score_bounds(&self) -> bool {
        self.prepared_bounds.is_some()
    }

    fn candidate_score_upper_bound(&self) -> Score {
        crate::observe::search_work!(phrase_bound_calls += 1);
        if self.current_doc == TERMINATED {
            return Score::INFINITY;
        }
        let Some(bounds) = &self.prepared_bounds else {
            return Score::INFINITY;
        };
        let Some(Lengths::Docs(lengths)) = &self.lengths else {
            return Score::INFINITY;
        };
        // Phrase frequency counts original-first starts, including duplicates.
        // Another term's smaller TF is not an upper bound on that multiplicity.
        let max_tf = self.posting_iters[0].term_freq();
        let length = lengths.length(self.current_doc).max(1);
        bounds.pair(max_tf, length)
    }

    fn advance_candidate(&mut self) -> DocId {
        if self.current_doc == TERMINATED {
            return TERMINATED;
        }
        self.posting_iters[self.lead].advance();
        self.find_next_candidate()
    }

    fn seek_candidate(&mut self, target: DocId) -> DocId {
        if target == TERMINATED || self.current_doc == TERMINATED {
            self.park(TERMINATED, Some(false));
            return TERMINATED;
        }
        self.posting_iters[self.lead].seek(target);
        self.find_next_candidate()
    }

    fn confirm_candidate(&mut self) -> bool {
        if self
            .budget
            .as_ref()
            .is_some_and(super::SharedThreshold::stop_if_expired)
        {
            self.park(TERMINATED, Some(false));
        }
        if self.current_doc == TERMINATED {
            return false;
        }
        if let Some(matched) = self.confirmed {
            return matched;
        }
        let matched = self.check_phrase_positions(self.current_doc);
        self.confirmed = Some(matched);
        matched
    }

    fn score(&self) -> Score {
        if self.current_doc == TERMINATED {
            return 0.0;
        }

        // BM25 over the phrase frequency with the summed idf of the terms
        // (Lucene semantics): a document with two occurrences of the phrase
        // outranks one with a single occurrence at equal length.
        let tf = self.exact_phrase_frequency().max(1) as f32;
        crate::observe::search_work!(phrase_score_units += 1);

        // Real unit length when the segment has it; otherwise the summed
        // term frequency stands in for the length (legacy segments).
        let doc_len = match &self.lengths {
            Some(lengths) => (lengths.length(self.current_doc) as f32).max(1.0),
            None => self
                .posting_iters
                .iter()
                .map(|it| it.term_freq() as f32)
                .sum::<f32>()
                .max(tf),
        };

        self.params.score(tf, self.idf, doc_len, self.avg_field_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phrase_bounds_preserve_duplicate_start_multiplicity_without_reading_positions() {
        use super::super::Bm25Params;
        use crate::structures::{PositionStreamEncoder, PostingCodec, PostingList};
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let mut lists = Vec::new();
            let mut positions = Vec::new();
            for mut values in [vec![0, 0, 0, 2, 4], vec![1, 3, 5]] {
                let mut list = PostingList::new();
                list.push(0, values.len() as u32);
                let mut bytes = Vec::new();
                let mut encoder = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
                encoder.push_doc(&mut values).unwrap();
                encoder.finish().unwrap();
                lists.push(
                    BlockPostingList::from_posting_list_with_options(&list, true, None, codec)
                        .unwrap(),
                );
                positions
                    .push(TermPositions::open(crate::directories::OwnedBytes::new(bytes)).unwrap());
            }
            for length in [0, 1, 9, u16::MAX] {
                for avg in [0.0, 1.0, 17.5, 10000.0] {
                    for params in [
                        Bm25Params::default(),
                        Bm25Params { k1: 0.0, b: 1.0 },
                        Bm25Params { k1: 7e20, b: 0.0 },
                        Bm25Params { k1: 3.0, b: 1.0 },
                    ] {
                        let mut scorer = PhraseScorer::unpositioned(
                            lists.clone(),
                            positions.clone(),
                            &[0, 1],
                            0,
                            1.2345,
                            avg,
                            None,
                        )
                        .with_params(params)
                        .with_lengths(Lengths::Docs(
                            crate::segment::chunk_map::DocLengths::from_lengths(&[length]),
                        ));
                        assert_eq!(scorer.find_next_candidate(), 0);
                        assert!(scorer.supports_candidate_score_bounds());
                        let bound = scorer.candidate_score_upper_bound();
                        let previous = params.upper_bound_with_impacts(5, 1.2345, avg, |a, b| {
                            Some((a + b * f64::from(length.max(1))) / 5.0)
                        });
                        assert_eq!(bound.to_bits(), previous.to_bits());
                        assert!(scorer.position_bufs.iter().all(Vec::is_empty));
                        assert!(scorer.confirm_candidate());
                        assert_eq!(
                            scorer.exact_phrase_frequency(),
                            5,
                            "smaller second-term TF must not cap duplicate starts"
                        );
                        assert!(
                            bound >= scorer.score(),
                            "{codec:?} length={length} avg={avg} params={params:?}"
                        );
                    }
                }
            }
            for (params, idf, avg) in [
                (Bm25Params { k1: -1.0, b: 0.75 }, 1.0, 10.0),
                (
                    Bm25Params {
                        k1: f32::MAX,
                        b: 0.75,
                    },
                    1.0,
                    10.0,
                ),
                (
                    Bm25Params {
                        k1: 1.2,
                        b: f32::NAN,
                    },
                    1.0,
                    10.0,
                ),
                (Bm25Params::default(), f32::NAN, 10.0),
                (Bm25Params::default(), -1.0, 10.0),
                (Bm25Params::default(), 1.0, f32::INFINITY),
            ] {
                let scorer = PhraseScorer::unpositioned(
                    lists.clone(),
                    positions.clone(),
                    &[0, 1],
                    0,
                    idf,
                    avg,
                    None,
                )
                .with_params(params)
                .with_lengths(Lengths::Docs(
                    crate::segment::chunk_map::DocLengths::from_lengths(&[10]),
                ));
                assert!(!scorer.supports_candidate_score_bounds());
            }
        }
    }

    #[test]
    fn selective_phrase_checks_an_intermediate_first_term_before_frequent_payloads() {
        use crate::structures::{PositionStreamEncoder, PostingList};
        let mut lists = Vec::new();
        let mut positions = Vec::new();
        for (term, docs) in [2, 16, 1].into_iter().enumerate() {
            let mut list = PostingList::new();
            let mut bytes = Vec::new();
            let mut encoder = PositionStreamEncoder::new(&mut bytes);
            for doc in 0..docs {
                let mut values = match term {
                    0 => vec![4, 14],
                    1 => (0..512).collect::<Vec<u32>>(),
                    _ => vec![2, 12],
                };
                list.push(doc, values.len() as u32);
                encoder.push_doc(&mut values).unwrap();
            }
            encoder.finish().unwrap();
            lists.push(BlockPostingList::from_posting_list_with(&list, true, None).unwrap());
            positions
                .push(TermPositions::open(crate::directories::OwnedBytes::new(bytes)).unwrap());
        }
        let mut scorer =
            PhraseScorer::unpositioned(lists, positions, &[0, 1, 2], 0, 1.0, 10.0, None);
        assert!(!scorer.check_phrase_positions(0));
        assert!(
            scorer.position_bufs[1].is_empty(),
            "the original first term must reject before a less selective middle term is read"
        );
        assert_eq!(
            scorer.position_bufs[0],
            [4, 14],
            "filtering must not mutate the original first-term occurrences"
        );
    }

    #[test]
    fn selective_phrase_anchors_preserve_original_multiplicity_and_extreme_offsets() {
        use crate::structures::{PositionStreamEncoder, PostingCodec, PostingList};
        let values = [
            vec![0, 0, 5, 20, u32::MAX - 8, u32::MAX - 8],
            vec![1, 6, 21, u32::MAX - 7],
            vec![0, 3, 8, u32::MAX - 5],
            vec![0, 8, 13, u32::MAX],
        ];
        let offsets = [0, 1, 3, 8];
        let expected = values[0]
            .iter()
            .filter(|&&start| {
                (1..values.len()).all(|term| {
                    values[term]
                        .iter()
                        .any(|&p| u64::from(p) == u64::from(start) + u64::from(offsets[term]))
                })
            })
            .count() as u32;
        assert_eq!(expected, 5);
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            for rare in 0..values.len() {
                let mut lists = Vec::new();
                let mut positions = Vec::new();
                for (term, first_values) in values.iter().enumerate() {
                    let mut list = PostingList::new();
                    let mut bytes = Vec::new();
                    let mut encoder = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
                    let docs = if term == rare { 1 } else { 3 + term as u32 };
                    for doc in 0..docs {
                        let mut positions = if doc == 0 {
                            first_values.clone()
                        } else {
                            vec![offsets[term]]
                        };
                        list.push(doc, positions.len() as u32);
                        encoder.push_doc(&mut positions).unwrap();
                    }
                    encoder.finish().unwrap();
                    lists.push(
                        BlockPostingList::from_posting_list_with_options(&list, true, None, codec)
                            .unwrap(),
                    );
                    positions.push(
                        TermPositions::open(crate::directories::OwnedBytes::new(bytes)).unwrap(),
                    );
                }
                let mut scorer =
                    PhraseScorer::unpositioned(lists, positions, &offsets, 0, 1.0, 10.0, None);
                assert!(
                    scorer.check_phrase_positions(0),
                    "codec {codec:?}, rare {rare}"
                );
                assert!(scorer.frequency.get().is_none());
                assert_eq!(
                    scorer.exact_phrase_frequency(),
                    expected,
                    "codec {codec:?}, rare {rare}"
                );
                let doc_len = values.iter().map(|p| p.len() as f32).sum::<f32>();
                assert_eq!(
                    scorer.score().to_bits(),
                    super::super::Bm25Params::default()
                        .score(expected as f32, 1.0, doc_len, 10.0)
                        .to_bits()
                );
                assert_eq!(scorer.exact_phrase_frequency(), expected);
            }
        }
    }

    #[test]
    fn selective_phrase_filters_avoid_reading_frequent_first_term_positions() {
        use crate::structures::{PositionStreamEncoder, PostingCodec, PostingList};
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let mut lists = Vec::new();
            let mut positions = Vec::new();
            for (term, docs) in [16, 2, 1].into_iter().enumerate() {
                let mut list = PostingList::new();
                let mut bytes = Vec::new();
                let mut encoder = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
                for doc in 0..docs {
                    let mut values = match term {
                        0 => (0..512).collect::<Vec<u32>>(),
                        1 => vec![4, 14],
                        _ => vec![2, 12],
                    };
                    list.push(doc, values.len() as u32);
                    encoder.push_doc(&mut values).unwrap();
                }
                encoder.finish().unwrap();
                lists.push(
                    BlockPostingList::from_posting_list_with_options(&list, true, None, codec)
                        .unwrap(),
                );
                positions
                    .push(TermPositions::open(crate::directories::OwnedBytes::new(bytes)).unwrap());
            }
            let mut scorer =
                PhraseScorer::unpositioned(lists, positions, &[0, 1, 2], 0, 1.0, 10.0, None);
            assert!(!scorer.check_phrase_positions(0));
            assert!(
                scorer.position_bufs[0].is_empty(),
                "an impossible rare-term intersection must reject before reading the frequent first term"
            );
            assert_eq!(scorer.exact_phrase_frequency(), 0);
        }
    }

    #[test]
    fn exact_phrase_rejects_empty_prefix_before_reading_later_position_payloads() {
        use crate::structures::{PositionStreamEncoder, PostingCodec, PostingList};
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let values = [
                [vec![0, 10], vec![0, 10, 20]],
                [vec![4, 14], vec![1, 11, 21]],
                [vec![2, 12], vec![2, 12, 22]],
            ];
            let mut lists = Vec::new();
            let mut positions = Vec::new();
            for docs in &values {
                let mut list = PostingList::new();
                let mut bytes = Vec::new();
                let mut encoder = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
                for (doc, values) in docs.iter().enumerate() {
                    list.push(doc as u32, values.len() as u32);
                    encoder.push_doc(&mut values.clone()).unwrap();
                }
                encoder.finish().unwrap();
                lists.push(
                    BlockPostingList::from_posting_list_with_options(&list, true, None, codec)
                        .unwrap(),
                );
                positions
                    .push(TermPositions::open(crate::directories::OwnedBytes::new(bytes)).unwrap());
            }
            let mut scorer =
                PhraseScorer::unpositioned(lists, positions, &[0, 1, 2], 0, 1.0, 10.0, None);
            assert!(!scorer.check_phrase_positions(0));
            assert!(
                scorer.position_bufs[2].is_empty(),
                "a failed prefix must not read a later term's positions"
            );
            assert_eq!(scorer.exact_phrase_frequency(), 0);
            for cursor in &mut scorer.posting_iters {
                assert_eq!(cursor.seek(1), 1);
            }
            assert!(scorer.check_phrase_positions(1));
            scorer.current_doc = 1;
            assert_eq!(scorer.exact_phrase_frequency(), 3);
            assert_eq!(
                scorer.score().to_bits(),
                super::super::Bm25Params::default()
                    .score(3.0, 1.0, 9.0, 10.0)
                    .to_bits()
            );
            assert_eq!(scorer.exact_phrase_frequency(), 3);
        }
    }

    #[test]
    fn exact_phrase_intersections_preserve_first_start_multiplicity_offsets_and_resume() {
        for terms in 2..10 {
            for seed in 0..64u32 {
                let mut bufs: Vec<Vec<u32>> = (0..terms)
                    .map(|term| {
                        let mut values: Vec<_> = (0..256u32)
                            .filter(|p| (p * (term as u32 + 3) + seed * 17) % (term as u32 + 4) < 2)
                            .map(|p| p * 3 + seed % 4)
                            .collect();
                        values.extend([u32::MAX - 3, u32::MAX, u32::MAX]);
                        if term == 0 && seed.is_multiple_of(3) {
                            values = values.into_iter().flat_map(|p| [p, p]).collect();
                        }
                        if seed.is_multiple_of(17) {
                            values.clear();
                        }
                        values
                    })
                    .collect();
                let deltas: Vec<_> = (0..terms)
                    .map(|i| match seed % 5 {
                        0 => 0,
                        1 => u32::MAX,
                        _ => i as u32 * 2,
                    })
                    .collect();
                let expected: Vec<_> = bufs[0]
                    .iter()
                    .copied()
                    .filter(|&start| {
                        (1..terms).all(|i| {
                            bufs[i]
                                .iter()
                                .any(|&p| u64::from(p) == u64::from(start) + u64::from(deltas[i]))
                        })
                    })
                    .collect();
                let mut order: Vec<_> = (1..terms).collect();
                if seed.is_multiple_of(2) {
                    order.reverse();
                }
                for &term in &order[..order.len() - 1] {
                    let (left, right) = bufs.split_at_mut(term);
                    retain_exact_phrase_starts(&mut left[0], &right[0], deltas[term]);
                }
                let last = *order.last().unwrap();
                let mut next = 0;
                let mut position = 0;
                let first = next_exact_phrase_match(
                    &bufs[0],
                    &bufs[last],
                    deltas[last],
                    &mut next,
                    &mut position,
                );
                let mut actual = Vec::new();
                if let Some(index) = first {
                    actual.push(bufs[0][index]);
                }
                let mut resumed_next = next;
                let mut resumed_position = position;
                while let Some(index) = next_exact_phrase_match(
                    &bufs[0],
                    &bufs[last],
                    deltas[last],
                    &mut resumed_next,
                    &mut resumed_position,
                ) {
                    actual.push(bufs[0][index]);
                }
                assert_eq!(actual, expected, "terms={terms},seed={seed}");
            }
        }
    }

    #[test]
    fn incremental_exact_phrase_scoring_matches_occurrences_across_codecs_and_false_candidates() {
        use crate::structures::{PositionStreamEncoder, PostingCodec, PostingList};
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            for terms in [2, 3, 5, 9] {
                let offsets: Vec<_> = (0..terms).map(|term| (term / 2) as u32 * 2).collect();
                let mut all_values = Vec::new();
                let mut lists = Vec::new();
                let mut positions = Vec::new();
                for (term, &offset) in offsets.iter().enumerate() {
                    let mut list = PostingList::new();
                    let mut bytes = Vec::new();
                    let mut encoder = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
                    let mut docs = Vec::new();
                    for doc in 0..260u32 {
                        let values: Vec<_> = (0..(129 + doc % 19))
                            .filter(|p| term == 0 || (p + doc) % (term as u32 + 2) != 0)
                            .map(|p| {
                                p * 32
                                    + doc % 3
                                    + offset
                                    + u32::from(doc.is_multiple_of(5) && term == 1)
                            })
                            .collect();
                        list.push(doc * 17, values.len() as u32);
                        encoder.push_doc(&mut values.clone()).unwrap();
                        docs.push(values);
                    }
                    encoder.finish().unwrap();
                    lists.push(
                        BlockPostingList::from_posting_list_with_options(&list, true, None, codec)
                            .unwrap(),
                    );
                    positions.push(
                        TermPositions::open(crate::directories::OwnedBytes::new(bytes)).unwrap(),
                    );
                    all_values.push(docs);
                }
                let mut scorer =
                    PhraseScorer::unpositioned(lists, positions, &offsets, 0, 1.5, 97.0, None);
                for doc in 0..260usize {
                    let target = doc as u32 * 17;
                    for cursor in &mut scorer.posting_iters {
                        assert_eq!(cursor.seek(target), target);
                    }
                    let expected_tf = all_values[0][doc]
                        .iter()
                        .filter(|&&start| {
                            (1..terms).all(|term| {
                                all_values[term][doc].contains(&(start + offsets[term]))
                            })
                        })
                        .count() as u32;
                    assert_eq!(
                        scorer.check_phrase_positions(target),
                        expected_tf > 0,
                        "codec={codec:?},terms={terms},doc={doc}"
                    );
                    assert_eq!(
                        scorer.exact_phrase_frequency(),
                        expected_tf,
                        "codec={codec:?},terms={terms},doc={doc}"
                    );
                    scorer.current_doc = target;
                    if expected_tf > 0 {
                        let length = all_values.iter().map(|docs| docs[doc].len() as f32).sum();
                        let expected = super::super::Bm25Params::default().score(
                            expected_tf as f32,
                            1.5,
                            length,
                            97.0,
                        );
                        assert_eq!(scorer.score().to_bits(), expected.to_bits());
                        assert_eq!(scorer.score().to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn point_phrase_backfill_invalidates_frequency_when_position_buffers_change() {
        use crate::structures::{PositionStreamEncoder, PostingList};
        let docs = [(0, 1), (50, 3), (99, 2), (240, 5)];
        let mut lists = Vec::new();
        let mut positions = Vec::new();
        for term in 0..2 {
            let mut list = PostingList::new();
            let mut bytes = Vec::new();
            let mut encoder = PositionStreamEncoder::new(&mut bytes);
            for &(doc, tf) in &docs {
                list.push(doc, tf);
                let mut values: Vec<_> = (0..tf).map(|i| i * 4 + term).collect();
                encoder.push_doc(&mut values).unwrap();
            }
            encoder.finish().unwrap();
            lists.push(BlockPostingList::from_posting_list_with(&list, true, None).unwrap());
            positions
                .push(TermPositions::open(crate::directories::OwnedBytes::new(bytes)).unwrap());
        }
        let mut scorer = PhraseScorer::unpositioned(lists, positions, &[0, 1], 0, 1.0, 10.0, None);
        for target in [0, 10, 50, 99, 99, 240, 999] {
            // Point backfill parks each term only on a nominated target; it does
            // not expand a missing target into the next posting intersection.
            let mut matches = true;
            for cursor in &mut scorer.posting_iters {
                matches &= cursor.seek(target) == target;
            }
            if matches {
                assert!(scorer.check_phrase_positions(target));
                scorer.current_doc = target;
                let tf = docs.iter().find(|e| e.0 == target).unwrap().1 as f32;
                let expected = super::super::Bm25Params::default().score(tf, 1.0, 2.0 * tf, 10.0);
                assert_eq!(
                    scorer.score().to_bits(),
                    expected.to_bits(),
                    "target={target}"
                );
            } else {
                assert!(!docs.iter().any(|e| e.0 == target));
            }
        }
    }

    #[test]
    fn phrase_frequency_resumes_after_first_match_without_recounting_prefixes() {
        for terms in 0..6 {
            for seed in 0..32u32 {
                let mut bufs: Vec<Vec<u32>> = (0..terms)
                    .map(|term| {
                        let mut positions: Vec<_> = (0..128u32)
                            .filter(|p| (p * (term as u32 + 3) + seed * 17) % (term as u32 + 4) < 2)
                            .map(|p| p * 3 + seed % 4)
                            .collect();
                        positions.extend([u32::MAX - 3, u32::MAX]);
                        positions
                    })
                    .collect();
                if terms > 0 && seed.is_multiple_of(13) {
                    bufs[0].clear();
                }
                if terms > 2 && seed.is_multiple_of(17) {
                    bufs[terms - 1].clear();
                }
                let deltas: Vec<_> = (0..terms)
                    .map(|i| {
                        if seed.is_multiple_of(5) {
                            0
                        } else {
                            i as u32 * 2
                        }
                    })
                    .collect();
                for slop in [0, 1, 5, u32::MAX] {
                    let expected = bufs.first().map_or(0, |first| {
                        first
                            .iter()
                            .filter(|&&start| {
                                (1..bufs.len()).all(|i| {
                                    let target = u64::from(start) + u64::from(deltas[i]);
                                    let low = target.saturating_sub(u64::from(slop));
                                    let high = target + u64::from(slop);
                                    bufs[i]
                                        .iter()
                                        .any(|&p| (low..=high).contains(&u64::from(p)))
                                })
                            })
                            .count() as u32
                    });
                    let mut indices = vec![0; terms];
                    let mut next = 0;
                    assert_eq!(
                        scan_phrase_matches(&bufs, &deltas, slop, &mut indices, &mut next, 0),
                        0
                    );
                    assert_eq!(next, 0);
                    let first =
                        scan_phrase_matches(&bufs, &deltas, slop, &mut indices, &mut next, 1);
                    assert_eq!(first, u32::from(expected > 0));
                    let mut resumed = indices.clone();
                    let mut resumed_start = next;
                    let remaining = scan_phrase_matches(
                        &bufs,
                        &deltas,
                        slop,
                        &mut resumed,
                        &mut resumed_start,
                        u32::MAX,
                    );
                    assert_eq!(
                        first + remaining,
                        expected,
                        "terms={terms},seed={seed},slop={slop}"
                    );
                    assert_eq!(
                        count_phrase_matches(&bufs, &deltas, slop, &mut indices),
                        expected
                    );
                }
            }
        }
    }

    struct ChunkHits {
        hits: Vec<(u32, f32)>,
        at: usize,
        advances: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl DocSet for ChunkHits {
        fn doc(&self) -> DocId {
            self.hits.get(self.at).map_or(TERMINATED, |h| h.0)
        }
        fn advance(&mut self) -> DocId {
            self.advances
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.at = (self.at + 1).min(self.hits.len());
            self.doc()
        }
        fn seek(&mut self, target: DocId) -> DocId {
            self.at += self.hits[self.at..].partition_point(|h| h.0 < target);
            self.doc()
        }
        fn size_hint(&self) -> u32 {
            (self.hits.len() - self.at) as u32
        }
    }
    impl Scorer for ChunkHits {
        fn score(&self) -> Score {
            self.hits.get(self.at).map_or(0.0, |h| h.1)
        }
    }

    fn test_chunk_map(owners: &[(u32, u16)]) -> crate::segment::chunk_map::ChunkMap {
        use crate::segment::chunk_map::{ChunkMapBuilder, read_chunk_maps, write_chunk_maps};
        let mut builder = ChunkMapBuilder::default();
        for &(doc, ordinal) in owners {
            builder.push(doc, ordinal, 10).unwrap();
        }
        let mut bytes = Vec::new();
        write_chunk_maps(&mut bytes, &[(0, &builder)], &[]).unwrap();
        read_chunk_maps(crate::directories::OwnedBytes::new(bytes))
            .unwrap()
            .chunk_maps
            .remove(&0)
            .unwrap()
    }

    fn chunk_hits(hits: Vec<(u32, f32)>) -> ChunkHits {
        ChunkHits {
            hits,
            at: 0,
            advances: Arc::default(),
        }
    }

    #[test]
    fn lazy_phrase_fold_matches_stable_eager_oracle_including_reordered_ordinals() {
        for owners in [
            vec![],
            vec![(0, 0)],
            vec![(2, 2), (2, 0), (2, 1), (5, 1), (5, 0), (9, 0)],
            vec![(5, 1), (2, 2), (9, 0), (2, 0), (5, 0), (2, 1)],
        ] {
            let map = test_chunk_map(&owners);
            assert_eq!(
                map.is_doc_ordered(),
                owners.windows(2).all(|p| p[0].0 <= p[1].0)
            );
            for stride in [1, 2, 3] {
                let hits: Vec<_> = (0..owners.len() as u32)
                    .step_by(stride)
                    .map(|vid| (vid, (vid % 3) as f32 * 0.5))
                    .collect();
                let raw: Vec<_> = hits
                    .iter()
                    .map(|&(vid, score)| {
                        let (doc, ord) = map.resolve(vid);
                        (doc, ord, score)
                    })
                    .collect();
                let expected = crate::segment::combine_ordinal_results(
                    raw,
                    super::super::MultiValueCombiner::Max,
                    usize::MAX,
                );
                let mut expected = super::super::vector::VectorResultScorer::new(expected, 7);
                let mut actual = fold_chunked_phrase_scorer(chunk_hits(hits), map.clone(), 7, None);
                while expected.doc() != TERMINATED {
                    assert_eq!(actual.doc(), expected.doc());
                    assert_eq!(actual.score().to_bits(), expected.score().to_bits());
                    let signature = |s: &dyn Scorer| {
                        s.matched_positions()
                            .unwrap()
                            .into_iter()
                            .map(|(field, positions)| {
                                (
                                    field,
                                    positions
                                        .into_iter()
                                        .map(|p| (p.position, p.score.to_bits()))
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(signature(actual.as_ref()), signature(&expected));
                    actual.advance();
                    expected.advance();
                }
                assert_eq!(actual.doc(), TERMINATED);
                assert_eq!(actual.advance(), TERMINATED);
                assert_eq!(actual.score(), 0.0);
            }
        }
    }

    #[test]
    fn lazy_phrase_fold_only_consumes_one_document_and_can_skip_to_late_matches() {
        let owners: Vec<_> = (0..100)
            .flat_map(|doc| [(doc * 2, 0), (doc * 2, 1)])
            .collect();
        let inner = chunk_hits((0..200).map(|vid| (vid, vid as f32)).collect());
        let advances = inner.advances.clone();
        let mut scorer = fold_chunked_phrase_scorer(inner, test_chunk_map(&owners), 0, None);
        assert_eq!(advances.load(std::sync::atomic::Ordering::Relaxed), 2);
        assert_eq!(scorer.doc(), 0);
        assert_eq!(scorer.seek(179), 180);
        assert_eq!(advances.load(std::sync::atomic::Ordering::Relaxed), 4);
        assert_eq!(scorer.score(), 181.0);
        assert_eq!(scorer.seek(179), 180);
        assert_eq!(scorer.seek(199), TERMINATED);
        assert!(scorer.matched_positions().is_none());
        assert_eq!(scorer.advance(), TERMINATED);
    }

    #[test]
    fn lazy_phrase_fold_discards_current_result_at_budget_boundary() {
        let inner = chunk_hits(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let advances = inner.advances.clone();
        let mut scorer = ChunkedPhraseScorer {
            inner,
            chunk_map: test_chunk_map(&[(0, 0), (1, 0), (1, 1)]),
            field_id: 0,
            budget: None,
            current_doc: TERMINATED,
            score: 0.0,
            ordinals: crate::segment::VectorOrdinals::new(),
        };
        assert_eq!(scorer.fold_next_document(), 0);
        assert_eq!(scorer.score(), 1.0);
        let budget = super::super::SharedThreshold::for_limit(1)
            .with_deadline(Some(std::time::Instant::now()));
        scorer.budget = Some(budget.clone());
        assert_eq!(scorer.advance(), TERMINATED);
        assert_eq!(scorer.score(), 0.0);
        assert!(scorer.matched_positions().is_none());
        assert!(budget.truncated());
        assert_eq!(advances.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn phrase_alignment_preserves_offsets_counts_and_seeks_with_each_term_rarest() {
        use crate::structures::{PositionStreamEncoder, PostingList};

        for rare in 0..3 {
            let present = |term, doc: u32| {
                if term == rare {
                    doc.is_multiple_of(7)
                } else {
                    !doc.is_multiple_of(5 + term as u32)
                }
            };
            let offsets = [0, 3, 8];
            let mut lists = Vec::new();
            let mut positions = Vec::new();
            for (term, &offset) in offsets.iter().enumerate() {
                let mut list = PostingList::new();
                let mut bytes = Vec::new();
                let mut encoder = PositionStreamEncoder::new(&mut bytes);
                for doc in 0..5000 {
                    if !present(term, doc) {
                        continue;
                    }
                    list.push(doc, 2);
                    let shift = if term == 2 && doc.is_multiple_of(11) {
                        100
                    } else {
                        0
                    };
                    encoder
                        .push_doc(&mut [offset + shift, 20 + offset + shift])
                        .unwrap();
                }
                encoder.finish().unwrap();
                lists.push(BlockPostingList::from_posting_list_with(&list, true, None).unwrap());
                positions
                    .push(TermPositions::open(crate::directories::OwnedBytes::new(bytes)).unwrap());
            }
            let candidates: Vec<_> = (0..5000)
                .filter(|&doc| (0..3).all(|term| present(term, doc)))
                .collect();
            let expected: Vec<_> = candidates
                .iter()
                .copied()
                .filter(|doc| !doc.is_multiple_of(11))
                .collect();
            let make = || {
                PhraseScorer::unpositioned(
                    lists.clone(),
                    positions.clone(),
                    &offsets,
                    0,
                    1.0,
                    10.0,
                    None,
                )
            };
            let score = super::super::Bm25Params::default().score(2.0, 1.0, 6.0, 10.0);
            let mut scorer = make();
            scorer.find_next_phrase_match();
            for &doc in &expected {
                assert_eq!(scorer.doc(), doc);
                assert!(scorer.frequency.get().is_none());
                assert_eq!(
                    scorer.next_start, 1,
                    "membership stops after the first occurrence"
                );
                #[cfg(feature = "native")]
                if doc == expected[0] {
                    std::thread::scope(|scope| {
                        for _ in 0..4 {
                            let shared = &scorer;
                            scope.spawn(move || {
                                assert_eq!(shared.score().to_bits(), score.to_bits())
                            });
                        }
                    });
                }
                assert_eq!(scorer.exact_phrase_frequency(), 2);
                assert_eq!(scorer.score().to_bits(), score.to_bits());
                scorer.advance();
            }
            assert_eq!(scorer.doc(), TERMINATED);
            assert_eq!(scorer.advance(), TERMINATED);

            let mut scorer = make();
            scorer.find_next_candidate();
            for &doc in &candidates {
                assert_eq!(scorer.doc(), doc);
                assert_eq!(scorer.confirm_candidate(), !doc.is_multiple_of(11));
                assert!(
                    scorer.frequency.get().is_none(),
                    "membership does not finish phrase frequency"
                );
                scorer.advance_candidate();
            }
            assert_eq!(scorer.doc(), TERMINATED);

            let mut scorer = make();
            scorer.find_next_phrase_match();
            for target in [0, 1, 127, 127, 2000, 4096, 4999, TERMINATED] {
                let next = expected
                    .iter()
                    .copied()
                    .find(|&doc| doc >= target)
                    .unwrap_or(TERMINATED);
                assert_eq!(scorer.seek(target), next);
            }
            assert_eq!(scorer.seek(0), TERMINATED);
        }
    }

    #[test]
    fn phrase_candidates_require_confirmation_and_exact_seek_stays_terminal() {
        use crate::structures::{PositionStreamEncoder, PostingList};
        let mut lists = Vec::new();
        let mut positions = Vec::new();
        for term in 0..2 {
            let mut list = PostingList::new();
            let mut bytes = Vec::new();
            let mut encoder = PositionStreamEncoder::new(&mut bytes);
            for doc in 0..1000 {
                list.push(doc, 1);
                encoder
                    .push_doc(&mut [if [0, 100, 999].contains(&doc) {
                        term
                    } else {
                        term * 10
                    }])
                    .unwrap();
            }
            encoder.finish().unwrap();
            lists.push(BlockPostingList::from_posting_list_with(&list, true, None).unwrap());
            positions
                .push(TermPositions::open(crate::directories::OwnedBytes::new(bytes)).unwrap());
        }
        let mut expired = PhraseScorer::unpositioned(
            lists.clone(),
            positions.clone(),
            &[0, 1],
            0,
            1.0,
            2.0,
            None,
        );
        let mut scorer = PhraseScorer::unpositioned(lists, positions, &[0, 1], 0, 1.0, 2.0, None);
        scorer.find_next_phrase_match();
        let score = scorer.score();
        assert_eq!(scorer.advance_candidate(), 1);
        assert_eq!(
            scorer.position_bufs,
            vec![vec![0], vec![1]],
            "candidate traversal does not decode positions"
        );
        assert!(!scorer.confirm_candidate());
        assert!(!scorer.confirm_candidate());
        assert_eq!(scorer.exact_phrase_frequency(), 0);
        assert_eq!(scorer.seek_candidate(100), 100);
        assert!(scorer.confirm_candidate());
        assert!(scorer.confirm_candidate());
        assert_eq!(scorer.score(), score);
        assert_eq!(scorer.seek(101), 999, "ordinary seek skips false positives");
        assert_eq!(scorer.score(), score);
        assert_eq!(expired.seek_candidate(100), 100);
        assert!(expired.confirm_candidate());
        let budget = super::super::SharedThreshold::for_limit(10)
            .with_deadline(Some(std::time::Instant::now()));
        expired.budget = Some(budget.clone());
        assert!(
            !expired.confirm_candidate(),
            "cancellation invalidates cached confirmation"
        );
        assert!(budget.truncated());
        assert_eq!(expired.doc(), TERMINATED);
        assert_eq!(expired.score(), 0.0);
        assert_eq!(expired.seek(0), TERMINATED);
        assert_eq!(scorer.seek(TERMINATED), TERMINATED);
        assert_eq!(
            scorer.seek(0),
            TERMINATED,
            "exhausted streams cannot resurrect"
        );
    }

    #[test]
    fn phrase_stops_when_budget_expires_after_first_match() {
        use super::super::docset::DocSet;
        use crate::structures::{PositionStreamEncoder, PostingList};
        let mut lists = Vec::new();
        let mut positions = Vec::new();
        for term in 0..2 {
            let mut list = PostingList::new();
            let mut bytes = Vec::new();
            let mut encoder = PositionStreamEncoder::new(&mut bytes);
            for doc in 0..1000 {
                list.push(doc, 1);
                encoder
                    .push_doc(&mut [if doc == 0 { term } else { term * 10 }])
                    .unwrap();
            }
            encoder.finish().unwrap();
            lists.push(BlockPostingList::from_posting_list_with(&list, true, None).unwrap());
            positions
                .push(TermPositions::open(crate::directories::OwnedBytes::new(bytes)).unwrap());
        }
        let mut scorer = PhraseScorer::unpositioned(lists, positions, &[0, 1], 0, 1.0, 2.0, None);
        scorer.find_next_phrase_match();
        assert_eq!(scorer.doc(), 0);
        let budget = super::super::SharedThreshold::for_limit(10)
            .with_deadline(Some(std::time::Instant::now()));
        scorer.budget = Some(budget.clone());
        assert_eq!(scorer.advance(), TERMINATED);
        assert_eq!(scorer.score(), 0.0);
        assert!(budget.truncated());
        assert_eq!(
            scorer.position_bufs,
            vec![vec![0], vec![1]],
            "no further positions decoded"
        );
    }

    #[test]
    fn monotone_phrase_frequency_matches_naive_offsets_slop_and_repeated_terms() {
        for seed in 0..100u32 {
            let a: Vec<_> = (0..200).filter(|i| (i * 17 + seed) % 11 < 5).collect();
            let b: Vec<_> = (0..200).filter(|i| (i * 13 + seed) % 19 < 4).collect();
            for bufs in [
                vec![a.clone(), b.clone()],
                vec![a.clone(), b.clone(), a.clone()],
            ] {
                for slop in [0, 1, 3, 100] {
                    let deltas = [0, 2, 7];
                    let expected = bufs[0]
                        .iter()
                        .filter(|&&start| {
                            bufs.iter().enumerate().skip(1).all(|(i, positions)| {
                                positions
                                    .iter()
                                    .any(|&p| p.abs_diff(start + deltas[i]) <= slop)
                            })
                        })
                        .count() as u32;
                    assert_eq!(
                        count_phrase_matches(&bufs, &deltas, slop, &mut [0; 3]),
                        expected
                    );
                }
            }
        }
        assert_eq!(
            count_phrase_matches(&[vec![0, 0, 5], vec![1, 6]], &[0, 1], 0, &mut [0; 2]),
            3
        );
        assert_eq!(
            count_phrase_matches(&[vec![u32::MAX], vec![0]], &[0, 1], 0, &mut [0; 2]),
            0
        );
        assert_eq!(
            count_phrase_matches(&[vec![1], vec![]], &[0, 1], 3, &mut [0; 2]),
            0
        );
    }
}
