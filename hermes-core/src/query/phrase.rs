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
fn fold_chunked_phrase_scorer<'a, S: Scorer + 'a>(
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
    if reader.is_chunked_field(field) {
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
) -> crate::Result<Box<dyn Scorer + 'a>> {
    scorer.find_next_phrase_match();
    if let Some(map) = reader.chunk_map(field) {
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
    let stats = query.global_stats.as_ref().or(stats);
    if query.terms.len() == 1 {
        return super::term::score_term_candidates(
            reader,
            query.field,
            &[(query.terms[0].clone(), 1.0)],
            targets,
            stats,
        )
        .await;
    }
    let mut scores = vec![0.0; targets.len()];
    if query.terms.is_empty() {
        return Ok(scores);
    }
    if !reader.has_positions(query.field) {
        return Err(crate::Error::Query(
            "phrase score backfill requires positions".into(),
        ));
    }
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
    Ok(scores)
}

// ── Shared early-return checks for phrase scorer ─────────────────────────
//
// Handles: empty terms, single-term delegation, no-positions fallback.
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
        if !$reader.has_positions($field) {
            let mut bq = super::BooleanQuery::new();
            for t in $terms.iter() {
                bq = bq.must(super::TermQuery::new($field, t.clone()));
            }
            return bq.$scorer_fn($reader, $limit, $options) $(. $aw)* ;
        }
    };
}

impl Query for PhraseQuery {
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
            finish_phrase_scorer(scorer, reader, field, options.shared_threshold)
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
        finish_phrase_scorer(scorer, reader, self.field, options.shared_threshold)
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
            // A one-term phrase is the term itself; walk its postings and
            // resolve chunk ids to documents where needed.
            let list = reader
                .get_postings_sync(self.field, &self.terms[0])
                .ok()??;
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
    budget: Option<super::SharedThreshold>,
    /// Posting iterators for each term
    posting_iters: Vec<BlockPostingIterator<'static>>,
    /// Positions of each term (legacy list or cursor-addressed stream)
    position_lists: Vec<crate::structures::postings::TermPositionCursor>,
    /// Position-list cursors advance monotonically within a matching unit.
    position_indices: Vec<usize>,
    /// Required distance of each term from the first one (`offsets[i] -
    /// offsets[0]`); `deltas[0]` is 0.
    deltas: Vec<u32>,
    /// Max slop between terms
    slop: u32,
    /// Current matching document
    current_doc: DocId,
    /// Number of phrase occurrences in the current document (phrase
    /// frequency), the `tf` of the phrase for BM25.
    current_matches: u32,
    /// Combined IDF
    idf: f32,
    /// Per-field k1/b.
    params: super::Bm25Params,
    /// Average field length
    avg_field_len: f32,
    /// Real lengths of the scoring units. `None` keeps the historic
    /// `tf`-as-length approximation.
    lengths: Option<Lengths>,
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
            budget: budget.filter(|b| b.deadline().is_some()),
            posting_iters,
            position_lists: position_lists
                .into_iter()
                .map(TermPositions::into_cursor)
                .collect(),
            position_indices: vec![0; num_terms],
            deltas,
            slop,
            current_doc: 0,
            current_matches: 0,
            params: super::Bm25Params::default(),
            idf,
            avg_field_len,
            lengths: None,
            position_bufs: (0..num_terms).map(|_| Vec::new()).collect(),
        }
    }

    /// Score with the real length of each scoring unit.
    fn with_lengths(mut self, lengths: Lengths) -> Self {
        self.lengths = Some(lengths);
        self
    }

    /// Score with the field's BM25 parameters.
    fn with_params(mut self, params: super::Bm25Params) -> Self {
        self.params = params;
        self
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
            if self
                .budget
                .as_ref()
                .is_some_and(super::SharedThreshold::stop_if_expired)
            {
                return TERMINATED;
            }
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
    fn check_phrase_positions(&mut self, doc_id: DocId) -> bool {
        // Get positions for each term into reusable buffers (zero allocation).
        // The doc-posting iterator of every term is parked on `doc_id`, so
        // its cursor and term frequency address the term's position stream.
        for i in 0..self.position_lists.len() {
            let cursor = self.posting_iters[i].position_cursor();
            let tf = self.posting_iters[i].term_freq();
            if !self.position_lists[i].read_into(doc_id, cursor, tf, &mut self.position_bufs[i]) {
                return false;
            }
        }

        // Count the occurrences: every position of the first term that
        // starts a full match. The count is the phrase frequency BM25 scores.
        self.current_matches = self.count_phrase_matches_in_bufs();
        self.current_matches > 0
    }

    /// Number of phrase occurrences in the internal reusable buffers.
    fn count_phrase_matches_in_bufs(&mut self) -> u32 {
        count_phrase_matches(
            &self.position_bufs,
            &self.deltas,
            self.slop,
            &mut self.position_indices,
        )
    }
}

/// Count starts with a match in every term's independent slop interval.
/// Ascending starts make each interval monotone: O(terms * starts + positions),
/// preserving repeated terms and the existing (not edit-distance) slop rule.
fn count_phrase_matches(
    bufs: &[Vec<u32>],
    deltas: &[u32],
    slop: u32,
    indices: &mut [usize],
) -> u32 {
    let Some(first) = bufs.first() else {
        return 0;
    };
    indices.fill(0);
    let mut matches = 0;
    'starts: for &start in first {
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
    }
    matches
}

impl super::docset::DocSet for PhraseScorer {
    fn doc(&self) -> DocId {
        self.current_doc
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

impl Scorer for PhraseScorer {
    fn score(&self) -> Score {
        if self.current_doc == TERMINATED {
            return 0.0;
        }

        // BM25 over the phrase frequency with the summed idf of the terms
        // (Lucene semantics): a document with two occurrences of the phrase
        // outranks one with a single occurrence at equal length.
        let tf = self.current_matches.max(1) as f32;

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
