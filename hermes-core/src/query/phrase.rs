//! Phrase query - matches documents containing terms in consecutive positions

use std::sync::Arc;

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::structures::{BlockPostingIterator, BlockPostingList, TERMINATED, TermPositions};
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

/// Phrase over a chunked field: match and score every chunk (posting ids are
/// virtual chunk ids, positions restart per chunk so a phrase never spans two
/// chunks), then fold the chunk hits into documents with per-ordinal scores.
///
/// The phrase is a conjunction, so draining the positional scorer costs one
/// pass over the matching chunks only.
fn build_chunked_phrase_scorer<'a>(
    term_data: Vec<(BlockPostingList, TermPositions)>,
    offsets: &[u32],
    slop: u32,
    reader: &SegmentReader,
    field: Field,
    limit: usize,
) -> crate::Result<Box<dyn Scorer + 'a>> {
    let Some(chunk_map) = reader.chunk_map(field) else {
        return Err(crate::Error::Corruption(format!(
            "chunked text field '{}' has postings but segment {:016x} carries no chunk map",
            reader.schema().get_field_name(field).unwrap_or("?"),
            reader.meta().id,
        )));
    };
    let num_chunks = chunk_map.num_chunks() as f32;
    let idf: f32 = term_data
        .iter()
        .map(|(p, _)| super::bm25_idf(p.doc_count() as f32, num_chunks))
        .sum();
    let (postings, positions): (Vec<_>, Vec<_>) = term_data.into_iter().unzip();
    let mut scorer =
        PhraseScorer::new(postings, positions, offsets, slop, idf, chunk_map.avg_len())
            .with_lengths(Lengths::Chunks(chunk_map.clone()))
            .with_params(super::Bm25Params::for_field(reader.schema(), field));

    use super::docset::DocSet as _;
    let mut raw: Vec<(u32, u16, f32)> = Vec::new();
    while scorer.doc() != TERMINATED {
        let (doc_id, ordinal) = chunk_map.resolve(scorer.doc());
        raw.push((doc_id, ordinal, scorer.score()));
        scorer.advance();
    }
    // Every matching document is kept: a phrase is also used as a MUST
    // constraint (verifier or bitset), where truncating to `limit` would
    // silently reject documents that do contain the phrase.
    let _ = limit;
    let combined =
        crate::segment::combine_ordinal_results(raw, super::MultiValueCombiner::Max, usize::MAX);
    Ok(Box::new(super::vector::VectorResultScorer::new(combined, field.0)) as Box<dyn Scorer + 'a>)
}

/// Build a PhraseScorer from already-fetched term data.
fn build_phrase_scorer<'a>(
    term_data: Vec<(BlockPostingList, TermPositions)>,
    offsets: &[u32],
    slop: u32,
    reader: &SegmentReader,
    field: Field,
) -> Box<dyn Scorer + 'a> {
    let idf: f32 = term_data
        .iter()
        .map(|(p, _)| {
            let num_docs = reader.num_docs() as f32;
            let doc_freq = p.doc_count() as f32;
            super::bm25_idf(doc_freq, num_docs)
        })
        .sum();
    let avg_field_len = reader.avg_field_len(field);
    let (postings, positions): (Vec<_>, Vec<_>) = term_data.into_iter().unzip();
    let mut scorer = PhraseScorer::new(postings, positions, offsets, slop, idf, avg_field_len)
        .with_params(super::Bm25Params::for_field(reader.schema(), field));
    if let Some(lengths) = reader.doc_lengths(field) {
        scorer = scorer.with_lengths(Lengths::Docs(lengths.clone()));
    }
    Box::new(scorer)
}

// ── Shared early-return checks for phrase scorer ─────────────────────────
//
// Handles: empty terms, single-term delegation, no-positions fallback.
// Parameterised on the option-aware scorer function plus async/sync awaiting.
macro_rules! phrase_early_returns {
    ($field:expr, $terms:expr, $reader:expr, $limit:expr,
     $scorer_fn:ident, $options:expr $(, $aw:tt)*) => {
        if $terms.is_empty() {
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

            if reader.is_chunked_field(field) {
                return build_chunked_phrase_scorer(
                    term_data, &offsets, slop, reader, field, limit,
                );
            }
            Ok(build_phrase_scorer(
                term_data, &offsets, slop, reader, field,
            ))
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

        if reader.is_chunked_field(self.field) {
            return build_chunked_phrase_scorer(
                term_data,
                &self.offsets,
                self.slop,
                reader,
                self.field,
                limit,
            );
        }
        Ok(build_phrase_scorer(
            term_data,
            &self.offsets,
            self.slop,
            reader,
            self.field,
        ))
    }

    /// Every document containing the phrase, as a bitset (documents, also
    /// for chunked fields). Lets the planner push a quoted span into the
    /// MaxScore executors as an O(1) predicate instead of a verifier.
    #[cfg(feature = "sync")]
    fn as_doc_bitset(&self, reader: &SegmentReader) -> Option<super::DocBitset> {
        if self.terms.is_empty() {
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
                let doc = chunk_map.map_or(it.doc(), |map| map.doc_id(it.doc()));
                bitset.set(doc);
                it.advance();
            }
            return Some(bitset);
        }
        let mut scorer = self
            .scorer_sync_with_options(reader, usize::MAX, super::ScorerOptions::with_positions())
            .ok()?;
        while scorer.doc() != TERMINATED {
            bitset.set(scorer.doc());
            scorer.advance();
        }
        Some(bitset)
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
            Lengths::Chunks(map) => map.length(id),
            Lengths::Docs(lengths) => lengths.length(id),
        }
    }
}

/// Scorer that checks phrase positions
struct PhraseScorer {
    /// Posting iterators for each term
    posting_iters: Vec<BlockPostingIterator<'static>>,
    /// Positions of each term (legacy list or cursor-addressed stream)
    position_lists: Vec<TermPositions>,
    /// One decoded position block, reused across documents and terms
    position_scratch: Vec<u32>,
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
    fn new(
        posting_lists: Vec<BlockPostingList>,
        position_lists: Vec<TermPositions>,
        offsets: &[u32],
        slop: u32,
        idf: f32,
        avg_field_len: f32,
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
        let mut scorer = Self {
            posting_iters,
            position_lists,
            position_scratch: Vec::new(),
            deltas,
            slop,
            current_doc: 0,
            current_matches: 0,
            params: super::Bm25Params::default(),
            idf,
            avg_field_len,
            lengths: None,
            position_bufs: (0..num_terms).map(|_| Vec::new()).collect(),
        };

        scorer.find_next_phrase_match();
        scorer
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
            if !self.position_lists[i].positions_into(
                doc_id,
                cursor,
                tf,
                &mut self.position_scratch,
                &mut self.position_bufs[i],
            ) {
                return false;
            }
        }

        // Count the occurrences: every position of the first term that
        // starts a full match. The count is the phrase frequency BM25 scores.
        self.current_matches = self.count_phrase_matches_in_bufs();
        self.current_matches > 0
    }

    /// Number of phrase occurrences in the internal reusable buffers.
    fn count_phrase_matches_in_bufs(&self) -> u32 {
        if self.position_bufs.is_empty() || self.position_bufs[0].is_empty() {
            return 0;
        }
        self.position_bufs[0]
            .iter()
            .filter(|&&first_pos| self.check_phrase_from_position(first_pos, &self.position_bufs))
            .count() as u32
    }

    /// Check if a phrase exists starting from the given position
    fn check_phrase_from_position(&self, start_pos: u32, term_positions: &[Vec<u32>]) -> bool {
        for (i, positions) in term_positions.iter().enumerate() {
            if i == 0 {
                continue; // Skip first term, already matched
            }

            let expected_pos = start_pos + self.deltas[i];

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
