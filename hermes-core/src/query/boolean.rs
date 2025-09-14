//! Boolean query with MUST, SHOULD, and MUST_NOT clauses

use std::sync::Arc;

use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::{DocId, Score};

use super::planner::{
    build_combined_bitset, build_sparse_bmp_results, build_sparse_bmp_results_filtered,
    build_sparse_maxscore_executor, cap_terms, chain_predicates, combine_sparse_results,
    compute_idf, extract_all_sparse_infos, finish_chunked_text_maxscore, finish_text_maxscore,
    prepare_per_field_grouping, prepare_text_maxscore, text_maxscore_allowed,
};
use super::{CountFuture, EmptyScorer, GlobalStats, Query, Scorer, ScorerFuture};

/// Boolean query with MUST, SHOULD, and MUST_NOT clauses
///
/// When all clauses are SHOULD term queries on the same field, automatically
/// uses MaxScore optimization for efficient top-k retrieval.
#[derive(Clone)]
pub struct BooleanQuery {
    pub must: Vec<Arc<dyn Query>>,
    pub should: Vec<Arc<dyn Query>>,
    pub must_not: Vec<Arc<dyn Query>>,
    /// Optional global statistics for cross-segment IDF
    global_stats: Option<Arc<GlobalStats>>,
    /// Proximity rescoring of the text MaxScore result (SHOULD terms in
    /// query order); `None` = off.
    proximity: Option<super::ProximityConfig>,
    /// Approximate text MaxScore: threshold divided by `heap_factor`
    /// (< 1 prunes beyond rank safety, like sparse). 1.0 = exact.
    text_heap_factor: f32,
    /// Keep only the rarest `max_terms` SHOULD text terms of a field group
    /// (0 = all): long-query cap.
    max_terms: usize,
}

fn shared_or_extract_sparse_infos<'a>(
    plan: Option<&'a Arc<super::bmp::LspSegmentPlan>>,
    should: &[Arc<dyn Query>],
) -> Option<std::borrow::Cow<'a, [super::SparseTermQueryInfo]>> {
    plan.map(|plan| std::borrow::Cow::Borrowed(plan.infos.as_ref()))
        .or_else(|| extract_all_sparse_infos(should).map(std::borrow::Cow::Owned))
}

impl std::fmt::Debug for BooleanQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BooleanQuery")
            .field("must_count", &self.must.len())
            .field("should_count", &self.should.len())
            .field("must_not_count", &self.must_not.len())
            .field("has_global_stats", &self.global_stats.is_some())
            .field("proximity", &self.proximity)
            .finish()
    }
}

impl std::fmt::Display for BooleanQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Boolean(")?;
        let mut first = true;
        for q in &self.must {
            if !first {
                write!(f, " ")?;
            }
            write!(f, "+{}", q)?;
            first = false;
        }
        for q in &self.should {
            if !first {
                write!(f, " ")?;
            }
            write!(f, "{}", q)?;
            first = false;
        }
        for q in &self.must_not {
            if !first {
                write!(f, " ")?;
            }
            write!(f, "-{}", q)?;
            first = false;
        }
        if let Some(proximity) = &self.proximity {
            write!(f, " ~proximity({}, {})", proximity.weight, proximity.window)?;
        }
        if self.text_heap_factor < 1.0 {
            write!(f, " ~heap({})", self.text_heap_factor)?;
        }
        if self.max_terms > 0 {
            write!(f, " ~max_terms({})", self.max_terms)?;
        }
        write!(f, ")")
    }
}

impl Default for BooleanQuery {
    fn default() -> Self {
        Self {
            must: Vec::new(),
            should: Vec::new(),
            must_not: Vec::new(),
            global_stats: None,
            proximity: None,
            text_heap_factor: 1.0,
            max_terms: 0,
        }
    }
}

impl BooleanQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn must(mut self, query: impl Query + 'static) -> Self {
        self.must.push(Arc::new(query));
        self
    }

    pub fn should(mut self, query: impl Query + 'static) -> Self {
        self.should.push(Arc::new(query));
        self
    }

    pub fn must_not(mut self, query: impl Query + 'static) -> Self {
        self.must_not.push(Arc::new(query));
        self
    }

    /// Set global statistics for cross-segment IDF
    pub fn with_global_stats(mut self, stats: Arc<GlobalStats>) -> Self {
        self.global_stats = Some(stats);
        self
    }

    /// Rescore the text MaxScore top candidates with term proximity
    /// (`docs`: `query::proximity`). Applies when the SHOULD clauses are text
    /// terms of one field, in query order.
    pub fn with_proximity(mut self, config: super::ProximityConfig) -> Self {
        self.proximity = config.is_active().then_some(config);
        self
    }

    /// Approximate text MaxScore (threshold / `heap_factor`), like sparse.
    /// 1 is exact; [0, 1) prunes more aggressively, with an effective 0.01
    /// floor. Non-finite values and values outside [0, 1] fail construction
    /// of the scorer. RPC zero/unset is normalized to 1 by the adapter.
    pub fn with_text_heap_factor(mut self, heap_factor: f32) -> Self {
        self.text_heap_factor = heap_factor;
        self
    }

    /// Cap the text terms scored per field group to the `max_terms` rarest
    /// (highest idf) ones; 0 = no cap.
    pub fn with_max_terms(mut self, max_terms: usize) -> Self {
        self.max_terms = max_terms;
        self
    }
}

/// Flatten nested pure-SHOULD Boolean queries into one SHOULD list.
///
/// `OR(OR(a, b), c)` scores exactly like `OR(a, b, c)`, and only the flat
/// form reaches MaxScore and filter push-down. The nested form would be an
/// opaque sub-scorer whose top-k truncation can hide matches from the outer
/// query.
fn flatten_should(should: &[Arc<dyn Query>]) -> std::borrow::Cow<'_, [Arc<dyn Query>]> {
    if !should.iter().any(|query| query.should_children().is_some()) {
        return std::borrow::Cow::Borrowed(should);
    }

    fn push_flat(out: &mut Vec<Arc<dyn Query>>, query: &Arc<dyn Query>) {
        match query.should_children() {
            Some(children) => children.iter().for_each(|child| push_flat(out, child)),
            None => out.push(Arc::clone(query)),
        }
    }

    let mut flat = Vec::with_capacity(should.len());
    should.iter().for_each(|query| push_flat(&mut flat, query));
    std::borrow::Cow::Owned(flat)
}

/// Build a SHOULD-only scorer from a vec of optimized scorers.
fn build_should_scorer<'a>(scorers: Vec<Box<dyn Scorer + 'a>>) -> Box<dyn Scorer + 'a> {
    if scorers.is_empty() {
        return Box::new(EmptyScorer);
    }
    if scorers.len() == 1 {
        return scorers.into_iter().next().unwrap();
    }
    let mut scorer = BooleanScorer {
        must: vec![],
        should: scorers,
        must_not: vec![],
        current_doc: 0,
        lead: 0,
        doc_limit: 0,
    };
    scorer.initialize();
    Box::new(scorer)
}

// ── Planner macro ────────────────────────────────────────────────────────
//
// Unified planner for both async and sync paths.  Parameterised on:
//   $scorer_fn      – scorer_with_options | scorer_sync_with_options
//   $get_postings_fn – get_postings | get_postings_sync
//   $execute_fn     – execute | execute_sync
//   $($aw)*         – .await  (present for async, absent for sync)
//
// Decision order:
//   1. Single-clause unwrap
//   2. Pure OR → text MaxScore | sparse MaxScore | per-field MaxScore
//   3. Filter push-down → predicate-aware sparse MaxScore | PredicatedScorer
//   4. Standard BooleanScorer fallback
macro_rules! boolean_plan {
    ($must:expr, $should:expr, $must_not:expr, $global_stats:expr, $proximity:expr, $text_tuning:expr,
     $reader:expr, $limit:expr, $scorer_options:expr,
     $scorer_fn:ident, $get_postings_fn:ident, $execute_fn:ident
     $(, $aw:tt)*) => {{
        let must: &[Arc<dyn Query>] = &$must;
        let should_flat = flatten_should(&$should);
        let should_all: &[Arc<dyn Query>] = &should_flat;
        let must_not: &[Arc<dyn Query>] = &$must_not;
        let global_stats: Option<&Arc<GlobalStats>> = $global_stats;
        let reader: &SegmentReader = $reader;
        let limit: usize = $limit;
        let mut scorer_options: super::ScorerOptions = $scorer_options;
        // A Boolean node's resolved statistics apply to its complete child
        // streams too. Explicit statistics on a child still take precedence.
        scorer_options.global_stats = global_stats.cloned();
        if !$text_tuning.0.is_finite() || !(0.0..=1.0).contains(&$text_tuning.0) {
            return Err(crate::Error::Query(
                "Text heap_factor must be finite and between 0 and 1".into(),
            ));
        }
        if scorer_options.stop_if_expired() {
            return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>);
        }

        // Cap SHOULD clauses to MAX_QUERY_TERMS, but only count queries that need
        // posting-list cursors. Fast-field predicates (O(1) per doc) are exempt.
        let should_capped: Vec<Arc<dyn Query>>;
        let should: &[Arc<dyn Query>] = if should_all.len() > super::MAX_QUERY_TERMS {
            let is_predicate: Vec<bool> = should_all
                .iter()
                .map(|q| q.is_filter() || q.as_doc_predicate(reader).is_some())
                .collect();
            let cursor_count = is_predicate.iter().filter(|&&p| !p).count();

            if cursor_count > super::MAX_QUERY_TERMS {
                let mut kept = Vec::with_capacity(should_all.len());
                let mut cursor_kept = 0usize;
                for (q, &is_pred) in should_all.iter().zip(is_predicate.iter()) {
                    if is_pred {
                        kept.push(q.clone());
                    } else if cursor_kept < super::MAX_QUERY_TERMS {
                        kept.push(q.clone());
                        cursor_kept += 1;
                    }
                }
                log::warn!(
                    "BooleanQuery: capping cursor SHOULD from {} to {} ({} fast-field predicates exempt); dropped clauses do not match or score",
                    cursor_count,
                    super::MAX_QUERY_TERMS,
                    kept.len() - cursor_kept,
                );
                should_capped = kept;
                &should_capped
            } else {
                log::debug!(
                    "BooleanQuery: {} SHOULD clauses OK ({} need cursors, {} fast-field predicates)",
                    should_all.len(),
                    cursor_count,
                    should_all.len() - cursor_count,
                );
                should_all
            }
        } else {
            should_all
        };

        // ── 1. Single-clause optimisation ────────────────────────────────
        if must_not.is_empty() {
            if must.len() == 1 && should.is_empty() {
                return must[0].$scorer_fn(reader, limit, scorer_options) $(.  $aw)* ;
            }
            if should.len() == 1 && must.is_empty() && $text_tuning.0 == 1.0 {
                return should[0].$scorer_fn(reader, limit, scorer_options) $(. $aw)* ;
            }
        }

        if (scorer_options.complete_text_matches || scorer_options.physical_text_field.is_some()) && must.is_empty() && must_not.is_empty()
            && !should.is_empty()
            && let Some((infos, field, avg_field_len, num_docs)) = prepare_text_maxscore(should, reader, global_stats)
            && text_maxscore_allowed(reader, field, scorer_options.collect_positions)
        {
            if $proximity.is_some() {
                return Err(crate::Error::Query("proximity scoring inside a required text clause is not supported".into()));
            }
            let mut postings = Vec::with_capacity(infos.len());
            for info in infos {
                if let Some(pl) = reader.$get_postings_fn(info.field, &info.term) $(. $aw)* ? {
                    let idf = compute_idf(&pl, field, &info.term, num_docs, global_stats) * info.weight;
                    postings.push((pl, idf));
                }
            }
            return super::term::complete_text_scorer(postings, avg_field_len, reader, field, &scorer_options);
        }

        // Plain ranked text terms on one field share the bounded text window
        // executor. Semantic membership (every MUST term, plus optional SHOULD
        // terms) is imposed inside the executor before its top-k cutoff.
        // Complete streams, positions, boosts, tuning, proximity, chunked
        // fields and compositions with their own semantics stay below.
        let counted_limit = scorer_options.ranked_count_limit.filter(|_| should.is_empty()
            && scorer_options.shared_threshold.as_ref().and_then(super::SharedThreshold::deadline).is_none());
        let ranked_limit = counted_limit.unwrap_or(limit);
        if !must.is_empty() && must_not.is_empty()
            && must.len() + should.len() <= super::MAX_QUERY_TERMS
            && ranked_limit < reader.num_docs() as usize
            && (!scorer_options.complete_text_matches || counted_limit.is_some()) && !scorer_options.collect_positions
            && $proximity.is_none() && $text_tuning.0 == 1.0 && $text_tuning.1 == 0
            && let Some((required, field, _, _)) = prepare_text_maxscore(must, reader, global_stats)
            && let Some(optional) = ranked_optional_terms(should, field, reader, global_stats)
            && required.iter().chain(&optional).all(|info| info.weight == 1.0)
            && (!reader.has_text_mapping(field)
                || (scorer_options.physical_text_field == Some(field) && reader.alive_docs().is_none()))
        {
            let required_count = required.len();
            let params = super::Bm25Params::for_field(reader.schema(), field);
            let mut cursors = Vec::with_capacity(required_count + optional.len());
            for (index, info) in required.into_iter().chain(optional).enumerate() {
                let Some(postings) = reader.$get_postings_fn(field, &info.term) $(. $aw)* ? else {
                    if index < required_count {
                        log::debug!("BooleanQuery planner: required term absent → empty result");
                        return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>);
                    }
                    continue;
                };
                let (idf, avg_len) = super::term::compute_term_idf(
                    &postings, field, reader, global_stats, &info.term,
                );
                cursors.push(super::TermCursor::text_with_params(
                    postings, idf, avg_len,
                    reader.chunk_map(field).map(super::LengthSource::Chunks)
                        .or_else(|| reader.doc_lengths(field).map(super::LengthSource::Docs)), params,
                ));
            }
            log::debug!(
                "BooleanQuery planner: ranked text windows, {} required + {} optional terms",
                required_count,
                cursors.len() - required_count
            );
            let executor = ranked_text_executor(cursors, required_count, ranked_limit, reader, field, &scorer_options);
            if counted_limit.is_some() {
                let (mut results, count) = executor.execute_counted_conjunction()?;
                super::text_mapping::physical_results(&mut results, reader, scorer_options.physical_text_field);
                return Ok(Box::new(super::planner::TopKResultScorer::new(results).with_exact_count(count)) as Box<dyn Scorer + '_>);
            }
            let mut results = executor.$execute_fn() $(. $aw)* ?;
            super::text_mapping::physical_results(&mut results, reader, scorer_options.physical_text_field);
            return Ok(Box::new(super::planner::TopKResultScorer::new(results)) as Box<dyn Scorer + '_>);
        }

        // Other physical compositions keep complete child streams.
        if scorer_options.physical_text_field.is_none() {
        // ── 2. Pure OR → MaxScore optimisations ──────────────────────────
        if must.is_empty() && must_not.is_empty()
            && (should.len() >= 2 || (should.len() == 1 && $text_tuning.0 < 1.0)) {
            // 2a. Text MaxScore (single-field, all term queries)
            if !scorer_options.complete_text_matches
                && let Some((mut infos, text_field, avg_field_len, num_docs)) =
                prepare_text_maxscore(should, reader, global_stats)
                && text_maxscore_allowed(reader, text_field, scorer_options.collect_positions)
            {
                let mut posting_lists = Vec::with_capacity(infos.len());
                let mut term_bytes: Vec<Vec<u8>> = Vec::new();
                for info in infos.drain(..) {
                    if let Some(pl) = reader.$get_postings_fn(info.field, &info.term)
                        $(. $aw)* ?
                    {
                        let idf = compute_idf(&pl, info.field, &info.term, num_docs, global_stats) * info.weight;
                        posting_lists.push((pl, idf));
                        term_bytes.push(info.term.clone());
                    }
                }
                cap_terms(&mut posting_lists, &mut term_bytes, $text_tuning.1);
                // Chunked field: score chunks, fold to documents with ordinals.
                if reader.has_text_mapping(text_field) {
                    return finish_chunked_text_maxscore(
                        posting_lists, avg_field_len, limit, reader, text_field, eligibility_predicate(&scorer_options),
                        $proximity.map(|config| (config, term_bytes)),
                        $text_tuning.0,
                        scorer_options.shared_threshold.as_ref(),
                    );
                }
                // Seed from the cross-segment floor: this path scores final
                // per-doc BM25 into a top-`limit` heap, so a floor carried from
                // an already-searched segment prunes exactly (see
                // SharedThreshold). The per-field path below stays at 0.0 —
                // its per-field partial scores are not the final doc score.
                let shared_threshold = std::cell::Cell::new(scorer_options.initial_threshold);
                return finish_text_maxscore(
                    posting_lists,
                    avg_field_len,
                    reader.doc_lengths(text_field),
                    limit,
                    &shared_threshold,
                    reader,
                    text_field,
                    eligibility_predicate(&scorer_options),
                    super::Bm25Params::for_field(reader.schema(), text_field),
                    $proximity.map(|config| (config, term_bytes)),
                    $text_tuning.0,
                    scorer_options.shared_threshold.as_ref(),
                );
            }

            // 2b. Sparse (single-field, all sparse term queries)
            // Auto-detect: BMP executor if field has BMP index, else MaxScore
            if let Some(infos) =
                shared_or_extract_sparse_infos(scorer_options.lsp_plan.as_ref(), should)
            {
                if let Some((raw, info)) =
                    build_sparse_bmp_results(&infos, reader, limit, &scorer_options)?
                {
                    return Ok(combine_sparse_results(raw, info.combiner, info.field, limit));
                }
                if let Some((executor, info)) =
                    build_sparse_maxscore_executor(&infos, reader, limit, None, &scorer_options)
                {
                    let raw = executor.$execute_fn() $(. $aw)* ?;
                    return Ok(combine_sparse_results(raw, info.combiner, info.field, limit));
                }
            }

            // 2c. Per-field text MaxScore (multi-field term grouping)
            if !scorer_options.complete_text_matches
                && let Some(grouping) = prepare_per_field_grouping(
                should,
                reader,
                limit,
                global_stats,
                scorer_options.collect_positions,
            ) {
                let mut scorers: Vec<Box<dyn Scorer + '_>> = Vec::new();
                // Query-local cross-group threshold seeding (see finish_text_maxscore)
                let shared_threshold = std::cell::Cell::new(0.0f32);
                for (field, avg_field_len, infos) in &grouping.multi_term_groups {
                    // Chunked fields: IDF over chunks, not documents.
                    let corpus_size = reader.text_corpus_size(*field);
                    let mut posting_lists = Vec::with_capacity(infos.len());
                let mut term_bytes: Vec<Vec<u8>> = Vec::new();
                    for info in infos {
                        if let Some(pl) = reader.$get_postings_fn(info.field, &info.term)
                            $(. $aw)* ?
                        {
                            let idf = compute_idf(
                                &pl, *field, &info.term, corpus_size, global_stats,
                            ) * info.weight;
                            posting_lists.push((pl, idf));
                        term_bytes.push(info.term.clone());
                        }
                    }
                    cap_terms(&mut posting_lists, &mut term_bytes, $text_tuning.1);
                    if reader.has_text_mapping(*field) {
                        scorers.push(finish_chunked_text_maxscore(
                            posting_lists,
                            *avg_field_len,
                            grouping.per_field_limit,
                            reader,
                            *field,
                            eligibility_predicate(&scorer_options),
                            $proximity.map(|config| (config, term_bytes)),
                            $text_tuning.0,
                            scorer_options.shared_threshold.as_ref(),
                        )?);
                    } else if !posting_lists.is_empty() {
                        scorers.push(finish_text_maxscore(
                            posting_lists,
                            *avg_field_len,
                            reader.doc_lengths(*field),
                            grouping.per_field_limit,
                            &shared_threshold,
                            reader,
                            *field,
                            eligibility_predicate(&scorer_options),
                            super::Bm25Params::for_field(reader.schema(), *field),
                            $proximity.map(|config| (config, term_bytes)),
                            $text_tuning.0,
                            scorer_options.shared_threshold.as_ref(),
                        )?);
                    }
                }
                // A child's own top-k is not a safe candidate set for a summed
                // parent: request its complete stream.
                for &idx in &grouping.fallback_indices {
                    scorers.push(should[idx].$scorer_fn(
                        reader,
                        limit,
                        scorer_options.for_required_clause(),
                    ) $(. $aw)* ?);
                }
                return Ok(build_should_scorer(scorers));
            }
        }

        // ── 3. Filter push-down (MUST + SHOULD) ─────────────────────────
        //
        // Position collection no longer disables this path: fast-field
        // predicates carry no positions to lose and verifier scorers keep
        // theirs. Only the posting-list bitset shortcut is skipped when
        // positions are requested, because a bitset cannot report them.
        if (!scorer_options.complete_text_matches || extract_all_sparse_infos(should).is_some())
            && !should.is_empty() && (!must.is_empty() || !must_not.is_empty()) {
            // ── 3-text. Text SHOULD with materializable filters ──────────
            //
            // When every SHOULD clause is a text term and the MUST/MUST_NOT
            // clauses combine into one document bitset (term filters, ranges,
            // quoted phrases via `PhraseQuery::as_doc_bitset`), the text
            // MaxScore executors run with the bitset as a predicate: the
            // top-k is exact over the filtered documents (bounds are unaffected
            // by a filter), instead of an over-fetched unfiltered top-k that a
            // PredicatedScorer thins out afterwards. Documents matching only
            // the filters (score 0) fill the tail when fewer than `limit`
            // scored documents survive, keeping Boolean semantics.
            let text_groups: Option<Vec<(crate::Field, Vec<super::TermQueryInfo>)>> = {
                let mut groups: Vec<(crate::Field, Vec<super::TermQueryInfo>)> = Vec::new();
                let mut all_text = true;
                for q in should {
                    match q.decompose() {
                        super::QueryDecomposition::TextTerm(info)
                            if info.global_stats.is_none() && text_maxscore_allowed(
                                reader, info.field, scorer_options.collect_positions,
                            ) =>
                        {
                            match groups.iter_mut().find(|(f, _)| *f == info.field) {
                                Some((_, infos)) => infos.push(info),
                                None => groups.push((info.field, vec![info])),
                            }
                        }
                        _ => {
                            all_text = false;
                            break;
                        }
                    }
                }
                all_text.then_some(groups)
            };
            if must.iter().all(|query| {
                query.is_filter()
                    || query.as_doc_predicate(reader).is_some()
                    || (!matches!(
                        query.decompose(),
                        super::QueryDecomposition::TextTerm(_)
                    ) && scorer_options.doc_bitset(query.as_ref(), reader).is_some())
            })
                && let Some(groups) = text_groups
                && (groups.len() == 1
                    || ($proximity.is_none()
                        && groups
                            .iter()
                            .all(|(field, _)| !reader.has_text_mapping(*field))))
                && let Some(bitset) = build_combined_bitset(must, must_not, reader, &scorer_options)
            {
                if scorer_options.stop_if_expired() {
                    return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>);
                }
                let bitset = std::sync::Arc::new(bitset);
                let single_field = groups.len() == 1;

                // Scores from different fields are additive. Running a
                // separate top-k per field and merging those windows is not
                // exact: a document just below every local cutoff can still
                // win after its field scores are summed. Non-chunked text
                // fields share document ids, so put all of their cursors in
                // one executor and apply the filter there.
                if !single_field {
                    let mut cursors = Vec::new();
                    for (field, infos) in groups {
                        let corpus_size = reader.text_corpus_size(field);
                        let avg_field_len = global_stats
                            .map(|stats| stats.avg_field_len(field))
                            .unwrap_or_else(|| reader.avg_field_len(field));
                        let params = super::Bm25Params::for_field(reader.schema(), field);
                        let mut posting_lists = Vec::with_capacity(infos.len());
                        let mut term_bytes = Vec::with_capacity(infos.len());
                        for info in &infos {
                            if let Some(postings) =
                                reader.$get_postings_fn(field, &info.term) $(. $aw)* ?
                            {
                                let idf = compute_idf(
                                    &postings,
                                    field,
                                    &info.term,
                                    corpus_size,
                                    global_stats,
                                ) * info.weight;
                                posting_lists.push((postings, idf));
                                term_bytes.push(info.term.clone());
                            }
                        }
                        cap_terms(&mut posting_lists, &mut term_bytes, $text_tuning.1);
                        cursors.extend(posting_lists.into_iter().map(|(postings, idf)| {
                            super::TermCursor::text_with_params(
                                postings,
                                idf,
                                avg_field_len,
                                reader.doc_lengths(field).map(super::LengthSource::Docs),
                                params,
                            )
                        }));
                    }

                    let filter = bitset.clone();
                    let predicate: super::DocPredicate<'_> =
                        Box::new(move |doc_id| filter.contains(doc_id));
                    let mut executor = super::MaxScoreExecutor::new(
                        cursors,
                        limit,
                        $text_tuning.0,
                    )
                    .with_metric_labels(reader.schema().index_label(), "<multiple>")
                    .with_predicate(predicate)
                    .with_budget(scorer_options.shared_threshold.clone());
                    if $text_tuning.0 == 1.0 && scorer_options.initial_threshold > 0.0 {
                        executor.seed_threshold(scorer_options.initial_threshold);
                    }
                    let results = executor.execute_sync()?;
                    let found = results.len() as u32;
                    let should_scorer: Box<dyn Scorer + '_> =
                        Box::new(super::planner::TopKResultScorer::new(results));
                    if !must.is_empty() && (found as usize) < limit && bitset.count() > found {
                        return Ok(Box::new(super::planner::BitsetFillScorer::new(
                            should_scorer,
                            bitset,
                        )));
                    }
                    return Ok(should_scorer);
                }

                let group_limit = if single_field {
                    limit
                } else {
                    super::max_candidate_limit(limit)
                        .min(reader.num_docs() as usize)
                        .max(1)
                };
                // Cross-segment floor only when the group score is the final
                // document score (single field); per-field partial scores
                // start at 0.0 like path 2c.
                let shared_threshold = std::cell::Cell::new(if single_field {
                    scorer_options.initial_threshold
                } else {
                    0.0
                });
                let mut scorers: Vec<Box<dyn Scorer + '_>> = Vec::new();
                let mut found = 0u32;
                let mut complete = true;
                for (field, infos) in groups {
                    let corpus_size = reader.text_corpus_size(field);
                    let avg_field_len = global_stats
                        .map(|s| s.avg_field_len(field))
                        .unwrap_or_else(|| reader.avg_field_len(field));
                    let mut posting_lists = Vec::with_capacity(infos.len());
                let mut term_bytes: Vec<Vec<u8>> = Vec::new();
                    for info in &infos {
                        if let Some(pl) = reader.$get_postings_fn(field, &info.term) $(. $aw)* ? {
                            let idf = compute_idf(&pl, field, &info.term, corpus_size, global_stats) * info.weight;
                            posting_lists.push((pl, idf));
                        term_bytes.push(info.term.clone());
                        }
                    }
                    cap_terms(&mut posting_lists, &mut term_bytes, $text_tuning.1);
                    let filter = bitset.clone();
                    let predicate: super::DocPredicate<'_> =
                        Box::new(move |doc_id| filter.contains(doc_id));
                    let scorer = if reader.has_text_mapping(field) {
                        finish_chunked_text_maxscore(
                            posting_lists, avg_field_len, group_limit, reader, field, Some(predicate),
                            $proximity.map(|config| (config, term_bytes)),
                            $text_tuning.0,
                            scorer_options.shared_threshold.as_ref(),
                        )?
                    } else {
                        finish_text_maxscore(
                            posting_lists,
                            avg_field_len,
                            reader.doc_lengths(field),
                            group_limit,
                            &shared_threshold,
                            reader,
                            field,
                            Some(predicate),
                            super::Bm25Params::for_field(reader.schema(), field),
                            $proximity.map(|config| (config, term_bytes)),
                            $text_tuning.0,
                            scorer_options.shared_threshold.as_ref(),
                        )?
                    };
                    let hits = scorer.size_hint();
                    found = found.saturating_add(hits);
                    if hits as usize >= group_limit {
                        complete = false;
                    }
                    scorers.push(scorer);
                }
                log::debug!(
                    "BooleanQuery planner: bitset-aware text MaxScore, {} field group(s), \
                     {} filtered docs, {} scored hits",
                    scorers.len(),
                    bitset.count(),
                    found
                );
                let should_scorer = build_should_scorer(scorers);
                if !must.is_empty()
                    && complete
                    && (found as usize) < limit
                    && bitset.count() > found
                {
                    return Ok(Box::new(super::planner::BitsetFillScorer::new(
                        should_scorer,
                        bitset,
                    )));
                }
                return Ok(should_scorer);
            }

            // Pre-check: is SHOULD all-sparse? This determines whether we can
            // use bitset fallback for MUST clauses that lack fast-field predicates.
            // For sparse SHOULD, the predicate is pushed into BMP/MaxScore traversal
            // so all qualifying docs are found. For text SHOULD, we must NOT convert
            // MUST to a predicate (PredicatedScorer would drop MUST-only docs that
            // don't match SHOULD), so those go to verifier → BooleanScorer.
            let should_is_sparse = scorer_options.lsp_plan.is_some()
                || extract_all_sparse_infos(should).is_some();
            let bitset_predicates_allowed = should_is_sparse && !scorer_options.collect_positions;

            // 3a. Compile MUST → predicates (O(1)) vs verifier scorers (seek)
            //
            // Priority: as_doc_predicate (fast-field O(1)) > as_doc_bitset
            // (posting-list materialization, O(1) lookup, sparse-SHOULD only)
            // > verifier scorer (seek).
            super::planner::push_down_text_predicates(
                must, should, must_not, reader, &mut scorer_options,
            )?;
            if scorer_options.stop_if_expired()
                || scorer_options.eligibility.as_ref()
                    .is_some_and(|bits| bits.next_set_bit(0).is_none()) {
                return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>);
            }
            let mut predicates: Vec<super::DocPredicate<'_>> = Vec::new();
            let mut must_verifiers: Vec<Box<dyn super::Scorer + '_>> = Vec::new();
            for q in must {
                if let Some(pred) = q.as_doc_predicate(reader) {
                    log::debug!("BooleanQuery planner 3a: MUST clause → predicate ({})", q);
                    predicates.push(pred);
                } else if bitset_predicates_allowed {
                    if let Some(bitset) = scorer_options.doc_bitset(q.as_ref(), reader) {
                        log::debug!("BooleanQuery planner 3a: MUST clause → bitset predicate ({})", q);
                        predicates.push(Box::new(move |doc_id| bitset.contains(doc_id)));
                    } else {
                        log::debug!("BooleanQuery planner 3a: MUST clause → verifier scorer ({})", q);
                        must_verifiers.push(q.$scorer_fn(
                            reader, limit, scorer_options.for_required_clause()
                        ) $(. $aw)* ?);
                    }
                } else {
                    log::debug!("BooleanQuery planner 3a: MUST clause → verifier scorer ({})", q);
                    must_verifiers.push(q.$scorer_fn(
                        reader, limit, scorer_options.for_required_clause()
                    ) $(. $aw)* ?);
                }
            }
            // Compile MUST_NOT → negated predicates vs verifier scorers
            let mut must_not_verifiers: Vec<Box<dyn super::Scorer + '_>> = Vec::new();
            for q in must_not {
                if let Some(pred) = q.as_doc_predicate(reader) {
                    let negated: super::DocPredicate<'_> =
                        Box::new(move |doc_id| !pred(doc_id));
                    predicates.push(negated);
                } else if bitset_predicates_allowed {
                    if let Some(bitset) = scorer_options.doc_bitset(q.as_ref(), reader) {
                        log::debug!("BooleanQuery planner 3a: MUST_NOT clause → bitset predicate ({})", q);
                        predicates.push(Box::new(move |doc_id| !bitset.contains(doc_id)));
                    } else {
                        must_not_verifiers.push(q.$scorer_fn(
                            reader, limit, scorer_options.for_required_clause()
                        ) $(. $aw)* ?);
                    }
                } else {
                    must_not_verifiers.push(q.$scorer_fn(
                        reader, limit, scorer_options.for_required_clause()
                    ) $(. $aw)* ?);
                }
            }

            // 3b. Fast path: pure predicates + sparse SHOULD → BMP or MaxScore w/ predicate
            if scorer_options.stop_if_expired() {
                return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>);
            }
            if must_verifiers.is_empty()
                && must_not_verifiers.is_empty()
                && !predicates.is_empty()
            {
                let sparse_infos =
                    shared_or_extract_sparse_infos(scorer_options.lsp_plan.as_ref(), should);
                if let Some(infos) = sparse_infos {
                    // Try BMP with bitset first: build compact bitset from MUST/MUST_NOT
                    // posting lists (O(M) for term queries) for fast per-slot lookup.
                    let bitset_result = build_combined_bitset(must, must_not, reader, &scorer_options);
                    if scorer_options.stop_if_expired() {
                        return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>);
                    }
                    if let Some(ref bitset) = bitset_result {
                        let bitset_pred = |doc_id: crate::DocId| bitset.contains(doc_id);
                        if let Some((raw, info)) =
                            build_sparse_bmp_results_filtered(
                                &infos, reader, limit, &bitset_pred, &scorer_options
                            )?
                        {
                            log::debug!(
                                "BooleanQuery planner: bitset-aware sparse BMP, {} dims, {} matching docs",
                                infos.len(),
                                bitset.count()
                            );
                            return Ok(combine_sparse_results(raw, info.combiner, info.field, limit));
                        }
                    }

                    // Fallback: closure predicate (for queries that don't support bitsets)
                    let combined = chain_predicates(predicates);
                    if let Some((raw, info)) =
                        build_sparse_bmp_results_filtered(
                            &infos, reader, limit, &*combined, &scorer_options
                        )?
                    {
                        log::debug!(
                            "BooleanQuery planner: predicate-aware sparse BMP, {} dims",
                            infos.len()
                        );
                        return Ok(combine_sparse_results(raw, info.combiner, info.field, limit));
                    }
                    // Try MaxScore with predicate
                    if let Some((executor, info)) =
                        build_sparse_maxscore_executor(&infos, reader, limit, Some(combined), &scorer_options)
                    {
                        log::debug!(
                            "BooleanQuery planner: predicate-aware sparse MaxScore, {} dims",
                            infos.len()
                        );
                        let raw = executor.$execute_fn() $(. $aw)* ?;
                        return Ok(combine_sparse_results(raw, info.combiner, info.field, limit));
                    }
                    // predicates consumed — cannot fall through; rebuild them
                    // (this path only triggers if neither sparse index exists)
                    // should_is_sparse is true here (we're inside extract_all_sparse_infos)
                    predicates = Vec::new();
                    for q in must {
                        if let Some(pred) = q.as_doc_predicate(reader) {
                            predicates.push(pred);
                        } else if let Some(bitset) = scorer_options.doc_bitset(q.as_ref(), reader) {
                            predicates.push(Box::new(move |doc_id| bitset.contains(doc_id)));
                        }
                    }
                    for q in must_not {
                        if let Some(pred) = q.as_doc_predicate(reader) {
                            let negated: super::DocPredicate<'_> =
                                Box::new(move |doc_id| !pred(doc_id));
                            predicates.push(negated);
                        } else if let Some(bitset) = scorer_options.doc_bitset(q.as_ref(), reader) {
                            predicates.push(Box::new(move |doc_id| !bitset.contains(doc_id)));
                        }
                    }
                }
            }

            // 3c. Generic fallback — never filter a truncated SHOULD window.
            // Sparse retrieval keeps its combined candidate executor. Other
            // query shapes use the individual SHOULD streams so filters and
            // scoring requirements see the complete document streams.
            let mut should_options = if must_verifiers.is_empty() && must_not_verifiers.is_empty() {
                scorer_options.without_threshold()
            } else {
                scorer_options.for_required_clause()
            };
            if should_is_sparse {
                // The outer decomposition built this plan from the complete
                // sparse SHOULD expression. Filters cannot increase scores,
                // so retain global γ even when a verifier prevents predicate
                // push-down. Thresholds still belong to the outer score space
                // and remain cleared.
                should_options.lsp_plan = scorer_options.lsp_plan.clone();
            }
            let proximity_should = $proximity.is_some();
            if proximity_should {
                // This existing path explicitly uses the full text corpus as
                // sub_limit below, so it already preserves required matches.
                should_options.complete_text_matches = false;
            }
            let combined_should = should.len() == 1 || should_is_sparse || proximity_should;
            let should_scorer: Option<Box<dyn Scorer + '_>> = if should.len() == 1 {
                Some(should[0].$scorer_fn(reader, limit, should_options.clone()) $(. $aw)* ?)
            } else if should_is_sparse || proximity_should {
                let sub = BooleanQuery {
                    must: Vec::new(),
                    should: should.to_vec(),
                    must_not: Vec::new(),
                    global_stats: global_stats.cloned(),
                    proximity: $proximity,
                    text_heap_factor: $text_tuning.0,
                    max_terms: $text_tuning.1,
                };
                // Proximity is a positive second-stage bonus. Preserve the
                // complete SHOULD stream before applying outer requirements;
                // a bounded BM25-only window can omit the document whose
                // proximity bonus would promote it. Chunked fields use their
                // virtual-id corpus size, plain fields their document count.
                let sub_limit = if proximity_should {
                    should
                        .first()
                        .and_then(|query| match query.decompose() {
                            super::QueryDecomposition::TextTerm(info) => {
                                Some(reader.text_corpus_size(info.field) as usize)
                            }
                            _ => None,
                        })
                        .unwrap_or(reader.num_docs() as usize)
                        .max(limit)
                } else {
                    super::max_candidate_limit(limit)
                };
                Some(sub.$scorer_fn(
                    reader,
                    sub_limit,
                    should_options.clone(),
                ) $(. $aw)* ?)
            } else {
                None
            };
            let should_scorers: Vec<Box<dyn Scorer + '_>> = match should_scorer {
                Some(scorer) => vec![scorer],
                None => {
                    let mut scorers = Vec::with_capacity(should.len());
                    for query in should {
                        scorers.push(query.$scorer_fn(
                            reader,
                            limit,
                            should_options.clone(),
                        ) $(. $aw)* ?);
                    }
                    scorers
                }
            };

            if must_verifiers.is_empty() {
                let should_scorer = build_should_scorer(should_scorers);
                log::debug!(
                    "BooleanQuery planner: PredicatedScorer {} preds + {} must_not_v, \
                     SHOULD size_hint={}, combined={}",
                    predicates.len(), must_not_verifiers.len(),
                    should_scorer.size_hint(), combined_should
                );
                return Ok(Box::new(super::PredicatedScorer::new(
                    should_scorer, predicates, Vec::new(), must_not_verifiers,
                )));
            }

            // Scoring MUST clauses drive the conjunction; SHOULD is optional.
            log::debug!(
                "BooleanQuery planner: required-clause BooleanScorer {} must + {} should, \
                 {} preds + {} must_not_v",
                must_verifiers.len(), should_scorers.len(),
                predicates.len(), must_not_verifiers.len()
            );
            let mut driver = BooleanScorer {
                must: must_verifiers,
                should: should_scorers,
                must_not: Vec::new(),
                current_doc: 0,
        lead: 0,
        doc_limit: reader.num_docs(),
            };
            driver.initialize();
            return Ok(Box::new(super::PredicatedScorer::new(
                Box::new(driver),
                predicates,
                Vec::new(),
                must_not_verifiers,
            )));
        }

        }

        // ── 4. Standard BooleanScorer fallback ───────────────────────────
        if scorer_options.physical_text_field.is_none() {
            super::planner::push_down_text_predicates(
                must, should, must_not, reader, &mut scorer_options,
            )?;
        }
        if scorer_options.stop_if_expired()
            || scorer_options.eligibility.as_ref()
                .is_some_and(|bits| bits.next_set_bit(0).is_none()) {
            return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + '_>);
        }
        let mut must_scorers = Vec::with_capacity(must.len());
        // A child top-k is not a safe candidate set for a summed parent:
        // a document outside every child heap may still have the best total.
        let child_options = if scorer_options.complete_text_matches
            || should.len() > 1
            || !should.is_empty() && !must.is_empty()
            || must.iter().filter(|query| query.as_doc_predicate(reader).is_none()).count() > 1
            || must_not.iter().any(|query| query.as_doc_predicate(reader).is_none())
        {
            scorer_options.for_required_clause()
        } else {
            scorer_options.without_threshold()
        };
        if must.is_empty() && should.is_empty() && !must_not.is_empty() {
            must_scorers.push(Box::new(super::AllDocSet::new(reader.num_docs()))
                as Box<dyn Scorer + '_>);
        }
        for q in must {
            must_scorers.push(q.$scorer_fn(
                reader, limit, child_options.clone()
            ) $(. $aw)* ?);
        }
        let mut should_scorers = Vec::with_capacity(should.len());
        for q in should {
            should_scorers.push(q.$scorer_fn(
                reader, limit, child_options.clone()
            ) $(. $aw)* ?);
        }
        let mut must_not_scorers = Vec::with_capacity(must_not.len());
        for q in must_not {
            must_not_scorers.push(q.$scorer_fn(
                reader, limit, scorer_options.for_required_clause()
            ) $(. $aw)* ?);
        }
        let mut scorer = BooleanScorer {
            must: must_scorers,
            should: should_scorers,
            must_not: must_not_scorers,
            current_doc: 0,
        lead: 0,
        doc_limit: reader.num_docs(),
        };
        scorer.initialize();
        Ok(Box::new(scorer) as Box<dyn Scorer + '_>)
    }};
}

impl Query for BooleanQuery {
    fn physical_text_field(&self, reader: &SegmentReader, complete: bool) -> Option<crate::Field> {
        if self.proximity.is_some() || self.text_heap_factor != 1.0 || self.max_terms != 0 {
            return None;
        }
        // Preserve the existing ranked union executor and its stable-ID heap.
        if !complete
            && self.must.is_empty()
            && self.must_not.is_empty()
            && self
                .should
                .iter()
                .all(|q| matches!(q.decompose(), super::QueryDecomposition::TextTerm(_)))
        {
            return None;
        }
        let mut clauses = self.must.iter().chain(&self.should).chain(&self.must_not);
        let field = clauses.next()?.physical_text_field(reader, true)?;
        clauses
            .all(|q| q.physical_text_field(reader, true) == Some(field))
            .then_some(field)
    }
    fn candidate_query(&self) -> crate::Result<crate::query::CandidateQuery> {
        if !self.must.is_empty() || !self.must_not.is_empty() || self.proximity.is_some() {
            return Err(crate::Error::Query("L1 scoring branches support SHOULD composition; move required/excluded constraints into fusion.filters and use explicit phrase branches for proximity".into()));
        }
        if let super::QueryDecomposition::SparseTerms(infos) = self.decompose() {
            return super::CandidateQuery::from_decomposition(
                super::QueryDecomposition::SparseTerms(infos),
            );
        }
        super::CandidateQuery::sum(self.should.iter().map(|query| query.candidate_query()))
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
        let must = self.must.clone();
        let should = self.should.clone();
        let must_not = self.must_not.clone();
        let global_stats = self
            .global_stats
            .clone()
            .or_else(|| options.global_stats.clone());
        let proximity = self.proximity;
        let text_tuning = (self.text_heap_factor, self.max_terms);
        Box::pin(async move {
            boolean_plan!(
                must,
                should,
                must_not,
                global_stats.as_ref(),
                proximity,
                text_tuning,
                reader,
                limit,
                options,
                scorer_with_options,
                get_postings,
                execute,
                await
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
        let global_stats = self
            .global_stats
            .clone()
            .or_else(|| options.global_stats.clone());
        boolean_plan!(
            self.must,
            self.should,
            self.must_not,
            global_stats.as_ref(),
            self.proximity,
            (self.text_heap_factor, self.max_terms),
            reader,
            limit,
            options,
            scorer_sync_with_options,
            get_postings_sync,
            execute_sync
        )
    }

    fn text_terms(&self, out: &mut Vec<(crate::dsl::Field, Vec<u8>)>) {
        for clause in self.must.iter().chain(&self.should).chain(&self.must_not) {
            clause.text_terms(out);
        }
    }

    fn decompose(&self) -> super::QueryDecomposition {
        if !self.must.is_empty() || !self.must_not.is_empty() {
            return super::QueryDecomposition::Opaque;
        }
        self.lsp_decomposition()
    }

    fn count_equivalent_term(&self) -> Option<super::TermQueryInfo> {
        if self.must.len() == 1
            && self.must_not.is_empty()
            && self.proximity.is_none()
            && self.text_heap_factor == 1.0
            && self.max_terms == 0
            && self
                .should
                .iter()
                .all(|query| matches!(query.decompose(), super::QueryDecomposition::TextTerm(_)))
        {
            self.must[0].count_equivalent_term()
        } else {
            None
        }
    }

    fn supports_ranked_conjunction_count(&self) -> bool {
        self.must.len() >= 2 && self.should.is_empty() && self.must_not.is_empty()
            && self.proximity.is_none() && self.text_heap_factor == 1.0 && self.max_terms == 0
            && self.must.iter().all(|query| matches!(query.decompose(),
                super::QueryDecomposition::TextTerm(info) if info.weight == 1.0 && info.global_stats.is_none()))
    }

    fn ranked_count_equivalent_term(&self) -> Option<super::TermQueryInfo> {
        let count = self.count_equivalent_term()?;
        let super::QueryDecomposition::TextTerm(required) = self.must[0].decompose() else {
            return None;
        };
        (required.weight.is_finite()
            && required.weight > 0.0
            && self.should.iter().all(|query| {
                matches!(query.decompose(), super::QueryDecomposition::TextTerm(info)
                    if info.field == required.field && info.weight.is_finite() && info.weight > 0.0)
            }))
        .then_some(count)
    }

    fn lsp_decomposition(&self) -> super::QueryDecomposition {
        // LSP/0 selection depends only on the sparse scoring clauses. Pure
        // filters may remove documents but cannot increase their score, so a
        // query-global superblock plan remains valid and must be shared across
        // segments for filtered sparse queries too. A scoring MUST clause can
        // change final ordering, therefore keep that shape opaque.
        if self.should.is_empty() || self.must.iter().any(|query| !query.is_filter()) {
            return super::QueryDecomposition::Opaque;
        }
        extract_all_sparse_infos(&self.should)
            .map(super::QueryDecomposition::SparseTerms)
            .unwrap_or(super::QueryDecomposition::Opaque)
    }

    fn should_children(&self) -> Option<&[Arc<dyn Query>]> {
        if self.must.is_empty()
            && self.must_not.is_empty()
            && !self.should.is_empty()
            && self.proximity.is_none()
            && self.global_stats.is_none()
            && self.text_heap_factor == 1.0
            && self.max_terms == 0
        {
            Some(&self.should)
        } else {
            None
        }
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
        if self.must.is_empty() && self.should.is_empty() && self.must_not.is_empty() {
            return None;
        }

        let num_docs = reader.num_docs();

        // MUST clauses: intersect bitsets (AND)
        let mut result = (self.must.is_empty() && self.should.is_empty())
            .then(|| super::DocBitset::all(num_docs));
        for q in &self.must {
            let bs = options.doc_bitset(q.as_ref(), reader)?;
            match result {
                None => result = Some(bs),
                Some(ref mut acc) => acc.intersect_with(&bs),
            }
        }

        // SHOULD clauses: union bitsets (OR), then intersect with MUST result
        if !self.should.is_empty() {
            let mut should_union = super::DocBitset::new(num_docs);
            for q in &self.should {
                let bs = options.doc_bitset(q.as_ref(), reader)?;
                should_union.union_with(&bs);
            }
            match result {
                None => result = Some(should_union),
                Some(ref mut acc) => {
                    // When MUST clauses exist, SHOULD is optional (doesn't filter).
                    // When no MUST clauses, at least one SHOULD must match.
                    if self.must.is_empty() {
                        *acc = should_union;
                    }
                }
            }
        }

        // MUST_NOT clauses: subtract bitsets (ANDNOT)
        if let Some(ref mut acc) = result {
            for q in &self.must_not {
                {
                    let bs = options.doc_bitset(q.as_ref(), reader)?;
                    acc.subtract(&bs);
                }
            }
        }

        if options.stop_if_expired() {
            None
        } else {
            result
        }
    }

    fn as_doc_predicate<'a>(&self, reader: &'a SegmentReader) -> Option<super::DocPredicate<'a>> {
        // Need at least some clauses
        if self.must.is_empty() && self.should.is_empty() && self.must_not.is_empty() {
            return None;
        }

        // Try converting all clauses to predicates; bail if any child can't
        let must_preds: Vec<_> = self
            .must
            .iter()
            .map(|q| q.as_doc_predicate(reader))
            .collect::<Option<Vec<_>>>()?;
        let should_preds: Vec<_> = self
            .should
            .iter()
            .map(|q| q.as_doc_predicate(reader))
            .collect::<Option<Vec<_>>>()?;
        let must_not_preds: Vec<_> = self
            .must_not
            .iter()
            .map(|q| q.as_doc_predicate(reader))
            .collect::<Option<Vec<_>>>()?;

        let has_must = !must_preds.is_empty();

        Some(Box::new(move |doc_id| {
            // All MUST predicates must pass
            if !must_preds.iter().all(|p| p(doc_id)) {
                return false;
            }
            // When there are no MUST clauses, at least one SHOULD must pass
            if !has_must && !should_preds.is_empty() && !should_preds.iter().any(|p| p(doc_id)) {
                return false;
            }
            // No MUST_NOT predicate should pass
            must_not_preds.iter().all(|p| !p(doc_id))
        }))
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let must = self.must.clone();
        let should = self.should.clone();
        let has_exclusions = !self.must_not.is_empty();

        Box::pin(async move {
            if !must.is_empty() {
                let mut estimates = Vec::with_capacity(must.len());
                for q in &must {
                    estimates.push(q.count_estimate(reader).await?);
                }
                estimates
                    .into_iter()
                    .min()
                    .ok_or_else(|| crate::Error::Corruption("Empty must clause".to_string()))
            } else if !should.is_empty() {
                let mut sum = 0u32;
                for q in &should {
                    sum = sum.saturating_add(q.count_estimate(reader).await?);
                }
                Ok(sum)
            } else if has_exclusions {
                Ok(reader.num_docs())
            } else {
                Ok(0)
            }
        })
    }
}

pub(super) struct BooleanScorer<'a> {
    must: Vec<Box<dyn Scorer + 'a>>,
    should: Vec<Box<dyn Scorer + 'a>>,
    must_not: Vec<Box<dyn Scorer + 'a>>,
    current_doc: DocId,
    /// Most selective required cursor; preserve score summation order.
    lead: usize,
    /// Segment document space, used only to estimate window setup cost.
    doc_limit: u32,
}

impl<'a> BooleanScorer<'a> {
    pub(super) fn disjunction(should: Vec<Box<dyn Scorer + 'a>>) -> Self {
        let mut scorer = Self {
            must: Vec::new(),
            should,
            must_not: Vec::new(),
            current_doc: 0,
            lead: 0,
            doc_limit: 0,
        };
        scorer.initialize();
        scorer
    }

    fn initialize(&mut self) {
        self.lead = self
            .must
            .iter()
            .enumerate()
            .min_by_key(|(_, scorer)| match scorer.size_hint() {
                0 => u32::MAX,
                cost => cost,
            })
            .map_or(0, |(index, _)| index);
        self.current_doc = self.find_next_match();
    }

    fn find_next_match(&mut self) -> DocId {
        if self.must.is_empty() && self.should.is_empty() {
            return TERMINATED;
        }

        loop {
            let candidate = if !self.must.is_empty() {
                let mut candidate = self.must[self.lead].doc();
                'align: loop {
                    if candidate == TERMINATED {
                        return TERMINATED;
                    }
                    for index in 0..self.must.len() {
                        if index == self.lead {
                            continue;
                        }
                        let doc = self.must[index].seek_candidate(candidate);
                        if doc > candidate {
                            // No intersection exists before the rejecting
                            // cursor's next candidate. Advance the selected
                            // driver without revisiting the rejected prefix.
                            candidate = self.must[self.lead].seek_candidate(doc);
                            continue 'align;
                        }
                    }
                    break candidate;
                }
            } else {
                self.should
                    .iter()
                    .map(|s| s.doc())
                    .filter(|&d| d != TERMINATED)
                    .min()
                    .unwrap_or(TERMINATED)
            };

            if candidate == TERMINATED {
                return TERMINATED;
            }

            if !self
                .must
                .iter_mut()
                .all(|scorer| scorer.confirm_candidate())
            {
                self.must[self.lead].advance_candidate();
                continue;
            }

            let excluded = self.must_not.iter_mut().any(|scorer| {
                let doc = scorer.seek(candidate);
                doc == candidate
            });

            if !excluded {
                // Seek SHOULD scorers to candidate so score() can see their contributions
                for scorer in &mut self.should {
                    scorer.seek(candidate);
                }
                self.current_doc = candidate;
                return candidate;
            }

            // Advance past excluded candidate
            if !self.must.is_empty() {
                self.must[self.lead].advance_candidate();
            } else {
                // For SHOULD-only: seek all scorers past the excluded candidate
                for scorer in &mut self.should {
                    if scorer.doc() <= candidate && scorer.doc() != TERMINATED {
                        scorer.seek(candidate + 1);
                    }
                }
            }
        }
    }
}

impl super::docset::DocSet for BooleanScorer<'_> {
    fn supports_doc_batches(&self) -> bool {
        self.must.len() > 1
            && self.should.is_empty()
            && self.must_not.is_empty()
            && self.must.iter().all(|child| child.supports_doc_batches())
    }

    fn fill_doc_batch(&mut self, docs: &mut super::docset::DocBatch) -> usize {
        if !self.supports_doc_batches() {
            return super::docset::fill_batch(self, docs);
        }
        if self.current_doc == TERMINATED {
            return 0;
        }
        let mut count = self.must[self.lead].fill_doc_batch(docs);
        for (index, child) in self.must.iter_mut().enumerate() {
            if index != self.lead {
                count = child.retain_doc_batch(docs, count);
                if count == 0 {
                    break;
                }
            }
        }
        self.current_doc = self.find_next_match();
        count
    }

    fn supports_doc_windows(&self) -> bool {
        self.must_not.is_empty()
            && if self.must.is_empty() {
                self.should.iter().all(|child| child.supports_doc_windows())
            } else {
                self.should.is_empty()
                    && u64::from(self.must[self.lead].size_hint())
                        * u64::from(super::docset::DOC_WINDOW_SIZE)
                        >= u64::from(self.doc_limit) * super::docset::DOC_WINDOW_WORDS as u64
                    && self.must.iter().all(|child| child.supports_doc_windows())
            }
    }

    fn fill_doc_window(&mut self, base: DocId, bits: &mut super::docset::DocWindow) {
        if !self.supports_doc_windows() {
            return super::docset::fill_window(self, base, bits);
        }
        bits.fill(0);
        if self.current_doc == TERMINATED {
            return;
        }
        let mut child_bits = [0; super::docset::DOC_WINDOW_WORDS];
        if self.must.is_empty() {
            for child in &mut self.should {
                child.fill_doc_window(base, &mut child_bits);
                for (out, child_word) in bits.iter_mut().zip(child_bits) {
                    *out |= child_word;
                }
            }
        } else {
            self.must[self.lead].fill_doc_window(base, bits);
            for (index, child) in self.must.iter_mut().enumerate() {
                if index == self.lead {
                    continue;
                }
                let candidates: u32 = bits.iter().map(|word| word.count_ones()).sum();
                if candidates == 0 {
                    break;
                }
                // Dense masks amortize a full child batch over bitmap words;
                // selective masks probe only the surviving candidates.
                if candidates > super::docset::DOC_WINDOW_WORDS as u32 {
                    child.fill_doc_window(base, &mut child_bits);
                    for (out, child_word) in bits.iter_mut().zip(child_bits) {
                        *out &= child_word;
                    }
                } else {
                    for (word_index, word) in bits.iter_mut().enumerate() {
                        let mut remaining = *word;
                        while remaining != 0 {
                            let bit = remaining.trailing_zeros();
                            let doc = base + word_index as u32 * 64 + bit;
                            if child.seek(doc) != doc {
                                *word &= !(1u64 << bit);
                            }
                            remaining &= remaining - 1;
                        }
                    }
                }
            }
        }
        self.current_doc = self.find_next_match();
    }

    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn advance(&mut self) -> DocId {
        if !self.must.is_empty() {
            self.must[self.lead].advance_candidate();
        } else {
            for scorer in &mut self.should {
                if scorer.doc() == self.current_doc {
                    scorer.advance();
                }
            }
        }

        self.current_doc = self.find_next_match();
        self.current_doc
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.must.is_empty() {
            for scorer in &mut self.should {
                scorer.seek(target);
            }
        } else {
            self.must[self.lead].seek_candidate(target);
        }

        self.current_doc = self.find_next_match();
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        if !self.must.is_empty() {
            self.must.iter().map(|s| s.size_hint()).min().unwrap_or(0)
        } else {
            self.should
                .iter()
                .fold(0u32, |total, s| total.saturating_add(s.size_hint()))
        }
    }
}

impl Scorer for BooleanScorer<'_> {
    fn supports_score_batches(&self) -> bool {
        !self.must.is_empty()
            && (self.must.len() > 1 || !self.must_not.is_empty())
            && self.should.is_empty()
            && self.must[self.lead].size_hint() > super::docset::DOC_BATCH_SIZE as u32
            && self.must.iter().all(|child| child.supports_score_batches())
    }

    fn fill_score_batch(
        &mut self,
        docs: &mut super::docset::DocBatch,
        scores: &mut super::ScoreBatch,
    ) -> usize {
        if !self.supports_score_batches() {
            return super::traits::fill_score_batch_scalar(self, docs, scores);
        }
        if self.current_doc == TERMINATED {
            return 0;
        }
        let mut lead_scores = [0.0; super::docset::DOC_BATCH_SIZE];
        let mut count = self.must[self.lead].fill_score_batch(docs, &mut lead_scores);
        let mut origins: [u8; super::docset::DOC_BATCH_SIZE] = std::array::from_fn(|i| i as u8);
        let mut values = [0.0; super::docset::DOC_BATCH_SIZE];
        let mut matches = [0; 2];
        scores[..count].fill(0.0);
        for (index, child) in self.must.iter_mut().enumerate() {
            if index == self.lead {
                for row in 0..count {
                    scores[row] += lead_scores[origins[row] as usize];
                }
                continue;
            }
            child.score_batch_matches(docs, count, &mut values, &mut matches);
            let mut kept = 0;
            for row in 0..count {
                if matches[row / 64] & (1 << (row % 64)) != 0 {
                    docs[kept] = docs[row];
                    scores[kept] = scores[row] + values[row];
                    origins[kept] = origins[row];
                    kept += 1;
                }
            }
            count = kept;
            if count == 0 {
                break;
            }
        }
        if !self.must_not.is_empty() {
            let mut kept = 0;
            for row in 0..count {
                let doc = docs[row];
                if self.must_not.iter_mut().all(|child| child.seek(doc) != doc) {
                    docs[kept] = doc;
                    scores[kept] = scores[row];
                    kept += 1;
                }
            }
            count = kept;
        }
        self.current_doc = self.find_next_match();
        count
    }

    fn supports_filtered_windows(&self) -> bool {
        super::docset::DocSet::supports_doc_windows(self)
            && (self.must.len() > 1 || self.should.len() > 1)
    }

    fn supports_score_windows(&self) -> bool {
        self.must.is_empty()
            && self.should.len() > 1
            && super::docset::DocSet::supports_doc_windows(self)
    }

    fn fill_score_window(
        &mut self,
        base: DocId,
        scores: &mut [Score; super::docset::DOC_WINDOW_SIZE as usize],
        bits: &mut super::docset::DocWindow,
    ) {
        scores.fill(0.0);
        bits.fill(0);
        if !self.supports_score_windows() {
            self.accumulate_score_window(base, scores, bits);
            return;
        }
        if self.current_doc == TERMINATED {
            return;
        }
        // Preserve the scalar scorer's child order, including the complete
        // score of any nested child. Never distribute a nested floating sum.
        for child in &mut self.should {
            child.accumulate_score_window(base, scores, bits);
        }
        self.current_doc = self.find_next_match();
    }

    fn score(&self) -> Score {
        let mut total = 0.0;

        for scorer in &self.must {
            if scorer.doc() == self.current_doc {
                total += scorer.score();
            }
        }

        for scorer in &self.should {
            if scorer.doc() == self.current_doc {
                total += scorer.score();
            }
        }

        total
    }

    fn matched_positions(&self) -> Option<super::MatchedPositions> {
        let mut all_positions: super::MatchedPositions = Vec::new();

        for scorer in &self.must {
            if scorer.doc() == self.current_doc
                && let Some(positions) = scorer.matched_positions()
            {
                all_positions.extend(positions);
            }
        }

        for scorer in &self.should {
            if scorer.doc() == self.current_doc
                && let Some(positions) = scorer.matched_positions()
            {
                all_positions.extend(positions);
            }
        }

        if all_positions.is_empty() {
            None
        } else {
            Some(merge_matched_positions(all_positions))
        }
    }
}

/// Coalesce the position lists that several clauses reported for one field.
///
/// Two term clauses on the same chunked field each report the chunk ordinal
/// they matched; the union must present one entry per chunk whose score is
/// the sum of the clause contributions (the chunk's BM25 score), not the same
/// ordinal twice. Distinct positions are left untouched, so token positions of
/// `positions`-mode fields keep their per-term scores.
pub(super) fn merge_matched_positions(
    positions: super::MatchedPositions,
) -> super::MatchedPositions {
    if positions.len() < 2 {
        return positions;
    }
    let mut merged: super::MatchedPositions = Vec::with_capacity(positions.len());
    for (field_id, scored) in positions {
        match merged
            .iter_mut()
            .find(|(existing, _)| *existing == field_id)
        {
            Some((_, existing)) => existing.extend(scored),
            None => merged.push((field_id, scored)),
        }
    }
    for (_, scored) in &mut merged {
        if scored.len() < 2 {
            continue;
        }
        scored.sort_by_key(|sp| sp.position);
        let mut write = 0usize;
        for read in 1..scored.len() {
            if scored[read].position == scored[write].position {
                scored[write].score += scored[read].score;
            } else {
                write += 1;
                scored[write] = scored[read];
            }
        }
        scored.truncate(write + 1);
    }
    merged
}

/// Optional SHOULD terms for the ranked text window executor: all plain text
/// terms on `field`, or an empty list when there are none.
fn ranked_optional_terms(
    should: &[Arc<dyn Query>],
    field: crate::Field,
    reader: &SegmentReader,
    global_stats: Option<&Arc<GlobalStats>>,
) -> Option<Vec<super::TermQueryInfo>> {
    if should.is_empty() {
        return Some(Vec::new());
    }
    let (optional, optional_field, _, _) = prepare_text_maxscore(should, reader, global_stats)?;
    (optional_field == field).then_some(optional)
}

/// Configure the shared text window executor for ranked term queries whose
/// first `required_count` cursors are semantic MUST terms. All-required
/// queries with at least two cursors use the conjunction driver.
fn ranked_text_executor<'a>(
    cursors: Vec<super::TermCursor<'a>>,
    required_count: usize,
    limit: usize,
    reader: &'a SegmentReader,
    field: crate::Field,
    options: &super::ScorerOptions,
) -> super::MaxScoreExecutor<'a> {
    let all_required = required_count == cursors.len() && cursors.len() >= 2;
    let executor = super::MaxScoreExecutor::new(cursors, limit, 1.0);
    let mut executor = if all_required {
        executor.require_all_terms()
    } else {
        executor.require_prefix_terms(required_count)
    }
    .with_metric_labels(
        reader.schema().index_label(),
        reader.schema().get_field_name(field).unwrap_or("?"),
    )
    .with_budget(options.shared_threshold.clone());
    if options.physical_text_field == Some(field) {
        executor =
            executor.with_document_map(reader.chunk_map(field).expect("admitted document map"));
    }
    if let Some(predicate) = eligibility_predicate(options) {
        executor = executor.with_predicate(predicate);
    }
    if options.initial_threshold > 0.0 {
        executor.seed_threshold(options.initial_threshold);
    }
    executor
}

fn eligibility_predicate(options: &super::ScorerOptions) -> Option<super::DocPredicate<'static>> {
    options.eligibility.as_ref().map(|filter| {
        let filter = filter.clone();
        Box::new(move |doc| filter.contains(doc)) as super::DocPredicate<'static>
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::Field;
    use crate::query::{DocSet, QueryDecomposition, TermQuery};

    #[test]
    fn ranked_count_hints_require_an_exact_compatible_text_plan() {
        struct CountOnly(TermQuery);
        impl std::fmt::Display for CountOnly {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(f)
            }
        }
        impl Query for CountOnly {
            fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
                self.0.scorer(reader, limit)
            }
            fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
                self.0.count_estimate(reader)
            }
            fn count_equivalent_term(&self) -> Option<super::super::TermQueryInfo> {
                self.0.count_equivalent_term()
            }
        }
        let field = Field(0);
        let compatible = BooleanQuery::new()
            .must(TermQuery::text(field, "required"))
            .should(TermQuery::text(field, "optional"));
        assert!(compatible.ranked_count_equivalent_term().is_some());
        assert!(
            Arc::new(compatible.clone())
                .ranked_count_equivalent_term()
                .is_some()
        );
        for query in [
            compatible
                .clone()
                .must_not(TermQuery::text(field, "excluded")),
            compatible.clone().must(TermQuery::text(field, "second")),
            compatible.clone().with_text_heap_factor(0.5),
            compatible.clone().with_max_terms(1),
            compatible
                .clone()
                .should(TermQuery::text(Field(1), "other-field")),
            compatible.should(super::super::BoostQuery::new(
                TermQuery::text(field, "negative"),
                -1.0,
            )),
            BooleanQuery::new()
                .must(CountOnly(TermQuery::text(field, "opaque")))
                .should(TermQuery::text(field, "optional")),
        ] {
            assert!(query.ranked_count_equivalent_term().is_none());
        }
        let opaque = CountOnly(TermQuery::text(field, "opaque"));
        assert!(opaque.count_equivalent_term().is_some());
        assert!(opaque.ranked_count_equivalent_term().is_none());
    }

    #[test]
    fn compact_score_batches_preserve_query_order_and_nested_scores() {
        use crate::query::docset::{DOC_BATCH_SIZE, DocSet, SortedVecDocSet};
        struct ExactScore {
            docs: SortedVecDocSet,
            value: f32,
        }
        impl DocSet for ExactScore {
            fn doc(&self) -> u32 {
                self.docs.doc()
            }
            fn advance(&mut self) -> u32 {
                self.docs.advance()
            }
            fn seek(&mut self, target: u32) -> u32 {
                self.docs.seek(target)
            }
            fn size_hint(&self) -> u32 {
                self.docs.size_hint()
            }
        }
        impl Scorer for ExactScore {
            fn score(&self) -> f32 {
                self.value
            }
            fn supports_score_batches(&self) -> bool {
                true
            }
        }
        fn leaf(step: u32, value: f32) -> Box<dyn Scorer> {
            Box::new(ExactScore {
                docs: SortedVecDocSet::new(Arc::new(
                    (0..20000).filter(|d| d % step == 0).collect(),
                )),
                value,
            })
        }
        fn conjunction(children: Vec<Box<dyn Scorer>>) -> BooleanScorer<'static> {
            let mut result = BooleanScorer::disjunction(Vec::new());
            result.must = children;
            result.initialize();
            result
        }
        fn make(nested: bool, excluded: bool, single: bool) -> BooleanScorer<'static> {
            let middle: Box<dyn Scorer> = if nested {
                Box::new(conjunction(vec![leaf(7, 1.0e10), leaf(7, 1.0)]))
            } else {
                leaf(7, 1.0e10)
            };
            let mut scorer = if single {
                conjunction(vec![leaf(3, 1.0)])
            } else {
                conjunction(vec![leaf(3, -1.0e10), middle, leaf(11, 1.0)])
            };
            if excluded {
                scorer.must_not = vec![leaf(2, 500.0), leaf(13, 500.0)];
                scorer.initialize();
            }
            scorer
        }
        for (nested, excluded, single) in [
            (false, false, false),
            (true, false, false),
            (false, true, false),
            (true, true, false),
            (false, true, true),
        ] {
            let mut scalar = make(nested, excluded, single);
            let mut batched = make(nested, excluded, single);
            assert_eq!(batched.lead, if single { 0 } else { 2 });
            assert!(batched.supports_score_batches());
            let mut docs = [0; DOC_BATCH_SIZE];
            let mut scores = [0.0; DOC_BATCH_SIZE];
            assert_eq!(batched.seek(17), scalar.seek(17));
            while batched.doc() != TERMINATED {
                let count = batched.fill_score_batch(&mut docs, &mut scores);
                assert!(count > 0);
                for i in 0..count {
                    assert_eq!(docs[i], scalar.doc());
                    assert_eq!(scores[i].to_bits(), scalar.score().to_bits());
                    assert_eq!(
                        scores[i], 1.0,
                        "lead-first or distributed sums change this value"
                    );
                    scalar.advance();
                }
                assert_eq!(batched.doc(), scalar.doc());
                assert_eq!(batched.advance(), scalar.advance());
            }
            assert_eq!(scalar.doc(), TERMINATED);
            assert_eq!(batched.fill_score_batch(&mut docs, &mut scores), 0);
        }
    }

    #[test]
    fn compact_boolean_batches_preserve_scalar_resume_and_nested_membership() {
        use crate::query::docset::{DOC_BATCH_SIZE, DocSet, SortedVecDocSet};
        struct ExactDocs(SortedVecDocSet);
        impl DocSet for ExactDocs {
            fn doc(&self) -> u32 {
                self.0.doc()
            }
            fn advance(&mut self) -> u32 {
                self.0.advance()
            }
            fn seek(&mut self, target: u32) -> u32 {
                self.0.seek(target)
            }
            fn size_hint(&self) -> u32 {
                self.0.size_hint()
            }
            fn supports_doc_batches(&self) -> bool {
                true
            }
        }
        impl Scorer for ExactDocs {
            fn score(&self) -> f32 {
                self.doc() as f32
            }
        }
        fn leaf(step: u32) -> Box<dyn Scorer> {
            Box::new(ExactDocs(SortedVecDocSet::new(Arc::new(
                (0..5000).filter(|doc| doc % step == 0).collect(),
            ))))
        }
        fn make(nested: bool) -> BooleanScorer<'static> {
            let mut scorer = BooleanScorer::disjunction(Vec::new());
            scorer.must = vec![leaf(3), leaf(7)];
            if nested {
                scorer.must.push(Box::new(make(false)));
            }
            scorer.doc_limit = 1_000_000;
            scorer.initialize();
            scorer
        }
        for nested in [false, true] {
            let mut scalar = make(nested);
            let mut batched = make(nested);
            assert!(batched.supports_doc_batches());
            assert_eq!(batched.seek(17), scalar.seek(17));
            let mut docs = [0; DOC_BATCH_SIZE];
            while batched.doc() != TERMINATED {
                assert_eq!(batched.doc(), scalar.doc());
                assert_eq!(batched.score().to_bits(), scalar.score().to_bits());
                let count = batched.fill_doc_batch(&mut docs);
                assert!(count > 0);
                for &doc in &docs[..count] {
                    assert_eq!(doc, scalar.doc());
                    scalar.advance();
                }
                assert_eq!(batched.doc(), scalar.doc());
                assert_eq!(batched.advance(), scalar.advance());
            }
            assert_eq!(scalar.doc(), TERMINATED);
            assert_eq!(batched.fill_doc_batch(&mut docs), 0);
        }
    }

    #[test]
    fn score_windows_preserve_nested_sums_seek_and_exhaustion() {
        struct Scores {
            hits: Vec<(DocId, Score)>,
            index: usize,
        }
        impl DocSet for Scores {
            fn doc(&self) -> DocId {
                self.hits.get(self.index).map_or(TERMINATED, |hit| hit.0)
            }
            fn advance(&mut self) -> DocId {
                self.index = (self.index + 1).min(self.hits.len());
                self.doc()
            }
            fn size_hint(&self) -> u32 {
                self.hits.len() as u32
            }
            fn supports_doc_windows(&self) -> bool {
                true
            }
        }
        impl Scorer for Scores {
            fn score(&self) -> Score {
                self.hits[self.index].1
            }
        }
        fn child(seed: u32) -> Box<dyn Scorer> {
            let hits = (0..16_400)
                .chain([u32::MAX - 4097, u32::MAX - 2])
                .filter(|doc| (doc % (seed + 2)) != 1)
                .map(|doc| {
                    (
                        doc,
                        match seed {
                            0 => 16_777_216.0,
                            1 => 1.0,
                            2 => -16_777_216.0,
                            _ => -0.0,
                        },
                    )
                })
                .collect();
            Box::new(Scores { hits, index: 0 })
        }
        fn build(shape: u8) -> BooleanScorer<'static> {
            let children = if shape == 2 {
                let mut conjunction = BooleanScorer {
                    must: vec![child(1), child(2)],
                    should: Vec::new(),
                    must_not: Vec::new(),
                    current_doc: 0,
                    lead: 0,
                    doc_limit: 0,
                };
                conjunction.initialize();
                vec![child(0), Box::new(conjunction), child(3)]
            } else if shape == 1 {
                vec![
                    child(0),
                    Box::new(BooleanScorer::disjunction(vec![child(1), child(2)])),
                    child(3),
                ]
            } else {
                (0..4).map(child).collect()
            };
            BooleanScorer::disjunction(children)
        }
        for shape in 0..3 {
            for start in [0, 17, 4095, 8193, u32::MAX - 4098, TERMINATED] {
                let mut scalar = build(shape);
                let mut batched = build(shape);
                scalar.seek(start);
                batched.seek(start);
                assert!(batched.supports_score_windows());
                let mut scores =
                    Box::new([f32::NAN; super::super::docset::DOC_WINDOW_SIZE as usize]);
                let mut bits = [u64::MAX; super::super::docset::DOC_WINDOW_WORDS];
                while batched.doc() != TERMINATED {
                    let base = batched.doc();
                    let end = base.saturating_add(super::super::docset::DOC_WINDOW_SIZE);
                    batched.fill_score_window(base, &mut scores, &mut bits);
                    for (index, &word) in bits.iter().enumerate() {
                        let mut remaining = word;
                        while remaining != 0 {
                            let slot = index * 64 + remaining.trailing_zeros() as usize;
                            assert_eq!(base + slot as u32, scalar.doc());
                            assert_eq!(
                                scores[slot].to_bits(),
                                scalar.score().to_bits(),
                                "shape={shape} start={start} doc={}",
                                scalar.doc()
                            );
                            scalar.advance();
                            remaining &= remaining - 1;
                        }
                    }
                    assert!(scalar.doc() >= end);
                    assert_eq!(scalar.doc(), batched.doc());
                    if scalar.doc() != TERMINATED {
                        assert_eq!(scalar.score().to_bits(), batched.score().to_bits());
                    }
                }
                batched.fill_score_window(TERMINATED, &mut scores, &mut bits);
                assert!(bits.iter().all(|word| *word == 0));
                assert_eq!(batched.advance(), TERMINATED);
                assert_eq!(batched.seek(0), TERMINATED);
            }
        }
    }

    #[tokio::test]
    async fn document_windows_preserve_nested_boolean_membership_and_next_scores() {
        use crate::query::{Collector, CountCollector, ScorerOptions, collect_segment};
        use crate::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId};
        use crate::structures::PostingCodec;
        use std::sync::Arc;
        struct Ids(Vec<u32>);
        impl Collector for Ids {
            fn needs_scores(&self) -> bool {
                false
            }
            fn collect(
                &mut self,
                doc: u32,
                score: f32,
                positions: &[(u32, Vec<crate::query::ScoredPosition>)],
            ) {
                assert_eq!(score, 0.0);
                assert!(positions.is_empty());
                self.0.push(doc);
            }
        }
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let dir = crate::RamDirectory::new();
            let mut schema = crate::SchemaBuilder::default();
            let field = schema.add_text_field("text", true, false);
            schema.set_positions(field, crate::dsl::PositionMode::TokenPosition);
            let schema = Arc::new(schema.build());
            let config = SegmentBuilderConfig {
                posting_codec: codec,
                ..Default::default()
            };
            let mut builder = SegmentBuilder::new(schema.clone(), config).unwrap();
            for doc_id in 0..12_345 {
                let mut text = String::from("padding ");
                for (term, divisor) in [("alpha", 2), ("beta", 3), ("gamma", 11), ("rare", 997)] {
                    if doc_id % divisor == 0 {
                        text.push_str(&format!("{term} {term} "));
                    }
                }
                let mut doc = crate::Document::new();
                doc.add_text(field, text);
                builder.add_document(doc).unwrap();
            }
            let id = SegmentId::new();
            builder.build(&dir, id, None).await.unwrap();
            let reader = SegmentReader::open(&dir, id, schema, 16).await.unwrap();
            let term = |name| TermQuery::text(field, name);
            for shape in 0..9 {
                let query = match shape {
                    0 => BooleanQuery::new()
                        .should(term("alpha"))
                        .should(term("beta")),
                    1 => BooleanQuery::new()
                        .must(term("alpha"))
                        .must(term("beta"))
                        .should(term("gamma")),
                    2 => BooleanQuery::new()
                        .must(
                            BooleanQuery::new()
                                .should(term("alpha"))
                                .should(term("beta")),
                        )
                        .must_not(term("gamma")),
                    3 => BooleanQuery::new()
                        .must(term("rare"))
                        .must(
                            BooleanQuery::new()
                                .should(term("alpha"))
                                .should(term("beta")),
                        )
                        .must_not(term("gamma")),
                    4 => BooleanQuery::new().must(term("alpha")).should(term("beta")),
                    5 => BooleanQuery::new().must(term("alpha")).must(term("beta")),
                    6 => BooleanQuery::new().must(term("rare")).must(
                        BooleanQuery::new()
                            .should(term("alpha"))
                            .should(term("beta")),
                    ),
                    7 => BooleanQuery::new()
                        .must(term("alpha"))
                        .must(term("missing")),
                    _ => BooleanQuery::new()
                        .must(BooleanQuery::new().must(term("alpha")).must(term("beta")))
                        .must(term("gamma")),
                };
                let expected: Vec<u32> = (0..12_345)
                    .filter(|doc| {
                        let a = doc % 2 == 0;
                        let b = doc % 3 == 0;
                        let g = doc % 11 == 0;
                        match shape {
                            0 => a || b,
                            1 => a && b,
                            2 => (a || b) && !g,
                            3 => doc % 997 == 0 && (a || b) && !g,
                            4 => a,
                            5 => a && b,
                            6 => doc % 997 == 0 && (a || b),
                            7 => false,
                            _ => a && b && g,
                        }
                    })
                    .collect();
                let options = ScorerOptions {
                    complete_text_matches: true,
                    collect_positions: true,
                    ..Default::default()
                };
                let mut scalar = query
                    .scorer_with_options(&reader, 10, options.clone())
                    .await
                    .unwrap();
                let mut batched = query
                    .scorer_with_options(&reader, 10, options)
                    .await
                    .unwrap();
                assert_eq!(
                    batched.supports_doc_windows(),
                    matches!(shape, 0 | 5 | 8),
                    "shape={shape}"
                );
                let mut actual = Vec::new();
                while batched.doc() != TERMINATED {
                    let base = batched.doc();
                    let mut bits = [u64::MAX; super::super::docset::DOC_WINDOW_WORDS];
                    batched.fill_doc_window(base, &mut bits);
                    for (index, word) in bits.into_iter().enumerate() {
                        for bit in 0..64 {
                            if word & (1 << bit) != 0 {
                                actual.push(base + index as u32 * 64 + bit);
                            }
                        }
                    }
                    scalar.seek(base.saturating_add(super::super::docset::DOC_WINDOW_SIZE));
                    assert_eq!(batched.doc(), scalar.doc());
                    if scalar.doc() != TERMINATED {
                        assert_eq!(batched.score().to_bits(), scalar.score().to_bits());
                        let positions = |value: Option<crate::query::MatchedPositions>| {
                            value.map(|fields| {
                                fields
                                    .into_iter()
                                    .map(|(field, values)| {
                                        (
                                            field,
                                            values
                                                .into_iter()
                                                .map(|p| (p.position, p.score.to_bits()))
                                                .collect::<Vec<_>>(),
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            })
                        };
                        assert_eq!(
                            positions(batched.matched_positions()),
                            positions(scalar.matched_positions())
                        );
                    }
                }
                assert_eq!(actual, expected, "codec={codec:?} shape={shape}");
                let mut count = CountCollector::new();
                collect_segment(&reader, &query, &mut count).await.unwrap();
                assert_eq!(count.count() as usize, expected.len());
                let mut ids = Ids(Vec::new());
                collect_segment(&reader, &query, &mut ids).await.unwrap();
                assert_eq!(ids.0, expected);
                #[cfg(feature = "sync")]
                {
                    let mut sync = query
                        .scorer_sync_with_options(
                            &reader,
                            10,
                            ScorerOptions {
                                complete_text_matches: true,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                    let mut sync_ids = Vec::new();
                    while sync.doc() != TERMINATED {
                        let base = sync.doc();
                        let mut bits = [0; super::super::docset::DOC_WINDOW_WORDS];
                        sync.fill_doc_window(base, &mut bits);
                        for (index, word) in bits.into_iter().enumerate() {
                            for bit in 0..64 {
                                if word & (1 << bit) != 0 {
                                    sync_ids.push(base + index as u32 * 64 + bit);
                                }
                            }
                        }
                    }
                    assert_eq!(sync_ids, expected);
                }
            }
        }
    }

    #[tokio::test]
    async fn conjunction_chooses_the_rare_term_over_a_common_phrase() {
        let dir = crate::RamDirectory::new();
        let mut schema = crate::SchemaBuilder::default();
        let field = schema.add_text_field("text", true, false);
        schema.set_positions(field, crate::dsl::PositionMode::TokenPosition);
        schema.set_default_fields(vec!["text".into()]);
        let config = crate::IndexConfig {
            num_indexing_threads: 1,
            num_threads: 1,
            ..Default::default()
        };
        let mut writer = crate::IndexWriter::create(dir.clone(), schema.build(), config.clone())
            .await
            .unwrap();
        for doc_id in 0..32 {
            let mut doc = crate::Document::new();
            doc.add_text(
                field,
                if doc_id == 24 {
                    "alpha beta rare"
                } else {
                    "alpha beta"
                },
            );
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        writer.shutdown().await.unwrap();
        let index = crate::Index::open(dir, config).await.unwrap();
        let reader = index.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        let segment = &searcher.segment_readers()[0];
        let parser = searcher.query_parser();
        let phrase = parser.parse_strict("\"alpha beta\"").unwrap();
        let term = parser.parse_strict("rare").unwrap();
        let mut scorer = BooleanScorer {
            must: vec![
                phrase.scorer(segment, 100).await.unwrap(),
                term.scorer(segment, 100).await.unwrap(),
            ],
            should: Vec::new(),
            must_not: Vec::new(),
            current_doc: 0,
            lead: 0,
            doc_limit: 0,
        };
        scorer.initialize();
        assert_eq!(
            scorer.lead, 1,
            "the rarer term must drive candidate traversal"
        );
        assert_eq!(scorer.doc(), 24);
        assert!(scorer.score() > 0.0);
        assert_eq!(scorer.advance(), TERMINATED);
    }

    struct ObservedDocSet {
        docs: Vec<u32>,
        offset: usize,
        cost: u32,
        advances: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl super::super::DocSet for ObservedDocSet {
        fn doc(&self) -> u32 {
            self.docs.get(self.offset).copied().unwrap_or(TERMINATED)
        }
        fn advance(&mut self) -> u32 {
            self.advances
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.offset = (self.offset + 1).min(self.docs.len());
            self.doc()
        }
        fn seek(&mut self, target: u32) -> u32 {
            self.offset += self.docs[self.offset..].partition_point(|doc| *doc < target);
            self.doc()
        }
        fn size_hint(&self) -> u32 {
            self.cost
        }
    }
    impl Scorer for ObservedDocSet {
        fn score(&self) -> f32 {
            1.0
        }
    }

    #[test]
    fn conjunction_skips_expensive_advances_without_changing_matches_or_scores() {
        let expensive = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cheap = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut scorer = BooleanScorer {
            must: vec![
                Box::new(ObservedDocSet {
                    docs: (0..201).collect(),
                    offset: 0,
                    cost: 201,
                    advances: expensive.clone(),
                }),
                Box::new(ObservedDocSet {
                    docs: vec![0, 100, 200],
                    offset: 0,
                    cost: 3,
                    advances: cheap.clone(),
                }),
            ],
            should: Vec::new(),
            must_not: Vec::new(),
            current_doc: 0,
            lead: 0,
            doc_limit: 0,
        };
        scorer.initialize();
        assert_eq!(scorer.doc(), 0);
        assert_eq!(scorer.score(), 2.0);
        assert_eq!(scorer.advance(), 100);
        assert_eq!(scorer.score(), 2.0);
        assert_eq!(scorer.seek(150), 200);
        assert_eq!(scorer.score(), 2.0);
        assert_eq!(scorer.advance(), TERMINATED);
        assert_eq!(expensive.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(cheap.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    struct ObservedTwoPhase {
        doc: DocId,
        confirmations: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl super::super::docset::DocSet for ObservedTwoPhase {
        fn doc(&self) -> DocId {
            self.doc
        }
        fn advance(&mut self) -> DocId {
            self.seek(self.doc.saturating_add(1))
        }
        fn seek(&mut self, target: DocId) -> DocId {
            self.seek_candidate(target);
            while self.doc != TERMINATED && !self.confirm_candidate() {
                self.advance_candidate();
            }
            self.doc
        }
        fn size_hint(&self) -> u32 {
            201
        }
    }

    impl Scorer for ObservedTwoPhase {
        fn score(&self) -> Score {
            1.0
        }
        fn advance_candidate(&mut self) -> DocId {
            self.seek_candidate(self.doc.saturating_add(1))
        }
        fn seek_candidate(&mut self, target: DocId) -> DocId {
            self.doc = self.doc.max(target);
            if self.doc > 200 {
                self.doc = TERMINATED;
            }
            self.doc
        }
        fn confirm_candidate(&mut self) -> bool {
            self.confirmations
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.doc != TERMINATED && self.doc.is_multiple_of(50)
        }
    }

    #[test]
    fn conjunction_confirms_only_aligned_candidates_and_rejects_false_positives() {
        let confirmations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut scorer = BooleanScorer {
            must: vec![
                Box::new(ObservedTwoPhase {
                    doc: 0,
                    confirmations: confirmations.clone(),
                }),
                Box::new(ObservedDocSet {
                    docs: vec![0, 20, 100, 200],
                    offset: 0,
                    cost: 4,
                    advances: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                }),
            ],
            should: Vec::new(),
            must_not: vec![Box::new(ObservedDocSet {
                docs: vec![100],
                offset: 0,
                cost: 1,
                advances: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            })],
            current_doc: 0,
            lead: 0,
            doc_limit: 0,
        };
        scorer.initialize();
        assert_eq!(scorer.doc(), 0);
        assert_eq!(scorer.score(), 2.0);
        assert_eq!(scorer.advance(), 200);
        assert_eq!(scorer.score(), 2.0);
        assert_eq!(scorer.advance(), TERMINATED);
        assert_eq!(confirmations.load(std::sync::atomic::Ordering::Relaxed), 4);
    }

    #[test]
    fn disjunction_cardinality_hints_saturate_instead_of_overflowing() {
        let scorer = BooleanScorer::disjunction(
            (0..2)
                .map(|_| {
                    Box::new(ObservedDocSet {
                        docs: vec![0],
                        offset: 0,
                        cost: u32::MAX,
                        advances: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                    }) as Box<dyn Scorer>
                })
                .collect(),
        );
        assert_eq!(scorer.size_hint(), u32::MAX);
    }

    #[test]
    fn test_maxscore_eligible_pure_or_same_field() {
        // Pure OR query with multiple terms in same field should be MaxScore-eligible
        let query = BooleanQuery::new()
            .should(TermQuery::text(Field(0), "hello"))
            .should(TermQuery::text(Field(0), "world"))
            .should(TermQuery::text(Field(0), "foo"));

        // All clauses should return term info
        assert!(
            query
                .should
                .iter()
                .all(|q| matches!(q.decompose(), QueryDecomposition::TextTerm(_)))
        );

        // All should be same field
        let infos: Vec<_> = query
            .should
            .iter()
            .filter_map(|q| match q.decompose() {
                QueryDecomposition::TextTerm(info) => Some(info),
                _ => None,
            })
            .collect();
        assert_eq!(infos.len(), 3);
        assert!(infos.iter().all(|i| i.field == Field(0)));
    }

    #[test]
    fn test_maxscore_not_eligible_different_fields() {
        // OR query with terms in different fields should NOT use MaxScore
        let query = BooleanQuery::new()
            .should(TermQuery::text(Field(0), "hello"))
            .should(TermQuery::text(Field(1), "world")); // Different field!

        let infos: Vec<_> = query
            .should
            .iter()
            .filter_map(|q| match q.decompose() {
                QueryDecomposition::TextTerm(info) => Some(info),
                _ => None,
            })
            .collect();
        assert_eq!(infos.len(), 2);
        // Fields are different, MaxScore should not be used
        assert!(infos[0].field != infos[1].field);
    }

    #[test]
    fn test_term_query_info_extraction() {
        let term_query = TermQuery::text(Field(42), "test");
        match term_query.decompose() {
            QueryDecomposition::TextTerm(info) => {
                assert_eq!(info.field, Field(42));
                assert_eq!(info.term, b"test");
            }
            _ => panic!("Expected TextTerm decomposition"),
        }
    }

    #[test]
    fn test_boolean_query_no_term_info() {
        // BooleanQuery itself should not return term info
        let query = BooleanQuery::new().should(TermQuery::text(Field(0), "hello"));

        assert!(matches!(query.decompose(), QueryDecomposition::Opaque));
    }
}

#[cfg(test)]
#[path = "boolean/conjunction_window_tests.rs"]
mod conjunction_window_tests;
