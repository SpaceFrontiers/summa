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
    };
    scorer.current_doc = scorer.find_next_match();
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
        let scorer_options: super::ScorerOptions = $scorer_options;
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
                log::debug!(
                    "BooleanQuery: capping cursor SHOULD from {} to {} ({} fast-field predicates exempt)",
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

        // ── 2. Pure OR → MaxScore optimisations ──────────────────────────
        if must.is_empty() && must_not.is_empty()
            && (should.len() >= 2 || (should.len() == 1 && $text_tuning.0 < 1.0)) {
            // 2a. Text MaxScore (single-field, all term queries)
            if let Some((mut infos, text_field, avg_field_len, num_docs)) =
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
                if reader.is_chunked_field(text_field) {
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
                    build_sparse_maxscore_executor(&infos, reader, limit, None)
                {
                    let raw = executor.$execute_fn() $(. $aw)* ?;
                    return Ok(combine_sparse_results(raw, info.combiner, info.field, limit));
                }
            }

            // 2c. Per-field text MaxScore (multi-field term grouping)
            if let Some(grouping) = prepare_per_field_grouping(
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
                    if reader.is_chunked_field(*field) {
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
                for &idx in &grouping.fallback_indices {
                    scorers.push(should[idx].$scorer_fn(
                        reader,
                        limit,
                        scorer_options.without_threshold(),
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
        if !should.is_empty() && (!must.is_empty() || !must_not.is_empty()) {
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
                            if text_maxscore_allowed(
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
                            .all(|(field, _)| !reader.is_chunked_field(*field))))
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
                    let scorer = if reader.is_chunked_field(field) {
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
                            reader, limit, scorer_options.without_threshold()
                        ) $(. $aw)* ?);
                    }
                } else {
                    log::debug!("BooleanQuery planner 3a: MUST clause → verifier scorer ({})", q);
                    must_verifiers.push(q.$scorer_fn(
                        reader, limit, scorer_options.without_threshold()
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
                            reader, limit, scorer_options.without_threshold()
                        ) $(. $aw)* ?);
                    }
                } else {
                    must_not_verifiers.push(q.$scorer_fn(
                        reader, limit, scorer_options.without_threshold()
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
                        build_sparse_maxscore_executor(&infos, reader, limit, Some(combined))
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
            let mut should_options = scorer_options.without_threshold();
            if should_is_sparse {
                // The outer decomposition built this plan from the complete
                // sparse SHOULD expression. Filters cannot increase scores,
                // so retain global γ even when a verifier prevents predicate
                // push-down. Thresholds still belong to the outer score space
                // and remain cleared.
                should_options.lsp_plan = scorer_options.lsp_plan.clone();
            }
            let proximity_should = $proximity.is_some();
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
            };
            driver.current_doc = driver.find_next_match();
            return Ok(Box::new(super::PredicatedScorer::new(
                Box::new(driver),
                predicates,
                Vec::new(),
                must_not_verifiers,
            )));
        }

        // ── 4. Standard BooleanScorer fallback ───────────────────────────
        let mut must_scorers = Vec::with_capacity(must.len());
        if must.is_empty() && should.is_empty() && !must_not.is_empty() {
            must_scorers.push(Box::new(super::AllDocSet::new(reader.num_docs()))
                as Box<dyn Scorer + '_>);
        }
        for q in must {
            must_scorers.push(q.$scorer_fn(
                reader, limit, scorer_options.without_threshold()
            ) $(. $aw)* ?);
        }
        let mut should_scorers = Vec::with_capacity(should.len());
        for q in should {
            should_scorers.push(q.$scorer_fn(
                reader, limit, scorer_options.without_threshold()
            ) $(. $aw)* ?);
        }
        let mut must_not_scorers = Vec::with_capacity(must_not.len());
        for q in must_not {
            must_not_scorers.push(q.$scorer_fn(
                reader, limit, scorer_options.without_threshold()
            ) $(. $aw)* ?);
        }
        let mut scorer = BooleanScorer {
            must: must_scorers,
            should: should_scorers,
            must_not: must_not_scorers,
            current_doc: 0,
        };
        scorer.current_doc = scorer.find_next_match();
        Ok(Box::new(scorer) as Box<dyn Scorer + '_>)
    }};
}

impl Query for BooleanQuery {
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

struct BooleanScorer<'a> {
    must: Vec<Box<dyn Scorer + 'a>>,
    should: Vec<Box<dyn Scorer + 'a>>,
    must_not: Vec<Box<dyn Scorer + 'a>>,
    current_doc: DocId,
}

impl BooleanScorer<'_> {
    fn find_next_match(&mut self) -> DocId {
        if self.must.is_empty() && self.should.is_empty() {
            return TERMINATED;
        }

        loop {
            let candidate = if !self.must.is_empty() {
                let mut max_doc = self
                    .must
                    .iter()
                    .map(|s| s.doc())
                    .max()
                    .unwrap_or(TERMINATED);

                if max_doc == TERMINATED {
                    return TERMINATED;
                }

                loop {
                    let mut all_match = true;
                    for scorer in &mut self.must {
                        let doc = scorer.seek(max_doc);
                        if doc == TERMINATED {
                            return TERMINATED;
                        }
                        if doc > max_doc {
                            max_doc = doc;
                            all_match = false;
                            break;
                        }
                    }
                    if all_match {
                        break;
                    }
                }
                max_doc
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
                for scorer in &mut self.must {
                    scorer.advance();
                }
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
    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn advance(&mut self) -> DocId {
        if !self.must.is_empty() {
            for scorer in &mut self.must {
                scorer.advance();
            }
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
        for scorer in &mut self.must {
            scorer.seek(target);
        }

        for scorer in &mut self.should {
            scorer.seek(target);
        }

        self.current_doc = self.find_next_match();
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        if !self.must.is_empty() {
            self.must.iter().map(|s| s.size_hint()).min().unwrap_or(0)
        } else {
            self.should.iter().map(|s| s.size_hint()).sum()
        }
    }
}

impl Scorer for BooleanScorer<'_> {
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
    use crate::query::{QueryDecomposition, TermQuery};

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
    fn test_maxscore_not_eligible_with_must() {
        // Query with MUST clause should NOT use MaxScore optimization
        let query = BooleanQuery::new()
            .must(TermQuery::text(Field(0), "required"))
            .should(TermQuery::text(Field(0), "hello"))
            .should(TermQuery::text(Field(0), "world"));

        // Has MUST clause, so MaxScore optimization should not kick in
        assert!(!query.must.is_empty());
    }

    #[test]
    fn test_maxscore_not_eligible_with_must_not() {
        // Query with MUST_NOT clause should NOT use MaxScore optimization
        let query = BooleanQuery::new()
            .should(TermQuery::text(Field(0), "hello"))
            .should(TermQuery::text(Field(0), "world"))
            .must_not(TermQuery::text(Field(0), "excluded"));

        // Has MUST_NOT clause, so MaxScore optimization should not kick in
        assert!(!query.must_not.is_empty());
    }

    #[test]
    fn test_maxscore_not_eligible_single_term() {
        // Single SHOULD clause should NOT use MaxScore (no benefit)
        let query = BooleanQuery::new().should(TermQuery::text(Field(0), "hello"));

        // Only one term, MaxScore not beneficial
        assert_eq!(query.should.len(), 1);
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
