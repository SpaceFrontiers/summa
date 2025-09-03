//! Boolean query planner — shared helpers for MaxScore and filter push-down optimisation
//!
//! Extracted from `boolean.rs` to keep the planner logic separate from the
//! BooleanQuery struct, builder, and scorer types.

use std::sync::Arc;

use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::{DocId, Score};

use super::{
    DocPredicate, EmptyScorer, GlobalStats, MaxScoreExecutor, MultiValueCombiner, Query, ScoredDoc,
    Scorer, SparseTermQueryInfo, TermQueryInfo,
};

// ── IDF ──────────────────────────────────────────────────────────────────

/// Compute IDF for a posting list, preferring global stats.
pub(super) fn compute_idf(
    posting_list: &crate::structures::BlockPostingList,
    field: crate::Field,
    term: &[u8],
    num_docs: f32,
    global_stats: Option<&Arc<GlobalStats>>,
) -> f32 {
    if let Some(stats) = global_stats {
        let global_idf = stats.text_idf(field, &String::from_utf8_lossy(term));
        if global_idf > 0.0 {
            return global_idf;
        }
    }
    let doc_freq = posting_list.doc_count() as f32;
    super::bm25_idf(doc_freq, num_docs)
}

// ── Text MaxScore helpers ────────────────────────────────────────────────

/// Shared pre-check for text MaxScore: extract term infos, field, avg_field_len, num_docs.
/// Returns None if not all SHOULD clauses are single-field term queries.
pub(super) fn prepare_text_maxscore(
    should: &[Arc<dyn Query>],
    reader: &SegmentReader,
    global_stats: Option<&Arc<GlobalStats>>,
) -> Option<(Vec<TermQueryInfo>, crate::Field, f32, f32)> {
    let infos: Vec<_> = should
        .iter()
        .filter_map(|q| match q.decompose() {
            super::QueryDecomposition::TextTerm(info) => Some(info),
            _ => None,
        })
        .collect();
    if infos.len() != should.len() {
        return None;
    }
    let field = infos[0].field;
    if !infos.iter().all(|t| t.field == field) {
        return None;
    }
    let avg_field_len = global_stats
        .map(|s| s.avg_field_len(field))
        .unwrap_or_else(|| reader.avg_field_len(field));
    // Chunked fields: IDF over chunks, not documents.
    let num_docs = reader.text_corpus_size(field);
    Some((infos, field, avg_field_len, num_docs))
}

/// Build a TopK scorer from fetched posting lists via text MaxScore.
///
/// `shared_threshold` is a QUERY-EXECUTION-local cell: when one query has
/// multiple per-field MaxScore groups (path 2c), field A's result seeds
/// field B's pruning. It must never be shared across queries — a
/// per-segment cell here caused cross-query threshold leaks under
/// concurrent searches (one query's threshold wrongly pruning another's
/// results).
#[allow(clippy::too_many_arguments)]
pub(super) fn finish_text_maxscore<'a>(
    posting_lists: Vec<(crate::structures::BlockPostingList, f32)>,
    avg_field_len: f32,
    lengths: Option<&'a crate::segment::chunk_map::DocLengths>,
    limit: usize,
    shared_threshold: &std::cell::Cell<f32>,
    reader: &'a SegmentReader,
    field: crate::Field,
    predicate: Option<DocPredicate<'a>>,
    params: super::Bm25Params,
    proximity: Option<(super::ProximityConfig, Vec<Vec<u8>>)>,
    heap_factor: f32,
    budget: Option<&super::SharedThreshold>,
) -> crate::Result<Box<dyn Scorer + 'a>> {
    if posting_lists.is_empty() {
        return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + 'a>);
    }
    // Proximity rescoring works on an over-fetched candidate pool and adds
    // a non-negative bonus, so a BM25 floor from other segments cannot seed
    // the pass and no floor is published from it.
    let proximity = proximity.filter(|(config, terms)| config.is_active() && terms.len() >= 2);
    let executor_limit = if proximity.is_some() {
        limit
            .saturating_mul(super::proximity::PROXIMITY_OVER_FETCH)
            .max(64)
    } else {
        limit
    };
    let idfs: Vec<f32> = posting_lists.iter().map(|(_, idf)| *idf).collect();
    let mut executor = MaxScoreExecutor::text(
        posting_lists,
        avg_field_len,
        executor_limit,
        lengths,
        params,
        heap_factor,
    )
    .with_metric_labels(
        reader.schema().index_label(),
        reader.schema().get_field_name(field).unwrap_or("?"),
    );
    if let Some(predicate) = predicate {
        executor = executor.with_predicate(predicate);
    }
    executor = executor.with_budget(budget.cloned());
    // An approximate pass neither consumes nor publishes the exact floor.
    let exact = heap_factor == 1.0;
    let initial = shared_threshold.get();
    if initial > 0.0 && proximity.is_none() && exact {
        executor.seed_threshold(initial);
    }
    let mut results = executor.execute_sync()?;
    if let Some((config, terms)) = proximity {
        let terms: Vec<(Vec<u8>, f32)> = terms.into_iter().zip(idfs).collect();
        super::proximity::rescore_sync(
            reader,
            field,
            &terms,
            params,
            lengths.map(super::LengthSource::Docs),
            avg_field_len,
            config,
            &mut results,
        )?;
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.doc_id.cmp(&b.doc_id))
        });
        results.truncate(limit);
    } else if exact
        && results.len() >= limit
        && let Some(last) = results.last()
        && last.score > shared_threshold.get()
    {
        shared_threshold.set(last.score);
    }
    Ok(Box::new(TopKResultScorer::new(results)) as Box<dyn Scorer + 'a>)
}

/// Long-query cap: keep the `max_terms` rarest terms (highest idf) of a
/// field group, in their original order, dropping the rest. `0` keeps all.
pub(super) fn cap_terms(
    posting_lists: &mut Vec<(crate::structures::BlockPostingList, f32)>,
    term_bytes: &mut Vec<Vec<u8>>,
    max_terms: usize,
) {
    if max_terms == 0 || posting_lists.len() <= max_terms {
        return;
    }
    let mut by_idf: Vec<(usize, f32)> = posting_lists
        .iter()
        .enumerate()
        .map(|(i, (_, idf))| (i, *idf))
        .collect();
    by_idf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep = vec![false; posting_lists.len()];
    for (i, _) in by_idf.into_iter().take(max_terms) {
        keep[i] = true;
    }
    let mut index = 0;
    posting_lists.retain(|_| {
        let kept = keep[index];
        index += 1;
        kept
    });
    if term_bytes.len() == keep.len() {
        let mut index = 0;
        term_bytes.retain(|_| {
            let kept = keep[index];
            index += 1;
            kept
        });
    }
}

/// Over-fetch factor for chunked text MaxScore: the executor ranks chunks and
/// `Max` folding collapses a document's chunks into one hit, so fetch more
/// raw chunks than documents requested (same budget as sparse vectors).
const CHUNKED_TEXT_OVER_FETCH_FACTOR: f32 = 2.0;

/// Whether the text MaxScore fast path may serve a field when the caller
/// asked for positions.
///
/// Without positions it always may. With positions it may unless the field's
/// position mode tracks element ordinals (`positions` / `ordinal`): those
/// callers expect the raw encoded positions of every hit, which the top-k
/// executor does not produce. Phrase-only (`token_position`) fields report no
/// positions and chunked fields report chunk ordinals, both via MaxScore.
pub(super) fn text_maxscore_allowed(
    reader: &SegmentReader,
    field: crate::Field,
    collect_positions: bool,
) -> bool {
    if !collect_positions || reader.is_chunked_field(field) {
        return true;
    }
    !reader
        .schema()
        .get_field_entry(field)
        .and_then(|entry| entry.positions)
        .is_some_and(|mode| mode.tracks_ordinal())
}

/// Run BM25 MaxScore over a chunked field and fold the chunk hits into
/// documents with per-ordinal scores.
///
/// Posting ids are virtual chunk ids; every hit is resolved through the
/// segment's chunk map, grouped by document with `Max`, and served by a
/// `VectorResultScorer` so `matched_positions` carries `(ordinal, chunk
/// score)` pairs — the same shape sparse vectors produce. Cross-segment
/// threshold seeding is deliberately not applied: the k-th chunk score of a
/// full heap is not a document-level floor after `Max` folding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_chunked_text_maxscore<'a>(
    posting_lists: Vec<(crate::structures::BlockPostingList, f32)>,
    limit: usize,
    reader: &'a SegmentReader,
    field: crate::Field,
    predicate: Option<DocPredicate<'a>>,
    proximity: Option<(super::ProximityConfig, Vec<Vec<u8>>)>,
    heap_factor: f32,
    budget: Option<&super::SharedThreshold>,
) -> crate::Result<Box<dyn Scorer + 'a>> {
    if posting_lists.is_empty() {
        return Ok(Box::new(EmptyScorer) as Box<dyn Scorer + 'a>);
    }
    let proximity = proximity.filter(|(config, terms)| config.is_active() && terms.len() >= 2);
    let Some(chunk_map) = reader.chunk_map(field) else {
        return Err(crate::Error::Corruption(format!(
            "chunked text field '{}' has postings but segment {:016x} carries no chunk map",
            reader.schema().get_field_name(field).unwrap_or("?"),
            reader.meta().id,
        )));
    };
    let over_fetch = if proximity.is_some() {
        CHUNKED_TEXT_OVER_FETCH_FACTOR * super::proximity::PROXIMITY_OVER_FETCH as f32
    } else {
        CHUNKED_TEXT_OVER_FETCH_FACTOR
    };
    let executor_limit = bounded_sparse_executor_limit(limit, over_fetch)
        .min(chunk_map.num_chunks() as usize)
        .max(1);
    let idfs: Vec<f32> = posting_lists.iter().map(|(_, idf)| *idf).collect();
    let params = super::Bm25Params::for_field(reader.schema(), field);
    let mut executor = MaxScoreExecutor::text_chunked(
        posting_lists,
        chunk_map.avg_len(),
        executor_limit,
        chunk_map,
        params,
        heap_factor,
    )
    .with_metric_labels(
        reader.schema().index_label(),
        reader.schema().get_field_name(field).unwrap_or("?"),
    );
    if let Some(predicate) = predicate {
        // The executor walks virtual chunk ids; filters are per document.
        executor = executor.with_predicate(Box::new(move |vid| predicate(chunk_map.doc_id(vid))));
    }
    executor = executor.with_budget(budget.cloned());
    let mut raw = executor.execute_sync()?;
    if let Some((config, terms)) = proximity {
        let terms: Vec<(Vec<u8>, f32)> = terms.into_iter().zip(idfs).collect();
        super::proximity::rescore_sync(
            reader,
            field,
            &terms,
            params,
            Some(super::LengthSource::Chunks(chunk_map)),
            chunk_map.avg_len(),
            config,
            &mut raw,
        )?;
    }
    let combined = crate::segment::combine_ordinal_results(
        raw.into_iter().map(|hit| {
            let (doc_id, ordinal) = chunk_map.resolve(hit.doc_id);
            (doc_id, ordinal, hit.score)
        }),
        MultiValueCombiner::Max,
        limit,
    );
    Ok(Box::new(super::vector::VectorResultScorer::new(combined, field.0)) as Box<dyn Scorer + 'a>)
}

// ── Per-field grouping ───────────────────────────────────────────────────

/// Shared grouping result for per-field MaxScore.
pub(super) struct PerFieldGrouping {
    /// (field, avg_field_len, term_infos) for groups with 2+ terms
    pub multi_term_groups: Vec<(crate::Field, f32, Vec<TermQueryInfo>)>,
    /// Original indices of single-term and non-term SHOULD clauses (fallback scorers)
    pub fallback_indices: Vec<usize>,
    /// Limit per field group (over-fetched to compensate for cross-field scoring)
    pub per_field_limit: usize,
}

/// Group SHOULD clauses by field for per-field MaxScore.
/// Returns None if no group has 2+ terms (no optimization benefit).
pub(super) fn prepare_per_field_grouping(
    should: &[Arc<dyn Query>],
    reader: &SegmentReader,
    limit: usize,
    global_stats: Option<&Arc<GlobalStats>>,
    collect_positions: bool,
) -> Option<PerFieldGrouping> {
    let mut field_groups: rustc_hash::FxHashMap<crate::Field, Vec<(usize, TermQueryInfo)>> =
        rustc_hash::FxHashMap::default();
    let mut non_term_indices: Vec<usize> = Vec::new();

    for (i, q) in should.iter().enumerate() {
        if let super::QueryDecomposition::TextTerm(info) = q.decompose()
            && text_maxscore_allowed(reader, info.field, collect_positions)
        {
            field_groups.entry(info.field).or_default().push((i, info));
        } else {
            non_term_indices.push(i);
        }
    }

    if !field_groups.values().any(|g| g.len() >= 2) {
        return None;
    }

    let per_field_limit = super::max_candidate_limit(limit).min(reader.num_docs() as usize);

    let mut multi_term_groups = Vec::new();
    let mut fallback_indices = non_term_indices;

    for group in field_groups.into_values() {
        if group.len() >= 2 {
            let field = group[0].1.field;
            let avg_field_len = global_stats
                .map(|s| s.avg_field_len(field))
                .unwrap_or_else(|| reader.avg_field_len(field));
            let infos: Vec<_> = group.into_iter().map(|(_, info)| info).collect();
            multi_term_groups.push((field, avg_field_len, infos));
        } else {
            fallback_indices.push(group[0].0);
        }
    }

    Some(PerFieldGrouping {
        multi_term_groups,
        fallback_indices,
        per_field_limit,
    })
}

// ── Sparse MaxScore helpers ──────────────────────────────────────────────

const MAX_SPARSE_EXECUTOR_RESULTS: usize = 200_000;

pub(super) fn bounded_sparse_executor_limit(limit: usize, over_fetch_factor: f32) -> usize {
    let factor = if over_fetch_factor.is_finite() && over_fetch_factor >= 1.0 {
        over_fetch_factor.min(super::MAX_CANDIDATE_OVERSUBSCRIPTION as f32) as f64
    } else {
        1.0
    };
    let derived = (limit as f64 * factor).ceil();
    if !derived.is_finite() || derived >= usize::MAX as f64 {
        return MAX_SPARSE_EXECUTOR_RESULTS;
    }
    (derived as usize).min(MAX_SPARSE_EXECUTOR_RESULTS)
}

/// Physical single-value BMP segments need no ordinal over-fetch: every raw
/// hit is already a distinct final document. Genuine multi-value segments
/// retain the configured budget because aggregation can collapse many raw
/// ordinals into one document.
pub(crate) fn bmp_executor_limit(
    limit: usize,
    over_fetch_factor: f32,
    bmp: &crate::segment::reader::bmp::BmpIndex,
) -> usize {
    bmp_executor_limit_for_counts(
        limit,
        over_fetch_factor,
        bmp.is_single_valued(),
        bmp.num_real_docs() as usize,
    )
}

fn bmp_executor_limit_for_counts(
    limit: usize,
    over_fetch_factor: f32,
    single_valued: bool,
    num_real_docs: usize,
) -> usize {
    if single_valued {
        limit.min(num_real_docs)
    } else {
        bounded_sparse_executor_limit(limit, over_fetch_factor).min(num_real_docs)
    }
}

fn bmp_threshold<'a>(
    options: &'a super::ScorerOptions,
    combiner: MultiValueCombiner,
    single_valued: bool,
    heap_covers_limit: bool,
) -> super::bmp::BmpThreshold<'a> {
    if !single_valued && combiner != MultiValueCombiner::Max {
        return super::bmp::BmpThreshold::default();
    }
    super::bmp::BmpThreshold {
        initial: options.initial_threshold,
        shared: options.shared_threshold.as_ref(),
        // A full heap's k-th score is a valid cross-segment floor only when
        // the heap is as deep as the requested result window. A segment with
        // fewer documents than `limit` fills its (clamped) heap early; its
        // k-th score says nothing about the global top-`limit` set and used
        // to erase every lower-scoring segment's results.
        publish: single_valued && heap_covers_limit,
    }
}

/// Build a sparse MaxScoreExecutor from decomposed sparse infos.
///
/// Returns the executor + representative info (for combiner/field), or None
/// if the sparse index doesn't exist or no query dims match.
pub(crate) fn build_sparse_maxscore_executor<'a>(
    infos: &[SparseTermQueryInfo],
    reader: &'a SegmentReader,
    limit: usize,
    predicate: Option<DocPredicate<'a>>,
) -> Option<(MaxScoreExecutor<'a>, SparseTermQueryInfo)> {
    let field = infos[0].field;
    let si = reader.sparse_index(field)?;
    let query_terms: Vec<(u32, f32)> = infos
        .iter()
        .filter(|info| info.candidate && si.has_dimension(info.dim_id))
        .map(|info| (info.dim_id, info.weight))
        .collect();
    if query_terms.is_empty() {
        return None;
    }
    // Sparse postings contain one entry per stored ordinal, so the raw heap
    // may legitimately need to exceed the real document count before ordinal
    // scores can be combined back into documents.
    let executor_limit = bounded_sparse_executor_limit(limit, infos[0].over_fetch_factor)
        .min(si.total_vectors as usize);
    let mut executor =
        MaxScoreExecutor::sparse(si, query_terms, executor_limit, infos[0].heap_factor)
            .with_metric_labels(
                reader.schema().index_label(),
                reader.schema().get_field_name(field).unwrap_or("?"),
            );
    if let Some(pred) = predicate {
        executor = executor.with_predicate(pred);
    }
    Some((executor, infos[0]))
}

/// Build a sparse BMP executor from decomposed sparse infos.
///
/// Auto-detected: called when the field has a BMP index. Returns scored
/// results directly (BMP is always synchronous), or None if no BMP index
/// exists for the field.
pub(crate) fn build_sparse_bmp_results(
    infos: &[SparseTermQueryInfo],
    reader: &SegmentReader,
    limit: usize,
    options: &super::ScorerOptions,
) -> crate::Result<Option<(Vec<ScoredDoc>, SparseTermQueryInfo)>> {
    build_sparse_bmp_results_inner(infos, reader, limit, None, options)
}

/// Build a sparse BMP executor with a document predicate filter.
///
/// The predicate is applied during BMP scoring (not post-filter), ensuring
/// the collector only contains valid documents and the threshold evolves correctly.
pub(crate) fn build_sparse_bmp_results_filtered(
    infos: &[SparseTermQueryInfo],
    reader: &SegmentReader,
    limit: usize,
    predicate: &dyn Fn(crate::DocId) -> bool,
    options: &super::ScorerOptions,
) -> crate::Result<Option<(Vec<ScoredDoc>, SparseTermQueryInfo)>> {
    build_sparse_bmp_results_inner(infos, reader, limit, Some(predicate), options)
}

fn build_sparse_bmp_results_inner(
    infos: &[SparseTermQueryInfo],
    reader: &SegmentReader,
    limit: usize,
    predicate: Option<&dyn Fn(crate::DocId) -> bool>,
    options: &super::ScorerOptions,
) -> crate::Result<Option<(Vec<ScoredDoc>, SparseTermQueryInfo)>> {
    let infos = options
        .lsp_plan
        .as_ref()
        .map_or(infos, |plan| plan.infos.as_ref());
    let Some(&info) = infos.first() else {
        return Ok(None);
    };
    let field = info.field;
    let Some(bmp) = reader.bmp_index(field) else {
        return Ok(None);
    };
    // Global LSP already resolved and quantized the complete query. Rebuilding
    // these two vectors in every segment used to duplicate bounded query work
    // and allocations. Direct/single-segment calls still prepare locally.
    let (candidate_terms, scoring_terms) = if options.lsp_plan.is_some() {
        (Vec::new(), Vec::new())
    } else {
        let candidate_terms: Vec<_> = infos
            .iter()
            .filter(|info| info.candidate)
            .map(|info| (info.dim_id, info.weight))
            .collect();
        if candidate_terms.is_empty() {
            return Ok(None);
        }
        let scoring_terms = infos
            .iter()
            .map(|info| (info.dim_id, info.weight))
            .collect();
        (candidate_terms, scoring_terms)
    };
    let executor_limit = bmp_executor_limit(limit, info.over_fetch_factor, bmp);
    let lsp_gamma = info
        .lsp_gamma
        .unwrap_or_else(|| super::bmp::recommended_lsp_gamma(executor_limit));
    let field_label = reader.schema().get_field_name(field).unwrap_or("?");
    // The per-segment `limit` may already be clamped to the segment's doc
    // count, so validate the heap depth against the *query* window carried by
    // the shared floor, not against `limit`.
    let heap_covers_floor = options
        .shared_threshold
        .as_ref()
        .is_none_or(|shared| shared.covers(executor_limit));
    let threshold = bmp_threshold(
        options,
        info.combiner,
        bmp.is_single_valued(),
        heap_covers_floor,
    );
    let results = if let Some(predicate) = predicate {
        super::bmp::execute_bmp_filtered_with_threshold(
            bmp,
            reader.schema().index_label(),
            field_label,
            &candidate_terms,
            &scoring_terms,
            executor_limit,
            info.heap_factor,
            lsp_gamma,
            options.lsp_plan.as_deref(),
            predicate,
            threshold,
        )
    } else {
        super::bmp::execute_bmp_with_threshold(
            bmp,
            reader.schema().index_label(),
            field_label,
            &candidate_terms,
            &scoring_terms,
            executor_limit,
            info.heap_factor,
            lsp_gamma,
            options.lsp_plan.as_deref(),
            threshold,
        )
    }?;
    Ok(Some((results, info)))
}

/// Combine raw MaxScore results with ordinal deduplication into a scorer.
pub(crate) fn combine_sparse_results<'a>(
    raw: Vec<ScoredDoc>,
    combiner: MultiValueCombiner,
    field: crate::Field,
    limit: usize,
) -> Box<dyn Scorer + 'a> {
    let combined = crate::segment::combine_ordinal_results(
        raw.into_iter().map(|r| (r.doc_id, r.ordinal, r.score)),
        combiner,
        limit,
    );
    Box::new(super::vector::VectorResultScorer::new(combined, field.0))
}

/// Extract all sparse term infos from SHOULD clauses, flattening SparseVectorQuery.
///
/// Returns `None` if any SHOULD clause is not decomposable into sparse term queries
/// or if the resulting infos span multiple fields.
pub(super) fn extract_all_sparse_infos(
    should: &[Arc<dyn Query>],
) -> Option<Vec<SparseTermQueryInfo>> {
    let mut all = Vec::new();
    for q in should {
        match q.decompose() {
            super::QueryDecomposition::SparseTerms(infos) => all.extend(infos),
            _ => return None,
        }
    }
    if all.is_empty() {
        return None;
    }
    let field = all[0].field;
    if !all.iter().all(|i| i.field == field) {
        return None;
    }
    Some(all)
}

// ── Predicate helpers ────────────────────────────────────────────────────

/// Chain multiple predicates into a single combined predicate.
pub(super) fn chain_predicates<'a>(predicates: Vec<DocPredicate<'a>>) -> DocPredicate<'a> {
    if predicates.len() == 1 {
        return predicates.into_iter().next().unwrap();
    }
    Box::new(move |doc_id| predicates.iter().all(|p| p(doc_id)))
}

/// Refining a small accumulator beats materializing a clause's full bitset
/// when the clause is estimated to match at least this many times more docs
/// than the accumulator holds. Probes cost ~30-40ns (fast-field closure) vs
/// ~2-5ns per materialized entry, hence the margin.
const PROBE_ADVANTAGE: u64 = 8;

/// Build a combined DocBitset from MUST and MUST_NOT clause bitsets.
///
/// Selectivity-aware: MUST clauses are evaluated narrowest-first (posting
/// doc counts are exact, fast-field ranges are sampled), and once the
/// accumulator is much smaller than a remaining clause's estimate, that
/// clause is applied as O(|acc|) per-doc predicate probes instead of being
/// materialized. A `type = X` filter matching millions of docs is then never
/// iterated when a recent-dates range keeps only a few thousand candidates —
/// each surviving doc just probes the `type` fast field once.
///
/// Returns None if a clause that must be materialized doesn't support bitset
/// creation (probed clauses only need `as_doc_predicate`).
/// The resulting bitset enables ~2ns per-doc lookups in BMP (vs ~30-40ns for closures).
pub(super) fn build_combined_bitset(
    must: &[std::sync::Arc<dyn super::Query>],
    must_not: &[std::sync::Arc<dyn super::Query>],
    reader: &crate::segment::SegmentReader,
    options: &super::ScorerOptions,
) -> Option<super::DocBitset> {
    if options.stop_if_expired() {
        return None;
    }
    if must.is_empty() && must_not.is_empty() {
        return None;
    }

    let num_docs = reader.num_docs();

    // Order MUST clauses by estimated match count, narrowest first. Unknown
    // estimates sort last (pessimistically treated as matching everything).
    let mut order: Vec<(usize, u64)> = must
        .iter()
        .enumerate()
        .map(|(i, q)| {
            (
                i,
                q.bitset_cardinality_estimate(reader)
                    .unwrap_or(num_docs as u64),
            )
        })
        .collect();
    order.sort_unstable_by_key(|&(_, est)| est);

    let mut result: Option<super::DocBitset> = None;
    let mut acc_count: u64 = 0;

    for (idx, est) in order {
        let q = &must[idx];
        match result {
            None => {
                // Seed: materialize the narrowest clause.
                let bs = options.doc_bitset(q.as_ref(), reader)?;
                acc_count = bs.count() as u64;
                result = Some(bs);
            }
            Some(ref mut acc) => {
                let mut probed = false;
                if acc_count.saturating_mul(PROBE_ADVANTAGE) <= est
                    && let Some(pred) = q.as_doc_predicate(reader)
                {
                    acc.retain(&*pred);
                    probed = true;
                }
                if !probed {
                    let bs = options.doc_bitset(q.as_ref(), reader)?;
                    acc.intersect_with(&bs);
                }
                acc_count = acc.count() as u64;
                log::debug!(
                    "[planner] MUST clause {}: est={} probed={} acc={}",
                    idx,
                    est,
                    probed,
                    acc_count,
                );
            }
        }
        if acc_count == 0 {
            // AND is already empty — nothing can revive it.
            break;
        }
    }

    // Subtract MUST_NOT bitsets (probe-refined when the accumulator is small)
    for q in must_not {
        match result {
            None => {
                // No MUST clauses — start with all-ones, then subtract
                let bs = options.doc_bitset(q.as_ref(), reader)?;
                let mut all = super::DocBitset::new(num_docs);
                all.bits.fill(u64::MAX);
                // Clear bits beyond num_docs
                let tail_bits = num_docs as usize % 64;
                if tail_bits > 0 && !all.bits.is_empty() {
                    let last = all.bits.len() - 1;
                    all.bits[last] &= (1u64 << tail_bits) - 1;
                }
                all.subtract(&bs);
                acc_count = all.count() as u64;
                result = Some(all);
            }
            Some(ref mut acc) => {
                let est = q
                    .bitset_cardinality_estimate(reader)
                    .unwrap_or(num_docs as u64);
                let mut probed = false;
                if acc_count.saturating_mul(PROBE_ADVANTAGE) <= est
                    && let Some(pred) = q.as_doc_predicate(reader)
                {
                    acc.retain(&|doc| !pred(doc));
                    probed = true;
                }
                if !probed {
                    let bs = options.doc_bitset(q.as_ref(), reader)?;
                    acc.subtract(&bs);
                }
                acc_count = acc.count() as u64;
            }
        }
    }

    if options.stop_if_expired() {
        None
    } else {
        result
    }
}

// ── Result scorers ───────────────────────────────────────────────────────

/// Union of a scored result stream with the documents of a filter bitset:
/// documents the stream does not produce are yielded with score 0. Used when
/// a filtered text MaxScore pass finds fewer than `limit` scored documents,
/// so documents matching only the MUST clauses still fill the result list
/// (Boolean semantics: SHOULD is optional once a MUST clause exists).
pub(super) struct BitsetFillScorer<'a> {
    inner: Box<dyn Scorer + 'a>,
    bitset: std::sync::Arc<super::DocBitset>,
    /// Next bitset document not yet consumed.
    next_bit: Option<DocId>,
    current: DocId,
    on_inner: bool,
}

impl<'a> BitsetFillScorer<'a> {
    pub(super) fn new(
        inner: Box<dyn Scorer + 'a>,
        bitset: std::sync::Arc<super::DocBitset>,
    ) -> Self {
        let next_bit = bitset.next_set_bit(0);
        let mut scorer = Self {
            inner,
            bitset,
            next_bit,
            current: 0,
            on_inner: false,
        };
        scorer.settle();
        scorer
    }

    /// Position on the smaller of the two heads.
    fn settle(&mut self) {
        let inner_doc = self.inner.doc();
        let bit_doc = self.next_bit.unwrap_or(TERMINATED);
        self.current = inner_doc.min(bit_doc);
        self.on_inner = inner_doc == self.current && inner_doc != TERMINATED;
    }
}

impl super::docset::DocSet for BitsetFillScorer<'_> {
    fn doc(&self) -> DocId {
        self.current
    }

    fn advance(&mut self) -> DocId {
        if self.current == TERMINATED {
            return TERMINATED;
        }
        if self.on_inner {
            self.inner.advance();
        }
        if self.next_bit == Some(self.current) {
            self.next_bit = self
                .current
                .checked_add(1)
                .and_then(|d| self.bitset.next_set_bit(d));
        }
        self.settle();
        self.current
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if target <= self.current {
            return self.current;
        }
        self.inner.seek(target);
        self.next_bit = self.bitset.next_set_bit(target);
        self.settle();
        self.current
    }

    fn size_hint(&self) -> u32 {
        self.bitset.count().max(self.inner.size_hint())
    }
}

impl Scorer for BitsetFillScorer<'_> {
    fn score(&self) -> Score {
        if self.on_inner {
            self.inner.score()
        } else {
            0.0
        }
    }

    fn matched_positions(&self) -> Option<super::MatchedPositions> {
        if self.on_inner {
            self.inner.matched_positions()
        } else {
            None
        }
    }
}

/// Scorer that iterates over pre-computed top-k results
pub(super) struct TopKResultScorer {
    results: Vec<ScoredDoc>,
    position: usize,
}

impl TopKResultScorer {
    pub(super) fn new(mut results: Vec<ScoredDoc>) -> Self {
        // Sort by doc_id ascending — required for DocSet seek() correctness
        results.sort_unstable_by_key(|r| r.doc_id);
        Self {
            results,
            position: 0,
        }
    }
}

impl super::docset::DocSet for TopKResultScorer {
    fn doc(&self) -> DocId {
        if self.position < self.results.len() {
            self.results[self.position].doc_id
        } else {
            TERMINATED
        }
    }

    fn advance(&mut self) -> DocId {
        self.position += 1;
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let remaining = &self.results[self.position..];
        self.position += remaining.partition_point(|r| r.doc_id < target);
        self.doc()
    }

    fn size_hint(&self) -> u32 {
        (self.results.len() - self.position) as u32
    }
}

impl Scorer for TopKResultScorer {
    fn score(&self) -> Score {
        if self.position < self.results.len() {
            self.results[self.position].score
        } else {
            0.0
        }
    }
}

// Sparse executors share `crate::query::vector::VectorResultScorer` with the
// dense queries (see `combine_sparse_results`).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bmp_single_value_limit_does_not_overfetch() {
        assert_eq!(bounded_sparse_executor_limit(320, 99.0), 640);
        assert_eq!(bmp_executor_limit_for_counts(320, 2.0, true, 10_000), 320);
        assert_eq!(bmp_executor_limit_for_counts(320, 2.0, false, 10_000), 640);
    }

    #[test]
    fn bmp_threshold_is_only_used_in_final_score_space() {
        let shared = super::super::SharedThreshold::new();
        shared.raise(7.0);
        let options = super::super::ScorerOptions {
            collect_positions: false,
            initial_threshold: 5.0,
            shared_threshold: Some(shared),
            lsp_plan: None,
            global_stats: None,
        };

        let single_sum = bmp_threshold(&options, MultiValueCombiner::Sum, true, true);
        assert_eq!(single_sum.initial, 5.0);
        assert!(single_sum.shared.is_some());
        assert!(single_sum.publish);

        let multi_max = bmp_threshold(&options, MultiValueCombiner::Max, false, true);
        assert!(multi_max.shared.is_some());
        assert!(!multi_max.publish);

        let multi_sum = bmp_threshold(&options, MultiValueCombiner::Sum, false, true);
        assert_eq!(multi_sum.initial, 0.0);
        assert!(multi_sum.shared.is_none());
        assert!(!multi_sum.publish);
    }

    /// A segment with fewer real docs than the requested window clamps its
    /// heap below `limit`; its full-heap threshold must then stay private.
    #[test]
    fn bmp_threshold_from_a_clamped_heap_is_never_published() {
        let shared = super::super::SharedThreshold::new();
        let options = super::super::ScorerOptions {
            collect_positions: false,
            initial_threshold: 0.0,
            shared_threshold: Some(shared),
            lsp_plan: None,
            global_stats: None,
        };

        let clamped = bmp_threshold(&options, MultiValueCombiner::Sum, true, false);
        assert!(clamped.shared.is_some(), "reading a valid floor stays safe");
        assert!(
            !clamped.publish,
            "a heap shallower than the result window must not publish a floor"
        );
    }
}
