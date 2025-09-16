//! Seismic nomination over immutable segment runs, with one exact forward scorer.
use super::{MultiValueCombiner, ScoreCollector, ScorerOptions, SparseTermQueryInfo};
use crate::segment::seismic::{SeismicIndex, TermRun};
use crate::segment::{VectorOrdinals, VectorSearchResult};
use crate::{Error, Result};

mod required;
mod scoring;
pub(super) use required::{membership, required_scorer};
pub(super) use scoring::score_candidates;
use scoring::{PreparedQuery, document_ordinals, score_row};

/// Bounded nomination scratch per segment, independent of corpus size.
const MAX_NOMINATED_DOCUMENTS: usize = 262_144;
const MAX_VISITED_CLUSTERS: usize = 262_144;
const MAX_EXACT_FILTER_DOCUMENTS: u32 = 4096;

pub(super) fn validate_options(cut: usize, factor: f32) -> Result<()> {
    if cut == 0
        || cut > super::MAX_QUERY_TERMS
        || !factor.is_finite()
        || !(0.0..=1.0).contains(&factor)
    {
        return Err(Error::Query(
            "Seismic requires cut in [1, 64] and finite factor in [0, 1]".into(),
        ));
    }
    Ok(())
}

/// Summaries have no shared heap state. Split only inside the caller's existing
/// search pool, with one aggregate score buffer and at most four leaf tasks.
fn score_term_runs(
    runs: &[TermRun<'_>],
    query: &[(u32, f32)],
    scores: &mut [f32],
    options: &ScorerOptions,
) -> bool {
    #[cfg(feature = "sync")]
    if runs.len() > 1 && rayon::current_thread_index().is_some() {
        return score_term_runs_parallel(
            runs,
            query,
            scores,
            options,
            rayon::current_num_threads().min(4),
        );
    }
    score_term_runs_sequential(runs, query, scores, options)
}

fn score_term_runs_sequential(
    runs: &[TermRun<'_>],
    query: &[(u32, f32)],
    mut scores: &mut [f32],
    options: &ScorerOptions,
) -> bool {
    for &run in runs {
        if options.stop_if_expired() {
            return false;
        }
        let (run_scores, remaining) = scores.split_at_mut(run.cluster_count());
        run.score_summaries(query, run_scores);
        scores = remaining;
    }
    !options.stop_if_expired()
}

#[cfg(feature = "sync")]
fn score_term_runs_parallel(
    runs: &[TermRun<'_>],
    query: &[(u32, f32)],
    scores: &mut [f32],
    options: &ScorerOptions,
    tasks: usize,
) -> bool {
    if tasks < 2 || runs.len() < 2 {
        return score_term_runs_sequential(runs, query, scores, options);
    }
    let middle = runs.len() / 2;
    let (left, right) = runs.split_at(middle);
    let split = left.iter().map(|run| run.cluster_count()).sum();
    let (left_scores, right_scores) = scores.split_at_mut(split);
    let (left_complete, right_complete) = rayon::join(
        || score_term_runs_parallel(left, query, left_scores, options, tasks / 2),
        || score_term_runs_parallel(right, query, right_scores, options, tasks - tasks / 2),
    );
    left_complete && right_complete
}

/// Collect documents only after combining every matching ordinal. This avoids
/// losing a document whose many moderate ordinals outrank one strong ordinal.
pub(super) fn execute(
    index: &SeismicIndex,
    infos: &[SparseTermQueryInfo],
    k: usize,
    predicate: &dyn Fn(u32) -> bool,
    options: &ScorerOptions,
    filtered: bool,
    labels: (&str, &str),
) -> Result<Vec<VectorSearchResult>> {
    let timer = crate::observe::Timer::start();
    let Some(info) = infos.first() else {
        return Ok(Vec::new());
    };
    validate_options(info.seismic_cut, info.seismic_factor)?;
    if k == 0 || index.is_empty() {
        return Ok(Vec::new());
    }
    let terms: Vec<_> = infos
        .iter()
        .filter(|info| info.weight != 0.0)
        .map(|info| (info.dim_id, info.weight))
        .collect();
    if terms.len() > super::MAX_QUERY_TERMS || terms.iter().any(|(_, w)| !w.is_finite()) {
        return Err(Error::Query("invalid Seismic scoring dimensions".into()));
    }
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    let query = PreparedQuery::new(terms);
    let mut heap = ScoreCollector::new(k);
    heap.seed_threshold(options.initial_threshold);
    let mut ordinals = VectorOrdinals::new();
    let mut scored = 0usize;
    let single_valued = index.is_single_valued();
    let mut collect = |doc: u32, row: u32, heap: &mut ScoreCollector| -> Result<()> {
        if !predicate(doc) {
            return Ok(());
        }
        let score = if single_valued {
            let Some(score) = score_row(index, row, &query)? else {
                return Ok(());
            };
            score
        } else {
            document_ordinals(index, doc, &query, &mut ordinals, None)?;
            if ordinals.is_empty() {
                return Ok(());
            }
            info.combiner.combine(&ordinals)
        };
        if !score.is_finite() {
            return Err(Error::Query("Seismic combined score overflow".into()));
        }
        heap.insert(doc, score);
        if heap.len() == k
            && let Some(shared) = &options.shared_threshold
            && shared.covers(k)
        {
            shared.raise(heap.threshold());
        }
        scored += 1;
        Ok(())
    };
    let exhaustive = info.exhaustive || options.complete_text_matches;
    let selective = options
        .eligibility
        .as_ref()
        .filter(|bits| bits.count() <= MAX_EXACT_FILTER_DOCUMENTS);
    let mut clusters = 0usize;
    let mut truncated = false;
    let mut filter_scan = false;
    if !exhaustive && let Some(bits) = selective {
        filter_scan = true;
        let mut next = bits.next_set_bit(0);
        while let Some(doc) = next {
            if options.stop_if_expired() {
                break;
            }
            if let Some(row) = index.rows_for_document(doc).next() {
                collect(doc, row, &mut heap)?;
            }
            next = doc.checked_add(1).and_then(|next| bits.next_set_bit(next));
        }
    } else if exhaustive {
        let mut previous = None;
        for row in 0..index.len() {
            let doc = index.key(row).doc;
            if previous == Some(doc) {
                continue;
            }
            previous = Some(doc);
            if row.is_multiple_of(1024) && options.stop_if_expired() {
                break;
            }
            collect(doc, row, &mut heap)?;
        }
    } else {
        let mut candidate_terms: Vec<_> = infos
            .iter()
            .filter(|info| info.candidate && info.weight != 0.0)
            .map(|info| (info.dim_id, info.weight))
            .collect();
        candidate_terms
            .sort_unstable_by(|a, b| b.1.abs().total_cmp(&a.1.abs()).then(a.0.cmp(&b.0)));
        let mut dimensions = Vec::with_capacity(candidate_terms.len());
        candidate_terms.retain(|&(dim, _)| {
            if dimensions.contains(&dim) {
                false
            } else {
                dimensions.push(dim);
                true
            }
        });
        candidate_terms.truncate(info.seismic_cut);
        let mut seen = rustc_hash::FxHashSet::default();
        let mut order = Vec::new();
        let mut summary_scores = Vec::new();
        let mut runs = smallvec::SmallVec::<[TermRun<'_>; 4]>::new();
        'terms: for (position, (dimension, _)) in candidate_terms.into_iter().enumerate() {
            order.clear();
            runs.clear();
            let mut cluster_count = 0;
            for run in index.term_runs(dimension) {
                if options.stop_if_expired() {
                    break 'terms;
                }
                if run.cluster_count() == 0 {
                    continue;
                }
                // Admit the aggregate scratch before allocating or dispatching.
                if run.cluster_count()
                    > MAX_VISITED_CLUSTERS.saturating_sub(clusters + cluster_count)
                {
                    truncated = true;
                    break;
                }
                cluster_count += run.cluster_count();
                runs.push(run);
            }
            summary_scores.resize(cluster_count, 0.0);
            if !score_term_runs(&runs, query.terms(), &mut summary_scores, options) {
                break;
            }
            order.reserve(cluster_count);
            order.extend(
                summary_scores
                    .iter()
                    .copied()
                    .zip(runs.iter().flat_map(|run| run.clusters())),
            );
            if position == 0 {
                // Establish one strong threshold across all preserved runs,
                // rather than exhausting a weak source before seeing the next.
                order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            }
            for (bound, cluster) in order.drain(..) {
                clusters += 1;
                if clusters.is_multiple_of(1024) && options.stop_if_expired() {
                    break 'terms;
                }
                // This summary is a nomination proxy, not an exact bound.
                // Multi-value sums cannot use a per-row bound safely.
                let prune = single_valued || matches!(info.combiner, MultiValueCombiner::Max);
                if prune && heap.len() == k && bound < info.seismic_factor * heap.threshold() {
                    continue;
                }
                for row in cluster.rows() {
                    if seen.len().is_multiple_of(1024) && options.stop_if_expired() {
                        break 'terms;
                    }
                    let doc = index.key(row).doc;
                    if seen.len() == MAX_NOMINATED_DOCUMENTS && !seen.contains(&doc) {
                        truncated = true;
                        break 'terms;
                    }
                    if seen.insert(doc) {
                        collect(doc, row, &mut heap)?;
                    }
                }
            }
            if truncated {
                break 'terms;
            }
        }
        if filtered && heap.len() < k && !options.stop_if_expired() {
            // A selective predicate can reject every retained top-L nominee.
            // Backfill from the shared owner instead of reporting false empty
            // results solely because the nomination lists were pruned.
            filter_scan = true;
            let mut previous = None;
            for row in 0..index.len() {
                let doc = index.key(row).doc;
                if previous == Some(doc) {
                    continue;
                }
                previous = Some(doc);
                if row.is_multiple_of(1024) && options.stop_if_expired() {
                    break;
                }
                if !seen.contains(&doc) {
                    collect(doc, row, &mut heap)?;
                }
            }
        }
    }
    #[cfg(feature = "query-diagnostics")]
    crate::search_diagnostics::update(|work| {
        work.seismic_clusters += clusters as u64;
        work.seismic_documents += scored as u64;
        work.seismic_budget_truncations += u64::from(truncated);
        work.seismic_filter_scans += u64::from(filter_scan);
    });
    if truncated {
        if let Some(shared) = &options.shared_threshold {
            shared.mark_truncated();
        }
        log::warn!(
            "Seismic nomination budget reached: scored_documents={scored}, visited_clusters={clusters}; use exhaustive search for complete coverage"
        );
    }
    log::debug!(
        "Seismic search: exhaustive={exhaustive}, filter_scan={filter_scan}, scored_documents={scored}, visited_clusters={clusters}, truncated={truncated}"
    );
    // Only winning documents retain ordinal lists. Candidate scratch is reused
    // above, so memory does not grow with candidates times values per document.
    let mut results = Vec::with_capacity(heap.len());
    let mut matched_ordinals = VectorOrdinals::new();
    for (doc, score, _) in heap.into_sorted_results() {
        if !options.collect_positions {
            results.push(VectorSearchResult::single(doc, score));
            continue;
        }
        document_ordinals(
            index,
            doc,
            &query,
            &mut ordinals,
            Some(&mut matched_ordinals),
        )?;
        results.push(VectorSearchResult::with_ordinals(
            doc,
            score,
            matched_ordinals.clone(),
        ));
    }
    crate::observe::seismic_query(
        labels.0,
        labels.1,
        timer.secs(),
        clusters,
        scored,
        truncated,
    );
    Ok(results)
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "native"))]
mod integration_tests;
