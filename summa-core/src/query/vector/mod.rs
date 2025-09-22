//! Vector query types for dense, binary dense, and sparse vector search

mod binary_dense;
mod combiner;
mod dense;
mod sparse;

pub use binary_dense::BinaryDenseVectorQuery;
pub use combiner::MultiValueCombiner;
pub use dense::{
    DEFAULT_DENSE_RERANK_FACTOR, DenseVectorQuery, MAX_DENSE_NPROBE, MAX_DENSE_RERANK_FACTOR,
};
pub use sparse::{SparseTermQuery, SparseVectorQuery};

use crate::segment::VectorSearchResult;
use crate::{DocId, Score, TERMINATED};

use super::traits::{MatchedPositions, Scorer};
use super::{ScoredPosition, SearchResult};

/// Convert a result's per-ordinal contributions into scored positions.
#[inline]
fn ordinal_positions(result: &VectorSearchResult) -> Vec<ScoredPosition> {
    result
        .ordinals
        .iter()
        .map(|&(ordinal, score)| ScoredPosition::new(ordinal, score))
        .collect()
}

/// Shared scorer for ranked vector results (dense, binary dense and sparse
/// executors all produce a ranked `Vec<VectorSearchResult>`).
///
/// The executor returns results ranked by score. The `DocSet` contract needs
/// doc-id order, but a vector query that is the *top-level* query never needs
/// to be walked at all: `precomputed_top_k` hands the ranked list straight to
/// the searcher, skipping the doc-id sort, the `TopKCollector` heap and the
/// final re-sort by score. The doc-id sort therefore happens lazily on the
/// first `advance`/`seek`; until then the scorer reports the minimum doc id,
/// which is exactly where a doc-ordered walk would start.
pub(crate) struct VectorResultScorer {
    results: Vec<VectorSearchResult>,
    /// Cursor into `results` once they are doc-ordered.
    position: usize,
    /// Index of the minimum doc id while `results` are still score-ordered.
    head: usize,
    doc_ordered: bool,
    field_id: u32,
}

impl VectorResultScorer {
    pub(crate) fn new(results: Vec<VectorSearchResult>, field_id: u32) -> Self {
        let head = results
            .iter()
            .enumerate()
            .min_by_key(|(_, result)| result.doc_id)
            .map(|(index, _)| index)
            .unwrap_or(0);
        Self {
            results,
            position: 0,
            head,
            doc_ordered: false,
            field_id,
        }
    }

    #[inline]
    fn ensure_doc_order(&mut self) {
        if !self.doc_ordered {
            debug_assert_eq!(self.position, 0);
            self.results.sort_unstable_by_key(|result| result.doc_id);
            self.doc_ordered = true;
            self.head = 0;
        }
    }

    #[inline]
    fn current(&self) -> Option<&VectorSearchResult> {
        let index = if self.doc_ordered {
            self.position
        } else {
            self.head
        };
        self.results.get(index)
    }
}

impl super::docset::DocSet for VectorResultScorer {
    fn doc(&self) -> DocId {
        self.current().map_or(TERMINATED, |result| result.doc_id)
    }

    fn advance(&mut self) -> DocId {
        self.ensure_doc_order();
        self.position = (self.position + 1).min(self.results.len());
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.ensure_doc_order();
        let remaining = &self.results[self.position..];
        let offset = remaining.partition_point(|result| result.doc_id < target);
        self.position += offset;
        self.doc()
    }

    fn size_hint(&self) -> u32 {
        (self.results.len() - self.position) as u32
    }
}

impl Scorer for VectorResultScorer {
    fn score(&self) -> Score {
        self.current().map_or(0.0, |result| result.score)
    }

    fn matched_positions(&self) -> Option<MatchedPositions> {
        let result = self.current()?;
        Some(vec![(self.field_id, ordinal_positions(result))])
    }

    fn precomputed_top_k(
        &mut self,
        limit: usize,
        collect_positions: bool,
    ) -> Option<(Vec<SearchResult>, u32)> {
        if self.doc_ordered || self.position != 0 {
            // Already being driven as a DocSet.
            return None;
        }
        let mut results = std::mem::take(&mut self.results);
        let total_seen = results.len() as u32;
        // Same order a TopKCollector produces: score desc, then doc id asc.
        // Executors already emit this order, so the sort is a linear check.
        results.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        results.truncate(limit);
        let field_id = self.field_id;
        let ranked = results
            .into_iter()
            .map(|result| SearchResult {
                doc_id: result.doc_id,
                score: result.score,
                segment_id: 0,
                positions: if collect_positions {
                    vec![(field_id, ordinal_positions(&result))]
                } else {
                    Vec::new()
                },
            })
            .collect();
        Some((ranked, total_seen))
    }
}

#[cfg(test)]
mod tests {
    use super::super::docset::DocSet;
    use super::*;
    use crate::query::{Collector, TopKCollector};

    /// `(doc_id, score, ordinals)` as an executor would rank them.
    type Ranked<'a> = &'a [(u32, f32, &'a [(u32, f32)])];

    fn ranked(entries: Ranked<'_>) -> Vec<VectorSearchResult> {
        entries
            .iter()
            .map(|&(doc_id, score, ordinals)| {
                VectorSearchResult::new(doc_id, score, ordinals.to_vec())
            })
            .collect()
    }

    /// The standalone hand-over must produce exactly what driving the scorer
    /// through a position-collecting `TopKCollector` produces, including tie
    /// order (equal scores → lower doc id first) and per-ordinal positions.
    #[test]
    fn precomputed_top_k_matches_driven_top_k_collector_including_ties() {
        let entries: Ranked<'_> = &[
            (7, 0.9, &[(0, 0.9)]),
            (3, 0.9, &[(1, 0.9), (0, 0.2)]),
            (12, 0.5, &[(0, 0.5)]),
            (1, 0.5, &[(2, 0.5)]),
            (9, 0.1, &[(0, 0.1)]),
        ];
        for limit in [0usize, 1, 3, 5, 10] {
            for collect_positions in [false, true] {
                let mut driven = VectorResultScorer::new(ranked(entries), 4);
                let mut collector = if collect_positions {
                    TopKCollector::with_positions(limit)
                } else {
                    TopKCollector::new(limit)
                };
                let mut doc = driven.doc();
                while doc != TERMINATED {
                    let score = driven.score();
                    if collect_positions && collector.would_collect(doc, score) {
                        let positions = driven.matched_positions().unwrap_or_default();
                        collector.collect_owned(doc, score, positions);
                    } else {
                        collector.collect(doc, score, &[]);
                    }
                    doc = driven.advance();
                }
                let expected = collector.into_results_with_count();

                let mut standalone = VectorResultScorer::new(ranked(entries), 4);
                let actual = standalone
                    .precomputed_top_k(limit, collect_positions)
                    .expect("undriven scorer hands over its ranked list");

                assert_eq!(actual.1, expected.1, "total_seen (limit {limit})");
                assert_eq!(actual.0.len(), expected.0.len(), "len (limit {limit})");
                for (left, right) in actual.0.iter().zip(&expected.0) {
                    assert_eq!(left.doc_id, right.doc_id, "doc order (limit {limit})");
                    assert_eq!(left.score.to_bits(), right.score.to_bits());
                    assert_eq!(left.segment_id, right.segment_id);
                    assert_eq!(left.positions.len(), right.positions.len());
                    for ((lf, lp), (rf, rp)) in left.positions.iter().zip(&right.positions) {
                        assert_eq!(lf, rf);
                        let lp: Vec<_> =
                            lp.iter().map(|p| (p.position, p.score.to_bits())).collect();
                        let rp: Vec<_> =
                            rp.iter().map(|p| (p.position, p.score.to_bits())).collect();
                        assert_eq!(lp, rp, "ordinal positions (limit {limit})");
                    }
                }
            }
        }
    }

    /// Before the lazy doc-id sort, `doc()`/`score()`/`matched_positions()`
    /// must describe the minimum doc id — the first document of a doc-ordered
    /// walk — and `seek` must land on the same documents as an eager sort.
    #[test]
    fn lazy_doc_order_reports_minimum_doc_before_first_advance() {
        let entries: Ranked<'_> = &[
            (40, 0.9, &[(0, 0.9)]),
            (5, 0.8, &[(3, 0.8)]),
            (17, 0.7, &[(0, 0.7)]),
        ];
        let mut scorer = VectorResultScorer::new(ranked(entries), 2);
        assert_eq!(scorer.doc(), 5);
        assert_eq!(scorer.score(), 0.8);
        let positions = scorer.matched_positions().unwrap();
        assert_eq!(positions[0].0, 2);
        assert_eq!(positions[0].1[0].position, 3);
        assert_eq!(scorer.size_hint(), 3);

        assert_eq!(scorer.advance(), 17);
        assert_eq!(scorer.score(), 0.7);
        assert_eq!(scorer.seek(18), 40);
        assert_eq!(scorer.advance(), TERMINATED);
        assert_eq!(scorer.score(), 0.0);
        assert!(scorer.matched_positions().is_none());
        assert!(
            scorer.precomputed_top_k(10, true).is_none(),
            "a driven scorer must not hand over a partially consumed list"
        );

        let mut seeking = VectorResultScorer::new(ranked(entries), 2);
        assert_eq!(seeking.seek(6), 17);
        assert_eq!(seeking.seek(17), 17);
        assert_eq!(seeking.seek(41), TERMINATED);
    }
}
