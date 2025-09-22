//! Bounded score-floor proofs for expensive mapped phrase traversals.

use super::PhraseScorer;
use crate::query::{Collector, Scorer, TopKCollector, docset::DocSet};
use crate::structures::TERMINATED;
use crate::{DocId, Score};

impl PhraseScorer {
    pub(super) fn seed_score(&mut self, limit: usize) -> Option<Score> {
        if !(1..=128).contains(&limit) || self.current_doc == TERMINATED {
            return None;
        }
        // Small heaps benefit from wider coverage with few position checks;
        // larger heaps need enough confirmed matches to prove any floor.
        let selective = limit <= 16;
        let block_limit = if selective { 64 } else { 8 };
        self.prepared_bounds.as_ref()?;
        let original = self.current_doc;
        let (list, current) = self.posting_iters[self.bound_term].current_block_metadata()?;
        let mut blocks: Vec<(Score, DocId, DocId)> = Vec::with_capacity(block_limit);
        let stride = (list.num_blocks() - current - 1).div_ceil(4096).max(1);
        for (visited, block) in (current + 1..list.num_blocks()).step_by(stride).enumerate() {
            crate::observe::search_work!(phrase_seed_metadata_blocks += 1);
            if visited.is_multiple_of(64)
                && self
                    .budget
                    .as_ref()
                    .is_some_and(crate::query::SharedThreshold::stop_if_expired)
            {
                return None;
            }
            // Short-document bounds only nominate blocks. They do not prove
            // a score floor: only the real matches in `heap` below can do that.
            let score = -(list.block_bounds(block)?.1.unwrap_or(1) as Score);
            let at = blocks.partition_point(|&(bound, _, _)| bound >= score);
            if at < block_limit {
                if blocks.len() == block_limit {
                    blocks.pop();
                }
                blocks.insert(
                    at,
                    (
                        score,
                        list.block_first_doc(block)?,
                        list.block_last_doc(block)?,
                    ),
                );
            }
        }
        let mut heap = TopKCollector::new(limit);
        let mut candidates: Vec<(Score, DocId)> = Vec::with_capacity(if selective { 8 } else { 0 });
        for (_, first, last) in blocks {
            if self
                .budget
                .as_ref()
                .is_some_and(crate::query::SharedThreshold::stop_if_expired)
            {
                self.restore_candidate(original);
                return None;
            }
            candidates.clear();
            self.rewind_postings(first);
            let doc = self.find_next_and_match_through::<true>(last);
            self.park(doc, None);
            while self.doc() <= last {
                if self
                    .budget
                    .as_ref()
                    .is_some_and(crate::query::SharedThreshold::stop_if_expired)
                {
                    self.restore_candidate(original);
                    return None;
                }
                if selective {
                    let bound = self.candidate_score_upper_bound();
                    let at = candidates.partition_point(|&(score, _)| score >= bound);
                    if at < 8 {
                        if candidates.len() == 8 {
                            candidates.pop();
                        }
                        candidates.insert(at, (bound, self.doc()));
                    }
                } else {
                    crate::observe::search_work!(phrase_seed_candidates += 1);
                    if self.confirm_candidate() {
                        heap.collect(self.doc(), self.score(), &[]);
                    }
                }
                self.posting_iters[self.lead].advance();
                let doc = self.find_next_and_match_through::<true>(last);
                self.park(doc, None);
            }
            candidates.sort_unstable_by_key(|&(_, doc)| doc);
            for &(_, target) in &candidates {
                if self
                    .budget
                    .as_ref()
                    .is_some_and(crate::query::SharedThreshold::stop_if_expired)
                {
                    self.restore_candidate(original);
                    return None;
                }
                crate::observe::search_work!(phrase_seed_candidates += 1);
                self.restore_candidate(target);
                if self.doc() == target && self.confirm_candidate() {
                    heap.collect(target, self.score(), &[]);
                }
            }
        }
        // These distinct matches prove a score, but emit no results. Ordinary
        // traversal must still collect them and resolve logical-ID ties.
        self.restore_candidate(original);
        if !heap.is_full() {
            return None;
        }
        heap.into_sorted_results().last().map(|hit| hit.score)
    }

    fn restore_candidate(&mut self, target: DocId) {
        self.rewind_postings(target);
        self.find_next_candidate();
    }

    fn rewind_postings(&mut self, target: DocId) {
        self.intersection.reset();
        for cursor in &mut self.posting_iters {
            cursor.seek_physical(target);
        }
        self.park(TERMINATED, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directories::OwnedBytes;
    use crate::query::phrase::Lengths;
    use crate::segment::chunk_map::DocLengths;
    use crate::structures::{
        BlockPostingList, PositionStreamEncoder, PostingCodec, PostingList, TermPositions,
    };

    #[test]
    fn phrase_score_pilot_proves_a_floor_and_restores_position_cursors() {
        for (starts, follows, slop, frequency) in [
            (&[0][..], &[1][..], 0, 1.0),
            (&[0, 0][..], &[1][..], 0, 2.0),
            (&[0, 0][..], &[2][..], 2, 2.0),
        ] {
            for codec in [
                PostingCodec::Rounded,
                PostingCodec::Packed,
                PostingCodec::Pfor,
                PostingCodec::Simd4x,
            ] {
                let lengths: Vec<u16> = (0..2181)
                    .map(|doc| if doc < 2048 { 100 } else { 3 })
                    .collect();
                let mut lists = Vec::new();
                let mut positions = Vec::new();
                for values in [starts, follows] {
                    let mut values = values.to_vec();
                    let mut list = PostingList::new();
                    let mut bytes = Vec::new();
                    let mut encoder = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
                    for doc in 0..2181 {
                        list.push(doc, values.len() as u32);
                        encoder.push_doc(&mut values).unwrap();
                    }
                    encoder.finish().unwrap();
                    lists.push(
                        BlockPostingList::from_posting_list_with_options(
                            &list,
                            true,
                            Some(&|doc| u32::from(lengths[doc as usize])),
                            codec,
                        )
                        .unwrap(),
                    );
                    positions.push(TermPositions::open(OwnedBytes::new(bytes)).unwrap());
                }
                let mut scorer = PhraseScorer::unpositioned(
                    lists,
                    positions,
                    &[0, 1],
                    slop,
                    1.2345,
                    100.0,
                    None,
                )
                .with_lengths(Lengths::Docs(DocLengths::from_lengths(&lengths)));
                scorer.find_next_candidate();
                assert!(scorer.confirm_candidate());
                let first_score = scorer.score();
                for k in [0, 1, 10, 100, 128, 129] {
                    let seeded = scorer.seed_ranked_score(k);
                    let best = scorer.params.score(frequency, 1.2345, 3.0, 100.0);
                    match k {
                        1 | 10 => assert_eq!(seeded, Some(best)),
                        100 | 128 => assert!(seeded.is_some_and(|floor| {
                            floor.to_bits() == best.to_bits()
                                || floor.to_bits() == first_score.to_bits()
                        })),
                        _ => assert_eq!(seeded, None),
                    }
                    assert_eq!(scorer.doc(), 0);
                    assert!(scorer.confirm_candidate());
                    assert_eq!(scorer.score().to_bits(), first_score.to_bits());
                }
                for doc in 1..2181 {
                    scorer.advance_candidate();
                    assert_eq!(scorer.doc(), doc);
                    assert!(scorer.confirm_candidate());
                    assert_eq!(
                        scorer.score().to_bits(),
                        scorer
                            .params
                            .score(frequency, 1.2345, f32::from(lengths[doc as usize]), 100.0)
                            .to_bits()
                    );
                }
                assert_eq!(scorer.seed_ranked_score(10), None);
            }
        }
    }

    #[test]
    fn phrase_score_pilot_requires_distinct_matches_and_observes_cancellation() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let mut lists = Vec::new();
            let mut positions = Vec::new();
            for term in 0..2 {
                let mut list = PostingList::new();
                let mut bytes = Vec::new();
                let mut encoder = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
                for doc in 0..153 {
                    list.push(doc, 1);
                    let position = if term == 0 {
                        0
                    } else if [0, 128, 129].contains(&doc) {
                        1
                    } else {
                        2
                    };
                    encoder.push_doc(&mut [position]).unwrap();
                }
                encoder.finish().unwrap();
                lists.push(
                    BlockPostingList::from_posting_list_with_options(
                        &list,
                        true,
                        Some(&|_| 3),
                        codec,
                    )
                    .unwrap(),
                );
                positions.push(TermPositions::open(OwnedBytes::new(bytes)).unwrap());
            }
            let mut scorer =
                PhraseScorer::unpositioned(lists, positions, &[0, 1], 0, 1.2345, 100.0, None)
                    .with_lengths(Lengths::Docs(DocLengths::from_lengths(&[3; 153])));
            scorer.find_next_candidate();
            assert!(scorer.confirm_candidate());
            let expected = scorer.score();
            for k in [1, 2, 3, 100] {
                assert_eq!(scorer.seed_ranked_score(k), (k <= 2).then_some(expected));
                assert_eq!(scorer.doc(), 0);
                assert!(scorer.confirm_candidate());
                assert_eq!(scorer.score(), expected);
            }
            let budget = crate::query::SharedThreshold::for_limit(10)
                .with_deadline(Some(std::time::Instant::now()));
            scorer.budget = Some(budget.clone());
            assert_eq!(scorer.seed_ranked_score(1), None);
            assert!(budget.truncated());
            assert_eq!(scorer.doc(), 0);
            scorer.budget = None;
            assert_eq!(scorer.advance(), 128);
            assert_eq!(scorer.advance(), 129);
            assert_eq!(scorer.advance(), TERMINATED);
        }
    }
}
