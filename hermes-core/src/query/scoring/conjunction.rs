//! Typed text intersection and ranked/counting admission.

use super::{
    CursorVariant, DocId, MaxScoreExecutor, ScoredDoc, SharedThreshold, WindowScratch,
    score_text_run, with_window_scratch,
};
use log::debug;

impl MaxScoreExecutor<'_> {
    /// Pruning is useful only where both lists carry selective block bounds.
    /// Unsupported scorers and approximate heap factors retain exhaustive AND.
    pub(super) fn can_prune_text_pair(&self) -> bool {
        self.document_map.is_some()
            && self.inv_heap_factor == 1.0
            && self.cursors.len() == 2
            && self.cursors.iter().all(|cursor| {
                cursor.max_score.is_finite()
                    && matches!(&cursor.variant,
                        CursorVariant::Text { list, length_bounds: true, prepared_bounds: Some(_), .. }
                        if list.has_ratio_bounds())
            })
    }

    /// Semantic AND aligns typed cursors and batches only actual matches.
    /// Ranked mapped pairs can skip strictly losing blocks; counted traversal
    /// always uses the exhaustive instantiation below.
    pub(super) fn execute_conjunction(&mut self) -> crate::Result<Vec<ScoredDoc>> {
        with_window_scratch(|scratch| self.run_ranked_conjunction(scratch))
    }

    /// Share ranked admission with a union tail that has proved every term
    /// necessary. The returned work count is deliberately discarded here;
    /// exact counted traversal always selects the exhaustive implementation.
    pub(super) fn run_ranked_conjunction(
        &mut self,
        scratch: &mut WindowScratch,
    ) -> crate::Result<Vec<ScoredDoc>> {
        if self.can_prune_text_pair() {
            self.run_conjunction::<true>(scratch)
        } else {
            self.run_conjunction::<false>(scratch)
        }
        .map(|(hits, _)| hits)
    }

    pub(in crate::query) fn execute_counted_conjunction(
        mut self,
    ) -> crate::Result<(Vec<ScoredDoc>, u64)> {
        if !self.all_required {
            return Err(crate::Error::Query(
                "Exact counted traversal requires a text conjunction".into(),
            ));
        }
        let Some(timer) = self.begin()? else {
            return Ok((Vec::new(), 0));
        };
        let result = with_window_scratch(|scratch| self.run_conjunction::<false>(scratch));
        if let Ok((ref hits, _)) = result {
            crate::observe::maxscore_query(
                self.metric_index,
                self.metric_field,
                timer.secs(),
                hits.len(),
            );
        }
        result
    }

    pub(super) fn run_conjunction<const PRUNE: bool>(
        &mut self,
        scratch: &mut WindowScratch,
    ) -> crate::Result<(Vec<ScoredDoc>, u64)> {
        const BATCH: usize = crate::structures::postings::POSTING_BLOCK_SIZE;
        if self.collector.k == 0
            || self
                .budget
                .as_ref()
                .is_some_and(SharedThreshold::stop_if_expired)
        {
            return Ok((self.finish(), 0));
        }
        let n = self.cursors.len();
        debug_assert!(!PRUNE || n == 2);
        let mut order: [usize; crate::query::MAX_QUERY_TERMS] = std::array::from_fn(|i| i);
        order[..n].sort_unstable_by_key(|&i| match &self.cursors[i].variant {
            CursorVariant::Text { list, .. } => list.doc_count(),
            _ => unreachable!("semantic conjunctions contain text cursors"),
        });
        let lead = order[0];
        for cursor in &mut self.cursors {
            if !cursor.ensure_block_loaded_sync()? {
                return Ok((self.finish(), 0));
            }
        }
        let mut docs = [0; BATCH];
        scratch.prepare_conjunction(n);
        let tfs = &mut scratch.conjunction_tfs;
        let mut iterations = 0u64;
        let mut matched = 0u64;
        let started = crate::observe::WallTimer::start();
        loop {
            // Later clauses may already prove that an entire future pair
            // prefix cannot match. Preserve that skip before forming a batch.
            if let Some(next) = order[2..n].iter().map(|&i| self.cursors[i].doc()).max() {
                if next == u32::MAX {
                    break;
                }
                self.cursors[lead].seek_sync(next)?;
            }
            let threshold = if PRUNE {
                self.pruning_threshold()
            } else {
                f32::NEG_INFINITY
            };
            let pair_count = self.fill_conjunction_pair::<PRUNE>(
                &mut docs,
                tfs,
                &mut iterations,
                [lead, order[1]],
                threshold,
            )?;
            let mut count = pair_count;
            if n > 2 {
                let mut origins: [u8; BATCH] = std::array::from_fn(|i| i as u8);
                for &i in &order[2..n] {
                    let cursor = &mut self.cursors[i];
                    let mut kept = 0;
                    for row in 0..count {
                        if row.is_multiple_of(64)
                            && self
                                .budget
                                .as_ref()
                                .is_some_and(SharedThreshold::stop_if_expired)
                        {
                            return Ok((self.finish(), matched));
                        }
                        if cursor.seek_sync(docs[row])? == docs[row] {
                            cursor.decode_deferred_tfs();
                            tfs[i * BATCH + origins[row] as usize] = cursor.tfs[cursor.pos];
                            docs[kept] = docs[row];
                            origins[kept] = origins[row];
                            kept += 1;
                        }
                    }
                    count = kept;
                    if count == 0 {
                        break;
                    }
                }
                for i in 0..n {
                    for (row, &origin) in origins[..count].iter().enumerate() {
                        tfs[i * BATCH + row] = tfs[i * BATCH + origin as usize];
                    }
                }
            }
            matched += count as u64;
            if count > 0 && !self.collect_conjunction_batch(&docs[..count], tfs) {
                break;
            }
            if pair_count < BATCH {
                break;
            }
        }
        let results = self.finish();
        crate::observe::search_work!(conjunction_candidates += matched);
        debug!(
            "MaxScoreExecutor(conjunction): {}ms, cursors={}, iterations={}, matched={}, returned={}",
            (started.secs() * 1000.0) as u64,
            n,
            iterations,
            matched,
            results.len()
        );
        Ok((results, matched))
    }

    /// Intersect loaded blocks before returning a bounded batch to the shared
    /// scorer. Only block boundaries use the general cursor protocol.
    fn fill_conjunction_pair<const PRUNE: bool>(
        &mut self,
        docs: &mut [DocId],
        tfs: &mut [u32],
        iterations: &mut u64,
        pair: [usize; 2],
        threshold: f32,
    ) -> crate::Result<usize> {
        let [left, right] = self
            .cursors
            .get_disjoint_mut(pair)
            .expect("distinct conjunction cursors");
        let seek_driven = match (&left.variant, &right.variant) {
            (CursorVariant::Text { list: left, .. }, CursorVariant::Text { list: right, .. }) => {
                // Below one rare posting per common block, seeking can avoid
                // scanning most of that block. Denser intersections amortize
                // the SIMD merge over several matches instead.
                left.doc_count()
                    .saturating_mul(crate::structures::postings::POSTING_BLOCK_SIZE as u32)
                    < right.doc_count()
            }
            _ => unreachable!("conjunctions contain text cursors ordered by cost"),
        };
        let mut pairs = [(0u8, 0u8); crate::structures::postings::POSTING_BLOCK_SIZE];
        let mut count = 0;
        while count < docs.len() && !left.exhausted && !right.exhausted {
            if self
                .budget
                .as_ref()
                .is_some_and(SharedThreshold::stop_if_expired)
            {
                break;
            }
            if left.block_last_doc(left.block_idx) < right.doc() {
                left.seek_sync(right.doc())?;
            }
            if right.block_last_doc(right.block_idx) < left.doc() {
                right.seek_sync(left.doc())?;
            }
            if PRUNE && (left.exhausted || right.exhausted) {
                break;
            }
            if PRUNE
                && left.text_block_bound(left.block_idx) + right.text_block_bound(right.block_idx)
                    < threshold
            {
                // Every possible intersection in the shorter block loses.
                // Advancing it alone preserves progress without decoding TFs.
                let left_last = left.block_last_doc(left.block_idx);
                let right_last = right.block_last_doc(right.block_idx);
                if left_last <= right_last {
                    left.skip_to_next_block();
                }
                if right_last <= left_last {
                    right.skip_to_next_block();
                }
                continue;
            }
            if !left.ensure_block_loaded_sync()? || !right.ensure_block_loaded_sync()? {
                break;
            }
            let mut a = left.pos;
            let mut b = right.pos;
            // RGB can cluster a globally rare term into dense local runs.
            // Compare the decoded spans too, retaining the SIMD merge there.
            let sparse_block = seek_driven
                && left.doc_ids.last().unwrap() - left.doc_ids[0]
                    > (right.doc_ids.last().unwrap() - right.doc_ids[0])
                        .saturating_mul(crate::structures::postings::POSTING_BLOCK_SIZE as u32);
            let found = if sparse_block {
                // The rarer cursor supplies candidate IDs. Seeking avoids a
                // linear walk across the common term's decoded block; the
                // same bounded batch below owns frequencies and scoring.
                let target = left.doc();
                let other = right.seek_sync(target)?;
                if other != target {
                    left.seek_sync(other)?;
                    continue;
                }
                a = left.pos + 1;
                b = right.pos + 1;
                pairs[0] = (left.pos as u8, right.pos as u8);
                1
            } else {
                crate::structures::simd::intersect_posting_blocks(
                    &left.doc_ids,
                    &mut a,
                    &right.doc_ids,
                    &mut b,
                    &mut pairs[..docs.len() - count],
                )
            };
            *iterations += (a - left.pos + b - right.pos) as u64;
            let mut frequencies_loaded = false;
            for &(a, b) in &pairs[..found] {
                let doc = left.doc_ids[a as usize];
                if self
                    .predicate
                    .as_ref()
                    .is_none_or(|predicate| predicate(doc))
                {
                    if !frequencies_loaded {
                        left.decode_deferred_tfs();
                        right.decode_deferred_tfs();
                        frequencies_loaded = true;
                    }
                    docs[count] = doc;
                    tfs[pair[0] * docs.len() + count] = left.tfs[a as usize];
                    tfs[pair[1] * docs.len() + count] = right.tfs[b as usize];
                    count += 1;
                }
            }
            for (cursor, pos) in [(&mut *left, a), (&mut *right, b)] {
                if pos == cursor.doc_ids.len() {
                    cursor.pos = pos - 1;
                    cursor.advance_pos();
                } else {
                    cursor.pos = pos;
                }
            }
        }
        Ok(count)
    }

    /// Rows retain cursor identity; canonical reduction follows query order.
    fn collect_conjunction_batch(&mut self, docs: &[DocId], tfs: &[u32]) -> bool {
        const BATCH: usize = crate::structures::postings::POSTING_BLOCK_SIZE;
        let mut totals = [0.0; BATCH];
        let mut values = [0.0; BATCH];
        for &i in &self.score_order {
            if self
                .budget
                .as_ref()
                .is_some_and(SharedThreshold::stop_if_expired)
            {
                return false;
            }
            let CursorVariant::Text {
                idf,
                avg_len,
                params,
                lengths,
                normalization,
                ..
            } = &self.cursors[i].variant
            else {
                unreachable!("semantic conjunctions contain text cursors");
            };
            score_text_run(
                *params,
                *idf,
                *avg_len,
                *lengths,
                normalization.as_deref(),
                docs,
                &tfs[i * BATCH..i * BATCH + docs.len()],
                &mut values[..docs.len()],
            );
            for (total, &value) in totals.iter_mut().zip(&values[..docs.len()]) {
                *total += value;
            }
        }
        if let Some(map) = self.document_map {
            self.collector
                .insert_text_run_with_mapping(docs, &totals[..docs.len()], |doc| map.doc_id(doc));
        } else {
            self.collector.insert_text_run(docs, &totals[..docs.len()]);
        }
        true
    }
}
