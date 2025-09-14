//! Bounded text windows, local MaxScore partitioning, and canonical reduction.

use super::{
    CursorVariant, DocId, ExecutorStats, MaxScoreExecutor, ScoredDoc, SharedThreshold, TermCursor,
    WINDOW_IDS, WindowScratch, filter_competitive, with_window_scratch,
};
use log::{debug, warn};

impl MaxScoreExecutor<'_> {
    /// Probe once per common group interval. Bounds are summed in query order,
    /// so positive f32 addition preserves the per-term upper-bound invariant.
    pub(super) fn skip_noncompetitive_group(
        &mut self,
        from: DocId,
        fine_end: DocId,
        threshold: f32,
        bounds: &mut [f32],
        checked_end: &mut Option<DocId>,
    ) -> crate::Result<bool> {
        let mut first = [u32::MAX; crate::query::MAX_QUERY_TERMS];
        let mut end = u32::MAX;
        for (i, cursor) in self.cursors.iter().enumerate() {
            bounds[i] = 0.0;
            if let Some((start, last, bound)) = cursor.text_group_span_from(from) {
                first[i] = start;
                end = end.min(last);
                // Unsupported negative bounds cannot bound absent clauses.
                bounds[i] = if bound < 0.0 { f32::INFINITY } else { bound };
            }
        }
        if end == u32::MAX {
            return Ok(false);
        }
        *checked_end = Some(end);
        if end <= fine_end {
            return Ok(false);
        }
        let mut bound = 0.0f32;
        for &i in &self.score_order {
            if first[i] <= end {
                bound += bounds[i];
            }
        }
        if bound < threshold {
            for cursor in &mut self.cursors {
                if !cursor.exhausted && cursor.doc() <= end {
                    cursor.skip_past_sync(end)?;
                }
            }
            return Ok(true);
        }
        Ok(false)
    }

    /// Window-at-a-time Block-Max MaxScore for text cursors.
    ///
    /// The id space is walked in windows of at most [`WINDOW_IDS`] ids that
    /// start at the first id a globally essential cursor can still reach and
    /// end at the smallest current block end among those cursors. Per window
    /// (Lucene `MaxScoreBulkScorer`; turbopuffer "batched iterator
    /// advancement"):
    ///
    /// 1. every cursor's bound over the window is read from its skip entries
    ///    (`window_upper_bound`), and the cursors are re-partitioned into
    ///    essential and non-essential by those bounds, so the partition is
    ///    the block-max one, not the list-max one;
    /// 2. a window whose summed bounds cannot reach the threshold is skipped
    ///    by every cursor without decoding a block;
    /// 3. each essential cursor scores all of its postings in the window in
    ///    one pass over its decoded block (dense `scores[id - from]` buffer
    ///    plus a match bitset), so the same iterator advances many times in
    ///    a row instead of alternating with the others;
    /// 4. the candidates are filtered branch-free against the threshold
    ///    minus what the remaining cursors could still add, and each
    ///    non-essential cursor is then sought to the survivors in id order
    ///    (again one iterator at a time), strongest bound first.
    ///
    /// Rank-safe: only documents whose window-essential score plus the
    /// non-essential bounds cannot reach the threshold are dropped. The
    /// approximate `heap_factor` mode scales the threshold as in the
    /// document-at-a-time loop.
    pub(super) fn execute_windowed(&mut self) -> crate::Result<Vec<ScoredDoc>> {
        self.execute_text_windows::<false>()
    }

    /// Shared body of the windowed executor; `REQUIRED` selects the semantic
    /// MUST-prefix variant driven by a required cursor.
    ///
    /// Canonical reduction: cursors are visited in window-bound order, which
    /// changes from window to window, so summing scores as cursors are
    /// visited would make a document's score depend on the window it landed
    /// in. When all terms are essential, admitted nonnegative scorers visit
    /// them in query order and the accumulator is already canonical. Otherwise
    /// every cursor's contribution is retained per term
    /// (`contributions` plus the presence bits in `contribution_masks`) and a
    /// surviving candidate's score is folded in query order (`score_order`)
    /// starting from `0.0`. This keeps windowed scores bit-identical to
    /// query-order summation and to the exhaustive scorer; it is intentional
    /// and retained.
    pub(super) fn execute_text_windows<const REQUIRED: bool>(
        &mut self,
    ) -> crate::Result<Vec<ScoredDoc>> {
        with_window_scratch(|scratch| self.run_text_windows::<REQUIRED>(scratch))
    }

    fn run_text_windows<const REQUIRED: bool>(
        &mut self,
        scratch: &mut WindowScratch,
    ) -> crate::Result<Vec<ScoredDoc>> {
        debug_assert!(self.all_text(), "text windows drive text cursors only");
        if self.collector.k == 0 {
            return Ok(Vec::new());
        }
        if self
            .budget
            .as_ref()
            .is_some_and(SharedThreshold::stop_if_expired)
        {
            return Ok(Vec::new());
        }
        let n = self.cursors.len();
        let required_lead = if REQUIRED {
            (0..n)
                .filter(|&i| self.required_mask & (1u64 << i) != 0)
                .min_by_key(|&i| match &self.cursors[i].variant {
                    CursorVariant::Text { list, .. } => list.doc_count(),
                    _ => unreachable!("required windows contain text cursors"),
                })
                .expect("required windows have a semantic driver")
        } else {
            0
        };

        for cursor in &mut self.cursors {
            cursor.ensure_block_loaded_sync()?;
        }
        scratch.prepare_windows(n);
        let WindowScratch {
            window_scores,
            window_mask,
            // Single-term scores already have canonical arithmetic. Multi-term
            // windows retain contributions until surviving candidates can be
            // summed in query order, without rewinding any posting cursor.
            // Presence bits distinguish a current contribution from an old
            // value at the same window slot, without clearing the score
            // arrays (which also outlive the query in the scratch).
            contributions,
            contribution_masks,
            cand_docs,
            cand_scores,
            wmax,
            order,
            wprefix,
            ..
        } = scratch;
        // If even the largest sum omitting one term loses, every term is
        // necessary. Prepare this query-wide proof once; keep score arithmetic
        // canonical f32 and round this nonnegative bound upward.
        let conjunction_tail_bound = (!REQUIRED
            && self.document_map.is_some()
            && n > 1
            && self.inv_heap_factor == 1.0
            && self
                .cursors
                .iter()
                .all(|cursor| cursor.max_score.is_finite() && cursor.max_score >= 0.0))
        .then(|| {
            let sum = self
                .cursors
                .iter()
                .map(|cursor| f64::from(cursor.max_score))
                .sum::<f64>();
            let minimum = self
                .cursors
                .iter()
                .map(|cursor| cursor.max_score)
                .fold(f32::INFINITY, f32::min);
            ((sum - f64::from(minimum)) as f32).next_up()
        });
        // With two finite nonnegative contributions, addition is commutative.
        // The accumulated score already matches the query-order fold from +0.
        let nonnegative_scores = self.cursors.iter().all(|cursor| {
            cursor.max_score.is_finite()
                && matches!(
                    &cursor.variant,
                    CursorVariant::Text {
                        prepared_bounds: Some(_),
                        ..
                    }
                )
        });
        let retain_contributions =
            n > 1 && !(n == 2 && self.document_map.is_some() && nonnegative_scores);
        // Prefer selective candidate drivers under the same prefix-bound
        // proof. Costs are query-constant; the fixed array adds at most
        // MAX_QUERY_TERMS * 8 bytes of temporary stack, not a retained cache.
        let cost_partition = !REQUIRED && self.document_map.is_some() && nonnegative_scores;
        let mut inverse_costs = [0.0f64; crate::query::MAX_QUERY_TERMS];
        if cost_partition {
            for (cost, cursor) in inverse_costs.iter_mut().zip(&self.cursors) {
                let CursorVariant::Text { list, .. } = &cursor.variant else {
                    unreachable!("text windows contain text cursors");
                };
                *cost = 1.0 / f64::from(list.doc_count().max(1));
            }
        }
        let mask_words = WINDOW_IDS / 64;
        let group_pruning = self.cursors.iter().any(TermCursor::has_group_impacts);
        let mut checked_group_end = None;
        let mut windows = 0u64;
        let mut windows_skipped = 0u64;
        let mut groups_skipped = 0u64;
        let mut candidates = 0u64;
        let mut docs_scored = 0u64;
        let mut optional_leads = 0u64;
        let mut min_window_ids = 1u32;
        let mut evaluated_windows = 0u64;
        let mut window_candidates = 0u64;
        let started = crate::observe::WallTimer::start();

        'windows: loop {
            windows += 1;
            if windows & 0x3F == 0
                && let Some(budget) = &self.budget
                && budget.expired()
            {
                budget.mark_truncated();
                log::debug!(
                    "MaxScoreExecutor(windowed): deadline reached after {} windows, {} scored",
                    windows,
                    docs_scored
                );
                break;
            }
            let (from, to) = if REQUIRED {
                let cursor = &self.cursors[required_lead];
                if cursor.exhausted {
                    break;
                }
                (cursor.doc(), cursor.block_last_doc(cursor.block_idx))
            } else {
                let partition = self.find_partition();
                if partition >= n {
                    break;
                }
                // Window: from the first id a globally essential cursor can
                // still reach to the smallest current block end among them.
                let mut from = u32::MAX;
                let mut to = u32::MAX;
                for cursor in &self.cursors[partition..] {
                    if cursor.exhausted {
                        continue;
                    }
                    from = from.min(cursor.doc());
                    to = to.min(cursor.block_last_doc(cursor.block_idx));
                }
                // Amortize metadata/partition setup when several drivers
                // repeatedly form tiny windows. The full enlarged interval
                // still receives bounds below and fits the existing scratch.
                if self.document_map.is_some() && n - partition > 1 {
                    min_window_ids =
                        if window_candidates < evaluated_windows.saturating_mul(32 * n as u64) {
                            (min_window_ids * 2).min(WINDOW_IDS as u32)
                        } else {
                            1
                        };
                    to = to.max(from.saturating_add(min_window_ids - 1));
                }
                (from, to)
            };
            if from == u32::MAX {
                break;
            }
            let to = to.max(from).min(from.saturating_add(WINDOW_IDS as u32 - 1));
            let width = (to - from) as usize + 1;
            let words = width.div_ceil(64);

            // Block-max partition over the window.
            let heap_full = self.collector.len() >= self.collector.k;
            let threshold = if heap_full {
                self.pruning_threshold()
            } else {
                0.0
            };
            if group_pruning && heap_full && checked_group_end.is_none_or(|end| from > end) {
                // A coarse skip represents more postings than a fine window.
                // Check its deadline before doing that extra unit of work.
                if self
                    .budget
                    .as_ref()
                    .is_some_and(SharedThreshold::stop_if_expired)
                {
                    break;
                }
                if self.skip_noncompetitive_group(
                    from,
                    to,
                    threshold,
                    wmax,
                    &mut checked_group_end,
                )? {
                    groups_skipped += 1;
                    continue;
                }
            }
            for (i, bound) in wmax.iter_mut().enumerate() {
                *bound = self.cursors[i].window_upper_bound(from, to);
            }
            order.sort_unstable_by(|&a, &b| {
                if cost_partition {
                    (f64::from(wmax[a]) * inverse_costs[a])
                        .total_cmp(&(f64::from(wmax[b]) * inverse_costs[b]))
                } else {
                    wmax[a].total_cmp(&wmax[b])
                }
            });
            let mut sum = 0.0f32;
            for (rank, &i) in order.iter().enumerate() {
                sum += wmax[i];
                wprefix[rank] = sum;
            }
            let mut wpartition = if heap_full {
                wprefix.partition_point(|&s| s < threshold)
            } else {
                0
            };
            if wpartition >= n {
                // Nothing in the window can compete: every cursor jumps past it.
                for cursor in &mut self.cursors {
                    if !cursor.exhausted && cursor.doc() <= to {
                        cursor.skip_past_sync(to)?;
                    }
                }
                windows_skipped += 1;
                continue;
            }

            if REQUIRED {
                let mut lead = required_lead;
                let cost = |i: usize| match &self.cursors[i].variant {
                    CursorVariant::Text { list, .. } => list.doc_count(),
                    _ => unreachable!("required windows contain text cursors"),
                };
                // A semantic requirement always covers every matching document.
                // An optional term can drive only when its absence is strictly
                // noncompetitive under a canonical sum of conservative bounds.
                if heap_full {
                    for i in 0..n {
                        if cost(i) >= cost(lead) {
                            continue;
                        }
                        let without = self
                            .score_order
                            .iter()
                            .fold(0.0f32, |sum, &j| sum + if i == j { 0.0 } else { wmax[j] });
                        if without < threshold {
                            lead = i;
                        }
                    }
                }
                if lead != required_lead {
                    optional_leads += 1;
                }
                let position = order.iter().position(|&i| i == lead).unwrap();
                order[position..].rotate_left(1);
                let mut sum = 0.0;
                for (rank, &i) in order.iter().enumerate() {
                    sum += wmax[i];
                    wprefix[rank] = sum;
                }
                wpartition = n - 1;
            }

            // When every term is essential, canonical traversal makes the
            // accumulator exact without retaining another copy per term.
            let retain_contributions = if wpartition == 0 && nonnegative_scores {
                order.copy_from_slice(&self.score_order);
                false
            } else {
                retain_contributions
            };

            // One essential cursor already produces the exact candidate order.
            // Other partitions retain the dense union representation.
            let single_essential = n - wpartition == 1;
            cand_docs.clear();
            cand_scores.clear();
            if !single_essential {
                window_scores[..width].fill(0.0);
                window_mask[..words].fill(0);
            }
            if retain_contributions {
                for present in contribution_masks[..n * mask_words].chunks_exact_mut(mask_words) {
                    present[..words].fill(0);
                }
            }
            for &i in &order[wpartition..] {
                let cursor = &mut self.cursors[i];
                if cursor.exhausted {
                    continue;
                }
                if cursor.doc() < from {
                    cursor.seek_sync(from)?;
                }
                if cursor.exhausted || cursor.doc() > to {
                    continue;
                }
                let stored = retain_contributions.then(|| {
                    (
                        &mut contributions[i * WINDOW_IDS..i * WINDOW_IDS + width],
                        &mut contribution_masks[i * mask_words..i * mask_words + words],
                    )
                });
                if single_essential {
                    cursor.append_scored_window_sync(from, to, cand_docs, cand_scores, stored)?;
                } else {
                    cursor.score_window_sync(
                        from,
                        to,
                        &mut window_scores[..width],
                        &mut window_mask[..words],
                        stored,
                    )?;
                }
            }

            // Candidates in id order.
            if !single_essential {
                for (word_idx, word) in window_mask[..words].iter().enumerate() {
                    let mut bits = *word;
                    while bits != 0 {
                        let slot = (word_idx << 6) | bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        cand_docs.push(from + slot as u32);
                        cand_scores.push(window_scores[slot]);
                    }
                }
            }
            if let Some(pred) = &self.predicate {
                let mut kept = 0usize;
                for j in 0..cand_docs.len() {
                    let doc = cand_docs[j];
                    cand_docs[kept] = doc;
                    cand_scores[kept] = cand_scores[j];
                    kept += pred(doc) as usize;
                }
                cand_docs.truncate(kept);
                cand_scores.truncate(kept);
            }

            evaluated_windows += 1;
            window_candidates += cand_docs.len() as u64;

            // Non-essential cursors on the survivors, strongest bound first.
            let mut remaining = if wpartition > 0 {
                wprefix[wpartition - 1]
            } else {
                0.0
            };
            for rank in (0..wpartition).rev() {
                let i = order[rank];
                if heap_full {
                    filter_competitive(cand_docs, cand_scores, remaining, threshold);
                }
                if cand_docs.is_empty() {
                    break;
                }
                let required = REQUIRED && self.required_mask & (1u64 << i) != 0;
                if (required || wmax[i] > 0.0)
                    && !self.cursors[i].score_candidates_sync(
                        from,
                        cand_docs,
                        cand_scores,
                        required,
                        retain_contributions.then(|| {
                            (
                                &mut contributions[i * WINDOW_IDS..i * WINDOW_IDS + width],
                                &mut contribution_masks[i * mask_words..i * mask_words + words],
                            )
                        }),
                        self.budget.as_ref(),
                    )?
                {
                    break 'windows;
                }
                remaining -= wmax[i];
            }
            if heap_full {
                filter_competitive(cand_docs, cand_scores, 0.0, threshold);
            }
            if REQUIRED {
                let cursor = &mut self.cursors[required_lead];
                if !cursor.exhausted && cursor.doc() <= to {
                    cursor.skip_past_sync(to)?;
                }
            }
            candidates += cand_docs.len() as u64;
            for (doc, score) in cand_docs.iter().zip(cand_scores.iter()) {
                let score = if retain_contributions {
                    // Canonical query-order reduction (see the doc comment).
                    let slot = (*doc - from) as usize;
                    self.score_order.iter().fold(0.0f32, |total, &i| {
                        let value = contributions[i * WINDOW_IDS + slot];
                        let present = contribution_masks[i * mask_words + (slot >> 6)]
                            & (1u64 << (slot & 63))
                            != 0;
                        total + if present { value } else { 0.0 }
                    })
                } else {
                    *score
                };
                if self
                    .collector
                    .insert_with_ordinal(self.result_doc(*doc), score, 0)
                {
                    docs_scored += 1;
                }
            }
            // The previous window's threshold is conservative after collecting
            // this window. Once it proves every term necessary, reuse the typed
            // intersection and existing heap for the remaining stream.
            if heap_full && conjunction_tail_bound.is_some_and(|bound| bound < threshold) {
                for cursor in &mut self.cursors {
                    cursor.skip_past_sync(to)?;
                }
                self.stats = ExecutorStats {
                    windows,
                    windows_skipped,
                    groups_skipped,
                    candidates,
                    docs_scored,
                    optional_leads,
                    ..ExecutorStats::default()
                };
                return self.run_ranked_conjunction(scratch);
            }
        }

        let results = self.finish();
        self.stats = ExecutorStats {
            windows,
            windows_skipped,
            groups_skipped,
            candidates,
            docs_scored,
            optional_leads,
            ..ExecutorStats::default()
        };
        let stats = self.stats;
        let elapsed_ms = (started.secs() * 1000.0) as u64;
        if elapsed_ms > 500 {
            warn!(
                "slow windowed MaxScore: {}ms, cursors={}, windows={}, windows_skipped={}, groups_skipped={}, optional_leads={}, candidates={}, scored={}, returned={}, top_score={:.4}",
                elapsed_ms,
                n,
                stats.windows,
                stats.windows_skipped,
                stats.groups_skipped,
                stats.optional_leads,
                stats.candidates,
                stats.docs_scored,
                results.len(),
                results.first().map(|r| r.score).unwrap_or(0.0)
            );
        } else {
            debug!(
                "MaxScoreExecutor(windowed): {}ms, cursors={}, windows={}, windows_skipped={}, groups_skipped={}, optional_leads={}, candidates={}, scored={}, returned={}, top_score={:.4}",
                elapsed_ms,
                n,
                stats.windows,
                stats.windows_skipped,
                stats.groups_skipped,
                stats.optional_leads,
                stats.candidates,
                stats.docs_scored,
                results.len(),
                results.first().map(|r| r.score).unwrap_or(0.0)
            );
        }
        Ok(results)
    }
}
