//! Sequential-dependence proximity rescoring for BM25 text queries.
//!
//! A second stage over the top candidates of a text MaxScore pass: for every
//! pair of adjacent query terms the document's positions are scanned for
//! ordered windows (the second term directly after the first) and unordered
//! windows (both within `window` positions). Each count is saturated with the
//! field's BM25 parameters and length, weighted by the pair's mean idf, and
//! added to the BM25 score scaled by `weight` (Metzler & Croft, "A Markov
//! random field model for term dependencies", SIGIR 2005; ordered windows
//! count full weight, unordered windows half).
//!
//! The stage is approximate on purpose: the MaxScore pass over-fetches
//! `PROXIMITY_OVER_FETCH` times the requested limit, the bonus is added to
//! those candidates only, and cross-segment threshold seeding is disabled
//! for the pass because the bonus lifts scores above the BM25 floor.

#[cfg(feature = "sync")]
use crate::DocId;
use crate::dsl::Field;
use crate::segment::SegmentReader;

/// Proximity rescoring of a text query (`MatchQuery.proximity_weight`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProximityConfig {
    /// Multiplier of the window bonus; 0 disables the stage.
    pub weight: f32,
    /// Maximum distance of an unordered window.
    pub window: u32,
}

impl ProximityConfig {
    /// Default unordered window (Metzler & Croft used 8).
    pub const DEFAULT_WINDOW: u32 = 8;

    pub fn new(weight: f32, window: u32) -> Self {
        Self {
            weight,
            window: if window == 0 {
                Self::DEFAULT_WINDOW
            } else {
                window
            },
        }
    }

    pub fn is_active(&self) -> bool {
        self.weight > 0.0
    }
}

/// Candidates fetched per requested hit before rescoring.
pub(crate) const PROXIMITY_OVER_FETCH: usize = 4;

/// Ordered (`b` right after `a`) and unordered (within `window`) windows
/// between two ascending position lists.
#[cfg_attr(not(feature = "sync"), allow(dead_code))]
pub(crate) fn count_windows(a: &[u32], b: &[u32], window: u32) -> (u32, u32) {
    let mut ordered = 0u32;
    let mut unordered = 0u32;
    let (mut lo, mut hi, mut adj) = (0usize, 0usize, 0usize);
    for &pa in a {
        let low = pa.saturating_sub(window);
        let high = pa.saturating_add(window);
        while lo < b.len() && b[lo] < low {
            lo += 1;
        }
        if hi < lo {
            hi = lo;
        }
        while hi < b.len() && b[hi] <= high {
            hi += 1;
        }
        unordered += (hi - lo) as u32;
        while adj < b.len() && b[adj] < pa + 1 {
            adj += 1;
        }
        if adj < b.len() && b[adj] == pa + 1 {
            ordered += 1;
        }
    }
    (ordered, unordered)
}

/// Add the proximity bonus to `hits` (documents or chunk ids of `field`).
/// `terms` are the query terms in query order with their idf.
#[cfg(feature = "sync")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rescore_sync(
    reader: &SegmentReader,
    field: Field,
    terms: &[(Vec<u8>, f32)],
    params: super::Bm25Params,
    lengths: Option<super::LengthSource<'_>>,
    avg_len: f32,
    config: ProximityConfig,
    hits: &mut [super::ScoredDoc],
) -> crate::Result<()> {
    use crate::structures::TERMINATED;

    if !config.is_active() || terms.len() < 2 || hits.is_empty() {
        return Ok(());
    }
    let mut cursors = Vec::with_capacity(terms.len());
    for (term, _) in terms {
        let list = reader.get_postings_sync(field, term)?;
        let positions = reader.get_positions_sync(field, term)?;
        cursors.push(match (list, positions) {
            (Some(list), Some(positions)) => Some((list.into_iterator(), positions)),
            _ => None,
        });
    }
    let mut order: Vec<usize> = (0..hits.len()).collect();
    order.sort_unstable_by_key(|&i| hits[i].doc_id);
    let mut scratch: Vec<u32> = Vec::new();
    let mut bufs: Vec<Vec<u32>> = vec![Vec::new(); terms.len()];
    let avg = avg_len.max(1.0);
    for i in order {
        let doc: DocId = hits[i].doc_id;
        for (t, cursor) in cursors.iter_mut().enumerate() {
            bufs[t].clear();
            if let Some((it, positions)) = cursor
                && it.doc() != TERMINATED
                && it.seek(doc) == doc
            {
                positions.positions_into(
                    doc,
                    it.position_cursor(),
                    it.term_freq(),
                    &mut scratch,
                    &mut bufs[t],
                );
            }
        }
        let len = lengths
            .map(|source| source.length(doc) as f32)
            .filter(|len| *len > 0.0)
            .unwrap_or(avg);
        let mut bonus = 0.0f32;
        for pair in 0..terms.len() - 1 {
            let (a, b) = (&bufs[pair], &bufs[pair + 1]);
            if a.is_empty() || b.is_empty() {
                continue;
            }
            let (ordered, unordered) = count_windows(a, b, config.window);
            let idf = (terms[pair].1 + terms[pair + 1].1) * 0.5;
            if ordered > 0 {
                bonus += params.score(ordered as f32, idf, len, avg);
            }
            if unordered > 0 {
                bonus += 0.5 * params.score(unordered as f32, idf, len, avg);
            }
        }
        hits[i].score += config.weight * bonus;
    }
    Ok(())
}

/// Without synchronous file handles the stage cannot read positions per
/// candidate; the BM25 ranking is returned unchanged.
#[cfg(not(feature = "sync"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rescore_sync(
    _reader: &SegmentReader,
    _field: Field,
    _terms: &[(Vec<u8>, f32)],
    _params: super::Bm25Params,
    _lengths: Option<super::LengthSource<'_>>,
    _avg_len: f32,
    _config: ProximityConfig,
    _hits: &mut [super::ScoredDoc],
) -> crate::Result<()> {
    log::debug!("proximity rescoring needs the `sync` feature; returning BM25 order");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_count_ordered_and_unordered_pairs() {
        // a at 0, 10; b at 1, 3, 12: a@0 sees b@1 (ordered) and b@3 within 8,
        // a@10 sees b@3 and b@12 within 8 (b@12 is not adjacent).
        assert_eq!(count_windows(&[0, 10], &[1, 3, 12], 8), (1, 4));
        assert_eq!(count_windows(&[0, 10], &[1, 3, 12], 1), (1, 1));
        assert_eq!(count_windows(&[5], &[4], 2), (0, 1));
        assert_eq!(count_windows(&[5], &[], 2), (0, 0));
        // Window 0 counts co-located positions only; ordered pairs are the
        // three adjacent ones.
        assert_eq!(count_windows(&[0, 1, 2], &[1, 2, 3], 0), (3, 2));
    }

    #[test]
    fn config_defaults_window() {
        assert_eq!(ProximityConfig::new(1.0, 0).window, 8);
        assert!(!ProximityConfig::new(0.0, 4).is_active());
    }
}
