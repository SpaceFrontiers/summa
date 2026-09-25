//! Borrowed scores from the actual named retrieval branches. Logical identity
//! includes branch and scope; a score from another branch cannot fill this cell.
use super::*;
use crate::query::SearchResult;
use std::collections::BTreeSet;

pub(super) struct RetrievedScores<'a> {
    rows: Vec<(usize, &'a SearchResult)>,
}

impl<'a> RetrievedScores<'a> {
    pub(super) fn new(
        branches: &[(usize, &'a [SearchResult])],
        plan: &CandidateScoringPlan,
        addresses: &BTreeSet<(u128, u32)>,
    ) -> Result<Self> {
        let slots = branches
            .iter()
            .try_fold(0usize, |total, (_, hits)| total.checked_add(hits.len()))
            .filter(|&n| n <= crate::query::MAX_FUSION_CANDIDATE_SLOTS)
            .ok_or_else(|| Error::Query("retrieved L1 score budget exceeded".into()))?;
        let mut rows = Vec::with_capacity(slots);
        let mut features = BTreeSet::new();
        let mut positions = 0usize;
        for &(feature, hits) in branches {
            if feature >= plan.features.len() || !features.insert(feature) {
                return Err(Error::Query(
                    "invalid or duplicate retrieved L1 branch".into(),
                ));
            }
            for hit in hits {
                if !addresses.contains(&(hit.segment_id, hit.doc_id)) || !hit.score.is_finite() {
                    return Err(Error::Query(
                        "retrieved L1 score has an invalid address or value".into(),
                    ));
                }
                for (_, values) in &hit.positions {
                    positions = positions.saturating_add(values.len());
                    if positions > crate::query::MAX_FUSION_CHUNK_SLOTS
                        || values
                            .iter()
                            .any(|p| p.position > u16::MAX as u32 || !p.score.is_finite())
                    {
                        return Err(Error::Query(
                            "retrieved L1 passage score budget or value is invalid".into(),
                        ));
                    }
                }
                rows.push((feature, hit));
            }
        }
        rows.sort_unstable_by_key(|(feature, hit)| (*feature, hit.segment_id, hit.doc_id));
        if rows.windows(2).any(|p| {
            (p[0].0, p[0].1.segment_id, p[0].1.doc_id) == (p[1].0, p[1].1.segment_id, p[1].1.doc_id)
        }) {
            return Err(Error::Query("duplicate retrieved L1 document score".into()));
        }
        Ok(Self { rows })
    }

    pub(super) fn get(&self, feature: usize, segment: u128, doc: u32) -> Option<&SearchResult> {
        self.rows
            .binary_search_by_key(&(feature, segment, doc), |(f, hit)| {
                (*f, hit.segment_id, hit.doc_id)
            })
            .ok()
            .map(|i| self.rows[i].1)
    }
}
