//! Shared passage/context inference and document reduction.
use crate::{Error, Result};
/// One passage/context merge and document reduction for every L1 model.
pub(super) fn score_candidate_with(
    names: &[&str],
    features: &mut super::CandidateScores,
    combiner: crate::query::MultiValueCombiner,
    rrf: Option<&crate::query::RrfScore>,
    mut score: impl FnMut(&[Option<f32>], Option<f32>) -> Result<f32>,
) -> Result<f32> {
    use crate::query::MultiValueCombiner;
    combiner.validate().map_err(Error::Query)?;
    if names.len() > crate::query::MAX_FUSION_SUB_QUERIES
        || features.document.len() != names.len()
        || features.passages.len() > features.scored_passages
    {
        return Err(Error::Query("L1 candidate feature shape mismatch".into()));
    }
    if features.scored_passages == 0 {
        return score(&features.document, rrf.map(|rrf| rrf.score));
    }
    let required = match combiner {
        MultiValueCombiner::Max => 1,
        MultiValueCombiner::WeightedTopK { k, .. } => k.min(features.scored_passages),
        _ => features.scored_passages,
    };
    if features.passages.len() < required {
        return Err(Error::Query(
            "L1 document combiner needs more passage rows than were exported".into(),
        ));
    }
    let context = rrf.map_or(0.0, |rrf| rrf.document_context());
    let mut values = [None; crate::query::MAX_FUSION_SUB_QUERIES];
    let values = &mut values[..names.len()];
    let mut scores = smallvec::SmallVec::<[(u32, f32); 16]>::with_capacity(features.passages.len());
    for row in &features.passages {
        if row.values.len() != names.len() {
            return Err(Error::Query(
                "L1 passage feature shape/ordinal mismatch".into(),
            ));
        }
        scores.push((u32::from(row.ordinal), 0.0));
    }
    scores.sort_unstable_by_key(|&(ordinal, _)| ordinal);
    if scores.windows(2).any(|rows| rows[0].0 == rows[1].0) {
        return Err(Error::Query(
            "L1 passage feature shape/ordinal mismatch".into(),
        ));
    }
    scores.clear();
    for row in &mut features.passages {
        for ((value, &chunk), &document) in
            values.iter_mut().zip(&row.values).zip(&features.document)
        {
            *value = chunk.or(document);
        }
        row.score = score(
            values,
            rrf.map(|rrf| rrf.passage_score(u32::from(row.ordinal), context)),
        )?;
        scores.push((u32::from(row.ordinal), row.score));
    }
    // The order on the wire may be score order; physical/export ordering
    // must not alter strict floating-point reductions.
    scores.sort_unstable_by_key(|&(ordinal, _)| ordinal);
    let score = combiner.combine(&scores);
    if !score.is_finite() {
        return Err(Error::Query("L1 document score reduction overflow".into()));
    }
    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::super::{CandidateScores, PassageFeatures, RankingModel};
    use crate::query::MultiValueCombiner;

    #[test]
    fn passage_reduction_preserves_ordinal_order_when_inline_scratch_spills() {
        let model = RankingModel::compile("p + d", &["p", "d"], &Default::default()).unwrap();
        for count in [1, 16, 17, 65] {
            let expected: Vec<_> = (0..count)
                .map(|i| (i, [10_000_000.0, -10_000_000.0, 0.1][i as usize % 3]))
                .collect();
            for combiner in [
                MultiValueCombiner::Max,
                MultiValueCombiner::Sum,
                MultiValueCombiner::Avg,
                MultiValueCombiner::weighted_top_k(),
                MultiValueCombiner::log_sum_exp(),
            ] {
                let mut features = CandidateScores {
                    document: vec![None, Some(0.0)],
                    passages: expected
                        .iter()
                        .rev()
                        .map(|&(ordinal, value)| PassageFeatures {
                            ordinal: ordinal as u16,
                            score: 0.0,
                            values: vec![Some(value), None],
                        })
                        .collect(),
                    scored_passages: count as usize,
                };
                let result = model
                    .score_candidate(&["p", "d"], &mut features, combiner, None)
                    .unwrap();
                assert_eq!(result.to_bits(), combiner.combine(&expected).to_bits());
            }
        }
    }
}
