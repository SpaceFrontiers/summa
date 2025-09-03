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
    if features.document.len() != names.len() || features.passages.len() > features.scored_passages
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
    let mut values = vec![None; names.len()];
    let mut scores = Vec::with_capacity(features.passages.len());
    let mut ordinals = std::collections::BTreeSet::new();
    for row in &mut features.passages {
        if row.values.len() != names.len() || !ordinals.insert(row.ordinal) {
            return Err(Error::Query(
                "L1 passage feature shape/ordinal mismatch".into(),
            ));
        }
        for ((value, &chunk), &document) in
            values.iter_mut().zip(&row.values).zip(&features.document)
        {
            *value = chunk.or(document);
        }
        row.score = score(
            &values,
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
