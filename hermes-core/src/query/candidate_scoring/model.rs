//! Portable request coefficients; query names are the feature schema.
use crate::{Error, Result};
use std::collections::BTreeMap;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureTransform {
    #[serde(default)]
    pub signed_log1p: bool,
    #[serde(default = "one")]
    pub scale: f64,
    #[serde(default)]
    pub offset: f64,
}
fn one() -> f64 {
    1.0
}
impl Default for FeatureTransform {
    fn default() -> Self {
        Self {
            signed_log1p: false,
            scale: 1.0,
            offset: 0.0,
        }
    }
}
impl FeatureTransform {
    fn apply(&self, value: f32) -> f64 {
        let raw = f64::from(value);
        let raw = if self.signed_log1p {
            raw.signum() * raw.abs().ln_1p()
        } else {
            raw
        };
        raw * self.scale + self.offset
    }
}

/// Linear coefficients reference names on the actual nomination/scoring queries.
/// Missing coefficients mean zero, never a default weight. No page/shard-derived
/// normalization is applied. A missing value contributes zero before transforms.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearModel {
    pub weights: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub bias: f64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub transforms: BTreeMap<String, FeatureTransform>,
}
fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

impl LinearModel {
    /// Shared shard/coordinator inference. A shard may export only its best
    /// passage rows for MAX or weighted-top-k; AVG/SUM/softmax require every
    /// scored row. Document context is broadcast once per passage.
    pub fn score_candidate(
        &self,
        names: &[&str],
        features: &mut super::CandidateScores,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<f32> {
        use crate::query::MultiValueCombiner;
        combiner.validate().map_err(Error::Query)?;
        if features.document.len() != names.len()
            || features.passages.len() > features.scored_passages
        {
            return Err(Error::Query("L1 candidate feature shape mismatch".into()));
        }
        if features.scored_passages == 0 {
            return self.score(names, &features.document);
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
            row.score = self.score(names, &values)?;
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

    pub fn validate(&self, names: &[&str]) -> Result<()> {
        if !self.bias.is_finite()
            || self.weights.is_empty()
            || self.weights.values().all(|&w| w == 0.0)
        {
            return Err(Error::Query(
                "l1 requires finite bias and at least one nonzero weight".into(),
            ));
        }
        for (name, weight) in &self.weights {
            if !names.contains(&name.as_str()) {
                return Err(Error::Query(format!(
                    "l1 weight references unknown query branch '{name}'"
                )));
            }
            if !weight.is_finite() {
                return Err(Error::Query(format!("l1 weight '{name}' must be finite")));
            }
        }
        for (name, transform) in &self.transforms {
            if !self.weights.contains_key(name) {
                return Err(Error::Query(format!(
                    "l1 transform '{name}' has no coefficient"
                )));
            }
            if !transform.scale.is_finite()
                || transform.scale <= 0.0
                || !transform.offset.is_finite()
            {
                return Err(Error::Query(format!(
                    "l1 transform '{name}' needs a finite positive scale and finite offset"
                )));
            }
        }
        Ok(())
    }

    /// Strict IEEE/f64 reduction in declared query order, shared with artifact
    /// inference. The caller validates name/value lengths and the model once.
    pub fn score(&self, names: &[&str], values: &[Option<f32>]) -> Result<f32> {
        if names.len() != values.len() {
            return Err(Error::Query("l1 feature/name count mismatch".into()));
        }
        let mut score = self.bias;
        for (&name, &value) in names.iter().zip(values) {
            let Some(value) = value else { continue };
            if !value.is_finite() {
                return Err(Error::Query(format!(
                    "non-finite score for branch '{name}'"
                )));
            }
            let weight = self.weights.get(name).copied().unwrap_or(0.0);
            if weight == 0.0 {
                continue;
            }
            let transformed = self
                .transforms
                .get(name)
                .map_or(f64::from(value), |t| t.apply(value));
            score += weight * transformed;
        }
        let score = score as f32;
        if !score.is_finite() {
            return Err(Error::Query("l1 formula overflowed a finite score".into()));
        }
        Ok(score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_branches_and_empty_models_fail_instead_of_falling_back_to_rrf() {
        let mut model = LinearModel::default();
        assert!(model.validate(&["dense"]).is_err());
        model.weights.insert("typo".into(), 1.0);
        assert!(model.validate(&["dense"]).is_err());
        model.weights.clear();
        model.weights.insert("dense".into(), 0.0);
        assert!(model.validate(&["dense"]).is_err());
    }
    #[test]
    fn missing_values_skip_affine_offsets_while_real_zeros_and_negatives_are_valid() {
        let model = LinearModel {
            weights: BTreeMap::from([("dense".into(), 2.0)]),
            bias: 0.5,
            transforms: BTreeMap::from([(
                "dense".into(),
                FeatureTransform {
                    offset: 1.0,
                    ..Default::default()
                },
            )]),
        };
        model.validate(&["dense", "bm25"]).unwrap();
        assert_eq!(
            model
                .score(&["dense", "bm25"], &[None, Some(1000.0)])
                .unwrap(),
            0.5
        );
        assert_eq!(
            model
                .score(&["dense", "bm25"], &[Some(0.0), Some(1000.0)])
                .unwrap(),
            2.5
        );
        assert_eq!(
            model
                .score(&["dense", "bm25"], &[Some(-0.5), Some(1000.0)])
                .unwrap(),
            1.5
        );
    }
}
