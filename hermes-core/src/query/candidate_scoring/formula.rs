//! Bounded, immutable symbolic scoring. Parsing happens at request admission;
//! raw features bind by index, with no per-passage name maps or shared scratch.
use super::CandidateScores;
use crate::query::{MultiValueCombiner, RrfScore};
use crate::{Error, Result};
use exmex::{Express, FlatEx, FloatOpsFactory, MakeOperators, Operator};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
struct MathOps;
impl MakeOperators<f64> for MathOps {
    fn make<'a>() -> Vec<Operator<'a, f64>> {
        let mut ops = FloatOpsFactory::<f64>::make();
        ops.push(Operator::make_unary("log1p", f64::ln_1p));
        ops.push(Operator::make_unary("expm1", f64::exp_m1));
        ops
    }
}

#[derive(Debug)]
struct Formula {
    expression: FlatEx<f64, MathOps>,
    names: Vec<String>,
    // None denotes the reserved RRF variable; defaults follow expression order.
    bindings: Vec<Option<usize>>,
    defaults: Vec<f64>,
}

/// Compiled, immutable L1 formula shared by shards and the coordinator.
#[derive(Clone, Debug)]
pub struct RankingModel(Arc<Formula>);

impl RankingModel {
    pub fn compile(
        formula: &str,
        names: &[&str],
        missing_values: &BTreeMap<String, f64>,
    ) -> Result<Self> {
        Ok(Self(Arc::new(Formula::compile(
            formula,
            names,
            missing_values,
        )?)))
    }

    pub fn validate(&self, names: &[&str]) -> Result<()> {
        if self
            .0
            .names
            .iter()
            .map(String::as_str)
            .ne(names.iter().copied())
        {
            return Err(Error::Query("L1 formula feature schema mismatch".into()));
        }
        Ok(())
    }

    pub fn needs_rrf(&self) -> bool {
        self.0.bindings.iter().any(Option::is_none)
    }

    pub fn score_candidate(
        &self,
        names: &[&str],
        features: &mut CandidateScores,
        combiner: MultiValueCombiner,
        rrf: Option<&RrfScore>,
    ) -> Result<f32> {
        if self.needs_rrf()
            && (rrf.is_none() || features.passages.len() != features.scored_passages)
        {
            return Err(Error::Query(
                "RRF L1 requires complete passage features and organic votes".into(),
            ));
        }
        super::model::score_candidate_with(names, features, combiner, rrf, |values, rrf| {
            self.0.score(values, rrf)
        })
    }
}

impl Formula {
    fn compile(text: &str, names: &[&str], missing: &BTreeMap<String, f64>) -> Result<Self> {
        validate_shape(text)?;
        if names.len() > crate::query::MAX_FUSION_SUB_QUERIES
            || names.iter().enumerate().any(|(index, name)| {
                name.is_empty()
                    || name.len() > 128
                    || !name.is_ascii()
                    || *name == "rrf"
                    || names[..index].contains(name)
            })
        {
            return Err(Error::Query(
                "L1 formula needs up to 16 unique ASCII branch names of 1..128 bytes; 'rrf' is reserved".into(),
            ));
        }
        let mut expression = FlatEx::<f64, MathOps>::parse_wo_compile(text)
            .map_err(|error| Error::Query(format!("invalid l1.formula: {error}")))?;
        let mut bindings = Vec::with_capacity(expression.var_names().len());
        let mut defaults = Vec::with_capacity(expression.var_names().len());
        for name in expression.var_names() {
            bindings.push(if name == "rrf" {
                None
            } else {
                Some(
                    names
                        .iter()
                        .position(|candidate| candidate == name)
                        .ok_or_else(|| {
                            Error::Query(format!("l1.formula references unknown branch '{name}'"))
                        })?,
                )
            });
            defaults.push(missing.get(name).copied().unwrap_or(0.0));
        }
        for (name, value) in missing {
            if !value.is_finite() || name == "rrf" || !expression.var_names().contains(name) {
                return Err(Error::Query(format!(
                    "l1 missing value '{name}' needs a formula variable and finite raw default"
                )));
            }
        }
        expression.compile();
        let formula = Self {
            expression,
            names: names.iter().map(|name| (*name).to_owned()).collect(),
            bindings,
            defaults,
        };
        if formula.bindings.is_empty() {
            formula.score(&vec![None; names.len()], None)?;
        }
        Ok(formula)
    }

    fn score(&self, values: &[Option<f32>], rrf: Option<f32>) -> Result<f32> {
        if values.len() != self.names.len() {
            return Err(Error::Query("L1 formula feature count mismatch".into()));
        }
        let mut variables = [0.0f64; crate::query::MAX_FUSION_SUB_QUERIES + 1];
        for (i, binding) in self.bindings.iter().enumerate() {
            variables[i] =
                match binding {
                    Some(index) => values[*index].map(f64::from).unwrap_or(self.defaults[i]),
                    None => f64::from(rrf.ok_or_else(|| {
                        Error::Query("L1 formula requires organic RRF votes".into())
                    })?),
                };
            if !variables[i].is_finite() {
                return Err(Error::Query(
                    "L1 formula received a non-finite feature".into(),
                ));
            }
        }
        let score = self
            .expression
            .eval(&variables[..self.bindings.len()])
            .map_err(|error| Error::Query(format!("L1 formula evaluation failed: {error}")))?;
        if !score.is_finite() || !(score as f32).is_finite() {
            return Err(Error::Query(
                "L1 formula produced a non-finite score (domain error or overflow)".into(),
            ));
        }
        Ok(score as f32)
    }
}

/// Count lexical units before the parser can allocate. This is a shape budget,
/// not a second grammar: the library validates the actual expression syntax.
fn validate_shape(text: &str) -> Result<()> {
    if text.is_empty() || text.len() > 4096 || !text.is_ascii() {
        return Err(Error::Query(
            "l1.formula requires 1..4096 ASCII bytes".into(),
        ));
    }
    let mut tokens = 0usize;
    let mut depth = 0usize;
    let mut word = false;
    for byte in text.bytes() {
        let next_word = byte.is_ascii_alphanumeric() || b"_.".contains(&byte);
        if !byte.is_ascii_whitespace() && (!next_word || !word) {
            tokens += 1;
        }
        word = next_word;
        match byte {
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            _ => {}
        }
        if tokens > 256 || depth > 32 {
            return Err(Error::Query(
                "l1.formula exceeds 256 tokens or 32 parenthesis levels".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn formula(text: &str, missing: BTreeMap<String, f64>) -> Result<Formula> {
        Formula::compile(text, &["dense", "body.bm25", "sparse"], &missing)
    }

    #[test]
    fn symbolic_formula_binds_raw_scores_and_defaults_without_changing_features() {
        let formula = formula("2 * ln(1 + {body.bm25}) + sqrt(abs(dense)) + log2(8) + log10(100) + max(sparse, 0) + 100 * rrf", BTreeMap::from([("sparse".into(), -2.0)])).unwrap();
        let values = [Some(-4.0), Some(3.0), None];
        let score = formula.score(&values, Some(0.02)).unwrap();
        assert_eq!(
            score,
            (2.0 * 4.0f64.ln() + 2.0 + 3.0 + 2.0 + 100.0 * f64::from(0.02f32)) as f32
        );
        assert_eq!(values, [Some(-4.0), Some(3.0), None]);
        assert_eq!(
            self::formula("sparse", BTreeMap::new())
                .unwrap()
                .score(&values, None)
                .unwrap(),
            0.0
        );
        assert_eq!(
            self::formula("log1p(dense) + expm1(sparse)", BTreeMap::new())
                .unwrap()
                .score(&[Some(1.0), None, Some(0.0)], None)
                .unwrap(),
            2.0f64.ln() as f32
        );
    }

    #[test]
    fn formula_rejects_unknown_names_complexity_and_nonfinite_predictions() {
        for text in [
            "",
            "typo * 0",
            "print(dense)",
            "dense = 1",
            "min(dense, sparse, 0)",
            "sqrt(-1)",
            "1/0",
        ] {
            assert!(formula(text, BTreeMap::new()).is_err(), "{text}");
        }
        assert!(
            formula(
                &format!("{}dense{}", "(".repeat(33), ")".repeat(33)),
                BTreeMap::new()
            )
            .is_err()
        );
        assert!(formula(&"dense+".repeat(200), BTreeMap::new()).is_err());
        assert!(formula(&" ".repeat(4097), BTreeMap::new()).is_err());
        for (text, dense) in [
            ("ln(dense)", -1.0),
            ("1/dense", 0.0),
            ("exp(dense)", 1000.0),
        ] {
            assert!(
                formula(text, BTreeMap::new())
                    .unwrap()
                    .score(&[Some(dense), None, None], None)
                    .is_err()
            );
        }
    }

    #[test]
    fn nonlinear_rrf_formula_selects_passage_before_document_max() {
        let model =
            RankingModel::compile("dense + 1000 * rrf", &["dense"], &BTreeMap::new()).unwrap();
        let mut features = CandidateScores {
            document: vec![None],
            scored_passages: 2,
            passages: vec![
                super::super::PassageFeatures {
                    ordinal: 0,
                    score: 0.0,
                    values: vec![Some(2.0)],
                },
                super::super::PassageFeatures {
                    ordinal: 1,
                    score: 0.0,
                    values: vec![Some(1.9)],
                },
            ],
        };
        let rrf = RrfScore {
            score: 1.0 / 61.0,
            contributions: vec![
                crate::query::RrfContribution {
                    query_index: 0,
                    rank: 2,
                    score: 1.0 / 62.0,
                    ordinal: Some(0),
                },
                crate::query::RrfContribution {
                    query_index: 0,
                    rank: 1,
                    score: 1.0 / 61.0,
                    ordinal: Some(1),
                },
            ],
        };
        let score = model
            .score_candidate(
                &["dense"],
                &mut features,
                MultiValueCombiner::Max,
                Some(&rrf),
            )
            .unwrap();
        assert_eq!(score, features.passages[1].score);
        assert!(score > features.passages[0].score);
        features.passages.pop();
        assert!(
            model
                .score_candidate(
                    &["dense"],
                    &mut features,
                    MultiValueCombiner::Max,
                    Some(&rrf)
                )
                .is_err()
        );
    }
}
