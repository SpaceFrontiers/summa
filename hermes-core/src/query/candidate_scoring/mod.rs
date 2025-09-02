//! Complete cross-vertical scores over a bounded candidate union, independent
//! of retrieval nomination. Linear ranking and RRF are alternative policies.
mod execution;
mod model;
pub use model::{FeatureTransform, LinearModel};

use super::{PhraseQuery, QueryDecomposition};
use crate::{Error, Field, Result};

/// Explicit alignment contract: fields declared Chunk share logical ordinals.
/// A document feature is reduced once with MAX and broadcast to its passages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreScope {
    Document,
    Chunk,
}

#[derive(Clone, Debug)]
pub(crate) enum ScoreComponent {
    Text(Vec<(Vec<u8>, f32)>),
    Phrase(PhraseQuery),
    Sparse(Vec<(u32, f32)>),
    Dense(Vec<f32>),
    Binary(Vec<u8>),
}

/// Prepared from an ordinary query through its owning Query implementation.
/// Scoring compositions must use one field; eligibility belongs in the common
/// fusion filter. This avoids silently treating a profile ordinal as body zero.
#[derive(Clone, Debug)]
pub struct CandidateQuery {
    pub(crate) field: Field,
    pub(crate) components: Vec<(ScoreComponent, f32)>,
}
impl CandidateQuery {
    pub fn field(&self) -> Field {
        self.field
    }
    pub(crate) fn new(field: Field, component: ScoreComponent) -> Self {
        Self {
            field,
            components: vec![(component, 1.0)],
        }
    }
    pub(crate) fn from_decomposition(decomposition: QueryDecomposition) -> Result<Self> {
        match decomposition {
            QueryDecomposition::TextTerm(term) => Ok(Self::new(term.field, ScoreComponent::Text(vec![(term.term, term.weight)]))),
            QueryDecomposition::SparseTerms(infos) if !infos.is_empty() => {
                let field = infos[0].field;
                if infos.iter().any(|info| info.field != field) { return Err(Error::Query("L1 branch mixes sparse fields".into())); }
                Ok(Self::new(field, ScoreComponent::Sparse(infos.into_iter().map(|info| (info.dim_id, info.weight)).collect())))
            }
            _ => Err(Error::Query("query cannot be backfilled as an L1 score; use text/phrase/vector scoring branches and the common fusion filter for eligibility".into())),
        }
    }
    pub(crate) fn boosted(mut self, weight: f32) -> Result<Self> {
        for (_, boost) in &mut self.components {
            *boost *= weight;
            if !boost.is_finite() {
                return Err(Error::Query("L1 query boost overflow".into()));
            }
        }
        Ok(self)
    }
    pub(crate) fn sum(queries: impl IntoIterator<Item = Result<Self>>) -> Result<Self> {
        let mut result: Option<Self> = None;
        for query in queries {
            let query = query?;
            if let Some(current) = &mut result {
                if current.field != query.field {
                    return Err(Error::Query("L1 scoring branch must use one field; name separate branches for separate fields".into()));
                }
                current.components.extend(query.components);
            } else {
                result = Some(query);
            }
        }
        result.ok_or_else(|| Error::Query("L1 scoring branch is empty".into()))
    }
    pub fn text_terms(&self, out: &mut Vec<(Field, Vec<u8>)>) {
        for (component, _) in &self.components {
            match component {
                ScoreComponent::Text(terms) => {
                    out.extend(terms.iter().map(|(term, _)| (self.field, term.clone())))
                }
                ScoreComponent::Phrase(query) => {
                    out.extend(query.terms.iter().map(|term| (self.field, term.clone())))
                }
                _ => {}
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct CandidateFeature {
    pub name: String,
    pub scope: ScoreScope,
    pub query: CandidateQuery,
}

#[derive(Clone, Debug)]
pub struct CandidateScoringPlan {
    pub features: Vec<CandidateFeature>,
    /// None exports raw features; Some ranks directly with this model.
    pub model: Option<LinearModel>,
    /// Bounds only response feature rows; every candidate passage is scored
    /// before top passages are selected. This does not change indexed chunks.
    pub export_passages: usize,
    /// Diagnostics may request every stored passage. Search normally scores
    /// only the union of nominated passage ordinals, plus document context.
    pub all_passages: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PassageFeatures {
    pub ordinal: u16,
    pub score: f32,
    pub values: Vec<Option<f32>>,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CandidateScores {
    /// Values in declared branch order. Entries for chunk features are None.
    pub document: Vec<Option<f32>>,
    pub passages: Vec<PassageFeatures>,
    pub scored_passages: usize,
}
#[derive(Clone, Debug)]
pub struct ScoredCandidate {
    pub result: super::SearchResult,
    pub features: CandidateScores,
}

#[cfg(test)]
mod tests;
