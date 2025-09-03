//! Named scores over a bounded candidate union: preserve organic retrieval
//! values, optionally backfill missing cells, and apply the shared linear model.
mod execution;
mod model;
mod retrieved;
pub use model::{FeatureTransform, LinearModel};

use super::{MultiValueCombiner, PhraseQuery, QueryDecomposition};
use crate::{Error, Field, Result};

/// Explicit alignment contract: fields declared Chunk share logical ordinals.
/// A document feature is reduced by its query and broadcast to its passages.
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
    document: DocumentExpression,
}

/// Keep expression order for document features. In particular MAX(a)+MAX(b)
/// and -MAX(a) cannot be flattened into MAX(a+b) and MAX(-a).
#[derive(Clone, Debug)]
enum DocumentExpression {
    Component(usize, MultiValueCombiner),
    Sum(Vec<Self>),
    Boost(Box<Self>, f32),
}
impl DocumentExpression {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Component(_, combiner) => combiner.validate().map_err(Error::Query),
            Self::Sum(children) => children.iter().try_for_each(Self::validate),
            Self::Boost(child, boost) => {
                if !boost.is_finite() {
                    return Err(Error::Query("L1 query boost must be finite".into()));
                }
                child.validate()
            }
        }
    }
    fn rebase(&mut self, offset: usize) {
        match self {
            Self::Component(index, _) => *index += offset,
            Self::Sum(children) => children.iter_mut().for_each(|child| child.rebase(offset)),
            Self::Boost(child, _) => child.rebase(offset),
        }
    }
    fn score(&self, components: &[Vec<f32>], locations: &[(u32, usize)]) -> Result<f32> {
        let value = match self {
            Self::Component(index, combiner) => {
                let values: Vec<_> = locations
                    .iter()
                    .map(|&(ordinal, position)| (ordinal, components[*index][position]))
                    .collect();
                combiner.combine(&values)
            }
            Self::Sum(children) => {
                let mut value = 0.0;
                for child in children {
                    value += child.score(components, locations)?;
                }
                value
            }
            Self::Boost(child, boost) => child.score(components, locations)? * boost,
        };
        if !value.is_finite() {
            return Err(Error::Query(
                "L1 document feature reduction overflow".into(),
            ));
        }
        Ok(value)
    }
}
impl CandidateQuery {
    pub fn field(&self) -> Field {
        self.field
    }
    pub(crate) fn new(field: Field, component: ScoreComponent) -> Self {
        Self {
            field,
            components: vec![(component, 1.0)],
            document: DocumentExpression::Component(0, MultiValueCombiner::Max),
        }
    }
    pub(crate) fn with_combiner(mut self, combiner: MultiValueCombiner) -> Self {
        self.document = DocumentExpression::Component(0, combiner);
        self
    }
    pub(crate) fn from_decomposition(decomposition: QueryDecomposition) -> Result<Self> {
        match decomposition {
            QueryDecomposition::TextTerm(term) => Ok(Self::new(term.field, ScoreComponent::Text(vec![(term.term, term.weight)]))),
            QueryDecomposition::SparseTerms(infos) if !infos.is_empty() => {
                let field = infos[0].field;
                if infos.iter().any(|info| info.field != field) { return Err(Error::Query("L1 branch mixes sparse fields".into())); }
                let combiner = infos[0].combiner;
                if infos.iter().any(|info| info.combiner != combiner) { return Err(Error::Query("L1 sparse composition mixes document combiners".into())); }
                Ok(Self::new(field, ScoreComponent::Sparse(infos.into_iter().map(|info| (info.dim_id, info.weight)).collect())).with_combiner(combiner))
            }
            _ => Err(Error::Query("query cannot be backfilled as an L1 score; use text/phrase/vector scoring branches and the common fusion filter for eligibility".into())),
        }
    }
    pub(crate) fn boosted(mut self, weight: f32) -> Result<Self> {
        self.document = DocumentExpression::Boost(Box::new(self.document), weight);
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
            let mut query = query?;
            if let Some(current) = &mut result {
                if current.field != query.field {
                    return Err(Error::Query("L1 scoring branch must use one field; name separate branches for separate fields".into()));
                }
                query.document.rebase(current.components.len());
                current.components.extend(query.components);
                let previous =
                    std::mem::replace(&mut current.document, DocumentExpression::Sum(Vec::new()));
                current.document = match previous {
                    DocumentExpression::Sum(mut children) => {
                        children.push(query.document);
                        DocumentExpression::Sum(children)
                    }
                    previous => DocumentExpression::Sum(vec![previous, query.document]),
                };
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
    /// Probe only cells missing from retrieval. False preserves missing values.
    pub backfill: bool,
    /// None exports raw features; Some ranks directly with this model.
    pub model: Option<LinearModel>,
    /// Bounds only response feature rows; every candidate passage is scored
    /// before top passages are selected. This does not change indexed chunks.
    pub export_passages: usize,
    /// Diagnostics may request every stored passage. Search normally scores
    /// only the union of nominated passage ordinals, plus document context.
    pub all_passages: bool,
    /// Reduce all final passage predictions before export truncation/top-K.
    pub document_combiner: MultiValueCombiner,
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
