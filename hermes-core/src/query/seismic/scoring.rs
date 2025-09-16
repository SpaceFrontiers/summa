//! Shared exact sparse-vector scoring and query preparation.
use crate::segment::VectorOrdinals;
use crate::segment::seismic::SeismicIndex;
use crate::{Error, Result};

/// A query-local table removes the merge walk from common-vocabulary scoring.
/// Its allocation is bounded independently of index vocabulary/corpus size.
pub(super) const MAX_LOOKUP_DIMENSIONS: usize = 65_536;

pub(super) enum PreparedQuery {
    Lookup {
        weights: Vec<f32>,
        terms: Vec<(u32, f32)>,
    },
    Sorted(Vec<(u32, f32)>),
}
impl PreparedQuery {
    pub(super) fn new(mut terms: Vec<(u32, f32)>) -> Self {
        terms.retain(|&(_, weight)| weight != 0.0);
        terms.sort_unstable_by_key(|&(dimension, _)| dimension);
        let dimensions = terms
            .last()
            .map_or(0, |&(dimension, _)| u64::from(dimension) + 1);
        if dimensions <= MAX_LOOKUP_DIMENSIONS as u64
            && terms.windows(2).all(|pair| pair[0].0 != pair[1].0)
        {
            let mut lookup = vec![0.0; dimensions as usize];
            for &(dimension, weight) in &terms {
                lookup[dimension as usize] = weight;
            }
            Self::Lookup {
                weights: lookup,
                terms,
            }
        } else {
            // Aggregating repeated query coordinates changes floating-point
            // arithmetic. Keep their individual multiply/add operations.
            Self::Sorted(terms)
        }
    }

    pub(super) fn terms(&self) -> &[(u32, f32)] {
        match self {
            Self::Lookup { terms, .. } | Self::Sorted(terms) => terms,
        }
    }

    /// `None` distinguishes a missing match from a matching zero (including
    /// cancellation and a stored zero weight). Query zero weights are absent.
    pub(super) fn score_vector(
        &self,
        values: impl Iterator<Item = (u32, f32)>,
    ) -> Result<Option<f32>> {
        let score = self.dot_product::<false>(values);
        if score.is_some_and(|score| !score.is_finite()) {
            return Err(Error::Query("Seismic sparse score overflow".into()));
        }
        Ok(score)
    }

    /// Coordinate maxima can come from different rows. Their nonnegative
    /// nomination proxy may overflow while every exact document score remains
    /// finite; infinity keeps such a cluster eligible.
    #[cfg(test)]
    pub(super) fn summary_score(&self, values: impl Iterator<Item = (u32, f32)>) -> f32 {
        self.dot_product::<true>(values).unwrap_or(0.0)
    }

    fn dot_product<const ABSOLUTE: bool>(
        &self,
        values: impl Iterator<Item = (u32, f32)>,
    ) -> Option<f32> {
        let mut score = 0.0f32;
        let mut matched = false;
        match self {
            Self::Lookup {
                weights: lookup, ..
            } => {
                values.for_each(|(dimension, value)| {
                    let weight = lookup.get(dimension as usize).copied().unwrap_or(0.0);
                    if weight != 0.0 {
                        matched = true;
                        let weight = if ABSOLUTE { weight.abs() } else { weight };
                        score = score.algebraic_add(value.algebraic_mul(weight));
                    }
                });
            }
            Self::Sorted(terms) => {
                let mut query = terms.as_slice();
                values.for_each(|(dimension, value)| {
                    while query.first().is_some_and(|&(dim, _)| dim < dimension) {
                        query = &query[1..];
                    }
                    for &(_, weight) in query.iter().take_while(|&&(dim, _)| dim == dimension) {
                        matched = true;
                        let weight = if ABSOLUTE { weight.abs() } else { weight };
                        score = score.algebraic_add(value.algebraic_mul(weight));
                    }
                });
            }
        }
        matched.then_some(score)
    }
}

pub(super) fn score_row(
    index: &SeismicIndex,
    row: u32,
    query: &PreparedQuery,
) -> Result<Option<f32>> {
    let vector = index.vector(row);
    #[cfg(feature = "query-diagnostics")]
    crate::search_diagnostics::update(|work| {
        work.seismic_forward_rows += 1;
        work.seismic_forward_bytes += vector.byte_len() as u64;
    });
    if let Some(values) = vector.raw_iter() {
        query.score_vector(values)
    } else {
        query.score_vector(vector.iter())
    }
}

pub(in crate::query) fn score_candidates(
    index: &SeismicIndex,
    terms: &[(u32, f32)],
    rows: &[u32],
) -> Result<Vec<f32>> {
    let query = PreparedQuery::new(terms.to_vec());
    rows.iter()
        .map(|&row| score_row(index, row, &query).map(|score| score.unwrap_or(0.0)))
        .collect()
}

pub(super) fn document_ordinals(
    index: &SeismicIndex,
    doc: u32,
    query: &PreparedQuery,
    out: &mut VectorOrdinals,
    mut matches: Option<&mut VectorOrdinals>,
) -> Result<()> {
    out.clear();
    if let Some(matches) = &mut matches {
        matches.clear();
    }
    let mut matched = false;
    for row in index.rows_for_document(doc) {
        let score = score_row(index, row, query)?;
        matched |= score.is_some();
        let ordinal = u32::from(index.key(row).ordinal);
        if let Some(score) = score
            && let Some(matches) = &mut matches
        {
            matches.push((ordinal, score));
        }
        out.push((ordinal, score.unwrap_or(0.0)));
    }
    if !matched {
        out.clear();
    }
    Ok(())
}
