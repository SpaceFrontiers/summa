//! L2 reranker: rerank L1 candidates by exact dense vector distance on stored vectors

use crate::dsl::Field;
use crate::structures::simd::squared_euclidean_distance;

use super::{MultiValueCombiner, SearchResult};

/// Configuration for L2 dense vector reranking
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Dense vector field (must be stored)
    pub field: Field,
    /// Query vector
    pub vector: Vec<f32>,
    /// How to combine scores for multi-valued documents
    pub combiner: MultiValueCombiner,
}

/// Score a single document against the query vector.
///
/// Returns `None` if the document has no values for the given field.
fn score_document(doc: &crate::dsl::Document, config: &RerankerConfig) -> Option<f32> {
    let values: Vec<(u32, f32)> = doc
        .get_all(config.field)
        .enumerate()
        .filter_map(|(ordinal, fv)| {
            let vec = fv.as_dense_vector()?;
            let dist = squared_euclidean_distance(&config.vector, vec);
            let score = 1.0 / (1.0 + dist);
            Some((ordinal as u32, score))
        })
        .collect();

    if values.is_empty() {
        return None;
    }

    Some(config.combiner.combine(&values))
}

/// Rerank L1 candidates by exact dense vector distance.
///
/// For each candidate, loads the stored document, extracts the dense vector field,
/// computes squared Euclidean distance, and converts to a similarity score via
/// `1 / (1 + dist)`. Multi-valued fields are combined using `config.combiner`.
///
/// Documents missing the vector field are skipped.
pub async fn rerank<D: crate::directories::Directory + 'static>(
    searcher: &crate::index::Searcher<D>,
    candidates: &[SearchResult],
    config: &RerankerConfig,
    final_limit: usize,
) -> crate::error::Result<Vec<SearchResult>> {
    let mut scored: Vec<SearchResult> = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        if let Some(doc) = searcher.doc(candidate.doc_id).await?
            && let Some(score) = score_document(&doc, config)
        {
            scored.push(SearchResult {
                doc_id: candidate.doc_id,
                score,
                positions: Vec::new(),
            });
        }
    }

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(final_limit);

    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::{Document, Field};

    fn make_config(vector: Vec<f32>, combiner: MultiValueCombiner) -> RerankerConfig {
        RerankerConfig {
            field: Field(0),
            vector,
            combiner,
        }
    }

    #[test]
    fn test_score_document_single_value() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![1.0, 0.0, 0.0]);

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max);
        let score = score_document(&doc, &config).unwrap();
        // Distance = 0, score = 1 / (1 + 0) = 1.0
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_document_distance_correctness() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![3.0, 0.0, 0.0]);

        let config = make_config(vec![0.0, 0.0, 0.0], MultiValueCombiner::Max);
        let score = score_document(&doc, &config).unwrap();
        // Distance = 9.0, score = 1 / (1 + 9) = 0.1
        assert!((score - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_score_document_multi_value_max() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![1.0, 0.0, 0.0]); // dist=0, score=1.0
        doc.add_dense_vector(Field(0), vec![3.0, 0.0, 0.0]); // dist=4, score=0.2

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max);
        let score = score_document(&doc, &config).unwrap();
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_document_multi_value_avg() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![1.0, 0.0, 0.0]); // dist=0, score=1.0
        doc.add_dense_vector(Field(0), vec![3.0, 0.0, 0.0]); // dist=4, score=0.2

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Avg);
        let score = score_document(&doc, &config).unwrap();
        // avg(1.0, 0.2) = 0.6
        assert!((score - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_score_document_missing_field() {
        let mut doc = Document::new();
        // Add to field 1, not field 0
        doc.add_dense_vector(Field(1), vec![1.0, 0.0, 0.0]);

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max);
        assert!(score_document(&doc, &config).is_none());
    }

    #[test]
    fn test_score_document_wrong_field_type() {
        let mut doc = Document::new();
        doc.add_text(Field(0), "not a vector");

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max);
        assert!(score_document(&doc, &config).is_none());
    }
}
