//! L2 reranker: rerank L1 candidates by exact dense vector distance on stored vectors

use crate::dsl::Field;
use crate::structures::simd::cosine_similarity;

use super::{MultiValueCombiner, ScoredPosition, SearchResult};

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

/// Score a single document against the query vector (used by tests).
#[cfg(test)]
fn score_document(
    doc: &crate::dsl::Document,
    config: &RerankerConfig,
) -> Option<(f32, Vec<ScoredPosition>)> {
    let query_dim = config.vector.len();
    let mut values: Vec<(u32, f32)> = doc
        .get_all(config.field)
        .filter_map(|fv| fv.as_dense_vector())
        .enumerate()
        .filter_map(|(ordinal, vec)| {
            if vec.len() != query_dim {
                return None;
            }
            let score = cosine_similarity(&config.vector, vec);
            Some((ordinal as u32, score))
        })
        .collect();

    if values.is_empty() {
        return None;
    }

    let combined = config.combiner.combine(&values);

    // Sort ordinals by score descending (best chunk first)
    values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let positions: Vec<ScoredPosition> = values
        .into_iter()
        .map(|(ordinal, score)| ScoredPosition::new(ordinal, score))
        .collect();

    Some((combined, positions))
}

/// Rerank L1 candidates by exact dense vector distance.
///
/// Reads vectors directly from flat vector data (mmap) instead of loading
/// full documents from the store. This avoids store block decompression
/// and document deserialization — typically 10-50× faster.
///
/// For each candidate, resolves the segment and flat vector index via
/// binary search, reads the raw vector, dequantizes to f32, and scores
/// with SIMD cosine similarity.
///
/// Documents missing the vector field are skipped.
pub async fn rerank<D: crate::directories::Directory + 'static>(
    searcher: &crate::index::Searcher<D>,
    candidates: &[SearchResult],
    config: &RerankerConfig,
    final_limit: usize,
) -> crate::error::Result<Vec<SearchResult>> {
    if config.vector.is_empty() || candidates.is_empty() {
        return Ok(Vec::new());
    }

    let t0 = std::time::Instant::now();
    let field_id = config.field.0;
    let query = &config.vector;
    let query_dim = query.len();
    let segments = searcher.segment_readers();
    let seg_by_id = searcher.segment_map();

    // For each candidate, batch-read all its ordinals in one mmap call
    // (vectors for the same doc are contiguous in the flat store)
    let t_read = std::time::Instant::now();
    let mut ordinal_scores: Vec<Vec<(u32, f32)>> = vec![Vec::new(); candidates.len()];
    let mut skipped = 0u32;
    let mut total_vectors = 0usize;
    let mut vec_buf = vec![0f32; query_dim];

    for (ci, candidate) in candidates.iter().enumerate() {
        let Some(&si) = seg_by_id.get(&candidate.segment_id) else {
            skipped += 1;
            continue;
        };

        let local_doc_id = candidate.doc_id - segments[si].doc_id_offset();
        let Some(lazy_flat) = segments[si].flat_vectors().get(&field_id) else {
            skipped += 1;
            continue;
        };

        if lazy_flat.dim != query_dim {
            skipped += 1;
            continue;
        }

        let (start, entries) = lazy_flat.flat_indexes_for_doc(local_doc_id);
        if entries.is_empty() {
            skipped += 1;
            continue;
        }

        let count = entries.len();
        total_vectors += count;

        // One batch read for all ordinals of this doc (contiguous in flat store)
        let batch = match lazy_flat.read_vectors_batch(start, count).await {
            Ok(b) => b,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        let vbs = lazy_flat.vector_byte_size();
        let raw = batch.as_slice();

        for (j, &(_doc_id, ordinal)) in entries.iter().enumerate() {
            let offset = j * vbs;
            crate::segment::dequantize_raw(
                &raw[offset..offset + vbs],
                lazy_flat.quantization,
                query_dim,
                &mut vec_buf,
            );
            let score = cosine_similarity(query, &vec_buf);
            ordinal_scores[ci].push((ordinal as u32, score));
        }
    }

    let resolve_elapsed = t0.elapsed();
    let read_score_elapsed = t_read.elapsed();

    if total_vectors == 0 {
        log::debug!(
            "[reranker] field {}: {} candidates, all skipped (no flat vectors)",
            field_id,
            candidates.len()
        );
        return Ok(Vec::new());
    }

    // Combine per-candidate ordinal scores and build results
    let mut scored: Vec<SearchResult> = Vec::with_capacity(candidates.len());
    for (ci, ordinals) in ordinal_scores.into_iter().enumerate() {
        if ordinals.is_empty() {
            continue;
        }
        let combined = config.combiner.combine(&ordinals);
        let mut positions: Vec<ScoredPosition> = ordinals
            .into_iter()
            .map(|(ord, score)| ScoredPosition::new(ord, score))
            .collect();
        positions.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.push(SearchResult {
            doc_id: candidates[ci].doc_id,
            score: combined,
            segment_id: candidates[ci].segment_id,
            positions: vec![(field_id, positions)],
        });
    }

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(final_limit);

    log::debug!(
        "[reranker] field {}: {} candidates -> {} results (skipped {}, {} vectors): resolve={:.1}ms read+score={:.1}ms total={:.1}ms",
        field_id,
        candidates.len(),
        scored.len(),
        skipped,
        total_vectors,
        resolve_elapsed.as_secs_f64() * 1000.0,
        read_score_elapsed.as_secs_f64() * 1000.0,
        t0.elapsed().as_secs_f64() * 1000.0,
    );

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
        let (score, positions) = score_document(&doc, &config).unwrap();
        // cosine([1,0,0], [1,0,0]) = 1.0
        assert!((score - 1.0).abs() < 1e-6);
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].position, 0); // ordinal 0
    }

    #[test]
    fn test_score_document_orthogonal() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![0.0, 1.0, 0.0]);

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max);
        let (score, _) = score_document(&doc, &config).unwrap();
        // cosine([1,0,0], [0,1,0]) = 0.0
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn test_score_document_multi_value_max() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![1.0, 0.0, 0.0]); // cos=1.0 (same direction)
        doc.add_dense_vector(Field(0), vec![0.0, 1.0, 0.0]); // cos=0.0 (orthogonal)

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max);
        let (score, positions) = score_document(&doc, &config).unwrap();
        assert!((score - 1.0).abs() < 1e-6);
        // Best chunk first
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].position, 0); // ordinal 0 scored highest
        assert!((positions[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_document_multi_value_avg() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![1.0, 0.0, 0.0]); // cos=1.0
        doc.add_dense_vector(Field(0), vec![0.0, 1.0, 0.0]); // cos=0.0

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Avg);
        let (score, _) = score_document(&doc, &config).unwrap();
        // avg(1.0, 0.0) = 0.5
        assert!((score - 0.5).abs() < 1e-6);
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

    #[test]
    fn test_score_document_dimension_mismatch() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![1.0, 0.0]); // 2D

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max); // 3D query
        assert!(score_document(&doc, &config).is_none());
    }

    #[test]
    fn test_score_document_empty_query_vector() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![1.0, 0.0, 0.0]);

        let config = make_config(vec![], MultiValueCombiner::Max);
        // Empty query can't match any stored vector (dimension mismatch)
        assert!(score_document(&doc, &config).is_none());
    }
}
