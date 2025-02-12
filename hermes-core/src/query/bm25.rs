//! BM25/BM25F scoring constants and utilities
//!
//! Shared BM25 parameters used across full-text scoring implementations.
//! All posting list formats and scoring executors should use these functions.

/// BM25 k1 parameter - controls term frequency saturation
/// Higher values give more weight to term frequency
pub const BM25_K1: f32 = 1.2;

/// BM25 b parameter - controls length normalization
/// 0 = no length normalization, 1 = full normalization
pub const BM25_B: f32 = 0.75;

/// Compute BM25 score for a term occurrence
///
/// # Arguments
/// * `tf` - Term frequency in document
/// * `idf` - Inverse document frequency
/// * `doc_len` - Document length (or field length)
/// * `avg_doc_len` - Average document length
#[inline]
pub fn bm25_score(tf: f32, idf: f32, doc_len: f32, avg_doc_len: f32) -> f32 {
    let length_norm = 1.0 - BM25_B + BM25_B * (doc_len / avg_doc_len.max(1.0));
    let tf_norm = (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * length_norm);
    idf * tf_norm
}

/// Compute BM25F score with field boost
///
/// # Arguments
/// * `tf` - Term frequency in document
/// * `idf` - Inverse document frequency
/// * `doc_len` - Document length (or field length)
/// * `avg_doc_len` - Average document length
/// * `field_boost` - Field-specific boost factor
#[inline]
pub fn bm25f_score(tf: f32, idf: f32, doc_len: f32, avg_doc_len: f32, field_boost: f32) -> f32 {
    let length_norm = 1.0 - BM25_B + BM25_B * (doc_len / avg_doc_len.max(1.0));
    let tf_norm = (tf * field_boost * (BM25_K1 + 1.0)) / (tf * field_boost + BM25_K1 * length_norm);
    idf * tf_norm
}

/// Compute BM25 upper bound score for MaxScore pruning
///
/// Uses conservative assumptions for maximum possible score:
/// - Maximum TF from posting list
/// - Minimum length normalization (shortest possible document)
#[inline]
pub fn bm25_upper_bound(max_tf: f32, idf: f32) -> f32 {
    let min_length_norm = 1.0 - BM25_B;
    let tf_norm = (max_tf * (BM25_K1 + 1.0)) / (max_tf + BM25_K1 * min_length_norm);
    idf * tf_norm
}

/// Compute BM25F upper bound score for MaxScore pruning with field boost
///
/// Uses conservative assumptions for maximum possible score:
/// - Maximum TF from posting list
/// - Minimum length normalization (shortest possible document)
/// - Field boost factor
#[inline]
pub fn bm25f_upper_bound(max_tf: f32, idf: f32, field_boost: f32) -> f32 {
    let min_length_norm = 1.0 - BM25_B;
    let tf_norm = (max_tf * field_boost * (BM25_K1 + 1.0))
        / (max_tf * field_boost + BM25_K1 * min_length_norm);
    idf * tf_norm
}

/// Compute IDF (Inverse Document Frequency) using BM25 variant
///
/// # Arguments
/// * `doc_freq` - Number of documents containing the term
/// * `total_docs` - Total number of documents in collection
#[inline]
pub fn bm25_idf(doc_freq: f32, total_docs: f32) -> f32 {
    ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln()
}
