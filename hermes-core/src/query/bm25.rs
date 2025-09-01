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

/// Per-field BM25 parameters (`indexed<k1: ..., b: ...>` in the schema).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self {
            k1: BM25_K1,
            b: BM25_B,
        }
    }
}

impl Bm25Params {
    /// Resolve a field's parameters from its schema entry.
    pub fn for_field(schema: &crate::dsl::Schema, field: crate::dsl::Field) -> Self {
        let entry = schema.get_field_entry(field);
        Self {
            k1: entry.and_then(|e| e.bm25_k1).unwrap_or(BM25_K1),
            b: entry.and_then(|e| e.bm25_b).unwrap_or(BM25_B),
        }
    }

    /// BM25 score of one term occurrence set.
    #[inline]
    pub fn score(self, tf: f32, idf: f32, doc_len: f32, avg_doc_len: f32) -> f32 {
        let length_norm = 1.0 - self.b + self.b * (doc_len / avg_doc_len.max(1.0));
        let tf_norm = (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm);
        idf * tf_norm
    }

    /// BM25F score with a field boost.
    #[inline]
    pub fn score_boosted(
        self,
        tf: f32,
        idf: f32,
        doc_len: f32,
        avg_doc_len: f32,
        field_boost: f32,
    ) -> f32 {
        let length_norm = 1.0 - self.b + self.b * (doc_len / avg_doc_len.max(1.0));
        let tf_norm =
            (tf * field_boost * (self.k1 + 1.0)) / (tf * field_boost + self.k1 * length_norm);
        idf * tf_norm
    }

    /// Upper bound with the shortest possible unit (length 0).
    #[inline]
    pub fn upper_bound(self, max_tf: f32, idf: f32) -> f32 {
        let min_length_norm = 1.0 - self.b;
        let tf_norm = (max_tf * (self.k1 + 1.0)) / (max_tf + self.k1 * min_length_norm);
        idf * tf_norm
    }

    /// Upper bound with a known minimum unit length.
    #[inline]
    pub fn upper_bound_with_len(self, max_tf: f32, idf: f32, min_len: f32, avg_len: f32) -> f32 {
        let length_norm = 1.0 - self.b + self.b * (min_len / avg_len.max(1.0));
        let tf_norm = (max_tf * (self.k1 + 1.0)) / (max_tf + self.k1 * length_norm);
        idf * tf_norm
    }
}

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

/// BM25 upper bound with a known minimum length of the scoring units the
/// bound covers (a block or a whole list): the shortest unit has the
/// weakest length normalisation, so it bounds every longer one.
#[inline]
pub fn bm25_upper_bound_with_len(max_tf: f32, idf: f32, min_len: f32, avg_len: f32) -> f32 {
    let length_norm = 1.0 - BM25_B + BM25_B * (min_len / avg_len.max(1.0));
    let tf_norm = (max_tf * (BM25_K1 + 1.0)) / (max_tf + BM25_K1 * length_norm);
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
