//! WAND (Weak AND) data structures for efficient top-k retrieval
//!
//! This module provides pre-computed data structures that enable WAND and BlockMax WAND
//! query optimization. The key insight is that we can pre-compute upper bound scores
//! for each term, allowing us to skip documents that can't possibly make it into the
//! top-k results.
//!
//! # Usage
//!
//! 1. Pre-compute term statistics using `hermes-tool term-stats`:
//!    ```bash
//!    cat docs.jsonl | hermes-tool term-stats --field content > wand_stats.json
//!    ```
//!
//! 2. Load the statistics during indexing or query time:
//!    ```rust,ignore
//!    let wand_data = WandData::from_json_file("wand_stats.json")?;
//!    let idf = wand_data.get_idf("content", "hello").unwrap_or(0.0);
//!    ```

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Per-term statistics for WAND optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermWandInfo {
    /// Document frequency (number of documents containing this term)
    pub df: u32,
    /// Total term frequency across all documents
    pub total_tf: u64,
    /// Maximum term frequency in any single document
    pub max_tf: u32,
    /// IDF value: log((N - df + 0.5) / (df + 0.5))
    pub idf: f32,
    /// Upper bound score for this term (BM25 with max_tf and conservative length norm)
    pub upper_bound: f32,
}

/// Collection-level WAND data
///
/// Contains pre-computed statistics needed for efficient WAND query processing.
/// This data is typically computed offline using `hermes-tool term-stats` and
/// loaded at index open time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WandData {
    /// Total number of documents in the collection
    pub total_docs: u64,
    /// Total number of tokens across all documents
    pub total_tokens: u64,
    /// Average document length (tokens per document)
    pub avg_doc_len: f32,
    /// BM25 k1 parameter used for computing upper bounds
    pub bm25_k1: f32,
    /// BM25 b parameter used for computing upper bounds
    pub bm25_b: f32,
    /// Per-term statistics, keyed by "field:term"
    #[serde(skip)]
    term_map: HashMap<String, TermWandInfo>,
    /// Raw term list (for serialization)
    terms: Vec<TermEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TermEntry {
    term: String,
    df: u32,
    total_tf: u64,
    max_tf: u32,
    idf: f32,
    upper_bound: f32,
}

impl WandData {
    /// Create empty WAND data
    pub fn new(total_docs: u64, avg_doc_len: f32) -> Self {
        Self {
            total_docs,
            total_tokens: (total_docs as f32 * avg_doc_len) as u64,
            avg_doc_len,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            term_map: HashMap::new(),
            terms: Vec::new(),
        }
    }

    /// Load WAND data from a JSON file
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(Error::Io)?;
        let reader = std::io::BufReader::new(file);
        Self::from_json_reader(reader)
    }

    /// Load WAND data from a JSON reader
    pub fn from_json_reader<R: Read>(reader: R) -> Result<Self> {
        let mut data: WandData =
            serde_json::from_reader(reader).map_err(|e| Error::Serialization(e.to_string()))?;
        data.build_term_map();
        Ok(data)
    }

    /// Load WAND data from JSON bytes
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let mut data: WandData =
            serde_json::from_slice(bytes).map_err(|e| Error::Serialization(e.to_string()))?;
        data.build_term_map();
        Ok(data)
    }

    /// Save WAND data to a JSON file
    pub fn to_json_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = std::fs::File::create(path).map_err(Error::Io)?;
        let writer = std::io::BufWriter::new(file);
        self.to_json_writer(writer)
    }

    /// Write WAND data to a JSON writer
    pub fn to_json_writer<W: Write>(&self, writer: W) -> Result<()> {
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        Ok(())
    }

    /// Build the term map from the terms vector
    fn build_term_map(&mut self) {
        self.term_map.clear();
        for entry in &self.terms {
            self.term_map.insert(
                entry.term.clone(),
                TermWandInfo {
                    df: entry.df,
                    total_tf: entry.total_tf,
                    max_tf: entry.max_tf,
                    idf: entry.idf,
                    upper_bound: entry.upper_bound,
                },
            );
        }
    }

    /// Get IDF for a term in a field
    ///
    /// Returns None if the term is not found in the pre-computed data.
    /// In that case, you should compute IDF on-the-fly using the segment's
    /// document count and term document frequency.
    pub fn get_idf(&self, field: &str, term: &str) -> Option<f32> {
        let key = format!("{}:{}", field, term);
        self.term_map.get(&key).map(|info| info.idf)
    }

    /// Get full term info for a term in a field
    pub fn get_term_info(&self, field: &str, term: &str) -> Option<&TermWandInfo> {
        let key = format!("{}:{}", field, term);
        self.term_map.get(&key)
    }

    /// Get upper bound score for a term
    pub fn get_upper_bound(&self, field: &str, term: &str) -> Option<f32> {
        let key = format!("{}:{}", field, term);
        self.term_map.get(&key).map(|info| info.upper_bound)
    }

    /// Compute IDF for a term given its document frequency
    ///
    /// Uses the BM25 IDF formula: log((N - df + 0.5) / (df + 0.5))
    pub fn compute_idf(&self, df: u32) -> f32 {
        let n = self.total_docs as f32;
        let df = df as f32;
        ((n - df + 0.5) / (df + 0.5)).ln()
    }

    /// Compute upper bound score for a term given max_tf and IDF
    ///
    /// Uses conservative length normalization (assumes shortest possible document)
    pub fn compute_upper_bound(&self, max_tf: u32, idf: f32) -> f32 {
        let tf = max_tf as f32;
        let min_length_norm = 1.0 - self.bm25_b;
        let tf_norm = (tf * (self.bm25_k1 + 1.0)) / (tf + self.bm25_k1 * min_length_norm);
        idf * tf_norm
    }

    /// Add or update term statistics
    pub fn add_term(&mut self, field: &str, term: &str, df: u32, total_tf: u64, max_tf: u32) {
        let idf = self.compute_idf(df);
        let upper_bound = self.compute_upper_bound(max_tf, idf);
        let key = format!("{}:{}", field, term);

        let info = TermWandInfo {
            df,
            total_tf,
            max_tf,
            idf,
            upper_bound,
        };

        self.term_map.insert(key.clone(), info.clone());
        self.terms.push(TermEntry {
            term: key,
            df,
            total_tf,
            max_tf,
            idf,
            upper_bound,
        });
    }

    /// Get the number of terms in the WAND data
    pub fn num_terms(&self) -> usize {
        self.term_map.len()
    }

    /// Check if WAND data is available for a term
    pub fn has_term(&self, field: &str, term: &str) -> bool {
        let key = format!("{}:{}", field, term);
        self.term_map.contains_key(&key)
    }
}

impl Default for WandData {
    fn default() -> Self {
        Self::new(0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wand_data_basic() {
        let mut wand = WandData::new(1000, 100.0);
        wand.add_term("content", "hello", 100, 500, 10);
        wand.add_term("content", "world", 50, 200, 5);

        assert!(wand.has_term("content", "hello"));
        assert!(wand.has_term("content", "world"));
        assert!(!wand.has_term("content", "missing"));

        let hello_idf = wand.get_idf("content", "hello").unwrap();
        let world_idf = wand.get_idf("content", "world").unwrap();

        // "world" appears in fewer docs, so should have higher IDF
        assert!(world_idf > hello_idf);
    }

    #[test]
    fn test_wand_data_serialization() {
        let mut wand = WandData::new(1000, 100.0);
        wand.add_term("title", "test", 50, 100, 3);

        let json = serde_json::to_string(&wand).unwrap();
        let restored = WandData::from_json_bytes(json.as_bytes()).unwrap();

        assert_eq!(restored.total_docs, wand.total_docs);
        assert_eq!(restored.avg_doc_len, wand.avg_doc_len);
        assert!(restored.has_term("title", "test"));
    }

    #[test]
    fn test_compute_idf() {
        let wand = WandData::new(1000, 100.0);

        // Rare term (df=10) should have high IDF
        let rare_idf = wand.compute_idf(10);
        // Common term (df=500) should have low IDF
        let common_idf = wand.compute_idf(500);

        assert!(rare_idf > common_idf);
        assert!(rare_idf > 0.0);
    }
}
