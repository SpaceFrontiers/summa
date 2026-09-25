//! HuggingFace tokenizer support for sparse vector queries
//!
//! Provides query-time tokenization using HuggingFace tokenizers.
//! Used when a sparse vector field has `query_tokenizer` configured.
//!
//! Supports both native and WASM targets:
//! - Native: Full support with `onig` regex and HTTP hub downloads
//! - WASM: Limited to `from_bytes()` loading (no HTTP, no onig regex)

use std::collections::HashMap;
#[cfg(feature = "native")]
use std::sync::Arc;

#[cfg(feature = "native")]
use parking_lot::RwLock;
use tokenizers::Tokenizer;

use log::debug;

use crate::Result;
use crate::error::Error;

/// Cached HuggingFace tokenizer
pub struct HfTokenizer {
    pub(crate) tokenizer: Tokenizer,
}

/// Tokenizer source - where to load the tokenizer from
#[derive(Debug, Clone)]
pub enum TokenizerSource {
    /// Load from HuggingFace hub (e.g., "bert-base-uncased") - native only
    #[cfg(not(target_arch = "wasm32"))]
    HuggingFace(String),
    /// Load from local file path - native only
    #[cfg(not(target_arch = "wasm32"))]
    LocalFile(String),
    /// Load from index directory (relative path within index)
    IndexDirectory(String),
}

impl TokenizerSource {
    /// Parse a tokenizer path string into a TokenizerSource
    ///
    /// - Paths starting with `index://` are relative to index directory
    /// - On native: Paths starting with `/` are absolute local paths
    /// - On native: Other paths are treated as HuggingFace hub identifiers
    #[cfg(not(target_arch = "wasm32"))]
    pub fn parse(path: &str) -> Self {
        if let Some(relative) = path.strip_prefix("index://") {
            TokenizerSource::IndexDirectory(relative.to_string())
        } else if path.starts_with('/') {
            TokenizerSource::LocalFile(path.to_string())
        } else {
            TokenizerSource::HuggingFace(path.to_string())
        }
    }

    /// Parse a tokenizer path string into a TokenizerSource (WASM version)
    ///
    /// On WASM, only index:// paths are supported
    #[cfg(target_arch = "wasm32")]
    pub fn parse(path: &str) -> Self {
        if let Some(relative) = path.strip_prefix("index://") {
            TokenizerSource::IndexDirectory(relative.to_string())
        } else {
            // On WASM, treat all paths as index-relative
            TokenizerSource::IndexDirectory(path.to_string())
        }
    }
}

impl HfTokenizer {
    /// Load a tokenizer from HuggingFace hub or local path (native only)
    ///
    /// Examples:
    /// - `"Alibaba-NLP/gte-Qwen2-1.5B-instruct"` - from HuggingFace hub
    /// - `"/path/to/tokenizer.json"` - from local file
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(name_or_path: &str) -> Result<Self> {
        let tokenizer = if name_or_path.contains('/') && !name_or_path.starts_with('/') {
            // Looks like a HuggingFace hub identifier
            Tokenizer::from_pretrained(name_or_path, None).map_err(|e| {
                Error::Tokenizer(format!(
                    "Failed to load tokenizer '{}': {}",
                    name_or_path, e
                ))
            })?
        } else {
            // Local file path
            Tokenizer::from_file(name_or_path).map_err(|e| {
                Error::Tokenizer(format!(
                    "Failed to load tokenizer from '{}': {}",
                    name_or_path, e
                ))
            })?
        };

        Ok(Self { tokenizer })
    }

    /// Load a tokenizer from bytes (e.g., read from Directory)
    ///
    /// This allows loading tokenizers from any Directory implementation,
    /// including remote storage like S3 or HTTP.
    /// This is the primary method for WASM targets.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(bytes).map_err(|e| {
            Error::Tokenizer(format!("Failed to parse tokenizer from bytes: {}", e))
        })?;
        Ok(Self { tokenizer })
    }

    /// Load from a TokenizerSource (native only)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_source(source: &TokenizerSource) -> Result<Self> {
        match source {
            TokenizerSource::HuggingFace(name) => {
                let tokenizer = Tokenizer::from_pretrained(name, None).map_err(|e| {
                    Error::Tokenizer(format!("Failed to load tokenizer '{}': {}", name, e))
                })?;
                Ok(Self { tokenizer })
            }
            TokenizerSource::LocalFile(path) => {
                let tokenizer = Tokenizer::from_file(path).map_err(|e| {
                    Error::Tokenizer(format!("Failed to load tokenizer from '{}': {}", path, e))
                })?;
                Ok(Self { tokenizer })
            }
            TokenizerSource::IndexDirectory(_) => {
                // For index directory, caller must use from_bytes with data read from Directory
                Err(Error::Tokenizer(
                    "IndexDirectory source requires using from_bytes with Directory read"
                        .to_string(),
                ))
            }
        }
    }

    /// Resolve a token ID back to its text representation
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Tokenize text and return token IDs
    ///
    /// Returns a vector of (token_id, count) pairs where count is the
    /// number of times each token appears in the text.
    pub fn tokenize(&self, text: &str) -> Result<Vec<(u32, u32)>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(format!("Tokenization failed: {}", e)))?;

        // Count token occurrences
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &id in encoding.get_ids() {
            *counts.entry(id).or_insert(0) += 1;
        }

        let result: Vec<(u32, u32)> = counts.into_iter().collect();
        let paired: Vec<_> = encoding
            .get_tokens()
            .iter()
            .zip(encoding.get_ids())
            .map(|(tok, id)| format!("({:?},{})", tok, id))
            .collect();
        debug!(
            "Tokenized query: text={:?} tokens=[{}] unique_count={}",
            text,
            paired.join(", "),
            result.len()
        );

        Ok(result)
    }

    /// Tokenize text and return unique token IDs (for weighting: one)
    pub fn tokenize_unique(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(format!("Tokenization failed: {}", e)))?;

        // Get unique token IDs
        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        ids.sort_unstable();
        ids.dedup();

        let paired: Vec<_> = encoding
            .get_tokens()
            .iter()
            .zip(encoding.get_ids())
            .map(|(tok, id)| format!("({:?},{})", tok, id))
            .collect();
        debug!(
            "Tokenized query (unique): text={:?} tokens=[{}] unique_count={}",
            text,
            paired.join(", "),
            ids.len()
        );

        Ok(ids)
    }
}

/// Global tokenizer cache for reuse across queries
#[cfg(feature = "native")]
pub struct TokenizerCache {
    cache: RwLock<HashMap<String, Arc<HfTokenizer>>>,
}

#[cfg(feature = "native")]
impl Default for TokenizerCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "native")]
impl TokenizerCache {
    /// Create a new tokenizer cache
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get or load a tokenizer
    pub fn get_or_load(&self, name_or_path: &str) -> Result<Arc<HfTokenizer>> {
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(tokenizer) = cache.get(name_or_path) {
                return Ok(Arc::clone(tokenizer));
            }
        }

        // Load and cache
        let tokenizer = Arc::new(HfTokenizer::load(name_or_path)?);
        {
            let mut cache = self.cache.write();
            cache.insert(name_or_path.to_string(), Arc::clone(&tokenizer));
        }

        Ok(tokenizer)
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
    }
}

/// Global tokenizer cache instance
#[cfg(feature = "native")]
static TOKENIZER_CACHE: std::sync::OnceLock<TokenizerCache> = std::sync::OnceLock::new();

/// Get the global tokenizer cache
#[cfg(feature = "native")]
pub fn tokenizer_cache() -> &'static TokenizerCache {
    TOKENIZER_CACHE.get_or_init(TokenizerCache::new)
}

#[cfg(test)]
#[cfg(feature = "native")]
mod tests {
    use super::*;

    // Note: These tests require network access to download tokenizers
    // They are ignored by default to avoid CI issues

    #[test]
    #[ignore]
    fn test_load_tokenizer_from_hub() {
        let tokenizer = HfTokenizer::load("bert-base-uncased").unwrap();
        let tokens = tokenizer.tokenize("hello world").unwrap();
        assert!(!tokens.is_empty());
    }

    #[test]
    #[ignore]
    fn test_tokenize_unique() {
        let tokenizer = HfTokenizer::load("bert-base-uncased").unwrap();
        let ids = tokenizer.tokenize_unique("the quick brown fox").unwrap();
        // Should have unique tokens
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(ids.len(), sorted.len());
    }

    #[test]
    fn test_tokenizer_cache() {
        let cache = TokenizerCache::new();
        // Just test that the cache structure works
        assert!(cache.cache.read().is_empty());
    }
}
