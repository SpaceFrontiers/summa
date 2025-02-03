//! Pre-computed IDF weights from model's `idf.json`
//!
//! Neural sparse models (e.g., opensearch-neural-sparse-encoding-multilingual-v1)
//! ship `idf.json` with IDF values calibrated during training. Using these weights
//! instead of index-derived IDF produces correct rankings for doc-only models.
//!
//! The idf.json format maps **token strings** to IDF values:
//! `{"[PAD]": 0.607, "hemoglobin": 8.12, "##ing": 1.05, ...}`
//!
//! At load time, we resolve token strings to numeric IDs via the model's tokenizer,
//! then store as a flat `Vec<f32>` for O(1) lookup by token_id.

#[cfg(feature = "native")]
use std::collections::HashMap;
#[cfg(feature = "native")]
use std::sync::Arc;

#[cfg(feature = "native")]
use log::{debug, warn};
#[cfg(feature = "native")]
use parking_lot::RwLock;

#[cfg(feature = "native")]
use crate::Result;
#[cfg(feature = "native")]
use crate::error::Error;

/// Pre-computed IDF weights indexed by token_id
///
/// Stored as a flat `Vec<f32>` for O(1) lookup by token_id.
/// For mBERT's 105K vocab this uses ~420KB of memory.
#[cfg(feature = "native")]
pub struct IdfWeights {
    weights: Vec<f32>,
}

#[cfg(feature = "native")]
impl IdfWeights {
    /// Get the IDF weight for a token_id
    ///
    /// Returns 1.0 for out-of-range token_ids (neutral weight).
    #[inline]
    pub fn get(&self, token_id: u32) -> f32 {
        self.weights.get(token_id as usize).copied().unwrap_or(1.0)
    }

    /// Load IDF weights from a JSON object, resolving token strings to IDs
    /// via the provided tokenizer.
    ///
    /// The idf.json maps token strings → IDF values. We use `token_to_id`
    /// to convert each key to a numeric token ID for O(1) lookup.
    fn from_json_with_tokenizer(
        json_bytes: &[u8],
        tokenizer: &tokenizers::Tokenizer,
    ) -> Result<Self> {
        let map: HashMap<String, f64> = serde_json::from_slice(json_bytes)
            .map_err(|e| Error::Tokenizer(format!("Failed to parse idf.json: {}", e)))?;

        if map.is_empty() {
            return Err(Error::Tokenizer("idf.json is empty".to_string()));
        }

        // Resolve token strings to IDs and find max ID
        let mut resolved: Vec<(u32, f32)> = Vec::with_capacity(map.len());
        let mut missed = 0u32;
        for (token_str, value) in &map {
            if let Some(id) = tokenizer.token_to_id(token_str) {
                resolved.push((id, *value as f32));
            } else {
                missed += 1;
            }
        }

        if resolved.is_empty() {
            return Err(Error::Tokenizer(
                "idf.json: no tokens could be resolved to IDs via tokenizer".to_string(),
            ));
        }

        let max_id = resolved.iter().map(|(id, _)| *id).max().unwrap();

        // Initialize with 1.0 (neutral weight) for unmapped tokens
        let mut weights = vec![1.0f32; (max_id + 1) as usize];
        for &(id, value) in &resolved {
            weights[id as usize] = value;
        }

        debug!(
            "Loaded {} IDF weights via tokenizer (vec size: {}, unresolved: {})",
            resolved.len(),
            weights.len(),
            missed,
        );

        Ok(Self { weights })
    }
}

/// Global cache for IDF weights, keyed by model name
#[cfg(feature = "native")]
pub struct IdfWeightsCache {
    cache: RwLock<HashMap<String, Arc<IdfWeights>>>,
}

#[cfg(feature = "native")]
impl Default for IdfWeightsCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "native")]
impl IdfWeightsCache {
    /// Create a new IDF weights cache
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get or load IDF weights for a model
    ///
    /// Downloads `idf.json` from the HuggingFace model repo if not cached.
    /// Returns `None` if `idf.json` is not available (graceful fallback).
    pub fn get_or_load(&self, model_name: &str) -> Option<Arc<IdfWeights>> {
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(weights) = cache.get(model_name) {
                return Some(Arc::clone(weights));
            }
        }

        // Try to load from HF hub
        match self.download_and_parse(model_name) {
            Ok(weights) => {
                let weights = Arc::new(weights);
                let mut cache = self.cache.write();
                cache.insert(model_name.to_string(), Arc::clone(&weights));
                Some(weights)
            }
            Err(e) => {
                warn!(
                    "Could not load idf.json for model '{}': {}. Falling back to index-derived IDF.",
                    model_name, e
                );
                None
            }
        }
    }

    /// Download idf.json from HuggingFace hub and parse it
    ///
    /// Also loads the model's tokenizer to resolve token strings → IDs.
    fn download_and_parse(&self, model_name: &str) -> Result<IdfWeights> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| Error::Tokenizer(format!("Failed to create HF hub API: {}", e)))?;
        let repo = api.model(model_name.to_string());
        let idf_path = repo.get("idf.json").map_err(|e| {
            Error::Tokenizer(format!(
                "Failed to download idf.json from '{}': {}",
                model_name, e
            ))
        })?;

        debug!(
            "Downloaded idf.json from '{}' to {:?}",
            model_name, idf_path
        );

        let json_bytes = std::fs::read(&idf_path).map_err(|e| {
            Error::Tokenizer(format!("Failed to read idf.json at {:?}: {}", idf_path, e))
        })?;

        // Load tokenizer to resolve token strings → numeric IDs
        let tokenizer = super::tokenizer_cache().get_or_load(model_name)?;

        IdfWeights::from_json_with_tokenizer(&json_bytes, &tokenizer.tokenizer)
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
    }
}

/// Global IDF weights cache instance
#[cfg(feature = "native")]
static IDF_WEIGHTS_CACHE: std::sync::OnceLock<IdfWeightsCache> = std::sync::OnceLock::new();

/// Get the global IDF weights cache
#[cfg(feature = "native")]
pub fn idf_weights_cache() -> &'static IdfWeightsCache {
    IDF_WEIGHTS_CACHE.get_or_init(IdfWeightsCache::new)
}

#[cfg(test)]
#[cfg(feature = "native")]
mod tests {
    use super::*;

    /// Build a minimal tokenizer with a known vocab for testing
    fn test_tokenizer() -> tokenizers::Tokenizer {
        use tokenizers::models::wordpiece::WordPiece;
        let wp = WordPiece::builder()
            .vocab([
                ("[UNK]".to_string(), 0),
                ("hello".to_string(), 1),
                ("world".to_string(), 2),
                ("foo".to_string(), 5),
                ("bar".to_string(), 100),
            ])
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        tokenizers::Tokenizer::new(wp)
    }

    #[test]
    fn test_idf_weights_from_json_with_tokenizer() {
        let json = br#"{"hello": 1.5, "world": 2.0, "foo": 0.5, "bar": 3.0}"#;
        let tokenizer = test_tokenizer();
        let weights = IdfWeights::from_json_with_tokenizer(json, &tokenizer).unwrap();

        // hello=1, world=2, foo=5, bar=100
        assert!((weights.get(1) - 1.5).abs() < f32::EPSILON);
        assert!((weights.get(2) - 2.0).abs() < f32::EPSILON);
        assert!((weights.get(5) - 0.5).abs() < f32::EPSILON);
        assert!((weights.get(100) - 3.0).abs() < f32::EPSILON);

        // Unmapped tokens get 1.0
        assert!((weights.get(3) - 1.0).abs() < f32::EPSILON);
        assert!((weights.get(50) - 1.0).abs() < f32::EPSILON);

        // Out-of-range tokens get 1.0
        assert!((weights.get(999) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_idf_weights_unresolvable_tokens_skipped() {
        // "unknown_xyz" is not in the vocab, should be skipped
        let json = br#"{"hello": 1.5, "unknown_xyz": 9.9}"#;
        let tokenizer = test_tokenizer();
        let weights = IdfWeights::from_json_with_tokenizer(json, &tokenizer).unwrap();

        assert!((weights.get(1) - 1.5).abs() < f32::EPSILON); // hello resolved
    }

    #[test]
    fn test_idf_weights_empty_json() {
        let json = br#"{}"#;
        let tokenizer = test_tokenizer();
        assert!(IdfWeights::from_json_with_tokenizer(json, &tokenizer).is_err());
    }

    #[test]
    fn test_idf_weights_invalid_json() {
        let json = br#"not json"#;
        let tokenizer = test_tokenizer();
        assert!(IdfWeights::from_json_with_tokenizer(json, &tokenizer).is_err());
    }

    #[test]
    fn test_idf_weights_cache_structure() {
        let cache = IdfWeightsCache::new();
        assert!(cache.cache.read().is_empty());
    }

    #[test]
    fn test_idf_weights_cache_miss_graceful() {
        let cache = IdfWeightsCache::new();
        // Non-existent model should return None gracefully
        let result = cache.get_or_load("nonexistent-model-xyz-12345");
        assert!(result.is_none());
    }
}
