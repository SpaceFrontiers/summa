//! Pre-computed IDF weights from model's `idf.json`
//!
//! Neural sparse models (e.g., opensearch-neural-sparse-encoding-multilingual-v1)
//! ship `idf.json` with IDF values calibrated during training. Using these weights
//! instead of index-derived IDF produces correct rankings for doc-only models.

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

    /// Load IDF weights from a JSON object `{"token_id_str": float, ...}`
    fn from_json(json_bytes: &[u8]) -> Result<Self> {
        let map: HashMap<String, f64> = serde_json::from_slice(json_bytes)
            .map_err(|e| Error::Tokenizer(format!("Failed to parse idf.json: {}", e)))?;

        if map.is_empty() {
            return Err(Error::Tokenizer("idf.json is empty".to_string()));
        }

        // Find max token_id to size the vector
        let max_id = map
            .keys()
            .filter_map(|k| k.parse::<u32>().ok())
            .max()
            .ok_or_else(|| Error::Tokenizer("idf.json contains no valid token IDs".to_string()))?;

        // Initialize with 1.0 (neutral weight) for unmapped tokens
        let mut weights = vec![1.0f32; (max_id + 1) as usize];
        let mut count = 0;
        for (key, value) in &map {
            if let Ok(id) = key.parse::<u32>() {
                weights[id as usize] = *value as f32;
                count += 1;
            }
        }

        debug!("Loaded {} IDF weights (vec size: {})", count, weights.len());

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

        IdfWeights::from_json(&json_bytes)
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

    #[test]
    fn test_idf_weights_from_json() {
        let json = br#"{"0": 1.5, "1": 2.0, "5": 0.5, "100": 3.0}"#;
        let weights = IdfWeights::from_json(json).unwrap();

        assert!((weights.get(0) - 1.5).abs() < f32::EPSILON);
        assert!((weights.get(1) - 2.0).abs() < f32::EPSILON);
        assert!((weights.get(5) - 0.5).abs() < f32::EPSILON);
        assert!((weights.get(100) - 3.0).abs() < f32::EPSILON);

        // Unmapped tokens get 1.0
        assert!((weights.get(2) - 1.0).abs() < f32::EPSILON);
        assert!((weights.get(50) - 1.0).abs() < f32::EPSILON);

        // Out-of-range tokens get 1.0
        assert!((weights.get(999) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_idf_weights_empty_json() {
        let json = br#"{}"#;
        assert!(IdfWeights::from_json(json).is_err());
    }

    #[test]
    fn test_idf_weights_invalid_json() {
        let json = br#"not json"#;
        assert!(IdfWeights::from_json(json).is_err());
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
