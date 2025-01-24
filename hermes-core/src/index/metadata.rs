//! Unified index metadata - segments list + vector index state
//!
//! This module manages all index-level metadata in a single `metadata.json` file:
//! - List of committed segments
//! - Vector index state per field (Flat/Built)
//! - Trained centroids/codebooks paths
//!
//! The workflow is:
//! 1. During accumulation: segments store Flat vectors, state is Flat
//! 2. When threshold crossed: train ONCE, update state to Built
//! 3. On index open: load metadata, skip re-training if already built

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::dsl::VectorIndexType;
use crate::error::{Error, Result};

/// Metadata file name at index level
pub const INDEX_META_FILENAME: &str = "metadata.json";

/// State of vector index for a field
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum VectorIndexState {
    /// Accumulating vectors - using Flat (brute-force) search
    #[default]
    Flat,
    /// Index structures built - using ANN search
    Built {
        /// Total vector count when training happened
        vector_count: usize,
        /// Number of clusters used
        num_clusters: usize,
    },
}

/// Per-field vector index metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldVectorMeta {
    /// Field ID
    pub field_id: u32,
    /// Configured index type (target type when built)
    pub index_type: VectorIndexType,
    /// Current state
    pub state: VectorIndexState,
    /// Path to centroids file (relative to index dir)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub centroids_file: Option<String>,
    /// Path to codebook file (relative to index dir, for ScaNN)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codebook_file: Option<String>,
}

/// Unified index metadata - replaces segments.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Version for compatibility
    pub version: u32,
    /// List of committed segment IDs (hex strings)
    pub segments: Vec<String>,
    /// Per-field vector index metadata
    #[serde(default)]
    pub vector_fields: HashMap<u32, FieldVectorMeta>,
    /// Total vectors across all segments (updated on commit)
    #[serde(default)]
    pub total_vectors: usize,
}

impl Default for IndexMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexMetadata {
    /// Create empty metadata
    pub fn new() -> Self {
        Self {
            version: 1,
            segments: Vec::new(),
            vector_fields: HashMap::new(),
            total_vectors: 0,
        }
    }

    /// Create from existing segments list (migration from old format)
    pub fn from_segments(segments: Vec<String>) -> Self {
        Self {
            version: 1,
            segments,
            vector_fields: HashMap::new(),
            total_vectors: 0,
        }
    }

    /// Check if a field has been built
    pub fn is_field_built(&self, field_id: u32) -> bool {
        self.vector_fields
            .get(&field_id)
            .map(|f| matches!(f.state, VectorIndexState::Built { .. }))
            .unwrap_or(false)
    }

    /// Get field metadata
    pub fn get_field_meta(&self, field_id: u32) -> Option<&FieldVectorMeta> {
        self.vector_fields.get(&field_id)
    }

    /// Initialize field metadata (called when field is first seen)
    pub fn init_field(&mut self, field_id: u32, index_type: VectorIndexType) {
        self.vector_fields
            .entry(field_id)
            .or_insert(FieldVectorMeta {
                field_id,
                index_type,
                state: VectorIndexState::Flat,
                centroids_file: None,
                codebook_file: None,
            });
    }

    /// Mark field as built with trained structures
    pub fn mark_field_built(
        &mut self,
        field_id: u32,
        vector_count: usize,
        num_clusters: usize,
        centroids_file: String,
        codebook_file: Option<String>,
    ) {
        if let Some(field) = self.vector_fields.get_mut(&field_id) {
            field.state = VectorIndexState::Built {
                vector_count,
                num_clusters,
            };
            field.centroids_file = Some(centroids_file);
            field.codebook_file = codebook_file;
        }
    }

    /// Check if field should be built based on threshold
    pub fn should_build_field(&self, field_id: u32, threshold: usize) -> bool {
        // Don't build if already built
        if self.is_field_built(field_id) {
            return false;
        }
        // Build if we have enough vectors
        self.total_vectors >= threshold
    }

    /// Add a segment
    pub fn add_segment(&mut self, segment_id: String) {
        if !self.segments.contains(&segment_id) {
            self.segments.push(segment_id);
        }
    }

    /// Remove segments
    pub fn remove_segments(&mut self, to_remove: &[String]) {
        self.segments.retain(|s| !to_remove.contains(s));
    }

    /// Load from directory (with migration from old segments.json)
    pub async fn load<D: crate::directories::Directory>(dir: &D) -> Result<Self> {
        let path = Path::new(INDEX_META_FILENAME);
        match dir.open_read(path).await {
            Ok(slice) => {
                let bytes = slice.read_bytes().await?;
                serde_json::from_slice(bytes.as_slice())
                    .map_err(|e| Error::Serialization(e.to_string()))
            }
            Err(_) => {
                // Try migration from old segments.json format
                let old_path = Path::new("segments.json");
                if let Ok(slice) = dir.open_read(old_path).await
                    && let Ok(bytes) = slice.read_bytes().await
                    && let Ok(segments) = serde_json::from_slice::<Vec<String>>(bytes.as_slice())
                {
                    Ok(Self::from_segments(segments))
                } else {
                    Ok(Self::new())
                }
            }
        }
    }

    /// Save to directory
    pub async fn save<D: crate::directories::DirectoryWriter>(&self, dir: &D) -> Result<()> {
        let path = Path::new(INDEX_META_FILENAME);
        let bytes =
            serde_json::to_vec_pretty(self).map_err(|e| Error::Serialization(e.to_string()))?;
        dir.write(path, &bytes).await.map_err(Error::Io)
    }

    /// Load trained centroids and codebooks from index-level files
    ///
    /// Returns (centroids_map, codebooks_map) for fields that are Built
    pub async fn load_trained_structures<D: crate::directories::Directory>(
        &self,
        dir: &D,
    ) -> (
        rustc_hash::FxHashMap<u32, std::sync::Arc<crate::structures::CoarseCentroids>>,
        rustc_hash::FxHashMap<u32, std::sync::Arc<crate::structures::PQCodebook>>,
    ) {
        use std::sync::Arc;

        let mut centroids = rustc_hash::FxHashMap::default();
        let mut codebooks = rustc_hash::FxHashMap::default();

        for (field_id, field_meta) in &self.vector_fields {
            if !matches!(field_meta.state, VectorIndexState::Built { .. }) {
                continue;
            }

            // Load centroids
            if let Some(ref file) = field_meta.centroids_file
                && let Ok(slice) = dir.open_read(Path::new(file)).await
                && let Ok(bytes) = slice.read_bytes().await
                && let Ok(c) =
                    serde_json::from_slice::<crate::structures::CoarseCentroids>(bytes.as_slice())
            {
                centroids.insert(*field_id, Arc::new(c));
            }

            // Load codebook (for ScaNN)
            if let Some(ref file) = field_meta.codebook_file
                && let Ok(slice) = dir.open_read(Path::new(file)).await
                && let Ok(bytes) = slice.read_bytes().await
                && let Ok(c) =
                    serde_json::from_slice::<crate::structures::PQCodebook>(bytes.as_slice())
            {
                codebooks.insert(*field_id, Arc::new(c));
            }
        }

        (centroids, codebooks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_init() {
        let mut meta = IndexMetadata::new();
        assert_eq!(meta.total_vectors, 0);
        assert!(meta.segments.is_empty());
        assert!(!meta.is_field_built(0));

        meta.init_field(0, VectorIndexType::IvfRaBitQ);
        assert!(!meta.is_field_built(0));
        assert!(meta.vector_fields.contains_key(&0));
    }

    #[test]
    fn test_metadata_segments() {
        let mut meta = IndexMetadata::new();
        meta.add_segment("abc123".to_string());
        meta.add_segment("def456".to_string());
        assert_eq!(meta.segments.len(), 2);

        // No duplicates
        meta.add_segment("abc123".to_string());
        assert_eq!(meta.segments.len(), 2);

        meta.remove_segments(&["abc123".to_string()]);
        assert_eq!(meta.segments.len(), 1);
        assert_eq!(meta.segments[0], "def456");
    }

    #[test]
    fn test_mark_field_built() {
        let mut meta = IndexMetadata::new();
        meta.init_field(0, VectorIndexType::IvfRaBitQ);
        meta.total_vectors = 10000;

        assert!(!meta.is_field_built(0));

        meta.mark_field_built(0, 10000, 256, "field_0_centroids.bin".to_string(), None);

        assert!(meta.is_field_built(0));
        let field = meta.get_field_meta(0).unwrap();
        assert_eq!(
            field.centroids_file.as_deref(),
            Some("field_0_centroids.bin")
        );
    }

    #[test]
    fn test_should_build_field() {
        let mut meta = IndexMetadata::new();
        meta.init_field(0, VectorIndexType::IvfRaBitQ);

        // Below threshold
        meta.total_vectors = 500;
        assert!(!meta.should_build_field(0, 1000));

        // Above threshold
        meta.total_vectors = 1500;
        assert!(meta.should_build_field(0, 1000));

        // Already built - should not build again
        meta.mark_field_built(0, 1500, 256, "centroids.bin".to_string(), None);
        assert!(!meta.should_build_field(0, 1000));
    }

    #[test]
    fn test_serialization() {
        let mut meta = IndexMetadata::new();
        meta.add_segment("seg1".to_string());
        meta.init_field(0, VectorIndexType::IvfRaBitQ);
        meta.total_vectors = 5000;

        let json = serde_json::to_string_pretty(&meta).unwrap();
        let loaded: IndexMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.segments, meta.segments);
        assert_eq!(loaded.total_vectors, meta.total_vectors);
        assert!(loaded.vector_fields.contains_key(&0));
    }
}
