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
use crate::schema::Schema;

/// Metadata file name at index level
pub const INDEX_META_FILENAME: &str = "metadata.json";
/// Temp file for atomic writes (write here, then rename to INDEX_META_FILENAME)
const INDEX_META_TMP_FILENAME: &str = "metadata.json.tmp";

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

/// Per-segment metadata stored in index metadata
/// This allows merge decisions without loading segment files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMetaInfo {
    /// Number of documents in this segment
    pub num_docs: u32,
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

/// Unified index metadata - single source of truth for index state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Version for compatibility
    pub version: u32,
    /// Index schema
    pub schema: Schema,
    /// Segment metadata: segment_id -> info (doc count, etc.)
    /// Using HashMap allows O(1) lookup and stores doc counts for merge decisions
    #[serde(default)]
    pub segment_metas: HashMap<String, SegmentMetaInfo>,
    /// Per-field vector index metadata
    #[serde(default)]
    pub vector_fields: HashMap<u32, FieldVectorMeta>,
    /// Total vectors across all segments (updated on commit)
    #[serde(default)]
    pub total_vectors: usize,
}

impl IndexMetadata {
    /// Create new metadata with schema
    pub fn new(schema: Schema) -> Self {
        Self {
            version: 1,
            schema,
            segment_metas: HashMap::new(),
            vector_fields: HashMap::new(),
            total_vectors: 0,
        }
    }

    /// Get segment IDs as a sorted Vec (deterministic ordering for doc_id_offset assignment)
    pub fn segment_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.segment_metas.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Add or update a segment with its doc count
    pub fn add_segment(&mut self, segment_id: String, num_docs: u32) {
        self.segment_metas
            .insert(segment_id, SegmentMetaInfo { num_docs });
    }

    /// Remove a segment
    pub fn remove_segment(&mut self, segment_id: &str) {
        self.segment_metas.remove(segment_id);
    }

    /// Check if segment exists
    pub fn has_segment(&self, segment_id: &str) -> bool {
        self.segment_metas.contains_key(segment_id)
    }

    /// Get segment doc count
    pub fn segment_doc_count(&self, segment_id: &str) -> Option<u32> {
        self.segment_metas.get(segment_id).map(|m| m.num_docs)
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

    /// Load from directory
    ///
    /// If `metadata.json` is missing but `metadata.json.tmp` exists (crash
    /// between write and rename), recovers from the temp file.
    pub async fn load<D: crate::directories::Directory>(dir: &D) -> Result<Self> {
        let path = Path::new(INDEX_META_FILENAME);
        match dir.open_read(path).await {
            Ok(slice) => {
                let bytes = slice.read_bytes().await?;
                serde_json::from_slice(bytes.as_slice())
                    .map_err(|e| Error::Serialization(e.to_string()))
            }
            Err(_) => {
                // Try recovering from temp file (crash between write and rename)
                let tmp_path = Path::new(INDEX_META_TMP_FILENAME);
                let slice = dir.open_read(tmp_path).await?;
                let bytes = slice.read_bytes().await?;
                let meta: Self = serde_json::from_slice(bytes.as_slice())
                    .map_err(|e| Error::Serialization(e.to_string()))?;
                log::warn!("Recovered metadata from temp file (previous crash during save)");
                Ok(meta)
            }
        }
    }

    /// Save to directory (atomic: write temp file, then rename)
    ///
    /// Uses write-then-rename so a crash mid-write won't corrupt the
    /// existing metadata file. On POSIX, rename is atomic.
    pub async fn save<D: crate::directories::DirectoryWriter>(&self, dir: &D) -> Result<()> {
        let tmp_path = Path::new(INDEX_META_TMP_FILENAME);
        let final_path = Path::new(INDEX_META_FILENAME);
        let bytes =
            serde_json::to_vec_pretty(self).map_err(|e| Error::Serialization(e.to_string()))?;
        dir.write(tmp_path, &bytes).await.map_err(Error::Io)?;
        dir.rename(tmp_path, final_path).await.map_err(Error::Io)?;
        Ok(())
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
                log::debug!("[trained] field {} not in Built state, skipping", field_id);
                continue;
            }

            // Load centroids
            match &field_meta.centroids_file {
                None => {
                    log::warn!(
                        "[trained] field {} is Built but centroids_file is None",
                        field_id
                    );
                }
                Some(file) => match dir.open_read(Path::new(file)).await {
                    Err(e) => {
                        log::warn!(
                            "[trained] field {} centroids file '{}' open failed: {}",
                            field_id,
                            file,
                            e
                        );
                    }
                    Ok(slice) => match slice.read_bytes().await {
                        Err(e) => {
                            log::warn!(
                                "[trained] field {} centroids file '{}' read failed: {}",
                                field_id,
                                file,
                                e
                            );
                        }
                        Ok(bytes) => {
                            match serde_json::from_slice::<crate::structures::CoarseCentroids>(
                                bytes.as_slice(),
                            ) {
                                Ok(c) => {
                                    log::debug!(
                                        "[trained] field {} loaded centroids ({} clusters)",
                                        field_id,
                                        c.num_clusters
                                    );
                                    centroids.insert(*field_id, Arc::new(c));
                                }
                                Err(e) => {
                                    log::warn!(
                                        "[trained] field {} centroids deserialize failed: {}",
                                        field_id,
                                        e
                                    );
                                }
                            }
                        }
                    },
                },
            }

            // Load codebook (for ScaNN)
            match &field_meta.codebook_file {
                None => {} // Not all index types have codebooks
                Some(file) => match dir.open_read(Path::new(file)).await {
                    Err(e) => {
                        log::warn!(
                            "[trained] field {} codebook file '{}' open failed: {}",
                            field_id,
                            file,
                            e
                        );
                    }
                    Ok(slice) => match slice.read_bytes().await {
                        Err(e) => {
                            log::warn!(
                                "[trained] field {} codebook file '{}' read failed: {}",
                                field_id,
                                file,
                                e
                            );
                        }
                        Ok(bytes) => {
                            match serde_json::from_slice::<crate::structures::PQCodebook>(
                                bytes.as_slice(),
                            ) {
                                Ok(c) => {
                                    log::debug!("[trained] field {} loaded codebook", field_id);
                                    codebooks.insert(*field_id, Arc::new(c));
                                }
                                Err(e) => {
                                    log::warn!(
                                        "[trained] field {} codebook deserialize failed: {}",
                                        field_id,
                                        e
                                    );
                                }
                            }
                        }
                    },
                },
            }
        }

        // Fallback: if vector_fields didn't yield centroids, scan schema for
        // dense vector fields and probe well-known file paths on disk.
        // This handles cases where vector_fields is empty (e.g. old metadata
        // format) but centroids files exist from a previous ANN build.
        if centroids.is_empty() {
            for (field, entry) in self.schema.fields() {
                let config = match &entry.dense_vector_config {
                    Some(c) => c,
                    None => continue,
                };
                let field_id = field.0;

                // Skip if not an ANN type
                if !matches!(
                    config.index_type,
                    VectorIndexType::IvfRaBitQ | VectorIndexType::ScaNN
                ) {
                    continue;
                }

                let centroids_file = format!("field_{}_centroids.bin", field_id);
                if let Ok(slice) = dir.open_read(Path::new(&centroids_file)).await
                    && let Ok(bytes) = slice.read_bytes().await
                {
                    match serde_json::from_slice::<crate::structures::CoarseCentroids>(
                        bytes.as_slice(),
                    ) {
                        Ok(c) => {
                            log::info!(
                                "[trained] field {} loaded centroids from disk fallback ({} clusters)",
                                field_id,
                                c.num_clusters
                            );
                            centroids.insert(field_id, Arc::new(c));
                        }
                        Err(e) => {
                            log::warn!(
                                "[trained] field {} centroids fallback deserialize failed: {}",
                                field_id,
                                e
                            );
                        }
                    }
                }

                // Try codebook (for ScaNN)
                if matches!(config.index_type, VectorIndexType::ScaNN) {
                    let codebook_file = format!("field_{}_codebook.bin", field_id);
                    if let Ok(slice) = dir.open_read(Path::new(&codebook_file)).await
                        && let Ok(bytes) = slice.read_bytes().await
                    {
                        match serde_json::from_slice::<crate::structures::PQCodebook>(
                            bytes.as_slice(),
                        ) {
                            Ok(c) => {
                                log::info!(
                                    "[trained] field {} loaded codebook from disk fallback",
                                    field_id
                                );
                                codebooks.insert(field_id, Arc::new(c));
                            }
                            Err(e) => {
                                log::warn!(
                                    "[trained] field {} codebook fallback deserialize failed: {}",
                                    field_id,
                                    e
                                );
                            }
                        }
                    }
                }
            }
        }

        (centroids, codebooks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_schema() -> Schema {
        Schema::default()
    }

    #[test]
    fn test_metadata_init() {
        let mut meta = IndexMetadata::new(test_schema());
        assert_eq!(meta.total_vectors, 0);
        assert!(meta.segment_metas.is_empty());
        assert!(!meta.is_field_built(0));

        meta.init_field(0, VectorIndexType::IvfRaBitQ);
        assert!(!meta.is_field_built(0));
        assert!(meta.vector_fields.contains_key(&0));
    }

    #[test]
    fn test_metadata_segments() {
        let mut meta = IndexMetadata::new(test_schema());
        meta.add_segment("abc123".to_string(), 50);
        meta.add_segment("def456".to_string(), 100);
        assert_eq!(meta.segment_metas.len(), 2);
        assert_eq!(meta.segment_doc_count("abc123"), Some(50));
        assert_eq!(meta.segment_doc_count("def456"), Some(100));

        // Overwrites existing
        meta.add_segment("abc123".to_string(), 75);
        assert_eq!(meta.segment_metas.len(), 2);
        assert_eq!(meta.segment_doc_count("abc123"), Some(75));

        meta.remove_segment("abc123");
        assert_eq!(meta.segment_metas.len(), 1);
        assert!(meta.has_segment("def456"));
        assert!(!meta.has_segment("abc123"));
    }

    #[test]
    fn test_mark_field_built() {
        let mut meta = IndexMetadata::new(test_schema());
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
        let mut meta = IndexMetadata::new(test_schema());
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
        let mut meta = IndexMetadata::new(test_schema());
        meta.add_segment("seg1".to_string(), 100);
        meta.init_field(0, VectorIndexType::IvfRaBitQ);
        meta.total_vectors = 5000;

        let json = serde_json::to_string_pretty(&meta).unwrap();
        let loaded: IndexMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.segment_ids().len(), meta.segment_ids().len());
        assert_eq!(loaded.segment_doc_count("seg1"), Some(100));
        assert_eq!(loaded.total_vectors, meta.total_vectors);
        assert!(loaded.vector_fields.contains_key(&0));
    }
}
