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
use std::io::Write;
use std::path::Path;

use crate::dsl::{Schema, VectorIndexType};
use crate::error::{Error, Result};

/// Metadata file name at index level
pub const INDEX_META_FILENAME: &str = "metadata.json";
/// Temp file for atomic writes (write here, then rename to INDEX_META_FILENAME)
const INDEX_META_TMP_FILENAME: &str = "metadata.json.tmp";

/// Index-level centroids/codebooks are deliberately bounded before they are
/// read or decoded. Besides limiting ordinary corruption damage, the matching
/// bincode limit prevents a tiny forged collection length from requesting an
/// effectively unbounded allocation.
pub(crate) const MAX_TRAINED_ARTIFACT_BYTES: usize = 512 * 1024 * 1024;

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

fn default_true() -> bool {
    true
}

/// Per-segment metadata stored in index metadata
/// This allows merge decisions without loading segment files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMetaInfo {
    /// Number of documents in this segment
    pub num_docs: u32,
    /// Parent segment IDs that were merged to produce this segment (empty for fresh segments)
    pub ancestors: Vec<String>,
    /// Merge generation: 0 for fresh segments, max(parent generations) + 1 for merged segments
    pub generation: u32,
    /// Whether this segment has been reordered via Recursive Graph Bisection (BP).
    /// Fresh segments and block-copy merges are not reordered. Only segments that have
    /// been explicitly reordered (via background optimizer or reorder command) are marked true.
    #[serde(default)]
    pub reordered: bool,
    /// Whether the last BP reorder pass ran to natural convergence. False when
    /// a wall-clock BP budget ended the pass early — the segment is ordered
    /// better than before, and a later warm-started pass can deepen it.
    /// Old metadata (field absent) deserializes as converged.
    #[serde(default = "default_true")]
    pub bp_converged: bool,
    /// Number of consecutive budget-exhausted BP rewrites in this segment's
    /// current reordered lineage. Carried across replacement IDs so the
    /// optimizer can impose a hard follow-up bound instead of rewriting forever.
    #[serde(default)]
    pub bp_unconverged_passes: u32,
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
    /// Aggregate vector count recorded by all built vector fields.
    ///
    /// The per-field `VectorIndexState::Built::vector_count` values are the
    /// source of truth. This cached aggregate is refreshed whenever a field is
    /// marked built, rather than being overwritten with whichever field was
    /// trained last.
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

    /// Get segment IDs as a sorted Vec (deterministic ordering)
    pub fn segment_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.segment_metas.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Add a fresh segment (gen=0, no ancestors, not reordered)
    pub fn add_segment(&mut self, segment_id: String, num_docs: u32) {
        self.segment_metas.insert(
            segment_id,
            SegmentMetaInfo {
                num_docs,
                ancestors: Vec::new(),
                generation: 0,
                reordered: false,
                bp_converged: true,
                bp_unconverged_passes: 0,
            },
        );
    }

    /// Add a merged segment with lineage info
    pub fn add_merged_segment(
        &mut self,
        segment_id: String,
        num_docs: u32,
        ancestors: Vec<String>,
        generation: u32,
        reordered: bool,
        bp_converged: bool,
    ) {
        self.add_segment_meta(
            segment_id,
            SegmentMetaInfo {
                num_docs,
                ancestors,
                generation,
                reordered,
                bp_converged,
                bp_unconverged_passes: 0,
            },
        );
    }

    /// Insert fully constructed lifecycle metadata. Merge/reorder code uses
    /// this to carry bounded BP lineage; ordinary callers use the safer
    /// constructors above, which start a fresh lineage.
    pub(crate) fn add_segment_meta(&mut self, segment_id: String, info: SegmentMetaInfo) {
        self.segment_metas.insert(segment_id, info);
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
            self.refresh_total_vectors();
        }
    }

    /// Refresh the cached aggregate from the authoritative per-field states.
    ///
    /// Saturation keeps this infallible metadata helper safe even if it is
    /// called after loading externally modified metadata with impossible
    /// counts.
    pub(crate) fn refresh_total_vectors(&mut self) {
        self.total_vectors = self
            .vector_fields
            .values()
            .filter_map(|field| match field.state {
                VectorIndexState::Built { vector_count, .. } => Some(vector_count),
                VectorIndexState::Flat => None,
            })
            .fold(0usize, usize::saturating_add);
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
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Try recovering from temp file (crash between write and rename)
                let tmp_path = Path::new(INDEX_META_TMP_FILENAME);
                let slice = dir.open_read(tmp_path).await?;
                let bytes = slice.read_bytes().await?;
                let meta: Self = serde_json::from_slice(bytes.as_slice())
                    .map_err(|e| Error::Serialization(e.to_string()))?;
                log::warn!("Recovered metadata from temp file (previous crash during save)");
                Ok(meta)
            }
            Err(e) => Err(Error::Io(e)),
        }
    }

    /// Save to directory (atomic: write temp file, then rename)
    ///
    /// Uses write-then-rename so a crash mid-write won't corrupt the
    /// existing metadata file. On POSIX, rename is atomic.
    pub async fn save<D: crate::directories::DirectoryWriter>(&self, dir: &D) -> Result<()> {
        let bytes = self.serialize_to_bytes()?;
        Self::save_bytes(dir, &bytes).await
    }

    /// Serialize metadata to bytes (cheap, no I/O).
    /// Useful when you need to release a lock before doing disk I/O.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(self).map_err(|e| Error::Serialization(e.to_string()))
    }

    /// Write pre-serialized metadata bytes to directory (atomic rename + fsync).
    ///
    /// The fsync ensures durability: without it, a power failure after rename
    /// could lose the metadata update on systems with volatile write caches.
    pub async fn save_bytes<D: crate::directories::DirectoryWriter>(
        dir: &D,
        bytes: &[u8],
    ) -> Result<()> {
        let tmp_path = Path::new(INDEX_META_TMP_FILENAME);
        let final_path = Path::new(INDEX_META_FILENAME);
        // Metadata is tiny, but `DirectoryWriter::write` does not guarantee
        // the file contents themselves are fsynced. Finish the streaming
        // writer first (filesystem implementations call `File::sync_all`),
        // then atomically publish the durable temp file by rename.
        let mut writer = dir.streaming_writer(tmp_path).await.map_err(Error::Io)?;
        writer.write_all(bytes).map_err(Error::Io)?;
        writer.finish().map_err(Error::Io)?;
        // Rename is the logical commit point: after it succeeds, readers can
        // observe the new generation and callers must publish the matching
        // in-memory/tracker state. Directory fsync only strengthens crash
        // durability. It cannot safely turn an already-visible rename into a
        // reported pre-commit failure, because cleanup could then delete files
        // referenced by the metadata now on disk.
        dir.rename(tmp_path, final_path).await.map_err(Error::Io)?;
        if let Err(error) = dir.sync().await {
            log::error!(
                "[metadata] directory fsync failed after committed rename: {}. \
                 Continuing with the renamed generation; crash durability is not guaranteed",
                error,
            );
        }
        Ok(())
    }

    /// Compatibility loader for callers that only have the persisted field
    /// map. Invalid/incomplete state is logged and returns `None` rather than a
    /// partial set.
    ///
    /// Index open/build paths use the fallible, schema-aware
    /// [`Self::try_load_trained_from_fields`] method below.
    pub async fn load_trained_from_fields<D: crate::directories::Directory>(
        vector_fields: &HashMap<u32, FieldVectorMeta>,
        dir: &D,
    ) -> Option<crate::segment::TrainedVectorStructures> {
        match Self::load_trained_from_fields_impl(vector_fields, None, dir).await {
            Ok(trained) => trained,
            Err(error) => {
                log::error!("[trained] refusing incomplete/corrupt artifact set: {error}");
                None
            }
        }
    }

    /// Fallible schema-aware loader used for lifecycle publication.
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub(crate) async fn try_load_trained_from_fields<D: crate::directories::Directory>(
        vector_fields: &HashMap<u32, FieldVectorMeta>,
        schema: &Schema,
        dir: &D,
    ) -> Result<Option<crate::segment::TrainedVectorStructures>> {
        Self::load_trained_from_fields_impl(vector_fields, Some(schema), dir).await
    }

    /// Load and validate the complete trained-artifact set described by a
    /// `vector_fields` snapshot.
    ///
    /// This is intentionally all-or-nothing. A `Built` field is a durable
    /// promise that every artifact required by its configured index exists and
    /// is compatible with the schema. Returning a partial map would let some
    /// segment builders publish ANN data while another field was silently
    /// unusable, and would make the same index behave differently after a
    /// restart.
    async fn load_trained_from_fields_impl<D: crate::directories::Directory>(
        vector_fields: &HashMap<u32, FieldVectorMeta>,
        schema: Option<&Schema>,
        dir: &D,
    ) -> Result<Option<crate::segment::TrainedVectorStructures>> {
        use std::sync::Arc;

        let mut centroids = rustc_hash::FxHashMap::default();
        let mut codebooks = rustc_hash::FxHashMap::default();
        let mut built_fields: Vec<_> = vector_fields
            .iter()
            .filter(|(_, meta)| matches!(meta.state, VectorIndexState::Built { .. }))
            .collect();
        built_fields.sort_unstable_by_key(|(field_id, _)| **field_id);

        log::debug!(
            "[trained] loading trained structures, vector_fields={:?}",
            vector_fields.keys().collect::<Vec<_>>()
        );

        for (field_id, field_meta) in built_fields {
            log::debug!(
                "[trained] field {} state={:?} centroids_file={:?} codebook_file={:?}",
                field_id,
                field_meta.state,
                field_meta.centroids_file,
                field_meta.codebook_file,
            );
            if field_meta.field_id != *field_id {
                return Err(Error::Corruption(format!(
                    "trained vector metadata key {field_id} contains field_id {}",
                    field_meta.field_id
                )));
            }

            let schema_config = match schema {
                None => None,
                Some(schema) => {
                    let entry = schema
                        .get_field_entry(crate::dsl::Field(*field_id))
                        .ok_or_else(|| {
                            Error::Corruption(format!(
                                "trained vector metadata references missing field {field_id}"
                            ))
                        })?;
                    if entry.field_type != crate::dsl::FieldType::DenseVector {
                        return Err(Error::Corruption(format!(
                            "trained vector metadata field {field_id} has non-dense schema type {:?}",
                            entry.field_type
                        )));
                    }
                    let config = entry.dense_vector_config.as_ref().ok_or_else(|| {
                        Error::Corruption(format!(
                            "trained vector metadata field {field_id} has no dense-vector configuration"
                        ))
                    })?;
                    if field_meta.index_type != config.index_type {
                        return Err(Error::Corruption(format!(
                            "trained vector metadata field {field_id} uses {:?}, schema requires {:?}",
                            field_meta.index_type, config.index_type
                        )));
                    }
                    Some(config)
                }
            };
            if !matches!(
                field_meta.index_type,
                VectorIndexType::IvfRaBitQ | VectorIndexType::ScaNN
            ) {
                return Err(Error::Corruption(format!(
                    "field {field_id} is Built for {:?}, which has no index-level trained artifacts",
                    field_meta.index_type
                )));
            }

            let expected_clusters = match field_meta.state {
                VectorIndexState::Built { num_clusters, .. } if num_clusters > 0 => num_clusters,
                VectorIndexState::Built { .. } => {
                    return Err(Error::Corruption(format!(
                        "trained vector metadata field {field_id} has zero clusters"
                    )));
                }
                VectorIndexState::Flat => unreachable!("built_fields contains only Built entries"),
            };

            let centroids_file = field_meta.centroids_file.as_deref().ok_or_else(|| {
                Error::Corruption(format!(
                    "trained vector metadata field {field_id} is Built but has no centroids_file"
                ))
            })?;
            let c: crate::structures::CoarseCentroids =
                load_trained_artifact(dir, *field_id, "centroids", centroids_file).await?;
            let expected_dim = schema_config.map_or(c.dim, |config| config.dim);
            let actual_clusters = c.num_clusters as usize;
            let expected_values = actual_clusters.checked_mul(expected_dim).ok_or_else(|| {
                Error::Corruption(format!(
                    "trained centroid dimensions overflow for field {field_id}"
                ))
            })?;
            if actual_clusters == 0
                || actual_clusters > expected_clusters
                || c.dim == 0
                || c.dim != expected_dim
                || c.centroids.len() != expected_values
                || c.centroids.iter().any(|value| !value.is_finite())
            {
                return Err(Error::Corruption(format!(
                    "trained centroids for field {field_id} do not match metadata/schema: \
                     clusters={} (metadata maximum {expected_clusters}), dim={} (expected {}), \
                     values={} (expected {expected_values})",
                    c.num_clusters,
                    c.dim,
                    expected_dim,
                    c.centroids.len(),
                )));
            }
            if actual_clusters < expected_clusters {
                // Older writers persisted the requested cluster count even
                // though the trainer clamps it to the available sample. This
                // shape is safe and self-describing in the artifact; accepting
                // it keeps pre-fix indexes openable. New writers persist the
                // actual count, so no new mismatch is produced.
                log::warn!(
                    "[trained] field {} legacy cluster-count clamp: metadata={}, artifact={}",
                    field_id,
                    expected_clusters,
                    actual_clusters,
                );
            }
            log::debug!(
                "[trained] field {} loaded centroids ({} clusters)",
                field_id,
                c.num_clusters
            );

            if field_meta.index_type == VectorIndexType::ScaNN {
                let codebook_file = field_meta.codebook_file.as_deref().ok_or_else(|| {
                    Error::Corruption(format!(
                        "trained vector metadata field {field_id} is ScaNN Built but has no codebook_file"
                    ))
                })?;
                let codebook: crate::structures::PQCodebook =
                    load_trained_artifact(dir, *field_id, "codebook", codebook_file).await?;
                codebook.validate().map_err(|error| {
                    Error::Corruption(format!(
                        "invalid trained codebook for field {field_id}: {error}"
                    ))
                })?;
                if codebook.config.dim != expected_dim {
                    return Err(Error::Corruption(format!(
                        "trained codebook for field {field_id} has dimension {}, expected {}",
                        codebook.config.dim, expected_dim
                    )));
                }
                log::debug!("[trained] field {} loaded codebook", field_id);
                codebooks.insert(*field_id, Arc::new(codebook));
            }
            centroids.insert(*field_id, Arc::new(c));
        }

        if centroids.is_empty() {
            Ok(None)
        } else {
            Ok(Some(crate::segment::TrainedVectorStructures {
                centroids,
                codebooks,
            }))
        }
    }
}

fn validate_trained_artifact_path(field_id: u32, kind: &str, filename: &str) -> Result<()> {
    use std::path::Component;

    let path = Path::new(filename);
    if filename.is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(Error::Corruption(format!(
            "trained {kind} path for field {field_id} is not a safe relative path: '{filename}'"
        )));
    }
    Ok(())
}

async fn load_trained_artifact<T, D>(
    dir: &D,
    field_id: u32,
    kind: &str,
    filename: &str,
) -> Result<T>
where
    T: serde::de::DeserializeOwned,
    D: crate::directories::Directory,
{
    validate_trained_artifact_path(field_id, kind, filename)?;
    let path = Path::new(filename);
    let file_size = dir.file_size(path).await.map_err(|error| {
        Error::Corruption(format!(
            "failed to stat trained {kind} '{filename}' for field {field_id}: {error}"
        ))
    })?;
    validate_trained_artifact_size(field_id, kind, filename, file_size)?;
    let slice = dir.open_read(path).await.map_err(|error| {
        Error::Corruption(format!(
            "failed to open trained {kind} '{filename}' for field {field_id}: {error}"
        ))
    })?;
    validate_trained_artifact_size(field_id, kind, filename, slice.len())?;
    let bytes = slice.read_bytes().await.map_err(|error| {
        Error::Corruption(format!(
            "failed to read trained {kind} '{filename}' for field {field_id}: {error}"
        ))
    })?;
    let (artifact, consumed) = bincode::serde::decode_from_slice::<T, _>(
        bytes.as_slice(),
        bincode::config::standard().with_limit::<MAX_TRAINED_ARTIFACT_BYTES>(),
    )
    .map_err(|error| {
        Error::Corruption(format!(
            "failed to deserialize trained {kind} '{filename}' for field {field_id}: {error}"
        ))
    })?;
    if consumed != bytes.len() {
        return Err(Error::Corruption(format!(
            "trained {kind} '{filename}' for field {field_id} has {} trailing bytes",
            bytes.len() - consumed
        )));
    }
    Ok(artifact)
}

fn validate_trained_artifact_size(
    field_id: u32,
    kind: &str,
    filename: &str,
    file_size: u64,
) -> Result<()> {
    if file_size > MAX_TRAINED_ARTIFACT_BYTES as u64 {
        return Err(Error::Corruption(format!(
            "trained {kind} '{filename}' for field {field_id} is {file_size} bytes, \
             exceeding the {MAX_TRAINED_ARTIFACT_BYTES}-byte safety limit"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directories::DirectoryWriter;

    #[derive(Clone, Default)]
    struct SyncFailDirectory(crate::directories::RamDirectory);

    #[async_trait::async_trait]
    impl crate::directories::Directory for SyncFailDirectory {
        async fn exists(&self, path: &Path) -> std::io::Result<bool> {
            self.0.exists(path).await
        }

        async fn file_size(&self, path: &Path) -> std::io::Result<u64> {
            self.0.file_size(path).await
        }

        async fn open_read(&self, path: &Path) -> std::io::Result<crate::directories::FileHandle> {
            self.0.open_read(path).await
        }

        async fn read_range(
            &self,
            path: &Path,
            range: std::ops::Range<u64>,
        ) -> std::io::Result<crate::directories::OwnedBytes> {
            self.0.read_range(path, range).await
        }

        async fn list_files(&self, prefix: &Path) -> std::io::Result<Vec<std::path::PathBuf>> {
            self.0.list_files(prefix).await
        }

        async fn open_lazy(&self, path: &Path) -> std::io::Result<crate::directories::FileHandle> {
            self.0.open_lazy(path).await
        }
    }

    #[async_trait::async_trait]
    impl crate::directories::DirectoryWriter for SyncFailDirectory {
        async fn write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
            self.0.write(path, data).await
        }

        async fn delete(&self, path: &Path) -> std::io::Result<()> {
            self.0.delete(path).await
        }

        async fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
            self.0.rename(from, to).await
        }

        async fn sync(&self) -> std::io::Result<()> {
            Err(std::io::Error::other("injected directory fsync failure"))
        }

        async fn streaming_writer(
            &self,
            path: &Path,
        ) -> std::io::Result<Box<dyn crate::directories::StreamingWriter>> {
            self.0.streaming_writer(path).await
        }
    }

    fn test_schema() -> Schema {
        Schema::default()
    }

    fn dense_schema(index_type: VectorIndexType) -> (Schema, crate::dsl::Field) {
        let mut builder = crate::dsl::SchemaBuilder::default();
        let config = match index_type {
            VectorIndexType::IvfRaBitQ => crate::dsl::DenseVectorConfig::with_ivf(2, Some(1), 1),
            VectorIndexType::ScaNN => crate::dsl::DenseVectorConfig::with_scann(2, Some(1), 1),
            other => panic!("unsupported trained test index type: {other:?}"),
        };
        let field = builder.add_dense_vector_field_with_config("embedding", true, true, config);
        (builder.build(), field)
    }

    fn test_centroids() -> crate::structures::CoarseCentroids {
        crate::structures::CoarseCentroids {
            num_clusters: 1,
            dim: 2,
            centroids: vec![0.25, 0.75],
            version: 7,
            soar_config: None,
        }
    }

    async fn write_bincode(
        directory: &crate::directories::RamDirectory,
        filename: &str,
        value: &impl serde::Serialize,
    ) {
        let bytes = bincode::serde::encode_to_vec(value, bincode::config::standard()).unwrap();
        directory.write(Path::new(filename), &bytes).await.unwrap();
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

    #[tokio::test]
    async fn save_treats_post_rename_sync_failure_as_committed() {
        let directory = SyncFailDirectory::default();
        let mut metadata = IndexMetadata::new(test_schema());
        metadata.add_segment("committed".to_string(), 7);

        metadata.save(&directory).await.unwrap();

        let loaded = IndexMetadata::load(&directory).await.unwrap();
        assert_eq!(loaded.segment_doc_count("committed"), Some(7));
    }

    #[tokio::test]
    async fn trained_artifacts_load_only_when_the_complete_built_set_is_valid() {
        let mut builder = crate::dsl::SchemaBuilder::default();
        let config = crate::dsl::DenseVectorConfig::with_ivf(2, Some(1), 1);
        let first = builder.add_dense_vector_field_with_config(
            "first_embedding",
            true,
            true,
            config.clone(),
        );
        let second =
            builder.add_dense_vector_field_with_config("second_embedding", true, true, config);
        let schema = builder.build();
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema.clone());
        metadata.init_field(first.0, VectorIndexType::IvfRaBitQ);
        metadata.init_field(second.0, VectorIndexType::IvfRaBitQ);
        metadata.mark_field_built(first.0, 10, 1, "field_0_centroids.bin".into(), None);
        metadata.mark_field_built(second.0, 10, 1, "field_1_centroids.bin".into(), None);
        write_bincode(&directory, "field_0_centroids.bin", &test_centroids()).await;

        let error = IndexMetadata::try_load_trained_from_fields(
            &metadata.vector_fields,
            &schema,
            &directory,
        )
        .await
        .err()
        .expect("missing artifact must fail the complete load")
        .to_string();
        assert!(error.contains("field_1_centroids.bin"), "{error}");
        assert!(error.contains("field 1"), "{error}");
        assert!(
            IndexMetadata::load_trained_from_fields(&metadata.vector_fields, &directory)
                .await
                .is_none(),
            "the compatibility API must also fail closed instead of returning the valid subset"
        );
    }

    #[tokio::test]
    async fn index_open_fails_closed_when_built_artifact_is_missing() {
        let (schema, field) = dense_schema(VectorIndexType::IvfRaBitQ);
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema);
        metadata.init_field(field.0, VectorIndexType::IvfRaBitQ);
        metadata.mark_field_built(field.0, 10, 1, "missing_centroids.bin".into(), None);
        metadata.save(&directory).await.unwrap();

        let error = match crate::index::Index::open(directory, crate::index::IndexConfig::default())
            .await
        {
            Ok(_) => panic!("Index::open accepted a Built field with no artifact"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("missing_centroids.bin"), "{error}");
    }

    #[tokio::test]
    async fn scann_built_state_requires_a_codebook() {
        let (schema, field) = dense_schema(VectorIndexType::ScaNN);
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema.clone());
        metadata.init_field(field.0, VectorIndexType::ScaNN);
        metadata.mark_field_built(field.0, 10, 1, "field_0_centroids.bin".into(), None);
        write_bincode(&directory, "field_0_centroids.bin", &test_centroids()).await;

        let error = IndexMetadata::try_load_trained_from_fields(
            &metadata.vector_fields,
            &schema,
            &directory,
        )
        .await
        .err()
        .expect("ScaNN Built state without a codebook must fail")
        .to_string();
        assert!(error.contains("has no codebook_file"), "{error}");
    }

    #[tokio::test]
    async fn legacy_requested_cluster_count_accepts_a_clamped_artifact() {
        let mut builder = crate::dsl::SchemaBuilder::default();
        let field = builder.add_dense_vector_field_with_config(
            "embedding",
            true,
            true,
            crate::dsl::DenseVectorConfig::with_ivf(2, Some(4), 1),
        );
        let schema = builder.build();
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema.clone());
        metadata.init_field(field.0, VectorIndexType::IvfRaBitQ);
        metadata.mark_field_built(field.0, 1, 4, "field_0_centroids.bin".into(), None);
        write_bincode(&directory, "field_0_centroids.bin", &test_centroids()).await;

        let trained = IndexMetadata::try_load_trained_from_fields(
            &metadata.vector_fields,
            &schema,
            &directory,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(trained.centroids[&field.0].num_clusters, 1);
    }

    #[tokio::test]
    async fn trained_artifact_loader_rejects_trailing_data() {
        let (schema, field) = dense_schema(VectorIndexType::IvfRaBitQ);
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema.clone());
        metadata.init_field(field.0, VectorIndexType::IvfRaBitQ);
        metadata.mark_field_built(field.0, 10, 1, "field_0_centroids.bin".into(), None);
        let mut bytes =
            bincode::serde::encode_to_vec(test_centroids(), bincode::config::standard()).unwrap();
        bytes.extend_from_slice(&[0xaa, 0xbb]);
        directory
            .write(Path::new("field_0_centroids.bin"), &bytes)
            .await
            .unwrap();

        let error = IndexMetadata::try_load_trained_from_fields(
            &metadata.vector_fields,
            &schema,
            &directory,
        )
        .await
        .err()
        .expect("trailing artifact bytes must fail validation")
        .to_string();
        assert!(error.contains("trailing bytes"), "{error}");
    }

    #[test]
    fn trained_artifact_size_limit_rejects_before_reading() {
        let error = validate_trained_artifact_size(
            3,
            "centroids",
            "field_3_centroids.bin",
            MAX_TRAINED_ARTIFACT_BYTES as u64 + 1,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("exceeding"), "{error}");
        assert!(error.contains("field 3"), "{error}");
    }

    #[tokio::test]
    async fn trained_artifact_decode_limit_rejects_forged_collection_length() {
        let (schema, field) = dense_schema(VectorIndexType::IvfRaBitQ);
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema.clone());
        metadata.init_field(field.0, VectorIndexType::IvfRaBitQ);
        metadata.mark_field_built(field.0, 10, 1, "field_0_centroids.bin".into(), None);

        // CoarseCentroids begins with num_clusters=1, dim=2, then the Vec
        // length. Bincode's standard varint marker 253 introduces a u64; this
        // tiny payload claims an impossible f32 vector and must hit the decode
        // limit before any large allocation is attempted.
        let mut bytes = vec![1, 2, 253];
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        directory
            .write(Path::new("field_0_centroids.bin"), &bytes)
            .await
            .unwrap();

        let error = IndexMetadata::try_load_trained_from_fields(
            &metadata.vector_fields,
            &schema,
            &directory,
        )
        .await
        .err()
        .expect("forged collection length must fail the bounded decoder")
        .to_string();
        assert!(error.contains("failed to deserialize"), "{error}");
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
    fn total_vectors_is_aggregate_of_built_field_counts() {
        let mut meta = IndexMetadata::new(test_schema());
        meta.init_field(7, VectorIndexType::IvfRaBitQ);
        meta.init_field(3, VectorIndexType::ScaNN);

        // Build in reverse field-id order to ensure the result is not tied to
        // HashMap or training iteration order.
        meta.mark_field_built(7, 400, 20, "field_7_centroids.bin".to_string(), None);
        assert_eq!(meta.total_vectors, 400);
        meta.mark_field_built(
            3,
            250,
            15,
            "field_3_centroids.bin".to_string(),
            Some("field_3_codebook.bin".to_string()),
        );
        assert_eq!(meta.total_vectors, 650);

        // Rebuilding a field replaces its contribution; it does not add a
        // duplicate training snapshot.
        meta.mark_field_built(7, 425, 20, "field_7_centroids.bin".to_string(), None);
        assert_eq!(meta.total_vectors, 675);
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

    #[test]
    fn old_metadata_defaults_the_bp_retry_counter() {
        let mut meta = IndexMetadata::new(test_schema());
        meta.add_segment("legacy".to_string(), 10);
        let mut json = serde_json::to_value(&meta).unwrap();
        json["segment_metas"]["legacy"]
            .as_object_mut()
            .unwrap()
            .remove("bp_unconverged_passes");

        let loaded: IndexMetadata = serde_json::from_value(json).unwrap();
        assert_eq!(loaded.segment_metas["legacy"].bp_unconverged_passes, 0);
    }

    #[test]
    fn test_merged_segment_lineage() {
        let mut meta = IndexMetadata::new(test_schema());
        meta.add_segment("a".to_string(), 50);
        meta.add_segment("b".to_string(), 75);

        // Fresh segments: gen=0, no ancestors
        assert_eq!(meta.segment_metas["a"].generation, 0);
        assert!(meta.segment_metas["a"].ancestors.is_empty());

        // Merge a+b → c
        meta.add_merged_segment(
            "c".to_string(),
            125,
            vec!["a".to_string(), "b".to_string()],
            1,
            false,
            true,
        );
        assert_eq!(meta.segment_metas["c"].generation, 1);
        assert_eq!(meta.segment_metas["c"].ancestors, vec!["a", "b"]);
        assert_eq!(meta.segment_doc_count("c"), Some(125));

        // Merge c+d → e (gen should be 2)
        meta.add_segment("d".to_string(), 30);
        meta.add_merged_segment(
            "e".to_string(),
            155,
            vec!["c".to_string(), "d".to_string()],
            2,
            false,
            true,
        );
        assert_eq!(meta.segment_metas["e"].generation, 2);
    }
}
