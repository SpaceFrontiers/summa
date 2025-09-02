//! Unified index metadata - segments list + vector index state
//!
//! This module manages all index-level metadata in a single `metadata.json` file:
//! - List of committed segments
//! - Vector index state per field (Flat/Built)
//! - Trained centroid artifact paths
//!
//! The workflow is:
//! 1. During initial accumulation, segments store flat vectors.
//! 2. A manual build trains the first coarse-centroid ANN generation.
//! 3. A manual retrain stages and atomically publishes a replacement generation.
//! 4. On index open, metadata loads the currently published artifacts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use crate::dsl::{BinaryIndexType, Schema, VectorIndexType};
use crate::error::{Error, Result};

/// Metadata file name at index level
pub const INDEX_META_FILENAME: &str = "metadata.json";
/// Temp file for atomic writes (write here, then rename to INDEX_META_FILENAME)
const INDEX_META_TMP_FILENAME: &str = "metadata.json.tmp";

/// Current metadata.json format version written by this build.
///
/// `load` requires this exact version. Metadata/segment compatibility is a
/// clean rebuild boundary; serde_json would otherwise silently drop fields it
/// does not know and a later save could destructively rewrite index state.
pub const INDEX_META_FORMAT_VERSION: u32 = 6;

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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "index", rename_all = "snake_case")]
pub enum VectorFieldIndexType {
    Float(VectorIndexType),
    Binary(BinaryIndexType),
}

impl From<VectorIndexType> for VectorFieldIndexType {
    fn from(value: VectorIndexType) -> Self {
        Self::Float(value)
    }
}

impl From<BinaryIndexType> for VectorFieldIndexType {
    fn from(value: BinaryIndexType) -> Self {
        Self::Binary(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldVectorMeta {
    /// Field ID
    pub field_id: u32,
    /// Configured index type (target type when built)
    pub index_type: VectorFieldIndexType,
    /// Current state
    pub state: VectorIndexState,
    /// Path to centroids file (relative to index dir)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub centroids_file: Option<String>,
    /// Legacy: path to a trained IVF-PQ codebook. Always `None` for current
    /// formats; kept so pre-removal metadata deserializes into an actionable
    /// error instead of dropping the field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codebook_file: Option<String>,
    /// ScaNN global model generation encoded into every segment payload.
    /// Absent for legacy IVF/TQ fields and older metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_generation: Option<u64>,
    /// Content fingerprint of the exact ScaNN model encoded into every
    /// segment payload. This prevents same-generation accidental mixing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<u64>,
}

/// Unified index metadata - single source of truth for index state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Version for compatibility
    pub version: u32,
    /// Monotonic publication identity for schema/vector generations. Segment
    /// commits are already detected by their IDs; this also makes query-only
    /// ALTERs (for example `nprobe`) visible to cached readers.
    #[serde(default)]
    pub publication_generation: u64,
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
            version: INDEX_META_FORMAT_VERSION,
            publication_generation: 0,
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
    pub fn init_field(&mut self, field_id: u32, index_type: impl Into<VectorFieldIndexType>) {
        let index_type = index_type.into();
        self.vector_fields
            .entry(field_id)
            .or_insert(FieldVectorMeta {
                field_id,
                index_type,
                state: VectorIndexState::Flat,
                centroids_file: None,
                codebook_file: None,
                artifact_generation: None,
                artifact_id: None,
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
            field.artifact_generation = None;
            field.artifact_id = None;
            self.refresh_total_vectors();
        }
    }

    /// Mark a ScaNN field built against one immutable global model. Reuses
    /// `centroids_file` as the trained-artifact path for wire compatibility;
    /// the explicit generation/fingerprint fields make its semantics loud.
    pub fn mark_scann_field_built(
        &mut self,
        field_id: u32,
        vector_count: usize,
        num_leaves: usize,
        artifact_file: String,
        artifact_generation: u64,
        artifact_id: u64,
    ) -> Result<()> {
        if artifact_generation == 0 || artifact_id == 0 {
            return Err(Error::Corruption(format!(
                "ScaNN field {field_id} cannot publish a zero generation or artifact fingerprint"
            )));
        }
        let field = self.vector_fields.get_mut(&field_id).ok_or_else(|| {
            Error::Corruption(format!(
                "ScaNN field {field_id} must be initialized before it is marked built"
            ))
        })?;
        if !matches!(
            field.index_type,
            VectorFieldIndexType::Float(VectorIndexType::Scann)
                | VectorFieldIndexType::Binary(BinaryIndexType::Scann)
        ) {
            return Err(Error::Corruption(format!(
                "field {field_id} is not configured as ScaNN"
            )));
        }
        field.state = VectorIndexState::Built {
            vector_count,
            num_clusters: num_leaves,
        };
        field.centroids_file = Some(artifact_file);
        field.codebook_file = None;
        field.artifact_generation = Some(artifact_generation);
        field.artifact_id = Some(artifact_id);
        self.refresh_total_vectors();
        Ok(())
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
                Self::deserialize_versioned(bytes.as_slice())
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Try recovering from temp file (crash between write and rename)
                let tmp_path = Path::new(INDEX_META_TMP_FILENAME);
                let slice = dir.open_read(tmp_path).await?;
                let bytes = slice.read_bytes().await?;
                let meta = Self::deserialize_versioned(bytes.as_slice())?;
                log::warn!("Recovered metadata from temp file (previous crash during save)");
                Ok(meta)
            }
            Err(e) => Err(Error::Io(e)),
        }
    }

    /// Deserialize only the current format. Vector artifacts are intentionally
    /// rebuilt when the ANN format changes; silently accepting older metadata
    /// would mix incompatible segment and global-codebook generations.
    fn deserialize_versioned(bytes: &[u8]) -> Result<Self> {
        let meta: Self =
            serde_json::from_slice(bytes).map_err(|e| Error::Serialization(e.to_string()))?;
        crate::dsl::reject_removed_vector_index_types(&meta.schema).map_err(Error::Schema)?;
        if meta.version != INDEX_META_FORMAT_VERSION {
            return Err(Error::Corruption(format!(
                "metadata.json format version {} is incompatible with required version {}; \
                 rebuild and republish the index with this Hermes version",
                meta.version, INDEX_META_FORMAT_VERSION
            )));
        }
        Ok(meta)
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

    /// Fallible schema-aware loader used for lifecycle publication.
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub(crate) async fn try_load_trained_from_fields<D: crate::directories::Directory>(
        vector_fields: &HashMap<u32, FieldVectorMeta>,
        schema: &Schema,
        dir: &D,
    ) -> Result<Option<crate::segment::TrainedVectorStructures>> {
        Self::load_trained_from_fields_impl(vector_fields, schema, dir).await
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
        schema: &Schema,
        dir: &D,
    ) -> Result<Option<crate::segment::TrainedVectorStructures>> {
        use std::sync::Arc;

        let mut centroids = rustc_hash::FxHashMap::default();
        let mut binary_quantizers = rustc_hash::FxHashMap::default();
        let mut scann_artifacts = rustc_hash::FxHashMap::default();

        let mut built_fields: Vec<_> = vector_fields
            .iter()
            .filter(|(_, meta)| matches!(meta.state, VectorIndexState::Built { .. }))
            .collect();
        built_fields.sort_unstable_by_key(|(field_id, _)| **field_id);

        log::debug!(
            "[trained] index={} loading trained structures, dense_vector_fields={:?}",
            schema.index_label(),
            vector_fields.keys().collect::<Vec<_>>()
        );

        for (field_id, field_meta) in built_fields {
            log::debug!(
                "[trained] index={} field {} state={:?} centroids_file={:?} codebook_file={:?}",
                schema.index_label(),
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
            match field_meta.index_type {
                VectorFieldIndexType::Float(VectorIndexType::IvfPq) => {
                    return Err(Error::Corruption(format!(
                        "field {field_id} was trained as IVF-PQ, which is no longer \
                         supported; recreate the index with `ivf_tq` and reindex \
                         (docs/turboquant-quantization.md)"
                    )));
                }
                VectorFieldIndexType::Float(index_type @ VectorIndexType::IvfTq) => {
                    let entry = schema
                        .get_field_entry(crate::dsl::Field(*field_id))
                        .ok_or_else(|| {
                            Error::Corruption(format!(
                                "trained vector metadata references missing field {field_id}"
                            ))
                        })?;
                    let schema_config = entry
                        .dense_vector_config
                        .as_ref()
                        .filter(|_| entry.field_type == crate::dsl::FieldType::DenseVector)
                        .ok_or_else(|| {
                            Error::Corruption(format!(
                                "trained vector metadata field {field_id} is not a float dense field"
                            ))
                        })?;
                    if schema_config.index_type != index_type {
                        return Err(Error::Corruption(format!(
                            "trained vector metadata field {field_id} uses {index_type:?}, schema requires {:?}",
                            schema_config.index_type
                        )));
                    }
                    let c: crate::structures::CoarseCentroids =
                        load_trained_artifact(dir, *field_id, "centroids", centroids_file).await?;
                    let expected_dim = schema_config.dim;
                    let actual_clusters = c.num_clusters as usize;
                    let expected_values =
                        actual_clusters.checked_mul(expected_dim).ok_or_else(|| {
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
                            "trained centroids for field {field_id} do not match metadata/schema"
                        )));
                    }
                    if !crate::structures::is_ivf_tq_cosine_generation(c.version) {
                        return Err(Error::Corruption(format!(
                            "trained IVF-TQ centroids for field {field_id} use an \
                             unsupported legacy generation; rebuild the index"
                        )));
                    }
                    c.validate_routing(schema_config.ivf_routing)
                        .map_err(|error| {
                            Error::Corruption(format!(
                                "invalid trained centroid routing for field {field_id}: {error}"
                            ))
                        })?;
                    // The TQ leaf codec is derived, never trained; ensure
                    // `index_type` stays referenced for future variants.
                    let _ = index_type;
                    if field_meta.codebook_file.is_some() {
                        return Err(Error::Corruption(format!(
                            "trained IVF-TQ field {field_id} unexpectedly references a codebook file"
                        )));
                    }
                    centroids.insert(*field_id, Arc::new(c));
                }
                VectorFieldIndexType::Binary(BinaryIndexType::Ivf) => {
                    let entry = schema
                        .get_field_entry(crate::dsl::Field(*field_id))
                        .ok_or_else(|| {
                            Error::Corruption(format!(
                                "trained vector metadata references missing field {field_id}"
                            ))
                        })?;
                    let schema_config = entry
                        .binary_dense_vector_config
                        .as_ref()
                        .filter(|config| {
                            entry.field_type == crate::dsl::FieldType::BinaryDenseVector
                                && config.index_type == BinaryIndexType::Ivf
                        })
                        .ok_or_else(|| {
                            Error::Corruption(format!(
                                "trained vector metadata field {field_id} is not a binary IVF field"
                            ))
                        })?;
                    let quantizer: crate::structures::BinaryCoarseQuantizer =
                        load_trained_artifact(dir, *field_id, "binary centroids", centroids_file)
                            .await?;
                    quantizer.validate().map_err(|error| {
                        Error::Corruption(format!(
                            "invalid binary coarse quantizer for field {field_id}: {error}"
                        ))
                    })?;
                    let actual_clusters = quantizer.num_clusters as usize;
                    if actual_clusters > expected_clusters
                        || schema_config.dim != quantizer.dim_bits
                    {
                        return Err(Error::Corruption(format!(
                            "binary coarse quantizer for field {field_id} does not match metadata/schema"
                        )));
                    }
                    quantizer
                        .validate_routing(schema_config.ivf_routing)
                        .map_err(|error| {
                            Error::Corruption(format!(
                                "invalid binary centroid routing for field {field_id}: {error}"
                            ))
                        })?;
                    binary_quantizers.insert(*field_id, Arc::new(quantizer));
                }
                VectorFieldIndexType::Float(VectorIndexType::Scann)
                | VectorFieldIndexType::Binary(BinaryIndexType::Scann) => {
                    if field_meta.codebook_file.is_some() {
                        return Err(Error::Corruption(format!(
                            "trained ScaNN field {field_id} unexpectedly references a separate codebook file"
                        )));
                    }
                    let expected_generation = field_meta.artifact_generation.ok_or_else(|| {
                        Error::Corruption(format!(
                            "trained ScaNN field {field_id} has no artifact generation"
                        ))
                    })?;
                    let expected_artifact_id = field_meta.artifact_id.ok_or_else(|| {
                        Error::Corruption(format!(
                            "trained ScaNN field {field_id} has no artifact fingerprint"
                        ))
                    })?;
                    validate_trained_artifact_path(
                        field_id.to_owned(),
                        "ScaNN artifact",
                        centroids_file,
                    )?;
                    let path = Path::new(centroids_file);
                    let slice = dir.open_read(path).await.map_err(|error| {
                        Error::Corruption(format!(
                            "failed to open trained ScaNN artifact '{centroids_file}' for field {field_id}: {error}"
                        ))
                    })?;
                    let raw = slice.read_bytes().await.map_err(|error| {
                        Error::Corruption(format!(
                            "failed to map trained ScaNN artifact '{centroids_file}' for field {field_id}: {error}"
                        ))
                    })?;
                    let artifact =
                        crate::segment::ScannTrainedArtifactBytes::open(raw).map_err(|error| {
                            Error::Corruption(format!(
                                "invalid trained ScaNN artifact for field {field_id}: {error}"
                            ))
                        })?;
                    if artifact.generation() != expected_generation
                        || artifact.artifact_id() != expected_artifact_id
                        || artifact.config().num_leaves as usize != expected_clusters
                    {
                        return Err(Error::Corruption(format!(
                            "trained ScaNN artifact for field {field_id} does not match metadata"
                        )));
                    }
                    let entry = schema
                        .get_field_entry(crate::dsl::Field(*field_id))
                        .ok_or_else(|| {
                            Error::Corruption(format!(
                                "trained vector metadata references missing field {field_id}"
                            ))
                        })?;
                    let schema_matches = match field_meta.index_type {
                        VectorFieldIndexType::Float(VectorIndexType::Scann) => entry
                            .dense_vector_config
                            .as_ref()
                            .filter(|_| entry.field_type == crate::dsl::FieldType::DenseVector)
                            .is_some_and(|config| {
                                config.index_type == VectorIndexType::Scann
                                    && config.dim == artifact.config().dimension as usize
                                    && scann_explicit_geometry_matches(
                                        config.num_clusters,
                                        config.tree_levels,
                                        artifact.config().num_leaves as usize,
                                        artifact.config().tree_levels,
                                    )
                                    && matches!(
                                        artifact.config().encoding,
                                        crate::structures::vector::scann::ScannEncoding::AsymmetricHash { .. }
                                    )
                            }),
                        VectorFieldIndexType::Binary(BinaryIndexType::Scann) => entry
                            .binary_dense_vector_config
                            .as_ref()
                            .filter(|_| {
                                entry.field_type == crate::dsl::FieldType::BinaryDenseVector
                            })
                            .is_some_and(|config| {
                                config.index_type == BinaryIndexType::Scann
                                    && config.dim == artifact.config().dimension as usize
                                    && scann_explicit_geometry_matches(
                                        config.num_clusters,
                                        config.tree_levels,
                                        artifact.config().num_leaves as usize,
                                        artifact.config().tree_levels,
                                    )
                                    && artifact.config().encoding
                                        == crate::structures::vector::scann::ScannEncoding::BinaryHamming
                            }),
                        _ => false,
                    };
                    if !schema_matches {
                        return Err(Error::Corruption(format!(
                            "trained ScaNN artifact for field {field_id} does not match schema geometry/encoding"
                        )));
                    }
                    scann_artifacts.insert(*field_id, Arc::new(artifact));
                }
                unsupported => {
                    return Err(Error::Corruption(format!(
                        "field {field_id} is Built for {unsupported:?}, which has no global IVF artifacts"
                    )));
                }
            }
        }

        if centroids.is_empty() && binary_quantizers.is_empty() && scann_artifacts.is_empty() {
            Ok(None)
        } else {
            let trained = crate::segment::TrainedVectorStructures {
                #[cfg(feature = "native")]
                _ann_pins: Default::default(),
                centroids,
                binary_quantizers,
                scann_artifacts,
            };
            #[cfg(feature = "native")]
            let trained = {
                let mut trained = trained;
                trained.pin_ann_structures(crate::segment::pin::pin_policy());
                trained
            };
            Ok(Some(trained))
        }
    }
}

/// Omitted ScaNN geometry is autopilot: the trained artifact's resolved
/// values are authoritative and are also pinned by `VectorIndexState::Built`.
/// Explicit schema values remain strict compatibility requirements.
fn scann_explicit_geometry_matches(
    configured_leaves: Option<usize>,
    configured_levels: Option<u8>,
    resolved_leaves: usize,
    resolved_levels: u8,
) -> bool {
    configured_leaves.is_none_or(|leaves| leaves == resolved_leaves)
        && configured_levels.is_none_or(|levels| levels == resolved_levels)
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
            VectorIndexType::IvfTq => crate::dsl::DenseVectorConfig::ivf_tq(2, Some(1), 1),
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
            version: crate::structures::mark_ivf_tq_cosine_generation(7),
            soar_config: None,
            routing_index: None,
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

        meta.init_field(0, VectorIndexType::IvfTq);
        assert!(!meta.is_field_built(0));
        assert!(meta.vector_fields.contains_key(&0));
    }

    #[tokio::test]
    async fn load_refuses_metadata_stamped_with_a_newer_format_version() {
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(test_schema());
        metadata.version = INDEX_META_FORMAT_VERSION + 1;
        metadata.save(&directory).await.unwrap();

        let error = IndexMetadata::load(&directory)
            .await
            .expect_err("metadata from a newer format version must be refused, not silently pruned")
            .to_string();
        assert!(error.contains("version 6"), "{error}");
        assert!(error.contains("incompatible"), "{error}");
    }

    #[tokio::test]
    async fn tmp_recovery_refuses_metadata_stamped_with_a_newer_format_version() {
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(test_schema());
        metadata.version = INDEX_META_FORMAT_VERSION + 1;
        let bytes = metadata.serialize_to_bytes().unwrap();
        // Simulate a crash between write and rename: only the temp file exists.
        directory
            .write(Path::new(INDEX_META_TMP_FILENAME), &bytes)
            .await
            .unwrap();

        let error = IndexMetadata::load(&directory)
            .await
            .expect_err("temp-file recovery must apply the same version gate")
            .to_string();
        assert!(error.contains("version 6"), "{error}");
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
        let config = crate::dsl::DenseVectorConfig::ivf_tq(2, Some(1), 1);
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
        metadata.init_field(first.0, VectorIndexType::IvfTq);
        metadata.init_field(second.0, VectorIndexType::IvfTq);
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
    }

    #[tokio::test]
    async fn index_open_fails_closed_when_built_artifact_is_missing() {
        let (schema, field) = dense_schema(VectorIndexType::IvfTq);
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema);
        metadata.init_field(field.0, VectorIndexType::IvfTq);
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
    async fn ivf_tq_built_state_rejects_a_codebook_file() {
        let (schema, field) = dense_schema(VectorIndexType::IvfTq);
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema.clone());
        metadata.init_field(field.0, VectorIndexType::IvfTq);
        metadata.mark_field_built(
            field.0,
            10,
            1,
            "field_0_centroids.bin".into(),
            Some("field_0_codebook.bin".into()),
        );
        write_bincode(&directory, "field_0_centroids.bin", &test_centroids()).await;

        let error = IndexMetadata::try_load_trained_from_fields(
            &metadata.vector_fields,
            &schema,
            &directory,
        )
        .await
        .err()
        .expect("IVF-TQ Built state with a codebook file must fail")
        .to_string();
        assert!(error.contains("codebook"), "{error}");
    }

    #[tokio::test]
    async fn legacy_ivf_tq_centroid_generation_is_rejected_while_loading() {
        let (schema, field) = dense_schema(VectorIndexType::IvfTq);
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema.clone());
        metadata.init_field(field.0, VectorIndexType::IvfTq);
        metadata.mark_field_built(field.0, 10, 1, "field_0_centroids.bin".into(), None);
        let mut legacy = test_centroids();
        legacy.version = 7;
        write_bincode(&directory, "field_0_centroids.bin", &legacy).await;

        let error = IndexMetadata::try_load_trained_from_fields(
            &metadata.vector_fields,
            &schema,
            &directory,
        )
        .await
        .err()
        .expect("legacy IVF-TQ centroid state must fail while loading")
        .to_string();
        assert!(error.contains("unsupported legacy generation"), "{error}");
        assert!(error.contains("rebuild the index"), "{error}");
    }

    #[tokio::test]
    async fn legacy_ivf_pq_trained_field_fails_with_actionable_error() {
        // Simulates metadata written by a pre-removal version: the schema
        // gate rejects `ivf_pq` fields, so build the raw field-state map
        // directly against a current-format schema.
        let (schema, field) = dense_schema(VectorIndexType::IvfTq);
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema.clone());
        metadata.init_field(field.0, VectorIndexType::IvfTq);
        metadata.mark_field_built(field.0, 10, 1, "field_0_centroids.bin".into(), None);
        // Overwrite the recorded type the way pre-removal metadata carries it
        // (init_field never downgrades an existing entry).
        metadata
            .vector_fields
            .get_mut(&field.0)
            .expect("field initialized")
            .index_type = VectorFieldIndexType::Float(VectorIndexType::IvfPq);
        write_bincode(&directory, "field_0_centroids.bin", &test_centroids()).await;

        let error = IndexMetadata::try_load_trained_from_fields(
            &metadata.vector_fields,
            &schema,
            &directory,
        )
        .await
        .err()
        .expect("legacy IVF-PQ trained state must fail loudly")
        .to_string();
        assert!(error.contains("no longer"), "{error}");
        assert!(error.contains("ivf_tq"), "{error}");
    }

    #[tokio::test]
    async fn requested_cluster_count_accepts_a_quality_clamped_artifact() {
        let mut builder = crate::dsl::SchemaBuilder::default();
        let field = builder.add_dense_vector_field_with_config(
            "embedding",
            true,
            true,
            crate::dsl::DenseVectorConfig::ivf_tq(2, Some(4), 1),
        );
        let schema = builder.build();
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema.clone());
        metadata.init_field(field.0, VectorIndexType::IvfTq);
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
        let (schema, field) = dense_schema(VectorIndexType::IvfTq);
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema.clone());
        metadata.init_field(field.0, VectorIndexType::IvfTq);
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
        let (schema, field) = dense_schema(VectorIndexType::IvfTq);
        let directory = crate::directories::RamDirectory::new();
        let mut metadata = IndexMetadata::new(schema.clone());
        metadata.init_field(field.0, VectorIndexType::IvfTq);
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
        meta.init_field(0, VectorIndexType::IvfTq);
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
    fn scann_metadata_persists_generation_and_fingerprint_and_defaults_old_json() {
        let mut meta = IndexMetadata::new(test_schema());
        meta.init_field(3, VectorIndexType::Scann);
        meta.mark_scann_field_built(
            3,
            100_000,
            1_000,
            "field_3_scann_17.bin".to_string(),
            17,
            0xdecafbad,
        )
        .unwrap();

        let bytes = meta.serialize_to_bytes().unwrap();
        let decoded: IndexMetadata = serde_json::from_slice(&bytes).unwrap();
        let field = decoded.get_field_meta(3).unwrap();
        assert_eq!(field.artifact_generation, Some(17));
        assert_eq!(field.artifact_id, Some(0xdecafbad));

        let mut legacy_json = serde_json::to_value(&decoded).unwrap();
        legacy_json["vector_fields"]["3"]
            .as_object_mut()
            .unwrap()
            .remove("artifact_generation");
        legacy_json["vector_fields"]["3"]
            .as_object_mut()
            .unwrap()
            .remove("artifact_id");
        let legacy: IndexMetadata = serde_json::from_value(legacy_json).unwrap();
        let legacy_field = legacy.get_field_meta(3).unwrap();
        assert_eq!(legacy_field.artifact_generation, None);
        assert_eq!(legacy_field.artifact_id, None);
    }

    #[test]
    fn scann_metadata_refuses_zero_or_non_scann_generation() {
        let mut meta = IndexMetadata::new(test_schema());
        meta.init_field(0, VectorIndexType::Scann);
        assert!(
            meta.mark_scann_field_built(0, 100_000, 1_000, "artifact.bin".into(), 0, 1)
                .is_err()
        );
        meta.init_field(1, VectorIndexType::IvfTq);
        assert!(
            meta.mark_scann_field_built(1, 100_000, 1_000, "artifact.bin".into(), 1, 2)
                .is_err()
        );
    }

    #[test]
    fn scann_autopilot_accepts_resolved_billion_scale_three_level_geometry() {
        assert!(scann_explicit_geometry_matches(None, None, 10_000_000, 3));
        assert!(!scann_explicit_geometry_matches(
            Some(1_000_000),
            None,
            10_000_000,
            3
        ));
        assert!(!scann_explicit_geometry_matches(
            None,
            Some(1),
            10_000_000,
            3
        ));
    }

    #[test]
    fn total_vectors_is_aggregate_of_built_field_counts() {
        let mut meta = IndexMetadata::new(test_schema());
        meta.init_field(7, VectorIndexType::IvfTq);
        meta.init_field(3, VectorIndexType::IvfTq);

        // Build in reverse field-id order to ensure the result is not tied to
        // HashMap or training iteration order.
        meta.mark_field_built(7, 400, 20, "field_7_centroids.bin".to_string(), None);
        assert_eq!(meta.total_vectors, 400);
        meta.mark_field_built(3, 250, 15, "field_3_centroids.bin".to_string(), None);
        assert_eq!(meta.total_vectors, 650);

        // Rebuilding a field replaces its contribution; it does not add a
        // duplicate training snapshot.
        meta.mark_field_built(7, 425, 20, "field_7_centroids.bin".to_string(), None);
        assert_eq!(meta.total_vectors, 675);
    }

    #[test]
    fn test_should_build_field() {
        let mut meta = IndexMetadata::new(test_schema());
        meta.init_field(0, VectorIndexType::IvfTq);

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
        meta.init_field(0, VectorIndexType::IvfTq);
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
