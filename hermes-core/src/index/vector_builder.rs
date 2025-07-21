//! Vector index building for IndexWriter
//!
//! Training is **manual-only** — decoupled from commit.
//! `build_vector_index()` creates missing codebooks; `retrain_vector_index()`
//! replaces existing codebooks. Both finish every committed ANN segment.

use std::io::Write;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::directories::DirectoryWriter;
use crate::dsl::{
    BinaryDenseVectorConfig, BinaryIndexType, DenseVectorConfig, Field, FieldType, VectorIndexType,
};
use crate::error::{Error, Result};
use crate::segment::{SegmentFiles, SegmentId, SegmentMeta};

use super::IndexWriter;

/// Maximum supported IVF centroid count. Query-side `nprobe` and serialized
/// cluster identifiers use the same practical bound.
const MAX_IVF_CLUSTERS: usize = 1_048_576;
/// Faiss-style clustering quality floor: fewer points per centroid generally
/// overfits the training sample and leaves unstable/empty cells.
const MIN_TRAINING_POINTS_PER_CENTROID: usize = 39;
/// Bound transient I/O/dequantization buffers independently of the configured
/// total training sample budget.
const MAX_SAMPLE_READ_BYTES: usize = 64 * 1024 * 1024;
const SAMPLE_BLOCK: usize = 256;
/// Generation-qualified filenames make retraining crash-safe: the currently
/// published metadata never points at a file being overwritten in place.
const VECTOR_ARTIFACT_PREFIX: &str = "vector_artifact_";

struct TrainedFieldUpdate {
    field_id: u32,
    index_type: super::metadata::VectorFieldIndexType,
    vector_count: usize,
    num_clusters: usize,
    centroids_file: String,
    codebook_file: Option<String>,
}

enum TrainedFieldArtifacts {
    Float {
        centroids: crate::structures::CoarseCentroids,
        codebook: crate::structures::PQCodebook,
        pq_sample_count: usize,
    },
    /// IVF-TQ: only the coarse router is trained; the TQ leaf codec is
    /// derived from the dimension.
    FloatCentroids(crate::structures::CoarseCentroids),
    Binary(crate::structures::BinaryCoarseQuantizer),
}

struct TrainedFieldModel {
    update: TrainedFieldUpdate,
    artifacts: TrainedFieldArtifacts,
}

#[derive(Clone)]
enum IvfFieldConfig {
    Float(DenseVectorConfig),
    Binary(BinaryDenseVectorConfig),
}

impl IvfFieldConfig {
    fn dim(&self) -> usize {
        match self {
            Self::Float(config) => config.dim,
            Self::Binary(config) => config.dim,
        }
    }

    fn index_type(&self) -> super::metadata::VectorFieldIndexType {
        match self {
            Self::Float(config) => config.index_type.into(),
            Self::Binary(config) => config.index_type.into(),
        }
    }

    fn num_clusters(&self) -> Option<usize> {
        match self {
            Self::Float(config) => config.num_clusters,
            Self::Binary(config) => config.num_clusters,
        }
    }

    fn optimal_num_clusters(&self, vector_count: usize) -> usize {
        match self {
            Self::Float(config) => config.optimal_num_clusters(vector_count),
            Self::Binary(config) => config.optimal_num_clusters(vector_count),
        }
    }
}

enum TrainingSample {
    Float(Vec<Vec<f32>>),
    Binary(Vec<u8>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VectorGenerationMode {
    BuildMissing,
    RetrainAll,
}

impl TrainingSample {
    fn len(&self, dim: usize) -> usize {
        match self {
            Self::Float(vectors) => vectors.len(),
            Self::Binary(codes) => codes.len() / dim.div_ceil(8),
        }
    }
}

/// Write adapter that rejects an artifact before its serialized form exceeds
/// the same bound enforced by the loader. Encoding directly through this
/// adapter avoids materializing a second, potentially hundreds-of-megabytes
/// copy of the trained structure.
struct SizeLimitedWriter<'a, W: Write + ?Sized> {
    inner: &'a mut W,
    written: usize,
    limit: usize,
}

impl<'a, W: Write + ?Sized> SizeLimitedWriter<'a, W> {
    fn new(inner: &'a mut W, limit: usize) -> Self {
        Self {
            inner,
            written: 0,
            limit,
        }
    }
}

impl<W: Write + ?Sized> Write for SizeLimitedWriter<'_, W> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let next_size = self
            .written
            .checked_add(buffer.len())
            .ok_or_else(|| std::io::Error::other("trained artifact size overflow"))?;
        if next_size > self.limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "trained artifact exceeds the {}-byte safety limit",
                    self.limit
                ),
            ));
        }
        let written = self.inner.write(buffer)?;
        self.written += written;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn validate_explicit_cluster_count(num_clusters: Option<usize>) -> Result<()> {
    match num_clusters {
        Some(0) => Err(Error::Schema(
            "dense vector num_clusters must be at least 1".to_string(),
        )),
        Some(value) if value > MAX_IVF_CLUSTERS => Err(Error::Schema(format!(
            "dense vector num_clusters must not exceed {MAX_IVF_CLUSTERS}, got {value}"
        ))),
        _ => Ok(()),
    }
}

fn effective_field_num_clusters(
    config: &IvfFieldConfig,
    corpus_count: usize,
    sample_count: usize,
) -> Result<usize> {
    if sample_count == 0 {
        return Err(Error::Schema(
            "cannot train an IVF vector index without sample vectors".to_string(),
        ));
    }
    validate_explicit_cluster_count(config.num_clusters())?;
    let centroid_bytes = match config {
        IvfFieldConfig::Float(config) => config.dim.saturating_mul(size_of::<f32>()),
        IvfFieldConfig::Binary(config) => config.dim.div_ceil(8),
    };
    let artifact_limit = super::metadata::MAX_TRAINED_ARTIFACT_BYTES
        .saturating_sub(1024)
        .checked_div(centroid_bytes.max(1))
        .unwrap_or(0)
        .max(1);
    let quality_limit = if config.num_clusters().is_some() {
        sample_count
    } else {
        (sample_count / MIN_TRAINING_POINTS_PER_CENTROID)
            .max(16)
            .min(sample_count)
    };
    let requested = config.optimal_num_clusters(corpus_count);
    if config.num_clusters().is_some() && requested > artifact_limit {
        return Err(Error::Schema(format!(
            "configured IVF codebook needs {} bytes for {} centroids, exceeding the {}-byte artifact limit",
            requested.saturating_mul(centroid_bytes),
            requested,
            super::metadata::MAX_TRAINED_ARTIFACT_BYTES,
        )));
    }
    Ok(requested.min(quality_limit).min(artifact_limit))
}

fn training_sample_limit(
    max_samples: usize,
    max_bytes: usize,
    bytes_per_sample: usize,
) -> Result<usize> {
    if max_samples == 0 || max_bytes == 0 || bytes_per_sample == 0 {
        return Err(Error::Schema(
            "vector training sample count, memory budget, and vector size must be greater than zero"
                .into(),
        ));
    }
    let memory_limited = max_bytes / bytes_per_sample;
    if memory_limited == 0 {
        return Err(Error::Schema(format!(
            "vector training memory budget ({max_bytes} bytes) cannot hold one {bytes_per_sample}-byte sample"
        )));
    }
    Ok(max_samples.min(memory_limited))
}

/// Validate the configured centroid count and cap it to the training sample.
///
/// Corpus size drives the automatic heuristic, but training cannot produce
/// more distinct centroids than the number of sampled vectors. Keeping this
/// decision here avoids relying on a panic-prone, implicit clamp inside the
/// trainer and gives callers a schema error for invalid explicit values.
#[cfg(test)]
fn effective_ivf_num_clusters(
    config: &DenseVectorConfig,
    corpus_count: usize,
    sample_count: usize,
) -> Result<usize> {
    if sample_count == 0 {
        return Err(Error::Schema(
            "cannot train an IVF vector index without sample vectors".to_string(),
        ));
    }

    effective_field_num_clusters(
        &IvfFieldConfig::Float(config.clone()),
        corpus_count,
        sample_count,
    )
}

impl<D: DirectoryWriter + 'static> IndexWriter<D> {
    /// Train vector index from accumulated Flat vectors (manual, not auto-triggered).
    ///
    /// 1. Acquires a stable segment snapshot.
    /// 2. Trains missing global codebooks.
    /// 3. Stages ANN replacements for every affected segment.
    /// 4. Publishes the complete segment/codebook generation atomically.
    pub async fn build_vector_index(&self) -> Result<()> {
        self.build_vector_generation(VectorGenerationMode::BuildMissing)
            .await
    }

    /// Train a fresh global codebook from the current corpus and rebuild every
    /// ANN segment into that generation. The replacement is atomic for search
    /// readers: the old segment/codebook pair remains live until all new files
    /// have been staged and durably committed together.
    pub async fn retrain_vector_index(&self) -> Result<()> {
        self.build_vector_generation(VectorGenerationMode::RetrainAll)
            .await
    }

    async fn build_vector_generation(&self, mode: VectorGenerationMode) -> Result<()> {
        let dense_fields = self.get_ivf_vector_fields();
        if dense_fields.is_empty() {
            log::info!("No dense vector fields configured for ANN indexing");
            return Ok(());
        }

        let artifact_update = self.segment_manager.begin_vector_artifact_update().await?;
        self.cleanup_unreferenced_vector_artifacts().await;

        let fields_to_train = match mode {
            VectorGenerationMode::BuildMissing => self.get_fields_to_build(&dense_fields).await,
            VectorGenerationMode::RetrainAll => dense_fields.clone(),
        };
        for (_, config) in &fields_to_train {
            validate_explicit_cluster_count(config.num_clusters())?;
        }

        let snapshot = self.segment_manager.acquire_snapshot().await;
        if snapshot.is_empty() {
            if mode == VectorGenerationMode::RetrainAll {
                return Err(Error::Schema(
                    "cannot retrain vector codebooks without committed segments".into(),
                ));
            }
            return Ok(());
        }

        let mut candidate_metadata = self.segment_manager.read_metadata(Clone::clone).await;
        if !fields_to_train.is_empty() {
            let total_vectors = self
                .count_vectors_for_training(
                    snapshot.segment_ids(),
                    &fields_to_train,
                    mode == VectorGenerationMode::BuildMissing,
                )
                .await?;
            let artifact_generation = SegmentId::new().to_hex();
            let updates = self
                .train_fields(
                    snapshot.segment_ids(),
                    &fields_to_train,
                    &total_vectors,
                    &artifact_generation,
                )
                .await?;
            for update in &updates {
                candidate_metadata.init_field(update.field_id, update.index_type);
                candidate_metadata.mark_field_built(
                    update.field_id,
                    update.vector_count,
                    update.num_clusters,
                    update.centroids_file.clone(),
                    update.codebook_file.clone(),
                );
            }
        }

        let target_field_ids = dense_fields
            .iter()
            .filter_map(|(field, _)| {
                candidate_metadata
                    .is_field_built(field.0)
                    .then_some(field.0)
            })
            .collect::<Vec<_>>();
        if target_field_ids.is_empty() {
            return Ok(());
        }

        let candidate_trained = super::IndexMetadata::try_load_trained_from_fields(
            &candidate_metadata.vector_fields,
            self.schema.as_ref(),
            self.directory.as_ref(),
        )
        .await?
        .map(Arc::new)
        .ok_or_else(|| Error::Internal("candidate vector generation has no artifacts".into()))?;

        let staged = self
            .segment_manager
            .stage_vector_generation(
                &artifact_update,
                snapshot.segment_ids(),
                &target_field_ids,
                Arc::clone(&candidate_trained),
                mode == VectorGenerationMode::RetrainAll,
            )
            .await?;
        self.segment_manager
            .publish_vector_generation(
                &artifact_update,
                candidate_metadata.vector_fields,
                candidate_trained,
                staged,
            )
            .await?;

        // Old readers retain the old snapshot and deserialized codebook. Once
        // this local training snapshot drops, retired source files can be
        // reclaimed. Reopening producers after the lease sees only the new set.
        drop(snapshot);
        drop(artifact_update);

        // A producer that started while training was gated writes flat data.
        // Catch already committed outputs; later commits carry their own
        // targeted upgrade marker in PreparedSegment.
        self.segment_manager
            .rewrite_vector_segments(&target_field_ids)
            .await?;
        self.cleanup_unreferenced_vector_artifacts().await;
        log::info!(
            "Dense vector ANN generation {:?} complete for {} field(s)",
            mode,
            target_field_ids.len(),
        );
        Ok(())
    }

    async fn train_fields(
        &self,
        segment_ids: &[String],
        fields: &[(Field, IvfFieldConfig)],
        total_vectors: &FxHashMap<u32, usize>,
        artifact_generation: &str,
    ) -> Result<Vec<TrainedFieldUpdate>> {
        let training_pool = self.segment_manager.background_cpu_pool();
        let mut missing = Vec::new();
        let mut updates = Vec::with_capacity(fields.len());
        for (field, config) in fields {
            // Sample collection and training are both field-serial. At most
            // one bounded sample, one field's clustering scratch, and one
            // generated artifact set can coexist.
            let corpus_count = total_vectors.get(&field.0).copied().unwrap_or(0);
            let Some(sample) = self
                .collect_training_sample(segment_ids, *field, config, corpus_count)
                .await?
            else {
                missing.push(field.0);
                continue;
            };
            let model = crate::segment::block_in_place_if_multithread(|| {
                training_pool.install(|| {
                    Self::train_field_model(
                        *field,
                        config,
                        &sample,
                        corpus_count,
                        artifact_generation,
                    )
                })
            })?;
            // Training artifacts own everything needed for persistence. Drop
            // the potentially multi-gigabyte sample before async file I/O.
            drop(sample);
            updates.push(self.save_trained_field(model).await?);
        }
        if updates.is_empty() && !fields.is_empty() {
            return Err(Error::Schema(format!(
                "cannot train vector codebooks: no committed vectors for field(s) {missing:?}"
            )));
        }
        if !missing.is_empty() {
            log::info!(
                "Skipping dense vector field(s) {missing:?}: the current corpus contains no vectors"
            );
        }
        Ok(updates)
    }

    /// Remove abandoned generation-qualified codebooks from cancelled or
    /// crash-interrupted attempts. The metadata references are the complete
    /// live set, and the exclusive update lease prevents another trainer from
    /// creating a candidate concurrently with this sweep.
    async fn cleanup_unreferenced_vector_artifacts(&self) {
        let referenced = self
            .segment_manager
            .read_metadata(|metadata| {
                metadata
                    .vector_fields
                    .values()
                    .flat_map(|field| {
                        field
                            .centroids_file
                            .iter()
                            .chain(field.codebook_file.iter())
                    })
                    .cloned()
                    .collect::<std::collections::HashSet<_>>()
            })
            .await;
        let files = match self.directory.list_files(std::path::Path::new("")).await {
            Ok(files) => files,
            Err(error) => {
                log::warn!("[trained] failed listing abandoned dense vector artifacts: {error}");
                return;
            }
        };
        for path in files {
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !name.starts_with(VECTOR_ARTIFACT_PREFIX)
                || referenced.contains(path.to_string_lossy().as_ref())
            {
                continue;
            }
            if let Err(error) = self.directory.delete(&path).await
                && error.kind() != std::io::ErrorKind::NotFound
            {
                log::warn!("[trained] failed deleting abandoned artifact {path:?}: {error}");
            }
        }
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    fn reject_ann_fields(ann_fields: &[u32], id_str: &str, field_ids: &[u32]) -> Result<()> {
        for &field_id in field_ids {
            if ann_fields.binary_search(&field_id).is_ok() {
                return Err(Error::Schema(format!(
                    "metadata-flat field {field_id} already has ANN data in segment {id_str}; \
                     recreate the index instead of mixing vector generations"
                )));
            }
        }
        Ok(())
    }

    /// Open only selected flat-vector fields plus the tiny segment metadata.
    /// Training does not need term dictionaries, stores, sparse structures, or
    /// corpus-sized ANN run columns, and must not pin those transient readers.
    async fn load_training_vectors(
        &self,
        segment_id: SegmentId,
        field_ids: &[u32],
    ) -> Result<crate::segment::reader::loader::VectorsFileData> {
        let files = SegmentFiles::new(segment_id.0);
        let meta_bytes = self
            .directory
            .open_read(&files.meta)
            .await?
            .read_bytes()
            .await?;
        let meta = SegmentMeta::deserialize(meta_bytes.as_slice())?;
        if meta.id != segment_id.0 {
            return Err(Error::Corruption(format!(
                "segment metadata ID {:032x} does not match file ID {}",
                meta.id,
                segment_id.to_hex(),
            )));
        }
        crate::segment::reader::loader::load_flat_vectors_file(
            self.directory.as_ref(),
            &files,
            self.schema.as_ref(),
            meta.num_docs,
            field_ids,
        )
        .await
    }

    /// Get all dense vector fields that need ANN indexes
    fn get_ivf_vector_fields(&self) -> Vec<(Field, IvfFieldConfig)> {
        self.schema
            .fields()
            .filter_map(|(field, entry)| {
                if entry.field_type == FieldType::DenseVector && entry.indexed {
                    entry
                        .dense_vector_config
                        .as_ref()
                        // Flat is a pre-build storage state; the production ANN
                        // path is trained once and shared by every segment.
                        .filter(|c| c.uses_ivf())
                        .map(|c| (field, IvfFieldConfig::Float(c.clone())))
                } else if entry.field_type == FieldType::BinaryDenseVector && entry.indexed {
                    entry
                        .binary_dense_vector_config
                        .as_ref()
                        .filter(|config| config.index_type == BinaryIndexType::Ivf)
                        .map(|config| (field, IvfFieldConfig::Binary(config.clone())))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get fields that need building (not already built)
    async fn get_fields_to_build(
        &self,
        dense_fields: &[(Field, IvfFieldConfig)],
    ) -> Vec<(Field, IvfFieldConfig)> {
        let field_ids: Vec<u32> = dense_fields.iter().map(|(f, _)| f.0).collect();
        let built: Vec<u32> = self
            .segment_manager
            .read_metadata(|meta| {
                field_ids
                    .iter()
                    .filter(|fid| meta.is_field_built(**fid))
                    .copied()
                    .collect()
            })
            .await;
        dense_fields
            .iter()
            .filter(|(field, _)| !built.contains(&field.0))
            .cloned()
            .collect()
    }

    /// Count every configured field without reading any vector payload bytes.
    async fn count_vectors_for_training(
        &self,
        segment_ids: &[String],
        fields_to_build: &[(Field, IvfFieldConfig)],
        require_flat_generation: bool,
    ) -> Result<FxHashMap<u32, usize>> {
        let mut total_vectors: FxHashMap<u32, usize> = FxHashMap::default();
        let field_ids: Vec<u32> = fields_to_build.iter().map(|(field, _)| field.0).collect();

        // Initial construction rejects
        // ANN payloads for metadata-flat fields; an explicit retrain reads the
        // exact flat vectors retained beside the current ANN generation.
        for id_str in segment_ids {
            let segment_id = SegmentId::from_hex(id_str)
                .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {}", id_str)))?;
            let vectors = self.load_training_vectors(segment_id, &field_ids).await?;

            if require_flat_generation {
                Self::reject_ann_fields(&vectors.ann_fields, id_str, &field_ids)?;
            }

            for (field, _) in fields_to_build {
                if let Some(flat) = vectors.flat_vectors.get(&field.0) {
                    let total = total_vectors.entry(field.0).or_default();
                    *total = total.checked_add(flat.num_vectors).ok_or_else(|| {
                        Error::Corruption(format!(
                            "vector count overflows usize for field {}",
                            field.0,
                        ))
                    })?;
                }
            }
        }
        Ok(total_vectors)
    }

    /// Fetch one deterministic, uniform field sample from the pinned segment
    /// snapshot. Only selected ranges are read; all other corpus vectors stay
    /// on disk. The caller trains and drops this sample before moving to the
    /// next field.
    async fn collect_training_sample(
        &self,
        segment_ids: &[String],
        field: Field,
        config: &IvfFieldConfig,
        total: usize,
    ) -> Result<Option<TrainingSample>> {
        if total == 0 {
            return Ok(None);
        }
        let bytes_per_sample = match config {
            IvfFieldConfig::Float(config) => config
                .dim
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| Error::Schema("float training vector size overflows".into()))?,
            IvfFieldConfig::Binary(config) => config.dim.div_ceil(8),
        };
        let limit = training_sample_limit(
            self.config.vector_training_max_samples,
            self.config.vector_training_memory_bytes,
            bytes_per_sample,
        )?;
        let take = total.min(limit);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
            0x4845_524d_4553_4956 ^ field.0 as u64 ^ total as u64,
        );
        let mut ordinals = Vec::with_capacity(take);
        if take == total {
            ordinals.extend(0..total);
        } else {
            let blocks = take.div_ceil(SAMPLE_BLOCK);
            for block in 0..blocks {
                let block_len = SAMPLE_BLOCK.min(take - ordinals.len());
                let stratum_start = block.saturating_mul(total) / blocks;
                let stratum_end = (block + 1).saturating_mul(total) / blocks;
                let latest_start = stratum_end.saturating_sub(block_len);
                let start = if latest_start > stratum_start {
                    rand::Rng::random_range(&mut rng, stratum_start..=latest_start)
                } else {
                    stratum_start
                };
                ordinals.extend(start..start + block_len);
            }
        }

        let mut sample = match config {
            IvfFieldConfig::Float(_) => TrainingSample::Float(Vec::with_capacity(take)),
            IvfFieldConfig::Binary(_) => TrainingSample::Binary(Vec::with_capacity(
                take.checked_mul(bytes_per_sample)
                    .ok_or_else(|| Error::Schema("binary training sample size overflows".into()))?,
            )),
        };
        let max_read_vectors = (MAX_SAMPLE_READ_BYTES / bytes_per_sample.max(1)).max(1);
        let mut global_offset = 0usize;
        let mut cursor = 0usize;
        let field_ids = [field.0];

        for id_str in segment_ids {
            let segment_id = SegmentId::from_hex(id_str)
                .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {id_str}")))?;
            let vectors = self.load_training_vectors(segment_id, &field_ids).await?;

            let Some(lazy_flat) = vectors.flat_vectors.get(&field.0) else {
                continue;
            };
            let base = global_offset;
            let end = base.checked_add(lazy_flat.num_vectors).ok_or_else(|| {
                Error::Corruption(format!("vector offset overflows for field {}", field.0))
            })?;
            global_offset = end;
            let first = cursor;
            while cursor < ordinals.len() && ordinals[cursor] < end {
                cursor += 1;
            }
            let selected = &ordinals[first..cursor];
            let mut run_start = 0;
            while run_start < selected.len() {
                let mut run_end = run_start + 1;
                while run_end < selected.len()
                    && run_end - run_start < max_read_vectors
                    && selected[run_end] == selected[run_end - 1] + 1
                {
                    run_end += 1;
                }
                let local_start = selected[run_start] - base;
                let run_len = run_end - run_start;
                let bytes = lazy_flat
                    .read_vectors_batch(local_start, run_len)
                    .await
                    .map_err(crate::Error::Io)?;
                match &mut sample {
                    TrainingSample::Binary(codes) => {
                        let expected = run_len.checked_mul(bytes_per_sample).ok_or_else(|| {
                            Error::Corruption("binary sample read size overflows".into())
                        })?;
                        if bytes.len() != expected {
                            return Err(Error::Corruption(format!(
                                "binary sample read returned {} bytes, expected {expected}",
                                bytes.len(),
                            )));
                        }
                        codes.extend_from_slice(bytes.as_slice());
                    }
                    TrainingSample::Float(vectors) => {
                        let dim = lazy_flat.dim;
                        let float_count = run_len.checked_mul(dim).ok_or_else(|| {
                            Error::Corruption("float sample read size overflows".into())
                        })?;
                        let mut decoded = vec![0.0; float_count];
                        crate::segment::dequantize_raw(
                            bytes.as_slice(),
                            lazy_flat.quantization,
                            decoded.len(),
                            &mut decoded,
                        )
                        .map_err(crate::Error::Io)?;
                        vectors.extend(decoded.chunks_exact(dim).map(<[f32]>::to_vec));
                    }
                }
                run_start = run_end;
            }
        }

        let collected = sample.len(config.dim());
        if global_offset != total || cursor != take || collected != take {
            return Err(Error::Corruption(format!(
                "training sample coverage mismatch for field {}: counted={total}, traversed={global_offset}, selected={cursor}, collected={collected}",
                field.0,
            )));
        }
        if collected < total {
            log::info!(
                "Sampled {} / {} dense vectors for field {} (max {} vectors / {} resident)",
                collected,
                total,
                field.0,
                self.config.vector_training_max_samples,
                crate::format_bytes(self.config.vector_training_memory_bytes as u64),
            );
        }
        Ok(Some(sample))
    }

    /// Train one field. Called from the shared bounded Rayon pool, so fields
    /// and each field's internal clustering work compose without extra pools.
    fn train_field_model(
        field: Field,
        config: &IvfFieldConfig,
        sample: &TrainingSample,
        corpus_count: usize,
        artifact_generation: &str,
    ) -> Result<TrainedFieldModel> {
        let field_id = field.0;
        let dim = config.dim();
        let sample_count = sample.len(dim);
        if sample_count == 0 || corpus_count == 0 {
            return Err(Error::Internal(format!(
                "empty training sample for non-empty field {field_id}"
            )));
        }
        let num_clusters = effective_field_num_clusters(config, corpus_count, sample_count)?;

        log::info!(
            "Training dense vector index for field {} with {} sampled / {} total vectors, {} clusters (dim={})",
            field_id,
            sample_count,
            corpus_count,
            num_clusters,
            dim,
        );

        let centroids_filename =
            format!("{VECTOR_ARTIFACT_PREFIX}{artifact_generation}_field_{field_id}_centroids.bin");
        let mut codebook_filename = None;

        let artifacts = match (config, sample) {
            (IvfFieldConfig::Float(config), TrainingSample::Float(vectors))
                if config.index_type == VectorIndexType::IvfPq =>
            {
                codebook_filename = Some(format!(
                    "{VECTOR_ARTIFACT_PREFIX}{artifact_generation}_field_{field_id}_codebook.bin"
                ));
                let (centroids, codebook, pq_sample_count) = Self::train_ivf_pq_model(
                    dim,
                    num_clusters,
                    config.ivf_routing,
                    config.soar.clone(),
                    vectors,
                );
                TrainedFieldArtifacts::Float {
                    centroids,
                    codebook,
                    pq_sample_count,
                }
            }
            (IvfFieldConfig::Float(config), TrainingSample::Float(vectors))
                if config.index_type == VectorIndexType::IvfTq =>
            {
                let mut coarse_config = crate::structures::CoarseConfig::new(dim, num_clusters)
                    .with_routing(config.ivf_routing);
                if let Some(soar) = config.soar.clone() {
                    coarse_config = coarse_config.with_soar(soar);
                }
                TrainedFieldArtifacts::FloatCentroids(crate::structures::CoarseCentroids::train(
                    &coarse_config,
                    vectors,
                ))
            }
            (IvfFieldConfig::Binary(config), TrainingSample::Binary(codes)) => {
                let mut binary_config = crate::structures::BinaryIvfConfig::new(dim, num_clusters);
                binary_config.max_train_samples = sample_count;
                binary_config.routing = config.ivf_routing;
                TrainedFieldArtifacts::Binary(
                    crate::structures::BinaryCoarseQuantizer::train(
                        binary_config,
                        codes,
                        sample_count,
                    )
                    .map_err(Error::Io)?,
                )
            }
            _ => {
                return Err(Error::Internal(format!(
                    "training sample kind does not match field {field_id}"
                )));
            }
        };

        let actual_num_clusters = match &artifacts {
            TrainedFieldArtifacts::Float { centroids, .. } => centroids.num_clusters as usize,
            TrainedFieldArtifacts::FloatCentroids(centroids) => centroids.num_clusters as usize,
            TrainedFieldArtifacts::Binary(quantizer) => quantizer.num_clusters as usize,
        };
        Ok(TrainedFieldModel {
            update: TrainedFieldUpdate {
                field_id,
                index_type: config.index_type(),
                vector_count: corpus_count,
                num_clusters: actual_num_clusters,
                centroids_file: centroids_filename,
                codebook_file: codebook_filename,
            },
            artifacts,
        })
    }

    async fn save_trained_field(&self, model: TrainedFieldModel) -> Result<TrainedFieldUpdate> {
        let TrainedFieldModel { update, artifacts } = model;
        match artifacts {
            TrainedFieldArtifacts::Float {
                centroids,
                codebook,
                pq_sample_count,
            } => {
                let codebook_file = update.codebook_file.as_deref().ok_or_else(|| {
                    Error::Internal(format!(
                        "trained IVF-PQ field {} has no codebook filename",
                        update.field_id,
                    ))
                })?;
                tokio::try_join!(
                    self.save_trained_artifact(&centroids, &update.centroids_file),
                    self.save_trained_artifact(&codebook, codebook_file),
                )?;
                log::info!(
                    "Saved IVF-PQ artifacts for field {} ({} clusters, {} PQ samples)",
                    update.field_id,
                    centroids.num_clusters,
                    pq_sample_count,
                );
            }
            TrainedFieldArtifacts::FloatCentroids(centroids) => {
                self.save_trained_artifact(&centroids, &update.centroids_file)
                    .await?;
                log::info!(
                    "Saved IVF-TQ coarse artifact for field {} ({} clusters; leaf codec is derived)",
                    update.field_id,
                    centroids.num_clusters,
                );
            }
            TrainedFieldArtifacts::Binary(quantizer) => {
                self.save_trained_artifact(&quantizer, &update.centroids_file)
                    .await?;
                log::info!(
                    "Saved binary IVF artifact for field {} ({} clusters)",
                    update.field_id,
                    quantizer.num_clusters,
                );
            }
        }
        Ok(update)
    }

    /// Serialize a trained structure to bincode and save to an index-level file.
    async fn save_trained_artifact(
        &self,
        artifact: &impl serde::Serialize,
        filename: &str,
    ) -> Result<()> {
        let temp_filename = format!("{filename}.tmp");
        let temp_path = std::path::Path::new(&temp_filename);
        let final_path = std::path::Path::new(filename);
        let mut writer = self.directory.streaming_writer(temp_path).await?;
        let encode_result = {
            let mut limited = SizeLimitedWriter::new(
                writer.as_mut(),
                super::metadata::MAX_TRAINED_ARTIFACT_BYTES,
            );
            bincode::serde::encode_into_std_write(
                artifact,
                &mut limited,
                bincode::config::standard(),
            )
        };
        if let Err(error) = encode_result {
            drop(writer);
            let _ = self.directory.delete(temp_path).await;
            return Err(Error::Serialization(format!(
                "failed to serialize trained artifact '{filename}': {error}"
            )));
        }
        if let Err(error) = writer.finish() {
            let _ = self.directory.delete(temp_path).await;
            return Err(Error::Io(error));
        }
        if let Err(error) = self.directory.rename(temp_path, final_path).await {
            let _ = self.directory.delete(temp_path).await;
            return Err(Error::Io(error));
        }
        self.directory.sync().await?;
        Ok(())
    }

    /// Train the global IVF-PQ centroids and residual codebook. The caller is
    /// already running inside the bounded background Rayon pool.
    fn train_ivf_pq_model(
        dim: usize,
        num_clusters: usize,
        routing: crate::dsl::IvfRoutingMode,
        soar: Option<crate::structures::SoarConfig>,
        vectors: &[Vec<f32>],
    ) -> (
        crate::structures::CoarseCentroids,
        crate::structures::PQCodebook,
        usize,
    ) {
        let mut coarse_config =
            crate::structures::CoarseConfig::new(dim, num_clusters).with_routing(routing);
        if let Some(soar) = soar {
            coarse_config = coarse_config.with_soar(soar);
        }
        let centroids = crate::structures::CoarseCentroids::train(&coarse_config, vectors);

        // Faiss' established PQ training ceiling is 256 samples per one-byte
        // subquantizer centroid. More points multiply every subspace's Lloyd
        // work without improving the 256-way codebook materially. Train on
        // residuals, because segment encoding also quantizes x - coarse(x).
        const PQ_CENTROIDS: usize = 256;
        const PQ_TRAINING_POINTS_PER_CENTROID: usize = 256;
        let pq_sample_count = vectors
            .len()
            .min(PQ_CENTROIDS * PQ_TRAINING_POINTS_PER_CENTROID);
        use rayon::prelude::*;
        let pq_residuals = (0..pq_sample_count)
            .into_par_iter()
            .map(|sample_index| {
                let vector_index = sample_index.saturating_mul(vectors.len()) / pq_sample_count;
                let vector = &vectors[vector_index];
                let cluster = centroids
                    .probe(vector, 1, routing)
                    .cluster_ids
                    .first()
                    .copied()
                    .unwrap_or(0);
                centroids.compute_residual(vector, cluster)
            })
            .collect::<Vec<_>>();
        let pq_config = crate::structures::PQConfig::new(dim);
        let codebook = crate::structures::PQCodebook::train(pq_config, &pq_residuals, 10);
        (centroids, codebook, pq_sample_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ivf_config(num_clusters: Option<usize>) -> DenseVectorConfig {
        DenseVectorConfig::with_ivf_pq(8, num_clusters, 4)
    }

    #[test]
    fn effective_clusters_follow_corpus_heuristic_but_fit_sample() {
        let config = ivf_config(None);

        assert_eq!(
            effective_ivf_num_clusters(&config, 1_000_000, 73).unwrap(),
            16
        );
        assert_eq!(
            effective_ivf_num_clusters(&config, 10_000, 1_000).unwrap(),
            25
        );
    }

    #[test]
    fn effective_clusters_clamp_explicit_value_to_sample() {
        let config = ivf_config(Some(256));
        assert_eq!(
            effective_ivf_num_clusters(&config, 1_000_000, 17).unwrap(),
            17
        );
    }

    #[test]
    fn effective_clusters_reject_invalid_explicit_bounds() {
        let zero = effective_ivf_num_clusters(&ivf_config(Some(0)), 10_000, 100)
            .unwrap_err()
            .to_string();
        assert!(zero.contains("at least 1"));

        let too_many =
            effective_ivf_num_clusters(&ivf_config(Some(MAX_IVF_CLUSTERS + 1)), 10_000, 100)
                .unwrap_err()
                .to_string();
        assert!(too_many.contains("must not exceed 1048576"));
    }

    #[test]
    fn effective_clusters_reject_empty_training_sample() {
        let error = effective_ivf_num_clusters(&ivf_config(None), 10_000, 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("without sample vectors"));
    }

    #[test]
    fn training_sample_limit_honors_both_cli_bounds() {
        assert_eq!(training_sample_limit(10_000_000, 4_096, 4).unwrap(), 1_024);
        assert_eq!(training_sample_limit(100, 4_096, 4).unwrap(), 100);
        let error = training_sample_limit(100, 3, 4).unwrap_err().to_string();
        assert!(error.contains("cannot hold one"), "{error}");
    }

    #[test]
    fn artifact_writer_enforces_limit_without_writing_past_it() {
        let mut output = Vec::new();
        let mut writer = SizeLimitedWriter::new(&mut output, 3);
        writer.write_all(&[1, 2]).unwrap();
        let error = writer.write_all(&[3, 4]).unwrap_err().to_string();
        assert!(error.contains("3-byte safety limit"), "{error}");
        assert_eq!(output, vec![1, 2]);
    }

    // ===== rebuild destructive-downgrade regression tests =====

    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::directories::{
        Directory, DirectoryWriter as DirectoryWriterTrait, FileHandle, RamDirectory, RangeReadFn,
    };
    use crate::dsl::{Document, SchemaBuilder};
    use crate::index::{IndexConfig, IndexWriter};

    const READ_FAIL_DOCS: usize = 5;
    const READ_FAIL_DIM: usize = 4;
    /// Flat entry layout of a single-field, flat-only `.vectors` file written
    /// by the segment builder (data-first format): header (16 bytes) + raw f32
    /// vectors + doc-id map + TOC + footer. Only the raw vector region is read
    /// by training collection; segment open touches the header, doc-id map,
    /// TOC, and footer, which all live outside this byte range.
    const VEC_REGION_START: u64 = 16;
    const VEC_REGION_END: u64 = VEC_REGION_START + (READ_FAIL_DOCS * READ_FAIL_DIM * 4) as u64;

    /// RamDirectory wrapper whose `.vectors` handles fail range reads of the
    /// raw vector region while `fail_vector_reads` is armed. Segment open
    /// keeps succeeding, so exactly the training-collection batch reads fail —
    /// the I/O the rebuild path used to swallow with `if let Ok`.
    #[derive(Clone, Default)]
    struct VectorReadFailDirectory {
        inner: RamDirectory,
        fail_vector_reads: Arc<AtomicBool>,
        fail_all_vector_reads: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl Directory for VectorReadFailDirectory {
        async fn exists(&self, path: &Path) -> std::io::Result<bool> {
            self.inner.exists(path).await
        }

        async fn file_size(&self, path: &Path) -> std::io::Result<u64> {
            self.inner.file_size(path).await
        }

        async fn open_read(&self, path: &Path) -> std::io::Result<FileHandle> {
            self.inner.open_read(path).await
        }

        async fn read_range(
            &self,
            path: &Path,
            range: std::ops::Range<u64>,
        ) -> std::io::Result<crate::directories::OwnedBytes> {
            self.inner.read_range(path, range).await
        }

        async fn list_files(&self, prefix: &Path) -> std::io::Result<Vec<std::path::PathBuf>> {
            self.inner.list_files(prefix).await
        }

        async fn open_lazy(&self, path: &Path) -> std::io::Result<FileHandle> {
            let handle = self.inner.open_lazy(path).await?;
            if path.extension().is_some_and(|ext| ext == "vectors") {
                let armed = Arc::clone(&self.fail_vector_reads);
                let fail_all = Arc::clone(&self.fail_all_vector_reads);
                let len = handle.len();
                let read_fn: RangeReadFn = Arc::new(move |range: std::ops::Range<u64>| {
                    let handle = handle.clone();
                    let armed = Arc::clone(&armed);
                    let fail_all = Arc::clone(&fail_all);
                    Box::pin(async move {
                        if fail_all.load(Ordering::SeqCst)
                            || (armed.load(Ordering::SeqCst)
                                && range.start >= VEC_REGION_START
                                && range.end <= VEC_REGION_END)
                        {
                            return Err(std::io::Error::other("injected vector data read failure"));
                        }
                        handle.read_bytes_range(range).await
                    })
                });
                return Ok(FileHandle::lazy(len, read_fn));
            }
            Ok(handle)
        }
    }

    #[async_trait::async_trait]
    impl DirectoryWriterTrait for VectorReadFailDirectory {
        async fn write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
            self.inner.write(path, data).await
        }

        async fn delete(&self, path: &Path) -> std::io::Result<()> {
            self.inner.delete(path).await
        }

        async fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
            self.inner.rename(from, to).await
        }

        async fn sync(&self) -> std::io::Result<()> {
            self.inner.sync().await
        }

        async fn streaming_writer(
            &self,
            path: &Path,
        ) -> std::io::Result<Box<dyn crate::directories::StreamingWriter>> {
            self.inner.streaming_writer(path).await
        }
    }

    /// A failed read from the flat staging generation must abort training
    /// before artifacts or Built metadata are published.
    #[tokio::test]
    async fn build_propagates_vector_read_errors_without_publishing_artifacts() {
        let mut sb = SchemaBuilder::default();
        let embedding = sb.add_dense_vector_field_with_config(
            "embedding",
            true,
            true,
            DenseVectorConfig::with_ivf_pq(READ_FAIL_DIM, Some(1), 1),
        );
        let schema = sb.build();

        let dir = VectorReadFailDirectory::default();
        let config = IndexConfig {
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            num_indexing_threads: 1,
            ..Default::default()
        };
        let mut writer = IndexWriter::create(dir.clone(), schema, config)
            .await
            .unwrap();
        for i in 0..READ_FAIL_DOCS {
            let mut doc = Document::new();
            doc.add_dense_vector(embedding, vec![i as f32 + 1.0; READ_FAIL_DIM]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        dir.fail_vector_reads.store(true, Ordering::SeqCst);
        let error = writer
            .build_vector_index()
            .await
            .expect_err("failed sample collection must fail the build")
            .to_string();
        assert!(
            error.contains("injected vector data read failure"),
            "{error}"
        );

        assert!(
            !writer
                .segment_manager
                .read_metadata(|meta| meta.is_field_built(embedding.0))
                .await,
            "a failed build must not publish Built metadata"
        );
        assert!(
            writer.segment_manager.trained().is_none(),
            "a failed build must not publish trained artifacts"
        );
    }

    #[tokio::test]
    async fn retrain_read_failure_keeps_the_complete_published_generation() {
        let mut sb = SchemaBuilder::default();
        let embedding = sb.add_dense_vector_field_with_config(
            "embedding",
            true,
            true,
            DenseVectorConfig::with_ivf_pq(READ_FAIL_DIM, Some(1), 1),
        );
        let schema = sb.build();
        let dir = VectorReadFailDirectory::default();
        let config = IndexConfig {
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            num_indexing_threads: 1,
            ..Default::default()
        };
        let mut writer = IndexWriter::create(dir.clone(), schema, config)
            .await
            .unwrap();
        for i in 0..READ_FAIL_DOCS {
            let mut doc = Document::new();
            doc.add_dense_vector(embedding, vec![i as f32 + 1.0; READ_FAIL_DIM]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        writer.build_vector_index().await.unwrap();

        let old_ids = writer.segment_manager.get_segment_ids().await;
        let old_meta = writer
            .segment_manager
            .read_metadata(|metadata| metadata.get_field_meta(embedding.0).cloned())
            .await
            .unwrap();
        let old_version = writer.segment_manager.trained().unwrap().codebooks[&embedding.0].version;

        dir.fail_all_vector_reads.store(true, Ordering::SeqCst);
        let error = writer
            .retrain_vector_index()
            .await
            .expect_err("failed sample collection must abort the retrain")
            .to_string();
        assert!(
            error.contains("injected vector data read failure"),
            "{error}"
        );
        assert_eq!(writer.segment_manager.get_segment_ids().await, old_ids);
        assert_eq!(
            writer
                .segment_manager
                .read_metadata(|metadata| metadata
                    .get_field_meta(embedding.0)
                    .map(|field| (field.centroids_file.clone(), field.codebook_file.clone())))
                .await,
            Some((old_meta.centroids_file, old_meta.codebook_file)),
        );
        assert_eq!(
            writer.segment_manager.trained().unwrap().codebooks[&embedding.0].version,
            old_version,
        );
    }
}
