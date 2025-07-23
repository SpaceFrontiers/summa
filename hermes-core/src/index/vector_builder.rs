//! Vector index building for IndexWriter
//!
//! Training is **manual-only** — decoupled from commit.
//! `build_vector_index()` trains missing coarse-centroid generations;
//! `retrain_vector_index()` replaces them. Both finish every committed ANN
//! segment. Leaf codecs (TurboQuant) are derived, never trained.

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
/// Faiss-style clustering ceiling: more points per centroid multiply Lloyd
/// cost without materially improving the codebook.
const COARSE_TRAINING_POINTS_PER_CENTROID: usize = 256;
/// Bound transient I/O/dequantization buffers independently of the configured
/// total training sample budget.
const MAX_SAMPLE_READ_BYTES: usize = 64 * 1024 * 1024;
/// Coalesce nearby point-sample reads only while the extra I/O remains bounded.
/// This keeps point-level sampling statistically useful without turning a dense
/// sample into one range read per vector.
const MAX_SAMPLE_READ_AMPLIFICATION: usize = 4;
/// A held-out sample is large enough to expose routing/occupancy tails but stays
/// bounded when codebooks are trained from millions of points.
const VALIDATION_SAMPLE_DENOMINATOR: usize = 10;
const MAX_VALIDATION_SAMPLES: usize = 65_536;
/// Bound the exact centroid scan used to measure router recall. Even for very
/// large codebooks, retain at least one held-out row when the sample permits.
const MAX_VALIDATION_COORDINATE_WORK: usize = 512_000_000;
/// A second deterministic initialization is useful on modest training jobs.
/// Large jobs retain the same quality-report scaffold without silently doubling
/// their already substantial Lloyd cost.
const MODEL_SELECTION_SEEDS: [u64; 2] = [42, 0x9e37_79b9_7f4a_7c15];
const MAX_MULTI_SEED_COORDINATE_WORK: usize = 4_000_000_000;
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
    /// Contiguous row-major matrix, retained in this form through training.
    Float(Vec<f32>),
    Binary(Vec<u8>),
}

#[derive(Clone, Copy, Debug)]
struct OccupancyQuality {
    p95: usize,
    p99: usize,
    max: usize,
    empty: usize,
    penalty: f64,
}

#[derive(Clone, Copy, Debug)]
struct FloatBuildQuality {
    objective: f64,
    mean_exact_distortion: f64,
    mean_routed_distortion: f64,
    router_recall_at_1: f64,
    mean_construction_assignments: f64,
    residual_p50: f32,
    residual_p95: f32,
    residual_p99: f32,
    occupancy: OccupancyQuality,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VectorGenerationMode {
    BuildMissing,
    RetrainAll,
}

impl TrainingSample {
    fn len(&self, dim: usize) -> usize {
        match self {
            Self::Float(values) => values.len() / dim,
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

/// Resolve the codebook size against the complete configured sample budget,
/// then select that final training sample directly. In particular, callers no
/// longer collect a larger block-correlated sample and stride-thin it later.
fn final_training_sample_count(
    config: &IvfFieldConfig,
    corpus_count: usize,
    sample_limit: usize,
) -> Result<usize> {
    let available = corpus_count.min(sample_limit);
    if available == 0 {
        return Ok(0);
    }
    let clusters = effective_field_num_clusters(config, corpus_count, available)?;
    Ok(available.min(clusters.saturating_mul(COARSE_TRAINING_POINTS_PER_CENTROID)))
}

/// Uniform point sample without replacement, sorted only after selection so
/// storage reads remain monotonic. `rand::seq::index::sample` uses bounded
/// memory proportional to the selected set rather than materializing a corpus
/// permutation.
fn deterministic_sample_ordinals(total: usize, take: usize, seed: u64) -> Vec<usize> {
    debug_assert!(take <= total);
    if take == 0 {
        return Vec::new();
    }
    if take == total {
        return (0..total).collect();
    }
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
    let mut ordinals = rand::seq::index::sample(&mut rng, total, take).into_vec();
    ordinals.sort_unstable();
    ordinals
}

fn validation_sample_count(
    sample_count: usize,
    num_clusters: usize,
    values_per_vector: usize,
) -> usize {
    if sample_count <= num_clusters {
        return 0;
    }
    let quality_floor = num_clusters.saturating_mul(MIN_TRAINING_POINTS_PER_CENTROID);
    let minimum_training = if sample_count >= quality_floor {
        quality_floor
    } else {
        num_clusters
    };
    let exact_scan_limit = MAX_VALIDATION_COORDINATE_WORK
        .checked_div(num_clusters.saturating_mul(values_per_vector).max(1))
        .unwrap_or(0)
        .max(1);
    sample_count
        .div_ceil(VALIDATION_SAMPLE_DENOMINATOR)
        .clamp(1, MAX_VALIDATION_SAMPLES)
        .min(exact_scan_limit)
        .min(sample_count - minimum_training)
}

fn model_selection_seeds(
    training_count: usize,
    num_clusters: usize,
    dim: usize,
    has_validation: bool,
) -> &'static [u64] {
    if !has_validation {
        return &MODEL_SELECTION_SEEDS[..1];
    }
    let distance_passes = crate::structures::vector::estimated_euclidean_kmeans_distance_multiplier(
        training_count,
        num_clusters,
        25,
    );
    let work = training_count
        .saturating_mul(num_clusters)
        .saturating_mul(dim)
        .saturating_mul(distance_passes)
        .saturating_mul(MODEL_SELECTION_SEEDS.len());
    if work <= MAX_MULTI_SEED_COORDINATE_WORK {
        &MODEL_SELECTION_SEEDS
    } else {
        &MODEL_SELECTION_SEEDS[..1]
    }
}

fn percentile_index(len: usize, percentile: usize) -> usize {
    debug_assert!(len > 0 && percentile <= 100);
    (len - 1).saturating_mul(percentile).div_ceil(100)
}

fn occupancy_quality(mut counts: Vec<usize>, observations: usize) -> OccupancyQuality {
    if counts.is_empty() {
        return OccupancyQuality {
            p95: 0,
            p99: 0,
            max: 0,
            empty: 0,
            penalty: 0.0,
        };
    }
    counts.sort_unstable();
    let p95 = counts[percentile_index(counts.len(), 95)];
    let p99 = counts[percentile_index(counts.len(), 99)];
    let max = counts.last().copied().unwrap_or(0);
    let empty = counts.partition_point(|&count| count == 0);
    let expected = observations as f64 / counts.len() as f64;
    let denominator = expected.max(1.0);
    let p99_excess = (p99 as f64 / denominator - 1.0).max(0.0);
    let max_excess = (max as f64 / denominator - 1.0).max(0.0);
    let empty_fraction = empty as f64 / counts.len() as f64;
    // Distortion remains the dominant selection signal. These terms only
    // reject seeds with materially worse posting-list tails at similar error.
    let penalty = 0.02 * p99_excess + 0.005 * max_excess + 0.05 * empty_fraction;
    OccupancyQuality {
        p95,
        p99,
        max,
        empty,
        penalty,
    }
}

fn float_model_selection_objective(
    mean_exact_distortion: f64,
    mean_routed_distortion: f64,
    occupancy_penalty: f64,
) -> f64 {
    let scale = mean_exact_distortion.max(f64::from(f32::EPSILON));
    let routed_distortion_excess = (mean_routed_distortion - mean_exact_distortion).max(0.0);
    mean_exact_distortion + scale * occupancy_penalty + routed_distortion_excess
}

/// Move a deterministic uniform holdout into the matrix suffix and return the
/// element offset separating training and validation rows. Row swaps avoid a
/// second vector allocation and keep the original sample capacity available to
/// the trainer.
fn partition_contiguous_holdout_suffix<T>(
    values: &mut [T],
    values_per_vector: usize,
    validation_count: usize,
    seed: u64,
) -> usize {
    assert!(values_per_vector > 0);
    assert_eq!(values.len() % values_per_vector, 0);
    let sample_count = values.len() / values_per_vector;
    assert!(validation_count <= sample_count);
    if validation_count == 0 {
        return values.len();
    }
    let validation_indices =
        deterministic_sample_ordinals(sample_count, validation_count, seed ^ 0x5641_4c49_4441_5445);
    let split_row = sample_count - validation_count;
    let prefix_validation_count = validation_indices.partition_point(|&index| index < split_row);
    let mut right = sample_count;
    for &left in &validation_indices[..prefix_validation_count] {
        loop {
            right -= 1;
            if validation_indices.binary_search(&right).is_err() {
                break;
            }
        }
        debug_assert!(right >= split_row);
        for component in 0..values_per_vector {
            values.swap(
                left * values_per_vector + component,
                right * values_per_vector + component,
            );
        }
    }
    split_row * values_per_vector
}

fn evaluate_float_build_quality(
    centroids: &crate::structures::CoarseCentroids,
    validation: &[f32],
    routing: crate::dsl::IvfRoutingMode,
) -> Option<FloatBuildQuality> {
    let dim = centroids.dim;
    let validation_count = validation.len() / dim;
    if validation_count == 0 {
        return None;
    }
    let mut occupancy = vec![0usize; centroids.num_clusters as usize];
    let mut residual_scales = Vec::with_capacity(validation_count);
    let mut exact_distortion_sum = 0.0f64;
    let mut routed_distortion_sum = 0.0f64;
    let mut router_hits = 0usize;
    let mut construction_assignments = 0usize;
    let effective_routing = crate::structures::vector::ivf::routing::effective_routing_mode(
        routing,
        centroids.num_clusters as usize,
    );
    let exact_routing = effective_routing == crate::dsl::IvfRoutingMode::Flat;
    for vector in validation.chunks_exact(dim) {
        let (exact_cluster_id, routed_cluster_id) = if exact_routing {
            if centroids.soar_config.is_some() {
                // Flat SOAR already performs the exact all-centroid pass needed
                // for its primary and secondary assignments. Its primary is
                // therefore both the exact and query-routed nearest centroid.
                let construction_assignment = centroids.assign_with_routing(vector, routing);
                let exact_cluster_id = construction_assignment.primary_cluster;
                for cluster_id in construction_assignment.all_clusters() {
                    occupancy[cluster_id as usize] += 1;
                    construction_assignments += 1;
                }
                (exact_cluster_id, exact_cluster_id)
            } else {
                // With neither an approximate router nor SOAR, one exact pass
                // supplies exact quality, query routing, and construction
                // occupancy.
                let exact_cluster_id = centroids.find_nearest(vector);
                occupancy[exact_cluster_id as usize] += 1;
                construction_assignments += 1;
                (exact_cluster_id, exact_cluster_id)
            }
        } else {
            let exact_cluster_id = centroids.find_nearest(vector);
            let routed_cluster_id = centroids.probe(vector, 1, routing).cluster_ids[0];
            let construction_assignment = centroids.assign_with_routing(vector, routing);
            for cluster_id in construction_assignment.all_clusters() {
                occupancy[cluster_id as usize] += 1;
                construction_assignments += 1;
            }
            (exact_cluster_id, routed_cluster_id)
        };
        router_hits += usize::from(routed_cluster_id == exact_cluster_id);
        let exact_distance = vector
            .iter()
            .zip(centroids.get_centroid(exact_cluster_id))
            .map(|(&value, &center)| {
                let delta = value - center;
                delta * delta
            })
            .sum::<f32>();
        let routed_distance = if routed_cluster_id == exact_cluster_id {
            exact_distance
        } else {
            vector
                .iter()
                .zip(centroids.get_centroid(routed_cluster_id))
                .map(|(&value, &center)| {
                    let delta = value - center;
                    delta * delta
                })
                .sum::<f32>()
        };
        exact_distortion_sum += f64::from(exact_distance);
        routed_distortion_sum += f64::from(routed_distance);
        residual_scales.push(exact_distance.max(0.0).sqrt());
    }
    residual_scales.sort_unstable_by(f32::total_cmp);
    let occupancy = occupancy_quality(occupancy, construction_assignments);
    let mean_exact_distortion = exact_distortion_sum / validation_count as f64;
    let mean_routed_distortion = routed_distortion_sum / validation_count as f64;
    let router_recall_at_1 = router_hits as f64 / validation_count as f64;
    let mean_construction_assignments = construction_assignments as f64 / validation_count as f64;
    // Keep exact codebook distortion as the primary signal. Price approximate
    // routing by its measured excess distortion rather than treating all
    // misses equally; retain recall@1 as a separately reported diagnostic.
    let objective = float_model_selection_objective(
        mean_exact_distortion,
        mean_routed_distortion,
        occupancy.penalty,
    );
    Some(FloatBuildQuality {
        objective,
        mean_exact_distortion,
        mean_routed_distortion,
        router_recall_at_1,
        mean_construction_assignments,
        residual_p50: residual_scales[percentile_index(validation_count, 50)],
        residual_p95: residual_scales[percentile_index(validation_count, 95)],
        residual_p99: residual_scales[percentile_index(validation_count, 99)],
        occupancy,
    })
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
    /// 2. Trains missing coarse-centroid generations.
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
            log::info!(
                "[vector_training] no dense vector fields configured for ANN indexing: index={}",
                self.schema.index_label()
            );
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
                    "cannot retrain vector centroids without committed segments".into(),
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
            "[vector_training] ANN generation {:?} complete: index={} {} field(s)",
            mode,
            self.schema.index_label(),
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
        let index_label = self.schema.index_label();
        let mut missing = Vec::new();
        let mut updates = Vec::with_capacity(fields.len());
        for (field, config) in fields {
            // Sample collection and training are both field-serial. At most
            // one bounded sample, one field's clustering scratch, and one
            // generated artifact set can coexist.
            let corpus_count = total_vectors.get(&field.0).copied().unwrap_or(0);
            let Some(mut sample) = self
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
                        &mut sample,
                        corpus_count,
                        artifact_generation,
                        index_label,
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
                "cannot train vector centroids: no committed vectors for field(s) {missing:?}"
            )));
        }
        if !missing.is_empty() {
            log::info!(
                "[vector_training] skipping dense vector field(s) {missing:?}: index={index_label} has no vectors in the current corpus"
            );
        }
        Ok(updates)
    }

    /// Remove abandoned generation-qualified artifacts from cancelled or
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
                log::warn!(
                    "[trained] index={} failed listing abandoned dense vector artifacts: {error}",
                    self.schema.index_label()
                );
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
                log::warn!(
                    "[trained] index={} failed deleting abandoned artifact {path:?}: {error}",
                    self.schema.index_label()
                );
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
        let take = final_training_sample_count(config, total, limit)?;
        let sample_seed = 0x4845_524d_4553_4956 ^ field.0 as u64 ^ total as u64;
        let ordinals = deterministic_sample_ordinals(total, take, sample_seed);

        let mut sample = match config {
            IvfFieldConfig::Float(config) => TrainingSample::Float(Vec::with_capacity(
                take.checked_mul(config.dim)
                    .ok_or_else(|| Error::Schema("float training sample size overflows".into()))?,
            )),
            IvfFieldConfig::Binary(_) => TrainingSample::Binary(Vec::with_capacity(
                take.checked_mul(bytes_per_sample)
                    .ok_or_else(|| Error::Schema("binary training sample size overflows".into()))?,
            )),
        };
        let max_read_vectors = (MAX_SAMPLE_READ_BYTES / bytes_per_sample.max(1)).max(1);
        let mut zero_codes = 0usize;
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
                while run_end < selected.len() {
                    let selected_count = run_end - run_start + 1;
                    let span = selected[run_end] - selected[run_start] + 1;
                    if span > max_read_vectors
                        || span > selected_count.saturating_mul(MAX_SAMPLE_READ_AMPLIFICATION)
                    {
                        break;
                    }
                    run_end += 1;
                }
                let local_start = selected[run_start] - base;
                let read_len = selected[run_end - 1] - selected[run_start] + 1;
                let bytes = lazy_flat
                    .read_vectors_batch(local_start, read_len)
                    .await
                    .map_err(crate::Error::Io)?;
                match &mut sample {
                    TrainingSample::Binary(codes) => {
                        let expected = read_len.checked_mul(bytes_per_sample).ok_or_else(|| {
                            Error::Corruption("binary sample read size overflows".into())
                        })?;
                        if bytes.len() != expected {
                            return Err(Error::Corruption(format!(
                                "binary sample read returned {} bytes, expected {expected}",
                                bytes.len(),
                            )));
                        }
                        for &ordinal in &selected[run_start..run_end] {
                            let relative = ordinal - selected[run_start];
                            let offset = relative * bytes_per_sample;
                            let code = &bytes.as_slice()[offset..offset + bytes_per_sample];
                            // All-zero codes are never indexed, so training on
                            // them only spends centroids that can never be
                            // assigned: a production field turned ~30% of a
                            // 163k codebook into duplicate zero centroids.
                            if code.iter().all(|&byte| byte == 0) {
                                zero_codes += 1;
                                continue;
                            }
                            codes.extend_from_slice(code);
                        }
                    }
                    TrainingSample::Float(values) => {
                        let dim = lazy_flat.dim;
                        let float_count = read_len.checked_mul(dim).ok_or_else(|| {
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
                        for &ordinal in &selected[run_start..run_end] {
                            let relative = ordinal - selected[run_start];
                            let offset = relative * dim;
                            values.extend_from_slice(&decoded[offset..offset + dim]);
                        }
                    }
                }
                run_start = run_end;
            }
        }

        let collected = sample.len(config.dim());
        // Coverage is checked against what was *selected*; withheld all-zero
        // codes are subtracted explicitly so a real traversal bug still trips.
        if global_offset != total || cursor != take || collected + zero_codes != take {
            return Err(Error::Corruption(format!(
                "training sample coverage mismatch for field {}: counted={total}, traversed={global_offset}, selected={cursor}, collected={collected}, zero={zero_codes}",
                field.0,
            )));
        }
        if zero_codes > 0 {
            log::warn!(
                "[vector_training] index={} field={}: {zero_codes} of {take} sampled vectors \
                 ({:.1}%) are all-zero and were excluded from training — they cannot be assigned \
                 to any leaf, so training on them only wastes centroids",
                self.schema.index_label(),
                field.0,
                100.0 * zero_codes as f64 / take.max(1) as f64,
            );
            crate::observe::binary_zero_vectors(
                self.schema.index_label(),
                field.0,
                zero_codes,
                take,
            );
        }
        if collected == 0 {
            log::warn!(
                "[vector_training] index={} field={}: every sampled vector is all-zero; \
                 skipping ANN training for this field",
                self.schema.index_label(),
                field.0,
            );
            return Ok(None);
        }
        if collected < total {
            log::info!(
                "[vector_training] sampled {} / {} dense vectors: index={} field={} (max {} vectors / {} resident)",
                collected,
                total,
                self.schema.index_label(),
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
        sample: &mut TrainingSample,
        corpus_count: usize,
        artifact_generation: &str,
        index_label: &str,
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
            "[vector_training] training model: index={} field={} with {} sampled / {} total vectors, {} clusters (dim={})",
            index_label,
            field_id,
            sample_count,
            corpus_count,
            num_clusters,
            dim,
        );

        let centroids_filename =
            format!("{VECTOR_ARTIFACT_PREFIX}{artifact_generation}_field_{field_id}_centroids.bin");

        let artifacts = match (config, sample) {
            (IvfFieldConfig::Float(config), TrainingSample::Float(values))
                if config.index_type == VectorIndexType::IvfTq =>
            {
                values
                    .chunks_exact_mut(dim)
                    .for_each(crate::structures::vector::ivf::routing::normalize_cosine_in_place);
                let candidate_validation_count =
                    validation_sample_count(sample_count, num_clusters, dim);
                let candidate_training_count = sample_count - candidate_validation_count;
                let seeds = model_selection_seeds(
                    candidate_training_count,
                    num_clusters,
                    dim,
                    candidate_validation_count > 0,
                );
                let split_seed =
                    MODEL_SELECTION_SEEDS[0] ^ u64::from(field_id) ^ corpus_count as u64;
                let split = if seeds.len() > 1 {
                    partition_contiguous_holdout_suffix(
                        values.as_mut_slice(),
                        dim,
                        candidate_validation_count,
                        split_seed,
                    )
                } else {
                    values.len()
                };
                let (training_values, validation) = values.as_slice().split_at(split);
                let training_count = training_values.len() / dim;
                let validation_count = validation.len() / dim;
                if seeds.len() > 1 {
                    log::info!(
                        "[vector_training] model selection: index={index_label} field={field_id}, {} training + {} held-out vectors, {} deterministic centroid seed(s)",
                        training_count,
                        validation_count,
                        seeds.len(),
                    );
                } else {
                    log::info!(
                        "[vector_training] model selection: index={index_label} field={field_id}, all {} sampled vectors with one deterministic \
                         centroid seed; model-selection holdout disabled",
                        training_count,
                    );
                }

                let mut base_config = crate::structures::CoarseConfig::new(dim, num_clusters)
                    .with_routing(config.ivf_routing);
                if let Some(soar) = config.soar.clone() {
                    base_config = base_config.with_soar(soar);
                }

                let mut selected: Option<(
                    crate::structures::CoarseCentroids,
                    Option<FloatBuildQuality>,
                    u64,
                )> = None;
                for &seed in seeds {
                    let candidate = crate::structures::CoarseCentroids::train_contiguous(
                        &base_config.clone().with_seed(seed),
                        training_values,
                        training_count,
                        index_label,
                    );
                    let quality =
                        evaluate_float_build_quality(&candidate, validation, config.ivf_routing);
                    if let Some(quality) = quality {
                        log::info!(
                            "[vector_training] IVF candidate: index={index_label} field={field_id} seed={seed}, objective={:.6}, \
                             exact/routed_mean_distortion={:.6}/{:.6}, router_recall@1={:.4}, \
                             construction_postings/vector={:.3}, \
                             residual_scale[p50/p95/p99]={:.4}/{:.4}/{:.4}, \
                             construction_occupancy[p95/p99/max/empty]={}/{}/{}/{}",
                            quality.objective,
                            quality.mean_exact_distortion,
                            quality.mean_routed_distortion,
                            quality.router_recall_at_1,
                            quality.mean_construction_assignments,
                            quality.residual_p50,
                            quality.residual_p95,
                            quality.residual_p99,
                            quality.occupancy.p95,
                            quality.occupancy.p99,
                            quality.occupancy.max,
                            quality.occupancy.empty,
                        );
                    }
                    let replace = selected.as_ref().is_none_or(|(_, best, _)| {
                        quality
                            .map(|quality| quality.objective)
                            .unwrap_or(f64::INFINITY)
                            .total_cmp(
                                &best
                                    .map(|quality| quality.objective)
                                    .unwrap_or(f64::INFINITY),
                            )
                            .is_lt()
                    });
                    if replace {
                        selected = Some((candidate, quality, seed));
                    }
                }
                let (mut centroids, quality, seed) =
                    selected.expect("the fixed centroid seed bank is non-empty");
                if let Some(quality) = quality {
                    log::info!(
                        "[vector_training] selected IVF seed: index={index_label} field={field_id} seed={seed}, held-out objective {:.6} \
                         (occupancy penalty {:.4})",
                        quality.objective,
                        quality.occupancy.penalty,
                    );
                } else {
                    log::info!(
                        "[vector_training] selected IVF seed: index={index_label} field={field_id} seed={seed}, without a model-selection \
                         holdout",
                    );
                }
                centroids.version =
                    crate::structures::mark_ivf_tq_cosine_generation(centroids.version);
                TrainedFieldArtifacts::FloatCentroids(centroids)
            }
            (IvfFieldConfig::Binary(config), TrainingSample::Binary(codes)) => {
                let byte_len = dim.div_ceil(8);
                let training_count = codes.len() / byte_len;
                let mut binary_config = crate::structures::BinaryIvfConfig::new(dim, num_clusters);
                binary_config.max_train_samples = training_count;
                binary_config.routing = config.ivf_routing;
                let quantizer = crate::structures::BinaryCoarseQuantizer::train(
                    binary_config,
                    codes,
                    training_count,
                    index_label,
                )
                .map_err(Error::Io)?;
                TrainedFieldArtifacts::Binary(quantizer)
            }
            _ => {
                return Err(Error::Internal(format!(
                    "training sample kind does not match field {field_id}"
                )));
            }
        };

        let actual_num_clusters = match &artifacts {
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
                codebook_file: None,
            },
            artifacts,
        })
    }

    async fn save_trained_field(&self, model: TrainedFieldModel) -> Result<TrainedFieldUpdate> {
        let TrainedFieldModel { update, artifacts } = model;
        match artifacts {
            TrainedFieldArtifacts::FloatCentroids(centroids) => {
                self.save_trained_artifact(&centroids, &update.centroids_file)
                    .await?;
                log::info!(
                    "[vector_training] saved IVF-TQ coarse artifact: index={} field={} ({} clusters; leaf codec is derived)",
                    self.schema.index_label(),
                    update.field_id,
                    centroids.num_clusters,
                );
            }
            TrainedFieldArtifacts::Binary(quantizer) => {
                self.save_trained_artifact(&quantizer, &update.centroids_file)
                    .await?;
                log::info!(
                    "[vector_training] saved binary IVF artifact: index={} field={} ({} clusters)",
                    self.schema.index_label(),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ivf_config(num_clusters: Option<usize>) -> DenseVectorConfig {
        DenseVectorConfig::ivf_tq(8, num_clusters, 4)
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
    fn final_sample_is_selected_at_the_points_per_centroid_ceiling() {
        let config = IvfFieldConfig::Float(DenseVectorConfig::ivf_tq(8, Some(4), 1));
        assert_eq!(
            final_training_sample_count(&config, 10_000, 10_000).unwrap(),
            4 * COARSE_TRAINING_POINTS_PER_CENTROID,
        );
        assert_eq!(
            final_training_sample_count(&config, 10_000, 512).unwrap(),
            512,
        );
    }

    #[test]
    fn holdout_preserves_the_training_points_per_centroid_floor() {
        assert_eq!(validation_sample_count(390, 10, 8), 0);
        assert_eq!(validation_sample_count(391, 10, 8), 1);

        let config = IvfFieldConfig::Float(ivf_config(None));
        let sample_count = 1_000;
        let clusters = effective_field_num_clusters(&config, 10_000, sample_count).unwrap();
        let held_out = validation_sample_count(sample_count, clusters, config.dim());
        assert!(
            sample_count - held_out >= clusters.saturating_mul(MIN_TRAINING_POINTS_PER_CENTROID)
        );
    }

    #[test]
    fn deterministic_point_sample_is_sorted_unique_and_repeatable() {
        let first = deterministic_sample_ordinals(10_000, 1_000, 7);
        let repeated = deterministic_sample_ordinals(10_000, 1_000, 7);
        let other_seed = deterministic_sample_ordinals(10_000, 1_000, 8);
        assert_eq!(first, repeated);
        assert_ne!(first, other_seed);
        assert_eq!(first.len(), 1_000);
        assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(first.iter().all(|&ordinal| ordinal < 10_000));
    }

    #[test]
    fn model_selection_accounts_for_initialization_and_all_lloyd_passes() {
        assert_eq!(model_selection_seeds(1_000, 16, 8, true).len(), 2);
        assert_eq!(model_selection_seeds(1_000, 16, 8, false).len(), 1);
        // A single assignment pass is only 180M coordinate comparisons, but
        // two complete initialization/refinement candidates exceed the budget.
        assert_eq!(model_selection_seeds(1_800, 1_000, 100, true).len(), 1);
        assert_eq!(
            model_selection_seeds(MAX_MULTI_SEED_COORDINATE_WORK, 2, 1, true).len(),
            1,
        );
    }

    #[test]
    fn flat_float_sample_partitions_a_deterministic_holdout_suffix() {
        let original: Vec<f32> = (0..20).map(|value| value as f32).collect();
        let mut first = original.clone();
        let mut repeated = original.clone();
        let validation_count = validation_sample_count(10, 2, 2);
        let first_split = partition_contiguous_holdout_suffix(&mut first, 2, validation_count, 11);
        let repeated_split =
            partition_contiguous_holdout_suffix(&mut repeated, 2, validation_count, 11);

        assert_eq!(first, repeated);
        assert_eq!(first_split, repeated_split);
        assert_eq!(first_split, 18);
        assert_eq!(first.len(), original.len());
        assert_eq!(first[first_split..].len(), 2);

        let mut first_components: Vec<u32> =
            first.chunks_exact(2).map(|row| row[0] as u32).collect();
        first_components.sort_unstable();
        assert_eq!(first_components, vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18]);
        assert!(first.chunks_exact(2).all(|row| row[1] == row[0] + 1.0));
    }

    #[test]
    fn occupancy_report_exposes_tail_and_empty_cells() {
        let report = occupancy_quality(vec![0, 1, 2, 7], 10);
        assert_eq!(report.p95, 7);
        assert_eq!(report.p99, 7);
        assert_eq!(report.max, 7);
        assert_eq!(report.empty, 1);
        assert!(report.penalty > 0.0);
    }

    #[test]
    fn model_selection_objective_prices_routed_distortion_excess() {
        let objective = float_model_selection_objective(2.0, 3.0, 0.1);
        assert!((objective - 3.2).abs() < f64::EPSILON);
        assert_eq!(float_model_selection_objective(2.0, 1.5, 0.0), 2.0);
    }

    #[test]
    fn float_quality_reports_exact_query_router_recall() {
        let training = [0.0, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0];
        let config = crate::structures::CoarseConfig::new(2, 2);
        let centroids = crate::structures::CoarseCentroids::train_contiguous(
            &config,
            &training,
            4,
            "test-index",
        );
        for routing in [
            crate::dsl::IvfRoutingMode::Flat,
            crate::dsl::IvfRoutingMode::Auto,
        ] {
            let quality =
                evaluate_float_build_quality(&centroids, &[0.05, 0.0, 10.05, 10.0], routing)
                    .unwrap();

            assert_eq!(quality.router_recall_at_1, 1.0);
            assert!(
                (quality.mean_exact_distortion - quality.mean_routed_distortion).abs()
                    < f64::EPSILON
            );
            assert_eq!(quality.mean_construction_assignments, 1.0);
            assert_eq!(quality.occupancy.empty, 0);
        }
    }

    #[test]
    fn float_quality_counts_soar_secondary_postings_in_occupancy() {
        let training = [0.0, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0];
        let config = crate::structures::CoarseConfig::new(2, 2)
            .with_routing(crate::dsl::IvfRoutingMode::Flat)
            .with_soar(crate::structures::SoarConfig::full());
        let centroids = crate::structures::CoarseCentroids::train_contiguous(
            &config,
            &training,
            4,
            "test-index",
        );
        let quality = evaluate_float_build_quality(
            &centroids,
            &[0.05, 0.0, 10.05, 10.0],
            crate::dsl::IvfRoutingMode::Flat,
        )
        .unwrap();

        assert_eq!(quality.mean_construction_assignments, 2.0);
        assert_eq!(quality.occupancy.empty, 0);
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

    #[test]
    fn ivf_tq_training_marks_generation_normalizes_and_calibrates_default_soar() {
        let config = IvfFieldConfig::Float(DenseVectorConfig::ivf_tq(2, Some(1), 1));
        let mut sample = TrainingSample::Float(vec![3.0, 4.0, 30.0, 40.0, 300.0, 400.0]);
        let model = IndexWriter::<crate::directories::RamDirectory>::train_field_model(
            Field(3),
            &config,
            &mut sample,
            3,
            "test",
            "test-index",
        )
        .unwrap();
        let TrainedFieldArtifacts::FloatCentroids(centroids) = model.artifacts else {
            panic!("expected float IVF-TQ centroids");
        };

        assert!(crate::structures::is_ivf_tq_cosine_generation(
            centroids.version
        ));
        assert!((centroids.centroids[0] - 0.6).abs() < 1e-6);
        assert!((centroids.centroids[1] - 0.8).abs() < 1e-6);
        let soar = centroids
            .soar_config
            .as_ref()
            .expect("default SOAR should propagate into the trained router");
        assert_eq!(soar.num_secondary, 1);
        assert!(soar.selective);
        assert!(
            soar.spill_threshold > 0.0,
            "the negative 30% target tag should be replaced by a calibrated threshold"
        );
        assert_eq!(soar.calibration_target(), None);
    }

    #[test]
    fn explicitly_disabled_soar_stays_off_during_ivf_tq_training() {
        let config = IvfFieldConfig::Float(DenseVectorConfig::ivf_tq(2, Some(1), 1).without_soar());
        let mut sample = TrainingSample::Float(vec![3.0, 4.0, 30.0, 40.0, 300.0, 400.0]);
        let model = IndexWriter::<crate::directories::RamDirectory>::train_field_model(
            Field(3),
            &config,
            &mut sample,
            3,
            "test-no-soar",
            "test-index",
        )
        .unwrap();
        let TrainedFieldArtifacts::FloatCentroids(centroids) = model.artifacts else {
            panic!("expected float IVF-TQ centroids");
        };

        assert!(centroids.soar_config.is_none());
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
            DenseVectorConfig::ivf_tq(READ_FAIL_DIM, Some(1), 1),
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
            DenseVectorConfig::ivf_tq(READ_FAIL_DIM, Some(1), 1),
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
        let old_version = writer.segment_manager.trained().unwrap().centroids[&embedding.0].version;

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
            writer.segment_manager.trained().unwrap().centroids[&embedding.0].version,
            old_version,
        );
    }
}
