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
    BinaryDenseVectorConfig, BinaryIndexType, DenseVectorConfig, Field, FieldType,
    VectorIndexAlter, VectorIndexType,
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
/// Inspect a bounded deterministic reserve when binary quality filtering drops
/// selected rows. The resident training matrix remains capped by `take`; this
/// only adds ordinal metadata and reads, and prevents a single constant code
/// from making an otherwise viable geometry retry forever.
const MAX_BINARY_REPLENISHMENT_CANDIDATES: usize = 1_000_000;
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
    scann_generation: Option<u64>,
    scann_artifact_id: Option<u64>,
}

enum TrainedFieldArtifacts {
    /// IVF-TQ: only the coarse router is trained; the TQ leaf codec is
    /// derived from the dimension.
    FloatCentroids(crate::structures::CoarseCentroids),
    Binary(crate::structures::BinaryCoarseQuantizer),
    Scann(crate::structures::vector::scann::ScannTrainedArtifact),
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

    fn target_vectors(&self) -> Option<u64> {
        match self {
            Self::Float(config) => config.target_vectors,
            Self::Binary(config) => config.target_vectors,
        }
    }

    fn uses_target_sized_ivf(&self) -> bool {
        self.num_clusters().is_none()
            && self.target_vectors().is_some()
            && (matches!(self, Self::Float(config) if config.index_type == VectorIndexType::IvfTq)
                || matches!(self, Self::Binary(config) if config.index_type == BinaryIndexType::Ivf))
    }

    fn supports_deferred_flat(&self) -> bool {
        self.uses_target_sized_ivf()
            || matches!(self, Self::Float(config) if config.index_type == VectorIndexType::Scann)
            || matches!(self, Self::Binary(config) if config.index_type == BinaryIndexType::Scann)
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlterVectorIndexState {
    Built,
    DeferredFlat,
    ParametersOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlterVectorIndexOutcome {
    pub publication_generation: u64,
    pub state: AlterVectorIndexState,
}

fn same_soar_layout(
    left: Option<&crate::structures::SoarConfig>,
    right: Option<&crate::structures::SoarConfig>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.num_secondary == right.num_secondary
                && left.selective == right.selective
                && left.spill_threshold.to_bits() == right.spill_threshold.to_bits()
        }
        _ => false,
    }
}

fn alter_requires_rebuild(current: &IvfFieldConfig, target: &IvfFieldConfig) -> bool {
    match (current, target) {
        (IvfFieldConfig::Float(current), IvfFieldConfig::Float(target)) => {
            current.index_type != target.index_type
                || current.num_clusters != target.num_clusters
                || (current.num_clusters.is_none()
                    && target.num_clusters.is_none()
                    && current.target_vectors != target.target_vectors)
                || current.tree_levels != target.tree_levels
                || current.ivf_routing != target.ivf_routing
                || current.unit_norm != target.unit_norm
                || !same_soar_layout(current.soar.as_ref(), target.soar.as_ref())
        }
        (IvfFieldConfig::Binary(current), IvfFieldConfig::Binary(target)) => {
            current.index_type != target.index_type
                || current.num_clusters != target.num_clusters
                || (current.num_clusters.is_none()
                    && target.num_clusters.is_none()
                    && current.target_vectors != target.target_vectors)
                || current.tree_levels != target.tree_levels
                || current.ivf_routing != target.ivf_routing
                || !same_soar_layout(current.soar.as_ref(), target.soar.as_ref())
        }
        _ => true,
    }
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
    let stable_automatic_topology =
        config.num_clusters().is_none() && config.target_vectors().is_some();
    if (config.num_clusters().is_some() || stable_automatic_topology) && requested > artifact_limit
    {
        return Err(Error::Schema(format!(
            "configured IVF codebook needs {} bytes for {} centroids, exceeding the {}-byte artifact limit",
            requested.saturating_mul(centroid_bytes),
            requested,
            super::metadata::MAX_TRAINED_ARTIFACT_BYTES,
        )));
    }
    if stable_automatic_topology && requested > quality_limit {
        let required = requested.saturating_mul(MIN_TRAINING_POINTS_PER_CENTROID);
        return Err(Error::Schema(format!(
            "target-sized IVF geometry selected {requested} leaves and needs at least {required} training samples (hardcoded {MIN_TRAINING_POINTS_PER_CENTROID} samples/leaf), but the builder can supply {sample_count}"
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

fn training_sample_bytes(config: &IvfFieldConfig) -> Result<usize> {
    match config {
        IvfFieldConfig::Float(config) => config
            .dim
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| Error::Schema("float training vector size overflows".into())),
        IvfFieldConfig::Binary(config) => Ok(config.dim.div_ceil(8)),
    }
}

fn required_scann_training_sample(num_leaves: u32) -> Result<u64> {
    u64::from(num_leaves)
        .checked_mul(crate::structures::vector::scann::MIN_PARTITION_TRAINING_POINTS_PER_LEAF)
        .map(|required| required.max(crate::structures::vector::scann::MIN_POINTS_FOR_PARTITIONING))
        .ok_or_else(|| Error::Schema("ScaNN minimum training sample overflows u64".into()))
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

fn scann_geometry(
    config: &DenseVectorConfig,
    corpus_count: usize,
) -> Result<crate::structures::vector::scann::ScannGeometry> {
    let dimension = u32::try_from(config.dim)
        .map_err(|_| Error::Schema("ScaNN dimension exceeds u32".into()))?;
    let Some(leaves) = config.num_clusters else {
        let sizing_count = config.target_vectors.unwrap_or(0).max(corpus_count as u64);
        return crate::structures::vector::scann::derive_geometry_with_levels(
            sizing_count,
            dimension,
            config.tree_levels,
        )
        .map_err(|error| Error::Schema(error.to_string()));
    };
    let leaves = u32::try_from(leaves)
        .map_err(|_| Error::Schema("ScaNN num_clusters exceeds u32".into()))?;
    crate::structures::vector::scann::geometry_for_leaves_with_auto_depth(
        leaves,
        dimension,
        config.tree_levels,
    )
    .map_err(|error| Error::Schema(error.to_string()))
}

fn binary_scann_geometry(
    config: &BinaryDenseVectorConfig,
    corpus_count: usize,
) -> Result<crate::structures::vector::scann::ScannGeometry> {
    let leaves = config.num_clusters;
    let Some(leaves) = leaves else {
        let sizing_count = config.target_vectors.unwrap_or(0).max(corpus_count as u64);
        return crate::structures::vector::scann::derive_geometry_with_levels(
            sizing_count,
            config.dim as u32,
            config.tree_levels,
        )
        .map_err(|error| Error::Schema(error.to_string()));
    };
    crate::structures::vector::scann::geometry_for_leaves_with_auto_depth(
        u32::try_from(leaves)
            .map_err(|_| Error::Schema("binary ScaNN num_clusters exceeds u32".into()))?,
        config.dim as u32,
        config.tree_levels,
    )
    .map_err(|error| Error::Schema(error.to_string()))
}

fn scann_training_sample_count(
    field: Field,
    total: usize,
    limit: usize,
    bytes_per_sample: usize,
    geometry: &crate::structures::vector::scann::ScannGeometry,
    binary: bool,
) -> Result<usize> {
    let desired = crate::structures::vector::scann::desired_training_sample(
        total as u64,
        geometry.num_leaves,
    ) as usize;
    let take = desired.min(limit);
    let minimum = usize::try_from(required_scann_training_sample(geometry.num_leaves)?)
        .map_err(|_| Error::Schema("ScaNN minimum sample exceeds usize".into()))?;
    if take < minimum {
        let minimum_bytes = minimum
            .checked_mul(bytes_per_sample)
            .ok_or_else(|| Error::Schema("ScaNN minimum training memory overflows".into()))?;
        let kind = if binary { "binary ScaNN" } else { "ScaNN" };
        return Err(Error::Schema(format!(
            "{kind} geometry for field {} needs at least {} sampled vectors ({} leaves at the hardcoded {} samples/leaf; {} bytes), but the builder limits allow {}; raise vector_training_memory_bytes/vector_training_max_samples",
            field.0,
            minimum,
            geometry.num_leaves,
            crate::structures::vector::scann::MIN_PARTITION_TRAINING_POINTS_PER_LEAF,
            minimum_bytes,
            take
        )));
    }
    Ok(take)
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

fn binary_replenishment_candidate_count(total: usize, take: usize, minimum_usable: usize) -> usize {
    let reserve = minimum_usable.clamp(1_024, MAX_BINARY_REPLENISHMENT_CANDIDATES);
    take.saturating_add(reserve).min(total)
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
        let exact_distance = crate::structures::simd::squared_l2_f32(
            vector,
            centroids.get_centroid(exact_cluster_id),
        );
        let routed_distance = if routed_cluster_id == exact_cluster_id {
            exact_distance
        } else {
            crate::structures::simd::squared_l2_f32(
                vector,
                centroids.get_centroid(routed_cluster_id),
            )
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
    /// Atomically replace one field's ANN algorithm or parameters.
    ///
    /// Stored vectors are retained by every segment, so IVF-TQ and ScaNN can
    /// rebuild each other without reindexing source documents. ScaNN targets
    /// below their geometry-derived corpus floor publish as Flat and can be
    /// completed later by `build_vector_index`.
    pub async fn alter_vector_index(
        &mut self,
        field: Field,
        alter: VectorIndexAlter,
    ) -> Result<AlterVectorIndexOutcome> {
        // Workers retain a partially filled SegmentBuilder for an entire
        // commit cycle. Flush that cycle before changing schemas so no output
        // built with the old ANN layout can be published after this ALTER.
        self.commit().await?;
        let current_generation = self.segment_manager.published_generation();
        let current_entry = current_generation
            .schema
            .get_field_entry(field)
            .ok_or_else(|| Error::Schema(format!("unknown vector field {}", field.0)))?;
        let current_config = match current_entry.field_type {
            FieldType::DenseVector => current_entry
                .dense_vector_config
                .clone()
                .map(IvfFieldConfig::Float),
            FieldType::BinaryDenseVector => current_entry
                .binary_dense_vector_config
                .clone()
                .map(IvfFieldConfig::Binary),
            _ => None,
        }
        .ok_or_else(|| Error::Schema(format!("field {} is not a vector field", field.0)))?;
        let candidate_schema = Arc::new(
            current_generation
                .schema
                .with_vector_index_alter(field, alter)
                .map_err(Error::Schema)?,
        );
        let target_entry = candidate_schema
            .get_field_entry(field)
            .expect("validated ALTER preserves the field");
        let target_config = match target_entry.field_type {
            FieldType::DenseVector => IvfFieldConfig::Float(
                target_entry
                    .dense_vector_config
                    .clone()
                    .expect("validated dense ALTER has a config"),
            ),
            FieldType::BinaryDenseVector => IvfFieldConfig::Binary(
                target_entry
                    .binary_dense_vector_config
                    .clone()
                    .expect("validated binary ALTER has a config"),
            ),
            _ => unreachable!("validated vector ALTER changed field type"),
        };

        let artifact_update = self.segment_manager.begin_vector_artifact_update().await?;
        if !alter_requires_rebuild(&current_config, &target_config) {
            self.segment_manager
                .publish_vector_schema_only(&artifact_update, candidate_schema)
                .await?;
            drop(artifact_update);
            return Ok(AlterVectorIndexOutcome {
                publication_generation: self.segment_manager.publication_id(),
                state: AlterVectorIndexState::ParametersOnly,
            });
        }

        self.cleanup_unreferenced_vector_artifacts().await;
        let snapshot = self.segment_manager.acquire_snapshot().await;
        let fields = vec![(field, target_config.clone())];
        let total_vectors = self
            .count_vectors_for_training(
                snapshot.segment_ids(),
                &fields,
                false,
                &current_generation.schema,
            )
            .await?;
        let corpus_count = total_vectors.get(&field.0).copied().unwrap_or(0);
        let mut candidate_metadata = self.segment_manager.read_metadata(Clone::clone).await;
        candidate_metadata.init_field(field.0, target_config.index_type());
        let field_metadata = candidate_metadata
            .vector_fields
            .get_mut(&field.0)
            .expect("ALTER initialized vector field metadata");
        field_metadata.index_type = target_config.index_type();
        field_metadata.state = super::VectorIndexState::Flat;
        field_metadata.centroids_file = None;
        field_metadata.codebook_file = None;
        field_metadata.artifact_generation = None;
        field_metadata.artifact_id = None;
        candidate_metadata.refresh_total_vectors();

        let artifact_generation = SegmentId::new().to_hex();
        let updates = self
            .train_fields(
                snapshot.segment_ids(),
                &fields,
                &total_vectors,
                &artifact_generation,
                &current_generation.schema,
            )
            .await?;
        for update in &updates {
            if let (Some(generation), Some(artifact_id)) =
                (update.scann_generation, update.scann_artifact_id)
            {
                candidate_metadata.mark_scann_field_built(
                    update.field_id,
                    update.vector_count,
                    update.num_clusters,
                    update.centroids_file.clone(),
                    generation,
                    artifact_id,
                )?;
            } else {
                candidate_metadata.mark_field_built(
                    update.field_id,
                    update.vector_count,
                    update.num_clusters,
                    update.centroids_file.clone(),
                    update.codebook_file.clone(),
                );
            }
        }
        let built = candidate_metadata.is_field_built(field.0);
        if !built && !target_config.supports_deferred_flat() && corpus_count > 0 {
            return Err(Error::Schema(format!(
                "cannot train target vector index for field {} from {corpus_count} vectors",
                field.0
            )));
        }

        let candidate_trained = super::IndexMetadata::try_load_trained_from_fields(
            &candidate_metadata.vector_fields,
            candidate_schema.as_ref(),
            self.directory.as_ref(),
        )
        .await?
        .map(Arc::new);
        let finalize_ann_segments = candidate_trained.is_some();
        let rewrite_trained = candidate_trained
            .clone()
            .unwrap_or_else(|| Arc::new(crate::segment::TrainedVectorStructures::default()));
        let staged = self
            .segment_manager
            .stage_vector_generation_with_schema(
                &artifact_update,
                snapshot.segment_ids(),
                &[field.0],
                rewrite_trained,
                true,
                Arc::clone(&candidate_schema),
            )
            .await?;
        self.segment_manager
            .publish_vector_generation_with_schema(
                &artifact_update,
                candidate_schema,
                candidate_metadata.vector_fields,
                candidate_trained,
                staged,
            )
            .await?;
        drop(snapshot);
        drop(artifact_update);
        if finalize_ann_segments {
            self.segment_manager
                .rewrite_vector_segments(&[field.0])
                .await?;
        }
        self.cleanup_unreferenced_vector_artifacts().await;

        Ok(AlterVectorIndexOutcome {
            publication_generation: self.segment_manager.publication_id(),
            state: if built {
                AlterVectorIndexState::Built
            } else {
                AlterVectorIndexState::DeferredFlat
            },
        })
    }

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
        let artifact_update = self.segment_manager.begin_vector_artifact_update().await?;
        let generation = self.segment_manager.published_generation();
        let schema = generation.schema.clone();
        let dense_fields = Self::get_ivf_vector_fields(&schema);
        if dense_fields.is_empty() {
            log::info!(
                "[vector_training] no dense vector fields configured for ANN indexing: index={}",
                schema.index_label()
            );
            return Ok(());
        }

        self.cleanup_unreferenced_vector_artifacts().await;

        let fields_to_train = match mode {
            VectorGenerationMode::BuildMissing => self.get_fields_to_build(&dense_fields).await,
            VectorGenerationMode::RetrainAll => dense_fields.clone(),
        };
        for (_, config) in &fields_to_train {
            if !matches!(
                config,
                IvfFieldConfig::Float(config) if config.index_type == VectorIndexType::Scann
            ) && !matches!(
                config,
                IvfFieldConfig::Binary(config) if config.index_type == BinaryIndexType::Scann
            ) {
                validate_explicit_cluster_count(config.num_clusters())?;
            }
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
                    &schema,
                )
                .await?;
            let artifact_generation = SegmentId::new().to_hex();
            let updates = self
                .train_fields(
                    snapshot.segment_ids(),
                    &fields_to_train,
                    &total_vectors,
                    &artifact_generation,
                    &schema,
                )
                .await?;
            for update in &updates {
                candidate_metadata.init_field(update.field_id, update.index_type);
                if let (Some(generation), Some(artifact_id)) =
                    (update.scann_generation, update.scann_artifact_id)
                {
                    candidate_metadata.mark_scann_field_built(
                        update.field_id,
                        update.vector_count,
                        update.num_clusters,
                        update.centroids_file.clone(),
                        generation,
                        artifact_id,
                    )?;
                } else {
                    candidate_metadata.mark_field_built(
                        update.field_id,
                        update.vector_count,
                        update.num_clusters,
                        update.centroids_file.clone(),
                        update.codebook_file.clone(),
                    );
                }
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
            schema.as_ref(),
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
        schema: &Arc<crate::dsl::Schema>,
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
            if config.uses_target_sized_ivf() {
                let leaves = config.optimal_num_clusters(corpus_count);
                let required = leaves.saturating_mul(MIN_TRAINING_POINTS_PER_CENTROID);
                if corpus_count < required {
                    log::info!(
                        "[vector_training] deferring target-sized IVF field {}: index={} has {} vectors; selected {}-leaf topology requires at least {}",
                        field.0,
                        index_label,
                        corpus_count,
                        leaves,
                        required,
                    );
                    continue;
                }
            }
            if let IvfFieldConfig::Float(scann) = config
                && scann.index_type == VectorIndexType::Scann
            {
                let geometry = if corpus_count == 0 {
                    None
                } else {
                    Some(scann_geometry(scann, corpus_count)?)
                };
                let required = geometry.as_ref().map_or(
                    crate::structures::vector::scann::MIN_POINTS_FOR_PARTITIONING,
                    |geometry| {
                        crate::structures::vector::scann::MIN_POINTS_FOR_PARTITIONING.max(
                            required_scann_training_sample(geometry.num_leaves)
                                .expect("validated ScaNN leaves fit u64"),
                        )
                    },
                );
                if corpus_count < required as usize {
                    log::info!(
                        "[vector_training] deferring ScaNN field {}: index={} has {} vectors; selected geometry requires {} (max of hardcoded partition floor {} and {} samples/leaf)",
                        field.0,
                        index_label,
                        corpus_count,
                        required,
                        crate::structures::vector::scann::MIN_POINTS_FOR_PARTITIONING,
                        crate::structures::vector::scann::MIN_PARTITION_TRAINING_POINTS_PER_LEAF,
                    );
                    continue;
                }
                let geometry = geometry.expect("non-empty ready corpus has ScaNN geometry");
                debug_assert!(geometry.centroid_levels > 0);
                if scann.soar.is_some() {
                    return Err(Error::Schema(format!(
                        "ScaNN field {} enables SOAR, but ScaNN SOAR secondary assignments are not implemented; set soar: null before building",
                        field.0
                    )));
                }
            }
            if let IvfFieldConfig::Binary(scann) = config
                && scann.index_type == BinaryIndexType::Scann
            {
                let required = if corpus_count == 0 {
                    crate::structures::vector::scann::MIN_POINTS_FOR_PARTITIONING
                } else {
                    let geometry = binary_scann_geometry(scann, corpus_count)?;
                    crate::structures::vector::scann::MIN_POINTS_FOR_PARTITIONING
                        .max(required_scann_training_sample(geometry.num_leaves)?)
                };
                if corpus_count < required as usize {
                    log::info!(
                        "[vector_training] deferring binary ScaNN field {}: index={} has {} vectors; selected geometry requires {}",
                        field.0,
                        index_label,
                        corpus_count,
                        required,
                    );
                    continue;
                }
            }
            let sampled = self
                .collect_training_sample(segment_ids, *field, config, corpus_count, schema)
                .await?;
            let Some(mut sample) = sampled else {
                if matches!(config, IvfFieldConfig::Float(config) if config.index_type == VectorIndexType::Scann)
                    || matches!(config, IvfFieldConfig::Binary(config) if config.index_type == BinaryIndexType::Scann)
                {
                    log::warn!(
                        "[vector_training] deferring ScaNN field {}: index={} has no usable sampled vectors after quality filtering",
                        field.0,
                        index_label,
                    );
                    continue;
                }
                missing.push(field.0);
                continue;
            };
            let minimum_usable = match config {
                IvfFieldConfig::Float(config) if config.index_type == VectorIndexType::Scann => {
                    usize::try_from(required_scann_training_sample(
                        scann_geometry(config, corpus_count)?.num_leaves,
                    )?)
                    .map_err(|_| Error::Schema("ScaNN minimum sample exceeds usize".into()))?
                }
                IvfFieldConfig::Binary(config) if config.index_type == BinaryIndexType::Scann => {
                    usize::try_from(required_scann_training_sample(
                        binary_scann_geometry(config, corpus_count)?.num_leaves,
                    )?)
                    .map_err(|_| {
                        Error::Schema("binary ScaNN minimum sample exceeds usize".into())
                    })?
                }
                _ => 0,
            };
            let usable = sample.len(config.dim());
            if usable < minimum_usable {
                log::warn!(
                    "[vector_training] deferring ScaNN field {}: index={} has {} usable sampled vectors after filtering, selected geometry requires at least {}",
                    field.0,
                    index_label,
                    usable,
                    minimum_usable,
                );
                continue;
            }
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
        if updates.is_empty() && !fields.is_empty() && !missing.is_empty() {
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
        schema: &crate::dsl::Schema,
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
            schema,
            meta.num_docs,
            field_ids,
        )
        .await
    }

    /// Get all dense vector fields that need ANN indexes
    fn get_ivf_vector_fields(schema: &crate::dsl::Schema) -> Vec<(Field, IvfFieldConfig)> {
        schema
            .fields()
            .filter_map(|(field, entry)| {
                if entry.field_type == FieldType::DenseVector && entry.indexed {
                    entry
                        .dense_vector_config
                        .as_ref()
                        // Flat is a pre-build storage state; the production ANN
                        // path is trained once and shared by every segment.
                        .filter(|c| c.uses_ivf() || c.index_type == VectorIndexType::Scann)
                        .map(|c| (field, IvfFieldConfig::Float(c.clone())))
                } else if entry.field_type == FieldType::BinaryDenseVector && entry.indexed {
                    entry
                        .binary_dense_vector_config
                        .as_ref()
                        .filter(|config| {
                            matches!(
                                config.index_type,
                                BinaryIndexType::Ivf | BinaryIndexType::Scann
                            )
                        })
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
        schema: &crate::dsl::Schema,
    ) -> Result<FxHashMap<u32, usize>> {
        let mut total_vectors: FxHashMap<u32, usize> = FxHashMap::default();
        let field_ids: Vec<u32> = fields_to_build.iter().map(|(field, _)| field.0).collect();

        // Initial construction rejects
        // ANN payloads for metadata-flat fields; an explicit retrain reads the
        // exact flat vectors retained beside the current ANN generation.
        for id_str in segment_ids {
            let segment_id = SegmentId::from_hex(id_str)
                .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {}", id_str)))?;
            let vectors = self
                .load_training_vectors(segment_id, &field_ids, schema)
                .await?;

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
        schema: &crate::dsl::Schema,
    ) -> Result<Option<TrainingSample>> {
        if total == 0 {
            return Ok(None);
        }
        let bytes_per_sample = training_sample_bytes(config)?;
        let limit = training_sample_limit(
            self.config.vector_training_max_samples,
            self.config.vector_training_memory_bytes,
            bytes_per_sample,
        )?;
        let mut binary_scann_minimum = None;
        let take = match config {
            IvfFieldConfig::Float(scann) if scann.index_type == VectorIndexType::Scann => {
                let geometry = scann_geometry(scann, total)?;
                scann_training_sample_count(
                    field,
                    total,
                    limit,
                    bytes_per_sample,
                    &geometry,
                    false,
                )?
            }
            IvfFieldConfig::Binary(scann) if scann.index_type == BinaryIndexType::Scann => {
                let geometry = binary_scann_geometry(scann, total)?;
                let take = scann_training_sample_count(
                    field,
                    total,
                    limit,
                    bytes_per_sample,
                    &geometry,
                    true,
                )?;
                binary_scann_minimum = Some(
                    usize::try_from(required_scann_training_sample(geometry.num_leaves)?).map_err(
                        |_| Error::Schema("binary ScaNN minimum sample exceeds usize".into()),
                    )?,
                );
                take
            }
            _ => final_training_sample_count(config, total, limit)?,
        };
        let sample_seed = 0x4845_524d_4553_4956 ^ field.0 as u64 ^ total as u64;
        let candidate_count = binary_scann_minimum.map_or(take, |minimum| {
            binary_replenishment_candidate_count(total, take, minimum)
        });
        let ordinals = deterministic_sample_ordinals(total, candidate_count, sample_seed);

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
        let mut ones_codes = 0usize;
        let mut surplus_codes = 0usize;
        let mut global_offset = 0usize;
        let mut cursor = 0usize;
        let field_ids = [field.0];

        for id_str in segment_ids {
            let segment_id = SegmentId::from_hex(id_str)
                .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {id_str}")))?;
            let vectors = self
                .load_training_vectors(segment_id, &field_ids, schema)
                .await?;

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
                            // Degenerate constant codes are withheld from
                            // training: k-majority dedicates centroids to
                            // them, which only institutionalizes the producer
                            // bug. One production field turned ~30% of a 163k
                            // codebook into duplicate zero centroids; another
                            // trained centroid 0 to exactly 0xFF from two
                            // years of signbit-packed NaN vectors. (They are
                            // still *indexed* — payload/flat parity — just
                            // not trained on.)
                            if code.iter().all(|&byte| byte == 0) {
                                zero_codes += 1;
                                continue;
                            }
                            if code.iter().all(|&byte| byte == 0xff) {
                                ones_codes += 1;
                                continue;
                            }
                            if codes.len() / bytes_per_sample < take {
                                codes.extend_from_slice(code);
                            } else {
                                surplus_codes += 1;
                            }
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
        // Coverage is checked against what was *selected*; withheld degenerate
        // codes are subtracted explicitly so a real traversal bug still trips.
        if global_offset != total
            || cursor != candidate_count
            || collected + zero_codes + ones_codes + surplus_codes != candidate_count
        {
            return Err(Error::Corruption(format!(
                "training sample coverage mismatch for field {}: counted={total}, traversed={global_offset}, selected={cursor}, collected={collected}, zero={zero_codes}, ones={ones_codes}, surplus={surplus_codes}",
                field.0,
            )));
        }
        if let Some(minimum) = binary_scann_minimum
            && collected < minimum
        {
            return Err(Error::Schema(format!(
                "binary ScaNN field {} has only {collected} usable training vectors after excluding {zero_codes} all-zero and {ones_codes} all-ones codes from {candidate_count} deterministic candidates; selected geometry requires at least {minimum}. Fix the binary embedding producer/data or choose a smaller geometry",
                field.0,
            )));
        }
        if zero_codes > 0 {
            log::warn!(
                "[vector_training] index={} field={}: {zero_codes} of {candidate_count} sampled candidates \
                 ({:.1}%) are all-zero and were excluded from training — they cannot be assigned \
                 to any leaf, so training on them only wastes centroids",
                self.schema.index_label(),
                field.0,
                100.0 * zero_codes as f64 / candidate_count.max(1) as f64,
            );
        }
        if ones_codes > 0 {
            log::warn!(
                "[vector_training] index={} field={}: {ones_codes} of {candidate_count} sampled candidates \
                 ({:.1}%) are all-ones and were excluded from training — training on the \
                 saturated constant only dedicates centroids to a producer bug",
                self.schema.index_label(),
                field.0,
                100.0 * ones_codes as f64 / candidate_count.max(1) as f64,
            );
        }
        if collected == 0 {
            log::warn!(
                "[vector_training] index={} field={}: every sampled vector is degenerate \
                 (all-zero or all-ones); skipping ANN training for this field",
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
        let num_clusters = match config {
            IvfFieldConfig::Float(config) if config.index_type == VectorIndexType::Scann => {
                scann_geometry(config, corpus_count)?.num_leaves as usize
            }
            IvfFieldConfig::Binary(config) if config.index_type == BinaryIndexType::Scann => {
                binary_scann_geometry(config, corpus_count)?.num_leaves as usize
            }
            _ => effective_field_num_clusters(config, corpus_count, sample_count)?,
        };

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
            (IvfFieldConfig::Float(config), TrainingSample::Float(values))
                if config.index_type == VectorIndexType::Scann =>
            {
                if config.soar.is_some() {
                    return Err(Error::Schema(format!(
                        "ScaNN field {field_id} enables SOAR, but ScaNN SOAR secondary assignments are not implemented; set soar: null before building"
                    )));
                }
                values
                    .chunks_exact_mut(dim)
                    .for_each(crate::structures::vector::ivf::routing::normalize_cosine_in_place);
                let geometry = scann_geometry(config, corpus_count)?;
                let dimensions_per_block = 2usize.min(dim);
                let seed = MODEL_SELECTION_SEEDS[0] ^ u64::from(field_id) ^ corpus_count as u64;
                let generation = u64::from_str_radix(
                    &artifact_generation[artifact_generation.len().saturating_sub(16)..],
                    16,
                )
                .unwrap_or(seed)
                .max(1);
                let model = crate::structures::vector::scann::FloatScannModel::train_model(
                    values,
                    sample_count,
                    dim,
                    &geometry.level_counts,
                    dimensions_per_block,
                    25,
                    seed,
                    crate::structures::vector::scann::DEFAULT_ANISOTROPIC_THRESHOLD,
                )
                .map_err(|error| Error::Internal(format!("ScaNN training failed: {error}")))?;
                let artifact = crate::structures::vector::scann::ScannTrainedArtifact::new(
                    generation,
                    sample_count as u64,
                    crate::structures::vector::scann::ScannConfig {
                        dimension: dim as u32,
                        tree_levels: geometry.centroid_levels,
                        num_leaves: geometry.num_leaves,
                        encoding: crate::structures::vector::scann::ScannEncoding::AsymmetricHash {
                            dimensions_per_block: dimensions_per_block as u16,
                            bits_per_code: 4,
                        },
                    },
                    model.routing.to_quantized_levels(),
                    Some(model.codebook.to_artifact()),
                )
                .map_err(|error| Error::Internal(format!("ScaNN artifact failed: {error}")))?;
                TrainedFieldArtifacts::Scann(artifact)
            }
            (IvfFieldConfig::Binary(config), TrainingSample::Binary(codes))
                if config.index_type == BinaryIndexType::Ivf =>
            {
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
            (IvfFieldConfig::Binary(config), TrainingSample::Binary(codes))
                if config.index_type == BinaryIndexType::Scann =>
            {
                let geometry = binary_scann_geometry(config, corpus_count)?;
                let seed = MODEL_SELECTION_SEEDS[0] ^ u64::from(field_id) ^ corpus_count as u64;
                let generation = u64::from_str_radix(
                    &artifact_generation[artifact_generation.len().saturating_sub(16)..],
                    16,
                )
                .unwrap_or(seed)
                .max(1);
                let training = crate::structures::vector::scann::BinaryScannTraining {
                    dim_bits: config.dim as u32,
                    geometry,
                    train_iters: 25,
                    seed,
                };
                let model = crate::structures::vector::scann::BinaryScannModel::train(
                    &training,
                    codes,
                    sample_count,
                    index_label,
                )
                .map_err(|error| {
                    Error::Internal(format!("binary ScaNN training failed: {error}"))
                })?;
                TrainedFieldArtifacts::Scann(
                    model
                        .to_artifact(generation, sample_count as u64)
                        .map_err(|error| {
                            Error::Internal(format!("binary ScaNN artifact failed: {error}"))
                        })?,
                )
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
            TrainedFieldArtifacts::Scann(artifact) => artifact.config.num_leaves as usize,
        };
        let (scann_generation, scann_artifact_id) = match &artifacts {
            TrainedFieldArtifacts::Scann(artifact) => {
                (Some(artifact.generation), Some(artifact.artifact_id))
            }
            _ => (None, None),
        };
        Ok(TrainedFieldModel {
            update: TrainedFieldUpdate {
                field_id,
                index_type: config.index_type(),
                vector_count: corpus_count,
                num_clusters: actual_num_clusters,
                centroids_file: centroids_filename,
                codebook_file: None,
                scann_generation,
                scann_artifact_id,
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
            TrainedFieldArtifacts::Scann(artifact) => {
                self.save_scann_artifact(&artifact, &update.centroids_file)
                    .await?;
                log::info!(
                    "[vector_training] saved ScaNN artifact: index={} field={} ({} leaves, {} levels, generation={})",
                    self.schema.index_label(),
                    update.field_id,
                    artifact.config.num_leaves,
                    artifact.config.tree_levels,
                    artifact.generation,
                );
            }
        }
        Ok(update)
    }

    async fn save_scann_artifact(
        &self,
        artifact: &crate::structures::vector::scann::ScannTrainedArtifact,
        filename: &str,
    ) -> Result<()> {
        let temp_filename = format!("{filename}.tmp");
        let temp_path = std::path::Path::new(&temp_filename);
        let final_path = std::path::Path::new(filename);
        let mut writer = self.directory.streaming_writer(temp_path).await?;
        artifact
            .write_to(&mut writer)
            .map_err(|error| Error::Serialization(error.to_string()))?;
        writer.finish()?;
        if let Err(error) = self.directory.rename(temp_path, final_path).await {
            let _ = self.directory.delete(temp_path).await;
            return Err(Error::Io(error));
        }
        self.directory.sync().await?;
        Ok(())
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

    fn scann_config(dim: usize, leaves: Option<usize>) -> DenseVectorConfig {
        DenseVectorConfig {
            dim,
            index_type: VectorIndexType::Scann,
            quantization: crate::dsl::DenseVectorQuantization::F32,
            num_clusters: leaves,
            target_vectors: None,
            tree_levels: Some(1),
            ivf_routing: crate::dsl::IvfRoutingMode::Auto,
            nprobe: 1,
            unit_norm: true,
            soar: None,
        }
    }

    #[test]
    fn billion_scale_scann_autopilot_preserves_geometry_across_builder_budgets() {
        let defaults = crate::IndexConfig::default();

        let mut float = scann_config(1_024, None);
        float.tree_levels = None;
        let float = IvfFieldConfig::Float(float);
        let float_limit = training_sample_limit(
            defaults.vector_training_max_samples,
            defaults.vector_training_memory_bytes,
            training_sample_bytes(&float).unwrap(),
        )
        .unwrap();
        let float_geometry = match &float {
            IvfFieldConfig::Float(config) => scann_geometry(config, 1_000_000_000).unwrap(),
            _ => unreachable!(),
        };
        assert_eq!(float_limit, 1_048_576);
        assert_eq!(float_geometry.level_counts, [1_000, 1_000_000]);
        let float_error = scann_training_sample_count(
            Field(7),
            1_000_000_000,
            float_limit,
            training_sample_bytes(&float).unwrap(),
            &float_geometry,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(float_error.contains("1000000 leaves"));
        assert!(float_error.contains("8000000 sampled vectors"));

        let mut binary = BinaryDenseVectorConfig::new(2_560);
        binary.index_type = BinaryIndexType::Scann;
        let binary = IvfFieldConfig::Binary(binary);
        let binary_limit = training_sample_limit(
            defaults.vector_training_max_samples,
            defaults.vector_training_memory_bytes,
            training_sample_bytes(&binary).unwrap(),
        )
        .unwrap();
        let binary_geometry = match &binary {
            IvfFieldConfig::Binary(config) => binary_scann_geometry(config, 1_000_000_000).unwrap(),
            _ => unreachable!(),
        };
        assert_eq!(binary_limit, 10_000_000);
        assert_eq!(binary_geometry.level_counts, [1_000, 1_000_000]);
        assert!(
            required_scann_training_sample(binary_geometry.num_leaves).unwrap()
                <= binary_limit as u64
        );
    }

    #[test]
    fn explicit_scann_leaves_have_stable_automatic_depth() {
        for leaves in [31_622, 31_623] {
            let mut float = scann_config(1_024, Some(leaves));
            float.tree_levels = None;
            assert_eq!(
                scann_geometry(&float, 15_000_000).unwrap().level_counts,
                [178, leaves as u32]
            );

            let mut binary = BinaryDenseVectorConfig::new(2_560);
            binary.index_type = BinaryIndexType::Scann;
            binary.num_clusters = Some(leaves);
            assert_eq!(
                binary_scann_geometry(&binary, 15_000_000)
                    .unwrap()
                    .level_counts,
                [178, leaves as u32]
            );
        }

        let mut float = scann_config(1_024, Some(1_000_000));
        float.tree_levels = None;
        assert_eq!(
            scann_geometry(&float, 15_000_000).unwrap().level_counts,
            [1_000, 1_000_000]
        );
    }

    #[test]
    fn target_vectors_selects_future_scann_topology_without_faking_readiness() {
        let mut binary = BinaryDenseVectorConfig::new(2_560);
        binary.index_type = BinaryIndexType::Scann;
        binary.target_vectors = Some(1_000_000_000);

        let geometry = binary_scann_geometry(&binary, 1_000_000).unwrap();
        assert_eq!(geometry.level_counts, [1_000, 1_000_000]);
        assert_eq!(
            required_scann_training_sample(geometry.num_leaves).unwrap(),
            8_000_000
        );
        let below_floor =
            scann_training_sample_count(Field(7), 7_999_999, 8_000_000, 320, &geometry, true)
                .unwrap_err()
                .to_string();
        assert!(below_floor.contains("needs at least 8000000 sampled vectors"));
        assert!(below_floor.contains("builder limits allow 7999999"));
        assert_eq!(
            scann_training_sample_count(Field(7), 8_000_000, 8_000_000, 320, &geometry, true,)
                .unwrap(),
            8_000_000
        );

        let mut explicit = binary.clone();
        explicit.num_clusters = Some(4_096);
        assert_eq!(
            binary_scann_geometry(&explicit, 1_000_000)
                .unwrap()
                .num_leaves,
            4_096,
            "explicit leaves override target-sized automatic geometry"
        );

        let mut unhinted = binary.clone();
        unhinted.target_vectors = None;
        assert_eq!(
            binary_scann_geometry(&binary, 2_000_000_000).unwrap(),
            binary_scann_geometry(&unhinted, 2_000_000_000).unwrap(),
            "target_vectors is a lower bound and cannot shrink live-corpus geometry"
        );
    }

    #[test]
    fn target_sized_ivf_geometry_waits_for_its_hardcoded_sample_floor() {
        let binary = IvfFieldConfig::Binary(
            BinaryDenseVectorConfig::new(256).with_target_vectors(1_000_000_000),
        );
        let leaves = 31_623;
        let required = leaves * MIN_TRAINING_POINTS_PER_CENTROID;

        let error = effective_field_num_clusters(&binary, 1_000_000, required - 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("target-sized IVF geometry"), "{error}");
        assert!(error.contains(&required.to_string()), "{error}");
        assert_eq!(
            effective_field_num_clusters(&binary, 1_000_000, required).unwrap(),
            leaves
        );
    }

    #[test]
    fn target_vector_alters_rebuild_only_when_the_hint_controls_geometry() {
        let binary_auto = IvfFieldConfig::Binary(
            BinaryDenseVectorConfig::new(256).with_target_vectors(15_000_000),
        );
        let binary_auto_changed = IvfFieldConfig::Binary(
            BinaryDenseVectorConfig::new(256).with_target_vectors(1_000_000_000),
        );
        assert!(alter_requires_rebuild(&binary_auto, &binary_auto_changed));

        let binary_explicit = IvfFieldConfig::Binary(
            BinaryDenseVectorConfig::new(256)
                .with_target_vectors(15_000_000)
                .with_ivf(Some(4_096), 64),
        );
        let binary_explicit_changed = IvfFieldConfig::Binary(
            BinaryDenseVectorConfig::new(256)
                .with_target_vectors(1_000_000_000)
                .with_ivf(Some(4_096), 64),
        );
        assert!(!alter_requires_rebuild(
            &binary_explicit,
            &binary_explicit_changed
        ));
        assert!(alter_requires_rebuild(
            &binary_explicit_changed,
            &binary_auto_changed
        ));

        let float_explicit = IvfFieldConfig::Float(
            DenseVectorConfig::ivf_tq(128, Some(4_096), 64).with_target_vectors(15_000_000),
        );
        let float_explicit_changed = IvfFieldConfig::Float(
            DenseVectorConfig::ivf_tq(128, Some(4_096), 64).with_target_vectors(1_000_000_000),
        );
        assert!(!alter_requires_rebuild(
            &float_explicit,
            &float_explicit_changed
        ));
    }

    #[test]
    fn explicit_scann_geometry_rejects_an_inadequate_builder_sample() {
        let geometry = crate::structures::vector::scann::geometry_for_leaves(20_000, 1).unwrap();
        let error =
            scann_training_sample_count(Field(7), 1_000_000, 159_999, 4_096, &geometry, false)
                .unwrap_err()
                .to_string();
        assert!(error.contains("needs at least 160000 sampled vectors"));
        assert!(error.contains("8 samples/leaf"));
        assert!(error.contains("builder limits allow 159999"));
    }

    #[test]
    fn scann_build_rejects_soar_instead_of_silently_ignoring_it() {
        let mut config = scann_config(2, Some(2));
        config.soar = Some(crate::structures::SoarConfig::default());
        let mut sample = TrainingSample::Float(vec![0.5; 100_000 * 2]);
        let result = IndexWriter::<crate::directories::RamDirectory>::train_field_model(
            Field(7),
            &IvfFieldConfig::Float(config),
            &mut sample,
            100_000,
            "00000000000000000000000000000007",
            "test",
        );
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("ScaNN SOAR must fail before training"),
        };
        assert!(error.to_string().contains("SOAR"));
        assert!(error.to_string().contains("soar: null"));
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
    fn binary_scann_quality_filter_has_deterministic_replenishment_capacity() {
        // When the requested sample is exactly the hard geometry floor, a
        // filtered constant code must not make the same undersized sample get
        // retried forever. Candidate reads grow, but the resident sample stays
        // capped at `take` in collect_training_sample.
        assert_eq!(binary_replenishment_candidate_count(101, 100, 100), 101);
        assert_eq!(
            binary_replenishment_candidate_count(10_000, 100, 100),
            1_124
        );
        assert_eq!(
            binary_replenishment_candidate_count(20_000_000, 10_000_000, 4_000_000),
            11_000_000
        );
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

    #[tokio::test]
    async fn alter_target_sized_ivf_below_hard_floor_publishes_deferred_flat() {
        let mut sb = SchemaBuilder::default();
        let hash = sb.add_binary_dense_vector_field_with_config(
            "hash",
            true,
            true,
            BinaryDenseVectorConfig::new(256).with_ivf(Some(1), 1),
        );
        let dir = RamDirectory::new();
        let config = IndexConfig {
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            num_indexing_threads: 1,
            ..Default::default()
        };
        let mut writer = IndexWriter::create(dir.clone(), sb.build(), config)
            .await
            .unwrap();
        for row in 0u8..64 {
            let mut doc = Document::new();
            doc.add_binary_dense_vector(hash, vec![row; 32]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        writer.build_vector_index().await.unwrap();

        let target = BinaryDenseVectorConfig::new(256).with_target_vectors(1_000_000_000);
        let required = target.optimal_num_clusters(64) * MIN_TRAINING_POINTS_PER_CENTROID;
        assert!(
            64 < required,
            "test corpus must stay below the hardcoded floor"
        );
        let outcome = writer
            .alter_vector_index(hash, VectorIndexAlter::Binary(target))
            .await
            .unwrap();

        assert_eq!(outcome.state, AlterVectorIndexState::DeferredFlat);
        let generation = writer.segment_manager.published_generation();
        let config = generation
            .schema
            .get_field_entry(hash)
            .unwrap()
            .binary_dense_vector_config
            .as_ref()
            .unwrap();
        assert_eq!(config.index_type, BinaryIndexType::Ivf);
        assert_eq!(config.num_clusters, None);
        assert_eq!(config.target_vectors, Some(1_000_000_000));
        assert!(
            !writer
                .segment_manager
                .read_metadata(|metadata| metadata.is_field_built(hash.0))
                .await
        );
        assert!(generation.trained_vectors.is_none());
        for id in writer.segment_manager.get_segment_ids().await {
            let segment = crate::segment::SegmentReader::open(
                &dir,
                SegmentId::from_hex(&id).unwrap(),
                generation.schema.clone(),
                16,
            )
            .await
            .unwrap();
            assert!(segment.get_vector_index(hash).is_none());
        }
    }

    #[tokio::test]
    async fn alter_target_sized_float_ivf_below_hard_floor_publishes_deferred_flat() {
        let mut sb = SchemaBuilder::default();
        let embedding = sb.add_dense_vector_field_with_config(
            "embedding",
            true,
            true,
            DenseVectorConfig::ivf_tq(4, Some(1), 1),
        );
        let dir = RamDirectory::new();
        let config = IndexConfig {
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            num_indexing_threads: 1,
            ..Default::default()
        };
        let mut writer = IndexWriter::create(dir.clone(), sb.build(), config)
            .await
            .unwrap();
        for row in 0..64 {
            let mut doc = Document::new();
            doc.add_dense_vector(embedding, vec![row as f32 + 1.0, 1.0, 0.5, 0.25]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        writer.build_vector_index().await.unwrap();

        let target = DenseVectorConfig::ivf_tq(4, None, 1).with_target_vectors(1_000_000_000);
        let outcome = writer
            .alter_vector_index(embedding, VectorIndexAlter::Dense(target))
            .await
            .unwrap();

        assert_eq!(outcome.state, AlterVectorIndexState::DeferredFlat);
        let generation = writer.segment_manager.published_generation();
        let config = generation
            .schema
            .get_field_entry(embedding)
            .unwrap()
            .dense_vector_config
            .as_ref()
            .unwrap();
        assert_eq!(config.index_type, VectorIndexType::IvfTq);
        assert_eq!(config.num_clusters, None);
        assert_eq!(config.target_vectors, Some(1_000_000_000));
        assert!(
            !writer
                .segment_manager
                .read_metadata(|metadata| metadata.is_field_built(embedding.0))
                .await
        );
        assert!(generation.trained_vectors.is_none());
        for id in writer.segment_manager.get_segment_ids().await {
            let segment = crate::segment::SegmentReader::open(
                &dir,
                SegmentId::from_hex(&id).unwrap(),
                generation.schema.clone(),
                16,
            )
            .await
            .unwrap();
            assert!(segment.get_vector_index(embedding).is_none());
        }
    }

    #[tokio::test]
    async fn alter_ivf_scann_deferred_and_back_is_atomic() {
        let mut sb = SchemaBuilder::default();
        let embedding = sb.add_dense_vector_field_with_config(
            "embedding",
            true,
            true,
            DenseVectorConfig::ivf_tq(4, Some(1), 1),
        );
        let dir = crate::directories::RamDirectory::new();
        let config = crate::IndexConfig {
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            num_indexing_threads: 1,
            ..Default::default()
        };
        let mut writer = IndexWriter::create(dir.clone(), sb.build(), config)
            .await
            .unwrap();
        for row in 0..64 {
            let mut doc = Document::new();
            doc.add_dense_vector(embedding, vec![row as f32 + 1.0, 1.0, 0.5, 0.25]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        writer.build_vector_index().await.unwrap();

        // Leave a live worker cycle on the old schema. ALTER must flush this
        // generation before publication rather than letting an old IVF
        // SegmentBuilder cross the schema boundary.
        for row in 64..72 {
            let mut doc = Document::new();
            doc.add_dense_vector(embedding, vec![row as f32 + 1.0, 1.0, 0.5, 0.25]);
            writer.add_document(doc).unwrap();
        }

        let deferred = writer
            .alter_vector_index(embedding, VectorIndexAlter::Dense(scann_config(4, Some(2))))
            .await
            .unwrap();
        assert_eq!(deferred.state, AlterVectorIndexState::DeferredFlat);
        let deferred_generation = writer.segment_manager.published_generation();
        assert_eq!(
            deferred_generation
                .schema
                .get_field_entry(embedding)
                .unwrap()
                .dense_vector_config
                .as_ref()
                .unwrap()
                .index_type,
            VectorIndexType::Scann
        );
        assert!(
            !writer
                .segment_manager
                .read_metadata(|metadata| metadata.is_field_built(embedding.0))
                .await
        );
        assert!(deferred_generation.trained_vectors.is_none());

        // Workers resumed after ALTER must construct builders from the newly
        // published schema. A stale IVF builder here would either leave an IVF
        // ANN payload behind or make the segment unreadable as ScaNN.
        for row in 72..80 {
            let mut doc = Document::new();
            doc.add_dense_vector(embedding, vec![row as f32 + 1.0, 1.0, 0.5, 0.25]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        assert_eq!(
            writer
                .schema()
                .get_field_entry(embedding)
                .unwrap()
                .dense_vector_config
                .as_ref()
                .unwrap()
                .index_type,
            VectorIndexType::Scann
        );
        for id in writer.segment_manager.get_segment_ids().await {
            let segment = crate::segment::SegmentReader::open(
                &dir,
                SegmentId::from_hex(&id).unwrap(),
                deferred_generation.schema.clone(),
                16,
            )
            .await
            .unwrap();
            assert!(segment.get_vector_index(embedding).is_none());
        }

        let rebuilt = writer
            .alter_vector_index(
                embedding,
                VectorIndexAlter::Dense(DenseVectorConfig::ivf_tq(4, Some(1), 1)),
            )
            .await
            .unwrap();
        assert_eq!(rebuilt.state, AlterVectorIndexState::Built);
        assert!(rebuilt.publication_generation > deferred.publication_generation);
        let rebuilt_generation = writer.segment_manager.published_generation();
        assert_eq!(
            rebuilt_generation
                .schema
                .get_field_entry(embedding)
                .unwrap()
                .dense_vector_config
                .as_ref()
                .unwrap()
                .index_type,
            VectorIndexType::IvfTq
        );
        assert!(
            rebuilt_generation
                .trained_vectors
                .as_ref()
                .is_some_and(|trained| trained.centroids.contains_key(&embedding.0))
        );
        for id in writer.segment_manager.get_segment_ids().await {
            let segment = crate::segment::SegmentReader::open(
                &dir,
                SegmentId::from_hex(&id).unwrap(),
                rebuilt_generation.schema.clone(),
                16,
            )
            .await
            .unwrap();
            assert!(matches!(
                segment.get_vector_index(embedding),
                Some(crate::segment::VectorIndex::IvfTq { .. })
            ));
        }

        let ids_before_parameter_change = writer.segment_manager.get_segment_ids().await;
        let parameters_only = writer
            .alter_vector_index(
                embedding,
                VectorIndexAlter::Dense(DenseVectorConfig::ivf_tq(4, Some(1), 7)),
            )
            .await
            .unwrap();
        assert_eq!(parameters_only.state, AlterVectorIndexState::ParametersOnly);
        assert!(parameters_only.publication_generation > rebuilt.publication_generation);
        assert_eq!(
            writer.segment_manager.get_segment_ids().await,
            ids_before_parameter_change
        );
        assert_eq!(
            writer
                .schema()
                .get_field_entry(embedding)
                .unwrap()
                .dense_vector_config
                .as_ref()
                .unwrap()
                .nprobe,
            7
        );
    }
}
