//! Binary ScaNN routing and exact leaf scanning.
//!
//! Binary embeddings stay packed from training through serving. Routing uses
//! a configurable one-to-three-level k-majority tree and leaf scans compute
//! exact Hamming distances with Hermes' resolved AVX-512/AVX2/NEON/scalar
//! kernel. The trained tree is global; segment objects contain only leaf-local
//! document columns and exact packed codes, so compatible merges never train.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Range;

use rand::SeedableRng;

use super::{
    MAX_SCANN_TREE_LEVELS, MIN_PARTITION_TRAINING_POINTS_PER_LEAF, MIN_POINTS_FOR_PARTITIONING,
    ScannConfig, ScannEncoding, ScannFormatError, ScannGeometry, ScannLeafRun, ScannResult,
    ScannRoutingLevel, ScannSegmentPayload, ScannTrainedArtifact, ScannTrainedArtifactView,
    ScannTrainingState, desired_training_sample,
};
use crate::dsl::IvfRoutingMode;
use crate::structures::simd::HammingKernel;
use crate::structures::vector::index::{BinaryIvfConfig, train_binary_k_majority_codebook};
use crate::structures::vector::ivf::SoarConfig;
use crate::structures::vector::ivf::routing::allocate_child_clusters;

const HAMMING_SCAN_BLOCK: usize = 1_024;
const MAX_LOCAL_K_MAJORITY_BRANCHES: usize = 64;
const BINARY_SPILL_ASSIGNMENT_CANDIDATES: usize = 8;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BinaryScannTrainingStats {
    pub splits: usize,
    pub max_split_clusters: usize,
    pub max_depth: usize,
    pub retained_groups: usize,
    /// Largest temporary packed code matrix materialized for a non-contiguous
    /// row group. The complete retained sample is borrowed in place.
    pub max_materialized_training_bytes: usize,
}

/// Training controls for a global binary ScaNN tree.
///
/// Readiness and sample size are intentionally absent: both are hardcoded and
/// derived from `geometry`, matching the float ScaNN builder contract.
#[derive(Clone, Debug)]
pub struct BinaryScannTraining {
    pub dim_bits: u32,
    pub geometry: ScannGeometry,
    pub train_iters: usize,
    pub seed: u64,
}

impl BinaryScannTraining {
    pub fn validate(&self) -> ScannResult<()> {
        if self.dim_bits == 0 || !self.dim_bits.is_multiple_of(8) {
            return Err(ScannFormatError::new(
                "binary ScaNN dimension must be a positive multiple of eight bits",
            ));
        }
        let levels = usize::from(self.geometry.centroid_levels);
        if levels == 0
            || self.geometry.centroid_levels > MAX_SCANN_TREE_LEVELS
            || self.geometry.level_counts.len() != levels
            || self.geometry.level_counts.last().copied() != Some(self.geometry.num_leaves)
            || self.geometry.level_counts.contains(&0)
            || self
                .geometry
                .level_counts
                .windows(2)
                .any(|counts| counts[0] > counts[1])
        {
            return Err(ScannFormatError::new(
                "binary ScaNN needs a valid one-to-three-level cumulative geometry",
            ));
        }
        if self.train_iters == 0 {
            return Err(ScannFormatError::new(
                "binary ScaNN k-majority iterations must be positive",
            ));
        }
        Ok(())
    }

    /// The corpus floor is fixed in code and raised only when the chosen
    /// geometry needs the hardcoded minimum sample coverage for every
    /// terminal leaf.
    pub fn training_state(&self, observed: u64) -> ScannResult<ScannTrainingState> {
        self.validate()?;
        let geometry_required = u64::from(self.geometry.num_leaves)
            .checked_mul(MIN_PARTITION_TRAINING_POINTS_PER_LEAF)
            .ok_or_else(|| {
                ScannFormatError::new("binary ScaNN minimum training sample overflows u64")
            })?;
        let required = MIN_POINTS_FOR_PARTITIONING.max(geometry_required);
        Ok(if observed < required {
            ScannTrainingState::AwaitingData { observed, required }
        } else {
            ScannTrainingState::Ready { observed, required }
        })
    }

    pub fn desired_training_vectors(&self, observed: u64) -> ScannResult<u64> {
        self.validate()?;
        Ok(desired_training_sample(observed, self.geometry.num_leaves))
    }
}

#[derive(Clone, Debug)]
struct BinaryRoutingLevel {
    /// Packed child centroids, ordered by parent node.
    centroids: Vec<u8>,
    /// `parent_offsets[p]..parent_offsets[p + 1]` is parent `p`'s child run.
    parent_offsets: Vec<u32>,
}

/// Index-generation-scoped packed Hamming routing model.
#[derive(Clone, Debug)]
pub struct BinaryScannModel {
    dim_bits: u32,
    num_leaves: u32,
    levels: Vec<BinaryRoutingLevel>,
    fingerprint: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct QuantizedBinaryRoutingLevel {
    centroid_count: usize,
    centroid_codes: Range<usize>,
    parent_offsets: Vec<u32>,
}

/// Small executable metadata for mmap-backed packed-Hamming routing. The
/// potentially multi-gigabyte centroid planes remain in artifact storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedBinaryScannModel {
    dim_bits: u32,
    num_leaves: u32,
    artifact_id: u64,
    artifact_len: usize,
    levels: Vec<QuantizedBinaryRoutingLevel>,
    fingerprint: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedBinaryScannModelView<'a> {
    model: &'a QuantizedBinaryScannModel,
    artifact_bytes: &'a [u8],
}

impl BinaryScannModel {
    pub fn to_artifact(
        &self,
        generation: u64,
        trained_vectors: u64,
    ) -> ScannResult<ScannTrainedArtifact> {
        self.validate()?;
        let levels = self
            .levels
            .iter()
            .enumerate()
            .map(|(index, level)| ScannRoutingLevel {
                centroid_count: (level.centroids.len() / self.byte_len()) as u32,
                centroid_codes: level.centroids.clone(),
                minimums: Vec::new(),
                steps: Vec::new(),
                child_offsets: self
                    .levels
                    .get(index + 1)
                    .map_or_else(Vec::new, |next| next.parent_offsets.clone()),
            })
            .collect();
        ScannTrainedArtifact::new(
            generation,
            trained_vectors,
            ScannConfig {
                dimension: self.dim_bits,
                tree_levels: self.levels.len() as u8,
                num_leaves: self.num_leaves,
                encoding: ScannEncoding::BinaryHamming,
            },
            levels,
            None,
        )
    }

    pub fn from_artifact(artifact: &ScannTrainedArtifact) -> ScannResult<Self> {
        artifact.validate()?;
        if artifact.config.encoding != ScannEncoding::BinaryHamming {
            return Err(ScannFormatError::new(
                "float ScaNN artifact cannot be opened as a binary model",
            ));
        }
        let byte_len = artifact.config.dimension as usize / 8;
        let mut levels = Vec::with_capacity(artifact.levels.len());
        for (index, level) in artifact.levels.iter().enumerate() {
            let parent_offsets = if index == 0 {
                vec![0, level.centroid_count]
            } else {
                artifact.levels[index - 1].child_offsets.clone()
            };
            if level.centroid_codes.len() != level.centroid_count as usize * byte_len {
                return Err(ScannFormatError::new(
                    "binary ScaNN artifact centroid plane is inconsistent",
                ));
            }
            levels.push(BinaryRoutingLevel {
                centroids: level.centroid_codes.clone(),
                parent_offsets,
            });
        }
        let mut model = Self {
            dim_bits: artifact.config.dimension,
            num_leaves: artifact.config.num_leaves,
            levels,
            fingerprint: 0,
        };
        model.fingerprint = model.compute_fingerprint();
        model.validate()?;
        Ok(model)
    }

    pub fn train(
        training: &BinaryScannTraining,
        codes: &[u8],
        num_vectors: usize,
        index_label: &str,
    ) -> ScannResult<Self> {
        Self::train_with_stats(training, codes, num_vectors, index_label).map(|(model, _)| model)
    }

    pub fn train_with_stats(
        training: &BinaryScannTraining,
        codes: &[u8],
        num_vectors: usize,
        index_label: &str,
    ) -> ScannResult<(Self, BinaryScannTrainingStats)> {
        training.validate()?;
        match training.training_state(num_vectors as u64)? {
            ScannTrainingState::AwaitingData { observed, required } => {
                return Err(ScannFormatError::new(format!(
                    "binary ScaNN training deferred: geometry requires {required} vectors, observed {observed}"
                )));
            }
            ScannTrainingState::Ready { .. } => {}
        }
        let byte_len = usize::try_from(training.dim_bits / 8)
            .map_err(|_| ScannFormatError::new("binary ScaNN row size exceeds usize"))?;
        let expected = num_vectors
            .checked_mul(byte_len)
            .ok_or_else(|| ScannFormatError::new("binary ScaNN training matrix overflows"))?;
        if codes.len() != expected {
            return Err(ScannFormatError::new(format!(
                "binary ScaNN training matrix is truncated: expected {expected} bytes, got {}",
                codes.len()
            )));
        }

        let sample_count = usize::try_from(training.desired_training_vectors(num_vectors as u64)?)
            .map_err(|_| ScannFormatError::new("binary ScaNN sample count exceeds usize"))?;
        let mut groups = vec![deterministic_sample_rows(
            num_vectors,
            sample_count,
            training.seed,
        )];
        let mut levels = Vec::with_capacity(training.geometry.level_counts.len());
        let mut stats = BinaryScannTrainingStats::default();

        for (level_index, &level_count) in training.geometry.level_counts.iter().enumerate() {
            let group_sizes: Vec<usize> = groups.iter().map(BinaryTrainingRows::len).collect();
            let child_counts = allocate_child_clusters(&group_sizes, level_count as usize);
            if child_counts.iter().sum::<usize>() != level_count as usize {
                return Err(ScannFormatError::new(format!(
                    "binary ScaNN geometry level {level_index} cannot allocate {level_count} centroids from {sample_count} samples"
                )));
            }

            let mut parent_offsets = Vec::with_capacity(groups.len() + 1);
            let centroid_bytes = (level_count as usize)
                .checked_mul(byte_len)
                .ok_or_else(|| ScannFormatError::new("binary ScaNN centroid matrix overflows"))?;
            let mut centroids = Vec::with_capacity(centroid_bytes);
            let mut next_groups = Vec::with_capacity(level_count as usize);
            parent_offsets.push(0);
            let current_groups = std::mem::take(&mut groups);
            for (parent, (group, &children)) in
                current_groups.into_iter().zip(&child_counts).enumerate()
            {
                if children > 0 {
                    let partition = train_binary_partition(
                        codes,
                        &group,
                        byte_len,
                        training.dim_bits,
                        children,
                        training.train_iters,
                        derived_seed(training.seed, level_index, parent),
                        0,
                        level_index + 1 < training.geometry.level_counts.len(),
                        index_label,
                        &mut stats,
                    )?;
                    centroids.extend_from_slice(&partition.centroids);
                    if level_index + 1 < training.geometry.level_counts.len() {
                        next_groups.extend(partition.groups);
                    }
                }
                parent_offsets.push(u32::try_from(centroids.len() / byte_len).map_err(|_| {
                    ScannFormatError::new("binary ScaNN centroid identifier exceeds u32")
                })?);
            }
            debug_assert_eq!(centroids.len(), centroid_bytes);

            let is_leaf_level = level_index + 1 == training.geometry.level_counts.len();
            if !is_leaf_level {
                groups = next_groups;
            }
            levels.push(BinaryRoutingLevel {
                centroids,
                parent_offsets,
            });
        }

        let mut model = Self {
            dim_bits: training.dim_bits,
            num_leaves: training.geometry.num_leaves,
            levels,
            fingerprint: 0,
        };
        model.fingerprint = model.compute_fingerprint();
        model.validate()?;
        Ok((model, stats))
    }

    pub fn dim_bits(&self) -> u32 {
        self.dim_bits
    }

    pub fn num_leaves(&self) -> u32 {
        self.num_leaves
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub fn validate(&self) -> ScannResult<()> {
        if self.dim_bits == 0
            || !self.dim_bits.is_multiple_of(8)
            || self.levels.is_empty()
            || self.levels.len() > usize::from(MAX_SCANN_TREE_LEVELS)
        {
            return Err(ScannFormatError::new("invalid binary ScaNN model header"));
        }
        let byte_len = self.byte_len();
        let mut parents = 1usize;
        for level in &self.levels {
            if level.parent_offsets.len() != parents + 1
                || level.parent_offsets.first() != Some(&0)
                || level
                    .parent_offsets
                    .windows(2)
                    .any(|pair| pair[0] > pair[1])
            {
                return Err(ScannFormatError::new(
                    "invalid binary ScaNN parent directory",
                ));
            }
            let children = level.centroids.len() / byte_len;
            if level.centroids.len() % byte_len != 0
                || level.parent_offsets.last().copied() != Some(children as u32)
            {
                return Err(ScannFormatError::new(
                    "invalid binary ScaNN centroid matrix",
                ));
            }
            parents = children;
        }
        if parents != self.num_leaves as usize || self.compute_fingerprint() != self.fingerprint {
            return Err(ScannFormatError::new(
                "binary ScaNN leaf count or fingerprint is inconsistent",
            ));
        }
        Ok(())
    }

    /// Route once against the global tree. The resulting plan can be reused
    /// across every immutable segment in the active generation.
    pub fn probe(
        &self,
        query: &[u8],
        nprobe: usize,
        beam_width: usize,
        scratch: &mut BinaryScannSearchScratch,
    ) -> ScannResult<BinaryScannProbePlan> {
        if query.len() != self.byte_len() {
            return Err(ScannFormatError::new(
                "binary ScaNN query dimension does not match the model",
            ));
        }
        if nprobe == 0 || beam_width == 0 {
            return Err(ScannFormatError::new(
                "binary ScaNN nprobe and beam width must be positive",
            ));
        }
        let kernel = HammingKernel::resolve();
        scratch.frontier.clear();
        scratch.frontier.push(0);

        for (level_index, level) in self.levels.iter().enumerate() {
            scratch.candidates.clear();
            for &parent in &scratch.frontier {
                let parent = parent as usize;
                let start = level.parent_offsets[parent] as usize;
                let end = level.parent_offsets[parent + 1] as usize;
                let rows = end - start;
                scratch.distances.clear();
                scratch.distances.resize(rows, 0);
                kernel.distances(
                    query,
                    &level.centroids[start * self.byte_len()..end * self.byte_len()],
                    self.byte_len(),
                    &mut scratch.distances,
                );
                scratch
                    .candidates
                    .extend(
                        scratch
                            .distances
                            .iter()
                            .enumerate()
                            .map(|(local, &distance)| RouteCandidate {
                                node: (start + local) as u32,
                                distance,
                            }),
                    );
            }
            scratch.candidates.sort_unstable();
            let width = if level_index + 1 == self.levels.len() {
                nprobe.min(self.num_leaves as usize)
            } else {
                super::routing_prefix_for_child_coverage(
                    &scratch.candidates,
                    &self.levels[level_index + 1].parent_offsets,
                    beam_width,
                    nprobe,
                    |candidate| candidate.node as usize,
                )
            };
            scratch.frontier.clear();
            scratch.frontier.extend(
                scratch
                    .candidates
                    .iter()
                    .take(width)
                    .map(|candidate| candidate.node),
            );
            if scratch.frontier.is_empty() {
                return Err(ScannFormatError::new(
                    "binary ScaNN routing reached an empty branch",
                ));
            }
        }
        Ok(BinaryScannProbePlan {
            model_fingerprint: self.fingerprint,
            leaf_ids: scratch.frontier.clone(),
        })
    }

    pub fn assign(&self, code: &[u8], scratch: &mut BinaryScannSearchScratch) -> ScannResult<u32> {
        self.probe(code, 1, 1, scratch)?
            .leaf_ids
            .first()
            .copied()
            .ok_or_else(|| ScannFormatError::new("binary ScaNN assignment returned no leaf"))
    }

    /// Choose the normal primary leaf and the best alternate leaf reachable by
    /// a small widened tree probe. Packed bits do not have a meaningful float
    /// residual projection, so binary spilling uses exact centroid Hamming
    /// distance while retaining SOAR's one-secondary storage policy.
    pub fn spill_assignment(
        &self,
        code: &[u8],
        scratch: &mut BinaryScannSearchScratch,
    ) -> ScannResult<BinaryScannSpillAssignment> {
        let primary_leaf = self.assign(code, scratch)?;
        let candidate_count = BINARY_SPILL_ASSIGNMENT_CANDIDATES
            .min(self.num_leaves as usize)
            .max(1);
        let plan = self.probe(code, candidate_count, candidate_count, scratch)?;
        let kernel = HammingKernel::resolve();
        let leaf_centroids = &self
            .levels
            .last()
            .expect("validated binary ScaNN model has a terminal level")
            .centroids;
        let centroid = |leaf_id: u32| {
            let start = leaf_id as usize * self.byte_len();
            &leaf_centroids[start..start + self.byte_len()]
        };
        let primary_distance = kernel.distance(code, centroid(primary_leaf));
        let secondary_leaf = plan
            .leaf_ids
            .into_iter()
            .filter(|&leaf_id| leaf_id != primary_leaf)
            .map(|leaf_id| (kernel.distance(code, centroid(leaf_id)), leaf_id))
            .min()
            .map(|(_, leaf_id)| leaf_id);
        Ok(BinaryScannSpillAssignment {
            primary_leaf,
            secondary_leaf,
            primary_distance,
        })
    }

    /// Search any number of compatible segments with one shared routing plan.
    /// `doc_base` rebases segment-local IDs without touching their payload.
    pub fn search_segments(
        &self,
        query: &[u8],
        k: usize,
        nprobe: usize,
        beam_width: usize,
        segments: &[(&BinaryScannSegment, u32)],
        scratch: &mut BinaryScannSearchScratch,
    ) -> ScannResult<Vec<BinaryScannHit>> {
        let plan = self.probe(query, nprobe, beam_width, scratch)?;
        scratch.best_hit_keys.clear();
        let mut best = BinaryHeap::with_capacity(k.min(8_192));
        for &(segment, doc_base) in segments {
            segment.validate_for(self)?;
            segment.scan(query, &plan, doc_base, k, &mut best, scratch)?;
        }
        let mut hits = best.into_vec();
        hits.sort_unstable();
        Ok(hits)
    }

    fn byte_len(&self) -> usize {
        self.dim_bits as usize / 8
    }

    fn compute_fingerprint(&self) -> u64 {
        let mut hash = Fingerprint::new();
        hash.write(&self.dim_bits.to_le_bytes());
        hash.write(&self.num_leaves.to_le_bytes());
        hash.write(&(self.levels.len() as u32).to_le_bytes());
        for level in &self.levels {
            for offset in &level.parent_offsets {
                hash.write(&offset.to_le_bytes());
            }
            hash.write(&level.centroids);
        }
        hash.finish()
    }
}

impl QuantizedBinaryScannModel {
    pub fn from_artifact_view(artifact: &ScannTrainedArtifactView<'_>) -> ScannResult<Self> {
        if artifact.config.encoding != ScannEncoding::BinaryHamming {
            return Err(ScannFormatError::new(
                "float ScaNN artifact cannot be opened as a binary mmap model",
            ));
        }
        let byte_len = artifact.config.dimension as usize / 8;
        let mut levels = Vec::with_capacity(artifact.level_count());
        for index in 0..artifact.level_count() {
            let level = artifact.level(index).ok_or_else(|| {
                ScannFormatError::new("binary ScaNN artifact routing level disappeared")
            })?;
            let centroid_codes = artifact.level_centroid_codes_range(index).ok_or_else(|| {
                ScannFormatError::new("binary ScaNN artifact centroid range disappeared")
            })?;
            if centroid_codes.len() != level.centroid_count as usize * byte_len {
                return Err(ScannFormatError::new(
                    "binary ScaNN artifact centroid plane is inconsistent",
                ));
            }
            let parent_offsets = if index == 0 {
                vec![0, level.centroid_count]
            } else {
                artifact
                    .level(index - 1)
                    .expect("previous validated routing level exists")
                    .child_offsets()
                    .collect()
            };
            levels.push(QuantizedBinaryRoutingLevel {
                centroid_count: level.centroid_count as usize,
                centroid_codes,
                parent_offsets,
            });
        }
        let mut model = Self {
            dim_bits: artifact.config.dimension,
            num_leaves: artifact.config.num_leaves,
            artifact_id: artifact.artifact_id,
            artifact_len: artifact.bytes().len(),
            levels,
            fingerprint: 0,
        };
        model.fingerprint = model.compute_fingerprint(artifact.bytes());
        model.validate_metadata()?;
        Ok(model)
    }

    pub fn view<'a>(
        &'a self,
        artifact_bytes: &'a [u8],
    ) -> ScannResult<QuantizedBinaryScannModelView<'a>> {
        let stored_id = artifact_bytes
            .get(12..20)
            .and_then(|bytes| <[u8; 8]>::try_from(bytes).ok())
            .map(u64::from_le_bytes);
        if artifact_bytes.len() != self.artifact_len || stored_id != Some(self.artifact_id) {
            return Err(ScannFormatError::new(
                "quantized binary ScaNN model was paired with a different artifact mapping",
            ));
        }
        Ok(QuantizedBinaryScannModelView {
            model: self,
            artifact_bytes,
        })
    }

    pub fn dim_bits(&self) -> u32 {
        self.dim_bits
    }

    pub fn num_leaves(&self) -> u32 {
        self.num_leaves
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub fn estimated_memory_bytes(&self) -> usize {
        self.levels.iter().fold(0usize, |total, level| {
            total.saturating_add(level.parent_offsets.len() * std::mem::size_of::<u32>())
        })
    }

    fn byte_len(&self) -> usize {
        self.dim_bits as usize / 8
    }

    fn validate_metadata(&self) -> ScannResult<()> {
        if self.dim_bits == 0
            || !self.dim_bits.is_multiple_of(8)
            || self.levels.is_empty()
            || self.levels.len() > usize::from(MAX_SCANN_TREE_LEVELS)
            || self.levels.last().map(|level| level.centroid_count)
                != Some(self.num_leaves as usize)
        {
            return Err(ScannFormatError::new(
                "invalid quantized binary ScaNN model header",
            ));
        }
        let mut parents = 1usize;
        for level in &self.levels {
            if level.centroid_codes.end > self.artifact_len
                || level.centroid_codes.len()
                    != level.centroid_count.saturating_mul(self.byte_len())
                || level.parent_offsets.len() != parents + 1
                || level.parent_offsets.first() != Some(&0)
                || level.parent_offsets.last().copied() != Some(level.centroid_count as u32)
                || level
                    .parent_offsets
                    .windows(2)
                    .any(|pair| pair[0] > pair[1])
            {
                return Err(ScannFormatError::new(
                    "invalid quantized binary ScaNN routing level",
                ));
            }
            parents = level.centroid_count;
        }
        Ok(())
    }

    fn compute_fingerprint(&self, artifact_bytes: &[u8]) -> u64 {
        let mut hash = Fingerprint::new();
        hash.write(&self.dim_bits.to_le_bytes());
        hash.write(&self.num_leaves.to_le_bytes());
        hash.write(&(self.levels.len() as u32).to_le_bytes());
        for level in &self.levels {
            for offset in &level.parent_offsets {
                hash.write(&offset.to_le_bytes());
            }
            hash.write(&artifact_bytes[level.centroid_codes.clone()]);
        }
        hash.finish()
    }
}

impl QuantizedBinaryScannModelView<'_> {
    pub fn dim_bits(&self) -> u32 {
        self.model.dim_bits
    }

    pub fn num_leaves(&self) -> u32 {
        self.model.num_leaves
    }

    pub fn fingerprint(&self) -> u64 {
        self.model.fingerprint
    }

    pub fn probe(
        &self,
        query: &[u8],
        nprobe: usize,
        beam_width: usize,
        scratch: &mut BinaryScannSearchScratch,
    ) -> ScannResult<BinaryScannProbePlan> {
        if query.len() != self.model.byte_len() {
            return Err(ScannFormatError::new(
                "binary ScaNN query dimension does not match the model",
            ));
        }
        if nprobe == 0 || beam_width == 0 {
            return Err(ScannFormatError::new(
                "binary ScaNN nprobe and beam width must be positive",
            ));
        }
        let kernel = HammingKernel::resolve();
        scratch.frontier.clear();
        scratch.frontier.push(0);
        for (level_index, level) in self.model.levels.iter().enumerate() {
            scratch.candidates.clear();
            let centroids = &self.artifact_bytes[level.centroid_codes.clone()];
            for &parent in &scratch.frontier {
                let parent = parent as usize;
                let start = level.parent_offsets[parent] as usize;
                let end = level.parent_offsets[parent + 1] as usize;
                let rows = end - start;
                scratch.distances.clear();
                scratch.distances.resize(rows, 0);
                kernel.distances(
                    query,
                    &centroids[start * self.model.byte_len()..end * self.model.byte_len()],
                    self.model.byte_len(),
                    &mut scratch.distances,
                );
                scratch
                    .candidates
                    .extend(
                        scratch
                            .distances
                            .iter()
                            .enumerate()
                            .map(|(local, &distance)| RouteCandidate {
                                node: (start + local) as u32,
                                distance,
                            }),
                    );
            }
            scratch.candidates.sort_unstable();
            let width = if level_index + 1 == self.model.levels.len() {
                nprobe.min(self.model.num_leaves as usize)
            } else {
                super::routing_prefix_for_child_coverage(
                    &scratch.candidates,
                    &self.model.levels[level_index + 1].parent_offsets,
                    beam_width,
                    nprobe,
                    |candidate| candidate.node as usize,
                )
            };
            scratch.frontier.clear();
            scratch.frontier.extend(
                scratch
                    .candidates
                    .iter()
                    .take(width)
                    .map(|candidate| candidate.node),
            );
            if scratch.frontier.is_empty() {
                return Err(ScannFormatError::new(
                    "binary ScaNN routing reached an empty branch",
                ));
            }
        }
        Ok(BinaryScannProbePlan {
            model_fingerprint: self.model.fingerprint,
            leaf_ids: scratch.frontier.clone(),
        })
    }

    pub fn assign(&self, code: &[u8], scratch: &mut BinaryScannSearchScratch) -> ScannResult<u32> {
        self.probe(code, 1, 1, scratch)?
            .leaf_ids
            .first()
            .copied()
            .ok_or_else(|| ScannFormatError::new("binary ScaNN assignment returned no leaf"))
    }

    /// Mmap-backed equivalent of [`BinaryScannModel::spill_assignment`]. The
    /// centroid plane remains borrowed from the artifact mapping.
    pub fn spill_assignment(
        &self,
        code: &[u8],
        scratch: &mut BinaryScannSearchScratch,
    ) -> ScannResult<BinaryScannSpillAssignment> {
        let primary_leaf = self.assign(code, scratch)?;
        let candidate_count = BINARY_SPILL_ASSIGNMENT_CANDIDATES
            .min(self.model.num_leaves as usize)
            .max(1);
        let plan = self.probe(code, candidate_count, candidate_count, scratch)?;
        let kernel = HammingKernel::resolve();
        let terminal = self
            .model
            .levels
            .last()
            .expect("validated binary ScaNN model has a terminal level");
        let leaf_centroids = &self.artifact_bytes[terminal.centroid_codes.clone()];
        let centroid = |leaf_id: u32| {
            let start = leaf_id as usize * self.model.byte_len();
            &leaf_centroids[start..start + self.model.byte_len()]
        };
        let primary_distance = kernel.distance(code, centroid(primary_leaf));
        let secondary_leaf = plan
            .leaf_ids
            .into_iter()
            .filter(|&leaf_id| leaf_id != primary_leaf)
            .map(|leaf_id| (kernel.distance(code, centroid(leaf_id)), leaf_id))
            .min()
            .map(|(_, leaf_id)| leaf_id);
        Ok(BinaryScannSpillAssignment {
            primary_leaf,
            secondary_leaf,
            primary_distance,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BinaryScannProbePlan {
    pub model_fingerprint: u64,
    pub leaf_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RouteCandidate {
    node: u32,
    distance: u32,
}

impl Ord for RouteCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for RouteCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Per-query allocations retained by the caller and reused across segments.
#[derive(Default, Debug)]
pub struct BinaryScannSearchScratch {
    frontier: Vec<u32>,
    candidates: Vec<RouteCandidate>,
    distances: Vec<u32>,
    /// Logical vector IDs currently represented in the top-k heap. Tracking
    /// only retained hits keeps secondary-posting deduplication bounded by k,
    /// rather than by the number of postings scanned.
    best_hit_keys: rustc_hash::FxHashSet<(u32, u16)>,
}

/// Deterministic packed-Hamming primary and optional secondary candidate.
/// Policy code decides whether to retain the secondary under its storage cap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BinaryScannSpillAssignment {
    pub primary_leaf: u32,
    pub secondary_leaf: Option<u32>,
    pub primary_distance: u32,
}

#[derive(Clone, Debug)]
struct BinaryScannLeaf {
    leaf_id: u32,
    doc_ids: Vec<u32>,
    ordinals: Vec<u16>,
    codes: Vec<u8>,
}

fn push_binary_posting(
    grouped: &mut rustc_hash::FxHashMap<u32, BinaryScannLeaf>,
    leaf_id: u32,
    doc_id: u32,
    ordinal: u16,
    code: &[u8],
) {
    let leaf = grouped.entry(leaf_id).or_insert_with(|| BinaryScannLeaf {
        leaf_id,
        doc_ids: Vec::new(),
        ordinals: Vec::new(),
        codes: Vec::new(),
    });
    leaf.doc_ids.push(doc_id);
    leaf.ordinals.push(ordinal);
    leaf.codes.extend_from_slice(code);
}

/// Immutable segment-local exact binary payload.
#[derive(Clone, Debug)]
pub struct BinaryScannSegment {
    dim_bits: u32,
    model_fingerprint: u64,
    num_leaves: u32,
    leaves: Vec<BinaryScannLeaf>,
    /// Logical vectors before optional secondary posting expansion.
    len: usize,
    /// Physical postings, bounded to at most two per logical vector.
    stored_len: usize,
}

impl BinaryScannSegment {
    pub fn build(
        model: &BinaryScannModel,
        codes: &[u8],
        doc_id_ordinals: &[(u32, u16)],
        scratch: &mut BinaryScannSearchScratch,
    ) -> ScannResult<Self> {
        Self::build_internal(model, codes, doc_id_ordinals, None, scratch)
    }

    /// Build with deterministic one-secondary binary spilling.
    ///
    /// A negative `spill_threshold` keeps `SoarConfig`'s target-fraction tag:
    /// the most poorly represented primary assignments are retained up to a
    /// strict segment-local storage budget. Explicit non-negative thresholds
    /// use the same primary residual rule as float SOAR, with squared L2 over
    /// bits represented exactly by Hamming distance.
    pub fn build_with_soar(
        model: &BinaryScannModel,
        codes: &[u8],
        doc_id_ordinals: &[(u32, u16)],
        soar: &SoarConfig,
        scratch: &mut BinaryScannSearchScratch,
    ) -> ScannResult<Self> {
        Self::build_internal(model, codes, doc_id_ordinals, Some(soar), scratch)
    }

    fn build_internal(
        model: &BinaryScannModel,
        codes: &[u8],
        doc_id_ordinals: &[(u32, u16)],
        soar: Option<&SoarConfig>,
        scratch: &mut BinaryScannSearchScratch,
    ) -> ScannResult<Self> {
        let expected = doc_id_ordinals
            .len()
            .checked_mul(model.byte_len())
            .ok_or_else(|| ScannFormatError::new("binary ScaNN segment size overflows"))?;
        if codes.len() != expected {
            return Err(ScannFormatError::new(
                "binary ScaNN segment code and label columns are inconsistent",
            ));
        }
        if soar.is_some_and(|config| !config.spill_threshold.is_finite()) {
            return Err(ScannFormatError::new(
                "binary ScaNN spill threshold must be finite",
            ));
        }

        let spill_enabled =
            soar.is_some_and(|config| config.num_secondary > 0) && model.num_leaves > 1;
        if !spill_enabled {
            // Preserve the allocation profile of the established primary-only
            // builder. Spill ranking state is paid only when spilling is
            // explicitly enabled for this segment.
            let mut grouped = rustc_hash::FxHashMap::<u32, BinaryScannLeaf>::default();
            for (&(doc_id, ordinal), code) in doc_id_ordinals
                .iter()
                .zip(codes.chunks_exact(model.byte_len()))
            {
                let primary_leaf = model.assign(code, scratch)?;
                push_binary_posting(&mut grouped, primary_leaf, doc_id, ordinal, code);
            }
            let mut leaves: Vec<_> = grouped.into_values().collect();
            leaves.sort_unstable_by_key(|leaf| leaf.leaf_id);
            let segment = Self {
                dim_bits: model.dim_bits,
                model_fingerprint: model.fingerprint,
                num_leaves: model.num_leaves,
                leaves,
                len: doc_id_ordinals.len(),
                stored_len: doc_id_ordinals.len(),
            };
            segment.validate_for(model)?;
            return Ok(segment);
        }

        let mut assignments = Vec::with_capacity(doc_id_ordinals.len());
        for code in codes.chunks_exact(model.byte_len()) {
            assignments.push(model.spill_assignment(code, scratch)?);
        }

        if let Some(config) = soar {
            if let Some(target_fraction) = config.calibration_target() {
                // Floor, rather than round, makes the target a strict storage
                // ceiling even for tiny streaming segments.
                let spill_budget = ((assignments.len() as f64 * f64::from(target_fraction)).floor()
                    as usize)
                    .min(assignments.len());
                let mut ranked: Vec<usize> = assignments
                    .iter()
                    .enumerate()
                    .filter_map(|(row, assignment)| assignment.secondary_leaf.map(|_| row))
                    .collect();
                ranked.sort_unstable_by(|&left, &right| {
                    assignments[right]
                        .primary_distance
                        .cmp(&assignments[left].primary_distance)
                        .then_with(|| left.cmp(&right))
                });
                for &row in ranked.iter().skip(spill_budget) {
                    assignments[row].secondary_leaf = None;
                }
            } else if config.selective {
                let threshold_sq = f64::from(config.spill_threshold).powi(2);
                for assignment in &mut assignments {
                    if f64::from(assignment.primary_distance) < threshold_sq {
                        assignment.secondary_leaf = None;
                    }
                }
            }
        }

        let mut grouped = rustc_hash::FxHashMap::<u32, BinaryScannLeaf>::default();
        for ((&(doc_id, ordinal), code), assignment) in doc_id_ordinals
            .iter()
            .zip(codes.chunks_exact(model.byte_len()))
            .zip(assignments)
        {
            push_binary_posting(&mut grouped, assignment.primary_leaf, doc_id, ordinal, code);
            if let Some(secondary_leaf) = assignment.secondary_leaf {
                push_binary_posting(&mut grouped, secondary_leaf, doc_id, ordinal, code);
            }
        }
        let mut leaves: Vec<_> = grouped.into_values().collect();
        leaves.sort_unstable_by_key(|leaf| leaf.leaf_id);
        let stored_len = leaves.iter().map(|leaf| leaf.doc_ids.len()).sum();
        let segment = Self {
            dim_bits: model.dim_bits,
            model_fingerprint: model.fingerprint,
            num_leaves: model.num_leaves,
            leaves,
            len: doc_id_ordinals.len(),
            stored_len,
        };
        segment.validate_for(model)?;
        Ok(segment)
    }

    /// Leaf-wise compatible merge. Codes are copied verbatim and no routing or
    /// training runs; only segment-local document IDs are rebased.
    pub fn merge_compatible(
        model: &BinaryScannModel,
        segments: &[(&Self, u32)],
    ) -> ScannResult<Self> {
        for &(segment, _) in segments {
            segment.validate_for(model)?;
        }
        let mut cursors = vec![0usize; segments.len()];
        let mut queue = BinaryHeap::new();
        for (segment_index, (segment, _)) in segments.iter().enumerate() {
            if let Some(first) = segment.leaves.first() {
                queue.push(Reverse((first.leaf_id, segment_index)));
            }
        }
        let mut leaves: Vec<BinaryScannLeaf> = Vec::new();
        let len = segments.iter().try_fold(0usize, |total, (segment, _)| {
            total
                .checked_add(segment.len)
                .ok_or_else(|| ScannFormatError::new("binary ScaNN merge count overflows"))
        })?;
        let mut stored_len = 0usize;
        while let Some(Reverse((leaf_id, segment_index))) = queue.pop() {
            let (segment, doc_base) = segments[segment_index];
            let source = &segment.leaves[cursors[segment_index]];
            if leaves.last().is_none_or(|leaf| leaf.leaf_id != leaf_id) {
                leaves.push(BinaryScannLeaf {
                    leaf_id,
                    doc_ids: Vec::new(),
                    ordinals: Vec::new(),
                    codes: Vec::new(),
                });
            }
            let target = leaves.last_mut().expect("leaf was just inserted");
            target.doc_ids.reserve(source.doc_ids.len());
            for &doc_id in &source.doc_ids {
                target
                    .doc_ids
                    .push(doc_id.checked_add(doc_base).ok_or_else(|| {
                        ScannFormatError::new("binary ScaNN merge document ID overflows u32")
                    })?);
            }
            target.ordinals.extend_from_slice(&source.ordinals);
            target.codes.extend_from_slice(&source.codes);
            stored_len = stored_len
                .checked_add(source.doc_ids.len())
                .ok_or_else(|| ScannFormatError::new("binary ScaNN merge count overflows"))?;
            cursors[segment_index] += 1;
            if let Some(next) = segment.leaves.get(cursors[segment_index]) {
                queue.push(Reverse((next.leaf_id, segment_index)));
            }
        }
        let merged = Self {
            dim_bits: model.dim_bits,
            model_fingerprint: model.fingerprint,
            num_leaves: model.num_leaves,
            leaves,
            len,
            stored_len,
        };
        merged.validate_for(model)?;
        Ok(merged)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of physical leaf postings after secondary spill expansion.
    pub fn stored_len(&self) -> usize {
        self.stored_len
    }

    pub fn to_payload(
        &self,
        model: &BinaryScannModel,
        artifact: &ScannTrainedArtifact,
        doc_count: u32,
    ) -> ScannResult<ScannSegmentPayload> {
        self.validate_for(model)?;
        let reopened = BinaryScannModel::from_artifact(artifact)?;
        if reopened.fingerprint != model.fingerprint {
            return Err(ScannFormatError::new(
                "binary ScaNN segment model does not match the persisted artifact",
            ));
        }
        let mut runs = Vec::with_capacity(self.leaves.len());
        for leaf in &self.leaves {
            runs.push(ScannLeafRun::from_rows(
                leaf.leaf_id,
                0,
                &leaf.doc_ids,
                &leaf.ordinals,
                leaf.codes.clone(),
                ScannEncoding::BinaryHamming,
                self.dim_bits,
            )?);
        }
        ScannSegmentPayload::new(artifact, doc_count, runs)
    }

    fn validate_for(&self, model: &BinaryScannModel) -> ScannResult<()> {
        if self.dim_bits != model.dim_bits
            || self.model_fingerprint != model.fingerprint
            || self.num_leaves != model.num_leaves
        {
            return Err(ScannFormatError::new(
                "binary ScaNN segment belongs to a different trained generation",
            ));
        }
        let byte_len = model.byte_len();
        let mut previous = None;
        let mut total = 0usize;
        for leaf in &self.leaves {
            if leaf.leaf_id >= self.num_leaves
                || previous.is_some_and(|previous| previous >= leaf.leaf_id)
                || leaf.doc_ids.len() != leaf.ordinals.len()
                || leaf.codes.len() != leaf.doc_ids.len().saturating_mul(byte_len)
            {
                return Err(ScannFormatError::new(
                    "binary ScaNN leaf directory or columns are inconsistent",
                ));
            }
            previous = Some(leaf.leaf_id);
            total = total
                .checked_add(leaf.doc_ids.len())
                .ok_or_else(|| ScannFormatError::new("binary ScaNN segment count overflows"))?;
        }
        let maximum_stored = self
            .len
            .checked_mul(2)
            .ok_or_else(|| ScannFormatError::new("binary ScaNN segment count overflows"))?;
        if total != self.stored_len
            || self.stored_len < self.len
            || self.stored_len > maximum_stored
        {
            return Err(ScannFormatError::new(
                "binary ScaNN segment vector count is inconsistent",
            ));
        }
        Ok(())
    }

    fn scan(
        &self,
        query: &[u8],
        plan: &BinaryScannProbePlan,
        doc_base: u32,
        k: usize,
        best: &mut BinaryHeap<BinaryScannHit>,
        scratch: &mut BinaryScannSearchScratch,
    ) -> ScannResult<()> {
        if plan.model_fingerprint != self.model_fingerprint {
            return Err(ScannFormatError::new(
                "binary ScaNN probe plan belongs to a different trained generation",
            ));
        }
        let byte_len = self.dim_bits as usize / 8;
        let kernel = HammingKernel::resolve();
        for &leaf_id in &plan.leaf_ids {
            let Ok(position) = self
                .leaves
                .binary_search_by_key(&leaf_id, |leaf| leaf.leaf_id)
            else {
                continue;
            };
            let leaf = &self.leaves[position];
            for start in (0..leaf.doc_ids.len()).step_by(HAMMING_SCAN_BLOCK) {
                let rows = HAMMING_SCAN_BLOCK.min(leaf.doc_ids.len() - start);
                scratch.distances.clear();
                scratch.distances.resize(rows, 0);
                kernel.distances(
                    query,
                    &leaf.codes[start * byte_len..(start + rows) * byte_len],
                    byte_len,
                    &mut scratch.distances,
                );
                for (local, &distance) in scratch.distances.iter().enumerate() {
                    if k == 0 {
                        continue;
                    }
                    let index = start + local;
                    let doc_id = leaf.doc_ids[index].checked_add(doc_base).ok_or_else(|| {
                        ScannFormatError::new("binary ScaNN query document ID overflows u32")
                    })?;
                    let ordinal = leaf.ordinals[index];
                    if scratch.best_hit_keys.contains(&(doc_id, ordinal)) {
                        continue;
                    }
                    let hit = BinaryScannHit {
                        doc_id,
                        ordinal,
                        distance,
                    };
                    if best.len() < k {
                        best.push(hit);
                        scratch.best_hit_keys.insert((doc_id, ordinal));
                    } else if best.peek().is_some_and(|worst| hit < *worst) {
                        let evicted = best.pop().expect("non-empty top-k heap has a worst hit");
                        scratch
                            .best_hit_keys
                            .remove(&(evicted.doc_id, evicted.ordinal));
                        best.push(hit);
                        scratch.best_hit_keys.insert((doc_id, ordinal));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Exact Hamming result. Lower distance wins; ties are stable by document and
/// ordinal so segment layout and merge order cannot change the answer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BinaryScannHit {
    pub doc_id: u32,
    pub ordinal: u16,
    pub distance: u32,
}

impl Ord for BinaryScannHit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
            .then_with(|| self.ordinal.cmp(&other.ordinal))
    }
}

impl PartialOrd for BinaryScannHit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct BinaryPartition {
    centroids: Vec<u8>,
    groups: Vec<BinaryTrainingRows>,
}

/// A partition is an ordered view into the caller-owned packed code matrix.
/// The initial complete sample is a range and costs no additional memory;
/// child partitions retain only row identifiers instead of cloning codes.
#[derive(Debug)]
enum BinaryTrainingRows {
    Contiguous(Range<usize>),
    Indexed(Vec<usize>),
}

impl BinaryTrainingRows {
    fn len(&self) -> usize {
        match self {
            Self::Contiguous(range) => range.len(),
            Self::Indexed(indices) => indices.len(),
        }
    }

    fn source_row(&self, position: usize) -> usize {
        match self {
            Self::Contiguous(range) => range.start + position,
            Self::Indexed(indices) => indices[position],
        }
    }

    fn from_indices(indices: Vec<usize>) -> Self {
        let Some(&first) = indices.first() else {
            return Self::Indexed(indices);
        };
        if indices
            .iter()
            .enumerate()
            .all(|(offset, &row)| row == first + offset)
        {
            Self::Contiguous(first..first + indices.len())
        } else {
            Self::Indexed(indices)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn train_binary_partition(
    source_codes: &[u8],
    rows: &BinaryTrainingRows,
    byte_len: usize,
    dim_bits: u32,
    clusters: usize,
    train_iters: usize,
    seed: u64,
    depth: usize,
    retain_groups: bool,
    index_label: &str,
    stats: &mut BinaryScannTrainingStats,
) -> ScannResult<BinaryPartition> {
    let row_count = rows.len();
    if row_count == 0 || byte_len == 0 || clusters == 0 || clusters > row_count {
        return Err(ScannFormatError::new(
            "invalid recursive binary ScaNN partition shape",
        ));
    }
    stats.max_depth = stats.max_depth.max(depth);
    if clusters == row_count {
        let groups = if retain_groups {
            let groups: Vec<BinaryTrainingRows> = (0..row_count)
                .map(|position| {
                    let row = rows.source_row(position);
                    BinaryTrainingRows::Contiguous(row..row + 1)
                })
                .collect();
            stats.retained_groups = stats.retained_groups.saturating_add(groups.len());
            groups
        } else {
            Vec::new()
        };
        return Ok(BinaryPartition {
            centroids: materialize_training_rows(source_codes, rows, byte_len)?,
            groups,
        });
    }

    let branches = training_branch_factor(clusters).min(row_count);
    let mut config = BinaryIvfConfig::new(dim_bits as usize, branches);
    config.routing = IvfRoutingMode::Flat;
    config.train_iters = train_iters;
    config.max_train_samples = row_count;
    config.seed = seed;
    let local_centroids =
        train_binary_codebook_for_rows(&config, source_codes, rows, byte_len, index_label, stats)?;
    stats.splits = stats.splits.saturating_add(1);
    stats.max_split_clusters = stats.max_split_clusters.max(branches);
    if branches == clusters {
        let groups = if retain_groups {
            let groups =
                partition_one_group_nonempty(source_codes, rows, &local_centroids, byte_len);
            stats.retained_groups = stats.retained_groups.saturating_add(groups.len());
            groups
        } else {
            Vec::new()
        };
        return Ok(BinaryPartition {
            groups,
            centroids: local_centroids,
        });
    }

    let local_groups = partition_one_group_nonempty(source_codes, rows, &local_centroids, byte_len);
    let sizes: Vec<usize> = local_groups.iter().map(BinaryTrainingRows::len).collect();
    let allocations = allocate_child_clusters(&sizes, clusters);
    if allocations.iter().sum::<usize>() != clusters || allocations.contains(&0) {
        return Err(ScannFormatError::new(
            "recursive binary ScaNN centroid allocation is inconsistent",
        ));
    }
    let mut centroids = Vec::with_capacity(clusters.saturating_mul(byte_len));
    let mut groups = Vec::with_capacity(clusters);
    for (branch, (group, &allocation)) in local_groups.iter().zip(&allocations).enumerate() {
        let child = train_binary_partition(
            source_codes,
            group,
            byte_len,
            dim_bits,
            allocation,
            train_iters,
            derived_seed(seed, depth, branch),
            depth + 1,
            retain_groups,
            index_label,
            stats,
        )?;
        centroids.extend_from_slice(&child.centroids);
        if retain_groups {
            groups.extend(child.groups);
        }
    }
    Ok(BinaryPartition { centroids, groups })
}

fn training_branch_factor(clusters: usize) -> usize {
    if clusters <= MAX_LOCAL_K_MAJORITY_BRANCHES {
        clusters
    } else {
        ((clusters as f64).sqrt().ceil() as usize).clamp(2, MAX_LOCAL_K_MAJORITY_BRANCHES)
    }
}

fn deterministic_sample_rows(
    num_vectors: usize,
    sample_count: usize,
    seed: u64,
) -> BinaryTrainingRows {
    if sample_count >= num_vectors {
        return BinaryTrainingRows::Contiguous(0..num_vectors);
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut indices = rand::seq::index::sample(&mut rng, num_vectors, sample_count).into_vec();
    indices.sort_unstable();
    BinaryTrainingRows::from_indices(indices)
}

fn materialize_training_rows(
    source_codes: &[u8],
    rows: &BinaryTrainingRows,
    byte_len: usize,
) -> ScannResult<Vec<u8>> {
    let capacity = rows
        .len()
        .checked_mul(byte_len)
        .ok_or_else(|| ScannFormatError::new("binary ScaNN group matrix overflows"))?;
    let mut packed = Vec::with_capacity(capacity);
    for position in 0..rows.len() {
        let row = rows.source_row(position);
        let start = row
            .checked_mul(byte_len)
            .ok_or_else(|| ScannFormatError::new("binary ScaNN source row overflows"))?;
        let code = source_codes
            .get(start..start + byte_len)
            .ok_or_else(|| ScannFormatError::new("binary ScaNN source row is truncated"))?;
        packed.extend_from_slice(code);
    }
    Ok(packed)
}

fn train_binary_codebook_for_rows(
    config: &BinaryIvfConfig,
    source_codes: &[u8],
    rows: &BinaryTrainingRows,
    byte_len: usize,
    index_label: &str,
    stats: &mut BinaryScannTrainingStats,
) -> ScannResult<Vec<u8>> {
    match rows {
        BinaryTrainingRows::Contiguous(range) => {
            let start = range
                .start
                .checked_mul(byte_len)
                .ok_or_else(|| ScannFormatError::new("binary ScaNN group offset overflows"))?;
            let end = range
                .end
                .checked_mul(byte_len)
                .ok_or_else(|| ScannFormatError::new("binary ScaNN group offset overflows"))?;
            let codes = source_codes
                .get(start..end)
                .ok_or_else(|| ScannFormatError::new("binary ScaNN group is truncated"))?;
            train_binary_k_majority_codebook(config, codes, rows.len(), index_label)
                .map_err(|error| ScannFormatError::new(error.to_string()))
        }
        BinaryTrainingRows::Indexed(_) => {
            let packed = materialize_training_rows(source_codes, rows, byte_len)?;
            stats.max_materialized_training_bytes =
                stats.max_materialized_training_bytes.max(packed.len());
            train_binary_k_majority_codebook(config, &packed, rows.len(), index_label)
                .map_err(|error| ScannFormatError::new(error.to_string()))
        }
    }
}

fn derived_seed(seed: u64, level: usize, parent: usize) -> u64 {
    seed ^ (level as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
        ^ (parent as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

fn partition_one_group_nonempty(
    source_codes: &[u8],
    rows: &BinaryTrainingRows,
    centroids: &[u8],
    byte_len: usize,
) -> Vec<BinaryTrainingRows> {
    let kernel = HammingKernel::resolve();
    let child_count = centroids.len() / byte_len;
    let row_count = rows.len();
    let mut assignments = vec![0usize; row_count];
    let mut assignment_distances = vec![0u32; row_count];
    let mut counts = vec![0usize; child_count];
    let mut distances = vec![0u32; child_count];
    for position in 0..row_count {
        let source_row = rows.source_row(position);
        let offset = source_row * byte_len;
        let code = &source_codes[offset..offset + byte_len];
        kernel.distances(code, centroids, byte_len, &mut distances);
        let (child, &distance) = distances
            .iter()
            .enumerate()
            .min_by_key(|&(child, distance)| (*distance, child))
            .expect("a populated routing parent has children");
        assignments[position] = child;
        assignment_distances[position] = distance;
        counts[child] += 1;
    }
    for empty in 0..child_count {
        if counts[empty] != 0 {
            continue;
        }
        let replacement = (0..row_count)
            .filter(|&row| counts[assignments[row]] > 1)
            .max_by_key(|&row| (assignment_distances[row], Reverse(row)))
            .expect("training readiness guarantees one sample per centroid");
        counts[assignments[replacement]] -= 1;
        assignments[replacement] = empty;
        assignment_distances[replacement] = 0;
        counts[empty] = 1;
    }
    let mut groups: Vec<Vec<usize>> = counts
        .iter()
        .map(|&count| Vec::with_capacity(count))
        .collect();
    for (position, &child) in assignments.iter().enumerate() {
        groups[child].push(rows.source_row(position));
    }
    groups
        .into_iter()
        .map(BinaryTrainingRows::from_indices)
        .collect()
}

struct Fingerprint(u64);

impl Fingerprint {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::simd::hamming_distance;

    fn corpus(rows: usize) -> Vec<u8> {
        let anchors = [
            [0x00, 0x00],
            [0xff, 0xff],
            [0x0f, 0x0f],
            [0xf0, 0xf0],
            [0xaa, 0xaa],
            [0x55, 0x55],
            [0x33, 0xcc],
            [0xcc, 0x33],
        ];
        let mut codes = Vec::with_capacity(rows * 2);
        for row in 0..rows {
            let mut code = anchors[row % anchors.len()];
            code[(row / anchors.len()) % 2] ^= 1 << ((row / 16) % 8);
            codes.extend_from_slice(&code);
        }
        codes
    }

    fn training() -> BinaryScannTraining {
        BinaryScannTraining {
            dim_bits: 16,
            geometry: ScannGeometry {
                centroid_levels: 3,
                num_leaves: 8,
                level_counts: vec![2, 4, 8],
            },
            train_iters: 2,
            seed: 73,
        }
    }

    fn two_leaf_model() -> BinaryScannModel {
        let mut model = BinaryScannModel {
            dim_bits: 8,
            num_leaves: 2,
            levels: vec![BinaryRoutingLevel {
                centroids: vec![0x00, 0xff],
                parent_offsets: vec![0, 2],
            }],
            fingerprint: 0,
        };
        model.fingerprint = model.compute_fingerprint();
        model.validate().unwrap();
        model
    }

    #[test]
    fn readiness_is_derived_from_geometry_not_user_configuration() {
        let training = training();
        assert_eq!(
            training.training_state(99_999).unwrap(),
            ScannTrainingState::AwaitingData {
                observed: 99_999,
                required: 100_000,
            }
        );
        assert_eq!(training.desired_training_vectors(500_000).unwrap(), 100_000);
    }

    #[test]
    fn large_binary_partition_never_flat_trains_terminal_leaf_count() {
        let codes = corpus(100_000);
        let training = BinaryScannTraining {
            dim_bits: 16,
            geometry: ScannGeometry {
                centroid_levels: 1,
                num_leaves: 1_024,
                level_counts: vec![1_024],
            },
            train_iters: 1,
            seed: 91,
        };
        let (model, stats) =
            BinaryScannModel::train_with_stats(&training, &codes, 100_000, "test").unwrap();
        assert_eq!(model.num_leaves(), 1_024);
        assert!(stats.splits > 1);
        assert!(
            stats.max_split_clusters <= MAX_LOCAL_K_MAJORITY_BRANCHES,
            "binary local split widened to {} clusters",
            stats.max_split_clusters,
        );
        assert!(stats.max_split_clusters < 1_024);
        assert_eq!(
            stats.retained_groups, 0,
            "terminal training must not allocate one Vec per leaf",
        );
        assert!(
            stats.max_materialized_training_bytes < codes.len(),
            "binary training cloned the complete {}-byte retained sample",
            codes.len(),
        );
    }

    #[test]
    fn full_probe_widens_past_the_legacy_sixty_four_parent_beam() {
        let root_count = 65usize;
        let leaf_count = root_count * root_count;
        let mut leaf_offsets = Vec::with_capacity(root_count + 1);
        for parent in 0..=root_count {
            leaf_offsets.push((parent * root_count) as u32);
        }
        let mut model = BinaryScannModel {
            dim_bits: 8,
            num_leaves: leaf_count as u32,
            levels: vec![
                BinaryRoutingLevel {
                    centroids: vec![0; root_count],
                    parent_offsets: vec![0, root_count as u32],
                },
                BinaryRoutingLevel {
                    centroids: vec![0; leaf_count],
                    parent_offsets: leaf_offsets,
                },
            ],
            fingerprint: 0,
        };
        model.fingerprint = model.compute_fingerprint();
        model.validate().unwrap();

        let mut scratch = BinaryScannSearchScratch::default();
        let owned = model.probe(&[0], leaf_count, 64, &mut scratch).unwrap();
        assert_eq!(owned.leaf_ids.len(), leaf_count);
        assert_eq!(owned.leaf_ids, (0..leaf_count as u32).collect::<Vec<_>>());

        let artifact = model.to_artifact(7, 100_000).unwrap();
        let bytes = artifact.to_bytes().unwrap();
        let artifact = ScannTrainedArtifactView::parse(&bytes).unwrap();
        let quantized = QuantizedBinaryScannModel::from_artifact_view(&artifact).unwrap();
        let view = quantized.view(&bytes).unwrap();
        let mapped = view.probe(&[0], leaf_count, 64, &mut scratch).unwrap();
        assert_eq!(mapped.leaf_ids, owned.leaf_ids);
    }

    #[test]
    fn packed_hamming_search_is_exact_and_merge_independent() {
        let training_codes = corpus(100_000);
        let model = BinaryScannModel::train(&training(), &training_codes, 100_000, "test").unwrap();
        let rebuilt =
            BinaryScannModel::train(&training(), &training_codes, 100_000, "test").unwrap();
        assert_eq!(rebuilt.fingerprint(), model.fingerprint());
        let mut scratch = BinaryScannSearchScratch::default();
        let query = [0b1010_1011, 0b1010_1010];
        let first = model.probe(&query, 4, 2, &mut scratch).unwrap();
        let second = rebuilt.probe(&query, 4, 2, &mut scratch).unwrap();
        assert_eq!(first, second);
        let artifact = model.to_artifact(11, 100_000).unwrap();
        let artifact_bytes = artifact.to_bytes().unwrap();
        let artifact_view = ScannTrainedArtifactView::parse(&artifact_bytes).unwrap();
        let quantized = QuantizedBinaryScannModel::from_artifact_view(&artifact_view).unwrap();
        let quantized_view = quantized.view(&artifact_bytes).unwrap();
        let range_backed = quantized_view.probe(&query, 4, 2, &mut scratch).unwrap();
        assert_eq!(range_backed, first);
        assert_eq!(quantized.fingerprint(), model.fingerprint());
        assert!(quantized.estimated_memory_bytes() < artifact_bytes.len());

        let codes = corpus(512);
        let labels: Vec<_> = (0..512).map(|doc_id| (doc_id, 0)).collect();
        let monolith = BinaryScannSegment::build(&model, &codes, &labels, &mut scratch).unwrap();

        let split = 193;
        let left_labels: Vec<_> = (0..split as u32).map(|doc_id| (doc_id, 0)).collect();
        let right_labels: Vec<_> = (0..(512 - split) as u32)
            .map(|doc_id| (doc_id, 0))
            .collect();
        let left =
            BinaryScannSegment::build(&model, &codes[..split * 2], &left_labels, &mut scratch)
                .unwrap();
        let right =
            BinaryScannSegment::build(&model, &codes[split * 2..], &right_labels, &mut scratch)
                .unwrap();
        let merged =
            BinaryScannSegment::merge_compatible(&model, &[(&left, 0), (&right, split as u32)])
                .unwrap();

        let expected = model
            .search_segments(
                &query,
                25,
                model.num_leaves() as usize,
                8,
                &[(&monolith, 0)],
                &mut scratch,
            )
            .unwrap();
        let split_hits = model
            .search_segments(
                &query,
                25,
                model.num_leaves() as usize,
                8,
                &[(&left, 0), (&right, split as u32)],
                &mut scratch,
            )
            .unwrap();
        let merged_hits = model
            .search_segments(
                &query,
                25,
                model.num_leaves() as usize,
                8,
                &[(&merged, 0)],
                &mut scratch,
            )
            .unwrap();
        assert_eq!(split_hits, expected);
        assert_eq!(merged_hits, expected);

        let mut brute_force: Vec<_> = codes
            .chunks_exact(2)
            .enumerate()
            .map(|(doc_id, code)| BinaryScannHit {
                doc_id: doc_id as u32,
                ordinal: 0,
                distance: hamming_distance(&query, code),
            })
            .collect();
        brute_force.sort_unstable();
        brute_force.truncate(25);
        assert_eq!(expected, brute_force);
    }

    #[test]
    fn selective_binary_spill_is_bounded_deterministic_and_deduplicated() {
        let model = two_leaf_model();
        let artifact = model.to_artifact(17, 100_000).unwrap();
        let codes = vec![0x00, 0x01, 0x03, 0x07, 0x0f, 0xff, 0xfe, 0xfc, 0xf8, 0xf0];
        let labels: Vec<_> = (0..codes.len() as u32).map(|doc_id| (doc_id, 0)).collect();
        let soar = SoarConfig::new().target_spill_fraction(0.30);
        let mut scratch = BinaryScannSearchScratch::default();

        let artifact_bytes = artifact.to_bytes().unwrap();
        let artifact_view = ScannTrainedArtifactView::parse(&artifact_bytes).unwrap();
        let quantized = QuantizedBinaryScannModel::from_artifact_view(&artifact_view).unwrap();
        let quantized = quantized.view(&artifact_bytes).unwrap();
        for code in &codes {
            let code = std::slice::from_ref(code);
            assert_eq!(
                model.spill_assignment(code, &mut scratch).unwrap(),
                quantized.spill_assignment(code, &mut scratch).unwrap(),
            );
        }

        let first =
            BinaryScannSegment::build_with_soar(&model, &codes, &labels, &soar, &mut scratch)
                .unwrap();
        let second =
            BinaryScannSegment::build_with_soar(&model, &codes, &labels, &soar, &mut scratch)
                .unwrap();
        assert_eq!(first.len(), codes.len());
        assert_eq!(first.stored_len(), codes.len() + 3);

        let first_payload = first
            .to_payload(&model, &artifact, codes.len() as u32)
            .unwrap();
        let second_payload = second
            .to_payload(&model, &artifact, codes.len() as u32)
            .unwrap();
        assert_eq!(first_payload, second_payload);
        let encoded = first_payload.to_bytes().unwrap();
        let decoded = ScannSegmentPayload::from_bytes(&encoded).unwrap();
        assert_eq!(decoded, first_payload);
        decoded.validate_against(&artifact).unwrap();

        let query = [0x0f];
        let hits = model
            .search_segments(&query, codes.len(), 2, 2, &[(&first, 0)], &mut scratch)
            .unwrap();
        assert_eq!(
            hits.len(),
            codes.len(),
            "secondary postings must not duplicate hits"
        );
        let mut expected: Vec<_> = codes
            .iter()
            .enumerate()
            .map(|(doc_id, code)| BinaryScannHit {
                doc_id: doc_id as u32,
                ordinal: 0,
                distance: hamming_distance(&query, std::slice::from_ref(code)),
            })
            .collect();
        expected.sort_unstable();
        assert_eq!(hits, expected);

        let split = 5;
        let local_labels: Vec<_> = (0..split as u32).map(|doc_id| (doc_id, 0)).collect();
        let left = BinaryScannSegment::build_with_soar(
            &model,
            &codes[..split],
            &local_labels,
            &soar,
            &mut scratch,
        )
        .unwrap();
        let right = BinaryScannSegment::build_with_soar(
            &model,
            &codes[split..],
            &local_labels,
            &soar,
            &mut scratch,
        )
        .unwrap();
        let fingerprint = model.fingerprint();
        let merged =
            BinaryScannSegment::merge_compatible(&model, &[(&left, 0), (&right, split as u32)])
                .unwrap();
        assert_eq!(model.fingerprint(), fingerprint, "merge must not retrain");
        assert_eq!(merged.len(), codes.len());
        assert_eq!(merged.stored_len(), codes.len() + 2);
        let merged_hits = model
            .search_segments(&query, codes.len(), 2, 2, &[(&merged, 0)], &mut scratch)
            .unwrap();
        assert_eq!(merged_hits, expected);

        // A boundary vector assigned primarily to leaf zero remains reachable
        // when a nearby query routes only to leaf one.
        let boundary_code = [0x0f];
        let boundary_label = [(42, 0)];
        let primary_only =
            BinaryScannSegment::build(&model, &boundary_code, &boundary_label, &mut scratch)
                .unwrap();
        let fully_spilled = BinaryScannSegment::build_with_soar(
            &model,
            &boundary_code,
            &boundary_label,
            &SoarConfig::full(),
            &mut scratch,
        )
        .unwrap();
        let nearby_query = [0x1f];
        assert!(
            model
                .search_segments(&nearby_query, 1, 1, 1, &[(&primary_only, 0)], &mut scratch)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            model
                .search_segments(&nearby_query, 1, 1, 1, &[(&fully_spilled, 0)], &mut scratch)
                .unwrap(),
            vec![BinaryScannHit {
                doc_id: 42,
                ordinal: 0,
                distance: 1,
            }],
        );
    }
}
