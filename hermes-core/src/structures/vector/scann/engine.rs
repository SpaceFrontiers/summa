//! Deterministic hierarchical k-means training and routing.
//!
//! This is the executable float routing core. The trained tree is global to an
//! index generation; immutable segments only store terminal leaf identifiers.

use std::ops::Range;

use rand::{Rng, SeedableRng};

use super::{
    AhCodebook, AhQuery, DEFAULT_ANISOTROPIC_THRESHOLD, MAX_SCANN_TREE_LEVELS, ScannEncoding,
    ScannFormatError, ScannResult, ScannRoutingLevel, ScannTrainedArtifact,
    ScannTrainedArtifactView,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoutedLeaf {
    pub leaf: u32,
    pub squared_distance: f32,
}

#[derive(Clone, Debug, Default)]
pub struct RoutingScratch {
    active: Vec<(usize, f32)>,
    next: Vec<(usize, f32)>,
}

/// Reusable storage for assigning and AH-encoding float vectors. Segment
/// builders keep one of these for their lifetime so encoding a row does not
/// allocate dimension-sized residuals or code buffers.
#[derive(Clone, Debug, Default)]
pub struct FloatEncodeScratch {
    routing: RoutingScratch,
    routed: Vec<RoutedLeaf>,
    residual: Vec<f32>,
    codes: Vec<u8>,
    ah: super::AhEncodeScratch,
}

/// Query routing intentionally explores a wider beam than the final probe
/// count when probes are small. Large probe requests widen this floor just
/// enough to keep the requested terminal leaves reachable.
const QUERY_INTERMEDIATE_ROUTING_BEAM: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct RoutingTraining {
    pub tree: FloatRoutingTree,
    /// Final training assignment in the reordered terminal-leaf namespace.
    pub assignments: Vec<u32>,
    /// Deterministic accounting for the bounded recursive trainer.
    pub stats: RoutingTrainingStats,
}

/// Work accounting for hierarchical routing training. `max_split_clusters`
/// is the scale invariant: it is bounded by local training fanout and does not
/// grow to the terminal leaf count.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RoutingTrainingStats {
    pub splits: usize,
    pub max_split_clusters: usize,
    pub max_depth: usize,
    pub distance_evaluations: u64,
    /// Membership recovery replaces a terminal `points * leaves` scan.
    pub assignment_distance_evaluations: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FloatRoutingTree {
    dimension: usize,
    levels: Vec<Vec<f32>>,
    child_offsets: Vec<Vec<u32>>,
}

/// Executable float ScaNN model shared by all immutable segments in one index
/// generation.
#[derive(Clone, Debug, PartialEq)]
pub struct FloatScannModel {
    pub routing: FloatRoutingTree,
    pub codebook: AhCodebook,
    anisotropic_threshold: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct QuantizedFloatRoutingLevel {
    centroid_count: usize,
    centroid_codes: Range<usize>,
    minimums: Vec<f32>,
    steps: Vec<f32>,
    child_offsets: Vec<u32>,
}

/// Small executable metadata for a persisted float ScaNN model. Quantized
/// centroid planes remain in the caller-owned artifact bytes; only per-level
/// scale vectors, child directories, and the AH codebook are resident.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedFloatScannModel {
    dimension: usize,
    num_leaves: usize,
    artifact_id: u64,
    artifact_len: usize,
    levels: Vec<QuantizedFloatRoutingLevel>,
    codebook: AhCodebook,
    anisotropic_threshold: f32,
}

/// Borrowed executable pairing of small model metadata with the mmap/file
/// bytes that own its quantized centroid planes.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedFloatScannModelView<'a> {
    model: &'a QuantizedFloatScannModel,
    artifact_bytes: &'a [u8],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedFloatVector {
    pub leaf: u32,
    /// One unpacked 4-bit code per AH block.
    pub codes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FloatScannQuery {
    routed_leaves: Vec<u32>,
    centroid_dots: Vec<f32>,
    ah: AhQuery,
}

impl FloatScannModel {
    /// Reconstruct the executable float model from a validated persisted
    /// generation. The anisotropic threshold is an implementation invariant,
    /// not a schema or artifact knob, so version 1 artifacts use the same
    /// hardcoded value as training.
    pub fn from_artifact(artifact: &ScannTrainedArtifact) -> ScannResult<Self> {
        artifact.validate()?;
        let dimensions_per_block = match artifact.config.encoding {
            ScannEncoding::AsymmetricHash {
                dimensions_per_block,
                bits_per_code: 4,
            } => usize::from(dimensions_per_block),
            _ => {
                return Err(ScannFormatError::new(
                    "float ScaNN model requires a 4-bit asymmetric-hash artifact",
                ));
            }
        };
        let codebook_artifact = artifact.ah_codebook.as_ref().ok_or_else(|| {
            ScannFormatError::new("float ScaNN artifact is missing its AH codebook")
        })?;
        if usize::from(codebook_artifact.dimensions_per_block) != dimensions_per_block {
            return Err(ScannFormatError::new(
                "ScaNN routing encoding and AH codebook block geometry differ",
            ));
        }
        let dimension = artifact.config.dimension as usize;
        Ok(Self {
            routing: FloatRoutingTree::from_quantized_levels(&artifact.levels, dimension)?,
            codebook: AhCodebook::from_artifact(dimension, codebook_artifact)?,
            anisotropic_threshold: DEFAULT_ANISOTROPIC_THRESHOLD,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train(
        data: &[f32],
        points: usize,
        dimension: usize,
        level_counts: &[u32],
        dimensions_per_block: usize,
        iterations: usize,
        seed: u64,
        anisotropic_threshold: f32,
    ) -> ScannResult<(Self, Vec<EncodedFloatVector>)> {
        let model = Self::train_model(
            data,
            points,
            dimension,
            level_counts,
            dimensions_per_block,
            iterations,
            seed,
            anisotropic_threshold,
        )?;
        let encoded = data
            .chunks_exact(dimension)
            .map(|vector| model.encode(vector))
            .collect::<ScannResult<Vec<_>>>()?;
        Ok((model, encoded))
    }

    /// Train only the index-global model. Production generation training uses
    /// this path because training-sample encodings are immediately discarded;
    /// immutable segment builders encode the actual corpus later.
    #[allow(clippy::too_many_arguments)]
    pub fn train_model(
        data: &[f32],
        points: usize,
        dimension: usize,
        level_counts: &[u32],
        dimensions_per_block: usize,
        iterations: usize,
        seed: u64,
        anisotropic_threshold: f32,
    ) -> ScannResult<Self> {
        if !anisotropic_threshold.is_finite() || !(0.0..1.0).contains(&anisotropic_threshold) {
            return Err(ScannFormatError::new(
                "ScaNN anisotropic threshold must be in [0, 1)",
            ));
        }
        let routing = train_routing_tree(data, points, dimension, level_counts, iterations, seed)?;
        let codebook = AhCodebook::train_from_assigned_vectors(
            data,
            &routing.assignments,
            routing.tree.leaf_centroids(),
            points,
            dimension,
            dimensions_per_block,
            iterations,
            seed.wrapping_add(0xd1b5_4a32_d192_ed03),
        )?;
        Ok(Self {
            routing: routing.tree,
            codebook,
            anisotropic_threshold,
        })
    }

    pub fn encode(&self, vector: &[f32]) -> ScannResult<EncodedFloatVector> {
        let mut scratch = FloatEncodeScratch::default();
        let (leaf, codes) = self.encode_with_scratch(vector, &mut scratch)?;
        Ok(EncodedFloatVector {
            leaf,
            codes: codes.to_vec(),
        })
    }

    pub fn encode_with_scratch<'a>(
        &self,
        vector: &[f32],
        scratch: &'a mut FloatEncodeScratch,
    ) -> ScannResult<(u32, &'a [u8])> {
        if vector.len() != self.routing.dimension {
            return Err(ScannFormatError::new(
                "ScaNN vector dimension does not match trained model",
            ));
        }
        let FloatEncodeScratch {
            routing,
            routed,
            residual,
            codes,
            ah,
        } = scratch;
        self.routing
            .route_with_scratch(vector, 1, routing, routed)?;
        let leaf = routed[0].leaf;
        let centroid = &self.routing.leaf_centroids()
            [leaf as usize * self.routing.dimension..(leaf as usize + 1) * self.routing.dimension];
        residual.resize(self.routing.dimension, 0.0);
        for ((value, &original), &center) in residual.iter_mut().zip(vector).zip(centroid) {
            *value = original - center;
        }
        codes.resize(self.codebook.blocks(), 0);
        self.codebook.encode_with_scratch(
            residual,
            vector,
            self.anisotropic_threshold,
            codes,
            ah,
        )?;
        Ok((leaf, codes))
    }

    pub fn prepare_query(&self, query: &[f32], probes: usize) -> ScannResult<FloatScannQuery> {
        let routed = self.routing.route(query, probes)?;
        let mut routed_leaves = Vec::with_capacity(routed.len());
        let mut centroid_dots = Vec::with_capacity(routed.len());
        for routed_leaf in routed {
            let centroid = &self.routing.leaf_centroids()[routed_leaf.leaf as usize
                * self.routing.dimension
                ..(routed_leaf.leaf as usize + 1) * self.routing.dimension];
            routed_leaves.push(routed_leaf.leaf);
            centroid_dots.push(crate::structures::simd::dot_product_f32(
                query,
                centroid,
                query.len(),
            ));
        }
        Ok(FloatScannQuery {
            routed_leaves,
            centroid_dots,
            ah: self.codebook.query_dot_product(query)?,
        })
    }

    pub fn anisotropic_threshold(&self) -> f32 {
        self.anisotropic_threshold
    }
}

impl QuantizedFloatScannModel {
    pub fn from_artifact_view(artifact: &ScannTrainedArtifactView<'_>) -> ScannResult<Self> {
        let dimensions_per_block = match artifact.config.encoding {
            ScannEncoding::AsymmetricHash {
                dimensions_per_block,
                bits_per_code: 4,
            } => usize::from(dimensions_per_block),
            _ => {
                return Err(ScannFormatError::new(
                    "quantized float ScaNN model requires a 4-bit AH artifact",
                ));
            }
        };
        let dimension = artifact.config.dimension as usize;
        let codebook_ref = artifact.ah_codebook().ok_or_else(|| {
            ScannFormatError::new("float ScaNN artifact is missing its AH codebook")
        })?;
        if usize::from(codebook_ref.dimensions_per_block) != dimensions_per_block {
            return Err(ScannFormatError::new(
                "ScaNN routing encoding and AH codebook block geometry differ",
            ));
        }
        let codebook = AhCodebook::from_artifact_ref(dimension, codebook_ref)?;
        let mut levels = Vec::with_capacity(artifact.level_count());
        for index in 0..artifact.level_count() {
            let level = artifact
                .level(index)
                .ok_or_else(|| ScannFormatError::new("ScaNN artifact routing level disappeared"))?;
            let centroid_codes = artifact.level_centroid_codes_range(index).ok_or_else(|| {
                ScannFormatError::new("ScaNN artifact centroid range disappeared")
            })?;
            levels.push(QuantizedFloatRoutingLevel {
                centroid_count: level.centroid_count as usize,
                centroid_codes,
                minimums: level.minimums().collect(),
                steps: level.steps().collect(),
                child_offsets: level.child_offsets().collect(),
            });
        }
        let model = Self {
            dimension,
            num_leaves: artifact.config.num_leaves as usize,
            artifact_id: artifact.artifact_id,
            artifact_len: artifact.bytes().len(),
            levels,
            codebook,
            anisotropic_threshold: DEFAULT_ANISOTROPIC_THRESHOLD,
        };
        model.validate_metadata()?;
        Ok(model)
    }

    /// Pair this metadata with the exact validated artifact mapping it was
    /// created from. The fingerprint slot check is O(1); full hashing happened
    /// once when `ScannTrainedArtifactView` was parsed.
    pub fn view<'a>(
        &'a self,
        artifact_bytes: &'a [u8],
    ) -> ScannResult<QuantizedFloatScannModelView<'a>> {
        let stored_id = artifact_bytes
            .get(12..20)
            .and_then(|bytes| <[u8; 8]>::try_from(bytes).ok())
            .map(u64::from_le_bytes);
        if artifact_bytes.len() != self.artifact_len || stored_id != Some(self.artifact_id) {
            return Err(ScannFormatError::new(
                "quantized float ScaNN model was paired with a different artifact mapping",
            ));
        }
        Ok(QuantizedFloatScannModelView {
            model: self,
            artifact_bytes,
        })
    }

    pub fn estimated_memory_bytes(&self) -> usize {
        self.levels
            .iter()
            .fold(self.codebook.estimated_memory_bytes(), |total, level| {
                total
                    .saturating_add(level.minimums.len() * std::mem::size_of::<f32>())
                    .saturating_add(level.steps.len() * std::mem::size_of::<f32>())
                    .saturating_add(level.child_offsets.len() * std::mem::size_of::<u32>())
            })
    }

    fn validate_metadata(&self) -> ScannResult<()> {
        if self.dimension == 0
            || self.levels.is_empty()
            || self.levels.len() > usize::from(MAX_SCANN_TREE_LEVELS)
            || self.levels.last().map(|level| level.centroid_count) != Some(self.num_leaves)
            || self.codebook.dimension() != self.dimension
        {
            return Err(ScannFormatError::new(
                "invalid quantized float ScaNN model metadata",
            ));
        }
        for (index, level) in self.levels.iter().enumerate() {
            if level.minimums.len() != self.dimension
                || level.steps.len() != self.dimension
                || level.centroid_codes.len() != level.centroid_count.saturating_mul(self.dimension)
                || level.centroid_codes.end > self.artifact_len
            {
                return Err(ScannFormatError::new(format!(
                    "invalid quantized float ScaNN routing level {index}",
                )));
            }
            if index + 1 == self.levels.len() {
                if !level.child_offsets.is_empty() {
                    return Err(ScannFormatError::new(
                        "quantized ScaNN leaf level must not have children",
                    ));
                }
            } else if level.child_offsets.len() != level.centroid_count + 1
                || level.child_offsets.first() != Some(&0)
                || level.child_offsets.last().copied()
                    != Some(self.levels[index + 1].centroid_count as u32)
            {
                return Err(ScannFormatError::new(format!(
                    "invalid quantized float ScaNN child directory at level {index}",
                )));
            }
        }
        Ok(())
    }
}

impl QuantizedFloatScannModelView<'_> {
    pub fn encode(&self, vector: &[f32]) -> ScannResult<EncodedFloatVector> {
        let mut scratch = FloatEncodeScratch::default();
        let (leaf, codes) = self.encode_with_scratch(vector, &mut scratch)?;
        Ok(EncodedFloatVector {
            leaf,
            codes: codes.to_vec(),
        })
    }

    pub fn encode_with_scratch<'a>(
        &self,
        vector: &[f32],
        scratch: &'a mut FloatEncodeScratch,
    ) -> ScannResult<(u32, &'a [u8])> {
        if vector.len() != self.model.dimension {
            return Err(ScannFormatError::new(
                "ScaNN vector dimension does not match trained model",
            ));
        }
        let FloatEncodeScratch {
            routing,
            routed,
            residual,
            codes: encoded,
            ah,
        } = scratch;
        self.route_with_scratch(vector, 1, routing, routed)?;
        let leaf = routed[0].leaf;
        let level = self
            .model
            .levels
            .last()
            .expect("validated non-empty levels");
        let codes = self.level_codes(level);
        let row = &codes
            [leaf as usize * self.model.dimension..(leaf as usize + 1) * self.model.dimension];
        residual.resize(self.model.dimension, 0.0);
        for (coordinate, (&value, residual)) in vector.iter().zip(residual.iter_mut()).enumerate() {
            *residual = value
                - (level.minimums[coordinate]
                    + level.steps[coordinate] * f32::from(row[coordinate]));
        }
        encoded.resize(self.model.codebook.blocks(), 0);
        self.model.codebook.encode_with_scratch(
            residual,
            vector,
            self.model.anisotropic_threshold,
            encoded,
            ah,
        )?;
        Ok((leaf, encoded))
    }

    pub fn prepare_query(&self, query: &[f32], probes: usize) -> ScannResult<FloatScannQuery> {
        let routed = self.route(query, probes)?;
        let level = self
            .model
            .levels
            .last()
            .expect("validated non-empty levels");
        let codes = self.level_codes(level);
        let mut routed_leaves = Vec::with_capacity(routed.len());
        let mut centroid_dots = Vec::with_capacity(routed.len());
        for routed_leaf in routed {
            let row = &codes[routed_leaf.leaf as usize * self.model.dimension
                ..(routed_leaf.leaf as usize + 1) * self.model.dimension];
            let dot = query
                .iter()
                .enumerate()
                .map(|(coordinate, &value)| {
                    value
                        * (level.minimums[coordinate]
                            + level.steps[coordinate] * f32::from(row[coordinate]))
                })
                .sum();
            routed_leaves.push(routed_leaf.leaf);
            centroid_dots.push(dot);
        }
        Ok(FloatScannQuery {
            routed_leaves,
            centroid_dots,
            ah: self.model.codebook.query_dot_product(query)?,
        })
    }

    pub fn anisotropic_threshold(&self) -> f32 {
        self.model.anisotropic_threshold
    }

    fn route(&self, query: &[f32], probes: usize) -> ScannResult<Vec<RoutedLeaf>> {
        let mut output = Vec::with_capacity(probes);
        self.route_with_scratch(query, probes, &mut RoutingScratch::default(), &mut output)?;
        Ok(output)
    }

    fn route_with_scratch(
        &self,
        query: &[f32],
        probes: usize,
        scratch: &mut RoutingScratch,
        output: &mut Vec<RoutedLeaf>,
    ) -> ScannResult<()> {
        if query.len() != self.model.dimension || query.iter().any(|value| !value.is_finite()) {
            return Err(ScannFormatError::new(
                "ScaNN routing query has the wrong dimension or non-finite values",
            ));
        }
        if probes == 0 {
            return Err(ScannFormatError::new(
                "ScaNN routing probes must be positive",
            ));
        }
        scratch.active.clear();
        self.score_range_into(
            &self.model.levels[0],
            0,
            self.model.levels[0].centroid_count,
            query,
            &mut scratch.active,
        );
        if self.model.levels.len() == 1 {
            keep_best(&mut scratch.active, probes);
        } else {
            sort_routing_candidates(&mut scratch.active);
            let initial_width = super::routing_prefix_for_child_coverage(
                &scratch.active,
                &self.model.levels[0].child_offsets,
                intermediate_routing_beam(probes),
                probes,
                |candidate| candidate.0,
            );
            scratch.active.truncate(initial_width);
        }
        for level_index in 1..self.model.levels.len() {
            let level = &self.model.levels[level_index];
            let offsets = &self.model.levels[level_index - 1].child_offsets;
            scratch.next.clear();
            for &(parent, _) in &scratch.active {
                self.score_range_into(
                    level,
                    offsets[parent] as usize,
                    offsets[parent + 1] as usize,
                    query,
                    &mut scratch.next,
                );
            }
            if level_index + 1 == self.model.levels.len() {
                keep_best(&mut scratch.next, probes);
            } else {
                sort_routing_candidates(&mut scratch.next);
                let width = super::routing_prefix_for_child_coverage(
                    &scratch.next,
                    &level.child_offsets,
                    intermediate_routing_beam(probes),
                    probes,
                    |candidate| candidate.0,
                );
                scratch.next.truncate(width);
            }
            std::mem::swap(&mut scratch.active, &mut scratch.next);
        }
        output.clear();
        output.extend(
            scratch
                .active
                .iter()
                .map(|&(leaf, squared_distance)| RoutedLeaf {
                    leaf: leaf as u32,
                    squared_distance,
                }),
        );
        Ok(())
    }

    fn score_range_into(
        &self,
        level: &QuantizedFloatRoutingLevel,
        start: usize,
        end: usize,
        query: &[f32],
        output: &mut Vec<(usize, f32)>,
    ) {
        let codes = self.level_codes(level);
        output.reserve(end.saturating_sub(start));
        output.extend((start..end).map(|centroid| {
            let row =
                &codes[centroid * self.model.dimension..(centroid + 1) * self.model.dimension];
            let distance = query
                .iter()
                .enumerate()
                .map(|(coordinate, &value)| {
                    let decoded = level.minimums[coordinate]
                        + level.steps[coordinate] * f32::from(row[coordinate]);
                    let difference = value - decoded;
                    difference * difference
                })
                .sum();
            (centroid, distance)
        }));
    }

    fn level_codes(&self, level: &QuantizedFloatRoutingLevel) -> &[u8] {
        &self.artifact_bytes[level.centroid_codes.clone()]
    }
}

impl FloatScannQuery {
    pub fn routed_leaves(&self) -> &[u32] {
        &self.routed_leaves
    }

    /// Centroid contribution for one routed leaf. Segment scanners use this
    /// alongside the shared AH lookup table without allocating row wrappers.
    pub fn centroid_dot(&self, leaf: u32) -> Option<f32> {
        self.routed_leaves
            .iter()
            .position(|&candidate| candidate == leaf)
            .map(|position| self.centroid_dots[position])
    }

    /// Borrow the query-specific AH lookup table for packed/FastScan rows.
    pub fn ah_query(&self) -> &AhQuery {
        &self.ah
    }

    /// Returns `None` when the row's leaf was not selected by the query beam.
    pub fn score(&self, vector: &EncodedFloatVector) -> ScannResult<Option<f32>> {
        let Some(position) = self
            .routed_leaves
            .iter()
            .position(|&leaf| leaf == vector.leaf)
        else {
            return Ok(None);
        };
        self.ah
            .score_unpacked(&vector.codes, self.centroid_dots[position])
            .map(Some)
    }
}

impl FloatRoutingTree {
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn level_counts(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.levels.iter().map(|level| level.len() / self.dimension)
    }

    pub fn leaf_centroids(&self) -> &[f32] {
        self.levels.last().map_or(&[], Vec::as_slice)
    }

    pub fn levels(&self) -> &[Vec<f32>] {
        &self.levels
    }

    pub fn child_offsets(&self) -> &[Vec<u32>] {
        &self.child_offsets
    }

    /// Route to the closest terminal partitions with a bounded beam.
    pub fn route(&self, query: &[f32], probes: usize) -> ScannResult<Vec<RoutedLeaf>> {
        let mut output = Vec::with_capacity(probes);
        self.route_with_scratch(query, probes, &mut RoutingScratch::default(), &mut output)?;
        Ok(output)
    }

    /// Allocation-reusing serving-path form of [`Self::route`].
    pub fn route_with_scratch(
        &self,
        query: &[f32],
        probes: usize,
        scratch: &mut RoutingScratch,
        output: &mut Vec<RoutedLeaf>,
    ) -> ScannResult<()> {
        if query.len() != self.dimension || query.iter().any(|value| !value.is_finite()) {
            return Err(ScannFormatError::new(
                "ScaNN routing query has the wrong dimension or non-finite values",
            ));
        }
        if probes == 0 {
            return Err(ScannFormatError::new(
                "ScaNN routing probes must be positive",
            ));
        }

        scratch.active.clear();
        score_range_into(
            &self.levels[0],
            self.dimension,
            0,
            self.level_counts().next().unwrap(),
            query,
            &mut scratch.active,
        );
        if self.levels.len() == 1 {
            keep_best(&mut scratch.active, probes);
        } else {
            sort_routing_candidates(&mut scratch.active);
            let initial_width = super::routing_prefix_for_child_coverage(
                &scratch.active,
                &self.child_offsets[0],
                intermediate_routing_beam(probes),
                probes,
                |candidate| candidate.0,
            );
            scratch.active.truncate(initial_width);
        }
        for level in 1..self.levels.len() {
            let offsets = &self.child_offsets[level - 1];
            scratch.next.clear();
            for &(parent, _) in &scratch.active {
                let start = offsets[parent] as usize;
                let end = offsets[parent + 1] as usize;
                score_range_into(
                    &self.levels[level],
                    self.dimension,
                    start,
                    end,
                    query,
                    &mut scratch.next,
                );
            }
            if level + 1 == self.levels.len() {
                keep_best(&mut scratch.next, probes);
            } else {
                sort_routing_candidates(&mut scratch.next);
                let width = super::routing_prefix_for_child_coverage(
                    &scratch.next,
                    &self.child_offsets[level],
                    intermediate_routing_beam(probes),
                    probes,
                    |candidate| candidate.0,
                );
                scratch.next.truncate(width);
            }
            std::mem::swap(&mut scratch.active, &mut scratch.next);
        }
        output.clear();
        output.extend(
            scratch
                .active
                .iter()
                .map(|&(leaf, squared_distance)| RoutedLeaf {
                    leaf: leaf as u32,
                    squared_distance,
                }),
        );
        Ok(())
    }

    /// Quantize every level independently to the persisted u8 centroid plane.
    pub fn to_quantized_levels(&self) -> Vec<ScannRoutingLevel> {
        self.levels
            .iter()
            .enumerate()
            .map(|(level_index, centroids)| {
                let count = centroids.len() / self.dimension;
                let mut minimums = vec![f32::INFINITY; self.dimension];
                let mut maximums = vec![f32::NEG_INFINITY; self.dimension];
                for centroid in centroids.chunks_exact(self.dimension) {
                    for coordinate in 0..self.dimension {
                        minimums[coordinate] = minimums[coordinate].min(centroid[coordinate]);
                        maximums[coordinate] = maximums[coordinate].max(centroid[coordinate]);
                    }
                }
                let steps: Vec<f32> = minimums
                    .iter()
                    .zip(&maximums)
                    .map(|(&minimum, &maximum)| {
                        let range = maximum - minimum;
                        if range.is_finite() && range > 0.0 {
                            range / 255.0
                        } else {
                            1.0
                        }
                    })
                    .collect();
                let centroid_codes = centroids
                    .chunks_exact(self.dimension)
                    .flat_map(|centroid| {
                        centroid.iter().enumerate().map(|(coordinate, &value)| {
                            ((value - minimums[coordinate]) / steps[coordinate])
                                .round()
                                .clamp(0.0, 255.0) as u8
                        })
                    })
                    .collect();
                ScannRoutingLevel {
                    centroid_count: count as u32,
                    centroid_codes,
                    minimums,
                    steps,
                    child_offsets: self
                        .child_offsets
                        .get(level_index)
                        .cloned()
                        .unwrap_or_default(),
                }
            })
            .collect()
    }

    pub fn from_quantized_levels(
        levels: &[ScannRoutingLevel],
        dimension: usize,
    ) -> ScannResult<Self> {
        if dimension == 0 || levels.is_empty() || levels.len() > usize::from(MAX_SCANN_TREE_LEVELS)
        {
            return Err(ScannFormatError::new(
                "invalid ScaNN quantized routing shape",
            ));
        }
        let mut decoded = Vec::with_capacity(levels.len());
        let mut child_offsets = Vec::with_capacity(levels.len().saturating_sub(1));
        for (index, level) in levels.iter().enumerate() {
            if level.minimums.len() != dimension
                || level.steps.len() != dimension
                || level.centroid_codes.len() != level.centroid_count as usize * dimension
                || level
                    .minimums
                    .iter()
                    .chain(&level.steps)
                    .any(|value| !value.is_finite())
            {
                return Err(ScannFormatError::new(format!(
                    "invalid ScaNN quantized routing level {index}"
                )));
            }
            let centroids = level
                .centroid_codes
                .chunks_exact(dimension)
                .flat_map(|centroid| {
                    centroid.iter().enumerate().map(|(coordinate, &code)| {
                        level.minimums[coordinate] + level.steps[coordinate] * f32::from(code)
                    })
                })
                .collect();
            decoded.push(centroids);
            if index + 1 < levels.len() {
                if level.child_offsets.len() != level.centroid_count as usize + 1
                    || level.child_offsets.first() != Some(&0)
                    || level.child_offsets.last() != Some(&levels[index + 1].centroid_count)
                    || level.child_offsets.windows(2).any(|pair| pair[0] > pair[1])
                {
                    return Err(ScannFormatError::new(format!(
                        "invalid ScaNN child offsets at routing level {index}"
                    )));
                }
                child_offsets.push(level.child_offsets.clone());
            } else if !level.child_offsets.is_empty() {
                return Err(ScannFormatError::new(
                    "terminal ScaNN routing level must not have children",
                ));
            }
        }
        Ok(Self {
            dimension,
            levels: decoded,
            child_offsets,
        })
    }
}

/// Maximum number of centroids considered by one Lloyd assignment. Large
/// requested partitions are produced recursively, so training work scales
/// with this local fanout rather than `points * terminal_leaves`.
const MAX_LOCAL_KMEANS_BRANCHES: usize = 64;

/// Train a nested routing tree with bounded recursive partitioning. Terminal
/// centroids are generated by local k-means splits, then upper persisted levels
/// are fitted bottom-up with the same bounded algorithm. Carrying each upper
/// permutation through all descendants preserves contiguous child ranges.
pub fn train_routing_tree(
    data: &[f32],
    points: usize,
    dimension: usize,
    level_counts: &[u32],
    iterations: usize,
    seed: u64,
) -> ScannResult<RoutingTraining> {
    if points == 0
        || dimension == 0
        || data.len() != points.saturating_mul(dimension)
        || data.iter().any(|value| !value.is_finite())
        || level_counts.is_empty()
        || level_counts.len() > usize::from(MAX_SCANN_TREE_LEVELS)
        || level_counts.contains(&0)
        || level_counts.windows(2).any(|pair| pair[0] > pair[1])
        || level_counts.last().copied().unwrap_or_default() as usize > points
    {
        return Err(ScannFormatError::new(
            "invalid ScaNN routing training data or geometry",
        ));
    }

    let leaf_count = *level_counts.last().unwrap() as usize;
    let mut stats = RoutingTrainingStats::default();
    let PartitionTraining {
        centroids: leaf_centroids,
        group_sizes: leaf_group_sizes,
        point_order: leaf_point_order,
    } = train_partition(
        data, points, dimension, leaf_count, iterations, seed, 0, &mut stats,
    )?;
    let mut assignments = vec![u32::MAX; points];
    let mut cursor = 0usize;
    for (leaf, group_size) in leaf_group_sizes.into_iter().enumerate() {
        let end = cursor + group_size;
        for &point in &leaf_point_order[cursor..end] {
            assignments[point] = leaf as u32;
        }
        cursor = end;
    }
    debug_assert_eq!(cursor, points);
    debug_assert!(!assignments.contains(&u32::MAX));
    let mut leaf_current_to_original: Vec<usize> = (0..leaf_count).collect();
    let mut levels = vec![leaf_centroids];
    let mut child_offsets = Vec::with_capacity(level_counts.len().saturating_sub(1));

    for (round, &parent_count) in level_counts[..level_counts.len() - 1]
        .iter()
        .rev()
        .enumerate()
    {
        let children = levels[0].len() / dimension;
        let partition = train_partition(
            &levels[0],
            children,
            dimension,
            parent_count as usize,
            iterations,
            seed.wrapping_add(0x9e37_79b9_u64.wrapping_mul(round as u64 + 1)),
            round + 1,
            &mut stats,
        )?;
        let leaf_new_to_old = reorder_descendants(
            &mut levels,
            &mut child_offsets,
            &partition.point_order,
            dimension,
        )?;
        leaf_current_to_original = leaf_new_to_old
            .into_iter()
            .map(|old| leaf_current_to_original[old])
            .collect();
        child_offsets.insert(0, group_offsets(&partition.group_sizes)?);
        levels.insert(0, partition.centroids);
    }

    let tree = FloatRoutingTree {
        dimension,
        levels,
        child_offsets,
    };
    let mut original_to_current = vec![0u32; leaf_count];
    for (current, original) in leaf_current_to_original.into_iter().enumerate() {
        original_to_current[original] = current as u32;
    }
    for assignment in &mut assignments {
        *assignment = original_to_current[*assignment as usize];
    }
    Ok(RoutingTraining {
        tree,
        assignments,
        stats,
    })
}

struct PartitionTraining {
    centroids: Vec<f32>,
    group_sizes: Vec<usize>,
    point_order: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
fn train_partition(
    data: &[f32],
    points: usize,
    dimension: usize,
    clusters: usize,
    iterations: usize,
    seed: u64,
    depth: usize,
    stats: &mut RoutingTrainingStats,
) -> ScannResult<PartitionTraining> {
    if points == 0
        || dimension == 0
        || data.len() != points.saturating_mul(dimension)
        || clusters == 0
        || clusters > points
    {
        return Err(ScannFormatError::new(
            "invalid ScaNN recursive partition shape",
        ));
    }
    let mut point_order: Vec<usize> = (0..points).collect();
    let mut centroids = Vec::with_capacity(clusters.saturating_mul(dimension));
    let mut group_sizes = Vec::with_capacity(clusters);
    train_partition_node(
        data,
        &mut point_order,
        dimension,
        clusters,
        iterations,
        seed,
        depth,
        &mut centroids,
        &mut group_sizes,
        stats,
    );
    if centroids.len() != clusters.saturating_mul(dimension)
        || group_sizes.len() != clusters
        || group_sizes.iter().sum::<usize>() != points
    {
        return Err(ScannFormatError::new(
            "ScaNN recursive partition produced the wrong shape",
        ));
    }
    Ok(PartitionTraining {
        centroids,
        group_sizes,
        point_order,
    })
}

#[allow(clippy::too_many_arguments)]
fn train_partition_node(
    data: &[f32],
    point_ids: &mut [usize],
    dimension: usize,
    clusters: usize,
    iterations: usize,
    seed: u64,
    depth: usize,
    output: &mut Vec<f32>,
    group_sizes: &mut Vec<usize>,
    stats: &mut RoutingTrainingStats,
) {
    let points = point_ids.len();
    stats.max_depth = stats.max_depth.max(depth);
    if clusters == 1 {
        append_mean(data, point_ids, dimension, output);
        group_sizes.push(points);
        return;
    }
    if clusters == points {
        for &point_id in point_ids.iter() {
            output.extend_from_slice(&data[point_id * dimension..(point_id + 1) * dimension]);
        }
        group_sizes.resize(group_sizes.len() + points, 1);
        return;
    }

    let branches = training_branch_factor(clusters).min(points);
    let model = train_local_kmeans(
        data, point_ids, dimension, branches, iterations, seed, stats,
    );
    stats.splits = stats.splits.saturating_add(1);
    stats.max_split_clusters = stats.max_split_clusters.max(branches);
    let sizes: Vec<usize> = model
        .member_offsets
        .windows(2)
        .map(|range| range[1] - range[0])
        .collect();
    let allocations = apportion_clusters(&sizes, clusters);
    reorder_point_ids(point_ids, &model.assignments, &model.member_offsets);
    for (branch, &allocation) in allocations.iter().enumerate() {
        let start = model.member_offsets[branch];
        let end = model.member_offsets[branch + 1];
        train_partition_node(
            data,
            &mut point_ids[start..end],
            dimension,
            allocation,
            iterations,
            mix_seed(seed, depth, branch),
            depth + 1,
            output,
            group_sizes,
            stats,
        );
    }
}

struct LocalKMeans {
    assignments: Vec<usize>,
    member_offsets: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
fn train_local_kmeans(
    data: &[f32],
    point_ids: &[usize],
    dimension: usize,
    clusters: usize,
    iterations: usize,
    seed: u64,
    stats: &mut RoutingTrainingStats,
) -> LocalKMeans {
    debug_assert!(clusters > 1 && clusters < point_ids.len());
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let points = point_ids.len();
    let mut selected = std::collections::BTreeSet::new();
    let mut selected_rows = Vec::with_capacity(clusters);
    for upper in points - clusters..points {
        let candidate = rng.random_range(0..=upper);
        let row = if selected.insert(candidate) {
            candidate
        } else {
            selected.insert(upper);
            upper
        };
        selected_rows.push(row);
    }
    let mut centroids: Vec<f32> = selected_rows
        .into_iter()
        .flat_map(|row| {
            let point_id = point_ids[row];
            data[point_id * dimension..(point_id + 1) * dimension]
                .iter()
                .copied()
        })
        .collect();
    let mut assignments = vec![usize::MAX; points];
    for _ in 0..iterations.max(1) {
        let mut distances = vec![0.0f32; points];
        for (row, &point_id) in point_ids.iter().enumerate() {
            let point = &data[point_id * dimension..(point_id + 1) * dimension];
            let (cluster, distance) = nearest_centroid(&centroids, dimension, point);
            assignments[row] = cluster;
            distances[row] = distance;
        }
        stats.distance_evaluations = stats
            .distance_evaluations
            .saturating_add(u64::try_from(points.saturating_mul(clusters)).unwrap_or(u64::MAX));
        ensure_non_empty(&mut assignments, &distances, clusters);
        let mut sums = vec![0.0f32; clusters * dimension];
        let mut counts = vec![0usize; clusters];
        for (&point_id, &cluster) in point_ids.iter().zip(&assignments) {
            let point = &data[point_id * dimension..(point_id + 1) * dimension];
            counts[cluster] += 1;
            for coordinate in 0..dimension {
                sums[cluster * dimension + coordinate] += point[coordinate];
            }
        }
        for cluster in 0..clusters {
            let inverse = (counts[cluster] as f32).recip();
            for coordinate in 0..dimension {
                sums[cluster * dimension + coordinate] *= inverse;
            }
        }
        if sums == centroids {
            break;
        }
        centroids = sums;
    }
    let mut distances = vec![0.0f32; points];
    for (row, &point_id) in point_ids.iter().enumerate() {
        let point = &data[point_id * dimension..(point_id + 1) * dimension];
        let (cluster, distance) = nearest_centroid(&centroids, dimension, point);
        assignments[row] = cluster;
        distances[row] = distance;
    }
    stats.distance_evaluations = stats
        .distance_evaluations
        .saturating_add(u64::try_from(points.saturating_mul(clusters)).unwrap_or(u64::MAX));
    ensure_non_empty(&mut assignments, &distances, clusters);
    let mut member_offsets = vec![0usize; clusters + 1];
    for &cluster in &assignments {
        member_offsets[cluster + 1] += 1;
    }
    for cluster in 0..clusters {
        member_offsets[cluster + 1] += member_offsets[cluster];
    }
    LocalKMeans {
        assignments,
        member_offsets,
    }
}

fn training_branch_factor(clusters: usize) -> usize {
    if clusters <= MAX_LOCAL_KMEANS_BRANCHES {
        clusters
    } else {
        ((clusters as f64).sqrt().ceil() as usize).clamp(2, MAX_LOCAL_KMEANS_BRANCHES)
    }
}

fn append_mean(data: &[f32], point_ids: &[usize], dimension: usize, output: &mut Vec<f32>) {
    let start = output.len();
    output.resize(start + dimension, 0.0);
    for &point_id in point_ids {
        let point = &data[point_id * dimension..(point_id + 1) * dimension];
        for (sum, &value) in output[start..].iter_mut().zip(point) {
            *sum += value;
        }
    }
    let inverse = (point_ids.len() as f32).recip();
    for value in &mut output[start..] {
        *value *= inverse;
    }
}

fn apportion_clusters(sizes: &[usize], total_clusters: usize) -> Vec<usize> {
    debug_assert!(!sizes.is_empty());
    debug_assert!(sizes.iter().all(|&size| size > 0));
    debug_assert!(total_clusters >= sizes.len());
    debug_assert!(total_clusters <= sizes.iter().sum());
    let total_points: usize = sizes.iter().sum();
    let mut allocations = vec![1usize; sizes.len()];
    let mut assigned = sizes.len();
    for (allocation, &size) in allocations.iter_mut().zip(sizes) {
        let target = (total_clusters.saturating_mul(size) / total_points)
            .max(1)
            .min(size);
        assigned += target - 1;
        *allocation = target;
    }
    while assigned < total_clusters {
        let next = (0..sizes.len())
            .filter(|&index| allocations[index] < sizes[index])
            .max_by(|&left, &right| {
                let left_deficit = (total_clusters as i128) * (sizes[left] as i128)
                    - (allocations[left] as i128) * (total_points as i128);
                let right_deficit = (total_clusters as i128) * (sizes[right] as i128)
                    - (allocations[right] as i128) * (total_points as i128);
                left_deficit
                    .cmp(&right_deficit)
                    .then_with(|| right.cmp(&left))
            })
            .expect("remaining points provide centroid capacity");
        allocations[next] += 1;
        assigned += 1;
    }
    while assigned > total_clusters {
        let next = (0..sizes.len())
            .filter(|&index| allocations[index] > 1)
            .max_by(|&left, &right| {
                let left_excess = (allocations[left] as i128) * (total_points as i128)
                    - (total_clusters as i128) * (sizes[left] as i128);
                let right_excess = (allocations[right] as i128) * (total_points as i128)
                    - (total_clusters as i128) * (sizes[right] as i128);
                left_excess
                    .cmp(&right_excess)
                    .then_with(|| right.cmp(&left))
            })
            .expect("at least one branch can release a centroid");
        allocations[next] -= 1;
        assigned -= 1;
    }
    allocations
}

fn reorder_point_ids(point_ids: &mut [usize], assignments: &[usize], offsets: &[usize]) {
    let original = point_ids.to_vec();
    let mut cursors = offsets[..offsets.len() - 1].to_vec();
    for (&point_id, &cluster) in original.iter().zip(assignments) {
        point_ids[cursors[cluster]] = point_id;
        cursors[cluster] += 1;
    }
}

fn group_offsets(group_sizes: &[usize]) -> ScannResult<Vec<u32>> {
    let mut offsets = Vec::with_capacity(group_sizes.len() + 1);
    offsets.push(0);
    let mut cursor = 0usize;
    for &size in group_sizes {
        cursor = cursor
            .checked_add(size)
            .ok_or_else(|| ScannFormatError::new("ScaNN child count overflows usize"))?;
        offsets.push(
            u32::try_from(cursor)
                .map_err(|_| ScannFormatError::new("ScaNN child count exceeds u32"))?,
        );
    }
    Ok(offsets)
}

fn reorder_descendants(
    levels: &mut [Vec<f32>],
    child_offsets: &mut [Vec<u32>],
    top_order: &[usize],
    dimension: usize,
) -> ScannResult<Vec<usize>> {
    if levels.len() != child_offsets.len() + 1
        || levels.first().map_or(0, |level| level.len() / dimension) != top_order.len()
    {
        return Err(ScannFormatError::new(
            "ScaNN bottom-up subtree shape mismatch",
        ));
    }
    let mut parent_order = top_order.to_vec();
    let mut permutation = Vec::new();
    for depth in 0..child_offsets.len() {
        let old_offsets = &child_offsets[depth];
        if old_offsets.len() != parent_order.len() + 1 {
            return Err(ScannFormatError::new(
                "ScaNN bottom-up child directory mismatch",
            ));
        }
        let mut next_order = Vec::with_capacity(levels[depth + 1].len() / dimension);
        let mut next_offsets = Vec::with_capacity(parent_order.len() + 1);
        next_offsets.push(0);
        for &old_parent in &parent_order {
            let start = old_offsets[old_parent] as usize;
            let end = old_offsets[old_parent + 1] as usize;
            if start > end || end > levels[depth + 1].len() / dimension {
                return Err(ScannFormatError::new(
                    "ScaNN bottom-up child range is invalid",
                ));
            }
            next_order.extend(start..end);
            next_offsets.push(
                u32::try_from(next_order.len())
                    .map_err(|_| ScannFormatError::new("ScaNN descendant count exceeds u32"))?,
            );
        }
        if next_order.len() != levels[depth + 1].len() / dimension {
            return Err(ScannFormatError::new(
                "ScaNN bottom-up child permutation is incomplete",
            ));
        }
        reorder_rows(
            &mut levels[depth + 1],
            dimension,
            &next_order,
            &mut permutation,
        );
        child_offsets[depth] = next_offsets;
        parent_order = next_order;
    }
    reorder_rows(&mut levels[0], dimension, top_order, &mut permutation);
    Ok(parent_order)
}

fn reorder_rows(
    data: &mut [f32],
    dimension: usize,
    new_to_old: &[usize],
    permutation: &mut Vec<usize>,
) {
    debug_assert_eq!(data.len(), new_to_old.len() * dimension);
    permutation.clear();
    permutation.resize(new_to_old.len(), usize::MAX);
    for (new, &old) in new_to_old.iter().enumerate() {
        permutation[old] = new;
    }
    debug_assert!(!permutation.contains(&usize::MAX));
    for index in 0..new_to_old.len() {
        while permutation[index] != index {
            let other = permutation[index];
            for coordinate in 0..dimension {
                data.swap(
                    index * dimension + coordinate,
                    other * dimension + coordinate,
                );
            }
            permutation.swap(index, other);
        }
    }
}

fn mix_seed(seed: u64, depth: usize, branch: usize) -> u64 {
    seed ^ (depth as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (branch as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9)
}

fn ensure_non_empty(assignments: &mut [usize], distances: &[f32], clusters: usize) {
    let mut counts = vec![0usize; clusters];
    for &cluster in assignments.iter() {
        counts[cluster] += 1;
    }
    for empty in 0..clusters {
        if counts[empty] != 0 {
            continue;
        }
        let donor = (0..assignments.len())
            .filter(|&row| counts[assignments[row]] > 1)
            .max_by(|&left, &right| {
                distances[left]
                    .total_cmp(&distances[right])
                    .then_with(|| right.cmp(&left))
            })
            .expect("clusters do not exceed points");
        counts[assignments[donor]] -= 1;
        assignments[donor] = empty;
        counts[empty] = 1;
    }
}

fn nearest_centroid(centroids: &[f32], dimension: usize, point: &[f32]) -> (usize, f32) {
    centroids
        .chunks_exact(dimension)
        .enumerate()
        .map(|(index, centroid)| (index, squared_l2(point, centroid)))
        .min_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        })
        .unwrap()
}

fn score_range_into(
    centroids: &[f32],
    dimension: usize,
    start: usize,
    end: usize,
    query: &[f32],
    output: &mut Vec<(usize, f32)>,
) {
    output.reserve(end.saturating_sub(start));
    output.extend((start..end).map(|index| {
        (
            index,
            squared_l2(
                &centroids[index * dimension..(index + 1) * dimension],
                query,
            ),
        )
    }));
}

fn keep_best(values: &mut Vec<(usize, f32)>, count: usize) {
    let compare = |left: &(usize, f32), right: &(usize, f32)| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    };
    let keep = count.min(values.len());
    if keep == 0 {
        values.clear();
        return;
    }
    if keep < values.len() {
        values.select_nth_unstable_by(keep, compare);
        values.truncate(keep);
    }
    // Stable output order is part of deterministic routing, while the large
    // rejected tail no longer pays O(n log n) sorting work.
    values.sort_unstable_by(compare);
}

fn sort_routing_candidates(values: &mut [(usize, f32)]) {
    values.sort_unstable_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
}

#[inline]
fn intermediate_routing_beam(probes: usize) -> usize {
    if probes == 1 {
        // Corpus assignment follows exactly one path through the tree.
        1
    } else {
        QUERY_INTERMEDIATE_ROUTING_BEAM
    }
}

#[inline]
fn squared_l2(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| {
            let difference = a - b;
            difference * difference
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clustered_points() -> Vec<f32> {
        let mut points = Vec::new();
        for cluster in 0..8 {
            for row in 0..16 {
                points.push(cluster as f32 * 10.0 + row as f32 * 0.01);
                points.push((cluster % 3) as f32 * 5.0 - row as f32 * 0.005);
            }
        }
        points
    }

    #[test]
    fn hierarchical_training_is_deterministic_and_nested() {
        let data = clustered_points();
        let first = train_routing_tree(&data, 128, 2, &[2, 8], 8, 17).unwrap();
        let second = train_routing_tree(&data, 128, 2, &[2, 8], 8, 17).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.tree.level_counts().collect::<Vec<_>>(), [2, 8]);
        assert_eq!(first.tree.child_offsets()[0].first(), Some(&0));
        assert_eq!(first.tree.child_offsets()[0].last(), Some(&8));
    }

    #[test]
    fn query_beam_is_recall_oriented_but_bounded() {
        assert_eq!(intermediate_routing_beam(1), 1);
        assert_eq!(
            intermediate_routing_beam(2),
            QUERY_INTERMEDIATE_ROUTING_BEAM
        );
        assert_eq!(
            intermediate_routing_beam(usize::MAX),
            QUERY_INTERMEDIATE_ROUTING_BEAM
        );
    }

    #[test]
    fn full_probe_reaches_every_leaf_past_sixty_four_root_parents() {
        let root_count = 65usize;
        let leaf_count = root_count * root_count;
        let mut offsets = Vec::with_capacity(root_count + 1);
        for parent in 0..=root_count {
            offsets.push((parent * root_count) as u32);
        }
        let tree = FloatRoutingTree {
            dimension: 1,
            levels: vec![vec![0.0; root_count], vec![0.0; leaf_count]],
            child_offsets: vec![offsets],
        };
        let routed = tree.route(&[0.0], leaf_count).unwrap();
        assert_eq!(routed.len(), leaf_count);
        assert_eq!(
            routed.iter().map(|leaf| leaf.leaf).collect::<Vec<_>>(),
            (0..leaf_count as u32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn model_only_training_matches_compatibility_training() {
        let data = clustered_points();
        let model = FloatScannModel::train_model(
            &data,
            128,
            2,
            &[2, 8],
            1,
            4,
            0x5ca1,
            DEFAULT_ANISOTROPIC_THRESHOLD,
        )
        .unwrap();
        let (compatibility_model, encoded) = FloatScannModel::train(
            &data,
            128,
            2,
            &[2, 8],
            1,
            4,
            0x5ca1,
            DEFAULT_ANISOTROPIC_THRESHOLD,
        )
        .unwrap();

        assert_eq!(model, compatibility_model);
        assert_eq!(encoded.len(), 128);
    }

    #[test]
    fn float_encode_scratch_reuses_dimension_and_code_buffers() {
        let data = clustered_points();
        let model = FloatScannModel::train_model(
            &data,
            128,
            2,
            &[2, 8],
            1,
            4,
            0x5ca1,
            DEFAULT_ANISOTROPIC_THRESHOLD,
        )
        .unwrap();
        let mut scratch = FloatEncodeScratch::default();

        model.encode_with_scratch(&data[..2], &mut scratch).unwrap();
        let residual_allocation = scratch.residual.as_ptr();
        let code_allocation = scratch.codes.as_ptr();
        let residual_capacity = scratch.residual.capacity();
        let code_capacity = scratch.codes.capacity();

        let (leaf, codes) = model
            .encode_with_scratch(&data[2..4], &mut scratch)
            .unwrap();
        assert!(leaf < 8);
        assert_eq!(codes.len(), model.codebook.blocks());
        assert_eq!(scratch.residual.as_ptr(), residual_allocation);
        assert_eq!(scratch.codes.as_ptr(), code_allocation);
        assert_eq!(scratch.residual.capacity(), residual_capacity);
        assert_eq!(scratch.codes.capacity(), code_capacity);
    }

    #[test]
    fn recursive_training_work_is_bounded_by_local_fanout() {
        let points = 4_096usize;
        let dimension = 4usize;
        let data: Vec<f32> = (0..points * dimension)
            .map(|index| {
                let row = index / dimension;
                let coordinate = index % dimension;
                (((row * 73 + coordinate * 151 + row * coordinate * 19) % 997) as f32 / 498.5) - 1.0
            })
            .collect();
        let trained =
            train_routing_tree(&data, points, dimension, &[16, 1_024], 2, 0x5ca1).unwrap();

        assert_eq!(trained.tree.level_counts().collect::<Vec<_>>(), [16, 1_024]);
        assert!(trained.stats.splits > 1);
        assert!(
            trained.stats.max_split_clusters <= MAX_LOCAL_KMEANS_BRANCHES,
            "local split widened to {} clusters",
            trained.stats.max_split_clusters,
        );
        assert!(trained.stats.max_split_clusters < 1_024);
        assert_eq!(trained.stats.assignment_distance_evaluations, 0);
        assert!(trained.assignments.iter().all(|&leaf| leaf < 1_024));
        let one_flat_assignment = (points * 1_024) as u64;
        assert!(
            trained.stats.distance_evaluations < one_flat_assignment,
            "recursive trainer performed {} distance evaluations vs {one_flat_assignment} for one flat leaf pass",
            trained.stats.distance_evaluations,
        );
    }

    #[test]
    fn quantized_tree_roundtrip_preserves_routing_recall() {
        let data = clustered_points();
        let trained = train_routing_tree(&data, 128, 2, &[2, 8], 8, 91).unwrap();
        let restored =
            FloatRoutingTree::from_quantized_levels(&trained.tree.to_quantized_levels(), 2)
                .unwrap();
        let mut recalled = 0usize;
        for point in data.chunks_exact(2) {
            let exact = nearest_centroid(trained.tree.leaf_centroids(), 2, point).0 as u32;
            if restored
                .route(point, 4)
                .unwrap()
                .iter()
                .any(|leaf| leaf.leaf == exact)
            {
                recalled += 1;
            }
        }
        assert!(recalled as f32 / 128.0 >= 0.98);
    }

    #[test]
    fn executable_model_reopens_from_the_persisted_generation() {
        let data = clustered_points();
        let (model, _) = FloatScannModel::train(
            &data,
            128,
            2,
            &[2, 8],
            1,
            8,
            117,
            DEFAULT_ANISOTROPIC_THRESHOLD,
        )
        .unwrap();
        let artifact = ScannTrainedArtifact::new(
            19,
            100_000,
            super::super::ScannConfig {
                dimension: 2,
                tree_levels: 2,
                num_leaves: 8,
                encoding: ScannEncoding::AsymmetricHash {
                    dimensions_per_block: 1,
                    bits_per_code: 4,
                },
            },
            model.routing.to_quantized_levels(),
            Some(model.codebook.to_artifact()),
        )
        .unwrap();

        let reopened = FloatScannModel::from_artifact(&artifact).unwrap();
        let artifact_bytes = artifact.to_bytes().unwrap();
        let artifact_view = ScannTrainedArtifactView::parse(&artifact_bytes).unwrap();
        let quantized = QuantizedFloatScannModel::from_artifact_view(&artifact_view).unwrap();
        let quantized_view = quantized.view(&artifact_bytes).unwrap();
        assert_eq!(reopened.codebook, model.codebook);
        assert_eq!(
            reopened.anisotropic_threshold(),
            DEFAULT_ANISOTROPIC_THRESHOLD
        );
        assert_eq!(reopened.routing.level_counts().collect::<Vec<_>>(), [2, 8]);
        for point in data.chunks_exact(2).take(16) {
            let encoded = reopened.encode(point).unwrap();
            let query = reopened.prepare_query(point, 8).unwrap();
            assert!(query.score(&encoded).unwrap().unwrap().is_finite());
            assert_eq!(quantized_view.encode(point).unwrap(), encoded);
            assert_eq!(quantized_view.prepare_query(point, 8).unwrap(), query);
        }
    }

    #[test]
    fn float_scann_end_to_end_recall_is_deterministic() {
        let mut data = Vec::with_capacity(256 * 8);
        for row in 0..256 {
            let mut vector: Vec<f32> = (0..8)
                .map(|coordinate| {
                    let raw = ((row * 73 + coordinate * 151 + row * coordinate * 19) % 997) as f32;
                    raw / 498.5 - 1.0
                })
                .collect();
            let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
            vector.iter_mut().for_each(|value| *value /= norm);
            data.extend(vector);
        }
        let (model, encoded) =
            FloatScannModel::train(&data, 256, 8, &[2, 8], 2, 7, 123, 0.2).unwrap();
        let mut recalled = 0usize;
        let queries = 32usize;
        let k = 10usize;
        for query in data.chunks_exact(8).take(queries) {
            let mut exact: Vec<(usize, f32)> = data
                .chunks_exact(8)
                .enumerate()
                .map(|(row, vector)| {
                    (
                        row,
                        crate::structures::simd::dot_product_f32(query, vector, 8),
                    )
                })
                .collect();
            exact.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
            let prepared = model.prepare_query(query, 8).unwrap();
            let mut approximate: Vec<(usize, f32)> = encoded
                .iter()
                .enumerate()
                .map(|(row, vector)| (row, prepared.score(vector).unwrap().unwrap()))
                .collect();
            approximate.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
            recalled += approximate[..k]
                .iter()
                .filter(|(row, _)| exact[..k].iter().any(|(exact_row, _)| exact_row == row))
                .count();
        }
        let recall = recalled as f32 / (queries * k) as f32;
        assert!(recall >= 0.70, "unexpected float ScaNN recall@10: {recall}");
    }
}
