//! Shared ANN index constants and construction helpers.
//!
//! Constants are available on all platforms (including WASM).
//! Builder/serialization functions are native-only.

/// Index type discriminants stored in the vectors file TOC.
/// Type 2 was IVF-PQ, removed after IVF-TQ superseded it; the loader still
/// recognizes it to fail with an actionable message. Never reuse it.
pub const LEGACY_IVF_PQ_TYPE: u8 = 2;
pub const FLAT_TYPE: u8 = 4;
/// Binary IVF payload backed by an index-level global quantizer.
pub const BINARY_IVF_TYPE: u8 = 6;
/// TurboQuant flat payload; training-free, no global artifacts.
pub const TQ_FLAT_TYPE: u8 = 7;
/// IVF-TQ payload: trained coarse router, TurboQuant residual leaves.
pub const IVF_TQ_TYPE: u8 = 8;
/// ScaNN float-AH payload tied to one global trained model.
pub const SCANN_AH_TYPE: u8 = 9;
/// ScaNN binary-Hamming payload tied to one global trained model.
pub const SCANN_BINARY_TYPE: u8 = 10;

// --- Native-only builder/serialization functions ---

#[cfg(feature = "native")]
use crate::structures::CoarseCentroids;

/// Encode one segment's vectors into IVF-TQ leaves against the trained global
/// cosine-space centroids.
#[cfg(feature = "native")]
pub fn build_ivf_tq(
    dim: usize,
    routing: crate::dsl::IvfRoutingMode,
    centroids: &CoarseCentroids,
    doc_id_ordinals: &[(u32, u16)],
    vectors: &[f32],
) -> crate::Result<Vec<u8>> {
    if !crate::structures::is_ivf_tq_cosine_generation(centroids.version) {
        return Err(crate::Error::Corruption(
            "legacy raw IVF-TQ centroids cannot encode a correct cosine index; \
             rebuild the index with a current Hermes version"
                .into(),
        ));
    }
    let codec = crate::structures::vector::quantization::tq_shared_codec(dim);
    let mut index = crate::structures::IvfTqIndex::new(dim, routing, centroids.version, codec);
    index
        .add_vectors_parallel(centroids, doc_id_ordinals, vectors)
        .map_err(|error| crate::Error::Internal(format!("IVF-TQ encode failed: {error}")))?;
    let mut bytes = Vec::new();
    crate::segment::ann_disk::write_built_ivf_tq(&index, centroids.num_clusters, &mut bytes)
        .map_err(crate::Error::Io)?;
    Ok(bytes)
}

/// Encode one segment's vectors with the training-free TurboQuant codec.
#[cfg(feature = "native")]
pub fn build_tq_flat(
    dim: usize,
    doc_id_ordinals: &[(u32, u16)],
    vectors: &[f32],
) -> crate::Result<Vec<u8>> {
    let codec = crate::structures::vector::quantization::tq_shared_codec(dim);
    let mut builder = crate::structures::TqFlatBuilder::new(codec);
    builder
        .add_batch(doc_id_ordinals, vectors)
        .map_err(|error| crate::Error::Internal(format!("TQ encode failed: {error}")))?;
    builder.finish();
    let mut bytes = Vec::new();
    crate::segment::ann_disk::write_built_tq_flat(&builder, &mut bytes)
        .map_err(crate::Error::Io)?;
    Ok(bytes)
}

/// Encode one immutable segment against the index-global float ScaNN model.
/// The payload contains only leaf-local AH codes and document labels; the
/// routing tree/codebook stays in the generation artifact.
#[cfg(feature = "native")]
#[derive(Default)]
struct ScannAhLeafBuilder {
    doc_ids: Vec<u32>,
    ordinals: Vec<u16>,
    unpacked_codes: Vec<u8>,
}

#[cfg(feature = "native")]
pub(crate) struct ScannAhSegmentBuilder<'a> {
    artifact: &'a crate::segment::ScannTrainedArtifactBytes,
    leaves: std::collections::BTreeMap<u32, ScannAhLeafBuilder>,
    encode_scratch: crate::structures::vector::scann::FloatEncodeScratch,
}

#[cfg(feature = "native")]
impl<'a> ScannAhSegmentBuilder<'a> {
    pub(crate) fn new(artifact: &'a crate::segment::ScannTrainedArtifactBytes) -> Self {
        Self {
            artifact,
            leaves: std::collections::BTreeMap::new(),
            encode_scratch: Default::default(),
        }
    }

    pub(crate) fn add_batch(
        &mut self,
        doc_id_ordinals: &[(u32, u16)],
        vectors: &[f32],
    ) -> crate::Result<()> {
        let model = self.artifact.float_model().map_err(crate::Error::Io)?;
        let dim = self.artifact.config().dimension as usize;
        if vectors.len() != doc_id_ordinals.len().saturating_mul(dim) {
            return Err(crate::Error::Corruption(
                "ScaNN segment labels and float vectors have different lengths".into(),
            ));
        }
        let mut normalized = vec![0.0f32; dim];
        for (&(doc_id, ordinal), vector) in doc_id_ordinals.iter().zip(vectors.chunks_exact(dim)) {
            normalized.copy_from_slice(vector);
            crate::structures::vector::ivf::routing::normalize_cosine_in_place(&mut normalized);
            let (leaf_id, codes) = model
                .encode_with_scratch(&normalized, &mut self.encode_scratch)
                .map_err(|error| {
                    crate::Error::Internal(format!("ScaNN AH encode failed: {error}"))
                })?;
            let leaf = self.leaves.entry(leaf_id).or_default();
            leaf.doc_ids.push(doc_id);
            leaf.ordinals.push(ordinal);
            leaf.unpacked_codes.extend_from_slice(codes);
        }
        Ok(())
    }

    pub(crate) fn finish(
        self,
        doc_count: u32,
    ) -> crate::Result<crate::structures::vector::scann::ScannSegmentPayload> {
        use crate::structures::vector::scann::{
            FAST_SCAN_LANES, ScannEncoding, ScannLeafRun, ScannSegmentPayload, pack_fast_scan_block,
        };

        let artifact = self.artifact;
        let dim = artifact.config().dimension as usize;
        let encoding = artifact.config().encoding;
        let dimensions_per_block = match encoding {
            ScannEncoding::AsymmetricHash {
                dimensions_per_block,
                bits_per_code: 4,
            } => dimensions_per_block,
            _ => {
                return Err(crate::Error::Corruption(
                    "float ScaNN segment requires a 4-bit AH artifact".into(),
                ));
            }
        };
        let blocks = dim.div_ceil(usize::from(dimensions_per_block));
        let mut runs = Vec::with_capacity(self.leaves.len());
        for (leaf_id, leaf) in self.leaves {
            let ScannAhLeafBuilder {
                doc_ids,
                ordinals,
                unpacked_codes: unpacked,
            } = leaf;
            let mut codes = Vec::with_capacity(
                encoding
                    .leaf_code_bytes(artifact.config().dimension, doc_ids.len())
                    .map_err(|error| crate::Error::Internal(error.to_string()))?,
            );
            let full_rows = doc_ids.len() / FAST_SCAN_LANES * FAST_SCAN_LANES;
            for rows in unpacked[..full_rows * blocks].chunks_exact(FAST_SCAN_LANES * blocks) {
                pack_fast_scan_block(rows, blocks, &mut codes)
                    .map_err(|error| crate::Error::Internal(error.to_string()))?;
            }
            for row in unpacked[full_rows * blocks..].chunks_exact(blocks) {
                for pair in row.chunks(2) {
                    codes.push(pair[0] | (pair.get(1).copied().unwrap_or(0) << 4));
                }
            }
            runs.push(
                ScannLeafRun::from_rows(
                    leaf_id,
                    0,
                    &doc_ids,
                    &ordinals,
                    codes,
                    encoding,
                    artifact.config().dimension,
                )
                .map_err(|error| crate::Error::Internal(error.to_string()))?,
            );
        }
        if runs.is_empty() {
            return Err(crate::Error::Internal(
                "cannot build an empty ScaNN segment payload".into(),
            ));
        }
        ScannSegmentPayload::from_generation(
            artifact.config(),
            artifact.generation(),
            artifact.artifact_id(),
            doc_count,
            runs,
        )
        .map_err(|error| crate::Error::Internal(error.to_string()))
    }
}

#[cfg(feature = "native")]
pub(crate) fn build_scann_ah(
    artifact: &crate::segment::ScannTrainedArtifactBytes,
    doc_id_ordinals: &[(u32, u16)],
    vectors: &[f32],
) -> crate::Result<Vec<u8>> {
    let doc_count = doc_id_ordinals
        .iter()
        .map(|&(doc_id, _)| doc_id)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| crate::Error::Corruption("ScaNN segment doc count overflows".into()))?;
    let mut builder = ScannAhSegmentBuilder::new(artifact);
    builder.add_batch(doc_id_ordinals, vectors)?;
    let payload = builder.finish(doc_count)?;
    let mut bytes = Vec::new();
    crate::segment::ann_disk::write_built_scann(&payload, &mut bytes).map_err(crate::Error::Io)?;
    Ok(bytes)
}

#[cfg(feature = "native")]
#[derive(Default)]
struct BinaryScannLeafBuilder {
    doc_ids: Vec<u32>,
    ordinals: Vec<u16>,
    codes: Vec<u8>,
}

#[cfg(feature = "native")]
#[derive(Clone, Copy)]
struct BinaryScannSecondaryCandidate {
    primary_leaf: u32,
    primary_row: u32,
    secondary_leaf: u32,
    primary_distance: u32,
    input_order: u64,
}

#[cfg(feature = "native")]
pub(crate) struct BinaryScannPayloadBuilder<'a> {
    artifact: &'a crate::segment::ScannTrainedArtifactBytes,
    leaves: std::collections::BTreeMap<u32, BinaryScannLeafBuilder>,
    scratch: crate::structures::vector::scann::BinaryScannSearchScratch,
    soar: Option<crate::structures::SoarConfig>,
    secondary_candidates: Vec<BinaryScannSecondaryCandidate>,
    logical_vectors: u64,
}

#[cfg(feature = "native")]
impl<'a> BinaryScannPayloadBuilder<'a> {
    pub(crate) fn new(
        artifact: &'a crate::segment::ScannTrainedArtifactBytes,
        soar: Option<&crate::structures::SoarConfig>,
    ) -> Self {
        Self {
            artifact,
            leaves: std::collections::BTreeMap::new(),
            scratch: Default::default(),
            soar: soar.cloned(),
            secondary_candidates: Vec::new(),
            logical_vectors: 0,
        }
    }

    pub(crate) fn add_batch(&mut self, labels: &[(u32, u16)], codes: &[u8]) -> crate::Result<()> {
        let model = self.artifact.binary_model().map_err(crate::Error::Io)?;
        let byte_len = model.dim_bits() as usize / 8;
        if codes.len() != labels.len().saturating_mul(byte_len) {
            return Err(crate::Error::Corruption(
                "binary ScaNN labels and codes have different lengths".into(),
            ));
        }
        if self
            .soar
            .as_ref()
            .is_some_and(|config| !config.spill_threshold.is_finite())
        {
            return Err(crate::Error::Schema(
                "binary ScaNN spill threshold must be finite".into(),
            ));
        }
        let spill_enabled = self
            .soar
            .as_ref()
            .is_some_and(|config| config.num_secondary > 0)
            && model.num_leaves() > 1;
        for (&(doc_id, ordinal), code) in labels.iter().zip(codes.chunks_exact(byte_len)) {
            let assignment = if spill_enabled {
                model
                    .spill_assignment(code, &mut self.scratch)
                    .map_err(|error| crate::Error::Internal(error.to_string()))?
            } else {
                crate::structures::vector::scann::BinaryScannSpillAssignment {
                    primary_leaf: model
                        .assign(code, &mut self.scratch)
                        .map_err(|error| crate::Error::Internal(error.to_string()))?,
                    secondary_leaf: None,
                    primary_distance: 0,
                }
            };
            let leaf = self.leaves.entry(assignment.primary_leaf).or_default();
            let primary_row = u32::try_from(leaf.doc_ids.len()).map_err(|_| {
                crate::Error::Corruption("binary ScaNN leaf row count exceeds u32".into())
            })?;
            leaf.doc_ids.push(doc_id);
            leaf.ordinals.push(ordinal);
            leaf.codes.extend_from_slice(code);
            if let Some(secondary_leaf) = assignment.secondary_leaf {
                self.secondary_candidates
                    .push(BinaryScannSecondaryCandidate {
                        primary_leaf: assignment.primary_leaf,
                        primary_row,
                        secondary_leaf,
                        primary_distance: assignment.primary_distance,
                        input_order: self.logical_vectors,
                    });
            }
            self.logical_vectors = self.logical_vectors.checked_add(1).ok_or_else(|| {
                crate::Error::Corruption("binary ScaNN vector count overflows".into())
            })?;
        }
        Ok(())
    }

    pub(crate) fn finish(
        mut self,
        doc_count: u32,
    ) -> crate::Result<crate::structures::vector::scann::ScannSegmentPayload> {
        if let Some(soar) = self.soar.as_ref() {
            if let Some(target_fraction) = soar.calibration_target() {
                let budget = ((self.logical_vectors as f64 * f64::from(target_fraction)).floor()
                    as usize)
                    .min(self.secondary_candidates.len());
                let by_spill_priority =
                    |left: &BinaryScannSecondaryCandidate,
                     right: &BinaryScannSecondaryCandidate| {
                        right
                            .primary_distance
                            .cmp(&left.primary_distance)
                            .then_with(|| left.input_order.cmp(&right.input_order))
                    };
                if budget < self.secondary_candidates.len() {
                    self.secondary_candidates
                        .select_nth_unstable_by(budget, by_spill_priority);
                }
                self.secondary_candidates.truncate(budget);
            } else if soar.selective {
                let threshold_squared = f64::from(soar.spill_threshold).powi(2);
                self.secondary_candidates
                    .retain(|candidate| f64::from(candidate.primary_distance) >= threshold_squared);
            }
        } else {
            self.secondary_candidates.clear();
        }

        let byte_len = self.artifact.config().dimension as usize / 8;
        self.secondary_candidates.sort_unstable_by(|left, right| {
            left.secondary_leaf
                .cmp(&right.secondary_leaf)
                .then_with(|| left.input_order.cmp(&right.input_order))
        });
        let mut cursor = 0usize;
        while cursor < self.secondary_candidates.len() {
            let secondary_leaf = self.secondary_candidates[cursor].secondary_leaf;
            // A secondary leaf is never its candidate's primary leaf. Removing
            // one destination therefore permits immutable source-leaf reads
            // and direct appends into the final allocation, without a
            // corpus-sized Vec<Vec<u8>> or a second copy of every packed code.
            let mut destination = self.leaves.remove(&secondary_leaf).unwrap_or_default();
            while cursor < self.secondary_candidates.len()
                && self.secondary_candidates[cursor].secondary_leaf == secondary_leaf
            {
                let candidate = self.secondary_candidates[cursor];
                if candidate.primary_leaf == secondary_leaf {
                    return Err(crate::Error::Corruption(
                        "binary ScaNN secondary assignment repeats its primary leaf".into(),
                    ));
                }
                let primary = self.leaves.get(&candidate.primary_leaf).ok_or_else(|| {
                    crate::Error::Corruption("binary ScaNN spill source leaf is missing".into())
                })?;
                let row = candidate.primary_row as usize;
                let start = row.checked_mul(byte_len).ok_or_else(|| {
                    crate::Error::Corruption("binary ScaNN spill code offset overflows".into())
                })?;
                let end = start.checked_add(byte_len).ok_or_else(|| {
                    crate::Error::Corruption("binary ScaNN spill code end overflows".into())
                })?;
                let code = primary.codes.get(start..end).ok_or_else(|| {
                    crate::Error::Corruption("binary ScaNN spill source row is invalid".into())
                })?;
                let doc_id = primary.doc_ids.get(row).copied().ok_or_else(|| {
                    crate::Error::Corruption("binary ScaNN spill source document is invalid".into())
                })?;
                let ordinal = primary.ordinals.get(row).copied().ok_or_else(|| {
                    crate::Error::Corruption("binary ScaNN spill source ordinal is invalid".into())
                })?;
                destination.doc_ids.push(doc_id);
                destination.ordinals.push(ordinal);
                destination.codes.extend_from_slice(code);
                cursor += 1;
            }
            self.leaves.insert(secondary_leaf, destination);
        }

        let mut runs = Vec::with_capacity(self.leaves.len());
        for (leaf_id, leaf) in self.leaves {
            runs.push(
                crate::structures::vector::scann::ScannLeafRun::from_rows(
                    leaf_id,
                    0,
                    &leaf.doc_ids,
                    &leaf.ordinals,
                    leaf.codes,
                    crate::structures::vector::scann::ScannEncoding::BinaryHamming,
                    self.artifact.config().dimension,
                )
                .map_err(|error| crate::Error::Internal(error.to_string()))?,
            );
        }
        crate::structures::vector::scann::ScannSegmentPayload::from_generation(
            self.artifact.config(),
            self.artifact.generation(),
            self.artifact.artifact_id(),
            doc_count,
            runs,
        )
        .map_err(|error| crate::Error::Internal(error.to_string()))
    }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    use crate::directories::OwnedBytes;
    use crate::dsl::IvfRoutingMode;
    use crate::segment::ann_disk::{AnnDiskIndex, AnnKind};

    #[test]
    fn ivf_tq_header_preserves_cosine_generation_marker() {
        let marked = crate::structures::mark_ivf_tq_cosine_generation(7);
        let centroids = CoarseCentroids {
            num_clusters: 1,
            dim: 2,
            centroids: vec![1.0, 0.0],
            version: marked,
            soar_config: None,
            routing_index: None,
        };
        let bytes = build_ivf_tq(
            2,
            IvfRoutingMode::Flat,
            &centroids,
            &[(0, 0)],
            &[100.0, 0.0],
        )
        .unwrap();
        let disk = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::IvfTq, 1).unwrap();

        assert_eq!(disk.header().quantizer_version, marked);
        assert!(crate::structures::is_ivf_tq_cosine_generation(
            disk.header().quantizer_version
        ));
    }

    #[test]
    fn ivf_tq_build_rejects_legacy_raw_generation() {
        let centroids = CoarseCentroids {
            num_clusters: 1,
            dim: 2,
            centroids: vec![1.0, 0.0],
            version: 7,
            soar_config: None,
            routing_index: None,
        };
        let error = build_ivf_tq(2, IvfRoutingMode::Flat, &centroids, &[(0, 0)], &[1.0, 0.0])
            .expect_err("legacy IVF-TQ generation must not encode new segments")
            .to_string();
        assert!(error.contains("legacy raw IVF-TQ"), "{error}");
        assert!(error.contains("rebuild the index"), "{error}");
    }

    #[test]
    fn scann_ah_segment_build_and_fast_scan_share_the_global_generation() {
        use crate::structures::vector::scann::{
            DEFAULT_ANISOTROPIC_THRESHOLD, FloatScannModel, ScannConfig, ScannEncoding,
            ScannTrainedArtifact,
        };

        let dim = 4usize;
        let points = 256usize;
        let vectors: Vec<f32> = (0..points)
            .flat_map(|row| {
                let sign = if row < points / 2 { -1.0 } else { 1.0 };
                [sign, row as f32 / points as f32, sign * 0.5, 0.25]
            })
            .collect();
        let (model, _) = FloatScannModel::train(
            &vectors,
            points,
            dim,
            &[2],
            2,
            4,
            17,
            DEFAULT_ANISOTROPIC_THRESHOLD,
        )
        .unwrap();
        let artifact = ScannTrainedArtifact::new(
            7,
            crate::structures::vector::scann::MIN_POINTS_FOR_PARTITIONING,
            ScannConfig {
                dimension: dim as u32,
                tree_levels: 1,
                num_leaves: 2,
                encoding: ScannEncoding::AsymmetricHash {
                    dimensions_per_block: 2,
                    bits_per_code: 4,
                },
            },
            model.routing.to_quantized_levels(),
            Some(model.codebook.to_artifact()),
        )
        .unwrap();
        let holder = crate::segment::ScannTrainedArtifactBytes::open(OwnedBytes::new(
            artifact.to_bytes().unwrap(),
        ))
        .unwrap();
        let labels: Vec<(u32, u16)> = (0..points as u32).map(|doc| (doc, 0)).collect();
        let bytes = build_scann_ah(&holder, &labels, &vectors).unwrap();
        let disk =
            AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::ScannAh, points as u32).unwrap();
        disk.validate_scann_generation(holder.config(), 7, holder.artifact_id())
            .unwrap();
        let query = holder
            .float_model()
            .unwrap()
            .prepare_query(&vectors[..dim], 2)
            .unwrap();
        let candidates = disk
            .search_scann_ah_combined_documents(8, &query, crate::query::MultiValueCombiner::Max)
            .unwrap();
        assert!(candidates.iter().any(|candidate| candidate.doc_id == 0));
    }

    #[test]
    fn binary_scann_artifact_segment_and_disk_search_round_trip() {
        use crate::structures::vector::scann::{
            BinaryScannModel, BinaryScannSearchScratch, BinaryScannTraining, geometry_for_leaves,
        };

        let training_codes: Vec<u8> = (0..100_000).map(|row| row as u8).collect();
        let model = BinaryScannModel::train(
            &BinaryScannTraining {
                dim_bits: 8,
                geometry: geometry_for_leaves(2, 1).unwrap(),
                train_iters: 2,
                seed: 9,
            },
            &training_codes,
            100_000,
            "test",
        )
        .unwrap();
        let artifact = model.to_artifact(11, 100_000).unwrap();
        let holder = crate::segment::ScannTrainedArtifactBytes::open(OwnedBytes::new(
            artifact.to_bytes().unwrap(),
        ))
        .unwrap();
        let labels: Vec<(u32, u16)> = (0..64).map(|doc| (doc, 0)).collect();
        let codes: Vec<u8> = (0..64).map(|row| row as u8).collect();
        let soar = crate::structures::SoarConfig::new().target_spill_fraction(0.30);
        let mut builder = BinaryScannPayloadBuilder::new(&holder, Some(&soar));
        // The target is one segment-wide budget, not the sum of independently
        // rounded batch budgets: floor(64 * .30) = 19, while this 1/63 split
        // would retain only 18 if each batch were budgeted separately.
        builder.add_batch(&labels[..1], &codes[..1]).unwrap();
        builder.add_batch(&labels[1..], &codes[1..]).unwrap();
        let payload = builder.finish(64).unwrap();
        let mut monolithic = BinaryScannPayloadBuilder::new(&holder, Some(&soar));
        monolithic.add_batch(&labels, &codes).unwrap();
        assert_eq!(payload, monolithic.finish(64).unwrap());
        let mut bytes = Vec::new();
        crate::segment::ann_disk::write_built_scann(&payload, &mut bytes).unwrap();
        let disk = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::ScannBinary, 64).unwrap();
        assert_eq!(
            disk.header().vector_count,
            64 + (64.0f32 * 0.30).floor() as usize
        );
        let mut scratch = BinaryScannSearchScratch::default();
        let plan = holder
            .binary_model()
            .unwrap()
            .probe(&[0], 2, 2, &mut scratch)
            .unwrap();
        let hits = disk
            .search_binary_clusters::<false>(&[0], 1, &plan.leaf_ids)
            .unwrap();
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits[0].2, 1.0);

        let all_hits = disk
            .search_binary_clusters::<false>(&[0], 64, &plan.leaf_ids)
            .unwrap();
        assert_eq!(
            all_hits.len(),
            64,
            "spilled postings must be deduplicated by logical vector"
        );
    }
}
