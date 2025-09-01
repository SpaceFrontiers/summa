//! IVF-TQ: inverted-file search with TurboQuant-coded centroid residuals.
//!
//! Two-level index (design: `docs/turboquant-quantization.md`):
//! - Level 1: the trained global coarse quantizer (same router as IVF-PQ).
//! - Level 2: per-leaf TQ codes of `normalized(vector) − centroid` residuals
//!   plus a per-vector residual scale, so
//!   `⟨q̂, x̂⟩ = ⟨q̂, c⟩ + scale · ⟨q̂, r̂⟩`.
//!
//! Unlike IVF-PQ, the leaf codec is training-free and the residual estimate
//! uses one query-wide LUT: probed clusters differ only by the scalar
//! `⟨q̂, c⟩`, so plan build is O(P·16 + nprobe·dim) instead of nprobe ADC
//! tables. Segment persistence and pure-copy merging live in
//! `segment::ann_disk`; this type is build-only.

use crate::dsl::IvfRoutingMode;
use crate::structures::vector::ivf::routing::{
    cosine_probe_fingerprint, normalize_cosine_in_place, normalized_cosine_query,
};
use crate::structures::vector::ivf::{CoarseCentroids, IvfProbePlan};
use crate::structures::vector::quantization::{TqCodec, TqEncodeScratch, TqQueryPlan};

// Preserve the serialized centroid and ANN-header layouts by carrying the
// cosine-generation format in the existing u64 version. Timestamp-generated
// legacy versions cannot reach this prefix for centuries; the mixed 52-bit
// payload retains ample generation entropy without exposing timestamp bits.
const IVF_TQ_COSINE_VERSION_MASK: u64 = 0xfff0_0000_0000_0000;
const IVF_TQ_COSINE_VERSION_TAG: u64 = 0xc050_0000_0000_0000;

/// Mark an IVF-TQ centroid version whose centroids and residuals use normalized
/// cosine-space vectors. The transformation is idempotent.
pub fn mark_ivf_tq_cosine_generation(version: u64) -> u64 {
    if is_ivf_tq_cosine_generation(version) {
        return version;
    }
    let mut mixed = version;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^= mixed >> 31;
    IVF_TQ_COSINE_VERSION_TAG | (mixed & !IVF_TQ_COSINE_VERSION_MASK)
}

/// Whether a persisted IVF-TQ generation was trained and encoded in normalized
/// cosine space. Readers use this marker to reject incompatible artifacts.
#[inline]
pub fn is_ivf_tq_cosine_generation(version: u64) -> bool {
    version & IVF_TQ_COSINE_VERSION_MASK == IVF_TQ_COSINE_VERSION_TAG
}

/// Struct-of-arrays payload for one non-empty IVF-TQ leaf. `rows` holds one
/// unpacked nibble value (0..=15) per padded coordinate per vector; blocks
/// are packed lazily at serialization time.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TqCluster {
    pub(crate) doc_ids: Vec<u32>,
    pub(crate) ordinals: Vec<u16>,
    pub(crate) scales: Vec<f32>,
    pub(crate) gammas: Vec<f32>,
    pub(crate) rows: Vec<u8>,
}

impl TqCluster {
    #[cfg(feature = "native")]
    fn append_owned(&mut self, mut source: Self) {
        self.doc_ids.append(&mut source.doc_ids);
        self.ordinals.append(&mut source.ordinals);
        self.scales.append(&mut source.scales);
        self.gammas.append(&mut source.gammas);
        self.rows.append(&mut source.rows);
    }
}

/// Query-global IVF-TQ work shared by every segment: one probe route, one
/// set of TQ LUTs, and one `⟨q̂, centroid⟩` scalar per probed cluster.
pub struct TqIvfQueryPlan {
    pub quantizer_version: u64,
    /// TQ codec fingerprint (carried as `codebook_version` on disk).
    pub fingerprint: u64,
    pub request_fingerprint: u64,
    pub cluster_ids: std::sync::Arc<[u32]>,
    cluster_dots: Vec<f32>,
    tq: TqQueryPlan,
}

impl std::fmt::Debug for TqIvfQueryPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TqIvfQueryPlan")
            .field("quantizer_version", &self.quantizer_version)
            .field("fingerprint", &self.fingerprint)
            .field("cluster_count", &self.cluster_ids.len())
            .finish()
    }
}

impl TqIvfQueryPlan {
    pub fn build(
        coarse_centroids: &CoarseCentroids,
        codec: &TqCodec,
        query: &[f32],
        nprobe: usize,
        routing: IvfRoutingMode,
    ) -> Self {
        assert!(
            is_ivf_tq_cosine_generation(coarse_centroids.version),
            "legacy raw IVF-TQ generations cannot build a query plan; rebuild the index"
        );
        let normalized_query = normalized_cosine_query(query);
        let route: IvfProbePlan = coarse_centroids.probe(&normalized_query, nprobe, routing);
        let effective_nprobe = nprobe.clamp(1, coarse_centroids.num_clusters as usize);
        if effective_nprobe != nprobe {
            log::debug!(
                "[ivf_tq] nprobe {nprobe} clamped to {effective_nprobe}: codebook has {} leaves",
                coarse_centroids.num_clusters
            );
        }
        let cluster_dots = route
            .cluster_ids
            .iter()
            .map(|&cluster_id| {
                let centroid = coarse_centroids.get_centroid(cluster_id);
                crate::structures::simd::dot_product_f32(
                    &normalized_query,
                    centroid,
                    normalized_query.len(),
                )
            })
            .collect();
        Self {
            quantizer_version: route.quantizer_version,
            fingerprint: codec.fingerprint(),
            request_fingerprint: Self::request_fingerprint_for(
                coarse_centroids,
                query,
                effective_nprobe,
                routing,
            ),
            cluster_ids: route.cluster_ids,
            cluster_dots,
            tq: TqQueryPlan::build(codec, &normalized_query),
        }
    }

    /// Cache identity matching normalized cosine routing geometry.
    pub fn request_fingerprint_for(
        coarse_centroids: &CoarseCentroids,
        query: &[f32],
        nprobe: usize,
        routing: IvfRoutingMode,
    ) -> u64 {
        assert!(
            is_ivf_tq_cosine_generation(coarse_centroids.version),
            "legacy raw IVF-TQ generations cannot build a query fingerprint; rebuild the index"
        );
        let effective_nprobe = nprobe.clamp(1, coarse_centroids.num_clusters as usize);
        cosine_probe_fingerprint(query, effective_nprobe, routing)
    }

    #[inline]
    pub(crate) fn tq_plan(&self) -> &TqQueryPlan {
        &self.tq
    }

    pub(crate) fn cluster_dots(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.cluster_ids
            .iter()
            .copied()
            .zip(self.cluster_dots.iter().copied())
    }
}

/// IVF-TQ build payload for a single segment.
#[derive(Debug, Clone)]
pub struct IvfTqIndex {
    pub dim: usize,
    pub routing: IvfRoutingMode,
    /// Version of coarse centroids used (for merge compatibility).
    pub centroids_version: u64,
    codec: std::sync::Arc<TqCodec>,
    pub(crate) clusters: rustc_hash::FxHashMap<u32, TqCluster>,
    len: usize,
}

impl IvfTqIndex {
    pub fn new(
        dim: usize,
        routing: IvfRoutingMode,
        centroids_version: u64,
        codec: std::sync::Arc<TqCodec>,
    ) -> Self {
        assert_eq!(codec.dim(), dim, "IVF-TQ codec/config dimension mismatch");
        assert!(
            is_ivf_tq_cosine_generation(centroids_version),
            "legacy raw IVF-TQ generations cannot encode vectors; rebuild the index"
        );
        Self {
            dim,
            routing,
            centroids_version,
            codec,
            clusters: rustc_hash::FxHashMap::default(),
            len: 0,
        }
    }

    #[inline]
    pub fn codec(&self) -> &TqCodec {
        &self.codec
    }

    /// Add a single vector: assign to the trained router (with SOAR when
    /// configured) and TQ-encode the residual against every assigned leaf.
    pub fn add_vector(
        &mut self,
        coarse_centroids: &CoarseCentroids,
        doc_id: u32,
        ordinal: u16,
        vector: &[f32],
        scratch: &mut TqIvfEncodeScratch,
    ) {
        assert_eq!(
            coarse_centroids.version, self.centroids_version,
            "IVF-TQ centroid generation mismatch"
        );
        let mut normalized = std::mem::take(&mut scratch.normalized);
        normalized.clear();
        normalized.extend_from_slice(vector);
        normalize_cosine_in_place(&mut normalized);
        self.add_prepared_vector(coarse_centroids, doc_id, ordinal, &normalized, scratch);
        scratch.normalized = normalized;
    }

    fn add_prepared_vector(
        &mut self,
        coarse_centroids: &CoarseCentroids,
        doc_id: u32,
        ordinal: u16,
        vector: &[f32],
        scratch: &mut TqIvfEncodeScratch,
    ) {
        let assignment = coarse_centroids.assign_with_routing(vector, self.routing);
        self.add_to_cluster(
            coarse_centroids,
            assignment.primary_cluster,
            doc_id,
            ordinal,
            vector,
            scratch,
        );
        for &cluster_id in &assignment.secondary_clusters {
            self.add_to_cluster(
                coarse_centroids,
                cluster_id,
                doc_id,
                ordinal,
                vector,
                scratch,
            );
        }
    }

    fn add_to_cluster(
        &mut self,
        coarse_centroids: &CoarseCentroids,
        cluster_id: u32,
        doc_id: u32,
        ordinal: u16,
        vector: &[f32],
        scratch: &mut TqIvfEncodeScratch,
    ) {
        let centroid = coarse_centroids.get_centroid(cluster_id);
        scratch.residual.clear();
        scratch.residual.extend(
            vector
                .iter()
                .zip(centroid)
                .map(|(value, center)| value - center),
        );
        scratch.nibbles.resize(self.codec.padded_dim(), 0);
        let (scale, gamma) = self.codec.encode_residual_into(
            &scratch.residual,
            &mut scratch.nibbles,
            &mut scratch.tq,
        );

        let cluster = self.clusters.entry(cluster_id).or_default();
        cluster.doc_ids.push(doc_id);
        cluster.ordinals.push(ordinal);
        cluster.scales.push(scale);
        cluster.gammas.push(gamma);
        cluster.rows.extend_from_slice(&scratch.nibbles);
        self.len += 1;
    }

    /// Add one contiguous vector batch in parallel while preserving input
    /// order inside every leaf.
    #[cfg(feature = "native")]
    pub fn add_vectors_parallel(
        &mut self,
        coarse_centroids: &CoarseCentroids,
        doc_id_ordinals: &[(u32, u16)],
        vectors: &[f32],
    ) -> Result<(), &'static str> {
        use rayon::prelude::*;

        let vector_count = doc_id_ordinals.len();
        let expected = vector_count
            .checked_mul(self.dim)
            .ok_or("IVF-TQ input size overflow")?;
        if vectors.len() != expected {
            return Err("IVF-TQ vector and label matrices are inconsistent");
        }
        if coarse_centroids.version != self.centroids_version {
            return Err("IVF-TQ centroid generation mismatch");
        }
        if vector_count == 0 {
            return Ok(());
        }

        let target_tasks = rayon::current_num_threads().saturating_mul(4).max(1);
        let chunk_vectors = vector_count.div_ceil(target_tasks).max(64);
        let dim = self.dim;
        let routing = self.routing;
        let centroids_version = self.centroids_version;
        let codec = std::sync::Arc::clone(&self.codec);
        let partials: Vec<Self> = doc_id_ordinals
            .par_chunks(chunk_vectors)
            .enumerate()
            .map(|(chunk_index, labels)| {
                let first = chunk_index * chunk_vectors;
                let chunk = &vectors[first * dim..(first + labels.len()) * dim];
                let mut partial = Self::new(
                    dim,
                    routing,
                    centroids_version,
                    std::sync::Arc::clone(&codec),
                );
                let mut scratch = TqIvfEncodeScratch::default();
                for (&(doc_id, ordinal), vector) in labels.iter().zip(chunk.chunks_exact(dim)) {
                    partial.add_vector(coarse_centroids, doc_id, ordinal, vector, &mut scratch);
                }
                partial
            })
            .collect();

        for partial in partials {
            self.append_owned(partial)?;
        }
        Ok(())
    }

    #[cfg(feature = "native")]
    fn append_owned(&mut self, mut other: Self) -> Result<(), &'static str> {
        if self.centroids_version != other.centroids_version
            || self.dim != other.dim
            || self.codec.fingerprint() != other.codec.fingerprint()
        {
            return Err("Cannot merge IVF-TQ payloads from different generations");
        }
        let other_len = other.len;
        for (cluster_id, source) in other.clusters.drain() {
            match self.clusters.entry(cluster_id) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().append_owned(source);
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(source);
                }
            }
        }
        self.len = self
            .len
            .checked_add(other_len)
            .ok_or("IVF-TQ vector count overflow during parallel build")?;
        Ok(())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn estimated_memory_bytes(&self) -> usize {
        self.clusters
            .values()
            .map(|cluster| {
                cluster.rows.len()
                    + cluster.doc_ids.len() * size_of::<u32>()
                    + cluster.ordinals.len() * size_of::<u16>()
                    + (cluster.scales.len() + cluster.gammas.len()) * size_of::<f32>()
            })
            .sum()
    }
}

/// Reusable per-thread IVF-TQ encode buffers.
#[derive(Debug, Default)]
pub struct TqIvfEncodeScratch {
    normalized: Vec<f32>,
    residual: Vec<f32>,
    nibbles: Vec<u8>,
    tq: TqEncodeScratch,
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    use crate::structures::vector::ivf::CoarseConfig;
    use crate::structures::vector::quantization::{
        TQ_BLOCK_LANES, tq_pack_ivf_block, tq_score_ivf_block,
    };

    fn seeded_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        let mut values: Vec<f32> = (0..dim)
            .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
            .collect();
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        values.iter_mut().for_each(|v| *v /= norm);
        values
    }

    /// The scaled residual estimator must approximate the true inner product
    /// of the normalized-vector residual decomposition.
    #[test]
    fn scaled_residual_estimator_tracks_true_dot() {
        let dim = 64;
        let codec = std::sync::Arc::new(TqCodec::new(dim));
        let vectors: Vec<Vec<f32>> = (0..64).map(|i| seeded_unit_vector(dim, 100 + i)).collect();
        let mut centroids = CoarseCentroids::train(&CoarseConfig::new(dim, 4), &vectors, "test");
        centroids.version = mark_ivf_tq_cosine_generation(centroids.version);

        let mut index = IvfTqIndex::new(
            dim,
            IvfRoutingMode::Flat,
            centroids.version,
            std::sync::Arc::clone(&codec),
        );
        let mut scratch = TqIvfEncodeScratch::default();
        for (i, vector) in vectors.iter().enumerate() {
            index.add_vector(&centroids, i as u32, 0, vector, &mut scratch);
        }
        assert_eq!(index.len(), vectors.len());

        let query = seeded_unit_vector(dim, 7);
        let plan = TqIvfQueryPlan::build(&centroids, &codec, &query, 4, IvfRoutingMode::Flat);

        let mut checked = 0usize;
        let mut squared_error = 0.0f64;
        for (cluster_id, cluster_dot) in plan.cluster_dots() {
            let Some(cluster) = index.clusters.get(&cluster_id) else {
                continue;
            };
            let padded = codec.padded_dim();
            for block_start in (0..cluster.doc_ids.len()).step_by(TQ_BLOCK_LANES) {
                let lanes = TQ_BLOCK_LANES.min(cluster.doc_ids.len() - block_start);
                let rows: Vec<&[u8]> = (0..lanes)
                    .map(|lane| {
                        &cluster.rows
                            [(block_start + lane) * padded..(block_start + lane + 1) * padded]
                    })
                    .collect();
                let mut block = Vec::new();
                tq_pack_ivf_block(
                    &rows,
                    &cluster.scales[block_start..block_start + lanes],
                    &cluster.gammas[block_start..block_start + lanes],
                    padded,
                    &mut block,
                );
                let mut scores = [0.0f32; TQ_BLOCK_LANES];
                tq_score_ivf_block(plan.tq_plan(), &block, cluster_dot, &mut scores);
                for (lane, &score) in scores.iter().enumerate().take(lanes) {
                    let doc = cluster.doc_ids[block_start + lane] as usize;
                    let truth: f32 = vectors[doc].iter().zip(&query).map(|(a, b)| a * b).sum();
                    squared_error += f64::from(score - truth).powi(2);
                    checked += 1;
                }
            }
        }
        assert!(checked >= vectors.len(), "every vector must be scored");
        let rmse = (squared_error / checked as f64).sqrt();
        assert!(rmse < 0.08, "IVF-TQ estimator RMSE too large: {rmse}");
    }

    #[test]
    fn query_plans_are_scale_invariant() {
        let dim = 2;
        let codec = TqCodec::new(dim);
        let query = vec![1.0, 0.0];
        let scaled_query: Vec<f32> = query.iter().map(|value| value * 100.0).collect();
        for routing in [IvfRoutingMode::Auto, IvfRoutingMode::Flat] {
            let centroids = CoarseCentroids {
                num_clusters: 2,
                dim,
                // An intentionally non-unit fixture makes accidental raw-L2
                // routing select a different leaf for the scaled query.
                centroids: vec![1.0, 0.0, 10.0, 0.0],
                version: mark_ivf_tq_cosine_generation(7),
                soar_config: None,
                routing_index: None,
            };
            let plan = TqIvfQueryPlan::build(&centroids, &codec, &query, 1, routing);
            let scaled_plan = TqIvfQueryPlan::build(&centroids, &codec, &scaled_query, 1, routing);

            assert_eq!(&*plan.cluster_ids, &[0]);
            assert_eq!(plan.cluster_ids, scaled_plan.cluster_ids);
            assert_eq!(plan.request_fingerprint, scaled_plan.request_fingerprint);
            assert_eq!(plan.cluster_dots, scaled_plan.cluster_dots);
        }
    }

    #[test]
    fn vector_encoding_always_normalizes_input() {
        let codec = std::sync::Arc::new(TqCodec::new(2));
        let centroids = CoarseCentroids {
            num_clusters: 1,
            dim: 2,
            centroids: vec![1.0, 0.0],
            version: mark_ivf_tq_cosine_generation(7),
            soar_config: None,
            routing_index: None,
        };
        let mut index = IvfTqIndex::new(
            2,
            IvfRoutingMode::Flat,
            centroids.version,
            std::sync::Arc::clone(&codec),
        );
        let mut scratch = TqIvfEncodeScratch::default();

        index.add_vector(&centroids, 0, 0, &[2.0, 0.0], &mut scratch);
        index.add_vector(&centroids, 1, 0, &[128.0, 0.0], &mut scratch);

        let cluster = &index.clusters[&0];
        assert_eq!(cluster.doc_ids, [0, 1]);
        assert_eq!(cluster.scales, [0.0, 0.0]);
        assert_eq!(cluster.gammas[0], cluster.gammas[1]);
        assert_eq!(
            &cluster.rows[..index.codec.padded_dim()],
            &cluster.rows[index.codec.padded_dim()..]
        );
    }

    #[test]
    #[should_panic(expected = "legacy raw IVF-TQ generations cannot build a query plan")]
    fn query_plan_rejects_legacy_generation() {
        let centroids = CoarseCentroids {
            num_clusters: 1,
            dim: 2,
            centroids: vec![1.0, 0.0],
            version: 7,
            soar_config: None,
            routing_index: None,
        };
        let codec = TqCodec::new(2);
        let _ = TqIvfQueryPlan::build(&centroids, &codec, &[1.0, 0.0], 1, IvfRoutingMode::Flat);
    }

    #[test]
    #[should_panic(expected = "legacy raw IVF-TQ generations cannot encode vectors")]
    fn index_builder_rejects_legacy_generation() {
        let _ = IvfTqIndex::new(
            2,
            IvfRoutingMode::Flat,
            7,
            std::sync::Arc::new(TqCodec::new(2)),
        );
    }

    #[test]
    fn cosine_generation_marker_is_stable_and_distinct_from_unmarked() {
        let marked = mark_ivf_tq_cosine_generation(7);
        assert!(!is_ivf_tq_cosine_generation(7));
        assert!(is_ivf_tq_cosine_generation(marked));
        assert_eq!(mark_ivf_tq_cosine_generation(marked), marked);
        assert_ne!(marked, mark_ivf_tq_cosine_generation(8));
    }
}
