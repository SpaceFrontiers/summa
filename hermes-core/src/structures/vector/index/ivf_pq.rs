//! IVF-PQ: inverted-file search with product-quantized residuals.
//!
//! Two-level index for vector search:
//! - Level 1: Coarse quantizer (k-means centroids)
//! - Level 2: Product Quantization codes per cluster
//!
//! Key feature: Segments sharing the same coarse centroids and PQ codebook
//! can be merged in O(1) by concatenating cluster data.

use serde::{Deserialize, Serialize};
use std::io;

use crate::dsl::IvfRoutingMode;
use crate::structures::vector::ivf::{CoarseCentroids, IvfProbePlan, MultiAssignment};
use crate::structures::vector::quantization::{DistanceTable, PQCodebook};

/// Struct-of-arrays payload for one non-empty float IVF leaf. PQ codes are a
/// single `count * code_size` byte column: no allocation, length prefix, or
/// `Vec` header exists per vector.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct PqCluster {
    pub(crate) doc_ids: Vec<u32>,
    ordinals: Vec<u16>,
    codes: Vec<u8>,
}

/// Query-global IVF-PQ work shared by every segment. Both leaf routing and
/// ADC tables depend only on the query and index-level artifacts, so doing
/// either per segment multiplies identical work by the segment count.
pub struct IvfPqQueryPlan {
    pub quantizer_version: u64,
    pub codebook_version: u64,
    pub request_fingerprint: u64,
    pub cluster_ids: std::sync::Arc<[u32]>,
    distance_tables: Vec<DistanceTable>,
}

impl std::fmt::Debug for IvfPqQueryPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("IvfPqQueryPlan")
            .field("quantizer_version", &self.quantizer_version)
            .field("codebook_version", &self.codebook_version)
            .field("request_fingerprint", &self.request_fingerprint)
            .field("cluster_count", &self.cluster_ids.len())
            .field("distance_table_count", &self.distance_tables.len())
            .finish()
    }
}

impl IvfPqQueryPlan {
    pub fn build(
        coarse_centroids: &CoarseCentroids,
        codebook: &PQCodebook,
        query: &[f32],
        nprobe: usize,
        routing: IvfRoutingMode,
    ) -> Self {
        let route: IvfProbePlan = coarse_centroids.probe(query, nprobe, routing);
        let distance_tables = route
            .cluster_ids
            .iter()
            .map(|&cluster_id| {
                DistanceTable::build(
                    codebook,
                    query,
                    Some(coarse_centroids.get_centroid(cluster_id)),
                )
            })
            .collect();
        Self {
            quantizer_version: route.quantizer_version,
            codebook_version: codebook.version,
            request_fingerprint: route.request_fingerprint,
            cluster_ids: route.cluster_ids,
            distance_tables,
        }
    }
}

impl PqCluster {
    fn append(&mut self, source: &Self, doc_id_offset: u32) {
        self.doc_ids.reserve(source.doc_ids.len());
        self.ordinals.reserve(source.ordinals.len());
        self.codes.reserve(source.codes.len());
        self.doc_ids
            .extend(source.doc_ids.iter().map(|doc_id| doc_id + doc_id_offset));
        self.ordinals.extend_from_slice(&source.ordinals);
        self.codes.extend_from_slice(&source.codes);
    }

    #[cfg(feature = "native")]
    fn append_owned(&mut self, mut source: Self) {
        self.doc_ids.append(&mut source.doc_ids);
        self.ordinals.append(&mut source.ordinals);
        self.codes.append(&mut source.codes);
    }
}

/// Configuration for IVF-PQ index
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IVFPQConfig {
    /// Vector dimension
    pub dim: usize,
    /// PQ bytes per vector (one centroid ID per subspace).
    pub code_size: usize,
    /// Assignment routing used when constructing segment payloads.
    pub routing: IvfRoutingMode,
}

impl IVFPQConfig {
    pub fn new(dim: usize, code_size: usize) -> Self {
        Self {
            dim,
            code_size,
            routing: IvfRoutingMode::Auto,
        }
    }

    pub fn with_routing(mut self, routing: IvfRoutingMode) -> Self {
        self.routing = routing;
        self
    }
}

/// IVF-PQ index for a single segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFPQIndex {
    /// Configuration
    pub config: IVFPQConfig,
    /// Version of coarse centroids used (for merge compatibility)
    pub centroids_version: u64,
    /// Version of PQ codebook used (for merge compatibility)
    pub codebook_version: u64,
    /// Non-empty leaves only. Each leaf stores contiguous PQ bytes.
    pub(crate) clusters: rustc_hash::FxHashMap<u32, PqCluster>,
    len: usize,
    /// Build-only scratch, omitted from the persisted segment payload.
    #[serde(skip)]
    residual_scratch: Vec<f32>,
    #[serde(skip)]
    rotated_scratch: Vec<f32>,
}

impl IVFPQIndex {
    /// Create a new empty IVF-PQ index
    pub fn new(config: IVFPQConfig, centroids_version: u64, codebook_version: u64) -> Self {
        Self {
            config,
            centroids_version,
            codebook_version,
            clusters: rustc_hash::FxHashMap::default(),
            len: 0,
            residual_scratch: Vec::new(),
            rotated_scratch: Vec::new(),
        }
    }

    /// Build index from vectors using provided coarse centroids and PQ codebook
    ///
    /// `doc_id_ordinals`: (doc_id, ordinal) pairs. If None, uses (index, 0).
    pub fn build(
        config: IVFPQConfig,
        coarse_centroids: &CoarseCentroids,
        codebook: &PQCodebook,
        vectors: &[Vec<f32>],
        doc_id_ordinals: Option<&[(u32, u16)]>,
    ) -> Self {
        let mut index = Self::new(config.clone(), coarse_centroids.version, codebook.version);

        for (i, vector) in vectors.iter().enumerate() {
            let (doc_id, ordinal) = doc_id_ordinals.map(|ids| ids[i]).unwrap_or((i as u32, 0));
            index.add_vector(coarse_centroids, codebook, doc_id, ordinal, vector);
        }

        index
    }

    /// Add a single vector to the index
    pub fn add_vector(
        &mut self,
        coarse_centroids: &CoarseCentroids,
        codebook: &PQCodebook,
        doc_id: u32,
        ordinal: u16,
        vector: &[f32],
    ) {
        // Get cluster assignment (with SOAR if configured)
        let assignment = coarse_centroids.assign_with_routing(vector, self.config.routing);

        // Add to primary cluster
        self.add_to_cluster(
            coarse_centroids,
            codebook,
            &assignment,
            doc_id,
            ordinal,
            vector,
        );

        // Add to secondary clusters (SOAR)
        for &cluster_id in &assignment.secondary_clusters {
            let secondary_assignment = MultiAssignment {
                primary_cluster: cluster_id,
                secondary_clusters: Vec::new(),
            };
            self.add_to_cluster(
                coarse_centroids,
                codebook,
                &secondary_assignment,
                doc_id,
                ordinal,
                vector,
            );
        }
    }

    /// Add one contiguous vector batch in parallel while preserving input
    /// order inside every leaf. Callers can install this work on their bounded
    /// Rayon pool; normal segment builds use the current Rayon pool.
    #[cfg(feature = "native")]
    pub fn add_vectors_parallel(
        &mut self,
        coarse_centroids: &CoarseCentroids,
        codebook: &PQCodebook,
        doc_id_ordinals: &[(u32, u16)],
        vectors: &[f32],
    ) -> Result<(), &'static str> {
        use rayon::prelude::*;

        let vector_count = doc_id_ordinals.len();
        let expected = vector_count
            .checked_mul(self.config.dim)
            .ok_or("IVF-PQ input size overflow")?;
        if vectors.len() != expected {
            return Err("IVF-PQ vector and label matrices are inconsistent");
        }
        if vector_count == 0 {
            return Ok(());
        }

        // A small multiple of the worker count supplies enough tasks for load
        // balancing without creating one hash map per vector. Indexed collect
        // retains chunk order, so merging the partials also retains the input
        // order within each leaf.
        let target_tasks = rayon::current_num_threads().saturating_mul(4).max(1);
        let chunk_vectors = vector_count.div_ceil(target_tasks).max(64);
        let config = self.config.clone();
        let centroids_version = self.centroids_version;
        let codebook_version = self.codebook_version;
        let partials: Vec<Self> = doc_id_ordinals
            .par_chunks(chunk_vectors)
            .enumerate()
            .map(|(chunk_index, labels)| {
                let first = chunk_index * chunk_vectors;
                let chunk = &vectors[first * config.dim..(first + labels.len()) * config.dim];
                let mut partial = Self::new(config.clone(), centroids_version, codebook_version);
                for (&(doc_id, ordinal), vector) in
                    labels.iter().zip(chunk.chunks_exact(config.dim))
                {
                    partial.add_vector(coarse_centroids, codebook, doc_id, ordinal, vector);
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
            || self.codebook_version != other.codebook_version
            || self.config != other.config
        {
            return Err("Cannot merge IVF-PQ payloads from different generations");
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
            .ok_or("IVF-PQ vector count overflow during parallel build")?;
        Ok(())
    }

    fn add_to_cluster(
        &mut self,
        coarse_centroids: &CoarseCentroids,
        codebook: &PQCodebook,
        assignment: &MultiAssignment,
        doc_id: u32,
        ordinal: u16,
        vector: &[f32],
    ) {
        let cluster_id = assignment.primary_cluster;
        let centroid = coarse_centroids.get_centroid(cluster_id);

        // Encode residual with PQ
        let cluster = self.clusters.entry(cluster_id).or_default();
        cluster.doc_ids.push(doc_id);
        cluster.ordinals.push(ordinal);
        codebook.encode_into(
            vector,
            Some(centroid),
            &mut cluster.codes,
            &mut self.residual_scratch,
            &mut self.rotated_scratch,
        );
        self.len += 1;
    }

    /// Search a probe plan already computed from the global quantizer.
    pub fn search_distinct_documents(
        &self,
        k: usize,
        plan: &IvfPqQueryPlan,
    ) -> Vec<(u32, u16, f32)> {
        let mut candidates = super::BoundedAnnCollector::<true, false>::new(k);
        self.visit_cluster_distances(plan, |doc_id, ordinal, distance| {
            candidates.insert(doc_id, ordinal, distance)
        });
        candidates.into_sorted_results()
    }

    fn visit_cluster_distances(&self, plan: &IvfPqQueryPlan, mut visit: impl FnMut(u32, u16, f32)) {
        for (&cluster_id, distance_table) in plan.cluster_ids.iter().zip(&plan.distance_tables) {
            if let Some(cluster) = self.clusters.get(&cluster_id) {
                // Score all vectors in cluster using ADC (Asymmetric Distance Computation)
                for (index, code) in cluster
                    .codes
                    .chunks_exact(self.config.code_size)
                    .enumerate()
                {
                    let dist = distance_table.compute_distance(code);
                    visit(cluster.doc_ids[index], cluster.ordinals[index], dist);
                }
            }
        }
    }

    /// Merge another index into this one (instance method)
    pub fn merge_into(
        &mut self,
        other: &IVFPQIndex,
        doc_id_offset: u32,
    ) -> Result<(), &'static str> {
        if self.centroids_version != other.centroids_version {
            return Err("Cannot merge indexes with different centroid versions");
        }
        if self.codebook_version != other.codebook_version {
            return Err("Cannot merge indexes with different codebook versions");
        }
        if self.config != other.config {
            return Err("Cannot merge IVF-PQ payloads with different configurations");
        }

        for (&cluster_id, source) in &other.clusters {
            self.clusters
                .entry(cluster_id)
                .or_default()
                .append(source, doc_id_offset);
        }
        self.len = self
            .len
            .checked_add(other.len)
            .ok_or("IVF-PQ vector count overflow during merge")?;
        Ok(())
    }

    /// Number of indexed vectors
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of non-empty clusters
    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Memory usage estimate
    pub fn size_bytes(&self) -> usize {
        self.clusters
            .values()
            .map(|cluster| {
                cluster.codes.len()
                    + cluster.doc_ids.len() * size_of::<u32>()
                    + cluster.ordinals.len() * size_of::<u16>()
            })
            .sum()
    }

    /// Estimated memory usage in bytes (alias for size_bytes)
    pub fn estimated_memory_bytes(&self) -> usize {
        self.size_bytes()
    }

    fn validate(&self) -> Result<(), String> {
        if self.config.dim == 0
            || self.config.code_size == 0
            || self.centroids_version == 0
            || self.codebook_version == 0
        {
            return Err("invalid IVF-PQ segment metadata".to_string());
        }
        let mut total = 0usize;
        for cluster in self.clusters.values() {
            if cluster.doc_ids.len() != cluster.ordinals.len()
                || cluster.codes.len()
                    != cluster
                        .doc_ids
                        .len()
                        .checked_mul(self.config.code_size)
                        .ok_or_else(|| "IVF-PQ code column size overflow".to_string())?
            {
                return Err("invalid IVF-PQ cluster columns or codes".to_string());
            }
            total = total
                .checked_add(cluster.doc_ids.len())
                .ok_or_else(|| "IVF-PQ vector count overflow".to_string())?;
        }
        if total != self.len {
            return Err("inconsistent IVF-PQ vector count".to_string());
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }

    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let index: Self = crate::structures::vector::decode_ann_bincode_exact(data, "IVF-PQ")?;
        index
            .validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        Ok(index)
    }

    /// Merge multiple compatible segment payloads into one.
    #[allow(clippy::should_implement_trait)]
    pub fn merge(indexes: &[&IVFPQIndex], doc_offsets: &[u32]) -> Result<Self, &'static str> {
        if indexes.is_empty() {
            return Err("Cannot merge empty list of indexes");
        }

        let first = indexes[0];
        let mut merged = Self::new(
            first.config.clone(),
            first.centroids_version,
            first.codebook_version,
        );

        for (idx, &index) in indexes.iter().enumerate() {
            let offset = doc_offsets.get(idx).copied().unwrap_or(0);
            merged.merge_into(index, offset)?;
        }

        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::vector::ivf::CoarseConfig;
    use crate::structures::vector::quantization::PQConfig;
    use rand::prelude::*;

    #[test]
    #[ignore] // Long-running test
    fn test_ivf_pq_basic() {
        let dim = 64;
        let n = 500;
        let num_clusters = 16;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        // Train coarse centroids
        let coarse_config = CoarseConfig::new(dim, num_clusters);
        let coarse_centroids = CoarseCentroids::train(&coarse_config, &vectors);

        // Train PQ codebook
        let pq_config = PQConfig::new(dim).with_opq(false, 0);
        let codebook = PQCodebook::train(pq_config, &vectors, 10);

        // Build index
        let config = IVFPQConfig::new(dim, codebook.config.num_subspaces);
        let index = IVFPQIndex::build(config, &coarse_centroids, &codebook, &vectors, None);

        assert_eq!(index.len(), n);
    }

    #[test]
    #[ignore] // Long-running test
    fn test_ivf_pq_search() {
        let dim = 32;
        let n = 200;
        let k = 10;
        let num_clusters = 8;

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let coarse_config = CoarseConfig::new(dim, num_clusters);
        let coarse_centroids = CoarseCentroids::train(&coarse_config, &vectors);

        let pq_config = PQConfig::new(dim).with_opq(false, 0);
        let codebook = PQCodebook::train(pq_config, &vectors, 10);

        let config = IVFPQConfig::new(dim, codebook.config.num_subspaces);
        let index = IVFPQIndex::build(config, &coarse_centroids, &codebook, &vectors, None);

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
        let plan = IvfPqQueryPlan::build(
            &coarse_centroids,
            &codebook,
            &query,
            4,
            IvfRoutingMode::Flat,
        );
        let results = index.search_distinct_documents(k, &plan);

        assert_eq!(results.len(), k);

        // Verify sorted by distance
        for i in 1..results.len() {
            assert!(results[i].2 >= results[i - 1].2);
        }
    }

    #[test]
    #[ignore] // Long-running test
    fn test_ivf_pq_recall() {
        let dim = 128;
        let n = 1000;
        let k = 10;
        let num_clusters = 32;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let coarse_config = CoarseConfig::new(dim, num_clusters);
        let coarse_centroids = CoarseCentroids::train(&coarse_config, &vectors);

        // Test with default PQ config (64 subspaces, 2 dims each)
        let pq_config = PQConfig::new(dim).with_opq(false, 0);
        let codebook = PQCodebook::train(pq_config, &vectors, 25);

        let config = IVFPQConfig::new(dim, codebook.config.num_subspaces);
        let index = IVFPQIndex::build(config, &coarse_centroids, &codebook, &vectors, None);

        // Run multiple queries and compute recall
        let num_queries = 50;
        let mut total_recall = 0.0f32;

        for _ in 0..num_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();

            // Exact k-NN
            let mut exact: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = query.iter().zip(v).map(|(&a, &b)| (a - b) * (a - b)).sum();
                    (i, d)
                })
                .collect();
            exact.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            let exact_top_k: std::collections::HashSet<usize> =
                exact[..k].iter().map(|(i, _)| *i).collect();

            // IVF-PQ search
            let plan = IvfPqQueryPlan::build(
                &coarse_centroids,
                &codebook,
                &query,
                32,
                IvfRoutingMode::Flat,
            );
            let results = index.search_distinct_documents(k, &plan);
            let pq_top_k: std::collections::HashSet<usize> =
                results.iter().map(|(i, _, _)| *i as usize).collect();

            let recall = exact_top_k.intersection(&pq_top_k).count() as f32 / k as f32;
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f32;
        println!("IVF-PQ Recall@{}: {:.1}%", k, avg_recall * 100.0);

        // With proper config, recall should be reasonable (>30%)
        assert!(
            avg_recall > 0.25,
            "IVF-PQ recall too low: {:.1}%",
            avg_recall * 100.0
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn parallel_batch_build_matches_sequential_multi_value_payload() {
        let dim = 8;
        let vector_count = 512;
        let vectors: Vec<Vec<f32>> = (0..vector_count)
            .map(|index| {
                (0..dim)
                    .map(|column| ((index * 31 + column * 17) as f32).sin())
                    .collect()
            })
            .collect();
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        let labels: Vec<(u32, u16)> = (0..vector_count)
            .map(|index| ((index / 3) as u32, (index % 3) as u16))
            .collect();
        let centroids = CoarseCentroids::train(&CoarseConfig::new(dim, 16), &vectors);
        let codebook = PQCodebook::train(PQConfig::new(dim).with_opq(false, 0), &vectors, 2);
        let config = IVFPQConfig::new(dim, codebook.config.num_subspaces);

        let mut sequential = IVFPQIndex::new(config.clone(), centroids.version, codebook.version);
        for (&(doc_id, ordinal), vector) in labels.iter().zip(vectors.iter()) {
            sequential.add_vector(&centroids, &codebook, doc_id, ordinal, vector);
        }

        let mut parallel = IVFPQIndex::new(config, centroids.version, codebook.version);
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                parallel
                    .add_vectors_parallel(&centroids, &codebook, &labels, &flat)
                    .unwrap();
            });

        assert_eq!(parallel.len(), sequential.len());
        assert_eq!(parallel.to_bytes().unwrap(), sequential.to_bytes().unwrap());
    }

    #[test]
    fn test_ivf_pq_merge() {
        let dim = 32;
        let n = 100;
        let num_clusters = 4;

        let mut rng = rand::rngs::StdRng::seed_from_u64(456);
        let vectors1: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();
        let vectors2: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        let coarse_config = CoarseConfig::new(dim, num_clusters);
        let coarse_centroids = CoarseCentroids::train(&coarse_config, &vectors1);

        let pq_config = PQConfig::new(dim).with_opq(false, 0);
        let codebook = PQCodebook::train(pq_config, &vectors1, 10);

        let config = IVFPQConfig::new(dim, codebook.config.num_subspaces);
        let mut index1 = IVFPQIndex::build(
            config.clone(),
            &coarse_centroids,
            &codebook,
            &vectors1,
            None,
        );
        let index2 = IVFPQIndex::build(config, &coarse_centroids, &codebook, &vectors2, None);

        index1.merge_into(&index2, n as u32).unwrap();

        assert_eq!(index1.len(), 2 * n);
        assert_eq!(
            index1.size_bytes(),
            index1.len() * (codebook.config.num_subspaces + size_of::<u32>() + size_of::<u16>())
        );
        assert!(index1.clusters.values().all(|cluster| {
            cluster.codes.len() == cluster.doc_ids.len() * codebook.config.num_subspaces
        }));

        let bytes = index1.to_bytes().unwrap();
        let decoded = IVFPQIndex::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.len(), index1.len());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(IVFPQIndex::from_bytes(&trailing).is_err());
    }
}
