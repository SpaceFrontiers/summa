//! Product Quantization with an Optimized Product Quantization rotation.
//!
//! Implementation inspired by Google's ScaNN (Scalable Nearest Neighbors):
//! - **OPQ rotation**: learns optimal rotation matrix before quantization
//! - **Product quantization** with learned codebooks
//! - **SIMD-accelerated** asymmetric distance computation

use serde::{Deserialize, Serialize};

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "x86_64", feature = "native"))]
#[allow(unused_imports)]
use std::arch::x86_64::*;

/// Default number of centroids per subspace (K) - must be 256 for u8 codes
pub const DEFAULT_NUM_CENTROIDS: usize = 256;

/// Configuration for residual Product Quantization with OPQ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Vector dimension
    pub dim: usize,
    /// Number of subspaces (M) - computed from dim / dims_per_block
    pub num_subspaces: usize,
    /// Dimensions per subspace block.
    pub dims_per_block: usize,
    /// Number of centroids per subspace (K) - typically 256 for u8 codes
    pub num_centroids: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Use OPQ rotation matrix (learned via SVD)
    pub use_opq: bool,
    /// Number of OPQ iterations
    pub opq_iters: usize,
}

impl PQConfig {
    /// Production profile: at most one byte per eight dimensions (96 bytes for
    /// 768D), with OPQ training. Raw vectors remain available for the bounded
    /// exact rerank stage.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "PQ dimension must be non-zero");
        let dims_per_block = (1..=8)
            .rev()
            .find(|candidate| dim.is_multiple_of(*candidate))
            .unwrap_or(1);
        Self {
            dim,
            num_subspaces: dim / dims_per_block,
            dims_per_block,
            num_centroids: DEFAULT_NUM_CENTROIDS,
            seed: 42,
            use_opq: true,
            opq_iters: 4,
        }
    }

    #[cfg(test)]
    pub fn with_opq(mut self, enabled: bool, iters: usize) -> Self {
        self.use_opq = enabled;
        self.opq_iters = iters;
        self
    }

    /// Dimension of each subspace
    pub fn subspace_dim(&self) -> usize {
        self.dims_per_block
    }
}

/// Learned codebook for Product Quantization with OPQ rotation
///
/// Trained once, shared across all segments (like CoarseCentroids).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCodebook {
    /// Configuration
    pub config: PQConfig,
    /// OPQ rotation matrix (dim × dim), stored row-major
    pub rotation_matrix: Option<Vec<f32>>,
    /// Centroids: M subspaces × K centroids × subspace_dim
    pub centroids: Vec<f32>,
    /// Version for merge compatibility checking
    pub version: u64,
}

impl PQCodebook {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let config = &self.config;
        if config.dim == 0
            || config.num_subspaces == 0
            || config.dims_per_block == 0
            || config.num_centroids == 0
            || config.num_centroids > 256
        {
            return Err("PQ codebook has invalid zero/unbounded dimensions".to_string());
        }
        let covered_dim = config
            .num_subspaces
            .checked_mul(config.dims_per_block)
            .ok_or_else(|| "PQ subspace dimension overflow".to_string())?;
        if covered_dim != config.dim {
            return Err(format!(
                "PQ subspaces cover {covered_dim} dimensions, expected {}",
                config.dim
            ));
        }
        let expected_centroids = config
            .num_subspaces
            .checked_mul(config.num_centroids)
            .and_then(|count| count.checked_mul(config.dims_per_block))
            .ok_or_else(|| "PQ centroid size overflow".to_string())?;
        if self.centroids.len() != expected_centroids
            || self.centroids.iter().any(|value| !value.is_finite())
        {
            return Err(format!(
                "PQ centroid table is invalid: got {}, expected {expected_centroids}",
                self.centroids.len()
            ));
        }
        if let Some(rotation) = &self.rotation_matrix {
            let expected = config
                .dim
                .checked_mul(config.dim)
                .ok_or_else(|| "PQ rotation size overflow".to_string())?;
            if rotation.len() != expected || rotation.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "PQ rotation matrix is invalid: got {}, expected {expected}",
                    rotation.len()
                ));
            }
        }
        if config.use_opq != self.rotation_matrix.is_some() {
            return Err("PQ OPQ configuration does not match its rotation matrix".to_string());
        }
        Ok(())
    }

    /// Train the global residual codebook and OPQ rotation.
    pub fn train(config: PQConfig, vectors: &[Vec<f32>], max_iters: usize) -> Self {
        #[cfg(not(feature = "native"))]
        let config = PQConfig {
            use_opq: false,
            opq_iters: 0,
            ..config
        };

        assert!(!vectors.is_empty(), "Cannot train on empty vector set");
        assert!(
            vectors
                .iter()
                .all(|vector| vector.len() == config.dim
                    && vector.iter().all(|value| value.is_finite())),
            "Vector dimension mismatch or non-finite training value"
        );

        let m = config.num_subspaces;
        let k = config.num_centroids;
        let sub_dim = config.subspace_dim();
        let n = vectors.len();

        // Step 1: Learn OPQ rotation matrix if enabled
        #[cfg(feature = "native")]
        let rotation_matrix = (config.use_opq && config.opq_iters > 0)
            .then(|| Self::learn_opq_rotation(&config, vectors, max_iters));
        #[cfg(not(feature = "native"))]
        let rotation_matrix: Option<Vec<f32>> = None;

        // Step 2: Apply rotation to vectors
        let rotated_vectors: Vec<Vec<f32>> = if let Some(ref r) = rotation_matrix {
            vectors
                .iter()
                .map(|v| Self::apply_rotation(r, v, config.dim))
                .collect()
        } else {
            vectors.to_vec()
        };

        // Step 3: Train k-means for each subspace
        let mut centroids = Vec::with_capacity(m * k * sub_dim);

        for subspace_idx in 0..m {
            let offset = subspace_idx * sub_dim;

            let subdata: Vec<f32> = rotated_vectors
                .iter()
                .flat_map(|v| v[offset..offset + sub_dim].iter().copied())
                .collect();

            let actual_k = k.min(n);

            let trained = crate::structures::vector::kmeans::train_euclidean_kmeans(
                &subdata,
                n,
                sub_dim,
                actual_k,
                max_iters,
                config.seed ^ (subspace_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            );
            centroids.extend(trained.centroids);

            // Pad if needed
            while centroids.len() < (subspace_idx + 1) * k * sub_dim {
                let last_start = centroids.len() - sub_dim;
                let last: Vec<f32> = centroids[last_start..].to_vec();
                centroids.extend(last);
            }
        }

        let version = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            config,
            rotation_matrix,
            centroids,
            version,
        }
    }

    /// Learn OPQ rotation matrix using SVD
    #[cfg(feature = "native")]
    fn learn_opq_rotation(config: &PQConfig, vectors: &[Vec<f32>], max_iters: usize) -> Vec<f32> {
        use nalgebra::DMatrix;

        let dim = config.dim;
        let n = vectors.len();

        let mut rotation = DMatrix::<f32>::identity(dim, dim);
        let data: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let x = DMatrix::from_row_slice(n, dim, &data);

        for _iter in 0..config.opq_iters.min(max_iters) {
            // Vectors are rows here while the runtime applies `R * x` to a
            // column vector, hence the transpose.
            let rotated = &x * rotation.transpose();
            let reconstructed = Self::reconstruct_pq(
                config,
                &rotated,
                5,
                config.seed ^ (_iter as u64).wrapping_mul(0x517c_c1b7_2722_0a95),
            );

            let xtx_hat = x.transpose() * &reconstructed;
            let svd = xtx_hat.svd(true, true);
            if let (Some(u), Some(vt)) = (svd.u, svd.v_t) {
                // Orthogonal Procrustes gives Q=U*V^T for X*Q ~= Y.
                // Runtime stores R=Q^T.
                rotation = vt.transpose() * u.transpose();
            }
        }

        let mut row_major = Vec::with_capacity(dim * dim);
        for row in 0..dim {
            for column in 0..dim {
                row_major.push(rotation[(row, column)]);
            }
        }
        row_major
    }

    #[cfg(feature = "native")]
    fn reconstruct_pq(
        config: &PQConfig,
        rotated: &nalgebra::DMatrix<f32>,
        iterations: usize,
        seed: u64,
    ) -> nalgebra::DMatrix<f32> {
        let m = config.num_subspaces;
        let k = config.num_centroids.min(rotated.nrows());
        let sub_dim = config.subspace_dim();
        let n = rotated.nrows();
        let mut reconstructed = nalgebra::DMatrix::<f32>::zeros(n, config.dim);

        for subspace_idx in 0..m {
            let mut subdata: Vec<f32> = Vec::with_capacity(n * sub_dim);
            for row in 0..n {
                for col in 0..sub_dim {
                    subdata.push(rotated[(row, subspace_idx * sub_dim + col)]);
                }
            }

            let trained = crate::structures::vector::kmeans::train_euclidean_kmeans(
                &subdata,
                n,
                sub_dim,
                k,
                iterations,
                seed ^ (subspace_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            );
            for (row, &assignment) in trained.assignments.iter().enumerate() {
                let centroid = &trained.centroids[assignment * sub_dim..(assignment + 1) * sub_dim];
                for (column, &value) in centroid.iter().enumerate() {
                    reconstructed[(row, subspace_idx * sub_dim + column)] = value;
                }
            }
        }
        reconstructed
    }

    /// Apply rotation matrix to vector (SIMD-accelerated dot product per row)
    fn apply_rotation(rotation: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; dim];
        for i in 0..dim {
            result[i] = crate::structures::simd::dot_product_f32(
                &rotation[i * dim..(i + 1) * dim],
                vector,
                dim,
            );
        }
        result
    }

    /// Find nearest centroid index
    fn find_nearest(centroids: &[f32], vector: &[f32], sub_dim: usize) -> usize {
        let num_centroids = centroids.len() / sub_dim;
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for c in 0..num_centroids {
            let offset = c * sub_dim;
            let dist: f32 = vector
                .iter()
                .zip(&centroids[offset..offset + sub_dim])
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();

            if dist < best_dist {
                best_dist = dist;
                best_idx = c;
            }
        }

        best_idx
    }

    /// Append one vector's PQ codes to a contiguous cluster column while
    /// reusing caller-owned rotation scratch across the entire segment build.
    pub(crate) fn encode_into(
        &self,
        vector: &[f32],
        centroid: Option<&[f32]>,
        codes: &mut Vec<u8>,
        residual: &mut Vec<f32>,
        rotated: &mut Vec<f32>,
    ) {
        let m = self.config.num_subspaces;
        let k = self.config.num_centroids;
        let sub_dim = self.config.subspace_dim();

        residual.clear();
        residual.reserve(self.config.dim);
        if let Some(centroid) = centroid {
            residual.extend(
                vector
                    .iter()
                    .zip(centroid)
                    .map(|(&value, &center)| value - center),
            );
        } else {
            residual.extend_from_slice(vector);
        }

        let vec_to_encode = if let Some(ref r) = self.rotation_matrix {
            rotated.clear();
            rotated.resize(self.config.dim, 0.0);
            for (row, output) in rotated.iter_mut().enumerate() {
                *output = crate::structures::simd::dot_product_f32(
                    &r[row * self.config.dim..(row + 1) * self.config.dim],
                    residual,
                    self.config.dim,
                );
            }
            rotated.as_slice()
        } else {
            residual.as_slice()
        };

        codes.reserve(m);

        for subspace_idx in 0..m {
            let vec_offset = subspace_idx * sub_dim;
            let subvec = &vec_to_encode[vec_offset..vec_offset + sub_dim];

            let centroid_base = subspace_idx * k * sub_dim;
            let centroids_slice = &self.centroids[centroid_base..centroid_base + k * sub_dim];

            let nearest = Self::find_nearest(centroids_slice, subvec, sub_dim);
            codes.push(nearest as u8);
        }
    }

    /// Encode one vector into an owned code for diagnostics and external use.
    /// Segment payload construction uses [`Self::encode_into`] to avoid one
    /// allocation per indexed vector.
    pub fn encode(&self, vector: &[f32], centroid: Option<&[f32]>) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.config.num_subspaces);
        self.encode_into(
            vector,
            centroid,
            &mut codes,
            &mut Vec::new(),
            &mut Vec::new(),
        );
        codes
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let m = self.config.num_subspaces;
        let k = self.config.num_centroids;
        let sub_dim = self.config.subspace_dim();

        let mut rotated_vector = Vec::with_capacity(self.config.dim);

        for (subspace_idx, &code) in codes.iter().enumerate().take(m) {
            let centroid_base = subspace_idx * k * sub_dim;
            let centroid_offset = centroid_base + (code as usize) * sub_dim;
            rotated_vector
                .extend_from_slice(&self.centroids[centroid_offset..centroid_offset + sub_dim]);
        }

        // Apply inverse rotation if present
        if let Some(ref r) = self.rotation_matrix {
            Self::apply_rotation_transpose(r, &rotated_vector, self.config.dim)
        } else {
            rotated_vector
        }
    }

    /// Apply transpose of rotation matrix
    fn apply_rotation_transpose(rotation: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; dim];
        for i in 0..dim {
            for j in 0..dim {
                result[i] += rotation[j * dim + i] * vector[j];
            }
        }
        result
    }

    /// Get centroid for a specific subspace and code
    #[inline]
    pub fn get_centroid(&self, subspace_idx: usize, code: u8) -> &[f32] {
        let k = self.config.num_centroids;
        let sub_dim = self.config.subspace_dim();
        let offset = subspace_idx * k * sub_dim + (code as usize) * sub_dim;
        &self.centroids[offset..offset + sub_dim]
    }

    /// Rotate a query vector
    pub fn rotate_query(&self, query: &[f32]) -> Vec<f32> {
        if let Some(ref r) = self.rotation_matrix {
            Self::apply_rotation(r, query, self.config.dim)
        } else {
            query.to_vec()
        }
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        let centroids_size = self.centroids.len() * 4;
        let rotation_size = self
            .rotation_matrix
            .as_ref()
            .map(|r| r.len() * 4)
            .unwrap_or(0);
        centroids_size + rotation_size + 64
    }

    /// Visit the small, index-global tables required to build every IVF-PQ
    /// query plan and distance table.
    pub(crate) fn visit_resident_regions(&self, visit: &mut dyn FnMut(&'static str, &[u8])) {
        if let Some(rotation) = &self.rotation_matrix {
            visit(
                "PQ rotation matrix",
                crate::structures::vector::ivf::routing::bytes_of_slice(rotation),
            );
        }
        visit(
            "PQ centroid table",
            crate::structures::vector::ivf::routing::bytes_of_slice(&self.centroids),
        );
    }
}

/// Precomputed distance table for fast asymmetric distance computation
#[derive(Debug, Clone)]
pub struct DistanceTable {
    /// M × K table of squared distances
    pub distances: Vec<f32>,
    /// Number of subspaces
    pub num_subspaces: usize,
    /// Number of centroids per subspace
    pub num_centroids: usize,
}

impl DistanceTable {
    /// Build distance table for a query vector
    pub fn build(codebook: &PQCodebook, query: &[f32], centroid: Option<&[f32]>) -> Self {
        let m = codebook.config.num_subspaces;
        let k = codebook.config.num_centroids;
        let sub_dim = codebook.config.subspace_dim();

        // Compute residual if centroid provided
        let residual: Vec<f32> = if let Some(c) = centroid {
            query.iter().zip(c).map(|(&v, &c)| v - c).collect()
        } else {
            query.to_vec()
        };

        // Apply rotation if present
        let rotated_query = codebook.rotate_query(&residual);

        let mut distances = Vec::with_capacity(m * k);

        for subspace_idx in 0..m {
            let query_offset = subspace_idx * sub_dim;
            let query_sub = &rotated_query[query_offset..query_offset + sub_dim];

            let centroid_base = subspace_idx * k * sub_dim;

            for centroid_idx in 0..k {
                let centroid_offset = centroid_base + centroid_idx * sub_dim;
                let centroid = &codebook.centroids[centroid_offset..centroid_offset + sub_dim];

                let dist: f32 = query_sub
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();

                distances.push(dist);
            }
        }

        Self {
            distances,
            num_subspaces: m,
            num_centroids: k,
        }
    }

    /// Compute approximate distance using PQ codes
    #[inline]
    pub fn compute_distance(&self, codes: &[u8]) -> f32 {
        let k = self.num_centroids;
        let mut total = 0.0f32;

        for (subspace_idx, &code) in codes.iter().enumerate() {
            let table_offset = subspace_idx * k + code as usize;
            total += self.distances[table_offset];
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_pq_config() {
        let config = PQConfig::new(128);
        assert_eq!(config.dim, 128);
        assert_eq!(config.dims_per_block, 8);
        assert_eq!(config.num_subspaces, 16);
    }

    #[test]
    fn test_pq_encode_decode() {
        let dim = 32;
        let config = PQConfig::new(dim).with_opq(false, 0);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let codebook = PQCodebook::train(config, &vectors, 10);

        let test_vec: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
        let code = codebook.encode(&test_vec, None);

        assert_eq!(code.len(), 4); // 32 dims / 8 dims_per_block
    }

    #[test]
    fn test_distance_table() {
        let dim = 16;
        let config = PQConfig::new(dim).with_opq(false, 0);

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        let codebook = PQCodebook::train(config, &vectors, 5);

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let table = DistanceTable::build(&codebook, &query, None);

        let code = codebook.encode(&vectors[0], None);
        let dist = table.compute_distance(&code);

        assert!(dist >= 0.0);
    }
}
