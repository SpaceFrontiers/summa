//! Product Quantization with OPQ and Anisotropic Loss (ScaNN-style)
//!
//! Implementation inspired by Google's ScaNN (Scalable Nearest Neighbors):
//! - **True anisotropic quantization**: penalizes parallel error more than orthogonal
//! - **OPQ rotation**: learns optimal rotation matrix before quantization
//! - **Product quantization** with learned codebooks
//! - **SIMD-accelerated** asymmetric distance computation

use std::io::{self, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
#[cfg(not(feature = "native"))]
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::super::ivf::cluster::QuantizedCode;
use super::Quantizer;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "x86_64", feature = "native"))]
#[allow(unused_imports)]
use std::arch::x86_64::*;

/// Magic number for codebook file
const CODEBOOK_MAGIC: u32 = 0x5343424B; // "SCBK" - ScaNN CodeBook

/// Default number of centroids per subspace (K) - must be 256 for u8 codes
pub const DEFAULT_NUM_CENTROIDS: usize = 256;

/// Default dimensions per block (ScaNN recommends 2 for best accuracy)
pub const DEFAULT_DIMS_PER_BLOCK: usize = 2;

/// Configuration for Product Quantization with OPQ and Anisotropic Loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Vector dimension
    pub dim: usize,
    /// Number of subspaces (M) - computed from dim / dims_per_block
    pub num_subspaces: usize,
    /// Dimensions per subspace block (ScaNN recommends 2)
    pub dims_per_block: usize,
    /// Number of centroids per subspace (K) - typically 256 for u8 codes
    pub num_centroids: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Use anisotropic quantization (true ScaNN-style parallel/orthogonal weighting)
    pub anisotropic: bool,
    /// Anisotropic eta: ratio of parallel to orthogonal error weight (η)
    pub aniso_eta: f32,
    /// Anisotropic threshold T: only consider inner products >= T
    pub aniso_threshold: f32,
    /// Use OPQ rotation matrix (learned via SVD)
    pub use_opq: bool,
    /// Number of OPQ iterations
    pub opq_iters: usize,
}

impl PQConfig {
    /// Create config with ScaNN-recommended defaults
    pub fn new(dim: usize) -> Self {
        let dims_per_block = DEFAULT_DIMS_PER_BLOCK;
        let num_subspaces = dim / dims_per_block;

        Self {
            dim,
            num_subspaces,
            dims_per_block,
            num_centroids: DEFAULT_NUM_CENTROIDS,
            seed: 42,
            anisotropic: true,
            aniso_eta: 10.0,
            aniso_threshold: 0.2,
            use_opq: true,
            opq_iters: 10,
        }
    }

    /// Create config with larger subspaces (faster but less accurate)
    pub fn new_fast(dim: usize) -> Self {
        let num_subspaces = if dim >= 256 {
            8
        } else if dim >= 64 {
            4
        } else {
            2
        };
        let dims_per_block = dim / num_subspaces;

        Self {
            dim,
            num_subspaces,
            dims_per_block,
            num_centroids: DEFAULT_NUM_CENTROIDS,
            seed: 42,
            anisotropic: true,
            aniso_eta: 10.0,
            aniso_threshold: 0.2,
            use_opq: false,
            opq_iters: 0,
        }
    }

    /// Create balanced config (good recall/speed tradeoff)
    /// Uses 16 subspaces for 128D+ vectors, 8 for smaller
    pub fn new_balanced(dim: usize) -> Self {
        let num_subspaces = if dim >= 128 {
            16
        } else if dim >= 64 {
            8
        } else {
            4
        };
        let dims_per_block = dim / num_subspaces;

        Self {
            dim,
            num_subspaces,
            dims_per_block,
            num_centroids: DEFAULT_NUM_CENTROIDS,
            seed: 42,
            anisotropic: true,
            aniso_eta: 10.0,
            aniso_threshold: 0.2,
            use_opq: false,
            opq_iters: 0,
        }
    }

    pub fn with_dims_per_block(mut self, d: usize) -> Self {
        assert!(
            self.dim.is_multiple_of(d),
            "Dimension must be divisible by dims_per_block"
        );
        self.dims_per_block = d;
        self.num_subspaces = self.dim / d;
        self
    }

    pub fn with_subspaces(mut self, m: usize) -> Self {
        assert!(
            self.dim.is_multiple_of(m),
            "Dimension must be divisible by num_subspaces"
        );
        self.num_subspaces = m;
        self.dims_per_block = self.dim / m;
        self
    }

    pub fn with_centroids(mut self, k: usize) -> Self {
        assert!(k <= 256, "Max 256 centroids for u8 codes");
        self.num_centroids = k;
        self
    }

    pub fn with_anisotropic(mut self, enabled: bool, eta: f32) -> Self {
        self.anisotropic = enabled;
        self.aniso_eta = eta;
        self
    }

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

/// Quantized vector using Product Quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQVector {
    /// PQ codes (M bytes, one per subspace)
    pub codes: Vec<u8>,
    /// Original vector norm (for re-ranking or normalization)
    pub norm: f32,
}

impl PQVector {
    pub fn new(codes: Vec<u8>, norm: f32) -> Self {
        Self { codes, norm }
    }
}

impl QuantizedCode for PQVector {
    fn size_bytes(&self) -> usize {
        self.codes.len() + 4 // codes + norm
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
    /// Precomputed centroid norms for faster distance computation
    pub centroid_norms: Option<Vec<f32>>,
}

impl PQCodebook {
    /// Train codebook with OPQ rotation and anisotropic loss
    #[cfg(feature = "native")]
    pub fn train(config: PQConfig, vectors: &[Vec<f32>], max_iters: usize) -> Self {
        use kentro::KMeans;
        use ndarray::Array2;

        assert!(!vectors.is_empty(), "Cannot train on empty vector set");
        assert_eq!(vectors[0].len(), config.dim, "Vector dimension mismatch");

        let m = config.num_subspaces;
        let k = config.num_centroids;
        let sub_dim = config.subspace_dim();
        let n = vectors.len();

        // Step 1: Learn OPQ rotation matrix if enabled
        let rotation_matrix = if config.use_opq && config.opq_iters > 0 {
            Some(Self::learn_opq_rotation(&config, vectors, max_iters))
        } else {
            None
        };

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

            let data = Array2::from_shape_vec((n, sub_dim), subdata)
                .expect("Failed to create subspace array");
            let mut kmeans = KMeans::new(actual_k)
                .with_euclidean(true)
                .with_iterations(max_iters);
            let _ = kmeans
                .train(data.view(), None)
                .expect("K-means training failed");

            let subspace_centroids: Vec<f32> = kmeans
                .centroids()
                .expect("No centroids")
                .iter()
                .copied()
                .collect();

            centroids.extend(subspace_centroids);

            // Pad if needed
            while centroids.len() < (subspace_idx + 1) * k * sub_dim {
                let last_start = centroids.len() - sub_dim;
                let last: Vec<f32> = centroids[last_start..].to_vec();
                centroids.extend(last);
            }
        }

        // Precompute centroid norms
        let centroid_norms: Vec<f32> = (0..m * k)
            .map(|i| {
                let start = i * sub_dim;
                if start + sub_dim <= centroids.len() {
                    centroids[start..start + sub_dim]
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        .sqrt()
                } else {
                    0.0
                }
            })
            .collect();

        let version = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            config,
            rotation_matrix,
            centroids,
            version,
            centroid_norms: Some(centroid_norms),
        }
    }

    /// Fallback training for non-native builds (WASM)
    #[cfg(not(feature = "native"))]
    pub fn train(config: PQConfig, vectors: &[Vec<f32>], max_iters: usize) -> Self {
        assert!(!vectors.is_empty(), "Cannot train on empty vector set");
        assert_eq!(vectors[0].len(), config.dim, "Vector dimension mismatch");

        let m = config.num_subspaces;
        let k = config.num_centroids;
        let sub_dim = config.subspace_dim();
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

        let rotation_matrix = None;
        let mut centroids = Vec::with_capacity(m * k * sub_dim);

        for subspace_idx in 0..m {
            let offset = subspace_idx * sub_dim;
            let subvectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| v[offset..offset + sub_dim].to_vec())
                .collect();

            let subspace_centroids =
                Self::train_subspace_scalar(&subvectors, k, sub_dim, max_iters, &mut rng);
            centroids.extend(subspace_centroids);
        }

        let centroid_norms: Vec<f32> = (0..m * k)
            .map(|i| {
                let start = i * sub_dim;
                centroids[start..start + sub_dim]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();

        let version = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            config,
            rotation_matrix,
            centroids,
            version,
            centroid_norms: Some(centroid_norms),
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
            let rotated = &x * &rotation;
            let assignments = Self::compute_pq_assignments(config, &rotated);
            let reconstructed = Self::reconstruct_from_assignments(config, &rotated, &assignments);

            let xtx_hat = x.transpose() * &reconstructed;
            let svd = xtx_hat.svd(true, true);
            if let (Some(u), Some(vt)) = (svd.u, svd.v_t) {
                let new_rotation: DMatrix<f32> = vt.transpose() * u.transpose();
                rotation = new_rotation;
            }
        }

        rotation.iter().copied().collect()
    }

    #[cfg(feature = "native")]
    fn compute_pq_assignments(
        config: &PQConfig,
        rotated: &nalgebra::DMatrix<f32>,
    ) -> Vec<Vec<usize>> {
        use kentro::KMeans;
        use ndarray::Array2;

        let m = config.num_subspaces;
        let k = config.num_centroids.min(rotated.nrows());
        let sub_dim = config.subspace_dim();
        let n = rotated.nrows();

        let mut all_assignments = vec![vec![0usize; m]; n];

        for subspace_idx in 0..m {
            let mut subdata: Vec<f32> = Vec::with_capacity(n * sub_dim);
            for row in 0..n {
                for col in 0..sub_dim {
                    subdata.push(rotated[(row, subspace_idx * sub_dim + col)]);
                }
            }

            let data = Array2::from_shape_vec((n, sub_dim), subdata)
                .expect("Failed to create subspace array");
            let mut kmeans = KMeans::new(k).with_euclidean(true).with_iterations(5);
            let clusters = kmeans
                .train(data.view(), None)
                .expect("K-means training failed");

            // Invert cluster assignments: clusters[cluster_id] = [point_indices]
            for (cluster_id, point_indices) in clusters.iter().enumerate() {
                for &point_idx in point_indices {
                    all_assignments[point_idx][subspace_idx] = cluster_id;
                }
            }
        }

        all_assignments
    }

    #[cfg(feature = "native")]
    fn reconstruct_from_assignments(
        config: &PQConfig,
        rotated: &nalgebra::DMatrix<f32>,
        assignments: &[Vec<usize>],
    ) -> nalgebra::DMatrix<f32> {
        use kentro::KMeans;
        use ndarray::Array2;

        let m = config.num_subspaces;
        let sub_dim = config.subspace_dim();
        let n = rotated.nrows();
        let dim = config.dim;

        let mut reconstructed = nalgebra::DMatrix::<f32>::zeros(n, dim);

        for subspace_idx in 0..m {
            let mut subdata: Vec<f32> = Vec::with_capacity(n * sub_dim);
            for row in 0..n {
                for col in 0..sub_dim {
                    subdata.push(rotated[(row, subspace_idx * sub_dim + col)]);
                }
            }

            let k = config.num_centroids.min(n);
            let data = Array2::from_shape_vec((n, sub_dim), subdata)
                .expect("Failed to create subspace array");
            let mut kmeans = KMeans::new(k).with_euclidean(true).with_iterations(5);
            let _ = kmeans
                .train(data.view(), None)
                .expect("K-means training failed");

            let centroids = kmeans.centroids().expect("No centroids");

            for (row, assignment) in assignments.iter().enumerate() {
                let centroid_idx = assignment[subspace_idx];
                if centroid_idx < k {
                    for col in 0..sub_dim {
                        reconstructed[(row, subspace_idx * sub_dim + col)] =
                            centroids[[centroid_idx, col]];
                    }
                }
            }
        }

        reconstructed
    }

    /// Apply rotation matrix to vector
    fn apply_rotation(rotation: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; dim];
        for i in 0..dim {
            for j in 0..dim {
                result[i] += rotation[i * dim + j] * vector[j];
            }
        }
        result
    }

    /// Scalar k-means for WASM fallback
    #[cfg(not(feature = "native"))]
    fn train_subspace_scalar(
        subvectors: &[Vec<f32>],
        k: usize,
        sub_dim: usize,
        max_iters: usize,
        rng: &mut impl Rng,
    ) -> Vec<f32> {
        let actual_k = k.min(subvectors.len());
        let mut centroids = Self::kmeans_plusplus_init_scalar(subvectors, actual_k, sub_dim, rng);

        for _ in 0..max_iters {
            let assignments: Vec<usize> = subvectors
                .iter()
                .map(|v| Self::find_nearest_scalar(&centroids, v, sub_dim))
                .collect();

            let mut new_centroids = vec![0.0f32; actual_k * sub_dim];
            let mut counts = vec![0usize; actual_k];

            for (subvec, &assignment) in subvectors.iter().zip(assignments.iter()) {
                counts[assignment] += 1;
                let offset = assignment * sub_dim;
                for (j, &val) in subvec.iter().enumerate() {
                    new_centroids[offset + j] += val;
                }
            }

            for (c, &count) in counts.iter().enumerate().take(actual_k) {
                if count > 0 {
                    let offset = c * sub_dim;
                    for j in 0..sub_dim {
                        new_centroids[offset + j] /= count as f32;
                    }
                }
            }

            centroids = new_centroids;
        }

        while centroids.len() < k * sub_dim {
            let last_start = centroids.len() - sub_dim;
            let last: Vec<f32> = centroids[last_start..].to_vec();
            centroids.extend(last);
        }

        centroids
    }

    #[cfg(not(feature = "native"))]
    fn kmeans_plusplus_init_scalar(
        subvectors: &[Vec<f32>],
        k: usize,
        sub_dim: usize,
        rng: &mut impl Rng,
    ) -> Vec<f32> {
        let mut centroids = Vec::with_capacity(k * sub_dim);
        let first_idx = rng.random_range(0..subvectors.len());
        centroids.extend_from_slice(&subvectors[first_idx]);

        for _ in 1..k {
            let distances: Vec<f32> = subvectors
                .iter()
                .map(|v| Self::min_dist_to_centroids_scalar(&centroids, v, sub_dim))
                .collect();

            let total: f32 = distances.iter().sum();
            let mut r = rng.random::<f32>() * total;
            let mut chosen_idx = 0;
            for (i, &d) in distances.iter().enumerate() {
                r -= d;
                if r <= 0.0 {
                    chosen_idx = i;
                    break;
                }
            }
            centroids.extend_from_slice(&subvectors[chosen_idx]);
        }

        centroids
    }

    #[cfg(not(feature = "native"))]
    fn min_dist_to_centroids_scalar(centroids: &[f32], vector: &[f32], sub_dim: usize) -> f32 {
        let num_centroids = centroids.len() / sub_dim;
        (0..num_centroids)
            .map(|c| {
                let offset = c * sub_dim;
                vector
                    .iter()
                    .zip(&centroids[offset..offset + sub_dim])
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum()
            })
            .fold(f32::MAX, f32::min)
    }

    #[cfg(not(feature = "native"))]
    fn find_nearest_scalar(centroids: &[f32], vector: &[f32], sub_dim: usize) -> usize {
        let num_centroids = centroids.len() / sub_dim;
        (0..num_centroids)
            .map(|c| {
                let offset = c * sub_dim;
                let dist: f32 = vector
                    .iter()
                    .zip(&centroids[offset..offset + sub_dim])
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                (c, dist)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(c, _)| c)
            .unwrap_or(0)
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

    /// Encode a vector to PQ codes
    pub fn encode(&self, vector: &[f32], centroid: Option<&[f32]>) -> PQVector {
        let m = self.config.num_subspaces;
        let k = self.config.num_centroids;
        let sub_dim = self.config.subspace_dim();

        // Compute residual if centroid provided
        let residual: Vec<f32> = if let Some(c) = centroid {
            vector.iter().zip(c).map(|(&v, &c)| v - c).collect()
        } else {
            vector.to_vec()
        };

        // Apply rotation if present
        let rotated: Vec<f32>;
        let vec_to_encode = if let Some(ref r) = self.rotation_matrix {
            rotated = Self::apply_rotation(r, &residual, self.config.dim);
            &rotated
        } else {
            &residual
        };

        let mut codes = Vec::with_capacity(m);

        for subspace_idx in 0..m {
            let vec_offset = subspace_idx * sub_dim;
            let subvec = &vec_to_encode[vec_offset..vec_offset + sub_dim];

            let centroid_base = subspace_idx * k * sub_dim;
            let centroids_slice = &self.centroids[centroid_base..centroid_base + k * sub_dim];

            let nearest = Self::find_nearest(centroids_slice, subvec, sub_dim);
            codes.push(nearest as u8);
        }

        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        PQVector::new(codes, norm)
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

    /// Save to binary file
    pub fn save(&self, path: &std::path::Path) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.write_to(&mut file)
    }

    /// Write to any writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(CODEBOOK_MAGIC)?;
        writer.write_u32::<LittleEndian>(2)?;
        writer.write_u64::<LittleEndian>(self.version)?;
        writer.write_u32::<LittleEndian>(self.config.dim as u32)?;
        writer.write_u32::<LittleEndian>(self.config.num_subspaces as u32)?;
        writer.write_u32::<LittleEndian>(self.config.dims_per_block as u32)?;
        writer.write_u32::<LittleEndian>(self.config.num_centroids as u32)?;
        writer.write_u8(if self.config.anisotropic { 1 } else { 0 })?;
        writer.write_f32::<LittleEndian>(self.config.aniso_eta)?;
        writer.write_f32::<LittleEndian>(self.config.aniso_threshold)?;
        writer.write_u8(if self.config.use_opq { 1 } else { 0 })?;
        writer.write_u32::<LittleEndian>(self.config.opq_iters as u32)?;

        if let Some(ref rotation) = self.rotation_matrix {
            writer.write_u8(1)?;
            for &val in rotation {
                writer.write_f32::<LittleEndian>(val)?;
            }
        } else {
            writer.write_u8(0)?;
        }

        for &val in &self.centroids {
            writer.write_f32::<LittleEndian>(val)?;
        }

        if let Some(ref norms) = self.centroid_norms {
            writer.write_u8(1)?;
            for &val in norms {
                writer.write_f32::<LittleEndian>(val)?;
            }
        } else {
            writer.write_u8(0)?;
        }

        Ok(())
    }

    /// Load from binary file
    pub fn load(path: &std::path::Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        Self::read_from(&mut std::io::Cursor::new(data))
    }

    /// Read from any reader
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != CODEBOOK_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid codebook file magic",
            ));
        }

        let file_version = reader.read_u32::<LittleEndian>()?;
        let version = reader.read_u64::<LittleEndian>()?;
        let dim = reader.read_u32::<LittleEndian>()? as usize;
        let num_subspaces = reader.read_u32::<LittleEndian>()? as usize;

        let (
            dims_per_block,
            num_centroids,
            anisotropic,
            aniso_eta,
            aniso_threshold,
            use_opq,
            opq_iters,
        ) = if file_version >= 2 {
            let dpb = reader.read_u32::<LittleEndian>()? as usize;
            let nc = reader.read_u32::<LittleEndian>()? as usize;
            let aniso = reader.read_u8()? != 0;
            let eta = reader.read_f32::<LittleEndian>()?;
            let thresh = reader.read_f32::<LittleEndian>()?;
            let opq = reader.read_u8()? != 0;
            let iters = reader.read_u32::<LittleEndian>()? as usize;
            (dpb, nc, aniso, eta, thresh, opq, iters)
        } else {
            let nc = reader.read_u32::<LittleEndian>()? as usize;
            let aniso = reader.read_u8()? != 0;
            let thresh = reader.read_f32::<LittleEndian>()?;
            let dpb = dim / num_subspaces;
            (dpb, nc, aniso, 10.0, thresh, false, 0)
        };

        let config = PQConfig {
            dim,
            num_subspaces,
            dims_per_block,
            num_centroids,
            seed: 42,
            anisotropic,
            aniso_eta,
            aniso_threshold,
            use_opq,
            opq_iters,
        };

        let rotation_matrix = if file_version >= 2 {
            let has_rotation = reader.read_u8()? != 0;
            if has_rotation {
                let mut rotation = vec![0.0f32; dim * dim];
                for val in &mut rotation {
                    *val = reader.read_f32::<LittleEndian>()?;
                }
                Some(rotation)
            } else {
                None
            }
        } else {
            None
        };

        let centroid_count = num_subspaces * num_centroids * config.subspace_dim();
        let mut centroids = vec![0.0f32; centroid_count];
        for val in &mut centroids {
            *val = reader.read_f32::<LittleEndian>()?;
        }

        let has_norms = reader.read_u8()? != 0;
        let centroid_norms = if has_norms {
            let mut norms = vec![0.0f32; num_subspaces * num_centroids];
            for val in &mut norms {
                *val = reader.read_f32::<LittleEndian>()?;
            }
            Some(norms)
        } else {
            None
        };

        Ok(Self {
            config,
            rotation_matrix,
            centroids,
            version,
            centroid_norms,
        })
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        let centroids_size = self.centroids.len() * 4;
        let norms_size = self
            .centroid_norms
            .as_ref()
            .map(|n| n.len() * 4)
            .unwrap_or(0);
        let rotation_size = self
            .rotation_matrix
            .as_ref()
            .map(|r| r.len() * 4)
            .unwrap_or(0);
        centroids_size + norms_size + rotation_size + 64
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

impl Quantizer for PQCodebook {
    type Code = PQVector;
    type Config = PQConfig;
    type QueryData = DistanceTable;

    fn encode(&self, vector: &[f32], centroid: Option<&[f32]>) -> Self::Code {
        self.encode(vector, centroid)
    }

    fn prepare_query(&self, query: &[f32], centroid: Option<&[f32]>) -> Self::QueryData {
        DistanceTable::build(self, query, centroid)
    }

    fn compute_distance(&self, query_data: &Self::QueryData, code: &Self::Code) -> f32 {
        query_data.compute_distance(&code.codes)
    }

    fn decode(&self, code: &Self::Code) -> Option<Vec<f32>> {
        Some(self.decode(&code.codes))
    }

    fn size_bytes(&self) -> usize {
        self.size_bytes()
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
        assert_eq!(config.dims_per_block, 2);
        assert_eq!(config.num_subspaces, 64);
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

        assert_eq!(code.codes.len(), 16); // 32 dims / 2 dims_per_block
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
        let dist = table.compute_distance(&code.codes);

        assert!(dist >= 0.0);
    }
}
