//! Coarse centroids for IVF partitioning
//!
//! Provides k-means clustering for the first level of IVF indexing.
//! Trained once, shared across all segments for O(1) merge compatibility.

use std::io::{self, Cursor, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
#[cfg(not(feature = "native"))]
use rand::SeedableRng;
#[cfg(not(feature = "native"))]
use rand::prelude::SliceRandom;
use serde::{Deserialize, Serialize};

use super::soar::{MultiAssignment, SoarConfig};

/// Magic number for coarse centroids file
const CENTROIDS_MAGIC: u32 = 0x48435643; // "CVCH" - Coarse Vector Centroids Hermes

/// Configuration for coarse quantizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoarseConfig {
    /// Number of clusters
    pub num_clusters: usize,
    /// Vector dimension
    pub dim: usize,
    /// Maximum k-means iterations
    pub max_iters: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// SOAR configuration (optional)
    pub soar: Option<SoarConfig>,
}

impl CoarseConfig {
    pub fn new(dim: usize, num_clusters: usize) -> Self {
        Self {
            num_clusters,
            dim,
            max_iters: 25,
            seed: 42,
            soar: None,
        }
    }

    pub fn with_soar(mut self, config: SoarConfig) -> Self {
        self.soar = Some(config);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_max_iters(mut self, iters: usize) -> Self {
        self.max_iters = iters;
        self
    }
}

/// Coarse centroids for IVF - trained once, shared across all segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoarseCentroids {
    /// Number of clusters
    pub num_clusters: u32,
    /// Vector dimension
    pub dim: usize,
    /// Centroids stored as flat array (num_clusters × dim)
    pub centroids: Vec<f32>,
    /// Version for compatibility checking during merge
    pub version: u64,
    /// SOAR configuration (if enabled)
    pub soar_config: Option<SoarConfig>,
}

impl CoarseCentroids {
    /// Train coarse centroids using k-means algorithm
    ///
    /// Uses kmeans crate with SIMD acceleration (native feature).
    #[cfg(feature = "native")]
    pub fn train(config: &CoarseConfig, vectors: &[Vec<f32>]) -> Self {
        use kmeans::{EuclideanDistance, KMeans, KMeansConfig};

        assert!(!vectors.is_empty(), "Cannot train on empty vector set");
        assert!(config.num_clusters > 0, "Need at least 1 cluster");

        let actual_clusters = config.num_clusters.min(vectors.len());
        let dim = config.dim;

        // Flatten vectors for kmeans crate (expects flat slice)
        let samples: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        // Run k-means with k-means++ initialization
        // KMeans<f32, 8, _> uses 8-lane SIMD (AVX256)
        let kmean: KMeans<f32, 8, _> = KMeans::new(&samples, vectors.len(), dim, EuclideanDistance);
        let result = kmean.kmeans_lloyd(
            actual_clusters,
            config.max_iters,
            KMeans::init_kmeanplusplus,
            &KMeansConfig::default(),
        );

        // Extract centroids from StrideBuffer to flat Vec
        let centroids: Vec<f32> = result
            .centroids
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect();

        let version = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            num_clusters: actual_clusters as u32,
            dim,
            centroids,
            version,
            soar_config: config.soar.clone(),
        }
    }

    /// Fallback k-means for non-native builds (WASM)
    #[cfg(not(feature = "native"))]
    pub fn train(config: &CoarseConfig, vectors: &[Vec<f32>]) -> Self {
        assert!(!vectors.is_empty(), "Cannot train on empty vector set");
        assert!(config.num_clusters > 0, "Need at least 1 cluster");

        let actual_clusters = config.num_clusters.min(vectors.len());
        let dim = config.dim;
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

        // Simple random initialization
        let mut indices: Vec<usize> = (0..vectors.len()).collect();
        indices.shuffle(&mut rng);

        let mut centroids: Vec<f32> = indices[..actual_clusters]
            .iter()
            .flat_map(|&i| vectors[i].iter().copied())
            .collect();

        // K-means iterations
        for _ in 0..config.max_iters {
            let assignments: Vec<usize> = vectors
                .iter()
                .map(|v| Self::find_nearest_idx_static(v, &centroids, dim))
                .collect();

            let mut new_centroids = vec![0.0f32; actual_clusters * dim];
            let mut counts = vec![0usize; actual_clusters];

            for (vec_idx, &cluster_id) in assignments.iter().enumerate() {
                counts[cluster_id] += 1;
                let offset = cluster_id * dim;
                for (i, &val) in vectors[vec_idx].iter().enumerate() {
                    new_centroids[offset + i] += val;
                }
            }

            for (cluster_id, &count) in counts.iter().enumerate().take(actual_clusters) {
                if count > 0 {
                    let offset = cluster_id * dim;
                    for i in 0..dim {
                        new_centroids[offset + i] /= count as f32;
                    }
                }
            }

            centroids = new_centroids;
        }

        let version = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            num_clusters: actual_clusters as u32,
            dim,
            centroids,
            version,
            soar_config: config.soar.clone(),
        }
    }

    /// Find nearest centroid index for a vector (static helper)
    fn find_nearest_idx_static(vector: &[f32], centroids: &[f32], dim: usize) -> usize {
        let num_clusters = centroids.len() / dim;
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for c in 0..num_clusters {
            let offset = c * dim;
            let dist: f32 = vector
                .iter()
                .zip(&centroids[offset..offset + dim])
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();

            if dist < best_dist {
                best_dist = dist;
                best_idx = c;
            }
        }

        best_idx
    }

    /// Find nearest cluster for a query vector
    pub fn find_nearest(&self, vector: &[f32]) -> u32 {
        Self::find_nearest_idx_static(vector, &self.centroids, self.dim) as u32
    }

    /// Find k nearest clusters for a query vector
    pub fn find_k_nearest(&self, vector: &[f32], k: usize) -> Vec<u32> {
        let mut distances: Vec<(u32, f32)> = (0..self.num_clusters)
            .map(|c| {
                let offset = c as usize * self.dim;
                let dist: f32 = vector
                    .iter()
                    .zip(&self.centroids[offset..offset + self.dim])
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                (c, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        distances.into_iter().map(|(c, _)| c).collect()
    }

    /// Find k nearest clusters with their distances
    pub fn find_k_nearest_with_distances(&self, vector: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut distances: Vec<(u32, f32)> = (0..self.num_clusters)
            .map(|c| {
                let offset = c as usize * self.dim;
                let dist: f32 = vector
                    .iter()
                    .zip(&self.centroids[offset..offset + self.dim])
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                (c, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        distances
    }

    /// Assign vector with SOAR (if configured) or standard assignment
    pub fn assign(&self, vector: &[f32]) -> MultiAssignment {
        if let Some(ref soar_config) = self.soar_config {
            self.assign_with_soar(vector, soar_config)
        } else {
            MultiAssignment {
                primary_cluster: self.find_nearest(vector),
                secondary_clusters: Vec::new(),
            }
        }
    }

    /// SOAR-style assignment: find secondary clusters with orthogonal residuals
    pub fn assign_with_soar(&self, vector: &[f32], config: &SoarConfig) -> MultiAssignment {
        // 1. Find primary cluster (nearest centroid)
        let primary = self.find_nearest(vector);
        let primary_centroid = self.get_centroid(primary);

        // 2. Compute primary residual r = x - c
        let residual: Vec<f32> = vector
            .iter()
            .zip(primary_centroid)
            .map(|(v, c)| v - c)
            .collect();

        let residual_norm_sq: f32 = residual.iter().map(|x| x * x).sum();

        // 3. Check if we should spill (selective spilling)
        if config.selective && residual_norm_sq < config.spill_threshold * config.spill_threshold {
            return MultiAssignment {
                primary_cluster: primary,
                secondary_clusters: Vec::new(),
            };
        }

        // 4. Find secondary clusters that MINIMIZE |⟨r, r'⟩| (orthogonal residuals)
        let mut candidates: Vec<(u32, f32)> = (0..self.num_clusters)
            .filter(|&c| c != primary)
            .map(|c| {
                let centroid = self.get_centroid(c);
                // Compute r' = x - c'
                // Then compute |⟨r, r'⟩| - we want this SMALL (orthogonal)
                let dot: f32 = vector
                    .iter()
                    .zip(centroid)
                    .zip(&residual)
                    .map(|((v, c), r)| (v - c) * r)
                    .sum();
                (c, dot.abs())
            })
            .collect();

        // Sort by orthogonality (smallest dot product first)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        MultiAssignment {
            primary_cluster: primary,
            secondary_clusters: candidates
                .iter()
                .take(config.num_secondary)
                .map(|(c, _)| *c)
                .collect(),
        }
    }

    /// Get centroid for a cluster
    pub fn get_centroid(&self, cluster_id: u32) -> &[f32] {
        let offset = cluster_id as usize * self.dim;
        &self.centroids[offset..offset + self.dim]
    }

    /// Compute residual vector (vector - centroid)
    pub fn compute_residual(&self, vector: &[f32], cluster_id: u32) -> Vec<f32> {
        let centroid = self.get_centroid(cluster_id);
        vector.iter().zip(centroid).map(|(&v, &c)| v - c).collect()
    }

    /// Save to binary file
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.write_to(&mut file)
    }

    /// Write to any writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(CENTROIDS_MAGIC)?;
        writer.write_u32::<LittleEndian>(2)?; // version 2 with SOAR support
        writer.write_u64::<LittleEndian>(self.version)?;
        writer.write_u32::<LittleEndian>(self.num_clusters)?;
        writer.write_u32::<LittleEndian>(self.dim as u32)?;

        // Write SOAR config
        if let Some(ref soar) = self.soar_config {
            writer.write_u8(1)?;
            writer.write_u32::<LittleEndian>(soar.num_secondary as u32)?;
            writer.write_u8(if soar.selective { 1 } else { 0 })?;
            writer.write_f32::<LittleEndian>(soar.spill_threshold)?;
        } else {
            writer.write_u8(0)?;
        }

        for &val in &self.centroids {
            writer.write_f32::<LittleEndian>(val)?;
        }

        Ok(())
    }

    /// Load from binary file
    pub fn load(path: &Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        Self::read_from(&mut Cursor::new(data))
    }

    /// Read from any reader
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != CENTROIDS_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid centroids file magic",
            ));
        }

        let file_version = reader.read_u32::<LittleEndian>()?;
        let version = reader.read_u64::<LittleEndian>()?;
        let num_clusters = reader.read_u32::<LittleEndian>()?;
        let dim = reader.read_u32::<LittleEndian>()? as usize;

        // Read SOAR config (version 2+)
        let soar_config = if file_version >= 2 {
            let has_soar = reader.read_u8()? != 0;
            if has_soar {
                let num_secondary = reader.read_u32::<LittleEndian>()? as usize;
                let selective = reader.read_u8()? != 0;
                let spill_threshold = reader.read_f32::<LittleEndian>()?;
                Some(SoarConfig {
                    num_secondary,
                    selective,
                    spill_threshold,
                })
            } else {
                None
            }
        } else {
            None
        };

        let mut centroids = vec![0.0f32; num_clusters as usize * dim];
        for val in &mut centroids {
            *val = reader.read_f32::<LittleEndian>()?;
        }

        Ok(Self {
            num_clusters,
            dim,
            centroids,
            version,
            soar_config,
        })
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        self.write_to(&mut buf)?;
        Ok(buf)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        Self::read_from(&mut Cursor::new(data))
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        self.centroids.len() * 4 + 64 // centroids + overhead
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_coarse_centroids_basic() {
        let dim = 64;
        let n = 1000;
        let num_clusters = 16;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let config = CoarseConfig::new(dim, num_clusters);
        let centroids = CoarseCentroids::train(&config, &vectors);

        assert_eq!(centroids.num_clusters, num_clusters as u32);
        assert_eq!(centroids.dim, dim);
    }

    #[test]
    fn test_find_nearest() {
        let dim = 32;
        let n = 500;
        let num_clusters = 8;

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        let config = CoarseConfig::new(dim, num_clusters);
        let centroids = CoarseCentroids::train(&config, &vectors);

        // Test that find_nearest returns valid cluster IDs
        for v in &vectors {
            let cluster = centroids.find_nearest(v);
            assert!(cluster < centroids.num_clusters);
        }
    }

    #[test]
    fn test_soar_assignment() {
        let dim = 32;
        let n = 100;
        let num_clusters = 8;

        let mut rng = rand::rngs::StdRng::seed_from_u64(456);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        let soar_config = SoarConfig {
            num_secondary: 2,
            selective: false,
            spill_threshold: 0.0,
        };
        let config = CoarseConfig::new(dim, num_clusters).with_soar(soar_config);
        let centroids = CoarseCentroids::train(&config, &vectors);

        // Test SOAR assignment
        let assignment = centroids.assign(&vectors[0]);
        assert!(assignment.primary_cluster < centroids.num_clusters);
        assert_eq!(assignment.secondary_clusters.len(), 2);

        // Secondary clusters should be different from primary
        for &sec in &assignment.secondary_clusters {
            assert_ne!(sec, assignment.primary_cluster);
        }
    }

    #[test]
    fn test_serialization() {
        let dim = 16;
        let n = 50;
        let num_clusters = 4;

        let mut rng = rand::rngs::StdRng::seed_from_u64(789);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        let config = CoarseConfig::new(dim, num_clusters);
        let centroids = CoarseCentroids::train(&config, &vectors);

        // Serialize and deserialize
        let bytes = centroids.to_bytes().unwrap();
        let loaded = CoarseCentroids::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.num_clusters, centroids.num_clusters);
        assert_eq!(loaded.dim, centroids.dim);
        assert_eq!(loaded.centroids.len(), centroids.centroids.len());
    }
}
