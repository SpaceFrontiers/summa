//! Unified data loading for benchmarks

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Dense embeddings with metadata
pub struct DenseData {
    pub vectors: Vec<Vec<f32>>,
    pub dim: usize,
}

/// Sparse embeddings (indices, values)
pub struct SparseData {
    pub vectors: Vec<(Vec<u32>, Vec<f32>)>,
}

/// Text data for BM25 benchmarking
pub struct TextData {
    pub texts: Vec<String>,
}

/// Ground truth for recall computation
pub struct GroundTruth {
    pub neighbors: Vec<Vec<u32>>,
    pub k: usize,
}

/// Relevance judgments for IR metrics (MRR, NDCG)
pub struct Qrels {
    pub relevance: HashMap<u32, Vec<u32>>,
}

impl DenseData {
    /// Load dense embeddings from binary file
    /// Format: num_vectors (u32), dim (u32), vectors (f32 * num_vectors * dim)
    pub fn load(path: &Path) -> Option<Self> {
        let file = File::open(path).ok()?;
        let mut reader = BufReader::new(file);

        let mut header = [0u8; 8];
        reader.read_exact(&mut header).ok()?;

        let num_vectors = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let dim = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

        let mut data = vec![0u8; num_vectors * dim * 4];
        reader.read_exact(&mut data).ok()?;

        let vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|i| {
                let offset = i * dim * 4;
                (0..dim)
                    .map(|j| {
                        let idx = offset + j * 4;
                        f32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]])
                    })
                    .collect()
            })
            .collect();

        println!(
            "Loaded {} dense vectors (dim={}) from {:?}",
            num_vectors, dim, path
        );
        Some(Self { vectors, dim })
    }

    /// Limit to first n vectors
    pub fn take(&self, n: usize) -> Self {
        Self {
            vectors: self.vectors.iter().take(n).cloned().collect(),
            dim: self.dim,
        }
    }
}

impl Clone for GroundTruth {
    fn clone(&self) -> Self {
        Self {
            neighbors: self.neighbors.clone(),
            k: self.k,
        }
    }
}

impl GroundTruth {
    /// Load ground truth from binary file
    /// Format: num_queries (u32), k (u32), indices (u32 * num_queries * k)
    pub fn load(path: &Path) -> Option<Self> {
        let file = File::open(path).ok()?;
        let mut reader = BufReader::new(file);

        let mut header = [0u8; 8];
        reader.read_exact(&mut header).ok()?;

        let num_queries = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let k = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

        let mut data = vec![0u8; num_queries * k * 4];
        reader.read_exact(&mut data).ok()?;

        let neighbors: Vec<Vec<u32>> = (0..num_queries)
            .map(|i| {
                let offset = i * k * 4;
                (0..k)
                    .map(|j| {
                        let idx = offset + j * 4;
                        u32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]])
                    })
                    .collect()
            })
            .collect();

        println!("Loaded ground truth: {} queries, k={}", num_queries, k);
        Some(Self { neighbors, k })
    }

    /// Limit to first n queries
    pub fn take(&self, n: usize) -> Self {
        Self {
            neighbors: self.neighbors.iter().take(n).cloned().collect(),
            k: self.k,
        }
    }
}

impl SparseData {
    /// Load sparse embeddings from binary file
    /// Format: num_vectors (u32), then for each: num_nonzero (u32), indices (u32*), values (f32*)
    pub fn load(path: &Path) -> Option<Self> {
        let file = File::open(path).ok()?;
        let mut reader = BufReader::new(file);

        let mut header = [0u8; 4];
        reader.read_exact(&mut header).ok()?;
        let num_vectors = u32::from_le_bytes(header) as usize;

        let mut vectors = Vec::with_capacity(num_vectors);
        let mut total_nnz = 0usize;

        for _ in 0..num_vectors {
            let mut nnz_buf = [0u8; 4];
            reader.read_exact(&mut nnz_buf).ok()?;
            let nnz = u32::from_le_bytes(nnz_buf) as usize;
            total_nnz += nnz;

            let mut indices_buf = vec![0u8; nnz * 4];
            reader.read_exact(&mut indices_buf).ok()?;
            let indices: Vec<u32> = (0..nnz)
                .map(|i| {
                    let idx = i * 4;
                    u32::from_le_bytes([
                        indices_buf[idx],
                        indices_buf[idx + 1],
                        indices_buf[idx + 2],
                        indices_buf[idx + 3],
                    ])
                })
                .collect();

            let mut values_buf = vec![0u8; nnz * 4];
            reader.read_exact(&mut values_buf).ok()?;
            let values: Vec<f32> = (0..nnz)
                .map(|i| {
                    let idx = i * 4;
                    f32::from_le_bytes([
                        values_buf[idx],
                        values_buf[idx + 1],
                        values_buf[idx + 2],
                        values_buf[idx + 3],
                    ])
                })
                .collect();

            vectors.push((indices, values));
        }

        let avg_nnz = total_nnz as f64 / num_vectors as f64;
        println!(
            "Loaded {} sparse vectors (avg nnz={:.1}) from {:?}",
            num_vectors, avg_nnz, path
        );
        Some(Self { vectors })
    }
}

impl TextData {
    /// Load text data from file (one text per line)
    pub fn load(path: &Path) -> Option<Self> {
        let file = File::open(path).ok()?;
        let reader = BufReader::new(file);
        let texts: Vec<String> = reader.lines().collect::<Result<_, _>>().ok()?;
        println!("Loaded {} texts from {:?}", texts.len(), path);
        Some(Self { texts })
    }
}

impl Qrels {
    /// Load qrels from binary file
    /// Format: num_queries (u32), then for each: query_idx (u32), num_relevant (u32), relevant_idxs (u32*)
    pub fn load(path: &Path) -> Option<Self> {
        let file = File::open(path).ok()?;
        let mut reader = BufReader::new(file);

        let mut header = [0u8; 4];
        reader.read_exact(&mut header).ok()?;
        let num_queries = u32::from_le_bytes(header) as usize;

        let mut relevance = HashMap::new();
        let mut total_relevant = 0usize;

        for _ in 0..num_queries {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf).ok()?;
            let query_idx = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
            let num_relevant = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
            total_relevant += num_relevant;

            let mut data = vec![0u8; num_relevant * 4];
            reader.read_exact(&mut data).ok()?;
            let relevant: Vec<u32> = (0..num_relevant)
                .map(|i| {
                    let idx = i * 4;
                    u32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]])
                })
                .collect();
            relevance.insert(query_idx, relevant);
        }

        println!(
            "Loaded qrels: {} queries, {} total relevant",
            num_queries, total_relevant
        );
        Some(Self { relevance })
    }
}
