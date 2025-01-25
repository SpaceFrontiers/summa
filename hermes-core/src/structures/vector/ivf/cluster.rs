//! Generic cluster data storage for IVF indexes
//!
//! Provides a unified storage structure for cluster data that works
//! with any quantization method (RaBitQ, PQ, etc.).

use serde::{Deserialize, Serialize};

/// Trait for quantized vector codes
pub trait QuantizedCode: Clone + Send + Sync {
    /// Size in bytes of this code
    fn size_bytes(&self) -> usize;
}

/// Generic cluster data storage
///
/// Stores document IDs, element ordinals, and quantized codes for vectors in a cluster.
/// Can optionally store raw vectors for re-ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterData<C: Clone> {
    /// Document IDs (local to segment)
    pub doc_ids: Vec<u32>,
    /// Element ordinals for multi-valued fields (0 for single-valued)
    /// Stored as u16 to support up to 65535 values per document per field
    pub ordinals: Vec<u16>,
    /// Quantized vector codes
    pub codes: Vec<C>,
    /// Raw vectors for re-ranking (optional)
    pub raw_vectors: Option<Vec<Vec<f32>>>,
}

impl<C: Clone> Default for ClusterData<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Clone> ClusterData<C> {
    pub fn new() -> Self {
        Self {
            doc_ids: Vec::new(),
            ordinals: Vec::new(),
            codes: Vec::new(),
            raw_vectors: None,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            doc_ids: Vec::with_capacity(capacity),
            ordinals: Vec::with_capacity(capacity),
            codes: Vec::with_capacity(capacity),
            raw_vectors: None,
        }
    }

    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Add a vector to the cluster
    pub fn add(&mut self, doc_id: u32, ordinal: u16, code: C, raw_vector: Option<Vec<f32>>) {
        self.doc_ids.push(doc_id);
        self.ordinals.push(ordinal);
        self.codes.push(code);

        if let Some(raw) = raw_vector {
            self.raw_vectors.get_or_insert_with(Vec::new).push(raw);
        }
    }

    /// Append another cluster's data (for merging segments)
    pub fn append(&mut self, other: &ClusterData<C>, doc_id_offset: u32) {
        for &doc_id in &other.doc_ids {
            self.doc_ids.push(doc_id + doc_id_offset);
        }
        self.ordinals.extend(other.ordinals.iter().copied());
        self.codes.extend(other.codes.iter().cloned());

        if let Some(ref other_raw) = other.raw_vectors {
            let raw = self.raw_vectors.get_or_insert_with(Vec::new);
            raw.extend(other_raw.iter().cloned());
        }
    }

    /// Get iterator over (doc_id, ordinal, code) tuples
    pub fn iter(&self) -> impl Iterator<Item = (u32, u16, &C)> {
        self.doc_ids
            .iter()
            .copied()
            .zip(self.ordinals.iter().copied())
            .zip(self.codes.iter())
            .map(|((doc_id, ordinal), code)| (doc_id, ordinal, code))
    }

    /// Get iterator over (doc_id, ordinal, code, optional raw_vector) tuples
    pub fn iter_with_raw(&self) -> impl Iterator<Item = (u32, u16, &C, Option<&Vec<f32>>)> {
        let raw_iter = self.raw_vectors.as_ref();
        self.doc_ids
            .iter()
            .copied()
            .zip(self.ordinals.iter().copied())
            .zip(self.codes.iter())
            .enumerate()
            .map(move |(i, ((doc_id, ordinal), code))| {
                let raw = raw_iter.and_then(|r| r.get(i));
                (doc_id, ordinal, code, raw)
            })
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.doc_ids.clear();
        self.ordinals.clear();
        self.codes.clear();
        if let Some(ref mut raw) = self.raw_vectors {
            raw.clear();
        }
    }

    /// Reserve capacity
    pub fn reserve(&mut self, additional: usize) {
        self.doc_ids.reserve(additional);
        self.ordinals.reserve(additional);
        self.codes.reserve(additional);
        if let Some(ref mut raw) = self.raw_vectors {
            raw.reserve(additional);
        }
    }
}

impl<C: Clone + QuantizedCode> ClusterData<C> {
    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        use std::mem::size_of;
        let doc_ids_size = self.doc_ids.len() * size_of::<u32>();
        let ordinals_size = self.ordinals.len() * size_of::<u16>();
        let codes_size: usize = self.codes.iter().map(|c| c.size_bytes()).sum();
        let raw_size = self
            .raw_vectors
            .as_ref()
            .map(|vecs| vecs.iter().map(|v| v.len() * size_of::<f32>()).sum())
            .unwrap_or(0);

        doc_ids_size + ordinals_size + codes_size + raw_size
    }
}

/// Storage for multiple clusters (HashMap wrapper with utilities)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStorage<C: Clone> {
    /// Cluster data indexed by cluster ID
    pub clusters: std::collections::HashMap<u32, ClusterData<C>>,
    /// Total number of vectors across all clusters
    pub total_vectors: usize,
}

impl<C: Clone> Default for ClusterStorage<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Clone> ClusterStorage<C> {
    pub fn new() -> Self {
        Self {
            clusters: std::collections::HashMap::new(),
            total_vectors: 0,
        }
    }

    pub fn with_capacity(num_clusters: usize) -> Self {
        Self {
            clusters: std::collections::HashMap::with_capacity(num_clusters),
            total_vectors: 0,
        }
    }

    /// Add a vector to a cluster
    pub fn add(
        &mut self,
        cluster_id: u32,
        doc_id: u32,
        ordinal: u16,
        code: C,
        raw_vector: Option<Vec<f32>>,
    ) {
        self.clusters
            .entry(cluster_id)
            .or_default()
            .add(doc_id, ordinal, code, raw_vector);
        self.total_vectors += 1;
    }

    /// Get cluster data
    pub fn get(&self, cluster_id: u32) -> Option<&ClusterData<C>> {
        self.clusters.get(&cluster_id)
    }

    /// Get mutable cluster data
    pub fn get_mut(&mut self, cluster_id: u32) -> Option<&mut ClusterData<C>> {
        self.clusters.get_mut(&cluster_id)
    }

    /// Get or create cluster data
    pub fn get_or_create(&mut self, cluster_id: u32) -> &mut ClusterData<C> {
        self.clusters.entry(cluster_id).or_default()
    }

    /// Number of non-empty clusters
    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Total number of vectors
    pub fn len(&self) -> usize {
        self.total_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.total_vectors == 0
    }

    /// Iterate over all clusters
    pub fn iter(&self) -> impl Iterator<Item = (u32, &ClusterData<C>)> {
        self.clusters.iter().map(|(&id, data)| (id, data))
    }

    /// Merge another storage into this one
    pub fn merge(&mut self, other: &ClusterStorage<C>, doc_id_offset: u32) {
        for (&cluster_id, other_data) in &other.clusters {
            self.clusters
                .entry(cluster_id)
                .or_default()
                .append(other_data, doc_id_offset);
        }
        self.total_vectors += other.total_vectors;
    }

    /// Clear all clusters
    pub fn clear(&mut self) {
        self.clusters.clear();
        self.total_vectors = 0;
    }
}

impl<C: Clone + QuantizedCode> ClusterStorage<C> {
    /// Total memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        self.clusters.values().map(|c| c.size_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test code type
    #[derive(Clone, Debug)]
    struct TestCode(Vec<u8>);

    impl QuantizedCode for TestCode {
        fn size_bytes(&self) -> usize {
            self.0.len()
        }
    }

    #[test]
    fn test_cluster_data_basic() {
        let mut cluster: ClusterData<TestCode> = ClusterData::new();

        cluster.add(0, 0, TestCode(vec![1, 2, 3]), None);
        cluster.add(1, 0, TestCode(vec![4, 5, 6]), None);

        assert_eq!(cluster.len(), 2);
        assert!(!cluster.is_empty());
    }

    #[test]
    fn test_cluster_data_with_raw() {
        let mut cluster: ClusterData<TestCode> = ClusterData::new();

        cluster.add(0, 0, TestCode(vec![1]), Some(vec![1.0, 2.0, 3.0]));
        cluster.add(1, 0, TestCode(vec![2]), Some(vec![4.0, 5.0, 6.0]));

        assert!(cluster.raw_vectors.is_some());
        assert_eq!(cluster.raw_vectors.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_cluster_data_with_ordinals() {
        let mut cluster: ClusterData<TestCode> = ClusterData::new();

        // Multi-valued field: doc 0 has 3 vectors
        cluster.add(0, 0, TestCode(vec![1]), None);
        cluster.add(0, 1, TestCode(vec![2]), None);
        cluster.add(0, 2, TestCode(vec![3]), None);

        assert_eq!(cluster.len(), 3);
        assert_eq!(cluster.ordinals, vec![0, 1, 2]);
    }

    #[test]
    fn test_cluster_data_append() {
        let mut cluster1: ClusterData<TestCode> = ClusterData::new();
        cluster1.add(0, 0, TestCode(vec![1]), None);
        cluster1.add(1, 0, TestCode(vec![2]), None);

        let mut cluster2: ClusterData<TestCode> = ClusterData::new();
        cluster2.add(0, 0, TestCode(vec![3]), None);
        cluster2.add(1, 0, TestCode(vec![4]), None);

        cluster1.append(&cluster2, 100);

        assert_eq!(cluster1.len(), 4);
        assert_eq!(cluster1.doc_ids, vec![0, 1, 100, 101]);
    }

    #[test]
    fn test_cluster_storage() {
        let mut storage: ClusterStorage<TestCode> = ClusterStorage::new();

        storage.add(0, 10, 0, TestCode(vec![1]), None);
        storage.add(0, 11, 0, TestCode(vec![2]), None);
        storage.add(1, 20, 0, TestCode(vec![3]), None);

        assert_eq!(storage.num_clusters(), 2);
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.get(0).unwrap().len(), 2);
        assert_eq!(storage.get(1).unwrap().len(), 1);
    }

    #[test]
    fn test_cluster_storage_merge() {
        let mut storage1: ClusterStorage<TestCode> = ClusterStorage::new();
        storage1.add(0, 0, 0, TestCode(vec![1]), None);

        let mut storage2: ClusterStorage<TestCode> = ClusterStorage::new();
        storage2.add(0, 0, 0, TestCode(vec![2]), None);
        storage2.add(1, 0, 0, TestCode(vec![3]), None);

        storage1.merge(&storage2, 100);

        assert_eq!(storage1.len(), 3);
        assert_eq!(storage1.get(0).unwrap().len(), 2);
        assert_eq!(storage1.get(0).unwrap().doc_ids, vec![0, 100]);
    }
}
