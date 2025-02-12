//! Vector index building for IndexWriter
//!
//! Training is **manual-only** — decoupled from commit.
//! Call `build_vector_index()` explicitly when ready.
//! ANN indexes are built naturally during subsequent merges.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::directories::DirectoryWriter;
use crate::dsl::{DenseVectorConfig, Field, FieldType, VectorIndexType};
use crate::error::{Error, Result};
use crate::segment::{SegmentId, SegmentReader};

use super::IndexWriter;

impl<D: DirectoryWriter + 'static> IndexWriter<D> {
    /// Train vector index from accumulated Flat vectors (manual, not auto-triggered).
    ///
    /// 1. Acquires a snapshot (segments safe to read)
    /// 2. Collects vectors for training
    /// 3. Trains centroids/codebooks
    /// 4. Updates metadata (marks fields as Built)
    /// 5. Publishes to ArcSwap — merges will use these automatically
    ///
    /// Existing flat segments get ANN during normal merges. No rebuild needed.
    pub async fn build_vector_index(&self) -> Result<()> {
        let dense_fields = self.get_dense_vector_fields();
        if dense_fields.is_empty() {
            log::info!("No dense vector fields configured for ANN indexing");
            return Ok(());
        }

        // Check which fields need building (skip already built)
        let fields_to_build = self.get_fields_to_build(&dense_fields).await;
        if fields_to_build.is_empty() {
            log::info!("All vector fields already built, skipping training");
            return Ok(());
        }

        // Acquire snapshot — segments won't be deleted while we read them
        let snapshot = self.segment_manager.acquire_snapshot().await;
        let segment_ids = snapshot.segment_ids();
        if segment_ids.is_empty() {
            return Ok(());
        }

        // Collect vectors for training
        let all_vectors = self
            .collect_vectors_for_training(segment_ids, &fields_to_build)
            .await?;

        // Train centroids/codebooks for each field
        for (field, config) in &fields_to_build {
            self.train_field_index(*field, config, &all_vectors).await?;
        }

        // Publish to ArcSwap — merges and new segment builds will use these
        self.segment_manager.load_and_publish_trained().await;

        log::info!("Vector index training complete, ANN will be built during merges");

        Ok(())
    }

    /// Rebuild vector index by retraining centroids/codebooks.
    ///
    /// Resets Built state to Flat, clears trained structures, then trains fresh.
    pub async fn rebuild_vector_index(&self) -> Result<()> {
        let dense_fields = self.get_dense_vector_fields();
        if dense_fields.is_empty() {
            return Ok(());
        }
        let dense_fields: Vec<Field> = dense_fields.into_iter().map(|(f, _)| f).collect();

        // Reset fields to Flat and collect files to delete
        let dense_field_ids: Vec<u32> = dense_fields.iter().map(|f| f.0).collect();
        let mut files_to_delete = Vec::new();
        self.segment_manager
            .update_metadata(|meta| {
                for field_id in &dense_field_ids {
                    if let Some(field_meta) = meta.vector_fields.get_mut(field_id) {
                        field_meta.state = super::VectorIndexState::Flat;
                        if let Some(ref f) = field_meta.centroids_file {
                            files_to_delete.push(f.clone());
                        }
                        if let Some(ref f) = field_meta.codebook_file {
                            files_to_delete.push(f.clone());
                        }
                        field_meta.centroids_file = None;
                        field_meta.codebook_file = None;
                    }
                }
            })
            .await?;

        // Delete old files
        for file in files_to_delete {
            let _ = self.directory.delete(std::path::Path::new(&file)).await;
        }

        // Clear ArcSwap so workers produce flat segments during retraining
        self.segment_manager.clear_trained();

        log::info!("Reset vector index state to Flat, triggering rebuild...");

        self.build_vector_index().await
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    /// Get all dense vector fields that need ANN indexes
    fn get_dense_vector_fields(&self) -> Vec<(Field, DenseVectorConfig)> {
        self.schema
            .fields()
            .filter_map(|(field, entry)| {
                if entry.field_type == FieldType::DenseVector && entry.indexed {
                    entry
                        .dense_vector_config
                        .as_ref()
                        .filter(|c| !c.is_flat())
                        .map(|c| (field, c.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get fields that need building (not already built)
    async fn get_fields_to_build(
        &self,
        dense_fields: &[(Field, DenseVectorConfig)],
    ) -> Vec<(Field, DenseVectorConfig)> {
        let field_ids: Vec<u32> = dense_fields.iter().map(|(f, _)| f.0).collect();
        let built: Vec<u32> = self
            .segment_manager
            .read_metadata(|meta| {
                field_ids
                    .iter()
                    .filter(|fid| meta.is_field_built(**fid))
                    .copied()
                    .collect()
            })
            .await;
        dense_fields
            .iter()
            .filter(|(field, _)| !built.contains(&field.0))
            .cloned()
            .collect()
    }

    /// Collect vectors from segments for training, with sampling for large datasets.
    ///
    /// K-means clustering converges well with ~100K samples, so we cap collection
    /// per field to avoid loading millions of vectors into memory.
    async fn collect_vectors_for_training(
        &self,
        segment_ids: &[String],
        fields_to_build: &[(Field, DenseVectorConfig)],
    ) -> Result<FxHashMap<u32, Vec<Vec<f32>>>> {
        /// Maximum vectors per field for training. K-means converges well with ~100K samples.
        const MAX_TRAINING_VECTORS: usize = 100_000;

        let mut all_vectors: FxHashMap<u32, Vec<Vec<f32>>> = FxHashMap::default();
        let mut total_skipped = 0usize;

        for id_str in segment_ids {
            let segment_id = SegmentId::from_hex(id_str)
                .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {}", id_str)))?;
            let reader = SegmentReader::open(
                self.directory.as_ref(),
                segment_id,
                Arc::clone(&self.schema),
                self.config.term_cache_blocks,
            )
            .await?;

            for (field_id, lazy_flat) in reader.flat_vectors() {
                if !fields_to_build.iter().any(|(f, _)| f.0 == *field_id) {
                    continue;
                }
                let entry = all_vectors.entry(*field_id).or_default();
                let remaining = MAX_TRAINING_VECTORS.saturating_sub(entry.len());

                if remaining == 0 {
                    total_skipped += lazy_flat.num_vectors;
                    continue;
                }

                let n = lazy_flat.num_vectors;
                let dim = lazy_flat.dim;
                let quant = lazy_flat.quantization;

                // Determine which vector indices to collect
                let indices: Vec<usize> = if n <= remaining {
                    (0..n).collect()
                } else {
                    let step = (n / remaining).max(1);
                    (0..n).step_by(step).take(remaining).collect()
                };

                if indices.len() < n {
                    total_skipped += n - indices.len();
                }

                // Batch-read and dequantize instead of one-by-one get_vector()
                const BATCH: usize = 1024;
                let mut f32_buf = vec![0f32; BATCH * dim];
                for chunk in indices.chunks(BATCH) {
                    // For contiguous ranges, use batch read
                    let start = chunk[0];
                    let end = *chunk.last().unwrap();
                    if end - start + 1 == chunk.len() {
                        // Contiguous — single batch read
                        if let Ok(batch_bytes) =
                            lazy_flat.read_vectors_batch(start, chunk.len()).await
                        {
                            let floats = chunk.len() * dim;
                            f32_buf.resize(floats, 0.0);
                            crate::segment::dequantize_raw(
                                batch_bytes.as_slice(),
                                quant,
                                floats,
                                &mut f32_buf,
                            );
                            for i in 0..chunk.len() {
                                entry.push(f32_buf[i * dim..(i + 1) * dim].to_vec());
                            }
                        }
                    } else {
                        // Non-contiguous (sampled) — read individually but reuse buffer
                        f32_buf.resize(dim, 0.0);
                        for &idx in chunk {
                            if let Ok(()) = lazy_flat.read_vector_into(idx, &mut f32_buf).await {
                                entry.push(f32_buf[..dim].to_vec());
                            }
                        }
                    }
                }
            }
        }

        if total_skipped > 0 {
            let collected: usize = all_vectors.values().map(|v| v.len()).sum();
            log::info!(
                "Sampled {} vectors for training (skipped {}, max {} per field)",
                collected,
                total_skipped,
                MAX_TRAINING_VECTORS,
            );
        }

        Ok(all_vectors)
    }

    /// Train index for a single field
    async fn train_field_index(
        &self,
        field: Field,
        config: &DenseVectorConfig,
        all_vectors: &FxHashMap<u32, Vec<Vec<f32>>>,
    ) -> Result<()> {
        let field_id = field.0;
        let vectors = match all_vectors.get(&field_id) {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(()),
        };

        let dim = config.dim;
        let num_vectors = vectors.len();
        let num_clusters = config.optimal_num_clusters(num_vectors);

        log::info!(
            "Training vector index for field {} with {} vectors, {} clusters (dim={})",
            field_id,
            num_vectors,
            num_clusters,
            dim,
        );

        let centroids_filename = format!("field_{}_centroids.bin", field_id);
        let mut codebook_filename: Option<String> = None;

        match config.index_type {
            VectorIndexType::IvfRaBitQ => {
                self.train_ivf_rabitq(field_id, dim, num_clusters, vectors, &centroids_filename)
                    .await?;
            }
            VectorIndexType::ScaNN => {
                codebook_filename = Some(format!("field_{}_codebook.bin", field_id));
                self.train_scann(
                    field_id,
                    dim,
                    num_clusters,
                    vectors,
                    &centroids_filename,
                    codebook_filename.as_ref().unwrap(),
                )
                .await?;
            }
            _ => {
                // RaBitQ or Flat - no pre-training needed
                return Ok(());
            }
        }

        // Update metadata to mark this field as built
        self.segment_manager
            .update_metadata(|meta| {
                meta.init_field(field_id, config.index_type);
                meta.total_vectors = num_vectors;
                meta.mark_field_built(
                    field_id,
                    num_vectors,
                    num_clusters,
                    centroids_filename.clone(),
                    codebook_filename.clone(),
                );
            })
            .await?;

        Ok(())
    }

    /// Serialize a trained structure to bincode and save to an index-level file.
    async fn save_trained_artifact(
        &self,
        artifact: &impl serde::Serialize,
        filename: &str,
    ) -> Result<()> {
        let bytes = bincode::serde::encode_to_vec(artifact, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;
        self.directory
            .write(std::path::Path::new(filename), &bytes)
            .await?;
        Ok(())
    }

    /// Train IVF-RaBitQ centroids
    async fn train_ivf_rabitq(
        &self,
        field_id: u32,
        dim: usize,
        num_clusters: usize,
        vectors: &[Vec<f32>],
        centroids_filename: &str,
    ) -> Result<()> {
        let coarse_config = crate::structures::CoarseConfig::new(dim, num_clusters);
        let centroids = crate::structures::CoarseCentroids::train(&coarse_config, vectors);
        self.save_trained_artifact(&centroids, centroids_filename)
            .await?;

        log::info!(
            "Saved IVF-RaBitQ centroids for field {} ({} clusters)",
            field_id,
            centroids.num_clusters
        );
        Ok(())
    }

    /// Train ScaNN (IVF-PQ) centroids and codebook
    async fn train_scann(
        &self,
        field_id: u32,
        dim: usize,
        num_clusters: usize,
        vectors: &[Vec<f32>],
        centroids_filename: &str,
        codebook_filename: &str,
    ) -> Result<()> {
        let coarse_config = crate::structures::CoarseConfig::new(dim, num_clusters);
        let centroids = crate::structures::CoarseCentroids::train(&coarse_config, vectors);
        self.save_trained_artifact(&centroids, centroids_filename)
            .await?;

        let pq_config = crate::structures::PQConfig::new(dim);
        let codebook = crate::structures::PQCodebook::train(pq_config, vectors, 10);
        self.save_trained_artifact(&codebook, codebook_filename)
            .await?;

        log::info!(
            "Saved ScaNN centroids and codebook for field {} ({} clusters)",
            field_id,
            centroids.num_clusters
        );
        Ok(())
    }
}
