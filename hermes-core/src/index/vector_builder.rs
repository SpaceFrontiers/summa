//! Vector index building for IndexWriter
//!
//! This module handles:
//! - Training centroids/codebooks from accumulated Flat vectors
//! - Rebuilding segments with ANN indexes
//! - Threshold-based auto-triggering of vector index builds

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::directories::DirectoryWriter;
use crate::dsl::{DenseVectorConfig, Field, FieldType, VectorIndexType};
use crate::error::{Error, Result};
use crate::segment::{SegmentId, SegmentReader};

use super::IndexWriter;

impl<D: DirectoryWriter + 'static> IndexWriter<D> {
    /// Check if any dense vector field should be built and trigger training
    pub(super) async fn maybe_build_vector_index(&self) -> Result<()> {
        let dense_fields = self.get_dense_vector_fields();
        if dense_fields.is_empty() {
            return Ok(());
        }

        // Quick check: if all fields are already built, skip entirely
        // This avoids loading segments just to count vectors when index is already built
        let all_built = {
            let metadata_arc = self.segment_manager.metadata();
            let meta = metadata_arc.read().await;
            dense_fields
                .iter()
                .all(|(field, _)| meta.is_field_built(field.0))
        };
        if all_built {
            // Ensure workers have trained structures (handles from_index cold start)
            if self
                .trained_structures
                .read()
                .ok()
                .is_none_or(|g| g.is_none())
            {
                self.publish_trained_structures().await;
            }
            return Ok(());
        }

        // Count total vectors across all segments
        let segment_ids = self.segment_manager.get_segment_ids().await;
        let total_vectors = self.count_flat_vectors(&segment_ids).await;

        // Update total in metadata and check if any field should be built
        let should_build = {
            let metadata_arc = self.segment_manager.metadata();
            let mut meta = metadata_arc.write().await;
            meta.total_vectors = total_vectors;
            dense_fields.iter().any(|(field, config)| {
                let threshold = config.build_threshold.unwrap_or(1000);
                meta.should_build_field(field.0, threshold)
            })
        };

        if should_build {
            log::info!(
                "Threshold crossed ({} vectors), auto-triggering vector index build",
                total_vectors
            );
            self.build_vector_index().await?;
        }

        Ok(())
    }

    /// Build vector index from accumulated Flat vectors (trains ONCE)
    ///
    /// This trains centroids/codebooks from ALL vectors across all segments.
    /// Training happens only ONCE - subsequent calls are no-ops if already built.
    ///
    /// **Note:** This is auto-triggered by `commit()` when threshold is crossed.
    /// You typically don't need to call this manually.
    ///
    /// The process:
    /// 1. Check if already built (skip if so)
    /// 2. Collect all vectors from all segments
    /// 3. Train centroids/codebooks based on schema's index_type
    /// 4. Update metadata to mark as built (prevents re-training)
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

        let segment_ids = self.segment_manager.get_segment_ids().await;
        if segment_ids.is_empty() {
            return Ok(());
        }

        // Collect all vectors from all segments for fields that need building
        let all_vectors = self
            .collect_vectors_for_training(&segment_ids, &fields_to_build)
            .await?;

        // Train centroids/codebooks for each field
        for (field, config) in &fields_to_build {
            self.train_field_index(*field, config, &all_vectors).await?;
        }

        // Publish trained structures to workers so new segments get ANN inline.
        // Existing flat segments acquire ANN during regular background merges.
        self.publish_trained_structures().await;

        log::info!("Vector index training complete, new segments will have ANN inline");

        Ok(())
    }

    /// Publish trained structures to shared worker state so new segment builds
    /// include ANN indexes inline. Called after training completes.
    pub(super) async fn publish_trained_structures(&self) {
        let trained = {
            let metadata_arc = self.segment_manager.metadata();
            let meta = metadata_arc.read().await;
            meta.load_trained_structures(self.directory.as_ref()).await
        };
        if let Some(trained) = trained
            && let Ok(mut guard) = self.trained_structures.write()
        {
            log::info!(
                "[writer] published trained structures to workers ({} fields)",
                trained.centroids.len()
            );
            *guard = Some(trained);
        }
    }

    /// Get total vector count across all segments (for threshold checking)
    pub async fn total_vector_count(&self) -> usize {
        let metadata_arc = self.segment_manager.metadata();
        metadata_arc.read().await.total_vectors
    }

    /// Check if vector index has been built for a field
    pub async fn is_vector_index_built(&self, field: Field) -> bool {
        let metadata_arc = self.segment_manager.metadata();
        metadata_arc.read().await.is_field_built(field.0)
    }

    /// Rebuild vector index by retraining centroids/codebooks
    ///
    /// Use this when:
    /// - Significant new data has been added and you want better centroids
    /// - You want to change the number of clusters
    /// - The vector distribution has changed significantly
    ///
    /// This resets the Built state to Flat, then triggers a fresh training.
    pub async fn rebuild_vector_index(&self) -> Result<()> {
        let dense_fields = self.get_dense_vector_fields();
        if dense_fields.is_empty() {
            return Ok(());
        }
        let dense_fields: Vec<Field> = dense_fields.into_iter().map(|(f, _)| f).collect();

        // Collect files to delete and reset fields to Flat state
        let files_to_delete = {
            let metadata_arc = self.segment_manager.metadata();
            let mut meta = metadata_arc.write().await;
            let mut files = Vec::new();
            for field in &dense_fields {
                if let Some(field_meta) = meta.vector_fields.get_mut(&field.0) {
                    field_meta.state = super::VectorIndexState::Flat;
                    if let Some(ref f) = field_meta.centroids_file {
                        files.push(f.clone());
                    }
                    if let Some(ref f) = field_meta.codebook_file {
                        files.push(f.clone());
                    }
                    field_meta.centroids_file = None;
                    field_meta.codebook_file = None;
                }
            }
            meta.save(self.directory.as_ref()).await?;
            files
        };

        // Delete old centroids/codebook files
        for file in files_to_delete {
            let _ = self.directory.delete(std::path::Path::new(&file)).await;
        }

        // Clear shared trained structures so workers produce flat segments
        // during retraining (avoids stale centroid mismatch)
        if let Ok(mut guard) = self.trained_structures.write() {
            *guard = None;
        }

        log::info!("Reset vector index state to Flat, triggering rebuild...");

        // Now build fresh
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
        let metadata_arc = self.segment_manager.metadata();
        let meta = metadata_arc.read().await;
        dense_fields
            .iter()
            .filter(|(field, _)| !meta.is_field_built(field.0))
            .cloned()
            .collect()
    }

    /// Count flat vectors across all segments
    /// Only loads segments that have a vectors file to avoid unnecessary I/O
    async fn count_flat_vectors(&self, segment_ids: &[String]) -> usize {
        let mut total_vectors = 0usize;
        let mut doc_offset = 0u32;

        for id_str in segment_ids {
            let Some(segment_id) = SegmentId::from_hex(id_str) else {
                continue;
            };

            // Quick check: skip segments without vectors file
            let files = crate::segment::SegmentFiles::new(segment_id.0);
            if !self.directory.exists(&files.vectors).await.unwrap_or(false) {
                // No vectors file - segment has no vectors, skip loading
                continue;
            }

            // Only load segments that have vectors
            if let Ok(reader) = SegmentReader::open(
                self.directory.as_ref(),
                segment_id,
                Arc::clone(&self.schema),
                doc_offset,
                self.config.term_cache_blocks,
            )
            .await
            {
                for flat_data in reader.flat_vectors().values() {
                    total_vectors += flat_data.num_vectors;
                }
                doc_offset += reader.meta().num_docs;
            }
        }

        total_vectors
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
        let mut doc_offset = 0u32;
        let mut total_skipped = 0usize;

        for id_str in segment_ids {
            let segment_id = SegmentId::from_hex(id_str)
                .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {}", id_str)))?;
            let reader = SegmentReader::open(
                self.directory.as_ref(),
                segment_id,
                Arc::clone(&self.schema),
                doc_offset,
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

            doc_offset += reader.meta().num_docs;
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

    /// Serialize a trained structure to JSON and save to an index-level file.
    async fn save_trained_artifact(
        &self,
        artifact: &impl serde::Serialize,
        filename: &str,
    ) -> Result<()> {
        let bytes =
            serde_json::to_vec(artifact).map_err(|e| Error::Serialization(e.to_string()))?;
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
