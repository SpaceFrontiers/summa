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
use crate::segment::{SegmentId, SegmentMerger, SegmentReader, TrainedVectorStructures};

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

        // Wait for any background merges to complete before training.
        // rebuild_segments_with_ann() calls replace_segments() which clears ALL
        // segments atomically — concurrent merges would lose data or operate on
        // stale/deleted segments.
        self.segment_manager.wait_for_merges().await;

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

        log::info!("Vector index training complete. Rebuilding segments with ANN indexes...");

        // Rebuild segments with ANN indexes using trained structures
        self.rebuild_segments_with_ann().await?;

        Ok(())
    }

    /// Rebuild all segments with ANN indexes using trained centroids/codebooks
    pub(super) async fn rebuild_segments_with_ann(&self) -> Result<()> {
        // Pause background merges and wait for any in-flight ones to finish.
        // rebuild replaces ALL segments atomically — concurrent merges would
        // operate on stale/deleted segments or lose their output.
        self.segment_manager.pause_merges();
        self.segment_manager.wait_for_merges().await;

        let result = self.rebuild_segments_with_ann_inner().await;

        // Always resume merges, even on error
        self.segment_manager.resume_merges();

        result
    }

    async fn rebuild_segments_with_ann_inner(&self) -> Result<()> {
        let segment_ids = self.segment_manager.get_segment_ids().await;
        if segment_ids.is_empty() {
            return Ok(());
        }

        // Load trained structures from metadata
        let (trained_centroids, trained_codebooks) = {
            let metadata_arc = self.segment_manager.metadata();
            let meta = metadata_arc.read().await;
            meta.load_trained_structures(self.directory.as_ref()).await
        };

        if trained_centroids.is_empty() {
            log::info!("No trained structures to rebuild with");
            return Ok(());
        }

        let trained = TrainedVectorStructures {
            centroids: trained_centroids,
            codebooks: trained_codebooks,
        };

        // Load all segment readers
        let readers = self.load_segment_readers(&segment_ids).await?;

        // Calculate total doc count for the merged segment
        let total_docs: u32 = readers.iter().map(|r| r.meta().num_docs).sum();

        // Merge all segments into one with ANN indexes
        let merger = SegmentMerger::new(Arc::clone(&self.schema));
        let new_segment_id = SegmentId::new();
        merger
            .merge_with_ann(self.directory.as_ref(), &readers, new_segment_id, &trained)
            .await?;

        // Atomically update segments and delete old ones via SegmentManager
        self.segment_manager
            .replace_segments(vec![(new_segment_id.to_hex(), total_docs)], segment_ids)
            .await?;

        log::info!("Segments rebuilt with ANN indexes");
        Ok(())
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
        let dense_fields: Vec<Field> = self
            .schema
            .fields()
            .filter_map(|(field, entry)| {
                if entry.field_type == FieldType::DenseVector && entry.indexed {
                    Some(field)
                } else {
                    None
                }
            })
            .collect();

        if dense_fields.is_empty() {
            return Ok(());
        }

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
                if n <= remaining {
                    // Take all vectors from this segment (async reads)
                    for i in 0..n {
                        if let Ok(vec) = lazy_flat.get_vector(i).await {
                            entry.push(vec);
                        }
                    }
                } else {
                    // Uniform sample: take every Nth vector
                    let step = (n / remaining).max(1);
                    for i in 0..n {
                        if i % step == 0
                            && entry.len() < MAX_TRAINING_VECTORS
                            && let Ok(vec) = lazy_flat.get_vector(i).await
                        {
                            entry.push(vec);
                        }
                    }
                    total_skipped += n - remaining;
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

    /// Load segment readers for given IDs
    pub(super) async fn load_segment_readers(
        &self,
        segment_ids: &[String],
    ) -> Result<Vec<SegmentReader>> {
        let mut readers = Vec::new();
        let mut doc_offset = 0u32;

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
            doc_offset += reader.meta().num_docs;
            readers.push(reader);
        }

        Ok(readers)
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

        // Save centroids to index-level file
        let centroids_path = std::path::Path::new(centroids_filename);
        let centroids_bytes =
            serde_json::to_vec(&centroids).map_err(|e| Error::Serialization(e.to_string()))?;
        self.directory
            .write(centroids_path, &centroids_bytes)
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
        // Train coarse centroids
        let coarse_config = crate::structures::CoarseConfig::new(dim, num_clusters);
        let centroids = crate::structures::CoarseCentroids::train(&coarse_config, vectors);

        // Train PQ codebook
        let pq_config = crate::structures::PQConfig::new(dim);
        let codebook = crate::structures::PQCodebook::train(pq_config, vectors, 10);

        // Save centroids
        let centroids_path = std::path::Path::new(centroids_filename);
        let centroids_bytes =
            serde_json::to_vec(&centroids).map_err(|e| Error::Serialization(e.to_string()))?;
        self.directory
            .write(centroids_path, &centroids_bytes)
            .await?;

        // Save codebook
        let codebook_path = std::path::Path::new(codebook_filename);
        let codebook_bytes =
            serde_json::to_vec(&codebook).map_err(|e| Error::Serialization(e.to_string()))?;
        self.directory.write(codebook_path, &codebook_bytes).await?;

        log::info!(
            "Saved ScaNN centroids and codebook for field {} ({} clusters)",
            field_id,
            centroids.num_clusters
        );

        Ok(())
    }
}
