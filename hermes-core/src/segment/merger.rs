//! Segment merger for combining multiple segments

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::builder::{SegmentBuilder, SegmentBuilderConfig};
use super::reader::SegmentReader;
use super::store::StoreMerger;
use super::types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};
use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::{FieldType, Schema};
use crate::structures::{
    BlockPostingList, PostingList, RaBitQConfig, RaBitQIndex, SSTableWriter, TERMINATED, TermInfo,
};

/// Statistics for merge operations
#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    /// Number of terms processed
    pub terms_processed: usize,
    /// Number of postings merged
    pub postings_merged: usize,
    /// Peak memory usage in bytes (estimated)
    pub peak_memory_bytes: usize,
    /// Current memory usage in bytes (estimated)
    pub current_memory_bytes: usize,
    /// Term dictionary output size
    pub term_dict_bytes: usize,
    /// Postings output size
    pub postings_bytes: usize,
    /// Store output size
    pub store_bytes: usize,
    /// Vector index output size
    pub vectors_bytes: usize,
}

impl MergeStats {
    /// Format memory as human-readable string
    pub fn format_memory(bytes: usize) -> String {
        if bytes >= 1024 * 1024 * 1024 {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        } else if bytes >= 1024 * 1024 {
            format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
        } else if bytes >= 1024 {
            format!("{:.2} KB", bytes as f64 / 1024.0)
        } else {
            format!("{} B", bytes)
        }
    }
}

/// Entry for k-way merge heap
struct MergeEntry {
    key: Vec<u8>,
    term_info: TermInfo,
    segment_idx: usize,
    doc_offset: u32,
}

impl PartialEq for MergeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for MergeEntry {}

impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (BinaryHeap is max-heap by default)
        other.key.cmp(&self.key)
    }
}

/// Segment merger - merges multiple segments into one
pub struct SegmentMerger {
    schema: Arc<Schema>,
}

impl SegmentMerger {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema }
    }

    /// Merge segments - uses optimized store stacking when possible
    pub async fn merge<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        new_segment_id: SegmentId,
    ) -> Result<SegmentMeta> {
        let (meta, _stats) = self.merge_with_stats(dir, segments, new_segment_id).await?;
        Ok(meta)
    }

    /// Merge segments with memory tracking - returns merge statistics
    pub async fn merge_with_stats<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        new_segment_id: SegmentId,
    ) -> Result<(SegmentMeta, MergeStats)> {
        // Check if we can use optimized store stacking (no dictionaries)
        let can_stack_stores = segments.iter().all(|s| !s.store_has_dict());

        // Check if any segment has positions - if so, use rebuild merge
        // (positions require doc ID remapping which optimized merge doesn't handle)
        let has_positions = self
            .schema
            .fields()
            .any(|(_, entry)| entry.positions.is_some());

        if can_stack_stores && !has_positions {
            self.merge_optimized_with_stats(dir, segments, new_segment_id)
                .await
        } else {
            self.merge_rebuild_with_stats(dir, segments, new_segment_id)
                .await
        }
    }

    /// Optimized merge with stats tracking
    async fn merge_optimized_with_stats<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        new_segment_id: SegmentId,
    ) -> Result<(SegmentMeta, MergeStats)> {
        let mut stats = MergeStats::default();
        let files = SegmentFiles::new(new_segment_id.0);

        // Build merged term dictionary and postings
        let mut term_dict_data = Vec::new();
        let mut postings_data = Vec::new();
        let terms_processed = self
            .merge_postings_with_stats(
                segments,
                &mut term_dict_data,
                &mut postings_data,
                &mut stats,
            )
            .await?;
        stats.terms_processed = terms_processed;
        stats.term_dict_bytes = term_dict_data.len();
        stats.postings_bytes = postings_data.len();

        // Track peak memory (term dict + postings buffers)
        let current_mem = term_dict_data.capacity() + postings_data.capacity();
        stats.current_memory_bytes = current_mem;
        stats.peak_memory_bytes = stats.peak_memory_bytes.max(current_mem);

        // Stack store files without recompression
        let mut store_data = Vec::new();
        {
            let mut store_merger = StoreMerger::new(&mut store_data);
            for segment in segments {
                let raw_blocks = segment.store_raw_blocks();
                let data_slice = segment.store_data_slice();
                store_merger.append_store(data_slice, &raw_blocks).await?;
            }
            store_merger.finish()?;
        }
        stats.store_bytes = store_data.len();

        // Update peak memory
        let current_mem =
            term_dict_data.capacity() + postings_data.capacity() + store_data.capacity();
        stats.peak_memory_bytes = stats.peak_memory_bytes.max(current_mem);

        // Write files
        dir.write(&files.term_dict, &term_dict_data).await?;
        dir.write(&files.postings, &postings_data).await?;
        dir.write(&files.store, &store_data).await?;

        // Free memory after writing
        drop(term_dict_data);
        drop(postings_data);
        drop(store_data);

        // Merge dense vector indexes
        let vectors_bytes = self
            .merge_dense_vectors_with_stats(dir, segments, &files)
            .await?;
        stats.vectors_bytes = vectors_bytes;

        // Merge field stats
        let mut merged_field_stats: FxHashMap<u32, FieldStats> = FxHashMap::default();
        for segment in segments {
            for (&field_id, field_stats) in &segment.meta().field_stats {
                let entry = merged_field_stats.entry(field_id).or_default();
                entry.total_tokens += field_stats.total_tokens;
                entry.doc_count += field_stats.doc_count;
            }
        }

        let total_docs: u32 = segments.iter().map(|s| s.num_docs()).sum();
        let meta = SegmentMeta {
            id: new_segment_id.0,
            num_docs: total_docs,
            field_stats: merged_field_stats,
        };

        dir.write(&files.meta, &meta.serialize()?).await?;

        log::info!(
            "Merge complete: {} terms, output: term_dict={}, postings={}, store={}, vectors={}",
            stats.terms_processed,
            MergeStats::format_memory(stats.term_dict_bytes),
            MergeStats::format_memory(stats.postings_bytes),
            MergeStats::format_memory(stats.store_bytes),
            MergeStats::format_memory(stats.vectors_bytes),
        );

        Ok((meta, stats))
    }

    /// Fallback merge with stats tracking
    async fn merge_rebuild_with_stats<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        new_segment_id: SegmentId,
    ) -> Result<(SegmentMeta, MergeStats)> {
        let mut stats = MergeStats::default();

        let mut builder =
            SegmentBuilder::new((*self.schema).clone(), SegmentBuilderConfig::default())?;

        for segment in segments {
            for doc_id in 0..segment.num_docs() {
                if let Some(doc) = segment.doc(doc_id).await? {
                    builder.add_document(doc)?;
                }

                // Track memory periodically
                if doc_id % 10000 == 0 {
                    let builder_stats = builder.stats();
                    stats.current_memory_bytes = builder_stats.estimated_memory_bytes;
                    stats.peak_memory_bytes =
                        stats.peak_memory_bytes.max(stats.current_memory_bytes);
                }
            }
        }

        let meta = builder.build(dir, new_segment_id).await?;
        Ok((meta, stats))
    }

    /// Merge postings from multiple segments using streaming k-way merge
    ///
    /// This implementation uses a min-heap to merge terms from all segments
    /// in sorted order without loading all terms into memory at once.
    /// Memory usage is O(num_segments) instead of O(total_terms).
    ///
    /// Optimization: For terms that exist in only one segment, we copy the
    /// posting data directly without decode/encode. Only terms that exist
    /// in multiple segments need full merge.
    ///
    /// Returns the number of terms processed.
    async fn merge_postings_with_stats(
        &self,
        segments: &[SegmentReader],
        term_dict: &mut Vec<u8>,
        postings_out: &mut Vec<u8>,
        stats: &mut MergeStats,
    ) -> Result<usize> {
        // Calculate doc offsets for each segment
        let mut doc_offsets = Vec::with_capacity(segments.len());
        let mut offset = 0u32;
        for segment in segments {
            doc_offsets.push(offset);
            offset += segment.num_docs();
        }

        // Create iterators for each segment's term dictionary
        let mut iterators: Vec<_> = segments.iter().map(|s| s.term_dict_iter()).collect();

        // Initialize min-heap with first entry from each segment
        let mut heap: BinaryHeap<MergeEntry> = BinaryHeap::new();
        for (seg_idx, iter) in iterators.iter_mut().enumerate() {
            if let Some((key, term_info)) = iter.next().await.map_err(crate::Error::from)? {
                heap.push(MergeEntry {
                    key,
                    term_info,
                    segment_idx: seg_idx,
                    doc_offset: doc_offsets[seg_idx],
                });
            }
        }

        // Collect results for SSTable writing (need to buffer due to async)
        let mut term_results: Vec<(Vec<u8>, TermInfo)> = Vec::new();
        let mut terms_processed = 0usize;

        while !heap.is_empty() {
            // Get the smallest key
            let first = heap.pop().unwrap();
            let current_key = first.key.clone();

            // Collect all entries with the same key
            let mut sources: Vec<(usize, TermInfo, u32)> =
                vec![(first.segment_idx, first.term_info, first.doc_offset)];

            // Advance the iterator that provided this entry
            if let Some((key, term_info)) = iterators[first.segment_idx]
                .next()
                .await
                .map_err(crate::Error::from)?
            {
                heap.push(MergeEntry {
                    key,
                    term_info,
                    segment_idx: first.segment_idx,
                    doc_offset: doc_offsets[first.segment_idx],
                });
            }

            // Check if other segments have the same key
            while let Some(entry) = heap.peek() {
                if entry.key != current_key {
                    break;
                }
                let entry = heap.pop().unwrap();
                sources.push((entry.segment_idx, entry.term_info, entry.doc_offset));

                // Advance this iterator too
                if let Some((key, term_info)) = iterators[entry.segment_idx]
                    .next()
                    .await
                    .map_err(crate::Error::from)?
                {
                    heap.push(MergeEntry {
                        key,
                        term_info,
                        segment_idx: entry.segment_idx,
                        doc_offset: doc_offsets[entry.segment_idx],
                    });
                }
            }

            // Process this term
            let term_info = if sources.len() == 1 {
                // Optimization: Term exists in only one segment - copy directly
                let (seg_idx, source_info, seg_doc_offset) = &sources[0];
                self.copy_term_posting(
                    &segments[*seg_idx],
                    source_info,
                    *seg_doc_offset,
                    postings_out,
                )
                .await?
            } else {
                // Term exists in multiple segments - need full merge
                self.merge_term_postings(segments, &sources, postings_out)
                    .await?
            };

            term_results.push((current_key, term_info));
            terms_processed += 1;

            // Log progress every 100k terms
            if terms_processed.is_multiple_of(100_000) {
                log::debug!("Merge progress: {} terms processed", terms_processed);
            }
        }

        log::info!(
            "Merge complete: {} terms processed from {} segments",
            terms_processed,
            segments.len()
        );

        // Track memory for term_results buffer
        let results_mem = term_results.capacity() * std::mem::size_of::<(Vec<u8>, TermInfo)>();
        stats.current_memory_bytes = results_mem + postings_out.capacity();
        stats.peak_memory_bytes = stats.peak_memory_bytes.max(stats.current_memory_bytes);

        // Write to SSTable (sync, no await points)
        let mut writer = SSTableWriter::<TermInfo>::new(term_dict);
        for (key, term_info) in term_results {
            writer.insert(&key, &term_info)?;
        }
        writer.finish()?;

        Ok(terms_processed)
    }

    /// Copy a term's posting data directly from source segment (no decode/encode)
    /// Only adjusts doc IDs by adding the segment's doc offset
    async fn copy_term_posting(
        &self,
        segment: &SegmentReader,
        source_info: &TermInfo,
        doc_offset: u32,
        postings_out: &mut Vec<u8>,
    ) -> Result<TermInfo> {
        // Handle inline postings - need to remap doc IDs
        if let Some((doc_ids, term_freqs)) = source_info.decode_inline() {
            let remapped_ids: Vec<u32> = doc_ids.iter().map(|&id| id + doc_offset).collect();
            if let Some(inline) = TermInfo::try_inline(&remapped_ids, &term_freqs) {
                return Ok(inline);
            }
            // If can't inline after remapping (shouldn't happen), fall through to external
            let mut pl = PostingList::with_capacity(remapped_ids.len());
            for (doc_id, tf) in remapped_ids.into_iter().zip(term_freqs.into_iter()) {
                pl.push(doc_id, tf);
            }
            let posting_offset = postings_out.len() as u64;
            let block_list = BlockPostingList::from_posting_list(&pl)?;
            let mut encoded = Vec::new();
            block_list.serialize(&mut encoded)?;
            postings_out.extend_from_slice(&encoded);
            return Ok(TermInfo::external(
                posting_offset,
                encoded.len() as u32,
                pl.doc_count(),
            ));
        }

        // External posting - read, decode, remap doc IDs, re-encode
        // Note: We can't just copy bytes because doc IDs are delta-encoded
        let (offset, len) = source_info.external_info().unwrap();
        let posting_bytes = segment.read_postings(offset, len).await?;
        let source_postings = BlockPostingList::deserialize(&mut posting_bytes.as_slice())?;

        // Remap doc IDs
        let mut remapped = PostingList::with_capacity(source_postings.doc_count() as usize);
        let mut iter = source_postings.iterator();
        while iter.doc() != TERMINATED {
            remapped.add(iter.doc() + doc_offset, iter.term_freq());
            iter.advance();
        }

        // Try to inline if small enough
        let doc_ids: Vec<u32> = remapped.iter().map(|p| p.doc_id).collect();
        let term_freqs: Vec<u32> = remapped.iter().map(|p| p.term_freq).collect();

        if let Some(inline) = TermInfo::try_inline(&doc_ids, &term_freqs) {
            return Ok(inline);
        }

        // Write to postings file
        let posting_offset = postings_out.len() as u64;
        let block_list = BlockPostingList::from_posting_list(&remapped)?;
        let mut encoded = Vec::new();
        block_list.serialize(&mut encoded)?;
        postings_out.extend_from_slice(&encoded);

        Ok(TermInfo::external(
            posting_offset,
            encoded.len() as u32,
            remapped.doc_count(),
        ))
    }

    /// Merge postings for a term that exists in multiple segments
    /// Uses block-level concatenation for O(num_blocks) instead of O(num_postings)
    async fn merge_term_postings(
        &self,
        segments: &[SegmentReader],
        sources: &[(usize, TermInfo, u32)],
        postings_out: &mut Vec<u8>,
    ) -> Result<TermInfo> {
        // Sort sources by doc_offset to ensure postings are added in sorted order
        let mut sorted_sources: Vec<_> = sources.to_vec();
        sorted_sources.sort_by_key(|(_, _, doc_offset)| *doc_offset);

        // Check if all sources are external (can use block concatenation)
        let all_external = sorted_sources
            .iter()
            .all(|(_, term_info, _)| term_info.external_info().is_some());

        if all_external && sorted_sources.len() > 1 {
            // Fast path: block-level concatenation
            let mut block_sources = Vec::with_capacity(sorted_sources.len());

            for (seg_idx, term_info, doc_offset) in &sorted_sources {
                let segment = &segments[*seg_idx];
                let (offset, len) = term_info.external_info().unwrap();
                let posting_bytes = segment.read_postings(offset, len).await?;
                let source_postings = BlockPostingList::deserialize(&mut posting_bytes.as_slice())?;
                block_sources.push((source_postings, *doc_offset));
            }

            let merged_blocks = BlockPostingList::concatenate_blocks(&block_sources)?;
            let posting_offset = postings_out.len() as u64;
            let mut encoded = Vec::new();
            merged_blocks.serialize(&mut encoded)?;
            postings_out.extend_from_slice(&encoded);

            return Ok(TermInfo::external(
                posting_offset,
                encoded.len() as u32,
                merged_blocks.doc_count(),
            ));
        }

        // Slow path: full decode/encode for inline postings or single source
        let mut merged = PostingList::new();

        for (seg_idx, term_info, doc_offset) in &sorted_sources {
            let segment = &segments[*seg_idx];

            if let Some((doc_ids, term_freqs)) = term_info.decode_inline() {
                // Inline posting list
                for (doc_id, tf) in doc_ids.into_iter().zip(term_freqs.into_iter()) {
                    merged.add(doc_id + doc_offset, tf);
                }
            } else {
                // External posting list
                let (offset, len) = term_info.external_info().unwrap();
                let posting_bytes = segment.read_postings(offset, len).await?;
                let source_postings = BlockPostingList::deserialize(&mut posting_bytes.as_slice())?;

                let mut iter = source_postings.iterator();
                while iter.doc() != TERMINATED {
                    merged.add(iter.doc() + doc_offset, iter.term_freq());
                    iter.advance();
                }
            }
        }

        // Try to inline small posting lists
        let doc_ids: Vec<u32> = merged.iter().map(|p| p.doc_id).collect();
        let term_freqs: Vec<u32> = merged.iter().map(|p| p.term_freq).collect();

        if let Some(inline) = TermInfo::try_inline(&doc_ids, &term_freqs) {
            return Ok(inline);
        }

        // Write to postings file
        let posting_offset = postings_out.len() as u64;
        let block_list = BlockPostingList::from_posting_list(&merged)?;
        let mut encoded = Vec::new();
        block_list.serialize(&mut encoded)?;
        postings_out.extend_from_slice(&encoded);

        Ok(TermInfo::external(
            posting_offset,
            encoded.len() as u32,
            merged.doc_count(),
        ))
    }
    /// Merge dense vector indexes with stats tracking - returns output size in bytes
    ///
    /// For ScaNN (IVF-PQ): O(1) merge by concatenating cluster data (same codebook)
    /// For IVF-RaBitQ: O(1) merge by concatenating cluster data (same centroids)
    /// For RaBitQ: Must rebuild with new centroid from all vectors
    async fn merge_dense_vectors_with_stats<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
    ) -> Result<usize> {
        use byteorder::{LittleEndian, WriteBytesExt};

        // (field_id, index_type, data)
        let mut field_indexes: Vec<(u32, u8, Vec<u8>)> = Vec::new();

        for (field, entry) in self.schema.fields() {
            if !matches!(entry.field_type, FieldType::DenseVector) {
                continue;
            }

            // Check if all segments have ScaNN indexes for this field
            let scann_indexes: Vec<_> = segments
                .iter()
                .filter_map(|s| s.get_scann_vector_index(field))
                .collect();

            if scann_indexes.len()
                == segments
                    .iter()
                    .filter(|s| s.has_dense_vector_index(field))
                    .count()
                && !scann_indexes.is_empty()
            {
                // All segments have ScaNN - use O(1) cluster merge!
                let refs: Vec<&crate::structures::IVFPQIndex> =
                    scann_indexes.iter().map(|(idx, _)| idx.as_ref()).collect();

                // Calculate doc_id offsets
                let mut doc_offsets = Vec::with_capacity(segments.len());
                let mut offset = 0u32;
                for segment in segments {
                    doc_offsets.push(offset);
                    offset += segment.num_docs();
                }

                match crate::structures::IVFPQIndex::merge(&refs, &doc_offsets) {
                    Ok(merged) => {
                        let bytes = merged
                            .to_bytes()
                            .map_err(|e| crate::Error::Serialization(e.to_string()))?;
                        field_indexes.push((field.0, 2u8, bytes)); // 2 = ScaNN
                        continue;
                    }
                    Err(e) => {
                        log::warn!("ScaNN merge failed: {}, falling back to IVF", e);
                    }
                }
            }

            // Check if all segments have IVF indexes for this field
            let ivf_indexes: Vec<_> = segments
                .iter()
                .filter_map(|s| s.get_ivf_vector_index(field))
                .collect();

            if ivf_indexes.len()
                == segments
                    .iter()
                    .filter(|s| s.has_dense_vector_index(field))
                    .count()
                && !ivf_indexes.is_empty()
            {
                // All segments have IVF - use O(1) cluster merge!
                let refs: Vec<&crate::structures::IVFRaBitQIndex> =
                    ivf_indexes.iter().map(|arc| arc.as_ref()).collect();

                // Calculate doc_id offsets
                let mut doc_offsets = Vec::with_capacity(segments.len());
                let mut offset = 0u32;
                for segment in segments {
                    doc_offsets.push(offset);
                    offset += segment.num_docs();
                }

                match crate::structures::IVFRaBitQIndex::merge(&refs, &doc_offsets) {
                    Ok(merged) => {
                        let bytes = merged
                            .to_bytes()
                            .map_err(|e| crate::Error::Serialization(e.to_string()))?;
                        field_indexes.push((field.0, 1u8, bytes)); // 1 = IVF-RaBitQ
                        continue;
                    }
                    Err(e) => {
                        log::warn!("IVF merge failed: {}, falling back to rebuild", e);
                    }
                }
            }

            // Fall back to RaBitQ rebuild (collect raw vectors)
            let mut all_vectors: Vec<Vec<f32>> = Vec::new();

            for segment in segments {
                if let Some(index) = segment.get_dense_vector_index(field)
                    && let Some(raw_vecs) = &index.raw_vectors
                {
                    all_vectors.extend(raw_vecs.iter().cloned());
                }
            }

            if !all_vectors.is_empty() {
                let dim = all_vectors[0].len();
                let config = RaBitQConfig::new(dim);
                let merged_index = RaBitQIndex::build(config, &all_vectors, true);

                let index_bytes = serde_json::to_vec(&merged_index)
                    .map_err(|e| crate::Error::Serialization(e.to_string()))?;

                field_indexes.push((field.0, 0u8, index_bytes)); // 0 = RaBitQ
            }
        }

        // Write unified vectors file with index_type
        if !field_indexes.is_empty() {
            field_indexes.sort_by_key(|(id, _, _)| *id);

            // Header: num_fields + (field_id, index_type, offset, len) per field
            let header_size = 4 + field_indexes.len() * (4 + 1 + 8 + 8);
            let mut output = Vec::new();

            output.write_u32::<LittleEndian>(field_indexes.len() as u32)?;

            let mut current_offset = header_size as u64;
            for (field_id, index_type, data) in &field_indexes {
                output.write_u32::<LittleEndian>(*field_id)?;
                output.write_u8(*index_type)?;
                output.write_u64::<LittleEndian>(current_offset)?;
                output.write_u64::<LittleEndian>(data.len() as u64)?;
                current_offset += data.len() as u64;
            }

            for (_, _, data) in field_indexes {
                output.extend_from_slice(&data);
            }

            let output_size = output.len();
            dir.write(&files.vectors, &output).await?;
            return Ok(output_size);
        }

        Ok(0)
    }
}

/// Delete segment files from directory
pub async fn delete_segment<D: Directory + DirectoryWriter>(
    dir: &D,
    segment_id: SegmentId,
) -> Result<()> {
    let files = SegmentFiles::new(segment_id.0);
    let _ = dir.delete(&files.term_dict).await;
    let _ = dir.delete(&files.postings).await;
    let _ = dir.delete(&files.store).await;
    let _ = dir.delete(&files.meta).await;
    let _ = dir.delete(&files.vectors).await;
    Ok(())
}
