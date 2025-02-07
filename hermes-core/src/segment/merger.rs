//! Segment merger for combining multiple segments

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::Write;
use std::mem::size_of;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::reader::SegmentReader;
use super::store::StoreMerger;
use super::types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};
use crate::Result;
use crate::directories::{Directory, DirectoryWriter, StreamingWriter};
use crate::dsl::{FieldType, Schema};
use crate::structures::{
    BlockPostingList, PositionPostingList, PostingList, RaBitQConfig, RaBitQIndex, SSTableWriter,
    TERMINATED, TermInfo,
};

/// Write adapter that tracks bytes written.
///
/// Concrete type so it works with generic `serialize<W: Write>` functions
/// (unlike `dyn StreamingWriter` which isn't `Sized`).
pub(crate) struct OffsetWriter {
    inner: Box<dyn StreamingWriter>,
    offset: u64,
}

impl OffsetWriter {
    fn new(inner: Box<dyn StreamingWriter>) -> Self {
        Self { inner, offset: 0 }
    }

    /// Current write position (total bytes written so far).
    fn offset(&self) -> u64 {
        self.offset
    }

    /// Finalize the underlying streaming writer.
    fn finish(self) -> std::io::Result<()> {
        self.inner.finish()
    }
}

impl Write for OffsetWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.offset += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Format byte count as human-readable string
fn format_bytes(bytes: usize) -> String {
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

/// Compute per-segment doc ID offsets (each segment's docs start after the previous)
fn doc_offsets(segments: &[SegmentReader]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(segments.len());
    let mut acc = 0u32;
    for seg in segments {
        offsets.push(acc);
        acc += seg.num_docs();
    }
    offsets
}

/// Statistics for merge operations
#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    /// Number of terms processed
    pub terms_processed: usize,
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
    /// Sparse vector index output size
    pub sparse_bytes: usize,
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

/// Trained vector index structures for rebuilding segments with ANN indexes
pub struct TrainedVectorStructures {
    /// Trained centroids per field_id
    pub centroids: rustc_hash::FxHashMap<u32, Arc<crate::structures::CoarseCentroids>>,
    /// Trained PQ codebooks per field_id (for ScaNN)
    pub codebooks: rustc_hash::FxHashMap<u32, Arc<crate::structures::PQCodebook>>,
}

/// Strategy for handling dense vector indexes during merge
pub enum DenseVectorStrategy<'a> {
    /// Merge existing indexes (ScaNN/IVF cluster merge or RaBitQ rebuild)
    MergeExisting,
    /// Build ANN indexes from flat vectors using trained structures
    BuildAnn(&'a TrainedVectorStructures),
}

/// Segment merger - merges multiple segments into one
pub struct SegmentMerger {
    schema: Arc<Schema>,
}

impl SegmentMerger {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema }
    }

    /// Merge segments into one, streaming postings/positions/store directly to files.
    pub async fn merge<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        new_segment_id: SegmentId,
    ) -> Result<(SegmentMeta, MergeStats)> {
        self.merge_core(
            dir,
            segments,
            new_segment_id,
            DenseVectorStrategy::MergeExisting,
        )
        .await
    }

    /// Core merge: handles all mandatory parts (postings, positions, store, sparse, field stats, meta)
    /// and delegates dense vector handling to the provided strategy.
    ///
    /// Uses streaming writers so postings, positions, and store data flow directly
    /// to files instead of buffering everything in memory. Only the term dictionary
    /// (compact key+TermInfo entries) is buffered.
    async fn merge_core<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        new_segment_id: SegmentId,
        dense_strategy: DenseVectorStrategy<'_>,
    ) -> Result<(SegmentMeta, MergeStats)> {
        let mut stats = MergeStats::default();
        let files = SegmentFiles::new(new_segment_id.0);

        // === Phase 1: merge postings + positions (streaming) ===
        let mut postings_writer = OffsetWriter::new(dir.streaming_writer(&files.postings).await?);
        let mut positions_writer = OffsetWriter::new(dir.streaming_writer(&files.positions).await?);
        let mut term_dict_writer = OffsetWriter::new(dir.streaming_writer(&files.term_dict).await?);

        let terms_processed = self
            .merge_postings(
                segments,
                &mut term_dict_writer,
                &mut postings_writer,
                &mut positions_writer,
                &mut stats,
            )
            .await?;
        stats.terms_processed = terms_processed;
        stats.postings_bytes = postings_writer.offset() as usize;
        stats.term_dict_bytes = term_dict_writer.offset() as usize;
        let positions_bytes = positions_writer.offset();

        postings_writer.finish()?;
        term_dict_writer.finish()?;
        if positions_bytes > 0 {
            positions_writer.finish()?;
        } else {
            drop(positions_writer);
            let _ = dir.delete(&files.positions).await;
        }

        // === Phase 2: merge store files (streaming) ===
        {
            let mut store_writer = OffsetWriter::new(dir.streaming_writer(&files.store).await?);
            {
                let mut store_merger = StoreMerger::new(&mut store_writer);
                for segment in segments {
                    if segment.store_has_dict() {
                        store_merger
                            .append_store_recompressing(segment.store())
                            .await
                            .map_err(crate::Error::Io)?;
                    } else {
                        let raw_blocks = segment.store_raw_blocks();
                        let data_slice = segment.store_data_slice();
                        store_merger.append_store(data_slice, &raw_blocks).await?;
                    }
                }
                store_merger.finish()?;
            }
            stats.store_bytes = store_writer.offset() as usize;
            store_writer.finish()?;
        }

        // === Dense vectors: strategy-dependent ===
        let vectors_bytes = match &dense_strategy {
            DenseVectorStrategy::MergeExisting => {
                self.merge_dense_vectors(dir, segments, &files).await?
            }
            DenseVectorStrategy::BuildAnn(trained) => {
                self.build_ann_vectors(dir, segments, &files, trained)
                    .await?
            }
        };
        stats.vectors_bytes = vectors_bytes;

        // === Mandatory: merge sparse vectors ===
        let sparse_bytes = self.merge_sparse_vectors(dir, segments, &files).await?;
        stats.sparse_bytes = sparse_bytes;

        // === Mandatory: merge field stats + write meta ===
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

        let label = match &dense_strategy {
            DenseVectorStrategy::MergeExisting => "Merge",
            DenseVectorStrategy::BuildAnn(_) => "ANN merge",
        };
        log::info!(
            "{} complete: {} docs, {} terms, term_dict={}, postings={}, store={}, vectors={}, sparse={}",
            label,
            total_docs,
            stats.terms_processed,
            format_bytes(stats.term_dict_bytes),
            format_bytes(stats.postings_bytes),
            format_bytes(stats.store_bytes),
            format_bytes(stats.vectors_bytes),
            format_bytes(stats.sparse_bytes),
        );

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
    async fn merge_postings(
        &self,
        segments: &[SegmentReader],
        term_dict: &mut OffsetWriter,
        postings_out: &mut OffsetWriter,
        positions_out: &mut OffsetWriter,
        stats: &mut MergeStats,
    ) -> Result<usize> {
        let doc_offs = doc_offsets(segments);

        // Bulk-prefetch all term dict blocks (1 I/O per segment instead of ~160)
        for (i, segment) in segments.iter().enumerate() {
            log::debug!("Prefetching term dict for segment {} ...", i);
            segment.prefetch_term_dict().await?;
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
                    doc_offset: doc_offs[seg_idx],
                });
            }
        }

        // Buffer term results - needed because SSTableWriter can't be held across await points
        // Memory is bounded by unique terms (typically much smaller than postings)
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
                    doc_offset: doc_offs[first.segment_idx],
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
                        doc_offset: doc_offs[entry.segment_idx],
                    });
                }
            }

            // Process this term (handles both single-source and multi-source)
            let term_info = self
                .merge_term(segments, &sources, postings_out, positions_out)
                .await?;

            term_results.push((current_key, term_info));
            terms_processed += 1;

            // Log progress every 100k terms
            if terms_processed.is_multiple_of(100_000) {
                log::debug!("Merge progress: {} terms processed", terms_processed);
            }
        }

        // Track memory (only term_results is buffered; postings/positions stream to disk)
        let results_mem = term_results.capacity() * std::mem::size_of::<(Vec<u8>, TermInfo)>();
        stats.current_memory_bytes = results_mem;
        stats.peak_memory_bytes = stats.peak_memory_bytes.max(stats.current_memory_bytes);

        log::info!(
            "[merge] complete: terms={}, segments={}, term_buffer={:.2} MB, postings={}, positions={}",
            terms_processed,
            segments.len(),
            results_mem as f64 / (1024.0 * 1024.0),
            format_bytes(postings_out.offset() as usize),
            format_bytes(positions_out.offset() as usize),
        );

        // Write to SSTable (sync, no await points)
        let mut writer = SSTableWriter::<TermInfo>::new(term_dict);
        for (key, term_info) in term_results {
            writer.insert(&key, &term_info)?;
        }
        writer.finish()?;

        Ok(terms_processed)
    }

    /// Merge a single term's postings + positions from one or more source segments.
    ///
    /// Fast path: when all sources are external and there are multiple,
    /// uses block-level concatenation (O(blocks) instead of O(postings)).
    /// Otherwise: full decode → remap doc IDs → re-encode.
    async fn merge_term(
        &self,
        segments: &[SegmentReader],
        sources: &[(usize, TermInfo, u32)],
        postings_out: &mut OffsetWriter,
        positions_out: &mut OffsetWriter,
    ) -> Result<TermInfo> {
        let mut sorted: Vec<_> = sources.to_vec();
        sorted.sort_by_key(|(_, _, off)| *off);

        let any_positions = sorted.iter().any(|(_, ti, _)| ti.position_info().is_some());
        let all_external = sorted.iter().all(|(_, ti, _)| ti.external_info().is_some());

        // === Merge postings ===
        let (posting_offset, posting_len, doc_count) = if all_external && sorted.len() > 1 {
            // Fast path: block-level concatenation
            let mut block_sources = Vec::with_capacity(sorted.len());
            for (seg_idx, ti, doc_off) in &sorted {
                let (off, len) = ti.external_info().unwrap();
                let bytes = segments[*seg_idx].read_postings(off, len).await?;
                let bpl = BlockPostingList::deserialize(&mut bytes.as_slice())?;
                block_sources.push((bpl, *doc_off));
            }
            let merged = BlockPostingList::concatenate_blocks(&block_sources)?;
            let offset = postings_out.offset();
            let mut buf = Vec::new();
            merged.serialize(&mut buf)?;
            postings_out.write_all(&buf)?;
            (offset, buf.len() as u32, merged.doc_count())
        } else {
            // Decode all sources into a flat PostingList, remap doc IDs
            let mut merged = PostingList::new();
            for (seg_idx, ti, doc_off) in &sorted {
                if let Some((ids, tfs)) = ti.decode_inline() {
                    for (id, tf) in ids.into_iter().zip(tfs) {
                        merged.add(id + doc_off, tf);
                    }
                } else {
                    let (off, len) = ti.external_info().unwrap();
                    let bytes = segments[*seg_idx].read_postings(off, len).await?;
                    let bpl = BlockPostingList::deserialize(&mut bytes.as_slice())?;
                    let mut it = bpl.iterator();
                    while it.doc() != TERMINATED {
                        merged.add(it.doc() + doc_off, it.term_freq());
                        it.advance();
                    }
                }
            }
            // Try to inline (only when no positions)
            if !any_positions {
                let ids: Vec<u32> = merged.iter().map(|p| p.doc_id).collect();
                let tfs: Vec<u32> = merged.iter().map(|p| p.term_freq).collect();
                if let Some(inline) = TermInfo::try_inline(&ids, &tfs) {
                    return Ok(inline);
                }
            }
            let offset = postings_out.offset();
            let block = BlockPostingList::from_posting_list(&merged)?;
            let mut buf = Vec::new();
            block.serialize(&mut buf)?;
            postings_out.write_all(&buf)?;
            (offset, buf.len() as u32, merged.doc_count())
        };

        // === Merge positions (if any source has them) ===
        if any_positions {
            let mut pos_sources = Vec::new();
            for (seg_idx, ti, doc_off) in &sorted {
                if let Some((pos_off, pos_len)) = ti.position_info()
                    && let Some(bytes) = segments[*seg_idx]
                        .read_position_bytes(pos_off, pos_len)
                        .await?
                {
                    let pl = PositionPostingList::deserialize(&mut bytes.as_slice())
                        .map_err(crate::Error::Io)?;
                    pos_sources.push((pl, *doc_off));
                }
            }
            if !pos_sources.is_empty() {
                let merged = PositionPostingList::concatenate_blocks(&pos_sources)
                    .map_err(crate::Error::Io)?;
                let offset = positions_out.offset();
                let mut buf = Vec::new();
                merged.serialize(&mut buf).map_err(crate::Error::Io)?;
                positions_out.write_all(&buf)?;
                return Ok(TermInfo::external_with_positions(
                    posting_offset,
                    posting_len,
                    doc_count,
                    offset,
                    buf.len() as u32,
                ));
            }
        }

        Ok(TermInfo::external(posting_offset, posting_len, doc_count))
    }

    /// Merge dense vector indexes - returns output size in bytes
    ///
    /// For ScaNN (IVF-PQ): O(1) merge by concatenating cluster data (same codebook)
    /// For IVF-RaBitQ: O(1) merge by concatenating cluster data (same centroids)
    /// For RaBitQ: Must rebuild with new centroid from all vectors
    async fn merge_dense_vectors<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
    ) -> Result<usize> {
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

                let doc_offs = doc_offsets(segments);

                match crate::structures::IVFPQIndex::merge(&refs, &doc_offs) {
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

                let doc_offs = doc_offsets(segments);

                match crate::structures::IVFRaBitQIndex::merge(&refs, &doc_offs) {
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

        write_vector_file(dir, files, field_indexes).await
    }

    /// Merge sparse vector indexes using block stacking
    ///
    /// This is O(blocks) instead of O(postings) - we stack blocks directly
    /// and only adjust the first_doc_id in each block header by the doc offset.
    /// Deltas within blocks remain unchanged since they're relative.
    async fn merge_sparse_vectors<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
    ) -> Result<usize> {
        use crate::structures::BlockSparsePostingList;
        use byteorder::{LittleEndian, WriteBytesExt};

        let doc_offs = doc_offsets(segments);
        for (i, seg) in segments.iter().enumerate() {
            log::debug!(
                "Sparse merge: segment {} has {} docs, doc_offset={}",
                i,
                seg.num_docs(),
                doc_offs[i]
            );
        }

        // Collect all sparse vector fields from schema
        let sparse_fields: Vec<_> = self
            .schema
            .fields()
            .filter(|(_, entry)| matches!(entry.field_type, FieldType::SparseVector))
            .map(|(field, entry)| (field, entry.sparse_vector_config.clone()))
            .collect();

        if sparse_fields.is_empty() {
            return Ok(0);
        }

        // Collect field data: (field_id, quantization, max_dim_id, dim_id -> merged_posting_list)
        type SparseFieldData = (
            u32,
            crate::structures::WeightQuantization,
            u32,
            FxHashMap<u32, Vec<u8>>,
        );
        let mut field_data: Vec<SparseFieldData> = Vec::new();

        for (field, sparse_config) in &sparse_fields {
            // Get quantization from config
            let quantization = sparse_config
                .as_ref()
                .map(|c| c.weight_quantization)
                .unwrap_or(crate::structures::WeightQuantization::Float32);

            // Collect all dimensions across all segments for this field
            let mut all_dims: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
            for segment in segments {
                if let Some(sparse_index) = segment.sparse_indexes().get(&field.0) {
                    for dim_id in sparse_index.active_dimensions() {
                        all_dims.insert(dim_id);
                    }
                }
            }

            if all_dims.is_empty() {
                continue;
            }

            // Bulk-read ALL posting lists per segment in one I/O call each.
            // This replaces 80K+ individual get_posting() calls with ~4 bulk reads.
            let mut segment_postings: Vec<FxHashMap<u32, Arc<BlockSparsePostingList>>> =
                Vec::with_capacity(segments.len());
            for (seg_idx, segment) in segments.iter().enumerate() {
                if let Some(sparse_index) = segment.sparse_indexes().get(&field.0) {
                    log::debug!(
                        "Sparse merge field {}: bulk-reading {} dims from segment {}",
                        field.0,
                        sparse_index.num_dimensions(),
                        seg_idx
                    );
                    let postings = sparse_index.read_all_postings_bulk().await?;
                    segment_postings.push(postings);
                } else {
                    segment_postings.push(FxHashMap::default());
                }
            }

            let mut dim_bytes: FxHashMap<u32, Vec<u8>> = FxHashMap::default();

            // Merge from in-memory data — no I/O in this loop
            for dim_id in all_dims {
                let mut posting_arcs: Vec<(Arc<BlockSparsePostingList>, u32)> = Vec::new();

                for (seg_idx, postings) in segment_postings.iter().enumerate() {
                    if let Some(posting_list) = postings.get(&dim_id) {
                        posting_arcs.push((Arc::clone(posting_list), doc_offs[seg_idx]));
                    }
                }

                if posting_arcs.is_empty() {
                    continue;
                }

                let lists_with_offsets: Vec<(&BlockSparsePostingList, u32)> = posting_arcs
                    .iter()
                    .map(|(pl, offset)| (pl.as_ref(), *offset))
                    .collect();

                let merged = BlockSparsePostingList::merge_with_offsets(&lists_with_offsets);

                let mut bytes = Vec::new();
                merged.serialize(&mut bytes).map_err(crate::Error::Io)?;
                dim_bytes.insert(dim_id, bytes);
            }

            // Drop bulk data before accumulating output
            drop(segment_postings);

            // Store num_dims (active count) instead of max_dim_id
            field_data.push((field.0, quantization, dim_bytes.len() as u32, dim_bytes));
        }

        if field_data.is_empty() {
            return Ok(0);
        }

        // Sort by field_id
        field_data.sort_by_key(|(id, _, _, _)| *id);

        // Compute header size and per-dimension offsets before writing
        // num_fields(u32) + per-field: field_id(u32) + quant(u8) + num_dims(u32)
        // per-dim: dim_id(u32) + offset(u64) + length(u32)
        let per_dim_entry = size_of::<u32>() + size_of::<u64>() + size_of::<u32>();
        let per_field_header = size_of::<u32>() + size_of::<u8>() + size_of::<u32>();
        let mut header_size = size_of::<u32>() as u64;
        for (_, _, num_dims, _) in &field_data {
            header_size += per_field_header as u64;
            header_size += (*num_dims as u64) * per_dim_entry as u64;
        }

        // Pre-compute offset tables (small — just dim_id + offset + length per dim)
        let mut current_offset = header_size;
        let mut field_tables: Vec<Vec<(u32, u64, u32)>> = Vec::new();
        for (_, _, _, dim_bytes) in &field_data {
            let mut table: Vec<(u32, u64, u32)> = Vec::with_capacity(dim_bytes.len());
            let mut dims: Vec<_> = dim_bytes.keys().copied().collect();
            dims.sort();
            for dim_id in dims {
                let bytes = &dim_bytes[&dim_id];
                table.push((dim_id, current_offset, bytes.len() as u32));
                current_offset += bytes.len() as u64;
            }
            field_tables.push(table);
        }

        // Stream header + tables + data directly to disk
        let mut writer = OffsetWriter::new(dir.streaming_writer(&files.sparse).await?);

        writer.write_u32::<LittleEndian>(field_data.len() as u32)?;
        for (i, (field_id, quantization, num_dims, _)) in field_data.iter().enumerate() {
            writer.write_u32::<LittleEndian>(*field_id)?;
            writer.write_u8(*quantization as u8)?;
            writer.write_u32::<LittleEndian>(*num_dims)?;
            for &(dim_id, offset, length) in &field_tables[i] {
                writer.write_u32::<LittleEndian>(dim_id)?;
                writer.write_u64::<LittleEndian>(offset)?;
                writer.write_u32::<LittleEndian>(length)?;
            }
        }

        // Stream posting data per-field, per-dimension (drop each after writing)
        for (_, _, _, dim_bytes) in field_data {
            let mut dims: Vec<_> = dim_bytes.keys().copied().collect();
            dims.sort();
            for dim_id in dims {
                writer.write_all(&dim_bytes[&dim_id])?;
            }
        }

        let output_size = writer.offset() as usize;
        writer.finish()?;

        log::info!(
            "Sparse vector merge complete: {} fields, {} bytes",
            field_tables.len(),
            output_size
        );

        Ok(output_size)
    }

    /// Merge segments and rebuild dense vectors with ANN indexes using trained structures.
    pub async fn merge_with_ann<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        new_segment_id: SegmentId,
        trained: &TrainedVectorStructures,
    ) -> Result<(SegmentMeta, MergeStats)> {
        self.merge_core(
            dir,
            segments,
            new_segment_id,
            DenseVectorStrategy::BuildAnn(trained),
        )
        .await
    }

    /// Build ANN indexes from Flat vectors using trained centroids
    async fn build_ann_vectors<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
        trained: &TrainedVectorStructures,
    ) -> Result<usize> {
        use crate::dsl::VectorIndexType;

        let mut field_indexes: Vec<(u32, u8, Vec<u8>)> = Vec::new();

        for (field, entry) in self.schema.fields() {
            if !matches!(entry.field_type, FieldType::DenseVector) || !entry.indexed {
                continue;
            }

            let config = match &entry.dense_vector_config {
                Some(c) => c,
                None => continue,
            };

            // Collect all Flat vectors from segments
            let mut all_vectors: Vec<Vec<f32>> = Vec::new();
            let mut all_doc_ids: Vec<(u32, u16)> = Vec::new();
            let mut doc_offset = 0u32;

            for segment in segments {
                if let Some(super::VectorIndex::Flat(flat_data)) =
                    segment.vector_indexes().get(&field.0)
                {
                    for (vec, (local_doc_id, ordinal)) in
                        flat_data.vectors.iter().zip(flat_data.doc_ids.iter())
                    {
                        all_vectors.push(vec.clone());
                        all_doc_ids.push((doc_offset + local_doc_id, *ordinal));
                    }
                }
                doc_offset += segment.num_docs();
            }

            if all_vectors.is_empty() {
                continue;
            }

            let dim = config.index_dim();

            // Extract just doc_ids for ANN indexes (they don't track ordinals yet)
            let ann_doc_ids: Vec<u32> = all_doc_ids.iter().map(|(doc_id, _)| *doc_id).collect();

            // Build ANN index based on index type and available trained structures
            match config.index_type {
                VectorIndexType::IvfRaBitQ => {
                    if let Some(centroids) = trained.centroids.get(&field.0) {
                        // Create RaBitQ codebook for the dimension
                        let rabitq_config = crate::structures::RaBitQConfig::new(dim);
                        let codebook = crate::structures::RaBitQCodebook::new(rabitq_config);

                        // Build IVF-RaBitQ index
                        let ivf_config = crate::structures::IVFRaBitQConfig::new(dim)
                            .with_store_raw(config.store_raw);
                        let ivf_index = crate::structures::IVFRaBitQIndex::build(
                            ivf_config,
                            centroids,
                            &codebook,
                            &all_vectors,
                            Some(&ann_doc_ids),
                        );

                        let index_data = super::builder::IVFRaBitQIndexData {
                            centroids: (**centroids).clone(),
                            codebook,
                            index: ivf_index,
                        };
                        let bytes = index_data
                            .to_bytes()
                            .map_err(|e| crate::Error::Serialization(e.to_string()))?;
                        field_indexes.push((field.0, 1u8, bytes)); // 1 = IVF-RaBitQ

                        log::info!(
                            "Built IVF-RaBitQ index for field {} with {} vectors",
                            field.0,
                            all_vectors.len()
                        );
                        continue;
                    }
                }
                VectorIndexType::ScaNN => {
                    if let (Some(centroids), Some(codebook)) = (
                        trained.centroids.get(&field.0),
                        trained.codebooks.get(&field.0),
                    ) {
                        // Build ScaNN (IVF-PQ) index
                        let ivf_pq_config = crate::structures::IVFPQConfig::new(dim);
                        let ivf_pq_index = crate::structures::IVFPQIndex::build(
                            ivf_pq_config,
                            centroids,
                            codebook,
                            &all_vectors,
                            Some(&ann_doc_ids),
                        );

                        let index_data = super::builder::ScaNNIndexData {
                            centroids: (**centroids).clone(),
                            codebook: (**codebook).clone(),
                            index: ivf_pq_index,
                        };
                        let bytes = index_data
                            .to_bytes()
                            .map_err(|e| crate::Error::Serialization(e.to_string()))?;
                        field_indexes.push((field.0, 2u8, bytes)); // 2 = ScaNN

                        log::info!(
                            "Built ScaNN index for field {} with {} vectors",
                            field.0,
                            all_vectors.len()
                        );
                        continue;
                    }
                }
                _ => {}
            }

            // Fallback: keep as Flat if no trained structures available
            let flat_data = super::builder::FlatVectorData {
                dim,
                vectors: all_vectors,
                doc_ids: all_doc_ids,
            };
            let bytes = flat_data.to_binary_bytes();
            field_indexes.push((field.0, 4u8, bytes)); // 4 = Flat Binary
        }

        write_vector_file(dir, files, field_indexes).await
    }
}

/// Write a vector index file with per-field header + data.
/// Streams header then each field's data directly to disk, avoiding a single
/// giant concatenation buffer.
async fn write_vector_file<D: Directory + DirectoryWriter>(
    dir: &D,
    files: &SegmentFiles,
    mut field_indexes: Vec<(u32, u8, Vec<u8>)>,
) -> Result<usize> {
    use byteorder::{LittleEndian, WriteBytesExt};

    if field_indexes.is_empty() {
        return Ok(0);
    }

    field_indexes.sort_by_key(|(id, _, _)| *id);

    let mut writer = OffsetWriter::new(dir.streaming_writer(&files.vectors).await?);

    // num_fields(u32) + per-field: field_id(u32) + index_type(u8) + offset(u64) + length(u64)
    let per_field_entry = size_of::<u32>() + size_of::<u8>() + size_of::<u64>() + size_of::<u64>();
    let header_size = size_of::<u32>() + field_indexes.len() * per_field_entry;
    writer.write_u32::<LittleEndian>(field_indexes.len() as u32)?;

    let mut current_offset = header_size as u64;
    for (field_id, index_type, data) in &field_indexes {
        writer.write_u32::<LittleEndian>(*field_id)?;
        writer.write_u8(*index_type)?;
        writer.write_u64::<LittleEndian>(current_offset)?;
        writer.write_u64::<LittleEndian>(data.len() as u64)?;
        current_offset += data.len() as u64;
    }

    // Stream each field's data (drop after writing)
    for (_, _, data) in field_indexes {
        writer.write_all(&data)?;
    }

    let output_size = writer.offset() as usize;
    writer.finish()?;
    Ok(output_size)
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
