//! Segment merger for combining multiple segments

mod dense_vectors;
mod sparse_vectors;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::Write;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::reader::SegmentReader;
use super::store::StoreMerger;
use super::types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};
use crate::Result;
use crate::directories::{Directory, DirectoryWriter, StreamingWriter};
use crate::dsl::Schema;
use crate::structures::{
    BlockPostingList, PositionPostingList, PostingList, SSTableWriter, TERMINATED, TermInfo,
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
#[derive(Clone)]
pub struct TrainedVectorStructures {
    /// Trained centroids per field_id
    pub centroids: rustc_hash::FxHashMap<u32, Arc<crate::structures::CoarseCentroids>>,
    /// Trained PQ codebooks per field_id (for ScaNN)
    pub codebooks: rustc_hash::FxHashMap<u32, Arc<crate::structures::PQCodebook>>,
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
    ///
    /// If `trained` is provided, dense vectors use O(1) cluster merge when possible
    /// (homogeneous IVF/ScaNN), otherwise rebuilds ANN from trained structures.
    /// Without trained structures, only flat vectors are merged.
    ///
    /// Uses streaming writers so postings, positions, and store data flow directly
    /// to files instead of buffering everything in memory. Only the term dictionary
    /// (compact key+TermInfo entries) is buffered.
    pub async fn merge<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        new_segment_id: SegmentId,
        trained: Option<&TrainedVectorStructures>,
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

        // === Dense vectors ===
        let vectors_bytes = self
            .merge_dense_vectors(dir, segments, &files, trained)
            .await?;
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

        let label = if trained.is_some() {
            "ANN merge"
        } else {
            "Merge"
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
        let mut serialize_buf: Vec<u8> = Vec::new();

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
                .merge_term(
                    segments,
                    &sources,
                    postings_out,
                    positions_out,
                    &mut serialize_buf,
                )
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
        stats.peak_memory_bytes = stats.peak_memory_bytes.max(results_mem);

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
        buf: &mut Vec<u8>,
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
            buf.clear();
            merged.serialize(buf)?;
            postings_out.write_all(buf)?;
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
            buf.clear();
            block.serialize(buf)?;
            postings_out.write_all(buf)?;
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
                buf.clear();
                merged.serialize(buf).map_err(crate::Error::Io)?;
                positions_out.write_all(buf)?;
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
    let _ = dir.delete(&files.sparse).await;
    let _ = dir.delete(&files.positions).await;
    Ok(())
}
