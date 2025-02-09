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

// TrainedVectorStructures is defined in super::types (available on all platforms)
pub use super::types::TrainedVectorStructures;

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
        let phase1_start = std::time::Instant::now();
        let mut postings_writer = OffsetWriter::new(dir.streaming_writer(&files.postings).await?);
        let mut positions_writer = OffsetWriter::new(dir.streaming_writer(&files.positions).await?);
        let mut term_dict_writer = OffsetWriter::new(dir.streaming_writer(&files.term_dict).await?);

        let terms_processed = self
            .merge_postings(
                segments,
                &mut term_dict_writer,
                &mut postings_writer,
                &mut positions_writer,
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
        log::info!(
            "[merge] postings done: {} terms, term_dict={}, postings={}, positions={} in {:.1}s",
            terms_processed,
            format_bytes(stats.term_dict_bytes),
            format_bytes(stats.postings_bytes),
            format_bytes(positions_bytes as usize),
            phase1_start.elapsed().as_secs_f64()
        );

        // === Phase 2: merge store files (streaming) ===
        let phase2_start = std::time::Instant::now();
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
        log::info!(
            "[merge] store done: {} in {:.1}s",
            format_bytes(stats.store_bytes),
            phase2_start.elapsed().as_secs_f64()
        );

        // === Phase 3: Dense vectors ===
        let phase3_start = std::time::Instant::now();
        let vectors_bytes = self
            .merge_dense_vectors(dir, segments, &files, trained)
            .await?;
        stats.vectors_bytes = vectors_bytes;
        log::info!(
            "[merge] dense vectors done: {} in {:.1}s",
            format_bytes(stats.vectors_bytes),
            phase3_start.elapsed().as_secs_f64()
        );

        // === Phase 4: merge sparse vectors ===
        let phase4_start = std::time::Instant::now();
        let sparse_bytes = self.merge_sparse_vectors(dir, segments, &files).await?;
        stats.sparse_bytes = sparse_bytes;
        log::info!(
            "[merge] sparse vectors done: {} in {:.1}s",
            format_bytes(stats.sparse_bytes),
            phase4_start.elapsed().as_secs_f64()
        );

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
    ///
    /// Optimization: For terms that exist in only one segment, we copy the
    /// posting data directly without decode/encode. Only terms that exist
    /// in multiple segments need full merge.
    ///
    /// SSTable entries are written inline during the merge loop (no buffering).
    /// This is possible because SSTableWriter<W> is Send when W is Send.
    ///
    /// Returns the number of terms processed.
    async fn merge_postings(
        &self,
        segments: &[SegmentReader],
        term_dict: &mut OffsetWriter,
        postings_out: &mut OffsetWriter,
        positions_out: &mut OffsetWriter,
    ) -> Result<usize> {
        let doc_offs = doc_offsets(segments);

        // Parallel prefetch all term dict blocks
        let prefetch_start = std::time::Instant::now();
        let mut futs = Vec::with_capacity(segments.len());
        for segment in segments.iter() {
            futs.push(segment.prefetch_term_dict());
        }
        let results = futures::future::join_all(futs).await;
        for (i, res) in results.into_iter().enumerate() {
            res.map_err(|e| {
                log::error!("Prefetch failed for segment {}: {}", i, e);
                e
            })?;
        }
        log::debug!(
            "Prefetched {} term dicts in {:.1}s",
            segments.len(),
            prefetch_start.elapsed().as_secs_f64()
        );

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

        // Write SSTable entries inline — no buffering needed since
        // SSTableWriter<&mut OffsetWriter> is Send (OffsetWriter is Send).
        let mut term_dict_writer = SSTableWriter::<&mut OffsetWriter, TermInfo>::new(term_dict);
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

            // Write directly to SSTable (no buffering)
            term_dict_writer
                .insert(&current_key, &term_info)
                .map_err(crate::Error::Io)?;
            terms_processed += 1;

            // Log progress every 100k terms
            if terms_processed.is_multiple_of(100_000) {
                log::debug!("Merge progress: {} terms processed", terms_processed);
            }
        }

        term_dict_writer.finish().map_err(crate::Error::Io)?;

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
            // Fast path: streaming merge (blocks → output writer, no buffering)
            let mut raw_sources: Vec<(Vec<u8>, u32)> = Vec::with_capacity(sorted.len());
            for (seg_idx, ti, doc_off) in &sorted {
                let (off, len) = ti.external_info().unwrap();
                let bytes = segments[*seg_idx].read_postings(off, len).await?;
                raw_sources.push((bytes, *doc_off));
            }
            let refs: Vec<(&[u8], u32)> = raw_sources
                .iter()
                .map(|(b, off)| (b.as_slice(), *off))
                .collect();
            let offset = postings_out.offset();
            let (doc_count, bytes_written) =
                BlockPostingList::concatenate_streaming(&refs, postings_out)?;
            (offset, bytes_written as u32, doc_count)
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
                    let bpl = BlockPostingList::deserialize(&bytes)?;
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
            let mut raw_pos: Vec<(Vec<u8>, u32)> = Vec::new();
            for (seg_idx, ti, doc_off) in &sorted {
                if let Some((pos_off, pos_len)) = ti.position_info()
                    && let Some(bytes) = segments[*seg_idx]
                        .read_position_bytes(pos_off, pos_len)
                        .await?
                {
                    raw_pos.push((bytes, *doc_off));
                }
            }
            if !raw_pos.is_empty() {
                let refs: Vec<(&[u8], u32)> = raw_pos
                    .iter()
                    .map(|(b, off)| (b.as_slice(), *off))
                    .collect();
                let offset = positions_out.offset();
                let (_doc_count, bytes_written) =
                    PositionPostingList::concatenate_streaming(&refs, positions_out)
                        .map_err(crate::Error::Io)?;
                return Ok(TermInfo::external_with_positions(
                    posting_offset,
                    posting_len,
                    doc_count,
                    offset,
                    bytes_written as u32,
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
