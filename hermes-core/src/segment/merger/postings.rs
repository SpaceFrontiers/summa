//! Postings merge via streaming k-way merge.
//!
//! Uses a min-heap to merge terms from all segments in sorted order
//! without loading all terms into memory at once.
//!
//! Optimization: For terms that exist in only one segment, we copy the
//! posting data directly without decode/encode. Only terms that exist
//! in multiple segments need full merge.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::Write;

use super::OffsetWriter;
use super::SegmentMerger;
use super::doc_offsets;
use crate::Result;
use crate::segment::reader::SegmentReader;
use crate::structures::{
    BlockPostingList, PositionPostingList, PostingList, SSTableWriter, TERMINATED, TermInfo,
};

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

impl SegmentMerger {
    /// Merge postings from multiple segments using streaming k-way merge
    ///
    /// SSTable entries are written inline during the merge loop (no buffering).
    /// This is possible because SSTableWriter<W> is Send when W is Send.
    ///
    /// Returns the number of terms processed.
    pub(super) async fn merge_postings(
        &self,
        segments: &[SegmentReader],
        term_dict: &mut OffsetWriter,
        postings_out: &mut OffsetWriter,
        positions_out: &mut OffsetWriter,
    ) -> Result<usize> {
        let doc_offs = doc_offsets(segments)?;

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
        // Pre-allocate sources buffer outside loop — reused for every term
        let mut sources: Vec<(usize, TermInfo, u32)> = Vec::with_capacity(segments.len());

        while !heap.is_empty() {
            // Get the smallest key (move, not clone)
            let first = heap.pop().unwrap();
            let current_key = first.key;

            // Collect all entries with the same key
            sources.clear();
            sources.push((first.segment_idx, first.term_info, first.doc_offset));

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
                    &mut sources,
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
        sources: &mut [(usize, TermInfo, u32)],
        postings_out: &mut OffsetWriter,
        positions_out: &mut OffsetWriter,
        buf: &mut Vec<u8>,
    ) -> Result<TermInfo> {
        sources.sort_by_key(|(_, _, off)| *off);

        let any_positions = sources
            .iter()
            .any(|(_, ti, _)| ti.position_info().is_some());
        let all_external = sources
            .iter()
            .all(|(_, ti, _)| ti.external_info().is_some());

        // === Merge postings ===
        let (posting_offset, posting_len, doc_count) = if all_external && sources.len() > 1 {
            // Fast path: streaming merge (blocks → output writer, no buffering)
            // Read all segments' postings in parallel
            let read_futs: Vec<_> = sources
                .iter()
                .map(|(seg_idx, ti, doc_off)| {
                    let (off, len) = ti.external_info().unwrap();
                    let seg = &segments[*seg_idx];
                    let doc_off = *doc_off;
                    async move {
                        let bytes = seg.read_postings(off, len).await?;
                        Ok::<_, crate::Error>((bytes, doc_off))
                    }
                })
                .collect();
            let raw_sources: Vec<(Vec<u8>, u32)> = futures::future::try_join_all(read_futs).await?;
            let refs: Vec<(&[u8], u32)> = raw_sources
                .iter()
                .map(|(b, off)| (b.as_slice(), *off))
                .collect();
            let offset = postings_out.offset();
            let (doc_count, bytes_written) =
                BlockPostingList::concatenate_streaming(&refs, postings_out)?;
            (offset, bytes_written as u64, doc_count)
        } else {
            // Decode all sources into a flat PostingList, remap doc IDs
            let mut merged = PostingList::new();
            for (seg_idx, ti, doc_off) in sources.iter() {
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
            if !any_positions
                && let Some(inline) = TermInfo::try_inline_iter(
                    merged.doc_count() as usize,
                    merged.iter().map(|p| (p.doc_id, p.term_freq)),
                )
            {
                return Ok(inline);
            }
            let offset = postings_out.offset();
            let block = BlockPostingList::from_posting_list(&merged)?;
            buf.clear();
            block.serialize(buf)?;
            postings_out.write_all(buf)?;
            (offset, buf.len() as u64, merged.doc_count())
        };

        // === Merge positions (if any source has them) ===
        if any_positions {
            // Read all position data in parallel
            let pos_futs: Vec<_> = sources
                .iter()
                .filter_map(|(seg_idx, ti, doc_off)| {
                    let (pos_off, pos_len) = ti.position_info()?;
                    let seg = &segments[*seg_idx];
                    let doc_off = *doc_off;
                    Some(async move {
                        match seg.read_position_bytes(pos_off, pos_len).await? {
                            Some(bytes) => Ok::<_, crate::Error>(Some((bytes, doc_off))),
                            None => Ok(None),
                        }
                    })
                })
                .collect();
            let raw_pos: Vec<(Vec<u8>, u32)> = futures::future::try_join_all(pos_futs)
                .await?
                .into_iter()
                .flatten()
                .collect();
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
                    bytes_written as u64,
                ));
            }
        }

        Ok(TermInfo::external(posting_offset, posting_len, doc_count))
    }
}
