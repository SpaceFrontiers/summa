//! Postings merge via streaming k-way merge.
//!
//! Uses a min-heap to merge terms from all segments in sorted order
//! without loading all terms into memory at once.
//!
//! Encoded posting and position blocks are copied directly for both single-
//! and multi-segment terms. Inline postings are promoted to tiny encoded
//! blocks only when the merged term no longer fits inline.

use super::OffsetWriter;
use super::SegmentMerger;
use super::chunk_maps::chunk_offsets;
use super::doc_offsets;
use crate::Result;
use crate::directories::OwnedBytes;
use crate::segment::reader::SegmentReader;
use crate::structures::{BlockPostingList, PositionStream, PostingList, SSTableWriter, TermInfo};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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
        // Chunked text fields key their postings by virtual chunk id, so their
        // terms stack with the field's chunk-count offsets, not document offsets.
        let chunk_offs = chunk_offsets(&self.schema, segments)?;
        let offset_for = |key: &[u8], segment_idx: usize| -> u32 {
            let field_id = u32::from_le_bytes([key[0], key[1], key[2], key[3]]);
            match chunk_offs.get(&field_id) {
                Some(offsets) => offsets[segment_idx],
                None => doc_offs[segment_idx],
            }
        };

        // Parallel prefetch all term dict blocks
        let prefetch_start = std::time::Instant::now();
        let mut futs = Vec::with_capacity(segments.len());
        for segment in segments.iter() {
            futs.push(segment.prefetch_term_dict());
        }
        let results = futures::future::join_all(futs).await;
        for (i, res) in results.into_iter().enumerate() {
            res.map_err(|e| {
                log::error!(
                    "[merge] index={} prefetch failed for segment {}: {}",
                    self.schema.index_label(),
                    i,
                    e
                );
                e
            })?;
        }
        log::debug!(
            "[merge] index={} prefetched {} term dicts in {:.1}s",
            self.schema.index_label(),
            segments.len(),
            prefetch_start.elapsed().as_secs_f64()
        );

        // Create iterators for each segment's term dictionary
        let mut iterators: Vec<_> = segments.iter().map(|s| s.term_dict_iter()).collect();

        // Initialize min-heap with first entry from each segment
        let mut heap: BinaryHeap<MergeEntry> = BinaryHeap::new();
        for (seg_idx, iter) in iterators.iter_mut().enumerate() {
            if let Some((key, term_info)) = iter.next().await.map_err(crate::Error::from)? {
                let doc_offset = offset_for(&key, seg_idx);
                heap.push(MergeEntry {
                    key,
                    term_info,
                    segment_idx: seg_idx,
                    doc_offset,
                });
            }
        }

        // Write SSTable entries inline — no buffering needed since
        // SSTableWriter<&mut OffsetWriter> is Send (OffsetWriter is Send).
        let mut term_dict_writer = SSTableWriter::<&mut OffsetWriter, TermInfo>::with_config(
            term_dict,
            crate::structures::SSTableWriterConfig::from_optimization(self.optimization),
        );
        let mut terms_processed = 0usize;
        // Pre-allocate sources buffer outside loop — reused for every term
        let mut sources: Vec<(usize, TermInfo, u32)> = Vec::with_capacity(segments.len());

        while !heap.is_empty() {
            self.ensure_not_cancelled()?;
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
                let doc_offset = offset_for(&key, first.segment_idx);
                heap.push(MergeEntry {
                    key,
                    term_info,
                    segment_idx: first.segment_idx,
                    doc_offset,
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
                    let doc_offset = offset_for(&key, entry.segment_idx);
                    heap.push(MergeEntry {
                        key,
                        term_info,
                        segment_idx: entry.segment_idx,
                        doc_offset,
                    });
                }
            }

            // Process this term (handles both single-source and multi-source)
            let term_info = self
                .merge_term(segments, &mut sources, postings_out, positions_out)
                .await?;

            // Write directly to SSTable (no buffering)
            term_dict_writer
                .insert(&current_key, &term_info)
                .map_err(crate::Error::Io)?;
            terms_processed += 1;

            // Log progress every 100k terms
            if terms_processed.is_multiple_of(100_000) {
                log::debug!(
                    "[merge] index={} progress: {} terms processed",
                    self.schema.index_label(),
                    terms_processed
                );
            }
        }

        term_dict_writer.finish().map_err(crate::Error::Io)?;

        Ok(terms_processed)
    }

    /// Merge a single term's postings + positions from one or more segments.
    ///
    /// Existing external posting and position blocks are always copied in
    /// their encoded form. Inline postings remain inline when the combined
    /// value fits; otherwise each tiny inline source is encoded as one block
    /// and concatenated with the untouched external blocks.
    pub(crate) async fn merge_term(
        &self,
        segments: &[SegmentReader],
        sources: &mut [(usize, TermInfo, u32)],
        postings_out: &mut OffsetWriter,
        positions_out: &mut OffsetWriter,
    ) -> Result<TermInfo> {
        sources.sort_by_key(|(_, _, off)| *off);

        let has_positions = sources
            .first()
            .is_some_and(|(_, info, _)| info.position_info().is_some());
        if sources
            .iter()
            .any(|(_, info, _)| info.position_info().is_some() != has_positions)
        {
            return Err(crate::Error::Corruption(
                "cannot merge a term with inconsistent position data".into(),
            ));
        }

        // Preserve genuinely tiny terms inline. Decoding here is bounded by
        // MAX_INLINE_POSTINGS and never touches an external posting list.
        if !has_positions
            && sources
                .iter()
                .all(|(_, info, _)| matches!(info, TermInfo::Inline { .. }))
        {
            let mut postings = Vec::new();
            for (_, info, doc_offset) in sources.iter() {
                let (ids, tfs) = info.decode_inline().expect("checked inline source");
                postings.extend(
                    ids.into_iter()
                        .zip(tfs)
                        .map(|(doc, tf)| (doc + doc_offset, tf)),
                );
            }
            if let Some(inline) =
                TermInfo::try_inline_iter(postings.len(), postings.iter().copied())
            {
                return Ok(inline);
            }
        }

        // Range reads return Arc/mmap-backed slices, so ordinary external
        // sources are not copied into an intermediate Vec. Reads still run in
        // parallel for lazy/remote directories.
        let read_futs: Vec<_> = sources
            .iter()
            .map(|(seg_idx, ti, _)| {
                let external = ti.external_info();
                let seg = &segments[*seg_idx];
                async move {
                    Ok::<_, crate::Error>(match external {
                        Some((off, len)) => Some(seg.read_postings(off, len).await?),
                        None => None,
                    })
                }
            })
            .collect();
        let external_sources: Vec<Option<OwnedBytes>> =
            futures::future::try_join_all(read_futs).await?;

        let mut posting_sources = Vec::with_capacity(sources.len());
        for ((_, info, doc_offset), external) in sources.iter().zip(external_sources) {
            let bytes = match external {
                Some(bytes) => bytes,
                None => {
                    let (ids, tfs) = info.decode_inline().ok_or_else(|| {
                        crate::Error::Corruption(
                            "term has neither inline nor external postings".into(),
                        )
                    })?;
                    let mut postings = PostingList::with_capacity(ids.len());
                    for (doc, tf) in ids.into_iter().zip(tfs) {
                        postings.push(doc, tf);
                    }
                    let block = BlockPostingList::from_posting_list_with_options(
                        &postings,
                        false,
                        None,
                        self.posting_codec,
                    )?;
                    let mut encoded = Vec::new();
                    block.serialize(&mut encoded)?;
                    OwnedBytes::new(encoded)
                }
            };
            if BlockPostingList::has_cursors_bytes(bytes.as_slice()) != has_positions {
                return Err(crate::Error::Corruption(
                    "posting position cursors do not match term position data".into(),
                ));
            }
            posting_sources.push((bytes, *doc_offset));
        }

        let posting_refs: Vec<_> = posting_sources
            .iter()
            .map(|(bytes, offset)| (bytes.as_slice(), *offset))
            .collect();
        let posting_offset = postings_out.offset();
        let (doc_count, posting_len) =
            BlockPostingList::concatenate_streaming(&posting_refs, postings_out)?;

        if has_positions {
            let pos_futs: Vec<_> = sources
                .iter()
                .map(|(seg_idx, ti, _)| {
                    let (pos_off, pos_len) = ti
                        .position_info()
                        .expect("position consistency checked above");
                    let seg = &segments[*seg_idx];
                    async move {
                        seg.read_position_bytes(pos_off, pos_len)
                            .await?
                            .ok_or_else(|| {
                                crate::Error::Corruption(
                                    "term has positions but the segment has no position file"
                                        .into(),
                                )
                            })
                    }
                })
                .collect();
            let position_sources = futures::future::try_join_all(pos_futs).await?;
            let position_refs: Vec<_> = position_sources
                .iter()
                .map(|bytes| bytes.as_slice())
                .collect();
            let position_offset = positions_out.offset();
            let (_, position_len) =
                PositionStream::concatenate_streaming(&position_refs, positions_out)?;
            return Ok(TermInfo::external_with_positions(
                posting_offset,
                posting_len as u64,
                doc_count,
                position_offset,
                position_len,
            ));
        }

        Ok(TermInfo::external(
            posting_offset,
            posting_len as u64,
            doc_count,
        ))
    }
}
