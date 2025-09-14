//! Postings merge via streaming k-way merge.
//!
//! Uses a min-heap to merge terms from all segments in sorted order
//! without loading all terms into memory at once.
//!
//! Encoded posting and position blocks are copied directly for both single-
//! and multi-segment terms. Inline postings are promoted to tiny encoded
//! blocks only when the merged term no longer fits inline.

use super::MergedTerms;
use super::OffsetWriter;
use super::SegmentMerger;
use super::chunk_maps::chunk_offsets;
use super::doc_offsets;
use crate::Result;
use crate::directories::OwnedBytes;
use crate::segment::reader::SegmentReader;
use crate::structures::{BlockPostingList, PositionStream, PostingList, SSTableWriter, TermInfo};

/// Outcome of one postings merge pass.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PostingMergeStats {
    /// Terms written to the merged dictionary.
    pub terms_processed: usize,
    /// Blocks written into a bounded list with an unknown ratio or impact
    /// record: legacy external blocks copied next to bounded sources, and
    /// promoted inline blocks that cannot carry an impact envelope. See
    /// [`super::MergeStats::posting_blocks_without_bounds`].
    pub blocks_without_bounds: usize,
}

/// Bound metadata carried by the external sources of one merged term.
#[derive(Debug, Default, Clone, Copy)]
struct SourceBounds {
    ratio: bool,
    impact: bool,
}

impl SegmentMerger {
    /// Merge postings from multiple segments using streaming k-way merge
    ///
    /// SSTable entries are written inline during the merge loop (no buffering).
    /// This is possible because SSTableWriter<W> is Send when W is Send.
    ///
    /// Returns the number of terms processed and the legacy-block counter.
    pub(super) async fn merge_postings(
        &self,
        segments: &[SegmentReader],
        term_dict: &mut OffsetWriter,
        postings_out: &mut OffsetWriter,
        positions_out: &mut OffsetWriter,
        plans: &[crate::segment::text_reorder::TextReorderPlan],
    ) -> Result<PostingMergeStats> {
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

        // Warm bounded initial dictionary ranges with bounded I/O fan-out.
        let prefetch_start = std::time::Instant::now();
        for (batch, group) in segments
            .chunks(crate::structures::AsyncSSTableReader::<TermInfo>::PREFETCH_LEADING_FANOUT)
            .enumerate()
        {
            self.ensure_not_cancelled()?;
            let mut reads = Vec::with_capacity(group.len());
            for segment in group {
                reads.push(segment.prefetch_term_dict());
            }
            // Drain all started reads before an error releases source/output
            // ownership. Later batches are not admitted after failure.
            for (offset, result) in futures::future::join_all(reads)
                .await
                .into_iter()
                .enumerate()
            {
                result.map_err(|error| {
                    log::error!(
                        "[merge] index={} prefetch failed for segment {}: {}",
                        self.schema.index_label(),
                        batch * crate::structures::AsyncSSTableReader::<TermInfo>::PREFETCH_LEADING_FANOUT + offset,
                        error
                    );
                    error
                })?;
            }
        }
        log::debug!(
            "[merge] index={} prefetched {} term dicts in {:.1}s",
            self.schema.index_label(),
            segments.len(),
            prefetch_start.elapsed().as_secs_f64()
        );

        let mut terms = MergedTerms::new(segments, self.cancellation.as_deref()).await?;

        // Write SSTable entries inline — no buffering needed since
        // SSTableWriter<&mut OffsetWriter> is Send (OffsetWriter is Send).
        let mut term_dict_writer = SSTableWriter::<&mut OffsetWriter, TermInfo>::with_config(
            term_dict,
            crate::structures::SSTableWriterConfig {
                block_size: self.term_dict_block_size,
                ..crate::structures::SSTableWriterConfig::from_optimization(self.optimization)
            },
        );
        let mut stats = PostingMergeStats::default();
        // Pre-allocate sources buffer outside loop — reused for every term
        let mut sources: Vec<(usize, TermInfo, u32)> = Vec::with_capacity(segments.len());

        while let Some(current_key) = terms.next(&mut sources).await? {
            self.ensure_not_cancelled()?;
            for (source, _, offset) in &mut sources {
                *offset = offset_for(&current_key, *source);
            }

            // Process this term (handles both single-source and multi-source)
            let field = crate::Field(u32::from_le_bytes([
                current_key[0],
                current_key[1],
                current_key[2],
                current_key[3],
            ]));
            let term_info = if let Some(plan) = plans.iter().find(|plan| plan.field == field) {
                crate::segment::text_reorder::reorder_term(
                    segments,
                    &sources,
                    plan,
                    postings_out,
                    positions_out,
                    self.posting_codec,
                    self.bp_memory_budget
                        .saturating_sub(crate::segment::text_reorder::plan_bytes(plans)),
                    self.cancellation.as_deref(),
                )
                .await?
            } else {
                self.merge_term(
                    segments,
                    field,
                    &mut sources,
                    postings_out,
                    positions_out,
                    &mut stats,
                )
                .await?
            };

            // Write directly to SSTable (no buffering)
            term_dict_writer
                .insert(&current_key, &term_info)
                .map_err(crate::Error::Io)?;
            stats.terms_processed += 1;

            // Log progress every 100k terms
            if stats.terms_processed.is_multiple_of(100_000) {
                log::debug!(
                    "[merge] index={} progress: {} terms processed",
                    self.schema.index_label(),
                    stats.terms_processed
                );
            }
        }

        term_dict_writer.finish().map_err(crate::Error::Io)?;

        if stats.blocks_without_bounds > 0 {
            log::info!(
                "[merge] index={} {} posting blocks merged without ratio/impact metadata \
                 (legacy sources); their L1 groups keep an unknown bound until the \
                 data is rebuilt with posting_ratio_bounds/posting_impact_bounds",
                self.schema.index_label(),
                stats.blocks_without_bounds
            );
        }

        Ok(stats)
    }

    /// Merge a single term's postings + positions from one or more segments.
    ///
    /// Existing external posting and position blocks are always copied in
    /// their encoded form. Inline postings remain inline when the combined
    /// value fits; otherwise each tiny inline source is encoded as one block
    /// and concatenated with the untouched external blocks. A promoted block
    /// carries the same ratio/impact metadata as the external sources it
    /// joins, computed from its own segment's scoring lengths, so it never
    /// zeroes the bound of its L1 group. `field` identifies the length source
    /// (`key[0..4]` of the dictionary entry).
    pub(crate) async fn merge_term(
        &self,
        segments: &[SegmentReader],
        field: crate::Field,
        sources: &mut [(usize, TermInfo, u32)],
        postings_out: &mut OffsetWriter,
        positions_out: &mut OffsetWriter,
        stats: &mut PostingMergeStats,
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

        // Single external sources are copied verbatim and keep whatever
        // metadata they have. Multi-source terms decide the output layout
        // from every external footer: `concatenate_streaming` emits ratio /
        // impact records when any source has them, and a source without
        // them contributes an unknown (zero) record to its L1 group.
        let bounds = if sources.len() > 1 {
            Self::source_bounds(&external_sources, stats)?
        } else {
            SourceBounds::default()
        };

        let mut posting_sources = Vec::with_capacity(sources.len());
        for ((seg_idx, info, doc_offset), external) in sources.iter().zip(external_sources) {
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
                    let block = Self::encode_promoted_inline(
                        &segments[*seg_idx],
                        field,
                        &postings,
                        bounds,
                        self.posting_codec,
                    )?;
                    // Impact envelopes exist only for multi-block lists, so a
                    // promoted block joins an impact list with an unknown
                    // record (its ratio bound still applies). Keep that visible.
                    if bounds.impact && !block.has_impact_bounds() {
                        stats.blocks_without_bounds += 1;
                    }
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

    /// Bound metadata of a multi-source term's external sources. Sources
    /// without metadata that join a bounded list are counted as legacy blocks.
    /// Parsing validates only the directory (no block decoding).
    fn source_bounds(
        external_sources: &[Option<OwnedBytes>],
        stats: &mut PostingMergeStats,
    ) -> Result<SourceBounds> {
        let mut bounds = SourceBounds::default();
        let mut unbounded_blocks = 0usize;
        for bytes in external_sources.iter().flatten() {
            let list = BlockPostingList::deserialize_zero_copy(bytes.clone()).map_err(|error| {
                crate::Error::Corruption(format!("invalid posting source: {error}"))
            })?;
            bounds.ratio |= list.has_ratio_bounds();
            bounds.impact |= list.has_impact_bounds();
            if !list.has_ratio_bounds() && !list.has_impact_bounds() {
                unbounded_blocks += list.num_blocks();
            }
        }
        if bounds.ratio || bounds.impact {
            stats.blocks_without_bounds += unbounded_blocks;
        }
        Ok(bounds)
    }

    /// Encode one promoted inline source with the layout of the list it joins.
    ///
    /// Lengths come from the source segment (local ids, before the merge
    /// offset is applied). Raw chunk/document lengths mirror the segment
    /// builder: they never exceed the effective scoring length, whatever
    /// chunk-length floor the merged segment derives, so the bound stays valid.
    fn encode_promoted_inline(
        segment: &SegmentReader,
        field: crate::Field,
        postings: &PostingList,
        bounds: SourceBounds,
        codec: crate::structures::PostingCodec,
    ) -> Result<BlockPostingList> {
        let length_of = |id: u32| -> u32 {
            segment.chunk_map(field).map_or_else(
                || segment.doc_lengths(field).map_or(0, |norm| norm.length(id)),
                |map| map.length(id),
            )
        };
        let list = if bounds.impact {
            BlockPostingList::from_posting_list_with_impact_bounds(
                postings,
                false,
                Some(&length_of),
                codec,
            )?
        } else if bounds.ratio {
            BlockPostingList::from_posting_list_with_ratio_bounds(
                postings,
                false,
                Some(&length_of),
                codec,
            )?
        } else {
            BlockPostingList::from_posting_list_with_options(postings, false, None, codec)?
        };
        Ok(list)
    }
}
