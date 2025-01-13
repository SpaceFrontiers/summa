//! Segment merger for combining multiple segments

use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::builder::{SegmentBuilder, SegmentBuilderConfig};
use super::reader::SegmentReader;
use super::store::StoreMerger;
use super::types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};
use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::Schema;
use crate::structures::{BlockPostingList, PostingList, SSTableWriter, TERMINATED, TermInfo};

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
        // Check if we can use optimized store stacking (no dictionaries)
        let can_stack_stores = segments.iter().all(|s| !s.store_has_dict());

        if can_stack_stores {
            self.merge_optimized(dir, segments, new_segment_id).await
        } else {
            self.merge_rebuild(dir, segments, new_segment_id).await
        }
    }

    /// Optimized merge: stack compressed store blocks, rebuild only term dict/postings
    async fn merge_optimized<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        new_segment_id: SegmentId,
    ) -> Result<SegmentMeta> {
        let files = SegmentFiles::new(new_segment_id.0);

        // Build merged term dictionary and postings (still need to rebuild these)
        let mut term_dict_data = Vec::new();
        let mut postings_data = Vec::new();
        self.merge_postings(segments, &mut term_dict_data, &mut postings_data)
            .await?;

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

        // Write files
        dir.write(&files.term_dict, &term_dict_data).await?;
        dir.write(&files.postings, &postings_data).await?;
        dir.write(&files.store, &store_data).await?;

        // Merge field stats
        let mut merged_field_stats: FxHashMap<u32, FieldStats> = FxHashMap::default();
        for segment in segments {
            for (&field_id, stats) in &segment.meta().field_stats {
                let entry = merged_field_stats.entry(field_id).or_default();
                entry.total_tokens += stats.total_tokens;
                entry.doc_count += stats.doc_count;
            }
        }

        let total_docs: u32 = segments.iter().map(|s| s.num_docs()).sum();
        let meta = SegmentMeta {
            id: new_segment_id.0,
            num_docs: total_docs,
            field_stats: merged_field_stats,
        };

        dir.write(&files.meta, &meta.serialize()?).await?;

        Ok(meta)
    }

    /// Fallback merge: rebuild everything from documents
    async fn merge_rebuild<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        new_segment_id: SegmentId,
    ) -> Result<SegmentMeta> {
        let mut builder =
            SegmentBuilder::new((*self.schema).clone(), SegmentBuilderConfig::default())?;

        for segment in segments {
            for doc_id in 0..segment.num_docs() {
                if let Some(doc) = segment.doc(doc_id).await? {
                    builder.add_document(doc)?;
                }
            }
        }

        builder.build(dir, new_segment_id).await
    }

    /// Merge postings from multiple segments using hybrid stacking
    ///
    /// Optimization: For terms that exist in only one segment, we copy the
    /// posting data directly without decode/encode. Only terms that exist
    /// in multiple segments need full merge.
    async fn merge_postings(
        &self,
        segments: &[SegmentReader],
        term_dict: &mut Vec<u8>,
        postings_out: &mut Vec<u8>,
    ) -> Result<()> {
        use std::collections::BTreeMap;

        // Phase 1: Collect all terms and track which segments contain each term
        // Key -> Vec<(segment_index, term_info, doc_offset)>
        let mut term_sources: BTreeMap<Vec<u8>, Vec<(usize, TermInfo, u32)>> = BTreeMap::new();
        let mut doc_offset = 0u32;

        for (seg_idx, segment) in segments.iter().enumerate() {
            let terms = segment.all_terms().await?;
            for (key, term_info) in terms {
                term_sources
                    .entry(key)
                    .or_default()
                    .push((seg_idx, term_info, doc_offset));
            }
            doc_offset += segment.num_docs();
        }

        // Phase 2: Process terms - collect all results first (async)
        // This avoids holding SSTableWriter across await points
        let mut term_results: Vec<(Vec<u8>, TermInfo)> = Vec::with_capacity(term_sources.len());

        for (key, sources) in term_sources {
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

            term_results.push((key, term_info));
        }

        // Phase 3: Write to SSTable (sync, no await points)
        let mut writer = SSTableWriter::<TermInfo>::new(term_dict);
        for (key, term_info) in term_results {
            writer.insert(&key, &term_info)?;
        }
        writer.finish()?;

        Ok(())
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
    Ok(())
}
