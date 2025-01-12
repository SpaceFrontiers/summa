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

    /// Merge postings from multiple segments
    async fn merge_postings(
        &self,
        segments: &[SegmentReader],
        term_dict: &mut Vec<u8>,
        postings_out: &mut Vec<u8>,
    ) -> Result<()> {
        use std::collections::BTreeMap;

        // Collect all terms across all segments with doc ID remapping
        let mut all_terms: BTreeMap<Vec<u8>, PostingList> = BTreeMap::new();
        let mut doc_offset = 0u32;

        for segment in segments {
            // Iterate through all terms in this segment
            let terms = segment.all_terms().await?;
            for (key, term_info) in terms {
                // Read the posting list - handle both inline and external
                let source_postings = if let Some((doc_ids, term_freqs)) = term_info.decode_inline()
                {
                    // Inline posting list - no I/O needed
                    let mut pl = PostingList::with_capacity(doc_ids.len());
                    for (doc_id, tf) in doc_ids.into_iter().zip(term_freqs.into_iter()) {
                        pl.push(doc_id, tf);
                    }
                    BlockPostingList::from_posting_list(&pl)?
                } else {
                    // External posting list
                    let (offset, len) = term_info.external_info().unwrap();
                    let posting_bytes = segment.read_postings(offset, len).await?;
                    BlockPostingList::deserialize(&mut posting_bytes.as_slice())?
                };

                // Merge into combined posting list with remapped doc IDs
                let merged = all_terms.entry(key).or_default();
                let mut iter = source_postings.iterator();
                while iter.doc() != TERMINATED {
                    merged.add(iter.doc() + doc_offset, iter.term_freq());
                    iter.advance();
                }
            }
            doc_offset += segment.num_docs();
        }

        // Write merged term dictionary and postings
        let mut writer = SSTableWriter::<TermInfo>::new(term_dict);

        for (key, posting_list) in &all_terms {
            // Try to inline small posting lists
            let doc_ids: Vec<u32> = posting_list.iter().map(|p| p.doc_id).collect();
            let term_freqs: Vec<u32> = posting_list.iter().map(|p| p.term_freq).collect();

            let term_info = if let Some(inline) = TermInfo::try_inline(&doc_ids, &term_freqs) {
                // Small posting list - inline it directly
                inline
            } else {
                // Large posting list - write to external file
                let posting_offset = postings_out.len() as u64;
                let block_list = BlockPostingList::from_posting_list(posting_list)?;
                let mut encoded = Vec::new();
                block_list.serialize(&mut encoded)?;
                postings_out.extend_from_slice(&encoded);
                TermInfo::external(
                    posting_offset,
                    encoded.len() as u32,
                    posting_list.doc_count(),
                )
            };

            writer.insert(key, &term_info)?;
        }

        writer.finish()?;
        Ok(())
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
