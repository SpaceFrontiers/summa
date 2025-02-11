//! Postings and positions streaming build.
//!
//! Includes in-memory posting builder types (TermKey, CompactPosting,
//! PostingListBuilder, PositionPostingListBuilder) and the streaming
//! serialization functions that flush them to disk.

use std::io::Write;
use std::mem::size_of;

use hashbrown::HashMap;
use lasso::{Rodeo, Spur};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::structures::{PostingList, SSTableWriter, TermInfo};
use crate::{DocId, Result};

// ============================================================================
// In-memory posting builder types
// ============================================================================

/// Interned term key combining field and term
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct TermKey {
    pub field: u32,
    pub term: Spur,
}

/// Compact posting entry for in-memory storage
#[derive(Clone, Copy)]
pub(super) struct CompactPosting {
    pub doc_id: DocId,
    pub term_freq: u16,
}

/// In-memory posting list for a term
pub(super) struct PostingListBuilder {
    /// In-memory postings
    pub postings: Vec<CompactPosting>,
}

impl PostingListBuilder {
    pub fn new() -> Self {
        Self {
            postings: Vec::with_capacity(4),
        }
    }

    /// Add a posting, merging if same doc_id as last
    #[inline]
    pub fn add(&mut self, doc_id: DocId, term_freq: u32) {
        // Check if we can merge with the last posting
        if let Some(last) = self.postings.last_mut()
            && last.doc_id == doc_id
        {
            last.term_freq = last.term_freq.saturating_add(term_freq as u16);
            return;
        }
        self.postings.push(CompactPosting {
            doc_id,
            term_freq: term_freq.min(u16::MAX as u32) as u16,
        });
    }

    pub fn len(&self) -> usize {
        self.postings.len()
    }
}

/// In-memory position posting list for a term (for fields with record_positions=true)
pub(super) struct PositionPostingListBuilder {
    /// Doc ID -> list of positions (encoded as element_ordinal << 20 | token_position)
    pub postings: Vec<(DocId, Vec<u32>)>,
}

impl PositionPostingListBuilder {
    pub fn new() -> Self {
        Self {
            postings: Vec::new(),
        }
    }

    /// Add a position for a document
    #[inline]
    pub fn add_position(&mut self, doc_id: DocId, position: u32) {
        if let Some((last_doc, positions)) = self.postings.last_mut()
            && *last_doc == doc_id
        {
            positions.push(position);
            return;
        }
        let mut positions = Vec::with_capacity(4);
        positions.push(position);
        self.postings.push((doc_id, positions));
    }
}

/// Intermediate result for parallel posting serialization
pub(super) enum SerializedPosting {
    /// Inline posting (small enough to fit in TermInfo)
    Inline(TermInfo),
    /// External posting with serialized bytes
    External { bytes: Vec<u8>, doc_count: u32 },
}

// ============================================================================
// Streaming build functions
// ============================================================================

/// Stream postings directly to disk.
///
/// Parallel serialization of posting lists, then sequential streaming of
/// term dict and postings data directly to writers (no Vec<u8> accumulation).
pub(super) fn build_postings_streaming(
    inverted_index: HashMap<TermKey, PostingListBuilder>,
    term_interner: Rodeo,
    position_offsets: &FxHashMap<Vec<u8>, (u64, u64)>,
    term_dict_writer: &mut dyn Write,
    postings_writer: &mut dyn Write,
) -> Result<()> {
    // Phase 1: Consume HashMap into sorted Vec (frees HashMap overhead)
    let mut term_entries: Vec<(Vec<u8>, PostingListBuilder)> = inverted_index
        .into_iter()
        .map(|(term_key, posting_list)| {
            let term_str = term_interner.resolve(&term_key.term);
            let mut key = Vec::with_capacity(4 + term_str.len());
            key.extend_from_slice(&term_key.field.to_le_bytes());
            key.extend_from_slice(term_str.as_bytes());
            (key, posting_list)
        })
        .collect();

    drop(term_interner);

    term_entries.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));

    // Phase 2: Parallel serialization
    // For inline-eligible terms (no positions, few postings), extract doc_ids/tfs
    // directly from CompactPosting without creating an intermediate PostingList.
    let serialized: Vec<(Vec<u8>, SerializedPosting)> = term_entries
        .into_par_iter()
        .map(|(key, posting_builder)| {
            let has_positions = position_offsets.contains_key(&key);

            // Fast path: try inline first (avoids PostingList + BlockPostingList allocs)
            // Uses try_inline_iter to avoid allocating two Vec<u32> per term.
            if !has_positions
                && let Some(inline) = TermInfo::try_inline_iter(
                    posting_builder.postings.len(),
                    posting_builder
                        .postings
                        .iter()
                        .map(|p| (p.doc_id, p.term_freq as u32)),
                )
            {
                return Ok((key, SerializedPosting::Inline(inline)));
            }

            // Slow path: build full PostingList → BlockPostingList → serialize
            let mut full_postings = PostingList::with_capacity(posting_builder.len());
            for p in &posting_builder.postings {
                full_postings.push(p.doc_id, p.term_freq as u32);
            }

            let mut posting_bytes = Vec::new();
            let block_list =
                crate::structures::BlockPostingList::from_posting_list(&full_postings)?;
            block_list.serialize(&mut posting_bytes)?;
            let result = SerializedPosting::External {
                bytes: posting_bytes,
                doc_count: full_postings.doc_count(),
            };

            Ok((key, result))
        })
        .collect::<Result<Vec<_>>>()?;

    // Phase 3: Stream directly to writers (no intermediate Vec<u8> accumulation)
    let mut postings_offset = 0u64;
    let mut writer = SSTableWriter::<_, TermInfo>::new(term_dict_writer);

    for (key, serialized_posting) in serialized {
        let term_info = match serialized_posting {
            SerializedPosting::Inline(info) => info,
            SerializedPosting::External { bytes, doc_count } => {
                let posting_len = bytes.len() as u64;
                postings_writer.write_all(&bytes)?;

                let info = if let Some(&(pos_offset, pos_len)) = position_offsets.get(&key) {
                    TermInfo::external_with_positions(
                        postings_offset,
                        posting_len,
                        doc_count,
                        pos_offset,
                        pos_len,
                    )
                } else {
                    TermInfo::external(postings_offset, posting_len, doc_count)
                };
                postings_offset += posting_len;
                info
            }
        };

        writer.insert(&key, &term_info)?;
    }

    let _ = writer.finish()?;
    Ok(())
}

/// Stream positions directly to disk, returning only the offset map.
///
/// Consumes the position_index and writes each position posting list
/// directly to the writer, tracking offsets for the postings phase.
pub(super) fn build_positions_streaming(
    position_index: HashMap<TermKey, PositionPostingListBuilder>,
    term_interner: &Rodeo,
    writer: &mut dyn Write,
) -> Result<FxHashMap<Vec<u8>, (u64, u64)>> {
    use crate::structures::PositionPostingList;

    let mut position_offsets: FxHashMap<Vec<u8>, (u64, u64)> = FxHashMap::default();

    // Consume HashMap into Vec for sorting (owned, no borrowing)
    let mut entries: Vec<(Vec<u8>, PositionPostingListBuilder)> = position_index
        .into_iter()
        .map(|(term_key, pos_builder)| {
            let term_str = term_interner.resolve(&term_key.term);
            let mut key = Vec::with_capacity(size_of::<u32>() + term_str.len());
            key.extend_from_slice(&term_key.field.to_le_bytes());
            key.extend_from_slice(term_str.as_bytes());
            (key, pos_builder)
        })
        .collect();

    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut current_offset = 0u64;
    let mut buf = Vec::new();

    for (key, pos_builder) in entries {
        let mut pos_list = PositionPostingList::with_capacity(pos_builder.postings.len());
        for (doc_id, positions) in pos_builder.postings {
            pos_list.push(doc_id, positions);
        }

        // Serialize to reusable buffer, then write
        buf.clear();
        pos_list.serialize(&mut buf).map_err(crate::Error::Io)?;
        writer.write_all(&buf)?;

        position_offsets.insert(key, (current_offset, buf.len() as u64));
        current_offset += buf.len() as u64;
    }

    Ok(position_offsets)
}
