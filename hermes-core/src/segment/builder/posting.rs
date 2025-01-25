//! Posting list builders for inverted index

use lasso::Spur;

use crate::DocId;
use crate::structures::TermInfo;

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
            postings: Vec::new(),
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
        self.postings.push((doc_id, vec![position]));
    }
}

/// Intermediate result for parallel posting serialization
pub(super) enum SerializedPosting {
    /// Inline posting (small enough to fit in TermInfo)
    Inline(TermInfo),
    /// External posting with serialized bytes
    External { bytes: Vec<u8>, doc_count: u32 },
}
