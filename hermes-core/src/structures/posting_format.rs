//! Unified posting list format with automatic compression selection
//!
//! Automatically selects the best compression method based on posting list characteristics:
//! - **Inline**: 1-3 postings stored directly in TermInfo (no separate I/O)
//! - **Bitpacked**: Standard block-based compression for medium lists
//! - **Elias-Fano**: Near-optimal compression for long lists (>10K docs)
//! - **Roaring**: Bitmap-based for very frequent terms (>1% of corpus)

use byteorder::{ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

use super::bitpacking::BitpackedPostingList;
use super::elias_fano::EliasFanoPostingList;
use super::roaring::RoaringPostingList;

/// Thresholds for format selection
pub const INLINE_THRESHOLD: usize = 3;
pub const ELIAS_FANO_THRESHOLD: usize = 10_000;
pub const ROARING_THRESHOLD_RATIO: f32 = 0.01; // 1% of corpus

/// Format tag bytes for serialization
const FORMAT_BITPACKED: u8 = 0;
const FORMAT_ELIAS_FANO: u8 = 1;
const FORMAT_ROARING: u8 = 2;

/// Posting list format selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostingFormat {
    /// Standard bitpacked format (default)
    Bitpacked,
    /// Elias-Fano for long posting lists
    EliasFano,
    /// Roaring bitmap for very frequent terms
    Roaring,
}

impl PostingFormat {
    /// Select optimal format based on posting list characteristics
    pub fn select(doc_count: usize, total_docs: usize) -> Self {
        let frequency_ratio = doc_count as f32 / total_docs.max(1) as f32;

        if frequency_ratio >= ROARING_THRESHOLD_RATIO && doc_count > ELIAS_FANO_THRESHOLD {
            // Very frequent term - use Roaring for fast intersections
            PostingFormat::Roaring
        } else if doc_count >= ELIAS_FANO_THRESHOLD {
            // Long posting list - use Elias-Fano for better compression
            PostingFormat::EliasFano
        } else {
            // Standard case - use bitpacked
            PostingFormat::Bitpacked
        }
    }
}

/// Unified posting list that can use any compression format
#[derive(Debug, Clone)]
pub enum CompressedPostingList {
    Bitpacked(BitpackedPostingList),
    EliasFano(EliasFanoPostingList),
    Roaring(RoaringPostingList),
}

impl CompressedPostingList {
    /// Create from raw postings with automatic format selection
    pub fn from_postings(doc_ids: &[u32], term_freqs: &[u32], total_docs: usize, idf: f32) -> Self {
        let format = PostingFormat::select(doc_ids.len(), total_docs);

        match format {
            PostingFormat::Bitpacked => CompressedPostingList::Bitpacked(
                BitpackedPostingList::from_postings(doc_ids, term_freqs, idf),
            ),
            PostingFormat::EliasFano => CompressedPostingList::EliasFano(
                EliasFanoPostingList::from_postings(doc_ids, term_freqs),
            ),
            PostingFormat::Roaring => CompressedPostingList::Roaring(
                RoaringPostingList::from_postings(doc_ids, term_freqs),
            ),
        }
    }

    /// Create with explicit format selection
    pub fn from_postings_with_format(
        doc_ids: &[u32],
        term_freqs: &[u32],
        format: PostingFormat,
        idf: f32,
    ) -> Self {
        match format {
            PostingFormat::Bitpacked => CompressedPostingList::Bitpacked(
                BitpackedPostingList::from_postings(doc_ids, term_freqs, idf),
            ),
            PostingFormat::EliasFano => CompressedPostingList::EliasFano(
                EliasFanoPostingList::from_postings(doc_ids, term_freqs),
            ),
            PostingFormat::Roaring => CompressedPostingList::Roaring(
                RoaringPostingList::from_postings(doc_ids, term_freqs),
            ),
        }
    }

    /// Get document count
    pub fn doc_count(&self) -> u32 {
        match self {
            CompressedPostingList::Bitpacked(p) => p.doc_count,
            CompressedPostingList::EliasFano(p) => p.len(),
            CompressedPostingList::Roaring(p) => p.len(),
        }
    }

    /// Get maximum term frequency
    pub fn max_tf(&self) -> u32 {
        match self {
            CompressedPostingList::Bitpacked(p) => p.max_score as u32, // Approximation
            CompressedPostingList::EliasFano(p) => p.max_tf,
            CompressedPostingList::Roaring(p) => p.max_tf,
        }
    }

    /// Get format type
    pub fn format(&self) -> PostingFormat {
        match self {
            CompressedPostingList::Bitpacked(_) => PostingFormat::Bitpacked,
            CompressedPostingList::EliasFano(_) => PostingFormat::EliasFano,
            CompressedPostingList::Roaring(_) => PostingFormat::Roaring,
        }
    }

    /// Serialize
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        match self {
            CompressedPostingList::Bitpacked(p) => {
                writer.write_u8(FORMAT_BITPACKED)?;
                p.serialize(writer)
            }
            CompressedPostingList::EliasFano(p) => {
                writer.write_u8(FORMAT_ELIAS_FANO)?;
                p.serialize(writer)
            }
            CompressedPostingList::Roaring(p) => {
                writer.write_u8(FORMAT_ROARING)?;
                p.serialize(writer)
            }
        }
    }

    /// Deserialize
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let format = reader.read_u8()?;
        match format {
            FORMAT_BITPACKED => Ok(CompressedPostingList::Bitpacked(
                BitpackedPostingList::deserialize(reader)?,
            )),
            FORMAT_ELIAS_FANO => Ok(CompressedPostingList::EliasFano(
                EliasFanoPostingList::deserialize(reader)?,
            )),
            FORMAT_ROARING => Ok(CompressedPostingList::Roaring(
                RoaringPostingList::deserialize(reader)?,
            )),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown posting list format: {}", format),
            )),
        }
    }

    /// Create an iterator
    pub fn iterator(&self) -> CompressedPostingIterator<'_> {
        match self {
            CompressedPostingList::Bitpacked(p) => {
                CompressedPostingIterator::Bitpacked(p.iterator())
            }
            CompressedPostingList::EliasFano(p) => {
                CompressedPostingIterator::EliasFano(p.iterator())
            }
            CompressedPostingList::Roaring(p) => {
                let mut iter = p.iterator();
                iter.init();
                CompressedPostingIterator::Roaring(iter)
            }
        }
    }
}

/// Unified iterator over any posting list format
pub enum CompressedPostingIterator<'a> {
    Bitpacked(super::bitpacking::BitpackedPostingIterator<'a>),
    EliasFano(super::elias_fano::EliasFanoPostingIterator<'a>),
    Roaring(super::roaring::RoaringPostingIterator<'a>),
}

impl<'a> CompressedPostingIterator<'a> {
    /// Current document ID
    pub fn doc(&self) -> u32 {
        match self {
            CompressedPostingIterator::Bitpacked(i) => i.doc(),
            CompressedPostingIterator::EliasFano(i) => i.doc(),
            CompressedPostingIterator::Roaring(i) => i.doc(),
        }
    }

    /// Current term frequency
    pub fn term_freq(&self) -> u32 {
        match self {
            CompressedPostingIterator::Bitpacked(i) => i.term_freq(),
            CompressedPostingIterator::EliasFano(i) => i.term_freq(),
            CompressedPostingIterator::Roaring(i) => i.term_freq(),
        }
    }

    /// Advance to next document
    pub fn advance(&mut self) -> u32 {
        match self {
            CompressedPostingIterator::Bitpacked(i) => i.advance(),
            CompressedPostingIterator::EliasFano(i) => i.advance(),
            CompressedPostingIterator::Roaring(i) => i.advance(),
        }
    }

    /// Seek to first doc >= target
    pub fn seek(&mut self, target: u32) -> u32 {
        match self {
            CompressedPostingIterator::Bitpacked(i) => i.seek(target),
            CompressedPostingIterator::EliasFano(i) => i.seek(target),
            CompressedPostingIterator::Roaring(i) => i.seek(target),
        }
    }

    /// Check if exhausted
    pub fn is_exhausted(&self) -> bool {
        self.doc() == u32::MAX
    }
}

/// Statistics about compression format usage
#[derive(Debug, Default, Clone)]
pub struct CompressionStats {
    pub bitpacked_count: u32,
    pub bitpacked_docs: u64,
    pub elias_fano_count: u32,
    pub elias_fano_docs: u64,
    pub roaring_count: u32,
    pub roaring_docs: u64,
    pub inline_count: u32,
    pub inline_docs: u64,
}

impl CompressionStats {
    pub fn record(&mut self, format: PostingFormat, doc_count: u32) {
        match format {
            PostingFormat::Bitpacked => {
                self.bitpacked_count += 1;
                self.bitpacked_docs += doc_count as u64;
            }
            PostingFormat::EliasFano => {
                self.elias_fano_count += 1;
                self.elias_fano_docs += doc_count as u64;
            }
            PostingFormat::Roaring => {
                self.roaring_count += 1;
                self.roaring_docs += doc_count as u64;
            }
        }
    }

    pub fn record_inline(&mut self, doc_count: u32) {
        self.inline_count += 1;
        self.inline_docs += doc_count as u64;
    }

    pub fn total_terms(&self) -> u32 {
        self.bitpacked_count + self.elias_fano_count + self.roaring_count + self.inline_count
    }

    pub fn total_postings(&self) -> u64 {
        self.bitpacked_docs + self.elias_fano_docs + self.roaring_docs + self.inline_docs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_selection() {
        // Small list -> Bitpacked
        assert_eq!(
            PostingFormat::select(100, 1_000_000),
            PostingFormat::Bitpacked
        );

        // Medium list -> Bitpacked
        assert_eq!(
            PostingFormat::select(5_000, 1_000_000),
            PostingFormat::Bitpacked
        );

        // Long list but not frequent enough -> Elias-Fano
        // 15K / 1M = 1.5% which is > 1% threshold, so it's Roaring
        // Use a larger corpus to get Elias-Fano
        assert_eq!(
            PostingFormat::select(15_000, 10_000_000),
            PostingFormat::EliasFano
        );

        // Very frequent term (>1% of corpus AND >10K docs) -> Roaring
        assert_eq!(
            PostingFormat::select(50_000, 1_000_000),
            PostingFormat::Roaring
        );
    }

    #[test]
    fn test_compressed_posting_list_bitpacked() {
        let doc_ids: Vec<u32> = (0..100).map(|i| i * 2).collect();
        let term_freqs: Vec<u32> = vec![1; 100];

        let list = CompressedPostingList::from_postings(&doc_ids, &term_freqs, 1_000_000, 1.0);

        assert_eq!(list.format(), PostingFormat::Bitpacked);
        assert_eq!(list.doc_count(), 100);

        let mut iter = list.iterator();
        for (i, &expected) in doc_ids.iter().enumerate() {
            assert_eq!(iter.doc(), expected, "Mismatch at {}", i);
            iter.advance();
        }
    }

    #[test]
    fn test_compressed_posting_list_elias_fano() {
        let doc_ids: Vec<u32> = (0..15_000).map(|i| i * 2).collect();
        let term_freqs: Vec<u32> = vec![1; 15_000];

        // Use large corpus so 15K docs is < 1% (Elias-Fano, not Roaring)
        let list = CompressedPostingList::from_postings(&doc_ids, &term_freqs, 10_000_000, 1.0);

        assert_eq!(list.format(), PostingFormat::EliasFano);
        assert_eq!(list.doc_count(), 15_000);
    }

    #[test]
    fn test_compressed_posting_list_serialization() {
        let doc_ids: Vec<u32> = (0..500).map(|i| i * 3).collect();
        let term_freqs: Vec<u32> = (0..500).map(|i| (i % 5) + 1).collect();

        let list = CompressedPostingList::from_postings(&doc_ids, &term_freqs, 1_000_000, 1.0);

        let mut buffer = Vec::new();
        list.serialize(&mut buffer).unwrap();

        let restored = CompressedPostingList::deserialize(&mut &buffer[..]).unwrap();

        assert_eq!(restored.format(), list.format());
        assert_eq!(restored.doc_count(), list.doc_count());

        // Verify iteration
        let mut iter1 = list.iterator();
        let mut iter2 = restored.iterator();

        while iter1.doc() != u32::MAX {
            assert_eq!(iter1.doc(), iter2.doc());
            assert_eq!(iter1.term_freq(), iter2.term_freq());
            iter1.advance();
            iter2.advance();
        }
    }

    #[test]
    fn test_iterator_seek() {
        let doc_ids: Vec<u32> = vec![10, 20, 30, 100, 200, 300, 1000, 2000];
        let term_freqs: Vec<u32> = vec![1; 8];

        let list = CompressedPostingList::from_postings(&doc_ids, &term_freqs, 1_000_000, 1.0);
        let mut iter = list.iterator();

        assert_eq!(iter.seek(25), 30);
        assert_eq!(iter.seek(100), 100);
        assert_eq!(iter.seek(500), 1000);
        assert_eq!(iter.seek(3000), u32::MAX);
    }
}
