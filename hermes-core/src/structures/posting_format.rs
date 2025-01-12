//! Unified posting list format with automatic compression selection
//!
//! Automatically selects the best compression method based on posting list characteristics:
//! - **Inline**: 1-3 postings stored directly in TermInfo (no separate I/O)
//! - **HorizontalBP128**: Horizontal layout with SIMD (fastest encoding + decoding)
//! - **Partitioned EF**: Best compression for large lists (>20K)
//! - **Roaring**: Bitmap-based for very frequent terms (>1% of corpus), fastest iteration
//!
//! Supports three optimization modes:
//! - **Adaptive**: Balanced compression/speed (default, zstd level 7)
//! - **SizeOptimized**: Best compression ratio (OptP4D + zstd level 22)
//! - **PerformanceOptimized**: Fastest decoding (Roaring + zstd level 3)

use byteorder::{ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

use super::elias_fano::EliasFanoPostingList;
use super::horizontal_bp128::HorizontalBP128PostingList;
use super::opt_p4d::OptP4DPostingList;
use super::partitioned_ef::PartitionedEFPostingList;
use super::roaring::RoaringPostingList;
use super::vertical_bp128::VerticalBP128PostingList;

/// Thresholds for format selection (based on benchmarks)
pub const INLINE_THRESHOLD: usize = 3;
pub const ROARING_THRESHOLD_RATIO: f32 = 0.01; // 1% of corpus
/// Threshold for using Partitioned EF (best compression)
pub const PARTITIONED_EF_THRESHOLD: usize = 20_000;

/// Index optimization mode for balancing compression ratio vs speed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexOptimization {
    /// Balanced compression and speed (zstd level 7)
    /// Uses adaptive format selection based on posting list characteristics
    #[default]
    Adaptive,
    /// Optimize for smallest index size (zstd level 22)
    /// Prefers OptP4D for best compression ratio (optimal bit-width + patched exceptions)
    SizeOptimized,
    /// Optimize for fastest query performance (zstd level 3)
    /// Prefers Roaring bitmaps for fastest iteration
    PerformanceOptimized,
}

impl IndexOptimization {
    /// Get the zstd compression level for this optimization mode
    pub fn zstd_level(&self) -> i32 {
        match self {
            IndexOptimization::Adaptive => 7,
            IndexOptimization::SizeOptimized => 22,
            IndexOptimization::PerformanceOptimized => 3,
        }
    }

    /// Parse from string (for CLI)
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "adaptive" | "balanced" | "default" => Some(IndexOptimization::Adaptive),
            "size" | "size-optimized" | "small" | "compact" => {
                Some(IndexOptimization::SizeOptimized)
            }
            "performance" | "perf" | "fast" | "speed" => {
                Some(IndexOptimization::PerformanceOptimized)
            }
            _ => None,
        }
    }
}

/// Format tag bytes for serialization
const FORMAT_HORIZONTAL_BP128: u8 = 0;
const FORMAT_ELIAS_FANO: u8 = 1;
const FORMAT_ROARING: u8 = 2;
const FORMAT_VERTICAL_BP128: u8 = 3;
const FORMAT_PARTITIONED_EF: u8 = 4;
const FORMAT_OPT_P4D: u8 = 5;

/// Posting list format selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostingFormat {
    /// Horizontal BP128 layout (fastest encoding + decoding with SIMD)
    HorizontalBP128,
    /// Vertical BP128 with bit-interleaved layout (optimal for SIMD scatter)
    VerticalBP128,
    /// Elias-Fano for long posting lists
    EliasFano,
    /// Partitioned Elias-Fano (very long lists, better compression)
    PartitionedEF,
    /// Roaring bitmap for very frequent terms
    Roaring,
    /// OptP4D (Optimized Patched Frame-of-Reference Delta) - optimal bit-width selection with exceptions
    OptP4D,
}

impl PostingFormat {
    /// Select optimal format based on posting list characteristics and optimization mode
    ///
    /// Format selection varies by optimization mode:
    /// - **Adaptive**: Balanced selection based on list size and frequency
    /// - **SizeOptimized**: Prefer PartitionedEF for best compression
    /// - **PerformanceOptimized**: Prefer Roaring for fastest iteration
    pub fn select(doc_count: usize, total_docs: usize) -> Self {
        Self::select_with_optimization(doc_count, total_docs, IndexOptimization::Adaptive)
    }

    /// Select format with explicit optimization mode
    pub fn select_with_optimization(
        doc_count: usize,
        total_docs: usize,
        optimization: IndexOptimization,
    ) -> Self {
        let frequency_ratio = doc_count as f32 / total_docs.max(1) as f32;

        match optimization {
            IndexOptimization::Adaptive => {
                // Balanced: use best format for each size range
                if frequency_ratio >= ROARING_THRESHOLD_RATIO
                    && doc_count >= PARTITIONED_EF_THRESHOLD
                {
                    PostingFormat::Roaring
                } else if doc_count >= PARTITIONED_EF_THRESHOLD {
                    PostingFormat::PartitionedEF
                } else {
                    PostingFormat::HorizontalBP128
                }
            }
            IndexOptimization::SizeOptimized => {
                // Size: prefer OptP4D for best compression (optimal bit-width + patched exceptions)
                if doc_count >= 128 {
                    PostingFormat::OptP4D
                } else {
                    PostingFormat::HorizontalBP128
                }
            }
            IndexOptimization::PerformanceOptimized => {
                // Performance: prefer Roaring for fastest iteration
                if doc_count >= 64 {
                    PostingFormat::Roaring
                } else {
                    PostingFormat::HorizontalBP128
                }
            }
        }
    }
}

/// Unified posting list that can use any compression format
#[derive(Debug, Clone)]
pub enum CompressedPostingList {
    HorizontalBP128(HorizontalBP128PostingList),
    VerticalBP128(VerticalBP128PostingList),
    EliasFano(EliasFanoPostingList),
    PartitionedEF(PartitionedEFPostingList),
    Roaring(RoaringPostingList),
    OptP4D(OptP4DPostingList),
}

impl CompressedPostingList {
    /// Create from raw postings with automatic format selection
    pub fn from_postings(doc_ids: &[u32], term_freqs: &[u32], total_docs: usize, idf: f32) -> Self {
        let format = PostingFormat::select(doc_ids.len(), total_docs);

        match format {
            PostingFormat::HorizontalBP128 => CompressedPostingList::HorizontalBP128(
                HorizontalBP128PostingList::from_postings(doc_ids, term_freqs, idf),
            ),
            PostingFormat::VerticalBP128 => CompressedPostingList::VerticalBP128(
                VerticalBP128PostingList::from_postings(doc_ids, term_freqs, idf),
            ),
            PostingFormat::EliasFano => CompressedPostingList::EliasFano(
                EliasFanoPostingList::from_postings(doc_ids, term_freqs),
            ),
            PostingFormat::PartitionedEF => CompressedPostingList::PartitionedEF(
                PartitionedEFPostingList::from_postings_with_idf(doc_ids, term_freqs, idf),
            ),
            PostingFormat::Roaring => CompressedPostingList::Roaring(
                RoaringPostingList::from_postings(doc_ids, term_freqs),
            ),
            PostingFormat::OptP4D => CompressedPostingList::OptP4D(
                OptP4DPostingList::from_postings(doc_ids, term_freqs, idf),
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
            PostingFormat::HorizontalBP128 => CompressedPostingList::HorizontalBP128(
                HorizontalBP128PostingList::from_postings(doc_ids, term_freqs, idf),
            ),
            PostingFormat::VerticalBP128 => CompressedPostingList::VerticalBP128(
                VerticalBP128PostingList::from_postings(doc_ids, term_freqs, idf),
            ),
            PostingFormat::EliasFano => CompressedPostingList::EliasFano(
                EliasFanoPostingList::from_postings(doc_ids, term_freqs),
            ),
            PostingFormat::PartitionedEF => CompressedPostingList::PartitionedEF(
                PartitionedEFPostingList::from_postings_with_idf(doc_ids, term_freqs, idf),
            ),
            PostingFormat::Roaring => CompressedPostingList::Roaring(
                RoaringPostingList::from_postings(doc_ids, term_freqs),
            ),
            PostingFormat::OptP4D => CompressedPostingList::OptP4D(
                OptP4DPostingList::from_postings(doc_ids, term_freqs, idf),
            ),
        }
    }

    /// Get document count
    pub fn doc_count(&self) -> u32 {
        match self {
            CompressedPostingList::HorizontalBP128(p) => p.doc_count,
            CompressedPostingList::VerticalBP128(p) => p.doc_count,
            CompressedPostingList::EliasFano(p) => p.len(),
            CompressedPostingList::PartitionedEF(p) => p.len(),
            CompressedPostingList::Roaring(p) => p.len(),
            CompressedPostingList::OptP4D(p) => p.len(),
        }
    }

    /// Get maximum term frequency
    pub fn max_tf(&self) -> u32 {
        match self {
            CompressedPostingList::HorizontalBP128(p) => p.max_score as u32, // Approximation
            CompressedPostingList::VerticalBP128(p) => {
                p.blocks.iter().map(|b| b.max_tf).max().unwrap_or(0)
            }
            CompressedPostingList::EliasFano(p) => p.max_tf,
            CompressedPostingList::PartitionedEF(p) => p.max_tf,
            CompressedPostingList::Roaring(p) => p.max_tf,
            CompressedPostingList::OptP4D(p) => {
                p.blocks.iter().map(|b| b.max_tf).max().unwrap_or(0)
            }
        }
    }

    /// Get format type
    pub fn format(&self) -> PostingFormat {
        match self {
            CompressedPostingList::HorizontalBP128(_) => PostingFormat::HorizontalBP128,
            CompressedPostingList::VerticalBP128(_) => PostingFormat::VerticalBP128,
            CompressedPostingList::EliasFano(_) => PostingFormat::EliasFano,
            CompressedPostingList::PartitionedEF(_) => PostingFormat::PartitionedEF,
            CompressedPostingList::Roaring(_) => PostingFormat::Roaring,
            CompressedPostingList::OptP4D(_) => PostingFormat::OptP4D,
        }
    }

    /// Serialize
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        match self {
            CompressedPostingList::HorizontalBP128(p) => {
                writer.write_u8(FORMAT_HORIZONTAL_BP128)?;
                p.serialize(writer)
            }
            CompressedPostingList::VerticalBP128(p) => {
                writer.write_u8(FORMAT_VERTICAL_BP128)?;
                p.serialize(writer)
            }
            CompressedPostingList::EliasFano(p) => {
                writer.write_u8(FORMAT_ELIAS_FANO)?;
                p.serialize(writer)
            }
            CompressedPostingList::PartitionedEF(p) => {
                writer.write_u8(FORMAT_PARTITIONED_EF)?;
                p.serialize(writer)
            }
            CompressedPostingList::Roaring(p) => {
                writer.write_u8(FORMAT_ROARING)?;
                p.serialize(writer)
            }
            CompressedPostingList::OptP4D(p) => {
                writer.write_u8(FORMAT_OPT_P4D)?;
                p.serialize(writer)
            }
        }
    }

    /// Deserialize
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let format = reader.read_u8()?;
        match format {
            FORMAT_HORIZONTAL_BP128 => Ok(CompressedPostingList::HorizontalBP128(
                HorizontalBP128PostingList::deserialize(reader)?,
            )),
            FORMAT_VERTICAL_BP128 => Ok(CompressedPostingList::VerticalBP128(
                VerticalBP128PostingList::deserialize(reader)?,
            )),
            FORMAT_ELIAS_FANO => Ok(CompressedPostingList::EliasFano(
                EliasFanoPostingList::deserialize(reader)?,
            )),
            FORMAT_PARTITIONED_EF => Ok(CompressedPostingList::PartitionedEF(
                PartitionedEFPostingList::deserialize(reader)?,
            )),
            FORMAT_ROARING => Ok(CompressedPostingList::Roaring(
                RoaringPostingList::deserialize(reader)?,
            )),
            FORMAT_OPT_P4D => Ok(CompressedPostingList::OptP4D(
                OptP4DPostingList::deserialize(reader)?,
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
            CompressedPostingList::HorizontalBP128(p) => {
                CompressedPostingIterator::HorizontalBP128(p.iterator())
            }
            CompressedPostingList::VerticalBP128(p) => {
                CompressedPostingIterator::VerticalBP128(p.iterator())
            }
            CompressedPostingList::EliasFano(p) => {
                CompressedPostingIterator::EliasFano(p.iterator())
            }
            CompressedPostingList::PartitionedEF(p) => {
                CompressedPostingIterator::PartitionedEF(p.iterator())
            }
            CompressedPostingList::Roaring(p) => {
                let mut iter = p.iterator();
                iter.init();
                CompressedPostingIterator::Roaring(iter)
            }
            CompressedPostingList::OptP4D(p) => CompressedPostingIterator::OptP4D(p.iterator()),
        }
    }
}

/// Unified iterator over any posting list format
pub enum CompressedPostingIterator<'a> {
    HorizontalBP128(super::horizontal_bp128::HorizontalBP128Iterator<'a>),
    VerticalBP128(super::vertical_bp128::VerticalBP128Iterator<'a>),
    EliasFano(super::elias_fano::EliasFanoPostingIterator<'a>),
    PartitionedEF(super::partitioned_ef::PartitionedEFPostingIterator<'a>),
    Roaring(super::roaring::RoaringPostingIterator<'a>),
    OptP4D(super::opt_p4d::OptP4DIterator<'a>),
}

impl<'a> CompressedPostingIterator<'a> {
    /// Current document ID
    pub fn doc(&self) -> u32 {
        match self {
            CompressedPostingIterator::HorizontalBP128(i) => i.doc(),
            CompressedPostingIterator::VerticalBP128(i) => i.doc(),
            CompressedPostingIterator::EliasFano(i) => i.doc(),
            CompressedPostingIterator::PartitionedEF(i) => i.doc(),
            CompressedPostingIterator::Roaring(i) => i.doc(),
            CompressedPostingIterator::OptP4D(i) => i.doc(),
        }
    }

    /// Current term frequency
    pub fn term_freq(&self) -> u32 {
        match self {
            CompressedPostingIterator::HorizontalBP128(i) => i.term_freq(),
            CompressedPostingIterator::VerticalBP128(i) => i.term_freq(),
            CompressedPostingIterator::EliasFano(i) => i.term_freq(),
            CompressedPostingIterator::PartitionedEF(i) => i.term_freq(),
            CompressedPostingIterator::Roaring(i) => i.term_freq(),
            CompressedPostingIterator::OptP4D(i) => i.term_freq(),
        }
    }

    /// Advance to next document
    pub fn advance(&mut self) -> u32 {
        match self {
            CompressedPostingIterator::HorizontalBP128(i) => i.advance(),
            CompressedPostingIterator::VerticalBP128(i) => i.advance(),
            CompressedPostingIterator::EliasFano(i) => i.advance(),
            CompressedPostingIterator::PartitionedEF(i) => i.advance(),
            CompressedPostingIterator::Roaring(i) => i.advance(),
            CompressedPostingIterator::OptP4D(i) => i.advance(),
        }
    }

    /// Seek to first doc >= target
    pub fn seek(&mut self, target: u32) -> u32 {
        match self {
            CompressedPostingIterator::HorizontalBP128(i) => i.seek(target),
            CompressedPostingIterator::VerticalBP128(i) => i.seek(target),
            CompressedPostingIterator::EliasFano(i) => i.seek(target),
            CompressedPostingIterator::PartitionedEF(i) => i.seek(target),
            CompressedPostingIterator::Roaring(i) => i.seek(target),
            CompressedPostingIterator::OptP4D(i) => i.seek(target),
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
    pub simd_bp128_count: u32,
    pub simd_bp128_docs: u64,
    pub elias_fano_count: u32,
    pub elias_fano_docs: u64,
    pub partitioned_ef_count: u32,
    pub partitioned_ef_docs: u64,
    pub roaring_count: u32,
    pub roaring_docs: u64,
    pub inline_count: u32,
    pub inline_docs: u64,
}

impl CompressionStats {
    pub fn record(&mut self, format: PostingFormat, doc_count: u32) {
        match format {
            PostingFormat::HorizontalBP128 => {
                self.bitpacked_count += 1;
                self.bitpacked_docs += doc_count as u64;
            }
            PostingFormat::VerticalBP128 => {
                self.simd_bp128_count += 1;
                self.simd_bp128_docs += doc_count as u64;
            }
            PostingFormat::EliasFano => {
                self.elias_fano_count += 1;
                self.elias_fano_docs += doc_count as u64;
            }
            PostingFormat::PartitionedEF => {
                self.partitioned_ef_count += 1;
                self.partitioned_ef_docs += doc_count as u64;
            }
            PostingFormat::Roaring => {
                self.roaring_count += 1;
                self.roaring_docs += doc_count as u64;
            }
            PostingFormat::OptP4D => {
                // Track OptP4D under bitpacked for now (similar compression family)
                self.bitpacked_count += 1;
                self.bitpacked_docs += doc_count as u64;
            }
        }
    }

    pub fn record_inline(&mut self, doc_count: u32) {
        self.inline_count += 1;
        self.inline_docs += doc_count as u64;
    }

    pub fn total_terms(&self) -> u32 {
        self.bitpacked_count
            + self.simd_bp128_count
            + self.elias_fano_count
            + self.partitioned_ef_count
            + self.roaring_count
            + self.inline_count
    }

    pub fn total_postings(&self) -> u64 {
        self.bitpacked_docs
            + self.simd_bp128_docs
            + self.elias_fano_docs
            + self.partitioned_ef_docs
            + self.roaring_docs
            + self.inline_docs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_selection() {
        // Small list -> HorizontalBP128 (fastest encoding + decoding with SIMD)
        assert_eq!(
            PostingFormat::select(100, 1_000_000),
            PostingFormat::HorizontalBP128
        );

        // Medium list -> HorizontalBP128
        assert_eq!(
            PostingFormat::select(500, 1_000_000),
            PostingFormat::HorizontalBP128
        );

        assert_eq!(
            PostingFormat::select(5_000, 1_000_000),
            PostingFormat::HorizontalBP128
        );

        assert_eq!(
            PostingFormat::select(15_000, 10_000_000),
            PostingFormat::HorizontalBP128
        );

        // Large list (>=20K) -> Partitioned EF
        assert_eq!(
            PostingFormat::select(25_000, 10_000_000),
            PostingFormat::PartitionedEF
        );

        // Very frequent term (>1% of corpus AND >=20K docs) -> Roaring
        assert_eq!(
            PostingFormat::select(50_000, 1_000_000),
            PostingFormat::Roaring
        );
    }

    #[test]
    fn test_compressed_posting_list_small() {
        let doc_ids: Vec<u32> = (0..100).map(|i| i * 2).collect();
        let term_freqs: Vec<u32> = vec![1; 100];

        let list = CompressedPostingList::from_postings(&doc_ids, &term_freqs, 1_000_000, 1.0);

        // Small lists use HorizontalBP128 (fastest encoding + decoding with SIMD)
        assert_eq!(list.format(), PostingFormat::HorizontalBP128);
        assert_eq!(list.doc_count(), 100);

        let mut iter = list.iterator();
        for (i, &expected) in doc_ids.iter().enumerate() {
            assert_eq!(iter.doc(), expected, "Mismatch at {}", i);
            iter.advance();
        }
    }

    #[test]
    fn test_compressed_posting_list_bitpacked() {
        let doc_ids: Vec<u32> = (0..15_000).map(|i| i * 2).collect();
        let term_freqs: Vec<u32> = vec![1; 15_000];

        // 15K docs with large corpus -> HorizontalBP128 (under 20K threshold)
        let list = CompressedPostingList::from_postings(&doc_ids, &term_freqs, 10_000_000, 1.0);

        assert_eq!(list.format(), PostingFormat::HorizontalBP128);
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

    #[test]
    fn test_opt_p4d_via_unified_interface() {
        let doc_ids: Vec<u32> = (0..500).map(|i| i * 3).collect();
        let term_freqs: Vec<u32> = (0..500).map(|i| (i % 5) + 1).collect();

        // Create OptP4D explicitly via the unified interface
        let list = CompressedPostingList::from_postings_with_format(
            &doc_ids,
            &term_freqs,
            PostingFormat::OptP4D,
            1.0,
        );

        assert_eq!(list.format(), PostingFormat::OptP4D);
        assert_eq!(list.doc_count(), 500);

        // Test serialization roundtrip
        let mut buffer = Vec::new();
        list.serialize(&mut buffer).unwrap();
        let restored = CompressedPostingList::deserialize(&mut &buffer[..]).unwrap();

        assert_eq!(restored.format(), PostingFormat::OptP4D);
        assert_eq!(restored.doc_count(), 500);

        // Verify iteration
        let mut iter1 = list.iterator();
        let mut iter2 = restored.iterator();

        while iter1.doc() != u32::MAX {
            assert_eq!(iter1.doc(), iter2.doc());
            assert_eq!(iter1.term_freq(), iter2.term_freq());
            iter1.advance();
            iter2.advance();
        }

        // Test seek
        let mut iter = list.iterator();
        assert_eq!(iter.seek(100), 102); // 34 * 3 = 102
        assert_eq!(iter.seek(500), 501); // 167 * 3 = 501
    }
}
