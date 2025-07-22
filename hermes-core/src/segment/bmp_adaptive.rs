//! Adaptive per-block payload codec for BMP V19.
//!
//! A block keeps sorted `u32` dimensions, but adapts the rest of its header
//! and payload to the data:
//!
//! ```text
//! raw_num_terms: u32
//! dimensions:    [u32; num_terms]
//! byte_offsets:  [u16 | u32; num_terms + 1]
//! maxima_u4:     [u8; ceil(num_terms / 2)]
//! payload:       sparse `(slot, impact)` pairs or dense impact rows
//! ```
//!
//! `raw_num_terms`' high bit selects 32-bit offsets; otherwise offsets are
//! 16-bit. The high bit of each term's start offset selects a dense impact
//! row. The remaining bits are byte offsets into `payload`. A dense row is
//! chosen exactly when it is no larger than the sparse pairs and slots are
//! unique, so the representation never grows because of adaptation.

const WIDE_BLOCK_FLAG: u32 = 1 << 31;
const NARROW_DENSE_FLAG: u16 = 1 << 15;
const WIDE_DENSE_FLAG: u32 = 1 << 31;
const NARROW_OFFSET_MASK: u16 = !NARROW_DENSE_FLAG;
const WIDE_OFFSET_MASK: u32 = !WIDE_DENSE_FLAG;
const MAXIMUM_QUANTUM: u8 = 17;

/// A single posting in BMP's block-local inverted index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct BmpPosting {
    pub(crate) local_slot: u8,
    pub(crate) impact: u8,
}

const _: [(); 2] = [(); std::mem::size_of::<BmpPosting>()];
const _: [(); 1] = [(); std::mem::align_of::<BmpPosting>()];

/// Reusable bounded scratch for encoding one adaptive block.
#[derive(Default)]
pub(crate) struct AdaptiveEncodeScratch {
    offsets: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AdaptiveEncodeStats {
    pub(crate) dense_terms: usize,
    pub(crate) wide_offsets: bool,
}

impl AdaptiveEncodeScratch {
    /// Encode one non-empty block.
    ///
    /// `sparse_postings` contains two bytes per logical posting in the same
    /// term-major order described by `posting_counts`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &mut self,
        block_size: usize,
        enable_dense_rows: bool,
        dimensions: &[u32],
        posting_counts: &[u32],
        exact_maxima: &[u8],
        sparse_postings: &[u8],
        output: &mut Vec<u8>,
    ) -> std::io::Result<AdaptiveEncodeStats> {
        if !sparse_postings.len().is_multiple_of(2) {
            return Err(invalid_data(
                "BMP adaptive sparse posting payload has an odd byte count",
            ));
        }
        self.encode_postings(
            block_size,
            enable_dense_rows,
            dimensions,
            posting_counts,
            exact_maxima,
            sparse_postings.len() / 2,
            |index| BmpPosting {
                local_slot: sparse_postings[index * 2],
                impact: sparse_postings[index * 2 + 1],
            },
            output,
        )
    }

    /// Encode from an indexable logical-posting source. Reorder uses this to
    /// read routed tuples in place instead of materializing a second pair
    /// buffer for every output block.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_postings(
        &mut self,
        block_size: usize,
        enable_dense_rows: bool,
        dimensions: &[u32],
        posting_counts: &[u32],
        exact_maxima: &[u8],
        total_postings: usize,
        mut posting_at: impl FnMut(usize) -> BmpPosting,
        output: &mut Vec<u8>,
    ) -> std::io::Result<AdaptiveEncodeStats> {
        if !(1..=256).contains(&block_size) {
            return Err(invalid_data("BMP adaptive block size must be in 1..=256"));
        }
        let num_terms = dimensions.len();
        if num_terms == 0 || posting_counts.len() != num_terms || exact_maxima.len() != num_terms {
            return Err(invalid_data(
                "BMP adaptive block arrays have inconsistent lengths",
            ));
        }
        if num_terms > WIDE_OFFSET_MASK as usize {
            return Err(invalid_data(
                "BMP adaptive block term count exceeds the u31 format limit",
            ));
        }

        self.offsets.clear();
        self.offsets.reserve_exact(num_terms);
        let mut posting_start = 0usize;
        let mut payload_len = 0usize;
        let mut dense_terms = 0usize;
        let mut previous_dimension = None;

        for term in 0..num_terms {
            let dimension = dimensions[term];
            if previous_dimension.is_some_and(|previous| dimension <= previous) {
                return Err(invalid_data(
                    "BMP adaptive block dimensions are not strictly increasing",
                ));
            }
            previous_dimension = Some(dimension);

            let count = posting_counts[term] as usize;
            if count == 0 {
                return Err(invalid_data("BMP adaptive term has no postings"));
            }
            let posting_end = posting_start.checked_add(count).ok_or_else(|| {
                invalid_data("BMP adaptive logical posting range overflows usize")
            })?;
            if posting_end > total_postings {
                return Err(invalid_data(
                    "BMP adaptive posting counts exceed the logical payload",
                ));
            }
            let source_bytes = count.checked_mul(2).ok_or_else(|| {
                invalid_data("BMP adaptive sparse posting byte count overflows usize")
            })?;

            let mut observed_max = 0u8;
            let mut occupied = [0u64; 4];
            let mut unique_slots = true;
            for index in posting_start..posting_end {
                let posting = posting_at(index);
                let slot = posting.local_slot as usize;
                let impact = posting.impact;
                if slot >= block_size {
                    return Err(invalid_data(
                        "BMP adaptive posting slot exceeds the block size",
                    ));
                }
                if impact == 0 {
                    return Err(invalid_data("BMP adaptive posting has zero impact"));
                }
                let bit = 1u64 << (slot % 64);
                let word = &mut occupied[slot / 64];
                unique_slots &= *word & bit == 0;
                *word |= bit;
                observed_max = observed_max.max(impact);
            }
            if exact_maxima[term] != observed_max {
                return Err(invalid_data(
                    "BMP adaptive term maximum does not match its postings",
                ));
            }

            let dense = enable_dense_rows && count >= block_size.div_ceil(2) && unique_slots;
            dense_terms += usize::from(dense);
            let term_bytes = if dense { block_size } else { source_bytes };
            let raw_offset = u32::try_from(payload_len)
                .map_err(|_| invalid_data("BMP adaptive payload exceeds the u32 format limit"))?;
            if raw_offset > WIDE_OFFSET_MASK {
                return Err(invalid_data(
                    "BMP adaptive payload exceeds the u31 format limit",
                ));
            }
            self.offsets
                .push(raw_offset | if dense { WIDE_DENSE_FLAG } else { 0 });
            payload_len = payload_len
                .checked_add(term_bytes)
                .ok_or_else(|| invalid_data("BMP adaptive payload byte count overflows usize"))?;
            posting_start = posting_end;
        }
        if posting_start != total_postings {
            return Err(invalid_data(
                "BMP adaptive logical payload has trailing postings",
            ));
        }
        if payload_len > WIDE_OFFSET_MASK as usize {
            return Err(invalid_data(
                "BMP adaptive payload exceeds the u31 format limit",
            ));
        }

        let wide = payload_len > NARROW_OFFSET_MASK as usize;
        let offset_width = if wide { 4usize } else { 2usize };
        let offsets_bytes = num_terms
            .checked_add(1)
            .and_then(|count| count.checked_mul(offset_width))
            .ok_or_else(|| invalid_data("BMP adaptive offset header overflows usize"))?;
        let header_len = 4usize
            .checked_add(
                num_terms
                    .checked_mul(4)
                    .ok_or_else(|| invalid_data("BMP adaptive dimension header overflows usize"))?,
            )
            .and_then(|len| len.checked_add(offsets_bytes))
            .and_then(|len| len.checked_add(num_terms.div_ceil(2)))
            .ok_or_else(|| invalid_data("BMP adaptive header overflows usize"))?;
        let encoded_len = header_len
            .checked_add(payload_len)
            .ok_or_else(|| invalid_data("BMP adaptive block size overflows usize"))?;

        output.clear();
        output.reserve_exact(encoded_len);
        let raw_num_terms = num_terms as u32 | if wide { WIDE_BLOCK_FLAG } else { 0 };
        output.extend_from_slice(&raw_num_terms.to_le_bytes());
        for &dimension in dimensions {
            output.extend_from_slice(&dimension.to_le_bytes());
        }
        if wide {
            for &encoded in &self.offsets {
                output.extend_from_slice(&encoded.to_le_bytes());
            }
            output.extend_from_slice(&(payload_len as u32).to_le_bytes());
        } else {
            for &wide_encoded in &self.offsets {
                let offset = wide_encoded & WIDE_OFFSET_MASK;
                let dense = wide_encoded & WIDE_DENSE_FLAG != 0;
                let encoded = offset as u16 | if dense { NARROW_DENSE_FLAG } else { 0 };
                output.extend_from_slice(&encoded.to_le_bytes());
            }
            output.extend_from_slice(&(payload_len as u16).to_le_bytes());
        }
        for maxima in exact_maxima.chunks(2) {
            let low = quantize_maximum(maxima[0]);
            let high = maxima.get(1).copied().map_or(0, quantize_maximum);
            output.push(low | high << 4);
        }

        posting_start = 0;
        for (term, &count) in posting_counts.iter().enumerate() {
            let posting_end = posting_start + count as usize;
            if self.offsets[term] & WIDE_DENSE_FLAG != 0 {
                let row_start = output.len();
                output.resize(row_start + block_size, 0);
                let row = &mut output[row_start..];
                for index in posting_start..posting_end {
                    let posting = posting_at(index);
                    row[posting.local_slot as usize] = posting.impact;
                }
            } else {
                for index in posting_start..posting_end {
                    let posting = posting_at(index);
                    output.extend_from_slice(&[posting.local_slot, posting.impact]);
                }
            }
            posting_start = posting_end;
        }
        debug_assert_eq!(output.len(), encoded_len);
        Ok(AdaptiveEncodeStats {
            dense_terms,
            wide_offsets: wide,
        })
    }
}

/// A parsed zero-copy adaptive block.
#[derive(Clone, Copy)]
pub(crate) struct AdaptiveBlock<'a> {
    bytes: &'a [u8],
    block_size: usize,
    num_terms: usize,
    dimensions_start: usize,
    offsets_start: usize,
    maxima_start: usize,
    payload_start: usize,
    offset_width: usize,
}

impl<'a> AdaptiveBlock<'a> {
    /// Parse the constant-time structural envelope needed by the query path.
    ///
    /// Per-term ranges and postings are checked lazily by `postings`; exhaustive
    /// rewrite validation uses `validate`.
    #[inline(always)]
    pub(crate) fn parse(bytes: &'a [u8], block_size: usize) -> Option<Self> {
        if bytes.len() < 4 || !(1..=256).contains(&block_size) {
            return None;
        }
        let raw_num_terms = read_u32(bytes, 0)?;
        let wide = raw_num_terms & WIDE_BLOCK_FLAG != 0;
        let num_terms = (raw_num_terms & WIDE_OFFSET_MASK) as usize;
        if num_terms == 0 {
            return None;
        }
        let offset_width = if wide { 4usize } else { 2usize };
        let dimensions_start = 4usize;
        let offsets_start = dimensions_start.checked_add(num_terms.checked_mul(4)?)?;
        let maxima_start =
            offsets_start.checked_add(num_terms.checked_add(1)?.checked_mul(offset_width)?)?;
        let payload_start = maxima_start.checked_add(num_terms.div_ceil(2))?;
        if payload_start > bytes.len() {
            return None;
        }
        let block = Self {
            bytes,
            block_size,
            num_terms,
            dimensions_start,
            offsets_start,
            maxima_start,
            payload_start,
            offset_width,
        };
        let (first, _) = block.offset(0)?;
        let (sentinel, sentinel_dense) = block.offset(num_terms)?;
        if first != 0 || sentinel_dense || sentinel != bytes.len() - payload_start {
            return None;
        }
        Some(block)
    }

    #[inline(always)]
    pub(crate) fn find_dimension(&self, dimension: u32) -> Option<usize> {
        // Match the standard library's branch-minimized binary search. Term
        // lookups dominate wide sparse queries, and a three-way branch at
        // every level is measurably slower on mmap-backed headers.
        let mut size = self.num_terms;
        let mut base = 0usize;
        while size > 1 {
            let half = size / 2;
            let middle = base + half;
            // SAFETY: `middle < num_terms`; parse validated the complete
            // dimension array once when constructing this block view.
            let value = unsafe { self.dimension_unchecked(middle) };
            base = std::hint::select_unpredictable(value > dimension, base, middle);
            size -= half;
        }
        // SAFETY: AdaptiveBlock::parse rejects zero-term blocks, and the loop
        // maintains `base < num_terms`.
        if unsafe { self.dimension_unchecked(base) } == dimension {
            Some(base)
        } else {
            None
        }
    }

    /// Early-exit binary search for narrow queries. When most looked-up terms
    /// are present, returning at the matching level avoids enough header loads
    /// to beat the fixed-depth branch-minimized search.
    #[inline(always)]
    pub(crate) fn find_dimension_branching(&self, dimension: u32) -> Option<usize> {
        let mut low = 0usize;
        let mut high = self.num_terms;
        while low < high {
            let middle = low + (high - low) / 2;
            // SAFETY: `middle < num_terms`; parse validated the dimension
            // array once when constructing this view.
            let value = unsafe { self.dimension_unchecked(middle) };
            match value.cmp(&dimension) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Equal => return Some(middle),
                std::cmp::Ordering::Greater => high = middle,
            }
        }
        None
    }

    #[inline(always)]
    pub(crate) fn dimension(&self, term: usize) -> Option<u32> {
        if term >= self.num_terms {
            return None;
        }
        // SAFETY: The term bound and parsed header envelope cover this read.
        Some(unsafe { self.dimension_unchecked(term) })
    }

    /// Reconstruct the conservative u8 representative of the packed u4 max.
    ///
    /// Re-quantizing this value to either a u4 or u2 pruning grid gives the
    /// same cell as quantizing the original exact maximum.
    #[inline(always)]
    pub(crate) fn max_impact(&self, term: usize) -> Option<u8> {
        if term >= self.num_terms {
            return None;
        }
        // SAFETY: The term bound and parsed header envelope cover this byte.
        let packed = unsafe { *self.bytes.get_unchecked(self.maxima_start + term / 2) };
        let quantized = if term.is_multiple_of(2) {
            packed & 0x0f
        } else {
            packed >> 4
        };
        Some(quantized * MAXIMUM_QUANTUM)
    }

    #[inline(always)]
    pub(crate) fn postings(&self, term: usize) -> Option<AdaptivePostings<'a>> {
        if term >= self.num_terms {
            return None;
        }
        // SAFETY: Both entries belong to the parsed offset table.
        let (start, dense) = unsafe { self.offset_unchecked(term) };
        let (end, _) = unsafe { self.offset_unchecked(term + 1) };
        let payload_len = self.bytes.len() - self.payload_start;
        if end <= start || end > payload_len {
            return None;
        }
        // SAFETY: `end <= payload_len` proves the complete range is inside
        // the block. `start < end` proves its start is inside as well.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.bytes.as_ptr().add(self.payload_start + start),
                end - start,
            )
        };
        if dense {
            if bytes.len() != self.block_size {
                return None;
            }
            Some(AdaptivePostings::Dense(bytes))
        } else {
            if !bytes.len().is_multiple_of(2) {
                return None;
            }
            // SAFETY: BmpPosting contains two u8 fields and therefore has
            // alignment one. The byte count was checked to be even.
            let postings = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr().cast::<BmpPosting>(), bytes.len() / 2)
            };
            Some(AdaptivePostings::Sparse(postings))
        }
    }

    #[inline]
    pub(crate) fn terms(self) -> AdaptiveTermIter<'a> {
        AdaptiveTermIter {
            block: self,
            current: 0,
        }
    }

    /// Exhaustively validate a block before background rewrite or graph work.
    #[cfg(any(feature = "native", test))]
    pub(crate) fn validate(self, dimensions: u32) -> Result<(), String> {
        if self.num_terms == 0 {
            return Err("block contains no terms".into());
        }
        if self.num_terms % 2 == 1 && self.bytes[self.maxima_start + self.num_terms / 2] & 0xf0 != 0
        {
            return Err("unused packed-maximum nibble is non-zero".into());
        }

        let mut previous_dimension = None;
        for term in 0..self.num_terms {
            let dimension = self
                .dimension(term)
                .ok_or_else(|| format!("term {term} has no dimension"))?;
            if dimension >= dimensions
                || previous_dimension.is_some_and(|previous| dimension <= previous)
            {
                return Err(format!(
                    "term {term} has invalid/non-increasing dimension {dimension}"
                ));
            }
            previous_dimension = Some(dimension);

            let postings = self
                .postings(term)
                .ok_or_else(|| format!("term {term} has an invalid payload range"))?;
            let mut observed_max = 0u8;
            let mut count = 0usize;
            match postings {
                AdaptivePostings::Sparse(values) => {
                    for posting in values {
                        if posting.local_slot as usize >= self.block_size {
                            return Err(format!(
                                "term {term} has local slot {} outside block size {}",
                                posting.local_slot, self.block_size
                            ));
                        }
                        if posting.impact == 0 {
                            return Err(format!("term {term} has a zero sparse impact"));
                        }
                        observed_max = observed_max.max(posting.impact);
                        count += 1;
                    }
                }
                AdaptivePostings::Dense(values) => {
                    for &impact in values {
                        if impact != 0 {
                            observed_max = observed_max.max(impact);
                            count += 1;
                        }
                    }
                }
            }
            if count == 0 {
                return Err(format!("term {term} has no postings"));
            }
            let stored = self
                .max_impact(term)
                .ok_or_else(|| format!("term {term} has no maximum"))?;
            if stored != u4_representative(observed_max) {
                return Err(format!(
                    "term {term} packed maximum {stored} does not cover observed {observed_max}"
                ));
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn offset(&self, term: usize) -> Option<(usize, bool)> {
        if term > self.num_terms {
            return None;
        }
        // SAFETY: The term bound and parsed header envelope cover this entry.
        Some(unsafe { self.offset_unchecked(term) })
    }

    #[inline(always)]
    unsafe fn dimension_unchecked(&self, term: usize) -> u32 {
        unsafe { read_u32_unchecked(self.bytes, self.dimensions_start + term * 4) }
    }

    #[inline(always)]
    unsafe fn offset_unchecked(&self, term: usize) -> (usize, bool) {
        let byte = self.offsets_start + term * self.offset_width;
        if self.offset_width == 2 {
            let raw = unsafe { read_u16_unchecked(self.bytes, byte) };
            (
                (raw & NARROW_OFFSET_MASK) as usize,
                raw & NARROW_DENSE_FLAG != 0,
            )
        } else {
            let raw = unsafe { read_u32_unchecked(self.bytes, byte) };
            (
                (raw & WIDE_OFFSET_MASK) as usize,
                raw & WIDE_DENSE_FLAG != 0,
            )
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum AdaptivePostings<'a> {
    Sparse(&'a [BmpPosting]),
    Dense(&'a [u8]),
}

impl AdaptivePostings<'_> {
    #[cfg(any(feature = "native", test))]
    #[inline]
    pub(crate) fn len(self) -> usize {
        match self {
            Self::Sparse(postings) => postings.len(),
            Self::Dense(impacts) => impacts.iter().filter(|&&impact| impact != 0).count(),
        }
    }
}

pub(crate) enum AdaptivePostingIter<'a> {
    Sparse(std::iter::Copied<std::slice::Iter<'a, BmpPosting>>),
    Dense(std::iter::Enumerate<std::slice::Iter<'a, u8>>),
}

impl Iterator for AdaptivePostingIter<'_> {
    type Item = BmpPosting;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Sparse(postings) => postings.next(),
            Self::Dense(impacts) => {
                for (slot, &impact) in impacts.by_ref() {
                    if impact != 0 {
                        return Some(BmpPosting {
                            local_slot: slot as u8,
                            impact,
                        });
                    }
                }
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Sparse(postings) => postings.size_hint(),
            Self::Dense(impacts) => (0, Some(impacts.len())),
        }
    }
}

impl<'a> IntoIterator for AdaptivePostings<'a> {
    type Item = BmpPosting;
    type IntoIter = AdaptivePostingIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::Sparse(postings) => AdaptivePostingIter::Sparse(postings.iter().copied()),
            Self::Dense(impacts) => AdaptivePostingIter::Dense(impacts.iter().enumerate()),
        }
    }
}

pub(crate) struct AdaptiveTermIter<'a> {
    block: AdaptiveBlock<'a>,
    current: usize,
}

impl<'a> Iterator for AdaptiveTermIter<'a> {
    type Item = (u32, u8, AdaptivePostings<'a>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.block.num_terms {
            return None;
        }
        let term = self.current;
        self.current += 1;
        Some((
            self.block.dimension(term)?,
            self.block.max_impact(term)?,
            self.block.postings(term)?,
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.block.num_terms - self.current;
        (remaining, Some(remaining))
    }
}

#[inline]
fn quantize_maximum(maximum: u8) -> u8 {
    u4_representative(maximum) / MAXIMUM_QUANTUM
}

#[inline]
fn u4_representative(maximum: u8) -> u8 {
    if maximum == 0 {
        0
    } else {
        u16::from(maximum).div_ceil(u16::from(MAXIMUM_QUANTUM)) as u8 * MAXIMUM_QUANTUM
    }
}

#[inline]
fn read_u32(bytes: &[u8], start: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(start..start.checked_add(4)?)?.try_into().ok()?,
    ))
}

#[inline(always)]
unsafe fn read_u16_unchecked(bytes: &[u8], start: usize) -> u16 {
    unsafe { u16::from_le((bytes.as_ptr().add(start) as *const u16).read_unaligned()) }
}

#[inline(always)]
unsafe fn read_u32_unchecked(bytes: &[u8], start: usize) -> u32 {
    unsafe { u32::from_le((bytes.as_ptr().add(start) as *const u32).read_unaligned()) }
}

fn invalid_data(message: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(
        block_size: usize,
        dimensions: &[u32],
        counts: &[u32],
        maxima: &[u8],
        postings: &[u8],
    ) -> Vec<u8> {
        let mut scratch = AdaptiveEncodeScratch::default();
        let mut bytes = Vec::new();
        scratch
            .encode(
                block_size, true, dimensions, counts, maxima, postings, &mut bytes,
            )
            .unwrap();
        bytes
    }

    #[test]
    fn narrow_sparse_and_dense_terms_round_trip() {
        let mut postings = vec![1, 19, 7, 255];
        for slot in 0..16u8 {
            postings.extend_from_slice(&[slot, slot.saturating_add(1)]);
        }
        let bytes = encode(32, &[3, 9], &[2, 16], &[255, 16], &postings);
        let block = AdaptiveBlock::parse(&bytes, 32).unwrap();
        assert_eq!(block.num_terms, 2);
        assert_eq!(block.find_dimension(3), Some(0));
        assert_eq!(block.find_dimension(9), Some(1));
        assert_eq!(block.find_dimension(8), None);
        assert_eq!(block.max_impact(0), Some(255));
        assert_eq!(block.max_impact(1), Some(17));
        assert!(matches!(
            block.postings(0),
            Some(AdaptivePostings::Sparse(_))
        ));
        assert!(matches!(
            block.postings(1),
            Some(AdaptivePostings::Dense(_))
        ));
        assert_eq!(
            block.postings(0).unwrap().into_iter().collect::<Vec<_>>(),
            vec![
                BmpPosting {
                    local_slot: 1,
                    impact: 19
                },
                BmpPosting {
                    local_slot: 7,
                    impact: 255
                }
            ]
        );
        assert_eq!(block.postings(1).unwrap().len(), 16);
        block.validate(10).unwrap();
    }

    #[test]
    fn duplicate_slots_stay_sparse() {
        let postings = [1, 10, 1, 20, 2, 30, 2, 40];
        let bytes = encode(4, &[7], &[4], &[40], &postings);
        let block = AdaptiveBlock::parse(&bytes, 4).unwrap();
        assert!(matches!(
            block.postings(0),
            Some(AdaptivePostings::Sparse(_))
        ));
        assert_eq!(block.postings(0).unwrap().len(), 4);
    }

    #[test]
    fn adaptive_encoding_never_exceeds_v18_flat_blocks() {
        for num_terms in 1..=128usize {
            let dimensions: Vec<u32> = (0..num_terms as u32).collect();
            let counts: Vec<u32> = (0..num_terms)
                .map(|term| (term * 7 % 32 + 1) as u32)
                .collect();
            let maxima = vec![255; num_terms];
            let mut postings = Vec::new();
            for &count in &counts {
                for slot in 0..count as u8 {
                    postings.extend_from_slice(&[slot, 255]);
                }
            }
            let bytes = encode(32, &dimensions, &counts, &maxima, &postings);
            let v18_bytes = 8 + 9 * num_terms + postings.len();
            assert!(
                bytes.len() <= v18_bytes,
                "terms={num_terms}: V19={} > V18={v18_bytes}",
                bytes.len(),
            );
        }
    }

    #[test]
    fn wide_offsets_round_trip() {
        let dimensions: Vec<u32> = (0..130).collect();
        let counts = vec![128; dimensions.len()];
        let maxima = vec![255; dimensions.len()];
        let mut postings = Vec::with_capacity(dimensions.len() * 256);
        for _ in &dimensions {
            for slot in 0..128u8 {
                postings.extend_from_slice(&[slot, 255]);
            }
        }
        let bytes = encode(256, &dimensions, &counts, &maxima, &postings);
        assert_ne!(
            u32::from_le_bytes(bytes[..4].try_into().unwrap()) & WIDE_BLOCK_FLAG,
            0
        );
        let block = AdaptiveBlock::parse(&bytes, 256).unwrap();
        assert_eq!(block.find_dimension(129), Some(129));
        assert_eq!(block.postings(129).unwrap().len(), 128);
        block.validate(130).unwrap();
    }

    #[test]
    fn packed_u4_requantizes_identically_for_u4_and_u2_grids() {
        for exact in 0..=u8::MAX {
            let representative = u4_representative(exact);
            assert_eq!(quantize_maximum(exact), quantize_maximum(representative));
            let exact_u2 = if exact == 0 {
                0
            } else {
                (u16::from(exact) * 3).div_ceil(255) as u8
            };
            let representative_u2 = if representative == 0 {
                0
            } else {
                (u16::from(representative) * 3).div_ceil(255) as u8
            };
            assert_eq!(exact_u2, representative_u2);
        }
    }

    #[test]
    fn malformed_offset_is_rejected_without_panicking() {
        let mut bytes = encode(32, &[1], &[1], &[7], &[3, 7]);
        // Narrow sentinel follows raw count, one dimension, and one start.
        bytes[10..12].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(AdaptiveBlock::parse(&bytes, 32).is_none());
    }

    #[test]
    fn branch_minimized_dimension_search_matches_slice_binary_search() {
        for len in 1..=300usize {
            let dimensions: Vec<u32> = (0..len as u32).map(|value| value * 3 + 1).collect();
            let counts = vec![1; len];
            let maxima = vec![1; len];
            let postings = [0, 1].repeat(len);
            let bytes = encode(32, &dimensions, &counts, &maxima, &postings);
            let block = AdaptiveBlock::parse(&bytes, 32).unwrap();
            for target in 0..=(len as u32 * 3 + 2) {
                let expected = dimensions.binary_search(&target).ok();
                assert_eq!(
                    block.find_dimension(target),
                    expected,
                    "length={len}, target={target}",
                );
                assert_eq!(
                    block.find_dimension_branching(target),
                    expected,
                    "branching length={len}, target={target}",
                );
            }
        }
    }
}
