//! Random-access bit-packed BMP maximum grids.
//!
//! Every dimension is stored as independently decodable groups of 256 cells.
//! A row header contains one compact metadata record per 32 groups:
//!
//! ```text
//! checkpoint: u32-LE       // payload offset in 32-byte units
//! widths:     packed-u4    // local bit width for each 256-cell group
//! ```
//!
//! The payload for a group is exactly `width * 32` bytes. Thus one metadata
//! record and a bounded sum of at most 31 width nibbles locate any group
//! without walking earlier payload. All-zero groups have width zero and no
//! payload. Row offsets are a small, separately pinnable `u64` table.

use std::io::Write;

use crate::directories::OwnedBytes;

pub(crate) const GRID_GROUP_CELLS: usize = 256;
/// LSP/0 uses four-bit maximum weights at both pruning levels.
pub(crate) const LSP_SUPERBLOCK_GRID_BITS: u8 = 4;
const META_GROUPS: usize = 32;
const META_SELECTOR_BYTES: usize = META_GROUPS / 2;
const META_RECORD_BYTES: usize = 4 + META_SELECTOR_BYTES;
const PAYLOAD_UNIT_BYTES: usize = GRID_GROUP_CELLS / 8;

#[inline]
fn selector_bytes(groups: usize) -> usize {
    groups.div_ceil(2)
}

#[inline]
fn selector_width(selectors: &[u8], group: usize) -> u8 {
    let byte = selectors[group / 2];
    if group.is_multiple_of(2) {
        byte & 0x0f
    } else {
        byte >> 4
    }
}

/// Dequantization multiplier for a block-grid cell.
#[inline]
pub(crate) fn block_grid_scale(grid_bits: u8) -> u32 {
    match grid_bits {
        2 => 85,
        _ => 17,
    }
}

/// Ceiling-quantize an exact u8 maximum without underestimating it.
#[inline]
pub(crate) fn quantize_block_maximum(value: u8, grid_bits: u8) -> u8 {
    if value == 0 {
        return 0;
    }
    match grid_bits {
        2 => (u16::from(value) * 3).div_ceil(255) as u8,
        _ => (u16::from(value) * 15).div_ceil(255) as u8,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CompressedGridLayout {
    dims: usize,
    cells: usize,
    groups: usize,
}

impl CompressedGridLayout {
    pub(crate) fn new(dims: usize, cells: usize) -> Self {
        let groups = cells.div_ceil(GRID_GROUP_CELLS);
        Self {
            dims,
            cells,
            groups,
        }
    }

    #[inline]
    pub(crate) fn dims(self) -> usize {
        self.dims
    }

    #[inline]
    pub(crate) fn cells(self) -> usize {
        self.cells
    }

    #[inline]
    pub(crate) fn groups(self) -> usize {
        self.groups
    }

    #[inline]
    pub(crate) fn row_header_bytes(self) -> usize {
        let complete = self.groups / META_GROUPS;
        let remainder = self.groups % META_GROUPS;
        complete * META_RECORD_BYTES
            + usize::from(remainder != 0) * (std::mem::size_of::<u32>() + selector_bytes(remainder))
    }

    pub(crate) fn row_bytes(self, widths: &[u8]) -> crate::Result<u64> {
        if widths.len() != self.groups {
            return Err(crate::Error::Internal(format!(
                "BMP compressed-grid width count {} != expected {}",
                widths.len(),
                self.groups
            )));
        }
        let payload_units = widths.iter().try_fold(0u64, |total, &width| {
            total.checked_add(u64::from(width)).ok_or_else(|| {
                crate::Error::Internal("BMP compressed-grid row size overflows u64".into())
            })
        })?;
        let payload_bytes = payload_units
            .checked_mul(PAYLOAD_UNIT_BYTES as u64)
            .ok_or_else(|| {
                crate::Error::Internal("BMP compressed-grid row size overflows u64".into())
            })?;
        (self.row_header_bytes() as u64)
            .checked_add(payload_bytes)
            .ok_or_else(|| {
                crate::Error::Internal("BMP compressed-grid row size overflows u64".into())
            })
    }

    /// Write the row-offset table. Offsets are relative to the first row byte.
    pub(crate) fn write_row_offsets(
        self,
        row_sizes: &[u64],
        writer: &mut dyn Write,
    ) -> std::io::Result<u64> {
        debug_assert_eq!(row_sizes.len(), self.dims);
        let mut offset = 0u64;
        writer.write_all(&offset.to_le_bytes())?;
        for &row_size in row_sizes {
            offset = offset.checked_add(row_size).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "BMP compressed-grid section exceeds u64",
                )
            })?;
            writer.write_all(&offset.to_le_bytes())?;
        }
        Ok((row_sizes.len() as u64 + 1) * 8)
    }

    /// Write the fixed-size metadata header for one row.
    pub(crate) fn write_row_header(
        self,
        widths: &[u8],
        max_width: u8,
        writer: &mut dyn Write,
    ) -> std::io::Result<()> {
        if widths.len() != self.groups {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid BMP compressed-grid width count",
            ));
        }
        let mut payload_units = 0u32;
        for chunk in widths.chunks(META_GROUPS) {
            writer.write_all(&payload_units.to_le_bytes())?;
            let mut packed_selectors = [0u8; META_SELECTOR_BYTES];
            for (index, &width) in chunk.iter().enumerate() {
                if width > max_width {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("BMP compressed-grid width {width} exceeds maximum {max_width}"),
                    ));
                }
                payload_units = payload_units.checked_add(u32::from(width)).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "BMP compressed-grid row payload exceeds u32 units",
                    )
                })?;
                packed_selectors[index / 2] |= width << ((index % 2) * 4);
            }
            writer.write_all(&packed_selectors[..selector_bytes(chunk.len())])?;
        }
        Ok(())
    }
}

/// One independently addressable 256-cell payload.
#[derive(Clone, Copy)]
pub(crate) struct PackedGridGroup<'a> {
    width: u8,
    bytes: &'a [u8],
}

impl<'a> PackedGridGroup<'a> {
    #[inline]
    pub(crate) fn width(self) -> u8 {
        self.width
    }

    #[inline]
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub(crate) fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Decode a range within this group into one byte per value.
    pub(crate) fn decode(self, start: usize, count: usize, output: &mut [u8]) {
        debug_assert!(start + count <= GRID_GROUP_CELLS);
        debug_assert!(output.len() >= count);
        if self.width == 0 {
            output[..count].fill(0);
            return;
        }
        if self.width == 8 {
            output[..count].copy_from_slice(&self.bytes[start..start + count]);
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("bmi2") && (start * usize::from(self.width)).is_multiple_of(8) {
            // SAFETY: BMI2 was detected at runtime. Each complete chunk reads
            // exactly `width` payload bytes and writes eight output bytes.
            unsafe {
                decode_u8_bmi2(self.bytes, start, count, self.width, output);
            }
            return;
        }

        let width = usize::from(self.width);
        let mask = (1u64 << width) - 1;
        let aligned_values = if (start * width).is_multiple_of(8) {
            count / 8 * 8
        } else {
            0
        };
        if aligned_values != 0 {
            let source_start = start * width / 8;
            for chunk in 0..aligned_values / 8 {
                let packed = load_packed_eight(self.bytes, source_start + chunk * width, width);
                unpack_eight_u8(packed, width, mask, &mut output[chunk * 8..chunk * 8 + 8]);
            }
        }
        for (index, value) in output[aligned_values..count].iter_mut().enumerate() {
            let index = aligned_values + index;
            let bit = (start + index) * width;
            let byte = bit / 8;
            let shift = bit % 8;
            let low = u16::from(self.bytes[byte]);
            let high = self
                .bytes
                .get(byte + 1)
                .copied()
                .map(u16::from)
                .unwrap_or(0);
            *value = (((low | (high << 8)) >> shift) & mask as u16) as u8;
        }
    }

    /// Decode a u2/u4 block-grid range into ordinary packed-u4 bytes.
    ///
    /// BMP visits eight-cell ranges aligned inside a 256-cell group. Width four
    /// is a direct copy; x86 BMI2 expands widths one through three eight cells
    /// at a time. A bounded eight-cell unpacker handles AArch64 and the
    /// portable fallback without per-cell division.
    pub(crate) fn decode_u4_packed(self, start: usize, count: usize, output: &mut [u8]) {
        debug_assert!(self.width <= 4);
        debug_assert!(start + count <= GRID_GROUP_CELLS);
        let output_len = count.div_ceil(2);
        debug_assert!(output.len() >= output_len);
        output[..output_len].fill(0);
        if self.width == 0 || count == 0 {
            return;
        }
        if self.width == 4 && start.is_multiple_of(2) {
            output[..output_len].copy_from_slice(&self.bytes[start / 2..start / 2 + output_len]);
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("bmi2")
            && (start * usize::from(self.width)).is_multiple_of(8)
            && count.is_multiple_of(8)
        {
            // SAFETY: BMI2 was detected at runtime. The format guarantees a
            // width*32-byte payload, and each eight-cell chunk consumes
            // exactly `width` bytes and emits four bytes.
            unsafe {
                decode_u4_packed_bmi2(self.bytes, start, count, self.width, output);
            }
            return;
        }

        let width = usize::from(self.width);
        let mask = (1u64 << width) - 1;
        let aligned_values = if (start * width).is_multiple_of(8) {
            count / 8 * 8
        } else {
            0
        };
        if aligned_values != 0 {
            let source_start = start * width / 8;
            for chunk in 0..aligned_values / 8 {
                let packed = load_packed_eight(self.bytes, source_start + chunk * width, width);
                let nibbles = pack_eight_u4(packed, width, mask).to_le_bytes();
                output[chunk * 4..chunk * 4 + 4].copy_from_slice(&nibbles);
            }
        }
        for index in aligned_values..count {
            let bit = (start + index) * width;
            let byte = bit / 8;
            let shift = bit % 8;
            let low = u16::from(self.bytes[byte]);
            let high = self
                .bytes
                .get(byte + 1)
                .copied()
                .map(u16::from)
                .unwrap_or(0);
            let value = (((low | high << 8) >> shift) & mask as u16) as u8;
            if index.is_multiple_of(2) {
                output[index / 2] |= value;
            } else {
                output[index / 2] |= value << 4;
            }
        }
    }
}

/// Load the packed representation of eight consecutive cells.
///
/// Eight width-`w` values occupy exactly `w` bytes, so this never reads past
/// the final chunk. Chunking removes division, modulus, and cross-byte loads
/// from the AArch64/portable per-cell loop.
#[inline(always)]
fn load_packed_eight(payload: &[u8], source: usize, width: usize) -> u64 {
    debug_assert!((1..=7).contains(&width));
    debug_assert!(source + width <= payload.len());
    let mut packed = 0u64;
    // SAFETY: callers request complete eight-cell chunks and `group()`
    // validates the width-sized payload.
    unsafe {
        std::ptr::copy_nonoverlapping(
            payload.as_ptr().add(source),
            (&mut packed as *mut u64).cast::<u8>(),
            width,
        );
    }
    u64::from_le(packed)
}

#[inline(always)]
fn unpack_eight_u8(packed: u64, width: usize, mask: u64, output: &mut [u8]) {
    debug_assert!(output.len() >= 8);
    output[0] = (packed & mask) as u8;
    output[1] = (packed >> width & mask) as u8;
    output[2] = (packed >> (width * 2) & mask) as u8;
    output[3] = (packed >> (width * 3) & mask) as u8;
    output[4] = (packed >> (width * 4) & mask) as u8;
    output[5] = (packed >> (width * 5) & mask) as u8;
    output[6] = (packed >> (width * 6) & mask) as u8;
    output[7] = (packed >> (width * 7) & mask) as u8;
}

#[inline(always)]
fn pack_eight_u4(packed: u64, width: usize, mask: u64) -> u32 {
    (packed & mask) as u32
        | ((packed >> width & mask) as u32) << 4
        | ((packed >> (width * 2) & mask) as u32) << 8
        | ((packed >> (width * 3) & mask) as u32) << 12
        | ((packed >> (width * 4) & mask) as u32) << 16
        | ((packed >> (width * 5) & mask) as u32) << 20
        | ((packed >> (width * 6) & mask) as u32) << 24
        | ((packed >> (width * 7) & mask) as u32) << 28
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn decode_u8_bmi2(payload: &[u8], start: usize, count: usize, width: u8, output: &mut [u8]) {
    use std::arch::x86_64::_pdep_u64;

    let deposit_mask = u64::MAX / 255 * ((1u64 << width) - 1);
    let source_start = start * usize::from(width) / 8;
    let chunks = count / 8;
    for chunk in 0..chunks {
        let source = source_start + chunk * usize::from(width);
        let mut packed = 0u64;
        std::ptr::copy_nonoverlapping(
            payload.as_ptr().add(source),
            (&mut packed as *mut u64).cast::<u8>(),
            usize::from(width),
        );
        let unpacked = _pdep_u64(packed, deposit_mask).to_le_bytes();
        let destination = chunk * 8;
        output
            .get_unchecked_mut(destination..destination + 8)
            .copy_from_slice(&unpacked);
    }

    let width_usize = usize::from(width);
    let value_mask = (1u16 << width) - 1;
    for index in chunks * 8..count {
        let bit = (start + index) * width_usize;
        let byte = bit / 8;
        let shift = bit % 8;
        let low = u16::from(*payload.get_unchecked(byte));
        let high = payload.get(byte + 1).copied().map(u16::from).unwrap_or(0);
        *output.get_unchecked_mut(index) = (((low | high << 8) >> shift) & value_mask) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn decode_u4_packed_bmi2(
    payload: &[u8],
    start: usize,
    count: usize,
    width: u8,
    output: &mut [u8],
) {
    use std::arch::x86_64::_pdep_u32;

    let deposit_mask = match width {
        1 => 0x1111_1111,
        2 => 0x3333_3333,
        3 => 0x7777_7777,
        _ => unreachable!(),
    };
    let source_start = start * usize::from(width) / 8;
    for chunk in 0..count / 8 {
        let source = source_start + chunk * usize::from(width);
        let packed = match width {
            1 => u32::from(*payload.get_unchecked(source)),
            2 => u32::from(u16::from_le_bytes([
                *payload.get_unchecked(source),
                *payload.get_unchecked(source + 1),
            ])),
            3 => {
                u32::from(*payload.get_unchecked(source))
                    | u32::from(*payload.get_unchecked(source + 1)) << 8
                    | u32::from(*payload.get_unchecked(source + 2)) << 16
            }
            _ => unreachable!(),
        };
        let unpacked = _pdep_u32(packed, deposit_mask).to_le_bytes();
        let destination = chunk * 4;
        output
            .get_unchecked_mut(destination..destination + 4)
            .copy_from_slice(&unpacked);
    }
}

/// Accumulate one packed-u4 range and return its non-zero-cell bitset.
///
/// The optional secondary accumulator is updated in the same SIMD pass for
/// BMP's phase-one dimensions. This is the only production u4 accumulation
/// kernel: compressed widths 1–3 are normalized by `decode_u4_packed`, and a
/// width-4 group uses the same representation directly.
#[inline]
pub(crate) fn accumulate_packed_u4(
    packed: &[u8],
    count: usize,
    multiplier: u32,
    output: &mut [u32],
    mut secondary: Option<&mut [u32]>,
) -> u64 {
    debug_assert!(count <= 64);
    debug_assert!(packed.len() >= count.div_ceil(2));
    debug_assert!(output.len() >= count);
    debug_assert!(
        secondary
            .as_ref()
            .is_none_or(|values| values.len() >= count)
    );

    #[cfg(target_arch = "aarch64")]
    {
        // AArch64 mandates NEON.
        let secondary_ptr = secondary.as_mut().map(|values| values.as_mut_ptr());
        // SAFETY: all pointers cover `count` elements and AArch64 has NEON.
        unsafe {
            accumulate_packed_u4_neon(
                packed,
                count,
                multiplier,
                output.as_mut_ptr(),
                secondary_ptr,
            )
        }
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.1") {
        let secondary_ptr = secondary.as_mut().map(|values| values.as_mut_ptr());
        // SAFETY: runtime detection proves SSE4.1 support; pointers cover
        // `count` elements.
        return unsafe {
            accumulate_packed_u4_sse41(
                packed,
                count,
                multiplier,
                output.as_mut_ptr(),
                secondary_ptr,
            )
        };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut presence = 0u64;
        for index in 0..count {
            let byte = packed[index / 2];
            let value = if index.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            };
            if value == 0 {
                continue;
            }
            presence |= 1u64 << index;
            let contribution = u32::from(value) * multiplier;
            output[index] += contribution;
            if let Some(values) = secondary.as_deref_mut() {
                values[index] += contribution;
            }
        }
        presence
    }
}

/// Add decoded grid cells to integer bounds.
#[inline]
pub(crate) fn accumulate_u8(values: &[u8], count: usize, multiplier: u32, output: &mut [u32]) {
    debug_assert!(values.len() >= count);
    debug_assert!(output.len() >= count);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 mandates NEON and all slices cover `count`.
        unsafe {
            accumulate_u8_neon(values.as_ptr(), count, multiplier, output.as_mut_ptr());
        }
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.1") {
        // SAFETY: runtime detection proves SSE4.1 support and all slices cover
        // `count`.
        unsafe {
            accumulate_u8_sse41(values.as_ptr(), count, multiplier, output.as_mut_ptr());
        }
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    for index in 0..count {
        let value = values[index];
        output[index] += u32::from(value) * multiplier;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn accumulate_u8_sse41(values: *const u8, count: usize, multiplier: u32, output: *mut u32) {
    use std::arch::x86_64::*;

    let zero = _mm_setzero_si128();
    let multiplier_vector = _mm_set1_epi32(multiplier as i32);
    let chunks = count / 16;
    for chunk in 0..chunks {
        let offset = chunk * 16;
        let bytes = _mm_loadu_si128(values.add(offset) as *const __m128i);
        let low = _mm_unpacklo_epi8(bytes, zero);
        let high = _mm_unpackhi_epi8(bytes, zero);
        let vectors = [
            _mm_unpacklo_epi16(low, zero),
            _mm_unpackhi_epi16(low, zero),
            _mm_unpacklo_epi16(high, zero),
            _mm_unpackhi_epi16(high, zero),
        ];
        for (vector, decoded) in vectors.into_iter().enumerate() {
            let destination = output.add(offset + vector * 4) as *mut __m128i;
            let contribution = _mm_mullo_epi32(decoded, multiplier_vector);
            _mm_storeu_si128(
                destination,
                _mm_add_epi32(_mm_loadu_si128(destination), contribution),
            );
        }
    }
    for index in chunks * 16..count {
        let value = *values.add(index);
        *output.add(index) += u32::from(value) * multiplier;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn accumulate_u8_neon(values: *const u8, count: usize, multiplier: u32, output: *mut u32) {
    use std::arch::aarch64::*;

    let chunks = count / 16;
    for chunk in 0..chunks {
        let offset = chunk * 16;
        let bytes = vld1q_u8(values.add(offset));
        let low = vmovl_u8(vget_low_u8(bytes));
        let high = vmovl_u8(vget_high_u8(bytes));
        let vectors = [
            vmovl_u16(vget_low_u16(low)),
            vmovl_u16(vget_high_u16(low)),
            vmovl_u16(vget_low_u16(high)),
            vmovl_u16(vget_high_u16(high)),
        ];
        for (vector, decoded) in vectors.into_iter().enumerate() {
            let destination = output.add(offset + vector * 4);
            vst1q_u32(
                destination,
                vmlaq_n_u32(vld1q_u32(destination), decoded, multiplier),
            );
        }
    }
    for index in chunks * 16..count {
        let value = *values.add(index);
        *output.add(index) += u32::from(value) * multiplier;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn spread_16_bits(mut value: u32) -> u32 {
    value = (value | value << 8) & 0x00ff_00ff;
    value = (value | value << 4) & 0x0f0f_0f0f;
    value = (value | value << 2) & 0x3333_3333;
    (value | value << 1) & 0x5555_5555
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn accumulate_packed_u4_sse41(
    packed: &[u8],
    count: usize,
    multiplier: u32,
    output: *mut u32,
    secondary: Option<*mut u32>,
) -> u64 {
    use std::arch::x86_64::*;

    let nibble_mask = _mm_set1_epi8(0x0f);
    let zero = _mm_setzero_si128();
    let multiplier_vector = _mm_set1_epi32(multiplier as i32);
    let chunks = count / 32;
    let mut presence = 0u64;

    for chunk in 0..chunks {
        let bytes = _mm_loadu_si128(packed.as_ptr().add(chunk * 16) as *const __m128i);
        let low = _mm_and_si128(bytes, nibble_mask);
        let high = _mm_and_si128(_mm_srli_epi16::<4>(bytes), nibble_mask);
        let low_mask = _mm_movemask_epi8(_mm_cmpgt_epi8(low, zero)) as u32;
        let high_mask = _mm_movemask_epi8(_mm_cmpgt_epi8(high, zero)) as u32;
        let chunk_presence = spread_16_bits(low_mask) | spread_16_bits(high_mask) << 1;
        presence |= u64::from(chunk_presence) << (chunk * 32);

        let first = _mm_unpacklo_epi8(low, high);
        let second = _mm_unpackhi_epi8(low, high);
        let first_low = _mm_unpacklo_epi8(first, zero);
        let first_high = _mm_unpackhi_epi8(first, zero);
        let second_low = _mm_unpacklo_epi8(second, zero);
        let second_high = _mm_unpackhi_epi8(second, zero);
        let vectors = [
            _mm_unpacklo_epi16(first_low, zero),
            _mm_unpackhi_epi16(first_low, zero),
            _mm_unpacklo_epi16(first_high, zero),
            _mm_unpackhi_epi16(first_high, zero),
            _mm_unpacklo_epi16(second_low, zero),
            _mm_unpackhi_epi16(second_low, zero),
            _mm_unpacklo_epi16(second_high, zero),
            _mm_unpackhi_epi16(second_high, zero),
        ];
        for (vector, values) in vectors.into_iter().enumerate() {
            let offset = chunk * 32 + vector * 4;
            let contribution = _mm_mullo_epi32(values, multiplier_vector);
            let destination = output.add(offset) as *mut __m128i;
            _mm_storeu_si128(
                destination,
                _mm_add_epi32(_mm_loadu_si128(destination), contribution),
            );
            if let Some(secondary) = secondary {
                let destination = secondary.add(offset) as *mut __m128i;
                _mm_storeu_si128(
                    destination,
                    _mm_add_epi32(_mm_loadu_si128(destination), contribution),
                );
            }
        }
    }

    // LSP uses eight blocks per superblock. Handle that common tail with two
    // vector accumulations instead of falling back to eight scalar updates.
    let mut base = chunks * 32;
    while base + 8 <= count {
        let packed_word = (packed.as_ptr().add(base / 2) as *const u32).read_unaligned();
        let bytes = _mm_cvtsi32_si128(packed_word as i32);
        let low = _mm_and_si128(bytes, nibble_mask);
        let high = _mm_and_si128(_mm_srli_epi16::<4>(bytes), nibble_mask);
        let interleaved = _mm_unpacklo_epi8(low, high);
        let widened = _mm_cvtepu8_epi16(interleaved);
        let vectors = [
            _mm_cvtepu16_epi32(widened),
            _mm_cvtepu16_epi32(_mm_srli_si128::<8>(widened)),
        ];
        for (vector, values) in vectors.into_iter().enumerate() {
            let offset = base + vector * 4;
            let contribution = _mm_mullo_epi32(values, multiplier_vector);
            let destination = output.add(offset) as *mut __m128i;
            _mm_storeu_si128(
                destination,
                _mm_add_epi32(_mm_loadu_si128(destination), contribution),
            );
            if let Some(secondary) = secondary {
                let destination = secondary.add(offset) as *mut __m128i;
                _mm_storeu_si128(
                    destination,
                    _mm_add_epi32(_mm_loadu_si128(destination), contribution),
                );
            }
        }
        for local_byte in 0..4 {
            let byte = *packed.get_unchecked(base / 2 + local_byte);
            let pair = u64::from(byte & 0x0f != 0) | (u64::from(byte >> 4 != 0) << 1);
            presence |= pair << (base + local_byte * 2);
        }
        base += 8;
    }
    for index in base..count {
        let byte = *packed.get_unchecked(index / 2);
        let value = if index.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        };
        if value == 0 {
            continue;
        }
        presence |= 1u64 << index;
        let contribution = u32::from(value) * multiplier;
        *output.add(index) += contribution;
        if let Some(secondary) = secondary {
            *secondary.add(index) += contribution;
        }
    }
    presence
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn accumulate_packed_u4_neon(
    packed: &[u8],
    count: usize,
    multiplier: u32,
    output: *mut u32,
    secondary: Option<*mut u32>,
) -> u64 {
    use std::arch::aarch64::*;

    let nibble_mask = vdupq_n_u8(0x0f);
    let chunks = count / 32;
    let mut presence = 0u64;
    for chunk in 0..chunks {
        let bytes = vld1q_u8(packed.as_ptr().add(chunk * 16));
        let low = vandq_u8(bytes, nibble_mask);
        let high = vshrq_n_u8::<4>(bytes);
        let first = vzip1q_u8(low, high);
        let second = vzip2q_u8(low, high);
        let first_low = vmovl_u8(vget_low_u8(first));
        let first_high = vmovl_u8(vget_high_u8(first));
        let second_low = vmovl_u8(vget_low_u8(second));
        let second_high = vmovl_u8(vget_high_u8(second));
        let vectors = [
            vmovl_u16(vget_low_u16(first_low)),
            vmovl_u16(vget_high_u16(first_low)),
            vmovl_u16(vget_low_u16(first_high)),
            vmovl_u16(vget_high_u16(first_high)),
            vmovl_u16(vget_low_u16(second_low)),
            vmovl_u16(vget_high_u16(second_low)),
            vmovl_u16(vget_low_u16(second_high)),
            vmovl_u16(vget_high_u16(second_high)),
        ];
        for (vector, values) in vectors.into_iter().enumerate() {
            let offset = chunk * 32 + vector * 4;
            let destination = output.add(offset);
            vst1q_u32(
                destination,
                vmlaq_n_u32(vld1q_u32(destination), values, multiplier),
            );
            if let Some(secondary) = secondary {
                let destination = secondary.add(offset);
                vst1q_u32(
                    destination,
                    vmlaq_n_u32(vld1q_u32(destination), values, multiplier),
                );
            }
        }

        // Presence is consumed as one u64 per query dimension. Extract both
        // nibble flags per byte in one pass; this avoids a NEON-to-GPR
        // reduction and rereading each packed byte.
        for local_byte in 0..16 {
            let byte = *packed.get_unchecked(chunk * 16 + local_byte);
            let pair = u64::from(byte & 0x0f != 0) | (u64::from(byte >> 4 != 0) << 1);
            presence |= pair << (chunk * 32 + local_byte * 2);
        }
    }

    // LSP uses eight blocks per superblock. A four-byte unaligned read is
    // sufficient for eight nibbles, and keeps the load in bounds even for the
    // final partial compressed-grid group.
    let mut base = chunks * 32;
    while base + 8 <= count {
        let packed_word = (packed.as_ptr().add(base / 2) as *const u32).read_unaligned();
        let bytes = vreinterpret_u8_u32(vdup_n_u32(packed_word));
        let low = vand_u8(bytes, vdup_n_u8(0x0f));
        let high = vshr_n_u8::<4>(bytes);
        let interleaved = vzip1_u8(low, high);
        let widened = vmovl_u8(interleaved);
        let vectors = [
            vmovl_u16(vget_low_u16(widened)),
            vmovl_u16(vget_high_u16(widened)),
        ];
        for (vector, values) in vectors.into_iter().enumerate() {
            let offset = base + vector * 4;
            let destination = output.add(offset);
            vst1q_u32(
                destination,
                vmlaq_n_u32(vld1q_u32(destination), values, multiplier),
            );
            if let Some(secondary) = secondary {
                let destination = secondary.add(offset);
                vst1q_u32(
                    destination,
                    vmlaq_n_u32(vld1q_u32(destination), values, multiplier),
                );
            }
        }
        for local_byte in 0..4 {
            let byte = *packed.get_unchecked(base / 2 + local_byte);
            let pair = u64::from(byte & 0x0f != 0) | (u64::from(byte >> 4 != 0) << 1);
            presence |= pair << (base + local_byte * 2);
        }
        base += 8;
    }
    for index in base..count {
        let byte = *packed.get_unchecked(index / 2);
        let value = if index.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        };
        if value == 0 {
            continue;
        }
        presence |= 1u64 << index;
        let contribution = u32::from(value) * multiplier;
        *output.add(index) += contribution;
        if let Some(secondary) = secondary {
            *secondary.add(index) += contribution;
        }
    }
    presence
}

/// Mmap-backed compressed grid section.
#[derive(Clone)]
pub(crate) struct CompressedGrid {
    row_offsets: OwnedBytes,
    rows: OwnedBytes,
    layout: CompressedGridLayout,
    max_width: u8,
}

impl CompressedGrid {
    pub(crate) fn empty() -> Self {
        Self {
            row_offsets: OwnedBytes::empty(),
            rows: OwnedBytes::empty(),
            layout: CompressedGridLayout::new(0, 0),
            max_width: 0,
        }
    }

    pub(crate) fn parse(
        section: OwnedBytes,
        dims: usize,
        cells: usize,
        max_width: u8,
        label: &str,
    ) -> crate::Result<Self> {
        let layout = CompressedGridLayout::new(dims, cells);
        let table_bytes = dims
            .checked_add(1)
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| {
                crate::Error::Corruption(format!("{label} row-offset table overflows usize"))
            })?;
        if section.len() < table_bytes {
            return Err(crate::Error::Corruption(format!(
                "{label} is {} bytes, shorter than its {}-byte row-offset table",
                section.len(),
                table_bytes
            )));
        }
        let row_offsets = section.slice(0..table_bytes);
        let rows = section.slice(table_bytes..section.len());

        let offsets = row_offsets.as_slice();
        let mut previous = 0u64;
        for row in 0..=dims {
            let offset = row * 8;
            let current = u64::from_le_bytes(offsets[offset..offset + 8].try_into().unwrap());
            if current < previous || current > rows.len() as u64 {
                return Err(crate::Error::Corruption(format!(
                    "invalid {label} row offset {row}: {current} (previous={previous}, rows={})",
                    rows.len()
                )));
            }
            if row == 0 && current != 0 {
                return Err(crate::Error::Corruption(format!(
                    "{label} first row offset is {current}, expected 0"
                )));
            }
            if row > 0 && current - previous < layout.row_header_bytes() as u64 {
                return Err(crate::Error::Corruption(format!(
                    "{label} row {} is shorter than its {}-byte metadata header",
                    row - 1,
                    layout.row_header_bytes()
                )));
            }
            previous = current;
        }
        if previous != rows.len() as u64 {
            return Err(crate::Error::Corruption(format!(
                "{label} rows end at {previous}, section contains {} bytes",
                rows.len()
            )));
        }

        Ok(Self {
            row_offsets,
            rows,
            layout,
            max_width,
        })
    }

    #[inline]
    pub(crate) fn cells(&self) -> usize {
        self.layout.cells()
    }

    #[inline]
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub(crate) fn groups(&self) -> usize {
        self.layout.groups()
    }

    #[inline]
    pub(crate) fn dims(&self) -> usize {
        self.layout.dims()
    }

    #[inline]
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub(crate) fn max_width(&self) -> u8 {
        self.max_width
    }

    #[inline]
    fn row_range(&self, dimension: usize) -> crate::Result<(usize, usize)> {
        if dimension >= self.layout.dims() {
            return Err(crate::Error::Corruption(format!(
                "BMP compressed-grid dimension {dimension} exceeds {}",
                self.layout.dims()
            )));
        }
        let offsets = self.row_offsets.as_slice();
        let start_offset = dimension * 8;
        let start = u64::from_le_bytes(offsets[start_offset..start_offset + 8].try_into().unwrap());
        let end = u64::from_le_bytes(
            offsets[start_offset + 8..start_offset + 16]
                .try_into()
                .unwrap(),
        );
        Ok((start as usize, end as usize))
    }

    pub(crate) fn group(
        &self,
        dimension: usize,
        group: usize,
    ) -> crate::Result<PackedGridGroup<'_>> {
        if group >= self.layout.groups() {
            return Err(crate::Error::Corruption(format!(
                "BMP compressed-grid group {group} exceeds {}",
                self.layout.groups()
            )));
        }
        let (row_start, row_end) = self.row_range(dimension)?;
        let chunk = group / META_GROUPS;
        let within = group % META_GROUPS;
        let record_start = row_start + chunk * META_RECORD_BYTES;
        let selectors = META_GROUPS.min(self.layout.groups() - chunk * META_GROUPS);
        let record_end = record_start + std::mem::size_of::<u32>() + selector_bytes(selectors);
        if record_end > row_end {
            return Err(crate::Error::Corruption(
                "BMP compressed-grid metadata extends beyond row".into(),
            ));
        }
        let rows = self.rows.as_slice();
        let record = &rows[record_start..record_end];
        let checkpoint = u32::from_le_bytes(record[..4].try_into().unwrap()) as usize;
        if chunk == 0 && checkpoint != 0 {
            return Err(crate::Error::Corruption(format!(
                "BMP compressed-grid first checkpoint is {checkpoint}, expected 0"
            )));
        }
        let selectors = &record[4..];
        let width = selector_width(selectors, within);
        if width > self.max_width {
            return Err(crate::Error::Corruption(format!(
                "BMP compressed-grid width {width} exceeds {}",
                self.max_width
            )));
        }
        let mut preceding = 0usize;
        for &packed in &selectors[..within / 2] {
            let low = packed & 0x0f;
            let high = packed >> 4;
            if low > self.max_width || high > self.max_width {
                return Err(crate::Error::Corruption(format!(
                    "BMP compressed-grid selector {packed:#04x} exceeds width {}",
                    self.max_width
                )));
            }
            preceding += usize::from(low + high);
        }
        if !within.is_multiple_of(2) {
            let value = selectors[within / 2] & 0x0f;
            if value > self.max_width {
                return Err(crate::Error::Corruption(format!(
                    "BMP compressed-grid width {value} exceeds {}",
                    self.max_width
                )));
            }
            preceding += usize::from(value);
        }
        let payload_units = checkpoint.checked_add(preceding).ok_or_else(|| {
            crate::Error::Corruption("BMP compressed-grid payload offset overflows usize".into())
        })?;
        let payload_byte_offset =
            payload_units
                .checked_mul(PAYLOAD_UNIT_BYTES)
                .ok_or_else(|| {
                    crate::Error::Corruption(
                        "BMP compressed-grid payload byte offset overflows usize".into(),
                    )
                })?;
        let payload_start = row_start
            .checked_add(self.layout.row_header_bytes())
            .and_then(|base| base.checked_add(payload_byte_offset))
            .ok_or_else(|| {
                crate::Error::Corruption(
                    "BMP compressed-grid payload offset overflows usize".into(),
                )
            })?;
        let payload_len = usize::from(width)
            .checked_mul(PAYLOAD_UNIT_BYTES)
            .ok_or_else(|| {
                crate::Error::Corruption(
                    "BMP compressed-grid payload length overflows usize".into(),
                )
            })?;
        let payload_end = payload_start.checked_add(payload_len).ok_or_else(|| {
            crate::Error::Corruption("BMP compressed-grid payload end overflows usize".into())
        })?;
        if payload_end > row_end {
            return Err(crate::Error::Corruption(format!(
                "BMP compressed-grid payload {payload_start}..{payload_end} exceeds row end {row_end}"
            )));
        }
        Ok(PackedGridGroup {
            width,
            bytes: &rows[payload_start..payload_end],
        })
    }

    /// Visit every group in one row using a single sequential metadata pass.
    ///
    /// Full-row scans should use this rather than random `group()` calls:
    /// it advances one payload cursor, avoids repeated checkpoint-prefix
    /// sums, performs no allocation, and validates the complete row.
    pub(crate) fn try_for_each_row_group<'a>(
        &'a self,
        dimension: usize,
        mut visitor: impl FnMut(usize, PackedGridGroup<'a>) -> crate::Result<()>,
    ) -> crate::Result<()> {
        let (row_start, row_end) = self.row_range(dimension)?;
        let rows = self.rows.as_slice();
        let header_end = row_start
            .checked_add(self.layout.row_header_bytes())
            .ok_or_else(|| {
                crate::Error::Corruption("BMP compressed-grid row header overflows usize".into())
            })?;
        if header_end > row_end {
            return Err(crate::Error::Corruption(
                "BMP compressed-grid metadata extends beyond row".into(),
            ));
        }
        let mut payload = header_end;
        for group in 0..self.layout.groups() {
            let record = group / META_GROUPS;
            let within = group % META_GROUPS;
            if within == 0 {
                let record_start = row_start + record * META_RECORD_BYTES;
                let checkpoint =
                    u32::from_le_bytes(rows[record_start..record_start + 4].try_into().unwrap())
                        as usize;
                let expected = (payload - header_end) / PAYLOAD_UNIT_BYTES;
                if checkpoint != expected {
                    return Err(crate::Error::Corruption(format!(
                        "BMP compressed-grid checkpoint {checkpoint} != expected {expected}"
                    )));
                }
            }
            let selector = row_start + record * META_RECORD_BYTES + 4 + within / 2;
            if selector >= header_end {
                return Err(crate::Error::Corruption(
                    "BMP compressed-grid selector extends beyond row header".into(),
                ));
            }
            let selector = rows[selector];
            let width = if within.is_multiple_of(2) {
                selector & 0x0f
            } else {
                selector >> 4
            };
            if width > self.max_width {
                return Err(crate::Error::Corruption(format!(
                    "BMP compressed-grid width {width} exceeds {}",
                    self.max_width
                )));
            }
            let payload_end = payload
                .checked_add(usize::from(width) * PAYLOAD_UNIT_BYTES)
                .ok_or_else(|| {
                    crate::Error::Corruption(
                        "BMP compressed-grid row payload overflows usize".into(),
                    )
                })?;
            if payload_end > row_end {
                return Err(crate::Error::Corruption(format!(
                    "BMP compressed-grid payload {payload}..{payload_end} exceeds row end {row_end}"
                )));
            }
            visitor(
                group,
                PackedGridGroup {
                    width,
                    bytes: &rows[payload..payload_end],
                },
            )?;
            payload = payload_end;
        }
        if payload != row_end {
            return Err(crate::Error::Corruption(format!(
                "BMP compressed-grid row payload ends at {payload}, expected {row_end}"
            )));
        }
        Ok(())
    }

    /// Append a row's group descriptors for merge paths that combine several
    /// source rows.
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub(crate) fn append_row_groups<'a>(
        &'a self,
        dimension: usize,
        output: &mut Vec<PackedGridGroup<'a>>,
    ) -> crate::Result<()> {
        output.reserve(self.layout.groups());
        self.try_for_each_row_group(dimension, |_, group| {
            output.push(group);
            Ok(())
        })
    }

    pub(crate) fn encoded_bytes(&self) -> usize {
        self.row_offsets.len() + self.rows.len()
    }

    #[cfg(feature = "native")]
    pub(crate) fn madvise_rows(&self, advice: i32) {
        self.rows.madvise(advice);
    }

    #[cfg(feature = "native")]
    pub(crate) fn pin_offsets(
        &mut self,
        label: &'static str,
        mode: crate::segment::pin::PinMode,
        remaining: &mut u64,
        report: &mut crate::segment::pin::PinReport,
    ) {
        crate::segment::pin::pin_section(&mut self.row_offsets, label, mode, remaining, report);
    }

    #[cfg(feature = "native")]
    pub(crate) fn pin_all(
        &mut self,
        offsets_label: &'static str,
        rows_label: &'static str,
        mode: crate::segment::pin::PinMode,
        remaining: &mut u64,
        report: &mut crate::segment::pin::PinReport,
    ) {
        self.pin_offsets(offsets_label, mode, remaining, report);
        crate::segment::pin::pin_section(&mut self.rows, rows_label, mode, remaining, report);
    }
}

#[inline]
pub(crate) fn bit_width(value: u8) -> u8 {
    (u8::BITS - value.leading_zeros()) as u8
}

/// Pack exactly 256 values using the supplied local width.
///
/// Callers zero-pad the unused tail of the last logical group. The returned
/// payload always occupies `width * 32` bytes.
pub(crate) fn pack_group(
    values: &[u8; GRID_GROUP_CELLS],
    width: u8,
    output: &mut [u8; GRID_GROUP_CELLS],
) -> crate::Result<usize> {
    if width > 8 {
        return Err(crate::Error::Internal(format!(
            "BMP compressed-grid width {width} exceeds 8"
        )));
    }
    let output_len = usize::from(width) * PAYLOAD_UNIT_BYTES;
    output[..output_len].fill(0);
    if width == 0 {
        return Ok(0);
    }
    if width == 8 {
        output.copy_from_slice(values);
        return Ok(GRID_GROUP_CELLS);
    }
    let mask = (1u16 << width) - 1;
    if u16::from(values.iter().copied().max().unwrap_or(0)) > mask {
        return Err(crate::Error::Internal(format!(
            "BMP compressed-grid value does not fit width {width}"
        )));
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("bmi2") {
        // SAFETY: BMI2 was detected at runtime; input and output are fixed
        // 256-byte group buffers.
        unsafe {
            pack_group_bmi2(values, width, output);
        }
        return Ok(output_len);
    }

    let width = usize::from(width);
    for (chunk, values) in values.chunks_exact(8).enumerate() {
        let mut packed = 0u64;
        for (index, &value) in values.iter().enumerate() {
            packed |= u64::from(value) << (index * width);
        }
        let bytes = packed.to_le_bytes();
        let destination = chunk * width;
        output[destination..destination + width].copy_from_slice(&bytes[..width]);
    }
    Ok(output_len)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pack_group_bmi2(
    values: &[u8; GRID_GROUP_CELLS],
    width: u8,
    output: &mut [u8; GRID_GROUP_CELLS],
) {
    use std::arch::x86_64::_pext_u64;

    let extract_mask = u64::MAX / 255 * ((1u64 << width) - 1);
    for chunk in 0..GRID_GROUP_CELLS / 8 {
        let source = u64::from_le((values.as_ptr().add(chunk * 8) as *const u64).read_unaligned());
        let packed = _pext_u64(source, extract_mask).to_le_bytes();
        let destination = chunk * usize::from(width);
        output
            .get_unchecked_mut(destination..destination + usize::from(width))
            .copy_from_slice(&packed[..usize::from(width)]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded_section(widths_by_row: &[Vec<u8>], values_by_row: &[Vec<u8>]) -> Vec<u8> {
        let dims = widths_by_row.len();
        let cells = values_by_row[0].len();
        let layout = CompressedGridLayout::new(dims, cells);
        let row_sizes: Vec<u64> = widths_by_row
            .iter()
            .map(|widths| layout.row_bytes(widths).unwrap())
            .collect();
        let mut bytes = Vec::new();
        layout.write_row_offsets(&row_sizes, &mut bytes).unwrap();
        let mut packed = [0u8; GRID_GROUP_CELLS];
        let mut group_values = [0u8; GRID_GROUP_CELLS];
        for (widths, values) in widths_by_row.iter().zip(values_by_row) {
            layout.write_row_header(widths, 8, &mut bytes).unwrap();
            for (group, &width) in widths.iter().enumerate() {
                group_values.fill(0);
                let start = group * GRID_GROUP_CELLS;
                let count = GRID_GROUP_CELLS.min(cells - start);
                group_values[..count].copy_from_slice(&values[start..start + count]);
                let len = pack_group(&group_values, width, &mut packed).unwrap();
                bytes.extend_from_slice(&packed[..len]);
            }
        }
        bytes
    }

    #[test]
    fn every_width_round_trips_with_random_group_access() {
        let cells = GRID_GROUP_CELLS * 9 + 17;
        let mut values_by_row = Vec::new();
        let mut widths_by_row = Vec::new();
        for row in 0..9u8 {
            let width = row;
            let mask = if width == 0 { 0 } else { (1u16 << width) - 1 };
            let values: Vec<u8> = (0..cells)
                .map(|index| ((index * 37 + usize::from(row)) as u16 & mask) as u8)
                .collect();
            values_by_row.push(values);
            widths_by_row.push(vec![width; cells.div_ceil(GRID_GROUP_CELLS)]);
        }
        let bytes = encoded_section(&widths_by_row, &values_by_row);
        let grid = CompressedGrid::parse(OwnedBytes::new(bytes), 9, cells, 8, "test grid").unwrap();
        let mut decoded = [0u8; GRID_GROUP_CELLS];
        for (row, expected_row) in values_by_row.iter().enumerate() {
            for group in (0..grid.groups()).rev() {
                let count = GRID_GROUP_CELLS.min(cells - group * GRID_GROUP_CELLS);
                grid.group(row, group)
                    .unwrap()
                    .decode(0, count, &mut decoded);
                assert_eq!(
                    &decoded[..count],
                    &expected_row[group * GRID_GROUP_CELLS..group * GRID_GROUP_CELLS + count]
                );
                if count > 8 {
                    let range_start = 3;
                    let range_len = count - 7;
                    grid.group(row, group)
                        .unwrap()
                        .decode(range_start, range_len, &mut decoded);
                    assert_eq!(
                        &decoded[..range_len],
                        &expected_row[group * GRID_GROUP_CELLS + range_start
                            ..group * GRID_GROUP_CELLS + range_start + range_len]
                    );
                }
            }
        }
    }

    #[test]
    fn final_metadata_record_has_no_unused_selector_padding() {
        assert_eq!(META_RECORD_BYTES, 20);
        let one_group = CompressedGridLayout::new(1, GRID_GROUP_CELLS);
        assert_eq!(one_group.row_header_bytes(), 5);
        let thirty_two_groups = CompressedGridLayout::new(1, GRID_GROUP_CELLS * META_GROUPS);
        assert_eq!(thirty_two_groups.row_header_bytes(), 20);
        let thirty_three_groups =
            CompressedGridLayout::new(1, GRID_GROUP_CELLS * (META_GROUPS + 1));
        assert_eq!(thirty_three_groups.row_header_bytes(), 25);
    }

    #[test]
    fn malformed_width_is_rejected_on_access() {
        let layout = CompressedGridLayout::new(1, GRID_GROUP_CELLS);
        let mut bytes = Vec::new();
        layout.write_row_offsets(&[5], &mut bytes).unwrap();
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.push(9);
        let grid = CompressedGrid::parse(OwnedBytes::new(bytes), 1, 256, 8, "test").unwrap();
        assert!(matches!(grid.group(0, 0), Err(crate::Error::Corruption(_))));
    }

    #[test]
    fn malformed_checkpoint_is_rejected_on_access() {
        let cells = GRID_GROUP_CELLS * (META_GROUPS + 1);
        let layout = CompressedGridLayout::new(1, cells);
        let widths = vec![0u8; layout.groups()];
        let row_size = layout.row_bytes(&widths).unwrap();
        let mut bytes = Vec::new();
        layout.write_row_offsets(&[row_size], &mut bytes).unwrap();
        layout.write_row_header(&widths, 4, &mut bytes).unwrap();
        let second_record = 16 + META_RECORD_BYTES;
        bytes[second_record..second_record + 4].copy_from_slice(&1u32.to_le_bytes());
        let grid = CompressedGrid::parse(OwnedBytes::new(bytes), 1, cells, 4, "test").unwrap();
        assert!(matches!(
            grid.group(0, META_GROUPS),
            Err(crate::Error::Corruption(_))
        ));
    }

    #[test]
    fn every_block_width_decodes_to_packed_u4_at_each_superblock_offset() {
        let mut values = [0u8; GRID_GROUP_CELLS];
        let mut encoded = [0u8; GRID_GROUP_CELLS];
        let mut decoded = [0u8; 32];
        for width in 0..=4u8 {
            let mask = if width == 0 { 0 } else { (1u8 << width) - 1 };
            for (index, value) in values.iter_mut().enumerate() {
                *value = (index as u8).wrapping_mul(13) & mask;
            }
            let len = pack_group(&values, width, &mut encoded).unwrap();
            let group = PackedGridGroup {
                width,
                bytes: &encoded[..len],
            };
            for start in [0, 64, 128, 192] {
                group.decode_u4_packed(start, 64, &mut decoded);
                for index in 0..64 {
                    let byte = decoded[index / 2];
                    let actual = if index.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    assert_eq!(
                        actual,
                        values[start + index],
                        "width={width}, index={index}"
                    );
                }
            }
            group.decode_u4_packed(3, 59, &mut decoded);
            for index in 0..59 {
                let byte = decoded[index / 2];
                let actual = if index.is_multiple_of(2) {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                assert_eq!(
                    actual,
                    values[3 + index],
                    "unaligned width={width}, index={index}"
                );
            }
        }
    }

    #[test]
    fn fused_u4_accumulator_returns_presence_and_updates_both_outputs() {
        let values: Vec<u8> = (0..64)
            .map(|index| {
                if index % 7 == 0 {
                    0
                } else {
                    (index % 16) as u8
                }
            })
            .collect();
        let mut packed = [0u8; 32];
        for (index, &value) in values.iter().enumerate() {
            if index.is_multiple_of(2) {
                packed[index / 2] = value;
            } else {
                packed[index / 2] |= value << 4;
            }
        }
        let mut primary = [0u32; 64];
        let mut secondary = [11u32; 64];
        let presence = accumulate_packed_u4(
            &packed,
            values.len(),
            37,
            &mut primary,
            Some(&mut secondary),
        );
        for (index, &value) in values.iter().enumerate() {
            assert_eq!(primary[index], u32::from(value) * 37);
            assert_eq!(secondary[index], 11 + u32::from(value) * 37);
            assert_eq!((presence >> index) & 1, u64::from(value != 0));
        }
    }

    #[test]
    fn fused_u4_accumulator_handles_lsp_and_partial_tails() {
        for count in [1usize, 7, 8, 9, 15, 16, 24, 31, 32, 33, 63, 64] {
            let values: Vec<u8> = (0..count)
                .map(|index| ((index * 11 + 3) % 16) as u8)
                .collect();
            let mut packed = [0u8; 32];
            for (index, &value) in values.iter().enumerate() {
                if index.is_multiple_of(2) {
                    packed[index / 2] = value;
                } else {
                    packed[index / 2] |= value << 4;
                }
            }
            let mut primary = [5u32; 64];
            let mut secondary = [17u32; 64];
            let presence = accumulate_packed_u4(
                &packed[..count.div_ceil(2)],
                count,
                29,
                &mut primary[..count],
                Some(&mut secondary[..count]),
            );
            for (index, &value) in values.iter().enumerate() {
                assert_eq!(primary[index], 5 + u32::from(value) * 29);
                assert_eq!(secondary[index], 17 + u32::from(value) * 29);
                assert_eq!((presence >> index) & 1, u64::from(value != 0));
            }
        }
    }

    #[test]
    fn u8_accumulator_handles_high_bit_values() {
        let values: Vec<u8> = (0..257)
            .map(|index| match index % 5 {
                0 => 0,
                1 => 1,
                2 => 127,
                3 => 128,
                _ => 255,
            })
            .collect();
        let mut output = vec![9u32; values.len()];
        accumulate_u8(&values, values.len(), 41, &mut output);
        for (index, &value) in values.iter().enumerate() {
            assert_eq!(output[index], 9 + u32::from(value) * 41);
        }
    }
}
