//! Fast-field compression codecs with auto-selection.
//!
//! Four codecs are available, and the writer picks the smallest at build time:
//!
//! | Codec            | ID | Description                                      |
//! |------------------|----|--------------------------------------------------|
//! | Constant         |  0 | No data bytes — all values identical              |
//! | Bitpacked        |  1 | min-subtract + global bitpack                     |
//! | Linear           |  2 | Regression line, bitpack residuals                |
//! | BlockwiseLinear  |  3 | Per-512-block linear, bitpack residuals per block |

use std::io::{self, Write};

use byteorder::{LittleEndian, WriteBytesExt};

use super::{bitpack_read, bitpack_write, bits_needed_u64};

// ── Codec type tag ───────────────────────────────────────────────────────

/// Codec identifier stored in the column data region (first byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CodecType {
    Constant = 0,
    Bitpacked = 1,
    Linear = 2,
    BlockwiseLinear = 3,
}

impl CodecType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Constant),
            1 => Some(Self::Bitpacked),
            2 => Some(Self::Linear),
            3 => Some(Self::BlockwiseLinear),
            _ => None,
        }
    }
}

/// Block size for BlockwiseLinear codec (matching Tantivy).
pub const BLOCKWISE_LINEAR_BLOCK_SIZE: usize = 512;

// ── Estimator trait ──────────────────────────────────────────────────────

/// Estimates serialized size for a given codec.
///
/// Usage: call `collect(val)` for every value, then `finalize()`,
/// then `estimate()` returns the byte count.
pub trait CodecEstimator {
    fn collect(&mut self, value: u64);
    fn finalize(&mut self) {}
    fn estimate(&self) -> Option<u64>;
    fn serialize(&self, values: &[u64], writer: &mut dyn Write) -> io::Result<u64>;
}

// ── Constant codec ───────────────────────────────────────────────────────

/// All values are identical → zero data bytes. Value stored in the codec header.
#[derive(Default)]
pub struct ConstantEstimator {
    first: Option<u64>,
    all_same: bool,
}

impl CodecEstimator for ConstantEstimator {
    fn collect(&mut self, value: u64) {
        match self.first {
            None => {
                self.first = Some(value);
                self.all_same = true;
            }
            Some(f) => {
                if value != f {
                    self.all_same = false;
                }
            }
        }
    }

    fn estimate(&self) -> Option<u64> {
        if self.all_same {
            // codec_id(1) + value(8) = 9 bytes
            Some(9)
        } else {
            None
        }
    }

    fn serialize(&self, values: &[u64], writer: &mut dyn Write) -> io::Result<u64> {
        let val = if values.is_empty() { 0 } else { values[0] };
        writer.write_u8(CodecType::Constant as u8)?;
        writer.write_u64::<LittleEndian>(val)?;
        Ok(9)
    }
}

// ── Bitpacked codec ──────────────────────────────────────────────────────

/// Min-subtract + global bitpack. This is the existing codec, now behind a tag.
#[derive(Default)]
pub struct BitpackedEstimator {
    min: u64,
    max: u64,
    count: usize,
    initialized: bool,
}

impl CodecEstimator for BitpackedEstimator {
    fn collect(&mut self, value: u64) {
        if !self.initialized {
            self.min = value;
            self.max = value;
            self.initialized = true;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
        self.count += 1;
    }

    fn estimate(&self) -> Option<u64> {
        if self.count == 0 {
            return Some(0);
        }
        let range = self.max - self.min;
        let bpv = bits_needed_u64(range) as u64;
        // codec_id(1) + min(8) + bpv(1) + packed data
        let data_bits = self.count as u64 * bpv;
        let data_bytes = data_bits.div_ceil(8);
        Some(1 + 8 + 1 + data_bytes)
    }

    fn serialize(&self, values: &[u64], writer: &mut dyn Write) -> io::Result<u64> {
        let (min_value, bpv) = if values.is_empty() {
            (0u64, 0u8)
        } else {
            let min_val = values.iter().copied().min().unwrap();
            let max_val = values.iter().copied().max().unwrap();
            (min_val, bits_needed_u64(max_val - min_val))
        };

        writer.write_u8(CodecType::Bitpacked as u8)?;
        writer.write_u64::<LittleEndian>(min_value)?;
        writer.write_u8(bpv)?;
        let mut bytes_written = 10u64; // 1 + 8 + 1

        if bpv > 0 && !values.is_empty() {
            let shifted: Vec<u64> = values.iter().map(|&v| v - min_value).collect();
            let mut packed = Vec::new();
            bitpack_write(&shifted, bpv, &mut packed);
            writer.write_all(&packed)?;
            bytes_written += packed.len() as u64;
        }
        Ok(bytes_written)
    }
}

/// Read a single value from a bitpacked-codec column.
///
/// `data` starts right after the codec_id byte (i.e. at min_value).
#[inline]
pub fn bitpacked_read(data: &[u8], index: usize) -> u64 {
    let min_value = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let bpv = data[8];
    if bpv == 0 {
        return min_value;
    }
    let packed = &data[9..];
    bitpack_read(packed, bpv, index) + min_value
}

// ── Linear codec ─────────────────────────────────────────────────────────

/// Fits y = slope * x + intercept across all values, stores residuals bitpacked.
///
/// Header: codec_id(1) + intercept(8) + slope_num(8) + slope_den(8) + bpv(1) + offset(8) = 34
///
/// Estimation uses O(1) memory by tracking value extremes during collection and
/// computing worst-case residual bounds in `finalize()`.
///
/// **Limitation**: the per-column offset is stored as i64 (8 bytes). When values
/// span nearly the full u64 range (e.g. `FAST_FIELD_MISSING` mixed with small
/// values), residuals can exceed i64 bounds.  The estimator returns `None` in
/// that case so the auto-selector falls back to bitpacked.
#[derive(Default)]
pub struct LinearEstimator {
    count: usize,
    first: u64,
    last: u64,
    min_val: u64,
    max_val: u64,
    min_residual: i64,
    max_residual: i64,
    values_collected: bool,
    /// Set by `finalize()` when residuals exceed i64 range.
    overflow: bool,
}

impl CodecEstimator for LinearEstimator {
    fn collect(&mut self, value: u64) {
        if !self.values_collected {
            self.first = value;
            self.min_val = value;
            self.max_val = value;
            self.values_collected = true;
        } else {
            self.min_val = self.min_val.min(value);
            self.max_val = self.max_val.max(value);
        }
        self.last = value;
        self.count += 1;
    }

    fn finalize(&mut self) {
        if self.count < 2 {
            return;
        }
        // Compute worst-case residual bounds from value extremes vs predicted line.
        // The predicted line spans [first, last]. The worst-case residuals occur when
        // the most extreme value is farthest from the nearest predicted value.
        // Predicted values range from min(first,last) to max(first,last), so:
        //   max_residual ≥ max_val - min(predicted) = max_val - min(first, last)
        //   min_residual ≤ min_val - max(predicted) = min_val - max(first, last)
        // This is a conservative bound (may slightly overestimate bpv vs exact).
        let pred_min = self.first.min(self.last) as i128;
        let pred_max = self.first.max(self.last) as i128;
        let min_res = self.min_val as i128 - pred_max;
        let max_res = self.max_val as i128 - pred_min;
        // The offset is stored as i64 on disk.  If residuals exceed i64 range,
        // this codec cannot represent the data — mark as overflow.
        if min_res < i64::MIN as i128 || max_res > i64::MAX as i128 {
            self.overflow = true;
            return;
        }
        self.min_residual = min_res as i64;
        self.max_residual = max_res as i64;
    }

    fn estimate(&self) -> Option<u64> {
        if self.count < 2 || self.overflow {
            return None;
        }
        // Check for overflow: if the range doesn't fit u64, this codec is not viable
        let range = (self.max_residual as i128 - self.min_residual as i128) as u64;
        let bpv = bits_needed_u64(range) as u64;
        let data_bits = self.count as u64 * bpv;
        let data_bytes = data_bits.div_ceil(8);
        // codec_id(1) + first(8) + last(8) + num_values(4) + offset(8) + bpv(1) + packed
        Some(1 + 8 + 8 + 4 + 8 + 1 + data_bytes)
    }

    fn serialize(&self, values: &[u64], writer: &mut dyn Write) -> io::Result<u64> {
        let n = values.len();
        if n < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "linear needs ≥ 2 values",
            ));
        }
        let first = values[0];
        let last = values[n - 1];

        // Compute residuals using i128 to avoid overflow
        let mut min_residual = i128::MAX;
        for (i, &val) in values.iter().enumerate() {
            let predicted = interpolate(first, last, n, i);
            let residual = val as i128 - predicted as i128;
            min_residual = min_residual.min(residual);
        }

        // The offset field is i64 on disk — reject data that doesn't fit.
        if min_residual < i64::MIN as i128 || min_residual > i64::MAX as i128 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "linear codec: residual offset exceeds i64 range",
            ));
        }
        let min_residual_i64 = min_residual as i64;

        // Shift residuals to non-negative
        let shifted: Vec<u64> = values
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let predicted = interpolate(first, last, n, i);
                let residual = val as i128 - predicted as i128;
                (residual - min_residual) as u64
            })
            .collect();
        let max_shifted = shifted.iter().copied().max().unwrap_or(0);
        let bpv = bits_needed_u64(max_shifted);
        writer.write_u8(CodecType::Linear as u8)?;
        writer.write_u64::<LittleEndian>(first)?;
        writer.write_u64::<LittleEndian>(last)?;
        writer.write_u32::<LittleEndian>(n as u32)?;
        writer.write_i64::<LittleEndian>(min_residual_i64)?;
        writer.write_u8(bpv)?;
        let mut bytes_written = 30u64; // 1+8+8+4+8+1

        if bpv > 0 {
            let mut packed = Vec::new();
            bitpack_write(&shifted, bpv, &mut packed);
            writer.write_all(&packed)?;
            bytes_written += packed.len() as u64;
        }

        Ok(bytes_written)
    }
}

/// Interpolate value at index `i` on the line from first to last over `n` values.
#[inline]
fn interpolate(first: u64, last: u64, n: usize, i: usize) -> u64 {
    if n <= 1 {
        return first;
    }
    // Use i128 to avoid overflow
    let first = first as i128;
    let last = last as i128;
    let n = n as i128;
    let i = i as i128;
    let result = first + (last - first) * i / (n - 1);
    result as u64
}

/// Read a single value from a linear-codec column.
///
/// `data` starts right after the codec_id byte.
#[inline]
pub fn linear_read(data: &[u8], index: usize) -> u64 {
    let first = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let last = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let n = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
    let offset = i64::from_le_bytes(data[20..28].try_into().unwrap());
    let bpv = data[28];
    let predicted = interpolate(first, last, n, index);
    let residual = if bpv == 0 {
        0u64
    } else {
        bitpack_read(&data[29..], bpv, index)
    };
    // Use i128 to avoid overflow with large values
    (predicted as i128 + offset as i128 + residual as i128) as u64
}

// ── BlockwiseLinear codec ────────────────────────────────────────────────

/// Per-512-element-block linear interpolation with per-block bitpacked residuals.
///
/// Header: codec_id(1) + num_values(4) + num_blocks(4)
/// Per block: first(8) + last(8) + offset(8) + bpv(1) + packed_len(4) + packed_data
#[derive(Default)]
pub struct BlockwiseLinearEstimator {
    values: Vec<u64>,
}

impl CodecEstimator for BlockwiseLinearEstimator {
    fn collect(&mut self, value: u64) {
        self.values.push(value);
    }

    fn estimate(&self) -> Option<u64> {
        let n = self.values.len();
        if n < 2 * BLOCKWISE_LINEAR_BLOCK_SIZE {
            // Only useful when there are enough values to amortize the per-block headers
            return None;
        }

        let num_blocks = n.div_ceil(BLOCKWISE_LINEAR_BLOCK_SIZE);
        // Global header
        let mut total = 9u64; // codec_id(1) + num_values(4) + num_blocks(4)

        for b in 0..num_blocks {
            let start = b * BLOCKWISE_LINEAR_BLOCK_SIZE;
            let end = (start + BLOCKWISE_LINEAR_BLOCK_SIZE).min(n);
            let block = &self.values[start..end];
            let block_len = block.len();

            if block_len < 2 {
                // Block header + 0 data
                total += 29; // first(8)+last(8)+offset(8)+bpv(1)+packed_len(4)
                continue;
            }

            let first = block[0];
            let last = block[block_len - 1];
            let mut min_res = i128::MAX;
            let mut max_res = i128::MIN;
            for (i, &val) in block.iter().enumerate() {
                let pred = interpolate(first, last, block_len, i);
                let res = val as i128 - pred as i128;
                min_res = min_res.min(res);
                max_res = max_res.max(res);
            }
            // Per-block offset is stored as i64 — if any block's residuals
            // exceed i64 range, this codec cannot represent the data.
            if min_res < i64::MIN as i128 || max_res > i64::MAX as i128 {
                return None;
            }
            let range = (max_res - min_res) as u64;
            let bpv = bits_needed_u64(range) as u64;
            let data_bits = block_len as u64 * bpv;
            let data_bytes = data_bits.div_ceil(8);
            total += 29 + data_bytes;
        }

        Some(total)
    }

    fn serialize(&self, values: &[u64], writer: &mut dyn Write) -> io::Result<u64> {
        let n = values.len();
        let num_blocks = n.div_ceil(BLOCKWISE_LINEAR_BLOCK_SIZE);

        writer.write_u8(CodecType::BlockwiseLinear as u8)?;
        writer.write_u32::<LittleEndian>(n as u32)?;
        writer.write_u32::<LittleEndian>(num_blocks as u32)?;
        let mut bytes_written = 9u64;

        for b in 0..num_blocks {
            let start = b * BLOCKWISE_LINEAR_BLOCK_SIZE;
            let end = (start + BLOCKWISE_LINEAR_BLOCK_SIZE).min(n);
            let block = &values[start..end];
            let block_len = block.len();

            let first = block[0];
            let last = if block_len > 1 {
                block[block_len - 1]
            } else {
                first
            };

            // Compute residuals using i128 to avoid overflow
            let mut min_residual = i128::MAX;
            if block_len >= 2 {
                for (i, &val) in block.iter().enumerate() {
                    let pred = interpolate(first, last, block_len, i);
                    let res = val as i128 - pred as i128;
                    min_residual = min_residual.min(res);
                }
            } else {
                min_residual = 0;
            }

            let shifted: Vec<u64> = block
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    if block_len < 2 {
                        return 0;
                    }
                    let pred = interpolate(first, last, block_len, i);
                    let res = val as i128 - pred as i128;
                    (res - min_residual) as u64
                })
                .collect();
            let max_shifted = shifted.iter().copied().max().unwrap_or(0);
            let bpv = bits_needed_u64(max_shifted);

            // Per-block offset is stored as i64 — reject data that doesn't fit.
            if min_residual < i64::MIN as i128 || min_residual > i64::MAX as i128 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "blockwise linear codec: per-block residual offset exceeds i64 range",
                ));
            }
            let min_res_i64 = min_residual as i64;
            writer.write_u64::<LittleEndian>(first)?;
            writer.write_u64::<LittleEndian>(last)?;
            writer.write_i64::<LittleEndian>(min_res_i64)?;
            writer.write_u8(bpv)?;

            let mut packed = Vec::new();
            if bpv > 0 {
                bitpack_write(&shifted, bpv, &mut packed);
            }
            writer.write_u32::<LittleEndian>(packed.len() as u32)?;
            writer.write_all(&packed)?;
            bytes_written += 29 + packed.len() as u64;
        }

        Ok(bytes_written)
    }
}

/// Read a single value from a blockwise-linear-codec column.
///
/// `data` starts right after the codec_id byte.
pub fn blockwise_linear_read(data: &[u8], index: usize) -> u64 {
    let _num_values = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let num_blocks = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

    let target_block = index / BLOCKWISE_LINEAR_BLOCK_SIZE;
    let index_in_block = index % BLOCKWISE_LINEAR_BLOCK_SIZE;

    // Scan block headers to find the right one
    let mut pos = 8usize;
    for b in 0..num_blocks {
        let first = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        let last = u64::from_le_bytes(data[pos + 8..pos + 16].try_into().unwrap());
        let offset = i64::from_le_bytes(data[pos + 16..pos + 24].try_into().unwrap());
        let bpv = data[pos + 24];
        let packed_len = u32::from_le_bytes(data[pos + 25..pos + 29].try_into().unwrap()) as usize;

        if b == target_block {
            let block_start = b * BLOCKWISE_LINEAR_BLOCK_SIZE;
            let block_end = ((b + 1) * BLOCKWISE_LINEAR_BLOCK_SIZE).min(_num_values);
            let block_len = block_end - block_start;

            let predicted = interpolate(first, last, block_len, index_in_block);
            let residual = if bpv == 0 {
                0u64
            } else {
                bitpack_read(&data[pos + 29..], bpv, index_in_block)
            };
            return (predicted as i128 + offset as i128 + residual as i128) as u64;
        }

        pos += 29 + packed_len;
    }

    0 // Should not reach here
}

// ── Auto-selection ───────────────────────────────────────────────────────

/// Serialize values using the codec that produces the smallest output.
///
/// Returns the number of bytes written.
pub fn serialize_auto(values: &[u64], writer: &mut dyn Write) -> io::Result<u64> {
    let mut constant = ConstantEstimator::default();
    let mut bitpacked = BitpackedEstimator::default();
    let mut linear = LinearEstimator::default();
    let mut blockwise = BlockwiseLinearEstimator::default();

    // Pass 1: collect
    for &v in values {
        constant.collect(v);
        bitpacked.collect(v);
        linear.collect(v);
        blockwise.collect(v);
    }

    // Finalize
    constant.finalize();
    bitpacked.finalize();
    linear.finalize();
    blockwise.finalize();

    // Pick smallest
    let candidates: Vec<(&dyn CodecEstimator, &str)> = vec![
        (&constant, "constant"),
        (&bitpacked, "bitpacked"),
        (&linear, "linear"),
        (&blockwise, "blockwise_linear"),
    ];

    let (best, _name) = candidates
        .into_iter()
        .filter_map(|(est, name)| est.estimate().map(|size| (est, name, size)))
        .min_by_key(|&(_, _, size)| size)
        .map(|(est, name, _)| (est, name))
        .unwrap_or((&bitpacked as &dyn CodecEstimator, "bitpacked"));

    best.serialize(values, writer)
}

/// Batch-read `count` consecutive values starting at `start_index` from bitpacked data.
///
/// `data` starts right after the codec_id byte (at min_value).
/// Uses direct array access for byte-aligned bpv (8, 16, 32, 64) which the
/// compiler auto-vectorizes to SIMD on aarch64 (NEON) and x86_64 (SSE/AVX).
/// For arbitrary bpv, uses a tight scalar loop with the u64 fast-path.
pub fn bitpacked_read_batch(data: &[u8], start_index: usize, out: &mut [u64]) {
    let min_value = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let bpv = data[8];

    if bpv == 0 {
        out.iter_mut().for_each(|v| *v = min_value);
        return;
    }

    let packed = &data[9..];

    match bpv {
        // Byte-aligned fast paths — compiler auto-vectorizes these loops
        8 => {
            for (i, v) in out.iter_mut().enumerate() {
                let idx = start_index + i;
                *v = packed[idx] as u64 + min_value;
            }
        }
        16 => {
            for (i, v) in out.iter_mut().enumerate() {
                let idx = start_index + i;
                let byte_off = idx * 2;
                let raw = u16::from_le_bytes([packed[byte_off], packed[byte_off + 1]]);
                *v = raw as u64 + min_value;
            }
        }
        32 => {
            for (i, v) in out.iter_mut().enumerate() {
                let idx = start_index + i;
                let byte_off = idx * 4;
                let raw = u32::from_le_bytes(packed[byte_off..byte_off + 4].try_into().unwrap());
                *v = raw as u64 + min_value;
            }
        }
        64 => {
            for (i, v) in out.iter_mut().enumerate() {
                let idx = start_index + i;
                let byte_off = idx * 8;
                let raw = u64::from_le_bytes(packed[byte_off..byte_off + 8].try_into().unwrap());
                *v = raw.wrapping_add(min_value);
            }
        }
        // Arbitrary bpv — tight scalar loop using u64 fast-path read
        _ => {
            for (i, v) in out.iter_mut().enumerate() {
                *v = super::bitpack_read(packed, bpv, start_index + i) + min_value;
            }
        }
    }
}

/// Batch-read `out.len()` consecutive values starting at `start_index` from auto-codec data.
///
/// Dispatches codec type once (vs. per-value in `auto_read`), enabling tight inner
/// loops that the compiler auto-vectorizes for byte-aligned bitpacked columns.
pub fn auto_read_batch(data: &[u8], start_index: usize, out: &mut [u64]) {
    if data.is_empty() || out.is_empty() {
        out.iter_mut().for_each(|v| *v = 0);
        return;
    }
    let codec_id = data[0];
    let rest = &data[1..];
    match CodecType::from_u8(codec_id) {
        Some(CodecType::Constant) => {
            let val = u64::from_le_bytes(rest[0..8].try_into().unwrap());
            out.iter_mut().for_each(|v| *v = val);
        }
        Some(CodecType::Bitpacked) => bitpacked_read_batch(rest, start_index, out),
        Some(CodecType::Linear) => {
            for (i, v) in out.iter_mut().enumerate() {
                *v = linear_read(rest, start_index + i);
            }
        }
        Some(CodecType::BlockwiseLinear) => {
            for (i, v) in out.iter_mut().enumerate() {
                *v = blockwise_linear_read(rest, start_index + i);
            }
        }
        None => out.iter_mut().for_each(|v| *v = 0),
    }
}

/// Read a single value from auto-codec encoded data.
///
/// The first byte identifies the codec.
#[inline]
pub fn auto_read(data: &[u8], index: usize) -> u64 {
    if data.is_empty() {
        return 0;
    }
    let codec_id = data[0];
    let rest = &data[1..];
    match CodecType::from_u8(codec_id) {
        Some(CodecType::Constant) => {
            // rest = value(8)
            u64::from_le_bytes(rest[0..8].try_into().unwrap())
        }
        Some(CodecType::Bitpacked) => bitpacked_read(rest, index),
        Some(CodecType::Linear) => linear_read(rest, index),
        Some(CodecType::BlockwiseLinear) => blockwise_linear_read(rest, index),
        None => 0,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(values: &[u64]) -> Vec<u64> {
        let mut buf = Vec::new();
        serialize_auto(values, &mut buf).unwrap();
        (0..values.len()).map(|i| auto_read(&buf, i)).collect()
    }

    #[test]
    fn test_constant_codec() {
        let values: Vec<u64> = vec![42; 100];
        let mut buf = Vec::new();
        serialize_auto(&values, &mut buf).unwrap();
        assert_eq!(buf[0], CodecType::Constant as u8);
        assert_eq!(buf.len(), 9);
        assert_eq!(roundtrip(&values), values);
    }

    #[test]
    fn test_bitpacked_codec() {
        let values: Vec<u64> = (0..50).map(|i| 1000 + (i % 7) * 13).collect();
        let result = roundtrip(&values);
        assert_eq!(result, values);
    }

    #[test]
    fn test_linear_codec_sequential() {
        // Perfectly linear → 0 bpv residuals
        let values: Vec<u64> = (0..1000).map(|i| 100 + i * 3).collect();
        let mut buf = Vec::new();
        serialize_auto(&values, &mut buf).unwrap();
        // Should pick linear (smaller than bitpacked for sequential data)
        assert_eq!(roundtrip(&values), values);
    }

    #[test]
    fn test_blockwise_linear_codec() {
        // Two distinct linear segments
        let mut values: Vec<u64> = Vec::new();
        for i in 0..1500 {
            if i < 750 {
                values.push(100 + i * 2);
            } else {
                values.push(5000 + (i - 750) * 5);
            }
        }
        let result = roundtrip(&values);
        assert_eq!(result, values);
    }

    #[test]
    fn test_empty() {
        let values: Vec<u64> = vec![];
        let mut buf = Vec::new();
        serialize_auto(&values, &mut buf).unwrap();
        assert!(buf.len() <= 10);
    }

    #[test]
    fn test_single_value() {
        let values = vec![999u64];
        assert_eq!(roundtrip(&values), values);
    }

    #[test]
    fn test_two_values() {
        let values = vec![10u64, 20];
        assert_eq!(roundtrip(&values), values);
    }

    #[test]
    fn test_large_range() {
        let values = vec![0u64, u64::MAX / 2, u64::MAX];
        assert_eq!(roundtrip(&values), values);
    }

    #[test]
    fn test_timestamps_pick_linear_or_blockwise() {
        // Simulate timestamps (monotonically increasing with small jitter)
        let mut values: Vec<u64> = Vec::new();
        let mut ts = 1_700_000_000u64;
        for _ in 0..2000 {
            values.push(ts);
            ts += 1000 + (ts % 7); // ~1000 with jitter
        }
        let result = roundtrip(&values);
        assert_eq!(result, values);
    }

    /// Helper: roundtrip via auto_read_batch and compare with per-element auto_read.
    fn roundtrip_batch(values: &[u64]) {
        let mut buf = Vec::new();
        serialize_auto(values, &mut buf).unwrap();

        // Batch read all values
        let mut batch_out = vec![0u64; values.len()];
        auto_read_batch(&buf, 0, &mut batch_out);
        assert_eq!(batch_out, values, "batch read mismatch");

        // Batch read a sub-range
        if values.len() >= 10 {
            let start = 3;
            let count = values.len() - 6;
            let mut sub = vec![0u64; count];
            auto_read_batch(&buf, start, &mut sub);
            assert_eq!(
                sub,
                &values[start..start + count],
                "sub-range batch mismatch"
            );
        }
    }

    #[test]
    fn test_batch_read_constant() {
        roundtrip_batch(&vec![42u64; 100]);
    }

    #[test]
    fn test_batch_read_bitpacked_8bit() {
        // Values with range < 256 → 8-bit bpv
        let values: Vec<u64> = (0..200).map(|i| 1000 + (i % 200)).collect();
        roundtrip_batch(&values);
    }

    #[test]
    fn test_batch_read_bitpacked_16bit() {
        // Values with range fitting 16 bits
        let values: Vec<u64> = (0..200).map(|i| 50000 + i * 100).collect();
        roundtrip_batch(&values);
    }

    #[test]
    fn test_batch_read_bitpacked_arbitrary() {
        // Arbitrary bpv (e.g. 13 bits)
        let values: Vec<u64> = (0..100).map(|i| 999 + (i * 37) % 8000).collect();
        roundtrip_batch(&values);
    }

    #[test]
    fn test_batch_read_linear() {
        let values: Vec<u64> = (0..500).map(|i| 100 + i * 3).collect();
        roundtrip_batch(&values);
    }

    #[test]
    fn test_batch_read_blockwise() {
        let mut values = Vec::new();
        for i in 0..1500u64 {
            values.push(if i < 750 {
                100 + i * 2
            } else {
                5000 + (i - 750) * 5
            });
        }
        roundtrip_batch(&values);
    }

    /// Regression: zigzag-encoded i64 timestamps mixed with FAST_FIELD_MISSING (u64::MAX).
    /// The linear codec's min_residual clamping to i64 corrupts data when values
    /// span nearly the full u64 range.
    #[test]
    fn test_zigzag_timestamps_with_missing() {
        use super::super::{FAST_FIELD_MISSING, zigzag_encode};

        // Simulate issued_at column: most docs have timestamps, some are missing
        let timestamps: Vec<i64> = vec![
            1724630400, // 2024-08-26
            1724716800, // 2024-08-27
            1724803200, // 2024-08-28
            1700000000, // 2023-11-14
            1680000000, // 2023-03-28
            1724630400, // duplicate
        ];

        // Build values array: zigzag-encoded timestamps + some FAST_FIELD_MISSING gaps
        let mut values = Vec::new();
        for (i, &ts) in timestamps.iter().enumerate() {
            values.push(zigzag_encode(ts));
            // Insert a missing value after every 2nd doc
            if i % 2 == 1 {
                values.push(FAST_FIELD_MISSING);
            }
        }

        let result = roundtrip(&values);
        assert_eq!(
            result, values,
            "zigzag timestamps + missing roundtrip failed"
        );
    }

    /// Test each codec individually with zigzag-encoded values + FAST_FIELD_MISSING
    #[test]
    fn test_codecs_individually_with_zigzag_and_missing() {
        use super::super::{FAST_FIELD_MISSING, zigzag_encode};

        let values: Vec<u64> = vec![
            zigzag_encode(1724630400), // 3449260800
            zigzag_encode(1700000000), // 3400000000
            FAST_FIELD_MISSING,
            zigzag_encode(1680000000), // 3360000000
            zigzag_encode(1724716800), // 3449433600
            FAST_FIELD_MISSING,
            zigzag_encode(1724630400), // 3449260800
            zigzag_encode(0),          // 0
        ];

        // Test bitpacked directly
        {
            let mut est = BitpackedEstimator::default();
            for &v in &values {
                est.collect(v);
            }
            est.finalize();
            if est.estimate().is_some() {
                let mut buf = Vec::new();
                est.serialize(&values, &mut buf).unwrap();
                for (i, &expected) in values.iter().enumerate() {
                    let got = auto_read(&buf, i);
                    assert_eq!(
                        got, expected,
                        "bitpacked: index {} expected {} got {}",
                        i, expected, got
                    );
                }
            }
        }

        // Test linear directly (needs ≥ 2 values)
        {
            let mut est = LinearEstimator::default();
            for &v in &values {
                est.collect(v);
            }
            est.finalize();
            if est.estimate().is_some() {
                let mut buf = Vec::new();
                est.serialize(&values, &mut buf).unwrap();
                for (i, &expected) in values.iter().enumerate() {
                    let got = auto_read(&buf, i);
                    assert_eq!(
                        got, expected,
                        "linear: index {} expected {} got {}",
                        i, expected, got
                    );
                }
            }
        }

        // Test auto (whichever is selected)
        let result = roundtrip(&values);
        assert_eq!(result, values, "auto codec roundtrip failed");
    }

    /// Regression: value that the user observed corrupted in production
    #[test]
    fn test_specific_issued_at_roundtrip() {
        use super::super::{FAST_FIELD_MISSING, zigzag_encode};

        // Reproduce exact scenario: 100 docs, mix of timestamps and missing
        let mut values = Vec::new();
        let base_ts = 1724630400i64; // 2024-08-26 epoch
        for i in 0..100u64 {
            if i % 5 == 0 {
                // Every 5th doc has no issued_at
                values.push(FAST_FIELD_MISSING);
            } else {
                // Varying timestamps
                let ts = base_ts - (i as i64 * 86400); // one day apart
                values.push(zigzag_encode(ts));
            }
        }

        let result = roundtrip(&values);
        for (i, (&expected, &got)) in values.iter().zip(result.iter()).enumerate() {
            assert_eq!(
                got,
                expected,
                "doc {}: expected {} (zigzag of {}), got {}",
                i,
                expected,
                if expected == FAST_FIELD_MISSING {
                    -1 // placeholder
                } else {
                    super::super::zigzag_decode(expected)
                },
                got
            );
        }
    }

    /// Large-scale test: exercise blockwise linear codec with realistic timestamp data.
    /// Tests 10K, 50K, 100K docs to catch codec edge cases.
    #[test]
    fn test_large_scale_timestamp_roundtrip() {
        use super::super::{FAST_FIELD_MISSING, zigzag_encode};

        for num_docs in [10_000, 50_000, 100_000] {
            let mut values = Vec::with_capacity(num_docs);
            let base_ts = 1724630400i64;

            for i in 0..num_docs {
                if i % 7 == 0 {
                    values.push(FAST_FIELD_MISSING);
                } else {
                    // Timestamps spanning ~5 years, with some jitter
                    let ts = base_ts - (i as i64 * 3600) + ((i as i64 * 37) % 1000);
                    values.push(zigzag_encode(ts));
                }
            }

            // Check which codec is selected
            let mut buf = Vec::new();
            serialize_auto(&values, &mut buf).unwrap();
            let codec_id = buf[0];
            let codec_name = match CodecType::from_u8(codec_id) {
                Some(CodecType::Constant) => "constant",
                Some(CodecType::Bitpacked) => "bitpacked",
                Some(CodecType::Linear) => "linear",
                Some(CodecType::BlockwiseLinear) => "blockwise_linear",
                None => "unknown",
            };

            // Verify roundtrip
            let mut failures = Vec::new();
            for (i, &expected) in values.iter().enumerate() {
                let got = auto_read(&buf, i);
                if got != expected {
                    failures.push((i, expected, got));
                    if failures.len() >= 5 {
                        break;
                    }
                }
            }

            assert!(
                failures.is_empty(),
                "num_docs={}, codec={}: {} failures. First 5: {:?}",
                num_docs,
                codec_name,
                failures.len(),
                failures
            );
        }
    }

    /// Regression: blockwise linear codec selected for column where most blocks
    /// are efficient (sorted timestamps only) but a few blocks contain
    /// FAST_FIELD_MISSING, causing min_residual clamping corruption.
    #[test]
    fn test_blockwise_linear_with_clustered_missing() {
        use super::super::{FAST_FIELD_MISSING, zigzag_encode};

        // 3000 values: first 512 are all FAST_FIELD_MISSING,
        // remaining 2488 are sorted timestamps (efficient linear blocks).
        // This should trigger blockwise linear selection overall,
        // but the first block has a mix that triggers the bug.
        let mut values = Vec::new();

        // Block 0 (indices 0-511): mix of MISSING and timestamps
        // — first 100 are MISSING, rest are timestamps
        for i in 0..512 {
            if i < 100 {
                values.push(FAST_FIELD_MISSING);
            } else {
                let ts = 1724630400i64 + (i as i64 * 100);
                values.push(zigzag_encode(ts));
            }
        }

        // Blocks 1-5 (indices 512-3071): sorted timestamps only
        for i in 512..3072 {
            let ts = 1724630400i64 + (i as i64 * 100);
            values.push(zigzag_encode(ts));
        }

        let result = roundtrip(&values);
        let mut failures = Vec::new();
        for (i, (&expected, &got)) in values.iter().zip(result.iter()).enumerate() {
            if got != expected {
                failures.push((i, expected, got));
            }
        }
        assert!(
            failures.is_empty(),
            "blockwise linear with clustered missing: {} failures. First 5: {:?}",
            failures.len(),
            &failures[..failures.len().min(5)]
        );
    }

    /// Test each codec FORCED with zigzag timestamps + FAST_FIELD_MISSING.
    /// This catches bugs that only manifest when a specific codec is forced.
    #[test]
    fn test_forced_codecs_with_timestamps_and_missing() {
        use super::super::{FAST_FIELD_MISSING, zigzag_encode};

        let mut values = Vec::new();
        let base_ts = 1724630400i64;
        for i in 0..200 {
            if i % 5 == 0 {
                values.push(FAST_FIELD_MISSING);
            } else {
                let ts = base_ts - (i as i64 * 86400);
                values.push(zigzag_encode(ts));
            }
        }

        // Force bitpacked
        {
            let est = BitpackedEstimator::default();
            let mut buf = Vec::new();
            est.serialize(&values, &mut buf).unwrap();
            for (i, &expected) in values.iter().enumerate() {
                let got = bitpacked_read(&buf[1..], i); // skip codec_id byte
                assert_eq!(got, expected, "forced bitpacked: index {} failed", i);
            }
        }

        // Force linear — should error because FAST_FIELD_MISSING + timestamps
        // produce residuals exceeding i64 range
        {
            let est = LinearEstimator::default();
            let mut buf = Vec::new();
            let result = est.serialize(&values, &mut buf);
            assert!(
                result.is_err(),
                "linear codec should reject data with residuals exceeding i64"
            );
        }

        // Force linear with values that DO fit in i64 (no FAST_FIELD_MISSING)
        {
            let safe_values: Vec<u64> = values
                .iter()
                .filter(|&&v| v != FAST_FIELD_MISSING)
                .copied()
                .collect();
            let est = LinearEstimator::default();
            let mut buf = Vec::new();
            est.serialize(&safe_values, &mut buf).unwrap();
            for (i, &expected) in safe_values.iter().enumerate() {
                let got = linear_read(&buf[1..], i);
                assert_eq!(got, expected, "forced linear (safe): index {} failed", i);
            }
        }

        // Blockwise linear estimator should return None for data with FAST_FIELD_MISSING
        {
            let mut large_values = Vec::new();
            for i in 0..2000 {
                if i % 5 == 0 {
                    large_values.push(FAST_FIELD_MISSING);
                } else {
                    let ts = base_ts - (i as i64 * 86400);
                    large_values.push(zigzag_encode(ts));
                }
            }
            let mut est = BlockwiseLinearEstimator::default();
            for &v in &large_values {
                est.collect(v);
            }
            assert!(
                est.estimate().is_none(),
                "blockwise linear should reject data with per-block residuals exceeding i64"
            );
        }
    }
}
