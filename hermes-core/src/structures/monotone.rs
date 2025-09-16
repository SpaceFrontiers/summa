//! Borrowed, independently addressable blocks of monotone U32 integers.
use super::fast_field::{bitpack_read, bitpack_write, bits_needed_u64};
use crate::{Error, Result};

pub(crate) const BLOCK: usize = 128;
const ENTRY: usize = 16;

fn invalid() -> Error {
    Error::Corruption("invalid monotone integer block".into())
}
fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn packed(count: usize, bits: u8) -> usize {
    (count * usize::from(bits)).div_ceil(8)
}

#[derive(Clone, Copy)]
pub(crate) struct Monotone<'a> {
    bytes: &'a [u8],
    count: usize,
}
impl<'a> Monotone<'a> {
    /// Borrow immutable writer-produced blocks with their stored cardinality.
    pub(crate) fn new(bytes: &'a [u8], count: usize) -> Self {
        Self { bytes, count }
    }
    fn block(self, block: usize) -> Block<'a> {
        let entry = &self.bytes[block * ENTRY..][..ENTRY];
        let count = (self.count - block * BLOCK).min(BLOCK);
        let mut length = packed(count, entry[12]);
        if entry[13] == 1 {
            length += ((u64::from(word(entry, 4) - word(entry, 0)) >> entry[12]) as usize + count)
                .div_ceil(8);
        }
        Block {
            bytes: &self.bytes[word(entry, 8) as usize..][..length],
            base: word(entry, 0),
            count,
            bits: entry[12],
            codec: entry[13],
        }
    }
    pub(crate) fn get(self, index: usize) -> u32 {
        assert!(index < self.count);
        self.block(index / BLOCK).get(index % BLOCK)
    }
    /// Decode a bounded block without repeating select for each integer.
    pub(crate) fn decode_block(self, block: usize, out: &mut [u32]) {
        let block = self.block(block);
        assert_eq!(out.len(), block.count);
        block.for_each_offset(|i, offset| out[i] = block.base + offset as u32);
    }
    pub(crate) fn lower_bound(self, value: u32) -> usize {
        let blocks = self.count.div_ceil(BLOCK);
        let mut low = 0;
        let mut high = blocks;
        while low < high {
            let mid = low + (high - low) / 2;
            if word(self.bytes, mid * ENTRY + 4) < value {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if low == blocks {
            return self.count;
        }
        let base = low * BLOCK;
        let block = self.block(low);
        let mut low = 0;
        let mut high = block.count;
        while low < high {
            let mid = low + (high - low) / 2;
            if block.get(mid) < value {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        base + low
    }
}

struct Block<'a> {
    bytes: &'a [u8],
    base: u32,
    count: usize,
    bits: u8,
    codec: u8,
}
impl Block<'_> {
    fn for_each_offset(&self, mut visit: impl FnMut(usize, u64)) {
        let low_bytes = packed(self.count, self.bits);
        if self.codec == 0 {
            for i in 0..self.count {
                visit(i, bitpack_read(&self.bytes[..low_bytes], self.bits, i));
            }
            return;
        }
        let mut i = 0;
        for (word, bytes) in self.bytes[low_bytes..].chunks(8).enumerate() {
            let mut storage = [0u8; 8];
            storage[..bytes.len()].copy_from_slice(bytes);
            let mut upper = u64::from_le_bytes(storage);
            while upper != 0 {
                let high = word * 64 + upper.trailing_zeros() as usize - i;
                let low = bitpack_read(&self.bytes[..low_bytes], self.bits, i);
                visit(i, ((high as u64) << self.bits) | low);
                upper &= upper - 1;
                i += 1;
            }
        }
    }
    fn get(&self, index: usize) -> u32 {
        self.base + self.offset(index) as u32
    }
    fn offset(&self, index: usize) -> u64 {
        let low_bytes = packed(self.count, self.bits);
        let low = bitpack_read(&self.bytes[..low_bytes], self.bits, index) as u32;
        if self.codec == 0 {
            return u64::from(low);
        }
        // The high bitmap is at most 3 * BLOCK bits. Never scan another block.
        let mut remaining = index as u32;
        for (i, bytes) in self.bytes[low_bytes..].chunks(8).enumerate() {
            let mut storage = [0u8; 8];
            storage[..bytes.len()].copy_from_slice(bytes);
            let mut bits = u64::from_le_bytes(storage);
            let count = bits.count_ones();
            if remaining < count {
                for _ in 0..remaining {
                    bits &= bits - 1;
                }
                let high = i * 64 + bits.trailing_zeros() as usize - index;
                return ((high as u64) << self.bits) | u64::from(low);
            }
            remaining -= count;
        }
        unreachable!("writer-produced monotone block contains every encoded value")
    }
}

pub(crate) fn encode(values: &[u64]) -> Result<Vec<u8>> {
    if values.iter().any(|&v| v > u64::from(u32::MAX)) || values.windows(2).any(|w| w[0] > w[1]) {
        return Err(invalid());
    }
    let mut output = vec![0; values.len().div_ceil(BLOCK) * ENTRY];
    for (i, values) in values.chunks(BLOCK).enumerate() {
        let base = values[0];
        let maximum = *values.last().unwrap();
        let span = maximum - base;
        let width = bits_needed_u64(span);
        let ratio = (span + 1) / values.len() as u64;
        let lower = bits_needed_u64(ratio).saturating_sub(1);
        let upper_bits = (span >> lower) as usize + values.len();
        let ef_bytes = packed(values.len(), lower) + upper_bits.div_ceil(8);
        let codec = u8::from(ef_bytes < packed(values.len(), width));
        let bits = if codec == 1 { lower } else { width };
        let offset = u32::try_from(output.len()).map_err(|_| invalid())?;
        let entry = &mut output[i * ENTRY..][..ENTRY];
        entry[..4].copy_from_slice(&(base as u32).to_le_bytes());
        entry[4..8].copy_from_slice(&(maximum as u32).to_le_bytes());
        entry[8..12].copy_from_slice(&offset.to_le_bytes());
        entry[12] = bits;
        entry[13] = codec;
        let mask = (1u64 << bits) - 1;
        let mut lows = [0u64; BLOCK];
        for (low, &value) in lows.iter_mut().zip(values) {
            *low = (value - base) & mask;
        }
        bitpack_write(&lows[..values.len()], bits, &mut output);
        if codec == 1 {
            let start = output.len();
            output.resize(start + upper_bits.div_ceil(8), 0);
            for (j, &value) in values.iter().enumerate() {
                let bit = ((value - base) >> bits) as usize + j;
                output[start + bit / 8] |= 1 << (bit % 8);
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
pub(crate) fn admit(bytes: &[u8], count: usize) -> Result<()> {
    let directory = count
        .div_ceil(BLOCK)
        .checked_mul(ENTRY)
        .ok_or_else(invalid)?;
    if bytes.len() < directory {
        return Err(invalid());
    }
    let mut next = directory;
    let mut previous = None;
    for (i, entry) in bytes[..directory].chunks_exact(ENTRY).enumerate() {
        let base = word(entry, 0);
        let maximum = word(entry, 4);
        let n = (count - i * BLOCK).min(BLOCK);
        let bits = entry[12];
        let codec = entry[13];
        if base > maximum
            || previous.is_some_and(|p| p > base)
            || word(entry, 8) as usize != next
            || bits > 32
            || codec > 1
            || entry[14..] != [0, 0]
        {
            return Err(invalid());
        }
        let mut size = packed(n, bits);
        if codec == 1 {
            let high_bits = ((u64::from(maximum - base) >> bits) + n as u64) as usize;
            if high_bits > 3 * n {
                return Err(invalid());
            }
            size += high_bits.div_ceil(8);
        }
        let end = next.checked_add(size).ok_or_else(invalid)?;
        let payload = bytes.get(next..end).ok_or_else(invalid)?;
        if codec == 1 {
            let high = &payload[packed(n, bits)..];
            if high.iter().map(|b| b.count_ones() as usize).sum::<usize>() != n {
                return Err(invalid());
            }
        }
        let block = Block {
            bytes: payload,
            base: 0,
            count: n,
            bits,
            codec,
        };
        let mut last = 0;
        let mut valid = true;
        block.for_each_offset(|j, value| {
            if value < last || value > u64::from(maximum - base) || (j == 0 && value != 0) {
                valid = false;
            }
            last = value;
        });
        if !valid || last != u64::from(maximum - base) {
            return Err(invalid());
        }
        previous = Some(maximum);
        next = end;
    }
    if next != bytes.len() {
        return Err(invalid());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotone_blocks_preserve_duplicates_tails_and_u32_edges() {
        for values in [
            vec![],
            vec![u32::MAX as u64],
            vec![7; 300],
            (0..513).map(|i| i * 7 + 1000).collect(),
            vec![0, 1, 1, u32::MAX as u64],
            (0..300).map(|i| i * 13_000_000).collect(),
        ] {
            let bytes = encode(&values).unwrap();
            admit(&bytes, values.len()).unwrap();
            let view = Monotone::new(&bytes, values.len());
            for (block, chunk) in values.chunks(BLOCK).enumerate() {
                let mut decoded = [0u32; BLOCK];
                view.decode_block(block, &mut decoded[..chunk.len()]);
                assert!(decoded.iter().zip(chunk).all(|(&a, &b)| u64::from(a) == b));
            }
            for (i, &value) in values.iter().enumerate() {
                assert_eq!(view.get(i), value as u32);
                for probe in [
                    value as u32,
                    (value as u32).saturating_sub(1),
                    (value as u32).saturating_add(1),
                ] {
                    assert_eq!(
                        view.lower_bound(probe),
                        values.partition_point(|&v| v < u64::from(probe))
                    );
                }
            }
        }
    }

    #[test]
    fn corrupt_monotone_blocks_fail_without_panicking() {
        let values: Vec<_> = (0..259).map(|i| i * 9).collect();
        let bytes = encode(&values).unwrap();
        for end in 0..bytes.len() {
            assert!(admit(&bytes[..end], values.len()).is_err());
        }
        for at in 0..bytes.len() {
            for mask in [1, 128, 255] {
                let mut corrupt = bytes.clone();
                corrupt[at] ^= mask;
                let _ = admit(&corrupt, values.len());
            }
        }
        assert!(encode(&[2, 1]).is_err());
        assert!(encode(&[u64::MAX]).is_err());
    }
}
