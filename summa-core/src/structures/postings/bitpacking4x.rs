//! Canonical little-endian BitPacker4x blocks and existing horizontal tails.
//!
//! Callers validate widths and extents before admitting persisted byte views.
//! All full blocks hold 128 values; tails never allocate padded full blocks.

use bitpacking::{BitPacker, BitPacker4x};

const BLOCK: usize = BitPacker4x::BLOCK_LEN;
const _: () = assert!(BLOCK == 128);

pub(super) fn encoded_len(count: usize, width: u8) -> usize {
    (count * width as usize).div_ceil(8)
}

pub(super) fn encode(values: &[u32], output: &mut Vec<u8>) -> u8 {
    assert!(values.len() <= BLOCK);
    let width = crate::structures::simd::bits_needed(values.iter().copied().max().unwrap_or(0));
    if values.len() == BLOCK {
        let start = output.len();
        output.resize(start + encoded_len(BLOCK, width), 0);
        BitPacker4x::new().compress(values, &mut output[start..], width);
        #[cfg(target_endian = "big")]
        reverse_words(&mut output[start..]);
    } else {
        super::horizontal_bp128::pack_block_n(values, width, output);
    }
    width
}

pub(super) fn decode(input: &[u8], width: u8, output: &mut [u32]) {
    if output.len() == BLOCK {
        with_native_words(input, |bytes| {
            BitPacker4x::new().decompress(bytes, output, width);
        });
    } else {
        super::horizontal_bp128::unpack_block_n(input, width, output, output.len());
    }
}

/// Full blocks include a zero first delta so the immutable first-doc header
/// alone controls document rebasing. Tail blocks omit that first value.
pub(super) fn encode_gaps(deltas: &[u32], output: &mut Vec<u8>) -> u8 {
    assert!(deltas.len() < BLOCK);
    let mut gaps = [0u32; BLOCK];
    for (out, &delta) in gaps[1..].iter_mut().zip(deltas) {
        *out = delta
            .checked_sub(1)
            .expect("posting documents must be strictly increasing");
    }
    if deltas.len() == BLOCK - 1 {
        encode(&gaps, output)
    } else {
        encode(&gaps[1..1 + deltas.len()], output)
    }
}

pub(super) fn decode_docs(input: &[u8], width: u8, first: u32, output: &mut [u32]) {
    if output.len() == BLOCK {
        with_native_words(input, |bytes| {
            BitPacker4x::new().decompress_strictly_sorted(
                first.checked_sub(1),
                bytes,
                output,
                width,
            );
        });
    } else {
        output[0] = first;
        decode(input, width, &mut output[1..]);
        for i in 1..output.len() {
            output[i] = output[i].wrapping_add(output[i - 1]).wrapping_add(1);
        }
    }
}

/// The first lane begins in the low bits of the first canonical u32 word.
/// Checking its reserved zero does not require decoding a posting block.
pub(super) fn first_gap_is_zero(input: &[u8], width: u8) -> bool {
    if width == 0 {
        return true;
    }
    width <= 32
        && input.get(..4).is_some_and(|bytes| {
            u32::from_le_bytes(bytes.try_into().unwrap()) & (u32::MAX >> (32 - width)) == 0
        })
}

#[cfg(target_endian = "big")]
fn reverse_words(bytes: &mut [u8]) {
    for word in bytes.chunks_exact_mut(4) {
        word.reverse();
    }
}

fn with_native_words<R>(input: &[u8], decode: impl FnOnce(&[u8]) -> R) -> R {
    #[cfg(target_endian = "big")]
    {
        let mut native = [0u8; BLOCK * 4];
        let native = &mut native[..input.len()];
        native.copy_from_slice(input);
        reverse_words(native);
        decode(native)
    }
    #[cfg(target_endian = "little")]
    decode(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Format oracle independent of the library: four interleaved streams of
    // little-endian words, each containing every fourth input value.
    fn reference_bytes(values: &[u32], width: u8) -> Vec<u8> {
        let mut bytes = vec![0u8; encoded_len(values.len(), width)];
        for (i, &value) in values.iter().enumerate() {
            for bit in 0..width as usize {
                if value & (1 << bit) == 0 {
                    continue;
                }
                let position = if values.len() == BLOCK {
                    let lane_bit = (i / 4) * width as usize + bit;
                    (lane_bit / 32) * 128 + (i % 4) * 32 + lane_bit % 32
                } else {
                    i * width as usize + bit
                };
                bytes[position / 8] |= 1 << (position % 8);
            }
        }
        bytes
    }

    #[test]
    fn every_width_and_tail_matches_canonical_little_endian_bytes() {
        for count in 0..=BLOCK {
            for width in 0..=32u8 {
                let mask = if width == 0 {
                    0
                } else {
                    u32::MAX >> (32 - width)
                };
                let mut values: Vec<_> = (0..count as u32)
                    .map(|i| i.wrapping_mul(0x9e37_79b9).rotate_left(i % 32) & mask)
                    .collect();
                if let Some(last) = values.last_mut() {
                    *last = mask;
                }
                let effective_width = if count == 0 { 0 } else { width };
                let mut bytes = vec![0xa5];
                assert_eq!(encode(&values, &mut bytes), effective_width);
                assert_eq!(bytes[0], 0xa5);
                assert_eq!(&bytes[1..], reference_bytes(&values, effective_width));
                let mut decoded = vec![u32::MAX; count];
                // Deliberately unaligned bytes exercise the persisted reader.
                decode(&bytes[1..], effective_width, &mut decoded);
                assert_eq!(decoded, values, "count {count}, width {width}");
            }
        }
    }

    #[test]
    fn document_bases_tails_large_gaps_and_rebasing_preserve_strict_order() {
        for count in 1..=BLOCK {
            for large_gap in [1u32, 2, 255, 65535, u32::MAX - 1024] {
                let mut docs = Vec::with_capacity(count);
                docs.push(0u32);
                for i in 1..count {
                    docs.push(docs[i - 1] + if i == 1 { large_gap } else { 1 });
                }
                let deltas: Vec<_> = docs.windows(2).map(|d| d[1] - d[0]).collect();
                let mut bytes = Vec::new();
                let width = encode_gaps(&deltas, &mut bytes);
                let mut reference: Vec<_> = deltas.iter().map(|d| d - 1).collect();
                if count == BLOCK {
                    reference.insert(0, 0);
                    assert!(first_gap_is_zero(&bytes, width));
                    if width != 0 {
                        let mut bad = bytes.clone();
                        bad[0] |= 1;
                        assert!(!first_gap_is_zero(&bad, width));
                    }
                }
                assert_eq!(bytes, reference_bytes(&reference, width));
                let original = bytes.clone();
                for base in [0, 1, u32::MAX - 1 - docs[count - 1]] {
                    let mut output = vec![u32::MAX; count];
                    decode_docs(&bytes, width, base, &mut output);
                    assert_eq!(output, docs.iter().map(|d| d + base).collect::<Vec<_>>());
                    assert_eq!(bytes, original, "rebasing never changes packed deltas");
                }
            }
        }
    }
}
