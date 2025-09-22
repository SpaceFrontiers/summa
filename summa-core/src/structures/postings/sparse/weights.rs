//! Shared sparse precision encoding and random-access decoding.
use super::WeightQuantization;
use super::block::MAX_BLOCK_SIZE;
use crate::structures::simd;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Cursor};

pub(crate) fn encode_weights(weights: &[f32], quant: WeightQuantization) -> io::Result<Vec<u8>> {
    if weights.is_empty() && matches!(quant, WeightQuantization::UInt8 | WeightQuantization::UInt4)
    {
        let mut data = Vec::with_capacity(8);
        data.write_f32::<LittleEndian>(1.0)?;
        data.write_f32::<LittleEndian>(0.0)?;
        return Ok(data);
    }
    let encoded_len = match quant {
        WeightQuantization::Float32 => weights.len().saturating_mul(4),
        WeightQuantization::Float16 => weights.len().saturating_mul(2),
        WeightQuantization::UInt8 => 8usize.saturating_add(weights.len()),
        WeightQuantization::UInt4 => 8usize.saturating_add(weights.len().div_ceil(2)),
    };
    let mut data = Vec::with_capacity(encoded_len);
    match quant {
        WeightQuantization::Float32 => {
            for &w in weights {
                data.write_f32::<LittleEndian>(w)?;
            }
        }
        WeightQuantization::Float16 => {
            use half::f16;
            for &w in weights {
                data.write_u16::<LittleEndian>(f16::from_f32(w).to_bits())?;
            }
        }
        WeightQuantization::UInt8 => {
            let min = weights.iter().copied().fold(f32::INFINITY, f32::min);
            let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            let scale = if range < f32::EPSILON {
                1.0
            } else {
                range / 255.0
            };
            data.write_f32::<LittleEndian>(scale)?;
            data.write_f32::<LittleEndian>(min)?;
            for &w in weights {
                data.write_u8(((w - min) / scale).round() as u8)?;
            }
        }
        WeightQuantization::UInt4 => {
            let min = weights.iter().copied().fold(f32::INFINITY, f32::min);
            let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            let scale = if range < f32::EPSILON {
                1.0
            } else {
                range / 15.0
            };
            data.write_f32::<LittleEndian>(scale)?;
            data.write_f32::<LittleEndian>(min)?;
            let mut i = 0;
            while i < weights.len() {
                let q1 = ((weights[i] - min) / scale).round() as u8 & 0x0F;
                let q2 = if i + 1 < weights.len() {
                    ((weights[i + 1] - min) / scale).round() as u8 & 0x0F
                } else {
                    0
                };
                data.write_u8((q2 << 4) | q1)?;
                i += 2;
            }
        }
    }
    Ok(data)
}

/// Decode one admitted encoded weight; bulk block reads use the same codec.
#[inline]
pub(crate) fn decode_weight_at(data: &[u8], quant: WeightQuantization, index: usize) -> f32 {
    match quant {
        WeightQuantization::Float32 => {
            f32::from_le_bytes(data[index * 4..index * 4 + 4].try_into().unwrap())
        }
        WeightQuantization::Float16 => half::f16::from_bits(u16::from_le_bytes(
            data[index * 2..index * 2 + 2].try_into().unwrap(),
        ))
        .to_f32(),
        WeightQuantization::UInt8 | WeightQuantization::UInt4 => {
            let scale = f32::from_le_bytes(data[..4].try_into().unwrap());
            let min = f32::from_le_bytes(data[4..8].try_into().unwrap());
            let q = if quant == WeightQuantization::UInt8 {
                data[8 + index]
            } else {
                (data[8 + index / 2] >> ((index % 2) * 4)) & 15
            };
            q as f32 * scale + min
        }
    }
}

/// Decode an admitted block (at most MAX_BLOCK_SIZE weights) with the existing
/// SIMD bulk paths. Callers clear the destination before decoding.
pub(super) fn decode_weights_into(
    data: &[u8],
    quant: WeightQuantization,
    count: usize,
    out: &mut Vec<f32>,
) {
    match quant {
        WeightQuantization::Float32 => {
            out.reserve(count);
            for chunk in data[..count * 4].as_chunks::<4>().0 {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        WeightQuantization::Float16 => {
            // Bulk convert: read u16 bits → f16 → batch convert_to_f32_slice
            // Uses SIMD F16C on x86_64 when available (half 2.x auto-detects)
            use half::f16;
            use half::slice::HalfFloatSliceExt;
            let byte_count = count * 2;
            let src = &data[..byte_count];
            let mut f16_buf = [f16::ZERO; MAX_BLOCK_SIZE];
            for (value, chunk) in f16_buf[..count].iter_mut().zip(src.as_chunks::<2>().0) {
                *value = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            let start = out.len();
            out.resize(start + count, 0.0);
            f16_buf[..count].convert_to_f32_slice(&mut out[start..start + count]);
        }
        WeightQuantization::UInt8 => {
            let mut cursor = Cursor::new(data);
            let scale = cursor.read_f32::<LittleEndian>().unwrap_or(1.0);
            let min_val = cursor.read_f32::<LittleEndian>().unwrap_or(0.0);
            let offset = cursor.position() as usize;
            out.resize(count, 0.0);
            simd::dequantize_uint8(&data[offset..], out, scale, min_val, count);
        }
        WeightQuantization::UInt4 => {
            let mut cursor = Cursor::new(data);
            let scale = cursor.read_f32::<LittleEndian>().unwrap_or(1.0);
            let min = cursor.read_f32::<LittleEndian>().unwrap_or(0.0);
            let mut i = 0;
            while i < count {
                let byte = cursor.read_u8().unwrap_or(0);
                out.push((byte & 0x0F) as f32 * scale + min);
                i += 1;
                if i < count {
                    out.push((byte >> 4) as f32 * scale + min);
                    i += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_codec_keeps_existing_block_bytes_and_bulk_decode_values() {
        let weights = [-2.0f32, 0.0, 1.0];
        for (precision, expected) in [
            (
                WeightQuantization::Float32,
                weights
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            (WeightQuantization::Float16, vec![0, 0xc0, 0, 0, 0, 0x3c]),
            (
                WeightQuantization::UInt8,
                [
                    (3.0f32 / 255.0).to_le_bytes().as_slice(),
                    (-2.0f32).to_le_bytes().as_slice(),
                    &[0, 170, 255],
                ]
                .concat(),
            ),
            (
                WeightQuantization::UInt4,
                [
                    (3.0f32 / 15.0).to_le_bytes().as_slice(),
                    (-2.0f32).to_le_bytes().as_slice(),
                    &[0xa0, 0x0f],
                ]
                .concat(),
            ),
        ] {
            let encoded = encode_weights(&weights, precision).unwrap();
            assert_eq!(encoded, expected, "{precision:?} encoded bytes");
            let mut decoded = Vec::new();
            decode_weights_into(&encoded, precision, weights.len(), &mut decoded);
            assert_eq!(decoded.len(), weights.len());
            for (i, value) in decoded.iter().enumerate() {
                assert!(
                    (*value - decode_weight_at(&encoded, precision, i)).abs() < 1e-6,
                    "{precision:?} scalar/bulk mismatch at {i}"
                );
            }
        }
    }

    #[test]
    fn shared_sparse_precision_codecs_preserve_signed_endpoints_and_empty_vectors() {
        for precision in [
            WeightQuantization::Float32,
            WeightQuantization::Float16,
            WeightQuantization::UInt8,
            WeightQuantization::UInt4,
        ] {
            let bytes = encode_weights(&[-2.0, 0.0, 1.0], precision).unwrap();
            for (i, expected) in [-2.0, 0.0, 1.0].into_iter().enumerate() {
                assert!((decode_weight_at(&bytes, precision, i) - expected).abs() < 1e-6);
            }
            let empty = encode_weights(&[], precision).unwrap();
            if matches!(
                precision,
                WeightQuantization::UInt8 | WeightQuantization::UInt4
            ) {
                assert_eq!(empty, [1.0f32.to_le_bytes(), 0.0f32.to_le_bytes()].concat());
            } else {
                assert!(empty.is_empty());
            }
        }
    }
}
