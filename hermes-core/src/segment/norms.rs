//! Version-one byte4 scoring norms. Physical chunk lengths are separate.

/// Monotone representatives: exact small lengths, then three mantissa bits.
pub(crate) const fn decode(code: u8) -> u32 {
    if code < 24 {
        code as u32
    } else {
        let value = code - 24;
        let shift = value >> 3;
        let mantissa = (value & 7) as u32;
        24 + if shift == 0 {
            mantissa
        } else {
            (mantissa | 8) << (shift - 1)
        }
    }
}

pub(crate) const REPRESENTATIVES: [u32; 256] = {
    let mut values = [0; 256];
    let mut i = 0;
    while i < 256 {
        values[i] = decode(i as u8);
        i += 1;
    }
    values
};

#[inline]
pub(crate) fn encode(length: u32) -> u8 {
    REPRESENTATIVES
        .partition_point(|&value| value <= length)
        .saturating_sub(1) as u8
}

#[cfg(any(feature = "native", feature = "wasm", test))]
#[inline]
pub(crate) fn quantize(length: u32) -> u32 {
    decode(encode(length))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_norms_round_down_monotonically_and_preserve_zero() {
        assert_eq!(decode(0), 0);
        assert_eq!(decode(255), 2_013_265_944);
        for code in 0..=255u8 {
            assert_eq!(encode(decode(code)), code);
            if code > 0 {
                assert_eq!(encode(decode(code) - 1), code - 1);
            }
        }
        let mut previous = 0;
        for length in 0..=u16::MAX as u32 {
            let value = quantize(length);
            assert!(value <= length && value >= previous);
            assert_eq!(value == 0, length == 0);
            previous = value;
        }
    }
}
