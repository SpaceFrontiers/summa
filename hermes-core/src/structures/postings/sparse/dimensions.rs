//! Per-vector dimension encodings. DotVByte's one-bit controls describe eight
//! U16 gaps with U32 prefix sums; absolute tail IDs are raw U32 (arXiv:2602.05445).
#[inline(always)]
fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

pub(crate) const RAW: u8 = 0;
const U16: u8 = 1;
const DOT: u8 = 2;
const U24: u8 = 3;

#[cfg(any(feature = "native", feature = "wasm", test))]
pub(crate) fn encode<T>(values: &[(u32, T)], compact: bool) -> (u8, Vec<u8>) {
    if !compact || values.is_empty() {
        return raw(values, 4);
    }
    let maximum = values.last().unwrap().0;
    let raw_width = if maximum <= u16::MAX as u32 {
        2
    } else if maximum < 1 << 24 {
        3
    } else {
        4
    };
    let groups = values.len() / 8;
    let base = values[0].0;
    let mut bytes = base.to_le_bytes().to_vec();
    bytes.resize(4 + groups, 0);
    let mut previous = base;
    let mut fits = true;
    for (i, &(dim, _)) in values[..groups * 8].iter().enumerate() {
        let gap = dim - previous;
        if gap > u16::MAX as u32 {
            fits = false;
            break;
        }
        previous = dim;
        bytes.push(gap as u8);
        if gap > 255 {
            bytes[4 + i / 8] |= 1 << (i % 8);
            bytes.push((gap >> 8) as u8);
        }
    }
    for &(dim, _) in &values[groups * 8..] {
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
    if fits && groups > 0 && bytes.len() < values.len() * raw_width {
        (DOT, bytes)
    } else {
        raw(values, raw_width)
    }
}

#[cfg(any(feature = "native", feature = "wasm", test))]
fn raw<T>(values: &[(u32, T)], width: usize) -> (u8, Vec<u8>) {
    let mut bytes = Vec::with_capacity(values.len() * width);
    for &(dim, _) in values {
        bytes.extend_from_slice(&dim.to_le_bytes()[..width]);
    }
    (
        match width {
            2 => U16,
            3 => U24,
            _ => RAW,
        },
        bytes,
    )
}

pub(crate) struct Dimensions<'a, const RAW_ONLY: bool = false> {
    bytes: &'a [u8],
    encoding: u8,
    count: usize,
    index: usize,
    offset: usize,
    decoded: [u32; 8],
    previous: u32,
}
impl<'a, const RAW_ONLY: bool> Dimensions<'a, RAW_ONLY> {
    pub(crate) fn new(bytes: &'a [u8], count: usize, encoding: u8) -> Self {
        let encoding = if RAW_ONLY { RAW } else { encoding };
        Self {
            bytes,
            encoding,
            count,
            index: 0,
            offset: 4 + count / 8,
            decoded: [0; 8],
            previous: if encoding == DOT { u32_at(bytes, 0) } else { 0 },
        }
    }
    #[inline]
    fn decode_group(&mut self, group: usize) {
        let control = self.bytes[4 + group];
        self.decoded = decode(control, &self.bytes[self.offset..], self.previous);
        self.offset += 8 + control.count_ones() as usize;
        self.previous = self.decoded[7];
    }

    pub(crate) fn position(&self) -> usize {
        self.index
    }
    // Keep the caller's constant weight precision visible inside the coordinate
    // loop. An outlined fold reintroduces precision dispatch per coordinate.
    #[inline(always)]
    pub(crate) fn fold_indexed<B, F: FnMut(B, usize, u32) -> B>(
        mut self,
        mut value: B,
        mut f: F,
    ) -> B {
        // One format dispatch per vector. Feed decoded groups directly into the
        // caller's weight decode/scoring closure; never materialize a vector.
        match if RAW_ONLY { RAW } else { self.encoding } {
            RAW => {
                for i in self.index..self.count {
                    value = f(value, i, u32_at(self.bytes, i * 4));
                }
            }
            U16 => {
                for (offset, bytes) in self.bytes[self.index * 2..].chunks_exact(2).enumerate() {
                    value = f(
                        value,
                        self.index + offset,
                        u16::from_le_bytes(bytes.try_into().unwrap()) as u32,
                    );
                }
            }
            U24 => {
                for (offset, bytes) in self.bytes[self.index * 3..].chunks_exact(3).enumerate() {
                    value = f(
                        value,
                        self.index + offset,
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]),
                    );
                }
            }
            _ => {
                let aligned = self.count / 8 * 8;
                while self.index < aligned && !self.index.is_multiple_of(8) {
                    value = f(value, self.index, self.decoded[self.index % 8]);
                    self.index += 1;
                }
                while self.index < aligned {
                    self.decode_group(self.index / 8);
                    for (lane, dim) in self.decoded.into_iter().enumerate() {
                        value = f(value, self.index + lane, dim);
                    }
                    self.index += 8;
                }
                for i in self.index..self.count {
                    value = f(value, i, u32_at(self.bytes, self.offset));
                    self.offset += 4;
                }
            }
        }
        value
    }
}

impl<const RAW_ONLY: bool> Iterator for Dimensions<'_, RAW_ONLY> {
    type Item = u32;
    #[inline]
    fn fold<B, F: FnMut(B, u32) -> B>(self, value: B, mut f: F) -> B {
        self.fold_indexed(value, |value, _, dim| f(value, dim))
    }
    #[inline(always)]
    fn next(&mut self) -> Option<u32> {
        if self.index == self.count {
            return None;
        }
        let i = self.index;
        self.index += 1;
        Some(match if RAW_ONLY { RAW } else { self.encoding } {
            RAW => u32_at(self.bytes, i * 4),
            U16 => u16_at(self.bytes, i * 2) as u32,
            U24 => u24_at(self.bytes, i * 3),
            _ if i < self.count / 8 * 8 => {
                if i.is_multiple_of(8) {
                    self.decode_group(i / 8);
                }
                self.decoded[i % 8]
            }
            _ => {
                let dim = u32_at(self.bytes, self.offset);
                self.offset += 4;
                dim
            }
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count - self.index;
        (remaining, Some(remaining))
    }
}
#[inline(always)]
fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
}

#[inline(always)]
fn u24_at(bytes: &[u8], at: usize) -> u32 {
    let b = &bytes[at..at + 3];
    u32::from_le_bytes([b[0], b[1], b[2], 0])
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const SHUFFLES: [[u8; 16]; 256] = {
    let mut table = [[255; 16]; 256];
    let mut control = 0;
    while control < 256 {
        let mut lane = 0;
        let mut pos = 0;
        while lane < 8 {
            table[control][lane * 2] = pos;
            pos += 1;
            if control & (1 << lane) != 0 {
                table[control][lane * 2 + 1] = pos;
                pos += 1;
            }
            lane += 1;
        }
        control += 1;
    }
    table
};

#[inline]
fn decode(control: u8, bytes: &[u8], previous: u32) -> [u32; 8] {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut tail = [0; 16];
        let input = padded(bytes, &mut tail);
        // AArch64 always has NEON. Both input and shuffle have at least 16 bytes;
        // the output has eight U32 lanes. Table lookup zeros absent high bytes.
        unsafe {
            let packed = vld1q_u8(input.as_ptr());
            let shuffle = vld1q_u8(SHUFFLES[control as usize].as_ptr());
            let gaps = vreinterpretq_u16_u8(vqtbl1q_u8(packed, shuffle));
            let mut low = vmovl_u16(vget_low_u16(gaps));
            let mut high = vmovl_u16(vget_high_u16(gaps));
            let zero = vdupq_n_u32(0);
            low = vaddq_u32(low, vextq_u32::<3>(zero, low));
            low = vaddq_u32(low, vextq_u32::<2>(zero, low));
            high = vaddq_u32(high, vextq_u32::<3>(zero, high));
            high = vaddq_u32(high, vextq_u32::<2>(zero, high));
            low = vaddq_u32(low, vdupq_n_u32(previous));
            high = vaddq_u32(high, vdupq_n_u32(vgetq_lane_u32::<3>(low)));
            let mut out = [0; 8];
            vst1q_u32(out.as_mut_ptr(), low);
            vst1q_u32(out.as_mut_ptr().add(4), high);
            out
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("ssse3") {
            let mut tail = [0; 16];
            // Runtime dispatch establishes SSSE3; padded guarantees 16 readable bytes.
            return unsafe { decode_ssse3(control, padded(bytes, &mut tail), previous) };
        }
        decode_scalar(control, bytes, previous)
    }
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn padded<'a>(bytes: &'a [u8], tail: &'a mut [u8; 16]) -> &'a [u8] {
    if bytes.len() >= 16 {
        bytes
    } else {
        tail[..bytes.len()].copy_from_slice(bytes);
        tail
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn decode_ssse3(control: u8, input: &[u8], previous: u32) -> [u32; 8] {
    use std::arch::x86_64::*;
    // Caller establishes the CPU feature and a complete 16-byte input window.
    // The control table contains 16 bytes and output holds two 16-byte registers.
    unsafe {
        let packed = _mm_loadu_si128(input.as_ptr().cast());
        let shuffle = _mm_loadu_si128(SHUFFLES[control as usize].as_ptr().cast());
        let gaps = _mm_shuffle_epi8(packed, shuffle);
        let zero = _mm_setzero_si128();
        let mut low = _mm_unpacklo_epi16(gaps, zero);
        let mut high = _mm_unpackhi_epi16(gaps, zero);
        low = _mm_add_epi32(low, _mm_slli_si128::<4>(low));
        low = _mm_add_epi32(low, _mm_slli_si128::<8>(low));
        high = _mm_add_epi32(high, _mm_slli_si128::<4>(high));
        high = _mm_add_epi32(high, _mm_slli_si128::<8>(high));
        low = _mm_add_epi32(low, _mm_set1_epi32(previous as i32));
        high = _mm_add_epi32(high, _mm_shuffle_epi32::<255>(low));
        let mut out = [0; 8];
        _mm_storeu_si128(out.as_mut_ptr().cast(), low);
        _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), high);
        out
    }
}

#[cfg(any(not(target_arch = "aarch64"), test))]
fn decode_scalar(control: u8, bytes: &[u8], mut previous: u32) -> [u32; 8] {
    let mut offset = 0;
    std::array::from_fn(|i| {
        let width = 1 + ((control >> i) & 1) as usize;
        let gap = if width == 1 {
            bytes[offset] as u16
        } else {
            u16_at(bytes, offset)
        };
        offset += width;
        previous += gap as u32;
        previous
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_control_decodes_identically_with_exact_extents() {
        for control in 0..=255u8 {
            let mut bytes = Vec::new();
            for lane in 0..8 {
                bytes.push(lane + 1);
                if control & (1 << lane) != 0 {
                    bytes.push(1);
                }
            }
            assert_eq!(
                decode(control, &bytes, 19),
                decode_scalar(control, &bytes, 19)
            );
        }
    }
    #[test]
    fn alignment_tails_partial_iteration_and_large_ids_roundtrip() {
        for count in 0..=137 {
            for stride in [1, 255, 256, 512, 65_535, 100_000] {
                let values: Vec<_> = (0..count).map(|i| (i as u32 * stride, 0.0)).collect();
                for compact in [false, true] {
                    let (encoding, bytes) = encode(&values, compact);
                    assert!(bytes.len() <= count * 4);
                    for consumed in [0, 1, 7, 8, 9, count] {
                        let mut dims = Dimensions::<false>::new(&bytes, count, encoding);
                        for _ in 0..consumed {
                            dims.next();
                        }
                        let actual = dims.fold(Vec::new(), |mut out, dim| {
                            out.push(dim);
                            out
                        });
                        assert_eq!(
                            actual,
                            values
                                .iter()
                                .skip(consumed)
                                .map(|p| p.0)
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn vocabularies_above_u16_compress_gaps_without_truncating_dimensions() {
        for base in [0, 70_000, u32::MAX - 10_000] {
            let values: Vec<_> = (0..137).map(|i| (base + i * 7, 0.0)).collect();
            let (encoding, bytes) = encode(&values, true);
            assert_eq!(encoding, DOT);
            assert!(
                Dimensions::<false>::new(&bytes, values.len(), encoding)
                    .eq(values.iter().map(|v| v.0))
            );
        }
    }
    #[test]
    fn fixed_width_fallbacks_keep_exact_u16_u24_and_u32_boundaries() {
        for (dimension, expected) in [
            (65535, U16),
            (65536, U24),
            (0x00ff_ffff, U24),
            (0x0100_0000, RAW),
            (u32::MAX, RAW),
        ] {
            let values = [(dimension, 1.0)];
            let (encoding, bytes) = encode(&values, true);
            assert_eq!(encoding, expected);
            assert_eq!(
                Dimensions::<false>::new(&bytes, 1, encoding).collect::<Vec<_>>(),
                [dimension]
            );
        }
    }
}
