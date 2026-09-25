use super::dimensions::Dimensions;
use super::*;
/// Borrowed vector in its configured weight precision and dimension encoding.
#[derive(Clone, Copy)]
pub(crate) struct ForwardVector<'a> {
    pub(super) bytes: &'a [u8],
    count: usize,
    pub(super) encoding: u8,
    quantization: WeightQuantization,
}
impl<'a> ForwardVector<'a> {
    pub(super) fn new(
        bytes: &'a [u8],
        count: usize,
        quantization: WeightQuantization,
        encoding: u8,
    ) -> Self {
        Self {
            bytes,
            count,
            encoding,
            quantization,
        }
    }
    #[cfg(any(feature = "native", test))]
    pub(crate) fn byte_len(&self) -> usize {
        self.bytes.len()
    }
    #[cfg(any(feature = "native", test))]
    pub(crate) fn len(&self) -> usize {
        self.count
    }
    #[inline]
    pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.iterator::<false>()
    }

    /// Specialize the shared iterator for raw rows, keeping compact-format
    /// dispatch and scratch out of the raw scoring loop.
    #[inline]
    pub(crate) fn raw_iter(&self) -> Option<impl Iterator<Item = (u32, f32)> + '_> {
        (self.encoding == dimensions::RAW).then(|| self.iterator::<true>())
    }

    #[inline]
    fn iterator<const RAW_ONLY: bool>(&self) -> ForwardIter<'_, RAW_ONLY> {
        let split = if RAW_ONLY {
            self.count * 4
        } else {
            self.bytes.len() - weight_bytes(self.count, self.quantization).unwrap()
        };
        ForwardIter {
            dimensions: Dimensions::new(&self.bytes[..split], self.count, self.encoding),
            weights: &self.bytes[split..],
            quantization: self.quantization,
        }
    }
}

struct ForwardIter<'a, const RAW_ONLY: bool> {
    dimensions: Dimensions<'a, RAW_ONLY>,
    weights: &'a [u8],
    quantization: WeightQuantization,
}

impl<const RAW_ONLY: bool> ForwardIter<'_, RAW_ONLY> {
    /// Inlined into one arm of the precision dispatch below. A constant format
    /// lets the shared scalar decoder reduce to that format's loads/conversion.
    #[inline(always)]
    fn fold_precision<B, F>(self, value: B, mut fold: F, precision: WeightQuantization) -> B
    where
        F: FnMut(B, (u32, f32)) -> B,
    {
        self.dimensions.fold_indexed(value, |value, i, dim| {
            let weight =
                crate::structures::postings::decode_sparse_weight_at(self.weights, precision, i);
            fold(value, (dim, weight))
        })
    }
}

impl<const RAW_ONLY: bool> Iterator for ForwardIter<'_, RAW_ONLY> {
    type Item = (u32, f32);

    // The release profile showed the previous map closure staying out of line
    // for every coordinate. Inline this measured hot call, keeping one codec.
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.dimensions.position();
        let dim = self.dimensions.next()?;
        let weight = crate::structures::postings::decode_sparse_weight_at(
            self.weights,
            self.quantization,
            i,
        );
        Some((dim, weight))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.dimensions.size_hint()
    }

    #[inline]
    fn fold<B, F>(self, value: B, fold: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        match self.quantization {
            WeightQuantization::Float32 => {
                self.fold_precision(value, fold, WeightQuantization::Float32)
            }
            WeightQuantization::Float16 => {
                self.fold_precision(value, fold, WeightQuantization::Float16)
            }
            WeightQuantization::UInt8 => {
                self.fold_precision(value, fold, WeightQuantization::UInt8)
            }
            WeightQuantization::UInt4 => {
                self.fold_precision(value, fold, WeightQuantization::UInt4)
            }
        }
    }
}
pub(super) fn weight_bytes(count: usize, quantization: WeightQuantization) -> Option<usize> {
    Some(match quantization {
        WeightQuantization::Float32 => count.checked_mul(4)?,
        WeightQuantization::Float16 => count.checked_mul(2)?,
        WeightQuantization::UInt8 => count.checked_add(8)?,
        WeightQuantization::UInt4 => count.div_ceil(2).checked_add(8)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_iterator_preserves_every_precision_and_partial_consumption() {
        for precision in [
            WeightQuantization::Float32,
            WeightQuantization::Float16,
            WeightQuantization::UInt8,
            WeightQuantization::UInt4,
        ] {
            for count in [0, 1, 7, 8, 9, 129] {
                for compact in [false, true] {
                    let weights: Vec<_> = (0..count).map(|i| (i % 17) as f32 - 8.0).collect();
                    let encoded =
                        crate::structures::postings::encode_sparse_weights(&weights, precision)
                            .unwrap();
                    let values: Vec<_> =
                        (0..count).map(|i| (i as u32 * 2 + 1, weights[i])).collect();
                    let (encoding, mut bytes) = dimensions::encode(&values, compact);
                    bytes.extend_from_slice(&encoded);
                    let vector = ForwardVector::new(&bytes, count, precision, encoding);
                    let expected: Vec<_> = (0..count)
                        .map(|i| {
                            (
                                i as u32 * 2 + 1,
                                crate::structures::postings::decode_sparse_weight_at(
                                    &encoded, precision, i,
                                )
                                .to_bits(),
                            )
                        })
                        .collect();
                    if let Some(raw) = vector.raw_iter() {
                        assert_eq!(
                            raw.map(|(d, w)| (d, w.to_bits())).collect::<Vec<_>>(),
                            expected
                        );
                        assert_eq!(
                            vector.raw_iter().unwrap().fold(0.0, |sum, (_, w)| sum + w),
                            vector.iter().fold(0.0, |sum, (_, w)| sum + w)
                        );
                    }
                    let mut actual = vector.iter();
                    for (i, expected) in expected.iter().enumerate() {
                        assert_eq!(actual.size_hint(), (count - i, Some(count - i)));
                        assert_eq!(
                            actual.next().map(|(dim, value)| (dim, value.to_bits())),
                            Some(*expected)
                        );
                    }
                    assert_eq!(actual.next(), None);
                    assert_eq!(actual.next(), None);
                    assert_eq!(actual.size_hint(), (0, Some(0)));

                    let mut partial = vector.iter();
                    let _ = partial.nth(2);
                    assert_eq!(
                        partial
                            .map(|(dim, value)| (dim, value.to_bits()))
                            .collect::<Vec<_>>(),
                        expected.iter().copied().skip(3).collect::<Vec<_>>()
                    );

                    // Exercise the specialized fold both from the start and after
                    // consuming coordinates, with an order-sensitive reduction.
                    for consumed in [0, 1, 3, count] {
                        let mut partial = vector.iter();
                        if consumed > 0 {
                            let _ = partial.next();
                        }
                        if consumed > 1 {
                            let _ = partial.nth(consumed - 2);
                        }
                        let combine = |hash: u64, (dimension, bits): (u32, u32)| {
                            hash.rotate_left(7) ^ (u64::from(dimension) << 32) ^ u64::from(bits)
                        };
                        let actual = partial.fold(19, |hash, (dimension, value)| {
                            combine(hash, (dimension, value.to_bits()))
                        });
                        let expected = expected.iter().copied().skip(consumed).fold(19, combine);
                        assert_eq!(actual, expected, "{precision:?}: consumed {consumed}");
                    }
                }
            }
        }
    }
}
