use super::*;
/// Borrowed vector in its configured precision; dimensions always use U32.
#[derive(Clone, Copy)]
pub(crate) struct ForwardVector<'a> {
    pub(super) bytes: &'a [u8],
    count: usize,
    quantization: WeightQuantization,
}
impl<'a> ForwardVector<'a> {
    pub(super) fn new(bytes: &'a [u8], count: usize, quantization: WeightQuantization) -> Self {
        Self {
            bytes,
            count,
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
    pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        ForwardIter {
            dimensions: &self.bytes[..self.count * 4],
            weights: &self.bytes[self.count * 4..],
            indices: 0..self.count,
            quantization: self.quantization,
        }
    }
}

struct ForwardIter<'a> {
    dimensions: &'a [u8],
    weights: &'a [u8],
    indices: std::ops::Range<usize>,
    quantization: WeightQuantization,
}

impl ForwardIter<'_> {
    /// Inlined into one arm of the precision dispatch below. A constant format
    /// lets the shared scalar decoder reduce to that format's loads/conversion.
    #[inline(always)]
    fn fold_precision<B, F>(self, mut value: B, mut fold: F, precision: WeightQuantization) -> B
    where
        F: FnMut(B, (u32, f32)) -> B,
    {
        for i in self.indices {
            let weight =
                crate::structures::postings::decode_sparse_weight_at(self.weights, precision, i);
            value = fold(value, (u32_at(self.dimensions, i * 4), weight));
        }
        value
    }
}

impl Iterator for ForwardIter<'_> {
    type Item = (u32, f32);

    // The release profile showed the previous map closure staying out of line
    // for every coordinate. Inline this measured hot call, keeping one codec.
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.indices.next()?;
        let weight = crate::structures::postings::decode_sparse_weight_at(
            self.weights,
            self.quantization,
            i,
        );
        Some((u32_at(self.dimensions, i * 4), weight))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
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
pub(super) fn vector_bytes(count: usize, quantization: WeightQuantization) -> Option<usize> {
    let weights = match quantization {
        WeightQuantization::Float32 => count.checked_mul(4)?,
        WeightQuantization::Float16 => count.checked_mul(2)?,
        WeightQuantization::UInt8 => count.checked_add(8)?,
        WeightQuantization::UInt4 => count.div_ceil(2).checked_add(8)?,
    };
    count.checked_mul(4)?.checked_add(weights)
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
            for count in [0, 1, 7, 129] {
                let weights: Vec<_> = (0..count).map(|i| (i % 17) as f32 - 8.0).collect();
                let encoded =
                    crate::structures::postings::encode_sparse_weights(&weights, precision)
                        .unwrap();
                let mut bytes = Vec::new();
                for i in 0..count {
                    bytes.extend_from_slice(&(i as u32 * 2 + 1).to_le_bytes());
                }
                bytes.extend_from_slice(&encoded);
                let vector = ForwardVector::new(&bytes, count, precision);
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
