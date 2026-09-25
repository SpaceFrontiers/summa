//! Bounded BMP row packets, sharing dimension encodings with Seismic.
#[cfg(any(feature = "native", feature = "wasm", test))]
use super::{Result, corrupt};
#[cfg(any(feature = "native", feature = "wasm", test))]
use crate::structures::postings::sparse_dimensions;
use crate::structures::postings::sparse_dimensions::Dimensions;

#[cfg(any(feature = "native", feature = "wasm", test))]
const PACKET_ENTRIES: usize = 128;

#[derive(Clone, Copy)]
pub(crate) struct ForwardVector<'a>(pub(super) &'a [u8]);
impl<'a> ForwardVector<'a> {
    #[cfg(any(feature = "native", feature = "wasm", test))]
    pub(crate) fn len(&self) -> usize {
        u32::from_le_bytes(self.0[self.0.len() - 4..].try_into().unwrap()) as usize
    }

    fn packets(self) -> impl Iterator<Item = (Dimensions<'a>, &'a [u8])> {
        let mut bytes = &self.0[..self.0.len() - 4];
        std::iter::from_fn(move || {
            if bytes.is_empty() {
                return None;
            }
            let count = bytes[0] as usize;
            let encoding = bytes[1];
            let dimension_bytes = u16::from_le_bytes(bytes[2..4].try_into().unwrap()) as usize;
            let end = 4 + dimension_bytes;
            let dimensions = Dimensions::<false>::new(&bytes[4..end], count, encoding);
            let impacts = &bytes[end..end + count];
            bytes = &bytes[end + count..];
            Some((dimensions, impacts))
        })
    }

    #[cfg(any(feature = "native", feature = "wasm", test))]
    pub(crate) fn iter(self) -> impl Iterator<Item = (u32, u8)> + 'a {
        self.packets()
            .flat_map(|(dimensions, impacts)| dimensions.zip(impacts.iter().copied()))
    }

    /// Dispatch the dimension encoding once per packet and fuse its decode with
    /// the caller's integer scoring. Iteration remains available to BP consumers.
    #[inline]
    pub(crate) fn fold<B>(self, value: B, mut f: impl FnMut(B, u32, u8) -> B) -> B {
        self.packets().fold(value, |value, (dimensions, impacts)| {
            dimensions.fold_indexed(value, |value, index, dimension| {
                f(value, dimension, impacts[index])
            })
        })
    }

    /// Explicit integrity/maintenance entry point. Query iteration trusts writer output.
    #[cfg(any(feature = "native", feature = "wasm", test))]
    pub(super) fn validate(self, dims: u32) -> Result<()> {
        let end = self
            .0
            .len()
            .checked_sub(4)
            .ok_or_else(|| corrupt("missing row count"))?;
        let mut bytes = &self.0[..end];
        let mut total = 0usize;
        while !bytes.is_empty() {
            let header = bytes.get(..4).ok_or_else(|| corrupt("incomplete packet"))?;
            let count = header[0] as usize;
            let length = u16::from_le_bytes(header[2..4].try_into().unwrap()) as usize;
            let payload = bytes
                .get(4..4 + length)
                .ok_or_else(|| corrupt("incomplete dimensions"))?;
            let expected = match header[1] {
                0 => count * 4,
                1 => count * 2,
                3 => count * 3,
                2 if count >= 8 => {
                    let groups = count / 8;
                    let controls = payload
                        .get(4..4 + groups)
                        .ok_or_else(|| corrupt("incomplete controls"))?;
                    4 + groups
                        + groups * 8
                        + controls
                            .iter()
                            .map(|c| c.count_ones() as usize)
                            .sum::<usize>()
                        + (count % 8) * 4
                }
                _ => return Err(corrupt("unknown dimension encoding")),
            };
            if count == 0 || count > PACKET_ENTRIES || length != expected {
                return Err(corrupt("invalid packet extent"));
            }
            bytes = bytes
                .get(4 + length + count..)
                .ok_or_else(|| corrupt("incomplete impacts"))?;
            total += count;
        }
        if total != self.len() || total == 0 {
            return Err(corrupt("invalid row count"));
        }
        let mut previous = None;
        for (dimension, impact) in self.iter() {
            if dimension >= dims || impact == 0 || previous.is_some_and(|p| p > dimension) {
                return Err(corrupt("invalid dimension order or impact"));
            }
            previous = Some(dimension);
        }
        Ok(())
    }
}

/// One packet of scratch regardless of vector/corpus length. Both ingestion and
/// explicit materialization call this writer; merge only copies its output.
#[cfg(any(feature = "native", feature = "wasm", test))]
pub(crate) struct RowWriter {
    entries: [(u32, u8); PACKET_ENTRIES],
    buffered: usize,
    count: u32,
}
#[cfg(any(feature = "native", feature = "wasm", test))]
impl Default for RowWriter {
    fn default() -> Self {
        Self {
            entries: [(0, 0); PACKET_ENTRIES],
            buffered: 0,
            count: 0,
        }
    }
}
#[cfg(any(feature = "native", feature = "wasm", test))]
impl RowWriter {
    pub(crate) fn push(
        &mut self,
        dimension: u32,
        impact: u8,
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<u64> {
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("BMP forward vector too large"))?;
        self.entries[self.buffered] = (dimension, impact);
        self.buffered += 1;
        if self.buffered == PACKET_ENTRIES {
            self.flush(writer)
        } else {
            Ok(0)
        }
    }

    fn flush(&mut self, writer: &mut dyn std::io::Write) -> std::io::Result<u64> {
        if self.buffered == 0 {
            return Ok(0);
        }
        let entries = &self.entries[..self.buffered];
        let (encoding, dimensions) = sparse_dimensions::encode(entries, true);
        writer.write_all(&[self.buffered as u8, encoding])?;
        writer.write_all(&(dimensions.len() as u16).to_le_bytes())?;
        writer.write_all(&dimensions)?;
        let mut impacts = [0u8; PACKET_ENTRIES];
        for (out, &(_, impact)) in impacts.iter_mut().zip(entries) {
            *out = impact;
        }
        writer.write_all(&impacts[..self.buffered])?;
        let bytes = 4 + dimensions.len() + self.buffered;
        self.buffered = 0;
        Ok(bytes as u64)
    }

    pub(crate) fn finish(&mut self, writer: &mut dyn std::io::Write) -> std::io::Result<u64> {
        let bytes = self.flush(writer)?;
        writer.write_all(&self.count.to_le_bytes())?;
        self.count = 0;
        Ok(bytes + 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_boundaries_and_raw_fallbacks_preserve_all_entries() {
        let mut writer = RowWriter::default();
        for count in [1, 7, 8, 9, 127, 128, 129, 256, 4097, 70_000] {
            for (base, gap) in [(0, 0), (65_530, 3), (100_000, 70_000), (1 << 25, 30_000)] {
                if u64::from(base) + count as u64 * u64::from(gap) > u64::from(u32::MAX) {
                    continue;
                }
                let expected: Vec<_> = (0..count)
                    .map(|i| (base + i * gap, (i % 255 + 1) as u8))
                    .collect();
                let mut bytes = Vec::new();
                let mut written = 0;
                for &(dim, impact) in &expected {
                    written += writer.push(dim, impact, &mut bytes).unwrap();
                }
                written += writer.finish(&mut bytes).unwrap();
                assert_eq!(written as usize, bytes.len());
                let row = ForwardVector(&bytes);
                row.validate(u32::MAX).unwrap();
                assert_eq!(row.len(), count as usize);
                assert_eq!(
                    row.fold(Vec::new(), |mut values, dim, impact| {
                        values.push((dim, impact));
                        values
                    }),
                    expected
                );
                assert_eq!(row.iter().collect::<Vec<_>>(), expected);
                assert_eq!(
                    row.iter().skip(7).take(123).collect::<Vec<_>>(),
                    expected
                        .iter()
                        .copied()
                        .skip(7)
                        .take(123)
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn row_writer_propagates_failure_at_packet_and_count_boundaries() {
        let mut row = RowWriter::default();
        for _ in 0..127 {
            row.push(10, 1, &mut Vec::new()).unwrap();
        }
        assert!(row.push(10, 1, &mut [0u8; 1].as_mut_slice()).is_err());
        let mut row = RowWriter::default();
        row.push(10, 1, &mut Vec::new()).unwrap();
        assert!(row.finish(&mut [0u8; 7].as_mut_slice()).is_err());
    }
}
