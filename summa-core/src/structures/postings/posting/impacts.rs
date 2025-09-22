//! Bounded, score-independent frequency/length envelopes for posting blocks.
//! Records reuse the shared unsigned-vint codec; empty records mean unknown.
use crate::directories::OwnedBytes;
use crate::structures::{read_vint, write_vint};
use std::io;

const MAX_POINTS: usize = 8;
pub(super) const MAX_RECORD_BYTES: usize = 1 + MAX_POINTS * 10;
type Point = (u32, u32);
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

// Orientation of (1/tf, length/tf), without rounding either coordinate.
fn turn(a: Point, b: Point, c: Point) -> i128 {
    let (at, al, bt, bl, ct, cl) = (
        i128::from(a.0),
        i128::from(a.1),
        i128::from(b.0),
        i128::from(b.1),
        i128::from(c.0),
        i128::from(c.1),
    );
    (at - bt) * (cl * at - al * ct) - (bl * at - al * bt) * (at - ct)
}

fn read_point(input: &mut &[u8]) -> io::Result<Point> {
    let tf =
        u32::try_from(read_vint(input)?).map_err(|_| invalid("impact frequency exceeds u32"))?;
    let len = u32::try_from(read_vint(input)?).map_err(|_| invalid("impact length exceeds u32"))?;
    if tf == 0 || len == 0 {
        return Err(invalid("impact coordinates must be positive"));
    }
    Ok((tf, len))
}

fn validate_record(record: &[u8]) -> io::Result<()> {
    if record.is_empty() {
        return Ok(());
    }
    if record.len() > MAX_RECORD_BYTES || !(1..=MAX_POINTS).contains(&(record[0] as usize)) {
        return Err(invalid("invalid impact point count or record extent"));
    }
    let mut input = &record[1..];
    let mut previous: Option<Point> = None;
    let mut before_previous: Option<Point> = None;
    for _ in 0..record[0] {
        let point = read_point(&mut input)?;
        if let Some(last) = previous {
            if point.0 >= last.0
                || point.1 >= last.1
                || u64::from(point.1) * u64::from(last.0) >= u64::from(last.1) * u64::from(point.0)
            {
                return Err(invalid("impact points are not strictly ordered"));
            }
            if before_previous.is_some_and(|before| turn(before, last, point) <= 0) {
                return Err(invalid("impact envelope is not convex"));
            }
        }
        before_previous = previous;
        previous = Some(point);
    }
    if !input.is_empty() {
        return Err(invalid("unaddressed impact record bytes"));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(super) struct ImpactTable {
    bytes: OwnedBytes,
    blocks: usize,
}
impl ImpactTable {
    pub(super) fn validate(bytes: &[u8], blocks: usize) -> io::Result<()> {
        Self::validate_with(bytes, blocks, || Ok(()))
    }

    pub(super) fn validate_with(
        bytes: &[u8],
        blocks: usize,
        mut check: impl FnMut() -> io::Result<()>,
    ) -> io::Result<()> {
        let directory = blocks
            .checked_add(1)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("impact directory overflow"))?;
        let payload = bytes
            .get(directory..)
            .ok_or_else(|| invalid("truncated impact directory"))?;
        let mut previous = 0;
        for (i, entry) in bytes[..directory].chunks_exact(4).enumerate() {
            if i.is_multiple_of(4096) {
                check()?;
            }
            let next = u32::from_le_bytes(entry.try_into().unwrap()) as usize;
            if next < previous || next > payload.len() || (i == 0 && next != 0) {
                return Err(invalid("invalid impact record offset"));
            }
            if i > 0 {
                validate_record(&payload[previous..next])?;
            }
            previous = next;
        }
        if previous != payload.len() {
            return Err(invalid("unaddressed impact payload bytes"));
        }
        Ok(())
    }
    pub(super) fn from_validated(bytes: OwnedBytes, blocks: usize) -> Self {
        Self { bytes, blocks }
    }
    pub(super) fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }
    pub(super) fn record_count(&self) -> usize {
        self.blocks
    }
    pub(super) fn record(&self, block: usize) -> Option<&[u8]> {
        if block >= self.blocks {
            return None;
        }
        Some(Self::record_from_validated(
            self.bytes.as_slice(),
            self.blocks,
            block,
        ))
    }
    pub(super) fn record_from_validated(raw: &[u8], blocks: usize, block: usize) -> &[u8] {
        let at = block * 4;
        let from = u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) as usize;
        let to = u32::from_le_bytes(raw[at + 4..at + 8].try_into().unwrap()) as usize;
        let base = (blocks + 1) * 4;
        &raw[base + from..base + to]
    }
    /// Geometry only; the query owner supplies nonnegative coefficients and
    /// converts the minimum into its guarded similarity bound.
    pub(super) fn minimum(&self, block: usize, reciprocal: f64, ratio: f64) -> Option<f64> {
        let record = self.record(block)?;
        let (&count, mut input) = record.split_first()?;
        let mut minimum = f64::INFINITY;
        for _ in 0..count {
            let (tf, len) = read_point(&mut input).expect("validated immutable impact record");
            minimum = minimum.min((reciprocal + ratio * f64::from(len)) / f64::from(tf));
        }
        Some(minimum)
    }
}

/// Metadata output only. The posting writer owns document encoding and decides
/// when a changed block needs construction versus an unchanged record copy.
pub(super) struct ImpactBuilder {
    bytes: Vec<u8>,
    blocks: usize,
    used: usize,
}
impl ImpactBuilder {
    #[cfg(feature = "native")]
    pub(super) fn budget_with_groups(blocks: usize) -> Option<usize> {
        Self::budget(blocks.checked_add(blocks.div_ceil(super::L1_INTERVAL))?)
    }
    pub(super) fn with_groups(blocks: usize) -> io::Result<Self> {
        Self::new(
            blocks
                .checked_add(blocks.div_ceil(super::L1_INTERVAL))
                .ok_or_else(|| invalid("impact group count overflow"))?,
        )
    }

    /// Aggregate only small metadata. The posting owner supplies a byte copy
    /// when an existing group keeps exactly the same membership.
    pub(super) fn append_groups_with<'a>(
        &mut self,
        blocks: usize,
        mut unchanged: impl FnMut(std::ops::Range<usize>) -> io::Result<Option<&'a [u8]>>,
    ) -> io::Result<()> {
        if self.used != blocks {
            return Err(invalid("impact groups require complete L0 output"));
        }
        for start in (0..blocks).step_by(super::L1_INTERVAL) {
            let end = (start + super::L1_INTERVAL).min(blocks);
            if let Some(record) = unchanged(start..end)? {
                self.append(record)?;
                continue;
            }
            let mut points = [(0u32, 0u32); super::L1_INTERVAL * MAX_POINTS];
            let mut count = 0;
            let mut complete = true;
            for block in start..end {
                let record = ImpactTable::record_from_validated(&self.bytes, self.blocks, block);
                let Some((&size, mut input)) = record.split_first() else {
                    complete = false;
                    break;
                };
                for _ in 0..size {
                    points[count] = read_point(&mut input)
                        .expect("canonical or validated copied impact record");
                    count += 1;
                }
            }
            if complete {
                self.append_points(&mut points[..count])?;
            } else {
                self.append(&[])?;
            }
        }
        Ok(())
    }
    pub(super) fn budget(blocks: usize) -> Option<usize> {
        blocks
            .checked_add(1)?
            .checked_mul(4)?
            .checked_add(blocks.checked_mul(MAX_RECORD_BYTES)?)
    }
    pub(super) fn new(blocks: usize) -> io::Result<Self> {
        blocks
            .checked_mul(MAX_RECORD_BYTES)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| invalid("impact payload exceeds u32 offsets"))?;
        let capacity = Self::budget(blocks).ok_or_else(|| invalid("impact directory overflow"))?;
        let mut bytes = Vec::with_capacity(capacity);
        bytes.resize((blocks + 1) * 4, 0);
        Ok(Self {
            bytes,
            blocks,
            used: 0,
        })
    }
    pub(super) fn append(&mut self, record: &[u8]) -> io::Result<()> {
        if self.used == self.blocks || record.len() > MAX_RECORD_BYTES {
            return Err(invalid("impact output exceeds admitted block budget"));
        }
        let base = (self.blocks + 1) * 4;
        let end = (self.bytes.len() - base)
            .checked_add(record.len())
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| invalid("impact payload exceeds u32 offsets"))?;
        self.bytes.extend_from_slice(record);
        self.used += 1;
        self.bytes[self.used * 4..self.used * 4 + 4].copy_from_slice(&end.to_le_bytes());
        Ok(())
    }
    pub(super) fn append_points(&mut self, points: &mut [Point]) -> io::Result<()> {
        if points.is_empty() || points.iter().any(|p| p.0 == 0 || p.1 == 0) {
            return self.append(&[]);
        }
        if points.len() > 128 {
            return Err(invalid("impact construction exceeds posting block size"));
        }
        points.sort_unstable_by_key(|&(tf, len)| (std::cmp::Reverse(tf), len));
        let mut hull = [(0u32, 0u32); 128];
        let mut size = 0;
        for &point in points.iter() {
            if size > 0 {
                let last = hull[size - 1];
                if u64::from(point.1) * u64::from(last.0) >= u64::from(last.1) * u64::from(point.0)
                {
                    continue;
                }
            }
            while size >= 2 && turn(hull[size - 2], hull[size - 1], point) <= 0 {
                size -= 1;
            }
            hull[size] = point;
            size += 1;
        }
        if size > MAX_POINTS {
            return self.append(&[]);
        }
        let mut bytes = [0u8; MAX_RECORD_BYTES];
        bytes[0] = size as u8;
        let mut output = io::Cursor::new(&mut bytes[1..]);
        for &(tf, len) in &hull[..size] {
            write_vint(&mut output, u64::from(tf))?;
            write_vint(&mut output, u64::from(len))?;
        }
        let len = output.position() as usize + 1;
        self.append(&bytes[..len])
    }
    pub(super) fn finish(mut self) -> Option<ImpactTable> {
        let reserved = (self.blocks + 1) * 4;
        let actual = (self.used + 1) * 4;
        if self.bytes.len() == reserved {
            return None;
        }
        let records = self.bytes.len() - reserved;
        if reserved != actual {
            self.bytes.copy_within(reserved.., actual);
            self.bytes.truncate(actual + records);
        }
        Some(ImpactTable::from_validated(
            OwnedBytes::new(self.bytes),
            self.used,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_envelopes_use_existing_vints_and_oversized_envelopes_are_unknown() {
        let mut builder = ImpactBuilder::new(2).unwrap();
        let mut eight: Vec<_> = (1..=8).map(|tf| (tf, tf * tf)).collect();
        let mut nine: Vec<_> = (1..=9).map(|tf| (tf, tf * tf)).collect();
        builder.append_points(&mut eight).unwrap();
        builder.append_points(&mut nine).unwrap();
        let table = builder.finish().unwrap();
        ImpactTable::validate(table.bytes(), 2).unwrap();
        assert_eq!(
            table.record(0).unwrap(),
            [8, 8, 64, 7, 49, 6, 36, 5, 25, 4, 16, 3, 9, 2, 4, 1, 1]
        );
        assert!(table.record(1).unwrap().is_empty());
        assert_eq!(table.minimum(1, 1.0, 1.0), None);
        let mut copied = ImpactBuilder::new(3).unwrap();
        copied.append(&[]).unwrap();
        for i in 0..2 {
            copied.append(table.record(i).unwrap()).unwrap();
        }
        let copied = copied.finish().unwrap();
        ImpactTable::validate(copied.bytes(), 3).unwrap();
        assert_eq!(table.record(0), copied.record(1));
        assert_eq!(table.record(1), copied.record(2));
    }
    #[test]
    fn encoded_envelopes_preserve_minima_across_positive_parameters_and_u32_extremes() {
        let mut seed = 7u64;
        for trial in 0..1000 {
            let mut next = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 32) as u32
            };
            let values = [1, 2, 3, 127, 65535, 65536, 16777217, u32::MAX];
            let mut points: Vec<_> = (0..[1, 3, 8, 127, 128][trial % 5])
                .map(|_| {
                    if trial % 2 == 0 {
                        (
                            values[next() as usize % values.len()],
                            values[next() as usize % values.len()],
                        )
                    } else {
                        (next().max(1), next().max(1))
                    }
                })
                .collect();
            let mut builder = ImpactBuilder::new(1).unwrap();
            builder.append_points(&mut points).unwrap();
            if let Some(table) = builder.finish() {
                ImpactTable::validate(table.bytes(), 1).unwrap();
                for a in [0.0, 0.01, 0.25, 1.0] {
                    for b in [0.0, 1e-39, 0.0001, 0.75, 1.0] {
                        let expected = points
                            .iter()
                            .map(|&(tf, len)| (a + b * f64::from(len)) / f64::from(tf))
                            .fold(f64::INFINITY, f64::min);
                        let actual = table.minimum(0, a, b).unwrap();
                        assert!(
                            (actual - expected).abs() <= expected.abs() * 8.0 * f64::EPSILON,
                            "trial={trial} a={a} b={b} expected={expected} actual={actual}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn malformed_or_incomplete_envelopes_never_become_validated_views() {
        for record in [
            vec![0],
            vec![9],
            vec![0x81, 1, 1],
            vec![1, 0, 1],
            vec![1, 1, 0],
            vec![1, 0x80],
            vec![1, 1, 1, 0],
            vec![2, 1, 1, 2, 4],                      // frequencies out of order
            vec![2, 4, 2, 2, 1],                      // dominated equal length/TF ratio
            vec![3, 4, 16, 3, 12, 2, 4],              // non-convex/dominated intermediate point
            vec![1, 0xff, 0xff, 0xff, 0xff, 0x1f, 1], // exceeds u32
            vec![
                1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 2, 1,
            ],
        ] {
            assert!(validate_record(&record).is_err(), "{record:?}");
        }
        let mut builder = ImpactBuilder::new(1).unwrap();
        builder.append_points(&mut [(u32::MAX, u32::MAX)]).unwrap();
        let table = builder.finish().unwrap();
        ImpactTable::validate(table.bytes(), 1).unwrap();
        for end in 0..table.bytes().len() {
            assert!(ImpactTable::validate(&table.bytes()[..end], 1).is_err());
        }
        assert!(ImpactTable::validate(table.bytes(), usize::MAX).is_err());
        assert!(ImpactBuilder::new(usize::MAX).is_err());
        for (offset, value) in [(0, 1u32), (4, 0), (4, u32::MAX)] {
            let mut bad = table.bytes().to_vec();
            bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(ImpactTable::validate(&bad, 1).is_err());
        }
        let mut bad = table.bytes().to_vec();
        bad.push(0);
        assert!(ImpactTable::validate(&bad, 1).is_err());
        assert_eq!(
            ImpactTable::validate_with(table.bytes(), 1, || Err(io::ErrorKind::Interrupted.into()))
                .unwrap_err()
                .kind(),
            io::ErrorKind::Interrupted
        );
    }

    #[test]
    fn impact_records_copy_across_legacy_merges_without_changing_encoded_postings() {
        use super::super::*;
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let mut postings = PostingList::new();
            for doc in 0..389 {
                postings.push(doc * 7, doc % 8 + 1);
            }
            let length = |doc: u32| (doc / 7 % 8 + 1).pow(2);
            let plain = BlockPostingList::from_posting_list_with_ratio_bounds(
                &postings,
                true,
                Some(&length),
                codec,
            )
            .unwrap();
            let impacts = BlockPostingList::from_posting_list_with_impact_bounds(
                &postings,
                true,
                Some(&length),
                codec,
            )
            .unwrap();
            assert_eq!(plain.stream.as_slice(), impacts.stream.as_slice());
            assert_eq!(plain.l0_bytes.as_slice(), impacts.l0_bytes.as_slice());
            assert_eq!(
                plain.ratios.as_ref().unwrap().as_slice(),
                impacts.ratios.as_ref().unwrap().as_slice()
            );
            let mut plain_bytes = Vec::new();
            plain.serialize(&mut plain_bytes).unwrap();
            let mut impact_bytes = Vec::new();
            impacts.serialize(&mut impact_bytes).unwrap();
            let reopened = BlockPostingList::deserialize(&impact_bytes).unwrap();
            let mut roundtrip = Vec::new();
            reopened.serialize(&mut roundtrip).unwrap();
            assert_eq!(roundtrip, impact_bytes);
            assert_eq!(reopened.block_impact_point_count(0), Some(8));
            assert_eq!(reopened.block_impact_point_count(99), None);
            let merged =
                BlockPostingList::concatenate_blocks(&[(plain, 0), (impacts.clone(), 10_000)])
                    .unwrap();
            let mut typed_bytes = Vec::new();
            merged.serialize(&mut typed_bytes).unwrap();
            let mut streamed = Vec::new();
            let (_, size) = BlockPostingList::concatenate_streaming(
                &[(&plain_bytes, 0), (&impact_bytes, 10_000)],
                &mut streamed,
            )
            .unwrap();
            assert_eq!(streamed, typed_bytes);
            assert_eq!(size, streamed.len());
            for i in 0..impacts.num_blocks() {
                assert_eq!(merged.block_impact_point_count(i), Some(0));
                assert_eq!(
                    merged
                        .impacts
                        .as_ref()
                        .unwrap()
                        .record(i + impacts.num_blocks()),
                    impacts.impacts.as_ref().unwrap().record(i)
                );
            }
            let footer = Footer::parse(&impact_bytes).unwrap();
            for at in [
                footer.ratios_end(),
                footer.ratios_end() + (footer.l0_count + 1) * 4,
                impact_bytes.len() - FOOTER_V2_SIZE - 1,
            ] {
                let mut bad = impact_bytes.clone();
                bad[at] = 255;
                assert!(BlockPostingList::deserialize(&bad).is_err());
                let mut output = Vec::new();
                assert!(
                    BlockPostingList::concatenate_streaming(&[(&bad, 0)], &mut output).is_err()
                );
                assert!(output.is_empty());
            }
            let mut short = PostingList::new();
            short.push(1, 1);
            let short = BlockPostingList::from_posting_list_with_impact_bounds(
                &short,
                false,
                Some(&|_| 1),
                codec,
            )
            .unwrap();
            assert!(!short.has_impact_bounds());
        }
    }
    #[test]
    fn group_envelopes_preserve_minima_and_unknown_constituents_without_posting_decode() {
        let mut builder = ImpactBuilder::with_groups(17).unwrap();
        for block in 0..17 {
            if block == 9 {
                builder.append(&[]).unwrap();
            } else {
                builder
                    .append_points(&mut [(1, 1 + block), (4, 16 + block)])
                    .unwrap();
            }
        }
        builder.append_groups_with(17, |_| Ok(None)).unwrap();
        let table = builder.finish().unwrap();
        ImpactTable::validate(table.bytes(), 20).unwrap();
        assert!(table.record(18).unwrap().is_empty());
        for group in [0, 2] {
            for a in [0.0, 0.25, 1.0] {
                for b in [0.0, 0.001, 0.75] {
                    let expected = (group * 8..((group + 1) * 8).min(17))
                        .map(|block| table.minimum(block, a, b).unwrap())
                        .fold(f64::INFINITY, f64::min);
                    assert_eq!(table.minimum(17 + group, a, b), Some(expected));
                }
            }
        }
        assert!(ImpactBuilder::with_groups(usize::MAX).is_err());
        let mut small = ImpactBuilder::new(8).unwrap();
        for _ in 0..8 {
            small.append(&[1, 1, 1]).unwrap();
        }
        assert!(small.append_groups_with(8, |_| Ok(None)).is_err());
    }

    #[test]
    fn aligned_groups_copy_original_vint_bytes_and_regrouping_keeps_l0_records_unchanged() {
        use super::super::*;
        let mut postings = PostingList::new();
        for doc in 0..1024 {
            postings.push(doc, 1);
        }
        let mut list = BlockPostingList::from_posting_list_with_impact_bounds(
            &postings,
            false,
            Some(&|_| 1),
            PostingCodec::Rounded,
        )
        .unwrap();
        // Legal nonminimal shared-vint representation distinguishes a byte
        // copy from reconstruction of the same numerical envelope.
        let original_group = [1, 0x81, 0, 0x81, 0];
        let mut records = ImpactBuilder::with_groups(8).unwrap();
        for block in 0..8 {
            records
                .append(list.impacts.as_ref().unwrap().record(block).unwrap())
                .unwrap();
        }
        records.append(&original_group).unwrap();
        list.impacts = records.finish();
        let mut original = Vec::new();
        list.serialize(&mut original).unwrap();
        BlockPostingList::deserialize(&original).unwrap();
        let mut old = list.clone();
        let mut records = ImpactBuilder::new(8).unwrap();
        for block in 0..8 {
            records
                .append(list.impacts.as_ref().unwrap().record(block).unwrap())
                .unwrap();
        }
        old.impacts = records.finish();
        assert!(!old.has_group_impact_bounds());
        let mut old_bytes = Vec::new();
        old.serialize(&mut old_bytes).unwrap();
        assert!(
            !BlockPostingList::deserialize(&old_bytes)
                .unwrap()
                .has_group_impact_bounds()
        );
        for sources in [
            vec![(list.clone(), 0), (list.clone(), 10_000)],
            vec![(old, 0), (list.clone(), 10_000)],
        ] {
            let merged = BlockPostingList::concatenate_blocks(&sources).unwrap();
            assert_eq!(
                merged.impacts.as_ref().unwrap().record(17),
                Some(original_group.as_slice())
            );
            let bytes: Vec<Vec<u8>> = sources
                .iter()
                .map(|(s, _)| {
                    let mut b = Vec::new();
                    s.serialize(&mut b).unwrap();
                    b
                })
                .collect();
            let refs: Vec<_> = bytes
                .iter()
                .zip(&sources)
                .map(|(b, (_, base))| (b.as_slice(), *base))
                .collect();
            let mut streamed = Vec::new();
            BlockPostingList::concatenate_streaming(&refs, &mut streamed).unwrap();
            let mut typed = Vec::new();
            merged.serialize(&mut typed).unwrap();
            assert_eq!(streamed, typed);
        }
        let mut prefix = PostingList::new();
        prefix.push(0, 1);
        let prefix = BlockPostingList::from_posting_list(&prefix).unwrap();
        let merged =
            BlockPostingList::concatenate_blocks(&[(prefix, 0), (list.clone(), 10)]).unwrap();
        assert_eq!(merged.group_impact_point_count(0), Some(0));
        assert_eq!(
            merged.impacts.as_ref().unwrap().record(10),
            Some([1u8, 1, 1].as_slice())
        );
        for block in 0..8 {
            assert_eq!(
                merged.impacts.as_ref().unwrap().record(block + 1),
                list.impacts.as_ref().unwrap().record(block)
            );
        }
        for flag in [FLAG_IMPACT_BOUNDS, FLAG_L1_BOUNDS] {
            let mut bad = original.clone();
            let at = bad.len() - 12;
            bad[at] &= !(flag as u8);
            assert!(BlockPostingList::deserialize(&bad).is_err());
            let mut out = Vec::new();
            assert!(BlockPostingList::concatenate_streaming(&[(&bad, 0)], &mut out).is_err());
            assert!(out.is_empty());
        }
    }
}
