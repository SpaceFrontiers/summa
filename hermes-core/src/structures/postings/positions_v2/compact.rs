//! Range copying through the existing position stream encoder.
use super::*;
use crate::directories::FileHandle;
use std::sync::atomic::{AtomicBool, Ordering};

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub(crate) struct PositionRangeSource {
    file: FileHandle,
    index: OwnedBytes,
    index_start: usize,
    blocks: usize,
    total: u64,
    cached: Option<usize>,
    raw: OwnedBytes,
    values: Vec<u32>,
    legacy: Option<TermPositionCursor>,
    budget: usize,
}

impl PositionRangeSource {
    pub(crate) async fn open(
        file: FileHandle,
        budget: usize,
        cancellation: Option<&AtomicBool>,
    ) -> io::Result<Self> {
        let tail = file
            .read_bytes_range(file.len().saturating_sub(FOOTER as u64)..file.len())
            .await?;
        let mut source = Self {
            file,
            index: OwnedBytes::new(Vec::new()),
            index_start: 0,
            blocks: 0,
            total: 0,
            cached: None,
            raw: OwnedBytes::new(Vec::new()),
            values: Vec::with_capacity(POSITION_STREAM_BLOCK),
            legacy: None,
            budget,
        };
        if !PositionStream::is_stream(tail.as_slice()) {
            if source.file.len() > (budget / 16) as u64 || tail.len() != FOOTER {
                return Err(invalid("legacy positions exceed compaction scratch budget"));
            }
            let bytes = source.file.read_bytes().await?;
            let data_len = u64::from_le_bytes(tail[..8].try_into().unwrap());
            let count = u32::from_le_bytes(tail[8..12].try_into().unwrap()) as u64;
            if data_len.checked_add(count * 20) != Some(source.file.len() - FOOTER as u64) {
                return Err(invalid("invalid legacy position directory"));
            }
            for entry in bytes[data_len as usize..bytes.len() - FOOTER].chunks_exact(20) {
                let offset = u64::from_le_bytes(entry[8..16].try_into().unwrap());
                let len = u32::from_le_bytes(entry[16..20].try_into().unwrap()) as u64;
                if offset.checked_add(len).is_none_or(|end| end > data_len) {
                    return Err(invalid("invalid legacy position block"));
                }
            }
            source.legacy = Some(TermPositions::open(bytes)?.into_cursor());
            return Ok(source);
        }
        let total_len = usize::try_from(source.file.len())
            .map_err(|_| invalid("position file exceeds address space"))?;
        let (blocks, index_start, total) =
            PositionStream::parse_layout_tail(tail.as_slice(), total_len)?;
        if blocks.saturating_mul(INDEX_ENTRY) > budget {
            return Err(invalid(
                "position directory exceeds compaction scratch budget",
            ));
        }
        source.index = source
            .file
            .read_bytes_range(index_start as u64..source.file.len() - FOOTER as u64)
            .await?;
        source.index_start = index_start;
        source.blocks = blocks;
        source.total = total;
        let mut previous = None;
        for i in 0..blocks {
            if i.is_multiple_of(4096)
                && cancellation.is_some_and(|flag| flag.load(Ordering::Acquire))
            {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "position compaction cancelled",
                ));
            }
            let entry = source.entry(i);
            if entry.0 >= index_start
                || entry.1 >= total
                || (i == 0 && entry != (0, 0))
                || previous.is_some_and(|(offset, value)| entry.0 <= offset || entry.1 <= value)
            {
                return Err(invalid("invalid position source directory"));
            }
            previous = Some(entry);
        }
        if (blocks == 0) != (total == 0) {
            return Err(invalid("invalid empty position stream"));
        }
        Ok(source)
    }

    pub(crate) fn total_positions(&self) -> u64 {
        self.total
    }
    pub(crate) fn is_stream(&self) -> bool {
        self.legacy.is_none()
    }
    fn entry(&self, i: usize) -> (usize, u64) {
        PositionStream::index_entry(self.index.as_slice(), 0, i)
    }

    pub(crate) async fn append_doc<W: Write>(
        &mut self,
        writer: &mut PositionStreamEncoder<W>,
        doc: DocId,
        cursor: u64,
        tf: u32,
        cancellation: Option<&AtomicBool>,
    ) -> io::Result<()> {
        if let Some(legacy) = &mut self.legacy {
            if (tf as usize).saturating_mul(4) > self.budget / 2 {
                return Err(invalid(
                    "legacy document positions exceed compaction scratch budget",
                ));
            }
            if !legacy.read_into(doc, cursor, tf, &mut self.values)
                || self.values.len() != tf as usize
            {
                return Err(invalid("invalid legacy document positions"));
            }
            return writer.push_doc(&mut self.values);
        }
        let end = cursor
            .checked_add(u64::from(tf))
            .ok_or_else(|| invalid("position cursor overflow"))?;
        self.append_range(writer, cursor..end, cancellation).await
    }

    pub(crate) async fn append_range<W: Write>(
        &mut self,
        writer: &mut PositionStreamEncoder<W>,
        range: std::ops::Range<u64>,
        cancellation: Option<&AtomicBool>,
    ) -> io::Result<()> {
        if range.start > range.end || range.end > self.total || self.legacy.is_some() {
            return Err(invalid("invalid position compaction range"));
        }
        if range.is_empty() {
            return Ok(());
        }
        let mut low = 0;
        let mut high = self.blocks;
        while low < high {
            let middle = low + (high - low) / 2;
            if self.entry(middle).1 <= range.start {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        let mut block = low
            .checked_sub(1)
            .ok_or_else(|| invalid("missing position block"))?;
        let mut cursor = range.start;
        while cursor < range.end {
            if cancellation.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "position compaction cancelled",
                ));
            }
            if block >= self.blocks {
                return Err(invalid("missing position block"));
            }
            let (offset, value_start) = self.entry(block);
            let (end, value_end) = if block + 1 == self.blocks {
                (self.index_start, self.total)
            } else {
                self.entry(block + 1)
            };
            if end < offset || end - offset > BLOCK_HEADER + POSITION_STREAM_BLOCK * 4 {
                return Err(invalid("invalid position block size"));
            }
            if self.cached != Some(block) {
                self.raw = self
                    .file
                    .read_bytes_range(offset as u64..end as u64)
                    .await?;
                if PositionStream::block_count(self.raw.as_slice()).map(|count| count as u64)
                    != Some(value_end - value_start)
                {
                    return Err(invalid("position block disagrees with directory"));
                }
                self.cached = Some(block);
                self.values.clear();
            }
            let take_end = range.end.min(value_end);
            if cursor == value_start && take_end == value_end {
                writer.append_encoded_block(self.raw.as_slice())?;
            } else {
                if self.values.is_empty()
                    && !PositionStream::decode_block_bytes(self.raw.as_slice(), &mut self.values)
                {
                    return Err(invalid("invalid position block"));
                }
                writer.push_values(
                    &self.values
                        [(cursor - value_start) as usize..(take_end - value_start) as usize],
                )?;
            }
            cursor = take_end;
            block += 1;
        }
        Ok(())
    }
}

impl<W: Write> PositionStreamEncoder<W> {
    pub(crate) fn finish_cancellable(
        self,
        cancellation: Option<&AtomicBool>,
    ) -> io::Result<(u64, u64)> {
        self.finish_checked(|| {
            if cancellation.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "position compaction cancelled",
                ))
            } else {
                Ok(())
            }
        })
    }

    pub(crate) fn with_budget(writer: W, budget: usize) -> Self {
        let mut encoder = Self::new(writer);
        encoder.index_limit = Some(budget / std::mem::size_of::<(u32, u64)>());
        encoder
    }

    fn append_encoded_block(&mut self, bytes: &[u8]) -> io::Result<()> {
        let count = PositionStream::block_count(bytes)
            .ok_or_else(|| invalid("invalid copied position block"))?;
        self.flush_block()?;
        self.reserve_index_entry()?;
        let offset = u32::try_from(self.written)
            .map_err(|_| invalid("position output exceeds u32 offsets"))?;
        self.index.push((offset, self.total));
        self.writer.write_all(bytes)?;
        self.written += bytes.len() as u64;
        self.total += count as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn position_ranges_preserve_values_across_short_blocks_and_copy_intact_encoding() {
        let values: Vec<_> = (0..700).map(|i| (i * 31 % 257) as u32).collect();
        let mut bytes = Vec::new();
        let mut encoder = PositionStreamEncoder::new(&mut bytes);
        encoder.push_values(&values).unwrap();
        encoder.finish().unwrap();
        for ranges in [
            std::iter::once(128..384).collect::<Vec<_>>(),
            vec![3..127, 131..275, 511..700],
        ] {
            let mut source = PositionRangeSource::open(
                FileHandle::from_bytes(OwnedBytes::new(bytes.clone())),
                4096,
                None,
            )
            .await
            .unwrap();
            let mut output = Vec::new();
            let mut writer = PositionStreamEncoder::with_budget(&mut output, 4096);
            let mut expected_values = Vec::new();
            for range in &ranges {
                source
                    .append_range(&mut writer, range.clone(), None)
                    .await
                    .unwrap();
                expected_values
                    .extend_from_slice(&values[range.start as usize..range.end as usize]);
            }
            writer.finish().unwrap();
            let actual = PositionStream::open(OwnedBytes::new(output.clone())).unwrap();
            let mut decoded = Vec::new();
            let mut scratch = Vec::new();
            for block in 0..actual.num_blocks() {
                assert!(actual.decode_block(block, &mut scratch));
                decoded.extend_from_slice(&scratch);
            }
            assert_eq!(decoded, expected_values);
            if ranges.len() == 1 {
                let mut expected = Vec::new();
                let mut encoder = PositionStreamEncoder::new(&mut expected);
                encoder.push_values(&expected_values).unwrap();
                encoder.finish().unwrap();
                assert_eq!(output, expected, "full position blocks changed bytes");
            }
        }
    }

    #[tokio::test]
    async fn position_copy_propagates_write_failure_cancellation_and_directory_budget() {
        let mut bytes = Vec::new();
        let mut encoder = PositionStreamEncoder::new(&mut bytes);
        encoder.push_values(&vec![7; 512]).unwrap();
        encoder.finish().unwrap();
        let file = FileHandle::from_bytes(OwnedBytes::new(bytes));
        assert!(
            PositionRangeSource::open(file.clone(), 1, None)
                .await
                .is_err()
        );
        struct Stop<'a>(&'a AtomicBool, bool);
        impl Write for Stop<'_> {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.1 {
                    return Err(io::Error::other("injected position write failure"));
                }
                self.0.store(true, Ordering::Release);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        for fail in [false, true] {
            let flag = AtomicBool::new(false);
            let mut source = PositionRangeSource::open(file.clone(), 4096, None)
                .await
                .unwrap();
            let mut encoder = PositionStreamEncoder::with_budget(Stop(&flag, fail), 4096);
            let error = source
                .append_range(&mut encoder, 0..512, Some(&flag))
                .await
                .unwrap_err();
            assert_eq!(
                error.kind(),
                if fail {
                    io::ErrorKind::Other
                } else {
                    io::ErrorKind::Interrupted
                }
            );
        }
        let mut source = PositionRangeSource::open(file, 4096, None).await.unwrap();
        let mut encoder = PositionStreamEncoder::with_budget(Vec::new(), 0);
        assert!(
            source
                .append_range(&mut encoder, 0..512, None)
                .await
                .is_err()
        );
    }
}
