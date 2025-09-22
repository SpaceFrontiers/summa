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
    compact: bool,
    repack_blocks: bool,
    unique_positions: bool,
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
            compact: directory::is_compact(tail.as_slice()),
            repack_blocks: false,
            unique_positions: false,
        };
        let total_len = usize::try_from(source.file.len())
            .map_err(|_| invalid("position file exceeds address space"))?;
        let (blocks, index_start, total) =
            PositionStream::parse_layout_tail(tail.as_slice(), total_len)?;
        source.unique_positions = has_unique_positions(tail.as_slice());
        if total_len - FOOTER - index_start > budget {
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
        if source.compact {
            directory::validate_with(source.index.as_slice(), blocks, index_start, total, || {
                if cancellation.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                    Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "position compaction cancelled",
                    ))
                } else {
                    Ok(())
                }
            })?;
            return Ok(source);
        }
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

    /// Explicit reorder rebuilds the stream: fill output blocks across
    /// permuted document boundaries instead of copying fragmented blocks.
    /// Compaction retains encoded-block copying by default.
    pub(crate) fn with_repacked_blocks(mut self) -> Self {
        self.repack_blocks = true;
        self
    }

    pub(crate) fn is_compact(&self) -> bool {
        self.compact
    }
    pub(crate) fn total_positions(&self) -> u64 {
        self.total
    }
    fn entry(&self, i: usize) -> (usize, u64) {
        if self.compact {
            directory::entry(self.index.as_slice(), self.blocks, i)
        } else {
            PositionStream::index_entry(self.index.as_slice(), 0, i)
        }
    }

    pub(crate) async fn append_doc<W: Write>(
        &mut self,
        writer: &mut PositionStreamEncoder<W>,
        cursor: u64,
        tf: u32,
        cancellation: Option<&AtomicBool>,
    ) -> io::Result<()> {
        let end = cursor
            .checked_add(u64::from(tf))
            .ok_or_else(|| invalid("position cursor overflow"))?;
        self.append_range(writer, cursor..end, cancellation).await
    }

    /// Copy a range spanning complete postings; callers own the doc/TF boundaries.
    /// The source certificate survives only when every appended source has it.
    pub(crate) async fn append_range<W: Write>(
        &mut self,
        writer: &mut PositionStreamEncoder<W>,
        range: std::ops::Range<u64>,
        cancellation: Option<&AtomicBool>,
    ) -> io::Result<()> {
        if range.start > range.end || range.end > self.total {
            return Err(invalid("invalid position compaction range"));
        }
        if range.is_empty() {
            return Ok(());
        }
        let unique_positions = writer.unique_positions && self.unique_positions;
        writer.unique_positions = false;
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
                if self.compact {
                    let (count, width, codec, _) = directory::parts(directory::tag(
                        self.index.as_slice(),
                        self.blocks,
                        block,
                    ))?;
                    let mut bytes = Vec::with_capacity(BLOCK_HEADER + self.raw.len());
                    bytes.extend_from_slice(&(count as u16).to_le_bytes());
                    bytes.extend_from_slice(&[width, codec]);
                    bytes.extend_from_slice(self.raw.as_slice());
                    self.raw = OwnedBytes::new(bytes);
                }
                if PositionStream::block_count(self.raw.as_slice()).map(|count| count as u64)
                    != Some(value_end - value_start)
                {
                    return Err(invalid("position block disagrees with directory"));
                }
                self.cached = Some(block);
                self.values.clear();
            }
            let take_end = range.end.min(value_end);
            if !self.repack_blocks && cursor == value_start && take_end == value_end {
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
        writer.unique_positions = unique_positions;
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

    pub(crate) fn with_budget(writer: W, budget: usize, codec: PostingCodec) -> Self {
        let mut encoder = Self::with_posting_codec(writer, codec);
        encoder.index_limit =
            Some(budget / (std::mem::size_of::<(u32, u64)>() + std::mem::size_of::<u16>()));
        encoder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn doc_aligned_position_copies_preserve_uniqueness_across_reorder_and_deletion() {
        for compact in [false, true] {
            for duplicate in [false, true] {
                let docs: Vec<Vec<u32>> = (0..300)
                    .map(|i| {
                        if duplicate && i == 13 {
                            vec![0, 0]
                        } else {
                            vec![0, i + 1]
                        }
                    })
                    .collect();
                let mut bytes = Vec::new();
                let mut encoder = PositionStreamEncoder::new(&mut bytes);
                if compact {
                    encoder = encoder.with_compact_directory();
                }
                for doc in &docs {
                    encoder.push_doc(&mut doc.clone()).unwrap();
                }
                encoder.finish().unwrap();
                for repack in [false, true] {
                    let mut source = PositionRangeSource::open(
                        FileHandle::from_bytes(OwnedBytes::new(bytes.clone())),
                        4096,
                        None,
                    )
                    .await
                    .unwrap();
                    if repack {
                        source = source.with_repacked_blocks();
                    }
                    let mut output = Vec::new();
                    let mut writer = PositionStreamEncoder::with_budget(
                        &mut output,
                        4096,
                        PostingCodec::Rounded,
                    );
                    if compact {
                        writer = writer.with_compact_directory();
                    }
                    for doc in [299u32, 0, 13, 200, 128] {
                        source
                            .append_doc(&mut writer, u64::from(doc) * 2, 2, None)
                            .await
                            .unwrap();
                    }
                    writer.finish().unwrap();
                    assert_eq!(has_unique_positions(&output), !duplicate);
                    let mut cursor = TermPositions::open(OwnedBytes::new(output))
                        .unwrap()
                        .into_cursor();
                    let mut actual = Vec::new();
                    for (i, doc) in [299, 0, 13, 200, 128].into_iter().enumerate() {
                        assert!(cursor.read_into(i as u64 * 2, 2, &mut actual));
                        assert_eq!(actual, docs[doc]);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn reordered_position_ranges_pack_identically_to_fresh_encoding() {
        let values: Vec<u32> = (0..1024).map(|i| i % 17).collect();
        // The first range leaves a short output block; the next contains
        // complete source blocks which must not fragment that output.
        let ranges = [769..1024, 0..513, 513..769];
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            for compact in [false, true] {
                let encode = |values: &[u32]| {
                    let mut bytes = Vec::new();
                    let mut writer = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
                    if compact {
                        writer = writer.with_compact_directory();
                    }
                    writer.push_values(values).unwrap();
                    writer.finish().unwrap();
                    bytes
                };
                let mut source = PositionRangeSource::open(
                    FileHandle::from_bytes(OwnedBytes::new(encode(&values))),
                    4096,
                    None,
                )
                .await
                .unwrap()
                .with_repacked_blocks();
                let mut output = Vec::new();
                let mut writer = PositionStreamEncoder::with_budget(&mut output, 4096, codec);
                if compact {
                    writer = writer.with_compact_directory();
                }
                let mut expected = Vec::new();
                for range in &ranges {
                    source
                        .append_range(&mut writer, range.clone(), None)
                        .await
                        .unwrap();
                    expected.extend_from_slice(&values[range.start as usize..range.end as usize]);
                }
                writer.finish().unwrap();
                assert_eq!(
                    output,
                    encode(&expected),
                    "codec={codec:?}, compact={compact}"
                );
            }
        }
    }

    #[tokio::test]
    async fn compact_position_ranges_copy_payloads_across_formats_and_respect_budgets() {
        let values: Vec<u32> = (0..700).map(|i| i % 251).collect();
        for input_compact in [false, true] {
            for output_compact in [false, true] {
                let mut bytes = Vec::new();
                let mut encoder = PositionStreamEncoder::new(&mut bytes);
                if input_compact {
                    encoder = encoder.with_compact_directory();
                }
                encoder.push_values(&values).unwrap();
                encoder.finish().unwrap();
                let source_stream = PositionStream::open(OwnedBytes::new(bytes.clone())).unwrap();
                let file = FileHandle::from_bytes(OwnedBytes::new(bytes.clone()));
                let cancelled = AtomicBool::new(true);
                assert!(
                    PositionRangeSource::open(file.clone(), 4096, Some(&cancelled))
                        .await
                        .is_err()
                );
                assert!(
                    PositionRangeSource::open(file.clone(), 1, None)
                        .await
                        .is_err()
                );
                let mut source = PositionRangeSource::open(file, 4096, None).await.unwrap();
                let mut output = Vec::new();
                let mut writer =
                    PositionStreamEncoder::with_budget(&mut output, 4096, PostingCodec::Rounded);
                if output_compact {
                    writer = writer.with_compact_directory();
                }
                source
                    .append_range(&mut writer, 128..384, None)
                    .await
                    .unwrap();
                source
                    .append_range(&mut writer, 511..699, None)
                    .await
                    .unwrap();
                writer.finish().unwrap();
                let result = PositionStream::open(OwnedBytes::new(output.clone())).unwrap();
                let mut actual = Vec::new();
                let mut block = Vec::new();
                for i in 0..result.num_blocks() {
                    assert!(result.decode_block(i, &mut block));
                    actual.extend_from_slice(&block);
                }
                assert_eq!(actual, [&values[128..384], &values[511..699]].concat());
                for i in 0..2 {
                    let (a, b, _) = source_stream.block_range(i + 1).unwrap();
                    let (c, d, _) = result.block_range(i).unwrap();
                    assert_eq!(
                        &bytes[a + if input_compact { 0 } else { 4 }..b],
                        &output[c + if output_compact { 0 } else { 4 }..d]
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn position_ranges_preserve_values_across_short_blocks_and_copy_intact_encoding() {
        for codec in [PostingCodec::Rounded, PostingCodec::Simd4x] {
            let values: Vec<_> = (0..700).map(|i| (i * 31 % 257) as u32).collect();
            let mut bytes = Vec::new();
            let mut encoder = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
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
                let mut writer = PositionStreamEncoder::with_budget(
                    &mut output,
                    4096,
                    if codec == PostingCodec::Rounded {
                        PostingCodec::Simd4x
                    } else {
                        PostingCodec::Rounded
                    },
                );
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
                    let mut encoder =
                        PositionStreamEncoder::with_posting_codec(&mut expected, codec);
                    encoder.push_values(&expected_values).unwrap();
                    encoder.finish().unwrap();
                    assert_eq!(output, expected, "full position blocks changed bytes");
                }
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
            let mut encoder =
                PositionStreamEncoder::with_budget(Stop(&flag, fail), 4096, PostingCodec::Rounded);
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
        let mut encoder = PositionStreamEncoder::with_budget(Vec::new(), 0, PostingCodec::Rounded);
        assert!(
            source
                .append_range(&mut encoder, 0..512, None)
                .await
                .is_err()
        );
    }
}
