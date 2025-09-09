//! Bounded encoded-block access and streaming output for row compaction.
use super::*;
use crate::directories::FileHandle;
use std::sync::atomic::{AtomicBool, Ordering};

fn check(cancellation: Option<&AtomicBool>) -> io::Result<()> {
    if cancellation.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "posting compaction cancelled",
        ));
    }
    Ok(())
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Retains only the encoded directory. Payload blocks are read on demand.
pub(crate) struct PostingBlockSource {
    file: FileHandle,
    footer: Footer,
    index: OwnedBytes,
}

impl PostingBlockSource {
    pub(crate) async fn open(
        file: FileHandle,
        budget: usize,
        cancellation: Option<&AtomicBool>,
    ) -> io::Result<Self> {
        check(cancellation)?;
        let len = usize::try_from(file.len()).map_err(|_| invalid("posting file is too large"))?;
        let tail = file
            .read_bytes_range(file.len().saturating_sub(FOOTER_V2_SIZE as u64)..file.len())
            .await?;
        let footer = Footer::parse_tail(tail.as_slice(), len)?;
        let index_len = footer.cursors_end() - footer.stream_len;
        if index_len > budget {
            return Err(invalid(
                "posting directory exceeds compaction scratch budget",
            ));
        }
        let index = file
            .read_bytes_range(footer.stream_len as u64..footer.cursors_end() as u64)
            .await?;
        let source = Self {
            file,
            footer,
            index,
        };
        let mut previous_last = None;
        let mut previous_offset = None;
        if source.footer.l1_count != source.len().div_ceil(L1_INTERVAL)
            || (source.len() == 0) != (source.doc_count() == 0)
            || u64::from(source.doc_count()) > source.len() as u64 * BLOCK_SIZE as u64
        {
            return Err(invalid("invalid posting source counts"));
        }
        for i in 0..source.len() {
            if i.is_multiple_of(4096) {
                check(cancellation)?;
            }
            let (first, last, offset, _) = source.entry(i);
            if (i == 0 && offset != 0)
                || first > last
                || last == TERMINATED
                || previous_last.is_some_and(|last| first <= last)
                || previous_offset.is_some_and(|old| offset <= old)
                || offset as usize >= source.footer.stream_len
            {
                return Err(invalid("invalid compacted source posting directory"));
            }
            source.position_span(i)?;
            previous_last = Some(last);
            previous_offset = Some(offset);
        }
        Ok(source)
    }

    fn entry(&self, i: usize) -> (u32, u32, u32, u32) {
        read_l0(self.index.as_slice(), i)
    }
    pub(crate) fn len(&self) -> usize {
        self.footer.l0_count
    }
    pub(crate) fn has_positions(&self) -> bool {
        self.footer.has_cursors
    }
    pub(crate) fn doc_count(&self) -> u32 {
        self.footer.doc_count
    }
    pub(crate) fn bounds(&self, i: usize) -> (u32, u32) {
        let (first, last, _, _) = self.entry(i);
        (first, last)
    }
    pub(crate) fn position_span(&self, i: usize) -> io::Result<std::ops::Range<u64>> {
        if !self.footer.has_cursors {
            return Ok(0..0);
        }
        let base = self.footer.l1_bounds_end() - self.footer.stream_len;
        let cursor = |j: usize| {
            let at = base + j * CURSOR_SIZE;
            u64::from_le_bytes(self.index[at..at + CURSOR_SIZE].try_into().unwrap())
        };
        let start = cursor(i);
        let end = if i + 1 == self.len() {
            self.footer.total_positions
        } else {
            cursor(i + 1)
        };
        if start > end || end > self.footer.total_positions || (i == 0 && start != 0) {
            return Err(invalid("invalid source posting position cursors"));
        }
        Ok(start..end)
    }

    /// Reuse the ordinary block decoder with offsets localized to this block.
    pub(crate) async fn read_block(&self, i: usize) -> io::Result<BlockPostingList> {
        let (first, last, offset, bounds) = self.entry(i);
        let end = if i + 1 == self.len() {
            self.footer.stream_len
        } else {
            self.entry(i + 1).2 as usize
        };
        if end < offset as usize || end - (offset as usize) > BLOCK_SIZE * 32 + 64 {
            return Err(invalid("posting block exceeds encoded block bound"));
        }
        let stream = self
            .file
            .read_bytes_range(u64::from(offset)..end as u64)
            .await?;
        if stream.len() < 8 {
            return Err(invalid("truncated posting block"));
        }
        let count = u16::from_le_bytes(stream[..2].try_into().unwrap()) as u32;
        if count == 0
            || count as usize > BLOCK_SIZE
            || count > last - first + 1
            || u32::from_le_bytes(stream[2..6].try_into().unwrap()) != first
        {
            return Err(invalid("posting block header disagrees with directory"));
        }
        let (codec, bits) = PostingCodec::from_header_byte(stream[6])?;
        let tf_bits = stream[7];
        if tf_bits > 32 {
            return Err(invalid("invalid posting frequency width"));
        }
        let payload_len = |payload: &[u8], count: usize, bits: u8| -> io::Result<usize> {
            let len = match codec {
                PostingCodec::Rounded => {
                    if !matches!(bits, 0 | 8 | 16 | 32) {
                        return Err(invalid("invalid rounded posting width"));
                    }
                    count * simd::RoundedBitWidth::from_u8(bits).bytes_per_value()
                }
                PostingCodec::Packed => packed_bytes(count, bits),
                PostingCodec::Pfor => pfor_payload_len(payload, count, bits)?,
            };
            if len > payload.len() {
                return Err(invalid("truncated posting payload"));
            }
            Ok(len)
        };
        let gaps = if count > 1 {
            payload_len(&stream[8..], count as usize - 1, bits)?
        } else {
            0
        };
        let tfs = payload_len(&stream[8 + gaps..], count as usize, tf_bits)?;
        if 8 + gaps + tfs != stream.len() {
            return Err(invalid("invalid posting payload length"));
        }
        let mut l0 = Vec::with_capacity(L0_SIZE);
        write_l0(&mut l0, first, last, 0, bounds);
        let span = self.position_span(i)?;
        Ok(BlockPostingList {
            stream,
            l0_bytes: OwnedBytes::new(l0),
            l0_count: 1,
            l1_docs: vec![last],
            l1_bounds: Vec::new(),
            doc_count: count,
            max_tf: self.footer.max_tf,
            pos_cursors: self.footer.has_cursors.then(|| OwnedBytes::new(vec![0; 8])),
            total_positions: span.end - span.start,
            len_bounds: self.footer.len_bounds,
            min_len: self.footer.min_len,
        })
    }
}

/// Copies encoded blocks and rebuilds only bounded skip/cursor metadata.
pub(crate) struct PostingStreamWriter<W: Write> {
    writer: W,
    l0: Vec<u8>,
    cursors: Vec<u8>,
    with_positions: bool,
    blocks_limit: usize,
    written: u64,
    docs: u32,
    max_tf: u32,
    min_len: u32,
    positions: u64,
    pending: PostingList,
    min_pending_length: u32,
    codec: PostingCodec,
}

impl<W: Write> PostingStreamWriter<W> {
    pub(crate) fn new(
        writer: W,
        max_blocks: usize,
        with_positions: bool,
        codec: PostingCodec,
        budget: usize,
    ) -> io::Result<Self> {
        if max_blocks.saturating_mul(32) > budget {
            return Err(invalid(
                "posting output directory exceeds compaction scratch budget",
            ));
        }
        Ok(Self {
            writer,
            l0: Vec::with_capacity(max_blocks * L0_SIZE),
            cursors: Vec::with_capacity(if with_positions {
                max_blocks * CURSOR_SIZE
            } else {
                0
            }),
            with_positions,
            blocks_limit: max_blocks,
            written: 0,
            docs: 0,
            max_tf: 0,
            min_len: u32::MAX,
            positions: 0,
            pending: PostingList::with_capacity(BLOCK_SIZE),
            min_pending_length: u32::MAX,
            codec,
        })
    }

    /// Accumulate mixed-block survivors so scattered deletions do not leave
    /// a directory entry and header for every handful of surviving postings.
    pub(crate) fn push(&mut self, doc: u32, tf: u32, length: u32) -> io::Result<()> {
        self.pending.push(doc, tf);
        self.min_pending_length = self.min_pending_length.min(length.max(1));
        if self.pending.len() == BLOCK_SIZE {
            self.flush_pending()?;
        }
        Ok(())
    }

    fn flush_pending(&mut self) -> io::Result<()> {
        let Some(first) = self.pending.iter().next().map(|posting| posting.doc_id) else {
            return Ok(());
        };
        // One block: its exact minimum is sufficient for the bound encoder.
        let length = |_: u32| self.min_pending_length;
        let block = BlockPostingList::from_posting_list_with_options(
            &self.pending,
            self.with_positions,
            Some(&length),
            self.codec,
        )?;
        self.append_block(&block, first)?;
        self.pending.postings.clear();
        self.min_pending_length = u32::MAX;
        Ok(())
    }

    pub(crate) fn append(&mut self, block: &BlockPostingList, first: u32) -> io::Result<()> {
        self.flush_pending()?;
        self.append_block(block, first)
    }

    fn append_block(&mut self, block: &BlockPostingList, first: u32) -> io::Result<()> {
        if block.num_blocks() != 1
            || block.has_position_cursors() != self.with_positions
            || self.l0.len() / L0_SIZE == self.blocks_limit
        {
            return Err(invalid("invalid streaming posting block admission"));
        }
        let (old_first, old_last, _, bounds) = block.read_l0_entry(0);
        let last = first
            .checked_add(old_last - old_first)
            .ok_or_else(|| invalid("posting address overflow"))?;
        if self.written > u32::MAX as u64
            || last == TERMINATED
            || (!self.l0.is_empty() && first <= read_l0(&self.l0, self.l0.len() / L0_SIZE - 1).1)
        {
            return Err(invalid("invalid streaming posting order or size"));
        }
        let (max_tf, min_len) = unpack_bounds(bounds, block.len_bounds);
        write_l0(
            &mut self.l0,
            first,
            last,
            self.written as u32,
            pack_bounds(max_tf, min_len.unwrap_or(1)),
        );
        if self.with_positions {
            self.cursors
                .extend_from_slice(&self.positions.to_le_bytes());
        }
        let mut header = [0u8; 8];
        header.copy_from_slice(&block.stream[..8]);
        header[2..6].copy_from_slice(&first.to_le_bytes());
        self.writer.write_all(&header)?;
        self.writer.write_all(&block.stream[8..])?;
        self.written += block.stream.len() as u64;
        self.docs = self
            .docs
            .checked_add(block.doc_count())
            .ok_or_else(|| invalid("posting count overflow"))?;
        self.positions = self
            .positions
            .checked_add(block.total_positions())
            .ok_or_else(|| invalid("position count overflow"))?;
        self.max_tf = self.max_tf.max(block.max_tf());
        self.min_len = self.min_len.min(min_len.unwrap_or(1));
        Ok(())
    }

    pub(crate) fn doc_count(&self) -> u32 {
        self.docs + self.pending.doc_count()
    }
    pub(crate) fn total_positions(&self) -> u64 {
        self.positions
            + if self.with_positions {
                self.pending
                    .iter()
                    .map(|posting| u64::from(posting.term_freq))
                    .sum::<u64>()
            } else {
                0
            }
    }

    pub(crate) fn finish(mut self, cancellation: Option<&AtomicBool>) -> io::Result<(u32, u64)> {
        check(cancellation)?;
        self.flush_pending()?;
        let blocks = self.l0.len() / L0_SIZE;
        let groups = blocks.div_ceil(L1_INTERVAL);
        for chunk in self.l0.chunks(64 * 1024) {
            check(cancellation)?;
            self.writer.write_all(chunk)?;
        }
        for group in 0..groups {
            if group.is_multiple_of(4096) {
                check(cancellation)?;
            }
            let last = ((group + 1) * L1_INTERVAL).min(blocks) - 1;
            self.writer
                .write_u32::<LittleEndian>(read_l0(&self.l0, last).1)?;
        }
        for bounds in group_bounds_from_l0(&self.l0, blocks) {
            self.writer.write_u32::<LittleEndian>(bounds)?;
        }
        for chunk in self.cursors.chunks(64 * 1024) {
            check(cancellation)?;
            self.writer.write_all(chunk)?;
        }
        BlockPostingList::write_footer(
            &mut self.writer,
            self.written,
            blocks,
            groups,
            self.docs,
            self.max_tf,
            self.positions,
            self.with_positions,
            Some(if blocks == 0 { 1 } else { self.min_len }),
            true,
        )?;
        Ok((
            self.docs,
            self.written
                + self.l0.len() as u64
                + groups as u64 * 8
                + self.cursors.len() as u64
                + FOOTER_V2_SIZE as u64,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[tokio::test]
    async fn rebased_posting_blocks_copy_payload_bytes_for_every_codec_and_read_only_requested_blocks()
     {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
        ] {
            let mut list = PostingList::new();
            for doc in 0..4096 {
                list.push(doc * 11, doc % 19 + 1);
            }
            let original =
                BlockPostingList::from_posting_list_with_options(&list, true, Some(&|_| 37), codec)
                    .unwrap();
            let mut bytes = Vec::new();
            original.serialize(&mut bytes).unwrap();
            let bytes = OwnedBytes::new(bytes);
            let reads = Arc::new(Mutex::new(Vec::new()));
            let captured = reads.clone();
            let file = FileHandle::lazy(
                bytes.len() as u64,
                Arc::new(move |range| {
                    captured.lock().unwrap().push(range.clone());
                    let result = bytes.slice(range.start as usize..range.end as usize);
                    Box::pin(async move { Ok(result) })
                }),
            );
            let source = PostingBlockSource::open(file, 4096, None).await.unwrap();
            assert_eq!(
                reads.lock().unwrap().len(),
                2,
                "opening must read only footer and directory"
            );
            let block = source.read_block(7).await.unwrap();
            assert_eq!(reads.lock().unwrap().len(), 3);
            let first = block.read_l0_entry(0).0 - 200;
            let mut bytes = Vec::new();
            let mut output = PostingStreamWriter::new(&mut bytes, 1, true, codec, 32).unwrap();
            output.append(&block, first).unwrap();
            output.finish(None).unwrap();
            let copied = BlockPostingList::deserialize(&bytes).unwrap();
            assert_eq!(&copied.stream[8..], &block.stream[8..]);
            let mut ids = Vec::new();
            let mut tfs = Vec::new();
            assert!(copied.decode_block_into(0, &mut ids, &mut tfs));
            assert_eq!(
                ids,
                (7 * 128..8 * 128)
                    .map(|doc| doc * 11 - 200)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                tfs,
                (7 * 128..8 * 128)
                    .map(|doc| doc % 19 + 1)
                    .collect::<Vec<_>>()
            );
            assert_eq!(copied.total_positions(), block.total_positions());
        }
    }

    #[tokio::test]
    async fn partial_posting_reads_reject_truncated_payloads_and_cancelled_directory_scans() {
        let mut list = PostingList::new();
        for doc in 0..128 {
            list.push(doc * 3, 256);
        }
        let block = BlockPostingList::from_posting_list_with_options(
            &list,
            false,
            None,
            PostingCodec::Rounded,
        )
        .unwrap();
        let mut bytes = Vec::new();
        block.serialize(&mut bytes).unwrap();
        let flag = AtomicBool::new(true);
        assert!(
            PostingBlockSource::open(
                FileHandle::from_bytes(OwnedBytes::new(bytes.clone())),
                4096,
                Some(&flag)
            )
            .await
            .is_err()
        );
        bytes[7] = 32; // Declared TF payload is longer than the encoded block.
        let source =
            PostingBlockSource::open(FileHandle::from_bytes(OwnedBytes::new(bytes)), 4096, None)
                .await
                .unwrap();
        assert!(source.read_block(0).await.is_err());
        let mut corrupt = vec![0; FOOTER_SIZE];
        corrupt[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(Footer::parse(&corrupt).is_err());
    }
}
