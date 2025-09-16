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
    impacts: Option<ImpactTable>,
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
        if tail.len() != len.min(FOOTER_V2_SIZE) {
            return Err(invalid(
                "posting footer read returned an incorrect byte count",
            ));
        }
        let footer = Footer::parse_tail(tail.as_slice(), len)?;
        let index_end = if footer.impact_bounds {
            len - FOOTER_V2_SIZE
        } else {
            footer.ratios_end()
        };
        let index_len = index_end - footer.stream_len;
        if index_len > budget {
            return Err(invalid(
                "posting directory exceeds compaction scratch budget",
            ));
        }
        let index = file
            .read_bytes_range(footer.stream_len as u64..index_end as u64)
            .await?;
        if index.len() != index_len {
            return Err(invalid(
                "posting directory read returned an incorrect byte count",
            ));
        }
        if footer.ratio_bounds {
            validate_ratios(
                &index[footer.cursors_end() - footer.stream_len
                    ..footer.ratios_end() - footer.stream_len],
            )?;
        }
        let impacts = if footer.impact_bounds {
            let bytes = index.slice(footer.ratios_end() - footer.stream_len..index.len());
            ImpactTable::validate_with(&bytes, footer.impact_record_count(), || {
                check(cancellation)
            })?;
            Some(ImpactTable::from_validated(
                bytes,
                footer.impact_record_count(),
            ))
        } else {
            None
        };
        let source = Self {
            impacts,
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
                || previous_offset
                    .is_some_and(|old| offset < old || (!footer.compact_headers && offset == old))
                || offset as usize > source.footer.stream_len
                || (!footer.compact_headers && offset as usize == source.footer.stream_len)
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
    pub(crate) fn is_compact(&self) -> bool {
        self.footer.compact_headers
    }
    pub(crate) fn len(&self) -> usize {
        self.footer.l0_count
    }
    pub(crate) fn has_ratio_bounds(&self) -> bool {
        self.footer.ratio_bounds
    }
    pub(crate) fn has_impact_bounds(&self) -> bool {
        self.impacts.is_some()
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
            let size = self.footer.cursor_size();
            let at = base + j * size;
            if self.footer.short_cursors {
                u64::from(u32::from_le_bytes(
                    self.index[at..at + size].try_into().unwrap(),
                ))
            } else {
                u64::from_le_bytes(self.index[at..at + size].try_into().unwrap())
            }
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
        if stream.len() != end - offset as usize {
            return Err(invalid(
                "posting block read returned an incorrect byte count",
            ));
        }
        let stream = if self.footer.compact_headers {
            let at = self.footer.l0_count * L0_SIZE + i * 4;
            let descriptor = &self.index[at..at + 4];
            validation::validate_descriptor(descriptor, first, last, stream.len())?;
            let mut bytes = Vec::with_capacity(8 + stream.len());
            bytes.extend_from_slice(&descriptor[..2]);
            bytes.extend_from_slice(&first.to_le_bytes());
            bytes.extend_from_slice(&descriptor[2..]);
            bytes.extend_from_slice(&stream);
            OwnedBytes::new(bytes)
        } else {
            stream
        };
        let count = validation::validate_block(&stream, first, last)? as u32;
        let mut l0 = Vec::with_capacity(L0_SIZE);
        write_l0(&mut l0, first, last, 0, bounds);
        let span = self.position_span(i)?;
        let impacts = if let Some(table) = &self.impacts {
            let mut builder = ImpactBuilder::new(1)?;
            builder.append(table.record(i).unwrap())?;
            builder.finish()
        } else {
            None
        };
        Ok(BlockPostingList {
            compact_headers: false,
            short_cursors: false,
            verify_content: true,
            content_error: None,
            impacts,
            stream,
            l0_bytes: OwnedBytes::new(l0),
            l0_count: 1,
            l1_docs: vec![last],
            l1_bounds: Vec::new(),
            ratios: self.footer.ratio_bounds.then(|| {
                let ratio = read_ratio(
                    &self.index[self.footer.cursors_end() - self.footer.stream_len..],
                    i,
                );
                let mut bytes = ratio.to_le_bytes().to_vec();
                bytes.extend_from_slice(&ratio.to_le_bytes());
                OwnedBytes::new(bytes)
            }),
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
    headers: Option<Vec<u8>>,
    writer: W,
    l0: Vec<u8>,
    cursors: Vec<u8>,
    ratios: Vec<u8>,
    has_ratios: bool,
    impacts: Option<ImpactBuilder>,
    pending_lengths: [u32; BLOCK_SIZE],
    pending_ratio: f32,
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
    directory_budget: usize,
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
            headers: None,
            l0: Vec::with_capacity(max_blocks * L0_SIZE),
            cursors: Vec::with_capacity(if with_positions {
                max_blocks * CURSOR_SIZE
            } else {
                0
            }),
            ratios: Vec::new(),
            has_ratios: false,
            impacts: None,
            pending_lengths: [0; BLOCK_SIZE],
            pending_ratio: f32::INFINITY,
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
            directory_budget: budget,
        })
    }

    pub(crate) fn enable_compact_headers(&mut self) -> io::Result<()> {
        if !self.l0.is_empty() || !self.pending.is_empty() || self.codec == PostingCodec::Pfor {
            return Err(invalid(
                "select compact headers before appending fixed-width blocks",
            ));
        }
        self.headers = Some(Vec::with_capacity(self.blocks_limit * 4));
        Ok(())
    }

    // Includes transient one-block envelope output and bounded construction
    // scratch. Payload/document scratch remains charged by the compaction owner.
    const IMPACT_SCRATCH_BYTES: usize = 4096;

    fn ratio_directory_bytes(&self) -> usize {
        let per_block = L0_SIZE
            + 4
            + if self.headers.is_some() { 4 } else { 0 }
            + if self.with_positions { CURSOR_SIZE } else { 0 };
        self.blocks_limit
            .saturating_mul(per_block)
            .saturating_add(self.blocks_limit.div_ceil(L1_INTERVAL).saturating_mul(12))
    }

    pub(crate) fn enable_impact_bounds(&mut self) -> io::Result<()> {
        if self.impacts.is_some() {
            return Ok(());
        }
        let required = self
            .ratio_directory_bytes()
            .saturating_add(
                ImpactBuilder::budget_with_groups(self.blocks_limit).unwrap_or(usize::MAX),
            )
            .saturating_add(Self::IMPACT_SCRATCH_BYTES);
        if required > self.directory_budget {
            return Err(invalid(
                "posting impact directory exceeds compaction scratch budget",
            ));
        }
        self.enable_ratio_bounds()?;
        let mut impacts = ImpactBuilder::with_groups(self.blocks_limit)?;
        for _ in 0..self.l0.len() / L0_SIZE {
            impacts.append(&[])?;
        }
        self.impacts = Some(impacts);
        Ok(())
    }

    pub(crate) fn enable_ratio_bounds(&mut self) -> io::Result<()> {
        let per_block = L0_SIZE
            + 4
            + if self.headers.is_some() { 4 } else { 0 }
            + if self.with_positions { CURSOR_SIZE } else { 0 };
        let required = self
            .blocks_limit
            .saturating_mul(per_block)
            .saturating_add(self.blocks_limit.div_ceil(L1_INTERVAL).saturating_mul(12));
        if required > self.directory_budget {
            return Err(invalid(
                "posting ratio directory exceeds compaction scratch budget",
            ));
        }
        if !self.has_ratios {
            // Pending legacy values have no ratio. Zero stays conservative.
            if !self.pending.is_empty() {
                self.pending_ratio = 0.0;
            }
            self.ratios
                .reserve_exact((self.blocks_limit + self.blocks_limit.div_ceil(L1_INTERVAL)) * 4);
            self.has_ratios = true;
        }
        Ok(())
    }

    /// Accumulate mixed-block survivors so scattered deletions do not leave
    /// a directory entry and header for every handful of surviving postings.
    pub(crate) fn push(&mut self, doc: u32, tf: u32, length: u32) -> io::Result<()> {
        self.pending_lengths[self.pending.len()] = if length == 0 { tf } else { length };
        self.pending.push(doc, tf);
        if self.has_ratios {
            self.pending_ratio = self
                .pending_ratio
                .min(lower_length_ratio(length.max(1), tf));
        }
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
        let mut block = BlockPostingList::from_posting_list_with_options(
            &self.pending,
            self.with_positions,
            Some(&length),
            self.codec,
        )?;
        if self.has_ratios {
            let mut ratios = self.pending_ratio.to_le_bytes().to_vec();
            ratios.extend_from_slice(&self.pending_ratio.to_le_bytes());
            block.ratios = Some(OwnedBytes::new(ratios));
        }
        if self.impacts.is_some() {
            let mut points = [(0u32, 0u32); BLOCK_SIZE];
            for ((point, posting), &length) in points
                .iter_mut()
                .zip(self.pending.iter())
                .zip(&self.pending_lengths)
            {
                *point = (posting.term_freq, length);
            }
            let mut builder = ImpactBuilder::new(1)?;
            builder.append_points(&mut points[..self.pending.len()])?;
            block.impacts = builder.finish();
        }
        self.append_block(&block, first)?;
        self.pending.postings.clear();
        self.min_pending_length = u32::MAX;
        self.pending_ratio = f32::INFINITY;
        Ok(())
    }

    pub(crate) fn append(&mut self, block: &BlockPostingList, first: u32) -> io::Result<()> {
        if block.has_impact_bounds() {
            self.enable_impact_bounds()?;
        } else if block.has_ratio_bounds() {
            self.enable_ratio_bounds()?;
        }
        self.flush_pending()?;
        self.append_block(block, first)
    }

    fn append_block(&mut self, block: &BlockPostingList, first: u32) -> io::Result<()> {
        if block.num_blocks() != 1
            || (self.headers.is_some() && block.block_codec(0) == Some(PostingCodec::Pfor))
            || block.has_position_cursors() != self.with_positions
            || self.l0.len() / L0_SIZE == self.blocks_limit
        {
            return Err(invalid("invalid streaming posting block admission"));
        }
        if block.has_impact_bounds() {
            self.enable_impact_bounds()?;
        } else if block.has_ratio_bounds() {
            self.enable_ratio_bounds()?;
        }
        if self.has_ratios {
            self.ratios.resize(self.l0.len() / L0_SIZE * 4, 0);
            self.ratios
                .extend_from_slice(&block.block_length_ratio(0).to_le_bytes());
        }
        if let Some(impacts) = &mut self.impacts {
            impacts.append(
                block
                    .impacts
                    .as_ref()
                    .and_then(|t| t.record(0))
                    .unwrap_or(&[]),
            )?;
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
        header.copy_from_slice(&block.block_header(0));
        header[2..6].copy_from_slice(&first.to_le_bytes());
        if let Some(headers) = &mut self.headers {
            headers.extend_from_slice(&header[..2]);
            headers.extend_from_slice(&header[6..]);
        } else {
            self.writer.write_all(&header)?;
        }
        self.writer.write_all(block.block_payload(0))?;
        self.written +=
            (if self.headers.is_some() { 0 } else { 8 } + block.block_payload(0).len()) as u64;
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
        if let Some(headers) = &self.headers {
            for chunk in headers.chunks(64 * 1024) {
                check(cancellation)?;
                self.writer.write_all(chunk)?;
            }
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
        let short_cursors =
            self.headers.is_some() && self.with_positions && self.positions <= u32::MAX as u64;
        if short_cursors {
            for (i, bytes) in self.cursors.chunks_exact(8).enumerate() {
                if i.is_multiple_of(4096) {
                    check(cancellation)?;
                }
                self.writer.write_u32::<LittleEndian>(
                    u64::from_le_bytes(bytes.try_into().unwrap()) as u32,
                )?;
            }
        } else {
            for chunk in self.cursors.chunks(64 * 1024) {
                check(cancellation)?;
                self.writer.write_all(chunk)?;
            }
        }
        if self.has_ratios {
            append_ratio_groups(&mut self.ratios, blocks);
            for chunk in self.ratios.chunks(64 * 1024) {
                check(cancellation)?;
                self.writer.write_all(chunk)?;
            }
        }
        let impacts = if let Some(mut impacts) = self.impacts.take() {
            impacts.append_groups_with(blocks, |_| {
                check(cancellation)?;
                Ok(None)
            })?;
            impacts.finish()
        } else {
            None
        };
        if let Some(impacts) = &impacts {
            for chunk in impacts.bytes().chunks(64 * 1024) {
                check(cancellation)?;
                self.writer.write_all(chunk)?;
            }
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
            self.has_ratios,
            impacts.is_some(),
            impacts.is_some(),
            if self.headers.is_some() {
                FLAG_COMPACT_HEADERS
            } else {
                0
            } | if short_cursors { FLAG_SHORT_CURSORS } else { 0 },
        )?;
        Ok((
            self.docs,
            self.written
                + self.l0.len() as u64
                + groups as u64 * 8
                + self.headers.as_ref().map_or(0, Vec::len) as u64
                + (if short_cursors {
                    self.cursors.len() / 2
                } else {
                    self.cursors.len()
                }) as u64
                + if self.has_ratios {
                    self.ratios.len() as u64
                } else {
                    0
                }
                + impacts.as_ref().map_or(0, |t| t.bytes().len() as u64)
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
            PostingCodec::Simd4x,
        ] {
            let mut list = PostingList::new();
            for doc in 0..4096 {
                list.push(doc * 11, doc % 19 + 1);
            }
            let original = BlockPostingList::from_posting_list_with_ratio_bounds(
                &list,
                true,
                Some(&|_| 37),
                codec,
            )
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
            let mut output = PostingStreamWriter::new(&mut bytes, 1, true, codec, 40).unwrap();
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
            assert_eq!(copied.block_length_ratio(0), block.block_length_ratio(0));
            assert!(copied.has_ratio_bounds());
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
    #[test]
    fn ratio_directory_budget_is_checked_before_copying_any_payload() {
        let mut postings = PostingList::new();
        postings.push(1, 3);
        let block = BlockPostingList::from_posting_list_with_ratio_bounds(
            &postings,
            true,
            Some(&|_| 10),
            PostingCodec::Rounded,
        )
        .unwrap();
        let mut bytes = Vec::new();
        let mut output =
            PostingStreamWriter::new(&mut bytes, 1, true, PostingCodec::Rounded, 32).unwrap();
        assert!(output.append(&block, 1).is_err());
        drop(output);
        assert!(bytes.is_empty());
        let mut output =
            PostingStreamWriter::new(&mut bytes, 1, true, PostingCodec::Rounded, 40).unwrap();
        output.enable_ratio_bounds().unwrap();
        output.push(1, 3, 10).unwrap();
        let cancelled = AtomicBool::new(true);
        assert_eq!(
            output.finish(Some(&cancelled)).unwrap_err().kind(),
            io::ErrorKind::Interrupted
        );
        assert!(bytes.is_empty());
    }
    #[tokio::test]
    async fn impact_compaction_admits_metadata_before_io_and_copies_complete_records() {
        let mut postings = PostingList::new();
        for doc in 0..512 {
            postings.push(doc * 3, doc % 8 + 1);
        }
        let original = BlockPostingList::from_posting_list_with_impact_bounds(
            &postings,
            true,
            Some(&|doc| (doc / 3 % 8 + 1).pow(2)),
            PostingCodec::Rounded,
        )
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
        assert!(
            PostingBlockSource::open(file.clone(), 16, None)
                .await
                .is_err()
        );
        assert_eq!(
            reads.lock().unwrap().len(),
            1,
            "budget failure must precede metadata IO"
        );
        let source = PostingBlockSource::open(file, 16 * 1024, None)
            .await
            .unwrap();
        assert!(source.has_impact_bounds());
        let block = source.read_block(1).await.unwrap();
        assert_eq!(
            block.impacts.as_ref().unwrap().record(0),
            original.impacts.as_ref().unwrap().record(1)
        );
        let mut output = Vec::new();
        let mut writer =
            PostingStreamWriter::new(&mut output, 1, true, PostingCodec::Rounded, 40).unwrap();
        assert!(writer.append(&block, 7).is_err());
        drop(writer);
        assert!(output.is_empty());
        let mut writer =
            PostingStreamWriter::new(&mut output, 4, true, PostingCodec::Rounded, 16 * 1024)
                .unwrap();
        writer.enable_impact_bounds().unwrap();
        writer.append(&block, 7).unwrap();
        for doc in 1000..1008 {
            let tf = doc % 8 + 1;
            writer.push(doc, tf, tf * tf).unwrap();
        }
        let (count, size) = writer.finish(None).unwrap();
        assert_eq!(count, 136);
        assert_eq!(size as usize, output.len());
        let rebuilt = BlockPostingList::deserialize(&output).unwrap();
        assert_eq!(rebuilt.num_blocks(), 2);
        assert_eq!(
            rebuilt.impacts.as_ref().unwrap().record(0),
            block.impacts.as_ref().unwrap().record(0)
        );
        assert_eq!(rebuilt.block_impact_point_count(1), Some(8));
        assert_eq!(rebuilt.block_impact_minimum(1, 1.0, 1.0), Some(2.0));
        assert_eq!(&rebuilt.stream[8..block.stream.len()], &block.stream[8..]);
    }

    #[test]
    fn cancelled_or_failed_impact_output_never_emits_a_valid_footer() {
        struct CancelOutput<'a> {
            bytes: &'a mut Vec<u8>,
            cancel: &'a AtomicBool,
        }
        impl Write for CancelOutput<'_> {
            fn write(&mut self, b: &[u8]) -> io::Result<usize> {
                self.bytes.extend_from_slice(b);
                self.cancel.store(true, Ordering::Release);
                Ok(b.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let flag = AtomicBool::new(false);
        let mut bytes = Vec::new();
        let mut writer = PostingStreamWriter::new(
            CancelOutput {
                bytes: &mut bytes,
                cancel: &flag,
            },
            1,
            false,
            PostingCodec::Rounded,
            16 * 1024,
        )
        .unwrap();
        writer.enable_impact_bounds().unwrap();
        writer.push(1, 3, 11).unwrap();
        assert_eq!(
            writer.finish(Some(&flag)).unwrap_err().kind(),
            io::ErrorKind::Interrupted
        );
        assert!(BlockPostingList::deserialize(&bytes).is_err());
        // Fixed-capacity writer accepts the block bytes but fails in metadata.
        let mut bytes = [0u8; 20];
        let mut remaining = bytes.as_mut_slice();
        let mut writer =
            PostingStreamWriter::new(&mut remaining, 1, false, PostingCodec::Rounded, 16 * 1024)
                .unwrap();
        writer.enable_impact_bounds().unwrap();
        writer.push(1, 3, 11).unwrap();
        assert_eq!(
            writer.finish(None).unwrap_err().kind(),
            io::ErrorKind::WriteZero
        );
        assert!(BlockPostingList::deserialize(&bytes).is_err());
    }
    #[tokio::test]
    async fn incomplete_compaction_metadata_reads_return_errors_without_panicking() {
        let mut postings = PostingList::new();
        for doc in 0..256 {
            postings.push(doc, 3);
        }
        let list = BlockPostingList::from_posting_list_with_impact_bounds(
            &postings,
            false,
            Some(&|_| 11),
            PostingCodec::Rounded,
        )
        .unwrap();
        let mut bytes = Vec::new();
        list.serialize(&mut bytes).unwrap();
        let bytes = OwnedBytes::new(bytes);
        let footer_start = (bytes.len() - FOOTER_V2_SIZE) as u64;
        for returned in [0, 1] {
            let source = bytes.clone();
            let file = FileHandle::lazy(
                source.len() as u64,
                Arc::new(move |range| {
                    let end = if range.start == footer_start {
                        range.end as usize
                    } else {
                        range.start as usize + returned
                    };
                    let result = source.slice(range.start as usize..end);
                    Box::pin(async move { Ok(result) })
                }),
            );
            assert!(
                PostingBlockSource::open(file, 16 * 1024, None)
                    .await
                    .is_err()
            );
        }
    }
}
