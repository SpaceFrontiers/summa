//! Lossless compaction of one immutable segment, through owning encoders.
use super::*;
use crate::segment::chunk_map::{ChunkMapBuilder, DocLengthsColumn, write_chunk_maps};
use crate::segment::row_map::RowMap;
use crate::structures::fast_field::{
    BLOCK_INDEX_ENTRY_SIZE, FastFieldReader, write_fast_field_toc_and_footer,
};
use crate::structures::{PositionStreamEncoder, SSTableWriter, TermInfo};
use std::io::Write;

fn admit(bytes: usize, budget: usize) -> Result<()> {
    if bytes > budget {
        return Err(crate::Error::Schema(format!(
            "compaction scratch requires {bytes} bytes; remaining budget is {budget}"
        )));
    }
    Ok(())
}

impl SegmentMerger {
    /// Remove deleted rows from one segment. The caller owns input/output IDs
    /// and publishes the result. `memory_budget` bounds decoded scratch, in
    /// addition to the immutable source reader's documented residency.
    pub async fn compact<D: DirectoryWriter>(
        &self,
        dir: &D,
        source: &SegmentReader,
        output: SegmentId,
        memory_budget: usize,
    ) -> Result<(SegmentMeta, MergeStats)> {
        self.ensure_not_cancelled()?;
        let rows = RowMap::from_visibility(
            source.num_docs(),
            source.num_live_docs(),
            source.alive_docs(),
            memory_budget / 4,
            || self.ensure_not_cancelled(),
        )?;
        let budget = memory_budget.saturating_sub(rows.memory_bytes());
        // Validate complete statistics before output: old standalone segments
        // without this column cannot distinguish missing and empty text.
        for (field, entry) in self.schema.fields() {
            if ((entry.indexed && entry.field_type == FieldType::Text)
                || entry.field_type == FieldType::SparseVector)
                && !source.row_stats().contains_key(&field.0)
            {
                return Err(crate::Error::Schema(format!(
                    "field '{}' lacks lossless row statistics; rebuild this pre-deletion-format segment before compaction",
                    entry.name
                )));
            }
        }
        let files = SegmentFiles::new(output.0);
        let mut stats = MergeStats::default();
        let (field_stats, chunk_maps) = self
            .compact_text_maps(dir, source, &rows, &files, budget)
            .await?;
        let posting_budget =
            budget.saturating_sub(chunk_maps.values().map(RowMap::memory_bytes).sum::<usize>());
        stats.terms_processed = self
            .compact_postings(dir, source, &rows, &chunk_maps, &files, posting_budget)
            .await?;
        drop(chunk_maps);
        stats.fast_bytes = self
            .compact_columns(dir, &files.fast, source.fast_fields(), &rows, budget)
            .await?;
        self.compact_columns(dir, &files.row_stats, source.row_stats(), &rows, budget)
            .await?;
        let mut store_out = OffsetWriter::new(dir.streaming_writer_cold(&files.store).await?);
        let mut store = crate::segment::StoreMerger::new(&mut store_out);
        store
            .append_compacted(source.store(), &rows, self.cancellation.as_deref(), budget)
            .await?;
        if store.finish()? != rows.len() {
            return Err(crate::Error::Corruption(
                "compacted store row count mismatch".into(),
            ));
        }
        stats.store_bytes = store_out.offset() as usize;
        store_out.finish()?;
        stats.vectors_bytes = self
            .compact_vectors(dir, source, &rows, &files, budget)
            .await?;
        stats.sparse_bytes = self
            .compact_sparse(dir, source, &rows, &files, budget)
            .await?;
        self.ensure_not_cancelled()?;
        let meta = SegmentMeta {
            id: output.0,
            num_docs: rows.len(),
            field_stats,
        };
        dir.write_durable(&files.meta, &meta.serialize()?).await?;
        Ok((meta, stats))
    }

    async fn compact_columns<D: DirectoryWriter>(
        &self,
        dir: &D,
        path: &std::path::Path,
        columns: &FxHashMap<u32, FastFieldReader>,
        rows: &RowMap,
        budget: usize,
    ) -> Result<usize> {
        use crate::structures::fast_field::BlockIndexEntry;
        struct OutputBlock {
            source: usize,
            range: std::ops::Range<u32>,
            header: BlockIndexEntry,
            copy: bool,
            cached: Option<Vec<u8>>,
        }
        if columns.is_empty() {
            return Ok(0);
        }
        let mut writer = OffsetWriter::new(dir.streaming_writer_cold(path).await?);
        let mut fields: Vec<_> = columns.keys().copied().collect();
        fields.sort_unstable();
        let mut toc = Vec::new();
        for field in fields {
            self.ensure_not_cancelled()?;
            let reader = &columns[&field];
            if reader.num_docs != rows.physical() {
                return Err(crate::Error::Corruption(
                    "compaction column row count mismatch".into(),
                ));
            }
            let capacity = reader
                .blocks()
                .iter()
                .map(|block| {
                    let live =
                        rows.count(block.cumulative_docs..block.cumulative_docs + block.num_docs);
                    if live == 0 {
                        0
                    } else if live == block.num_docs {
                        1
                    } else {
                        (live as usize).min((block.num_docs as usize).div_ceil(4096))
                    }
                })
                .sum::<usize>();
            admit(
                capacity.saturating_mul(std::mem::size_of::<OutputBlock>()),
                budget / 4,
            )?;
            let mut plan = Vec::with_capacity(capacity);
            let mut cached_bytes = 0usize;
            for (source, block) in reader.blocks().iter().enumerate() {
                self.ensure_not_cancelled()?;
                let live =
                    rows.count(block.cumulative_docs..block.cumulative_docs + block.num_docs);
                if live == 0 {
                    continue;
                }
                if live == block.num_docs {
                    plan.push(OutputBlock {
                        source,
                        range: 0..block.num_docs,
                        copy: true,
                        cached: None,
                        header: BlockIndexEntry {
                            num_docs: live,
                            data_len: block.data.len() as u32,
                            dict_count: block.dict.as_ref().map_or(0, |dict| dict.len()),
                            dict_len: block.raw_dict.len() as u32,
                        },
                    });
                    continue;
                }
                for start in (0..block.num_docs).step_by(4096) {
                    self.ensure_not_cancelled()?;
                    let end = start.saturating_add(4096).min(block.num_docs);
                    if rows.count(block.cumulative_docs + start..block.cumulative_docs + end) == 0 {
                        continue;
                    }
                    let bytes = block.compact_range(
                        reader.column_type,
                        reader.multi,
                        start..end,
                        |doc| rows.get(doc).is_some(),
                        budget / 2,
                    )?;
                    let header = BlockIndexEntry::read_from(&mut &bytes[4..])?;
                    let cached = if bytes.capacity() <= (budget / 4).saturating_sub(cached_bytes) {
                        cached_bytes += bytes.capacity();
                        Some(bytes)
                    } else {
                        None
                    };
                    plan.push(OutputBlock {
                        source,
                        range: start..end,
                        header,
                        copy: false,
                        cached,
                    });
                }
            }
            let offset = writer.offset();
            writer.write_all(&(plan.len() as u32).to_le_bytes())?;
            for block in &plan {
                block.header.write_to(&mut writer)?;
            }
            for block in plan {
                self.ensure_not_cancelled()?;
                let source = &reader.blocks()[block.source];
                if block.copy {
                    for data in [source.data.as_slice(), source.raw_dict.as_slice()] {
                        for chunk in data.chunks(4 * 1024 * 1024) {
                            self.ensure_not_cancelled()?;
                            writer.write_all(chunk)?;
                        }
                    }
                } else {
                    let bytes = match block.cached {
                        Some(bytes) => bytes,
                        None => source.compact_range(
                            reader.column_type,
                            reader.multi,
                            block.range,
                            |doc| rows.get(doc).is_some(),
                            budget / 2,
                        )?,
                    };
                    let mut header = Vec::with_capacity(BLOCK_INDEX_ENTRY_SIZE);
                    block.header.write_to(&mut header)?;
                    if bytes.get(4..4 + BLOCK_INDEX_ENTRY_SIZE) != Some(header.as_slice()) {
                        return Err(crate::Error::Corruption(
                            "non-deterministic compacted fast column".into(),
                        ));
                    }
                    writer.write_all(&bytes[4 + BLOCK_INDEX_ENTRY_SIZE..])?;
                }
            }
            toc.push(crate::structures::fast_field::FastFieldTocEntry {
                field_id: field,
                column_type: reader.column_type,
                multi: reader.multi,
                data_offset: offset,
                data_len: writer.offset() - offset,
                num_docs: rows.len(),
                dict_offset: 0,
                dict_count: 0,
            });
        }
        let offset = writer.offset();
        write_fast_field_toc_and_footer(&mut writer, offset, &toc)?;
        let bytes = writer.offset() as usize;
        writer.finish()?;
        Ok(bytes)
    }

    async fn compact_text_maps<D: DirectoryWriter>(
        &self,
        dir: &D,
        source: &SegmentReader,
        rows: &RowMap,
        files: &SegmentFiles,
        budget: usize,
    ) -> Result<(FxHashMap<u32, FieldStats>, FxHashMap<u32, RowMap>)> {
        let mut retained_bytes = 0usize;
        let mut statistics = FxHashMap::default();
        let mut chunks = Vec::new();
        let mut maps = FxHashMap::default();
        let mut norms = Vec::new();
        for (field, entry) in self.schema.fields() {
            if !entry.indexed || entry.field_type != FieldType::Text {
                continue;
            }
            self.ensure_not_cancelled()?;
            let exact = &source.row_stats()[&field.0];
            let mut stat = FieldStats::default();
            for (visited, old) in rows.iter().enumerate() {
                if visited.is_multiple_of(4096) {
                    self.ensure_not_cancelled()?;
                }
                let value = exact.get_u64(old);
                if value > 0 {
                    stat.doc_count += 1;
                    stat.total_tokens =
                        stat.total_tokens.checked_add(value - 1).ok_or_else(|| {
                            crate::Error::Corruption("compacted token count overflow".into())
                        })?;
                }
            }
            if entry.chunked {
                stat.doc_count = 0;
                if let Some(source_map) = source.chunk_map(field) {
                    let map = RowMap::try_new(
                        source_map.num_chunks(),
                        source_map.num_chunks(),
                        |vid| {
                            if vid.is_multiple_of(4096) {
                                self.ensure_not_cancelled()?;
                            }
                            Ok(rows.get(source_map.doc_id(vid)).is_some())
                        },
                        (budget / 2).saturating_sub(retained_bytes),
                    )?;
                    retained_bytes = retained_bytes
                        .saturating_add(map.memory_bytes())
                        .saturating_add(map.len() as usize * 8);
                    admit(retained_bytes, budget / 2)?;
                    let mut builder = ChunkMapBuilder::with_capacity(map.len() as usize);
                    for (visited, old) in map.iter().enumerate() {
                        if visited.is_multiple_of(4096) {
                            self.ensure_not_cancelled()?;
                        }
                        let (doc, ordinal) = source_map.resolve(old);
                        builder.push(rows.get(doc).unwrap(), ordinal, source_map.length(old))?;
                    }
                    stat.doc_count = map.len();
                    builder.set_total_tokens(stat.total_tokens);
                    chunks.push((field.0, builder));
                    maps.insert(field.0, map);
                }
            } else if let Some(lengths) = source.doc_lengths(field) {
                retained_bytes = retained_bytes.saturating_add(rows.len() as usize * 2);
                admit(retained_bytes, budget / 2)?;
                let mut values = Vec::with_capacity(rows.len() as usize);
                for (visited, old) in rows.iter().enumerate() {
                    if visited.is_multiple_of(4096) {
                        self.ensure_not_cancelled()?;
                    }
                    values.push(lengths.length(old).min(u16::MAX as u32) as u16);
                }
                norms.push((field.0, values, stat.total_tokens));
            }
            statistics.insert(field.0, stat);
        }
        let chunk_refs: Vec<_> = chunks
            .iter()
            .filter(|(_, map)| !map.is_empty())
            .map(|(field, map)| (*field, map))
            .collect();
        let norm_refs: Vec<_> = norms
            .iter()
            .map(|(field, lengths, total)| DocLengthsColumn {
                field_id: *field,
                lengths,
                total_tokens: *total,
            })
            .collect();
        if !chunk_refs.is_empty() || !norm_refs.is_empty() {
            let mut writer = dir.streaming_writer_cold(&files.chunks).await?;
            write_chunk_maps(&mut *writer, &chunk_refs, &norm_refs)?;
            writer.finish()?;
        }
        Ok((statistics, maps))
    }

    async fn compact_postings<D: DirectoryWriter>(
        &self,
        dir: &D,
        source: &SegmentReader,
        rows: &RowMap,
        chunks: &FxHashMap<u32, RowMap>,
        files: &SegmentFiles,
        budget: usize,
    ) -> Result<usize> {
        // Fixed posting/position decoder and encoder buffers are independent
        // of term frequency. Divide the rest between encoded directories.
        admit(16 * 1024, budget)?;
        let budget = budget - 16 * 1024;
        let mut postings = OffsetWriter::new(dir.streaming_writer_cold(&files.postings).await?);
        let mut positions = OffsetWriter::new(dir.streaming_writer_cold(&files.positions).await?);
        let mut terms_out = OffsetWriter::new(dir.streaming_writer_cold(&files.term_dict).await?);
        let mut terms = SSTableWriter::<_, TermInfo>::with_config(
            &mut terms_out,
            crate::structures::SSTableWriterConfig::from_optimization(self.optimization),
        );
        let mut iter = source.term_dict_iter();
        let mut count = 0;
        while let Some((key, info)) = iter.next().await? {
            self.ensure_not_cancelled()?;
            let field = crate::Field(u32::from_le_bytes(
                key.get(..4)
                    .ok_or_else(|| crate::Error::Corruption("invalid term field prefix".into()))?
                    .try_into()
                    .unwrap(),
            ));
            let map = chunks.get(&field.0).unwrap_or(rows);
            let info = if let Some((ids, tfs)) = info.decode_inline() {
                let entries: Vec<_> = ids
                    .into_iter()
                    .zip(tfs)
                    .filter_map(|(old, tf)| map.get(old).map(|new| (new, tf)))
                    .collect();
                if entries.is_empty() {
                    continue;
                }
                // Removing rows only reduces the inline count and addresses.
                TermInfo::try_inline_iter(entries.len(), entries.into_iter()).ok_or_else(|| {
                    crate::Error::Corruption("compacted inline posting no longer fits".into())
                })?
            } else if let Some((offset, len)) = info.external_info() {
                use crate::structures::postings::{
                    PositionRangeSource, PostingBlockSource, PostingStreamWriter,
                };
                let cancellation = self.cancellation.as_deref();
                let input = PostingBlockSource::open(
                    source.posting_file_range(offset, len)?,
                    budget / 4,
                    cancellation,
                )
                .await?;
                if input.doc_count() != info.doc_freq() || input.doc_count() > map.physical() {
                    return Err(crate::Error::Corruption(
                        "posting source count disagrees with term or row space".into(),
                    ));
                }
                let mut source_positions = match info.position_info() {
                    Some((offset, len)) => Some(
                        PositionRangeSource::open(
                            source.position_file_range(offset, len)?,
                            budget / 4,
                            cancellation,
                        )
                        .await?,
                    ),
                    None => None,
                };
                let has_positions = source_positions.is_some();
                if input.has_positions() && !has_positions
                    || source_positions
                        .as_ref()
                        .is_some_and(|positions| positions.is_stream() && !input.has_positions())
                {
                    return Err(crate::Error::Corruption(
                        "posting and position formats disagree".into(),
                    ));
                }
                if let Some(positions) = source_positions
                    .as_ref()
                    .filter(|positions| positions.is_stream())
                {
                    let total = if input.len() == 0 {
                        0
                    } else {
                        input.position_span(input.len() - 1)?.end
                    };
                    if positions.total_positions() != total {
                        return Err(crate::Error::Corruption(
                            "posting and position counts disagree".into(),
                        ));
                    }
                }
                let off = postings.offset();
                let pos_off = positions.offset();
                let mut output = PostingStreamWriter::new(
                    &mut postings,
                    input.len(),
                    has_positions,
                    self.posting_codec,
                    budget / 4,
                )?;
                let mut position_output =
                    PositionStreamEncoder::with_budget(&mut positions, budget / 4);
                let mut docs = Vec::with_capacity(crate::structures::postings::POSTING_BLOCK_SIZE);
                let mut tfs = Vec::with_capacity(crate::structures::postings::POSTING_BLOCK_SIZE);
                for i in 0..input.len() {
                    self.ensure_not_cancelled()?;
                    let (first, last) = input.bounds(i);
                    if last >= map.physical() {
                        return Err(crate::Error::Corruption(
                            "posting address exceeds compaction map".into(),
                        ));
                    }
                    let live = map.count(first..last + 1);
                    if live == 0 {
                        continue;
                    }
                    let block = input.read_block(i).await?;
                    let span = input.position_span(i)?;
                    if live == last - first + 1
                        && (!has_positions
                            || source_positions
                                .as_ref()
                                .is_some_and(|positions| positions.is_stream()))
                    {
                        output.append(&block, map.get(first).expect("live interval"))?;
                        if let Some(positions) = &mut source_positions {
                            positions
                                .append_range(&mut position_output, span, cancellation)
                                .await?;
                        }
                        continue;
                    }
                    if !block.decode_block_into(0, &mut docs, &mut tfs)
                        || docs.first() != Some(&first)
                        || docs.last() != Some(&last)
                        || !docs.windows(2).all(|pair| pair[0] < pair[1])
                        || tfs.contains(&0)
                    {
                        return Err(crate::Error::Corruption(
                            "invalid compacted posting block".into(),
                        ));
                    }
                    let mut cursor = span.start;
                    for (&old, &tf) in docs.iter().zip(&tfs) {
                        if let Some(new) = map.get(old) {
                            let length = source.chunk_map(field).map_or_else(
                                || source.doc_lengths(field).map_or(1, |norm| norm.length(old)),
                                |map| map.bm25_length(old),
                            );
                            output.push(new, tf, length)?;
                            if let Some(positions) = &mut source_positions {
                                positions
                                    .append_doc(&mut position_output, old, cursor, tf, cancellation)
                                    .await?;
                            }
                        }
                        cursor = cursor.checked_add(u64::from(tf)).ok_or_else(|| {
                            crate::Error::Corruption("position count overflow".into())
                        })?;
                    }
                    if input.has_positions() && cursor != span.end {
                        return Err(crate::Error::Corruption(
                            "position cursor disagrees with term frequencies".into(),
                        ));
                    }
                }
                if output.doc_count() == 0 {
                    continue;
                }
                if output.doc_count() > input.doc_count() {
                    return Err(crate::Error::Corruption(
                        "compacted posting count exceeds source".into(),
                    ));
                }
                let total_positions = output.total_positions();
                let (docs, len) = output.finish(cancellation)?;
                if has_positions {
                    self.ensure_not_cancelled()?;
                    let (total, pos_len) = position_output.finish_cancellable(cancellation)?;
                    if total != total_positions {
                        return Err(crate::Error::Corruption(
                            "compacted position count mismatch".into(),
                        ));
                    }
                    TermInfo::external_with_positions(off, len, docs, pos_off, pos_len)
                } else {
                    TermInfo::external(off, len, docs)
                }
            } else {
                return Err(crate::Error::Corruption(
                    "invalid term posting representation".into(),
                ));
            };
            terms.insert(&key, &info)?;
            count += 1;
        }
        terms.finish()?;
        terms_out.finish()?;
        postings.finish()?;
        if positions.offset() > 0 {
            positions.finish()?;
        } else {
            drop(positions);
            dir.delete(&files.positions).await?;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::BlockPostingList;
    use crate::structures::fast_field::{FastFieldColumnType, FastFieldWriter};

    #[tokio::test]
    async fn frequent_terms_compact_with_block_sized_decoding_and_preserve_positions() {
        use crate::directories::{Directory, RamDirectory};
        use crate::structures::{TERMINATED, TermPositions};
        let mut schema = crate::SchemaBuilder::default();
        let body = schema.add_text_field("body", true, false);
        schema.set_positions(body, crate::dsl::PositionMode::TokenPosition);
        let schema = schema.build();
        let dir = RamDirectory::new();
        let index = crate::Index::create(
            dir.clone(),
            schema.clone(),
            crate::IndexConfig {
                merge_policy: Box::new(crate::NoMergePolicy),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let mut writer = index.writer();
        for _ in 0..8192 {
            let mut doc = crate::Document::new();
            doc.add_text(body, "common common common common");
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        let reader = index.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        for sparse in [false, true] {
            let rows = RowMap::new(
                8192,
                8192,
                |doc| {
                    if sparse {
                        doc % 129 == 0
                    } else {
                        doc >= 256 && doc % 257 != 0
                    }
                },
                4096,
            )
            .unwrap();
            let merger = SegmentMerger::new(Arc::new(schema.clone()));
            let files = SegmentFiles::new(SegmentId::new().0);
            // The old whole-term decoded entries alone required 512 KiB.
            merger
                .compact_postings(
                    &dir,
                    &searcher.segment_readers()[0],
                    &rows,
                    &FxHashMap::default(),
                    &files,
                    64 * 1024,
                )
                .await
                .unwrap();
            let posting_bytes = dir
                .open_read(&files.postings)
                .await
                .unwrap()
                .read_bytes()
                .await
                .unwrap();
            let postings = BlockPostingList::deserialize(posting_bytes.as_slice()).unwrap();
            assert_eq!(postings.doc_count(), rows.len());
            if sparse {
                assert_eq!(
                    postings.num_blocks(),
                    1,
                    "sparse survivors should share a rebuilt block"
                );
            }
            let position_bytes = dir
                .open_read(&files.positions)
                .await
                .unwrap()
                .read_bytes()
                .await
                .unwrap();
            let positions = TermPositions::open(position_bytes).unwrap();
            let mut cursor = postings.iterator();
            let mut scratch = Vec::new();
            let mut values = Vec::new();
            for doc in 0..rows.len() {
                assert_eq!(cursor.doc(), doc);
                assert_eq!(cursor.term_freq(), 4);
                assert!(positions.positions_into(
                    doc,
                    cursor.position_cursor(),
                    4,
                    &mut scratch,
                    &mut values
                ));
                assert_eq!(values, [0, 1, 2, 3]);
                cursor.advance();
            }
            assert_eq!(cursor.doc(), TERMINATED);
        }
    }

    #[tokio::test]
    async fn intact_fast_blocks_keep_encoded_bytes_when_earlier_rows_are_deleted() {
        let n = 8193;
        let mut column = FastFieldWriter::new_numeric(FastFieldColumnType::U64);
        for i in 0..n {
            column.add_u64(i, u64::from(i) * 17);
        }
        let mut encoded = Vec::new();
        let (mut toc, _) = column.serialize(&mut encoded, 0).unwrap();
        let header = &encoded[4..4 + BLOCK_INDEX_ENTRY_SIZE];
        let payload = &encoded[4 + BLOCK_INDEX_ENTRY_SIZE..];
        let mut stacked = 2u32.to_le_bytes().to_vec();
        stacked.extend_from_slice(header);
        stacked.extend_from_slice(header);
        stacked.extend_from_slice(payload);
        stacked.extend_from_slice(payload);
        toc.num_docs = 2 * n;
        toc.data_len = stacked.len() as u64;
        let source =
            FastFieldReader::open(&crate::directories::OwnedBytes::new(stacked), &toc).unwrap();
        let columns = FxHashMap::from_iter([(0, source)]);
        let rows = RowMap::new(2 * n, n, |doc| doc >= n, 1024 * 1024).unwrap();
        let dir = crate::directories::RamDirectory::new();
        let merger = SegmentMerger::new(Arc::new(crate::SchemaBuilder::default().build()));
        let path = std::path::Path::new("copied.fast");
        merger
            .compact_columns(&dir, path, &columns, &rows, 4 * 1024 * 1024)
            .await
            .unwrap();
        let bytes = dir
            .open_read(path)
            .await
            .unwrap()
            .read_bytes()
            .await
            .unwrap();
        let (offset, count) =
            crate::structures::fast_field::read_fast_field_footer(bytes.as_slice()).unwrap();
        let toc =
            crate::structures::fast_field::read_fast_field_toc(bytes.as_slice(), offset, count)
                .unwrap();
        assert_eq!(toc[0].data_len as usize, encoded.len());
        assert_eq!(&bytes.as_slice()[..encoded.len()], encoded.as_slice());
    }

    #[tokio::test]
    async fn cached_and_reencoded_column_chunks_have_identical_bytes_and_missing_rows() {
        // 33 output chunks: incompressible values exceed the 1 MiB prefix
        // cache, and the final chunk contains only one row.
        let count = 2 * (32 * 4096 + 1);
        let mut column = FastFieldWriter::new_numeric(FastFieldColumnType::U64);
        for doc in 0..count {
            if doc % 97 != 0 {
                column.add_u64(
                    doc,
                    u64::from(doc)
                        .wrapping_mul(0x9e3779b97f4a7c15)
                        .rotate_left(23),
                );
            }
        }
        column.pad_to(count);
        let mut encoded = Vec::new();
        let (toc, _) = column.serialize(&mut encoded, 0).unwrap();
        let source =
            FastFieldReader::open(&crate::directories::OwnedBytes::new(encoded), &toc).unwrap();
        let columns = FxHashMap::from_iter([(0, source)]);
        let rows = RowMap::new(count, count / 2, |doc| doc % 2 == 1, 8 * 1024 * 1024).unwrap();
        let dir = crate::directories::RamDirectory::new();
        let merger = SegmentMerger::new(Arc::new(crate::SchemaBuilder::default().build()));
        let mut outputs = Vec::new();
        for (name, budget) in [
            ("partial.fast", 4 * 1024 * 1024),
            ("cached.fast", 16 * 1024 * 1024),
        ] {
            let path = std::path::Path::new(name);
            merger
                .compact_columns(&dir, path, &columns, &rows, budget)
                .await
                .unwrap();
            outputs.push(
                dir.open_read(path)
                    .await
                    .unwrap()
                    .read_bytes()
                    .await
                    .unwrap(),
            );
        }
        assert!(
            outputs[0].len() > 1024 * 1024,
            "fixture no longer exercises cache overflow"
        );
        assert_eq!(outputs[0].as_slice(), outputs[1].as_slice());
        let (offset, count) =
            crate::structures::fast_field::read_fast_field_footer(outputs[0].as_slice()).unwrap();
        let toc = crate::structures::fast_field::read_fast_field_toc(
            outputs[0].as_slice(),
            offset,
            count,
        )
        .unwrap();
        let actual = FastFieldReader::open(&outputs[0], &toc[0]).unwrap();
        assert_eq!(actual.num_docs, rows.len());
        for (new, old) in rows.iter().enumerate() {
            assert_eq!(actual.get_u64(new as u32), columns[&0].get_u64(old));
            assert_eq!(actual.has_value(new as u32), columns[&0].has_value(old));
        }
    }
}
