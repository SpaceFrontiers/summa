//! Lossless compaction of one immutable segment, through owning encoders.
use super::*;
use crate::segment::chunk_map::{ChunkMapBuilder, DocLengthsColumn, write_chunk_maps};
use crate::segment::row_map::RowMap;
use crate::structures::fast_field::{
    BLOCK_INDEX_ENTRY_SIZE, FastFieldColumnType, FastFieldReader, FastFieldWriter,
    write_fast_field_toc_and_footer,
};
use crate::structures::{
    BlockPostingList, PositionStreamEncoder, PostingList, SSTableWriter, TERMINATED, TermInfo,
    TermPositions,
};
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
        let rows = RowMap::new(
            source.num_docs(),
            source.num_live_docs(),
            |doc| source.is_alive(doc),
            memory_budget / 4,
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
        if columns.is_empty() {
            return Ok(0);
        }
        let mut writer = OffsetWriter::new(dir.streaming_writer_cold(path).await?);
        let mut fields: Vec<_> = columns.keys().copied().collect();
        fields.sort_unstable();
        let mut toc = Vec::new();
        for field in fields {
            let reader = &columns[&field];
            let offset = writer.offset();
            let chunks = rows.new_to_old.chunks(4096);
            admit(
                chunks.len().saturating_mul(BLOCK_INDEX_ENTRY_SIZE),
                budget / 4,
            )?;
            let mut headers = Vec::with_capacity(chunks.len());
            // Keep only a budgeted prefix of encoded chunks. The leading block
            // directory still requires a first pass over the whole column, but
            // cached chunks avoid a second encode without unbounded buffering.
            let cache_limit = budget / 4;
            let slots = chunks
                .len()
                .min(cache_limit / std::mem::size_of::<Vec<u8>>());
            let mut cached = Vec::with_capacity(slots);
            let mut cached_bytes = cached.capacity() * std::mem::size_of::<Vec<u8>>();
            for chunk in chunks {
                self.ensure_not_cancelled()?;
                let bytes = compact_column_chunk(reader, chunk, budget / 2)?;
                let can_cache = cached.len() == headers.len()
                    && cached.len() < cached.capacity()
                    && bytes.capacity() <= cache_limit.saturating_sub(cached_bytes);
                headers.push(
                    <[u8; BLOCK_INDEX_ENTRY_SIZE]>::try_from(&bytes[4..4 + BLOCK_INDEX_ENTRY_SIZE])
                        .unwrap(),
                );
                if can_cache {
                    cached_bytes += bytes.capacity();
                    cached.push(bytes);
                }
            }
            writer.write_all(&(headers.len() as u32).to_le_bytes())?;
            for header in &headers {
                writer.write_all(header)?;
            }
            let mut cached = cached.into_iter();
            for (chunk, header) in rows.new_to_old.chunks(4096).zip(&headers) {
                self.ensure_not_cancelled()?;
                let bytes = match cached.next() {
                    Some(bytes) => bytes,
                    None => compact_column_chunk(reader, chunk, budget / 2)?,
                };
                if &bytes[4..4 + BLOCK_INDEX_ENTRY_SIZE] != header {
                    return Err(crate::Error::Corruption(
                        "non-deterministic compacted fast column".into(),
                    ));
                }
                writer.write_all(&bytes[4 + BLOCK_INDEX_ENTRY_SIZE..])?;
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
        let estimate = source
            .chunk_maps()
            .values()
            .fold(0usize, |sum, map| {
                sum.saturating_add(map.num_chunks() as usize * 32)
            })
            .saturating_add(
                source
                    .row_stats()
                    .len()
                    .saturating_mul(rows.len() as usize)
                    .saturating_mul(4),
            );
        admit(estimate, budget / 2)?;
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
            for &old in &rows.new_to_old {
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
                    let map = RowMap::new(
                        source_map.num_chunks(),
                        source_map.num_chunks(),
                        |vid| rows.get(source_map.doc_id(vid)).is_some(),
                        budget / 2,
                    )?;
                    let mut builder = ChunkMapBuilder::with_capacity(map.len() as usize);
                    for &old in &map.new_to_old {
                        let (doc, ordinal) = source_map.resolve(old);
                        builder.push(rows.get(doc).unwrap(), ordinal, source_map.length(old))?;
                    }
                    stat.doc_count = map.len();
                    builder.set_total_tokens(stat.total_tokens);
                    chunks.push((field.0, builder));
                    maps.insert(field.0, map);
                }
            } else if let Some(lengths) = source.doc_lengths(field) {
                let values: Vec<u16> = rows
                    .new_to_old
                    .iter()
                    .map(|&old| lengths.length(old).min(u16::MAX as u32) as u16)
                    .collect();
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
            let mut entries = Vec::new();
            let mut position_values = Vec::new();
            let mut pos_scratch = Vec::new();
            let has_positions = info.position_info().is_some();
            let mut retained_bytes = 0usize;
            if let Some((ids, tfs)) = info.decode_inline() {
                for (old, tf) in ids.into_iter().zip(tfs) {
                    if let Some(new) = map.get(old) {
                        entries.push((new, tf));
                    }
                }
            } else if let Some((offset, len)) = info.external_info() {
                admit(len as usize, budget / 4)?;
                let bytes = source.read_postings(offset, len).await?;
                let list = BlockPostingList::deserialize(bytes.as_slice())?;
                admit((list.doc_count() as usize).saturating_mul(64), budget / 4)?;
                let source_positions = match info.position_info() {
                    Some((off, len)) => {
                        admit(len as usize, budget / 4)?;
                        Some(TermPositions::open(
                            source.read_position_bytes(off, len).await?.ok_or_else(|| {
                                crate::Error::Corruption("missing position data".into())
                            })?,
                        )?)
                    }
                    None => None,
                };
                let mut cursor = list.iterator();
                let mut visited = 0usize;
                while cursor.doc() != TERMINATED {
                    if visited.is_multiple_of(4096) {
                        self.ensure_not_cancelled()?;
                    }
                    visited += 1;
                    if let Some(new) = map.get(cursor.doc()) {
                        let tf = cursor.term_freq();
                        entries.push((new, tf));
                        if let Some(source_positions) = &source_positions {
                            retained_bytes = retained_bytes.saturating_add(tf as usize * 8 + 32);
                            admit(retained_bytes, budget / 4)?;
                            let mut pos = Vec::new();
                            if !source_positions.positions_into(
                                cursor.doc(),
                                cursor.position_cursor(),
                                tf,
                                &mut pos_scratch,
                                &mut pos,
                            ) {
                                return Err(crate::Error::Corruption(
                                    "invalid compaction positions".into(),
                                ));
                            }
                            position_values.push(pos);
                        }
                    }
                    cursor.advance();
                }
            }
            if entries.is_empty() {
                continue;
            }
            let info = if !has_positions
                && let Some(inline) =
                    TermInfo::try_inline_iter(entries.len(), entries.iter().copied())
            {
                inline
            } else {
                let mut list = PostingList::with_capacity(entries.len());
                for &(doc, tf) in &entries {
                    list.push(doc, tf);
                }
                let length = |new: u32| {
                    let old = map.new_to_old[new as usize];
                    source.chunk_map(field).map_or_else(
                        || source.doc_lengths(field).map_or(1, |norm| norm.length(old)),
                        |map| map.bm25_length(old),
                    )
                };
                let block = BlockPostingList::from_posting_list_with_options(
                    &list,
                    has_positions,
                    Some(&length),
                    self.posting_codec,
                )?;
                let off = postings.offset();
                block.serialize(&mut postings)?;
                let len = postings.offset() - off;
                if has_positions {
                    let pos_off = positions.offset();
                    let mut encoder = PositionStreamEncoder::new(&mut positions);
                    for pos in &mut position_values {
                        encoder.push_doc(pos)?;
                    }
                    let (_, pos_len) = encoder.finish()?;
                    TermInfo::external_with_positions(off, len, list.doc_count(), pos_off, pos_len)
                } else {
                    TermInfo::external(off, len, list.doc_count())
                }
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

fn compact_column_chunk(reader: &FastFieldReader, docs: &[u32], budget: usize) -> Result<Vec<u8>> {
    let text = reader.column_type == FastFieldColumnType::TextOrdinal;
    let mut column = match (text, reader.multi) {
        (true, false) => FastFieldWriter::new_text(),
        (true, true) => FastFieldWriter::new_text_multi(),
        (false, false) => FastFieldWriter::new_numeric(reader.column_type),
        (false, true) => FastFieldWriter::new_numeric_multi(reader.column_type),
    };
    let mut estimate = docs.len() * 32;
    for (new, &old) in docs.iter().enumerate() {
        if !reader.has_value(old) {
            continue;
        }
        let values = if reader.multi {
            reader.value_range(old)
        } else {
            (0, 1)
        };
        estimate = estimate.saturating_add((values.1 - values.0) as usize * 64);
        admit(estimate, budget / 4)?;
        let mut append = |value| -> Result<()> {
            if text {
                let value = reader
                    .text_dict()
                    .and_then(|dict| dict.get(value as u32))
                    .ok_or_else(|| crate::Error::Corruption("invalid fast text ordinal".into()))?;
                estimate = estimate.saturating_add(value.len().saturating_mul(4));
                admit(estimate, budget / 4)?;
                column.add_text(new as u32, value);
            } else {
                column.add_u64(new as u32, value);
            }
            Ok(())
        };
        if reader.multi {
            let mut result = Ok(());
            reader.for_each_multi_value(old, |value| {
                result = append(value);
                result.is_err()
            });
            result?;
        } else {
            append(reader.get_u64(old))?;
        }
    }
    column.pad_to(docs.len() as u32);
    let mut bytes = Vec::new();
    column.serialize(&mut bytes, 0)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        for (new, &old) in rows.new_to_old.iter().enumerate() {
            assert_eq!(actual.get_u64(new as u32), columns[&0].get_u64(old));
            assert_eq!(actual.has_value(new as u32), columns[&0].has_value(old));
        }
    }
}
