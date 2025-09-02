//! Fast-field merge: raw block stacking from source segments.
//!
//! Each source segment's fast-field column is a sequence of blocks.
//! Merge = concatenate blocks from all source segments via raw byte copy
//! (memcpy from mmap). No per-value decode/re-encode.
//!
//! For segments missing a field, a compact missing-value block is synthesized.

use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::FieldType;
use crate::segment::reader::SegmentReader;
use crate::segment::types::SegmentFiles;
use crate::structures::fast_field::{
    BLOCK_INDEX_ENTRY_SIZE, BlockIndexEntry, FastFieldColumnType, FastFieldTocEntry,
    missing_block_data, write_fast_field_toc_and_footer,
};

use super::SegmentMerger;

impl SegmentMerger {
    /// Merge fast-field columns from source segments into a new `.fast` file.
    ///
    /// Uses raw block stacking: copies block data+dict bytes directly from
    /// mmap'd source readers. Memory usage = O(block_index) per column.
    pub(super) async fn merge_fast_fields<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
    ) -> Result<usize> {
        // Collect fields with columnar storage (fast fields)
        let mut fast_fields: Vec<(u32, FieldType)> = self
            .schema
            .fields()
            .filter(|(_, entry)| {
                entry.fast
                    && matches!(
                        entry.field_type,
                        FieldType::U64 | FieldType::I64 | FieldType::F64 | FieldType::Text
                    )
            })
            .map(|(field, entry)| (field.0, entry.field_type.clone()))
            .collect();

        if fast_fields.is_empty() {
            return Ok(0);
        }

        // Check if any source segment actually has fast-field data
        let has_data = segments.iter().any(|s| !s.fast_fields().is_empty());
        if !has_data {
            return Ok(0);
        }

        let total_docs: u32 = segments.iter().map(|s| s.num_docs()).sum();

        // Sort field_ids for deterministic output
        fast_fields.sort_by_key(|&(id, _)| id);

        let mut fast_writer = dir.streaming_writer_cold(&files.fast).await?;
        let mut toc_entries: Vec<FastFieldTocEntry> = Vec::with_capacity(fast_fields.len());
        let mut current_offset = 0u64;

        for &(field_id, ref field_type) in &fast_fields {
            self.ensure_not_cancelled()?;
            let is_multi = self
                .schema
                .get_field_entry(crate::dsl::Field(field_id))
                .map(|e| e.multi)
                .unwrap_or(false);
            let column_type = match field_type {
                FieldType::U64 => FastFieldColumnType::U64,
                FieldType::I64 => FastFieldColumnType::I64,
                FieldType::F64 => FastFieldColumnType::F64,
                FieldType::Text => FastFieldColumnType::TextOrdinal,
                _ => continue,
            };

            // Collect block info from all source segments
            let mut all_blocks: Vec<SourceBlock> = Vec::new();

            for segment in segments.iter() {
                let num_docs = segment.num_docs();
                match segment.fast_field(field_id) {
                    Some(reader) => {
                        // Flatten blocks from source reader
                        for block in reader.blocks() {
                            all_blocks.push(SourceBlock::Raw {
                                num_docs: block.num_docs,
                                data: block.data.as_slice(),
                                dict_count: block.dict.as_ref().map(|d| d.len()).unwrap_or(0),
                                dict_bytes: block.raw_dict.as_slice(),
                            });
                        }
                    }
                    None => {
                        // No fast-field data — synthesize a missing-value block
                        if num_docs > 0 {
                            all_blocks.push(SourceBlock::Missing { num_docs });
                        }
                    }
                }
            }

            let bytes_written = write_merged_column(&mut *fast_writer, is_multi, &all_blocks)
                .map_err(crate::Error::Io)?;

            toc_entries.push(FastFieldTocEntry {
                field_id,
                column_type,
                multi: is_multi,
                data_offset: current_offset,
                data_len: bytes_written,
                num_docs: total_docs,
                dict_offset: 0,
                dict_count: 0,
            });
            current_offset += bytes_written;
        }

        let toc_offset = current_offset;
        write_fast_field_toc_and_footer(&mut *fast_writer, toc_offset, &toc_entries)
            .map_err(crate::Error::Io)?;
        fast_writer.finish()?;

        let total_bytes = toc_offset as usize + toc_entries.len() * 38 + 16;

        log::info!(
            "[merge] index={} fast-fields: {} columns, {} docs, {} (raw block stacking)",
            self.schema.index_label(),
            toc_entries.len(),
            total_docs,
            crate::format_bytes(total_bytes as u64)
        );

        Ok(total_bytes)
    }
}

/// A block from a source segment — either raw bytes or a synthetic missing-value block.
enum SourceBlock<'a> {
    /// Raw block data from an existing segment (memcpy)
    Raw {
        num_docs: u32,
        data: &'a [u8],
        dict_count: u32,
        dict_bytes: &'a [u8],
    },
    /// Segment had no data for this field — preserve missing semantics
    Missing { num_docs: u32 },
}

/// Write a merged blocked column: [num_blocks] [block_index] [block_data+dict...]
fn write_merged_column(
    writer: &mut dyn Write,
    is_multi: bool,
    blocks: &[SourceBlock],
) -> std::io::Result<u64> {
    // Every absent source has the same encoded representation regardless of
    // its document count. Keep one tiny payload, not per-source document arrays
    // or a second copy of the block directory.
    let missing_data = if blocks
        .iter()
        .any(|block| matches!(block, SourceBlock::Missing { .. }))
    {
        missing_block_data(is_multi)?
    } else {
        Vec::new()
    };
    let num_blocks = u32::try_from(blocks.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "too many fast-field blocks",
        )
    })?;
    writer.write_u32::<LittleEndian>(num_blocks)?;
    let mut total = 4 + u64::from(num_blocks) * BLOCK_INDEX_ENTRY_SIZE as u64;

    for block in blocks {
        let entry = match block {
            SourceBlock::Raw {
                num_docs,
                data,
                dict_count,
                dict_bytes,
            } => BlockIndexEntry {
                num_docs: *num_docs,
                data_len: data.len() as u32,
                dict_count: *dict_count,
                dict_len: dict_bytes.len() as u32,
            },
            SourceBlock::Missing { num_docs } => BlockIndexEntry {
                num_docs: *num_docs,
                data_len: missing_data.len() as u32,
                dict_count: 0,
                dict_len: 0,
            },
        };
        entry.write_to(writer)?;
    }

    for block in blocks {
        match block {
            SourceBlock::Raw {
                data, dict_bytes, ..
            } => {
                writer.write_all(data)?;
                writer.write_all(dict_bytes)?;
                total += (data.len() + dict_bytes.len()) as u64;
            }
            SourceBlock::Missing { .. } => {
                writer.write_all(&missing_data)?;
                total += missing_data.len() as u64;
            }
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directories::OwnedBytes;
    use crate::structures::fast_field::{FAST_FIELD_MISSING, FastFieldReader, FastFieldWriter};

    #[test]
    fn missing_fast_fields_match_builder_bytes_and_preserve_raw_blocks() {
        for column_type in [
            FastFieldColumnType::U64,
            FastFieldColumnType::I64,
            FastFieldColumnType::F64,
            FastFieldColumnType::TextOrdinal,
        ] {
            for multi in [false, true] {
                for num_docs in [1, 2, 513, 8_192] {
                    let mut reference = match (column_type, multi) {
                        (FastFieldColumnType::TextOrdinal, false) => FastFieldWriter::new_text(),
                        (FastFieldColumnType::TextOrdinal, true) => {
                            FastFieldWriter::new_text_multi()
                        }
                        (_, false) => FastFieldWriter::new_numeric(column_type),
                        (_, true) => FastFieldWriter::new_numeric_multi(column_type),
                    };
                    reference.pad_to(num_docs);
                    let mut expected = Vec::new();
                    let (mut toc, _) = reference.serialize(&mut expected, 0).unwrap();
                    let data = &expected[4 + BLOCK_INDEX_ENTRY_SIZE..];
                    let blocks = [
                        SourceBlock::Raw {
                            num_docs,
                            data,
                            dict_count: 0,
                            dict_bytes: &[],
                        },
                        SourceBlock::Missing { num_docs },
                        SourceBlock::Raw {
                            num_docs,
                            data,
                            dict_count: 0,
                            dict_bytes: &[],
                        },
                    ];
                    let mut merged = Vec::new();
                    let len = write_merged_column(&mut merged, multi, &blocks).unwrap();
                    assert_eq!(len as usize, merged.len());
                    // Same header for each block, same builder payload, including
                    // the raw blocks on both sides of the synthesized source.
                    assert_eq!(&merged[..4], &3u32.to_le_bytes());
                    for header in merged[4..4 + 3 * BLOCK_INDEX_ENTRY_SIZE]
                        .chunks_exact(BLOCK_INDEX_ENTRY_SIZE)
                    {
                        assert_eq!(header, &expected[4..4 + BLOCK_INDEX_ENTRY_SIZE]);
                    }
                    assert_eq!(&merged[4 + 3 * BLOCK_INDEX_ENTRY_SIZE..], data.repeat(3));
                    toc.num_docs = 3 * num_docs;
                    toc.data_len = len;
                    let reader = FastFieldReader::open(&OwnedBytes::new(merged), &toc).unwrap();
                    for id in [
                        0,
                        num_docs - 1,
                        num_docs,
                        2 * num_docs - 1,
                        2 * num_docs,
                        3 * num_docs - 1,
                    ] {
                        assert!(!reader.has_value(id));
                        if multi {
                            assert!(reader.get_multi_values(id).is_empty());
                        } else {
                            assert_eq!(reader.get_u64(id), FAST_FIELD_MISSING);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn billion_document_missing_fast_field_uses_constant_space() {
        for multi in [false, true] {
            let mut bytes = Vec::new();
            write_merged_column(
                &mut bytes,
                multi,
                &[SourceBlock::Missing {
                    num_docs: 1_000_000_000,
                }],
            )
            .unwrap();
            assert_eq!(
                bytes.len(),
                4 + BLOCK_INDEX_ENTRY_SIZE + if multi { 23 } else { 9 }
            );
        }
    }

    #[tokio::test]
    async fn merge_reopens_with_missing_columns_around_present_values() {
        use crate::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId};
        use crate::{Document, RamDirectory, SchemaBuilder};
        use std::sync::Arc;

        let schema_with_fast = |fast| {
            let mut builder = SchemaBuilder::default();
            let scalar = builder.add_u64_field("scalar", false, false);
            let labels = builder.add_text_field("labels", false, false);
            builder.set_fast(scalar, fast);
            builder.set_fast(labels, fast);
            builder.set_multi(labels, true);
            Arc::new(builder.build())
        };
        let schema = schema_with_fast(true);
        let scalar = schema.get_field("scalar").unwrap();
        let labels = schema.get_field("labels").unwrap();
        let dir = RamDirectory::new();
        let mut sources = Vec::new();
        for present in [false, true, false] {
            let mut builder =
                SegmentBuilder::new(schema_with_fast(present), SegmentBuilderConfig::default())
                    .unwrap();
            for id in 0..3 {
                let mut doc = Document::new();
                if present {
                    doc.add_u64(scalar, 100 + id);
                    doc.add_text(labels, "zebra");
                    doc.add_text(labels, "apple");
                }
                builder.add_document(doc).unwrap();
            }
            let id = SegmentId::new();
            builder.build(&dir, id, None).await.unwrap();
            sources.push(
                SegmentReader::open(&dir, id, Arc::clone(&schema), 0)
                    .await
                    .unwrap(),
            );
        }
        let output = SegmentId::new();
        SegmentMerger::new(Arc::clone(&schema))
            .merge(&dir, &sources, output, None)
            .await
            .unwrap();
        let reader = SegmentReader::open(&dir, output, schema, 0).await.unwrap();
        assert_eq!(reader.num_docs(), 9);
        let numeric = reader.fast_field(scalar.0).unwrap();
        let text = reader.fast_field(labels.0).unwrap();
        for id in 0..9 {
            if (3..6).contains(&id) {
                assert_eq!(numeric.get_u64(id), u64::from(100 + id - 3));
                let values = text.get_multi_values(id);
                let dict = text.text_dict().unwrap();
                let decoded: Vec<_> = values
                    .iter()
                    .map(|&value| dict.get(value as u32).unwrap())
                    .collect();
                assert_eq!(decoded, ["zebra", "apple"]);
            } else {
                assert!(!numeric.has_value(id));
                assert!(text.get_multi_values(id).is_empty());
            }
        }
    }
}
