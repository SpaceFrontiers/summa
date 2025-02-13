//! Fast-field merge: raw block stacking from source segments.
//!
//! Each source segment's fast-field column is a sequence of blocks.
//! Merge = concatenate blocks from all source segments via raw byte copy
//! (memcpy from mmap). No per-value decode/re-encode.
//!
//! For segments missing a field, a zero-filled single block is synthesized.

use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::FieldType;
use crate::segment::reader::AsyncSegmentReader as SegmentReader;
use crate::segment::types::SegmentFiles;
use crate::structures::fast_field::{
    BLOCK_INDEX_ENTRY_SIZE, BlockIndexEntry, FastFieldColumnType, FastFieldTocEntry,
    FastFieldWriter, write_fast_field_toc_and_footer,
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
        // Collect fast-enabled fields from schema
        let fast_fields: Vec<(u32, FieldType)> = self
            .schema
            .fields()
            .filter(|(_, entry)| entry.fast)
            .filter(|(_, entry)| {
                matches!(
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
        let mut sorted_fields = fast_fields.clone();
        sorted_fields.sort_by_key(|&(id, _)| id);

        let mut fast_writer = dir.streaming_writer(&files.fast).await?;
        let mut toc_entries: Vec<FastFieldTocEntry> = Vec::with_capacity(sorted_fields.len());
        let mut current_offset = 0u64;

        for &(field_id, ref field_type) in &sorted_fields {
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
                        // No fast-field data — synthesize a zero block
                        if num_docs > 0 {
                            all_blocks.push(SourceBlock::Missing {
                                num_docs,
                                is_multi,
                                column_type,
                            });
                        }
                    }
                }
            }

            let bytes_written = write_merged_column(
                &mut *fast_writer,
                field_id,
                column_type,
                is_multi,
                total_docs,
                &all_blocks,
            )
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
            "[merge] fast-fields: {} columns, {} docs, {} (raw block stacking)",
            toc_entries.len(),
            total_docs,
            super::format_bytes(total_bytes)
        );

        Ok(total_bytes)
    }
}

/// A block from a source segment — either raw bytes or a synthetic zero block.
enum SourceBlock<'a> {
    /// Raw block data from an existing segment (memcpy)
    Raw {
        num_docs: u32,
        data: &'a [u8],
        dict_count: u32,
        dict_bytes: &'a [u8],
    },
    /// Segment had no data for this field — synthesize zeros
    Missing {
        num_docs: u32,
        is_multi: bool,
        column_type: FastFieldColumnType,
    },
}

/// Write a merged blocked column: [num_blocks] [block_index] [block_data+dict...]
fn write_merged_column(
    writer: &mut dyn Write,
    _field_id: u32,
    _column_type: FastFieldColumnType,
    _is_multi: bool,
    _total_docs: u32,
    blocks: &[SourceBlock],
) -> std::io::Result<u64> {
    // Precompute block index entries (need to know data/dict sizes before writing)
    let mut index_entries: Vec<BlockIndexEntry> = Vec::with_capacity(blocks.len());
    let mut block_payloads: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(blocks.len());

    for block in blocks {
        match block {
            SourceBlock::Raw {
                num_docs,
                data,
                dict_count,
                dict_bytes,
            } => {
                index_entries.push(BlockIndexEntry {
                    num_docs: *num_docs,
                    data_len: data.len() as u32,
                    dict_count: *dict_count,
                    dict_len: dict_bytes.len() as u32,
                });
                // Raw data is written directly from source — no allocation
                block_payloads.push((Vec::new(), Vec::new())); // placeholder
            }
            SourceBlock::Missing {
                num_docs,
                is_multi: blk_multi,
                column_type: blk_type,
            } => {
                // Synthesize a zero-filled block
                let (data_buf, dict_buf, dict_count) =
                    synthesize_zero_block(*num_docs, *blk_multi, *blk_type)?;
                index_entries.push(BlockIndexEntry {
                    num_docs: *num_docs,
                    data_len: data_buf.len() as u32,
                    dict_count,
                    dict_len: dict_buf.len() as u32,
                });
                block_payloads.push((data_buf, dict_buf));
            }
        }
    }

    let mut total = 0u64;

    // Write num_blocks
    writer.write_u32::<LittleEndian>(blocks.len() as u32)?;
    total += 4;

    // Write block index
    for entry in &index_entries {
        entry.write_to(writer)?;
    }
    total += (blocks.len() * BLOCK_INDEX_ENTRY_SIZE) as u64;

    // Write block data + dicts
    for (i, block) in blocks.iter().enumerate() {
        match block {
            SourceBlock::Raw {
                data, dict_bytes, ..
            } => {
                writer.write_all(data)?;
                total += data.len() as u64;
                writer.write_all(dict_bytes)?;
                total += dict_bytes.len() as u64;
            }
            SourceBlock::Missing { .. } => {
                let (ref data_buf, ref dict_buf) = block_payloads[i];
                writer.write_all(data_buf)?;
                total += data_buf.len() as u64;
                writer.write_all(dict_buf)?;
                total += dict_buf.len() as u64;
            }
        }
    }

    Ok(total)
}

/// Synthesize a zero-filled block for a segment that lacks a fast field.
/// Returns (data_bytes, dict_bytes, dict_count).
fn synthesize_zero_block(
    num_docs: u32,
    is_multi: bool,
    column_type: FastFieldColumnType,
) -> std::io::Result<(Vec<u8>, Vec<u8>, u32)> {
    let mut writer = if is_multi {
        match column_type {
            FastFieldColumnType::TextOrdinal => FastFieldWriter::new_text_multi(),
            _ => FastFieldWriter::new_numeric_multi(column_type),
        }
    } else {
        match column_type {
            FastFieldColumnType::TextOrdinal => FastFieldWriter::new_text(),
            _ => FastFieldWriter::new_numeric(column_type),
        }
    };
    writer.pad_to(num_docs);

    // Serialize through the normal writer path, then strip the blocked header
    // (we just want the inner block data + dict)
    let mut buf = Vec::new();
    let (_toc, _total) = writer.serialize(&mut buf, 0)?;

    // The serialized format is: [num_blocks(4)] [BlockIndexEntry(16)] [data] [dict]
    // We need to extract just the data and dict portions
    if buf.len() < 4 + BLOCK_INDEX_ENTRY_SIZE {
        return Ok((Vec::new(), Vec::new(), 0));
    }
    let mut cursor = std::io::Cursor::new(&buf[4..4 + BLOCK_INDEX_ENTRY_SIZE]);
    let entry = BlockIndexEntry::read_from(&mut cursor)?;
    let data_start = 4 + BLOCK_INDEX_ENTRY_SIZE;
    let data_end = data_start + entry.data_len as usize;
    let dict_end = data_end + entry.dict_len as usize;

    let data_bytes = buf[data_start..data_end].to_vec();
    let dict_bytes = if dict_end > data_end {
        buf[data_end..dict_end].to_vec()
    } else {
        Vec::new()
    };

    Ok((data_bytes, dict_bytes, entry.dict_count))
}
