//! Fast-field merge: stackable columnar merge from source segments.
//!
//! **Numeric columns** (u64/i64/f64): values are read via `get_u64()` from the
//! zero-copy source readers and collected into a single `FastFieldWriter`.
//! The reader operates directly on the mmap'd `.fast` file — no extra copies.
//!
//! **Text columns**: dictionaries differ across segments, so we rebuild the
//! merged dictionary from the union of all unique strings. Ordinals are
//! re-mapped via the new sorted dictionary.

use rustc_hash::FxHashMap;

use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::FieldType;
use crate::segment::reader::AsyncSegmentReader as SegmentReader;
use crate::segment::types::SegmentFiles;
use crate::structures::fast_field::{
    FastFieldColumnType, FastFieldTocEntry, FastFieldWriter, write_fast_field_toc_and_footer,
};

use super::SegmentMerger;

impl SegmentMerger {
    /// Merge fast-field columns from source segments into a new `.fast` file.
    ///
    /// Source readers are zero-copy (backed by mmap/OwnedBytes), so the
    /// per-doc `get_u64` calls read directly from the file without
    /// intermediate allocations.
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

        // Create writers for each fast field
        let mut writers: FxHashMap<u32, FastFieldWriter> = FxHashMap::default();
        for &(field_id, ref field_type) in &fast_fields {
            let writer = match field_type {
                FieldType::U64 => FastFieldWriter::new_numeric(FastFieldColumnType::U64),
                FieldType::I64 => FastFieldWriter::new_numeric(FastFieldColumnType::I64),
                FieldType::F64 => FastFieldWriter::new_numeric(FastFieldColumnType::F64),
                FieldType::Text => FastFieldWriter::new_text(),
                _ => continue,
            };
            writers.insert(field_id, writer);
        }

        // Iterate source segments, reading from zero-copy mmap'd readers.
        // For numeric columns: get_u64 reads directly from the mmap'd bitpacked data.
        // For text columns: get_text borrows from the mmap'd dictionary.
        let mut merged_doc_id: u32 = 0;
        for segment in segments {
            let num_docs = segment.num_docs();
            for &(field_id, ref field_type) in &fast_fields {
                let src = segment.fast_field(field_id);
                let writer = match writers.get_mut(&field_id) {
                    Some(w) => w,
                    None => continue,
                };

                match (field_type, src) {
                    (FieldType::Text, Some(src_reader)) => {
                        for local in 0..num_docs {
                            if let Some(text) = src_reader.get_text(local) {
                                writer.add_text(merged_doc_id + local, text);
                            }
                            // Missing → leave as TEXT_MISSING_ORDINAL (default)
                        }
                    }
                    (_, Some(src_reader)) => {
                        // Numeric: read raw encoded u64 values directly
                        for local in 0..num_docs {
                            let val = src_reader.get_u64(local);
                            writer.add_u64(merged_doc_id + local, val);
                        }
                    }
                    (_, None) => {
                        // No fast-field data in this segment — pad with zeros
                        writer.pad_to(merged_doc_id + num_docs);
                    }
                }
            }
            merged_doc_id += num_docs;
        }

        // Serialize to .fast file (streaming)
        let mut fast_writer = dir.streaming_writer(&files.fast).await?;

        let mut field_ids: Vec<u32> = writers.keys().copied().collect();
        field_ids.sort_unstable();

        let mut toc_entries: Vec<FastFieldTocEntry> = Vec::with_capacity(field_ids.len());
        let mut current_offset = 0u64;

        for &field_id in &field_ids {
            let ff = writers.get_mut(&field_id).unwrap();
            ff.pad_to(total_docs);

            let (mut toc, bytes_written) = ff
                .serialize(&mut *fast_writer, current_offset)
                .map_err(crate::Error::Io)?;
            toc.field_id = field_id;
            current_offset += bytes_written;
            toc_entries.push(toc);
        }

        let toc_offset = current_offset;
        write_fast_field_toc_and_footer(&mut *fast_writer, toc_offset, &toc_entries)
            .map_err(crate::Error::Io)?;
        fast_writer.finish()?;

        let total_bytes = toc_offset as usize + toc_entries.len() * 38 + 16; // data + toc + footer

        log::info!(
            "[merge] fast-fields: {} columns, {} docs, {}",
            toc_entries.len(),
            total_docs,
            super::format_bytes(total_bytes)
        );

        Ok(total_bytes)
    }
}
