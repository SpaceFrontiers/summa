//! Merge of chunked-text virtual-id maps (`.chunks`).
//!
//! Virtual ids of a chunked field are segment-local and dense, so merged
//! postings stack with per-field chunk-count offsets (see `chunk_offsets`)
//! while the map itself is a plain section concatenation with the document
//! offset added to `doc_ids`. Ordinals and lengths are copied verbatim.

use rustc_hash::FxHashMap;

use super::SegmentMerger;
use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::Schema;
use crate::segment::chunk_map::{ChunkMapSource, DocLengthsSource, write_merged_chunk_maps};
use crate::segment::reader::SegmentReader;
use crate::segment::types::SegmentFiles;

/// Per chunked field, the virtual-id offset of every source segment: the sum
/// of the chunk counts of the segments before it (same order as `doc_offsets`).
pub(super) fn chunk_offsets(
    schema: &Schema,
    segments: &[SegmentReader],
) -> Result<FxHashMap<u32, Vec<u32>>> {
    let mut offsets: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for (field, entry) in schema.fields() {
        if !entry.chunked {
            continue;
        }
        let mut acc = 0u32;
        let mut per_segment = Vec::with_capacity(segments.len());
        for segment in segments {
            per_segment.push(acc);
            acc = acc.checked_add(segment.num_chunks(field)).ok_or_else(|| {
                crate::Error::Internal(format!(
                    "chunked field '{}' exceeds u32::MAX chunks after merge",
                    entry.name
                ))
            })?;
        }
        offsets.insert(field.0, per_segment);
    }
    Ok(offsets)
}

impl SegmentMerger {
    /// Concatenate the sources' chunk maps into the output `.chunks` file.
    /// Returns the bytes written (0 when no chunked field has data).
    pub(super) async fn merge_chunk_maps<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
    ) -> Result<u64> {
        if segments.iter().all(|s| !s.has_chunks_file()) {
            return Ok(0);
        }
        let mut chunked_fields: Vec<u32> = Vec::new();
        let mut plain_fields: Vec<u32> = Vec::new();
        for (field, entry) in self.schema.fields() {
            if !entry.indexed || entry.field_type != crate::dsl::FieldType::Text {
                continue;
            }
            if entry.chunked {
                chunked_fields.push(field.0);
            } else {
                plain_fields.push(field.0);
            }
        }
        chunked_fields.sort_unstable();
        plain_fields.sort_unstable();

        let doc_offs = super::doc_offsets(segments)?;
        let mut fields: Vec<(u32, Vec<ChunkMapSource<'_>>)> = Vec::new();
        for field_id in chunked_fields {
            let mut sources = Vec::new();
            for (segment, &doc_offset) in segments.iter().zip(doc_offs.iter()) {
                if let Some(map) = segment.chunk_map(crate::dsl::Field(field_id)) {
                    sources.push(ChunkMapSource { map, doc_offset });
                }
            }
            if !sources.is_empty() {
                fields.push((field_id, sources));
            }
        }
        // Norms are dense per document: a source without the section
        // contributes zeros for its documents so doc ids keep lining up.
        let mut norms: Vec<(u32, Vec<DocLengthsSource<'_>>)> = Vec::new();
        for field_id in plain_fields {
            let field = crate::dsl::Field(field_id);
            if segments.iter().all(|s| s.doc_lengths(field).is_none()) {
                continue;
            }
            let sources = segments
                .iter()
                .map(|segment| DocLengthsSource {
                    lengths: segment.doc_lengths(field),
                    num_docs: segment.num_docs(),
                })
                .collect();
            norms.push((field_id, sources));
        }
        if fields.is_empty() && norms.is_empty() {
            return Ok(0);
        }

        self.ensure_not_cancelled()?;
        let mut writer = dir.streaming_writer_cold(&files.chunks).await?;
        let bytes =
            write_merged_chunk_maps(&mut *writer, &fields, &norms).map_err(crate::Error::Io)?;
        writer.finish()?;
        log::info!(
            "[merge] index={} chunk maps done: {} chunked fields, {} norm columns, {}",
            self.schema.index_label(),
            fields.len(),
            norms.len(),
            crate::format_bytes(bytes),
        );
        Ok(bytes)
    }
}
