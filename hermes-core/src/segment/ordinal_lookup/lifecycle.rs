//! Derived lookups participate in the owning segment's existing publication
//! claim. Ordinary merges concatenate/remap sorted metadata; only an explicit
//! reorder prepares a new physical permutation. No per-query corpus scans.
use super::*;
use crate::Schema;
use crate::directories::{Directory, DirectoryWriter};
use crate::segment::chunk_map::ChunkMap;
use crate::segment::reader::bmp::BmpIndex;
use crate::segment::{SegmentFiles, SegmentMeta, SegmentReader};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
enum Map<'a> {
    Text(&'a ChunkMap),
    Sparse(&'a BmpIndex),
}
impl Map<'_> {
    fn kind(self) -> LookupKind {
        match self {
            Self::Text(_) => LookupKind::ChunkedText,
            Self::Sparse(_) => LookupKind::SparseBmp,
        }
    }
    fn slots(self) -> u32 {
        match self {
            Self::Text(map) => map.num_chunks(),
            Self::Sparse(map) => map.num_virtual_docs,
        }
    }
    fn count(self) -> u32 {
        match self {
            Self::Text(map) => map.num_chunks(),
            Self::Sparse(map) => map.num_real_docs(),
        }
    }
    fn ordered(self) -> bool {
        match self {
            Self::Text(map) => map.logically_ordered(),
            Self::Sparse(map) => map.logically_ordered(),
        }
    }
    fn resolve(self, physical: u32) -> Option<LogicalUnit> {
        let (doc, ordinal) = match self {
            Self::Text(map) => map.resolve(physical),
            Self::Sparse(map) => map.virtual_to_doc(physical),
        };
        (doc != u32::MAX).then_some(LogicalUnit { doc, ordinal })
    }
}
fn source_maps(reader: &SegmentReader) -> impl Iterator<Item = (u32, Map<'_>)> {
    reader
        .chunk_maps()
        .iter()
        .map(|(&field, map)| (field, Map::Text(map)))
        .chain(
            reader
                .bmp_indexes()
                .iter()
                .map(|(&field, map)| (field, Map::Sparse(map))),
        )
}
fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Call before the existing owner publishes the output generation. All files
/// remain owned by its cancellation guard, including failures during sorting.
/// `rebuild` identifies physical payloads the caller explicitly reordered.
pub(crate) async fn write_generation_lookups<D: Directory + DirectoryWriter>(
    dir: &D,
    schema: &Schema,
    meta: &SegmentMeta,
    sources: &[&SegmentReader],
    rebuild: (bool, bool),
    memory_budget: usize,
) -> Result<()> {
    let files = SegmentFiles::new(meta.id);
    // The ordinary build/copy/merge fast path needs no sidecar or output read.
    if !rebuild.0
        && !rebuild.1
        && sources
            .iter()
            .all(|reader| source_maps(reader).all(|(_, map)| map.ordered()))
    {
        return Ok(());
    }
    let chunks = if rebuild.0 {
        Some(crate::segment::reader::loader::load_chunk_maps_file(dir, &files, schema).await?)
    } else {
        None
    };
    let sparse = if rebuild.1 {
        Some(
            crate::segment::reader::loader::load_sparse_file(dir, &files, meta.num_docs, schema)
                .await?,
        )
    } else {
        None
    };
    let mut rewritten = BTreeMap::new();
    if let Some(chunks) = &chunks {
        rewritten.extend(
            chunks
                .chunk_maps
                .iter()
                .map(|(&field, map)| (field, Map::Text(map))),
        );
    }
    if let Some(sparse) = &sparse {
        rewritten.extend(
            sparse
                .bmp_indexes
                .iter()
                .map(|(&field, map)| (field, Map::Sparse(map))),
        );
    }
    let mut original: BTreeMap<u32, Vec<(&SegmentReader, Map<'_>, u32)>> = BTreeMap::new();
    let mut doc_offset = 0u32;
    for &reader in sources {
        for (field, map) in source_maps(reader) {
            original
                .entry(field)
                .or_default()
                .push((reader, map, doc_offset));
        }
        doc_offset = doc_offset
            .checked_add(reader.num_docs())
            .ok_or_else(|| Error::Corruption("lookup document offset overflow".into()))?;
    }
    if doc_offset != meta.num_docs {
        return Err(Error::Corruption(
            "lookup sources do not cover output documents".into(),
        ));
    }
    let mut sections = Vec::new();
    for (field, maps) in original {
        if let Some(&map) = rewritten.get(&field) {
            if map.ordered() {
                continue;
            }
            let scratch = (map.count() as usize)
                .checked_mul(4)
                .ok_or_else(|| Error::Query("lookup preparation scratch overflow".into()))?;
            if scratch > memory_budget {
                return Err(Error::Query(format!(
                    "candidate lookup for field {field} needs {scratch} scratch bytes; reorder budget is {memory_budget}"
                )));
            }
            sections.push(StreamSection {
                field,
                kind: map.kind(),
                count: map.count(),
                physical_slots: map.slots(),
                emit: Box::new(move |visit| {
                    let mut slots = Vec::with_capacity(map.count() as usize);
                    // The canonical BMP validator distinguishes corrupt slots from
                    // padding. Never skip an invalid real document as a missing value.
                    if let Map::Sparse(bmp) = map {
                        bmp.visit_real_slots_for_rewrite(|physical| {
                            slots.push(physical as u32);
                        })
                        .map_err(|error| invalid(error.to_string()))?;
                    } else {
                        slots.extend(0..map.slots());
                    }
                    slots.sort_unstable_by_key(|&slot| map.resolve(slot));
                    for slot in slots {
                        visit(
                            map.resolve(slot)
                                .ok_or_else(|| invalid("lookup source became padding"))?,
                            slot,
                        )?;
                    }
                    Ok(())
                }),
            });
            continue;
        }
        if maps.iter().all(|(_, map, _)| map.ordered()) {
            continue;
        }
        // Existing unprepared generations remain usable for ordinary search.
        // They advertise lack of L1 support and can be prepared by Reorder.
        if maps.iter().any(|(reader, map, _)| {
            !map.ordered()
                && reader
                    .prepared_lookup(crate::Field(field), map.kind(), map.slots())
                    .is_err()
        }) {
            log::warn!(
                "[candidate_lookup] index={} field {} has legacy reordered sources without lookup; ordinary merge preserves the representation; run Reorder before L1",
                schema.index_label(),
                field
            );
            continue;
        }
        let kind = maps[0].1.kind();
        let slots = maps.iter().try_fold(0u32, |n, (_, map, _)| {
            n.checked_add(map.slots())
                .ok_or_else(|| invalid("lookup physical offset overflow"))
        })?;
        let count = maps.iter().try_fold(0u32, |n, (_, map, _)| {
            n.checked_add(map.count())
                .ok_or_else(|| invalid("lookup row count overflow"))
        })?;
        sections.push(StreamSection {
            field,
            kind,
            count,
            physical_slots: slots,
            emit: Box::new(move |visit| {
                let mut physical_offset = 0u32;
                for &(reader, map, doc_offset) in &maps {
                    let mut emit = |unit: LogicalUnit, physical: u32| {
                        if map.resolve(physical) != Some(unit) {
                            return Err(invalid("lookup source identity mismatch"));
                        }
                        visit(
                            LogicalUnit {
                                doc: unit.doc + doc_offset,
                                ordinal: unit.ordinal,
                            },
                            physical + physical_offset,
                        )
                    };
                    if map.ordered() {
                        for physical in 0..map.slots() {
                            if let Some(unit) = map.resolve(physical) {
                                emit(unit, physical)?;
                            }
                        }
                    } else {
                        let lookup = reader
                            .prepared_lookup(crate::Field(field), kind, map.slots())
                            .map_err(|e| invalid(e.to_string()))?;
                        for (unit, physical) in lookup.rows() {
                            emit(unit, physical)?;
                        }
                    }
                    physical_offset += map.slots();
                }
                Ok(())
            }),
        });
    }
    if sections.is_empty() {
        return Ok(());
    }
    let mut writer = dir.streaming_writer_cold(&files.ordinal_lookup).await?;
    #[cfg(feature = "native")]
    let run = || {
        write_streamed(writer.as_mut(), meta.id, meta.num_docs, &sections)?;
        writer.finish()
    };
    #[cfg(feature = "native")]
    crate::segment::merger::block_in_place_if_multithread(run)?;
    #[cfg(not(feature = "native"))]
    {
        write_streamed(writer.as_mut(), meta.id, meta.num_docs, &sections)?;
        writer.finish()?;
    }
    Ok(())
}
