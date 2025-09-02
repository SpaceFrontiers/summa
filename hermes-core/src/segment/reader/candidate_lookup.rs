//! Candidate addressing shared by every point-scoring vertical.

use super::SegmentReader;
use crate::dsl::{Field, FieldType};
use crate::segment::ordinal_lookup::{LookupKind, OrdinalLookup};
use crate::{DocId, Error, Result};

#[derive(Clone, Copy, Debug)]
pub(crate) struct CandidateLocation {
    pub doc: DocId,
    pub ordinal: u16,
    /// Field-local ID; chunk/BMP physical IDs and flat-vector indexes need
    /// not agree even when they refer to the same logical document ordinal.
    pub physical: u32,
}

impl SegmentReader {
    pub(crate) fn prepared_lookup(
        &self,
        field: Field,
        kind: LookupKind,
        slots: u32,
    ) -> Result<&OrdinalLookup> {
        let lookup = self.ordinal_lookups.get().and_then(|lookups| {
            lookups.binary_search_by_key(&field.0, OrdinalLookup::field).ok().map(|i| &lookups[i])
        }).ok_or_else(|| Error::Query(format!(
            "candidate scoring field {} in segment {:032x} has a reordered map without an ordinal lookup; prepare candidate-scoring lookups before using L1",
            field.0, self.meta.id,
        )))?;
        let expected_count = match kind {
            LookupKind::ChunkedText => self.chunk_map(field).map_or(0, |map| map.num_chunks()),
            LookupKind::SparseBmp => self
                .bmp_indexes
                .get(&field.0)
                .map_or(0, |bmp| bmp.num_real_docs()),
        };
        if lookup.kind() != kind
            || lookup.physical_slots() != slots
            || lookup.len() != expected_count as usize
        {
            return Err(Error::Corruption(format!(
                "candidate lookup for field {} does not match its physical source",
                field.0,
            )));
        }
        Ok(lookup)
    }

    /// Resolve every stored value of the selected documents in one field.
    /// The caller controls the expansion budget; no field silently truncates
    /// a long document or invents a body ordinal for document context.
    pub(crate) fn candidate_locations(
        &self,
        field: Field,
        documents: &[DocId],
        max_locations: usize,
    ) -> Result<Vec<CandidateLocation>> {
        if documents.iter().any(|&doc| doc >= self.num_docs())
            || !documents.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err(Error::Query(
                "candidate documents must be valid, unique and sorted".into(),
            ));
        }
        let entry = self
            .schema
            .get_field_entry(field)
            .ok_or_else(|| Error::FieldNotFound(field.0.to_string()))?;
        let mut locations = Vec::with_capacity(documents.len().min(max_locations));
        let mut push = |doc, ordinal, physical| -> Result<()> {
            if locations.len() == max_locations {
                return Err(Error::Query(format!(
                    "candidate scoring expands beyond {max_locations} field values"
                )));
            }
            locations.push(CandidateLocation {
                doc,
                ordinal,
                physical,
            });
            Ok(())
        };
        match &entry.field_type {
            FieldType::Text if entry.chunked => {
                let Some(map) = self.chunk_map(field) else {
                    return Ok(locations);
                };
                for &doc in documents {
                    if map.logically_ordered() {
                        for (ordinal, physical) in map.ordered_slots_for_document(doc) {
                            push(doc, ordinal, physical)?;
                        }
                    } else {
                        let lookup =
                            self.prepared_lookup(field, LookupKind::ChunkedText, map.num_chunks())?;
                        for (ordinal, physical) in lookup.for_document(doc) {
                            if map.resolve(physical) != (doc, ordinal) {
                                return Err(Error::Corruption(
                                    "text lookup points to another logical chunk".into(),
                                ));
                            }
                            push(doc, ordinal, physical)?;
                        }
                    }
                }
            }
            FieldType::SparseVector => {
                let Some(bmp) = self.bmp_indexes.get(&field.0) else {
                    if self.sparse_indexes.contains_key(&field.0) {
                        return Err(Error::Query("candidate address lookup requires a BMP sparse field; legacy sparse scoring must use posting probes".into()));
                    }
                    return Ok(locations);
                };
                for &doc in documents {
                    if bmp.logically_ordered() {
                        for (ordinal, physical) in bmp.ordered_slots_for_document(doc) {
                            push(doc, ordinal, physical)?;
                        }
                    } else {
                        let lookup = self.prepared_lookup(
                            field,
                            LookupKind::SparseBmp,
                            bmp.num_virtual_docs,
                        )?;
                        for (ordinal, physical) in lookup.for_document(doc) {
                            if bmp.virtual_to_doc(physical) != (doc, ordinal) {
                                return Err(Error::Corruption(
                                    "sparse lookup points to another logical chunk".into(),
                                ));
                            }
                            push(doc, ordinal, physical)?;
                        }
                    }
                }
            }
            FieldType::DenseVector | FieldType::BinaryDenseVector => {
                let Some(flat) = self.flat_vectors.get(&field.0) else {
                    if self.vector_indexes.contains_key(&field.0) {
                        return Err(Error::Query(format!(
                            "candidate scoring needs stored vectors for field {}",
                            entry.name
                        )));
                    }
                    return Ok(locations);
                };
                for &doc in documents {
                    let (start, count) = flat.flat_indexes_for_doc_range(doc);
                    for physical in start..start + count {
                        let (actual_doc, ordinal) = flat.get_doc_id(physical);
                        if actual_doc != doc {
                            return Err(Error::Corruption(
                                "flat vector lookup points to another document".into(),
                            ));
                        }
                        push(
                            doc,
                            ordinal,
                            u32::try_from(physical).map_err(|_| {
                                Error::Query("candidate flat vector address exceeds u32".into())
                            })?,
                        )?;
                    }
                }
            }
            FieldType::Text => {
                if self
                    .meta
                    .field_stats
                    .get(&field.0)
                    .is_none_or(|stats| stats.total_tokens == 0)
                {
                    return Ok(locations);
                }
                let lengths = self.doc_lengths(field).ok_or_else(|| {
                    Error::Query(format!(
                        "candidate scoring requires field-length metadata for plain text field {}",
                        entry.name
                    ))
                })?;
                for &doc in documents {
                    if lengths.length(doc) == 0 {
                        continue;
                    }
                    push(doc, 0, doc)?;
                }
            }
            other => {
                return Err(Error::Query(format!(
                    "candidate scoring does not support field type {other:?}"
                )));
            }
        }
        Ok(locations)
    }
}

impl SegmentReader {
    /// Capability diagnostics for legacy/reordered fields. Missing values are
    /// distinct from a field whose representation cannot support backfill.
    pub fn unprepared_candidate_fields(&self) -> Vec<String> {
        self.schema
            .fields()
            .filter_map(|(field, entry)| {
                let prepared = if let Some(map) = self.chunk_map(field) {
                    map.logically_ordered()
                        || self
                            .prepared_lookup(field, LookupKind::ChunkedText, map.num_chunks())
                            .is_ok()
                } else if let Some(bmp) = self.bmp_indexes.get(&field.0) {
                    bmp.logically_ordered()
                        || self
                            .prepared_lookup(field, LookupKind::SparseBmp, bmp.num_virtual_docs)
                            .is_ok()
                } else {
                    !self.sparse_indexes.contains_key(&field.0)
                        && !(self.vector_indexes.contains_key(&field.0)
                            && !self.flat_vectors.contains_key(&field.0))
                };
                (!prepared).then(|| entry.name.clone())
            })
            .collect()
    }
}

impl SegmentReader {
    /// Resolve only nominated logical passages. Lookup costs depend on the
    /// candidate set, not on the number of chunks in a book.
    pub(crate) fn candidate_passage_locations(
        &self,
        field: Field,
        targets: &[crate::segment::ordinal_lookup::LogicalUnit],
    ) -> Result<Vec<CandidateLocation>> {
        use crate::segment::ordinal_lookup::{LogicalUnit, ordered_slot_for_unit};
        let mut locations = Vec::with_capacity(targets.len());
        for &target in targets {
            let physical = if let Some(map) = self.chunk_map(field) {
                if map.logically_ordered() {
                    ordered_slot_for_unit(map.num_chunks(), target, |physical| {
                        let (doc, ordinal) = map.resolve(physical);
                        Some(LogicalUnit { doc, ordinal })
                    })
                } else {
                    self.prepared_lookup(field, LookupKind::ChunkedText, map.num_chunks())?
                        .for_unit(target)
                }
            } else if let Some(bmp) = self.bmp_indexes.get(&field.0) {
                if bmp.logically_ordered() {
                    ordered_slot_for_unit(bmp.num_virtual_docs, target, |physical| {
                        let (doc, ordinal) = bmp.virtual_to_doc(physical);
                        (doc != u32::MAX).then_some(LogicalUnit { doc, ordinal })
                    })
                } else {
                    self.prepared_lookup(field, LookupKind::SparseBmp, bmp.num_virtual_docs)?
                        .for_unit(target)
                }
            } else if let Some(flat) = self.flat_vectors.get(&field.0) {
                let (start, count) = flat.flat_indexes_for_doc_range(target.doc);
                let mut low = start;
                let mut high = start + count;
                while low < high {
                    let mid = low + (high - low) / 2;
                    if flat.get_doc_id(mid).1 < target.ordinal {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                if low < start + count && flat.get_doc_id(low) == (target.doc, target.ordinal) {
                    Some(
                        u32::try_from(low)
                            .map_err(|_| Error::Query("flat vector address exceeds u32".into()))?,
                    )
                } else {
                    None
                }
            } else if self.sparse_indexes.contains_key(&field.0) {
                return Err(Error::Query("candidate scoring requires BMP sparse storage; migrate this legacy MaxScore field".into()));
            } else if self.vector_indexes.contains_key(&field.0) {
                return Err(Error::Query(
                    "candidate scoring needs stored flat vectors".into(),
                ));
            } else {
                None
            };
            if let Some(physical) = physical {
                let actual = if let Some(map) = self.chunk_map(field) {
                    map.resolve(physical)
                } else if let Some(bmp) = self.bmp_indexes.get(&field.0) {
                    bmp.virtual_to_doc(physical)
                } else {
                    self.flat_vectors[&field.0].get_doc_id(physical as usize)
                };
                if actual != (target.doc, target.ordinal) {
                    return Err(Error::Corruption(
                        "candidate lookup points to another passage".into(),
                    ));
                }
                locations.push(CandidateLocation {
                    doc: target.doc,
                    ordinal: target.ordinal,
                    physical,
                });
            }
        }
        Ok(locations)
    }
}
