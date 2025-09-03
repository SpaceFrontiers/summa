//! Candidate addressing shared by every point-scoring vertical.

use super::SegmentReader;
use crate::dsl::{Field, FieldType};

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
    /// Resolve every stored value of the selected documents in one field.
    /// The caller controls the expansion budget; no field silently truncates
    /// a long document or invents a body ordinal for document context.
    pub(crate) async fn candidate_locations(
        &self,
        field: Field,
        documents: &[DocId],
        max_locations: usize,
        budget: &mut super::SparseProbeBudget,
    ) -> Result<Vec<CandidateLocation>> {
        if documents.iter().any(|&doc| doc >= self.num_docs())
            || !documents.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err(Error::Query(
                "candidate documents must be valid, unique and sorted".into(),
            ));
        }
        if documents.is_empty() {
            return Ok(Vec::new());
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
                if !map.has_logical_addressing() {
                    return Err(Error::Query("legacy reordered text needs explicit Reorder to upgrade its chunk map for L1".into()));
                }
                for &doc in documents {
                    for (ordinal, physical) in map.slots_for_document(doc) {
                        push(doc, ordinal, physical)?;
                    }
                }
            }
            FieldType::SparseVector => {
                let Some(bmp) = self.bmp_indexes.get(&field.0) else {
                    if self.sparse_indexes.contains_key(&field.0) {
                        return self
                            .maxscore_candidate_locations(
                                field,
                                documents,
                                None,
                                max_locations,
                                budget,
                            )
                            .await;
                    }
                    return Ok(locations);
                };
                for &doc in documents {
                    if let Some(forward) = bmp.forward() {
                        for (ordinal, physical) in forward.for_document(doc) {
                            push(doc, ordinal, physical)?;
                        }
                    } else if bmp.logically_ordered() {
                        for (ordinal, physical) in bmp.ordered_slots_for_document(doc) {
                            push(doc, ordinal, physical)?;
                        }
                    } else {
                        return Err(Error::Query("reordered BMP backfill requires forward values; enable bmp_forward_index and explicitly reorder/rebuild, or disable L1 backfill".into()));
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
                    map.has_logical_addressing()
                } else if let Some(bmp) = self.bmp_indexes.get(&field.0) {
                    bmp.forward().is_some() || bmp.logically_ordered()
                } else {
                    !(self.vector_indexes.contains_key(&field.0)
                        && !self.flat_vectors.contains_key(&field.0))
                        && !(matches!(entry.field_type, FieldType::Text)
                            && !entry.chunked
                            && self
                                .meta
                                .field_stats
                                .get(&field.0)
                                .is_some_and(|stats| stats.total_tokens > 0)
                            && self.doc_lengths(field).is_none())
                };
                (!prepared).then(|| entry.name.clone())
            })
            .collect()
    }
}

impl SegmentReader {
    /// Resolve only nominated logical passages. Lookup costs depend on the
    /// candidate set, not on the number of chunks in a book.
    pub(crate) async fn candidate_passage_locations(
        &self,
        field: Field,
        targets: &[crate::segment::logical_address::LogicalUnit],
        budget: &mut super::SparseProbeBudget,
    ) -> Result<Vec<CandidateLocation>> {
        use crate::segment::logical_address::{LogicalUnit, ordered_slot_for_unit};
        if self.sparse_indexes.contains_key(&field.0) {
            let mut documents: Vec<_> = targets.iter().map(|t| t.doc).collect();
            documents.dedup();
            return self
                .maxscore_candidate_locations(
                    field,
                    &documents,
                    Some(targets),
                    targets.len(),
                    budget,
                )
                .await;
        }
        let mut locations = Vec::with_capacity(targets.len());
        for &target in targets {
            let physical = if let Some(map) = self.chunk_map(field) {
                if !map.has_logical_addressing() {
                    return Err(Error::Query("legacy reordered text needs explicit Reorder to upgrade its chunk map for L1".into()));
                }
                map.slot_for_unit(target)
            } else if let Some(bmp) = self.bmp_indexes.get(&field.0) {
                if let Some(forward) = bmp.forward() {
                    forward.find(target)
                } else if bmp.logically_ordered() {
                    ordered_slot_for_unit(bmp.num_virtual_docs, target, |physical| {
                        let (doc, ordinal) = bmp.virtual_to_doc(physical);
                        (doc != u32::MAX).then_some(LogicalUnit { doc, ordinal })
                    })
                } else {
                    return Err(Error::Query(
                        "reordered BMP backfill requires forward values; enable bmp_forward_index and explicitly reorder/rebuild, or disable L1 backfill"
                            .into(),
                    ));
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
                    if let Some(forward) = bmp.forward() {
                        let key = forward.key(physical);
                        (key.doc, key.ordinal)
                    } else {
                        bmp.virtual_to_doc(physical)
                    }
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

impl SegmentReader {
    async fn maxscore_candidate_locations(
        &self,
        field: Field,
        documents: &[DocId],
        targets: Option<&[crate::segment::logical_address::LogicalUnit]>,
        limit: usize,
        budget: &mut super::SparseProbeBudget,
    ) -> Result<Vec<CandidateLocation>> {
        use crate::segment::logical_address::LogicalUnit;
        let index = &self.sparse_indexes[&field.0];
        let mut units = std::collections::BTreeSet::new();
        index
            .probe_candidates(documents, None, budget, |doc, ordinal, _| {
                let key = LogicalUnit { doc, ordinal };
                if targets.is_some_and(|targets| targets.binary_search(&key).is_err()) {
                    return Ok(());
                }
                if !units.contains(&key) {
                    if units.len() == limit {
                        return Err(Error::Query(
                            "L1 sparse field-value expansion budget exceeded".into(),
                        ));
                    }
                    units.insert(key);
                }
                Ok(())
            })
            .await?;
        Ok(units
            .into_iter()
            .enumerate()
            .map(|(i, key)| CandidateLocation {
                doc: key.doc,
                ordinal: key.ordinal,
                physical: i as u32,
            })
            .collect())
    }
}

impl SegmentReader {
    /// Existing text readers return zero-copy views on mmap/RAM but materialize
    /// ranges on lazy backends. Admit those ranges before invoking the reader.
    pub(crate) async fn reserve_candidate_text_reads(
        &self,
        field: Field,
        term: &[u8],
        positions: bool,
        remaining: &mut u64,
    ) -> Result<()> {
        let lazy_postings = !self.postings_handle.is_sync();
        let lazy_positions =
            positions && self.positions_handle.as_ref().is_some_and(|h| !h.is_sync());
        if !lazy_postings && !lazy_positions {
            return Ok(());
        }
        let mut key = Vec::with_capacity(4 + term.len());
        key.extend_from_slice(&field.0.to_le_bytes());
        key.extend_from_slice(term);
        let Some(info) = self.term_dict.get(&key).await? else {
            return Ok(());
        };
        let posting_bytes = if lazy_postings {
            info.external_info().map_or(0, |(_, bytes)| bytes)
        } else {
            0
        };
        let position_bytes = if lazy_positions {
            info.position_info().map_or(0, |(_, bytes)| bytes)
        } else {
            0
        };
        *remaining = posting_bytes
            .checked_add(position_bytes)
            .and_then(|bytes| remaining.checked_sub(bytes))
            .ok_or_else(|| Error::Query("L1 lazy text read budget exceeded (256 MiB)".into()))?;
        Ok(())
    }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    #[tokio::test]
    async fn lazy_text_backfill_is_admitted_before_any_payload_read() {
        use crate::directories::{FileHandle, RamDirectory};
        use crate::{Document, Index, IndexConfig, IndexWriter, Schema};
        let mut schema = Schema::builder();
        let field = schema.add_text_field_with_tokenizer("body", true, false, "simple");
        let schema = std::sync::Arc::new(schema.build());
        let dir = RamDirectory::new();
        let config = IndexConfig::default();
        let mut writer = IndexWriter::create(dir.clone(), (*schema).clone(), config.clone())
            .await
            .unwrap();
        for _ in 0..256 {
            let mut doc = Document::new();
            doc.add_text(field, "common term");
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        let index = Index::open(dir.clone(), config).await.unwrap();
        let searcher = index.reader().await.unwrap().searcher().await.unwrap();
        let id = crate::segment::SegmentId(searcher.segment_readers()[0].meta().id);
        let mut reader = SegmentReader::open(&dir, id, schema, 4).await.unwrap();
        reader.postings_handle = FileHandle::lazy(
            reader.postings_handle.len(),
            std::sync::Arc::new(|_| Box::pin(async { panic!("payload I/O before admission") })),
        );
        let error = reader
            .reserve_candidate_text_reads(field, b"common", false, &mut 0)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("text read budget"));
        reader
            .reserve_candidate_text_reads(field, b"absent", false, &mut 0)
            .await
            .unwrap();
        // Statistics precede scoring admission, so they must read only the
        // dictionary even when a common term has an external posting list.
        assert_eq!(reader.text_doc_freq(field, b"common").await.unwrap(), 256);
        assert_eq!(reader.text_doc_freq(field, b"absent").await.unwrap(), 0);
    }
}
