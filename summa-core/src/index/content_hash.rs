//! Stored fingerprint reads and bounded primary-key ordinal-to-row resolution.
use std::sync::Arc;

use crate::dsl::{Document, FieldValue, Schema};
use crate::segment::AsyncStoreReader;
use crate::structures::fast_field::FastFieldReader;
use crate::{Error, Result};

use super::primary_key::PkSegmentData;

const MAX_ROW_LOOKUP_BYTES: usize = 64 * 1024 * 1024;
const NO_ROW: u32 = u32::MAX;

/// Missing fingerprints are deliberately not an equality assertion.
pub(super) fn document_hash<'a>(
    doc: &'a Document,
    schema: &Schema,
) -> Result<Option<&'a FieldValue>> {
    let Some(field) = schema.content_hash_field() else {
        return Ok(None);
    };
    let mut values = doc.get_all(field);
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some()
        || !matches!(
            (value, &schema.get_field_entry(field).unwrap().field_type),
            (FieldValue::Text(_), crate::dsl::FieldType::Text)
                | (FieldValue::Bytes(_), crate::dsl::FieldType::Bytes)
                | (FieldValue::U64(_), crate::dsl::FieldType::U64)
        )
    {
        return Err(Error::Document(
            "content_hash requires one value of its declared type".into(),
        ));
    }
    Ok(Some(value))
}

pub(super) struct ContentHashLookup {
    pub store: Arc<AsyncStoreReader>,
    // None selects the bounded-scratch scan fallback for oversized dictionaries.
    rows: Option<Vec<u32>>,
}

impl ContentHashLookup {
    pub fn new(
        store: Arc<AsyncStoreReader>,
        column: &FastFieldReader,
        data: &PkSegmentData,
    ) -> Result<Self> {
        let dict = column
            .text_dict()
            .ok_or_else(|| Error::Corruption("primary-key dictionary is missing".into()))?;
        if column.multi || column.num_docs != store.num_docs() {
            return Err(Error::Corruption(
                "primary-key column and document store disagree".into(),
            ));
        }
        if dict.len() as usize > MAX_ROW_LOOKUP_BYTES / 4 {
            log::warn!(
                "[content_hash] segment={} row lookup exceeds 64 MiB; using column scans",
                data.segment_id
            );
            return Ok(Self { store, rows: None });
        }
        let mut rows = vec![NO_ROW; dict.len() as usize];
        column.try_scan_single_values(|doc, ordinal| {
            if data
                .alive_docs
                .as_ref()
                .is_none_or(|alive| alive.contains(doc))
            {
                let row = usize::try_from(ordinal)
                    .ok()
                    .and_then(|ordinal| rows.get_mut(ordinal))
                    .ok_or_else(|| Error::Corruption("invalid primary-key ordinal".into()))?;
                if *row != NO_ROW {
                    return Err(Error::Corruption(
                        "multiple live rows share a primary key".into(),
                    ));
                }
                *row = doc;
            }
            Ok(())
        })?;
        Ok(Self {
            store,
            rows: Some(rows),
        })
    }

    pub fn memory_bytes(&self) -> usize {
        self.rows.as_ref().map_or(0, |rows| rows.capacity() * 4)
    }

    pub fn row(&self, ordinal: u64, column: &FastFieldReader, data: &PkSegmentData) -> Result<u32> {
        let row = if let Some(rows) = &self.rows {
            rows.get(ordinal as usize).copied().unwrap_or(NO_ROW)
        } else {
            let mut row = NO_ROW;
            column.try_scan_single_values(|doc, value| {
                if value == ordinal
                    && data
                        .alive_docs
                        .as_ref()
                        .is_none_or(|alive| alive.contains(doc))
                {
                    if row != NO_ROW {
                        return Err(Error::Corruption(
                            "multiple live rows share a primary key".into(),
                        ));
                    }
                    row = doc;
                }
                Ok(())
            })?;
            row
        };
        if row == NO_ROW {
            return Err(Error::Corruption(
                "live primary key has no document row".into(),
            ));
        }
        Ok(row)
    }
}

/// Owns the exact generation through asynchronous store I/O and topology refresh.
pub(super) struct ContentHashTarget {
    pub store: Arc<AsyncStoreReader>,
    pub row: u32,
    #[cfg(feature = "native")]
    pub snapshot: Option<Arc<crate::segment::SegmentSnapshot>>,
}

impl ContentHashTarget {
    pub async fn matches(&self, incoming: &FieldValue, schema: &Schema) -> Result<bool> {
        #[cfg(feature = "native")]
        let _keep_snapshot_alive = &self.snapshot;
        let field = schema.content_hash_field().unwrap();
        let stored = self
            .store
            .get_fields(self.row, schema, &[field.0])
            .await?
            .ok_or_else(|| {
                Error::Corruption("primary-key row is missing from document store".into())
            })?;
        let hash =
            document_hash(&stored, schema).map_err(|error| Error::Corruption(error.to_string()))?;
        let equal = hash == Some(incoming);
        if equal {
            log::debug!(
                "[content_hash] index={} skipped unchanged upsert",
                schema.index_label()
            );
        }
        Ok(equal)
    }
}
