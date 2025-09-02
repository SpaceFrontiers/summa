//! Field selection and bounded retained/encoded response accounting.

use crate::proto::*;
use hermes_core::FieldValue as CoreFieldValue;
use std::io::Write;
use tonic::Status;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ResolvedField {
    pub(super) id: hermes_core::dsl::Field,
    pub(super) name: String,
}

/// Resolve and deduplicate field names once. Unknown names retain the existing
/// API behavior (they are ignored), while aliases/duplicates cannot multiply
/// per-hit work or HashMap capacity.
pub(super) fn resolve_requested_fields(
    schema: &hermes_core::Schema,
    requested: &[String],
) -> Vec<ResolvedField> {
    let mut seen = rustc_hash::FxHashSet::default();
    let mut resolved = Vec::with_capacity(requested.len().min(schema.num_fields()));
    for name in requested {
        let Some(id) = schema.get_field(name) else {
            continue;
        };
        if !seen.insert(id.0) {
            continue;
        }
        let canonical_name = schema.get_field_name(id).unwrap_or(name).to_owned();
        resolved.push(ResolvedField {
            id,
            name: canonical_name,
        });
    }
    resolved
}

#[derive(Debug)]
pub(super) struct SearchResponseBudget {
    retained_bytes: usize,
    encoded_bytes: usize,
    maximum: usize,
}

impl SearchResponseBudget {
    pub(super) fn with_maximum(maximum: usize) -> Self {
        Self {
            retained_bytes: 0,
            encoded_bytes: 0,
            maximum,
        }
    }

    fn reserve(counter: &mut usize, bytes: usize, maximum: usize) -> Result<(), Status> {
        let next = counter.checked_add(bytes).ok_or_else(|| {
            Status::invalid_argument("Search response size accounting overflowed")
        })?;
        if next > maximum {
            // Deterministic for a given request shape: retrying cannot succeed,
            // so this must not be RESOURCE_EXHAUSTED (which clients treat as
            // retryable capacity pressure, e.g. "Search capacity is full").
            return Err(Status::invalid_argument(format!(
                "Search response exceeds the {maximum}-byte hydration budget; \
                 request fewer hits or fields"
            )));
        }
        *counter = next;
        Ok(())
    }

    pub(super) fn reserve_retained(&mut self, bytes: usize) -> Result<(), Status> {
        Self::reserve(&mut self.retained_bytes, bytes, self.maximum)
    }

    pub(super) fn reserve_hit(&mut self, hit: &SearchHit) -> Result<(), Status> {
        let payload = prost::Message::encoded_len(hit);
        let framed = payload
            .checked_add(protobuf_varint_len(payload))
            .and_then(|bytes| bytes.checked_add(1))
            .ok_or_else(|| Status::invalid_argument("Search response encoded size overflowed"))?;
        Self::reserve(&mut self.encoded_bytes, framed, self.maximum)
    }
}

fn protobuf_varint_len(mut value: usize) -> usize {
    let mut bytes = 1;
    while value >= 0x80 {
        value >>= 7;
        bytes += 1;
    }
    bytes
}

#[derive(Default)]
struct CountingWriter(usize);

impl Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("serialized JSON size overflow"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Conservative retained-heap estimate charged before cloning a stored value
/// into its protobuf counterpart. Doubling the value object accounts for Vec
/// growth slack when a field has many tiny values.
pub(super) fn retained_field_value_bytes(value: &CoreFieldValue) -> Result<usize, Status> {
    let payload = match value {
        CoreFieldValue::Text(text) => text.len(),
        CoreFieldValue::U64(_) | CoreFieldValue::I64(_) | CoreFieldValue::F64(_) => 0,
        CoreFieldValue::Bytes(bytes) | CoreFieldValue::BinaryDenseVector(bytes) => bytes.len(),
        CoreFieldValue::SparseVector(entries) => entries
            .len()
            .checked_mul(std::mem::size_of::<u32>() + std::mem::size_of::<f32>())
            .ok_or_else(|| Status::invalid_argument("Sparse field size overflowed"))?,
        CoreFieldValue::DenseVector(values) => values
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Status::invalid_argument("Dense field size overflowed"))?,
        CoreFieldValue::Json(json) => {
            let mut writer = CountingWriter::default();
            serde_json::to_writer(&mut writer, json).map_err(|error| {
                Status::internal(format!("Failed to size JSON response field: {error}"))
            })?;
            writer.0
        }
    };
    std::mem::size_of::<FieldValue>()
        .saturating_mul(2)
        .checked_add(payload)
        .ok_or_else(|| Status::invalid_argument("Response field size overflowed"))
}

pub(super) fn retained_hit_base_bytes(ordinal_count: usize) -> Result<usize, Status> {
    ordinal_count
        .checked_mul(std::mem::size_of::<OrdinalScore>())
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<SearchHit>()))
        .and_then(|bytes| bytes.checked_add(32)) // segment-id String backing bytes
        .ok_or_else(|| Status::invalid_argument("Search hit size overflowed"))
}

pub(super) fn retained_field_entry_bytes(name: &str) -> usize {
    // HashMap growth keeps spare buckets. Charge two entries so many tiny
    // fields cannot evade the payload budget through container overhead.
    (std::mem::size_of::<String>() + std::mem::size_of::<FieldValueList>())
        .saturating_mul(2)
        .saturating_add(name.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tonic::Code;
    #[test]
    fn requested_fields_are_resolved_and_deduplicated_once() {
        let mut builder = hermes_core::SchemaBuilder::default();
        let title = builder.add_text_field("title", true, true);
        let body = builder.add_text_field("body", true, true);
        let schema = builder.build();
        let requested = vec![
            "title".to_owned(),
            "missing".to_owned(),
            "title".to_owned(),
            "body".to_owned(),
        ];

        assert_eq!(
            resolve_requested_fields(&schema, &requested),
            vec![
                ResolvedField {
                    id: title,
                    name: "title".to_owned(),
                },
                ResolvedField {
                    id: body,
                    name: "body".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn response_budget_bounds_retained_and_encoded_bytes() {
        let value = CoreFieldValue::Bytes(vec![0; 64]);
        assert!(retained_field_value_bytes(&value).unwrap() > 64);

        let mut retained = SearchResponseBudget::with_maximum(100);
        retained.reserve_retained(80).unwrap();
        let err = retained.reserve_retained(21).unwrap_err();
        // Budget violations are deterministic caller errors, not transient
        // capacity pressure: clients retry ResourceExhausted with backoff,
        // which can never succeed for an oversized hydration request.
        assert_eq!(err.code(), Code::InvalidArgument);

        let hit = SearchHit {
            address: Some(DocAddress {
                segment_id: "0".repeat(32),
                doc_id: 1,
            }),
            score: 1.0,
            fields: HashMap::from([(
                "body".to_owned(),
                FieldValueList {
                    values: vec![FieldValue {
                        value: Some(field_value::Value::Text("x".repeat(128))),
                    }],
                },
            )]),
            ordinal_scores: Vec::new(),
        };
        let mut encoded = SearchResponseBudget::with_maximum(64);
        let err = encoded.reserve_hit(&hit).unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);
    }
}
