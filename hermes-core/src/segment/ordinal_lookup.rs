//! Logical document/ordinal lookup for field-local physical scoring IDs.
//!
//! Ordered source maps can be searched directly. Reordered maps use an
//! optional immutable `.lookup` sidecar: sorted logical keys plus physical IDs.
//! The sidecar contains no term/vector payload and never changes source IDs.

#[cfg(feature = "native")]
mod lifecycle;
#[cfg(feature = "native")]
pub(crate) use lifecycle::write_generation_lookups;

#[cfg(any(feature = "native", test))]
use std::io::{self, Write};

use crate::directories::OwnedBytes;
use crate::{DocId, Error, Result};

const MAGIC: &[u8; 4] = b"OLKP";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 32;
const FIELD_BYTES: usize = 24;
const ROW_BYTES: usize = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LookupKind {
    ChunkedText = 1,
    SparseBmp = 2,
}

/// A logical unit is a document ID and its actual field value ordinal.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct LogicalUnit {
    pub doc: DocId,
    pub ordinal: u16,
}

#[derive(Clone, Debug)]
pub(crate) struct OrdinalLookup {
    field: u32,
    kind: LookupKind,
    physical_slots: u32,
    rows: OwnedBytes,
}

impl OrdinalLookup {
    pub fn field(&self) -> u32 {
        self.field
    }
    pub fn kind(&self) -> LookupKind {
        self.kind
    }
    pub fn physical_slots(&self) -> u32 {
        self.physical_slots
    }
    pub fn len(&self) -> usize {
        self.rows.len() / ROW_BYTES
    }

    fn row(&self, index: usize) -> (LogicalUnit, u32) {
        let row = &self.rows.as_slice()[index * ROW_BYTES..(index + 1) * ROW_BYTES];
        (
            LogicalUnit {
                doc: u32::from_le_bytes(row[0..4].try_into().unwrap()),
                ordinal: u16::from_le_bytes(row[4..6].try_into().unwrap()),
            },
            u32::from_le_bytes(row[8..12].try_into().unwrap()),
        )
    }

    pub fn rows(&self) -> impl Iterator<Item = (LogicalUnit, u32)> + '_ {
        (0..self.len()).map(|index| self.row(index))
    }

    pub fn for_unit(&self, target: LogicalUnit) -> Option<u32> {
        let mut low = 0;
        let mut high = self.len();
        while low < high {
            let mid = low + (high - low) / 2;
            if self.row(mid).0 < target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        (low < self.len())
            .then(|| self.row(low))
            .and_then(|(unit, physical)| (unit == target).then_some(physical))
    }

    /// File-backed binary search; scratch does not scale with corpus size.
    pub fn for_document(&self, doc: DocId) -> impl Iterator<Item = (u16, u32)> + '_ {
        let mut low = 0;
        let mut high = self.len();
        while low < high {
            let mid = low + (high - low) / 2;
            if self.row(mid).0.doc < doc {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        (low..self.len())
            .map(|index| self.row(index))
            .take_while(move |(unit, _)| unit.doc == doc)
            .map(|(unit, physical)| (unit.ordinal, physical))
    }
}

/// Check order while ignoring BMP padding. No inverse allocation is needed
/// for ordinary build/concatenation maps, including padding between segments.
pub(crate) fn logically_ordered(units: impl IntoIterator<Item = Option<LogicalUnit>>) -> bool {
    let mut previous = None;
    for unit in units.into_iter().flatten() {
        if previous.is_some_and(|previous| previous >= unit) {
            return false;
        }
        previous = Some(unit);
    }
    true
}

/// Search an ordered physical-to-logical map directly. A padded slot takes
/// its preceding document's comparison key, preserving the lower-bound
/// predicate without treating a padding sentinel as a real document ID.
fn ordered_lower_bound(
    physical_slots: u32,
    target: LogicalUnit,
    resolve: &impl Fn(u32) -> Option<LogicalUnit>,
) -> u32 {
    let mut low = 0;
    let mut high = physical_slots;
    while low < high {
        let mid = low + (high - low) / 2;
        let mut prior = mid;
        let prior_unit = loop {
            if let Some(unit) = resolve(prior) {
                break Some(unit);
            }
            if prior == 0 {
                break None;
            }
            prior -= 1;
        };
        if prior_unit.is_none_or(|unit| unit < target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

pub(crate) fn ordered_document_slots(
    physical_slots: u32,
    doc: DocId,
    resolve: impl Fn(u32) -> Option<LogicalUnit>,
) -> impl Iterator<Item = (u16, u32)> {
    let low = ordered_lower_bound(physical_slots, LogicalUnit { doc, ordinal: 0 }, &resolve);
    (low..physical_slots)
        .filter_map(move |physical| resolve(physical).map(|unit| (unit, physical)))
        .take_while(move |(unit, _)| unit.doc == doc)
        .map(|(unit, physical)| (unit.ordinal, physical))
}

pub(crate) fn ordered_slot_for_unit(
    physical_slots: u32,
    target: LogicalUnit,
    resolve: impl Fn(u32) -> Option<LogicalUnit>,
) -> Option<u32> {
    let low = ordered_lower_bound(physical_slots, target, &resolve);
    (low..physical_slots)
        .find_map(|physical| resolve(physical).map(|unit| (unit, physical)))
        .and_then(|(unit, physical)| (unit == target).then_some(physical))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Corruption(message.into())
}

/// Validate identity, version, table bounds and sorted unique logical keys.
/// Physical-to-logical identity is additionally checked against the owning
/// field when a lookup is used, so a stale permutation cannot cross chunks.
pub(crate) fn read_lookups(
    bytes: OwnedBytes,
    segment: u128,
    num_docs: u32,
) -> Result<Vec<OrdinalLookup>> {
    let data = bytes.as_slice();
    if data.len() < HEADER_BYTES || &data[..4] != MAGIC {
        return Err(invalid("ordinal lookup header is missing or invalid"));
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(invalid(format!(
            "unsupported ordinal lookup version {version}"
        )));
    }
    if u128::from_le_bytes(data[8..24].try_into().unwrap()) != segment
        || u32::from_le_bytes(data[24..28].try_into().unwrap()) != num_docs
    {
        return Err(invalid(
            "ordinal lookup belongs to another segment generation",
        ));
    }
    let fields = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;
    let mut expected_offset = fields
        .checked_mul(FIELD_BYTES)
        .and_then(|n| n.checked_add(HEADER_BYTES))
        .filter(|&n| n <= data.len())
        .ok_or_else(|| invalid("ordinal lookup field table is truncated"))?;
    let mut result = Vec::with_capacity(fields);
    let mut previous_field = None;
    for field_index in 0..fields {
        let start = HEADER_BYTES + field_index * FIELD_BYTES;
        let entry = &data[start..start + FIELD_BYTES];
        let field = u32::from_le_bytes(entry[0..4].try_into().unwrap());
        if previous_field.is_some_and(|previous| previous >= field) {
            return Err(invalid("ordinal lookup fields are not unique and sorted"));
        }
        previous_field = Some(field);
        let kind = match u32::from_le_bytes(entry[4..8].try_into().unwrap()) {
            1 => LookupKind::ChunkedText,
            2 => LookupKind::SparseBmp,
            value => return Err(invalid(format!("unknown ordinal lookup kind {value}"))),
        };
        let count = u32::from_le_bytes(entry[8..12].try_into().unwrap()) as usize;
        let physical_slots = u32::from_le_bytes(entry[12..16].try_into().unwrap());
        let offset = usize::try_from(u64::from_le_bytes(entry[16..24].try_into().unwrap()))
            .map_err(|_| invalid("ordinal lookup offset exceeds address space"))?;
        if offset != expected_offset {
            return Err(invalid("ordinal lookup sections overlap or have gaps"));
        }
        let end = count
            .checked_mul(ROW_BYTES)
            .and_then(|n| offset.checked_add(n))
            .filter(|&n| n <= data.len())
            .ok_or_else(|| invalid("ordinal lookup rows are truncated"))?;
        let lookup = OrdinalLookup {
            field,
            kind,
            physical_slots,
            rows: bytes.slice(offset..end),
        };
        let mut previous = None;
        for (index, (unit, slot)) in lookup.rows().enumerate() {
            if unit.doc >= num_docs
                || slot >= physical_slots
                || previous.is_some_and(|previous| previous >= unit)
                || data[offset + index * ROW_BYTES + 6..offset + index * ROW_BYTES + 8] != [0, 0]
            {
                return Err(invalid(
                    "ordinal lookup has invalid, duplicate or unordered rows",
                ));
            }
            previous = Some(unit);
        }
        result.push(lookup);
        expected_offset = end;
    }
    if expected_offset != data.len() {
        return Err(invalid("ordinal lookup has trailing data"));
    }
    Ok(result)
}

#[cfg(test)]
pub(crate) struct LookupSection<'a> {
    pub field: u32,
    pub kind: LookupKind,
    pub physical_slots: u32,
    /// Sorted logical key order. Preparing this permutation is a bounded
    /// build/reorder operation; normal merges stream existing rows with offsets.
    pub slots: &'a [u32],
    pub resolve: &'a (dyn Fn(u32) -> LogicalUnit + Send + Sync),
}

#[cfg(test)]
pub(crate) fn write_lookups(
    writer: &mut dyn Write,
    segment: u128,
    num_docs: u32,
    sections: &[LookupSection<'_>],
) -> io::Result<()> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
    let mut previous_field = None;
    // Validate every input before emitting any bytes.
    for section in sections {
        if previous_field.is_some_and(|previous| previous >= section.field) {
            return Err(invalid("lookup fields must be unique and sorted"));
        }
        previous_field = Some(section.field);
        u32::try_from(section.slots.len()).map_err(|_| invalid("too many ordinal lookup rows"))?;
        let mut previous = None;
        for &slot in section.slots {
            if slot >= section.physical_slots {
                return Err(invalid("lookup physical slot out of range"));
            }
            let unit = (section.resolve)(slot);
            if unit.doc >= num_docs || previous.is_some_and(|previous| previous >= unit) {
                return Err(invalid(
                    "lookup logical keys must be valid, unique and sorted",
                ));
            }
            previous = Some(unit);
        }
    }
    let streamed: Vec<_> = sections
        .iter()
        .map(|section| StreamSection {
            field: section.field,
            kind: section.kind,
            physical_slots: section.physical_slots,
            count: section.slots.len() as u32,
            emit: Box::new(move |visit| {
                for &slot in section.slots {
                    visit((section.resolve)(slot), slot)?;
                }
                Ok(())
            }),
        })
        .collect();
    write_streamed(writer, segment, num_docs, &streamed)
}

// One format writer serves reorder preparation and streaming merge/copy.
// Scratch is owned by each source and released before the next field.
#[cfg(all(not(target_arch = "wasm32"), any(feature = "native", test)))]
type EmitRows<'a> = Box<
    dyn Fn(&mut dyn FnMut(LogicalUnit, u32) -> io::Result<()>) -> io::Result<()> + Send + Sync + 'a,
>;
#[cfg(all(target_arch = "wasm32", any(feature = "native", test)))]
type EmitRows<'a> =
    Box<dyn Fn(&mut dyn FnMut(LogicalUnit, u32) -> io::Result<()>) -> io::Result<()> + 'a>;
#[cfg(any(feature = "native", test))]
struct StreamSection<'a> {
    field: u32,
    kind: LookupKind,
    physical_slots: u32,
    count: u32,
    emit: EmitRows<'a>,
}
#[cfg(any(feature = "native", test))]
fn write_streamed(
    writer: &mut dyn Write,
    segment: u128,
    num_docs: u32,
    sections: &[StreamSection<'_>],
) -> io::Result<()> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidData, message);
    let mut offset = HEADER_BYTES as u64 + sections.len() as u64 * FIELD_BYTES as u64;
    if !sections.windows(2).all(|p| p[0].field < p[1].field) {
        return Err(invalid("lookup fields are not unique and sorted"));
    }
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&segment.to_le_bytes())?;
    writer.write_all(&num_docs.to_le_bytes())?;
    writer.write_all(&(sections.len() as u32).to_le_bytes())?;
    for section in sections {
        writer.write_all(&section.field.to_le_bytes())?;
        writer.write_all(&(section.kind as u32).to_le_bytes())?;
        writer.write_all(&section.count.to_le_bytes())?;
        writer.write_all(&section.physical_slots.to_le_bytes())?;
        writer.write_all(&offset.to_le_bytes())?;
        offset = offset
            .checked_add(u64::from(section.count) * ROW_BYTES as u64)
            .ok_or_else(|| invalid("lookup size overflow"))?;
    }
    for section in sections {
        let mut previous = None;
        let mut count = 0u32;
        (section.emit)(&mut |unit, physical| {
            if unit.doc >= num_docs
                || physical >= section.physical_slots
                || previous.is_some_and(|p| p >= unit)
            {
                return Err(invalid("invalid, duplicate or unordered lookup row"));
            }
            count = count
                .checked_add(1)
                .ok_or_else(|| invalid("lookup count overflow"))?;
            if count > section.count {
                return Err(invalid("too many lookup rows"));
            }
            previous = Some(unit);
            writer.write_all(&unit.doc.to_le_bytes())?;
            writer.write_all(&unit.ordinal.to_le_bytes())?;
            writer.write_all(&[0, 0])?;
            writer.write_all(&physical.to_le_bytes())
        })?;
        if count != section.count {
            return Err(invalid("missing lookup rows"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        // Different field-local permutations must resolve the same logical unit.
        let sparse = [(1, 0), (0, 2), (0, 0), (2, 1)];
        let text = [(0, 0), (0, 2), (1, 0), (2, 1)];
        let sparse_unit = |slot: u32| {
            let (doc, ordinal) = sparse[slot as usize];
            LogicalUnit { doc, ordinal }
        };
        let text_unit = |slot: u32| {
            let (doc, ordinal) = text[slot as usize];
            LogicalUnit { doc, ordinal }
        };
        let mut out = Vec::new();
        write_lookups(
            &mut out,
            42,
            4,
            &[
                LookupSection {
                    field: 1,
                    kind: LookupKind::SparseBmp,
                    physical_slots: 4,
                    slots: &[2, 1, 0, 3],
                    resolve: &sparse_unit,
                },
                LookupSection {
                    field: 2,
                    kind: LookupKind::ChunkedText,
                    physical_slots: 4,
                    slots: &[0, 1, 2, 3],
                    resolve: &text_unit,
                },
            ],
        )
        .unwrap();
        out
    }

    #[test]
    fn ordered_lookup_handles_internal_padding_missing_docs_and_sparse_ordinals() {
        let units = [
            None,
            Some(LogicalUnit { doc: 0, ordinal: 0 }),
            Some(LogicalUnit { doc: 0, ordinal: 2 }),
            None,
            None,
            Some(LogicalUnit { doc: 2, ordinal: 1 }),
            None,
        ];
        assert!(logically_ordered(units));
        for (doc, expected) in [
            (0, vec![(0, 1), (2, 2)]),
            (1, vec![]),
            (2, vec![(1, 5)]),
            (3, vec![]),
        ] {
            assert_eq!(
                ordered_document_slots(units.len() as u32, doc, |slot| units[slot as usize])
                    .collect::<Vec<_>>(),
                expected
            );
        }
        assert!(!logically_ordered([units[5], units[1]]));
        assert!(!logically_ordered([units[1], units[1]]));
    }

    #[test]
    fn field_local_reordering_preserves_logical_chunk_identity() {
        let fields = read_lookups(OwnedBytes::new(fixture()), 42, 4).unwrap();
        assert_eq!(
            fields[0].for_document(0).collect::<Vec<_>>(),
            [(0, 2), (2, 1)]
        );
        assert_eq!(
            fields[1].for_document(0).collect::<Vec<_>>(),
            [(0, 0), (2, 1)]
        );
        assert_eq!(fields[0].for_document(1).collect::<Vec<_>>(), [(0, 0)]);
        assert_eq!(fields[0].for_document(3).count(), 0);
        assert_eq!(fields[0].for_document(u32::MAX).count(), 0);
    }

    #[test]
    fn corrupt_or_foreign_lookup_cannot_silently_score_another_chunk() {
        let source = fixture();
        assert!(read_lookups(OwnedBytes::new(source.clone()), 43, 4).is_err());
        for length in [0, 31, source.len() - 1] {
            assert!(read_lookups(OwnedBytes::new(source[..length].to_vec()), 42, 4).is_err());
        }
        for (offset, value) in [
            (4, 2),
            (HEADER_BYTES + 4, 7),
            (HEADER_BYTES + 2 * FIELD_BYTES, 8),
            (HEADER_BYTES + 2 * FIELD_BYTES + 6, 1),
        ] {
            let mut broken = source.clone();
            broken[offset] = value;
            assert!(
                read_lookups(OwnedBytes::new(broken), 42, 4).is_err(),
                "offset {offset}"
            );
        }
    }

    #[test]
    fn writer_rejects_duplicate_logical_units_before_writing() {
        let resolve = |_| LogicalUnit { doc: 0, ordinal: 0 };
        let mut out = Vec::new();
        assert!(
            write_lookups(
                &mut out,
                42,
                1,
                &[LookupSection {
                    field: 1,
                    kind: LookupKind::SparseBmp,
                    physical_slots: 2,
                    slots: &[0, 1],
                    resolve: &resolve,
                }]
            )
            .is_err()
        );
        assert!(out.is_empty());
    }
}
