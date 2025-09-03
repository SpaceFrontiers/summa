//! Logical document/ordinal identities and ordered-map search.
use crate::DocId;

/// A logical unit is a document ID and its actual field value ordinal.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct LogicalUnit {
    pub doc: DocId,
    pub ordinal: u16,
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
