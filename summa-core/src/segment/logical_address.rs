//! Logical document/ordinal identities and ordered-map search.
use crate::DocId;

/// A logical unit is a document ID and its actual field value ordinal.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct LogicalUnit {
    pub doc: DocId,
    pub ordinal: u16,
}

/// First position in `0..len` whose document is at or after `doc`, for
/// positions sorted by document.
///
/// Values per document vary, but positions stay roughly proportional to
/// document ids, so a secant estimate lands near the answer. A plain bisection
/// of a directory far larger than memory reads a distinct cold page for most
/// of its upper levels on every lookup; this search usually reads two or
/// three. An interpolation probe that fails to halve the interval is followed
/// by a bisection probe, which bounds the work at `2 * log2(len)` probes.
pub(crate) fn first_position_of_document(
    len: usize,
    doc: DocId,
    doc_at: impl Fn(usize) -> DocId,
) -> usize {
    if len == 0 {
        return 0;
    }
    let target = u64::from(doc);
    // Documents at positions below `low` are before the target and at least
    // `low_doc`; documents from `high` on are at or after it, starting at
    // `high_doc`.
    let (mut low, mut high) = (0usize, len);
    let (mut low_doc, mut high_doc) = (0u64, u64::from(doc_at(len - 1)) + 1);
    if target >= high_doc {
        return len;
    }
    let mut bisect = false;
    while low < high {
        let span = high - low;
        let probe = if bisect || span <= 8 || high_doc <= low_doc {
            low + span / 2
        } else {
            let offset = (u128::from(target - low_doc) * span as u128
                / u128::from(high_doc - low_doc)) as usize;
            low + offset.min(span - 1)
        };
        let found = u64::from(doc_at(probe));
        if found < target {
            low = probe + 1;
            low_doc = found;
        } else {
            high = probe;
            high_doc = found;
        }
        bisect = !bisect && (high - low) * 2 > span;
    }
    low
}

/// Check order while ignoring unoccupied slots. No inverse allocation is needed
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// Positions of documents holding `values(doc)` values each.
    fn directory(docs: u32, values: impl Fn(u32) -> usize) -> Vec<DocId> {
        (0..docs)
            .flat_map(|doc| std::iter::repeat_n(doc, values(doc)))
            .collect()
    }

    #[test]
    fn interpolated_document_search_matches_bisection_on_skewed_directories() {
        let shapes: [(&str, Vec<DocId>); 5] = [
            ("empty", Vec::new()),
            ("single", vec![7]),
            ("uniform", directory(5_000, |_| 3)),
            // Mostly short documents, a few books, and documents without values.
            (
                "skewed",
                directory(20_000, |doc| match doc % 997 {
                    0 => 4_000,
                    1..=40 => 0,
                    _ => 1 + (doc as usize * 7919) % 5,
                }),
            ),
            // One giant document at the front bends the secant the other way.
            (
                "front-loaded",
                directory(3_000, |doc| if doc == 2 { 90_000 } else { 1 }),
            ),
        ];
        for (name, rows) in &shapes {
            let last = rows.last().copied().unwrap_or(0);
            let targets = (0..=last.saturating_add(2)).step_by(1 + rows.len() / 4_000);
            let (mut probes, mut lookups) = (0u64, 0u64);
            for doc in targets.chain([last, last.saturating_add(1), u32::MAX]) {
                let count = Cell::new(0u64);
                let found = first_position_of_document(rows.len(), doc, |position| {
                    count.set(count.get() + 1);
                    rows[position]
                });
                assert_eq!(
                    found,
                    rows.partition_point(|&row| row < doc),
                    "{name}: {doc}"
                );
                let bound = 2 * (usize::BITS - rows.len().leading_zeros()) as u64 + 3;
                assert!(count.get() <= bound, "{name}: {} probes", count.get());
                probes += count.get();
                lookups += 1;
            }
            // A near-proportional directory resolves in far fewer probes than
            // the ~log2(len) a bisection needs.
            if *name == "uniform" {
                assert!(probes <= 5 * lookups, "{name}: {probes} probes / {lookups}");
            }
        }
    }
}
