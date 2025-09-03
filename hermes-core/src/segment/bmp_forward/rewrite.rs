//! Representation-preserving merge/reorder, with explicit budgeted V19 upgrade.
use super::*;
use crate::segment::{BmpIndex, OffsetWriter};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

fn cancelled(cancellation: Option<&AtomicBool>) -> Result<()> {
    if cancellation.is_some_and(|c| c.load(Ordering::Acquire)) {
        Err(Error::IndexClosed)
    } else {
        Ok(())
    }
}

struct SourcePlan<'a> {
    bmp: &'a BmpIndex,
    doc_offset: u32,
    payload_offset: u64,
    /// Only the explicit V19 migration constructs a physical permutation.
    legacy_slots: Vec<u32>,
    legacy_offsets: Vec<u64>,
}

pub(crate) fn validate_copy_sources(sources: &[(&BmpIndex, u32)]) -> Result<bool> {
    let legacy = sources
        .iter()
        .filter(|(bmp, _)| bmp.forward().is_none())
        .count();
    if legacy != 0 && legacy != sources.len() {
        return Err(Error::Schema("cannot copy-merge BMP V19 and V20; explicitly reorder the V19 segments to add forward values first".into()));
    }
    Ok(legacy == 0)
}

/// Returns whether a V20 section was written. Ordinary all-V19 merge remains
/// V19. Mixed ordinary merges refuse to rebuild or discard a forward section.
pub(crate) fn write_forward_sources(
    sources: &[(&BmpIndex, u32)],
    paths: &[Option<&std::path::Path>],
    writer: &mut OffsetWriter,
    upgrade_budget: Option<usize>,
    cancellation: Option<&AtomicBool>,
) -> Result<bool> {
    if upgrade_budget.is_none() && !validate_copy_sources(sources)? {
        return Ok(false);
    }
    let mut remaining = upgrade_budget.unwrap_or(0);
    let mut plans = Vec::with_capacity(sources.len());
    let mut count = 0u32;
    let mut previous_last = None;
    for &(bmp, doc_offset) in sources {
        cancelled(cancellation)?;
        count = count
            .checked_add(bmp.num_real_docs())
            .ok_or_else(|| corrupt("merged vector count overflows u32"))?;
        let mut slots = Vec::new();
        let mut offsets = Vec::new();
        if bmp.forward().is_none() {
            let bytes = (bmp.num_real_docs() as usize)
                .checked_mul(12)
                .ok_or_else(|| corrupt("migration directory size overflow"))?;
            remaining = remaining.checked_sub(bytes).ok_or_else(|| Error::Schema(
                "BMP V19 forward migration exceeds the reorder memory budget; increase bp-memory-budget-mb".into()))?;
            slots = Vec::with_capacity(bmp.num_real_docs() as usize);
            bmp.visit_real_slots_for_rewrite(|slot| slots.push(slot as u32))?;
            slots.sort_unstable_by_key(|&slot| bmp.virtual_to_doc(slot));
            offsets = Vec::with_capacity(slots.len());
            log::info!(
                "[bmp_forward] migrating {} V19 vectors; directory scratch={} bytes",
                slots.len(),
                bytes
            );
        }
        // Validate disjoint logical ranges before emitting payload bytes.
        let mut previous = previous_last;
        for i in 0..bmp.num_real_docs() {
            if i % 4096 == 0 {
                cancelled(cancellation)?;
            }
            let key = if let Some(forward) = bmp.forward() {
                forward.key(i)
            } else {
                let (doc, ordinal) = bmp.virtual_to_doc(slots[i as usize]);
                LogicalUnit { doc, ordinal }
            };
            let key = LogicalUnit {
                doc: key
                    .doc
                    .checked_add(doc_offset)
                    .ok_or_else(|| corrupt("document offset overflow"))?,
                ..key
            };
            if previous.is_some_and(|p| p >= key) {
                return Err(corrupt("duplicate or overlapping source keys"));
            }
            previous = Some(key);
        }
        previous_last = previous;
        plans.push(SourcePlan {
            bmp,
            doc_offset,
            payload_offset: 0,
            legacy_slots: slots,
            legacy_offsets: offsets,
        });
    }
    let start = writer.offset();
    for (i, plan) in plans.iter_mut().enumerate() {
        cancelled(cancellation)?;
        plan.payload_offset = writer.offset() - start;
        if let Some(forward) = plan.bmp.forward() {
            crate::segment::merger::copy_local_range_or_bytes(
                writer,
                paths.get(i).copied().flatten(),
                plan.bmp.forward_payload_file_range(),
                forward.payload.as_slice(),
                cancellation,
                "BMP forward payload",
            )?;
        } else {
            // An explicit one-time migration probes source blocks in logical
            // order. Only its directory is buffered; payload streams directly.
            for &slot in &plan.legacy_slots {
                cancelled(cancellation)?;
                plan.legacy_offsets
                    .push(writer.offset() - start - plan.payload_offset);
                let block_id = slot / plan.bmp.bmp_block_size;
                plan.bmp.validate_block_for_rewrite(block_id)?;
                let local = (slot % plan.bmp.bmp_block_size) as u8;
                let mut found = false;
                for (dim, _, postings) in plan.bmp.iter_block_terms(block_id) {
                    for posting in postings {
                        if posting.local_slot == local {
                            writer.write_all(&dim.to_le_bytes())?;
                            writer.write_all(&[posting.impact])?;
                            found = true;
                        }
                    }
                }
                if !found {
                    return Err(corrupt("real legacy vector has no postings"));
                }
            }
        }
    }
    let payload_bytes = writer.offset() - start;
    let rows = plans.iter().flat_map(|plan| {
        (0..plan.bmp.num_real_docs()).map(move |i| {
            if cancellation.is_some_and(|c| c.load(Ordering::Acquire)) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "BMP forward write cancelled",
                ));
            }
            let (key, offset) = if let Some(forward) = plan.bmp.forward() {
                (forward.key(i), forward.offset(i))
            } else {
                let (doc, ordinal) = plan.bmp.virtual_to_doc(plan.legacy_slots[i as usize]);
                (
                    LogicalUnit { doc, ordinal },
                    plan.legacy_offsets[i as usize],
                )
            };
            Ok((
                LogicalUnit {
                    doc: key.doc + plan.doc_offset,
                    ..key
                },
                offset + plan.payload_offset,
            ))
        })
    });
    let result = write_directory(writer, rows, count, payload_bytes);
    cancelled(cancellation)?;
    result?;
    Ok(true)
}
