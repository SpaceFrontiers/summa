//! Explicit bounded consolidation and deletion rewrite; ordinary merge never
//! calls these routines. Forward payloads and untouched nominations are copied.
use super::*;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

// Keep continuation, membership and cancellation separate: they have different
// lifecycle semantics and must not be collapsed into an error-returning gate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_maintained_partition(
    index: &SeismicIndex,
    partition: usize,
    config: &SparseVectorConfig,
    should_continue: &impl Fn() -> bool,
    memory_budget: usize,
    is_alive: &impl Fn(u32) -> bool,
    check_cancel: &impl Fn() -> Result<()>,
    writer: &mut dyn Write,
) -> Result<(u64, u32)> {
    check_cancel()?;
    config.seismic.validate().map_err(Error::Schema)?;
    if config.weight_quantization != index.quantization
        || config.seismic.postings != index.postings as usize
        || config.seismic.cluster_size != index.cluster_size as usize
        || config.seismic.summary_energy.to_bits() != index.summary_energy.to_bits()
    {
        return Err(Error::Schema(
            "Seismic maintenance settings differ from the encoded generation; rebuild explicitly"
                .into(),
        ));
    }
    let part = index
        .partitions
        .get(partition)
        .and_then(Option::as_ref)
        .ok_or_else(|| corrupt("missing nomination partition"))?;
    if part.pending_terms == 0 {
        let bytes = write_partition_sources(&[(index, 0)], partition, writer, check_cancel)?;
        return Ok((bytes, part.pending_terms));
    }
    let directory_scratch = part
        .runs
        .iter()
        .try_fold(0usize, |sum, r| {
            sum.checked_add((r.terms as usize).checked_mul(256)?)
        })
        .ok_or_else(|| corrupt("maintenance directory overflow"))?;
    if directory_scratch > memory_budget {
        let bytes = write_partition_sources(&[(index, 0)], partition, writer, check_cancel)?;
        return Ok((bytes, part.pending_terms));
    }
    let memory_budget = memory_budget - directory_scratch;
    let mut terms: BTreeMap<u32, Vec<(&Run, usize)>> = BTreeMap::new();
    for r in &part.runs {
        for i in 0..r.terms as usize {
            let at = i * TERM_ENTRY;
            terms
                .entry(u32_at(&r.term_directory, at))
                .or_default()
                .push((r, at));
        }
    }
    let mut priorities: Vec<_> = terms
        .iter()
        .filter(|(_, runs)| runs.len() > 1)
        .map(|(&dim, runs)| {
            let bytes = runs
                .iter()
                .map(|(run, at)| u64_at(&run.term_directory, at + 16))
                .sum::<u64>();
            (Reverse(runs.len()), Reverse(bytes), dim)
        })
        .collect();
    priorities.sort_unstable();
    let order: Vec<_> = priorities
        .into_iter()
        .map(|p| p.2)
        .chain(
            terms
                .iter()
                .filter_map(|(&dim, runs)| (runs.len() == 1).then_some(dim)),
        )
        .collect();
    let mut offset = 0u64;
    let mut entries = Vec::new();
    let mut remaining = 0;
    let mut has_time = true;
    for dim in order {
        let old = terms.remove(&dim).expect("ordered nomination term exists");
        check_cancel()?;
        // Complete one admitted term at a time. Once the deadline expires,
        // retain every remaining encoded term without starting more clustering.
        if old.len() > 1 && has_time {
            has_time = should_continue();
        }
        let rebuilt = if old.len() > 1 && has_time {
            rebuild_term(
                index,
                dim,
                &old,
                config,
                memory_budget,
                is_alive,
                check_cancel,
            )?
        } else {
            None
        };
        check_cancel()?;
        if let Some((bytes, clusters)) = rebuilt {
            entries.push((
                dim,
                clusters,
                offset,
                bytes.len() as u64,
                0,
                config.seismic.postings as u32,
                old.iter()
                    .map(|(r, at)| u32_at(&r.term_directory, at + 32))
                    .sum::<u32>(),
            ));
            writer.write_all(&bytes)?;
            offset += bytes.len() as u64;
        } else {
            if old.len() > 1 {
                remaining += 1;
            }
            for (r, at) in old {
                let start = u64_at(&r.term_directory, at + 8) as usize;
                let len = u64_at(&r.term_directory, at + 16) as usize;
                let base = r
                    .row_base
                    .checked_add(u32_at(&r.term_directory, at + 24))
                    .ok_or_else(|| corrupt("term row base overflow"))?;
                entries.push((
                    dim,
                    u32_at(&r.term_directory, at + 4),
                    offset,
                    len as u64,
                    base,
                    u32_at(&r.term_directory, at + 28),
                    u32_at(&r.term_directory, at + 32),
                ));
                for chunk in r.bytes[start..start + len].chunks(4 * 1024 * 1024) {
                    check_cancel()?;
                    writer.write_all(chunk)?;
                }
                offset += len as u64;
            }
        }
    }
    check_cancel()?;
    entries.sort_unstable_by_key(|entry| (entry.0, entry.4));
    let term_offset = offset;
    for &(dim, clusters, at, len, base, coverage, frequency) in &entries {
        put32(writer, dim)?;
        put32(writer, clusters)?;
        put64(writer, at)?;
        put64(writer, len)?;
        put32(writer, base)?;
        put32(writer, coverage)?;
        put32(writer, frequency)?;
        put32(writer, 0)?;
        offset += TERM_ENTRY as u64;
    }
    let bytes = finish_run(
        writer,
        offset,
        0,
        term_offset,
        entries.len() as u32,
        index,
        Some(partition),
    )?;
    Ok((bytes, remaining))
}

fn rebuild_term(
    index: &SeismicIndex,
    dim: u32,
    old: &[(&Run, usize)],
    config: &SparseVectorConfig,
    budget: usize,
    is_alive: &impl Fn(u32) -> bool,
    check_cancel: &impl Fn() -> Result<()>,
) -> Result<Option<(Vec<u8>, u32)>> {
    let cap = config.seismic.postings;
    let minimum = cap
        .checked_mul(128)
        .ok_or_else(|| corrupt("maintenance size overflow"))?;
    if minimum > budget {
        return Ok(None);
    }
    let complete = old
        .iter()
        .all(|(run, at)| u32_at(&run.term_directory, at + 28) as usize >= cap)
        && index.term_runs(dim).all(|term| {
            term.clusters()
                .all(|cluster| cluster.rows().all(|row| is_alive(index.key(row).doc)))
        });
    let mut heap = BinaryHeap::with_capacity(cap);
    let mut visit = |row: u32| -> Result<()> {
        check_cancel()?;
        if !is_alive(index.key(row).doc) {
            return Ok(());
        }
        let weight = index
            .vector(row)
            .iter()
            .find(|p| p.0 >= dim)
            .filter(|p| p.0 == dim)
            .map_or(0.0, |p| p.1.abs());
        if weight == 0.0 {
            return Ok(());
        }
        let value = (weight.to_bits(), u32::MAX - row);
        if heap.len() < cap {
            heap.push(Reverse(value));
        } else if heap.peek().is_some_and(|p| value > p.0) {
            *heap.peek_mut().unwrap() = Reverse(value);
        }
        Ok(())
    };
    if complete {
        for term in index.term_runs(dim) {
            for cluster in term.clusters() {
                for row in cluster.rows() {
                    visit(row)?;
                }
            }
        }
    } else {
        log::debug!(
            "Seismic maintenance term {dim}: scanning forward values for incomplete/deleted nomination coverage"
        );
        for row in 0..index.len() {
            visit(row)?;
        }
    }
    let mut selected: Vec<_> = heap.into_iter().map(|p| u32::MAX - p.0.1).collect();
    selected.sort_unstable();
    let mut estimated = minimum;
    for &row in &selected {
        estimated = estimated
            .checked_add(vector_scratch(index, row)?)
            .ok_or_else(|| corrupt("maintenance size overflow"))?;
        if estimated > budget {
            return Ok(None);
        }
    }
    let rows: Vec<Vec<_>> = selected
        .iter()
        .map(|&r| index.vector(r).iter().collect())
        .collect();
    let assignment = build::AssignmentCoordinates::prepare(&rows);
    let local: Vec<_> = (0..rows.len() as u32).collect();
    let (mut bytes, clusters) = build::encode_term(
        &local,
        &rows,
        &assignment,
        config.seismic.cluster_size,
        config.seismic.summary_energy,
    )?;
    term::remap_rows(&mut bytes, &selected);
    Ok(Some((bytes, clusters)))
}

// Row vectors, centroid dictionaries, posting-vector allocation slack,
// coordinate maxima, the bounded transpose and output summaries coexist. Account for singleton
// clusters and dimensions as well as the common shared-vocabulary case.
fn vector_scratch(index: &SeismicIndex, row: u32) -> Result<usize> {
    index
        .vector(row)
        .len()
        .checked_mul(256)
        .and_then(|n| n.checked_add(128 + build::AssignmentCoordinates::BYTES_PER_ROW))
        .ok_or_else(|| corrupt("vector scratch estimate overflow"))
}

// Unlike a selected-term pass, compaction's finish_blob also retains the
// complete dimension -> candidate-list BTreeMap and the term directory while
// encode_term owns its clustering scratch. Bound distinct dimensions by the
// total coordinate count. Charge three map-entry widths for node occupancy and
// links, four candidate slots for the smallest Vec allocation, one output
// directory entry, and allocator slack. On 64-bit hosts this adds 192 bytes per
// coordinate to vector_scratch's 256; repeated dimensions only lower the cost.
fn compaction_vector_scratch(index: &SeismicIndex, row: u32) -> Result<usize> {
    let global_per_coordinate = 3 * std::mem::size_of::<(u32, Vec<(u32, f32)>)>()
        + 4 * std::mem::size_of::<(u32, f32)>()
        + std::mem::size_of::<(u32, u32, u64, u64, u32)>()
        + 32;
    vector_scratch(index, row)?
        .checked_add(
            index
                .vector(row)
                .len()
                .checked_mul(global_per_coordinate)
                .ok_or_else(|| corrupt("compaction scratch estimate overflow"))?,
        )
        .ok_or_else(|| corrupt("compaction scratch estimate overflow"))
}

/// Deletion compaction is an explicit rebuild, admitted before cloning values.
/// Nomination selection sees every retained forward row, enabling backfill.
pub(crate) fn write_compacted(
    index: &SeismicIndex,
    doc_map: &impl Fn(u32) -> Option<u32>,
    config: &SparseVectorConfig,
    memory_budget: usize,
    check_cancel: &impl Fn() -> Result<()>,
    writer: &mut impl SeismicWriter,
) -> Result<OutputLengths> {
    if config.weight_quantization != index.quantization {
        return Err(Error::Schema(
            "Seismic compaction cannot change forward precision".into(),
        ));
    }
    let mut estimated = 0usize;
    for row in 0..index.len() {
        if row % 4096 == 0 {
            check_cancel()?;
        }
        if doc_map(index.key(row).doc).is_some() {
            estimated = estimated
                .checked_add(compaction_vector_scratch(index, row)?)
                .ok_or_else(|| corrupt("compaction scratch overflow"))?;
            if estimated > memory_budget {
                return Err(Error::Schema(
                    "Seismic compaction exceeds its memory budget".into(),
                ));
            }
        }
    }
    let mut rows = Vec::new();
    let mut directory = Vec::new();
    let mut offset = 0u64;
    let mut previous = None;
    for row in 0..index.len() {
        check_cancel()?;
        let key = index.key(row);
        if let Some(doc) = doc_map(key.doc) {
            let key = LogicalUnit {
                doc,
                ordinal: key.ordinal,
            };
            if previous.is_some_and(|old| old >= key) {
                return Err(corrupt("compaction map is not ordered"));
            }
            previous = Some(key);
            let vector = index.vector(row);
            for chunk in vector.bytes.chunks(4 * 1024 * 1024) {
                check_cancel()?;
                writer.root().write_all(chunk)?;
            }
            directory.push((
                key,
                offset,
                vector.byte_len() as u32,
                vector.len() as u32,
                vector.encoding,
            ));
            offset += vector.byte_len() as u64;
            rows.push(vector.iter().collect());
        }
    }
    build::finish_blob(rows, directory, offset, index.dims(), config, writer)
}
