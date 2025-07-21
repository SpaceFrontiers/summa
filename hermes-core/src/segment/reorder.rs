//! Standalone segment reorder via Recursive Graph Bisection (BP).
//!
//! Copies unchanged segment files (postings, store, fast, dense vectors)
//! and rebuilds the sparse file with reordered BMP blocks. Non-BMP sparse
//! fields (MaxScore) are identity-copied.
//!
//! Background optimization is decoupled from the merge path. Merge-time
//! reordering is optional; when enabled, both paths share the same bounded
//! CPU pool and whole-pass concurrency gate.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::{FieldType, Schema};
use crate::segment::format::{SparseFieldToc, write_sparse_toc_and_footer};
use crate::segment::reader::SegmentReader;
use crate::segment::types::{SegmentFiles, SegmentId, SegmentMeta};
use crate::segment::{OffsetWriter, SegmentMerger, TrainedVectorStructures};
use crate::structures::SparseFormat;

/// Default memory budget for forward index during BP (24 GB). A cap, not an
/// allocation — usage is proportional to the segment being reordered
/// (~4 B/posting + ~32 B/doc). Sized from prod evidence: a 58M-doc /
/// 5B-posting pass estimated 20.1 GB, which smaller budgets trimmed by
/// dropping highest-df dims. Mirrored by
/// `IndexConfig::default().bp_memory_budget_bytes`.
pub const DEFAULT_MEMORY_BUDGET: usize = 24 * 1024 * 1024 * 1024;

fn cancellation_requested(cancellation: Option<&std::sync::atomic::AtomicBool>) -> bool {
    cancellation.is_some_and(|cancelled| cancelled.load(std::sync::atomic::Ordering::Acquire))
}

/// Unique prefix for a pass's spilled grid-run temp files.
///
/// MUST be unique per pass, not per (process, field): in a container the
/// server is always PID 1 and every index's sparse field tends to share the
/// same field id, so the old `pid_field` scheme collided across concurrent
/// reorders (documents + social both spilling field 3) — passes overwrote
/// each other's runs and deleted them on completion, failing the other pass
/// with ENOENT after it had already copied tens of GB (orphaned outputs
/// filled the production disk overnight).
fn grid_run_prefix(field_id: u32) -> String {
    format!(
        "hermes_grid_run_{}_{}_{}",
        std::process::id(),
        field_id,
        SegmentId::new().to_hex(),
    )
}

/// Owns spill paths from before their first write. Drop removes complete and
/// partially-written files on success, `?`, and panic unwind. UUID prefixes
/// also avoid colliding with leftovers from a prior process using the same
/// PID (common for PID 1 in containers).
struct GridScratchFiles {
    directory: PathBuf,
    prefix: String,
    run_paths: Vec<PathBuf>,
    auxiliary_paths: Vec<PathBuf>,
    all_paths: Vec<PathBuf>,
    next_run_id: usize,
}

impl GridScratchFiles {
    fn new(field_id: u32, output_path: PathBuf) -> Self {
        let directory = output_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(std::env::temp_dir);
        let output_name = output_path
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_else(|| "hermes".into());
        Self {
            directory,
            prefix: format!("{output_name}.grid_run_{}", grid_run_prefix(field_id)),
            run_paths: Vec::new(),
            auxiliary_paths: Vec::new(),
            all_paths: Vec::new(),
            next_run_id: 0,
        }
    }

    fn allocate_run(&mut self) -> PathBuf {
        let path = self
            .directory
            .join(format!("{}_run_{}.tmp", self.prefix, self.next_run_id));
        self.next_run_id += 1;
        self.run_paths.push(path.clone());
        self.all_paths.push(path.clone());
        path
    }

    fn allocate_auxiliary(&mut self, name: &str) -> PathBuf {
        let path = self.directory.join(format!(
            "{}_{}_{}.tmp",
            self.prefix,
            name,
            self.auxiliary_paths.len()
        ));
        self.auxiliary_paths.push(path.clone());
        self.all_paths.push(path.clone());
        path
    }

    fn runs_are_empty(&self) -> bool {
        self.run_paths.is_empty()
    }

    fn num_runs(&self) -> usize {
        self.run_paths.len()
    }

    fn runs(&self) -> impl Iterator<Item = &PathBuf> {
        self.run_paths.iter()
    }

    fn consolidate_runs(&mut self, max_fan_in: usize) -> Result<()> {
        while self.run_paths.len() > max_fan_in {
            let inputs = std::mem::take(&mut self.run_paths);
            for group in inputs.chunks(max_fan_in) {
                if group.len() == 1 {
                    self.run_paths.push(group[0].clone());
                    continue;
                }
                let output = self.allocate_run();
                crate::segment::builder::bmp::merge_grid_runs(group, &output)
                    .map_err(crate::Error::Io)?;
                // Retire each input group as soon as its replacement exists.
                // Waiting for a complete new generation temporarily doubled
                // multi-GB scratch usage.
                for input in group {
                    match std::fs::remove_file(input) {
                        Ok(()) => {}
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => return Err(crate::Error::Io(error)),
                    }
                }
            }
        }
        Ok(())
    }
}

impl Drop for GridScratchFiles {
    fn drop(&mut self) {
        for path in &self.all_paths {
            if let Err(error) = std::fs::remove_file(path)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                log::warn!("[reorder_bmp] failed to remove spill {:?}: {}", path, error);
            }
        }
    }
}

type BmpGridEntry = (u32, u32, u8);
const MAX_GRID_RUN_ENTRIES: usize = 16 * 1024 * 1024;
const REWRITE_IO_BUFFER_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy)]
struct RecordRoute {
    source: u32,
    old_block: u32,
    out_local: u32,
    old_slot: u8,
    new_slot: u8,
}

#[derive(Clone, Copy)]
struct SourceBlockJob {
    source: u32,
    old_block: u32,
    route_start: u32,
    route_end: u32,
}

#[derive(Clone, Copy, Default)]
struct RoutedPosting {
    out_local: u32,
    dimension: u32,
    slot: u8,
    impact: u8,
}

fn spill_grid_entries(
    entries: &mut Vec<BmpGridEntry>,
    scratch: &mut GridScratchFiles,
    field_id: u32,
    rayon_pool: Option<&rayon::ThreadPool>,
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    use rayon::prelude::*;
    install_on_pool(rayon_pool, || entries.par_sort_unstable());
    let path = scratch.allocate_run();
    crate::segment::builder::bmp::write_grid_run(entries, &path).map_err(crate::Error::Io)?;
    entries.clear();
    log::debug!(
        "[reorder_bmp] field {}: spilled grid run {} to disk",
        field_id,
        scratch.num_runs(),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_reorder_grids(
    mut entries: Vec<BmpGridEntry>,
    scratch: &mut GridScratchFiles,
    field_id: u32,
    dims: usize,
    num_blocks: usize,
    grid_bits: u8,
    rayon_pool: Option<&rayon::ThreadPool>,
    writer: &mut OffsetWriter,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> Result<(u64, u64, u64)> {
    use crate::segment::builder::bmp::{
        GridRunReader, stream_write_grids, stream_write_grids_merged,
    };

    if scratch.runs_are_empty() {
        use rayon::prelude::*;
        if cancellation_requested(cancellation) {
            return Err(crate::Error::IndexClosed);
        }
        install_on_pool(rayon_pool, || entries.par_sort_unstable());
        let result =
            stream_write_grids(&entries, dims, num_blocks, grid_bits, writer, cancellation);
        return if cancellation_requested(cancellation) {
            Err(crate::Error::IndexClosed)
        } else {
            result.map_err(crate::Error::Io)
        };
    }

    if !entries.is_empty() {
        if cancellation_requested(cancellation) {
            return Err(crate::Error::IndexClosed);
        }
        spill_grid_entries(&mut entries, scratch, field_id, rayon_pool)?;
    }
    drop(entries);
    // Bound both file descriptors and BufReader memory. Extremely large
    // fields can produce hundreds of sorted runs even with 16M-entry chunks.
    scratch.consolidate_runs(64)?;
    if cancellation_requested(cancellation) {
        return Err(crate::Error::IndexClosed);
    }
    let mut readers: Vec<GridRunReader> = scratch
        .runs()
        .map(|path| GridRunReader::open(path).map_err(crate::Error::Io))
        .collect::<Result<_>>()?;
    // The hierarchy rows must follow the complete block-grid section in the
    // output. Spool only the much smaller E/H rows beside the index while the
    // second and final merged-run pass emits all three projections.
    let superblock_spool = scratch.allocate_auxiliary("superblock");
    let coarse_spool = scratch.allocate_auxiliary("coarse");
    let result = stream_write_grids_merged(
        &mut readers,
        dims,
        num_blocks,
        grid_bits,
        &superblock_spool,
        &coarse_spool,
        writer,
        cancellation,
    );
    // Close readers before GridScratchFiles removes the spill paths.
    drop(readers);
    if cancellation_requested(cancellation) {
        Err(crate::Error::IndexClosed)
    } else {
        result.map_err(crate::Error::Io)
    }
}

fn install_on_pool<T: Send>(
    pool: Option<&rayon::ThreadPool>,
    operation: impl FnOnce() -> T + Send,
) -> T {
    match pool {
        Some(pool) => pool.install(operation),
        None => operation(),
    }
}

#[inline]
fn resolve_real_vid(
    global_real: usize,
    real_base: &[usize],
    real_to_virtual: &[Vec<u32>],
) -> (usize, usize) {
    let source = if real_to_virtual.len() == 1 {
        0
    } else {
        real_base.partition_point(|&base| base <= global_real) - 1
    };
    (
        source,
        real_to_virtual[source][global_real - real_base[source]] as usize,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_record_route_plan(
    sources: &[(crate::segment::BmpIndex, u32)],
    permutation: &[u32],
    real_base: &[usize],
    real_to_virtual: &[Vec<u32>],
    block_size: usize,
    num_real_docs: usize,
    window_start: usize,
    window_end: usize,
    rayon_pool: Option<&rayon::ThreadPool>,
) -> Result<(Vec<RecordRoute>, Vec<SourceBlockJob>)> {
    use rayon::prelude::*;

    let first_record = window_start
        .checked_mul(block_size)
        .ok_or_else(|| crate::Error::Internal("BMP route window start overflows usize".into()))?;
    let last_record = window_end
        .checked_mul(block_size)
        .map(|end| end.min(num_real_docs))
        .ok_or_else(|| crate::Error::Internal("BMP route window end overflows usize".into()))?;
    let mut routes = Vec::with_capacity(last_record.saturating_sub(first_record));

    for out_block in window_start..window_end {
        let new_vid_start = out_block * block_size;
        let new_vid_end = ((out_block + 1) * block_size).min(num_real_docs);
        let out_local = u32::try_from(out_block - window_start).map_err(|_| {
            crate::Error::Internal("BMP reorder window exceeds u32::MAX blocks".into())
        })?;
        for new_slot in 0..new_vid_end.saturating_sub(new_vid_start) {
            let global_real = permutation[new_vid_start + new_slot] as usize;
            let (source, old_vid) = resolve_real_vid(global_real, real_base, real_to_virtual);
            let source_block_size = sources[source].0.bmp_block_size as usize;
            routes.push(RecordRoute {
                source: u32::try_from(source).map_err(|_| {
                    crate::Error::Internal("BMP source index exceeds u32::MAX".into())
                })?,
                old_block: u32::try_from(old_vid / source_block_size).map_err(|_| {
                    crate::Error::Internal("BMP source block exceeds u32::MAX".into())
                })?,
                out_local,
                old_slot: (old_vid % source_block_size) as u8,
                new_slot: new_slot as u8,
            });
        }
    }
    install_on_pool(rayon_pool, || {
        routes.par_sort_unstable_by_key(|route| (route.source, route.old_block, route.old_slot));
    });

    let mut jobs = Vec::new();
    let mut start = 0usize;
    while start < routes.len() {
        let source = routes[start].source;
        let old_block = routes[start].old_block;
        let mut end = start + 1;
        while end < routes.len()
            && routes[end].source == source
            && routes[end].old_block == old_block
        {
            end += 1;
        }
        jobs.push(SourceBlockJob {
            source,
            old_block,
            route_start: u32::try_from(start)
                .map_err(|_| crate::Error::Internal("BMP route count exceeds u32::MAX".into()))?,
            route_end: u32::try_from(end)
                .map_err(|_| crate::Error::Internal("BMP route count exceeds u32::MAX".into()))?,
        });
        start = end;
    }
    Ok((routes, jobs))
}

fn source_job_route_maps(
    job: &SourceBlockJob,
    routes: &[RecordRoute],
) -> Result<([u32; 256], [u8; 256])> {
    let mut output_blocks = [u32::MAX; 256];
    let mut output_slots = [0u8; 256];
    let job_routes = routes
        .get(job.route_start as usize..job.route_end as usize)
        .ok_or_else(|| crate::Error::Internal("BMP source-job route range is invalid".into()))?;
    for route in job_routes {
        if route.source != job.source || route.old_block != job.old_block {
            return Err(crate::Error::Internal(
                "BMP source-job routes are not grouped".into(),
            ));
        }
        let slot = route.old_slot as usize;
        if output_blocks[slot] != u32::MAX {
            return Err(crate::Error::Internal(format!(
                "BMP permutation maps source block {} slot {} more than once",
                job.old_block, route.old_slot,
            )));
        }
        output_blocks[slot] = route.out_local;
        output_slots[slot] = route.new_slot;
    }
    Ok((output_blocks, output_slots))
}

fn count_source_job_postings(
    sources: &[(crate::segment::BmpIndex, u32)],
    job: &SourceBlockJob,
    routes: &[RecordRoute],
) -> Result<usize> {
    let (output_blocks, _) = source_job_route_maps(job, routes)?;
    let source = &sources[job.source as usize].0;
    let mut count = 0usize;
    for (_, _, postings) in source.iter_block_terms(job.old_block) {
        for posting in postings {
            if output_blocks[posting.local_slot as usize] != u32::MAX {
                count = count.checked_add(1).ok_or_else(|| {
                    crate::Error::Internal("BMP routed posting count overflows usize".into())
                })?;
            }
        }
    }
    Ok(count)
}

fn fill_source_job_postings(
    sources: &[(crate::segment::BmpIndex, u32)],
    job: &SourceBlockJob,
    routes: &[RecordRoute],
    output: &mut [RoutedPosting],
) -> Result<()> {
    let (output_blocks, output_slots) = source_job_route_maps(job, routes)?;
    let source = &sources[job.source as usize].0;
    let mut cursor = 0usize;
    for (dimension, _, postings) in source.iter_block_terms(job.old_block) {
        for posting in postings {
            let out_local = output_blocks[posting.local_slot as usize];
            if out_local == u32::MAX {
                continue;
            }
            let destination = output.get_mut(cursor).ok_or_else(|| {
                crate::Error::Internal("BMP routed posting count changed between passes".into())
            })?;
            *destination = RoutedPosting {
                out_local,
                dimension,
                slot: output_slots[posting.local_slot as usize],
                impact: posting.impact,
            };
            cursor += 1;
        }
    }
    if cursor != output.len() {
        return Err(crate::Error::Internal(format!(
            "BMP routed posting count changed between passes: counted {}, wrote {}",
            output.len(),
            cursor,
        )));
    }
    Ok(())
}

fn record_window_memory_bytes(
    routes: &Vec<RecordRoute>,
    jobs: &Vec<SourceBlockJob>,
    counts: &Vec<usize>,
    posting_count: usize,
) -> Option<usize> {
    let routes_bytes = routes
        .capacity()
        .checked_mul(std::mem::size_of::<RecordRoute>())?;
    let jobs_bytes = jobs
        .capacity()
        .checked_mul(std::mem::size_of::<SourceBlockJob>())?;
    let counts_bytes = counts
        .capacity()
        .checked_mul(std::mem::size_of::<usize>())?;
    let partitions_bytes = jobs
        .capacity()
        .checked_mul(std::mem::size_of::<(&SourceBlockJob, &mut [RoutedPosting])>())?;
    let routed_bytes = posting_count.checked_mul(std::mem::size_of::<RoutedPosting>())?;
    routes_bytes
        .checked_add(jobs_bytes)?
        .checked_add(counts_bytes)?
        .checked_add(partitions_bytes)?
        .checked_add(routed_bytes)
}

fn push_rewrite_bytes(buffer: &mut Vec<u8>, bytes: &[u8], writer: &mut OffsetWriter) -> Result<()> {
    if buffer.len() + bytes.len() > buffer.capacity() {
        writer.write_all(buffer).map_err(crate::Error::Io)?;
        buffer.clear();
    }
    if bytes.len() > buffer.capacity() {
        writer.write_all(bytes).map_err(crate::Error::Io)
    } else {
        buffer.extend_from_slice(bytes);
        Ok(())
    }
}

fn flush_rewrite_bytes(buffer: &mut Vec<u8>, writer: &mut OffsetWriter) -> Result<()> {
    if !buffer.is_empty() {
        writer.write_all(buffer).map_err(crate::Error::Io)?;
        buffer.clear();
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_sorted_routed_block(
    out_block: u32,
    postings: &[RoutedPosting],
    grid_entries: &mut Vec<BmpGridEntry>,
    grid_run_entries: usize,
    run_files: &mut GridScratchFiles,
    field_id: u32,
    rayon_pool: Option<&rayon::ThreadPool>,
    posting_buffer: &mut Vec<u8>,
    writer: &mut OffsetWriter,
) -> Result<(u64, usize)> {
    if postings.is_empty() {
        return Ok((0, 0));
    }
    debug_assert!(
        postings
            .iter()
            .all(|posting| posting.out_local == postings[0].out_local)
    );

    let num_terms = 1 + postings
        .windows(2)
        .filter(|pair| pair[0].dimension != pair[1].dimension)
        .count();
    let start_offset = writer.offset();
    posting_buffer.clear();
    push_rewrite_bytes(
        posting_buffer,
        &u32::try_from(num_terms)
            .map_err(|_| crate::Error::Internal("BMP block term count exceeds u32::MAX".into()))?
            .to_le_bytes(),
        writer,
    )?;

    push_rewrite_bytes(posting_buffer, &postings[0].dimension.to_le_bytes(), writer)?;
    for pair in postings.windows(2) {
        if pair[0].dimension != pair[1].dimension {
            push_rewrite_bytes(posting_buffer, &pair[1].dimension.to_le_bytes(), writer)?;
        }
    }

    push_rewrite_bytes(posting_buffer, &0u32.to_le_bytes(), writer)?;
    for (index, pair) in postings.windows(2).enumerate() {
        if pair[0].dimension != pair[1].dimension {
            push_rewrite_bytes(
                posting_buffer,
                &u32::try_from(index + 1)
                    .map_err(|_| {
                        crate::Error::Internal("BMP block posting prefix exceeds u32::MAX".into())
                    })?
                    .to_le_bytes(),
                writer,
            )?;
        }
    }
    push_rewrite_bytes(
        posting_buffer,
        &u32::try_from(postings.len())
            .map_err(|_| {
                crate::Error::Internal("BMP block posting prefix exceeds u32::MAX".into())
            })?
            .to_le_bytes(),
        writer,
    )?;

    let mut term_start = 0usize;
    while term_start < postings.len() {
        let dimension = postings[term_start].dimension;
        let mut term_end = term_start + 1;
        let mut max_impact = postings[term_start].impact;
        while term_end < postings.len() && postings[term_end].dimension == dimension {
            max_impact = max_impact.max(postings[term_end].impact);
            term_end += 1;
        }
        push_rewrite_bytes(posting_buffer, &[max_impact], writer)?;
        if grid_entries.len() == grid_run_entries {
            spill_grid_entries(grid_entries, run_files, field_id, rayon_pool)?;
        }
        grid_entries.push((dimension, out_block, max_impact));
        term_start = term_end;
    }

    for posting in postings {
        push_rewrite_bytes(posting_buffer, &[posting.slot, posting.impact], writer)?;
    }
    flush_rewrite_bytes(posting_buffer, writer)?;
    Ok((writer.offset() - start_offset, num_terms))
}

fn write_blockwise_doc_maps(
    sources: &[(crate::segment::BmpIndex, u32)],
    permuted_blocks: &[(u32, u32)],
    block_size: usize,
    writer: &mut OffsetWriter,
    rayon_pool: Option<&rayon::ThreadPool>,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> Result<()> {
    use rayon::prelude::*;

    let id_block_bytes = block_size * 4;
    let blocks_per_chunk = (REWRITE_IO_BUFFER_BYTES / id_block_bytes).max(1);
    let mut buffer = Vec::with_capacity(blocks_per_chunk * id_block_bytes);
    for blocks in permuted_blocks.chunks(blocks_per_chunk) {
        if cancellation_requested(cancellation) {
            return Err(crate::Error::IndexClosed);
        }
        buffer.resize(blocks.len() * id_block_bytes, 0);
        install_on_pool(rayon_pool, || {
            buffer
                .par_chunks_exact_mut(id_block_bytes)
                .zip(blocks.par_iter())
                .try_for_each(|(output, &(source, local_block))| -> Result<()> {
                    let (bmp, doc_offset) = &sources[source as usize];
                    let start = local_block as usize * id_block_bytes;
                    output.copy_from_slice(&bmp.doc_map_ids_slice()[start..start + id_block_bytes]);
                    if *doc_offset != 0 {
                        let (ids, _) = output.as_chunks_mut::<4>();
                        for id in ids {
                            let source_doc_id = u32::from_le_bytes(*id);
                            if source_doc_id != u32::MAX {
                                *id = source_doc_id
                                    .checked_add(*doc_offset)
                                    .ok_or_else(|| {
                                        crate::Error::Corruption(format!(
                                            "BMP doc-id offset overflow: {} + {}",
                                            source_doc_id, doc_offset,
                                        ))
                                    })?
                                    .to_le_bytes();
                            }
                        }
                    }
                    Ok(())
                })
        })?;
        writer.write_all(&buffer).map_err(crate::Error::Io)?;
    }

    let ordinal_block_bytes = block_size * 2;
    let blocks_per_chunk = (REWRITE_IO_BUFFER_BYTES / ordinal_block_bytes).max(1);
    buffer.clear();
    buffer.reserve(
        blocks_per_chunk
            .saturating_mul(ordinal_block_bytes)
            .saturating_sub(buffer.capacity()),
    );
    for blocks in permuted_blocks.chunks(blocks_per_chunk) {
        if cancellation_requested(cancellation) {
            return Err(crate::Error::IndexClosed);
        }
        buffer.resize(blocks.len() * ordinal_block_bytes, 0);
        install_on_pool(rayon_pool, || {
            buffer
                .par_chunks_exact_mut(ordinal_block_bytes)
                .zip(blocks.par_iter())
                .for_each(|(output, &(source, local_block))| {
                    let bmp = &sources[source as usize].0;
                    let start = local_block as usize * ordinal_block_bytes;
                    output.copy_from_slice(
                        &bmp.doc_map_ordinals_slice()[start..start + ordinal_block_bytes],
                    );
                });
        });
        writer.write_all(&buffer).map_err(crate::Error::Io)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_record_doc_maps(
    sources: &[(crate::segment::BmpIndex, u32)],
    permutation: &[u32],
    real_base: &[usize],
    real_to_virtual: &[Vec<u32>],
    num_virtual_docs: usize,
    writer: &mut OffsetWriter,
    rayon_pool: Option<&rayon::ThreadPool>,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> Result<()> {
    use rayon::prelude::*;

    let id_records_per_chunk = (REWRITE_IO_BUFFER_BYTES / 4).max(1);
    let mut buffer = Vec::with_capacity(
        permutation
            .len()
            .min(id_records_per_chunk)
            .saturating_mul(4),
    );
    for records in permutation.chunks(id_records_per_chunk) {
        if cancellation_requested(cancellation) {
            return Err(crate::Error::IndexClosed);
        }
        buffer.resize(records.len() * 4, 0);
        install_on_pool(rayon_pool, || {
            buffer
                .par_chunks_exact_mut(4)
                .zip(records.par_iter())
                .try_for_each(|(output, &global_real)| -> Result<()> {
                    let (source, virtual_id) =
                        resolve_real_vid(global_real as usize, real_base, real_to_virtual);
                    let ids = sources[source].0.doc_map_ids_slice();
                    let offset = virtual_id * 4;
                    let source_doc_id =
                        u32::from_le_bytes(ids[offset..offset + 4].try_into().unwrap());
                    let output_doc_id =
                        source_doc_id
                            .checked_add(sources[source].1)
                            .ok_or_else(|| {
                                crate::Error::Corruption(format!(
                                    "BMP doc-id offset overflow: {} + {}",
                                    source_doc_id, sources[source].1,
                                ))
                            })?;
                    output.copy_from_slice(&output_doc_id.to_le_bytes());
                    Ok(())
                })
        })?;
        writer.write_all(&buffer).map_err(crate::Error::Io)?;
    }
    let padding = num_virtual_docs
        .checked_sub(permutation.len())
        .ok_or_else(|| {
            crate::Error::Internal(format!(
                "BMP document-map permutation has {} records for {} virtual slots",
                permutation.len(),
                num_virtual_docs,
            ))
        })?;
    if padding > 0 {
        buffer.clear();
        buffer.resize(padding * 4, u8::MAX);
        writer.write_all(&buffer).map_err(crate::Error::Io)?;
    }

    let ordinal_records_per_chunk = (REWRITE_IO_BUFFER_BYTES / 2).max(1);
    buffer.clear();
    buffer.reserve(
        permutation
            .len()
            .min(ordinal_records_per_chunk)
            .saturating_mul(2)
            .saturating_sub(buffer.capacity()),
    );
    for records in permutation.chunks(ordinal_records_per_chunk) {
        if cancellation_requested(cancellation) {
            return Err(crate::Error::IndexClosed);
        }
        buffer.resize(records.len() * 2, 0);
        install_on_pool(rayon_pool, || {
            buffer
                .par_chunks_exact_mut(2)
                .zip(records.par_iter())
                .for_each(|(output, &global_real)| {
                    let (source, virtual_id) =
                        resolve_real_vid(global_real as usize, real_base, real_to_virtual);
                    let ordinals = sources[source].0.doc_map_ordinals_slice();
                    let offset = virtual_id * 2;
                    output.copy_from_slice(&ordinals[offset..offset + 2]);
                });
        });
        writer.write_all(&buffer).map_err(crate::Error::Io)?;
    }
    if padding > 0 {
        buffer.clear();
        buffer.resize(padding * 2, 0);
        writer.write_all(&buffer).map_err(crate::Error::Io)?;
    }
    Ok(())
}

/// Reorder granularity (see docs/block-level-reorder.md).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BpGranularity {
    /// Decide per field from footer stats: coherent blocks → `Blocks`,
    /// scrambled blocks → `Records`. The decision and its inputs are logged.
    #[default]
    Auto,
    /// Record-level BP: full quality (tight block UBs + superblock
    /// locality), scatter rewrite. Compacts interior padding.
    Records,
    /// Block-level BP: permute whole blocks to recover superblock locality
    /// at ~1/block_size of the BP cost and a memcpy-class rewrite. Block
    /// composition (and therefore block UBs) unchanged.
    Blocks,
}

/// `Auto` picks block-level reorder when normalized coherence — how far the
/// measured d sits between its random-order baseline and its perfect-packing
/// bound, 0..=1 — is at least this. Value-independent: unlike an absolute d
/// cutoff, it is not skewed by the corpus's dim-frequency distribution
/// (rare-dim corpora cap d low even when perfectly clustered; ubiquitous
/// dims inflate d even when scrambled). Measured values are logged per pass
/// for tuning.
pub const BLOCKWISE_NORM_COHERENCE_THRESHOLD: f32 = 0.5;

/// Scan cap for the coherence decision: above this many blocks the scan
/// samples every k-th block instead of all of them. 8192 blocks (~262k
/// vectors at the production block size 32) is statistically ample for a
/// 0.5 threshold and bounds the decision cost on multi-million-vector
/// segments.
const MAX_COHERENCE_SCAN_BLOCKS: usize = 8192;

/// Coherence statistics driving the `Auto` granularity decision.
#[derive(Clone, Copy, Debug)]
struct CoherenceStats {
    /// Raw average records per (block, dim) pair: ≈1 when blocks are
    /// scrambled, grows toward block_size as blocks become coherent.
    d: f32,
    /// Expected `d` if the same records were assigned to blocks uniformly at
    /// random: `P / Σ_t B·(1 − (1 − 1/B)^df_t)`.
    d_rand: f32,
    /// Upper bound on achievable `d`: every dim packed into its minimal
    /// `ceil(df_t / block_size)` blocks. Ignores joint constraints across
    /// dims, so it overestimates — which biases `norm` down, toward the
    /// higher-quality record-level path.
    d_max: f32,
    /// `(d − d_rand) / (d_max − d_rand)`, clamped to 0..=1. When the data
    /// offers no clustering headroom (`d_max ≈ d_rand`, e.g. only ubiquitous
    /// or singleton dims), reordering records cannot help, so this is 1.0.
    norm: f32,
    /// Blocks actually scanned vs total — differ when the sampling cap hit.
    scanned_blocks: usize,
    total_blocks: usize,
}

/// Compute [`CoherenceStats`] from a streaming pass over block headers:
/// per-dim record frequencies come from posting-slice lengths, no weight
/// decode. Rewrite validation has already proved every dim is below the
/// configured vocabulary size, so a compact direct-addressed table avoids
/// hashing up to a million sampled headers. Segments above
/// [`MAX_COHERENCE_SCAN_BLOCKS`] are stride-sampled and all aggregates
/// come from the sampled sub-population, which keeps the estimator
/// consistent at both the clustered and scrambled extremes.
///
/// Singleton dims (df=1) are excluded from every aggregate: they contribute
/// identically to the actual, random, and packed pair counts (one pair
/// each), so they carry zero ordering signal and would only compress the
/// three bounds together — on id-heavy corpora (every record has a unique
/// dim) enough to trip the no-headroom epsilon and mask real headroom in
/// the informative dims.
fn block_coherence(
    sources: &[(crate::segment::BmpIndex, u32)],
    block_size: usize,
    dims: usize,
) -> CoherenceStats {
    let total_blocks: usize = sources.iter().map(|(b, _)| b.num_blocks as usize).sum();
    if total_blocks == 0 {
        return CoherenceStats {
            d: 0.0,
            d_rand: 0.0,
            d_max: 0.0,
            norm: 1.0,
            scanned_blocks: 0,
            total_blocks,
        };
    }

    let stride = total_blocks.div_ceil(MAX_COHERENCE_SCAN_BLOCKS).max(1);
    // dim → (record count, block count) within the sampled blocks
    let mut df = vec![(0u64, 0u64); dims];
    let mut scanned_blocks = 0usize;
    let mut global_block = 0usize;
    for (bmp, _) in sources {
        for block_id in 0..bmp.num_blocks {
            if global_block.is_multiple_of(stride) {
                scanned_blocks += 1;
                for (dim_id, _, posts) in bmp.iter_block_terms(block_id) {
                    let e = &mut df[dim_id as usize];
                    e.0 += posts.len() as u64;
                    e.1 += 1;
                }
            }
            global_block += 1;
        }
    }

    let b = scanned_blocks as f64;
    let keep = 1.0 - 1.0 / b;
    let mut postings: u64 = 0;
    let mut terms: u64 = 0;
    let mut expected_rand_pairs = 0.0f64;
    let mut min_pairs = 0u64;
    for &(records, blocks) in &df {
        if records < 2 {
            continue; // singleton: inert for clustering, see above
        }
        postings += records;
        terms += blocks;
        expected_rand_pairs += b * (1.0 - keep.powf(records as f64));
        min_pairs += records.div_ceil(block_size as u64);
    }
    if terms == 0 || postings == 0 {
        // No dim shared by ≥2 records: BP has no signal at any granularity.
        return CoherenceStats {
            d: 0.0,
            d_rand: 0.0,
            d_max: 0.0,
            norm: 1.0,
            scanned_blocks,
            total_blocks,
        };
    }

    let d = postings as f32 / terms as f32;
    let d_rand = if expected_rand_pairs > 0.0 {
        (postings as f64 / expected_rand_pairs) as f32
    } else {
        d
    };
    let d_max = postings as f32 / min_pairs.max(1) as f32;
    let headroom = d_max - d_rand;
    // Relative epsilon: below ~5% headroom the bounds are within estimation
    // noise of each other and no ordering can move d meaningfully.
    let norm = if headroom <= 0.05 * d_rand {
        1.0
    } else {
        ((d - d_rand) / headroom).clamp(0.0, 1.0)
    };
    CoherenceStats {
        d,
        d_rand,
        d_max,
        norm,
        scanned_blocks,
        total_blocks,
    }
}

/// Reorder a single segment's BMP data via Recursive Graph Bisection (BP).
///
/// Creates a new segment with reordered BMP blocks for better pruning.
/// Non-BMP fields are copied unchanged via streaming file copy.
///
/// `rayon_pool`: optional bounded thread pool for BP computation. When `Some`,
/// all rayon parallel work (gain computation, recursive bisection) runs on this
/// pool instead of the global rayon pool. This prevents the optimizer from
/// saturating all CPU cores.
///
/// Returns `(new_segment_hex_id, num_docs, bp_converged)`. `bp_converged` is
/// false iff the BP wall-clock budget ended a field's pass early (the output
/// is still valid and better-ordered than the input; a later pass warm-starts
/// from it and deepens).
///
/// `granularity`: callers deepening an unconverged segment must pass
/// `BpGranularity::Records` — `Auto` would measure the partial pass's
/// residual coherence, potentially take the blockwise path, report converged,
/// and end the deepening cascade at partial quality.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn reorder_segment<D: Directory + DirectoryWriter>(
    dir: &D,
    schema: &Arc<Schema>,
    source_id: SegmentId,
    output_id: SegmentId,
    term_cache_blocks: usize,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    granularity: BpGranularity,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
    cancellation: Option<Arc<std::sync::atomic::AtomicBool>>,
) -> Result<(String, u32, bool)> {
    let reader = SegmentReader::open(dir, source_id, Arc::clone(schema), term_cache_blocks).await?;
    let num_docs = reader.num_docs();

    let src_files = SegmentFiles::new(source_id.0);
    let dst_files = SegmentFiles::new(output_id.0);

    log::info!(
        "[reorder] segment {} → {} ({} docs)",
        source_id.to_hex(),
        output_id.to_hex(),
        num_docs,
    );

    // Copy unchanged segment files
    let copy_start = std::time::Instant::now();
    for (src, dst, required) in [
        (&src_files.term_dict, &dst_files.term_dict, true),
        (&src_files.postings, &dst_files.postings, true),
        (
            &src_files.positions,
            &dst_files.positions,
            reader.has_positions_file(),
        ),
        (&src_files.store, &dst_files.store, true),
        (
            &src_files.fast,
            &dst_files.fast,
            !reader.fast_fields().is_empty(),
        ),
        (
            &src_files.vectors,
            &dst_files.vectors,
            !reader.vector_indexes().is_empty() || !reader.flat_vectors().is_empty(),
        ),
    ] {
        if cancellation_requested(cancellation.as_deref()) {
            return Err(crate::Error::IndexClosed);
        }
        clone_segment_file(dir, src, dst, required, cancellation.as_deref()).await?;
    }
    log::info!(
        "[reorder] copied files in {:.1}s",
        copy_start.elapsed().as_secs_f64(),
    );

    // Rebuild sparse file with reordered BMP data
    let bp_converged = reorder_sparse_file(
        dir,
        &reader,
        &dst_files,
        schema,
        memory_budget,
        bp_budget,
        cancellation,
        granularity,
        rayon_pool,
    )
    .await?;

    // Write new meta with output segment ID
    let src_meta = reader.meta();
    let meta = SegmentMeta {
        id: output_id.0,
        num_docs: src_meta.num_docs,
        field_stats: src_meta.field_stats.clone(),
    };
    // Durable: the reordered segment replaces its fsynced source, so a
    // non-durable .meta could be the only copy across a power failure.
    dir.write_durable(&dst_files.meta, &meta.serialize()?)
        .await?;

    Ok((output_id.to_hex(), num_docs, bp_converged))
}

/// Rewrite only a segment's dense-vector file while retaining every immutable
/// non-vector file. Filesystem backends hard-link unchanged data, so upgrading
/// a max-sized segment does not duplicate its postings, store, sparse index, or
/// fast fields. Backends without links use the bounded streaming fallback.
pub async fn rewrite_vector_segment<D: Directory + DirectoryWriter>(
    dir: &D,
    schema: &Arc<Schema>,
    source_id: SegmentId,
    output_id: SegmentId,
    term_cache_blocks: usize,
    trained: &TrainedVectorStructures,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<(String, u32)> {
    let reader = SegmentReader::open(dir, source_id, Arc::clone(schema), term_cache_blocks).await?;
    let num_docs = reader.num_docs();
    let src_files = SegmentFiles::new(source_id.0);
    let dst_files = SegmentFiles::new(output_id.0);

    log::info!(
        "[dense_vector_rewrite] finalizing segment {} → {} ({} docs)",
        source_id.to_hex(),
        output_id.to_hex(),
        num_docs,
    );

    for (src, dst, required) in [
        (&src_files.term_dict, &dst_files.term_dict, true),
        (&src_files.postings, &dst_files.postings, true),
        (
            &src_files.positions,
            &dst_files.positions,
            reader.has_positions_file(),
        ),
        (&src_files.store, &dst_files.store, true),
        (
            &src_files.fast,
            &dst_files.fast,
            !reader.fast_fields().is_empty(),
        ),
        (
            &src_files.sparse,
            &dst_files.sparse,
            !reader.sparse_indexes().is_empty() || !reader.bmp_indexes().is_empty(),
        ),
    ] {
        clone_segment_file(dir, src, dst, required, None).await?;
    }

    let merger = SegmentMerger::new(Arc::clone(schema)).with_background_pool(rayon_pool);
    let vector_bytes = merger
        .merge_dense_vectors(
            dir,
            std::slice::from_ref(&reader),
            &dst_files,
            Some(trained),
            super::merger::AnnWriteMode::Rebuild,
        )
        .await?;
    if vector_bytes == 0 {
        return Err(crate::Error::Corruption(format!(
            "vector rewrite source {} has no flat vector payload",
            source_id.to_hex(),
        )));
    }

    let src_meta = reader.meta();
    let meta = SegmentMeta {
        id: output_id.0,
        num_docs: src_meta.num_docs,
        field_stats: src_meta.field_stats.clone(),
    };
    dir.write_durable(&dst_files.meta, &meta.serialize()?)
        .await?;

    Ok((output_id.to_hex(), num_docs))
}

/// Retain an immutable segment file with a hard link when possible, falling
/// back to streaming I/O in 4 MB chunks.
///
/// No-op if an optional source file does not exist. A file observed by the
/// source reader is required: losing it during the copy is corruption, not an
/// empty optional field.
/// Empty files are still created (SegmentReader requires their existence).
async fn clone_segment_file<D: Directory + DirectoryWriter>(
    dir: &D,
    src: &Path,
    dst: &Path,
    required: bool,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> Result<()> {
    if cancellation_requested(cancellation) {
        return Err(crate::Error::IndexClosed);
    }
    match dir.link(src, dst).await {
        Ok(()) => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !required => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(crate::Error::Corruption(format!(
                "required segment source file {src:?} disappeared before link"
            )));
        }
        Err(error) => {
            log::debug!(
                "[segment_clone] link {:?} → {:?} unavailable ({}), streaming instead",
                src,
                dst,
                error,
            );
        }
    }

    let handle = match dir.open_read(src).await {
        Ok(h) => h,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !required => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(crate::Error::Corruption(format!(
                "required segment source file {src:?} disappeared before copy"
            )));
        }
        Err(error) => return Err(crate::Error::Io(error)),
    };
    let mut writer = dir
        .streaming_writer_cold(dst)
        .await
        .map_err(crate::Error::Io)?;
    const CHUNK: usize = 4 * 1024 * 1024;
    let source_len = handle.len();
    let mut offset = 0u64;
    while offset < source_len {
        if cancellation_requested(cancellation) {
            return Err(crate::Error::IndexClosed);
        }
        let end = (offset + CHUNK as u64).min(source_len);
        let data = handle
            .read_bytes_range(offset..end)
            .await
            .map_err(crate::Error::Io)?;
        writer = super::merger::block_in_place_if_multithread(move || {
            writer.write_all(data.as_slice())?;
            // Drop source pages behind the copy cursor — a whole-file copy
            // must not leave the complete source resident in page cache.
            #[cfg(feature = "native")]
            data.madvise_range(0..data.len(), libc::MADV_DONTNEED);
            Ok::<_, std::io::Error>(writer)
        })
        .map_err(crate::Error::Io)?;
        offset = end;
    }
    if writer.bytes_written() != source_len {
        return Err(crate::Error::Corruption(format!(
            "short segment clone from {:?} to {:?}: wrote {} of {} bytes",
            src,
            dst,
            writer.bytes_written(),
            source_len,
        )));
    }
    super::merger::block_in_place_if_multithread(move || writer.finish())
        .map_err(crate::Error::Io)?;

    Ok(())
}

/// Rebuild the sparse file with reordered BMP data.
///
/// - BMP fields: run BP reorder and write new blob
/// - MaxScore fields: identity-copy raw data from source
/// - No sparse fields: no-op
#[allow(clippy::too_many_arguments)]
async fn reorder_sparse_file<D: Directory + DirectoryWriter>(
    dir: &D,
    reader: &SegmentReader,
    dst_files: &SegmentFiles,
    schema: &Schema,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    cancellation: Option<Arc<std::sync::atomic::AtomicBool>>,
    granularity: BpGranularity,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<bool> {
    let sparse_fields: Vec<_> = schema
        .fields()
        .filter(|(_, entry)| matches!(entry.field_type, FieldType::SparseVector))
        .map(|(field, entry)| {
            (
                field,
                entry.sparse_vector_config.clone(),
                entry.reorder,
                entry.name.clone(),
            )
        })
        .collect();

    if sparse_fields.is_empty() {
        return Ok(true);
    }

    // Check if there's any BMP data to reorder. Per-field gate: only fields
    // with the `reorder` schema attribute get BP; others are copied unchanged.
    let has_bmp_data = sparse_fields.iter().any(|(field, config, reorder, _)| {
        *reorder
            && config.as_ref().map(|c| c.format) == Some(SparseFormat::Bmp)
            && reader.bmp_indexes().get(&field.0).is_some()
    });

    // Sparse files are legitimately absent when the schema has sparse fields
    // but this segment contains no sparse values. Do not turn that valid
    // optional-file case into a required-copy corruption error.
    if reader.sparse_indexes().is_empty() && reader.bmp_indexes().is_empty() {
        return Ok(true);
    }

    if !has_bmp_data {
        // No BMP field wants reordering — just copy the sparse file as-is
        log::info!(
            "[reorder] segment {:x}: no BMP field has the `reorder` schema attribute — sparse file copied unchanged",
            reader.meta().id,
        );
        let src_files = SegmentFiles::new(reader.meta().id);
        clone_segment_file(
            dir,
            &src_files.sparse,
            &dst_files.sparse,
            true,
            cancellation.as_deref(),
        )
        .await?;
        return Ok(true);
    }

    let mut all_converged = true;
    let scratch_path = dir.local_path(&dst_files.sparse).unwrap_or_else(|| {
        std::env::temp_dir().join(
            dst_files
                .sparse
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("hermes.sparse")),
        )
    });
    let mut writer = OffsetWriter::new(
        dir.streaming_writer_cold(&dst_files.sparse)
            .await
            .map_err(crate::Error::Io)?,
    );
    let skip_tmp = dst_files.sparse_skip_temp();
    let mut skip_writer = dir
        .streaming_writer_cold(&skip_tmp)
        .await
        .map_err(crate::Error::Io)?;
    let mut field_tocs: Vec<SparseFieldToc> = Vec::new();
    let mut skip_entry_buffer = Vec::with_capacity(crate::structures::SparseSkipEntry::SIZE);
    let mut skip_count: u32 = 0;

    for (field, sparse_config, reorder, field_name) in &sparse_fields {
        if cancellation_requested(cancellation.as_deref()) {
            return Err(crate::Error::IndexClosed);
        }
        let format = sparse_config.as_ref().map(|c| c.format).unwrap_or_default();
        let quantization = sparse_config
            .as_ref()
            .map(|c| c.weight_quantization)
            .unwrap_or(crate::structures::WeightQuantization::Float32);

        if format == SparseFormat::Bmp {
            if let Some(bmp_idx) = reader.bmp_indexes().get(&field.0) {
                if !*reorder {
                    // Field opted out of BP: copy its blob byte-identically.
                    log::info!(
                        "[reorder] field {}: `reorder` attribute not set — blob copied unchanged",
                        field.0,
                    );
                    copy_bmp_blob(
                        bmp_idx,
                        field.0,
                        bmp_idx.total_vectors,
                        &mut writer,
                        &mut field_tocs,
                    )?;
                    continue;
                }
                let effective_block_size = sparse_config
                    .as_ref()
                    .map(|c| c.bmp_block_size)
                    .unwrap_or(crate::structures::SparseVectorConfig::DEFAULT_BMP_BLOCK_SIZE)
                    .clamp(1, 256) as usize;
                let dims = sparse_config
                    .as_ref()
                    .and_then(|c| c.dims)
                    .unwrap_or_else(|| bmp_idx.dims());
                let grid_bits = sparse_config
                    .as_ref()
                    .map(|c| c.bmp_grid_bits)
                    .unwrap_or(crate::structures::SparseVectorConfig::DEFAULT_BMP_GRID_BITS);
                let max_weight_scale = sparse_config
                    .as_ref()
                    .and_then(|c| c.max_weight)
                    .unwrap_or(bmp_idx.max_weight_scale);
                let total_vectors = bmp_idx.total_vectors;

                // One field owns the output writer and BP working set at a
                // time. The field itself still uses the bounded Rayon pool.
                let bmp_sources = vec![(bmp_idx.clone(), 0u32)];
                let fid = field.0;
                let fname = field_name.clone();
                let ilabel = schema.index_label().to_owned();
                let pool = rayon_pool.clone();
                let scratch_path = scratch_path.clone();
                let field_bp_budget = bp_budget;
                let field_cancellation = cancellation.clone();
                let (w, ft, converged) = super::merger::block_in_place_if_multithread(move || {
                    reorder_bmp_field(
                        &bmp_sources,
                        fid,
                        &ilabel,
                        &fname,
                        dims,
                        effective_block_size,
                        grid_bits,
                        max_weight_scale,
                        total_vectors,
                        memory_budget,
                        field_bp_budget,
                        field_cancellation,
                        granularity,
                        scratch_path,
                        writer,
                        field_tocs,
                        pool,
                    )
                })?;
                writer = w;
                field_tocs = ft;
                all_converged &= converged;
            }
        } else {
            // MaxScore format: identity-copy raw data from source
            if let Some(sparse_idx) = reader.sparse_indexes().get(&field.0) {
                copy_maxscore_field(
                    sparse_idx,
                    field.0,
                    quantization,
                    &mut writer,
                    &mut field_tocs,
                    skip_writer.as_mut(),
                    &mut skip_entry_buffer,
                    &mut skip_count,
                )
                .await?;
            }
        }
    }
    skip_writer.finish().map_err(crate::Error::Io)?;

    if field_tocs.is_empty() {
        drop(writer);
        let _ = dir.delete(&dst_files.sparse).await;
        let _ = dir.delete(&skip_tmp).await;
        return Ok(all_converged);
    }

    // Write skip section (MaxScore fields only)
    let skip_offset = writer.offset();
    let skip_bytes = u64::from(skip_count)
        .checked_mul(crate::structures::SparseSkipEntry::SIZE as u64)
        .ok_or_else(|| crate::Error::Internal("sparse skip section exceeds u64::MAX".into()))?;
    super::merger::append_and_delete_temp(dir, &skip_tmp, skip_bytes, &mut writer).await?;

    // Write TOC + footer
    let toc_offset = writer.offset();
    write_sparse_toc_and_footer(&mut writer, skip_offset, toc_offset, &field_tocs)
        .map_err(crate::Error::Io)?;

    writer.finish().map_err(crate::Error::Io)?;

    let total_dims: usize = field_tocs.iter().map(|f| f.dims.len()).sum();
    log::info!(
        "[reorder] sparse file written: {} fields, {} dims, {} skip entries (bp_converged={})",
        field_tocs.len(),
        total_dims,
        skip_count,
        all_converged,
    );

    Ok(all_converged)
}

/// Block-level reorder: permute whole blocks so similar blocks share
/// superblocks — a permuted block copy, no record unpacking. See
/// docs/block-level-reorder.md. Interior padding is preserved (blocks are
/// copied verbatim); block UBs are unchanged by construction.
#[allow(clippy::too_many_arguments)]
fn reorder_bmp_field_blockwise(
    sources: &[(crate::segment::BmpIndex, u32)],
    field_id: u32,
    index_label: &str,
    field_name: &str,
    dims: u32,
    effective_block_size: usize,
    grid_bits: u8,
    max_weight_scale: f32,
    total_vectors: u32,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
    scratch_path: PathBuf,
    source_pages: &crate::segment::reader::bmp::BmpScanPageGuard<'_>,
    mut writer: OffsetWriter,
    mut field_tocs: Vec<SparseFieldToc>,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<(OffsetWriter, Vec<SparseFieldToc>, bool)> {
    use crate::segment::builder::bmp::write_bmp_footer;
    use crate::segment::builder::graph_bisection::{
        BpProgressLabel, build_forward_index_from_blocks, graph_bisection_with_progress,
    };
    use crate::segment::reader::bmp::BMP_SUPERBLOCK_SIZE;

    let bmp_refs: Vec<&crate::segment::BmpIndex> = sources.iter().map(|(b, _)| b).collect();
    if cancellation_requested(cancellation) {
        return Err(crate::Error::IndexClosed);
    }
    {
        use rayon::prelude::*;
        install_on_pool(rayon_pool.as_deref(), || {
            bmp_refs
                .par_iter()
                .try_for_each(|bmp| bmp.visit_real_slots_for_rewrite(|_| {}))
        })?;
    }
    let num_blocks_total = bmp_refs
        .iter()
        .try_fold(0usize, |total, bmp| {
            total.checked_add(bmp.num_blocks as usize)
        })
        .ok_or_else(|| {
            crate::Error::Internal("reordered BMP block count overflows usize".into())
        })?;
    if num_blocks_total == 0 {
        return Ok((writer, field_tocs, true));
    }
    if num_blocks_total > u32::MAX as usize {
        return Err(crate::Error::Internal(
            "reordered BMP block count exceeds the V18 u32 format limit".into(),
        ));
    }
    let num_virtual_docs = num_blocks_total
        .checked_mul(effective_block_size)
        .filter(|&count| count <= u32::MAX as usize)
        .ok_or_else(|| {
            crate::Error::Internal(
                "reordered BMP virtual-document count exceeds the V18 u32 format limit".into(),
            )
        })?;

    // Global block idx → source: prefix sums.
    let block_base: Vec<usize> = std::iter::once(0)
        .chain(bmp_refs.iter().scan(0usize, |acc, b| {
            *acc += b.num_blocks as usize;
            Some(*acc)
        }))
        .collect();
    let resolve = |global: usize| -> (usize, usize) {
        let src = block_base.partition_point(|&b| b <= global) - 1;
        (src, global - block_base[src])
    };

    // ── BP over blocks, down to superblock granularity ──────────────────
    let bp_start = std::time::Instant::now();
    let sb = BMP_SUPERBLOCK_SIZE as usize;
    // The budget's depth cap is in DOCS but BP entities here are BLOCKS —
    // convert units, or the optimizer's partial cap (e.g. 4096 docs) would
    // be read as 4096 blocks and silently stop BP above superblock depth.
    let block_min_partition = bp_budget
        .min_partition_docs
        .map(|docs| (docs / effective_block_size).max(1));
    // Forward-index build is parallel too — keep it on the bounded pool.
    let run_bp = || {
        if bp_budget.time_budget.is_some_and(|budget| budget.is_zero()) {
            return ((0..num_blocks_total as u32).collect(), false);
        }
        let fwd = build_forward_index_from_blocks(&bmp_refs, memory_budget);
        if fwd.num_terms > 0 && num_blocks_total > sb {
            // The user-visible time budget covers the forward-index build as
            // well as graph refinement. Previously a multi-minute build ran
            // before a fresh full budget started, defeating the lifecycle cap.
            let block_budget = crate::segment::BpBudget {
                min_partition_docs: block_min_partition,
                time_budget: bp_budget
                    .time_budget
                    .map(|budget| budget.saturating_sub(bp_start.elapsed())),
            };
            graph_bisection_with_progress(
                &fwd,
                sb,
                20,
                block_budget,
                cancellation,
                BpProgressLabel {
                    index: index_label,
                    field: field_name,
                    entity_kind: "blocks",
                },
            )
        } else {
            (
                (0..num_blocks_total as u32).collect(),
                num_blocks_total <= sb || !fwd.budget_limited(),
            )
        }
    };
    let (perm, converged) = if let Some(ref pool) = rayon_pool {
        pool.install(run_bp)
    } else {
        run_bp()
    };
    if cancellation_requested(cancellation) {
        return Err(crate::Error::IndexClosed);
    }
    // Resolving a global block through source prefix sums is cheap once, but
    // catastrophic inside the `dims × blocks` grid rewrite below (billions
    // of calls on production-sized fields). Cache the compact source/local
    // pair once per output block; this is 8 bytes/block.
    let permuted_blocks: Vec<(u32, u32)> = perm
        .iter()
        .map(|&old_global| {
            let (source, local_block) = resolve(old_global as usize);
            (source as u32, local_block as u32)
        })
        .collect();
    drop(perm);
    log::info!(
        "[reorder_bmp] field {}: blockwise BP over {} blocks in {:.1}ms (converged={})",
        field_id,
        num_blocks_total,
        bp_start.elapsed().as_secs_f64() * 1000.0,
        converged,
    );
    source_pages.switch_to_random();

    // ── Write blob: permuted block copy ─────────────────────────────────
    let rewrite_start = std::time::Instant::now();
    let blob_start = writer.offset();

    // Section B: block data verbatim, in permuted order
    let mut block_data_starts: Vec<u64> = Vec::with_capacity(num_blocks_total + 1);
    let mut cumulative: u64 = 0;
    let mut total_terms: u64 = 0;
    let mut total_postings: u64 = 0;
    for b in &bmp_refs {
        total_terms = total_terms.saturating_add(b.total_terms());
        total_postings = total_postings.saturating_add(b.total_postings());
    }
    let persistent_bytes = permuted_blocks
        .capacity()
        .saturating_mul(std::mem::size_of::<(u32, u32)>())
        .saturating_add(
            block_data_starts
                .capacity()
                .saturating_mul(std::mem::size_of::<u64>()),
        );
    let grid_budget = memory_budget
        .saturating_sub(persistent_bytes)
        .saturating_div(2)
        .clamp(1024 * 1024, 512 * 1024 * 1024);
    let grid_run_entries =
        (grid_budget / std::mem::size_of::<BmpGridEntry>()).clamp(1, MAX_GRID_RUN_ENTRIES);
    let mut grid_entries = Vec::with_capacity(grid_run_entries);
    let mut run_files = GridScratchFiles::new(field_id, scratch_path);

    for (new_block, &(src, lb)) in permuted_blocks.iter().enumerate() {
        if new_block.is_multiple_of(1024) && cancellation_requested(cancellation) {
            return Err(crate::Error::IndexClosed);
        }
        let bmp = bmp_refs[src as usize];
        let start = bmp.block_data_start(lb) as usize;
        let end = if lb + 1 < bmp.num_blocks {
            bmp.block_data_start(lb + 1) as usize
        } else {
            bmp.block_data_sentinel() as usize
        };
        block_data_starts.push(cumulative);
        let bytes = &bmp.block_data_slice()[start..end];
        writer.write_all(bytes).map_err(crate::Error::Io)?;
        cumulative += bytes.len() as u64;
        for (dimension, max_impact, _) in bmp.iter_block_terms(lb) {
            if grid_entries.len() == grid_run_entries {
                spill_grid_entries(
                    &mut grid_entries,
                    &mut run_files,
                    field_id,
                    rayon_pool.as_deref(),
                )?;
            }
            grid_entries.push((dimension, new_block as u32, max_impact));
        }
    }
    block_data_starts.push(cumulative);

    // Padding + Section A: block_data_starts
    let block_data_len = writer.offset() - blob_start;
    let padding = (8 - (block_data_len % 8) as usize) % 8;
    if padding > 0 {
        writer
            .write_all(&[0u8; 8][..padding])
            .map_err(crate::Error::Io)?;
    }
    crate::segment::builder::bmp::write_u64_slice_le(&mut writer, &block_data_starts)
        .map_err(crate::Error::Io)?;
    drop(block_data_starts);
    let block_data_elapsed = rewrite_start.elapsed();

    // Sections D + E + H: exact maxima from the copied blocks, compressed in one
    // bounded external-sort pass. Block payloads above remained byte-identical.
    let grids_start = std::time::Instant::now();
    let grid_offset = writer.offset() - blob_start;
    let (block_grid_bytes, superblock_grid_bytes, _) = write_reorder_grids(
        grid_entries,
        &mut run_files,
        field_id,
        dims as usize,
        num_blocks_total,
        grid_bits,
        rayon_pool.as_deref(),
        &mut writer,
        cancellation,
    )?;
    let sb_grid_offset = grid_offset + block_grid_bytes;
    let coarse_grid_offset = sb_grid_offset + superblock_grid_bytes;
    // Release potentially multi-GB sorted runs and hierarchy spools before
    // the doc-map phase starts consuming more output space.
    drop(run_files);
    let grids_elapsed = grids_start.elapsed();

    // Sections F + G: doc maps copied per block chunk, ids offset-patched
    let doc_maps_start = std::time::Instant::now();
    let doc_map_offset = writer.offset() - blob_start;
    let bs = effective_block_size;
    let mut num_real_docs: u32 = 0;
    for b in &bmp_refs {
        num_real_docs = num_real_docs
            .checked_add(b.num_real_docs())
            .ok_or_else(|| {
                crate::Error::Internal(
                    "reordered BMP real-document count exceeds the V18 u32 format limit".into(),
                )
            })?;
    }
    write_blockwise_doc_maps(
        sources,
        &permuted_blocks,
        bs,
        &mut writer,
        rayon_pool.as_deref(),
        cancellation,
    )?;
    let doc_maps_elapsed = doc_maps_start.elapsed();

    // Footer
    write_bmp_footer(
        &mut writer,
        total_terms,
        total_postings,
        grid_offset,
        sb_grid_offset,
        coarse_grid_offset,
        num_blocks_total as u32,
        dims,
        effective_block_size as u32,
        num_virtual_docs as u32,
        max_weight_scale,
        doc_map_offset,
        num_real_docs,
        grid_bits,
    )
    .map_err(crate::Error::Io)?;

    let blob_len = writer.offset() - blob_start;
    field_tocs.push(SparseFieldToc::bmp(
        field_id,
        total_vectors,
        blob_start,
        blob_len,
    ));

    log::info!(
        "[reorder_bmp] field {}: blockwise reorder done — {} blocks, {}, phases: data+offsets={:.1}s grids={:.1}s doc_maps={:.1}s total={:.1}s",
        field_id,
        num_blocks_total,
        crate::format_bytes(blob_len),
        block_data_elapsed.as_secs_f64(),
        grids_elapsed.as_secs_f64(),
        doc_maps_elapsed.as_secs_f64(),
        rewrite_start.elapsed().as_secs_f64(),
    );

    Ok((writer, field_tocs, converged))
}

/// Identity-copy a BMP field's raw blob (fields whose `reorder` schema
/// attribute is unset). Byte-identical: insertion order, padding, and footer
/// are all preserved.
fn copy_bmp_blob(
    bmp: &crate::segment::BmpIndex,
    field_id: u32,
    total_vectors: u32,
    writer: &mut OffsetWriter,
    field_tocs: &mut Vec<SparseFieldToc>,
) -> Result<()> {
    let blob = bmp.read_raw_blob().map_err(crate::Error::Io)?;
    let blob_start = writer.offset();
    const CHUNK: usize = 4 * 1024 * 1024;
    for chunk in blob.as_slice().chunks(CHUNK) {
        writer.write_all(chunk).map_err(crate::Error::Io)?;
    }
    let blob_len = writer.offset() - blob_start;

    field_tocs.push(SparseFieldToc::bmp(
        field_id,
        total_vectors,
        blob_start,
        blob_len,
    ));
    Ok(())
}

/// Identity-copy a MaxScore sparse field from source to destination.
#[allow(clippy::too_many_arguments)]
async fn copy_maxscore_field(
    sparse_idx: &crate::segment::SparseIndex,
    field_id: u32,
    quantization: crate::structures::WeightQuantization,
    writer: &mut OffsetWriter,
    field_tocs: &mut Vec<SparseFieldToc>,
    skip_writer: &mut (dyn Write + Send),
    skip_entry_buffer: &mut Vec<u8>,
    skip_count: &mut u32,
) -> Result<()> {
    let all_dims: Vec<u32> = sparse_idx.active_dimensions().collect();
    if all_dims.is_empty() {
        return Ok(());
    }

    let total_vectors = sparse_idx.total_vectors;
    let mut dim_toc_entries = Vec::with_capacity(all_dims.len());

    for &dim_id in &all_dims {
        let raw = match sparse_idx.read_dim_raw(dim_id).await? {
            Some(r) => r,
            None => continue,
        };

        if raw.raw_block_data.as_slice().is_empty() {
            continue;
        }

        let block_data_offset = writer.offset();
        let skip_start = *skip_count;
        let num_blocks = u32::try_from(raw.skip_entries.len()).map_err(|_| {
            crate::Error::Internal("sparse dimension skip-entry count exceeds u32::MAX".into())
        })?;

        // Write raw block data (identity copy)
        writer
            .write_all(raw.raw_block_data.as_slice())
            .map_err(crate::Error::Io)?;

        // Stream skip entries to the segment-scoped temp file. Offsets are
        // already correct for a single source.
        for entry in &raw.skip_entries {
            skip_entry_buffer.clear();
            entry.write_to_vec(skip_entry_buffer);
            skip_writer
                .write_all(skip_entry_buffer)
                .map_err(crate::Error::Io)?;
            *skip_count = skip_count.checked_add(1).ok_or_else(|| {
                crate::Error::Internal("sparse skip-entry count exceeds u32::MAX".into())
            })?;
        }

        dim_toc_entries.push(crate::segment::format::SparseDimTocEntry {
            dim_id,
            block_data_offset,
            skip_start,
            num_blocks,
            doc_count: raw.doc_count,
            max_weight: raw.global_max_weight,
        });
    }

    if !dim_toc_entries.is_empty() {
        field_tocs.push(SparseFieldToc {
            field_id,
            quantization: quantization as u8,
            total_vectors,
            dims: dim_toc_entries,
        });
    }

    Ok(())
}

/// Reorder one BMP field via Recursive Graph Bisection (BP), single- or
/// multi-source.
///
/// Reads block data from the source `BmpIndex`es (each paired with its doc-id
/// offset in the output segment), builds a combined forward index, runs BP to
/// compute a permutation over *real* (non-padding) records, then writes one
/// reordered blob. With a single `(bmp, 0)` source this is the standalone
/// segment reorder; with multiple sources it is the merge-time reorder path,
/// which replaces byte-level block stacking.
///
/// Realness is derived from each source's doc map (`build_vid_maps`), never
/// from `vid < num_real_docs` — block-copy merged sources carry interior
/// padding. Interior padding is compacted away in the output: the written
/// blob always has tail-only padding.
///
/// This is a synchronous function called from a Tokio block-in-place section,
/// so cancellation cannot detach its owned output writer. The
/// `OffsetWriter` streams directly to disk — no in-memory buffering of the
/// output blob.
///
/// When `rayon_pool` is `Some`, all rayon parallel work runs on that pool
/// instead of the global pool, bounding optimizer CPU usage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reorder_bmp_field(
    sources: &[(crate::segment::BmpIndex, u32)],
    field_id: u32,
    index_label: &str,
    field_name: &str,
    dims: u32,
    effective_block_size: usize,
    grid_bits: u8,
    max_weight_scale: f32,
    total_vectors: u32,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    cancellation: Option<Arc<std::sync::atomic::AtomicBool>>,
    granularity: BpGranularity,
    scratch_path: PathBuf,
    mut writer: OffsetWriter,
    mut field_tocs: Vec<SparseFieldToc>,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<(OffsetWriter, Vec<SparseFieldToc>, bool)> {
    if cancellation_requested(cancellation.as_deref()) {
        return Err(crate::Error::IndexClosed);
    }
    if !(1..=256).contains(&effective_block_size) {
        return Err(crate::Error::Internal(format!(
            "invalid BMP reorder block size {} (expected 1..=256)",
            effective_block_size
        )));
    }
    use crate::segment::builder::bmp::write_bmp_footer;
    use crate::segment::builder::graph_bisection::{
        BpProgressLabel, build_forward_index_from_bmps_with_maps, build_vid_maps,
        graph_bisection_with_progress,
    };

    if sources.is_empty() {
        return Ok((writer, field_tocs, true));
    }
    // Validation, coherence, forward construction, and rewrite are exhaustive
    // scans. Apply sequential page advice before the first one, not only after
    // two cold passes have already completed.
    let source_pages =
        crate::segment::reader::bmp::BmpScanPageGuard::new(sources.iter().map(|(bmp, _)| bmp));
    for (source, _) in sources {
        source.validate_rewrite_layout(
            "BMP reorder",
            dims,
            effective_block_size as u32,
            grid_bits,
            max_weight_scale,
        )?;
    }
    let validation_start = std::time::Instant::now();
    {
        use rayon::prelude::*;
        install_on_pool(rayon_pool.as_deref(), || {
            sources.par_iter().try_for_each(|(source, _)| {
                (0..source.num_blocks)
                    .into_par_iter()
                    .try_for_each(|block| source.validate_block_for_rewrite(block))
            })
        })?;
    }
    log::info!(
        "[reorder_bmp] field {}: validated {} source block(s) in {:.1}ms",
        field_id,
        sources
            .iter()
            .map(|(source, _)| u64::from(source.num_blocks))
            .sum::<u64>(),
        validation_start.elapsed().as_secs_f64() * 1000.0,
    );

    // ── Granularity decision (docs/block-level-reorder.md) ──────────────
    // The coherence scan runs only for `Auto`: explicit granularity (manual
    // override, or a deepening pass on an unconverged segment) must not pay
    // a header scan just to decide what is already decided.
    let effective_granularity = match granularity {
        BpGranularity::Auto => {
            let stats_start = std::time::Instant::now();
            let coherence = block_coherence(sources, effective_block_size, dims as usize);
            let chosen = if coherence.norm >= BLOCKWISE_NORM_COHERENCE_THRESHOLD {
                BpGranularity::Blocks
            } else {
                BpGranularity::Records
            };
            log::info!(
                "[reorder_bmp] field {}: coherence norm={:.3} (d={:.2}, rand={:.2}, max={:.2}, threshold {:.2}, {}/{} blocks scanned in {:.1}ms) → {:?} granularity",
                field_id,
                coherence.norm,
                coherence.d,
                coherence.d_rand,
                coherence.d_max,
                BLOCKWISE_NORM_COHERENCE_THRESHOLD,
                coherence.scanned_blocks,
                coherence.total_blocks,
                stats_start.elapsed().as_secs_f64() * 1000.0,
                chosen,
            );
            crate::observe::reorder_coherence(index_label, field_name, coherence.d, coherence.norm);
            chosen
        }
        explicit => {
            log::info!(
                "[reorder_bmp] field {}: {:?} granularity (explicit, coherence scan skipped)",
                field_id,
                explicit,
            );
            explicit
        }
    };
    crate::observe::reorder_granularity(
        index_label,
        field_name,
        match effective_granularity {
            BpGranularity::Blocks => "blocks",
            _ => "records",
        },
    );

    if effective_granularity == BpGranularity::Blocks {
        return reorder_bmp_field_blockwise(
            sources,
            field_id,
            index_label,
            field_name,
            dims,
            effective_block_size,
            grid_bits,
            max_weight_scale,
            total_vectors,
            memory_budget,
            bp_budget,
            cancellation.as_deref(),
            scratch_path,
            &source_pages,
            writer,
            field_tocs,
            rayon_pool,
        );
    }

    // Record-level BP needs both directions of every document map before it
    // can even start building the forward graph. Reserve that non-negotiable
    // peak up front; previously these vectors were allocated outside the
    // advertised budget. Fall back to a valid blockwise rewrite when the
    // record representation itself cannot fit, but keep the result marked
    // unconverged so a larger-budget/manual pass may still deepen it later.
    let record_map_bytes = sources.iter().fold(0usize, |total, (bmp, _)| {
        total
            .saturating_add((bmp.num_virtual_docs as usize).saturating_mul(4))
            .saturating_add((bmp.num_real_docs() as usize).saturating_mul(4))
    });
    let record_rewrite_fixed_bytes = sources.iter().fold(0usize, |total, (bmp, _)| {
        total
            // retained real→virtual map + output permutation
            .saturating_add((bmp.num_real_docs() as usize).saturating_mul(8))
            // output block-offset table
            .saturating_add((bmp.num_blocks as usize).saturating_mul(8))
    });
    let record_fixed_peak = record_map_bytes.max(record_rewrite_fixed_bytes);
    const MIN_RECORD_WORKSPACE: usize = 16 * 1024 * 1024;
    if record_fixed_peak > memory_budget.saturating_sub(MIN_RECORD_WORKSPACE) {
        log::warn!(
            "[reorder_bmp] field {}: record maps need {} of the {} total budget; falling back to blockwise order",
            field_id,
            crate::format_bytes(record_fixed_peak as u64),
            crate::format_bytes(memory_budget as u64),
        );
        let (writer, field_tocs, _) = reorder_bmp_field_blockwise(
            sources,
            field_id,
            index_label,
            field_name,
            dims,
            effective_block_size,
            grid_bits,
            max_weight_scale,
            total_vectors,
            memory_budget,
            bp_budget,
            cancellation.as_deref(),
            scratch_path,
            &source_pages,
            writer,
            field_tocs,
            rayon_pool,
        )?;
        return Ok((writer, field_tocs, false));
    }

    // ── Phase 1: Build forward index and run BP ─────────────────────────
    let bp_start = std::time::Instant::now();

    let bmp_refs: Vec<&crate::segment::BmpIndex> = sources.iter().map(|(b, _)| b).collect();
    // Build padding-aware maps once. Forward-index construction needs both
    // directions; output encoding reuses real→virtual instead of rescanning
    // every source document map.
    let mut vid_maps: Vec<(Vec<u32>, Vec<u32>)> = {
        use rayon::prelude::*;
        install_on_pool(rayon_pool.as_deref(), || {
            bmp_refs
                .par_iter()
                .map(|bmp| build_vid_maps(bmp))
                .collect::<Result<_>>()
        })?
    };
    // Prefix sums of real doc counts: source s owns global real ids
    // real_base[s]..real_base[s+1].
    let mut real_base = Vec::with_capacity(vid_maps.len() + 1);
    real_base.push(0usize);
    for (_, real_to_virtual) in &vid_maps {
        let next = real_base
            .last()
            .copied()
            .unwrap_or(0)
            .checked_add(real_to_virtual.len())
            .ok_or_else(|| {
                crate::Error::Internal("reordered BMP real-document count overflows usize".into())
            })?;
        real_base.push(next);
    }
    let num_real_docs = *real_base.last().unwrap();
    if num_real_docs == 0 {
        return Ok((writer, field_tocs, true));
    }
    if num_real_docs > u32::MAX as usize {
        return Err(crate::Error::Internal(
            "reordered BMP real-document count exceeds the V18 u32 format limit".into(),
        ));
    }

    log::info!(
        "[reorder_bmp] field {}: running BP on {} real docs from {} source(s)",
        field_id,
        num_real_docs,
        sources.len(),
    );

    // Zero time budget = identity permutation by construction — skip the
    // forward-index build entirely, it would be discarded unread. Reported
    // unconverged, matching budgeted-pass semantics (a follow-up pass
    // deepens).
    let zero_budget = bp_budget.time_budget.is_some_and(|d| d.is_zero());

    let (perm, converged) = if zero_budget {
        // No forward graph will consume virtual→real. Release it before
        // allocating the identity permutation; otherwise the zero-budget path
        // briefly holds both full map directions plus the permutation and can
        // exceed the same budget check that protects normal passes.
        for (virtual_to_real, _) in &mut vid_maps {
            *virtual_to_real = Vec::new();
        }
        log::info!(
            "[reorder_bmp] field {}: zero time budget — identity re-block, forward index skipped",
            field_id,
        );
        ((0..num_real_docs as u32).collect(), false)
    } else {
        let max_doc_freq = ((num_real_docs as f64) * 0.9) as usize;
        // Scale the df floor with segment size: a fixed 128 keeps only dims in
        // >=2.5% of docs on a 5k-doc segment — exactly the discriminative
        // mid-frequency dims BP needs — making reorder a near-no-op on small
        // segments. Rare dims are cheap (few postings), and the memory budget
        // already drops the highest-df dims when the forward index overflows.
        let min_doc_freq = (num_real_docs / 5000).clamp(2, 128);

        // Forward-index build is parallel too — keep it on the bounded pool.
        let forward_budget = memory_budget.saturating_sub(record_map_bytes);
        let build_forward = || {
            build_forward_index_from_bmps_with_maps(
                &bmp_refs,
                &vid_maps,
                min_doc_freq,
                max_doc_freq.max(1),
                forward_budget,
            )
        };
        let fwd = if let Some(ref pool) = rayon_pool {
            pool.install(build_forward)
        } else {
            build_forward()
        };

        log::info!(
            "[reorder_bmp] field {}: forward index built in {:.1}ms ({} terms, {} postings)",
            field_id,
            bp_start.elapsed().as_secs_f64() * 1000.0,
            fwd.num_terms,
            fwd.total_postings(),
        );

        // Forward construction no longer needs virtual→real. Release those
        // potentially hundreds of MB before graph scratch reaches its peak;
        // retain real→virtual for output encoding.
        for (virtual_to_real, _) in &mut vid_maps {
            *virtual_to_real = Vec::new();
        }

        if fwd.num_terms > 0 && num_real_docs > effective_block_size {
            let graph_start = std::time::Instant::now();
            let graph_budget = crate::segment::BpBudget {
                min_partition_docs: bp_budget.min_partition_docs,
                time_budget: bp_budget
                    .time_budget
                    .map(|budget| budget.saturating_sub(bp_start.elapsed())),
            };
            let run_graph = || {
                graph_bisection_with_progress(
                    &fwd,
                    effective_block_size,
                    20,
                    graph_budget,
                    cancellation.as_deref(),
                    BpProgressLabel {
                        index: index_label,
                        field: field_name,
                        entity_kind: "records",
                    },
                )
            };
            let (perm, converged) = if let Some(ref pool) = rayon_pool {
                pool.install(run_graph)
            } else {
                run_graph()
            };
            log::info!(
                "[reorder_bmp] field {}: BP completed in {:.1}ms (converged={})",
                field_id,
                graph_start.elapsed().as_secs_f64() * 1000.0,
                converged,
            );
            (perm, converged)
        } else {
            (
                (0..num_real_docs as u32).collect(),
                num_real_docs <= effective_block_size || !fwd.budget_limited(),
            )
        }
    };

    if cancellation_requested(cancellation.as_deref()) {
        return Err(crate::Error::IndexClosed);
    }

    let real_to_virtual: Vec<Vec<u32>> = vid_maps.into_iter().map(|(_, real)| real).collect();

    log::info!(
        "[reorder_bmp] field {}: writing reordered blob ({} blocks)",
        field_id,
        num_real_docs.div_ceil(effective_block_size),
    );
    source_pages.switch_to_random();

    // ── Phase 2: Write new blob with records in permuted order ──────────
    let new_num_blocks = num_real_docs.div_ceil(effective_block_size);
    if new_num_blocks > u32::MAX as usize {
        return Err(crate::Error::Internal(
            "reordered BMP block count exceeds the V18 u32 format limit".into(),
        ));
    }
    let new_num_virtual_docs = new_num_blocks
        .checked_mul(effective_block_size)
        .filter(|&count| count <= u32::MAX as usize)
        .ok_or_else(|| {
            crate::Error::Internal(
                "reordered BMP virtual-document count exceeds the V18 u32 format limit".into(),
            )
        })?;

    let rewrite_start = std::time::Instant::now();
    let blob_start = writer.offset();
    let mut block_data_starts: Vec<u64> = Vec::with_capacity(new_num_blocks + 1);
    let mut total_terms: u64 = 0;
    let mut total_postings: u64 = 0;
    let mut cumulative_bytes: u64 = 0;

    // Grid entries and parallel encoded blocks coexist with the retained
    // real→virtual map and permutation. Divide the actual remaining pass
    // budget between them instead of adding fixed 512+256 MB allocations on
    // top of `--bp-memory-budget-mb`.
    let persistent_rewrite_bytes = perm
        .capacity()
        .saturating_mul(std::mem::size_of::<u32>())
        .saturating_add(real_to_virtual.iter().fold(0usize, |total, map| {
            total.saturating_add(map.capacity().saturating_mul(std::mem::size_of::<u32>()))
        }))
        .saturating_add(
            block_data_starts
                .capacity()
                .saturating_mul(std::mem::size_of::<u64>()),
        );
    let rewrite_workspace = memory_budget.saturating_sub(persistent_rewrite_bytes);
    let grid_entries_budget =
        (rewrite_workspace.saturating_mul(2) / 3).clamp(1024 * 1024, 512 * 1024 * 1024);
    let encode_window_budget = rewrite_workspace
        .saturating_sub(grid_entries_budget)
        .clamp(1024 * 1024, 256 * 1024 * 1024);

    // Grid entries use bounded sorted runs. Reserving the complete run up
    // front prevents Vec's geometric growth from overshooting the advertised
    // memory budget just before a spill.
    let grid_run_entries =
        (grid_entries_budget / std::mem::size_of::<BmpGridEntry>()).clamp(1, MAX_GRID_RUN_ENTRIES);
    let mut grid_entries: Vec<BmpGridEntry> = Vec::with_capacity(grid_run_entries);
    let mut run_files = GridScratchFiles::new(field_id, scratch_path);

    // Transpose records through exact-counted flat tuples. Block encoding
    // streams directly to the buffered output, so admission covers only the
    // live routes/jobs/counts/partitions/tuples plus one reusable write buffer.
    let posting_buffer_bytes = (encode_window_budget / 16)
        .clamp(64 * 1024, REWRITE_IO_BUFFER_BYTES)
        .min(encode_window_budget.saturating_div(2))
        .max(2);
    let transpose_budget = encode_window_budget.saturating_sub(posting_buffer_bytes);
    let mut posting_buffer = Vec::with_capacity(posting_buffer_bytes);
    let total_source_postings = sources.iter().fold(0u64, |total, (source, _)| {
        total.saturating_add(source.total_postings())
    });
    let average_postings = usize::try_from(
        total_source_postings.div_ceil(u64::try_from(num_real_docs).unwrap_or(u64::MAX)),
    )
    .unwrap_or(usize::MAX);
    let estimated_record_bytes = std::mem::size_of::<RecordRoute>()
        .saturating_add(average_postings.saturating_mul(std::mem::size_of::<RoutedPosting>()))
        // A fully scattered output window can touch one distinct source block
        // per routed record, so speculative route planning must charge job,
        // count, and partition metadata per record too.
        .saturating_add(std::mem::size_of::<SourceBlockJob>())
        .saturating_add(std::mem::size_of::<usize>())
        .saturating_add(std::mem::size_of::<(&SourceBlockJob, &mut [RoutedPosting])>());
    let estimated_block_bytes = effective_block_size
        .saturating_mul(estimated_record_bytes)
        .max(1);
    let max_parallel_window = transpose_budget
        .checked_div(estimated_block_bytes)
        .unwrap_or(0)
        .max(1);
    let mut source_block_scans = 0u64;
    let mut window_start = 0usize;
    while window_start < new_num_blocks {
        if cancellation_requested(cancellation.as_deref()) {
            return Err(crate::Error::IndexClosed);
        }
        let initial_end = window_start
            .saturating_add(max_parallel_window)
            .min(new_num_blocks);
        let mut window_end = initial_end;
        let (routes, jobs, counts, posting_count, admitted_bytes) = loop {
            let (routes, jobs) = build_record_route_plan(
                sources,
                &perm,
                &real_base,
                &real_to_virtual,
                effective_block_size,
                num_real_docs,
                window_start,
                window_end,
                rayon_pool.as_deref(),
            )?;
            let counts = {
                use rayon::prelude::*;
                install_on_pool(rayon_pool.as_deref(), || {
                    jobs.par_iter()
                        .map(|job| count_source_job_postings(sources, job, &routes))
                        .collect::<Result<Vec<_>>>()
                })?
            };
            let posting_count = counts.iter().try_fold(0usize, |total, &count| {
                total.checked_add(count).ok_or_else(|| {
                    crate::Error::Internal("BMP routed posting count overflows usize".into())
                })
            })?;
            let required = record_window_memory_bytes(&routes, &jobs, &counts, posting_count)
                .ok_or_else(|| {
                    crate::Error::Internal("BMP route-window memory size overflows usize".into())
                })?;
            if required <= transpose_budget {
                break (routes, jobs, counts, posting_count, required);
            }
            let window_blocks = window_end - window_start;
            if window_blocks == 1 {
                return Err(crate::Error::Internal(format!(
                    "one BMP output block needs {} of the {} record-transpose budget",
                    crate::format_bytes(required as u64),
                    crate::format_bytes(transpose_budget as u64),
                )));
            }
            window_end = window_start + (window_blocks / 2).max(1);
        };
        if window_end < initial_end {
            log::debug!(
                "[reorder_bmp] field {}: record window reduced from {} to {} blocks ({} admitted of {} budget)",
                field_id,
                initial_end - window_start,
                window_end - window_start,
                crate::format_bytes(admitted_bytes as u64),
                crate::format_bytes(transpose_budget as u64),
            );
        }
        source_block_scans = source_block_scans.saturating_add(jobs.len() as u64);

        let mut routed = vec![RoutedPosting::default(); posting_count];
        let mut partitions = Vec::with_capacity(jobs.len());
        let mut remaining = routed.as_mut_slice();
        for (job, &count) in jobs.iter().zip(&counts) {
            let (output, tail) = remaining.split_at_mut(count);
            partitions.push((job, output));
            remaining = tail;
        }
        debug_assert!(remaining.is_empty());
        {
            use rayon::prelude::*;
            install_on_pool(rayon_pool.as_deref(), || {
                partitions.into_par_iter().try_for_each(|(job, output)| {
                    fill_source_job_postings(sources, job, &routes, output)
                })
            })?;
            install_on_pool(rayon_pool.as_deref(), || {
                routed.par_sort_unstable_by_key(|posting| {
                    (
                        posting.out_local,
                        posting.dimension,
                        posting.slot,
                        posting.impact,
                    )
                });
            });
        }

        let mut posting_cursor = 0usize;
        for out_local in 0..window_end - window_start {
            let block_posting_start = posting_cursor;
            while posting_cursor < routed.len()
                && routed[posting_cursor].out_local == out_local as u32
            {
                posting_cursor += 1;
            }
            let block_postings = &routed[block_posting_start..posting_cursor];
            block_data_starts.push(cumulative_bytes);
            if block_postings.is_empty() {
                continue;
            }
            let global_block = u32::try_from(window_start + out_local)
                .map_err(|_| crate::Error::Internal("BMP output block exceeds u32::MAX".into()))?;
            let (block_bytes, block_terms) = write_sorted_routed_block(
                global_block,
                block_postings,
                &mut grid_entries,
                grid_run_entries,
                &mut run_files,
                field_id,
                rayon_pool.as_deref(),
                &mut posting_buffer,
                &mut writer,
            )?;
            total_terms = total_terms.saturating_add(block_terms as u64);
            total_postings = total_postings.saturating_add(block_postings.len() as u64);
            cumulative_bytes = cumulative_bytes.saturating_add(block_bytes);
        }
        debug_assert_eq!(posting_cursor, routed.len());
        window_start = window_end;
    }

    // Sentinel
    block_data_starts.push(cumulative_bytes);

    if total_terms == 0 {
        return Ok((writer, field_tocs, converged));
    }

    // ── Write remaining sections ────────────────────────────────────────

    let block_data_len = writer.offset() - blob_start;
    let padding = (8 - (block_data_len % 8) as usize) % 8;
    if padding > 0 {
        writer
            .write_all(&[0u8; 8][..padding])
            .map_err(crate::Error::Io)?;
    }

    // Section A: block_data_starts
    crate::segment::builder::bmp::write_u64_slice_le(&mut writer, &block_data_starts)
        .map_err(crate::Error::Io)?;
    drop(block_data_starts);
    let block_data_elapsed = rewrite_start.elapsed();

    // Sections D+E+H: block, superblock, and coarse-superblock grids.
    if cancellation_requested(cancellation.as_deref()) {
        return Err(crate::Error::IndexClosed);
    }
    let grids_start = std::time::Instant::now();
    let grid_offset = writer.offset() - blob_start;
    let (packed_bytes, sb_bytes, _coarse_bytes) = write_reorder_grids(
        grid_entries,
        &mut run_files,
        field_id,
        dims as usize,
        new_num_blocks,
        grid_bits,
        rayon_pool.as_deref(),
        &mut writer,
        cancellation.as_deref(),
    )?;
    let sb_grid_offset = grid_offset + packed_bytes;
    let coarse_grid_offset = sb_grid_offset + sb_bytes;
    // The grids are durable in the output stream now; keep no external-sort
    // scratch alive during the doc-map/footer phases.
    drop(run_files);
    let grids_elapsed = grids_start.elapsed();

    // Sections F+G: doc_map [new_num_virtual_docs each]
    if cancellation_requested(cancellation.as_deref()) {
        return Err(crate::Error::IndexClosed);
    }
    let doc_maps_start = std::time::Instant::now();
    let doc_map_offset = writer.offset() - blob_start;

    write_record_doc_maps(
        sources,
        &perm[..num_real_docs],
        &real_base,
        &real_to_virtual,
        new_num_virtual_docs,
        &mut writer,
        rayon_pool.as_deref(),
        cancellation.as_deref(),
    )?;
    let doc_maps_elapsed = doc_maps_start.elapsed();

    // V18 footer.
    write_bmp_footer(
        &mut writer,
        total_terms,
        total_postings,
        grid_offset,
        sb_grid_offset,
        coarse_grid_offset,
        new_num_blocks as u32,
        dims,
        effective_block_size as u32,
        new_num_virtual_docs as u32,
        max_weight_scale,
        doc_map_offset,
        num_real_docs as u32,
        grid_bits,
    )
    .map_err(crate::Error::Io)?;

    let blob_len = writer.offset() - blob_start;

    field_tocs.push(SparseFieldToc::bmp(
        field_id,
        total_vectors,
        blob_start,
        blob_len,
    ));

    log::info!(
        "[reorder_bmp] field {}: done — {} blocks, {} terms, {} postings, {}, source-block scans={}, phases: transpose+data+offsets={:.1}s grids={:.1}s doc_maps={:.1}s total={:.1}s",
        field_id,
        new_num_blocks,
        total_terms,
        total_postings,
        crate::format_bytes(blob_len),
        source_block_scans,
        block_data_elapsed.as_secs_f64(),
        grids_elapsed.as_secs_f64(),
        doc_maps_elapsed.as_secs_f64(),
        rewrite_start.elapsed().as_secs_f64(),
    );

    Ok((writer, field_tocs, converged))
}

#[cfg(test)]
mod grid_run_prefix_tests {
    use crate::directories::{Directory, DirectoryWriter};

    #[tokio::test]
    async fn segment_copy_distinguishes_optional_and_required_missing_files() {
        let dir = crate::directories::RamDirectory::new();
        let missing = std::path::Path::new("missing");
        let output = std::path::Path::new("output");

        super::clone_segment_file(&dir, missing, output, false, None)
            .await
            .unwrap();
        assert!(!dir.exists(output).await.unwrap());

        let error = super::clone_segment_file(&dir, missing, output, true, None)
            .await
            .unwrap_err();
        assert!(matches!(error, crate::Error::Corruption(_)));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn segment_copy_preserves_all_bytes() {
        let dir = crate::directories::RamDirectory::new();
        let source = std::path::Path::new("source");
        let output = std::path::Path::new("output");
        let data = vec![0x5a; 4 * 1024 * 1024 + 17];
        dir.write(source, &data).await.unwrap();

        super::clone_segment_file(&dir, source, output, true, None)
            .await
            .unwrap();

        let copied = dir
            .open_read(output)
            .await
            .unwrap()
            .read_bytes()
            .await
            .unwrap();
        assert_eq!(copied.as_slice(), data);
    }

    /// Regression: run prefixes were `pid_field` — identical for every pass
    /// in a container (PID 1) reordering the same field id, so concurrent
    /// passes clobbered and deleted each other's spill files (prod ENOENT).
    #[test]
    fn test_grid_run_prefix_unique_per_pass() {
        let a = super::grid_run_prefix(3);
        let b = super::grid_run_prefix(3);
        assert_ne!(
            a, b,
            "two passes over the same field must not share spill files"
        );
    }

    #[test]
    fn grid_run_files_remove_partial_spills_on_drop() {
        let directory = tempfile::tempdir().unwrap();
        let path = {
            let mut runs =
                super::GridScratchFiles::new(7, directory.path().join("seg_test.sparse"));
            let path = runs.allocate_run();
            std::fs::write(&path, b"partial run").unwrap();
            assert!(path.exists());
            path
        };
        assert!(
            !path.exists(),
            "spill owner must remove files on unwind/drop"
        );
    }
}
