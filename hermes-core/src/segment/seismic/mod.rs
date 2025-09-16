//! Immutable Seismic nomination runs with a single logical sparse-vector owner.
//!
//! Runs are independently addressable. Merge copies their bytes and remaps the
//! small run directory; it never reclusters vectors or changes their precision.
use crate::directories::OwnedBytes;
use crate::segment::logical_address::LogicalUnit;
#[cfg(any(feature = "native", feature = "wasm", test))]
use crate::structures::SparseVectorConfig;
use crate::structures::WeightQuantization;
use crate::{Error, Result};
#[cfg(any(feature = "native", feature = "wasm", test))]
use std::io::Write;

#[cfg(any(feature = "native", feature = "wasm", test))]
mod build;
mod forward;
#[cfg(any(feature = "native", test))]
mod maintain;
mod occurrences;
mod parse;
#[cfg(feature = "native")]
mod residency;
mod term;
#[cfg(any(feature = "native", test))]
pub(crate) use maintain::{write_compacted, write_maintained_partition};
pub(crate) use term::TermRun;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use build::build_blob;
#[cfg(any(feature = "native", feature = "wasm", test))]
pub(crate) use build::build_blob_with_keys;
pub(crate) use forward::ForwardVector;
#[cfg(test)]
pub(crate) use tests::{MemoryWriter, copy_sources_for_test};

const VERSION: u32 = 5;
pub(crate) const PARTITIONS: usize = 16;

#[cfg(any(feature = "native", feature = "wasm", test))]
pub(crate) trait SeismicWriter {
    fn root(&mut self) -> &mut dyn Write;
    fn partition(&mut self, id: usize) -> &mut dyn Write;
}

#[cfg(any(feature = "native", feature = "wasm", test))]
#[derive(Default)]
pub(crate) struct OutputLengths {
    pub(crate) root: u64,
    pub(crate) partitions: [u64; PARTITIONS],
}

#[derive(Clone)]
struct Partition {
    bytes: OwnedBytes,
    #[cfg(any(feature = "native", test))]
    source_offset: u64,
    runs: Vec<Run>,
    #[cfg(any(feature = "native", test))]
    pending_terms: u32,
    #[cfg(any(feature = "native", test))]
    pending_bytes: u64,
}

const MAGIC: u32 = 0x314d5353;
const RUN_MAGIC: u32 = 0x31524d53;
const FOOTER: usize = 40;
const RUN_FOOTER: usize = 32;
const RUN_ENTRY: usize = 32;
const ROW_ENTRY: usize = 24;
const TERM_ENTRY: usize = 40;

fn corrupt(s: &str) -> Error {
    Error::Corruption(format!("Seismic: {s}; rebuild the index"))
}
fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}
#[cfg(any(feature = "native", feature = "wasm", test))]
fn put32(w: &mut (impl Write + ?Sized), n: u32) -> std::io::Result<()> {
    w.write_all(&n.to_le_bytes())
}
#[cfg(any(feature = "native", feature = "wasm", test))]
fn put64(w: &mut (impl Write + ?Sized), n: u64) -> std::io::Result<()> {
    w.write_all(&n.to_le_bytes())
}

#[derive(Clone)]
struct Run {
    bytes: OwnedBytes,
    #[cfg(any(feature = "native", test))]
    offset: u64,
    doc_base: u32,
    row_base: u32,
    #[cfg(any(feature = "native", test))]
    rows: u32,
    row_directory: OwnedBytes,
    term_directory: OwnedBytes,
    #[cfg(test)]
    term_offset: usize,
    terms: u32,
}

#[derive(Clone)]
pub(crate) struct SeismicIndex {
    bytes: OwnedBytes,
    source_offset: u64,
    runs: Vec<Run>,
    partitions: Vec<Option<Partition>>,
    dims: u32,
    quantization: WeightQuantization,
    postings: u32,
    cluster_size: u32,
    summary_energy: f32,
    rows: u32,
    single_valued: bool,
    pending_terms: u32,
    pending_bytes: u64,
    forward_entries: u64,
}
impl SeismicIndex {
    pub(crate) fn set_source_offset(&mut self, offset: u64) {
        self.source_offset = offset;
    }
    #[cfg(any(feature = "native", test))]
    pub(crate) fn source_offset(&self) -> u64 {
        self.source_offset
    }
    pub(crate) fn is_single_valued(&self) -> bool {
        self.single_valued
    }
    pub(crate) fn quantization(&self) -> WeightQuantization {
        self.quantization
    }
    pub(crate) fn vector_byte_len(&self, row: u32) -> Result<usize> {
        if row >= self.rows {
            return Err(corrupt("forward row out of bounds"));
        }
        let (r, at) = self.locate(row);
        Ok(u32_at(&r.row_directory, at + 16) as usize)
    }
    pub(crate) fn len(&self) -> u32 {
        self.rows
    }
    pub(crate) fn forward_entries(&self) -> u64 {
        self.forward_entries
    }
    pub(crate) fn total_vectors(&self) -> u32 {
        self.rows
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.rows == 0
    }
    pub(crate) fn dims(&self) -> u32 {
        self.dims
    }
    pub(crate) fn run_count(&self) -> usize {
        self.runs.len()
    }
    #[cfg(test)]
    pub(crate) fn run_ranges(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.runs.iter().map(|r| (r.offset, r.bytes.len() as u64))
    }
    pub(crate) fn encoded_bytes(&self) -> usize {
        self.bytes.len()
            + self
                .partitions
                .iter()
                .flatten()
                .map(|p| p.bytes.len())
                .sum::<usize>()
    }
    pub(crate) fn estimated_heap_bytes(&self) -> usize {
        (self.runs.capacity()
            + self
                .partitions
                .iter()
                .flatten()
                .map(|p| p.runs.capacity())
                .sum::<usize>())
            * std::mem::size_of::<Run>()
            + self.partitions.capacity() * std::mem::size_of::<Option<Partition>>()
    }
    fn locate(&self, row: u32) -> (&Run, usize) {
        assert!(row < self.rows);
        let i = self.runs.partition_point(|r| r.row_base <= row) - 1;
        let run = &self.runs[i];
        (run, (row - run.row_base) as usize * ROW_ENTRY)
    }
    pub(crate) fn key(&self, row: u32) -> LogicalUnit {
        let (r, at) = self.locate(row);
        LogicalUnit {
            doc: u32_at(&r.row_directory, at) + r.doc_base,
            ordinal: u16::from_le_bytes(r.row_directory[at + 4..at + 6].try_into().unwrap()),
        }
    }
    pub(crate) fn vector(&self, row: u32) -> ForwardVector<'_> {
        let (r, at) = self.locate(row);
        let start = u64_at(&r.row_directory, at + 8) as usize;
        let len = u32_at(&r.row_directory, at + 16) as usize;
        ForwardVector::new(
            &r.bytes[start..start + len],
            u32_at(&r.row_directory, at + 20) as usize,
            self.quantization,
        )
    }
    pub(crate) fn rows_for_document(&self, doc: u32) -> impl Iterator<Item = u32> + '_ {
        let mut lo = 0;
        let mut hi = self.rows;
        while lo < hi {
            let m = lo + (hi - lo) / 2;
            if self.key(m).doc < doc {
                lo = m + 1;
            } else {
                hi = m;
            }
        }
        (lo..self.rows).take_while(move |&r| self.key(r).doc == doc)
    }
    #[cfg(test)]
    pub(crate) fn for_document(&self, doc: u32) -> impl Iterator<Item = (u16, u32)> + '_ {
        self.rows_for_document(doc)
            .map(|r| (self.key(r).ordinal, r))
    }
    pub(crate) fn term_runs(&self, dim: u32) -> impl Iterator<Item = TermRun<'_>> {
        self.partitions[dim as usize % PARTITIONS]
            .iter()
            .flat_map(|p| &p.runs)
            .flat_map(move |r| {
                let mut lo = 0;
                let mut hi = r.terms as usize;
                while lo < hi {
                    let m = lo + (hi - lo) / 2;
                    if u32_at(&r.term_directory, m * TERM_ENTRY) < dim {
                        lo = m + 1;
                    } else {
                        hi = m;
                    }
                }
                (lo..r.terms as usize)
                    .take_while(move |&i| u32_at(&r.term_directory, i * TERM_ENTRY) == dim)
                    .map(move |i| {
                        let at = i * TERM_ENTRY;
                        let start = u64_at(&r.term_directory, at + 8) as usize;
                        let len = u64_at(&r.term_directory, at + 16) as usize;
                        TermRun::new(
                            &r.bytes[start..start + len],
                            r.row_base + u32_at(&r.term_directory, at + 24),
                        )
                    })
            })
    }
    /// Complete nonzero forward-vector frequency, independent of top-L pruning.
    pub(crate) fn doc_count(&self, dim: u32) -> u32 {
        self.partitions[dim as usize % PARTITIONS]
            .iter()
            .flat_map(|p| &p.runs)
            .map(|r| {
                let mut lo = 0;
                let mut hi = r.terms as usize;
                while lo < hi {
                    let m = lo + (hi - lo) / 2;
                    if u32_at(&r.term_directory, m * TERM_ENTRY) < dim {
                        lo = m + 1;
                    } else {
                        hi = m;
                    }
                }
                (lo..r.terms as usize)
                    .take_while(|&i| u32_at(&r.term_directory, i * TERM_ENTRY) == dim)
                    .map(|i| u32_at(&r.term_directory, i * TERM_ENTRY + 32))
                    .sum::<u32>()
            })
            .sum()
    }
    #[cfg(any(feature = "native", test))]
    pub(crate) fn partition_maintenance_priority(&self, id: usize) -> (u32, u64) {
        self.partitions[id]
            .as_ref()
            .map_or((0, 0), |p| (p.pending_terms, p.pending_bytes))
    }
    #[cfg(any(feature = "native", test))]
    pub(crate) fn partition_source_offset(&self, id: usize) -> u64 {
        self.partitions[id].as_ref().unwrap().source_offset
    }
    pub(crate) fn pending_terms(&self) -> u32 {
        self.pending_terms
    }
    pub(crate) fn cluster_count(&self) -> u64 {
        self.partitions
            .iter()
            .flatten()
            .flat_map(|p| &p.runs)
            .map(|r| {
                r.term_directory
                    .chunks_exact(TERM_ENTRY)
                    .map(|t| u32_at(t, 4) as u64)
                    .sum::<u64>()
            })
            .sum()
    }
    pub(crate) fn nomination_count(&self) -> u64 {
        self.partitions
            .iter()
            .flatten()
            .flat_map(|p| &p.runs)
            .map(|r| {
                (0..r.terms as usize)
                    .map(|i| {
                        let at = i * TERM_ENTRY;
                        let start = u64_at(&r.term_directory, at + 8) as usize;
                        let len = u64_at(&r.term_directory, at + 16) as usize;
                        TermRun::new(&r.bytes[start..start + len], r.row_base).nomination_count()
                            as u64
                    })
                    .sum::<u64>()
            })
            .sum()
    }
}

#[cfg(any(feature = "native", feature = "wasm", test))]
fn write_footer(
    w: &mut (impl Write + ?Sized),
    settings: &SeismicIndex,
    rows: u32,
    runs: u32,
    dims: u32,
    partition: Option<usize>,
) -> std::io::Result<()> {
    for v in [
        MAGIC,
        VERSION,
        dims,
        settings.quantization as u32,
        settings.postings,
        settings.cluster_size,
        settings.summary_energy.to_bits(),
        rows,
        runs,
        partition.map_or(0, |id| id as u32 + 1),
    ] {
        put32(w, v)?;
    }
    Ok(())
}
/// Finish a single forward or nomination run using the shared directory envelope.
#[cfg(any(feature = "native", feature = "wasm", test))]
fn finish_run(
    writer: &mut dyn Write,
    mut offset: u64,
    row_offset: u64,
    term_offset: u64,
    terms: u32,
    settings: &SeismicIndex,
    partition: Option<usize>,
) -> Result<u64> {
    put32(writer, settings.rows)?;
    put32(writer, terms)?;
    put64(writer, row_offset)?;
    put64(writer, term_offset)?;
    put32(writer, RUN_MAGIC)?;
    put32(writer, 0)?;
    offset += RUN_FOOTER as u64;
    put64(writer, 0)?;
    put64(writer, offset)?;
    for value in [0, 0, settings.rows, 0] {
        put32(writer, value)?;
    }
    write_footer(writer, settings, settings.rows, 1, settings.dims, partition)?;
    Ok(offset + RUN_ENTRY as u64 + FOOTER as u64)
}

/// Representation-preserving merge. Scratch is one fixed-width entry per run.
#[cfg(any(feature = "native", test))]
pub(crate) fn write_sources(
    sources: &[(&SeismicIndex, u32)],
    writer: &mut dyn Write,
    check_cancel: &impl Fn() -> Result<()>,
) -> Result<u64> {
    write_sources_with_copy(sources, writer, check_cancel, |_, _, bytes, writer| {
        for chunk in bytes.chunks(4 * 1024 * 1024) {
            check_cancel()?;
            writer.write_all(chunk)?;
        }
        Ok(())
    })
}

#[cfg(any(feature = "native", test))]
pub(crate) fn write_sources_with_copy<W: Write + ?Sized>(
    sources: &[(&SeismicIndex, u32)],
    writer: &mut W,
    check_cancel: &impl Fn() -> Result<()>,
    copy: impl FnMut(usize, u64, &[u8], &mut W) -> Result<()>,
) -> Result<u64> {
    write_component_sources(sources, None, writer, check_cancel, copy)
}

#[cfg(any(feature = "native", test))]
pub(crate) fn write_partition_sources_with_copy<W: Write + ?Sized>(
    sources: &[(&SeismicIndex, u32)],
    partition: usize,
    writer: &mut W,
    check_cancel: &impl Fn() -> Result<()>,
    copy: impl FnMut(usize, u64, &[u8], &mut W) -> Result<()>,
) -> Result<u64> {
    write_component_sources(sources, Some(partition), writer, check_cancel, copy)
}

#[cfg(any(feature = "native", test))]
pub(crate) fn write_partition_sources(
    sources: &[(&SeismicIndex, u32)],
    partition: usize,
    writer: &mut dyn Write,
    check_cancel: &impl Fn() -> Result<()>,
) -> Result<u64> {
    write_partition_sources_with_copy(
        sources,
        partition,
        writer,
        check_cancel,
        |_, _, bytes, w| {
            for chunk in bytes.chunks(4 * 1024 * 1024) {
                check_cancel()?;
                w.write_all(chunk)?;
            }
            Ok(())
        },
    )
}

#[cfg(any(feature = "native", test))]
fn write_component_sources<W: Write + ?Sized>(
    sources: &[(&SeismicIndex, u32)],
    partition: Option<usize>,
    writer: &mut W,
    check_cancel: &impl Fn() -> Result<()>,
    mut copy: impl FnMut(usize, u64, &[u8], &mut W) -> Result<()>,
) -> Result<u64> {
    let first = sources
        .first()
        .ok_or_else(|| corrupt("merge has no sources"))?
        .0;
    let mut entries = Vec::new();
    let mut offset = 0u64;
    let mut rows = 0u32;
    let mut last = None;
    for (source_index, &(source, doc_offset)) in sources.iter().enumerate() {
        if source.quantization != first.quantization
            || source.postings != first.postings
            || source.cluster_size != first.cluster_size
            || source.summary_energy.to_bits() != first.summary_energy.to_bits()
        {
            return Err(corrupt("incompatible merge settings"));
        }
        if !source.is_empty() {
            let first_key = source.key(0);
            let end_key = source.key(source.len() - 1);
            let start = LogicalUnit {
                doc: first_key
                    .doc
                    .checked_add(doc_offset)
                    .ok_or_else(|| corrupt("document overflow"))?,
                ..first_key
            };
            let end = LogicalUnit {
                doc: end_key
                    .doc
                    .checked_add(doc_offset)
                    .ok_or_else(|| corrupt("document overflow"))?,
                ..end_key
            };
            if last.is_some_and(|old| old >= start) {
                return Err(corrupt("overlapping source documents"));
            }
            last = Some(end);
        }
        let runs = match partition {
            Some(id) => {
                &source.partitions[id]
                    .as_ref()
                    .ok_or_else(|| corrupt("missing nomination partition"))?
                    .runs
            }
            None => &source.runs,
        };
        for run in runs {
            check_cancel()?;
            let base = run
                .doc_base
                .checked_add(doc_offset)
                .ok_or_else(|| corrupt("document overflow"))?;
            entries.push((offset, run.bytes.len() as u64, base, rows, run.rows));
            rows = rows
                .checked_add(run.rows)
                .ok_or_else(|| corrupt("row overflow"))?;
            copy(source_index, run.offset, &run.bytes, writer)?;
            offset = offset
                .checked_add(run.bytes.len() as u64)
                .ok_or_else(|| corrupt("length overflow"))?;
        }
    }
    for &(start, len, doc, row, count) in &entries {
        put64(writer, start)?;
        put64(writer, len)?;
        for v in [doc, row, count, 0] {
            put32(writer, v)?;
        }
    }
    write_footer(
        writer,
        first,
        rows,
        u32::try_from(entries.len()).map_err(|_| corrupt("too many runs"))?,
        sources
            .iter()
            .map(|(s, _)| s.dims())
            .max()
            .unwrap_or(first.dims()),
        partition,
    )?;
    Ok(offset + entries.len() as u64 * RUN_ENTRY as u64 + FOOTER as u64)
}
