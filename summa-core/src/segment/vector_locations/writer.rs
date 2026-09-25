use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use super::{BLOCK_SIZE, ENTRY_SIZE, HEADER_SIZE, MAGIC, SPAN_SIZE, VERSION, u64_at};

// 1 MiB of records; at most 16 buffered readers per merge. Multi-level
// carries bound open spill files as well as heap use for large corpora.
const RUN_ROWS: usize = 65_536;
const FAN_IN: usize = 16;
const IO_BUFFER: usize = 4 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Location {
    doc: u32,
    ordinal: u16,
    address: u64,
}

impl Location {
    fn write(self, out: &mut (impl Write + ?Sized)) -> io::Result<()> {
        out.write_all(&self.doc.to_le_bytes())?;
        out.write_all(&self.ordinal.to_le_bytes())?;
        out.write_all(&self.address.to_le_bytes())
    }

    fn read(input: &mut impl Read) -> io::Result<Option<Self>> {
        let mut bytes = [0; ENTRY_SIZE];
        loop {
            match input.read(&mut bytes[..1]) {
                Ok(0) => return Ok(None),
                Ok(_) => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        input.read_exact(&mut bytes[1..])?;
        Ok(Some(Self {
            doc: u32::from_le_bytes(bytes[..4].try_into().unwrap()),
            ordinal: u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
            address: u64::from_le_bytes(bytes[6..].try_into().unwrap()),
        }))
    }
}

fn check(cancel: Option<&AtomicBool>) -> io::Result<()> {
    if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "vector location write cancelled",
        ))
    } else {
        Ok(())
    }
}

fn merge_runs(
    runs: Vec<File>,
    mut emit: impl FnMut(Location) -> io::Result<()>,
    cancel: Option<&AtomicBool>,
) -> io::Result<()> {
    let mut readers: Vec<_> = runs
        .into_iter()
        .map(|file| BufReader::with_capacity(IO_BUFFER, file))
        .collect();
    let mut heap = BinaryHeap::with_capacity(readers.len());
    for (i, reader) in readers.iter_mut().enumerate() {
        reader.seek(SeekFrom::Start(0))?;
        if let Some(row) = Location::read(reader)? {
            heap.push(Reverse((row, i)));
        }
    }
    let mut count = 0usize;
    while let Some(Reverse((row, i))) = heap.pop() {
        if count.is_multiple_of(4096) {
            check(cancel)?;
        }
        emit(row)?;
        count += 1;
        if let Some(next) = Location::read(&mut readers[i])? {
            heap.push(Reverse((next, i)));
        }
    }
    Ok(())
}

fn write_header(
    out: &mut (impl Write + ?Sized),
    dim: usize,
    count: usize,
    ann_len: u64,
    blocks: usize,
) -> io::Result<()> {
    let narrow =
        |n| u32::try_from(n).map_err(|_| io::Error::other("vector lookup size exceeds u32"));
    for value in [MAGIC, narrow(dim)?, narrow(count)?, VERSION] {
        out.write_all(&value.to_le_bytes())?;
    }
    out.write_all(&ann_len.to_le_bytes())?;
    out.write_all(&narrow(blocks)?.to_le_bytes())?;
    out.write_all(&0u32.to_le_bytes())
}

fn write_block(
    out: &mut (impl Write + ?Sized),
    count: usize,
    doc_base: u32,
    spans: usize,
) -> io::Result<()> {
    for value in [
        u32::try_from(count).map_err(|_| io::Error::other("too many lookup rows"))?,
        doc_base,
        u32::try_from(spans).map_err(|_| io::Error::other("too many lookup spans"))?,
        0,
    ] {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

/// Copy immutable lookup rows. Only the small span/block directories are
/// relocated; no vector or per-vector address is decoded or rewritten.
pub(crate) fn write_copied_locations(
    sources: &[(&crate::segment::vector_data::LazyFlatVectorData, u32, u64)],
    dim: usize,
    count: usize,
    ann_len: u64,
    out: &mut (impl Write + ?Sized),
    cancel: Option<&AtomicBool>,
) -> io::Result<u64> {
    check(cancel)?;
    let actual_count = sources.iter().try_fold(0usize, |total, (flat, _, _)| {
        if flat.dim != dim {
            return Err(io::Error::other("vector lookup dimensions disagree"));
        }
        total
            .checked_add(flat.num_vectors)
            .ok_or_else(|| io::Error::other("vector lookup count overflow"))
    })?;
    if actual_count != count || count == 0 {
        return Err(io::Error::other("vector lookup count mismatch"));
    }
    let layouts: Vec<_> = sources
        .iter()
        .map(|(flat, doc_base, bias)| {
            flat.locations()
                .map(|layout| (flat, *doc_base, *bias, layout))
                .ok_or_else(|| io::Error::other("copy merge requires ANN-backed exact vectors"))
        })
        .collect::<io::Result<_>>()?;
    let blocks: usize = layouts
        .iter()
        .map(|(_, _, _, layout)| layout.blocks.len())
        .sum();
    write_header(out, dim, count, ann_len, blocks)?;
    let mut written = HEADER_SIZE as u64;
    for (flat, _, _, _) in &layouts {
        for chunk in flat.location_rows().chunks(1024 * 1024) {
            check(cancel)?;
            out.write_all(chunk)?;
            written += chunk.len() as u64;
        }
    }
    for (_, _, bias, layout) in &layouts {
        for (i, span) in layout.spans.chunks_exact(SPAN_SIZE).enumerate() {
            if i.is_multiple_of(4096) {
                check(cancel)?;
            }
            let offset = u64_at(span, 0)
                .checked_add(*bias)
                .ok_or_else(|| io::Error::other("vector span relocation overflow"))?;
            out.write_all(&offset.to_le_bytes())?;
            out.write_all(&span[8..])?;
            written += SPAN_SIZE as u64;
        }
    }
    for (_, doc_base, _, layout) in &layouts {
        for block in &layout.blocks {
            check(cancel)?;
            let base = block
                .doc_base
                .checked_add(*doc_base)
                .ok_or_else(|| io::Error::other("vector lookup document base overflow"))?;
            write_block(out, block.count, base, block.spans)?;
            written += BLOCK_SIZE as u64;
        }
    }
    check(cancel)?;
    Ok(written)
}

/// Scratch files are anonymous and owned before their first write. Closing
/// on error, cancellation, or unwind also removes their filesystem storage.
pub(crate) struct ExactLocations {
    rows: Vec<Location>,
    levels: Vec<Vec<File>>,
    spans: Vec<(u64, u32)>,
    row_limit: usize,
    cancellation: Option<std::sync::Arc<AtomicBool>>,
}

impl Default for ExactLocations {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            levels: Vec::new(),
            spans: Vec::new(),
            row_limit: RUN_ROWS,
            cancellation: None,
        }
    }
}

impl ExactLocations {
    pub(crate) fn with_budget(
        budget: usize,
        cancellation: Option<std::sync::Arc<AtomicBool>>,
    ) -> io::Result<Self> {
        if budget < 128 * 1024 {
            return Err(io::Error::other(
                "vector lookup scratch budget must be at least 128 KiB",
            ));
        }
        Ok(Self {
            // Leave room for sixteen input buffers, an output buffer,
            // and the merge heap even at the minimum supported budget.
            row_limit: (budget / 3 / std::mem::size_of::<Location>()).min(RUN_ROWS),
            cancellation,
            ..Self::default()
        })
    }

    pub(crate) fn span(&mut self, offset: u64, count: usize) -> io::Result<u32> {
        if count == 0 {
            return Err(io::Error::other("empty vector span"));
        }
        let id = u32::try_from(self.spans.len())
            .map_err(|_| io::Error::other("too many vector spans"))?;
        let count =
            u32::try_from(count).map_err(|_| io::Error::other("vector span exceeds u32"))?;
        self.spans.push((offset, count));
        Ok(id)
    }

    pub(crate) fn push(&mut self, doc: u32, ordinal: u16, address: u64) -> io::Result<()> {
        if self.rows.len().is_multiple_of(4096) {
            check(self.cancellation.as_deref())?;
        }
        if self.rows.len() == self.rows.capacity() {
            let additional = (self.row_limit - self.rows.len()).min(self.rows.capacity().max(16));
            self.rows.reserve_exact(additional);
        }
        self.rows.push(Location {
            doc,
            ordinal,
            address,
        });
        if self.rows.len() == self.row_limit {
            self.spill()?;
        }
        Ok(())
    }

    fn spill(&mut self) -> io::Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        self.rows.sort_unstable();
        let mut out = BufWriter::with_capacity(IO_BUFFER, tempfile::tempfile()?);
        for &row in &self.rows {
            row.write(&mut out)?;
        }
        self.rows.clear();
        let mut run = out.into_inner().map_err(|error| error.into_error())?;
        for level in 0.. {
            if self.levels.len() == level {
                self.levels.push(Vec::new());
            }
            self.levels[level].push(run);
            if self.levels[level].len() < FAN_IN {
                return Ok(());
            }
            let runs = std::mem::take(&mut self.levels[level]);
            let mut out = BufWriter::with_capacity(IO_BUFFER, tempfile::tempfile()?);
            merge_runs(
                runs,
                |row| row.write(&mut out),
                self.cancellation.as_deref(),
            )?;
            run = out.into_inner().map_err(|error| error.into_error())?;
        }
        unreachable!()
    }

    pub(crate) fn write(
        mut self,
        dim: usize,
        count: usize,
        ann_len: u64,
        out: &mut (impl Write + ?Sized),
        cancel: Option<&AtomicBool>,
    ) -> io::Result<u64> {
        check(cancel)?;
        write_header(out, dim, count, ann_len, 1)?;
        let mut written = 0u64;
        let mut previous = None;
        let mut emit = |row: Location| -> io::Result<()> {
            if written.is_multiple_of(4096) {
                check(cancel)?;
            }
            if previous == Some((row.doc, row.ordinal)) {
                return Ok(());
            }
            // SOAR repeats an exact vector in another leaf. Keep one
            // deterministic location, not a second logical value.
            previous = Some((row.doc, row.ordinal));
            row.write(out)?;
            written += 1;
            Ok(())
        };
        if self.levels.is_empty() {
            self.rows.sort_unstable();
            for row in self.rows {
                emit(row)?;
            }
        } else {
            self.spill()?;
            let mut runs: Vec<_> = self.levels.into_iter().flatten().collect();
            while runs.len() > FAN_IN {
                check(cancel)?;
                let group = runs.split_off(runs.len() - FAN_IN);
                let mut output = BufWriter::with_capacity(IO_BUFFER, tempfile::tempfile()?);
                merge_runs(group, |row| row.write(&mut output), cancel)?;
                runs.push(output.into_inner().map_err(|error| error.into_error())?);
            }
            merge_runs(runs, emit, cancel)?;
        }
        if written != count as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("ANN contains {written} distinct vectors, expected {count}"),
            ));
        }
        for (i, &(offset, count)) in self.spans.iter().enumerate() {
            if i.is_multiple_of(4096) {
                check(cancel)?;
            }
            out.write_all(&offset.to_le_bytes())?;
            out.write_all(&count.to_le_bytes())?;
        }
        write_block(out, count, 0, self.spans.len())?;
        check(cancel)?;
        Ok(HEADER_SIZE as u64
            + written * ENTRY_SIZE as u64
            + self.spans.len() as u64 * SPAN_SIZE as u64
            + BLOCK_SIZE as u64)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn build(limit: usize) -> Vec<u8> {
        let mut locations = ExactLocations {
            row_limit: limit,
            ..Default::default()
        };
        let span = locations.span(56, 1024).unwrap();
        for row in (0..1024u32).rev() {
            locations
                .push(row / 2, 0, (u64::from(span) << 32) | u64::from(row))
                .unwrap();
        }
        let mut bytes = Vec::new();
        locations.write(256, 512, 40_000, &mut bytes, None).unwrap();
        bytes
    }

    #[test]
    fn spilled_lookup_sort_matches_in_memory_and_deduplicates_ordinals() {
        assert_eq!(build(7), build(RUN_ROWS));
    }

    #[test]
    fn lookup_write_propagates_failure_and_cancellation() {
        struct Fails;
        impl Write for Fails {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("injected lookup write failure"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut locations = ExactLocations::default();
        locations.span(56, 1).unwrap();
        locations.push(0, 0, 0).unwrap();
        assert!(
            locations
                .write(256, 1, 100, &mut Fails, None)
                .unwrap_err()
                .to_string()
                .contains("injected")
        );
        let cancelled = std::sync::Arc::new(AtomicBool::new(true));
        let mut locations =
            ExactLocations::with_budget(128 * 1024, Some(cancelled.clone())).unwrap();
        assert_eq!(
            locations.push(0, 0, 0).unwrap_err().kind(),
            io::ErrorKind::Interrupted
        );
        assert_eq!(
            locations
                .write(256, 1, 100, &mut Vec::new(), Some(&cancelled))
                .unwrap_err()
                .kind(),
            io::ErrorKind::Interrupted
        );
    }
}
