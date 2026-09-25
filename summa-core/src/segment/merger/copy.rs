//! Byte-preserving range copies shared by encoded payloads and sparse scratch.

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use super::{OffsetWriter, block_in_place_if_multithread};
use crate::Result;
use crate::directories::DirectoryWriter;

const READ_CHUNK: usize = 4 * 1024 * 1024;

fn check_cancellation(cancellation: Option<&AtomicBool>) -> Result<()> {
    if cancellation.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(crate::Error::IndexClosed);
    }
    Ok(())
}

/// Attempt a byte-identical local range copy through the streaming writer's
/// kernel-assisted path. Returns `Ok(false)` without emitting bytes when the
/// backend/filesystem does not support it.
fn try_copy_local_file_range(
    writer: &mut OffsetWriter,
    source: &std::fs::File,
    source_path: &std::path::Path,
    source_range: std::ops::Range<u64>,
    cancellation: Option<&AtomicBool>,
    context: &str,
) -> Result<bool> {
    const COPY_CHUNK: usize = 16 * 1024 * 1024;

    let expected = source_range
        .end
        .checked_sub(source_range.start)
        .ok_or_else(|| crate::Error::Corruption(format!("{context} source range is inverted")))?;
    let mut source_offset = source_range.start;
    let mut copied = 0u64;
    while copied < expected {
        check_cancellation(cancellation)?;
        let requested = usize::try_from((expected - copied).min(COPY_CHUNK as u64))
            .expect("bounded copy chunk fits usize");
        match writer.copy_from_file_range(source, &mut source_offset, requested) {
            Ok(0) => {
                return Err(crate::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "kernel {context} copy stopped after {copied} of {expected} bytes from \
                         {source_path:?}",
                    ),
                )));
            }
            Ok(count) => {
                copied = copied.checked_add(count as u64).ok_or_else(|| {
                    crate::Error::Internal(format!("{context} copied-byte count exceeds u64"))
                })?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == std::io::ErrorKind::Unsupported && copied == 0 => {
                return Ok(false);
            }
            Err(error) => return Err(crate::Error::Io(error)),
        }
    }

    static LOGGED: AtomicBool = AtomicBool::new(false);
    if copied != 0 && !LOGGED.swap(true, Ordering::Relaxed) {
        log::info!("[merge] byte-identical local ranges use kernel-assisted file copies");
    }
    Ok(true)
}

/// Keep one source descriptor through kernel admission and local fallback.
fn copy_local_file_range(
    writer: &mut OffsetWriter,
    source_path: &std::path::Path,
    source_range: std::ops::Range<u64>,
    cancellation: Option<&AtomicBool>,
    context: &str,
) -> Result<()> {
    let mut source = std::fs::File::open(source_path)?;
    if try_copy_local_file_range(
        writer,
        &source,
        source_path,
        source_range.clone(),
        cancellation,
        context,
    )? {
        return Ok(());
    }
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        log::info!("[merge] unsupported local range copies use bounded file reads");
    }
    // Unsupported is accepted only before any kernel copy succeeded. Seek
    // the original range explicitly rather than relying on the fd cursor.
    copy_reader_range(writer, &mut source, source_range, cancellation)
}

fn copy_reader_range(
    writer: &mut OffsetWriter,
    source: &mut (impl Read + Seek),
    source_range: std::ops::Range<u64>,
    cancellation: Option<&AtomicBool>,
) -> Result<()> {
    let mut remaining = source_range
        .end
        .checked_sub(source_range.start)
        .ok_or_else(|| crate::Error::Corruption("local copy source range is inverted".into()))?;
    if remaining == 0 {
        return Ok(());
    }
    check_cancellation(cancellation)?;
    source.seek(SeekFrom::Start(source_range.start))?;
    let mut buffer = vec![0; remaining.min(READ_CHUNK as u64) as usize];
    while remaining > 0 {
        check_cancellation(cancellation)?;
        let count = remaining.min(buffer.len() as u64) as usize;
        // read_exact retries Interrupted and refuses a truncated source. A
        // failed chunk is never partly appended to the destination.
        source.read_exact(&mut buffer[..count])?;
        check_cancellation(cancellation)?;
        writer.write_all(&buffer[..count])?;
        remaining -= count as u64;
    }
    Ok(())
}

/// Copy an admitted byte-identical range. Local sources use the kernel or
/// bounded reads; abstract directories retain their mapped-byte fallback.
pub(in crate::segment) fn copy_local_range_or_bytes(
    writer: &mut OffsetWriter,
    source_path: Option<&std::path::Path>,
    source_range: std::ops::Range<u64>,
    bytes: &[u8],
    cancellation: Option<&AtomicBool>,
    context: &str,
) -> Result<()> {
    let range_len = source_range
        .end
        .checked_sub(source_range.start)
        .ok_or_else(|| crate::Error::Corruption(format!("{context} source range is inverted")))?;
    if range_len != bytes.len() as u64 {
        return Err(crate::Error::Corruption(format!(
            "{context} source range is {range_len} bytes but mapped section is {} bytes",
            bytes.len(),
        )));
    }
    if bytes.is_empty() {
        return Ok(());
    }

    if let Some(path) = source_path {
        return copy_local_file_range(writer, path, source_range, cancellation, context);
    }

    for chunk in bytes.chunks(READ_CHUNK) {
        check_cancellation(cancellation)?;
        writer.write_all(chunk).map_err(crate::Error::Io)?;
    }
    Ok(())
}

/// Append an exact-length temporary directory file to a segment output and
/// remove it. Used by sparse skip tables so neither merge nor BP rewrite
/// buffers a corpus-sized metadata section on heap.
pub(crate) async fn append_and_delete_temp<D: DirectoryWriter>(
    directory: &D,
    path: &std::path::Path,
    expected_bytes: u64,
    writer: &mut OffsetWriter,
    index_label: &str,
) -> Result<()> {
    let actual_bytes = directory.file_size(path).await?;
    if actual_bytes != expected_bytes {
        return Err(crate::Error::Corruption(format!(
            "temporary sparse section {:?} has {} bytes, expected {}",
            path, actual_bytes, expected_bytes,
        )));
    }
    if let Some(local_path) = directory.local_path(path) {
        block_in_place_if_multithread(|| {
            copy_local_file_range(
                writer,
                &local_path,
                0..expected_bytes,
                None,
                "sparse scratch",
            )
        })?;
    } else {
        let mut offset = 0u64;
        while offset < expected_bytes {
            let end = (offset + READ_CHUNK as u64).min(expected_bytes);
            let chunk = directory.read_range(path, offset..end).await?;
            writer
                .write_all(chunk.as_slice())
                .map_err(crate::Error::Io)?;
            offset = end;
        }
    }
    if let Err(error) = directory.delete(path).await {
        // The section is already complete in the output. This output-scoped
        // scratch file is safe for the startup orphan sweep and must not
        // invalidate an otherwise successful multi-hour merge.
        log::warn!(
            "[merge] index={} failed to remove temporary sparse section {:?}: {}",
            index_label,
            path,
            error,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directories::StreamingWriter;
    use std::collections::VecDeque;
    use std::io::{self, Cursor};
    use std::sync::{Arc, Mutex};

    enum KernelStep {
        Copy(usize),
        Error(io::ErrorKind),
    }

    #[derive(Default)]
    struct Recorded {
        bytes: Vec<u8>,
        kernel_requests: Vec<(u64, usize)>,
        write_sizes: Vec<usize>,
    }

    struct TestWriter {
        recorded: Arc<Mutex<Recorded>>,
        steps: VecDeque<KernelStep>,
        cancel_after_write: Option<Arc<AtomicBool>>,
        write_limit: usize,
        interrupt_write: bool,
    }

    impl Write for TestWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if std::mem::take(&mut self.interrupt_write) {
                return Err(io::Error::from(io::ErrorKind::Interrupted));
            }
            let bytes = &bytes[..bytes.len().min(self.write_limit)];
            let mut recorded = self.recorded.lock().unwrap();
            recorded.bytes.extend_from_slice(bytes);
            recorded.write_sizes.push(bytes.len());
            if let Some(cancelled) = &self.cancel_after_write {
                cancelled.store(true, Ordering::Release);
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl StreamingWriter for TestWriter {
        fn finish(self: Box<Self>) -> io::Result<()> {
            Ok(())
        }
        fn bytes_written(&self) -> u64 {
            self.recorded.lock().unwrap().bytes.len() as u64
        }
        fn copy_from_file_range(
            &mut self,
            source: &std::fs::File,
            offset: &mut u64,
            len: usize,
        ) -> io::Result<usize> {
            self.recorded
                .lock()
                .unwrap()
                .kernel_requests
                .push((*offset, len));
            match self
                .steps
                .pop_front()
                .unwrap_or(KernelStep::Error(io::ErrorKind::Unsupported))
            {
                KernelStep::Copy(count) => {
                    let count = count.min(len);
                    let mut source = source.try_clone()?;
                    source.seek(SeekFrom::Start(*offset))?;
                    let mut bytes = vec![0; count];
                    source.read_exact(&mut bytes)?;
                    self.recorded
                        .lock()
                        .unwrap()
                        .bytes
                        .extend_from_slice(&bytes);
                    *offset += count as u64;
                    Ok(count)
                }
                KernelStep::Error(kind) => Err(io::Error::from(kind)),
            }
        }
    }

    fn writer(steps: impl IntoIterator<Item = KernelStep>) -> (OffsetWriter, Arc<Mutex<Recorded>>) {
        let recorded = Arc::new(Mutex::new(Recorded::default()));
        (
            OffsetWriter::new(Box::new(TestWriter {
                recorded: Arc::clone(&recorded),
                steps: steps.into_iter().collect(),
                cancel_after_write: None,
                write_limit: usize::MAX,
                interrupt_write: false,
            })),
            recorded,
        )
    }

    #[test]
    fn truncated_local_range_fails_instead_of_copying_a_stale_mapped_snapshot() {
        let source = tempfile::NamedTempFile::new().unwrap();
        let admitted = (0..100u8).collect::<Vec<_>>();
        std::fs::write(source.path(), &admitted).unwrap();
        // An externally truncated immutable source must fail copying; the
        // previously admitted view does not justify publishing missing bytes.
        source.as_file().set_len(20).unwrap();
        let (mut output, recorded) = writer([]);
        let error = copy_local_range_or_bytes(
            &mut output,
            Some(source.path()),
            10..30,
            &admitted[10..30],
            None,
            "test",
        )
        .unwrap_err();
        assert!(
            matches!(error, crate::Error::Io(error) if error.kind() == io::ErrorKind::UnexpectedEof)
        );
        assert!(recorded.lock().unwrap().bytes.is_empty());
    }
    #[test]
    fn local_fallback_copies_an_exact_subrange_in_bounded_chunks() {
        let source = tempfile::NamedTempFile::new().unwrap();
        let bytes: Vec<_> = (0..READ_CHUNK * 2 + 83).map(|i| (i % 251) as u8).collect();
        std::fs::write(source.path(), &bytes).unwrap();
        let (mut output, recorded) = writer([]);
        output.write_all(b"prefix").unwrap();
        let range = 17..(bytes.len() - 11);
        copy_local_range_or_bytes(
            &mut output,
            Some(source.path()),
            range.start as u64..range.end as u64,
            &bytes[range.clone()],
            None,
            "test",
        )
        .unwrap();
        let recorded = recorded.lock().unwrap();
        assert_eq!(&recorded.bytes[..6], b"prefix");
        assert_eq!(&recorded.bytes[6..], &bytes[range.clone()]);
        assert_eq!(output.offset(), (range.len() + 6) as u64);
        assert_eq!(&recorded.write_sizes[1..], &[READ_CHUNK, READ_CHUNK, 55]);
        assert_eq!(recorded.kernel_requests, [(17, range.len())]);
    }

    #[test]
    fn kernel_short_copies_and_interruptions_preserve_source_offsets() {
        let source = tempfile::NamedTempFile::new().unwrap();
        let bytes = (0..40u8).collect::<Vec<_>>();
        std::fs::write(source.path(), &bytes).unwrap();
        let (mut output, recorded) = writer([
            KernelStep::Error(io::ErrorKind::Interrupted),
            KernelStep::Copy(3),
            KernelStep::Error(io::ErrorKind::Interrupted),
            KernelStep::Copy(7),
        ]);
        copy_local_range_or_bytes(
            &mut output,
            Some(source.path()),
            9..19,
            &bytes[9..19],
            None,
            "test",
        )
        .unwrap();
        let recorded = recorded.lock().unwrap();
        assert_eq!(recorded.bytes, bytes[9..19]);
        assert_eq!(
            recorded.kernel_requests,
            [(9, 10), (9, 10), (12, 7), (12, 7)]
        );
        assert!(
            recorded.write_sizes.is_empty(),
            "successful kernel copies must not stage or duplicate bytes"
        );
        assert_eq!(output.offset(), 10);
    }

    #[test]
    fn kernel_copy_failure_after_partial_output_never_restarts_fallback() {
        let source = tempfile::NamedTempFile::new().unwrap();
        let bytes = (0..40u8).collect::<Vec<_>>();
        std::fs::write(source.path(), &bytes).unwrap();
        for kind in [io::ErrorKind::Unsupported, io::ErrorKind::Other] {
            let (mut output, recorded) = writer([KernelStep::Copy(3), KernelStep::Error(kind)]);
            let error = copy_local_range_or_bytes(
                &mut output,
                Some(source.path()),
                9..19,
                &bytes[9..19],
                None,
                "test",
            )
            .unwrap_err();
            assert!(matches!(error, crate::Error::Io(error) if error.kind() == kind));
            let recorded = recorded.lock().unwrap();
            assert_eq!(recorded.bytes, bytes[9..12]);
            assert_eq!(recorded.kernel_requests, [(9, 10), (12, 7)]);
            assert!(recorded.write_sizes.is_empty());
            assert_eq!(output.offset(), 3);
        }
        let (mut output, recorded) = writer([KernelStep::Copy(0)]);
        let error = copy_local_range_or_bytes(
            &mut output,
            Some(source.path()),
            9..19,
            &bytes[9..19],
            None,
            "test",
        )
        .unwrap_err();
        assert!(
            matches!(error, crate::Error::Io(error) if error.kind() == io::ErrorKind::UnexpectedEof)
        );
        assert!(recorded.lock().unwrap().bytes.is_empty());
    }

    struct InterruptedReader {
        cursor: Cursor<Vec<u8>>,
        interrupted: bool,
        largest_request: usize,
    }
    impl Read for InterruptedReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.largest_request = self.largest_request.max(buffer.len());
            if !self.interrupted {
                self.interrupted = true;
                return Err(io::Error::from(io::ErrorKind::Interrupted));
            }
            let len = buffer.len().min(1024);
            self.cursor.read(&mut buffer[..len])
        }
    }
    impl Seek for InterruptedReader {
        fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
            self.cursor.seek(from)
        }
    }

    #[test]
    fn staged_reads_retry_interruptions_and_short_reads_without_exceeding_scratch_bound() {
        let bytes: Vec<_> = (0..READ_CHUNK + 83).map(|i| (i % 251) as u8).collect();
        let mut source = InterruptedReader {
            cursor: Cursor::new(bytes.clone()),
            interrupted: false,
            largest_request: 0,
        };
        let (mut output, recorded) = writer([]);
        copy_reader_range(
            &mut output,
            &mut source,
            17..(bytes.len() - 11) as u64,
            None,
        )
        .unwrap();
        assert!(source.interrupted);
        assert_eq!(source.largest_request, READ_CHUNK);
        assert_eq!(source.cursor.position(), (bytes.len() - 11) as u64);
        let recorded = recorded.lock().unwrap();
        assert_eq!(recorded.bytes, bytes[17..bytes.len() - 11]);
        assert_eq!(recorded.write_sizes, [READ_CHUNK, 55]);
    }

    #[test]
    fn staged_copy_retries_interrupted_and_short_destination_writes() {
        let recorded = Arc::new(Mutex::new(Recorded::default()));
        let mut output = OffsetWriter::new(Box::new(TestWriter {
            recorded: Arc::clone(&recorded),
            steps: VecDeque::new(),
            cancel_after_write: None,
            write_limit: 3,
            interrupt_write: true,
        }));
        let mut source = Cursor::new(b"0123456789abc".to_vec());
        copy_reader_range(&mut output, &mut source, 1..12, None).unwrap();
        let recorded = recorded.lock().unwrap();
        assert_eq!(recorded.bytes, b"123456789ab");
        assert_eq!(recorded.write_sizes, [3, 3, 3, 2]);
        assert_eq!(output.offset(), 11);
    }

    #[test]
    fn cancellation_stops_local_staging_before_the_next_chunk() {
        let source = tempfile::NamedTempFile::new().unwrap();
        let bytes = vec![7; READ_CHUNK + 1];
        std::fs::write(source.path(), &bytes).unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let recorded = Arc::new(Mutex::new(Recorded::default()));
        let mut output = OffsetWriter::new(Box::new(TestWriter {
            recorded: Arc::clone(&recorded),
            steps: VecDeque::new(),
            cancel_after_write: Some(Arc::clone(&cancelled)),
            write_limit: usize::MAX,
            interrupt_write: false,
        }));
        let error = copy_local_range_or_bytes(
            &mut output,
            Some(source.path()),
            0..bytes.len() as u64,
            &bytes,
            Some(&cancelled),
            "test",
        )
        .unwrap_err();
        assert!(matches!(error, crate::Error::IndexClosed));
        assert_eq!(recorded.lock().unwrap().write_sizes, [READ_CHUNK]);
        assert_eq!(output.offset(), READ_CHUNK as u64);
        let (mut output, recorded) = writer([]);
        let error = copy_local_range_or_bytes(
            &mut output,
            Some(source.path()),
            0..1,
            &bytes[..1],
            Some(&cancelled),
            "test",
        )
        .unwrap_err();
        assert!(matches!(error, crate::Error::IndexClosed));
        assert!(recorded.lock().unwrap().kernel_requests.is_empty());
    }

    #[test]
    fn abstract_and_empty_ranges_preserve_the_existing_mapped_fallback() {
        let (mut output, recorded) = writer([]);
        copy_local_range_or_bytes(&mut output, None, 27..30, b"abc", None, "test").unwrap();
        let directory = tempfile::tempdir().unwrap();
        copy_local_range_or_bytes(
            &mut output,
            Some(&directory.path().join("missing")),
            7..7,
            b"",
            None,
            "test",
        )
        .unwrap();
        assert_eq!(recorded.lock().unwrap().bytes, b"abc");
        assert!(recorded.lock().unwrap().kernel_requests.is_empty());
        assert!(matches!(
            copy_local_range_or_bytes(&mut output, None, 0..4, b"abc", None, "test"),
            Err(crate::Error::Corruption(_))
        ));
    }

    #[tokio::test]
    async fn temporary_sections_share_the_local_fallback_and_delete_only_after_copy() {
        use crate::directories::{MmapDirectory, RamDirectory};
        async fn check<D: DirectoryWriter>(dir: D) {
            let path = std::path::Path::new("section.tmp");
            dir.write(path, b"temporary section").await.unwrap();
            let (mut output, recorded) = writer([]);
            assert!(
                append_and_delete_temp(&dir, path, 1, &mut output, "test")
                    .await
                    .is_err()
            );
            assert!(dir.exists(path).await.unwrap());
            assert!(recorded.lock().unwrap().bytes.is_empty());
            append_and_delete_temp(&dir, path, 17, &mut output, "test")
                .await
                .unwrap();
            assert_eq!(recorded.lock().unwrap().bytes, b"temporary section");
            assert!(!dir.exists(path).await.unwrap());
        }
        check(RamDirectory::new()).await;
        let temp = tempfile::tempdir().unwrap();
        check(MmapDirectory::new(temp.path())).await;
    }
}
