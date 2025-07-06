//! Cold write path for bulk one-shot IO (merges, reorders).
//!
//! See `docs/cold-io.md`. Writing a multi-GB merged segment through the page
//! cache evicts the serving segments' warm pages; this writer drops its own
//! pages behind the write cursor so the steady-state cache footprint of a
//! merge is one 64 MB window instead of the whole output. This is the
//! default (and only) write path for merge/reorder outputs on filesystem
//! directories — output is byte-identical to the buffered writer.
//!
//! Mechanism (O_DIRECT-equivalent without the alignment tax):
//! - Linux: per 64 MB window `sync_file_range(WAIT_BEFORE|WRITE|WAIT_AFTER)`
//!   then `posix_fadvise(POSIX_FADV_DONTNEED)` on the clean window.
//! - macOS: `fcntl(fd, F_NOCACHE, 1)` at creation.
//! - Elsewhere: plain buffered writes (logged once, loudly).

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use super::directory::StreamingWriter;

/// Log the effective mode once at first cold-writer creation.
fn log_mode_once() {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        #[cfg(target_os = "linux")]
        log::info!(
            "[cold_io] merge writes drop page cache via sync_file_range + fadvise(DONTNEED)"
        );
        #[cfg(target_os = "macos")]
        log::info!("[cold_io] merge writes bypass buffer cache via F_NOCACHE");
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        log::warn!(
            "[cold_io] no page-cache-drop mechanism on this platform — merge writes stay buffered"
        );
    }
}

/// Buffer size (matches `FileStreamingWriter`).
const BUF_SIZE: usize = 8 * 1024 * 1024;
/// Drop window: force writeback + drop cache for completed 64 MB regions.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const DROP_WINDOW: u64 = 64 * 1024 * 1024;

/// StreamingWriter that keeps at most one drop-window of its output in the
/// page cache. Byte-for-byte identical output to `FileStreamingWriter`.
pub(crate) struct ColdStreamingWriter {
    file: std::fs::File,
    buf: Vec<u8>,
    /// Bytes flushed to the fd.
    written: u64,
    /// Start of the region not yet dropped from cache (linux windowed drop).
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    drop_cursor: u64,
}

impl ColdStreamingWriter {
    pub(crate) fn new(file: std::fs::File) -> Self {
        log_mode_once();
        #[cfg(target_os = "macos")]
        {
            use std::os::fd::AsRawFd;
            // Bypass the buffer cache for this fd; writes remain unaligned-safe.
            if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } != 0 {
                log::warn!(
                    "[cold_io] F_NOCACHE failed ({}); falling back to cached writes",
                    io::Error::last_os_error()
                );
            }
        }
        Self {
            file,
            buf: Vec::with_capacity(BUF_SIZE),
            written: 0,
            drop_cursor: 0,
        }
    }

    fn flush_buf(&mut self) -> io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        self.file.write_all(&self.buf)?;
        self.written += self.buf.len() as u64;
        self.buf.clear();
        self.maybe_drop(false);
        Ok(())
    }

    /// Force writeback and drop clean pages of completed windows.
    /// `final_drop` drops everything written so far (called from finish()).
    #[allow(unused_variables)]
    fn maybe_drop(&mut self, final_drop: bool) {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let end = if final_drop {
                self.written
            } else {
                // Only drop fully-completed windows.
                (self.written / DROP_WINDOW) * DROP_WINDOW
            };
            if end <= self.drop_cursor {
                return;
            }
            let fd = self.file.as_raw_fd();
            let off = self.drop_cursor as libc::off64_t;
            let len = (end - self.drop_cursor) as libc::off64_t;
            unsafe {
                // Write the window to storage so DONTNEED can evict it.
                libc::sync_file_range(
                    fd,
                    off,
                    len,
                    libc::SYNC_FILE_RANGE_WAIT_BEFORE
                        | libc::SYNC_FILE_RANGE_WRITE
                        | libc::SYNC_FILE_RANGE_WAIT_AFTER,
                );
                libc::posix_fadvise(fd, off, len, libc::POSIX_FADV_DONTNEED);
            }
            self.drop_cursor = end;
        }
        // macOS: F_NOCACHE handles it per-write; other platforms: no-op.
    }
}

impl io::Write for ColdStreamingWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if self.buf.len() + data.len() > BUF_SIZE {
            self.flush_buf()?;
        }
        if data.len() >= BUF_SIZE {
            // Large write: send straight to the fd.
            self.file.write_all(data)?;
            self.written += data.len() as u64;
            self.maybe_drop(false);
        } else {
            self.buf.extend_from_slice(data);
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buf()?;
        self.file.flush()
    }
}

impl StreamingWriter for ColdStreamingWriter {
    fn finish(mut self: Box<Self>) -> io::Result<()> {
        self.flush_buf()?;
        self.file.sync_all()?;
        self.maybe_drop(true);
        crate::observe::cold_write(self.written as usize);
        log::debug!(
            "[cold_io] wrote {:.1} MB with page cache dropped behind the cursor",
            self.written as f64 / (1024.0 * 1024.0)
        );
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        self.written + self.buf.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cold writer output must be byte-identical to a plain buffered writer
    /// across small writes, buffer-boundary writes, and >buffer bulk writes.
    #[test]
    fn test_cold_streaming_writer_writes_byte_identical_output() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cold.bin");
        let file = std::fs::File::create(&path).unwrap();
        let mut writer: Box<dyn StreamingWriter> = Box::new(ColdStreamingWriter::new(file));

        let mut expected: Vec<u8> = Vec::new();
        // Small writes
        for i in 0..1000u32 {
            let chunk = i.to_le_bytes();
            writer.write_all(&chunk).unwrap();
            expected.extend_from_slice(&chunk);
        }
        // Exactly buffer-sized write
        let big = vec![0xABu8; BUF_SIZE];
        writer.write_all(&big).unwrap();
        expected.extend_from_slice(&big);
        // Larger-than-buffer write
        let bigger = vec![0xCDu8; BUF_SIZE + 12345];
        writer.write_all(&bigger).unwrap();
        expected.extend_from_slice(&bigger);
        // Tail
        writer.write_all(b"tail").unwrap();
        expected.extend_from_slice(b"tail");

        assert_eq!(writer.bytes_written(), expected.len() as u64);
        writer.finish().unwrap();

        let got = std::fs::read(&path).unwrap();
        assert_eq!(got.len(), expected.len());
        assert_eq!(got, expected, "cold writer corrupted the byte stream");
    }
}
