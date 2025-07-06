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
//! - Linux: two-window write-behind — initiate async writeback
//!   (`sync_file_range(WRITE)`) for each completed 64 MB window and
//!   `posix_fadvise(DONTNEED)` the *previous* window (whose writeback was
//!   initiated a window ago and is clean by now). No synchronous disk wait
//!   on the writing thread; `finish()` fsyncs and drops everything.
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
    /// Start of the region whose writeback has not been initiated yet.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    writeback_cursor: u64,
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
            writeback_cursor: 0,
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

    /// Two-window write-behind: initiate async writeback for newly completed
    /// windows and drop the previous (now clean) window's pages. No
    /// synchronous disk wait on the writing thread — this runs on a tokio
    /// worker during merges. `final_drop` (from finish(), after fsync) drops
    /// everything.
    #[allow(unused_variables)]
    fn maybe_drop(&mut self, final_drop: bool) {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let fd = self.file.as_raw_fd();
            if final_drop {
                // Caller fsync'd — all pages are clean; drop the whole file.
                if self.written > self.drop_cursor {
                    unsafe {
                        libc::posix_fadvise(
                            fd,
                            self.drop_cursor as libc::off64_t,
                            (self.written - self.drop_cursor) as libc::off64_t,
                            libc::POSIX_FADV_DONTNEED,
                        );
                    }
                    self.drop_cursor = self.written;
                }
                return;
            }
            // Only fully-completed windows.
            let completed = (self.written / DROP_WINDOW) * DROP_WINDOW;
            if completed <= self.writeback_cursor {
                return;
            }
            unsafe {
                // Kick off async writeback for the newly completed window(s).
                libc::sync_file_range(
                    fd,
                    self.writeback_cursor as libc::off64_t,
                    (completed - self.writeback_cursor) as libc::off64_t,
                    libc::SYNC_FILE_RANGE_WRITE,
                );
                // Drop the previous window — its writeback was initiated a
                // full window of writing ago, so its pages are clean by now.
                // (DONTNEED on a still-dirty page is a no-op; it gets another
                // chance on the next call and at final_drop.)
                if self.writeback_cursor > self.drop_cursor {
                    libc::posix_fadvise(
                        fd,
                        self.drop_cursor as libc::off64_t,
                        (self.writeback_cursor - self.drop_cursor) as libc::off64_t,
                        libc::POSIX_FADV_DONTNEED,
                    );
                    self.drop_cursor = self.writeback_cursor;
                }
            }
            self.writeback_cursor = completed;
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
