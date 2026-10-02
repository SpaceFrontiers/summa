//! Bounded, shared local payload I/O. Metadata readers stay mapped.
pub(super) mod buffers;
mod pool;
#[cfg(all(target_os = "linux", feature = "io-uring"))]
mod ring;
pub(super) mod scope;
mod stats;
#[cfg(test)]
mod tests;

use super::{FileHandle, OwnedBytes, RangeReadFn};
use parking_lot::Mutex;
use stats::Counters;
pub use stats::PayloadReadStats;
use std::{fs::File, io, ops::Range, path::PathBuf, sync::Arc, thread::JoinHandle};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

const DEPTH: usize = FileHandle::MAX_BATCH_CONCURRENCY;
const BYTES: usize = FileHandle::MAX_BATCH_BYTES;
const MAX_RETURN_BYTES: usize = 64 * 1024 * 1024;

/// Explicit backend selection. Unavailable backends fail at construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PayloadReadBackend {
    /// Eight persistent positional-read workers, available on native platforms.
    Pool,
    /// One completion-driven Linux worker. Requires `io-uring` and Linux 6.0+.
    IoUring,
}

struct Reply {
    result: io::Result<OwnedBytes>,
    // A slow or departed receiver must not create an uncharged reply queue.
    _bytes: OwnedSemaphorePermit,
}
struct Job {
    buffers: Arc<buffers::BufferPool>,
    file: Arc<File>,
    range: Range<u64>,
    _slot: OwnedSemaphorePermit,
    bytes: OwnedSemaphorePermit,
    reply: oneshot::Sender<Reply>,
    _lease: Option<scope::ReadLease>,
}
impl Job {
    fn len(&self) -> usize {
        (self.range.end - self.range.start) as usize
    }
}
struct Read {
    job: Job,
    buffer: Vec<u8>,
}

struct Admission {
    slots: Arc<Semaphore>,
    bytes: Arc<Semaphore>,
}
impl Admission {
    fn close(&self) {
        self.slots.close();
        self.bytes.close();
    }
}

/// A process-shared payload service, attached explicitly to mapped directories.
///
/// Across all attached files, at most eight jobs and 8 MiB of read buffers are
/// admitted. A logical range may assemble up to 64 MiB; returned bytes belong to
/// the caller's cache/response budget. Dropping a future never frees a submitted
/// buffer. Directory unlink drains its accepted reads; retire directories before
/// removing their roots. Call [`Self::shutdown`] before stopping the runtime;
/// dropping the last owner also joins the workers, synchronously.
pub struct PayloadReadService {
    sender: async_channel::Sender<Job>,
    admission: Arc<Admission>,
    workers: Mutex<Vec<JoinHandle<io::Result<()>>>>,
    counters: Arc<Counters>,
    backend: PayloadReadBackend,
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    wake: Option<Arc<ring::Wake>>,
    scopes: scope::Registry,
    buffers: Arc<buffers::BufferPool>,
}

fn closed() -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, "payload service is closed")
}

fn prepare(job: Job, counters: &Counters) -> Option<Read> {
    if job.reply.is_closed() {
        counters.skipped();
        return None;
    }
    let len = job.len();
    let buffer = match job.buffers.take(job.bytes.num_permits()) {
        Ok(buffer) => buffer,
        Err(error) => {
            let _ = job.reply.send(Reply {
                result: Err(error),
                _bytes: job.bytes,
            });
            return None;
        }
    };
    counters.started(len);
    Some(Read { job, buffer })
}

fn finish(read: Read, result: io::Result<()>, counters: &Counters) {
    let len = read.job.len();
    counters.finished(result.is_err());
    drop(read.job._slot);
    if read
        .job
        .reply
        .send(Reply {
            result: result.map(|()| read.job.buffers.own(read.buffer, len)),
            _bytes: read.job.bytes,
        })
        .is_err()
    {
        counters.dropped();
    }
}

fn spawn_worker(
    name: &str,
    receiver: async_channel::Receiver<Job>,
    counters: Arc<Counters>,
    admission: Arc<Admission>,
    work: impl FnOnce() -> io::Result<()> + Send + 'static,
) -> io::Result<JoinHandle<io::Result<()>>> {
    std::thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
                .unwrap_or_else(|_| Err(io::Error::other("payload worker panicked")));
            if let Err(error) = &result {
                receiver.close();
                admission.close();
                // A closed channel still owns queued jobs while the service's
                // sender lives. Release their leases/permits before joining;
                // otherwise directory retirement can wait on a dead worker.
                while let Ok(job) = receiver.try_recv() {
                    counters.skipped();
                    drop(job);
                }
                counters.failed();
                log::error!("payload worker failed: {error}");
            }
            result
        })
}

impl PayloadReadService {
    /// Create once and share the returned owner between directories/indices.
    pub fn new(backend: PayloadReadBackend) -> io::Result<Arc<Self>> {
        Self::with_buffer_budget(backend, 0)
    }

    /// Select bounded idle buffer reuse (0 disables it, maximum 8 MiB).
    /// This residency budget is additional to admitted and caller-retained bytes.
    pub fn with_buffer_budget(
        backend: PayloadReadBackend,
        idle_bytes: usize,
    ) -> io::Result<Arc<Self>> {
        if idle_bytes > BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "idle payload buffer budget exceeds 8 MiB",
            ));
        }
        let (sender, receiver) = async_channel::bounded(DEPTH);
        let counters = Arc::new(Counters::default());
        let admission = Arc::new(Admission {
            slots: Arc::new(Semaphore::new(DEPTH)),
            bytes: Arc::new(Semaphore::new(BYTES)),
        });
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        let wake = if backend == PayloadReadBackend::IoUring {
            Some(Arc::new(ring::Wake::new(counters.clone())?))
        } else {
            None
        };
        let workers = match backend {
            PayloadReadBackend::Pool => pool::start(receiver, counters.clone(), admission.clone())?,
            PayloadReadBackend::IoUring => {
                #[cfg(all(target_os = "linux", feature = "io-uring"))]
                {
                    ring::start(
                        receiver,
                        counters.clone(),
                        admission.clone(),
                        wake.as_ref().unwrap().clone(),
                    )?
                }
                #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
                {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "io_uring requires Linux and the summa-core/io-uring feature",
                    ));
                }
            }
        };
        log::info!("payload I/O backend {backend:?}: {DEPTH} reads, {BYTES} in-flight bytes");
        Ok(Arc::new(Self {
            sender,
            admission,
            workers: Mutex::new(workers),
            counters,
            backend,
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            wake,
            scopes: scope::Registry::default(),
            buffers: buffers::BufferPool::new(idle_bytes),
        }))
    }

    pub(super) fn directory_scope(&self, root: &std::path::Path) -> Arc<scope::Scope> {
        self.scopes.attach(root)
    }

    pub(super) fn ensure_drained_resources(&self) -> io::Result<()> {
        if self.counters.snapshot().quarantined_bytes != 0 {
            Err(io::Error::other(
                "payload cancellation failed; files remain in use",
            ))
        } else {
            Ok(())
        }
    }

    /// Selected backend, without automatic fallback.
    pub fn backend(&self) -> PayloadReadBackend {
        self.backend
    }

    /// Approximate cumulative counters; snapshot fields are not transactional.
    pub fn stats(&self) -> PayloadReadStats {
        let mut stats = self.counters.snapshot();
        let (bytes, hits, misses) = self.buffers.snapshot();
        stats.idle_buffer_bytes = bytes;
        stats.buffer_reuses = hits;
        stats.buffer_allocations = misses;
        stats
    }

    /// Stop admission and join every worker. Cancelling this future after it is
    /// first polled does not detach ownership: the blocking join continues.
    pub async fn shutdown(self: &Arc<Self>) -> io::Result<()> {
        self.sender.close();
        self.admission.close();
        let service = self.clone();
        tokio::task::spawn_blocking(move || service.join())
            .await
            .map_err(io::Error::other)?
    }

    fn join(&self) -> io::Result<()> {
        // Hold the lock through joining so concurrent shutdowns observe the drain.
        let mut workers = self.workers.lock();
        for worker in workers.drain(..) {
            if worker.join().is_err() {
                self.counters.failed();
            }
        }
        self.buffers.close();
        self.ensure_drained_resources()?;
        if self.counters.snapshot().worker_failures != 0 {
            Err(io::Error::other(
                "payload worker failed; inspect payload I/O error logs",
            ))
        } else {
            Ok(())
        }
    }

    pub(super) async fn open(
        self: &Arc<Self>,
        path: PathBuf,
        label: Arc<str>,
        scope: Option<Arc<scope::Scope>>,
    ) -> io::Result<FileHandle> {
        if self.sender.is_closed() {
            return Err(closed());
        }
        let lease = match &scope {
            Some(scope) => Some(scope.read().await?),
            None => None,
        };
        let (file, len) = tokio::task::spawn_blocking(move || {
            let _lease = lease;
            let file = File::open(path)?;
            let metadata = file.metadata()?;
            if !metadata.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "payload must be a regular file",
                ));
            }
            Ok((Arc::new(file), metadata.len()))
        })
        .await
        .map_err(io::Error::other)??;
        let service = self.clone();
        let read: RangeReadFn = Arc::new(move |range| {
            let file = file.clone();
            let service = service.clone();
            let scope = scope.clone();
            Box::pin(async move { service.read_range(file, range, scope).await })
        });
        Ok(FileHandle::lazy_labeled(len, read, label))
    }

    async fn read_range(
        &self,
        file: Arc<File>,
        range: Range<u64>,
        scope: Option<Arc<scope::Scope>>,
    ) -> io::Result<OwnedBytes> {
        let len = range
            .end
            .checked_sub(range.start)
            .filter(|&n| n <= MAX_RETURN_BYTES as u64)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "payload range exceeds 64 MiB")
            })? as usize;
        if len <= BYTES {
            return self.read(file, range, scope).await;
        }
        let mut output = Vec::new();
        output.try_reserve_exact(len).map_err(io::Error::other)?;
        let mut start = range.start;
        while start < range.end {
            let end = start + (range.end - start).min(BYTES as u64);
            let chunk = self.read(file.clone(), start..end, scope.clone()).await?;
            output.extend_from_slice(chunk.as_slice());
            start = end;
        }
        Ok(OwnedBytes::new(output))
    }

    async fn read(
        &self,
        file: Arc<File>,
        range: Range<u64>,
        scope: Option<Arc<scope::Scope>>,
    ) -> io::Result<OwnedBytes> {
        let len = range
            .end
            .checked_sub(range.start)
            .filter(|&n| n <= BYTES as u64)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "payload job exceeds byte budget",
                )
            })? as usize;
        if len == 0 {
            return Ok(OwnedBytes::empty());
        }
        let lease = match scope {
            Some(scope) => Some(scope.read().await?),
            None => None,
        };
        let bytes = self
            .admission
            .bytes
            .clone()
            .acquire_many_owned(self.buffers.allocation_len(len) as u32)
            .await
            .map_err(|_| closed())?;
        let slot = self
            .admission
            .slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| closed())?;
        let (reply, done) = oneshot::channel();
        self.sender
            .send(Job {
                _lease: lease,
                buffers: self.buffers.clone(),
                file,
                range,
                _slot: slot,
                bytes,
                reply,
            })
            .await
            .map_err(|_| closed())?;
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        if let Some(wake) = &self.wake
            && let Err(error) = wake.notify()
        {
            self.sender.close();
            self.admission.close();
            return Err(error);
        }
        let response = done.await.map_err(|_| closed())?;
        response.result
    }
}

impl Drop for PayloadReadService {
    fn drop(&mut self) {
        self.sender.close();
        self.admission.close();
        if let Err(error) = self.join() {
            log::error!("{error}");
        }
    }
}
