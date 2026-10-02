//! Range validation and scheduling for immutable mapped and lazy files.
use super::OwnedBytes;
use std::{io, ops::Range, sync::Arc};

/// Callback type for nonempty lazy range reads. Return exactly the requested
/// bytes; started external I/O must retain its resources after future cancellation.
#[cfg(not(target_arch = "wasm32"))]
pub type RangeReadFn = Arc<
    dyn Fn(
            Range<u64>,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = io::Result<OwnedBytes>> + Send>>
        + Send
        + Sync,
>;

#[cfg(target_arch = "wasm32")]
pub type RangeReadFn = Arc<
    dyn Fn(
        Range<u64>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = io::Result<OwnedBytes>>>>,
>;

/// Unified file handle for both inline (mmap/RAM) and lazy (HTTP/filesystem) access.
///
/// Replaces the previous `FileSlice`, `LazyFileHandle`, and `LazyFileSlice` types.
/// - **Inline**: data is available synchronously (mmap, RAM). Sync reads via `read_bytes_range_sync`.
/// - **Lazy**: data is fetched on-demand via async callback (HTTP, filesystem).
///
/// Use `.slice()` to create sub-range views (zero-copy for Inline, offset-adjusted for Lazy).
#[derive(Clone)]
pub struct FileHandle {
    inner: FileHandleInner,
}

#[derive(Clone)]
enum FileHandleInner {
    /// Data available inline — sync reads possible (mmap, RAM)
    Inline {
        data: OwnedBytes,
        offset: u64,
        len: u64,
    },
    /// Data fetched on-demand via async callback (HTTP, filesystem)
    Lazy {
        read_fn: RangeReadFn,
        offset: u64,
        len: u64,
        /// Index name for the `summa_directory_read_*` metric labels.
        label: Arc<str>,
    },
}

/// Late-bound index name for Directory-layer metric labels
/// (`summa_directory_read_*`, `summa_cold_write_bytes_total`).
///
/// Directories are constructed before the schema is loaded, so the label is
/// attached afterwards: `Index::open`/`create` call
/// `Directory::set_index_label(schema.index_label())` on the index's
/// directory instance. Reads happen at handle/writer creation, not per IO.
#[derive(Clone, Debug)]
pub struct IndexLabel(Arc<std::sync::RwLock<Arc<str>>>);

impl Default for IndexLabel {
    fn default() -> Self {
        Self(Arc::new(std::sync::RwLock::new(Arc::from("unknown"))))
    }
}

impl IndexLabel {
    /// Current label ("unknown" until set).
    pub fn get(&self) -> Arc<str> {
        self.0.read().expect("IndexLabel lock poisoned").clone()
    }

    /// Set the label (idempotent; last write wins).
    pub fn set(&self, label: &str) {
        *self.0.write().expect("IndexLabel lock poisoned") = Arc::from(label);
    }
}

impl std::fmt::Debug for FileHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            FileHandleInner::Inline { len, offset, .. } => f
                .debug_struct("FileHandle::Inline")
                .field("offset", offset)
                .field("len", len)
                .finish(),
            FileHandleInner::Lazy { len, offset, .. } => f
                .debug_struct("FileHandle::Lazy")
                .field("offset", offset)
                .field("len", len)
                .finish(),
        }
    }
}

impl FileHandle {
    pub(crate) const MAX_BATCH_RANGES: usize = 32;
    pub(crate) const MAX_BATCH_CONCURRENCY: usize = 8;
    pub(crate) const MAX_BATCH_BYTES: usize = 8 * 1024 * 1024;

    /// Give a batch of borrowed ranges one local reference-count owner.
    /// Lazy handles keep their original bounded range-read behavior.
    pub(crate) fn with_local_owner(&self) -> Self {
        match &self.inner {
            FileHandleInner::Inline { data, offset, len } => Self {
                inner: FileHandleInner::Inline {
                    data: data.clone().with_local_owner(),
                    offset: *offset,
                    len: *len,
                },
            },
            FileHandleInner::Lazy { .. } => self.clone(),
        }
    }

    /// Create an inline file handle from owned bytes (mmap, RAM).
    /// Sync reads are available.
    pub fn from_bytes(data: OwnedBytes) -> Self {
        let len = data.len() as u64;
        Self {
            inner: FileHandleInner::Inline {
                data,
                offset: 0,
                len,
            },
        }
    }

    /// Create an empty file handle.
    pub fn empty() -> Self {
        Self::from_bytes(OwnedBytes::empty())
    }

    /// Create a lazy file handle from an async range-read callback.
    /// Only async reads are available. Reads emit `summa_directory_read_*`
    /// with `index="unknown"` — use [`FileHandle::lazy_labeled`] when the
    /// owning index is known.
    pub fn lazy(len: u64, read_fn: RangeReadFn) -> Self {
        Self::lazy_labeled(len, read_fn, Arc::from("unknown"))
    }

    /// [`FileHandle::lazy`] with an index name for metric labels.
    pub fn lazy_labeled(len: u64, read_fn: RangeReadFn, label: Arc<str>) -> Self {
        Self {
            inner: FileHandleInner::Lazy {
                read_fn,
                offset: 0,
                len,
                label,
            },
        }
    }

    /// Total length in bytes.
    #[inline]
    pub fn len(&self) -> u64 {
        match &self.inner {
            FileHandleInner::Inline { len, .. } => *len,
            FileHandleInner::Lazy { len, .. } => *len,
        }
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether synchronous reads are available (inline/mmap data).
    #[inline]
    pub fn is_sync(&self) -> bool {
        matches!(&self.inner, FileHandleInner::Inline { .. })
    }

    /// Create a sub-range view. Zero-copy for Inline, offset-adjusted for Lazy.
    ///
    /// # Panics
    /// Panics if the range is reversed or extends beyond this handle.
    pub fn slice(&self, range: Range<u64>) -> Self {
        assert!(
            range.start <= range.end && range.end <= self.len(),
            "file slice out of bounds"
        );
        match &self.inner {
            FileHandleInner::Inline { data, offset, len } => {
                let new_offset = offset + range.start;
                let new_len = range.end - range.start;
                debug_assert!(
                    new_offset + new_len <= offset + len,
                    "slice out of bounds: {}+{} > {}+{}",
                    new_offset,
                    new_len,
                    offset,
                    len
                );
                Self {
                    inner: FileHandleInner::Inline {
                        data: data.clone(),
                        offset: new_offset,
                        len: new_len,
                    },
                }
            }
            FileHandleInner::Lazy {
                read_fn,
                offset,
                len,
                label,
            } => {
                let new_offset = offset + range.start;
                let new_len = range.end - range.start;
                debug_assert!(
                    new_offset + new_len <= offset + len,
                    "slice out of bounds: {}+{} > {}+{}",
                    new_offset,
                    new_len,
                    offset,
                    len
                );
                Self {
                    inner: FileHandleInner::Lazy {
                        read_fn: Arc::clone(read_fn),
                        offset: new_offset,
                        len: new_len,
                        label: Arc::clone(label),
                    },
                }
            }
        }
    }

    /// Advise the kernel about the access pattern for a byte range of this handle.
    ///
    /// Only effective for Inline handles backed by mmap; no-op for Lazy
    /// handles (HTTP, filesystem callbacks) and heap-backed data.
    #[cfg(feature = "native")]
    pub fn madvise_range(&self, range: Range<u64>, advice: libc::c_int) {
        if let FileHandleInner::Inline { data, offset, len } = &self.inner {
            let end = range.end.min(*len);
            if range.start >= end {
                return;
            }
            let start = (*offset + range.start) as usize;
            let end = (*offset + end) as usize;
            data.madvise_range(start..end, advice);
        }
    }

    /// Async range read — works for both Inline and Lazy.
    pub async fn read_bytes_range(&self, range: Range<u64>) -> io::Result<OwnedBytes> {
        if let FileHandleInner::Lazy { label, .. } = &self.inner {
            let timer = crate::observe::Timer::start();
            let result = self.read_bytes_range_unmetered(range).await;
            if let Ok(bytes) = &result {
                crate::observe::directory_read(label, "lazy_range", timer.secs(), bytes.len());
            }
            result
        } else {
            self.read_bytes_range_unmetered(range).await
        }
    }

    // Cache wrappers meter the outer read once, including cache hits. Their
    // misses retain this handle's owner and validation without double counting.
    pub(super) async fn read_bytes_range_unmetered(
        &self,
        range: Range<u64>,
    ) -> io::Result<OwnedBytes> {
        self.check_range(&range)?;
        if range.is_empty() {
            return Ok(OwnedBytes::empty());
        }
        match &self.inner {
            FileHandleInner::Inline { data, offset, .. } => {
                let start = (*offset + range.start) as usize;
                let end = (*offset + range.end) as usize;
                Ok(data.slice(start..end))
            }
            FileHandleInner::Lazy {
                read_fn, offset, ..
            } => {
                let abs_start = offset + range.start;
                let abs_end = offset + range.end;
                let bytes = (read_fn)(abs_start..abs_end).await?;
                if bytes.len() as u64 != range.end - range.start {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "range reader returned an incorrect byte count",
                    ));
                }
                Ok(bytes)
            }
        }
    }

    /// Read a bounded batch in input order, overlapping up to eight lazy reads.
    /// Validates every range and the aggregate budget before starting any I/O.
    /// Ordinary errors drain the batch. On cancellation, lazy callbacks must
    /// retain resources owned by already-started external I/O until completion.
    pub async fn read_bytes_ranges(&self, ranges: &[Range<u64>]) -> io::Result<Vec<OwnedBytes>> {
        if ranges.len() > Self::MAX_BATCH_RANGES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read batch exceeds 32 ranges",
            ));
        }
        let requests: Vec<_> = ranges.iter().cloned().map(|range| (self, range)).collect();
        Self::read_many(&requests).await
    }

    /// Execute one global admission/concurrency budget across several files.
    pub(crate) async fn read_many(
        requests: &[(&FileHandle, Range<u64>)],
    ) -> io::Result<Vec<OwnedBytes>> {
        use futures::{StreamExt, stream};
        if requests.len() > Self::MAX_BATCH_RANGES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read batch exceeds 32 ranges",
            ));
        }
        let mut bytes = 0_u64;
        for (handle, range) in requests {
            handle.check_range(range)?;
            bytes = bytes.checked_add(range.end - range.start).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "read batch size overflow")
            })?;
        }
        if bytes > Self::MAX_BATCH_BYTES as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read batch exceeds 8 MiB",
            ));
        }
        // Materialize the bounded set of futures before awaiting. This also
        // keeps borrowed iterator closures out of the RPC future's Send bound.
        let reads: Vec<_> = requests
            .iter()
            .cloned()
            .enumerate()
            .map(|(position, request)| async move {
                (position, request.0.read_bytes_range(request.1).await)
            })
            .collect();
        // Replenish from any completed read. An ordered future buffer would
        // leave storage idle behind an early slow read, even with free slots.
        let mut results: Vec<Option<io::Result<OwnedBytes>>> =
            (0..reads.len()).map(|_| None).collect();
        let mut pending = stream::iter(reads).buffer_unordered(Self::MAX_BATCH_CONCURRENCY);
        while let Some((position, result)) = pending.next().await {
            results[position] = Some(result);
        }
        // Restore both result order and first-input-error selection after draining.
        results.into_iter().map(Option::unwrap).collect()
    }

    fn check_range(&self, range: &Range<u64>) -> io::Result<()> {
        if range.start > range.end || range.end > self.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("range {range:?} out of bounds (len: {})", self.len()),
            ));
        }
        Ok(())
    }

    /// Read all bytes.
    pub async fn read_bytes(&self) -> io::Result<OwnedBytes> {
        self.read_bytes_range(0..self.len()).await
    }

    /// Synchronous range read — only works for Inline handles.
    /// Returns `Err` if the handle is Lazy.
    #[inline]
    pub fn read_bytes_range_sync(&self, range: Range<u64>) -> io::Result<OwnedBytes> {
        self.check_range(&range)?;
        match &self.inner {
            FileHandleInner::Inline { data, offset, .. } => {
                let start = (*offset + range.start) as usize;
                let end = (*offset + range.end) as usize;
                Ok(data.slice(start..end))
            }
            FileHandleInner::Lazy { .. } => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Synchronous read not available on lazy file handle",
            )),
        }
    }

    /// Synchronous read of all bytes — only works for Inline handles.
    #[inline]
    pub fn read_bytes_sync(&self) -> io::Result<OwnedBytes> {
        self.read_bytes_range_sync(0..self.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn empty_lazy_ranges_do_not_call_the_backend() {
        let handle = FileHandle::lazy(8, Arc::new(|_| panic!("empty read reached backend")));
        assert!(handle.read_bytes_range(8..8).await.unwrap().is_empty());
        assert_eq!(
            handle.read_bytes_range(9..9).await.unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[tokio::test]
    async fn single_range_reads_reject_reversed_ranges_before_access() {
        let lazy = FileHandle::lazy(8, Arc::new(|_| panic!("invalid range reached backend")));
        let inline = FileHandle::from_bytes(OwnedBytes::new(vec![0; 8]));
        let range = Range { start: 5, end: 3 };
        for handle in [&lazy, &inline] {
            assert_eq!(
                handle
                    .read_bytes_range(range.clone())
                    .await
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::InvalidInput
            );
        }
        assert_eq!(
            inline.read_bytes_range_sync(range).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[tokio::test]
    async fn single_lazy_reads_reject_short_and_overlong_responses() {
        for actual in [1, 3] {
            let handle = FileHandle::lazy(
                8,
                Arc::new(move |_| Box::pin(async move { Ok(OwnedBytes::new(vec![0; actual])) })),
            );
            assert_eq!(
                handle.read_bytes_range(2..4).await.unwrap_err().kind(),
                io::ErrorKind::UnexpectedEof
            );
        }
    }

    #[tokio::test]
    async fn batched_reads_validate_all_ranges_before_starting_io() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let handle = FileHandle::lazy(
            16 * 1024 * 1024,
            Arc::new(move |range| {
                observed.fetch_add(1, Ordering::Relaxed);
                Box::pin(
                    async move { Ok(OwnedBytes::new(vec![0; (range.end - range.start) as usize])) },
                )
            }),
        );
        for ranges in [
            vec![0..1, Range { start: 4, end: 3 }],
            vec![0..1, 0..20_000_000],
            vec![0..1; 33],
            std::iter::once(0..9_000_000).collect(),
        ] {
            assert_eq!(
                handle.read_bytes_ranges(&ranges).await.unwrap_err().kind(),
                io::ErrorKind::InvalidInput
            );
            assert_eq!(calls.load(Ordering::Relaxed), 0);
        }
    }

    #[tokio::test]
    async fn batched_reads_overlap_preserve_order_and_drain_errors() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let completed = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let (done, running, maximum) = (completed.clone(), active.clone(), peak.clone());
        let handle = FileHandle::lazy(
            64,
            Arc::new(move |range| {
                let (done, running, maximum) = (done.clone(), running.clone(), maximum.clone());
                Box::pin(async move {
                    let n = running.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum.fetch_max(n, Ordering::SeqCst);
                    for _ in 0..(3 - range.start % 3) {
                        tokio::task::yield_now().await;
                    }
                    running.fetch_sub(1, Ordering::SeqCst);
                    done.fetch_add(1, Ordering::SeqCst);
                    if range.start == 63 {
                        return Err(io::Error::other("read failed"));
                    }
                    Ok(OwnedBytes::new(vec![
                        range.start as u8;
                        (range.end - range.start) as usize
                    ]))
                })
            }),
        );
        let slice = handle.slice(10..50);
        let ranges = [3..5, 0..1, 3..5, 4..4];
        let bytes = slice.read_bytes_ranges(&ranges).await.unwrap();
        assert_eq!(
            bytes.iter().map(|b| b.as_slice()).collect::<Vec<_>>(),
            vec![&[13, 13][..], &[10], &[13, 13], &[]]
        );
        assert!(peak.load(Ordering::SeqCst) > 1);
        completed.store(0, Ordering::SeqCst);
        let ranges: Vec<_> = std::iter::once(63..64)
            .chain((0..31).map(|i| i..i + 1))
            .collect();
        assert!(handle.read_bytes_ranges(&ranges).await.is_err());
        assert_eq!(completed.load(Ordering::SeqCst), 32);
        assert!(peak.load(Ordering::SeqCst) <= 8);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn batched_reads_replenish_slots_while_the_first_read_is_stalled() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let started = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let (observed, gate) = (started.clone(), release.clone());
        let handle = FileHandle::lazy(
            32,
            Arc::new(move |range| {
                let (observed, gate) = (observed.clone(), gate.clone());
                Box::pin(async move {
                    observed.fetch_add(1, Ordering::SeqCst);
                    if range.start == 0 {
                        gate.acquire().await.unwrap().forget();
                    }
                    Ok(OwnedBytes::new(vec![range.start as u8]))
                })
            }),
        );
        let ranges: Vec<_> = (0..32).map(|i| i..i + 1).collect();
        let mut pending = Box::pin(handle.read_bytes_ranges(&ranges));
        assert!(futures::poll!(&mut pending).is_pending());
        assert!(
            started.load(Ordering::SeqCst) > 8,
            "completed later reads must free admission without waiting for the first result"
        );
        release.add_permits(1);
        let bytes = pending.await.unwrap();
        for (i, bytes) in bytes.iter().enumerate() {
            assert_eq!(bytes.as_slice(), &[i as u8]);
        }
    }

    #[tokio::test]
    async fn batched_reads_return_the_first_input_error_after_draining_later_errors() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let completed = Arc::new(AtomicUsize::new(0));
        let observed = completed.clone();
        let handle = FileHandle::lazy(
            16,
            Arc::new(move |range| {
                let observed = observed.clone();
                Box::pin(async move {
                    if range.start == 0 {
                        tokio::task::yield_now().await;
                    }
                    observed.fetch_add(1, Ordering::SeqCst);
                    if range.start == 0 {
                        Err(io::Error::other("first input error"))
                    } else if range.start == 1 {
                        Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            "later error",
                        ))
                    } else {
                        Ok(OwnedBytes::new(vec![range.start as u8]))
                    }
                })
            }),
        );
        let ranges: Vec<_> = (0..16).map(|i| i..i + 1).collect();
        let error = handle.read_bytes_ranges(&ranges).await.unwrap_err();
        assert_eq!(error.to_string(), "first input error");
        assert_eq!(completed.load(Ordering::SeqCst), 16);
    }

    #[tokio::test]
    async fn batched_reads_share_one_concurrency_limit_across_files() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..32)
            .map(|file| {
                let (active, peak, completed) = (active.clone(), peak.clone(), completed.clone());
                FileHandle::lazy(
                    8,
                    Arc::new(move |range| {
                        let (active, peak, completed) =
                            (active.clone(), peak.clone(), completed.clone());
                        Box::pin(async move {
                            let n = active.fetch_add(1, Ordering::SeqCst) + 1;
                            peak.fetch_max(n, Ordering::SeqCst);
                            tokio::task::yield_now().await;
                            active.fetch_sub(1, Ordering::SeqCst);
                            completed.fetch_add(1, Ordering::SeqCst);
                            Ok(OwnedBytes::new(vec![
                                file;
                                (range.end - range.start) as usize
                            ]))
                        })
                    }),
                )
            })
            .collect();
        let mut requests: Vec<_> = handles.iter().map(|handle| (handle, 0..1)).collect();
        requests[31].1 = 0..9;
        assert!(FileHandle::read_many(&requests).await.is_err());
        assert_eq!(
            completed.load(Ordering::SeqCst),
            0,
            "reject the entire multi-file batch before starting any file"
        );
        requests[31].1 = 0..1;
        let bytes = FileHandle::read_many(&requests).await.unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), 8);
        assert_eq!(completed.load(Ordering::SeqCst), 32);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        for (file, bytes) in bytes.iter().enumerate() {
            assert_eq!(bytes.as_slice(), &[file as u8]);
        }
    }

    #[tokio::test]
    async fn batched_reads_cancel_unstarted_work_and_release_pending_owners() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct PendingOwner(Arc<AtomicUsize>);
        impl Drop for PendingOwner {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let active = Arc::new(AtomicUsize::new(0));
        let starts = Arc::new(AtomicUsize::new(0));
        let (owners, calls) = (active.clone(), starts.clone());
        let handle = FileHandle::lazy(
            32,
            Arc::new(move |_| {
                let (owners, calls) = (owners.clone(), calls.clone());
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    owners.fetch_add(1, Ordering::SeqCst);
                    let _owner = PendingOwner(owners);
                    futures::future::pending::<()>().await;
                    Ok(OwnedBytes::new(Vec::new()))
                })
            }),
        );
        let ranges: Vec<_> = (0..32).map(|i| i..i + 1).collect();
        let mut pending = Box::pin(handle.read_bytes_ranges(&ranges));
        assert!(futures::poll!(&mut pending).is_pending());
        assert_eq!(active.load(Ordering::SeqCst), 8);
        drop(pending);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert_eq!(starts.load(Ordering::SeqCst), 8);
    }

    #[tokio::test]
    async fn batched_inline_views_preserve_bytes_and_survive_handle_drop() {
        let handle = FileHandle::from_bytes(OwnedBytes::new((0..64).collect()));
        let slice = handle.slice(10..40);
        let bytes = slice
            .read_bytes_ranges(&[20..30, 0..5, 0..5, 10..10])
            .await
            .unwrap();
        drop(slice);
        drop(handle);
        assert_eq!(bytes[0].as_slice(), &(30..40).collect::<Vec<_>>());
        assert_eq!(bytes[1].as_slice(), &(10..15).collect::<Vec<_>>());
        assert_eq!(bytes[1].as_slice(), bytes[2].as_slice());
        assert!(bytes[3].is_empty());
    }

    #[tokio::test]
    async fn batched_reads_reject_short_payloads() {
        let handle = FileHandle::lazy(
            10,
            Arc::new(|_| Box::pin(async { Ok(OwnedBytes::new(vec![0])) })),
        );
        assert_eq!(
            handle
                .read_bytes_ranges(std::slice::from_ref(&(0..2)))
                .await
                .unwrap_err()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[tokio::test]
    async fn test_file_handle() {
        let data = OwnedBytes::new(b"hello world".to_vec());
        let handle = FileHandle::from_bytes(data);

        assert_eq!(handle.len(), 11);
        assert!(handle.is_sync());

        let sub = handle.slice(0..5);
        let bytes = sub.read_bytes().await.unwrap();
        assert_eq!(bytes.as_slice(), b"hello");

        let sub2 = handle.slice(6..11);
        let bytes2 = sub2.read_bytes().await.unwrap();
        assert_eq!(bytes2.as_slice(), b"world");

        // Sync reads work on inline handles
        let sync_bytes = handle.read_bytes_range_sync(0..5).unwrap();
        assert_eq!(sync_bytes.as_slice(), b"hello");
    }

    #[test]
    fn local_byte_owners_share_storage_without_repeated_global_refcounts() {
        let backing = Arc::new((0u8..64).collect::<Vec<_>>());
        let handle = FileHandle::from_bytes(OwnedBytes::from_arc_vec(backing.clone(), 3..61));
        let local = handle.slice(2..40).with_local_owner().with_local_owner();
        let count = Arc::strong_count(&backing);
        let views: Vec<_> = (0..20)
            .map(|i| local.read_bytes_range_sync(i..i + 3).unwrap())
            .collect();
        assert_eq!(Arc::strong_count(&backing), count);
        drop(local);
        drop(handle);
        let survivor = std::thread::spawn(move || {
            for (i, view) in views.iter().enumerate() {
                assert_eq!(
                    view.as_slice(),
                    &[(i + 5) as u8, (i + 6) as u8, (i + 7) as u8]
                );
            }
            views[7].clone()
        })
        .join()
        .unwrap();
        assert_eq!(survivor.as_slice(), &[12, 13, 14]);
        drop(survivor);
        assert_eq!(Arc::strong_count(&backing), 1);
    }
}
