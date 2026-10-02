use super::*;
use crate::directories::{Directory, DirectoryWriter, MmapDirectory};

fn backends() -> Vec<PayloadReadBackend> {
    let mut backends = vec![PayloadReadBackend::Pool];
    if cfg!(all(target_os = "linux", feature = "io-uring")) {
        backends.push(PayloadReadBackend::IoUring);
    }
    backends
}

#[tokio::test(flavor = "current_thread")]
async fn payload_backends_preserve_range_bytes_and_keep_metadata_mapped() {
    for backend in backends() {
        let root = tempfile::tempdir().unwrap();
        let service = PayloadReadService::new(backend).unwrap();
        let dir = MmapDirectory::new(root.path()).with_payload_reads(service.clone());
        let path = std::path::Path::new("payload");
        let expected: Vec<_> = (0..BYTES + 37).map(|i| (i % 251) as u8).collect();
        dir.write(path, &expected).await.unwrap();
        assert!(dir.open_read(path).await.unwrap().is_sync());
        assert!(dir.open_lazy(path).await.unwrap().is_sync());
        let handle = dir.open_payload(path).await.unwrap();
        assert!(!handle.is_sync());
        assert_eq!(service.backend(), backend);
        let bytes = handle
            .read_bytes_range(3..expected.len() as u64)
            .await
            .unwrap();
        assert_eq!(bytes.as_slice(), &expected[3..]);
        let ranges = [17..31, 5..10, 17..31, 0..0];
        let batch = handle.read_bytes_ranges(&ranges).await.unwrap();
        for (bytes, range) in batch.iter().zip(ranges) {
            assert_eq!(
                bytes.as_slice(),
                &expected[range.start as usize..range.end as usize]
            );
        }
        let before = service.stats().submitted;
        assert_eq!(
            service
                .read_range(
                    Arc::new(File::open(root.path().join(path)).unwrap()),
                    0..MAX_RETURN_BYTES as u64 + 1,
                    None,
                )
                .await
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(service.stats().submitted, before);
        service.shutdown().await.unwrap();
        assert_eq!(
            handle.read_bytes_range(0..1).await.unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
        assert_eq!(bytes.as_slice(), &expected[3..]);
        let stats = service.stats();
        assert_eq!(stats.submitted, stats.completed);
        assert!(stats.max_active <= DEPTH as u64);
        assert_eq!(stats.quarantined_bytes, 0);
    }
}

#[tokio::test]
async fn short_reads_fail_and_admission_is_reusable() {
    for backend in backends() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("payload");
        std::fs::write(&path, b"abcdefgh").unwrap();
        let service = PayloadReadService::new(backend).unwrap();
        let handle = service
            .open(path.clone(), Arc::from("test"), None)
            .await
            .unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(3)
            .unwrap();
        assert_eq!(
            handle.read_bytes_range(0..8).await.unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(
            handle.read_bytes_range(0..3).await.unwrap().as_slice(),
            b"abc"
        );
        service.shutdown().await.unwrap();
        assert_eq!(service.admission.bytes.available_permits(), BYTES);
        assert_eq!(service.admission.slots.available_permits(), DEPTH);
    }
}

#[tokio::test]
async fn cancellation_before_admission_starts_no_io_and_shutdown_wakes_waiters() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("payload");
    std::fs::write(&path, b"abc").unwrap();
    for backend in backends() {
        let service = PayloadReadService::new(backend).unwrap();
        let handle = service
            .open(path.clone(), Arc::from("test"), None)
            .await
            .unwrap();
        let permit = service
            .admission
            .bytes
            .clone()
            .acquire_many_owned(BYTES as u32)
            .await
            .unwrap();
        let mut read = Box::pin(handle.read_bytes_range(0..1));
        assert!(futures::poll!(&mut read).is_pending());
        drop(read);
        assert_eq!(service.stats().submitted, 0);
        let mut waiting = Box::pin(handle.read_bytes_range(0..1));
        assert!(futures::poll!(&mut waiting).is_pending());
        service.shutdown().await.unwrap();
        assert_eq!(waiting.await.unwrap_err().kind(), io::ErrorKind::BrokenPipe);
        drop(permit);
        assert_eq!(service.admission.bytes.available_permits(), BYTES);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn cancelled_submitted_reads_drain_before_shutdown_returns() {
    for backend in backends() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("payload");
        std::fs::write(&path, vec![42; BYTES]).unwrap();
        let service = PayloadReadService::new(backend).unwrap();
        let handle = service
            .open(path.clone(), Arc::from("test"), None)
            .await
            .unwrap();
        // Poll admits a job but does not poll its reply again. Even if the I/O
        // completes immediately, the owned reply still holds the byte permit.
        let mut read = Box::pin(handle.read_bytes_range(0..BYTES as u64));
        assert!(futures::poll!(&mut read).is_pending());
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while service.stats().submitted == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(read);
        drop(handle);
        service.shutdown().await.unwrap();
        let stats = service.stats();
        assert_eq!(stats.submitted, stats.completed);
        assert_eq!(stats.active, 0);
        assert_eq!(service.admission.bytes.available_permits(), BYTES);
        assert_eq!(service.admission.slots.available_permits(), DEPTH);
        assert_eq!(stats.quarantined_bytes, 0);
    }
}

#[tokio::test]
async fn completed_replies_retain_byte_admission_until_consumed() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("payload");
    std::fs::write(&path, vec![42; BYTES]).unwrap();
    for backend in backends() {
        let service = PayloadReadService::new(backend).unwrap();
        let handle = service
            .open(path.clone(), Arc::from("test"), None)
            .await
            .unwrap();
        let mut read = Box::pin(handle.read_bytes_range(0..BYTES as u64));
        assert!(futures::poll!(&mut read).is_pending());
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while service.stats().completed == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(service.admission.bytes.available_permits(), 0);
        let bytes = read.await.unwrap();
        assert_eq!(service.admission.bytes.available_permits(), BYTES);
        service.shutdown().await.unwrap();
        assert_eq!(bytes.as_slice(), &vec![42; BYTES]);
    }
}

#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
#[test]
fn unavailable_ring_is_an_explicit_error_without_fallback() {
    assert_eq!(
        PayloadReadService::new(PayloadReadBackend::IoUring)
            .err()
            .unwrap()
            .kind(),
        io::ErrorKind::Unsupported
    );
}

#[cfg(unix)]
#[tokio::test]
async fn open_payloads_retain_the_original_file_after_path_replacement() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("payload");
    for backend in backends() {
        std::fs::write(&path, b"old bytes").unwrap();
        let service = PayloadReadService::new(backend).unwrap();
        let handle = service
            .open(path.clone(), Arc::from("test"), None)
            .await
            .unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"new bytes").unwrap();
        assert_eq!(handle.read_bytes().await.unwrap().as_slice(), b"old bytes");
        let mut shutdown = Box::pin(service.shutdown());
        let _ = futures::poll!(&mut shutdown);
        drop(shutdown);
        // A cancelled shutdown still owns its join; a second call observes the
        // same completed drain instead of racing a detached worker.
        service.shutdown().await.unwrap();
        assert_eq!(service.stats().active, 0);
    }
}

#[tokio::test]
async fn failed_workers_release_queued_reads_and_directory_leases() {
    for panic in [false, true] {
        let (sender, receiver) = async_channel::bounded(DEPTH);
        let admission = Arc::new(Admission {
            slots: Arc::new(Semaphore::new(DEPTH)),
            bytes: Arc::new(Semaphore::new(BYTES)),
        });
        let stats = Arc::new(Counters::default());
        let scope = scope::Scope::default();
        let file = Arc::new(tempfile::tempfile().unwrap());
        let (reply, mut done) = oneshot::channel();
        sender
            .try_send(Job {
                buffers: buffers::BufferPool::new(0),
                file: file.clone(),
                range: 0..3,
                _slot: admission.slots.clone().acquire_owned().await.unwrap(),
                bytes: admission.bytes.clone().acquire_many_owned(3).await.unwrap(),
                reply,
                _lease: Some(scope.read().await.unwrap()),
            })
            .unwrap();
        let worker = spawn_worker(
            "injected-failure",
            receiver,
            stats.clone(),
            admission.clone(),
            move || {
                assert!(!panic, "injected worker panic");
                Err(io::Error::other("injected worker failure"))
            },
        )
        .unwrap();
        assert!(worker.join().unwrap().is_err());
        // Keep the service's sender alive: closing a channel alone retains jobs.
        let drained =
            tokio::time::timeout(std::time::Duration::from_secs(1), scope.exclusive()).await;
        assert!(
            drained.is_ok(),
            "failed worker retained a queued directory lease"
        );
        assert!(matches!(
            done.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
        assert!(sender.is_closed());
        assert!(admission.slots.is_closed());
        assert!(admission.bytes.is_closed());
        assert_eq!(admission.slots.available_permits(), DEPTH);
        assert_eq!(admission.bytes.available_permits(), BYTES);
        assert_eq!(Arc::strong_count(&file), 1);
        assert_eq!(stats.snapshot().worker_failures, 1);
        assert_eq!(stats.snapshot().skipped, 1);
    }
}

#[tokio::test]
async fn retiring_one_directory_rejects_stale_handles_without_stopping_another() {
    for backend in backends() {
        let root = tempfile::tempdir().unwrap();
        let service = PayloadReadService::with_buffer_budget(backend, BYTES).unwrap();
        let first =
            MmapDirectory::new(root.path().join("first")).with_payload_reads(service.clone());
        let second =
            MmapDirectory::new(root.path().join("second")).with_payload_reads(service.clone());
        let path = std::path::Path::new("data");
        first.write(path, b"first").await.unwrap();
        second.write(path, b"other").await.unwrap();
        let handle = first.open_payload(path).await.unwrap();
        let bytes = handle.read_bytes().await.unwrap();
        first.retire_payload_reads().await.unwrap();
        assert_eq!(
            handle.read_bytes().await.unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
        assert!(first.open_payload(path).await.is_err());
        assert_eq!(
            second
                .open_payload(path)
                .await
                .unwrap()
                .read_bytes()
                .await
                .unwrap()
                .as_slice(),
            b"other"
        );
        first.delete(path).await.unwrap();
        assert_eq!(bytes.as_slice(), b"first");
        drop(bytes);
        assert!(service.stats().buffer_reuses > 0 || service.stats().idle_buffer_bytes > 0);
        service.shutdown().await.unwrap();
        assert_eq!(service.stats().idle_buffer_bytes, 0);
    }
}

#[tokio::test]
async fn cancelled_receiver_keeps_unlink_waiting_until_its_job_finishes() {
    let root = tempfile::tempdir().unwrap();
    let service = PayloadReadService::new(PayloadReadBackend::Pool).unwrap();
    let dir = MmapDirectory::new(root.path()).with_payload_reads(service.clone());
    let path = std::path::Path::new("data");
    dir.write(path, b"abc").await.unwrap();
    let (reply, done) = oneshot::channel();
    let job = Job {
        file: Arc::new(File::open(root.path().join(path)).unwrap()),
        range: 0..3,
        buffers: service.buffers.clone(),
        _slot: service
            .admission
            .slots
            .clone()
            .acquire_owned()
            .await
            .unwrap(),
        bytes: service
            .admission
            .bytes
            .clone()
            .acquire_many_owned(3)
            .await
            .unwrap(),
        reply,
        _lease: Some(dir.payload_scope_for_test().read().await.unwrap()),
    };
    let read = prepare(job, &service.counters).unwrap();
    drop(done);
    let mut unlink = Box::pin(dir.delete(path));
    assert!(futures::poll!(&mut unlink).is_pending());
    assert!(root.path().join(path).exists());
    // Cancellation after preparation must not revoke the worker's lease.
    finish(
        read,
        Err(io::Error::other("injected read failure")),
        &service.counters,
    );
    unlink.await.unwrap();
    assert!(!root.path().join(path).exists());
    service.shutdown().await.unwrap();
}

#[tokio::test]
async fn recycled_reads_charge_capacity_before_io_and_release_admission_with_live_views() {
    for backend in backends() {
        let root = tempfile::tempdir().unwrap();
        let service = PayloadReadService::with_buffer_budget(backend, BYTES).unwrap();
        let dir = MmapDirectory::new(root.path()).with_payload_reads(service.clone());
        let path = std::path::Path::new("data");
        dir.write(path, &[42; 17]).await.unwrap();
        let handle = dir.open_payload(path).await.unwrap();
        // 17 visible bytes need a 32-byte initialized backing allocation.
        let held = service
            .admission
            .bytes
            .clone()
            .acquire_many_owned((BYTES - 31) as u32)
            .await
            .unwrap();
        let mut read = Box::pin(handle.read_bytes());
        assert!(futures::poll!(&mut read).is_pending());
        // Tokio reserves the 31 available permits while waiting for the 32nd;
        // charging only 17 would start I/O and leave 14 permits instead.
        assert_eq!(service.admission.bytes.available_permits(), 0);
        assert_eq!(service.stats().submitted, 0);
        drop(held);
        let bytes = read.await.unwrap();
        assert_eq!(bytes.as_slice(), &[42; 17]);
        assert_eq!(service.admission.bytes.available_permits(), BYTES);
        service.shutdown().await.unwrap();
        assert_eq!(bytes.as_slice(), &[42; 17]);
        drop(bytes);
        assert_eq!(service.stats().idle_buffer_bytes, 0);
    }
}

#[tokio::test]
async fn directory_drain_survives_cancelled_open_without_a_published_owner() {
    let root = tempfile::tempdir().unwrap();
    let service = PayloadReadService::new(PayloadReadBackend::Pool).unwrap();
    let dir = MmapDirectory::new(root.path()).with_payload_reads(service.clone());
    let path = std::path::Path::new("data");
    dir.write(path, b"abc").await.unwrap();
    // The accepted job retains only the gate guard after its opening future and
    // original directory disappear. A later delete must find that same gate.
    let job_lease = dir.payload_scope_for_test().read().await.unwrap();
    drop(dir);
    let deleting = MmapDirectory::new(root.path()).with_payload_reads(service.clone());
    let mut drain = Box::pin(deleting.retire_payload_reads());
    assert!(
        futures::poll!(&mut drain).is_pending(),
        "retirement lost the cancelled opener's outstanding job"
    );
    drop(job_lease);
    drain.await.unwrap();
    deleting.delete(path).await.unwrap();
    // Recreating the pathname gets a fresh generation, even if a stale directory
    // retains the retired gate. The old generation stays closed.
    let next = MmapDirectory::new(root.path()).with_payload_reads(service.clone());
    next.write(path, b"new").await.unwrap();
    assert!(deleting.open_payload(path).await.is_err());
    assert_eq!(
        next.open_payload(path)
            .await
            .unwrap()
            .read_bytes()
            .await
            .unwrap()
            .as_slice(),
        b"new"
    );
    service.shutdown().await.unwrap();
}
