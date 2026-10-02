use super::super::{PayloadReadBackend, PayloadReadService};
use super::*;
use std::{fs::File, io::Write, os::fd::FromRawFd, time::Duration};
use tokio::sync::{Semaphore, oneshot};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum WaitAt {
    Idle,
    BeforePoll,
    SubmittedPoll,
}

pub(super) struct WaitPause {
    pub(super) at: WaitAt,
    ready: oneshot::Sender<()>,
    resume: std::sync::mpsc::Receiver<()>,
}
impl WaitPause {
    pub(super) fn wait(self) {
        self.ready.send(()).unwrap();
        self.resume.recv_timeout(Duration::from_secs(5)).unwrap();
    }
}

fn pipe() -> (Arc<File>, File) {
    let mut fds = [-1; 2];
    // SAFETY: pipe2 initializes exactly these two descriptor slots on success.
    assert_eq!(unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) }, 0);
    // SAFETY: successful pipe2 transfers two distinct, newly owned descriptors.
    unsafe {
        (
            Arc::new(File::from_raw_fd(fds[0])),
            File::from_raw_fd(fds[1]),
        )
    }
}

#[tokio::test]
async fn notification_before_poll_registration_is_not_lost() {
    queued_read_progress(false).await;
}

#[tokio::test]
async fn queued_read_wakes_ring_while_prior_read_is_pending() {
    queued_read_progress(true).await;
}

async fn queued_read_progress(submitted: bool) {
    let service = PayloadReadService::new(PayloadReadBackend::Pool).unwrap();
    let wake = Arc::new(Wake::new(service.counters.clone()).unwrap());
    let mut reactor = Reactor::with_wake(service.counters.clone(), wake.clone()).unwrap();
    let (waiting, entered_wait) = oneshot::channel();
    let (resume, proceed) = std::sync::mpsc::channel();
    reactor.pause_for_test = Some(WaitPause {
        at: if submitted {
            WaitAt::SubmittedPoll
        } else {
            WaitAt::BeforePoll
        },
        ready: waiting,
        resume: proceed,
    });
    let (sender, receiver) = async_channel::bounded(DEPTH);
    let worker = std::thread::spawn(move || reactor.run(receiver));
    let (pending_file, mut release) = pipe();
    async fn job(
        file: Arc<File>,
        service: &PayloadReadService,
    ) -> (Job, oneshot::Receiver<super::super::Reply>) {
        let (reply, done) = oneshot::channel();
        (
            Job {
                file,
                range: 0..1,
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
                    .acquire_many_owned(1)
                    .await
                    .unwrap(),
                reply,
                _lease: None,
            },
            done,
        )
    }
    let (pending, pending_done) = job(pending_file, &service).await;
    sender.send(pending).await.unwrap();
    // The control counter is initially zero. This hook fires after queue
    // draining and before entering the kernel, so the late request cannot
    // be picked up by the worker's preceding try_recv loop.
    entered_wait.await.unwrap();
    let mut ready_file = tempfile::tempfile().unwrap();
    ready_file.write_all(b"r").unwrap();
    let (ready, ready_done) = job(Arc::new(ready_file), &service).await;
    sender.send(ready).await.unwrap();
    wake.notify().unwrap();
    resume.send(()).unwrap();
    let independent = tokio::time::timeout(std::time::Duration::from_secs(5), ready_done).await;
    // Always release the delayed read before asserting, including on timeout.
    release.write_all(b"p").unwrap();
    pending_done.await.unwrap().result.unwrap();
    sender.close();
    worker.join().unwrap().unwrap();
    service.shutdown().await.unwrap();
    assert_eq!(
        independent
            .expect("new read waited for unrelated pending I/O")
            .unwrap()
            .result
            .unwrap()
            .as_slice(),
        b"r"
    );
    assert!(service.stats().wakeups > 0);
    assert_eq!(service.stats().active, 0);
    assert_eq!(service.stats().quarantined_bytes, 0);
}

#[tokio::test]
async fn failed_cancel_quarantines_storage_and_refuses_unlink_and_shutdown() {
    use crate::directories::{DirectoryWriter, MmapDirectory};
    let root = tempfile::tempdir().unwrap();
    let service = PayloadReadService::new(PayloadReadBackend::Pool).unwrap();
    let directory = MmapDirectory::new(root.path()).with_payload_reads(service.clone());
    let path = std::path::Path::new("payload");
    directory.write(path, b"x").await.unwrap();
    let (reply, done) = oneshot::channel();
    let mut reactor = Reactor::new(service.counters.clone()).unwrap();
    reactor
        .enqueue(Job {
            file: Arc::new(std::fs::File::open(root.path().join(path)).unwrap()),
            range: 0..1,
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
                .acquire_many_owned(1)
                .await
                .unwrap(),
            reply,
            _lease: Some(directory.payload_scope_for_test().read().await.unwrap()),
        })
        .unwrap();
    // No request was submitted. Inject the cancellation outcome at the
    // resource-release boundary, without relying on a broken real kernel.
    drop(reactor.ring.take());
    reactor.release_slots(Err(io::Error::other("injected cancellation failure")));
    assert!(done.await.unwrap().result.is_err());
    assert_eq!(service.stats().quarantined_bytes, 1);
    assert!(directory.delete(path).await.is_err());
    assert!(root.path().join(path).exists());
    assert!(directory.retire_payload_reads().await.is_err());
    assert!(service.shutdown().await.is_err());
    // One byte and one descriptor intentionally survive until process exit.
}

#[tokio::test]
async fn unwinding_with_submitted_reads_drains_before_releasing_owned_resources() {
    let file = tempfile::tempfile().unwrap();
    file.set_len(1024 * 1024).unwrap();
    let file = Arc::new(file);
    let slots = Arc::new(Semaphore::new(DEPTH));
    let bytes = Arc::new(Semaphore::new(super::super::BYTES));
    let stats = Arc::new(Counters::default());
    let mut reactor = Reactor::new(stats.clone()).unwrap();
    let mut replies = Vec::new();
    for _ in 0..DEPTH {
        let (reply, done) = oneshot::channel();
        reactor
            .enqueue(Job {
                _lease: None,
                buffers: super::super::buffers::BufferPool::new(0),
                file: file.clone(),
                range: 0..65536,
                _slot: slots.clone().acquire_owned().await.unwrap(),
                bytes: bytes.clone().acquire_many_owned(65536).await.unwrap(),
                reply,
            })
            .unwrap();
        replies.push(done);
    }
    reactor.ring.as_mut().unwrap().submit().unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _reactor = reactor;
        panic!("injected worker failure before consuming original completions");
    }));
    assert!(result.is_err());
    for reply in replies {
        assert!(reply.await.unwrap().result.is_err());
    }
    assert_eq!(slots.available_permits(), DEPTH);
    assert_eq!(bytes.available_permits(), super::super::BYTES);
    assert_eq!(Arc::strong_count(&file), 1);
    assert_eq!(stats.snapshot().active, 0);
    assert_eq!(stats.snapshot().quarantined_bytes, 0);
}

#[tokio::test]
async fn notification_failure_closes_admission_and_drains_pending_reads() {
    failed_notifier_cleanup(false).await;
}

#[tokio::test]
async fn cancelled_reader_does_not_prevent_failed_notifier_cleanup() {
    failed_notifier_cleanup(true).await;
}

async fn failed_notifier_cleanup(cancel_reader: bool) {
    let service = PayloadReadService::new(PayloadReadBackend::IoUring).unwrap();
    let root = tempfile::tempdir().unwrap();
    let scope = service.directory_scope(root.path());
    let (pending_file, mut release) = pipe();
    let pending_service = service.clone();
    let pending_scope = scope.clone();
    let mut pending = tokio::spawn(async move {
        pending_service
            .read(pending_file, 0..1, Some(pending_scope))
            .await
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while service.stats().active == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    if cancel_reader {
        pending.abort();
    }
    service.wake.as_ref().unwrap().fail_next_notification();
    let ready = Arc::new(tempfile::tempfile().unwrap());
    ready.set_len(1).unwrap();
    assert!(service.read(ready.clone(), 0..1, None).await.is_err());
    let admission_closed = service.sender.is_closed() && service.admission.slots.is_closed();
    let refused = service.read(ready, 0..1, None).await.is_err();
    let drained = tokio::time::timeout(Duration::from_secs(2), &mut pending).await;
    // Cleanup even when testing a broken notifier implementation; no hung pipe.
    let _ = release.write_all(b"p");
    let shutdown = service.shutdown().await;
    assert!(
        admission_closed && refused,
        "notification failure left admission open"
    );
    if cancel_reader {
        assert!(drained.unwrap().unwrap_err().is_cancelled());
    } else {
        assert!(
            drained
                .expect("failed notifier stranded an accepted read")
                .unwrap()
                .is_err()
        );
    }
    assert!(shutdown.is_err());
    assert_eq!(service.stats().notification_failures, 1);
    assert_eq!(service.stats().worker_failures, 1);
    assert_eq!(service.stats().active, 0);
    assert_eq!(service.stats().quarantined_bytes, 0);
    assert_eq!(service.stats().submitted, service.stats().completed);
    assert_eq!(service.admission.slots.available_permits(), DEPTH);
    assert_eq!(
        service.admission.bytes.available_permits(),
        super::super::BYTES
    );
    // Accepted directory leases are gone, including when their caller vanished.
    drop(
        tokio::time::timeout(Duration::from_secs(1), scope.exclusive())
            .await
            .unwrap(),
    );
}

#[test]
fn dropping_an_unstarted_reactor_releases_its_notification_owner() {
    let counters = Arc::new(Counters::default());
    let wake = Arc::new(Wake::new(counters.clone()).unwrap());
    let owner = Arc::downgrade(&wake);
    let reactor = Reactor::with_wake(counters, wake).unwrap();
    drop(reactor);
    assert!(owner.upgrade().is_none());
}

#[test]
fn dropping_an_armed_idle_reactor_releases_its_notification_owner() {
    let counters = Arc::new(Counters::default());
    let wake = Arc::new(Wake::new(counters.clone()).unwrap());
    let owner = Arc::downgrade(&wake);
    let mut reactor = Reactor::with_wake(counters, wake).unwrap();
    // No payload jobs: timeout leaves just the control poll armed in the kernel.
    reactor.wait().unwrap();
    assert!(reactor.wake_armed);
    drop(reactor);
    assert!(owner.upgrade().is_none());
}

#[tokio::test]
async fn idle_queue_close_preserves_notification_failure() {
    let counters = Arc::new(Counters::default());
    let wake = Arc::new(Wake::new(counters.clone()).unwrap());
    let mut reactor = Reactor::with_wake(counters.clone(), wake.clone()).unwrap();
    let (ready, entered_idle) = oneshot::channel();
    let (resume, proceed) = std::sync::mpsc::channel();
    reactor.pause_for_test = Some(WaitPause {
        at: WaitAt::Idle,
        ready,
        resume: proceed,
    });
    let (sender, receiver) = async_channel::bounded(DEPTH);
    let worker = std::thread::spawn(move || reactor.run(receiver));
    entered_idle.await.unwrap();
    // Failure arrives after the loop's health check, before an empty closed
    // queue returns from recv_blocking. It must not become successful shutdown.
    wake.fail_next_notification();
    assert!(wake.notify().is_err());
    sender.close();
    resume.send(()).unwrap();
    assert!(
        worker.join().unwrap().is_err(),
        "idle queue close hid notification failure"
    );
    assert_eq!(counters.snapshot().notification_failures, 1);
    assert_eq!(counters.snapshot().active, 0);
}

#[tokio::test]
async fn healthy_read_survives_notification_health_timeouts() {
    let service = PayloadReadService::new(PayloadReadBackend::IoUring).unwrap();
    let reader = service.clone();
    let (file, mut release) = pipe();
    let mut pending = tokio::spawn(async move { reader.read(file, 0..1, None).await });
    let still_pending = tokio::time::timeout(Duration::from_millis(250), &mut pending).await;
    release.write_all(b"p").unwrap();
    assert!(
        still_pending.is_err(),
        "health timeout cancelled a healthy read"
    );
    assert_eq!(pending.await.unwrap().unwrap().as_slice(), b"p");
    service.shutdown().await.unwrap();
    let stats = service.stats();
    assert_eq!(stats.errors, 0);
    assert_eq!(stats.notification_failures, 0);
    assert_eq!(stats.submitted, stats.completed);
}
