//! Completion-driven ring worker. This module alone owns kernel buffer pointers.
use super::{Admission, Counters, DEPTH, Job, Read, finish, prepare};
use io_uring::{IoUring, opcode, types};
use std::{
    io,
    os::fd::AsRawFd,
    sync::{Arc, atomic::Ordering::Relaxed},
    thread::JoinHandle,
};

mod wake;
pub(super) use wake::Wake;

const WAKE_TOKEN: u64 = u64::MAX;

struct Reactor {
    ring: Option<IoUring>,
    slots: [Option<Read>; DEPTH],
    counters: Arc<Counters>,
    wake: Arc<Wake>,
    wake_armed: bool,
    #[cfg(test)]
    pause_for_test: Option<tests::WaitPause>,
}

/// The only submitter is this worker; SQPOLL is deliberately disabled.
fn cancel(ring: &IoUring) -> io::Result<()> {
    for _ in 0..8 {
        match ring
            .submitter()
            .register_sync_cancel(None, types::CancelBuilder::any().all())
        {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::Interrupted,
        "payload cancellation repeatedly interrupted",
    ))
}

impl Drop for Reactor {
    fn drop(&mut self) {
        let Some(ring) = self.ring.take() else {
            return;
        };
        if self.slots.iter().all(Option::is_none) {
            return;
        }
        // Exceptional teardown only. A successful synchronous cancel proves no
        // submitted read still uses its allocation. ENOENT means all completed.
        // Unsubmitted SQ entries cannot run: no SQPOLL or other submitter exists.
        let drained = cancel(&ring);
        drop(ring);
        self.release_slots(drained);
    }
}

impl Reactor {
    fn release_slots(&mut self, drained: io::Result<()>) {
        if let Err(error) = &drained {
            log::error!("payload ring drain failed; quarantining live buffers: {error}");
        }
        for slot in &mut self.slots {
            if let Some(mut read) = slot.take() {
                if drained.is_err() {
                    // Never guess that closing a ring completed its operations.
                    // Preserve the allocation and descriptor, but notify callers.
                    // Admission is closed on worker failure; retention is bounded
                    // by this service's 8 MiB and eight descriptors, not retries.
                    let buffer = std::mem::take(&mut read.buffer);
                    self.counters
                        .quarantined_bytes
                        .fetch_add(buffer.capacity() as u64, Relaxed);
                    std::mem::forget((buffer, read.job.file.clone()));
                }
                finish(
                    read,
                    Err(io::Error::other("payload ring stopped before completion")),
                    &self.counters,
                );
            }
        }
    }

    #[cfg(test)]
    fn new(counters: Arc<Counters>) -> io::Result<Self> {
        Self::with_wake(counters.clone(), Arc::new(Wake::new(counters)?))
    }
    fn with_wake(counters: Arc<Counters>, wake: Arc<Wake>) -> io::Result<Self> {
        let ring = IoUring::new((DEPTH * 2) as u32)?;
        // Refuse kernels where exceptional teardown has no proven drain path.
        cancel(&ring)?;
        let mut probe = io_uring::Probe::new();
        ring.submitter().register_probe(&mut probe)?;
        if !probe.is_supported(opcode::Read::CODE)
            || !probe.is_supported(opcode::PollAdd::CODE)
            || !ring.params().is_feature_ext_arg()
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "io_uring read, poll or extended wait unavailable",
            ));
        }
        Ok(Self {
            ring: Some(ring),
            slots: std::array::from_fn(|_| None),
            counters,
            wake,
            wake_armed: false,
            #[cfg(test)]
            pause_for_test: None,
        })
    }

    fn enqueue(&mut self, job: Job) -> io::Result<bool> {
        let Some(read) = prepare(job, &self.counters) else {
            return Ok(false);
        };
        let id = self
            .slots
            .iter()
            .position(Option::is_none)
            .ok_or_else(|| io::Error::other("payload ring admission invariant violated"))?;
        // Put ownership in the teardown guard BEFORE publishing any raw pointer.
        self.slots[id] = Some(read);
        let read = self.slots[id].as_mut().unwrap();
        let entry = opcode::Read::new(
            types::Fd(read.job.file.as_raw_fd()),
            read.buffer.as_mut_ptr(),
            read.job.len() as u32,
        )
        .offset(read.job.range.start)
        .build()
        .user_data(id as u64);
        // SAFETY: stable Vec allocation and open FD remain in this slot until its
        // original CQE, including receiver cancellation. Reactor::drop handles
        // errors/unwinding before slots can be freed. There is one submitter.
        unsafe { self.ring.as_mut().unwrap().submission().push(&entry) }
            .map_err(|_| io::Error::other("payload submission queue unexpectedly full"))?;
        Ok(true)
    }

    fn wait(&mut self) -> io::Result<()> {
        self.wake.check()?;
        #[cfg(test)]
        if self
            .pause_for_test
            .as_ref()
            .is_some_and(|pause| pause.at == tests::WaitAt::BeforePoll)
        {
            self.pause_for_test.take().unwrap().wait();
        }
        let ring = self.ring.as_mut().unwrap();
        if !self.wake_armed {
            let entry = opcode::PollAdd::new(types::Fd(self.wake.as_raw_fd()), libc::POLLIN as u32)
                .build()
                .user_data(WAKE_TOKEN);
            // SAFETY: eventfd owner lives through ring teardown; PollAdd holds
            // no userspace buffer. One control SQE plus at most DEPTH read SQEs.
            unsafe { ring.submission().push(&entry) }
                .map_err(|_| io::Error::other("wake submission queue full"))?;
            self.wake_armed = true;
        }
        #[cfg(test)]
        if let Some(pause) = self.pause_for_test.take() {
            ring.submit()?;
            pause.wait();
        }
        // A broken notifier must not strand accepted jobs behind unrelated I/O.
        // Expiry only checks worker health; it does not cancel healthy reads.
        let timeout = types::Timespec::new().nsec(100_000_000);
        let args = types::SubmitArgs::new().timespec(&timeout);
        for _ in 0..8 {
            match ring.submitter().submit_with_args(1, &args) {
                Ok(_) => return Ok(()),
                Err(error) if error.raw_os_error() == Some(libc::ETIME) => return Ok(()),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "payload submission repeatedly interrupted",
        ))
    }

    fn complete(&mut self) -> io::Result<usize> {
        let mut count = 0;
        for cqe in self.ring.as_mut().unwrap().completion() {
            if cqe.user_data() == WAKE_TOKEN {
                if !self.wake_armed {
                    return Err(io::Error::other("unexpected queue notification completion"));
                }
                self.wake_armed = false;
                if cqe.result() < 0 {
                    return Err(io::Error::from_raw_os_error(-cqe.result()));
                }
                if cqe.result() & i32::from(libc::POLLIN) == 0
                    || cqe.result() & i32::from(libc::POLLERR | libc::POLLHUP) != 0
                {
                    return Err(io::Error::other("invalid queue notification readiness"));
                }
                self.wake.drain()?;
                self.counters.woke();
                continue;
            }
            let id = usize::try_from(cqe.user_data()).map_err(io::Error::other)?;
            let read = self
                .slots
                .get_mut(id)
                .and_then(Option::take)
                .ok_or_else(|| io::Error::other("unexpected payload completion token"))?;
            let result = if cqe.result() == read.job.len() as i32 {
                Ok(())
            } else if cqe.result() < 0 {
                Err(io::Error::from_raw_os_error(-cqe.result()))
            } else {
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short payload read",
                ))
            };
            finish(read, result, &self.counters);
            count += 1;
        }
        Ok(count)
    }

    fn run(mut self, receiver: async_channel::Receiver<Job>) -> io::Result<()> {
        let mut outstanding = 0;
        loop {
            self.wake.check()?;
            let mut added = 0;
            if outstanding == 0 {
                #[cfg(test)]
                if self
                    .pause_for_test
                    .as_ref()
                    .is_some_and(|pause| pause.at == tests::WaitAt::Idle)
                {
                    self.pause_for_test.take().unwrap().wait();
                }
                match receiver.recv_blocking() {
                    Ok(job) => added += usize::from(self.enqueue(job)?),
                    Err(_) => return self.wake.check(),
                }
            }
            while outstanding + added < DEPTH {
                let Ok(job) = receiver.try_recv() else {
                    break;
                };
                added += usize::from(self.enqueue(job)?);
            }
            if added != 0 {
                self.counters.batch(added, outstanding != 0);
            }
            outstanding += added;
            if outstanding != 0 {
                self.wait()?;
                self.wake.check()?;
                outstanding -= self.complete()?;
            }
        }
    }
}

pub(super) fn start(
    receiver: async_channel::Receiver<Job>,
    counters: Arc<Counters>,
    admission: Arc<Admission>,
    wake: Arc<Wake>,
) -> io::Result<Vec<JoinHandle<io::Result<()>>>> {
    let reactor = Reactor::with_wake(counters.clone(), wake)?;
    let reads = receiver.clone();
    Ok(vec![super::spawn_worker(
        "summa-payload-uring",
        receiver,
        counters,
        admission,
        move || reactor.run(reads),
    )?])
}

#[cfg(test)]
mod tests;
