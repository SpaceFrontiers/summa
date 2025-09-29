//! Level-triggered queue notification. The reactor retains the descriptor owner.
use super::super::Counters;
use std::{
    io,
    os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub(in super::super) struct Wake {
    fd: OwnedFd,
    failed: AtomicBool,
    counters: Arc<Counters>,
    #[cfg(test)]
    fail_next: AtomicBool,
}

// Both eventfd operations transfer exactly one u64. EAGAIN means the counter is
// already readable (write) or drained (read), not a lost notification.
fn transfer(mut operation: impl FnMut() -> io::Result<usize>) -> io::Result<()> {
    for _ in 0..8 {
        match operation() {
            Ok(8) => return Ok(()),
            Ok(_) => return Err(io::Error::other("short eventfd transfer")),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::Interrupted,
        "eventfd repeatedly interrupted",
    ))
}

fn transferred(count: isize) -> io::Result<usize> {
    if count < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(count as usize)
    }
}

impl Wake {
    pub(in super::super) fn new(counters: Arc<Counters>) -> io::Result<Self> {
        // SAFETY: eventfd takes values, not pointers, and returns a fresh FD.
        let fd = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            // SAFETY: successful eventfd transfers exactly one descriptor owner.
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
            failed: AtomicBool::new(false),
            counters,
            #[cfg(test)]
            fail_next: AtomicBool::new(false),
        })
    }

    pub(in super::super) fn check(&self) -> io::Result<()> {
        if self.failed.load(Ordering::Acquire) {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "payload queue notification failed",
            ))
        } else {
            Ok(())
        }
    }

    pub(in super::super) fn notify(&self) -> io::Result<()> {
        self.check()?;
        let result = transfer(|| {
            #[cfg(test)]
            if self.fail_next.swap(false, Ordering::Relaxed) {
                return Err(io::Error::other("injected eventfd write failure"));
            }
            let count = 1u64;
            // SAFETY: one live u64 and a valid owned nonblocking descriptor.
            transferred(unsafe { libc::write(self.as_raw_fd(), (&count as *const u64).cast(), 8) })
        });
        if result.is_err() && !self.failed.swap(true, Ordering::AcqRel) {
            self.counters.notification_failed();
        }
        result
    }

    pub(super) fn drain(&self) -> io::Result<()> {
        transfer(|| {
            let mut count = 0u64;
            // SAFETY: one writable u64, valid descriptor, no pointer escapes.
            transferred(unsafe { libc::read(self.as_raw_fd(), (&mut count as *mut u64).cast(), 8) })
        })
    }

    #[cfg(test)]
    pub(super) fn fail_next_notification(&self) {
        self.fail_next.store(true, Ordering::Relaxed);
    }
}

impl AsRawFd for Wake {
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupted_notifications_are_bounded_and_short_transfers_are_errors() {
        let mut attempts = 0;
        let result = transfer(|| {
            attempts += 1;
            Err(io::Error::from(io::ErrorKind::Interrupted))
        });
        assert_eq!(attempts, 8);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::Interrupted);
        assert!(transfer(|| Ok(0)).is_err());
        let mut attempts = 0;
        transfer(|| {
            attempts += 1;
            if attempts < 3 {
                Err(io::Error::from(io::ErrorKind::Interrupted))
            } else {
                Ok(8)
            }
        })
        .unwrap();
        assert_eq!(attempts, 3);
    }

    #[test]
    fn saturated_notification_counter_remains_readable_and_reusable() {
        let wake = Wake::new(Arc::new(Counters::default())).unwrap();
        let count = u64::MAX - 1;
        // SAFETY: one live u64 and an owned eventfd; this saturates its counter.
        assert_eq!(
            unsafe { libc::write(wake.as_raw_fd(), (&count as *const u64).cast(), 8) },
            8
        );
        wake.notify().unwrap(); // EAGAIN is an already-pending wakeup.
        wake.drain().unwrap();
        wake.drain().unwrap(); // Empty reads never block.
        wake.notify().unwrap();
        wake.drain().unwrap();
        assert_eq!(wake.counters.snapshot().notification_failures, 0);
    }
}
