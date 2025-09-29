//! Idle storage only: in-flight admission and returned-byte budgets are separate.
use super::DEPTH;
use parking_lot::Mutex;
use std::{
    collections::VecDeque,
    io,
    sync::{Arc, Weak},
};

pub(in crate::directories) struct RecycledBytes {
    bytes: Vec<u8>,
    pool: Weak<BufferPool>,
}
// Views remain immutable across unwinding; the mutex is touched only on final drop.
impl std::panic::RefUnwindSafe for RecycledBytes {}
impl std::panic::UnwindSafe for RecycledBytes {}

impl RecycledBytes {
    pub(in crate::directories) fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}
impl Drop for RecycledBytes {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            pool.put(std::mem::take(&mut self.bytes));
        }
    }
}

#[derive(Default)]
struct Idle {
    buffers: VecDeque<Vec<u8>>,
    bytes: usize,
    hits: u64,
    misses: u64,
    closed: bool,
}
pub(super) struct BufferPool {
    budget: usize,
    idle: Mutex<Idle>,
}
impl BufferPool {
    pub(super) fn new(budget: usize) -> Arc<Self> {
        Arc::new(Self {
            budget,
            idle: Mutex::new(Idle::default()),
        })
    }
    /// Charge the backing capacity, not merely the visible range length.
    pub(super) fn allocation_len(&self, len: usize) -> usize {
        let class = len.next_power_of_two();
        if class <= self.budget { class } else { len }
    }
    pub(super) fn take(&self, len: usize) -> io::Result<Vec<u8>> {
        if self.budget != 0 {
            let mut idle = self.idle.lock();
            if let Some(index) = idle.buffers.iter().position(|b| b.capacity() == len) {
                let mut buffer = idle.buffers.remove(index).unwrap();
                idle.bytes -= buffer.capacity();
                idle.hits += 1;
                buffer.resize(len, 0);
                return Ok(buffer);
            }
            idle.misses += 1;
        }
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(len).map_err(io::Error::other)?;
        buffer.resize(len, 0);
        Ok(buffer)
    }
    pub(super) fn own(self: &Arc<Self>, mut bytes: Vec<u8>, len: usize) -> super::OwnedBytes {
        if self.budget == 0 {
            bytes.truncate(len);
            return super::OwnedBytes::new(bytes);
        }
        super::OwnedBytes::from_recycled(
            Arc::new(RecycledBytes {
                bytes,
                pool: Arc::downgrade(self),
            }),
            len,
        )
    }
    fn put(&self, bytes: Vec<u8>) {
        let mut idle = self.idle.lock();
        if idle.closed || bytes.capacity() > self.budget {
            return;
        }
        while idle.buffers.len() >= DEPTH || bytes.capacity() > self.budget - idle.bytes {
            let evicted = idle.buffers.pop_front().unwrap();
            idle.bytes -= evicted.capacity();
        }
        idle.bytes += bytes.capacity();
        idle.buffers.push_back(bytes);
    }
    pub(super) fn close(&self) {
        let mut idle = self.idle.lock();
        idle.closed = true;
        idle.buffers.clear();
        idle.bytes = 0;
    }
    pub(super) fn snapshot(&self) -> (usize, u64, u64) {
        let idle = self.idle.lock();
        (idle.bytes, idle.hits, idle.misses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_classes_reuse_initialized_storage_but_expose_only_requested_bytes() {
        let pool = BufferPool::new(64);
        assert_eq!(pool.allocation_len(17), 32);
        let mut buffer = pool.take(pool.allocation_len(17)).unwrap();
        buffer.fill(42);
        let bytes = pool.own(buffer, 17);
        assert_eq!(bytes.as_slice(), &[42; 17]);
        drop(bytes);
        let mut buffer = pool.take(pool.allocation_len(21)).unwrap();
        assert_eq!(pool.snapshot().1, 1);
        buffer[..21].fill(7);
        assert_eq!(pool.own(buffer, 21).as_slice(), &[7; 21]);
        assert_eq!(
            pool.allocation_len(65),
            65,
            "oversized classes bypass reuse"
        );
        assert_eq!(BufferPool::new(0).allocation_len(17), 17);
    }

    #[test]
    fn startup_sized_idle_buffers_do_not_prevent_payload_buffer_reuse() {
        let pool = BufferPool::new(64);
        drop(pool.own(pool.take(64).unwrap(), 64));
        assert_eq!(pool.snapshot().0, 64);
        drop(pool.own(pool.take(8).unwrap(), 8));
        let _reused = pool.take(8).unwrap();
        assert_eq!(
            pool.snapshot().1,
            1,
            "a stale large allocation must make room for recently returned payload storage"
        );
        assert!(pool.snapshot().0 <= 64);
    }

    #[test]
    fn final_subview_returns_storage_without_recycling_live_bytes() {
        let pool = BufferPool::new(64);
        let mut data = pool.take(32).unwrap();
        data.fill(42);
        let bytes = pool.own(data, 32);
        let view = bytes.slice(3..9);
        let empty = bytes.slice(0..0);
        drop(bytes);
        assert_eq!(pool.snapshot().0, 0);
        assert!(!view.is_mmap());
        assert_eq!(
            std::thread::spawn(move || view.to_vec()).join().unwrap(),
            vec![42; 6]
        );
        assert_eq!(pool.snapshot().0, 0);
        drop(empty);
        assert_eq!(pool.snapshot().0, 32);
        let reused = pool.take(32).unwrap();
        assert_eq!(reused, vec![42; 32]);
        assert_eq!(pool.snapshot(), (0, 1, 1));
        let bytes = pool.own(reused, 32);
        pool.close();
        drop(bytes);
        assert_eq!(pool.snapshot().0, 0);
    }

    #[test]
    fn idle_capacity_and_count_are_bounded_and_views_outlive_pool() {
        let pool = BufferPool::new(64);
        let held: Vec<_> = (0..32)
            .map(|_| pool.own(pool.take(8).unwrap(), 8))
            .collect();
        drop(held);
        assert_eq!(pool.snapshot().0, 64);
        assert_eq!(pool.idle.lock().buffers.len(), DEPTH);
        let bytes = pool.own(pool.take(128).unwrap(), 128);
        drop(pool);
        assert_eq!(bytes.len(), 128);
        assert_eq!(bytes.as_slice(), &[0; 128]);
    }
}
