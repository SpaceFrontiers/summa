// Copyright (C) 2022 Quickwit, Inc.
//
// Quickwit is offered under the AGPL v3.0 and as commercial software.
// For commercial licensing, contact us at hello@quickwit.io.
//
// AGPL:
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

use std::borrow::Borrow;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::Duration;

use lru::{KeyRef, LruCache};
use tantivy::directory::OwnedBytes;

use super::slice_address::{SliceAddress, SliceAddressKey, SliceAddressRef};
use super::stored_item::StoredItem;
use crate::metrics::CacheMetrics;

const MIN_TIME_SINCE_LAST_ACCESS: Duration = Duration::from_secs(300);

#[derive(Clone, Copy, Debug)]
enum Capacity {
    Unlimited,
    InBytes(usize),
}

impl Capacity {
    fn exceeds_capacity(&self, num_bytes: usize) -> bool {
        match *self {
            Capacity::Unlimited => false,
            Capacity::InBytes(capacity_in_bytes) => num_bytes > capacity_in_bytes,
        }
    }
}

struct NeedMutMemorySizedCache<K: Hash + Eq> {
    lru_cache: LruCache<K, StoredItem>,
    num_items: usize,
    num_bytes: u64,
    capacity: Capacity,
    cache_metrics: CacheMetrics,
}

impl<K: Hash + Eq> Drop for NeedMutMemorySizedCache<K> {
    fn drop(&mut self) {
        self.cache_metrics.in_cache_count.fetch_sub(self.num_items as i64, Ordering::Relaxed);
        self.cache_metrics.in_cache_num_bytes.fetch_sub(self.num_bytes as i64, Ordering::Relaxed);
    }
}

impl<K: Hash + Eq> NeedMutMemorySizedCache<K> {
    /// Creates a new NeedMutSliceCache with the given capacity.
    fn with_capacity(capacity: Capacity, cache_metrics: CacheMetrics) -> Self {
        NeedMutMemorySizedCache {
            // The limit will be decided by the amount of memory in the cache,
            // not the number of items in the cache.
            // Enforcing this limit is done in the `NeedMutCache` impl.
            lru_cache: LruCache::unbounded(),
            num_items: 0,
            num_bytes: 0,
            capacity,
            cache_metrics,
        }
    }

    pub fn record_item(&mut self, num_bytes: u64) {
        self.num_items += 1;
        self.num_bytes += num_bytes;
        self.cache_metrics.requests_count.fetch_add(1, Ordering::Relaxed);
        self.cache_metrics.requests_bytes.fetch_add(num_bytes, Ordering::Relaxed);
        self.cache_metrics.in_cache_count.fetch_add(1, Ordering::Relaxed);
        self.cache_metrics.in_cache_num_bytes.fetch_add(num_bytes as i64, Ordering::Relaxed);
    }

    pub fn drop_item(&mut self, num_bytes: u64) {
        self.num_items -= 1;
        self.num_bytes -= num_bytes;
        self.cache_metrics.in_cache_count.fetch_sub(1, Ordering::Relaxed);
        self.cache_metrics.in_cache_num_bytes.fetch_sub(num_bytes as i64, Ordering::Relaxed);
    }

    pub fn get<Q>(&mut self, cache_key: &Q) -> Option<OwnedBytes>
    where
        KeyRef<K>: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let item_opt = self.lru_cache.get_mut(cache_key);
        if let Some(item) = item_opt {
            self.cache_metrics.hits_num_items.fetch_add(1, Ordering::Relaxed);
            self.cache_metrics.hits_num_bytes.fetch_add(item.len() as u64, Ordering::Relaxed);
            Some(item.payload())
        } else {
            self.cache_metrics.misses_num_items.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Attempt to put the given amount of data in the cache.
    /// This may fail silently if the owned_bytes slice is larger than the cache
    /// capacity.
    fn put(&mut self, key: K, bytes: OwnedBytes) {
        if self.capacity.exceeds_capacity(bytes.len()) {
            // The value does not fit in the cache. We simply don't store it.
            return;
        }
        if let Some(previous_data) = self.lru_cache.pop(&key) {
            self.drop_item(previous_data.len() as u64);
        }

        let now = instant::Instant::now();
        while self.capacity.exceeds_capacity(self.num_bytes as usize + bytes.len()) {
            if let Some((_, candidate_for_eviction)) = self.lru_cache.peek_lru() {
                let time_since_last_access = now.duration_since(candidate_for_eviction.last_access_time());
                if time_since_last_access < MIN_TIME_SINCE_LAST_ACCESS {
                    // It is not worth doing an eviction.
                    // TODO: It is sub-optimal that we might have needlessly evicted items in this
                    // loop before just returning.
                    return;
                }
            }
            if let Some((_, bytes)) = self.lru_cache.pop_lru() {
                self.drop_item(bytes.len() as u64);
            } else {
                return;
            }
        }
        self.record_item(bytes.len() as u64);
        self.lru_cache.put(key, StoredItem::new(bytes, now));
    }
}

/// A simple in-resident memory slice cache.
pub struct MemorySizedCache<K: Hash + Eq = SliceAddress> {
    inner: Mutex<NeedMutMemorySizedCache<K>>,
}

impl<K: Hash + Eq> MemorySizedCache<K> {
    /// Creates an slice cache with the given capacity.
    pub fn with_capacity_in_bytes(capacity_in_bytes: usize, cache_metrics: CacheMetrics) -> Self {
        MemorySizedCache {
            inner: Mutex::new(NeedMutMemorySizedCache::with_capacity(Capacity::InBytes(capacity_in_bytes), cache_metrics)),
        }
    }

    /// Creates a slice cache that nevers removes any entry.
    pub fn with_infinite_capacity(cache_metrics: CacheMetrics) -> Self {
        MemorySizedCache {
            inner: Mutex::new(NeedMutMemorySizedCache::with_capacity(Capacity::Unlimited, cache_metrics)),
        }
    }

    /// If available, returns the cached view of the slice.
    pub fn get<Q>(&self, cache_key: &Q) -> Option<OwnedBytes>
    where
        KeyRef<K>: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.lock().expect("poisoned").get(cache_key)
    }

    /// Attempt to put the given amount of data in the cache.
    /// This may fail silently if the owned_bytes slice is larger than the cache
    /// capacity.
    pub fn put(&self, val: K, bytes: OwnedBytes) {
        self.inner.lock().expect("poisoned").put(val, bytes);
    }
}

impl MemorySizedCache<SliceAddress> {
    /// If available, returns the cached view of the slice.
    pub fn get_slice(&self, path: &Path, index: usize) -> Option<OwnedBytes> {
        let slice_address_ref = SliceAddressRef { path, index };
        self.get(&slice_address_ref as &dyn SliceAddressKey)
    }

    /// Attempt to put the given amount of data in the cache.
    /// This may fail silently if the owned_bytes slice is larger than the cache
    /// capacity.
    pub fn put_slice(&self, path: PathBuf, index: usize, bytes: OwnedBytes) {
        let slice_address = SliceAddress { path, index };
        self.put(slice_address, bytes);
    }
}
