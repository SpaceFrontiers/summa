//! Process-wide byte-bounded cache for decoded immutable blocks.
//!
//! Shared by document-store blocks and term-dictionary blocks. A per-reader
//! count cap multiplies by segment fan-out and by overlapping reader
//! generations; this cache has one hard byte ceiling across every reader
//! opened with the same policy. See `docs/term-dictionary-cache.md`.
use lru::LruCache;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Heap bytes a cached value keeps alive.
pub(crate) trait RetainedBytes {
    fn retained_bytes(&self) -> usize;
}

impl RetainedBytes for [u8] {
    fn retained_bytes(&self) -> usize {
        self.len()
    }
}

/// One segment's blocks. `directory` separates independent copies of an index
/// whose segment ids collide; readers of the same segment share entries.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BlockCacheNamespace {
    pub(crate) directory: usize,
    pub(crate) segment: u128,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BlockCacheKey {
    pub(crate) namespace: BlockCacheNamespace,
    /// Owner-defined block identity (first document id, block offset, ...).
    pub(crate) block: u64,
}

struct State<V: ?Sized> {
    blocks: LruCache<BlockCacheKey, Arc<V>>,
    retained_bytes: usize,
    namespace_bytes: FxHashMap<BlockCacheNamespace, usize>,
    namespace_readers: FxHashMap<BlockCacheNamespace, usize>,
}

impl<V: ?Sized> State<V> {
    fn charge(&mut self, namespace: BlockCacheNamespace, bytes: usize) {
        self.retained_bytes = self.retained_bytes.saturating_add(bytes);
        let namespace_bytes = self.namespace_bytes.entry(namespace).or_default();
        *namespace_bytes = namespace_bytes.saturating_add(bytes);
    }

    fn discharge(&mut self, namespace: BlockCacheNamespace, bytes: usize) {
        self.retained_bytes = self.retained_bytes.saturating_sub(bytes);
        if let Some(namespace_bytes) = self.namespace_bytes.get_mut(&namespace) {
            *namespace_bytes = namespace_bytes.saturating_sub(bytes);
            if *namespace_bytes == 0 {
                self.namespace_bytes.remove(&namespace);
            }
        }
    }
}

/// Hits take a shared read lock; eviction is insertion-ordered rather than
/// serializing every hit solely for exact LRU promotion.
pub(crate) struct SharedBlockCache<V: ?Sized> {
    state: RwLock<State<V>>,
    max_bytes: usize,
    /// Very large blocks have almost no spatial reuse and can evict thousands
    /// of ordinary blocks; they bypass retention.
    max_entry_bytes: usize,
}

impl<V: ?Sized> std::fmt::Debug for SharedBlockCache<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SharedBlockCache")
            .field("max_bytes", &self.max_bytes)
            .field("max_entry_bytes", &self.max_entry_bytes)
            .field("retained_bytes", &self.state.read().retained_bytes)
            .finish()
    }
}

impl<V: RetainedBytes + ?Sized> SharedBlockCache<V> {
    const MAX_ADMITTED_ENTRY_BYTES: usize = 8 * 1024 * 1024;

    pub(crate) fn new(max_bytes: usize) -> Self {
        Self::with_limits(max_bytes, max_bytes.min(Self::MAX_ADMITTED_ENTRY_BYTES))
    }

    pub(crate) fn with_limits(max_bytes: usize, max_entry_bytes: usize) -> Self {
        Self {
            state: RwLock::new(State {
                blocks: LruCache::unbounded(),
                retained_bytes: 0,
                namespace_bytes: FxHashMap::default(),
                namespace_readers: FxHashMap::default(),
            }),
            max_bytes,
            max_entry_bytes: max_entry_bytes.min(max_bytes),
        }
    }

    #[cfg(test)]
    pub(crate) fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Whether a block of `bytes` would be retained by [`Self::insert`].
    pub(crate) fn admits(&self, bytes: usize) -> bool {
        self.max_bytes != 0 && bytes != 0 && bytes <= self.max_entry_bytes
    }

    pub(crate) fn register(&self, namespace: BlockCacheNamespace) {
        if self.max_bytes == 0 {
            return;
        }
        let mut state = self.state.write();
        *state.namespace_readers.entry(namespace).or_default() += 1;
    }

    pub(crate) fn unregister(&self, namespace: BlockCacheNamespace) {
        if self.max_bytes == 0 {
            return;
        }
        let mut state = self.state.write();
        let Some(readers) = state.namespace_readers.get_mut(&namespace) else {
            return;
        };
        *readers -= 1;
        if *readers > 0 {
            return;
        }
        state.namespace_readers.remove(&namespace);

        // A merged-away segment will never hit these entries again. Remove
        // them immediately instead of waiting for unrelated searches to
        // create enough pressure for ordinary LRU eviction.
        let keys: Vec<_> = state
            .blocks
            .iter()
            .filter_map(|(key, _)| (key.namespace == namespace).then_some(*key))
            .collect();
        for key in keys {
            if let Some(block) = state.blocks.pop(&key) {
                state.retained_bytes = state.retained_bytes.saturating_sub(block.retained_bytes());
            }
        }
        state.namespace_bytes.remove(&namespace);
    }

    pub(crate) fn get(&self, key: BlockCacheKey) -> Option<Arc<V>> {
        // Do not serialize all process-wide hits merely to update exact LRU
        // order. Concurrent readers use a shared lock; insertion and
        // decompression-race resolution still promote entries.
        self.state.read().blocks.peek(&key).map(Arc::clone)
    }

    /// Admit one block and return the canonical cached allocation if another
    /// request won the decompression race.
    pub(crate) fn insert(&self, key: BlockCacheKey, block: Arc<V>) -> Arc<V> {
        let bytes = block.retained_bytes();
        if !self.admits(bytes) {
            return block;
        }

        let mut state = self.state.write();
        if let Some(existing) = state.blocks.get(&key) {
            return Arc::clone(existing);
        }

        state.blocks.put(key, Arc::clone(&block));
        state.charge(key.namespace, bytes);
        self.evict_over_budget(&mut state);
        block
    }

    /// Swap a cached block for an equivalent one that retains more (a block
    /// with a scan index), re-charging the difference and evicting as
    /// [`Self::insert`] does. Cached values never change their retained
    /// bytes in place, so eviction always subtracts what insertion charged.
    /// Returns `false`, changing nothing, if `key` is no longer cached or the
    /// new form is not admitted.
    pub(crate) fn replace(&self, key: BlockCacheKey, block: Arc<V>) -> bool {
        let bytes = block.retained_bytes();
        if !self.admits(bytes) {
            return false;
        }
        let mut state = self.state.write();
        // Promoted, so the larger form does not evict itself first.
        let Some(slot) = state.blocks.get_mut(&key) else {
            return false;
        };
        let previous = std::mem::replace(slot, block);
        state.discharge(key.namespace, previous.retained_bytes());
        state.charge(key.namespace, bytes);
        self.evict_over_budget(&mut state);
        true
    }

    fn evict_over_budget(&self, state: &mut State<V>) {
        while state.retained_bytes > self.max_bytes {
            let Some((evicted_key, evicted)) = state.blocks.pop_lru() else {
                state.retained_bytes = 0;
                state.namespace_bytes.clear();
                break;
            };
            state.discharge(evicted_key.namespace, evicted.retained_bytes());
        }
    }

    pub(crate) fn total_bytes(&self) -> usize {
        self.state.read().retained_bytes
    }

    pub(crate) fn total_blocks(&self) -> usize {
        self.state.read().blocks.len()
    }

    pub(crate) fn namespace_bytes(&self, namespace: BlockCacheNamespace) -> usize {
        self.state
            .read()
            .namespace_bytes
            .get(&namespace)
            .copied()
            .unwrap_or(0)
    }

    pub(crate) fn namespace_blocks(&self, namespace: BlockCacheNamespace) -> usize {
        self.state
            .read()
            .blocks
            .iter()
            .filter(|(key, _)| key.namespace == namespace)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(byte: u8, len: usize) -> Arc<[u8]> {
        vec![byte; len].into()
    }

    fn key(directory: usize, segment: u128, block: u64) -> BlockCacheKey {
        BlockCacheKey {
            namespace: BlockCacheNamespace { directory, segment },
            block,
        }
    }

    #[test]
    fn shared_block_cache_is_byte_bounded_and_read_concurrent() {
        let cache = SharedBlockCache::<[u8]>::with_limits(200, 100);
        cache.insert(key(1, 7, 1), block(1, 100));
        cache.insert(key(1, 7, 2), block(2, 100));
        assert!(cache.get(key(1, 7, 1)).is_some());
        cache.insert(key(1, 7, 3), block(3, 100));

        assert!(cache.get(key(1, 7, 1)).is_none());
        assert!(cache.get(key(1, 7, 2)).is_some());
        assert!(cache.get(key(1, 7, 3)).is_some());
        assert!(cache.total_bytes() <= 200);
    }

    #[test]
    fn shared_block_cache_bypasses_oversized_entries() {
        let cache = SharedBlockCache::<[u8]>::with_limits(1024, 99);
        assert!(!cache.admits(100));
        cache.insert(key(1, 9, 0), block(1, 100));
        assert_eq!(cache.total_bytes(), 0);
        assert!(cache.get(key(1, 9, 0)).is_none());
    }

    #[test]
    fn shared_block_cache_purges_closed_segment_namespace_after_last_reader() {
        let cache = SharedBlockCache::<[u8]>::with_limits(1024, 1024);
        let entry = key(1, 11, 0);
        cache.register(entry.namespace);
        cache.register(entry.namespace);
        cache.insert(entry, block(1, 10));
        cache.unregister(entry.namespace);
        assert_eq!(cache.namespace_bytes(entry.namespace), 10);
        cache.unregister(entry.namespace);
        assert_eq!(cache.total_bytes(), 0);
        assert!(cache.get(entry).is_none());
    }

    #[test]
    fn shared_block_cache_isolates_equal_segment_ids_across_directories() {
        let cache = SharedBlockCache::<[u8]>::with_limits(1024, 1024);
        let left = cache.insert(key(1, 42, 0), block(1, 10));
        let right = cache.insert(key(2, 42, 0), block(2, 10));

        assert!(!Arc::ptr_eq(&left, &right));
        assert_eq!(cache.total_blocks(), 2);
        assert_eq!(
            cache.namespace_blocks(BlockCacheNamespace {
                directory: 1,
                segment: 42
            }),
            1
        );
    }

    #[test]
    fn shared_block_cache_replace_recharges_exactly_and_evicts_older_blocks() {
        let cache = SharedBlockCache::<[u8]>::with_limits(100, 100);
        let namespace = key(1, 5, 0).namespace;
        cache.insert(key(1, 5, 0), block(1, 30));
        cache.insert(key(1, 5, 1), block(2, 30));
        cache.insert(key(1, 5, 2), block(3, 30));

        // A larger form of block 2 pushes out the oldest block, not itself.
        assert!(cache.replace(key(1, 5, 2), block(3, 50)));
        assert!(cache.get(key(1, 5, 0)).is_none());
        assert_eq!(cache.get(key(1, 5, 2)).unwrap().len(), 50);
        assert_eq!(cache.total_bytes(), 80);
        assert_eq!(cache.namespace_bytes(namespace), 80);

        // A block evicted in the meantime is not re-admitted.
        assert!(!cache.replace(key(1, 5, 0), block(1, 10)));
        assert!(cache.get(key(1, 5, 0)).is_none());
        assert_eq!(cache.total_bytes(), 80);

        // Eviction and namespace release subtract the replaced form's bytes.
        cache.insert(key(1, 6, 0), block(4, 20));
        assert_eq!(cache.total_bytes(), 100);
        cache.register(namespace);
        cache.unregister(namespace);
        assert_eq!(cache.total_bytes(), 20);
        assert_eq!(cache.namespace_bytes(namespace), 0);
    }

    #[test]
    fn shared_block_cache_returns_the_race_winner() {
        let cache = SharedBlockCache::<[u8]>::with_limits(1024, 1024);
        let first = cache.insert(key(1, 1, 0), block(1, 10));
        let second = cache.insert(key(1, 1, 0), block(2, 10));
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.total_bytes(), 10);
    }
}
