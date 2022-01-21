use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

/// Hold metrics for caches used in `ChunkedCachingDirectory`
#[derive(Clone, Default)]
pub struct CacheMetrics {
    pub in_cache_count: Arc<AtomicI64>,
    pub in_cache_num_bytes: Arc<AtomicI64>,
    pub hits_num_bytes: Arc<AtomicU64>,
    pub hits_num_items: Arc<AtomicU64>,
    pub misses_num_items: Arc<AtomicU64>,
    pub requests_count: Arc<AtomicU64>,
    pub requests_bytes: Arc<AtomicU64>,
}

impl Serialize for CacheMetrics {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Color", 3)?;
        state.serialize_field("in_cache_count", &self.in_cache_count.load(Ordering::Relaxed))?;
        state.serialize_field("in_cache_num_bytes", &self.in_cache_num_bytes.load(Ordering::Relaxed))?;
        state.serialize_field("hits_num_bytes", &self.hits_num_bytes.load(Ordering::Relaxed))?;
        state.serialize_field("hits_num_items", &self.hits_num_items.load(Ordering::Relaxed))?;
        state.serialize_field("misses_num_items", &self.misses_num_items.load(Ordering::Relaxed))?;
        state.serialize_field("requests_count", &self.requests_count.load(Ordering::Relaxed))?;
        state.serialize_field("requests_bytes", &self.requests_bytes.load(Ordering::Relaxed))?;
        state.end()
    }
}
