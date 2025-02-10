//! Merge scheduling constants
//!
//! Centralized tuning knobs for background and forced segment merging.

/// Maximum number of concurrent background merge operations.
/// Higher values reduce segment count faster but increase I/O and CPU pressure.
pub const MAX_CONCURRENT_MERGES: usize = 4;

/// Maximum segments to merge in a single force_merge pass.
/// Keeps file descriptor count and memory usage bounded when merging
/// hundreds of segments. force_merge iterates in batches of this size
/// until only one segment remains.
pub const FORCE_MERGE_BATCH: usize = 64;
