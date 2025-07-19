//! Segment types and metadata

use std::io::{self, Cursor};
#[cfg(feature = "native")]
use std::path::Path;
use std::path::PathBuf;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rustc_hash::FxHashMap;

use std::sync::Arc;

use crate::dsl::Field;

/// Trained vector index structures for rebuilding segments with ANN indexes
///
/// Defined here (not in merger) so it's available on all platforms including WASM.
#[derive(Clone, Default)]
pub struct TrainedVectorStructures {
    /// Must be declared before artifact `Arc`s so locks are released while all
    /// referenced allocations are still alive (Rust drops fields in order).
    #[cfg(feature = "native")]
    pub(crate) _ann_pins: Arc<crate::segment::pin::HeapPinSet>,
    /// Trained centroids per field_id
    pub centroids: FxHashMap<u32, Arc<crate::structures::CoarseCentroids>>,
    /// Global Hamming coarse quantizers per binary dense field.
    pub binary_quantizers: FxHashMap<u32, Arc<crate::structures::BinaryCoarseQuantizer>>,
    /// Index-global PQ codebooks per field ID.
    pub codebooks: FxHashMap<u32, Arc<crate::structures::PQCodebook>>,
}

impl TrainedVectorStructures {
    /// Lock the immutable index-global structures touched by ANN routing once
    /// per artifact generation. Segment payloads (PQ/exact codes, doc IDs) and
    /// raw rerank vectors are deliberately excluded: those scale with the
    /// corpus and are data, not lightweight routing metadata.
    #[cfg(feature = "native")]
    pub(crate) fn pin_ann_structures(&mut self, policy: &crate::segment::pin::PinPolicy) {
        if !policy.is_enabled() {
            return;
        }
        let mut pins = crate::segment::pin::HeapPinSet::default();
        let mut remaining = policy.budget_bytes;

        let mut float_fields: Vec<_> = self.centroids.keys().copied().collect();
        float_fields.sort_unstable();
        let mut binary_fields: Vec<_> = self.binary_quantizers.keys().copied().collect();
        binary_fields.sort_unstable();
        let mut codebook_fields: Vec<_> = self.codebooks.keys().copied().collect();
        codebook_fields.sort_unstable();

        // Priority 1: graph/topology and two-level parent data for every field.
        for field_id in &float_fields {
            pins.retain_owner(Arc::clone(&self.centroids[field_id]));
            self.centroids[field_id].visit_routing_regions(&mut |label, bytes| {
                pins.pin_slice(
                    bytes,
                    &format!("field {field_id} {label}"),
                    policy.mode,
                    &mut remaining,
                );
            });
        }
        for field_id in &binary_fields {
            pins.retain_owner(Arc::clone(&self.binary_quantizers[field_id]));
            self.binary_quantizers[field_id].visit_routing_regions(&mut |label, bytes| {
                pins.pin_slice(
                    bytes,
                    &format!("field {field_id} {label}"),
                    policy.mode,
                    &mut remaining,
                );
            });
        }

        // Priority 2: PQ/OPQ tables used for every float query plan.
        for field_id in &codebook_fields {
            pins.retain_owner(Arc::clone(&self.codebooks[field_id]));
            self.codebooks[field_id].visit_resident_regions(&mut |label, bytes| {
                pins.pin_slice(
                    bytes,
                    &format!("field {field_id} {label}"),
                    policy.mode,
                    &mut remaining,
                );
            });
        }

        // Priority 3: leaf centroids. These can be much larger at billion
        // scale, so they consume the budget only after all control structures.
        for field_id in &float_fields {
            self.centroids[field_id].visit_leaf_centroid_region(&mut |label, bytes| {
                pins.pin_slice(
                    bytes,
                    &format!("field {field_id} {label}"),
                    policy.mode,
                    &mut remaining,
                );
            });
        }
        for field_id in &binary_fields {
            self.binary_quantizers[field_id].visit_leaf_centroid_region(&mut |label, bytes| {
                pins.pin_slice(
                    bytes,
                    &format!("field {field_id} {label}"),
                    policy.mode,
                    &mut remaining,
                );
            });
        }

        let report = pins.report();
        if report.skipped_budget_bytes > 0 || report.failed_bytes > 0 {
            log::warn!(
                "[pin] ANN generation: resident {}/{} bytes ({:?}); budget skipped {}, mlock failed {}",
                report.pinned_bytes,
                report.intended_bytes,
                policy.mode,
                report.skipped_budget_bytes,
                report.failed_bytes,
            );
        } else if report.pinned_bytes > 0 {
            log::info!(
                "[pin] ANN generation: pinned {} bytes of routing structures ({:?})",
                report.pinned_bytes,
                policy.mode,
            );
        }
        self._ann_pins = Arc::new(pins);
    }

    #[cfg(feature = "native")]
    #[cfg(test)]
    pub(crate) fn ann_pin_report(&self) -> crate::segment::pin::PinReport {
        self._ann_pins.report()
    }
}

/// Unique segment identifier (UUID7-like: 48-bit timestamp + 80-bit random)
///
/// Stored as u128 internally for full 128-bit support.
/// Format: [48-bit timestamp ms][80-bit random]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentId(pub u128);

impl SegmentId {
    pub fn new() -> Self {
        // UUID7-like: 48 bits timestamp (ms) + 80 bits random
        #[cfg(not(target_arch = "wasm32"))]
        let timestamp_ms = {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        };
        #[cfg(target_arch = "wasm32")]
        let timestamp_ms = (instant::now() * 1000.0) as u128;

        let random_bits: u128 =
            ((rand::random::<u64>() as u128) << 16) | (rand::random::<u16>() as u128);

        // Combine: timestamp in upper 48 bits, random in lower 80 bits
        Self((timestamp_ms << 80) | random_bits)
    }

    pub fn from_u128(id: u128) -> Self {
        Self(id)
    }

    /// For backwards compatibility with u64-based IDs
    pub fn from_u64(id: u64) -> Self {
        Self(id as u128)
    }

    /// Create from hex string (32 chars)
    pub fn from_hex(s: &str) -> Option<Self> {
        u128::from_str_radix(s, 16).ok().map(Self)
    }

    /// Convert to hex string (32 chars, zero-padded)
    pub fn to_hex(&self) -> String {
        format!("{:032x}", self.0)
    }
}

impl Default for SegmentId {
    fn default() -> Self {
        Self::new()
    }
}

/// Field statistics for BM25F scoring
#[derive(Debug, Clone, Default)]
pub struct FieldStats {
    /// Total number of tokens across all documents for this field
    pub total_tokens: u64,
    /// Number of documents that have this field
    pub doc_count: u32,
}

impl FieldStats {
    /// Average field length (tokens per document)
    pub fn avg_field_len(&self) -> f32 {
        if self.doc_count == 0 {
            0.0
        } else {
            self.total_tokens as f32 / self.doc_count as f32
        }
    }
}

/// Segment metadata
#[derive(Debug, Clone)]
pub struct SegmentMeta {
    pub id: u128,
    pub num_docs: u32,
    /// Per-field statistics for BM25F scoring (field_id -> stats)
    pub field_stats: FxHashMap<u32, FieldStats>,
}

impl SegmentMeta {
    pub fn serialize(&self) -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        buf.write_u128::<LittleEndian>(self.id)?;
        buf.write_u32::<LittleEndian>(self.num_docs)?;

        // Write field stats
        buf.write_u32::<LittleEndian>(self.field_stats.len() as u32)?;
        for (&field_id, stats) in &self.field_stats {
            buf.write_u32::<LittleEndian>(field_id)?;
            buf.write_u64::<LittleEndian>(stats.total_tokens)?;
            buf.write_u32::<LittleEndian>(stats.doc_count)?;
        }

        Ok(buf)
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut reader = Cursor::new(data);
        let id = reader.read_u128::<LittleEndian>()?;
        let num_docs = reader.read_u32::<LittleEndian>()?;

        // Read field stats (handle legacy format without field stats)
        let mut field_stats = FxHashMap::default();
        if reader.position() < data.len() as u64 {
            let num_fields = reader.read_u32::<LittleEndian>()?;
            for _ in 0..num_fields {
                let field_id = reader.read_u32::<LittleEndian>()?;
                let total_tokens = reader.read_u64::<LittleEndian>()?;
                let doc_count = reader.read_u32::<LittleEndian>()?;
                field_stats.insert(
                    field_id,
                    FieldStats {
                        total_tokens,
                        doc_count,
                    },
                );
            }
        }

        Ok(Self {
            id,
            num_docs,
            field_stats,
        })
    }

    /// Get average field length for a field
    pub fn avg_field_len(&self, field: Field) -> f32 {
        self.field_stats
            .get(&field.0)
            .map(|s| s.avg_field_len())
            .unwrap_or(0.0)
    }
}

/// Paths for segment files
pub struct SegmentFiles {
    pub term_dict: PathBuf,
    pub postings: PathBuf,
    pub store: PathBuf,
    pub meta: PathBuf,
    /// Dense vector indexes (all fields in one file)
    pub vectors: PathBuf,
    /// Sparse vector posting lists (per field, per dimension)
    pub sparse: PathBuf,
    /// Token positions for phrase queries (fields with record_positions=true)
    pub positions: PathBuf,
    /// Fast-field columnar storage for O(1) doc→value access
    pub fast: PathBuf,
}

impl SegmentFiles {
    pub fn new(segment_id: u128) -> Self {
        let prefix = format!("seg_{:032x}", segment_id);
        Self {
            term_dict: PathBuf::from(format!("{}.terms", prefix)),
            postings: PathBuf::from(format!("{}.post", prefix)),
            store: PathBuf::from(format!("{}.store", prefix)),
            meta: PathBuf::from(format!("{}.meta", prefix)),
            vectors: PathBuf::from(format!("{}.vectors", prefix)),
            sparse: PathBuf::from(format!("{}.sparse", prefix)),
            positions: PathBuf::from(format!("{}.pos", prefix)),
            fast: PathBuf::from(format!("{}.fast", prefix)),
        }
    }

    /// Files every readable segment must contain.
    #[cfg(feature = "native")]
    pub(crate) fn mandatory_paths(&self) -> [&Path; 4] {
        [
            self.meta.as_path(),
            self.term_dict.as_path(),
            self.postings.as_path(),
            self.store.as_path(),
        ]
    }

    /// Temporary sparse skip table used while streaming large merges.
    ///
    /// Readers never open it, but abort and orphan cleanup must treat it as a
    /// segment-owned artifact.
    #[cfg(feature = "native")]
    pub(crate) fn sparse_skip_temp(&self) -> PathBuf {
        self.sparse.with_extension("skip.tmp")
    }

    /// Every permanent or temporary path owned by this segment ID.
    #[cfg(feature = "native")]
    pub(crate) fn lifecycle_paths(&self) -> [PathBuf; 9] {
        [
            self.term_dict.clone(),
            self.postings.clone(),
            self.store.clone(),
            self.meta.clone(),
            self.vectors.clone(),
            self.sparse.clone(),
            self.sparse_skip_temp(),
            self.positions.clone(),
            self.fast.clone(),
        ]
    }
}

#[cfg(all(test, feature = "native"))]
mod ann_pin_tests {
    use super::*;
    use crate::dsl::IvfRoutingMode;
    use crate::segment::pin::{PinMode, PinPolicy};
    use crate::structures::{CoarseCentroids, CoarseConfig};

    fn trained_float_artifacts() -> TrainedVectorStructures {
        let vectors = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let centroids = CoarseCentroids::train(
            &CoarseConfig::new(2, 2).with_routing(IvfRoutingMode::Flat),
            &vectors,
        );
        let mut trained = TrainedVectorStructures::default();
        trained.centroids.insert(7, Arc::new(centroids));
        trained
    }

    #[test]
    fn ann_heap_pinning_accounts_for_global_routing_arrays() {
        let mut trained = trained_float_artifacts();
        trained.pin_ann_structures(&PinPolicy {
            budget_bytes: 1 << 20,
            mode: PinMode::Copy,
        });
        let report = trained.ann_pin_report();
        assert!(report.intended_bytes > 0);
        assert_eq!(report.pinned_bytes, report.intended_bytes);
        assert_eq!(report.skipped_budget_bytes, 0);
        assert_eq!(report.failed_bytes, 0);
    }

    #[test]
    fn ann_heap_pinning_honors_generation_budget() {
        let mut trained = trained_float_artifacts();
        trained.pin_ann_structures(&PinPolicy {
            budget_bytes: 1,
            mode: PinMode::Copy,
        });
        let report = trained.ann_pin_report();
        assert!(report.intended_bytes > 0);
        assert_eq!(report.pinned_bytes, 0);
        assert_eq!(report.skipped_budget_bytes, report.intended_bytes);
    }
}
