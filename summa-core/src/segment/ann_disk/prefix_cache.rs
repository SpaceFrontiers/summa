//! Derived native cache; persisted ANN and exact-vector representations stay unchanged.
use std::io;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{AnnDiskIndex, AnnKind, AnnRun, invalid_data, read_u16, run_doc_id_validated};
use crate::segment::LazyFlatVectorData;

pub(super) const GATHER_BYTES: usize = 8192;
const COPY_BYTES: usize = 1024 * 1024;
const YIELD_ROWS: usize = 4096;

/// One native Index's prefix-cache accounting, including retained old readers.
/// These are requested heap bytes, not OS locked pages or process RSS.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BinaryPrefixCacheStats {
    pub budget_bytes: usize,
    pub allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub completed_builds: usize,
    pub refused_builds: usize,
    pub aborted_builds: usize,
}

#[derive(Debug)]
pub(crate) struct PrefixCacheBudget {
    limit: usize,
    used: AtomicUsize,
    peak: AtomicUsize,
    completed: AtomicUsize,
    refused: AtomicUsize,
    aborted: AtomicUsize,
    build_gate: tokio::sync::Mutex<()>,
}
impl PrefixCacheBudget {
    pub(crate) fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            completed: AtomicUsize::new(0),
            refused: AtomicUsize::new(0),
            aborted: AtomicUsize::new(0),
            build_gate: tokio::sync::Mutex::new(()),
        })
    }
    pub(crate) fn enabled(&self) -> bool {
        self.limit != 0
    }
    pub(crate) fn stats(&self) -> BinaryPrefixCacheStats {
        BinaryPrefixCacheStats {
            budget_bytes: self.limit,
            allocated_bytes: self.used.load(Ordering::Acquire),
            peak_allocated_bytes: self.peak.load(Ordering::Acquire),
            completed_builds: self.completed.load(Ordering::Acquire),
            refused_builds: self.refused.load(Ordering::Acquire),
            aborted_builds: self.aborted.load(Ordering::Acquire),
        }
    }
    fn reserve(self: &Arc<Self>, bytes: usize) -> Option<Reservation> {
        if self
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|&next| next <= self.limit)
            })
            .is_err()
        {
            self.refused.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.peak
            .fetch_max(self.used.load(Ordering::Acquire), Ordering::Relaxed);
        Some(Reservation {
            budget: Arc::clone(self),
            bytes,
            published: false,
        })
    }
}
#[derive(Debug)]
struct Reservation {
    budget: Arc<PrefixCacheBudget>,
    bytes: usize,
    published: bool,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
        if !self.published {
            self.budget.aborted.fetch_add(1, Ordering::Relaxed);
        }
    }
}
#[derive(Debug)]
struct RunAddress {
    codes: usize,
    base: usize,
}
#[derive(Debug)]
pub(super) struct ResidentBinaryPrefixes {
    primary: Box<[u8]>,
    primary_docs: Box<[u32]>,
    primary_ordinals: Box<[u16]>,
    references: Box<[u32]>,
    addresses: Box<[RunAddress]>,
    // The lease outlives all byte allocations, including visibility/eligibility clones.
    reservation: Reservation,
}
impl ResidentBinaryPrefixes {
    pub(super) fn allocated_bytes(&self) -> usize {
        self.reservation.bytes
    }
    pub(super) fn base(&self, run: &AnnRun) -> usize {
        let at = self
            .addresses
            .binary_search_by_key(&run.codes.start, |a| a.codes)
            .expect("cache and ANN run directory share one immutable owner");
        self.addresses[at].base
    }
    pub(super) fn labels(&self, run: &AnnRun, base: usize, row: usize) -> (u32, u16) {
        let index = if run.prefix_only {
            self.references[base + row] as usize
        } else {
            base + row
        };
        (self.primary_docs[index], self.primary_ordinals[index])
    }
    pub(super) fn block<'a>(
        &'a self,
        run: &AnnRun,
        base: usize,
        first: usize,
        count: usize,
        prefix: usize,
        scratch: Option<&'a mut [u8; GATHER_BYTES]>,
    ) -> &'a [u8] {
        if !run.prefix_only {
            return &self.primary[(base + first) * prefix..(base + first + count) * prefix];
        }
        let scratch = scratch.expect("secondary cache scan has bounded gather scratch");
        for row in 0..count {
            let index = self.references[base + first + row] as usize;
            scratch[row * prefix..(row + 1) * prefix]
                .copy_from_slice(&self.primary[index * prefix..(index + 1) * prefix]);
        }
        &scratch[..count * prefix]
    }
}
pub(crate) enum Admission {
    Disabled,
    Existing,
    Loaded(usize),
    Refused { bytes: usize, reason: &'static str },
}

fn reserve_vec<T>(count: usize) -> io::Result<Vec<T>> {
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(|error| {
        io::Error::other(format!("binary prefix cache allocation failed: {error}"))
    })?;
    Ok(result)
}

impl AnnDiskIndex {
    pub(crate) fn resident_prefix_bytes(&self) -> usize {
        self.prefix_cache
            .get()
            .map_or(0, ResidentBinaryPrefixes::allocated_bytes)
    }
    pub(crate) async fn prepare_prefix_cache(
        &self,
        flat: &LazyFlatVectorData,
        budget: &Arc<PrefixCacheBudget>,
    ) -> io::Result<Admission> {
        if !budget.enabled()
            || self.header.kind != AnnKind::BinaryIvf
            || self.header.prefix_bytes == 0
        {
            return Ok(Admission::Disabled);
        }
        if self.prefix_cache.get().is_some() {
            return Ok(Admission::Existing);
        }
        let _gate = budget.build_gate.lock().await;
        if self.prefix_cache.get().is_some() {
            return Ok(Admission::Existing);
        }
        let prefix = self.header.prefix_bytes;
        let primary_rows = self
            .runs
            .iter()
            .filter(|r| !r.prefix_only)
            .try_fold(0usize, |n, r| n.checked_add(r.count))
            .ok_or_else(|| invalid_data("prefix cache primary count overflow"))?;
        let secondary_rows = self
            .header
            .vector_count
            .checked_sub(primary_rows)
            .ok_or_else(|| invalid_data("prefix cache secondary count underflow"))?;
        if prefix > GATHER_BYTES || primary_rows > u32::MAX as usize {
            budget.refused.fetch_add(1, Ordering::Relaxed);
            return Ok(Admission::Refused {
                bytes: 0,
                reason: "prefix width or primary reference width",
            });
        }
        let prefix_len = primary_rows
            .checked_mul(prefix)
            .ok_or_else(|| invalid_data("prefix cache size overflow"))?;
        let bytes = secondary_rows
            .checked_mul(size_of::<u32>())
            .and_then(|n| n.checked_add(prefix_len))
            .and_then(|n| {
                primary_rows
                    .checked_mul(size_of::<u32>() + size_of::<u16>())
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| {
                self.runs
                    .len()
                    .checked_mul(size_of::<RunAddress>())
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| n.checked_add(size_of::<ResidentBinaryPrefixes>()))
            .ok_or_else(|| invalid_data("prefix cache allocation size overflow"))?;
        let Some(reservation) = budget.reserve(bytes) else {
            return Ok(Admission::Refused {
                bytes,
                reason: "retained cache budget",
            });
        };
        let mut primary = reserve_vec::<u8>(prefix_len)?;
        let mut primary_docs = reserve_vec::<u32>(primary_rows)?;
        let mut primary_ordinals = reserve_vec::<u16>(primary_rows)?;
        let mut references = reserve_vec::<u32>(secondary_rows)?;
        let mut addresses = reserve_vec::<RunAddress>(self.runs.len())?;
        let raw = self.raw.as_slice();
        let mut secondary_base = 0usize;
        for run in self.runs.iter() {
            let base = if run.prefix_only {
                let base = secondary_base;
                secondary_base += run.count;
                base
            } else {
                primary.len() / prefix
            };
            addresses.push(RunAddress {
                codes: run.codes.start,
                base,
            });
            if !run.prefix_only {
                for chunk in raw[run.prefix.clone()].chunks(COPY_BYTES) {
                    primary.extend_from_slice(chunk);
                    tokio::task::yield_now().await;
                }
                for row in 0..run.count {
                    primary_docs.push(run_doc_id_validated(raw, run, row));
                    primary_ordinals.push(read_u16(raw, run.ordinals.start + row * 2));
                    if (row + 1).is_multiple_of(YIELD_ROWS) {
                        tokio::task::yield_now().await;
                    }
                }
            }
        }
        for run in self.runs.iter().filter(|r| r.prefix_only) {
            for row in 0..run.count {
                let doc = run_doc_id_validated(raw, run, row);
                let ordinal = read_u16(raw, run.ordinals.start + row * 2);
                let location = flat.exact_code_location(doc, ordinal).ok_or_else(|| {
                    invalid_data("prefix cache secondary has no primary exact location")
                })?;
                let location = usize::try_from(location)
                    .map_err(|_| invalid_data("prefix cache location overflow"))?;
                let at = self
                    .split_runs
                    .partition_point(|&i| self.runs[i as usize].codes.start <= location);
                let ri = *self
                    .split_runs
                    .get(
                        at.checked_sub(1)
                            .ok_or_else(|| invalid_data("prefix cache location precedes codes"))?,
                    )
                    .ok_or_else(|| invalid_data("prefix cache missing primary run"))?
                    as usize;
                let source = &self.runs[ri];
                let relative = location - source.codes.start;
                let source_row = relative / self.header.code_size;
                if !relative.is_multiple_of(self.header.code_size)
                    || source_row >= source.count
                    || source.prefix_only
                    || run_doc_id_validated(raw, source, source_row) != doc
                    || read_u16(raw, source.ordinals.start + source_row * 2) != ordinal
                {
                    return Err(invalid_data(
                        "prefix cache exact location has mismatched primary labels",
                    ));
                }
                let index = addresses[ri].base + source_row;
                if primary[index * prefix..(index + 1) * prefix]
                    != raw[run.prefix.start + row * prefix..run.prefix.start + (row + 1) * prefix]
                {
                    return Err(invalid_data(
                        "prefix cache secondary differs from its primary",
                    ));
                }
                references.push(
                    u32::try_from(index)
                        .map_err(|_| invalid_data("prefix cache reference overflow"))?,
                );
                if (row + 1).is_multiple_of(YIELD_ROWS) {
                    tokio::task::yield_now().await;
                }
            }
            tokio::task::yield_now().await;
        }
        if primary.len() != prefix_len
            || primary_docs.len() != primary_rows
            || primary_ordinals.len() != primary_rows
            || references.len() != secondary_rows
        {
            return Err(invalid_data("prefix cache payload length mismatch"));
        }
        addresses.sort_unstable_by_key(|a| a.codes);
        let mut payload = ResidentBinaryPrefixes {
            primary: primary.into_boxed_slice(),
            primary_docs: primary_docs.into_boxed_slice(),
            primary_ordinals: primary_ordinals.into_boxed_slice(),
            references: references.into_boxed_slice(),
            addresses: addresses.into_boxed_slice(),
            reservation,
        };
        payload.reservation.published = true;
        if self.prefix_cache.set(payload).is_err() {
            // A different caller already attached an equivalent immutable cache.
            return Ok(Admission::Existing);
        }
        budget.completed.fetch_add(1, Ordering::Relaxed);
        Ok(Admission::Loaded(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_reservations_follow_all_retained_owners_and_rollback() {
        let budget = PrefixCacheBudget::new(100);
        let first = Arc::new(budget.reserve(100).unwrap());
        let held = Arc::clone(&first);
        assert!(budget.reserve(1).is_none());
        drop(first);
        assert_eq!(budget.stats().allocated_bytes, 100);
        drop(held);
        assert_eq!(budget.stats().allocated_bytes, 0);
        assert_eq!(budget.stats().aborted_builds, 1);
        let mut next = budget.reserve(100).unwrap();
        next.published = true;
        drop(next);
        assert_eq!(budget.stats().allocated_bytes, 0);
        assert_eq!(budget.stats().peak_allocated_bytes, 100);
    }

    use crate::directories::{OwnedBytes, RamDirectory};
    use crate::dsl::{BinaryDenseVectorConfig, Document, Field, SchemaBuilder};
    use crate::index::{Index, IndexConfig, IndexWriter};
    use crate::segment::VectorIndex;
    async fn soar_fixture() -> (Index<RamDirectory>, IndexWriter<RamDirectory>) {
        let mut schema = SchemaBuilder::default();
        let field = schema.add_binary_dense_vector_field_with_config(
            "v",
            true,
            false,
            BinaryDenseVectorConfig::new(2560)
                .with_ivf(Some(4), 4)
                .with_prefix(1024, Some(8))
                .with_soar(crate::structures::SoarConfig::full()),
        );
        let index = Index::create(
            RamDirectory::new(),
            schema.build(),
            IndexConfig {
                num_threads: 1,
                num_indexing_threads: 1,
                num_compression_threads: 1,
                merge_policy: Box::new(crate::NoMergePolicy),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let mut writer = index.writer();
        for row in 0..32u64 {
            let mut state = (row + 1) * 7919;
            let code = (0..320)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state as u8
                })
                .collect();
            let mut doc = Document::new();
            doc.add_binary_dense_vector(field, code);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        writer.build_vector_index().await.unwrap();
        (index, writer)
    }

    #[tokio::test]
    async fn prefix_cache_rejects_disagreeing_secondary_without_publishing_or_leaking() {
        let (index, mut writer) = soar_fixture().await;
        let searcher = index.reader().await.unwrap().searcher().await.unwrap();
        let segment = &searcher.segment_readers()[0];
        let Some(VectorIndex::BinaryIvf(ann)) = segment.get_vector_index(Field(0)) else {
            panic!("binary IVF")
        };
        let mut broken = ann.get().clone();
        broken.prefix_cache = Default::default();
        let mut bytes = broken.raw.as_slice().to_vec();
        let secondary = broken.runs.iter().find(|run| run.prefix_only).unwrap();
        bytes[secondary.prefix.start] ^= 1;
        broken.raw = OwnedBytes::new(bytes);
        let budget = PrefixCacheBudget::new(1024 * 1024);
        let error = broken
            .prepare_prefix_cache(&segment.flat_vectors()[&0], &budget)
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("secondary differs"));
        assert_eq!(broken.resident_prefix_bytes(), 0);
        assert_eq!(budget.stats().allocated_bytes, 0);
        assert_eq!(budget.stats().aborted_builds, 1);
        writer.shutdown().await.unwrap();
    }
    #[tokio::test]
    async fn resident_nomination_never_reads_persisted_prefixes_or_labels() {
        let (index, mut writer) = soar_fixture().await;
        let searcher = index.reader().await.unwrap().searcher().await.unwrap();
        let segment = &searcher.segment_readers()[0];
        let Some(VectorIndex::BinaryIvf(ann)) = segment.get_vector_index(Field(0)) else {
            panic!("binary IVF")
        };
        let cached = ann.get().clone();
        let budget = PrefixCacheBudget::new(1024 * 1024);
        cached
            .prepare_prefix_cache(&segment.flat_vectors()[&0], &budget)
            .await
            .unwrap();
        let mut poisoned = cached.clone();
        let mut bytes = poisoned.raw.as_slice().to_vec();
        for run in poisoned.runs.iter() {
            bytes[run.doc_ids.clone()].fill(255);
            bytes[run.ordinals.clone()].fill(255);
            bytes[run.prefix.clone()].fill(255);
        }
        poisoned.raw = OwnedBytes::new(bytes);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let clusters = [0, 1, 2, 3];
        for filtered in [false, true] {
            let mut reference = cached.clone();
            let mut actual = poisoned.clone();
            if filtered {
                let visibility = Arc::new(crate::query::DocBitset::from_predicate(32, &|doc| {
                    doc % 3 != 0
                }));
                reference.alive_docs = Some(Arc::clone(&visibility));
                actual.alive_docs = Some(visibility);
            }
            for byte in [0, 71, 255] {
                for rerank in [0, 1, 8, 1000] {
                    for threshold in [1, usize::MAX] {
                        let query = vec![byte; 320];
                        let expected = pool
                            .install(|| {
                                reference.binary_prefix_candidates(
                                    &query,
                                    &clusters,
                                    reference.header.vector_count,
                                    rerank,
                                    threshold,
                                )
                            })
                            .unwrap();
                        let got = pool
                            .install(|| {
                                actual.binary_prefix_candidates(
                                    &query,
                                    &clusters,
                                    actual.header.vector_count,
                                    rerank,
                                    threshold,
                                )
                            })
                            .unwrap();
                        assert_eq!(
                            got, expected,
                            "nomination must be independent of persisted prefix/label bytes"
                        );
                    }
                }
            }
        }
        writer.shutdown().await.unwrap();
    }
}
