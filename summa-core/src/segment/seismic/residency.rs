//! Budgeted residency for compact lookup directories. Encoded payload owners
//! remain untouched so merges copy the original runs without reconstruction.
use super::*;
use crate::segment::pin::{PinMode, PinReport, pin_section};

impl SeismicIndex {
    pub(crate) fn pin_term_directories(
        &mut self,
        mode: PinMode,
        remaining: &mut u64,
        report: &mut PinReport,
    ) {
        for run in self
            .partitions
            .iter_mut()
            .flatten()
            .flat_map(|p| &mut p.runs)
        {
            pin_section(
                &mut run.term_directory,
                "seismic.term_directory",
                mode,
                remaining,
                report,
            );
        }
    }

    pub(crate) fn pin_row_directories(
        &mut self,
        mode: PinMode,
        remaining: &mut u64,
        report: &mut PinReport,
    ) {
        for run in &mut self.runs {
            pin_section(
                &mut run.row_directory,
                "seismic.row_directory",
                mode,
                remaining,
                report,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn mmap_fixture() -> (tempfile::TempDir, SeismicIndex) {
        let directory = tempfile::tempdir().unwrap();
        let (_, source) = super::super::tests::fixture(WeightQuantization::Float32);
        let map = |name: &str, bytes: &[u8]| {
            let path = directory.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            let file = std::fs::File::open(&path).unwrap();
            // The fixture never changes an encoded file after mapping it.
            OwnedBytes::from_mmap(Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() }))
        };
        let mut index = SeismicIndex::parse(map("root", &source.bytes), 3, 3).unwrap();
        for (id, part) in source.partitions.iter().enumerate() {
            index
                .attach_partition(
                    id,
                    map(&format!("part-{id}"), &part.as_ref().unwrap().bytes),
                    0,
                )
                .unwrap();
        }
        index.finish_partitions().unwrap();
        (directory, index)
    }

    fn directory_bytes(index: &SeismicIndex) -> (u64, u64) {
        let terms = index
            .partitions
            .iter()
            .flatten()
            .flat_map(|p| &p.runs)
            .map(|r| r.term_directory.len() as u64)
            .sum();
        let rows = index
            .runs
            .iter()
            .map(|r| r.row_directory.len() as u64)
            .sum();
        (terms, rows)
    }

    fn pin(index: &mut SeismicIndex, mode: PinMode, budget: u64) -> PinReport {
        let mut remaining = budget;
        let mut report = PinReport::default();
        index.pin_term_directories(mode, &mut remaining, &mut report);
        index.pin_row_directories(mode, &mut remaining, &mut report);
        assert_eq!(remaining + report.pinned_bytes, budget);
        report
    }

    fn assert_payloads_mapped(index: &SeismicIndex) {
        assert!(index.bytes.is_mmap());
        for run in index
            .runs
            .iter()
            .chain(index.partitions.iter().flatten().flat_map(|p| &p.runs))
        {
            assert!(run.bytes.is_mmap());
        }
    }

    fn assert_equivalent(before: &SeismicIndex, after: &SeismicIndex) {
        assert_eq!(before.encoded_bytes(), after.encoded_bytes());
        assert_eq!(before.cluster_count(), after.cluster_count());
        assert_eq!(before.nomination_count(), after.nomination_count());
        for row in 0..before.len() {
            assert_eq!(before.key(row), after.key(row));
            assert_eq!(
                before.vector_byte_len(row).unwrap(),
                after.vector_byte_len(row).unwrap()
            );
            assert_eq!(
                before.vector(row).iter().collect::<Vec<_>>(),
                after.vector(row).iter().collect::<Vec<_>>()
            );
        }
        for doc in 0..3 {
            assert_eq!(
                before.for_document(doc).collect::<Vec<_>>(),
                after.for_document(doc).collect::<Vec<_>>()
            );
        }
        for dim in [2, 70_000, 999] {
            assert_eq!(before.doc_count(dim), after.doc_count(dim));
            let candidates = |index: &SeismicIndex| {
                let mut rows = Vec::new();
                for run in index.term_runs(dim) {
                    for cluster in run.clusters() {
                        rows.extend(cluster.rows());
                    }
                }
                rows
            };
            assert_eq!(candidates(before), candidates(after));
        }
        let mut left = super::super::MemoryWriter::default();
        let mut right = super::super::MemoryWriter::default();
        super::super::copy_sources_for_test(&[(before, 0)], &mut left, &|| Ok(())).unwrap();
        super::super::copy_sources_for_test(&[(after, 0)], &mut right, &|| Ok(())).unwrap();
        let left = left.parse(3, 3).unwrap();
        let right = right.parse(3, 3).unwrap();
        assert_eq!(&*left.bytes, &*right.bytes);
        for (a, b) in left.partitions.iter().zip(&right.partitions) {
            assert_eq!(&*a.as_ref().unwrap().bytes, &*b.as_ref().unwrap().bytes);
        }
    }

    #[test]
    fn copy_pins_only_directories_and_preserves_multivalue_lookup_and_merge_bytes() {
        let (_directory, mut index) = mmap_fixture();
        let before = index.clone();
        let (terms, rows) = directory_bytes(&index);
        let report = pin(&mut index, PinMode::Copy, u64::MAX);
        assert_eq!(report.intended_bytes, terms + rows);
        assert_eq!(report.pinned_bytes, terms + rows);
        assert_eq!(report.heap_copy_bytes, terms + rows);
        assert_eq!(report.skipped_budget_bytes, 0);
        assert_eq!(report.failed_bytes, 0);
        for run in &index.runs {
            assert!(!run.row_directory.is_mmap());
        }
        for run in index.partitions.iter().flatten().flat_map(|p| &p.runs) {
            if !run.term_directory.is_empty() {
                assert!(!run.term_directory.is_mmap());
            }
        }
        assert_payloads_mapped(&index);
        assert_equivalent(&before, &index);
    }

    #[test]
    fn cloned_readers_share_copied_directories_and_outlive_the_original() {
        let (_directory, mut original) = mmap_fixture();
        let unpinned = original.clone();
        let report = pin(&mut original, PinMode::Copy, u64::MAX);
        assert!(report.heap_copy_bytes > 0);
        let cloned = original.clone();
        for (left, right) in original.runs.iter().zip(&cloned.runs) {
            assert_eq!(left.row_directory.as_ptr(), right.row_directory.as_ptr());
            assert_eq!(left.bytes.as_ptr(), right.bytes.as_ptr());
        }
        for (left, right) in original.partitions.iter().zip(&cloned.partitions) {
            let (Some(left), Some(right)) = (left, right) else {
                continue;
            };
            for (left, right) in left.runs.iter().zip(&right.runs) {
                assert_eq!(left.term_directory.as_ptr(), right.term_directory.as_ptr());
                assert_eq!(left.bytes.as_ptr(), right.bytes.as_ptr());
            }
        }
        drop(original);
        assert_equivalent(&unpinned, &cloned);
    }

    #[test]
    fn directory_budget_prioritizes_terms_and_reports_disabled_pinning() {
        let (_directory, mut index) = mmap_fixture();
        let (terms, rows) = directory_bytes(&index);
        let disabled = pin(&mut index, PinMode::Copy, 0);
        assert_eq!(disabled.intended_bytes, terms + rows);
        assert_eq!(disabled.skipped_budget_bytes, terms + rows);
        assert_eq!(disabled.pinned_bytes, 0);
        let report = pin(&mut index, PinMode::Copy, terms);
        assert_eq!(report.pinned_bytes, terms);
        assert_eq!(report.skipped_budget_bytes, rows);
        assert!(index.runs.iter().all(|r| r.row_directory.is_mmap()));
        assert_payloads_mapped(&index);
    }

    #[test]
    fn mlock_keeps_mapping_views_and_accounts_for_unavailable_lock_budget() {
        let (_directory, mut index) = mmap_fixture();
        let before = index.clone();
        let (terms, rows) = directory_bytes(&index);
        let report = pin(&mut index, PinMode::Mlock, terms + rows);
        assert_eq!(report.intended_bytes, terms + rows);
        assert_eq!(report.pinned_bytes + report.failed_bytes, terms + rows);
        assert_eq!(report.heap_copy_bytes, 0);
        assert_eq!(report.skipped_budget_bytes, 0);
        assert!(index.runs.iter().all(|r| r.row_directory.is_mmap()));
        assert_payloads_mapped(&index);
        assert_equivalent(&before, &index);
    }

    #[test]
    fn heap_backed_directories_need_no_copy_or_pin_budget() {
        let (_, mut index) = super::super::tests::fixture(WeightQuantization::Float32);
        let report = pin(&mut index, PinMode::Copy, u64::MAX);
        assert_eq!(report.intended_bytes, 0);
        assert_eq!(report.pinned_bytes, 0);
        assert_eq!(report.heap_copy_bytes, 0);
    }
}
