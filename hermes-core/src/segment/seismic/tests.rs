use self::copy_sources_for_test as write_sources;
use super::*;
use rustc_hash::FxHashMap;

#[derive(Default)]
pub(crate) struct MemoryWriter {
    root: Vec<u8>,
    parts: [Vec<u8>; PARTITIONS],
}
impl SeismicWriter for MemoryWriter {
    fn root(&mut self) -> &mut dyn Write {
        &mut self.root
    }
    fn partition(&mut self, id: usize) -> &mut dyn Write {
        &mut self.parts[id]
    }
}
impl MemoryWriter {
    pub(crate) fn parse(self, docs: u32, rows: u32) -> Result<SeismicIndex> {
        let mut index = SeismicIndex::parse(OwnedBytes::new(self.root), docs, rows)?;
        for (id, bytes) in self.parts.into_iter().enumerate() {
            index.attach_partition(id, OwnedBytes::new(bytes), 0)?;
        }
        index.finish_partitions()?;
        Ok(index)
    }
    fn is_empty(&self) -> bool {
        self.root.is_empty() && self.parts.iter().all(Vec::is_empty)
    }
}
pub(crate) fn copy_sources_for_test(
    sources: &[(&SeismicIndex, u32)],
    output: &mut MemoryWriter,
    cancel: &impl Fn() -> Result<()>,
) -> Result<()> {
    super::write_sources(sources, &mut output.root, cancel)?;
    for id in 0..PARTITIONS {
        write_partition_sources(sources, id, &mut output.parts[id], cancel)?;
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn write_maintained(
    index: &SeismicIndex,
    config: &SparseVectorConfig,
    should_continue: &impl Fn() -> bool,
    budget: usize,
    is_alive: &impl Fn(u32) -> bool,
    cancel: &impl Fn() -> Result<()>,
    output: &mut MemoryWriter,
) -> Result<(u64, u32)> {
    output.root.extend_from_slice(&index.bytes);
    let mut debt = 0;
    for id in 0..PARTITIONS {
        let (_, remaining) = write_maintained_partition(
            index,
            id,
            config,
            should_continue,
            budget,
            is_alive,
            cancel,
            &mut output.parts[id],
        )?;
        debt += remaining;
    }
    Ok((
        output.root.len() as u64 + output.parts.iter().map(|p| p.len() as u64).sum::<u64>(),
        debt,
    ))
}

pub(super) fn fixture(quant: WeightQuantization) -> (SparseVectorConfig, SeismicIndex) {
    let config = SparseVectorConfig {
        dims: Some(100_000),
        weight_quantization: quant,
        ..Default::default()
    };
    let mut postings = FxHashMap::default();
    postings.insert(70_000, vec![(0, 0, -2.0), (0, 2, 3.0), (2, 0, 1.0)]);
    postings.insert(2, vec![(0, 0, 1.0), (2, 0, -4.0)]);
    let mut bytes = MemoryWriter::default();
    build_blob(postings, &config, &mut bytes).unwrap();
    let index = bytes.parse(3, 3).unwrap();
    (config, index)
}
#[test]
fn signed_values_large_dimensions_missing_documents_and_ordinals_survive_reopen() {
    for quant in [
        WeightQuantization::Float32,
        WeightQuantization::Float16,
        WeightQuantization::UInt8,
        WeightQuantization::UInt4,
    ] {
        let (_, index) = fixture(quant);
        assert_eq!(index.rows_for_document(1).count(), 0);
        assert_eq!(
            index.for_document(0).collect::<Vec<_>>(),
            vec![(0, 0), (2, 1)]
        );
        assert_eq!(
            index.vector(0).iter().collect::<Vec<_>>(),
            vec![(2, 1.0), (70_000, -2.0)]
        );
        assert!(!index.is_single_valued());
        assert_eq!(
            index
                .term_runs(70_000)
                .flat_map(|r| r.clusters())
                .flat_map(|c| c.rows().collect::<Vec<_>>())
                .count(),
            3
        );
    }
}
#[test]
fn copy_merge_preserves_run_bytes_and_rebases_documents_and_rows() {
    let (_, index) = fixture(WeightQuantization::Float32);
    let mut bytes = MemoryWriter::default();
    write_sources(&[(&index, 0), (&index, 3)], &mut bytes, &|| Ok(())).unwrap();
    let merged = bytes.parse(6, 6).unwrap();
    assert_eq!(
        merged.runs[0].bytes.as_slice(),
        index.runs[0].bytes.as_slice()
    );
    assert_eq!(
        merged.runs[1].bytes.as_slice(),
        index.runs[0].bytes.as_slice()
    );
    assert_eq!(merged.key(4), LogicalUnit { doc: 3, ordinal: 2 });
    assert_eq!(merged.pending_terms(), 2);
}
#[test]
fn bounded_maintenance_copies_forward_values_and_converges_term_debt() {
    let (config, index) = fixture(WeightQuantization::Float32);
    let mut bytes = MemoryWriter::default();
    write_sources(&[(&index, 0), (&index, 3)], &mut bytes, &|| Ok(())).unwrap();
    let merged = bytes.parse(6, 6).unwrap();
    let mut output = MemoryWriter::default();
    let admitted = std::cell::Cell::new(false);
    let (_, debt) = write_maintained(
        &merged,
        &config,
        &|| !admitted.replace(true),
        16 << 20,
        &|_| true,
        &|| Ok(()),
        &mut output,
    )
    .unwrap();
    assert_eq!(debt, 1);
    let partial = output.parse(6, 6).unwrap();
    for row in 0..merged.len() {
        assert_eq!(merged.vector(row).bytes, partial.vector(row).bytes);
        assert_eq!(merged.key(row), partial.key(row));
    }
    let mut output = MemoryWriter::default();
    let (_, debt) = write_maintained(
        &partial,
        &config,
        &|| true,
        16 << 20,
        &|_| true,
        &|| Ok(()),
        &mut output,
    )
    .unwrap();
    assert_eq!(debt, 0);
    let done = output.parse(6, 6).unwrap();
    assert_eq!(done.pending_terms(), 0);
}
#[test]
fn corrupt_envelopes_and_cancelled_copies_return_errors() {
    let (_, index) = fixture(WeightQuantization::Float32);
    let mut bad = index.bytes.to_vec();
    let last = bad.len() - FOOTER;
    bad[last] ^= 1;
    assert!(SeismicIndex::parse(OwnedBytes::new(bad), 3, 3).is_err());
    let mut out = MemoryWriter::default();
    assert!(write_sources(&[(&index, 0)], &mut out, &|| Err(Error::IndexClosed)).is_err());
    assert!(out.is_empty());
}
#[test]
fn delete_compaction_backfills_from_retained_forward_values() {
    for quant in [
        WeightQuantization::Float32,
        WeightQuantization::Float16,
        WeightQuantization::UInt8,
        WeightQuantization::UInt4,
    ] {
        let (config, index) = fixture(quant);
        let mut bytes = MemoryWriter::default();
        write_compacted(
            &index,
            &|doc| (doc == 2).then_some(0),
            &config,
            16 << 20,
            &|| Ok(()),
            &mut bytes,
        )
        .unwrap();
        let compacted = bytes.parse(1, 1).unwrap();
        assert_eq!(compacted.key(0).doc, 0);
        assert_eq!(compacted.vector(0).bytes, index.vector(2).bytes);
        assert_eq!(
            compacted.vector(0).iter().collect::<Vec<_>>(),
            index.vector(2).iter().collect::<Vec<_>>()
        );
    }
}

#[test]
fn empty_vectors_preserve_value_ordinals_without_inventing_missing_values() {
    for quant in [WeightQuantization::Float32, WeightQuantization::UInt4] {
        let config = SparseVectorConfig {
            weight_quantization: quant,
            ..Default::default()
        };
        let mut bytes = MemoryWriter::default();
        build_blob_with_keys(
            FxHashMap::default(),
            &[(0, 0), (0, 1), (2, 0)],
            &config,
            &mut bytes,
        )
        .unwrap();
        let index = bytes.parse(3, 3).unwrap();
        assert_eq!(index.len(), 3);
        assert_eq!(
            index.for_document(0).collect::<Vec<_>>(),
            vec![(0, 0), (1, 1)]
        );
        assert_eq!(index.rows_for_document(1).count(), 0);
        assert_eq!(index.vector(0).iter().count(), 0);
        let mut compacted = MemoryWriter::default();
        write_compacted(
            &index,
            &|d| (d == 0).then_some(0),
            &config,
            1 << 20,
            &|| Ok(()),
            &mut compacted,
        )
        .unwrap();
        let compacted = compacted.parse(1, 2).unwrap();
        assert_eq!(compacted.len(), 2);
    }
}

#[test]
fn maintenance_backfills_deleted_top_postings_from_complete_forward_values() {
    let mut config = SparseVectorConfig {
        dims: Some(2),
        ..Default::default()
    };
    config.seismic.postings = 1;
    config.seismic.cluster_size = 1;
    let mut postings = FxHashMap::default();
    postings.insert(0, vec![(0, 0, 5.0), (1, 0, 4.0), (2, 0, 3.0)]);
    let mut bytes = MemoryWriter::default();
    build_blob(postings, &config, &mut bytes).unwrap();
    let original = bytes.parse(3, 3).unwrap();
    let mut bytes = MemoryWriter::default();
    write_sources(&[(&original, 0), (&original, 3)], &mut bytes, &|| Ok(())).unwrap();
    let merged = bytes.parse(6, 6).unwrap();
    let mut bytes = MemoryWriter::default();
    let (_, debt) = write_maintained(
        &merged,
        &config,
        &|| true,
        1 << 20,
        &|doc| doc != 0 && doc != 3,
        &|| Ok(()),
        &mut bytes,
    )
    .unwrap();
    assert_eq!(debt, 0);
    let rebuilt = bytes.parse(6, 6).unwrap();
    let nominated: Vec<_> = rebuilt
        .term_runs(0)
        .flat_map(|t| t.clusters())
        .flat_map(|c| c.rows().collect::<Vec<_>>())
        .collect();
    assert_eq!(nominated, vec![1]);
}

#[test]
fn maintenance_with_no_scratch_budget_keeps_encoded_runs_and_reports_debt() {
    let (config, index) = fixture(WeightQuantization::Float32);
    let mut bytes = MemoryWriter::default();
    write_sources(&[(&index, 0), (&index, 3)], &mut bytes, &|| Ok(())).unwrap();
    let merged = bytes.parse(6, 6).unwrap();
    let mut bytes = MemoryWriter::default();
    let (_, debt) = write_maintained(
        &merged,
        &config,
        &|| true,
        0,
        &|_| true,
        &|| Ok(()),
        &mut bytes,
    )
    .unwrap();
    assert_eq!(debt, 2);
    assert_eq!(bytes.root.as_slice(), merged.bytes.as_slice());
    for (id, part) in merged.partitions.iter().enumerate() {
        assert_eq!(
            bytes.parts[id].as_slice(),
            part.as_ref().unwrap().bytes.as_slice()
        );
    }
}

#[test]
fn geometric_clusters_group_similar_vectors_instead_of_adjacent_rows() {
    let mut config = SparseVectorConfig {
        dims: Some(300),
        ..Default::default()
    };
    config.seismic.cluster_size = 2;
    config.seismic.summary_energy = 1.0;
    let mut postings = FxHashMap::default();
    postings.insert(0, (0..4).map(|doc| (doc, 0, 1.0)).collect());
    postings.insert(100, vec![(0, 0, 10.0), (2, 0, 10.0)]);
    postings.insert(200, vec![(1, 0, 10.0), (3, 0, 10.0)]);
    let mut bytes = MemoryWriter::default();
    build_blob(postings, &config, &mut bytes).unwrap();
    let index = bytes.parse(4, 4).unwrap();
    let groups: Vec<Vec<_>> = index
        .term_runs(0)
        .flat_map(|t| t.clusters())
        .map(|c| c.rows().collect())
        .collect();
    assert_eq!(groups, vec![vec![0, 2], vec![1, 3]]);
    assert_eq!(index.doc_count(0), 4);
}

#[test]
fn full_dimension_frequency_survives_top_l_pruning_copy_and_maintenance() {
    let mut config = SparseVectorConfig {
        dims: Some(2),
        ..Default::default()
    };
    config.seismic.postings = 1;
    config.seismic.cluster_size = 1;
    let mut postings = FxHashMap::default();
    postings.insert(0, vec![(0, 0, 5.0), (1, 0, 4.0), (2, 0, 3.0)]);
    let mut bytes = MemoryWriter::default();
    build_blob(postings, &config, &mut bytes).unwrap();
    let index = bytes.parse(3, 3).unwrap();
    assert_eq!(index.doc_count(0), 3);
    assert_eq!(index.nomination_count(), 1);
    let mut bytes = MemoryWriter::default();
    write_sources(&[(&index, 0), (&index, 3)], &mut bytes, &|| Ok(())).unwrap();
    let merged = bytes.parse(6, 6).unwrap();
    let mut bytes = MemoryWriter::default();
    write_maintained(
        &merged,
        &config,
        &|| true,
        1 << 20,
        &|_| true,
        &|| Ok(()),
        &mut bytes,
    )
    .unwrap();
    let done = bytes.parse(6, 6).unwrap();
    assert_eq!(done.doc_count(0), 6);
    assert_eq!(done.nomination_count(), 1);
}

#[test]
fn copy_merge_unions_inferred_vocabulary_bounds_without_reencoding_runs() {
    let config = SparseVectorConfig::default();
    let mut left = FxHashMap::default();
    left.insert(1, vec![(0, 0, 1.0)]);
    let mut right = FxHashMap::default();
    right.insert(90_000, vec![(0, 0, 2.0)]);
    let mut bytes = MemoryWriter::default();
    build_blob(left, &config, &mut bytes).unwrap();
    let left = bytes.parse(1, 1).unwrap();
    let mut bytes = MemoryWriter::default();
    build_blob(right, &config, &mut bytes).unwrap();
    let right = bytes.parse(1, 1).unwrap();
    let mut bytes = MemoryWriter::default();
    write_sources(&[(&left, 0), (&right, 1)], &mut bytes, &|| Ok(())).unwrap();
    let merged = bytes.parse(2, 2).unwrap();
    assert_eq!(merged.dims(), 90_001);
    assert_eq!(
        merged.runs[0].bytes.as_slice(),
        left.runs[0].bytes.as_slice()
    );
    assert_eq!(
        merged.runs[1].bytes.as_slice(),
        right.runs[0].bytes.as_slice()
    );
    assert_eq!(
        merged.vector(1).iter().collect::<Vec<_>>(),
        vec![(90_000, 2.0)]
    );
}

#[test]
fn incomplete_nomination_coverage_backfills_before_consolidating_lists() {
    let mut config = SparseVectorConfig {
        dims: Some(2),
        pruning: Some(0.1),
        min_terms: 0,
        ..Default::default()
    };
    config.seismic.postings = 3;
    config.seismic.cluster_size = 1;
    let mut postings = FxHashMap::default();
    postings.insert(0, vec![(0, 0, 5.0), (1, 0, 4.0), (2, 0, 3.0)]);
    let mut bytes = MemoryWriter::default();
    build_blob(postings, &config, &mut bytes).unwrap();
    let index = bytes.parse(3, 3).unwrap();
    assert_eq!(index.nomination_count(), 1);
    let mut bytes = MemoryWriter::default();
    write_sources(&[(&index, 0), (&index, 3)], &mut bytes, &|| Ok(())).unwrap();
    let merged = bytes.parse(6, 6).unwrap();
    let mut bytes = MemoryWriter::default();
    write_maintained(
        &merged,
        &config,
        &|| true,
        1 << 20,
        &|_| true,
        &|| Ok(()),
        &mut bytes,
    )
    .unwrap();
    let done = bytes.parse(6, 6).unwrap();
    let mut rows: Vec<_> = done
        .term_runs(0)
        .flat_map(|t| t.clusters())
        .flat_map(|c| c.rows().collect::<Vec<_>>())
        .collect();
    rows.sort_unstable();
    assert_eq!(rows, vec![0, 1, 3]);
}

#[test]
fn format_admission_rejects_incorrect_vector_count_and_zero_summary_energy() {
    let (_, index) = fixture(WeightQuantization::Float32);
    assert!(SeismicIndex::parse(index.bytes.clone(), 3, 4).is_err());
    assert!(SeismicIndex::parse(index.bytes.clone(), 3, 2).is_err());
    let mut bytes = index.bytes.to_vec();
    let at = bytes.len() - FOOTER + 24;
    bytes[at..at + 4].copy_from_slice(&0.0f32.to_bits().to_le_bytes());
    assert!(SeismicIndex::parse(OwnedBytes::new(bytes), 3, 3).is_err());
}

#[test]
fn wide_vector_compaction_admits_global_term_map_before_writing() {
    let coordinates = 128usize;
    let config = SparseVectorConfig {
        dims: Some(coordinates as u32),
        ..Default::default()
    };
    let postings = (0..coordinates as u32)
        .map(|dim| (dim, vec![(0, 0, dim as f32 + 1.0)]))
        .collect();
    let mut bytes = MemoryWriter::default();
    build_blob(postings, &config, &mut bytes).unwrap();
    let index = bytes.parse(1, 1).unwrap();
    // This admits the selected-vector clustering estimate, but cannot also
    // hold the global term map and output term directory used by compaction.
    let budget = 128 + coordinates * 300;
    let mut output = MemoryWriter::default();
    assert!(write_compacted(&index, &Some, &config, budget, &|| Ok(()), &mut output).is_err());
    assert!(output.is_empty(), "admission must precede forward output");
    write_compacted(&index, &Some, &config, 1 << 20, &|| Ok(()), &mut output).unwrap();
    let compacted = output.parse(1, 1).unwrap();
    assert_eq!(compacted.vector(0).bytes, index.vector(0).bytes);
}

#[test]
fn old_summary_format_version_is_rejected_without_fallback() {
    let config = SparseVectorConfig::default();
    let mut bytes = MemoryWriter::default();
    build_blob(
        [(7, vec![(0, 0, 1.0)])].into_iter().collect(),
        &config,
        &mut bytes,
    )
    .unwrap();
    let version = bytes.root.len() - FOOTER + 4;
    bytes.root[version..version + 4].copy_from_slice(&2u32.to_le_bytes());
    assert!(bytes.parse(1, 1).is_err());
}

#[test]
fn u32_dimensions_keep_the_existing_exclusive_vocabulary_limit() {
    let config = SparseVectorConfig::default();
    let mut bytes = MemoryWriter::default();
    build_blob(
        [(u32::MAX - 1, vec![(0, 0, 1.0)])].into_iter().collect(),
        &config,
        &mut bytes,
    )
    .unwrap();
    let index = bytes.parse(1, 1).unwrap();
    assert_eq!(
        index.vector(0).iter().collect::<Vec<_>>(),
        vec![(u32::MAX - 1, 1.0)]
    );
    let run = index.term_runs(u32::MAX - 1).next().unwrap();
    let mut scores = vec![0.0; run.cluster_count()];
    run.score_summaries(&[(u32::MAX - 1, 1.0)], &mut scores);
    assert_eq!(scores, vec![1.0]);
    let mut output = MemoryWriter::default();
    assert!(
        build_blob(
            [(u32::MAX, vec![(0, 0, 1.0)])].into_iter().collect(),
            &config,
            &mut output
        )
        .is_err()
    );
    assert!(output.is_empty());
}

#[test]
fn partition_maintenance_preserves_forward_and_unselected_partition_bytes() {
    let (config, original) = fixture(WeightQuantization::Float32);
    let mut output = MemoryWriter::default();
    write_sources(&[(&original, 0), (&original, 3)], &mut output, &|| Ok(())).unwrap();
    let merged = output.parse(6, 6).unwrap();
    let selected = 2;
    let mut bytes = Vec::new();
    let (written, debt) = write_maintained_partition(
        &merged,
        selected,
        &config,
        &|| true,
        16 << 20,
        &|_| true,
        &|| Ok(()),
        &mut bytes,
    )
    .unwrap();
    assert_eq!(written, bytes.len() as u64);
    assert!(written < merged.encoded_bytes() as u64);
    assert_eq!(debt, 0);
    let mut maintained = SeismicIndex::parse(merged.bytes.clone(), 6, 6).unwrap();
    for id in 0..PARTITIONS {
        let payload = if id == selected {
            OwnedBytes::new(bytes.clone())
        } else {
            merged.partitions[id].as_ref().unwrap().bytes.clone()
        };
        maintained.attach_partition(id, payload, 0).unwrap();
    }
    maintained.finish_partitions().unwrap();
    assert_eq!(maintained.pending_terms(), 1);
    assert_eq!(maintained.bytes.as_slice(), merged.bytes.as_slice());
    for id in 0..PARTITIONS {
        if id != selected {
            assert_eq!(
                maintained.partitions[id].as_ref().unwrap().bytes.as_slice(),
                merged.partitions[id].as_ref().unwrap().bytes.as_slice()
            );
        }
    }
    for row in 0..merged.len() {
        assert_eq!(maintained.vector(row).bytes, merged.vector(row).bytes);
        assert_eq!(maintained.key(row), merged.key(row));
    }
}

#[test]
fn partition_admission_rejects_missing_duplicate_misrouted_and_incompatible_parts() {
    let (_, index) = fixture(WeightQuantization::Float32);
    let mut root = SeismicIndex::parse(index.bytes.clone(), 3, 3).unwrap();
    assert!(root.finish_partitions().is_err());
    let original = index.partitions[2].as_ref().unwrap().bytes.clone();
    assert!(root.attach_partition(1, original.clone(), 0).is_err());
    let mut bad = original.to_vec();
    let footer = bad.len() - FOOTER;
    bad[footer + 16..footer + 20].copy_from_slice(&1u32.to_le_bytes());
    assert!(root.attach_partition(2, OwnedBytes::new(bad), 0).is_err());
    let mut bad = original.to_vec();
    let run_footer = bad.len() - FOOTER - RUN_ENTRY - RUN_FOOTER;
    let directory = u64_at(&bad, run_footer + 16) as usize;
    bad[directory..directory + 4].copy_from_slice(&3u32.to_le_bytes());
    assert!(root.attach_partition(2, OwnedBytes::new(bad), 0).is_err());
    root.attach_partition(2, original.clone(), 0).unwrap();
    assert!(root.attach_partition(2, original, 0).is_err());
}

#[test]
fn copy_merge_preserves_each_nomination_run_without_reencoding() {
    let (_, index) = fixture(WeightQuantization::Float32);
    let mut output = MemoryWriter::default();
    write_sources(&[(&index, 0), (&index, 3)], &mut output, &|| Ok(())).unwrap();
    let merged = output.parse(6, 6).unwrap();
    for id in 0..PARTITIONS {
        let source = index.partitions[id].as_ref().unwrap();
        let target = merged.partitions[id].as_ref().unwrap();
        assert_eq!(target.runs.len(), 2);
        for run in &target.runs {
            assert_eq!(run.bytes.as_slice(), source.runs[0].bytes.as_slice());
        }
    }
}

#[test]
fn expired_partition_pass_retains_debt_and_cancelled_pass_writes_nothing() {
    let (config, index) = fixture(WeightQuantization::Float32);
    let mut output = MemoryWriter::default();
    write_sources(&[(&index, 0), (&index, 3)], &mut output, &|| Ok(())).unwrap();
    let merged = output.parse(6, 6).unwrap();
    let mut bytes = Vec::new();
    let (_, debt) = write_maintained_partition(
        &merged,
        2,
        &config,
        &|| false,
        16 << 20,
        &|_| true,
        &|| Ok(()),
        &mut bytes,
    )
    .unwrap();
    assert_eq!(debt, 1);
    let mut cancelled = Vec::new();
    assert!(
        write_maintained_partition(
            &merged,
            2,
            &config,
            &|| true,
            16 << 20,
            &|_| true,
            &|| Err(Error::IndexClosed),
            &mut cancelled
        )
        .is_err()
    );
    assert!(cancelled.is_empty());
}

#[test]
fn maintenance_spends_expiring_budget_on_largest_terms_with_stable_dimension_ties() {
    let config = SparseVectorConfig {
        dims: Some(64),
        ..Default::default()
    };
    let postings = [
        (0, vec![(0, 0, 1.0)]),
        (16, (0..4).map(|doc| (doc, 0, 1.0)).collect()),
        (32, (0..4).map(|doc| (doc, 0, 1.0)).collect()),
    ]
    .into_iter()
    .collect();
    let mut output = MemoryWriter::default();
    build_blob(postings, &config, &mut output).unwrap();
    let original = output.parse(4, 4).unwrap();
    let mut output = MemoryWriter::default();
    write_sources(&[(&original, 0), (&original, 4)], &mut output, &|| Ok(())).unwrap();
    let merged = output.parse(8, 8).unwrap();
    let (count, bytes) = merged.partition_maintenance_priority(0);
    assert_eq!(count, 3);
    assert!(bytes > 0);
    let mut maintained = MemoryWriter::default();
    let admitted = std::cell::Cell::new(false);
    write_maintained(
        &merged,
        &config,
        &|| !admitted.replace(true),
        16 << 20,
        &|_| true,
        &|| Ok(()),
        &mut maintained,
    )
    .unwrap();
    let maintained = maintained.parse(8, 8).unwrap();
    assert_eq!(maintained.term_runs(0).count(), 2);
    assert_eq!(maintained.term_runs(16).count(), 1);
    assert_eq!(maintained.term_runs(32).count(), 2);
    assert_eq!(maintained.bytes.as_slice(), merged.bytes.as_slice());
    assert_eq!(
        term_payloads(&merged, 0, 0),
        term_payloads(&maintained, 0, 0)
    );
    assert_eq!(
        term_payloads(&merged, 0, 32),
        term_payloads(&maintained, 0, 32)
    );
}

#[test]
fn partition_admission_rejects_combined_term_frequency_exceeding_forward_rows() {
    let (config, index) = fixture(WeightQuantization::Float32);
    let mut output = MemoryWriter::default();
    write_sources(&[(&index, 0), (&index, 3)], &mut output, &|| Ok(())).unwrap();
    let merged = output.parse(6, 6).unwrap();
    let mut bytes = Vec::new();
    write_maintained_partition(
        &merged,
        2,
        &config,
        &|| false,
        16 << 20,
        &|_| true,
        &|| Ok(()),
        &mut bytes,
    )
    .unwrap();
    let run_footer = bytes.len() - FOOTER - RUN_ENTRY - RUN_FOOTER;
    let directory = u64_at(&bytes, run_footer + 16) as usize;
    let count = u32_at(&bytes, run_footer + 4) as usize;
    assert_eq!(count, 2);
    for entry in bytes[directory..run_footer].chunks_exact_mut(TERM_ENTRY) {
        entry[32..36].copy_from_slice(&6u32.to_le_bytes());
    }
    let mut root = SeismicIndex::parse(merged.bytes.clone(), 6, 6).unwrap();
    assert!(root.attach_partition(2, OwnedBytes::new(bytes), 0).is_err());
}

#[test]
fn partition_admission_rejects_overlapping_and_gapped_term_payloads() {
    let config = SparseVectorConfig {
        dims: Some(32),
        ..Default::default()
    };
    let mut output = MemoryWriter::default();
    build_blob(
        [(0, vec![(0, 0, 1.0)]), (16, vec![(0, 0, 1.0)])]
            .into_iter()
            .collect(),
        &config,
        &mut output,
    )
    .unwrap();
    let index = output.parse(1, 1).unwrap();
    let original = &index.partitions[0].as_ref().unwrap().bytes;
    let run_footer = original.len() - FOOTER - RUN_ENTRY - RUN_FOOTER;
    let term_offset = u64_at(original, run_footer + 16) as usize;
    let mut overlap = original.to_vec();
    overlap[term_offset + TERM_ENTRY + 8..term_offset + TERM_ENTRY + 16]
        .copy_from_slice(&0u64.to_le_bytes());
    let mut root = SeismicIndex::parse(index.bytes.clone(), 1, 1).unwrap();
    let error = root
        .attach_partition(0, OwnedBytes::new(overlap), 0)
        .err()
        .unwrap();
    assert!(error.to_string().contains("overlapping or gapped"));

    let mut gap = original.to_vec();
    gap.insert(term_offset, 0);
    let new_footer = run_footer + 1;
    gap[new_footer + 16..new_footer + 24].copy_from_slice(&(term_offset as u64 + 1).to_le_bytes());
    let run_entry = new_footer + RUN_FOOTER;
    let run_len = u64_at(&gap, run_entry + 8) + 1;
    gap[run_entry + 8..run_entry + 16].copy_from_slice(&run_len.to_le_bytes());
    let error = root
        .attach_partition(0, OwnedBytes::new(gap), 0)
        .err()
        .unwrap();
    assert!(error.to_string().contains("unowned term bytes"));
}

#[test]
fn partition_admission_rejects_nonempty_term_based_past_forward_rows() {
    let (_, index) = fixture(WeightQuantization::Float32);
    let original = &index.partitions[2].as_ref().unwrap().bytes;
    let run_footer = original.len() - FOOTER - RUN_ENTRY - RUN_FOOTER;
    let directory = u64_at(original, run_footer + 16) as usize;
    let rows = u32_at(original, run_footer);
    assert!(u32_at(original, directory + 4) > 0);
    for base in [rows, rows + 1] {
        let mut bytes = original.to_vec();
        bytes[directory + 24..directory + 28].copy_from_slice(&base.to_le_bytes());
        let mut root = SeismicIndex::parse(index.bytes.clone(), 3, 3).unwrap();
        let error = root
            .attach_partition(2, OwnedBytes::new(bytes), 0)
            .expect_err("nonempty term cannot start at or beyond the forward-row end");
        assert!(error.to_string().contains("invalid term directory"));
    }
}

#[test]
fn partition_admission_preserves_empty_term_at_forward_row_end() {
    let (config, index) = fixture(WeightQuantization::Float32);
    let mut output = MemoryWriter::default();
    write_sources(&[(&index, 0), (&index, 3)], &mut output, &|| Ok(())).unwrap();
    let merged = output.parse(6, 6).unwrap();
    let mut bytes = Vec::new();
    write_maintained_partition(
        &merged,
        2,
        &config,
        &|| true,
        16 << 20,
        &|_| false,
        &|| Ok(()),
        &mut bytes,
    )
    .unwrap();
    let run_footer = bytes.len() - FOOTER - RUN_ENTRY - RUN_FOOTER;
    let directory = u64_at(&bytes, run_footer + 16) as usize;
    let rows = u32_at(&bytes, run_footer);
    assert_eq!(u32_at(&bytes, directory + 4), 0);
    bytes[directory + 24..directory + 28].copy_from_slice(&rows.to_le_bytes());
    let mut maintained = SeismicIndex::parse(merged.bytes.clone(), 6, 6).unwrap();
    for partition in 0..PARTITIONS {
        let payload = if partition == 2 {
            OwnedBytes::new(bytes.clone())
        } else {
            merged.partitions[partition].as_ref().unwrap().bytes.clone()
        };
        maintained.attach_partition(partition, payload, 0).unwrap();
    }
    maintained.finish_partitions().unwrap();
    assert_eq!(maintained.term_runs(2).count(), 1);
    assert!(
        maintained
            .term_runs(2)
            .all(|term| term.clusters().next().is_none())
    );
    for row in 0..merged.len() {
        assert_eq!(maintained.key(row), merged.key(row));
        assert_eq!(maintained.vector(row).bytes, merged.vector(row).bytes);
    }
}

#[test]
fn maintenance_uses_available_budget_beyond_sixty_four_terms() {
    let config = SparseVectorConfig {
        dims: Some(80 * PARTITIONS as u32),
        ..Default::default()
    };
    let postings = (0..80)
        .map(|dim| (dim * PARTITIONS as u32, vec![(0, 0, 1.0)]))
        .collect();
    let mut output = MemoryWriter::default();
    build_blob(postings, &config, &mut output).unwrap();
    let original = output.parse(1, 1).unwrap();
    let mut output = MemoryWriter::default();
    write_sources(&[(&original, 0), (&original, 1)], &mut output, &|| Ok(())).unwrap();
    let merged = output.parse(2, 2).unwrap();
    let mut output = MemoryWriter::default();
    let (_, debt) = write_maintained(
        &merged,
        &config,
        &|| true,
        16 << 20,
        &|_| true,
        &|| Ok(()),
        &mut output,
    )
    .unwrap();
    assert_eq!(debt, 0);
    assert_eq!(output.root.as_slice(), merged.bytes.as_slice());
    let maintained = output.parse(2, 2).unwrap();
    assert_eq!(maintained.pending_terms(), 0);
    for dim in (0..80).map(|dim| dim * PARTITIONS as u32) {
        assert_eq!(maintained.term_runs(dim).count(), 1);
        assert_eq!(maintained.doc_count(dim), 2);
    }
}

#[test]
fn maintenance_copies_unaffordable_term_and_rebuilds_later_affordable_term() {
    let mut config = SparseVectorConfig {
        dims: Some(128 * PARTITIONS as u32),
        ..Default::default()
    };
    config.seismic.postings = 2;
    config.seismic.cluster_size = 1;
    let mut postings: FxHashMap<_, _> = (0..128)
        .map(|dim| (1 + dim * PARTITIONS as u32, vec![(0, 0, 1.0), (1, 0, 1.0)]))
        .collect();
    postings.insert(0, vec![(0, 0, 1.0), (1, 0, 1.0)]);
    postings.insert(16, vec![(2, 0, 1.0)]);
    let mut output = MemoryWriter::default();
    build_blob(postings, &config, &mut output).unwrap();
    let original = output.parse(3, 3).unwrap();
    let mut output = MemoryWriter::default();
    write_sources(&[(&original, 0), (&original, 3)], &mut output, &|| Ok(())).unwrap();
    let merged = output.parse(6, 6).unwrap();
    let mut bytes = Vec::new();
    let (_, debt) = write_maintained_partition(
        &merged,
        0,
        &config,
        &|| true,
        4096,
        &|_| true,
        &|| Ok(()),
        &mut bytes,
    )
    .unwrap();
    assert_eq!(debt, 1);
    let mut maintained = SeismicIndex::parse(merged.bytes.clone(), 6, 6).unwrap();
    for id in 0..PARTITIONS {
        let payload = if id == 0 {
            OwnedBytes::new(bytes.clone())
        } else {
            merged.partitions[id].as_ref().unwrap().bytes.clone()
        };
        maintained.attach_partition(id, payload, 0).unwrap();
    }
    maintained.finish_partitions().unwrap();
    assert_eq!(maintained.term_runs(0).count(), 2);
    assert_eq!(maintained.term_runs(16).count(), 1);
    assert_eq!(
        term_payloads(&merged, 0, 0),
        term_payloads(&maintained, 0, 0)
    );
}

fn term_payloads(index: &SeismicIndex, partition: usize, dim: u32) -> Vec<&[u8]> {
    index.partitions[partition]
        .as_ref()
        .unwrap()
        .runs
        .iter()
        .flat_map(|run| {
            (0..run.terms as usize).filter_map(move |entry| {
                let at = run.term_offset + entry * TERM_ENTRY;
                (u32_at(&run.bytes, at) == dim).then(|| {
                    let start = u64_at(&run.bytes, at + 8) as usize;
                    let len = u64_at(&run.bytes, at + 16) as usize;
                    &run.bytes[start..start + len]
                })
            })
        })
        .collect()
}

#[test]
fn cancellation_after_completed_term_aborts_partition_before_publication() {
    struct CancelAfterWrite<'a> {
        cancelled: &'a std::cell::Cell<bool>,
        bytes: Vec<u8>,
    }
    impl Write for CancelAfterWrite<'_> {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            self.cancelled.set(true);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let (config, original) = fixture(WeightQuantization::Float32);
    let mut output = MemoryWriter::default();
    write_sources(&[(&original, 0), (&original, 3)], &mut output, &|| Ok(())).unwrap();
    let merged = output.parse(6, 6).unwrap();
    let cancelled = std::cell::Cell::new(false);
    let mut writer = CancelAfterWrite {
        cancelled: &cancelled,
        bytes: Vec::new(),
    };
    let result = write_maintained_partition(
        &merged,
        2,
        &config,
        &|| true,
        16 << 20,
        &|_| true,
        &|| {
            if cancelled.get() {
                Err(Error::IndexClosed)
            } else {
                Ok(())
            }
        },
        &mut writer,
    );
    assert!(matches!(result, Err(Error::IndexClosed)));
    assert!(!writer.bytes.is_empty());
    let mut reopened = SeismicIndex::parse(merged.bytes.clone(), 6, 6).unwrap();
    assert!(
        reopened
            .attach_partition(2, OwnedBytes::new(writer.bytes), 0)
            .is_err()
    );
}

// Keep the dimension directory sorted while reversing the corresponding bytes,
// as maintenance can do when it writes terms in debt-priority order.
fn shuffled_partition_fixture() -> (SeismicIndex, Vec<u8>) {
    let mut config = SparseVectorConfig {
        dims: Some(64),
        ..Default::default()
    };
    config.seismic.summary_energy = 1.0;
    let mut output = MemoryWriter::default();
    build_blob(
        [
            (0, vec![(0, 0, 1.0), (1, 0, 2.0)]),
            (16, vec![(0, 0, 3.0), (1, 0, 1.0)]),
            (32, vec![(1, 0, 4.0)]),
        ]
        .into_iter()
        .collect(),
        &config,
        &mut output,
    )
    .unwrap();
    let index = output.parse(2, 2).unwrap();
    let part = index.partitions[0].as_ref().unwrap();
    let run = &part.runs[0];
    assert_eq!(run.terms, 3);
    let original = part.bytes.as_slice();
    let mut shuffled = original.to_vec();
    let mut offset = 0;
    for i in (0..run.terms as usize).rev() {
        let entry = run.term_offset + i * TERM_ENTRY;
        let start = u64_at(original, entry + 8) as usize;
        let len = u64_at(original, entry + 16) as usize;
        shuffled[offset..offset + len].copy_from_slice(&original[start..start + len]);
        shuffled[entry + 8..entry + 16].copy_from_slice(&(offset as u64).to_le_bytes());
        offset += len;
    }
    assert_eq!(offset, run.term_offset);
    assert!(u64_at(&shuffled, run.term_offset + 8) > 0);
    (index, shuffled)
}

#[test]
fn physical_order_admission_preserves_shuffled_term_bytes_and_scoring() {
    let (original, shuffled) = shuffled_partition_fixture();
    let mut reopened = SeismicIndex::parse(original.bytes.clone(), 2, 2).unwrap();
    for id in 0..PARTITIONS {
        let bytes = if id == 0 {
            OwnedBytes::new(shuffled.clone())
        } else {
            original.partitions[id].as_ref().unwrap().bytes.clone()
        };
        reopened.attach_partition(id, bytes, 0).unwrap();
    }
    reopened.finish_partitions().unwrap();
    assert_eq!(reopened.pending_terms(), original.pending_terms());
    assert_eq!(reopened.bytes.as_slice(), original.bytes.as_slice());
    assert_eq!(
        reopened.partitions[0].as_ref().unwrap().bytes.as_slice(),
        shuffled
    );
    for dim in [0, 16, 32] {
        assert_eq!(
            term_payloads(&original, 0, dim),
            term_payloads(&reopened, 0, dim)
        );
        let before = original.term_runs(dim).next().unwrap();
        let after = reopened.term_runs(dim).next().unwrap();
        let mut expected = vec![0.0; before.cluster_count()];
        let mut actual = vec![0.0; after.cluster_count()];
        before.score_summaries(&[(0, 0.5), (16, 2.0), (32, 1.0)], &mut expected);
        after.score_summaries(&[(0, 0.5), (16, 2.0), (32, 1.0)], &mut actual);
        assert_eq!(actual, expected);
        assert_eq!(
            before
                .clusters()
                .flat_map(|c| c.rows().collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            after
                .clusters()
                .flat_map(|c| c.rows().collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        );
    }
}

#[test]
fn opening_nomination_runs_does_not_read_summary_payloads() {
    let (original, mut shuffled) = shuffled_partition_fixture();
    let run = &original.partitions[0].as_ref().unwrap().runs[0];
    let entry = run.term_offset;
    let start = u64_at(&shuffled, entry + 8) as usize;
    let len = u64_at(&shuffled, entry + 16) as usize;
    // The normal reader trusts writer-owned payloads. Opening only needs the
    // run/term directories; summary format and contents are not inspected.
    shuffled[start..start + len].fill(255);
    let mut root = SeismicIndex::parse(original.bytes.clone(), 2, 2).unwrap();
    root.attach_partition(0, OwnedBytes::new(shuffled), 0)
        .unwrap();
    assert!(root.partitions[0].is_some());
}
