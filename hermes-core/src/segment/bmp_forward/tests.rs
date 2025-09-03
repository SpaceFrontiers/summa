use super::*;
use crate::directories::{Directory, DirectoryWriter, FileHandle, RamDirectory};
use crate::segment::{BmpIndex, OffsetWriter};
use std::collections::BTreeMap;
use std::path::Path;

fn fixture(docs: u32) -> BmpIndex {
    fixture_with_duplicates(docs, true)
}

fn fixture_with_duplicates(docs: u32, duplicates: bool) -> BmpIndex {
    fixture_with_storage(docs, duplicates, true)
}

fn fixture_with_storage(docs: u32, duplicates: bool, store_forward: bool) -> BmpIndex {
    let mut postings = rustc_hash::FxHashMap::default();
    let mut vectors = 0;
    for doc in 0..docs {
        for ordinal in [0, 2].into_iter().take(if doc % 2 == 0 { 2 } else { 1 }) {
            vectors += 1;
            postings
                .entry(doc % 16)
                .or_insert_with(Vec::new)
                .push((doc, ordinal, 1.0));
            postings
                .entry(31)
                .or_insert_with(Vec::new)
                .push((doc, ordinal, 0.1));
        }
    }
    if duplicates {
        postings.get_mut(&0).unwrap().insert(1, (0, 0, 0.3));
    }
    let mut bytes = Vec::new();
    crate::segment::builder::bmp::build_bmp_blob(
        postings,
        32,
        4,
        0.0,
        None,
        32,
        5.0,
        0,
        store_forward,
        &mut bytes,
    )
    .unwrap();
    parse(bytes, docs, vectors)
}

fn parse(bytes: Vec<u8>, docs: u32, vectors: u32) -> BmpIndex {
    let len = bytes.len() as u64;
    BmpIndex::parse(
        FileHandle::from_bytes(OwnedBytes::new(bytes)),
        0,
        len,
        docs,
        vectors,
    )
    .unwrap()
}

fn legacy(source: &BmpIndex, docs: u32) -> BmpIndex {
    let mut bytes = source.read_raw_blob().unwrap().to_vec();
    let footer = bytes.len() - crate::segment::format::BMP_BLOB_FOOTER_SIZE;
    let start = u64::from_le_bytes(bytes[footer + 60..footer + 68].try_into().unwrap()) as usize
        + source.num_virtual_docs as usize * 6;
    bytes.drain(start..footer);
    let end = bytes.len();
    bytes[end - 4..].copy_from_slice(&crate::segment::format::BMP_BLOB_MAGIC_V19.to_le_bytes());
    parse(bytes, docs, source.total_vectors)
}

fn inverted_values(source: &BmpIndex) -> BTreeMap<(u32, u16, u32), u32> {
    let mut result = BTreeMap::new();
    for block in 0..source.num_blocks {
        for (dim, _, postings) in source.iter_block_terms(block) {
            for posting in postings {
                let (doc, ordinal) = source
                    .virtual_to_doc(block * source.bmp_block_size + u32::from(posting.local_slot));
                *result.entry((doc, ordinal, dim)).or_default() += u32::from(posting.impact);
            }
        }
    }
    result
}

fn forward_values(source: &BmpForward) -> BTreeMap<(u32, u16, u32), u32> {
    let mut result = BTreeMap::new();
    for i in 0..source.len() {
        let key = source.key(i);
        for (dim, impact) in source.vector(i).unwrap().iter() {
            *result.entry((key.doc, key.ordinal, dim)).or_default() += u32::from(impact);
        }
    }
    result
}

#[test]
fn forward_values_preserve_quantized_postings_duplicates_and_real_ordinals() {
    let bmp = fixture(137);
    let forward = bmp.forward().unwrap();
    assert_eq!(forward_values(forward), inverted_values(&bmp));
    assert_eq!(forward.find(LogicalUnit { doc: 0, ordinal: 1 }), None);
    assert_eq!(
        forward.find(LogicalUnit {
            doc: 200,
            ordinal: 0
        }),
        None
    );
    assert!(forward.find(LogicalUnit { doc: 0, ordinal: 2 }).is_some());
}

#[test]
fn legacy_and_forward_candidate_scores_sum_all_duplicate_dimension_impacts() {
    let current = fixture(137);
    let old = legacy(&current, 137);
    let query = [(0, 1.0), (31, 0.3)];
    let forward = crate::query::bmp::score_bmp_candidates(&current, &query, &[0]).unwrap();
    let inverted = crate::query::bmp::score_bmp_candidates(&old, &query, &[0]).unwrap();
    assert_eq!(forward, inverted);
}

#[test]
fn forward_candidate_scores_preserve_repeated_query_dimensions() {
    let current = fixture(137);
    let old = legacy(&current, 137);
    // Both the query and the document repeat dimension zero. Quantization is
    // per query entry, so combining raw query weights would change the score.
    let query = [(0, 1.0), (0, 0.3), (31, 0.2)];
    let forward = crate::query::bmp::score_bmp_candidates(&current, &query, &[0]).unwrap();
    let inverted = crate::query::bmp::score_bmp_candidates(&old, &query, &[0]).unwrap();
    assert_eq!(forward, inverted);
}

#[tokio::test]
async fn forward_merge_copies_payload_and_remaps_only_directory_with_document_gaps() {
    let a = fixture(17);
    let b = fixture(33);
    let dir = RamDirectory::new();
    let mut writer = OffsetWriter::new(
        dir.streaming_writer_cold(Path::new("forward"))
            .await
            .unwrap(),
    );
    assert!(write_forward_sources(&[(&a, 0), (&b, 25)], &[], &mut writer, None, None).unwrap());
    writer.finish().unwrap();
    let bytes = dir
        .open_read(Path::new("forward"))
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap();
    let result = BmpForward::parse(bytes, a.num_real_docs() + b.num_real_docs(), 58, 32).unwrap();
    let expected: Vec<_> = a
        .forward()
        .unwrap()
        .payload
        .as_slice()
        .iter()
        .chain(b.forward().unwrap().payload.as_slice())
        .copied()
        .collect();
    assert_eq!(result.payload.as_slice(), expected);
    assert!(result.for_document(20).next().is_none());
    let mut expected = forward_values(a.forward().unwrap());
    expected.extend(
        forward_values(b.forward().unwrap())
            .into_iter()
            .map(|((doc, ord, dim), impact)| ((doc + 25, ord, dim), impact)),
    );
    assert_eq!(forward_values(&result), expected);
}

#[tokio::test]
async fn legacy_migration_is_explicit_budgeted_and_preserves_forward_bytes() {
    let current = fixture(137);
    let old = legacy(&current, 137);
    assert!(old.forward().is_none());
    assert_eq!(inverted_values(&old), inverted_values(&current));
    let dir = RamDirectory::new();
    let mut writer = OffsetWriter::new(
        dir.streaming_writer_cold(Path::new("forward"))
            .await
            .unwrap(),
    );
    assert!(!write_forward_sources(&[(&old, 0)], &[], &mut writer, None, None).unwrap());
    assert_eq!(writer.offset(), 0);
    assert!(
        write_forward_sources(&[(&old, 0), (&current, 137)], &[], &mut writer, None, None).is_err()
    );
    assert!(write_forward_sources(&[(&old, 0)], &[], &mut writer, Some(1), None).is_err());
    assert_eq!(writer.offset(), 0);
    let cancel = std::sync::atomic::AtomicBool::new(true);
    assert!(matches!(
        write_forward_sources(&[(&current, 0)], &[], &mut writer, None, Some(&cancel)),
        Err(Error::IndexClosed)
    ));
    assert_eq!(writer.offset(), 0);
    assert!(
        write_forward_sources(&[(&old, 0)], &[], &mut writer, Some(1024 * 1024), None).unwrap()
    );
    writer.finish().unwrap();
    let bytes = dir
        .open_read(Path::new("forward"))
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap();
    let migrated = BmpForward::parse(bytes, current.num_real_docs(), 137, 32).unwrap();
    assert_eq!(
        migrated.payload.as_slice(),
        current.forward().unwrap().payload.as_slice()
    );
    assert_eq!(
        migrated.rows.as_slice(),
        current.forward().unwrap().rows.as_slice()
    );
}

#[tokio::test]
async fn record_and_block_bp_copy_forward_bytes_and_preserve_quantized_values() {
    use crate::segment::reorder::{BpGranularity, reorder_bmp_field};
    let source = fixture(513);
    let expected = inverted_values(&source);
    for granularity in [BpGranularity::Records, BpGranularity::Blocks] {
        for old in [false, true] {
            let input = if old {
                legacy(&source, 513)
            } else {
                source.clone()
            };
            let dir = RamDirectory::new();
            let temp = tempfile::tempdir().unwrap();
            let writer = OffsetWriter::new(
                dir.streaming_writer_cold(Path::new("sparse"))
                    .await
                    .unwrap(),
            );
            let (writer, _, _) = reorder_bmp_field(
                &[(input, 0)],
                0,
                "test",
                "sparse",
                32,
                32,
                4,
                5.0,
                source.total_vectors,
                64 * 1024 * 1024,
                Default::default(),
                None,
                granularity,
                temp.path().into(),
                true,
                writer,
                Vec::new(),
                None,
            )
            .unwrap();
            writer.finish().unwrap();
            let bytes = dir
                .open_read(Path::new("sparse"))
                .await
                .unwrap()
                .read_bytes()
                .await
                .unwrap();
            let result = parse(bytes.to_vec(), 513, source.total_vectors);
            assert_eq!(inverted_values(&result), expected);
            assert_eq!(
                result.forward().unwrap().payload.as_slice(),
                source.forward().unwrap().payload.as_slice()
            );
            assert_eq!(
                result.forward().unwrap().rows.as_slice(),
                source.forward().unwrap().rows.as_slice()
            );
        }
    }
}

#[test]
fn forward_directory_and_selected_payload_reject_corruption() {
    let bmp = fixture(3);
    let forward = bmp.forward().unwrap();
    let encode = |payload: &[u8], rows: &[u8]| {
        let mut bytes = payload.to_vec();
        bytes.extend_from_slice(rows);
        bytes.extend_from_slice(&forward.len().to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        OwnedBytes::new(bytes)
    };
    let mut rows = forward.rows.to_vec();
    rows[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(
        BmpForward::parse(
            encode(forward.payload.as_slice(), &rows),
            forward.len(),
            3,
            32
        )
        .is_err()
    );
    let mut payload = forward.payload.to_vec();
    payload[..4].copy_from_slice(&32u32.to_le_bytes());
    let parsed = BmpForward::parse(
        encode(&payload, forward.rows.as_slice()),
        forward.len(),
        3,
        32,
    )
    .unwrap();
    assert!(parsed.vector(0).is_err());
}

#[test]
fn forward_storage_is_used_by_l1_and_record_bp_but_never_bmp_search() {
    let source = fixture(137);
    let mut bytes = source.read_raw_blob().unwrap().to_vec();
    let footer = bytes.len() - crate::segment::format::BMP_BLOB_FOOTER_SIZE;
    let payload = u64::from_le_bytes(bytes[footer + 60..footer + 68].try_into().unwrap()) as usize
        + source.num_virtual_docs as usize * 6;
    // Corrupt only the forward payload, leaving every inverted representation
    // valid. Its consumers must report corruption instead of changing readers.
    bytes[payload..payload + 4].copy_from_slice(&32u32.to_le_bytes());
    let corrupt = parse(bytes, 137, source.total_vectors);
    let query = [(0, 1.0), (31, 0.3)];
    assert!(matches!(
        crate::query::bmp::score_bmp_candidates(&corrupt, &query, &[0]),
        Err(Error::Corruption(_))
    ));
    assert!(matches!(
        crate::segment::builder::graph_bisection::build_forward_index_from_bmps(
            &[&corrupt],
            1,
            137,
            128 * 1024 * 1024,
        ),
        Err(Error::Corruption(_))
    ));
    let search = |bmp| {
        crate::query::bmp::execute_bmp_with_threshold(
            bmp,
            "test",
            "sparse",
            &query,
            &query,
            10,
            1.0,
            0,
            None,
            Default::default(),
        )
        .unwrap()
        .into_iter()
        .map(|hit| (hit.doc_id, hit.ordinal, hit.score.to_bits()))
        .collect::<Vec<_>>()
    };
    assert_eq!(search(&corrupt), search(&source));
}

#[test]
fn forward_copy_propagates_partial_writer_failure_and_mid_payload_cancellation() {
    use crate::directories::StreamingWriter;
    use std::io::{self, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    struct InterruptWriter {
        written: u64,
        fail_after: u64,
        cancel: Option<Arc<AtomicBool>>,
    }
    impl Write for InterruptWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.written == self.fail_after {
                return Err(io::Error::other("injected forward output failure"));
            }
            let count = bytes.len().min((self.fail_after - self.written) as usize);
            self.written += count as u64;
            if let Some(cancel) = &self.cancel {
                cancel.store(true, Ordering::Release);
            }
            Ok(count)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl StreamingWriter for InterruptWriter {
        fn finish(self: Box<Self>) -> io::Result<()> {
            Ok(())
        }
        fn bytes_written(&self) -> u64 {
            self.written
        }
    }
    let source = fixture(137);
    let mut writer = OffsetWriter::new(Box::new(InterruptWriter {
        written: 0,
        fail_after: 7,
        cancel: None,
    }));
    let error = write_forward_sources(&[(&source, 0)], &[], &mut writer, None, None).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected forward output failure")
    );
    assert_eq!(writer.offset(), 7);
    let cancel = Arc::new(AtomicBool::new(false));
    let mut writer = OffsetWriter::new(Box::new(InterruptWriter {
        written: 0,
        fail_after: u64::MAX,
        cancel: Some(cancel.clone()),
    }));
    assert!(matches!(
        write_forward_sources(&[(&source, 0)], &[], &mut writer, None, Some(&cancel)),
        Err(Error::IndexClosed)
    ));
    assert!(writer.offset() > 0);
    assert!(
        writer.offset() < source.forward().unwrap().encoded_bytes() as u64,
        "a cancelled copy cannot publish a complete directory"
    );
}

#[tokio::test]
async fn forward_duplicate_impacts_survive_copy_merge_and_legacy_upgrade() {
    for duplicate in [false, true] {
        let a = fixture_with_duplicates(17, false);
        let b = fixture_with_duplicates(17, duplicate);
        for migrate in [false, true] {
            let old = legacy(&b, 17);
            let source = if migrate { &old } else { &b };
            let dir = RamDirectory::new();
            let mut writer = OffsetWriter::new(
                dir.streaming_writer_cold(Path::new("forward"))
                    .await
                    .unwrap(),
            );
            write_forward_sources(
                &[(&a, 0), (source, 17)],
                &[],
                &mut writer,
                migrate.then_some(1024 * 1024),
                None,
            )
            .unwrap();
            writer.finish().unwrap();
            let bytes = dir
                .open_read(Path::new("forward"))
                .await
                .unwrap()
                .read_bytes()
                .await
                .unwrap();
            let forward =
                BmpForward::parse(bytes, a.num_real_docs() + b.num_real_docs(), 34, 32).unwrap();
            let expected: Vec<_> = a
                .forward()
                .unwrap()
                .payload
                .as_slice()
                .iter()
                .chain(b.forward().unwrap().payload.as_slice())
                .copied()
                .collect();
            assert_eq!(forward.payload.as_slice(), expected);
            forward.validate_payload().unwrap();
        }
    }
}

#[tokio::test]
async fn bp_can_omit_forward_storage_without_changing_inverted_values() {
    use crate::segment::reorder::{BpGranularity, reorder_bmp_field};
    let current = fixture(137);
    let old = legacy(&current, 137);
    for source in [&current, &old] {
        for granularity in [BpGranularity::Records, BpGranularity::Blocks] {
            let dir = RamDirectory::new();
            let writer = OffsetWriter::new(
                dir.streaming_writer_cold(Path::new("sparse"))
                    .await
                    .unwrap(),
            );
            let scratch = tempfile::tempdir().unwrap();
            let (writer, _, _) = reorder_bmp_field(
                &[(source.clone(), 0)],
                0,
                "optional",
                "sparse",
                32,
                32,
                4,
                5.0,
                source.total_vectors,
                128 * 1024 * 1024,
                Default::default(),
                None,
                granularity,
                scratch.path().join("bp"),
                false,
                writer,
                Vec::new(),
                None,
            )
            .unwrap();
            writer.finish().unwrap();
            let bytes = dir
                .open_read(Path::new("sparse"))
                .await
                .unwrap()
                .read_bytes()
                .await
                .unwrap();
            let result = parse(bytes.to_vec(), 137, source.total_vectors);
            assert!(result.forward().is_none());
            assert_eq!(inverted_values(&result), inverted_values(source));
            if granularity == BpGranularity::Blocks {
                let mut before: Vec<_> = (0..source.num_blocks)
                    .map(|block| {
                        let (start, end) = source.block_data_range(block);
                        source.read_raw_blob().unwrap().as_slice()[start as usize..end as usize]
                            .to_vec()
                    })
                    .collect();
                let mut after: Vec<_> = (0..result.num_blocks)
                    .map(|block| {
                        let (start, end) = result.block_data_range(block);
                        result.read_raw_blob().unwrap().as_slice()[start as usize..end as usize]
                            .to_vec()
                    })
                    .collect();
                before.sort();
                after.sort();
                assert_eq!(before, after);
            }
        }
    }
}

#[test]
fn disabling_forward_storage_preserves_every_inverted_byte() {
    let enabled = fixture_with_storage(137, true, true);
    let disabled = fixture_with_storage(137, true, false);
    assert!(disabled.forward().is_none());
    assert_eq!(
        disabled.read_raw_blob().unwrap().as_slice(),
        legacy(&enabled, 137).read_raw_blob().unwrap().as_slice()
    );
    assert_eq!(
        enabled.read_raw_blob().unwrap().len() - disabled.read_raw_blob().unwrap().len(),
        enabled.forward().unwrap().encoded_bytes()
    );
}
