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

fn without_forward(source: &BmpIndex, docs: u32) -> BmpIndex {
    let mut bytes = source.read_raw_blob().unwrap().to_vec();
    let footer = bytes.len() - crate::segment::format::BMP_BLOB_FOOTER_SIZE;
    let start = u64::from_le_bytes(bytes[footer + 60..footer + 68].try_into().unwrap()) as usize
        + source.num_virtual_docs as usize * 6;
    bytes.drain(start..footer);
    let mut marker = Vec::new();
    write_disabled(&mut marker).unwrap();
    bytes.splice(start..start, marker);
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
fn disabled_forward_storage_keeps_the_current_bmp_envelope() {
    let source = fixture_with_storage(137, true, false);
    let bytes = source.read_raw_blob().unwrap();
    assert_eq!(
        u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap()),
        crate::segment::format::BMP_BLOB_MAGIC,
    );
    assert!(source.forward().is_none());
}

#[test]
fn current_bmp_reader_rejects_a_legacy_envelope() {
    let current = fixture(137);
    let old = without_forward(&current, 137);
    let mut bytes = old.read_raw_blob().unwrap().to_vec();
    let footer = bytes.len() - crate::segment::format::BMP_BLOB_FOOTER_SIZE;
    bytes.drain(footer - TRAILER_BYTES..footer);
    let end = bytes.len();
    bytes[end - 4..].copy_from_slice(&0x39504d42u32.to_le_bytes());
    let bytes = OwnedBytes::new(bytes);
    let result = BmpIndex::parse(
        FileHandle::from_bytes(bytes.clone()),
        0,
        bytes.len() as u64,
        137,
        old.total_vectors,
    );
    assert!(result.is_err(), "legacy BMP must require offline migration");
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
fn inverted_and_forward_candidate_scores_sum_all_duplicate_dimension_impacts() {
    let current = fixture(137);
    let old = without_forward(&current, 137);
    let query = [(0, 1.0), (31, 0.3)];
    let forward = crate::query::bmp::score_bmp_candidates(&current, &query, &[0]).unwrap();
    let inverted = crate::query::bmp::score_bmp_candidates(&old, &query, &[0]).unwrap();
    assert_eq!(forward, inverted);
}

#[test]
fn forward_candidate_scores_preserve_repeated_query_dimensions() {
    let current = fixture(137);
    let old = without_forward(&current, 137);
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
    assert!(
        write_forward_sources(&[(&a, 0), (&b, 25)], &[], &mut writer, None, None, true).unwrap()
    );
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
async fn enabling_forward_storage_is_explicit_budgeted_and_preserves_forward_bytes() {
    let current = fixture(137);
    let old = without_forward(&current, 137);
    assert!(old.forward().is_none());
    assert_eq!(inverted_values(&old), inverted_values(&current));
    let dir = RamDirectory::new();
    let mut writer = OffsetWriter::new(
        dir.streaming_writer_cold(Path::new("forward"))
            .await
            .unwrap(),
    );
    assert!(!write_forward_sources(&[(&old, 0)], &[], &mut writer, None, None, true).unwrap());
    assert_eq!(writer.offset(), TRAILER_BYTES as u64);
    writer.finish().unwrap();
    let disabled = dir
        .open_read(Path::new("forward"))
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap();
    assert!(
        BmpForward::parse_optional(disabled, current.num_real_docs(), 137, 32)
            .unwrap()
            .is_none()
    );
    let mut writer = OffsetWriter::new(
        dir.streaming_writer_cold(Path::new("materialized"))
            .await
            .unwrap(),
    );
    assert!(
        write_forward_sources(
            &[(&old, 0), (&current, 137)],
            &[],
            &mut writer,
            None,
            None,
            true
        )
        .is_err()
    );
    assert!(write_forward_sources(&[(&old, 0)], &[], &mut writer, Some(1), None, true).is_err());
    assert_eq!(writer.offset(), 0);
    let cancel = std::sync::atomic::AtomicBool::new(true);
    assert!(matches!(
        write_forward_sources(
            &[(&current, 0)],
            &[],
            &mut writer,
            None,
            Some(&cancel),
            true
        ),
        Err(Error::IndexClosed)
    ));
    assert_eq!(writer.offset(), 0);
    assert!(
        write_forward_sources(
            &[(&old, 0)],
            &[],
            &mut writer,
            Some(1024 * 1024),
            None,
            true
        )
        .unwrap()
    );
    writer.finish().unwrap();
    let bytes = dir
        .open_read(Path::new("materialized"))
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
                without_forward(&source, 513)
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
    payload[4..6].copy_from_slice(&32u16.to_le_bytes());
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
fn explicit_forward_integrity_and_record_bp_reject_corruption_while_bmp_search_ignores_forward() {
    let source = fixture(137);
    let mut bytes = source.read_raw_blob().unwrap().to_vec();
    let footer = bytes.len() - crate::segment::format::BMP_BLOB_FOOTER_SIZE;
    let payload = u64::from_le_bytes(bytes[footer + 60..footer + 68].try_into().unwrap()) as usize
        + source.num_virtual_docs as usize * 6;
    // Corrupt only the forward payload, leaving every inverted representation
    // valid. Explicit integrity and BP validate it; ordinary BMP never reads it.
    bytes[payload + 4..payload + 6].copy_from_slice(&32u16.to_le_bytes());
    let corrupt = parse(bytes, 137, source.total_vectors);
    let query = [(0, 1.0), (31, 0.3)];
    assert!(matches!(
        corrupt.forward().unwrap().validate_payload(&|| Ok(())),
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

/// Run alone in release mode; this is measurement evidence, not a latency gate.
#[test]
#[ignore = "manual fixed-fixture BMP performance comparison"]
fn measure_forward_candidate_scoring_and_bp_graph() {
    use crate::segment::builder::graph_bisection::build_forward_index_from_bmps;
    use std::hint::black_box;
    use std::time::Instant;
    const DOCS: u32 = 16_384;
    const DIMS: u32 = 4096;
    const NNZ: u32 = 64;
    let mut postings = rustc_hash::FxHashMap::default();
    for doc in 0..DOCS {
        for k in 0..NNZ {
            let dim = ((doc * 37) + k * 61) % DIMS;
            postings.entry(dim).or_insert_with(Vec::new).push((
                doc,
                0,
                0.2 + (k % 13) as f32 * 0.1,
            ));
        }
    }
    let build = Instant::now();
    let mut bytes = Vec::new();
    crate::segment::builder::bmp::build_bmp_blob(
        postings, 128, 4, 0.0, None, DIMS, 5.0, 0, true, &mut bytes,
    )
    .unwrap();
    let build_ms = build.elapsed().as_secs_f64() * 1000.0;
    let current = parse(bytes, DOCS, DOCS);
    let old = without_forward(&current, DOCS);
    let query: Vec<_> = (0..64).map(|i| (i * 61, 0.7)).collect();
    let targets: Vec<_> = (0..128).map(|i| i * 127).collect();
    assert_eq!(
        crate::query::bmp::score_bmp_candidates(&old, &query, &targets).unwrap(),
        crate::query::bmp::score_bmp_candidates(&current, &query, &targets).unwrap()
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut graph_shape = None;
    eprintln!(
        "fixture: docs={DOCS} dims={DIMS} nnz={NNZ} build_ms={build_ms:.3} v19_bytes={} v20_bytes={} forward_bytes={}",
        old.read_raw_blob().unwrap().len(),
        current.read_raw_blob().unwrap().len(),
        current.forward().unwrap().encoded_bytes()
    );
    for (label, bmp) in [("without forward", &old), ("with forward", &current)] {
        let mut score_times = Vec::new();
        let mut graph_times = Vec::new();
        for _ in 0..11 {
            let started = Instant::now();
            for _ in 0..100 {
                black_box(
                    crate::query::bmp::score_bmp_candidates(
                        bmp,
                        black_box(&query),
                        black_box(&targets),
                    )
                    .unwrap(),
                );
            }
            score_times.push(started.elapsed().as_secs_f64() * 1_000_000.0 / 100.0);
            let started = Instant::now();
            let graph = pool
                .install(|| {
                    build_forward_index_from_bmps(&[bmp], 1, DOCS as usize, 128 * 1024 * 1024)
                })
                .unwrap();
            graph_times.push(started.elapsed().as_secs_f64() * 1000.0);
            let shape = (
                graph.num_docs(),
                graph.num_terms,
                graph.total_postings(),
                graph.budget_limited(),
            );
            if let Some(expected) = graph_shape {
                assert_eq!(shape, expected);
            } else {
                graph_shape = Some(shape);
            }
            assert_eq!(graph.total_postings(), u64::from(DOCS * NNZ));
            black_box(graph);
        }
        score_times.sort_by(f64::total_cmp);
        graph_times.sort_by(f64::total_cmp);
        eprintln!(
            "{label}: score_us median={:.3} min={:.3} max={:.3}; graph_ms median={:.3} min={:.3} max={:.3}; graph_terms_bytes={} graph_offsets_bytes={}",
            score_times[5],
            score_times[0],
            score_times[10],
            graph_times[5],
            graph_times[0],
            graph_times[10],
            DOCS * NNZ * 4,
            (u64::from(DOCS) + 1) * 8
        );
    }
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
    let error =
        write_forward_sources(&[(&source, 0)], &[], &mut writer, None, None, true).unwrap_err();
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
        write_forward_sources(&[(&source, 0)], &[], &mut writer, None, Some(&cancel), true),
        Err(Error::IndexClosed)
    ));
    assert!(writer.offset() > 0);
    assert!(
        writer.offset() < source.forward().unwrap().encoded_bytes() as u64,
        "a cancelled copy cannot publish a complete directory"
    );
}

#[tokio::test]
async fn forward_duplicate_impacts_survive_copy_merge_and_storage_enablement() {
    for duplicate in [false, true] {
        let a = fixture_with_duplicates(17, false);
        let b = fixture_with_duplicates(17, duplicate);
        for migrate in [false, true] {
            let old = without_forward(&b, 17);
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
                true,
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
            forward.validate_payload(&|| Ok(())).unwrap();
        }
    }
}

#[tokio::test]
async fn bp_can_omit_forward_storage_without_changing_inverted_values() {
    use crate::segment::reorder::{BpGranularity, reorder_bmp_field};
    let current = fixture(137);
    let old = without_forward(&current, 137);
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
        without_forward(&enabled, 137)
            .read_raw_blob()
            .unwrap()
            .as_slice()
    );
    assert_eq!(
        enabled.read_raw_blob().unwrap().len() - disabled.read_raw_blob().unwrap().len(),
        enabled.forward().unwrap().encoded_bytes() - TRAILER_BYTES
    );
}

#[test]
fn optional_storage_rejects_missing_markers_unknown_flags_and_hidden_payloads() {
    let source = fixture_with_storage(137, true, false);
    let bytes = source.read_raw_blob().unwrap().to_vec();
    let trailer = bytes.len() - crate::segment::format::BMP_BLOB_FOOTER_SIZE - TRAILER_BYTES;
    let reject = |bytes: Vec<u8>| {
        let len = bytes.len() as u64;
        assert!(
            BmpIndex::parse(
                FileHandle::from_bytes(OwnedBytes::new(bytes)),
                0,
                len,
                137,
                source.total_vectors
            )
            .is_err()
        );
    };
    for (offset, value) in [(0, 1u32), (4, 2), (4, 3), (8, 5)] {
        let mut corrupt = bytes.clone();
        corrupt[trailer + offset..trailer + offset + 4].copy_from_slice(&value.to_le_bytes());
        reject(corrupt);
    }
    let mut absent = bytes.clone();
    absent.drain(trailer..trailer + TRAILER_BYTES);
    reject(absent);
    let mut hidden = bytes;
    hidden.insert(trailer, 0);
    reject(hidden);

    let enabled = fixture(137);
    let mut payload = enabled.read_raw_blob().unwrap().to_vec();
    let flags = payload.len() - crate::segment::format::BMP_BLOB_FOOTER_SIZE - TRAILER_BYTES + 4;
    payload[flags..flags + 4].copy_from_slice(&STORAGE_DISABLED.to_le_bytes());
    reject(payload);
}

#[test]
fn empty_bmp_blobs_validate_their_storage_section_and_footer() {
    let mut bytes = Vec::new();
    write_disabled(&mut bytes).unwrap();
    crate::segment::builder::bmp::write_bmp_footer(
        &mut bytes, 0, 0, 0, 0, 0, 0, 32, 32, 0, 5.0, 0, 0, 4,
    )
    .unwrap();
    let read = |bytes: Vec<u8>| {
        let len = bytes.len() as u64;
        BmpIndex::parse(
            FileHandle::from_bytes(OwnedBytes::new(bytes)),
            0,
            len,
            137,
            0,
        )
    };
    let empty = read(bytes.clone()).unwrap();
    assert_eq!(empty.num_blocks, 0);
    assert!(empty.forward().is_none());
    for offset in [
        4,
        TRAILER_BYTES,
        TRAILER_BYTES + 8,
        TRAILER_BYTES + 16,
        TRAILER_BYTES + 52,
    ] {
        let mut corrupt = bytes.clone();
        corrupt[offset] = 2;
        assert!(read(corrupt).is_err());
    }
    let mut missing_section = bytes.clone();
    missing_section.drain(..TRAILER_BYTES);
    assert!(read(missing_section).is_err());
    let mut hidden_payload = bytes.clone();
    hidden_payload.insert(0, 0);
    assert!(read(hidden_payload).is_err());
    bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
    let empty = read(bytes).unwrap();
    assert_eq!(empty.forward().unwrap().len(), 0);
}

#[test]
fn bmpb_forward_storage_has_stable_encoded_bytes() {
    let source = fixture(137);
    let bytes = source.read_raw_blob().unwrap();
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    assert_eq!((bytes.len(), hash), (13_535, 0x17452c22d28d5c37));
}

#[test]
fn forward_construction_cancels_during_validation_counting_and_csr_fill() {
    use crate::segment::builder::graph_bisection::{
        build_forward_index_from_bmps_with_maps, build_vid_maps,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    for storage in [false, true] {
        let bmp = fixture_with_storage(128, false, storage);
        let maps = vec![build_vid_maps(&bmp, &|| Ok(())).unwrap()];
        let checks = AtomicUsize::new(0);
        let complete = build_forward_index_from_bmps_with_maps(
            &[&bmp],
            &maps,
            1,
            1000,
            16 * 1024 * 1024,
            &|| {
                checks.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        )
        .unwrap();
        assert!(complete.total_postings() > 0);
        let total = checks.load(Ordering::Relaxed);
        assert!(total > 4, "forward construction never checks cancellation");
        for stop in [1, total / 4, total / 2, total - 1] {
            checks.store(0, Ordering::Relaxed);
            let result = build_forward_index_from_bmps_with_maps(
                &[&bmp],
                &maps,
                1,
                1000,
                16 * 1024 * 1024,
                &|| {
                    if checks.fetch_add(1, Ordering::Relaxed) >= stop {
                        Err(crate::Error::IndexClosed)
                    } else {
                        Ok(())
                    }
                },
            );
            assert!(
                matches!(result, Err(crate::Error::IndexClosed)),
                "cancelled construction produced a graph at checkpoint {stop}/{total}"
            );
        }
    }
}

#[test]
fn block_forward_construction_cancels_before_returning_a_partial_graph() {
    use crate::segment::builder::graph_bisection::build_forward_index_from_blocks;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut postings = rustc_hash::FxHashMap::default();
    for doc in 0..2048 {
        postings
            .entry(doc / 128)
            .or_insert_with(Vec::new)
            .push((doc, 0, 1.0));
    }
    let mut bytes = Vec::new();
    crate::segment::builder::bmp::build_bmp_blob(
        postings, 32, 4, 0.0, None, 32, 5.0, 0, true, &mut bytes,
    )
    .unwrap();
    let bmp = parse(bytes, 2048, 2048);
    let checks = AtomicUsize::new(0);
    let complete = build_forward_index_from_blocks(&[&bmp], 16 * 1024 * 1024, &|| {
        checks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    })
    .unwrap();
    assert!(complete.num_docs() > 0);
    let total = checks.load(Ordering::Relaxed);
    for stop in [1, total / 4, total / 2, total - 1] {
        checks.store(0, Ordering::Relaxed);
        let result = build_forward_index_from_blocks(&[&bmp], 16 * 1024 * 1024, &|| {
            if checks.fetch_add(1, Ordering::Relaxed) >= stop {
                Err(crate::Error::IndexClosed)
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(crate::Error::IndexClosed)));
    }
}

#[test]
fn rewrite_document_map_scan_observes_cancellation_mid_scan() {
    use crate::segment::builder::graph_bisection::build_vid_maps;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let bmp = fixture(2048);
    let checks = AtomicUsize::new(0);
    let result = build_vid_maps(&bmp, &|| {
        if checks.fetch_add(1, Ordering::Relaxed) >= 3 {
            Err(crate::Error::IndexClosed)
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(crate::Error::IndexClosed)));
}

fn gap_fixture() -> BmpIndex {
    let mut postings = rustc_hash::FxHashMap::default();
    for i in 0..137 {
        let entries = postings.entry(65_530 + i * 3).or_insert_with(Vec::new);
        entries.push((0, 0, 1.0));
        entries.push((2, 3, 2.0));
        if i == 3 {
            entries.insert(1, (0, 0, 0.5));
        }
    }
    let mut bytes = Vec::new();
    crate::segment::builder::bmp::build_bmp_blob(
        postings, 32, 4, 0.0, None, 100_000, 5.0, 0, true, &mut bytes,
    )
    .unwrap();
    parse(bytes, 3, 2)
}

#[test]
fn forward_gap_packets_preserve_wide_ids_duplicates_and_integer_scores() {
    let bmp = gap_fixture();
    let forward = bmp.forward().unwrap();
    assert_eq!(forward_values(forward), inverted_values(&bmp));
    assert!(forward.payload.len() < (137 * 2 + 1) * 3);
    let inverted = without_forward(&bmp, 3);
    let query = [(65_539, 0.5), (65_539, 0.75), (65_938, 1.0)];
    assert_eq!(
        crate::query::bmp::score_bmp_candidates(&bmp, &query, &[0, 1]).unwrap(),
        crate::query::bmp::score_bmp_candidates(&inverted, &query, &[0, 1]).unwrap(),
    );
}

#[test]
fn bmpa_envelope_is_rejected_before_forward_payload_access() {
    let bmp = gap_fixture();
    let mut bytes = bmp.read_raw_blob().unwrap().to_vec();
    let end = bytes.len();
    bytes[end - 4..].copy_from_slice(&0x41504d42u32.to_le_bytes());
    assert!(
        BmpIndex::parse(
            FileHandle::from_bytes(OwnedBytes::new(bytes)),
            0,
            end as u64,
            3,
            2,
        )
        .is_err()
    );
}

#[tokio::test]
async fn gap_encoded_rows_survive_copy_merge_and_materialization_byte_identically() {
    let source = gap_fixture();
    let absent = without_forward(&source, 3);
    for input in [&source, &absent] {
        let dir = RamDirectory::new();
        let mut writer = OffsetWriter::new(
            dir.streaming_writer_cold(Path::new("forward"))
                .await
                .unwrap(),
        );
        write_forward_sources(
            &[(input, 0), (&source, 4)],
            &[],
            &mut writer,
            Some(1024 * 1024),
            None,
            true,
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
        let result = BmpForward::parse(bytes, 4, 7, 100_000).unwrap();
        let payload = source.forward().unwrap().payload.as_slice();
        assert_eq!(result.payload.as_slice(), [payload, payload].concat());
        result.validate_payload(&|| Ok(())).unwrap();
        assert_eq!(result.key(3), LogicalUnit { doc: 6, ordinal: 3 });
    }
}

#[tokio::test]
async fn gap_encoded_survivors_are_copied_without_reencoding_on_compaction() {
    let source = gap_fixture();
    let rows = crate::segment::row_map::RowMap::new(3, 1, |doc| doc == 2, 1024).unwrap();
    let dir = RamDirectory::new();
    let mut writer = OffsetWriter::new(
        dir.streaming_writer_cold(Path::new("forward"))
            .await
            .unwrap(),
    );
    write_compacted_forward(&source, &rows, &mut writer, None).unwrap();
    writer.finish().unwrap();
    let bytes = dir
        .open_read(Path::new("forward"))
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap();
    let result = BmpForward::parse(bytes, 1, 1, 100_000).unwrap();
    let forward = source.forward().unwrap();
    assert_eq!(
        result.payload.as_slice(),
        &forward.payload.as_slice()[forward.offset(1) as usize..]
    );
    assert_eq!(result.key(0), LogicalUnit { doc: 0, ordinal: 3 });
    assert_eq!(
        result.vector(0).unwrap().iter().collect::<Vec<_>>(),
        forward.vector(1).unwrap().iter().collect::<Vec<_>>()
    );
}
