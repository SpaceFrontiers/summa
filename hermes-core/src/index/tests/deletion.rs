use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig};
use crate::query::AllQuery;

#[tokio::test]
async fn mutation_merge_and_compaction_sequences_match_committed_rows_through_abort_and_reopen() {
    use std::collections::BTreeMap;
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    let body = schema.add_text_field("body", true, false);
    let version = schema.add_u64_field("version", false, false);
    schema.set_fast(version, true);
    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::NoMergePolicy),
        ..Default::default()
    };
    let index = Index::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    let reader = index.reader().await.unwrap();
    let mut expected = BTreeMap::new();
    for cycle in 0..24u64 {
        let previous = reader.searcher().await.unwrap();
        let old_expected = expected.clone();
        let mut staged = expected.clone();
        for operation in 0..8u64 {
            let key = format!("key{}", (cycle * 13 + operation * 7) % 41);
            let revision = cycle * 8 + operation;
            let mut doc = Document::new();
            doc.add_text(id, key.clone());
            doc.add_text(body, format!("present revision{revision}"));
            doc.add_u64(version, revision);
            match (cycle + operation) % 3 {
                0 => {
                    writer.delete_primary_key(&key).unwrap();
                    staged.remove(&key);
                }
                1 => {
                    writer.upsert_document(doc).unwrap();
                    staged.insert(key, revision);
                }
                _ => {
                    if let std::collections::btree_map::Entry::Vacant(entry) = staged.entry(key) {
                        writer.add_document(doc).unwrap();
                        entry.insert(revision);
                    } else {
                        assert!(matches!(
                            writer.add_document(doc),
                            Err(crate::Error::DuplicatePrimaryKey(_))
                        ));
                    }
                }
            }
        }
        if cycle % 5 == 0 {
            writer.prepare_commit().await.unwrap().abort();
        } else {
            writer.commit().await.unwrap();
            expected = staged;
        }
        match cycle % 4 {
            0 => writer.force_merge().await.unwrap(),
            1 => writer.force_merge_with_compaction(true).await.unwrap(),
            2 => {
                writer.compact(16 * 1024 * 1024).await.unwrap();
            }
            _ => {}
        }
        reader.reload().await.unwrap();
        let current = reader.searcher().await.unwrap();
        for (snapshot, model) in [(&previous, &old_expected), (&current, &expected)] {
            let mut actual = BTreeMap::new();
            let hits = snapshot
                .search(&crate::query::TermQuery::new(body, "present"), 100)
                .await
                .unwrap();
            assert_eq!(snapshot.num_docs() as usize, model.len());
            for hit in hits {
                let doc = snapshot
                    .doc(hit.segment_id, hit.doc_id)
                    .await
                    .unwrap()
                    .unwrap();
                let key = doc.get_first(id).unwrap().as_text().unwrap().to_owned();
                assert!(
                    doc.get_first(body).is_none(),
                    "fixture must remain indexed-only"
                );
                let segment = snapshot.segment_map()[&hit.segment_id];
                let value = snapshot.segment_readers()[segment]
                    .fast_field(version.0)
                    .unwrap()
                    .get_u64(hit.doc_id);
                assert!(
                    actual.insert(key, value).is_none(),
                    "duplicate live key at cycle {cycle}"
                );
            }
            assert_eq!(&actual, model, "cycle {cycle}");
        }
    }
    writer.shutdown().await.unwrap();
    drop(writer);
    let reopened = Index::open(dir, config).await.unwrap();
    let mut writer = reopened.writer();
    writer.init_primary_key_dedup().await.unwrap();
    assert_eq!(
        reopened
            .reader()
            .await
            .unwrap()
            .searcher()
            .await
            .unwrap()
            .num_docs() as usize,
        expected.len()
    );
    for key in expected.keys() {
        let mut doc = Document::new();
        doc.add_text(id, key);
        assert!(matches!(
            writer.add_document(doc),
            Err(crate::Error::DuplicatePrimaryKey(_))
        ));
    }
}

#[tokio::test]
async fn ordinary_force_merge_preserves_tombstones_including_a_single_segment() {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    let index = Index::create(
        RamDirectory::new(),
        schema.build(),
        IndexConfig {
            merge_policy: Box::new(crate::NoMergePolicy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for keys in [["a", "b"], ["c", "d"]] {
        for key in keys {
            let mut doc = Document::new();
            doc.add_text(id, key);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.delete_primary_key("a").unwrap();
    writer.commit().await.unwrap();
    for _ in 0..2 {
        writer.force_merge().await.unwrap();
        let reader = index.reader().await.unwrap();
        reader.reload().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        assert_eq!(searcher.segment_readers().len(), 1);
        assert_eq!(searcher.num_docs(), 3);
        assert_eq!(searcher.segment_readers()[0].num_docs(), 4);
        assert_eq!(searcher.search(&AllQuery, 10).await.unwrap().len(), 3);
    }
    writer.force_merge_with_compaction(true).await.unwrap();
    let reader = index.reader().await.unwrap();
    reader.reload().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    assert_eq!(searcher.segment_readers()[0].num_docs(), 3);
    assert!(searcher.segment_readers()[0].deletion_meta().is_none());
}

#[tokio::test]
async fn automatic_compaction_obeys_ratio_and_shared_maintenance_capacity() {
    use std::sync::Arc;
    let global = Arc::new(tokio::sync::Semaphore::new(1));
    let gate = Arc::new(crate::index::ReorderConcurrencyGate::new(2));
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    let index = Index::create(
        RamDirectory::new(),
        schema.build(),
        IndexConfig {
            merge_policy: Box::new(crate::NoMergePolicy),
            background_merge_permits: global.clone(),
            background_reorder_permits: gate.clone(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for key in ["a", "b", "c", "d", "e"] {
        let mut doc = Document::new();
        doc.add_text(id, key);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.delete_primary_key("a").unwrap();
    writer.delete_primary_key("b").unwrap();
    writer.commit().await.unwrap();
    let manager = writer.segment_manager();
    assert!(manager.compaction_candidates(f64::NAN).await.is_err());
    assert!(manager.compaction_candidates(0.0).await.unwrap().is_empty());
    assert!(manager.compaction_candidates(0.5).await.unwrap().is_empty());
    let candidates = manager.compaction_candidates(0.4).await.unwrap();
    assert_eq!(candidates.len(), 1);
    let (segment, physical, deleted) = &candidates[0];
    assert_eq!((*physical, *deleted), (5, 2));
    let held = global.acquire().await.unwrap();
    assert!(
        !manager
            .compact_segment_if_eligible(segment, 0.4, 16 * 1024 * 1024)
            .await
            .unwrap()
    );
    drop(held);
    let held = gate
        .acquire(crate::index::ReorderPriority::Optimizer)
        .await
        .unwrap();
    assert!(
        !manager
            .compact_segment_if_eligible(segment, 0.4, 16 * 1024 * 1024)
            .await
            .unwrap()
    );
    drop(held);
    let held = gate.begin_foreground().await.unwrap();
    assert!(
        !manager
            .compact_segment_if_eligible(segment, 0.4, 16 * 1024 * 1024)
            .await
            .unwrap()
    );
    drop(held);
    assert!(
        !manager
            .compact_segment_if_eligible(segment, 0.5, 16 * 1024 * 1024)
            .await
            .unwrap()
    );
    assert!(
        manager
            .compact_segment_if_eligible(segment, 0.4, 16 * 1024 * 1024)
            .await
            .unwrap()
    );
    assert!(manager.compaction_candidates(0.4).await.unwrap().is_empty());
    assert_eq!(index.num_docs().await.unwrap(), 3);
}

#[tokio::test]
async fn compaction_keeps_bmp_record_order_and_reordering_history_without_claiming_convergence() {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    let sparse = schema.add_sparse_vector_field_with_config(
        "v",
        true,
        false,
        crate::structures::SparseVectorConfig {
            format: crate::structures::SparseFormat::Bmp,
            bmp_block_size: 32,
            dims: Some(64),
            ..Default::default()
        },
    );
    schema.set_multi(sparse, true);
    schema.set_reorder(sparse, true);
    let index = Index::create(
        RamDirectory::new(),
        schema.build(),
        IndexConfig {
            merge_policy: Box::new(crate::NoMergePolicy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for i in 0..131 {
        let mut doc = Document::new();
        doc.add_text(id, i.to_string());
        doc.add_sparse_vector(sparse, vec![(i % 8, 1.0), (20, 0.5)]);
        doc.add_sparse_vector(sparse, vec![(8 + i % 8, 1.5), (21, 0.25)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.reorder().await.unwrap();
    // Start with interrupted BP debt so compaction must retain the attempt
    // count and preserve its actual current virtual order.
    let source = writer.segment_manager().get_segment_ids().await[0].clone();
    writer
        .segment_manager()
        .reorder_single_segment(
            &source,
            None,
            crate::segment::BpBudget {
                time_budget: Some(std::time::Duration::ZERO),
                ..crate::segment::BpBudget::full()
            },
        )
        .await
        .unwrap();
    let before_info = writer
        .segment_manager()
        .read_metadata(|m| m.segment_metas.values().next().unwrap().clone())
        .await;
    assert!(before_info.reordered);
    assert!(!before_info.bp_converged);
    assert_eq!(before_info.bp_unconverged_passes, 1);
    for i in (0..131).step_by(3) {
        writer.delete_primary_key(&i.to_string()).unwrap();
    }
    writer.commit().await.unwrap();
    let mask = writer
        .segment_manager()
        .read_metadata(|m| m.segment_metas.values().next().unwrap().deletions.clone())
        .await;
    let source = writer.segment_manager().get_segment_ids().await[0].clone();
    writer
        .segment_manager()
        .reorder_single_segment(
            &source,
            None,
            crate::segment::BpBudget {
                time_budget: Some(std::time::Duration::ZERO),
                ..crate::segment::BpBudget::full()
            },
        )
        .await
        .unwrap();
    let before_info = writer
        .segment_manager()
        .read_metadata(|m| m.segment_metas.values().next().unwrap().clone())
        .await;
    assert_eq!(
        before_info.deletions, mask,
        "field reorder must retain the exact row visibility generation"
    );
    assert_eq!(before_info.bp_unconverged_passes, 2);
    let reader = index.reader().await.unwrap();
    reader.reload().await.unwrap();
    let before = reader.searcher().await.unwrap();
    let segment = &before.segment_readers()[0];
    let bmp = &segment.bmp_indexes()[&sparse.0];
    let expected: Vec<_> = (0..bmp.num_virtual_docs)
        .map(|v| bmp.virtual_to_doc(v))
        .filter(|(doc, _)| *doc != u32::MAX && segment.is_alive(*doc))
        .map(|(doc, ordinal)| {
            (
                segment
                    .fast_field(id.0)
                    .unwrap()
                    .get_text(doc)
                    .unwrap()
                    .to_owned(),
                ordinal,
            )
        })
        .collect();
    writer.compact(32 * 1024 * 1024).await.unwrap();
    reader.reload().await.unwrap();
    let after = reader.searcher().await.unwrap();
    let segment = &after.segment_readers()[0];
    let bmp = &segment.bmp_indexes()[&sparse.0];
    let actual: Vec<_> = (0..bmp.num_virtual_docs)
        .map(|v| bmp.virtual_to_doc(v))
        .filter(|(doc, _)| *doc != u32::MAX)
        .map(|(doc, ordinal)| {
            (
                segment
                    .fast_field(id.0)
                    .unwrap()
                    .get_text(doc)
                    .unwrap()
                    .to_owned(),
                ordinal,
            )
        })
        .collect();
    assert_eq!(actual, expected);
    let after_info = writer
        .segment_manager()
        .read_metadata(|m| m.segment_metas.values().next().unwrap().clone())
        .await;
    assert!(after_info.reordered);
    assert!(!after_info.bp_converged);
    assert_eq!(
        after_info.bp_unconverged_passes,
        before_info.bp_unconverged_passes
    );
    writer.reorder().await.unwrap();
    let converged = writer
        .segment_manager()
        .read_metadata(|m| m.segment_metas.values().next().unwrap().clone())
        .await;
    assert!(converged.reordered && converged.bp_converged);
    assert_eq!(converged.bp_unconverged_passes, 0);
    assert_eq!(converged.num_deleted_docs(), 0);
    writer.delete_primary_key("1").unwrap();
    writer.commit().await.unwrap();
    writer.compact(32 * 1024 * 1024).await.unwrap();
    let filtered = writer
        .segment_manager()
        .read_metadata(|m| m.segment_metas.values().next().unwrap().clone())
        .await;
    assert!(filtered.reordered);
    assert!(
        !filtered.bp_converged,
        "filtering changes converged BMP blocks"
    );
    assert_eq!(
        filtered.bp_unconverged_passes, 0,
        "compaction is not a BP attempt"
    );
}

#[tokio::test]
async fn compaction_preserves_indexed_only_numbers_binary_vectors_and_stored_payloads() {
    use crate::query::{BinaryDenseVectorQuery, MultiValueCombiner, Query, RangeQuery, TermQuery};
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    let unsigned = schema.add_u64_field("unsigned", true, false);
    let signed = schema.add_i64_field("signed", true, false);
    let float = schema.add_f64_field("float", true, false);
    schema.set_fast(float, true);
    let labels = schema.add_text_field("labels", false, false);
    schema.set_fast(labels, true);
    schema.set_multi(labels, true);
    let binary = schema.add_binary_dense_vector_field("binary", 16, true, false);
    schema.set_multi(binary, true);
    let bytes = schema.add_bytes_field("bytes", true);
    let json = schema.add_json_field("json", true);
    let index = Index::create(RamDirectory::new(), schema.build(), IndexConfig::default())
        .await
        .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for i in 0..131u64 {
        let mut doc = Document::new();
        doc.add_text(id, i.to_string());
        doc.add_u64(unsigned, 42);
        doc.add_i64(signed, -7);
        doc.add_f64(float, 13.5);
        if i % 3 != 0 {
            doc.add_text(labels, "");
            doc.add_text(labels, format!("label-{i}"));
        }
        doc.add_binary_dense_vector(binary, vec![i as u8, 0]);
        doc.add_binary_dense_vector(binary, vec![255, i as u8]);
        doc.add_bytes(bytes, vec![i as u8; 100]);
        doc.add_json(json, serde_json::json!({"n": i, "nested": [null, "text"]}));
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    for i in (0..131).step_by(2) {
        writer.delete_primary_key(&i.to_string()).unwrap();
    }
    writer.commit().await.unwrap();
    let reader = index.reader().await.unwrap();
    let before = reader.searcher().await.unwrap();
    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(TermQuery::new(unsigned, "__num_42")),
        Box::new(TermQuery::new(signed, format!("__num_{}", -7i64 as u64))),
        Box::new(TermQuery::new(
            float,
            format!("__num_{}", 13.5f64.to_bits()),
        )),
        Box::new(RangeQuery::f64(float, Some(13.0), Some(14.0))),
        Box::new(
            BinaryDenseVectorQuery::new(binary, vec![0, 0]).with_combiner(MultiValueCombiner::Sum),
        ),
    ];
    let mut expected = Vec::new();
    for query in &queries {
        let hits = before.search(query.as_ref(), 200).await.unwrap();
        assert_eq!(hits.len(), 65);
        let mut rows = std::collections::BTreeMap::new();
        for hit in hits {
            let doc = before
                .doc(hit.segment_id, hit.doc_id)
                .await
                .unwrap()
                .unwrap();
            let key = doc.get_first(id).unwrap().as_text().unwrap().to_owned();
            rows.insert(
                key,
                (hit.score.to_bits(), serde_json::to_value(doc).unwrap()),
            );
        }
        expected.push(rows);
    }
    writer.compact(16 * 1024 * 1024).await.unwrap();
    reader.reload().await.unwrap();
    let after = reader.searcher().await.unwrap();
    let segment = &after.segment_readers()[0];
    assert_eq!(segment.num_docs(), 65);
    for (query_index, (query, expected)) in queries.iter().zip(expected).enumerate() {
        let hits = after.search(query.as_ref(), 200).await.unwrap();
        assert_eq!(hits.len(), expected.len());
        for hit in hits {
            let doc = after
                .doc(hit.segment_id, hit.doc_id)
                .await
                .unwrap()
                .unwrap();
            let key = doc.get_first(id).unwrap().as_text().unwrap();
            let (score, payload) = &expected[key];
            // Term/BM25 statistics change on compaction; numeric range and
            // exact binary scores must remain bit-identical.
            if query_index >= 3 {
                assert_eq!(hit.score.to_bits(), *score);
            }
            assert_eq!(serde_json::to_value(&doc).unwrap(), *payload);
            assert!(doc.get_first(unsigned).is_none());
            let number: u64 = key.parse().unwrap();
            let fast = segment.fast_field(labels.0).unwrap();
            if number.is_multiple_of(3) {
                assert!(!fast.has_value(hit.doc_id));
            } else {
                let values: Vec<_> = fast
                    .get_multi_values(hit.doc_id)
                    .into_iter()
                    .map(|ordinal| fast.text_dict().unwrap().get(ordinal as u32).unwrap())
                    .collect();
                assert_eq!(values, ["", &format!("label-{number}")]);
            }
        }
    }
}

#[tokio::test]
async fn deleted_rows_leave_top_k_and_old_searchers_keep_their_snapshot() {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_fast(id, true);
    schema.set_primary_key(id);
    let title = schema.add_text_field("title", true, true);
    let directory = RamDirectory::new();
    let index = Index::create(directory.clone(), schema.build(), IndexConfig::default())
        .await
        .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for key in ["a", "b", "c"] {
        let mut doc = Document::new();
        doc.add_text(id, key);
        doc.add_text(title, "hello");
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let reader = index.reader().await.unwrap();
    let old = reader.searcher().await.unwrap();
    writer.delete_primary_key("a").unwrap();
    writer.commit().await.unwrap();
    reader.reload().await.unwrap();
    let current = reader.searcher().await.unwrap();
    assert_eq!(old.num_docs(), 3);
    assert_eq!(current.num_docs(), 2);
    assert_eq!(current.search(&AllQuery, 2).await.unwrap().len(), 2);
    // The low-level physical-copy primitive must not silently discard masks.
    let segment = &current.segment_readers()[0];
    let source = crate::segment::SegmentReader::open_with_deletions(
        &directory,
        crate::segment::SegmentId(segment.meta().id),
        writer.schema(),
        0,
        segment.deletion_meta().cloned(),
    )
    .await
    .unwrap();
    let error = crate::segment::SegmentMerger::new(writer.schema())
        .merge(
            &directory,
            &[source],
            crate::segment::SegmentId::new(),
            None,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("cannot discard deletion masks"));
    let reopened = Index::open(directory, IndexConfig::default())
        .await
        .unwrap();
    assert_eq!(
        reopened
            .reader()
            .await
            .unwrap()
            .searcher()
            .await
            .unwrap()
            .num_docs(),
        2
    );
}

#[tokio::test]
async fn upserts_replace_atomically_and_single_segment_compaction_preserves_indexed_only_values() {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_fast(id, true);
    schema.set_primary_key(id);
    let title = schema.add_text_field("title", true, false);
    let number = schema.add_i64_field("number", true, true);
    schema.set_fast(number, true);
    schema.set_multi(number, true);
    let dir = RamDirectory::new();
    let index = Index::create(dir.clone(), schema.build(), IndexConfig::default())
        .await
        .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for (key, text) in [("a", "obsolete"), ("b", "survives survives"), ("c", "")] {
        let mut doc = Document::new();
        doc.add_text(id, key);
        doc.add_text(title, text);
        if key == "b" {
            doc.add_i64(number, -7);
            doc.add_i64(number, 42);
        }
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let reader = index.reader().await.unwrap();
    let old = reader.searcher().await.unwrap();
    let mut replacement = Document::new();
    replacement.add_text(id, "a");
    replacement.add_text(title, "replacement");
    writer.upsert_document(replacement).unwrap();
    writer.commit().await.unwrap();
    reader.reload().await.unwrap();
    let updated = reader.searcher().await.unwrap();
    assert_eq!(updated.num_docs(), 3);
    assert_eq!(
        updated
            .search(&crate::query::TermQuery::new(title, "obsolete"), 10)
            .await
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        updated
            .search(&crate::query::TermQuery::new(title, "replacement"), 10)
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        old.search(&crate::query::TermQuery::new(title, "obsolete"), 10)
            .await
            .unwrap()
            .len(),
        1
    );
    writer.force_merge_with_compaction(true).await.unwrap();
    reader.reload().await.unwrap();
    let merged = reader.searcher().await.unwrap();
    assert_eq!(merged.segment_readers().len(), 1);
    assert_eq!(merged.segment_readers()[0].num_docs(), 3);
    assert!(merged.segment_readers()[0].deletion_meta().is_none());
    let hits = merged
        .search(&crate::query::TermQuery::new(title, "survives"), 10)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    let segment = &merged.segment_readers()[0];
    let row = (0..3)
        .find(|&doc| segment.fast_field(id.0).unwrap().get_text(doc) == Some("b"))
        .unwrap();
    assert_eq!(
        segment
            .fast_field(number.0)
            .unwrap()
            .get_multi_values(row)
            .len(),
        2
    );
    assert_eq!(segment.meta().field_stats[&title.0].total_tokens, 3);
    assert_eq!(segment.meta().field_stats[&title.0].doc_count, 3);
    writer.delete_primary_key("b").unwrap();
    writer.commit().await.unwrap();
    assert_eq!(writer.compact(16 * 1024 * 1024).await.unwrap(), 1);
    reader.reload().await.unwrap();
    let compacted = reader.searcher().await.unwrap();
    assert_eq!(compacted.segment_readers().len(), 1);
    assert_eq!(compacted.segment_readers()[0].num_docs(), 2);
    assert_eq!(
        compacted.segment_readers()[0].meta().field_stats[&title.0].total_tokens,
        1
    );
    assert_eq!(
        compacted.segment_readers()[0].meta().field_stats[&title.0].doc_count,
        2
    );
    assert_eq!(merged.num_docs(), 3);
    for key in ["a", "c"] {
        writer.delete_primary_key(key).unwrap();
    }
    writer.commit().await.unwrap();
    writer.force_merge_with_compaction(true).await.unwrap();
    reader.reload().await.unwrap();
    let empty = reader.searcher().await.unwrap();
    assert_eq!(empty.num_docs(), 0);
    assert!(empty.search(&AllQuery, 10).await.unwrap().is_empty());
    assert_eq!(
        empty
            .segment_readers()
            .iter()
            .map(|s| s.num_docs())
            .sum::<u32>(),
        0
    );
    let reopened = Index::open(dir, IndexConfig::default()).await.unwrap();
    assert_eq!(
        reopened
            .reader()
            .await
            .unwrap()
            .searcher()
            .await
            .unwrap()
            .num_docs(),
        0
    );
}

#[tokio::test]
async fn deleted_dense_neighbours_do_not_consume_top_k_slots() {
    use crate::dsl::{DenseVectorConfig, VectorIndexType};
    for kind in [VectorIndexType::Flat, VectorIndexType::Tq] {
        let mut schema = SchemaBuilder::default();
        let id = schema.add_text_field("id", true, true);
        schema.set_fast(id, true);
        schema.set_primary_key(id);
        let vector = schema.add_dense_vector_field_with_config(
            "v",
            true,
            false,
            DenseVectorConfig {
                dim: 8,
                index_type: kind,
                quantization: crate::dsl::DenseVectorQuantization::F32,
                num_clusters: None,
                target_vectors: None,
                tree_levels: None,
                ivf_routing: crate::dsl::IvfRoutingMode::Auto,
                nprobe: 1,
                unit_norm: false,
                soar: None,
            },
        );
        let dir = RamDirectory::new();
        let index = Index::create(dir, schema.build(), IndexConfig::default())
            .await
            .unwrap();
        let mut writer = index.writer();
        writer.init_primary_key_dedup().await.unwrap();
        for i in 0..65 {
            let mut doc = Document::new();
            doc.add_text(id, i.to_string());
            doc.add_dense_vector(vector, vec![1.0 + i as f32; 8]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        let reader = index.reader().await.unwrap();
        for i in 0..64 {
            writer.delete_primary_key(&i.to_string()).unwrap();
        }
        writer.commit().await.unwrap();
        reader.reload().await.unwrap();
        let query = crate::query::DenseVectorQuery::new(vector, vec![1.0; 8]);
        let before = reader.searcher().await.unwrap();
        let hits = before.search(&query, 1).await.unwrap();
        assert_eq!(hits.len(), 1, "{kind:?}");
        assert_eq!(hits[0].doc_id, 64);
        let async_hits =
            crate::query::search_segment_with_count(&before.segment_readers()[0], &query, 1)
                .await
                .unwrap()
                .0;
        assert_eq!(async_hits.len(), 1);
        assert_eq!(async_hits[0].doc_id, 64);
        writer.compact(16 * 1024 * 1024).await.unwrap();
        reader.reload().await.unwrap();
        let after = reader.searcher().await.unwrap();
        let compacted = after.search(&query, 1).await.unwrap();
        assert_eq!(compacted.len(), 1);
        assert_eq!(compacted[0].score.to_bits(), hits[0].score.to_bits());
        assert_eq!(after.segment_readers()[0].num_docs(), 1);
        writer.delete_primary_key("64").unwrap();
        writer.compact(16 * 1024 * 1024).await.unwrap();
        reader.reload().await.unwrap();
        let empty = reader.searcher().await.unwrap();
        assert_eq!(empty.num_docs(), 0);
        assert!(empty.search(&query, 1).await.unwrap().is_empty());
    }
}

#[tokio::test]
async fn compaction_preserves_chunk_ordinals_positions_and_sparse_vector_scores() {
    use crate::query::{MultiValueCombiner, SparseVectorQuery, TermQuery};
    use crate::structures::{SparseFormat, SparseVectorConfig, WeightQuantization};
    for format in [SparseFormat::Bmp, SparseFormat::MaxScore] {
        let mut schema = SchemaBuilder::default();
        let id = schema.add_text_field("id", true, true);
        schema.set_fast(id, true);
        schema.set_primary_key(id);
        let text = schema.add_text_field("chunks", true, false);
        schema.set_chunked(text, true);
        schema.set_positions(text, crate::dsl::PositionMode::TokenPosition);
        let sparse = schema.add_sparse_vector_field_with_config(
            "sparse",
            true,
            false,
            SparseVectorConfig {
                format,
                weight_quantization: WeightQuantization::UInt8,
                bmp_block_size: 64,
                dims: Some(128),
                ..Default::default()
            },
        );
        schema.set_multi(sparse, true);
        let dir = RamDirectory::new();
        let index = Index::create(
            dir,
            schema.build(),
            IndexConfig {
                merge_policy: Box::new(crate::merge::NoMergePolicy),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let mut writer = index.writer();
        writer.init_primary_key_dedup().await.unwrap();
        for i in 0..131 {
            let mut doc = Document::new();
            doc.add_text(id, i.to_string());
            doc.add_text(text, "alpha");
            doc.add_text(text, "");
            doc.add_text(text, "alpha beta beta");
            doc.add_sparse_vector(sparse, vec![(1, 0.5 + i as f32 / 200.0), (2, 1.0)]);
            doc.add_sparse_vector(sparse, vec![(1, 2.0), (3, 0.3)]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        for i in 0..131 {
            if i % 3 != 2 {
                writer.delete_primary_key(&i.to_string()).unwrap();
            }
        }
        writer.commit().await.unwrap();
        let reader = index.reader().await.unwrap();
        let before = reader.searcher().await.unwrap();
        let query = SparseVectorQuery::new(sparse, vec![(1, 1.0), (2, 0.5)])
            .with_combiner(MultiValueCombiner::Sum)
            .with_lsp_gamma(0);
        let reference = before.search(&query, 200).await.unwrap();
        assert_eq!(reference.len(), 43, "{format:?}");
        let mut expected = std::collections::BTreeMap::new();
        for hit in reference {
            let key = before
                .doc(hit.segment_id, hit.doc_id)
                .await
                .unwrap()
                .unwrap()
                .get_first(id)
                .unwrap()
                .as_text()
                .unwrap()
                .to_owned();
            expected.insert(key, hit.score);
        }
        writer.compact(32 * 1024 * 1024).await.unwrap();
        reader.reload().await.unwrap();
        let after = reader.searcher().await.unwrap();
        assert_eq!(after.segment_readers()[0].num_docs(), 43);
        assert_eq!(
            after.segment_readers()[0].meta().field_stats[&text.0].doc_count,
            129
        );
        assert_eq!(
            after.segment_readers()[0].meta().field_stats[&text.0].total_tokens,
            172
        );
        let actual = after.search(&query, 200).await.unwrap();
        assert_eq!(actual.len(), expected.len());
        for hit in actual {
            let doc = after
                .doc(hit.segment_id, hit.doc_id)
                .await
                .unwrap()
                .unwrap();
            let key = doc.get_first(id).unwrap().as_text().unwrap();
            assert!(
                (hit.score - expected[key]).abs() < 1e-5,
                "{format:?} {key}: {} != {}",
                hit.score,
                expected[key]
            );
        }
        let (hits, _) = after
            .search_with_positions(&TermQuery::new(text, "beta"), 200)
            .await
            .unwrap();
        assert_eq!(hits.len(), 43);
        for hit in hits {
            assert_eq!(hit.positions[0].1[0].position, 2);
        }
        for key in expected.keys() {
            writer.delete_primary_key(key).unwrap();
        }
        writer.compact(32 * 1024 * 1024).await.unwrap();
        reader.reload().await.unwrap();
        let empty = reader.searcher().await.unwrap();
        assert_eq!(empty.num_docs(), 0);
        assert!(empty.search(&query, 10).await.unwrap().is_empty());
    }
}
