#![cfg(feature = "native")]

use std::sync::Arc;

use summa_core::query::{Query, RangeQuery, Scorer};
use summa_core::segment::{
    SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentMerger, SegmentReader,
};
use summa_core::structures::TERMINATED;
use summa_core::structures::fast_field::f64_to_sortable_u64;
use summa_core::{Document, Field, RamDirectory, SchemaBuilder};

fn scorer_hits(mut scorer: Box<dyn Scorer + '_>) -> Vec<u32> {
    let mut hits = Vec::new();
    while scorer.doc() != TERMINATED {
        assert_eq!(scorer.score().to_bits(), 1.0f32.to_bits());
        hits.push(scorer.doc());
        scorer.advance();
    }
    hits
}

fn assert_scorer_seeks(mut scorer: Box<dyn Scorer + '_>, expected: &[u32], num_docs: u32) {
    // Warm a dense batch before seeking within it and beyond its end.
    for _ in 0..12 {
        let current = scorer.doc();
        let next = expected
            .iter()
            .copied()
            .find(|&doc| doc > current)
            .unwrap_or(TERMINATED);
        assert_eq!(scorer.advance(), next);
    }
    for target in [
        0,
        1,
        7,
        8,
        9,
        63,
        64,
        65,
        255,
        256,
        257,
        511,
        512,
        513,
        num_docs.saturating_sub(1),
        num_docs,
        TERMINATED,
    ] {
        let floor = target.max(scorer.doc());
        let wanted = expected
            .iter()
            .copied()
            .find(|&doc| doc >= floor)
            .unwrap_or(TERMINATED);
        assert_eq!(scorer.seek(target), wanted, "seek to {target}");
        assert_eq!(scorer.seek(0), wanted, "backward seek must stay put");
        assert_eq!(scorer.seek(wanted), wanted, "equal seek must stay put");
        if wanted != TERMINATED {
            assert_eq!(scorer.score().to_bits(), 1.0f32.to_bits());
            let next = expected
                .iter()
                .copied()
                .find(|&doc| doc > wanted)
                .unwrap_or(TERMINATED);
            assert_eq!(scorer.advance(), next);
        }
    }
    for _ in 0..3 {
        assert_eq!(scorer.advance(), TERMINATED);
        assert_eq!(scorer.seek(0), TERMINATED);
        assert_eq!(scorer.size_hint(), 0);
    }
}

#[tokio::test]
async fn range_block_pruning_uses_global_text_ordinals_after_merge() {
    let dir = RamDirectory::new();
    let mut sb = SchemaBuilder::default();
    let field = sb.add_text_field("value", false, false);
    sb.set_fast(field, true);
    let schema = Arc::new(sb.build());
    let mut sources = Vec::new();
    for words in [["y", "z"], ["a", "b"]] {
        let mut builder =
            SegmentBuilder::new(Arc::clone(&schema), SegmentBuilderConfig::default()).unwrap();
        for word in words.into_iter().cycle().take(130) {
            let mut doc = Document::new();
            doc.add_text(field, word);
            builder.add_document(doc).unwrap();
        }
        let id = SegmentId::new();
        builder.build(&dir, id, None).await.unwrap();
        sources.push(
            SegmentReader::open(&dir, id, Arc::clone(&schema), 0)
                .await
                .unwrap(),
        );
    }
    let id = SegmentId::new();
    SegmentMerger::new(Arc::clone(&schema))
        .merge(&dir, &sources, id, None)
        .await
        .unwrap();
    let reader = SegmentReader::open(&dir, id, schema, 0).await.unwrap();
    // First block stores local ordinals 0,1, which become global ordinals 2,3.
    let query = RangeQuery::u64(field, Some(2), Some(3));
    let result = query.as_doc_bitset(&reader).unwrap();
    assert_eq!(
        (0..260)
            .filter(|&doc| result.contains(doc))
            .collect::<Vec<_>>(),
        (0..130).collect::<Vec<_>>()
    );
    assert_eq!(
        scorer_hits(query.scorer(&reader, 4).await.unwrap()),
        (0..130).collect::<Vec<_>>()
    );
    assert_scorer_seeks(
        query.scorer(&reader, 4).await.unwrap(),
        &(0..130).collect::<Vec<_>>(),
        260,
    );
    #[cfg(feature = "sync")]
    assert_eq!(
        scorer_hits(query.scorer_sync(&reader, 4).unwrap()),
        (0..130).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn range_bitsets_preserve_matches_across_every_copied_block_bit_offset() {
    let dir = RamDirectory::new();
    let mut sb = SchemaBuilder::default();
    let field = sb.add_u64_field("value", false, false);
    sb.set_fast(field, true);
    let schema = Arc::new(sb.build());
    let mut sources = Vec::new();
    let mut values = Vec::new();
    // Each copied block starts one bit later, exercises four full words and
    // a one-value batch tail, and shares an output word with its neighbour.
    for block in 0..64 {
        let mut builder =
            SegmentBuilder::new(Arc::clone(&schema), SegmentBuilderConfig::default()).unwrap();
        for local in 0..257 {
            let value = (local + block) % 5;
            let mut doc = Document::new();
            if value != 4 {
                doc.add_u64(field, value);
            }
            builder.add_document(doc).unwrap();
            values.push(value);
        }
        let id = SegmentId::new();
        builder.build(&dir, id, None).await.unwrap();
        sources.push(
            SegmentReader::open(&dir, id, Arc::clone(&schema), 0)
                .await
                .unwrap(),
        );
    }
    let id = SegmentId::new();
    SegmentMerger::new(Arc::clone(&schema))
        .merge(&dir, &sources, id, None)
        .await
        .unwrap();
    let reader = SegmentReader::open(&dir, id, schema, 0).await.unwrap();
    assert_eq!(reader.fast_field(field.0).unwrap().num_blocks(), 64);
    for (lo, hi) in [(0, 0), (1, 2), (0, 3), (4, 4), (3, 1)] {
        let query = RangeQuery::u64(field, Some(lo), Some(hi));
        let expected: Vec<_> = values
            .iter()
            .enumerate()
            .filter_map(|(doc, &value)| {
                (value != 4 && value >= lo && value <= hi).then_some(doc as u32)
            })
            .collect();
        let bits = query.as_doc_bitset(&reader).unwrap();
        let actual: Vec<_> = (0..reader.num_docs())
            .filter(|&doc| bits.contains(doc))
            .collect();
        assert_eq!(actual, expected, "{query}");
        assert_eq!(bits.count() as usize, expected.len());
        assert_eq!(
            scorer_hits(query.scorer(&reader, values.len()).await.unwrap()),
            expected
        );
        #[cfg(feature = "sync")]
        assert_eq!(
            scorer_hits(query.scorer_sync(&reader, values.len()).unwrap()),
            expected
        );
    }
}

#[tokio::test]
async fn range_bitsets_preserve_numeric_bounds_missing_values_and_first_multi_value_after_merge() {
    let dir = RamDirectory::new();
    let mut sb = SchemaBuilder::default();
    let unsigned = sb.add_u64_field("unsigned", false, false);
    let signed = sb.add_i64_field("signed", false, false);
    let float = sb.add_f64_field("float", false, false);
    let multi = sb.add_u64_field("multi", false, false);
    let constant = sb.add_u64_field("constant", false, false);
    let linear = sb.add_u64_field("linear", false, false);
    for field in [unsigned, signed, float, multi, constant, linear] {
        sb.set_fast(field, true);
    }
    sb.set_multi(multi, true);
    let schema = Arc::new(sb.build());
    let mut sources = Vec::new();
    // Uneven block and batch tails, including a whole absent-value block.
    let mut records = Vec::new();
    for (source, count) in [257u32, 513, 9].into_iter().enumerate() {
        let mut builder =
            SegmentBuilder::new(Arc::clone(&schema), SegmentBuilderConfig::default()).unwrap();
        for local in 0..count {
            let id = records.len() as u32;
            let present = source != 1 && local % 11 != 0;
            let u = u64::from((id * 137) % 1024);
            let i = [i64::MIN + 1, -100, -1, 0, 1, 100, i64::MAX][id as usize % 7];
            let f = [
                f64::NEG_INFINITY,
                -2.0,
                -0.0,
                0.0,
                1.0,
                f64::INFINITY,
                f64::NAN,
            ][id as usize % 7];
            let mut doc = Document::new();
            if present {
                doc.add_u64(unsigned, u);
                doc.add_i64(signed, i);
                doc.add_f64(float, f);
                doc.add_u64(multi, u);
                // A match in the second position alone must not match the range.
                doc.add_u64(multi, 42);
            }
            doc.add_u64(constant, 7);
            doc.add_u64(linear, u64::from(id) * 10);
            builder.add_document(doc).unwrap();
            records.push((present, u, i, f));
        }
        let id = SegmentId::new();
        builder.build(&dir, id, None).await.unwrap();
        sources.push(
            SegmentReader::open(&dir, id, Arc::clone(&schema), 0)
                .await
                .unwrap(),
        );
    }
    let id = SegmentId::new();
    SegmentMerger::new(Arc::clone(&schema))
        .merge(&dir, &sources, id, None)
        .await
        .unwrap();
    let merged = SegmentReader::open(&dir, id, schema, 0).await.unwrap();
    assert_eq!(merged.fast_field(unsigned.0).unwrap().num_blocks(), 3);

    let mut cases = Vec::new();
    for field in [unsigned, multi] {
        for (min, max) in [
            (None, None),
            (Some(42), Some(42)),
            (Some(700), None),
            (None, Some(20)),
            (Some(900), Some(3)),
            (Some(u64::MAX), Some(u64::MAX)),
        ] {
            let expected = records
                .iter()
                .enumerate()
                .filter_map(|(doc, &(present, u, _, _))| {
                    (present && u >= min.unwrap_or(0) && u <= max.unwrap_or(u64::MAX - 1))
                        .then_some(doc as u32)
                })
                .collect::<Vec<_>>();
            cases.push((RangeQuery::u64(field, min, max), expected));
        }
    }
    for (min, max) in [
        (None, None),
        (Some(-100), Some(0)),
        (Some(i64::MAX), None),
        (None, Some(i64::MIN + 1)),
        (Some(5), Some(-5)),
    ] {
        let expected = records
            .iter()
            .enumerate()
            .filter_map(|(doc, &(present, _, i, _))| {
                (present && i >= min.unwrap_or(i64::MIN) && i <= max.unwrap_or(i64::MAX))
                    .then_some(doc as u32)
            })
            .collect::<Vec<_>>();
        cases.push((RangeQuery::i64(signed, min, max), expected));
    }
    for (min, max) in [
        (None, None),
        (Some(-0.0), Some(0.0)),
        (Some(0.0), Some(0.0)),
        (Some(f64::NEG_INFINITY), Some(f64::INFINITY)),
        (Some(3.0), Some(-3.0)),
    ] {
        let expected = records
            .iter()
            .enumerate()
            .filter_map(|(doc, &(present, _, _, f))| {
                let raw = f64_to_sortable_u64(f);
                (present
                    && raw >= min.map(f64_to_sortable_u64).unwrap_or(0)
                    && raw <= max.map(f64_to_sortable_u64).unwrap_or(u64::MAX - 1))
                .then_some(doc as u32)
            })
            .collect::<Vec<_>>();
        cases.push((RangeQuery::f64(float, min, max), expected));
    }
    cases.push((
        RangeQuery::u64(constant, Some(7), Some(7)),
        (0..merged.num_docs()).collect(),
    ));
    cases.push((
        RangeQuery::u64(linear, Some(2560), Some(5130)),
        (256..=513).collect(),
    ));
    for (query, expected) in cases {
        assert_scorer_seeks(
            query.scorer(&merged, 10).await.unwrap(),
            &expected,
            merged.num_docs(),
        );
        #[cfg(feature = "sync")]
        assert_scorer_seeks(
            query.scorer_sync(&merged, 10).unwrap(),
            &expected,
            merged.num_docs(),
        );
        let bits = query.as_doc_bitset(&merged).unwrap();
        let actual: Vec<_> = (0..merged.num_docs())
            .filter(|&doc| bits.contains(doc))
            .collect();
        assert_eq!(actual, expected, "{query}");
        assert!(!bits.contains(merged.num_docs()));
        assert!(!bits.contains(u32::MAX));
        assert_eq!(bits.count() as usize, expected.len(), "{query}");
        assert_eq!(
            scorer_hits(query.scorer(&merged, records.len()).await.unwrap()),
            expected,
            "{query}"
        );
        #[cfg(feature = "sync")]
        assert_eq!(
            scorer_hits(query.scorer_sync(&merged, records.len()).unwrap()),
            expected,
            "{query}"
        );
    }
    assert!(
        RangeQuery::u64(Field(999), None, None)
            .as_doc_bitset(&merged)
            .is_none()
    );
}

#[tokio::test]
async fn lazy_range_scans_and_seeks_preserve_hits_across_compressed_record_and_merge_tails() {
    let dir = RamDirectory::new();
    let mut sb = SchemaBuilder::default();
    let field = sb.add_u64_field("value", false, false);
    sb.set_fast(field, true);
    let schema = Arc::new(sb.build());
    let mut sources = Vec::new();
    let mut expected = Vec::new();
    let mut doc_id = 0;
    for count in [2053u32, 4097] {
        let mut builder =
            SegmentBuilder::new(Arc::clone(&schema), SegmentBuilderConfig::default()).unwrap();
        for local_doc in 0..count {
            let block = u64::from(local_doc / 512);
            let local = u64::from(local_doc % 512);
            let value = ((block * 40503) & 65535) * 65536 + local * (3 + block % 3) + local % 7;
            let mut doc = Document::new();
            doc.add_u64(field, value);
            builder.add_document(doc).unwrap();
            if value <= 1 << 31 {
                expected.push(doc_id);
            }
            doc_id += 1;
        }
        let id = SegmentId::new();
        builder.build(&dir, id, None).await.unwrap();
        let reader = SegmentReader::open(&dir, id, Arc::clone(&schema), 0)
            .await
            .unwrap();
        assert_eq!(
            reader.fast_field(field.0).unwrap().blocks()[0]
                .data
                .as_slice()[0],
            3
        );
        sources.push(reader);
    }
    let id = SegmentId::new();
    SegmentMerger::new(Arc::clone(&schema))
        .merge(&dir, &sources, id, None)
        .await
        .unwrap();
    let reader = SegmentReader::open(&dir, id, schema, 0).await.unwrap();
    let query = RangeQuery::u64(field, None, Some(1 << 31));
    assert_eq!(
        scorer_hits(query.scorer(&reader, 10).await.unwrap()),
        expected
    );
    assert_scorer_seeks(query.scorer(&reader, 10).await.unwrap(), &expected, doc_id);
    #[cfg(feature = "sync")]
    {
        assert_eq!(
            scorer_hits(query.scorer_sync(&reader, 10).unwrap()),
            expected
        );
        assert_scorer_seeks(query.scorer_sync(&reader, 10).unwrap(), &expected, doc_id);
    }
}
