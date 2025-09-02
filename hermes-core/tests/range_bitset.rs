#![cfg(feature = "native")]

use std::sync::Arc;

use hermes_core::query::{Query, RangeQuery, Scorer};
use hermes_core::segment::{
    SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentMerger, SegmentReader,
};
use hermes_core::structures::TERMINATED;
use hermes_core::structures::fast_field::f64_to_sortable_u64;
use hermes_core::{Document, Field, RamDirectory, SchemaBuilder};

fn scorer_hits(mut scorer: Box<dyn Scorer + '_>) -> Vec<u32> {
    let mut hits = Vec::new();
    while scorer.doc() != TERMINATED {
        assert_eq!(scorer.score().to_bits(), 1.0f32.to_bits());
        hits.push(scorer.doc());
        scorer.advance();
    }
    hits
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
