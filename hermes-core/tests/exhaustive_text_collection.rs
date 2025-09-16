#![cfg(feature = "native")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use hermes_core::query::{BooleanQuery, CountCollector, TermQuery, TopKCollector, collect_segment};
use hermes_core::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentReader};
use hermes_core::{Document, RamDirectory, SchemaBuilder};

struct MeasuredAllocator;
static MEASURE: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for MeasuredAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if MEASURE.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if MEASURE.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(size, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: MeasuredAllocator = MeasuredAllocator;

// One test in this binary: the allocation window cannot overlap another test.
#[tokio::test(flavor = "current_thread")]
async fn exhaustive_text_count_keeps_scratch_bounded_and_preserves_top_k() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let text = schema.add_text_field("text", true, false);
    let schema = Arc::new(schema.build());
    let mut builder = SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
    for row in 0..32_768 {
        let mut doc = Document::new();
        doc.add_text(
            text,
            match row % 3 {
                0 => "alpha beta beta",
                1 => "alpha padding",
                _ => "beta padding padding",
            },
        );
        builder.add_document(doc).unwrap();
    }
    let id = SegmentId::new();
    builder.build(&dir, id, None).await.unwrap();
    let reader = SegmentReader::open(&dir, id, schema, 16).await.unwrap();
    // Warm immutable posting views outside the query allocation measurement.
    reader.get_postings(text, b"alpha").await.unwrap();
    reader.get_postings(text, b"beta").await.unwrap();
    let query = BooleanQuery::new()
        .should(TermQuery::text(text, "alpha"))
        .should(TermQuery::text(text, "beta"));
    let mut count = CountCollector::new();
    ALLOCATED.store(0, Ordering::Relaxed);
    MEASURE.store(true, Ordering::Relaxed);
    let result = collect_segment(&reader, &query, &mut count).await;
    MEASURE.store(false, Ordering::Relaxed);
    let allocated = ALLOCATED.load(Ordering::Relaxed);
    result.unwrap();
    eprintln!("query_allocated_bytes={allocated}");
    assert_eq!(count.count(), 32_768);
    assert!(
        allocated < 256 * 1024,
        "count allocated {allocated} bytes for 32k matches"
    );

    let mut count = CountCollector::new();
    let mut top = TopKCollector::new(10);
    ALLOCATED.store(0, Ordering::Relaxed);
    MEASURE.store(true, Ordering::Relaxed);
    let result = collect_segment(&reader, &query, &mut (&mut top, &mut count)).await;
    MEASURE.store(false, Ordering::Relaxed);
    let allocated = ALLOCATED.load(Ordering::Relaxed);
    result.unwrap();
    eprintln!("rank_and_count_allocated_bytes={allocated}");
    assert!(
        allocated < 256 * 1024,
        "rank and count allocated {allocated} bytes"
    );
    assert_eq!(count.count(), 32_768);
    let exhaustive = top.into_sorted_results();
    let (ranked, _) = hermes_core::query::search_segment_with_count(&reader, &query, 10)
        .await
        .unwrap();
    assert_eq!(exhaustive.len(), ranked.len());
    for (a, b) in exhaustive.iter().zip(ranked) {
        assert_eq!(a.doc_id, b.doc_id);
        assert!((a.score - b.score).abs() < 1e-6);
    }
    for term in ["alpha", "beta", "missing"] {
        let query = TermQuery::text(text, term);
        let mut exact = CountCollector::new();
        collect_segment(&reader, &query, &mut exact).await.unwrap();
        let mut enumerated = CountCollector::new();
        let mut top = TopKCollector::new(1);
        collect_segment(&reader, &query, &mut (&mut enumerated, &mut top))
            .await
            .unwrap();
        assert_eq!(exact.count(), enumerated.count(), "term {term}");
        // Bulk counts add to existing collector state across segments.
        collect_segment(&reader, &query, &mut exact).await.unwrap();
        assert_eq!(exact.count(), 2 * enumerated.count());
    }
    exhaustive_collection_preserves_deleted_rows_and_positioned_multivalues().await;
}

async fn exhaustive_collection_preserves_deleted_rows_and_positioned_multivalues() {
    use hermes_core::dsl::PositionMode;
    use hermes_core::index::{Index, IndexConfig, IndexWriter};
    use hermes_core::query::PhraseQuery;
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field_with_tokenizer("id", true, true, "raw");
    schema.set_primary_key(id);
    let text = schema.add_text_field("text", true, false);
    schema.set_multi(text, true);
    schema.set_positions(text, PositionMode::TokenPosition);
    let config = IndexConfig {
        num_indexing_threads: 1,
        num_threads: 1,
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    for (key, values) in [
        ("a", vec!["alpha beta", "gamma"]),
        ("b", vec!["alpha beta"]),
        ("c", vec!["alpha", "beta"]),
    ] {
        let mut doc = Document::new();
        doc.add_text(id, key);
        for value in values {
            doc.add_text(text, value);
        }
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.delete_primary_key("b").unwrap();
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let union = BooleanQuery::new()
        .should(TermQuery::text(text, "alpha"))
        .should(TermQuery::text(text, "beta"));
    let conjunction = BooleanQuery::new()
        .must(TermQuery::text(text, "alpha"))
        .must(TermQuery::text(text, "beta"));
    let phrase = PhraseQuery::text(text, "alpha beta");
    let term = TermQuery::text(text, "alpha");
    for (query, expected) in [
        (&union as &dyn hermes_core::query::Query, 2),
        (&conjunction, 2),
        (&phrase, 1),
        (&term, 2),
    ] {
        let mut count = CountCollector::new();
        for segment in searcher.segment_readers() {
            collect_segment(segment, query, &mut count).await.unwrap();
        }
        assert_eq!(count.count(), expected);
        let (async_hits, _) = searcher.search_with_count(query, 10).await.unwrap();
        #[cfg(feature = "sync")]
        {
            let (sync_hits, _) = searcher
                .search_with_offset_and_count_sync(query, 10, 0)
                .unwrap();
            assert_eq!(async_hits, sync_hits);
        }
        assert_eq!(async_hits.len(), expected as usize);
    }
}
