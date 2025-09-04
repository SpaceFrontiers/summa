use super::*;
use crate::query::{
    AllQuery, BooleanQuery, RangeQuery, SparseTermQuery, SparseVectorQuery, TermQuery,
};
use crate::structures::{SparseFormat, SparseVectorConfig};
use crate::{Document, Field, Index, IndexConfig, IndexWriter, RamDirectory, Schema};

async fn fixture() -> (Index<RamDirectory>, Field, Field, Field, Field) {
    let mut schema = Schema::builder();
    let text = schema.add_text_field_with_tokenizer("body", true, false, "simple");
    let allowed = schema.add_u64_field("allowed", true, false);
    schema.set_fast(allowed, true);
    let sparse: Vec<_> = [SparseFormat::Bmp, SparseFormat::MaxScore]
        .into_iter()
        .enumerate()
        .map(|(i, format)| {
            schema.add_sparse_vector_field_with_config(
                &format!("sparse{i}"),
                true,
                false,
                SparseVectorConfig {
                    format,
                    dims: Some(16),
                    ..Default::default()
                },
            )
        })
        .collect();
    let directory = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(directory.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for doc in 0..12 {
        let mut document = Document::new();
        document.add_text(text, "alpha ".repeat(12 - doc));
        document.add_u64(allowed, u64::from(doc >= 10));
        for &field in &sparse {
            document.add_sparse_vector(field, vec![(0, (12 - doc) as f32 / 12.0), (1, 0.1)]);
        }
        writer.add_document(document).unwrap();
    }
    writer.commit().await.unwrap();
    (
        Index::open(directory, config).await.unwrap(),
        text,
        allowed,
        sparse[0],
        sparse[1],
    )
}

fn eligible(query: impl Query + 'static, allowed: Field) -> FilteredQuery {
    FilteredQuery::new(
        Arc::new(query),
        vec![Arc::new(RangeQuery::u64(allowed, Some(1), Some(1)))],
    )
}

async fn assert_selected(
    index: &Index<RamDirectory>,
    query: &dyn Query,
    limit: usize,
    expected: &[u32],
) {
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    assert_eq!(searcher.segment_readers().len(), 1);
    let reader = &searcher.segment_readers()[0];
    let check = |mut scorer: Box<dyn Scorer + '_>| {
        let mut actual = Vec::new();
        while scorer.doc() != crate::TERMINATED {
            actual.push(scorer.doc());
            scorer.advance();
        }
        assert_eq!(actual, expected, "{query}");
    };
    check(
        query
            .scorer_with_options(reader, limit, ScorerOptions::default())
            .await
            .unwrap(),
    );
    #[cfg(feature = "sync")]
    check(
        query
            .scorer_sync_with_options(reader, limit, ScorerOptions::default())
            .unwrap(),
    );
    let results = index.search(query, limit).await.unwrap();
    let mut actual: Vec<_> = results.hits.iter().map(|hit| hit.address.doc_id).collect();
    actual.sort_unstable();
    assert_eq!(actual, expected, "index search: {query}");
}

#[tokio::test]
async fn small_limits_filter_required_plain_text_disjunctions_before_top_k() {
    let (index, text, allowed, _, _) = fixture().await;
    let query = BooleanQuery::new()
        .must(RangeQuery::u64(allowed, Some(1), Some(1)))
        .must(
            BooleanQuery::new()
                .should(TermQuery::text(text, "alpha"))
                .should(TermQuery::text(text, "absent")),
        );
    assert_selected(&index, &query, 1, &[10]).await;
}

#[tokio::test]
async fn common_filter_survives_nested_boolean_optimization() {
    let (index, text, allowed, bmp, maxscore) = fixture().await;
    let text_query = BooleanQuery::new()
        .should(eligible(TermQuery::text(text, "alpha"), allowed))
        .should(TermQuery::text(text, "absent"));
    assert_selected(&index, &text_query, 12, &[10, 11]).await;
    for field in [bmp, maxscore] {
        let query = BooleanQuery::new()
            .should(eligible(
                SparseVectorQuery::new(field, vec![(0, 1.0)]),
                allowed,
            ))
            .should(SparseTermQuery::new(field, 15, 1.0));
        assert_selected(&index, &query, 12, &[10, 11]).await;
    }
}

#[tokio::test]
async fn nested_boolean_sparse_filters_survive_scoring_decomposition() {
    let (index, _, allowed, bmp, maxscore) = fixture().await;
    for field in [bmp, maxscore] {
        for excluded in [false, true] {
            let inner = BooleanQuery::new().should(SparseVectorQuery::new(field, vec![(0, 1.0)]));
            let inner = if excluded {
                inner.must_not(RangeQuery::u64(allowed, Some(0), Some(0)))
            } else {
                inner.must(RangeQuery::u64(allowed, Some(1), Some(1)))
            };
            let query = BooleanQuery::new()
                .should(inner)
                .should(SparseTermQuery::new(field, 15, 1.0));
            assert_selected(&index, &query, 12, &[10, 11]).await;
        }
    }
}

#[tokio::test]
async fn filtered_bmp_keeps_one_global_superblock_budget_across_segments() {
    let mut schema = Schema::builder();
    let allowed = schema.add_u64_field("allowed", true, false);
    schema.set_fast(allowed, true);
    let sparse = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::Bmp,
            dims: Some(16),
            ..Default::default()
        },
    );
    let directory = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(directory.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for weight in [1.0, 0.1] {
        let mut document = Document::new();
        document.add_u64(allowed, 1);
        document.add_sparse_vector(sparse, vec![(0, weight)]);
        writer.add_document(document).unwrap();
        writer.commit().await.unwrap();
    }
    let index = Index::open(directory, config).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 2);
    let sparse_query = SparseVectorQuery::new(sparse, vec![(0, 1.0)]).with_lsp_gamma(1);
    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(eligible(sparse_query.clone(), allowed)),
        Box::new(
            BooleanQuery::new()
                .must(RangeQuery::u64(allowed, Some(1), Some(1)))
                .should(sparse_query),
        ),
    ];
    for query in queries {
        let hits = index.search(query.as_ref(), 2).await.unwrap().hits;
        assert_eq!(
            hits.len(),
            1,
            "one superblock across the entire index: {query}"
        );
        assert!(hits[0].score > 0.9);
    }
}

#[tokio::test]
async fn common_filter_precedes_boolean_text_top_k() {
    let (index, text, allowed, _, _) = fixture().await;
    let query = eligible(
        BooleanQuery::new()
            .must(AllQuery)
            .should(TermQuery::text(text, "alpha")),
        allowed,
    );
    assert_selected(&index, &query, 1, &[10]).await;
}

#[tokio::test]
async fn common_filter_precedes_boolean_bmp_top_k() {
    let (index, _, allowed, bmp, _) = fixture().await;
    let query = eligible(
        BooleanQuery::new()
            .must(AllQuery)
            .should(SparseTermQuery::new(bmp, 0, 1.0)),
        allowed,
    );
    assert_selected(&index, &query, 1, &[10]).await;
}

#[tokio::test]
async fn common_filter_precedes_maxscore_sparse_top_k_in_every_plan() {
    let (index, _, allowed, _, maxscore) = fixture().await;
    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(SparseVectorQuery::new(maxscore, vec![(0, 1.0), (1, 1.0)])),
        Box::new(
            BooleanQuery::new()
                .should(SparseTermQuery::new(maxscore, 0, 1.0))
                .should(SparseTermQuery::new(maxscore, 1, 1.0)),
        ),
        Box::new(
            BooleanQuery::new()
                .must(AllQuery)
                .should(SparseTermQuery::new(maxscore, 0, 1.0)),
        ),
    ];
    for query in queries {
        assert_selected(&index, &eligible(query, allowed), 1, &[10]).await;
    }
}

struct UnexpectedWork;

impl std::fmt::Display for UnexpectedWork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("unexpected payload work")
    }
}

impl Query for UnexpectedWork {
    fn scorer<'a>(&self, _reader: &'a SegmentReader, _limit: usize) -> ScorerFuture<'a> {
        Box::pin(async { Err(Error::Query("payload work should have been skipped".into())) })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        _reader: &'a SegmentReader,
        _limit: usize,
    ) -> Result<Box<dyn Scorer + 'a>> {
        Err(Error::Query("payload work should have been skipped".into()))
    }

    fn count_estimate<'a>(&self, _reader: &'a SegmentReader) -> crate::query::CountFuture<'a> {
        Box::pin(async { Ok(12) })
    }
}

#[tokio::test]
async fn empty_common_filter_skips_remaining_filters_and_scoring_payloads() {
    let (index, _, allowed, _, _) = fixture().await;
    let query = FilteredQuery::new(
        Arc::new(UnexpectedWork),
        vec![
            Arc::new(RangeQuery::u64(allowed, Some(2), Some(2))),
            Arc::new(UnexpectedWork),
        ],
    );
    assert_selected(&index, &query, 1, &[]).await;
}

#[tokio::test]
async fn expired_common_filter_skips_materialization_and_marks_truncation() {
    let (index, _, _, _, _) = fixture().await;
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    let reader = &searcher.segment_readers()[0];
    let query = FilteredQuery::new(Arc::new(UnexpectedWork), vec![Arc::new(UnexpectedWork)]);
    let budget =
        crate::query::SharedThreshold::new().with_deadline(Some(std::time::Instant::now()));
    let options = ScorerOptions {
        shared_threshold: Some(budget.clone()),
        ..Default::default()
    };
    let scorer = query
        .scorer_with_options(reader, 1, options.clone())
        .await
        .unwrap();
    assert_eq!(scorer.doc(), crate::TERMINATED);
    assert!(budget.truncated());
    #[cfg(feature = "sync")]
    assert_eq!(
        query
            .scorer_sync_with_options(reader, 1, options)
            .unwrap()
            .doc(),
        crate::TERMINATED
    );
}

#[cfg(feature = "sync")]
#[tokio::test]
async fn common_filter_supports_direct_sync_scorers() {
    let (index, _, allowed, _, _) = fixture().await;
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    let query = eligible(AllQuery, allowed);
    let mut scorer = query
        .scorer_sync(&searcher.segment_readers()[0], 12)
        .unwrap();
    assert_eq!(scorer.doc(), 10);
    assert_eq!(scorer.advance(), 11);
    assert_eq!(scorer.advance(), crate::TERMINATED);
}
