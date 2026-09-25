//! The complete document universe, including documents with missing fields.
use super::{AllDocSet, CountFuture, DocBitset, DocPredicate, DocSet, Query, Scorer, ScorerFuture};
use crate::segment::SegmentReader;
use crate::{DocId, Score};

/// Matches every document, including documents with missing fields.
#[derive(Debug, Clone, Copy)]
pub struct AllQuery;

impl std::fmt::Display for AllQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("All")
    }
}

impl Query for AllQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, _limit: usize) -> ScorerFuture<'a> {
        let scorer = AllScorer::new(reader.num_docs());
        Box::pin(async move { Ok(Box::new(scorer) as Box<dyn Scorer>) })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        _limit: usize,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        Ok(Box::new(AllScorer::new(reader.num_docs())))
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let count = reader.num_docs();
        Box::pin(async move { Ok(count) })
    }

    fn is_filter(&self) -> bool {
        true
    }

    fn as_doc_predicate<'a>(&self, reader: &'a SegmentReader) -> Option<DocPredicate<'a>> {
        let count = reader.num_docs();
        Some(Box::new(move |doc| doc < count))
    }

    fn as_doc_bitset(&self, reader: &SegmentReader) -> Option<DocBitset> {
        Some(DocBitset::all(reader.num_docs()))
    }
}

struct AllScorer(AllDocSet);

impl AllScorer {
    fn new(count: u32) -> Self {
        Self(AllDocSet::new(count))
    }
}

impl DocSet for AllScorer {
    fn doc(&self) -> DocId {
        self.0.doc()
    }

    fn advance(&mut self) -> DocId {
        self.0.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.0.seek(target)
    }

    fn size_hint(&self) -> u32 {
        self.0.size_hint()
    }
}

impl Scorer for AllScorer {
    fn score(&self) -> Score {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TERMINATED;
    #[tokio::test]
    async fn exclusion_filter_includes_documents_without_metadata_in_every_execution_path() {
        use crate::query::{BooleanQuery, FilteredQuery, RangeQuery};
        use crate::{Document, Index, IndexConfig, IndexWriter, RamDirectory, Schema};
        use std::sync::Arc;
        let mut schema = Schema::builder();
        let date = schema.add_i64_field("date", true, false);
        schema.set_fast(date, true);
        let dir = RamDirectory::new();
        let config = IndexConfig::default();
        let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
            .await
            .unwrap();
        for date_value in [Some(-1), Some(0), None] {
            let mut doc = Document::new();
            if let Some(value) = date_value {
                doc.add_i64(date, value);
            }
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        let index = Index::open(dir, config).await.unwrap();
        let searcher = index.reader().await.unwrap().searcher().await.unwrap();
        let reader = &searcher.segment_readers()[0];
        let filter =
            BooleanQuery::new()
                .must(AllQuery)
                .must_not(RangeQuery::i64(date, Some(0), Some(0)));
        let predicate = filter.as_doc_predicate(reader).unwrap();
        let bits = filter.as_doc_bitset(reader).unwrap();
        for doc in 0..3 {
            assert_eq!(predicate(doc), doc != 1);
            assert_eq!(bits.contains(doc), doc != 1);
        }
        let query = FilteredQuery::new(Arc::new(AllQuery), vec![Arc::new(filter)]);
        let check = |mut scorer: Box<dyn Scorer + '_>| {
            let mut docs = Vec::new();
            while scorer.doc() != TERMINATED {
                docs.push(scorer.doc());
                scorer.advance();
            }
            assert_eq!(docs, [0, 2]);
        };
        check(query.scorer(reader, 10).await.unwrap());
        #[cfg(feature = "sync")]
        check(
            query
                .scorer_sync_with_options(reader, 10, super::super::ScorerOptions::default())
                .unwrap(),
        );
    }

    #[test]
    fn complete_universe_has_no_tail_documents_and_seek_never_rewinds() {
        for count in [0, 1, 63, 64, 65, 129] {
            let bits = DocBitset::all(count);
            let mut scorer = AllScorer::new(count);
            for doc in 0..count {
                assert!(bits.contains(doc));
                assert_eq!(scorer.doc(), doc);
                assert_eq!(scorer.score(), 1.0);
                scorer.advance();
            }
            assert_eq!(bits.next_set_bit(count), None);
            assert_eq!(scorer.doc(), TERMINATED);
            assert_eq!(scorer.advance(), TERMINATED);
            assert_eq!(scorer.seek(0), TERMINATED);
        }
        let mut scorer = AllScorer::new(10);
        assert_eq!(scorer.seek(5), 5);
        assert_eq!(scorer.seek(2), 5);
        assert_eq!(scorer.advance(), 6);
    }
}
