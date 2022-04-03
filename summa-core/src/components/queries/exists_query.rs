use tantivy::query::{BitSetDocSet, ConstScorer, EnableScoring, Explanation, Query, Scorer, Weight};
use tantivy::schema::{Field, IndexRecordOption};
use tantivy::{DocId, Result, Score, SegmentReader, TantivyError};
use tantivy_common::BitSet;

/// An Exists Query matches all of the documents
/// containing a specific indexed field.
///
/// ```rust
/// use tantivy::collector::Count;
/// use summa_core::components::queries::ExistsQuery;
/// use tantivy::schema::{Schema, TEXT};
/// use tantivy::{doc, Index};
///
/// # fn test() -> tantivy::Result<()> {
/// let mut schema_builder = Schema::builder();
/// let title = schema_builder.add_text_field("title", TEXT);
/// let author = schema_builder.add_text_field("author", TEXT);
/// let schema = schema_builder.build();
/// let index = Index::create_in_ram(schema);
/// {
///     let mut index_writer = index.writer(3_000_000)?;
///     index_writer.add_document(doc!(
///         title => "The Name of the Wind",
///         author => "Patrick Rothfuss"
///     ))?;
///     index_writer.add_document(doc!(
///         title => "The Diary of Muadib",
///     ))?;
///     index_writer.add_document(doc!(
///         title => "A Dairy Cow",
///         author => "John Webster"
///     ))?;
///     index_writer.commit()?;
/// }
///
/// let reader = index.reader()?;
/// let searcher = reader.searcher();
///
/// let query = ExistsQuery::new(author);
/// let count = searcher.search(&query, &Count)?;
/// assert_eq!(count, 2);
/// Ok(())
/// # }
/// # assert!(test().is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct ExistsQuery {
    field: Field,
}

impl ExistsQuery {
    /// Creates a new ExistsQuery with a given field
    pub fn new(field: Field) -> Self {
        ExistsQuery { field }
    }
}

#[async_trait]
impl Query for ExistsQuery {
    fn weight(&self, _: EnableScoring<'_>) -> Result<Box<dyn Weight>> {
        Ok(Box::new(ExistsWeight { field: self.field }))
    }

    async fn weight_async(&self, _: EnableScoring<'_>) -> Result<Box<dyn Weight>> {
        Ok(Box::new(ExistsWeight { field: self.field }))
    }
}

/// Weight associated with the `ExistsQuery` query.
pub struct ExistsWeight {
    field: Field,
}

#[async_trait]
impl Weight for ExistsWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> Result<Box<dyn Scorer>> {
        let max_doc = reader.max_doc();
        let mut doc_bitset = BitSet::with_max_value(max_doc);

        let inverted_index = reader.inverted_index(self.field)?;
        let mut term_stream = inverted_index.terms().stream()?;

        while term_stream.advance() {
            let term_info = term_stream.value();

            let mut block_segment_postings = inverted_index.read_block_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;

            loop {
                let docs = block_segment_postings.docs();

                if docs.is_empty() {
                    break;
                }
                for &doc in block_segment_postings.docs() {
                    doc_bitset.insert(doc);
                }
                block_segment_postings.advance();
            }
        }

        let doc_bitset = BitSetDocSet::from(doc_bitset);
        Ok(Box::new(ConstScorer::new(doc_bitset, boost)))
    }

    async fn scorer_async(&self, reader: &SegmentReader, boost: Score) -> Result<Box<dyn Scorer>> {
        let max_doc = reader.max_doc();
        let mut doc_bitset = BitSet::with_max_value(max_doc);

        let inverted_index = reader.inverted_index_async(self.field).await?;
        let mut term_stream = inverted_index.terms().range().into_stream_async().await?;

        while term_stream.advance() {
            let term_info = term_stream.value();

            let mut block_segment_postings = inverted_index
                .read_block_postings_from_terminfo_async(term_info, IndexRecordOption::Basic)
                .await?;

            loop {
                let docs = block_segment_postings.docs();

                if docs.is_empty() {
                    break;
                }
                for &doc in block_segment_postings.docs() {
                    doc_bitset.insert(doc);
                }
                block_segment_postings.advance();
            }
        }

        let doc_bitset = BitSetDocSet::from(doc_bitset);
        Ok(Box::new(ConstScorer::new(doc_bitset, boost)))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) != doc {
            return Err(TantivyError::InvalidArgument(format!("Document #({}) does not match", doc)));
        }
        Ok(Explanation::new("ExistsQuery", 1.0))
    }
}
