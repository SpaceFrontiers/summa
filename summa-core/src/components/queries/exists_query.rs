use std::ops::RangeInclusive;

use tantivy::json_utils::JsonTermWriter;
use tantivy::query::{BitSetDocSet, ConstScorer, EnableScoring, Explanation, Query, Scorer, Weight};
use tantivy::schema::{Field, IndexRecordOption};
use tantivy::{DocId, Result, Score, SegmentReader, TantivyError, Term};
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
/// let query = ExistsQuery::new(author, "");
/// let count = searcher.search(&query, &Count)?;
/// assert_eq!(count, 2);
/// Ok(())
/// # }
/// # assert!(test().is_ok());
/// ```

pub const JSON_SEGMENT_UPPER_TERMINATOR: u8 = 2u8;
pub const JSON_SEGMENT_UPPER_TERMINATOR_STR: &str = unsafe { std::str::from_utf8_unchecked(&[JSON_SEGMENT_UPPER_TERMINATOR]) };

#[derive(Clone, Debug)]
pub struct ExistsQuery {
    field: Field,
    full_path: String,
}

impl ExistsQuery {
    /// Creates a new ExistsQuery with a given field
    pub fn new(field: Field, full_path: &str) -> Self {
        ExistsQuery {
            field,
            full_path: full_path.to_string(),
        }
    }
}

#[async_trait]
impl Query for ExistsQuery {
    fn weight(&self, _: EnableScoring<'_>) -> Result<Box<dyn Weight>> {
        Ok(Box::new(ExistsWeight {
            field: self.field,
            full_path: self.full_path.clone(),
        }))
    }

    async fn weight_async(&self, enable_scoring: EnableScoring<'_>) -> Result<Box<dyn Weight>> {
        self.weight(enable_scoring)
    }
}

/// Weight associated with the `ExistsQuery` query.
pub struct ExistsWeight {
    field: Field,
    full_path: String,
}

impl ExistsWeight {
    fn get_json_term(&self, json_path: &str) -> Term {
        let mut term = Term::with_capacity(128);
        let json_term_writer = JsonTermWriter::from_field_and_json_path(self.field, json_path, true, &mut term);
        json_term_writer.term().clone()
    }
    fn generate_json_term_range(&self) -> RangeInclusive<Term> {
        let start_term_str = format!("{}\0", self.full_path);
        let end_term_str = format!("{}{}", self.full_path, JSON_SEGMENT_UPPER_TERMINATOR_STR);
        self.get_json_term(&start_term_str)..=self.get_json_term(&end_term_str)
    }
}

#[async_trait]
impl Weight for ExistsWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> Result<Box<dyn Scorer>> {
        let max_doc = reader.max_doc();
        let mut doc_bitset = BitSet::with_max_value(max_doc);

        let inverted_index = reader.inverted_index(self.field)?;
        let terms = inverted_index.terms();
        let mut term_stream = if self.full_path.is_empty() {
            terms.stream()?
        } else {
            let json_term_range = self.generate_json_term_range();
            terms
                .range()
                .ge(json_term_range.start().serialized_value_bytes())
                .le(json_term_range.end().serialized_value_bytes())
                .into_stream()?
        };
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
        let terms = inverted_index.terms();
        let mut term_stream = if self.full_path.is_empty() {
            terms.range().into_stream_async().await?
        } else {
            let json_term_range = self.generate_json_term_range();
            terms
                .range()
                .ge(json_term_range.start().serialized_value_bytes())
                .le(json_term_range.end().serialized_value_bytes())
                .into_stream_async()
                .await?
        };
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
