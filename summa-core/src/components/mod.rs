mod custom_serializer;
mod default_tokenizers;
mod fruit_extractors;
mod index_holder;
mod index_registry;
mod index_writer_holder;
pub mod merge_policies;
pub mod queries;
mod query_parser;
mod segment_attributes;
mod snippet_generator;
mod summa_document;
mod summa_tokenizer;

pub use custom_serializer::NamedFieldDocument;
pub use default_tokenizers::{default_tokenizers, STOP_WORDS};
pub use fruit_extractors::{build_fruit_extractor, FruitExtractor, IntermediateExtractionResult};
pub use index_holder::{cleanup_index, IndexHolder};
pub use index_registry::IndexRegistry;
pub use index_writer_holder::IndexWriterHolder;
pub use query_parser::{MorphologyManager, ProtoQueryParser, QueryParser, QueryParserError};
pub use segment_attributes::SummaSegmentAttributes;
pub use summa_document::{DocumentParsingError, SummaDocument};
pub use summa_tokenizer::SummaTokenizer;

pub mod test_utils {
    use std::default::Default;
    use std::sync::atomic::{AtomicI64, Ordering};

    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use serde_json::json;
    use tantivy::doc;
    use tantivy::schema::{IndexRecordOption, JsonObjectOptions, Schema, TextFieldIndexing, TextOptions, FAST, INDEXED, STORED};

    use crate::components::SummaDocument;

    pub fn create_test_schema() -> Schema {
        let mut schema_builder = Schema::builder();

        schema_builder.add_i64_field("id", FAST | INDEXED | STORED);
        schema_builder.add_i64_field("issued_at", FAST | INDEXED | STORED);
        schema_builder.add_text_field(
            "title",
            TextOptions::default().set_stored().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("summa")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            ),
        );
        schema_builder.add_text_field(
            "body",
            TextOptions::default().set_stored().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("summa")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            ),
        );
        schema_builder.add_text_field(
            "tags",
            TextOptions::default()
                .set_stored()
                .set_indexing_options(TextFieldIndexing::default().set_tokenizer("summa").set_index_option(IndexRecordOption::Basic)),
        );
        schema_builder.add_json_field(
            "metadata",
            JsonObjectOptions::default()
                .set_stored()
                .set_indexing_options(
                    TextFieldIndexing::default()
                        .set_tokenizer("summa_without_stop_words")
                        .set_index_option(IndexRecordOption::Basic),
                )
                .set_expand_dots_enabled(),
        );
        schema_builder.add_text_field(
            "extra",
            TextOptions::default().set_stored().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("summa")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            ),
        );
        schema_builder.build()
    }

    #[inline]
    fn generate_term(rng: &mut SmallRng, prefix: &str, power: usize) -> String {
        if power > 0 {
            format!("{}{}", prefix, rng.gen_range(0..power))
        } else {
            prefix.to_string()
        }
    }

    #[inline]
    fn generate_sentence(rng: &mut SmallRng, prefix: &str, power: usize, length: usize) -> String {
        (0..length).map(|_| generate_term(rng, prefix, power)).join(" ")
    }

    pub fn generate_document<'a>(
        doc_id: Option<i64>,
        rng: &mut SmallRng,
        schema: &Schema,
        title_prefix: &'a str,
        title_power: usize,
        body_prefix: &'a str,
        body_power: usize,
        tag_prefix: &'a str,
        tag_power: usize,
    ) -> SummaDocument<'a> {
        static DOC_ID: AtomicI64 = AtomicI64::new(1);

        let issued_at = 1674041452i64 - rng.gen_range(100..1000);
        let doc_id = doc_id.unwrap_or_else(|| DOC_ID.fetch_add(1, Ordering::SeqCst));

        SummaDocument::TantivyDocument(doc!(
            schema.get_field("id").expect("no expected field") => doc_id,
            schema.get_field("title").expect("no expected field") => generate_sentence(rng, title_prefix, title_power, 3),
            schema.get_field("body").expect("no expected field") => generate_sentence(rng, body_prefix, body_power, 50),
            schema.get_field("tags").expect("no expected field") => generate_sentence(rng, tag_prefix, tag_power, 5),
            schema.get_field("issued_at").expect("no expected field") => issued_at,
            schema.get_field("metadata").expect("no expected field") => json!({"id": doc_id}),
        ))
    }

    pub fn generate_unique_document<'a>(schema: &'a Schema, title: &'a str) -> SummaDocument<'a> {
        generate_document(None, &mut SmallRng::seed_from_u64(42), schema, title, 0, "body", 1000, "tag", 100)
    }

    pub fn generate_documents(schema: &Schema, n: usize) -> Vec<SummaDocument> {
        let mut rng = SmallRng::seed_from_u64(42);
        (0..n)
            .map(|_| generate_document(None, &mut rng, schema, "title", 100, "body", 1000, "tag", 10))
            .collect()
    }

    pub fn generate_documents_with_doc_id_gen_and_rng<'a>(doc_id_gen: AtomicI64, rng: &mut SmallRng, schema: &'a Schema, n: usize) -> Vec<SummaDocument<'a>> {
        (0..n)
            .map(|_| {
                generate_document(
                    Some(doc_id_gen.fetch_add(1, Ordering::SeqCst)),
                    rng,
                    schema,
                    "title",
                    100,
                    "body",
                    1000,
                    "tag",
                    10,
                )
            })
            .collect()
    }
}
