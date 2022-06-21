use summa_proto::proto;
use tantivy::query::{DisjunctionMaxQuery, Query, TermQuery};
use tantivy::schema::{Field, FieldType, IndexRecordOption};

mod english;
mod manager;

pub use manager::MorphologyManager;

use crate::components::query_parser::utils::cast_field_to_term;

pub trait Morphology: MorphologyClone + Send + Sync {
    fn derive_tenses(&self, word: &str) -> Option<String>;
    fn derive_query(&self, config: proto::MorphologyConfig, field: &Field, full_path: &str, field_type: &FieldType, text: &str) -> Box<dyn Query> {
        let term = cast_field_to_term(field, full_path, field_type, text, false);
        if let Some(derive_tenses_coefficient) = config.derive_tenses_coefficient {
            if let Some(other_tense_token) = self.derive_tenses(text) {
                let other_term = cast_field_to_term(field, full_path, field_type, &other_tense_token, false);
                let disjunction_query = DisjunctionMaxQuery::with_tie_breaker(
                    vec![
                        Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>,
                        Box::new(TermQuery::new(other_term, IndexRecordOption::WithFreqs)) as Box<dyn Query>,
                    ],
                    derive_tenses_coefficient,
                );
                Box::new(disjunction_query) as Box<dyn Query>
            } else {
                Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>
            }
        } else {
            Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>
        }
    }
    fn detect_ners(&self, phrase: &str) -> Vec<String>;
}

pub trait MorphologyClone {
    fn clone_box(&self) -> Box<dyn Morphology>;
}

impl<T> MorphologyClone for T
where
    T: 'static + Morphology + Clone,
{
    fn clone_box(&self) -> Box<dyn Morphology> {
        Box::new(self.clone())
    }
}

// We can now implement Clone manually by forwarding to clone_box.
impl Clone for Box<dyn Morphology> {
    fn clone(&self) -> Box<dyn Morphology> {
        self.clone_box()
    }
}
