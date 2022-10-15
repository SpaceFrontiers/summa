use summa_proto::proto;
use tantivy::query::{DisjunctionMaxQuery, Query, TermQuery};
use tantivy::schema::{Field, FieldType, IndexRecordOption};

mod english;
mod manager;

pub use manager::MorphologyManager;

use crate::components::query_parser::utils::cast_field_to_term;

pub trait Morphology: MorphologyClone + Send + Sync {
    fn derive_tenses(&self, word: &str) -> Option<String>;
    fn derive_spelling(&self, word: &str) -> Option<String>;

    fn derive_query(&self, config: proto::MorphologyConfig, field: &Field, full_path: &str, field_type: &FieldType, text: &str) -> Box<dyn Query> {
        let derive_tenses_coefficient = if let Some(derive_tenses_coefficient) = config.derive_tenses_coefficient {
            derive_tenses_coefficient
        } else {
            let term = cast_field_to_term(field, full_path, field_type, text, false);
            return Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>;
        };
        let mut terms = vec![text.to_string()];
        if let Some(other_tense_text) = self.derive_tenses(text) {
            terms.push(other_tense_text);
        }
        let corrected_spellings: Vec<_> = terms.iter().filter_map(|t| self.derive_spelling(t)).collect();
        terms.extend(corrected_spellings);
        if terms.len() == 1 {
            Box::new(TermQuery::new(
                cast_field_to_term(field, full_path, field_type, &terms[0], false),
                IndexRecordOption::WithFreqs,
            )) as Box<dyn Query>
        } else {
            let queries = terms
                .into_iter()
                .map(|text| {
                    Box::new(TermQuery::new(
                        cast_field_to_term(field, full_path, field_type, &text, false),
                        IndexRecordOption::WithFreqs,
                    )) as Box<dyn Query>
                })
                .collect();
            let disjunction_query = DisjunctionMaxQuery::with_tie_breaker(queries, derive_tenses_coefficient);
            Box::new(disjunction_query) as Box<dyn Query>
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
