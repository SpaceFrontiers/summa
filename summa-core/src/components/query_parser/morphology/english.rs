use crate::components::query_parser::morphology::Morphology;

#[derive(Default, Clone)]
pub struct EnglishMorphology {}

impl Morphology for EnglishMorphology {
    fn derive_tenses(&self, word: &str) -> Option<String> {
        let is_singular = pluralize_rs::is_singular(word);
        let is_plural = pluralize_rs::is_plural(word);

        if is_singular {
            Some(pluralize_rs::to_plural(word))
        } else if is_plural {
            Some(pluralize_rs::to_singular(word))
        } else {
            None
        }
    }

    fn detect_ners(&self, _: &str) -> Vec<String> {
        vec![]
    }
}
