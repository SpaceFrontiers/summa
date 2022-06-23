use std::collections::HashSet;

use regex::RegexSet;

use crate::components::query_parser::morphology::Morphology;

#[derive(Default, Clone)]
pub struct EnglishMorphology {}

impl Morphology for EnglishMorphology {
    fn derive_tenses(&self, word: &str) -> Option<String> {
        thread_local! {
            static NOT_A_NOUN: (RegexSet, HashSet<&'static str>) = (RegexSet::new([
                r"\d$",
                r"ing$",
            ]).expect("cannot compile regex"), HashSet::from_iter(crate::components::default_tokenizers::STOP_WORDS.into_iter()));
        }
        NOT_A_NOUN.with(|(not_a_noun_regex, stop_words)| {
            if stop_words.contains(word) || not_a_noun_regex.is_match(word) {
                return None;
            }
            let is_singular = pluralize_rs::is_singular(word);
            let is_plural = pluralize_rs::is_plural(word);
            if is_singular {
                Some(pluralize_rs::to_plural(word))
            } else if is_plural {
                Some(pluralize_rs::to_singular(word))
            } else {
                None
            }
        })
    }

    fn detect_ners(&self, _: &str) -> Vec<String> {
        vec![]
    }
}
