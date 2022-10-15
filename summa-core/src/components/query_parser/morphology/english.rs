use std::collections::{HashMap, HashSet};

use regex::RegexSet;
use tracing::info;

use crate::components::query_parser::morphology::Morphology;

#[derive(Clone)]
pub struct EnglishMorphology {
    spelling_dict: HashMap<String, String>,
}

impl EnglishMorphology {
    pub fn new() -> Self {
        let mut spelling_set = vec![];
        let mut csv_reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_reader(include_bytes!("../../../../resources/spelling.csv").as_slice());
        for record in csv_reader.records() {
            let record = record.expect("incorrect record");
            let v1 = record[0].to_string();
            let v2 = record[1].to_string();
            spelling_set.push((v1.to_string(), v2.to_string()));
            spelling_set.push((v2, v1));
        }
        info!(
            action = "loaded_spelling_dictionary",
            size = spelling_set.len(),
            header = ?spelling_set[..10].iter().collect::<Vec<_>>()
        );
        EnglishMorphology {
            spelling_dict: HashMap::from_iter(spelling_set),
        }
    }
}

impl Default for EnglishMorphology {
    fn default() -> Self {
        Self::new()
    }
}

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

    fn derive_spelling(&self, word: &str) -> Option<String> {
        self.spelling_dict.get(word).cloned()
    }

    fn detect_ners(&self, _: &str) -> Vec<String> {
        vec![]
    }
}
