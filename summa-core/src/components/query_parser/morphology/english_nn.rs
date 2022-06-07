use rust_bert::pipelines::ner::{Entity, NERModel};
use rust_bert::pipelines::pos_tagging::POSModel;

use crate::components::query_parser::morphology::Morphology;

#[derive(Default, Clone)]
pub struct EnglishNNMorphology {}

impl Morphology for EnglishNNMorphology {
    fn detect_ners(&self, phrase: &str) -> Vec<String> {
        thread_local! {
            static NER_MODEL: NERModel = NERModel::new(Default::default()).expect("cannot create model");
        }
        NER_MODEL
            .with(|ner_model| ner_model.predict_full_entities(&[phrase]).into_iter().next().expect("cannot use model"))
            .into_iter()
            .map(|e| e.word.to_string())
            .collect()
    }

    fn derive_tenses(&self, word: &str) -> Option<String> {
        thread_local! {
            static POS_MODEL: POSModel = POSModel::new(Default::default()).expect("cannot create model");
        }
        let pos_tag = POS_MODEL.with(|pos_model| {
            let pos_tags = pos_model.predict(&[word]);
            pos_tags[0][0].label.clone()
        });
        match pos_tag.as_str() {
            "NN" => Some(pluralize_rs::to_plural(word)),
            "NNS" => Some(pluralize_rs::to_singular(word)),
            _ => None,
        }
    }
}
