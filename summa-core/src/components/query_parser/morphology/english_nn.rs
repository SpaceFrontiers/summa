use std::sync::Arc;

use parking_lot::Mutex;
use rust_bert::pipelines::ner::NERModel;
use rust_bert::pipelines::pos_tagging::POSModel;

use crate::components::query_parser::morphology::Morphology;

#[derive(Clone)]
pub struct EnglishNNMorphology {
    // ner_model: Arc<Mutex<NERModel>>,
    pos_model: Arc<Mutex<POSModel>>,
}

impl Default for EnglishNNMorphology {
    fn default() -> Self {
        EnglishNNMorphology {
            // ner_model: Arc::new(Mutex::new(NERModel::new(Default::default()).expect("cannot create model"))),
            pos_model: Arc::new(Mutex::new(POSModel::new(Default::default()).expect("cannot create model"))),
        }
    }
}

impl Morphology for EnglishNNMorphology {
    fn detect_ners(&self, phrase: &str) -> Vec<String> {
        vec![]
        /*self.ner_model
        .lock()
        .predict_full_entities(&[phrase])
        .into_iter()
        .next()
        .expect("cannot use model")
        .into_iter()
        .map(|e| e.word)
        .collect()*/
    }

    fn derive_tenses(&self, word: &str) -> Option<String> {
        let pos_tags = self.pos_model.lock().predict(&[word]);
        let pos_tag = pos_tags[0][0].label.clone();
        match pos_tag.as_str() {
            "NN" => Some(pluralize_rs::to_plural(word)),
            "NNS" => Some(pluralize_rs::to_singular(word)),
            _ => None,
        }
    }
}
