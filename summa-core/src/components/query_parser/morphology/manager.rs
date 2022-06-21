use std::collections::HashMap;

use crate::components::query_parser::morphology::english::EnglishMorphology;
use crate::components::query_parser::morphology::Morphology;

#[derive(Clone)]
pub struct MorphologyManager {
    morphologies: HashMap<String, Box<dyn Morphology>>,
}

impl Default for MorphologyManager {
    fn default() -> Self {
        let mut morphologies = HashMap::new();
        morphologies.insert("en".to_string(), Box::<EnglishMorphology>::default() as Box<dyn Morphology>);
        MorphologyManager { morphologies }
    }
}

impl MorphologyManager {
    pub fn get(&self, language: &str) -> Option<&Box<dyn Morphology>> {
        self.morphologies.get(language)
    }
}
