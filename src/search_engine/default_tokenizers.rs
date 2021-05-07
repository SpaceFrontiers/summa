use super::stop_words::STOP_WORDS;
use super::summa_tokenizer::SummaTokenizer;
use crate::configs::IndexConfig;
use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, StopWordFilter, TextAnalyzer};

pub fn default_tokenizers(index_config: &IndexConfig) -> [(String, TextAnalyzer); 2] {
    let stop_words = index_config
        .stop_words
        .as_ref()
        .map(|stop_words| stop_words.clone())
        .unwrap_or_else(|| STOP_WORDS.iter().map(|x| x.to_string()).collect());
    [
        (
            "summa".to_owned(),
            TextAnalyzer::from(SummaTokenizer)
                .filter(RemoveLongFilter::limit(100))
                .filter(LowerCaser)
                .filter(StopWordFilter::remove(stop_words.clone())),
        ),
        (
            "default".to_owned(),
            TextAnalyzer::from(SimpleTokenizer)
                .filter(RemoveLongFilter::limit(100))
                .filter(LowerCaser)
                .filter(StopWordFilter::remove(stop_words)),
        ),
    ]
}
