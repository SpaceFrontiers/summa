use super::stop_words::STOP_WORDS;
use super::summa_tokenizer::SummaTokenizer;
use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, StopWordFilter, TextAnalyzer};

pub fn default_tokenizers() -> [(String, TextAnalyzer); 2] {
    [
        (
            "summa".to_string(),
            TextAnalyzer::from(SummaTokenizer)
                .filter(RemoveLongFilter::limit(200))
                .filter(LowerCaser)
                .filter(StopWordFilter::remove(STOP_WORDS.iter().map(|x| x.to_string()).collect())),
        ),
        (
            "default".to_string(),
            TextAnalyzer::from(SimpleTokenizer)
                .filter(RemoveLongFilter::limit(200))
                .filter(LowerCaser)
                .filter(StopWordFilter::remove(STOP_WORDS.iter().map(|x| x.to_string()).collect())),
        ),
    ]
}
