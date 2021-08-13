use super::summa_tokenizer::SummaTokenizer;
use crate::configs::IndexConfig;
use tantivy::tokenizer::{RawTokenizer, LowerCaser, RemoveLongFilter, SimpleTokenizer, StopWordFilter, TextAnalyzer};

pub fn default_tokenizers(index_config: &IndexConfig) -> [(String, TextAnalyzer); 3] {
    let mut summa_tokenizer = TextAnalyzer::from(SummaTokenizer).filter(RemoveLongFilter::limit(100)).filter(LowerCaser);
    let mut default_tokenizer = TextAnalyzer::from(SimpleTokenizer).filter(RemoveLongFilter::limit(100)).filter(LowerCaser);
    let mut raw_tokenizer = TextAnalyzer::from(RawTokenizer).filter(LowerCaser);
    if let Some(stop_words) = index_config.stop_words.as_ref().cloned() {
        summa_tokenizer = summa_tokenizer.filter(StopWordFilter::remove(stop_words.clone()));
        default_tokenizer = default_tokenizer.filter(StopWordFilter::remove(stop_words));
    }
    [
        ("summa".to_owned(), summa_tokenizer),
        ("default".to_owned(), default_tokenizer),
        ("raw".to_owned(), raw_tokenizer),
    ]
}
