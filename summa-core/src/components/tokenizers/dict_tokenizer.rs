use aho_corasick::MatchKind;
use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

/// Tokenize the text by splitting on whitespaces and punctuation.
#[derive(Clone)]
pub struct DictTokenizer {
    ac: aho_corasick::AhoCorasick,
    words: Vec<String>,
    dict: Vec<usize>,
}

impl DictTokenizer {
    pub fn new() -> DictTokenizer {
        let mut synsets = vec![];
        let mut csv_reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_reader(include_bytes!("../../../resources/drugs.csv").as_slice());
        for record in csv_reader.records() {
            let mut synset = vec![];
            for word in record.expect("dictionary is broken").iter() {
                synset.push(word.to_string())
            }
            synsets.push(synset);
        }

        let mut base_offset = 0;
        let mut dict = vec![];
        let words: Vec<String> = synsets
            .into_iter()
            .flat_map(|synset| {
                dict.extend(std::iter::repeat(base_offset).take(synset.len()));
                base_offset += synset.len();
                synset
            })
            .collect();
        let ac = aho_corasick::AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .match_kind(MatchKind::LeftmostLongest)
            .build(words.iter());
        DictTokenizer { ac, words, dict }
    }
}

impl Default for DictTokenizer {
    fn default() -> Self {
        DictTokenizer::new()
    }
}

pub struct DictTokenStream<'a> {
    text: &'a str,
    words: &'a Vec<String>,
    dict: &'a Vec<usize>,
    ah_iter: aho_corasick::FindIter<'a, 'a, usize>,
    token: Token,
}

impl<'a> DictTokenStream<'a> {
    pub fn new(text: &'a str, words: &'a Vec<String>, dict: &'a Vec<usize>, ac: &'a aho_corasick::AhoCorasick) -> DictTokenStream<'a> {
        DictTokenStream {
            text,
            words,
            dict,
            ah_iter: ac.find_iter(text),
            token: Token::default(),
        }
    }
}

impl Tokenizer for DictTokenizer {
    type TokenStream<'a> = DictTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> DictTokenStream<'a> {
        DictTokenStream::new(text, &self.words, &self.dict, &self.ac)
    }
}

impl<'a> TokenStream for DictTokenStream<'a> {
    fn advance(&mut self) -> bool {
        self.token.text.clear();
        self.token.position = self.token.position.wrapping_add(1);
        for pattern in self.ah_iter.by_ref() {
            let properly_beginning = pattern.start() == 0
                || self.text.as_bytes()[pattern.start() - 1].is_ascii_punctuation()
                || self.text.as_bytes()[pattern.start() - 1].is_ascii_whitespace();
            let properly_ending = pattern.end() == self.text.len()
                || self.text.as_bytes()[pattern.end()].is_ascii_punctuation()
                || self.text.as_bytes()[pattern.end()].is_ascii_whitespace();
            if properly_beginning && properly_ending {
                self.token.offset_from = pattern.start();
                self.token.offset_to = pattern.end();
                self.token.text.push_str(&self.words[self.dict[pattern.pattern()]]);
                return true;
            }
        }
        false
    }

    fn token(&self) -> &Token {
        &self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.token
    }
}

#[cfg(test)]
pub mod tests {
    use tantivy::tokenizer::{TextAnalyzer, Token, TokenizerManager};

    use super::DictTokenizer;

    pub fn assert_token(token: &Token, position: usize, text: &str, from: usize, to: usize) {
        assert_eq!(token.position, position, "expected position {} but {:?}", position, token);
        assert_eq!(token.text, text, "expected text {} but {:?}", text, token);
        assert_eq!(token.offset_from, from, "expected offset_from {} but {:?}", from, token);
        assert_eq!(token.offset_to, to, "expected offset_to {} but {:?}", to, token);
    }

    #[test]
    fn test_dict_tokenizer() {
        let tokenizer_manager = TokenizerManager::default();
        tokenizer_manager.register("tokenizer", TextAnalyzer::builder(DictTokenizer::new()).build());
        let mut tokenizer = tokenizer_manager.get("tokenizer").unwrap();
        let mut tokens: Vec<Token> = vec![];
        {
            let mut add_token = |token: &Token| {
                tokens.push(token.clone());
            };
            tokenizer
                .token_stream("FOXP2 gene (not FOXP21) can be correlated with autism spectrum disorder or just autismo")
                .process(&mut add_token);
        }

        assert_eq!(tokens.len(), 1);
        assert_token(&tokens[0], 0, "foxp2", 0, 5);

        let mut tokens: Vec<Token> = vec![];
        {
            let mut add_token = |token: &Token| {
                tokens.push(token.clone());
            };
            tokenizer.token_stream("FOXP2ген связан с аутизмом").process(&mut add_token);
        }

        assert_eq!(tokens.len(), 0);
    }
}
