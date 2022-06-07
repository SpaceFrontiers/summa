use std::str::CharIndices;

use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

/// Tokenize the text by splitting on whitespaces and punctuation.
#[derive(Clone)]
pub struct SummaTokenizer;

pub struct SummaTokenStream<'a> {
    text: &'a str,
    current_char_index: Option<(usize, char)>,
    chars: CharIndices<'a>,
    token: Token,
}

impl Tokenizer for SummaTokenizer {
    type TokenStream<'a> = SummaTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> SummaTokenStream<'a> {
        let mut chars = text.char_indices();
        SummaTokenStream {
            text,
            current_char_index: chars.next(),
            chars,
            token: Token::default(),
        }
    }
}

#[inline]
fn is_cjk(c: &char) -> bool {
    (0x4e00 <= *c as u32 && *c as u32 <= 0x9FFF)
        || (0x3400 <= *c as u32 && *c as u32 <= 0x4DBF)
        || (0x20000 <= *c as u32 && *c as u32 <= 0x2A6DF)
        || (0x2A700 <= *c as u32 && *c as u32 <= 0x2B73F)
        || (0x2B740 <= *c as u32 && *c as u32 <= 0x2B81F)
}

impl<'a> SummaTokenStream<'a> {
    fn search_token_end(&mut self) -> usize {
        let cci = &mut self.current_char_index;
        (&mut self.chars)
            .map(|(offset, c)| {
                *cci = Some((offset, c));
                (offset, c)
            })
            .filter(|(_, c)| !(c.is_alphanumeric() || *c == '#' || *c == '+') || is_cjk(c))
            .map(|(offset, _)| offset)
            .next()
            .unwrap_or(self.text.len())
    }
}

impl<'a> TokenStream for SummaTokenStream<'a> {
    fn advance(&mut self) -> bool {
        self.token.text.clear();
        self.token.position = self.token.position.wrapping_add(1);
        while let Some((offset_from, c)) = self.current_char_index {
            if c.is_alphanumeric() {
                let offset_to = if !is_cjk(&c) {
                    let offset_to = self.search_token_end();
                    if !is_cjk(&self.current_char_index.expect("expected char").1) {
                        self.current_char_index = self.chars.next();
                    }
                    offset_to
                } else {
                    self.current_char_index = self.chars.next();
                    offset_from + c.len_utf8()
                };
                self.token.offset_from = offset_from;
                self.token.offset_to = offset_to;
                self.token.text.push_str(&self.text[offset_from..offset_to]);
                return true;
            }
            self.current_char_index = self.chars.next();
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
    use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, TextAnalyzer, Token, TokenizerManager};

    use super::SummaTokenizer;

    pub fn assert_token(token: &Token, position: usize, text: &str, from: usize, to: usize) {
        assert_eq!(token.position, position, "expected position {} but {:?}", position, token);
        assert_eq!(token.text, text, "expected text {} but {:?}", text, token);
        assert_eq!(token.offset_from, from, "expected offset_from {} but {:?}", from, token);
        assert_eq!(token.offset_to, to, "expected offset_to {} but {:?}", to, token);
    }

    #[test]
    fn test_en_tokenizer() {
        let tokenizer_manager = TokenizerManager::default();
        tokenizer_manager.register(
            "tokenizer",
            TextAnalyzer::builder(SummaTokenizer)
                .filter(RemoveLongFilter::limit(40))
                .filter(LowerCaser)
                .build(),
        );
        let mut tokenizer = tokenizer_manager.get("tokenizer").unwrap();
        let mut tokens: Vec<Token> = vec![];
        {
            let mut add_token = |token: &Token| {
                tokens.push(token.clone());
            };
            tokenizer.token_stream("Hello, world!").process(&mut add_token);
        }

        assert_eq!(tokens.len(), 2);
        assert_token(&tokens[0], 0, "hello", 0, 5);
        assert_token(&tokens[1], 1, "world", 7, 12);
    }

    #[test]
    fn test_zh_tokenizer() {
        let tokenizer_manager = TokenizerManager::default();
        tokenizer_manager.register(
            "tokenizer",
            TextAnalyzer::builder(SummaTokenizer)
                .filter(RemoveLongFilter::limit(40))
                .filter(LowerCaser)
                .build(),
        );
        let mut tokenizer = tokenizer_manager.get("tokenizer").unwrap();
        let mut tokens: Vec<Token> = vec![];
        {
            let mut add_token = |token: &Token| {
                tokens.push(token.clone());
            };
            tokenizer.token_stream("在查hello, worl土d动!").process(&mut add_token);
        }
        assert_eq!(tokens.len(), 7);
        assert_token(&tokens[0], 0, "在", 0, 3);
        assert_token(&tokens[1], 1, "查", 3, 6);
        assert_token(&tokens[2], 2, "hello", 6, 11);
        assert_token(&tokens[3], 3, "worl", 13, 17);
        assert_token(&tokens[4], 4, "土", 17, 20);
        assert_token(&tokens[5], 5, "d", 20, 21);
        assert_token(&tokens[6], 6, "动", 21, 24);
    }

    #[test]
    fn test_zh_tokenizer_2() {
        let tokenizer_manager = TokenizerManager::default();
        tokenizer_manager.register(
            "tokenizer",
            TextAnalyzer::builder(SummaTokenizer)
                .filter(RemoveLongFilter::limit(40))
                .filter(LowerCaser)
                .build(),
        );
        let mut tokenizer = tokenizer_manager.get("tokenizer").unwrap();
        let mut tokens: Vec<Token> = vec![];
        {
            let mut add_token = |token: &Token| {
                tokens.push(token.clone());
            };
            tokenizer.token_stream("Veri 在查hello, c查m p查 查lex  worl土d动!").process(&mut add_token);
        }
        assert_eq!(tokens.len(), 15);
        assert_token(&tokens[0], 0, "veri", 0, 4);
        assert_token(&tokens[1], 1, "在", 5, 8);
        assert_token(&tokens[2], 2, "查", 8, 11);
        assert_token(&tokens[3], 3, "hello", 11, 16);
        assert_token(&tokens[4], 4, "c", 18, 19);
        assert_token(&tokens[5], 5, "查", 19, 22);
        assert_token(&tokens[6], 6, "m", 22, 23);
        assert_token(&tokens[7], 7, "p", 24, 25);
        assert_token(&tokens[8], 8, "查", 25, 28);
        assert_token(&tokens[9], 9, "查", 29, 32);
        assert_token(&tokens[10], 10, "lex", 32, 35);
        assert_token(&tokens[11], 11, "worl", 37, 41);
        assert_token(&tokens[12], 12, "土", 41, 44);
        assert_token(&tokens[13], 13, "d", 44, 45);
        assert_token(&tokens[14], 14, "动", 45, 48);
    }
    #[test]
    fn test_zh_tokenizer_3() {
        let tokenizer_manager = TokenizerManager::default();
        tokenizer_manager.register(
            "tokenizer",
            TextAnalyzer::builder(SummaTokenizer)
                .filter(RemoveLongFilter::limit(40))
                .filter(LowerCaser)
                .build(),
        );
        let mut tokenizer = tokenizer_manager.get("tokenizer").unwrap();
        let mut tokens: Vec<Token> = vec![];
        {
            let mut add_token = |token: &Token| {
                tokens.push(token.clone());
            };
            tokenizer.token_stream("。").process(&mut add_token);
        }
        assert_eq!(tokens.len(), 0);
    }
}
