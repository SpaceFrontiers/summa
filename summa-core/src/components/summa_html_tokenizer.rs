use std::collections::HashSet;


use tantivy::tokenizer::{Token, TokenStream, Tokenizer};
use tantivy_common::HasLen;

use crate::components::summa_tokenizer::SummaTokenStream;

/// Tokenize the text by splitting on whitespaces and punctuation.
#[derive(Clone)]
pub struct SummaHtmlTokenizer {
    ignored_tags: HashSet<String>,
}

impl SummaHtmlTokenizer {
    pub fn new(ignored_tags: HashSet<String>) -> SummaHtmlTokenizer {
        SummaHtmlTokenizer { ignored_tags }
    }
}

pub struct SummaHtmlTokenStream<'a> {
    text: &'a str,
    html_tokenizer: xmlparser::Tokenizer<'a>,
    current_nested_token_stream: SummaTokenStream<'a>,
    ignored_tags: &'a HashSet<String>,
    position: usize,
}

impl Tokenizer for SummaHtmlTokenizer {
    type TokenStream<'a> = SummaHtmlTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> SummaHtmlTokenStream<'a> {
        let html_tokenizer = xmlparser::Tokenizer::from_fragment(text, 0..text.len());
        SummaHtmlTokenStream {
            text,
            html_tokenizer,
            current_nested_token_stream: SummaTokenStream::new(""),
            ignored_tags: &self.ignored_tags,
            position: usize::MAX,
        }
    }
}

impl<'a> TokenStream for SummaHtmlTokenStream<'a> {
    fn advance(&mut self) -> bool {
        loop {
            if self.current_nested_token_stream.advance() {
                self.position = self.current_nested_token_stream.token().position;
                return true;
            }
            loop {
                if let Some(Ok(token)) = self.html_tokenizer.next() {
                    match token {
                        xmlparser::Token::ElementStart { local: start, .. } => {
                            if self.ignored_tags.contains(start.as_str()) {
                                let mut depth = 1;
                                while let Some(Ok(next_token)) = self.html_tokenizer.next() {
                                    match next_token {
                                        xmlparser::Token::ElementStart { .. } => {
                                            depth += 1;
                                        }
                                        xmlparser::Token::ElementEnd {
                                            end: xmlparser::ElementEnd::Close(..),
                                            ..
                                        } => {
                                            depth -= 1;
                                        }
                                        _ => {}
                                    }
                                    if depth == 0 {
                                        break;
                                    }
                                }
                            }
                        }
                        xmlparser::Token::Text { text } => {
                            self.current_nested_token_stream =
                                SummaTokenStream::new_with_offset_and_position(&self.text[text.start()..text.end()], text.start(), self.position);
                            break;
                        }
                        _ => {}
                    }
                } else {
                    return false;
                }
            }
        }
    }

    fn token(&self) -> &Token {
        self.current_nested_token_stream.token()
    }

    fn token_mut(&mut self) -> &mut Token {
        self.current_nested_token_stream.token_mut()
    }
}

#[cfg(test)]
pub mod tests {
    use std::collections::HashSet;

    use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, TextAnalyzer, Token, TokenizerManager};

    use crate::components::summa_html_tokenizer::SummaHtmlTokenizer;
    use crate::components::SummaTokenizer;

    pub fn assert_token(token: &Token, position: usize, text: &str, from: usize, to: usize) {
        assert_eq!(token.position, position, "expected position {} but {:?}", position, token);
        assert_eq!(token.text, text, "expected text {} but {:?}", text, token);
        assert_eq!(token.offset_from, from, "expected offset_from {} but {:?}", from, token);
        assert_eq!(token.offset_to, to, "expected offset_to {} but {:?}", to, token);
    }

    #[test]
    fn test_html_tokenization() {
        let tokenizer_manager = TokenizerManager::default();
        tokenizer_manager.register(
            "tokenizer",
            TextAnalyzer::builder(SummaHtmlTokenizer::new(HashSet::from_iter(vec!["formula".to_string()].into_iter())))
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

        let mut tokenizer = tokenizer_manager.get("tokenizer").unwrap();
        let mut tokens: Vec<Token> = vec![];
        {
            let mut add_token = |token: &Token| {
                tokens.push(token.clone());
            };
            tokenizer.token_stream("<article>test1 <t2>test2 TEST3</t2></article>").process(&mut add_token);
        }

        assert_eq!(tokens.len(), 3);
        assert_token(&tokens[0], 0, "test1", 9, 14);
        assert_token(&tokens[1], 1, "test2", 19, 24);
        assert_token(&tokens[2], 2, "test3", 25, 30);

        let mut tokenizer = tokenizer_manager.get("tokenizer").unwrap();
        let mut tokens: Vec<Token> = vec![];
        {
            let mut add_token = |token: &Token| {
                tokens.push(token.clone());
            };
            tokenizer
                .token_stream("<article>test1 test2<p>link link2</p><formula>1 + 2</formula><p>link3 link4</p></article>")
                .process(&mut add_token);
        }

        assert_eq!(tokens.len(), 6);
        assert_eq!(format!("{:?}", tokens), "[Token { offset_from: 9, offset_to: 14, position: 0, text: \"test1\", position_length: 1 }, Token { offset_from: 15, offset_to: 20, position: 1, text: \"test2\", position_length: 1 }, Token { offset_from: 23, offset_to: 27, position: 2, text: \"link\", position_length: 1 }, Token { offset_from: 28, offset_to: 33, position: 3, text: \"link2\", position_length: 1 }, Token { offset_from: 64, offset_to: 69, position: 4, text: \"link3\", position_length: 1 }, Token { offset_from: 70, offset_to: 75, position: 5, text: \"link4\", position_length: 1 }]");

        let mut tokenizer = tokenizer_manager.get("tokenizer").unwrap();
        let mut tokens: Vec<Token> = vec![];
        {
            let mut add_token = |token: &Token| {
                tokens.push(token.clone());
            };
            tokenizer
                .token_stream("test1 test2<p>link link2<formula>1 + 2</formula><p>link3 link4")
                .process(&mut add_token);
        }

        assert_eq!(tokens.len(), 6);
        assert_eq!(format!("{:?}", tokens), "[Token { offset_from: 0, offset_to: 5, position: 0, text: \"test1\", position_length: 1 }, Token { offset_from: 6, offset_to: 11, position: 1, text: \"test2\", position_length: 1 }, Token { offset_from: 14, offset_to: 18, position: 2, text: \"link\", position_length: 1 }, Token { offset_from: 19, offset_to: 24, position: 3, text: \"link2\", position_length: 1 }, Token { offset_from: 51, offset_to: 56, position: 4, text: \"link3\", position_length: 1 }, Token { offset_from: 57, offset_to: 62, position: 5, text: \"link4\", position_length: 1 }]");
    }
}
