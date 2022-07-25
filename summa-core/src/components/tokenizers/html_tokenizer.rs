use std::cell::RefCell;
use std::collections::HashSet;

use super::tokenizer::TokenStream;

/// Tokenize the text by splitting on whitespaces and punctuation.
#[derive(Clone)]
pub struct HtmlTokenizer {
    ignored_tags: HashSet<String>,
    inlined_tags: HashSet<String>,
}

impl HtmlTokenizer {
    pub fn new(ignored_tags: HashSet<String>, inlined_tags: HashSet<String>) -> HtmlTokenizer {
        HtmlTokenizer { ignored_tags, inlined_tags }
    }
}

pub struct HtmlTokenStream<'a> {
    text: &'a str,
    html_tokenizer: xmlparser::Tokenizer<'a>,
    current_nested_token_stream: TokenStream<'a>,
    ignored_tags: &'a HashSet<String>,
    inlined_tags: &'a HashSet<String>,
    position: usize,
    skip_list: RefCell<Option<Vec<(usize, usize)>>>,
    current_state: HtmlTokenizerState,
    next_token: Option<Result<xmlparser::Token<'a>, xmlparser::Error>>,
}

impl HtmlTokenStream<'_> {
    pub fn add_new_skip(&self, start: usize, end: usize) {
        let mut skip_list = self.skip_list.borrow_mut();
        match skip_list.as_mut() {
            None => *skip_list = Some(vec![(start, end)]),
            Some(skip_list) => skip_list.push((start, end)),
        }
    }

    pub fn emit(&mut self, start: usize, end: usize) {
        self.current_nested_token_stream =
            TokenStream::new_with_offset_and_position(&self.text[start..end], start, self.position, self.skip_list.borrow_mut().take());
        self.current_state = HtmlTokenizerState::Emit;
    }

    pub fn skip_tag(&mut self) {
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
                self.next_token = self.html_tokenizer.next();
            }
        }
    }
}

impl tantivy::tokenizer::Tokenizer for HtmlTokenizer {
    type TokenStream<'a> = HtmlTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> HtmlTokenStream<'a> {
        let html_tokenizer = xmlparser::Tokenizer::from_fragment(text, 0..text.len());
        HtmlTokenStream {
            text,
            html_tokenizer,
            current_nested_token_stream: TokenStream::new(""),
            ignored_tags: &self.ignored_tags,
            inlined_tags: &self.inlined_tags,
            position: usize::MAX,
            skip_list: RefCell::default(),
            current_state: HtmlTokenizerState::BeginReading,
            next_token: None,
        }
    }
}

#[derive(Debug)]
enum CollectedToken {
    None,
    Ref { start: usize, end: usize },
}

#[derive(Debug)]
enum HtmlTokenizerState {
    BeginReading,
    CollectToken { collected_token: CollectedToken },
    Emit,
}

impl<'a> tantivy::tokenizer::TokenStream for HtmlTokenStream<'a> {
    fn advance(&mut self) -> bool {
        loop {
            match &self.current_state {
                HtmlTokenizerState::BeginReading => {
                    *self.skip_list.borrow_mut() = None;
                    self.next_token = self.html_tokenizer.next();
                    self.current_state = HtmlTokenizerState::CollectToken {
                        collected_token: CollectedToken::None,
                    };
                }
                HtmlTokenizerState::CollectToken { collected_token } => match self.next_token {
                    Some(next_token) => match next_token {
                        Ok(xmlparser::Token::Declaration { .. })
                        | Ok(xmlparser::Token::ProcessingInstruction { .. })
                        | Ok(xmlparser::Token::Comment { .. })
                        | Ok(xmlparser::Token::DtdStart { .. })
                        | Ok(xmlparser::Token::EmptyDtd { .. })
                        | Ok(xmlparser::Token::DtdEnd { .. })
                        | Ok(xmlparser::Token::Attribute { .. })
                        | Ok(xmlparser::Token::Cdata { .. })
                        | Ok(xmlparser::Token::EntityDeclaration { .. })
                        | Ok(xmlparser::Token::ElementEnd {
                            end: xmlparser::ElementEnd::Open,
                            ..
                        })
                        | Ok(xmlparser::Token::ElementEnd {
                            end: xmlparser::ElementEnd::Empty,
                            ..
                        }) => {
                            self.next_token = self.html_tokenizer.next();
                        }
                        Ok(xmlparser::Token::ElementStart { local: start, .. }) => {
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
                            } else if self.inlined_tags.contains(start.as_str()) {
                                while let Some(Ok(next_token)) = self.html_tokenizer.next() {
                                    if let xmlparser::Token::ElementEnd {
                                        end: xmlparser::ElementEnd::Open,
                                        ..
                                    } = next_token
                                    {
                                        break;
                                    }
                                }
                                self.next_token = self.html_tokenizer.next();
                                continue;
                            }
                            match collected_token {
                                CollectedToken::None => self.current_state = HtmlTokenizerState::BeginReading,
                                CollectedToken::Ref { start, end } => self.emit(*start, *end),
                            }
                        }
                        Ok(xmlparser::Token::ElementEnd {
                            end: xmlparser::ElementEnd::Close(_, local),
                            ..
                        }) => {
                            if self.inlined_tags.contains(local.as_str()) {
                                self.next_token = self.html_tokenizer.next();
                                continue;
                            }
                            match collected_token {
                                CollectedToken::None => self.current_state = HtmlTokenizerState::BeginReading,
                                CollectedToken::Ref { start, end } => self.emit(*start, *end),
                            }
                        }
                        Ok(xmlparser::Token::Text { text }) => {
                            let new_collected_token = match collected_token {
                                CollectedToken::None => CollectedToken::Ref {
                                    start: text.start(),
                                    end: text.end(),
                                },
                                CollectedToken::Ref { start, end } => {
                                    if *end < text.start() {
                                        self.add_new_skip(*end, text.start());
                                    }
                                    CollectedToken::Ref {
                                        start: *start,
                                        end: text.end(),
                                    }
                                }
                            };
                            self.current_state = HtmlTokenizerState::CollectToken {
                                collected_token: new_collected_token,
                            };
                            self.next_token = self.html_tokenizer.next();
                        }
                        Err(_) => match collected_token {
                            CollectedToken::None => self.current_state = HtmlTokenizerState::BeginReading,
                            CollectedToken::Ref { start, end } => self.emit(*start, *end),
                        },
                    },
                    None => match collected_token {
                        CollectedToken::None => return false,
                        CollectedToken::Ref { start, end } => self.emit(*start, *end),
                    },
                },
                HtmlTokenizerState::Emit => {
                    if self.current_nested_token_stream.advance() {
                        self.position = self.current_nested_token_stream.token().position;
                        return true;
                    }
                    self.current_state = HtmlTokenizerState::BeginReading;
                }
            }
        }
    }

    fn token(&self) -> &tantivy::tokenizer::Token {
        self.current_nested_token_stream.token()
    }

    fn token_mut(&mut self) -> &mut tantivy::tokenizer::Token {
        self.current_nested_token_stream.token_mut()
    }
}

#[cfg(test)]
pub mod tests {
    use std::collections::HashSet;

    use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, TextAnalyzer, Token, TokenizerManager};

    use super::HtmlTokenizer;
    use crate::components::tokenizers::tokenizer::tests::assert_tokenization;

    #[test]
    fn test_html_tokenization() {
        let tokenizer_manager = TokenizerManager::default();
        tokenizer_manager.register(
            "tokenizer",
            TextAnalyzer::builder(HtmlTokenizer::new(
                HashSet::from_iter(vec!["formula".to_string()].into_iter()),
                HashSet::from_iter(vec!["sup".to_string()].into_iter()),
            ))
            .filter(RemoveLongFilter::limit(40))
            .filter(LowerCaser)
            .build(),
        );
        let mut tokenizer = tokenizer_manager.get("tokenizer").unwrap();
        let t_ref = &mut tokenizer;
        assert_tokenization(
            t_ref,
            "Hello, world!",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 5,
                    position: 0,
                    text: "hello".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 7,
                    offset_to: 12,
                    position: 1,
                    text: "world".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            t_ref,
            "<article>test1 <t2>test2 TEST3</t2></article>",
            &[
                Token {
                    offset_from: 9,
                    offset_to: 14,
                    position: 0,
                    text: "test1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 19,
                    offset_to: 24,
                    position: 1,
                    text: "test2".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 25,
                    offset_to: 30,
                    position: 2,
                    text: "test3".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            t_ref,
            "<article>test1 test2<p>link link2</p><formula>1 + 2</formula><p>link3 link4</p></article>",
            &[
                Token {
                    offset_from: 9,
                    offset_to: 14,
                    position: 0,
                    text: "test1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 15,
                    offset_to: 20,
                    position: 1,
                    text: "test2".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 23,
                    offset_to: 27,
                    position: 2,
                    text: "link".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 28,
                    offset_to: 33,
                    position: 3,
                    text: "link2".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 64,
                    offset_to: 69,
                    position: 4,
                    text: "link3".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 70,
                    offset_to: 75,
                    position: 5,
                    text: "link4".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            t_ref,
            "test1 test2<p>link link2<formula>1 + 2</formula><p>link3 link4",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 5,
                    position: 0,
                    text: "test1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 6,
                    offset_to: 11,
                    position: 1,
                    text: "test2".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 14,
                    offset_to: 18,
                    position: 2,
                    text: "link".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 19,
                    offset_to: 24,
                    position: 3,
                    text: "link2".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 51,
                    offset_to: 56,
                    position: 4,
                    text: "link3".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 57,
                    offset_to: 62,
                    position: 5,
                    text: "link4".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            t_ref,
            "link link2<formula>1 + 2</formula>link3 link4",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 4,
                    position: 0,
                    text: "link".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 5,
                    offset_to: 10,
                    position: 1,
                    text: "link2".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 34,
                    offset_to: 39,
                    position: 2,
                    text: "link3".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 40,
                    offset_to: 45,
                    position: 3,
                    text: "link4".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            t_ref,
            "link link2<i>link</i>link3 link4",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 4,
                    position: 0,
                    text: "link".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 5,
                    offset_to: 10,
                    position: 1,
                    text: "link2".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 13,
                    offset_to: 17,
                    position: 2,
                    text: "link".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 21,
                    offset_to: 26,
                    position: 3,
                    text: "link3".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 27,
                    offset_to: 32,
                    position: 4,
                    text: "link4".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            t_ref,
            "link link2 <i>link</i>link3 link4",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 4,
                    position: 0,
                    text: "link".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 5,
                    offset_to: 10,
                    position: 1,
                    text: "link2".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 14,
                    offset_to: 18,
                    position: 2,
                    text: "link".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 22,
                    offset_to: 27,
                    position: 3,
                    text: "link3".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 28,
                    offset_to: 33,
                    position: 4,
                    text: "link4".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            t_ref,
            "link link2 <i>link</i> link3 link4",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 4,
                    position: 0,
                    text: "link".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 5,
                    offset_to: 10,
                    position: 1,
                    text: "link2".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 14,
                    offset_to: 18,
                    position: 2,
                    text: "link".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 23,
                    offset_to: 28,
                    position: 3,
                    text: "link3".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 29,
                    offset_to: 34,
                    position: 4,
                    text: "link4".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            t_ref,
            "link link2<i>link</i> link3 link4",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 4,
                    position: 0,
                    text: "link".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 5,
                    offset_to: 10,
                    position: 1,
                    text: "link2".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 13,
                    offset_to: 17,
                    position: 2,
                    text: "link".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 22,
                    offset_to: 27,
                    position: 3,
                    text: "link3".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 28,
                    offset_to: 33,
                    position: 4,
                    text: "link4".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            t_ref,
            "link<sup>1</sup>2 link<sup>3</sup>",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 17,
                    position: 0,
                    text: "link12".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 18,
                    offset_to: 28,
                    position: 1,
                    text: "link3".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            t_ref,
            "link<sup attr=\"1\">1</sup>",
            &[Token {
                offset_from: 0,
                offset_to: 19,
                position: 0,
                text: "link1".to_string(),
                position_length: 1,
            }],
        );
        assert_tokenization(
            t_ref,
            "link<mll:p attr=\"1\">1</mll:p>",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 4,
                    position: 0,
                    text: "link".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 20,
                    offset_to: 21,
                    position: 1,
                    text: "1".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            t_ref,
            "<p>test1 <sup>test2",
            &[
                Token {
                    offset_from: 3,
                    offset_to: 8,
                    position: 0,
                    text: "test1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 14,
                    offset_to: 19,
                    position: 1,
                    text: "test2".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            t_ref,
            "<p>test1<sup>test2",
            &[Token {
                offset_from: 3,
                offset_to: 18,
                position: 0,
                text: "test1test2".to_string(),
                position_length: 1,
            }],
        );
        assert_tokenization(
            t_ref,
            "test1<p <b>>test2</b>",
            &[Token {
                offset_from: 0,
                offset_to: 5,
                position: 0,
                text: "test1".to_string(),
                position_length: 1,
            }],
        );
    }
}
