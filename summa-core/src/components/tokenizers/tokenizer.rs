use std::str::CharIndices;

/// Tokenize the text by splitting on whitespaces and punctuation.
#[derive(Clone)]
pub struct Tokenizer;

pub struct TokenStream<'a> {
    skip_list: Option<Vec<(usize, usize)>>,
    skip_iter: usize,
    chars: CharIndices<'a>,
    token: tantivy::tokenizer::Token,
    stacked_char: Option<(char, usize)>,

    base_offset: usize,
}

#[inline]
pub fn accept_char(token: &mut tantivy::tokenizer::Token, c: char, offset: usize) {
    if token.offset_from == usize::MAX {
        token.offset_from = offset;
    }
    token.offset_to = offset + c.len_utf8();
    token.text.push(c);
}

impl<'a> TokenStream<'a> {
    pub fn new(text: &'a str) -> TokenStream<'a> {
        TokenStream {
            skip_list: None,
            skip_iter: 0,
            chars: text.char_indices(),
            token: tantivy::tokenizer::Token::default(),
            stacked_char: None,
            base_offset: 0,
        }
    }

    pub fn new_with_offset_and_position(text: &'a str, offset: usize, position: usize, skip_list: Option<Vec<(usize, usize)>>) -> TokenStream<'a> {
        let token = tantivy::tokenizer::Token {
            position,
            ..Default::default()
        };
        TokenStream {
            skip_list,
            skip_iter: 0,
            chars: text.char_indices(),
            token,
            stacked_char: None,
            base_offset: offset,
        }
    }

    pub fn token(&self) -> &tantivy::tokenizer::Token {
        &self.token
    }

    pub fn token_mut(&mut self) -> &mut tantivy::tokenizer::Token {
        &mut self.token
    }
}

impl tantivy::tokenizer::Tokenizer for Tokenizer {
    type TokenStream<'a> = TokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> TokenStream<'a> {
        TokenStream::new(text)
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

impl<'a> tantivy::tokenizer::TokenStream for TokenStream<'a> {
    fn advance(&mut self) -> bool {
        self.token.text.clear();
        self.token.position = self.token.position.wrapping_add(1);
        self.token.offset_from = usize::MAX;

        if let Some((stacked_char, stacked_offset)) = self.stacked_char.take() {
            accept_char(&mut self.token, stacked_char, self.base_offset + stacked_offset);
            if is_cjk(&stacked_char) {
                return true;
            }
        }

        for (offset, c) in &mut self.chars {
            let real_offset = self.base_offset + offset;
            if let Some(skip_list) = &self.skip_list {
                while self.skip_iter < skip_list.len() && skip_list[self.skip_iter].1 <= real_offset {
                    self.skip_iter += 1;
                }
                if self.skip_iter < skip_list.len() && skip_list[self.skip_iter].0 <= real_offset && real_offset < skip_list[self.skip_iter].1 {
                    continue;
                }
            }

            if is_cjk(&c) {
                if !self.token.text.is_empty() {
                    self.stacked_char = Some((c, offset));
                    return true;
                }
                accept_char(&mut self.token, c, real_offset);
                return true;
            } else if c.is_alphanumeric() || c == '#' || c == '+' {
                accept_char(&mut self.token, c, real_offset);
                continue;
            } else if !self.token.text.is_empty() {
                break;
            }
        }
        !self.token.text.is_empty()
    }

    fn token(&self) -> &tantivy::tokenizer::Token {
        &self.token
    }

    fn token_mut(&mut self) -> &mut tantivy::tokenizer::Token {
        &mut self.token
    }
}

#[cfg(test)]
pub mod tests {
    use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, TextAnalyzer, Token, TokenizerManager};

    use super::Tokenizer;

    pub fn assert_tokenization(tokenizer: &mut TextAnalyzer, text: &str, response: &[Token]) {
        let mut tokens: Vec<Token> = vec![];
        {
            let mut add_token = |token: &Token| {
                tokens.push(token.clone());
            };
            tokenizer.token_stream(text).process(&mut add_token);
        }
        assert_eq!(tokens, response);
    }

    #[test]
    fn test_en_tokenizer() {
        let tokenizer_manager = TokenizerManager::default();
        tokenizer_manager.register(
            "tokenizer",
            TextAnalyzer::builder(Tokenizer).filter(RemoveLongFilter::limit(40)).filter(LowerCaser).build(),
        );
        let mut tokenizer = tokenizer_manager.get("tokenizer").unwrap();

        assert_tokenization(
            &mut tokenizer,
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
    }

    #[test]
    fn test_zh_tokenizer() {
        let tokenizer_manager = TokenizerManager::default();
        tokenizer_manager.register(
            "tokenizer",
            TextAnalyzer::builder(Tokenizer).filter(RemoveLongFilter::limit(40)).filter(LowerCaser).build(),
        );
        let mut tokenizer = tokenizer_manager.get("tokenizer").unwrap();
        assert_tokenization(
            &mut tokenizer,
            "在查hello, worl土d动!",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 3,
                    position: 0,
                    text: "在".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 3,
                    offset_to: 6,
                    position: 1,
                    text: "查".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 6,
                    offset_to: 11,
                    position: 2,
                    text: "hello".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 13,
                    offset_to: 17,
                    position: 3,
                    text: "worl".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 17,
                    offset_to: 20,
                    position: 4,
                    text: "土".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 20,
                    offset_to: 21,
                    position: 5,
                    text: "d".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 21,
                    offset_to: 24,
                    position: 6,
                    text: "动".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "在查土d动",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 3,
                    position: 0,
                    text: "在".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 3,
                    offset_to: 6,
                    position: 1,
                    text: "查".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 6,
                    offset_to: 9,
                    position: 2,
                    text: "土".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 9,
                    offset_to: 10,
                    position: 3,
                    text: "d".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 10,
                    offset_to: 13,
                    position: 4,
                    text: "动".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "Veri 在查hello, c查m p查 查lex  worl土d动!",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 4,
                    position: 0,
                    text: "veri".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 5,
                    offset_to: 8,
                    position: 1,
                    text: "在".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 8,
                    offset_to: 11,
                    position: 2,
                    text: "查".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 11,
                    offset_to: 16,
                    position: 3,
                    text: "hello".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 18,
                    offset_to: 19,
                    position: 4,
                    text: "c".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 19,
                    offset_to: 22,
                    position: 5,
                    text: "查".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 22,
                    offset_to: 23,
                    position: 6,
                    text: "m".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 24,
                    offset_to: 25,
                    position: 7,
                    text: "p".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 25,
                    offset_to: 28,
                    position: 8,
                    text: "查".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 29,
                    offset_to: 32,
                    position: 9,
                    text: "查".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 32,
                    offset_to: 35,
                    position: 10,
                    text: "lex".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 37,
                    offset_to: 41,
                    position: 11,
                    text: "worl".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 41,
                    offset_to: 44,
                    position: 12,
                    text: "土".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 44,
                    offset_to: 45,
                    position: 13,
                    text: "d".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 45,
                    offset_to: 48,
                    position: 14,
                    text: "动".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(&mut tokenizer, "。", &[]);
    }
}
