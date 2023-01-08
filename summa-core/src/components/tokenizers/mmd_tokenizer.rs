use std::collections::{HashMap, HashSet};
use std::str::CharIndices;

/// Tokenize the text by splitting on whitespaces and punctuation.
#[derive(Clone)]
pub struct MmdTokenizer {
    skip_commands: HashMap<&'static str, &'static str>,
    skip_tokens: HashSet<&'static str>,
    drop_commands: HashSet<&'static str>,
    known_commands: HashSet<&'static str>,
}

impl Default for MmdTokenizer {
    fn default() -> Self {
        static SKIP_COMMANDS: [(&str, &str); 6] = [
            ("\\begin{table}", "\\end{table}"),
            ("\\(", "\\)"),
            ("\\[", "\\]"),
            ("\\begin{tabular}", "\\end{tabular}"),
            ("\\begin{figure}", "\\end{figure}"),
            ("$$", "$$"),
        ];
        static SKIP_TOKENS: [&str; 33] = [
            "#",
            "##",
            "###",
            "####",
            "#####",
            "######",
            "\\",
            "\\begin{theorem}",
            "\\end{theorem}",
            "\\begin{lemma}",
            "\\end{lemma}",
            "\\begin{itemize}",
            "\\end{itemize}",
            "\\begin{equation}",
            "\\end{equation}",
            "\\begin{equation*}",
            "\\end{equation*}",
            "\\begin{align}",
            "\\end{align}",
            "\\begin{align*}",
            "\\end{align*}",
            "\\begin{split}",
            "\\end{split}",
            "\\begin{split*}",
            "\\end{split*}",
            "\\begin{gather}",
            "\\end{gather}",
            "\\begin{gather*}",
            "\\end{gather*}",
            "\\end{table}",
            "\\end{tabular}",
            "\\end{figure}",
            "\\pagebreak",
        ];
        static DROP_COMMANDS: [&str; 17] = [
            "\\footnote",
            "\\footnotemark",
            "\\underline",
            "\\uline",
            "\\uwave",
            "\\dashuline",
            "\\dotuline",
            "\\sout",
            "\\xout",
            "\\title",
            "\\author",
            "\\section",
            "\\subsection",
            "\\subsubsection",
            "\\textit",
            "\\textbf",
            "\\url",
        ];
        static KNOWN_COMMANDS: [&str; 3] = ["\\pagebreak", "\\begin", "\\end"];
        MmdTokenizer {
            skip_commands: HashMap::from_iter(SKIP_COMMANDS),
            skip_tokens: HashSet::from_iter(SKIP_TOKENS),
            drop_commands: HashSet::from_iter(DROP_COMMANDS),
            known_commands: HashSet::from_iter(KNOWN_COMMANDS),
        }
    }
}

pub struct MmdTokenStream<'a> {
    skip_list: Option<Vec<(usize, usize)>>,
    skip_iter: usize,
    chars: CharIndices<'a>,
    token: tantivy::tokenizer::Token,
    stacked_char: Option<(char, usize)>,
    skip_commands: &'a HashMap<&'static str, &'static str>,
    skip_tokens: &'a HashSet<&'static str>,
    drop_commands: &'a HashSet<&'static str>,
    known_commands: &'a HashSet<&'static str>,
    base_offset: usize,
    maybe_link: bool,
}

#[inline]
pub fn accept_char(token: &mut tantivy::tokenizer::Token, c: char, offset: usize) {
    if token.offset_from == usize::MAX {
        token.offset_from = offset;
    }
    token.offset_to = offset + c.len_utf8();
    token.text.push(c);
}

impl<'a> MmdTokenStream<'a> {
    pub fn new(
        text: &'a str,
        skip_commands: &'a HashMap<&'static str, &'static str>,
        skip_tokens: &'a HashSet<&'static str>,
        drop_commands: &'a HashSet<&'static str>,
        known_commands: &'a HashSet<&'static str>,
    ) -> MmdTokenStream<'a> {
        MmdTokenStream {
            skip_list: None,
            skip_iter: 0,
            chars: text.char_indices(),
            token: tantivy::tokenizer::Token::default(),
            stacked_char: None,
            skip_commands,
            skip_tokens,
            drop_commands,
            known_commands,
            base_offset: 0,
            maybe_link: false,
        }
    }

    pub fn new_with_offset_and_position(
        text: &'a str,
        offset: usize,
        position: usize,
        skip_list: Option<Vec<(usize, usize)>>,
        skip_commands: &'a HashMap<&'static str, &'static str>,
        skip_tokens: &'a HashSet<&'static str>,
        drop_commands: &'a HashSet<&'static str>,
        known_commands: &'a HashSet<&'static str>,
    ) -> MmdTokenStream<'a> {
        let token = tantivy::tokenizer::Token {
            position,
            ..Default::default()
        };
        MmdTokenStream {
            skip_list,
            skip_iter: 0,
            chars: text.char_indices(),
            token,
            stacked_char: None,
            skip_commands,
            skip_tokens,
            drop_commands,
            known_commands,
            base_offset: offset,
            maybe_link: false,
        }
    }

    pub fn token(&self) -> &tantivy::tokenizer::Token {
        &self.token
    }

    pub fn token_mut(&mut self) -> &mut tantivy::tokenizer::Token {
        &mut self.token
    }

    fn advance_token(&mut self, update_position: bool) -> bool {
        self.token.text.clear();
        if update_position {
            self.token.position = self.token.position.wrapping_add(1);
        }
        self.token.offset_from = usize::MAX;
        let mut is_command = false;
        let mut spec_counter = 0;
        let mut start_skipping_round_bracket = false;
        let mut skipped_round_bracket = 0;
        let mut start_skipping_figure_bracket = false;
        let mut skipped_figure_bracket = 0;

        if let Some((stacked_char, stacked_offset)) = self.stacked_char.take() {
            accept_char(&mut self.token, stacked_char, self.base_offset + stacked_offset);
            if is_cjk(&stacked_char) {
                return true;
            }
            if stacked_char == '\\' {
                is_command = true;
            }
            if stacked_char == '[' {
                self.maybe_link = true;
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

            if start_skipping_round_bracket || skipped_round_bracket > 0 {
                start_skipping_round_bracket = false;
                if c == '(' {
                    skipped_round_bracket += 1;
                    continue;
                } else if c == ')' {
                    skipped_round_bracket -= 1;
                    if skipped_round_bracket == 0 {
                        start_skipping_figure_bracket = true;
                    }
                    continue;
                } else if skipped_round_bracket > 0 {
                    continue;
                }
            }

            if start_skipping_figure_bracket || skipped_figure_bracket > 0 {
                start_skipping_figure_bracket = false;
                if c == '{' {
                    skipped_figure_bracket += 1;
                    continue;
                } else if c == '}' {
                    skipped_figure_bracket -= 1;
                    continue;
                } else if skipped_figure_bracket > 0 {
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
            }

            if c == '\\' {
                if !self.token.text.is_empty() {
                    self.stacked_char = Some((c, offset));
                    return true;
                }
                is_command = true;
                accept_char(&mut self.token, c, real_offset);
            } else if c == '[' && !is_command {
                if !self.token.text.is_empty() {
                    self.stacked_char = Some((c, offset));
                    return true;
                }
                self.maybe_link = true;
            } else if c == ']' && self.maybe_link && !is_command {
                self.maybe_link = false;
                start_skipping_round_bracket = true;
            } else if c == '^' || c == '~' {
                self.token.offset_to += 1;
            } else if c == '*' || c == '_' {
                spec_counter += 1;
            } else if c.is_alphanumeric() || c == '#' || c == '+' {
                if spec_counter == 1 {
                    self.stacked_char = Some((c, offset));
                    return true;
                } else if spec_counter > 1 {
                    self.token.offset_to += spec_counter;
                    spec_counter = 0;
                };
                accept_char(&mut self.token, c, real_offset);
            } else if is_command && (c == '(' || c == ')' || c == '[' || c == ']') && self.token.text.len() == 1 {
                accept_char(&mut self.token, c, real_offset);
                break;
            } else if is_command && (c == '{' || c == '}') {
                if self.drop_commands.contains(&self.token.text.as_str()) {
                    is_command = false;
                    self.token.text.clear();
                    self.token.offset_from = usize::MAX;
                    continue;
                } else if c == '{' && !self.known_commands.contains(&self.token.text.as_str()) {
                    break;
                }
                accept_char(&mut self.token, c, real_offset);
                if c == '}' {
                    break;
                }
            } else if !self.token.text.is_empty() {
                break;
            }
        }
        !self.token.text.is_empty()
    }
}

impl tantivy::tokenizer::Tokenizer for MmdTokenizer {
    type TokenStream<'a> = MmdTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> MmdTokenStream<'a> {
        MmdTokenStream::new(text, &self.skip_commands, &self.skip_tokens, &self.drop_commands, &self.known_commands)
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

impl<'a> tantivy::tokenizer::TokenStream for MmdTokenStream<'a> {
    fn advance(&mut self) -> bool {
        let mut result = self.advance_token(true);
        while result {
            if self.skip_tokens.contains(&self.token.text.as_str()) {
                result = self.advance_token(false);
            } else if let Some(end_command) = self.skip_commands.get(self.token.text.as_str()) {
                while result && self.token.text != *end_command {
                    result = self.advance_token(false);
                }
                result = self.advance_token(false);
            } else {
                while self.token.text.starts_with('\\') {
                    self.token.offset_from += 1;
                    self.token.text = self.token.text[1..].to_string()
                }
                if self.token.text == "]" || self.token.text == "}" || self.token.text == ")" {
                    result = self.advance_token(false);
                } else {
                    break;
                }
            }
        }
        result
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

    use super::MmdTokenizer;

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
            TextAnalyzer::builder(MmdTokenizer::default())
                .filter(RemoveLongFilter::limit(40))
                .filter(LowerCaser)
                .build(),
        );
        let mut tokenizer = tokenizer_manager.get("tokenizer").unwrap();
        assert_tokenization(&mut tokenizer, "#", &[]);
        assert_tokenization(
            &mut tokenizer,
            "# Header1",
            &[Token {
                offset_from: 2,
                offset_to: 9,
                position: 0,
                text: "header1".to_string(),
                position_length: 1,
            }],
        );
        assert_tokenization(&mut tokenizer, "\\begin{table}\\end{table}", &[]);
        assert_tokenization(
            &mut tokenizer,
            "\\begin{table}\\end{table}a",
            &[Token {
                offset_from: 24,
                offset_to: 25,
                position: 0,
                text: "a".to_string(),
                position_length: 1,
            }],
        );
        assert_tokenization(&mut tokenizer, "\\begin{table}# Header 1\\end{table}", &[]);
        assert_tokenization(&mut tokenizer, "\\end{table}", &[]);
        assert_tokenization(
            &mut tokenizer,
            "# Header1\nHello, 1 \\ 2 world! \\begin{table}table content\\end{table}\n\\begin{theorem}\ntheorem content\\end{theorem}",
            &[
                Token {
                    offset_from: 2,
                    offset_to: 9,
                    position: 0,
                    text: "header1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 10,
                    offset_to: 15,
                    position: 1,
                    text: "hello".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 17,
                    offset_to: 18,
                    position: 2,
                    text: "1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 21,
                    offset_to: 22,
                    position: 3,
                    text: "2".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 23,
                    offset_to: 28,
                    position: 4,
                    text: "world".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 84,
                    offset_to: 91,
                    position: 5,
                    text: "theorem".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 92,
                    offset_to: 99,
                    position: 6,
                    text: "content".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "# Header1 \\footnote{footnote text}# Header2 \\uline{\\uline{double line}}",
            &[
                Token {
                    offset_from: 2,
                    offset_to: 9,
                    position: 0,
                    text: "header1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 20,
                    offset_to: 28,
                    position: 1,
                    text: "footnote".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 29,
                    offset_to: 33,
                    position: 2,
                    text: "text".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 36,
                    offset_to: 43,
                    position: 3,
                    text: "header2".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 58,
                    offset_to: 64,
                    position: 4,
                    text: "double".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 65,
                    offset_to: 69,
                    position: 5,
                    text: "line".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "\\noncommand \\noncommand2 \\",
            &[
                Token {
                    offset_from: 1,
                    offset_to: 11,
                    position: 0,
                    text: "noncommand".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 13,
                    offset_to: 24,
                    position: 1,
                    text: "noncommand2".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "\\command{weird}",
            &[
                Token {
                    offset_from: 1,
                    offset_to: 8,
                    position: 0,
                    text: "command".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 9,
                    offset_to: 14,
                    position: 1,
                    text: "weird".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "word1 \\(x_1 + x_2\\) \\word2",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 5,
                    position: 0,
                    text: "word1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 21,
                    offset_to: 26,
                    position: 1,
                    text: "word2".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "Love**is**bold",
            &[Token {
                offset_from: 0,
                offset_to: 14,
                position: 0,
                text: "loveisbold".to_string(),
                position_length: 1,
            }],
        );
        assert_tokenization(
            &mut tokenizer,
            "Love*is*bold",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 4,
                    position: 0,
                    text: "love".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 5,
                    offset_to: 7,
                    position: 1,
                    text: "is".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 8,
                    offset_to: 12,
                    position: 2,
                    text: "bold".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "Love **is*bold",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 4,
                    position: 0,
                    text: "love".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 7,
                    offset_to: 9,
                    position: 1,
                    text: "is".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 10,
                    offset_to: 14,
                    position: 2,
                    text: "bold".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "How to do x^2",
            &[
                Token {
                    offset_from: 0,
                    offset_to: 3,
                    position: 0,
                    text: "how".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 4,
                    offset_to: 6,
                    position: 1,
                    text: "to".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 7,
                    offset_to: 9,
                    position: 2,
                    text: "do".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 10,
                    offset_to: 13,
                    position: 3,
                    text: "x2".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(&mut tokenizer, "![]()", &[]);
        assert_tokenization(
            &mut tokenizer,
            "![image text](https://example.com/image.jpg){width=1}",
            &[
                Token {
                    offset_from: 2,
                    offset_to: 7,
                    position: 0,
                    text: "image".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 8,
                    offset_to: 12,
                    position: 1,
                    text: "text".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "[ref] (author)",
            &[
                Token {
                    offset_from: 1,
                    offset_to: 4,
                    position: 0,
                    text: "ref".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 7,
                    offset_to: 13,
                    position: 1,
                    text: "author".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "[ref]test [ref](l)test",
            &[
                Token {
                    offset_from: 1,
                    offset_to: 9,
                    position: 0,
                    text: "reftest".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 11,
                    offset_to: 22,
                    position: 1,
                    text: "reftest".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "\\title{This is title}\n\\author{Author}\n\\section{Section 1}\n\\subsection{Section 1.1}\n\\subsubsection{Section 1.1.1}",
            &[
                Token {
                    offset_from: 7,
                    offset_to: 11,
                    position: 0,
                    text: "this".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 12,
                    offset_to: 14,
                    position: 1,
                    text: "is".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 15,
                    offset_to: 20,
                    position: 2,
                    text: "title".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 30,
                    offset_to: 36,
                    position: 3,
                    text: "author".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 47,
                    offset_to: 54,
                    position: 4,
                    text: "section".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 55,
                    offset_to: 56,
                    position: 5,
                    text: "1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 70,
                    offset_to: 77,
                    position: 6,
                    text: "section".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 78,
                    offset_to: 79,
                    position: 7,
                    text: "1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 80,
                    offset_to: 81,
                    position: 8,
                    text: "1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 98,
                    offset_to: 105,
                    position: 9,
                    text: "section".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 106,
                    offset_to: 107,
                    position: 10,
                    text: "1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 108,
                    offset_to: 109,
                    position: 11,
                    text: "1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 110,
                    offset_to: 111,
                    position: 12,
                    text: "1".to_string(),
                    position_length: 1,
                },
            ],
        );
        assert_tokenization(
            &mut tokenizer,
            "![ref](hehe)-abc{} \\[34\\] \\] \\) \\} 1 ### abc \\(",
            &[
                Token {
                    offset_from: 2,
                    offset_to: 5,
                    position: 0,
                    text: "ref".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 13,
                    offset_to: 16,
                    position: 1,
                    text: "abc".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 35,
                    offset_to: 36,
                    position: 2,
                    text: "1".to_string(),
                    position_length: 1,
                },
                Token {
                    offset_from: 41,
                    offset_to: 44,
                    position: 3,
                    text: "abc".to_string(),
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
            TextAnalyzer::builder(MmdTokenizer::default())
                .filter(RemoveLongFilter::limit(40))
                .filter(LowerCaser)
                .build(),
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
