//! Tokenizer API for text processing

#[cfg(any(feature = "native", feature = "wasm"))]
mod hf_tokenizer;

#[cfg(feature = "native")]
mod idf_weights;

#[cfg(any(feature = "native", feature = "wasm"))]
pub use hf_tokenizer::{HfTokenizer, TokenizerSource};

#[cfg(feature = "native")]
pub use hf_tokenizer::{TokenizerCache, tokenizer_cache};

#[cfg(feature = "native")]
pub use idf_weights::{IdfWeights, IdfWeightsCache, idf_weights_cache};

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use rust_stemmers::Algorithm;
use serde::{Deserialize, Serialize};
use stop_words::LANGUAGE;

/// A token produced by tokenization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token {
    /// The text content of the token
    pub text: String,
    /// Position in the token stream (0-indexed)
    pub position: u32,
    /// Byte offset from start of original text
    pub offset_from: usize,
    /// Byte offset to end of token in original text
    pub offset_to: usize,
}

impl Token {
    pub fn new(text: String, position: u32, offset_from: usize, offset_to: usize) -> Self {
        Self {
            text,
            position,
            offset_from,
            offset_to,
        }
    }
}

/// Trait for tokenizers
pub trait Tokenizer: Send + Sync + Clone + 'static {
    /// Tokenize the input text into a vector of tokens
    fn tokenize(&self, text: &str) -> Vec<Token>;

    /// Tokenize with an optional caller-supplied hint.
    ///
    /// Static tokenizers ignore the hint. Dynamic tokenizers (see
    /// [`DynamicStemmer`]) interpret it — e.g. as a comma-separated list of
    /// language codes taken from a sibling document field at index time and
    /// from `tokenizer_hint` on the query at search time.
    fn tokenize_hinted(&self, text: &str, hint: Option<&str>) -> Vec<Token> {
        let _ = hint;
        self.tokenize(text)
    }
}

/// Simple tokenizer — splits on whitespace, strips non-alphanumeric, and lowercases.
///
/// "Hello, World!" → ["hello", "world"]
#[derive(Debug, Clone, Default)]
pub struct SimpleTokenizer;

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        tokenize_and_clean(text, std::convert::identity)
    }
}

/// Raw tokenizer — no tokenization at all.
///
/// The entire input text becomes a single token (trimmed).
#[derive(Debug, Clone, Default)]
pub struct RawTokenizer;

impl Tokenizer for RawTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }
        let offset = text.as_ptr() as usize;
        let trimmed_offset = trimmed.as_ptr() as usize - offset;
        vec![Token::new(
            trimmed.to_string(),
            0,
            trimmed_offset,
            trimmed_offset + trimmed.len(),
        )]
    }
}

/// Raw case-insensitive tokenizer — lowercases the entire input without splitting.
///
/// The entire input text becomes a single lowercased token (trimmed).
#[derive(Debug, Clone, Default)]
pub struct RawCiTokenizer;

impl Tokenizer for RawCiTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }
        let offset = text.as_ptr() as usize;
        let trimmed_offset = trimmed.as_ptr() as usize - offset;
        vec![Token::new(
            lowercase_word(trimmed),
            0,
            trimmed_offset,
            trimmed_offset + trimmed.len(),
        )]
    }
}

/// Lowercase a word, preserving all characters (no stripping).
///
/// ASCII fast-path avoids char decoding.
#[inline]
fn lowercase_word(word: &str) -> String {
    if word.is_ascii() {
        if word.bytes().all(|b| !b.is_ascii_uppercase()) {
            return word.to_string();
        }
        let mut s = word.to_string();
        s.make_ascii_lowercase();
        s
    } else {
        word.chars().flat_map(|c| c.to_lowercase()).collect()
    }
}

/// Strip non-alphanumeric characters and lowercase.
///
/// ASCII fast-path iterates bytes directly; falls back to full Unicode
/// `char` iteration only when the word contains non-ASCII bytes.
#[inline]
fn clean_word(word: &str) -> String {
    if word.is_ascii() {
        let bytes = word.as_bytes();
        // Super-fast path: word is already lowercase alphanumeric → single memcpy
        if bytes
            .iter()
            .all(|&b| b.is_ascii_lowercase() || b.is_ascii_digit())
        {
            return word.to_string();
        }
        // ASCII path – byte iteration, no char decoding
        let mut result = String::with_capacity(bytes.len());
        for &b in bytes {
            if b.is_ascii_alphanumeric() {
                result.push(b.to_ascii_lowercase() as char);
            }
        }
        result
    } else {
        // Unicode fallback
        word.chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect()
    }
}

/// Shared tokenization logic: split on whitespace, clean (remove punctuation + lowercase),
/// then apply a transform function to produce the final token text.
///
/// Used by `SimpleTokenizer` (identity transform) and `StemmerTokenizer` (stem transform).
/// The transform receives an owned `String` so identity transforms avoid extra allocations.
fn tokenize_and_clean(text: &str, transform: impl Fn(String) -> String) -> Vec<Token> {
    tokenize_and_clean_filtered(text, |word| Some(transform(word)))
}

/// Like [`tokenize_and_clean`], but `transform` may drop a word by returning
/// `None`. A dropped word still consumes a position, so the surviving tokens
/// keep their original distances (a phrase indexed as `quantum@0 art@3`
/// keeps the gap of the two stop words between them).
fn tokenize_and_clean_filtered(
    text: &str,
    transform: impl Fn(String) -> Option<String>,
) -> Vec<Token> {
    let mut tokens = Vec::with_capacity(text.len() / 5);
    let mut position = 0u32;
    for (offset, word) in split_whitespace_with_offsets(text) {
        if !word.is_empty() {
            let cleaned = clean_word(word);
            if !cleaned.is_empty() {
                if let Some(text) = transform(cleaned) {
                    tokens.push(Token::new(text, position, offset, offset + word.len()));
                }
                position += 1;
            }
        }
    }
    tokens
}

/// Split text on whitespace, returning (byte-offset, word) pairs.
///
/// Uses pointer arithmetic on the subslices returned by `split_whitespace`
/// instead of the previous O(n)-per-word `find()` approach.
fn split_whitespace_with_offsets(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let base = text.as_ptr() as usize;
    text.split_whitespace()
        .map(move |word| (word.as_ptr() as usize - base, word))
}

/// Supported stemmer languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
#[derive(Default)]
pub enum Language {
    Arabic,
    Danish,
    Dutch,
    #[default]
    English,
    Finnish,
    French,
    German,
    Greek,
    Hungarian,
    Italian,
    Norwegian,
    Portuguese,
    Romanian,
    Russian,
    Spanish,
    Swedish,
    Tamil,
    Turkish,
}

impl Language {
    fn to_algorithm(self) -> Algorithm {
        match self {
            Language::Arabic => Algorithm::Arabic,
            Language::Danish => Algorithm::Danish,
            Language::Dutch => Algorithm::Dutch,
            Language::English => Algorithm::English,
            Language::Finnish => Algorithm::Finnish,
            Language::French => Algorithm::French,
            Language::German => Algorithm::German,
            Language::Greek => Algorithm::Greek,
            Language::Hungarian => Algorithm::Hungarian,
            Language::Italian => Algorithm::Italian,
            Language::Norwegian => Algorithm::Norwegian,
            Language::Portuguese => Algorithm::Portuguese,
            Language::Romanian => Algorithm::Romanian,
            Language::Russian => Algorithm::Russian,
            Language::Spanish => Algorithm::Spanish,
            Language::Swedish => Algorithm::Swedish,
            Language::Tamil => Algorithm::Tamil,
            Language::Turkish => Algorithm::Turkish,
        }
    }

    fn to_stop_words_language(self) -> LANGUAGE {
        match self {
            Language::Arabic => LANGUAGE::Arabic,
            Language::Danish => LANGUAGE::Danish,
            Language::Dutch => LANGUAGE::Dutch,
            Language::English => LANGUAGE::English,
            Language::Finnish => LANGUAGE::Finnish,
            Language::French => LANGUAGE::French,
            Language::German => LANGUAGE::German,
            Language::Greek => LANGUAGE::Greek,
            Language::Hungarian => LANGUAGE::Hungarian,
            Language::Italian => LANGUAGE::Italian,
            Language::Norwegian => LANGUAGE::Norwegian,
            Language::Portuguese => LANGUAGE::Portuguese,
            Language::Romanian => LANGUAGE::Romanian,
            Language::Russian => LANGUAGE::Russian,
            Language::Spanish => LANGUAGE::Spanish,
            Language::Swedish => LANGUAGE::Swedish,
            Language::Tamil => LANGUAGE::Tamil,
            Language::Turkish => LANGUAGE::Turkish,
        }
    }
}

/// Stop word filter tokenizer - wraps another tokenizer and filters out stop words
///
/// Uses the stop-words crate for language-specific stop word lists.
#[derive(Debug, Clone)]
pub struct StopWordTokenizer<T: Tokenizer> {
    inner: T,
    stop_words: HashSet<String>,
}

use std::collections::HashSet;

impl<T: Tokenizer> StopWordTokenizer<T> {
    /// Create a new stop word tokenizer wrapping the given tokenizer
    pub fn new(inner: T, language: Language) -> Self {
        let stop_words: HashSet<String> = stop_words::get(language.to_stop_words_language())
            .iter()
            .map(|s| s.to_string())
            .collect();
        Self { inner, stop_words }
    }

    /// Create with English stop words
    pub fn english(inner: T) -> Self {
        Self::new(inner, Language::English)
    }

    /// Create with custom stop words
    pub fn with_custom_stop_words(inner: T, stop_words: HashSet<String>) -> Self {
        Self { inner, stop_words }
    }

    /// Check if a word is a stop word
    pub fn is_stop_word(&self, word: &str) -> bool {
        self.stop_words.contains(word)
    }
}

impl<T: Tokenizer> Tokenizer for StopWordTokenizer<T> {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        self.inner
            .tokenize(text)
            .into_iter()
            .filter(|token| !self.stop_words.contains(token.text.as_str()))
            .collect()
    }
}

/// Stemming tokenizer - splits on whitespace, lowercases, and applies stemming
///
/// Uses the Snowball stemming algorithm via rust-stemmers.
/// Supports multiple languages including English, German, French, Spanish, etc.
#[derive(Debug, Clone)]
pub struct StemmerTokenizer {
    language: Language,
}

impl StemmerTokenizer {
    /// Create a new stemmer tokenizer for the given language
    pub fn new(language: Language) -> Self {
        Self { language }
    }

    /// Create a new English stemmer tokenizer
    pub fn english() -> Self {
        Self::new(Language::English)
    }
}

impl Default for StemmerTokenizer {
    fn default() -> Self {
        Self::english()
    }
}

impl Tokenizer for StemmerTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let stemmer = rust_stemmers::Stemmer::create(self.language.to_algorithm());
        tokenize_and_clean(text, |s| stemmer.stem(&s).into_owned())
    }
}

/// Multi-language stemmer that can select language dynamically
///
/// This tokenizer holds stemmers for multiple languages and can tokenize
/// text using a specific language selected at runtime.
#[derive(Debug, Clone)]
pub struct MultiLanguageStemmer {
    default_language: Language,
}

impl MultiLanguageStemmer {
    /// Create a new multi-language stemmer with the given default language
    pub fn new(default_language: Language) -> Self {
        Self { default_language }
    }

    /// Tokenize text using a specific language
    pub fn tokenize_with_language(&self, text: &str, language: Language) -> Vec<Token> {
        let stemmer = rust_stemmers::Stemmer::create(language.to_algorithm());
        tokenize_and_clean(text, |s| stemmer.stem(&s).into_owned())
    }

    /// Get the default language
    pub fn default_language(&self) -> Language {
        self.default_language
    }
}

impl Default for MultiLanguageStemmer {
    fn default() -> Self {
        Self::new(Language::English)
    }
}

impl Tokenizer for MultiLanguageStemmer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        self.tokenize_with_language(text, self.default_language)
    }
}

/// Language-aware tokenizer that can be configured per-field
///
/// This allows selecting the stemmer language based on document metadata,
/// such as a "language" field in the document.
#[derive(Clone)]
pub struct LanguageAwareTokenizer<F>
where
    F: Fn(&str) -> Language + Clone + Send + Sync + 'static,
{
    language_selector: F,
    stemmer: MultiLanguageStemmer,
}

impl<F> LanguageAwareTokenizer<F>
where
    F: Fn(&str) -> Language + Clone + Send + Sync + 'static,
{
    /// Create a new language-aware tokenizer with a custom language selector
    ///
    /// The selector function receives a language hint (e.g., from a document field)
    /// and returns the appropriate Language to use for stemming.
    ///
    /// # Example
    /// ```ignore
    /// let tokenizer = LanguageAwareTokenizer::new(|hint| {
    ///     match hint {
    ///         "en" | "english" => Language::English,
    ///         "de" | "german" => Language::German,
    ///         "ru" | "russian" => Language::Russian,
    ///         _ => Language::English,
    ///     }
    /// });
    /// ```
    pub fn new(language_selector: F) -> Self {
        Self {
            language_selector,
            stemmer: MultiLanguageStemmer::default(),
        }
    }

    /// Tokenize text with a language hint
    ///
    /// The hint is passed to the language selector to determine which stemmer to use.
    pub fn tokenize_with_hint(&self, text: &str, language_hint: &str) -> Vec<Token> {
        let language = (self.language_selector)(language_hint);
        self.stemmer.tokenize_with_language(text, language)
    }
}

impl<F> Tokenizer for LanguageAwareTokenizer<F>
where
    F: Fn(&str) -> Language + Clone + Send + Sync + 'static,
{
    fn tokenize(&self, text: &str) -> Vec<Token> {
        // Default to English when no hint is provided
        self.stemmer.tokenize_with_language(text, Language::English)
    }

    fn tokenize_hinted(&self, text: &str, hint: Option<&str>) -> Vec<Token> {
        match hint {
            Some(hint) => self.tokenize_with_hint(text, hint),
            None => Tokenizer::tokenize(self, text),
        }
    }
}

/// Parse a language string into a Language enum
///
/// Supports common language codes and names.
pub fn parse_language(s: &str) -> Language {
    match s.to_lowercase().as_str() {
        "ar" | "arabic" => Language::Arabic,
        "da" | "danish" => Language::Danish,
        "nl" | "dutch" => Language::Dutch,
        "en" | "english" => Language::English,
        "fi" | "finnish" => Language::Finnish,
        "fr" | "french" => Language::French,
        "de" | "german" => Language::German,
        "el" | "greek" => Language::Greek,
        "hu" | "hungarian" => Language::Hungarian,
        "it" | "italian" => Language::Italian,
        "no" | "norwegian" => Language::Norwegian,
        "pt" | "portuguese" => Language::Portuguese,
        "ro" | "romanian" => Language::Romanian,
        "ru" | "russian" => Language::Russian,
        "es" | "spanish" => Language::Spanish,
        "sv" | "swedish" => Language::Swedish,
        "ta" | "tamil" => Language::Tamil,
        "tr" | "turkish" => Language::Turkish,
        _ => Language::English, // Default fallback
    }
}

/// Parse a language string into a Language, returning `None` for unknown values.
///
/// Accepts ISO 639-1 codes and English language names, case-insensitively.
/// Unlike [`parse_language`], unknown input is not silently mapped to English.
pub fn parse_language_opt(s: &str) -> Option<Language> {
    Some(match s.trim().to_lowercase().as_str() {
        "ar" | "arabic" => Language::Arabic,
        "da" | "danish" => Language::Danish,
        "nl" | "dutch" => Language::Dutch,
        "en" | "english" => Language::English,
        "fi" | "finnish" => Language::Finnish,
        "fr" | "french" => Language::French,
        "de" | "german" => Language::German,
        "el" | "greek" => Language::Greek,
        "hu" | "hungarian" => Language::Hungarian,
        "it" | "italian" => Language::Italian,
        "no" | "norwegian" => Language::Norwegian,
        "pt" | "portuguese" => Language::Portuguese,
        "ro" | "romanian" => Language::Romanian,
        "ru" | "russian" => Language::Russian,
        "es" | "spanish" => Language::Spanish,
        "sv" | "swedish" => Language::Swedish,
        "ta" | "tamil" => Language::Tamil,
        "tr" | "turkish" => Language::Turkish,
        _ => return None,
    })
}

/// Writing system of a token, used to route it to a stemmer of the same script.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    Latin,
    Cyrillic,
    Greek,
    Arabic,
    Tamil,
    /// Any other script (CJK, Hebrew, digits-only tokens, ...)
    Other,
}

impl Script {
    /// Script of the first alphabetic character of `token`; `Other` when none.
    pub fn of_token(token: &str) -> Script {
        let Some(c) = token.chars().find(|c| c.is_alphabetic()) else {
            return Script::Other;
        };
        Script::of_char(c)
    }

    fn of_char(c: char) -> Script {
        match c as u32 {
            // Basic Latin, Latin-1 Supplement, Latin Extended-A/B, Latin Extended Additional
            0x0041..=0x024F | 0x1E00..=0x1EFF => Script::Latin,
            0x0370..=0x03FF | 0x1F00..=0x1FFF => Script::Greek,
            0x0400..=0x052F => Script::Cyrillic,
            0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF => Script::Arabic,
            0x0B80..=0x0BFF => Script::Tamil,
            _ => Script::Other,
        }
    }
}

impl Language {
    /// Script the Snowball algorithm of this language operates on.
    pub fn script(self) -> Script {
        match self {
            Language::Russian => Script::Cyrillic,
            Language::Greek => Script::Greek,
            Language::Arabic => Script::Arabic,
            Language::Tamil => Script::Tamil,
            _ => Script::Latin,
        }
    }
}

thread_local! {
    static STEMMER_CACHE: std::cell::RefCell<HashMap<Language, rust_stemmers::Stemmer>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Stem an owned, already cleaned token, reusing its allocation when the
/// stemmer leaves it unchanged (the common case for short and stop-like
/// words, and for every token of a script the stemmer does not touch).
#[inline]
fn stem_owned(stemmer: &rust_stemmers::Stemmer, word: String) -> String {
    match stemmer.stem(&word) {
        std::borrow::Cow::Borrowed(_) => word,
        std::borrow::Cow::Owned(stemmed) => stemmed,
    }
}

/// Run `f` with the cached Snowball stemmers for `languages`, in order.
///
/// The thread-local cache is borrowed once per tokenization call instead of
/// once per token, so the hot loop pays no `RefCell` borrow or hash lookup.
fn with_stemmers<R>(languages: &[Language], f: impl FnOnce(&[&rust_stemmers::Stemmer]) -> R) -> R {
    STEMMER_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        for language in languages {
            cache
                .entry(*language)
                .or_insert_with(|| rust_stemmers::Stemmer::create(language.to_algorithm()));
        }
        let stemmers: Vec<&rust_stemmers::Stemmer> =
            languages.iter().map(|language| &cache[language]).collect();
        f(&stemmers)
    })
}

/// Word segmentation of a [`DynamicStemmer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Segmenter {
    /// Split on whitespace, strip every non-alphanumeric character
    /// (`float-zero` → `floatzero`, no CJK segmentation).
    #[default]
    Simple,
    /// UAX #29 word boundaries (`float-zero` → `float`, `zero`; `p53`,
    /// `co2` and `10.1007` stay whole), character bigrams over runs of Han,
    /// Hiragana and Katakana, and diacritic folding of Latin, Cyrillic and
    /// Greek tokens after stemming (`résumé` → `resume`).
    Unicode,
}

/// Whether `c` belongs to a script that is bigrammed instead of stemmed.
#[inline]
fn is_cjk_char(c: char) -> bool {
    matches!(
        c as u32,
        0x3040..=0x30FF      // Hiragana, Katakana
            | 0x31F0..=0x31FF // Katakana phonetic extensions
            | 0x3400..=0x4DBF // CJK extension A
            | 0x4E00..=0x9FFF // CJK unified ideographs
            | 0xF900..=0xFAFF // CJK compatibility ideographs
            | 0xFF66..=0xFF9F // half-width Katakana
            | 0x20000..=0x2FA1F // CJK extensions B..F, compatibility supplement
    )
}

/// Fold diacritics of Latin, Cyrillic and Greek text: compatibility
/// decomposition, then every combining mark is dropped. Other scripts are
/// returned unchanged (their combining marks are letters in their own right).
fn fold_diacritics(word: String) -> String {
    if word.is_ascii() {
        return word;
    }
    if !matches!(
        Script::of_token(&word),
        Script::Latin | Script::Cyrillic | Script::Greek
    ) {
        return word;
    }
    use unicode_normalization::UnicodeNormalization;
    use unicode_normalization::char::is_combining_mark;
    word.nfkd()
        .filter(|c| !is_combining_mark(*c))
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// UAX #29 tokenization with CJK bigrams and diacritic folding. `transform`
/// receives each cleaned (lowercased, punctuation-free) non-CJK word and may
/// drop it by returning `None`; dropped words still consume a position.
/// Folding happens after the transform, so stop lists and stemmers see the
/// original letters.
fn tokenize_unicode_filtered(
    text: &str,
    transform: impl Fn(String) -> Option<String>,
) -> Vec<Token> {
    use unicode_segmentation::UnicodeSegmentation;

    let mut tokens = Vec::with_capacity(text.len() / 5);
    let mut position = 0u32;
    // Contiguous run of CJK characters: (byte offset, char).
    let mut run: Vec<(usize, char)> = Vec::new();
    let mut run_end = 0usize;

    fn flush_run(run: &mut Vec<(usize, char)>, tokens: &mut Vec<Token>, position: &mut u32) {
        match run.len() {
            0 => {}
            1 => {
                let (offset, c) = run[0];
                tokens.push(Token::new(
                    c.to_string(),
                    *position,
                    offset,
                    offset + c.len_utf8(),
                ));
                *position += 1;
            }
            _ => {
                for pair in run.windows(2) {
                    let (start, a) = pair[0];
                    let (next, b) = pair[1];
                    let mut text = String::with_capacity(a.len_utf8() + b.len_utf8());
                    text.push(a);
                    text.push(b);
                    tokens.push(Token::new(text, *position, start, next + b.len_utf8()));
                    *position += 1;
                }
            }
        }
        run.clear();
    }

    for (offset, word) in text.unicode_word_indices() {
        if word.chars().all(is_cjk_char) {
            // UAX #29 yields one word per ideograph but keeps kana runs
            // together; either way the characters join the current run when
            // they are adjacent in the text.
            if !run.is_empty() && offset != run_end {
                flush_run(&mut run, &mut tokens, &mut position);
            }
            let mut at = offset;
            for c in word.chars() {
                run.push((at, c));
                at += c.len_utf8();
            }
            run_end = at;
            continue;
        }
        flush_run(&mut run, &mut tokens, &mut position);
        let cleaned = clean_word(word);
        if cleaned.is_empty() {
            continue;
        }
        if let Some(out) = transform(cleaned) {
            tokens.push(Token::new(
                fold_diacritics(out),
                position,
                offset,
                offset + word.len(),
            ));
        }
        position += 1;
    }
    flush_run(&mut run, &mut tokens, &mut position);
    tokens
}

/// Stop words of a language (NLTK lists), shared for the process lifetime.
fn stop_word_set(language: Language) -> Option<&'static HashSet<String>> {
    static SETS: std::sync::OnceLock<RwLock<HashMap<Language, &'static HashSet<String>>>> =
        std::sync::OnceLock::new();
    let sets = SETS.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(set) = sets.read().get(&language) {
        return Some(set);
    }
    let set: &'static HashSet<String> = Box::leak(Box::new(
        stop_words::get(language.to_stop_words_language())
            .iter()
            .map(|word| word.to_string())
            .collect(),
    ));
    Some(*sets.write().entry(language).or_insert(set))
}

/// Stemmer whose language is selected per call from the tokenizer hint.
///
/// The hint is a comma-separated list of language codes or names (`"ru,en"`).
/// Each token is stemmed with the first hinted language whose script matches
/// the token's script; tokens of a script no hinted language covers are kept
/// as cleaned (lowercased, punctuation stripped) text. Without a hint the
/// `default` language applies, or plain cleaning when it is `None`.
///
/// Because Snowball stemmers are script-local, one field can hold mixed
/// Cyrillic/Latin documents and still stem each part correctly.
///
/// With `stop_words` enabled, the stop words of the language a token is
/// routed to are dropped before stemming. Dropped words keep consuming
/// positions, so phrase distances are preserved (`"quantum of the art"`
/// becomes `quantum@0 art@3`).
///
/// Declared in SDL as
/// `text<stem(by: <field>, default: <language|simple>, stop_words: <bool>, segmenter: <simple|unicode>)>`;
/// see [`TokenizerSpec`].
#[derive(Debug, Clone, Default)]
pub struct DynamicStemmer {
    default: Option<Language>,
    stop_words: bool,
    segmenter: Segmenter,
    /// A spec without `by` has no per-document language and treats query
    /// hints the same way: every text goes through `default`.
    ignore_hints: bool,
}

impl DynamicStemmer {
    /// Create a dynamic stemmer; `default` applies when no hint is present.
    /// Stop words are kept unless [`DynamicStemmer::with_stop_words`] is set;
    /// words are split by [`Segmenter::Simple`] unless
    /// [`DynamicStemmer::with_segmenter`] is set.
    pub fn new(default: Option<Language>) -> Self {
        Self {
            default,
            stop_words: false,
            segmenter: Segmenter::Simple,
            ignore_hints: false,
        }
    }

    /// Ignore language hints (index-time field values and query hints
    /// alike): the tokenizer always applies `default`. This is the
    /// `stem(...)` spec without `by`.
    pub fn with_ignored_hints(mut self, ignore: bool) -> Self {
        self.ignore_hints = ignore;
        self
    }

    /// Whether hints are ignored (see [`DynamicStemmer::with_ignored_hints`]).
    pub fn ignores_hints(&self) -> bool {
        self.ignore_hints
    }

    /// Choose the word segmentation (see [`Segmenter`]).
    pub fn with_segmenter(mut self, segmenter: Segmenter) -> Self {
        self.segmenter = segmenter;
        self
    }

    /// The word segmentation in use.
    pub fn segmenter(&self) -> Segmenter {
        self.segmenter
    }

    /// Drop the routed language's stop words (positions are preserved).
    pub fn with_stop_words(mut self, enabled: bool) -> Self {
        self.stop_words = enabled;
        self
    }

    /// Whether stop words are dropped.
    pub fn strips_stop_words(&self) -> bool {
        self.stop_words
    }

    /// Default language used when no hint is given.
    pub fn default_language(&self) -> Option<Language> {
        self.default
    }

    /// Parse a hint into the ordered list of recognised languages.
    pub fn parse_hint(hint: &str) -> Vec<Language> {
        let mut languages = Vec::new();
        for part in hint.split(',') {
            if let Some(language) = parse_language_opt(part)
                && !languages.contains(&language)
            {
                languages.push(language);
            }
        }
        languages
    }

    fn segment(&self, text: &str, transform: impl Fn(String) -> Option<String>) -> Vec<Token> {
        match self.segmenter {
            Segmenter::Simple => tokenize_and_clean_filtered(text, transform),
            Segmenter::Unicode => tokenize_unicode_filtered(text, transform),
        }
    }

    fn tokenize_with_languages(&self, text: &str, languages: &[Language]) -> Vec<Token> {
        if languages.is_empty() {
            return self.segment(text, Some);
        }
        // Stop lists follow the same script routing as the stemmers: a token
        // is checked against (and then stemmed with) the first hinted
        // language of its script.
        let stops: Vec<Option<&'static HashSet<String>>> = languages
            .iter()
            .map(|language| self.stop_words.then(|| stop_word_set(*language)).flatten())
            .collect();
        with_stemmers(languages, |stemmers| {
            self.segment(text, |s| {
                let script = Script::of_token(&s);
                let Some(index) = languages
                    .iter()
                    .position(|language| language.script() == script)
                else {
                    return Some(s);
                };
                if stops[index].is_some_and(|set| set.contains(s.as_str())) {
                    return None;
                }
                Some(stem_owned(stemmers[index], s))
            })
        })
    }
}

impl Tokenizer for DynamicStemmer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        match self.default {
            Some(language) => self.tokenize_with_languages(text, &[language]),
            None => self.segment(text, Some),
        }
    }

    fn tokenize_hinted(&self, text: &str, hint: Option<&str>) -> Vec<Token> {
        if self.ignore_hints {
            return Tokenizer::tokenize(self, text);
        }
        match hint.map(str::trim).filter(|hint| !hint.is_empty()) {
            Some(hint) => {
                let languages = Self::parse_hint(hint);
                if languages.is_empty() {
                    // Hinted but nothing recognised: fall back to the default.
                    Tokenizer::tokenize(self, text)
                } else {
                    self.tokenize_with_languages(text, &languages)
                }
            }
            None => Tokenizer::tokenize(self, text),
        }
    }
}

/// Parsed form of a tokenizer name in the schema.
///
/// Plain names refer to a registered tokenizer (`simple`, `en_stem`, ...).
/// `stem(by: <field>, default: <language|simple>, stop_words: <bool>)`
/// declares a [`DynamicStemmer`] whose per-document hint is read from
/// `<field>` in the same document. Without `by` the same options apply to
/// every document and query (no language routing; `default: simple` is
/// then plain unicode tokenization with stop words and no stemming). The canonical string form is stored in
/// `FieldEntry::tokenizer`, so index metadata needs no new field; specs
/// without `stop_words` render exactly as before.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerSpec {
    /// A registered tokenizer name.
    Named(String),
    /// Dynamic stemmer hinted by another field of the document, or, without
    /// `by`, a fixed tokenizer with the same options (stop words, unicode
    /// segmentation) that ignores hints.
    DynamicStem {
        /// Field whose text values supply the language hint; `None` = no
        /// per-document language, hints are ignored, `default` always applies.
        by: Option<String>,
        /// Language applied when the hint field is absent; `None` = simple.
        default: Option<Language>,
        /// Drop the routed language's stop words (positions keep their gaps).
        stop_words: bool,
        /// Word segmentation; `Simple` is the historic behaviour.
        segmenter: Segmenter,
    },
}

impl TokenizerSpec {
    /// Parse a tokenizer name or `stem(...)` spec.
    pub fn parse(spec: &str) -> Result<TokenizerSpec, String> {
        let spec = spec.trim();
        let Some(rest) = spec.strip_prefix("stem(") else {
            if spec.is_empty() || spec.contains(['(', ')', ':', ',']) {
                return Err(format!("invalid tokenizer spec '{spec}'"));
            }
            return Ok(TokenizerSpec::Named(spec.to_string()));
        };
        let Some(params) = rest.strip_suffix(')') else {
            return Err(format!("tokenizer spec '{spec}' is missing ')'"));
        };
        let mut by = None;
        let mut default = None;
        let mut stop_words = false;
        let mut segmenter = Segmenter::Simple;
        for param in params.split(',') {
            let param = param.trim();
            if param.is_empty() {
                continue;
            }
            let Some((key, value)) = param.split_once(':') else {
                return Err(format!(
                    "tokenizer spec '{spec}': parameter '{param}' must be 'key: value'"
                ));
            };
            let (key, value) = (key.trim(), value.trim());
            match key {
                "by" if !value.is_empty() => by = Some(value.to_string()),
                "by" => return Err(format!("tokenizer spec '{spec}': 'by' needs a field name")),
                "default" => {
                    default = match value {
                        "simple" | "none" => None,
                        other => Some(parse_language_opt(other).ok_or_else(|| {
                            format!("tokenizer spec '{spec}': unknown default language '{other}'")
                        })?),
                    };
                }
                "stop_words" => {
                    stop_words = match value {
                        "true" => true,
                        "false" => false,
                        other => {
                            return Err(format!(
                                "tokenizer spec '{spec}': 'stop_words' must be true or false, got '{other}'"
                            ));
                        }
                    };
                }
                "segmenter" => {
                    segmenter = match value {
                        "simple" => Segmenter::Simple,
                        "unicode" => Segmenter::Unicode,
                        other => {
                            return Err(format!(
                                "tokenizer spec '{spec}': 'segmenter' must be simple or unicode, got '{other}'"
                            ));
                        }
                    };
                }
                other => {
                    return Err(format!(
                        "tokenizer spec '{spec}': unknown parameter '{other}'"
                    ));
                }
            }
        }
        Ok(TokenizerSpec::DynamicStem {
            by,
            default,
            stop_words,
            segmenter,
        })
    }

    /// Field whose values hint the tokenizer, for dynamic specs.
    pub fn hint_field(&self) -> Option<&str> {
        match self {
            TokenizerSpec::Named(_) => None,
            TokenizerSpec::DynamicStem { by, .. } => by.as_deref(),
        }
    }

    /// Build the tokenizer described by a dynamic spec.
    pub fn dynamic_tokenizer(&self) -> Option<BoxedTokenizer> {
        match self {
            TokenizerSpec::Named(_) => None,
            TokenizerSpec::DynamicStem {
                by,
                default,
                stop_words,
                segmenter,
            } => Some(Box::new(
                DynamicStemmer::new(*default)
                    .with_stop_words(*stop_words)
                    .with_segmenter(*segmenter)
                    .with_ignored_hints(by.is_none()),
            )),
        }
    }
}

impl std::fmt::Display for TokenizerSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerSpec::Named(name) => f.write_str(name),
            TokenizerSpec::DynamicStem {
                by,
                default,
                stop_words,
                segmenter,
            } => {
                let default = match default {
                    None => "simple".to_string(),
                    Some(language) => language_code(*language).to_string(),
                };
                match by {
                    Some(by) => write!(f, "stem(by: {by}, default: {default}")?,
                    None => write!(f, "stem(default: {default}")?,
                }
                if *stop_words {
                    write!(f, ", stop_words: true")?;
                }
                if *segmenter == Segmenter::Unicode {
                    write!(f, ", segmenter: unicode")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// ISO 639-1 code of a stemmer language.
pub fn language_code(language: Language) -> &'static str {
    match language {
        Language::Arabic => "ar",
        Language::Danish => "da",
        Language::Dutch => "nl",
        Language::English => "en",
        Language::Finnish => "fi",
        Language::French => "fr",
        Language::German => "de",
        Language::Greek => "el",
        Language::Hungarian => "hu",
        Language::Italian => "it",
        Language::Norwegian => "no",
        Language::Portuguese => "pt",
        Language::Romanian => "ro",
        Language::Russian => "ru",
        Language::Spanish => "es",
        Language::Swedish => "sv",
        Language::Tamil => "ta",
        Language::Turkish => "tr",
    }
}

/// Boxed tokenizer for dynamic dispatch
pub type BoxedTokenizer = Box<dyn TokenizerClone>;

pub trait TokenizerClone: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<Token>;
    /// Hinted tokenization; see [`Tokenizer::tokenize_hinted`].
    fn tokenize_hinted(&self, text: &str, hint: Option<&str>) -> Vec<Token>;
    fn clone_box(&self) -> BoxedTokenizer;
}

impl<T: Tokenizer> TokenizerClone for T {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        Tokenizer::tokenize(self, text)
    }

    fn tokenize_hinted(&self, text: &str, hint: Option<&str>) -> Vec<Token> {
        Tokenizer::tokenize_hinted(self, text, hint)
    }

    fn clone_box(&self) -> BoxedTokenizer {
        Box::new(self.clone())
    }
}

impl Clone for BoxedTokenizer {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Registry for named tokenizers
///
/// Allows registering tokenizers by name and retrieving them for use during indexing.
/// Pre-registers common tokenizers: "simple", "raw", "raw_ci", "en_stem", etc.
#[derive(Clone)]
pub struct TokenizerRegistry {
    tokenizers: Arc<RwLock<HashMap<String, BoxedTokenizer>>>,
    /// Parsed `stem(...)` specs, keyed by their spec string. Query conversion
    /// resolves the field tokenizer on every request; parsing the spec each
    /// time was measurable at query rates.
    dynamic: Arc<RwLock<HashMap<String, BoxedTokenizer>>>,
}

impl TokenizerRegistry {
    /// Create a new tokenizer registry with default tokenizers registered
    pub fn new() -> Self {
        let registry = Self {
            tokenizers: Arc::new(RwLock::new(HashMap::new())),
            dynamic: Arc::new(RwLock::new(HashMap::new())),
        };
        registry.register_defaults();
        registry
    }

    /// Register default tokenizers
    fn register_defaults(&self) {
        // Basic tokenizers ("default" is the documented alias of "simple")
        self.register("simple", SimpleTokenizer);
        self.register("default", SimpleTokenizer);
        self.register("raw", RawTokenizer);
        self.register("raw_ci", RawCiTokenizer);

        // English stemmer variants
        self.register("en_stem", StemmerTokenizer::new(Language::English));
        self.register("english", StemmerTokenizer::new(Language::English));

        // Other language stemmers
        self.register("ar_stem", StemmerTokenizer::new(Language::Arabic));
        self.register("arabic", StemmerTokenizer::new(Language::Arabic));
        self.register("da_stem", StemmerTokenizer::new(Language::Danish));
        self.register("danish", StemmerTokenizer::new(Language::Danish));
        self.register("nl_stem", StemmerTokenizer::new(Language::Dutch));
        self.register("dutch", StemmerTokenizer::new(Language::Dutch));
        self.register("fi_stem", StemmerTokenizer::new(Language::Finnish));
        self.register("finnish", StemmerTokenizer::new(Language::Finnish));
        self.register("fr_stem", StemmerTokenizer::new(Language::French));
        self.register("french", StemmerTokenizer::new(Language::French));
        self.register("de_stem", StemmerTokenizer::new(Language::German));
        self.register("german", StemmerTokenizer::new(Language::German));
        self.register("el_stem", StemmerTokenizer::new(Language::Greek));
        self.register("greek", StemmerTokenizer::new(Language::Greek));
        self.register("hu_stem", StemmerTokenizer::new(Language::Hungarian));
        self.register("hungarian", StemmerTokenizer::new(Language::Hungarian));
        self.register("it_stem", StemmerTokenizer::new(Language::Italian));
        self.register("italian", StemmerTokenizer::new(Language::Italian));
        self.register("no_stem", StemmerTokenizer::new(Language::Norwegian));
        self.register("norwegian", StemmerTokenizer::new(Language::Norwegian));
        self.register("pt_stem", StemmerTokenizer::new(Language::Portuguese));
        self.register("portuguese", StemmerTokenizer::new(Language::Portuguese));
        self.register("ro_stem", StemmerTokenizer::new(Language::Romanian));
        self.register("romanian", StemmerTokenizer::new(Language::Romanian));
        self.register("ru_stem", StemmerTokenizer::new(Language::Russian));
        self.register("russian", StemmerTokenizer::new(Language::Russian));
        self.register("es_stem", StemmerTokenizer::new(Language::Spanish));
        self.register("spanish", StemmerTokenizer::new(Language::Spanish));
        self.register("sv_stem", StemmerTokenizer::new(Language::Swedish));
        self.register("swedish", StemmerTokenizer::new(Language::Swedish));
        self.register("ta_stem", StemmerTokenizer::new(Language::Tamil));
        self.register("tamil", StemmerTokenizer::new(Language::Tamil));
        self.register("tr_stem", StemmerTokenizer::new(Language::Turkish));
        self.register("turkish", StemmerTokenizer::new(Language::Turkish));

        // Stop word filtered tokenizers (lowercase + stop words)
        self.register(
            "en_stop",
            StopWordTokenizer::new(SimpleTokenizer, Language::English),
        );
        self.register(
            "de_stop",
            StopWordTokenizer::new(SimpleTokenizer, Language::German),
        );
        self.register(
            "fr_stop",
            StopWordTokenizer::new(SimpleTokenizer, Language::French),
        );
        self.register(
            "ru_stop",
            StopWordTokenizer::new(SimpleTokenizer, Language::Russian),
        );
        self.register(
            "es_stop",
            StopWordTokenizer::new(SimpleTokenizer, Language::Spanish),
        );

        // Stop word + stemming tokenizers
        self.register(
            "en_stem_stop",
            StopWordTokenizer::new(StemmerTokenizer::new(Language::English), Language::English),
        );
        self.register(
            "de_stem_stop",
            StopWordTokenizer::new(StemmerTokenizer::new(Language::German), Language::German),
        );
        self.register(
            "fr_stem_stop",
            StopWordTokenizer::new(StemmerTokenizer::new(Language::French), Language::French),
        );
        self.register(
            "ru_stem_stop",
            StopWordTokenizer::new(StemmerTokenizer::new(Language::Russian), Language::Russian),
        );
        self.register(
            "es_stem_stop",
            StopWordTokenizer::new(StemmerTokenizer::new(Language::Spanish), Language::Spanish),
        );
    }

    /// Register a tokenizer with a name
    pub fn register<T: Tokenizer>(&self, name: &str, tokenizer: T) {
        let mut tokenizers = self.tokenizers.write();
        tokenizers.insert(name.to_string(), Box::new(tokenizer));
    }

    /// Get a tokenizer by name or by a `stem(by: ..., default: ...)` spec.
    ///
    /// Dynamic specs are parsed once per distinct spec string and cached; a
    /// malformed spec is not cached and yields `None` on every call.
    pub fn get(&self, name: &str) -> Option<BoxedTokenizer> {
        if name.starts_with("stem(") {
            if let Some(tokenizer) = self.dynamic.read().get(name) {
                return Some(tokenizer.clone());
            }
            let tokenizer = TokenizerSpec::parse(name)
                .ok()
                .and_then(|spec| spec.dynamic_tokenizer())?;
            self.dynamic
                .write()
                .entry(name.to_string())
                .or_insert_with(|| tokenizer.clone());
            return Some(tokenizer);
        }
        let tokenizers = self.tokenizers.read();
        tokenizers.get(name).cloned()
    }

    /// Check if a tokenizer is registered
    pub fn contains(&self, name: &str) -> bool {
        let tokenizers = self.tokenizers.read();
        tokenizers.contains_key(name)
    }

    /// List all registered tokenizer names
    pub fn names(&self) -> Vec<String> {
        let tokenizers = self.tokenizers.read();
        tokenizers.keys().cloned().collect()
    }
}

impl Default for TokenizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer() {
        let tokenizer = SimpleTokenizer;
        let tokens = Tokenizer::tokenize(&tokenizer, "Hello World");

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[1].position, 1);
    }

    #[test]
    fn test_raw_tokenizer() {
        let tokenizer = RawTokenizer;
        // Entire input becomes one token, preserving case and punctuation
        let tokens = Tokenizer::tokenize(&tokenizer, "Hello, World!");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "Hello, World!");
        assert_eq!(tokens[0].position, 0);
    }

    #[test]
    fn test_raw_tokenizer_trims() {
        let tokenizer = RawTokenizer;
        let tokens = Tokenizer::tokenize(&tokenizer, "  spaced  ");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "spaced");
        assert_eq!(tokens[0].offset_from, 2);
    }

    #[test]
    fn test_raw_tokenizer_empty() {
        let tokenizer = RawTokenizer;
        assert!(Tokenizer::tokenize(&tokenizer, "").is_empty());
        assert!(Tokenizer::tokenize(&tokenizer, "   ").is_empty());
    }

    #[test]
    fn test_raw_ci_tokenizer() {
        let tokenizer = RawCiTokenizer;
        // Entire input lowercased as one token
        let tokens = Tokenizer::tokenize(&tokenizer, "Hello, World!");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "hello, world!");
        assert_eq!(tokens[0].position, 0);
    }

    #[test]
    fn test_raw_ci_tokenizer_preserves_structure() {
        let tokenizer = RawCiTokenizer;
        let tokens = Tokenizer::tokenize(&tokenizer, "HTTPS://Example.COM/Page");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "https://example.com/page");
    }

    #[test]
    fn test_simple_tokenizer_strips_punctuation() {
        let tokenizer = SimpleTokenizer;
        let tokens = Tokenizer::tokenize(&tokenizer, "Hello, World!");

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_empty_text() {
        let tokenizer = SimpleTokenizer;
        let tokens = Tokenizer::tokenize(&tokenizer, "");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_stemmer_tokenizer_english() {
        let tokenizer = StemmerTokenizer::english();
        let tokens = Tokenizer::tokenize(&tokenizer, "Dogs are running quickly");

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "dog"); // dogs -> dog
        assert_eq!(tokens[1].text, "are"); // are -> are
        assert_eq!(tokens[2].text, "run"); // running -> run
        assert_eq!(tokens[3].text, "quick"); // quickly -> quick
    }

    #[test]
    fn test_stemmer_tokenizer_preserves_offsets() {
        let tokenizer = StemmerTokenizer::english();
        let tokens = Tokenizer::tokenize(&tokenizer, "Running dogs");

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "run");
        assert_eq!(tokens[0].offset_from, 0);
        assert_eq!(tokens[0].offset_to, 7); // "Running" is 7 chars
        assert_eq!(tokens[1].text, "dog");
        assert_eq!(tokens[1].offset_from, 8);
        assert_eq!(tokens[1].offset_to, 12); // "dogs" is 4 chars
    }

    #[test]
    fn test_stemmer_tokenizer_german() {
        let tokenizer = StemmerTokenizer::new(Language::German);
        let tokens = Tokenizer::tokenize(&tokenizer, "Häuser Bücher");

        assert_eq!(tokens.len(), 2);
        // German stemmer should stem these plural forms
        assert_eq!(tokens[0].text, "haus"); // häuser -> haus
        assert_eq!(tokens[1].text, "buch"); // bücher -> buch
    }

    #[test]
    fn test_stemmer_tokenizer_russian() {
        let tokenizer = StemmerTokenizer::new(Language::Russian);
        let tokens = Tokenizer::tokenize(&tokenizer, "бегущие собаки");

        assert_eq!(tokens.len(), 2);
        // Russian stemmer should stem these
        assert_eq!(tokens[0].text, "бегущ"); // бегущие -> бегущ
        assert_eq!(tokens[1].text, "собак"); // собаки -> собак
    }

    #[test]
    fn test_multi_language_stemmer() {
        let stemmer = MultiLanguageStemmer::new(Language::English);

        // Test with English
        let tokens = stemmer.tokenize_with_language("running dogs", Language::English);
        assert_eq!(tokens[0].text, "run");
        assert_eq!(tokens[1].text, "dog");

        // Test with German
        let tokens = stemmer.tokenize_with_language("Häuser Bücher", Language::German);
        assert_eq!(tokens[0].text, "haus");
        assert_eq!(tokens[1].text, "buch");

        // Test with Russian
        let tokens = stemmer.tokenize_with_language("бегущие собаки", Language::Russian);
        assert_eq!(tokens[0].text, "бегущ");
        assert_eq!(tokens[1].text, "собак");
    }

    #[test]
    fn test_language_aware_tokenizer() {
        let tokenizer = LanguageAwareTokenizer::new(parse_language);

        // English hint
        let tokens = tokenizer.tokenize_with_hint("running dogs", "en");
        assert_eq!(tokens[0].text, "run");
        assert_eq!(tokens[1].text, "dog");

        // German hint
        let tokens = tokenizer.tokenize_with_hint("Häuser Bücher", "de");
        assert_eq!(tokens[0].text, "haus");
        assert_eq!(tokens[1].text, "buch");

        // Russian hint
        let tokens = tokenizer.tokenize_with_hint("бегущие собаки", "russian");
        assert_eq!(tokens[0].text, "бегущ");
        assert_eq!(tokens[1].text, "собак");
    }

    #[test]
    fn test_parse_language() {
        assert_eq!(parse_language("en"), Language::English);
        assert_eq!(parse_language("english"), Language::English);
        assert_eq!(parse_language("English"), Language::English);
        assert_eq!(parse_language("de"), Language::German);
        assert_eq!(parse_language("german"), Language::German);
        assert_eq!(parse_language("ru"), Language::Russian);
        assert_eq!(parse_language("russian"), Language::Russian);
        assert_eq!(parse_language("unknown"), Language::English); // fallback
    }

    #[test]
    fn test_tokenizer_registry_defaults() {
        let registry = TokenizerRegistry::new();

        // Check default tokenizers are registered
        assert!(registry.contains("simple"));
        assert!(registry.contains("raw"));
        assert!(registry.contains("raw_ci"));
        assert!(registry.contains("raw"));
        assert!(registry.contains("raw_ci"));
        assert!(registry.contains("en_stem"));
        assert!(registry.contains("german"));
        assert!(registry.contains("russian"));
    }

    #[test]
    fn test_tokenizer_registry_get() {
        let registry = TokenizerRegistry::new();

        // Get and use a tokenizer
        let tokenizer = registry.get("en_stem").unwrap();
        let tokens = tokenizer.tokenize("running dogs");
        assert_eq!(tokens[0].text, "run");
        assert_eq!(tokens[1].text, "dog");

        // Get German stemmer
        let tokenizer = registry.get("german").unwrap();
        let tokens = tokenizer.tokenize("Häuser Bücher");
        assert_eq!(tokens[0].text, "haus");
        assert_eq!(tokens[1].text, "buch");
    }

    #[test]
    fn test_tokenizer_registry_custom() {
        let registry = TokenizerRegistry::new();

        // Register a custom tokenizer
        registry.register("my_tokenizer", SimpleTokenizer);

        assert!(registry.contains("my_tokenizer"));
        let tokenizer = registry.get("my_tokenizer").unwrap();
        let tokens = tokenizer.tokenize("Hello World");
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_tokenizer_registry_nonexistent() {
        let registry = TokenizerRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_stop_word_tokenizer_english() {
        let tokenizer = StopWordTokenizer::english(SimpleTokenizer);
        let tokens = Tokenizer::tokenize(&tokenizer, "The quick brown fox jumps over the lazy dog");

        // "the", "over" are stop words and should be filtered
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(!texts.contains(&"the"));
        assert!(!texts.contains(&"over"));
        assert!(texts.contains(&"quick"));
        assert!(texts.contains(&"brown"));
        assert!(texts.contains(&"fox"));
        assert!(texts.contains(&"jumps"));
        assert!(texts.contains(&"lazy"));
        assert!(texts.contains(&"dog"));
    }

    #[test]
    fn test_stop_word_tokenizer_with_stemmer() {
        // Note: StopWordTokenizer filters AFTER stemming, so stop words
        // that get stemmed may not be filtered. For proper stop word + stemming,
        // filter stop words before stemming or use a stemmed stop word list.
        let tokenizer = StopWordTokenizer::new(StemmerTokenizer::english(), Language::English);
        let tokens = Tokenizer::tokenize(&tokenizer, "elephants galaxies quantum");

        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        // Stemmed forms should be present (these are not stop words)
        assert!(texts.contains(&"eleph")); // elephants -> eleph
        assert!(texts.contains(&"galaxi")); // galaxies -> galaxi
        assert!(texts.contains(&"quantum")); // quantum -> quantum
    }

    #[test]
    fn test_stop_word_tokenizer_german() {
        let tokenizer = StopWordTokenizer::new(SimpleTokenizer, Language::German);
        let tokens = Tokenizer::tokenize(&tokenizer, "Der Hund und die Katze");

        // "der", "und", "die" are German stop words
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(!texts.contains(&"der"));
        assert!(!texts.contains(&"und"));
        assert!(!texts.contains(&"die"));
        assert!(texts.contains(&"hund"));
        assert!(texts.contains(&"katze"));
    }

    #[test]
    fn test_stop_word_tokenizer_custom() {
        let custom_stops: HashSet<String> = ["foo", "bar"].iter().map(|s| s.to_string()).collect();
        let tokenizer = StopWordTokenizer::with_custom_stop_words(SimpleTokenizer, custom_stops);
        let tokens = Tokenizer::tokenize(&tokenizer, "foo baz bar qux");

        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(!texts.contains(&"foo"));
        assert!(!texts.contains(&"bar"));
        assert!(texts.contains(&"baz"));
        assert!(texts.contains(&"qux"));
    }

    #[test]
    fn test_stop_word_tokenizer_is_stop_word() {
        let tokenizer = StopWordTokenizer::english(SimpleTokenizer);
        assert!(tokenizer.is_stop_word("the"));
        assert!(tokenizer.is_stop_word("and"));
        assert!(tokenizer.is_stop_word("is"));
        // These are definitely not stop words
        assert!(!tokenizer.is_stop_word("elephant"));
        assert!(!tokenizer.is_stop_word("quantum"));
    }

    #[test]
    fn test_tokenizer_registry_stop_word_tokenizers() {
        let registry = TokenizerRegistry::new();

        // Check stop word tokenizers are registered
        assert!(registry.contains("en_stop"));
        assert!(registry.contains("en_stem_stop"));
        assert!(registry.contains("de_stop"));
        assert!(registry.contains("ru_stop"));

        // Test en_stop filters stop words
        let tokenizer = registry.get("en_stop").unwrap();
        let tokens = tokenizer.tokenize("The quick fox");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(!texts.contains(&"the"));
        assert!(texts.contains(&"quick"));
        assert!(texts.contains(&"fox"));

        // Test en_stem_stop filters stop words AND stems
        let tokenizer = registry.get("en_stem_stop").unwrap();
        let tokens = tokenizer.tokenize("elephants galaxies");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"eleph")); // stemmed
        assert!(texts.contains(&"galaxi")); // stemmed
    }

    fn hinted<T: Tokenizer>(tokenizer: &T, text: &str, hint: Option<&str>) -> Vec<Token> {
        Tokenizer::tokenize_hinted(tokenizer, text, hint)
    }

    fn texts(tokens: &[Token]) -> Vec<&str> {
        tokens.iter().map(|t| t.text.as_str()).collect()
    }

    fn positions(tokens: &[Token]) -> Vec<u32> {
        tokens.iter().map(|t| t.position).collect()
    }

    #[test]
    fn unicode_segmenter_splits_on_word_boundaries_and_folds() {
        let plain = DynamicStemmer::new(None).with_segmenter(Segmenter::Unicode);
        let tokens = hinted(
            &plain,
            "Float-zero determinants: p53/CO2, 10.1007/s1 résumé",
            None,
        );
        assert_eq!(
            texts(&tokens),
            vec![
                "float",
                "zero",
                "determinants",
                "p53",
                "co2",
                "101007",
                "s1",
                "resume"
            ]
        );
        assert_eq!(positions(&tokens), (0..8).collect::<Vec<u32>>());
        // Offsets still point at the original words.
        assert_eq!(
            &"Float-zero determinants: p53/CO2, 10.1007/s1 résumé"
                [tokens[7].offset_from..tokens[7].offset_to],
            "résumé"
        );

        // Stemming sees the original letters, folding runs afterwards; the
        // simple segmenter keeps the historic behaviour.
        let english = DynamicStemmer::new(None).with_segmenter(Segmenter::Unicode);
        assert_eq!(
            texts(&hinted(&english, "Running foxes' café", Some("en"))),
            vec!["run", "fox", "cafe"]
        );
        assert_eq!(
            texts(&hinted(
                &DynamicStemmer::new(None),
                "Float-zero café",
                Some("en")
            )),
            vec!["floatzero", "café"]
        );
        // Cyrillic folding drops combining marks but keeps the letters.
        assert_eq!(texts(&hinted(&plain, "ёлка", None)), vec!["елка"]);
        // Stop words are removed before folding and keep their gap.
        let stopping = DynamicStemmer::new(None)
            .with_stop_words(true)
            .with_segmenter(Segmenter::Unicode);
        let tokens = hinted(&stopping, "state-of-the-art résumé", Some("en"));
        assert_eq!(tokens.len(), 3, "{:?}", texts(&tokens));
        assert_eq!(&texts(&tokens)[..2], ["state", "art"]);
        assert!(tokens[2].text.starts_with("resum"), "{:?}", tokens[2].text);
        assert!(tokens[2].text.is_ascii(), "folded after stemming");
        assert_eq!(positions(&tokens), vec![0, 3, 4]);
    }

    #[test]
    fn unicode_segmenter_bigrams_cjk_runs() {
        let t = DynamicStemmer::new(None).with_segmenter(Segmenter::Unicode);
        let tokens = hinted(&t, "東京都 tower", Some("en"));
        assert_eq!(texts(&tokens), vec!["東京", "京都", "tower"]);
        assert_eq!(positions(&tokens), vec![0, 1, 2]);
        assert_eq!(
            &"東京都 tower"[tokens[1].offset_from..tokens[1].offset_to],
            "京都"
        );
        // A lone ideograph is its own token; runs split at any non-CJK word.
        assert_eq!(
            texts(&hinted(&t, "東 tower 京都", None)),
            vec!["東", "tower", "京都"]
        );
        // Katakana runs are bigrammed too, and a run stops where the text
        // stops being contiguous.
        assert_eq!(
            texts(&hinted(&t, "トウキョウ タワー", None)),
            vec!["トウ", "ウキ", "キョ", "ョウ", "タワ", "ワー"]
        );
        // Mixed script without spaces: the ideographs form one run, the
        // Latin word follows.
        assert_eq!(texts(&hinted(&t, "東京tower", None)), vec!["東京", "tower"]);
        // Phrase offsets follow the bigram positions.
        let query = hinted(&t, "東京都", None);
        assert_eq!(positions(&query), vec![0, 1]);
    }

    #[test]
    fn dynamic_stemmer_drops_stop_words_but_keeps_positions() {
        let stemmer = DynamicStemmer::new(None).with_stop_words(true);
        let tokens = hinted(&stemmer, "Quantum of the Art", Some("en"));
        assert_eq!(texts(&tokens), vec!["quantum", "art"]);
        assert_eq!(positions(&tokens), vec![0, 3]);
        // Byte offsets still point at the surviving words.
        assert_eq!(tokens[1].offset_from, "Quantum of the ".len());

        // Stop words are matched before stemming, per script-routed language.
        let tokens = hinted(&stemmer, "бегущие и собаки the foxes", Some("ru,en"));
        assert_eq!(texts(&tokens), vec!["бегущ", "собак", "fox"]);
        assert_eq!(positions(&tokens), vec![0, 2, 4]);

        // Tokens of a script no hinted language covers are untouched.
        assert_eq!(
            texts(&hinted(&stemmer, "the 日本語 fox", Some("en"))),
            vec!["日本語", "fox"]
        );

        // No hint and no default: nothing is stemmed and nothing is dropped.
        assert_eq!(
            texts(&hinted(&stemmer, "the fox", None)),
            vec!["the", "fox"]
        );
        // A default language applies its stop list too.
        let english = DynamicStemmer::new(Some(Language::English)).with_stop_words(true);
        assert_eq!(texts(&hinted(&english, "the fox", None)), vec!["fox"]);
        // Off by default.
        assert_eq!(
            texts(&hinted(&DynamicStemmer::new(None), "the fox", Some("en"))),
            vec!["the", "fox"]
        );
        // Only stop words: no tokens at all.
        assert!(hinted(&stemmer, "to be or not to be", Some("en")).is_empty());
    }

    #[test]
    fn dynamic_stemmer_selects_language_from_hint() {
        let stemmer = DynamicStemmer::new(None);
        assert_eq!(
            texts(&hinted(&stemmer, "Running Foxes", Some("en"))),
            vec!["run", "fox"]
        );
        assert_eq!(
            texts(&hinted(&stemmer, "бегущие собаки", Some("ru"))),
            vec!["бегущ", "собак"]
        );
        // Unknown hint and no hint both fall back to the default (simple).
        assert_eq!(
            texts(&hinted(&stemmer, "Running Foxes", Some("xx"))),
            vec!["running", "foxes"]
        );
        assert_eq!(
            texts(&hinted(&stemmer, "Running Foxes", None)),
            vec!["running", "foxes"]
        );
        assert_eq!(
            texts(&Tokenizer::tokenize(&stemmer, "Running, Foxes!")),
            vec!["running", "foxes"]
        );
        // A default language applies when no hint is given.
        let english = DynamicStemmer::new(Some(Language::English));
        assert_eq!(
            texts(&hinted(&english, "Running Foxes", None)),
            vec!["run", "fox"]
        );
    }

    #[test]
    fn dynamic_stemmer_routes_tokens_by_script() {
        let stemmer = DynamicStemmer::new(None);
        // Mixed-script text: each token goes to the hinted language of its script.
        assert_eq!(
            texts(&hinted(&stemmer, "бегущие foxes", Some("ru,en"))),
            vec!["бегущ", "fox"]
        );
        assert_eq!(
            texts(&hinted(&stemmer, "бегущие foxes", Some("en, ru"))),
            vec!["бегущ", "fox"]
        );
        // A single hinted language never touches tokens of another script.
        assert_eq!(
            texts(&hinted(&stemmer, "бегущие foxes", Some("ru"))),
            vec!["бегущ", "foxes"]
        );
        assert_eq!(
            texts(&hinted(&stemmer, "бегущие foxes", Some("en"))),
            vec!["бегущие", "fox"]
        );
        // Same-script languages: the first listed one wins.
        assert_eq!(
            texts(&hinted(&stemmer, "running", Some("de,en"))),
            vec!["running"]
        );
        assert_eq!(
            texts(&hinted(&stemmer, "running", Some("en,de"))),
            vec!["run"]
        );
        // Positions stay sequential across scripts.
        let tokens = hinted(&stemmer, "бегущие foxes run", Some("ru,en"));
        assert_eq!(
            tokens.iter().map(|t| t.position).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn script_detection_covers_supported_stemmer_scripts() {
        assert_eq!(Script::of_token("hello"), Script::Latin);
        assert_eq!(Script::of_token("straße"), Script::Latin);
        assert_eq!(Script::of_token("собака"), Script::Cyrillic);
        assert_eq!(Script::of_token("γεια"), Script::Greek);
        assert_eq!(Script::of_token("مرحبا"), Script::Arabic);
        assert_eq!(Script::of_token("தமிழ்"), Script::Tamil);
        assert_eq!(Script::of_token("日本語"), Script::Other);
        assert_eq!(Script::of_token("2024"), Script::Other);
        assert_eq!(Language::Russian.script(), Script::Cyrillic);
        assert_eq!(Language::Turkish.script(), Script::Latin);
    }

    #[test]
    fn tokenizer_spec_parses_and_renders_canonically() {
        assert_eq!(
            TokenizerSpec::parse("en_stem").unwrap(),
            TokenizerSpec::Named("en_stem".to_string())
        );
        let spec = TokenizerSpec::parse("stem(by:languages,default:simple)").unwrap();
        assert_eq!(
            spec,
            TokenizerSpec::DynamicStem {
                by: Some("languages".to_string()),
                default: None,
                stop_words: false,
                segmenter: Segmenter::Simple,
            }
        );
        assert_eq!(spec.to_string(), "stem(by: languages, default: simple)");
        assert_eq!(spec.hint_field(), Some("languages"));

        let spec = TokenizerSpec::parse("stem(by: lang, default: english)").unwrap();
        assert_eq!(spec.to_string(), "stem(by: lang, default: en)");
        assert_eq!(
            TokenizerSpec::parse("stem(by: lang)").unwrap(),
            TokenizerSpec::DynamicStem {
                by: Some("lang".to_string()),
                default: None,
                stop_words: false,
                segmenter: Segmenter::Simple,
            }
        );

        let spec =
            TokenizerSpec::parse("stem(by: languages, default: simple, stop_words: true)").unwrap();
        assert_eq!(
            spec,
            TokenizerSpec::DynamicStem {
                by: Some("languages".to_string()),
                default: None,
                stop_words: true,
                segmenter: Segmenter::Simple,
            }
        );
        assert_eq!(
            spec.to_string(),
            "stem(by: languages, default: simple, stop_words: true)"
        );
        // `stop_words: false` is the default and renders as the historic form.
        assert_eq!(
            TokenizerSpec::parse("stem(by: lang, stop_words: false)")
                .unwrap()
                .to_string(),
            "stem(by: lang, default: simple)"
        );
        assert!(TokenizerSpec::parse("stem(by: lang, stop_words: maybe)").is_err());
        let spec =
            TokenizerSpec::parse("stem(by: languages, stop_words: true, segmenter: unicode)")
                .unwrap();
        assert_eq!(
            spec,
            TokenizerSpec::DynamicStem {
                by: Some("languages".to_string()),
                default: None,
                stop_words: true,
                segmenter: Segmenter::Unicode,
            }
        );
        assert_eq!(
            spec.to_string(),
            "stem(by: languages, default: simple, stop_words: true, segmenter: unicode)"
        );
        assert!(TokenizerSpec::parse("stem(by: lang, segmenter: icu)").is_err());

        // `by` is optional: a spec without it is a fixed tokenizer.
        assert_eq!(
            TokenizerSpec::parse("stem(default: en)")
                .unwrap()
                .hint_field(),
            None
        );
        assert!(TokenizerSpec::parse("stem(by: lang, default: klingon)").is_err());
        assert!(TokenizerSpec::parse("stem(by: lang").is_err());
        assert!(TokenizerSpec::parse("stem(by: lang, color: red)").is_err());
        assert!(TokenizerSpec::parse("en_stem(foo)").is_err());
        assert!(TokenizerSpec::parse("").is_err());
    }

    #[test]
    fn registry_builds_dynamic_stemmer_from_spec() {
        let registry = TokenizerRegistry::new();
        let tokenizer = registry
            .get("stem(by: languages, default: simple)")
            .expect("dynamic spec resolves without registration");
        assert_eq!(
            texts(&tokenizer.tokenize_hinted("Running Foxes", Some("en"))),
            vec!["run", "fox"]
        );
        assert_eq!(
            texts(&tokenizer.tokenize("Running Foxes")),
            vec!["running", "foxes"]
        );
        assert!(
            registry
                .get("stem(by: languages, default: klingon)")
                .is_none()
        );
        let stopping = registry
            .get("stem(by: languages, default: simple, stop_words: true)")
            .expect("stop-word spec resolves");
        let tokens = stopping.tokenize_hinted("the running foxes", Some("en"));
        assert_eq!(texts(&tokens), vec!["run", "fox"]);
        assert_eq!(positions(&tokens), vec![1, 2]);
        // Static tokenizers accept and ignore hints.
        let simple = registry.get("en_stem").unwrap();
        assert_eq!(
            texts(&simple.tokenize_hinted("Running Foxes", Some("ru"))),
            vec!["run", "fox"]
        );
    }

    #[test]
    fn spec_without_by_is_a_fixed_tokenizer_that_ignores_hints() {
        let spec =
            TokenizerSpec::parse("stem(default: simple, stop_words: true, segmenter: unicode)")
                .unwrap();
        assert_eq!(spec.hint_field(), None);
        assert_eq!(
            spec.to_string(),
            "stem(default: simple, stop_words: true, segmenter: unicode)"
        );
        assert_eq!(TokenizerSpec::parse(&spec.to_string()).unwrap(), spec);
        let tokenizer = spec.dynamic_tokenizer().unwrap();
        let plain: Vec<String> = tokenizer
            .tokenize("The running cells")
            .into_iter()
            .map(|t| t.text)
            .collect();
        // No language: nothing is stemmed and no stop list applies.
        assert_eq!(plain, vec!["the", "running", "cells"]);
        let hinted: Vec<String> = tokenizer
            .tokenize_hinted("The running cells", Some("en"))
            .into_iter()
            .map(|t| t.text)
            .collect();
        assert_eq!(hinted, plain, "hints are ignored without `by`");

        // A fixed default language stems every document and query alike.
        let english = TokenizerSpec::parse("stem(default: en, stop_words: true)").unwrap();
        let tokenizer = english.dynamic_tokenizer().unwrap();
        let stemmed: Vec<String> = tokenizer
            .tokenize_hinted("The running cells", Some("ru"))
            .into_iter()
            .map(|t| t.text)
            .collect();
        assert_eq!(stemmed, vec!["run", "cell"]);
    }

    #[test]
    fn parse_language_opt_rejects_unknown_codes() {
        assert_eq!(parse_language_opt(" RU "), Some(Language::Russian));
        assert_eq!(parse_language_opt("german"), Some(Language::German));
        assert_eq!(parse_language_opt("xx"), None);
        assert_eq!(parse_language_opt(""), None);
        for language in [Language::English, Language::Russian, Language::Tamil] {
            assert_eq!(parse_language_opt(language_code(language)), Some(language));
        }
    }
}
