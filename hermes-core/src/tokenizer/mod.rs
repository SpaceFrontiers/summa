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
}

/// Simple whitespace tokenizer
#[derive(Debug, Clone, Default)]
pub struct SimpleTokenizer;

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut position = 0u32;

        for (offset, word) in split_whitespace_with_offsets(text) {
            if !word.is_empty() {
                tokens.push(Token::new(
                    word.to_string(),
                    position,
                    offset,
                    offset + word.len(),
                ));
                position += 1;
            }
        }

        tokens
    }
}

/// Lowercase tokenizer - splits on whitespace and lowercases
#[derive(Debug, Clone, Default)]
pub struct LowercaseTokenizer;

impl Tokenizer for LowercaseTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        tokenize_and_clean(text, |s| s.to_string())
    }
}

/// Strip non-alphanumeric characters and lowercase.
///
/// ASCII fast-path iterates bytes directly; falls back to full Unicode
/// `char` iteration only when the word contains non-ASCII bytes.
#[inline]
fn clean_word(word: &str) -> String {
    if word.is_ascii() {
        // ASCII fast path – byte iteration, no char decoding
        let mut result = String::with_capacity(word.len());
        for &b in word.as_bytes() {
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
fn tokenize_and_clean(text: &str, transform: impl Fn(&str) -> String) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut position = 0u32;
    for (offset, word) in split_whitespace_with_offsets(text) {
        if !word.is_empty() {
            let cleaned = clean_word(word);
            if !cleaned.is_empty() {
                tokens.push(Token::new(
                    transform(&cleaned),
                    position,
                    offset,
                    offset + word.len(),
                ));
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
            Language::Tamil => LANGUAGE::English, // Tamil not supported, fallback to English
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
            .into_iter()
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
            .filter(|token| !self.stop_words.contains(&token.text))
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
        tokenize_and_clean(text, |s| stemmer.stem(s).into_owned())
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
        tokenize_and_clean(text, |s| stemmer.stem(s).into_owned())
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

/// Boxed tokenizer for dynamic dispatch
pub type BoxedTokenizer = Box<dyn TokenizerClone>;

pub trait TokenizerClone: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<Token>;
    fn clone_box(&self) -> BoxedTokenizer;
}

impl<T: Tokenizer> TokenizerClone for T {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        Tokenizer::tokenize(self, text)
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
/// Pre-registers common tokenizers: "default", "simple", "lowercase", "en_stem", etc.
#[derive(Clone)]
pub struct TokenizerRegistry {
    tokenizers: Arc<RwLock<HashMap<String, BoxedTokenizer>>>,
}

impl TokenizerRegistry {
    /// Create a new tokenizer registry with default tokenizers registered
    pub fn new() -> Self {
        let registry = Self {
            tokenizers: Arc::new(RwLock::new(HashMap::new())),
        };
        registry.register_defaults();
        registry
    }

    /// Register default tokenizers
    fn register_defaults(&self) {
        // Basic tokenizers
        self.register("default", LowercaseTokenizer);
        self.register("simple", SimpleTokenizer);
        self.register("lowercase", LowercaseTokenizer);
        self.register("raw", SimpleTokenizer);

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
            StopWordTokenizer::new(LowercaseTokenizer, Language::English),
        );
        self.register(
            "de_stop",
            StopWordTokenizer::new(LowercaseTokenizer, Language::German),
        );
        self.register(
            "fr_stop",
            StopWordTokenizer::new(LowercaseTokenizer, Language::French),
        );
        self.register(
            "ru_stop",
            StopWordTokenizer::new(LowercaseTokenizer, Language::Russian),
        );
        self.register(
            "es_stop",
            StopWordTokenizer::new(LowercaseTokenizer, Language::Spanish),
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

    /// Get a tokenizer by name
    pub fn get(&self, name: &str) -> Option<BoxedTokenizer> {
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
        let tokens = Tokenizer::tokenize(&tokenizer, "hello world");

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[1].position, 1);
    }

    #[test]
    fn test_lowercase_tokenizer() {
        let tokenizer = LowercaseTokenizer;
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
        assert!(registry.contains("default"));
        assert!(registry.contains("simple"));
        assert!(registry.contains("lowercase"));
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
        registry.register("my_tokenizer", LowercaseTokenizer);

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
        let tokenizer = StopWordTokenizer::english(LowercaseTokenizer);
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
        let tokenizer = StopWordTokenizer::new(LowercaseTokenizer, Language::German);
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
        let tokenizer = StopWordTokenizer::with_custom_stop_words(LowercaseTokenizer, custom_stops);
        let tokens = Tokenizer::tokenize(&tokenizer, "foo baz bar qux");

        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(!texts.contains(&"foo"));
        assert!(!texts.contains(&"bar"));
        assert!(texts.contains(&"baz"));
        assert!(texts.contains(&"qux"));
    }

    #[test]
    fn test_stop_word_tokenizer_is_stop_word() {
        let tokenizer = StopWordTokenizer::english(LowercaseTokenizer);
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
}
