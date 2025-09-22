//! The lexical text tokenizer: segmentation, normalisation, morphology and
//! same-position variants, declared in SDL as
//!
//! ```text
//! text<lex(by: <field>, default: <language|none>, stop_words: <bool>,
//!          segmenter: <icu|unicode|simple>, stem: <light|snowball|none>,
//!          variants: <bool>, fold: <bool>, max_token_length: <n>,
//!          han: <as_written|simplified>, cjk: <icu|dictionary>)>
//! ```
//!
//! Every option has a default (see [`LexOptions`]) and only non-defaults are
//! rendered, so `lex()` alone is the recommended tokenizer for a
//! language-agnostic text field and
//! `lex(by: languages, default: en, stop_words: true)` the recommended one
//! for a multilingual corpus that tags its documents.
//!
//! Three layers, identical at index and query time:
//!
//! 1. **Segmentation and normalisation** (language-agnostic). `icu` uses
//!    ICU4X's word segmenter (dictionary words for Chinese and Japanese, LSTM
//!    word breaks for Thai, Lao, Khmer and Burmese, UAX #29 elsewhere) and
//!    emits the bigrams of every dictionary word as same-position variants,
//!    so a differently segmented query still matches; runs the dictionary
//!    does not know fall back to character bigrams. `unicode` is UAX #29 with
//!    bigrams over every CJK run; `simple` splits on whitespace. Every token
//!    is NFKC-normalised and lowercased; Arabic tokens get the Lucene
//!    orthographic normalisation, Cyrillic `ё` becomes `е`; `han:
//!    simplified` folds traditional Chinese characters. Tokens longer than
//!    `max_token_length` characters are dropped (position kept). `cjk:
//!    dictionary` adds Japanese and Korean morphology (see
//!    `cjk_morph`).
//! 2. **Morphology** per language, routed by the token's script to the first
//!    hinted language of that script (`by` reads the hint from a sibling
//!    field at index time, the query passes `tokenizer_hint`; without `by`
//!    the `default` applies to everything and hints are ignored): `light`
//!    strips inflection only (see [`super::light_stem`]), `snowball` is the
//!    full algorithm, `none` keeps the word.
//! 3. **Variants** (`variants: true`): the written word is the indexed
//!    token; its stem and its diacritic-folded form (`fold`) are variants at
//!    the same position, so phrases and exact terms match the written form
//!    while match queries use the stem. Variants are marked
//!    [`super::Token::variant`] and do not count towards field length.
//!    Without variants the (folded) stem replaces the word.
//!
//! Query tokenization ([`super::Purpose::Match`], [`super::Purpose::Exact`])
//! emits one form per word and never a variant: the stem for a match query
//! (the written form when the language is unknown), the written form for a
//! phrase or an exact term. Because a stem shares its word's position, a
//! phrase term also matches words that stem to it.

use std::collections::{HashMap, HashSet};

use parking_lot::RwLock;

use super::{
    Language, Purpose, Script, Token, Tokenizer, cjk_morph, language_code, light_stem,
    parse_language_opt, split_whitespace_with_offsets, with_stemmers,
};

/// Default `max_token_length`: longer "words" are hashes, sequences and
/// URLs, which no query types and which bloat the dictionary.
pub const DEFAULT_MAX_TOKEN_LENGTH: usize = 64;

/// Segmenters see the text in windows of at most this many characters.
///
/// ICU's dictionary segmentation of Han text is quadratic in the length of
/// the run it is handed: one 2.4 MB Chinese book took 105 s as a single
/// call and 0.2 s in windows. A window ends at the first whitespace or
/// punctuation after `SEGMENT_WINDOW_SOFT` characters, or unconditionally at
/// `SEGMENT_WINDOW_HARD` (text with neither in thousands of characters is not
/// prose; a split there costs at most one word boundary).
const SEGMENT_WINDOW_SOFT: usize = 1024;
const SEGMENT_WINDOW_HARD: usize = 4096;

/// Windows of `text` as (byte offset, slice), contiguous and covering.
fn segment_windows(text: &str) -> Vec<(usize, &str)> {
    let mut windows = Vec::new();
    let mut start = 0usize;
    let mut chars_in_window = 0usize;
    for (offset, c) in text.char_indices() {
        chars_in_window += 1;
        let cut = chars_in_window >= SEGMENT_WINDOW_HARD
            || (chars_in_window >= SEGMENT_WINDOW_SOFT && (c.is_whitespace() || is_break_punct(c)));
        if cut {
            let end = offset + c.len_utf8();
            windows.push((start, &text[start..end]));
            start = end;
            chars_in_window = 0;
        }
    }
    if start < text.len() || windows.is_empty() {
        windows.push((start, &text[start..]));
    }
    windows
}

/// Split one segment into lowercase alphanumeric words at the punctuation
/// the segmenter left inside it, the way a standard analyzer with the
/// possessive and elision filters does:
///
/// - any character that is not a letter, digit or mark is a boundary
///   (`state-of-the-art` → `state of the art`, `HbA1c/HDL-c` → `hba1c hdl c`,
///   `end.of.sentence.Next` → `end of sentence next`);
/// - an apostrophe between letters splits the word into parts: a trailing
///   English contraction or possessive (`'s`, `'t`, `'re`, `'ve`, `'ll`,
///   `'d`, `'m`) is dropped (`John's` → `john`, `don't` → `don`), leading
///   elided articles and conjunctions (`l'`, `d'`, `qu'`, `dell'`, …) are
///   dropped (`l'homme` → `homme`, `qu'il` → `il`, `O'Neil` → `neil`), and
///   what remains are separate words (`aujourd'hui` → `aujourd hui`);
/// - a dotted acronym is joined (`U.S.A.` → `usa`, `e.g.` → `eg`);
/// - a dot between digits is kept (`3.14`, `0.05`, `1.2.3`) and a comma
///   between digits is dropped (`1,000` → `1000`);
/// - soft hyphens, zero-width joiners and other invisible joiners are
///   dropped without splitting (`soft\u{ad}hyphen` → `softhyphen`).
///
/// Returns `(from, to, piece)` byte spans into `segment`.
fn split_word(segment: &str) -> Vec<(usize, usize, String)> {
    // Fast path: already a clean lowercase word.
    if segment
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    {
        return vec![(0, segment.len(), segment.to_string())];
    }
    let chars: Vec<(usize, char)> = segment.char_indices().collect();
    let acronym = segment.contains('.') && is_dotted_acronym(segment);
    let mut pieces: Vec<(usize, usize, String)> = Vec::new();
    // Parts of the word joined by apostrophes so far (`rock'n'roll`).
    let mut group: Vec<(usize, usize, String)> = Vec::new();
    let mut piece = String::new();
    let mut piece_from = 0usize;
    let mut piece_to = 0usize;
    for (i, &(offset, c)) in chars.iter().enumerate() {
        let prev = (i > 0).then(|| chars[i - 1].1);
        let next = chars.get(i + 1).map(|(_, n)| *n);
        let between = |test: fn(char) -> bool| prev.is_some_and(test) && next.is_some_and(test);
        let action = if c.is_alphanumeric() || is_mark(c) {
            Piece::Keep
        } else if is_joiner(c) {
            Piece::Join
        } else if is_apostrophe(c) && between(char::is_alphabetic) {
            Piece::Apostrophe
        } else if c == '.' && acronym {
            Piece::Join
        } else if c == '.' && between(|d| d.is_ascii_digit()) {
            Piece::Keep
        } else if c == ',' && between(|d| d.is_ascii_digit()) {
            // Thousands separator: `1,000` → `1000`, the form people type.
            Piece::Join
        } else {
            Piece::Boundary
        };
        match action {
            Piece::Keep => {
                if piece.is_empty() {
                    piece_from = offset;
                }
                piece.extend(c.to_lowercase());
                piece_to = offset + c.len_utf8();
            }
            Piece::Join => {}
            Piece::Apostrophe => {
                group.push((piece_from, piece_to, std::mem::take(&mut piece)));
            }
            Piece::Boundary => {
                if !piece.is_empty() {
                    group.push((piece_from, piece_to, std::mem::take(&mut piece)));
                }
                resolve_apostrophes(&mut group, &mut pieces);
            }
        }
    }
    if !piece.is_empty() {
        group.push((piece_from, piece_to, piece));
    }
    resolve_apostrophes(&mut group, &mut pieces);
    pieces
}

enum Piece {
    Keep,
    Join,
    Apostrophe,
    Boundary,
}

/// Emit the parts of an apostrophe-joined word: drop a trailing English
/// contraction/possessive suffix and leading elided prefixes.
fn resolve_apostrophes(
    group: &mut Vec<(usize, usize, String)>,
    pieces: &mut Vec<(usize, usize, String)>,
) {
    if group.len() > 1 {
        if group
            .last()
            .is_some_and(|(_, _, part)| CONTRACTION_SUFFIXES.contains(&part.as_str()))
        {
            group.pop();
        }
        while group.len() > 1
            && group.first().is_some_and(|(_, _, part)| {
                part.chars().count() <= 2 || ELISION_PREFIXES.contains(&part.as_str())
            })
        {
            group.remove(0);
        }
    }
    pieces.append(group);
}

/// English contractions and the possessive: dropped after an apostrophe.
const CONTRACTION_SUFFIXES: &[&str] = &["s", "t", "re", "ve", "ll", "d", "m"];

/// Elided articles, prepositions and conjunctions of French, Italian and
/// Catalan longer than two letters (shorter ones are dropped by length).
const ELISION_PREFIXES: &[&str] = &[
    "qu", "lorsqu", "jusqu", "puisqu", "quoiqu", "dell", "nell", "sull", "dall", "all", "coll",
    "degl", "dagl", "negl", "sugl", "quest", "quell", "sant", "anch",
];

/// `U.S.A.`, `e.g.`: every run between dots is exactly one letter
/// (`Ph.D.` splits into `ph d` like a standard analyzer).
fn is_dotted_acronym(segment: &str) -> bool {
    let mut runs = 0;
    for run in segment.split('.') {
        if run.is_empty() {
            continue;
        }
        let mut letters = run.chars();
        match (letters.next(), letters.next()) {
            (Some(c), None) if c.is_alphabetic() => runs += 1,
            _ => return false,
        }
    }
    runs >= 2
}

fn is_mark(c: char) -> bool {
    // Combining marks (Mn/Mc/Me) that survive NFKC, e.g. in Indic scripts.
    matches!(c as u32, 0x0300..=0x036F | 0x0483..=0x0489 | 0x0591..=0x05BD | 0x0610..=0x061A
        | 0x064B..=0x065F | 0x0900..=0x0903 | 0x093A..=0x094F | 0x0951..=0x0957 | 0x0962..=0x0963
        | 0x0E31 | 0x0E34..=0x0E3A | 0x0E47..=0x0E4E | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF
        | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}

/// Invisible characters that join the letters around them.
fn is_joiner(c: char) -> bool {
    matches!(
        c,
        '\u{00AD}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}' | '\u{034F}'
    )
}

fn is_apostrophe(c: char) -> bool {
    matches!(c, '\'' | '\u{2019}' | '\u{2018}' | '\u{02BC}' | '\u{FF07}')
}

/// Punctuation that ends a segmentation window: ASCII sentence and clause
/// marks plus their CJK full-width forms.
fn is_break_punct(c: char) -> bool {
    matches!(
        c,
        '.' | ','
            | ';'
            | ':'
            | '!'
            | '?'
            | ')'
            | ']'
            | '}'
            | '。'
            | '，'
            | '、'
            | '；'
            | '：'
            | '！'
            | '？'
            | '」'
            | '』'
            | '）'
            | '】'
            | '〉'
            | '》'
    )
}

/// Word segmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Segmenter {
    /// ICU4X word segmentation: dictionary words for Chinese and Japanese
    /// (with their bigrams as variants), LSTM word breaks for Thai, Lao,
    /// Khmer and Burmese, UAX #29 for everything else.
    #[default]
    Icu,
    /// UAX #29 word boundaries (`float-zero` → `float`, `zero`; `p53`,
    /// `co2` and `10.1007` stay whole) and character bigrams over runs of
    /// Han, Hiragana and Katakana.
    Unicode,
    /// Split on whitespace, strip every non-alphanumeric character
    /// (`float-zero` → `floatzero`, no CJK segmentation).
    Simple,
}

/// Morphological normalisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StemMode {
    /// Inflection only (plural, case, gender), Lucene's light stemmers.
    #[default]
    Light,
    /// Full Snowball stemming.
    Snowball,
    /// Keep every word as written.
    None,
}

/// Treatment of Han characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HanForm {
    /// Index and query characters as written.
    #[default]
    AsWritten,
    /// Fold traditional characters to simplified (OpenCC table), index and
    /// query alike; Japanese dictionary tokens keep their surface and gain
    /// the folded form as a variant.
    Simplified,
}

/// Japanese and Korean analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CjkMode {
    /// ICU dictionary words (Chinese and Japanese) plus bigrams; Korean by
    /// UAX #29 (particles stay attached).
    #[default]
    Icu,
    /// Dictionary morphology (`cjk-dict` feature): Korean always, Japanese
    /// for kana runs and, when hinted `ja`, Han runs; particles and endings
    /// dropped, base forms as variants.
    Dictionary,
}

/// Options of a [`LexTokenizer`]; the parsed and rendered form of a
/// `lex(...)` spec. Only values that differ from the defaults are rendered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexOptions {
    /// Field whose text values supply the language hint; `None` = no
    /// per-document language, hints are ignored, `default` always applies.
    pub by: Option<String>,
    /// Language applied when no hint is present; `None` = no stemming or
    /// stop list.
    pub default: Option<Language>,
    /// Drop the routed language's stop words (positions keep their gaps).
    pub stop_words: bool,
    pub segmenter: Segmenter,
    pub stem: StemMode,
    /// Keep the written word as the token and index stems and folded forms
    /// as same-position variants.
    pub variants: bool,
    /// Fold diacritics of Latin, Cyrillic and Greek tokens.
    pub fold: bool,
    /// Drop tokens longer than this many characters (0 = unlimited).
    pub max_token_length: usize,
    pub han: HanForm,
    pub cjk: CjkMode,
}

impl Default for LexOptions {
    fn default() -> Self {
        Self {
            by: None,
            default: None,
            stop_words: false,
            segmenter: Segmenter::Icu,
            stem: StemMode::Light,
            variants: true,
            fold: true,
            max_token_length: DEFAULT_MAX_TOKEN_LENGTH,
            han: HanForm::AsWritten,
            cjk: CjkMode::Icu,
        }
    }
}

impl LexOptions {
    /// Parse the `key: value, ...` body of a `lex(...)` spec.
    pub fn parse(params: &str) -> Result<Self, String> {
        let mut options = Self::default();
        let spec = format!("lex({params})");
        let parse_bool = |key: &str, value: &str| -> Result<bool, String> {
            match value {
                "true" => Ok(true),
                "false" => Ok(false),
                other => Err(format!(
                    "tokenizer spec '{spec}': '{key}' must be true or false, got '{other}'"
                )),
            }
        };
        let choice = |key: &str, value: &str, allowed: &[&str]| -> Result<(), String> {
            if allowed.contains(&value) {
                Ok(())
            } else {
                Err(format!(
                    "tokenizer spec '{spec}': '{key}' must be one of {}, got '{value}'",
                    allowed.join(", ")
                ))
            }
        };
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
                "by" if !value.is_empty() => options.by = Some(value.to_string()),
                "by" => return Err(format!("tokenizer spec '{spec}': 'by' needs a field name")),
                "default" => {
                    options.default = match value {
                        "none" => None,
                        other => Some(parse_language_opt(other).ok_or_else(|| {
                            format!("tokenizer spec '{spec}': unknown default language '{other}'")
                        })?),
                    };
                }
                "stop_words" => options.stop_words = parse_bool(key, value)?,
                "variants" => options.variants = parse_bool(key, value)?,
                "fold" => options.fold = parse_bool(key, value)?,
                "segmenter" => {
                    choice(key, value, &["icu", "unicode", "simple"])?;
                    options.segmenter = match value {
                        "icu" => Segmenter::Icu,
                        "unicode" => Segmenter::Unicode,
                        _ => Segmenter::Simple,
                    };
                }
                "stem" => {
                    choice(key, value, &["light", "snowball", "none"])?;
                    options.stem = match value {
                        "light" => StemMode::Light,
                        "snowball" => StemMode::Snowball,
                        _ => StemMode::None,
                    };
                }
                "han" => {
                    choice(key, value, &["as_written", "simplified"])?;
                    options.han = if value == "simplified" {
                        HanForm::Simplified
                    } else {
                        HanForm::AsWritten
                    };
                }
                "cjk" => {
                    choice(key, value, &["icu", "dictionary"])?;
                    options.cjk = if value == "dictionary" {
                        if !cjk_morph::available() {
                            return Err(format!(
                                "tokenizer spec '{spec}': 'cjk: dictionary' needs a build with the cjk-dict feature (Japanese and Korean dictionaries)"
                            ));
                        }
                        CjkMode::Dictionary
                    } else {
                        CjkMode::Icu
                    };
                }
                "max_token_length" => {
                    options.max_token_length = value.parse::<usize>().map_err(|_| {
                        format!(
                            "tokenizer spec '{spec}': 'max_token_length' must be a number, got '{value}'"
                        )
                    })?;
                }
                other => {
                    return Err(format!(
                        "tokenizer spec '{spec}': unknown parameter '{other}'"
                    ));
                }
            }
        }
        Ok(options)
    }

    /// Render the `key: value, ...` body: non-default options in canonical
    /// order.
    fn render(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let defaults = Self::default();
        let mut parts: Vec<String> = Vec::new();
        if let Some(by) = &self.by {
            parts.push(format!("by: {by}"));
        }
        if let Some(language) = self.default {
            parts.push(format!("default: {}", language_code(language)));
        }
        if self.stop_words != defaults.stop_words {
            parts.push(format!("stop_words: {}", self.stop_words));
        }
        if self.segmenter != defaults.segmenter {
            parts.push(format!(
                "segmenter: {}",
                match self.segmenter {
                    Segmenter::Icu => "icu",
                    Segmenter::Unicode => "unicode",
                    Segmenter::Simple => "simple",
                }
            ));
        }
        if self.stem != defaults.stem {
            parts.push(format!(
                "stem: {}",
                match self.stem {
                    StemMode::Light => "light",
                    StemMode::Snowball => "snowball",
                    StemMode::None => "none",
                }
            ));
        }
        if self.variants != defaults.variants {
            parts.push(format!("variants: {}", self.variants));
        }
        if self.fold != defaults.fold {
            parts.push(format!("fold: {}", self.fold));
        }
        if self.max_token_length != defaults.max_token_length {
            parts.push(format!("max_token_length: {}", self.max_token_length));
        }
        if self.han != defaults.han {
            parts.push("han: simplified".to_string());
        }
        if self.cjk != defaults.cjk {
            parts.push("cjk: dictionary".to_string());
        }
        write!(f, "lex({})", parts.join(", "))
    }
}

impl std::fmt::Display for LexOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(f)
    }
}

/// Parsed form of a tokenizer name in the schema: a registered name
/// (`simple`, `en_stem`, ...) or a `lex(...)` spec. The canonical string
/// form is stored in `FieldEntry::tokenizer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerSpec {
    Named(String),
    Lex(LexOptions),
}

impl TokenizerSpec {
    /// Parse a tokenizer name or `lex(...)` spec.
    pub fn parse(spec: &str) -> Result<TokenizerSpec, String> {
        let spec = spec.trim();
        let Some(rest) = spec.strip_prefix("lex(") else {
            if spec.is_empty() || spec.contains(['(', ')', ':', ',']) {
                return Err(format!("invalid tokenizer spec '{spec}'"));
            }
            return Ok(TokenizerSpec::Named(spec.to_string()));
        };
        let Some(params) = rest.strip_suffix(')') else {
            return Err(format!("tokenizer spec '{spec}' is missing ')'"));
        };
        LexOptions::parse(params).map(TokenizerSpec::Lex)
    }

    /// The options of a `lex(...)` spec.
    pub fn lex(&self) -> Option<&LexOptions> {
        match self {
            TokenizerSpec::Named(_) => None,
            TokenizerSpec::Lex(options) => Some(options),
        }
    }

    /// Field whose values hint the tokenizer, for `lex` specs with `by`.
    pub fn hint_field(&self) -> Option<&str> {
        self.lex().and_then(|options| options.by.as_deref())
    }

    /// Whether the spec indexes originals next to their variants, so exact
    /// (phrase, term) queries match the written form and match queries the
    /// stem.
    pub fn keeps_original(&self) -> bool {
        self.lex().is_some_and(|options| options.variants)
    }

    /// Build the tokenizer described by a `lex` spec.
    pub fn dynamic_tokenizer(&self) -> Option<super::BoxedTokenizer> {
        self.lex()
            .map(|options| Box::new(LexTokenizer::new(options.clone())) as super::BoxedTokenizer)
    }
}

impl std::fmt::Display for TokenizerSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerSpec::Named(name) => f.write_str(name),
            TokenizerSpec::Lex(options) => options.render(f),
        }
    }
}

/// The tokenizer of a `lex(...)` field (module docs).
#[derive(Debug, Clone, Default)]
pub struct LexTokenizer {
    options: LexOptions,
}

impl LexTokenizer {
    pub fn new(options: LexOptions) -> Self {
        Self { options }
    }

    pub fn options(&self) -> &LexOptions {
        &self.options
    }

    /// Languages of a hint (`"ru,en"`), in order, plus the Japanese and
    /// Korean flags; the default language when the hint names none. Specs
    /// without `by` ignore hints.
    fn hints(&self, hint: Option<&str>) -> Hints {
        let mut hints = Hints::default();
        if self.options.by.is_some()
            && let Some(hint) = hint.map(str::trim).filter(|hint| !hint.is_empty())
        {
            for part in hint.split(',') {
                let part = part.trim();
                match part.to_ascii_lowercase().as_str() {
                    "ja" | "jpn" | "japanese" => hints.japanese = true,
                    "ko" | "kor" | "korean" => hints.korean = true,
                    _ => {
                        if let Some(language) = parse_language_opt(part)
                            && !hints.languages.contains(&language)
                        {
                            hints.languages.push(language);
                        }
                    }
                }
            }
        }
        if hints.languages.is_empty() {
            hints.languages.extend(self.options.default);
        }
        hints
    }

    fn run(&self, text: &str, hints: &Hints, purpose: Purpose) -> Vec<Token> {
        let stops: Vec<Option<&'static HashSet<String>>> = hints
            .languages
            .iter()
            .map(|language| {
                self.options
                    .stop_words
                    .then(|| stop_word_set(*language))
                    .flatten()
            })
            .collect();
        if hints.languages.is_empty() || self.options.stem != StemMode::Snowball {
            self.walk(text, &Ctx::new(hints, &stops, &[]), purpose)
        } else {
            with_stemmers(&hints.languages, |stemmers| {
                self.walk(text, &Ctx::new(hints, &stops, stemmers), purpose)
            })
        }
    }

    fn walk(&self, text: &str, ctx: &Ctx<'_>, purpose: Purpose) -> Vec<Token> {
        let mut emitter = Emitter {
            options: &self.options,
            ctx,
            purpose,
            tokens: Vec::with_capacity(text.len() / 5),
            position: 0,
            run: Vec::new(),
            run_end: 0,
        };
        match self.options.segmenter {
            Segmenter::Simple => {
                for (offset, word) in split_whitespace_with_offsets(text) {
                    emitter.word(offset, word);
                }
            }
            Segmenter::Unicode => {
                use unicode_segmentation::UnicodeSegmentation;
                for (offset, word) in text.unicode_word_indices() {
                    if word.chars().all(|c| CjkScript::of(c).is_cjk()) {
                        emitter.cjk_chars(offset, word);
                    } else {
                        emitter.word(offset, word);
                    }
                }
            }
            Segmenter::Icu if self.options.cjk == CjkMode::Dictionary => {
                for (start, end, kind) in morph_spans(text, ctx.hints) {
                    for (offset, window) in segment_windows(&text[start..end]) {
                        let base = start + offset;
                        match kind {
                            SpanKind::Japanese => {
                                emitter.morph_run(base, window, cjk_morph::japanese)
                            }
                            SpanKind::Korean => emitter.morph_run(base, window, cjk_morph::korean),
                            SpanKind::Icu => emitter.icu_span(base, window),
                        }
                    }
                }
            }
            Segmenter::Icu => {
                for (offset, window) in segment_windows(text) {
                    emitter.icu_span(offset, window);
                }
            }
        }
        emitter.flush_run();
        emitter.tokens
    }
}

impl Tokenizer for LexTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        self.run(text, &self.hints(None), Purpose::Index)
    }

    fn tokenize_with(&self, text: &str, hint: Option<&str>, purpose: Purpose) -> Vec<Token> {
        self.run(text, &self.hints(hint), purpose)
    }
}

/// Languages of one tokenization.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Hints {
    languages: Vec<Language>,
    japanese: bool,
    korean: bool,
}

/// Per-call context.
struct Ctx<'a> {
    hints: &'a Hints,
    stops: &'a [Option<&'static HashSet<String>>],
    /// Snowball stemmers aligned with `hints.languages` (empty unless the
    /// mode is Snowball).
    stemmers: &'a [&'a rust_stemmers::Stemmer],
}

impl<'a> Ctx<'a> {
    fn new(
        hints: &'a Hints,
        stops: &'a [Option<&'static HashSet<String>>],
        stemmers: &'a [&'a rust_stemmers::Stemmer],
    ) -> Self {
        Self {
            hints,
            stops,
            stemmers,
        }
    }
}

/// Script class of a character for CJK handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CjkScript {
    Han,
    Kana,
    Hangul,
    Other,
}

impl CjkScript {
    #[inline]
    fn of(c: char) -> Self {
        match c as u32 {
            0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F => Self::Han,
            0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => Self::Kana,
            0xAC00..=0xD7AF
            | 0x1100..=0x11FF
            | 0x3130..=0x318F
            | 0xA960..=0xA97F
            | 0xD7B0..=0xD7FF => Self::Hangul,
            _ => Self::Other,
        }
    }

    /// Han or kana: the scripts that are bigrammed instead of stemmed.
    #[inline]
    fn is_cjk(self) -> bool {
        matches!(self, Self::Han | Self::Kana)
    }
}

/// How a span of text is analysed under `cjk: dictionary`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpanKind {
    Japanese,
    Korean,
    Icu,
}

/// Split `text` into maximal spans: Hangul runs go to the Korean
/// dictionary, kana runs (plus Han when hinted Japanese) to the Japanese
/// one, everything else to ICU. Whitespace ends a run, so each Japanese
/// sentence or Korean word group is analysed whole.
fn morph_spans(text: &str, hints: &Hints) -> Vec<(usize, usize, SpanKind)> {
    let classify = |c: char| match CjkScript::of(c) {
        CjkScript::Hangul => SpanKind::Korean,
        CjkScript::Kana => SpanKind::Japanese,
        CjkScript::Han if hints.japanese => SpanKind::Japanese,
        _ => SpanKind::Icu,
    };
    let mut spans: Vec<(usize, usize, SpanKind)> = Vec::new();
    for (offset, c) in text.char_indices() {
        let kind = classify(c);
        let end = offset + c.len_utf8();
        match spans.last_mut() {
            Some((_, last_end, last_kind)) if *last_kind == kind && *last_end == offset => {
                *last_end = end;
            }
            _ => spans.push((offset, end, kind)),
        }
    }
    spans
}

struct Emitter<'a> {
    options: &'a LexOptions,
    ctx: &'a Ctx<'a>,
    purpose: Purpose,
    tokens: Vec<Token>,
    position: u32,
    /// Contiguous run of single CJK characters: (byte offset, char).
    run: Vec<(usize, char)>,
    run_end: usize,
}

impl Emitter<'_> {
    /// Segment `text` (at byte `base` of the document) with ICU and emit
    /// its words.
    fn icu_span(&mut self, base: usize, text: &str) {
        let segmenter = icu_word_segmenter();
        let mut start = 0usize;
        for (end, kind) in segmenter.segment_str(text).iter_with_word_type() {
            let segment = &text[start..end];
            let offset = base + start;
            start = end;
            if !kind.is_word_like() {
                continue;
            }
            if segment.chars().all(|c| CjkScript::of(c).is_cjk()) {
                self.cjk_word(offset, segment);
            } else {
                self.word(offset, segment);
            }
        }
    }

    /// A word from the segmenter (non-CJK): NFKC, split at punctuation the
    /// segmenter kept inside it, lowercase, then routed to its language.
    fn word(&mut self, offset: usize, raw: &str) {
        self.flush_run();
        if raw.is_empty() {
            return;
        }
        if raw.is_ascii() {
            for (from, to, piece) in split_word(raw) {
                self.emit_word(piece, offset + from, offset + to);
            }
        } else {
            use unicode_normalization::UnicodeNormalization;
            let normalized: String = raw.nfkc().collect();
            // Offsets of pieces point into the normalized form; report the
            // whole raw segment as the source span when NFKC changed it.
            let same_length = normalized.len() == raw.len();
            for (from, to, piece) in split_word(&normalized) {
                if same_length {
                    self.emit_word(piece, offset + from, offset + to);
                } else {
                    self.emit_word(piece, offset, offset + raw.len());
                }
            }
        }
    }

    /// Characters of a CJK segment without dictionary word boundaries: they
    /// join the current run and are bigrammed.
    fn cjk_chars(&mut self, offset: usize, word: &str) {
        if !self.run.is_empty() && offset != self.run_end {
            self.flush_run();
        }
        let mut at = offset;
        for c in word.chars() {
            self.run.push((at, c));
            at += c.len_utf8();
        }
        self.run_end = at;
    }

    /// A CJK segment from the ICU dictionary: a single character joins the
    /// bigram run; a word is one token with its bigrams as variants.
    fn cjk_word(&mut self, offset: usize, word: &str) {
        if word.chars().nth(1).is_none() {
            self.cjk_chars(offset, word);
            return;
        }
        self.flush_run();
        use unicode_normalization::UnicodeNormalization;
        let text = self.simplify(word.nfkc().collect());
        let end = offset + word.len();
        let position = self.position;
        self.tokens
            .push(Token::new(text.clone(), position, offset, end));
        if self.purpose == Purpose::Index {
            self.push_bigram_variants(&text, position, offset, end);
        }
        self.position += 1;
    }

    /// Bigrams of a CJK word of three or more characters, as variants.
    fn push_bigram_variants(&mut self, text: &str, position: u32, from: usize, to: usize) {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < 3 || !chars.iter().all(|c| CjkScript::of(*c).is_cjk()) {
            return;
        }
        for pair in chars.windows(2) {
            let mut bigram = String::with_capacity(8);
            bigram.push(pair[0]);
            bigram.push(pair[1]);
            self.tokens
                .push(Token::variant_of(bigram, position, from, to));
        }
    }

    fn flush_run(&mut self) {
        match self.run.len() {
            0 => {}
            1 => {
                let (offset, c) = self.run[0];
                let text = self.simplify(c.to_string());
                self.tokens.push(Token::new(
                    text,
                    self.position,
                    offset,
                    offset + c.len_utf8(),
                ));
                self.position += 1;
            }
            _ => {
                for pair in self.run.windows(2) {
                    let (start, a) = pair[0];
                    let (next, b) = pair[1];
                    let mut text = String::with_capacity(8);
                    text.push(a);
                    text.push(b);
                    let text = self.simplify(text);
                    self.tokens
                        .push(Token::new(text, self.position, start, next + b.len_utf8()));
                    self.position += 1;
                }
            }
        }
        self.run.clear();
    }

    /// Traditional-to-simplified folding of a Han token when enabled.
    fn simplify(&self, text: String) -> String {
        if self.options.han == HanForm::Simplified
            && text.chars().any(|c| CjkScript::of(c) == CjkScript::Han)
        {
            han_to_simplified(&text)
        } else {
            text
        }
    }

    /// Whether a token exceeds `max_token_length` (a byte-length check first,
    /// since a token cannot have more characters than bytes).
    fn too_long(&self, text: &str) -> bool {
        let max = self.options.max_token_length;
        max > 0 && text.len() > max && text.chars().count() > max
    }

    /// Morphemes of a Japanese or Korean run: function morphemes keep
    /// their position and are dropped; content morphemes are emitted with
    /// their base form, their simplified Han form and, when indexing, their
    /// bigrams as variants.
    fn morph_run(&mut self, base: usize, text: &str, analyse: fn(&str) -> Vec<cjk_morph::Morph>) {
        self.flush_run();
        for morph in analyse(text) {
            if !morph.content {
                self.position += 1;
                continue;
            }
            use unicode_normalization::UnicodeNormalization;
            let surface: String = morph.surface.nfkc().collect();
            if self.too_long(&surface) {
                self.position += 1;
                continue;
            }
            let (from, to) = (base + morph.start, base + morph.end);
            let position = self.position;
            match self.purpose {
                Purpose::Index => {
                    let start = self.tokens.len();
                    self.tokens
                        .push(Token::new(surface.clone(), position, from, to));
                    if let Some(lemma) = morph.lemma {
                        self.push_variant(start, lemma, position, from, to);
                    }
                    let simplified = self.simplify(surface.clone());
                    self.push_variant(start, simplified, position, from, to);
                    self.push_bigram_variants(&surface, position, from, to);
                }
                Purpose::Match => {
                    let form = morph.lemma.unwrap_or(surface);
                    self.tokens.push(Token::new(form, position, from, to));
                }
                Purpose::Exact => {
                    self.tokens.push(Token::new(surface, position, from, to));
                }
            }
            self.position += 1;
        }
    }

    /// A variant of the token at `self.tokens[start]`, unless the same text
    /// is already emitted at this position.
    fn push_variant(&mut self, start: usize, text: String, position: u32, from: usize, to: usize) {
        if self.tokens[start..].iter().any(|t| t.text == text) {
            return;
        }
        self.tokens
            .push(Token::variant_of(text, position, from, to));
    }

    /// Route a cleaned word to its language and emit the forms the purpose
    /// asks for. Every path consumes one position.
    fn emit_word(&mut self, word: String, from: usize, to: usize) {
        let options = self.options;
        let script = Script::of_token(&word);
        let route = self
            .ctx
            .hints
            .languages
            .iter()
            .position(|language| language.script() == script);

        // Orthographic normalisation of the written form itself.
        let word = match script {
            Script::Arabic => light_stem::arabic_normalize(&word).unwrap_or(word),
            Script::Cyrillic if word.contains('ё') => word.replace('ё', "е"),
            _ => word,
        };

        if let Some(index) = route
            && self.ctx.stops[index].is_some_and(|set| set.contains(word.as_str()))
        {
            self.position += 1;
            return;
        }
        if self.too_long(&word) {
            self.position += 1;
            return;
        }

        let stem: Option<String> = route.and_then(|index| match options.stem {
            StemMode::None => None,
            StemMode::Light => light_stem::light_stem(self.ctx.hints.languages[index], &word),
            StemMode::Snowball => {
                let stemmer = self.ctx.stemmers.get(index)?;
                match stemmer.stem(&word) {
                    std::borrow::Cow::Borrowed(_) => None,
                    std::borrow::Cow::Owned(stemmed) => (stemmed != word).then_some(stemmed),
                }
            }
        });

        let position = self.position;
        match self.purpose {
            Purpose::Index if options.variants => {
                let start = self.tokens.len();
                let folded = options.fold.then(|| fold_diacritics(&word)).flatten();
                let folded_stem = options
                    .fold
                    .then(|| stem.as_deref().and_then(fold_diacritics))
                    .flatten();
                self.tokens.push(Token::new(word, position, from, to));
                for variant in [stem, folded, folded_stem].into_iter().flatten() {
                    self.push_variant(start, variant, position, from, to);
                }
            }
            Purpose::Exact if options.variants => {
                // The written form is indexed as the token; its stem and
                // folded form are variants at the same position.
                self.tokens.push(Token::new(word, position, from, to));
            }
            Purpose::Index | Purpose::Match | Purpose::Exact => {
                // Without variants the index holds one form per word, the
                // (folded) stem, so every query form is that.
                let base = stem.unwrap_or(word);
                let out = if options.fold && !options.variants {
                    fold_diacritics(&base).unwrap_or(base)
                } else {
                    base
                };
                self.tokens.push(Token::new(out, position, from, to));
            }
        }
        self.position += 1;
    }
}

/// Character-level traditional-to-simplified conversion (OpenCC
/// `TSCharacters` table, see `han_t2s`).
fn han_to_simplified(text: &str) -> String {
    text.chars()
        .map(|c| super::han_t2s::to_simplified(c).unwrap_or(c))
        .collect()
}

/// Diacritic-free form of a Latin, Cyrillic or Greek word (compatibility
/// decomposition, combining marks dropped, lowercased), or `None` when the
/// word has no diacritics or belongs to another script (whose combining
/// marks are letters in their own right).
fn fold_diacritics(word: &str) -> Option<String> {
    if word.is_ascii() {
        return None;
    }
    if !matches!(
        Script::of_token(word),
        Script::Latin | Script::Cyrillic | Script::Greek
    ) {
        return None;
    }
    use unicode_normalization::UnicodeNormalization;
    use unicode_normalization::char::is_combining_mark;
    let folded: String = word
        .nfkd()
        .filter(|c| !is_combining_mark(*c))
        .flat_map(|c| c.to_lowercase())
        .collect();
    (folded != word).then_some(folded)
}

/// The process-wide ICU4X word segmenter (compiled data).
fn icu_word_segmenter() -> &'static icu_segmenter::WordSegmenterBorrowed<'static> {
    static SEGMENTER: std::sync::OnceLock<icu_segmenter::WordSegmenterBorrowed<'static>> =
        std::sync::OnceLock::new();
    SEGMENTER.get_or_init(|| {
        icu_segmenter::WordSegmenter::new_auto(
            icu_segmenter::options::WordBreakInvariantOptions::default(),
        )
    })
}

/// Stop words of a language (NLTK lists), shared for the process lifetime.
/// Entries of the NLTK lists that are pieces of contractions and elisions
/// (`don't` → `don`, `t`; `l'homme` → `l`, `homme`) rather than words. The
/// tokenizer never produces those pieces, and as stop words they would
/// delete real single-letter tokens: `vitamin D`, `T cell`, `Y chromosome`.
fn is_stop_list_fragment(language: Language, word: &str) -> bool {
    if word.contains('\'') || word.contains('\u{2019}') {
        return true;
    }
    match language {
        Language::English => matches!(
            word,
            "s" | "t" | "d" | "ll" | "m" | "o" | "re" | "ve" | "y" | "ain" | "ma"
        ),
        Language::French => matches!(word, "c" | "d" | "j" | "l" | "m" | "n" | "s" | "t" | "qu"),
        Language::Italian => matches!(word, "l" | "c" | "d" | "m" | "n" | "s" | "t" | "v"),
        _ => false,
    }
}

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
            .filter(|word| !is_stop_list_fragment(language, word))
            .map(|word| word.to_string())
            .collect(),
    ));
    Some(*sets.write().entry(language).or_insert(set))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(tokens: &[Token]) -> Vec<(u32, String, bool)> {
        tokens
            .iter()
            .map(|t| (t.position, t.text.clone(), t.variant))
            .collect()
    }

    fn lex(spec: &str) -> LexTokenizer {
        LexTokenizer::new(LexOptions::parse(spec).unwrap())
    }

    #[test]
    fn split_word_cleans_punctuation_like_a_standard_analyzer() {
        let words = |s: &str| {
            split_word(s)
                .into_iter()
                .map(|(_, _, w)| w)
                .collect::<Vec<_>>()
        };
        assert_eq!(words("state-of-the-art"), ["state", "of", "the", "art"]);
        assert_eq!(words("HbA1c/HDL-c"), ["hba1c", "hdl", "c"]);
        assert_eq!(
            words("end.of.sentence.Next"),
            ["end", "of", "sentence", "next"]
        );
        assert_eq!(words("don't"), ["don"]);
        assert_eq!(words("it's"), ["it"]);
        assert_eq!(words("we're"), ["we"]);
        assert_eq!(words("O'Neil"), ["neil"]);
        assert_eq!(words("rock'n'roll"), ["rock", "n", "roll"]);
        assert_eq!(words("John\u{2019}s"), ["john"]);
        assert_eq!(words("cats'"), ["cats"]);
        assert_eq!(words("l'homme"), ["homme"]);
        assert_eq!(words("qu'il"), ["il"]);
        assert_eq!(words("dell'acqua"), ["acqua"]);
        assert_eq!(words("aujourd'hui"), ["aujourd", "hui"]);
        assert_eq!(words("'quoted'"), ["quoted"]);
        assert_eq!(words("U.S.A."), ["usa"]);
        assert_eq!(words("e.g."), ["eg"]);
        assert_eq!(words("Ph.D."), ["ph", "d"]);
        assert_eq!(words("3.14"), ["3.14"]);
        assert_eq!(words("p<0.05"), ["p", "0.05"]);
        assert_eq!(words("1.2.3"), ["1.2.3"]);
        assert_eq!(words("1,000,000"), ["1000000"]);
        assert_eq!(words("word..."), ["word"]);
        assert_eq!(words("soft\u{ad}hyphen"), ["softhyphen"]);
        assert_eq!(words("zero\u{200d}width"), ["zerowidth"]);
        assert_eq!(words("(test)"), ["test"]);
        assert_eq!(words("C++"), ["c"]);
        assert_eq!(words("#tag"), ["tag"]);
        assert_eq!(words("α-synuclein"), ["α", "synuclein"]);
        assert_eq!(words("---"), Vec::<String>::new());
        // Byte spans point at the pieces in the segment.
        assert_eq!(
            split_word("Foo-Bar"),
            vec![(0, 3, "foo".to_string()), (4, 7, "bar".to_string())]
        );
    }

    #[test]
    fn segment_windows_cut_at_punctuation_after_the_soft_size_and_cover_the_text() {
        let sentence = "量子计算机的研究进展，";
        let text: String = sentence.repeat(2000);
        let windows = segment_windows(&text);
        assert!(windows.len() > 1);
        let mut expected_start = 0;
        for (offset, window) in &windows {
            assert_eq!(*offset, expected_start);
            expected_start += window.len();
            let chars = window.chars().count();
            assert!(chars <= SEGMENT_WINDOW_SOFT + sentence.chars().count());
            assert!(window.ends_with('，') || expected_start == text.len());
        }
        assert_eq!(expected_start, text.len());

        // No break characters at all: hard cuts, still covering.
        let solid: String = "的".repeat(10_000);
        let windows = segment_windows(&solid);
        assert_eq!(windows.len(), 10_000 / SEGMENT_WINDOW_HARD + 1);
        assert_eq!(
            windows.iter().map(|(_, w)| w.len()).sum::<usize>(),
            solid.len()
        );
        assert_eq!(segment_windows(""), vec![(0, "")]);
    }

    #[test]
    fn long_han_run_tokenizes_in_bounded_time_with_continuous_positions() {
        let tokenizer = lex("by: languages, default: en, han: simplified");
        let text: String = "量子计算机的研究进展".repeat(20_000);
        let started = std::time::Instant::now();
        let tokens = tokenizer.tokenize_with(&text, Some("zh"), Purpose::Index);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(20),
            "200k Han characters took {:?}",
            started.elapsed()
        );
        assert!(tokens.len() > 20_000);
        let mut last_position = 0;
        let mut last_end = 0;
        for token in tokens.iter().filter(|t| !t.variant) {
            assert!(token.position >= last_position);
            assert!(token.offset_from >= last_end || token.offset_from == last_end);
            last_position = token.position;
            last_end = token.offset_to;
        }
        assert_eq!(last_end, text.len());
    }

    #[test]
    fn variants_index_stem_and_folded_forms_next_to_the_written_word() {
        let tokenizer = lex("by: languages, default: en, stop_words: true");
        let tokens = tokenizer.tokenize("The cell membranes of résumés");
        assert_eq!(
            texts(&tokens),
            vec![
                (1, "cell".to_string(), false),
                (2, "membranes".to_string(), false),
                (2, "membrane".to_string(), true),
                (4, "résumés".to_string(), false),
                (4, "résumé".to_string(), true),
                (4, "resumes".to_string(), true),
                (4, "resume".to_string(), true),
            ]
        );
        // A match query uses the stem, a phrase the written form; never variants.
        let matched = tokenizer.tokenize_with("cell membranes", Some("en"), Purpose::Match);
        assert_eq!(
            texts(&matched),
            vec![
                (0, "cell".to_string(), false),
                (1, "membrane".to_string(), false)
            ]
        );
        let exact = tokenizer.tokenize_with("cell membranes", Some("en"), Purpose::Exact);
        assert_eq!(
            texts(&exact),
            vec![
                (0, "cell".to_string(), false),
                (1, "membranes".to_string(), false)
            ]
        );
        // An unrecognised hint falls back to the default language.
        let fallback = tokenizer.tokenize_with("membranes", Some("xx"), Purpose::Match);
        assert_eq!(fallback[0].text, "membrane");
        // No language at all: the written form.
        let none = lex("").tokenize_with("membranes", None, Purpose::Match);
        assert_eq!(none[0].text, "membranes");
    }

    #[test]
    fn without_variants_the_folded_stem_replaces_the_word_for_every_purpose() {
        let tokenizer = lex("default: en, stem: snowball, variants: false");
        for purpose in [Purpose::Index, Purpose::Match, Purpose::Exact] {
            let tokens = tokenizer.tokenize_with("Running cafés", None, purpose);
            assert_eq!(
                texts(&tokens),
                vec![
                    (0, "run".to_string(), false),
                    (1, "cafe".to_string(), false)
                ],
                "{purpose:?}"
            );
        }
    }

    #[test]
    fn icu_segments_cjk_words_with_bigram_variants_and_thai() {
        let tokenizer = lex("");
        let tokens = tokenizer.tokenize("量子コンピュータの研究");
        let words: Vec<(u32, &str, bool)> = tokens
            .iter()
            .map(|t| (t.position, t.text.as_str(), t.variant))
            .collect();
        assert_eq!(words[0], (0, "量子", false));
        // A dictionary word of three or more characters carries its bigrams.
        let computer: Vec<&(u32, &str, bool)> = words.iter().filter(|(p, _, _)| *p == 1).collect();
        assert_eq!(computer[0], &(1, "コンピュータ", false));
        assert!(computer.iter().skip(1).all(|(_, _, v)| *v));
        assert!(computer.iter().any(|(_, t, _)| *t == "コン"));
        assert!(words.contains(&(2, "の", false)));
        assert!(words.contains(&(3, "研究", false)));
        // Queries emit the words only.
        let query = tokenizer.tokenize_with("量子コンピュータ", None, Purpose::Match);
        assert!(query.iter().all(|t| !t.variant));
        assert_eq!(query.len(), 2);
        // Thai is split into words by the LSTM model.
        let thai = tokenizer.tokenize("สวัสดีครับ");
        assert!(thai.len() >= 2);
        assert!(thai.iter().all(|t| !t.variant));
    }

    #[test]
    fn unicode_and_simple_segmenters_keep_their_behaviour() {
        let unicode = lex("segmenter: unicode, stem: none");
        let tokens: Vec<String> = unicode
            .tokenize("Float-zero p53 日本語")
            .into_iter()
            .filter(|t| !t.variant)
            .map(|t| t.text)
            .collect();
        assert_eq!(tokens, vec!["float", "zero", "p53", "日本", "本語"]);
        let simple = lex("segmenter: simple, stem: none");
        let tokens: Vec<String> = simple
            .tokenize("Float-zero p53")
            .into_iter()
            .map(|t| t.text)
            .collect();
        assert_eq!(tokens, vec!["float", "zero", "p53"]);
    }

    #[test]
    fn single_letter_words_survive_stop_words() {
        let tokenizer = lex("by: languages, default: en, stop_words: true");
        let words = |text: &str, hint: &str| {
            tokenizer
                .tokenize_with(text, Some(hint), Purpose::Index)
                .into_iter()
                .filter(|t| !t.variant)
                .map(|t| t.text)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            words("vitamin D and T cells", "en"),
            ["vitamin", "d", "t", "cells"]
        );
        assert_eq!(words("the Y chromosome", "en"), ["y", "chromosome"]);
        assert_eq!(words("a cat", "en"), ["cat"]);
        assert_eq!(
            words("la vitamine D et l'homme", "fr"),
            ["vitamine", "d", "homme"]
        );
    }

    #[test]
    fn long_tokens_are_dropped_but_keep_their_position() {
        let tokenizer = lex("stem: none, max_token_length: 8");
        let tokens = tokenizer.tokenize("short averyveryverylongtoken next");
        assert_eq!(
            texts(&tokens),
            vec![
                (0, "short".to_string(), false),
                (2, "next".to_string(), false)
            ]
        );
        let unlimited = lex("stem: none, max_token_length: 0");
        assert_eq!(unlimited.tokenize("averyveryverylongtoken").len(), 1);
        // Multibyte tokens are measured in characters.
        let cyrillic = lex("stem: none, max_token_length: 8");
        assert_eq!(cyrillic.tokenize("исследование").len(), 0);
        assert_eq!(cyrillic.tokenize("исследов").len(), 1);
    }

    #[test]
    fn stem_modes_and_arabic_normalisation() {
        let word = "running";
        assert_eq!(
            lex("default: en, stem: none").tokenize(word)[0].text,
            "running"
        );
        assert_eq!(
            lex("default: en, stem: light").tokenize(word)[0].text,
            "running"
        );
        let snowball = lex("default: en, stem: snowball").tokenize(word);
        assert_eq!(
            texts(&snowball),
            vec![
                (0, "running".to_string(), false),
                (0, "run".to_string(), true)
            ]
        );

        let arabic = lex("default: ar").tokenize("الْكِتَابُ");
        assert_eq!(arabic[0].text, "الكتاب");
        assert!(!arabic[0].variant);
        assert_eq!(arabic[1].text, "كتاب");
        assert!(arabic[1].variant);
    }

    #[test]
    fn traditional_chinese_is_indexed_and_queried_as_simplified() {
        let tokenizer = lex("han: simplified");
        let words = |text: &str| -> Vec<String> {
            tokenizer
                .tokenize(text)
                .into_iter()
                .filter(|t| !t.variant)
                .map(|t| t.text)
                .collect()
        };
        assert_eq!(words("電腦網絡"), words("电脑网络"));
        assert!(words("電腦網絡").concat().contains("电脑"));
        let query: Vec<String> = tokenizer
            .tokenize_with("電腦", None, Purpose::Match)
            .into_iter()
            .map(|t| t.text)
            .collect();
        assert_eq!(query, vec!["电脑"]);
        assert_eq!(words("コンピュータ"), vec!["コンピュータ"]);
    }

    #[test]
    fn spec_round_trips_and_renders_only_non_defaults() {
        let text = "lex(by: languages, default: en, stop_words: true, segmenter: unicode, stem: snowball, variants: false, fold: false, max_token_length: 32, han: simplified)";
        let spec = TokenizerSpec::parse(text).unwrap();
        assert_eq!(spec.to_string(), text);
        assert_eq!(spec.hint_field(), Some("languages"));
        assert!(!spec.keeps_original());
        let options = spec.lex().unwrap();
        assert_eq!(options.stem, StemMode::Snowball);
        assert_eq!(options.segmenter, Segmenter::Unicode);
        assert_eq!(options.han, HanForm::Simplified);
        assert_eq!(options.max_token_length, 32);

        assert_eq!(TokenizerSpec::parse("lex()").unwrap().to_string(), "lex()");
        assert_eq!(
            TokenizerSpec::parse("lex(segmenter: icu, stem: light, variants: true, fold: true, max_token_length: 64, han: as_written, cjk: icu, default: none)")
                .unwrap()
                .to_string(),
            "lex()"
        );
        assert_eq!(
            TokenizerSpec::parse("lex(by:languages,default:english,stop_words:true)")
                .unwrap()
                .to_string(),
            "lex(by: languages, default: en, stop_words: true)"
        );
        assert_eq!(
            TokenizerSpec::parse("en_stem").unwrap(),
            TokenizerSpec::Named("en_stem".to_string())
        );
        for bad in [
            "lex(stem: aggressive)",
            "lex(max_token_length: many)",
            "lex(by: )",
            "lex(default: klingon)",
            "lex(segmenter: nope)",
            "lex(han: traditional)",
            "lex(colour: red)",
            "lex(by: lang",
            "en_stem(foo)",
            "",
        ] {
            assert!(TokenizerSpec::parse(bad).is_err(), "{bad}");
        }
        // A spec without `by` ignores hints.
        let fixed = lex("default: en, stem: snowball, variants: false");
        let ru = fixed.tokenize_with("running", Some("ru"), Purpose::Match);
        assert_eq!(ru[0].text, "run");
        assert_eq!(
            TokenizerSpec::parse("lex(cjk: dictionary)").is_ok(),
            cjk_morph::available()
        );
    }

    #[cfg(feature = "cjk-dict")]
    #[test]
    fn dictionary_morphology_for_japanese_and_korean() {
        let tokenizer = lex("by: languages, default: en, cjk: dictionary, han: simplified");
        // Japanese needs the `ja` hint for Han runs; kana runs always go
        // through the dictionary.
        let ja = tokenizer.tokenize_with("研究を食べました", Some("ja"), Purpose::Index);
        assert_eq!(
            texts(&ja),
            vec![
                (0, "研究".to_string(), false),
                (2, "食べ".to_string(), false),
                (2, "食べる".to_string(), true),
            ]
        );
        let matched = tokenizer.tokenize_with("食べました", Some("ja"), Purpose::Match);
        assert_eq!(texts(&matched), vec![(0, "食べる".to_string(), false)]);
        let exact = tokenizer.tokenize_with("食べました", Some("ja"), Purpose::Exact);
        assert_eq!(texts(&exact), vec![(0, "食べ".to_string(), false)]);
        // A Japanese surface with a traditional form gains the simplified
        // variant, so a hint-less Han query (ICU + folding) still matches.
        let learning = tokenizer.tokenize_with("學校", Some("ja"), Purpose::Index);
        assert!(learning.iter().any(|t| t.variant && t.text == "学校"));

        // Korean needs no hint: particles and endings keep their positions.
        let ko = tokenizer.tokenize("학교에서 친구들과 공부했습니다");
        assert_eq!(
            texts(&ko),
            vec![
                (0, "학교".to_string(), false),
                (2, "친구".to_string(), false),
                (5, "공부".to_string(), false),
            ]
        );
        // Mixed text: Latin words still go through ICU and the stemmers.
        let mixed = tokenizer.tokenize_with("cells 학교에서", Some("en"), Purpose::Index);
        assert_eq!(
            texts(&mixed),
            vec![
                (0, "cells".to_string(), false),
                (0, "cell".to_string(), true),
                (1, "학교".to_string(), false),
            ]
        );
    }
}
