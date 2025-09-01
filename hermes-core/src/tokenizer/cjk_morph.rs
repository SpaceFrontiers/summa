//! Japanese and Korean morphology through lindera dictionaries (`cjk-dict`
//! feature): UniDic for Japanese, ko-dic for Korean, embedded in the binary.
//!
//! A run of Japanese or Korean text is segmented into morphemes with
//! part-of-speech tags. Particles, auxiliaries, endings and punctuation are
//! dropped (they keep their positions like stop words); content morphemes
//! are emitted with their dictionary base form as a lemma, which the
//! tokenizer indexes as a same-position variant (`食べました` → `食べ` with
//! variant `食べる`; `학교에서` → `학교`, the particle `에서` dropped).
//!
//! Without the feature the module reports itself unavailable and a spec
//! asking for `cjk: dictionary` fails to parse, so an index built with
//! morphology is never opened by a binary that would tokenize queries
//! differently.

/// One morpheme of a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Morph {
    /// Surface form as written.
    pub surface: String,
    /// Dictionary base form when it differs from the surface.
    pub lemma: Option<String>,
    /// Whether the morpheme is content (indexed) or function (dropped, but
    /// it still consumes a position).
    pub content: bool,
    /// Byte range in the run.
    pub start: usize,
    pub end: usize,
}

/// Whether the dictionaries are compiled in.
pub const fn available() -> bool {
    cfg!(feature = "cjk-dict")
}

/// Load the dictionaries now (a server calls this at start-up so the first
/// Japanese or Korean request does not pay the decode).
pub fn warm_up() {
    if available() {
        let _ = japanese("研究");
        let _ = korean("학교");
    }
}

#[cfg(feature = "cjk-dict")]
mod imp {
    use super::Morph;
    use std::borrow::Cow;
    use std::sync::OnceLock;

    use lindera::dictionary::load_dictionary;
    use lindera::mode::Mode;
    use lindera::segmenter::Segmenter;

    /// How a dictionary's part-of-speech tags classify a morpheme:
    /// `None` = punctuation or whitespace (no position), `Some(false)` =
    /// function morpheme, `Some(true)` = content.
    type Classify = fn(pos: &str) -> Option<bool>;

    fn segmenter(uri: &'static str, cell: &'static OnceLock<Segmenter>) -> &'static Segmenter {
        cell.get_or_init(|| {
            let dictionary = load_dictionary(uri).expect("embedded lindera dictionary");
            Segmenter::new(Mode::Normal, dictionary, None)
        })
    }

    fn analyse(
        segmenter: &Segmenter,
        text: &str,
        classify: Classify,
        lemma_index: Option<usize>,
    ) -> Vec<Morph> {
        let Ok(mut tokens) = segmenter.segment(Cow::Borrowed(text)) else {
            return Vec::new();
        };
        tokens
            .iter_mut()
            .filter_map(|token| {
                let surface = token.surface.to_string();
                if surface.trim().is_empty() {
                    return None;
                }
                let details = token.details();
                let pos = details.first().copied().unwrap_or("");
                let content = classify(pos)?;
                let lemma = lemma_index
                    .and_then(|index| details.get(index).copied())
                    .filter(|base| !base.is_empty() && *base != "*" && *base != surface)
                    .map(str::to_string);
                Some(Morph {
                    surface,
                    lemma,
                    content,
                    start: token.byte_start,
                    end: token.byte_end,
                })
            })
            .collect()
    }

    /// UniDic: details[0] = part of speech, details[10] = orthographic base
    /// form (`orthBase`).
    pub(in crate::tokenizer) fn japanese(text: &str) -> Vec<Morph> {
        static CELL: OnceLock<Segmenter> = OnceLock::new();
        analyse(
            segmenter("embedded://unidic", &CELL),
            text,
            |pos| match pos {
                "補助記号" | "記号" | "空白" => None,
                "助詞" | "助動詞" => Some(false),
                _ => Some(true),
            },
            Some(10),
        )
    }

    /// ko-dic: details[0] = part of speech (Sejong tag set; `+` joins the
    /// tags of an inflected compound). Particles (J*), endings (E*) and
    /// suffixes (XS*) are function morphemes; nouns, verbs, adjectives,
    /// adverbs, roots, foreign words and numbers are content.
    pub(in crate::tokenizer) fn korean(text: &str) -> Vec<Morph> {
        static CELL: OnceLock<Segmenter> = OnceLock::new();
        analyse(
            segmenter("embedded://ko-dic", &CELL),
            text,
            |pos| {
                let first = pos.split('+').next().unwrap_or("");
                match first {
                    "SF" | "SP" | "SS" | "SE" | "SO" | "SC" | "SY" | "SW" => None,
                    _ if first.starts_with('J')
                        || first.starts_with('E')
                        || first.starts_with("XS") =>
                    {
                        Some(false)
                    }
                    _ => Some(true),
                }
            },
            None,
        )
    }
}

#[cfg(feature = "cjk-dict")]
pub(super) use imp::{japanese, korean};

#[cfg(not(feature = "cjk-dict"))]
pub(super) fn japanese(_text: &str) -> Vec<Morph> {
    Vec::new()
}

#[cfg(not(feature = "cjk-dict"))]
pub(super) fn korean(_text: &str) -> Vec<Morph> {
    Vec::new()
}

#[cfg(all(test, feature = "cjk-dict"))]
mod tests {
    use super::*;

    #[test]
    fn japanese_morphemes_carry_base_forms_and_drop_particles() {
        let morphs = japanese("量子コンピュータの研究を食べました");
        let content: Vec<(&str, Option<&str>)> = morphs
            .iter()
            .filter(|m| m.content)
            .map(|m| (m.surface.as_str(), m.lemma.as_deref()))
            .collect();
        assert_eq!(
            content,
            vec![
                ("量子", None),
                ("コンピュータ", None),
                ("研究", None),
                ("食べ", Some("食べる")),
            ]
        );
        assert!(morphs.iter().any(|m| m.surface == "の" && !m.content));
    }

    #[test]
    fn korean_particles_and_endings_are_function_morphemes() {
        let morphs = korean("학교에서 친구들과 공부했습니다");
        let content: Vec<&str> = morphs
            .iter()
            .filter(|m| m.content)
            .map(|m| m.surface.as_str())
            .collect();
        assert_eq!(content, vec!["학교", "친구", "공부"]);
        assert!(morphs.iter().any(|m| m.surface == "에서" && !m.content));
    }
}
