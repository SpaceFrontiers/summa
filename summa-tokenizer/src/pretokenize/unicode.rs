use icu_properties::CodePointSetData;
use icu_properties::props::{
    EnumeratedProperty, GeneralCategory, GeneralCategoryGroup, WhiteSpace,
};

#[inline]
pub(crate) fn get_general_category(c: char) -> GeneralCategory {
    GeneralCategory::for_char(c)
}

#[inline]
pub(crate) fn is_gc_letter(gc: GeneralCategory) -> bool {
    GeneralCategoryGroup::Letter.contains(gc)
}

#[inline]
pub(crate) fn is_gc_number(gc: GeneralCategory) -> bool {
    GeneralCategoryGroup::Number.contains(gc)
}

/// Unicode White_Space property — matches the same characters as `\s` in regex.
/// This includes GeneralCategory::Separator (Zs/Zl/Zp) PLUS control characters
/// like U+0009 (TAB), U+000A (LF), U+000D (CR), U+0085 (NEL), etc.
#[inline]
pub(crate) fn is_whitespace(c: char) -> bool {
    // The set is a static compiled-data lookup, but cache the borrowed handle
    // to avoid repeated constructor overhead.
    static WS: std::sync::LazyLock<icu_properties::CodePointSetDataBorrowed<'static>> =
        std::sync::LazyLock::new(CodePointSetData::new::<WhiteSpace>);
    WS.contains(c)
}

#[inline]
pub(crate) fn is_letter(c: char) -> bool {
    is_gc_letter(get_general_category(c))
}

#[inline]
pub(crate) fn is_number(c: char) -> bool {
    is_gc_number(get_general_category(c))
}

#[inline]
pub(crate) fn is_other_complete(c: char) -> bool {
    if c.is_ascii() {
        return !c.is_ascii_alphanumeric() && !c.is_ascii_whitespace();
    }
    let gc = get_general_category(c);
    !is_gc_letter(gc) && !is_gc_number(gc) && !is_whitespace(c)
}

// ---------------------------------------------------------------------------
// Packed codepoint → class table (hot-path classification)
// ---------------------------------------------------------------------------

/// Character class as used by the pretokenization regexes: `\p{L}`, `\p{N}`,
/// `\s` (White_Space), and everything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CharClass {
    Letter = 0,
    Number = 1,
    Whitespace = 2,
    Other = 3,
}

/// 2-bit class per codepoint, 4 codepoints per byte (~272 KiB total).
/// A single L1 load replaces the ICU GeneralCategory trie walk plus the
/// White_Space set binary search that the `is_*` predicates above pay per
/// call. Only the cache lines for scripts actually present in the input
/// stay resident.
static CLASS_TABLE: std::sync::LazyLock<Box<[u8]>> = std::sync::LazyLock::new(build_class_table);

fn build_class_table() -> Box<[u8]> {
    use icu_properties::CodePointMapData;
    const N: usize = 0x110000;
    let mut classes = vec![CharClass::Other as u8; N];
    let gc = CodePointMapData::<GeneralCategory>::new();
    for (group, class) in [
        (GeneralCategoryGroup::Letter, CharClass::Letter),
        (GeneralCategoryGroup::Number, CharClass::Number),
    ] {
        for range in gc.iter_ranges_for_group(group) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    // White_Space is disjoint from GC Letter/Number, so fill order is moot.
    for range in CodePointSetData::new::<WhiteSpace>().iter_ranges() {
        classes[*range.start() as usize..=*range.end() as usize].fill(CharClass::Whitespace as u8);
    }
    classes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| c[0] | (c[1] << 2) | (c[2] << 4) | (c[3] << 6))
        .collect()
}

/// Pre-resolved handle to the packed class table. The static is a
/// `LazyLock<Box<[u8]>>`, so every bare [`class_of`] call pays the
/// lazy-init state check plus a dependent load of the Box pointer before
/// the table load itself; per-char classify loops resolve the handle once
/// and index the slice directly.
#[derive(Clone, Copy)]
pub(crate) struct ClassTable(&'static [u8]);

impl ClassTable {
    #[inline]
    pub(crate) fn get() -> Self {
        Self(&CLASS_TABLE)
    }

    /// [`class_of`] without the per-call static resolution. `cp` must be
    /// a valid scalar value (guaranteed when decoded from valid UTF-8).
    #[inline(always)]
    pub(crate) fn class_of(self, cp: u32) -> CharClass {
        debug_assert!(cp < 0x110000);
        // SAFETY: `self.0` is CLASS_TABLE (the only constructor), sized
        // 0x110000 / 4; cp >> 2 is in range for any scalar value.
        let byte = unsafe { *self.0.get_unchecked((cp >> 2) as usize) };
        match (byte >> ((cp & 3) << 1)) & 3 {
            0 => CharClass::Letter,
            1 => CharClass::Number,
            2 => CharClass::Whitespace,
            _ => CharClass::Other,
        }
    }
}

/// Classify a codepoint with one table load. `cp` must be a valid scalar
/// value (guaranteed when decoded from valid UTF-8).
#[inline(always)]
pub(crate) fn class_of(cp: u32) -> CharClass {
    ClassTable::get().class_of(cp)
}

// ---------------------------------------------------------------------------
// DeepSeek character classes (finer split of `Other`)
// ---------------------------------------------------------------------------

/// Character class as used by the DeepSeek V3 main regex, which additionally
/// distinguishes `\p{M}` (joins letter runs) and `\p{P}`/`\p{S}` (punctuation
/// runs) from the remaining `Other` codepoints (controls, format chars,
/// unassigned), which the regex leaves unmatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum DsCharClass {
    Letter = 0,
    Number = 1,
    Whitespace = 2,
    Mark = 3,
    PunctSym = 4,
    Other = 5,
}

/// Four-class view for schemes whose regex joins `\p{M}` into letter
/// runs and excludes it from punctuation runs (Qwen3.5's
/// `[\p{L}\p{M}]+` / `[^\s\p{L}\p{M}\p{N}]+`): marks classify as
/// letters, everything else as in [`class_of`].
#[inline(always)]
pub(crate) fn class_of_marks_join(cp: u32) -> CharClass {
    DsClassTable::get().class_of_marks_join(cp)
}

/// 4-bit class per codepoint, 2 codepoints per byte (~544 KiB total).
static DS_CLASS_TABLE: std::sync::LazyLock<Box<[u8]>> =
    std::sync::LazyLock::new(build_ds_class_table);

fn build_ds_class_table() -> Box<[u8]> {
    use icu_properties::CodePointMapData;
    const N: usize = 0x110000;
    let mut classes = vec![DsCharClass::Other as u8; N];
    let gc = CodePointMapData::<GeneralCategory>::new();
    for (group, class) in [
        (GeneralCategoryGroup::Letter, DsCharClass::Letter),
        (GeneralCategoryGroup::Number, DsCharClass::Number),
        (GeneralCategoryGroup::Mark, DsCharClass::Mark),
        (GeneralCategoryGroup::Punctuation, DsCharClass::PunctSym),
        (GeneralCategoryGroup::Symbol, DsCharClass::PunctSym),
    ] {
        for range in gc.iter_ranges_for_group(group) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    // White_Space is disjoint from the groups above except Zs/Zl/Zp (which
    // are in none of them), so fill order is moot.
    for range in CodePointSetData::new::<WhiteSpace>().iter_ranges() {
        classes[*range.start() as usize..=*range.end() as usize]
            .fill(DsCharClass::Whitespace as u8);
    }
    classes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| c[0] | (c[1] << 4))
        .collect()
}

/// Pre-resolved handle to the packed DeepSeek class table — same
/// LazyLock-hoist rationale as [`ClassTable`].
#[derive(Clone, Copy)]
pub(crate) struct DsClassTable(&'static [u8]);

impl DsClassTable {
    #[inline]
    pub(crate) fn get() -> Self {
        Self(&DS_CLASS_TABLE)
    }

    /// [`ds_class_of`] without the per-call static resolution. `cp` must
    /// be a valid scalar value (guaranteed when decoded from valid UTF-8).
    #[inline(always)]
    pub(crate) fn ds_class_of(self, cp: u32) -> DsCharClass {
        debug_assert!(cp < 0x110000);
        // SAFETY: `self.0` is DS_CLASS_TABLE (the only constructor), sized
        // 0x110000 / 2; cp >> 1 is in range for any scalar value.
        let byte = unsafe { *self.0.get_unchecked((cp >> 1) as usize) };
        match (byte >> ((cp & 1) << 2)) & 0xF {
            0 => DsCharClass::Letter,
            1 => DsCharClass::Number,
            2 => DsCharClass::Whitespace,
            3 => DsCharClass::Mark,
            4 => DsCharClass::PunctSym,
            _ => DsCharClass::Other,
        }
    }

    /// [`class_of_marks_join`] without the per-call static resolution.
    #[inline(always)]
    pub(crate) fn class_of_marks_join(self, cp: u32) -> CharClass {
        match self.ds_class_of(cp) {
            DsCharClass::Letter | DsCharClass::Mark => CharClass::Letter,
            DsCharClass::Number => CharClass::Number,
            DsCharClass::Whitespace => CharClass::Whitespace,
            DsCharClass::PunctSym | DsCharClass::Other => CharClass::Other,
        }
    }
}

/// Classify a codepoint for the DeepSeek scheme with one table load. `cp`
/// must be a valid scalar value (guaranteed when decoded from valid UTF-8).
#[inline(always)]
pub(crate) fn ds_class_of(cp: u32) -> DsCharClass {
    DsClassTable::get().ds_class_of(cp)
}

// ---------------------------------------------------------------------------
// o200k character classes (case-aware split of Letter)
// ---------------------------------------------------------------------------

/// Character class as used by the o200k regex family (gpt-oss, Nemotron-3),
/// whose letter runs are case-structured:
/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+` etc.
/// `Upper` is Lu|Lt (the strict-uppercase classes that appear only in the
/// first bracket), `Lower` is Ll (only in the second), and `Caseless` is
/// Lm|Lo (in both). Marks (`\p{M}`) are their own class: they join letter
/// runs like `Caseless` but, being outside `\p{L}`, also continue
/// `[^\s\p{L}\p{N}]+` punctuation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum O200kCharClass {
    Upper = 0,
    Lower = 1,
    Caseless = 2,
    Mark = 3,
    Number = 4,
    Whitespace = 5,
    Other = 6,
}

/// 4-bit class per codepoint, 2 codepoints per byte (~544 KiB total).
static O200K_CLASS_TABLE: std::sync::LazyLock<Box<[u8]>> =
    std::sync::LazyLock::new(build_o200k_class_table);

fn build_o200k_class_table() -> Box<[u8]> {
    pack_nibbles(&o200k_classes_unpacked())
}

/// One `O200kCharClass` byte per codepoint (the unpacked form both the
/// o200k and Kimi table builders start from).
fn o200k_classes_unpacked() -> Vec<u8> {
    use icu_properties::CodePointMapData;
    const N: usize = 0x110000;
    let mut classes = vec![O200kCharClass::Other as u8; N];
    let gc = CodePointMapData::<GeneralCategory>::new();
    for (category, class) in [
        (GeneralCategory::UppercaseLetter, O200kCharClass::Upper),
        (GeneralCategory::TitlecaseLetter, O200kCharClass::Upper),
        (GeneralCategory::LowercaseLetter, O200kCharClass::Lower),
        (GeneralCategory::ModifierLetter, O200kCharClass::Caseless),
        (GeneralCategory::OtherLetter, O200kCharClass::Caseless),
    ] {
        for range in gc.iter_ranges_for_value(category) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    for (group, class) in [
        (GeneralCategoryGroup::Mark, O200kCharClass::Mark),
        (GeneralCategoryGroup::Number, O200kCharClass::Number),
    ] {
        for range in gc.iter_ranges_for_group(group) {
            classes[*range.start() as usize..=*range.end() as usize].fill(class as u8);
        }
    }
    // White_Space is disjoint from Letter/Mark/Number, so fill order is moot.
    for range in CodePointSetData::new::<WhiteSpace>().iter_ranges() {
        classes[*range.start() as usize..=*range.end() as usize]
            .fill(O200kCharClass::Whitespace as u8);
    }
    classes
}

/// Pack one-byte-per-codepoint classes into 4-bit nibbles, 2 per byte.
fn pack_nibbles(classes: &[u8]) -> Box<[u8]> {
    classes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| c[0] | (c[1] << 4))
        .collect()
}

/// Classify a codepoint for the o200k scheme family with one table load.
/// `cp` must be a valid scalar value (guaranteed when decoded from valid
/// UTF-8).
#[inline(always)]
pub(crate) fn o200k_class_of(cp: u32) -> O200kCharClass {
    debug_assert!(cp < 0x110000);
    let byte = unsafe { *O200K_CLASS_TABLE.get_unchecked((cp >> 1) as usize) };
    match (byte >> ((cp & 1) << 2)) & 0xF {
        0 => O200kCharClass::Upper,
        1 => O200kCharClass::Lower,
        2 => O200kCharClass::Caseless,
        3 => O200kCharClass::Mark,
        4 => O200kCharClass::Number,
        5 => O200kCharClass::Whitespace,
        _ => O200kCharClass::Other,
    }
}

// ---------------------------------------------------------------------------
// Kimi character classes (o200k classes with Script=Han split out)
// ---------------------------------------------------------------------------

/// Character class for the Kimi (moonshotai K2 family) regex: the o200k
/// classes with `\p{Han}` split out. The pattern gives Han runs their own
/// leading alternative (`[\p{Han}]+`) and excludes Han from both letter
/// brackets (`[...&&[^\p{Han}]]`), but the general-category rules are
/// otherwise Han-blind: `\p{N}{1,3}` still counts a Han numeral and
/// `[^\s\p{L}\p{N}]+` still spans a Han symbol mid-run. Each Han variant
/// therefore records which base class the char behaves as outside a Han
/// run ([`Self::base`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum KimiCharClass {
    Upper = 0,
    Lower = 1,
    Caseless = 2,
    Mark = 3,
    Number = 4,
    Whitespace = 5,
    Other = 6,
    /// Han letters (Lo, plus Lm like U+3005 々): the bulk of `\p{Han}`.
    /// Never letter-run members; a maximal run of Han-class chars starting
    /// a token is one `[\p{Han}]+` token.
    Han = 7,
    /// Han numerals (Nl: U+3007 〇, Suzhou numerals): `\p{N}` mid-number,
    /// Han-run members otherwise.
    HanNumber = 8,
    /// Han symbols and marks (So: Kangxi radicals; Mc: U+16FF0/1 reading
    /// marks, which `&&[^\p{Han}]` evicts from the letter brackets):
    /// punct-run members mid-run, Han-run members at a token start.
    HanOther = 9,
}

impl KimiCharClass {
    /// The o200k class the char behaves as in non-Han-run contexts.
    #[inline(always)]
    pub(crate) fn base(self) -> O200kCharClass {
        match self {
            KimiCharClass::Upper => O200kCharClass::Upper,
            KimiCharClass::Lower => O200kCharClass::Lower,
            KimiCharClass::Caseless | KimiCharClass::Han => O200kCharClass::Caseless,
            KimiCharClass::Mark => O200kCharClass::Mark,
            KimiCharClass::Number | KimiCharClass::HanNumber => O200kCharClass::Number,
            KimiCharClass::Whitespace => O200kCharClass::Whitespace,
            KimiCharClass::Other | KimiCharClass::HanOther => O200kCharClass::Other,
        }
    }

    /// Is the char in `\p{Han}` (a `[\p{Han}]+` run member)?
    #[inline(always)]
    pub(crate) fn is_han(self) -> bool {
        self as u8 >= KimiCharClass::Han as u8
    }
}

/// 4-bit class per codepoint, 2 codepoints per byte (~544 KiB total).
static KIMI_CLASS_TABLE: std::sync::LazyLock<Box<[u8]>> =
    std::sync::LazyLock::new(build_kimi_class_table);

fn build_kimi_class_table() -> Box<[u8]> {
    use icu_properties::CodePointMapData;
    use icu_properties::props::Script;
    let mut classes = o200k_classes_unpacked();
    let script = CodePointMapData::<Script>::new();
    for range in script.iter_ranges_for_value(Script::Han) {
        for cp in *range.start()..=*range.end() {
            let slot = &mut classes[cp as usize];
            *slot = match *slot {
                c if c == O200kCharClass::Number as u8 => KimiCharClass::HanNumber as u8,
                // Marks land in HanOther: `&&[^\p{Han}]` evicts them from
                // the letter brackets, leaving only their punct-run role.
                c if c == O200kCharClass::Other as u8 || c == O200kCharClass::Mark as u8 => {
                    KimiCharClass::HanOther as u8
                }
                // Letters (Lo/Lm); no Han char is Lu/Lt/Ll/Whitespace.
                _ => KimiCharClass::Han as u8,
            };
        }
    }
    pack_nibbles(&classes)
}

/// Classify a codepoint for the Kimi scheme with one table load. `cp` must
/// be a valid scalar value (guaranteed when decoded from valid UTF-8).
#[inline(always)]
pub(crate) fn kimi_class_of(cp: u32) -> KimiCharClass {
    debug_assert!(cp < 0x110000);
    let byte = unsafe { *KIMI_CLASS_TABLE.get_unchecked((cp >> 1) as usize) };
    match (byte >> ((cp & 1) << 2)) & 0xF {
        0 => KimiCharClass::Upper,
        1 => KimiCharClass::Lower,
        2 => KimiCharClass::Caseless,
        3 => KimiCharClass::Mark,
        4 => KimiCharClass::Number,
        5 => KimiCharClass::Whitespace,
        6 => KimiCharClass::Other,
        7 => KimiCharClass::Han,
        8 => KimiCharClass::HanNumber,
        _ => KimiCharClass::HanOther,
    }
}

/// The CJK ranges isolated by the DeepSeek pretokenizer's second Split:
/// `[\u{4E00}-\u{9FA5}\u{3040}-\u{309F}\u{30A0}-\u{30FF}]` (CJK unified
/// ideographs, hiragana, katakana — the two kana blocks are contiguous).
#[inline(always)]
pub(crate) fn is_deepseek_cjk(cp: u32) -> bool {
    (0x4E00..=0x9FA5).contains(&cp) || (0x3040..=0x30FF).contains(&cp)
}
