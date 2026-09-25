//! Light (inflection-only) stemmers.
//!
//! Ports of the Apache Lucene light and minimal stemmers (Apache License
//! 2.0; `lucene/analysis/common/.../{en,fr,de,es,it,pt,ru,fi,hu,sv,no,ar}`),
//! which in turn implement Jacques Savoy's light stemming rules (Savoy,
//! "Light stemming approaches for the French, Portuguese, German and
//! Hungarian languages", SAC 2006; Dolamic & Savoy for Russian, Finnish,
//! Swedish, Norwegian; Harman's S-stemmer for English). They strip plural,
//! case and gender endings of nouns and adjectives and leave derivation
//! alone, so `operation` stays `operation` while `cells` becomes `cell`.
//!
//! Each stemmer works on a `char` buffer exactly like the Java originals
//! (`len` is the logical length, edits happen in place), so the rules read
//! the same as upstream and can be checked against it line by line.

use super::Language;

/// Light stem of an already lowercased word, or `None` when the language has
/// no light stemmer or the rules leave the word unchanged.
pub fn light_stem(language: Language, word: &str) -> Option<String> {
    let mut s: Vec<char> = word.chars().collect();
    let len = s.len();
    let new_len = match language {
        Language::English => english_minimal(&mut s, len),
        Language::French => french(&mut s, len),
        Language::German => german(&mut s, len),
        Language::Spanish => spanish(&mut s, len),
        Language::Italian => italian(&mut s, len),
        Language::Portuguese => portuguese(&mut s, len),
        Language::Russian => russian(&mut s, len),
        Language::Finnish => finnish(&mut s, len),
        Language::Hungarian => hungarian(&mut s, len),
        Language::Swedish => swedish(&mut s, len),
        Language::Norwegian => norwegian(&mut s, len),
        Language::Arabic => arabic(&mut s, len),
        Language::Danish
        | Language::Dutch
        | Language::Greek
        | Language::Romanian
        | Language::Tamil
        | Language::Turkish => return None,
    };
    if new_len == len && s.iter().copied().eq(word.chars()) {
        return None;
    }
    Some(s[..new_len].iter().collect())
}

/// Whether a language has a light stemmer.
pub fn has_light_stemmer(language: Language) -> bool {
    !matches!(
        language,
        Language::Danish
            | Language::Dutch
            | Language::Greek
            | Language::Romanian
            | Language::Tamil
            | Language::Turkish
    )
}

// ── StemmerUtil ──────────────────────────────────────────────────────────

#[inline]
fn ends_with(s: &[char], len: usize, suffix: &str) -> bool {
    let mut i = len;
    for c in suffix.chars().rev() {
        if i == 0 {
            return false;
        }
        i -= 1;
        if s[i] != c {
            return false;
        }
    }
    true
}

/// Remove the character at `pos`, returning the new length.
#[inline]
fn delete(s: &mut [char], pos: usize, len: usize) -> usize {
    if pos < len - 1 {
        s.copy_within(pos + 1..len, pos);
    }
    len - 1
}

/// Remove `n` characters at `pos`, returning the new length.
#[inline]
fn delete_n(s: &mut [char], pos: usize, len: usize, n: usize) -> usize {
    if pos + n < len {
        s.copy_within(pos + n..len, pos);
    }
    len - n
}

// ── English (Harman S-stemmer) ────────────────────────────────────────────

fn english_minimal(s: &mut [char], len: usize) -> usize {
    if len < 3 || s[len - 1] != 's' {
        return len;
    }
    match s[len - 2] {
        'u' | 's' => len,
        'e' => {
            if len > 3 && s[len - 3] == 'i' && s[len - 4] != 'a' && s[len - 4] != 'e' {
                s[len - 3] = 'y';
                return len - 2;
            }
            if matches!(s[len - 3], 'i' | 'a' | 'o' | 'e') {
                return len;
            }
            len - 1
        }
        _ => len - 1,
    }
}

// ── French ────────────────────────────────────────────────────────────────

fn french(s: &mut [char], mut len: usize) -> usize {
    if len > 5 && s[len - 1] == 'x' {
        if s[len - 3] == 'a' && s[len - 2] == 'u' && s[len - 4] != 'e' {
            s[len - 2] = 'l';
        }
        len -= 1;
    }
    if len > 3 && s[len - 1] == 'x' {
        len -= 1;
    }
    if len > 3 && s[len - 1] == 's' {
        len -= 1;
    }
    if len > 9 && ends_with(s, len, "issement") {
        len -= 6;
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 8 && ends_with(s, len, "issant") {
        len -= 4;
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 6 && ends_with(s, len, "ement") {
        len -= 4;
        if len > 3 && ends_with(s, len, "ive") {
            len -= 1;
            s[len - 1] = 'f';
        }
        return french_norm(s, len);
    }
    if len > 11 && ends_with(s, len, "ficatrice") {
        len -= 5;
        s[len - 2] = 'e';
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 10 && ends_with(s, len, "ficateur") {
        len -= 4;
        s[len - 2] = 'e';
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 9 && ends_with(s, len, "catrice") {
        len -= 3;
        s[len - 4] = 'q';
        s[len - 3] = 'u';
        s[len - 2] = 'e';
        return french_norm(s, len);
    }
    if len > 8 && ends_with(s, len, "cateur") {
        len -= 2;
        s[len - 4] = 'q';
        s[len - 3] = 'u';
        s[len - 2] = 'e';
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 8 && ends_with(s, len, "atrice") {
        len -= 4;
        s[len - 2] = 'e';
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 7 && ends_with(s, len, "ateur") {
        len -= 3;
        s[len - 2] = 'e';
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 6 && ends_with(s, len, "trice") {
        len -= 1;
        s[len - 3] = 'e';
        s[len - 2] = 'u';
        s[len - 1] = 'r';
    }
    if len > 5 && ends_with(s, len, "ième") {
        return french_norm(s, len - 4);
    }
    if len > 7 && ends_with(s, len, "teuse") {
        len -= 2;
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 6 && ends_with(s, len, "teur") {
        len -= 1;
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 5 && ends_with(s, len, "euse") {
        return french_norm(s, len - 2);
    }
    if len > 8 && ends_with(s, len, "ère") {
        len -= 1;
        s[len - 2] = 'e';
        return french_norm(s, len);
    }
    if len > 7 && ends_with(s, len, "ive") {
        len -= 1;
        s[len - 1] = 'f';
        return french_norm(s, len);
    }
    if len > 4 && (ends_with(s, len, "folle") || ends_with(s, len, "molle")) {
        len -= 2;
        s[len - 1] = 'u';
        return french_norm(s, len);
    }
    if len > 9 && ends_with(s, len, "nnelle") {
        return french_norm(s, len - 5);
    }
    if len > 9 && ends_with(s, len, "nnel") {
        return french_norm(s, len - 3);
    }
    if len > 4 && ends_with(s, len, "ète") {
        len -= 1;
        s[len - 2] = 'e';
    }
    if len > 8 && ends_with(s, len, "ique") {
        len -= 4;
    }
    if len > 8 && ends_with(s, len, "esse") {
        return french_norm(s, len - 3);
    }
    if len > 7 && ends_with(s, len, "inage") {
        return french_norm(s, len - 3);
    }
    if len > 9 && ends_with(s, len, "isation") {
        len -= 7;
        if len > 5 && ends_with(s, len, "ual") {
            s[len - 2] = 'e';
        }
        return french_norm(s, len);
    }
    if len > 9 && ends_with(s, len, "isateur") {
        return french_norm(s, len - 7);
    }
    if len > 8 && ends_with(s, len, "ation") {
        return french_norm(s, len - 5);
    }
    if len > 8 && ends_with(s, len, "ition") {
        return french_norm(s, len - 5);
    }
    french_norm(s, len)
}

fn french_norm(s: &mut [char], mut len: usize) -> usize {
    if len > 4 {
        for c in s.iter_mut().take(len) {
            *c = match *c {
                'à' | 'á' | 'â' => 'a',
                'ô' => 'o',
                'è' | 'é' | 'ê' => 'e',
                'ù' | 'û' => 'u',
                'î' => 'i',
                'ç' => 'c',
                other => other,
            };
        }
        let mut ch = s[0];
        let mut i = 1;
        while i < len {
            if s[i] == ch && ch.is_alphabetic() {
                len = delete(s, i, len);
            } else {
                ch = s[i];
                i += 1;
            }
        }
    }
    if len > 4 && ends_with(s, len, "ie") {
        len -= 2;
    }
    if len > 4 {
        if s[len - 1] == 'r' {
            len -= 1;
        }
        if s[len - 1] == 'e' {
            len -= 1;
        }
        if s[len - 1] == 'e' {
            len -= 1;
        }
        if s[len - 1] == s[len - 2] && s[len - 1].is_alphabetic() {
            len -= 1;
        }
    }
    len
}

// ── German ────────────────────────────────────────────────────────────────

fn german(s: &mut [char], len: usize) -> usize {
    for c in s.iter_mut().take(len) {
        *c = match *c {
            'ä' | 'à' | 'á' | 'â' => 'a',
            'ö' | 'ò' | 'ó' | 'ô' => 'o',
            'ï' | 'ì' | 'í' | 'î' => 'i',
            'ü' | 'ù' | 'ú' | 'û' => 'u',
            other => other,
        };
    }
    let len = german_step1(s, len);
    german_step2(s, len)
}

#[inline]
fn german_st_ending(c: char) -> bool {
    matches!(c, 'b' | 'd' | 'f' | 'g' | 'h' | 'k' | 'l' | 'm' | 'n' | 't')
}

fn german_step1(s: &[char], len: usize) -> usize {
    if len > 5 && s[len - 3] == 'e' && s[len - 2] == 'r' && s[len - 1] == 'n' {
        return len - 3;
    }
    if len > 4 && s[len - 2] == 'e' && matches!(s[len - 1], 'm' | 'n' | 'r' | 's') {
        return len - 2;
    }
    if len > 3 && s[len - 1] == 'e' {
        return len - 1;
    }
    if len > 3 && s[len - 1] == 's' && german_st_ending(s[len - 2]) {
        return len - 1;
    }
    len
}

fn german_step2(s: &[char], len: usize) -> usize {
    if len > 5 && s[len - 3] == 'e' && s[len - 2] == 's' && s[len - 1] == 't' {
        return len - 3;
    }
    if len > 4 && s[len - 2] == 'e' && (s[len - 1] == 'r' || s[len - 1] == 'n') {
        return len - 2;
    }
    if len > 4 && s[len - 2] == 's' && s[len - 1] == 't' && german_st_ending(s[len - 3]) {
        return len - 2;
    }
    len
}

// ── Spanish ───────────────────────────────────────────────────────────────

fn romance_fold(c: char) -> char {
    match c {
        'à' | 'á' | 'â' | 'ä' => 'a',
        'ò' | 'ó' | 'ô' | 'ö' => 'o',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'ù' | 'ú' | 'û' | 'ü' => 'u',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        other => other,
    }
}

fn spanish(s: &mut [char], len: usize) -> usize {
    if len < 5 {
        return len;
    }
    for c in s.iter_mut().take(len) {
        *c = romance_fold(*c);
    }
    match s[len - 1] {
        'o' | 'a' | 'e' => len - 1,
        's' => {
            if s[len - 2] == 'e' && s[len - 3] == 's' && s[len - 4] == 'e' {
                return len - 2;
            }
            if s[len - 2] == 'e' && s[len - 3] == 'c' {
                s[len - 3] = 'z';
                return len - 2;
            }
            if matches!(s[len - 2], 'o' | 'a' | 'e') {
                return len - 2;
            }
            len
        }
        _ => len,
    }
}

// ── Italian ───────────────────────────────────────────────────────────────

fn italian(s: &mut [char], len: usize) -> usize {
    if len < 6 {
        return len;
    }
    for c in s.iter_mut().take(len) {
        *c = romance_fold(*c);
    }
    match s[len - 1] {
        'e' => {
            if s[len - 2] == 'i' || s[len - 2] == 'h' {
                len - 2
            } else {
                len - 1
            }
        }
        'i' => {
            if s[len - 2] == 'h' || s[len - 2] == 'i' {
                len - 2
            } else {
                len - 1
            }
        }
        'a' | 'o' => {
            if s[len - 2] == 'i' {
                len - 2
            } else {
                len - 1
            }
        }
        _ => len,
    }
}

// ── Portuguese ────────────────────────────────────────────────────────────

fn portuguese(s: &mut [char], mut len: usize) -> usize {
    if len < 4 {
        return len;
    }
    len = portuguese_remove_suffix(s, len);
    if len > 3 && s[len - 1] == 'a' {
        len = portuguese_norm_feminine(s, len);
    }
    if len > 4 && matches!(s[len - 1], 'e' | 'a' | 'o') {
        len -= 1;
    }
    for c in s.iter_mut().take(len) {
        *c = match *c {
            'à' | 'á' | 'â' | 'ä' | 'ã' => 'a',
            'ò' | 'ó' | 'ô' | 'ö' | 'õ' => 'o',
            'è' | 'é' | 'ê' | 'ë' => 'e',
            'ù' | 'ú' | 'û' | 'ü' => 'u',
            'ì' | 'í' | 'î' | 'ï' => 'i',
            'ç' => 'c',
            other => other,
        };
    }
    len
}

fn portuguese_remove_suffix(s: &mut [char], mut len: usize) -> usize {
    if len > 4 && ends_with(s, len, "es") && matches!(s[len - 3], 'r' | 's' | 'l' | 'z') {
        return len - 2;
    }
    if len > 3 && ends_with(s, len, "ns") {
        s[len - 2] = 'm';
        return len - 1;
    }
    if len > 4 && (ends_with(s, len, "eis") || ends_with(s, len, "éis")) {
        s[len - 3] = 'e';
        s[len - 2] = 'l';
        return len - 1;
    }
    if len > 4 && ends_with(s, len, "ais") {
        s[len - 2] = 'l';
        return len - 1;
    }
    if len > 4 && ends_with(s, len, "óis") {
        s[len - 3] = 'o';
        s[len - 2] = 'l';
        return len - 1;
    }
    if len > 4 && ends_with(s, len, "is") {
        s[len - 1] = 'l';
        return len;
    }
    if len > 3 && (ends_with(s, len, "ões") || ends_with(s, len, "ães")) {
        len -= 1;
        s[len - 2] = 'ã';
        s[len - 1] = 'o';
        return len;
    }
    if len > 6 && ends_with(s, len, "mente") {
        return len - 5;
    }
    if len > 3 && s[len - 1] == 's' {
        return len - 1;
    }
    len
}

fn portuguese_norm_feminine(s: &mut [char], len: usize) -> usize {
    if len > 7
        && (ends_with(s, len, "inha") || ends_with(s, len, "iaca") || ends_with(s, len, "eira"))
    {
        s[len - 1] = 'o';
        return len;
    }
    if len > 6 {
        if ends_with(s, len, "osa")
            || ends_with(s, len, "ica")
            || ends_with(s, len, "ida")
            || ends_with(s, len, "ada")
            || ends_with(s, len, "iva")
            || ends_with(s, len, "ama")
        {
            s[len - 1] = 'o';
            return len;
        }
        if ends_with(s, len, "ona") {
            s[len - 3] = 'ã';
            s[len - 2] = 'o';
            return len - 1;
        }
        if ends_with(s, len, "ora") {
            return len - 1;
        }
        if ends_with(s, len, "esa") {
            s[len - 3] = 'ê';
            return len - 1;
        }
        if ends_with(s, len, "na") {
            s[len - 1] = 'o';
            return len;
        }
    }
    len
}

// ── Russian ───────────────────────────────────────────────────────────────

fn russian(s: &mut [char], len: usize) -> usize {
    let len = russian_remove_case(s, len);
    russian_normalize(s, len)
}

fn russian_normalize(s: &[char], len: usize) -> usize {
    if len > 3 {
        match s[len - 1] {
            'ь' | 'и' => return len - 1,
            'н' if s[len - 2] == 'н' => return len - 1,
            _ => {}
        }
    }
    len
}

fn russian_remove_case(s: &[char], len: usize) -> usize {
    if len > 6 && (ends_with(s, len, "иями") || ends_with(s, len, "оями")) {
        return len - 4;
    }
    if len > 5
        && [
            "иям", "иях", "оях", "ями", "оям", "оьв", "ами", "его", "ему", "ери", "ими", "ого",
            "ому", "ыми", "оев",
        ]
        .iter()
        .any(|suffix| ends_with(s, len, suffix))
    {
        return len - 3;
    }
    if len > 4
        && [
            "ая", "яя", "ях", "юю", "ах", "ею", "их", "ия", "ию", "ьв", "ою", "ую", "ям", "ых",
            "ея", "ам", "ем", "ей", "ём", "ев", "ий", "им", "ое", "ой", "ом", "ов", "ые", "ый",
            "ым", "ми",
        ]
        .iter()
        .any(|suffix| ends_with(s, len, suffix))
    {
        return len - 2;
    }
    if len > 3
        && matches!(
            s[len - 1],
            'а' | 'е' | 'и' | 'о' | 'у' | 'й' | 'ы' | 'я' | 'ь'
        )
    {
        return len - 1;
    }
    len
}

// ── Finnish ───────────────────────────────────────────────────────────────

fn finnish(s: &mut [char], len: usize) -> usize {
    if len < 4 {
        return len;
    }
    for c in s.iter_mut().take(len) {
        *c = match *c {
            'ä' | 'å' => 'a',
            'ö' => 'o',
            other => other,
        };
    }
    let len = finnish_step1(s, len);
    let len = finnish_step2(s, len);
    let len = finnish_step3(s, len);
    let len = finnish_norm1(s, len);
    finnish_norm2(s, len)
}

#[inline]
fn finnish_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

fn finnish_step1(s: &[char], len: usize) -> usize {
    if len > 8 {
        if ends_with(s, len, "kin") {
            return finnish_step1(s, len - 3);
        }
        if ends_with(s, len, "ko") {
            return finnish_step1(s, len - 2);
        }
    }
    if len > 11 {
        if ends_with(s, len, "dellinen") {
            return len - 8;
        }
        if ends_with(s, len, "dellisuus") {
            return len - 9;
        }
    }
    len
}

fn finnish_step2(s: &[char], len: usize) -> usize {
    if len > 5 {
        if ends_with(s, len, "lla") || ends_with(s, len, "tse") || ends_with(s, len, "sti") {
            return len - 3;
        }
        if ends_with(s, len, "ni") {
            return len - 2;
        }
        if ends_with(s, len, "aa") {
            return len - 1;
        }
    }
    len
}

fn finnish_step3(s: &mut [char], len: usize) -> usize {
    if len > 8 {
        if ends_with(s, len, "nnen") {
            s[len - 4] = 's';
            return len - 3;
        }
        if ends_with(s, len, "ntena") {
            s[len - 5] = 's';
            return len - 4;
        }
        if ends_with(s, len, "tten") {
            return len - 4;
        }
        if ends_with(s, len, "eiden") {
            return len - 5;
        }
    }
    if len > 6 {
        if ends_with(s, len, "neen")
            || ends_with(s, len, "niin")
            || ends_with(s, len, "seen")
            || ends_with(s, len, "teen")
            || ends_with(s, len, "inen")
        {
            return len - 4;
        }
        if s[len - 3] == 'h' && finnish_vowel(s[len - 2]) && s[len - 1] == 'n' {
            return len - 3;
        }
        if ends_with(s, len, "den") {
            s[len - 3] = 's';
            return len - 2;
        }
        if ends_with(s, len, "ksen") {
            s[len - 4] = 's';
            return len - 3;
        }
        if ends_with(s, len, "ssa")
            || ends_with(s, len, "sta")
            || ends_with(s, len, "lla")
            || ends_with(s, len, "lta")
            || ends_with(s, len, "tta")
            || ends_with(s, len, "ksi")
            || ends_with(s, len, "lle")
        {
            return len - 3;
        }
    }
    if len > 5 {
        if ends_with(s, len, "na") || ends_with(s, len, "ne") {
            return len - 2;
        }
        if ends_with(s, len, "nei") {
            return len - 3;
        }
    }
    if len > 4 {
        if ends_with(s, len, "ja") || ends_with(s, len, "ta") {
            return len - 2;
        }
        if s[len - 1] == 'a' {
            return len - 1;
        }
        if s[len - 1] == 'n' && finnish_vowel(s[len - 2]) {
            return len - 2;
        }
        if s[len - 1] == 'n' {
            return len - 1;
        }
    }
    len
}

fn finnish_norm1(s: &mut [char], len: usize) -> usize {
    if len > 5 && ends_with(s, len, "hde") {
        s[len - 3] = 'k';
        s[len - 2] = 's';
        s[len - 1] = 'i';
    }
    if len > 4 && (ends_with(s, len, "ei") || ends_with(s, len, "at")) {
        return len - 2;
    }
    if len > 3 && matches!(s[len - 1], 't' | 's' | 'j' | 'e' | 'a' | 'i') {
        return len - 1;
    }
    len
}

fn finnish_norm2(s: &mut [char], mut len: usize) -> usize {
    if len > 8 && matches!(s[len - 1], 'e' | 'o' | 'u') {
        len -= 1;
    }
    if len > 4 {
        if s[len - 1] == 'i' {
            len -= 1;
        }
        if len > 4 {
            let mut ch = s[0];
            let mut i = 1;
            while i < len {
                if s[i] == ch && matches!(ch, 'k' | 'p' | 't') {
                    len = delete(s, i, len);
                } else {
                    ch = s[i];
                    i += 1;
                }
            }
        }
    }
    len
}

// ── Hungarian ─────────────────────────────────────────────────────────────

fn hungarian(s: &mut [char], len: usize) -> usize {
    for c in s.iter_mut().take(len) {
        *c = match *c {
            'á' => 'a',
            'ë' | 'é' => 'e',
            'í' => 'i',
            'ó' | 'ő' | 'õ' | 'ö' => 'o',
            'ú' | 'ű' | 'ũ' | 'û' | 'ü' => 'u',
            other => other,
        };
    }
    let len = hungarian_remove_case(s, len);
    let len = hungarian_remove_possessive(s, len);
    let len = hungarian_remove_plural(s, len);
    hungarian_normalize(s, len)
}

#[inline]
fn hungarian_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

fn hungarian_remove_case(s: &[char], len: usize) -> usize {
    if len > 6 && ends_with(s, len, "kent") {
        return len - 4;
    }
    if len > 5 {
        if [
            "nak", "nek", "val", "vel", "ert", "rol", "ban", "ben", "bol", "nal", "nel", "hoz",
            "hez", "tol",
        ]
        .iter()
        .any(|suffix| ends_with(s, len, suffix))
        {
            return len - 3;
        }
        if (ends_with(s, len, "al") || ends_with(s, len, "el"))
            && !hungarian_vowel(s[len - 3])
            && s[len - 3] == s[len - 4]
        {
            return len - 3;
        }
    }
    if len > 4 {
        if [
            "at", "et", "ot", "va", "ve", "ra", "re", "ba", "be", "ul", "ig",
        ]
        .iter()
        .any(|suffix| ends_with(s, len, suffix))
        {
            return len - 2;
        }
        if (ends_with(s, len, "on") || ends_with(s, len, "en")) && !hungarian_vowel(s[len - 3]) {
            return len - 2;
        }
        match s[len - 1] {
            't' | 'n' => return len - 1,
            'a' | 'e' if s[len - 2] == s[len - 3] && !hungarian_vowel(s[len - 2]) => {
                return len - 2;
            }
            _ => {}
        }
    }
    len
}

fn hungarian_remove_possessive(s: &[char], len: usize) -> usize {
    if len > 6 {
        if !hungarian_vowel(s[len - 5])
            && (ends_with(s, len, "atok") || ends_with(s, len, "otok") || ends_with(s, len, "etek"))
        {
            return len - 4;
        }
        if ends_with(s, len, "itek") || ends_with(s, len, "itok") {
            return len - 4;
        }
    }
    if len > 5 {
        if !hungarian_vowel(s[len - 4])
            && (ends_with(s, len, "unk") || ends_with(s, len, "tok") || ends_with(s, len, "tek"))
        {
            return len - 3;
        }
        if hungarian_vowel(s[len - 4]) && ends_with(s, len, "juk") {
            return len - 3;
        }
        if ends_with(s, len, "ink") {
            return len - 3;
        }
    }
    if len > 4 {
        if !hungarian_vowel(s[len - 3])
            && ["am", "em", "om", "ad", "ed", "od", "uk"]
                .iter()
                .any(|suffix| ends_with(s, len, suffix))
        {
            return len - 2;
        }
        if hungarian_vowel(s[len - 3])
            && (ends_with(s, len, "nk") || ends_with(s, len, "ja") || ends_with(s, len, "je"))
        {
            return len - 2;
        }
        if ends_with(s, len, "im") || ends_with(s, len, "id") || ends_with(s, len, "ik") {
            return len - 2;
        }
    }
    if len > 3 {
        match s[len - 1] {
            'a' | 'e' => {
                if !hungarian_vowel(s[len - 2]) {
                    return len - 1;
                }
            }
            'm' | 'd' => {
                if hungarian_vowel(s[len - 2]) {
                    return len - 1;
                }
            }
            'i' => return len - 1,
            _ => {}
        }
    }
    len
}

fn hungarian_remove_plural(s: &[char], len: usize) -> usize {
    if len > 3 && s[len - 1] == 'k' {
        if matches!(s[len - 2], 'a' | 'o' | 'e') && len > 4 {
            return len - 2;
        }
        return len - 1;
    }
    len
}

fn hungarian_normalize(s: &[char], len: usize) -> usize {
    if len > 3 && matches!(s[len - 1], 'a' | 'e' | 'i' | 'o') {
        return len - 1;
    }
    len
}

// ── Swedish ───────────────────────────────────────────────────────────────

fn swedish(s: &mut [char], mut len: usize) -> usize {
    if len > 4 && s[len - 1] == 's' {
        len -= 1;
    }
    if len > 7 && (ends_with(s, len, "elser") || ends_with(s, len, "heten")) {
        return len - 5;
    }
    if len > 6
        && ["arne", "erna", "ande", "else", "aste", "orna", "aren"]
            .iter()
            .any(|suffix| ends_with(s, len, suffix))
    {
        return len - 4;
    }
    if len > 5 && (ends_with(s, len, "are") || ends_with(s, len, "ast") || ends_with(s, len, "het"))
    {
        return len - 3;
    }
    if len > 4
        && ["ar", "er", "or", "en", "at", "te", "et"]
            .iter()
            .any(|suffix| ends_with(s, len, suffix))
    {
        return len - 2;
    }
    if len > 3 && matches!(s[len - 1], 't' | 'a' | 'e' | 'n') {
        return len - 1;
    }
    len
}

// ── Norwegian (Bokmål) ────────────────────────────────────────────────────

fn norwegian(s: &mut [char], mut len: usize) -> usize {
    if len > 4 && s[len - 1] == 's' {
        len -= 1;
    }
    if len > 7 && (ends_with(s, len, "heter") || ends_with(s, len, "heten")) {
        return len - 5;
    }
    if len > 5 && (ends_with(s, len, "dom") || ends_with(s, len, "het")) {
        return len - 3;
    }
    if len > 7 && (ends_with(s, len, "elser") || ends_with(s, len, "elsen")) {
        return len - 5;
    }
    if len > 6
        && (ends_with(s, len, "ende")
            || ends_with(s, len, "else")
            || ends_with(s, len, "este")
            || ends_with(s, len, "eren"))
    {
        return len - 4;
    }
    if len > 5 && (ends_with(s, len, "ere") || ends_with(s, len, "est") || ends_with(s, len, "ene"))
    {
        return len - 3;
    }
    if len > 4
        && (ends_with(s, len, "er")
            || ends_with(s, len, "en")
            || ends_with(s, len, "et")
            || ends_with(s, len, "st")
            || ends_with(s, len, "te"))
    {
        return len - 2;
    }
    if len > 3 && matches!(s[len - 1], 'a' | 'e' | 'n') {
        return len - 1;
    }
    len
}

// ── Arabic ────────────────────────────────────────────────────────────────

const ALEF: char = '\u{0627}';
const BEH: char = '\u{0628}';
const TEH_MARBUTA: char = '\u{0629}';
const TEH: char = '\u{062A}';
const FEH: char = '\u{0641}';
const KAF: char = '\u{0643}';
const LAM: char = '\u{0644}';
const NOON: char = '\u{0646}';
const HEH: char = '\u{0647}';
const WAW: char = '\u{0648}';
const YEH: char = '\u{064A}';

/// Arabic orthographic normalisation (Lucene `ArabicNormalizer`): alef
/// variants to bare alef, dotless yeh to yeh, teh marbuta to heh, tatweel
/// and harakat removed. Applied to every Arabic token, original included.
pub fn arabic_normalize(word: &str) -> Option<String> {
    let mut s: Vec<char> = word.chars().collect();
    let full = s.len();
    let len = arabic_normalize_buf(&mut s, full);
    if len == full && s.iter().copied().eq(word.chars()) {
        return None;
    }
    Some(s[..len].iter().collect())
}

fn arabic_normalize_buf(s: &mut [char], mut len: usize) -> usize {
    let mut i = 0;
    while i < len {
        match s[i] {
            '\u{0622}' | '\u{0623}' | '\u{0625}' => {
                s[i] = ALEF;
                i += 1;
            }
            '\u{0649}' => {
                s[i] = YEH;
                i += 1;
            }
            TEH_MARBUTA => {
                s[i] = HEH;
                i += 1;
            }
            '\u{0640}' | '\u{064B}' | '\u{064C}' | '\u{064D}' | '\u{064E}' | '\u{064F}'
            | '\u{0650}' | '\u{0651}' | '\u{0652}' => {
                len = delete(s, i, len);
            }
            _ => i += 1,
        }
    }
    len
}

fn arabic(s: &mut [char], len: usize) -> usize {
    let len = arabic_stem_prefix(s, len);
    arabic_stem_suffix(s, len)
}

fn arabic_stem_prefix(s: &mut [char], len: usize) -> usize {
    const PREFIXES: [&[char]; 7] = [
        &[ALEF, LAM],
        &[WAW, ALEF, LAM],
        &[BEH, ALEF, LAM],
        &[KAF, ALEF, LAM],
        &[FEH, ALEF, LAM],
        &[LAM, LAM],
        &[WAW],
    ];
    for prefix in PREFIXES {
        // The lone waw prefix needs at least three characters after it;
        // the others need two.
        let long_enough = if prefix.len() == 1 {
            len >= 4
        } else {
            len >= prefix.len() + 2
        };
        if long_enough && s[..prefix.len()] == *prefix {
            return delete_n(s, 0, len, prefix.len());
        }
    }
    len
}

fn arabic_stem_suffix(s: &mut [char], mut len: usize) -> usize {
    const SUFFIXES: [&[char]; 10] = [
        &[HEH, ALEF],
        &[ALEF, NOON],
        &[ALEF, TEH],
        &[WAW, NOON],
        &[YEH, NOON],
        &[YEH, HEH],
        &[YEH, TEH_MARBUTA],
        &[HEH],
        &[TEH_MARBUTA],
        &[YEH],
    ];
    for suffix in SUFFIXES {
        if len >= suffix.len() + 2 && s[len - suffix.len()..len] == *suffix {
            len = delete_n(s, len - suffix.len(), len, suffix.len());
        }
    }
    len
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stem(language: Language, word: &str) -> String {
        light_stem(language, word).unwrap_or_else(|| word.to_string())
    }

    #[test]
    fn english_minimal_strips_plurals_only() {
        assert_eq!(stem(Language::English, "cells"), "cell");
        assert_eq!(stem(Language::English, "flies"), "fly");
        assert_eq!(stem(Language::English, "boss"), "boss");
        assert_eq!(stem(Language::English, "bus"), "bus");
        assert_eq!(stem(Language::English, "membranes"), "membrane");
        // Lucene's S-stemmer quirk: `series` is treated as a plural of `sery`.
        assert_eq!(stem(Language::English, "series"), "sery");
        assert_eq!(stem(Language::English, "operation"), "operation");
        assert_eq!(stem(Language::English, "running"), "running");
        assert_eq!(stem(Language::English, "is"), "is");
        assert_eq!(light_stem(Language::English, "cell"), None);
    }

    #[test]
    fn french_light_folds_and_strips_inflection() {
        assert_eq!(stem(Language::French, "chevaux"), "cheval");
        assert_eq!(stem(Language::French, "maisons"), "maison");
        assert_eq!(stem(Language::French, "recherches"), "recherch");
        assert_eq!(stem(Language::French, "études"), "etud");
    }

    #[test]
    fn german_light_strips_plural_and_case() {
        assert_eq!(stem(Language::German, "häuser"), "haus");
        assert_eq!(stem(Language::German, "kinder"), "kind");
        assert_eq!(stem(Language::German, "buches"), "buch");
        assert_eq!(stem(Language::German, "stadt"), "stadt");
    }

    #[test]
    fn spanish_italian_portuguese_light() {
        assert_eq!(stem(Language::Spanish, "gatos"), "gat");
        assert_eq!(stem(Language::Spanish, "luces"), "luz");
        assert_eq!(stem(Language::Spanish, "casa"), "casa");
        assert_eq!(stem(Language::Italian, "ragazzi"), "ragazz");
        assert_eq!(stem(Language::Italian, "amiche"), "amic");
        assert_eq!(stem(Language::Portuguese, "coisas"), "cois");
        assert_eq!(stem(Language::Portuguese, "animais"), "animal");
        assert_eq!(stem(Language::Portuguese, "corações"), "coraca");
    }

    #[test]
    fn russian_light_removes_case_endings() {
        assert_eq!(stem(Language::Russian, "книгами"), "книг");
        assert_eq!(stem(Language::Russian, "книга"), "книг");
        assert_eq!(stem(Language::Russian, "исследования"), "исследован");
        assert_eq!(stem(Language::Russian, "нейронный"), "нейрон");
    }

    #[test]
    fn nordic_finnish_hungarian_light() {
        assert_eq!(stem(Language::Swedish, "böckerna"), "böck");
        assert_eq!(stem(Language::Swedish, "husen"), "hus");
        assert_eq!(stem(Language::Norwegian, "husene"), "hus");
        assert_eq!(stem(Language::Norwegian, "hemmeligheter"), "hemmelig");
        assert_eq!(stem(Language::Finnish, "taloissa"), "talo");
        assert_eq!(stem(Language::Hungarian, "házakban"), "haz");
    }

    #[test]
    fn arabic_normalizer_and_light_stemmer() {
        assert_eq!(arabic_normalize("الْكِتَابُ"), Some("الكتاب".to_string()));
        assert_eq!(arabic_normalize("كتاب"), None);
        // Definite article and plural suffix stripped.
        assert_eq!(stem(Language::Arabic, "الكتاب"), "كتاب");
        assert_eq!(stem(Language::Arabic, "مهندسون"), "مهندس");
    }

    #[test]
    fn unsupported_languages_have_no_light_stemmer() {
        assert!(!has_light_stemmer(Language::Dutch));
        assert_eq!(light_stem(Language::Dutch, "huizen"), None);
    }
}
