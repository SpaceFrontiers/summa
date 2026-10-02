//! Term encoding of common word pairs (`docs/common-word-pairs.md`).

/// Leading and separating byte of a pair term. It never occurs in UTF-8, so
/// pair terms cannot collide with words and sort after every word of a field.
pub(crate) const WORD_PAIR_MARK: u8 = 0xFF;

/// The term of the adjacent pair `first second`: `0xFF first 0xFF second`.
pub(crate) fn word_pair_term(first: &[u8], second: &[u8]) -> Vec<u8> {
    let mut term = Vec::with_capacity(first.len() + second.len() + 2);
    term.push(WORD_PAIR_MARK);
    term.extend_from_slice(first);
    term.push(WORD_PAIR_MARK);
    term.extend_from_slice(second);
    term
}

/// Whether `term` is a pair term rather than a word.
#[cfg_attr(not(feature = "native"), allow(dead_code))]
pub(crate) fn is_word_pair_term(term: &[u8]) -> bool {
    term.first() == Some(&WORD_PAIR_MARK)
}
