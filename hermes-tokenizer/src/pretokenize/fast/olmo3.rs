//! Fast pretokenizer for the Olmo 2/3 (dolma2) regex — on aarch64 (NEON)
//! and x86_64 with AVX-512 (runtime-detected) a mask scanner via the shared `cl100k_family::batch_masks` boundary algebra,
//! with the scalar `advance_pos` below as reference, no-SIMD fallback,
//! and bad-zone/tail executor:
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! This is the Qwen2 scheme with cl100k's number rule: `\p{N}{1,3}` matches
//! runs of up to THREE number chars (Qwen2 matches exactly one). Everything
//! else — contractions, letter-run prefixes, the `\s*[\r\n]+` newline rule
//! outranking end-of-input whitespace — is identical to Qwen2.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[cfg(target_arch = "aarch64")]
use super::cl100k_family::batch_masks;
#[cfg(target_arch = "x86_64")]
use super::cl100k_family::batch_masks_x86;
use super::mask::{MaskScheme, MaskState};
use super::{
    decode_cp, is_ascii_ws, is_digit, is_letter, letter_end_at, scan_letters_from, scan_newlines,
    scan_numbers_max3, scan_other_from,
};
use crate::pretokenize::Pretoken;
use crate::pretokenize::unicode::{self, CharClass};

pub(crate) struct Olmo3Scheme;

impl MaskScheme for Olmo3Scheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance_pos(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        // Class-table LazyLock resolved once per batch; the extended
        // path's per-char classify is then a bare slice index.
        let ct = unicode::ClassTable::get();
        batch_masks(bytes, scan, true, move |cp| ct.class_of(cp))
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // Class-table LazyLock resolved once per batch; the extended
        // path's per-char classify is then a bare slice index.
        let ct = unicode::ClassTable::get();
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { batch_masks_x86::<AVX512>(bytes, scan, true, move |cp| ct.class_of(cp)) }
    }
}

/// With SIMD support (aarch64 NEON, or x86_64 AVX-512 detected at runtime),
/// iteration runs the shared cl100k-family mask scanner (see
/// `cl100k_family::batch_masks`); elsewhere every token takes the scalar
/// `advance_pos`.
pub struct FastOlmo3Pretokenizer<'a> {
    bytes: &'a [u8],
    state: MaskState,
}

impl<'a> FastOlmo3Pretokenizer<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self::with_pos(bytes, 0)
    }

    /// Resume iteration at a byte offset previously returned by [`Self::pos`].
    #[inline]
    pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
        Self {
            bytes,
            state: MaskState::new(pos),
        }
    }

    /// Current position as a byte offset into the input.
    #[inline]
    pub fn pos(&self) -> usize {
        self.state.pos
    }
}

impl<'a> Iterator for FastOlmo3Pretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.state.next_span::<Olmo3Scheme>(self.bytes)?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

super::impl_mask_pretoken_spans!(FastOlmo3Pretokenizer, Olmo3Scheme);

/// Whitespace-led token starting at `start`, i.e. the alternatives
/// `\s*[\r\n]+` | `\s+(?!\S)` | `\s+`, in that priority.
/// Precondition: the letter-prefix (`[^\r\n\p{L}\p{N}]?\p{L}+`) and
/// space+punct (` ?[^\s\p{L}\p{N}]+...`) alternatives were ruled out.
#[inline(always)]
fn ws_token_end(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut p = start;
    let mut last_nl_end = 0usize; // 0 = run contains no \r\n
    let mut last_char_start = start;
    while p < len {
        let b = unsafe { *bytes.get_unchecked(p) };
        if b == b'\r' || b == b'\n' {
            last_char_start = p;
            p += 1;
            last_nl_end = p;
        } else if is_ascii_ws(b) {
            last_char_start = p;
            p += 1;
        } else if b >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == CharClass::Whitespace {
                last_char_start = p;
                p += l;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if last_nl_end != 0 {
        return last_nl_end; // `\s*[\r\n]+`: through the last newline, even at EOS
    }
    if p >= len {
        return p; // `\s+(?!\S)`: lookahead succeeds at EOS
    }
    if last_char_start > start {
        return last_char_start; // `\s+(?!\S)`: all but the last ws char
    }
    p // `\s+`: single whitespace char before content
}

/// Advance past one token starting at `pos`. Returns the new position.
/// `pos` must be < `bytes.len()`.
#[inline(always)]
fn advance_pos(bytes: &[u8], pos: usize) -> usize {
    let b0 = unsafe { *bytes.get_unchecked(pos) };

    // Hot path 1: ASCII letter — `\p{L}+` with empty prefix
    if is_letter(b0) {
        return scan_letters_from(bytes, pos + 1);
    }

    // Hot path 2: space prefix
    if b0 == b' ' {
        let Some(&b1) = bytes.get(pos + 1) else {
            return pos + 1; // trailing lone space (`\s+(?!\S)` at EOS)
        };
        if is_letter(b1) {
            return scan_letters_from(bytes, pos + 2); // " word"
        }
        if b1 < 0x80 {
            if is_digit(b1) {
                return pos + 1; // numbers never absorb the space
            }
            if is_ascii_ws(b1) {
                return ws_token_end(bytes, pos);
            }
            // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
            let p = scan_other_from(bytes, pos + 2);
            return scan_newlines(bytes, p);
        }
        let (cp, l) = unsafe { decode_cp(bytes, pos + 1) };
        let p1 = pos + 1 + l;
        match unicode::class_of(cp) {
            CharClass::Letter => return scan_letters_from(bytes, p1),
            CharClass::Whitespace => return ws_token_end(bytes, pos),
            CharClass::Number => return pos + 1,
            CharClass::Other => {
                let p = scan_other_from(bytes, p1);
                return scan_newlines(bytes, p);
            }
        }
    }

    // Non-ASCII
    if b0 >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        let p0 = pos + l;
        let class = unicode::class_of(cp);
        if class == CharClass::Letter {
            return scan_letters_from(bytes, p0);
        }
        if class == CharClass::Number {
            return scan_numbers_max3(bytes, p0, 1);
        }
        // Any non-letter/number char except \r\n may prefix a letter run
        if let Some(p) = letter_end_at(bytes, p0) {
            return scan_letters_from(bytes, p);
        }
        if class == CharClass::Whitespace {
            return ws_token_end(bytes, pos);
        }
        let p = scan_other_from(bytes, p0);
        return scan_newlines(bytes, p);
    }

    // ASCII digit: `\p{N}{1,3}`
    if is_digit(b0) {
        return scan_numbers_max3(bytes, pos + 1, 1);
    }

    // Apostrophe: case-insensitive contractions
    if b0 == b'\'' {
        match bytes.get(pos + 1).map(u8::to_ascii_lowercase) {
            Some(b's' | b'd' | b'm' | b't') => return pos + 2,
            Some(b'l') if bytes.get(pos + 2).map(u8::to_ascii_lowercase) == Some(b'l') => {
                return pos + 3;
            }
            Some(b'v') if bytes.get(pos + 2).map(u8::to_ascii_lowercase) == Some(b'e') => {
                return pos + 3;
            }
            Some(b'r') if bytes.get(pos + 2).map(u8::to_ascii_lowercase) == Some(b'e') => {
                return pos + 3;
            }
            _ => {}
        }
        // U+017F LATIN SMALL LETTER LONG S case-folds to 's' under `(?i)`
        if bytes.get(pos + 1) == Some(&0xC5) && bytes.get(pos + 2) == Some(&0xBF) {
            return pos + 3;
        }
        // Not a contraction: `'` can still prefix a letter run
        if let Some(p) = letter_end_at(bytes, pos + 1) {
            return scan_letters_from(bytes, p);
        }
        let p = scan_other_from(bytes, pos + 1);
        return scan_newlines(bytes, p);
    }

    // \r and \n are excluded from the letter-run prefix
    if b0 == b'\r' || b0 == b'\n' {
        return ws_token_end(bytes, pos);
    }

    // Other ASCII whitespace (\t, \x0b, \x0c) may prefix a letter run
    if is_ascii_ws(b0) {
        if let Some(p) = letter_end_at(bytes, pos + 1) {
            return scan_letters_from(bytes, p);
        }
        return ws_token_end(bytes, pos);
    }

    // ASCII punctuation/symbol
    if let Some(p) = letter_end_at(bytes, pos + 1) {
        return scan_letters_from(bytes, p);
    }
    let p = scan_other_from(bytes, pos + 1);
    scan_newlines(bytes, p)
}
