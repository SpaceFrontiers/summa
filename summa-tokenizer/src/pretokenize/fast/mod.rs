//! Fast scalar pretokenizers, one submodule per pretokenization scheme.
//!
//! Each scheme implements an advance function that consumes exactly one
//! pretoken, wrapped in a thin iterator struct. The byte predicates and
//! SWAR scans below are shared; a new scheme (e.g. o200k) should slot in
//! as another submodule reusing these primitives where its character
//! classes line up.

pub(crate) mod cl100k_family;
pub(crate) mod mask;
pub(crate) mod o200k_family;

pub mod cl100k;
pub mod deepseek_v3;
pub mod kimi;
pub mod nemotron;
pub mod o200k;
pub mod olmo3;
pub mod qwen2;
pub mod qwen3_5;
pub mod r50k;

pub use cl100k::FastCl100kPretokenizer;
pub use deepseek_v3::FastDeepSeekV3Pretokenizer;
pub use kimi::FastKimiPretokenizer;
pub use nemotron::FastNemotronPretokenizer;
pub use o200k::FastO200kPretokenizer;
pub use olmo3::FastOlmo3Pretokenizer;
pub use qwen2::FastQwen2Pretokenizer;
pub use qwen3_5::FastQwen35Pretokenizer;
pub use r50k::FastR50kPretokenizer;

use crate::pretokenize::SpanBatch;
use crate::pretokenize::unicode;

// -----------------------------------------------------------------------
// Shared chunked span pull for the mask-scanner pretokenizers
// -----------------------------------------------------------------------

/// The `PretokenSpans::fill_spans_keyed` body shared by every mask-scanner
/// pretokenizer (all of them wrap a `(bytes, MaskState)` pair). With a SIMD
/// scanner this is the two-phase chunk walker
/// ([`mask::MaskState::fill_spans_two_phase`]): boundary harvest into a
/// flat buffer, then a branch-free emission loop — the per-span refill
/// ladder and pack branches of the fused pull loop were the largest single
/// source of encode's discarded issue bandwidth. Without SIMD support it
/// pulls spans one at a time over `next_span`, fusing its
/// `#[inline(always)]` walker body into one tight loop. `#[inline(never)]`:
/// each monomorphization is its own out-of-line loop, keeping its register
/// allocation away from the (register-hungry) encode loop that calls it.
/// Routing this through `Iterator::next` instead measured ~23% of warm
/// encode time in un-inlined call overhead.
#[inline(never)]
pub(crate) fn fill_spans_keyed_mask<'a, S: mask::MaskScheme>(
    bytes: &'a [u8],
    state: &mut mask::MaskState,
    batch: &mut SpanBatch<'a>,
    prefetch: &impl Fn(u64),
) -> usize {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if mask::simd_scanner_available() {
        return state.fill_spans_two_phase::<S>(bytes, batch, prefetch);
    }
    crate::pretokenize::fill_spans_keyed_with_buf(
        bytes,
        // next_span returns in-bounds, nonempty span boundaries.
        || state.next_span::<S>(bytes),
        batch,
        prefetch,
    )
}

/// Implement the shared constructor, cursor, iterator, and chunked-span
/// surface for a mask-scanner pretokenizer.
///
/// The concrete module still declares the `{ bytes, state: MaskState }`
/// struct and its [`mask::MaskScheme`], keeping scheme-specific boundary
/// semantics next to their scalar and SIMD implementations. This macro owns
/// only the identical adapter layer between that scheme and the public
/// pretokenizer interfaces.
macro_rules! impl_mask_pretokenizer {
    ($pretokenizer:ident, $scheme:ty) => {
        impl<'a> $pretokenizer<'a> {
            #[inline]
            pub fn new(bytes: &'a [u8]) -> Self {
                Self::with_pos(bytes, 0)
            }

            /// Resume iteration at a byte offset previously returned by
            /// [`Self::pos`].
            #[inline]
            pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
                Self {
                    bytes,
                    state: crate::pretokenize::fast::mask::MaskState::new(pos),
                }
            }

            /// Current position as a byte offset into the input.
            #[inline]
            pub fn pos(&self) -> usize {
                self.state.pos
            }
        }

        impl<'a> Iterator for $pretokenizer<'a> {
            type Item = crate::pretokenize::Pretoken<'a>;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                let (start, end) = self.state.next_span::<$scheme>(self.bytes)?;
                Some(crate::pretokenize::Pretoken(&self.bytes[start..end]))
            }
        }

        // SAFETY: delegates to `fill_spans_keyed_mask`, whose bodies
        // (`fill_spans_keyed_with_buf` / `fill_spans_two_phase`) write
        // exactly the first `n` entries from live spans of `self.bytes`.
        unsafe impl<'a> crate::pretokenize::PretokenSpans<'a> for $pretokenizer<'a> {
            #[inline]
            fn fill_spans_keyed(
                &mut self,
                batch: &mut crate::pretokenize::SpanBatch<'a>,
                prefetch: &impl Fn(u64),
            ) -> usize {
                crate::pretokenize::fast::fill_spans_keyed_mask::<$scheme>(
                    self.bytes,
                    &mut self.state,
                    batch,
                    prefetch,
                )
            }
        }
    };
}
pub(crate) use impl_mask_pretokenizer;

// -----------------------------------------------------------------------
// Branchless byte predicates
// -----------------------------------------------------------------------

#[inline(always)]
pub(crate) fn is_letter(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') < 26
}

#[inline(always)]
pub(crate) fn is_digit(b: u8) -> bool {
    b.wrapping_sub(b'0') < 10
}

#[inline(always)]
pub(crate) fn is_ascii_ws(b: u8) -> bool {
    b == b' ' || b.wrapping_sub(9) < 5
}

#[inline(always)]
pub(crate) unsafe fn decode_non_ascii(bytes: &[u8]) -> char {
    unsafe {
        std::str::from_utf8_unchecked(bytes)
            .chars()
            .next()
            .unwrap_unchecked()
    }
}

/// Decode one non-ASCII scalar. Requires only `pos < bytes.len()` and
/// `bytes[pos] >= 0x80`; arbitrary (invalid) bytes are tolerated and
/// decode deterministically. Returns the codepoint and the number of
/// bytes consumed.
///
/// Invalid input is garbage-in/defined-garbage-out, with two hard
/// guarantees the walkers rely on:
///
/// - Never reads past `bytes.len()`, and the returned length never
///   overruns it: a multi-byte lead whose sequence is cut off by the
///   buffer end (a truncated tail) consumes exactly the bytes that
///   remain and yields [`CP_INVALID`]. (Pre-fix this read up to 3 bytes
///   past the slice and returned an end past `len` — walker panic on the
///   Iterator path, out-of-bounds span on the SpanBatch path.)
/// - The codepoint is always `<= 0x10FFFF`, so the packed class-table
///   lookups (`unicode::class_of` / `ds_class_of`, indexed unchecked)
///   stay in bounds: invalid leads 0xF5..=0xFF take the 4-byte branch
///   and can assemble "codepoints" up to 0x1FFFFF, which are clamped to
///   [`CP_INVALID`]. (Pre-fix the table lookup read up to ~246 KB past
///   the table — heap memory whose contents depend on other threads'
///   allocations, which is what made >65 KB invalid-UTF-8 pretokens
///   split nondeterministically between the walker paths.)
///
/// The clamp target U+10FFFF is unassigned (a noncharacter) — class
/// `Other` in every scheme's classifier — so truncated or
/// beyond-Unicode garbage classifies like any other unassigned
/// codepoint, identically on every path.
#[inline(always)]
pub(crate) unsafe fn decode_cp(bytes: &[u8], pos: usize) -> (u32, usize) {
    if pos + 4 > bytes.len() {
        // Within 3 bytes of the buffer end: the only region where a
        // sequence can be truncated. Cold: interior calls (the hot ones)
        // never take it, and the branch predicts not-taken.
        return decode_cp_near_end(bytes, pos);
    }
    // SAFETY: pos + 4 <= len just checked.
    unsafe { decode_cp_inbounds(bytes, pos) }
}

/// [`decode_cp`] without the buffer-end guard, for callers that already
/// guarantee `pos + 4 <= bytes.len()` structurally (the mask-scanner
/// batch helpers, whose `scan + 70 <= len` batch guard covers every call
/// site's worst case) — keeps the tail check out of the batch
/// classification path. Identical results to [`decode_cp`] on any input
/// where both are callable, including the [`CP_INVALID`] clamp for
/// beyond-Unicode garbage from invalid 4-byte leads.
#[inline(always)]
pub(crate) unsafe fn decode_cp_inbounds(bytes: &[u8], pos: usize) -> (u32, usize) {
    unsafe {
        let b0 = *bytes.get_unchecked(pos) as u32;
        let b1 = (*bytes.get_unchecked(pos + 1) & 0x3F) as u32;
        if b0 < 0xE0 {
            return (((b0 & 0x1F) << 6) | b1, 2);
        }
        let b2 = (*bytes.get_unchecked(pos + 2) & 0x3F) as u32;
        if b0 < 0xF0 {
            return (((b0 & 0x0F) << 12) | (b1 << 6) | b2, 3);
        }
        let b3 = (*bytes.get_unchecked(pos + 3) & 0x3F) as u32;
        (
            (((b0 & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3).min(CP_INVALID),
            4,
        )
    }
}

/// The codepoint reported for byte sequences that cannot be decoded
/// within bounds (truncated tails) or that assemble past the Unicode
/// range (invalid 4-byte-lead garbage). U+10FFFF: the largest scalar
/// value, an unassigned noncharacter, class `Other` in [`unicode::class_of`],
/// `class_of_marks_join`, and `ds_class_of` alike.
pub(crate) const CP_INVALID: u32 = 0x10FFFF;

/// [`decode_cp`]'s slow path for `pos + 4 > bytes.len()`: decodes with
/// per-byte bounds, identical results to the fast path for complete
/// sequences; a sequence truncated by the buffer end consumes exactly
/// the remaining bytes and yields [`CP_INVALID`].
#[cold]
#[inline(never)]
fn decode_cp_near_end(bytes: &[u8], pos: usize) -> (u32, usize) {
    let len = bytes.len();
    let b0 = bytes[pos] as u32;
    let need = if b0 < 0xE0 {
        2
    } else if b0 < 0xF0 {
        3
    } else {
        4
    };
    if pos + need > len {
        // Truncated tail: consume the rest of the buffer as one
        // unclassifiable char so every walker path terminates the final
        // pretoken at `len` the same way.
        return (CP_INVALID, len - pos);
    }
    let b1 = (bytes[pos + 1] & 0x3F) as u32;
    if need == 2 {
        return (((b0 & 0x1F) << 6) | b1, 2);
    }
    let b2 = (bytes[pos + 2] & 0x3F) as u32;
    if need == 3 {
        return (((b0 & 0x0F) << 12) | (b1 << 6) | b2, 3);
    }
    let b3 = (bytes[pos + 3] & 0x3F) as u32;
    (
        (((b0 & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3).min(CP_INVALID),
        4,
    )
}

/// `[\r\n]*`: advance past a run of CR/LF bytes (trailing newlines after a
/// punctuation run in the cl100k-family schemes).
#[inline(always)]
pub(crate) fn scan_newlines(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if b == b'\r' || b == b'\n' {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

/// End of a whitespace-led token for the common
/// `\s*[\r\n]+ | \s+(?!\S) | \s+` family of alternatives.
///
/// `NEWLINE_AT_EOS` captures the one regex-ordering difference between the
/// supported families: Qwen/OLMo/o200k let the newline alternative win even
/// at end of input, while cl100k's earlier `\s++$` keeps all trailing
/// whitespace together. `is_unicode_whitespace` supplies the scheme's packed
/// Unicode classifier and is monomorphized into each hot scalar walker.
///
/// Precondition: `start` begins a whitespace run and any higher-priority
/// letter-prefix or space-punctuation alternative has already been ruled out.
#[inline(always)]
pub(crate) fn whitespace_token_end<const NEWLINE_AT_EOS: bool>(
    bytes: &[u8],
    start: usize,
    is_unicode_whitespace: impl Fn(u32) -> bool,
) -> usize {
    let len = bytes.len();
    let mut pos = start;
    let mut last_newline_end = 0usize;
    let mut last_char_start = start;
    while pos < len {
        let byte = unsafe { *bytes.get_unchecked(pos) };
        if byte == b'\r' || byte == b'\n' {
            last_char_start = pos;
            pos += 1;
            last_newline_end = pos;
        } else if is_ascii_ws(byte) {
            last_char_start = pos;
            pos += 1;
        } else if byte >= 0x80 {
            let (codepoint, width) = unsafe { decode_cp(bytes, pos) };
            if is_unicode_whitespace(codepoint) {
                last_char_start = pos;
                pos += width;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if NEWLINE_AT_EOS && last_newline_end != 0 {
        return last_newline_end;
    }
    if pos >= len {
        return pos;
    }
    if last_newline_end != 0 {
        return last_newline_end;
    }
    if last_char_start > start {
        return last_char_start;
    }
    pos
}

/// End offset of the case-insensitive contraction beginning at
/// `apostrophe`, or `None` when the following bytes are not one of
/// `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, or `'d`.
///
/// U+017F LATIN SMALL LETTER LONG S is included because Unicode
/// case-insensitive matching folds it to `s`.
#[inline(always)]
pub(crate) fn contraction_end(bytes: &[u8], apostrophe: usize) -> Option<usize> {
    if bytes.get(apostrophe) != Some(&b'\'') {
        return None;
    }
    match bytes.get(apostrophe + 1).map(u8::to_ascii_lowercase) {
        Some(b's' | b'd' | b'm' | b't') => Some(apostrophe + 2),
        Some(b'l') if bytes.get(apostrophe + 2).map(u8::to_ascii_lowercase) == Some(b'l') => {
            Some(apostrophe + 3)
        }
        Some(b'v') if bytes.get(apostrophe + 2).map(u8::to_ascii_lowercase) == Some(b'e') => {
            Some(apostrophe + 3)
        }
        Some(b'r') if bytes.get(apostrophe + 2).map(u8::to_ascii_lowercase) == Some(b'e') => {
            Some(apostrophe + 3)
        }
        Some(0xC5) if bytes.get(apostrophe + 2) == Some(&0xBF) => Some(apostrophe + 3),
        _ => None,
    }
}

/// If the char at `pos` is a letter (`\p{L}` under the 4-way `CharClass`
/// classifier), return the offset just past it.
#[inline(always)]
pub(crate) fn letter_end_at(bytes: &[u8], pos: usize) -> Option<usize> {
    let &b = bytes.get(pos)?;
    if is_letter(b) {
        return Some(pos + 1);
    }
    if b >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        if unicode::class_of(cp) == unicode::CharClass::Letter {
            return Some(pos + l);
        }
    }
    None
}

// -----------------------------------------------------------------------
// SWAR
// -----------------------------------------------------------------------

pub(crate) const HI: u64 = 0x8080_8080_8080_8080;

/// Returns the high bit set in each lane that is NOT an ASCII letter,
/// computed directly (rather than as the complement of a letter mask) so
/// the scan loop can branch on `!= 0` and reuse the value for `trailing_zeros`.
#[inline(always)]
pub(crate) fn swar64_letter_nonmask(word: u64) -> u64 {
    let lowered = word | 0x2020_2020_2020_2020;
    let ge_a = (lowered | HI).wrapping_sub(0x6161_6161_6161_6161);
    let le_z = 0xFAFA_FAFA_FAFA_FAFA_u64.wrapping_sub(lowered);
    !(ge_a & le_z) & HI
}

/// SWAR letter scan: advances `pos` past ASCII letters.
/// Returns the updated pos.
#[inline(always)]
pub(crate) fn swar_scan_letters(bytes: &[u8], mut pos: usize) -> usize {
    let len = bytes.len();
    // SWAR: 8 bytes at a time
    while pos + 8 <= len {
        let word = unsafe { (bytes.as_ptr().add(pos) as *const u64).read_unaligned() };
        if word & HI != 0 {
            break;
        }
        let nonletter = swar64_letter_nonmask(word);
        if nonletter != 0 {
            return pos + nonletter.to_le().trailing_zeros() as usize / 8;
        }
        pos += 8;
    }
    // Scalar tail
    while pos < len {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_letter(b) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

/// NEON letter scan: 16 bytes per iteration. Non-ASCII bytes (>= 0x80) fail
/// the `<= 'z'` check after case-folding, so they stop the run exactly like
/// non-letters; the caller's unicode continuation handles them.
///
/// NOT used by `scan_letters_from`: measured 0.83x of the SWAR scan on OWT.
/// The `vshrn`-based movemask needs a vector→GPR transfer whose latency sits
/// on the serial per-token chain, and typical letter runs (~4-6 bytes) fit in
/// one SWAR iteration anyway. Kept as a reference / benchmark baseline.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn neon_scan_letters(bytes: &[u8], mut pos: usize) -> usize {
    use std::arch::aarch64::*;
    let len = bytes.len();
    while pos + 16 <= len {
        unsafe {
            let v = vld1q_u8(bytes.as_ptr().add(pos));
            let lowered = vorrq_u8(v, vdupq_n_u8(0x20));
            let ge_a = vcgeq_u8(lowered, vdupq_n_u8(b'a'));
            let le_z = vcleq_u8(lowered, vdupq_n_u8(b'z'));
            let nonletter = vmvnq_u8(vandq_u8(ge_a, le_z));
            // Narrowing movemask: 4 bits per lane, first set nibble = first
            // non-letter lane.
            let mask = vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(
                vreinterpretq_u16_u8(nonletter),
            )));
            if mask != 0 {
                return pos + (mask.trailing_zeros() >> 2) as usize;
            }
        }
        pos += 16;
    }
    while pos < len {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_letter(b) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

// -----------------------------------------------------------------------
// Shared run scans (`\p{L}+`, `\p{N}+`, `\p{N}{1,3}`, `[^\s\p{L}\p{N}]+`)
// -----------------------------------------------------------------------

/// `\p{N}{1,3}`: extend a number run that already matched `consumed` chars
/// to at most 3 chars total. Shared by the cl100k and olmo3 schemes.
#[inline(always)]
pub(crate) fn scan_numbers_max3(bytes: &[u8], mut pos: usize, mut consumed: u32) -> usize {
    let len = bytes.len();
    while consumed < 3 && pos < len {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_digit(b) {
            pos += 1;
            consumed += 1;
            continue;
        }
        if b >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, pos) };
            if unicode::class_of(cp) == unicode::CharClass::Number {
                pos += l;
                consumed += 1;
                continue;
            }
        }
        break;
    }
    pos
}

#[inline(always)]
pub(crate) fn scan_letters_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        p = swar_scan_letters(bytes, p);
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == unicode::CharClass::Letter {
                p += l;
                continue;
            }
        }
        return p;
    }
}

#[inline(always)]
pub(crate) fn scan_digits_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len && is_digit(unsafe { *bytes.get_unchecked(p) }) {
            p += 1;
        }
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == unicode::CharClass::Number {
                p += l;
                continue;
            }
        }
        return p;
    }
}

#[inline(always)]
pub(crate) fn scan_other_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len {
            let b = unsafe { *bytes.get_unchecked(p) };
            if b >= 0x80 {
                break;
            }
            if is_letter(b) || is_digit(b) || is_ascii_ws(b) {
                return p;
            }
            p += 1;
        }
        if p < len {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == unicode::CharClass::Other {
                p += l;
                continue;
            }
        }
        return p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::{Pretoken, PretokenSpans, SpanBatch};

    fn pieces<'a>(scanner: impl Iterator<Item = Pretoken<'a>>) -> Vec<&'a [u8]> {
        scanner.map(|pretoken| pretoken.0).collect()
    }

    #[test]
    fn every_mask_adapter_preserves_boundaries_when_resumed() {
        macro_rules! assert_resumable {
            ($scanner:ident) => {{
                let input = "we'RE 123\n\u{2003}漢字".as_bytes();
                let full = pieces($scanner::new(input));
                assert_eq!(full.concat(), input, stringify!($scanner));

                let first_end = full[0].len();
                let mut cursor = $scanner::new(input);
                assert_eq!(cursor.next().unwrap().0, full[0], stringify!($scanner));
                assert_eq!(cursor.pos(), first_end, stringify!($scanner));
                let resumed = pieces($scanner::with_pos(input, first_end));
                assert_eq!(resumed.as_slice(), &full[1..], stringify!($scanner));

                let mut chunked = $scanner::new(input);
                let mut batch = SpanBatch::new();
                let count = chunked.fill_spans_keyed(&mut batch, &|_| {});
                let chunked_spans = (0..count)
                    // SAFETY: `count` is the live prefix written by this fill.
                    .map(|index| unsafe { batch.span(index) })
                    .collect::<Vec<_>>();
                assert_eq!(chunked_spans, full, stringify!($scanner));
                assert_eq!(
                    chunked.fill_spans_keyed(&mut batch, &|_| {}),
                    0,
                    stringify!($scanner)
                );
            }};
        }

        assert_resumable!(FastR50kPretokenizer);
        assert_resumable!(FastCl100kPretokenizer);
        assert_resumable!(FastQwen2Pretokenizer);
        assert_resumable!(FastQwen35Pretokenizer);
        assert_resumable!(FastOlmo3Pretokenizer);
        assert_resumable!(FastO200kPretokenizer);
        assert_resumable!(FastNemotronPretokenizer);
        assert_resumable!(FastKimiPretokenizer);
    }

    #[test]
    fn shared_whitespace_walker_preserves_regex_priority() {
        let trailing = b" \n  ";
        let ascii_only = |_| false;
        assert_eq!(
            whitespace_token_end::<false>(trailing, 0, ascii_only),
            trailing.len(),
            "cl100k's end-of-input alternative keeps trailing whitespace together"
        );
        assert_eq!(
            whitespace_token_end::<true>(trailing, 0, ascii_only),
            2,
            "Qwen/OLMo/o200k newline alternatives win at end of input"
        );
        assert_eq!(
            pieces(FastCl100kPretokenizer::new(trailing)),
            [trailing.as_slice()]
        );
        assert_eq!(
            pieces(FastQwen2Pretokenizer::new(trailing)),
            [&trailing[..2], &trailing[2..]]
        );
    }

    #[test]
    fn shared_contraction_matcher_covers_ascii_case_and_long_s() {
        for token in ["'s", "'T", "'re", "'VE", "'m", "'Ll", "'d"] {
            assert_eq!(contraction_end(token.as_bytes(), 0), Some(token.len()));
        }
        assert_eq!(contraction_end(b"'\xC5\xBF", 0), Some(3));
        assert_eq!(contraction_end(b"'x", 0), None);
        assert_eq!(contraction_end(b"word", 0), None);

        assert_eq!(
            pieces(FastCl100kPretokenizer::new(b"we'RE")),
            [b"we".as_slice(), b"'RE".as_slice()]
        );
        assert_eq!(
            pieces(FastO200kPretokenizer::new(b"we'RE")),
            [b"we'RE".as_slice()]
        );
    }
}
