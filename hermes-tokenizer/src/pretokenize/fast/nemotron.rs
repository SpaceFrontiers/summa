//! Fast pretokenizer for the Nemotron-3 regex (nvidia Nemotron-3 family):
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! The o200k scheme without contraction suffixes and with single-char
//! `\p{N}` digit tokens. See `o200k_family` (`CONTRACTIONS = false`,
//! `DIGITS3 = false`).

use super::mask::{MaskScheme, MaskState};
use super::o200k_family;
use crate::pretokenize::Pretoken;

pub(crate) struct NemotronScheme;

impl MaskScheme for NemotronScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        o200k_family::advance_pos::<false, false, true, false>(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        o200k_family::batch_masks::<false, false, true, false>(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { o200k_family::batch_masks_x86::<AVX512, false, false, true, false>(bytes, scan) }
    }
}

/// With SIMD support (aarch64 NEON, or x86_64 AVX-512/AVX2 detected at
/// runtime), iteration runs the shared o200k-family mask scanner (see
/// `o200k_family::batch_masks`); elsewhere every token takes the scalar
/// `advance_pos`.
pub struct FastNemotronPretokenizer<'a> {
    bytes: &'a [u8],
    state: MaskState,
}

impl<'a> FastNemotronPretokenizer<'a> {
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

impl<'a> Iterator for FastNemotronPretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.state.next_span::<NemotronScheme>(self.bytes)?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

super::impl_mask_pretoken_spans!(FastNemotronPretokenizer, NemotronScheme);
