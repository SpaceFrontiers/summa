//! Fast pretokenizer for the GPT-2 (r50k_base) regex:
//! `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!
//! On aarch64 (NEON) and x86_64 with AVX-512 or AVX2 (runtime-detected)
//! the iterator runs a simdjson-style mask scanner: 64-byte
//! batches are classified with SIMD into per-byte u64 class masks, the
//! token-boundary bits are derived with shifted-mask algebra in scalar
//! registers (log step 17; the original vector-register algebra of step
//! 15 measured the same and was retired for the simpler form), and the
//! walker pops one bit per token — no per-token dispatch branches, which
//! sidesteps the ~8 cy/token branch-miss floor of the scalar scanner
//! (log step 13). Apostrophes get a contraction bit-fixup; batches with
//! any non-ASCII (~21% of OWT) take `extended_masks`, which classifies
//! every unicode char with the packed table so it joins the same
//! algebra. Chars straddling a batch edge are resolved with lookahead
//! and a prev-char walk-back, so bad zones (scalar re-derivation) remain
//! only for edge-straddling whitespace, contractions at the batch edge,
//! and invalid UTF-8 — ~0.4% of batches. Measured 2,460-2,600 MB/s on
//! 1 GB OWT (pretokenize_profile, min-of-N interleaved; 2,132 at step
//! 15, 983 for the scalar scanner).
//!
//! The scalar path (`advance_pos`, SWAR letter runs + arithmetic
//! predicates) remains the reference implementation, the no-SIMD
//! fallback, and the executor for bad zones and buffer tails.
//!
//! `advance_pos` is a pure free function (`(bytes, pos) -> end`) rather than
//! a `&mut self` method: keeping the cursor in a register instead of writing
//! `self.pos` at every scan step shortens the per-token dependency chain and
//! is worth ~30% throughput. Non-ASCII characters are classified with one
//! packed-table load (`unicode::class_of`) on a hand-rolled UTF-8 decode.
//!
//! A windowed multi-cursor variant (finding guaranteed boundaries and running
//! 2-4 independent `advance_pos` chains interleaved, queueing token ends) was
//! benchmarked at 0.80-0.95x of this streaming version: the queue traffic and
//! interleaved branch history cost more than the extra ILP recovers.

use super::mask::{self, MaskScheme, MaskState};
use super::{
    decode_cp, is_ascii_ws, is_digit, is_letter, scan_digits_from, scan_letters_from,
    scan_other_from,
};
use crate::pretokenize::unicode::{self, CharClass};

// -----------------------------------------------------------------------
// FastR50kPretokenizer
// -----------------------------------------------------------------------

/// Boundary and bad-zone bitmasks for `bytes[scan..scan+64]` (requires
/// `scan + 64 <= bytes.len()`). Bit `k` of `usable` = a trustworthy token
/// start at `scan + k`; `bad` marks bytes whose boundaries must be
/// re-derived by `advance_pos`, and no token may be emitted across an
/// unresolved bad zone.
///
/// NEON classifies the ASCII classes (letter, digit, space, whitespace:
/// 4 movemasks; apostrophe and non-ASCII behind horizontal any-tests)
/// and the boundary bits come from u64 shifted-mask algebra in scalar
/// registers, as in `cl100k_family::batch_masks`. Batches with any
/// non-ASCII byte (~21% on OWT, mostly curly quotes) take
/// [`extended_masks`]. Inlining that path here measured 0.98x, and
/// `#[inline(never)]` on this whole function 0.94x — the split keeps the
/// walker's register allocation clean (step 15's lesson) while the hot
/// ASCII algebra stays inline.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
    use std::arch::aarch64::*;
    let len = bytes.len();
    if scan + 70 > len {
        // Not enough lookahead for the batch-edge char classification
        // (up to a 4-byte char starting at scan + 66); scalar batch.
        return (0, u64::MAX);
    }
    unsafe {
        let p = bytes.as_ptr().add(scan);
        let zero = vdupq_n_u8(0);
        let mut lv = [zero; 4];
        let mut dv = [zero; 4];
        let mut sv = [zero; 4];
        let mut wsv = [zero; 4];
        let mut hiv = [zero; 4];
        let mut apv = [zero; 4];
        for i in 0..4 {
            let v = vld1q_u8(p.add(16 * i));
            let lowered = vorrq_u8(v, vdupq_n_u8(0x20));
            lv[i] = vcleq_u8(vsubq_u8(lowered, vdupq_n_u8(b'a')), vdupq_n_u8(25));
            dv[i] = vcleq_u8(vsubq_u8(v, vdupq_n_u8(b'0')), vdupq_n_u8(9));
            sv[i] = vceqq_u8(v, vdupq_n_u8(b' '));
            wsv[i] = vorrq_u8(sv[i], vcleq_u8(vsubq_u8(v, vdupq_n_u8(9)), vdupq_n_u8(4)));
            hiv[i] = vcltzq_s8(vreinterpretq_s8_u8(v));
            apv[i] = vceqq_u8(v, vdupq_n_u8(b'\''));
        }

        let lb = mask::movemask64(lv[0], lv[1], lv[2], lv[3]);
        let db = mask::movemask64(dv[0], dv[1], dv[2], dv[3]);
        let s64 = mask::movemask64(sv[0], sv[1], sv[2], sv[3]);
        let wsa = mask::movemask64(wsv[0], wsv[1], wsv[2], wsv[3]);
        // Apostrophes only matter for the contraction fixup below.
        let ap_any = vorrq_u8(vorrq_u8(apv[0], apv[1]), vorrq_u8(apv[2], apv[3]));
        let ap64 = if vmaxvq_u8(ap_any) != 0 {
            mask::movemask64(apv[0], apv[1], apv[2], apv[3])
        } else {
            0
        };

        // Any non-ASCII byte routes to the extended classifier, which
        // reuses the ASCII masks computed above.
        let hi_any = vorrq_u8(vorrq_u8(hiv[0], hiv[1]), vorrq_u8(hiv[2], hiv[3]));
        if vmaxvq_u8(hi_any) != 0 {
            let hi64 = mask::movemask64(hiv[0], hiv[1], hiv[2], hiv[3]);
            return extended_masks(bytes, scan, lb, db, s64, wsa, hi64, ap64);
        }

        ascii_batch_algebra(bytes, scan, lb, db, s64, wsa, ap64)
    }
}

/// x86 counterpart of the NEON `batch_masks`, monomorphized on the SIMD
/// tier: the classification is [`mask::ascii_masks_avx512`] (one 64-byte
/// load and one k-register compare per class) or
/// [`mask::ascii_masks_avx2`] (two 32-byte loads, compare + vpmovmskb per
/// class); the boundary algebra and the extended (non-ASCII) path are the
/// same shared scalar code. `#[inline(always)]` (with no `target_feature`
/// of its own) so the body fuses into whichever feature region calls it —
/// the tier-monomorphized fill wrappers
/// (`MaskState::fill_spans_two_phase_{avx512,avx2}_crc`), where LLVM's
/// cost model declined to inline the previous `#[target_feature]` form
/// and left a call per 64-byte batch, or the runtime-dispatched
/// `MaskScheme::batch_masks` that `next_span` uses.
///
/// # Safety
///
/// The selected tier must have been runtime-detected
/// ([`mask::avx512_scanner_available`] /
/// [`mask::avx2_scanner_available`]).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
    let len = bytes.len();
    if scan + 70 > len {
        // Not enough lookahead for the batch-edge char classification
        // (up to a 4-byte char starting at scan + 66); scalar batch.
        return (0, u64::MAX);
    }
    let am = if AVX512 {
        // SAFETY: the caller detected the AVX-512 tier (fn contract).
        unsafe { mask::ascii_masks_avx512(bytes, scan) }
    } else {
        // SAFETY: the caller detected the AVX2 tier (fn contract).
        unsafe { mask::ascii_masks_avx2(bytes, scan) }
    };
    let wsa = am.s | am.wt | am.n;
    if am.hi != 0 {
        // SAFETY: both detected tiers include the BMI1/BMI2/LZCNT/POPCNT
        // bit features `extended_masks` re-declares (fn contract).
        return unsafe { extended_masks(bytes, scan, am.l, am.d, am.s, wsa, am.hi, am.ap) };
    }
    ascii_batch_algebra(bytes, scan, am.l, am.d, am.s, wsa, am.ap)
}

/// Pure-ASCII boundary algebra shared by the NEON and AVX-512 batch
/// classifiers (the batch has no non-ASCII byte; `wsa` = all ASCII
/// whitespace). Everything here is platform-independent u64 bit math.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn ascii_batch_algebra(
    bytes: &[u8],
    scan: usize,
    lb: u64,
    db: u64,
    s64: u64,
    wsa: u64,
    ap64: u64,
) -> (u64, u64) {
    let ob = !(lb | db | wsa); // hi == 0 on this path

    // Bit-0 carries from the char before the batch. This batch is
    // pure ASCII, so a multi-byte prev char always ends exactly at
    // the boundary and the walk-back gives true carries — no bad
    // zone.
    let (pl, pd, ps, pws, po) = if scan == 0 {
        (0, 0, 0, 0, 0)
    } else {
        carries_at(bytes, scan)
    };

    let cont_same = (lb & ((lb << 1) | pl)) | (db & ((db << 1) | pd)) | (ob & ((ob << 1) | po));
    let after_sp = (s64 << 1) | ps;
    let nb = !wsa & !cont_same & !after_sp;

    // Ws-run split (`\s+(?!\S)`); bit 63 needs the real lookahead
    // char. The ASCII case is branchless — "is byte 63 ws" is a
    // ~20% coin flip on natural text, so testing it costs a
    // mispredict every few batches. Only a non-ASCII lookahead
    // byte (rare) branches, for the table-backed ws check.
    let mut split_ok = wsa & (!wsa >> 1); // bit 63: shifted-in 0
    let nb64 = bytes[scan + 64]; // in bounds: scan + 70 <= len
    if nb64 < 0x80 {
        split_ok |= (u64::from(!is_ascii_ws(nb64)) << 63) & wsa;
    } else if wsa >> 63 != 0
        // SAFETY: this classifier's scan + 70 <= len batch guard puts the
        // decode at scan + 64 in bounds (needs scan + 68 <= len).
        && unsafe { mask::nn_at_full(bytes, scan + 64) }
    {
        split_ok |= 1 << 63;
    }
    let pwsb = (wsa << 1) | pws;
    let wsboundary = wsa & (!pwsb | split_ok);
    let mut boundary = nb | wsboundary;

    let mut bad = 0u64;

    // Contraction fixup (see extended_masks for the rules).
    if ap64 != 0 {
        let mut cand = ap64 & boundary;
        while cand != 0 {
            let i = cand.trailing_zeros() as usize;
            cand &= cand - 1;
            if i >= 61 {
                bad |= u64::MAX << i;
                break;
            }
            let k = match bytes[scan + i + 1] {
                b's' | b'd' | b'm' | b't' => 2,
                b'l' if bytes[scan + i + 2] == b'l' => 3,
                b'v' if bytes[scan + i + 2] == b'e' => 3,
                b'r' if bytes[scan + i + 2] == b'e' => 3,
                _ => 0,
            };
            if k != 0 {
                boundary &= !(1u64 << (i + 1));
                boundary |= 1u64 << (i + k);
            }
        }
    }
    (boundary & !bad, bad)
}

/// Slow(er) path for batches containing non-ASCII: every unicode char in
/// (or straddling into/out of) the batch is classified with the packed
/// table via [`mask::classify_uni_chars`] — the same lookup the scalar
/// path would do — and joins the per-byte effective class masks, so
/// byte-adjacency == char-adjacency and the u64 boundary algebra applies
/// unchanged. Takes the ASCII class masks the caller already computed
/// (`ws64` = all ASCII whitespace).
///
/// Bad zones remain only for whitespace chars straddling a batch edge
/// (their `\s+(?!\S)` bookkeeping crosses the boundary), stray
/// continuation bytes (invalid UTF-8), and contractions at the batch
/// edge. An earlier version pattern-matched common leads with ~95 vector
/// ops (`mask::unicode_leads`) before falling back to per-char bad
/// zones; the direct table loop (typical hi batch: 1-3 unicode chars,
/// table hot in cache) plus edge-char resolution was worth ~13% end to
/// end. `#[inline(never)]`: inlining this into the walker wrecks the
/// clean path's register allocation (step 15).
///
/// On x86_64 the `target_feature` re-declaration keeps the bit-scan
/// loops on tzcnt/lzcnt/blsr in a baseline (non-native) build — this
/// function is out-of-line, so without it the ~21% of OWT batches
/// landing here would compile against baseline x86-64 even though every
/// caller is a SIMD batch classifier. Only the bit features are enabled
/// (not the callers' vector features): both the AVX-512 and AVX2 tiers
/// call this, so it must never emit instructions beyond the AVX2 tier's
/// set. Measured neutral on the OWT mask-compute diagnostic
/// (ASCII-dominated); kept for codegen parity on non-ASCII-heavy corpora
/// where this path dominates.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "bmi1,bmi2,lzcnt,popcnt")
)]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn extended_masks(
    bytes: &[u8],
    scan: usize,
    l64: u64,
    d64: u64,
    s64: u64,
    ws64: u64,
    hi64: u64,
    ap64: u64,
) -> (u64, u64) {
    let wsa = ws64;

    // Class-table LazyLock resolved once; the per-char classify below is
    // then a bare slice index.
    let ct = unicode::ClassTable::get();
    let class = move |cp| ct.class_of(cp);

    // Bit-0 carries via the prev-char walk-back; a char straddling into
    // this batch claims its continuation bytes with its class. Without
    // this, every batch following a unicode char became a bad zone at
    // bit 0, and a bad zone costs ~800 cycles in walker re-entries and
    // cold scalar gaps.
    let mut claim = mask::UniClasses::default();
    let (pl, pd, ps, pws, po) = if scan == 0 {
        (0, 0, 0, 0, 0)
    } else if bytes[scan - 1] < 0x80 {
        carries_at(bytes, scan)
    } else {
        // SAFETY: scan > 0 on this branch, and the classifier's
        // scan + 70 <= len batch guard covers pos + 3 <= len.
        let (cls, _lead, end) = unsafe { mask::char_through(bytes, scan, class) };
        let chm = if end > scan {
            (1u64 << (end - scan)) - 1
        } else {
            0
        };
        claim.cont = chm;
        match cls {
            CharClass::Letter => {
                claim.l = chm;
                (1, 0, 0, 0, 0)
            }
            CharClass::Number => {
                claim.n = chm;
                (0, 1, 0, 0, 0)
            }
            CharClass::Other => {
                claim.o = chm;
                (0, 0, 0, 0, 1)
            }
            CharClass::Whitespace => {
                // A ws char straddling in defers to the scalar path (its
                // run-split bookkeeping needs the pre-batch extent) but
                // still marks its true class for neighbors' algebra.
                claim.ws = chm;
                claim.resid = chm;
                (0, 0, u64::from(bytes[scan - 1] == b' '), 1, 0)
            }
        }
    };

    // SAFETY: this classifier's scan + 70 <= len batch guard is exactly
    // `classify_uni_chars`' contract.
    let uni =
        unsafe { mask::classify_uni_chars::<true, false>(bytes, scan, hi64 & !claim.cont, class) };

    // Effective per-byte classes: every byte of a classified char carries
    // the char's class, so the same algebra as the pure-ASCII path
    // applies.
    let lb = l64 | claim.l | uni.l;
    let db = d64 | claim.n | uni.n;
    let wsb = wsa | claim.ws | uni.ws;
    let ob = !(l64 | d64 | wsa | hi64) | claim.o | uni.o;
    let contm = claim.cont | uni.cont;
    let resid = claim.resid | uni.resid;

    let cont_same = (lb & ((lb << 1) | pl)) | (db & ((db << 1) | pd)) | (ob & ((ob << 1) | po));
    let after_sp = (s64 << 1) | ps;
    let nb = !wsb & !cont_same & !after_sp & !contm;

    // Ws-run split: char-length-aware "followed by non-ws" test. All ws
    // chars whose lookahead crosses the batch edge look at byte 64: an
    // ASCII ws at 63, a 2-byte ws led at 62, a 3-byte ws led at 61
    // (later leads straddle out and are already bad zones). The ASCII
    // case is branchless as in the fast path; multi-byte edge leads are
    // rare enough to branch.
    let nn = !wsb;
    let mut split_ok = (wsa & (nn >> 1)) | (uni.w2 & (nn >> 2)) | (uni.w3 & (nn >> 3));
    let ws_leads = wsa | uni.w2 | uni.w3;
    let edge_mb = (uni.w2 & (1 << 62)) | (uni.w3 & (1 << 61));
    let nb64 = bytes[scan + 64]; // in bounds: scan + 70 <= len
    if nb64 < 0x80 && edge_mb == 0 {
        split_ok = (split_ok & !(1 << 63)) | ((u64::from(!is_ascii_ws(nb64)) << 63) & wsa);
    } else {
        let edge = edge_mb | ((1 << 63) & wsa);
        if edge != 0 {
            // SAFETY: this classifier's scan + 70 <= len batch guard puts
            // the decode at scan + 64 in bounds (needs scan + 68 <= len).
            if unsafe { mask::nn_at_full(bytes, scan + 64) } {
                split_ok |= edge;
            } else {
                split_ok &= !edge;
            }
        }
    }
    let pwsb = (wsb << 1) | pws;
    let wsboundary = ws_leads & (!pwsb | split_ok);
    let mut boundary = nb | wsboundary;

    let mut bad = resid | resid << 1 | resid >> 1;

    // Contraction fixup: an apostrophe at a token start absorbs an
    // s/d/m/t/ll/ve/re suffix. One that could reach past bit 63 defers
    // to the scalar path (the next batch cannot see the moved boundary).
    let mut cand = ap64 & boundary & !bad;
    while cand != 0 {
        let i = cand.trailing_zeros() as usize;
        cand &= cand - 1;
        if i >= 61 {
            bad |= u64::MAX << i;
            break;
        }
        let k = match bytes[scan + i + 1] {
            b's' | b'd' | b'm' | b't' => 2,
            b'l' if bytes[scan + i + 2] == b'l' => 3,
            b'v' if bytes[scan + i + 2] == b'e' => 3,
            b'r' if bytes[scan + i + 2] == b'e' => 3,
            _ => 0,
        };
        if k != 0 {
            boundary &= !(1u64 << (i + 1));
            boundary |= 1u64 << (i + k);
        }
    }

    (boundary & !bad, bad)
}

/// `(pl, pd, ps, pws, po)` boundary carries for the char ending at
/// `scan - 1` (`scan > 0`), multi-byte aware via [`mask::char_through`].
/// `ps` (the ` ?` absorb) is ASCII 0x20 only. The ASCII case (almost
/// every call) is branchless — a class if-chain here is a per-batch
/// mispredict on natural text. Callers are the batch classifiers, whose
/// `scan + 70 <= len` guard covers the walk-back decode.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn carries_at(bytes: &[u8], scan: usize) -> (u64, u64, u64, u64, u64) {
    let b = bytes[scan - 1];
    if b < 0x80 {
        let (l, d, w) = (is_letter(b), is_digit(b), is_ascii_ws(b));
        let bit = |c: bool| u64::from(c);
        return (bit(l), bit(d), bit(b == b' '), bit(w), bit(!l && !d && !w));
    }
    // SAFETY: scan > 0 (this fn's caller contract), and the calling batch
    // classifier's scan + 70 <= len guard covers pos + 3 <= len.
    match unsafe { mask::char_through(bytes, scan, unicode::class_of) }.0 {
        CharClass::Letter => (1, 0, 0, 0, 0),
        CharClass::Number => (0, 1, 0, 0, 0),
        CharClass::Whitespace => (0, 0, 0, 1, 0),
        CharClass::Other => (0, 0, 0, 0, 1),
    }
}
pub(crate) struct R50kScheme;

impl MaskScheme for R50kScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance_pos(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        batch_masks(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { batch_masks_x86::<AVX512>(bytes, scan) }
    }
}

/// With SIMD support (aarch64 NEON, or x86_64 AVX-512/AVX2 detected at
/// runtime), iteration runs on the mask scanner above via the shared
/// [`MaskState`] batch walker; elsewhere every token takes `advance_pos`.
pub struct FastR50kPretokenizer<'a> {
    bytes: &'a [u8],
    state: MaskState,
}

super::impl_mask_pretokenizer!(FastR50kPretokenizer, R50kScheme);

/// Advance past one token starting at `start`; returns the token's end.
/// `start` must be < `bytes.len()` and a valid token start.
/// Uses direct comparison chains instead of LUT + jump table to avoid
/// GOT indirection and improve branch prediction on common patterns.
///
/// Byte loads here look redundant but are effectively free: their addresses
/// depend only on `start`, so they issue in parallel under speculation. A
/// variant that did one u64 load and extracted bytes/scanned letters
/// in-register measured 0.84x — the shifts serialize after the load, while
/// independent L1 loads don't.
#[inline(always)]
fn advance_pos(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let b0 = unsafe { *bytes.get_unchecked(start) };

    // Bare ASCII letter start (~5% of OWT tokens; most words carry a space)
    if is_letter(b0) {
        return scan_letters_from(bytes, start + 1);
    }

    // Hot path: space before content (~78% of tokens, ~75% space+letters)
    if b0 == b' ' {
        if start + 1 < len {
            let b1 = unsafe { *bytes.get_unchecked(start + 1) };
            if is_letter(b1) {
                return scan_letters_from(bytes, start + 2);
            }
            if is_digit(b1) {
                return scan_digits_from(bytes, start + 2);
            }
            if b1 >= 0x80 {
                let (cp, l) = unsafe { decode_cp(bytes, start + 1) };
                let p = start + 1 + l;
                return match unicode::class_of(cp) {
                    CharClass::Letter => scan_letters_from(bytes, p),
                    CharClass::Number => scan_digits_from(bytes, p),
                    CharClass::Whitespace => advance_ws(bytes, p, start),
                    CharClass::Other => scan_other_from(bytes, p),
                };
            }
            if is_ascii_ws(b1) {
                return advance_ws(bytes, start + 1, start);
            }
            return scan_other_from(bytes, start + 2);
        }
        return start + 1;
    }

    // Non-ASCII
    if b0 >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, start) };
        let p = start + l;
        return match unicode::class_of(cp) {
            CharClass::Letter => scan_letters_from(bytes, p),
            CharClass::Number => scan_digits_from(bytes, p),
            CharClass::Whitespace => advance_ws(bytes, p, start),
            CharClass::Other => scan_other_from(bytes, p),
        };
    }

    // Digit
    if is_digit(b0) {
        return scan_digits_from(bytes, start + 1);
    }

    // Apostrophe / contraction
    if b0 == b'\'' {
        match bytes.get(start + 1) {
            Some(b's' | b'd' | b'm' | b't') => return start + 2,
            Some(b'l') if bytes.get(start + 2) == Some(&b'l') => return start + 3,
            Some(b'v') if bytes.get(start + 2) == Some(&b'e') => return start + 3,
            Some(b'r') if bytes.get(start + 2) == Some(&b'e') => return start + 3,
            _ => return scan_other_from(bytes, start + 1),
        }
    }

    // Whitespace (tab, newline, etc.)
    if b0.wrapping_sub(9) < 5 {
        return advance_ws(bytes, start + 1, start);
    }

    // Other (punctuation, symbols)
    scan_other_from(bytes, start + 1)
}

/// Advance through whitespace. `scan_pos` is where to continue scanning,
/// `token_start` is where the token began (for the split-off-last-char logic).
#[inline(always)]
fn advance_ws(bytes: &[u8], scan_pos: usize, token_start: usize) -> usize {
    let len = bytes.len();
    let mut p = scan_pos;
    while p < len {
        let b = unsafe { *bytes.get_unchecked(p) };
        if is_ascii_ws(b) {
            p += 1;
        } else if b >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == CharClass::Whitespace {
                p += l;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if p < len {
        let ws_bytes = p - token_start;
        if ws_bytes >= 2 {
            let mut last = p - 1;
            while last > token_start && unsafe { *bytes.get_unchecked(last) } & 0xC0 == 0x80 {
                last -= 1;
            }
            if last > token_start {
                return last;
            }
        }
    }
    p
}
