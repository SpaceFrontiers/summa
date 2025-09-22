//! Shared mask-scanner boundary algebra for the cl100k regex family:
//! cl100k, olmo3, qwen2, and qwen3.5. Their patterns share the shape
//!
//! `'(?i:contractions)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3} or \p{N}|
//!  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! and differ only in the digit-group size (`digits3`), end-of-input
//! whitespace behavior (always in the scalar tail, never in a batch), and
//! which unicode classes join runs (`\p{M}`: punct-class for
//! cl100k/olmo3/qwen2, letter-class for qwen3.5 — expressed by the
//! codepoint classifier each scheme passes in).
//!
//! Boundary rules, derived in `pretokenizer_optimization_log.md` step 16:
//! - A letter starts a token unless it continues a letter run, follows
//!   space/tab-class whitespace (which always sits at a boundary before a
//!   non-ws char and absorbs one following letter run via the
//!   `[^\r\n\p{L}\p{N}]?` prefix), or follows a punct char that is itself
//!   at a boundary — i.e. whose own predecessor is neither punct nor a
//!   space (a two-chars-back test, made char-aware for multi-byte chars).
//! - Digits split every 1 or 3 chars from each run start and never absorb
//!   a preceding space.
//! - A punct char starts a token unless it continues a punct run or
//!   follows a space (` ?[^\s\p{L}\p{N}]+`).
//! - Newlines directly after a punct run are absorbed (`[\r\n]*`).
//! - A whitespace run containing newlines emits one token through its LAST
//!   newline (`\s*[\r\n]+` / `\s*[\r\n]`), then the r50k-style tail rules;
//!   NL-free runs split before their last char when followed by non-ws
//!   (`\s+(?!\S)`). A run touching the batch end resolves in-batch when
//!   the char at byte 64 is non-ws (the run demonstrably ends at the
//!   edge — ~16% of OWT batches end in a single space, and deferring
//!   them was the family's dominant cost, log step 19); a run actually
//!   crossing the edge is deferred to the scalar path (its "last
//!   newline" may lie in a later batch).

use super::is_ascii_ws;
use super::mask::{self, AsciiMasks};
use crate::pretokenize::unicode::CharClass;

/// Smear `seed` upward (toward higher bits) through contiguous set bits of
/// `within`, in log steps.
#[inline(always)]
fn smear_up(seed: u64, within: u64) -> u64 {
    let mut a = seed;
    let mut m = within;
    let mut sh = 1u32;
    while sh < 64 {
        a |= (a << sh) & m;
        m &= m << sh;
        sh <<= 1;
    }
    a
}

/// Boundary carries from the two chars before the batch: P1 ends at
/// `scan - 1`, P2 is the one before it (the two-chars-back absorb test).
#[derive(Clone, Copy, Default)]
struct Carries {
    /// P1 is a letter / space (0x20) / non-newline non-space ws / punct /
    /// any ws / digit.
    pl: u64,
    ps: u64,
    pwt: u64,
    po: u64,
    pws: u64,
    pd: u64,
    /// P2 is punct-or-space, for a char lead at bit 0 (P1 entirely before
    /// the batch).
    c2_os: u64,
    /// Same test positioned at the first lead AFTER a P1 that straddles
    /// into the batch (P1's own prev is then P2).
    b2b_in: u64,
}

/// Pure-ASCII carries (hot, branchless). Requires `scan > 0` and
/// `bytes[scan-1] < 0x80` (and `bytes[scan-2] < 0x80` when present).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn ascii_carries(bytes: &[u8], scan: usize) -> Carries {
    let b = bytes[scan - 1];
    let bit = |c: bool| u64::from(c);
    let (l, d, w) = (super::is_letter(b), super::is_digit(b), is_ascii_ws(b));
    let n = b == b'\r' || b == b'\n';
    let c2_os = if scan >= 2 {
        let b2 = bytes[scan - 2];
        bit(b2 == b' ' || (!super::is_letter(b2) && !super::is_digit(b2) && !is_ascii_ws(b2)))
    } else {
        0
    };
    Carries {
        pl: bit(l),
        ps: bit(b == b' '),
        pwt: bit(w && !n && b != b' '),
        po: bit(!l && !d && !w),
        pws: bit(w),
        pd: bit(d),
        c2_os,
        b2b_in: 0,
    }
}

/// `(usable, bad)` for `bytes[scan..scan+64]` under the cl100k-family
/// rules. `digits3`: `\p{N}{1,3}` (cl100k/olmo3) vs `\p{N}` (qwen2/3.5).
/// `class`: the scheme's codepoint classifier — `unicode::class_of` for
/// cl100k/olmo3/qwen2 (marks are punct-class), or
/// `unicode::class_of_marks_join` for qwen3.5 (`\p{M}` joins letter
/// runs).
///
/// Structured like the r50k scanner (log step 17): NEON classifies the
/// ASCII classes with 5 movemasks (letter, digit, space, whitespace,
/// newline; `wt` is derived in bit algebra, apostrophe and non-ASCII sit
/// behind horizontal any-tests) and the pure-ASCII boundary algebra stays
/// inline; batches with any non-ASCII byte in or just before them take
/// [`family_extended_masks`], `#[inline(never)]` so the hot path's
/// register allocation stays clean.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn batch_masks(
    bytes: &[u8],
    scan: usize,
    digits3: bool,
    class: impl Fn(u32) -> CharClass + Copy,
) -> (u64, u64) {
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
        let mut nv = [zero; 4];
        let mut hiv = [zero; 4];
        let mut apv = [zero; 4];
        for i in 0..4 {
            let v = vld1q_u8(p.add(16 * i));
            let lowered = vorrq_u8(v, vdupq_n_u8(0x20));
            lv[i] = vcleq_u8(vsubq_u8(lowered, vdupq_n_u8(b'a')), vdupq_n_u8(25));
            dv[i] = vcleq_u8(vsubq_u8(v, vdupq_n_u8(b'0')), vdupq_n_u8(9));
            sv[i] = vceqq_u8(v, vdupq_n_u8(b' '));
            wsv[i] = vorrq_u8(sv[i], vcleq_u8(vsubq_u8(v, vdupq_n_u8(9)), vdupq_n_u8(4)));
            nv[i] = vorrq_u8(
                vceqq_u8(v, vdupq_n_u8(b'\r')),
                vceqq_u8(v, vdupq_n_u8(b'\n')),
            );
            hiv[i] = vcltzq_s8(vreinterpretq_s8_u8(v));
            apv[i] = vceqq_u8(v, vdupq_n_u8(b'\''));
        }
        let l64 = mask::movemask64(lv[0], lv[1], lv[2], lv[3]);
        let d64 = mask::movemask64(dv[0], dv[1], dv[2], dv[3]);
        let s64 = mask::movemask64(sv[0], sv[1], sv[2], sv[3]);
        let wsa = mask::movemask64(wsv[0], wsv[1], wsv[2], wsv[3]);
        let n64 = mask::movemask64(nv[0], nv[1], nv[2], nv[3]);

        // Apostrophes only matter for the contraction fixup.
        let ap_any = vorrq_u8(vorrq_u8(apv[0], apv[1]), vorrq_u8(apv[2], apv[3]));
        let ap64 = if vmaxvq_u8(ap_any) != 0 {
            mask::movemask64(apv[0], apv[1], apv[2], apv[3])
        } else {
            0
        };

        let am = mask::AsciiMasks {
            l: l64,
            d: d64,
            s: s64,
            wt: wsa & !s64 & !n64,
            n: n64,
            hi: 0,
            ap: ap64,
        };

        // Any non-ASCII byte in the batch — or within the two carry bytes
        // before it — routes to the extended classifier.
        let hi_any = vorrq_u8(vorrq_u8(hiv[0], hiv[1]), vorrq_u8(hiv[2], hiv[3]));
        if vmaxvq_u8(hi_any) != 0
            || (scan >= 1 && bytes[scan - 1] >= 0x80)
            || (scan >= 2 && bytes[scan - 2] >= 0x80)
        {
            let mut am = am;
            am.hi = mask::movemask64(hiv[0], hiv[1], hiv[2], hiv[3]);
            return family_extended_masks(bytes, scan, digits3, class, am);
        }

        let cr = if scan == 0 {
            Carries::default()
        } else {
            ascii_carries(bytes, scan)
        };
        family_algebra(bytes, scan, digits3, am, cr, mask::UniClasses::default())
    }
}

/// x86-64 front-end for the family schemes: same contract as the NEON
/// `batch_masks` above, monomorphized on the SIMD tier (see
/// `MaskScheme::batch_masks_x86`, whose provided `batch_masks` supplies
/// the runtime-dispatched form). The classification is
/// [`mask::ascii_masks_avx512`] (one 64-byte load and one k-register
/// compare per class) or [`mask::ascii_masks_avx2`] (two 32-byte loads,
/// compare + vpmovmskb per class); the boundary algebra and the extended
/// (non-ASCII) path are the shared scalar code. `#[inline(always)]` (with
/// no `target_feature` of its own) so the body fuses into whichever
/// feature region calls it — LLVM's cost model declined to inline the
/// previous `#[target_feature]` form into the tier-monomorphized fill
/// wrappers and left a call per 64-byte batch.
///
/// # Safety
///
/// The selected tier must have been runtime-detected
/// ([`mask::avx512_scanner_available`] /
/// [`mask::avx2_scanner_available`]).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn batch_masks_x86<const AVX512: bool>(
    bytes: &[u8],
    scan: usize,
    digits3: bool,
    class: impl Fn(u32) -> CharClass + Copy,
) -> (u64, u64) {
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

    // Any non-ASCII byte in the batch — or within the two carry bytes
    // before it — routes to the extended classifier. (`am.hi` is exact
    // and already computed, unlike NEON's lazily-movemasked variant.)
    if am.hi != 0
        || (scan >= 1 && bytes[scan - 1] >= 0x80)
        || (scan >= 2 && bytes[scan - 2] >= 0x80)
    {
        // SAFETY: both detected tiers include the BMI1/BMI2/LZCNT/POPCNT
        // bit features `family_extended_masks` re-declares (fn contract).
        return unsafe { family_extended_masks(bytes, scan, digits3, class, am) };
    }

    let cr = if scan == 0 {
        Carries::default()
    } else {
        ascii_carries(bytes, scan)
    };
    family_algebra(bytes, scan, digits3, am, cr, mask::UniClasses::default())
}

/// Slow(er) path for batches with non-ASCII in or just before them: the
/// carries walk back through multi-byte chars and classify with the
/// packed table, so a batch following unicode text gets true carries
/// instead of a bit-0 bad zone; every unicode char in the batch is
/// classified with the same table and joins the effective class masks
/// ([`mask::classify_uni_chars`]); then the shared boundary algebra
/// applies unchanged. Only number chars (their `\p{N}{1,3}` grouping is
/// char-counted, inexpressible in byte masks), whitespace straddling the
/// batch end, and stray continuation bytes stay bad zones.
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
fn family_extended_masks(
    bytes: &[u8],
    scan: usize,
    digits3: bool,
    class: impl Fn(u32) -> CharClass + Copy,
    am: mask::AsciiMasks,
) -> (u64, u64) {
    // A P1 straddling into the batch claims its continuation bytes with
    // its class; `b2b_in` is the two-back test for the char right after
    // it, whose predecessor chain starts before the batch.
    let mut cl = mask::UniClasses::default();
    let cr = if scan == 0 {
        Carries::default()
    } else if bytes[scan - 1] < 0x80 && (scan < 2 || bytes[scan - 2] < 0x80) {
        ascii_carries(bytes, scan)
    } else {
        // A multi-byte char within two bytes of the batch start.
        // SAFETY: scan > 0 on this branch, and the classifier's
        // scan + 70 <= len batch guard covers pos + 3 <= len.
        let (c1, j1, e1) = unsafe { mask::char_through(bytes, scan, class) };
        let pb = bytes[scan - 1];
        let chm = if e1 > scan {
            (1u64 << (e1 - scan)) - 1
        } else {
            0
        };
        cl.cont = chm;
        let c2v = if j1 == 0 {
            0
        } else {
            // SAFETY: j1 > 0 just checked, and j1 < scan keeps the decode
            // within the classifier's scan + 70 <= len batch guard.
            let c2c = unsafe { mask::char_through(bytes, j1, class) }.0;
            u64::from(bytes[j1 - 1] == b' ' || c2c == CharClass::Other)
        };
        let mut c = Carries::default();
        if e1 > scan {
            c.b2b_in = c2v << (e1 - scan);
        } else {
            c.c2_os = c2v;
        }
        c.pd = u64::from(c1 == CharClass::Number);
        match c1 {
            CharClass::Letter => {
                cl.l = chm;
                c.pl = 1;
            }
            // A digit P1 sets no letter/punct carries: `\p{N}` groups
            // restart at token boundaries. A P1 entirely before the batch
            // is covered by the `pd` seed (bit 0 is then an ASCII digit
            // when the run continues). A digit char STRADDLING into the
            // batch defeats that seed — bit 0 is its continuation byte,
            // not an ASCII digit — so its claimed bytes defer via resid
            // and the bad<<1 seed catches the following run. (Found by
            // the o200k-family port's differential fuzz: "٢1234" with the
            // ٢ split across a batch edge mis-phased `\p{N}{1,3}`.)
            CharClass::Number => {
                cl.n = chm;
                cl.resid |= chm;
            }
            CharClass::Other => {
                cl.o = chm;
                c.po = 1;
            }
            CharClass::Whitespace => {
                cl.ws = chm;
                if e1 > scan {
                    // Straddling-in ws: run bookkeeping crosses the edge.
                    cl.resid = chm;
                }
                c.ps = u64::from(pb == b' ');
                let nl = pb == b'\r' || pb == b'\n';
                c.pwt = u64::from(pb != b' ' && !nl);
                c.pws = 1;
            }
        }
        c
    };

    let mut uni = if am.hi != 0 {
        // SAFETY: this classifier's scan + 70 <= len batch guard is
        // exactly `classify_uni_chars`' contract.
        unsafe { mask::classify_uni_chars::<false, true>(bytes, scan, am.hi & !cl.cont, class) }
    } else {
        mask::UniClasses::default()
    };
    uni.l |= cl.l;
    uni.n |= cl.n;
    uni.o |= cl.o;
    uni.ws |= cl.ws;
    uni.cont |= cl.cont;
    uni.resid |= cl.resid;

    family_algebra(bytes, scan, digits3, am, cr, uni)
}

/// The scheme family's shared u64 boundary algebra over per-byte class
/// masks. `uni` is all-zero on the pure-ASCII path (the constant folds
/// away every unicode term); the extended path passes real class masks
/// with straddle-in claims already merged in.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn family_algebra(
    bytes: &[u8],
    scan: usize,
    digits3: bool,
    am: mask::AsciiMasks,
    cr: Carries,
    uni: mask::UniClasses,
) -> (u64, u64) {
    let Carries {
        pl,
        ps,
        pwt,
        po,
        pws,
        pd,
        c2_os,
        b2b_in,
    } = cr;
    let contm = uni.cont;
    let resid = uni.resid;

    // Effective per-byte classes: every byte of a classified char carries
    // the char's class, so byte-adjacency == char-adjacency.
    let lb = am.l | uni.l;
    let sb = am.s; // the ` ?` / prefix "space" is ASCII 0x20 only
    let wtb = am.wt | uni.ws;
    let ob = !(am.l | am.d | am.s | am.wt | am.n | am.hi) | uni.o;
    let ws_all = sb | wtb | am.n;

    // --- Letters: `[^\r\n\p{L}\p{N}]?\p{L}+` -------------------------------
    // B: "the char two back is punct or space" — evaluated at each char's
    // lead by shifting the prev-byte test C by the PREV char's length.
    let len1 = !(contm | uni.lead2 | uni.lead3 | uni.lead4);
    let c_test = ((ob | sb) << 1) | po | ps; // bit 0: byte scan-1 in O|S
    let b2back = ((c_test & len1) << 1)
        | ((c_test & uni.lead2) << 2)
        | ((c_test & uni.lead3) << 3)
        | ((c_test & uni.lead4) << 4)
        | c2_os // prev char entirely before the batch
        | b2b_in; // prev char straddles in; its own prev is P2
    let p_l = (lb << 1) | pl;
    let p_s = (sb << 1) | ps;
    let p_wt = (wtb << 1) | pwt;
    let p_o = (ob << 1) | po;
    let absorb = p_o & !b2back;
    let b_letters = lb & !contm & !p_l & !p_s & !p_wt & !absorb;

    // --- Digits: `\p{N}{1,3}` or `\p{N}` -----------------------------------
    // The run-split hop loop only runs when a run of 2+ digits exists.
    let b_digits = if digits3 && am.d & (am.d >> 1) != 0 {
        mask::digit_run_splits3(am.d)
    } else {
        am.d
    };

    // --- Punct: ` ?[^\s\p{L}\p{N}]+` ----------------------------------------
    let b_punct = ob & !contm & !p_o & !p_s;

    // --- Whitespace ---------------------------------------------------------
    // Newlines directly after a punct run are absorbed (`[\r\n]*`). The
    // smear only runs on a nonzero seed (most batches have no
    // punct-adjacent newline).
    let abs_seed = am.n & ((ob << 1) | po);
    let abs_n = if abs_seed == 0 {
        0
    } else {
        smear_up(abs_seed, am.n)
    };
    let ws_eff = ws_all & !abs_n;

    let mut bad = resid | resid << 1 | resid >> 1;

    // Byte-64 lookahead: is the char at the next batch's first byte
    // non-ws? Decides whether ws-like runs touching bit 63 resolve
    // in-batch — the dominant deferral cause before this existed (~16% of
    // OWT batches ended in a single space). Branchless for an ASCII byte
    // 64 as in the r50k scanner ("is the edge ws" is a ~20% coin flip on
    // natural text); only a non-ASCII byte 64 (rare) branches, for the
    // table-backed check. Guarded on `bad >> 63`: a ws char straddling
    // out makes byte 64 a continuation byte, not a lead (only reachable
    // with a non-ASCII byte 64).
    let nb64 = bytes[scan + 64]; // in bounds: scan + 70 <= len
    let nn64 = if nb64 < 0x80 {
        !is_ascii_ws(nb64)
    } else {
        // SAFETY: the same scan + 70 <= len batch guard puts the decode at
        // scan + 64 in bounds (needs scan + 68 <= len).
        bad >> 63 == 0 && unsafe { mask::nn_at_full(bytes, scan + 64) }
    };
    let nn64m = u64::from(nn64).wrapping_neg(); // all-ones when non-ws

    // A punct-absorbed newline run touching the batch end: if the char at
    // byte 64 is ws, the token may continue (another newline), and even
    // when it doesn't, the next batch cannot tell the absorbed `\n` before
    // its bit 0 from a ws-run `\n` — defer to the scalar path. If byte 64
    // is non-ws, the punct token ends exactly at the batch edge.
    if abs_n >> 63 != 0 && !nn64 {
        bad |= 1u64 << 63;
    }

    // A ws run touching the batch end resolves in-batch when byte 64's
    // char is non-ws (the run's last newline and its `(?!\S)` split are
    // then all visible; `nn64m` feeds the lookahead bits below).
    // Otherwise it defers: its last newline (and the `\s+$`-style
    // end-of-input rules) may lie beyond this batch.
    let nonws = !ws_eff;
    if ws_eff >> 63 != 0 && !nn64 {
        if nonws == 0 {
            return (0, u64::MAX); // whole batch one ws run
        }
        let h = 63 - nonws.leading_zeros(); // highest non-ws bit (< 63)
        bad |= u64::MAX << (h + 1);
    }

    // A digit run crossing the batch END needs no deferral: its in-batch
    // `\p{N}{1,3}` splits are phased from the run's in-batch start (a
    // continuation from before the batch is the `pd` case below), and
    // they are token starts no matter how far the run continues — the
    // NEXT batch defers its own leading run via its `pd` seed and the
    // scalar path resumes from the last in-batch split.

    // A digit run whose grouping phase did not start inside this batch is
    // deferred too: `digit_run_splits3` phases each run from its first
    // in-batch digit, which is wrong when the run continues from before
    // the batch (`pd`: the walker stays on the 64-byte grid across scalar
    // overruns, so a batch can begin mid-run) or follows a bad zone that
    // may hold digit-class chars (e.g. Arabic-Indic digits, kept out of
    // the mask because `\p{N}{1,3}` counts chars, not bytes — a latent
    // bug that predates the table classifier, caught when Arabic-Indic
    // digits joined the fuzz corpus).
    if digits3 {
        let seed = (am.d & (bad << 1)) | (am.d & pd);
        if seed != 0 {
            bad |= smear_up(seed, am.d);
        }
    }

    // Base rule (correct for NL-free runs; NL runs are overridden below):
    // run start, or split before the last char when followed by non-ws.
    let ws_leads1 = (am.s | am.wt | am.n) & ws_eff;
    let ws_leads = (ws_leads1 | uni.w2 | uni.w3) & !abs_n;
    let p_ws = (ws_eff << 1) | pws; // prev byte ws (any kind)
    // Last-char `(?!\S)` split: in-batch via shifted nonws; the run
    // touching bit 63 uses the byte-64 lookahead (`nn64m`). A 2-byte ws
    // led at 62 or 3-byte ws led at 61 ends at the edge too.
    let edge_last = (ws_leads1 & (1 << 63)) | (uni.w2 & (1 << 62)) | (uni.w3 & (1 << 61));
    let split_ok = (ws_leads1 & (nonws >> 1))
        | (uni.w2 & (nonws >> 2))
        | (uni.w3 & (nonws >> 3))
        | (edge_last & nn64m);
    let mut b_ws = ws_leads & (!p_ws | split_ok);

    // Override every run that contains a (non-absorbed) newline: one token
    // through the run's last newline, then r50k-style tail rules. (A
    // branchless formulation via a downward smear — add the tail-start
    // bit after each run's last newline, clear the split bit sitting on
    // it — measured 0.95x: the smear's serial chain runs every batch
    // while this loop is skipped or predicted on most.)
    let mut runs_n = am.n & ws_eff & !bad;
    while runs_n != 0 {
        let f = runs_n.trailing_zeros();
        let below_gap = nonws & ((1u64 << f) - 1);
        let a = if below_gap == 0 {
            0
        } else {
            64 - below_gap.leading_zeros()
        };
        // First non-ws above f, or 64 for a run ending exactly at the
        // batch edge (only reachable when `nn64`).
        let e = (nonws & (u64::MAX << f)).trailing_zeros();
        let run_mask = (u64::MAX << a) & !u64::MAX.unbounded_shl(e);
        b_ws &= !run_mask;
        // Run start. Bit-0-leading runs with prev-byte ws cannot contain a
        // newline (scalar resumes only after `\s*[\r\n]+` tokens), so `a`
        // is always a true run start here.
        b_ws |= 1u64 << a;
        let q = 63 - (am.n & run_mask).leading_zeros(); // last NL in run
        if (q + 1) < e {
            // Tail after the last newline: starts a token, and its last
            // char splits off before the following non-ws char.
            b_ws |= 1u64 << (q + 1);
            let tail = run_mask & (u64::MAX << (q + 1));
            let tail_leads = ws_leads & tail;
            b_ws |= 1u64 << (63 - tail_leads.leading_zeros());
        }
        runs_n &= !run_mask;
    }

    let mut boundary = b_letters | b_digits | b_punct | b_ws;

    // --- Contractions: `'(?i:[sdmt]|ll|ve|re)` ------------------------------
    // ('ſ — U+017F — is non-ASCII, so it already sits in a bad zone.)
    let mut cand = am.ap & boundary & !bad;
    while cand != 0 {
        let i = cand.trailing_zeros() as usize;
        cand &= cand - 1;
        if i >= 61 {
            bad |= u64::MAX << i;
            break;
        }
        let b1 = bytes[scan + i + 1];
        if b1 >= 0x80 {
            // `(?i:'s)` also matches 'ſ (U+017F folds to s). With the
            // table now classifying ſ as a letter instead of leaving it
            // in a bad zone, an apostrophe before ANY non-ASCII char
            // must defer to the scalar path explicitly.
            bad |= 0b111u64 << i;
            continue;
        }
        let k = match b1 | 0x20 {
            b's' | b'd' | b'm' | b't' => 2,
            b'l' if bytes[scan + i + 2] | 0x20 == b'l' => 3,
            b'v' if bytes[scan + i + 2] | 0x20 == b'e' => 3,
            b'r' if bytes[scan + i + 2] | 0x20 == b'e' => 3,
            _ => 0,
        };
        if k != 0 {
            boundary &= !(1u64 << (i + 1));
            boundary |= 1u64 << (i + k);
        }
    }

    (boundary & !bad, bad)
}
