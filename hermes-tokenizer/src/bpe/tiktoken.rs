#[cfg(target_arch = "aarch64")]
use crate::bpe::bpe_merge_symbols_short_neon;
use crate::bpe::pretoken_cache::ShortPretokenCache;
use crate::bpe::{
    ByteRemapping, MergeScratch, PairRankTable, SHORT_MERGE_MAX, bpe_merge_symbols_by_rank,
    bpe_merge_symbols_ranked, bpe_merge_symbols_ranked_slice, bpe_merge_symbols_short_scalar,
    bpe_merge_symbols_with_scratch, simple_bpe_merge,
};
use crate::pretokenize::{
    FastCl100kPretokenizer, FastDeepSeekV3Pretokenizer, FastOlmo3Pretokenizer,
    FastQwen2Pretokenizer, FastQwen35Pretokenizer, FastR50kPretokenizer, PRETOKEN_CHUNK, Pretoken,
    PretokenSpans, PretokenizerType, SpanBatch, pack_pretoken_key, pretoken_key_hash,
};
use crate::token::TokenId;
use anyhow::Result;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

/// Byte-level BPE tokenizer (tiktoken / GPT-2 style).
///
/// Initial symbols are individual bytes (0–255).  Merge priority is
/// determined by the merged token's vocab ID (lower = first), which
/// equals the merge rank for tiktoken vocabularies.
pub struct Tokenizer {
    // The model tables (merges, pair_ranks, vocab, vocab_inv) are immutable
    // after construction and shared across forks behind `Arc`: parallel
    // workers read the same few MB of tables instead of holding one deep
    // clone each, which keeps a single copy resident per cache/cluster on
    // the cold miss path and makes forking the tables O(1). The rare
    // mutation (`add_special_token`) goes through `Arc::make_mut`.
    pub(crate) merges: Arc<HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>>,
    /// Flat pair-rank tables replacing `merges` lookups on the miss path's
    /// merge loop; `None` for vocabularies whose IDs don't fit its packed
    /// keys (those keep probing `merges`).
    pair_ranks: Option<Arc<PairRankTable>>,
    /// Explicit merge priorities for rank-mapped vocabularies (fairseq
    /// heritage: RoBERTa/OPT/DeBERTa, whose vocab IDs are frequency-ordered
    /// and carry no rank information). When set, `merges` is empty,
    /// `pair_ranks` is `None`, and every merge runs through the ranked loops
    /// with priority read from this table (`ranked_merge_key(a, b)` →
    /// `(merged, rank)`). `None` for id-as-rank vocabularies (everything
    /// tiktoken-style), whose fast paths are untouched.
    ranked_merges: Option<Arc<RankedMerges>>,
    pub(crate) vocab: Arc<Vec<Arc<[u8]>>>,
    pub(crate) vocab_inv: Arc<HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher>>,
    pub(crate) byte_remapping: Option<ByteRemapping>,
    /// Append-only arena of encoded token IDs. Cache entries for encodings
    /// of 5+ tokens store `(offset, len)` slices into this vector; shorter
    /// encodings (well over 99% of hit occurrences) live inline in the
    /// cache entry and never touch it.
    token_arena: Vec<TokenId>,
    /// Pretoken cache for the common case (≤ 15 bytes, ~99.9% of
    /// pretokens). The key packs the bytes into the low 15 bytes and the
    /// length into the top byte of a `u128`, so lookups are a single
    /// inlined 128-bit compare instead of a `memcmp` call. See
    /// `pretoken_cache.rs` for why this is a custom prefetchable table
    /// rather than a `HashMap`.
    pretoken_cache: ShortPretokenCache,
    /// Fallback cache for pretokens longer than 15 bytes.
    pretoken_cache_long: HashMap<Box<[u8]>, (u32, u32), rustc_hash::FxBuildHasher>,
    /// Scratch buffers reused across cache-missing pretokens so the merge loop
    /// performs no per-pretoken allocations.
    merge_scratch: MergeScratch,
    symbol_scratch: Vec<TokenId>,
    /// Pretokenization scheme used by [`Self::encode_with_added_tokens`].
    pub(crate) pretokenizer_type: PretokenizerType,
    /// Added tokens (special and non-special), matched atomically in the raw
    /// input before pretokenization, like HuggingFace's AddedVocabulary.
    added_tokens: Vec<AddedTokenDef>,
    /// Leftmost-longest Aho-Corasick automaton over `added_tokens` contents
    /// (pattern index == `added_tokens` index). A prebuilt automaton keeps the
    /// scan fast even when an added token starts with a byte that is common in
    /// text (ModernBERT has 23 space-run added tokens, so a first-byte
    /// candidate scan would probe on every space). Clones share the automaton
    /// via its internal `Arc`.
    added_matcher: Option<aho_corasick::AhoCorasick>,
    /// Apply NFC normalization to non-added-token segments before
    /// pretokenization, like HuggingFace's `NFC` normalizer (e.g. Qwen2).
    normalize_nfc: bool,
    /// HF `ByteLevel(add_prefix_space=true)` (RoBERTa-style exports): every
    /// non-empty added-token-split segment that does not already start with
    /// a space gets one prepended before pretokenization.
    add_prefix_space: bool,
    /// HF BPE `ignore_merges`: a pretoken whose whole byte string is a
    /// vocab entry encodes as that single ID without running the merge
    /// loop. Matters when the vocab has whole-word entries the merges
    /// would decompose differently (GLM-5.2 has ~97k such words); a plain
    /// merge walk diverges from HF on those.
    ignore_merges: bool,
}

/// NFC-normalize a segment if needed, using `buf` as scratch on the slow path.
///
/// ASCII and already-normalized segments are returned as-is. Invalid UTF-8 is
/// passed through unchanged (HF only ever sees `str`, so there is no parity
/// behavior to match).
fn nfc_segment<'a>(seg: &'a [u8], buf: &'a mut String) -> &'a [u8] {
    if seg.is_ascii() {
        return seg;
    }
    let Ok(s) = std::str::from_utf8(seg) else {
        return seg;
    };
    let nfc = icu_normalizer::ComposingNormalizer::new_nfc();
    if nfc.is_normalized(s) {
        return seg;
    }
    buf.clear();
    nfc.normalize_to(s, buf)
        .expect("writing to a String cannot fail");
    buf.as_bytes()
}

/// Cache-value packing (shared by the short-pretoken table and decode in
/// the encode loop). `val` low byte: token count in bits 0-6 plus a
/// "spilled" flag in bit 7. Inline values (1-4 tokens; only the first ID
/// must fit 24 bits — true of every real vocab) carry tokens 1-2 in `val`
/// bits 8-31 and 32-63 and tokens 3-4 in `ext`'s two u32 lanes; spilled
/// values carry the token-arena offset in `val`'s high 32 bits and leave
/// `ext` unused.
const VAL_SPILL: u64 = 0x80;

#[inline(always)]
fn pack_val_inline(symbols: &[TokenId]) -> Option<(u64, u64)> {
    match *symbols {
        [a] if a.0 < (1 << 24) => Some((1 | ((a.0 as u64) << 8), 0)),
        [a, b] if a.0 < (1 << 24) => Some((2 | ((a.0 as u64) << 8) | ((b.0 as u64) << 32), 0)),
        [a, b, c] if a.0 < (1 << 24) => {
            Some((3 | ((a.0 as u64) << 8) | ((b.0 as u64) << 32), c.0 as u64))
        }
        [a, b, c, d] if a.0 < (1 << 24) => Some((
            4 | ((a.0 as u64) << 8) | ((b.0 as u64) << 32),
            c.0 as u64 | ((d.0 as u64) << 32),
        )),
        _ => None,
    }
}

/// View a `TokenId` slice as its underlying `u32`s (repr(transparent)),
/// so bulk emits are `extend_from_slice` memcpys instead of per-element
/// iterator writes.
#[inline(always)]
fn token_ids_as_u32s(toks: &[TokenId]) -> &[u32] {
    // SAFETY: TokenId is #[repr(transparent)] over u32.
    unsafe { std::slice::from_raw_parts(toks.as_ptr() as *const u32, toks.len()) }
}

/// Unpack an inline value's four token lanes (lanes past the count are
/// another key's leftovers; callers truncate by the count).
#[inline(always)]
fn unpack_val_lanes(val: u64, ext: u64) -> [u32; 4] {
    [
        (val >> 8) as u32 & 0xFF_FFFF,
        (val >> 32) as u32,
        ext as u32,
        (ext >> 32) as u32,
    ]
}

/// One piece of the added-token pipeline walk (see
/// [`Tokenizer::for_each_piece`]): a between-occurrences text segment to
/// pretokenize and encode (paired with its source byte offset, which only
/// the verify-heavy differential reads), or an added token's ID to emit
/// verbatim.
enum Piece<'a> {
    Segment(&'a [u8], usize),
    Added(TokenId),
}

/// Explicit merge-priority table for rank-mapped vocabularies:
/// `ranked_merge_key(a, b)` → `(merged, rank)`. Same shape as the
/// SentencePiece engine's merge table.
pub(crate) type RankedMerges = HashMap<u64, (TokenId, u32), rustc_hash::FxBuildHasher>;

/// One added token as configured by the loader: byte content, emitted ID, and
/// HF `AddedToken` whitespace-stripping flags (`lstrip` absorbs whitespace
/// before a match, `rstrip` absorbs whitespace after it). Content is shared
/// (`Arc`) so forks clone entries cheaply.
#[derive(Clone, Debug)]
pub struct AddedTokenDef {
    pub content: Arc<[u8]>,
    pub id: TokenId,
    pub lstrip: bool,
    pub rstrip: bool,
}

/// Byte offset after the leading Unicode whitespace of `bytes` (the set of
/// `str::trim_start`, which is what HF's `\s*` sees). Invalid UTF-8 stops the
/// scan.
fn trim_ws_start(bytes: &[u8]) -> usize {
    let mut pos = 0;
    while pos < bytes.len() {
        let width = match bytes[pos] {
            0x00..=0x7F => 1,
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => break,
        };
        let Some(chunk) = bytes.get(pos..pos + width) else {
            break;
        };
        match std::str::from_utf8(chunk) {
            Ok(s) if s.chars().next().is_some_and(char::is_whitespace) => pos += width,
            _ => break,
        }
    }
    pos
}

/// Length of `bytes` after trimming trailing Unicode whitespace (the set of
/// `str::trim_end`). Invalid UTF-8 stops the scan.
fn trim_ws_end(bytes: &[u8]) -> usize {
    let mut end = bytes.len();
    while end > 0 {
        // Back up over at most 3 continuation bytes to the character start.
        let mut start = end - 1;
        while start > 0 && (bytes[start] & 0xC0) == 0x80 && end - start < 4 {
            start -= 1;
        }
        match std::str::from_utf8(&bytes[start..end]) {
            Ok(s) if s.chars().next().is_some_and(char::is_whitespace) => end = start,
            _ => break,
        }
    }
    end
}

/// Overwrite the short-cache entry of every added-token content of 1..=15
/// bytes that resolves in `vocab_inv` with that single ID, so a matching
/// pretoken encodes as the added token rather than its merge decomposition.
///
/// This function IS the cache-seed sync invariant: the short cache's
/// seed-level state is always "vocab seed, then these overwrites" — a pure
/// function of `(vocab, added_tokens)` — because both
/// [`Tokenizer::set_added_tokens`] (on the parent) and
/// [`Tokenizer::fork_sized`] (after a fork's fresh reseed) apply the
/// overwrites through this one body, so parent and forked workers always
/// agree on every short pretoken.
fn apply_added_token_overwrites(
    added_tokens: &[AddedTokenDef],
    vocab_inv: &HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher>,
    pretoken_cache: &mut ShortPretokenCache,
    token_arena: &mut Vec<TokenId>,
) {
    for tok in added_tokens {
        let content = &tok.content;
        if !(1..=15).contains(&content.len()) {
            continue;
        }
        let Some(&id) = vocab_inv.get(content) else {
            continue;
        };
        let key = pack_pretoken_key(content).expect("length checked <= 15");
        let h = pretoken_key_hash(key);
        let (val, ext) = Tokenizer::pack_val(&[id], token_arena);
        pretoken_cache.replace(key, h, val, ext);
    }
}

impl Tokenizer {
    pub fn new(
        merges: HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        vocab: Vec<Vec<u8>>,
        byte_remapping: Option<ByteRemapping>,
    ) -> Self {
        let vocab = vocab.into_iter().map(Into::into).collect();
        Self::from_tables(merges, None, vocab, byte_remapping)
    }

    /// Construct from an explicit-rank merge table (`ranked_merge_key(a, b)`
    /// → `(merged, rank)`), for vocabularies whose IDs do not follow merge
    /// order (see `ranked_merges`). Every merge runs through the ranked
    /// loops; the id-as-rank fast paths stay off.
    pub fn new_ranked(
        ranked_merges: RankedMerges,
        vocab: Vec<Vec<u8>>,
        byte_remapping: Option<ByteRemapping>,
    ) -> Self {
        let vocab = vocab.into_iter().map(Into::into).collect();
        Self::from_tables(
            HashMap::default(),
            Some(ranked_merges),
            vocab,
            byte_remapping,
        )
    }

    /// Shared construction tail ([`Self::new`], [`Self::new_ranked`] and
    /// [`Self::from_ranks`]): derive `vocab_inv` and the pair-rank table
    /// from the finished merges/vocab, seed the pretoken cache, and
    /// assemble the tokenizer with default pipeline settings (GPT-2
    /// pretokenization, no added tokens, no NFC).
    fn from_tables(
        merges: HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        ranked_merges: Option<RankedMerges>,
        vocab: Vec<Arc<[u8]>>,
        byte_remapping: Option<ByteRemapping>,
    ) -> Self {
        let vocab_inv: HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher> = vocab
            .iter()
            .cloned()
            .zip((0..).map(TokenId::from))
            .collect();
        let ranked_merges = ranked_merges.map(Arc::new);
        let pair_ranks = if ranked_merges.is_none() {
            PairRankTable::build(&merges, byte_remapping.as_ref(), vocab.len()).map(Arc::new)
        } else {
            None
        };
        let mut token_arena = Vec::new();
        let pretoken_cache = Self::seeded_pretoken_cache(
            &vocab,
            byte_remapping.as_ref(),
            pair_ranks.as_deref(),
            &merges,
            ranked_merges.as_deref(),
            false,
            &vocab_inv,
            &mut token_arena,
            0,
        );
        Tokenizer {
            merges: Arc::new(merges),
            pair_ranks,
            ranked_merges,
            vocab_inv: Arc::new(vocab_inv),
            vocab: Arc::new(vocab),
            byte_remapping,
            token_arena,
            pretoken_cache,
            pretoken_cache_long: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            merge_scratch: MergeScratch::default(),
            symbol_scratch: Vec::new(),
            pretokenizer_type: PretokenizerType::GPT2,
            added_tokens: Vec::new(),
            added_matcher: None,
            normalize_nfc: false,
            add_prefix_space: false,
            ignore_merges: false,
        }
    }

    /// A short-pretoken cache pre-seeded with the BPE encoding of every
    /// vocab entry of 1..=15 bytes: precomputed miss results, computed by
    /// the same [`Self::merge_short`] the miss path runs, so a seeded
    /// value is bit-identical to what a cold miss on those bytes would
    /// have produced and cached. Any short pretoken that is a whole vocab
    /// word then hits the cache outright, so the miss path never sees one.
    ///
    /// Without `ignore_merges`, the seed value must be the MERGE RESULT,
    /// not the entry's own ID: BPE encode semantics (HF `tokenizers`
    /// without `ignore_merges`, this repo's merge loop, and the pre-cache
    /// baseline 0e27c71) produce a whole-word token only when the merge
    /// rules can derive it, and vocabs may contain merge-UNREACHABLE
    /// entries — qwen3_5 has ~200 (multi-char CJK phrases, " Jap\u{f3}n",
    /// …) that must encode as their merge decomposition. Seeding
    /// `bytes -> [own id]` was a measured divergence from HF (see
    /// `verify_vocab_seeded_cache_matches_merge_decomposition`). For
    /// merge-reachable entries — all of gpt2/olmo3/qwen2/deepseek_v3 —
    /// the merge result is the single own ID, as before. Duplicate byte
    /// strings encode identically (the merge sees only bytes), so the
    /// insert-if-absent dedup is purely a work-skip.
    ///
    /// WITH `ignore_merges` the rule flips: HF emits the vocab entry's own
    /// ID for any whole-pretoken vocab hit, so every seed value is
    /// `[vocab_inv[bytes]]` (`vocab_inv` also resolves duplicate byte
    /// strings to the one ID a lookup would find).
    ///
    /// `min_slots` additionally floors the table size for a worker with a
    /// known workload (see [`Self::fork_sized`]); the table is built once
    /// at the max of the seed requirement and that floor, so seeding never
    /// grows it mid-way. Values of 5+ tokens (only possible for
    /// merge-unreachable entries) spill into `token_arena` like any other
    /// miss.
    #[allow(clippy::too_many_arguments)]
    fn seeded_pretoken_cache(
        vocab: &[Arc<[u8]>],
        byte_remapping: Option<&ByteRemapping>,
        pair_ranks: Option<&PairRankTable>,
        merges: &HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        ranked_merges: Option<&RankedMerges>,
        ignore_merges: bool,
        vocab_inv: &HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher>,
        token_arena: &mut Vec<TokenId>,
        min_slots: usize,
    ) -> ShortPretokenCache {
        let n_short = vocab
            .iter()
            .filter(|bytes| (1..=15).contains(&bytes.len()))
            .count();
        let mut cache = ShortPretokenCache::with_at_least(n_short, min_slots);
        let mut buf = [TokenId(0); SHORT_MERGE_MAX];
        for bytes in vocab {
            if !(1..=15).contains(&bytes.len()) {
                continue;
            }
            let key = pack_pretoken_key(bytes).expect("length checked <= 15");
            let h = pretoken_key_hash(key);
            // Duplicate byte strings seed the same value (see the doc
            // above), so insertion order is irrelevant and the
            // insert-if-absent check only skips redundant merges.
            if cache.get_or_slot(key, h).is_err() {
                let n = Self::seed_symbols_any(
                    byte_remapping,
                    pair_ranks,
                    merges,
                    ranked_merges,
                    ignore_merges,
                    vocab_inv,
                    bytes,
                    &mut buf,
                );
                let (val, ext) = Self::pack_val(&buf[..n], token_arena);
                cache.insert(key, h, val, ext);
            }
        }
        cache
    }

    /// Seed-level encoding of one short vocab byte string under the
    /// current `ignore_merges` setting: the single `vocab_inv` ID when the
    /// flag is set (HF's whole-pretoken vocab hit), the merge
    /// decomposition otherwise. One body shared by
    /// [`Self::seeded_pretoken_cache`] and [`Self::set_ignore_merges`] so
    /// a fork's fresh reseed and the parent's in-place rewrite always
    /// agree.
    #[inline]
    fn seed_symbols(
        byte_remapping: Option<&ByteRemapping>,
        pair_ranks: Option<&PairRankTable>,
        merges: &HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        ignore_merges: bool,
        vocab_inv: &HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher>,
        bytes: &[u8],
        buf: &mut [TokenId; SHORT_MERGE_MAX],
    ) -> usize {
        if ignore_merges && let Some(&id) = vocab_inv.get(bytes) {
            buf[0] = id;
            return 1;
        }
        Self::merge_short(byte_remapping, pair_ranks, merges, bytes, buf)
    }

    /// [`Self::seed_symbols`] for either merge-table shape: dispatches to
    /// the ranked variant when `ranked_merges` is set. Load-time call sites
    /// (cache seeding, flag flips) go through this; the per-pretoken miss
    /// path dispatches once per pretoken instead (see
    /// [`Self::encode_pretoken_miss`]), keeping the id-as-rank miss
    /// codegen identical to a build without ranked support.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn seed_symbols_any(
        byte_remapping: Option<&ByteRemapping>,
        pair_ranks: Option<&PairRankTable>,
        merges: &HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        ranked_merges: Option<&RankedMerges>,
        ignore_merges: bool,
        vocab_inv: &HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher>,
        bytes: &[u8],
        buf: &mut [TokenId; SHORT_MERGE_MAX],
    ) -> usize {
        match ranked_merges {
            Some(rm) => {
                Self::seed_symbols_ranked(byte_remapping, rm, ignore_merges, vocab_inv, bytes, buf)
            }
            None => Self::seed_symbols(
                byte_remapping,
                pair_ranks,
                merges,
                ignore_merges,
                vocab_inv,
                bytes,
                buf,
            ),
        }
    }

    /// Ranked-merge-table variant of [`Self::seed_symbols`].
    fn seed_symbols_ranked(
        byte_remapping: Option<&ByteRemapping>,
        ranked_merges: &RankedMerges,
        ignore_merges: bool,
        vocab_inv: &HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher>,
        bytes: &[u8],
        buf: &mut [TokenId; SHORT_MERGE_MAX],
    ) -> usize {
        if ignore_merges && let Some(&id) = vocab_inv.get(bytes) {
            buf[0] = id;
            return 1;
        }
        let n = bytes.len();
        debug_assert!((1..SHORT_MERGE_MAX).contains(&n));
        match byte_remapping {
            Some(br) => {
                for (dst, &b) in buf[..n].iter_mut().zip(bytes) {
                    *dst = br.mapping[b as usize];
                }
            }
            None => {
                for (dst, &b) in buf[..n].iter_mut().zip(bytes) {
                    *dst = TokenId(b as u32);
                }
            }
        }
        if n < 2 {
            return n;
        }
        bpe_merge_symbols_ranked_slice(ranked_merges, &mut buf[..n])
    }

    /// BPE-encode one short pretoken (1..=15 bytes) into `buf`, returning
    /// its token count: byte remapping, then the short merge loop. This is
    /// exactly the computation [`Self::encode_pretoken_miss`] performs for
    /// short keys — shared with [`Self::seeded_pretoken_cache`] so the
    /// vocab seed can never disagree with a cold miss.
    #[inline]
    fn merge_short(
        byte_remapping: Option<&ByteRemapping>,
        pair_ranks: Option<&PairRankTable>,
        merges: &HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        bytes: &[u8],
        buf: &mut [TokenId; SHORT_MERGE_MAX],
    ) -> usize {
        let n = bytes.len();
        debug_assert!((1..SHORT_MERGE_MAX).contains(&n));
        match byte_remapping {
            Some(br) => {
                for (dst, &b) in buf[..n].iter_mut().zip(bytes) {
                    *dst = br.mapping[b as usize];
                }
            }
            None => {
                for (dst, &b) in buf[..n].iter_mut().zip(bytes) {
                    *dst = TokenId(b as u32);
                }
            }
        }
        if n < 2 {
            return n;
        }
        match pair_ranks {
            #[cfg(target_arch = "aarch64")]
            Some(table) => bpe_merge_symbols_short_neon(table, buf, n),
            // x86-64 stays scalar ON PURPOSE: the AVX-512/AVX2 ports of the
            // min-rank scan (`bpe_merge_symbols_short_avx512/_avx2`, kept as
            // tested reference) measured ~1% SLOWER on cold encode_st (Zen 5,
            // gpt2, 100 MB and 1 GB OWT, interleaved min-of-5) — the x86
            // horizontal reduce is a 4-step dependent chain plus a
            // vector->GPR transfer on the serial merge chain, and the
            // `target_feature` boundary blocks inlining, while the scalar
            // scan's `rank < best` branches predict well on Zen 5. See
            // profiling/x86_port_plan.md §6.
            #[cfg(not(target_arch = "aarch64"))]
            Some(table) => bpe_merge_symbols_short_scalar(
                |a, b| table.rank(a, b),
                |a, b| table.prefetch_rank(a, b),
                buf,
                n,
            ),
            None => bpe_merge_symbols_short_scalar(
                |a, b| merges.get(&(a, b)).map_or(u32::MAX, |m| m.0),
                |_, _| {},
                buf,
                n,
            ),
        }
    }

    /// Pack a cache value: inline when possible, else spilled to the arena.
    #[inline(always)]
    fn pack_val(symbols: &[TokenId], token_arena: &mut Vec<TokenId>) -> (u64, u64) {
        pack_val_inline(symbols).unwrap_or_else(|| {
            let offset = token_arena.len() as u64;
            token_arena.extend_from_slice(symbols);
            (VAL_SPILL | symbols.len() as u64 | (offset << 32), 0)
        })
    }

    /// Given a list of tokens in rank order (by merge order), reconstructs the
    /// merges map and returns a Tokenizer.
    ///
    /// This process is necessary to load some tokenizers found in tiktoken.
    pub fn from_ranks(vocab: Vec<Vec<u8>>) -> Result<Self> {
        let mut merges: HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher> =
            HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        let vocab = vocab
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Arc<[u8]>>>();
        let vocab_inv: HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher> = vocab
            .iter()
            .cloned()
            .zip((0..).map(TokenId::from))
            .collect();

        for (token_idx, token_bytes) in vocab.iter().cloned().enumerate() {
            if token_bytes.len() < 2 {
                continue;
            }
            let byte_symbols: Vec<u8> = token_bytes
                .iter()
                .map(|b| vocab_inv.get(std::slice::from_ref(b)).unwrap().0 as u8)
                .collect();
            let tokenized = simple_bpe_merge(&merges, &byte_symbols);
            assert_eq!(tokenized.len(), 2);
            merges.insert((tokenized[0], tokenized[1]), TokenId::from(token_idx));
        }

        let byte_remapping = ByteRemapping::from_byte_vocab(&vocab)?;
        Ok(Self::from_tables(merges, None, vocab, byte_remapping))
    }

    /// Create a new tokenizer sharing the same model data but with a
    /// freshly seeded cache (no encoded pretokens beyond the vocab seed).
    /// Useful for per-thread encoding in parallel.
    pub fn fork(&self) -> Self {
        self.fork_sized(0)
    }

    /// [`Self::fork`] with the caches pre-sized for a worker expected to
    /// encode roughly `expected_bytes` of input. On a cold parallel run a
    /// default-sized worker rehashes its pretoken table through 6-7
    /// doublings — random scatter writes into a fresh zeroed allocation
    /// each time, on every worker at once; sizing from the input share
    /// pays for the table exactly once. The estimates are capacity hints
    /// only: every structure still grows past them as needed, and the
    /// clamps keep tiny inputs at the default size. The short-table size
    /// is a floor passed through the vocab seeding, so the seed
    /// requirement and the workload estimate resolve to one table
    /// construction (whichever is larger).
    pub(crate) fn fork_sized(&self, expected_bytes: usize) -> Self {
        // Distinct short pretokens follow Heaps' law: ~1.3M at 1 GB and
        // ~5.5M at 10 GB of OWT-like text gives distinct(n) ≈ 3.45·n^0.62.
        // Size for the Heaps estimate at the table's 3/4 growth load with
        // 1.4x headroom (self-paced chunk handout lets a fast core encode
        // more than its even share; the margin holds a >2x-oversubscribed
        // worker under the growth threshold before the table would resize).
        // Still a capacity hint: the table grows past it at 3/4 load on
        // corpora more diverse than the OWT calibration. Clamped to 2^22
        // slots (128 MB) per worker.
        let distinct = 3.45 * (expected_bytes as f64).powf(0.62);
        let cache_slots = ((distinct * (4.0 / 3.0) * 1.4) as usize)
            .clamp(1 << 16, 1 << 22)
            .next_power_of_two();
        let arena_cap = (expected_bytes / 256).min(1 << 24);
        let long_cap = (expected_bytes / 8192).min(1 << 20);
        let mut token_arena = Vec::with_capacity(arena_cap);
        let mut pretoken_cache = Self::seeded_pretoken_cache(
            &self.vocab,
            self.byte_remapping.as_ref(),
            self.pair_ranks.as_deref(),
            &self.merges,
            self.ranked_merges.as_deref(),
            self.ignore_merges,
            &self.vocab_inv,
            &mut token_arena,
            cache_slots,
        );
        // The vocab seed above holds the plain seed encoding of every short
        // byte string (merge result, or own ID under `ignore_merges`);
        // re-apply the added-token `[id]` overwrites so the fork's cache
        // matches the parent's seed-level state (the shared function is the
        // sync invariant — see `apply_added_token_overwrites`).
        apply_added_token_overwrites(
            &self.added_tokens,
            &self.vocab_inv,
            &mut pretoken_cache,
            &mut token_arena,
        );
        Tokenizer {
            merges: Arc::clone(&self.merges),
            pair_ranks: self.pair_ranks.clone(),
            ranked_merges: self.ranked_merges.clone(),
            vocab: Arc::clone(&self.vocab),
            vocab_inv: Arc::clone(&self.vocab_inv),
            byte_remapping: self.byte_remapping.clone(),
            token_arena,
            pretoken_cache,
            pretoken_cache_long: HashMap::with_capacity_and_hasher(
                long_cap,
                rustc_hash::FxBuildHasher {},
            ),
            merge_scratch: MergeScratch::default(),
            symbol_scratch: Vec::new(),
            pretokenizer_type: self.pretokenizer_type,
            added_tokens: self.added_tokens.clone(),
            added_matcher: self.added_matcher.clone(),
            normalize_nfc: self.normalize_nfc,
            add_prefix_space: self.add_prefix_space,
            ignore_merges: self.ignore_merges,
        }
    }

    /// Loader-phase mutator: like every `Tokenizer` mutation, this must
    /// run before any `WorkerPool` forks workers from this tokenizer —
    /// already-forked workers keep the old state (see [`WorkerPool`]).
    ///
    /// [`WorkerPool`]: crate::batch::WorkerPool
    pub fn set_pretokenizer_type(&mut self, pretokenizer_type: PretokenizerType) {
        self.pretokenizer_type = pretokenizer_type;
    }

    pub fn pretokenizer_type(&self) -> PretokenizerType {
        self.pretokenizer_type
    }

    /// Enable NFC normalization of non-added-token segments before
    /// pretokenization (HF `normalizer: {"type": "NFC"}`).
    pub fn set_normalize_nfc(&mut self, normalize_nfc: bool) {
        self.normalize_nfc = normalize_nfc;
    }

    /// Enable HF `ByteLevel(add_prefix_space=true)` semantics (see the
    /// `add_prefix_space` field).
    pub fn set_add_prefix_space(&mut self, add_prefix_space: bool) {
        self.add_prefix_space = add_prefix_space;
    }

    /// Enable HF BPE `ignore_merges` semantics: a pretoken whose whole
    /// byte string is a vocab entry encodes as that single ID, skipping
    /// the merge loop.
    ///
    /// Rewrites the vocab-seeded short-cache entries to the new flag's
    /// seed values (own ID vs merge decomposition — see
    /// [`Self::seed_symbols`]) and reasserts the added-token overwrites,
    /// so the cache stays a pure function of
    /// `(vocab, ignore_merges, added_tokens)` and matches what a fork's
    /// fresh reseed produces.
    ///
    /// Loader-phase mutator: must run before any `WorkerPool` forks
    /// workers from this tokenizer — already-forked workers keep the old
    /// state (see [`WorkerPool`]).
    ///
    /// [`WorkerPool`]: crate::batch::WorkerPool
    pub fn set_ignore_merges(&mut self, ignore_merges: bool) {
        if self.ignore_merges == ignore_merges {
            return;
        }
        self.ignore_merges = ignore_merges;
        let mut buf = [TokenId(0); SHORT_MERGE_MAX];
        for bytes in self.vocab.iter() {
            if !(1..=15).contains(&bytes.len()) {
                continue;
            }
            let key = pack_pretoken_key(bytes).expect("length checked <= 15");
            let h = pretoken_key_hash(key);
            let n = Self::seed_symbols_any(
                self.byte_remapping.as_ref(),
                self.pair_ranks.as_deref(),
                &self.merges,
                self.ranked_merges.as_deref(),
                ignore_merges,
                &self.vocab_inv,
                bytes,
                &mut buf,
            );
            let (val, ext) = Self::pack_val(&buf[..n], &mut self.token_arena);
            self.pretoken_cache.replace(key, h, val, ext);
        }
        apply_added_token_overwrites(
            &self.added_tokens,
            &self.vocab_inv,
            &mut self.pretoken_cache,
            &mut self.token_arena,
        );
    }

    /// Set the added tokens matched atomically by
    /// [`Self::encode_with_added_tokens`]. Empty contents are ignored.
    ///
    /// Loader-phase mutator: must run before any `WorkerPool` forks
    /// workers from this tokenizer — already-forked workers keep the old
    /// added-token set (see [`WorkerPool`]).
    ///
    /// [`WorkerPool`]: crate::batch::WorkerPool
    pub fn set_added_tokens(&mut self, added_tokens: Vec<AddedTokenDef>) {
        let mut added_tokens: Vec<AddedTokenDef> = added_tokens
            .into_iter()
            .filter(|t| !t.content.is_empty())
            .collect();
        added_tokens.sort_by_key(|t| std::cmp::Reverse(t.content.len()));
        self.added_matcher = (!added_tokens.is_empty()).then(|| {
            aho_corasick::AhoCorasick::builder()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(added_tokens.iter().map(|t| t.content.as_ref()))
                .expect("added-token automaton construction cannot fail")
        });
        let outgoing = std::mem::replace(&mut self.added_tokens, added_tokens);
        // Restore the plain seed value for the outgoing set first (only
        // contents that resolve in `vocab_inv` were ever overwritten, so
        // this replaces existing entries and never inserts), then apply
        // the incoming overwrites through the shared sync-invariant body
        // (see `apply_added_token_overwrites`).
        for tok in &outgoing {
            let content = &tok.content;
            if !(1..=15).contains(&content.len()) || self.vocab_inv.get(content).is_none() {
                continue;
            }
            let key = pack_pretoken_key(content).expect("length checked <= 15");
            let h = pretoken_key_hash(key);
            let mut buf = [TokenId(0); SHORT_MERGE_MAX];
            let n = Self::seed_symbols_any(
                self.byte_remapping.as_ref(),
                self.pair_ranks.as_deref(),
                &self.merges,
                self.ranked_merges.as_deref(),
                self.ignore_merges,
                &self.vocab_inv,
                content,
                &mut buf,
            );
            let (val, ext) = Self::pack_val(&buf[..n], &mut self.token_arena);
            self.pretoken_cache.replace(key, h, val, ext);
        }
        apply_added_token_overwrites(
            &self.added_tokens,
            &self.vocab_inv,
            &mut self.pretoken_cache,
            &mut self.token_arena,
        );
    }

    /// Register one additional added token, extending the decode vocab when
    /// its id lies outside the base ranks (mirrors the out-of-vocab
    /// added-token handling in the HF loader).
    ///
    /// Loader-phase mutator: must run before any `WorkerPool` forks
    /// workers from this tokenizer — already-forked workers keep the old
    /// vocab, matcher, and cache seed (see [`WorkerPool`]).
    ///
    /// [`WorkerPool`]: crate::batch::WorkerPool
    pub fn add_special_token(&mut self, content: Vec<u8>, id: TokenId) {
        let idx = id.0 as usize;
        // Loader-phase mutation of the shared model tables: `make_mut`
        // copies only when a fork holds the tables too (never during
        // loading, where this is called).
        let vocab = Arc::make_mut(&mut self.vocab);
        if idx >= vocab.len() {
            vocab.resize(idx + 1, Arc::from(Vec::new().as_slice()));
        }
        if vocab[idx].is_empty() {
            vocab[idx] = content.clone().into();
            // If `content` duplicates an already-present vocab byte
            // string, `vocab_inv` switches to the new ID (unconditional
            // overwrite). The short-cache overwrite that keeps a matching
            // pretoken resolving to `vocab_inv`'s answer happens in
            // `set_added_tokens` below, which re-derives every added-token
            // cache overwrite from the updated `vocab_inv` — the same
            // computation a fork's reseed + re-apply performs (see
            // [`Self::fork_sized`]), so parent and forked workers agree.
            Arc::make_mut(&mut self.vocab_inv).insert(vocab[idx].clone(), id);
        }
        let mut added = self.added_tokens.clone();
        added.push(AddedTokenDef {
            content: content.into(),
            id,
            lstrip: false,
            rstrip: false,
        });
        self.set_added_tokens(added);
    }

    /// Size of the vocabulary: one greater than the largest token ID,
    /// including added tokens (IDs with no assigned content count too).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Vocabulary entries as `(id, bytes)` pairs in ID order, including
    /// added tokens and skipping IDs with no assigned content.
    pub fn vocab_entries(&self) -> impl Iterator<Item = (u32, &[u8])> {
        super::vocab_entries(&self.vocab)
    }

    /// Merge rules as `(left, right)` byte pairs in merge-priority order
    /// (priority equals the merged token's ID for tiktoken vocabularies;
    /// rank-mapped vocabularies keep their explicit rank order).
    pub fn merge_entries(&self) -> Vec<(&[u8], &[u8])> {
        let mut ranked: Vec<(u32, u32, u32)> = match self.ranked_merges.as_deref() {
            Some(rm) => rm
                .iter()
                .map(|(&key, &(_, rank))| ((key >> 32) as u32, key as u32, rank))
                .collect(),
            None => self
                .merges
                .iter()
                .map(|(&(a, b), &m)| (a.0, b.0, m.0))
                .collect(),
        };
        ranked.sort_unstable_by_key(|&(.., priority)| priority);
        ranked
            .into_iter()
            .map(|(a, b, _)| {
                (
                    self.vocab[a as usize].as_ref(),
                    self.vocab[b as usize].as_ref(),
                )
            })
            .collect()
    }

    /// Added-token contents paired with their `rstrip` flag, for
    /// `pretokenize::safe_split_ranges`: an rstrip occurrence must not end
    /// exactly at a chunk boundary, or the whitespace it would absorb lands
    /// at the start of the next chunk and encodes as plain text.
    pub fn added_token_split_blockers(&self) -> Vec<(&[u8], bool)> {
        self.added_tokens
            .iter()
            .map(|t| (t.content.as_ref(), t.rstrip))
            .collect()
    }

    /// Find the leftmost added-token occurrence at or after `from`, taking
    /// the longest token when several match at the same position. Returns
    /// `(start, end, index into added_tokens)`.
    fn find_added_token(&self, bytes: &[u8], from: usize) -> Option<(usize, usize, usize)> {
        let m = self.added_matcher.as_ref()?.find(&bytes[from..])?;
        Some((from + m.start(), from + m.end(), m.pattern().as_usize()))
    }

    /// Shared piece walk of the added-token pipeline: split out added-token
    /// occurrences and hand each piece — the (possibly NFC-normalized)
    /// segment between occurrences, or the added token's ID — to `f` in
    /// input order. Scheme dispatch costs one enum match per 256-pretoken
    /// chunk fill (see [`PretokenizerType::pretokenize`] and
    /// `FastPretokenizerDispatch::fill_spans_keyed`), which delegates to
    /// the same out-of-line concrete fills a hardcoded pretokenizer uses.
    fn for_each_piece(&mut self, bytes: &[u8], mut f: impl FnMut(&mut Self, Piece<'_>)) {
        let normalize_nfc = self.normalize_nfc;
        let mut nfc_buf = String::new();
        let mut prefix_buf = Vec::new();
        let mut pos = 0;
        while pos < bytes.len() {
            let (mut seg_end, added) = match self.find_added_token(bytes, pos) {
                Some((start, end, idx)) => {
                    let t = &self.added_tokens[idx];
                    (start, Some((end, t.id, t.lstrip, t.rstrip)))
                }
                None => (bytes.len(), None),
            };
            // An lstrip added token absorbs the whitespace before it (HF's
            // `\s*` on the left of the match); drop it from the segment.
            if let Some((_, _, true, _)) = added {
                seg_end = pos + trim_ws_end(&bytes[pos..seg_end]);
            }
            let mut segment = if normalize_nfc {
                nfc_segment(&bytes[pos..seg_end], &mut nfc_buf)
            } else {
                &bytes[pos..seg_end]
            };
            if self.add_prefix_space && !segment.is_empty() && segment[0] != b' ' {
                prefix_buf.clear();
                prefix_buf.push(b' ');
                prefix_buf.extend_from_slice(segment);
                segment = &prefix_buf;
            }
            f(self, Piece::Segment(segment, pos));
            match added {
                Some((end, id, _, rstrip)) => {
                    f(self, Piece::Added(id));
                    // An rstrip added token absorbs the whitespace after it.
                    pos = if rstrip {
                        end + trim_ws_start(&bytes[end..])
                    } else {
                        end
                    };
                }
                None => break,
            }
        }
    }

    /// Encode raw text: split out added-token occurrences (emitted as their
    /// single token ID), pretokenize the segments between them with this
    /// tokenizer's pretokenization scheme, and BPE-encode each pretoken.
    /// This mirrors the full HuggingFace `tokenizers` encode pipeline.
    pub fn encode_with_added_tokens(&mut self, bytes: &[u8], mut f: impl FnMut(&[TokenId])) {
        let pt = self.pretokenizer_type;
        self.for_each_piece(bytes, |this, piece| match piece {
            Piece::Segment(segment, _) => this.memoized_encode(pt.pretokenize(segment), &mut f),
            Piece::Added(id) => f(&[id]),
        });
    }

    /// Flat variant of [`Self::encode_with_added_tokens`]: the identical
    /// token stream appended to `out` as raw u32 ids, routed through
    /// [`Self::memoized_encode_flat`] so segment tokens land directly in
    /// the caller's buffer (the batch engine's per-chunk id buffer).
    pub fn encode_with_added_tokens_flat(&mut self, bytes: &[u8], out: &mut Vec<u32>) {
        let pt = self.pretokenizer_type;
        self.for_each_piece(bytes, |this, piece| match piece {
            Piece::Segment(segment, _) => this.memoized_encode_flat(pt.pretokenize(segment), out),
            Piece::Added(id) => out.push(id.0),
        });
    }

    /// For each pretoken in the input iterator, looks up the string in the
    /// cache, and if not found, encodes it and inserts it into the cache.
    /// Calls `f` with the encoded token slice for each pretoken.
    ///
    /// A thin wrapper over the flat probe/emit machinery (see
    /// [`Self::memoized_encode_flat`], the path the batch engine and
    /// benches use): each chunk's tokens land in a reused L1-resident
    /// buffer with per-pretoken end offsets recorded on the side, then `f`
    /// receives one slice per pretoken.
    pub fn memoized_encode<'i>(
        &mut self,
        mut pretokens: impl PretokenSpans<'i>,
        mut f: impl FnMut(&[TokenId]),
    ) {
        let mut batch = SpanBatch::new();
        let mut out: Vec<u32> = Vec::new();
        let mut ends = [0usize; PRETOKEN_CHUNK];
        loop {
            let cache = &self.pretoken_cache;
            let n = pretokens.fill_spans_keyed(&mut batch, &|h| cache.prefetch_l2(h));
            if n == 0 {
                break;
            }
            out.clear();
            self.probe_emit_chunk(&batch, n, &mut out, |i, w| ends[i] = w);
            let mut start = 0;
            for &end in &ends[..n] {
                // SAFETY: TokenId is repr(transparent) over u32, and the
                // recorded ends partition `out` (0 <= start <= end <= len).
                f(unsafe {
                    std::slice::from_raw_parts(
                        out.as_ptr().add(start) as *const TokenId,
                        end - start,
                    )
                });
                start = end;
            }
            if n < PRETOKEN_CHUNK {
                break;
            }
        }
    }

    /// Flat variant of [`Self::memoized_encode`]: the identical token
    /// stream appended to `out` as raw u32 ids (bit-compatible with
    /// `TokenId`), with no per-pretoken delivery. This is the batch
    /// engine's output shape (`batch::encode_into` fills chunk id buffers),
    /// so the emit loop writes tokens straight into the final buffer.
    ///
    /// Runs in chunks of `PRETOKEN_CHUNK` pretokens through two phases —
    /// pull spans from the pretokenizer with keys/hashes derived and probe
    /// lines prefetched into L2 on the way out (out of line, fused with
    /// the span walker — see PretokenSpans), then probe and emit. The
    /// phase split keeps the walker's state register-allocated in one
    /// tight loop and gives every probe line a chunk of latency (hundreds
    /// of cycles, enough to cover DRAM) before its probe.
    pub fn memoized_encode_flat<'i>(
        &mut self,
        mut pretokens: impl PretokenSpans<'i>,
        out: &mut Vec<u32>,
    ) {
        let mut batch = SpanBatch::new();
        loop {
            let cache = &self.pretoken_cache;
            let n = pretokens.fill_spans_keyed(&mut batch, &|h| cache.prefetch_l2(h));
            if n == 0 {
                break;
            }
            self.probe_emit_chunk(&batch, n, out, |_, _| {});
            if n < PRETOKEN_CHUNK {
                break;
            }
        }
    }

    /// Probe-and-emit for one chunk: branchless flat emit with a single
    /// rare data-dependent branch per pretoken. Every iteration stores the
    /// probed value's four token lanes unconditionally at the write cursor
    /// and advances by the token count only when the fast predicate (pair
    /// hit ∧ inline value ∧ short key, ~99% of pretokens) holds; stores
    /// past the cursor are dead — overwritten by a later iteration or
    /// truncated by the final `set_len`. Everything else — probe walks
    /// past the home pair, arena spills, long pretokens, misses — takes
    /// the `#[cold]` slow path. `record(i, cursor)` runs once per pretoken
    /// (per-pretoken slicing in [`Self::memoized_encode`]; a no-op closure
    /// in the flat variant).
    ///
    /// Slack invariant: `out.capacity() >= cursor + 4 * (iterations
    /// left)`, established by the reserve below and re-established by the
    /// slow path after any reallocation, so the two 8-byte stores are
    /// always in bounds.
    #[inline(always)]
    fn probe_emit_chunk(
        &mut self,
        batch: &SpanBatch<'_>,
        n: usize,
        out: &mut Vec<u32>,
        mut record: impl FnMut(usize, usize),
    ) {
        // One check up front so `i`- and `pf`-indexing of the batch arrays
        // below is provably in bounds (removes two per-iteration compares).
        assert!(n <= PRETOKEN_CHUNK);
        if n == 0 {
            return;
        }
        out.reserve(4 * n);
        let mut w = out.len();
        // Loop-invariant raw cursors. The slow path's `&mut self` call is
        // the only thing that can move `out`'s buffer or the cache's slot
        // array, so both are refreshed there and nowhere else; without
        // these the compiler reloaded the Vec pointer, table base, and
        // mask from the stack on every iteration.
        let mut dst = out.as_mut_ptr();
        let mut table = self.pretoken_cache.probe_view();
        // Probe-stage prefetch: promote the pair's line L2 -> L1 a fixed
        // short distance ahead (the fill phase staged it into L2; D only
        // has to cover the L2 hit latency, a handful of iterations).
        const D: usize = 16;
        const _: () = assert!(D <= crate::pretokenize::SPAN_BATCH_SLACK);
        for i in 0..D.min(n) {
            table.prefetch(batch.entries[i].meta);
        }
        for i in 0..n {
            // Unclamped prefetch distance: the batch carries D slack
            // entries past a full chunk, so `i + D` always indexes into
            // the array and no per-pretoken bounds clamp is needed — the
            // load is one fixed-offset ldr off the walking entry pointer.
            // Tail iterations prefetch stale or zero `meta`, and long
            // entries a length, not a hash — either way a masked,
            // in-bounds table line: harmless.
            table.prefetch(batch.entries[i + D].meta);
            // One 32-byte entry: key + meta land in a single cache line
            // (the parallel-array layout walked three load streams here).
            let (key, h) = (batch.entries[i].key, batch.entries[i].meta);
            let (val, ext, found) = table.probe_pair(key, h);
            // `key != 0` folds the long-pretoken route in AND guards the
            // empty-slot sentinel (probe_pair matches key 0 against empty
            // slots); on !found the lanes below are another entry's, dead
            // because the cursor does not advance.
            let fast = found & (val & VAL_SPILL == 0) & (key != 0);
            // Lanes 1-2 packed into one u64 store, lanes 3-4 are `ext`
            // verbatim (little-endian lane order, like the raw key load in
            // `pack_pretoken_key`); the two u64 writes fuse into one 16 B
            // `stp`.
            let ab = ((val >> 8) & 0x00FF_FFFF) | (val & 0xFFFF_FFFF_0000_0000);
            // SAFETY: the slack invariant leaves >= 4 u32s past `w`.
            unsafe {
                let p = dst.add(w);
                (p as *mut u64).write_unaligned(ab);
                (p.add(2) as *mut u64).write_unaligned(ext);
            }
            w += if fast { (val & 0x7F) as usize } else { 0 };
            if !fast {
                // Cold: reconstruct the span from the entry. For key == 0
                // `h` is really the span length, but the slow path never
                // reads `h` on the long route (see probe_emit_slow), so it
                // passes through unfiltered — a select here got hoisted
                // into the hot loop as a per-pretoken cset.
                // SAFETY: entry `i` was written by this chunk's fill, so
                // `ptr` points at a live span of the input's lifetime.
                let bytes = unsafe { batch.span(i) };
                w = self.probe_emit_slow(bytes, key, h, out, w);
                dst = out.as_mut_ptr();
                table = self.pretoken_cache.probe_view();
            }
            record(i, w);
        }
        // SAFETY: w <= capacity by the slack invariant, and every element
        // below `w` was written (fast advances never skip lanes; the slow
        // path appends through Vec).
        unsafe { out.set_len(w) };
    }

    /// Everything [`Self::probe_emit_chunk`]'s fast predicate rejects.
    /// Appends this pretoken's tokens at cursor `w` and returns the new
    /// cursor, re-establishing the emit loop's slack invariant.
    ///
    /// `h` is only meaningful (and only read) when `key != 0`: the long
    /// route keys on `bytes` and passes literal zeros to the miss path.
    /// The emit loop relies on this and forwards the batch entry's `meta`
    /// (the span length when `key == 0`) without filtering it.
    #[cold]
    #[inline(never)]
    fn probe_emit_slow(
        &mut self,
        bytes: &[u8],
        key: u128,
        h: u64,
        out: &mut Vec<u32>,
        w: usize,
    ) -> usize {
        // SAFETY: elements below `w` are initialized and w <= capacity
        // (emit-loop invariant); Vec append methods need len in sync.
        unsafe { out.set_len(w) };
        if key != 0 {
            // A miss hands back the insert slot its walk found, so the
            // miss path's insert skips re-walking the (just-touched)
            // chain.
            match self.pretoken_cache.get_or_slot(key, h) {
                Ok((val, ext)) => {
                    let len = (val & 0x7F) as usize;
                    if val & VAL_SPILL == 0 {
                        out.extend_from_slice(&unpack_val_lanes(val, ext)[..len]);
                    } else {
                        let start = (val >> 32) as usize;
                        // SAFETY: recorded right after appending `len`
                        // tokens at `start`; the arena never shrinks.
                        let toks = unsafe { self.token_arena.get_unchecked(start..start + len) };
                        out.extend_from_slice(token_ids_as_u32s(toks));
                    }
                }
                Err(slot) => self.encode_pretoken_miss(bytes, key, h, slot, out),
            }
        } else {
            // Long pretokens (> 15 bytes, rare) always spill to the arena;
            // their token counts can exceed the packed-value range, so
            // they bypass it entirely.
            match self.pretoken_cache_long.get(bytes) {
                Some(&(offset, len)) => {
                    let start = offset as usize;
                    // SAFETY: as above.
                    let toks =
                        unsafe { self.token_arena.get_unchecked(start..start + len as usize) };
                    out.extend_from_slice(token_ids_as_u32s(toks));
                }
                None => self.encode_pretoken_miss(bytes, 0, 0, 0, out),
            }
        }
        out.reserve(4 * PRETOKEN_CHUNK);
        out.len()
    }

    /// Outlined miss path for rank-mapped vocabularies: the same cache
    /// bookkeeping as [`Self::encode_pretoken_miss`], with every merge
    /// running through the explicit-rank loops.
    #[cold]
    #[inline(never)]
    fn encode_pretoken_miss_ranked(
        &mut self,
        bytes: &[u8],
        key: u128,
        h: u64,
        slot: usize,
        out: &mut Vec<u32>,
    ) {
        let rm = self
            .ranked_merges
            .clone()
            .expect("caller checked ranked_merges");
        if key != 0 {
            let mut buf = [TokenId(0); SHORT_MERGE_MAX];
            let n = Self::seed_symbols_ranked(
                self.byte_remapping.as_ref(),
                &rm,
                self.ignore_merges,
                &self.vocab_inv,
                bytes,
                &mut buf,
            );
            let symbols = &buf[..n];
            let (val, ext) = Self::pack_val(symbols, &mut self.token_arena);
            self.pretoken_cache.insert_at(slot, key, h, val, ext);
            out.extend_from_slice(token_ids_as_u32s(symbols));
        } else {
            // Mirrors the long-pretoken arm of `encode_pretoken_miss`,
            // including the no-whole-pretoken-shortcut rule documented
            // there.
            let symbols = &mut self.symbol_scratch;
            symbols.clear();
            if self.ignore_merges
                && let Some(&id) = self.vocab_inv.get(bytes)
            {
                symbols.push(id);
            } else {
                match self.byte_remapping.as_ref() {
                    Some(br) => symbols.extend(bytes.iter().map(|&b| br.mapping[b as usize])),
                    None => symbols.extend(bytes.iter().map(|&b| TokenId::from(b as u32))),
                }
                bpe_merge_symbols_ranked(&rm, symbols);
            }
            let len = symbols.len() as u32;
            let offset = self.token_arena.len() as u32;
            self.token_arena.extend_from_slice(symbols);
            self.pretoken_cache_long.insert(bytes.into(), (offset, len));
            out.extend_from_slice(token_ids_as_u32s(symbols));
        }
    }

    /// Cache-miss path of the probe/emit loop: BPE-encode `bytes`, record
    /// it in the table `key` routes to (the short-pretoken table, or the
    /// long map when `key == 0`), and append its tokens to `out`. `slot`
    /// is the short-cache insert position reported by the failed
    /// `get_or_slot` probe (meaningful only when `key != 0`); nothing
    /// here touches the short cache before the insert, so it stays valid.
    #[inline(never)]
    fn encode_pretoken_miss(
        &mut self,
        bytes: &[u8],
        key: u128,
        h: u64,
        slot: usize,
        out: &mut Vec<u32>,
    ) {
        // Rank-mapped vocabularies take the outlined
        // ranked miss path; the branch is one perfectly-predicted test for
        // everything else, keeping this function's codegen identical to a
        // build without ranked support.
        if self.ranked_merges.is_some() {
            return self.encode_pretoken_miss_ranked(bytes, key, h, slot, out);
        }
        if key != 0 {
            // Short pretoken (≤ 15 bytes, the overwhelming majority of
            // misses): straight to byte symbols and the merge loop
            // (`merge_short`, shared with the vocab seed), in a stack
            // buffer instead of the `Vec` scratch. The cache is pre-seeded
            // with the seed encoding of every short vocab entry (see
            // `seeded_pretoken_cache`), so a miss here is never a whole
            // vocab word — but nothing depends on that: `seed_symbols`
            // computes the correct encoding for any bytes under either
            // `ignore_merges` setting.
            let mut buf = [TokenId(0); SHORT_MERGE_MAX];
            let n = Self::seed_symbols(
                self.byte_remapping.as_ref(),
                self.pair_ranks.as_deref(),
                &self.merges,
                self.ignore_merges,
                &self.vocab_inv,
                bytes,
                &mut buf,
            );
            let symbols = &buf[..n];
            let (val, ext) = Self::pack_val(symbols, &mut self.token_arena);
            self.pretoken_cache.insert_at(slot, key, h, val, ext);
            out.extend_from_slice(token_ids_as_u32s(symbols));
        } else {
            // Long pretoken (> 15 bytes, rare): remap and run the merge
            // loop. Without `ignore_merges`, deliberately NO
            // whole-pretoken reverse-vocab (`vocab_inv`) shortcut here — a
            // vocab entry is not guaranteed to be derivable from its own
            // merges (qwen3_5 has ~50 entries > 15 bytes, multi-char CJK
            // phrases, that HF `tokenizers` without `ignore_merges`
            // encodes as their merge decomposition, never as the single
            // ID), so any such shortcut diverges from HF and from the
            // pre-cache baseline. The same rule holds for short keys via
            // the seeded merge results above. Do not reintroduce it. With
            // `ignore_merges` set, the shortcut IS HF's semantics, so it
            // applies — gated on the flag.
            let symbols = &mut self.symbol_scratch;
            symbols.clear();
            if self.ignore_merges
                && let Some(&id) = self.vocab_inv.get(bytes)
            {
                symbols.push(id);
            } else {
                match self.byte_remapping.as_ref() {
                    Some(br) => symbols.extend(bytes.iter().map(|&b| br.mapping[b as usize])),
                    None => symbols.extend(bytes.iter().map(|&b| TokenId::from(b as u32))),
                }
                match self.pair_ranks.as_deref() {
                    Some(table) => bpe_merge_symbols_by_rank(
                        &|a, b| table.rank(a, b),
                        symbols,
                        &mut self.merge_scratch,
                    ),
                    None => bpe_merge_symbols_with_scratch(
                        &self.merges,
                        symbols,
                        &mut self.merge_scratch,
                    ),
                }
            }
            let len = symbols.len() as u32;
            let offset = self.token_arena.len() as u32;
            self.token_arena.extend_from_slice(symbols);
            self.pretoken_cache_long.insert(bytes.into(), (offset, len));
            out.extend_from_slice(token_ids_as_u32s(symbols));
        }
    }

    pub fn decode(&self, v: &[TokenId]) -> impl Iterator<Item = u8> {
        v.iter()
            .flat_map(|&token| self.vocab[token.0 as usize].as_ref())
            .copied()
    }

    /// Detailed cache stats for memory accounting (see examples/cache_memory.rs):
    /// (short_len, short_cap, long_len, long_cap, long_key_bytes, arena_len, arena_cap).
    pub fn cache_mem_stats(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
        let long_key_bytes: usize = self.pretoken_cache_long.keys().map(|k| k.len()).sum();
        (
            self.pretoken_cache.len(),
            self.pretoken_cache.capacity(),
            self.pretoken_cache_long.len(),
            self.pretoken_cache_long.capacity(),
            long_key_bytes,
            self.token_arena.len(),
            self.token_arena.capacity(),
        )
    }
}

impl Debug for Tokenizer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab.len())
            .field("merges_count", &self.merges.len())
            .field("pair_ranks", &self.pair_ranks.is_some())
            .field("byte_remapping", &self.byte_remapping.is_some())
            .finish()
    }
}
