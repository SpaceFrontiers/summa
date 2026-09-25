# Tokenizer backend compatibility

Summa treats `tokenizer.json` and its token IDs as part of the model artifact
contract. Changing the tokenizer implementation is safe for an existing
checkpoint only when encoding, decoding, added-token handling, and vocabulary
lookups remain identical.

## Current backend

`summa-llm` and `summa-train` use the first-party `summa-tokenizer` crate;
neither executable depends on Python Transformers or Hugging Face's Rust
`tokenizers` package. The shared wrapper loads a local `tokenizer.json`,
resolves EOS, encodes individual prompts and document batches, decodes
generated IDs, and exposes exact and display-friendly vocabulary pieces.

`summa-core` has a wider contract. It supports native Hugging Face Hub and
local loading, byte loading for index-contained tokenizers, WordPiece models
used by sparse retrieval, and a WASM build. A replacement for the LLM trainer
therefore does not automatically qualify as a replacement for `summa-core`.

## GigaToken evaluation — 2026-07-22

[GigaToken](https://github.com/marcelroed/gigatoken) 0.9.0 was evaluated
against the tokenizer from the retriever-100M training run. That tokenizer is a
GPT-2-style byte-level BPE with NFC normalization, ByteLevel pre-tokenization
and decoding, EOS and padding tokens, and added whitespace-run tokens.

Compatibility checks passed:

- GigaToken and Hugging Face produced identical token IDs for all 80 sampled
  Rust, Markdown, and JSON documents (0.87 MiB).
- Thirteen targeted cases covered empty input, English, Russian, Arabic, CJK,
  emoji, NFC/NFD forms, tabs, line endings, control bytes, long repeated input,
  whitespace-run tokens, and literal/embedded EOS tokens. Native and
  Hugging-Face-compatible GigaToken APIs both matched exactly.
- Decode behavior, special-token skipping, and ID-to-token lookup matched for
  81 checks. The resulting vocabulary size was 50,277 in both implementations.

On the same small document sample, GigaToken processed 22.71 MiB/s and Hugging
Face processed 16.33 MiB/s, a 1.39x tokenizer-only speedup. This is not an
end-to-end trainer result. GigaToken's much larger published gains use its
file-native API on multi-gigabyte inputs; its compatibility API has measurable
overhead. Summa currently parses JSONL itself, passes batches of strings to
the tokenizer, and reuses a persistent causal-token cache after the first
pass, so the published headline speedup does not transfer directly.

## Current extraction

The extraction has since been refreshed to GigaToken 0.10.0 at commit
`34a1599f0c0ae7d7cd0d1c530e6522320158b360`. See the
[provenance register](../summa-tokenizer/UPSTREAM.md) for the current revision
and refresh policy. The July evaluation above measured 0.9.0; it is not a
new measurement of the current extraction.

### Initial extraction — 2026-07-22

The optimized byte-level BPE engine, persistent pretoken cache, merge kernels,
and fast pretokenizers were extracted into `summa-tokenizer` from GigaToken
0.9.0 commit `542367a3efed134883fb4f1140b49c04e6fad3a3`. It is a narrow stable-Rust
crate, not a copy of the full application: Python bindings, CLI, data loaders,
Arrow/Parquet, networking, training code, reference implementations,
SentencePiece, and upstream tests tied to those components are excluded. The
upstream MIT notice and exact refresh revision are recorded in the crate.

The loader accepts current Hugging Face JSON byte-level BPE artifacts and
fails at load time for unsupported model or pipeline features. The migration
keeps `tokenizer.json` and its IDs canonical; checkpoints and token caches do
not need conversion. Differential tests cover generated fixtures, parallel
batch ordering, added and special tokens, Unicode normalization, decode and
vocabulary operations, plus targeted and deterministic randomized cases
against the step-19,000 tokenizer.

The extraction is intentionally limited to LLM training and inference:

- `summa-core` retains Hugging Face `tokenizers` for WordPiece sparse
  retrieval, Hub loading, index-contained byte artifacts, and WASM.
- SentencePiece/Unigram and BPE byte fallback are rejected rather than pulling
  their nightly SIMD path into the stable build.
- New upstream code is reviewed and recorded as an explicit revision refresh;
  the extraction is not advanced independently as a general GigaToken fork.

The relevant upstream work is tracked in GigaToken issues
[#26](https://github.com/marcelroed/gigatoken/issues/26),
[#27](https://github.com/marcelroed/gigatoken/issues/27), and
[#30](https://github.com/marcelroed/gigatoken/issues/30).

An end-to-end first-pass curriculum benchmark should still measure accelerator
utilization and training tokens per second; the 1.39x number above is a
tokenizer-only result. Keep the Hugging Face backend in `summa-core` until a
replacement also meets its WordPiece and WASM contracts.
