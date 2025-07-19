# Tokenizer backend compatibility

Hermes treats `tokenizer.json` and its token IDs as part of the model artifact
contract. Changing the tokenizer implementation is safe for an existing
checkpoint only when encoding, decoding, added-token handling, and vocabulary
lookups remain identical.

## Current backend

`hermes-llm` and `hermes-train` use Hugging Face's Rust `tokenizers` crate
directly; neither executable depends on Python Transformers. The shared wrapper
loads a local `tokenizer.json`, resolves EOS, encodes individual prompts and
1,000-document training batches, decodes generated IDs, and exposes exact and
display-friendly vocabulary pieces.

`hermes-core` has a wider contract. It supports native Hugging Face Hub and
local loading, byte loading for index-contained tokenizers, WordPiece models
used by sparse retrieval, and a WASM build. A replacement for the LLM trainer
therefore does not automatically qualify as a replacement for `hermes-core`.

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
overhead. Hermes currently parses JSONL itself, passes batches of strings to
the tokenizer, and reuses a persistent causal-token cache after the first
pass, so the published headline speedup does not transfer directly.

## Adoption status

Do not vendor or fork GigaToken, and do not replace Hugging Face repository-wide
at this time:

- The Rust package currently requires nightly Rust, while Hermes uses stable
  Rust 1.97.
- There is no versioned Rust crate on crates.io. The current package also pulls
  Python bindings, Arrow/Parquet, networking, and SentencePiece dependencies
  into a Rust build that only needs byte-level BPE.
- WordPiece is not supported, which excludes current sparse-retrieval
  tokenizers in `hermes-core`.
- There is no compatible WASM backend for the `hermes-core` byte-loading path.
- Inference tokenizes only a prompt at a time, and resumed training reads the
  existing token cache, so neither path is likely to benefit materially.

The relevant upstream work is tracked in GigaToken issues
[#26](https://github.com/marcelroed/gigatoken/issues/26),
[#27](https://github.com/marcelroed/gigatoken/issues/27), and
[#30](https://github.com/marcelroed/gigatoken/issues/30).