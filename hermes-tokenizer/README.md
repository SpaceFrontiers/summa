# hermes-tokenizer

`hermes-tokenizer` is Hermes' stable-Rust byte-level BPE tokenizer. It loads a
local Hugging Face `tokenizer.json`, keeps the artifact's token IDs unchanged,
and exposes the operations used by training, inference, and model
visualization:

```rust
let tokenizer = hermes_tokenizer::Tokenizer::from_file("tokenizer.json")?;
let ids = tokenizer.encode("A retrieval query", false)?;
let text = tokenizer.decode(&ids, true)?;
```

The current compatibility contract is intentionally narrow: byte-level BPE,
NFC normalization, supported GigaToken split-regex families, Hugging Face
added-token whitespace behavior, and the single-sequence parts of ByteLevel or
TemplateProcessing post-processors. Unsupported model and pipeline types fail
at load time instead of silently producing different token IDs.

The optimized merge engine and pretokenizers are extracted from GigaToken and
run without Python, networking, Arrow, SentencePiece, or nightly Rust. See
[`UPSTREAM.md`](UPSTREAM.md) for exact provenance and refresh policy.
