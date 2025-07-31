# Upstream provenance

The optimized byte-level BPE engine, pretoken cache, merge kernels, and fast
pretokenizers under `src/bpe/` and `src/pretokenize/` were extracted from
[`marcelroed/gigatoken`](https://github.com/marcelroed/gigatoken) commit
`34a1599f0c0ae7d7cd0d1c530e6522320158b360` (version 0.10.0).

The 0.10.0 refresh adds batch special-token registration so a loader rebuilds
the Aho-Corasick matcher and cache overwrites once per batch rather than once
per token. It also makes the supported pretokenizer scheme names a single
source of truth.

Upstream 0.10.0 also fixes raw `.tiktoken` loading: rank files do not contain
their split regex or special-token definitions, so those values must be
selected by the caller rather than guessed. Hermes deliberately does not load
raw rank files. Its local `tokenizer.json` loader reads and validates the
artifact's pretokenizer and added-token metadata, so the faulty upstream path
was never reachable here; scheme-sensitive differential cases guard that
contract.

The extraction removes the Python bindings, CLI, file/data-source layer,
tokenizer trainer, reference pretokenizers, network loading, and SentencePiece
engine. SentencePiece is the only upstream path that required nightly
`portable_simd`; the retained byte-level BPE path builds on Hermes' stable Rust
toolchain. `src/hf.rs`, the public wrapper in `src/lib.rs`, and the differential
tests are Hermes integration code.

Substantial upstream code remains under the original MIT terms reproduced in
`LICENSE-GIGATOKEN`. When refreshing the extraction, compare against upstream
first and record the new commit here instead of creating an untracked fork.
