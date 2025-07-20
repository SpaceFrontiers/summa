# Upstream provenance

The optimized byte-level BPE engine, pretoken cache, merge kernels, and fast
pretokenizers under `src/bpe/` and `src/pretokenize/` were extracted from
[`marcelroed/gigatoken`](https://github.com/marcelroed/gigatoken) commit
`542367a3efed134883fb4f1140b49c04e6fad3a3` (version 0.9.0).

The extraction removes the Python bindings, CLI, file/data-source layer,
tokenizer trainer, reference pretokenizers, network loading, and SentencePiece
engine. SentencePiece is the only upstream path that required nightly
`portable_simd`; the retained byte-level BPE path builds on Hermes' stable Rust
toolchain. `src/hf.rs`, the public wrapper in `src/lib.rs`, and the differential
tests are Hermes integration code.

Substantial upstream code remains under the original MIT terms reproduced in
`LICENSE-GIGATOKEN`. When refreshing the extraction, compare against upstream
first and record the new commit here instead of creating an untracked fork.
