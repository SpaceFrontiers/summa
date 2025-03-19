#!/usr/bin/env bash

set -e

PATH="/opt/homebrew/opt/llvm/bin/:$PATH" \
  CC=clang \
  AR=llvm-ar \
  wasm-pack build --target web --release

# Patch the generated JS to support Node.js environments (vitest, etc.)
# Node.js fetch() doesn't support file:// URLs, so we detect Node.js and use
# fs.readFile instead. See: https://github.com/nicolo-ribaudo/tc39-proposal-fetch-file
patch pkg/hermes_wasm.js < ./patches/fetch.patch
