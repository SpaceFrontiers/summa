#!/usr/bin/env bash

set -e

PATH="/opt/homebrew/opt/llvm/bin/:$PATH" \
  CC=/opt/homebrew/opt/llvm/bin/clang \
  AR=/opt/homebrew/opt/llvm/bin/llvm-ar \
  wasm-pack build --target web --release
