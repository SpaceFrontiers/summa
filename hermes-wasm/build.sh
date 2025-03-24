#!/usr/bin/env bash

set -e

# Set up LLVM tools for zstd cross-compilation to wasm32
if [ -d "/opt/homebrew/opt/llvm/bin" ]; then
  # macOS with Homebrew LLVM
  export PATH="/opt/homebrew/opt/llvm/bin/:$PATH"
elif [ -d "/usr/lib/llvm-18/bin" ]; then
  export PATH="/usr/lib/llvm-18/bin:$PATH"
elif [ -d "/usr/lib/llvm-17/bin" ]; then
  export PATH="/usr/lib/llvm-17/bin:$PATH"
fi

# Use llvm-ar if available, fall back to system ar
if command -v llvm-ar &>/dev/null; then
  export AR=llvm-ar
fi

export CC="${CC:-clang}"

wasm-pack build --target web --release

# Patch the generated JS to support Node.js environments (vitest, etc.)
# Node.js fetch() doesn't support file:// URLs, so we detect Node.js and use
# fs.readFile instead. See: https://github.com/nicolo-ribaudo/tc39-proposal-fetch-file
patch pkg/hermes_wasm.js < ./patches/fetch.patch
