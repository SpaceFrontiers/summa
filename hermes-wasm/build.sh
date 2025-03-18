#!/usr/bin/env bash

set -e

PATH="/opt/homebrew/opt/llvm/bin/:$PATH" \
  CC=/opt/homebrew/opt/llvm/bin/clang \
  AR=/opt/homebrew/opt/llvm/bin/llvm-ar \
  wasm-pack build --target web --release

# Patch the generated JS to support Node.js environments (vitest, etc.)
# Node.js fetch() doesn't support file:// URLs, so we detect Node.js and use
# fs.readFile instead. See: https://github.com/nicolo-ribaudo/tc39-proposal-fetch-file
node -e "
const fs = require('fs');
const path = require('path');
const file = path.join(__dirname, 'pkg/hermes_wasm.js');
let code = fs.readFileSync(file, 'utf8');
code = code.replace(
  'module_or_path = fetch(module_or_path);',
  \`if (typeof globalThis.process !== 'undefined' && globalThis.process.versions?.node) {
            const { readFile } = await import('node:fs/promises');
            const { fileURLToPath } = await import('node:url');
            const url = module_or_path instanceof URL ? module_or_path : new URL(module_or_path);
            module_or_path = url.protocol === 'file:' ? readFile(fileURLToPath(url)) : fetch(module_or_path);
        } else {
            module_or_path = fetch(module_or_path);
        }\`
);
fs.writeFileSync(file, code);
console.log('Patched hermes_wasm.js for Node.js compatibility');
"
