#!/usr/bin/env bash

#./node_modules/protobufjs-cli/bin/pbjs --filter proto.filter.json --es6 --no-encode --no-decode --no-verify --no-convert --no-delimited --no-typeurl --no-service  -w es6 --keep-case -t static-module -o ./src/proto.js -p ../summa-proto/proto/ ../summa-proto/proto/search_service.proto ../summa-proto/proto/index_service.proto
#./node_modules/protobufjs-cli/bin/pbts src/proto.js -o src/proto.d.ts


PATH="/usr/local/opt/llvm/bin/:$PATH" CC=/usr/local/opt/llvm/bin/clang AR=/usr/local/opt/llvm/bin/llvm-ar npm run build
sed -i '' 's/document.baseURI ||//g' dist/root-worker.js
sed -i '' 's/document.baseURI||//g' dist/root-worker.js
