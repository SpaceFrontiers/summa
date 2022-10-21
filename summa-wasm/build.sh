#!/usr/bin/env bash

npx protoc \
  --ts_out src/grpc-web \
  --ts_opt use_proto_field_name \
  --proto_path ../summa-proto/proto \
  ../summa-proto/proto/*.proto


PATH="/usr/local/opt/llvm/bin/:$PATH" CC=/usr/local/opt/llvm/bin/clang AR=/usr/local/opt/llvm/bin/llvm-ar npm run build
sed -i '' 's/document.baseURI ||//g' dist/root-worker.js
sed -i '' 's/document.baseURI||//g' dist/root-worker.js
