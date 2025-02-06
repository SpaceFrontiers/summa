#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROTO_DIR="$SCRIPT_DIR/../hermes-proto"
OUT_DIR="$SCRIPT_DIR/src/generated"

mkdir -p "$OUT_DIR"

npx grpc_tools_node_protoc \
  --plugin="protoc-gen-ts_proto=$SCRIPT_DIR/node_modules/.bin/protoc-gen-ts_proto" \
  --ts_proto_out="$OUT_DIR" \
  --ts_proto_opt=outputServices=generic-definitions \
  --ts_proto_opt=outputClientImpl=false \
  --ts_proto_opt=esModuleInterop=true \
  --ts_proto_opt=forceLong=number \
  --ts_proto_opt=snakeToCamel=true \
  --ts_proto_opt=useExactTypes=false \
  -I "$PROTO_DIR" \
  "$PROTO_DIR/hermes.proto"

echo "Generated TypeScript proto stubs in $OUT_DIR"
