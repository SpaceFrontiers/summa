#!/usr/bin/env bash

# docker pull pseudomuto/protoc-gen-doc
docker run --rm -v $(pwd)/markdown.tmpl:/tmp/markdown.tmpl \
-v $(pwd)/../docs/apis:/out \
-v $(pwd)/proto:/protos \
pseudomuto/protoc-gen-doc --doc_opt=/tmp/markdown.tmpl,grpc-api.md