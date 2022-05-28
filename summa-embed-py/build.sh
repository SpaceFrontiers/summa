#!/usr/bin/env bash

pip3 install -U build grpcio-tools twine
rm -rf summa_embed/proto/*
touch summa_embed/proto/__init__.py
python3 -m grpc_tools.protoc -I../summa-proto/proto --python_out=summa_embed/proto --pyi_out=summa_embed/proto ../summa-proto/proto/*.proto
sed -i '' 's/^\(import.*pb2\)/from . \1/g' summa_embed/proto/*.py
python3 -m build