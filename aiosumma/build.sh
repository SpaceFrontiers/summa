#!/usr/bin/env bash

pip3 install -U grpcio-tools twine
rm -rf aiosumma/proto/*
touch aiosumma/proto/__init__.py
python3 -m grpc_tools.protoc -I../summa-proto/proto --python_out=aiosumma/proto --pyi_out=aiosumma/proto --grpc_python_out=aiosumma/proto ../summa-proto/proto/*.proto
sed -i '' 's/^\(import.*pb2\)/from . \1/g' aiosumma/proto/*.py
python3 -m build