#!/usr/bin/env sh

rm -rf pkg/*
wasm-pack build --release --target web --out-name index
cp package-pkg.json pkg/package.json
cp src/*.ts pkg/
jq ".version = \"$1\"" <<< cat package-pkg.json > pkg/package.json