#!/usr/bin/env sh
sed -i '' 's/document.baseURI||//g' dist/worker.js
