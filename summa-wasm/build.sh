#!/usr/bin/env bash
npm run build
sed -i '' 's/document.baseURI ||//g' dist/worker.js
sed -i '' 's/document.baseURI||//g' dist/worker.js
