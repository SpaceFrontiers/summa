#!/usr/bin/env sh
sed -i '' 's/document.currentScript&&document.currentScript.src||document.baseURI/location.href/g' dist/assets/searcher.*.js