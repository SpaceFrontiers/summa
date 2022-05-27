#!/usr/bin/env bash
LLVM_PATH="/usr/local/opt/llvm" # or any other path
LLVM_VERSION="16"
export PATH="$LLVM_PATH/bin:$PATH"
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
export LD_LIBRARY_PATH="$LLVM_PATH/lib/:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="$LLVM_PATH/lib/:$DYLD_LIBRARY_PATH"
export CPATH="$LLVM_PATH/lib/clang/$LLVM_VERSION/include/"
export LDFLAGS="-L$LLVM_PATH/lib"
export CPPFLAGS="-I$LLVM_PATH/include"
export CC="$LLVM_PATH/bin/clang"
export CXX="$LLVM_PATH/bin/clang++"
npm run build
sed -i '' 's/document.baseURI ||//g' dist/root-worker.js
sed -i '' 's/document.baseURI||//g' dist/root-worker.js
