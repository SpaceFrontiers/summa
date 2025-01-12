#!/usr/bin/env bash
#
# Build WASM, install dependencies, and launch dev server
# ========================================================
#
# Usage:
#   ./scripts/dev.sh              # Build wasm + launch dev server
#   ./scripts/dev.sh --skip-wasm  # Skip wasm build, just launch dev server
#
# Environment variables:
#   SKIP_WASM  - Set to 1 to skip WASM build

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WEB_DIR/.." && pwd)"
WASM_DIR="$PROJECT_ROOT/hermes-wasm"

cd "$WEB_DIR"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() {
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

check_prerequisites() {
    local missing=0
    
    if ! command -v pnpm &> /dev/null; then
        log_error "pnpm not found. Install with: npm install -g pnpm"
        missing=1
    fi
    
    if ! command -v wasm-pack &> /dev/null; then
        log_error "wasm-pack not found. Install with: cargo install wasm-pack"
        missing=1
    fi
    
    if [[ $missing -eq 1 ]]; then
        exit 1
    fi
}

build_wasm() {
    log_step "Building WASM package"
    
    cd "$WASM_DIR"
    ./build.sh
    log_success "WASM package built at $WASM_DIR/pkg"
    cd "$WEB_DIR"
}

install_deps() {
    log_step "Installing dependencies"
    pnpm install
    log_success "Dependencies installed"
}

start_dev() {
    log_step "Starting dev server"
    
    echo ""
    echo -e "  Dev Server: ${BLUE}http://localhost:5173${NC}"
    echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop"
    echo ""
    
    pnpm dev
}

# Main
SKIP_WASM="${SKIP_WASM:-0}"

case "${1:-}" in
    --skip-wasm)
        SKIP_WASM=1
        ;;
    --help|-h)
        echo "Usage: $0 [--skip-wasm]"
        echo ""
        echo "Build WASM, install dependencies, and launch dev server."
        echo ""
        echo "Options:"
        echo "  --skip-wasm  Skip WASM build"
        exit 0
        ;;
esac

check_prerequisites

if [[ "$SKIP_WASM" != "1" ]]; then
    build_wasm
fi

install_deps
start_dev
