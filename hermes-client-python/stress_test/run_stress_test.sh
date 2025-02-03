#!/bin/bash
# Hermes Stress Test Runner
#
# This script starts the Hermes server with tracing enabled and runs the stress test.
# It collects server logs to analyze reload frequency and segment build patterns.
#
# Usage:
#   ./run_stress_test.sh [options]
#
# Options:
#   --docs N              Number of documents to index (default: 50000)
#   --batch-size N        Documents per batch (default: 500)
#   --index-workers N     Parallel indexing workers (default: 4)
#   --search-workers N    Parallel search workers (default: 4)
#   --search-qps N        Target queries per second (default: 50)
#   --duration N          Test duration in seconds (default: 120)
#   --reload-interval N   Server reload interval in ms (default: 1000)
#   --max-memory N        Max indexing memory in MB (default: 4096)
#   --indexing-threads N  Server indexing threads (default: 4)
#   --keep-server         Don't stop server after test
#   --server-only         Start server only, don't run test

set -e

# Default values
DOCS=50000
BATCH_SIZE=500
INDEX_WORKERS=4
SEARCH_WORKERS=4
SEARCH_QPS=50
DURATION=120
RELOAD_INTERVAL_MS=1000
MAX_MEMORY_MB=4096
INDEXING_THREADS=4
KEEP_SERVER=false
SERVER_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --docs)
            DOCS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --index-workers)
            INDEX_WORKERS="$2"
            shift 2
            ;;
        --search-workers)
            SEARCH_WORKERS="$2"
            shift 2
            ;;
        --search-qps)
            SEARCH_QPS="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --reload-interval)
            RELOAD_INTERVAL_MS="$2"
            shift 2
            ;;
        --max-memory)
            MAX_MEMORY_MB="$2"
            shift 2
            ;;
        --indexing-threads)
            INDEXING_THREADS="$2"
            shift 2
            ;;
        --keep-server)
            KEEP_SERVER=true
            shift
            ;;
        --server-only)
            SERVER_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Create data directory
DATA_DIR="$PROJECT_ROOT/data"
mkdir -p "$DATA_DIR"

# Log file for server output
LOG_FILE="$SCRIPT_DIR/server.log"

echo "========================================"
echo "Hermes Stress Test"
echo "========================================"
echo "Server Configuration:"
echo "  Max indexing memory: ${MAX_MEMORY_MB} MB"
echo "  Indexing threads:    ${INDEXING_THREADS}"
echo "  Reload interval:     ${RELOAD_INTERVAL_MS} ms"
echo ""
echo "Test Configuration:"
echo "  Documents:           ${DOCS}"
echo "  Batch size:          ${BATCH_SIZE}"
echo "  Index workers:       ${INDEX_WORKERS}"
echo "  Search workers:      ${SEARCH_WORKERS}"
echo "  Target QPS:          ${SEARCH_QPS}"
echo "  Duration:            ${DURATION}s"
echo "========================================"
echo ""

# Build the server
echo "Building hermes-server..."
cargo build --release -p hermes-server --manifest-path "$PROJECT_ROOT/Cargo.toml" 2>&1 | tail -5

# Start server in background with tracing
echo ""
echo "Starting server (logs: $LOG_FILE)..."
RUST_LOG=info "$PROJECT_ROOT/target/release/hermes-server" \
    --data-dir "$DATA_DIR" \
    --max-indexing-memory-mb "$MAX_MEMORY_MB" \
    --indexing-threads "$INDEXING_THREADS" \
    --reload-interval-ms "$RELOAD_INTERVAL_MS" \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
sleep 2

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server failed to start. Check logs:"
    tail -20 "$LOG_FILE"
    exit 1
fi

if [ "$SERVER_ONLY" = true ]; then
    echo ""
    echo "Server running. PID: $SERVER_PID"
    echo "To stop: kill $SERVER_PID"
    exit 0
fi

# Run stress test
echo ""
echo "Running stress test..."
cd "$PROJECT_ROOT/hermes-client-python"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python -m stress_test.main \
    --docs "$DOCS" \
    --batch-size "$BATCH_SIZE" \
    --index-workers "$INDEX_WORKERS" \
    --search-workers "$SEARCH_WORKERS" \
    --search-qps "$SEARCH_QPS" \
    --duration "$DURATION"

TEST_EXIT_CODE=$?

# Analyze server logs
echo ""
echo "========================================"
echo "Server Log Analysis"
echo "========================================"

# Count reload events
RELOADS=$(grep -c "\[index_reload\]" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Index reloads:     $RELOADS"

# Count segment builds
BUILDS_STARTED=$(grep -c "\[segment_build_started\]" "$LOG_FILE" 2>/dev/null || echo "0")
BUILDS_COMPLETED=$(grep -c "\[segment_build_completed\]" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Segments built:    $BUILDS_COMPLETED / $BUILDS_STARTED started"

# Show some build timing stats
if [ "$BUILDS_COMPLETED" -gt 0 ]; then
    echo ""
    echo "Last 5 segment builds:"
    grep "\[segment_build_completed\]" "$LOG_FILE" | tail -5 | sed 's/^/  /'
fi

echo "========================================"

# Stop server if not keeping
if [ "$KEEP_SERVER" = false ]; then
    echo ""
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Server stopped."
else
    echo ""
    echo "Server still running. PID: $SERVER_PID"
    echo "To stop: kill $SERVER_PID"
fi

exit $TEST_EXIT_CODE
