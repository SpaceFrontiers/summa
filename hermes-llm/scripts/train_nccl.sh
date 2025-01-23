#!/bin/bash
# NCCL-based distributed training launcher for hermes-llm
# Usage: ./scripts/train_nccl.sh <num_gpus> <data_file> <tokenizer> [model]
#
# This script launches multiple processes that communicate via NCCL
# for true gradient synchronization across GPUs.

set -e

NUM_GPUS=${1:-4}
DATA_FILE=${2:-"corpus.jsonl"}
TOKENIZER=${3:-"tok.json"}
MODEL=${4:-"gpt2-small"}
OUTPUT_DIR="checkpoints"
COMM_FILE="nccl_id.txt"

echo "=== NCCL Distributed Training ==="
echo "GPUs: $NUM_GPUS"
echo "Data: $DATA_FILE"
echo "Tokenizer: $TOKENIZER"
echo "Model: $MODEL"
echo ""

# Clean up any existing comm file
rm -f "$COMM_FILE"

# Build with CUDA and NCCL support
echo "Building with CUDA + NCCL support..."
cargo build --release -p hermes-llm --features nccl

# Calculate batch size and gradient accumulation
# With NCCL, effective batch = batch_size * grad_accum * num_gpus
BATCH_SIZE=32
GRAD_ACCUM=4
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))
echo "Effective batch size: $EFFECTIVE_BATCH (${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_GPUS})"
echo ""

# Launch training on each GPU
PIDS=()
for RANK in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching rank $RANK on GPU $RANK..."

    CUDA_VISIBLE_DEVICES=$RANK cargo run --release -p hermes-llm --features nccl -- train \
        --data "$DATA_FILE" \
        --tokenizer "$TOKENIZER" \
        --model "$MODEL" \
        --output "$OUTPUT_DIR" \
        --batch-size $BATCH_SIZE \
        --grad-accum $GRAD_ACCUM \
        --epochs 1 \
        --world-size $NUM_GPUS \
        --rank $RANK \
        --comm-file "$COMM_FILE" \
        2>&1 | sed "s/^/[Rank $RANK] /" &

    PIDS+=($!)

    # Small delay to ensure rank 0 creates comm file first
    if [ $RANK -eq 0 ]; then
        sleep 2
    fi
done

echo ""
echo "Training started on $NUM_GPUS GPUs with NCCL. PIDs: ${PIDS[*]}"
echo "Waiting for all processes to complete..."

# Wait for all processes
FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait $PID; then
        FAILED=1
    fi
done

# Clean up comm file
rm -f "$COMM_FILE"

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== Training complete ==="
    echo "Checkpoint saved to: $OUTPUT_DIR/"
else
    echo "=== Training failed ==="
    exit 1
fi
