#!/bin/bash
# Batch size sweep — compare tok/sec for B=64, B=128, B=256 on d6/MLX.
# Total batch size fixed at 524,288 tokens so training dynamics are identical.
# Grad accum steps: B=64→8, B=128→4, B=256→2.
#
# Usage:
#   bash runs/bench-batch-size.sh
#   tail -f /Users/geronimo/build/sp_theory/experiments/nanochat/bench-batch-size.log
#
# Reads tok/sec from steps 2-5 (after compile warmup at step 0) for each config.

set -euo pipefail

BASE_DIR="/Users/geronimo/build/sp_theory/experiments/nanochat"
LOG_FILE="$BASE_DIR/bench-batch-size.log"

cd "$(dirname "$0")/.."

run_bench() {
    local bsz=$1
    local label="B${bsz}"
    local run_log="$BASE_DIR/bench-${label}.log"

    echo "" | tee -a "$LOG_FILE"
    echo "=== $label (device-batch-size=$bsz) ===" | tee -a "$LOG_FILE"

    PYTHONUNBUFFERED=1 uv run nanochat \
        --config "$BASE_DIR/config.toml" \
        --backend=mlx \
        --wandb=local \
        --run="bench-bsz-${label}" \
        train base \
        --depth=6 \
        --aspect-ratio=64 \
        --head-dim=128 \
        --max-seq-len=1024 \
        --device-batch-size="$bsz" \
        --total-batch-size=524288 \
        --num-iterations=8 \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --eval-every=-1 \
        2>&1 | tee "$run_log"

    # Extract tok/sec from steps 2-7 (skip step 0 compile warmup, step 1 jit warmup)
    echo "--- tok/sec (steps 2-7) ---" | tee -a "$LOG_FILE"
    grep "tok/sec" "$run_log" | tail -n +3 | grep -o "tok/sec: [0-9,]*" | tee -a "$LOG_FILE"
}

echo "Batch size sweep — $(date)" > "$LOG_FILE"
echo "Fixed total-batch-size=524288, d6, MLX $(uv run python -c 'import mlx.core as mx; print(mx.__version__)' 2>/dev/null)" >> "$LOG_FILE"

run_bench 64
run_bench 128
run_bench 256

echo "" | tee -a "$LOG_FILE"
echo "Done. Full results: $LOG_FILE" | tee -a "$LOG_FILE"
