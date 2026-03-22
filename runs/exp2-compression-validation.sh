#!/bin/bash
# Experiment 2 — Compression Validation (d6, ~5h on MLX)
# Collects compression_ratio and val/bpb data points for correlation analysis.
# See docs/dev/phase-1.5.1-validation-checklist.md for details.
#
# Usage:
#   bash runs/exp2-compression-validation.sh
#   tail -f /Users/geronimo/build/sp_theory/experiments/nanochat/exp2.log

set -euo pipefail

BASE_DIR="/Users/geronimo/build/sp_theory/experiments/nanochat"
LOG_FILE="$BASE_DIR/exp2.log"

cd "$(dirname "$0")/.."

echo "Starting Experiment 2 — log: $LOG_FILE"

nohup uv run nanochat \
    --config "$BASE_DIR/config.toml" \
    --backend=mlx \
    --wandb=local \
    --run=compression-validation-d6 \
    train base \
    --depth=6 \
    --aspect-ratio=64 \
    --head-dim=128 \
    --max-seq-len=1024 \
    --device-batch-size=64 \
    --total-batch-size=524288 \
    --track-compression \
    --compression-log-every=50 \
    --eval-every=250 \
    --eval-tokens=5242880 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Monitor: tail -f $LOG_FILE"
