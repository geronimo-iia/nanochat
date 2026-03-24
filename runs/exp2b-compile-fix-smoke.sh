#!/bin/bash
# Experiment 2b — mx.compile fix smoke test (d6, ~5 min on MLX)
# Validates that the mx.compile stale-params fix works:
#   - train/loss must decrease (was flat at 10.5 before fix)
#   - val/bpb must decrease or stay stable (was increasing before fix)
#   - compression_ratio and val/bpb aligned at every eval step
#
# Usage:
#   bash runs/exp2b-compile-fix-smoke.sh
#   tail -f /Users/geronimo/build/sp_theory/experiments/nanochat/exp2b.log

set -euo pipefail

BASE_DIR="/Users/geronimo/build/sp_theory/experiments/nanochat"
LOG_FILE="$BASE_DIR/exp2b.log"

cd "$(dirname "$0")/.."

echo "Starting Experiment 2b — log: $LOG_FILE"

PYTHONUNBUFFERED=1 nohup uv run nanochat \
    --config "$BASE_DIR/config.toml" \
    --backend=mlx \
    --wandb=local \
    --run=compile-fix-smoke-d6 \
    train base \
    --depth=6 \
    --aspect-ratio=64 \
    --head-dim=128 \
    --max-seq-len=1024 \
    --device-batch-size=64 \
    --total-batch-size=524288 \
    --num-iterations=20 \
    --track-compression \
    --compression-log-every=10 \
    --eval-every=10 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Monitor: tail -f $LOG_FILE"
