#!/bin/bash
# Experiment 1b — Safetensors Checkpoint Smoke Test (d6, ~5 min on MLX)
# Verifies safetensors save and resume work end-to-end with MLX backend.
# See docs/dev/phase-1.5.1-validation-checklist.md for details.
#
# Usage:
#   bash runs/exp1b-safetensors-smoke.sh

set -euo pipefail

BASE_DIR="/Users/geronimo/build/sp_theory/experiments/nanochat"
LOG_FILE="$BASE_DIR/exp1b.log"

cd "$(dirname "$0")/.."

# Clean up any previous run
rm -f "$BASE_DIR/checkpoints/base/d6/model_"* \
       "$BASE_DIR/checkpoints/base/d6/meta_"* \
       "$BASE_DIR/checkpoints/base/d6/optim_"* \
       "$BASE_DIR/checkpoints/base/d6/config.toml" \
       "$LOG_FILE" 2>/dev/null || true

echo "=== Phase 1: run 10 steps, save at step 5 ==="
PYTHONUNBUFFERED=1 uv run nanochat \
    --config "$BASE_DIR/config.toml" \
    --backend=mlx --wandb=disabled --run=exp1b-safetensors \
    train base \
    --depth=6 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=64 --total-batch-size=524288 \
    --num-iterations=10 \
    --save-every=5 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=== Checkpoint files ==="
ls -lh "$BASE_DIR/checkpoints/base/d6/"

echo ""
echo "=== Phase 2: resume from step 5, run to step 10 ==="
PYTHONUNBUFFERED=1 uv run nanochat \
    --config "$BASE_DIR/config.toml" \
    --backend=mlx --wandb=disabled --run=exp1b-safetensors \
    train base \
    --depth=6 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=64 --total-batch-size=524288 \
    --num-iterations=10 \
    --resume-from-step=5 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=== Done — log: $LOG_FILE ==="
