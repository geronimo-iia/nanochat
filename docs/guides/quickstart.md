---
title: "Quickstart"
summary: "Get nanochat running from scratch: environment setup, data download, tokenizer training, and first training run."
read_when:
  - Setting up nanochat for the first time
  - Running on a new machine or GPU node
status: active
last_updated: "2026-06-14"
---

# Quickstart

## 1. Environment

```bash
# Install uv if not already installed
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtualenv and install dependencies
uv venv
uv sync --extra gpu        # GPU node
# uv sync --extra cpu      # CPU / Apple Silicon

source .venv/bin/activate
```

Set your base directory (where data, tokenizer, and checkpoints will be stored):

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
```

Pass it explicitly with `--base-dir $NANOCHAT_BASE_DIR` on every command, or drop a `config.toml` in that directory (see Step 2) so it is auto-discovered. See [data-layout.md](../data-layout.md) for the full directory structure.

## 2. Config

Generate a config file directly in your base directory:

```bash
nanochat --base-dir $NANOCHAT_BASE_DIR config init
```

This writes `$NANOCHAT_BASE_DIR/config.toml` with `base_dir` pre-filled. All subsequent commands will auto-discover it when you pass `--base-dir`.

See [configuration.md](../configuration.md) for all fields.

## 3. Data

Download training shards. Each shard is ~100MB of compressed text.

```bash
nanochat --base-dir $NANOCHAT_BASE_DIR data download -n 8      # ~800MB, enough for tokenizer training
nanochat --base-dir $NANOCHAT_BASE_DIR data download -n 170    # ~17GB, enough for GPT-2 capability pretraining
```

The last shard is always reserved as the validation split.

## 4. Tokenizer

Train the BPE tokenizer on the downloaded data:

```bash
# Quick test (~2 seconds, 50M characters)
nanochat --base-dir $NANOCHAT_BASE_DIR data tokenizer train --max-chars=50000000

# Production (~30–60 minutes, 2B characters)
nanochat --base-dir $NANOCHAT_BASE_DIR data tokenizer train

# Optional: check compression ratio
nanochat --base-dir $NANOCHAT_BASE_DIR data tokenizer eval
```

## 5. First Training Run

### POC / smoke test (~2 minutes, validates the training loop)

```bash
nanochat --base-dir $NANOCHAT_BASE_DIR --wandb=local train base \
    --depth=6 \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=524288 \
    --num-iterations=10 \
    --eval-every=1 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --window-pattern=L \
    --compression-log-every=1
```

### Quick test (CPU / Apple Silicon, ~30 minutes)

```bash
nanochat --base-dir $NANOCHAT_BASE_DIR --wandb=local train base \
    --depth=6 \
    --max-seq-len=512 \
    --device-batch-size=4 \
    --total-batch-size=16384 \
    --num-iterations=500 \
    --eval-every=100 \
    --core-metric-every=-1 \
    --sample-every=-1 
```

### GPU run (single GPU)

```bash
nanochat --base-dir $NANOCHAT_BASE_DIR --wandb=local train base \
    --depth=12 \
    --num-iterations=2000 \
    --eval-every=250 
```

### Multi-GPU run (8× H100, GPT-2 capability)

```bash
torchrun --standalone --nproc_per_node=8 -m nanochat.cli train base -- \
    --depth=24 \
    --target-param-data-ratio=9.5 \
    --device-batch-size=16 \
    --fp8 \
    --run=my-run
```

See [runs/speedrun.sh](../../runs/speedrun.sh) for the full reference pipeline including SFT and evaluation.

## 6. SFT

After pretraining, download identity data and run supervised fine-tuning:

```bash
curl -L -o $NANOCHAT_BASE_DIR/identity.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

nanochat --base-dir $NANOCHAT_BASE_DIR train sft --run=my-run
```

## 7. Chat

```bash
nanochat --base-dir $NANOCHAT_BASE_DIR chat -p "Why is the sky blue?"   # single prompt
nanochat --base-dir $NANOCHAT_BASE_DIR chat                              # interactive session
nanochat --base-dir $NANOCHAT_BASE_DIR serve                             # web UI at http://localhost:8000
```

## 8. Status Check

```bash
# Check GPU
nvidia-smi

# Run tests
uv run pytest tests/ -q
```
