---
title: "Tuning Guide"
summary: "Parameter recommendations for tokenizer training, pretraining, and SFT across hardware tiers."
read_when:
  - Deciding on vocab size, model depth, or batch size
  - Scaling up from a smoke test to a real run
  - Tuning SFT hyperparameters
status: active
last_updated: "2026-03-17"
---

# Tuning Guide

## Tokenizer

| Parameter | Default | Recommendation |
|-----------|---------|----------------|
| `vocab_size` | 32768 | Use 32768 for quick experiments. The original speedrun uses 65536 (`2**16`) and beats GPT-2 compression across the board. GPT-4 uses 100K and wins on code/math/multilingual — 50K–65K is the sweet spot to beat GPT-2 token efficiency. |
| `max_chars` | 2B | Keep at 2B for production. More shards alone don't help — you must also raise `max_chars`. |
| `doc_cap` | 10000 | Leave at default. Only lower if RAM is tight during tokenizer training. |

**Key insight**: `vocab_size` is the primary lever for token reduction. The default 32K is conservative — the original nanochat speedrun used 65536 and achieved ~4.8 chars/token, beating GPT-2 across most categories.

---

## Pretraining (`train base`)

### Model Architecture

| Parameter | Default | Recommendation |
|-----------|---------|----------------|
| `depth` | 20 | Start with 6 for smoke tests, 12 for single-GPU experiments (~6 min on 8×H100), 20–24 for serious runs. d24 beats GPT-2 XL CORE in ~3 hours on 8×H100. |
| `aspect_ratio` | 64 | `model_dim = depth × aspect_ratio`. Leave at 64 — validated across the entire miniseries. |
| `head_dim` | 128 | Leave at 128. Matches hardware-efficient attention kernels. |
| `max_seq_len` | 2048 | 512 for smoke tests (faster). 2048 for production — empirically the sweet spot balancing context and document diversity. |
| `window_pattern` | `SSSL` | `L` = all full context (simplest, slower). `SSSL` = 3 sliding-window layers (1024 tokens) + 1 full (2048 tokens), tiled across depth, final layer always full. Only use `L` for smoke tests or very short sequences. |

### Training Horizon

Exactly one of these three should be set; the others disabled (`-1`):

| Parameter | When to use |
|-----------|-------------|
| `num_iterations` | Smoke tests and fixed-budget runs. |
| `target_param_data_ratio` | Chinchilla-style scaling. nanochat's empirically measured optimal ratio is ~8–10 (not Chinchilla's 20 — Muon optimizer and architecture shift this). Default 10.5 is a good balance. Use 12 for the GPT-2 beating speedrun (slight overtrain for extra quality). |
| `target_flops` | When you have a strict compute budget in FLOPs. |

### Batch Size

| Parameter | Default | Recommendation |
|-----------|---------|----------------|
| `total_batch_size` | -1 (auto) | Auto computes a good value. The validated speedrun uses 524288 (512K tokens = `32 × 2048 × 8 GPUs`). Purely flops-wise ~256K is slightly better, but 512K wins on wall-clock for 8×H100. |
| `device_batch_size` | 32 | Reduce if you hit OOM. On MPS/CPU use 4–8. On H100: 32 for d≤20, 16 for d26, 8 for d30+. |

Gradient accumulation steps = `total_batch_size / (device_batch_size × seq_len × world_size)`.

### Optimizer

The defaults are well-tuned — only change these if you know what you're doing:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `matrix_lr` | 0.02 | Muon LR for weight matrices. Most sensitive param. |
| `embedding_lr` | 0.3 | AdamW LR for embeddings (`wte` + `value_embeds`). Scaled by `1/√(dim/768)` automatically. |
| `unembedding_lr` | 0.008 | AdamW LR for `lm_head`. Also scaled by `1/√(dim/768)`. |
| `weight_decay` | 0.28 | Cautious weight decay, linearly scheduled to 0 by end of training. |
| `warmup_steps` | 40 | Fine for most runs. Increase to 100–200 for very long runs. |
| `warmdown_ratio` | 0.65 | Last 65% of training is LR warmdown. The Jan29 speedrun used 0.5 (tuned up from 0.4). 0.65 is the current default after further tuning. |
| `final_lr_frac` | 0.05 | LR at end of warmdown as fraction of peak. |

### Evaluation

| Parameter | Default | Recommendation |
|-----------|---------|----------------|
| `eval_every` | 250 | Smoke tests: 1. Production: 250–500. |
| `eval_tokens` | ~42M | Smoke tests: 524288 (1 batch). Production: keep default. |
| `core_metric_every` | 2000 | Expensive — runs ARC/MMLU/GSM8K. Disable (`-1`) for short runs. |
| `sample_every` | 2000 | Text samples in wandb. Disable (`-1`) for short runs. |
| `save_every` | -1 | `-1` = only save at end. Set to e.g. 1000 for long runs with mid-run recovery. |

### Compression Tracking

| Parameter | Default | Notes |
|-----------|---------|-------|
| `track_compression` | false | Enables bpb compression metrics during training. Useful for research. |
| `compression_log_every` | 100 | How often to log compression metrics. Set to 1 for smoke tests. |
| `track_layer_compression` | false | Per-layer compression breakdown. Expensive — only for analysis runs. |

### FP8 (CUDA only)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `fp8` | false | Enable on H100 for ~1.5× throughput. Not supported on MPS/CPU. |
| `fp8_recipe` | `tensorwise` | `rowwise` is more accurate but slower. Use `tensorwise` by default. |

---

## Hardware Presets

### Apple Silicon (MPS, fp16)

**Memory tip**: Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to allow the GPU to use more unified memory before falling back to CPU. No measurable throughput gain at small model sizes (d6 peak: ~2.3GB), but may prevent OOM at larger depths.

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run nanochat --base-dir $NANOCHAT_BASE_DIR --wandb=local train base \
    --depth=6 \
    --max-seq-len=512 \
    --device-batch-size=4 \
    --total-batch-size=16384 \
    --window-pattern=L \
    --num-iterations=500 \
    --eval-every=100 \
    --core-metric-every=-1 \
    --sample-every=-1
```

### Single GPU (A100/H100, bf16)

```bash
nanochat --base-dir $NANOCHAT_BASE_DIR --wandb=local train base \
    --depth=12 \
    --device-batch-size=32 \
    --target-param-data-ratio=10.5 \
    --eval-every=250
```

### Multi-GPU (8× H100) — GPT-2 beating speedrun

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m nanochat train base -- \
    --depth=24 \
    --target-param-data-ratio=12 \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-every=3000
```

This reproduces the Jan 29 2026 record: CORE 0.258 (GPT-2 XL: 0.257) in ~3 hours (~$73 on 8×H100). Add `--fp8` for ~1.5× throughput on Hopper GPUs.

---

## SFT (`train sft`)

### Key Parameters

| Parameter | Default | Recommendation |
|-----------|---------|----------------|
| `num_iterations` | -1 (full epoch) | `-1` runs until the dataset is exhausted. Set explicitly for smoke tests (e.g. 10). |
| `max_seq_len` | inherited | Inherits from pretrain checkpoint. Override only if you need shorter sequences. |
| `load_optimizer` | true | Keep true — warm-starting momentum buffers from pretrain helps convergence. |
| `init_lr_frac` | 0.8 | Start SFT at 80% of pretrain peak LR. Lower (0.3–0.5) if loss spikes early. |
| `warmup_ratio` | 0.0 | No warmup by default (pretrain already warmed up). Add 0.05–0.1 if starting cold. |
| `warmdown_ratio` | 0.5 | Last 50% of SFT is LR warmdown. |

### Data Mix

| Parameter | Default | Notes |
|-----------|---------|-------|
| `mmlu_epochs` | 3 | Number of passes over MMLU auxiliary train (~100K rows each). Increase for better knowledge retention. |
| `gsm8k_epochs` | 4 | Number of passes over GSM8K train (~8K rows each). Increase for better math. |

### Evaluation

| Parameter | Default | Recommendation |
|-----------|---------|----------------|
| `eval_every` | 200 | How often to compute val bpb. |
| `eval_tokens` | ~21M | Smoke tests: 524288. Production: keep default. |
| `chatcore_every` | 200 | Runs ARC/MMLU/GSM8K/HumanEval/SpellingBee. Expensive — disable (`-1`) for smoke tests. |
| `chatcore_max_cat` | -1 | Max problems for categorical tasks (ARC, MMLU). `-1` = no limit. Set to 200–500 for faster eval. |
| `chatcore_max_sample` | 24 | Max problems for generative tasks (GSM8K, HumanEval). Keep low — these are slow. |
