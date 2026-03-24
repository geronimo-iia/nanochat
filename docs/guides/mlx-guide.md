---
title: "Apple Silicon Guide"
summary: "How nanochat runs on Apple Silicon — MLX (recommended) and PyTorch MPS (fallback) backends, practical training guidelines, and known limitations."
read_when:
  - Training on Apple Silicon (M-series) hardware
  - Choosing between the MLX and MPS backends
  - Debugging backend-specific issues or performance
  - Choosing batch size and sequence length for Apple Silicon
status: active
last_updated: "2025-07-24"
---

# Apple Silicon Guide

Nanochat supports Apple Silicon via two backends: **MLX** (recommended) and **PyTorch MPS**
(fallback). Both use the same unified memory hardware — the difference is the runtime.

Backend autodetection: if `mlx` is importable → `"mlx"`, otherwise → `"torch"`.

## Choosing a backend

|                 | MLX (`--backend=mlx`)                          | MPS (`--backend=torch --device-type=mps`) |
| --------------- | ---------------------------------------------- | ----------------------------------------- |
| Runtime         | MLX — purpose-built for Apple Silicon          | PyTorch with MPS device                   |
| Compile         | `mx.compile` — stable                          | `torch.compile` — ❌ NaN gradients, disabled |
| Precision       | bfloat16 / float16                             | float16 only (GradScaler required)        |
| FP8             | ❌ CUDA-only                                    | ❌ CUDA-only                               |
| DDP             | ❌ single-device                                | ❌ single-device                           |
| SFT / RL / eval | ❌ base training only                           | ✅ full feature set                        |
| Throughput      | Higher (compile + no GradScaler overhead)      | ~2.4× slower than CUDA                   |
| Autodetect      | ✅ default when MLX installed                   | fallback if MLX not installed             |

Use MLX for base pretraining on Apple Silicon. Use MPS if you need SFT, RL, or the
evaluation engine, or if MLX is not installed.

---

## MLX Backend

### Quick start

```bash
nanochat train base \
    --depth=12 \
    --max-seq-len=1024 \
    --device-batch-size=8 \
    --total-batch-size=524288
# --backend=mlx is autodetected when MLX is installed
```

### What works

| Feature               | Status | Notes |
| --------------------- | ------ | ----- |
| Base training         | ✅ | Full loop via `MLXTrainer` |
| `mx.compile`          | ✅ | Applied to loss function via `nn.value_and_grad` |
| Muon optimizer        | ✅ | Full Polar Express + NorMuon — see [mlx-muon-design.md](mlx-muon-design.md) |
| Grad accumulation     | ✅ | Manual `nn.utils.tree_map` accumulation |
| Compression tracking  | ✅ | `forward_logits()` returns numpy arrays |
| Checkpoint save/load  | ✅ | Use `format = "safetensors"` for cross-backend interop |
| SFT / RL              | ❌ | PyTorch only |
| Evaluation engine     | ❌ | KV cache not implemented in MLX GPT |
| FP8                   | ❌ | CUDA-only |
| DDP                   | ❌ | Single-device only |

### Batch size recommendations (128GB unified memory)

MLX uses bfloat16 (~2 bytes per parameter).

| Depth | Params | `--device-batch-size` | `--max-seq-len` | Notes |
| ----- | ------ | --------------------- | --------------- | ----- |
| 4     | ~10M   | 64                    | 2048            | Comfortable |
| 8     | ~42M   | 32                    | 2048            | Comfortable |
| 12    | ~110M  | 16                    | 1024            | Good for validation |
| 16    | ~235M  | 8                     | 1024            | Comfortable |
| 20    | ~400M  | 4                     | 1024            | Tight — reduce seq len if OOM |

Gradient accumulation via `--total-batch-size` maintains effective batch size regardless
of `--device-batch-size`.

### Overnight runs

```bash
nohup nanochat train base \
    --depth=12 \
    --max-seq-len=1024 \
    --device-batch-size=8 \
    --run=mlx-d12-overnight \
    --save-every=1000 \
    --wandb=disabled \
    > train.log 2>&1 &
```

---

## MPS Backend (PyTorch)

Use `--backend=torch` to force the PyTorch MPS path. Required for SFT, RL, and the
evaluation engine. Also the fallback when MLX is not installed.

```bash
nanochat --backend=torch --device-type=mps train base \
    --depth=12 \
    --max-seq-len=1024 \
    --device-batch-size=4 \
    --total-batch-size=524288
```

### Compute dtype

MPS uses **float16** via `torch.amp.autocast` (`common/dtype.py` detects MPS and returns
`torch.float16`). GradScaler is active — required for fp16 overflow handling. Memory per
parameter is ~2 bytes, same as bf16 on CUDA.

Override with `NANOCHAT_DTYPE=float32` if you hit fp16 stability issues.

### Attention

Flash Attention 3 requires Hopper (sm90). On MPS, nanochat falls back to PyTorch SDPA
automatically (`flash_attention.py`). Both full-context and sliding-window paths use the
same SDPA backend — no measurable speed difference on MPS (benchmarked: 4.08ms/call both
paths at d12, T=1024, fp16).

### What works

| Feature                               | Status | Notes |
| ------------------------------------- | ------ | ----- |
| Base training (`nanochat train base`) | ✅ | Full training loop |
| SFT training (`nanochat train sft`)   | ✅ | Full fine-tuning loop |
| Evaluation (`nanochat eval chat`)     | ✅ | All eval tasks |
| SDPA attention                        | ✅ | Automatic fallback from FA3 |
| Compression tracking                  | ✅ | `--track-compression` works |
| Checkpoint save/load                  | ✅ | bf16→fp32 conversion on load |
| Muon optimizer                        | ✅ | Polar Express + variance reduction in eager mode |
| `torch.mps.synchronize()`             | ✅ | Used for accurate step timing |
| `torch.mps.empty_cache()`             | ✅ | Called between eval and training steps |
| Memory reporting                      | ✅ | `torch.mps.current_allocated_memory()` |
| `torch.compile`                       | ❌ | NaN gradients on MPS — disabled automatically |
| FP8                                   | ❌ | CUDA-only |
| Flash Attention 3                     | ❌ | Hopper GPU required |
| bf16 compute                          | ❌ | MPS supports bf16 but nanochat uses fp16 for GradScaler compatibility |
| DDP / multi-device                    | ❌ | Single-device only |

`pin_memory` and `non_blocking` are disabled on MPS — Apple Silicon uses unified memory,
so there is no PCIe transfer to optimize.

### Known workarounds in code

**`torch.compile` disabled** (`training/base/setup.py`): the inductor backend produces NaN
gradients on MPS during gradient accumulation (confirmed PyTorch 2.9.1, M3 Max). Skipped
automatically when `device_type == "mps"`. Cost: ~2.4× slower throughput vs CUDA.

**int64 comparison** (`evaluation/loss_eval.py`): MPS lacks an int64 kernel for `< 0`.
Uses `y.int() < 0` (int32 cast) instead.

**Peak FLOPS**: `get_peak_flops` returns `float("inf")` for non-CUDA devices — MFU
reporting is disabled.

### Batch size recommendations (128GB unified memory)

Training uses fp16 (~2 bytes per parameter).

| Depth | Params | `--device-batch-size` | `--max-seq-len` | Notes |
| ----- | ------ | --------------------- | --------------- | ----- |
| 4     | ~10M   | 32                    | 2048            | Comfortable |
| 8     | ~42M   | 16                    | 2048            | Comfortable |
| 12    | ~110M  | 8                     | 1024            | Good for validation |
| 16    | ~235M  | 4                     | 1024            | Comfortable |
| 20    | ~400M  | 2                     | 1024            | Tight — reduce seq len if OOM |

### Overnight runs

```bash
nohup nanochat --backend=torch --device-type=mps train base \
    --depth=12 \
    --max-seq-len=1024 \
    --device-batch-size=4 \
    --run=mps-d12-overnight \
    --save-every=1000 \
    --wandb=disabled \
    > train.log 2>&1 &
```
