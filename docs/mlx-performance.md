---
title: MLX Performance — Base Training
summary: First performance comparison of MLX vs MPS backends for base pretraining on Apple Silicon. Methodology, benchmark configs, and measured results.
read_when:
  - Choosing between MLX and MPS for base training on Apple Silicon
  - Evaluating throughput before a long training run
  - Investigating performance regressions after backend changes
status: draft
last_updated: 2025-07-10
---

# MLX Performance — Base Training

First honest performance comparison between the MLX and MPS backends for base
pretraining on Apple Silicon. Numbers to be filled in after
[mlx-analysis-phase6.md](mlx-analysis-phase6.md) is implemented.

---

## Hardware

| Field | Value |
|-------|-------|
| Machine | Apple M3 Max |
| Unified memory | 128 GB |
| CPU cores | 16 (12P + 4E) |
| GPU cores | 40 |
| Neural Engine | 16-core |

---

## Measurement methodology

### What is measured

- `dt` — wall-clock time per training step in seconds, including forward, backward,
  optimizer update, and `mx.eval` / `synchronize` blocking call
- `tok/sec` — `total_batch_size / dt`
- Peak memory — reported at end of run (`get_max_memory()`)

### What is not measured

- Data loading time — the torch dataloader runs on CPU for both backends; its cost
  is amortized and not included in `dt`
- First-step JIT warmup — MLX and `mx.compile` have a compilation cost on step 0;
  numbers are taken from steps 5–20 after warmup

### Conditions

- `--wandb=disabled` — no network overhead
- `--eval-every=0` — no validation interruptions
- `--sample-every=0` — no sampling interruptions
- `--save-every=0` — no checkpoint I/O during measurement window
- `--num-iterations=25` — 25 steps total, discard first 5, average steps 5–20
- Fresh process each run — no cross-contamination between backends
- Machine otherwise idle — no background GPU workloads

### Commands

**MLX:**
```bash
nanochat train base \
    --depth=<D> \
    --max-seq-len=<T> \
    --device-batch-size=<B> \
    --total-batch-size=<BT> \
    --num-iterations=25 \
    --eval-every=0 \
    --sample-every=0 \
    --save-every=0 \
    --wandb=disabled
# --backend=mlx autodetected
```

**MPS:**
```bash
nanochat train base \
    --backend=torch \
    --device-type=mps \
    --depth=<D> \
    --max-seq-len=<T> \
    --device-batch-size=<B> \
    --total-batch-size=<BT> \
    --num-iterations=25 \
    --eval-every=0 \
    --sample-every=0 \
    --save-every=0 \
    --wandb=disabled
```

---

## Benchmark configs

Three configs covering small, medium, and large models on M3 Max.

| Config | depth | seq_len | params | flops/token | device_batch_size | total_batch_size |
|--------|-------|---------|--------|-------------|-------------------|------------------|
| d6     | 6     | 1024    | 74M    | 1.57e8      | TBD               | TBD              |
| d12    | 12    | 1024    | 286M   | 7.31e8      | TBD               | TBD              |
| d20    | 20    | 2048    | 897M   | 3.00e9      | TBD               | TBD              |

`device_batch_size` to be set to the largest value that fits in memory without OOM
on both backends (use the more constrained backend as the common baseline).

---

## Results

### d6 — 74M params, seq=1024

| Backend | Precision | dt (s) | tok/sec | Peak memory |
|---------|-----------|--------|---------|-------------|
| MLX     | bfloat16  | TBD    | TBD     | TBD         |
| MPS     | float16   | TBD    | TBD     | TBD         |
| Ratio MLX/MPS | — | —  | TBD     | —           |

### d12 — 286M params, seq=1024

| Backend | Precision | dt (s) | tok/sec | Peak memory |
|---------|-----------|--------|---------|-------------|
| MLX     | bfloat16  | TBD    | TBD     | TBD         |
| MPS     | float16   | TBD    | TBD     | TBD         |
| Ratio MLX/MPS | — | —  | TBD     | —           |

### d20 — 897M params, seq=2048

| Backend | Precision | dt (s) | tok/sec | Peak memory |
|---------|-----------|--------|---------|-------------|
| MLX     | bfloat16  | TBD    | TBD     | TBD         |
| MPS     | float16   | TBD    | TBD     | TBD         |
| Ratio MLX/MPS | — | —  | TBD     | —           |

---

## Known asymmetries between backends

These are structural differences that affect the comparison — not bugs.

| Aspect | MLX | MPS |
|--------|-----|-----|
| Compile | `mx.compile` on forward-backward (`_LossAndGrad`) | No compile (`torch.compile` disabled — NaN gradients) |
| Precision | bfloat16 | float16 + GradScaler |
| Optimizer | `@mx.compile` on `muon_step` | Eager PyTorch |
| Dataloader device | CPU tensors → `mx.array` | MPS tensors (direct) |
| Synchronization | `mx.eval()` at end of step | `torch.mps.synchronize()` |
| Memory cache clear | `mx.clear_cache()` | `torch.mps.empty_cache()` |

The MLX backend has a structural throughput advantage from `mx.compile`. The MPS
backend has no equivalent — `torch.compile` is disabled due to NaN gradients.
This is expected and documented in [m3-max-guide.md](m3-max-guide.md).

---

## Interpretation notes

- tok/sec is the primary metric — it captures the full step cost including optimizer
- Memory comparison is informative but not directly comparable: bfloat16 (MLX) vs
  float16 + GradScaler state (MPS) have different memory footprints
- d6 is the most reliable config for iteration — fast enough to run many times
- d20 is the stress test — most sensitive to memory bandwidth and compile quality
