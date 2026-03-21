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
pretraining on Apple Silicon.

Base dir: `/Users/geronimo/build/sp_theory/experiments/nanochat`
Nanochat dir: `/Users/geronimo/build/sp_theory/forge/nanochat`

---

## Config file

`/Users/geronimo/build/sp_theory/experiments/nanochat/config.toml` — used as base
for all benchmark commands. The relevant fields overridden by CLI flags are:
`--depth`, `--max-seq-len`, `--device-batch-size`, `--total-batch-size`,
`--num-iterations`, `--eval-every`, `--core-metric-every`, `--sample-every`,
`--save-every`, `--wandb`, `--track-compression`.

```toml
[common]
base_dir = "/Users/geronimo/build/sp_theory/experiments/nanochat"
device_type = ""              # empty = autodetect (mlx or mps depending on --backend)
run = "unnamed"
wandb = "local"
wandb_project = "nanochat"

[training]
depth = 8
aspect_ratio = 64
head_dim = 128
max_seq_len = 2048
window_pattern = "SSSL"
device_batch_size = 32
total_batch_size = -1
embedding_lr = 0.3
unembedding_lr = 0.008
matrix_lr = 0.02
scalar_lr = 0.5
weight_decay = 0.28
warmup_steps = 40
warmdown_ratio = 0.65
final_lr_frac = 0.05
```

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

## Benchmark configs

Three configs using the production architecture (`head_dim=128`, `aspect_ratio=64`,
`seq_len=1024`, `total_batch_size=524288`):

| Config | depth | seq_len | params | flops/token | device_batch_size |
|--------|-------|---------|--------|-------------|-------------------|
| d6     | 6     | 1024    | 74M    | 1.57e8      | TBD (probe below) |
| d8     | 8     | 1024    | 126M   | 2.83e8      | TBD (probe below) |
| d12    | 12    | 1024    | 286M   | 7.31e8      | TBD (probe below) |

d8 is the production model. d6 is the fast iteration baseline. d12 is the depth
stress test. All use `total_batch_size=524288` — grad accumulation adjusts automatically.

---

## Step 1 — Find max device_batch_size

Run the OOM probe for each config on **both backends**. Start high and halve until
it runs cleanly for 3 steps. Use the lower of the two backends as the common baseline.

### d6 probe

```bash
cd /Users/geronimo/build/sp_theory/forge/nanochat

# MLX — try B=64
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=mlx \
    train base --depth=6 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=64 --total-batch-size=524288 \
    --num-iterations=3 --eval-every=0 --core-metric-every=0 \
    --sample-every=0 --save-every=0 --wandb=disabled --track-compression=false

# MPS — try B=64
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=torch \
    train base --depth=6 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=64 --total-batch-size=524288 \
    --num-iterations=3 --eval-every=0 --core-metric-every=0 \
    --sample-every=0 --save-every=0 --wandb=disabled --track-compression=false
```

d6 max `device_batch_size`: MLX=**TBD** MPS=**TBD** → baseline=**TBD**

### d8 probe

```bash
# MLX — try B=32
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=mlx \
    train base --depth=8 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=32 --total-batch-size=524288 \
    --num-iterations=3 --eval-every=0 --core-metric-every=0 \
    --sample-every=0 --save-every=0 --wandb=disabled --track-compression=false

# MPS — try B=32
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=torch \
    train base --depth=8 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=32 --total-batch-size=524288 \
    --num-iterations=3 --eval-every=0 --core-metric-every=0 \
    --sample-every=0 --save-every=0 --wandb=disabled --track-compression=false
```

d8 max `device_batch_size`: MLX=**TBD** MPS=**TBD** → baseline=**TBD**

### d12 probe

```bash
# MLX — try B=16
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=mlx \
    train base --depth=12 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=16 --total-batch-size=524288 \
    --num-iterations=3 --eval-every=0 --core-metric-every=0 \
    --sample-every=0 --save-every=0 --wandb=disabled --track-compression=false

# MPS — try B=16
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=torch \
    train base --depth=12 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=16 --total-batch-size=524288 \
    --num-iterations=3 --eval-every=0 --core-metric-every=0 \
    --sample-every=0 --save-every=0 --wandb=disabled --track-compression=false
```

d12 max `device_batch_size`: MLX=**TBD** MPS=**TBD** → baseline=**TBD**

---

## Step 2 — Benchmark runs

Once `device_batch_size` is confirmed, run 25 iterations per config per backend.
Read tok/sec from steps 5–20 in the console output (skip step 0 JIT warmup).

### d6

```bash
# MLX
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=mlx \
    train base --depth=6 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=<B_d6> --total-batch-size=524288 \
    --num-iterations=25 --eval-every=0 --core-metric-every=0 \
    --sample-every=0 --save-every=0 --wandb=disabled --track-compression=false

# MPS
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=torch \
    train base --depth=6 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=<B_d6> --total-batch-size=524288 \
    --num-iterations=25 --eval-every=0 --core-metric-every=0 \
    --sample-every=0 --save-every=0 --wandb=disabled --track-compression=false
```

### d8

```bash
# MLX
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=mlx \
    train base --depth=8 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=<B_d8> --total-batch-size=524288 \
    --num-iterations=25 --eval-every=0 --core-metric-every=0 \
    --sample-every=0 --save-every=0 --wandb=disabled --track-compression=false

# MPS
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=torch \
    train base --depth=8 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=<B_d8> --total-batch-size=524288 \
    --num-iterations=25 --eval-every=0 --core-metric-every=0 \
    --sample-every=0 --save-every=0 --wandb=disabled --track-compression=false
```

### d12

```bash
# MLX
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=mlx \
    train base --depth=12 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=<B_d12> --total-batch-size=524288 \
    --num-iterations=25 --eval-every=0 --core-metric-every=0 \
    --sample-every=0 --save-every=0 --wandb=disabled --track-compression=false

# MPS
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=torch \
    train base --depth=12 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=<B_d12> --total-batch-size=524288 \
    --num-iterations=25 --eval-every=0 --core-metric-every=0 \
    --sample-every=0 --save-every=0 --wandb=disabled --track-compression=false
```

---

## Results

### d6 — 74M params, seq=1024, B=TBD

| Backend | Precision | dt (s) | tok/sec | Peak memory |
|---------|-----------|--------|---------|-------------|
| MLX     | bfloat16  | TBD    | TBD     | TBD         |
| MPS     | float16   | TBD    | TBD     | TBD         |
| Ratio MLX/MPS | — | —    | TBD     | —           |

### d8 — 126M params, seq=1024, B=TBD

| Backend | Precision | dt (s) | tok/sec | Peak memory |
|---------|-----------|--------|---------|-------------|
| MLX     | bfloat16  | TBD    | TBD     | TBD         |
| MPS     | float16   | TBD    | TBD     | TBD         |
| Ratio MLX/MPS | — | —    | TBD     | —           |

### d12 — 286M params, seq=1024, B=TBD

| Backend | Precision | dt (s) | tok/sec | Peak memory |
|---------|-----------|--------|---------|-------------|
| MLX     | bfloat16  | TBD    | TBD     | TBD         |
| MPS     | float16   | TBD    | TBD     | TBD         |
| Ratio MLX/MPS | — | —    | TBD     | —           |

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

- tok/sec is the primary metric — captures full step cost including optimizer
- d8 is the most important number — it is the production model
- Memory comparison is informative but not directly comparable: bfloat16 (MLX) vs
  float16 + GradScaler state (MPS) have different footprints
- d12 shows how the gap scales with depth — compile benefit grows with compute intensity
