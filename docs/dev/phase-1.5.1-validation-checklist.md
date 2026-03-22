---
title: "Phase 1.5.1 Validation Checklist"
summary: "Step-by-step checklist for running compression metrics validation experiments."
read_when: "Ready to validate compression metrics with actual training runs."
status: active
last_updated: "2025-07-11"
---

# Phase 1.5.1 Validation Checklist

**Implementation**: ✅ `CompressionMetrics` class + training loop integration complete  
**Pipeline smoke test**: ✅ base train → SFT → chat validated end-to-end on MPS (Apple Silicon M3 Max)  
**Hardware**: MLX only (Apple Silicon M3 Max) — d6 only for initial validation  
**Next step**: Run Experiment 1 (smoke test), then proceed in order.

## Pre-Flight

### 1. Environment

```bash
cd /Users/geronimo/build/sp_theory/forge/nanochat
uv run python -c "import mlx.core as mx; print('MLX:', mx.default_device())"
```

### 2. Data

```bash
cd /Users/geronimo/build/sp_theory/forge/nanochat
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    data download -n 8    # minimum for d6 validation
```

See [data-layout.md](data-layout.md) for where shards are stored.

### 3. Tokenizer

```bash
cd /Users/geronimo/build/sp_theory/forge/nanochat
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    data tokenizer train
```

### 4. Code

```bash
cd /Users/geronimo/build/sp_theory/forge/nanochat
uv run python -c "from nanochat.training.compression_metrics import CompressionMetrics; print('OK')"
uv run pytest tests/ -q
# Expected: 323 passed, 10 skipped (FA3 tests skipped on CPU)
```

## Experiments

### Experiment 1 — Smoke Test (d6, ~3 min on MLX)

Verify compression tracking and `val/bpb` evaluation work end-to-end without errors.

```bash
cd /Users/geronimo/build/sp_theory/forge/nanochat
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=mlx --wandb=local --run=compression-smoke-d6 \
    train base --depth=6 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=64 --total-batch-size=524288 \
    --num-iterations=20 \
    --track-compression --compression-log-every=5 \
    --eval-every=10 --eval-tokens=524288 \
    --core-metric-every=-1 --sample-every=-1
```

`--eval-tokens=524288` = 8 eval steps at this batch size — enough to verify `evaluate_bpb_mlx`
without OOM risk (default 41943040 = 640 steps would OOM before the fix).

- [x] No errors
- [x] `[compression]` lines appear in console every 5 steps
- [x] MFU consistent with baseline (~44-46k tok/sec)
- [x] `compression/compression_ratio` logged to wandb
- [x] `val/bpb` logged at step 10 and 20 without OOM

### Observations

Run: M3 Max 128GB, d6, 20 steps, warmup only (`lrm` 0.03→0.50).

| step | entropy | ratio | gzip  | efficiency | val/bpb |
|------|---------|-------|-------|------------|---------|
| 0    | 10.637  | 0.700 | 4.497 | 0.067      | 3.184   |
| 5    | 10.550  | 0.752 | 4.463 | 0.072      | —       |
| 10   | 10.495  | 0.615 | 4.477 | 0.059      | 3.561   |
| 15   | 10.602  | 0.596 | 4.418 | 0.057      | —       |
| 20   | —       | —     | —     | —          | 3.807   |

- **Loss flat at 10.5** — model hasn't started learning, still in warmup. No conclusions
  about loss/compression correlation possible at this scale.
- **Compression ratio moves before loss does** — `ratio` drops 0.700 → 0.596 while loss
  is still flat. If this holds at longer runs, compression could be an earlier convergence
  indicator than val/bpb. This is the core hypothesis for Phase 1.5.
- **val/bpb increasing** (3.18 → 3.56 → 3.81) — expected on a fresh model in warmup
  with very few tokens seen.
- **`efficiency` tracks `ratio` closely** — correlated by construction, not independently
  informative at this stage.
- **Peak memory 28GB** — healthy, well within 128GB unified memory budget.
- **Throughput ~46–47k tok/sec** — consistent with MLX baseline.

Conclusion: pipeline is healthy. Experiment 2 needed for real correlation analysis.

### Experiment 2 — Short Validation (d6, ~5h on MLX)

Collect enough data points to analyze correlation.

```bash
cd /Users/geronimo/build/sp_theory/forge/nanochat
uv run nanochat --config /Users/geronimo/build/sp_theory/experiments/nanochat/config.toml \
    --backend=mlx --wandb=local --run=compression-validation-d6 \
    train base --depth=6 --aspect-ratio=64 --head-dim=128 --max-seq-len=1024 \
    --device-batch-size=64 --total-batch-size=524288 \
    --track-compression --compression-log-every=50 \
    --eval-every=250 --core-metric-every=0 --sample-every=0 --save-every=-1
```

- [ ] Completes without error
- [ ] `val/bpb` logged every 250 steps
- [ ] `compression/compression_ratio` logged every 50 steps
- [ ] Can plot `compression_ratio` vs `val/bpb` at steps that are multiples of 250

## Analysis

After experiments, compute correlation between `compression/compression_ratio` and `val/bpb`.

- Pearson R² between compression ratio and val loss
- Whether compression plateau precedes loss plateau (early stopping signal)
- Per-layer compression distribution (if `--track-layer-compression` used)

Write results to `docs/phase-1.5.1-validation-report.md`.

## Success Criteria

| R² | Result | Action |
|---|---|---|
| > 0.7 | Strong | Proceed to Phase 1.5.2 (dataset quality) |
| 0.4–0.7 | Moderate | Refine metrics, investigate which ones correlate |
| < 0.4 | Weak | Re-evaluate approach or pivot |

## Exit Criteria

- [ ] All planned experiments completed
- [ ] R² calculated and documented
- [ ] Validation report written
- [ ] Decision made: proceed / refine / pivot
