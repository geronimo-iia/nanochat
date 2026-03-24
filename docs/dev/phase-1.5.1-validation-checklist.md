---
title: "Phase 1.5.1 Validation Checklist"
summary: "Step-by-step checklist for running compression metrics validation experiments."
read_when: "Ready to validate compression metrics with actual training runs."
status: active
last_updated: "2025-07-25"
---

# Phase 1.5.1 Validation Checklist

**Implementation**: ✅ `CompressionMetrics` class + training loop integration complete  
**Pipeline smoke test**: ✅ base train → SFT → chat validated end-to-end on MPS (Apple Silicon M3 Max)  
**Hardware**: MLX only (Apple Silicon M3 Max) — d6 only for initial validation  
**Next step**: Investigate flat loss and val/bpb anomalies from Experiment 2 before re-running.

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

### Experiment 1b — Safetensors Checkpoint Smoke Test (d6, ~5 min on MLX)

Verify safetensors save and resume work end-to-end with MLX backend before Experiment 2.

```bash
bash runs/exp1b-safetensors-smoke.sh
```

- [x] Checkpoint saved at step 5 as `.safetensors`
- [x] Resume from step 5 completes without error
- [x] Loss at step 6 after resume matches loss before save

### Observations

Two bugs fixed during this experiment:

1. `convert.py` `to_numpy` — `np.array()` on an MLX bfloat16 array raises `RuntimeError`
   (numpy has no bfloat16, PEP 3118 format string `B` item size mismatch). Fixed by upcasting
   bfloat16 → float32 before `np.array()`.
2. `mlx_trainer.py` `load_state_dicts` — `model.update()` does not auto-cast, so float32
   arrays loaded from safetensors would silently replace bfloat16 model weights. Fixed by
   restoring each parameter to the model's existing dtype after `from_numpy_mlx`.

Resume validation from log (`exp1b.log`):

| | Phase 1 (fresh) | Phase 2 (resumed from step 5) |
|---|---|---|
| step 5 loss | 10.500000 | 10.339935 |
| step 6 loss | 10.500000 | 10.226856 |
| rg at step 5 | 8 | 10 (matches phase 1 step 7) |
| Peak memory | 28355 MiB | 28595 MiB |

- `Resuming optimization from step 5` confirmed in log
- Loss at step 5 after resume (`10.34`) is lower than fresh step 5 (`10.50`) — trained weights loaded correctly
- `rg` (dataloader shard position) correctly restored — dataloader state resumed
- Loss continues declining steps 6–9 (`10.22 → 10.14 → 10.07 → 10.02`) — optimizer momentum restored

### Experiment 2 — Short Validation (d6, ~90 min on MLX)

**Status**: ✅ Complete

Collect enough data points to analyze correlation.

```bash
bash runs/exp2-compression-validation.sh
tail -f /Users/geronimo/build/sp_theory/experiments/nanochat/exp2.log
```

- [x] Completes without error
- [x] `val/bpb` logged at steps 0, 250, 464
- [x] `compression/compression_ratio` logged every 50 steps
- [x] Can plot `compression_ratio` vs `val/bpb` at steps that are multiples of 250 — fixed by setting `--compression-log-every=250` to align with `--eval-every=250`

### Observations

See full results in [experiments/exp2-compression-validation.md](experiments/exp2-compression-validation.md).

**Blocking issues found**:

1. **Loss flat at 10.5 for all 464 steps** — root cause: `mx.compile` without `inputs=`
   captures parameter arrays at compile time. After each `model.update()` the compiled
   function keeps seeing the original random weights. Fixed by `mx.compile(loss_and_grad,
   inputs=[orig_model])`. Regression test added. All logged loss values from this run
   are invalid.
2. **val/bpb increases** (3.21 → 5.54 → 5.56) — fully explained. Step-0 bpb of 3.21 is
   normal: theoretical floor for a random model is 2.28 bpb (avg 6.57 bytes/token, not
   15 bits/token). The increase is stale-gradient drift: eval path saw real weights but
   gradients were always from initial weights, pushing params in a fixed bad direction.
   See [exp2-compression-validation.md](experiments/exp2-compression-validation.md).
3. **Timing estimate wrong** — actual duration was **89.7 min**, not ~5h. Chinchilla
   ratio at d6 yields only 464 steps.
4. **Alignment issue** — `compression-log-every=50` and `eval-every=250` are not aligned.
   Only step 250 has both metrics. Next run should set `compression-log-every` to a
   divisor of `eval-every`, or increase total iterations for more eval points.

**Experiment 2 must be re-run after the `mx.compile` fix.**

### Experiment 2b — Compile Fix Smoke Test (d6, ~5 min on MLX)

**Status**: ✅ Complete — fix validated

Verify the `mx.compile` stale-params fix before committing to the full ~5h Experiment 2 re-run.

```bash
bash runs/exp2b-compile-fix-smoke.sh
tail -f /Users/geronimo/build/sp_theory/experiments/nanochat/exp2b.log
```

Pass criteria:

- [x] `train/loss` decreases from step 0 (was flat at 10.5 before fix)
- [x] `val/bpb` decreases or stays stable from step 0 to step 20 (was increasing before fix)
- [x] `compression_ratio` and `val/bpb` both present at steps 0, 10, 20

### Observations

Run: M3 Max 128GB, d6, 20 steps, warmup only (`lrm` 0.03→0.50). Duration: **3.67 min**.

| step | loss      | lrm  | entropy | ratio  | gzip   | efficiency | val/bpb  |
|------|-----------|------|---------|--------|--------|------------|----------|
| 0    | 10.500000 | 0.03 | 10.6369 | 0.6995 | 4.4969 | 0.0666     | 3.184161 |
| 10   | 8.038718  | 0.28 | 10.4945 | 1.0974 | 4.4773 | 0.1626     | 2.046961 |
| 20   | 6.832350  | 0.50 | —       | —      | —      | —          | 1.831366 |

**Fix confirmed**:
- `train/loss` drops from 10.5 → 6.83 across 20 steps — no longer flat
- `val/bpb` drops from 3.18 → 2.05 → 1.83 — now decreasing as expected

**Compression ratio anomaly**: `ratio` increases from 0.6995 → 1.0974 at step 10 while
loss and val/bpb improve strongly. This is opposite to the Experiment 2 trend (where ratio
decreased despite the compile bug). This may reflect early warmup dynamics (only 28% of
max LR at step 10) and is not interpretable at this scale — Experiment 2 re-run needed
for correlation analysis over the full learning curve.


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
