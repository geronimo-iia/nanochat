---
title: "Experiment 2 — Compression Validation Results"
summary: "Raw data and observations from the d6 compression validation run."
read_when: "Analyzing compression metric correlation with val/bpb."
status: active
last_updated: "2025-07-25"
---

# Experiment 2 — Compression Validation Results

**Run**: `compression-validation-d6`, M3 Max 128GB  
**Duration**: 89.7 min (464 steps)  
**Config**: d6, `compression-log-every=50`, `eval-every=250`, `eval-tokens=5242880`  
**Total tokens**: 243,269,632 (tokens:params ratio 10.49 — Chinchilla)

## Raw Data

### Compression metrics (every 50 steps)

| step | entropy | ratio  | gzip   | efficiency |
|------|---------|--------|--------|------------|
| 0    | 10.637  | 0.6995 | 4.4969 | 0.0666     |
| 50   | 10.598  | 0.6041 | 4.4624 | 0.0575     |
| 100  | 10.614  | 0.4583 | 4.4596 | 0.0436     |
| 150  | 10.565  | 0.4256 | 4.5318 | 0.0405     |
| 200  | 10.476  | 0.4066 | 4.5188 | 0.0387     |
| 250  | 10.566  | 0.4037 | 4.4375 | 0.0384     |
| 300  | 10.520  | 0.4000 | 4.5453 | 0.0381     |
| 350  | 10.566  | 0.4015 | 4.4429 | 0.0382     |
| 400  | 10.565  | 0.3987 | 4.5165 | 0.0380     |
| 450  | 10.555  | 0.4029 | 4.4592 | 0.0384     |

### Validation bpb (every 250 steps)

| step | val/bpb  |
|------|----------|
| 0    | 3.212615 |
| 250  | 5.537459 |
| 464  | 5.560884 |

### Training loss

Flat at `10.500000` for all 464 steps.

## Observations

### Loss flat at 10.5 — root cause identified and fixed

Loss is `10.500000` for every single step across the entire 464-step run. `ln(32768)` ≈
10.397 is the theoretical cross-entropy of a uniform distribution over the vocabulary —
this is exactly what a fresh random model produces.

**Root cause**: `mx.compile` captures array references at compile time. In `MLXTrainer.__init__`:

```python
loss_and_grad = _LossAndGrad(orig_model)
self._loss_and_grad = mx.compile(loss_and_grad)  # BUG: no inputs=
```

The compiled graph holds references to the model's parameter arrays as they existed at
construction time. When the optimizer calls `model.update(...)` each step, it replaces
those arrays with new ones. The compiled function never sees the replacements — it keeps
evaluating the original random weights. Loss stays at initialization entropy forever.

**Why compression ratio moved**: `forward_logits()` calls `self._orig_model(self._last_x)`
directly, bypassing the compiled function. It was seeing the updated weights. The model
was actually learning — only the loss computation was stale.

**Why val/bpb increased**: same reason. The eval path uses `eval_context()` which calls
the model directly. At step 0 the model is random (bpb ≈ 3.21 — suspiciously low, see
note below). By step 250 the model had learned something, but the val set is harder than
the step-0 eval batch, so bpb appears to increase. The step-0 bpb of 3.21 vs expected
~15 bpb for a random model is a separate anomaly worth investigating.

**Fix**: pass `inputs=[orig_model]` to `mx.compile` so the compiler re-reads model
parameters on each call:

```python
self._loss_and_grad = mx.compile(loss_and_grad, inputs=[orig_model])
```

`outputs=` is not needed — the optimizer updates params outside the compiled function.

**Impact**: this bug affected every MLX training run. The model was learning (optimizer
state, compression ratio, val/bpb all changed) but the logged `train/loss` was always
the initialization value. All Experiment 1 and Experiment 2 loss values in the logs are
invalid. Experiment 2 must be re-run after the fix.

**Regression test added**: `test_compiled_loss_sees_updated_params` in
`tests/test_training/test_mlx_trainer.py` — verifies that the compiled loss function
returns a different value after an optimizer step.

### val/bpb increasing (3.21 → 5.54 → 5.56)

val/bpb increases from step 0 to step 250, then plateaus. This is the opposite of expected
behavior. Two possible explanations:

1. The model is not learning (consistent with flat loss hypothesis above)
2. The eval set at step 0 is different from subsequent evals (e.g. different data split or
   the step-0 eval runs before any training, on a different distribution)

The step-0 bpb of `3.21` is suspiciously low compared to `5.54` at step 250. A fresh random
model should have bpb close to `log2(vocab_size)` = 15 bits. `3.21` suggests either the
step-0 eval is on a trivially easy subset, or there is a bug in the step-0 eval path.

### Compression ratio moves despite flat logged loss

`ratio` drops from `0.6995` → `0.40` by step 200 then plateaus. This is now explained
by the `mx.compile` bug: `forward_logits()` bypasses the compiled function and calls the
model directly, so it was seeing the actual updated weights. The compression metric was
valid — the model was learning. The logged loss was not.

### Timing

Actual duration: **89.7 min** — not ~5h as estimated. The Chinchilla ratio at d6 yields
only 464 steps, not thousands. Future experiment time estimates must account for the
`target_param_data_ratio` calculation.

### Alignment issue

`compression-log-every=50` and `eval-every=250` are not aligned — only step 250 has both
metrics. Steps 50/100/150/200/300/350/400/450 have compression data but no val/bpb to
correlate against. Next experiment should set `compression-log-every` to a divisor of
`eval-every` (e.g. `compression-log-every=250`), or increase total iterations to get more
`x*250` eval points.

## Next Steps

1. ~~Investigate loss logging~~ — root cause found and fixed (`mx.compile` stale params)
2. Investigate val/bpb at step 0 — why is it 3.21 when a fresh random model should be ~15 bpb?
3. Fix experiment config: align `compression-log-every` with `eval-every`
4. Re-run Experiment 2 with the fix to get valid loss and correlation data
