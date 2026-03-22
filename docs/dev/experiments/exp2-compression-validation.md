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
| ---- | ------- | ------ | ------ | ---------- |
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
| ---- | -------- |
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

**Fix**: pass both `inputs=[orig_model]` and `outputs=[orig_model]` to `mx.compile`:

```python
self._loss_and_grad = mx.compile(loss_and_grad, inputs=[orig_model], outputs=[orig_model])
```

`inputs=[orig_model]` ensures the compiler re-reads model parameters on each call.
`outputs=[orig_model]` is also required because `nn.value_and_grad` internally calls
`model.update(params)` — it modifies `orig_model`'s parameters as a side effect inside
the compiled function. Without `outputs=`, MLX doesn't track this write, and after any
`mx.eval()` on model outputs (e.g. during validation), the next compiled call produces
grad arrays disconnected from the computation graph, crashing with
`RuntimeError: Attempting to eval an array without a primitive`.

**Impact**: this bug affected every MLX training run. The model was learning (optimizer
state, compression ratio, val/bpb all changed) but the logged `train/loss` was always
the initialization value. All Experiment 1 and Experiment 2 loss values in the logs are
invalid. Experiment 2 must be re-run after the fix.

**Regression tests added** in `tests/test_training/test_mlx_trainer.py`:
- `test_compiled_loss_sees_updated_params` — verifies compiled loss sees updated weights
- `test_eval_context_restores_train_mode` — verifies `eval_context()` + `mx.eval` followed
  by `forward_backward()` does not crash (catches the missing `outputs=` bug)

### val/bpb increasing (3.21 → 5.54 → 5.56) — root cause identified

**Root cause**: stale-gradient drift from the `mx.compile` bug, plus a misread of what
"random model bpb" should be.

#### Why step-0 bpb is 3.21 (not anomalously low)

The experiment doc initially flagged 3.21 as suspicious by comparing it to
`log2(32768) = 15 bits/token`. That comparison is wrong — val/bpb is bits **per byte**,
not bits per token. The tokenizer byte distribution:

```
avg bytes per token: 6.57  (measured on token_bytes.pt)
theoretical bpb (uniform model): ln(32768) / (ln(2) * 6.57) ≈ 2.28 bpb
```

So the theoretical floor for a fully random model is **2.28 bpb**, not 15. A step-0 bpb
of 3.21 is slightly above this floor — consistent with a near-random model whose logits
are not perfectly uniform at initialization (softcap + non-zero weights). No anomaly.

#### Why val/bpb increases from 3.21 → 5.54 → 5.56

This is a direct consequence of the `mx.compile` stale-params bug, but the mechanism is
more subtle than just "loss is wrong":

- The compiled `_loss_and_grad` held references to the **initial** weight arrays
- Every step, gradients were computed at those **fixed initial weights**, not at the
  current model state
- The optimizer applied these stale-direction gradients to the actual model params,
  which were being replaced by `model.update(...)` each step
- Result: the model was pushed repeatedly in the gradient direction at initialization —
  effectively a fixed-direction drift uncoupled from the current model state

This explains the apparent contradiction between signals:

| Signal                          | What it measured                                          | Conclusion                            |
| ------------------------------- | --------------------------------------------------------- | ------------------------------------- |
| `train/loss = 10.5`             | Loss of initial (random) weights, always                  | Invalid — stale params                |
| `compression ratio 0.70 → 0.40` | Real model output via `forward_logits` (bypasses compile) | Valid — model IS changing             |
| `val/bpb 3.21 → 5.54 → 5.56`    | Real model output on val set (eval path bypasses compile) | Valid — model genuinely getting worse |
| `step-0 bpb = 3.21`             | Random init, avg 6.57 bytes/token                         | Expected — floor is 2.28              |

The compression ratio improved because `forward_logits()` calls `self._orig_model(x)`
directly. The actual model parameters were drifting — just in a pathological direction
driven by stale gradients from the initial weights. That direction happens to reduce theƒ
initial model's loss on training data while making the model worse at predicting real
val sequences.

The val loader always starts from the beginning of the val split (confirmed: `build_val_loader`
constructs a fresh loader on each eval call), so data distribution shift between evals is
ruled out.

**After the `inputs=[orig_model]` fix**, gradients are computed at the current model state
each step. Val/bpb should decrease monotonically from ~3.21 as training proceeds.

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
2. ~~Investigate val/bpb at step 0~~ — resolved: theoretical floor is 2.28 bpb (avg 6.57 bytes/token); 3.21 is normal. Val/bpb increase is stale-gradient drift from the compile bug.
3. Fix experiment config: align `compression-log-every` with `eval-every`
4. Re-run Experiment 2 with the fix to get valid loss and correlation data
