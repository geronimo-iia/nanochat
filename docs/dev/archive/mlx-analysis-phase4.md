---
title: "MLX Backend Analysis — Phase 4"
summary: "Fresh analysis of the MLX backend after Phase 3. Five findings: two latent divergences between orig_model/model, one loss reporting inconsistency, one import style issue, one misleading log output."
read_when:
  - Reviewing MLX training correctness before deeper development
  - Investigating loss curve differences between torch and MLX backends
  - Working on MLXTrainer step/checkpoint/logging behaviour
status: archived
last_updated: "2025-07-25"
---

# MLX Backend Analysis — Phase 4

Post Phase 3 review. The structural foundation and trainer lifecycle are solid.
Five application-level findings — all resolved. Archived after Phase 4 implementation.

---

## Finding 1 — `forward_logits` uses `self._model`, not `self._orig_model`

### What's wrong

```python
def forward_logits(self):
    logits = mx.stop_gradient(self._model(self._last_x))
```

Every other inference method (`eval_context`, `model_state_dict`, `load_state_dicts`)
correctly uses `self._orig_model`. `forward_logits` is the only outlier. Today
`_model is _orig_model` so it is silent, but it is inconsistent with the established
pattern and will silently use the wrong model if a compiled wrapper is ever introduced.

### Decision

Change `forward_logits` to use `self._orig_model`. One-line fix.

### Resolution

Fixed in Task 1. `self._model(self._last_x)` → `self._orig_model(self._last_x)`.

---

## Finding 2 — `MLXTrainer.forward_backward` returns mean loss; `TorchTrainer` returns last microbatch loss

### What's wrong

`TorchTrainer.forward_backward`:
```python
train_loss = loss.detach()          # last microbatch loss, not averaged
loss = loss / self._grad_accum_steps  # only the backward pass is scaled
return StepResult(loss=train_loss.item(), ...)
```

`MLXTrainer.forward_backward`:
```python
total_loss += loss.item()
...
mean_loss = total_loss / self._grad_accum_steps
return StepResult(loss=mean_loss, ...)
```

The two backends report semantically different values under the same `StepResult.loss`
field. The loop logs this value directly to wandb and the console. With
`grad_accum_steps > 1`, the torch path is noisier (last microbatch only) and the MLX
path is smoother (mean). Loss curves are not comparable across backends.

### Decision

Option B: align MLX to torch — return the last microbatch loss from `MLXTrainer`,
matching the original torch behaviour. Both backends are consistent. The mean is
arguably more informative, but following the original torch path is the right call:
it avoids touching the torch path, keeps the semantics identical, and `grad_accum_steps`
is typically 1 on MLX (single device, unified memory) so the difference is rarely
observable in practice.

Change `MLXTrainer.forward_backward` to capture and return the last microbatch loss:

```python
train_loss = 0.0
for _ in range(self._grad_accum_steps):
    loss, grads = self._loss_and_grad(self._x, self._y)
    mx.eval(loss, grads)
    train_loss = loss.item()   # overwrite each microbatch — last one wins
    ...
return StepResult(loss=train_loss, ...)
```

### Resolution

Fixed in Task 1. `total_loss` accumulation replaced with `train_loss` overwrite.
`mean_loss` division removed.

---

## Finding 3 — `step` writes to `self._model`; `model_state_dict` reads from `self._orig_model`

### What's wrong

```python
def step(self, ...):
    self._optimizer.update(self._model, self._accumulated_grads)
    mx.eval(self._model.parameters(), ...)
```

```python
def model_state_dict(self):
    flat = dict(nn.utils.tree_flatten(self._orig_model.parameters()))
```

Writes go to `_model`, reads come from `_orig_model`. Today `_model is _orig_model`
so this is silent. If they ever diverge (compiled wrapper, EMA, quantization),
checkpoints will silently save stale weights.

The `orig_model`/`model` pattern mirrors torch: `orig_model` is the unwrapped source
of truth. `step` should write to `_orig_model` to be consistent with that contract.

### Decision

Change `step` to update `self._orig_model`:

```python
self._optimizer.update(self._orig_model, self._accumulated_grads)
mx.eval(self._orig_model.parameters(), self._optimizer.state())
```

### Resolution

Fixed in Task 1. Both `optimizer.update` and `mx.eval` now target `self._orig_model`.

---

## Finding 4 — `get_mlx_compute_dtype` import is orphaned in `_setup_mlx`

### What's wrong

```python
def _setup_mlx(...):
    from nanochat.models.mlx_gpt import GPT as MLXGPT
    from nanochat.training.mlx_optimizer import MuonAdamW, build_param_groups
    from nanochat.training.mlx_trainer import MLXTrainer

    from nanochat.common import get_mlx_compute_dtype   # ← separated by blank line
```

The `get_mlx_compute_dtype` import was added in Task 3 but landed after a blank line,
separated from the other local imports. Style only, but inconsistent.

### Decision

Consolidate into the existing import block — remove the blank line separator.

### Resolution

Fixed in Task 2. Import moved into the existing local import block at the top of
`_setup_mlx`.

---

## Finding 5 — `bf16_mfu: 0.00` logged every step on the MLX path

### What's wrong

```python
print0(f"... bf16_mfu: {mfu:.2f} ...")
```

`mfu` is computed as `flops_per_sec / (s.gpu_peak_flops * s.ddp_world_size)`. On the
MLX path `gpu_peak_flops = float("inf")`, so `mfu` is always `0.0`. The label
`bf16_mfu: 0.00` is printed every step — misleading rather than informative. The label
is also torch-specific (`bf16` refers to CUDA BF16 FLOPS).

### Decision

Option A: suppress the mfu field on the MLX path. Conditionally omit it from the
format string when `s.gpu_peak_flops == float("inf")`:

```python
mfu_str = f" | bf16_mfu: {mfu:.2f}" if s.gpu_peak_flops != float("inf") else ""
print0(f"step ... {mfu_str} | epoch: ...")
```

### Resolution

Fixed in Task 3. `mfu_str` built conditionally and spliced into the `print0` format
string in `loop.py`.

---

## Priority

| # | Finding | Risk | Effort | Decision |
|---|---|---|---|---|
| 1 | `forward_logits` uses `_model` not `_orig_model` | Latent, silent on divergence | Trivial | Fixed — Task 1 |
| 2 | Loss reporting semantics differ between backends | Observability / comparability | Small | Fixed — Task 1, Option B |
| 3 | `step` writes to `_model`, reads from `_orig_model` | Latent, silent on divergence | Trivial | Fixed — Task 1 |
| 4 | `get_mlx_compute_dtype` import orphaned | Style only | Trivial | Fixed — Task 2 |
| 5 | `bf16_mfu: 0.00` logged every step on MLX | Misleading output | Small | Fixed — Task 3, Option A |

---

## Implementation plan

Findings 1, 2, 3 touch `mlx_trainer.py`. Finding 4 touches `setup.py`. Finding 5
touches `loop.py`. Do them in three focused tasks.

**Task 1 — `mlx_trainer.py`: fix `forward_logits`, loss reporting, and `step` target**

File: `src/nanochat/training/mlx_trainer.py`

- `forward_logits`: change `self._model(self._last_x)` → `self._orig_model(self._last_x)`.
- `forward_backward`: replace `total_loss` accumulation with a `train_loss` variable
  that is overwritten each microbatch (last one wins), matching `TorchTrainer`.
  Remove `mean_loss = total_loss / self._grad_accum_steps`.
- `step`: change `self._optimizer.update(self._model, ...)` and
  `mx.eval(self._model.parameters(), ...)` to use `self._orig_model`.

**Task 2 — `setup.py`: consolidate orphaned import**

File: `src/nanochat/training/base/setup.py`

- Move `from nanochat.common import get_mlx_compute_dtype` into the existing local
  import block at the top of `_setup_mlx`, removing the blank line separator.

**Task 3 — `loop.py`: suppress `bf16_mfu` on MLX path**

File: `src/nanochat/training/base/loop.py`

- Build `mfu_str` conditionally: non-empty only when `s.gpu_peak_flops != float("inf")`.
- Splice `mfu_str` into the `print0` format string in place of the hardcoded
  `bf16_mfu: {mfu:.2f}` segment.

**Verification**

- `pytest tests/test_training/test_mlx_trainer.py` — 13 passed.
- Full `pytest` suite — 323 passed, 10 skipped.
