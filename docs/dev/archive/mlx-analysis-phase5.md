---
title: "MLX Backend Analysis — Phase 5"
summary: "Fresh analysis after Phase 4. Five findings: one design gap around mx.compile at the model level, one latent crash on uninitialized accumulated grads, one stale batch after resume, one spurious wandb metric, one dead docstring reference."
read_when:
  - Reviewing MLX training correctness or performance before deeper development
  - Planning mx.compile integration at the model/forward-backward level
  - Investigating resume correctness or wandb metric quality on the MLX path
status: archived
last_updated: "2025-07-25"
---

# MLX Backend Analysis — Phase 5

Post Phase 4 review. The trainer lifecycle, dtype handling, loader ownership, and
logging are solid. Five findings — all resolved. Archived after Phase 5 implementation.

---

## Finding 1 — `orig_model`/`model` split is nominal: no `mx.compile` wrapper exists yet

### What's wrong

```python
trainer: BaseTrainer = MLXTrainer(model, model, optimizer, grad_accum_steps, train_loader)
```

Both arguments are the same object. The `orig_model`/`model` split was introduced in
Phase 3 as future-proofing, mirroring the torch path where `torch.compile(model)`
produces a genuinely distinct compiled wrapper. On the MLX path, no equivalent
compilation has been applied at the model level — `mx.compile` is only used at the
`muon_step` function level today.

This is not a bug. But the split is purely nominal until a compiled wrapper is
introduced. The torch path's pattern is:

```
orig_model  — unwrapped, used for state_dict, eval, grad computation
model       — compiled wrapper, used for the forward pass in training
```

### What MLX compile looks like at this level

`mx.compile` works on callables. The correct pattern for compiling the
forward-backward pass is:

```python
loss_and_grad = nn.value_and_grad(orig_model, orig_model)
compiled_loss_and_grad = mx.compile(
    loss_and_grad,
    inputs=orig_model.parameters(),
    outputs=orig_model.parameters(),
)
```

`inputs` captures the current parameter state so the compiled graph sees updates
between steps. `outputs` allows the compiled function to mutate the parameter tree
in-place. This is exactly the pattern already used for `muon_step`.

Alternatively, `mx.compile` can wrap the model callable directly:

```python
compiled_model = mx.compile(model, inputs=model.parameters(), outputs=model.parameters())
```

Then `model` (the compiled wrapper) is passed as the second argument to `MLXTrainer`,
while `orig_model` remains the unwrapped source of truth — matching the torch pattern
exactly.

### Decision

Introduce `mx.compile` at the model level in `_setup_mlx`:

```python
orig_model = MLXGPT(gpt_config)
orig_model = orig_model.astype(compute_dtype)
model = mx.compile(orig_model, inputs=orig_model.parameters(), outputs=orig_model.parameters())
trainer = MLXTrainer(orig_model, model, optimizer, grad_accum_steps, train_loader)
```

`MLXTrainer` already uses `orig_model` for `_loss_and_grad`, `model_state_dict`,
`load_state_dicts`, and `eval_context`. `model` (the compiled wrapper) is used in
`step` via `optimizer.update`. The split becomes real and meaningful.

### Resolution

Implemented in Task 1. `_LossAndGrad` thin `nn.Module` wrapper introduced so
`mx.compile` can automatically capture model parameters as state. `_loss_and_grad`
is now `mx.compile(_LossAndGrad(orig_model))` in `MLXTrainer.__init__`.

Note: `mx.compile` on a raw function with captured arrays raises
`[compile] Attempting to compile a function with uncaptured inputs is not allowed`
in MLX 0.31. Wrapping in `nn.Module` is the correct pattern — MLX handles state
capture automatically for modules.

---

## Finding 2 — `_accumulated_grads` not initialised in `__init__`

### What's wrong

`self._accumulated_grads` is assigned in `forward_backward` and consumed in `step`.
It is not initialised in `__init__`. If `step` is called before `forward_backward`
(e.g. in a test, a resume path, or any future refactor that reorders the calls),
it raises `AttributeError: 'MLXTrainer' object has no attribute '_accumulated_grads'`.

`TorchTrainer` has the same pattern but PyTorch's autograd makes it harder to call
`step` without a preceding backward — the optimizer would just apply zero gradients.
MLX has no such implicit guard.

### Resolution

Fixed in Task 1. `self._accumulated_grads: dict | None = None` initialised in
`__init__`. `assert self._accumulated_grads is not None` added at the top of `step`.

---

## Finding 3 — Stale batch after resume

### What's wrong

In `__init__`, the loader is primed unconditionally:

```python
self._x, self._y, self._loader_state = self._next_batch()
```

In `_setup_mlx`, after constructing the trainer, `load_state_dicts` is called on
resume:

```python
trainer = MLXTrainer(model, model, optimizer, grad_accum_steps, train_loader)
if resuming:
    trainer.load_state_dicts(resume_ckpt.model_state, resume_ckpt.optimizer_state)
```

`load_state_dicts` restores model and optimizer state but does not touch `self._x`,
`self._y`, or `self._loader_state`. These still hold the batch pulled during `__init__`
priming — which was drawn from the loader before it was positioned at the resume point.

The torch dataloader is resumed via `dataloader_resume_state_dict` passed at
construction, so it will produce the correct next batch. But `self._x`/`self._y`
already consumed one batch from the wrong position. The first `forward_backward` after
resume trains on that stale batch silently.

### Resolution

Fixed in Task 2. `reprime()` public method added to `MLXTrainer`. Called from
`_setup_mlx` after `load_state_dicts` on the resume path.

---

## Finding 4 — `train/mfu: 0.0` logged to wandb every step on MLX

### What's wrong

```python
s.wandb_run.log({"train/mfu": mfu, ...}, step=state.step)
```

`mfu` is `flops_per_sec / (s.gpu_peak_flops * s.ddp_world_size)`. On the MLX path
`gpu_peak_flops = float("inf")`, so `mfu` is always `0.0`. The console print was
fixed in Phase 4 to suppress `bf16_mfu` when `gpu_peak_flops == inf`, but the wandb
log was not updated. Every MLX training run sends `train/mfu: 0.0` to wandb for every
step — a meaningless metric that pollutes the run dashboard.

### Resolution

Fixed in Task 3. `train/mfu` conditionally included in the wandb log dict only when
`s.gpu_peak_flops != float("inf")`.

---

## Finding 5 — Dead docstring reference in `MLXTrainer`

### What's wrong

```python
"""
MLXTrainer — BaseTrainer implementation for Apple Silicon via MLX.

Single-device only. No FP8, no DDP, no GradScaler.
See docs/mlx-training-patterns.md for grad accumulation and mx.eval() cadence.
"""
```

`docs/mlx-training-patterns.md` does not exist. The archived analysis docs are in
`docs/archive/`. This reference was never created.

### Resolution

Fixed in Task 1. Dead reference removed from module docstring.

---

## Priority

| # | Finding | Risk | Effort | Decision |
|---|---|---|---|---|
| 1 | `orig_model`/`model` split nominal — no `mx.compile` on forward-backward | Performance gap | Medium | Fixed — Task 1, `_LossAndGrad` + `mx.compile` |
| 2 | `_accumulated_grads` uninitialised — `AttributeError` if `step` before `forward_backward` | Latent crash | Trivial | Fixed — Task 1 |
| 3 | Stale batch after resume | Silent correctness issue | Small | Fixed — Task 2, `reprime()` |
| 4 | `train/mfu: 0.0` sent to wandb every step on MLX | Misleading metric | Trivial | Fixed — Task 3 |
| 5 | Dead docstring reference to non-existent doc | Documentation rot | Trivial | Fixed — Task 1 |

---

## Implementation plan

**Task 1 — `mlx_trainer.py`: compile `_loss_and_grad`, init `_accumulated_grads`, fix docstring**

File: `src/nanochat/training/mlx_trainer.py`

- Remove the dead `See docs/mlx-training-patterns.md` line from the module docstring.
- In `__init__`, after building `loss_and_grad`, wrap it with `mx.compile`:
  ```python
  loss_and_grad = nn.value_and_grad(orig_model, orig_model)
  self._loss_and_grad = mx.compile(
      loss_and_grad,
      inputs=orig_model.parameters(),
      outputs=orig_model.parameters(),
  )
  ```
- Initialise `self._accumulated_grads: dict | None = None` in `__init__`.
- Add `assert self._accumulated_grads is not None` at the top of `step`.

**Task 2 — `mlx_trainer.py` + `setup.py`: expose `reprime()`, call it after resume**

File: `src/nanochat/training/mlx_trainer.py`

- Add `reprime()` public method:
  ```python
  def reprime(self) -> None:
      self._x, self._y, self._loader_state = self._next_batch()
  ```

File: `src/nanochat/training/base/setup.py`

- After `load_state_dicts` on the resume path, call `trainer.reprime()`:
  ```python
  if resuming:
      trainer.load_state_dicts(resume_ckpt.model_state, resume_ckpt.optimizer_state)
      del resume_ckpt.model_state, resume_ckpt.optimizer_state
      trainer.reprime()
  ```

**Task 3 — `loop.py`: guard wandb mfu log**

File: `src/nanochat/training/base/loop.py`

- Conditionally include `train/mfu` in the wandb log dict only when
  `s.gpu_peak_flops != float("inf")`.

**Verification**

- `pytest tests/test_training/test_mlx_trainer.py` — 13 passed.
- Full `pytest` suite — 323 passed, 10 skipped.
