---
title: "MLX Backend Analysis"
summary: "Fresh analysis of the MLX backend after Phase 1+2 integration work: three findings at the application level with proposals. All findings resolved."
read_when:
  - Understanding why MLXTrainer owns torch→mlx conversion
  - Understanding the orig_model/model split rationale
  - Historical reference for Phase 3 MLX work
status: archived
last_updated: "2025-07-25"
---

# MLX Backend Analysis

Post Phase 1+2 review. The structural foundation (`common/mlx.py`, `_BackendSetup`,
`mlx_compute_init`, `clear_device_cache`, `get_device_sync`) is solid. Three application-level
findings remain.

**All findings resolved. Archived after Phase 3 implementation.**

---

## Finding 1 — `mlx_loader` is an exhaustible generator

### What's wrong

In `_setup_mlx`, the dataloader wrapper is a generator function called once:

```python
def mlx_loader():
    for x, y, state in train_loader:
        yield mx.array(x.numpy()), mx.array(y.numpy()), state

trainer = MLXTrainer(model, optimizer, grad_accum_steps, mlx_loader())
```

`mlx_loader()` produces a one-shot generator. `MLXTrainer.__init__` immediately calls
`next(train_loader)` to prime it. If `MLXTrainer` is ever reconstructed (e.g. after a
resume that re-enters `_setup_mlx`), the generator passed in is already exhausted.

The torch path uses a stateful dataloader object that can be iterated multiple times and
carries resumable state. The MLX path has no equivalent — the generator is consumed and
gone.

This is not triggered today because `_setup_mlx` is called once per process. But it is
a latent correctness risk if the setup path is ever refactored or tested in isolation.

### Decision

Move the `mx.array` conversion inside `MLXTrainer` so it owns the stateful torch loader
directly, mirroring `TorchTrainer`:

```python
class MLXTrainer:
    def __init__(self, model, optimizer, grad_accum_steps, torch_loader):
        self._torch_loader = torch_loader  # stateful, resumable
        ...
    def _next_batch(self):
        x, y, state = next(self._torch_loader)
        return mx.array(x.numpy()), mx.array(y.numpy()), state
```

`_setup_mlx` passes the raw torch loader. The conversion is an implementation detail
of the trainer, not a concern of the setup path.

### Resolution

Implemented in Task 1. `mlx_loader` generator removed from `_setup_mlx`. `MLXTrainer`
now owns `_torch_loader` and `_next_batch`. Loader state preservation verified by
`test_loader_state_preserved_across_forward_backward` — confirms init primes once and
each `forward_backward` advances the loader by exactly `grad_accum_steps` calls.

---

## Finding 2 — `get_mlx_compute_dtype` is defined but never used

### What's wrong

`common/mlx.py` exports `get_mlx_compute_dtype()` and it is wired into `common/__init__.py`,
but nothing calls it. The MLX model dtype is whatever MLX defaults to at init time
(effectively `float32` for parameters, with ops running in the precision MLX chooses).

The torch path logs `COMPUTE_DTYPE` at startup and respects `NANOCHAT_DTYPE`. The MLX
path logs nothing about dtype and has no override mechanism in practice.

Two consequences:
- No visibility into what precision the MLX model is actually running at.
- `NANOCHAT_DTYPE` is silently ignored on the MLX path even though the env var is set.

### Decision

Option A: call `get_mlx_compute_dtype()` in `_setup_mlx`, log it, then cast the model
with `model.astype(compute_dtype)` after init:

```python
compute_dtype = get_mlx_compute_dtype()
print0(f"COMPUTE_DTYPE: {compute_dtype} (MLX)")
model = model.astype(compute_dtype)
```

Simple, one line cast, no architectural change. Full parameter copy at startup only.

### Resolution

Implemented in Task 3. `get_mlx_compute_dtype()` called in `_setup_mlx` after model
construction, logged, and model cast with `model.astype(compute_dtype)` before
`build_param_groups` so the optimizer sees the cast parameters.

---

## Finding 3 — `_loss_and_grad` captures `model` by reference at construction

### What's wrong

```python
self._loss_and_grad = nn.value_and_grad(model, model)
```

Both arguments are the same `model` instance captured at `MLXTrainer.__init__` time.
The first argument is the function to differentiate; the second is the module whose
parameters to differentiate with respect to.

If the model is ever wrapped after construction (quantization, EMA shadow, weight tying
changes), `_loss_and_grad` still holds the original reference and will differentiate
with respect to the original parameters — silently computing wrong gradients.

The torch path avoids this by keeping `orig_model` as an explicit separate reference
and passing it to `TorchTrainer` at construction. The MLX path has no such separation.

### Decision

Adopt the same `orig_model` / `model` split as `TorchTrainer`. `_setup_mlx` passes both;
`MLXTrainer` uses `orig_model` for `_loss_and_grad`, `eval_context`, `model_state_dict`,
and `load_state_dicts`. `model` is reserved for a future compiled wrapper.

Today `orig_model` and `model` are the same object — `mx.compile` is applied at the
`muon_step` function level, not at the model level. The split costs nothing and makes
the intent explicit and future-proof.

```python
class MLXTrainer:
    def __init__(self, orig_model, model, optimizer, grad_accum_steps, torch_loader):
        self._orig_model = orig_model
        self._model = model
        self._loss_and_grad = nn.value_and_grad(orig_model, orig_model)
        ...
```

### Resolution

Implemented in Task 1. Constructor takes `orig_model, model`. `_loss_and_grad`,
`eval_context`, `model_state_dict`, and `load_state_dicts` all use `self._orig_model`.

---

## Finding 4 — grad accumulation tree allocation (noted, won't fix)

In `forward_backward`, accumulated gradients are built with:

```python
accumulated_grads = nn.utils.tree_map(lambda a, b: a + b, accumulated_grads, grads)
```

Each iteration creates a new tree of `mx.array` references. Since `mx.eval(loss, grads)`
is called after each microbatch, the lazy graph is already bounded — the N-1 extra
`tree_map` calls are Python-level tree traversals, not Metal allocations. The real cost
is negligible at the typical `grad_accum_steps` of 1–4 on MLX.

Summing on the fly (current approach) is already correct and clear. Not worth changing.

---

## Priority

| Finding | Risk | Effort | Decision |
|---|---|---|---|
| `mlx_loader` exhaustible generator | Latent correctness bug | Small | Fixed — Task 1 |
| `get_mlx_compute_dtype` unused | Missing visibility + silent env var ignore | Small | Fixed — Task 3 |
| `_loss_and_grad` stale capture | Latent, not active | Small | Fixed — Task 1 |
| Grad accum tree allocation | Python overhead only, bounded | — | Won't fix |

---

## Implementation plan

All three fixes touch `MLXTrainer` and `_setup_mlx`. Do them in one pass.

**Task 1 — `MLXTrainer`: own the torch→mlx conversion and introduce `orig_model`**

File: `src/nanochat/training/mlx_trainer.py`

- Change constructor signature to `__init__(self, orig_model, model, optimizer, grad_accum_steps, torch_loader)`.
- Store `self._orig_model = orig_model` and `self._model = model` (today both are the
  same object; the split is for future-proofing).
- Store `self._torch_loader = torch_loader` (raw stateful torch dataloader).
- Remove the `train_loader: Any` parameter that previously accepted a pre-converted generator.
- Add `_next_batch(self)` private method that pulls from `self._torch_loader` and converts:
  ```python
  def _next_batch(self):
      x, y, state = next(self._torch_loader)
      return mx.array(x.numpy()), mx.array(y.numpy()), state
  ```
- Replace all `next(self._train_loader)` calls with `self._next_batch()`.
- Update `_loss_and_grad` to use `orig_model`:
  ```python
  self._loss_and_grad = nn.value_and_grad(orig_model, orig_model)
  ```
- Update `eval_context`, `model_state_dict`, `load_state_dicts` to use `self._orig_model`
  where they currently use `self._model` (same object today, explicit tomorrow).

**Task 2 — `_setup_mlx`: remove generator wrapper, pass torch loader directly**

File: `src/nanochat/training/base/setup.py`

- Remove the `mlx_loader` generator function entirely.
- Pass `train_loader` (the raw torch loader) directly to `MLXTrainer`.
- Update the `MLXTrainer(...)` call to pass `orig_model=model, model=model` (same object
  for now).

**Task 3 — `_setup_mlx`: wire `get_mlx_compute_dtype` and cast model**

File: `src/nanochat/training/base/setup.py`

- Import `get_mlx_compute_dtype` (already exported from `common`).
- After model construction, add:
  ```python
  compute_dtype = get_mlx_compute_dtype()
  print0(f"COMPUTE_DTYPE: {compute_dtype} (MLX)")
  model = model.astype(compute_dtype)
  ```
- This must happen before `build_param_groups` so the optimizer sees the cast parameters.

**Verification**

- `pytest tests/test_training/test_mlx_trainer.py` — 13 passed (12 original + 1 new).
- Full `pytest` suite — 323 passed, 10 skipped.
- `test_loader_state_preserved_across_forward_backward` confirms `_next_batch` advances
  the loader monotonically: 1 call at init, then `grad_accum_steps` calls per
  `forward_backward`.
