---
title: "MLX Muon Design"
summary: "Design and requirements for porting the Muon + AdamW optimizer to MLX."
read_when:
  - Implementing or reviewing mlx_optimizer.py
  - Understanding why MuonAdamW doesn't subclass mlx.optimizers.Optimizer
  - Debugging optimizer step mismatches between backends
status: draft
last_updated: "2025-07-22"
---

# MLX Muon Design

Step 4 of the [dual-trainer architecture](dual-trainer-architecture.md). Port `training/optimizer.py`
(AdamW + Muon) to MLX. The MLX optimizer is the last piece before `MLXTrainer` can be assembled.

**Status**: ✅ implemented in `src/nanochat/training/mlx_optimizer.py`, 11 tests passing.

---

## File

`src/nanochat/training/mlx_optimizer.py` — standalone, no dependency on `optimizer.py`.

---

## Architecture decision: don't subclass `mlx.optimizers.Optimizer`

`mlx.optimizers.Optimizer` is tree-based: `apply_gradients` calls `apply_single` per leaf
via `tree_map`. Muon cannot fit this model — it must operate on stacked groups of same-shape
parameters simultaneously (Polar Express requires the full matrix, not individual scalars).

Instead, `MuonAdamW` is a plain class with:
- One `mlx.optimizers.AdamW` instance per AdamW param group (different LR/betas per group)
- Manual Muon state (`momentum_buffer`, `second_momentum_buffer`) stored as plain dicts
- A single `update(model, grads)` method matching the MLX optimizer convention

---

## AdamW groups

`mlx.optimizers.AdamW` is used directly per group. Key difference from PyTorch:

| Aspect | PyTorch `adamw_step_fused` | MLX `optim.AdamW` |
|---|---|---|
| Bias correction | Always applied | `bias_correction=False` by default |
| Weight decay | Decoupled, applied before update | Decoupled, applied in `apply_single` |
| Per-group LR | Set on group dict | Passed to constructor |

MLX `AdamW` must be constructed with `bias_correction=True` to match PyTorch behavior.
Each AdamW group gets its own `optim.AdamW` instance — MLX optimizers hold state internally
and cannot be shared across groups with different hyperparameters.

---

## Muon step — op mapping

All PyTorch ops in `muon_step_fused` have direct MLX equivalents:

| PyTorch op | MLX equivalent | Notes |
|---|---|---|
| `tensor.lerp_(end, weight)` | `a + weight * (b - a)` | No in-place ops in MLX |
| `x.bfloat16()` | `x.astype(mx.bfloat16)` | Direct |
| `x.mT` | `mx.swapaxes(x, -2, -1)` | Matrix transpose |
| `x.norm(dim=(-2,-1), keepdim=True)` | `mx.linalg.norm(x, axis=(-2,-1), keepdims=True)` | Verified |
| `x.float()` | `x.astype(mx.float32)` | Direct |
| `x.square()` | `mx.square(x)` | Direct |
| `x.mean(dim=d, keepdim=True)` | `mx.mean(x, axis=d, keepdims=True)` | Direct |
| `x.clamp_min(v)` | `mx.maximum(x, v)` | Verified |
| `x.rsqrt()` | `mx.rsqrt(x)` | Verified |
| `torch.stack(params)` | `mx.stack(params)` | Verified |
| `stacked.unbind(0)` | `list(stacked)` or index loop | MLX arrays are iterable |
| `torch._foreach_copy_` | assign back via list | No in-place; return updated arrays |
| `@torch.compile` | `@mx.compile` on `muon_step` | Added after correctness validated |

---

## Muon step — immutability

MLX arrays are immutable — every op returns a new array. The Muon step returns updated
`(stacked_params, momentum_buffer, second_momentum_buffer)` rather than mutating in place.
State dicts are updated by reassignment after each step.

---

## Param group interface

Same `kind="muon"` / `kind="adamw"` dict structure as PyTorch `MuonAdamW`, so `GPT.setup_optimizer`
can be ported with minimal changes. Muon groups must still have all params with the same shape
(caller's responsibility, same as PyTorch).

`ve_keys` prefix is `"value_embeds.ve_"` — matches the `"ve_N"` key convention in `mlx_gpt.py`.

---

## Validation

Tests in `tests/test_training/test_mlx_optimizer.py`, skipped when MLX not installed.

1. Instantiate both PyTorch and MLX models with identical weights (copy via numpy)
2. Compute gradients on both with the same input
3. Run one optimizer step on both
4. Assert `max(abs(mlx_params - torch_params)) < 1e-3` for all parameter tensors
5. Run a second step, assert params still track

---

## Deferred

- `DistMuonAdamW` — single-device only, no DDP in MLX
