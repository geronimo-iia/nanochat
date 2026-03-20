---
title: "MLX Muon Reference"
summary: "Architecture reference for mlx_optimizer.py — MuonAdamW design, op mapping, and param group interface."
read_when:
  - Reviewing or modifying mlx_optimizer.py
  - Understanding why MuonAdamW doesn't subclass mlx.optimizers.Optimizer
  - Debugging optimizer step mismatches between backends
status: active
last_updated: "2025-07-24"
---

# MLX Muon Reference

`src/nanochat/training/mlx_optimizer.py` — MLX port of `training/optimizer.py`.
Standalone: no dependency on `optimizer.py`.

---

## Why not subclass `mlx.optimizers.Optimizer`

`mlx.optimizers.Optimizer` is tree-based: `apply_gradients` calls `apply_single` per leaf
via `tree_map`. Muon cannot fit this model — it must operate on stacked groups of same-shape
parameters simultaneously (Polar Express requires the full matrix, not individual scalars).

`MuonAdamW` is a plain class with:
- One `mlx.optimizers.AdamW` instance per AdamW param group
- Manual Muon state (`momentum_buffer`, `second_momentum_buffer`) stored as plain dicts
- A single `update(model, grads)` method

---

## AdamW groups

| Aspect | PyTorch `adamw_step_fused` | MLX `optim.AdamW` |
|---|---|---|
| Bias correction | Always applied | `bias_correction=False` by default — must pass `bias_correction=True` |
| Weight decay | Decoupled | Decoupled |
| Per-group LR | Set on group dict | Passed to constructor, updated via `adamw.learning_rate` |

Each AdamW group gets its own `optim.AdamW` instance — MLX optimizers hold state internally
and cannot be shared across groups with different hyperparameters.

---

## Muon step — op mapping

| PyTorch op | MLX equivalent |
|---|---|
| `tensor.lerp_(end, weight)` | `a + weight * (b - a)` |
| `x.bfloat16()` | `x.astype(mx.bfloat16)` |
| `x.mT` | `mx.swapaxes(x, -2, -1)` |
| `x.norm(dim=(-2,-1), keepdim=True)` | `mx.linalg.norm(x, axis=(-2,-1), keepdims=True)` |
| `x.float()` | `x.astype(mx.float32)` |
| `x.square()` | `mx.square(x)` |
| `x.mean(dim=d, keepdim=True)` | `mx.mean(x, axis=d, keepdims=True)` |
| `x.clamp_min(v)` | `mx.maximum(x, v)` |
| `x.rsqrt()` | `mx.rsqrt(x)` |
| `torch.stack(params)` | `mx.stack(params)` |
| `stacked.unbind(0)` | iterate over `stacked` (MLX arrays are iterable) |
| `@torch.compile` | `@mx.compile` on `muon_step` |

MLX arrays are immutable — the Muon step returns updated
`(stacked_params, momentum_buffer, second_momentum_buffer)` rather than mutating in place.

---

## Param group interface

Same `kind="muon"` / `kind="adamw"` dict structure as PyTorch `MuonAdamW`, with one addition:
each group carries a `_keys` field listing the flat parameter key strings
(as returned by `nn.utils.tree_flatten`) that belong to the group.

Use `build_param_groups(model, ...)` to construct groups with `_keys` injected.
Muon groups must have all params with the same shape — caller's responsibility, same as PyTorch.

`value_embeds` keys use prefix `"value_embeds.ve_"` — matches the `"ve_N"` convention in `mlx_gpt.py`.

---

## Validation

Tests in `tests/test_training/test_mlx_optimizer.py`, skipped when MLX is not installed.

- Identical weights copied torch → mlx via numpy
- PyTorch gradients injected into MLX grad tree
- One optimizer step on both
- All params match within `1e-2`
- Second step params still track
