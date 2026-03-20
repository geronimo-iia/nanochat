---
title: "MLX GPT Reference"
summary: "Architecture reference for mlx_gpt.py — how the PyTorch GPT maps to MLX, key implementation decisions, and validation."
read_when:
  - Reviewing or modifying mlx_gpt.py
  - Debugging numeric mismatches between backends
  - Understanding attention layout or mask differences
status: active
last_updated: "2025-07-24"
---

# MLX GPT Reference

`src/nanochat/models/mlx_gpt.py` — MLX port of `models/gpt.py` for Apple Silicon training.
Standalone: no dependency on `models/gpt.py`. Shares only `GPTConfig` from `models/config.py`.

---

## Architecture mapping

| PyTorch component | MLX equivalent | Notes |
| --- | --- | --- |
| `nn.Embedding` | `mx.nn.Embedding` | Direct equivalent |
| `nn.Linear` (no bias) | `mx.nn.Linear(bias=False)` | Same weight layout `(out, in)` |
| `F.rms_norm` | `mx.fast.rms_norm` | No learnable weight — same as PyTorch `norm()` |
| `F.relu(x).square()` | `mx.nn.relu(x) ** 2` | ReLU² activation |
| `F.cross_entropy` | `mx.nn.losses.cross_entropy` | Same semantics |
| `nn.Parameter` | `mx.array` attribute | Leaf arrays are trainable by default via `model.trainable_parameters()` |
| Rotary embeddings | Same math, MLX ops | `mx.cos`, `mx.sin`, `mx.outer`, `mx.concatenate` |
| Softcap | `mx.tanh` | Direct |

---

## Attention

`mx.fast.scaled_dot_product_attention` supports GQA natively — keys and values are not
pre-tiled. Input layout is `[B, N, T, D]` (heads before sequence), which differs from
FA3's `[B, T, N, D]` — Q, K, V are transposed before the call.

Sliding window has no native parameter. A boolean causal mask is built per layer:

```python
def causal_window_mask(T: int, window: int) -> mx.array:
    i = mx.arange(T)[:, None]
    j = mx.arange(T)[None, :]
    return (j <= i) & ((i - j) < window)
```

Full-context layers (`window_pattern = "L"`) pass `mask="causal"` directly.
Sliding window layers (`window_pattern = "S"`) pass the precomputed boolean mask.

---

## Key implementation decisions

- `cos`/`sin` rotary buffers frozen via `self.freeze(keys=["cos", "sin"])` — excluded from `trainable_parameters()`
- `value_embeds` dict keyed `"ve_N"` not `"N"` — MLX `tree_unflatten` interprets numeric string keys as list indices
- `@mx.compile` not applied to the model — `MLXTrainer` compiles the loss function via `nn.value_and_grad`

---

## Features

| Feature | Notes |
| --- | --- |
| Token embedding + norm | Straightforward |
| Smear gate | Small linear + sigmoid |
| Value embeddings | Alternating layers, `has_ve()` logic unchanged |
| Rotary embeddings | Precomputed buffers as frozen `mx.array` |
| Per-layer residual scalars | `resid_lambdas`, `x0_lambdas` as scalar `mx.array` |
| Backout | Mid-layer residual subtraction |
| Sliding window | Boolean causal mask per layer |
| Logit softcap | `softcap * tanh(logits / softcap)` |

KV cache inference is not implemented — training forward pass only.

---

## Validation

Tests in `tests/test_models/test_mlx_gpt.py`, skipped when MLX is not installed.

- Identical weights copied torch → mlx via numpy
- Same input tokens → forward pass on both
- `max(abs(mlx_logits - torch_logits)) < 1e-3`
- Backward pass gradient magnitudes in the same range
