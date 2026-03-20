---
title: "MLX GPT Design"
summary: "Design and requirements for porting models/gpt.py to mlx.nn with numerically matching forward pass."
read_when:
  - Implementing or reviewing mlx_gpt.py
  - Understanding how the MLX model maps to the PyTorch architecture
  - Debugging numeric mismatches between backends
status: draft
last_updated: "2025-07-22"
---

# MLX GPT Design

Step 3 of the [dual-trainer architecture](dual-trainer-architecture.md). Port `models/gpt.py`
to MLX (`mlx.nn`) with identical architecture and numerically matching forward pass output.
The MLX model is the foundation for `MLXTrainer` — if the forward pass doesn't match PyTorch,
nothing downstream is trustworthy.

**Status**: ✅ implemented in `src/nanochat/models/mlx_gpt.py`, 9 tests passing.

---

## File

`src/nanochat/models/mlx_gpt.py` — standalone, no dependency on `models/gpt.py`.
Shares only `GPTConfig` from `models/config.py`.

---

## Architecture mapping

| PyTorch component | MLX equivalent | Notes |
| --- | --- | --- |
| `nn.Embedding` | `mx.nn.Embedding` | Direct equivalent |
| `nn.Linear` (no bias) | `mx.nn.Linear(bias=False)` | MLX Linear stores weight as `(out, in)` — same as PyTorch |
| `F.rms_norm` | `mx.fast.rms_norm` | Available in MLX, no learnable weight (same as PyTorch `norm()`) |
| `F.relu(x).square()` | `mx.nn.relu(x) ** 2` | ReLU² activation |
| `F.cross_entropy` | `mx.nn.losses.cross_entropy` | Same semantics |
| `nn.Parameter` | `mx.array` stored as attribute | MLX has no `nn.Parameter` — leaf arrays are trainable by default via `model.trainable_parameters()` |
| Rotary embeddings | Same math, MLX ops | `mx.cos`, `mx.sin`, `mx.outer`, `mx.concatenate` |
| Softcap (`tanh` logit cap) | `mx.tanh` | Direct |

---

## Attention

`mx.fast.scaled_dot_product_attention` supports GQA natively — keys and values should not
be pre-tiled to match queries. Input layout is `[B, N, T, D]` (heads before sequence),
which differs from FA3's `[B, T, N, D]` — queries, keys, and values need to be transposed.

No native sliding window parameter exists. Instead, build a boolean causal mask with the
window applied:

```python
def causal_window_mask(T: int, window: int) -> mx.array:
    i = mx.arange(T)[:, None]
    j = mx.arange(T)[None, :]
    return (j <= i) & (i - j < window)
```

For full-context layers (`window_pattern = "L"`), pass `mask="causal"` directly.
For sliding window layers (`window_pattern = "S"`), pass the precomputed boolean mask.

FA3 and DDP are irrelevant for MLX — single-device only, `mx.compile` replaces `torch.compile`.

---

## Features ported

| Feature | PyTorch location | Notes |
| --- | --- | --- |
| Token embedding + norm | `transformer.wte` + `norm(x)` | Straightforward |
| Smear gate | `smear_gate`, `smear_lambda` | Small linear + sigmoid, no issues |
| Value embeddings | `value_embeds` | Alternating layers, `has_ve()` logic unchanged |
| Rotary embeddings | `_precompute_rotary_embeddings` | Precomputed buffers — MLX uses `mx.array` |
| Per-layer residual scalars | `resid_lambdas`, `x0_lambdas` | Scalar `mx.array` parameters |
| Backout | `backout_lambda`, `x_backout` | Mid-layer residual subtraction |
| Sliding window sizes | `_compute_window_sizes` | Pure Python, reusable as-is |
| Logit softcap | `softcap * tanh(logits / softcap)` | Direct |
| KV cache inference | `kv_cache` path in `forward` | Deferred — training only |

---

## Key implementation notes

- `cos`/`sin` rotary buffers frozen via `self.freeze(keys=["cos", "sin"])` — prevents them appearing in `trainable_parameters()`
- `value_embeds` dict keyed `"ve_N"` not `"N"` — MLX `tree_unflatten` interprets numeric string keys as list indices
- `@mx.compile` not applied — forward pass only, no KV cache yet

---

## Deferred

- KV cache inference (`generate`) — training forward pass only in the first iteration
- FP8 — CUDA-only, irrelevant for MLX
- DDP / `DistMuonAdamW` — single-device only

---

## Validation

Tests in `tests/test_models/test_mlx_gpt.py`, skipped when MLX is not installed.

1. Instantiate both models with identical `GPTConfig` and identical weights (copy via numpy)
2. Run a forward pass with the same input tokens
3. Assert `max(abs(mlx_logits - torch_logits)) < 1e-3` (tolerance for bf16 vs float32 differences)
4. Run a backward pass on both, assert gradient magnitudes are in the same range

### Open questions resolved

- ✅ `mx.fast.scaled_dot_product_attention` supports GQA natively
- ✅ Sliding window implemented via boolean causal mask — no native parameter needed
- ✅ `nn.value_and_grad(model, fn)` is the correct API for training
- ✅ Grad accumulation is manual — each call returns a fresh independent grad tree, no state on parameters
- ✅ `mx.eval()` must be called after each microbatch — see [MLX training patterns](mlx-training-patterns.md)
