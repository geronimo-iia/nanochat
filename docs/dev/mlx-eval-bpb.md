---
title: MLX Eval BPB — Design & Fix
summary: Documents the evaluate_bpb incompatibility on the MLX path and the design for the fix.
read_when:
  - Implementing or reviewing val/bpb evaluation on the MLX backend
  - Investigating why --eval-every fails with --backend=mlx
status: draft
last_updated: 2025-07-11
---

# MLX Eval BPB — Design & Fix

## The bug

`evaluate_bpb` in `evaluation/loss_eval.py` is torch-only. When `--eval-every` is
set on the MLX path, `loop.py` calls:

```python
with s.trainer.eval_context() as model:
    state.val_bpb = evaluate_bpb(model, val_loader, eval_steps, s.token_bytes)
```

`eval_context` yields the MLX `GPT`. `evaluate_bpb` immediately does:

```python
total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
```

MLX `GPT` has no `get_device()` → `AttributeError`.

Even if that were fixed, the rest of the function uses `torch.tensor`, `torch.no_grad`,
`dist.all_reduce`, and indexes a torch `token_bytes` tensor — none of which work with
an MLX model or MLX arrays.

## What evaluate_bpb does

```
for each val batch (x, y):
    loss2d = model(x, y, loss_reduction="none")  # (B, T) per-token loss
    num_bytes = token_bytes[y]                   # bytes per target token
    total_nats += (loss2d * (num_bytes > 0)).sum()
    total_bytes += num_bytes.sum()

bpb = total_nats / (log(2) * total_bytes)
```

The val loader yields `(x, y)` as torch CPU tensors (same dataloader as training,
`device=cpu` on MLX path). `token_bytes` is a 1D torch tensor `(vocab_size,)` on CPU.

## Fix design

Add `evaluate_bpb_mlx` in `evaluation/loss_eval.py`. The logic is identical — only
the tensor ops change.

### Inputs on the MLX path

| Input | Current (torch) | MLX |
|-------|----------------|-----|
| `model` | torch `GPT` | MLX `GPT` |
| `val_loader` | yields torch CPU tensors | yields torch CPU tensors (unchanged) |
| `token_bytes` | torch CPU tensor | convert to `mx.array` once before loop |
| `model(x, y, ...)` | torch tensor | `mx.array` — but MLX `GPT.__call__` takes `mx.array` input |

The val loader yields torch CPU tensors. We convert `x`, `y` to `mx.array` inside
the loop (same as `_next_batch` in `MLXTrainer`). `token_bytes` is converted once
before the loop.

### MLX GPT call signature difference

The torch `GPT` supports `loss_reduction="none"` to get per-token losses `(B, T)`.
The MLX `GPT.__call__` currently only returns scalar mean loss:

```python
loss = mx.mean(nn.losses.cross_entropy(...))
return loss
```

It needs to support `loss_reduction="none"` to return `(B, T)` losses, or we compute
it inline in `evaluate_bpb_mlx`.

Simplest fix: compute cross-entropy inline in `evaluate_bpb_mlx` using the logits
path (`model(x)` without targets), then apply the byte weighting manually. This avoids
touching `mlx_gpt.py`.

```python
logits = model(x_mx)                          # (B, T, vocab) — no targets
logits_2d = logits.reshape(-1, vocab_size)    # (B*T, vocab)
y_1d = y_mx.reshape(-1)                       # (B*T,)
loss_2d = nn.losses.cross_entropy(logits_2d, y_1d, reduction="none")  # (B*T,)
```

### No DDP on MLX

`evaluate_bpb` has a `dist.all_reduce` for multi-GPU. MLX is always single-device —
skip it entirely.

### Implementation

```python
def evaluate_bpb_mlx(model: object, batches: object, steps: int, token_bytes_torch: torch.Tensor) -> float:
    import math
    import mlx.core as mx
    import mlx.nn as nn

    token_bytes_mx = mx.array(token_bytes_torch.numpy())  # (vocab_size,) — once
    vocab_size = token_bytes_mx.shape[0]

    total_nats = mx.array(0.0)
    total_bytes = mx.array(0)
    batch_iter = iter(batches)

    for _ in range(steps):
        x, y = next(batch_iter)
        x_mx = mx.array(x.numpy())
        y_mx = mx.array(y.numpy())

        logits = model(x_mx)                                              # (B, T, vocab)
        loss_1d = nn.losses.cross_entropy(
            logits.reshape(-1, vocab_size), y_mx.reshape(-1), reduction="none"
        )                                                                  # (B*T,)
        y_1d = y_mx.reshape(-1)
        num_bytes = token_bytes_mx[y_1d]                                  # (B*T,)
        total_nats = total_nats + (loss_1d * (num_bytes > 0)).sum()
        total_bytes = total_bytes + num_bytes.sum()
        mx.eval(total_nats, total_bytes)  # flush every step — MLX is lazy;
                                          # without this, eval_steps forward passes
                                          # accumulate in one graph → OOM

    nats = total_nats.item()
    byt = total_bytes.item()
    if byt == 0:
        return float("inf")
    return nats / (math.log(2) * byt)
```

## OOM root cause

MLX uses lazy evaluation — ops are not executed until `mx.eval()` is called. Without
`mx.eval` inside the loop, every iteration appends to the same computation graph.
`eval_steps` can be in the hundreds (e.g. `eval_tokens=41943040`, `device_batch_size=32`,
`max_seq_len=2048` → 640 steps) — 640 full forward passes queued in one graph causes
a full system OOM.

The fix mirrors `MLXTrainer.forward_backward` which calls `mx.eval(loss, grads)` per
microbatch for the same reason.

### Changes to loop.py

Dispatch on `device_type`:

```python
from nanochat.evaluation.loss_eval import evaluate_bpb, evaluate_bpb_mlx

with s.trainer.eval_context() as model:
    if s.device_type == "mlx":
        state.val_bpb = evaluate_bpb_mlx(model, val_loader, eval_steps, s.token_bytes)
    else:
        state.val_bpb = evaluate_bpb(model, val_loader, eval_steps, s.token_bytes)
```

### Val loader on MLX path

The val loader is built in `setup.py`:

```python
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, config.training.device_batch_size, config.training.max_seq_len,
    split="val", device=device
)
```

On the MLX path `device=torch.device("cpu")` — yields torch CPU tensors. The
`evaluate_bpb_mlx` function converts them to `mx.array` inline. No change needed
to the val loader.

## Files to change

| File | Change |
|------|--------|
| `evaluation/loss_eval.py` | Add `evaluate_bpb_mlx` |
| `training/base/loop.py` | Dispatch to `evaluate_bpb_mlx` when `device_type == "mlx"` |

No changes to `mlx_gpt.py`, `setup.py`, or the dataloader.

## What this unblocks

- `--eval-every` works on the MLX path → `val/bpb` logged to wandb
- Phase 1.5.1 Experiment 2 can run: `compression_ratio` vs `val/bpb` correlation
