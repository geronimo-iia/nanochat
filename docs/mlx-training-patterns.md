---
title: "MLX Training Patterns"
summary: "Reference patterns for grad accumulation, mx.eval() cadence, and tree utilities in MLX training code."
read_when:
  - Writing or reviewing MLXTrainer.forward_backward
  - Debugging memory growth or stale graph issues in MLX
  - Understanding why mx.eval() placement matters
status: draft
last_updated: "2025-07-22"
---

# MLX Training Patterns

Reference for `MLXTrainer.forward_backward` and any MLX training code.
These patterns were validated during [MLX GPT](mlx-gpt-design.md) and [MLX Muon](mlx-muon-design.md) development.

---

## Grad accumulation

`nn.value_and_grad` does **not** accumulate. Each call returns a fresh, independent grad tree.
MLX has no `.grad` attribute on parameters — there is no state to accumulate into.

Manual accumulation with `nn.utils.tree_map`:

```python
loss_and_grad = nn.value_and_grad(model, model)

accumulated_grads = None
total_loss = 0.0

for x, y in microbatches:
    loss, grads = loss_and_grad(x, y)
    mx.eval(loss, grads)  # must eval each microbatch — see below
    total_loss += loss.item()
    if accumulated_grads is None:
        accumulated_grads = grads
    else:
        accumulated_grads = nn.utils.tree_map(lambda a, b: a + b, accumulated_grads, grads)

mean_grads = nn.utils.tree_map(lambda g: g / n_microbatches, accumulated_grads)
optimizer.update(model, mean_grads)
mx.eval(model.parameters())
```

---

## `mx.eval()` cadence

MLX is lazy — operations build a compute graph but nothing executes until `mx.eval()` is called.
Without an explicit eval between microbatches, the graph grows unboundedly in memory before
anything runs. The correct cadence:

| When | What to eval |
|---|---|
| After each microbatch forward/backward | `mx.eval(loss, grads)` |
| After optimizer update | `mx.eval(model.parameters())` |
| Before logging a loss value | `loss.item()` triggers eval implicitly |

Do **not** defer all evals to the end of the accumulation loop.

---

## Tree utilities

`nn.utils.tree_map` is the correct function for operating over grad trees.
`mx.tree_map` does not exist — using it will raise `AttributeError`.
