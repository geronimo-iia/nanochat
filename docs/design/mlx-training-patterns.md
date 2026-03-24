---
title: "MLX Training Patterns"
summary: "Reference patterns for grad accumulation, mx.eval() cadence, and tree utilities in MLX training code."
read_when:
  - Writing or reviewing MLXTrainer.forward_backward
  - Debugging memory growth or stale graph issues in MLX
  - Understanding why mx.eval() placement matters
  - Understanding the fused _MultiStepLossAndGrad design
status: active
last_updated: "2026-03-23"
---

# MLX Training Patterns

Reference for `MLXTrainer.forward_backward` and any MLX training code.
These patterns were validated during [MLX GPT](mlx-gpt-design.md) and [MLX Muon](mlx-muon-design.md) development.

---

## Grad accumulation — lazy N-call approach (current)

The production pattern calls the compiled function N times without `mx.eval` between
steps, accumulating gradient trees as lazy MLX expressions. A single `mx.eval` at the
end flushes all N forward/backward passes in one GPU submission.

```python
# Compile once at trainer init — single-step function
loss_and_grad = mx.compile(_LossAndGrad(model), inputs=[model], outputs=[model])

# Each optimizer step: N calls, eval (loss, grads) per step
accumulated_grads = None
for x, y in microbatches:
    loss, grads = loss_and_grad(x, y)
    mx.eval(loss, grads)               # one Metal submission per step — materializes grads
    accumulated_grads = (
        grads if accumulated_grads is None
        else nn.utils.tree_map(lambda a, b: a + b, accumulated_grads, grads)
    )
    # NOTE: do NOT eval accumulated_grads here — the lazy chain now only contains
    # cheap additions of already-materialized arrays; defer to optimizer step below.

mean_grads = nn.utils.tree_map(lambda g: g / N, accumulated_grads)
optimizer.update(model, mean_grads)
mx.eval(model.parameters(), optimizer.state())   # evaluates additions + optimizer together
```

**Why `mx.eval(loss, grads)` per step, not `mx.eval(accumulated_grads)`**:
Evaluating `loss, grads` materializes the compiled function's outputs (forward + backward
pass). The lazy `accumulated_grads` chain that builds on top only contains additions of
already-materialized arrays — its leaves are all concrete values. This chain is cheap to
defer: no activation memory is held live, and all N additions are batched into the final
`mx.eval(model.parameters(), optimizer.state())` along with the optimizer update.

Evaluating `accumulated_grads` per step instead adds N unnecessary intermediate Metal
command buffer submissions (one per addition), which measured ~10-15% slower on d6/M3 Max
(~39k vs ~45k tok/sec). The lazy chain of materialized-leaf additions is not a performance
problem — depth N-1 of simple add kernels is negligible compared to the forward/backward.

**Correctness**: per-step eval of grads produces numerically equivalent gradients to
the no-eval approach (max_diff < 1e-3 across all parameters). See
`tests/test_training/test_mlx_trainer.py::test_lazy_accum_matches_eager`.

### Why not fuse all N steps into one compiled call?

The `_MultiStepLossAndGrad` class (kept in `mlx_trainer.py`) fuses K steps into one
Metal program. Two approaches were tested on d6/M3 Max (N=8, `_CHUNK_SIZE` in the source):

**K=8 (fully fused)**: Crashes on d6.
```
RuntimeError: [compile] Too many inputs/outputs fused in the Metal Compiled primitive
which exhausted the available argument buffers for the kernel.
```
Metal's per-kernel argument buffer limit is exceeded: N×n_params gradient buffers
(8 × 74 × 2 = 1184 buffers) in a single kernel.

**K=2 (chunked fused)**: Compiles successfully (296 buffers, within limit) but is
**~25% slower** than K=1 (~30k vs ~41k tok/sec on d6/M3 Max). Cause: fusing 2 steps
keeps both forward-pass activation tensors live simultaneously, increasing bandwidth
pressure on Apple Silicon's unified memory. Compile time at step 0 also grows 2.7×.

**Conclusion**: K=1 (current, `_CHUNK_SIZE = 1`) is fastest. Increase `_CHUNK_SIZE`
only if a future MLX version improves activation recompute/offload. See
[docs/dev/lazy-grad-accumulation.md](../dev/lazy-grad-accumulation.md) for the full
fused design and validation data.

---

## `mx.eval()` cadence

MLX is lazy — operations build a compute graph but nothing executes until `mx.eval()` is called.

With the fused compiled approach, the correct cadence is:

| When | What to eval |
|---|---|
| After the single compiled call (all N microbatches) | `mx.eval(loss, mean_grads)` |
| After optimizer update | `mx.eval(model.parameters(), optimizer.state())` |

The fused approach intentionally defers all N microbatch evals to a single call. This is
safe because the entire accumulation is inside one compiled graph — MLX knows the full
computation ahead of time and does not grow an unbounded graph.

**Memory**: with N microbatches compiled into one graph, activation tensors for all N
forward passes are live simultaneously until the single eval. For N=8 on d6, this is
~500 MB of extra peak activation memory vs per-step eval. On M3 Max 128 GB this is
negligible.

---

## `mx.compile` with `inputs=` and `outputs=`

Both flags are required when compiling a function that reads and writes model parameters:

```python
mx.compile(fn, inputs=[model], outputs=[model])
```

| Flag | Effect | Without it |
|---|---|---|
| `inputs=[model]` | Re-reads model params at each call | Params baked as constants at compile time → loss flat forever (stale-params bug) |
| `outputs=[model]` | Allows grad computation to update param array refs | Next `mx.eval()` crashes: "eval array without primitive" |

Both flags were required to fix the original `mx.compile` bugs (see
[docs/dev/experiments/exp2-compression-validation.md](../dev/experiments/exp2-compression-validation.md)).

---

## Tree utilities

`nn.utils.tree_map` is the correct function for operating over grad trees.
`mx.tree_map` does not exist — using it will raise `AttributeError`.
