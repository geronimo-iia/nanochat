---
title: "Lazy Gradient Accumulation for MLX"
summary: "Design analysis for removing per-step mx.eval() sync barriers in MLXTrainer.forward_backward()."
read_when: "Considering MLX training performance optimizations."
status: fully validated — ready to implement
last_updated: "2026-03-23"
---

# Lazy Gradient Accumulation for MLX

## Context

`MLXTrainer.forward_backward()` runs `grad_accum_steps` (typically 8) mini-forward/backward
passes before one optimizer step. Each pass currently ends with a hard sync:

```python
# Current — mlx_trainer.py
for i in range(self._grad_accum_steps):
    loss, grads = self._loss_and_grad(self._x, self._y)
    mx.eval(loss, grads)          # ← GPU sync #i
    train_loss = loss.item()      # ← forces CPU to wait for GPU
    ...
    accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, grads)
    self._x, self._y, self._loader_state = self._next_batch()
```

With 8 accumulation steps, this creates **8 GPU sync points per optimizer step**.

## What "lazy" means here

MLX evaluates arrays lazily: operations build a computation graph in memory but do not
execute on the GPU until `mx.eval()` is called (or until `.item()` forces it). By
removing the per-step `mx.eval(loss, grads)`, all 8 forward/backward passes queue up as
a single compound graph, which is then flushed to the GPU with one `mx.eval` call at the
end.

## Proposed implementation

```python
def forward_backward(self) -> StepResult:
    self._last_x = self._x
    self._last_y = self._y
    self._nan_detected = False

    losses: list[mx.array] = []
    accumulated_grads = None

    for i in range(self._grad_accum_steps):
        loss, grads = self._loss_and_grad(self._x, self._y)
        losses.append(loss)                             # stay lazy
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = nn.utils.tree_map(
                lambda a, b: a + b, accumulated_grads, grads
            )
        self._x, self._y, self._loader_state = self._next_batch()

    assert accumulated_grads is not None
    if self._grad_accum_steps > 1:
        accumulated_grads = nn.utils.tree_map(
            lambda g: g / self._grad_accum_steps, accumulated_grads
        )

    # Single eval for all grad_accum_steps — replaces N sync barriers with 1.
    mx.eval(losses, accumulated_grads)

    # NaN check: losses already evaluated, no extra sync cost.
    train_loss = losses[-1].item()
    for i, loss in enumerate(losses):
        lv = loss.item()
        if not math.isfinite(lv):
            key = _first_nan_key(accumulated_grads)
            print(
                f"[NaN] forward_backward accum={i}/{self._grad_accum_steps}: "
                f"loss={lv}  first_nan_grad={key}"
            )
            self._nan_detected = True
            break

    self._accumulated_grads = accumulated_grads
    return StepResult(loss=train_loss, dataloader_state_dict=self._loader_state)
```

Key changes vs current:
- `mx.eval(loss, grads)` removed from the loop
- `losses.append(loss)` replaces `train_loss = loss.item()`
- Single `mx.eval(losses, accumulated_grads)` after the loop
- NaN check iterates `losses` after they are already evaluated (zero extra sync)

## Pros

### 1. Fewer GPU sync barriers (primary win)
N syncs → 1 sync per optimizer step. Each `mx.eval` call flushes Metal command buffers
and stalls the CPU until the GPU completes. Removing 7 of 8 stalls reduces the
CPU-GPU ping-pong overhead.

On Apple Silicon (unified memory), each sync is cheaper than discrete GPU but still
costs ~0.1–0.5ms in scheduler overhead. At 8 accum steps and ~12s per step, this is a
small absolute saving — but it opens the door to the larger win below.

### 2. Larger fused Metal kernel (primary win)
MLX's Metal compiler sees all 8 forward/backward graphs at once and can fuse
element-wise ops across mini-steps. In particular:
- The `tree_map(a + b)` accumulations across 8 grad dicts are simple additions;
  the compiler can merge them into a single kernel pass over each parameter's gradient.
- The 8 cross-entropy loss computations (one per mini-batch) share the same embedding
  table lookup and can be batched.

For `grad_accum_steps=8` on d6, the gradient accumulation adds 5 element-wise additions
per parameter per step. With lazy eval these 5 additions are compiled together with the
backward passes; with eager eval each addition is a separate small kernel.

### 3. Better CPU/GPU overlap for data loading
`self._next_batch()` is CPU I/O (parquet read + tokenization). With lazy eval, each
call to `_next_batch()` runs while the GPU computation from the previous step is still
queued but not yet executing (because no `mx.eval` was called). This means:

```
Current (eager):                  Lazy:
GPU: [fwd1][bwd1][SYNC]           GPU: [fwd1][bwd1][fwd2][bwd2]...[fwd8][bwd8][SYNC]
CPU:                 [load2]      CPU: [load2][load3]...[load8]
```

The lazy pattern lets CPU data loading overlap with GPU compute during the single long
GPU burst.

### 4. Cleaner NaN diagnostics preserved
The NaN check (`math.isfinite`) runs AFTER the single `mx.eval`. The losses are already
evaluated at that point, so there is no extra sync. The per-accum-step index is still
reported (via the `for i, loss in enumerate(losses)` loop).

## Cons and risks

### 1. Higher peak memory usage
With 8 lazy computation graphs in flight simultaneously, MLX holds the unevaluated
intermediate arrays for all 8 passes in memory until the single `mx.eval` call.

For d6 (all parameters):
- Each forward/backward creates activations proportional to `B × T × n_embd = 64 × 1024 × 512`
  in bfloat16 → ~67 MB of activations per mini-step
- 8 steps × 67 MB = ~536 MB of extra peak activation memory vs eager eval

The gradient accumulation arrays themselves are not duplicated — `tree_map(a + b)` creates
a single accumulated array per parameter, not 8 separate copies.

On M3 Max 128 GB this is not a concern. For smaller devices (e.g., M2 16 GB) this could
push into swap. Check peak memory before enabling on low-RAM devices.

### 2. Dependency chain through compiled function outputs

`mx.compile(loss_and_grad, inputs=[model], outputs=[model])` marks model parameters as
both inputs (re-read each call) and outputs (potentially updated). With 8 sequential
lazy calls:

- Call 1: reads P₀, outputs grads G₁ and model params P₁ (lazy)
- Call 2: reads P₁ (lazy), computes G₂ and P₂
- ...

`nn.value_and_grad` is a pure function with respect to model params — it computes
gradients but does NOT update the parameters. Each call's "output params" are the same
values as its "input params", just with the gradient graph attached. The lazy dependency
chain is correct: all 8 calls see P₀ values, which is the right behaviour for gradient
accumulation.

**Risk**: If MLX's compiler does not correctly handle multiple lazy calls through
`outputs=[model]` (i.e., if the in-place update mechanism breaks the graph), the
accumulated gradients could be wrong. This must be validated empirically:
- Compare accumulated gradient values between eager and lazy implementations on a
  fixed seed for 2–3 steps.
- A simple sanity check: run the model for N steps with each path and compare final
  loss curves.

### 3. Loss value reported is last mini-step only
Both the current and proposed implementations report `train_loss = losses[-1].item()`.
If the first 7 mini-steps had high loss and only the last had low loss (or vice versa),
the logged value would be misleading. This is not a new behaviour — it exists in the
current code too (`train_loss = loss.item()` is overwritten each iteration, keeping only
the last).

If average loss across mini-steps is desired:
```python
train_loss = sum(l.item() for l in losses) / len(losses)
```
This is zero extra sync cost (losses already evaluated).

### 4. Harder to profile per-mini-step
With eager eval, Metal profiling (Instruments, `mx.metal.start_capture()`) shows
individual forward/backward passes as separate GPU work items. With lazy eval, all 8
are fused into one large Metal command buffer. Per-mini-step timing information is lost.
This is a tradeoff worth accepting for production training; re-enable eager for profiling.

## Expected performance impact

Rough estimate for d6 on M3 Max:

| Component | Eager (current) | Lazy (proposed) | Delta |
|-----------|-----------------|-----------------|-------|
| GPU sync overhead (8 × ~0.3ms) | ~2.4ms/step | ~0.3ms/step | −2.1ms |
| Metal kernel fusion benefit | baseline | +1–5% throughput | +0.5–2.5ms saved |
| CPU/data-load overlap | partial | full | +1–3ms saved |
| **Total estimated gain** | | | **+3–8ms/step** |

At a current step time of ~12,000ms, this is a ~0.03–0.07% improvement — below
measurement noise. The optimization is more meaningful at smaller `grad_accum_steps`
(e.g., 1–2 where the per-step overhead is a larger fraction of total time).

**However**: the Metal kernel fusion benefit is hard to estimate without profiling. If the
8 gradient accumulation additions and 8 backward passes fuse into significantly fewer
Metal dispatches, the gain could be higher.

## Recommendation

Implement the lazy version but validate correctness first:

1. Add a test `tests/test_training/test_mlx_lazy_accum.py` that runs 3 optimizer steps
   with both `eager=True` and `eager=False` paths from the same seed and asserts that
   the accumulated gradients and final parameter values match within float32 tolerance.
2. If validated, land the change and benchmark with `--num-iterations=20` comparing
   tok/sec between the two paths.
3. The change is easily reverted: re-add `mx.eval(loss, grads)` to the loop body.

## Comparison: lazy accum vs. other MLX optimizations

| Optimization | Expected gain | Complexity | Reversible |
|---|---|---|---|
| Lazy grad accumulation (this doc) | <1% | Low | Yes |
| float32 Polar Express (landed) | correctness fix | Trivial | Yes |
| MLX-native dataloader (numpy path) | <1% | Medium | Yes |
| Multi-step compiled function (fuse all accum inside compile) | 5–15% | High | No |
| Async dataloader (background thread) | 2–5% | High | No |

The largest potential gain is **fusing all accumulation steps inside a single compiled
function** — this would let MLX compile the entire N-step accumulation loop as one
Metal program, eliminating all intermediate array allocations. This requires restructuring
`_LossAndGrad` to accept a batch-of-batches input and sum the losses internally. See
the next section for the full design.

---

## Fusing accumulation inside `mx.compile` — full design

### Why it gives more than lazy accumulation

With lazy accumulation, we still call the compiled function **N separate times**. MLX
traces each call independently and produces N separate Metal programs that happen to
be scheduled in sequence. The single `mx.eval` at the end flushes them all, but they
are still N distinct command buffers.

With a fused compiled function, the Python `for` loop over `range(N)` is **unrolled at
MLX trace time** into a single static computation graph. MLX sees the entire accumulation
— 8 forward passes, 8 backward passes, and 7 pairwise gradient additions — as one Metal
program. This unlocks:

1. **Gradient accumulation fusion**: the 7 `a + b` additions per parameter are visible
   to the Metal compiler as a chain `g₀ + g₁ + ... + g₇`. It can lower this to a
   single accumulation pass, avoiding 6 intermediate gradient materialisations per
   parameter tensor.

2. **Activation memory reuse**: step i's activations (stored for backprop) are only
   needed during step i's backward pass. With the full loop visible, the compiler knows
   it can reuse activation buffers across steps rather than keeping all 8 alive
   simultaneously. (This depends on MLX's compiler sophistication; JAX/XLA does this.)

3. **One Metal command buffer**: instead of N dispatches to the GPU, a single large
   command buffer is submitted. Metal's command encoder overhead (~0.1ms per dispatch)
   is paid once.

### How MLX trace-time loop unrolling works

`mx.compile` works like `jax.jit`: it traces through Python code at compile time,
executes Python control flow (if/else, for loops), and records only MLX array operations
into a static graph.

```python
for i in range(self._N):   # Python int N — evaluated at trace time
    loss_i, grads_i = self._lag(xs[i], ys[i])
```

`range(self._N)` is a Python-level loop. At trace time, MLX executes it 8 times,
each time appending 8 MLX ops (forward pass, backward pass) to the static graph.
The result is a graph with 8 × (forward + backward) unrolled in sequence — one program.

This is **different** from the lazy-accumulation approach, where 8 separate compiled
programs are evaluated together. Here there is only **one** compiled program.

### New class: `_MultiStepLossAndGrad`

Replaces `_LossAndGrad` in `mlx_trainer.py`:

```python
class _MultiStepLossAndGrad(nn.Module):
    """Accumulates value and gradient over N mini-batches in a single compiled call.

    The Python for loop is unrolled at mx.compile trace time, creating one static
    Metal program for all N forward + backward passes and their gradient accumulation.
    """

    def __init__(self, model: "GPT", grad_accum_steps: int) -> None:
        super().__init__()
        self._N = grad_accum_steps              # Python int — compile-time constant
        self._lag = nn.value_and_grad(model, model)

    def __call__(self, xs: mx.array, ys: mx.array):
        # xs: (N, B, T), ys: (N, B, T) — all mini-batches stacked
        # range(self._N) is a Python loop → unrolled at trace time into one graph.
        accumulated_grads = None
        total_loss = mx.array(0.0)

        for i in range(self._N):
            loss_i, grads_i = self._lag(xs[i], ys[i])
            total_loss = total_loss + loss_i
            if accumulated_grads is None:
                accumulated_grads = grads_i
            else:
                accumulated_grads = nn.utils.tree_map(
                    lambda a, b: a + b, accumulated_grads, grads_i
                )

        assert accumulated_grads is not None
        return (
            total_loss / self._N,
            nn.utils.tree_map(lambda g: g / self._N, accumulated_grads),
        )
```

The `if accumulated_grads is None` branch is a **Python-level** branch: at trace time
`i=0` is the `None` branch, `i=1..N-1` are the `tree_map` branch. MLX traces both and
records the correct ops. No dynamic branching in the graph.

### Construction in `MLXTrainer.__init__`

```python
# Before:
loss_and_grad = _LossAndGrad(orig_model)
self._loss_and_grad = mx.compile(loss_and_grad, inputs=[orig_model], outputs=[orig_model])

# After:
multi_lag = _MultiStepLossAndGrad(orig_model, grad_accum_steps)
self._loss_and_grad = mx.compile(multi_lag, inputs=[orig_model], outputs=[orig_model])
```

`inputs` and `outputs` serve the same role as before: re-read model params on each call,
and allow `nn.value_and_grad` to correctly track them through the compiled graph.

### Updated `forward_backward`

```python
def forward_backward(self) -> StepResult:
    self._last_x = self._x
    self._last_y = self._y
    self._nan_detected = False

    # Pre-load all N mini-batches on the CPU before issuing the GPU call.
    xs_list: list[mx.array] = [self._x]
    ys_list: list[mx.array] = [self._y]
    for _ in range(self._grad_accum_steps - 1):
        self._x, self._y, self._loader_state = self._next_batch()
        xs_list.append(self._x)
        ys_list.append(self._y)
    # Advance loader one more time so next forward_backward starts on the right batch.
    self._x, self._y, self._loader_state = self._next_batch()

    xs = mx.stack(xs_list)   # (N, B, T) — lazy stack, no copy
    ys = mx.stack(ys_list)   # (N, B, T)

    # Single compiled call: N forward + N backward + N-1 grad accumulations.
    loss, self._accumulated_grads = self._loss_and_grad(xs, ys)

    # Single eval for the entire step.
    mx.eval(loss, self._accumulated_grads)

    train_loss = loss.item()
    if not math.isfinite(train_loss):
        key = _first_nan_key(self._accumulated_grads)
        print(f"[NaN] forward_backward: loss={train_loss}  first_nan_grad={key}")
        self._nan_detected = True

    return StepResult(loss=train_loss, dataloader_state_dict=self._loader_state)
```

Key differences from the lazy-accum version:

| | Lazy accum | Fused compiled |
|---|---|---|
| Compiled calls per step | N (compiled fn called N times) | 1 |
| Metal programs | N separate | 1 fused |
| `mx.eval` per step | 1 | 1 |
| Per-step NaN report | yes (per mini-batch) | no (averaged loss only) |
| CPU/GPU overlap during load | yes (batches load between GPU ops) | no (all loads before GPU call) |

### Why CPU/GPU overlap loss is acceptable on Apple Silicon

On discrete GPU setups, losing CPU/GPU overlap for data loading would be significant
(PCIe data transfer is slow). On Apple Silicon, the tokenized data already lives in
unified memory — loading the next batch is a few pointer operations and numpy array
allocations, typically <1ms for 8 × (64 × 1024) int32 batches. The GPU is busy for
~12,000ms per step. The overlap window was never meaningful.

### Technical risk: `nn.value_and_grad` called N times inside a compiled trace

The critical unknown is whether MLX correctly handles N sequential calls to
`self._lag(xs[i], ys[i])` inside an `mx.compile`-traced function where
`self._lag = nn.value_and_grad(model, model)`.

Each call to `self._lag` produces `(loss_i, grads_i)`. From MLX's perspective, it is
tracing a function that calls `value_and_grad` N times. The `outputs=[model]` on the
outer compile tells MLX to track model param mutations. But `nn.value_and_grad` only
**reads** params — it does not update them. The `outputs=[model]` mechanism exists to
handle the case where the internal gradient plumbing creates new array objects for the
params; with N calls, there would be N rounds of this.

Two scenarios:
- **Safe**: MLX traces through all N calls correctly, treating each as a pure function
  reading (possibly lazy) param arrays and producing grad arrays. The final compiled
  graph has 8 forward passes and 8 backward passes in sequence. This is the expected
  behaviour.
- **Unsafe**: The `outputs=[model]` in-place update mechanism interferes with
  sequential tracing — e.g., call 2's `inputs=[model]` reads call 1's (unevaluated)
  output params, creating a spurious dependency chain that changes gradient values.

### Validation results ✓

**Gradient equivalence test** implemented in
`docs/dev/experiments/test_compiled_nan.py` (`check_fused_grads_match()`).
Uses `TinyGPT` (4 layers, bfloat16, N=4 accum steps) with both paths starting from
identical weights and identical input batches. The eager path uses
`mx.compile(..., inputs=[model], outputs=[model])` called N times — exactly matching
the current `MLXTrainer` behaviour.

Results (`uv run python docs/dev/experiments/test_compiled_nan.py`):

```
[OK]   blocks.0.attn.c_k.weight:    max_diff=0.000000
[OK]   blocks.0.attn.c_proj.weight: max_diff=0.000000
[OK]   blocks.0.attn.c_q.weight:    max_diff=0.000000
[OK]   blocks.0.attn.c_v.weight:    max_diff=0.000000
[OK]   blocks.0.mlp.c_fc.weight:    max_diff=0.000000
[OK]   blocks.0.mlp.c_proj.weight:  max_diff=0.000000
... (all 28 parameter tensors)
[OK]   wte.weight:                  max_diff=0.000004
[OK]   x0_lambdas:                  max_diff=0.000000

Fused grads match eager grads — OK (max_diff=0.000004)
```

All 28 parameter gradients match exactly (0.0 diff) except `wte.weight` at 4e-6.
This is float32 accumulation-order noise from summing 4 bfloat16 embedding gradient
tables in different association orders — not a correctness issue. The same level of
noise appears when computing `(g₀ + (g₁ + (g₂ + g₃)))` vs `((g₀ + g₁) + (g₂ + g₃))`.

**Conclusion**: the technical risk identified in this document is resolved.
MLX correctly handles N sequential calls to `nn.value_and_grad` inside an
`mx.compile` trace with `outputs=[model]`. The fused path produces numerically
equivalent gradients to the eager path.

**Weight trajectory check** (F — 20 optimizer steps, same batches and initial weights):

```
[OK]  blocks.0.attn.c_k.weight:    max_diff=0.001953
[OK]  blocks.0.attn.c_q.weight:    max_diff=0.003174
[OK]  blocks.0.attn.c_v.weight:    max_diff=0.003418
...
[OK]  wte.weight:                  max_diff=0.006836   ← largest
[OK]  x0_lambdas:                  max_diff=0.000122

Weight trajectories match — OK (max_diff=0.006836)
```

The 4e-6 per-step gradient difference (accumulation order) compounds over 20 AdamW
steps to ~7e-3. This is ~1 bfloat16 ULP for weights near 1.0 — numerical noise, not
algorithmic error. All 28 parameters pass the 1e-2 threshold.

**Performance benchmark** (G — TinyGPT, N=4 accum steps, 15 timed steps after 5 warmup):

```
Eager (N separate calls) :    174,576 tok/sec  (avg 46.9 ms/step)
Fused (1 compiled call)  :    187,560 tok/sec  (avg 43.7 ms/step)
```

**+7.4% tok/sec** on TinyGPT (256 embd, 4 layers). The measured gain falls in the
middle of the 5–15% estimate. TinyGPT is compute-light so Metal dispatch overhead is
proportionally larger; the relative gain on d6 (larger model, more compute per step)
may be slightly lower, but the gradient accumulation fusion benefit scales with the
number of parameters and accum steps.

**All validation checks passed. The fused path is ready to land in `mlx_trainer.py`.**

### Performance gain

| Source | Estimate | Measured (TinyGPT, N=4) |
|---|---|---|
| Metal command buffer overhead (N−1 dispatches eliminated) | +0.5–1% | — |
| Gradient accumulation fusion (7 additions per param fused) | +2–5% | — |
| Activation memory reuse across steps (if compiler does it) | +2–8% | — |
| **Total** | **+5–15%** | **+7.4%** (174k → 187k tok/sec) |

Measured on TinyGPT (N_EMBD=256, 4 layers, N=4 accum steps, bfloat16).
The gain on d6 (larger model, 8 accum steps) may differ — more compute per step means
dispatch overhead is a smaller fraction, but more parameters means more gradient
accumulation ops to fuse. Benchmark on the real model after landing to confirm.

### Implementation — landed ✓

Changes in `src/nanochat/training/mlx_trainer.py`:
- Removed `_LossAndGrad` (replaced by `_MultiStepLossAndGrad`)
- `MLXTrainer.__init__`: constructs `_MultiStepLossAndGrad(orig_model, grad_accum_steps)`
  and compiles it with `inputs=[orig_model], outputs=[orig_model]`
- `forward_backward`: pre-loads all N batches, stacks to `(N, B, T)`, single compiled
  call, single `mx.eval`

Changes in `tests/test_training/test_mlx_trainer.py`:
- `test_nan_guard_skips_bad_step`: monkeypatch updated from `(x, y)` to `(xs, ys)`
  to match the new stacked input API
- `test_loader_state_preserved_across_forward_backward`: comment updated (call count
  math unchanged — still N loads per step)
- `test_fused_accum_matches_eager`: new test — runs N=2 separate compiled calls vs
  one fused call on identical weights and data, asserts max_diff < 1e-3 per parameter

All 16 tests pass:

```
uv run pytest tests/test_training/test_mlx_trainer.py -x -q
................
16 passed in 1.79s
```

The change is isolated to `mlx_trainer.py` — no changes to the optimizer, model,
dataloader, or training loop.
