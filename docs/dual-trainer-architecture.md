---
title: "Dual Trainer Architecture"
summary: "Plan to introduce a BaseTrainer protocol with TorchTrainer (current code) and MLXTrainer (MLX + Muon on Apple Silicon)."
read_when:
  - Implementing or reviewing the TorchTrainer / MLXTrainer split
  - Adding a new training backend
  - Understanding how MLX training integrates with nanochat
status: draft
last_updated: "2025-07-22"
---

# Dual Trainer Architecture

Goal: define a `BaseTrainer` protocol that encapsulates model + optimizer behind four methods,
then provide two implementations — `TorchTrainer` (current code) and `MLXTrainer` (MLX + Muon
on Apple Silicon). `train_base.py` calls the protocol and doesn't know which backend is running.

---

## Motivation

The current `train_base` function is a single monolithic PyTorch path. On Apple Silicon the
experience is degraded: `torch.compile` causes NaN gradients on MPS, FP8 is CUDA-only, and
DDP is irrelevant. MLX is purpose-built for Apple Silicon and avoids all of these issues.

A shared protocol lets both backends reuse the same training loop, CLI, config, checkpoint
format, evaluation harness, and logging — only the forward/backward/optimizer step differs.

---

## BaseTrainer Protocol

```python
class BaseTrainer(Protocol):
    def forward_backward(self, x, y) -> float: ...
    def step(self, lr_multiplier, momentum, weight_decay): ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, d: dict): ...
```

Each implementation owns its model and optimizer internally. The training loop becomes:

```python
for step in range(num_iterations):
    loss = trainer.forward_backward(x, y)
    trainer.step(lrm, momentum, weight_decay)
```

The loop state lives in `TrainingState`, the model+optimizer state lives behind the trainer protocol.

---

## TorchTrainer — current code

Wraps the existing PyTorch model + `MuonAdamW` / `DistMuonAdamW`:

| Aspect | Detail |
|---|---|
| Model | `GPT` (PyTorch) |
| Compile | `torch.compile` (skipped on MPS) |
| Precision | bf16 / fp16 + GradScaler / FP8 (CUDA only) |
| Optimizer | `MuonAdamW` / `DistMuonAdamW` |
| Attention | FA3 when available, SDPA fallback |
| Distribution | DDP via `torchrun` |

`forward_backward` runs the grad accumulation loop internally (autocast, loss scaling, etc.).
`step` applies LR/momentum/weight-decay to all param groups and calls `optimizer.step()`.

---

## MLXTrainer — MLX backend

New implementation targeting Apple Silicon natively:

| Aspect | Detail |
|---|---|
| Model | `GPT` ported to `mlx.nn` (same architecture, MLX ops) |
| Compile | `mx.compile` (stable on Apple Silicon) |
| Precision | float16 / bfloat16 (native unified memory) |
| Optimizer | Muon ported to MLX (Polar Express + NorMuon) + AdamW from `mlx.optimizers` |
| Attention | `mx.fast.scaled_dot_product_attention` with sliding window |
| Distribution | single-device only |

`forward_backward` uses `mx.nn.value_and_grad`, returns the loss float.
`step` applies the Muon/AdamW update with the given hyperparameters.

---

## Checkpoint interop

Two options for cross-backend checkpoint exchange:

### Option A — numpy arrays

`state_dict()` returns numpy arrays. Both PyTorch and MLX can read/write numpy natively:

- **PyTorch → MLX**: `torch.Tensor.numpy()` → `mx.array()`
- **MLX → PyTorch**: `np.array(mx_tensor)` → `torch.from_numpy()`

### Option B — safetensors

Both frameworks support safetensors natively:

- **PyTorch**: `safetensors.torch.save_file` / `safetensors.torch.load_file`
- **MLX**: `mx.save_safetensors` / `mx.load`

Safetensors is memory-mapped, zero-copy, and already the standard format for HuggingFace
models. It also enables compatibility with `mlx-lm` tooling for inference and quantization.

Either option avoids custom conversion utilities. The choice depends on whether we want
minimal dependencies (numpy) or ecosystem compatibility (safetensors).

---

## What changes in train_base.py

`train_base.py` becomes backend-agnostic. It:

1. Builds a `TrainingState` (fresh or from checkpoint)
2. Constructs the appropriate trainer (`TorchTrainer` or `MLXTrainer`) based on `--backend`
3. Runs the training loop calling only protocol methods
4. Handles eval, logging, checkpointing, and scheduling — all of which are backend-independent

Everything that currently lives inside the `train_base_model()` closure that touches the model
or optimizer directly gets pushed behind the protocol.

---

## File layout

```
src/nanochat/
├── training/
│   ├── base_trainer.py      # BaseTrainer protocol
│   ├── torch_trainer.py     # TorchTrainer implementation
│   ├── mlx_trainer.py       # MLXTrainer implementation
│   ├── train_state.py       # TrainingState dataclass
│   ├── train_base.py        # backend-agnostic training loop
│   └── ...
├── models/
│   ├── gpt.py               # existing PyTorch GPT
│   ├── mlx_gpt.py           # MLX GPT (same architecture)
│   └── ...
```

---

## Implementation order

1. ✅ **Entry-point refactor** — training/evaluation split into sub-packages with co-located state classes (`PretrainingState`, `SFTState`, `RLState`)
2. ✅ **Checkpoint manager** — `CheckpointManager` protocol, typed metadata, `model_factory.py`
3. ✅ **MLX GPT** — port `models/gpt.py` to `mlx.nn`, validate forward pass matches PyTorch
4. ✅ **MLX Muon** — port optimizer, validate update step matches PyTorch
5. **BaseTrainer protocol + TorchTrainer + MLXTrainer** — abstraction grounded in what MLX actually needs
6. **Backend-agnostic `train_base.py`** — loop calls only protocol methods
7. **Checkpoint interop** — numpy or safetensors cross-backend handoff
8. ✅ **CLI integration** — `--backend torch|mlx` flag, auto-detection, `CommonConfig.validate()`

---

## BaseTrainer protocol + TorchTrainer implementation proposal

### Risks and drawbacks

The following issues were identified by reviewing the proposal against the actual `loop.py` and `setup.py` code.

| # | Risk | Severity | Blocker? |
|---|---|---|---|
| 1 | Dataloader state dict not returned from `forward_backward` | Medium | Yes — checkpoint resume position silently lost |
| 2 | `eval_model() -> GPT` return type unsatisfiable for MLXTrainer | Design | No for phase 1.5, yes for MLXTrainer |
| 3 | `model.train()` / `model.eval()` toggling unspecified | Medium | Yes — silent training mode bug |
| 4 | `initial_lr` on param groups is an undocumented invariant | Low | No, but needs assertion |
| 5 | Scaler + DDP inf all-reduce ordering broken if DDP guard missing | Medium | Yes — hangs on single-process run |
| 6 | `CompressionMetrics` numpy refactor: `log_softmax` + `gather` not trivially numpy | Medium | Yes — needs parity test before shipping |
| 7 | `forward_logits` runs a full extra forward pass every `compression_log_every` step | Low | No — known cost, same as today |

**Risk 1 — dataloader state dict**: `loop.py` reads `state.dataloader_state_dict` from the loader result inside the accumulation loop (`x, y, state.dataloader_state_dict = next(s.train_loader)`). This state dict is written to the checkpoint and used for resume. If the loader lives inside `TorchTrainer` and `forward_backward` only returns `float`, the resume position is silently dropped on every step.

**Risk 2 — `eval_model()` return type**: `MLXTrainer` holds an `mlx.nn.Module`, not a `GPT`. The protocol method `eval_model() -> GPT` cannot be satisfied. For phase 1.5 this is fine (Torch-only eval), but it means `loop.py` cannot call `eval_model()` generically without a type guard when MLXTrainer is added.

**Risk 3 — train/eval mode**: `loop.py` calls `s.model.eval()` before validation/CORE/sampling and `s.model.train()` after. With the trainer abstraction, `loop.py` no longer holds `s.model`. The protocol must expose `set_eval_mode()` / `set_train_mode()`, or the caller must toggle the model returned by `eval_model()` — which must be stated explicitly or it will be missed.

**Risk 4 — `initial_lr` invariant**: `TorchTrainer.step` sets `group["lr"] = group["initial_lr"] * lrm`. This assumes every param group has `"initial_lr"` set at construction time by `MuonAdamW.setup_optimizer`. If a different optimizer is passed, it silently fails at runtime with a `KeyError`.

**Risk 5 — scaler + DDP inf all-reduce ordering**: The scaler path in `loop.py` is `unscale_` → DDP all-reduce of inf flags → `scaler.step()`. The DDP all-reduce must be guarded by `is_ddp_initialized()` — calling `dist.all_reduce` on a single-process run hangs. Moving this block into `TorchTrainer.step` without the guard is a correctness regression.

**Risk 6 — `CompressionMetrics` numpy refactor**: `compute_conditional_entropy` uses `torch.log_softmax` and `.gather()`. The numpy equivalents (`scipy.special.log_softmax` or manual) must produce bit-identical results. A parity test asserting `abs(torch_result - numpy_result) < 1e-6` is required before the refactor ships.

**Risk 7 — double forward pass**: On every `compression_log_every` step, `forward_logits` runs a full extra forward pass. This is the same cost as today (the current code already does this), so it is not a regression — but it is a known overhead.

---

### Production-grade design

#### `StepResult` — forward_backward return type

Resolves risk 1. `forward_backward` returns a dataclass instead of a bare `float`:

```python
@dataclass(frozen=True)
class StepResult:
    loss: float
    dataloader_state_dict: dict[str, object]
```

`loop.py` unpacks it:

```python
result = s.trainer.forward_backward()
state.dataloader_state_dict = result.dataloader_state_dict
train_loss_f = result.loss
```

Note: `x` and `y` are no longer passed by `loop.py` — the trainer owns the loader and fetches batches internally. The initial `x, y` pre-fetch before the loop is also removed; `TorchTrainer.__init__` primes the loader on construction.

#### `eval_context` — train/eval mode and eval_model type

Resolves risks 2 and 3. Instead of `eval_model() -> GPT`, the protocol exposes a context manager:

```python
class BaseTrainer(Protocol):
    @contextmanager
    def eval_context(self) -> Iterator[Any]: ...
```

`loop.py` uses it as:

```python
with s.trainer.eval_context() as model:
    state.val_bpb = evaluate_bpb(model, val_loader, eval_steps, s.token_bytes)
```

`TorchTrainer.eval_context` calls `self._orig_model.eval()`, yields `self._orig_model`, then calls `self._model.train()` in the finally block. `MLXTrainer.eval_context` yields its `mlx.nn.Module` — the return type is `Any`, which is correct since `evaluate_bpb` will need a backend-specific path anyway.

For FP8 disable, `TorchTrainer.eval_context` wraps the yield in `disable_fp8(self._orig_model)` internally — the caller never sees it.

#### Full protocol

```python
@dataclass(frozen=True)
class StepResult:
    loss: float
    dataloader_state_dict: dict[str, object]


class BaseTrainer(Protocol):
    def forward_backward(self) -> StepResult: ...
    def step(self, lr_multiplier: float, momentum: float, weight_decay: float) -> None: ...
    def forward_logits(self) -> tuple[np.ndarray, np.ndarray]: ...  # (logits, tokens), numpy
    def model_state_dict(self) -> dict[str, Any]: ...
    def optimizer_state_dict(self) -> dict[str, Any]: ...
    def load_state_dicts(self, model_state: dict[str, Any], optimizer_state: dict[str, Any]) -> None: ...
    @contextmanager
    def eval_context(self) -> Iterator[Any]: ...
```

`forward_logits` takes no arguments — the trainer uses the same current batch stored as `self._last_x`, `self._last_y` after each `forward_backward` call. No extra dataloader advance, no argument threading.

#### `TorchTrainer` internals

`__init__` takes: `orig_model`, `model` (compiled), `optimizer`, `scaler`, `grad_accum_steps`, `device_type`, `train_loader`.

On construction:
- Assert every param group has `"initial_lr"` key (resolves risk 4)
- Prime the loader: `self._x, self._y, self._loader_state = next(train_loader)`

`forward_backward`:
- Runs the accumulation loop over `self._grad_accum_steps` microsteps
- Each microstep: autocast → forward → `loss / grad_accum_steps` → scaler scale → backward → advance loader
- Stores final `self._x, self._y, self._loader_state` for next call
- Returns `StepResult(loss=train_loss.item(), dataloader_state_dict=self._loader_state)`

`step` (resolves risk 5):
```python
for group in self._optimizer.param_groups:
    group["lr"] = group["initial_lr"] * lr_multiplier
    if group["kind"] == "muon":
        group["momentum"] = momentum
        group["weight_decay"] = weight_decay
if self._scaler is not None:
    self._scaler.unscale_(self._optimizer)
    if is_ddp_initialized():  # guard — resolves risk 5
        for v in self._scaler._found_inf_per_device(self._optimizer).values():
            dist.all_reduce(v, op=dist.ReduceOp.MAX)
    self._scaler.step(self._optimizer)
    self._scaler.update()
else:
    self._optimizer.step()
self._model.zero_grad(set_to_none=True)
```

`forward_logits`:
- Runs a single forward pass on `self._last_x` using `self._orig_model` inside `disable_fp8` + autocast
- Returns `logits.float().cpu().numpy(), self._last_y.cpu().numpy()`
- `self._last_x` / `self._last_y` are the batch stored after the most recent `forward_backward` call

`eval_context`:
```python
@contextmanager
def eval_context(self) -> Iterator[GPT]:
    self._orig_model.eval()
    try:
        with disable_fp8(self._orig_model):
            yield self._orig_model
    finally:
        self._model.train()
```

#### `CompressionMetrics` numpy refactor (resolves risk 6)

Change all method signatures to accept `np.ndarray`. Replace the two PyTorch ops:

- `torch.log_softmax(logits, dim=-1)` → `logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)`
- `.gather(dim=-1, index=tokens[..., None]).squeeze(-1)` → `logits[np.arange(B)[:, None], np.arange(T)[None, :], tokens]`

Add a parity test in `tests/test_training/test_compression_metrics.py`:
```python
def test_conditional_entropy_numpy_matches_torch():
    # generate random logits and tokens, compute via both paths, assert abs diff < 1e-5
```

This test must pass before the refactor is merged.

#### `BaseTrainingSetup` changes

Remove from `__slots__` / `__init__`: `orig_model`, `model`, `optimizer`, `scaler`, `grad_accum_steps`, `train_loader`.
Add: `trainer: BaseTrainer`.

`setup()` constructs `TorchTrainer(...)` and stores it. The initial `x, y` pre-fetch that currently precedes the loop in `loop.py` is removed — `TorchTrainer.__init__` primes the loader.

#### `loop.py` training step after refactor

```python
# compression logits (only when scheduled)
if compression_tracker and state.step % s.config.training.compression_log_every == 0:
    logits_np, tokens_np = s.trainer.forward_logits()

# training step
s.synchronize()
t0 = time.time()
result = s.trainer.forward_backward()
state.dataloader_state_dict = result.dataloader_state_dict

lrm = s.get_lr_multiplier(state.step)
muon_momentum = s.get_muon_momentum(state.step)
muon_weight_decay = s.get_weight_decay(state.step)
s.trainer.step(lrm, muon_momentum, muon_weight_decay)
train_loss_f = result.loss
```

Checkpoint save:
```python
checkpoint_manager.save(
    state,
    s.trainer.model_state_dict(),
    s.trainer.optimizer_state_dict(),
    rank=s.ddp_rank,
)
```

Eval / sampling:
```python
with s.trainer.eval_context() as model:
    state.val_bpb = evaluate_bpb(model, val_loader, eval_steps, s.token_bytes)
```

#### What stays in `loop.py`

Eval scheduling, bpb evaluation, CORE metric, sampling, checkpointing, compression metrics logging, wandb, GC management — all backend-independent, none moves.

#### What stays in `setup.py`

Model construction, FP8 conversion, `torch.compile`, optimizer construction, dataloader construction, scheduler construction — all passed into `TorchTrainer.__init__`.

#### Validation checklist before merging step 5

- [ ] `test_compression_metrics.py` — numpy parity test passes
- [ ] `TorchTrainer.__init__` asserts `"initial_lr"` present on all param groups
- [ ] Full suite passes with `TorchTrainer` wired into `BaseTrainingSetup`
- [ ] Resume from checkpoint restores correct dataloader position (epoch/pq/rg match)
- [ ] Single-process run (no DDP) completes without hang in `step`
- [ ] `eval_context` leaves model in train mode after exit (including on exception)
## MLX GPT — design and requirements

### Goal

Port `models/gpt.py` to MLX (`mlx.nn`) with identical architecture and numerically matching
forward pass output. The MLX model is the foundation for `MLXTrainer` — if the forward pass
doesn't match PyTorch, nothing downstream is trustworthy.

### File

`src/nanochat/models/mlx_gpt.py` — standalone, no dependency on `models/gpt.py`.
Shares only `GPTConfig` from `models/config.py`.

### Architecture mapping

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

### Attention

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

### Features to port

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
| KV cache inference | `kv_cache` path in `forward` | Defer — training only for now |

### What to defer

- KV cache inference (`generate`) — training forward pass only in the first iteration
- FP8 — CUDA-only, irrelevant for MLX
- DDP / `DistMuonAdamW` — single-device only

### Validation plan

After implementing `mlx_gpt.py`, validate numerics against PyTorch before any training:

1. Instantiate both models with identical `GPTConfig` and identical weights (copy via numpy)
2. Run a forward pass with the same input tokens
3. Assert `max(abs(mlx_logits - torch_logits)) < 1e-3` (tolerance for bf16 vs float32 differences)
4. Run a backward pass on both, assert gradient magnitudes are in the same range

This validation lives in `tests/test_models/test_mlx_gpt.py` and is skipped when MLX is not installed.

### Open questions resolved

- ✅ `mx.fast.scaled_dot_product_attention` supports GQA natively
- ✅ Sliding window implemented via boolean causal mask — no native parameter needed
- ✅ `nn.value_and_grad(model, fn)` is the correct API for training
- ✅ Grad accumulation is manual — each call returns a fresh independent grad tree, no state on parameters
- ✅ `mx.eval()` must be called after each microbatch — see MLX Training Patterns section

---

## MLX Training Patterns

These patterns apply to `MLXTrainer.forward_backward` and any MLX training code.

### Grad accumulation

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

### `mx.eval()` cadence

MLX is lazy — operations build a compute graph but nothing executes until `mx.eval()` is called.
Without an explicit eval between microbatches, the graph grows unboundedly in memory before
anything runs. The correct cadence:

| When | What to eval |
|---|---|
| After each microbatch forward/backward | `mx.eval(loss, grads)` |
| After optimizer update | `mx.eval(model.parameters())` |
| Before logging a loss value | `loss.item()` triggers eval implicitly |

Do **not** defer all evals to the end of the accumulation loop.

### Tree utilities

`nn.utils.tree_map` is the correct function for operating over grad trees.
`mx.tree_map` does not exist — using it will raise `AttributeError`.

---

## MLX Muon — design and requirements

### Goal

Port `training/optimizer.py` (AdamW + Muon) to MLX. The MLX optimizer is the last piece
before `MLXTrainer` can be assembled. Validation: run one update step on both backends with
identical weights and gradients, assert parameter delta matches within 1e-3.

### File

`src/nanochat/training/mlx_optimizer.py` — standalone, no dependency on `optimizer.py`.

### Architecture decision: don't subclass `mlx.optimizers.Optimizer`

`mlx.optimizers.Optimizer` is tree-based: `apply_gradients` calls `apply_single` per leaf
via `tree_map`. Muon cannot fit this model — it must operate on stacked groups of same-shape
parameters simultaneously (Polar Express requires the full matrix, not individual scalars).

Instead, `MuonAdamW` is a plain class with:
- One `mlx.optimizers.AdamW` instance per AdamW param group (different LR/betas per group)
- Manual Muon state (`momentum_buffer`, `second_momentum_buffer`) stored as plain dicts
- A single `update(model, grads)` method matching the MLX optimizer convention

### AdamW groups

`mlx.optimizers.AdamW` is used directly per group. Key difference from PyTorch:

| Aspect | PyTorch `adamw_step_fused` | MLX `optim.AdamW` |
|---|---|---|
| Bias correction | Always applied | `bias_correction=False` by default |
| Weight decay | Decoupled, applied before update | Decoupled, applied in `apply_single` |
| Per-group LR | Set on group dict | Passed to constructor |

MLX `AdamW` must be constructed with `bias_correction=True` to match PyTorch behavior.
Each AdamW group gets its own `optim.AdamW` instance — MLX optimizers hold state internally
and cannot be shared across groups with different hyperparameters.

### Muon step — op mapping

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
| `@torch.compile` | none needed | MLX JIT-compiles lazily |

### Muon step — immutability

MLX arrays are immutable — every op returns a new array. The Muon step returns updated
`(stacked_params, momentum_buffer, second_momentum_buffer)` rather than mutating in place.
State dicts are updated by reassignment after each step.

### Param group interface

Same `kind="muon"` / `kind="adamw"` dict structure as PyTorch `MuonAdamW`, so `GPT.setup_optimizer`
can be ported with minimal changes. Muon groups must still have all params with the same shape
(caller's responsibility, same as PyTorch).

### Validation plan

1. Instantiate both PyTorch and MLX models with identical weights (copy via numpy)
2. Compute gradients on both with the same input
3. Run one optimizer step on both
4. Assert `max(abs(mlx_params - torch_params)) < 1e-3` for all parameter tensors
5. Run a second step, assert params still track

Tests in `tests/test_training/test_mlx_optimizer.py`, skipped when MLX not installed.

### What to defer

- `DistMuonAdamW` — single-device only, no DDP in MLX
- `@mx.compile` on the Muon step — add after correctness is validated

---

## CLI integration — `--backend` in `CommonConfig`

Added ahead of step 5 so the config plumbing exists before the trainer dispatch is wired.

### What was added

`CommonConfig` gains one field:

```python
backend: str = ""  # torch | mlx (empty = autodetect)
```

`autodetect_backend()` in `common/distributed.py`:

```python
def autodetect_backend() -> str:
    try:
        import mlx.core  # noqa: F401
        return "mlx"
    except ImportError:
        return "torch"
```

On Apple Silicon with MLX installed → `"mlx"`. Everywhere else → `"torch"`. No platform sniffing — import presence is the only signal that matters.

CLI arg in `CommonConfig.update_parser`:

```
--backend torch|mlx   training backend (empty = autodetect)
```

### Why `backend` and `device_type` are separate fields

`device_type` controls which PyTorch device is used (`cuda`, `mps`, `cpu`). `backend` controls which runtime is used at all. On Apple Silicon you can run `backend=torch, device_type=mps` or `backend=mlx` (device is implicit — MLX always uses the Apple GPU). Collapsing them would make `device_type=mlx` a lie and break `compute_init`.

### What `setup.py` will do with it (step 5)

```python
backend = config.common.backend or autodetect_backend()
if backend == "mlx":
    # skip compute_init, DDP, device setup entirely
    trainer = build_mlx_trainer(config)
else:
    device_type = config.common.device_type or autodetect_device_type()
    _, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    trainer = build_torch_trainer(config, device, ddp_rank, ddp_world_size)
```

`compute_init` is never called for the MLX path — it asserts `device_type in ["cuda", "mps", "cpu"]` and would fail or be misleading.

### Resolved open questions

- ✅ `CompressionMetrics` tracker supported by both backends via `forward_logits() -> tuple[np.ndarray, np.ndarray]` — not deferred
