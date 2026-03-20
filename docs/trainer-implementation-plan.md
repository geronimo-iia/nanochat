---
title: "BaseTrainer + TorchTrainer Implementation Plan"
summary: "Risks, production-grade design, implementation steps, and validation checklist for step 5 of the dual-trainer architecture."
read_when:
  - Implementing BaseTrainer protocol or TorchTrainer
  - Reviewing the CompressionMetrics numpy refactor
  - Checking what needs to be done before MLXTrainer can be wired
status: draft
last_updated: "2025-07-23"
---

# BaseTrainer + TorchTrainer Implementation Plan

Step 5 of the [dual-trainer architecture](dual-trainer-architecture.md). Defines the protocol,
resolves all identified risks, and provides an ordered implementation plan.

---

## Risks and drawbacks

The following issues were identified by reviewing the initial proposal against the actual `loop.py` and `setup.py` code.

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

## Production-grade design

### `StepResult` — forward_backward return type

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

### `eval_context` — train/eval mode and eval_model type

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

### Full protocol

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

### `TorchTrainer` internals

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

### `CompressionMetrics` split (resolves risk 6)

`CompressionMetrics` currently mixes two concerns that diverge once the numpy refactor lands:

- **Computation** — the math (entropy, conditional entropy, compression ratio, gzip). Pure functions, no state, no framework dependency.
- **Tracking** — `self.history`, `detect_overfitting()`, `get_summary()`, `log_metrics()`. Stateful wrapper driven by `loop.py`.

Split into two files:

```
src/nanochat/training/
├── compression_math.py     # pure numpy functions, no state, no torch
└── compression_metrics.py  # CompressionMetrics — stateful tracker, calls compression_math
```

`compression_math.py` exports plain functions taking `np.ndarray`, returning `float`:

```python
def compute_entropy(tokens: np.ndarray) -> float: ...
def compute_conditional_entropy(logits: np.ndarray, tokens: np.ndarray) -> float: ...
def compute_compression_ratio(logits: np.ndarray, tokens: np.ndarray) -> float: ...
def compute_gzip_compression(tokens: np.ndarray) -> float: ...
```

Replace the two PyTorch ops in `compute_conditional_entropy`:
- `torch.log_softmax(logits, dim=-1)` → `logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)`
- `.gather(dim=-1, index=tokens[..., None]).squeeze(-1)` → `logits[np.arange(B)[:, None], np.arange(T)[None, :], tokens]`

`CompressionMetrics` becomes a thin stateful wrapper that calls `compression_math` functions and appends to `self.history`. The torch import disappears from it entirely.

#### Why not split by backend (torch vs mlx)

The only backend-specific ops are `log_softmax` and `gather` — two lines in one function. Splitting by backend would mean two implementations of the same formula that must stay in sync, and any fix applied twice. The numpy path works for both backends because the conversion to numpy happens at the boundary (`forward_logits`), not inside the math:

- `TorchTrainer.forward_logits` → `logits.float().cpu().numpy()` (device transfer, negligible at `compression_log_every=100`)
- `MLXTrainer.forward_logits` → `np.array(logits)` (zero-copy on Apple Silicon unified memory)

One numpy implementation, both backends feed it identically.

#### Parity test gate

Add `tests/test_training/test_compression_math.py`:
```python
def test_conditional_entropy_matches_torch():
    # random logits + tokens, compute via compression_math and via torch.log_softmax+gather
    # assert abs diff < 1e-5
```

This test must pass before the refactor is merged.

### `BaseTrainingSetup` changes

Remove from `__slots__` / `__init__`: `orig_model`, `model`, `optimizer`, `scaler`, `grad_accum_steps`, `train_loader`.
Add: `trainer: BaseTrainer`.

`setup()` constructs `TorchTrainer(...)` and stores it. The initial `x, y` pre-fetch that currently precedes the loop in `loop.py` is removed — `TorchTrainer.__init__` primes the loader.

### `loop.py` training step after refactor

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

### What stays in `loop.py`

Eval scheduling, bpb evaluation, CORE metric, sampling, checkpointing, compression metrics logging, wandb, GC management — all backend-independent, none moves.

### What stays in `setup.py`

Model construction, FP8 conversion, `torch.compile`, optimizer construction, dataloader construction, scheduler construction — all passed into `TorchTrainer.__init__`.

---

## Implementation steps

1. ✅ **`compression_math.py`** — pure numpy functions + parity test. Committed `df92f2b`.

2. ✅ **`CompressionMetrics` refactor** — delegates all math to `compression_math`, torch import removed, accepts `np.ndarray`. Committed `c908f12`.

3. ✅ **`training/base/trainer.py`** — `StepResult` dataclass and `BaseTrainer` protocol. Committed `f9de1a8`.

4. ✅ **`TorchTrainer`** — full implementation in `training/base/trainer.py` with 11 unit tests. Committed `060c43f`.

5. ✅ **`BaseTrainingSetup` swap** — holds `trainer: BaseTrainer`, `grad_accum_steps` removed (owned by `TorchTrainer`). Committed `8449683`.

6. ✅ **`loop.py` refactor** — training step, eval, sampling, checkpoint all use protocol methods. Full suite 301 passed. Committed `c46f904`.

7. **End-to-end parity test** — run 10 steps with the refactored loop, assert loss trajectory matches a reference run captured before the refactor (or run both old and new loop in the same test with identical seeds). Gate: loss values match within 1e-6.

---

## Validation checklist before merging step 5

- [x] `test_compression_math.py` — numpy parity test passes (< 1e-5 vs torch reference)
- [x] `compression_metrics.py` delegates to `compression_math`, no torch import
- [x] `TorchTrainer.__init__` asserts `"initial_lr"` present on all param groups
- [x] Full suite passes with `TorchTrainer` wired into `BaseTrainingSetup`
- [x] Resume from checkpoint restores correct dataloader position — no-repeat verified (`dev/verify_dataloader_resume.py`). Note: resume is approximate (buffer-based packing), exact batch reproduction is not guaranteed by design.
- [x] Single-process run (no DDP) completes without hang in `step` — verified (`dev/verify_single_process_step.py`)
- [x] `eval_context` leaves model in train mode after exit (including on exception)
- [ ] End-to-end parity test: 10-step loss trajectory matches pre-refactor reference
