---
title: "Trainer Protocol"
summary: "Reference for the BaseTrainer protocol, StepResult, TorchTrainer internals, and how to implement a new backend."
read_when:
  - Reviewing or implementing a trainer backend
  - Understanding how loop.py interacts with the trainer
  - Debugging forward/backward or optimizer step behavior
status: active
last_updated: "2025-07-24"
---

# Trainer Protocol

`BaseTrainer` is the protocol that decouples `loop.py` from any specific ML framework.
`TorchTrainer` and `MLXTrainer` both satisfy it. `loop.py` calls only protocol methods
and never imports torch or mlx directly.

---

## StepResult

```python
@dataclass(frozen=True)
class StepResult:
    loss: float
    dataloader_state_dict: dict[str, object]
```

Returned by `forward_backward`. The loop stores `dataloader_state_dict` in `PretrainingState`
so it can be checkpointed and used to resume the dataloader.

---

## Protocol methods

```python
class BaseTrainer(Protocol):
    def forward_backward(self) -> StepResult: ...
    def step(self, lr_multiplier: float, momentum: float, weight_decay: float) -> None: ...
    def forward_logits(self) -> tuple[np.ndarray, np.ndarray]: ...
    def model_state_dict(self) -> dict[str, Any]: ...
    def optimizer_state_dict(self) -> dict[str, Any]: ...
    def load_state_dicts(self, model_state, optimizer_state) -> None: ...
    @contextmanager
    def eval_context(self) -> Iterator[Any]: ...
```

| Method | Contract |
|---|---|
| `forward_backward()` | Runs the full accumulation loop, returns mean loss and final loader state |
| `step(lrm, mom, wd)` | Applies `initial_lr * lrm` to all groups, updates Muon momentum/weight_decay, calls optimizer |
| `forward_logits()` | Returns `(logits, targets)` as numpy arrays for the batch snapshotted at the start of the last `forward_backward` call |
| `model_state_dict()` | Returns model weights in the trainer's native array type |
| `optimizer_state_dict()` | Returns serializable optimizer state |
| `load_state_dicts(m, o)` | Loads model and optimizer state — accepts numpy arrays (converts internally) |
| `eval_context()` | Context manager: sets model to eval mode, yields model, restores train mode in `finally` |

---

## loop.py call sequence

```mermaid
sequenceDiagram
    participant Loop as loop.py
    participant Trainer as BaseTrainer
    participant Eval as evaluate_bpb / evaluate_core

    Loop->>Trainer: eval_context()
    Trainer-->>Loop: model (eval mode)
    Loop->>Eval: evaluate_bpb(model, ...)
    Eval-->>Loop: val_bpb
    Loop->>Trainer: [exit eval_context — restores train mode]

    loop training step
        Loop->>Trainer: forward_logits() [optional, compression]
        Loop->>Trainer: forward_backward()
        Trainer-->>Loop: StepResult(loss, dataloader_state_dict)
        Loop->>Trainer: step(lr_multiplier, momentum, weight_decay)
    end

    Loop->>Trainer: model_state_dict()
    Loop->>Trainer: optimizer_state_dict()
    Loop->>Manager: save(state, model_state, optimizer_state)
```

---

## TorchTrainer internals

- `__init__` primes the loader: calls `next(train_loader)` once so the first batch is ready
- `__init__` asserts `initial_lr` on all param groups — set by `model.setup_optimizer()`
- `forward_backward` snapshots `_last_x`/`_last_y` at the start of the step (before the accumulation loop advances the loader) — `forward_logits` uses these
- `step` uses `group["initial_lr"] * lr_multiplier` — never compounds
- `eval_context` uses `disable_fp8` to swap Float8Linear → Linear for BF16 eval, restores in `finally`
- Scaler path: `GradScaler` active on fp16 (MPS and some CUDA configs); DDP all-reduce of `found_inf` before `scaler.step`

## MLXTrainer internals

- `__init__` snapshots `initial_lr` on all groups and primes the loader
- `forward_backward` uses `nn.value_and_grad` + manual `nn.utils.tree_map` accumulation; `mx.eval(loss, grads)` after each microbatch — see [mlx-training-patterns.md](mlx-training-patterns.md)
- `step` calls `optimizer.update(model, accumulated_grads)` then `mx.eval(model.parameters(), optimizer.state())`
- `forward_logits` uses `mx.stop_gradient` — no grad tape pollution
- `load_state_dicts` accepts numpy arrays and converts via `from_numpy_mlx`

---

## Implementing a new backend

1. Implement all 7 protocol methods
2. Snapshot `initial_lr` per group in `__init__` — `step` must use `initial_lr * lr_multiplier`
3. Prime the loader in `__init__` — first batch ready before the loop starts
4. `forward_logits` must use the batch from the start of the last `forward_backward` call
5. `load_state_dicts` must accept `dict[str, np.ndarray]` (safetensors manager output)
6. `eval_context` must restore train mode in `finally`
7. Add a branch in `training/base/setup.py` `_setup_<backend>` and dispatch in `setup()`
