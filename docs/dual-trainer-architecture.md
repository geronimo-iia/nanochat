---
title: "Dual Trainer Architecture"
summary: "Index for the BaseTrainer protocol with TorchTrainer and MLXTrainer backends."
read_when:
  - Getting an overview of the dual-trainer architecture
  - Finding the right sub-document for a specific component
  - Reviewing implementation order and status
status: active
last_updated: "2025-07-23"
---

# Dual Trainer Architecture

Goal: define a `BaseTrainer` protocol that encapsulates model + optimizer behind a small set
of methods, then provide two implementations — `TorchTrainer` (current code) and `MLXTrainer`
(MLX + Muon on Apple Silicon). `loop.py` calls the protocol and doesn't know which backend
is running.

---

## Motivation

The current `train_base` function is a single monolithic PyTorch path. On Apple Silicon the
experience is degraded: `torch.compile` causes NaN gradients on MPS, FP8 is CUDA-only, and
DDP is irrelevant. MLX is purpose-built for Apple Silicon and avoids all of these issues.

A shared protocol lets both backends reuse the same training loop, CLI, config, checkpoint
format, evaluation harness, and logging — only the forward/backward/optimizer step differs.

---

## Protocol summary

```python
@dataclass(frozen=True)
class StepResult:
    loss: float
    dataloader_state_dict: dict[str, object]

class BaseTrainer(Protocol):
    def forward_backward(self) -> StepResult: ...
    def step(self, lr_multiplier: float, momentum: float, weight_decay: float) -> None: ...
    def forward_logits(self) -> tuple[np.ndarray, np.ndarray]: ...
    def model_state_dict(self) -> dict[str, Any]: ...
    def optimizer_state_dict(self) -> dict[str, Any]: ...
    def load_state_dicts(self, model_state: dict[str, Any], optimizer_state: dict[str, Any]) -> None: ...
    @contextmanager
    def eval_context(self) -> Iterator[Any]: ...
```

The loop state lives in `PretrainingState`, the model+optimizer state lives behind the protocol.
See [trainer-implementation-plan.md](trainer-implementation-plan.md) for the full design rationale.

---

## Backends

### TorchTrainer

| Aspect | Detail |
|---|---|
| Model | `GPT` (PyTorch) |
| Compile | `torch.compile` (skipped on MPS) |
| Precision | bf16 / fp16 + GradScaler / FP8 (CUDA only) |
| Optimizer | `MuonAdamW` / `DistMuonAdamW` |
| Attention | FA3 when available, SDPA fallback |
| Distribution | DDP via `torchrun` |

### MLXTrainer

| Aspect | Detail |
|---|---|
| Model | `GPT` ported to `mlx.nn` — see [mlx-gpt-design.md](mlx-gpt-design.md) |
| Compile | `mx.compile` (stable on Apple Silicon) |
| Precision | float16 / bfloat16 (native unified memory) |
| Optimizer | Muon + AdamW ported to MLX — see [mlx-muon-design.md](mlx-muon-design.md) |
| Attention | `mx.fast.scaled_dot_product_attention` with sliding window |
| Distribution | single-device only |

---

## Checkpoint interop

Two options for cross-backend checkpoint exchange:

**Option A — numpy arrays**: `state_dict()` returns numpy arrays. Both PyTorch and MLX can read/write numpy natively.
- PyTorch → MLX: `torch.Tensor.numpy()` → `mx.array()`
- MLX → PyTorch: `np.array(mx_tensor)` → `torch.from_numpy()`

**Option B — safetensors**: Both frameworks support safetensors natively. Memory-mapped, zero-copy, compatible with `mlx-lm` tooling. Preferred for ecosystem compatibility.

---

## File layout

```
src/nanochat/
├── training/
│   ├── base/
│   │   ├── trainer.py       # BaseTrainer protocol + StepResult + TorchTrainer
│   │   ├── loop.py          # backend-agnostic training loop
│   │   ├── setup.py         # constructs TorchTrainer or MLXTrainer
│   │   └── ...
│   ├── mlx_trainer.py       # MLXTrainer implementation
│   ├── mlx_optimizer.py     # MLX Muon + AdamW
│   ├── compression_math.py  # pure numpy functions (backend-agnostic)
│   ├── compression_metrics.py  # stateful tracker, delegates to compression_math
│   └── ...
├── models/
│   ├── gpt.py               # PyTorch GPT
│   ├── mlx_gpt.py           # MLX GPT
│   └── ...
```

---

## Implementation order

1. ✅ **Entry-point refactor** — training/evaluation split into sub-packages with co-located state classes
2. ✅ **Checkpoint manager** — `CheckpointManager` protocol, typed metadata, `model_factory.py`
3. ✅ **MLX GPT** — see [mlx-gpt-design.md](mlx-gpt-design.md)
4. ✅ **MLX Muon** — see [mlx-muon-design.md](mlx-muon-design.md)
5. ✅ **BaseTrainer + TorchTrainer** — see [trainer-implementation-plan.md](trainer-implementation-plan.md)
6. ✅ **Backend-agnostic `loop.py`** — loop calls only protocol methods
7. **Checkpoint interop** — numpy or safetensors cross-backend handoff
8. ✅ **CLI integration** — see [cli-backend-integration.md](cli-backend-integration.md)
