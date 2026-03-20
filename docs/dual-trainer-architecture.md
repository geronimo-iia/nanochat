---
title: "Dual Trainer Architecture"
summary: "Index for the BaseTrainer protocol with TorchTrainer and MLXTrainer backends."
read_when:
  - Getting an overview of the dual-trainer architecture
  - Finding the right sub-document for a specific component
  - Reviewing implementation order and status
status: active
last_updated: "2025-07-24"
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

Goal: a checkpoint written by `TorchTrainer` can be loaded by `MLXTrainer` and vice versa,
without either trainer knowing about the other's array type.

### Design

The manager is the conversion boundary. `load()` always returns `dict[str, np.ndarray]` for
model state — numpy is the common currency both frameworks can consume. Each trainer's
`load_state_dicts` converts from numpy to its native type. Neither trainer needs to know what
format was on disk or what the other backend uses.

Option B (safetensors) is the on-disk format: memory-mapped, zero-copy, compatible with
`mlx-lm` tooling, and natively supported by both PyTorch and MLX. Optimizer state uses `.npz`
(numpy archive) since optimizer tensors are framework-specific and not shared across backends.

### Conversion boundary

```
disk (.safetensors)  →  manager.load()  →  dict[str, np.ndarray]
                                                    ↓
                              TorchTrainer.load_state_dicts()  →  torch.Tensor
                              MLXTrainer.load_state_dicts()    →  mx.array
```

Saving is the mirror: each trainer's `model_state_dict()` returns its native type, and
`convert.to_numpy()` is called by the manager before writing to disk.

### Files

| File | Role |
|---|---|
| `checkpoint/convert.py` | `to_numpy(state_dict)` and `from_numpy(state_dict, framework)` — handles `torch.Tensor`, `mx.array`, `np.ndarray` |
| `checkpoint/safetensors_manager.py` | `SafetensorsCheckpointManager` — model as `.safetensors`, optimizer as `.pt`, meta as `.json` |
| `checkpoint/factory.py` | add `"safetensors"` branch |
| `checkpoint/__init__.py` | re-export `SafetensorsCheckpointManager` |
| `pyproject.toml` | add `safetensors` dependency |
| `tests/test_checkpoint/test_safetensors_manager.py` | round-trip, weights restored, prune, exists |

`compat.py` torch-specific patches are unchanged — they are called from `TorchTrainer.load_state_dicts`
which already speaks torch tensors after conversion.

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
7. ✅ **Checkpoint interop** — numpy or safetensors cross-backend handoff
8. ✅ **CLI integration** — see [cli-backend-integration.md](cli-backend-integration.md)
9. **Documentation** — update and create reference docs reflecting the completed architecture

---

## Step 9 — Documentation plan

### Updates to existing docs

| File | Changes |
|---|---|
| `code-structure.md` | Add `convert.py`, `safetensors_manager.py` to checkpoint table; add `compression_math.py`, `base/trainer.py` to training table; update CLI→training flow to show `BaseTrainer`/`TorchTrainer`; update dependency rules |
| `m3-max-guide.md` | Add MLX section: `--backend=mlx` flag, MLX vs MPS comparison, when to use each, batch size recommendations |
| `mlx-gpt-design.md` | Convert from planning doc to reference doc: remove "Step 3 of...", remove Deferred section, remove Open questions resolved, status `active` |
| `mlx-muon-design.md` | Same treatment as `mlx-gpt-design.md`: remove planning framing, remove Deferred section, status `active` |

### New docs

**`checkpoint-interop.md`** — checkpoint system reference
- `CheckpointManager` protocol and `Checkpoint` / `CheckpointMetadata` types
- `TorchCheckpointManager` vs `SafetensorsCheckpointManager` — when to use each
- `convert.py` conversion boundary — `to_numpy`, `from_numpy_torch`, `from_numpy_mlx`
- Mermaid diagram: save flow (trainer → manager → disk) and load flow (disk → manager → trainer)
- Optimizer state note: currently `torch.save`/`torch.load`, future split into safetensors + JSON

**`trainer-protocol.md`** — `BaseTrainer` protocol reference
- `StepResult` dataclass
- All 7 protocol methods with signatures and contracts
- `TorchTrainer` internals: accumulation loop, scaler path, `_last_x`/`_last_y` snapshot
- Mermaid diagram: `loop.py` → `BaseTrainer` call sequence per training step
- How to implement a new backend (checklist)

**`mlx-backend.md`** — MLX stack overview tying the pieces together
- MLX GPT forward pass parity (references `mlx-gpt-design.md`)
- MLX Muon optimizer parity (references `mlx-muon-design.md`)
- What `MLXTrainer` will wire up: `forward_backward`, `step`, `eval_context`
- Mermaid diagram: full MLX stack from `loop.py` call to `mx.eval()`
