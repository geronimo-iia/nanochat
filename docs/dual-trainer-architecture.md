---
title: "Dual Trainer Architecture"
summary: "Plan to introduce a BaseTrainer protocol with TorchTrainer (current code) and MLXTrainer (MLX + Muon on Apple Silicon)."
read_when:
  - Implementing or reviewing the TorchTrainer / MLXTrainer split
  - Adding a new training backend
  - Understanding how MLX training integrates with nanochat
status: draft
last_updated: "2025-07-19"
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
3. **BaseTrainer protocol + TorchTrainer** — extract current model/optimizer code, verify nothing breaks
4. **Backend-agnostic train_base.py** — loop calls only protocol methods
5. **MLX model** — port GPT to `mlx.nn`, validate forward pass matches PyTorch
6. **MLX Muon** — port optimizer, validate update step matches PyTorch
7. **MLXTrainer** — wire model + optimizer, run a d4 smoke test
8. **Numpy checkpoint interop** — verify resume and handoff across backends
9. **CLI integration** — `--backend torch|mlx` flag, auto-detection

---

## Open questions

- Should MLXTrainer support the compression metrics tracker, or defer that?
- MLX `scaled_dot_product_attention` sliding window support — verify API availability.
- Does `mx.nn.value_and_grad` handle the grad accumulation pattern, or do we accumulate manually?
