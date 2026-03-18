---
title: "Checkpoint Manager Design"
summary: "Design for a CheckpointManager protocol that abstracts checkpoint I/O, enabling multi-format support and backend-agnostic training."
read_when:
  - Implementing or reviewing the checkpoint manager refactor
  - Adding a new checkpoint format (safetensors, numpy)
  - Understanding how checkpoints integrate with the dual-trainer architecture
status: draft
last_updated: "2025-07-14"
---

# Checkpoint Manager Design

Goal: replace the current bag of free functions in `checkpoint.py` with a `CheckpointManager`
protocol and concrete implementations. This decouples checkpoint format from training logic,
enables multi-format support (torch, safetensors, numpy), and is a prerequisite for the
[dual-trainer architecture](dual-trainer-architecture.md).

---

## Problems with current design

### 1. Format is hardcoded to torch

`save_checkpoint` uses `torch.save`, `load_checkpoint` uses `torch.load`. An MLX trainer
can't use these — it would need `mx.save_safetensors` / `mx.load`. Every new format
requires touching every call site.

### 2. Mixed responsibilities

`checkpoint.py` currently handles:
- Low-level I/O (`save_checkpoint`, `load_checkpoint`)
- Model construction (`build_model`) — imports `GPT`, `GPTConfig`, `get_tokenizer`
- Directory discovery (`find_largest_model`, `find_last_step`)
- Convenience wrappers (`load_model_from_dir`, `load_model`, `load_optimizer_state`)
- Backward compatibility patching (`_patch_missing_config_keys`, `_patch_missing_keys`)

### 3. No logging abstraction

Uses a module-level `logger` and a custom `log0` that checks `RANK` env var. Logging
behavior is baked into the I/O functions.

### 4. Caller builds metadata dict inline

Each training script constructs its own metadata dict with different keys:

```python
# train_base.py
{"step": step, "val_bpb": val_bpb, "model_config": ..., "user_config": ...,
 "dataloader_state_dict": ..., "loop_state": {"min_val_bpb": ..., ...}}

# train_sft.py
{"step": step, "val_bpb": val_bpb, "model_config": ..., "user_config": ...}

# train_rl.py
{"model_config": model_config_kwargs}
```

No schema, no validation — easy to silently drop a key.

---

## Design

### CheckpointManager protocol

```python
class CheckpointManager(Protocol):
    def save(
        self,
        step: int,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any] | None,
        metadata: CheckpointMetadata,
        rank: int = 0,
    ) -> None: ...

    def load(
        self,
        step: int,
        device: Any,
        load_optimizer: bool = False,
        rank: int = 0,
    ) -> Checkpoint: ...

    def find_last_step(self) -> int: ...

    def exists(self, step: int) -> bool: ...
```

### Checkpoint dataclass

Returned by `load`, bundles everything together:

```python
@dataclass
class Checkpoint:
    step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    metadata: CheckpointMetadata
```

### CheckpointMetadata dataclass

Typed schema for the metadata JSON — replaces the ad-hoc dicts:

```python
@dataclass
class CheckpointMetadata:
    step: int
    model_config: dict[str, Any]
    user_config: dict[str, Any]
    val_bpb: float | None = None
    loop_state: LoopState | None = None
    dataloader_state_dict: dict | None = None

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> Self: ...
```

`LoopState` is the subset that `TrainingState.to_checkpoint()` produces:

```python
@dataclass
class LoopState:
    min_val_bpb: float
    smooth_train_loss: float
    total_training_time: float
```

### CheckpointLogger protocol

Decouples logging from I/O:

```python
class CheckpointLogger(Protocol):
    def info(self, message: str) -> None: ...
    def warning(self, message: str) -> None: ...
```

Two implementations:
- `RankZeroLogger` — logs only on rank 0 (current behavior, uses `logging`)
- `SilentLogger` — no-op (for testing)

---

## Implementations

### TorchCheckpointManager

Current behavior, extracted into the protocol:

- `save`: `torch.save` for model/optimizer `.pt`, `json.dump` for metadata
- `load`: `torch.load` + `json.load`
- `find_last_step`: glob `model_*.pt`
- Handles `_orig_mod.` prefix stripping, dtype conversion for CPU/MPS

### SafetensorsCheckpointManager

For cross-backend interop (dual-trainer):

- `save`: `safetensors.torch.save_file` for model, `json.dump` for metadata
- `load`: `safetensors.torch.load_file` or `mx.load` depending on caller
- Same directory layout, different file extension (`.safetensors` instead of `.pt`)

### NumpyCheckpointManager

Minimal, for testing and simple interop:

- `save`: `numpy.savez` for model, `json.dump` for metadata
- `load`: `numpy.load`

---

## What stays outside the manager

### Model construction

`build_model` (constructing a `GPT` from config + state_dict) stays separate. The manager
only handles serialization — it returns raw state dicts, not model instances. Model
construction belongs in a factory or the trainer itself.

### Directory discovery

`find_largest_model` (scanning phase dirs for model tags) stays as a standalone utility.
It's about directory layout, not checkpoint format.

### Backward compatibility patching

`_patch_missing_config_keys` and `_patch_missing_keys` stay as standalone functions.
They're called after loading, before model construction — not the manager's concern.

---

## File layout

```
src/nanochat/training/
├── checkpoint/
│   ├── __init__.py              # re-exports
│   ├── protocol.py              # CheckpointManager protocol, Checkpoint, CheckpointMetadata
│   ├── torch_manager.py         # TorchCheckpointManager
│   ├── safetensors_manager.py   # SafetensorsCheckpointManager (future)
│   ├── logger.py                # CheckpointLogger, RankZeroLogger, SilentLogger
│   ├── discovery.py             # find_largest_model, find_last_step
│   ├── compat.py                # _patch_missing_config_keys, _patch_missing_keys
│   └── model_builder.py         # build_model, load_model_from_dir
```

---

## Usage

### Training loop (train_base.py)

```python
ckpt_dir = workspace.checkpoint_dir("base", model_tag)
manager = TorchCheckpointManager(ckpt_dir, logger=RankZeroLogger())

# Resume
if resuming:
    ckpt = manager.load(step=resume_step, device=device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(ckpt.model_state)
    optimizer.load_state_dict(ckpt.optimizer_state)
    state = PretrainingState.from_checkpoint(ckpt.metadata)

# Save
manager.save(
    step=state.step,
    model_state=orig_model.state_dict(),
    optimizer_state=optimizer.state_dict(),
    metadata=CheckpointMetadata(
        step=state.step,
        model_config=model_config_kwargs,
        user_config=user_config,
        val_bpb=state.val_bpb,
        loop_state=state.to_loop_state(),
        dataloader_state_dict=state.dataloader_state_dict,
    ),
    rank=ddp_rank,
)
```

### Dual-trainer checkpoint handoff

```python
# Train with MLX, save as safetensors
mlx_manager = SafetensorsCheckpointManager(ckpt_dir)
mlx_manager.save(step=step, model_state=trainer.state_dict(), ...)

# Load in PyTorch for SFT
torch_manager = SafetensorsCheckpointManager(ckpt_dir)
ckpt = torch_manager.load(step=step, device=device)
model.load_state_dict(ckpt.model_state)
```

---

## Implementation order

1. Define `protocol.py` — `CheckpointManager`, `Checkpoint`, `CheckpointMetadata`, `LoopState`
2. Define `logger.py` — `CheckpointLogger`, `RankZeroLogger`, `SilentLogger`
3. Extract `TorchCheckpointManager` from current `checkpoint.py`
4. Extract `discovery.py`, `compat.py`, `model_builder.py` from current `checkpoint.py`
5. Update all call sites (`train_base`, `train_sft`, `train_rl`, `base_eval`, `chat_eval`)
6. Delete old `checkpoint.py`
7. Add `SafetensorsCheckpointManager` when dual-trainer lands

---

## Dependencies

- Requires [TrainingState refactor](training-state-refactor.md) for `LoopState` / `to_loop_state()`
- Benefits from [workspace module](workspace-design.md) — `workspace.checkpoint_dir()` replaces `checkpoint_dir(base_dir, ...)` threading
- Enables [dual-trainer architecture](dual-trainer-architecture.md) checkpoint interop
- Independent of [scheduler placement](scheduler-placement-study.md)
