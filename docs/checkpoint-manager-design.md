---
title: "Checkpoint Manager Design"
summary: "Design for a CheckpointManager protocol that abstracts checkpoint I/O, enabling multi-format support and backend-agnostic training."
read_when:
  - Implementing or reviewing the checkpoint manager refactor
  - Adding a new checkpoint format (safetensors, numpy)
  - Understanding how checkpoints integrate with the dual-trainer architecture
status: draft
last_updated: "2025-07-18"
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

Each training script constructs its own metadata dict with different keys — no schema,
no validation, easy to silently drop a key.

### 5. Checkpoint config scattered across training modes

`save_every`, `resume_from_step`, `model_tag` are duplicated across `TrainingConfig`,
`SFTConfig`, `RLConfig`, and `EvaluationConfig` with no single source of truth.

---

## Config changes

### `CommonConfig` — add `model_tag`

`model_tag` identifies which model to load. It is used by training, evaluation, and chat
alike — it belongs in `CommonConfig`, not duplicated per training mode.

```toml
[common]
model_tag = ""   # empty = auto-detect largest model
```

Remove `model_tag` from `TrainingConfig`, `SFTConfig`, `RLConfig`, `EvaluationConfig`.

### New `CheckpointConfig` section

```toml
[checkpoint]
format = "torch"        # torch | safetensors | numpy  (default: torch)
save_every = -1         # steps between saves, -1 = only at end
resume_from_step = -1   # step to resume from, -1 = disabled
keep_last_n = -1        # number of checkpoints to retain, -1 = keep all
```

`save_every` and `resume_from_step` are moved from `TrainingConfig`, `SFTConfig`, `RLConfig`.
`keep_last_n` is new — the manager prunes old checkpoint files after each save.

---

## Protocols

### `CheckpointStateProtocol`

Declares that a state class supports checkpoint serialization. All `xxxState` classes
satisfy this protocol.

```python
class CheckpointStateProtocol(Protocol):
    step: int

    @classmethod
    def fresh(cls) -> Self: ...

    def to_metadata(self) -> CheckpointMetadata: ...

    @classmethod
    def from_metadata(cls, meta: CheckpointMetadata) -> Self: ...
```

### `CheckpointManager` protocol

The manager only knows `CheckpointMetadata` — never concrete state types.

```python
class CheckpointManager(Protocol):
    def save(
        self,
        state: CheckpointStateProtocol,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any] | None,
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

### `CheckpointLogger` protocol

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

## Data classes

### `Checkpoint`

Returned by `manager.load()`, bundles everything together:

```python
@dataclass
class Checkpoint:
    step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    metadata: CheckpointMetadata
```

### `CheckpointMetadata`

Typed schema for the metadata JSON — replaces the ad-hoc dicts. Owned by the manager,
populated via `state.to_metadata()`.

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

### `LoopState`

The loop-specific slice of metadata, produced by `PretrainingState.to_metadata()`:

```python
@dataclass
class LoopState:
    min_val_bpb: float
    smooth_train_loss: float
    total_training_time: float
```

---

## State class changes

All `xxxState` classes replace `to_checkpoint()`/`from_checkpoint()` with
`to_metadata()`/`from_metadata()` to satisfy `CheckpointStateProtocol`.

### `PretrainingState`

```python
def to_metadata(self) -> CheckpointMetadata:
    return CheckpointMetadata(
        step=self.step,
        model_config=...,   # passed in from setup
        user_config=...,
        val_bpb=self.val_bpb,
        loop_state=LoopState(
            min_val_bpb=self.min_val_bpb,
            smooth_train_loss=self.smooth_train_loss,
            total_training_time=self.total_training_time,
        ),
        dataloader_state_dict=self.dataloader_state_dict,
    )

@classmethod
def from_metadata(cls, meta: CheckpointMetadata) -> Self:
    loop = meta.loop_state
    return cls(
        step=meta.step,
        val_bpb=meta.val_bpb,
        min_val_bpb=loop.min_val_bpb if loop else float("inf"),
        smooth_train_loss=loop.smooth_train_loss if loop else 0.0,
        total_training_time=loop.total_training_time if loop else 0.0,
        dataloader_state_dict=meta.dataloader_state_dict,
    )
```

`SFTState` and `RLState` follow the same pattern with their respective fields.

---

## Manager factory

A factory function reads `CheckpointConfig.format` and returns the right implementation:

```python
def make_checkpoint_manager(
    checkpoint_dir: str,
    config: CheckpointConfig,
    logger: CheckpointLogger | None = None,
) -> CheckpointManager: ...
```

---

## Implementations

### `TorchCheckpointManager`

Current behavior, extracted into the protocol:

- `save`: `torch.save` for model/optimizer `.pt`, `json.dump` for metadata
- `load`: `torch.load` + `json.load`
- `find_last_step`: glob `model_*.pt`
- Handles `_orig_mod.` prefix stripping, dtype conversion for CPU/MPS
- `keep_last_n`: prunes oldest `model_*.pt` + `optim_*.pt` + `meta_*.json` after save

### `SafetensorsCheckpointManager`

For cross-backend interop (dual-trainer):

- `save`: `safetensors.torch.save_file` for model, `json.dump` for metadata
- `load`: `safetensors.torch.load_file` or `mx.load` depending on caller
- Same directory layout, `.safetensors` extension instead of `.pt`

### `NumpyCheckpointManager`

Minimal, for testing and simple interop:

- `save`: `numpy.savez` for model, `json.dump` for metadata
- `load`: `numpy.load`

---

## What stays outside the manager

### Model construction

`build_model` stays separate — the manager returns raw state dicts, not model instances.
Model construction lives in `src/nanochat/model_factory.py` — a top-level utility used by
`training/`, `evaluation/`, and `chat/` alike. Keeping it outside `training/checkpoint/`
avoids a cross-package dependency from `chat/` and `evaluation/` into a training sub-package.

### Directory discovery

`find_largest_model` stays as a standalone utility — it's about directory layout, not
checkpoint format.

### Backward compatibility patching

`_patch_missing_config_keys` and `_patch_missing_keys` stay as standalone functions.
Called after loading, before model construction — not the manager's concern.

---

## File layout

```
src/nanochat/
├── model_factory.py             # build_model, load_model_from_dir, load_model
│                                #   used by training/, evaluation/, chat/
├── checkpoint/
│   ├── __init__.py              # re-exports: make_checkpoint_manager, CheckpointManager,
│   │                            #   Checkpoint, CheckpointMetadata, CheckpointStateProtocol
│   ├── protocol.py              # CheckpointManager, CheckpointStateProtocol,
│   │                            #   Checkpoint, CheckpointMetadata, LoopState
│   ├── torch_manager.py         # TorchCheckpointManager
│   ├── safetensors_manager.py   # SafetensorsCheckpointManager (future)
│   ├── logger.py                # CheckpointLogger, RankZeroLogger, SilentLogger
│   ├── factory.py               # make_checkpoint_manager
│   ├── discovery.py             # find_largest_model, find_last_step
│   └── compat.py                # _patch_missing_config_keys, _patch_missing_keys
├── training/
│   └── ...
```

---

## Usage

### Training loop

```python
manager = make_checkpoint_manager(
    checkpoint_dir=workspace.checkpoint_dir("base", config.common.model_tag),
    config=config.checkpoint,
    logger=RankZeroLogger(),
)

# Resume
if config.checkpoint.resume_from_step > 0:
    ckpt = manager.load(step=config.checkpoint.resume_from_step, device=device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(ckpt.model_state)
    optimizer.load_state_dict(ckpt.optimizer_state)
    state = PretrainingState.from_metadata(ckpt.metadata)
else:
    state = PretrainingState.fresh()

# Save
manager.save(state, orig_model.state_dict(), optimizer.state_dict(), rank=ddp_rank)
```

### Dual-trainer checkpoint handoff

```python
# Train with MLX, save as safetensors
mlx_manager = make_checkpoint_manager(ckpt_dir, CheckpointConfig(format="safetensors"))
mlx_manager.save(state, trainer.state_dict(), None)

# Load in PyTorch for SFT
torch_manager = make_checkpoint_manager(ckpt_dir, CheckpointConfig(format="safetensors"))
ckpt = torch_manager.load(step=step, device=device)
model.load_state_dict(ckpt.model_state)
state = SFTState.from_metadata(ckpt.metadata)
```

---

## Implementation notes

### Python 3.12 type annotations

All files use Python 3.12 native annotations. No `from __future__ import annotations`, no
`typing.Optional`, no `typing.Union`. Use `X | Y`, `X | None`, `list[X]`, `dict[K, V]`
directly. `Callable` and `Generator` from `collections.abc`, not `typing`.

### No dynamic imports

`factory.py` selects the manager implementation at call time but uses static imports at
the top of the module — no `importlib`, no `__import__`, no inline `import` statements.

```python
# correct
from nanochat.checkpoint.torch_manager import TorchCheckpointManager
from nanochat.checkpoint.safetensors_manager import SafetensorsCheckpointManager

def make_checkpoint_manager(checkpoint_dir: str, config: CheckpointConfig, ...) -> CheckpointManager:
    if config.format == "safetensors":
        return SafetensorsCheckpointManager(checkpoint_dir, ...)
    return TorchCheckpointManager(checkpoint_dir, ...)
```

### Tests

All new modules get unit tests under `tests/test_checkpoint/`.

| Test file                  | Coverage                                                              |
| -------------------------- | --------------------------------------------------------------------- |
| `test_protocol.py`         | `CheckpointMetadata.to_dict()` / `from_dict()` round-trip, `LoopState` |
| `test_torch_manager.py`    | save/load round-trip, `keep_last_n` pruning, `find_last_step`, `exists()` |
| `test_factory.py`          | `make_checkpoint_manager` returns correct type per `format`           |
| `test_discovery.py`        | `find_largest_model`, `find_last_step`                                |
| `test_compat.py`           | `_patch_missing_config_keys`, `_patch_missing_keys`                   |
| `test_model_factory.py`    | `build_model` round-trip with a tiny GPT config                       |

Existing `tests/test_training/test_checkpoint.py` updated to cover the new API.
`SilentLogger` used throughout tests — no real logging.

## Implementation order

### Step 1 — Config changes

- Create `src/nanochat/config/checkpoint.py` — `CheckpointConfig` with `format`, `save_every`, `resume_from_step`, `keep_last_n`
- `common.py` — add `model_tag: str | None`; drop `from __future__`, `Optional` → `| None`
- `training.py` — remove `model_tag`, `save_every`, `resume_from_step`; fix annotations
- `sft.py` — remove `model_tag`; rename `model_step` → `source_step`; fix annotations
- `rl.py` — remove `model_tag`, `save_every`; rename `model_step` → `source_step`; fix annotations
- `evaluation.py` — remove `model_tag`; fix annotations
- `config.py` — add `checkpoint: CheckpointConfig`, register in `SECTION_CLS`; fix annotations
- `loader.py` — add `add_checkpoint()`; fix annotations
- `__init__.py` — export `CheckpointConfig`

### Step 2 — `src/nanochat/checkpoint/` package

- `protocol.py` — `CheckpointStateProtocol`, `CheckpointManager`, `Checkpoint`, `CheckpointMetadata`, `LoopState`
- `logger.py` — `CheckpointLogger`, `RankZeroLogger`, `SilentLogger`
- `torch_manager.py` — `TorchCheckpointManager`
- `factory.py` — `make_checkpoint_manager`
- `discovery.py` — `find_largest_model`, `find_last_step`
- `compat.py` — `_patch_missing_config_keys`, `_patch_missing_keys`
- `__init__.py` — re-exports

### Step 3 — `src/nanochat/model_factory.py`

- Promote `build_model`, `load_model_from_dir`, `load_model`, `load_optimizer_state` from `training/checkpoint.py`
- Update all callers: `chat/`, `evaluation/base/`, `evaluation/chat/`, `training/sft/`, `training/rl/`

### Step 4 — Update state classes

- `training/base/state.py` — `to_checkpoint()` → `to_metadata()`, `from_checkpoint()` → `from_metadata()`; satisfy `CheckpointStateProtocol`
- `training/sft/state.py` — same
- `training/rl/state.py` — same

### Step 5 — Update training loops

- `training/base/loop.py` — use `make_checkpoint_manager`, `state.to_metadata()`, `State.from_metadata()`; read `save_every`/`resume_from_step` from `config.checkpoint`
- `training/sft/loop.py` — same; read `source_step` from `config.sft`
- `training/rl/loop.py` — same; read `source_step` from `config.rl`

### Step 6 — Update evaluation and chat

- `evaluation/base/loop.py` — use `model_factory` and `config.common.model_tag`
- `evaluation/chat/loop.py` — same
- `chat/chat_cli.py` — use `model_factory` and `config.common.model_tag`
- `chat/server/worker_pool.py` — same

### Step 7 — Delete `training/checkpoint.py`

- Verify no remaining imports, then delete

### Step 8 — Tests

- Create `tests/test_checkpoint/` with `test_protocol.py`, `test_torch_manager.py`, `test_factory.py`, `test_discovery.py`, `test_compat.py`
- Create `tests/test_model_factory.py`
- Update `tests/test_training/test_checkpoint.py` to cover new API
- Update `tests/test_config/` for `CheckpointConfig`, `model_tag` in `CommonConfig`, `source_step` rename

### Step 9 — Docs

- `docs/code-structure.md` — add `checkpoint/` and `model_factory.py` to package map and dependency rules
- `docs/roadmap.md` — move checkpoint manager from deferred to completed
- `CHANGELOG.md` — add unreleased entry

---

## Dependencies

- Entry point refactor ✅ — state classes (`PretrainingState`, `SFTState`, `RLState`) already exist
- Workspace module ✅ — `workspace.checkpoint_dir()` already in place
- `nanochat/checkpoint/` imports only from `common/`, `config/`, `workspace` — no dependency on `training/` or `evaluation/`
- `training/`, `evaluation/`, `chat/` all import from `nanochat/checkpoint/` and `nanochat/model_factory`
- Enables [dual-trainer architecture](dual-trainer-architecture.md) checkpoint interop
