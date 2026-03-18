---
title: "TrainingState Refactor Plan"
summary: "Plan to extract mutable loop state into dataclasses across all training and evaluation entry points."
read_when:
  - Implementing or reviewing the TrainingState refactor
  - Understanding the mutable state pattern across train_base, train_sft, train_rl, base_eval, chat_eval
status: draft
last_updated: "2025-07-14"
---

# TrainingState Refactor Plan

Goal: extract mutable loop variables into explicit dataclasses across all entry points.
Currently every training/eval function scatters state as bare local variables — this makes
resumption fragile, testing hard, and the dual-trainer refactor impossible.

---

## Current state inventory

Each entry point has its own set of mutable loop variables:

### train_base.py — `train_base_model()` closure

| Variable | Type | Checkpoint? |
|---|---|---|
| `step` | int | ✅ `meta_data["step"]` |
| `val_bpb` | float \| None | ✅ `meta_data["val_bpb"]` |
| `min_val_bpb` | float | ✅ `loop_state["min_val_bpb"]` |
| `smooth_train_loss` | float | ✅ `loop_state["smooth_train_loss"]` |
| `total_training_time` | float | ✅ `loop_state["total_training_time"]` |
| `dataloader_state_dict` | dict \| None | ✅ `meta_data["dataloader_state_dict"]` |

### train_sft.py — flat locals

| Variable | Type | Checkpoint? |
|---|---|---|
| `step` | int | ❌ |
| `val_bpb` | float \| None | ❌ |
| `min_val_bpb` | float | ❌ |
| `smooth_train_loss` | float | ❌ |
| `total_training_time` | float | ❌ |
| `last_step` | bool | ❌ |
| `approx_progress` | float | ❌ |
| `current_epoch` | int | ❌ |
| `progress` | float | ❌ |

### train_rl.py — flat locals

| Variable | Type | Checkpoint? |
|---|---|---|
| `step` | int (loop var) | ❌ |
| `rewards_list` | list[float] | ❌ |
| `sequence_lengths` | list[int] | ❌ |

### base_eval.py — flat locals

| Variable | Type |
|---|---|
| `core_results` | dict \| None |
| `bpb_results` | dict |
| `samples` | list[str] |
| `unconditioned_samples` | list[str] |

### chat_eval.py — flat locals

| Variable | Type |
|---|---|
| `results` | dict[str, float] |

---

## Step 1 — Create state dataclasses in `training/train_state.py`

### BaseTrainingState

Shared fields across base and SFT training:

```python
@dataclass
class BaseTrainingState:
    step: int
    val_bpb: float | None
    min_val_bpb: float
    smooth_train_loss: float
    total_training_time: float

    @classmethod
    def fresh(cls) -> Self: ...

    def to_checkpoint(self) -> dict: ...

    @classmethod
    def from_checkpoint(cls, meta_data: dict) -> Self: ...
```

### PretrainingState

Extends base with checkpoint-resumable dataloader state:

```python
@dataclass
class PretrainingState(BaseTrainingState):
    dataloader_state_dict: dict | None
```

### SFTState

Extends base with SFT-specific progress tracking:

```python
@dataclass
class SFTState(BaseTrainingState):
    last_step: bool
    approx_progress: float
    current_epoch: int
    progress: float
```

### RLState

Minimal — RL doesn't checkpoint loop state:

```python
@dataclass
class RLState:
    step: int
```

---

## Step 2 — Create eval result dataclasses in `evaluation/eval_state.py`

### BaseEvalResult

```python
@dataclass
class BaseEvalResult:
    core_results: dict | None = None
    bpb_results: dict = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)
    unconditioned_samples: list[str] = field(default_factory=list)
```

### ChatEvalResult

```python
@dataclass
class ChatEvalResult:
    results: dict[str, float] = field(default_factory=dict)
    chatcore: float | None = None
```

---

## Step 3 — Update entry points

### 3a. train_base.py

- Import `PretrainingState`
- Replace `if resuming / else` block with `PretrainingState.fresh()` / `.from_checkpoint()`
- Replace bare variable access with `state.*`
- Replace `loop_state` dict in `save_checkpoint` with `state.to_checkpoint()`
- Eliminate the `train_base_model()` closure — promote to module-level function

### 3b. train_sft.py

- Import `SFTState`
- Replace scattered locals with `state.*`
- The `nonlocal last_step, approx_progress, current_epoch` in the data generator
  becomes reads/writes on `state.*` (pass state to the generator)

### 3c. train_rl.py

- Import `RLState`
- Minimal change — just wrap `step` in the dataclass for consistency

### 3d. base_eval.py

- Import `BaseEvalResult`
- Replace scattered result locals with `result.*`
- Return `BaseEvalResult` instead of logging inline

### 3e. chat_eval.py

- Import `ChatEvalResult`
- Replace `results` dict with `result.*`
- Return `ChatEvalResult` instead of logging inline

---

## Step 4 — Verify checkpoint round-trip (base training)

Confirm `PretrainingState.from_checkpoint` reads exactly the keys that
`state.to_checkpoint()` writes:

| `from_checkpoint` reads | `to_checkpoint` writes |
|---|---|
| `meta_data["step"]` | `"step": state.step` |
| `meta_data["val_bpb"]` | `"val_bpb": state.val_bpb` |
| `meta_data["loop_state"]["min_val_bpb"]` | `"loop_state": {"min_val_bpb": ...}` |
| `meta_data["loop_state"]["smooth_train_loss"]` | `"loop_state": {"smooth_train_loss": ...}` |
| `meta_data["loop_state"]["total_training_time"]` | `"loop_state": {"total_training_time": ...}` |
| `meta_data["dataloader_state_dict"]` | `"dataloader_state_dict": ...` |

---

## Implementation order

1. `training/train_state.py` — all training state dataclasses
2. `evaluation/eval_state.py` — all eval result dataclasses
3. `train_base.py` — highest value (closure elimination + checkpoint resumption)
4. `train_sft.py` — second highest (eliminates `nonlocal` hack in data generator)
5. `base_eval.py` + `chat_eval.py` — clean up eval result passing
6. `train_rl.py` — minimal, for consistency

---

## Files touched

| File | Change |
|---|---|
| `src/nanochat/training/train_state.py` | New — `PretrainingState`, `SFTState`, `RLState` |
| `src/nanochat/evaluation/eval_state.py` | New — `BaseEvalResult`, `ChatEvalResult` |
| `src/nanochat/training/train_base.py` | Use `PretrainingState`, eliminate closure |
| `src/nanochat/training/train_sft.py` | Use `SFTState`, eliminate `nonlocal` |
| `src/nanochat/training/train_rl.py` | Use `RLState` |
| `src/nanochat/evaluation/base_eval.py` | Use `BaseEvalResult` |
| `src/nanochat/evaluation/chat_eval.py` | Use `ChatEvalResult` |
