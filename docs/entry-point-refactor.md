---
title: "Entry Point Refactor Plan"
summary: "Split large training and evaluation entry points into focused sub-packages with co-located state dataclasses."
read_when:
  - Implementing or reviewing the entry point refactor
  - Understanding the package split rationale
  - Implementing TrainingState extraction
status: draft
last_updated: "2025-07-16"
---

# Entry Point Refactor Plan

## Motivation

The five entry points are too large to edit safely:

| File                      | Lines |
| ------------------------- | ----- |
| `training/train_base.py`  | 777   |
| `training/train_sft.py`   | 577   |
| `training/train_rl.py`    | 347   |
| `evaluation/base_eval.py` | 185   |
| `evaluation/chat_eval.py` | 340   |

Files above ~200 lines are risky to edit — context loss, misaligned replacements, partial rewrites.
The fix: for each entry point, create its package with a co-located `state.py`, implement the modules using that state, then delete the old flat file. Each resulting file stays under ~200 lines.

---

## Current state inventory

Exact mutable variables in each entry point today, and whether they survive a checkpoint.

### train_base.py — inside `train_base_model()` closure

| Variable                | Type            | Checkpointed?                                      |
| ----------------------- | --------------- | -------------------------------------------------- |
| `step`                  | `int`           | ✅ `meta_data["step"]`                              |
| `val_bpb`               | `float \| None` | ✅ `meta_data["val_bpb"]`                           |
| `min_val_bpb`           | `float`         | ✅ `meta_data["loop_state"]["min_val_bpb"]`         |
| `smooth_train_loss`     | `float`         | ✅ `meta_data["loop_state"]["smooth_train_loss"]`   |
| `total_training_time`   | `float`         | ✅ `meta_data["loop_state"]["total_training_time"]` |
| `dataloader_state_dict` | `dict \| None`  | ✅ `meta_data["dataloader_state_dict"]`             |

### train_sft.py — flat locals + `nonlocal`

| Variable              | Type            | Checkpointed?                             |
| --------------------- | --------------- | ----------------------------------------- |
| `step`                | `int`           | ❌                                         |
| `val_bpb`             | `float \| None` | ❌                                         |
| `min_val_bpb`         | `float`         | ❌                                         |
| `smooth_train_loss`   | `float`         | ❌                                         |
| `total_training_time` | `float`         | ❌                                         |
| `last_step`           | `bool`          | ❌ — shared via `nonlocal` with dataloader |
| `approx_progress`     | `float`         | ❌ — shared via `nonlocal` with dataloader |
| `current_epoch`       | `int`           | ❌ — shared via `nonlocal` with dataloader |
| `progress`            | `float`         | ❌                                         |

### train_rl.py — flat locals

| Variable           | Type             | Checkpointed? |
| ------------------ | ---------------- | ------------- |
| `step`             | `int` (loop var) | ❌             |
| `rewards_list`     | `list[float]`    | ❌             |
| `sequence_lengths` | `list[int]`      | ❌             |

### base_eval.py — flat locals

| Variable                | Type           |
| ----------------------- | -------------- |
| `core_results`          | `dict \| None` |
| `bpb_results`           | `dict`         |
| `samples`               | `list[str]`    |
| `unconditioned_samples` | `list[str]`    |

### chat_eval.py — flat locals

| Variable  | Type               |
| --------- | ------------------ |
| `results` | `dict[str, float]` |

---

## State dataclass designs

All state dataclasses, one per entry point package.

### training/base/state.py — PretrainingState

See [Phase 1](#phase-1--trainingbase-package) for full implementation with `fresh()`, `to_checkpoint()`, `from_checkpoint()`.

### training/sft/state.py — SFTState

```python
@dataclass
class SFTState:
    step: int
    val_bpb: float | None
    min_val_bpb: float
    smooth_train_loss: float
    total_training_time: float
    last_step: bool          # replaces nonlocal — passed into dataloader
    approx_progress: float   # replaces nonlocal — updated by dataloader
    current_epoch: int       # replaces nonlocal — updated by dataloader
    progress: float

    @classmethod
    def fresh(cls) -> Self:
        return cls(
            step=0, val_bpb=None, min_val_bpb=float("inf"),
            smooth_train_loss=0.0, total_training_time=0.0,
            last_step=False, approx_progress=0.0, current_epoch=1, progress=0.0,
        )
```

The dataloader signature becomes `sft_data_generator_bos_bestfit(state: SFTState, ...)` — it mutates `state.last_step`, `state.approx_progress`, `state.current_epoch` directly. No `nonlocal`.

### training/rl/state.py — RLState

```python
@dataclass
class RLState:
    step: int

    @classmethod
    def fresh(cls) -> Self:
        return cls(step=0)
```

Minimal — RL does not checkpoint loop state. Exists for consistency and future extension.

### evaluation/base/state.py — BaseEvalResult

```python
@dataclass
class BaseEvalResult:
    core_results: dict | None = None
    bpb_results: dict = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)
    unconditioned_samples: list[str] = field(default_factory=list)
```

### evaluation/chat/state.py — ChatEvalResult

```python
@dataclass
class ChatEvalResult:
    results: dict[str, float] = field(default_factory=dict)
```

---

## Target layout

```
training/
  base/
    __init__.py      # exposes train_base()
    state.py         # PretrainingState
    fp8.py           # disable_fp8() context manager
    schedulers.py    # base_lr_scheduler, base_muon_momentum_scheduler, base_weight_decay_scheduler
    setup.py         # compute init, model build, optimizer, dataloaders
    loop.py          # training loop, uses PretrainingState
  sft/
    __init__.py      # exposes train_sft()
    state.py         # SFTState
    schedulers.py    # sft_lr_scheduler, sft_muon_momentum_scheduler
    dataloader.py    # sft_data_generator_bos_bestfit, uses SFTState (no nonlocal)
    setup.py         # compute init, model load, optimizer
    loop.py          # training loop, uses SFTState
  rl/
    __init__.py      # exposes train_rl()
    state.py         # RLState
    schedulers.py    # rl_lr_scheduler
    rollout.py       # get_batch()
    eval.py          # run_gsm8k_eval()
    loop.py          # training loop, uses RLState

evaluation/
  base/
    __init__.py      # exposes run_base_eval()
    state.py         # BaseEvalResult
    loop.py          # uses BaseEvalResult
  chat/
    __init__.py      # exposes run_chat_eval()
    state.py         # ChatEvalResult
    loop.py          # uses ChatEvalResult
```

### Notes

- `fp8.py` lives under `training/base/` — training concern (BF16 eval within the training loop), not a model concern. `models/` stays pure architecture.
- Old flat files (`train_base.py`, `train_sft.py`, `train_rl.py`, `base_eval.py`, `chat_eval.py`) are deleted only after their replacement is complete and tested.
- All external callers (`scripts/`, `dev/`) import from the package `__init__` — import path changes only.
- `training/checkpoint.py`, `training/dataloader.py`, `training/scaling.py`, `training/optimizer.py`, `training/compression_metrics.py` are unchanged.

---

## Phase 1 — training/base/ package

For each module: implement with `PretrainingState`, then delete `train_base.py`.

| Module          | Content                                                                                                               | Est. lines |
| --------------- | --------------------------------------------------------------------------------------------------------------------- | ---------- |
| `state.py`      | `PretrainingState` — `fresh()`, `to_checkpoint()`, `from_checkpoint()`                                                | ~40        |
| `fp8.py`        | `disable_fp8()` context manager                                                                                       | ~40        |
| `schedulers.py` | `base_lr_scheduler`, `base_muon_momentum_scheduler`, `base_weight_decay_scheduler`                                    | ~40        |
| `setup.py`      | compute init, `build_model_meta()`, FP8 conversion, `torch.compile`, optimizer, dataloaders, scaling law computations | ~200       |
| `loop.py`       | training loop — uses `PretrainingState`, eliminates closure                                                           | ~200       |
| `__init__.py`   | `train_base(config)` — orchestrates setup then loop                                                                   | ~30        |

### State design example — PretrainingState

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Self, cast


@dataclass
class PretrainingState:
    step: int
    val_bpb: float | None
    min_val_bpb: float
    smooth_train_loss: float
    total_training_time: float
    dataloader_state_dict: dict | None

    @classmethod
    def fresh(cls) -> Self:
        return cls(
            step=0,
            val_bpb=None,
            min_val_bpb=float("inf"),
            smooth_train_loss=0.0,
            total_training_time=0.0,
            dataloader_state_dict=None,
        )

    @classmethod
    def from_checkpoint(cls, meta_data: dict) -> Self:
        loop = cast(dict, meta_data["loop_state"])
        return cls(
            step=meta_data["step"],
            val_bpb=meta_data["val_bpb"],
            min_val_bpb=loop["min_val_bpb"],
            smooth_train_loss=loop["smooth_train_loss"],
            total_training_time=loop["total_training_time"],
            dataloader_state_dict=meta_data["dataloader_state_dict"],
        )

    def to_checkpoint(self, model_config: dict, user_config: dict, batch_config: dict) -> dict:
        """Produces the full meta_data dict passed to save_checkpoint()."""
        return {
            "step": self.step,
            "val_bpb": self.val_bpb,
            "model_config": model_config,
            "user_config": user_config,
            **batch_config,  # device_batch_size, max_seq_len, total_batch_size
            "dataloader_state_dict": self.dataloader_state_dict,
            "loop_state": {
                "min_val_bpb": self.min_val_bpb,
                "smooth_train_loss": self.smooth_train_loss,
                "total_training_time": self.total_training_time,
            },
        }
```

Usage in `loop.py`:

```python
# init
state = PretrainingState.fresh() if not resuming else PretrainingState.from_checkpoint(meta_data)

# save
save_checkpoint(ckpt_dir, state.step, model.state_dict(), optimizer.state_dict(),
                state.to_checkpoint(model_config_kwargs, user_config, batch_config), rank=ddp_rank)
```

### Checkpoint round-trip

Exact key mapping between `from_checkpoint` and `to_checkpoint`:

| `from_checkpoint` reads                          | `to_checkpoint` writes                                            |
| ------------------------------------------------ | ----------------------------------------------------------------- |
| `meta_data["step"]`                              | `"step": self.step`                                               |
| `meta_data["val_bpb"]`                           | `"val_bpb": self.val_bpb`                                         |
| `meta_data["loop_state"]["min_val_bpb"]`         | `"loop_state": {"min_val_bpb": self.min_val_bpb}`                 |
| `meta_data["loop_state"]["smooth_train_loss"]`   | `"loop_state": {"smooth_train_loss": self.smooth_train_loss}`     |
| `meta_data["loop_state"]["total_training_time"]` | `"loop_state": {"total_training_time": self.total_training_time}` |
| `meta_data["dataloader_state_dict"]`             | `"dataloader_state_dict": self.dataloader_state_dict`             |

### Checkpoint file layout (on disk)

For a base training run at step 1000, rank 0 and rank 1:

```
checkpoints/base/d12/
  model_001000.pt          # model state_dict — rank 0 only
  optim_001000_rank0.pt    # optimizer shard — rank 0
  optim_001000_rank1.pt    # optimizer shard — rank 1
  meta_001000.json         # PretrainingState + model/user/batch config
  config.toml              # full Config snapshot
```

`meta_001000.json` structure:

```json
{
  "step": 1000,
  "val_bpb": 1.234,
  "model_config": { "n_layer": 12, "n_head": 12, ... },
  "user_config": { "training": { "depth": 12, ... } },
  "device_batch_size": 32,
  "max_seq_len": 1024,
  "total_batch_size": 524288,
  "dataloader_state_dict": { "epoch": 2, "pq_idx": 47, "rg_idx": 3 },
  "loop_state": {
    "min_val_bpb": 1.198,
    "smooth_train_loss": 1.241,
    "total_training_time": 3612.4
  }
}
```

### Checkpoint concerns to address

**1. `save_checkpoint` receives a raw dict** — currently `loop.py` constructs the meta dict inline and passes it to `save_checkpoint`. After this refactor, `state.to_checkpoint(...)` owns that construction. `save_checkpoint` signature is unchanged.

**2. `build_model` re-inits weights before loading** — `model.init_weights()` is called before `load_state_dict` to initialize rotary embeddings. This is noted as a TODO in the code. Not blocking for this refactor but worth tracking.

**3. Optimizer sharding** — `optim_<step>_rank<N>.pt` is saved by every rank independently. `load_optimizer_state` loads only the shard for the current rank. `PretrainingState` does not need to know about this — it stays in `checkpoint.py`.

**4. `_patch_missing_config_keys` / `_patch_missing_keys`** — backward compat patches for old checkpoints. These live in `checkpoint.py` and are unaffected by this refactor.

**5. SFT does not checkpoint loop state** — current `train_sft.py` only saves at `last_step` and does not include `loop_state` in meta. `SFTState` will add `to_checkpoint()` for consistency but the SFT checkpoint format gains new keys — not a breaking change since `from_checkpoint` is only called on resume and SFT does not currently support resume.

---

## Phase 2 — training/sft/ package

`SFTState` is passed into `dataloader.py` — eliminates `nonlocal`. Then delete `train_sft.py`.

| Module          | Content                                                                       | Est. lines |
| --------------- | ----------------------------------------------------------------------------- | ---------- |
| `state.py`      | `SFTState`                                                                    | ~30        |
| `schedulers.py` | `sft_lr_scheduler`, `sft_muon_momentum_scheduler`                             | ~20        |
| `dataloader.py` | `sft_data_generator_bos_bestfit(state: SFTState, ...)` — no `nonlocal`        | ~100       |
| `setup.py`      | compute init, model load, hyperparameter inheritance, optimizer, task mixture | ~150       |
| `loop.py`       | training loop — uses `SFTState`                                               | ~150       |
| `__init__.py`   | `train_sft(config)` — orchestrates setup then loop                            | ~30        |

---

## Phase 3 — training/rl/ package

Promote nested functions to module level. Then delete `train_rl.py`.

| Module          | Content                                                    | Est. lines |
| --------------- | ---------------------------------------------------------- | ---------- |
| `state.py`      | `RLState`                                                  | ~10        |
| `schedulers.py` | `rl_lr_scheduler`                                          | ~10        |
| `rollout.py`    | `get_batch(...)` generator — promoted from nested function | ~80        |
| `eval.py`       | `run_gsm8k_eval(...)` — promoted from nested function      | ~50        |
| `loop.py`       | training loop — uses `RLState`                             | ~120       |
| `__init__.py`   | `train_rl(config)` — orchestrates setup then loop          | ~30        |

---

## Phase 4 — evaluation/base/ package

Then delete `base_eval.py`.

| Module        | Content                                               | Est. lines |
| ------------- | ----------------------------------------------------- | ---------- |
| `state.py`    | `BaseEvalResult`                                      | ~15        |
| `loop.py`     | core eval, bpb eval, sampling — uses `BaseEvalResult` | ~150       |
| `__init__.py` | `run_base_eval(config)`                               | ~20        |

---

## Phase 5 — evaluation/chat/ package

Then delete `chat_eval.py`.

| Module        | Content                                                 | Est. lines |
| ------------- | ------------------------------------------------------- | ---------- |
| `state.py`    | `ChatEvalResult`                                        | ~10        |
| `loop.py`     | chat eval loop, ChatCORE metric — uses `ChatEvalResult` | ~200       |
| `__init__.py` | `run_chat_eval(...)`                                    | ~20        |

---

## Phase 6 — Update clients and documentation

### cli.py — import path updates only

`src/nanochat/cli.py` is the only caller of all five entry point functions. Update imports:

```python
# before
from nanochat.evaluation.base_eval import base_eval
from nanochat.evaluation.chat_eval import chat_eval
from nanochat.training.train_base import train_base
from nanochat.training.train_rl import train_rl
from nanochat.training.train_sft import train_sft

# after
from nanochat.evaluation.base import run_base_eval as base_eval
from nanochat.evaluation.chat import run_chat_eval as chat_eval
from nanochat.training.base import train_base
from nanochat.training.rl import train_rl
from nanochat.training.sft import train_sft
```

No other changes to `cli.py` — function signatures are unchanged.

### docs/code-structure.md

- Update `training/` package map: replace flat file rows with sub-package rows (`base/`, `sft/`, `rl/`)
- Update `evaluation/` package map: replace `base_eval.py` and `chat_eval.py` rows with `base/` and `chat/` sub-packages
- Update CLI flow diagram: `train_base(config)` call path now goes through `training/base/__init__.py`
- Update dependency rules: `training/base/` imports from `training/` shared modules (`checkpoint.py`, `dataloader.py`, etc.)

### docs/roadmap.md

- Replace the `TrainingState refactor` deferred item with a link to this document
- Update the link from `training-state-refactor.md` to `entry-point-refactor.md`

### CHANGELOG.md

Add entry under `[Unreleased]`:

```markdown
### Changed
- `training/train_base.py`, `training/train_sft.py`, `training/train_rl.py` split into
  `training/base/`, `training/sft/`, `training/rl/` sub-packages with co-located state dataclasses
- `evaluation/base_eval.py`, `evaluation/chat_eval.py` split into
  `evaluation/base/`, `evaluation/chat/` sub-packages
- Mutable loop state extracted into `PretrainingState`, `SFTState`, `RLState`, `BaseEvalResult`, `ChatEvalResult`
- `train_base` closure eliminated — loop promoted to module-level function
- `train_sft` `nonlocal` hack eliminated — `SFTState` passed explicitly into dataloader
```

## Files touched

| File                      | Change                                     |
| ------------------------- | ------------------------------------------ |
| `training/base/`          | New package — replaces `train_base.py`     |
| `training/sft/`           | New package — replaces `train_sft.py`      |
| `training/rl/`            | New package — replaces `train_rl.py`       |
| `evaluation/base/`        | New package — replaces `base_eval.py`      |
| `evaluation/chat/`        | New package — replaces `chat_eval.py`      |
| `training/train_base.py`  | Deleted (end of phase 1)                   |
| `training/train_sft.py`   | Deleted (end of phase 2)                   |
| `training/train_rl.py`    | Deleted (end of phase 3)                   |
| `evaluation/base_eval.py` | Deleted (end of phase 4)                   |
| `evaluation/chat_eval.py` | Deleted (end of phase 5)                   |
| `cli.py`                  | Import paths updated (phase 6)             |
| `docs/code-structure.md`  | Package map and CLI flow updated (phase 6) |
| `docs/roadmap.md`         | Deferred item link updated (phase 6)       |
| `CHANGELOG.md`            | Unreleased entry added (phase 6)           |
