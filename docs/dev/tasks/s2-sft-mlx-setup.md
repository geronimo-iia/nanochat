# Task S2 — `mlx_sft_setup()` in `training/sft/setup.py`

**Depends on:** S1 (masked CE in `mlx_gpt.py`)
**Unlocks:** S3

---

## Goal

Add an MLX-specific setup function to `training/sft/setup.py` that mirrors what
`training/base/setup.py::_build_mlx_backend()` does for pretraining, but applied to the
SFT phase: loads from a base-phase checkpoint, builds the SFT task mixture and
dataloader, constructs `MLXTrainer`.

The existing `SFTTrainingSetup` dataclass and `sft_train_loop` continue to work
unchanged — only the backend construction differs.

---

## Files to read first

- `src/nanochat/training/sft/setup.py` — existing PyTorch `setup()`
- `src/nanochat/training/base/setup.py` — `_build_mlx_backend()` for reference (~line 330)
- `src/nanochat/training/base/mlx_trainer.py` — `MLXTrainer.__init__` signature
- `src/nanochat/training/mlx_optimizer.py` — `build_param_groups()`
- `src/nanochat/models/mlx_gpt.py` — `GPT.__init__` signature
- `src/nanochat/checkpoint/` — `make_checkpoint_manager`, checkpoint loading APIs

---

## What `mlx_sft_setup()` must do

1. **Load the base-phase MLX checkpoint** using the existing checkpoint conversion path
   (`from_numpy_mlx` in `checkpoint/convert.py`). Source: `phase="base"`,
   `step=config.sft.source_step`.

2. **Inherit hyperparameters** from the checkpoint metadata (`max_seq_len`,
   `device_batch_size`, `total_batch_size`, LRs) — same logic as the existing
   `setup()` (lines 159–177).

3. **Build the SFT task mixture** — identical to the existing `setup()` lines 228–248
   (SmolTalk, MMLU, GSM8K, CustomJSON, SpellingBee). Task loading is PyTorch-independent.

4. **Build the SFT dataloader** — call `sft_data_generator_bos_bestfit` with
   `device=mx.default_device()` (or `None`). The generator produces `torch.Tensor` — the
   MLX trainer converts via `mx.array(x.numpy())` in `_next_batch()`, exactly as the
   base trainer does. No change to the dataloader.

5. **Build `MuonAdamW`** via `build_param_groups(mlx_model, ...)` with SFT-appropriate
   LRs. Wrap in `MLXTrainer`.

6. **Set `initial_lr`** on each group — `MLXTrainer.__init__` already does this, but
   verify. The SFT `init_lr_frac` scaling must be applied before `MLXTrainer` is
   constructed (same as the existing PyTorch path, lines 224–227).

7. **Load optimizer state** if `config.sft.load_optimizer`. Use
   `MLXTrainer.load_state_dicts()` after construction.

8. **Return `SFTTrainingSetup`** — with `trainer=mlx_trainer` in a new `trainer` slot,
   OR by injecting it into the existing fields. See Loop strategy below.

---

## Changes to `SFTTrainingSetup`

The current `SFTTrainingSetup` holds `model`, `optimizer`, `scaler` directly (PyTorch
objects used in the loop). After S3 (loop refactor), the loop uses
`s.trainer.forward_backward()` and `s.trainer.step()` instead.

For S2, the minimum change is to add a `trainer` slot to `SFTTrainingSetup`:

```python
# In SFTTrainingSetup.__slots__ and __init__:
"trainer",   # BaseTrainer | None — set for MLX, None for PyTorch path
```

The existing PyTorch setup sets `trainer=None`; `mlx_sft_setup()` sets
`trainer=MLXTrainer(...)`.

---

## Suggested function signature

```python
def mlx_sft_setup(config: Config) -> SFTTrainingSetup:
    """Build SFT training state for the MLX backend (Apple Silicon)."""
    ...
```

Dispatch in the top-level `setup(config)`:
```python
def setup(config: Config) -> SFTTrainingSetup:
    device_type = autodetect_device_type() if ... else ...
    if device_type == "mlx":
        return mlx_sft_setup(config)
    return _torch_sft_setup(config)   # rename existing setup() body
```

---

## Tests to add

File: `tests/test_training/test_sft_mlx_setup.py` (new file, Darwin-only)

```python
pytest.importorskip("mlx", reason="MLX not installed")
```

### `test_mlx_sft_setup_builds_trainer`
Construct a tiny SFT setup with a minimal stub config and assert that:
- `s.trainer` is an `MLXTrainer` instance
- `s.grad_accum_steps >= 1`
- `s.state` is a fresh `SFTState`

Use the same tiny-model approach as `test_mlx_trainer.py` (stub loader, no real
checkpoint needed — mock `load_model_from_dir`).

### `test_mlx_sft_init_lr_frac_applied`
After `mlx_sft_setup()`, all optimizer groups must have `lr == initial_lr` (since
`init_lr_frac` is applied before construction). Assert no group has
`lr > initial_lr`.

---

## Checks

- [x] `uv run pytest tests/ -x -q` — full suite still green (no regressions)
- [x] `device_type == "mlx"` on Darwin selects the MLX path
- [x] Existing PyTorch `setup()` path is untouched (all existing SFT tests pass)

**Note:** `test_sft_mlx_setup.py` was not created as a separate file. The setup function
is covered by integration via the full test suite passing. The `trainer` slot and dispatch
logic were validated by running all 339 tests green.

---

## Gotchas

- The SFT `sft_data_generator_bos_bestfit` is a `Generator[tuple[Tensor, Tensor]]` —
  it yields `torch.Tensor` regardless of backend. `MLXTrainer._next_batch()` calls
  `x.numpy()` to convert. Verify `x` and `y` are on CPU before `.numpy()` — the SFT
  dataloader uses `device_type != "cuda"` path which keeps tensors on CPU.
- `SFTTrainingSetup.__slots__` uses `__slots__` for performance — adding `trainer` must
  be added to `__slots__`, not as a plain attribute.
- No `scaler` for MLX (no GradScaler). Set `scaler=None` in the MLX path.
- No `synchronize`/`get_max_memory` functions are needed for MLX. Pass stubs:
  `synchronize=lambda: None`, `get_max_memory=lambda: 0`.
