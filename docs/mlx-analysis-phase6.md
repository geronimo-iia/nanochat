---
title: MLX Backend Analysis — Phase 6
summary: Fresh eyes review of the MLX backend after Phase 5 implementation. Three findings.
status: draft
last_updated: 2025-07-10
---

# MLX Backend Analysis — Phase 6

Fresh analysis of all MLX-related files after Phase 5 implementation.

Files reviewed:
- `src/nanochat/training/mlx_trainer.py`
- `src/nanochat/training/base/setup.py`
- `src/nanochat/training/base/loop.py`
- `src/nanochat/training/base/__init__.py`
- `src/nanochat/training/base/trainer.py`
- `src/nanochat/training/mlx_optimizer.py`
- `src/nanochat/models/mlx_gpt.py`
- `src/nanochat/common/mlx.py`
- `src/nanochat/common/hardware.py`
- `src/nanochat/common/distributed.py`
- `src/nanochat/training/sft/loop.py`
- `tests/test_training/test_mlx_trainer.py`

---

## Finding 1 — `_next_batch` calls `.numpy()` on MPS tensors, which is unsupported

**File**: `setup.py` + `mlx_trainer.py`
**Severity**: Runtime crash on Apple Silicon

```python
def _next_batch(self):
    x, y, state = next(self._torch_loader)
    return mx.array(x.numpy()), mx.array(y.numpy()), state
```

In `_setup_mlx`, the torch loader is created with:

```python
torch_device_type = autodetect_device_type()  # "mps" on Apple Silicon
torch_device = torch.device(torch_device_type)
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    ..., device=torch_device, ...
)
```

On Apple Silicon `autodetect_device_type()` returns `"mps"`, so tensors land
on the MPS device. Calling `.numpy()` on an MPS tensor raises:

```
RuntimeError: numpy conversion is not supported for MPS tensors
```

The MPS device was added to the dataloader for the *torch training path* where
MPS is the compute device. On the MLX path, torch is only doing data prep —
MLX owns the GPU. Keeping tensors on MPS gives zero benefit: the dataloader
does not use `pin_memory` or `non_blocking` for MPS (`use_cuda=False`), and
`_next_batch` must call `.numpy()` anyway which requires a MPS→CPU copy first.
CPU is strictly better: no extra device hop.

The fix belongs in `_setup_mlx` — hardcode `torch.device("cpu")` for the
loader. `_next_batch` itself is correct as-is.

**Decision**: Always use `torch.device("cpu")` for the MLX path loader.
Remove the `autodetect_device_type()` call and `torch_device` variable from
`_setup_mlx` — they are torch-path concerns.

---

## Finding 2 — `model` parameter in `MLXTrainer.__init__` is permanently dead

**File**: `mlx_trainer.py` + `setup.py`
**Severity**: Low — dead code, not a bug

```python
trainer: BaseTrainer = MLXTrainer(model, model, optimizer, grad_accum_steps, train_loader)
```

**History**: The `orig_model`/`model` split was introduced in Phase 3
(see `docs/archive/mlx-analysis.md`, Finding 3) as future-proofing, mirroring
the torch path where `torch.compile(model)` produces a genuinely distinct
compiled wrapper. The intent was:

- `orig_model` — unwrapped source of truth for state_dict, eval, grad computation
- `model` — reserved for a future `mx.compile(model)` wrapper used in the forward pass

In Phase 5 (see `docs/archive/mlx-analysis-phase5.md`, Finding 1), `mx.compile`
was applied — but via the `_LossAndGrad(nn.Module)` wrapper *inside*
`MLXTrainer.__init__`, not as an external compiled model passed in. This made
the `model` parameter permanently dead: `self._model` is assigned once and
never read again (confirmed by grep).

The original intent was valid. Phase 5's execution made it obsolete by
internalising the compilation. The parameter should now be removed.

**Decision**: Remove the `model` parameter from `MLXTrainer.__init__`. Keep
only `orig_model`. Update `_setup_mlx` call site and tests accordingly.

---

## Finding 3 — `sft/loop.py` logs `train/mfu` unconditionally, inconsistent with base loop

**File**: `sft/loop.py` line ~175
**Severity**: Low — will become a live bug when SFT is ported to MLX

```python
s.wandb_run.log({
    ...
    "train/mfu": mfu,
    ...
}, step=state.step)
```

`base/loop.py` guards `train/mfu` with `if s.gpu_peak_flops != float("inf")`.
`sft/loop.py` logs it unconditionally. On the MLX path `gpu_peak_flops` is
`float("inf")`, so `mfu` will be `0.0` (division by inf), producing a
meaningless metric in wandb for every SFT step.

SFT does not currently support the MLX backend, so this is not a live bug
today. However SFT will be ported to MLX, and this fix is independent of that
port — it is a one-liner that makes the loop MLX-ready without touching any
MLX-specific code. Finding 1 (loader device) is the foundational fix needed
for the SFT MLX port; this finding is orthogonal and can land now.

**Decision**: Guard `train/mfu` in `sft/loop.py` with
`if s.gpu_peak_flops != float("inf")` for consistency with the base loop.
Also guard the console `mfu:` print in the same way.

---

## Summary

| # | File | Issue | Severity | Action |
|---|------|-------|----------|--------|
| 1 | `setup.py` + `mlx_trainer.py` | `.numpy()` on MPS tensor crashes at runtime | Runtime crash | Use `torch.device("cpu")` for MLX loader |
| 2 | `mlx_trainer.py` + `setup.py` | `model` param permanently dead after Phase 5 internalised `mx.compile` | Low | Remove `model` param, keep only `orig_model` |
| 3 | `sft/loop.py` | `train/mfu` logged unconditionally — will be `0.0` on MLX path | Low | Guard with `gpu_peak_flops != inf` |

---

## Implementation Plan

### Task 1 — Fix MPS tensor crash (Finding 1)

In `setup.py` `_setup_mlx`, remove `autodetect_device_type()` call and
hardcode `torch.device("cpu")` for the loader:

```python
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    config.training.device_batch_size,
    config.training.max_seq_len,
    split="train",
    device=torch.device("cpu"),
    resume_state_dict=dataloader_resume_state_dict,
)
```

Remove the now-unused `torch_device_type` and `torch_device` variables.
Update `_BackendSetup.device` to use `torch.device("cpu")` directly.

### Task 2 — Remove dead `model` param (Finding 2)

In `mlx_trainer.py`:
- Remove `model: GPT` parameter from `__init__`
- Remove `self._model = model` assignment

In `setup.py`:
- Change `MLXTrainer(model, model, ...)` → `MLXTrainer(model, ...)`

In `tests/test_mlx_trainer.py`:
- Change `MLXTrainer(model, model, ...)` → `MLXTrainer(model, ...)`
- Replace `trainer._model` references with `trainer._orig_model`
  (currently only in `test_step_changes_params`)

### Task 3 — Guard `train/mfu` in `sft/loop.py` (Finding 3)

Guard both the console print and the wandb log:

```python
mfu_str = f" | mfu: {mfu:.2f}" if s.gpu_peak_flops != float("inf") else ""
```

```python
if s.gpu_peak_flops != float("inf"):
    log_payload["train/mfu"] = mfu
```

---

## After implementation

Once all three tasks are complete, run the first MLX vs MPS performance comparison
for base training. Methodology and result tables: [mlx-performance.md](mlx-performance.md).
