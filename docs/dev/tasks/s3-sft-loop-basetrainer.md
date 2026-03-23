# Task S3 — Refactor `sft/loop.py` to use `BaseTrainer` protocol

**Depends on:** S2
**Unlocks:** MLX SFT training end-to-end

---

## Goal

`sft/loop.py` currently calls PyTorch primitives directly (`s.model(x, y)`,
`loss.backward()`, `s.optimizer.step()`). Replacing these with
`s.trainer.forward_backward()` / `s.trainer.step()` makes the loop backend-agnostic.
MLX and PyTorch then diverge only inside the trainer object, not in the loop.

This is equivalent to what `base/loop.py` already does — compare them side by side.

---

## Files to read first

- `src/nanochat/training/sft/loop.py` — the current loop (full file)
- `src/nanochat/training/base/loop.py` — reference for how `BaseTrainer` is used
- `src/nanochat/training/base/trainer.py` — `BaseTrainer` protocol definition
- `src/nanochat/training/base/mlx_trainer.py` — `MLXTrainer` as the concrete MLX impl
- `src/nanochat/training/sft/setup.py` — `SFTTrainingSetup` fields

---

## Exact changes to `sft/loop.py`

### 1. Replace the training step block

Current (lines ~121–153):
```python
s.synchronize()
t0 = time.time()
train_loss = torch.zeros(1, device=s.device)
for _ in range(s.grad_accum_steps):
    loss = s.model(x, y)
    train_loss = loss.detach()
    loss = loss / s.grad_accum_steps
    if s.scaler is not None:
        s.scaler.scale(loss).backward()
    else:
        loss.backward()
    x, y = next(s.train_loader)
    state.progress = max(state.progress, state.approx_progress)

lrm = s.get_lr_multiplier(state.progress)
muon_momentum = s.get_muon_momentum(state.step)
for group in s.optimizer.param_groups:
    group["lr"] = group["initial_lr"] * lrm
    if group["kind"] == "muon":
        group["momentum"] = muon_momentum
if s.scaler is not None:
    ...
    s.scaler.step(s.optimizer)
    s.scaler.update()
else:
    s.optimizer.step()
s.model.zero_grad(set_to_none=True)
```

Replace with:
```python
s.synchronize()
t0 = time.time()

if s.trainer is not None:
    # MLX path — BaseTrainer handles accumulation, optimizer, and data loading
    result = s.trainer.forward_backward()
    train_loss_val = result.loss
    state.progress = max(state.progress, state.approx_progress)
    lrm = s.get_lr_multiplier(state.progress)
    muon_momentum = s.get_muon_momentum(state.step)
    s.trainer.step(lrm, muon_momentum, weight_decay=0.0)
else:
    # PyTorch path — unchanged
    train_loss = torch.zeros(1, device=s.device)
    for _ in range(s.grad_accum_steps):
        loss = s.model(x, y)
        train_loss = loss.detach()
        loss = loss / s.grad_accum_steps
        if s.scaler is not None:
            s.scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y = next(s.train_loader)
        state.progress = max(state.progress, state.approx_progress)
    lrm = s.get_lr_multiplier(state.progress)
    muon_momentum = s.get_muon_momentum(state.step)
    for group in s.optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
    if s.scaler is not None:
        ...
    else:
        s.optimizer.step()
    s.model.zero_grad(set_to_none=True)
    train_loss_val = train_loss.item()
```

Then use `train_loss_val` (float) in the logging block instead of `train_loss.item()`.

### 2. Guard ChatCORE eval for MLX

ChatCORE requires `Engine` (PyTorch KV-cache). Skip it when `s.trainer is not None`:

```python
if s.config.sft.chatcore_every > 0 and s.trainer is None and (...):
    # existing ChatCORE block
```

### 3. Guard `s.device_type == "mps"` cache clear

The `torch.mps.empty_cache()` call near the top of the loop should only run when
not using the MLX backend:

```python
if s.device_type == "mps":
    torch.mps.empty_cache()
```

This is already conditional on `device_type`, so no change needed if `device_type`
is `"mlx"` for the MLX path.

### 4. Val loader for MLX

The val bpb eval at the top of the loop calls `evaluate_bpb` (PyTorch). Replace with
`evaluate_bpb_mlx` when `s.trainer is not None`, using the `eval_context` from the
trainer — same pattern as `base/loop.py` lines 39–43.

---

## Tests to add

File: `tests/test_training/test_sft_loop_mlx.py` (new, Darwin-only)

### `test_sft_loop_mlx_one_step`
Build a minimal `SFTTrainingSetup` with a stub MLX trainer (monkeypatched
`forward_backward` and `step`), run one iteration of the loop body, and assert:
- `forward_backward` was called once
- `step` was called once with finite `lr_multiplier`
- `state.step == 1`

### `test_sft_pytorch_path_unchanged`
Run the PyTorch path (`s.trainer = None`) through one step with a stub torch model,
assert the loop behaves identically to before.

---

## Checks

- [x] `uv run pytest tests/test_training/ -x -q` — all existing SFT and MLX trainer tests pass
- [x] `uv run pytest tests/ -x -q` — full suite green
- [ ] Manual smoke: `bash runs/sft-smoke.sh` with `--backend=mlx` runs 5 steps without crash
- [ ] PyTorch SFT path: `bash runs/sft-smoke.sh` with `--backend=torch` still works

---

## Gotchas

- The loop primes the dataloader with `x, y = next(s.train_loader)` at the top.
  The MLX trainer owns its own loader and primes it in `__init__`. When `s.trainer is
  not None`, skip the top-level `x, y = next(s.train_loader)` prime. Add a guard:
  ```python
  if s.trainer is None:
      x, y = next(s.train_loader)
  ```
- `state.approx_progress` is updated inside `sft_data_generator_bos_bestfit` (it mutates
  `state` directly). With MLX, the trainer owns the loader — `state.approx_progress`
  will still be updated because the same generator is passed in. Verify by printing
  `state.approx_progress` after the first MLX step.
- `state.last_step` is also mutated by the generator. Same concern — it should still work
  because the generator is the same object.
- `weight_decay=0.0` is the SFT default (no weight decay during fine-tuning). Hardcode
  or pass from config.
