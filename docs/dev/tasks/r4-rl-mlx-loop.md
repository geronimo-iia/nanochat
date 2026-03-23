# Task R4 — RL MLX loop and setup

**Depends on:** R3 (`MLXRLTrainer`)
**Unlocks:** end-to-end MLX RL training

---

## Goal

Wire `MLXRLTrainer` into a complete RL training loop. Either refactor `rl/loop.py` to
accept `BaseTrainer` (like the S3 approach for SFT), or write a dedicated
`rl/mlx_rl_loop.py`. Also add `mlx_rl_setup()` to `rl/setup.py`.

---

## Files to read first

- `src/nanochat/training/rl/loop.py` — full file
- `src/nanochat/training/rl/setup.py` — `RLTrainingSetup`
- `src/nanochat/training/rl/eval.py` — `run_gsm8k_eval`
- `src/nanochat/training/base/mlx_trainer.py` — for `MLXTrainer` as a pattern
- `src/nanochat/training/rl/mlx_rl_trainer.py` — from R3

---

## Loop strategy

**Recommendation: new `rl/mlx_rl_loop.py`**, not a refactor of `rl/loop.py`.

Rationale: `rl/loop.py` has deep DDP logic (`dist.all_reduce` for rewards, pass@k
aggregation, rank-sharded rollout iteration). Untangling this safely without breaking
the PyTorch path is risky. The MLX loop is structurally simpler (single device, no DDP)
and a clean separate file is safer.

---

## `mlx_rl_setup()`

Add to `src/nanochat/training/rl/setup.py`:

```python
def mlx_rl_setup(config: Config) -> RLTrainingSetup:
    """Build RL training state for the MLX backend."""
    from nanochat.training.rl.mlx_rl_trainer import MLXRLTrainer
    from nanochat.training.rl.mlx_engine import MLXEngine
    from nanochat.models.mlx_gpt import GPT as MLXGPT
    from nanochat.training.mlx_optimizer import MuonAdamW, build_param_groups
    ...
```

Key differences from PyTorch `setup()`:
1. Load MLX checkpoint (from `phase="sft"`)
2. Build `MLXEngine` instead of `Engine`
3. `batch_iterator = mlx_get_batch(...)` — the MLX rollout generator
4. `examples_per_rank = config.rl.examples_per_step` (no DDP rank division)
5. No `torch.compile`, no scaler
6. Return `RLTrainingSetup` with `trainer=MLXRLTrainer(...)` in a new `trainer` slot

Add `trainer` slot to `RLTrainingSetup.__slots__` (same pattern as S2 for SFT).

---

## `mlx_rl_loop.py` structure

```python
def mlx_rl_train_loop(s: RLTrainingSetup) -> None:
    checkpoint_manager = make_checkpoint_manager(s.ckpt_dir, s.config.checkpoint)

    for step in range(s.num_steps):
        s.state.step = step

        # Evaluate (pass@k)
        if step % s.config.rl.eval_every == 0:
            with s.trainer.eval_context() as model:
                records = list(run_gsm8k_eval_mlx(
                    task=s.val_task,
                    tokenizer=s.tokenizer,
                    engine=s.mlx_engine,       # MLXEngine
                    config=s.config,
                    max_examples=s.config.rl.eval_examples,
                    num_samples=s.config.rl.device_batch_size,
                ))
            # compute pass@k locally (no dist.all_reduce)
            ...
            s.wandb_run.log(log_passk, step=step)

        # Forward/backward + step
        result = s.trainer.forward_backward()
        lrm = s.get_lr_multiplier(step)
        s.trainer.step(lrm, momentum=0.95, weight_decay=s.config.rl.weight_decay)

        # Logging
        s.wandb_run.log({"reward": ..., "lrm": lrm}, step=step)

        # Checkpoint
        if step % s.config.checkpoint.save_every == 0 or step == s.num_steps - 1:
            checkpoint_manager.save(s.state, s.trainer.model_state_dict(), None)
```

### `run_gsm8k_eval_mlx`

Add to `rl/eval.py` alongside the existing `run_gsm8k_eval`:

```python
def run_gsm8k_eval_mlx(
    task, tokenizer, engine: MLXEngine, config, max_examples, num_samples, temperature=0.0
) -> Generator[dict, None, None]:
    """Single-device GSM8K eval using MLXEngine."""
    max_examples = min(max_examples, len(task))
    for idx in range(max_examples):
        # identical structure to run_gsm8k_eval but with MLXEngine
        ...
```

---

## Dispatch in `rl/setup.py`

```python
def setup(config: Config) -> RLTrainingSetup:
    device_type = autodetect_device_type() if ... else ...
    if device_type == "mlx":
        return mlx_rl_setup(config)
    return _torch_rl_setup(config)   # rename existing setup() body
```

---

## Tests to add

File: `tests/test_training/test_rl_mlx_loop.py` (Darwin-only)

### `test_mlx_rl_loop_one_step`
Build a minimal `RLTrainingSetup` with stubbed `trainer` and `engine`. Run one iteration
of `mlx_rl_train_loop`. Assert `state.step == 1`, `trainer.forward_backward` called
once, `trainer.step` called once.

### `test_mlx_rl_checkpoint_saved`
Run loop to `num_steps` using a temp directory. Assert checkpoint file exists.

### `test_rl_mlx_setup_no_ddp`
Assert that `mlx_rl_setup` does not set `ddp=True` and `ddp_world_size == 1`.

---

## Checks

- [x] `uv run pytest tests/ -x -q` — full suite green (339 passed)
- [x] PyTorch RL loop path untouched: existing `rl/loop.py` tests pass
- [x] No `dist.all_reduce` or `torch.distributed` imports in any `mlx_rl_*.py` file
- [ ] `test_rl_mlx_loop.py` unit tests (loop-level stubs) — not yet written
- [ ] Smoke run: `bash runs/rl-mlx-smoke.sh` — 3 steps, no crash, reward logged

**Implementation notes:**
- Separate `mlx_rl_loop.py` used (as recommended). `rl/loop.py` is untouched.
- The `examples_per_rank` inner loop in `mlx_rl_train_loop` calls `forward_backward()`
  once per example rather than once per full batch — each call internally handles the
  micro-pass loop over `device_batch_size` slices.
- Advantage normalisation is done inside `mlx_rl_rollout.get_batch_mlx()` in numpy
  before `mx.array` conversion — not inside the compiled graph.
- `run_gsm8k_eval_mlx` added to `rl/eval.py` alongside existing `run_gsm8k_eval`.
- `mlx_rl_setup()` added to `rl/setup.py`; `setup()` dispatches on `device_type=="mlx"`.
- `trainer` slot added to both `SFTTrainingSetup` and `RLTrainingSetup`.

---

## Gotchas

- `run_gsm8k_eval_mlx` must call `mx.eval` after each generation batch. The `MLXEngine`
  handles this internally, but verify the calling code doesn't accumulate unreduced arrays.
- The `mlx_rl_loop.py` eval block computes pass@k from a local list of records (no
  all-reduce). The denominator is `len(records)` (local), not a globally-reduced count.
- `checkpoint_manager.save` expects a model state dict as a flat `dict[str, np.ndarray]`.
  `MLXRLTrainer.model_state_dict()` returns `{k: mx.array}`. Add numpy conversion in
  `model_state_dict()` (same pattern as `MLXTrainer`).
- Muon momentum for RL: `rl/loop.py` uses a fixed momentum (no `get_muon_momentum`
  scheduler). Pass a constant to `step()`: `s.trainer.step(lrm, momentum=0.95, weight_decay=...)`.
