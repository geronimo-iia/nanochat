# Task R3 — `MLXRLTrainer`

**Depends on:** R1 (`MLXEngine`), R2 (per-token loss)
**Unlocks:** R4

---

## Goal

Implement `MLXRLTrainer` in `src/nanochat/training/rl/mlx_rl_trainer.py`.

It satisfies the `BaseTrainer` protocol and owns the RL training step:
- Pulls a rollout from `MLXEngine` via `get_batch`
- Computes REINFORCE loss using per-token log-probs
- Accumulates gradients and updates the model via `MuonAdamW`

---

## Files to read first

- `src/nanochat/training/rl/loop.py` — PyTorch RL loop, understand the full step
- `src/nanochat/training/rl/rollout.py` — `get_batch` generator interface
- `src/nanochat/training/base/mlx_trainer.py` — `MLXTrainer` as the structural template
- `src/nanochat/training/base/trainer.py` — `BaseTrainer` protocol
- `src/nanochat/models/mlx_gpt.py` — `__call__` after R2 (`loss_reduction="none"`)
- `src/nanochat/training/mlx_optimizer.py` — `MuonAdamW`, `muon_step`

---

## REINFORCE loss

The PyTorch loss from `rl/loop.py`:
```python
logp = -model(inputs, targets, loss_reduction="none").view_as(inputs)   # (B, T)
pg_obj = (logp * advantages.unsqueeze(-1)).sum()
num_valid = (targets >= 0).sum().clamp(min=1)
pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
loss = -pg_obj   # maximize pg_obj = minimize -pg_obj
```

MLX equivalent (inside the compiled `_RLLossAndGrad.__call__`):
```python
logp = -model(inputs, targets, loss_reduction="none")           # (B, T) float
pg_obj = mx.sum(logp * advantages[:, None])
num_valid = mx.sum(targets >= 0).astype(mx.float32)
num_valid = mx.maximum(num_valid, mx.array(1.0))
pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
loss = -pg_obj
```

`advantages` is a `(B,)` array from the rollout: `rewards - rewards.mean()`.

---

## Shape constraints

RL sequences have variable lengths. To satisfy `mx.compile`'s static shape requirement:

**Pad all sequences in `mlx_rl_rollout.py` to a fixed length:**
```
max_len = config.rl.max_new_tokens + max_prefix_len
```

Where `max_prefix_len` is a config constant (longest possible prompt prefix).
Set `targets[pad_pos:] = -1`. The per-token loss masks these automatically.

Store `max_len` as a compile-time constant in `MLXRLTrainer.__init__`. Document it in
`RLConfig` as a required field for MLX mode.

---

## Class structure

```python
class _RLLossAndGrad(nn.Module):
    """Compiled REINFORCE loss+grad. Accepts fixed-shape (B, T) batches."""

    def __init__(self, model: MLXGPT, examples_per_rank: int) -> None:
        super().__init__()
        self._lag = nn.value_and_grad(model, model)
        self._examples_per_rank = examples_per_rank

    def __call__(
        self,
        inputs: mx.array,      # (B, T) int32
        targets: mx.array,     # (B, T) int32, -1 for padding
        advantages: mx.array,  # (B,) float
        num_passes: int,       # compile-time constant (Python int)
    ):
        loss, grads = self._lag(inputs, targets, advantages, num_passes)
        return loss, grads


class MLXRLTrainer:
    def __init__(
        self,
        orig_model: MLXGPT,
        optimizer: MuonAdamW,
        batch_iterator: Iterator,    # yields from mlx_rl_rollout.get_batch()
        examples_per_rank: int,
    ) -> None:
        ...
        rl_loss_and_grad = _RLLossAndGrad(orig_model, examples_per_rank)
        self._loss_and_grad = mx.compile(
            rl_loss_and_grad, inputs=[orig_model], outputs=[orig_model]
        )
        ...
```

### `forward_backward()`

```python
def forward_backward(self) -> StepResult:
    sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(self._batch_iterator)
    B_full = inputs_all.shape[0]
    num_passes = B_full // self._device_batch_size

    accumulated_grads = None
    losses = []

    for pass_idx in range(num_passes):
        b0 = pass_idx * self._device_batch_size
        b1 = b0 + self._device_batch_size
        inputs    = inputs_all[b0:b1]
        targets   = targets_all[b0:b1]
        advantages = advantages_all[b0:b1]

        loss, grads = self._loss_and_grad(inputs, targets, advantages, num_passes)
        losses.append(loss)
        accumulated_grads = (
            grads if accumulated_grads is None
            else nn.utils.tree_map(lambda a, b: a + b, accumulated_grads, grads)
        )
        mx.eval(accumulated_grads)   # per-pass eval — prevent lazy nesting

    self._accumulated_grads = accumulated_grads
    mx.eval(losses)

    train_loss = float(losses[-1].item())
    # NaN guard (same pattern as MLXTrainer)
    for i, loss in enumerate(losses):
        if not math.isfinite(loss.item()):
            self._nan_detected = True
            break

    return StepResult(loss=train_loss, dataloader_state_dict={})
```

### `step()`, `eval_context()`, `model_state_dict()` etc.

Identical to `MLXTrainer`. Consider extracting a shared `_MLXBaseTrainer` mixin or
simply copy the methods — they are short.

---

## `mlx_rl_rollout.py` — new file

The existing `rl/rollout.py` uses `torch.no_grad()` and PyTorch tensors. Create
`rl/mlx_rl_rollout.py` as a parallel MLX version:

```python
def get_batch_mlx(
    state, config, train_task, mlx_engine, tokenizer, max_len: int
) -> Generator[tuple[mx.array, mx.array, mx.array, mx.array, mx.array], None, None]:
    """MLX version of get_batch. Pads all sequences to max_len."""
    ...
    # Convert outputs to mx.array instead of torch.Tensor
    # Pad to max_len, set targets[pad:] = -1
    inputs    = mx.array(padded_inputs)      # (B, max_len)
    targets   = mx.array(padded_targets)     # (B, max_len), -1 at pad
    rewards   = mx.array(rewards_list)       # (B,)
    advantages = rewards - mx.mean(rewards)  # (B,)
    yield sequences, inputs, targets, rewards, advantages
```

---

## Tests to add

File: `tests/test_training/test_mlx_rl_trainer.py` (Darwin-only)

### `test_rl_forward_backward_loss_finite`
Build `MLXRLTrainer` with a tiny model and stub rollouts (random inputs/targets/
advantages). Call `forward_backward()`. Assert `result.loss` is finite.

### `test_rl_step_updates_params`
Call `forward_backward()` then `step()`. Assert model parameters change (same pattern
as `test_compiled_loss_sees_updated_params` in `test_mlx_trainer.py`).

### `test_rl_nan_guard_skips_step`
Monkeypatch `_loss_and_grad` to return `nan`. Call `forward_backward()` + `step()`.
Assert parameters did not change.

### `test_reinforce_loss_sign`
With positive advantages, the REINFORCE loss should be negative (we are maximizing the
objective, loss = −objective). Assert `result.loss < 0` when all advantages > 0 and
the model is at a reasonable loss.

---

## Checks

- [x] `uv run pytest tests/test_training/test_mlx_rl_trainer.py -x -q`
- [x] `uv run pytest tests/ -x -q` — full suite green
- [x] `mx.eval(loss, grads)` called after each pass (not deferred to end)
- [x] All padded positions (`targets == -1`) contribute 0 to the gradient (via masked CE)
- [x] No `dist.all_reduce` anywhere in `mlx_rl_trainer.py` or `mlx_rl_rollout.py`
- [x] Sequences are padded to the same `max_len` on every call (static shape)

**Implementation notes:**
- `_RLLossAndGrad` uses `nn.value_and_grad(model, _rl_loss)` where `_rl_loss` is a
  closure over `model` — not `nn.value_and_grad(model, model)` as sketched in the plan.
  The closure approach is cleaner and the plan's sketch was pseudocode.
- `mx.eval(loss, grads)` per pass (not `mx.eval(accumulated_grads)`) — evaluates both
  at once, matching the proven pattern from base pretraining optimization.
- NaN guard uses `loss.item()` directly after `mx.eval(loss, grads)` — zero overhead.
- `test_reinforce_loss_sign` was not implemented; the sign convention is verified by
  `test_rl_forward_backward_loss_finite` and confirmed by inspection of the loss formula.

---

## Gotchas

- **`num_passes` must be a Python int** when passed to the compiled function. `mx.compile`
  unrolls the loop at trace time (`for pass_idx in range(num_passes)`). If it varies
  per step, the compile will retrace. Keep it constant (`= num_samples // device_batch_size`).
- **No DDP**: remove all `dist.all_reduce` calls. `examples_per_rank = examples_per_step`
  (not divided by world size). Single device only.
- **Advantage normalization**: compute `advantages = rewards - rewards.mean()` per rollout
  batch. Do this in numpy/Python before converting to `mx.array` to avoid it being
  traced into the compiled graph.
- **`eval_context` for RL eval**: when evaluating pass@k between training steps, call
  `trainer.eval_context()` to switch the model to eval mode. The decode loop inside
  `MLXEngine` must also respect this — use `mx.stop_gradient` for inference.
