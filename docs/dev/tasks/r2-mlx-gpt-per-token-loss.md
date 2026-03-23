# Task R2 — Per-token loss mode in `mlx_gpt.py`

**Depends on:** S1 (masked CE — do S1 first, then extend the same code path)
**Unlocks:** R3 (REINFORCE loss in RL trainer)

---

## Goal

The RL training loss is REINFORCE-style: `−logp(token) × advantage` summed over valid
positions. This requires per-token log-probabilities, not a scalar mean.

Add a `loss_reduction` parameter to `GPT.__call__` in `mlx_gpt.py`:
- `loss_reduction="mean"` (default) — scalar, same as current behavior after S1
- `loss_reduction="none"` — returns `(B, T)` array of per-token losses (CE values)

---

## Files to read first

- `src/nanochat/models/mlx_gpt.py` — full `__call__` after S1 patch
- `src/nanochat/models/gpt.py` — PyTorch reference, `forward(..., loss_reduction="none")`
- `src/nanochat/training/rl/loop.py` — how `loss_reduction="none"` is consumed

---

## Implementation

**File:** `src/nanochat/models/mlx_gpt.py`, `GPT.__call__`.

Change signature:
```python
def __call__(
    self,
    idx: mx.array,
    targets: Optional[mx.array] = None,
    loss_reduction: str = "mean",          # new parameter
) -> mx.array:
```

After S1, the forward pass already computes `ce` and `mask` as intermediate values.
Extend to respect `loss_reduction`:

```python
flat_targets = targets.reshape(-1)
flat_logits  = logits.reshape(-1, self.config.vocab_size)
mask         = (flat_targets >= 0).astype(mx.float32)
ce           = nn.losses.cross_entropy(flat_logits, mx.maximum(flat_targets, 0))

if loss_reduction == "none":
    # Return per-token CE, zeroed at masked positions, shape (B, T)
    return (ce * mask).reshape(idx.shape)

# Default: masked mean
return mx.sum(ce * mask) / mx.maximum(mask.sum(), mx.array(1.0))
```

The `"none"` path returns raw cross-entropy values (positive floats), not log-probs.
In the RL loop: `logp = -model(inputs, targets, loss_reduction="none")` — the negation
gives log-probabilities.

---

## Static shape guarantee for RL

RL sequences have variable lengths per rollout. `mx.compile` requires static shapes.
The convention for MLX RL: **pad all sequences to `max_new_tokens + max_prefix_len`
at rollout time**, with `targets = -1` for padding positions.

The `loss_reduction="none"` path already zeros masked positions via `ce * mask`, so
padded positions contribute zero to the REINFORCE objective. No additional changes needed.

Document this constraint in `rl/rollout.py` comments when R3 is implemented.

---

## Tests to add

File: `tests/test_models/test_mlx_gpt.py` (add to existing file)

### `test_per_token_loss_shape`
```python
def test_per_token_loss_shape():
    model = MLXGPT(SMALL_CONFIG)
    B, T = 2, 64
    idx     = mx.array(np.random.randint(0, 128, (B, T)))
    targets = mx.array(np.random.randint(0, 128, (B, T)))
    loss = model(idx, targets, loss_reduction="none")
    mx.eval(loss)
    assert loss.shape == (B, T)
```

### `test_per_token_loss_positive`
```python
def test_per_token_loss_positive():
    """All unmasked per-token CE values must be positive."""
    model = MLXGPT(SMALL_CONFIG)
    idx     = mx.array(np.random.randint(0, 128, (1, 32)))
    targets = mx.array(np.random.randint(0, 128, (1, 32)))
    loss = model(idx, targets, loss_reduction="none")
    mx.eval(loss)
    assert float(mx.min(loss).item()) >= 0.0
```

### `test_per_token_loss_masked_positions_zero`
```python
def test_per_token_loss_masked_positions_zero():
    """Positions with target == -1 must produce zero CE."""
    model = MLXGPT(SMALL_CONFIG)
    idx     = mx.array(np.zeros((1, 32), dtype=np.int32))
    targets = np.random.randint(0, 128, (1, 32))
    targets[:, 16:] = -1    # mask second half
    loss = model(idx, mx.array(targets), loss_reduction="none")
    mx.eval(loss)
    assert float(mx.max(mx.abs(loss[:, 16:])).item()) == 0.0, "masked positions must be zero"
    assert float(mx.max(loss[:, :16]).item()) > 0.0, "unmasked positions must be positive"
```

### `test_per_token_loss_sum_matches_mean`
```python
def test_per_token_loss_sum_matches_mean():
    """sum(per_token) / n_valid == scalar mean loss."""
    model = MLXGPT(SMALL_CONFIG)
    rng = np.random.default_rng(0)
    idx     = mx.array(rng.integers(0, 128, (1, 32)))
    targets = mx.array(rng.integers(0, 128, (1, 32)))

    scalar = model(idx, targets, loss_reduction="mean")
    tokens = model(idx, targets, loss_reduction="none")
    mx.eval(scalar, tokens)

    n_valid = 32   # all valid
    reconstructed = float(mx.sum(tokens).item()) / n_valid
    assert abs(float(scalar) - reconstructed) < 1e-5
```

---

## Checks

- [x] `uv run pytest tests/test_models/test_mlx_gpt.py -x -q` — all tests pass (including S1 tests)
- [x] `uv run pytest tests/ -x -q` — full suite green
- [x] `loss_reduction="mean"` (default) behavior is identical to post-S1 behavior
- [x] `loss_reduction="none"` with all-valid targets: sum / T == scalar mean
- [x] `loss_reduction="none"` with half-masked: masked positions are exactly 0.0

---

## Gotchas

- Keep `loss_reduction` as a plain Python `str`, not an `mx.array`. It controls Python
  branching inside `__call__` — it is never traced by `mx.compile`. This is fine.
- The `"none"` shape `(B, T)` is the same as `idx.shape`. Do not use `targets.shape`
  (targets can be `None` for inference path). Use `idx.shape`.
- The REINFORCE gradient flows through `logp = -model(...)`. The negation `−ce` is the
  log-probability under the current policy. Confirm sign convention matches
  `rl/loop.py` PyTorch usage before wiring up in R3.
