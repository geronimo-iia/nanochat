# Task S1 — Masked cross-entropy in `mlx_gpt.py`

**Depends on:** nothing
**Unlocks:** S2, S3, R2

---

## Goal

The SFT dataloader sets `targets = -1` for padding positions and non-assistant tokens.
The current `mlx_gpt.py` forward pass uses `mx.mean(cross_entropy(...))` which averages
over all positions including masked ones — producing wrong SFT loss and gradients.

Add `ignore_index=-1` semantics to `mlx_gpt.py`. The change must be backward-compatible:
base pretraining targets are always ≥ 0, so the mean must be identical to the current
computation when no mask is present.

---

## Files to read first

- `src/nanochat/models/mlx_gpt.py` — lines 221–264 (`GPT.__call__`)
- `src/nanochat/models/gpt.py` — PyTorch reference, search for `ignore_index`
- `tests/test_models/test_mlx_gpt.py` — existing test patterns

---

## Implementation

**File:** `src/nanochat/models/mlx_gpt.py`, in `GPT.__call__` near the end.

Replace:
```python
loss = mx.mean(nn.losses.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)))
```

With:
```python
flat_targets = targets.reshape(-1)                                           # (B*T,)
flat_logits  = logits.reshape(-1, self.config.vocab_size)                    # (B*T, V)
mask         = (flat_targets >= 0).astype(mx.float32)                        # 1 for valid, 0 for -1
ce           = nn.losses.cross_entropy(flat_logits, mx.maximum(flat_targets, 0))  # clip OOB index
loss         = mx.sum(ce * mask) / mx.maximum(mask.sum(), mx.array(1.0))
```

Key points:
- `mx.maximum(flat_targets, 0)` clips `-1` to `0` before indexing — avoids out-of-bounds
  access in `cross_entropy`. The contribution of those positions is zeroed by `mask`.
- `mx.maximum(mask.sum(), mx.array(1.0))` prevents division by zero if all targets are -1.
- When `flat_targets` is all ≥ 0 (base pretraining), `mask` is all-ones and the result is
  `mean(ce)` — identical to the current implementation.

No change to the function signature. No change to the `targets is None` branch.

---

## Tests to add

File: `tests/test_models/test_mlx_gpt.py`

### `test_masked_loss_ignores_minus_one`
```python
def test_masked_loss_ignores_minus_one():
    """Loss with -1 targets on half the positions equals loss on the valid half only."""
    model = MLXGPT(SMALL_CONFIG)
    rng = np.random.default_rng(42)
    idx     = mx.array(rng.integers(0, 128, (2, 64)))
    targets = mx.array(rng.integers(0, 128, (2, 64)))

    # Mask out the second half of every sequence
    masked_targets = np.array(targets.tolist())
    masked_targets[:, 32:] = -1

    loss_full   = model(idx, targets)
    loss_masked = model(idx, mx.array(masked_targets))
    mx.eval(loss_full, loss_masked)

    # Losses must differ when mask is applied
    assert abs(float(loss_full) - float(loss_masked)) > 1e-4, (
        "Masked loss must differ from full loss"
    )
    assert float(loss_masked) > 0
```

### `test_all_masked_loss_stable`
```python
def test_all_masked_loss_stable():
    """All-masked targets (-1 everywhere) must not produce NaN or inf."""
    model = MLXGPT(SMALL_CONFIG)
    idx     = mx.array(np.zeros((1, 64), dtype=np.int32))
    targets = mx.array(np.full((1, 64), -1, dtype=np.int32))
    loss    = model(idx, targets)
    mx.eval(loss)
    assert math.isfinite(float(loss))
```

### `test_masked_loss_matches_unmasked_on_valid_positions`
```python
def test_masked_loss_matches_unmasked_on_valid_positions():
    """loss(all valid) == loss(same positions, rest masked) when model is shared."""
    model = MLXGPT(SMALL_CONFIG)
    rng = np.random.default_rng(7)
    idx = mx.array(rng.integers(0, 128, (2, 32)))
    t1  = rng.integers(0, 128, (2, 32))

    # mask second row entirely
    t2 = t1.copy()
    t2[1, :] = -1

    loss_row0_only  = model(idx[:1], mx.array(t1[:1]))       # only row 0
    loss_row0_masked = model(idx, mx.array(t2))               # row 1 zeroed

    mx.eval(loss_row0_only, loss_row0_masked)
    diff = abs(float(loss_row0_only) - float(loss_row0_masked))
    assert diff < 1e-4, f"Expected same loss for single-row, got diff={diff}"
```

---

## Checks

- [x] `uv run pytest tests/test_models/test_mlx_gpt.py -x -q` — all existing tests pass
- [x] Three new tests pass
- [x] `uv run pytest tests/ -x -q` — full suite green
- [x] Manually verify: `float(loss_full) != float(loss_masked)` in the first new test

---

## Gotchas

- `nn.losses.cross_entropy` in MLX does **not** have an `ignore_index` argument — the
  masking must be done manually as above.
- Do not use `mx.where(mask, ce, 0)` — prefer `ce * mask` to avoid a conditional branch
  in the Metal graph; multiplication is cheaper.
- The `mx.maximum(flat_targets, 0)` clip is essential — a `-1` index into the vocab
  embedding would access out-of-bounds memory silently or produce garbage gradients.
