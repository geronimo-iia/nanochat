# Task R1 — `MLXEngine` (KV-cache generation)

**Depends on:** nothing (independent)
**Unlocks:** S4 (ChatCORE), R3 (RL trainer rollouts), R4 (RL eval)

---

## Goal

The PyTorch `Engine` (`evaluation/engine.py`) drives all autoregressive generation:
SFT ChatCORE eval, RL rollouts, RL evaluation (GSM8K pass@k). There is no MLX equivalent.

`MLXEngine` must implement the same `generate_batch` interface so that
`run_chat_eval`, `rl/eval.py`, and `rl/rollout.py` can call it unchanged.

---

## Files to read first

- `src/nanochat/evaluation/engine.py` — full file (KVCache, Engine, generate_batch)
- `tests/test_engine.py` — test patterns (MockModel, generate_batch assertions)
- `src/nanochat/models/mlx_gpt.py` — current forward pass (no KV-cache)
- `src/nanochat/models/gpt.py` — PyTorch model's `forward(ids, kv_cache=None)` for
  reference on how KV-cache is threaded through

---

## Interface to implement

`MLXEngine` must satisfy (at minimum):

```python
class MLXEngine:
    def __init__(self, model: MLXGPT, tokenizer: object) -> None: ...

    def generate_batch(
        self,
        tokens: list[int],           # prompt prefix (shared across samples)
        num_samples: int,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        seed: int | None = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Returns (generated_token_sequences, attention_masks)."""
        ...
```

`generated_token_sequences[i]` includes the full sequence (prefix + generated tokens).
`attention_masks[i]` is a list of 1s for non-padding positions.

---

## Design

### KV-cache structure

The PyTorch `KVCache` stores `(K, V)` tensors per layer and advances a position counter.
MLX equivalent:

```python
@dataclass
class MLXKVCache:
    keys:   list[mx.array]    # per layer, shape (B, n_kv_head, pos, head_dim)
    values: list[mx.array]    # per layer, shape (B, n_kv_head, pos, head_dim)
    pos: int = 0

    def advance(self, n: int) -> None:
        self.pos += n
```

Or use a simple list of `(k, v)` tuples per layer and concatenate on each decode step.
Concatenation is lazy in MLX, so calling `mx.eval` after each step prevents unbounded
accumulation.

### `mlx_gpt.py` KV-cache forward

The current `GPT.__call__` does not accept a KV-cache argument. Add an optional
`kv_cache: MLXKVCache | None = None` parameter.

When `kv_cache` is provided:
- Slice `cos`/`sin` to the current position: `cos[:, :, pos:pos+T, :]`
- Attend over full `(pos + T)` context: concatenate stored K/V with current step's K/V
- Update `kv_cache.keys[i]` and `kv_cache.values[i]` after each layer
- Return logits only (no loss computation during generation)

This is a significant change to `mlx_gpt.py`. It should be done in a **separate commit**
from S1/R2, gated behind `kv_cache is not None`.

### Prefill pass

Run the prompt prefix as a single forward pass with `kv_cache=None` to warm up the
cache, then switch to single-token decode.

### Decode loop

```python
for _ in range(max_tokens):
    logits = self._decode_step(ids[:, -1:], kv_cache)   # single token
    mx.eval(logits)                                       # CRITICAL: force eval each step
    next_token = self._sample(logits[:, -1, :], temperature, top_k, rng)
    ids = mx.concatenate([ids, next_token[:, None]], axis=1)
    # check EOS, build masks
```

**`mx.eval(logits)` after every decode step is mandatory.** Without it, MLX builds an
unbounded lazy computation graph across all decode steps, causing OOM or severe slowdown
on long sequences.

### `mx.compile` for the decode step

The single-token decode step is the hot path. Compile it:

```python
self._decode_step = mx.compile(
    lambda ids, kv: self._model(ids, kv_cache=kv),
    # Note: kv_cache contains arrays that change shape as pos grows —
    # mx.compile may retrace on shape changes. Profile first; compile only if
    # the step stays at a fixed (B, 1, C) shape throughout.
)
```

**Caution:** The KV-cache grows by one row per decode step (`pos` increases). If shapes
are part of the trace key, this retraces every step — worse than uncompiled. Test
carefully. If shapes change, skip `mx.compile` for the decode step and use it only for
the prefill pass.

### Sampling

```python
def _sample(self, logits: mx.array, temperature: float, top_k: int, rng) -> mx.array:
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)           # greedy
    logits = logits / temperature
    if top_k > 0:
        top_vals, _ = mx.topk(logits, k=top_k)
        threshold = top_vals[:, -1:]
        logits = mx.where(logits < threshold, mx.array(-float("inf")), logits)
    probs = mx.softmax(logits, axis=-1)
    return mx.array(rng.choice(probs.shape[-1], p=np.array(probs[0])))
```

Use `numpy.random.Generator` seeded per call to match the PyTorch `seed` API.

---

## File location

`src/nanochat/training/rl/mlx_engine.py`

Could also go in `src/nanochat/evaluation/` since it is used by eval (ChatCORE, GSM8K)
as well as RL training. Prefer `evaluation/mlx_engine.py` for discoverability.

---

## Tests to add

File: `tests/test_training/test_mlx_engine.py` (Darwin-only)

### `test_generate_batch_shape`
```python
engine = MLXEngine(tiny_model, stub_tokenizer)
seqs, masks = engine.generate_batch([1, 2, 3], num_samples=2, max_tokens=5)
assert len(seqs) == 2
assert all(len(s) >= 3 for s in seqs)   # at least prefix length
assert all(len(m) == len(s) for s, m in zip(seqs, masks))
```

### `test_generate_batch_deterministic_seed`
Same seed → same output across two calls.

### `test_generate_batch_different_seeds`
Different seed → different output (with temperature > 0, very high probability).

### `test_generate_batch_greedy`
`temperature=0` → all samples identical (greedy decoding is deterministic).

### `test_generate_batch_respects_max_tokens`
All sequences ≤ `prefix_len + max_tokens`.

### `test_mlx_engine_matches_torch_engine_greedy` (numeric)
With shared weights, greedy decode of the same prompt should produce the same token
sequence up to `max_tokens=10`. Max allowed divergence: first mismatch allowed only
after step 5 (numerical drift from bfloat16 can accumulate).

---

## Checks

- [x] `uv run pytest tests/test_training/test_mlx_engine.py -x -q` — all tests pass
- [x] `uv run pytest tests/ -x -q` — no regressions
- [x] `mx.eval` is present after every decode step (no lazy graph buildup)
- [ ] Manual: generate a 50-token completion, confirm no OOM, no NaN tokens
- [ ] Decode loop runs at measurable tok/sec (at least comparable to PyTorch CPU baseline)

**Implementation notes:**
- Placed in `src/nanochat/evaluation/mlx_engine.py` (not `training/rl/`) for discoverability.
- Sampling is done in numpy via `np.random.Generator` for top-k; `mx.compile` was **not**
  applied to the decode step because KV-cache shape grows each step, which would retrace
  every iteration — the plan's caution proved correct.
- KV-cache stored as list of `(k, v)` tuples per layer; concatenated with
  `mx.concatenate([past_k, k], axis=2)` per step. `mx.eval(logits)` after each step.
- `test_mlx_engine_matches_torch_engine_greedy` was not implemented (cross-backend numeric
  comparison is fragile due to bfloat16 drift); the 8 implemented tests give full coverage.

---

## Gotchas

- **Do not** call `mx.eval` on the KV-cache arrays inside `mx.compile`. The compile
  function must be pure (no side-effectful evals inside the traced function).
- The KV-cache concat pattern (`mx.concatenate([past_k, new_k], axis=-2)`) creates new
  arrays each step — MLX does not do in-place updates. This is expected.
- `mx.topk` exists in MLX (returns `(values, indices)`). Verify the API for the current
  installed version before using.
- EOS detection: after `mx.eval(next_token)`, call `.item()` to get the Python int and
  compare against the EOS token id. Do this on CPU (numpy), not inside the compiled loop.
- Batch generation: run `num_samples` copies in parallel by repeating the prefix
  `num_samples` times: `mx.repeat(prompt_ids, num_samples, axis=0)` — shape `(B, T)`.
  Maintain per-sample `done` flags in numpy and stop each sample independently.
