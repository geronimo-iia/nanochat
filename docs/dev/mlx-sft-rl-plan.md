# MLX SFT and RL — Design Plan

## Context

Base pretraining on MLX is stable and fast. This document plans how to extend the MLX
backend to SFT and RL training, grounding every decision in what we learned during base
training. Both SFT and RL share the same MLX compute stack (same model, same optimizer,
same `mx.compile` pattern) but each phase introduces its own blockers.

---

## When to use MLX vs cloud

The M3 Max GPU has ~14 TFLOPS BF16. A d24 forward+backward pass costs ~9.8B FLOPs/token.
At ~33% MFU that yields ~300 tok/sec — fine for experiments, inadequate for pretraining.

**Production target is d24** (validated in `dev/LOG.md` 2026-03-04: ClimbMix brought the
GPT-2 capability model from d26 to d24). Key numbers derived from the LOG scaling table
and FP8 training results (d26 on 8×H100 at 630K tok/sec):

| Phase | d24 tokens | M3 Max time | Single H100 | Cost |
|---|---|---|---|---|
| Pretrain | ~8B | ~300 days | ~26h | ~$70 |
| SFT | ~160M (2%) | ~6 days | ~2h | ~$8/run |
| RL | ~80M (1%) | ~3 days | ~1h | ~$4/run |
| d6 experiment | ~500M | ~3–6h | ~10min | ~$0.50/run |

**Rule of thumb:**
- **Pretrain at d24**: always use cloud. One-shot run, ~$70 on a single H100.
- **SFT and RL**: use M3 Max. Each phase requires many iterations (data mixture tuning,
  LR schedules, RL algorithm experiments). 10 SFT runs = $80 on cloud vs free locally.
- **Architecture experiments**: always use M3 Max at d6. The entire `dev/LOG.md` research
  loop (value embeddings, sliding window, Muon variants, dataset comparisons) was driven
  by fast d6 iteration. Hours per run, no queue, no cost.

**The right workflow:**
```
Pretrain d24 on H100 (~26h, ~$70)
       ↓
Iterate SFT on M3 Max via MLX (~6 days/run, free)
       ↓
Iterate RL on M3 Max via MLX (~3 days/run, free)
```

The MLX SFT and RL work in this document is what makes the second and third steps
practical without an ongoing cloud bill.

---

## Architecture already in the codebase

All of the following are already implemented and validated — no architectural debt:

| Feature | Where | LOG entry |
|---|---|---|
| RoPE (rotary embeddings) | `mlx_gpt.py:32`, `_precompute_rotary` | Baseline |
| GQA (grouped-query attention) | `n_kv_head` separate from `n_head` | 2026-01-17 |
| RMS norm (no learnable weight) | `mx.fast.rms_norm` | Explicit over LayerNorm |
| Sliding window (`SSSL` pattern) | `causal_window_mask`, `_masks` | 2026-01-11 |
| Value embeddings (alternating layers) | `value_embeds`, `has_ve()` | 2026-01-17 |
| Per-layer residual scalars | `resid_lambdas`, `x0_lambdas` | 2026-01-11 |
| Logit softcap | `SOFTCAP * tanh(x / SOFTCAP)` | 2026-03-02 |
| ReLU² MLP | `relu(x) ** 2` | 2026-02-05 (SwiGLU negative) |
| Muon + NorMuon + Polar Express | `mlx_optimizer.py` | 2026-01-10 |

Tried and **rejected** (not in code, do not re-add without new evidence):
MoE (wall-clock negative at d18), SwiGLU (worse than ReLU²), MTP (memory overhead, worse),
bigram embeddings (reverted at d25+), grad clipping (always inactive, 2% overhead).

---

## Lessons carried forward from base MLX training

These constraints apply to every new MLX trainer:

| Lesson | Rule |
|---|---|
| `mx.compile` requires `inputs=[model], outputs=[model]` | Both args mandatory — `inputs` re-reads params each call; `outputs` allows gradient flow |
| Deep lazy nesting kills throughput | Call `mx.eval(accumulated_grads)` after each accumulation step. At N=8 steps without eval, 7×n\_params extra kernel launches. |
| Metal argument buffer limit | Fused K-step compile crashes at K=8 for d6 (~74 param tensors). Keep K≤4. |
| Polar Express must run in float32 | bfloat16 (7 mantissa bits) causes catastrophic cancellation at step ~25 with coefficients a≈8, b≈−22, c≈15. Already fixed in `mlx_optimizer.py`. |
| NaN guard | Check `loss.item()` after eval; skip optimizer if non-finite. Already in `MLXTrainer`. |
| `nn.value_and_grad(model, model)` | Gradient is w.r.t. `model.trainable_parameters()`, returned as a model-shaped nested dict. |
| Static shapes required | `mx.compile` traces a static Metal program at first call. Variable batch shapes re-trigger recompilation. Pad to a fixed maximum. |

---

## Part 1 — SFT MLX

### What changes versus base training

SFT is the closest phase to base pretraining. The training loop structure, optimizer,
and data loading are all identical at the Python level. Three things differ:

#### 1. Masked targets (`ignore_index = -1`)

The SFT dataloader (`sft/dataloader.py`) sets `targets[mask == 0] = -1` for padding
positions and non-assistant tokens. The current `mlx_gpt.py` forward pass is:

```python
loss = mx.mean(nn.losses.cross_entropy(logits.reshape(-1, V), targets.reshape(-1)))
```

`mx.mean` averages over **all** positions including masked ones — this is wrong for SFT.
The PyTorch model uses `ignore_index=-1` in `F.cross_entropy`. MLX equivalent:

```python
flat_targets = targets.reshape(-1)
flat_logits  = logits.reshape(-1, self.config.vocab_size)
mask         = flat_targets >= 0                            # (B*T,) bool
ce           = nn.losses.cross_entropy(flat_logits, flat_targets.clip(0))  # clip avoids OOB index
loss         = mx.sum(ce * mask) / mx.maximum(mask.sum(), 1)
```

**This is the only required change to `mlx_gpt.py` for SFT.** It is backward-compatible
with base pretraining (all base targets are ≥ 0, so mask is all-ones and the result is
identical to the current `mx.mean`).

#### 2. ChatCORE evaluation

SFT `loop.py` calls `run_chat_eval()` which drives `Engine` — a PyTorch model with
KV-cache autoregressive generation. There is no `MLXEngine` yet.

**Decision: skip ChatCORE on MLX for now.** Run validation bpb only (already works via
`evaluate_bpb_mlx`). ChatCORE becomes available once `MLXEngine` is implemented (see
Part 2).

#### 3. Optimizer warm-start from pretrained checkpoint

SFT loads Muon momentum buffers from the base checkpoint. `MLXTrainer.load_state_dicts`
already handles this via `MuonAdamW._muon_state`. No change needed.

### New files

```
training/sft/
  mlx_sft_trainer.py    # MLXSFTTrainer — identical to MLXTrainer, delegates to
                        # sft-patched mlx_gpt forward
  setup.py              # add mlx_sft_setup() — mirrors base mlx_setup() but loads
                        # from phase="sft", builds SFT task mixture and dataloader
```

**`MLXSFTTrainer`** can be a thin subclass of `MLXTrainer` or an independent copy. Given
that the only difference is the loss function (handled in `mlx_gpt.py`, not the trainer),
it is likely that no new class is needed at all — the existing `MLXTrainer` + the patched
`mlx_gpt.py` is sufficient. This should be verified before creating a new file.

### Loop strategy

Two options:

**Option A** — Refactor `sft/loop.py` to accept the `BaseTrainer` protocol.
Replace `s.model(x, y) + loss.backward()` with `s.trainer.forward_backward()` and
`s.trainer.step(lrm, momentum, wd)`. The rest of the loop (eval, ChatCORE, checkpoint,
logging) stays unchanged. MLX and PyTorch diverge only at backend selection in `setup()`.

**Option B** — New `sft/mlx_sft_loop.py` mirroring `base/loop.py`.
Duplicate the loop for SFT-specific eval/checkpoint logic.

**Recommendation: Option A.** `sft/loop.py` is already very close to `base/loop.py`
structurally. Refactoring it to accept `BaseTrainer` reduces duplication and gives MLX
support for free. The only SFT-specific concern is ChatCORE, which is already
conditionally guarded by `chatcore_every > 0`.

### SFT implementation phases

| Phase | Work | Blocker |
|---|---|---|
| S1 | Patch `mlx_gpt.py` with masked CE | None |
| S2 | Add `mlx_sft_setup()` to `sft/setup.py` | S1 |
| S3 | Refactor `sft/loop.py` to use `BaseTrainer` protocol | S2 |
| S4 | ChatCORE on MLX | Requires `MLXEngine` (Part 2) |

---

## Part 2 — RL MLX

### What changes versus SFT

RL training is structurally different from both base pretraining and SFT:

| Concern | Base/SFT | RL |
|---|---|---|
| Forward pass input | Fixed `(B, T)` | Variable-length rollouts |
| Loss | Mean cross-entropy | REINFORCE: `−logp × advantages` per token |
| Data source | Pre-tokenized dataset | Live rollouts from model generation |
| Generation | Not needed in training | Core of every step (policy rollouts) |
| DDP | Supported | Single-device only in MLX |

### Blocker 1 — `MLXEngine` (generation with KV-cache)

Every RL training step calls `engine.generate_batch()` to produce rollouts. The current
`Engine` is PyTorch-only (KV-cache, top-k sampling, seed-based). Without an MLX
equivalent, we cannot do rollouts on-device.

**`MLXEngine` required functionality:**
- Prefill pass: encode the prompt prefix once
- Decode loop: autoregressive token generation with KV-cache
- `generate_batch(tokens, num_samples, max_tokens, temperature, top_k, seed)` interface
- `mx.eval` after each decode step (MLX lazy eval requires explicit materialization in
  autoregressive loops)

`MLXEngine` is a standalone module. It does not affect the training loop until RL. It
should be implemented and validated independently before any RL trainer work begins.

**KV-cache and `mx.compile`:** The decode step (single-token forward with KV-cache
update) is a good candidate for `mx.compile`. The KV-cache tensors must be passed as
`inputs=` so MLX reads the updated cache on each call. Shape is fixed if we fix
`max_new_tokens` at compile time.

### Blocker 2 — Per-token loss mode in `mlx_gpt.py`

The RL loss is:
```python
logp = -model(inputs, targets, loss_reduction="none")  # (B, T) per-token logprobs
pg_obj = (logp * advantages.unsqueeze(-1)).sum()
```

The current `mlx_gpt.py` only returns a scalar mean loss. A `loss_reduction` parameter
needs to be added:

```python
# In mlx_gpt.py __call__:
if targets is None:
    return logits

ce = nn.losses.cross_entropy(flat_logits, flat_targets.clip(0))  # (B*T,)
ce = ce * (flat_targets >= 0)                                     # mask padding

if loss_reduction == "none":
    return ce.reshape(B, T)   # (B, T)
return ce.sum() / mx.maximum((flat_targets >= 0).sum(), 1)        # scalar
```

### Blocker 3 — Variable sequence lengths and `mx.compile`

Rollout sequences have different lengths per example. `mx.compile` produces a static
Metal program — variable shapes trigger re-tracing, which is expensive.

**Solution: pad all sequences to `max_new_tokens + prefix_length` at rollout time.**

This is already partially true: `rollout.py` pads to `max(len(seq) for seq in batch)`.
For MLX, pad to the compile-time constant `config.rl.max_new_tokens + max_prefix_len`.
Padding positions get `targets = -1` (masked). The loss skips them via the masked CE
above.

This keeps shapes static and lets `mx.compile` produce a reusable kernel.

### Blocker 4 — No DDP in MLX

`rl/loop.py` and `rl/rollout.py` use `dist.all_reduce` to aggregate rewards across ranks.
MLX is single-device. The MLX RL loop runs on one GPU only:
- `examples_per_rank = examples_per_step` (no rank splitting)
- Remove all `dist.all_reduce` calls in the MLX path
- Reward aggregation is just the local mean

This limits RL throughput but is acceptable for single-machine experimentation.

### New files

```
training/rl/
  mlx_engine.py         # MLXEngine — KV-cache generation (Phase R1, prerequisite)
  mlx_rl_trainer.py     # MLXRLTrainer — REINFORCE forward_backward (Phase R3)
  setup.py              # add mlx_rl_setup() (Phase R3)
```

### RL implementation phases

| Phase | Work | Blocker |
|---|---|---|
| R1 | `MLXEngine` — KV-cache prefill + decode | None (independent) |
| R2 | `mlx_gpt.py` per-token loss (`loss_reduction="none"`) + static padding | None |
| R3 | `MLXRLTrainer.forward_backward()` — REINFORCE loss, `mx.compile` | R1 + R2 |
| R4 | `mlx_rl_setup()`, `mlx_rl_loop()` | R3 |

---

## Shared architecture decisions

### Keep the same `BaseTrainer` protocol

`MLXSFTTrainer` and `MLXRLTrainer` should both satisfy the `BaseTrainer` protocol from
`training/base/trainer.py`. This lets the existing checkpoint, eval, and logging
infrastructure work unchanged.

The RL trainer differs in that `forward_backward()` takes rollout batches rather than
consuming from an internal loader. Two options:

**Option A** — Extend the protocol with `forward_backward(xs, ys, advantages)` for RL.
This breaks the current signature.

**Option B** — `MLXRLTrainer` owns its `batch_iterator` internally (as `MLXTrainer` owns
its loader), and `forward_backward()` calls `next(self._batch_iterator)` to get the
rollout. The public protocol signature stays unchanged.

**Recommendation: Option B.** The protocol stays clean. The trainer is self-contained.

### Do not create an `MLXSFTTrainer` class unless needed

Since the only SFT-specific change (`ignore_index=-1`) lives in `mlx_gpt.py`, the
existing `MLXTrainer` may work unchanged for SFT. Validate this before adding a new class.

### `evaluate_bpb_mlx` already works for SFT

The base loop already calls `evaluate_bpb_mlx` when `device_type == "mlx"`. The SFT val
loader produces masked targets; `evaluate_bpb_mlx` passes them through `mlx_gpt.py`
forward. After the S1 patch to `mlx_gpt.py`, validation bpb on SFT will be correct.

---

## Summary: what to implement and in what order

```
Priority 1 (unblocked, enables SFT)
  S1  mlx_gpt.py     — masked CE (ignore_index=-1), backward-compatible       ✅
  S2  sft/setup.py   — mlx_sft_setup()                                        ✅
  S3  sft/loop.py    — refactor to use BaseTrainer protocol                   ✅

Priority 2 (enables ChatCORE + RL rollouts)
  R1  evaluation/mlx_engine.py  — MLXEngine (KV-cache prefill + decode)       ✅

Priority 3 (enables full RL on MLX)
  R2  mlx_gpt.py     — per-token loss (loss_reduction="none") + static padding ✅
  R3  rl/mlx_rl_trainer.py  — MLXRLTrainer (REINFORCE)                        ✅
  R4  rl/setup.py    — mlx_rl_setup()                                         ✅
  R4  rl/mlx_rl_loop.py (dedicated file, not a refactor of rl/loop.py)        ✅
```

---

## Implementation status

**All 7 tasks complete.** Branch `feat/mlx-gpt`, 5 commits:

| Commit | Tasks | Files |
|---|---|---|
| `e19b4ac` | S1 + R2 | `models/mlx_gpt.py`, `tests/test_models/test_mlx_gpt.py` |
| `0d0fbfb` | S2 + S3 | `training/sft/setup.py`, `training/sft/loop.py` |
| `9e27bc2` | R1 | `evaluation/mlx_engine.py`, `tests/test_training/test_mlx_engine.py` |
| `e8f7ad4` | R3 | `training/rl/mlx_rl_trainer.py`, `tests/test_training/test_mlx_rl_trainer.py` |
| `9d7e524` | R4 | `training/rl/mlx_rl_rollout.py`, `training/rl/mlx_rl_loop.py`, `training/rl/setup.py`, `training/rl/eval.py` |

**Test suite:** 339 passed, 10 skipped. **Pyright:** 22 errors (below pre-existing 25).

### Decisions that differed from the plan

- **`MLXEngine` location:** placed in `evaluation/mlx_engine.py` (plan's own preference),
  not `training/rl/mlx_engine.py`.
- **No `MLXSFTTrainer` class:** the existing `MLXTrainer` in `training/base/mlx_trainer.py`
  works unchanged for SFT — the only SFT-specific change is in `mlx_gpt.py` (S1). Plan
  anticipated this and recommended verifying first.
- **`_RLLossAndGrad` uses a closure:** `nn.value_and_grad(model, _rl_loss)` where
  `_rl_loss` is a closure over `model`. The plan's pseudocode sketch used
  `nn.value_and_grad(model, model)` which is not valid API.
- **`mx.eval(loss, grads)` per pass** (not `mx.eval(accumulated_grads)`): evaluates both
  at once — proven more efficient from base pretraining benchmarks.
- **`mx.compile` not applied to decode step:** KV-cache shape grows each step, which
  would retrace on every iteration. The plan's own caution was confirmed correct.
- **`examples_per_rank` loop in `mlx_rl_loop.py`:** each call to `forward_backward()`
  handles its own micro-pass loop; the outer loop iterates over examples.

### Remaining manual checks (not automated)

- Smoke run: `bash runs/sft-smoke.sh` with MLX backend — 5 steps, no crash
- Smoke run: `bash runs/rl-mlx-smoke.sh` — 3 steps, reward logged to wandb
- Greedy decoding end-to-end: `nanochat chat` using MLX model
