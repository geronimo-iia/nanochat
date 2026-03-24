promt:

could you look at the root cause about ### val/bpb increasing (3.21 → 5.54 → 5.56)
of forge/nanochat/docs/dev/experiments/exp2-compression-validation.md
I had keeped some note about a previous analysis here forge/nanochat/docs/dev/experiments/note-val-bnp-increasing.md


notes:


In evaluate_bpb_mlx, the model is called as:


logits = model(x_mx)  # no targets — returns logits only

Then cross-entropy is computed manually. But in _LossAndGrad / training, the model is called as model(x, y) which returns a scalar loss. The eval path calls model(x_mx) with no targets — this returns logits, which is correct.

at build_val_loader:

build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, config.training.device_batch_size, config.training.max_seq_len, split="val", device=device
)


For MLX, device=device where device = torch.device("cpu"). That's fine. But notice: build_val_loader is called fresh every time — a new loader is constructed at each eval. This means the val loader always starts from the beginning of the val split. So all three evals (step 0, 250, 464) are on the same 80 val steps from the same data . That rules out data distribution shift between evals.

Now the key question: why does a fresh random model score 3.21 bpb when the theoretical maximum entropy for a uniform distribution over 32768 tokens is log2(32768) = 15 bpb?

The answer is in evaluate_bpb_mlx — it computes bits per byte , not bits per token. The denominator is total_bytes — the sum of byte lengths of all target tokens. A token that represents 4 bytes contributes 4 to the denominator but only 1 to the loss numerator. So bpb = nats / (ln(2) * bytes). For a random model, nats per token ≈ ln(32768) ≈ 10.4. If the average token is ~3.3 bytes long, then bpb ≈ 10.4 / (0.693 * 3.3) ≈ 4.55. That's still not 3.21.

But wait — the step-0 eval happens before any training step , on a fresh random model. A fresh random model with softcap=15 doesn't produce uniform logits — tanh squashes them, and the initial weights are near-zero, so logits are near-zero, which gives a near-uniform distribution. That should give bpb close to the theoretical maximum, not 3.21.

Unless... the step-0 eval is running on the model after MLXTrainer.__init__ primes the loader and calls _next_batch(), which calls mx.array(x.numpy()) — but that doesn't affect the model weights.

Let me look more carefully at evaluate_bpb_mlx. It calls model(x_mx) directly — but model here is self._orig_model passed through eval_context(). At step 0, this is the freshly initialized model. The bpb of 3.21 is genuinely anomalous for a random model.

One more thing to check — token_bytes. If token_bytes has very large values for most tokens (e.g. multi-byte tokens dominate), the denominator grows large and bpb shrinks. Let me check what get_token_bytes returns for this tokenizer.


Check token byte distribution and compute theoretical bpb for a random model

```shell
cd /Users/geronimo/build/sp_theory/forge/nanochat && uv run python -c "
import torch
import os
tok_dir = '/Users/geronimo/build/sp_theory/experiments/nanochat/tokenizer'
token_bytes = torch.load(os.path.join(tok_dir, 'token_bytes.pt'), weights_only=True)
nonzero = token_bytes[token_bytes > 0].float()
print('vocab_size:', token_bytes.shape[0])
print('num special (0 bytes):', (token_bytes == 0).sum().item())
print('avg bytes per token:', nonzero.mean().item())
print('max bytes:', token_bytes.max().item())
print('distribution:', {i: (token_bytes == i).sum().item() for i in range(0, 9)})

# Compute theoretical bpb for a random model
import math
avg_bytes = nonzero.mean().item()
nats_per_token = math.log(32768)
bpb_random = nats_per_token / (math.log(2) * avg_bytes)
print(f'theoretical bpb (random model): {bpb_random:.4f}')
"

vocab_size: 32768
num special (0 bytes): 9
avg bytes per token: 6.57513952255249
max bytes: 32
distribution: {0: 9, 1: 128, 2: 1315, 3: 2763, 4: 4252, 5: 4748, 6: 4305, 7: 3930, 8: 3381}
theoretical bpb (random model): 2.2813
```

## Conclusions

### Step-0 bpb of 3.21 is not anomalous

The initial estimate of "expected ~15 bpb" was wrong — it compared to `log2(32768) = 15 bits/token`
instead of bits/byte. The correct theoretical floor for a random model is:

```
bpb = ln(32768) / (ln(2) * avg_bytes_per_token) = 10.397 / (0.693 * 6.575) ≈ 2.28
```

A step-0 bpb of 3.21 is slightly above 2.28 — consistent with a near-random model whose
logits are not perfectly uniform (softcap + non-zero init weights). No anomaly.

Data distribution shift is also ruled out: `build_val_loader` constructs a fresh loader
on each eval call, so all three evals run on the same val split from the beginning.

### Val/bpb increase (3.21 → 5.54 → 5.56) — two bugs

The increase is caused by two `mx.compile` bugs acting together:

**Bug 1 — missing `inputs=[orig_model]`** (fixed in commit `a39a7c3`):
The compiled `_loss_and_grad` captured parameter arrays at construction time. After each
`model.update(...)` the compiled function kept evaluating the original random weights,
so gradients were always computed at initialization. The model drifted in a fixed
gradient direction, making val/bpb worse each eval.

**Bug 2 — missing `outputs=[orig_model]`** (found during Experiment 2b smoke test):
`nn.value_and_grad` internally calls `model.update(params)` as a side effect inside the
compiled function. Without `outputs=[orig_model]`, MLX doesn't track this write. After
`eval_context()` runs `mx.eval()` on model outputs (making arrays concrete), the next
compiled call produces grad arrays disconnected from the computation graph:

```
RuntimeError: [eval] Attempting to eval an array without a primitive.
```

**Full fix**:

```python
self._loss_and_grad = mx.compile(loss_and_grad, inputs=[orig_model], outputs=[orig_model])
```

- `inputs=`: re-reads current params at the start of each compiled call
- `outputs=`: tells MLX that `orig_model`'s parameters are written inside the function
  (by `nn.value_and_grad`'s internal `model.update(params)`), preventing disconnected arrays

**Regression tests** in `tests/test_training/test_mlx_trainer.py`:
- `test_compiled_loss_sees_updated_params` — catches Bug 1
- `test_eval_context_restores_train_mode` — catches Bug 2 (runs `mx.eval` inside eval_context
  then calls `forward_backward()`)